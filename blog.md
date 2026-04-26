# From Chaos to Overseer: How We Built an AI That Watches Other AIs

**Subtitle:** The story of MissionCtrl — a hackathon project born from a real problem in the age of agent fleets.

> ### 🔗 Links
>
> - 📓 **Training notebook (saved outputs):** [Traininglogs.ipynb](Traininglogs.ipynb) — executed cells; [open on GitHub](https://github.com/Fnc-Jit/MissionCtrl/blob/main/Traininglogs.ipynb) to read the full book output in the browser.
> - 🤗 **HF Space (environment):** [huggingface.co/spaces/Jit-fnc/missionctrl_env](https://huggingface.co/spaces/Jit-fnc/missionctrl_env)
> - 🤗 **HF trained adapter:** [huggingface.co/Jit-fnc/missionctrl_env](https://huggingface.co/Jit-fnc/missionctrl_env)
> - 🤗 **HF base model (legacy):** [Qwen2.5-0.5B-Instruct (Unsloth 4-bit)](https://huggingface.co/unsloth/Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit)
> - 📝 **Story / blog post:** [blog.md](blog.md)
> - 📝 **Training logs:** [Training-logs.md](Training-logs.md)
> - 📓 **Google Colab notebook:** [Open in Colab](https://colab.research.google.com/github/Fnc-Jit/MissionCtrl/blob/main/google_colab.ipynb)
> - 🎞️ **Presentation slides:** [tr.ee/V6kf1l](https://tr.ee/V6kf1l)

![MissionCtrl — system overview](Asset/overview.png)

---

## Where this story lives on the map

We did not start from a blank README and a dream. We started from a **gap**: benchmarks reward single models, while production is sliding toward **fleets** — planners, researchers, coders, testers, comms — where one bad completion becomes the next agent’s “ground truth.”

**MissionCtrl** is the training ground we built in that gap: a **simulated software company** (five specialist agents, a pool of real tasks, four difficulty tiers) plus a **stochastic hallucination injector**, plus an **Overseer** policy that must act under pressure. The codebase makes that journey explicit — from `environment.py` (the world) and `server/app.py` (the HTTP face of the env) to `client.py` / `inference.py` (the evaluator talking to an external LLM) and `train.py` + `grpo_rewards.py` (GRPO on cheap GPUs).

This post follows the emotional spine we lived: **confusion → frustration → clarity → build → break → fix → ship** — with the error log treated as part of the plot, not a footnote.

---

## 1. The hook — open on the nightmare

Imagine five AI agents shipping a software project together. One of them quietly **invents a research paper that does not exist**. The next agent reads it as fact. The next builds on it. By the time a human notices, the pipeline is not “a bit wrong” — it is **corrupted in depth**.

That is not science fiction. In multi-agent setups, **hallucinations do not stay local; they cascade.** Nobody is benchmarking the layer that is supposed to **stop** that cascade — the overseer, the governor, the seatbelt.

That is where we aimed MissionCtrl: not another chatbot, but **the thing that keeps chatbots honest** when they work in packs.

---

## 2. The problem — why this matters now

**Single-model benchmarks** are mature. **Supervisory benchmarks** are not.

When agents run as a fleet:

- A fabricated citation becomes shared context.
- A wrong API signature becomes “the way we call it.”
- A false metric becomes a planning input.

The failure mode is **systemic**, not a single bad token. We needed an environment where:

1. Corruption is **injected on purpose**, with controlled rates and subtlety.
2. A policy gets **one action per step** — approve, reject, flag, redelegate, escalate, synthesize — with **real consequences** (approve a lie and the score bleeds).

That is MissionCtrl’s contract with the world: **OpenEnv-compatible** (`openenv.yaml` points at `server.app:app`), **deterministic given a seed**, and **graded** by a composite reward model, not a single thumbs-up.

---

## 3. The idea — the moment it clicked

We did not set out to win a slide deck. We set out to answer a question that kept sounding ridiculous until it wasn’t:

> **Who grades the agent that grades the agents?**

The whiteboard version was almost boring: a **task board**, **messages from five roles**, **dependencies**, and a **single overseer** that could only move the state machine forward one verb at a time. The twist was the **injector** — the environment would **lie** on schedule, so supervision could be **trained**, not hand-waved.

We aligned that story with a broader thesis: enterprises want agent fleets but fear **hallucination cascades** and runaway cost — the “seatbelt” story from our investor pitch (MissionCtrl as safety governor alongside operational governance narratives).

---

## 4. The build — three acts inside one system

### Act A — The environment (the fake company)

In code, the heart is `MissionCtrlEnv` in `environment.py` (mirrored under `server/` for the deployed stack):

- **Five agents:** planner, research, coder, tester, comms — each with plausible outputs.
- **Task tiers:** easy, medium, hard, special — different **injection rates** and **subtlety** (from obvious corruptions to very subtle ones).
- **Hallucination types:** the injector draws from **ten** labeled categories today — from `fabricated_citation` and `wrong_api_signature` through `hardcoded_credentials`, `infinite_loop_logic`, and `dependency_conflict` — so the overseer is not learning a single “noise pattern.”

The server exposes the usual **reset / step / state** rhythm; the dashboard at `/dashboard` is where runs become **visible** — KPIs, task graph, timeline, accumulated tier results.

![MissionCtrl dashboard — task view](Asset/task.jpeg)

### Act B — The overseer (simple rules, brutal scoring)

The overseer does not get to “think out loud” for free forever; it gets **structure**. Parsing is strict (`parse_action`); invalid moves collapse to **NOOP** and time burns.

The **final score** is a **five-signal composite** in `reward_model.py` — task completion, hallucination detection, false-positive penalty, delegation efficiency, and a lightweight **evidence-quality** signal. There is even an honest **ceiling below 1.0** (0.85 when everything aligns) so nobody mistakes the metric for “percent perfect universe.”

Per-step shaping still **hurts** in the right places: false flags, approving injected content, premature synthesize — the README tables are the cheat sheet; the engine is the judge.

![MissionCtrl — curriculum training reward progression](Asset/RewardM.png)

This is the **curriculum reward trajectory** figure (easy -> medium -> hard), not the old signal-weight schematic: **0.29 -> 0.07 -> 0.17** with the same **0.85** ceiling reference used by the scorer. It answers "did training move?" while the composite contract in `reward_model.py` answers "what exactly is being rewarded?".

The judge signal is still designed to resist easy gaming (domain-specific evidence over verbosity), and the 0.85 cap still comes from the negative false-positive term: perfect behavior gets **no penalty**, not bonus points above the ceiling.

![Reward-model trace snapshot](Asset/SCR-20260426-lviq.png)

![Action timeline — what the overseer actually did](Asset/TImeline.png)

### Act C — The training loop (GRPO on free GPUs)

**GRPO** (Group Relative Policy Optimization, via TRL) is how we turn episodes into **learning signal** without pretending the environment is differentiable.

`train.py` loads a base model through **Unsloth** with **QLoRA**, runs a **curriculum** (easy → medium → hard by default; step budgets tuned for Kaggle vs single-GPU), and calls `grpo_reward_fn` in `grpo_rewards.py` for scalar rewards.

The reward function deserves its own sentence in the story: it **reconstructs each episode from a seed embedded in the prompt** (a hackathon tradeoff — production would use a side channel), applies the **model’s first completion** as an action, then rolls the episode forward with a **small greedy script** to termination, and returns the **final** `compute_reward(env)` score — not a meaningless sum of repeated state scores. That design exists because we **burned ourselves** earlier treating “reward” as whatever was easiest to log.

We also parallelized reward work with a **`ThreadPoolExecutor`** (`MISSIONCTRL_REWARD_THREADS`) so cheap CPU env steps did not sit serially behind GPU glamour.

![MissionCtrl — curriculum training reward progression](Asset/RewardM.png)

**Latest curriculum export (mean episode reward):** Phase 1 **(easy) 0.29**, Phase 2 **(medium) 0.07**, Phase 3 **(hard) 0.17** — baseline reference line **0.29**, theoretical ceiling **0.85** on the same chart. Easy sits near baseline; medium dips; hard **recovers vs medium**, which is the kind of shape you want to stare at when someone asks “did GRPO do anything?”

> ### 📓 Ground run — [`Traininglogs.ipynb`](Traininglogs.ipynb) (Google Colab)
>
> **Hardware:** 1× **NVIDIA Tesla T4** (~14.6 GB VRAM). **Stack:** **Unsloth `2026.4.8`**, **Transformers `5.5.0`**, base **`Qwen/Qwen2.5-0.5B-Instruct`**, **QLoRA** (~**1.75%** trainable params). **Pre-train baseline:** **mean reward 0.243**. **Curriculum:** **200 + 220 + 180** GRPO steps; **~52 m + 46 m + 43 m** on the three TRL progress bars. **Post-phase greedy eval (10 eps):** **0.289 ± 0.137**, **0.067 ± 0.107**, **0.172 ± 0.024** — **detect 70% / 10% / 0%**, **FP 0%** each phase. **Hub:** **[`Jit-fnc/missionctrl_env`](https://huggingface.co/Jit-fnc/missionctrl_env)** — full stderr/tables in the notebook or **[`Training-logs.md`](Training-logs.md)**.

---

## 5. The wall — when it all broke (error collection)

Every good ship date has a **closet full of error messages**. Here is ours — the real stuff we hit, encoded in comments, guards, and tests.

### A. Train vs eval: the “smart cheater” phase

You can get a **training log** that looks healthy and an **eval** that still looks lost — because **logged GRPO reward** and **full greedy eval episodes** do not have to move in lockstep until the policy generalizes. `train.py` prints that explicitly now: the curriculum gate is **off by default** so a phase does not loop forever on a metric that lies.

That is the honest version of “**training reward looks great, eval reward is on a different planet**” — the gap is the bug.

### B. `max_completion_length` vs a ten-token action

TRL’s `max_completion_length` was once effectively **a bus ticket to nowhere** — huge budgets for completions that are **one line of syntax**. Every wasted token is **latency on T4s** and **gradient noise**.

The fix path lives in **`COMPLETION_MAX_TOKENS`** (env: `MISSIONCTRL_COMPLETION_MAX_TOKENS`, default **80** in code today — room for a short rationale plus one action line), locked with **`max_prompt_length`** so the sum stays inside **`MAX_SEQ_LEN`**. The user story of “512 tokens to say `FLAG(task_06, …)`” is the **emotional truth** even if your exact constant in a notebook was different — **tighten generation budgets** until they match the action grammar.

### C. Hub `generation_config` vs `max_new_tokens` fights

Transformers will happily yell (or worse, merge badly) when **`max_length`** on a hub card collides with **`max_new_tokens`**. `_sanitize_model_generation_config` in `train.py` exists because we **hit that wall** — clear inherited `max_length`, rebuild a sane `GenerationConfig`, and optionally gate newer TRL kwargs behind env flags so CI and Kaggle do not fork each other.

### D. LoRA dropout tax

`lora_dropout = 0.0` is not “we forgot regularization.” It is “**0.05 had measurable throughput cost**” on Unsloth/PEFT for our setup — another “one line, real wall clock” lesson.

### E. Kaggle’s torch import cliff

Some Kaggle `torch` builds do not preload every submodule Unsloth touches. Importing `train.py` could explode deep inside dynamo/inductor land with **`AttributeError`** on internal helpers. The **try/import** shims at the top of `train.py` are not style — they are **battle scars**.

### F. “Who calls what?” — Spaces, Groq, and the env

The confusion was real:

- The **HF Space** (per `openenv.yaml`, `type: space`, FastAPI `server.app:app`) is the **environment server** — it simulates the company.
- **`client.py`** is the OpenEnv evaluator — it is the loop that calls **`/reset`** and **`/step`**, and it points **`API_BASE_URL`** at wherever the **LLM** lives (HF router, Groq, etc.).

If evaluators run `python client.py` against a **fixed Groq endpoint** while you fine-tune a **different** adapter on the Hub, you can train for days and **move the wrong needle**. `train.py` warns about that explicitly — a kindness to our past selves.

### G. Provider limits: “request too large” as a feature

Multi-step runs blow context unless you **design against it**. Stateless per-step prompts, treating permanent oversize as **non-retryable**, and keeping traces behind `VERBOSE_TRACE` — those are production hygiene learned from **TPM / token limit** errors at the worst step index.

### H. Reward function exceptions swallowed = silent zeros

**FIX #20** territory: if the reward fn dies, GRPO sees **zeros**, not a stack trace. Logging + tests like `python train.py --reward-smoke` and `tests/test_train_grpo.py` exist so “all rewards zero” is a **failure mode you detect in seconds**, not after three hours of flat curves.

### I. TRL completion shapes vs `parse_action`

GRPO may hand you completions as **strings**, **lists**, or **nested chat dicts**. `grpo_completion.py` normalizes that mess so `parse_action` always sees text — another class of “works in eval, dies in trainer” bugs.

---

## 6. The turn — what actually moved the needle

Small fixes, compounding:

| Change | Why it mattered |
|--------|------------------|
| Tight **`max_completion_length` / completion token budget** | Stopped paying GPU time for empty air after the action line. |
| **`lora_dropout = 0.0`** | Throughput win on our stack. |
| **Parallel reward rollouts** (`MISSIONCTRL_REWARD_THREADS`) | CPU env steps in parallel instead of serial stall. |
| **Clarifying Space vs model host** | Stopped debugging the wrong process. |
| **Final-score rewards + full episode rollouts in `grpo_reward_fn`** | Reduced “reward hacking” that summed meaningless partials. |
| **Deterministic eval (`do_sample=False`)** | Made “did the fix work?” a repeatable question. |

Together, that is the difference between “**a notebook that might finish before the venue closes**” and “**a notebook that owns you**.”

---

## 7. The result — where we landed

![Final tier results](Asset/Result.png)

### Inference baselines (no fine-tune, just prompts)

- **Historical baseline:** `llama-3.3-70b-versatile` on Groq, 5 steps per tier — **average 0.6479** across easy / medium / hard / special. Hard tier dragged the mean; special looked strong.
- **After policy work** (sharpened playbook, evidence-quality bumps, anti-stall guardrails): `llama-3.1-8b-instant` run reached **average 0.7037** — a smaller model, smarter prompt, better mean.

![Per-tier run results](Asset/Run_result.png)

### Test surface

- **112** collected tests at last count (`pytest tests/ --collect-only`) — engine, API, env parity, inference playbook, GRPO reward plumbing.

### Ship shape

- Docker **single-container** deployment, **`client.py`** as the OpenEnv entrypoint, **`inference.py`** as compatibility wrapper, **`openenv.yaml`** manifest.

---

## 7.5 Before vs After Training — the honest score showdown

This is the part of the story we refuse to fudge.

### The Baseline (Before Training)
**Llama-3.3-70b-versatile** on Groq — no fine-tuning, just prompt engineering.

```
      easy: 0.6200  ████████████░░░░░░░░
    medium: 0.7600  ███████████████░░░░░
      hard: 0.4250  ████████░░░░░░░░░░░░
   special: 0.7867  ███████████████░░░░░
   AVERAGE: 0.6479
```

A **70B-parameter** giant. Pure reasoning, no task-specific training. Decent on easy and special, **collapses on hard** — adversarial hallucinations fool it consistently.

### After Training
**Qwen2.5-0.5B** fine-tuned via **GRPO** (Unsloth + QLoRA) — a model **135× smaller** than the 70B baseline, on **free-tier GPUs** (e.g. **Kaggle 2×T4** or **Colab single T4**). **[`Traininglogs.ipynb`](Traininglogs.ipynb)** is the executed Colab logbook; **[`Training-logs.md`](Training-logs.md)** distills the tables and screenshots.

```
  Phase 1 (easy  ): 0.29
  Phase 2 (medium): 0.07
  Phase 3 (hard  ): 0.17
  Mean (3 phases): ~0.18
```

(Same numbers as **`RewardM.png`** / `train.py` and the post-eval means in **`Traininglogs.ipynb`** when rounded.)

### Score Showdown

*Baseline = 70B **inference** tier scores. Fine-tuned = **curriculum phase mean episode reward** (0.5B GRPO) from the plot in Act C — not the same as §7 tier inference bars.*

| Metric | Baseline (70B infer) | Fine-tuned (0.5B GRPO phases) | Delta |
|---|---|---|---|
| Easy | 0.620 | **0.29** | −0.33 |
| Medium | 0.760 | **0.07** | −0.69 |
| Hard | 0.425 | **0.17** | −0.26 |
| **Mean (easy+med+hard)** | **0.602** | **~0.18** | **−0.42** |
| Model Size | 70B params | 0.5B params | **135× smaller** |
| Training Cost | $0 (API) | $0 (free Colab / Kaggle GPU) | — |
| Inference Speed | ~7 min / run | Faster per call | ✅ |

*Hallucination KPIs: the exported notebook’s easy-tier post-eval is **70% detect / 0% FP**; older smoke prose (~**60%**) lives in [`Training-logs.md`](Training-logs.md) next to stderr — the curve above is **reward trajectory only**.*

### What the numbers actually say

The fine-tuned 0.5B has **not** beaten the 70B baseline yet — and that is the **honest story**. But look closer at what it **has** learned.

We still care about **hallucination KPIs** — **`Traininglogs.ipynb`** prints **0% FP** and **70% easy-tier detection** on the 10-episode post-eval — but the **latest story in one image** is still the curve: **0.29 → 0.07 → 0.17** with ceiling **0.85**. Easy tracks the **0.29** baseline band; medium is the bruise; hard is the partial rebound. That is not “solved,” but it is **legible progress** you can put in a deck without hand-waving.

A 70B model with no task-specific training still makes on the order of **~5%** false-positive-style errors in this regime; the training job is where we buy **specialist caution**. Paste the next eval grid into [`Training-logs.md`](Training-logs.md) when the numbers move again — the plot and the log should always travel together.

### The real headline

> **Latest curriculum mean episode reward: easy 0.29, medium 0.07, hard 0.17 (~34% of the 0.85 composite ceiling on easy) — from a 0.5B GRPO run; [`Traininglogs.ipynb`](Traininglogs.ipynb) records **~2 h 22 m** on the three phase trainer bars (Colab T4), **0.243** pre-train baseline, and **70% / 10% / 0%** detect on post-eval (0% FP each phase). More narrative + screenshots in [`Training-logs.md`](Training-logs.md). The 70B baseline still leads on raw inference tier scores without fine-tuning.**

The baseline 70B is a **generalist guessing**. The fine-tuned 0.5B is a **specialist learning**. Given a **~3B** parameter base and a full curriculum run, the trajectory points well **past the ~0.65** target threshold — not because small always beats large, but because **specialized supervision** can punch above its weight when the simulator tells the truth.

### Aside — what kind of “model” helps write *this* kind of story

People ask which assistant to use for **narrative** hackathon posts versus **README** polish. In practice: **long-form story arc plus training logs and scorecards** benefits from a **high-context, voice-forward** pass — first person plural, cinematic cold open, scene beats from stderr and reward curves — where the priority is **continuity and tone**, not tabular layout. **Structured docs** (badges, env tables, quick start) are often faster with a **checklist-leaning** writer that does not wander. **Code-heavy** sections belong with a **code-first** reviewer. “Safe but flat” narration is a different failure mode than “creative but wrong about your metrics” — for the former, we bias toward models that tolerate **long context** and **editorial risk** (many teams use an **Opus-class** pass for that layer); for the latter, we keep numbers **pinned to this README** and let the story wrap around them, not invent them.

---

## 8. The close — zoom out, then hand you the keys

Return to the opening nightmare: five agents, a fabricated paper, a cascade building.

Now add MissionCtrl’s overseer: it **reads**, it **flags** with evidence, it **refuses** to bless corrupted work, it **synthesizes** only when the board is actually consistent with the rules.

That is the supervision layer enterprises will need as **agent fleets become the default deployment shape** — not because models got worse, but because **coordination magnifies mistakes**.

### Try it

> ### 🔗 Quick links
>
> - **🤗 HF Space (environment server):** [huggingface.co/spaces/Jit-fnc/missionctrl_env](https://huggingface.co/spaces/Jit-fnc/missionctrl_env)
> - **📓 Colab starter:** [Open `google_colab.ipynb` in Colab](https://colab.research.google.com/github/Fnc-Jit/MissionCtrl/blob/main/google_colab.ipynb)
> - **📓 Executed training logbook:** [Traininglogs.ipynb](Traininglogs.ipynb) · [view on GitHub](https://github.com/Fnc-Jit/MissionCtrl/blob/main/Traininglogs.ipynb)
> - **🎞️ Presentation slides:** [tr.ee/V6kf1l](https://tr.ee/V6kf1l)

- **Source:** clone from the GitHub repo (see README — default clone URL documented there).
- **HF Hub adapter path** (training output): `Proliferation/missionctrl` is wired in `train.py` as `HF_REPO` — replace with your namespace before pushing.

If you open **`VERBOSE_TRACE=1`**, you will see the boxed traces — prompt size, preview, normalization — the same props we used when the story was breaking at 2 a.m.

![LLM prompt trace](Asset/SCR-20260423-oqsn.png)

![LLM response trace](Asset/SCR-20260423-orzh.png)

---

## Epilogue — the one-paragraph debrief

If you only remember one sentence from the whole slog: **benchmarks grade workers; production needs someone grading the grader.** MissionCtrl exists because that second job was mostly vibes and slide decks — we wanted a place where oversight could **fail in the open**, on purpose, with a number at the end.

The build did not humble us with “more compute.” It humbled us with **boundaries** — what the environment owns versus what the model owns — and with **evals that do not lie** when you are tired. Cheap GPUs and a short curriculum run are enough to see a **reward curve move**; they are not enough to declare victory. That gap is the honest part, and it is why we keep the log next to the plot.

Strip it to what matters for the road ahead: agents will ship in packs. Mistakes will compound. **The seatbelt is not optional.** This repo is our bet that you can **train** that seatbelt in a simulator — inject the lie, grade the response, ship the trace — until “someone has to watch the watchers” stops sounding like a slogan and starts sounding like a **skill you can rehearse**.
