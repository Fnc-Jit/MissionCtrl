# OpenEnv Project Playbook (General Template)

This document is a reusable guide for building a strong OpenEnv project from scratch.
It is intentionally general so you can use it for your final submission and future environments.

## 1. What A Good OpenEnv Project Must Do

1. Expose a stable API contract.
2. Define multiple tasks with clear difficulty progression.
3. Provide deterministic, reproducible grading.
4. Keep final task scores strictly in the required range.
5. Ship with a baseline evaluator and test suite.
6. Build and run cleanly in Docker.

## 2. Recommended Task Ladder

Use 4 task tiers to show breadth and increasing complexity.

1. Easy
- Single objective.
- Low risk state transitions.
- Clear reward mapping.
- Used to verify that the agent can follow basic mechanics.

2. Medium
- Multi-step optimization objective.
- Tradeoff between short-term and long-term reward.
- Adds one operational constraint (budget, latency, safety, etc).

3. Hard
- Incident or dynamic stress scenario.
- Requires anticipatory actions and risk management.
- Includes failure modes that can end the episode.

4. Special task
- Distinct objective category that shows originality.
- Examples: sustainability, compliance, fairness, reliability under uncertainty, cost vs quality balancing.
- Should use a materially different grader component from the first three tasks.

## 3. API Contract Requirements (Validator-Safe)

Keep these endpoints and response shapes consistent.

1. GET /
- Must return HTTP 200.
- Should include status and available endpoints.

2. POST /reset
- Accepts task id.
- Returns observation and done false.
- Must return HTTP 200 for valid requests.

3. POST /step
- Accepts a valid action payload.
- Returns observation, reward, done, and info.
- Must return HTTP 200 for valid requests.

4. GET /health
- Simple readiness check.

5. Optional but recommended
- GET /schema
- GET /state
- GET /history
- GET /dashboard

## 4. Two Critical Failure Patterns To Avoid

## A) Endpoint contract breakage (the "HTTP 200" class of failures)

Typical symptom:
- Validation fails early because root or API routes are missing, non-deterministic, or not returning expected status.

Common causes:
- Stateless server wiring that loses episode state between reset and step.
- Route mismatch between docs and actual server.
- Invalid response schema for reset or step.

Fix pattern:
1. Use one persistent environment instance per server process (unless multi-session is intentionally implemented).
2. Ensure GET / returns 200 and a simple JSON heartbeat.
3. Ensure POST /reset and POST /step return stable, JSON-serializable output.
4. Add API tests that assert status code and payload fields.

## B) Score range failure (strict open interval)

Typical symptom:
- Phase 2 style failure: score out of range.
- Requirement is strict: score must be greater than 0 and less than 1.

Fix pattern:
1. Clamp all final grader outputs to an open interval using epsilon.
2. Apply the clamp in every grading path, not only one task.
3. Validate in tests across all tasks.
4. Validate in inference output path too, not only engine internals.

Reference clamp logic:
- epsilon = 0.01 is practical to avoid display rounding to boundaries.
- final_score = max(epsilon, min(1.0 - epsilon, raw_score)).

## 5. Grader Design Standard

Design each task grader with explicit weighted components.

Recommended component families:
1. Goal completion component.
2. Safety or SLA component.
3. Cost or efficiency component.
4. Special objective component (for special task).
5. Communication or human-in-loop bonus (optional).

Rules:
1. Keep each component independently inspectable.
2. Include penalties for severe failures.
3. Return explainable breakdown in info at episode end.

Recommended final info fields:
1. grader_score
2. score_breakdown
3. reward_breakdown

## 6. Explainability Standard (Judge-Friendly)

At minimum, include:
1. raw score
2. final clamped score
3. per-component contribution
4. penalties and bonuses
5. task-specific metrics used in grading

This improves trust, debugging speed, and review quality.

## 7. Baseline Evaluator Requirements

Your inference runner should:
1. Use required environment variables only.
2. Emit consistent machine-parseable logs.
3. Handle retries and transient API failures.
4. Always emit an end line even if task crashes.
5. Assert final task scores are in strict open interval.

Recommended log lines:
1. [START]
2. [STEP]
3. [END]

## 8. Testing Checklist (Must-Have)

1. API tests
- Root returns 200.
- Reset returns valid observation.
- Step returns valid transition payload.

2. Engine tests
- Reset state is clean.
- Action mechanics and penalties.
- Episode boundaries.
- Deterministic replay.
- Grader boundaries for every task.

3. Submission-critical tests
- All task scores strictly between 0 and 1.
- Docker build and run smoke test.
- OpenEnv manifest sanity checks.

## 9. Optimization Playbook

Use this sequence for iterative improvement.

1. Stabilize contract first.
- No flaky routes.
- No schema drift.

2. Stabilize score validity second.
- Open interval clamp in one place.
- Tests enforce bounds.

3. Improve policy quality third.
- Better prompt or heuristic policy.
- Better action guardrails.
- Better task-specific strategy.

4. Improve judge readability fourth.
- Explainable score outputs.
- Dashboard and history clarity.
- Clear README with reproducible commands.

## 10. Advanced Improvement Guide

Use this section when your project already passes basic validation and you want a stronger final score and better judge impression.

### A) Better Prompt Strategy

1. Use explicit role + objective + constraints + output schema in the system prompt.
2. Add hard rules for invalid actions and repeated mistakes.
3. Include a short priority decision framework (top-down) for deterministic behavior.
4. Add task-specific micro-strategies so policy changes by task difficulty.
5. Require strict JSON output and define a safe fallback action for parse failures.

Prompt quality checks:
1. Does the prompt explicitly define allowed commands?
2. Does it define forbidden actions and penalties?
3. Does it include state-aware logic (for example, avoid terminated targets)?
4. Does it define decision priority order, not just generic advice?

### B) Memory Improvement (Agent + Environment)

1. Maintain action history for the current episode (step, action, reward).
2. Include compact summaries of previous failures in each next-step prompt.
3. Add explicit lists of invalid or terminated targets in observation summaries.
4. Keep memory bounded and structured (avoid long unfiltered transcripts).
5. Preserve deterministic environment histories for replay and debugging.

Memory quality checks:
1. Agent should not repeat the same invalid action after penalty.
2. Same initial state + same action sequence should replay identically.
3. Context window should stay stable over long episodes.

### C) Decision Policy Improvement

1. Add pre-action safety guards (target validity, status checks, projected risk).
2. Add one-step lookahead for delayed effects (queued scaling, redistribution impact).
3. Encode risk thresholds (SLA danger zones, budget critical zones).
4. Prefer conservative recovery actions under uncertainty.
5. Add task-specific branching logic rather than one global heuristic.

Decision quality checks:
1. Breach prevention logic triggers before hard failures.
2. Cost actions do not silently create SLA failures.
3. High-risk states reduce exploratory actions.

### D) Baseline Improvement

1. Keep one deterministic heuristic baseline as a control.
2. Keep one LLM baseline using the required evaluator protocol.
3. Compare both on all tasks and report per-task gaps.
4. Add retry, timeout, and fallback behavior in inference.
5. Log all decisions in machine-readable format for analysis.

Baseline quality checks:
1. Baseline never violates output schema.
2. Baseline logs always include START, STEP, END events.
3. Scores stay in strict open interval for all tasks.

### E) Edge-Case Testing Expansion

Add targeted tests for:
1. Near-boundary numeric states (0, 1, epsilon, max thresholds).
2. Invalid or missing action fields.
3. Repeat-action penalties and no-op corner cases.
4. Episode-end behavior (step after done, reset consistency).
5. Determinism under identical seeds and trajectories.
6. Multi-factor stress states (high load + low budget + pending actions).

Testing quality checks:
1. Every grader path has boundary tests.
2. Every terminal condition has at least one explicit test.
3. Contract tests assert HTTP status and response shape.

### F) Dashboard Improvement

1. Show episode timeline (actions, rewards, done events).
2. Show score breakdown panels (cost, SLA, special objective, penalties/bonuses).
3. Add trend charts for critical state metrics per step.
4. Surface validator-relevant fields (final score, bounds, pass indicators).
5. Add replay controls for deterministic debugging demonstrations.

Dashboard quality checks:
1. Judges can understand "why score changed" within 10 seconds.
2. Key failure events are visually obvious.
3. Data shown in dashboard matches API payload values exactly.

## 11. How To Build The Dashboard (From Scratch)

Use this section as a reusable build template for any OpenEnv project dashboard.

### A) Dashboard Goals

1. Show current environment state in one glance.
2. Show what action happened and why reward changed.
3. Show progress toward task objective and failure thresholds.
4. Show final explainable score and component breakdown.

### B) Minimal Backend Endpoints

Expose these read-focused endpoints for dashboard rendering:
1. GET /state
- Current observation snapshot.

2. GET /history
- Chronological action and reward timeline.

3. GET /schema (optional)
- Useful for dynamic UI validation and developer tooling.

4. GET /health
- Used for readiness indicator in UI.

Dashboard correctness rule:
1. The dashboard must display only server-sourced values, never locally guessed values.

### C) Core UI Panels

Build these blocks first:
1. Header
- Task id, step number, done status, health status.

2. KPI strip
- Budget remaining, cumulative reward, traffic/load, risk indicator.

3. Server table
- id, type, status, cpu, memory, cost, recent trend.

4. Action timeline
- Step, action, target, reward delta, error/null status.

5. Score panel
- grader score, score breakdown components, penalties and bonuses.

6. Incident panel
- SLA breaches and other hard-failure events.

### D) Polling and State Sync Strategy

1. Poll state and history at a fixed interval (for example 1-2 seconds).
2. Pause polling when episode is done, or switch to slow polling.
3. Keep one source of truth in client state.
4. Use id-based de-duplication for timeline rows.

Sync quality checks:
1. No duplicated steps in timeline.
2. No stale values after reset.
3. Step counter and history length remain consistent.

### E) Visual Design Pattern (Judge-Friendly)

1. Use color semantics consistently.
- Green = good progress, amber = warning, red = critical.

2. Highlight threshold crossings.
- Example: cpu >= risk threshold, budget below danger threshold.

3. Use compact trend signals.
- Small sparklines or delta badges are enough.

4. Keep typography high contrast and readable.

### F) Explainable Scoring Integration

On final step, render:
1. final score and raw score
2. component contributions
3. grading penalties
4. reward bonuses and penalties accumulated during episode

If score breakdown is absent, show a fallback panel with explicit "breakdown unavailable" status.

### G) Reset and Replay UX

1. Add reset controls for each task.
2. Clear local dashboard state on reset.
3. Keep a replay mode that reads saved history and replays step-by-step.
4. Add speed control for replay to support demos.

### H) Dashboard Test Checklist

1. Endpoint failures show graceful error banners.
2. Empty history state renders without crashing.
3. Done state freezes timeline and shows final score panel.
4. Reset fully re-initializes charts, KPIs, and table rows.
5. Rendered values match API payload values exactly.

## 12. Minimal From-Scratch Build Sequence

1. Define environment concept and objective.
2. Create 4 tasks: easy, medium, hard, special.
3. Implement deterministic simulator with reset and step.
4. Implement grading per task with weighted components.
5. Clamp final scores to strict open interval.
6. Build FastAPI service and stable endpoints.
7. Add baseline inference runner with strict log format.
8. Add tests for API, mechanics, determinism, grading bounds.
9. Package Docker image and run local smoke tests.
10. Document usage and evaluation assumptions in README.

## 13. Pre-Submission Gate

Pass all checks before submit:
1. API contract passes local tests.
2. Docker build succeeds.
3. Reset and step work end-to-end.
4. Every task score is strictly between 0 and 1.
5. Baseline logs are correctly formatted.
6. README instructions run exactly as written.

## 14. Reusable Acceptance Criteria

A project is ready when:
1. It is reproducible.
2. It is validator-safe.
3. It is explainable.
4. It is robust under error conditions.
5. It demonstrates meaningful task progression including one special task.
