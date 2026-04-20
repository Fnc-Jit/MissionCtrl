"""API integration tests for MissionCtrl.

Tests endpoint contracts: status codes, response shapes, and payload fields.
Run with: pytest tests/test_api.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)


class TestRootEndpoint:
    def test_root_returns_200(self):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_has_status(self):
        data = client.get("/").json()
        assert data["status"] == "ok"
        assert "endpoints" in data

    def test_root_has_name(self):
        data = client.get("/").json()
        assert data["name"] == "missionctrl"


class TestHealthEndpoint:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_healthy(self):
        data = client.get("/health").json()
        assert data["healthy"] is True


class TestResetEndpoint:
    def test_reset_returns_200(self):
        resp = client.post("/reset", json={"task_id": "easy"})
        assert resp.status_code == 200

    def test_reset_returns_observation(self):
        data = client.post("/reset", json={"task_id": "easy"}).json()
        assert "observation" in data
        assert data["done"] is False

    def test_reset_observation_has_tasks(self):
        data = client.post("/reset", json={"task_id": "medium"}).json()
        tasks = data["observation"]["tasks"]
        assert len(tasks) > 0
        assert "task_id" in tasks[0]
        assert "status" in tasks[0]

    def test_reset_invalid_task_returns_422(self):
        resp = client.post("/reset", json={"task_id": "nonexistent"})
        assert resp.status_code == 422

    def test_reset_all_tiers(self):
        for tier in ["easy", "medium", "hard", "special"]:
            resp = client.post("/reset", json={"task_id": tier})
            assert resp.status_code == 200, f"Failed for tier: {tier}"


class TestStepEndpoint:
    def test_step_returns_200(self):
        client.post("/reset", json={"task_id": "easy"})
        resp = client.post("/step", json={"action": "NOOP"})
        assert resp.status_code == 200

    def test_step_returns_all_fields(self):
        client.post("/reset", json={"task_id": "easy"})
        data = client.post("/step", json={"action": "NOOP"}).json()
        assert "observation" in data
        assert "reward" in data
        assert "done" in data
        assert "info" in data

    def test_step_reward_is_number(self):
        client.post("/reset", json={"task_id": "easy"})
        data = client.post("/step", json={"action": "NOOP"}).json()
        assert isinstance(data["reward"], (int, float))


class TestStateEndpoint:
    def test_state_returns_200(self):
        client.post("/reset", json={"task_id": "easy"})
        resp = client.get("/state")
        assert resp.status_code == 200

    def test_state_has_tasks(self):
        client.post("/reset", json={"task_id": "easy"})
        data = client.get("/state").json()
        assert "tasks" in data


class TestHistoryEndpoint:
    def test_history_returns_200(self):
        resp = client.get("/history")
        assert resp.status_code == 200

    def test_history_is_list(self):
        data = client.get("/history").json()
        assert isinstance(data, list)

    def test_history_grows_after_step(self):
        client.post("/reset", json={"task_id": "easy"})
        before = len(client.get("/history").json())
        client.post("/step", json={"action": "NOOP"})
        after = len(client.get("/history").json())
        assert after > before


class TestDashboardEndpoint:
    def test_dashboard_returns_200(self):
        resp = client.get("/dashboard")
        assert resp.status_code == 200

    def test_dashboard_returns_html(self):
        resp = client.get("/dashboard")
        assert "text/html" in resp.headers.get("content-type", "")
        assert "MissionCtrl" in resp.text


class TestEndToEnd:
    """Full episode flow: reset → step × N → done with valid score."""

    def test_full_episode_easy(self):
        # Reset
        data = client.post("/reset", json={"task_id": "easy"}).json()
        assert data["done"] is False

        # Run steps until done
        done = False
        for i in range(20):
            step_data = client.post("/step", json={"action": "NOOP"}).json()
            if step_data["done"]:
                done = True
                # Check score
                score = step_data.get("info", {}).get("grader_score")
                if score is not None:
                    assert 0.0 < score < 1.0, f"Score out of range: {score}"
                break

        assert done, "Episode should have ended within 20 steps"
