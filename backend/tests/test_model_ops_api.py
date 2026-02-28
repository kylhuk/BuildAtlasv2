from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import app, get_artifact_base_path


def _prepare_client(tmp_path: Path) -> TestClient:
    app.dependency_overrides.clear()
    app.dependency_overrides[get_artifact_base_path] = lambda: tmp_path
    client = TestClient(app)
    return client


def _close_client(client: TestClient) -> None:
    client.close()
    app.dependency_overrides.clear()


def test_model_ops_status_empty(tmp_path: Path) -> None:
    client = _prepare_client(tmp_path)
    try:
        response = client.get("/ops/model-status")
        assert response.status_code == 200
        payload = response.json()
        assert payload["active_model"] is None
        assert payload["last_training"] is None
        assert payload["checkpoint"]["state"] == "missing"
        assert payload["rollback"]["state"] == "missing"
        assert payload["warnings"] == []
        assert payload["generated_at"]
    finally:
        _close_client(client)


def test_model_ops_status_populated(tmp_path: Path) -> None:
    client = _prepare_client(tmp_path)
    try:
        model_meta = {
            "model_id": "ops-model",
            "trained_at_utc": "2025-04-01T12:00:00Z",
            "record": {"version": "v1"},
            "secret": "should-not-expose",
        }
        model_meta_path = tmp_path / "models" / "ops" / "model_meta.json"
        model_meta_path.parent.mkdir(parents=True, exist_ok=True)
        model_meta_path.write_text(json.dumps(model_meta), encoding="utf-8")

        checkpoint_payload = {
            "checkpoint_id": "ckpt-ops",
            "timestamp": "2025-05-01T00:00:00Z",
            "extra": "drop-me",
        }
        checkpoint_path = tmp_path / "checkpoints" / "run" / "checkpoint-latest.json"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps(checkpoint_payload), encoding="utf-8")

        rollback_payload = {
            "rollback_id": "rb-ops",
            "created_at": "2025-05-02T06:00:00Z",
            "extra": "drop-me",
        }
        rollback_path = tmp_path / "rollback" / "rollback-latest.json"
        rollback_path.parent.mkdir(parents=True, exist_ok=True)
        rollback_path.write_text(json.dumps(rollback_payload), encoding="utf-8")

        response = client.get("/ops/model-status")
        assert response.status_code == 200
        payload = response.json()
        active_model = payload["active_model"]
        assert active_model["model_id"] == "ops-model"
        assert active_model["trained_at_utc"] == "2025-04-01T12:00:00Z"
        assert active_model["metadata"]["model_id"] == "ops-model"
        assert active_model["metadata"]["trained_at_utc"] == "2025-04-01T12:00:00Z"
        assert active_model["metadata"]["record"]["version"] == "v1"
        assert active_model["metadata"]["secret"] == "[REDACTED]"
        assert payload["last_training"]["model_id"] == "ops-model"

        checkpoint = payload["checkpoint"]
        assert checkpoint["state"] == "available"
        assert checkpoint["timestamp"] == checkpoint_payload["timestamp"]
        assert checkpoint["metadata"]["checkpoint_id"] == checkpoint_payload["checkpoint_id"]
        assert checkpoint["metadata"]["timestamp"] == checkpoint_payload["timestamp"]
        assert checkpoint["metadata"]["extra"] == "[REDACTED]"
        assert checkpoint["path"].endswith("checkpoint-latest.json")

        rollback = payload["rollback"]
        assert rollback["state"] == "available"
        assert rollback["timestamp"] == rollback_payload["created_at"]
        assert rollback["metadata"]["rollback_id"] == rollback_payload["rollback_id"]
        assert rollback["metadata"]["created_at"] == rollback_payload["created_at"]
        assert rollback["metadata"]["extra"] == "[REDACTED]"
        assert rollback["path"].endswith("rollback-latest.json")
        assert payload["warnings"] == []
    finally:
        _close_client(client)


def test_model_ops_status_prefers_ml_loop_meta(tmp_path: Path) -> None:
    client = _prepare_client(tmp_path)
    try:
        root_meta = {
            "model_id": "root-model",
            "trained_at_utc": "2025-05-01T00:00:00Z",
            "record": {"version": "root"},
        }
        root_meta_path = tmp_path / "models" / "ops" / "model_meta.json"
        root_meta_path.parent.mkdir(parents=True, exist_ok=True)
        root_meta_path.write_text(json.dumps(root_meta), encoding="utf-8")

        loop_meta = {
            "model_id": "loop-model",
            "trained_at_utc": "2025-06-01T00:00:00Z",
            "record": {"version": "loop"},
        }
        loop_meta_path = (
            tmp_path / "ml_loops" / "loop-alpha" / "models" / "ops" / "model_meta.json"
        )
        loop_meta_path.parent.mkdir(parents=True, exist_ok=True)
        loop_meta_path.write_text(json.dumps(loop_meta), encoding="utf-8")

        response = client.get("/ops/model-status")
        assert response.status_code == 200
        payload = response.json()
        active_model = payload["active_model"]
        assert active_model["model_id"] == "loop-model"
        assert payload["last_training"]["model_id"] == "loop-model"
        assert active_model["metadata"]["model_id"] == "loop-model"
        assert active_model["metadata"]["record"]["version"] == "loop"
        assert payload["warnings"] == []
    finally:
        _close_client(client)


def test_model_ops_status_metadata_empty_after_sanitization(tmp_path: Path) -> None:
    client = _prepare_client(tmp_path)
    try:
        checkpoint_payload = {"custom": "value"}
        checkpoint_path = tmp_path / "checkpoints" / "run" / "checkpoint-empty.json"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps(checkpoint_payload), encoding="utf-8")

        response = client.get("/ops/model-status")
        assert response.status_code == 200
        payload = response.json()
        checkpoint = payload["checkpoint"]
        assert checkpoint["state"] == "available"
        assert checkpoint["metadata"]["custom"] == "[REDACTED]"
        assert payload["warnings"] == []
    finally:
        _close_client(client)



def test_model_ops_status_malformed_json(tmp_path: Path) -> None:
    client = _prepare_client(tmp_path)
    try:
        malformed_path = tmp_path / "checkpoints" / "malformed.json"
        malformed_path.parent.mkdir(parents=True, exist_ok=True)
        malformed_path.write_text("{missing", encoding="utf-8")
        response = client.get("/ops/model-status")
        assert response.status_code == 200
        payload = response.json()
        checkpoint = payload["checkpoint"]
        assert checkpoint["state"] == "available"
        assert checkpoint["metadata"] is None
        assert any("malformed JSON" in warning for warning in payload["warnings"])
    finally:
        _close_client(client)
