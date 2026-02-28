from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path
from typing import Any
from unittest.mock import patch

from backend.engine.generation import runner as generation_runner

from fastapi.testclient import TestClient

from backend.app.api.evaluator import BuildEvaluator
from backend.app.db.ch import BuildInsertPayload
from backend.app.main import (
    app,
    get_artifact_base_path,
    get_build_evaluator,
    get_repository,
)


def _ruleset_id() -> str:
    return "pob:local|scenarios:pinnacle@v0|prices:local"


class FakeRepository:
    def __init__(self) -> None:
        self._builds: dict[str, dict] = {}
        self._scenario_metrics: dict[str, list[dict]] = {}

    def insert_build(self, payload: BuildInsertPayload) -> None:
        self._builds[payload.build_id] = payload.model_dump(by_alias=True)

    def get_build(self, build_id: str) -> dict | None:
        return self._builds.get(build_id)

    def list_builds(self, *args, **kwargs) -> list[dict]:  # pragma: no cover - stub
        return list(self._builds.values())

    def insert_scenario_metrics(self, rows: list) -> None:
        for row in rows:
            self._scenario_metrics.setdefault(row.build_id, []).append(row.model_dump())

    def list_scenario_metrics(self, build_id: str) -> list[dict]:
        return list(self._scenario_metrics.get(build_id, []))

    def get_latest_build_cost(self, build_id: str) -> dict[str, Any] | None:
        return None

    def update_build_status(self, build_id: str, status: str) -> None:
        if build_id in self._builds:
            self._builds[build_id]["status"] = status

    def update_build_constraints(
        self,
        build_id: str,
        constraint_status: str | None = None,
        constraint_reason_code: str | None = None,
        violated_constraints: list[str] | None = None,
        constraint_checked_at: Any | None = None,
    ) -> None:
        entry = self._builds.get(build_id)
        if entry is None:
            return
        if constraint_status is not None:
            entry["constraint_status"] = constraint_status
        if constraint_reason_code is not None:
            entry["constraint_reason_code"] = constraint_reason_code
        if violated_constraints is not None:
            entry["violated_constraints"] = list(violated_constraints)
        if constraint_checked_at is not None:
            entry["constraint_checked_at"] = constraint_checked_at

    def purge_build(self, build_id: str) -> None:
        self._builds.pop(build_id, None)
        self._scenario_metrics.pop(build_id, None)


def _prepare_client(tmp_path: Path) -> tuple[TestClient, FakeRepository]:
    fake_repo = FakeRepository()
    app.dependency_overrides.clear()
    app.dependency_overrides[get_repository] = lambda: fake_repo
    app.dependency_overrides[get_artifact_base_path] = lambda: tmp_path
    app.dependency_overrides[get_build_evaluator] = lambda: BuildEvaluator(
        repo=fake_repo,
        base_path=tmp_path,
    )
    return TestClient(app), fake_repo


_ORIGINAL_DEFAULT_METRICS_GENERATOR = generation_runner._default_metrics_generator


def _verified_default_metrics_generator(seed: int, templates: list[Any]) -> dict[str, Any]:
    payload = _ORIGINAL_DEFAULT_METRICS_GENERATOR(seed, templates)
    for scenario in payload.values():
        if isinstance(scenario, dict):
            scenario.pop("warnings", None)
    return payload


def _decode_share_code(payload: str) -> str:
    encoded = payload.strip()
    padding = "=" * ((4 - len(encoded) % 4) % 4)
    compressed = base64.urlsafe_b64decode(encoded + padding)
    return zlib.decompress(compressed).decode("utf-8", errors="replace")


def test_generation_endpoints(tmp_path: Path) -> None:
    client, _repo = _prepare_client(tmp_path)
    try:
        payload = {
            "count": 1,
            "seed_start": 2,
            "ruleset_id": _ruleset_id(),
            "profile_id": "pinnacle",
        }
        with patch.object(
            generation_runner,
            "_default_metrics_generator",
            _verified_default_metrics_generator,
        ):
            response = client.post("/generation", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["generation"]["attempted"] == len(data["generation"]["attempt_records"])
        assert data["generation"]["attempt_records"]
        run_id = data["run_id"]
        assert run_id
        summary = client.get(f"/runs/{run_id}")
        assert summary.status_code == 200
        body = summary.json()
        assert body["run_id"] == run_id
        assert body["generation"]["attempted"] == data["generation"]["attempted"]
        assert body["generation"]["attempt_records"] == data["generation"]["attempt_records"]
        assert "benchmark" in body
        assert body["benchmark"]["scenarios"]
        assert "ml_lifecycle" in body
        assert not body["ml_lifecycle"]["enabled"]
        assert body["ml_lifecycle"]["metadata"]["error"] == "surrogate disabled"
        build_id = data["generation"]["records"][0]["build_id"]
        build_detail_response = client.get(f"/builds/{build_id}")
        assert build_detail_response.status_code == 200
        build_detail_body = build_detail_response.json()
        assert "constraints" in build_detail_body
        assert build_detail_body["constraints"] is None

        code_export_response = client.get(f"/builds/{build_id}/export/code")
        assert code_export_response.status_code == 200
        decoded_export = _decode_share_code(code_export_response.text)
        assert decoded_export.startswith("<PathOfBuilding>")

        missing = client.get("/runs/missing")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "run_not_found"
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_generation_validation_error(tmp_path: Path) -> None:
    client, _repo = _prepare_client(tmp_path)
    try:
        payload = {
            "count": 1,
            "seed_start": 0,
            "ruleset_id": _ruleset_id(),
            "profile_id": "missing-profile",
        }
        with patch.object(
            generation_runner,
            "_default_metrics_generator",
            _verified_default_metrics_generator,
        ):
            response = client.post("/generation", json=payload)
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_generation"
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_generation_does_not_require_worker_eval_flag(tmp_path: Path) -> None:
    client, _repo = _prepare_client(tmp_path)
    try:
        payload = {
            "count": 1,
            "seed_start": 0,
            "ruleset_id": _ruleset_id(),
            "profile_id": "pinnacle",
        }
        response = client.post("/generation", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["status"] in {"completed", "partial", "failed"}
        assert "ENABLE_WORKER_EVAL=true" not in json.dumps(body)
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_generation_rejects_invalid_run_id(tmp_path: Path) -> None:
    client, _repo = _prepare_client(tmp_path)
    try:
        payload = {
            "count": 1,
            "seed_start": 0,
            "ruleset_id": _ruleset_id(),
            "profile_id": "pinnacle",
            "run_id": "../evil",
        }
        with patch.object(
            generation_runner,
            "_default_metrics_generator",
            _verified_default_metrics_generator,
        ):
            response = client.post("/generation", json=payload)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_generation_api_accepts_optimizer_mode(tmp_path: Path) -> None:
    client, _repo = _prepare_client(tmp_path)
    try:
        payload = {
            "count": 1,
            "seed_start": 2,
            "ruleset_id": _ruleset_id(),
            "profile_id": "pinnacle",
            "run_mode": "optimizer",
            "optimizer_iterations": 1,
            "optimizer_elite_count": 1,
        }
        with patch.object(
            generation_runner,
            "_default_metrics_generator",
            _verified_default_metrics_generator,
        ):
            response = client.post("/generation", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["parameters"]["run_mode"] == "optimizer"
        assert data["optimizer"]["enabled"]
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_generation_constraints_round_trip(tmp_path: Path) -> None:
    client, _repo = _prepare_client(tmp_path)
    try:
        constraints_payload = {
            "schema_version": 1,
            "rules": [
                {
                    "code": "full_dps_positive",
                    "metric_path": "metrics.full_dps",
                    "operator": ">=",
                    "threshold": 0,
                    "reason_code": "full_dps_nonnegative",
                }
            ],
        }
        payload = {
            "count": 1,
            "seed_start": 3,
            "ruleset_id": _ruleset_id(),
            "profile_id": "pinnacle",
            "constraints": constraints_payload,
        }
        with patch.object(
            generation_runner,
            "_default_metrics_generator",
            _verified_default_metrics_generator,
        ):
            response = client.post("/generation", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["parameters"]["constraints"] == constraints_payload
        assert data["constraints"] == constraints_payload
        build_id = data["generation"]["records"][0]["build_id"]
        build_detail_response = client.get(f"/builds/{build_id}")
        assert build_detail_response.status_code == 200
        detail_body = build_detail_response.json()
        assert detail_body["constraints"] is not None
        assert detail_body["constraints"]["spec"]["rules"][0]["code"] == "full_dps_positive"
        assert detail_body["constraints"]["evaluation"]["status"] == "pass"
    finally:
        client.close()
        app.dependency_overrides.clear()
