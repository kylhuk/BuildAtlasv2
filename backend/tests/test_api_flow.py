from __future__ import annotations

import base64
import json
import shutil
import subprocess
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.app.api.evaluator import BuildEvaluator
from backend.app.db.ch import BuildInsertPayload, BuildListFilters
from backend.app.main import (
    ML_LOOP_REGISTRY,
    app,
    get_artifact_base_path,
    get_build_evaluator,
    get_repository,
)


class FakeRepository:
    def __init__(self) -> None:
        self._builds: dict[str, dict] = {}
        self._scenario_metrics: dict[str, list[dict]] = {}
        self._latest_build_costs: dict[str, dict] = {}
        self.last_list_filters: BuildListFilters | None = None

    def insert_build(self, payload: BuildInsertPayload) -> None:
        self._builds[payload.build_id] = payload.model_dump(by_alias=True)

    def get_build(self, build_id: str) -> dict | None:
        return self._builds.get(build_id)

    def list_builds(
        self,
        filters=None,
        sort_by=None,
        sort_dir=None,
        limit=None,
        offset=None,
    ) -> list[dict]:
        self.last_list_filters = filters
        builds = list(self._builds.values())
        if filters is not None and not getattr(filters, "include_stale", False):
            builds = [build for build in builds if not build.get("is_stale")]
        if filters is not None and getattr(filters, "status", None):
            builds = [build for build in builds if build.get("status") == filters.status]
        if filters is not None and getattr(filters, "verified_only", False):
            builds = [build for build in builds if build.get("status") in {"evaluated", "failed"}]
        start = offset or 0
        end = start + limit if limit is not None else None
        return builds[start:end]

    def insert_scenario_metrics(self, rows: list) -> None:
        for row in rows:
            self._scenario_metrics.setdefault(row.build_id, []).append(row.model_dump())

    def list_scenario_metrics(self, build_id: str) -> list[dict]:
        return list(self._scenario_metrics.get(build_id, []))

    def get_latest_build_cost(self, build_id: str) -> dict | None:
        return self._latest_build_costs.get(build_id)

    def set_latest_build_cost(self, build_id: str, row: dict) -> None:
        self._latest_build_costs[build_id] = row

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


def _prepare_client(tmp_path) -> tuple[TestClient, FakeRepository]:
    app.dependency_overrides.clear()
    fake_repo = FakeRepository()
    app.dependency_overrides[get_repository] = lambda: fake_repo
    app.dependency_overrides[get_artifact_base_path] = lambda: tmp_path
    app.dependency_overrides[get_build_evaluator] = lambda: BuildEvaluator(
        repo=fake_repo,
        base_path=tmp_path,
        worker_cmd="luajit",
        worker_args="pob/worker/worker.lua",
        worker_cwd=".",
    )
    return TestClient(app), fake_repo


def _write_cost_breakdown(tmp_path, build_id: str) -> None:
    slot_payload = {
        "total_cost_chaos": 321.5,
        "unknown_cost_count": 1,
        "slots": [
            {
                "slot": "Body",
                "name": "Headhunter",
                "cost_chaos": 200.0,
                "matched": True,
            },
            {
                "slot": "Helmet",
                "name": "Shavis",
                "cost_chaos": None,
                "matched": False,
            },
        ],
    }
    gem_payload = {
        "total_cost_chaos": 321.5,
        "unknown_cost_count": 1,
        "gems": [
            {
                "name": "Frostbolt",
                "level": 20,
                "quality": 23,
                "price": 50.0,
                "matched": True,
            },
            {
                "name": "Unknown Gem",
                "level": None,
                "quality": None,
                "price": None,
                "matched": False,
            },
        ],
    }
    cost_dir = tmp_path / "data" / "builds" / build_id
    cost_dir.mkdir(parents=True, exist_ok=True)
    (cost_dir / "slot_costs.json").write_text(json.dumps(slot_payload))
    (cost_dir / "gem_costs.json").write_text(json.dumps(gem_payload))


def _metrics_payload_for_profile(profile_id: str) -> dict[str, Any]:
    pinnacle_payload = {
        "metrics": {"full_dps": 6000.0, "max_hit": 6000.0, "utility_score": 1.0},
        "defense": {
            "armour": 4500.0,
            "evasion": 3000.0,
            "resists": {"fire": 90, "cold": 90, "lightning": 90, "chaos": 80},
        },
        "resources": {"life": 9500.0, "mana": 2500.0},
        "reservation": {"reserved_percent": 60, "available_percent": 90},
        "attributes": {"strength": 200, "dexterity": 200, "intelligence": 200},
    }
    if profile_id == "pinnacle":
        return {"pinnacle": pinnacle_payload}
    if profile_id == "delve":
        return {
            "delve_tier_1": pinnacle_payload,
            "delve_tier_2": {
                **pinnacle_payload,
                "metrics": {"full_dps": 5800.0, "max_hit": 6200.0, "utility_score": 1.5},
            },
            "delve_tier_3": {
                **pinnacle_payload,
                "metrics": {"full_dps": 5600.0, "max_hit": 6500.0, "utility_score": 2.0},
            },
        }
    if profile_id == "support":
        return {
            "support_party": {
                **pinnacle_payload,
                "metrics": {"full_dps": 1800.0, "max_hit": 3000.0, "utility_score": 4.0},
                "resources": {"life": 8200.0, "mana": 2200.0},
                "reservation": {"reserved_percent": 88, "available_percent": 96},
                "attributes": {"strength": 180, "dexterity": 170, "intelligence": 180},
            }
        }
    return {}


def _write_metrics(tmp_path, build_id: str, profile_id: str = "pinnacle") -> None:
    metrics_payload = _metrics_payload_for_profile(profile_id)
    metrics_file = tmp_path / "data" / "builds" / build_id / "metrics_raw.json"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(metrics_payload))


def _write_prediction(tmp_path, build_id: str) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "model_id": "ep-v4-baseline",
        "model_path": "/models/ep-v4.json",
        "predicted_metrics": {
            "full_dps": 5500.0,
            "max_hit": 5950.0,
        },
        "pass_probability": 0.78,
        "selection_reason": "surrogate_top",
        "timestamp": "2026-02-26T12:00:00Z",
    }
    prediction_path = tmp_path / "data" / "builds" / build_id / "surrogate_prediction.json"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text(json.dumps(payload))
    return payload


def _import_payload(profile_id: str = "pinnacle", ruleset_id: str | None = None) -> dict:
    active_ruleset = ruleset_id or f"pob:abc|scenarios:{profile_id}@v0|prices:demo"
    return {
        "xml": "<build><id>sample</id></build>",
        "metadata": {
            "ruleset_id": active_ruleset,
            "profile_id": profile_id,
            "class": "Marauder",
            "ascendancy": "Slayer",
            "main_skill": "Sunder",
            "damage_type": "physical",
            "defence_type": "armor",
            "complexity_bucket": "low",
            "tags": ["test"],
        },
    }


def _share_import_payload() -> dict:
    payload = _import_payload()
    payload.pop("xml", None)
    payload["share_code"] = "$PoB$sharecode"
    return payload


def _decode_share_code(payload: str) -> str:
    encoded = payload.strip()
    padding = "=" * ((4 - len(encoded) % 4) % 4)
    compressed = base64.urlsafe_b64decode(encoded + padding)
    return zlib.decompress(compressed).decode("utf-8", errors="replace")


def test_api_flow_import_evaluate_list_detail_export(tmp_path) -> None:
    client, fake_repo = _prepare_client(tmp_path)
    try:
        payload = _import_payload()
        import_response = client.post("/import", json=payload)
        assert import_response.status_code == 201
        response_body = import_response.json()
        build_id = response_body["build_id"]
        _write_metrics(tmp_path, build_id)
        prediction_payload = _write_prediction(tmp_path, build_id)

        predicted_metrics = prediction_payload["predicted_metrics"]

        evaluate_response = client.post(f"/evaluate/{build_id}")
        assert evaluate_response.status_code == 200
        assert evaluate_response.json()["status"] == "evaluated"
        _write_cost_breakdown(tmp_path, build_id)
        fake_repo.set_latest_build_cost(
            build_id,
            {
                "build_id": build_id,
                "ruleset_id": payload["metadata"]["ruleset_id"],
                "price_snapshot_id": "sentinel-2026-02-26T120000Z",
                "total_cost_chaos": 321.5,
                "unknown_cost_count": 1,
                "slot_costs_json_path": f"data/builds/{build_id}/slot_costs.json",
                "gem_costs_json_path": f"data/builds/{build_id}/gem_costs.json",
            },
        )
        list_response = client.get("/builds")
        assert list_response.status_code == 200
        builds = list_response.json()["builds"]
        assert builds and builds[0]["build_id"] == build_id

        stats_response = client.get("/builds/stats")
        assert stats_response.status_code == 200
        stats_payload = stats_response.json()
        assert stats_payload["total_builds"] >= 1
        assert stats_payload["status_counts"].get("evaluated", 0) >= 1

        prediction_summary = builds[0].get("prediction")
        assert prediction_summary
        assert prediction_summary["predicted_full_dps"] == predicted_metrics["full_dps"]
        assert prediction_summary["predicted_max_hit"] == predicted_metrics["max_hit"]
        assert prediction_summary["pass_probability"] == prediction_payload["pass_probability"]
        assert prediction_summary["selection_reason"] == prediction_payload["selection_reason"]

        detail_response = client.get(f"/builds/{build_id}")
        assert detail_response.status_code == 200
        detail_json = detail_response.json()
        assert detail_json["build"]["build_id"] == build_id
        assert detail_json["scenario_metrics"]
        assert detail_json["scenarios_used"]
        assert detail_json["costs"]
        cost_block = detail_json["costs"]
        assert cost_block["price_snapshot_id"] == "sentinel-2026-02-26T120000Z"
        assert cost_block["total_cost_chaos"] == 321.5
        assert cost_block["unknown_cost_count"] == 1
        assert any(not slot.get("matched") for slot in cost_block["slot_costs"])
        assert any(not gem.get("matched") for gem in cost_block["gem_costs"])

        detail_prediction = detail_json.get("prediction")
        assert detail_prediction
        assert detail_prediction["verified_full_dps"] == 6000.0
        assert detail_prediction["error_full_dps"] == abs(6000.0 - predicted_metrics["full_dps"])
        assert detail_prediction["error_max_hit"] == abs(6000.0 - predicted_metrics["max_hit"])
        assert detail_prediction["pass_probability"] == prediction_payload["pass_probability"]
        assert detail_json["build"]["prediction"]
        assert detail_json["build"]["prediction"]["model_id"] == detail_prediction["model_id"]

        scenario_response = client.get(f"/builds/{build_id}/scenarios")
        assert scenario_response.status_code == 200
        assert scenario_response.json()["build_id"] == build_id
        assert scenario_response.json()["scenarios_used"]

        xml_export = client.get(f"/builds/{build_id}/export/xml")
        assert xml_export.status_code == 200
        assert xml_export.text == "<build><id>sample</id></build>"

        code_export = client.get(f"/builds/{build_id}/export/code")
        assert code_export.status_code == 200
        decoded = _decode_share_code(code_export.text)
        assert decoded == "<build><id>sample</id></build>"

        requeue = client.post(f"/evaluate/{build_id}")
        assert requeue.status_code == 200
        assert requeue.json()["status"] == "evaluated"

        batch_response = client.post("/evaluate-batch", json={"build_ids": [build_id]})
        assert batch_response.status_code == 200
        results = batch_response.json()["results"]
        assert results and results[0]["status"] == "evaluated"
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_evaluate_error_cases(tmp_path) -> None:
    client, _repo = _prepare_client(tmp_path)
    try:
        missing = client.post("/evaluate/missing-build")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "build_not_found"

        import_response = client.post("/import", json=_import_payload())
        assert import_response.status_code == 201
        build_id = import_response.json()["build_id"]

        evaluate = client.post(f"/evaluate/{build_id}")
        assert evaluate.status_code in {400, 502}
        error_payload = evaluate.json()["error"]
        assert error_payload["code"] in {
            "missing_metrics",
            "worker_unavailable",
            "worker_response_error",
        }
        assert "ENABLE_WORKER_EVAL=true" not in json.dumps(error_payload)
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_share_code_import_exports_only_code(tmp_path) -> None:
    client, _repo = _prepare_client(tmp_path)
    try:
        payload = _share_import_payload()
        import_response = client.post("/import", json=payload)
        assert import_response.status_code == 201
        build_id = import_response.json()["build_id"]

        xml_export = client.get(f"/builds/{build_id}/export/xml")
        assert xml_export.status_code == 404
        assert xml_export.json()["error"]["code"] == "artifact_missing"

        build_dir = tmp_path / "data" / "builds" / build_id
        assert not (build_dir / "build.xml.gz").exists()

        code_export = client.get(f"/builds/{build_id}/export/code")
        assert code_export.status_code == 200
        assert code_export.text == payload["share_code"]
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_list_builds_skips_missing_code_artifacts(tmp_path) -> None:
    client, _repo = _prepare_client(tmp_path)
    try:
        import_response = client.post("/import", json=_import_payload())
        assert import_response.status_code == 201
        build_id = import_response.json()["build_id"]

        artifact_dir = tmp_path / "data" / "builds" / build_id
        shutil.rmtree(artifact_dir)

        list_response = client.get("/builds")
        assert list_response.status_code == 200
        returned_build_ids = {build["build_id"] for build in list_response.json()["builds"]}
        assert build_id not in returned_build_ids

        export_response = client.get(f"/builds/{build_id}/export/code")
        assert export_response.status_code == 404
        error_payload = export_response.json().get("error") or {}
        assert error_payload.get("code") == "artifacts_missing"
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_evaluate_falls_back_to_stub_metrics_when_worker_unavailable(tmp_path) -> None:
    client, _repo = _prepare_client(tmp_path)
    try:
        import_response = client.post("/import", json=_import_payload())
        assert import_response.status_code == 201
        build_id = import_response.json()["build_id"]

        _write_metrics(tmp_path, build_id)
        metrics_file = tmp_path / "data" / "builds" / build_id / "metrics_raw.json"
        metrics_payload = json.loads(metrics_file.read_text())
        for payload in metrics_payload.values():
            if isinstance(payload, dict):
                payload["warnings"] = ["generation_stub_metrics"]
        metrics_file.write_text(json.dumps(metrics_payload))

        with patch.object(BuildEvaluator, "_collect_worker_metrics", return_value={}):
            evaluate_response = client.post(f"/evaluate/{build_id}")

        assert evaluate_response.status_code == 200
        body = evaluate_response.json()
        assert body["build_id"] == build_id
        assert body["status"] in {"evaluated", "failed"}
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_evaluate_batch_errors_use_structured_payload(tmp_path) -> None:
    client, _repo = _prepare_client(tmp_path)
    try:
        batch_response = client.post("/evaluate-batch", json={"build_ids": ["missing-build"]})
        assert batch_response.status_code == 200
        results = batch_response.json()["results"]
        assert results
        error = results[0]["error"]
        assert error["code"] == "build_not_found"
        assert isinstance(error["message"], str)
        assert "details" in error
        assert error["details"] is None
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_list_builds_prediction_mode_verified_only(tmp_path) -> None:
    client, _repo = _prepare_client(tmp_path)
    try:
        first_import = client.post("/import", json=_import_payload())
        assert first_import.status_code == 201
        first_build_id = first_import.json()["build_id"]
        _write_metrics(tmp_path, first_build_id)
        first_eval = client.post(f"/evaluate/{first_build_id}")
        assert first_eval.status_code == 200

        second_import = client.post("/import", json=_import_payload())
        assert second_import.status_code == 201
        second_build_id = second_import.json()["build_id"]
        _write_prediction(tmp_path, second_build_id)

        third_import = client.post("/import", json=_import_payload())
        assert third_import.status_code == 201
        third_build_id = third_import.json()["build_id"]
        _write_metrics(tmp_path, third_build_id)
        third_eval = client.post(f"/evaluate/{third_build_id}")
        assert third_eval.status_code == 200

        all_response = client.get("/builds")
        assert all_response.status_code == 200
        all_ids = {entry["build_id"] for entry in all_response.json()["builds"]}
        assert {first_build_id, second_build_id, third_build_id}.issubset(all_ids)

        verified_page_one = client.get(
            "/builds", params={"prediction_mode": "verified_only", "limit": 1, "offset": 0}
        )
        assert verified_page_one.status_code == 200
        page_one_builds = verified_page_one.json()["builds"]
        assert len(page_one_builds) == 1

        verified_page_two = client.get(
            "/builds", params={"prediction_mode": "verified_only", "limit": 1, "offset": 1}
        )
        assert verified_page_two.status_code == 200
        page_two_builds = verified_page_two.json()["builds"]
        assert len(page_two_builds) == 1

        verified_ids = {page_one_builds[0]["build_id"], page_two_builds[0]["build_id"]}
        assert verified_ids == {first_build_id, third_build_id}
        assert page_one_builds[0]["status"] in {"evaluated", "failed"}
        assert page_two_builds[0]["status"] in {"evaluated", "failed"}
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_list_builds_filters_stale(tmp_path) -> None:
    client, fake_repo = _prepare_client(tmp_path)
    try:
        payload = _import_payload()
        import_response = client.post("/import", json=payload)
        assert import_response.status_code == 201
        build_id = import_response.json()["build_id"]
        fake_repo._builds[build_id]["is_stale"] = 1

        default_listing = client.get("/builds")
        assert default_listing.status_code == 200
        assert all(entry["build_id"] != build_id for entry in default_listing.json()["builds"])

        stale_listing = client.get("/builds", params={"include_stale": "true"})
        assert stale_listing.status_code == 200
        assert any(entry["build_id"] == build_id for entry in stale_listing.json()["builds"])
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_list_builds_constraints_filter_wiring(tmp_path) -> None:
    client, fake_repo = _prepare_client(tmp_path)
    try:
        import_response = client.post("/import", json=_import_payload())
        assert import_response.status_code == 201
        build_id = import_response.json()["build_id"]
        constraint_checked_at = datetime(2026, 2, 25, 15, 30, tzinfo=timezone.utc)
        fake_repo.update_build_constraints(
            build_id,
            constraint_status="locked",
            constraint_reason_code="inventory_locked",
            violated_constraints=["passives"],
            constraint_checked_at=constraint_checked_at,
        )
        checked_after = datetime(2026, 2, 25, 14, 30, tzinfo=timezone.utc)
        checked_before = datetime(2026, 2, 25, 16, 30, tzinfo=timezone.utc)
        params = {
            "constraint_status": "locked",
            "constraint_reason_code": "inventory_locked",
            "violated_constraint": "passives",
            "constraint_checked_after": checked_after.isoformat(),
            "constraint_checked_before": checked_before.isoformat(),
        }
        list_response = client.get("/builds", params=params)
        assert list_response.status_code == 200
        filters = fake_repo.last_list_filters
        assert filters is not None
        assert filters.constraint_status == "locked"
        assert filters.constraint_reason_code == "inventory_locked"
        assert filters.violated_constraint == "passives"
        assert filters.constraint_checked_after == checked_after
        assert filters.constraint_checked_before == checked_before

        builds = list_response.json()["builds"]
        assert builds and len(builds) == 1
        summary = builds[0]
        assert summary["build_id"] == build_id
        assert summary["constraint_status"] == "locked"
        assert summary["constraint_reason_code"] == "inventory_locked"
        assert summary["violated_constraints"] == ["passives"]
        assert datetime.fromisoformat(summary["constraint_checked_at"]) == constraint_checked_at
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_re_evaluation_preserves_metrics_history(tmp_path) -> None:
    client, fake_repo = _prepare_client(tmp_path)
    try:
        payload = _import_payload()
        import_response = client.post("/import", json=payload)
        assert import_response.status_code == 201
        build_id = import_response.json()["build_id"]
        _write_metrics(tmp_path, build_id)

        first_eval = client.post(f"/evaluate/{build_id}")
        assert first_eval.status_code == 200

        _write_metrics(tmp_path, build_id)
        second_eval = client.post(f"/evaluate/{build_id}")
        assert second_eval.status_code == 200

        rows = fake_repo._scenario_metrics.get(build_id, [])
        assert len(rows) == 2
        parsed = [
            datetime.fromisoformat(row["evaluated_at"])
            if isinstance(row["evaluated_at"], str)
            else row["evaluated_at"]
            for row in rows
        ]
        assert parsed[1] > parsed[0]
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_api_flow_evaluates_delve_and_support_profiles(tmp_path) -> None:
    client, _repo = _prepare_client(tmp_path)
    try:
        delve_import = client.post(
            "/import",
            json=_import_payload(
                profile_id="delve",
                ruleset_id="pob:abc|scenarios:delve_tier_3@v0|prices:demo",
            ),
        )
        assert delve_import.status_code == 201
        delve_build_id = delve_import.json()["build_id"]
        _write_metrics(tmp_path, delve_build_id, profile_id="delve")

        delve_eval = client.post(f"/evaluate/{delve_build_id}")
        assert delve_eval.status_code == 200
        delve_body = delve_eval.json()
        assert delve_body["status"] == "evaluated"
        delve_results = delve_body["scenario_results"]
        assert {entry["scenario_id"] for entry in delve_results} == {
            "delve_tier_1",
            "delve_tier_2",
            "delve_tier_3",
        }
        assert all(entry["utility_score"] > 2.0 for entry in delve_results)

        support_import = client.post(
            "/import",
            json=_import_payload(
                profile_id="support",
                ruleset_id="pob:abc|scenarios:support_party@v0|prices:demo",
            ),
        )
        assert support_import.status_code == 201
        support_build_id = support_import.json()["build_id"]
        _write_metrics(tmp_path, support_build_id, profile_id="support")

        support_eval = client.post(f"/evaluate/{support_build_id}")
        assert support_eval.status_code == 200
        support_body = support_eval.json()
        assert support_body["status"] == "evaluated"
        support_results = support_body["scenario_results"]
        assert len(support_results) == 1
        assert support_results[0]["scenario_id"] == "support_party"
        assert support_results[0]["utility_score"] > 4.0
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_ml_loop_status_endpoint_reads_latest_iteration(tmp_path) -> None:
    client, _repo = _prepare_client(tmp_path)
    try:
        loop_root = tmp_path / "ml_loops" / "loop-a"
        loop_root.mkdir(parents=True, exist_ok=True)
        state_payload = {
            "schema_version": 1,
            "loop_id": "loop-a",
            "status": "running",
            "phase": "evaluate",
            "iteration": 2,
            "total_iterations": 5,
            "stop_requested": False,
            "started_at_utc": "2026-01-01T00:00:00+00:00",
            "updated_at_utc": "2026-01-01T00:05:00+00:00",
            "last_error": None,
            "last_run_id": "loop-a-iter-0002",
            "last_snapshot_id": "iter-0002",
            "last_model_path": "models/loop-a-iter-0002/model.json",
            "last_improvement": {
                "improved": True,
                "metric_mae_deltas": {"full_dps": 0.4},
                "pass_probability_mean_delta": 0.03,
            },
        }
        (loop_root / "state.json").write_text(json.dumps(state_payload), encoding="utf-8")
        iteration_one = {
            "iteration": 1,
            "run_id": "loop-a-iter-0001",
            "run_status": "completed",
            "evaluation": {
                "current": {"metric_mae": {"full_dps": 1.2}},
                "improvement": {"improved": False},
            },
        }
        iteration_two = {
            "iteration": 2,
            "run_id": "loop-a-iter-0002",
            "run_status": "completed",
            "evaluation": {
                "current": {"metric_mae": {"full_dps": 0.8}},
                "improvement": {"improved": True},
            },
        }
        iterations_path = loop_root / "iterations.jsonl"
        iterations_path.write_text(
            "\n".join([json.dumps(iteration_one), json.dumps(iteration_two)]) + "\n",
            encoding="utf-8",
        )

        response = client.get("/ops/ml-loop-status")
        assert response.status_code == 200
        payload = response.json()
        assert payload["loop_id"] == "loop-a"
        assert payload["latest_iteration"]["iteration"] == 2
        assert payload["previous_iteration"]["iteration"] == 1
        assert payload["warnings"] == []

        explicit = client.get("/ops/ml-loop-status", params={"loop_id": "loop-a"})
        assert explicit.status_code == 200
        explicit_payload = explicit.json()
        assert explicit_payload["loop_id"] == "loop-a"
        assert explicit_payload["latest_iteration"]["run_id"] == "loop-a-iter-0002"
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_ml_loop_start_endpoint_spawns_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ML_LOOP_REGISTRY.clear()
    client, _repo = _prepare_client(tmp_path)
    try:
        calls: list[list[str]] = []

        class FakeProcess:
            def __init__(self) -> None:
                self.pid = 1234

            def poll(self) -> None:
                return None

        def fake_popen(
            cmd: list[str],
            stdout: subprocess.Popen | None = None,
            stderr: subprocess.Popen | None = None,
        ) -> FakeProcess:
            calls.append(cmd)
            return FakeProcess()

        monkeypatch.setattr("backend.app.main.subprocess.Popen", fake_popen)
        response = client.post(
            "/ops/ml-loop-start",
            json={
                "loop_id": "api-loop",
                "count": 2,
                "seed_start": 5,
                "profile_id": "pinnacle",
                "surrogate_backend": "cpu",
                "endless": True,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["loop_id"] == "api-loop"
        assert payload["endless"] is True
        assert payload["status"] == "started"
        assert payload["pid"] == 1234
        assert calls
        assert "--iterations" in calls[0]
        assert "0" in calls[0]
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_ml_loop_stop_endpoint_defaults_to_last_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ML_LOOP_REGISTRY.clear()
    client, _repo = _prepare_client(tmp_path)
    try:

        class FakeProcess:
            def __init__(self) -> None:
                self.pid = 4321

            def poll(self) -> None:
                return None

        monkeypatch.setattr(
            "backend.app.main.subprocess.Popen", lambda *args, **kwargs: FakeProcess()
        )
        start_response = client.post(
            "/ops/ml-loop-start",
            json={
                "loop_id": "stop-loop",
                "count": 1,
                "seed_start": 1,
                "profile_id": "pinnacle",
                "surrogate_backend": "cpu",
                "endless": True,
            },
        )
        assert start_response.status_code == 200

        run_calls: list[list[str]] = []
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        def fake_run(cmd, capture_output=True, text=True):
            run_calls.append(cmd)
            return completed

        monkeypatch.setattr("backend.app.main.subprocess.run", fake_run)
        stop_response = client.post("/ops/ml-loop-stop", json={})
        assert stop_response.status_code == 200
        payload = stop_response.json()
        assert payload["loop_id"] == "stop-loop"
        assert payload["stop_requested"] is True
        assert run_calls
        assert "--loop-id" in run_calls[0]
        assert "stop-loop" in run_calls[0]
    finally:
        client.close()
        app.dependency_overrides.clear()
