from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.app.api.evaluator import BuildEvaluator
from backend.app.main import (
    app,
    get_artifact_base_path,
    get_build_evaluator,
    get_repository,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class FakeRepository:
    def __init__(self) -> None:
        self._builds: dict[str, dict[str, Any]] = {}
        self._scenario_metrics: dict[str, list[dict[str, Any]]] = {}

    def insert_build(self, payload: Any) -> None:
        self._builds[payload.build_id] = payload.model_dump(by_alias=True)

    def get_build(self, build_id: str) -> dict[str, Any] | None:
        return self._builds.get(build_id)

    def list_builds(
        self,
        filters=None,
        sort_by=None,
        sort_dir=None,
        limit=None,
        offset=None,
    ) -> list[dict[str, Any]]:
        builds = list(self._builds.values())
        start = offset or 0
        end = start + limit if limit is not None else None
        return builds[start:end]

    def insert_scenario_metrics(self, rows: list[Any]) -> None:
        for row in rows:
            self._scenario_metrics.setdefault(row.build_id, []).append(row.model_dump())

    def list_scenario_metrics(self, build_id: str) -> list[dict[str, Any]]:
        return list(self._scenario_metrics.get(build_id, []))

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


def _prepare_client(tmp_path: Path) -> tuple[TestClient, FakeRepository]:
    app.dependency_overrides.clear()
    fake_repo = FakeRepository()
    app.dependency_overrides[get_repository] = lambda: fake_repo
    app.dependency_overrides[get_artifact_base_path] = lambda: tmp_path
    app.dependency_overrides[get_build_evaluator] = lambda: BuildEvaluator(
        repo=fake_repo,
        base_path=tmp_path,
    )
    return TestClient(app), fake_repo


def _load_fixture_text(name: str) -> str:
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


def _load_fixture_json(name: str) -> dict[str, Any]:
    return json.loads(_load_fixture_text(name))


def _write_fixture_metrics(tmp_path: Path, build_id: str, fixture_name: str) -> None:
    payload = _load_fixture_json(fixture_name)
    metrics_file = tmp_path / "data" / "builds" / build_id / "metrics_raw.json"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(payload, indent=2))


def _evaluate_build(client: TestClient, build_id: str) -> dict[str, Any]:
    response = client.post(f"/evaluate/{build_id}")
    assert response.status_code == 200, response.text
    return response.json()


def _find_scenario_result(results: list[dict[str, Any]], scenario_id: str) -> dict[str, Any]:
    for result in results:
        if result.get("scenario_id") == scenario_id:
            return result
    raise AssertionError(f"scenario {scenario_id} missing in results")


def test_e2e_smoke_baseline_builds(tmp_path: Path) -> None:
    client, _repo = _prepare_client(tmp_path)
    try:
        share_code_text = _load_fixture_text("sample_share_code.txt").strip()
        assert share_code_text, "share code fixture must provide content"
        mapping_payload = {
            "share_code": share_code_text,
            "metadata": {
                "ruleset_id": "pob:smoke|scenarios:mapping_t16@v0|prices:demo",
                "profile_id": "mapping",
                "class": "Marauder",
                "ascendancy": "Slayer",
                "main_skill": "Sunder",
                "damage_type": "physical",
                "defence_type": "armor",
                "complexity_bucket": "low",
                "tags": ["smoke"],
            },
        }
        mapping_response = client.post("/import", json=mapping_payload)
        assert mapping_response.status_code == 201
        mapping_build_id = mapping_response.json()["build_id"]
        _write_fixture_metrics(tmp_path, mapping_build_id, "mapping_t16_metrics.json")
        mapping_eval = _evaluate_build(client, mapping_build_id)
        mapping_results = mapping_eval["scenario_results"]
        assert mapping_eval["status"] == "evaluated"
        assert mapping_results, "expected scenario metrics for mapping fixture"
        mapping_row = _find_scenario_result(mapping_results, "mapping_t16")
        assert mapping_row["ruleset_id"] == mapping_payload["metadata"]["ruleset_id"]
        assert mapping_row["gate_pass"] is True
        assert float(mapping_row["full_dps"]) > 5000
        assert float(mapping_row["max_hit"]) > 4500
        assert not mapping_row["gate_fail_reasons"], "mapping fixture should pass gates"

        xml_text = _load_fixture_text("sample_build.xml")
        uber_payload = {
            "xml": xml_text,
            "metadata": {
                "ruleset_id": "pob:smoke|scenarios:uber_pinnacle@v0|prices:demo",
                "profile_id": "uber_pinnacle",
                "class": "Ranger",
                "ascendancy": "Deadeye",
                "main_skill": "Tornado",
                "damage_type": "physical",
                "defence_type": "evasion",
                "complexity_bucket": "high",
                "tags": ["smoke", "uber"],
            },
        }
        uber_response = client.post("/import", json=uber_payload)
        assert uber_response.status_code == 201
        uber_build_id = uber_response.json()["build_id"]
        _write_fixture_metrics(tmp_path, uber_build_id, "uber_pinnacle_metrics.json")
        uber_eval = _evaluate_build(client, uber_build_id)
        uber_results = uber_eval["scenario_results"]
        assert uber_eval["status"] == "evaluated"
        assert uber_results, "expected scenario metrics for uber fixture"
        uber_row = _find_scenario_result(uber_results, "uber_pinnacle")
        assert uber_row["ruleset_id"] == uber_payload["metadata"]["ruleset_id"]
        assert uber_row["gate_pass"] is False
        assert uber_row["gate_fail_reasons"], "uber fixture should fail at least one gate"

        mapping_full = float(mapping_row["full_dps"])
        uber_full = float(uber_row["full_dps"])
        mapping_hit = float(mapping_row["max_hit"])
        uber_hit = float(uber_row["max_hit"])
        drift_segments = []
        for metric, source, target in (
            ("full_dps", mapping_full, uber_full),
            ("max_hit", mapping_hit, uber_hit),
        ):
            drift_segments.append(f"{metric}: {source:.1f}->{target:.1f} ({source - target:+.1f})")
        drift_report = "; ".join(drift_segments)
        drift_caption = f"{mapping_row['scenario_id']}->{uber_row['scenario_id']}"
        assert mapping_full - uber_full >= 1500.0, (
            f"{drift_caption} drift insufficient; {drift_report}"
        )
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_lua_worker_ping_runtime() -> None:
    lua_path = shutil.which("lua")
    if not lua_path:
        pytest.skip("Lua runtime not installed; skipping worker ping smoke test")
    cjson_check = subprocess.run(
        [lua_path, "-e", "require('cjson.safe')"],
        capture_output=True,
        text=True,
    )
    if cjson_check.returncode != 0:
        reason = cjson_check.stderr.strip() or cjson_check.stdout.strip()
        pytest.skip(
            "Lua runtime cannot load cjson.safe; skipping worker ping smoke test"
            + (f" ({reason})" if reason else "")
        )
    worker_script = Path("pob") / "worker" / "worker.lua"
    assert worker_script.exists(), "worker entrypoint missing"
    process = subprocess.Popen(
        [lua_path, str(worker_script)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdin is not None and process.stdout is not None
        payload = json.dumps({"id": 1, "method": "ping", "params": {}}) + "\n"
        process.stdin.write(payload)
        process.stdin.flush()
        response_line = process.stdout.readline()
        assert response_line, "lua worker ping returned no data"
        response = json.loads(response_line)
        assert response.get("ok") is True
        assert response.get("result", {}).get("protocol") == "ndjson"
    finally:
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
