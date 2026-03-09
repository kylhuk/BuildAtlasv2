from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from backend.app.api.errors import APIError
from backend.app.api.evaluator import BuildEvaluator
from backend.app.api.models import BuildStatus
from backend.engine.metrics_source import METRICS_SOURCE_POB, METRICS_SOURCE_STUB
from backend.engine.scenarios.loader import (
    ScenarioGateThresholds,
    ScenarioReservationThreshold,
    ScenarioTemplate,
)


# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownLambdaType=false
class FakeRepository:
    def __init__(self) -> None:
        self._builds: dict[str, dict[str, Any]] = {
            "build-1": {"build_id": "build-1", "status": BuildStatus.imported.value}
        }
        self.status_updates: list[str] = []
        self.inserted_rows: list[Any] = []

    def get_build(self, build_id: str) -> dict[str, Any] | None:
        return self._builds.get(build_id)

    def update_build_status(self, build_id: str, status: str) -> None:
        if build_id in self._builds:
            self._builds[build_id]["status"] = status
            self.status_updates.append(status)

    def insert_scenario_metrics(self, rows: list[Any]) -> None:
        self.inserted_rows = list(rows)


def test_evaluate_build_marks_evaluated_when_gate_fails(tmp_path: Path, monkeypatch) -> None:
    repo = FakeRepository()
    evaluator = BuildEvaluator(repo=repo, base_path=tmp_path)
    gate_failed_row = SimpleNamespace(gate_pass=False)
    monkeypatch.setattr(
        evaluator,
        "_collect_scenario_rows",
        lambda _build, *, gate_thresholds=None: [gate_failed_row],
    )

    status, rows = evaluator.evaluate_build("build-1")

    assert status is BuildStatus.evaluated
    assert rows == [gate_failed_row]
    assert repo.inserted_rows == [gate_failed_row]
    assert repo.status_updates == [BuildStatus.queued.value, BuildStatus.evaluated.value]


def test_worker_configuration_defaults_to_settings(monkeypatch, tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    worker_script = "backend/pob_worker/pob_worker.lua"

    monkeypatch.setattr("backend.app.api.evaluator.settings.pob_worker_cmd", "luajit")
    monkeypatch.setattr(
        "backend.app.api.evaluator.settings.pob_worker_args",
        f"{worker_script} --log-level debug",
    )
    monkeypatch.setattr("backend.app.api.evaluator.settings.pob_worker_cwd", "backend")
    monkeypatch.setattr("backend.app.api.evaluator.settings.pob_worker_pool_size", 3)

    evaluator = BuildEvaluator(repo=FakeRepository(), base_path=tmp_path)

    assert evaluator._worker_cmd == "luajit"
    assert evaluator._worker_args[0] == str(project_root / worker_script)
    assert evaluator._worker_args[1:] == ["--log-level", "debug"]
    assert evaluator._worker_cwd == str(project_root / "backend")
    assert evaluator._worker_pool_size == 3


def test_metrics_source_detects_stub_from_warnings(tmp_path: Path) -> None:
    evaluator = BuildEvaluator(repo=FakeRepository(), base_path=tmp_path)

    metrics_source = evaluator._worker_payload_metrics_source(
        {
            "warnings": ["generation_stub_metrics"],
        }
    )

    assert metrics_source == METRICS_SOURCE_STUB


def test_metrics_source_preserves_pob_when_explicit(tmp_path: Path) -> None:
    evaluator = BuildEvaluator(repo=FakeRepository(), base_path=tmp_path)

    metrics_source = evaluator._worker_payload_metrics_source(
        {
            "metrics_source": METRICS_SOURCE_POB,
            "warnings": [],
        }
    )

    assert metrics_source == METRICS_SOURCE_POB


def test_apply_gate_threshold_overrides_accepts_dataclass_override(tmp_path: Path) -> None:
    evaluator = BuildEvaluator(repo=FakeRepository(), base_path=tmp_path)
    base_template = ScenarioTemplate(
        scenario_id="pinnacle",
        version="v0",
        profile_id="pinnacle",
        pob_config={"boss": "Pinnacle"},
        gate_thresholds=ScenarioGateThresholds(
            resists={"fire": 75.0, "cold": 75.0, "lightning": 75.0, "chaos": 0.0},
            attributes={},
            reservation=ScenarioReservationThreshold(max_percent=100.0),
            min_max_hit=3000.0,
            min_full_dps=100000.0,
        ),
    )
    override = ScenarioGateThresholds(
        resists={"fire": 0.0, "cold": 0.0, "lightning": 0.0, "chaos": 0.0},
        attributes={},
        reservation=ScenarioReservationThreshold(max_percent=100.0),
        min_max_hit=0.0,
        min_full_dps=0.0,
    )

    overridden = evaluator._apply_gate_threshold_overrides([base_template], override)

    assert overridden[0].gate_thresholds == override


def test_prepare_build_evaluation_reuses_cached_templates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = FakeRepository()
    evaluator = BuildEvaluator(repo=repo, base_path=tmp_path)
    build = {"build_id": "build-1", "profile_id": "pinnacle"}

    monkeypatch.setattr(
        "backend.app.api.evaluator.read_build_artifacts",
        lambda build_id, *, base_path: {"build_id": build_id, "base_path": str(base_path)},
    )

    template = ScenarioTemplate(
        scenario_id="pinnacle",
        version="v0",
        profile_id="pinnacle",
        pob_config={"boss": "Pinnacle"},
        gate_thresholds=ScenarioGateThresholds(
            resists={"fire": 75.0, "cold": 75.0, "lightning": 75.0, "chaos": 0.0},
            attributes={},
            reservation=ScenarioReservationThreshold(max_percent=100.0),
            min_max_hit=3000.0,
            min_full_dps=100000.0,
        ),
    )
    calls = {"count": 0}

    def fake_list_templates() -> list[ScenarioTemplate]:
        calls["count"] += 1
        return [template]

    monkeypatch.setattr("backend.app.api.evaluator.list_templates", fake_list_templates)

    first = evaluator._prepare_build_evaluation(build)
    second = evaluator._prepare_build_evaluation(build)

    assert calls["count"] == 1
    assert [t.scenario_id for t in first.templates] == ["pinnacle"]
    assert [t.scenario_id for t in second.templates] == ["pinnacle"]


def test_prepare_build_evaluation_raises_for_missing_profile_templates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = FakeRepository()
    evaluator = BuildEvaluator(repo=repo, base_path=tmp_path)
    build = {"build_id": "build-1", "profile_id": "unknown-profile"}

    monkeypatch.setattr(
        "backend.app.api.evaluator.read_build_artifacts",
        lambda build_id, *, base_path: {"build_id": build_id, "base_path": str(base_path)},
    )
    monkeypatch.setattr("backend.app.api.evaluator.list_templates", lambda: [])

    with pytest.raises(APIError, match="no scenario templates configured"):
        evaluator._prepare_build_evaluation(build)
