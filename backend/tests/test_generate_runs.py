from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import Mock, patch

import pytest

from backend.app.api.errors import APIError
from backend.app.api.evaluator import BuildEvaluator, EvaluationProvenance
from backend.app.db.ch import BuildInsertPayload, ScenarioMetricRow
from backend.engine.archive import load_archive_artifact
from backend.engine.artifacts.store import read_build_artifacts
from backend.engine.generation import runner as generation_runner
from backend.engine.genome import GenomeV0
from backend.engine.items.templates import (
    ItemTemplatePlan,
    RepairReport,
    RequirementDeficits,
    build_item_templates,
)
from backend.engine.sockets.planner import PlanIssue, SocketPlan, plan_sockets
from backend.tools import generate_runs as generate_runs_cli


def _fake_passive_tree_plan(genome: GenomeV0) -> generation_runner.PassiveTreePlan:
    from backend.engine.passives.builder import PassiveGraphNode, PassiveTreePlan

    return PassiveTreePlan(
        genome=genome,
        nodes=(PassiveGraphNode(id="fake_start", kind="start", pob_id="1"),),
        required_targets=(),
    )


class FakeRepository:
    def __init__(self) -> None:
        self._builds: dict[str, dict[str, Any]] = {}
        self._scenario_metrics: dict[str, list[dict[str, Any]]] = {}

    def insert_build(self, payload: BuildInsertPayload) -> None:
        self._builds[payload.build_id] = payload.model_dump(by_alias=True)

    def get_build(self, build_id: str) -> dict[str, Any] | None:
        return self._builds.get(build_id)

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

    def insert_scenario_metrics(self, rows: list[Any]) -> None:
        for row in rows:
            self._scenario_metrics.setdefault(row.build_id, []).append(row.model_dump())

    def list_scenario_metrics(self, build_id: str) -> list[dict[str, Any]]:
        return list(self._scenario_metrics.get(build_id, []))

    def purge_build(self, build_id: str) -> None:
        self._builds.pop(build_id, None)
        self._scenario_metrics.pop(build_id, None)


def _fake_evaluator(tmp_path: Path, repo: FakeRepository) -> BuildEvaluator:
    return BuildEvaluator(
        repo=repo,
        base_path=tmp_path,
        worker_args="pob_worker/mock_worker.lua",
    )


def _ruleset_id() -> str:
    return "pob:local|scenarios:pinnacle@v0|prices:local"


def _budget_constraint_spec() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "rules": [
            {
                "code": "budget_cost",
                "metric_path": "metrics.total_cost_chaos",
                "operator": "<=",
                "threshold": 100,
                "reason_code": "budget_within_limit",
                "missing_data_reason": "budget_data_missing",
            }
        ],
    }


def _metrics_with_cost(seed: int, templates: list[Any]) -> dict[str, Any]:
    payload = _verified_metrics_generator(seed, templates)
    cost_value = 60 if seed % 2 == 0 else 180
    for scenario in payload.values():
        metrics = scenario.setdefault("metrics", {})
        metrics["total_cost_chaos"] = cost_value
    return payload


def _strip_stub_warnings(payload: dict[str, Any]) -> None:
    for scenario in payload.values():
        if isinstance(scenario, dict):
            scenario.pop("warnings", None)


def _verified_metrics_generator(seed: int, templates: list[Any]) -> dict[str, Any]:
    payload = generation_runner._default_metrics_generator(seed, templates)
    _strip_stub_warnings(payload)
    return payload


def _build_candidate_for_selection(
    *,
    build_id: str,
    seed: int,
    class_name: str,
    main_skill: str,
    ascendancy: str = "Chieftain",
    defense_archetype: str = "armour",
    identity: dict[str, str] | None = None,
) -> generation_runner.Candidate:
    genome = GenomeV0(
        seed=seed,
        class_name=class_name,
        ascendancy=ascendancy,
        main_skill_package=main_skill,
        defense_archetype=defense_archetype,
        budget_tier="endgame",
        profile_id="pinnacle",
    )
    build_details_payload = {"identity": identity} if identity is not None else {}
    return generation_runner.Candidate(
        seed=seed,
        build_id=build_id,
        main_skill_package=main_skill,
        class_name=class_name,
        ascendancy=ascendancy,
        budget_tier="endgame",
        failures=[],
        metrics_payload={},
        genome=genome,
        code_payload="",
        build_details_payload=build_details_payload,
    )


@pytest.mark.parametrize(
    ("character_level", "expected_budget"),
    [
        (0, 55),
        (1, 55),
        (2, 70),
        (3, 90),
        (4, 123),
        (5, 123),
    ],
)
def test_generation_uses_stage_passive_budget(
    tmp_path: Path,
    character_level: int,
    expected_budget: int,
) -> None:
    repo = FakeRepository()
    evaluator = _fake_evaluator(tmp_path, repo)
    captured_budgets: list[int] = []

    def _capture_budget(genome: GenomeV0, point_budget: int, *args, **kwargs):
        captured_budgets.append(point_budget)
        return _fake_passive_tree_plan(genome)

    with patch.object(generation_runner, "build_passive_tree_plan", _capture_budget):
        generation_runner.run_generation(
            count=1,
            seed_start=1,
            ruleset_id=_ruleset_id(),
            profile_id="pinnacle",
            base_path=tmp_path,
            repo=repo,
            evaluator=evaluator,
            metrics_generator=_verified_metrics_generator,
            run_id=f"stage-budget-{character_level}",
            candidate_pool_size=1,
            character_level=character_level,
        )

    assert captured_budgets
    assert set(captured_budgets) == {expected_budget}


def test_generation_run_summary(tmp_path: Path) -> None:
    repo = FakeRepository()
    evaluator = _fake_evaluator(tmp_path, repo)
    summary = generation_runner.run_generation(
        count=2,
        seed_start=1,
        ruleset_id=_ruleset_id(),
        profile_id="pinnacle",
        base_path=tmp_path,
        repo=repo,
        evaluator=evaluator,
        metrics_generator=_verified_metrics_generator,
        run_id="test-run",
    )

    assert summary["run_id"] == "test-run"
    assert isinstance(summary["paths"]["summary"], str)
    path = Path(summary["paths"]["summary"])
    assert path.exists()

    precheck = summary["generation"]["precheck_failures"]
    assert set(precheck) == set(generation_runner.PRECHECK_CATEGORIES)
    processed = summary["generation"]["processed"]
    total_prechecks = sum(precheck.values())
    assert summary["evaluation"]["attempted"] == processed - total_prechecks
    assert summary["evaluation"]["successes"] >= 0
    surrogate = summary["surrogate"]
    assert surrogate["status"] == "disabled"
    assert surrogate["counts"]["selected"] == summary["evaluation"]["attempted"]
    assert surrogate["counts"]["pruned"] == 0

    benchmark = summary["benchmark"]
    assert benchmark["scenarios"]
    benchmark_path = Path(summary["paths"]["benchmark_summary"])
    assert benchmark_path.exists()

    ml = summary["ml_lifecycle"]
    assert not ml["enabled"]
    metadata = ml["metadata"]
    assert metadata["model_meta"] is None
    assert metadata["error"] == "surrogate disabled"
    ml_path = Path(summary["paths"]["ml_lifecycle"])
    assert ml_path.exists()


def test_generation_logs_run_phases(tmp_path: Path, caplog) -> None:
    repo = FakeRepository()
    evaluator = _fake_evaluator(tmp_path, repo)

    with caplog.at_level(logging.INFO, logger=generation_runner.__name__):
        generation_runner.run_generation(
            count=2,
            seed_start=1,
            ruleset_id=_ruleset_id(),
            profile_id="pinnacle",
            base_path=tmp_path,
            repo=repo,
            evaluator=evaluator,
            metrics_generator=_verified_metrics_generator,
            run_id="progress-log-run",
        )

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == generation_runner.__name__ and "progress-log-run" in record.getMessage()
    ]
    assert any("starting (mode=" in message for message in messages)
    assert any("phase=synthesis_complete" in message for message in messages)
    assert any("phase=selection_complete" in message for message in messages)
    assert any(
        "starting PoB evaluation" in message
        or "has no candidates selected for evaluation" in message
        for message in messages
    )
    assert any("phase=evaluation_complete" in message for message in messages)
    assert any("phase=archive_complete" in message for message in messages)
    assert any("phase=artifact_write_start" in message for message in messages)


def test_generation_completes_when_metrics_ok_but_no_gate_pass(tmp_path: Path) -> None:
    repo = FakeRepository()
    evaluator = _fake_evaluator(tmp_path, repo)

    def _evaluate_builds_batched(
        build_ids: list[str],
        gate_thresholds: Any = None,
        progress_label: str | None = None,
    ) -> dict[str, Any]:
        del gate_thresholds, progress_label
        return {
            "results": {
                build_id: (
                    generation_runner.BuildStatus.evaluated,
                    [
                        ScenarioMetricRow(
                            build_id=build_id,
                            ruleset_id=_ruleset_id(),
                            scenario_id="pinnacle_boss",
                            gate_pass=False,
                            gate_fail_reasons=["min_max_hit"],
                            pob_warnings=[],
                            evaluated_at=datetime.now(UTC),
                            full_dps=1200.0,
                            max_hit=1800.0,
                            armour=1000.0,
                            evasion=500.0,
                            life=4000.0,
                            mana=700.0,
                            utility_score=44.0,
                            metrics_source="pob",
                        )
                    ],
                )
                for build_id in build_ids
            },
            "errors": {},
        }

    mock_evaluate_builds_batched = Mock(side_effect=_evaluate_builds_batched)
    with patch.object(evaluator, "evaluate_builds_batched", mock_evaluate_builds_batched):
        summary = generation_runner.run_generation(
            count=1,
            seed_start=13,
            ruleset_id=_ruleset_id(),
            profile_id="pinnacle",
            base_path=tmp_path,
            repo=repo,
            evaluator=evaluator,
            run_id="no-gate-pass-run",
            metrics_generator=_verified_metrics_generator,
        )

    assert mock_evaluate_builds_batched.call_count == 1
    build_ids = mock_evaluate_builds_batched.call_args.args[0]
    assert len(build_ids) == 1

    assert summary["status"] == "completed"
    reason = summary.get("status_reason")
    assert reason and reason.get("code") == "no_gate_pass_builds"

    assert summary["evaluation"]["successes"] == summary["evaluation"]["attempted"]
    assert summary["evaluation"]["metrics_ok_count"] == summary["evaluation"]["attempted"]
    assert summary["evaluation"]["gate_pass_count"] == 0
    assert summary["evaluation"]["gate_fail_count"] == summary["evaluation"]["attempted"]
    assert summary["evaluation"]["failures"] == 0
    assert summary["evaluation"]["errors"] == 0
    scenarios = summary["benchmark"]["scenarios"]
    assert scenarios
    assert all(payload["gate_pass_rate"] == 0.0 for payload in scenarios.values())
    assert summary["benchmark"]["gate_fail_reason_counts"]
    assert summary["generation"]["records"] == []
    attempt_records = summary["generation"]["attempt_records"]
    assert attempt_records
    assert all(record.get("persisted") is True for record in attempt_records)
    assert all(record.get("evaluation_status") == "evaluated" for record in attempt_records)
    assert all(record.get("gate_pass") is False for record in attempt_records)


def test_generation_uses_settings_worker_config_when_evaluator_not_provided(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = FakeRepository()
    monkeypatch.setattr(generation_runner.settings, "pob_worker_cmd", "luajit-custom")
    monkeypatch.setattr(
        generation_runner.settings,
        "pob_worker_args",
        "backend/pob_worker/pob_worker.lua --trace",
    )
    monkeypatch.setattr(generation_runner.settings, "pob_worker_cwd", "PathOfBuilding/src")
    monkeypatch.setattr(generation_runner.settings, "pob_worker_pool_size", 7)

    evaluator_instance = Mock()

    def _evaluate_build(
        build_id: str,
    ) -> tuple[generation_runner.BuildStatus, list[ScenarioMetricRow]]:
        row = ScenarioMetricRow(
            build_id=build_id,
            ruleset_id=_ruleset_id(),
            scenario_id="pinnacle_boss",
            gate_pass=True,
            gate_fail_reasons=[],
            pob_warnings=[],
            evaluated_at=datetime.now(UTC),
            full_dps=2200.0,
            max_hit=3200.0,
            armour=1000.0,
            evasion=500.0,
            life=4000.0,
            mana=700.0,
            utility_score=44.0,
            metrics_source="pob",
        )
        return generation_runner.BuildStatus.evaluated, [row]

    def _evaluate_builds_batched(
        build_ids: list[str],
        gate_thresholds: Any = None,
        progress_label: str | None = None,
    ) -> dict[str, Any]:
        del gate_thresholds, progress_label
        return {
            "results": {
                build_id: (
                    generation_runner.BuildStatus.evaluated,
                    [
                        ScenarioMetricRow(
                            build_id=build_id,
                            ruleset_id=_ruleset_id(),
                            scenario_id="pinnacle_boss",
                            gate_pass=True,
                            gate_fail_reasons=[],
                            pob_warnings=[],
                            evaluated_at=datetime.now(UTC),
                            full_dps=2200.0,
                            max_hit=3200.0,
                            armour=1000.0,
                            evasion=500.0,
                            life=4000.0,
                            mana=700.0,
                            utility_score=44.0,
                            metrics_source="pob",
                        )
                    ],
                )
                for build_id in build_ids
            },
            "errors": {},
        }

    evaluator_instance.evaluate_builds_batched.side_effect = _evaluate_builds_batched

    with patch.object(generation_runner, "BuildEvaluator", return_value=evaluator_instance) as ctor:
        summary = generation_runner.run_generation(
            count=1,
            seed_start=44,
            ruleset_id=_ruleset_id(),
            profile_id="pinnacle",
            base_path=tmp_path,
            repo=repo,
            evaluator=None,
            run_id="settings-evaluator-run",
            metrics_generator=_verified_metrics_generator,
        )

    assert summary["status"] == "completed"
    kwargs = ctor.call_args.kwargs
    assert kwargs["worker_cmd"] == "luajit-custom"
    assert kwargs["worker_args"] == "backend/pob_worker/pob_worker.lua --trace"
    assert kwargs["worker_cwd"] == "PathOfBuilding/src"
    assert kwargs["worker_pool_size"] == 7
    assert evaluator_instance.evaluate_builds_batched.call_count == 1
    build_ids = evaluator_instance.evaluate_builds_batched.call_args.args[0]
    assert len(build_ids) == 1
    evaluator_instance.close.assert_called_once()


def test_generation_closes_internal_evaluator_on_artifact_persistence_error(
    tmp_path: Path,
) -> None:
    repo = FakeRepository()

    evaluator_instance = Mock()

    def _evaluate_builds_batched(
        build_ids: list[str],
        gate_thresholds: Any = None,
        progress_label: str | None = None,
    ) -> dict[str, Any]:
        del gate_thresholds, progress_label
        return {
            "results": {
                build_id: (
                    generation_runner.BuildStatus.evaluated,
                    [
                        ScenarioMetricRow(
                            build_id=build_id,
                            ruleset_id=_ruleset_id(),
                            scenario_id="pinnacle_boss",
                            gate_pass=True,
                            gate_fail_reasons=[],
                            pob_warnings=[],
                            evaluated_at=datetime.now(UTC),
                            full_dps=1100.0,
                            max_hit=2100.0,
                            armour=1000.0,
                            evasion=500.0,
                            life=4000.0,
                            mana=700.0,
                            utility_score=44.0,
                            metrics_source="pob",
                        )
                        for build_id in build_ids
                    ],
                )
                for build_id in build_ids
            },
            "errors": {},
        }

    evaluator_instance.evaluate_builds_batched.side_effect = _evaluate_builds_batched

    with (
        patch.object(generation_runner, "BuildEvaluator", return_value=evaluator_instance),
        patch.object(
            generation_runner,
            "_persist_run_artifact",
            side_effect=RuntimeError("artifact write failed"),
        ),
    ):
        with pytest.raises(RuntimeError, match="artifact write failed"):
            generation_runner.run_generation(
                count=1,
                seed_start=101,
                ruleset_id=_ruleset_id(),
                profile_id="pinnacle",
                base_path=tmp_path,
                repo=repo,
                evaluator=None,
                run_id="artifact-persist-fail",
                metrics_generator=_verified_metrics_generator,
            )

    evaluator_instance.close.assert_called_once()


def test_generation_does_not_close_caller_supplied_evaluator_on_persist_error(
    tmp_path: Path,
) -> None:
    repo = FakeRepository()

    evaluator = Mock()

    def _evaluate_builds_batched(
        build_ids: list[str],
        gate_thresholds: Any = None,
        progress_label: str | None = None,
    ) -> dict[str, Any]:
        del gate_thresholds, progress_label
        return {
            "results": {
                build_id: (
                    generation_runner.BuildStatus.evaluated,
                    [
                        ScenarioMetricRow(
                            build_id=build_id,
                            ruleset_id=_ruleset_id(),
                            scenario_id="pinnacle_boss",
                            gate_pass=True,
                            gate_fail_reasons=[],
                            pob_warnings=[],
                            evaluated_at=datetime.now(UTC),
                            full_dps=1200.0,
                            max_hit=2300.0,
                            armour=1000.0,
                            evasion=500.0,
                            life=4000.0,
                            mana=700.0,
                            utility_score=44.0,
                            metrics_source="pob",
                        )
                        for build_id in build_ids
                    ],
                )
                for build_id in build_ids
            },
            "errors": {},
        }

    evaluator.evaluate_builds_batched.side_effect = _evaluate_builds_batched

    with patch.object(
        generation_runner,
        "_persist_run_artifact",
        side_effect=RuntimeError("artifact write failed"),
    ):
        with pytest.raises(RuntimeError, match="artifact write failed"):
            generation_runner.run_generation(
                count=1,
                seed_start=102,
                ruleset_id=_ruleset_id(),
                profile_id="pinnacle",
                base_path=tmp_path,
                repo=repo,
                evaluator=evaluator,
                run_id="artifact-persist-fail-caller",
                metrics_generator=_verified_metrics_generator,
            )

    evaluator.close.assert_not_called()


def test_stub_tripwire_detects_placeholder_metrics_when_gate_fails(tmp_path: Path) -> None:
    repo = FakeRepository()
    evaluator = _fake_evaluator(tmp_path, repo)
    call_count = 0
    seed_start = 90

    def _evaluate_build(
        build_id: str,
    ) -> tuple[generation_runner.BuildStatus, list[ScenarioMetricRow]]:
        nonlocal call_count
        call_count += 1
        seed = seed_start + call_count - 1
        row = ScenarioMetricRow(
            build_id=build_id,
            ruleset_id=_ruleset_id(),
            scenario_id="pinnacle_boss",
            gate_pass=False,
            gate_fail_reasons=["min_max_hit"],
            pob_warnings=[],
            evaluated_at=datetime.now(UTC),
            full_dps=float(120 * seed),
            max_hit=float(4500 + 2 * seed),
            armour=1000.0,
            evasion=500.0,
            life=4000.0,
            mana=700.0,
            utility_score=0.0,
            metrics_source="pob",
        )
        return generation_runner.BuildStatus.evaluated, [row]

    with patch.object(evaluator, "evaluate_build", side_effect=_evaluate_build):
        summary = generation_runner.run_generation(
            count=4,
            seed_start=seed_start,
            ruleset_id=_ruleset_id(),
            profile_id="pinnacle",
            base_path=tmp_path,
            repo=repo,
            evaluator=evaluator,
            run_id="stub-tripwire-gate-fail",
            metrics_generator=_verified_metrics_generator,
        )

    assert summary["status"] == "failed"
    reason = summary.get("status_reason")
    assert reason is not None
    assert reason["code"] == "stub_metrics_detected"


def test_generation_aborts_on_evaluation_infrastructure_error(tmp_path: Path) -> None:
    repo = FakeRepository()
    evaluator = _fake_evaluator(tmp_path, repo)

    error = APIError(
        500, "evaluation_error", "failed to evaluate build", details={"reason": "worker_hiccup"}
    )

    def _evaluate_builds_batched(
        build_ids: list[str],
        gate_thresholds: Any = None,
        progress_label: str | None = None,
    ) -> dict[str, Any]:
        del gate_thresholds, progress_label
        return {
            "results": {},
            "errors": {build_ids[0]: error},
        }

    mock_evaluate_builds_batched = Mock(side_effect=_evaluate_builds_batched)
    with patch.object(evaluator, "evaluate_builds_batched", mock_evaluate_builds_batched):
        summary = generation_runner.run_generation(
            count=3,
            seed_start=501,
            ruleset_id=_ruleset_id(),
            profile_id="pinnacle",
            base_path=tmp_path,
            repo=repo,
            evaluator=evaluator,
            run_id="infra-error-run",
            metrics_generator=_verified_metrics_generator,
        )

    assert mock_evaluate_builds_batched.call_count == 1
    build_ids = mock_evaluate_builds_batched.call_args.args[0]
    assert len(build_ids) == 3
    evaluation = summary["evaluation"]
    assert evaluation["attempted"] == 1
    assert evaluation["errors"] == 1
    assert evaluation["failures"] == 1
    reason = summary.get("status_reason")
    assert reason is not None
    assert reason["code"] == "evaluation_infrastructure_error"
    assert reason["evaluation"]["attempted"] == 1
    assert reason["details"]["error"]["code"] == "evaluation_error"
    assert reason["details"]["error"]["details"] == {"reason": "worker_hiccup"}


def test_generation_tripwire_stub_metrics_abort(tmp_path: Path) -> None:
    repo = FakeRepository()
    evaluator = _fake_evaluator(tmp_path, repo)

    error = APIError(
        400,
        "stub_metrics_disallowed",
        "stub metrics disallowed for profile",
        details={"reason": "worker_stub_metrics_only"},
    )

    def _evaluate_builds_batched(
        build_ids: list[str],
        gate_thresholds: Any = None,
        progress_label: str | None = None,
    ) -> dict[str, Any]:
        del gate_thresholds, progress_label
        return {
            "results": {},
            "errors": {build_ids[0]: error},
        }

    mock_evaluate_builds_batched = Mock(side_effect=_evaluate_builds_batched)
    with patch.object(evaluator, "evaluate_builds_batched", mock_evaluate_builds_batched):
        summary = generation_runner.run_generation(
            count=3,
            seed_start=701,
            ruleset_id=_ruleset_id(),
            profile_id="pinnacle",
            base_path=tmp_path,
            repo=repo,
            evaluator=evaluator,
            run_id="tripwire-stub-metrics",
            metrics_generator=_verified_metrics_generator,
            enforce_worker_tripwire=True,
        )

    assert mock_evaluate_builds_batched.call_count == 1
    build_ids = mock_evaluate_builds_batched.call_args.args[0]
    assert len(build_ids) == 3
    evaluation = summary["evaluation"]
    assert evaluation["attempted"] == 1
    assert evaluation["errors"] == 1
    assert evaluation["failures"] == 1
    assert evaluation["fallback_stub_count"] == 1
    assert evaluation["worker_error_count"] == 0
    assert evaluation["last_worker_error"] == "stub metrics detected for worker-required profile"
    reason = summary.get("status_reason")
    assert reason is not None
    assert reason["code"] == "evaluation_non_pob_metrics"
    assert reason["details"]["error"]["code"] == "stub_metrics_disallowed"


def test_generation_aborts_on_non_pob_metrics(tmp_path: Path) -> None:
    repo = FakeRepository()
    evaluator = _fake_evaluator(tmp_path, repo)

    def _evaluate_builds_batched(
        build_ids: list[str],
        gate_thresholds: Any = None,
        progress_label: str | None = None,
    ) -> dict[str, Any]:
        del gate_thresholds, progress_label
        first_build_id = build_ids[0]
        row = ScenarioMetricRow(
            build_id=first_build_id,
            ruleset_id=_ruleset_id(),
            scenario_id="pinnacle_boss",
            gate_pass=False,
            gate_fail_reasons=["non_pob_metrics"],
            pob_warnings=[],
            evaluated_at=datetime.now(UTC),
            full_dps=1200.0,
            max_hit=4520.0,
            armour=1000.0,
            evasion=500.0,
            life=4000.0,
            mana=700.0,
            utility_score=44.0,
            metrics_source="fallback",
        )
        return {
            "results": {first_build_id: (generation_runner.BuildStatus.evaluated, [row])},
            "errors": {},
        }

    mock_evaluate_builds_batched = Mock(side_effect=_evaluate_builds_batched)
    with patch.object(evaluator, "evaluate_builds_batched", mock_evaluate_builds_batched):
        summary = generation_runner.run_generation(
            count=3,
            seed_start=601,
            ruleset_id=_ruleset_id(),
            profile_id="pinnacle",
            base_path=tmp_path,
            repo=repo,
            evaluator=evaluator,
            run_id="non-pob-source-run",
            metrics_generator=_verified_metrics_generator,
        )

    assert mock_evaluate_builds_batched.call_count == 1
    build_ids = mock_evaluate_builds_batched.call_args.args[0]
    assert len(build_ids) == 3
    evaluation = summary["evaluation"]
    assert evaluation["attempted"] == 1
    assert evaluation["successes"] == 0
    assert evaluation["failures"] == 1
    assert evaluation["errors"] == 0

    reason = summary.get("status_reason")
    assert reason is not None
    assert reason["code"] == "evaluation_non_pob_metrics"
    assert reason["details"]["metrics_sources"] == ["fallback"]


def test_generation_tripwire_counts_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = FakeRepository()
    evaluator = _fake_evaluator(tmp_path, repo)
    base_time = datetime.now(UTC)

    row = ScenarioMetricRow(
        build_id="tripwire-build",
        ruleset_id=_ruleset_id(),
        scenario_id="pinnacle_boss",
        gate_pass=True,
        gate_fail_reasons=[],
        pob_warnings=[],
        evaluated_at=base_time,
        full_dps=1000.0,
        max_hit=2500.0,
        armour=900.0,
        evasion=450.0,
        life=3200.0,
        mana=600.0,
        utility_score=42.0,
        metrics_source="pob",
    )

    def fake_evaluate_builds_batched(
        build_ids: list[str],
        gate_thresholds: Any = None,
        progress_label: str | None = None,
    ) -> dict[str, Any]:
        del gate_thresholds, progress_label
        return {
            "results": {
                build_id: (generation_runner.BuildStatus.evaluated, [row]) for build_id in build_ids
            },
            "errors": {},
        }

    metadata = EvaluationProvenance(
        stub_warning_count=2,
        stub_warning_scenarios=["pinnacle_boss", "pinnacle_mob"],
        worker_metrics_used_count=1,
        worker_metadata_missing_count=1,
        worker_metadata_missing_scenarios=["pinnacle_boss"],
    )
    provenance_stack = [metadata]

    def fake_pop() -> EvaluationProvenance | None:
        return provenance_stack.pop(0) if provenance_stack else None

    monkeypatch.setattr(evaluator, "evaluate_builds_batched", fake_evaluate_builds_batched)
    monkeypatch.setattr(evaluator, "pop_last_evaluation_provenance", fake_pop)

    summary = generation_runner.run_generation(
        count=1,
        seed_start=701,
        ruleset_id=_ruleset_id(),
        profile_id="pinnacle",
        base_path=tmp_path,
        repo=repo,
        evaluator=evaluator,
        run_id="tripwire-counts",
        metrics_generator=_verified_metrics_generator,
        enforce_worker_tripwire=True,
    )

    evaluation = summary["evaluation"]
    assert evaluation["worker_metrics_used_count"] == 0
    assert evaluation["fallback_stub_count"] == 0
    assert evaluation["worker_error_count"] == 0
    assert evaluation["last_worker_error"] is None
    assert summary["status"] == "completed"


def test_uber_optimizer_requires_non_stub_metrics(tmp_path: Path) -> None:
    repo = FakeRepository()
    evaluator = _fake_evaluator(tmp_path, repo)

    def _empty_worker_metrics_batched(
        prepared_builds: Mapping[str, Any],
        grouped_build_ids: Mapping[tuple[str, str], list[str]],
        progress_label: str | None = None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, APIError]]:
        del grouped_build_ids, progress_label
        return ({build_id: {} for build_id in prepared_builds}, {})

    with patch.object(
        BuildEvaluator,
        "_collect_worker_metrics_batched",
        side_effect=_empty_worker_metrics_batched,
    ):
        summary = generation_runner.run_generation(
            count=1,
            seed_start=101,
            ruleset_id=_ruleset_id(),
            profile_id="uber_pinnacle",
            base_path=tmp_path,
            repo=repo,
            evaluator=evaluator,
            run_id="uber-optimizer-stub",
            run_mode="optimizer",
            metrics_generator=generation_runner._default_metrics_generator,
        )

    assert summary["status"] == "failed"
    evaluation = summary["evaluation"]
    assert evaluation["successes"] == 0
    assert evaluation["errors"] == evaluation["attempted"]
    assert evaluation["attempted"] >= 1
    records = evaluation["records"]
    assert records
    stub_errors = [record["error"] for record in records if record.get("error")]
    assert stub_errors
    stub_error = stub_errors[0]
    assert stub_error["code"] == "stub_metrics_disallowed"
    assert "stub metrics" in stub_error["message"].lower()
    assert stub_error["details"]
    assert stub_error["details"].get("profile_id") == "uber_pinnacle"


def test_optimizer_non_uber_profile_requires_worker_metrics(tmp_path: Path) -> None:
    repo = FakeRepository()
    evaluator = _fake_evaluator(tmp_path, repo)

    def _empty_worker_metrics_batched(
        prepared_builds: Mapping[str, Any],
        grouped_build_ids: Mapping[tuple[str, str], list[str]],
        progress_label: str | None = None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, APIError]]:
        del grouped_build_ids, progress_label
        return ({build_id: {} for build_id in prepared_builds}, {})

    with patch.object(
        BuildEvaluator,
        "_collect_worker_metrics_batched",
        side_effect=_empty_worker_metrics_batched,
    ):
        summary = generation_runner.run_generation(
            count=1,
            seed_start=202,
            ruleset_id=_ruleset_id(),
            profile_id="pinnacle",
            base_path=tmp_path,
            repo=repo,
            evaluator=evaluator,
            run_id="optimizer-pinnacle",
            run_mode="optimizer",
            metrics_generator=generation_runner._default_metrics_generator,
        )

    assert summary["status"] == "failed"
    evaluation = summary["evaluation"]
    assert evaluation["successes"] == 0
    assert evaluation["errors"] == evaluation["attempted"]
    records = evaluation["records"]
    assert records
    errors = [record["error"] for record in records if record.get("error")]
    assert errors
    missing_error = errors[0]
    assert missing_error["code"] == "missing_metrics"
    assert missing_error["details"]["reason"] == "worker_metrics_missing"


def test_uber_optimizer_rejects_worker_stub_metrics(tmp_path: Path) -> None:
    repo = FakeRepository()
    evaluator = _fake_evaluator(tmp_path, repo)

    def _worker_stub_metrics_batched(
        prepared_builds: Mapping[str, Any],
        grouped_build_ids: Mapping[tuple[str, str], list[str]],
        progress_label: str | None = None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, APIError]]:
        del grouped_build_ids, progress_label
        return (
            {
                build_id: {
                    "pinnacle_boss": {
                        "warnings": ["generation_stub_metrics"],
                        "metrics": {"full_dps": 123.0},
                    }
                }
                for build_id in prepared_builds
            },
            {},
        )

    with patch.object(
        BuildEvaluator,
        "_collect_worker_metrics_batched",
        side_effect=_worker_stub_metrics_batched,
    ):
        summary = generation_runner.run_generation(
            count=1,
            seed_start=203,
            ruleset_id=_ruleset_id(),
            profile_id="uber_pinnacle",
            base_path=tmp_path,
            repo=repo,
            evaluator=evaluator,
            run_id="optimizer-uber-worker-stub",
            run_mode="optimizer",
            metrics_generator=generation_runner._default_metrics_generator,
        )

    assert summary["status"] == "failed"
    evaluation = summary["evaluation"]
    assert evaluation["successes"] == 0
    records = evaluation["records"]
    assert records
    first_error = records[0]["error"]
    assert first_error["code"] == "stub_metrics_disallowed"
    assert first_error["details"]["reason"] == "worker_stub_metrics_only"


def test_generation_records_capture_failure_categories(tmp_path: Path) -> None:
    repo = FakeRepository()
    evaluator = _fake_evaluator(tmp_path, repo)

    def socket_with_issue(catalog, genome):  # type: ignore
        plan = plan_sockets(catalog, genome)
        issue = PlanIssue(code="forced", message="forced issue")
        return SocketPlan(
            genome_seed=plan.genome_seed,
            main_group_id=plan.main_group_id,
            main_slot_id=plan.main_slot_id,
            main_link_requirement=plan.main_link_requirement,
            assignments=plan.assignments,
            slots=plan.slots,
            issues=(issue,),
            hints=plan.hints,
        )

    def template_with_deficits(genome, gem_plan, passive_plan, socket_plan):  # type: ignore
        plan = build_item_templates(genome, gem_plan, passive_plan, socket_plan)
        deficits = plan.repair_report.remaining_deficits
        forced = RequirementDeficits(
            fire=deficits.fire,
            cold=deficits.cold,
            lightning=deficits.lightning,
            chaos=deficits.chaos,
            strength=max(1, deficits.strength),
            dexterity=max(1, deficits.dexterity),
            intelligence=max(1, deficits.intelligence),
            life=deficits.life,
            energy_shield=deficits.energy_shield,
        )
        report = RepairReport(
            iterations=plan.repair_report.iterations,
            initial_deficits=plan.repair_report.initial_deficits,
            remaining_deficits=forced,
        )
        return ItemTemplatePlan(genome=plan.genome, templates=plan.templates, repair_report=report)

    def metrics_with_reservation(seed: int, templates: list[Any]) -> dict[str, Any]:
        payload = _verified_metrics_generator(seed, templates)  # type: ignore[attr-defined]
        template = templates[0]
        reservation = template.gate_thresholds.reservation.max_percent + 10
        scenario_id = template.scenario_id
        payload.setdefault(scenario_id, {})
        payload[scenario_id]["reservation"]["reserved_percent"] = reservation
        payload[scenario_id]["reservation"]["available_percent"] = reservation + 5
        return payload

    summary = generation_runner.run_generation(
        count=1,
        seed_start=32,
        ruleset_id=_ruleset_id(),
        profile_id="pinnacle",
        base_path=tmp_path,
        repo=repo,
        evaluator=evaluator,
        socket_planner=socket_with_issue,
        template_builder=template_with_deficits,
        metrics_generator=metrics_with_reservation,
        run_id="fail-run",
    )

    records = summary["generation"]["attempt_records"]
    assert records
    record = records[0]
    assert "socket_precheck_failed" in record["precheck_failures"]
    assert "attribute_precheck_failed" in record["precheck_failures"]
    assert "reservation_precheck_failed" in record["precheck_failures"]
    assert summary["generation"]["precheck_failures"]["socket_precheck_failed"] >= 1
    assert summary["generation"]["precheck_failures"]["attribute_precheck_failed"] >= 1
    assert summary["generation"]["precheck_failures"]["reservation_precheck_failed"] >= 1


def test_constraints_metadata_recorded_and_persisted(tmp_path: Path) -> None:
    repo = FakeRepository()
    evaluator = _fake_evaluator(tmp_path, repo)

    def worker_metrics_stub(
        *args: Any,
        templates: list[Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        template_list = templates
        if template_list is None and len(args) >= 4:
            template_list = args[3]
        return _verified_metrics_generator(0, template_list or [])  # type: ignore[attr-defined]

    with patch.object(
        BuildEvaluator,
        "_collect_worker_metrics",
        side_effect=worker_metrics_stub,
    ):
        summary = generation_runner.run_generation(
            count=2,
            seed_start=10,
            ruleset_id=_ruleset_id(),
            profile_id="pinnacle",
            base_path=tmp_path,
            repo=repo,
            evaluator=evaluator,
            constraints=_budget_constraint_spec(),
            metrics_generator=_metrics_with_cost,
            run_id="constraint-run",
        )

    records = summary["generation"]["records"]
    assert len(records) == 2
    assert any(record["constraint_status"] == "pass" for record in records)
    assert any(record["constraint_status"] == "fail" for record in records)
    failing = next(record for record in records if record["constraint_status"] == "fail")
    assert "budget_cost" in failing["violated_constraints"]
    assert failing["constraint_reason_code"] == "budget_within_limit"
    assert records[0]["constraint_reason_code"] == "budget_within_limit"

    for record in records:
        build_dir = tmp_path / "data" / "builds" / record["build_id"]
        constraints_path = build_dir / "constraints.json"
        assert constraints_path.exists()
        payload = json.loads(constraints_path.read_text())
        assert payload["evaluation"]["status"] == record["constraint_status"]
        assert payload["evaluation"]["checked_at"] == record["constraint_checked_at"]


def test_constraints_missing_data_is_unknown(tmp_path: Path) -> None:
    repo = FakeRepository()
    evaluator = _fake_evaluator(tmp_path, repo)

    summary = generation_runner.run_generation(
        count=2,
        seed_start=20,
        ruleset_id=_ruleset_id(),
        profile_id="pinnacle",
        base_path=tmp_path,
        repo=repo,
        evaluator=evaluator,
        constraints=_budget_constraint_spec(),
        metrics_generator=_verified_metrics_generator,
        run_id="constraint-missing",
    )

    records = summary["generation"]["records"]
    assert all(record.get("constraint_status") == "unknown" for record in records)
    assert all(record.get("constraint_reason_code") == "budget_data_missing" for record in records)
    for record in records:
        assert record.get("violated_constraints") == []
        constraints_path = tmp_path / "data" / "builds" / record["build_id"] / "constraints.json"
        assert constraints_path.exists()
        payload = json.loads(constraints_path.read_text())
        assert payload["evaluation"]["status"] == "unknown"
        assert payload["evaluation"]["reason_code"] == "budget_data_missing"


def test_surrogate_predictor_prunes_candidates(tmp_path: Path) -> None:
    repo = FakeRepository()
    evaluator = _fake_evaluator(tmp_path, repo)

    def predictor(rows: list[dict[str, Any]]) -> list[float]:
        return [float(row["seed"]) for row in rows]

    summary = generation_runner.run_generation(
        count=4,
        seed_start=5,
        ruleset_id=_ruleset_id(),
        profile_id="pinnacle",
        base_path=tmp_path,
        repo=repo,
        evaluator=evaluator,
        surrogate_enabled=True,
        surrogate_top_k=2,
        surrogate_exploration_pct=0.5,
        surrogate_predictor=predictor,
        metrics_generator=_verified_metrics_generator,
        run_id="surrogate-active",
    )

    surrogate = summary["surrogate"]
    assert surrogate["status"] in {"active", "fallback"}
    if surrogate["status"] == "fallback":
        assert summary["surrogate"].get("fallback_reason") == "degenerate_predictions"
    assert surrogate["counts"]["selected"] == summary["evaluation"]["attempted"]
    assert surrogate["counts"]["pruned"] == 1
    assert surrogate["selection_params"]["top_k"] == 2
    assert surrogate["selection_params"]["exploration_pct"] == 0.5
    pruned_records = [
        record
        for record in summary["generation"]["attempt_records"]
        if record.get("surrogate_selection_reason") == "surrogate_pruned"
    ]
    assert len(pruned_records) == surrogate["counts"]["pruned"]
    assert (
        len(
            [
                record
                for record in summary["generation"]["attempt_records"]
                if record.get("surrogate_selection_reason") == "surrogate_exploration"
            ]
        )
        == 1
    )
    assert (
        len(
            [
                record
                for record in summary["generation"]["attempt_records"]
                if record.get("surrogate_selection_reason") == "surrogate_top"
            ]
        )
        == 2
    )
    predictions_path = Path(summary["paths"]["surrogate_predictions"])
    assert predictions_path.exists()
    payload = json.loads(predictions_path.read_text())
    assert payload["status"] == "active"
    assert payload["counts"]["selected"] == surrogate["counts"]["selected"]
    assert payload["counts"]["pruned"] == surrogate["counts"]["pruned"]
    assert len(payload["candidates"]) == surrogate["counts"]["candidates"]

    builds_dir = tmp_path / "data" / "builds"
    for entry in builds_dir.iterdir():
        if not entry.is_dir():
            continue
        prediction_file = entry / "surrogate_prediction.json"
        assert prediction_file.exists()
        record = json.loads(prediction_file.read_text())
        assert record["schema_version"] == generation_runner.SURROGATE_PREDICTION_SCHEMA_VERSION
        assert record.get("model_id") == surrogate["model_id"]
        assert "predicted_metrics" in record
        assert "selection_reason" in record


def test_surrogate_model_path_predictions(tmp_path: Path) -> None:
    surrogate_model_path = tmp_path / "tiny-surrogate.json"
    surrogate_model_path.write_text(
        json.dumps(
            {
                "model_id": "tiny",
                "dataset_snapshot_id": "snapshot",
                "feature_schema_version": "1.0",
                "global_metrics": {
                    "full_dps": {"mean": 100.0, "std": 1.0, "min": 50.0, "max": 150.0, "count": 1},
                    "max_hit": {"mean": 200.0, "std": 1.0, "min": 100.0, "max": 250.0, "count": 1},
                    "utility_score": {"mean": 1.0, "std": 0.5, "min": 0.0, "max": 2.0, "count": 1},
                },
                "main_skill_metrics": {
                    "sunder": {"full_dps": 150.0, "max_hit": 250.0, "utility_score": 2.0}
                },
                "feature_stats": {},
                "feature_weights": {},
                "identity_token_effects": {},
                "identity_cross_token_effects": {},
                "pass_metric": "full_dps",
                "backend": "ep-v4-baseline",
                "backend_version": "0.2.0",
                "compute_backend": "cpu",
                "token_learner_backend": "torch_sparse_sgd",
                "trained_at_utc": "2024-01-01T00:00:00Z",
                "target_transforms": {
                    "full_dps": "log1p",
                    "max_hit": "log1p",
                    "utility_score": "identity",
                },
                "classifier": None,
            }
        )
    )

    repo = FakeRepository()
    evaluator = _fake_evaluator(tmp_path, repo)
    summary = generation_runner.run_generation(
        count=3,
        seed_start=13,
        ruleset_id=_ruleset_id(),
        profile_id="pinnacle",
        base_path=tmp_path,
        repo=repo,
        evaluator=evaluator,
        surrogate_enabled=True,
        surrogate_model_path=surrogate_model_path,
        surrogate_top_k=2,
        surrogate_exploration_pct=0.0,
        metrics_generator=_verified_metrics_generator,
        run_id="surrogate-model-path",
    )

    surrogate = summary["surrogate"]
    assert surrogate["status"] in {"active", "fallback"}
    if surrogate["status"] == "fallback":
        assert summary["surrogate"].get("fallback_reason") == "degenerate_predictions"
    predictions_path = Path(summary["paths"]["surrogate_predictions"])
    assert predictions_path.exists()
    payload = json.loads(predictions_path.read_text())
    assert payload["status"] == surrogate["status"]
    candidates = payload.get("candidates", [])
    assert candidates
    full_dps_values = {candidate["predicted_metrics"]["full_dps"] for candidate in candidates}
    assert any(value != 0.0 for value in full_dps_values)


def test_surrogate_degenerate_predictions_warn(tmp_path: Path, caplog) -> None:
    repo = FakeRepository()
    evaluator = _fake_evaluator(tmp_path, repo)

    def constant_predictor(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"metrics": {"full_dps": 5.0}, "pass_probability": None} for _ in rows]

    run_id = "degenerate-warning-run"
    caplog.set_level(logging.WARNING)
    with caplog.at_level(logging.WARNING, logger=generation_runner.__name__):
        generation_runner.run_generation(
            count=3,
            seed_start=29,
            ruleset_id=_ruleset_id(),
            profile_id="pinnacle",
            base_path=tmp_path,
            repo=repo,
            evaluator=evaluator,
            surrogate_enabled=True,
            surrogate_predictor=constant_predictor,
            surrogate_top_k=2,
            surrogate_exploration_pct=0.0,
            metrics_generator=_verified_metrics_generator,
            run_id=run_id,
        )

    warning_messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == generation_runner.__name__
    ]
    assert any(
        run_id in message and "surrogate predictions constant" in message
        for message in warning_messages
    )
    assert any("full_dps" in message for message in warning_messages)
    assert any("candidates" in message for message in warning_messages)


def test_surrogate_degenerate_predictions_fallback(tmp_path: Path) -> None:
    repo = FakeRepository()
    evaluator = _fake_evaluator(tmp_path, repo)

    def constant_predictor(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"metrics": {"full_dps": 5.0}, "pass_probability": None} for _ in rows]

    summary = generation_runner.run_generation(
        count=3,
        seed_start=40,
        ruleset_id=_ruleset_id(),
        profile_id="pinnacle",
        base_path=tmp_path,
        repo=repo,
        evaluator=evaluator,
        surrogate_enabled=True,
        surrogate_predictor=constant_predictor,
        surrogate_top_k=2,
        surrogate_exploration_pct=0.0,
        metrics_generator=_verified_metrics_generator,
        run_id="degenerate-fallback",
    )

    surrogate = summary["surrogate"]
    assert surrogate["status"] == "fallback"
    assert surrogate["fallback_reason"] == "degenerate_predictions"
    assert surrogate["counts"]["pruned"] == 1
    assert surrogate["counts"]["selected"] == 2
    assert summary["evaluation"]["attempted"] == 2
    assert surrogate["counts"]["selected"] == summary["evaluation"]["attempted"]


def test_surrogate_degenerate_predictions_limits_top_k(tmp_path: Path) -> None:
    repo = FakeRepository()
    evaluator = _fake_evaluator(tmp_path, repo)

    def constant_predictor(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"metrics": {"full_dps": 5.0}, "pass_probability": None} for _ in rows]

    summary = generation_runner.run_generation(
        count=10,
        seed_start=50,
        ruleset_id=_ruleset_id(),
        profile_id="pinnacle",
        base_path=tmp_path,
        repo=repo,
        evaluator=evaluator,
        surrogate_enabled=True,
        surrogate_predictor=constant_predictor,
        surrogate_top_k=3,
        surrogate_exploration_pct=0.0,
        metrics_generator=_verified_metrics_generator,
        run_id="degenerate-fallback-top-k",
    )

    surrogate = summary["surrogate"]
    assert surrogate["status"] == "fallback"
    assert surrogate["fallback_reason"] == "degenerate_predictions"
    assert surrogate["counts"]["selected"] == 3
    assert surrogate["counts"]["pruned"] == 7
    assert summary["evaluation"]["attempted"] == 3
    assert surrogate["counts"]["selected"] == summary["evaluation"]["attempted"]


def test_surrogate_missing_model_fallback(tmp_path: Path) -> None:
    repo = FakeRepository()
    evaluator = _fake_evaluator(tmp_path, repo)
    missing_model = tmp_path / "missing-model.json"

    summary = generation_runner.run_generation(
        count=3,
        seed_start=10,
        ruleset_id=_ruleset_id(),
        profile_id="pinnacle",
        base_path=tmp_path,
        repo=repo,
        evaluator=evaluator,
        surrogate_enabled=True,
        surrogate_model_path=missing_model,
        metrics_generator=_verified_metrics_generator,
        run_id="surrogate-fallback",
    )

    surrogate = summary["surrogate"]
    assert surrogate["status"] == "fallback"
    assert surrogate["counts"]["selected"] == summary["evaluation"]["attempted"]
    assert surrogate["counts"]["pruned"] == 0
    assert surrogate["fallback_reason"]
    predictions_path = Path(summary["paths"]["surrogate_predictions"])
    assert predictions_path.exists()
    payload = json.loads(predictions_path.read_text())
    assert payload["status"] == "fallback"
    assert payload["fallback_reason"] == surrogate["fallback_reason"]
    assert payload["candidates"] == []

    ml = summary["ml_lifecycle"]
    assert ml["enabled"]
    metadata = ml["metadata"]
    assert metadata["model_meta"] is None
    assert metadata["meta_path"]
    assert "model_meta.json" in metadata["error"]
    assert Path(summary["paths"]["ml_lifecycle"]).exists()
    total_prechecks = sum(summary["generation"]["precheck_failures"].values())
    assert summary["evaluation"]["attempted"] == (
        summary["generation"]["processed"] - total_prechecks
    )


def test_select_diverse_top_candidates_prioritizes_unique_niches() -> None:
    c1 = _build_candidate_for_selection(
        build_id="niche-1",
        seed=1,
        class_name="Marauder",
        main_skill="sunder",
        ascendancy="Chieftain",
        identity={"class": "Marauder", "main_skill": "sunder"},
    )
    c2 = _build_candidate_for_selection(
        build_id="niche-1b",
        seed=2,
        class_name="Marauder",
        main_skill="sunder",
        ascendancy="Juggernaut",
        identity={"class": "Marauder", "main_skill": "sunder"},
    )
    c3 = _build_candidate_for_selection(
        build_id="niche-2",
        seed=3,
        class_name="Ranger",
        main_skill="tornado_shot",
        ascendancy="Deadeye",
        identity={"class": "Ranger", "main_skill": "tornado_shot"},
    )
    c4 = _build_candidate_for_selection(
        build_id="niche-3",
        seed=4,
        class_name="Witch",
        main_skill="essence_drain",
        ascendancy="Elementalist",
        identity={"class": "Witch", "main_skill": "essence_drain"},
    )

    selected = generation_runner._select_diverse_top_candidates([c1, c2, c3, c4], top_k=3)
    assert [candidate.build_id for candidate in selected] == ["niche-1", "niche-2", "niche-3"]


def test_select_diverse_top_candidates_fills_duplicates_when_unique_niches_are_exhausted() -> None:
    c1 = _build_candidate_for_selection(
        build_id="niche-1",
        seed=1,
        class_name="Marauder",
        main_skill="sunder",
        ascendancy="Chieftain",
        identity={"class": "Marauder", "main_skill": "sunder"},
    )
    c2 = _build_candidate_for_selection(
        build_id="niche-1b",
        seed=2,
        class_name="Marauder",
        main_skill="sunder",
        ascendancy="Juggernaut",
        identity={"class": "Marauder", "main_skill": "sunder"},
    )
    c3 = _build_candidate_for_selection(
        build_id="niche-2",
        seed=3,
        class_name="Ranger",
        main_skill="tornado_shot",
        ascendancy="Deadeye",
        identity={"class": "Ranger", "main_skill": "tornado_shot"},
    )
    c4 = _build_candidate_for_selection(
        build_id="niche-3",
        seed=4,
        class_name="Witch",
        main_skill="essence_drain",
        ascendancy="Elementalist",
        identity={"class": "Witch", "main_skill": "essence_drain"},
    )

    selected = generation_runner._select_diverse_top_candidates([c1, c2, c3, c4], top_k=4)
    assert [candidate.build_id for candidate in selected] == [
        "niche-1",
        "niche-2",
        "niche-3",
        "niche-1b",
    ]


def test_select_diverse_top_candidates_uses_candidate_fields_when_identity_missing() -> None:
    c1 = _build_candidate_for_selection(
        build_id="missing-1",
        seed=5,
        class_name="Ranger",
        main_skill="tornado_shot",
        ascendancy="Deadeye",
        identity=None,
    )
    c2 = _build_candidate_for_selection(
        build_id="missing-2",
        seed=6,
        class_name="Marauder",
        main_skill="sunder",
        ascendancy="Juggernaut",
        identity=None,
    )

    selected = generation_runner._select_diverse_top_candidates([c1, c2], top_k=2)
    assert [candidate.build_id for candidate in selected] == ["missing-1", "missing-2"]
    niches = {
        (candidate.class_name.lower(), candidate.main_skill_package.lower())
        for candidate in selected
    }
    assert niches == {("ranger", "tornado_shot"), ("marauder", "sunder")}


def test_optimizer_mode_tracks_parents(tmp_path: Path) -> None:
    repo = FakeRepository()
    evaluator = _fake_evaluator(tmp_path, repo)

    summary = generation_runner.run_generation(
        count=2,
        seed_start=42,
        ruleset_id=_ruleset_id(),
        profile_id="pinnacle",
        base_path=tmp_path,
        repo=repo,
        evaluator=evaluator,
        run_mode="optimizer",
        optimizer_iterations=1,
        optimizer_elite_count=1,
        metrics_generator=_verified_metrics_generator,
        run_id="optimizer-run",
    )

    optimizer = summary["optimizer"]
    assert optimizer["enabled"]
    assert optimizer["mode"] == "optimizer"
    assert optimizer["status"] == "completed"
    stages = {stage["stage"] for stage in optimizer["stage_counts"]}
    assert "warmup" in stages
    assert any(stage["stage"].startswith("iteration_") for stage in optimizer["stage_counts"])
    selection = optimizer["selection_history"]
    assert selection
    assert selection[0]["stage"] == "warmup"
    iteration_entries = [entry for entry in selection if entry["stage"].startswith("iteration_")]
    assert iteration_entries
    assert iteration_entries[0]["elites"]
    records = summary["generation"]["records"]
    iteration_records = [record for record in records if record.get("optimizer_iteration", 0) > 0]
    assert iteration_records
    assert iteration_records[0]["parent_build_id"] is not None
    assert iteration_records[0]["parent_seed"] is not None


def test_optimizer_selection_history_reports_stub_fallback(tmp_path: Path) -> None:
    repo = FakeRepository()
    evaluator = _fake_evaluator(tmp_path, repo)

    summary = generation_runner.run_generation(
        count=1,
        seed_start=200,
        ruleset_id=_ruleset_id(),
        profile_id="pinnacle",
        base_path=tmp_path,
        repo=repo,
        evaluator=evaluator,
        run_mode="optimizer",
        optimizer_iterations=1,
        optimizer_elite_count=1,
        run_id="optimizer-stub-history",
    )

    selection = summary["optimizer"]["selection_history"]
    assert selection
    for stage in selection:
        assert stage["elites"]
        objectives = stage["elites"][0]["objectives"]
        assert objectives["selection_basis"] == "stub_fallback"
        assert objectives["full_dps"] is None
        assert objectives["max_hit"] is None
        assert objectives["cost"] is None


def test_load_run_summary_rejects_path_traversal(tmp_path: Path) -> None:
    try:
        generation_runner.load_run_summary("../evil", base_path=tmp_path)
    except ValueError as exc:
        assert "run_id" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("expected ValueError for invalid run_id")


def test_archive_contains_only_verified_candidates(tmp_path: Path) -> None:
    repo = FakeRepository()
    evaluator = _fake_evaluator(tmp_path, repo)

    evaluation_call_index = 0

    def _evaluate_builds_batched(
        build_ids: list[str],
        gate_thresholds: Any = None,
        progress_label: str | None = None,
    ) -> dict[str, Any]:
        nonlocal evaluation_call_index
        del gate_thresholds, progress_label
        results: dict[str, tuple[generation_runner.BuildStatus, list[ScenarioMetricRow]]] = {}
        for build_id in build_ids:
            gate_pass = evaluation_call_index > 0
            evaluation_call_index += 1
            row = ScenarioMetricRow(
                build_id=build_id,
                ruleset_id=_ruleset_id(),
                scenario_id="pinnacle_boss",
                gate_pass=gate_pass,
                gate_fail_reasons=[] if gate_pass else ["min_max_hit"],
                pob_warnings=[],
                evaluated_at=datetime.now(UTC),
                full_dps=2000.0,
                max_hit=3200.0,
                armour=1000.0,
                evasion=500.0,
                life=4000.0,
                mana=700.0,
                utility_score=44.0,
                metrics_source="pob",
            )
            results[build_id] = (generation_runner.BuildStatus.evaluated, [row])
        return {
            "results": results,
            "errors": {},
        }

    mock_evaluate_builds_batched = Mock(side_effect=_evaluate_builds_batched)
    with patch.object(evaluator, "evaluate_builds_batched", mock_evaluate_builds_batched):
        summary = generation_runner.run_generation(
            count=2,
            seed_start=10,
            ruleset_id=_ruleset_id(),
            profile_id="pinnacle",
            base_path=tmp_path,
            repo=repo,
            evaluator=evaluator,
            metrics_generator=_verified_metrics_generator,
            run_id="archive-verified-only",
        )

    assert mock_evaluate_builds_batched.call_count == 1
    build_ids = mock_evaluate_builds_batched.call_args.args[0]
    assert len(build_ids) == 2

    archive_payload = load_archive_artifact(
        "archive-verified-only",
        base_path=tmp_path,
    )
    archived_build_ids = {
        entry["build_id"]
        for entry in archive_payload.get("bins", [])
        if isinstance(entry, dict) and "build_id" in entry
    }
    assert archived_build_ids

    attempt_records = summary["generation"]["attempt_records"]
    assert attempt_records
    gate_failed_attempts = [
        record for record in attempt_records if record.get("gate_pass") is False
    ]
    assert gate_failed_attempts
    assert all(record.get("evaluation_status") == "evaluated" for record in gate_failed_attempts)
    assert all(record.get("persisted") is True for record in gate_failed_attempts)

    records = summary["generation"]["records"]
    assert all(record.get("evaluation_status") == "evaluated" for record in records)
    assert all(record.get("gate_pass") is True for record in records)
    assert all(record.get("persisted") is True for record in records)

    gate_failed_ids = {record["build_id"] for record in gate_failed_attempts}
    assert archived_build_ids.isdisjoint(gate_failed_ids)


def test_generated_build_persists_build_details(tmp_path: Path) -> None:
    repo = FakeRepository()
    evaluator = _fake_evaluator(tmp_path, repo)

    summary = generation_runner.run_generation(
        count=1,
        seed_start=21,
        ruleset_id=_ruleset_id(),
        profile_id="mapping",
        base_path=tmp_path,
        repo=repo,
        evaluator=evaluator,
        metrics_generator=_verified_metrics_generator,
        run_id="build-details-generated",
    )

    records = summary["generation"]["records"]
    if records:
        build_id = records[0]["build_id"]
    else:
        attempt_records = summary["generation"]["attempt_records"]
        assert attempt_records
        build_id = attempt_records[0]["build_id"]

    artifacts = read_build_artifacts(build_id, base_path=tmp_path)
    details = artifacts.build_details
    assert isinstance(details, dict)
    assert details.get("source") == "generator_plan"

    identity = details.get("identity")
    assert isinstance(identity, dict)
    assert identity.get("class") not in (None, "", "unknown")
    assert identity.get("ascendancy") not in (None, "", "unknown")
    assert identity.get("main_skill") not in (None, "", "unknown")

    items = details.get("items")
    assert isinstance(items, dict)
    assert isinstance(items.get("slot_templates"), list)
    assert items["slot_templates"]

    passives = details.get("passives")
    assert isinstance(passives, dict)
    assert isinstance(passives.get("node_ids"), list)
    assert passives["node_ids"]

    gems = details.get("gems")
    assert isinstance(gems, dict)
    assert isinstance(gems.get("groups"), list)
    assert gems["groups"]

    resistances = details.get("resistances")
    assert isinstance(resistances, dict)
    assert set(resistances) == {"fire", "cold", "lightning", "chaos"}

    attributes = details.get("attributes")
    assert isinstance(attributes, dict)
    assert set(attributes) == {"strength", "dexterity", "intelligence"}

    stats = details.get("stats")
    assert isinstance(stats, dict)
    assert {"life", "energy_shield", "ehp"}.issubset(stats)
    assert isinstance(stats["life"], (int, float))
    assert isinstance(stats["energy_shield"], (int, float))
    assert isinstance(stats["ehp"], (int, float))
    assert stats["life"] >= 0
    assert stats["energy_shield"] >= 0
    assert stats["ehp"] >= 0

    reservation = details.get("reservation")
    total_mana = details.get("total_mana")
    assert isinstance(reservation, int)
    assert isinstance(total_mana, int)
    assert total_mana >= 100


def test_generation_persists_repaired_build_details(tmp_path: Path) -> None:
    repo = FakeRepository()
    evaluator = _fake_evaluator(tmp_path, repo)

    original_builder = generation_runner.build_details_from_generation
    mutated_metrics: dict[str, int] = {}

    def _repairable_build_details(*args: Any, **kwargs: Any) -> dict[str, Any]:
        payload = original_builder(*args, **kwargs)
        assert isinstance(payload.get("items"), dict)

        resistances = payload.get("resistances")
        assert isinstance(resistances, dict)
        resistances["fire"] = 10
        resistances["cold"] = 0
        resistances["lightning"] = 0
        resistances["chaos"] = 0

        attributes = payload.get("attributes")
        assert isinstance(attributes, dict)
        attributes["strength"] = 0
        attributes["dexterity"] = 0
        attributes["intelligence"] = 0

        stats = payload.get("stats")
        assert isinstance(stats, dict)
        stats["life"] = 100
        stats["energy_shield"] = 0
        stats["ehp"] = 100

        payload["reservation"] = 1200
        payload["total_mana"] = 200

        items = payload["items"]
        assert isinstance(items, dict)
        slot_templates = items.get("slot_templates")
        assert isinstance(slot_templates, list) and slot_templates
        for index, template in enumerate(slot_templates):
            template["adjustable"] = True
            template["contributions"] = template.get("contributions") or {}
            if index == 0:
                template["requirements"] = {
                    "Strength": 200,
                    "Dexterity": 200,
                    "Intelligence": 200,
                }

        gems = payload.get("gems")
        assert isinstance(gems, dict)
        groups = gems.get("groups")
        assert isinstance(groups, list) and len(groups) >= 2

        mutated_metrics["pre_repair_reservation_groups"] = len(groups)
        return payload

    with patch.object(
        generation_runner,
        "build_details_from_generation",
        _repairable_build_details,
    ):
        summary = generation_runner.run_generation(
            count=1,
            seed_start=31,
            ruleset_id=_ruleset_id(),
            profile_id="mapping",
            base_path=tmp_path,
            repo=repo,
            evaluator=evaluator,
            metrics_generator=_verified_metrics_generator,
            run_id="repair-payload-run",
        )

    records = summary["generation"]["records"]
    if records:
        build_id = records[0]["build_id"]
    else:
        attempt_records = summary["generation"]["attempt_records"]
        assert attempt_records
        build_id = attempt_records[0]["build_id"]
    artifacts = read_build_artifacts(build_id, base_path=tmp_path)
    details = artifacts.build_details
    assert isinstance(details, dict)

    resistances = details.get("resistances")
    assert isinstance(resistances, dict)
    assert resistances["fire"] >= 60
    assert resistances["cold"] >= 60
    assert resistances["lightning"] >= 60
    assert resistances["chaos"] >= 45

    attributes = details.get("attributes")
    assert isinstance(attributes, dict)
    assert attributes["strength"] >= 200
    assert attributes["dexterity"] >= 200
    assert attributes["intelligence"] >= 200

    stats = details.get("stats")
    assert isinstance(stats, dict)
    assert stats["ehp"] > 100

    reservation = details.get("reservation")
    total_mana = details.get("total_mana")
    assert isinstance(reservation, int)
    assert isinstance(total_mana, int)
    assert reservation <= total_mana

    gems = details.get("gems")
    assert isinstance(gems, dict)
    groups = gems.get("groups")
    assert isinstance(groups, list)
    assert len(groups) < int(mutated_metrics["pre_repair_reservation_groups"])


def test_generate_runs_cli_constraints(tmp_path: Path) -> None:
    constraints_file = tmp_path / "constraints.json"
    constraints_payload = {
        "schema_version": 1,
        "rules": [],
    }
    constraints_file.write_text(json.dumps(constraints_payload))
    summary = {"run_id": "cli-run", "paths": {"summary": str(tmp_path / "summary.json")}}
    with patch("backend.tools.generate_runs.run_generation", return_value=summary) as mock_run:
        result = generate_runs_cli.main(
            [
                "--ruleset-id",
                _ruleset_id(),
                "--constraints-file",
                str(constraints_file),
                "--data-path",
                str(tmp_path),
                "--count",
                "1",
            ]
        )
    assert result == 0
    assert mock_run.call_args.kwargs["constraints"] == constraints_payload


def test_generate_runs_cli_constraints_missing_file(tmp_path: Path, caplog) -> None:
    constraints_file = tmp_path / "missing.json"
    summary = {"run_id": "cli-run", "paths": {"summary": str(tmp_path / "summary.json")}}
    caplog.set_level(logging.ERROR)
    with patch("backend.tools.generate_runs.run_generation", return_value=summary) as mock_run:
        result = generate_runs_cli.main(
            [
                "--ruleset-id",
                _ruleset_id(),
                "--constraints-file",
                str(constraints_file),
                "--data-path",
                str(tmp_path),
                "--count",
                "1",
            ]
        )
    assert result == 1
    assert not mock_run.called
    assert str(constraints_file) in caplog.text
    assert "unable to read constraints file" in caplog.text


def test_generate_runs_cli_constraints_malformed_json(tmp_path: Path, caplog) -> None:
    constraints_file = tmp_path / "constraints.json"
    constraints_file.write_text("{notjson}")
    summary = {"run_id": "cli-run", "paths": {"summary": str(tmp_path / "summary.json")}}
    caplog.set_level(logging.ERROR)
    with patch("backend.tools.generate_runs.run_generation", return_value=summary) as mock_run:
        result = generate_runs_cli.main(
            [
                "--ruleset-id",
                _ruleset_id(),
                "--constraints-file",
                str(constraints_file),
                "--data-path",
                str(tmp_path),
                "--count",
                "1",
            ]
        )
    assert result == 1
    assert not mock_run.called
    assert str(constraints_file) in caplog.text
    assert "malformed JSON in constraints file" in caplog.text
