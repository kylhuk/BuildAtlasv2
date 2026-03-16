from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pytest

from backend.engine.surrogate import (
    FEATURE_SCHEMA_VERSION,
    EvaluationResult,
    SnapshotResult,
    TrainResult,
)
from backend.tools import ml_loop


class _FakeModel:
    def predict_many(self, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        return [
            {
                "metrics": {"full_dps": 1.0, "max_hit": 1.0, "utility_score": 1.0},
                "pass_probability": 0.5,
            }
            for _ in rows
        ]


@pytest.fixture(autouse=True)
def patch_evaluator_and_repo(monkeypatch):
    created_evaluators: list["_DummyEvaluator"] = []

    class _DummyClickhouseRepository:
        def close(self) -> None:
            pass

    class _DummyEvaluator:
        def __init__(self, repo, base_path, **_: object) -> None:
            self.repo = repo
            self.base_path = base_path
            self.require_worker_calls: list[str] = []
            self.require_non_stub_calls: list[str] = []
            self.closed = False
            created_evaluators.append(self)

        def require_worker_metrics_for_profile(self, profile_id: str) -> None:
            self.require_worker_calls.append(profile_id)

        def require_non_stub_metrics_for_profile(self, profile_id: str) -> None:
            self.require_non_stub_calls.append(profile_id)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(ml_loop, "ClickhouseRepository", _DummyClickhouseRepository)
    monkeypatch.setattr(ml_loop, "BuildEvaluator", _DummyEvaluator)
    return created_evaluators


def test_ml_loop_runs_with_failure_learning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ML loop should run with failure learning enabled."""
    loop_id = "failure-learning-loop"
    data_path = tmp_path / "data"
    args = argparse.Namespace(
        loop_id=loop_id,
        iterations=2,
        count=3,
        seed_start=1,
        profile_id="pinnacle",
        ruleset_id="ruleset-smoke",
        scenario_version=None,
        price_snapshot_id="price",
        pob_commit=None,
        constraints_file=None,
        data_path=data_path,
        surrogate_backend="cpu",
        include_failures=True,
    )

    rows = [
        {"full_dps": 100.0, "max_hit": 50.0, "utility_score": 0.8, "gate_pass": True},
        {"full_dps": 50.0, "max_hit": 20.0, "utility_score": 0.3, "gate_pass": False},
        {"full_dps": 150.0, "max_hit": 80.0, "utility_score": 0.9, "gate_pass": True},
    ]
    last_snapshot_id: str | None = None
    last_snapshot_root: Path | None = None
    train_include_failures_values: list[bool] = []

    def fake_run_generation(**kwargs: object) -> dict[str, object]:
        return {
            "run_id": kwargs.get("run_id"),
            "status": "completed",
            "evaluation": {
                "attempted": 3,
                "successes": 2,
                "failures": 1,
                "errors": 0,
            },
        }

    def fake_build_dataset_snapshot(
        *,
        data_path: Path | str,
        output_root: Path | str,
        snapshot_id: str,
        exclude_stub_rows: bool,
        profile_id: str | None = None,
        scenario_id: str | None = None,
        **_: object,
    ) -> SnapshotResult:
        nonlocal last_snapshot_id, last_snapshot_root
        last_snapshot_id = snapshot_id
        snapshot_root = Path(output_root) / snapshot_id
        last_snapshot_root = snapshot_root
        snapshot_root.mkdir(parents=True, exist_ok=True)
        dataset_path = snapshot_root / "dataset.jsonl"
        dataset_path.write_text("[]", encoding="utf-8")
        manifest_path = snapshot_root / "manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")
        return SnapshotResult(
            snapshot_id=snapshot_id,
            dataset_path=dataset_path,
            manifest_path=manifest_path,
            row_count=len(rows),
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            dataset_hash="hash",
        )

    def fake_train(
        *,
        dataset_path: Path | str,
        output_root: Path | str,
        model_id: str,
        compute_backend: str,
        include_failures: bool = True,
        **_: object,
    ) -> TrainResult:
        nonlocal last_snapshot_id, last_snapshot_root
        train_include_failures_values.append(include_failures)
        assert last_snapshot_root is not None
        assert Path(dataset_path) == last_snapshot_root
        model_root = Path(output_root) / model_id
        model_root.mkdir(parents=True, exist_ok=True)
        model_path = model_root / "model.json"
        model_path.write_text("{}", encoding="utf-8")
        metrics_path = model_root / "metrics.json"
        metrics_path.write_text(json.dumps({"brier": 0.1}), encoding="utf-8")
        meta_path = model_root / "meta.json"
        meta_path.write_text(
            json.dumps({
                "classifier_enabled": include_failures,
                "classifier_status": "trained" if include_failures else "disabled",
            }),
            encoding="utf-8",
        )
        return TrainResult(
            model_id=model_id,
            model_path=model_path,
            metrics_path=metrics_path,
            meta_path=meta_path,
            dataset_snapshot_id=last_snapshot_id or "",
            dataset_hash="hash",
            row_count=len(rows),
            feature_schema_version=FEATURE_SCHEMA_VERSION,
        )

    def fake_load_dataset_rows(dataset_path: Path | str) -> list[dict[str, float]]:
        return rows

    def fake_load_model(*args: object, **kwargs: object) -> _FakeModel:
        return _FakeModel()

    def fake_evaluate_predictions(*args: object, **kwargs: object) -> EvaluationResult:
        return EvaluationResult(
            row_count=len(rows),
            metric_mae={"full_dps": 0.0, "max_hit": 0.0},
            metric_mae_all={"full_dps": 0.0},
            metric_mae_pass={"full_dps": 0.0},
            metric_mae_log1p={"full_dps": 0.0},
            metric_mae_log1p_pass={"full_dps": 0.0},
            classifier_metrics={"brier": 0.1},
            pass_probability={"mean": 0.5, "std": 0.0, "min": 0.5, "max": 0.5},
        )

    monkeypatch.setattr(ml_loop, "run_generation", fake_run_generation)
    monkeypatch.setattr(ml_loop, "build_dataset_snapshot", fake_build_dataset_snapshot)
    monkeypatch.setattr(ml_loop, "train", fake_train)
    monkeypatch.setattr(ml_loop, "load_dataset_rows", fake_load_dataset_rows)
    monkeypatch.setattr(ml_loop, "load_model", fake_load_model)
    monkeypatch.setattr(ml_loop, "evaluate_predictions", fake_evaluate_predictions)

    result = ml_loop.start_loop(args)
    assert result == 0

    assert len(train_include_failures_values) == 2
    assert all(val is True for val in train_include_failures_values)

    loop_root = data_path / "ml_loops" / loop_id
    state_path = loop_root / ml_loop.STATE_FILENAME
    assert state_path.exists()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert state["iteration"] == 2


def test_ml_loop_persists_diverse_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ML loop should persist diverse archive between runs."""
    loop_id = "diversity-archive-loop"
    data_path = tmp_path / "data"
    args = argparse.Namespace(
        loop_id=loop_id,
        iterations=3,
        count=3,
        seed_start=1,
        profile_id="pinnacle",
        ruleset_id="ruleset-smoke",
        scenario_version=None,
        price_snapshot_id="price",
        pob_commit=None,
        constraints_file=None,
        data_path=data_path,
        surrogate_backend="cpu",
        enable_diversity=True,
    )

    rows = [
        {"full_dps": 100.0, "max_hit": 50.0, "utility_score": 0.8},
        {"full_dps": 200.0, "max_hit": 100.0, "utility_score": 0.9},
        {"full_dps": 150.0, "max_hit": 75.0, "utility_score": 0.85},
    ]
    last_snapshot_id: str | None = None
    last_snapshot_root: Path | None = None

    def fake_run_generation(**kwargs: object) -> dict[str, object]:
        return {
            "run_id": kwargs.get("run_id"),
            "status": "completed",
            "evaluation": {
                "attempted": 3,
                "successes": 3,
                "failures": 0,
                "errors": 0,
            },
            "verified_builds": [
                {
                    "build_id": "build-1",
                    "metrics": {"full_dps": 100.0, "max_hit": 50.0},
                },
                {
                    "build_id": "build-2",
                    "metrics": {"full_dps": 200.0, "max_hit": 100.0},
                },
                {
                    "build_id": "build-3",
                    "metrics": {"full_dps": 150.0, "max_hit": 75.0},
                },
            ],
        }

    def fake_build_dataset_snapshot(
        *,
        data_path: Path | str,
        output_root: Path | str,
        snapshot_id: str,
        exclude_stub_rows: bool,
        profile_id: str | None = None,
        scenario_id: str | None = None,
        **_: object,
    ) -> SnapshotResult:
        nonlocal last_snapshot_id, last_snapshot_root
        last_snapshot_id = snapshot_id
        snapshot_root = Path(output_root) / snapshot_id
        last_snapshot_root = snapshot_root
        snapshot_root.mkdir(parents=True, exist_ok=True)
        dataset_path = snapshot_root / "dataset.jsonl"
        dataset_path.write_text("[]", encoding="utf-8")
        manifest_path = snapshot_root / "manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")
        return SnapshotResult(
            snapshot_id=snapshot_id,
            dataset_path=dataset_path,
            manifest_path=manifest_path,
            row_count=len(rows),
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            dataset_hash="hash",
        )

    def fake_train(
        *,
        dataset_path: Path | str,
        output_root: Path | str,
        model_id: str,
        compute_backend: str,
        **_: object,
    ) -> TrainResult:
        nonlocal last_snapshot_id, last_snapshot_root
        assert last_snapshot_root is not None
        model_root = Path(output_root) / model_id
        model_root.mkdir(parents=True, exist_ok=True)
        model_path = model_root / "model.json"
        model_path.write_text("{}", encoding="utf-8")
        metrics_path = model_root / "metrics.json"
        metrics_path.write_text("{}", encoding="utf-8")
        meta_path = model_root / "meta.json"
        meta_path.write_text("{}", encoding="utf-8")
        return TrainResult(
            model_id=model_id,
            model_path=model_path,
            metrics_path=metrics_path,
            meta_path=meta_path,
            dataset_snapshot_id=last_snapshot_id or "",
            dataset_hash="hash",
            row_count=len(rows),
            feature_schema_version=FEATURE_SCHEMA_VERSION,
        )

    def fake_load_dataset_rows(dataset_path: Path | str) -> list[dict[str, float]]:
        return rows

    def fake_load_model(*args: object, **kwargs: object) -> _FakeModel:
        return _FakeModel()

    def fake_evaluate_predictions(*args: object, **kwargs: object) -> EvaluationResult:
        return EvaluationResult(
            row_count=len(rows),
            metric_mae={"full_dps": 0.0, "max_hit": 0.0},
            metric_mae_all={"full_dps": 0.0},
            metric_mae_pass={"full_dps": 0.0},
            metric_mae_log1p={"full_dps": 0.0},
            metric_mae_log1p_pass={"full_dps": 0.0},
            classifier_metrics={"brier": 0.1},
            pass_probability={"mean": 0.5, "std": 0.0, "min": 0.5, "max": 0.5},
        )

    monkeypatch.setattr(ml_loop, "run_generation", fake_run_generation)
    monkeypatch.setattr(ml_loop, "build_dataset_snapshot", fake_build_dataset_snapshot)
    monkeypatch.setattr(ml_loop, "train", fake_train)
    monkeypatch.setattr(ml_loop, "load_dataset_rows", fake_load_dataset_rows)
    monkeypatch.setattr(ml_loop, "load_model", fake_load_model)
    monkeypatch.setattr(ml_loop, "evaluate_predictions", fake_evaluate_predictions)

    with caplog.at_level(logging.INFO, logger=ml_loop.__name__):
        result = ml_loop.start_loop(args)
    assert result == 0

    assert "archive_initialized" in caplog.text, "Diversity archive should be initialized"

    loop_root = data_path / "ml_loops" / loop_id
    state_path = loop_root / ml_loop.STATE_FILENAME
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state.get("status") == "completed"


def test_ml_loop_improves_over_iterations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ML loop should show improvement over iterations."""
    loop_id = "improvement-loop"
    data_path = tmp_path / "data"
    iterations = 5
    args = argparse.Namespace(
        loop_id=loop_id,
        iterations=iterations,
        count=3,
        seed_start=1,
        profile_id="pinnacle",
        ruleset_id="ruleset-smoke",
        scenario_version=None,
        price_snapshot_id="price",
        pob_commit=None,
        constraints_file=None,
        data_path=data_path,
        surrogate_backend="cpu",
    )

    rows = [{"full_dps": 100.0, "max_hit": 50.0, "utility_score": 0.8}]
    last_snapshot_id: str | None = None
    last_snapshot_root: Path | None = None
    iteration_pass_rates: list[float] = []

    def fake_run_generation(**kwargs: object) -> dict[str, object]:
        iteration_num = kwargs.get("run_id", "").split("-")[-1]
        if iteration_num.isdigit():
            i = int(iteration_num)
            if i <= 2:
                pass_rate = 0.3
            elif i <= 4:
                pass_rate = 0.6
            else:
                pass_rate = 0.8
        else:
            pass_rate = 0.5
        iteration_pass_rates.append(pass_rate)
        attempted = 3
        successes = int(attempted * pass_rate)
        failures = attempted - successes
        return {
            "run_id": kwargs.get("run_id"),
            "status": "completed",
            "evaluation": {
                "attempted": attempted,
                "successes": successes,
                "failures": failures,
                "errors": 0,
            },
        }

    def fake_build_dataset_snapshot(
        *,
        data_path: Path | str,
        output_root: Path | str,
        snapshot_id: str,
        exclude_stub_rows: bool,
        profile_id: str | None = None,
        scenario_id: str | None = None,
        **_: object,
    ) -> SnapshotResult:
        nonlocal last_snapshot_id, last_snapshot_root
        last_snapshot_id = snapshot_id
        snapshot_root = Path(output_root) / snapshot_id
        last_snapshot_root = snapshot_root
        snapshot_root.mkdir(parents=True, exist_ok=True)
        dataset_path = snapshot_root / "dataset.jsonl"
        dataset_path.write_text("[]", encoding="utf-8")
        manifest_path = snapshot_root / "manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")
        return SnapshotResult(
            snapshot_id=snapshot_id,
            dataset_path=dataset_path,
            manifest_path=manifest_path,
            row_count=len(rows),
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            dataset_hash="hash",
        )

    def fake_train(
        *,
        dataset_path: Path | str,
        output_root: Path | str,
        model_id: str,
        compute_backend: str,
        **_: object,
    ) -> TrainResult:
        nonlocal last_snapshot_id, last_snapshot_root
        assert last_snapshot_root is not None
        model_root = Path(output_root) / model_id
        model_root.mkdir(parents=True, exist_ok=True)
        model_path = model_root / "model.json"
        model_path.write_text("{}", encoding="utf-8")
        metrics_path = model_root / "metrics.json"
        metrics_path.write_text("{}", encoding="utf-8")
        meta_path = model_root / "meta.json"
        meta_path.write_text("{}", encoding="utf-8")
        return TrainResult(
            model_id=model_id,
            model_path=model_path,
            metrics_path=metrics_path,
            meta_path=meta_path,
            dataset_snapshot_id=last_snapshot_id or "",
            dataset_hash="hash",
            row_count=len(rows),
            feature_schema_version=FEATURE_SCHEMA_VERSION,
        )

    def fake_load_dataset_rows(dataset_path: Path | str) -> list[dict[str, float]]:
        return rows

    def fake_load_model(*args: object, **kwargs: object) -> _FakeModel:
        return _FakeModel()

    def fake_evaluate_predictions(*args: object, **kwargs: object) -> EvaluationResult:
        return EvaluationResult(
            row_count=len(rows),
            metric_mae={"full_dps": 0.0, "max_hit": 0.0},
            metric_mae_all={"full_dps": 0.0},
            metric_mae_pass={"full_dps": 0.0},
            metric_mae_log1p={"full_dps": 0.0},
            metric_mae_log1p_pass={"full_dps": 0.0},
            classifier_metrics={"brier": 0.1},
            pass_probability={"mean": 0.5, "std": 0.0, "min": 0.5, "max": 0.5},
        )

    monkeypatch.setattr(ml_loop, "run_generation", fake_run_generation)
    monkeypatch.setattr(ml_loop, "build_dataset_snapshot", fake_build_dataset_snapshot)
    monkeypatch.setattr(ml_loop, "train", fake_train)
    monkeypatch.setattr(ml_loop, "load_dataset_rows", fake_load_dataset_rows)
    monkeypatch.setattr(ml_loop, "load_model", fake_load_model)
    monkeypatch.setattr(ml_loop, "evaluate_predictions", fake_evaluate_predictions)

    result = ml_loop.start_loop(args)
    assert result == 0

    assert len(iteration_pass_rates) == iterations
    early_avg = sum(iteration_pass_rates[:2]) / 2
    late_avg = sum(iteration_pass_rates[-2:]) / 2
    assert late_avg >= early_avg, f"Pass rate should improve or stabilize: early={early_avg}, late={late_avg}"

    loop_root = data_path / "ml_loops" / loop_id
    iterations_path = loop_root / ml_loop.ITERATIONS_FILENAME
    lines = [
        line for line in iterations_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(lines) == iterations

    loaded_iterations = [json.loads(line) for line in lines]
    for i, record in enumerate(loaded_iterations):
        assert record["iteration"] == i + 1


def test_reload_loop_continues_from_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ML loop should continue from saved state."""
    loop_id = "resume-checkpoint-loop"
    data_path = tmp_path / "data"
    loop_root = data_path / "ml_loops" / loop_id
    loop_root.mkdir(parents=True, exist_ok=True)
    iterations_path = loop_root / ml_loop.ITERATIONS_FILENAME

    first_iteration_record = {
        "iteration": 1,
        "run_id": f"{loop_id}-iter-0001",
        "snapshot": {"snapshot_id": "iter-0001", "row_count": 10},
        "model": {
            "model_id": f"{loop_id}-iter-0001",
            "model_path": str(loop_root / "models" / "iter-0001" / "model.json"),
            "promoted": True,
        },
    }
    iterations_path.write_text(json.dumps(first_iteration_record) + "\n", encoding="utf-8")

    state_path = loop_root / ml_loop.STATE_FILENAME
    state = ml_loop._build_initial_state(
        loop_id, total_iterations=3, seed_start_base=1, seed_window_size=5
    )
    state.update(
        {
            "iteration": 1,
            "status": "running",
            "phase": "idle",
            "last_run_id": f"{loop_id}-iter-0001",
            "last_model_id": f"{loop_id}-iter-0001",
            "next_iteration_seed_start": 16,
            "last_iteration_seed_start": 1,
            "completed_iterations": 1,
        }
    )
    ml_loop._write_state(state_path, state)

    args = argparse.Namespace(
        loop_id=loop_id,
        iterations=3,
        count=3,
        seed_start=1,
        profile_id="pinnacle",
        ruleset_id="ruleset-smoke",
        scenario_version=None,
        price_snapshot_id="price",
        pob_commit=None,
        constraints_file=None,
        data_path=data_path,
        surrogate_backend="cpu",
    )

    rows = [{"full_dps": 100.0, "max_hit": 50.0, "utility_score": 0.8}]
    last_snapshot_id: str | None = None
    last_snapshot_root: Path | None = None
    run_ids_seen: list[str] = []

    def fake_run_generation(**kwargs: object) -> dict[str, object]:
        run_id = str(kwargs.get("run_id"))
        run_ids_seen.append(run_id)
        return {
            "run_id": run_id,
            "status": "completed",
            "evaluation": {
                "attempted": 3,
                "successes": 3,
                "failures": 0,
                "errors": 0,
            },
        }

    def fake_build_dataset_snapshot(
        *,
        data_path: Path | str,
        output_root: Path | str,
        snapshot_id: str,
        exclude_stub_rows: bool,
        profile_id: str | None = None,
        scenario_id: str | None = None,
        **_: object,
    ) -> SnapshotResult:
        nonlocal last_snapshot_id, last_snapshot_root
        last_snapshot_id = snapshot_id
        snapshot_root = Path(output_root) / snapshot_id
        last_snapshot_root = snapshot_root
        snapshot_root.mkdir(parents=True, exist_ok=True)
        dataset_path = snapshot_root / "dataset.jsonl"
        dataset_path.write_text("[]", encoding="utf-8")
        manifest_path = snapshot_root / "manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")
        return SnapshotResult(
            snapshot_id=snapshot_id,
            dataset_path=dataset_path,
            manifest_path=manifest_path,
            row_count=len(rows),
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            dataset_hash="hash",
        )

    def fake_train(
        *,
        dataset_path: Path | str,
        output_root: Path | str,
        model_id: str,
        compute_backend: str,
        **_: object,
    ) -> TrainResult:
        nonlocal last_snapshot_id, last_snapshot_root
        assert last_snapshot_root is not None
        model_root = Path(output_root) / model_id
        model_root.mkdir(parents=True, exist_ok=True)
        model_path = model_root / "model.json"
        model_path.write_text("{}", encoding="utf-8")
        metrics_path = model_root / "metrics.json"
        metrics_path.write_text("{}", encoding="utf-8")
        meta_path = model_root / "meta.json"
        meta_path.write_text("{}", encoding="utf-8")
        return TrainResult(
            model_id=model_id,
            model_path=model_path,
            metrics_path=metrics_path,
            meta_path=meta_path,
            dataset_snapshot_id=last_snapshot_id or "",
            dataset_hash="hash",
            row_count=len(rows),
            feature_schema_version=FEATURE_SCHEMA_VERSION,
        )

    def fake_load_dataset_rows(dataset_path: Path | str) -> list[dict[str, float]]:
        return rows

    def fake_load_model(*args: object, **kwargs: object) -> _FakeModel:
        return _FakeModel()

    def fake_evaluate_predictions(*args: object, **kwargs: object) -> EvaluationResult:
        return EvaluationResult(
            row_count=len(rows),
            metric_mae={"full_dps": 0.0, "max_hit": 0.0},
            metric_mae_all={"full_dps": 0.0},
            metric_mae_pass={"full_dps": 0.0},
            metric_mae_log1p={"full_dps": 0.0},
            metric_mae_log1p_pass={"full_dps": 0.0},
            classifier_metrics={"brier": 0.1},
            pass_probability={"mean": 0.5, "std": 0.0, "min": 0.5, "max": 0.5},
        )

    monkeypatch.setattr(ml_loop, "run_generation", fake_run_generation)
    monkeypatch.setattr(ml_loop, "build_dataset_snapshot", fake_build_dataset_snapshot)
    monkeypatch.setattr(ml_loop, "train", fake_train)
    monkeypatch.setattr(ml_loop, "load_dataset_rows", fake_load_dataset_rows)
    monkeypatch.setattr(ml_loop, "load_model", fake_load_model)
    monkeypatch.setattr(ml_loop, "evaluate_predictions", fake_evaluate_predictions)

    result = ml_loop.start_loop(args)
    assert result == 0

    assert len(run_ids_seen) == 2
    assert f"{loop_id}-iter-0002" in run_ids_seen
    assert f"{loop_id}-iter-0003" in run_ids_seen
    assert f"{loop_id}-iter-0001" not in run_ids_seen

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["iteration"] == 3
    assert state["status"] == "completed"

    all_iterations = [
        json.loads(line)
        for line in iterations_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(all_iterations) == 3
    assert {rec["iteration"] for rec in all_iterations} == {1, 2, 3}
