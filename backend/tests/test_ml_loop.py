from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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


@pytest.mark.parametrize("iterations", [2])
def test_start_loop_executes_two_iterations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, iterations: int
) -> None:
    loop_id = "ml-loop-test"
    data_path = tmp_path / "data"
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

    rows = [{"full_dps": 1.0, "max_hit": 2.0, "utility_score": 3.0}]
    last_snapshot_id: str | None = None
    seed_calls: list[int] = []
    surrogate_enabled_calls: list[bool] = []

    last_snapshot_root: Path | None = None
    run_count = 0

    def fake_run_generation(**kwargs: object) -> dict[str, object]:
        nonlocal run_count
        run_count += 1
        assert kwargs.get("run_mode") == "standard"
        assert kwargs.get("count") == 12
        assert kwargs.get("candidate_pool_size") == 12
        assert kwargs.get("optimizer_iterations") == 3
        assert kwargs.get("optimizer_elite_count") == 16
        seed_start = kwargs.get("seed_start")
        assert isinstance(seed_start, int)
        seed_calls.append(seed_start)
        count_value = kwargs.get("count")
        assert isinstance(count_value, int)
        surrogate_enabled = bool(kwargs.get("surrogate_enabled"))
        surrogate_enabled_calls.append(surrogate_enabled)
        if run_count == 1:
            assert not surrogate_enabled
            assert kwargs.get("surrogate_top_k") is None
        else:
            assert surrogate_enabled
            assert kwargs.get("surrogate_top_k") == 128
            assert kwargs.get("surrogate_exploration_pct") == 0.10
        surrogate_summary = {
            "enabled": surrogate_enabled,
            "status": "active" if surrogate_enabled else "disabled",
            "counts": {
                "selected": kwargs.get("surrogate_top_k") or count_value,
                "pruned": 0,
            },
        }
        return {
            "run_id": kwargs.get("run_id"),
            "status": "completed",
            "evaluation": {"attempted": 1, "successes": 1, "failures": 0, "errors": 0},
            "surrogate": surrogate_summary,
        }

    def fake_build_dataset_snapshot(
        *,
        data_path: Path | str,
        output_root: Path | str,
        snapshot_id: str,
        exclude_stub_rows: bool,
        profile_id: str | None = None,
        scenario_id: str | None = None,
    ) -> SnapshotResult:
        nonlocal last_snapshot_id, last_snapshot_root
        assert exclude_stub_rows is True
        assert profile_id == "pinnacle"
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
        assert last_snapshot_root is not None
        assert Path(dataset_path) == last_snapshot_root
        assert compute_backend == "cpu"
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
        assert last_snapshot_root is not None
        assert Path(dataset_path) == last_snapshot_root
        return rows

    def fake_load_model(*args: object, **kwargs: object) -> _FakeModel:
        return _FakeModel()

    def fake_evaluate_predictions(*args: object, **kwargs: object) -> EvaluationResult:
        return EvaluationResult(
            row_count=len(rows),
            metric_mae={"full_dps": 0.0, "max_hit": 0.0},
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

    assert seed_calls == [1, 13]
    assert surrogate_enabled_calls == [False, True]

    loop_root = data_path / "ml_loops" / loop_id
    state_path = loop_root / ml_loop.STATE_FILENAME
    assert state_path.exists()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert state["iteration"] == iterations
    assert state["last_improvement"].get("improved") is False
    assert state["last_model_id"] == "ml-loop-test-iter-0001"
    assert str(state["last_model_path"]).endswith("ml-loop-test-iter-0001/model.json")

    iterations_path = loop_root / ml_loop.ITERATIONS_FILENAME
    lines = [
        line for line in iterations_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(lines) == iterations
    loaded_iterations = [json.loads(line) for line in lines]
    assert {entry["iteration"] for entry in loaded_iterations} == set(range(1, iterations + 1))
    assert loaded_iterations[0]["model"]["promoted"] is True
    assert loaded_iterations[1]["model"]["promoted"] is False
    assert loaded_iterations[-1]["model"]["promoted_model_id"] == "ml-loop-test-iter-0001"

    for iteration in range(1, iterations + 1):
        checkpoint = loop_root / ml_loop.CHECKPOINTS_DIR_NAME / f"iter-{iteration:04d}.json"
        assert checkpoint.exists()
        record = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert record["iteration"] == iteration


def test_start_loop_keeps_champion_when_challenger_regresses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop_id = "champion-loop"
    iterations = 2
    data_path = tmp_path / "data"
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

    rows = [{"full_dps": 1.0, "max_hit": 2.0, "utility_score": 3.0}]
    last_snapshot_id: str | None = None
    last_snapshot_root: Path | None = None

    def fake_run_generation(**kwargs: object) -> dict[str, object]:
        assert kwargs.get("run_mode") == "standard"
        return {
            "run_id": kwargs.get("run_id"),
            "status": "completed",
            "evaluation": {"attempted": 1, "successes": 1, "failures": 0, "errors": 0},
        }

    def fake_build_dataset_snapshot(
        *,
        data_path: Path | str,
        output_root: Path | str,
        snapshot_id: str,
        exclude_stub_rows: bool,
        profile_id: str | None = None,
        scenario_id: str | None = None,
    ) -> SnapshotResult:
        nonlocal last_snapshot_id, last_snapshot_root
        assert exclude_stub_rows is True
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
        assert last_snapshot_root is not None
        assert Path(dataset_path) == last_snapshot_root
        assert compute_backend == "cpu"
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
        assert last_snapshot_root is not None
        assert Path(dataset_path) == last_snapshot_root
        return rows

    class _TaggedModel:
        def __init__(self, tag: str) -> None:
            self.tag = tag

        def predict_many(self, rows: list[dict[str, object]]) -> list[dict[str, object]]:
            _ = rows
            return [{"model_tag": self.tag}]

    def fake_load_model(model_path: Path | str) -> _TaggedModel:
        model_root = Path(model_path).parent
        return _TaggedModel(model_root.name)

    def fake_evaluate_predictions(
        rows: list[dict[str, float]],
        predictions: list[dict[str, object]],
    ) -> EvaluationResult:
        _ = rows
        tag = str(predictions[0].get("model_tag")) if predictions else ""
        if tag == "champion-loop-iter-0002":
            return EvaluationResult(
                row_count=1,
                metric_mae={"full_dps": 1.2, "max_hit": 0.2},
                pass_probability={"mean": 0.4, "std": 0.0, "min": 0.4, "max": 0.4},
            )
        return EvaluationResult(
            row_count=1,
            metric_mae={"full_dps": 1.0, "max_hit": 0.1},
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

    loop_root = data_path / "ml_loops" / loop_id
    state = json.loads((loop_root / ml_loop.STATE_FILENAME).read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert state["last_model_id"] == "champion-loop-iter-0001"
    assert str(state["last_model_path"]).endswith("champion-loop-iter-0001/model.json")
    assert state["last_improvement"]["improved"] is False

    records = [
        json.loads(line)
        for line in (loop_root / ml_loop.ITERATIONS_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(records) == 2
    assert records[0]["model"]["promoted"] is True
    assert records[0]["model"]["promoted_model_id"] == "champion-loop-iter-0001"
    assert records[1]["model"]["promoted"] is False
    assert records[1]["model"]["promoted_model_id"] == "champion-loop-iter-0001"


def test_start_loop_resumes_with_next_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loop_id = "resume-loop"
    data_path = tmp_path / "data"
    loop_root = data_path / "ml_loops" / loop_id
    loop_root.mkdir(parents=True, exist_ok=True)
    iterations_path = loop_root / ml_loop.ITERATIONS_FILENAME
    iterations_path.write_text('{"iteration": 1}\n', encoding="utf-8")
    state_path = loop_root / ml_loop.STATE_FILENAME
    state = ml_loop._build_initial_state(
        loop_id, total_iterations=2, seed_start_base=1, seed_window_size=5
    )
    state.update(
        {
            "iteration": 1,
            "status": "running",
            "phase": "idle",
            "last_run_id": f"{loop_id}-iter-0001",
            "last_model_id": f"{loop_id}-iter-0001",
            "next_iteration_seed_start": 42,
            "last_iteration_seed_start": 37,
        }
    )
    ml_loop._write_state(state_path, state)

    args = argparse.Namespace(
        loop_id=loop_id,
        iterations=2,
        count=5,
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

    rows = [{"full_dps": 1.0, "max_hit": 2.0, "utility_score": 3.0}]
    last_snapshot_id: str | None = None
    last_snapshot_root: Path | None = None
    seed_calls: list[int] = []

    def fake_run_generation(**kwargs: object) -> dict[str, object]:
        seed_start = kwargs.get("seed_start")
        assert isinstance(seed_start, int)
        seed_calls.append(seed_start)
        return {
            "run_id": kwargs.get("run_id"),
            "status": "completed",
            "evaluation": {"attempted": 1, "successes": 1, "failures": 0, "errors": 0},
        }

    def fake_build_dataset_snapshot(
        *,
        data_path: Path | str,
        output_root: Path | str,
        snapshot_id: str,
        exclude_stub_rows: bool,
        profile_id: str | None = None,
        scenario_id: str | None = None,
    ) -> SnapshotResult:
        nonlocal last_snapshot_id, last_snapshot_root
        assert exclude_stub_rows is True
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
        assert last_snapshot_root is not None
        assert Path(dataset_path) == last_snapshot_root
        assert compute_backend == "cpu"
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
        assert last_snapshot_root is not None
        assert Path(dataset_path) == last_snapshot_root
        return rows

    def fake_load_model(*args: object, **kwargs: object) -> _FakeModel:
        return _FakeModel()

    def fake_evaluate_predictions(*args: object, **kwargs: object) -> EvaluationResult:
        return EvaluationResult(
            row_count=len(rows),
            metric_mae={"full_dps": 0.0, "max_hit": 0.0},
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
    assert seed_calls == [42]

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["iteration"] == 2
    assert state["last_iteration_seed_start"] == 42
    assert state["next_iteration_seed_start"] == 62
    assert state["status"] == "completed"


def test_start_loop_records_failure_checkpoint_on_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop_id = "failure-loop"
    data_path = tmp_path / "data"
    args = argparse.Namespace(
        loop_id=loop_id,
        iterations=1,
        count=4,
        seed_start=2,
        profile_id="pinnacle",
        ruleset_id="ruleset-smoke",
        scenario_version=None,
        price_snapshot_id="price",
        pob_commit=None,
        constraints_file=None,
        data_path=data_path,
        surrogate_backend="cpu",
    )

    def fake_run_generation(**kwargs: object) -> dict[str, object]:
        return {
            "run_id": kwargs.get("run_id"),
            "status": "completed",
            "evaluation": {"attempted": 1, "successes": 1, "failures": 0, "errors": 0},
        }

    def fake_build_dataset_snapshot(
        *,
        data_path: Path | str,
        output_root: Path | str,
        snapshot_id: str,
        exclude_stub_rows: bool,
        profile_id: str | None = None,
        scenario_id: str | None = None,
    ) -> SnapshotResult:
        raise RuntimeError("boom")

    monkeypatch.setattr(ml_loop, "run_generation", fake_run_generation)
    monkeypatch.setattr(ml_loop, "build_dataset_snapshot", fake_build_dataset_snapshot)

    result = ml_loop.start_loop(args)
    assert result == 1

    state_path = data_path / "ml_loops" / loop_id / ml_loop.STATE_FILENAME
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["failed_iteration"] == 1
    assert state["failed_phase"] == "snapshot"
    assert state["failed_at_utc"] is not None
    failure_path = Path(state["last_failure_checkpoint_path"])
    assert failure_path.exists()
    payload = json.loads(failure_path.read_text(encoding="utf-8"))
    assert payload["iteration"] == 1
    assert payload["phase"] == "snapshot"
    assert payload["error"] == "boom"
    assert payload["seed_start"] == args.seed_start
    assert payload["run_id"] == f"{loop_id}-iter-0001"
    context = payload.get("state_context", {})
    assert context.get("last_run_id") == payload["run_id"]
    assert context.get("next_iteration_seed_start") == args.seed_start + (args.count * 4)


def test_start_loop_retries_snapshot_without_profile_filter_when_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop_id = "snapshot-filter-retry"
    data_path = tmp_path / "data"
    args = argparse.Namespace(
        loop_id=loop_id,
        iterations=1,
        count=2,
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

    rows = [{"full_dps": 1.0, "max_hit": 2.0, "utility_score": 3.0}]
    last_snapshot_id: str | None = None
    last_snapshot_root: Path | None = None
    snapshot_call_profiles: list[str | None] = []

    def fake_run_generation(**kwargs: object) -> dict[str, object]:
        return {
            "run_id": kwargs.get("run_id"),
            "status": "completed",
            "evaluation": {"attempted": 2, "successes": 2, "failures": 0, "errors": 0},
        }

    def fake_build_dataset_snapshot(
        *,
        data_path: Path | str,
        output_root: Path | str,
        snapshot_id: str,
        exclude_stub_rows: bool,
        profile_id: str | None = None,
        scenario_id: str | None = None,
    ) -> SnapshotResult:
        nonlocal last_snapshot_id, last_snapshot_root
        assert exclude_stub_rows is True
        snapshot_call_profiles.append(profile_id)
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
            row_count=0 if len(snapshot_call_profiles) == 1 else len(rows),
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
        assert last_snapshot_root is not None
        assert Path(dataset_path) == last_snapshot_root
        assert compute_backend == "cpu"
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
        assert last_snapshot_root is not None
        assert Path(dataset_path) == last_snapshot_root
        return rows

    def fake_load_model(*args: object, **kwargs: object) -> _FakeModel:
        return _FakeModel()

    def fake_evaluate_predictions(*args: object, **kwargs: object) -> EvaluationResult:
        return EvaluationResult(
            row_count=len(rows),
            metric_mae={"full_dps": 0.0, "max_hit": 0.0},
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
    assert snapshot_call_profiles == ["pinnacle", None]


def test_start_loop_skips_when_generation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop_id = "generation-failure"
    data_path = tmp_path / "data"
    args = argparse.Namespace(
        loop_id=loop_id,
        iterations=1,
        count=4,
        seed_start=2,
        profile_id="pinnacle",
        ruleset_id="ruleset-smoke",
        scenario_version=None,
        price_snapshot_id="price",
        pob_commit=None,
        constraints_file=None,
        data_path=data_path,
        surrogate_backend="cpu",
    )

    def fake_run_generation(**kwargs: object) -> dict[str, object]:
        return {
            "run_id": kwargs.get("run_id"),
            "status": "failed",
            "evaluation": {"attempted": 2, "successes": 0, "failures": 2, "errors": 0},
            "status_reason": {
                "code": "no_verified_evaluations",
                "message": "no verified builds",
            },
        }

    def fake_fail(*args: object, **kwargs: object) -> None:
        pytest.fail("should not run after generation failure")

    monkeypatch.setattr(ml_loop, "run_generation", fake_run_generation)
    monkeypatch.setattr(ml_loop, "build_dataset_snapshot", fake_fail)
    monkeypatch.setattr(ml_loop, "train", fake_fail)
    monkeypatch.setattr(ml_loop, "load_dataset_rows", fake_fail)
    monkeypatch.setattr(ml_loop, "load_model", fake_fail)
    monkeypatch.setattr(ml_loop, "evaluate_predictions", fake_fail)

    result = ml_loop.start_loop(args)
    assert result == 0

    state_path = data_path / "ml_loops" / loop_id / ml_loop.STATE_FILENAME
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert state["phase"] == "idle"
    assert state["iteration"] == 1
    assert state["last_generation_status"] == "failed"
    assert state["last_generation_status_reason_code"] == "no_verified_evaluations"
    assert state["last_generation_verified"] == 0
    assert state["last_generation_attempted"] == 2
    assert state["last_iteration_outcome"] == "skipped_generation_unhealthy"
    assert state["skipped_iterations_total"] == 1
    assert state["last_skipped_iteration"] == 1
    assert state["last_skip_reason_code"] == "no_verified_evaluations"

    loop_root = data_path / "ml_loops" / loop_id
    records = [
        json.loads(line)
        for line in (loop_root / ml_loop.ITERATIONS_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    record = records[0]
    assert record["iteration"] == 1
    assert record["iteration_outcome"] == "skipped_generation_unhealthy"
    assert record["run_status"] == "failed"
    assert record["skip_reason_code"] == "no_verified_evaluations"
    assert record["snapshot"] is None
    assert record["model"] is None
    assert record["evaluation"] is None


def test_start_loop_continues_after_unhealthy_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop_id = "generation-recovery"
    data_path = tmp_path / "data"
    args = argparse.Namespace(
        loop_id=loop_id,
        iterations=2,
        count=4,
        seed_start=2,
        profile_id="pinnacle",
        ruleset_id="ruleset-smoke",
        scenario_version=None,
        price_snapshot_id="price",
        pob_commit=None,
        constraints_file=None,
        data_path=data_path,
        surrogate_backend="cpu",
    )

    rows = [{"full_dps": 1.0, "max_hit": 2.0, "utility_score": 3.0}]
    last_snapshot_id: str | None = None
    last_snapshot_root: Path | None = None
    seed_calls: list[int] = []
    generation_calls = 0
    snapshot_calls = 0
    train_calls = 0

    def fake_run_generation(**kwargs: object) -> dict[str, object]:
        nonlocal generation_calls
        assert kwargs.get("run_mode") == "standard"
        generation_calls += 1
        seed_start = kwargs.get("seed_start")
        assert isinstance(seed_start, int)
        seed_calls.append(seed_start)
        if generation_calls == 1:
            return {
                "run_id": kwargs.get("run_id"),
                "status": "failed",
                "evaluation": {"attempted": 2, "successes": 0, "failures": 2, "errors": 0},
                "status_reason": {
                    "code": "no_verified_evaluations",
                    "message": "no verified builds",
                },
            }
        return {
            "run_id": kwargs.get("run_id"),
            "status": "completed",
            "evaluation": {"attempted": 2, "successes": 2, "failures": 0, "errors": 0},
        }

    def fake_build_dataset_snapshot(
        *,
        data_path: Path | str,
        output_root: Path | str,
        snapshot_id: str,
        exclude_stub_rows: bool,
        profile_id: str | None = None,
        scenario_id: str | None = None,
    ) -> SnapshotResult:
        nonlocal snapshot_calls, last_snapshot_id, last_snapshot_root
        snapshot_calls += 1
        assert exclude_stub_rows is True
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
        nonlocal train_calls
        train_calls += 1
        assert last_snapshot_root is not None
        assert Path(dataset_path) == last_snapshot_root
        assert compute_backend == "cpu"
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
        assert last_snapshot_root is not None
        assert Path(dataset_path) == last_snapshot_root
        return rows

    def fake_load_model(*args: object, **kwargs: object) -> _FakeModel:
        return _FakeModel()

    def fake_evaluate_predictions(*args: object, **kwargs: object) -> EvaluationResult:
        return EvaluationResult(
            row_count=len(rows),
            metric_mae={"full_dps": 0.0, "max_hit": 0.0},
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
    assert seed_calls == [2, 18]
    assert snapshot_calls == 1
    assert train_calls == 1

    loop_root = data_path / "ml_loops" / loop_id
    state = json.loads((loop_root / ml_loop.STATE_FILENAME).read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert state["iteration"] == 2
    assert state["skipped_iterations_total"] == 1
    assert state["last_skipped_iteration"] == 1
    assert state["last_model_id"] == "generation-recovery-iter-0002"

    records = [
        json.loads(line)
        for line in (loop_root / ml_loop.ITERATIONS_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(records) == 2
    assert records[0]["iteration"] == 1
    assert records[0]["iteration_outcome"] == "skipped_generation_unhealthy"
    assert records[0]["run_status"] == "failed"
    assert records[0]["snapshot"] is None
    assert records[1]["iteration"] == 2
    assert records[1]["iteration_outcome"] == "completed"
    assert records[1]["run_status"] == "completed"
    assert records[1]["model"]["promoted"] is True


def test_start_loop_endless_respects_stop_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop_id = "endless-loop"
    data_path = tmp_path / "data"
    args = argparse.Namespace(
        loop_id=loop_id,
        iterations=0,
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

    rows = [{"full_dps": 1.0, "max_hit": 2.0, "utility_score": 3.0}]
    last_snapshot_id: str | None = None
    last_snapshot_root: Path | None = None

    def fake_run_generation(**kwargs: object) -> dict[str, object]:
        assert kwargs.get("run_mode") == "standard"
        return {
            "run_id": kwargs.get("run_id"),
            "status": "completed",
            "evaluation": {"attempted": 1, "successes": 1, "failures": 0, "errors": 0},
        }

    def fake_build_dataset_snapshot(
        *,
        data_path: Path | str,
        output_root: Path | str,
        snapshot_id: str,
        exclude_stub_rows: bool,
        profile_id: str | None = None,
        scenario_id: str | None = None,
    ) -> SnapshotResult:
        nonlocal last_snapshot_id, last_snapshot_root
        assert exclude_stub_rows is True
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
        assert last_snapshot_root is not None
        assert Path(dataset_path) == last_snapshot_root
        assert compute_backend == "cpu"
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
        assert last_snapshot_root is not None
        assert Path(dataset_path) == last_snapshot_root
        return rows

    def fake_load_model(*args: object, **kwargs: object) -> _FakeModel:
        return _FakeModel()

    def fake_evaluate_predictions(*args: object, **kwargs: object) -> EvaluationResult:
        return EvaluationResult(
            row_count=len(rows),
            metric_mae={"full_dps": 0.0, "max_hit": 0.0},
            pass_probability={"mean": 0.5, "std": 0.0, "min": 0.5, "max": 0.5},
        )

    stop_flag_set = False
    original_persist_state = ml_loop._persist_state

    def fake_persist_state(
        state_path: Path, state: dict[str, Any], **updates: Any
    ) -> dict[str, Any]:
        nonlocal stop_flag_set
        result = original_persist_state(state_path, state, **updates)
        if not stop_flag_set and updates.get("phase") == "idle" and result.get("iteration") == 1:
            result["stop_requested"] = True
            ml_loop._write_state(state_path, result)
            stop_flag_set = True
        return result

    monkeypatch.setattr(ml_loop, "_persist_state", fake_persist_state)
    monkeypatch.setattr(ml_loop, "run_generation", fake_run_generation)
    monkeypatch.setattr(ml_loop, "build_dataset_snapshot", fake_build_dataset_snapshot)
    monkeypatch.setattr(ml_loop, "train", fake_train)
    monkeypatch.setattr(ml_loop, "load_dataset_rows", fake_load_dataset_rows)
    monkeypatch.setattr(ml_loop, "load_model", fake_load_model)
    monkeypatch.setattr(ml_loop, "evaluate_predictions", fake_evaluate_predictions)

    result = ml_loop.start_loop(args)
    assert result == 0

    loop_root = data_path / "ml_loops" / loop_id
    state_path = loop_root / ml_loop.STATE_FILENAME
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["stop_requested"] is True
    assert state["status"] == "stopped"

    iterations_path = loop_root / ml_loop.ITERATIONS_FILENAME
    lines = [
        line for line in iterations_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["iteration"] == 1


def test_compute_improvement_uses_pass_log1p_and_classifier_brier() -> None:
    current = EvaluationResult(
        row_count=1,
        metric_mae={"full_dps": 1.0, "max_hit": 1.0},
        metric_mae_log1p_pass={"full_dps": 0.40, "max_hit": 0.30},
        classifier_metrics={"brier": 0.20},
        pass_probability={"mean": 0.5, "std": 0.0, "min": 0.5, "max": 0.5},
    )
    previous = EvaluationResult(
        row_count=1,
        metric_mae={"full_dps": 1.0, "max_hit": 1.0},
        metric_mae_log1p_pass={"full_dps": 0.52, "max_hit": 0.41},
        classifier_metrics={"brier": 0.24},
        pass_probability={"mean": 0.5, "std": 0.0, "min": 0.5, "max": 0.5},
    )
    improvement = ml_loop._compute_improvement(current, previous)
    assert improvement["improved"] is True
    assert improvement["promotion_score_delta"] is not None
    assert float(improvement["promotion_score_delta"]) > 0


def test_compute_improvement_requires_epsilon_gap() -> None:
    current = EvaluationResult(
        row_count=1,
        metric_mae={"full_dps": 1.0, "max_hit": 1.0},
        metric_mae_log1p_pass={"full_dps": 0.50, "max_hit": 0.50},
        classifier_metrics={"brier": 0.20},
        pass_probability={"mean": 0.5, "std": 0.0, "min": 0.5, "max": 0.5},
    )
    previous = EvaluationResult(
        row_count=1,
        metric_mae={"full_dps": 1.0, "max_hit": 1.0},
        metric_mae_log1p_pass={"full_dps": 0.50, "max_hit": 0.50},
        classifier_metrics={"brier": 0.20},
        pass_probability={"mean": 0.5, "std": 0.0, "min": 0.5, "max": 0.5},
    )
    improvement = ml_loop._compute_improvement(current, previous)
    assert improvement["improved"] is False


def test_compute_improvement_records_classifier_skip_reason_for_single_class() -> None:
    current = EvaluationResult(
        row_count=1,
        metric_mae={"full_dps": 0.9, "max_hit": 0.5},
        metric_mae_log1p_pass={"full_dps": 0.35, "max_hit": 0.25},
        classifier_metrics={
            "brier": 0.15,
            "labeled_count": 2,
            "positive_count": 0,
            "negative_count": 2,
        },
        pass_probability={"mean": 0.4, "std": 0.0, "min": 0.4, "max": 0.4},
    )
    previous = EvaluationResult(
        row_count=1,
        metric_mae={"full_dps": 1.1, "max_hit": 0.7},
        metric_mae_log1p_pass={"full_dps": 0.40, "max_hit": 0.30},
        classifier_metrics={"brier": 0.18},
        pass_probability={"mean": 0.35, "std": 0.0, "min": 0.35, "max": 0.35},
    )
    improvement = ml_loop._compute_improvement(current, previous)
    assert improvement["pass_probability_mean_delta"] is None
    reason = improvement["current_classifier_skip_reason"]
    assert isinstance(reason, str) and "single-class" in reason
    assert improvement["previous_classifier_skip_reason"] is None


def test_stop_loop_sets_stop_requested(tmp_path: Path) -> None:
    loop_id = "stop-loop"
    data_path = tmp_path / "data"
    loop_root = data_path / "ml_loops" / loop_id
    loop_root.mkdir(parents=True, exist_ok=True)
    state_path = loop_root / ml_loop.STATE_FILENAME
    initial_state = ml_loop._build_initial_state(
        loop_id, total_iterations=5, seed_start_base=1, seed_window_size=5
    )
    ml_loop._write_state(state_path, initial_state)

    args = argparse.Namespace(loop_id=loop_id, data_path=data_path)
    result = ml_loop.stop_loop(args)
    assert result == 0

    final_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert final_state["stop_requested"] is True
    assert final_state["phase"] == "stop_requested"


def test_status_loop_reports_missing_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = argparse.Namespace(loop_id="missing-loop", data_path=tmp_path, output_format="human")
    result = ml_loop.status_loop(args)
    assert result == 1

    output = capsys.readouterr().out.strip()
    payload = json.loads(output)
    assert payload["error"] == "loop_not_found"
    assert payload["loop_id"] == "missing-loop"


def test_status_loop_human_readable_iteration_comparison(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    loop_id = "status-human"
    data_path = tmp_path / "data"
    loop_root = data_path / "ml_loops" / loop_id
    loop_root.mkdir(parents=True, exist_ok=True)
    state_path = loop_root / ml_loop.STATE_FILENAME
    state = ml_loop._build_initial_state(
        loop_id,
        total_iterations=None,
        seed_start_base=1,
        seed_window_size=5,
    )
    state.update(
        {
            "status": "running",
            "phase": "train",
            "iteration": 2,
            "last_run_id": f"{loop_id}-iter-0002",
            "last_snapshot_id": "iter-0002",
            "last_model_id": f"{loop_id}-iter-0002",
            "last_model_path": str(loop_root / "models" / f"{loop_id}-iter-0002" / "model.json"),
            "last_generation_status": "completed",
            "last_generation_attempted": 5,
            "last_generation_verified": 3,
            "last_generation_failures": 0,
            "last_generation_errors": 0,
        }
    )
    ml_loop._write_state(state_path, state)

    iterations_path = loop_root / ml_loop.ITERATIONS_FILENAME
    record_1 = {
        "iteration": 1,
        "run_status": "completed",
        "generation": {
            "status": "completed",
            "status_reason_code": None,
            "status_reason_message": None,
            "attempted": 5,
            "successes": 3,
            "failures": 0,
            "errors": 0,
        },
        "model": {
            "model_id": f"{loop_id}-iter-0001",
            "promoted": True,
            "compute_backend_resolved": "cuda",
            "token_learner_backend": "torch_sparse_sgd",
        },
        "evaluation": {
            "current": {
                "metric_mae": {"full_dps": 1.1, "max_hit": 0.7},
                "pass_probability": {"mean": 0.42},
            },
            "previous": None,
            "improvement": {"improved": False},
        },
    }
    record_2 = {
        "iteration": 2,
        "run_status": "completed",
        "generation": {
            "status": "completed",
            "status_reason_code": None,
            "status_reason_message": None,
            "attempted": 5,
            "successes": 3,
            "failures": 0,
            "errors": 0,
        },
        "model": {
            "model_id": f"{loop_id}-iter-0002",
            "promoted": True,
            "compute_backend_resolved": "cuda",
            "token_learner_backend": "torch_sparse_sgd",
        },
        "evaluation": {
            "current": {
                "metric_mae": {"full_dps": 0.9, "max_hit": 0.6},
                "pass_probability": {"mean": 0.50},
            },
            "previous": {
                "metric_mae": {"full_dps": 1.1, "max_hit": 0.7},
                "pass_probability": {"mean": 0.42},
            },
            "improvement": {"improved": True},
        },
    }
    iterations_path.write_text(
        f"{json.dumps(record_1)}\n{json.dumps(record_2)}\n",
        encoding="utf-8",
    )

    args = argparse.Namespace(
        loop_id=loop_id,
        data_path=data_path,
        output_format="human",
        history=5,
    )
    result = ml_loop.status_loop(args)
    assert result == 0

    output = capsys.readouterr().out
    assert "ML Loop Status: status-human" in output
    assert "Latest vs Previous (latest - previous):" in output
    assert "pass_prob_mean" in output
    assert "+0.0800" in output
    assert "full_dps_mae" in output
    assert "-0.2000" in output
    assert "Recent iterations:" in output
    assert "Best pass_probability.mean so far:" in output
    assert "Last generation: status=completed" in output
    assert "verified=3/5" in output
    assert "reason=N/A" in output
    assert "run_status" in output
    assert "verified/attempted" in output


def test_status_loop_human_reports_classifier_skip_reason_and_best_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    loop_id = "status-skip-human"
    data_path = tmp_path / "data"
    loop_root = data_path / "ml_loops" / loop_id
    loop_root.mkdir(parents=True, exist_ok=True)
    state_path = loop_root / ml_loop.STATE_FILENAME
    state = ml_loop._build_initial_state(
        loop_id,
        total_iterations=None,
        seed_start_base=1,
        seed_window_size=5,
    )
    state.update(
        {
            "status": "running",
            "phase": "idle",
            "iteration": 2,
            "last_run_id": f"{loop_id}-iter-0002",
            "last_snapshot_id": "iter-0002",
            "last_model_id": f"{loop_id}-iter-0002",
            "last_model_path": str(loop_root / "models" / f"{loop_id}-iter-0002" / "model.json"),
        }
    )
    ml_loop._write_state(state_path, state)

    iterations_path = loop_root / ml_loop.ITERATIONS_FILENAME
    record_1 = {
        "iteration": 1,
        "run_status": "completed",
        "generation": {
            "status": "completed",
            "status_reason_code": None,
            "status_reason_message": None,
            "attempted": 4,
            "successes": 3,
            "failures": 0,
            "errors": 0,
        },
        "model": {
            "model_id": f"{loop_id}-iter-0001",
            "promoted": True,
            "compute_backend_resolved": "cuda",
            "token_learner_backend": "torch_sparse_sgd",
        },
        "evaluation": {
            "current": {
                "metric_mae": {"full_dps": 1.0, "max_hit": 0.5},
                "pass_probability": {"mean": 0.42},
            },
            "previous": None,
            "improvement": {"improved": False},
        },
    }
    record_2 = {
        "iteration": 2,
        "run_status": "completed",
        "generation": {
            "status": "completed",
            "status_reason_code": None,
            "status_reason_message": None,
            "attempted": 4,
            "successes": 3,
            "failures": 0,
            "errors": 0,
        },
        "model": {
            "model_id": f"{loop_id}-iter-0002",
            "promoted": True,
            "compute_backend_resolved": "cuda",
            "token_learner_backend": "torch_sparse_sgd",
        },
        "evaluation": {
            "current": {
                "metric_mae": {"full_dps": 0.8, "max_hit": 0.4},
                "pass_probability": {"mean": 0.55},
                "classifier_metrics": {
                    "brier": 0.12,
                    "labeled_count": 3,
                    "positive_count": 0,
                    "negative_count": 3,
                },
            },
            "previous": {
                "metric_mae": {"full_dps": 1.0, "max_hit": 0.5},
                "pass_probability": {"mean": 0.42},
                "classifier_metrics": {"brier": 0.18},
            },
            "improvement": {"improved": True},
        },
    }
    iterations_path.write_text(
        f"{json.dumps(record_1)}\n{json.dumps(record_2)}\n",
        encoding="utf-8",
    )

    args = argparse.Namespace(
        loop_id=loop_id,
        data_path=data_path,
        output_format="human",
        history=5,
    )
    result = ml_loop.status_loop(args)
    assert result == 0

    output = capsys.readouterr().out
    assert "pass_prob_mean" in output
    assert "Current classifier skip reason: single-class gate_pass labels: 0=3, 1=0" in output
    assert "Best pass_probability.mean so far:" in output
    best_line = output.split("Best pass_probability.mean so far:")[-1].splitlines()[0]
    assert "iter=1" in best_line
    assert "N/A" in output


def test_status_loop_json_format_returns_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    loop_id = "status-json"
    data_path = tmp_path / "data"
    loop_root = data_path / "ml_loops" / loop_id
    loop_root.mkdir(parents=True, exist_ok=True)
    state_path = loop_root / ml_loop.STATE_FILENAME
    state = ml_loop._build_initial_state(
        loop_id, total_iterations=5, seed_start_base=1, seed_window_size=5
    )
    state.update({"phase": "idle", "iteration": 3})
    ml_loop._write_state(state_path, state)

    args = argparse.Namespace(loop_id=loop_id, data_path=data_path, output_format="json", history=5)
    result = ml_loop.status_loop(args)
    assert result == 0

    output = capsys.readouterr().out.strip()
    payload = json.loads(output)
    assert payload["loop_id"] == loop_id
    assert payload["iteration"] == 3
    assert payload["phase"] == "idle"


def test_status_loop_human_ignores_malformed_iteration_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    loop_id = "status-malformed"
    data_path = tmp_path / "data"
    loop_root = data_path / "ml_loops" / loop_id
    loop_root.mkdir(parents=True, exist_ok=True)
    state_path = loop_root / ml_loop.STATE_FILENAME
    state = ml_loop._build_initial_state(
        loop_id, total_iterations=None, seed_start_base=1, seed_window_size=5
    )
    ml_loop._write_state(state_path, state)

    valid_record = {
        "iteration": 1,
        "model": {"model_id": "status-malformed-iter-0001", "promoted": False},
        "evaluation": {
            "current": {
                "metric_mae": {"full_dps": 1.0, "max_hit": 0.5},
                "pass_probability": {"mean": 0.33},
            },
            "previous": None,
            "improvement": {"improved": False},
        },
    }
    iterations_path = loop_root / ml_loop.ITERATIONS_FILENAME
    iterations_path.write_text(
        "{not-valid-json}\n" + json.dumps(valid_record) + "\n",
        encoding="utf-8",
    )

    args = argparse.Namespace(
        loop_id=loop_id,
        data_path=data_path,
        output_format="human",
        history=5,
    )
    result = ml_loop.status_loop(args)
    assert result == 0

    output = capsys.readouterr().out
    assert "ML Loop Status: status-malformed" in output
    assert "Recent iterations:" in output
    assert " 1" in output


def test_main_dispatches_status(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = ml_loop.main(["status", "--loop-id", "missing-main", "--data-path", str(tmp_path)])
    assert result == 1
    output = capsys.readouterr().out.strip()
    payload = json.loads(output)
    assert payload["error"] == "loop_not_found"
    assert payload["loop_id"] == "missing-main"
