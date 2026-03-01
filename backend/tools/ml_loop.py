from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from backend.app.settings import settings
from backend.engine.generation.runner import run_generation
from backend.engine.ruleset import (
    DEFAULT_PRICE_SNAPSHOT_ID,
    derive_ruleset_id,
    read_pob_commit,
    scenario_version_from_profile,
)
from backend.engine.surrogate import (
    EvaluationResult,
    SnapshotResult,
    TrainResult,
    build_dataset_snapshot,
    evaluate_predictions,
    load_dataset_rows,
    load_model,
    train,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

LOOP_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
LOOPS_DIR_NAME = "ml_loops"
STATE_FILENAME = "state.json"
ITERATIONS_FILENAME = "iterations.jsonl"
SNAPSHOTS_DIR_NAME = "snapshots"
MODELS_DIR_NAME = "models"
CHECKPOINTS_DIR_NAME = "checkpoints"
STATE_SCHEMA_VERSION = "ml-loop.v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_loop_id(loop_id: str) -> str:
    if not LOOP_ID_PATTERN.fullmatch(loop_id):
        raise ValueError(
            "loop_id must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_' or '-'"
        )
    return loop_id


def _loop_root(data_path: Path, loop_id: str) -> Path:
    base = data_path.resolve()
    loop_root = (base / LOOPS_DIR_NAME / loop_id).resolve()
    if base not in loop_root.parents and loop_root != base:
        raise ValueError("loop_id resolved outside data path")
    return loop_root


def _load_constraints(path: Path | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    try:
        payload = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"unable to read constraints file {path}: {exc}") from exc
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON in constraints file {path}: {exc}") from exc


def _write_state(state_path: Path, payload: Mapping[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        raise FileNotFoundError(f"state file not found for loop at {state_path}")
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("malformed loop state")
    return payload


def _seed_window_and_base(state: Mapping[str, Any], args: argparse.Namespace) -> tuple[int, int]:
    seed_start_base = state.get("seed_start_base")
    if not isinstance(seed_start_base, int):
        seed_start_base = args.seed_start
    seed_window_size = state.get("seed_window_size")
    if not isinstance(seed_window_size, int):
        seed_window_size = args.count
    return seed_start_base, seed_window_size


def _ensure_seed_metadata(state_path: Path, state: dict[str, Any], args: argparse.Namespace) -> None:
    seed_start_base, seed_window_size = _seed_window_and_base(state, args)
    updates: dict[str, Any] = {}
    if state.get("seed_start_base") is None:
        updates["seed_start_base"] = seed_start_base
    if state.get("seed_window_size") is None:
        updates["seed_window_size"] = seed_window_size
    if state.get("next_iteration_seed_start") is None:
        iterations_done = max(0, state.get("iteration", 0))
        updates["next_iteration_seed_start"] = seed_start_base + iterations_done * seed_window_size
    if updates:
        _persist_state(state_path, state, **updates)


def _resolve_iteration_seed_start(
    state: Mapping[str, Any], iteration: int, args: argparse.Namespace
) -> int:
    next_seed = state.get("next_iteration_seed_start")
    if isinstance(next_seed, int):
        return next_seed
    seed_start_base, seed_window_size = _seed_window_and_base(state, args)
    offset = max(0, iteration - 1)
    return seed_start_base + offset * seed_window_size


def _persist_state(state_path: Path, state: dict[str, Any], **updates: Any) -> dict[str, Any]:
    if updates:
        state.update(updates)
    state["updated_at_utc"] = _now()
    _write_state(state_path, state)
    return state


def _build_initial_state(
    loop_id: str,
    total_iterations: int | None,
    seed_start_base: int,
    seed_window_size: int,
) -> dict[str, Any]:
    now = _now()
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "loop_id": loop_id,
        "status": "running",
        "phase": "initialized",
        "iteration": 0,
        "total_iterations": total_iterations,
        "stop_requested": False,
        "started_at_utc": now,
        "updated_at_utc": now,
        "last_error": None,
        "last_run_id": None,
        "last_snapshot_id": None,
        "last_model_id": None,
        "last_model_path": None,
        "last_improvement": None,
        "seed_start_base": seed_start_base,
        "seed_window_size": seed_window_size,
        "last_iteration_seed_start": None,
        "next_iteration_seed_start": seed_start_base,
        "failed_iteration": None,
        "failed_phase": None,
        "failed_at_utc": None,
        "last_failure_checkpoint_path": None,
    }

def _append_iteration_record(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _count_completed_iterations(iterations_path: Path) -> int:
    if not iterations_path.exists():
        return 0
    count = 0
    for line in iterations_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            count += 1
    return count


def _failure_checkpoint_path(loop_root: Path, iteration: int) -> Path:
    name = f"iter-{iteration:04d}.failure.json"
    return loop_root / CHECKPOINTS_DIR_NAME / name


def _write_failure_checkpoint(
    loop_root: Path,
    iteration: int,
    phase: str,
    error: str,
    seed_start: int | None,
    run_id: str | None,
    state: Mapping[str, Any],
    timestamp: str,
) -> Path:
    path = _failure_checkpoint_path(loop_root, iteration)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "iteration": iteration,
        "phase": phase,
        "error": error,
        "seed_start": seed_start,
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "state_context": {
            "last_run_id": state.get("last_run_id"),
            "last_snapshot_id": state.get("last_snapshot_id"),
            "last_model_id": state.get("last_model_id"),
            "last_model_path": state.get("last_model_path"),
            "last_iteration_seed_start": state.get("last_iteration_seed_start"),
            "next_iteration_seed_start": state.get("next_iteration_seed_start"),
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path

def _iteration_checkpoint_path(loop_root: Path, iteration: int) -> Path:
    name = f"iter-{iteration:04d}.json"
    return loop_root / CHECKPOINTS_DIR_NAME / name


def _compute_improvement(
    current: EvaluationResult, previous: EvaluationResult | None
) -> dict[str, Any]:
    if previous is None:
        return {
            "improved": False,
            "metric_mae_deltas": {},
            "pass_probability_mean_delta": None,
        }
    deltas: dict[str, float] = {}
    for metric, current_value in current.metric_mae.items():
        previous_value = previous.metric_mae.get(metric)
        if previous_value is None:
            continue
        deltas[metric] = previous_value - current_value
    prev_pass_mean = previous.pass_probability.get("mean", 0.0)
    current_pass_mean = current.pass_probability.get("mean", 0.0)
    pass_delta = current_pass_mean - prev_pass_mean
    metrics_improved = all(value >= 0.0 for value in deltas.values()) if deltas else True
    improved = metrics_improved and pass_delta >= 0.0
    return {
        "improved": improved,
        "metric_mae_deltas": deltas,
        "pass_probability_mean_delta": pass_delta,
    }


def _build_iteration_record(
    iteration: int,
    timestamp: str,
    run_summary: Mapping[str, Any],
    snapshot: SnapshotResult,
    train_result: TrainResult,
    evaluation: EvaluationResult,
    previous_evaluation: EvaluationResult | None,
    improvement: Mapping[str, Any],
    promoted: bool,
    promoted_model_id: str,
    promoted_model_path: str,
) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "timestamp_utc": timestamp,
        "run_id": run_summary.get("run_id"),
        "run_status": run_summary.get("status"),
        "snapshot": {
            "snapshot_id": snapshot.snapshot_id,
            "dataset_path": str(snapshot.dataset_path),
            "manifest_path": str(snapshot.manifest_path),
            "row_count": snapshot.row_count,
            "dataset_hash": snapshot.dataset_hash,
            "feature_schema_version": snapshot.feature_schema_version,
        },
        "model": {
            "model_id": train_result.model_id,
            "model_path": str(train_result.model_path),
            "metrics_path": str(train_result.metrics_path),
            "meta_path": str(train_result.meta_path),
            "dataset_snapshot_id": train_result.dataset_snapshot_id,
            "row_count": train_result.row_count,
            "feature_schema_version": train_result.feature_schema_version,
            "compute_backend_preference": train_result.compute_backend_preference,
            "compute_backend_resolved": train_result.compute_backend_resolved,
            "compute_backend_fallback_reason": train_result.compute_backend_fallback_reason,
            "token_learner_backend": train_result.token_learner_backend,
            "token_learner_fallback_reason": train_result.token_learner_fallback_reason,
            "promoted": promoted,
            "promoted_model_id": promoted_model_id,
            "promoted_model_path": promoted_model_path,
        },
        "evaluation": {
            "current": {
                "row_count": evaluation.row_count,
                "metric_mae": evaluation.metric_mae,
                "pass_probability": evaluation.pass_probability,
            },
            "previous": (
                {
                    "row_count": previous_evaluation.row_count,
                    "metric_mae": previous_evaluation.metric_mae,
                    "pass_probability": previous_evaluation.pass_probability,
                }
                if previous_evaluation
                else None
            ),
            "improvement": improvement,
        },
    }


def _build_summary_payload(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "loop_id": state.get("loop_id"),
        "status": state.get("status"),
        "phase": state.get("phase"),
        "iteration": state.get("iteration"),
        "total_iterations": state.get("total_iterations"),
        "stop_requested": state.get("stop_requested"),
        "last_run_id": state.get("last_run_id"),
        "last_snapshot_id": state.get("last_snapshot_id"),
        "last_model_path": state.get("last_model_path"),
        "last_improvement": state.get("last_improvement"),
        "last_error": state.get("last_error"),
    }


def _resolve_ruleset_id(args: argparse.Namespace) -> str:
    if args.ruleset_id:
        return args.ruleset_id
    pob_commit = args.pob_commit or read_pob_commit()
    scenario_version = args.scenario_version or scenario_version_from_profile(args.profile_id)
    return derive_ruleset_id(
        pob_commit=pob_commit,
        scenario_version=scenario_version,
        price_snapshot_id=args.price_snapshot_id,
    )


def start_loop(args: argparse.Namespace) -> int:
    loop_id = _validate_loop_id(args.loop_id)
    if args.iterations < 0:
        raise ValueError("iterations must be non-negative")
    endless_mode = args.iterations == 0
    total_iterations = None if endless_mode else args.iterations
    data_root = Path(args.data_path)
    data_root.mkdir(parents=True, exist_ok=True)
    loop_root = _loop_root(data_root, loop_id)
    loop_root.mkdir(parents=True, exist_ok=True)
    snapshots_root = loop_root / SNAPSHOTS_DIR_NAME
    models_root = loop_root / MODELS_DIR_NAME
    checkpoints_root = loop_root / CHECKPOINTS_DIR_NAME
    artifacts_root = data_root / "data"
    snapshots_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    iterations_path = loop_root / ITERATIONS_FILENAME
    if not iterations_path.exists():
        iterations_path.write_text("", encoding="utf-8")
    state_path = loop_root / STATE_FILENAME

    constraints = _load_constraints(args.constraints_file)
    ruleset_id = _resolve_ruleset_id(args)

    completed_iterations = _count_completed_iterations(iterations_path)
    start_iteration = completed_iterations + 1
    state_exists = state_path.exists()
    if state_exists:
        state = _load_state(state_path)
    else:
        state = _build_initial_state(loop_id, total_iterations, args.seed_start, args.count)
        _write_state(state_path, state)

    _ensure_seed_metadata(state_path, state, args)
    state = _load_state(state_path)

    if total_iterations is not None and total_iterations <= completed_iterations:
        _persist_state(
            state_path,
            state,
            status="completed",
            phase="idle",
            iteration=completed_iterations,
            total_iterations=total_iterations,
            stop_requested=False,
            last_error=None,
        )
        summary_state = _load_state(state_path)
        print(json.dumps(_build_summary_payload(summary_state)))
        return 0

    if state_exists:
        _persist_state(
            state_path,
            state,
            status="running",
            phase="initialized",
            iteration=completed_iterations,
            total_iterations=total_iterations,
            stop_requested=False,
            last_error=None,
        )

    stop_triggered = False
    current_iteration = 0
    iteration = start_iteration
    current_run_id: str | None = None
    iteration_seed_start: int | None = None
    try:
        while True:
            if total_iterations is not None and iteration > total_iterations:
                break
            current_iteration = iteration
            state = _load_state(state_path)
            if state.get("stop_requested"):
                stop_triggered = True
                _persist_state(state_path, state, status="stopped", phase="stopped")
                break
            iteration_seed_start = _resolve_iteration_seed_start(state, iteration, args)
            _, seed_window_size = _seed_window_and_base(state, args)
            next_seed_start = iteration_seed_start + seed_window_size
            _persist_state(
                state_path,
                state,
                phase="generation",
                iteration=iteration,
                last_iteration_seed_start=iteration_seed_start,
                next_iteration_seed_start=next_seed_start,
            )
            current_run_id = f"{loop_id}-iter-{iteration:04d}"
            run_summary = run_generation(
                count=args.count,
                seed_start=iteration_seed_start,
                ruleset_id=ruleset_id,
                profile_id=args.profile_id,
                run_id=current_run_id,
                base_path=data_root,
                constraints=constraints,
                run_mode="optimizer",
            )
            _persist_state(
                state_path,
                state,
                phase="snapshot",
                last_run_id=run_summary.get("run_id"),
            )
            snapshot_id = f"iter-{iteration:04d}"
            snapshot = build_dataset_snapshot(
                data_path=artifacts_root,
                output_root=snapshots_root,
                snapshot_id=snapshot_id,
                exclude_stub_rows=True,
            )
            _persist_state(
                state_path,
                state,
                phase="train",
                last_snapshot_id=snapshot.snapshot_id,
            )
            snapshot_root = snapshot.dataset_path.parent
            if snapshot.row_count <= 0:
                raise ValueError(
                    f"snapshot {snapshot.snapshot_id} has no rows; no verified builds available for training"
                )
            previous_model_path = state.get("last_model_path")
            model_id = f"{loop_id}-iter-{iteration:04d}"
            train_result = train(
                dataset_path=snapshot_root,
                output_root=models_root,
                model_id=model_id,
                compute_backend=args.surrogate_backend,
            )
            current_model_path = str(train_result.model_path)
            _persist_state(
                state_path,
                state,
                phase="evaluate",
            )
            rows = load_dataset_rows(snapshot_root)
            current_model = load_model(train_result.model_path)
            current_predictions = current_model.predict_many(rows)
            current_evaluation = evaluate_predictions(rows, current_predictions)
            previous_evaluation: EvaluationResult | None = None
            previous_model_id = state.get("last_model_id")
            if previous_model_path:
                previous_model = load_model(previous_model_path)
                previous_predictions = previous_model.predict_many(rows)
                previous_evaluation = evaluate_predictions(rows, previous_predictions)
            improvement = _compute_improvement(current_evaluation, previous_evaluation)
            promoted = previous_model_path is None or bool(improvement.get("improved"))
            promoted_model_path = current_model_path if promoted else str(previous_model_path)
            if promoted:
                promoted_model_id = train_result.model_id
            elif previous_model_id:
                promoted_model_id = str(previous_model_id)
            else:
                promoted_model_id = Path(promoted_model_path).parent.name
            timestamp = _now()
            record = _build_iteration_record(
                iteration=iteration,
                timestamp=timestamp,
                run_summary=run_summary,
                snapshot=snapshot,
                train_result=train_result,
                evaluation=current_evaluation,
                previous_evaluation=previous_evaluation,
                improvement=improvement,
                promoted=promoted,
                promoted_model_id=promoted_model_id,
                promoted_model_path=promoted_model_path,
            )
            _append_iteration_record(iterations_path, record)
            checkpoint_path = _iteration_checkpoint_path(loop_root, iteration)
            checkpoint_path.write_text(json.dumps(record), encoding="utf-8")
            print(json.dumps(record))
            _persist_state(
                state_path,
                state,
                phase="idle",
                iteration=iteration,
                last_model_id=promoted_model_id,
                last_model_path=promoted_model_path,
                last_improvement=improvement,
                last_error=None,
            )
            iteration += 1
        if not stop_triggered and total_iterations is not None:
            final_state = _load_state(state_path)
            _persist_state(
                state_path,
                final_state,
                status="completed",
                phase="idle",
                total_iterations=total_iterations,
            )
        summary_state = _load_state(state_path)
        print(json.dumps(_build_summary_payload(summary_state)))
        return 0
    except Exception as exc:
        logger.exception("ml loop %s failed at iteration %s", loop_id, current_iteration)
        failure_timestamp = _now()
        failure_phase = "failed"
        failure_state: dict[str, Any]
        if state_path.exists():
            failure_state = _load_state(state_path)
            failure_phase = failure_state.get("phase") or failure_phase
        else:
            failure_state = _build_initial_state(loop_id, total_iterations, args.seed_start, args.count)
            _write_state(state_path, failure_state)
        failure_checkpoint = _write_failure_checkpoint(
            loop_root=loop_root,
            iteration=current_iteration,
            phase=failure_phase,
            error=str(exc),
            seed_start=iteration_seed_start,
            run_id=current_run_id,
            state=failure_state,
            timestamp=failure_timestamp,
        )
        _persist_state(
            state_path,
            failure_state,
            status="failed",
            phase=failure_phase,
            last_error=str(exc),
            failed_iteration=current_iteration,
            failed_phase=failure_phase,
            failed_at_utc=failure_timestamp,
            last_failure_checkpoint_path=str(failure_checkpoint),
        )
        summary_state = _load_state(state_path)
        print(json.dumps(_build_summary_payload(summary_state)))
        return 1


def stop_loop(args: argparse.Namespace) -> int:
    loop_id = _validate_loop_id(args.loop_id)
    data_root = Path(args.data_path)
    data_root.mkdir(parents=True, exist_ok=True)
    loop_root = _loop_root(data_root, loop_id)
    state_path = loop_root / STATE_FILENAME
    if not state_path.exists():
        print(json.dumps({"error": "loop_not_found", "loop_id": loop_id}))
        return 1
    state = _load_state(state_path)
    _persist_state(
        state_path,
        state,
        stop_requested=True,
        phase="stop_requested",
    )
    print(json.dumps({"loop_id": loop_id, "stop_requested": True}))
    return 0


def status_loop(args: argparse.Namespace) -> int:
    loop_id = _validate_loop_id(args.loop_id)
    data_root = Path(args.data_path)
    data_root.mkdir(parents=True, exist_ok=True)
    loop_root = _loop_root(data_root, loop_id)
    state_path = loop_root / STATE_FILENAME
    if not state_path.exists():
        print(json.dumps({"error": "loop_not_found", "loop_id": loop_id}))
        return 1
    state = _load_state(state_path)
    print(json.dumps(state, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ML loop controller")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start", help="Start an ML loop")
    start_parser.add_argument("--loop-id", required=True, help="User-facing loop identifier")
    start_parser.add_argument(
        "--iterations",
        type=int,
        required=True,
        help="Number of generation/train/eval iterations to run (0 for endless mode)",
    )
    start_parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of candidates to generate per iteration",
    )
    start_parser.add_argument(
        "--seed-start",
        type=int,
        default=1,
        help="Deterministic seed offset for generation",
    )
    start_parser.add_argument("--profile-id", default="pinnacle", help="Scenario profile")
    start_parser.add_argument("--ruleset-id", default=None, help="Explicit ruleset id")
    start_parser.add_argument(
        "--scenario-version",
        default=None,
        help="Optional scenario version when deriving ruleset",
    )
    start_parser.add_argument(
        "--price-snapshot-id",
        default=DEFAULT_PRICE_SNAPSHOT_ID,
        help="Price snapshot identifier",
    )
    start_parser.add_argument(
        "--pob-commit",
        default=None,
        help="Optional PoB commit hash for ruleset derivation",
    )
    start_parser.add_argument(
        "--constraints-file",
        type=Path,
        default=None,
        help="JSON payload of generation constraints",
    )
    start_parser.add_argument(
        "--data-path",
        type=Path,
        default=settings.data_path,
        help="Root directory for artifacts and loop state",
    )
    start_parser.add_argument(
        "--surrogate-backend",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="compute backend preference for surrogate training",
    )

    stop_parser = subparsers.add_parser("stop", help="Request an ML loop to stop")
    stop_parser.add_argument("--loop-id", required=True, help="Loop identifier")
    stop_parser.add_argument(
        "--data-path",
        type=Path,
        default=settings.data_path,
        help="Root directory for artifacts and loop state",
    )

    status_parser = subparsers.add_parser("status", help="Show ML loop state")
    status_parser.add_argument("--loop-id", required=True, help="Loop identifier")
    status_parser.add_argument(
        "--data-path",
        type=Path,
        default=settings.data_path,
        help="Root directory for artifacts and loop state",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "start":
        return start_loop(args)
    if args.command == "stop":
        return stop_loop(args)
    return status_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
