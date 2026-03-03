from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Mapping, Sequence

from backend.app.settings import settings
from backend.engine.generation.runner import run_generation
from backend.engine.ruleset import (
    DEFAULT_PRICE_SNAPSHOT_ID,
    derive_ruleset_id,
    read_pob_commit,
    scenario_version_from_profile,
)
from backend.engine.scenarios.loader import list_templates
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
from backend.engine.surrogate.model import METRIC_TARGETS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

IMPROVEMENT_EPSILON = 1e-6
PROMOTION_CLASSIFIER_BRIER_WEIGHT = 0.05
VAL_SPLIT_MOD = 5
VAL_SPLIT_REMAINDER = 0


def _log_loop_phase(
    loop_id: str,
    phase: str,
    *,
    iteration: int | None = None,
    detail: str | None = None,
) -> None:
    message = f"ml loop {loop_id} phase={phase}"
    if iteration is not None:
        message = f"ml loop {loop_id} iteration={iteration} phase={phase}"
    if detail:
        message = f"{message} {detail}"
    if logger.isEnabledFor(logging.INFO):
        logger.info(message)
        return
    logger.warning(message)


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


def _resolve_evaluation_budget(args: argparse.Namespace) -> int:
    budget = _coerce_int(getattr(args, "count", 1))
    if budget is None:
        return 1
    return max(1, budget)


def _resolve_pool_multiplier(args: argparse.Namespace) -> int:
    multiplier = _coerce_int(getattr(args, "pool_multiplier", 4))
    if multiplier is None:
        return 4
    return max(1, multiplier)


def _resolve_generate_count(args: argparse.Namespace) -> int:
    generated = _coerce_int(getattr(args, "generate_count", None))
    if generated is not None and generated > 0:
        return generated
    return _resolve_evaluation_budget(args) * _resolve_pool_multiplier(args)


def _seed_window_and_base(state: Mapping[str, Any], args: argparse.Namespace) -> tuple[int, int]:
    seed_start_base = state.get("seed_start_base")
    if not isinstance(seed_start_base, int):
        seed_start_base = args.seed_start
    seed_window_size = state.get("seed_window_size")
    if not isinstance(seed_window_size, int) or seed_window_size <= 0:
        seed_window_size = _resolve_generate_count(args)
    return seed_start_base, seed_window_size


def _ensure_seed_metadata(
    state_path: Path, state: dict[str, Any], args: argparse.Namespace
) -> None:
    seed_start_base = state.get("seed_start_base")
    if not isinstance(seed_start_base, int):
        seed_start_base = args.seed_start
    seed_window_size = _resolve_generate_count(args)
    updates: dict[str, Any] = {}
    if state.get("seed_start_base") != seed_start_base:
        updates["seed_start_base"] = seed_start_base
    if state.get("seed_window_size") != seed_window_size:
        updates["seed_window_size"] = seed_window_size
    if state.get("next_iteration_seed_start") is None:
        iterations_done = max(0, _coerce_int(state.get("iteration")) or 0)
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


def _stable_hash_mod(value: str, mod: int) -> int:
    if mod <= 0:
        return 0
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest, 16) % mod


def _validation_split_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    train_rows: list[Mapping[str, Any]] = []
    val_rows: list[Mapping[str, Any]] = []
    for row in rows:
        build_id = str(row.get("build_id") or "")
        if not build_id:
            build_id = (
                f"{row.get('scenario_id') or ''}|"
                f"{row.get('profile_id') or ''}|"
                f"{row.get('main_skill_package') or ''}"
            )
        if _stable_hash_mod(build_id, VAL_SPLIT_MOD) == VAL_SPLIT_REMAINDER:
            val_rows.append(row)
        else:
            train_rows.append(row)
    if not val_rows and rows:
        val_rows = [rows[0]]
        train_rows = list(rows[1:]) or list(rows)
    if not train_rows and rows:
        train_rows = list(rows)
    return train_rows, val_rows


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
        "last_generation_status": None,
        "last_generation_status_reason_code": None,
        "last_generation_status_reason_message": None,
        "last_generation_attempted": 0,
        "last_generation_verified": 0,
        "last_generation_failures": 0,
        "last_generation_errors": 0,
        "last_iteration_outcome": None,
        "skipped_iterations_total": 0,
        "last_skipped_iteration": None,
        "last_skip_reason_code": None,
        "last_skip_reason_message": None,
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
    current_score = _promotion_score(current)
    current_skip_reason = _classifier_skip_reason_from_evaluation_result(current)
    if previous is None:
        return {
            "improved": False,
            "metric_mae_deltas": {},
            "pass_probability_mean_delta": None,
            "promotion_score_current": current_score.get("total"),
            "promotion_score_previous": None,
            "promotion_score_delta": None,
            "promotion_score_components": {"current": current_score, "previous": None},
            "current_classifier_skip_reason": current_skip_reason,
            "previous_classifier_skip_reason": None,
        }
    previous_score = _promotion_score(previous)
    deltas: dict[str, float] = {}
    for metric, current_value in current.metric_mae.items():
        previous_value = previous.metric_mae.get(metric)
        if previous_value is None:
            continue
        delta = previous_value - current_value
        deltas[metric] = delta
    prev_pass_mean = _numeric(previous.pass_probability.get("mean"))
    current_pass_mean = _numeric(current.pass_probability.get("mean"))
    previous_skip_reason = _classifier_skip_reason_from_evaluation_result(previous)
    pass_delta: float | None = None
    if (
        prev_pass_mean is not None
        and current_pass_mean is not None
        and not previous_skip_reason
        and not current_skip_reason
    ):
        pass_delta = current_pass_mean - prev_pass_mean
    current_total = _numeric(current_score.get("total"))
    previous_total = _numeric(previous_score.get("total"))
    score_delta = (
        previous_total - current_total
        if previous_total is not None and current_total is not None
        else None
    )
    improved = bool(score_delta is not None and score_delta > IMPROVEMENT_EPSILON)
    return {
        "improved": improved,
        "metric_mae_deltas": deltas,
        "pass_probability_mean_delta": pass_delta,
        "promotion_score_current": current_total,
        "promotion_score_previous": previous_total,
        "promotion_score_delta": score_delta,
        "promotion_score_components": {
            "current": current_score,
            "previous": previous_score,
        },
        "current_classifier_skip_reason": current_skip_reason,
        "previous_classifier_skip_reason": previous_skip_reason,
    }


def _promotion_score(evaluation: EvaluationResult) -> dict[str, float | None]:
    log1p_pass = (
        dict(evaluation.metric_mae_log1p_pass)
        if evaluation.metric_mae_log1p_pass
        else dict(evaluation.metric_mae_log1p)
    )
    full_dps = _numeric(log1p_pass.get("full_dps"))
    max_hit = _numeric(log1p_pass.get("max_hit"))
    regression_terms = [value for value in (full_dps, max_hit) if value is not None]
    regression_component = (
        sum(regression_terms) / len(regression_terms) if regression_terms else None
    )
    classifier_metrics = dict(evaluation.classifier_metrics)
    brier = _numeric(classifier_metrics.get("brier"))
    classifier_penalty = brier * PROMOTION_CLASSIFIER_BRIER_WEIGHT if brier is not None else 0.0
    total = regression_component + classifier_penalty if regression_component is not None else None
    return {
        "regression_log1p_pass": regression_component,
        "classifier_brier": brier,
        "classifier_penalty": classifier_penalty,
        "total": total,
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
    run_status_reason = _as_dict(run_summary.get("status_reason"))
    evaluation_info = _as_dict(run_summary.get("evaluation"))
    current_skip_reason = improvement.get("current_classifier_skip_reason")
    previous_skip_reason = improvement.get("previous_classifier_skip_reason")
    return {
        "iteration": iteration,
        "timestamp_utc": timestamp,
        "run_id": run_summary.get("run_id"),
        "run_status": run_summary.get("status"),
        "iteration_outcome": "completed",
        "generation": {
            "status": run_summary.get("status"),
            "status_reason_code": run_status_reason.get("code"),
            "status_reason_message": run_status_reason.get("message"),
            "attempted": _coerce_int(evaluation_info.get("attempted")),
            "successes": _coerce_int(evaluation_info.get("successes")),
            "failures": _coerce_int(evaluation_info.get("failures")),
            "errors": _coerce_int(evaluation_info.get("errors")),
        },
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
                "metric_mae_all": evaluation.metric_mae_all,
                "metric_mae_pass": evaluation.metric_mae_pass,
                "metric_mae_log1p": evaluation.metric_mae_log1p,
                "metric_mae_log1p_pass": evaluation.metric_mae_log1p_pass,
                "classifier_metrics": evaluation.classifier_metrics,
                "classifier_skip_reason": current_skip_reason,
                "pass_probability": evaluation.pass_probability,
            },
            "previous": (
                {
                    "row_count": previous_evaluation.row_count,
                    "metric_mae": previous_evaluation.metric_mae,
                    "metric_mae_all": previous_evaluation.metric_mae_all,
                    "metric_mae_pass": previous_evaluation.metric_mae_pass,
                    "metric_mae_log1p": previous_evaluation.metric_mae_log1p,
                    "metric_mae_log1p_pass": previous_evaluation.metric_mae_log1p_pass,
                    "classifier_metrics": previous_evaluation.classifier_metrics,
                    "classifier_skip_reason": previous_skip_reason,
                    "pass_probability": previous_evaluation.pass_probability,
                }
                if previous_evaluation
                else None
            ),
            "improvement": improvement,
        },
    }


def _build_generation_skip_record(
    iteration: int,
    timestamp: str,
    run_summary: Mapping[str, Any],
    *,
    reason_code: str,
    reason_message: str,
    iteration_outcome: str = "skipped_generation_unhealthy",
) -> dict[str, Any]:
    run_status_reason = _as_dict(run_summary.get("status_reason"))
    evaluation_info = _as_dict(run_summary.get("evaluation"))
    return {
        "iteration": iteration,
        "timestamp_utc": timestamp,
        "run_id": run_summary.get("run_id"),
        "run_status": run_summary.get("status"),
        "iteration_outcome": iteration_outcome,
        "skip_reason_code": reason_code,
        "skip_reason_message": reason_message,
        "skipped_phases": ["snapshot", "train", "evaluate"],
        "generation": {
            "status": run_summary.get("status"),
            "status_reason_code": run_status_reason.get("code"),
            "status_reason_message": run_status_reason.get("message"),
            "attempted": _coerce_int(evaluation_info.get("attempted")),
            "successes": _coerce_int(evaluation_info.get("successes")),
            "failures": _coerce_int(evaluation_info.get("failures")),
            "errors": _coerce_int(evaluation_info.get("errors")),
        },
        "snapshot": None,
        "model": None,
        "evaluation": None,
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
        "last_iteration_outcome": state.get("last_iteration_outcome"),
        "skipped_iterations_total": state.get("skipped_iterations_total"),
        "last_skipped_iteration": state.get("last_skipped_iteration"),
        "last_skip_reason_code": state.get("last_skip_reason_code"),
        "last_skip_reason_message": state.get("last_skip_reason_message"),
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
    evaluation_budget = _resolve_evaluation_budget(args)
    args.count = evaluation_budget
    pool_multiplier = _resolve_pool_multiplier(args)
    generate_count = evaluation_budget * pool_multiplier
    args.generate_count = generate_count
    default_surrogate_top_k = evaluation_budget
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

    profile_id = str(getattr(args, "profile_id", "pinnacle"))
    scenario_templates = [
        template for template in list_templates() if template.profile_id == profile_id
    ]
    snapshot_scenario_id = (
        scenario_templates[0].scenario_id if len(scenario_templates) == 1 else None
    )
    candidate_pool_size_arg = getattr(args, "candidate_pool_size", None)
    resolved_candidate_pool_size = (
        int(candidate_pool_size_arg)
        if candidate_pool_size_arg and int(candidate_pool_size_arg) > 0
        else generate_count
    )
    surrogate_top_k_arg = getattr(args, "surrogate_top_k", None)
    surrogate_top_k_cli_value = _coerce_int(surrogate_top_k_arg)
    surrogate_top_k_disable_requested = (
        surrogate_top_k_arg is not None
        and surrogate_top_k_cli_value is not None
        and surrogate_top_k_cli_value <= 0
    )
    surrogate_top_k_override = (
        surrogate_top_k_cli_value
        if surrogate_top_k_cli_value is not None and surrogate_top_k_cli_value > 0
        else None
    )
    surrogate_exploration_pct = min(
        0.5,
        max(0.0, float(getattr(args, "surrogate_exploration_pct", 0.10))),
    )
    optimizer_iterations = int(getattr(args, "optimizer_iterations", 3))
    optimizer_elite_count = int(getattr(args, "optimizer_elite_count", 16))

    completed_iterations = _count_completed_iterations(iterations_path)
    start_iteration = completed_iterations + 1
    state_exists = state_path.exists()
    if state_exists:
        state = _load_state(state_path)
    else:
        state = _build_initial_state(loop_id, total_iterations, args.seed_start, generate_count)
        _write_state(state_path, state)

    _ensure_seed_metadata(state_path, state, args)
    state = _load_state(state_path)
    target_text = "endless" if total_iterations is None else str(total_iterations)
    _log_loop_phase(
        loop_id,
        "initialized",
        iteration=completed_iterations,
        detail=f"start_iteration={start_iteration} target_iterations={target_text}",
    )

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
            failed_iteration=None,
            failed_phase=None,
            failed_at_utc=None,
            last_failure_checkpoint_path=None,
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
                _log_loop_phase(loop_id, "stopped", iteration=iteration, detail="stop requested")
                break
            iteration_seed_start = _resolve_iteration_seed_start(state, iteration, args)
            _, seed_window_size = _seed_window_and_base(state, args)
            next_seed_start = iteration_seed_start + seed_window_size
            current_run_id = f"{loop_id}-iter-{iteration:04d}"
            _persist_state(
                state_path,
                state,
                phase="generation",
                iteration=iteration,
                last_iteration_seed_start=iteration_seed_start,
                next_iteration_seed_start=next_seed_start,
            )
            _log_loop_phase(
                loop_id,
                "generation",
                iteration=iteration,
                detail=(
                    f"run_id={current_run_id} seed_start={iteration_seed_start} "
                    f"eval_budget={evaluation_budget} generate_count={generate_count}"
                ),
            )
            generation_started_at = monotonic()
            active_surrogate_model_path = state.get("last_model_path")
            surrogate_top_k_value: int | None = None
            surrogate_predictor = None
            if active_surrogate_model_path:
                if surrogate_top_k_disable_requested:
                    surrogate_top_k_value = None
                elif surrogate_top_k_override is not None:
                    surrogate_top_k_value = surrogate_top_k_override
                else:
                    surrogate_top_k_value = default_surrogate_top_k
                try:
                    surrogate_model = load_model(active_surrogate_model_path)
                except Exception as exc:
                    logger.warning(
                        "ml loop %s iteration %d surrogate model load failed for %s; falling back to path-based loading: %s",
                        loop_id,
                        iteration,
                        active_surrogate_model_path,
                        exc,
                    )
                else:
                    predictor_callable = surrogate_model.predict_many

                    def _surrogate_predictor_wrapper(rows, *, _predictor=predictor_callable):
                        return _predictor(rows)

                    predictor_name = getattr(surrogate_model, "model_id", None)
                    _surrogate_predictor_wrapper.__name__ = (
                        predictor_name if predictor_name else "surrogate_predictor"
                    )
                    surrogate_predictor = _surrogate_predictor_wrapper
            run_summary = run_generation(
                count=generate_count,
                seed_start=iteration_seed_start,
                ruleset_id=ruleset_id,
                profile_id=profile_id,
                run_id=current_run_id,
                base_path=data_root,
                constraints=constraints,
                run_mode="standard",
                surrogate_enabled=bool(active_surrogate_model_path),
                surrogate_model_path=active_surrogate_model_path,
                surrogate_predictor=surrogate_predictor,
                candidate_pool_size=resolved_candidate_pool_size,
                surrogate_top_k=surrogate_top_k_value,
                surrogate_exploration_pct=surrogate_exploration_pct,
                optimizer_iterations=optimizer_iterations,
                optimizer_elite_count=optimizer_elite_count,
            )
            _log_loop_phase(
                loop_id,
                "generation_complete",
                iteration=iteration,
                detail=(
                    f"run_id={run_summary.get('run_id')} status={run_summary.get('status')} "
                    f"verified={run_summary.get('evaluation', {}).get('successes')} "
                    f"elapsed={max(0.0, monotonic() - generation_started_at):.1f}s"
                ),
            )
            status_reason = _as_dict(run_summary.get("status_reason"))
            evaluation_summary = _as_dict(run_summary.get("evaluation"))
            attempted_count = _coerce_int(evaluation_summary.get("attempted")) or 0
            verified_count = _coerce_int(evaluation_summary.get("successes")) or 0
            failures_count = _coerce_int(evaluation_summary.get("failures")) or 0
            errors_count = _coerce_int(evaluation_summary.get("errors")) or 0
            generation_status = run_summary.get("status")
            state = _persist_state(
                state_path,
                state,
                phase="generation",
                last_run_id=run_summary.get("run_id"),
                last_generation_status=generation_status,
                last_generation_status_reason_code=status_reason.get("code"),
                last_generation_status_reason_message=status_reason.get("message"),
                last_generation_attempted=attempted_count,
                last_generation_verified=verified_count,
                last_generation_failures=failures_count,
                last_generation_errors=errors_count,
            )
            if generation_status != "completed" or verified_count <= 0:
                reason_code = status_reason.get("code") or "unknown"
                reason_message = status_reason.get("message") or "unknown"
                failure_message = (
                    f"generation run {run_summary.get('run_id')} failed "
                    f"(status={generation_status} verified={verified_count}/{attempted_count} "
                    f"reason={reason_code}:{reason_message})"
                )
                logger.warning(
                    "ml loop %s iteration %d skipping unhealthy generation result: %s",
                    loop_id,
                    iteration,
                    failure_message,
                )
                timestamp = _now()
                skipped_record = _build_generation_skip_record(
                    iteration=iteration,
                    timestamp=timestamp,
                    run_summary=run_summary,
                    reason_code=reason_code,
                    reason_message=reason_message,
                )
                _append_iteration_record(iterations_path, skipped_record)
                checkpoint_path = _iteration_checkpoint_path(loop_root, iteration)
                checkpoint_path.write_text(json.dumps(skipped_record), encoding="utf-8")
                print(json.dumps(skipped_record))
                skipped_total = (_coerce_int(state.get("skipped_iterations_total")) or 0) + 1
                _persist_state(
                    state_path,
                    state,
                    phase="idle",
                    iteration=iteration,
                    last_error=failure_message,
                    last_iteration_outcome="skipped_generation_unhealthy",
                    skipped_iterations_total=skipped_total,
                    last_skipped_iteration=iteration,
                    last_skip_reason_code=reason_code,
                    last_skip_reason_message=reason_message,
                    failed_iteration=None,
                    failed_phase=None,
                    failed_at_utc=None,
                    last_failure_checkpoint_path=None,
                )
                _log_loop_phase(
                    loop_id,
                    "generation_skipped",
                    iteration=iteration,
                    detail=(
                        f"run_id={run_summary.get('run_id')} status={generation_status} "
                        f"verified={verified_count}/{attempted_count} reason={reason_code}"
                    ),
                )
                iteration += 1
                continue
            _persist_state(
                state_path,
                state,
                phase="snapshot",
                last_run_id=run_summary.get("run_id"),
            )
            snapshot_id = f"iter-{iteration:04d}"
            _log_loop_phase(
                loop_id,
                "snapshot",
                iteration=iteration,
                detail=f"snapshot_id={snapshot_id}",
            )
            snapshot_started_at = monotonic()
            snapshot = build_dataset_snapshot(
                data_path=artifacts_root,
                output_root=snapshots_root,
                snapshot_id=snapshot_id,
                exclude_stub_rows=True,
                profile_id=profile_id,
                scenario_id=snapshot_scenario_id,
            )
            if snapshot.row_count <= 0 and profile_id:
                _log_loop_phase(
                    loop_id,
                    "snapshot_retry",
                    iteration=iteration,
                    detail=(
                        "filtered snapshot empty; retrying without profile filter "
                        f"profile_id={profile_id} scenario_id={snapshot_scenario_id}"
                    ),
                )
                snapshot = build_dataset_snapshot(
                    data_path=artifacts_root,
                    output_root=snapshots_root,
                    snapshot_id=snapshot_id,
                    exclude_stub_rows=True,
                    profile_id=None,
                    scenario_id=snapshot_scenario_id,
                )
            if snapshot.row_count <= 0:
                reason_code = "snapshot_not_trained"
                reason_message = (
                    f"snapshot {snapshot.snapshot_id} filtered to zero verified builds; not trained"
                )
                _log_loop_phase(
                    loop_id,
                    "snapshot_skipped",
                    iteration=iteration,
                    detail=reason_message,
                )
                timestamp = _now()
                skip_record = _build_generation_skip_record(
                    iteration=iteration,
                    timestamp=timestamp,
                    run_summary=run_summary,
                    reason_code=reason_code,
                    reason_message=reason_message,
                    iteration_outcome="skipped_snapshot_empty",
                )
                skip_record["snapshot"] = {
                    "snapshot_id": snapshot.snapshot_id,
                    "dataset_path": str(snapshot.dataset_path),
                    "manifest_path": str(snapshot.manifest_path),
                    "row_count": snapshot.row_count,
                    "dataset_hash": snapshot.dataset_hash,
                    "feature_schema_version": snapshot.feature_schema_version,
                }
                _append_iteration_record(iterations_path, skip_record)
                checkpoint_path = _iteration_checkpoint_path(loop_root, iteration)
                checkpoint_path.write_text(json.dumps(skip_record), encoding="utf-8")
                print(json.dumps(skip_record))
                skipped_total = (_coerce_int(state.get("skipped_iterations_total")) or 0) + 1
                _persist_state(
                    state_path,
                    state,
                    phase="idle",
                    iteration=iteration,
                    last_error=reason_message,
                    last_iteration_outcome="skipped_snapshot_empty",
                    skipped_iterations_total=skipped_total,
                    last_skipped_iteration=iteration,
                    last_skip_reason_code=reason_code,
                    last_skip_reason_message=reason_message,
                    last_snapshot_id=snapshot.snapshot_id,
                    failed_iteration=None,
                    failed_phase=None,
                    failed_at_utc=None,
                    last_failure_checkpoint_path=None,
                )
                iteration += 1
                continue

            _log_loop_phase(
                loop_id,
                "snapshot_complete",
                iteration=iteration,
                detail=(
                    f"snapshot_id={snapshot.snapshot_id} rows={snapshot.row_count} "
                    f"elapsed={max(0.0, monotonic() - snapshot_started_at):.1f}s"
                ),
            )
            _persist_state(
                state_path,
                state,
                phase="train",
                last_snapshot_id=snapshot.snapshot_id,
            )
            snapshot_root = snapshot.dataset_path.parent
            previous_model_path = state.get("last_model_path")
            model_id = f"{loop_id}-iter-{iteration:04d}"
            _log_loop_phase(loop_id, "train", iteration=iteration, detail=f"model_id={model_id}")
            train_started_at = monotonic()
            train_result = train(
                dataset_path=snapshot_root,
                output_root=models_root,
                model_id=model_id,
                compute_backend=args.surrogate_backend,
                split_mod=VAL_SPLIT_MOD,
                split_remainder=VAL_SPLIT_REMAINDER,
            )
            current_model_path = str(train_result.model_path)
            _log_loop_phase(
                loop_id,
                "train_complete",
                iteration=iteration,
                detail=(
                    f"model_id={train_result.model_id} "
                    f"backend={train_result.compute_backend_resolved} "
                    f"elapsed={max(0.0, monotonic() - train_started_at):.1f}s"
                ),
            )
            _persist_state(
                state_path,
                state,
                phase="evaluate",
            )
            _log_loop_phase(loop_id, "evaluate", iteration=iteration)
            evaluate_started_at = monotonic()
            rows = load_dataset_rows(snapshot_root)
            _, validation_rows = _validation_split_rows(rows)
            evaluation_rows = validation_rows if validation_rows else rows
            current_model = load_model(train_result.model_path)
            current_predictions = current_model.predict_many(evaluation_rows)
            current_evaluation = evaluate_predictions(evaluation_rows, current_predictions)
            previous_evaluation: EvaluationResult | None = None
            previous_model_id = state.get("last_model_id")
            if previous_model_path:
                previous_model = load_model(previous_model_path)
                previous_predictions = previous_model.predict_many(evaluation_rows)
                previous_evaluation = evaluate_predictions(evaluation_rows, previous_predictions)
            improvement = _compute_improvement(current_evaluation, previous_evaluation)
            promoted = previous_model_path is None or bool(improvement.get("improved"))
            promoted_model_path = current_model_path if promoted else str(previous_model_path)
            if promoted:
                promoted_model_id = train_result.model_id
            elif previous_model_id:
                promoted_model_id = str(previous_model_id)
            else:
                promoted_model_id = Path(promoted_model_path).parent.name
            _log_loop_phase(
                loop_id,
                "evaluate_complete",
                iteration=iteration,
                detail=(
                    f"promoted={promoted} promoted_model_id={promoted_model_id} "
                    f"rows={current_evaluation.row_count} "
                    f"elapsed={max(0.0, monotonic() - evaluate_started_at):.1f}s"
                ),
            )
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
                last_iteration_outcome="completed",
                last_skip_reason_code=None,
                last_skip_reason_message=None,
                failed_iteration=None,
                failed_phase=None,
                failed_at_utc=None,
                last_failure_checkpoint_path=None,
            )
            _log_loop_phase(loop_id, "idle", iteration=iteration, detail="iteration complete")
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
            _log_loop_phase(
                loop_id,
                "completed",
                iteration=total_iterations,
                detail="requested iterations finished",
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
            failure_state = _build_initial_state(
                loop_id,
                total_iterations,
                args.seed_start,
                _resolve_generate_count(args),
            )
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


def _load_iteration_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload_text = line.strip()
        if not payload_text:
            continue
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _format_float(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def _format_delta(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.4f}"


def _format_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return "N/A"


def _format_count(value: Any) -> str:
    coerced = _coerce_int(value)
    return str(coerced) if coerced is not None else "N/A"


def _format_verified_ratio(successes: Any, attempted: Any) -> str:
    success = _coerce_int(successes)
    attempt = _coerce_int(attempted)
    if success is None or attempt is None:
        return "N/A"
    return f"{success}/{attempt}"


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _trimmed_string(value: Any) -> str | None:
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed:
            return trimmed
    return None


def _classifier_skip_reason_from_metrics(metrics: Mapping[str, Any] | None) -> str | None:
    candidate = _as_dict(metrics)
    reason = _trimmed_string(candidate.get("classifier_skip_reason"))
    if reason:
        return reason
    reason = _trimmed_string(candidate.get("skip_reason"))
    if reason:
        return reason
    labeled = _coerce_int(candidate.get("labeled_count"))
    if labeled is None or labeled <= 0:
        return None
    positive = _coerce_int(candidate.get("positive_count"))
    negative = _coerce_int(candidate.get("negative_count"))
    if positive is None or negative is None:
        return None
    if positive <= 0 or negative <= 0:
        return f"single-class gate_pass labels: 0={negative}, 1={positive}"
    return None


def _classifier_skip_reason_from_payload(payload: Mapping[str, Any] | None) -> str | None:
    if payload is None:
        return None
    reason = _trimmed_string(payload.get("classifier_skip_reason"))
    if reason:
        return reason
    reason = _trimmed_string(payload.get("classifier_skip_reason_message"))
    if reason:
        return reason
    reason = _trimmed_string(payload.get("skip_reason"))
    if reason:
        return reason
    return _classifier_skip_reason_from_metrics(payload.get("classifier_metrics"))


def _classifier_skip_reason_from_evaluation_result(
    evaluation: EvaluationResult | None,
) -> str | None:
    if evaluation is None:
        return None
    return _classifier_skip_reason_from_metrics(dict(evaluation.classifier_metrics))


def _record_pass_mean(record: Mapping[str, Any]) -> float | None:
    evaluation = _as_dict(record.get("evaluation"))
    current = _as_dict(evaluation.get("current"))
    if _classifier_skip_reason_from_payload(current):
        return None
    pass_probability = _as_dict(current.get("pass_probability"))
    return _numeric(pass_probability.get("mean"))


def _render_status_human(
    loop_id: str,
    state: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    history: int,
) -> str:
    status = state.get("status", "unknown")
    phase = state.get("phase", "unknown")
    iteration = state.get("iteration")
    total_iterations = state.get("total_iterations")
    total_text = "endless" if total_iterations is None else str(total_iterations)
    summary_lines = [
        f"ML Loop Status: {loop_id}",
        (
            "State: "
            f"{status} (phase={phase}) | iteration={iteration}/{total_text} "
            f"| stop_requested={state.get('stop_requested')}"
        ),
        f"Started: {state.get('started_at_utc')}",
        f"Updated: {state.get('updated_at_utc')}",
        (
            "Last IDs: "
            f"run={state.get('last_run_id')} snapshot={state.get('last_snapshot_id')} "
            f"model={state.get('last_model_id')}"
        ),
    ]
    last_generation_status = state.get("last_generation_status") or "unknown"
    last_generation_verified = _coerce_int(state.get("last_generation_verified"))
    last_generation_attempted = _coerce_int(state.get("last_generation_attempted"))
    last_generation_ratio = _format_verified_ratio(
        last_generation_verified, last_generation_attempted
    )
    last_generation_failures = _coerce_int(state.get("last_generation_failures"))
    last_generation_errors = _coerce_int(state.get("last_generation_errors"))
    last_generation_reason = state.get("last_generation_status_reason_code") or "N/A"
    summary_lines.append(
        (
            "Last generation: "
            f"status={last_generation_status} "
            f"verified={last_generation_ratio} "
            f"failures={_format_count(last_generation_failures)} "
            f"errors={_format_count(last_generation_errors)} "
            f"reason={last_generation_reason}"
        )
    )
    reason_message = state.get("last_generation_status_reason_message")
    if reason_message:
        summary_lines.append(f"Last generation reason: {reason_message}")

    last_error = state.get("last_error")
    if last_error:
        summary_lines.append(f"Last error: {last_error}")

    if not records:
        summary_lines.append("No iteration records found yet.")
        return "\n".join(summary_lines)

    latest = records[-1]
    latest_iteration = latest.get("iteration", "?")
    latest_model = _as_dict(latest.get("model"))
    latest_eval = _as_dict(latest.get("evaluation"))
    current_eval = _as_dict(latest_eval.get("current"))
    previous_eval = _as_dict(latest_eval.get("previous"))
    current_metric_mae = _as_dict(current_eval.get("metric_mae"))
    previous_metric_mae = _as_dict(previous_eval.get("metric_mae"))
    current_pass = _as_dict(current_eval.get("pass_probability"))
    previous_pass = _as_dict(previous_eval.get("pass_probability"))
    current_skip_reason = _classifier_skip_reason_from_payload(current_eval)
    previous_skip_reason = _classifier_skip_reason_from_payload(previous_eval)

    comparison_rows: list[tuple[str, float | None, float | None, float | None, str]] = []
    prev_pass_mean = _numeric(previous_pass.get("mean"))
    curr_pass_mean = _numeric(current_pass.get("mean"))
    if current_skip_reason:
        curr_pass_mean = None
    if previous_skip_reason:
        prev_pass_mean = None
    pass_delta = (
        curr_pass_mean - prev_pass_mean
        if prev_pass_mean is not None and curr_pass_mean is not None
        else None
    )
    pass_better = "yes" if pass_delta is not None and pass_delta >= 0.0 else "no"
    comparison_rows.append(
        ("pass_prob_mean", prev_pass_mean, curr_pass_mean, pass_delta, pass_better)
    )

    for metric in METRIC_TARGETS:
        prev_value = _numeric(previous_metric_mae.get(metric))
        curr_value = _numeric(current_metric_mae.get(metric))
        delta = (
            curr_value - prev_value if prev_value is not None and curr_value is not None else None
        )
        better = "yes" if delta is not None and delta <= 0.0 else "no"
        comparison_rows.append((f"{metric}_mae", prev_value, curr_value, delta, better))

    reason_lines: list[str] = []
    if current_skip_reason:
        reason_lines.append(f"Current classifier skip reason: {current_skip_reason}")
    if previous_skip_reason:
        reason_lines.append(f"Previous classifier skip reason: {previous_skip_reason}")
    if reason_lines:
        summary_lines.append("")
        summary_lines.extend(reason_lines)

    summary_lines.append("")
    summary_lines.append(
        (
            "Latest iteration: "
            f"iter={latest_iteration} model={latest_model.get('model_id')} "
            f"promoted={_format_bool(latest_model.get('promoted'))} "
            f"backend={latest_model.get('compute_backend_resolved')} "
            f"token_learner={latest_model.get('token_learner_backend')}"
        )
    )
    summary_lines.append("Latest vs Previous (latest - previous):")
    summary_lines.append("metric               previous    latest      delta       better")
    for metric_name, previous_value, latest_value, delta_value, better in comparison_rows:
        summary_lines.append(
            f"{metric_name:<20} {_format_float(previous_value):>10}"
            f" {_format_float(latest_value):>10} {_format_delta(delta_value):>11} {better:>8}"
        )

    summary_lines.append("")
    summary_lines.append("Recent iterations:")
    metric_headers = " ".join(f"mae.{metric}" for metric in METRIC_TARGETS)
    summary_lines.append(
        (f"iter  run_status   verified/attempted  promoted improved pass_mean  {metric_headers}")
    )
    tail = records[-max(1, history) :]
    for record in tail:
        evaluation = _as_dict(record.get("evaluation"))
        current = _as_dict(evaluation.get("current"))
        improvement = _as_dict(evaluation.get("improvement"))
        metrics = _as_dict(current.get("metric_mae"))
        pass_probability = _as_dict(current.get("pass_probability"))
        model = _as_dict(record.get("model"))
        generation = _as_dict(record.get("generation"))
        run_status_value = generation.get("status") or record.get("run_status") or "N/A"
        verified_attempted_value = _format_verified_ratio(
            generation.get("successes"), generation.get("attempted")
        )
        row_skip_reason = _classifier_skip_reason_from_payload(current)
        row_pass_mean = _numeric(pass_probability.get("mean"))
        row_pass_display = None if row_skip_reason else row_pass_mean
        metric_values = " ".join(
            f"{_format_float(_numeric(metrics.get(metric))):>12}" for metric in METRIC_TARGETS
        )
        summary_lines.append(
            f"{record.get('iteration', '?'):>4} "
            f"{run_status_value:>11} "
            f"{verified_attempted_value:>19} "
            f"{_format_bool(model.get('promoted')):>9} "
            f"{_format_bool(improvement.get('improved')):>8} "
            f"{_format_float(row_pass_display):>9} {metric_values}"
        )

    best_record = max(records, key=lambda record: _record_pass_mean(record) or -1.0)
    best_eval = _as_dict(best_record.get("evaluation"))
    best_current = _as_dict(best_eval.get("current"))
    best_pass_probability = _as_dict(best_current.get("pass_probability"))
    summary_lines.append("")
    summary_lines.append(
        "Best pass_probability.mean so far: "
        f"iter={best_record.get('iteration', '?')} "
        f"value={_format_float(_numeric(best_pass_probability.get('mean')))}"
    )
    return "\n".join(summary_lines)


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
    output_format = getattr(args, "output_format", "human")
    if output_format == "json":
        print(json.dumps(state, indent=2))
        return 0
    iterations_path = loop_root / ITERATIONS_FILENAME
    records = _load_iteration_records(iterations_path)
    history = getattr(args, "history", 5)
    print(_render_status_human(loop_id, state, records, history=max(1, int(history))))
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
        help="PoB evaluation budget per iteration",
    )
    start_parser.add_argument(
        "--pool-multiplier",
        type=int,
        default=4,
        help="Generation pool multiplier (generate_count=count*pool_multiplier)",
    )
    start_parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=None,
        help="Optional explicit candidate pool override (default=generate_count)",
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
        "--surrogate-top-k",
        type=int,
        default=None,
        help="Number of surrogate-ranked candidates to keep when a surrogate is available (default=resolved evaluation budget; <=0 disables pruning)",
    )
    start_parser.add_argument(
        "--surrogate-exploration-pct",
        type=float,
        default=0.10,
        help="Exploration fraction for surrogate selection (clamped to 0..0.5)",
    )
    start_parser.add_argument(
        "--optimizer-iterations",
        type=int,
        default=3,
        help="Iterations to run when optimizer mode is enabled",
    )
    start_parser.add_argument(
        "--optimizer-elite-count",
        type=int,
        default=16,
        help="Elites preserved per optimizer iteration",
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
    status_parser.add_argument(
        "--format",
        dest="output_format",
        choices=("human", "json"),
        default="human",
        help="output format for status command",
    )
    status_parser.add_argument(
        "--json",
        dest="output_format",
        action="store_const",
        const="json",
        help="shortcut for --format json",
    )
    status_parser.add_argument(
        "--history",
        type=int,
        default=5,
        help="recent iteration rows to print in human output",
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
