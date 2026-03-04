from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import re
import statistics
import tarfile
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Mapping, Sequence

from backend.app.api.evaluator import BuildEvaluator
from backend.app.db.ch import ClickhouseRepository
from backend.app.settings import settings
from backend.engine.generation.runner import (
    _run_summary_path,
    load_run_summary,
    run_generation,
)
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
        "last_worker_metrics_used_count": 0,
        "last_fallback_stub_count": 0,
        "last_worker_error_count": 0,
        "last_worker_error": None,
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
            "worker_metrics_used_count": _coerce_int(
                evaluation_info.get("worker_metrics_used_count")
            ),
            "fallback_stub_count": _coerce_int(evaluation_info.get("fallback_stub_count")),
            "worker_error_count": _coerce_int(evaluation_info.get("worker_error_count")),
            "last_worker_error": evaluation_info.get("last_worker_error"),
        },
        "gate_fail_analysis": run_summary.get("gate_fail_analysis"),
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
            "worker_metrics_used_count": _coerce_int(
                evaluation_info.get("worker_metrics_used_count")
            ),
            "fallback_stub_count": _coerce_int(evaluation_info.get("fallback_stub_count")),
            "worker_error_count": _coerce_int(evaluation_info.get("worker_error_count")),
            "last_worker_error": evaluation_info.get("last_worker_error"),
        },
        "gate_fail_analysis": run_summary.get("gate_fail_analysis"),
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
        "last_worker_metrics_used_count": state.get("last_worker_metrics_used_count"),
        "last_fallback_stub_count": state.get("last_fallback_stub_count"),
        "last_worker_error_count": state.get("last_worker_error_count"),
        "last_worker_error": state.get("last_worker_error"),
    }


def compute_gate_fail_histogram(evaluations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Count failures per gate across all evaluations."""
    fail_counts: Counter[str] = Counter()
    total_evaluated = len(evaluations)

    for ev in evaluations:
        for reason in ev.get("gate_fail_reasons", []):
            fail_counts[reason] += 1

    # Sort by count descending
    sorted_fails = sorted(fail_counts.items(), key=lambda item: item[1], reverse=True)

    return {
        "total_evaluated": total_evaluated,
        "gate_fail_histogram": dict(sorted_fails),
        "gate_fail_percentages": {
            reason: f"{(count / total_evaluated * 100):.1f}%" if total_evaluated > 0 else "0.0%"
            for reason, count in sorted_fails
        },
    }


def compute_slack_quantiles(evaluations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute quantiles for min_gate_slack across all evaluations."""
    if not evaluations:
        return {}

    slacks = []
    for ev in evaluations:
        gate_slacks = ev.get("gate_slacks")
        if isinstance(gate_slacks, Mapping):
            slack = gate_slacks.get("min_gate_slack")
            if slack is not None:
                slacks.append(float(slack))

    if not slacks:
        return {}

    slacks.sort()
    n = len(slacks)

    def get_quantile(q: float) -> float:
        if not slacks:
            return 0.0
        position = (n - 1) * q
        lower_index = int(position)
        upper_index = lower_index + 1 if lower_index < n - 1 else lower_index
        lower_value = slacks[lower_index]
        upper_value = slacks[upper_index]
        if lower_index == upper_index:
            return float(lower_value)
        weight = position - lower_index
        return float(lower_value + (upper_value - lower_value) * weight)

    return {
        "min_gate_slack": {
            "min": float(slacks[0]),
            "p10": get_quantile(0.1),
            "median": float(statistics.median(slacks)),
            "p90": get_quantile(0.9),
            "max": float(slacks[-1]),
            "mean": float(statistics.mean(slacks)),
        }
    }


def generate_run_summary(evaluations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Generates a summary of the run including gate failure analysis."""
    return {
        "gate_fail_analysis": compute_gate_fail_histogram(evaluations),
        "slack_quantiles": compute_slack_quantiles(evaluations),
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

    try:
        constraints = _load_constraints(args.constraints_file)
    except Exception as exc:
        logger.error("ml loop %s constraints load failed: %s", loop_id, exc)
        constraints = None
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
        try:
            state = _load_state(state_path)
        except Exception as exc:
            logger.error("ml loop %s state load failed, rebuilding state: %s", loop_id, exc)
            backup_path = state_path.with_suffix(".corrupt.json")
            try:
                state_path.replace(backup_path)
            except OSError as backup_exc:
                logger.warning(
                    "ml loop %s unable to preserve corrupt state file %s: %s",
                    loop_id,
                    state_path,
                    backup_exc,
                )
            state = _build_initial_state(loop_id, total_iterations, args.seed_start, generate_count)
            _write_state(state_path, state)
            state_exists = False
    else:
        state = _build_initial_state(loop_id, total_iterations, args.seed_start, generate_count)
        _write_state(state_path, state)

    _ensure_seed_metadata(state_path, state, args)
    try:
        state = _load_state(state_path)
    except Exception as exc:
        logger.error(
            "ml loop %s state reload failed, continuing with in-memory state: %s", loop_id, exc
        )
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

    repo: ClickhouseRepository | None = None
    evaluator: BuildEvaluator | None = None

    stop_triggered = False
    current_iteration = 0
    iteration = start_iteration
    current_run_id: str | None = None
    iteration_seed_start: int | None = None
    try:
        repo = ClickhouseRepository()
        evaluator = BuildEvaluator(repo=repo, base_path=data_root)
        evaluator.require_worker_metrics_for_profile(profile_id)
        evaluator.require_non_stub_metrics_for_profile(profile_id)
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
                        "ml loop %s iteration %d surrogate model load failed for %s; "
                        "falling back to path-based loading: %s",
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
                repo=repo,
                evaluator=evaluator,
                run_mode="standard",
                surrogate_enabled=bool(active_surrogate_model_path),
                surrogate_model_path=active_surrogate_model_path,
                surrogate_predictor=surrogate_predictor,
                candidate_pool_size=resolved_candidate_pool_size,
                surrogate_top_k=surrogate_top_k_value,
                surrogate_exploration_pct=surrogate_exploration_pct,
                optimizer_iterations=optimizer_iterations,
                optimizer_elite_count=optimizer_elite_count,
                enforce_worker_tripwire=True,
            )
            # TASK 0.3: Add gate fail analysis to run summary
            gate_evals_raw = run_summary.get("evaluation", {}).get("gate_evaluations", [])
            gate_evaluations = [
                {
                    "gate_pass": e.get("gate_pass"),
                    "gate_fail_reasons": e.get("gate_fail_reasons", []),
                    "gate_slacks": e.get("gate_slacks", {}),
                }
                for e in gate_evals_raw
            ]
            run_summary.update(generate_run_summary(gate_evaluations))

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
            worker_metrics_used_count = (
                _coerce_int(evaluation_summary.get("worker_metrics_used_count")) or 0
            )
            fallback_stub_count = _coerce_int(evaluation_summary.get("fallback_stub_count")) or 0
            worker_error_count = _coerce_int(evaluation_summary.get("worker_error_count")) or 0
            last_worker_error = evaluation_summary.get("last_worker_error")
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
                last_worker_metrics_used_count=worker_metrics_used_count,
                last_fallback_stub_count=fallback_stub_count,
                last_worker_error_count=worker_error_count,
                last_worker_error=last_worker_error,
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
                if reason_code == "evaluation_non_pob_metrics":
                    raise RuntimeError(failure_message)
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
    finally:
        if evaluator is not None:
            evaluator.close()
        if repo is not None:
            repo.close()


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


REPORT_CSV_SEPARATOR = ";"
REPORT_METRICS = tuple(METRIC_TARGETS)
REPORT_FIELD_ORDER = [
    "iteration",
    "run_id",
    "snapshot_id",
    "model_id",
    "promoted",
    "generation.candidate_pool_size",
    "generation.evaluation_budget",
    "generation.surrogate_enabled",
    "generation.surrogate_model_id",
    "generation.surrogate_status",
    "generation.fallback_reason",
    "generation.counts.candidates",
    "generation.counts.selected",
    "generation.counts.pruned",
    "generation.counts.exploration",
    "surrogate.full_dps.min",
    "surrogate.full_dps.median",
    "surrogate.full_dps.p95",
    "surrogate.full_dps.max",
    "surrogate.full_dps.std",
    "surrogate.full_dps.uniq",
    "surrogate.full_dps.degenerate",
    "surrogate.max_hit.min",
    "surrogate.max_hit.median",
    "surrogate.max_hit.p95",
    "surrogate.max_hit.max",
    "surrogate.max_hit.std",
    "surrogate.max_hit.uniq",
    "surrogate.max_hit.degenerate",
    "evaluation.attempted",
    "evaluation.verified",
    "evaluation.failures",
    "evaluation.errors",
    "evaluation.gate_pass_rate",
    "evaluation.gate_fail_reason_counts",
    "evaluation.metrics_source_counts",
    "evaluation.full_dps.min",
    "evaluation.full_dps.median",
    "evaluation.full_dps.p95",
    "evaluation.full_dps.max",
    "evaluation.full_dps.std",
    "evaluation.full_dps.uniq",
    "evaluation.max_hit.min",
    "evaluation.max_hit.median",
    "evaluation.max_hit.p95",
    "evaluation.max_hit.max",
    "evaluation.max_hit.std",
    "evaluation.max_hit.uniq",
    "evaluation.gate_pass_0",
    "evaluation.gate_pass_1",
    "training.snapshot_row_count",
    "training.label_counts",
    "training.feature_sparsity.nonzero_signals",
    "training.feature_sparsity.nonzero_tokens",
    "training.backend",
    "training.token_learner",
    "training.train_seconds",
    "model_quality.metric_mae_log1p_pass.full_dps",
    "model_quality.metric_mae_log1p_pass.max_hit",
    "model_quality.classifier_metrics",
    "warnings",
]


def _safe_load_state(state_path: Path) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    if not state_path.exists():
        warnings.append("state_missing")
        return {}, warnings
    try:
        return _load_state(state_path), warnings
    except Exception as exc:
        warnings.append(f"state_invalid:{exc}")
        return {}, warnings


def _try_load_json(
    path: Path, description: str, warnings: list[str]
) -> Mapping[str, Any] | list[Any] | None:
    try:
        if not path.exists():
            warnings.append(f"{description}_missing")
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        warnings.append(f"{description}_invalid:{exc}")
    except OSError as exc:
        warnings.append(f"{description}_io_error:{exc}")
    except Exception as exc:
        warnings.append(f"{description}_error:{exc}")
    return None


def _safe_percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * percentile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower_value = sorted_values[int(lower_index)]
    upper_value = sorted_values[int(upper_index)]
    if lower_index == upper_index:
        return float(lower_value)
    weight = position - lower_index
    return float(lower_value + (upper_value - lower_value) * weight)


def _metric_stats(values: Sequence[float]) -> dict[str, float | int | bool | None]:
    if not values:
        return {
            "min": None,
            "median": None,
            "p95": None,
            "max": None,
            "std": None,
            "uniq": None,
            "degenerate": None,
        }
    sorted_values = sorted(values)
    uniq_values = len(set(sorted_values))
    std_value = statistics.pstdev(sorted_values)
    return {
        "min": float(sorted_values[0]),
        "median": float(statistics.median(sorted_values)),
        "p95": _safe_percentile(sorted_values, 0.95),
        "max": float(sorted_values[-1]),
        "std": float(std_value),
        "uniq": uniq_values,
        "degenerate": len(sorted_values) <= 1 or std_value <= 1e-6,
    }


def _collect_surrogate_metric_stats(
    payload: Mapping[str, Any] | None, warnings: list[str]
) -> dict[str, dict[str, float | int | bool | None]]:
    stats = {metric: _metric_stats([]) for metric in REPORT_METRICS}
    if not isinstance(payload, Mapping):
        return stats
    candidates = payload.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return stats
    values: dict[str, list[float]] = {metric: [] for metric in REPORT_METRICS}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        predicted = candidate.get("predicted_metrics")
        if not isinstance(predicted, Mapping):
            continue
        for metric in REPORT_METRICS:
            value = _numeric(predicted.get(metric))
            if value is not None:
                values[metric].append(value)
    if not any(values[metric] for metric in REPORT_METRICS):
        warnings.append("surrogate_predictions_no_values")
    for metric in REPORT_METRICS:
        stats[metric] = _metric_stats(values[metric])
    return stats


def _exploration_count(attempt_records: Sequence[Mapping[str, Any]] | None) -> int:
    if not attempt_records:
        return 0
    return sum(
        1
        for record in attempt_records
        if record.get("surrogate_selection_reason") == "surrogate_exploration"
    )


def _collect_gate_pass_counts(
    records: Sequence[Mapping[str, Any]] | None,
) -> tuple[int | None, int | None]:
    if not records:
        return None, None
    zeros = ones = 0
    seen = 0
    for entry in records:
        gate_value = entry.get("gate_pass")
        if gate_value is True:
            ones += 1
            seen += 1
        elif gate_value is False:
            zeros += 1
            seen += 1
    if seen == 0:
        return None, None
    return zeros, ones


def _derive_feature_sparsity(
    model_payload: Mapping[str, Any] | None,
) -> tuple[int | None, int | None]:
    if not isinstance(model_payload, Mapping):
        return None, None
    feature_stats = model_payload.get("feature_stats")
    nonzero_signals: int | None = None
    if isinstance(feature_stats, Mapping):
        count = 0
        for value in feature_stats.values():
            if isinstance(value, Mapping):
                parsed = _numeric(value.get("count"))
                if parsed is not None and parsed > 0:
                    count += 1
        nonzero_signals = count
    identity_effects = model_payload.get("identity_token_effects")
    cross_effects = model_payload.get("identity_cross_token_effects")
    tokens: set[str] = set()
    for effect_map in (identity_effects, cross_effects):
        if not isinstance(effect_map, Mapping):
            continue
        for token_values in effect_map.values():
            if not isinstance(token_values, Mapping):
                continue
            for token, value in token_values.items():
                numeric_value = _numeric(value)
                if numeric_value is not None and numeric_value != 0.0:
                    tokens.add(str(token))
    nonzero_tokens = len(tokens)
    return nonzero_signals, nonzero_tokens


def _extract_metric_value(entry: Mapping[str, Any], metric: str) -> float | None:
    direct = _numeric(entry.get(metric))
    if direct is not None:
        return direct
    for key in ("metrics", "actual_metrics", "predicted_metrics", "best_metrics"):
        nested = entry.get(key)
        if not isinstance(nested, Mapping):
            continue
        nested_value = _numeric(nested.get(metric))
        if nested_value is not None:
            return nested_value
    return None


def _collect_evaluation_metric_stats(
    run_summary: Mapping[str, Any], warnings: list[str]
) -> dict[str, dict[str, float | int | bool | None]]:
    values: dict[str, list[float]] = {metric: [] for metric in REPORT_METRICS}

    evaluation = _as_dict(run_summary.get("evaluation"))
    generation = _as_dict(run_summary.get("generation"))
    record_sets: list[Sequence[Any]] = []

    evaluation_records = evaluation.get("records")
    if isinstance(evaluation_records, Sequence) and not isinstance(
        evaluation_records, (str, bytes)
    ):
        record_sets.append(evaluation_records)

    generation_records = generation.get("records")
    if isinstance(generation_records, Sequence) and not isinstance(
        generation_records, (str, bytes)
    ):
        record_sets.append(generation_records)

    generation_attempt_records = generation.get("attempt_records")
    if isinstance(generation_attempt_records, Sequence) and not isinstance(
        generation_attempt_records, (str, bytes)
    ):
        record_sets.append(generation_attempt_records)

    for records in record_sets:
        for entry in records:
            if not isinstance(entry, Mapping):
                continue
            for metric in REPORT_METRICS:
                value = _extract_metric_value(entry, metric)
                if value is not None:
                    values[metric].append(value)

    if not any(values.get(metric) for metric in REPORT_METRICS):
        benchmark = _as_dict(run_summary.get("benchmark"))
        scenarios = benchmark.get("scenarios")
        if isinstance(scenarios, Mapping):
            for scenario_payload in scenarios.values():
                if not isinstance(scenario_payload, Mapping):
                    continue
                full_dps = _numeric(scenario_payload.get("median_full_dps"))
                max_hit = _numeric(scenario_payload.get("median_max_hit"))
                if full_dps is not None and "full_dps" in values:
                    values["full_dps"].append(full_dps)
                if max_hit is not None and "max_hit" in values:
                    values["max_hit"].append(max_hit)
        if any(values.get(metric) for metric in REPORT_METRICS):
            warnings.append("evaluation_stats_from_benchmark_medians")

    if not any(values.get(metric) for metric in REPORT_METRICS):
        warnings.append("evaluation_actual_metrics_missing")

    return {metric: _metric_stats(values[metric]) for metric in REPORT_METRICS}


def _mapping_numeric(payload: Mapping[str, Any] | None, keys: Sequence[str]) -> float | None:
    if not isinstance(payload, Mapping):
        return None
    for key in keys:
        value = _numeric(payload.get(key))
        if value is not None:
            return value
    return None


def _resolve_train_seconds(
    meta_payload: Mapping[str, Any] | None,
    run_summary: Mapping[str, Any],
) -> float | None:
    common_keys = ("train_seconds", "training_seconds", "elapsed_seconds", "duration_seconds")
    value = _mapping_numeric(meta_payload, common_keys)
    if value is not None:
        return value
    meta_timings = _as_dict(_as_dict(meta_payload).get("timings"))
    value = _mapping_numeric(meta_timings, common_keys)
    if value is not None:
        return value

    ml_lifecycle = _as_dict(run_summary.get("ml_lifecycle"))
    metadata = _as_dict(ml_lifecycle.get("metadata"))
    model_meta = _as_dict(metadata.get("model_meta"))
    value = _mapping_numeric(model_meta, common_keys)
    if value is not None:
        return value
    model_meta_timings = _as_dict(model_meta.get("timings"))
    return _mapping_numeric(model_meta_timings, common_keys)


def _build_report_rows(
    data_root: Path,
    loop_root: Path,
    iterations: Sequence[dict[str, Any]],
    last: int | None,
) -> list[dict[str, Any]]:
    if last and last > 0:
        selected = iterations[-last:]
    else:
        selected = list(iterations)
    state_path = loop_root / STATE_FILENAME
    state_payload, state_warnings = _safe_load_state(state_path)
    summary_cache: dict[str, dict[str, Any]] = {}
    summary_warnings: dict[str, list[str]] = {}
    prediction_cache: dict[str, Mapping[str, Any] | None] = {}
    prediction_warnings: dict[str, list[str]] = {}

    def _load_summary(run_id: str) -> dict[str, Any]:
        if run_id in summary_cache:
            return summary_cache[run_id]
        warnings: list[str] = []
        try:
            summary = load_run_summary(run_id, base_path=data_root)
        except FileNotFoundError:
            summary = {}
            warnings.append(f"run_summary_missing:{run_id}")
        except Exception as exc:
            summary = {}
            warnings.append(f"run_summary_error:{exc}")
        summary_cache[run_id] = summary
        summary_warnings[run_id] = warnings
        return summary

    def _load_predictions(run_id: str) -> Mapping[str, Any] | None:
        if run_id in prediction_cache:
            return prediction_cache[run_id]
        warnings_list: list[str] = []
        try:
            summary_path = _run_summary_path(run_id, base_path=data_root)
            payload_raw = _try_load_json(
                summary_path.parent / "surrogate_predictions.json",
                "surrogate_predictions",
                warnings_list,
            )
            if isinstance(payload_raw, Mapping):
                payload: Mapping[str, Any] | None = payload_raw
            else:
                payload = None
                if payload_raw is not None:
                    warnings_list.append(f"surrogate_predictions_invalid_shape:{run_id}")
        except FileNotFoundError:
            payload = None
            warnings_list.append(f"surrogate_predictions_missing:{run_id}")
        except Exception as exc:
            payload = None
            warnings_list.append(f"surrogate_predictions_error:{exc}")
        prediction_cache[run_id] = payload
        prediction_warnings[run_id] = warnings_list
        return payload

    rows: list[dict[str, Any]] = []
    for record in selected:
        row_warnings = list(state_warnings)
        run_id_value = record.get("run_id")
        run_id = str(run_id_value) if run_id_value else ""
        run_summary: dict[str, Any] = {}
        predictions_payload: Mapping[str, Any] | None = None
        if run_id:
            run_summary = _load_summary(run_id)
            row_warnings.extend(summary_warnings.get(run_id, []))
            predictions_payload = _load_predictions(run_id)
            row_warnings.extend(prediction_warnings.get(run_id, []))
        else:
            row_warnings.append("run_id_missing")
        row: dict[str, Any] = {key: None for key in REPORT_FIELD_ORDER}
        row["iteration"] = record.get("iteration")
        row["run_id"] = run_id or run_id_value
        snapshot_info = record.get("snapshot") or {}
        row["snapshot_id"] = snapshot_info.get("snapshot_id")
        model_info = record.get("model") or {}
        row["model_id"] = model_info.get("model_id")
        row["promoted"] = model_info.get("promoted")
        parameters = _as_dict(run_summary.get("parameters"))
        surrogate = _as_dict(run_summary.get("surrogate"))
        generation = _as_dict(run_summary.get("generation"))
        counts = _as_dict(surrogate.get("counts"))
        row["generation.candidate_pool_size"] = parameters.get("candidate_pool_size")
        row["generation.evaluation_budget"] = parameters.get("count")
        row["generation.surrogate_enabled"] = parameters.get("surrogate_enabled")
        row["generation.surrogate_model_id"] = surrogate.get("model_id")
        row["generation.surrogate_status"] = surrogate.get("status")
        row["generation.fallback_reason"] = surrogate.get("fallback_reason")
        row["generation.counts.candidates"] = counts.get("candidates")
        row["generation.counts.selected"] = counts.get("selected")
        row["generation.counts.pruned"] = counts.get("pruned")
        generation_attempt_records = generation.get("attempt_records")
        gate_source: Sequence[Mapping[str, Any]] | None = None
        if isinstance(generation_attempt_records, Sequence) and not isinstance(
            generation_attempt_records, (str, bytes)
        ):
            gate_source = [
                entry for entry in generation_attempt_records if isinstance(entry, Mapping)
            ]
        generation_records = generation.get("records")
        if (
            gate_source is None
            and isinstance(generation_records, Sequence)
            and not isinstance(generation_records, (str, bytes))
        ):
            gate_source = [entry for entry in generation_records if isinstance(entry, Mapping)]
        row["generation.counts.exploration"] = _exploration_count(gate_source)
        surrogate_stats = _collect_surrogate_metric_stats(predictions_payload, row_warnings)
        for metric in REPORT_METRICS:
            metric_stats = surrogate_stats.get(metric, {})
            row[f"surrogate.{metric}.min"] = metric_stats.get("min")
            row[f"surrogate.{metric}.median"] = metric_stats.get("median")
            row[f"surrogate.{metric}.p95"] = metric_stats.get("p95")
            row[f"surrogate.{metric}.max"] = metric_stats.get("max")
            row[f"surrogate.{metric}.std"] = metric_stats.get("std")
            row[f"surrogate.{metric}.uniq"] = metric_stats.get("uniq")
            row[f"surrogate.{metric}.degenerate"] = metric_stats.get("degenerate")
        evaluation_summary = _as_dict(run_summary.get("evaluation"))
        benchmark_summary = _as_dict(run_summary.get("benchmark"))
        row["evaluation.attempted"] = _coerce_int(evaluation_summary.get("attempted"))
        row["evaluation.verified"] = _coerce_int(evaluation_summary.get("successes"))
        row["evaluation.failures"] = _coerce_int(evaluation_summary.get("failures"))
        row["evaluation.errors"] = _coerce_int(evaluation_summary.get("errors"))
        gate_zero, gate_one = _collect_gate_pass_counts(gate_source)
        row["evaluation.gate_pass_0"] = gate_zero
        row["evaluation.gate_pass_1"] = gate_one
        gate_pass_rate = _numeric(benchmark_summary.get("gate_pass_rate"))
        gate_zero_count = _coerce_int(gate_zero) or 0
        gate_one_count = _coerce_int(gate_one) or 0
        if gate_pass_rate is None:
            denominator = gate_zero_count + gate_one_count
            if denominator > 0:
                gate_pass_rate = gate_one_count / denominator
            else:
                scenarios_payload = benchmark_summary.get("scenarios")
                if isinstance(scenarios_payload, Mapping):
                    weighted_total = 0.0
                    sample_total = 0
                    for scenario_payload in scenarios_payload.values():
                        if not isinstance(scenario_payload, Mapping):
                            continue
                        samples = _coerce_int(scenario_payload.get("samples"))
                        scenario_rate = _numeric(scenario_payload.get("gate_pass_rate"))
                        if samples is None or samples <= 0 or scenario_rate is None:
                            continue
                        sample_total += samples
                        weighted_total += scenario_rate * samples
                    if sample_total > 0:
                        gate_pass_rate = weighted_total / sample_total
        row["evaluation.gate_pass_rate"] = gate_pass_rate
        gate_fail_reason_counts = benchmark_summary.get("gate_fail_reason_counts")
        if isinstance(gate_fail_reason_counts, Mapping):
            row["evaluation.gate_fail_reason_counts"] = dict(gate_fail_reason_counts)
        metrics_source_counts = evaluation_summary.get("metrics_source_counts")
        if not isinstance(metrics_source_counts, Mapping):
            metrics_source_counts = benchmark_summary.get("metrics_source_counts")
        if isinstance(metrics_source_counts, Mapping):
            row["evaluation.metrics_source_counts"] = dict(metrics_source_counts)
        evaluation_stats = _collect_evaluation_metric_stats(run_summary, row_warnings)
        for metric in REPORT_METRICS:
            metric_stats = evaluation_stats.get(metric, {})
            row[f"evaluation.{metric}.min"] = metric_stats.get("min")
            row[f"evaluation.{metric}.median"] = metric_stats.get("median")
            row[f"evaluation.{metric}.p95"] = metric_stats.get("p95")
            row[f"evaluation.{metric}.max"] = metric_stats.get("max")
            row[f"evaluation.{metric}.std"] = metric_stats.get("std")
            row[f"evaluation.{metric}.uniq"] = metric_stats.get("uniq")
        row["training.snapshot_row_count"] = snapshot_info.get("row_count")
        row["training.backend"] = model_info.get("compute_backend_resolved")
        row["training.token_learner"] = model_info.get("token_learner_backend")
        meta_payload: Mapping[str, Any] | None = None
        meta_path = model_info.get("meta_path")
        if meta_path:
            loaded_meta = _try_load_json(Path(meta_path), "training_meta", row_warnings)
            if isinstance(loaded_meta, Mapping):
                meta_payload = loaded_meta
            elif loaded_meta is not None:
                row_warnings.append("training_meta_invalid_shape")
        else:
            row_warnings.append("training_meta_missing")
        if isinstance(meta_payload, Mapping):
            classifier_distribution = meta_payload.get("classifier_label_distribution")
            if isinstance(classifier_distribution, Mapping):
                row["training.label_counts"] = {
                    str(key): int(value) if isinstance(value, (int, float)) else value
                    for key, value in classifier_distribution.items()
                }
            backend_value = meta_payload.get("compute_backend_resolved")
            token_value = meta_payload.get("token_learner_backend")
            if backend_value is not None:
                row["training.backend"] = backend_value
            if token_value is not None:
                row["training.token_learner"] = token_value
        else:
            row["training.label_counts"] = None
        model_payload: Mapping[str, Any] | None = None
        model_path = model_info.get("model_path")
        if model_path:
            loaded_model = _try_load_json(Path(model_path), "training_model", row_warnings)
            if isinstance(loaded_model, Mapping):
                model_payload = loaded_model
            elif loaded_model is not None:
                row_warnings.append("training_model_invalid_shape")
        sparsity_signals, sparsity_tokens = _derive_feature_sparsity(model_payload)
        row["training.feature_sparsity.nonzero_signals"] = sparsity_signals
        row["training.feature_sparsity.nonzero_tokens"] = sparsity_tokens
        row["training.train_seconds"] = _resolve_train_seconds(meta_payload, run_summary)
        current_evaluation = _as_dict(_as_dict(record.get("evaluation")).get("current"))
        mae_log1p_pass = _as_dict(current_evaluation.get("metric_mae_log1p_pass"))
        row["model_quality.metric_mae_log1p_pass.full_dps"] = _numeric(
            mae_log1p_pass.get("full_dps")
        )
        row["model_quality.metric_mae_log1p_pass.max_hit"] = _numeric(mae_log1p_pass.get("max_hit"))
        classifier_metrics = current_evaluation.get("classifier_metrics")
        if isinstance(classifier_metrics, Mapping):
            row["model_quality.classifier_metrics"] = dict(classifier_metrics)
        else:
            row["model_quality.classifier_metrics"] = None
        row["warnings"] = row_warnings
        rows.append(row)
    return rows


def _format_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _write_report_file(rows: Sequence[dict[str, Any]], path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    if fmt == "csv":
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REPORT_FIELD_ORDER)
            writer.writeheader()
            for row in rows:
                csv_row = {}
                for field in REPORT_FIELD_ORDER:
                    if field == "warnings":
                        csv_row[field] = REPORT_CSV_SEPARATOR.join(row.get(field) or [])
                        continue
                    csv_row[field] = _format_csv_value(row.get(field))
                writer.writerow(csv_row)
        return
    raise ValueError(f"unsupported report format: {fmt}")


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
    summary_lines.append(
        (
            "Tripwire counts: "
            f"worker_metrics={_format_count(state.get('last_worker_metrics_used_count'))} "
            f"fallback_stub={_format_count(state.get('last_fallback_stub_count'))} "
            f"worker_errors={_format_count(state.get('last_worker_error_count'))} "
            f"last_worker_error={state.get('last_worker_error') or 'N/A'}"
        )
    )

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
        (
            "iter  run_status   verified/attempted  promoted improved pass_mean"
            " worker/fallback/errors  "
            f"{metric_headers}"
        )
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
        worker_tripwire_counts = (
            f"{_format_count(generation.get('worker_metrics_used_count'))}/"
            f"{_format_count(generation.get('fallback_stub_count'))}/"
            f"{_format_count(generation.get('worker_error_count'))}"
        )
        metric_values = " ".join(
            f"{_format_float(_numeric(metrics.get(metric))):>12}" for metric in METRIC_TARGETS
        )
        summary_lines.append(
            f"{record.get('iteration', '?'):>4} "
            f"{run_status_value:>11} "
            f"{verified_attempted_value:>19} "
            f"{_format_bool(model.get('promoted')):>9} "
            f"{_format_bool(improvement.get('improved')):>8} "
            f"{_format_float(row_pass_display):>9} "
            f"{worker_tripwire_counts:>22} {metric_values}"
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


def report_loop(args: argparse.Namespace) -> int:
    loop_id = _validate_loop_id(args.loop_id)
    data_root = Path(args.data_path)
    data_root.mkdir(parents=True, exist_ok=True)
    loop_root = _loop_root(data_root, loop_id)
    iterations_path = loop_root / ITERATIONS_FILENAME
    iterations = _load_iteration_records(iterations_path)
    last_value = getattr(args, "last", None)
    window = last_value if last_value and last_value > 0 else None
    rows = _build_report_rows(data_root, loop_root, iterations, window)
    output_path = Path(args.out)
    _write_report_file(rows, output_path, args.format)
    return 0


def bundle_loop(args: argparse.Namespace) -> int:
    loop_id = _validate_loop_id(args.loop_id)
    data_root = Path(args.data_path)
    data_root.mkdir(parents=True, exist_ok=True)
    loop_root = _loop_root(data_root, loop_id)
    iterations_path = loop_root / ITERATIONS_FILENAME
    iterations = _load_iteration_records(iterations_path)
    last_value = getattr(args, "last", None)
    window = last_value if last_value and last_value > 0 else None
    rows = _build_report_rows(data_root, loop_root, iterations, window)
    if window and window > 0:
        selected_iterations = iterations[-window:]
    else:
        selected_iterations = list(iterations)
    state_path = loop_root / STATE_FILENAME
    state_payload, _ = _safe_load_state(state_path)

    def _determine_champion() -> Path | None:
        last_model = state_payload.get("last_model_path")
        if last_model:
            return Path(last_model)
        for record in reversed(selected_iterations):
            model_info = record.get("model")
            if not isinstance(model_info, Mapping):
                continue
            model_path = model_info.get("model_path")
            if model_path:
                return Path(model_path)
        return None

    champion_model_path = _determine_champion()
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_root = Path(temp_dir)
        report_json = tmp_root / "report.json"
        report_csv = tmp_root / "report.csv"
        _write_report_file(rows, report_json, "json")
        _write_report_file(rows, report_csv, "csv")
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        warnings_list: list[str] = []
        files_info: list[dict[str, Any]] = []
        repo_root = Path(__file__).resolve().parents[2]
        with tarfile.open(output_path, "w:gz") as tar:

            def _add_file(
                src: Path,
                arcname: str,
                description: str,
                *,
                append_entry: bool = True,
                override_entry: dict[str, Any] | None = None,
            ) -> None:
                entry = override_entry or {
                    "path": arcname,
                    "source": str(src),
                    "description": description,
                }
                if not src.exists():
                    entry["status"] = "missing"
                    if append_entry:
                        files_info.append(entry)
                    warnings_list.append(f"{description}_missing")
                    return
                tar.add(src, arcname=arcname)
                entry["status"] = "included"
                if append_entry:
                    files_info.append(entry)

            _add_file(loop_root / STATE_FILENAME, f"ml_loops/{loop_id}/state.json", "loop_state")
            _add_file(
                loop_root / ITERATIONS_FILENAME,
                f"ml_loops/{loop_id}/iterations.jsonl",
                "iteration_records",
            )
            _add_file(report_json, "report.json", "loop_report_json")
            _add_file(report_csv, "report.csv", "loop_report_csv")
            backend_dir = repo_root / "backend"
            _add_file(backend_dir, "backend", "backend_source")
            run_ids = sorted(
                run_id
                for run_id in {row.get("run_id") for row in rows}
                if isinstance(run_id, str) and run_id
            )
            for run_id in run_ids:
                try:
                    summary_path = _run_summary_path(run_id, base_path=data_root)
                except Exception as exc:
                    warnings_list.append(f"run_summary_path_error:{run_id}:{exc}")
                    continue
                _add_file(
                    summary_path,
                    f"runs/{run_id}/summary.json",
                    f"run_summary:{run_id}",
                )
                _add_file(
                    summary_path.parent / "surrogate_predictions.json",
                    f"runs/{run_id}/surrogate_predictions.json",
                    f"surrogate_predictions:{run_id}",
                )
            if champion_model_path:
                _add_file(
                    champion_model_path,
                    f"ml_loops/{loop_id}/champion_model.json",
                    "champion_model",
                )
            else:
                warnings_list.append("champion_model_missing")
            manifest_path = tmp_root / "bundle_manifest.json"
            manifest_entry = {
                "path": "bundle_manifest.json",
                "source": str(manifest_path),
                "description": "bundle_manifest",
                "status": "included",
            }
            files_info.append(manifest_entry)
            manifest_data = {
                "loop_id": loop_id,
                "selected_iterations": [record.get("iteration") for record in selected_iterations],
                "generated_at": _now(),
                "warnings": warnings_list,
                "files": list(files_info),
            }
            manifest_path.write_text(
                json.dumps(manifest_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            _add_file(
                manifest_path,
                "bundle_manifest.json",
                "bundle_manifest",
                append_entry=False,
                override_entry=manifest_entry,
            )
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
        help=(
            "Number of surrogate-ranked candidates to keep when a surrogate is "
            "available (default=resolved evaluation budget; <=0 disables pruning)"
        ),
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

    report_parser = subparsers.add_parser("report", help="Generate an ML loop report")
    report_parser.add_argument("--loop-id", required=True, help="Loop identifier")
    report_parser.add_argument(
        "--data-path",
        type=Path,
        default=settings.data_path,
        help="Root directory for artifacts and loop state",
    )
    report_parser.add_argument(
        "--format",
        choices=("json", "csv"),
        default="json",
        help="Output format for the report",
    )
    report_parser.add_argument("--out", type=Path, required=True, help="Report output path")
    report_parser.add_argument(
        "--last",
        type=int,
        default=None,
        help="Limit the report to the final N iterations",
    )

    bundle_parser = subparsers.add_parser("bundle", help="Bundle ML loop artifacts")
    bundle_parser.add_argument("--loop-id", required=True, help="Loop identifier")
    bundle_parser.add_argument(
        "--data-path",
        type=Path,
        default=settings.data_path,
        help="Root directory for artifacts and loop state",
    )
    bundle_parser.add_argument("--out", type=Path, required=True, help="Tarball output path")
    bundle_parser.add_argument(
        "--last",
        type=int,
        default=None,
        help="Include only the final N iterations",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "start":
        return start_loop(args)
    if args.command == "stop":
        return stop_loop(args)
    if args.command == "status":
        return status_loop(args)
    if args.command == "report":
        return report_loop(args)
    if args.command == "bundle":
        return bundle_loop(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
