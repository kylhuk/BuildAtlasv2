"""Baseline surrogate artifacts for EP-V4 scenarios.

This module ships a deterministic, lightweight regression+probability
baseline that only relies on the `data/datasets/ep-v4/<snapshot>/`
artifacts. The surrogate is explicitly a fallback proposal generator;
when the dataset has no rows or confidence is low, callers should fall
back to the authoritative PoB oracle instead of treating these
predictions as ground truth.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import exp, expm1, floor, isfinite, log, log1p
from pathlib import Path
from statistics import mean, median, pstdev
from time import monotonic
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore

from backend.engine.surrogate.dataset import (
    FEATURE_IDENTITY_CROSS_TOKENS,
    FEATURE_IDENTITY_TOKENS,
    FEATURE_SCHEMA_VERSION,
    FEATURE_SIGNAL_KEYS,
)

MODEL_BACKEND = "ep-v4-baseline"
MODEL_VERSION = "0.2.0"
METRIC_TARGETS = ("full_dps", "max_hit")
PASS_METRIC = "full_dps"
TARGET_TRANSFORM_IDENTITY = "identity"
TARGET_TRANSFORM_LOG1P = "log1p"
DEFAULT_TARGET_TRANSFORMS: Mapping[str, str] = {
    "full_dps": TARGET_TRANSFORM_LOG1P,
    "max_hit": TARGET_TRANSFORM_LOG1P,
}
BACKEND_PREFERENCE_AUTO = "auto"
BACKEND_PREFERENCE_CPU = "cpu"
BACKEND_PREFERENCE_CUDA = "cuda"
COMPUTE_BACKEND_CPU = "cpu"
COMPUTE_BACKEND_CUDA = "cuda"
VALID_BACKEND_PREFERENCES = {
    BACKEND_PREFERENCE_AUTO,
    BACKEND_PREFERENCE_CPU,
    BACKEND_PREFERENCE_CUDA,
}
_MISSING_SKILL = "__missing__"
_DATASET_FILENAME = "dataset.jsonl"
_MANIFEST_FILENAME = "manifest.json"
_MODEL_FILENAME = "model.json"
_METRICS_FILENAME = "metrics.json"
_META_FILENAME = "model_meta.json"
IDENTITY_TOKEN_EFFECT_LIMIT = 256
CROSS_TOKEN_EFFECT_LIMIT = 256
_TOKEN_SHRINKAGE_PRIOR = 5.0
TOKEN_LEARNER_RESIDUAL_MEAN = "residual_mean"
TOKEN_LEARNER_TORCH_SPARSE = "torch_sparse_sgd"
CLASSIFIER_BACKEND = "sgd_logistic_v1"
CLASSIFIER_HASH_DIM = 256
CLASSIFIER_EPOCHS = 120
CLASSIFIER_LEARNING_RATE = 0.05
CLASSIFIER_L2 = 1e-4
CLASSIFIER_LR_DECAY = 0.99

DEFAULT_VAL_SPLIT_MOD = 5
DEFAULT_VAL_SPLIT_REMAINDER = 0

DETERMINISTIC_BALANCE_VERSION = "1.0"
DETERMINISTIC_BALANCE_SEED = 0
DETERMINISTIC_BALANCE_KEY_FIELDS = ("class", "main_skill_package")
DETERMINISTIC_BALANCE_MIN_TOTAL_ROWS = 50
DETERMINISTIC_BALANCE_MIN_NICHE_COUNT = 5
DETERMINISTIC_BALANCE_DOMINANT_SHARE = 0.55
DETERMINISTIC_BALANCE_CAP_FLOOR = 20
DETERMINISTIC_BALANCE_MEDIAN_MULTIPLIER = 2.0
DETERMINISTIC_BALANCE_MIN_OUTPUT_ROWS = 30
DETERMINISTIC_BALANCE_MIN_RETENTION_RATIO = 0.6
DETERMINISTIC_BALANCE_HASH_SALT = "ep-v4-balance-v1"
DETERMINISTIC_BALANCE_UNKNOWN_LABEL = _MISSING_SKILL

logger = logging.getLogger(__name__)


def _train_phase_log(message: str, *args: object) -> None:
    if logger.isEnabledFor(logging.INFO):
        logger.info(message, *args)
        return
    logger.warning(message, *args)


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    std: float
    minimum: float
    maximum: float
    count: int

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.minimum,
            "max": self.maximum,
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MetricSummary":
        return cls(
            mean=float(payload.get("mean", 0.0)),
            std=float(payload.get("std", 0.0)),
            minimum=float(payload.get("min", 0.0)),
            maximum=float(payload.get("max", 0.0)),
            count=int(payload.get("count", 0)),
        )


@dataclass(frozen=True)
class FeatureStats:
    mean: float
    std: float
    minimum: float
    maximum: float
    count: int

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.minimum,
            "max": self.maximum,
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureStats":
        return cls(
            mean=float(payload.get("mean", 0.0)),
            std=float(payload.get("std", 0.0)),
            minimum=float(payload.get("min", 0.0)),
            maximum=float(payload.get("max", 0.0)),
            count=int(payload.get("count", 0)),
        )


@dataclass(frozen=True)
class TrainResult:
    model_id: str
    model_path: Path
    metrics_path: Path
    meta_path: Path
    dataset_snapshot_id: str
    dataset_hash: str
    row_count: int
    feature_schema_version: str
    compute_backend_preference: str = BACKEND_PREFERENCE_AUTO
    compute_backend_resolved: str = COMPUTE_BACKEND_CPU
    compute_backend_fallback_reason: str | None = None
    token_learner_backend: str = TOKEN_LEARNER_RESIDUAL_MEAN
    token_learner_fallback_reason: str | None = None


@dataclass(frozen=True)
class EvaluationResult:
    row_count: int
    metric_mae: Mapping[str, float]
    pass_probability: Mapping[str, float]
    metric_median_ae: Mapping[str, float] = field(default_factory=dict)
    metric_trimmed_mae: Mapping[str, float] = field(default_factory=dict)
    metric_mae_log1p: Mapping[str, float] = field(default_factory=dict)
    metric_mae_all: Mapping[str, float] = field(default_factory=dict)
    metric_mae_pass: Mapping[str, float] = field(default_factory=dict)
    metric_mae_log1p_pass: Mapping[str, float] = field(default_factory=dict)
    classifier_metrics: Mapping[str, float | None] = field(default_factory=dict)
    slack_regressor_metrics: Mapping[str, float | None] = field(default_factory=dict)
    labeled_count_pass: int = 0
    gate_slack_metrics: Mapping[str, float | None] = field(default_factory=dict)


if nn is not None:

    class SlackRegressor(nn.Module):
        """Predicts min_gate_slack (continuous)"""

        def __init__(self, input_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),  # Predict scalar slack
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

else:

    class SlackRegressor:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any):
            pass

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return None

        def to(self, *args: Any, **kwargs: Any) -> "SlackRegressor":
            return self

        def load_state_dict(self, *args: Any, **kwargs: Any) -> None:
            pass

        def eval(self) -> None:
            pass

        def train(self) -> None:
            pass

        def state_dict(self) -> dict[str, Any]:
            return {}

        def parameters(self) -> Iterable[Any]:
            return []


@dataclass
class SurrogateModel:
    model_id: str
    dataset_snapshot_id: str
    feature_schema_version: str
    global_metrics: Mapping[str, MetricSummary]
    main_skill_metrics: Mapping[str, Mapping[str, float]]
    feature_stats: Mapping[str, FeatureStats]
    feature_weights: Mapping[str, Mapping[str, float]]
    pass_metric: str
    backend: str
    backend_version: str
    compute_backend: str
    token_learner_backend: str
    trained_at_utc: str
    identity_token_effects: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    identity_cross_token_effects: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    target_transforms: Mapping[str, str] = field(default_factory=dict)
    classifier: Mapping[str, Any] | None = None
    slack_regressor: Mapping[str, Any] | None = None

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "model_id": self.model_id,
            "dataset_snapshot_id": self.dataset_snapshot_id,
            "feature_schema_version": self.feature_schema_version,
            "global_metrics": {
                key: summary.to_dict() for key, summary in self.global_metrics.items()
            },
            "main_skill_metrics": self.main_skill_metrics,
            "feature_stats": {key: stats.to_dict() for key, stats in self.feature_stats.items()},
            "feature_weights": self.feature_weights,
            "identity_token_effects": self.identity_token_effects,
            "identity_cross_token_effects": self.identity_cross_token_effects,
            "pass_metric": self.pass_metric,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "compute_backend": self.compute_backend,
            "token_learner_backend": self.token_learner_backend,
            "trained_at_utc": self.trained_at_utc,
            "target_transforms": self.target_transforms,
            "classifier": self.classifier,
            "slack_regressor": self.slack_regressor,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SurrogateModel":
        global_metrics = {
            key: MetricSummary.from_dict(values)
            for key, values in payload.get("global_metrics", {}).items()
        }
        feature_stats = {
            key: FeatureStats.from_dict(values)
            for key, values in payload.get("feature_stats", {}).items()
            if isinstance(values, Mapping)
        }
        feature_weights: dict[str, dict[str, float]] = {}
        raw_weights = payload.get("feature_weights", {})
        if isinstance(raw_weights, Mapping):
            for metric, features in raw_weights.items():
                if not isinstance(features, Mapping):
                    continue
                feature_weights[str(metric)] = {
                    str(feature): float(value)
                    for feature, value in features.items()
                    if value is not None
                }

        identity_token_effects = _parse_token_effects(payload.get("identity_token_effects"))
        identity_cross_token_effects = _parse_token_effects(
            payload.get("identity_cross_token_effects")
        )
        target_transforms = _parse_target_transforms(payload.get("target_transforms"))
        classifier_payload = payload.get("classifier")
        classifier: Mapping[str, Any] | None = (
            dict(classifier_payload) if isinstance(classifier_payload, Mapping) else None
        )
        slack_regressor_payload = payload.get("slack_regressor")
        slack_regressor: Mapping[str, Any] | None = (
            dict(slack_regressor_payload) if isinstance(slack_regressor_payload, Mapping) else None
        )

        return cls(
            model_id=str(payload.get("model_id", "")),
            dataset_snapshot_id=str(payload.get("dataset_snapshot_id", "")),
            feature_schema_version=str(payload.get("feature_schema_version", "")),
            global_metrics=global_metrics,
            main_skill_metrics={
                str(skill): {metric: float(value) for metric, value in metrics.items()}
                for skill, metrics in payload.get("main_skill_metrics", {}).items()
            },
            feature_stats=feature_stats,
            feature_weights=feature_weights,
            identity_token_effects=identity_token_effects,
            identity_cross_token_effects=identity_cross_token_effects,
            pass_metric=str(payload.get("pass_metric", PASS_METRIC)),
            backend=str(payload.get("backend", MODEL_BACKEND)),
            backend_version=str(payload.get("backend_version", MODEL_VERSION)),
            compute_backend=str(payload.get("compute_backend", COMPUTE_BACKEND_CPU)),
            token_learner_backend=str(
                payload.get("token_learner_backend", TOKEN_LEARNER_RESIDUAL_MEAN)
            ),
            trained_at_utc=str(payload.get("trained_at_utc", "")),
            target_transforms=target_transforms,
            classifier=classifier,
            slack_regressor=slack_regressor,
        )

    def predict_many(self, rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        results: list[Mapping[str, Any]] = []
        for row in rows:
            metrics = {metric: self._predict_metric(row, metric) for metric in METRIC_TARGETS}
            classifier_probability = self._classifier_probability(row)
            slack_prediction = self._slack_prediction(row)
            results.append(
                {
                    "metrics": metrics,
                    "pass_probability": (
                        classifier_probability
                        if classifier_probability is not None
                        else self._pass_probability(metrics.get(self.pass_metric))
                    ),
                    "min_gate_slack": slack_prediction,
                }
            )
        return results

    def _predict_metric(self, row: Mapping[str, Any], metric: str) -> float | None:
        skill_key = _main_skill_key(row)
        skill_metrics = self.main_skill_metrics.get(skill_key)
        baseline: float | None = None
        if skill_metrics:
            value = skill_metrics.get(metric)
            if value is not None:
                baseline = value
        if baseline is None:
            summary = self.global_metrics.get(metric)
            if not summary or summary.count == 0:
                return None
            baseline = summary.mean
        adjusted = self._apply_feature_adjustment(row, metric, baseline)
        identity_effects = self.identity_token_effects.get(metric, {})
        for token in _token_sequence(row.get(FEATURE_IDENTITY_TOKENS)):
            effect = identity_effects.get(token)
            if effect is not None:
                adjusted += effect
        cross_effects = self.identity_cross_token_effects.get(metric, {})
        for token in _token_sequence(row.get(FEATURE_IDENTITY_CROSS_TOKENS)):
            effect = cross_effects.get(token)
            if effect is not None:
                adjusted += effect
        summary = self.global_metrics.get(metric)
        bounded = _clamp_transformed_prediction(adjusted, summary)
        raw_prediction = _apply_inverse_target_transform(metric, bounded, self.target_transforms)
        if raw_prediction is not None:
            return raw_prediction
        fallback_value = summary.mean if summary and summary.count > 0 else bounded
        return _apply_inverse_target_transform(metric, fallback_value, self.target_transforms)

    def _apply_feature_adjustment(
        self, row: Mapping[str, Any], metric: str, baseline: float
    ) -> float:
        weights = self.feature_weights.get(metric)
        if not weights:
            return baseline
        stats = self.feature_stats
        adjustment = 0.0
        for feature, weight in weights.items():
            if not weight:
                continue
            feature_value = _to_number(row.get(feature))
            if feature_value is None:
                continue
            summary = stats.get(feature)
            mean_value = summary.mean if summary else 0.0
            adjustment += weight * (feature_value - mean_value)
        return baseline + adjustment

    def _pass_probability(self, value: float | None) -> float:
        summary = self.global_metrics.get(self.pass_metric)
        if not summary or summary.count == 0:
            return 0.0
        target = _apply_target_transform(self.pass_metric, value, self.target_transforms)
        if target is None:
            target = summary.mean
        denom = summary.std if summary.std > 1e-9 else 1.0
        normalized = (target - summary.mean) / denom
        return _sigmoid(normalized)

    def _classifier_probability(self, row: Mapping[str, Any]) -> float | None:
        payload = self.classifier
        if not isinstance(payload, Mapping):
            return None
        if str(payload.get("backend", "")) != CLASSIFIER_BACKEND:
            return None
        signal_features = payload.get("signal_features")
        if not isinstance(signal_features, Sequence) or isinstance(signal_features, (str, bytes)):
            signal_features = FEATURE_SIGNAL_KEYS
        signal_feature_keys = [str(feature) for feature in signal_features]
        weights_payload = payload.get("weights")
        if not isinstance(weights_payload, Sequence) or isinstance(weights_payload, (str, bytes)):
            return None
        weights = [float(weight) for weight in weights_payload]
        intercept = float(payload.get("intercept", 0.0))
        identity_hash_dim = int(payload.get("identity_hash_dim", CLASSIFIER_HASH_DIM))
        cross_hash_dim = int(payload.get("cross_hash_dim", CLASSIFIER_HASH_DIM))
        features = _classifier_feature_vector(
            row,
            signal_feature_keys=signal_feature_keys,
            identity_hash_dim=max(0, identity_hash_dim),
            cross_hash_dim=max(0, cross_hash_dim),
        )
        logit = intercept
        for index, value in features.items():
            if 0 <= index < len(weights):
                logit += weights[index] * value
        return _sigmoid(logit)

    def _slack_prediction(self, row: Mapping[str, Any]) -> float | None:
        payload = self.slack_regressor
        if not isinstance(payload, Mapping) or torch is None or nn is None:
            return None

        input_dim = payload.get("input_dim")
        state_dict_raw = payload.get("state_dict")
        if not input_dim or not isinstance(state_dict_raw, Mapping):
            return None

        signal_features = payload.get("signal_features")
        if not isinstance(signal_features, Sequence) or isinstance(signal_features, (str, bytes)):
            signal_features = FEATURE_SIGNAL_KEYS
        signal_feature_keys = [str(feature) for feature in signal_features]

        identity_hash_dim = int(payload.get("identity_hash_dim", CLASSIFIER_HASH_DIM))
        cross_hash_dim = int(payload.get("cross_hash_dim", CLASSIFIER_HASH_DIM))

        # Reconstruct model
        model = SlackRegressor(input_dim)
        state_dict = {k: torch.tensor(v) for k, v in state_dict_raw.items()}
        model.load_state_dict(state_dict)
        model.eval()

        features = _classifier_feature_vector(
            row,
            signal_feature_keys=signal_feature_keys,
            identity_hash_dim=max(0, identity_hash_dim),
            cross_hash_dim=max(0, cross_hash_dim),
        )

        x = torch.zeros((1, input_dim), dtype=torch.float32)
        for idx, val in features.items():
            if 0 <= idx < input_dim:
                x[0, idx] = val

        with torch.no_grad():
            prediction = model(x)
            return float(prediction.item())


def _normalize_niche_label(value: Any) -> str:
    if value is None:
        return DETERMINISTIC_BALANCE_UNKNOWN_LABEL
    normalized = str(value).strip()
    return normalized.lower() if normalized else DETERMINISTIC_BALANCE_UNKNOWN_LABEL


def _niche_key_from_row(row: Mapping[str, Any]) -> tuple[str, str]:
    return (
        _normalize_niche_label(row.get("class")),
        _normalize_niche_label(row.get("main_skill_package")),
    )


def _serialize_niche_key(key: tuple[str, str]) -> str:
    return f"{key[0]}|{key[1]}"


def _group_rows_by_niche(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = _niche_key_from_row(row)
        groups.setdefault(key, []).append(row)
    return groups


def _serialize_niche_counts(
    counts: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
) -> dict[str, int]:
    return {_serialize_niche_key(key): len(values) for key, values in sorted(counts.items())}


def _balance_row_rank(row: Mapping[str, Any]) -> int:
    build_id = str(row.get("build_id") or "")
    scenario_id = str(row.get("scenario_id") or "")
    balance_str = f"{build_id}|{scenario_id}|{DETERMINISTIC_BALANCE_HASH_SALT}"
    digest = hashlib.sha256(balance_str.encode("utf-8")).hexdigest()
    return int(digest, 16)


def _apply_deterministic_balance(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    original_rows = list(rows)
    total_rows = len(original_rows)
    counts = _group_rows_by_niche(original_rows)
    niche_counts_before = _serialize_niche_counts(counts)
    eligible_niches = {
        key: values
        for key, values in counts.items()
        if len(values) >= DETERMINISTIC_BALANCE_MIN_NICHE_COUNT
    }
    eligible_count = len(eligible_niches)
    dominant_count = max((len(group) for group in counts.values()), default=0)
    dominant_share = (dominant_count / total_rows) if total_rows else 0.0
    final_rows = original_rows
    gating_passed = False
    fallback = False
    cap_value: int | None = None
    reason = ""
    if total_rows < DETERMINISTIC_BALANCE_MIN_TOTAL_ROWS:
        reason = f"total rows {total_rows} below minimum {DETERMINISTIC_BALANCE_MIN_TOTAL_ROWS}"
    elif eligible_count < 2:
        reason = f"eligible niches {eligible_count} below minimum 2"
    elif dominant_share <= DETERMINISTIC_BALANCE_DOMINANT_SHARE:
        reason = f"dominant share {dominant_share:.3f} <= {DETERMINISTIC_BALANCE_DOMINANT_SHARE}"
    else:
        gating_passed = True
        eligible_counts = [len(values) for values in eligible_niches.values()]
        median_value = median(eligible_counts)
        cap_value = max(
            DETERMINISTIC_BALANCE_CAP_FLOOR,
            floor(median_value * DETERMINISTIC_BALANCE_MEDIAN_MULTIPLIER),
        )
        truncated_rows: list[Mapping[str, Any]] = []
        for key, group in counts.items():
            if key in eligible_niches and len(group) > cap_value:
                selected = sorted(group, key=_balance_row_rank)[:cap_value]
            else:
                selected = list(group)
            truncated_rows.extend(selected)
        output_count = len(truncated_rows)
        retention_ratio = output_count / total_rows if total_rows else 0.0
        if output_count < DETERMINISTIC_BALANCE_MIN_OUTPUT_ROWS:
            fallback = True
            reason = (
                f"balanced rows {output_count} below minimum "
                f"{DETERMINISTIC_BALANCE_MIN_OUTPUT_ROWS}; reverting to original dataset"
            )
        elif retention_ratio < DETERMINISTIC_BALANCE_MIN_RETENTION_RATIO:
            fallback = True
            reason = (
                f"retention ratio {retention_ratio:.3f} below threshold "
                f"{DETERMINISTIC_BALANCE_MIN_RETENTION_RATIO}; reverting to original dataset"
            )
        else:
            reason = f"dominant share {dominant_share:.3f} > {DETERMINISTIC_BALANCE_DOMINANT_SHARE}"
        final_rows = original_rows if fallback else truncated_rows
    final_count = len(final_rows)
    retention_ratio = final_count / total_rows if total_rows else 0.0
    dropped_row_count = total_rows - final_count
    applied = gating_passed and not fallback and dropped_row_count > 0
    niche_counts_after = _serialize_niche_counts(_group_rows_by_niche(final_rows))
    metadata = {
        "version": DETERMINISTIC_BALANCE_VERSION,
        "seed": DETERMINISTIC_BALANCE_SEED,
        "key_fields": list(DETERMINISTIC_BALANCE_KEY_FIELDS),
        "thresholds": {
            "min_total_rows": DETERMINISTIC_BALANCE_MIN_TOTAL_ROWS,
            "min_niche_count": DETERMINISTIC_BALANCE_MIN_NICHE_COUNT,
            "dominant_share": DETERMINISTIC_BALANCE_DOMINANT_SHARE,
            "min_output_rows": DETERMINISTIC_BALANCE_MIN_OUTPUT_ROWS,
            "min_retention_ratio": DETERMINISTIC_BALANCE_MIN_RETENTION_RATIO,
            "cap_floor": DETERMINISTIC_BALANCE_CAP_FLOOR,
        },
        "applied": applied,
        "reason": reason,
        "input_row_count": total_rows,
        "output_row_count": final_count,
        "dropped_row_count": dropped_row_count,
        "retention_ratio": retention_ratio,
        "eligible_niche_count": eligible_count,
        "cap_per_niche": cap_value if gating_passed else None,
        "dominant_share_before": dominant_share,
        "niche_counts_before": niche_counts_before,
        "niche_counts_after": niche_counts_after,
    }
    return final_rows, metadata


def train(
    dataset_path: Path | str,
    output_root: Path | str,
    model_id: str | None = None,
    compute_backend: str = BACKEND_PREFERENCE_AUTO,
    include_failures: bool = True,
    train_rows: Sequence[Mapping[str, Any]] | None = None,
    val_rows: Sequence[Mapping[str, Any]] | None = None,
    split_mod: int = DEFAULT_VAL_SPLIT_MOD,
    split_remainder: int = DEFAULT_VAL_SPLIT_REMAINDER,
    split_fn: Callable[[Mapping[str, Any]], bool] | None = None,
) -> TrainResult:
    train_started_at = monotonic()
    dataset_root = resolve_snapshot_root(Path(dataset_path))
    manifest = _load_manifest(dataset_root)
    rows = list(_load_dataset_rows(dataset_root))
    raw_row_count = len(rows)
    rows, balance_meta = _apply_deterministic_balance(rows)
    resolved_train_rows, resolved_val_rows = _resolve_train_val_split(
        rows,
        train_rows=train_rows,
        val_rows=val_rows,
        split_mod=split_mod,
        split_remainder=split_remainder,
        split_fn=split_fn,
    )
    regression_rows = [row for row in resolved_train_rows if row.get("gate_pass") is True]
    if not regression_rows:
        regression_rows = list(resolved_train_rows)

    model_id = model_id or _default_model_id()
    backend_preference = _normalize_backend_preference(compute_backend)
    resolved_backend, fallback_reason = _resolve_compute_backend(backend_preference)
    model_root = Path(output_root) / model_id
    _train_phase_log(
        "surrogate train %s phase=start "
        "(snapshot=%s rows=%d raw_rows=%d train_rows=%d val_rows=%d "
        "backend_preference=%s backend_resolved=%s)",
        model_id,
        manifest.get("snapshot_id", ""),
        len(regression_rows),
        raw_row_count,
        len(resolved_train_rows),
        len(resolved_val_rows),
        backend_preference,
        resolved_backend,
    )
    if fallback_reason:
        logger.warning(
            "surrogate train %s compute backend fallback: %s",
            model_id,
            fallback_reason,
        )

    feature_stage_started_at = monotonic()
    global_accumulators = {metric: _MetricAccumulator() for metric in METRIC_TARGETS}
    feature_accumulators = {feature: _MetricAccumulator() for feature in FEATURE_SIGNAL_KEYS}
    feature_metric_pairs = {
        metric: {feature: [] for feature in FEATURE_SIGNAL_KEYS} for metric in METRIC_TARGETS
    }
    for row in regression_rows:
        for metric in METRIC_TARGETS:
            global_accumulators[metric].add(
                _apply_target_transform(metric, row.get(metric), DEFAULT_TARGET_TRANSFORMS)
            )
        for feature in FEATURE_SIGNAL_KEYS:
            feature_accumulators[feature].add(_to_number(row.get(feature)))
        for metric in METRIC_TARGETS:
            metric_value = _apply_target_transform(
                metric, row.get(metric), DEFAULT_TARGET_TRANSFORMS
            )
            if metric_value is None:
                continue
            for feature in FEATURE_SIGNAL_KEYS:
                feature_value = _to_number(row.get(feature))
                if feature_value is None:
                    continue
                feature_metric_pairs[metric][feature].append((feature_value, metric_value))
    global_metrics = {
        metric: accumulator.summary() for metric, accumulator in global_accumulators.items()
    }

    feature_summaries = {
        feature: accumulator.summary() for feature, accumulator in feature_accumulators.items()
    }
    feature_stats = {
        feature: FeatureStats(
            mean=summary.mean,
            std=summary.std,
            minimum=summary.minimum,
            maximum=summary.maximum,
            count=summary.count,
        )
        for feature, summary in feature_summaries.items()
    }

    main_skill_metrics = _aggregate_main_skill_metrics(
        regression_rows,
        target_transforms=DEFAULT_TARGET_TRANSFORMS,
    )
    feature_weights: dict[str, dict[str, float]] = {}
    for metric in METRIC_TARGETS:
        weights_for_metric: dict[str, float] = {}
        for feature in FEATURE_SIGNAL_KEYS:
            pairs = feature_metric_pairs[metric][feature]
            if len(pairs) < 2:
                weights_for_metric[feature] = 0.0
                continue
            feature_values = [value for value, _ in pairs]
            metric_values = [value for _, value in pairs]
            weights_for_metric[feature] = _estimate_linear_weight(
                feature_values=feature_values,
                metric_values=metric_values,
                backend=resolved_backend,
            )
        feature_weights[metric] = weights_for_metric
    _train_phase_log(
        "surrogate train %s phase=feature_regression_complete (rows=%d elapsed=%.1fs)",
        model_id,
        len(regression_rows),
        max(0.0, monotonic() - feature_stage_started_at),
    )

    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    model_kwargs = {
        "model_id": model_id,
        "dataset_snapshot_id": manifest.get("snapshot_id", ""),
        "feature_schema_version": manifest.get("feature_schema_version", FEATURE_SCHEMA_VERSION),
        "global_metrics": global_metrics,
        "main_skill_metrics": main_skill_metrics,
        "feature_stats": feature_stats,
        "feature_weights": feature_weights,
        "pass_metric": PASS_METRIC,
        "backend": MODEL_BACKEND,
        "backend_version": MODEL_VERSION,
        "compute_backend": resolved_backend,
        "trained_at_utc": trained_at,
        "target_transforms": DEFAULT_TARGET_TRANSFORMS,
    }
    base_model = SurrogateModel(
        **model_kwargs,
        identity_token_effects={},
        identity_cross_token_effects={},
        token_learner_backend=TOKEN_LEARNER_RESIDUAL_MEAN,
    )
    token_stage_started_at = monotonic()
    baseline_predictions = base_model.predict_many(regression_rows)
    (
        identity_effects,
        cross_effects,
        token_learner_backend,
        token_learner_fallback_reason,
    ) = _learn_token_effects(regression_rows, baseline_predictions, resolved_backend)
    classifier_payload: Mapping[str, Any] | None = None
    model = SurrogateModel(
        **model_kwargs,
        identity_token_effects=identity_effects,
        identity_cross_token_effects=cross_effects,
        token_learner_backend=token_learner_backend,
        classifier=classifier_payload,
    )
    _train_phase_log(
        "surrogate train %s phase=token_effects_complete "
        "(token_backend=%s token_fallback=%s elapsed=%.1fs)",
        model_id,
        token_learner_backend,
        token_learner_fallback_reason or "none",
        max(0.0, monotonic() - token_stage_started_at),
    )

    classifier_model = None
    classifier_metrics = None
    slack_regressor_payload = None
    slack_regressor_metrics = None
    classifier_stage_started_at: float | None = None
    classifier_meta = {
        "classifier_enabled": include_failures,
        "classifier_status": "disabled" if not include_failures else "skipped",
        "classifier_skip_reason": ("include_failures disabled" if not include_failures else None),
        "classifier_train_samples": 0,
        "classifier_label_distribution": {},
        "classifier_cv_requested_folds": 0,
        "classifier_cv_used_folds": 0,
        "classifier_cv_status": "skipped",
        "classifier_cv_skip_reason": "not implemented for sgd logistic",
        "slack_regressor_status": "disabled" if not include_failures else "skipped",
        "slack_regressor_skip_reason": (
            "include_failures disabled" if not include_failures else None
        ),
    }
    if include_failures:
        classifier_stage_started_at = monotonic()
        _train_phase_log(
            "surrogate train %s phase=classifier_start (include_failures=true)",
            model_id,
        )
        classification_rows = [
            row for row in resolved_train_rows if row.get("gate_pass") is not None
        ]
        classifier_meta["classifier_train_samples"] = len(classification_rows)
        if not classification_rows:
            classifier_meta["classifier_skip_reason"] = "no gate_pass labels available"
            classifier_meta["slack_regressor_skip_reason"] = "no gate_pass labels available"
        else:
            labels = [1 if row.get("gate_pass") else 0 for row in classification_rows]
            label_counts = Counter(labels)
            classifier_meta["classifier_label_distribution"] = {
                str(key): int(value) for key, value in sorted(label_counts.items())
            }
            if len(label_counts) < 2:
                classifier_meta["classifier_skip_reason"] = (
                    "single-class gate_pass labels: "
                    + ", ".join(f"{label}={count}" for label, count in sorted(label_counts.items()))
                )
            else:
                try:
                    classifier_payload, classifier_metrics = _train_logistic_classifier(
                        classification_rows,
                        signal_feature_keys=list(FEATURE_SIGNAL_KEYS),
                        identity_hash_dim=CLASSIFIER_HASH_DIM,
                        cross_hash_dim=CLASSIFIER_HASH_DIM,
                        epochs=CLASSIFIER_EPOCHS,
                        learning_rate=CLASSIFIER_LEARNING_RATE,
                        l2=CLASSIFIER_L2,
                        lr_decay=CLASSIFIER_LR_DECAY,
                    )
                except ValueError as exc:
                    classifier_meta["classifier_skip_reason"] = str(exc)
                else:
                    classifier_model = CLASSIFIER_BACKEND
                    classifier_meta["classifier_status"] = "trained"
                    classifier_meta["classifier_skip_reason"] = None

            # Train Slack Regressor
            if torch is not None:
                try:
                    slack_regressor_payload, slack_regressor_metrics = _train_slack_regressor(
                        classification_rows,
                        signal_feature_keys=list(FEATURE_SIGNAL_KEYS),
                        identity_hash_dim=CLASSIFIER_HASH_DIM,
                        cross_hash_dim=CLASSIFIER_HASH_DIM,
                        compute_backend=resolved_backend,
                    )
                except Exception as exc:
                    classifier_meta["slack_regressor_skip_reason"] = str(exc)
                    classifier_meta["slack_regressor_status"] = "failed"
                else:
                    classifier_meta["slack_regressor_status"] = "trained"
                    classifier_meta["slack_regressor_skip_reason"] = None
            else:
                classifier_meta["slack_regressor_skip_reason"] = "torch not available"

            model = SurrogateModel(
                **model_kwargs,
                identity_token_effects=identity_effects,
                identity_cross_token_effects=cross_effects,
                token_learner_backend=token_learner_backend,
                classifier=classifier_payload,
                slack_regressor=slack_regressor_payload,
            )

    classifier_elapsed = (
        max(0.0, monotonic() - classifier_stage_started_at)
        if classifier_stage_started_at is not None
        else 0.0
    )
    _train_phase_log(
        "surrogate train %s phase=classifier_complete "
        "(status=%s cv_status=%s samples=%d elapsed=%.1fs reason=%s)",
        model_id,
        classifier_meta.get("classifier_status"),
        classifier_meta.get("classifier_cv_status"),
        classifier_meta.get("classifier_train_samples"),
        classifier_elapsed,
        classifier_meta.get("classifier_skip_reason") or "none",
    )

    evaluation_stage_started_at = monotonic()
    _train_phase_log("surrogate train %s phase=evaluation_start", model_id)
    evaluation_rows = list(resolved_val_rows)
    if not evaluation_rows:
        evaluation_rows = list(resolved_train_rows)
    predictions = model.predict_many(evaluation_rows)
    evaluation = evaluate_predictions(evaluation_rows, predictions)
    metrics_payload = {
        "row_count": evaluation.row_count,
        "metric_mae": evaluation.metric_mae,
        "metric_mae_all": evaluation.metric_mae_all,
        "metric_mae_pass": evaluation.metric_mae_pass,
        "metric_mae_log1p": evaluation.metric_mae_log1p,
        "metric_mae_log1p_pass": evaluation.metric_mae_log1p_pass,
        "pass_probability": evaluation.pass_probability,
        "classifier_metrics": evaluation.classifier_metrics,
        "slack_regressor_metrics": evaluation.slack_regressor_metrics,
        "split": {
            "train_rows": len(resolved_train_rows),
            "val_rows": len(resolved_val_rows),
            "regression_rows": len(regression_rows),
            "split_mod": split_mod,
            "split_remainder": split_remainder,
        },
    }
    _train_phase_log(
        "surrogate train %s phase=evaluation_complete (rows=%d elapsed=%.1fs)",
        model_id,
        evaluation.row_count,
        max(0.0, monotonic() - evaluation_stage_started_at),
    )

    meta_payload = {
        "model_id": model_id,
        "dataset_snapshot_id": manifest.get("snapshot_id", ""),
        "dataset_hash": manifest.get("dataset_hash", ""),
        "dataset_row_count": manifest.get("row_count", len(rows)),
        "feature_schema_version": manifest.get("feature_schema_version", FEATURE_SCHEMA_VERSION),
        "model_backend": MODEL_BACKEND,
        "model_backend_version": MODEL_VERSION,
        "compute_backend_preference": backend_preference,
        "compute_backend_resolved": resolved_backend,
        "compute_backend_fallback_reason": fallback_reason,
        "token_learner_backend": token_learner_backend,
        "token_learner_fallback_reason": token_learner_fallback_reason,
        "trained_at_utc": trained_at,
        "source_snapshot_path": str(dataset_root),
    }
    meta_payload.update(classifier_meta)
    meta_payload["deterministic_balance"] = balance_meta
    meta_payload["split"] = {
        "train_rows": len(resolved_train_rows),
        "val_rows": len(resolved_val_rows),
        "regression_rows": len(regression_rows),
        "split_mod": split_mod,
        "split_remainder": split_remainder,
    }
    if classifier_model and classifier_metrics:
        meta_payload["classifier"] = classifier_model
        meta_payload["classifier_metrics"] = classifier_metrics
    if slack_regressor_payload and slack_regressor_metrics:
        meta_payload["slack_regressor"] = "torch_mlp_v1"
        meta_payload["slack_regressor_metrics"] = slack_regressor_metrics

    model_path, metrics_path, meta_path = _write_model_artifacts_atomic(
        model_root,
        model_payload=model.to_dict(),
        metrics_payload=metrics_payload,
        meta_payload=meta_payload,
    )

    _train_phase_log(
        "surrogate train %s phase=completed (backend=%s token_backend=%s elapsed=%.1fs)",
        model_id,
        resolved_backend,
        token_learner_backend,
        max(0.0, monotonic() - train_started_at),
    )

    return TrainResult(
        model_id=model_id,
        model_path=model_path,
        metrics_path=metrics_path,
        meta_path=meta_path,
        dataset_snapshot_id=manifest.get("snapshot_id", ""),
        dataset_hash=manifest.get("dataset_hash", ""),
        row_count=evaluation.row_count,
        feature_schema_version=manifest.get("feature_schema_version", FEATURE_SCHEMA_VERSION),
        compute_backend_preference=backend_preference,
        compute_backend_resolved=resolved_backend,
        compute_backend_fallback_reason=fallback_reason,
        token_learner_backend=token_learner_backend,
        token_learner_fallback_reason=token_learner_fallback_reason,
    )


def load_model(model_path: Path | str) -> SurrogateModel:
    resolved = Path(model_path)
    if resolved.is_dir():
        resolved = resolved / _MODEL_FILENAME
    if not resolved.exists():
        raise FileNotFoundError(f"model file not found at {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    return SurrogateModel.from_dict(payload)


def load_dataset_rows(dataset_path: Path | str) -> list[Mapping[str, Any]]:
    dataset_root = resolve_snapshot_root(Path(dataset_path))
    return list(_load_dataset_rows(dataset_root))


def resolve_snapshot_root(path: Path | str) -> Path:
    candidate = Path(path)
    if candidate.exists() and _is_snapshot(candidate):
        return candidate
    if not candidate.exists() or not candidate.is_dir():
        raise FileNotFoundError(f"dataset path not found: {candidate}")
    snapshots = [entry for entry in sorted(candidate.iterdir()) if _is_snapshot(entry)]
    if not snapshots:
        raise FileNotFoundError(f"no dataset snapshot found under {candidate}")
    return snapshots[-1]


def evaluate_predictions(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> EvaluationResult:
    metric_errors_all: dict[str, list[float]] = {metric: [] for metric in METRIC_TARGETS}
    metric_errors_pass: dict[str, list[float]] = {metric: [] for metric in METRIC_TARGETS}
    metric_log_errors_all: dict[str, list[float]] = {metric: [] for metric in METRIC_TARGETS}
    metric_log_errors_pass: dict[str, list[float]] = {metric: [] for metric in METRIC_TARGETS}
    pass_probs: list[float] = []
    slack_errors: list[float] = []
    classifier_labels: list[int] = []
    classifier_probs: list[float] = []
    row_list = list(rows)
    for row, prediction in zip(row_list, predictions, strict=True):
        pass_probability = float(prediction.get("pass_probability", 0.0))
        pass_probs.append(min(1.0, max(0.0, pass_probability)))
        gate_pass = row.get("gate_pass")
        if isinstance(gate_pass, bool):
            classifier_labels.append(1 if gate_pass else 0)
            classifier_probs.append(pass_probs[-1])

        actual_slack = _to_number(row.get("min_gate_slack"))
        predicted_slack = _to_number(prediction.get("min_gate_slack"))
        if actual_slack is not None and predicted_slack is not None:
            slack_errors.append(abs(predicted_slack - actual_slack))

        predicted_metrics = prediction.get("metrics", {})
        is_pass_row = gate_pass is True
        for metric in METRIC_TARGETS:
            actual = _to_number(row.get(metric))
            predicted = _to_number(predicted_metrics.get(metric))
            if actual is None or predicted is None:
                continue
            metric_errors_all[metric].append(abs(predicted - actual))
            transformed_actual = _safe_log1p(actual)
            transformed_predicted = _safe_log1p(predicted)
            metric_log_errors_all[metric].append(abs(transformed_predicted - transformed_actual))
            if is_pass_row:
                metric_errors_pass[metric].append(abs(predicted - actual))
                metric_log_errors_pass[metric].append(
                    abs(transformed_predicted - transformed_actual)
                )

    metric_mae_all = {
        metric: mean(errors) if errors else 0.0 for metric, errors in metric_errors_all.items()
    }
    metric_mae_pass = {
        metric: mean(errors) if errors else 0.0 for metric, errors in metric_errors_pass.items()
    }
    metric_mae_log1p_all = {
        metric: mean(errors) if errors else 0.0 for metric, errors in metric_log_errors_all.items()
    }
    metric_mae_log1p_pass = {
        metric: mean(errors) if errors else 0.0 for metric, errors in metric_log_errors_pass.items()
    }
    pass_summary = _stat_summary(pass_probs)
    classifier_metrics = _classifier_eval_metrics(classifier_labels, classifier_probs)
    slack_regressor_metrics = {
        "mae": mean(slack_errors) if slack_errors else None,
        "count": float(len(slack_errors)),
    }
    gate_slack_metrics = {
        "min_gate_slack_mae": slack_regressor_metrics.get("mae"),
    }
    labeled_count_pass = int(classifier_metrics.get("positive_count") or 0)
    return EvaluationResult(
        row_count=len(row_list),
        metric_mae=metric_mae_all,
        pass_probability=pass_summary,
        metric_mae_all=metric_mae_all,
        metric_mae_pass=metric_mae_pass,
        metric_mae_log1p=metric_mae_log1p_all,
        metric_mae_log1p_pass=metric_mae_log1p_pass,
        classifier_metrics=classifier_metrics,
        slack_regressor_metrics=slack_regressor_metrics,
        labeled_count_pass=labeled_count_pass,
        gate_slack_metrics=gate_slack_metrics,
    )


@dataclass
class EnsembleSurrogate:
    """Ensemble of 5 surrogate models with uncertainty quantification.

    Combines predictions from multiple models to estimate both mean prediction
    and uncertainty (variance across models). Supports UCB acquisition function
    for active learning: mean + 2*std.
    """

    models: list[SurrogateModel]
    ensemble_id: str = field(default_factory=lambda: f"ensemble-{_default_model_id()}")

    def __post_init__(self) -> None:
        if len(self.models) != 5:
            raise ValueError(f"EnsembleSurrogate requires exactly 5 models, got {len(self.models)}")

    def predict_many(self, rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        """Predict with uncertainty quantification.

        Returns predictions with:
        - metrics: mean predictions across ensemble
        - metrics_std: standard deviation of predictions
        - pass_probability: mean pass probability
        - pass_probability_std: std of pass probabilities
        - ucb_full_dps: upper confidence bound for full_dps (mean + 2*std)
        - min_gate_slack: mean slack prediction
        """
        results: list[Mapping[str, Any]] = []

        for row in rows:
            # Collect predictions from all models
            model_predictions = [model.predict_many([row])[0] for model in self.models]

            # Extract metrics from each model
            metrics_by_model: dict[str, list[float]] = {metric: [] for metric in METRIC_TARGETS}
            pass_probs: list[float] = []
            slack_predictions: list[float | None] = []

            for pred in model_predictions:
                pred_metrics = pred.get("metrics", {})
                for metric in METRIC_TARGETS:
                    value = pred_metrics.get(metric)
                    if value is not None:
                        metrics_by_model[metric].append(value)

                pass_prob = pred.get("pass_probability")
                if pass_prob is not None:
                    pass_probs.append(pass_prob)

                slack = pred.get("min_gate_slack")
                slack_predictions.append(slack)

            # Compute mean and std for metrics
            metrics_mean: dict[str, float] = {}
            metrics_std: dict[str, float] = {}
            for metric in METRIC_TARGETS:
                values = metrics_by_model[metric]
                if values:
                    metrics_mean[metric] = mean(values)
                    metrics_std[metric] = pstdev(values) if len(values) > 1 else 0.0
                else:
                    metrics_mean[metric] = 0.0
                    metrics_std[metric] = 0.0

            # Compute pass probability stats
            pass_prob_mean = mean(pass_probs) if pass_probs else 0.0
            pass_prob_std = pstdev(pass_probs) if len(pass_probs) > 1 else 0.0

            # Compute slack prediction (mean of non-None values)
            valid_slacks = [s for s in slack_predictions if s is not None]
            slack_mean = mean(valid_slacks) if valid_slacks else None

            # UCB acquisition: mean + 2*std for exploration
            ucb_full_dps = metrics_mean.get("full_dps", 0.0) + 2.0 * metrics_std.get(
                "full_dps", 0.0
            )

            results.append(
                {
                    "metrics": metrics_mean,
                    "metrics_std": metrics_std,
                    "pass_probability": pass_prob_mean,
                    "pass_probability_std": pass_prob_std,
                    "ucb_full_dps": ucb_full_dps,
                    "min_gate_slack": slack_mean,
                }
            )

        return results

    def to_dict(self) -> Mapping[str, Any]:
        """Serialize ensemble to dictionary."""
        return {
            "ensemble_id": self.ensemble_id,
            "model_count": len(self.models),
            "models": [model.to_dict() for model in self.models],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EnsembleSurrogate":
        """Deserialize ensemble from dictionary."""
        models = [
            SurrogateModel.from_dict(model_payload) for model_payload in payload.get("models", [])
        ]
        return cls(
            models=models,
            ensemble_id=str(payload.get("ensemble_id", f"ensemble-{_default_model_id()}")),
        )


def _normalize_target_transform(value: Any) -> str:
    if isinstance(value, str) and value.strip().lower() == TARGET_TRANSFORM_LOG1P:
        return TARGET_TRANSFORM_LOG1P
    return TARGET_TRANSFORM_IDENTITY


def _parse_target_transforms(payload: Any | None) -> dict[str, str]:
    if not isinstance(payload, Mapping):
        return {str(metric): TARGET_TRANSFORM_IDENTITY for metric in METRIC_TARGETS}
    transforms = {
        str(metric): _normalize_target_transform(
            DEFAULT_TARGET_TRANSFORMS.get(metric, TARGET_TRANSFORM_IDENTITY)
        )
        for metric in METRIC_TARGETS
    }
    for metric, transform in payload.items():
        metric_name = str(metric)
        transforms[metric_name] = _normalize_target_transform(transform)
    return transforms


def _target_transform_name(metric: str, target_transforms: Mapping[str, str] | None = None) -> str:
    if target_transforms and metric in target_transforms:
        return _normalize_target_transform(target_transforms.get(metric))
    return TARGET_TRANSFORM_IDENTITY


def _apply_target_transform(
    metric: str,
    value: Any,
    target_transforms: Mapping[str, str] | None = None,
) -> float | None:
    numeric = _to_number(value)
    if numeric is None or not isfinite(numeric):
        return None
    transform_name = _target_transform_name(metric, target_transforms)
    if transform_name == TARGET_TRANSFORM_LOG1P:
        if numeric <= -1.0:
            return None
        return log1p(max(0.0, numeric))
    return numeric


def _apply_inverse_target_transform(
    metric: str,
    value: float | None,
    target_transforms: Mapping[str, str] | None = None,
) -> float | None:
    if value is None or not isfinite(value):
        return None
    transform_name = _target_transform_name(metric, target_transforms)
    if transform_name == TARGET_TRANSFORM_LOG1P:
        raw = expm1(value)
        if not isfinite(raw):
            return None
        return max(0.0, raw)
    return value


def _clamp_transformed_prediction(value: float, summary: MetricSummary | None) -> float:
    if summary and summary.count > 0:
        lower = min(summary.minimum, summary.maximum)
        upper = max(summary.minimum, summary.maximum)
        fallback = summary.mean
    else:
        lower = value
        upper = value
        fallback = 0.0
    bounded = value if isfinite(value) else fallback
    return min(upper, max(lower, bounded))


def _safe_log1p(value: float) -> float:
    bounded = value if isfinite(value) else 0.0
    return log1p(max(0.0, bounded))


def _stable_hash_mod(value: str, mod: int) -> int:
    if mod <= 0:
        return 0
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest, 16) % mod


def _resolve_train_val_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    train_rows: Sequence[Mapping[str, Any]] | None,
    val_rows: Sequence[Mapping[str, Any]] | None,
    split_mod: int,
    split_remainder: int,
    split_fn: Callable[[Mapping[str, Any]], bool] | None,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    if train_rows is not None or val_rows is not None:
        resolved_train = list(train_rows or [])
        resolved_val = list(val_rows or [])
        if not resolved_train and resolved_val:
            resolved_train = list(resolved_val)
        if not resolved_val and resolved_train:
            resolved_val = list(resolved_train)
        return resolved_train, resolved_val

    source_rows = list(rows)
    if not source_rows:
        return [], []

    normalized_mod = max(1, int(split_mod))
    normalized_remainder = int(split_remainder) % normalized_mod
    resolved_train: list[Mapping[str, Any]] = []
    resolved_val: list[Mapping[str, Any]] = []
    for row in source_rows:
        if split_fn is not None:
            use_val = bool(split_fn(row))
        else:
            build_id = str(row.get("build_id") or "")
            if not build_id:
                build_id = (
                    f"{row.get('scenario_id') or ''}|"
                    f"{row.get('profile_id') or ''}|"
                    f"{row.get('main_skill_package') or ''}"
                )
            use_val = _stable_hash_mod(build_id, normalized_mod) == normalized_remainder
        if use_val:
            resolved_val.append(row)
        else:
            resolved_train.append(row)

    if not resolved_val:
        resolved_val = [source_rows[0]]
        resolved_train = source_rows[1:] or list(source_rows)
    if not resolved_train:
        resolved_train = list(source_rows)
    return resolved_train, resolved_val


def _hash_bucket(value: str, mod: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest, 16) % mod


def _classifier_feature_vector(
    row: Mapping[str, Any],
    *,
    signal_feature_keys: Sequence[str],
    identity_hash_dim: int,
    cross_hash_dim: int,
) -> dict[int, float]:
    vector: dict[int, float] = {}
    for index, feature in enumerate(signal_feature_keys):
        numeric = _to_number(row.get(feature))
        if numeric is None or not isfinite(numeric) or numeric == 0.0:
            continue
        vector[index] = numeric

    offset = len(signal_feature_keys)
    if identity_hash_dim > 0:
        for token in _token_sequence(row.get(FEATURE_IDENTITY_TOKENS)):
            bucket = _hash_bucket(f"identity:{token}", identity_hash_dim)
            feature_index = offset + bucket
            vector[feature_index] = vector.get(feature_index, 0.0) + 1.0
    offset += max(0, identity_hash_dim)
    if cross_hash_dim > 0:
        for token in _token_sequence(row.get(FEATURE_IDENTITY_CROSS_TOKENS)):
            bucket = _hash_bucket(f"cross:{token}", cross_hash_dim)
            feature_index = offset + bucket
            vector[feature_index] = vector.get(feature_index, 0.0) + 1.0

    return vector


def _train_slack_regressor(
    rows: Sequence[Mapping[str, Any]],
    *,
    signal_feature_keys: Sequence[str],
    identity_hash_dim: int,
    cross_hash_dim: int,
    epochs: int = 100,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    compute_backend: str = COMPUTE_BACKEND_CPU,
) -> tuple[dict[str, Any], dict[str, float]]:
    if torch is None or nn is None:
        raise ImportError("torch is required for SlackRegressor")

    device = torch.device("cuda" if compute_backend == COMPUTE_BACKEND_CUDA else "cpu")

    # Filter rows that have min_gate_slack
    valid_rows = []
    targets = []
    for row in rows:
        slack = _to_number(row.get("min_gate_slack"))
        if slack is not None:
            valid_rows.append(row)
            targets.append(slack)

    if not valid_rows:
        raise ValueError("no rows with min_gate_slack available for regression")

    input_dim = len(signal_feature_keys) + max(0, identity_hash_dim) + max(0, cross_hash_dim)
    model = SlackRegressor(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Vectorize features
    vectors = [
        _classifier_feature_vector(
            row,
            signal_feature_keys=signal_feature_keys,
            identity_hash_dim=identity_hash_dim,
            cross_hash_dim=cross_hash_dim,
        )
        for row in valid_rows
    ]

    # Convert to tensors
    x = torch.zeros((len(vectors), input_dim), dtype=torch.float32, device=device)
    for i, vec in enumerate(vectors):
        for idx, val in vec.items():
            if 0 <= idx < input_dim:
                x[i, idx] = val

    y = torch.tensor(targets, dtype=torch.float32, device=device).view(-1, 1)

    model.train()
    for _epoch in range(epochs):
        permutation = torch.randperm(x.size()[0])
        for i in range(0, x.size()[0], batch_size):
            indices = permutation[i : i + batch_size]
            batch_x, batch_y = x[indices], y[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(x)
        mse = criterion(predictions, y).item()
        mae = torch.mean(torch.abs(predictions - y)).item()

    # Serialize state dict
    state_dict = {k: v.cpu().tolist() for k, v in model.state_dict().items()}

    payload = {
        "backend": "torch_mlp_v1",
        "input_dim": input_dim,
        "signal_features": list(signal_feature_keys),
        "identity_hash_dim": identity_hash_dim,
        "cross_hash_dim": cross_hash_dim,
        "state_dict": state_dict,
    }

    metrics = {
        "train_samples": float(len(valid_rows)),
        "train_mse": float(mse),
        "train_mae": float(mae),
    }

    return payload, metrics


def _train_logistic_classifier(
    rows: Sequence[Mapping[str, Any]],
    *,
    signal_feature_keys: Sequence[str],
    identity_hash_dim: int,
    cross_hash_dim: int,
    epochs: int,
    learning_rate: float,
    l2: float,
    lr_decay: float,
) -> tuple[dict[str, Any], dict[str, float]]:
    labels = [1 if row.get("gate_pass") else 0 for row in rows]
    vectors = [
        _classifier_feature_vector(
            row,
            signal_feature_keys=signal_feature_keys,
            identity_hash_dim=identity_hash_dim,
            cross_hash_dim=cross_hash_dim,
        )
        for row in rows
    ]
    feature_count = len(signal_feature_keys) + max(0, identity_hash_dim) + max(0, cross_hash_dim)
    weights = [0.0 for _ in range(feature_count)]
    intercept = 0.0

    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count <= 0 or negative_count <= 0:
        raise ValueError("classifier requires both positive and negative gate_pass labels")
    positive_weight = len(labels) / (2.0 * positive_count)
    negative_weight = len(labels) / (2.0 * negative_count)

    step_size = max(1e-6, learning_rate)
    for _ in range(max(1, epochs)):
        for label, vector in zip(labels, vectors, strict=True):
            logit = intercept
            for index, value in vector.items():
                if 0 <= index < feature_count:
                    logit += weights[index] * value
            probability = _sigmoid(logit)
            class_weight = positive_weight if label == 1 else negative_weight
            error = (probability - float(label)) * class_weight
            for index, value in vector.items():
                if 0 <= index < feature_count:
                    gradient = (error * value) + (l2 * weights[index])
                    weights[index] -= step_size * gradient
            intercept -= step_size * error
        step_size = max(1e-6, step_size * lr_decay)

    predictions: list[float] = []
    for vector in vectors:
        logit = intercept
        for index, value in vector.items():
            if 0 <= index < feature_count:
                logit += weights[index] * value
        predictions.append(_sigmoid(logit))

    eps = 1e-9
    train_log_loss = mean(
        [
            -((label * log(max(eps, prob))) + ((1 - label) * log(max(eps, 1.0 - prob))))
            for label, prob in zip(labels, predictions, strict=True)
        ]
    )
    train_brier = mean(
        [(prob - float(label)) ** 2 for label, prob in zip(labels, predictions, strict=True)]
    )
    train_auc = _roc_auc_score(labels, predictions)

    payload = {
        "backend": CLASSIFIER_BACKEND,
        "signal_features": list(signal_feature_keys),
        "identity_hash_dim": max(0, identity_hash_dim),
        "cross_hash_dim": max(0, cross_hash_dim),
        "weights": weights,
        "intercept": intercept,
    }
    metrics: dict[str, float] = {
        "train_samples": float(len(rows)),
        "train_log_loss": float(train_log_loss),
        "train_brier": float(train_brier),
    }
    if train_auc is not None:
        metrics["train_auc"] = float(train_auc)
    return payload, metrics


def _classifier_eval_metrics(
    labels: Sequence[int], probabilities: Sequence[float]
) -> dict[str, float | None]:
    if not labels:
        return {
            "brier": None,
            "auc": None,
            "labeled_count": 0.0,
            "positive_count": 0.0,
            "negative_count": 0.0,
        }
    brier = mean(
        [
            (probability - float(label)) ** 2
            for label, probability in zip(labels, probabilities, strict=True)
        ]
    )
    auc = _roc_auc_score(labels, probabilities)
    positive = sum(labels)
    negative = len(labels) - positive
    return {
        "brier": float(brier),
        "auc": float(auc) if auc is not None else None,
        "labeled_count": float(len(labels)),
        "positive_count": float(positive),
        "negative_count": float(negative),
    }


def _roc_auc_score(labels: Sequence[int], probabilities: Sequence[float]) -> float | None:
    if len(labels) != len(probabilities) or not labels:
        return None
    positive = sum(labels)
    negative = len(labels) - positive
    if positive <= 0 or negative <= 0:
        return None

    ranked = sorted(zip(probabilities, labels, strict=True), key=lambda item: item[0])
    rank = 1
    positive_rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        tie_count = end - index
        average_rank = (rank + (rank + tie_count - 1)) / 2.0
        positive_rank_sum += average_rank * sum(label for _, label in ranked[index:end])
        rank += tie_count
        index = end

    return (positive_rank_sum - (positive * (positive + 1) / 2.0)) / (positive * negative)


def _aggregate_main_skill_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_transforms: Mapping[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    groups: dict[str, dict[str, _MetricAccumulator]] = {}
    for row in rows:
        key = _main_skill_key(row)
        if key not in groups:
            groups[key] = {metric: _MetricAccumulator() for metric in METRIC_TARGETS}
        for metric in METRIC_TARGETS:
            groups[key][metric].add(
                _apply_target_transform(metric, row.get(metric), target_transforms)
            )

    result: dict[str, dict[str, float]] = {}
    for key, metrics in groups.items():
        summary: dict[str, float] = {}
        for metric, accumulator in metrics.items():
            if accumulator.count > 0:
                summary[metric] = accumulator.summary().mean
        if summary:
            result[key] = summary
    return result


def _load_manifest(snapshot_root: Path) -> Mapping[str, Any]:
    manifest_path = snapshot_root / _MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest missing at {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_dataset_rows(snapshot_root: Path) -> Iterable[Mapping[str, Any]]:
    dataset_file = snapshot_root / _DATASET_FILENAME
    if not dataset_file.exists():
        return []
    for line in dataset_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        yield json.loads(line)


def _write_model_artifacts_atomic(
    model_root: Path,
    *,
    model_payload: Mapping[str, Any],
    metrics_payload: Mapping[str, Any],
    meta_payload: Mapping[str, Any],
) -> tuple[Path, Path, Path]:
    model_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f".{model_root.name}-", dir=str(model_root.parent)))
    try:
        model_path = staging_root / _MODEL_FILENAME
        metrics_path = staging_root / _METRICS_FILENAME
        meta_path = staging_root / _META_FILENAME
        model_path.write_text(json.dumps(model_payload, indent=2), encoding="utf-8")
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
        if model_root.exists():
            shutil.rmtree(model_root)
        os.replace(staging_root, model_root)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return (
        model_root / _MODEL_FILENAME,
        model_root / _METRICS_FILENAME,
        model_root / _META_FILENAME,
    )


def _is_snapshot(path: Path) -> bool:
    return (path / _DATASET_FILENAME).exists() and (path / _MANIFEST_FILENAME).exists()


def _main_skill_key(row: Mapping[str, Any]) -> str:
    scenario = _normalize_niche_label(row.get("scenario_id"))
    profile = _normalize_niche_label(row.get("profile_id"))
    skill = _normalize_niche_label(row.get("main_skill_package"))
    return f"{scenario}|{profile}|{skill}"


def _sigmoid(value: float) -> float:
    try:
        return 1.0 / (1.0 + exp(-value))
    except OverflowError:
        return 0.0 if value < 0 else 1.0


def _stat_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _normalize_backend_preference(value: str) -> str:
    normalized = (value or BACKEND_PREFERENCE_AUTO).strip().lower()
    if normalized not in VALID_BACKEND_PREFERENCES:
        allowed = ", ".join(sorted(VALID_BACKEND_PREFERENCES))
        raise ValueError(f"invalid compute backend '{value}'; expected one of: {allowed}")
    return normalized


def _resolve_compute_backend(preference: str) -> tuple[str, str | None]:
    if preference == BACKEND_PREFERENCE_CPU:
        return COMPUTE_BACKEND_CPU, None
    if _torch_cuda_available():
        return COMPUTE_BACKEND_CUDA, None
    if preference == BACKEND_PREFERENCE_CUDA:
        return COMPUTE_BACKEND_CPU, "cuda unavailable; falling back to cpu"
    return COMPUTE_BACKEND_CPU, None


def _torch_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        return False


def _estimate_linear_weight(
    feature_values: Sequence[float],
    metric_values: Sequence[float],
    backend: str,
) -> float:
    if len(feature_values) < 2 or len(metric_values) < 2:
        return 0.0
    if backend == COMPUTE_BACKEND_CUDA:
        try:
            import torch

            device = torch.device("cuda")
            x = torch.tensor(feature_values, dtype=torch.float64, device=device)
            y = torch.tensor(metric_values, dtype=torch.float64, device=device)
            x_delta = x - torch.mean(x)
            y_delta = y - torch.mean(y)
            variance = torch.mean(x_delta * x_delta)
            variance_value = float(variance.item())
            if variance_value <= 1e-12:
                return 0.0
            covariance = torch.mean(x_delta * y_delta)
            return float((covariance / variance).item())
        except Exception:
            return _estimate_linear_weight_cpu(feature_values, metric_values)
    return _estimate_linear_weight_cpu(feature_values, metric_values)


def _estimate_linear_weight_cpu(
    feature_values: Sequence[float],
    metric_values: Sequence[float],
) -> float:
    feature_mean = mean(feature_values)
    metric_mean = mean(metric_values)
    variance = mean([(value - feature_mean) ** 2 for value in feature_values])
    if variance <= 1e-12:
        return 0.0
    covariance = mean(
        [
            (feature_value - feature_mean) * (metric_value - metric_mean)
            for feature_value, metric_value in zip(feature_values, metric_values, strict=True)
        ]
    )
    return covariance / variance


class _MetricAccumulator:
    def __init__(self) -> None:
        self._values: list[float] = []

    @property
    def count(self) -> int:
        return len(self._values)

    def add(self, value: float | None) -> None:
        if value is None:
            return
        self._values.append(value)

    def summary(self) -> MetricSummary:
        if not self._values:
            return MetricSummary(0.0, 0.0, 0.0, 0.0, 0)
        values = self._values
        return MetricSummary(
            mean(values),
            pstdev(values) if len(values) > 1 else 0.0,
            min(values),
            max(values),
            len(values),
        )


class _TokenAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value

    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0


def _accumulate_token(
    accumulators: dict[str, _TokenAccumulator],
    token: str,
    residual: float,
) -> None:
    stats = accumulators.setdefault(token, _TokenAccumulator())
    stats.add(residual)


def _token_shrinkage(count: int) -> float:
    return float(count) / (count + _TOKEN_SHRINKAGE_PRIOR)


def _select_token_effects(
    accumulators: dict[str, dict[str, _TokenAccumulator]],
    limit: int,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for metric, tokens in accumulators.items():
        entries: list[tuple[str, float]] = []
        for token, stats in tokens.items():
            if stats.count == 0:
                continue
            effect = stats.mean() * _token_shrinkage(stats.count)
            if not effect:
                continue
            entries.append((token, effect))
        if not entries:
            continue
        entries.sort(key=lambda entry: (-abs(entry[1]), entry[0]))
        results[metric] = {token: effect for token, effect in entries[:limit]}
    return results


def _learn_token_effects(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    compute_backend: str,
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    str,
    str | None,
]:
    if _torch_available_for_token_learner(compute_backend):
        try:
            identity, cross = _learn_token_effects_torch_sparse(
                rows,
                predictions,
                compute_backend,
            )
            return identity, cross, TOKEN_LEARNER_TORCH_SPARSE, None
        except Exception as exc:  # pragma: no cover - defensive fallback
            identity, cross = _learn_token_effects_residual_mean(rows, predictions)
            return (
                identity,
                cross,
                TOKEN_LEARNER_RESIDUAL_MEAN,
                f"torch token learner failed: {exc.__class__.__name__}",
            )

    identity, cross = _learn_token_effects_residual_mean(rows, predictions)
    fallback_reason: str | None = None
    if compute_backend == COMPUTE_BACKEND_CUDA:
        fallback_reason = "torch unavailable; token learner fell back to residual mean"
    return identity, cross, TOKEN_LEARNER_RESIDUAL_MEAN, fallback_reason


def _learn_token_effects_residual_mean(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    identity_accumulators = {metric: {} for metric in METRIC_TARGETS}
    cross_accumulators = {metric: {} for metric in METRIC_TARGETS}
    for row, prediction in zip(rows, predictions, strict=True):
        predicted_metrics = prediction.get("metrics", {})
        for metric in METRIC_TARGETS:
            actual = _apply_target_transform(metric, row.get(metric), DEFAULT_TARGET_TRANSFORMS)
            predicted = _apply_target_transform(
                metric,
                predicted_metrics.get(metric),
                DEFAULT_TARGET_TRANSFORMS,
            )
            if actual is None or predicted is None:
                continue
            residual = actual - predicted
            for token in _token_sequence(row.get(FEATURE_IDENTITY_TOKENS)):
                _accumulate_token(identity_accumulators[metric], token, residual)
            for token in _token_sequence(row.get(FEATURE_IDENTITY_CROSS_TOKENS)):
                _accumulate_token(cross_accumulators[metric], token, residual)
    return (
        _select_token_effects(identity_accumulators, IDENTITY_TOKEN_EFFECT_LIMIT),
        _select_token_effects(cross_accumulators, CROSS_TOKEN_EFFECT_LIMIT),
    )


def _torch_available_for_token_learner(compute_backend: str) -> bool:
    try:
        import torch
    except ImportError:
        return False
    if compute_backend == COMPUTE_BACKEND_CUDA:
        try:
            return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
        except Exception:
            return False
    return True


def _learn_token_effects_torch_sparse(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    compute_backend: str,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    import torch
    import torch.nn.functional as functional

    stage_started_at = monotonic()
    device = torch.device("cuda" if compute_backend == COMPUTE_BACKEND_CUDA else "cpu")
    identity_vocab, identity_rows = _build_token_vocab(rows, FEATURE_IDENTITY_TOKENS)
    cross_vocab, cross_rows = _build_token_vocab(rows, FEATURE_IDENTITY_CROSS_TOKENS)
    _train_phase_log(
        "surrogate token learner phase=start (backend=%s rows=%d identity_vocab=%d cross_vocab=%d)",
        compute_backend,
        len(rows),
        len(identity_vocab),
        len(cross_vocab),
    )
    identity_effects: dict[str, dict[str, float]] = {}
    cross_effects: dict[str, dict[str, float]] = {}
    for metric in METRIC_TARGETS:
        metric_started_at = monotonic()
        targets: list[float] = []
        row_indices: list[int] = []
        for row_idx, (row, prediction) in enumerate(zip(rows, predictions, strict=True)):
            actual = _apply_target_transform(metric, row.get(metric), DEFAULT_TARGET_TRANSFORMS)
            predicted = _apply_target_transform(
                metric,
                (prediction.get("metrics") or {}).get(metric),
                DEFAULT_TARGET_TRANSFORMS,
            )
            if actual is None or predicted is None:
                continue
            targets.append(actual - predicted)
            row_indices.append(row_idx)

        if not row_indices:
            _train_phase_log(
                "surrogate token learner phase=metric_skipped (metric=%s reason=no_valid_rows)",
                metric,
            )
            continue

        _train_phase_log(
            "surrogate token learner phase=metric_start (metric=%s rows=%d)",
            metric,
            len(row_indices),
        )

        identity_flat: torch.Tensor | None = None
        identity_offsets: torch.Tensor | None = None
        if len(identity_vocab) > 0:
            identity_values: list[int] = []
            identity_offset_values: list[int] = [0]
            for row_idx in row_indices:
                indices = identity_rows[row_idx]
                if indices:
                    identity_values.extend(indices)
                identity_offset_values.append(len(identity_values))
            identity_flat = torch.tensor(identity_values, dtype=torch.long, device=device)
            identity_offsets = torch.tensor(identity_offset_values, dtype=torch.long, device=device)

        cross_flat: torch.Tensor | None = None
        cross_offsets: torch.Tensor | None = None
        if len(cross_vocab) > 0:
            cross_values: list[int] = []
            cross_offset_values: list[int] = [0]
            for row_idx in row_indices:
                indices = cross_rows[row_idx]
                if indices:
                    cross_values.extend(indices)
                cross_offset_values.append(len(cross_values))
            cross_flat = torch.tensor(cross_values, dtype=torch.long, device=device)
            cross_offsets = torch.tensor(cross_offset_values, dtype=torch.long, device=device)

        identity_weights = torch.zeros(len(identity_vocab), dtype=torch.float32, device=device)
        identity_weights.requires_grad_(len(identity_vocab) > 0)
        cross_weights = torch.zeros(len(cross_vocab), dtype=torch.float32, device=device)
        cross_weights.requires_grad_(len(cross_vocab) > 0)
        trainable = [
            weights for weights in (identity_weights, cross_weights) if weights.requires_grad
        ]
        if not trainable:
            continue
        optimizer = torch.optim.Adam(trainable, lr=0.05)
        target_tensor = torch.tensor(targets, dtype=torch.float32, device=device)
        optimizer_steps = 40
        last_progress_log_at = metric_started_at
        for step in range(optimizer_steps):
            optimizer.zero_grad()
            prediction_tensor = torch.zeros(len(row_indices), dtype=torch.float32, device=device)
            if identity_flat is not None and identity_offsets is not None:
                prediction_tensor = prediction_tensor + functional.embedding_bag(
                    identity_flat,
                    identity_weights.unsqueeze(1),
                    identity_offsets,
                    mode="sum",
                    include_last_offset=True,
                ).squeeze(1)
            if cross_flat is not None and cross_offsets is not None:
                prediction_tensor = prediction_tensor + functional.embedding_bag(
                    cross_flat,
                    cross_weights.unsqueeze(1),
                    cross_offsets,
                    mode="sum",
                    include_last_offset=True,
                ).squeeze(1)
            loss = torch.mean((prediction_tensor - target_tensor) ** 2)
            loss.backward()
            optimizer.step()

            now = monotonic()
            should_log_progress = (
                step + 1 == optimizer_steps
                or (step + 1) % 10 == 0
                or (now - last_progress_log_at) >= 30.0
            )
            if should_log_progress:
                _train_phase_log(
                    "surrogate token learner phase=metric_progress "
                    "(metric=%s step=%d/%d loss=%.6f elapsed=%.1fs)",
                    metric,
                    step + 1,
                    optimizer_steps,
                    float(loss.detach().item()),
                    max(0.0, now - metric_started_at),
                )
                last_progress_log_at = now

        raw_identity = {
            token: float(identity_weights[index].item())
            for token, index in identity_vocab.items()
            if float(identity_weights[index].item()) != 0.0
        }
        raw_cross = {
            token: float(cross_weights[index].item())
            for token, index in cross_vocab.items()
            if float(cross_weights[index].item()) != 0.0
        }
        selected_identity = _select_effect_entries(raw_identity, IDENTITY_TOKEN_EFFECT_LIMIT)
        selected_cross = _select_effect_entries(raw_cross, CROSS_TOKEN_EFFECT_LIMIT)
        if selected_identity:
            identity_effects[metric] = selected_identity
        if selected_cross:
            cross_effects[metric] = selected_cross
        _train_phase_log(
            "surrogate token learner phase=metric_complete "
            "(metric=%s identity_effects=%d cross_effects=%d elapsed=%.1fs)",
            metric,
            len(selected_identity),
            len(selected_cross),
            max(0.0, monotonic() - metric_started_at),
        )

    _train_phase_log(
        "surrogate token learner phase=completed (elapsed=%.1fs)",
        max(0.0, monotonic() - stage_started_at),
    )

    return identity_effects, cross_effects


def _build_token_vocab(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> tuple[dict[str, int], list[list[int]]]:
    vocab: dict[str, int] = {}
    row_indices: list[list[int]] = []
    for row in rows:
        indices: list[int] = []
        for token in _token_sequence(row.get(field)):
            index = vocab.get(token)
            if index is None:
                index = len(vocab)
                vocab[token] = index
            indices.append(index)
        row_indices.append(indices)
    return vocab, row_indices


def _select_effect_entries(effects: Mapping[str, float], limit: int) -> dict[str, float]:
    entries = [(token, effect) for token, effect in effects.items() if effect]
    if not entries:
        return {}
    entries.sort(key=lambda entry: (-abs(entry[1]), entry[0]))
    return {token: effect for token, effect in entries[:limit]}


def _token_sequence(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(entry) for entry in value if entry is not None]


def _parse_token_effects(payload: Any | None) -> dict[str, dict[str, float]]:
    effects: dict[str, dict[str, float]] = {}
    if not isinstance(payload, Mapping):
        return effects
    for metric, tokens in payload.items():
        if not isinstance(tokens, Mapping):
            continue
        metric_effects: dict[str, float] = {}
        for token, value in tokens.items():
            if value is None:
                continue
            metric_effects[str(token)] = float(value)
        if metric_effects:
            effects[str(metric)] = metric_effects
    return effects


def _to_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _default_model_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"baseline-{timestamp}"
