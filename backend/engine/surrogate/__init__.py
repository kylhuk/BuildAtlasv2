"""Surrogate helpers bundle."""

from .dataset import FEATURE_SCHEMA_VERSION, SnapshotResult, build_dataset_snapshot
from .model import (
    EvaluationResult,
    SurrogateModel,
    TrainResult,
    evaluate_predictions,
    load_dataset_rows,
    load_model,
    resolve_snapshot_root,
    train,
)

__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "SnapshotResult",
    "build_dataset_snapshot",
    "EvaluationResult",
    "SurrogateModel",
    "TrainResult",
    "evaluate_predictions",
    "load_dataset_rows",
    "load_model",
    "resolve_snapshot_root",
    "train",
]
