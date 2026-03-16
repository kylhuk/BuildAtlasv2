"""Surrogate helpers bundle."""

from .dataset import (
    FEATURE_SCHEMA_VERSION,
    SnapshotResult,
    build_dataset_snapshot,
    DatasetValidationResult,
    validate_dataset,
    compute_dataset_statistics,
)
from .model import (
    EnsembleSurrogate,
    EvaluationResult,
    SurrogateModel,
    TrainResult,
    evaluate_predictions,
    incremental_train,
    load_dataset_rows,
    load_model,
    resolve_snapshot_root,
    train,
)

__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "SnapshotResult",
    "build_dataset_snapshot",
    "DatasetValidationResult",
    "validate_dataset",
    "compute_dataset_statistics",
    "EnsembleSurrogate",
    "EvaluationResult",
    "SurrogateModel",
    "TrainResult",
    "evaluate_predictions",
    "incremental_train",
    "load_dataset_rows",
    "load_model",
    "resolve_snapshot_root",
    "train",
]
