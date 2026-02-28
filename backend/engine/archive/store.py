from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from math import prod
from pathlib import Path
from typing import Any, Sequence, Tuple

from backend.app.settings import settings

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
ARCHIVE_ARTIFACT_SCHEMA_VERSION = 1


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class DescriptorAxisSpec:
    name: str
    metric_key: str
    bins: int
    min_value: float
    max_value: float

    def __post_init__(self) -> None:
        if self.bins < 1:
            raise ValueError("bins must be positive")

    def index_for_value(self, value: float) -> int:
        if self.max_value <= self.min_value:
            return 0
        clamped = max(self.min_value, min(self.max_value, value))
        ratio = (clamped - self.min_value) / (self.max_value - self.min_value)
        index = int(ratio * self.bins)
        if index >= self.bins:
            index = self.bins - 1
        return max(0, index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric_key": self.metric_key,
            "bins": self.bins,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }


DEFAULT_DESCRIPTOR_AXES: Tuple[DescriptorAxisSpec, ...] = (
    DescriptorAxisSpec("damage", "full_dps", bins=8, min_value=0.0, max_value=4000.0),
    DescriptorAxisSpec("utility", "utility_score", bins=5, min_value=0.0, max_value=200.0),
)


@dataclass(frozen=True)
class ArchiveMetrics:
    bins_filled: int
    total_bins: int
    coverage: float
    qd_score: float


@dataclass(frozen=True)
class ArchiveStoreEntry:
    bin_key: str
    build_id: str
    score: float
    descriptor_values: Tuple[float, ...]
    metadata: dict[str, Any]


def descriptor_values_from_metrics(
    metrics_payload: Mapping[str, Any],
    axes: Sequence[DescriptorAxisSpec],
) -> Tuple[float, ...]:
    values: list[float] = []
    for axis in axes:
        collected: list[float] = []
        for entry in metrics_payload.values():
            if isinstance(entry, Mapping):
                payload = entry.get("metrics") or entry
            else:
                continue
            if not isinstance(payload, Mapping):
                continue
            candidate = _coerce_float(payload.get(axis.metric_key))
            if candidate is not None:
                collected.append(candidate)
        axis_value = sum(collected) / len(collected) if collected else 0.0
        values.append(axis_value)
    return tuple(values)


def score_from_metrics(metrics_payload: Mapping[str, Any]) -> float:
    best: float | None = None
    for entry in metrics_payload.values():
        if isinstance(entry, Mapping):
            payload = entry.get("metrics") or entry
        else:
            continue
        if not isinstance(payload, Mapping):
            continue
        candidate = _coerce_float(payload.get("full_dps"))
        if candidate is None:
            continue
        if best is None or candidate > best:
            best = candidate
    return best or 0.0


class ArchiveStore:
    def __init__(self, axes: Sequence[DescriptorAxisSpec] | None = None) -> None:
        self._axes = tuple(axes or DEFAULT_DESCRIPTOR_AXES)
        self._entries: dict[str, ArchiveStoreEntry] = {}

    @property
    def axes(self) -> tuple[DescriptorAxisSpec, ...]:
        return self._axes

    @property
    def total_bins(self) -> int:
        if not self._axes:
            return 0
        return prod(axis.bins for axis in self._axes)

    def insert(
        self,
        build_id: str,
        score: float,
        descriptor: Sequence[float],
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        if len(descriptor) != len(self._axes):
            raise ValueError("descriptor length does not match axis count")
        descriptor_tuple = tuple(float(value) for value in descriptor)
        bin_indices = [
            axis.index_for_value(value)
            for axis, value in zip(self._axes, descriptor_tuple, strict=True)
        ]
        bin_key = "-".join(str(index) for index in bin_indices)
        metadata_dict = dict(metadata or {})
        entry = ArchiveStoreEntry(
            bin_key=bin_key,
            build_id=build_id,
            score=float(score),
            descriptor_values=descriptor_tuple,
            metadata=metadata_dict,
        )
        existing = self._entries.get(bin_key)
        if existing is None or self._should_replace(entry, existing):
            self._entries[bin_key] = entry
            return True
        return False

    def entry_for_bin(self, bin_key: str) -> ArchiveStoreEntry | None:
        return self._entries.get(bin_key)

    def entries(self) -> list[ArchiveStoreEntry]:
        return [self._entries[key] for key in sorted(self._entries.keys())]

    def metrics(self) -> ArchiveMetrics:
        total = self.total_bins
        filled = len(self._entries)
        coverage = (filled / total) if total else 0.0
        qd_score = sum(entry.score for entry in self._entries.values())
        return ArchiveMetrics(
            bins_filled=filled,
            total_bins=total,
            coverage=coverage,
            qd_score=qd_score,
        )

    def metrics_dict(self) -> dict[str, float | int]:
        metrics = self.metrics()
        return {
            "bins_filled": metrics.bins_filled,
            "total_bins": metrics.total_bins,
            "coverage": metrics.coverage,
            "qd_score": metrics.qd_score,
        }

    def _should_replace(self, candidate: ArchiveStoreEntry, current: ArchiveStoreEntry) -> bool:
        if candidate.score > current.score:
            return True
        if candidate.score == current.score and candidate.build_id < current.build_id:
            return True
        return False


def _validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_' or '-'",
        )
    return run_id


def _archive_root(base_path: Path | None = None) -> Path:
    return (Path(base_path or settings.data_path) / "runs").resolve()


def archive_artifact_path(run_id: str, base_path: Path | None = None) -> Path:
    safe_run_id = _validate_run_id(run_id)
    root = _archive_root(base_path)
    artifact = (root / safe_run_id / "archive.json").resolve()
    if root not in artifact.parents:
        raise ValueError("run_id resolved outside run directory")
    return artifact


def persist_archive(
    run_id: str,
    store: ArchiveStore,
    base_path: Path | None = None,
    created_at: str | None = None,
) -> Path:
    created = created_at or datetime.now(timezone.utc).isoformat()
    payload = {
        "schema_version": ARCHIVE_ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": created,
        "axes": [axis.to_dict() for axis in store.axes],
        "metrics": store.metrics_dict(),
        "bins": [
            {
                "bin_key": entry.bin_key,
                "build_id": entry.build_id,
                "score": entry.score,
                "descriptor": {
                    axis.name: entry.descriptor_values[idx] for idx, axis in enumerate(store.axes)
                },
                "metadata": entry.metadata,
            }
            for entry in store.entries()
        ],
    }
    path = archive_artifact_path(run_id, base_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def load_archive_artifact(run_id: str, base_path: Path | None = None) -> dict[str, Any]:
    path = archive_artifact_path(run_id, base_path)
    if not path.exists():
        raise FileNotFoundError(f"archive for {run_id} not found")
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("schema_version")
    if version != ARCHIVE_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported archive schema version")
    return payload


__all__ = [
    "ArchiveStore",
    "ArchiveStoreEntry",
    "ArchiveMetrics",
    "DEFAULT_DESCRIPTOR_AXES",
    "descriptor_values_from_metrics",
    "score_from_metrics",
    "persist_archive",
    "load_archive_artifact",
    "archive_artifact_path",
    "DescriptorAxisSpec",
]
