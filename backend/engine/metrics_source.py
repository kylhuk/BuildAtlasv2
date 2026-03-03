"""Common helpers for evaluating PoB metrics source annotations."""

from __future__ import annotations

from typing import Any

METRICS_SOURCE_POB = "pob"
METRICS_SOURCE_STUB = "stub"
METRICS_SOURCE_FALLBACK = "fallback"
METRICS_SOURCE_VALUES = {
    METRICS_SOURCE_POB,
    METRICS_SOURCE_STUB,
    METRICS_SOURCE_FALLBACK,
}

def normalize_metrics_source(value: Any | None, default: str | None = None) -> str | None:
    """Normalize arbitrary input to a known metrics source string."""

    if not isinstance(value, str):
        return default
    candidate = value.strip().lower()
    return candidate if candidate in METRICS_SOURCE_VALUES else default

def default_pob_source(value: Any | None) -> str:
    """Normalize metrics source with a default of PoB."""

    return normalize_metrics_source(value, METRICS_SOURCE_POB) or METRICS_SOURCE_POB
