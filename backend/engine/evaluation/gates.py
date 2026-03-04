from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from backend.engine.evaluation.normalized import NormalizedMetrics
from backend.engine.scenarios.loader import ScenarioGateThresholds

GATE_REASON_RESIST_FIRE = "resist_fire_shortfall"
GATE_REASON_RESIST_COLD = "resist_cold_shortfall"
GATE_REASON_RESIST_LIGHTNING = "resist_lightning_shortfall"
GATE_REASON_RESIST_CHAOS = "resist_chaos_shortfall"
GATE_REASON_MAX_HIT = "max_hit_too_low"
GATE_REASON_RESERVATION = "reservation_infeasible"
GATE_REASON_ATTRIBUTES = "attributes_requirements"
GATE_REASON_FULL_DPS = "full_dps_too_low"

_RESIST_REASON_MAPPING = {
    "fire": GATE_REASON_RESIST_FIRE,
    "cold": GATE_REASON_RESIST_COLD,
    "lightning": GATE_REASON_RESIST_LIGHTNING,
    "chaos": GATE_REASON_RESIST_CHAOS,
}


@dataclass(frozen=True)
class GateEvaluation:
    gate_pass: bool
    gate_fail_reasons: tuple[str, ...]


def _extract_resist_failures(
    metrics: NormalizedMetrics,
    thresholds: ScenarioGateThresholds,
) -> Iterable[str]:
    for name, minimum in thresholds.resists.items():
        actual = metrics.resists.get(name)
        if actual < minimum:
            reason = _RESIST_REASON_MAPPING.get(name.lower())
            if reason:
                yield reason


def _extract_attribute_failure(
    metrics: NormalizedMetrics,
    thresholds: ScenarioGateThresholds,
) -> Iterable[str]:
    for name, minimum in thresholds.attributes.items():
        actual = metrics.attributes.get(name)
        if actual < minimum:
            yield GATE_REASON_ATTRIBUTES
            break


def _extract_reservation_failure(
    metrics: NormalizedMetrics,
    thresholds: ScenarioGateThresholds,
) -> Iterable[str]:
    reservation = metrics.reservation
    if reservation.available_percent < reservation.reserved_percent:
        yield GATE_REASON_RESERVATION
        return
    if reservation.reserved_percent > thresholds.reservation.max_percent:
        yield GATE_REASON_RESERVATION


def _extract_max_hit_failure(
    metrics: NormalizedMetrics,
    thresholds: ScenarioGateThresholds,
) -> Iterable[str]:
    if metrics.max_hit < thresholds.min_max_hit:
        yield GATE_REASON_MAX_HIT


def _extract_full_dps_failure(
    metrics: NormalizedMetrics,
    thresholds: ScenarioGateThresholds,
) -> Iterable[str]:
    if metrics.full_dps < thresholds.min_full_dps:
        yield GATE_REASON_FULL_DPS


def evaluate_gates(
    metrics: NormalizedMetrics,
    thresholds: ScenarioGateThresholds,
) -> GateEvaluation:
    failures: list[str] = []
    failures.extend(_extract_resist_failures(metrics, thresholds))
    failures.extend(_extract_attribute_failure(metrics, thresholds))
    failures.extend(_extract_reservation_failure(metrics, thresholds))
    failures.extend(_extract_max_hit_failure(metrics, thresholds))
    failures.extend(_extract_full_dps_failure(metrics, thresholds))

    unique_reasons = tuple(dict.fromkeys(failures))
    return GateEvaluation(gate_pass=not unique_reasons, gate_fail_reasons=unique_reasons)


__all__ = [
    "GateEvaluation",
    "evaluate_gates",
    "GATE_REASON_RESIST_FIRE",
    "GATE_REASON_RESIST_COLD",
    "GATE_REASON_RESIST_LIGHTNING",
    "GATE_REASON_RESIST_CHAOS",
    "GATE_REASON_MAX_HIT",
    "GATE_REASON_FULL_DPS",
    "GATE_REASON_RESERVATION",
    "GATE_REASON_ATTRIBUTES",
]
