from __future__ import annotations

import logging
from collections.abc import Mapping as AbcMapping
from collections.abc import Sequence as AbcSequence
from dataclasses import dataclass
from typing import Any, Mapping

logger = logging.getLogger(__name__)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, AbcMapping):
        return value
    return None


def _flatten_warnings(value: Any) -> tuple[str, ...]:
    if isinstance(value, AbcSequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item) for item in value)
    if value is None:
        return ()
    return (str(value),)


@dataclass(frozen=True)
class ResistSnapshot:
    fire: float
    cold: float
    lightning: float
    chaos: float

    def get(self, name: str) -> float:
        normalized = name.lower()
        values = {
            "fire": self.fire,
            "cold": self.cold,
            "lightning": self.lightning,
            "chaos": self.chaos,
        }
        if normalized in values:
            return values[normalized]
        logger.warning("Unknown resist name requested: %s", name)
        return 0.0


@dataclass(frozen=True)
class NormalizedAttributes:
    strength: float
    dexterity: float
    intelligence: float

    def get(self, name: str) -> float:
        normalized = name.lower()
        return getattr(self, normalized, 0.0)


@dataclass(frozen=True)
class NormalizedReservation:
    reserved_percent: float
    available_percent: float


@dataclass(frozen=True)
class NormalizedMetrics:
    full_dps: float
    max_hit: float
    armour: float
    evasion: float
    life: float
    mana: float
    utility_score: float
    resists: ResistSnapshot
    reservation: NormalizedReservation
    attributes: NormalizedAttributes
    warnings: tuple[str, ...]


def _extract_resists(payload: Mapping[str, Any]) -> ResistSnapshot:
    defense = _as_mapping(payload.get("defense")) or {}
    resist_data = _as_mapping(defense.get("resists")) or _as_mapping(payload.get("resists")) or {}
    return ResistSnapshot(
        fire=_coerce_float(resist_data.get("fire")),
        cold=_coerce_float(resist_data.get("cold")),
        lightning=_coerce_float(resist_data.get("lightning")),
        chaos=_coerce_float(resist_data.get("chaos")),
    )


def _extract_reservation(payload: Mapping[str, Any]) -> NormalizedReservation:
    reservation = _as_mapping(payload.get("reservation")) or {}
    return NormalizedReservation(
        reserved_percent=_coerce_float(reservation.get("reserved_percent")),
        available_percent=_coerce_float(reservation.get("available_percent")),
    )


def _extract_attributes(payload: Mapping[str, Any]) -> NormalizedAttributes:
    attributes = _as_mapping(payload.get("attributes")) or {}
    return NormalizedAttributes(
        strength=_coerce_float(
            attributes.get("strength", attributes.get("str", attributes.get("Str")))
        ),
        dexterity=_coerce_float(
            attributes.get("dexterity", attributes.get("dex", attributes.get("Dex")))
        ),
        intelligence=_coerce_float(
            attributes.get("intelligence", attributes.get("int", attributes.get("Int")))
        ),
    )


def map_worker_output(payload: Mapping[str, Any]) -> NormalizedMetrics:
    metrics = _as_mapping(payload.get("metrics")) or {}
    defense = _as_mapping(payload.get("defense")) or {}
    resources = _as_mapping(payload.get("resources")) or {}
    utility_branch = _as_mapping(payload.get("utility")) or {}

    return NormalizedMetrics(
        full_dps=_coerce_float(metrics.get("full_dps")),
        max_hit=_coerce_float(metrics.get("max_hit")),
        armour=_coerce_float(defense.get("armour")),
        evasion=_coerce_float(defense.get("evasion")),
        life=_coerce_float(resources.get("life")),
        mana=_coerce_float(resources.get("mana")),
        utility_score=_coerce_float(metrics.get("utility_score", utility_branch.get("score"))),
        resists=_extract_resists(payload),
        reservation=_extract_reservation(payload),
        attributes=_extract_attributes(payload),
        warnings=_flatten_warnings(payload.get("warnings") or payload.get("pob_warnings")),
    )


__all__ = [
    "NormalizedMetrics",
    "NormalizedReservation",
    "NormalizedAttributes",
    "ResistSnapshot",
    "map_worker_output",
]
