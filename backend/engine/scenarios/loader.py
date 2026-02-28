from __future__ import annotations

import json
from collections.abc import Mapping as AbcMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from backend.engine.artifacts import format_ruleset_id

TEMPLATES_DIR = Path(__file__).parent / "templates"


@dataclass(frozen=True)
class ScenarioReservationThreshold:
    max_percent: float


@dataclass(frozen=True)
class ScenarioGateThresholds:
    resists: Mapping[str, float]
    reservation: ScenarioReservationThreshold
    attributes: Mapping[str, float]
    min_max_hit: float
    min_full_dps: float


@dataclass(frozen=True)
class ScenarioTemplate:
    scenario_id: str
    version: str
    profile_id: str
    pob_config: Mapping[str, Any]
    gate_thresholds: ScenarioGateThresholds


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_threshold_map(values: Mapping[str, Any] | None) -> Mapping[str, float]:
    if not values:
        return {}
    return {key.lower(): _coerce_float(value) for key, value in values.items()}


def _expect_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _template_paths() -> Sequence[Path]:
    if not TEMPLATES_DIR.exists():
        return ()
    return tuple(sorted(TEMPLATES_DIR.glob("*.json")))


def _load_template_from_path(path: Path) -> ScenarioTemplate:
    raw = json.loads(path.read_text(encoding="utf-8"))
    scenario_id = _expect_str(raw.get("scenario_id"), "scenario_id")
    version = _expect_str(raw.get("version"), "version")
    profile_id = _expect_str(raw.get("profile_id"), "profile_id")
    pob_config = raw.get("pob_config")
    if not isinstance(pob_config, AbcMapping):
        raise ValueError("pob_config must be a mapping")

    gate_data = raw.get("gate_thresholds")
    if not isinstance(gate_data, AbcMapping):
        raise ValueError("gate_thresholds must be a mapping")

    resists = _normalize_threshold_map(gate_data.get("resists"))
    attributes = _normalize_threshold_map(gate_data.get("attributes"))
    reservation_data = gate_data.get("reservation")
    if not isinstance(reservation_data, AbcMapping):
        raise ValueError("reservation gate thresholds must be defined")
    reservation_threshold = ScenarioReservationThreshold(
        max_percent=_coerce_float(reservation_data.get("max_percent")),
    )
    min_max_hit = _coerce_float(gate_data.get("min_max_hit"))
    min_full_dps = _coerce_float(gate_data.get("min_full_dps"))

    gate_thresholds = ScenarioGateThresholds(
        resists=resists,
        reservation=reservation_threshold,
        attributes=attributes,
        min_max_hit=min_max_hit,
        min_full_dps=min_full_dps,
    )

    return ScenarioTemplate(
        scenario_id=scenario_id,
        version=version,
        profile_id=profile_id,
        pob_config=pob_config,
        gate_thresholds=gate_thresholds,
    )


def list_templates() -> Sequence[ScenarioTemplate]:
    templates = [_load_template_from_path(path) for path in _template_paths()]
    templates.sort(key=lambda template: (template.scenario_id, template.version))
    return tuple(templates)


def load_template(scenario_id: str, version: str | None = None) -> ScenarioTemplate:
    resolved_id = _expect_str(scenario_id, "scenario_id")
    available = [template for template in list_templates() if template.scenario_id == resolved_id]
    if not available:
        raise ValueError(f"no templates found for scenario_id={resolved_id}")
    if version is None:
        return available[-1]
    for template in available:
        if template.version == version:
            return template
    raise ValueError(f"no template for scenario_id={resolved_id} with version={version}")


def scenario_version_tag(template: ScenarioTemplate) -> str:
    return f"{template.scenario_id}@{template.version}"


def compose_ruleset_id(pob_commit: str, template: ScenarioTemplate, price_snapshot_id: str) -> str:
    scenario_version = scenario_version_tag(template)
    return format_ruleset_id(pob_commit, scenario_version, price_snapshot_id)


__all__ = [
    "ScenarioTemplate",
    "ScenarioGateThresholds",
    "ScenarioReservationThreshold",
    "list_templates",
    "load_template",
    "scenario_version_tag",
    "compose_ruleset_id",
]
