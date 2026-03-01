from __future__ import annotations

import json
import logging
import random
import re
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable, Literal, Mapping, Sequence

from backend.app.api.evaluator import BuildEvaluator
from backend.app.api.models import BuildStatus
from backend.app.db.ch import BuildInsertPayload, ClickhouseRepository, ScenarioMetricRow
from backend.app.settings import settings
from backend.engine.archive import (
    ArchiveStore,
    ExploitEmitter,
    NoveltyEmitter,
    UncertaintyEmitter,
    descriptor_values_from_metrics,
    deterministic_allocator,
    persist_archive,
    score_from_metrics,
)
from backend.engine.artifacts.store import (
    artifact_paths,
    purge_build_artifacts,
    write_build_artifacts,
    write_build_constraints,
)
from backend.engine.build_details import build_details_from_generation
from backend.engine.surrogate.dataset import extract_feature_signals
from backend.engine.constraints import (
    ConstraintSpec,
    constraint_artifact_payload,
    evaluate_constraints,
)
from backend.engine.genome import GenomeV0, deterministic_genome_from_seed
from backend.engine.items.templates import (
    ItemTemplatePlan,
    RepairReport,
    build_item_templates,
)
from backend.engine.passives.builder import PassiveTreePlan, build_passive_tree_plan
from backend.engine.scenarios.loader import ScenarioTemplate, list_templates
from backend.engine.skills.catalog import GemPlan, SkillCatalog, load_default_skill_catalog
from backend.engine.sockets.planner import SocketPlan, plan_sockets

logger = logging.getLogger(__name__)

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

DEFAULT_POINT_BUDGETS: dict[str, int] = {
    "starter": 55,
    "midgame": 70,
    "endgame": 90,
}
SURROGATE_PREDICTION_SCHEMA_VERSION = 1


PRECHECK_CATEGORIES: tuple[str, ...] = (
    "socket_precheck_failed",
    "attribute_precheck_failed",
    "reservation_precheck_failed",
    "gem_completeness_precheck_failed",
    "item_completeness_precheck_failed",
)

POB_TARGET_VERSION_DEFAULT = "3_0"
_POB_TARGET_VERSION_PATTERN = re.compile(r"liveTargetVersion\s*=\s*\"([^\"]+)\"")
_POB_GAME_VERSIONS_PATH = settings.pob_path / "src" / "GameVersions.lua"


def _load_pob_target_version() -> str:
    try:
        content = _POB_GAME_VERSIONS_PATH.read_text(encoding="utf-8")
    except OSError:
        return POB_TARGET_VERSION_DEFAULT
    match = _POB_TARGET_VERSION_PATTERN.search(content)
    return match.group(1) if match else POB_TARGET_VERSION_DEFAULT


POB_TARGET_VERSION = _load_pob_target_version()

CANONICAL_ITEM_DESCRIPTIONS: tuple[tuple[str, str], ...] = (
    ("Weapon 1", "Rarity: Normal\nDriftwood Wand\n--------\nItem Level: 1"),
    ("Weapon 2", "Rarity: Normal\nDriftwood Sceptre\n--------\nItem Level: 1"),
    ("Helmet", "Rarity: Normal\nIron Hat\n--------\nItem Level: 1"),
    ("Body Armour", "Rarity: Normal\nSimple Robe\n--------\nItem Level: 1"),
    ("Gloves", "Rarity: Normal\nWool Gloves\n--------\nItem Level: 1"),
    ("Boots", "Rarity: Normal\nRawhide Boots\n--------\nItem Level: 1"),
    ("Amulet", "Rarity: Normal\nJade Amulet\n--------\nItem Level: 1"),
    ("Ring 1", "Rarity: Normal\nIron Ring\n--------\nItem Level: 1"),
    ("Ring 2", "Rarity: Normal\nCoral Ring\n--------\nItem Level: 1"),
    ("Belt", "Rarity: Normal\nLeather Belt\n--------\nItem Level: 1"),
    ("Flask 1", "Rarity: Normal\nSmall Life Flask\n--------\nItem Level: 1"),
    ("Flask 2", "Rarity: Normal\nSmall Mana Flask\n--------\nItem Level: 1"),
    ("Flask 3", "Rarity: Normal\nQuicksilver Flask\n--------\nItem Level: 1"),
    ("Flask 4", "Rarity: Normal\nRuby Flask\n--------\nItem Level: 1"),
    ("Flask 5", "Rarity: Normal\nDiamond Flask\n--------\nItem Level: 1"),
)
CANONICAL_ITEM_SLOTS = tuple(slot for slot, _ in CANONICAL_ITEM_DESCRIPTIONS)
SKILL_SLOT_ORDER = ("Weapon 1", "Body Armour", "Helmet", "Gloves", "Boots")
SLOT_ID_TO_SKILL_SLOT: dict[str, str] = {
    "weapon_2h": "Weapon 1",
    "body_armour": "Body Armour",
    "helmet": "Helmet",
    "gloves": "Gloves",
    "boots": "Boots",
}
FALLBACK_GEMS_BY_SLOT: dict[str, tuple[str, ...]] = {
    "Weapon 1": ("arc", "controlled_destruction_support"),
    "Body Armour": ("molten_shell",),
    "Helmet": ("arctic_armour",),
    "Gloves": ("fortify_support",),
    "Boots": ("dash",),
}
DEFAULT_GEM_LEVEL = "20"
DEFAULT_GEM_QUALITY = "0"


@dataclass
class Candidate:
    seed: int
    build_id: str
    main_skill_package: str
    class_name: str
    ascendancy: str | None
    budget_tier: str
    failures: list[str]
    metrics_payload: Mapping[str, Any]
    genome: GenomeV0
    code_payload: str
    build_details_payload: Mapping[str, Any]
    predicted_full_dps: float | None = None
    predicted_metrics: dict[str, float] | None = None
    pass_probability: float | None = None
    selection_reason: str | None = None
    selected_for_evaluation: bool = False
    evaluation_status: str | None = None
    gate_pass: bool | None = None
    evaluation_error: dict[str, Any] | None = None
    verified_metrics_payload: Mapping[str, Any] | None = None
    stage_label: str = "random_seed"
    source: str = "random"
    optimizer_iteration: int = 0
    parent_build_id: str | None = None
    parent_seed: int | None = None
    constraint_status: str | None = None
    constraint_reason_code: str | None = None
    violated_constraints: list[str] | None = None
    constraint_checked_at: str | None = None
    surrogate_prediction_payload: dict[str, Any] | None = None
    persisted: bool = False

    @property
    def evaluable(self) -> bool:
        return not self.failures


def _candidate_feature_row(candidate: Candidate) -> dict[str, Any]:
    row = {
        "seed": candidate.seed,
        "main_skill_package": candidate.main_skill_package,
        "class_name": candidate.class_name,
        "ascendancy": candidate.ascendancy,
        "budget_tier": candidate.budget_tier,
    }
    row.update(extract_feature_signals(candidate.build_details_payload))
    return row


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metrics_payload_from_scenario_rows(rows: Sequence[Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for row in rows:
        scenario_id = getattr(row, "scenario_id", None)
        if scenario_id is None and isinstance(row, Mapping):
            scenario_id = row.get("scenario_id")
        if not scenario_id:
            continue
        full_dps = _coerce_float(getattr(row, "full_dps", None))
        if full_dps is None and isinstance(row, Mapping):
            full_dps = _coerce_float(row.get("full_dps"))
        max_hit = _coerce_float(getattr(row, "max_hit", None))
        if max_hit is None and isinstance(row, Mapping):
            max_hit = _coerce_float(row.get("max_hit"))
        utility_score = _coerce_float(getattr(row, "utility_score", None))
        if utility_score is None and isinstance(row, Mapping):
            utility_score = _coerce_float(row.get("utility_score"))
        payload[str(scenario_id)] = {
            "metrics": {
                "full_dps": full_dps or 0.0,
                "max_hit": max_hit or 0.0,
                "utility_score": utility_score or 0.0,
            }
        }
    return payload


def _select_optimizer_elites(
    candidates: Sequence[Candidate], elite_count: int
) -> list[tuple[Candidate, dict[str, Any]]]:
    scored: list[tuple[Candidate, float, float, float, dict[str, Any]]] = []
    for candidate in candidates:
        if not candidate.evaluable:
            continue
        full_dps, max_hit, cost_sort, summary = _optimizer_objectives_from_payload(
            candidate.metrics_payload
        )
        scored.append((candidate, full_dps, max_hit, cost_sort, summary))
    if not scored:
        return []
    scored.sort(key=lambda item: (-item[1], item[2], item[3], item[0].seed))
    selected_entries: list[tuple[Candidate, float, float, float, dict[str, Any]]] = []
    for entry in scored:
        if len(selected_entries) >= elite_count:
            break
        candidate, full_dps, max_hit, cost_sort, summary = entry
        dominated = False
        for existing in selected_entries:
            _, existing_full, existing_max, existing_cost, _ = existing
            if (
                existing_full >= full_dps
                and existing_max <= max_hit
                and existing_cost <= cost_sort
                and (
                    existing_full > full_dps or existing_max < max_hit or existing_cost < cost_sort
                )
            ):
                dominated = True
                break
        if not dominated:
            selected_entries.append(entry)
    selected: list[tuple[Candidate, dict[str, Any]]] = []
    selected_ids: set[str] = set()
    for candidate, _, _, _, summary in selected_entries:
        selected.append((candidate, summary))
        selected_ids.add(candidate.build_id)
        if len(selected) >= elite_count:
            break
    idx = 0
    while len(selected) < elite_count and idx < len(scored):
        candidate, _, _, _, summary = scored[idx]
        if candidate.build_id not in selected_ids:
            selected.append((candidate, summary))
            selected_ids.add(candidate.build_id)
        idx += 1
    return selected[:elite_count]


def _optimizer_selection_record(
    stage: str,
    iteration: int,
    elites: list[tuple[Candidate, dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "stage": stage,
        "iteration": iteration,
        "elites": [
            {
                "build_id": candidate.build_id,
                "seed": candidate.seed,
                "objectives": {
                    "full_dps": summary.get("full_dps"),
                    "max_hit": summary.get("max_hit"),
                    "cost": summary.get("cost"),
                    "selection_basis": summary.get("selection_basis"),
                },
            }
            for candidate, summary in elites
        ],
    }


def _derive_optimizer_seed(
    parent_seed: int,
    iteration: int,
    rank: int,
    used_seeds: set[int],
) -> int | None:
    for attempt in range(16):
        candidate = (
            parent_seed * 6364136223846793005
            + iteration * 1442695040888963407
            + rank * 0x9E3779B97F4A7C15
            + attempt * 0x165667B19E3779F9
        ) & 0xFFFFFFFF
        if candidate not in used_seeds:
            return candidate
    return None


def _normalize_prediction_entry(entry: Any) -> tuple[dict[str, float], float | None]:
    metrics: dict[str, float] = {}
    pass_probability: float | None = None
    if isinstance(entry, Mapping):
        candidate_metrics = (
            entry.get("metrics") or entry.get("predicted_metrics") or entry.get("scores") or entry
        )
        if isinstance(candidate_metrics, Mapping):
            for key, raw_value in candidate_metrics.items():
                coerced = _coerce_float(raw_value)
                if coerced is not None:
                    metrics[str(key)] = coerced
        else:
            coerced = _coerce_float(candidate_metrics)
            if coerced is not None:
                metrics["full_dps"] = coerced
        pass_probability = _coerce_float(
            entry.get("pass_probability") or entry.get("pass_prob") or entry.get("probability")
        )
        fallback = _coerce_float(entry.get("score") or entry.get("predicted_full_dps"))
        if not metrics and fallback is not None:
            metrics["full_dps"] = fallback
    else:
        coerced = _coerce_float(entry)
        if coerced is not None:
            metrics["full_dps"] = coerced
    return metrics, pass_probability


def _load_surrogate_predictor(
    path: Path,
) -> tuple[str | None, Callable[[Sequence[dict[str, Any]]], Sequence[float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    model_id = payload.get("model_id") or payload.get("id") or path.name
    mapping = payload.get("predictions") or payload.get("scores") or {}
    if not isinstance(mapping, Mapping):
        raise ValueError("surrogate predictions payload must be a mapping")
    default_score = float(payload.get("default", 0.0))
    normalized = {str(key): float(value) for key, value in mapping.items()}

    def predictor(rows: Sequence[dict[str, Any]]) -> Sequence[float]:
        results: list[float] = []
        for row in rows:
            key = str(row.get("main_skill_package") or "")
            results.append(normalized.get(key, default_score))
        return results

    return model_id, predictor


SocketPlanner = Callable[[SkillCatalog, GenomeV0], SocketPlan]
PassivePlanner = Callable[[GenomeV0, int], PassiveTreePlan]
TemplateBuilder = Callable[[GenomeV0, GemPlan, PassiveTreePlan, SocketPlan], ItemTemplatePlan]
MetricsGenerator = Callable[[int, Sequence[ScenarioTemplate]], Mapping[str, Any]]


def _point_budget_for(genome: GenomeV0) -> int:
    return DEFAULT_POINT_BUDGETS.get(genome.budget_tier, 70)


def _default_metrics_generator(
    seed: int,
    templates: Sequence[ScenarioTemplate],
) -> Mapping[str, Any]:
    base = max(seed, 1)
    payload: dict[str, Any] = {}
    for template in templates:
        thresholds = template.gate_thresholds
        reserved = max(0.0, thresholds.reservation.max_percent - 5)
        available = min(100.0, reserved + 10)
        resists = {
            key: float(thresholds.resists.get(key, 70) + 10)
            for key in ("fire", "cold", "lightning", "chaos")
        }
        attributes = {
            key: float(thresholds.attributes.get(key, 150) + 20)
            for key in ("strength", "dexterity", "intelligence")
        }
        payload[template.scenario_id] = {
            "metrics": {
                "full_dps": float(base * 120),
                "max_hit": float(max(thresholds.min_max_hit, 1000) + base * 2),
                "utility_score": float((base % 100) + 10),
            },
            "defense": {
                "armour": float(base * 6),
                "evasion": float(base * 5),
                "resists": resists,
            },
            "resources": {"life": float(base * 10), "mana": float(base * 3)},
            "reservation": {"reserved_percent": reserved, "available_percent": available},
            "attributes": attributes,
            "warnings": ["generation_stub_metrics"],
        }
    return payload


def _entry_is_stub_metrics(entry: Mapping[str, Any]) -> bool:
    warnings = entry.get("warnings") or entry.get("pob_warnings")
    if warnings is None:
        return False
    if isinstance(warnings, Sequence) and not isinstance(warnings, (str, bytes)):
        entries = warnings
    else:
        entries = (warnings,)
    for candidate in entries:
        try:
            normalized = str(candidate).strip().lower()
        except Exception:
            continue
        if normalized == "generation_stub_metrics":
            return True
    return False


def _optimizer_objectives_from_payload(
    metrics_payload: Mapping[str, Any],
) -> tuple[float, float, float, dict[str, Any]]:
    best_full_dps = 0.0
    best_max_hit = float("inf")
    best_cost = float("inf")
    cost_known = False
    metrics_found = False
    if isinstance(metrics_payload, Mapping):
        entries = metrics_payload.values()
    else:
        entries = ()
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        if _entry_is_stub_metrics(entry):
            continue
        candidate_metrics = entry.get("metrics") or entry
        if not isinstance(candidate_metrics, Mapping):
            continue
        metrics_found = True
        full_dps = _coerce_float(candidate_metrics.get("full_dps"))
        if full_dps is not None:
            best_full_dps = max(best_full_dps, full_dps)
        max_hit = _coerce_float(candidate_metrics.get("max_hit"))
        if max_hit is not None:
            best_max_hit = min(best_max_hit, max_hit)
        cost_value = _coerce_float(
            candidate_metrics.get("total_cost_chaos")
            or candidate_metrics.get("cost")
            or candidate_metrics.get("cost_chaos")
        )
        if cost_value is not None:
            best_cost = min(best_cost, cost_value)
            cost_known = True
    if not metrics_found:
        return (
            0.0,
            float("inf"),
            float("inf"),
            {
                "full_dps": None,
                "max_hit": None,
                "cost": None,
                "selection_basis": "stub_fallback",
            },
        )
    summary: dict[str, Any] = {
        "full_dps": best_full_dps,
        "max_hit": None if best_max_hit == float("inf") else best_max_hit,
        "cost": None if not cost_known else best_cost,
        "selection_basis": "objective",
    }
    sort_max_hit = best_max_hit
    sort_cost = best_cost if cost_known else float("inf")
    return best_full_dps, sort_max_hit, sort_cost, summary


def _render_build_code(
    genome: GenomeV0,
    socket_plan: SocketPlan,
    template_plan: ItemTemplatePlan,
    metrics: Mapping[str, Any],
) -> str:
    payload = {
        "seed": genome.seed,
        "genome": asdict(genome),
        "socket_slots": [slot.id for slot in socket_plan.slots],
        "socket_issues": [issue.code for issue in socket_plan.issues],
        "repair_iterations": template_plan.repair_report.iterations,
        "metrics_preview": {scenario: metrics.get(scenario) for scenario in metrics},
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _format_xml_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _xml_attr(value: str) -> str:
    return (
        value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
    )


def _gem_name_spec(gem_id: str) -> str:
    normalized = gem_id.strip().replace("-", "_")
    if not normalized:
        return gem_id
    words = [part for part in normalized.split("_") if part]
    if not words:
        return gem_id.strip()
    return " ".join(word.capitalize() for word in words)


def _slot_id_to_xml_slot(slot_id: str) -> str:
    """Map slot_id to XML slot name."""
    mapping = {
        "weapon_2h": "Weapon 1",
        "weapon_1h": "Weapon 1",
        "offhand": "Weapon 2",
        "body_armour": "Body Armour",
        "helmet": "Helmet",
        "gloves": "Gloves",
        "boots": "Boots",
        "amulet": "Amulet",
        "ring_1": "Ring 1",
        "ring_2": "Ring 2",
        "belt": "Belt",
        "flask_1": "Flask 1",
        "flask_2": "Flask 2",
        "flask_3": "Flask 3",
        "flask_4": "Flask 4",
        "flask_5": "Flask 5",
    }
    return mapping.get(slot_id, slot_id.title())


def _render_items_section(candidate: Candidate) -> list[str]:
    """Render items section from candidate's build_details_payload.

    Produces Rare items with proper base types instead of fallback Normal items.
    """
    lines = ['  <Items activeItemSet="1" useSecondWeaponSet="false" showStatDifferences="true">']
    item_refs: list[tuple[str, str]] = []

    # Try to get actual items from build_details_payload
    items_data = (
        candidate.build_details_payload.get("items")
        if isinstance(candidate.build_details_payload, Mapping)
        else None
    )
    slot_templates = None
    if isinstance(items_data, Mapping):
        slot_templates = items_data.get("slot_templates")

    # Build a map of slot_id -> template
    template_by_slot: dict[str, dict] = {}
    if isinstance(slot_templates, Sequence):
        for template in slot_templates:
            if isinstance(template, Mapping):
                slot_id = template.get("slot_id")
                if slot_id:
                    template_by_slot[str(slot_id)] = template

    # Map canonical slots to slot_ids
    canonical_to_slotid = {
        "Weapon 1": "weapon_2h",
        "Weapon 2": "offhand",
        "Body Armour": "body_armour",
        "Helmet": "helmet",
        "Gloves": "gloves",
        "Boots": "boots",
        "Amulet": "amulet",
        "Ring 1": "ring_1",
        "Ring 2": "ring_2",
        "Belt": "belt",
        "Flask 1": "flask_1",
        "Flask 2": "flask_2",
        "Flask 3": "flask_3",
        "Flask 4": "flask_4",
        "Flask 5": "flask_5",
    }

    # Default base types for each slot (fallback)
    default_bases = {
        "weapon_2h": "Steel Khopesh",
        "offhand": "Spirit Shield",
        "body_armour": "Chestplate",
        "helmet": "Iron Cap",
        "gloves": "Chain Gloves",
        "boots": "Iron Greaves",
        "amulet": "Paua Amulet",
        "ring_1": "Gold Ring",
        "ring_2": "Gold Ring",
        "belt": "Heavy Belt",
        "flask_1": "Large Life Flask",
        "flask_2": "Large Mana Flask",
        "flask_3": "Quicksilver Flask",
        "flask_4": "Ruby Flask",
        "flask_5": "Diamond Flask",
    }

    # Render items for each canonical slot
    for idx, (slot, _) in enumerate(CANONICAL_ITEM_DESCRIPTIONS, start=1):
        item_id = str(idx)
        item_refs.append((slot, item_id))

        slot_id = canonical_to_slotid.get(slot)
        template = template_by_slot.get(slot_id) if slot_id else None

        # Get base type from template or use default
        if template and isinstance(template, Mapping):
            base_type = template.get("base_type", default_bases.get(slot_id, "Unknown Item"))
        else:
            base_type = default_bases.get(slot_id, "Unknown Item")

        # Build contribution-based affix lines
        affix_lines: list[str] = []
        if template and isinstance(template, Mapping):
            contributions = template.get("contributions", {})
            if isinstance(contributions, Mapping):
                life = contributions.get("life", 0)
                if life > 0:
                    affix_lines.append(f"+{life} to maximum Life")
                es = contributions.get("energy_shield", 0)
                if es > 0:
                    affix_lines.append(f"+{es} to maximum Energy Shield")
                strength = contributions.get("strength", 0)
                if strength > 0:
                    affix_lines.append(f"+{strength} to Strength")
                dexterity = contributions.get("dexterity", 0)
                if dexterity > 0:
                    affix_lines.append(f"+{dexterity} to Dexterity")
                intelligence = contributions.get("intelligence", 0)
                if intelligence > 0:
                    affix_lines.append(f"+{intelligence} to Intelligence")
                fire = contributions.get("fire", 0)
                if fire > 0:
                    affix_lines.append(f"+{fire}% to Fire Resistance")
                cold = contributions.get("cold", 0)
                if cold > 0:
                    affix_lines.append(f"+{cold}% to Cold Resistance")
                lightning = contributions.get("lightning", 0)
                if lightning > 0:
                    affix_lines.append(f"+{lightning}% to Lightning Resistance")
                chaos = contributions.get("chaos", 0)
                if chaos > 0:
                    affix_lines.append(f"+{chaos}% to Chaos Resistance")

        # Render the item as Rare
        lines.append(f'    <Item id="{item_id}">')
        lines.append("      Rarity: Rare")
        lines.append(f"      {base_type}")
        lines.append("      --------")
        lines.append("      Item Level: 100")
        for affix in affix_lines:
            lines.append(f"      {affix}")
        lines.append("    </Item>")

    lines.append('    <ItemSet id="1">')
    for slot, item_id in item_refs:
        lines.append(f'      <Slot name="{_xml_attr(slot)}" itemId="{item_id}"/>')
    lines.append("    </ItemSet>")
    lines.append("  </Items>")
    return lines


def _render_skills_section(candidate: Candidate) -> list[str]:
    gems_payload = candidate.build_details_payload.get("gems")
    if not isinstance(gems_payload, Mapping):
        gems_payload = {}
    groups: dict[str, Mapping[str, Any]] = {}
    raw_groups = gems_payload.get("groups")
    if isinstance(raw_groups, Sequence) and not isinstance(raw_groups, (str, bytes)):
        for raw_group in raw_groups:
            if not isinstance(raw_group, Mapping):
                continue
            group_id = raw_group.get("id")
            if group_id is None:
                continue
            groups[str(group_id)] = raw_group
    assignments: Sequence[Any] = ()
    socket_plan = gems_payload.get("socket_plan")
    if isinstance(socket_plan, Mapping):
        raw_assignments = socket_plan.get("assignments")
        if isinstance(raw_assignments, Sequence) and not isinstance(raw_assignments, (str, bytes)):
            assignments = raw_assignments
    slot_assignments: defaultdict[str, list[str]] = defaultdict(list)
    for assignment in assignments:
        if not isinstance(assignment, Mapping):
            continue
        slot_id = assignment.get("slot_id")
        group_id = assignment.get("group_id")
        if not slot_id or not group_id:
            continue
        slot_label = SLOT_ID_TO_SKILL_SLOT.get(slot_id)
        if not slot_label:
            continue
        slot_assignments[slot_label].append(str(group_id))
    lines = ['  <Skills activeSkillSet="1">', '    <SkillSet id="1" title="Generated">']
    for slot_label in SKILL_SLOT_ORDER:
        gem_ids: list[str] = []
        for group_id in slot_assignments.get(slot_label, []):
            group = groups.get(group_id)
            if not isinstance(group, Mapping):
                continue
            raw_gems = group.get("gems")
            if not (isinstance(raw_gems, Sequence) and not isinstance(raw_gems, (str, bytes))):
                continue
            for gem in raw_gems:
                if gem is None:
                    continue
                gem_ids.append(str(gem))
        if not gem_ids:
            fallback = FALLBACK_GEMS_BY_SLOT.get(
                slot_label, FALLBACK_GEMS_BY_SLOT.get("Weapon 1", ())
            )
            gem_ids = [str(gem) for gem in fallback if gem]
        deduped_gem_ids: list[str] = []
        seen_gems: set[str] = set()
        for gem_id in gem_ids:
            cleaned = str(gem_id).strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen_gems:
                continue
            seen_gems.add(key)
            deduped_gem_ids.append(cleaned)
        lines.append(f'      <Skill slot="{_xml_attr(slot_label)}" enabled="true">')
        for gem_id in deduped_gem_ids:
            name_spec = _gem_name_spec(gem_id)
            lines.append(
                '        <Gem nameSpec="'
                + _xml_attr(name_spec)
                + '" level="'
                + DEFAULT_GEM_LEVEL
                + '" quality="'
                + DEFAULT_GEM_QUALITY
                + '" enabled="true"/>'
            )
        lines.append("      </Skill>")
    lines.append("    </SkillSet>")
    lines.append("  </Skills>")
    return lines


def _build_worker_xml_payload(candidate: Candidate) -> str:
    scenario_payload: Mapping[str, Any] | None = None
    if isinstance(candidate.metrics_payload, Mapping):
        for entry in candidate.metrics_payload.values():
            if isinstance(entry, Mapping):
                scenario_payload = entry
                break
    scenario_payload = scenario_payload or {}

    metrics = scenario_payload.get("metrics") if isinstance(scenario_payload, Mapping) else {}
    defense = scenario_payload.get("defense") if isinstance(scenario_payload, Mapping) else {}
    resources = scenario_payload.get("resources") if isinstance(scenario_payload, Mapping) else {}
    reservation = (
        scenario_payload.get("reservation") if isinstance(scenario_payload, Mapping) else {}
    )
    attributes = scenario_payload.get("attributes") if isinstance(scenario_payload, Mapping) else {}
    if not isinstance(metrics, Mapping):
        metrics = {}
    if not isinstance(defense, Mapping):
        defense = {}
    if not isinstance(resources, Mapping):
        resources = {}
    if not isinstance(reservation, Mapping):
        reservation = {}
    if not isinstance(attributes, Mapping):
        attributes = {}

    resists = defense.get("resists") if isinstance(defense, Mapping) else {}
    if not isinstance(resists, Mapping):
        resists = {}

    def _metric(source: Mapping[str, Any], key: str, default: float = 0.0) -> float:
        value = _coerce_float(source.get(key))
        return default if value is None else float(value)

    full_dps = _metric(metrics, "full_dps", 0.0)
    max_hit = _metric(metrics, "max_hit", 0.0)
    life = _metric(resources, "life", 0.0)
    mana = _metric(resources, "mana", 0.0)
    armour = _metric(defense, "armour", 0.0)
    evasion = _metric(defense, "evasion", 0.0)
    energy_shield = _metric(resources, "energy_shield", 0.0)
    fire = _metric(resists, "fire", 0.0)
    cold = _metric(resists, "cold", 0.0)
    lightning = _metric(resists, "lightning", 0.0)
    chaos = _metric(resists, "chaos", 0.0)
    strength = _metric(attributes, "strength", 0.0)
    dexterity = _metric(attributes, "dexterity", 0.0)
    intelligence = _metric(attributes, "intelligence", 0.0)

    reserved_percent = _metric(reservation, "reserved_percent", 0.0)
    available_percent = _metric(
        reservation, "available_percent", max(0.0, 100.0 - reserved_percent)
    )
    mana_unreserved = max(0.0, min(100.0, available_percent))
    life_unreserved = max(0.0, min(100.0, 100.0 - reserved_percent))

    identity = (
        candidate.build_details_payload.get("identity")
        if isinstance(candidate.build_details_payload, Mapping)
        else {}
    )
    if not isinstance(identity, Mapping):
        identity = {}
    class_name = str(
        identity.get("class_name")
        or identity.get("class")
        or candidate.genome.class_name
        or "Witch"
    )
    ascendancy = str(identity.get("ascendancy") or candidate.genome.ascendancy or "")

    stats = [
        ("FullDPS", full_dps),
        ("MaximumHitTaken", max_hit),
        ("Life", life),
        ("Mana", mana),
        ("Armour", armour),
        ("Evasion", evasion),
        ("EnergyShield", energy_shield),
        ("FireResist", fire),
        ("ColdResist", cold),
        ("LightningResist", lightning),
        ("ChaosResist", chaos),
        ("Str", strength),
        ("Dex", dexterity),
        ("Int", intelligence),
        ("LifeUnreservedPercent", life_unreserved),
        ("ManaUnreservedPercent", mana_unreserved),
        ("EffectiveMovementSpeedMod", 1.0),
        ("BlockChance", 0.0),
        ("SpellBlockChance", 0.0),
    ]
    build_line = (
        '  <Build level="100" targetVersion="'
        + _xml_attr(POB_TARGET_VERSION)
        + '" className="'
        + _xml_attr(class_name)
        + '" ascendClassName="'
        + _xml_attr(ascendancy)
        + '" mainSocketGroup="1"/>'
    )
    lines = [
        "<PathOfBuilding>",
        build_line,
    ]
    lines.extend(_render_items_section(candidate))
    lines.extend(_render_skills_section(candidate))
    for stat_name, stat_value in stats:
        lines.append(
            '  <PlayerStat stat="'
            + stat_name
            + '" value="'
            + _format_xml_number(stat_value)
            + '"/>'
        )
    lines.append("</PathOfBuilding>")
    return "\n".join(lines)


def _socket_precheck_failed(plan: SocketPlan) -> bool:
    return bool(plan.issues)


def _attribute_precheck_failed(report: RepairReport) -> bool:
    deficits = report.remaining_deficits
    return any(
        getattr(deficits, attribute, 0) > 0
        for attribute in ("strength", "dexterity", "intelligence")
    )


def _reservation_precheck_failed(
    metrics: Mapping[str, Any], templates: Sequence[ScenarioTemplate]
) -> bool:
    for template in templates:
        payload = metrics.get(template.scenario_id)
        if not isinstance(payload, Mapping):
            continue
        reservation = payload.get("reservation")
        if not isinstance(reservation, Mapping):
            continue
        reserved_value = _coerce_float(reservation.get("reserved_percent"))
        if reserved_value is None:
            continue
        if reserved_value > template.gate_thresholds.reservation.max_percent:
            return True
    return False


def _gem_completeness_precheck_failed(build_details_payload: Mapping[str, Any]) -> bool:
    gems = build_details_payload.get("gems")
    if not isinstance(gems, Mapping):
        return True
    groups = gems.get("groups")
    if not isinstance(groups, Sequence) or not groups:
        return True
    normalized_groups = [group for group in groups if isinstance(group, Mapping)]
    if not normalized_groups:
        return True
    full_dps_group_id = gems.get("full_dps_group_id")
    main_group = None
    if full_dps_group_id is not None:
        for group in normalized_groups:
            if str(group.get("id")) == str(full_dps_group_id):
                main_group = group
                break
    if main_group is None:
        main_group = next(
            (
                group
                for group in normalized_groups
                if str(group.get("group_type", "")).strip().lower() == "damage"
            ),
            None,
        )
    if main_group is None:
        return True
    gems_value = main_group.get("gems")
    if isinstance(gems_value, Sequence) and not isinstance(gems_value, (str, bytes)):
        main_gems = [gem for gem in gems_value if gem is not None]
    else:
        main_gems = []
    return len(main_gems) < 2


def _item_completeness_precheck_failed(build_details_payload: Mapping[str, Any]) -> bool:
    items = build_details_payload.get("items")
    if not isinstance(items, Mapping):
        return True
    slot_templates = items.get("slot_templates")
    if isinstance(slot_templates, Sequence):
        normalized = [entry for entry in slot_templates if isinstance(entry, Mapping)]
        if len(normalized) < 5:
            return True
        for template in normalized:
            slot_id = str(template.get("slot_id", "")).strip()
            base_type = str(template.get("base_type", "")).strip()
            if not slot_id or not base_type:
                return True
        return False
    imported_items = items.get("items")
    if isinstance(imported_items, Sequence):
        normalized_imported = [entry for entry in imported_items if isinstance(entry, Mapping)]
        if len(normalized_imported) < 5:
            return True
        for entry in normalized_imported:
            slot = str(entry.get("slot", "")).strip()
            name = str(entry.get("name", "")).strip()
            if not slot or not name:
                return True
        return False
    return True


def _normalize_identity_label_value(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized or None


def _candidate_identity_label(
    candidate: Candidate,
    fields: str | Sequence[str],
    candidate_attr: str | None = None,
) -> str:
    if isinstance(fields, str):
        field_iter: Sequence[str] = (fields,)
    else:
        field_iter = fields
    build_details_payload = candidate.build_details_payload
    if isinstance(build_details_payload, Mapping):
        identity = build_details_payload.get("identity")
        if isinstance(identity, Mapping):
            for field in field_iter:
                normalized = _normalize_identity_label_value(identity.get(field))
                if normalized:
                    return normalized
    if candidate_attr:
        attr_value = getattr(candidate, candidate_attr, None)
        normalized = _normalize_identity_label_value(attr_value)
        if normalized:
            return normalized
    return "unknown"


def _candidate_niche_key(candidate: Candidate) -> tuple[str, str]:
    class_label = _candidate_identity_label(
        candidate,
        ("class", "class_name"),
        candidate_attr="class_name",
    )
    main_skill_label = _candidate_identity_label(
        candidate,
        ("main_skill", "main_skill_package"),
        candidate_attr="main_skill_package",
    )
    return class_label, main_skill_label


def _candidate_skill_label(candidate: Candidate) -> str:
    return _candidate_identity_label(
        candidate,
        ("main_skill", "main_skill_package"),
        candidate_attr="main_skill_package",
    )


def _candidate_ascendancy_label(candidate: Candidate) -> str:
    return _candidate_identity_label(candidate, "ascendancy", candidate_attr="ascendancy")


def _candidate_defense_label(candidate: Candidate) -> str:
    label = _candidate_identity_label(candidate, "defense_archetype")
    if label != "unknown":
        return label
    return (
        _normalize_identity_label_value(getattr(candidate.genome, "defense_archetype", None))
        or "unknown"
    )


def _select_diverse_top_candidates(
    ranked_candidates: Sequence[Candidate],
    *,
    top_k: int,
) -> list[Candidate]:
    if top_k <= 0 or not ranked_candidates:
        return []
    skill_cap = max(1, (top_k + 1) // 2)
    ascendancy_cap = max(1, (top_k + 2) // 3)
    defense_cap = max(1, (top_k + 2) // 3)
    skill_counts: dict[str, int] = {}
    ascendancy_counts: dict[str, int] = {}
    defense_counts: dict[str, int] = {}
    selected: list[Candidate] = []
    selected_build_ids: set[str] = set()

    def _increment_counts(candidate: Candidate) -> None:
        skill = _candidate_skill_label(candidate)
        ascendancy = _candidate_ascendancy_label(candidate)
        defense = _candidate_defense_label(candidate)
        skill_counts[skill] = skill_counts.get(skill, 0) + 1
        ascendancy_counts[ascendancy] = ascendancy_counts.get(ascendancy, 0) + 1
        defense_counts[defense] = defense_counts.get(defense, 0) + 1

    def _select_candidate(candidate: Candidate) -> None:
        selected.append(candidate)
        selected_build_ids.add(candidate.build_id)
        _increment_counts(candidate)

    selected_niches: set[tuple[str, str]] = set()

    for candidate in ranked_candidates:
        if len(selected) >= top_k:
            break
        niche = _candidate_niche_key(candidate)
        if niche in selected_niches:
            continue
        _select_candidate(candidate)
        selected_niches.add(niche)

    if len(selected) >= top_k:
        return selected

    for candidate in ranked_candidates:
        if len(selected) >= top_k:
            break
        if candidate.build_id in selected_build_ids:
            continue
        skill = _candidate_skill_label(candidate)
        ascendancy = _candidate_ascendancy_label(candidate)
        defense = _candidate_defense_label(candidate)
        if skill_counts.get(skill, 0) >= skill_cap:
            continue
        if ascendancy_counts.get(ascendancy, 0) >= ascendancy_cap:
            continue
        if defense_counts.get(defense, 0) >= defense_cap:
            continue
        _select_candidate(candidate)

    if len(selected) >= top_k:
        return selected

    for candidate in ranked_candidates:
        if len(selected) >= top_k:
            break
        if candidate.build_id in selected_build_ids:
            continue
        _select_candidate(candidate)

    return selected


def _build_generation_record(
    seed: int,
    build_id: str,
    precheck_failures: list[str],
    evaluation_status: str | None,
    gate_pass: bool | None,
    evaluation_error: dict[str, Any] | None,
    stage_label: str,
    source: str,
    optimizer_iteration: int,
    parent_build_id: str | None,
    parent_seed: int | None,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "build_id": build_id,
        "precheck_failures": precheck_failures,
        "evaluation_status": evaluation_status,
        "gate_pass": gate_pass,
        "evaluation_error": evaluation_error,
        "stage_label": stage_label,
        "source": source,
        "optimizer_iteration": optimizer_iteration,
        "parent_build_id": parent_build_id,
        "parent_seed": parent_seed,
    }


def _default_socket_planner(catalog: SkillCatalog, genome: GenomeV0) -> SocketPlan:
    return plan_sockets(catalog, genome)


def run_generation(
    *,
    count: int,
    seed_start: int,
    ruleset_id: str,
    profile_id: str,
    run_id: str | None = None,
    base_path: Path | None = None,
    repo: ClickhouseRepository | None = None,
    evaluator: BuildEvaluator | None = None,
    socket_planner: SocketPlanner | None = None,
    passive_planner: Callable[[GenomeV0, int], PassiveTreePlan] | None = None,
    template_builder: TemplateBuilder | None = None,
    metrics_generator: MetricsGenerator | None = None,
    surrogate_enabled: bool = False,
    surrogate_model_path: Path | str | None = None,
    surrogate_exploration_pct: float = 0.2,
    surrogate_top_k: int | None = None,
    surrogate_predictor: Callable[[Sequence[dict[str, Any]]], Sequence[float]] | None = None,
    run_mode: Literal["standard", "optimizer"] = "standard",
    optimizer_iterations: int = 1,
    optimizer_elite_count: int = 2,
    constraints: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if count <= 0:
        raise ValueError("count must be positive")
    if seed_start < 0:
        raise ValueError("seed_start must be non-negative")
    if run_mode not in {"standard", "optimizer"}:
        raise ValueError("run_mode must be either 'standard' or 'optimizer'")
    if optimizer_iterations <= 0:
        raise ValueError("optimizer_iterations must be positive")
    if optimizer_elite_count <= 0:
        raise ValueError("optimizer_elite_count must be positive")
    exploration_pct = max(0.0, min(1.0, surrogate_exploration_pct))
    top_k = surrogate_top_k if surrogate_top_k and surrogate_top_k > 0 else None

    constraint_spec = ConstraintSpec.from_payload(constraints)

    base_path = Path(base_path) if base_path is not None else settings.data_path
    base_path.mkdir(parents=True, exist_ok=True)

    surrogate_model_path_obj = Path(surrogate_model_path) if surrogate_model_path else None

    catalog = load_default_skill_catalog()
    templates = [template for template in list_templates() if template.profile_id == profile_id]
    if not templates:
        raise ValueError(f"no scenario templates for profile_id={profile_id}")

    repo = repo or ClickhouseRepository()
    evaluator = evaluator or BuildEvaluator(repo=repo, base_path=base_path)
    if run_mode == "optimizer":
        evaluator.require_worker_metrics_for_profile(profile_id)
        if profile_id.strip().lower() == "uber_pinnacle":
            evaluator.require_non_stub_metrics_for_profile(profile_id)
    assert repo is not None
    assert evaluator is not None
    session_id = run_id or uuid.uuid4().hex
    candidates: list[Candidate] = []
    precheck_counts: dict[str, int] = {category: 0 for category in PRECHECK_CATEGORIES}

    socket_planner = socket_planner or _default_socket_planner
    passive_planner = passive_planner or (
        lambda genome, budget: build_passive_tree_plan(genome, budget)
    )
    template_builder = template_builder or build_item_templates
    metrics_generator = metrics_generator or _default_metrics_generator
    used_seeds: set[int] = set()
    optimizer_stage_reports: list[dict[str, Any]] = []
    optimizer_selection_history: list[dict[str, Any]] = []

    def _build_candidate(
        seed: int,
        stage_label: str | None = None,
        source: str | None = None,
        optimizer_iteration: int = 0,
        parent_build_id: str | None = None,
        parent_seed: int | None = None,
    ) -> Candidate:
        genome = deterministic_genome_from_seed(seed)
        gem_plan = catalog.build_plan(genome)
        passive_plan = passive_planner(genome, _point_budget_for(genome))
        socket_plan = socket_planner(catalog, genome)
        template_plan = template_builder(genome, gem_plan, passive_plan, socket_plan)
        build_details_payload = build_details_from_generation(
            genome=genome,
            gem_plan=gem_plan,
            passive_plan=passive_plan,
            socket_plan=socket_plan,
            template_plan=template_plan,
        )
        metrics_payload = metrics_generator(seed, templates)

        failures: list[str] = []
        if _socket_precheck_failed(socket_plan):
            failures.append("socket_precheck_failed")
        if _attribute_precheck_failed(template_plan.repair_report):
            failures.append("attribute_precheck_failed")
        if _reservation_precheck_failed(metrics_payload, templates):
            failures.append("reservation_precheck_failed")
        if _gem_completeness_precheck_failed(build_details_payload):
            failures.append("gem_completeness_precheck_failed")
        if _item_completeness_precheck_failed(build_details_payload):
            failures.append("item_completeness_precheck_failed")
        for category in failures:
            if category in precheck_counts:
                precheck_counts[category] += 1

        build_id = uuid.uuid4().hex
        code_payload = _render_build_code(genome, socket_plan, template_plan, metrics_payload)
        candidate = Candidate(
            seed=seed,
            build_id=build_id,
            main_skill_package=genome.main_skill_package,
            class_name=genome.class_name,
            ascendancy=genome.ascendancy,
            budget_tier=genome.budget_tier,
            failures=list(failures),
            metrics_payload=metrics_payload,
            genome=genome,
            code_payload=code_payload,
            build_details_payload=build_details_payload,
        )
        if stage_label:
            candidate.stage_label = stage_label
        if source:
            candidate.source = source
        candidate.optimizer_iteration = optimizer_iteration
        candidate.parent_build_id = parent_build_id
        candidate.parent_seed = parent_seed
        used_seeds.add(seed)
        candidates.append(candidate)
        return candidate

    def _persist_candidate(candidate: Candidate) -> None:
        if candidate.persisted:
            return
        xml_payload = _build_worker_xml_payload(candidate)
        artifacts = write_build_artifacts(
            candidate.build_id,
            xml=xml_payload,
            code=candidate.code_payload,
            genome=asdict(candidate.genome),
            raw_metrics=candidate.metrics_payload,
            build_details=candidate.build_details_payload,
            base_path=base_path,
        )
        status_value = (
            BuildStatus.failed.value if candidate.failures else BuildStatus.imported.value
        )
        repo.insert_build(
            BuildInsertPayload(
                build_id=candidate.build_id,
                created_at=datetime.now(timezone.utc),
                ruleset_id=ruleset_id,
                profile_id=profile_id,
                class_=candidate.genome.class_name,
                ascendancy=candidate.genome.ascendancy,
                main_skill=candidate.genome.main_skill_package,
                damage_type=candidate.genome.main_skill_package,
                defence_type=candidate.genome.defense_archetype,
                complexity_bucket=candidate.genome.budget_tier,
                pob_xml_path=str(artifacts.paths.build_xml),
                pob_code_path=str(artifacts.paths.code),
                genome_path=str(artifacts.paths.genome),
                tags=["generated"],
                status=status_value,
            )
        )
        candidate.persisted = True

    def _cleanup_candidate(candidate: Candidate) -> None:
        if not candidate.persisted:
            return
        database_rows_purged = False
        try:
            repo.purge_build(candidate.build_id)
            database_rows_purged = True
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning(
                "failed to purge database rows for %s: %s",
                candidate.build_id,
                exc,
            )
        if database_rows_purged:
            try:
                purge_build_artifacts(candidate.build_id, base_path)
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning(
                    "failed to purge artifacts for %s: %s",
                    candidate.build_id,
                    exc,
                )
        candidate.persisted = False

    if run_mode == "standard":
        for index in range(count):
            _build_candidate(seed_start + index)
    else:
        warmup_candidates: list[Candidate] = []
        for index in range(count):
            warmup_candidates.append(
                _build_candidate(
                    seed_start + index,
                    stage_label="optimizer_warmup",
                    source="optimizer_warmup",
                )
            )
        optimizer_stage_reports.append({"stage": "warmup", "count": len(warmup_candidates)})
        elites = _select_optimizer_elites(candidates, optimizer_elite_count)
        optimizer_selection_history.append(_optimizer_selection_record("warmup", 0, elites))
        for iteration in range(1, optimizer_iterations + 1):
            if not elites:
                break
            new_stage_candidates: list[Candidate] = []
            for rank, (elite_candidate, _) in enumerate(elites):
                child_seed = _derive_optimizer_seed(
                    elite_candidate.seed, iteration, rank, used_seeds
                )
                if child_seed is None:
                    continue
                new_stage_candidates.append(
                    _build_candidate(
                        child_seed,
                        stage_label="optimizer_iteration",
                        source="optimizer_iteration",
                        optimizer_iteration=iteration,
                        parent_build_id=elite_candidate.build_id,
                        parent_seed=elite_candidate.seed,
                    )
                )
            if not new_stage_candidates:
                break
            optimizer_stage_reports.append(
                {"stage": f"iteration_{iteration}", "count": len(new_stage_candidates)}
            )
            elites = _select_optimizer_elites(candidates, optimizer_elite_count)
            optimizer_selection_history.append(
                _optimizer_selection_record(f"iteration_{iteration}", iteration, elites)
            )
    evaluable_candidates = [candidate for candidate in candidates if candidate.evaluable]

    surrogate_summary: dict[str, Any] = {
        "enabled": surrogate_enabled,
        "status": "disabled",
        "model_path": str(surrogate_model_path_obj) if surrogate_model_path_obj else None,
        "model_id": None,
        "selection_params": {
            "top_k": top_k,
            "exploration_pct": exploration_pct,
        },
        "counts": {
            "candidates": len(evaluable_candidates),
            "selected": 0,
            "pruned": 0,
        },
        "fallback_reason": None,
    }
    prediction_records: list[dict[str, Any]] = []
    predictor_model_id: str | None = None
    prediction_timestamp: str | None = None
    selected_candidates: list[Candidate] = []

    if surrogate_enabled and evaluable_candidates:
        prediction_timestamp = datetime.now(timezone.utc).isoformat()
        try:
            if surrogate_predictor is not None:
                predictor = surrogate_predictor
                predictor_model_id = getattr(surrogate_predictor, "__name__", "custom_predictor")
            else:
                if not surrogate_model_path_obj:
                    raise ValueError(
                        (
                            "surrogate_model_path is required when surrogate is enabled "
                            "without a predictor"
                        )
                    )
                predictor_model_id, predictor = _load_surrogate_predictor(surrogate_model_path_obj)
            feature_rows = [_candidate_feature_row(candidate) for candidate in evaluable_candidates]
            predictions = predictor(feature_rows)
            if len(predictions) != len(feature_rows):
                raise ValueError(
                    "surrogate predictor returned %d predictions for %d candidates"
                    % (len(predictions), len(feature_rows))
                )
            for candidate, prediction in zip(evaluable_candidates, predictions, strict=True):
                metrics, probability = _normalize_prediction_entry(prediction)
                candidate.predicted_metrics = metrics or {}
                candidate.pass_probability = probability
                candidate.predicted_full_dps = candidate.predicted_metrics.get("full_dps", 0.0)
            scored_candidates = sorted(
                evaluable_candidates,
                key=lambda item: item.predicted_full_dps or 0.0,
                reverse=True,
            )
            if top_k is None:
                top_candidates = scored_candidates
                remaining = []
            else:
                top_candidates = _select_diverse_top_candidates(
                    scored_candidates,
                    top_k=top_k,
                )
                selected_ids = {candidate.build_id for candidate in top_candidates}
                remaining = [
                    candidate
                    for candidate in scored_candidates
                    if candidate.build_id not in selected_ids
                ]
            exploration_candidates: list[Candidate] = []
            if remaining and exploration_pct > 0:
                target = int(len(remaining) * exploration_pct)
                if target == 0:
                    target = 1
                exploration_count = min(target, len(remaining))
                rng = random.Random(seed_start)
                exploration_candidates = rng.sample(remaining, exploration_count)
            for candidate in top_candidates:
                candidate.selected_for_evaluation = True
                candidate.selection_reason = "surrogate_top"
            for candidate in exploration_candidates:
                candidate.selected_for_evaluation = True
                candidate.selection_reason = "surrogate_exploration"
            pruned_candidates = [
                candidate for candidate in remaining if candidate not in exploration_candidates
            ]
            for candidate in pruned_candidates:
                candidate.selection_reason = "surrogate_pruned"
            selected_candidates = top_candidates + exploration_candidates
            surrogate_summary["model_id"] = predictor_model_id
            surrogate_summary["status"] = "active"
            for candidate in evaluable_candidates:
                candidate.surrogate_prediction_payload = _prediction_payload(
                    candidate,
                    surrogate_summary["model_id"],
                    surrogate_summary["model_path"],
                    prediction_timestamp,
                )
            prediction_records = [
                _build_prediction_record(
                    candidate,
                    SURROGATE_PREDICTION_SCHEMA_VERSION,
                    surrogate_summary["model_id"],
                    surrogate_summary["model_path"],
                    prediction_timestamp,
                )
                for candidate in evaluable_candidates
            ]
        except Exception as exc:  # pylint: disable=broad-except
            surrogate_summary["status"] = "fallback"
            surrogate_summary["fallback_reason"] = str(exc)
            selected_candidates = evaluable_candidates[:]
            for candidate in selected_candidates:
                candidate.selected_for_evaluation = True
    else:
        selected_candidates = evaluable_candidates[:]
        for candidate in selected_candidates:
            candidate.selected_for_evaluation = True

    selected_count = sum(1 for candidate in candidates if candidate.selected_for_evaluation)
    surrogate_summary["counts"]["selected"] = selected_count
    surrogate_summary["counts"]["pruned"] = max(0, len(evaluable_candidates) - selected_count)

    evaluation_rows: list[ScenarioMetricRow] = []
    evaluation_records: list[dict[str, Any]] = []
    evaluation_attempted = 0
    evaluation_successes = 0
    evaluation_failures = 0
    evaluation_errors = 0

    for candidate in candidates:
        if not candidate.selected_for_evaluation:
            continue
        evaluation_attempted += 1
        try:
            _persist_candidate(candidate)
            status, rows = evaluator.evaluate_build(candidate.build_id)
            candidate.evaluation_status = status.value
            if rows:
                candidate.gate_pass = all(getattr(row, "gate_pass", True) for row in rows)
            candidate.verified_metrics_payload = _metrics_payload_from_scenario_rows(rows)
            evaluation_rows.extend(rows)
            if status is BuildStatus.evaluated:
                evaluation_successes += 1
                if candidate.surrogate_prediction_payload:
                    _persist_surrogate_prediction(
                        candidate.build_id,
                        base_path,
                        candidate.surrogate_prediction_payload,
                    )
            else:
                evaluation_failures += 1
            evaluation_records.append(
                {
                    "build_id": candidate.build_id,
                    "status": candidate.evaluation_status,
                    "error": None,
                }
            )
        except Exception as exc:  # pylint: disable=broad-except
            candidate.evaluation_error = {
                "code": getattr(exc, "code", "generation_error"),
                "message": getattr(exc, "message", str(exc)),
                "details": getattr(exc, "details", None),
            }
            evaluation_records.append(
                {
                    "build_id": candidate.build_id,
                    "status": None,
                    "error": candidate.evaluation_error,
                }
            )
            evaluation_errors += 1
            evaluation_failures += 1
            candidate.evaluation_status = BuildStatus.failed.value

    valid_generation_records: list[dict[str, Any]] = []
    generation_attempt_records: list[dict[str, Any]] = []
    for candidate in candidates:
        if constraint_spec:
            base_metrics = candidate.verified_metrics_payload or candidate.metrics_payload
            evaluation = evaluate_constraints(base_metrics, constraint_spec)
            if (
                evaluation.status == "unknown"
                and candidate.metrics_payload
                and candidate.metrics_payload is not base_metrics
            ):
                fallback = evaluate_constraints(candidate.metrics_payload, constraint_spec)
                if fallback.status != "unknown" or fallback.violated_constraints:
                    evaluation = fallback
            candidate.constraint_status = evaluation.status
            candidate.constraint_reason_code = evaluation.reason_code
            candidate.violated_constraints = list(evaluation.violated_constraints)
            candidate.constraint_checked_at = evaluation.checked_at
            artifact_payload = constraint_artifact_payload(constraint_spec, evaluation)
            checked_at_value: datetime | None = None
            if candidate.constraint_checked_at:
                try:
                    checked_at_value = datetime.fromisoformat(candidate.constraint_checked_at)
                except ValueError:
                    checked_at_value = None
            if candidate.persisted and candidate.evaluation_status == BuildStatus.evaluated.value:
                write_build_constraints(
                    candidate.build_id,
                    artifact_payload,
                    base_path=base_path,
                )
                repo.update_build_constraints(
                    candidate.build_id,
                    constraint_status=candidate.constraint_status,
                    constraint_reason_code=candidate.constraint_reason_code,
                    violated_constraints=candidate.violated_constraints or [],
                    constraint_checked_at=checked_at_value,
                )
        record = _build_generation_record(
            candidate.seed,
            candidate.build_id,
            candidate.failures,
            candidate.evaluation_status,
            candidate.gate_pass,
            candidate.evaluation_error,
            candidate.stage_label,
            candidate.source,
            candidate.optimizer_iteration,
            candidate.parent_build_id,
            candidate.parent_seed,
        )
        if candidate.selection_reason:
            record["surrogate_selection_reason"] = candidate.selection_reason
        if candidate.constraint_status is not None:
            record["constraint_status"] = candidate.constraint_status
            record["constraint_reason_code"] = candidate.constraint_reason_code
            record["violated_constraints"] = candidate.violated_constraints or []
            record["constraint_checked_at"] = candidate.constraint_checked_at
        record["persisted"] = candidate.persisted
        generation_attempt_records.append(record)
        if candidate.persisted and candidate.evaluation_status == BuildStatus.evaluated.value:
            valid_generation_records.append(record)

    archive_store = ArchiveStore()
    for candidate in candidates:
        if candidate.evaluation_status != BuildStatus.evaluated.value:
            continue
        metrics_payload = candidate.verified_metrics_payload or candidate.metrics_payload
        descriptor = descriptor_values_from_metrics(metrics_payload, archive_store.axes)
        score = score_from_metrics(metrics_payload)
        metadata: dict[str, Any] = {
            "seed": candidate.seed,
            "budget_tier": candidate.budget_tier,
            "status": candidate.evaluation_status,
            "metrics_source": "verified",
        }
        if candidate.failures:
            metadata["failures"] = list(candidate.failures)
        archive_store.insert(candidate.build_id, score, descriptor, metadata=metadata)

    emitters = (ExploitEmitter(), NoveltyEmitter(), UncertaintyEmitter())
    budgets = deterministic_allocator(len(candidates), emitters)
    entries = archive_store.entries()
    emitter_summaries: list[dict[str, int]] = []
    archive_summary_payload: dict[str, Any] | None = None
    archive_path: Path | None = None
    archive_created_at: str | None = None

    if entries:
        for emitter in emitters:
            budget = budgets.get(emitter.name, 0)
            selected = emitter.select(entries, budget)
            emitter_summaries.append(
                {
                    "name": emitter.name,
                    "budget": budget,
                    "selected": len(selected),
                }
            )
        archive_created_at = datetime.now(timezone.utc).isoformat()
        archive_path = persist_archive(
            session_id,
            archive_store,
            base_path=base_path,
            created_at=archive_created_at,
        )
        archive_summary_payload = {
            "metrics": archive_store.metrics_dict(),
            "axes": [axis.to_dict() for axis in archive_store.axes],
            "created_at": archive_created_at,
        }
    else:
        for emitter in emitters:
            emitter_summaries.append(
                {
                    "name": emitter.name,
                    "budget": budgets.get(emitter.name, 0),
                    "selected": 0,
                }
            )

    evaluation_stats: dict[str, int] = {
        "attempted": evaluation_attempted,
        "successes": evaluation_successes,
        "failures": evaluation_failures,
        "errors": evaluation_errors,
    }
    status_reason: dict[str, Any] | None = None
    if evaluation_attempted == 0:
        run_status = "failed"
        status_reason = {
            "code": "no_evaluation_attempts",
            "message": "no evaluations were attempted",
            "evaluation": evaluation_stats,
        }
    elif evaluation_successes == 0:
        run_status = "failed"
        status_reason = {
            "code": "no_verified_evaluations",
            "message": "no PoB-verified evaluations succeeded",
            "evaluation": evaluation_stats,
        }
    elif evaluation_failures > 0:
        run_status = "partial"
        status_reason = {
            "code": "partial_evaluation_results",
            "message": f"{evaluation_failures} evaluation(s) failed",
            "evaluation": evaluation_stats,
        }
    else:
        run_status = "completed"

    summary = {
        "run_id": session_id,
        "status": run_status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "count": count,
            "seed_start": seed_start,
            "ruleset_id": ruleset_id,
            "profile_id": profile_id,
            "run_mode": run_mode,
            "optimizer_iterations": optimizer_iterations,
            "optimizer_elite_count": optimizer_elite_count,
            "surrogate_enabled": surrogate_enabled,
            "surrogate_model_path": (
                str(surrogate_model_path_obj) if surrogate_model_path_obj else None
            ),
            "surrogate_top_k": top_k,
            "surrogate_exploration_pct": exploration_pct,
        },
        "generation": {
            "count": count,
            "processed": len(candidates),
            "precheck_failures": precheck_counts,
            "records": valid_generation_records,
            "attempted": evaluation_attempted,
            "attempt_records": generation_attempt_records,
        },
        "evaluation": {
            "attempted": evaluation_attempted,
            "successes": evaluation_successes,
            "failures": evaluation_failures,
            "errors": evaluation_errors,
            "records": [
                record for record in evaluation_records if record["status"] or record["error"]
            ],
        },
        "surrogate": {
            "enabled": surrogate_summary["enabled"],
            "status": surrogate_summary["status"],
            "model_id": surrogate_summary["model_id"],
            "model_path": surrogate_summary["model_path"],
            "selection_params": surrogate_summary["selection_params"],
            "counts": surrogate_summary["counts"],
        },
        "paths": {},
    }
    summary["parameters"]["constraints"] = constraints
    summary["constraints"] = constraints
    if status_reason:
        summary["status_reason"] = status_reason

    if surrogate_summary["fallback_reason"]:
        summary["surrogate"]["fallback_reason"] = surrogate_summary["fallback_reason"]

    summary["optimizer"] = {
        "enabled": run_mode == "optimizer",
        "mode": run_mode,
        "iterations": optimizer_iterations,
        "elite_count": optimizer_elite_count,
        "stage_counts": optimizer_stage_reports,
        "selection_history": optimizer_selection_history,
        "status": "completed" if run_mode == "optimizer" else "standard",
    }

    summary["archive"] = archive_summary_payload
    summary["emitters"] = emitter_summaries
    if archive_path:
        summary["paths"]["archive"] = str(archive_path)

    invalid_evaluations = evaluation_failures + evaluation_errors
    if evaluation_successes == 0:
        logger.warning(
            "generation run %s emitted no PoB-verified builds (attempted=%d invalid=%d)",
            session_id,
            evaluation_attempted,
            invalid_evaluations,
        )
    else:
        logger.info(
            "generation run %s emitted %d verified build(s) (attempted=%d invalid=%d, status=%s)",
            session_id,
            evaluation_successes,
            evaluation_attempted,
            invalid_evaluations,
            run_status,
        )

    benchmark_payload = _benchmark_summary_from_rows(evaluation_rows)
    benchmark_path = _persist_run_artifact(
        session_id,
        benchmark_payload,
        "benchmark_summary.json",
        base_path=base_path,
    )
    summary["benchmark"] = benchmark_payload
    summary["paths"]["benchmark_summary"] = str(benchmark_path)

    ml_payload = _build_ml_lifecycle_payload(
        surrogate_enabled=surrogate_enabled,
        surrogate_summary=surrogate_summary,
        surrogate_model_path=surrogate_model_path_obj,
    )
    ml_path = _persist_run_artifact(
        session_id,
        ml_payload,
        "ml_lifecycle.json",
        base_path=base_path,
    )
    summary["ml_lifecycle"] = ml_payload
    summary["paths"]["ml_lifecycle"] = str(ml_path)

    summary_path = _run_summary_path(session_id, base_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    predictions_path: Path | None = None
    if surrogate_enabled:
        predictions_path = summary_path.parent / "surrogate_predictions.json"
        artifact_payload: dict[str, Any] = {
            "schema_version": SURROGATE_PREDICTION_SCHEMA_VERSION,
            "timestamp": prediction_timestamp or datetime.now(timezone.utc).isoformat(),
            "model_id": surrogate_summary["model_id"],
            "model_path": surrogate_summary["model_path"],
            "status": surrogate_summary["status"],
            "selection_params": surrogate_summary["selection_params"],
            "counts": surrogate_summary["counts"],
            "fallback_reason": surrogate_summary["fallback_reason"],
            "candidates": prediction_records,
        }
        predictions_path.write_text(
            json.dumps(artifact_payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        summary["paths"]["surrogate_predictions"] = str(predictions_path)

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary["paths"]["summary"] = str(summary_path)
    logger.info("generation run %s written to %s", session_id, summary_path)
    return summary


def _run_summary_path(run_id: str, base_path: Path | None = None) -> Path:
    safe_run_id = _validate_run_id(run_id)
    runs_root = (Path(base_path or settings.data_path) / "runs").resolve()
    summary_path = (runs_root / safe_run_id / "summary.json").resolve()
    if runs_root not in summary_path.parents:
        raise ValueError("run_id resolved outside run directory")
    return summary_path


def _validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_' or '-'"
        )
    return run_id


def _benchmark_summary_from_rows(
    rows: Sequence[ScenarioMetricRow],
) -> dict[str, Any]:
    grouped: dict[str, list[ScenarioMetricRow]] = defaultdict(list)
    for row in rows:
        grouped[row.scenario_id].append(row)
    summary_payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenarios": {},
    }
    for scenario_id, entries in grouped.items():
        sample_count = len(entries)
        summary_payload["scenarios"][scenario_id] = {
            "samples": sample_count,
            "median_full_dps": _median_or_zero([entry.full_dps for entry in entries]),
            "median_max_hit": _median_or_zero([entry.max_hit for entry in entries]),
            "median_utility_score": _median_or_zero([entry.utility_score for entry in entries]),
            "gate_pass_rate": (
                sum(1 for entry in entries if entry.gate_pass) / sample_count
                if sample_count
                else 0.0
            ),
        }
    return summary_payload


def _median_or_zero(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(median(values))


def _build_ml_lifecycle_payload(
    *,
    surrogate_enabled: bool,
    surrogate_summary: dict[str, Any],
    surrogate_model_path: Path | None,
) -> dict[str, Any]:
    model_meta, meta_error, meta_path = _load_model_meta(
        surrogate_model_path,
        surrogate_enabled,
    )
    return {
        "enabled": surrogate_enabled,
        "status": surrogate_summary["status"],
        "model": {
            "model_id": surrogate_summary["model_id"],
            "path": str(surrogate_model_path) if surrogate_model_path else None,
        },
        "selection_params": surrogate_summary["selection_params"],
        "counts": surrogate_summary["counts"],
        "fallback_reason": surrogate_summary.get("fallback_reason"),
        "metadata": {
            "model_meta": model_meta,
            "meta_path": str(meta_path) if meta_path else None,
            "error": meta_error,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _load_model_meta(
    model_path: Path | None,
    surrogate_enabled: bool,
) -> tuple[dict[str, Any] | None, str | None, Path | None]:
    if not surrogate_enabled:
        return None, "surrogate disabled", None
    if not model_path:
        return None, "surrogate_model_path not provided", None
    candidate_dir = model_path if model_path.is_dir() else model_path.parent
    meta_path = candidate_dir / "model_meta.json"
    if not meta_path.exists():
        return None, f"model_meta.json not found at {meta_path}", meta_path
    try:
        return json.loads(meta_path.read_text(encoding="utf-8")), None, meta_path
    except Exception as exc:
        return None, f"model_meta.json read error: {exc}", meta_path


def _run_artifact_path(
    run_id: str,
    filename: str,
    base_path: Path | None = None,
) -> Path:
    safe_run_id = _validate_run_id(run_id)
    runs_root = (Path(base_path or settings.data_path) / "runs").resolve()
    artifact_path = (runs_root / safe_run_id / filename).resolve()
    if runs_root not in artifact_path.parents:
        raise ValueError("run_id resolved outside run directory")
    return artifact_path


def _persist_run_artifact(
    run_id: str,
    payload: dict[str, Any],
    filename: str,
    base_path: Path | None = None,
) -> Path:
    artifact_path = _run_artifact_path(run_id, filename, base_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return artifact_path


def load_run_summary(run_id: str, base_path: Path | None = None) -> dict[str, Any]:
    path = _run_summary_path(run_id, base_path)
    if not path.exists():
        raise FileNotFoundError(f"run summary {run_id} not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction_payload(
    candidate: Candidate,
    model_id: str | None,
    model_path: str | None,
    timestamp: str,
) -> dict[str, Any]:
    return {
        "schema_version": SURROGATE_PREDICTION_SCHEMA_VERSION,
        "model_id": model_id,
        "model_path": model_path,
        "predicted_metrics": candidate.predicted_metrics or {},
        "pass_probability": candidate.pass_probability,
        "selection_reason": candidate.selection_reason,
        "timestamp": timestamp,
    }


def _build_prediction_record(
    candidate: Candidate,
    schema_version: int,
    model_id: str | None,
    model_path: str | None,
    timestamp: str,
) -> dict[str, Any]:
    payload = _prediction_payload(candidate, model_id, model_path, timestamp)
    payload.update(
        {
            "schema_version": schema_version,
            "build_id": candidate.build_id,
            "seed": candidate.seed,
            "main_skill_package": candidate.main_skill_package,
            "selected": candidate.selected_for_evaluation,
        }
    )
    return payload


def _persist_surrogate_prediction(build_id: str, base_path: Path, payload: dict[str, Any]) -> None:
    prediction_path = artifact_paths(build_id, base_path).surrogate_prediction
    try:
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning(
            "failed to write surrogate prediction for %s: %s",
            build_id,
            exc,
        )


__all__ = [
    "run_generation",
    "load_run_summary",
    "_run_summary_path",
    "PRECHECK_CATEGORIES",
]
