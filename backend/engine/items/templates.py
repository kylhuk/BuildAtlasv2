"""Deterministic item template helpers for EP-V2-05."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

from ..genome import GenomeV0
from ..passives.builder import PassiveTreePlan
from ..skills.catalog import GemPlan
from ..sockets.planner import SocketPlan

RESIST_KEYS = ("fire", "cold", "lightning", "chaos")
ATTRIBUTE_KEYS = ("strength", "dexterity", "intelligence")
ALL_STAT_KEYS = (*RESIST_KEYS, *ATTRIBUTE_KEYS, "life", "energy_shield")

RESIST_BASELINE = {"fire": 90, "cold": 90, "lightning": 90, "chaos": 60}
ATTRIBUTE_BASELINE = {"strength": 170, "dexterity": 140, "intelligence": 140}
LIFE_BASELINE = 900
ES_BASELINE = 600

MAX_REPAIR_PASSES = 4

ARCHETYPE_PRIORITY_MAP = {
    "armour": (
        "strength",
        "life",
        "fire_res",
        "cold_res",
        "lightning_res",
        "chaos_res",
    ),
    "evasion": (
        "dexterity",
        "life",
        "fire_res",
        "cold_res",
        "lightning_res",
    ),
    "energy_shield": (
        "intelligence",
        "energy_shield",
        "lightning_res",
        "chaos_res",
    ),
    "hybrid": (
        "strength",
        "dexterity",
        "life",
        "energy_shield",
        "fire_res",
    ),
}

LIFE_METRIC_BY_ARCHETYPE = {
    "armour": "life",
    "evasion": "life",
    "energy_shield": "energy_shield",
    "hybrid": "life",
}

_ATTRIBUTE_FOCUS_KEYWORDS = {
    "strength": ("cyclone", "blade", "flurry", "smash", "strike", "mace", "sunder"),
    "dexterity": ("tornado", "shot", "projectile", "rain", "toxic", "bow"),
    "intelligence": (
        "arc",
        "vortex",
        "essence",
        "drain",
        "summon",
        "frost",
        "spell",
        "orb",
    ),
}


@dataclass(frozen=True)
class SlotProfile:
    base_type: str
    adjustable: bool
    base_resists: Tuple[int, int, int, int]
    base_attributes: Tuple[int, int, int]
    base_life: int
    base_energy_shield: int


@dataclass
class TemplateContributions:
    fire: int = 0
    cold: int = 0
    lightning: int = 0
    chaos: int = 0
    strength: int = 0
    dexterity: int = 0
    intelligence: int = 0
    life: int = 0
    energy_shield: int = 0

    def increment(self, field: str, amount: int) -> None:
        if amount <= 0:
            return
        current = getattr(self, field)
        setattr(self, field, current + amount)


@dataclass
class SlotTemplate:
    slot_id: str
    base_type: str
    archetype_priorities: Tuple[str, ...]
    contributions: TemplateContributions
    adjustable: bool


@dataclass(frozen=True)
class RequirementDeficits:
    fire: int
    cold: int
    lightning: int
    chaos: int
    strength: int
    dexterity: int
    intelligence: int
    life: int
    energy_shield: int

    def total(self) -> int:
        return (
            self.fire
            + self.cold
            + self.lightning
            + self.chaos
            + self.strength
            + self.dexterity
            + self.intelligence
            + self.life
            + self.energy_shield
        )


@dataclass(frozen=True)
class RepairReport:
    iterations: int
    initial_deficits: RequirementDeficits
    remaining_deficits: RequirementDeficits


@dataclass(frozen=True)
class ItemTemplatePlan:
    genome: GenomeV0
    templates: Tuple[SlotTemplate, ...]
    repair_report: RepairReport


SLOT_PROFILES: dict[str, SlotProfile] = {
    "weapon_2h": SlotProfile(
        base_type="Karui Maul",
        adjustable=False,
        base_resists=(2, 1, 1, 1),
        base_attributes=(25, 5, 0),
        base_life=30,
        base_energy_shield=0,
    ),
    "body_armour": SlotProfile(
        base_type="Plate Vest",
        adjustable=True,
        base_resists=(8, 8, 8, 4),
        base_attributes=(10, 5, 5),
        base_life=120,
        base_energy_shield=30,
    ),
    "helmet": SlotProfile(
        base_type="Iron Hat",
        adjustable=True,
        base_resists=(6, 6, 6, 3),
        base_attributes=(8, 3, 8),
        base_life=45,
        base_energy_shield=20,
    ),
    "gloves": SlotProfile(
        base_type="Iron Gauntlets",
        adjustable=True,
        base_resists=(4, 3, 4, 2),
        base_attributes=(5, 8, 3),
        base_life=35,
        base_energy_shield=15,
    ),
    "boots": SlotProfile(
        base_type="Rawhide Boots",
        adjustable=True,
        base_resists=(4, 5, 4, 2),
        base_attributes=(3, 10, 4),
        base_life=40,
        base_energy_shield=10,
    ),
}

DEFAULT_SLOT_PROFILE = SlotProfile(
    base_type="Gear",
    adjustable=True,
    base_resists=(3, 3, 3, 2),
    base_attributes=(5, 5, 5),
    base_life=30,
    base_energy_shield=10,
)


def build_item_templates(
    genome: GenomeV0,
    gem_plan: GemPlan,
    passive_plan: PassiveTreePlan,
    socket_plan: SocketPlan,
) -> ItemTemplatePlan:
    assignment_map: dict[str, int] = {}
    for assignment in socket_plan.assignments:
        assignment_map.setdefault(assignment.slot_id, 0)
        assignment_map[assignment.slot_id] += assignment.link_count

    templates: list[SlotTemplate] = []
    for slot in socket_plan.slots:
        profile = SLOT_PROFILES.get(slot.id, DEFAULT_SLOT_PROFILE)
        templates.append(
            _build_slot_template(
                slot_id=slot.id,
                profile=profile,
                genome=genome,
                gem_plan=gem_plan,
                passive_plan=passive_plan,
                socket_plan=socket_plan,
                link_bonus=assignment_map.get(slot.id, 0),
            )
        )

    report = repair_templates(templates, genome.defense_archetype)
    return ItemTemplatePlan(genome=genome, templates=tuple(templates), repair_report=report)


def export_slot_template_text(template: SlotTemplate) -> str:
    lines: list[str] = [f"Rare {template.base_type}", f"Slot: {template.slot_id}"]
    for attr in ATTRIBUTE_KEYS:
        value = getattr(template.contributions, attr)
        if value:
            lines.append(f"+{value} to {attr.capitalize()}")

    for resist in RESIST_KEYS:
        value = getattr(template.contributions, resist)
        if value:
            lines.append(f"+{value}% to {resist.capitalize()} Resistance")

    if template.contributions.life:
        lines.append(f"+{template.contributions.life} to Maximum Life")
    if template.contributions.energy_shield:
        lines.append(f"+{template.contributions.energy_shield} to Maximum Energy Shield")
    return "\n".join(lines)


def repair_templates(
    templates: Sequence[SlotTemplate], archetype: str, max_passes: int = MAX_REPAIR_PASSES
) -> RepairReport:
    mutable = list(templates)
    initial = compute_requirement_deficits(mutable)
    last_total = initial.total()
    iterations = 0
    for pass_index in range(1, max_passes + 1):
        deficits = compute_requirement_deficits(mutable)
        if deficits.total() == 0:
            break
        applied = _reduce_deficits_once(mutable, archetype, deficits)
        iterations = pass_index
        if applied == 0:
            break
        new_deficits = compute_requirement_deficits(mutable)
        if new_deficits.total() >= last_total:
            break
        last_total = new_deficits.total()
    remaining = compute_requirement_deficits(mutable)
    return RepairReport(
        iterations=iterations,
        initial_deficits=initial,
        remaining_deficits=remaining,
    )


def compute_requirement_deficits(templates: Iterable[SlotTemplate]) -> RequirementDeficits:
    totals: dict[str, int] = {key: 0 for key in ALL_STAT_KEYS}
    for template in templates:
        contribs = template.contributions
        totals["fire"] += contribs.fire
        totals["cold"] += contribs.cold
        totals["lightning"] += contribs.lightning
        totals["chaos"] += contribs.chaos
        totals["strength"] += contribs.strength
        totals["dexterity"] += contribs.dexterity
        totals["intelligence"] += contribs.intelligence
        totals["life"] += contribs.life
        totals["energy_shield"] += contribs.energy_shield
    return RequirementDeficits(
        fire=_clamp_zero(RESIST_BASELINE["fire"] - totals["fire"]),
        cold=_clamp_zero(RESIST_BASELINE["cold"] - totals["cold"]),
        lightning=_clamp_zero(RESIST_BASELINE["lightning"] - totals["lightning"]),
        chaos=_clamp_zero(RESIST_BASELINE["chaos"] - totals["chaos"]),
        strength=_clamp_zero(ATTRIBUTE_BASELINE["strength"] - totals["strength"]),
        dexterity=_clamp_zero(ATTRIBUTE_BASELINE["dexterity"] - totals["dexterity"]),
        intelligence=_clamp_zero(ATTRIBUTE_BASELINE["intelligence"] - totals["intelligence"]),
        life=_clamp_zero(LIFE_BASELINE - totals["life"]),
        energy_shield=_clamp_zero(ES_BASELINE - totals["energy_shield"]),
    )


def _reduce_deficits_once(
    templates: Sequence[SlotTemplate], archetype: str, deficits: RequirementDeficits
) -> int:
    applied = 0
    for resist in RESIST_KEYS:
        value = getattr(deficits, resist)
        applied += _allocate_to_adjustables(templates, resist, value)
    for attr in ATTRIBUTE_KEYS:
        value = getattr(deficits, attr)
        applied += _allocate_to_adjustables(templates, attr, value)
    life_metric = LIFE_METRIC_BY_ARCHETYPE.get(archetype, "life")
    if life_metric == "energy_shield":
        applied += _allocate_to_adjustables(templates, "energy_shield", deficits.energy_shield)
    else:
        applied += _allocate_to_adjustables(templates, "life", getattr(deficits, life_metric))
        if archetype == "hybrid":
            applied += _allocate_to_adjustables(templates, "energy_shield", deficits.energy_shield)
    if archetype == "energy_shield" and life_metric != "energy_shield":
        applied += _allocate_to_adjustables(templates, "energy_shield", deficits.energy_shield)
    return applied


def _allocate_to_adjustables(templates: Sequence[SlotTemplate], field: str, deficit: int) -> int:
    if deficit <= 0:
        return 0
    adjustable = [template for template in templates if template.adjustable]
    if not adjustable:
        return 0
    needed = deficit
    per_slot = max(1, (needed + len(adjustable) - 1) // len(adjustable))
    allocated = 0
    for slot in sorted(adjustable, key=lambda entry: entry.slot_id):
        chunk = min(per_slot, needed - allocated)
        slot.contributions.increment(field, chunk)
        allocated += chunk
        if allocated >= needed:
            break
    return allocated


def _clamp_zero(value: int) -> int:
    return value if value > 0 else 0


def _build_slot_template(
    slot_id: str,
    profile: SlotProfile,
    genome: GenomeV0,
    gem_plan: GemPlan,
    passive_plan: PassiveTreePlan,
    socket_plan: SocketPlan,
    link_bonus: int,
) -> SlotTemplate:
    passive_nodes = len(passive_plan.nodes)
    target_count = len(passive_plan.required_targets)
    gem_sum = sum(len(group.gems) for group in gem_plan.groups)
    seed_offset = _slot_seed_offset(slot_id, genome.seed)
    dominant_attr = _dominant_attribute_for_plan(gem_plan)
    skill_bonus_common = 1 + (passive_nodes // 10) + (target_count // 4) + (gem_sum // 24)
    attribute_bonus_common = 1 + (passive_nodes // 12) + (target_count // 5) + (gem_sum // 20)
    contributions = TemplateContributions(
        fire=_calc_resist(
            profile.base_resists[0],
            0,
            skill_bonus_common,
            link_bonus,
            seed_offset,
            slot_id == socket_plan.main_slot_id,
        ),
        cold=_calc_resist(
            profile.base_resists[1],
            1,
            skill_bonus_common,
            link_bonus,
            seed_offset,
            slot_id == socket_plan.main_slot_id,
        ),
        lightning=_calc_resist(
            profile.base_resists[2],
            2,
            skill_bonus_common,
            link_bonus,
            seed_offset,
            slot_id == socket_plan.main_slot_id,
        ),
        chaos=_calc_resist(
            profile.base_resists[3],
            3,
            skill_bonus_common,
            link_bonus,
            seed_offset,
            slot_id == socket_plan.main_slot_id,
        ),
        strength=_calc_attribute(
            profile.base_attributes[0],
            "strength",
            attribute_bonus_common,
            link_bonus,
            seed_offset,
            slot_id == socket_plan.main_slot_id,
            dominant_attr,
        ),
        dexterity=_calc_attribute(
            profile.base_attributes[1],
            "dexterity",
            attribute_bonus_common,
            link_bonus,
            seed_offset,
            slot_id == socket_plan.main_slot_id,
            dominant_attr,
        ),
        intelligence=_calc_attribute(
            profile.base_attributes[2],
            "intelligence",
            attribute_bonus_common,
            link_bonus,
            seed_offset,
            slot_id == socket_plan.main_slot_id,
            dominant_attr,
        ),
        life=_calc_life(
            profile.base_life,
            passive_nodes,
            target_count,
            link_bonus,
            seed_offset,
            slot_id == socket_plan.main_slot_id,
            genome.defense_archetype,
        ),
        energy_shield=_calc_energy_shield(
            profile.base_energy_shield,
            passive_nodes,
            link_bonus,
            seed_offset,
            slot_id == socket_plan.main_slot_id,
            genome.defense_archetype,
        ),
    )
    return SlotTemplate(
        slot_id=slot_id,
        base_type=profile.base_type,
        archetype_priorities=ARCHETYPE_PRIORITY_MAP.get(
            genome.defense_archetype, ARCHETYPE_PRIORITY_MAP["armour"]
        ),
        contributions=contributions,
        adjustable=profile.adjustable,
    )


def _calc_resist(
    base: int,
    index: int,
    common_bonus: int,
    link_bonus: int,
    seed_offset: int,
    is_main: bool,
) -> int:
    addition = common_bonus + (link_bonus // 4) + seed_offset
    addition += (1 if is_main else 0) + (index % 2)
    return base + addition


def _calc_attribute(
    base: int,
    attr_name: str,
    common_bonus: int,
    link_bonus: int,
    seed_offset: int,
    is_main: bool,
    dominant: str,
) -> int:
    addition = common_bonus + (link_bonus // 5) + seed_offset
    if attr_name == dominant:
        addition += 2
    if is_main:
        addition += 1
    return base + addition


def _calc_life(
    base: int,
    passive_nodes: int,
    target_count: int,
    link_bonus: int,
    seed_offset: int,
    is_main: bool,
    archetype: str,
) -> int:
    addition = (passive_nodes // 3) + (target_count * 2) + (link_bonus * 2) + seed_offset
    if archetype == "armour":
        addition += 3
    if is_main:
        addition += 5
    return base + addition


def _calc_energy_shield(
    base: int,
    passive_nodes: int,
    link_bonus: int,
    seed_offset: int,
    is_main: bool,
    archetype: str,
) -> int:
    addition = (passive_nodes // 4) + link_bonus + seed_offset
    if archetype == "energy_shield":
        addition += 4
    if is_main:
        addition += 2
    return base + addition


def _dominant_attribute_for_plan(plan: GemPlan) -> str:
    candidate = plan.full_dps_group_id.lower()
    group_names = " ".join(group.name for group in plan.groups).lower()
    for attr, keywords in _ATTRIBUTE_FOCUS_KEYWORDS.items():
        if any(keyword in candidate for keyword in keywords):
            return attr
        if any(keyword in group_names for keyword in keywords):
            return attr
    return "strength"


def _slot_seed_offset(slot_id: str, seed: int) -> int:
    return (sum(ord(ch) for ch in slot_id) + seed) % 3
