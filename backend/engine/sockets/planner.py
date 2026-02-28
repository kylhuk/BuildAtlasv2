"""Deterministic socket planner for EP-V2-04."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, Tuple

from ..genome import GenomeV0
from ..skills import GemGroup, SkillCatalog, SkillCatalogError

ColorType = Literal["red", "green", "blue"]


@dataclass(frozen=True)
class SocketSlot:
    id: str
    capacity: int
    label: str


@dataclass(frozen=True)
class GemGroupAssignment:
    group_id: str
    slot_id: str
    link_count: int
    color: ColorType
    gem_count: int
    slot_capacity: int


@dataclass(frozen=True)
class PlanIssue:
    code: str
    message: str


@dataclass(frozen=True)
class PlanHint:
    code: str
    message: str


@dataclass(frozen=True)
class SocketPlan:
    genome_seed: int
    main_group_id: str
    main_slot_id: str
    main_link_requirement: int
    assignments: Tuple[GemGroupAssignment, ...]
    slots: Tuple[SocketSlot, ...]
    issues: Tuple[PlanIssue, ...]
    hints: Tuple[PlanHint, ...]


DEFAULT_SLOT_DEFINITIONS: Tuple[SocketSlot, ...] = (
    SocketSlot(id="weapon_2h", capacity=6, label="Two-Hand Weapon"),
    SocketSlot(id="body_armour", capacity=6, label="Body Armour"),
    SocketSlot(id="helmet", capacity=4, label="Helmet"),
    SocketSlot(id="gloves", capacity=4, label="Gloves"),
    SocketSlot(id="boots", capacity=4, label="Boots"),
)

_LINK_REQUIREMENTS_BY_BUDGET = {
    "starter": 4,
    "midgame": 5,
    "endgame": 6,
}

_COLOR_KEYWORDS: dict[ColorType, Tuple[str, ...]] = {
    "red": (
        "melee",
        "physical",
        "strike",
        "smash",
        "brutal",
        "blade",
        "fortify",
    ),
    "green": (
        "ranged",
        "projectile",
        "attack",
        "poison",
        "dash",
        "movement",
        "chain",
    ),
    "blue": (
        "caster",
        "spell",
        "lightning",
        "cold",
        "elemental",
        "aura",
        "shield",
    ),
}

DEFAULT_UTILITY_LINKS = 3


def determine_main_link_requirement(budget_tier: str) -> int:
    """Map a genome budget tier to the planned number of main links."""

    return _LINK_REQUIREMENTS_BY_BUDGET.get(budget_tier, 4)


def _select_main_slot_id(slots: Sequence[SocketSlot], prefer_two_handed: bool) -> str:
    ids = {slot.id for slot in slots}
    if prefer_two_handed and "weapon_2h" in ids:
        return "weapon_2h"
    if "body_armour" in ids:
        return "body_armour"
    if slots:
        return slots[0].id
    raise SkillCatalogError("no socket slots are available for planning")


def _determine_group_color(group: GemGroup, tags: Sequence[str]) -> ColorType:
    tokens = [group.name, group.id, *group.gems, *tags]
    normalized = [token.lower() for token in tokens if token]
    scores: dict[ColorType, int] = {"red": 0, "green": 0, "blue": 0}
    for token in normalized:
        for color, keywords in _COLOR_KEYWORDS.items():
            if any(keyword in token for keyword in keywords):
                scores[color] += 1
    if any(score > 0 for score in scores.values()):
        highest = max(scores.values())
        for color in ("red", "green", "blue"):
            if scores[color] == highest:
                return color
    checksum = sum(ord(ch) for token in normalized for ch in token)
    fallback = ("red", "green", "blue")[checksum % 3]
    return fallback


def plan_sockets(
    catalog: SkillCatalog,
    genome: GenomeV0,
    prefer_two_handed: bool = True,
    slot_definitions: Sequence[SocketSlot] | None = None,
) -> SocketPlan:
    """Build a deterministic socket plan aligned with the given genome."""

    slots = tuple(slot_definitions or DEFAULT_SLOT_DEFINITIONS)
    slot_map = {slot.id: slot for slot in slots}
    plan = catalog.build_plan(genome)
    package = catalog.main_packages.get(genome.main_skill_package)
    if package is None:
        raise SkillCatalogError(f"unknown package: {genome.main_skill_package}")
    utility_packs = catalog.select_utility_packs(genome.seed)
    group_by_id = {group.id: group for group in plan.groups}
    tag_lookup: dict[str, Sequence[str]] = {}
    for group in package.gem_groups:
        tag_lookup[group.id] = tuple(package.tags)
    for pack in utility_packs:
        for group in pack.gem_groups:
            tag_lookup[group.id] = tuple(pack.tags)
    colors: dict[str, ColorType] = {
        group_id: _determine_group_color(group_by_id[group_id], tag_lookup.get(group_id, ()))
        for group_id in group_by_id
    }
    main_slot_id = _select_main_slot_id(slots, prefer_two_handed)
    main_slot = slot_map.get(main_slot_id)
    if main_slot is None:
        raise SkillCatalogError(f"selected main slot {main_slot_id} not defined")
    main_link_requirement = determine_main_link_requirement(genome.budget_tier)
    main_group_id = plan.full_dps_group_id
    main_group = group_by_id[main_group_id]
    main_assignment = GemGroupAssignment(
        group_id=main_group_id,
        slot_id=main_slot_id,
        link_count=min(main_slot.capacity, main_link_requirement),
        color=colors[main_group_id],
        gem_count=len(main_group.gems),
        slot_capacity=main_slot.capacity,
    )
    aux_slots = [slot for slot in slots if slot.id != main_slot_id]
    utility_groups = [group for group in plan.groups if group.id != main_group_id]
    assignments: list[GemGroupAssignment] = [main_assignment]
    reuses = 0
    for index, group in enumerate(utility_groups):
        if aux_slots:
            slot = aux_slots[index] if index < len(aux_slots) else aux_slots[-1]
            if index >= len(aux_slots):
                reuses += 1
        else:
            slot = main_slot
        assignments.append(
            GemGroupAssignment(
                group_id=group.id,
                slot_id=slot.id,
                link_count=min(slot.capacity, DEFAULT_UTILITY_LINKS),
                color=colors[group.id],
                gem_count=len(group.gems),
                slot_capacity=slot.capacity,
            )
        )
    issues, hints = _evaluate_feasibility(
        slots=slots,
        main_slot=main_slot,
        main_link_requirement=main_link_requirement,
        utility_groups=utility_groups,
        auxiliary_slots=aux_slots,
        utility_reuses=reuses,
    )
    return SocketPlan(
        genome_seed=genome.seed,
        main_group_id=main_group_id,
        main_slot_id=main_slot_id,
        main_link_requirement=main_link_requirement,
        assignments=tuple(assignments),
        slots=slots,
        issues=issues,
        hints=hints,
    )


def _evaluate_feasibility(
    *,
    slots: Sequence[SocketSlot],
    main_slot: SocketSlot,
    main_link_requirement: int,
    utility_groups: Sequence[GemGroup],
    auxiliary_slots: Sequence[SocketSlot],
    utility_reuses: int,
) -> Tuple[Tuple[PlanIssue, ...], Tuple[PlanHint, ...]]:
    issues: list[PlanIssue] = []
    hints: list[PlanHint] = []
    seen_issues: set[str] = set()
    seen_hints: set[str] = set()

    def append_issue(code: str, message: str) -> None:
        if code not in seen_issues:
            seen_issues.add(code)
            issues.append(PlanIssue(code=code, message=message))

    def append_hint(code: str, message: str) -> None:
        if code not in seen_hints:
            seen_hints.add(code)
            hints.append(PlanHint(code=code, message=message))

    if main_slot.capacity < main_link_requirement:
        append_issue(
            "main_slot_capacity",
            (
                f"main slot {main_slot.id} has {main_slot.capacity} sockets "
                f"but requires {main_link_requirement}"
            ),
        )
        append_hint(
            "main_slot_capacity",
            "equipping a larger socketed item or lowering link ambitions reduces pressure",
        )

    total_capacity = sum(slot.capacity for slot in slots)
    estimated_links = main_link_requirement + len(utility_groups) * DEFAULT_UTILITY_LINKS
    if total_capacity < estimated_links:
        append_issue(
            "total_capacity",
            (
                f"available socket capacity ({total_capacity}) is less than "
                f"estimated demand ({estimated_links})"
            ),
        )
        append_hint(
            "total_capacity",
            "drop one utility group or swap to a higher-capacity cluster to stay feasible",
        )

    if len(utility_groups) > len(auxiliary_slots):
        append_issue(
            "utility_slots",
            "not enough auxiliary slots for all utility groups; some slots are reused",
        )
        append_hint(
            "utility_slots",
            "prioritize your strongest utility groups when auxiliary slots are constrained",
        )

    if utility_reuses > 0:
        append_issue(
            "slot_reuse",
            "utility groups share slots due to limited auxiliary gear",
        )

    return tuple(issues), tuple(hints)
