"""Skill package catalog helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from ..genome import DeterministicRng, GenomeV0

GemGroupType = Literal["damage", "utility"]
UtilityPackType = Literal["movement", "aura", "defense", "utility"]

DEFAULT_UTILITY_SELECTION_TYPES: tuple[UtilityPackType, ...] = ("movement", "aura")
DEFAULT_CATALOG_PATH = Path(__file__).resolve().parent / "packages_v0.json"


class SkillCatalogError(ValueError):
    """Base exception for catalog problems."""


class SkillCatalogValidationError(SkillCatalogError):
    """Raised when the catalog JSON is invalid."""


@dataclass(frozen=True)
class SupportGem:
    id: str
    name: str

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any], context: str) -> "SupportGem":
        return cls(
            id=_extract_string(raw, "id", context),
            name=_extract_string(raw, "name", context),
        )


@dataclass(frozen=True)
class GemGroup:
    id: str
    name: str
    group_type: GemGroupType
    gems: tuple[str, ...]

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any], context: str) -> "GemGroup":
        group_id = _extract_string(raw, "id", context)
        gems = _extract_string_list(raw, "gems", context)
        if not gems:
            raise SkillCatalogValidationError(f"{context}: gems list must not be empty")
        group_type = raw.get("type")
        if group_type not in ("damage", "utility"):
            raise SkillCatalogValidationError(
                f"{context}: invalid group type {group_type!r}, expected 'damage' or 'utility'"
            )
        return cls(
            id=group_id,
            name=_extract_string(raw, "name", context),
            group_type=group_type,
            gems=tuple(gems),
        )


def _extract_string(raw: Mapping[str, Any], key: str, context: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise SkillCatalogValidationError(f"{context}: missing or invalid field '{key}'")
    return value


def _extract_string_list(raw: Mapping[str, Any], key: str, context: str) -> tuple[str, ...]:
    value = raw.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise SkillCatalogValidationError(f"{context}: field '{key}' must be a list")
    return tuple(str(item) for item in value if item is not None)


@dataclass(frozen=True)
class SkillPackage:
    id: str
    name: str
    profile_ids: tuple[str, ...]
    defense_archetypes: tuple[str, ...]
    tags: tuple[str, ...]
    support_gems: tuple[SupportGem, ...]
    gem_groups: tuple[GemGroup, ...]

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> "SkillPackage":
        context = f"main_packages[{raw.get('id') or 'unknown'}]"
        groups_raw = raw.get("gem_groups")
        if not isinstance(groups_raw, list) or not groups_raw:
            raise SkillCatalogValidationError(f"{context}: gem_groups must be a non-empty list")
        gem_groups = tuple(GemGroup.from_raw(g, context) for g in groups_raw)
        return cls(
            id=_extract_string(raw, "id", context),
            name=_extract_string(raw, "name", context),
            profile_ids=_extract_string_list(raw, "profile_ids", context),
            defense_archetypes=_extract_string_list(raw, "defense_archetypes", context),
            tags=_extract_string_list(raw, "tags", context),
            support_gems=_parse_support_gems(raw, context),
            gem_groups=gem_groups,
        )


@dataclass(frozen=True)
class UtilityPack:
    id: str
    name: str
    type: UtilityPackType
    description: str
    profile_ids: tuple[str, ...]
    defense_archetypes: tuple[str, ...]
    tags: tuple[str, ...]
    support_gems: tuple[SupportGem, ...]
    gem_groups: tuple[GemGroup, ...]

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> "UtilityPack":
        context = f"utility_packs[{raw.get('id') or 'unknown'}]"
        gem_groups_raw = raw.get("gem_groups")
        if not isinstance(gem_groups_raw, list) or not gem_groups_raw:
            raise SkillCatalogValidationError(f"{context}: gem_groups must be a non-empty list")
        type_value = raw.get("type")
        if type_value not in ("movement", "aura", "defense", "utility"):
            raise SkillCatalogValidationError(
                f"{context}: invalid utility pack type {type_value!r}"
            )
        return cls(
            id=_extract_string(raw, "id", context),
            name=_extract_string(raw, "name", context),
            type=type_value,
            description=raw.get("description") or "",
            profile_ids=_extract_string_list(raw, "profile_ids", context),
            defense_archetypes=_extract_string_list(raw, "defense_archetypes", context),
            tags=_extract_string_list(raw, "tags", context),
            support_gems=_parse_support_gems(raw, context),
            gem_groups=tuple(GemGroup.from_raw(g, context) for g in gem_groups_raw),
        )


def _parse_support_gems(raw: Mapping[str, Any], context: str) -> tuple[SupportGem, ...]:
    gems_raw = raw.get("support_gems")
    if not isinstance(gems_raw, list):
        raise SkillCatalogValidationError(f"{context}: support_gems must be a list")
    return tuple(SupportGem.from_raw(g, context) for g in gems_raw)


@dataclass(frozen=True)
class GemPlan:
    groups: tuple[GemGroup, ...]
    full_dps_group_id: str


@dataclass(frozen=True)
class SkillCatalog:
    main_packages: Mapping[str, SkillPackage]
    utility_packs: tuple[UtilityPack, ...]
    _utility_by_type: Mapping[str, tuple[UtilityPack, ...]] = field(init=False, repr=False)

    def __post_init__(self):
        grouped: dict[str, list[UtilityPack]] = {}
        for pack in self.utility_packs:
            grouped.setdefault(pack.type, []).append(pack)
        object.__setattr__(self, "_utility_by_type", {k: tuple(v) for k, v in grouped.items()})

    @classmethod
    def load_from_path(cls, path: Path | str) -> "SkillCatalog":
        resolved = Path(path)
        raw = json.loads(resolved.read_text(encoding="utf-8"))
        main_entries = raw.get("main_packages")
        utility_entries = raw.get("utility_packs")
        if not isinstance(main_entries, list):
            raise SkillCatalogValidationError("main_packages must be a list")
        if not isinstance(utility_entries, list):
            raise SkillCatalogValidationError("utility_packs must be a list")
        main_packages = tuple(SkillPackage.from_raw(entry) for entry in main_entries)
        utility_packs = tuple(UtilityPack.from_raw(entry) for entry in utility_entries)
        _ensure_unique_ids((pkg.id for pkg in main_packages), "main_packages")
        _ensure_unique_ids((pack.id for pack in utility_packs), "utility_packs")
        catalog = cls({pkg.id: pkg for pkg in main_packages}, utility_packs)
        catalog.validate()
        return catalog

    def validate(self) -> None:
        if not self.main_packages:
            raise SkillCatalogValidationError("catalog must contain at least one main package")
        for pkg in self.main_packages.values():
            damage_groups = [g for g in pkg.gem_groups if g.group_type == "damage"]
            if not damage_groups:
                raise SkillCatalogValidationError(
                    f"package {pkg.id} must have at least one damage group"
                )
        missing_types = [
            utility
            for utility in DEFAULT_UTILITY_SELECTION_TYPES
            if utility not in self._utility_by_type
        ]
        if missing_types:
            raise SkillCatalogValidationError(
                f"catalog missing utility packs for types: {', '.join(missing_types)}"
            )

    def select_utility_packs(
        self,
        seed: int,
        utility_types: Sequence[UtilityPackType] | None = None,
    ) -> tuple[UtilityPack, ...]:
        rng = DeterministicRng(seed)
        picks: list[UtilityPack] = []
        for utility_type in utility_types or DEFAULT_UTILITY_SELECTION_TYPES:
            candidates = self._utility_by_type.get(utility_type)
            if not candidates:
                continue
            picks.append(rng.choice(candidates))
        return tuple(picks)

    def build_plan(self, genome: GenomeV0) -> GemPlan:
        package = self.main_packages.get(genome.main_skill_package)
        if package is None:
            raise SkillCatalogError(f"unknown package: {genome.main_skill_package}")
        damage_groups = [g for g in package.gem_groups if g.group_type == "damage"]
        if not damage_groups:
            raise SkillCatalogError(f"package {package.id} has no damage groups")
        utility_packs = self.select_utility_packs(genome.seed)
        utility_groups = tuple(g for pack in utility_packs for g in pack.gem_groups)
        groups = package.gem_groups + utility_groups
        return GemPlan(groups=groups, full_dps_group_id=damage_groups[0].id)


def _ensure_unique_ids(values: Iterable[str], context: str) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen:
            duplicates.append(value)
        else:
            seen.add(value)
    if duplicates:
        raise SkillCatalogValidationError(
            f"{context} contains duplicate ids: {', '.join(sorted(set(duplicates)))}"
        )


def load_default_skill_catalog() -> SkillCatalog:
    return SkillCatalog.load_from_path(DEFAULT_CATALOG_PATH)


__all__ = [
    "SupportGem",
    "GemGroup",
    "SkillPackage",
    "UtilityPack",
    "GemPlan",
    "SkillCatalog",
    "SkillCatalogError",
    "SkillCatalogValidationError",
    "load_default_skill_catalog",
]
