"""Tests for the skill catalog subsystem."""

from pathlib import Path

import pytest

from backend.engine import (
    SkillCatalog,
    SkillCatalogValidationError,
    load_default_skill_catalog,
)
from backend.engine.genome import deterministic_genome_from_seed


def test_catalog_loads_and_has_required_packs() -> None:
    catalog = load_default_skill_catalog()
    assert len(catalog.main_packages) >= 10
    assert "movement" in {pack.type for pack in catalog.utility_packs}
    assert "aura" in {pack.type for pack in catalog.utility_packs}


def test_gem_plan_is_deterministic_for_same_seed() -> None:
    genome = deterministic_genome_from_seed(1_234)
    catalog = load_default_skill_catalog()
    plan_a = catalog.build_plan(genome)
    plan_b = catalog.build_plan(genome)
    assert plan_a == plan_b


def test_full_dps_group_is_main_damage_and_not_utility() -> None:
    genome = deterministic_genome_from_seed(9_876)
    catalog = load_default_skill_catalog()
    plan = catalog.build_plan(genome)
    main_package = catalog.main_packages[genome.main_skill_package]
    damage_group_ids = [
        group.id for group in main_package.gem_groups if group.group_type == "damage"
    ]
    assert plan.full_dps_group_id in damage_group_ids
    main_group_count = len(main_package.gem_groups)
    utility_group_ids = {group.id for group in plan.groups[main_group_count:]}
    assert plan.full_dps_group_id not in utility_group_ids


def test_invalid_catalog_fixture_fails_validation() -> None:
    fixture = Path(__file__).resolve().parent / "fixtures" / "invalid_skill_catalog.json"
    with pytest.raises(SkillCatalogValidationError):
        SkillCatalog.load_from_path(fixture)
