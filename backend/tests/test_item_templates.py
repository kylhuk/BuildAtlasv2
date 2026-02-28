"""Tests for item template builder helpers."""

from backend.engine import (
    ARCHETYPE_PRIORITY_MAP,
    MAX_REPAIR_PASSES,
    build_item_templates,
    export_slot_template_text,
    load_default_skill_catalog,
)
from backend.engine.genome import deterministic_genome_from_seed
from backend.engine.items import RequirementDeficits
from backend.engine.passives.builder import build_passive_tree_plan
from backend.engine.sockets import plan_sockets


def _build_dependencies(seed: int):
    catalog = load_default_skill_catalog()
    genome = deterministic_genome_from_seed(seed)
    gem_plan = catalog.build_plan(genome)
    passive_plan = build_passive_tree_plan(genome, point_budget=30)
    socket_plan = plan_sockets(catalog, genome)
    return genome, gem_plan, passive_plan, socket_plan


def test_item_templates_are_deterministic():
    genome, gem_plan, passive_plan, socket_plan = _build_dependencies(2026)
    first = build_item_templates(genome, gem_plan, passive_plan, socket_plan)
    second = build_item_templates(genome, gem_plan, passive_plan, socket_plan)
    assert first == second


def test_requirement_satisfier_non_worsening_deficits():
    genome, gem_plan, passive_plan, socket_plan = _build_dependencies(2027)
    result = build_item_templates(genome, gem_plan, passive_plan, socket_plan)
    initial = result.repair_report.initial_deficits
    remaining = result.repair_report.remaining_deficits
    for field in RequirementDeficits.__dataclass_fields__:
        assert getattr(remaining, field) <= getattr(initial, field)


def test_archetype_priorities_reflect_genome():
    genome, gem_plan, passive_plan, socket_plan = _build_dependencies(2028)
    result = build_item_templates(genome, gem_plan, passive_plan, socket_plan)
    expected = ARCHETYPE_PRIORITY_MAP[genome.defense_archetype]
    assert all(template.archetype_priorities == expected for template in result.templates)


def test_pob_text_export_is_stable():
    genome, gem_plan, passive_plan, socket_plan = _build_dependencies(2029)
    result = build_item_templates(genome, gem_plan, passive_plan, socket_plan)
    text = export_slot_template_text(result.templates[0])
    assert text == export_slot_template_text(result.templates[0])
    assert text.startswith("Rare")
    assert "Slot:" in text


def test_repair_report_iterations_and_deficits():
    genome, gem_plan, passive_plan, socket_plan = _build_dependencies(2030)
    result = build_item_templates(genome, gem_plan, passive_plan, socket_plan)
    report = result.repair_report
    assert report.remaining_deficits.total() <= report.initial_deficits.total()
    assert 0 <= report.iterations <= MAX_REPAIR_PASSES
