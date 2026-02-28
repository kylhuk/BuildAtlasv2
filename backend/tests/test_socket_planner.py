from backend.engine import load_default_skill_catalog
from backend.engine.genome import GenomeV0, deterministic_genome_from_seed
from backend.engine.sockets import (
    SocketSlot,
    determine_main_link_requirement,
    plan_sockets,
)


def test_deterministic_plan_matches():
    catalog = load_default_skill_catalog()
    genome = deterministic_genome_from_seed(2024)
    first_plan = plan_sockets(catalog, genome)
    second_plan = plan_sockets(catalog, genome)
    assert first_plan == second_plan


def test_link_requirements_by_budget():
    assert determine_main_link_requirement("starter") == 4
    assert determine_main_link_requirement("midgame") == 5
    assert determine_main_link_requirement("endgame") == 6


def test_main_slot_selection_respects_preference():
    catalog = load_default_skill_catalog()
    genome = deterministic_genome_from_seed(42)

    plan_two_handed = plan_sockets(catalog, genome, prefer_two_handed=True)
    main_two_handed = next(
        assignment
        for assignment in plan_two_handed.assignments
        if assignment.group_id == plan_two_handed.main_group_id
    )
    assert main_two_handed.slot_id == "weapon_2h"

    plan_chest = plan_sockets(catalog, genome, prefer_two_handed=False)
    main_chest = next(
        assignment
        for assignment in plan_chest.assignments
        if assignment.group_id == plan_chest.main_group_id
    )
    assert main_chest.slot_id == "body_armour"


def test_feasibility_issues_for_low_capacity():
    catalog = load_default_skill_catalog()
    base_genome = deterministic_genome_from_seed(7)
    genome = GenomeV0(
        seed=base_genome.seed,
        class_name=base_genome.class_name,
        ascendancy=base_genome.ascendancy,
        main_skill_package=base_genome.main_skill_package,
        defense_archetype=base_genome.defense_archetype,
        budget_tier="endgame",
        profile_id=base_genome.profile_id,
    )
    tiny_slots = (
        SocketSlot(id="body_armour", capacity=3, label="Tiny Armour"),
        SocketSlot(id="helmet", capacity=2, label="Tiny Head"),
    )
    plan = plan_sockets(catalog, genome, prefer_two_handed=False, slot_definitions=tiny_slots)

    assert plan.issues, "expected feasibility issues when capacity is low"
    assert plan.hints, "expected recovery hints"
    assert any(issue.code == "main_slot_capacity" for issue in plan.issues)


def test_assignments_never_overflow_slot_capacity():
    catalog = load_default_skill_catalog()
    genome = deterministic_genome_from_seed(99)
    plan = plan_sockets(catalog, genome)

    for assignment in plan.assignments:
        assert assignment.link_count <= assignment.slot_capacity
