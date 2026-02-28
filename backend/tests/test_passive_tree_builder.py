import pytest

from backend.engine.genome import CLASS_ASCENDANCIES, GenomeV0
from backend.engine.passives import (
    PassiveTreeGraph,
    PassiveTreeGraphValidationError,
    PassiveTreePlanError,
    build_passive_tree_plan,
    export_passive_tree_plan,
    load_default_passive_tree_graph,
)


def _make_genome(class_name: str, defense_archetype: str) -> GenomeV0:
    ascendancy = CLASS_ASCENDANCIES[class_name][0]
    return GenomeV0(
        seed=1,
        class_name=class_name,
        ascendancy=ascendancy,
        main_skill_package="cyclone",
        defense_archetype=defense_archetype,
        budget_tier="starter",
        profile_id="alpha",
    )


def test_load_default_graph_has_start_nodes_for_every_class() -> None:
    graph = load_default_passive_tree_graph()
    assert set(graph.start_nodes) == set(CLASS_ASCENDANCIES)
    for start_node in graph.start_nodes.values():
        assert start_node in graph.nodes


def test_build_plan_includes_required_target() -> None:
    genome = _make_genome("marauder", "armour")
    plan = build_passive_tree_plan(genome, point_budget=10)
    assert "node_armour_required" in [node.id for node in plan.nodes]


def test_export_plan_node_ids_are_sorted_and_prefer_pob_ids() -> None:
    genome = _make_genome("marauder", "armour")
    plan = build_passive_tree_plan(genome, point_budget=10)
    exported = export_passive_tree_plan(plan)
    assert exported["node_ids"] == ["A1", "ARM_REQ", "node_central"]


def test_point_budget_failure_when_budget_too_small() -> None:
    genome = _make_genome("marauder", "armour")
    with pytest.raises(PassiveTreePlanError):
        build_passive_tree_plan(genome, point_budget=2)


def test_graph_loader_detects_duplicate_nodes() -> None:
    payload = {
        "schema_version": "v0",
        "nodes": [
            {"id": "duplicate", "kind": "start"},
            {"id": "duplicate", "kind": "start"},
        ],
        "edges": [],
        "start_nodes": {class_name: "duplicate" for class_name in CLASS_ASCENDANCIES},
    }
    with pytest.raises(PassiveTreeGraphValidationError):
        PassiveTreeGraph.from_dict(payload)
