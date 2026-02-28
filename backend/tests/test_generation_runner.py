from xml.etree import ElementTree as ET

from backend.engine.generation.runner import (
    CANONICAL_ITEM_SLOTS,
    Candidate,
    POB_TARGET_VERSION,
    _optimizer_objectives_from_payload,
    _build_worker_xml_payload,
)
from backend.engine.genome import GenomeV0


def _dummy_candidate() -> Candidate:
    genome = GenomeV0(
        seed=42,
        class_name="Witch",
        ascendancy="Occultist",
        main_skill_package="arc",
        defense_archetype="armour",
        budget_tier="starter",
        profile_id="alpha",
    )
    build_details_payload = {
        "identity": {"class_name": "Witch", "ascendancy": "Occultist"},
        "gems": {
            "groups": [
                {
                    "id": "weapon_main",
                    "name": "Arc Chain",
                    "group_type": "damage",
                    "gems": ["arc", "controlled_destruction_support"],
                },
                {
                    "id": "body_util",
                    "name": "Body Guard",
                    "group_type": "utility",
                    "gems": ["molten_shell"],
                },
                {
                    "id": "helmet_util",
                    "name": "Helmet Guard",
                    "group_type": "utility",
                    "gems": ["arctic_armour"],
                },
                {
                    "id": "gloves_util",
                    "name": "Gloves Guard",
                    "group_type": "utility",
                    "gems": ["fortify_support"],
                },
                {
                    "id": "boots_util",
                    "name": "Boots Guard",
                    "group_type": "utility",
                    "gems": ["dash"],
                },
            ],
            "socket_plan": {
                "assignments": [
                    {"group_id": "weapon_main", "slot_id": "weapon_2h"},
                    {"group_id": "body_util", "slot_id": "body_armour"},
                    {"group_id": "helmet_util", "slot_id": "helmet"},
                    {"group_id": "gloves_util", "slot_id": "gloves"},
                    {"group_id": "boots_util", "slot_id": "boots"},
                ]
            },
        },
    }
    return Candidate(
        seed=genome.seed,
        build_id="candidate-42",
        main_skill_package=genome.main_skill_package,
        class_name=genome.class_name,
        ascendancy=genome.ascendancy,
        budget_tier=genome.budget_tier,
        failures=[],
        metrics_payload={
            "mapping_t16": {
                "metrics": {"full_dps": 1500.0, "max_hit": 750.0},
                "defense": {"armour": 900.0, "resists": {"fire": 70.0}},
                "resources": {"life": 4200.0, "mana": 1300.0},
                "reservation": {"reserved_percent": 30.0, "available_percent": 70.0},
                "attributes": {"strength": 150.0, "dexterity": 120.0, "intelligence": 80.0},
            }
        },
        genome=genome,
        code_payload="{}",
        build_details_payload=build_details_payload,
    )


def test_worker_xml_includes_target_version_attribute() -> None:
    candidate = _dummy_candidate()
    payload = _build_worker_xml_payload(candidate)
    root = ET.fromstring(payload)
    build_element = root.find("Build")
    assert build_element is not None
    assert build_element.attrib.get("targetVersion") == POB_TARGET_VERSION

    items = root.find("Items")
    assert items is not None
    item_set = items.find("ItemSet")
    assert item_set is not None
    slot_names = {
        slot.attrib.get("name")
        for slot in item_set.findall("Slot")
        if slot.attrib.get("name")
    }
    assert set(CANONICAL_ITEM_SLOTS).issubset(slot_names)
    assert len(items.findall("Item")) >= len(CANONICAL_ITEM_SLOTS)

    skill_set = root.find("Skills/SkillSet")
    assert skill_set is not None
    skills = skill_set.findall("Skill")
    assert skills
    first_skill = skills[0]
    assert first_skill.attrib.get("slot")
    first_gem = first_skill.find("Gem")
    assert first_gem is not None
    assert first_gem.attrib.get("nameSpec")




def _scenario_metrics_entry(full_dps: float, max_hit: float, cost: float, warnings: list[str] | None = None) -> dict[str, Any]:
    entry = {
        "metrics": {
            "full_dps": full_dps,
            "max_hit": max_hit,
            "total_cost_chaos": cost,
        }
    }
    if warnings is not None:
        entry["warnings"] = warnings
    return entry


def test_optimizer_objectives_skip_stub_warnings() -> None:
    payload = {
        "stub": _scenario_metrics_entry(1200.0, 500.0, 40.0, warnings=["generation_stub_metrics"]),
    }
    full_dps, max_hit, cost_sort, summary = _optimizer_objectives_from_payload(payload)
    assert full_dps == 0.0
    assert max_hit == float("inf")
    assert cost_sort == float("inf")
    assert summary["selection_basis"] == "stub_fallback"
    assert summary["full_dps"] is None
    assert summary["max_hit"] is None
    assert summary["cost"] is None


def test_optimizer_objectives_prefers_real_metrics_over_stub() -> None:
    payload = {
        "stub": _scenario_metrics_entry(250.0, 120.0, 30.0, warnings=["generation_stub_metrics"]),
        "real": _scenario_metrics_entry(2000.0, 1000.0, 80.0),
    }
    full_dps, max_hit, cost_sort, summary = _optimizer_objectives_from_payload(payload)
    assert full_dps == 2000.0
    assert max_hit == 1000.0
    assert cost_sort == 80.0
    assert summary["selection_basis"] == "objective"
    assert summary["full_dps"] == 2000.0
    assert summary["max_hit"] == 1000.0
    assert summary["cost"] == 80.0
