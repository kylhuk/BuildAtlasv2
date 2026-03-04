import pytest
from backend.engine.repair.operators import (
    ResistanceRepair,
    LifeRepair,
    AttributeRepair,
    ReservationRepair,
)


def test_resistance_repair():
    repair = ResistanceRepair()
    build = {
        "resistances": {"fire": 50, "cold": 75, "lightning": 75, "chaos": 45},
        "items": [
            {"slot_id": "helmet", "adjustable": True, "contributions": {"fire": 10}},
            {"slot_id": "gloves", "adjustable": False, "contributions": {"fire": 10}},
        ],
    }

    assert repair.needs_repair(build) is True
    repaired = repair.apply(build)
    # RESIST_BASELINE["fire"] is 60
    assert repaired["resistances"]["fire"] >= 60
    assert repaired["items"][0]["contributions"]["fire"] > 10
    assert repaired["items"][1]["contributions"]["fire"] == 10  # Not adjustable


def test_life_repair():
    repair = LifeRepair(ehp_threshold=3000)
    build = {
        "stats": {"ehp": 2000, "life": 800},
        "items": [
            {"slot_id": "helmet", "adjustable": True, "contributions": {"life": 40}},
        ],
    }

    assert repair.needs_repair(build) is True
    repaired = repair.apply(build)
    assert repaired["stats"]["ehp"] > 2000
    assert repaired["stats"]["life"] > 800
    assert repaired["items"][0]["contributions"]["life"] > 40


def test_attribute_repair():
    repair = AttributeRepair()
    build = {
        "attributes": {"strength": 100, "dexterity": 100, "intelligence": 100},
        "items": [
            {"slot_id": "weapon", "requirements": {"Strength": 150}, "adjustable": False},
            {"slot_id": "helmet", "adjustable": True, "contributions": {}},
        ],
    }

    assert repair.needs_repair(build) is True
    repaired = repair.apply(build)
    assert repaired["attributes"]["strength"] >= 150
    assert repaired["items"][1]["contributions"]["strength"] >= 50


def test_reservation_repair():
    repair = ReservationRepair()
    build = {
        "reservation": 1100,
        "total_mana": 1000,
        "gems": {
            "full_dps_group_id": "main_skill",
            "groups": [
                {"id": "main_skill", "name": "Cyclone"},
                {"id": "aura_1", "name": "Determination"},
            ],
        },
    }

    assert repair.needs_repair(build) is True
    repaired = repair.apply(build)
    assert repaired["reservation"] <= 1000
    assert len(repaired["gems"]["groups"]) == 1
    assert repaired["gems"]["groups"][0]["id"] == "main_skill"


def test_no_repair_needed():
    res_repair = ResistanceRepair()
    build = {"resistances": {"fire": 75, "cold": 75, "lightning": 75, "chaos": 75}, "items": []}
    assert res_repair.needs_repair(build) is False

    res_repair_apply = res_repair.apply(build)
    assert res_repair_apply == build
