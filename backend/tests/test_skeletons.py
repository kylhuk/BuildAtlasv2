import pytest
from backend.engine.skeletons.schema import Skeleton


def test_skeleton_validation_success():
    skeleton = Skeleton(
        skeleton_id="test_id",
        class_name="marauder",
        ascendancy="Juggernaut",
        main_skill="Cyclone",
        skill_links=["Cyclone", "Melee Physical Damage"],
        aura_package=["Determination"],
        defense_layer="armour",
        budget_tier="starter",
        target_gates={"dps": 1000000},
        required_uniques=[],
        tree_path="some/path",
    )
    skeleton.validate()


def test_skeleton_validation_invalid_class():
    skeleton = Skeleton(
        skeleton_id="test_id",
        class_name="invalid_class",
        ascendancy="Juggernaut",
        main_skill="Cyclone",
        skill_links=[],
        aura_package=[],
        defense_layer="armour",
        budget_tier="starter",
        target_gates={},
        required_uniques=[],
        tree_path="some/path",
    )
    with pytest.raises(ValueError, match="invalid class_name"):
        skeleton.validate()


def test_skeleton_validation_invalid_ascendancy():
    skeleton = Skeleton(
        skeleton_id="test_id",
        class_name="marauder",
        ascendancy="Elementalist",
        main_skill="Cyclone",
        skill_links=[],
        aura_package=[],
        defense_layer="armour",
        budget_tier="starter",
        target_gates={},
        required_uniques=[],
        tree_path="some/path",
    )
    with pytest.raises(ValueError, match="invalid ascendancy"):
        skeleton.validate()


def test_skeleton_from_to_dict():
    data = {
        "skeleton_id": "test_id",
        "class_name": "marauder",
        "ascendancy": "Juggernaut",
        "main_skill": "Cyclone",
        "skill_links": ["Cyclone"],
        "aura_package": ["Determination"],
        "defense_layer": "armour",
        "budget_tier": "starter",
        "target_gates": {"dps": 1000000},
        "required_uniques": ["Unique Item"],
        "tree_path": "some/path",
    }
    skeleton = Skeleton.from_dict(data)
    assert skeleton.skeleton_id == "test_id"
    assert skeleton.to_dict() == data
