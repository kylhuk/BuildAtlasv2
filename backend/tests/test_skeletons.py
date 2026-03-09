import json

import pytest

from backend.app.settings import settings
from backend.engine.skeletons.loader import list_skeletons
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


def test_list_skeletons(tmp_path, monkeypatch):

    # Create a temporary skeletons directory
    skeletons_dir = tmp_path / "skeletons"
    skeletons_dir.mkdir()

    # Create a dummy skeleton file
    skeleton_data = {
        "skeleton_id": "test_id",
        "class_name": "marauder",
        "ascendancy": "Juggernaut",
        "main_skill": "Cyclone",
        "skill_links": ["Cyclone"],
        "aura_package": ["Determination"],
        "defense_layer": "armour",
        "budget_tier": "starter",
        "target_gates": {"dps": 1000000},
        "required_uniques": [],
        "tree_path": "some/path",
    }
    with open(skeletons_dir / "test_id.json", "w") as f:
        json.dump(skeleton_data, f)

    # Mock settings.data_path
    monkeypatch.setattr(settings, "data_path", tmp_path)

    skeletons = list_skeletons()
    assert len(skeletons) == 1
    assert skeletons[0].skeleton_id == "test_id"
