import json

from backend.app.settings import settings
from backend.engine.skeletons.expansion import expand_skeleton


def test_expand_skeleton(tmp_path, monkeypatch):
    # Create a temporary skeletons directory
    skeletons_dir = tmp_path / "skeletons"
    skeletons_dir.mkdir()

    # Create a dummy skeleton file
    skeleton_id = "test_skeleton"
    skeleton_data = {
        "skeleton_id": skeleton_id,
        "class_name": "marauder",
        "ascendancy": "Juggernaut",
        "main_skill": "Cyclone",
        "skill_links": ["Melee Physical Damage", "Brutality"],
        "aura_package": ["Determination", "Pride"],
        "defense_layer": "armour",
        "budget_tier": "starter",
        "target_gates": {"dps": 1000000},
        "required_uniques": ["Brass Dome"],
        "tree_path": "1,2,3,4,5",
    }
    with open(skeletons_dir / f"{skeleton_id}.json", "w") as f:
        json.dump(skeleton_data, f)

    # Mock settings.data_path
    monkeypatch.setattr(settings, "data_path", tmp_path)

    # Expand the skeleton
    dna = expand_skeleton(skeleton_id)

    # Verify DNA
    assert dna.skeleton_id == skeleton_id
    assert dna.class_name == "marauder"
    assert dna.ascendancy == "Juggernaut"
    assert dna.main_skill == "Cyclone"

    # Verify tree nodes
    assert dna.tree_nodes == [1, 2, 3, 4, 5]

    # Verify items
    assert len(dna.items) == 1
    assert "Brass Dome" in dna.items[0]
    assert "Rarity: UNIQUE" in dna.items[0]

    # Verify gem groups
    assert len(dna.gem_groups) == 2

    # Main skill group
    main_group = next(g for g in dna.gem_groups if g["slot"] == "Body Armour")
    assert len(main_group["gems"]) == 3
    assert main_group["gems"][0]["name"] == "Cyclone"
    assert main_group["gems"][1]["name"] == "Melee Physical Damage"
    assert main_group["gems"][2]["name"] == "Brutality"

    # Aura group
    aura_group = next(g for g in dna.gem_groups if g["slot"] == "Weapon 1")
    assert len(aura_group["gems"]) == 2
    assert aura_group["gems"][0]["name"] == "Determination"
    assert aura_group["gems"][1]["name"] == "Pride"

    # Verify metadata
    assert dna.defense_layer == "armour"
    assert dna.budget_tier == "starter"
    assert dna.target_gates == {"dps": 1000000}
