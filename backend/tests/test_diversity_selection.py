"""Tests for diversity/MAP-Elites selection."""

from __future__ import annotations

import pytest

from backend.engine.generation.diversity import (
    NicheAssignment,
    assign_niche,
    select_diverse_elites,
    DiversityArchive,
)


def test_assign_niche_returns_correct_structure():
    """assign_niche should return NicheAssignment with all fields."""
    build = {
        "class": "Witch",
        "damage_type": "chaos",
        "defense_archetype": "energy_shield",
        "genome": {"main_skill_package": "contagion"},
    }

    result = assign_niche(build)

    assert isinstance(result, NicheAssignment)
    assert result.class_name == "Witch"
    assert result.damage_type == "chaos"
    assert result.defense_type == "energy_shield"
    assert result.skill_type == "spell"


def test_assign_niche_with_spell_skill():
    """assign_niche should correctly identify spell skills."""
    build = {
        "class": "Templar",
        "genome": {"main_skill_package": "安卡"},
        "defense_archetype": "armor",
    }

    result = assign_niche(build)

    assert result.class_name == "Templar"
    assert result.skill_type == "spell"
    assert result.defense_type == "armor"


def test_assign_niche_with_attack_skill():
    """assign_niche should correctly identify attack skills."""
    build = {
        "class": "Ranger",
        "genome": {"main_skill_package": "lightning_arrow"},
        "defense_archetype": "evasion",
    }

    result = assign_niche(build)

    assert result.class_name == "Ranger"
    assert result.skill_type == "spell"
    assert result.defense_type == "evasion"


def test_assign_niche_with_minion_skill():
    """assign_niche should correctly identify minion skills."""
    build = {
        "class": "Necromancer",
        "genome": {"main_skill_package": "summon_skeleton"},
        "defense_archetype": "hybrid",
    }

    result = assign_niche(build)

    assert result.skill_type == "minion"


def test_diverse_elites_covers_multiple_niches():
    """select_diverse_elites should select from multiple niches."""
    candidates = [
        {"class": "Witch", "genome": {"main_skill_package": "arc"}, "defense_archetype": "armor", "score": 1000.0},
        {"class": "Ranger", "genome": {"main_skill_package": "lightning_arrow"}, "defense_archetype": "evasion", "score": 900.0},
        {"class": "Marauder", "genome": {"main_skill_package": "earthquake"}, "defense_archetype": "armor", "score": 800.0},
        {"class": "Shadow", "genome": {"main_skill_package": "blade_vortex"}, "defense_archetype": "evasion", "score": 700.0},
        {"class": "Templar", "genome": {"main_skill_package": "ball_lightning"}, "defense_archetype": "armor", "score": 600.0},
    ]

    result = select_diverse_elites(candidates, elite_count=5)

    assert len(result) <= 5

    niches = set()
    for candidate, _ in result:
        niche = assign_niche(candidate)
        niches.add(niche)

    assert len(niches) >= 3, f"Expected at least 3 different niches, got {len(niches)}"


def test_diverse_elites_with_empty_list():
    """select_diverse_elites should handle empty list."""
    result = select_diverse_elites([], elite_count=5)
    assert result == []


def test_diverse_elites_with_archive():
    """select_diverse_elites should use existing archive for empty niches."""
    candidates = [
        {"class": "Witch", "genome": {"main_skill_package": "arc"}, "defense_archetype": "armor", "score": 1000.0},
    ]

    archive = {
        NicheAssignment("Ranger", "elemental", "evasion", "attack"): {
            "class": "Ranger", "genome": {"main_skill_package": "lightning_arrow"}, "score": 800.0
        },
    }

    result = select_diverse_elites(candidates, elite_count=3, archive=archive)

    assert len(result) <= 3


def test_archive_insert_replaces_if_better():
    """Archive should replace entry if new build is better."""
    archive = DiversityArchive()

    build1 = {
        "class": "Witch",
        "genome": {"main_skill_package": "arc"},
        "defense_archetype": "armor",
        "full_dps": 1000.0,
    }
    result1 = archive.update(build1)
    assert result1 is True
    assert len(archive) == 1

    build2 = {
        "class": "Witch",
        "genome": {"main_skill_package": "arc"},
        "defense_archetype": "armor",
        "full_dps": 2000.0,
    }
    result2 = archive.update(build2)
    assert result2 is True
    assert len(archive) == 1

    niche = NicheAssignment("Witch", "hybrid", "armor", "spell")
    best = archive.get_best_for_niche(niche)
    assert best is not None
    assert best["full_dps"] == 2000.0


def test_archive_insert_keeps_if_worse():
    """Archive should keep existing if new build is worse."""
    archive = DiversityArchive()

    build1 = {
        "class": "Witch",
        "genome": {"main_skill_package": "arc"},
        "defense_archetype": "armor",
        "full_dps": 2000.0,
    }
    archive.update(build1)

    build2 = {
        "class": "Witch",
        "genome": {"main_skill_package": "arc"},
        "defense_archetype": "armor",
        "full_dps": 1000.0,
    }
    result = archive.update(build2)
    assert result is False
    assert len(archive) == 1

    niche = NicheAssignment("Witch", "hybrid", "armor", "spell")
    best = archive.get_best_for_niche(niche)
    assert best is not None
    assert best["full_dps"] == 2000.0


def test_archive_insert_different_niches():
    """Archive should keep entries from different niches."""
    archive = DiversityArchive()

    build1 = {
        "class": "Witch",
        "genome": {"main_skill_package": "arc"},
        "defense_archetype": "armor",
        "full_dps": 1000.0,
    }
    archive.update(build1)

    build2 = {
        "class": "Ranger",
        "genome": {"main_skill_package": "lightning_arrow"},
        "defense_archetype": "evasion",
        "full_dps": 1000.0,
    }
    archive.update(build2)

    assert len(archive) == 2
    assert archive.get_niche_count() == 2


def test_archive_persists_to_disk(tmp_path):
    """Archive should save to disk correctly."""
    archive = DiversityArchive()

    build1 = {
        "class": "Witch",
        "genome": {"main_skill_package": "arc"},
        "defense_archetype": "armor",
        "full_dps": 1000.0,
    }
    build2 = {
        "class": "Ranger",
        "genome": {"main_skill_package": "lightning_arrow"},
        "defense_archetype": "evasion",
        "full_dps": 900.0,
    }
    archive.update(build1)
    archive.update(build2)

    archive_path = tmp_path / "diversity_archive.json"
    archive.save(archive_path)

    assert archive_path.exists()

    new_archive = DiversityArchive()
    new_archive.load(archive_path)

    assert len(new_archive) == 2
    assert new_archive.get_niche_count() == 2


def test_archive_persists_empty_archive(tmp_path):
    """Archive should handle saving empty archive."""
    archive = DiversityArchive()

    archive_path = tmp_path / "empty_archive.json"
    archive.save(archive_path)

    assert archive_path.exists()


def test_archive_load_nonexistent_file():
    """Archive should handle loading nonexistent file gracefully."""
    from pathlib import Path
    archive = DiversityArchive()
    archive.load(Path("/nonexistent/path.json"))
    assert len(archive) == 0


def test_get_diverse_sample():
    """Archive should return diverse sample of builds."""
    archive = DiversityArchive()

    build1 = {"class": "Witch", "genome": {"main_skill_package": "arc"}, "defense_archetype": "armor", "full_dps": 1000.0}
    build2 = {"class": "Ranger", "genome": {"main_skill_package": "lightning_arrow"}, "defense_archetype": "evasion", "full_dps": 900.0}
    build3 = {"class": "Marauder", "genome": {"main_skill_package": "earthquake"}, "defense_archetype": "armor", "full_dps": 800.0}

    archive.update(build1)
    archive.update(build2)
    archive.update(build3)

    sample = archive.get_diverse_sample(2)

    assert len(sample) == 2


def test_niche_assignment_to_dict():
    """NicheAssignment should serialize to dict correctly."""
    niche = NicheAssignment(
        class_name="Witch",
        damage_type="chaos",
        defense_type="energy_shield",
        skill_type="spell",
    )

    d = niche.to_dict()

    assert d["class"] == "Witch"
    assert d["damage_type"] == "chaos"
    assert d["defense_type"] == "energy_shield"
    assert d["skill_type"] == "spell"


def test_niche_assignment_from_dict():
    """NicheAssignment should deserialize from dict correctly."""
    data = {
        "class": "Ranger",
        "damage_type": "physical",
        "defense_type": "evasion",
        "skill_type": "attack",
    }

    niche = NicheAssignment.from_dict(data)

    assert niche.class_name == "Ranger"
    assert niche.damage_type == "physical"
    assert niche.defense_type == "evasion"
    assert niche.skill_type == "attack"


def test_niche_assignment_str():
    """NicheAssignment should have readable string representation."""
    niche = NicheAssignment(
        class_name="Witch",
        damage_type="chaos",
        defense_type="energy_shield",
        skill_type="spell",
    )

    assert str(niche) == "Witch/chaos/energy_shield/spell"


def test_archive_get_niches():
    """Archive should return list of filled niches."""
    archive = DiversityArchive()

    build1 = {"class": "Witch", "genome": {"main_skill_package": "arc"}, "defense_archetype": "armor", "full_dps": 1000.0}
    build2 = {"class": "Ranger", "genome": {"main_skill_package": "lightning_arrow"}, "defense_archetype": "evasion", "full_dps": 900.0}

    archive.update(build1)
    archive.update(build2)

    niches = archive.get_niches()

    assert len(niches) == 2
    assert all(isinstance(n, NicheAssignment) for n in niches)


def test_archive_get_best_for_niche():
    """Archive should return best build for specific niche."""
    archive = DiversityArchive()

    build = {"class": "Witch", "genome": {"main_skill_package": "arc"}, "defense_archetype": "armor", "full_dps": 1000.0}
    archive.update(build)

    niche = NicheAssignment("Witch", "hybrid", "armor", "spell")
    best = archive.get_best_for_niche(niche)

    assert best is not None
    assert best["full_dps"] == 1000.0


def test_archive_get_best_for_empty_niche():
    """Archive should return None for empty niche."""
    archive = DiversityArchive()

    niche = NicheAssignment("Witch", "hybrid", "armor", "spell")
    best = archive.get_best_for_niche(niche)

    assert best is None
