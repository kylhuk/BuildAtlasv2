"""Tests for the EP-V4 surrogate dataset snapshot builder."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Mapping

from backend.engine.surrogate.dataset import (
    FEATURE_AFFIX_TOTAL_LINES,
    FEATURE_GEM_MAIN_LINK_REQUIREMENT,
    FEATURE_GEM_MAX_LINK_COUNT,
    FEATURE_GEM_TOTAL_COUNT,
    FEATURE_ITEM_SLOT_COUNT,
    FEATURE_PASSIVE_NODE_COUNT,
    FEATURE_PASSIVE_REQUIRED_TARGETS,
    FEATURE_SCHEMA_VERSION,
    FEATURE_IDENTITY_TOKENS,
    FEATURE_IDENTITY_CROSS_TOKENS,
    build_dataset_snapshot,
)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _create_build(
    builds_root: Path,
    build_id: str,
    genome: Mapping[str, object],
    metrics: Mapping[str, object],
    build_details: Mapping[str, object] | None = None,
) -> None:
    build_dir = builds_root / build_id
    build_dir.mkdir(parents=True, exist_ok=True)
    _write_json(build_dir / "genome.json", genome)
    _write_json(build_dir / "metrics_raw.json", metrics)
    if build_details is not None:
        _write_json(build_dir / "build_details.json", build_details)


def test_snapshot_round_trip(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    builds_root = data_root / "builds"
    genome = {
        "schema_version": "v0",
        "seed": 42,
        "class": "Marauder",
        "ascendancy": "Chieftain",
        "main_skill_package": "sunder",
        "defense_archetype": "armour",
        "budget_tier": "endgame",
        "profile_id": "alpha",
    }
    metrics = {
        "mapping": {
            "metrics": {
                "full_dps": 5200,
                "max_hit": 4700,
                "utility_score": 1.2,
            },
            "defense": {
                "armour": 6500,
                "evasion": 2100,
                "resists": {
                    "fire": 78,
                    "cold": 79.5,
                    "lightning": 80,
                    "chaos": 65,
                },
            },
            "resources": {"life": 9600, "mana": 2800},
            "reservation": {"reserved_percent": 62, "available_percent": 88},
            "attributes": {"strength": 185, "dexterity": 140, "intelligence": 125},
        }
    }
    _create_build(builds_root, "sample", genome, metrics)

    output_root = data_root / "datasets" / "ep-v4"
    result = build_dataset_snapshot(data_root, output_root, "snapshot-roundtrip")

    lines = result.dataset_path.read_text(encoding="utf-8").splitlines()
    assert result.row_count == len(lines)
    assert result.feature_schema_version == FEATURE_SCHEMA_VERSION

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert manifest["row_count"] == result.row_count
    assert manifest["dataset_hash"] == result.dataset_hash

    computed_hash = sha256()
    for line in lines:
        computed_hash.update(line.encode("utf-8"))
        computed_hash.update(b"\n")
    assert result.dataset_hash == computed_hash.hexdigest()


def test_ordering_and_hash_stability(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    builds_root = data_root / "builds"

    base_genome = {
        "schema_version": "v0",
        "seed": 12,
        "class": "Ranger",
        "ascendancy": "Deadeye",
        "main_skill_package": "tornado_shot",
        "defense_archetype": "evasion",
        "budget_tier": "midgame",
        "profile_id": "bravo",
    }

    metrics_zeta = {
        "zeta": {
            "metrics": {
                "full_dps": 1000,
                "max_hit": 1200,
                "utility_score": 0.9,
            },
            "defense": {
                "armour": 2500,
                "evasion": 3400,
                "resists": {
                    "fire": 60,
                    "cold": 60,
                    "lightning": 60,
                    "chaos": 50,
                },
            },
            "resources": {"life": 4200, "mana": 800},
            "reservation": {"reserved_percent": 55, "available_percent": 95},
            "attributes": {"strength": 90, "dexterity": 140, "intelligence": 120},
        },
        "alpha": {
            "metrics": {
                "full_dps": 1100,
                "max_hit": 1300,
                "utility_score": 1.0,
            },
            "defense": {
                "armour": 2600,
                "evasion": 3200,
                "resists": {
                    "fire": 65,
                    "cold": 65,
                    "lightning": 65,
                    "chaos": 55,
                },
            },
            "resources": {"life": 4300, "mana": 820},
            "reservation": {"reserved_percent": 57, "available_percent": 93},
            "attributes": {"strength": 92, "dexterity": 142, "intelligence": 122},
        },
    }

    metrics_alpha = {
        "beta": metrics_zeta["alpha"],
        "omega": metrics_zeta["zeta"],
    }

    _create_build(builds_root, "zzz-build", base_genome, metrics_zeta)
    _create_build(builds_root, "aaa-build", {**base_genome, "seed": 99}, metrics_alpha)

    output_root = data_root / "datasets" / "ep-v4"
    first = build_dataset_snapshot(data_root, output_root, "order-01")
    second = build_dataset_snapshot(data_root, output_root, "order-02")

    assert first.row_count == second.row_count
    assert first.dataset_hash == second.dataset_hash

    lines = first.dataset_path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    assert rows, "dataset should have rows"
    first_row = rows[0]
    assert first_row["build_id"] == "aaa-build"
    assert first_row["scenario_id"] == "beta"

    last_row = rows[-1]
    assert last_row["build_id"] == "zzz-build"
    zzz_rows = [row for row in rows if row["build_id"] == "zzz-build"]
    assert len(zzz_rows) == 2
    assert zzz_rows[0]["scenario_id"] == "alpha"
    assert zzz_rows[1]["scenario_id"] == "zeta"


def test_missing_artifacts_skipped(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    builds_root = data_root / "builds"

    genome = {
        "schema_version": "v0",
        "seed": 7,
        "class": "Shadow",
        "ascendancy": "Saboteur",
        "main_skill_package": "blade_flurry",
        "defense_archetype": "hybrid",
        "budget_tier": "starter",
        "profile_id": "charlie",
    }
    metrics = {
        "pinnacle": {
            "metrics": {
                "full_dps": 2600,
                "max_hit": 1800,
                "utility_score": 0.6,
            },
            "defense": {
                "armour": 1800,
                "evasion": 2400,
                "resists": {
                    "fire": 50,
                    "cold": 50,
                    "lightning": 50,
                    "chaos": 40,
                },
            },
            "resources": {"life": 3200, "mana": 600},
            "reservation": {"reserved_percent": 48, "available_percent": 88},
            "attributes": {"strength": 80, "dexterity": 100, "intelligence": 110},
        }
    }

    _create_build(builds_root, "valid", genome, metrics)

    missing_genome_dir = builds_root / "missing-genome"
    missing_genome_dir.mkdir(parents=True, exist_ok=True)
    _write_json(missing_genome_dir / "metrics_raw.json", metrics)

    missing_metrics_dir = builds_root / "missing-metrics"
    missing_metrics_dir.mkdir(parents=True, exist_ok=True)
    _write_json(missing_metrics_dir / "genome.json", genome)

    output_root = data_root / "datasets" / "ep-v4"
    result = build_dataset_snapshot(data_root, output_root, "missing-artifacts")

    assert result.row_count == 1
    lines = result.dataset_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["build_id"] == "valid"


def test_rows_include_build_details_features(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    builds_root = data_root / "builds"
    genome = {
        "schema_version": "v0",
        "seed": 99,
        "class": "Witch",
        "ascendancy": "Occultist",
        "main_skill_package": "arc",
        "defense_archetype": "energy_shield",
        "budget_tier": "endgame",
        "profile_id": "feature-test",
    }
    metrics = {
        "artefact": {
            "metrics": {
                "full_dps": 3000,
                "max_hit": 3500,
                "utility_score": 1.4,
            },
            "defense": {"armour": 1000, "evasion": 1000},
            "resources": {"life": 5300, "mana": 1200},
            "reservation": {"reserved_percent": 60, "available_percent": 70},
            "attributes": {"strength": 100, "dexterity": 100, "intelligence": 180},
        }
    }
    build_details = {
        "items": {
            "slot_templates": [
                {
                    "slot_id": "weapon_2h",
                    "base_type": "Sword",
                    "contributions": {
                        "fire": 5,
                        "strength": 7,
                        "life": 12,
                        "energy_shield": 3,
                    },
                    "adjustable": True,
                }
            ]
        },
        "passives": {
            "node_ids": ["node-1", "node-2", "keystone-chaos-inoculation"],
            "required_targets": ["target"],
            "node_details": [
                {"id": "node-1", "kind": "small", "pob_id": "11"},
                {"id": "mastery-lightning", "kind": "mastery", "pob_id": "12"},
                {
                    "id": "keystone-chaos-inoculation",
                    "kind": "keystone",
                    "pob_id": "13",
                },
            ],
            "mastery_ids": ["mastery-lightning"],
            "keystone_ids": ["keystone-chaos-inoculation"],
        },
        "gems": {
            "groups": [
                {
                    "id": "damage",
                    "group_type": "damage",
                    "gems": ["arcane", "added_lightning"],
                },
                {
                    "id": "utility",
                    "group_type": "utility",
                    "gems": ["blood_magic", "controlled_destruction"],
                },
            ],
            "full_dps_group_id": "damage",
            "socket_plan": {
                "assignments": [
                    {"group_id": "damage", "slot_id": "weapon_2h", "link_count": 4},
                    {"group_id": "utility", "slot_id": "helmet", "link_count": 3},
                ],
                "main_link_requirement": 5,
            },
        },
    }
    _create_build(builds_root, "feature-build", genome, metrics, build_details=build_details)
    output_root = data_root / "datasets" / "ep-v4"
    result = build_dataset_snapshot(data_root, output_root, "features")
    rows = [
        json.loads(line)
        for line in result.dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    row = rows[0]
    assert row[FEATURE_ITEM_SLOT_COUNT] == 1.0
    assert row[FEATURE_AFFIX_TOTAL_LINES] == 4.0
    assert row[FEATURE_PASSIVE_NODE_COUNT] == 3.0
    assert row[FEATURE_PASSIVE_REQUIRED_TARGETS] == 1.0
    assert row[FEATURE_GEM_TOTAL_COUNT] == 4.0
    assert row[FEATURE_GEM_MAX_LINK_COUNT] == 4.0
    assert row[FEATURE_GEM_MAIN_LINK_REQUIREMENT] == 5.0
    identity_tokens = set(row[FEATURE_IDENTITY_TOKENS])
    cross_tokens = set(row[FEATURE_IDENTITY_CROSS_TOKENS])
    assert "slot:weapon_2h" in identity_tokens
    assert "base_type:sword" in identity_tokens
    assert "slot:weapon_2h:adjustable" in identity_tokens
    assert "passive_node:node-1" in identity_tokens
    assert "passive_target:target" in identity_tokens
    assert "passive_mastery:mastery-lightning" in identity_tokens
    assert "passive_keystone:keystone-chaos-inoculation" in identity_tokens
    assert "passive_kind:mastery" in identity_tokens
    assert "passive_kind:keystone" in identity_tokens
    assert "slot_base_affix:weapon_2h|base_type:sword|affix:resists" in identity_tokens
    assert "slot_affix:weapon_2h|affix:life" in identity_tokens
    assert "main_gem:arcane" in identity_tokens
    assert "main_support:added_lightning" in identity_tokens
    assert "main_slot:weapon_2h" in identity_tokens
    assert "slot:weapon_2h|gem:arcane" in identity_tokens
    assert "group:damage|gem_pair:added_lightning+arcane" in identity_tokens
    assert "link_requirement:5-6" in identity_tokens
    assert "base_type:sword|main_gem:arcane" in cross_tokens
    assert "main_gem:arcane|passive_node:node-1" in cross_tokens
    assert "main_gem:arcane|main_support:added_lightning" in cross_tokens
    assert "main_support:added_lightning|passive_target:target" in cross_tokens
    assert "main_gem:arcane|main_slot:weapon_2h" in cross_tokens
    assert "main_gem:arcane|passive_keystone:keystone-chaos-inoculation" in cross_tokens
    assert "main_support:added_lightning|passive_mastery:mastery-lightning" in cross_tokens
    assert "main_gem:arcane|main_support:added_lightning|passive_node:node-1" in cross_tokens


def test_snapshot_filters_by_profile_id(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    builds_root = data_root / "builds"
    genome = {
        "schema_version": "v0",
        "seed": 1,
        "class": "Marauder",
        "ascendancy": "Chieftain",
        "main_skill_package": "sunder",
        "defense_archetype": "armour",
        "budget_tier": "endgame",
        "profile_id": "filtered",
    }
    metrics = {
        "alpha": {
            "metrics": {
                "full_dps": 1500,
                "max_hit": 1400,
                "utility_score": 1.2,
            },
            "defense": {
                "armour": 2000,
                "evasion": 1200,
            },
            "resources": {
                "life": 5000,
                "mana": 1000,
            },
            "reservation": {
                "reserved_percent": 50,
                "available_percent": 50,
            },
            "attributes": {
                "strength": 120,
                "dexterity": 110,
                "intelligence": 130,
            },
        }
    }
    _create_build(builds_root, "filtered-build", genome, metrics)
    _create_build(builds_root, "other-build", {**genome, "profile_id": "other", "seed": 2}, metrics)
    output_root = data_root / "datasets" / "ep-v4"
    result = build_dataset_snapshot(data_root, output_root, "profile-filter", profile_id="filtered")
    rows = [
        json.loads(line)
        for line in result.dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["profile_id"] == "filtered"


def test_snapshot_filters_by_scenario_id(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    builds_root = data_root / "builds"
    genome = {
        "schema_version": "v0",
        "seed": 1,
        "class": "Marauder",
        "ascendancy": "Chieftain",
        "main_skill_package": "sunder",
        "defense_archetype": "armour",
        "budget_tier": "endgame",
        "profile_id": "filtered",
    }
    base_payload = {
        "metrics": {
            "full_dps": 1500,
            "max_hit": 1400,
            "utility_score": 1.2,
        },
        "defense": {
            "armour": 2000,
            "evasion": 1200,
        },
        "resources": {
            "life": 5000,
            "mana": 1000,
        },
        "reservation": {
            "reserved_percent": 50,
            "available_percent": 50,
        },
        "attributes": {
            "strength": 120,
            "dexterity": 110,
            "intelligence": 130,
        },
    }
    _create_build(builds_root, "alpha-build", genome, {"alpha": base_payload})
    _create_build(
        builds_root,
        "beta-build",
        {**genome, "seed": 2},
        {"beta": {**base_payload, "metrics": {**base_payload["metrics"], "full_dps": 1700}}},
    )
    output_root = data_root / "datasets" / "ep-v4"
    result = build_dataset_snapshot(data_root, output_root, "scenario-filter", scenario_id="beta")
    rows = [
        json.loads(line)
        for line in result.dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["scenario_id"] == "beta"


def test_categorical_identity_tokens(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    builds_root = data_root / "builds"
    genome = {
        "schema_version": "v0",
        "seed": 7,
        "class": "Marauder",
        "ascendancy": "Chieftain",
        "main_skill_package": "sunder",
        "defense_archetype": "armour",
        "budget_tier": "endgame",
        "profile_id": "TokenProfile",
    }
    metrics = {
        "omega": {
            "metrics": {
                "full_dps": 2100,
                "max_hit": 1800,
                "utility_score": 1.5,
            },
            "defense": {
                "armour": 2500,
                "evasion": 1500,
            },
            "resources": {
                "life": 5200,
                "mana": 1200,
            },
            "reservation": {
                "reserved_percent": 55,
                "available_percent": 60,
            },
            "attributes": {
                "strength": 130,
                "dexterity": 120,
                "intelligence": 140,
            },
        }
    }
    _create_build(builds_root, "token-row", genome, metrics)
    output_root = data_root / "datasets" / "ep-v4"
    result = build_dataset_snapshot(data_root, output_root, "categorical-tokens")
    rows = [
        json.loads(line)
        for line in result.dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    tokens = set(rows[0][FEATURE_IDENTITY_TOKENS])
    expected = {
        "profile_id:tokenprofile",
        "scenario_id:omega",
        "class:marauder",
        "ascendancy:chieftain",
        "defense:armour",
        "budget:endgame",
        "main_skill:sunder",
    }
    assert expected.issubset(tokens)


def test_snapshot_id_rejects_path_traversal(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    output_root = data_root / "datasets" / "ep-v4"

    try:
        build_dataset_snapshot(data_root, output_root, "../escape")
    except ValueError as exc:
        assert "snapshot_id" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("expected ValueError for invalid snapshot id")
