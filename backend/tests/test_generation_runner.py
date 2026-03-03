import json
import random
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from typing import Any, Sequence

from backend.engine.generation.runner import (
    CANONICAL_ITEM_SLOTS,
    Candidate,
    POB_TARGET_VERSION,
    _predict_candidates,
    _optimizer_objectives_from_payload,
    _select_optimizer_elites,
    _select_surrogate_optimizer_elites,
    _assert_no_stub_metrics,
    _build_worker_xml_payload,
    _load_surrogate_predictor,
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


def _make_optimizer_candidate(
    build_id: str,
    seed: int,
    actual_full_dps: float,
    predicted_full_dps: float,
    predicted_max_hit: float,
    predicted_pass_probability: float | None = None,
) -> Candidate:
    genome = GenomeV0(
        seed=seed,
        class_name="Marauder",
        ascendancy="Chieftain",
        main_skill_package="sunder",
        defense_archetype="armour",
        budget_tier="endgame",
        profile_id="alpha",
    )
    candidate = Candidate(
        seed=seed,
        build_id=build_id,
        main_skill_package=genome.main_skill_package,
        class_name=genome.class_name,
        ascendancy=genome.ascendancy,
        budget_tier=genome.budget_tier,
        failures=[],
        metrics_payload={
            "alpha": {
                "metrics": {
                    "full_dps": actual_full_dps,
                    "max_hit": 1000.0,
                    "utility_score": 1.0,
                    "total_cost_chaos": 10.0,
                }
            }
        },
        genome=genome,
        code_payload="{}",
        build_details_payload={},
    )
    candidate.predicted_metrics = {
        "full_dps": predicted_full_dps,
        "max_hit": predicted_max_hit,
    }
    candidate.predicted_full_dps = predicted_full_dps
    candidate.pass_probability = predicted_pass_probability
    return candidate


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
        slot.attrib.get("name") for slot in item_set.findall("Slot") if slot.attrib.get("name")
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


def _scenario_metrics_entry(
    full_dps: float, max_hit: float, cost: float, warnings: list[str] | None = None
) -> dict[str, Any]:
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


def test_surrogate_optimizer_elites_prefers_predicted_metrics() -> None:
    actual = _make_optimizer_candidate(
        build_id="actual",
        seed=1,
        actual_full_dps=1500.0,
        predicted_full_dps=500.0,
        predicted_max_hit=1200.0,
        predicted_pass_probability=0.2,
    )
    surrogate = _make_optimizer_candidate(
        build_id="surrogate",
        seed=2,
        actual_full_dps=1200.0,
        predicted_full_dps=700.0,
        predicted_max_hit=1400.0,
        predicted_pass_probability=0.4,
    )

    def stub_predictor(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        assert len(rows) == 2
        return [
            {
                "metrics": {"full_dps": 900.0, "max_hit": 1200.0},
                "pass_probability": 0.25,
            },
            {
                "metrics": {"full_dps": 2200.0, "max_hit": 800.0},
                "pass_probability": 0.9,
            },
        ]

    _predict_candidates(stub_predictor, [actual, surrogate])

    actual_elites = _select_optimizer_elites([actual, surrogate], 1)
    assert actual_elites[0][0].build_id == "actual"
    surrogate_elites = _select_surrogate_optimizer_elites([actual, surrogate], 1)
    assert surrogate_elites[0][0].build_id == "surrogate"


def test_load_surrogate_predictor_legacy_mapping(tmp_path: Path) -> None:
    payload = {
        "model_id": "legacy",
        "predictions": {"sunder": 12.5},
        "default": 2.0,
    }
    path = tmp_path / "legacy-surrogate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    model_id, predictor = _load_surrogate_predictor(path)

    assert model_id == "legacy"
    scores = list(predictor([{"main_skill_package": "sunder"}, {"main_skill_package": "other"}]))
    assert scores == [12.5, 2.0]


def test_load_surrogate_predictor_legacy_scores_mapping(tmp_path: Path) -> None:
    payload = {
        "model_id": "legacy-scores",
        "scores": {"mace": 2.5},
        "default": 1.0,
    }
    path = tmp_path / "legacy-surrogate-scores.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    model_id, predictor = _load_surrogate_predictor(path)

    assert model_id == "legacy-scores"
    scores = list(predictor([{"main_skill_package": "mace"}, {"main_skill_package": "other"}]))
    assert scores == [2.5, 1.0]


def test_load_surrogate_predictor_surrogate_model_payload(tmp_path: Path) -> None:
    payload = {
        "model_id": "surrogate-latest",
        "dataset_snapshot_id": "snapshot",
        "feature_schema_version": "1.0",
        "global_metrics": {
            "full_dps": {"mean": 50.0, "std": 1.0, "min": 10.0, "max": 80.0, "count": 1},
            "max_hit": {"mean": 200.0, "std": 1.0, "min": 150.0, "max": 250.0, "count": 1},
            "utility_score": {"mean": 1.0, "std": 0.1, "min": 0.0, "max": 2.0, "count": 1},
        },
        "main_skill_metrics": {
            "mapping_t16|alpha|arc": {
                "full_dps": 120.0,
                "max_hit": 300.0,
                "utility_score": 2.0,
            }
        },
        "feature_stats": {},
        "feature_weights": {},
        "identity_token_effects": {},
        "identity_cross_token_effects": {},
        "pass_metric": "full_dps",
        "backend": "ep-v4-baseline",
        "backend_version": "0.1.0",
        "compute_backend": "cpu",
        "token_learner_backend": "torch_sparse_sgd",
        "trained_at_utc": "2025-01-01T00:00:00Z",
        "target_transforms": {
            "full_dps": "identity",
            "max_hit": "identity",
            "utility_score": "identity",
        },
    }
    path = tmp_path / "surrogate-model.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    model_id, predictor = _load_surrogate_predictor(path)

    assert model_id == "surrogate-latest"
    rows = [
        {
            "scenario_id": "mapping_t16",
            "profile_id": "alpha",
            "main_skill_package": "arc",
        },
        {
            "scenario_id": "mapping_t16",
            "profile_id": "alpha",
            "main_skill_package": "other",
        },
    ]
    predictions = list(predictor(rows))
    assert len(predictions) == len(rows)
    assert all("metrics" in prediction for prediction in predictions)
    full_dps_values = [prediction["metrics"]["full_dps"] for prediction in predictions]
    assert len(set(full_dps_values)) > 1
    assert any(value != 0.0 for value in full_dps_values)
    assert full_dps_values[1] == 50.0
    assert full_dps_values[0] > full_dps_values[1]


def test_select_surrogate_elites_ignores_missing_pass_probability() -> None:
    high = _make_optimizer_candidate(
        build_id="high",
        seed=1,
        actual_full_dps=100.0,
        predicted_full_dps=200.0,
        predicted_max_hit=120.0,
        predicted_pass_probability=None,
    )
    low = _make_optimizer_candidate(
        build_id="low",
        seed=2,
        actual_full_dps=90.0,
        predicted_full_dps=150.0,
        predicted_max_hit=110.0,
        predicted_pass_probability=0.5,
    )

    elites = _select_surrogate_optimizer_elites([low, high], 2, rng=random.Random(0))
    assert elites[0][0] is high


def test_select_surrogate_elites_tie_breaker_depends_on_rng() -> None:
    tie_a = _make_optimizer_candidate(
        build_id="tie-a",
        seed=3,
        actual_full_dps=100.0,
        predicted_full_dps=160.0,
        predicted_max_hit=130.0,
        predicted_pass_probability=0.3,
    )
    tie_b = _make_optimizer_candidate(
        build_id="tie-b",
        seed=4,
        actual_full_dps=105.0,
        predicted_full_dps=160.0,
        predicted_max_hit=130.0,
        predicted_pass_probability=0.3,
    )

    elites = _select_surrogate_optimizer_elites([tie_a, tie_b], 2, rng=random.Random(0))
    order = [candidate.build_id for candidate, _ in elites]
    assert order != ["tie-a", "tie-b"]



def test_stub_tripwire_detects_exact_stub_metrics() -> None:
    entries = [
        (1, 120.0, 4502.0),
        (2, 240.0, 4504.0),
        (3, 360.0, 4506.0),
    ]
    with pytest.raises(ValueError, match="PoB evaluation inactive; stub metrics detected"):
        _assert_no_stub_metrics(entries)



def test_stub_tripwire_detects_linear_stub_metrics() -> None:
    entries = [
        (5, 600.1, 4510.0),
        (10, 1200.2, 4520.0),
        (15, 1800.3, 4530.0),
        (20, 2400.4, 4540.0),
    ]
    with pytest.raises(ValueError, match="PoB evaluation inactive; stub metrics detected"):
        _assert_no_stub_metrics(entries)
