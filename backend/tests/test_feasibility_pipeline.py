import json
from unittest.mock import MagicMock, patch

import pytest

from backend.engine.generation.runner import generate_from_skeleton, run_generation
from backend.engine.scenarios.loader import (
    ScenarioGateThresholds,
    ScenarioReservationThreshold,
    ScenarioTemplate,
)


@pytest.fixture
def mock_skeleton(tmp_path, monkeypatch):
    skeleton_data = {
        "skeleton_id": "test_skeleton",
        "class_name": "marauder",
        "ascendancy": "Juggernaut",
        "main_skill": "cyclone",
        "skill_links": ["cyclone", "Melee Physical Damage"],
        "aura_package": ["Determination"],
        "defense_layer": "armour",
        "budget_tier": "starter",
        "target_gates": {"dps": 1000000},
        "required_uniques": [],
        "tree_path": "1,2,3",
    }

    skeletons_dir = tmp_path / "skeletons"
    skeletons_dir.mkdir()
    skeleton_file = skeletons_dir / "test_skeleton.json"
    with open(skeleton_file, "w") as f:
        json.dump(skeleton_data, f)

    from backend.app.settings import settings

    monkeypatch.setattr(settings, "data_path", tmp_path)

    return skeleton_data


@pytest.fixture
def mock_templates():
    return [
        ScenarioTemplate(
            scenario_id="test_scenario",
            version="v0",
            profile_id="pinnacle",
            pob_config={},
            gate_thresholds=ScenarioGateThresholds(
                min_full_dps=100000,
                min_max_hit=5000,
                reservation=ScenarioReservationThreshold(max_percent=95.0),
                resists={"fire": 75, "cold": 75, "lightning": 75, "chaos": 0},
                attributes={"strength": 100, "dexterity": 100, "intelligence": 100},
            ),
        )
    ]


def test_generate_from_skeleton_pipeline(mock_skeleton, mock_templates):
    evaluator = MagicMock()

    candidate = generate_from_skeleton(
        skeleton_id="test_skeleton",
        seed=1,
        profile_id="pinnacle",
        templates=mock_templates,
        evaluator=evaluator,
    )

    assert candidate.class_name == "marauder"
    assert candidate.main_skill_package == "cyclone"
    assert hasattr(candidate, "failures")


def test_run_generation_with_skeleton(mock_skeleton, mock_templates, tmp_path):
    repo = MagicMock()
    evaluator = MagicMock()
    evaluator.evaluate_build.return_value = (MagicMock(value="evaluated"), [])
    evaluator.pop_last_evaluation_provenance.return_value = None

    with patch("backend.engine.generation.runner.list_templates", return_value=mock_templates):
        summary = run_generation(
            count=1,
            seed_start=1,
            ruleset_id="test_ruleset",
            profile_id="pinnacle",
            skeleton_id="test_skeleton",
            base_path=tmp_path,
            repo=repo,
            evaluator=evaluator,
        )

    assert summary["status"] in ["completed", "partial", "failed"]
    assert summary["generation"]["processed"] == 1
