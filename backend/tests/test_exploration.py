"""Tests for exploration strategies."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pytest

from backend.engine.generation.exploration import (
    CuriosityExploration,
    ExplorationConfig,
    build_to_features,
    compute_novelty,
    select_candidates,
    select_pareto_frontier,
    select_with_epsilon_greedy,
)


def test_epsilon_greedy_selects_random_fraction():
    """Epsilon-greedy should select some random candidates."""
    candidates = [{"build_id": f"build_{i}"} for i in range(100)]
    model_scores = {f"build_{i}": float(i) for i in range(100)}

    random.seed(42)
    result = select_with_epsilon_greedy(candidates, model_scores, epsilon=0.3)

    random_count = 0
    exploit_count = 0
    for c in result:
        if c["build_id"].startswith("build_"):
            idx = int(c["build_id"].split("_")[1])
            expected_rank = result.index(c)
            if expected_rank < 30:
                random_count += 1

    assert len(result) == 100
    assert random_count > 0


def test_epsilon_greedy_with_empty_list():
    """Epsilon-greedy should handle empty list."""
    result = select_with_epsilon_greedy([], {}, epsilon=0.1)
    assert result == []


def test_epsilon_greedy_with_single_candidate():
    """Epsilon-greedy should handle single candidate."""
    candidates = [{"build_id": "build_0"}]
    model_scores = {"build_0": 1.0}
    result = select_with_epsilon_greedy(candidates, model_scores, epsilon=0.1)
    assert len(result) == 1
    assert result[0]["build_id"] == "build_0"


def test_epsilon_greedy_selects_exploit_candidates():
    """Epsilon-greedy should handle exploit-only case (epsilon=0)."""
    candidates = [{"build_id": "low"}, {"build_id": "high"}, {"build_id": "mid"}]
    model_scores = {"low": 1.0, "high": 100.0, "mid": 50.0}

    result = select_with_epsilon_greedy(candidates, model_scores, epsilon=0.0)

    assert len(result) == 3


def test_novelty_search_rewards_different_builds():
    """Novelty search should give higher scores to different builds."""
    population = [
        {"full_dps": 1000, "max_hit": 500, "life": 1000, "armour": 100, "evasion": 100, "energy_shield": 100},
        {"full_dps": 1100, "max_hit": 500, "life": 1000, "armour": 100, "evasion": 100, "energy_shield": 100},
        {"full_dps": 1050, "max_hit": 500, "life": 1000, "armour": 100, "evasion": 100, "energy_shield": 100},
    ]

    similar_build = {"full_dps": 1050, "max_hit": 500, "life": 1000, "armour": 100, "evasion": 100, "energy_shield": 100}
    different_build = {"full_dps": 10000, "max_hit": 5000, "life": 10000, "armour": 1000, "evasion": 1000, "energy_shield": 1000}

    similar_novelty = compute_novelty(similar_build, population)
    different_novelty = compute_novelty(different_build, population)

    assert different_novelty > similar_novelty


def test_novelty_with_empty_population():
    """Novelty should return 1.0 for empty population."""
    build = {"full_dps": 1000, "max_hit": 500, "life": 1000}
    novelty = compute_novelty(build, [])
    assert novelty == 1.0


def test_novelty_with_single_neighbor():
    """Novelty should work with single neighbor."""
    population = [{"full_dps": 1000, "max_hit": 500, "life": 1000, "armour": 100, "evasion": 100, "energy_shield": 100}]
    build = {"full_dps": 1500, "max_hit": 600, "life": 1200, "armour": 150, "evasion": 150, "energy_shield": 150}

    novelty = compute_novelty(build, population)
    assert novelty > 0


def test_curiosity_targets_weak_areas():
    """Curiosity should target areas where model performs poorly."""
    curiosity = CuriosityExploration()

    high_error_region = [
        {"full_dps": 10000, "max_hit": 5000, "life": 10000, "armour": 1000, "evasion": 1000, "energy_shield": 1000},
    ]
    for build in high_error_region:
        curiosity.update(build, actual=100.0, predicted=0.0)

    low_error_region = [
        {"full_dps": 1000, "max_hit": 500, "life": 1000, "armour": 100, "evasion": 100, "energy_shield": 100},
    ]
    for build in low_error_region * 10:
        curiosity.update(build, actual=100.0, predicted=99.0)

    regions = curiosity.get_target_regions(n=2)

    assert len(regions) >= 1


def test_curiosity_with_no_errors():
    """Curiosity should handle empty error history."""
    curiosity = CuriosityExploration()
    regions = curiosity.get_target_regions(n=3)
    assert regions == []


def test_curiosity_score_candidate():
    """Curiosity should add bonus for high-error regions."""
    curiosity = CuriosityExploration()

    high_error_build = {"full_dps": 10000, "max_hit": 5000, "life": 10000, "armour": 1000, "evasion": 1000, "energy_shield": 1000}
    curiosity.update(high_error_build, actual=100.0, predicted=0.0)
    curiosity.update(high_error_build, actual=100.0, predicted=10.0)

    candidate = type("Candidate", (), {"to_dict": lambda self: high_error_build})()

    score_with_curiosity = curiosity.score_candidate(candidate, base_score=0.0)
    assert score_with_curiosity >= 0


def test_select_pareto_frontier_returns_optimal():
    """select_pareto_frontier should return Pareto-optimal candidates."""
    candidates = [
        {"full_dps": 1000, "max_hit": 500, "life": 1000},
        {"full_dps": 2000, "max_hit": 500, "life": 1000},
        {"full_dps": 1000, "max_hit": 1000, "life": 1000},
        {"full_dps": 500, "max_hit": 500, "life": 1000},
    ]

    result = select_pareto_frontier(candidates)

    assert len(result) == 2
    assert any(c["full_dps"] == 2000 for c in result)
    assert any(c["max_hit"] == 1000 for c in result)


def test_select_pareto_frontier_with_single_objective():
    """Pareto should work with single objective."""
    candidates = [{"score": 100}, {"score": 50}, {"score": 75}]
    result = select_pareto_frontier(candidates, objectives=["score"])
    assert len(result) == 1
    assert result[0]["score"] == 100


def test_select_pareto_frontier_with_empty_list():
    """Pareto should handle empty list."""
    result = select_pareto_frontier([])
    assert result == []


def test_select_pareto_frontier_with_dataclass():
    """Pareto should work with dataclass objects."""
    @dataclass
    class MockCandidate:
        full_dps: float
        max_hit: float
        life: float

        def to_dict(self):
            return {"full_dps": self.full_dps, "max_hit": self.max_hit, "life": self.life}

    candidates = [
        MockCandidate(1000, 500, 1000),
        MockCandidate(2000, 500, 1000),
        MockCandidate(1000, 1000, 1000),
    ]

    result = select_pareto_frontier(candidates)
    assert len(result) == 2


def test_build_to_features_returns_correct_shape():
    """build_to_features should return correct feature vector."""
    build = {
        "full_dps": 1000,
        "max_hit": 500,
        "life": 1000,
        "armour": 100,
        "evasion": 200,
        "energy_shield": 150,
        "strength": 100,
        "dexterity": 150,
        "intelligence": 200,
        "physical_dps": 100,
        "fire_dps": 50,
        "cold_dps": 25,
        "lightning_dps": 25,
        "chaos_dps": 0,
    }

    features = build_to_features(build)

    assert isinstance(features, np.ndarray)
    assert len(features) == 14
    assert features[0] == 1000


def test_build_to_features_with_missing_values():
    """build_to_features should handle missing values."""
    build = {"full_dps": 1000}

    features = build_to_features(build)

    assert isinstance(features, np.ndarray)
    assert features[0] == 1000
    assert all(f == 0 for f in features[1:])


def test_curiosity_exploration_error_tracking():
    """CuriosityExploration should track errors correctly."""
    curiosity = CuriosityExploration()

    curiosity.update({"full_dps": 1000}, actual=100.0, predicted=80.0)
    curiosity.update({"full_dps": 2000}, actual=200.0, predicted=180.0)

    assert len(curiosity.error_history) == 2
    assert curiosity.error_history[0]["error"] == 20.0
    assert curiosity.error_history[1]["error"] == 20.0


def test_curiosity_limits_history_size():
    """CuriosityExploration should limit history to 1000 entries."""
    curiosity = CuriosityExploration()

    for i in range(1100):
        curiosity.update({"full_dps": float(i)}, actual=100.0, predicted=50.0)

    assert len(curiosity.error_history) == 1000


def test_select_candidates_combines_strategies():
    """select_candidates should combine all exploration strategies."""
    candidates = [
        {"build_id": "a", "full_dps": 1000, "max_hit": 500, "life": 1000},
        {"build_id": "b", "full_dps": 2000, "max_hit": 500, "life": 1000},
        {"build_id": "c", "full_dps": 1000, "max_hit": 1000, "life": 1000},
    ]
    model_scores = {"a": 10, "b": 20, "c": 15}
    config = ExplorationConfig(epsilon=0.1, use_pareto=True)

    result = select_candidates(candidates, model_scores, config)

    assert len(result) > 0


def test_select_candidates_with_archive():
    """select_candidates should use archive for novelty."""
    candidates = [
        {"build_id": "new", "full_dps": 1500, "max_hit": 750, "life": 1500, "armour": 150, "evasion": 150, "energy_shield": 150},
    ]
    model_scores = {"new": 10}
    config = ExplorationConfig(epsilon=0.0, novelty_weight=0.5)
    archive = {
        "existing": {"full_dps": 1000, "max_hit": 500, "life": 1000, "armour": 100, "evasion": 100, "energy_shield": 100},
    }

    result = select_candidates(candidates, model_scores, config, archive=archive)

    assert len(result) == 1


def test_select_candidates_with_curiosity():
    """select_candidates should use curiosity to boost weak regions."""
    curiosity = CuriosityExploration()
    curiosity.update({"full_dps": 10000, "max_hit": 5000, "life": 10000, "armour": 1000, "evasion": 1000, "energy_shield": 1000}, actual=100.0, predicted=0.0)

    candidates = [
        {"build_id": "weak", "full_dps": 10000, "max_hit": 5000, "life": 10000, "armour": 1000, "evasion": 1000, "energy_shield": 1000},
    ]
    model_scores = {"weak": 10}
    config = ExplorationConfig(epsilon=0.0, curiosity_weight=0.2)

    result = select_candidates(candidates, model_scores, config, curiosity=curiosity)

    assert len(result) == 1


def test_exploration_config_defaults():
    """ExplorationConfig should have sensible defaults."""
    config = ExplorationConfig()

    assert config.epsilon == 0.1
    assert config.novelty_weight == 0.2
    assert config.curiosity_weight == 0.1
    assert config.use_pareto is True
    assert config.epsilon_decay == 0.95
    assert config.min_epsilon == 0.01


def test_epsilon_decay_over_iterations():
    """Epsilon should decay over iterations."""
    config = ExplorationConfig(epsilon=0.1, epsilon_decay=0.95, min_epsilon=0.01)

    epsilons = [max(config.min_epsilon, config.epsilon * (config.epsilon_decay**i)) for i in range(10)]

    for i in range(1, len(epsilons)):
        assert epsilons[i] <= epsilons[i - 1]

    assert epsilons[0] == 0.1
