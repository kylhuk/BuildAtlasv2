"""
Exploration strategies for ML training.

Implements various exploration strategies:
- Epsilon-greedy: random exploration with exploitation
- Novelty search: reward builds different from seen ones
- Curiosity-driven: target areas where model is weak
- Pareto optimization: multi-objective selection
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ExplorationConfig:
    """Configuration for exploration strategy."""

    epsilon: float = 0.1  # Random exploration rate (0.1 = 10% random)
    novelty_weight: float = 0.2  # Weight for novelty bonus
    curiosity_weight: float = 0.1  # Weight for curiosity bonus
    use_pareto: bool = True  # Use Pareto selection

    # Decay schedule
    epsilon_decay: float = 0.95  # Multiply epsilon each iteration
    min_epsilon: float = 0.01  # Minimum exploration rate


def select_with_epsilon_greedy(
    candidates: list[Any],
    model_scores: dict[str, float],
    epsilon: float = 0.1,
) -> list[Any]:
    """Select candidates with epsilon-greedy strategy.

    Args:
        candidates: Available candidates
        model_scores: Dictionary mapping candidate_id -> score
        epsilon: Probability of random selection (exploration)

    Returns:
        Selected candidates
    """
    n = len(candidates)
    if n == 0:
        return []

    n_random = max(1, int(n * epsilon))

    # Random selection (exploration)
    random.shuffle(candidates)
    random_selection = candidates[:n_random]

    # Model-based selection (exploitation)
    remaining = candidates[n_random:]
    scored = [(c, model_scores.get(_get_candidate_id(c), 0)) for c in remaining]
    scored.sort(key=lambda x: x[1], reverse=True)
    exploit_selection = [c for c, _ in scored]

    return random_selection + exploit_selection


def compute_novelty(
    build: dict[str, Any],
    population: list[dict[str, Any]],
    k: int = 5,
) -> float:
    """Compute novelty score based on distance to k-nearest neighbors.

    Novelty = average distance to k nearest neighbors in feature space

    Args:
        build: Build to compute novelty for
        population: Existing builds to compare against
        k: Number of neighbors to consider

    Returns:
        Novelty score (higher = more novel)
    """
    if not population:
        return 1.0  # Most novel if nothing to compare to

    # Feature vector
    features = build_to_features(build)
    pop_features = [build_to_features(p) for p in population]

    if not pop_features:
        return 1.0

    # Compute distances
    distances = [np.linalg.norm(features - pf) for pf in pop_features]
    distances.sort()

    # Average distance to k nearest
    k_distances = distances[: min(k, len(distances))]
    if not k_distances:
        return 0.0

    return sum(k_distances) / len(k_distances)


def build_to_features(build: dict[str, Any]) -> np.ndarray:
    """Convert build to feature vector for distance computation.

    Args:
        build: Build dictionary

    Returns:
        Feature vector as numpy array
    """
    # Extract key features
    features = [
        build.get("full_dps", 0) or 0,
        build.get("max_hit", 0) or 0,
        build.get("life", 0) or 0,
        build.get("armour", 0) or 0,
        build.get("evasion", 0) or 0,
        build.get("energy_shield", 0) or 0,
        # Attribute requirements
        build.get("strength", 0) or 0,
        build.get("dexterity", 0) or 0,
        build.get("intelligence", 0) or 0,
        # Damage types
        build.get("physical_dps", 0) or 0,
        build.get("fire_dps", 0) or 0,
        build.get("cold_dps", 0) or 0,
        build.get("lightning_dps", 0) or 0,
        build.get("chaos_dps", 0) or 0,
    ]
    return np.array(features, dtype=np.float64)


def _get_candidate_id(candidate: Any) -> str:
    """Get candidate ID from various candidate types."""
    if hasattr(candidate, "build_id"):
        return candidate.build_id
    elif hasattr(candidate, "id"):
        return candidate.id
    elif isinstance(candidate, dict):
        return candidate.get("build_id", "")
    return str(candidate)


def select_pareto_frontier(
    candidates: list[Any],
    objectives: list[str] = None,
) -> list[Any]:
    """Select Pareto-optimal candidates across multiple objectives.

    Args:
        candidates: List of candidates (with .to_dict() or .__dict__)
        objectives: List of objective keys to optimize

    Returns:
        List of Pareto-optimal candidates
    """
    if objectives is None:
        objectives = ["full_dps", "max_hit", "life"]

    def get_value(candidate: Any, key: str) -> float:
        """Extract value from candidate."""
        if hasattr(candidate, "to_dict"):
            c_dict = candidate.to_dict()
        elif hasattr(candidate, "__dict__"):
            c_dict = candidate.__dict__
        else:
            c_dict = candidate

        return c_dict.get(key, 0) or 0

    def dominates(a: dict, b: dict) -> bool:
        """A dominates B if A is >= in all objectives and > in at least one."""
        better_in_any = False
        for obj in objectives:
            if get_value(a, obj) < get_value(b, obj):
                return False
            if get_value(a, obj) > get_value(b, obj):
                better_in_any = True
        return better_in_any

    # Convert to dicts
    c_dicts = []
    for c in candidates:
        if hasattr(c, "to_dict"):
            c_dicts.append(c.to_dict())
        elif hasattr(c, "__dict__"):
            c_dicts.append(c.__dict__)
        else:
            c_dicts.append(c)

    # Find Pareto frontier
    pareto = []
    for _i, candidate_dict in enumerate(c_dicts):
        dominated = False
        for existing_dict in pareto:
            if dominates(existing_dict, candidate_dict):
                dominated = True
                break
        if not dominated:
            # Remove any that this candidate dominates
            new_pareto = []
            for existing_dict in pareto:
                if not dominates(candidate_dict, existing_dict):
                    new_pareto.append(existing_dict)
            pareto = new_pareto
            pareto.append(candidate_dict)

    # Map back to candidates
    pareto_candidates = []
    for candidate_dict in pareto:
        # Find corresponding candidate
        for c in candidates:
            if hasattr(c, "to_dict"):
                if c.to_dict() == candidate_dict:
                    pareto_candidates.append(c)
                    break
            elif hasattr(c, "__dict__"):
                if c.__dict__ == candidate_dict:
                    pareto_candidates.append(c)
                    break
            elif c == candidate_dict:
                pareto_candidates.append(c)
                break

    return pareto_candidates


class CuriosityExploration:
    """Explore areas where model performs poorly."""

    def __init__(self):
        self.error_history: list[dict[str, Any]] = []

    def update(self, build: dict[str, Any], actual: float, predicted: float):
        """Record prediction error for this build.

        Args:
            build: Build dictionary
            actual: Actual metric value
            predicted: Predicted metric value
        """
        error = abs(actual - predicted)
        self.error_history.append(
            {
                "build": build,
                "error": error,
                "features": build_to_features(build),
            }
        )

        # Keep only last 1000 entries
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]

    def get_target_regions(self, n: int = 3) -> list[dict[str, Any]]:
        """Get regions of feature space where model is weakest.

        Args:
            n: Number of regions to return

        Returns:
            List of region centers with metadata
        """
        if not self.error_history:
            return []

        # Find builds with highest error
        sorted_errors = sorted(self.error_history, key=lambda x: x["error"], reverse=True)

        # Take top 50 for clustering
        high_error = [e["features"] for e in sorted_errors[:50]]

        if len(high_error) < n:
            return []

        # Simple clustering: split into n groups
        n = min(n, len(high_error))
        clusters = np.array_split(high_error, n)

        return [
            {
                "center": cluster.mean(axis=0).tolist(),
                "size": len(cluster),
                "avg_error": np.mean([e["error"] for e in sorted_errors[: len(cluster)]]),
            }
            for cluster in clusters
        ]

    def score_candidate(
        self,
        candidate: Any,
        base_score: float,
    ) -> float:
        """Score candidate with curiosity bonus for unexplored regions.

        Args:
            candidate: Candidate to score
            base_score: Base model score

        Returns:
            Score with curiosity bonus
        """
        # Get build dict
        if hasattr(candidate, "to_dict"):
            c_dict = candidate.to_dict()
        elif hasattr(candidate, "__dict__"):
            c_dict = candidate.__dict__
        else:
            c_dict = candidate

        features = build_to_features(c_dict)

        # How close to high-error regions?
        bonus = 0.0
        for region in self.get_target_regions():
            center = np.array(region["center"])
            distance = np.linalg.norm(features - center)
            # Inverse distance bonus
            bonus += 1.0 / (1.0 + distance)

        return base_score + bonus * 0.1  # 10% curiosity weight


def select_candidates(
    candidates: list[Any],
    model_scores: dict[str, float] | None = None,
    config: ExplorationConfig | None = None,
    archive: dict[str, dict[str, Any]] | None = None,
    curiosity: CuriosityExploration | None = None,
    iteration: int = 0,
) -> list[Any]:
    """Unified candidate selection with exploration.

    Combines:
    - Epsilon-greedy random exploration
    - Model-based exploitation
    - Novelty search for diversity
    - Curiosity-driven targeting of weak areas
    - Pareto frontier for multi-objective

    Args:
        candidates: Available candidates
        model_scores: Optional model scores for each candidate
        config: Exploration configuration
        archive: Archive of best builds per niche
        curiosity: Curiosity tracking
        iteration: Current iteration number (for decay)

    Returns:
        Selected candidates sorted by score
    """
    if config is None:
        config = ExplorationConfig()

    if model_scores is None:
        model_scores = {}

    # Apply epsilon decay
    epsilon = max(config.min_epsilon, config.epsilon * (config.epsilon_decay**iteration))

    # Get build dicts
    build_dicts = []
    for c in candidates:
        if hasattr(c, "to_dict"):
            build_dicts.append(c.to_dict())
        elif hasattr(c, "__dict__"):
            build_dicts.append(c.__dict__)
        else:
            build_dicts.append(c)

    # Score all candidates
    scored = []
    for i, candidate in enumerate(candidates):
        build_dict = build_dicts[i]

        # Base score from model or default
        cid = _get_candidate_id(candidate)
        base_score = model_scores.get(cid, 0)

        # Add novelty bonus
        if config.novelty_weight > 0 and archive:
            archive_list = list(archive.values())
            novelty = compute_novelty(build_dict, archive_list)
            base_score += novelty * config.novelty_weight

        # Add curiosity bonus
        if config.curiosity_weight > 0 and curiosity:
            base_score = curiosity.score_candidate(candidate, base_score)

        scored.append((candidate, base_score))

    # Sort by score
    scored.sort(key=lambda x: x[1], reverse=True)

    # Epsilon-greedy selection
    n_random = max(1, int(len(scored) * epsilon))
    random.shuffle(scored)
    random_candidates = [c for c, _ in scored[:n_random]]
    exploit_candidates = [c for c, _ in scored[n_random:]]

    # Pareto filter if enabled
    if config.use_pareto and exploit_candidates:
        exploit_candidates = select_pareto_frontier(exploit_candidates)

    # Combine: random first (for exploration), then exploitation
    return random_candidates + exploit_candidates
