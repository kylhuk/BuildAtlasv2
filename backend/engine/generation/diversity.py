"""
Diversity selection for ML training.

Implements MAP-Elites / Quality-Diversity approach to maintain diverse
build archives across different niches (class, damage type, defense type, etc.)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Niche dimensions for PoE builds
NICHE_DIMENSIONS = {
    "class": ["Marauder", "Ranger", "Witch", "Templar", "Shadow", "Scion"],
    "damage_type": ["physical", "chaos", "elemental", "hybrid"],
    "defense_type": ["armor", "evasion", "energy_shield", "hybrid"],
    "main_skill_type": ["spell", "attack", "minion", "totem", "trap", "brand"],
}


@dataclass(frozen=True)
class NicheAssignment:
    """Represents a build's niche assignment."""

    class_name: str
    damage_type: str
    defense_type: str
    skill_type: str

    def __str__(self) -> str:
        return f"{self.class_name}/{self.damage_type}/{self.defense_type}/{self.skill_type}"

    def to_dict(self) -> dict[str, str]:
        return {
            "class": self.class_name,
            "damage_type": self.damage_type,
            "defense_type": self.defense_type,
            "skill_type": self.skill_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "NicheAssignment":
        return cls(
            class_name=data.get("class", "unknown"),
            damage_type=data.get("damage_type", "hybrid"),
            defense_type=data.get("defense_type", "hybrid"),
            skill_type=data.get("skill_type", "unknown"),
        )


def assign_niche(build: dict[str, Any]) -> NicheAssignment:
    """Assign a build to a niche based on its characteristics.

    Args:
        build: Build dictionary with genome and metrics

    Returns:
        NicheAssignment representing the build's niche
    """
    # Extract class
    class_name = build.get("class") or build.get("class_name", "unknown")

    # Extract or infer damage type
    damage_type = _infer_damage_type(build)

    # Extract defense type
    defense_type = build.get("defense_archetype", "hybrid")
    if defense_type not in NICHE_DIMENSIONS["defense_type"]:
        defense_type = "hybrid"

    # Extract or infer skill type
    skill_type = _infer_skill_type(build)

    return NicheAssignment(
        class_name=class_name,
        damage_type=damage_type,
        defense_type=defense_type,
        skill_type=skill_type,
    )


def _infer_damage_type(build: dict[str, Any]) -> str:
    """Infer the primary damage type from build characteristics."""
    # Check genome for explicit damage type
    genome = build.get("genome", {})
    if genome:
        damage_type = genome.get("damage_type") or genome.get("main_skill_package", "")
        if damage_type:
            return _categorize_damage_type(damage_type)

    # Check metrics for elemental vs physical bias
    metrics = build.get("metrics", {})
    if metrics:
        # This is a simplified heuristic
        fire = metrics.get("fire_dps", 0) or 0
        cold = metrics.get("cold_dps", 0) or 0
        lightning = metrics.get("lightning_dps", 0) or 0
        chaos = metrics.get("chaos_dps", 0) or 0
        physical = metrics.get("physical_dps", 0) or 0

        total_ele = fire + cold + lightning
        if chaos > total_ele * 0.5:
            return "chaos"
        elif total_ele > physical:
            return "elemental"
        elif physical > total_ele:
            return "physical"

    return "hybrid"


def _categorize_damage_type(damage_type_str: str) -> str:
    """Categorize a damage type string into one of our categories."""
    dmg = damage_type_str.lower()
    if any(x in dmg for x in ["fire", "cold", "lightning", "elemental", "pyre", "arcanist"]):
        return "elemental"
    if any(x in dmg for x in ["chaos", "poison", "blight", "contagion"]):
        return "chaos"
    if any(x in dmg for x in ["physical", "bleed", "impale", "melee", "attack"]):
        return "physical"
    return "hybrid"


def _infer_skill_type(build: dict[str, Any]) -> str:
    """Infer the primary skill type from build characteristics."""
    # Check genome for main skill
    genome = build.get("genome", {})
    if genome:
        skill = genome.get("main_skill_package", "").lower()
        if skill:
            return _categorize_skill_type(skill)

    # Check tags
    tags = build.get("tags", [])
    if isinstance(tags, list):
        for tag in tags:
            if "minion" in tag.lower():
                return "minion"
            if "totem" in tag.lower():
                return "totem"
            if "trap" in tag.lower() or "mine" in tag.lower():
                return "trap"
            if "brand" in tag.lower():
                return "brand"

    return "spell"  # Default assumption


def _categorize_skill_type(skill_str: str) -> str:
    """Categorize a skill string into one of our categories."""
    skill = skill_str.lower()
    if any(x in skill for x in ["minion", "summon", "raise", "spectre", "golem"]):
        return "minion"
    if any(x in skill for x in ["totem", "ballista"]):
        return "totem"
    if any(x in skill for x in ["trap", "mine", "seismic", "exsanguinate"]):
        return "trap"
    if any(x in skill for x in ["brand", "armageddon", "winterball"]):
        return "brand"
    if any(x in skill for x in ["attack", "strike", "shoot", "bow", "melee"]):
        return "attack"
    return "spell"


def select_diverse_elites(
    candidates: list[Any],
    elite_count: int,
    archive: dict[NicheAssignment, dict[str, Any]] | None = None,
) -> list[tuple[Any, float]]:
    """Select elites that cover diverse niches.

    This implements MAP-Elites style selection:
    - Try to fill underrepresented niches first
    - Then select best overall

    Args:
        candidates: Available candidates with .score attribute
        elite_count: How many elites to select
        archive: Existing archive of best per niche

    Returns:
        List of (candidate, score) tuples, diversified across niches
    """
    if not candidates:
        return []

    if archive is None:
        archive = {}

    # Assign each candidate to a niche
    niche_candidates: dict[NicheAssignment, list[tuple[Any, float]]] = {}

    for candidate in candidates:
        # Handle both Candidate objects and dicts
        if hasattr(candidate, "to_dict"):
            c_dict = candidate.to_dict()
        elif hasattr(candidate, "__dict__"):
            c_dict = candidate.__dict__
        else:
            c_dict = candidate

        # Get score
        if hasattr(candidate, "score"):
            score = candidate.score
        elif isinstance(candidate, dict):
            score = candidate.get("score", 0) or candidate.get("full_dps", 0) or 0
        else:
            score = 0

        niche = assign_niche(c_dict)

        if niche not in niche_candidates:
            niche_candidates[niche] = []
        niche_candidates[niche].append((candidate, score))

    selected: list[tuple[Any, float]] = []
    selected_niches: set[NicheAssignment] = set()

    # Phase 1: First, fill empty niches from archive (exploitation of existing best)
    for niche in archive:
        if len(selected) >= elite_count:
            break
        if niche not in niche_candidates:
            # Use archive's best for this niche
            archive_build = archive[niche]
            score = archive_build.get("full_dps", 0) or archive_build.get("score", 0) or 0
            # Create a mock candidate from archive
            selected.append((archive_build, score))
            selected_niches.add(niche)

    # Phase 2: Select best from each niche that has candidates
    for niche, scored_candidates in niche_candidates.items():
        if len(selected) >= elite_count:
            break
        if niche in selected_niches:
            continue
        # Select best from this niche
        best = max(scored_candidates, key=lambda x: x[1])
        selected.append(best)
        selected_niches.add(niche)

    # Phase 3: If we need more, fill with highest scoring remaining
    remaining = []
    for niche, scored_candidates in niche_candidates.items():
        if niche not in selected_niches:
            for c, s in scored_candidates:
                remaining.append((c, s))

    # Sort by score descending
    remaining.sort(key=lambda x: x[1], reverse=True)

    while len(selected) < elite_count and remaining:
        candidate, score = remaining.pop(0)
        # Get niche to check
        if hasattr(candidate, "to_dict"):
            c_dict = candidate.to_dict()
        elif hasattr(candidate, "__dict__"):
            c_dict = candidate.__dict__
        else:
            c_dict = candidate
        niche = assign_niche(c_dict)

        if niche not in selected_niches:
            selected.append((candidate, score))
            selected_niches.add(niche)

    return selected[:elite_count]


class DiversityArchive:
    """MAP-Elites style archive maintaining best per niche."""

    def __init__(
        self,
        niche_dimensions: dict[str, list[str]] | None = None,
    ):
        self.niche_dims = niche_dimensions or NICHE_DIMENSIONS
        self.archive: dict[NicheAssignment, dict[str, Any]] = {}

    def update(self, build: dict[str, Any]) -> bool:
        """Add build to archive if it improves its niche.

        Args:
            build: Build dictionary with genome and metrics

        Returns:
            True if build was added to archive (new or better)
        """
        niche = assign_niche(build)
        score = build.get("full_dps", 0) or build.get("score", 0) or 0

        existing = self.archive.get(niche)
        if existing is None:
            self.archive[niche] = build
            return True

        existing_score = existing.get("full_dps", 0) or existing.get("score", 0) or 0
        if score > existing_score:
            self.archive[niche] = build
            return True
        return False

    def get_diverse_sample(self, n: int) -> list[dict[str, Any]]:
        """Get n builds, one per niche if possible.

        Args:
            n: Number of builds to return

        Returns:
            List of build dictionaries
        """
        niches = list(self.archive.keys())
        if not niches:
            return []

        # Shuffle for variety
        random.shuffle(niches)

        result = []
        for niche in niches:
            if len(result) >= n:
                break
            result.append(self.archive[niche])

        return result

    def get_niche_count(self) -> int:
        """Get number of filled niches."""
        return len(self.archive)

    def get_niches(self) -> list[NicheAssignment]:
        """Get list of all filled niches."""
        return list(self.archive.keys())

    def get_best_for_niche(self, niche: NicheAssignment) -> dict[str, Any] | None:
        """Get the best build for a specific niche."""
        return self.archive.get(niche)

    def save(self, path: Path) -> None:
        """Save archive to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-serializable format
        serialized = {}
        for niche, build in self.archive.items():
            serialized[str(niche)] = build

        path.write_text(json.dumps(serialized, indent=2, ensure_ascii=False))

    def load(self, path: Path) -> None:
        """Load archive from disk."""
        if not path.exists():
            return

        data = json.loads(path.read_text())
        self.archive = {}
        for niche_str, build in data.items():
            # Parse niche from string
            parts = niche_str.split("/")
            if len(parts) == 4:
                niche = NicheAssignment(
                    class_name=parts[0],
                    damage_type=parts[1],
                    defense_type=parts[2],
                    skill_type=parts[3],
                )
                self.archive[niche] = build

    def __len__(self) -> int:
        return len(self.archive)

    def __repr__(self) -> str:
        return f"DiversityArchive(filled_niches={len(self.archive)})"


def compute_diversity_score(
    builds: list[dict[str, Any]],
) -> float:
    """Compute a diversity score for a list of builds.

    Uses entropy-like measure based on niche distribution.

    Args:
        builds: List of build dictionaries

    Returns:
        Diversity score (higher = more diverse)
    """
    if not builds:
        return 0.0

    # Assign niches
    niches = [assign_niche(b) for b in builds]

    # Count unique niches
    unique_niches = len(set(niches))

    # Calculate coverage
    max_possible_niches = (
        len(NICHE_DIMENSIONS["class"])
        * len(NICHE_DIMENSIONS["damage_type"])
        * len(NICHE_DIMENSIONS["defense_type"])
        * len(NICHE_DIMENSIONS["main_skill_type"])
    )

    coverage = unique_niches / max_possible_niches if max_possible_niches > 0 else 0

    return coverage
