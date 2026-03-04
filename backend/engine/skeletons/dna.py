from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BuildDNA:
    """
    Full build DNA representing a complete character state.
    This is the expanded form of a Skeleton.
    """

    skeleton_id: str
    class_name: str
    ascendancy: str
    main_skill: str

    # Passive tree state
    tree_nodes: list[int] = field(default_factory=list)

    # Items: list of item texts (PoB format)
    items: list[str] = field(default_factory=list)

    # Gems: list of gem groups
    # Each group is a dict with: { "slot": str, "gems": [ { "name": str, "level": int, "quality": int, "enabled": bool } ] }
    gem_groups: list[dict[str, Any]] = field(default_factory=list)

    # Configuration/Metadata
    defense_layer: str = ""
    budget_tier: str = ""
    target_gates: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skeleton_id": self.skeleton_id,
            "class_name": self.class_name,
            "ascendancy": self.ascendancy,
            "main_skill": self.main_skill,
            "tree_nodes": list(self.tree_nodes),
            "items": list(self.items),
            "gem_groups": list(self.gem_groups),
            "defense_layer": self.defense_layer,
            "budget_tier": self.budget_tier,
            "target_gates": dict(self.target_gates),
        }
