from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.engine.genome import BUDGET_TIERS, CLASS_ASCENDANCIES, DEFENSE_ARCHETYPES


@dataclass(frozen=True)
class Skeleton:
    skeleton_id: str
    class_name: str
    ascendancy: str
    main_skill: str
    skill_links: list[str]
    aura_package: list[str]
    defense_layer: str
    budget_tier: str
    target_gates: dict[str, Any]
    required_uniques: list[str]
    tree_path: str

    def validate(self) -> None:
        """Validate the skeleton data."""
        if not self.skeleton_id:
            raise ValueError("skeleton_id is required")

        if self.class_name not in CLASS_ASCENDANCIES:
            raise ValueError(f"invalid class_name: {self.class_name}")

        if self.ascendancy not in CLASS_ASCENDANCIES[self.class_name]:
            raise ValueError(f"invalid ascendancy {self.ascendancy} for class {self.class_name}")

        if not self.main_skill:
            raise ValueError("main_skill is required")

        if not isinstance(self.skill_links, (list, tuple)):
            raise ValueError("skill_links must be a list or tuple")

        if not isinstance(self.aura_package, (list, tuple)):
            raise ValueError("aura_package must be a list or tuple")

        if self.defense_layer not in DEFENSE_ARCHETYPES:
            raise ValueError(f"invalid defense_layer: {self.defense_layer}")

        if self.budget_tier not in BUDGET_TIERS:
            raise ValueError(f"invalid budget_tier: {self.budget_tier}")

        if not isinstance(self.target_gates, dict):
            raise ValueError("target_gates must be a dict")

        if not isinstance(self.required_uniques, (list, tuple)):
            raise ValueError("required_uniques must be a list or tuple")

        if not self.tree_path:
            raise ValueError("tree_path is required")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Skeleton:
        """Create a Skeleton from a dictionary."""
        return cls(
            skeleton_id=str(data["skeleton_id"]),
            class_name=str(data["class_name"]),
            ascendancy=str(data["ascendancy"]),
            main_skill=str(data["main_skill"]),
            skill_links=list(data.get("skill_links", [])),
            aura_package=list(data.get("aura_package", [])),
            defense_layer=str(data.get("defense_layer", "")),
            budget_tier=str(data.get("budget_tier", "")),
            target_gates=dict(data.get("target_gates", {})),
            required_uniques=list(data.get("required_uniques", [])),
            tree_path=str(data.get("tree_path", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the Skeleton to a dictionary."""
        return {
            "skeleton_id": self.skeleton_id,
            "class_name": self.class_name,
            "ascendancy": self.ascendancy,
            "main_skill": self.main_skill,
            "skill_links": list(self.skill_links),
            "aura_package": list(self.aura_package),
            "defense_layer": self.defense_layer,
            "budget_tier": self.budget_tier,
            "target_gates": dict(self.target_gates),
            "required_uniques": list(self.required_uniques),
            "tree_path": self.tree_path,
        }
