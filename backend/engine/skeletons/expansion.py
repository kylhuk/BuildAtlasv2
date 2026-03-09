from __future__ import annotations

from typing import Any

from .dna import BuildDNA
from .loader import load_skeleton
from .schema import Skeleton


def expand_skeleton(skeleton_id: str) -> BuildDNA:
    """
    Expand a skeleton into a full BuildDNA.

    This fills in items, passive tree, and gems based on the skeleton's requirements.
    """
    skeleton = load_skeleton(skeleton_id)

    # 1. Passive Tree
    tree_nodes = _load_tree_path(skeleton.tree_path)

    # 2. Items (Required Uniques)
    items = _generate_items(skeleton.required_uniques)

    # 3. Gems (Main Skill + Auras)
    gem_groups = _generate_gem_groups(skeleton)

    return BuildDNA(
        skeleton_id=skeleton.skeleton_id,
        class_name=skeleton.class_name,
        ascendancy=skeleton.ascendancy,
        main_skill=skeleton.main_skill,
        tree_nodes=tree_nodes,
        items=items,
        gem_groups=gem_groups,
        defense_layer=skeleton.defense_layer,
        budget_tier=skeleton.budget_tier,
        target_gates=skeleton.target_gates,
    )


def _load_tree_path(tree_path: str) -> list[int]:
    """
    Load passive tree nodes from a tree path.
    For now, we assume tree_path is a comma-separated list of node IDs or a path to a file.
    If it's a file path, we read it.
    """
    # Simple implementation: if it contains commas, treat as list.
    # Otherwise, check if it's a file.
    if "," in tree_path:
        try:
            return [int(x.strip()) for x in tree_path.split(",") if x.strip()]
        except ValueError:
            pass

    # Try to treat as file path relative to data/trees/
    # (This is a placeholder for actual tree loading logic)
    return []


def _generate_items(required_uniques: list[str]) -> list[str]:
    """
    Generate PoB item texts for required uniques.
    """
    items = []
    for unique_name in required_uniques:
        # Minimal PoB item format for a unique
        item_text = f"Rarity: UNIQUE\n{unique_name}"
        items.append(item_text)
    return items


def _generate_gem_groups(skeleton: Skeleton) -> list[dict[str, Any]]:
    """
    Generate gem groups for main skill and auras.
    """
    groups: list[dict[str, Any]] = []

    # Main Skill Group
    main_gems: list[dict[str, Any]] = [
        {"name": skeleton.main_skill, "level": 20, "quality": 20, "enabled": True}
    ]
    for support in skeleton.skill_links:
        main_gems.append({"name": support, "level": 20, "quality": 20, "enabled": True})

    groups.append({"slot": "Body Armour", "gems": main_gems})

    # Aura Package
    if skeleton.aura_package:
        aura_gems: list[dict[str, Any]] = []
        for aura in skeleton.aura_package:
            aura_gems.append({"name": aura, "level": 20, "quality": 0, "enabled": True})

        groups.append({"slot": "Weapon 1", "gems": aura_gems})

    return groups
