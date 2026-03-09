from typing import Any

from backend.engine.items.templates import RESIST_BASELINE


class ResistanceRepair:
    """Add resistance affixes if below cap."""

    def needs_repair(self, build: dict[str, Any]) -> bool:
        resists = build.get("resistances", {})
        for key, cap in RESIST_BASELINE.items():
            if resists.get(key, 0) < cap:
                return True
        return False

    def apply(self, build: dict[str, Any]) -> dict[str, Any]:
        resists = build.get("resistances", {})
        items_data = build.get("items", [])

        if isinstance(items_data, dict):
            item_list = items_data.get("slot_templates", [])
        else:
            item_list = items_data

        adjustable_items = [item for item in item_list if item.get("adjustable", False)]
        if not adjustable_items:
            return build

        for key, cap in RESIST_BASELINE.items():
            current = resists.get(key, 0)
            if current < cap:
                deficit = cap - current
                # Distribute deficit among adjustable items
                per_item = (deficit + len(adjustable_items) - 1) // len(adjustable_items)
                for item in adjustable_items:
                    contribs = item.get("contributions", {})
                    # Handle both dict and dataclass-like objects
                    if isinstance(contribs, dict):
                        contribs[key] = contribs.get(key, 0) + per_item
                        item["contributions"] = contribs
                    else:
                        current_val = getattr(contribs, key, 0)
                        setattr(contribs, key, current_val + per_item)

                    resists[key] = resists.get(key, 0) + per_item
                    if resists[key] >= cap:
                        break

        build["resistances"] = resists
        return build


class LifeRepair:
    """Add life rolls if eHP < threshold."""

    def __init__(self, ehp_threshold: int = 3000):
        self.ehp_threshold = ehp_threshold

    def needs_repair(self, build: dict[str, Any]) -> bool:
        stats = build.get("stats", {})
        ehp = stats.get("ehp", 0)
        return ehp < self.ehp_threshold

    def apply(self, build: dict[str, Any]) -> dict[str, Any]:
        items_data = build.get("items", [])

        if isinstance(items_data, dict):
            item_list = items_data.get("slot_templates", [])
        else:
            item_list = items_data

        adjustable_items = [item for item in item_list if item.get("adjustable", False)]
        if not adjustable_items:
            return build

        # Add life to each adjustable item
        added_life = 0
        for item in adjustable_items:
            contribs = item.get("contributions", {})
            increment = 50
            if isinstance(contribs, dict):
                contribs["life"] = contribs.get("life", 0) + increment
                item["contributions"] = contribs
            else:
                current_val = getattr(contribs, "life", 0)
                contribs.life = current_val + increment
            added_life += increment

        # Update stats (simplified)
        stats = build.get("stats", {})
        stats["life"] = stats.get("life", 0) + added_life
        stats["ehp"] = stats.get("ehp", 0) + (added_life * 3)  # Rough estimate: 1 life -> 3 eHP
        build["stats"] = stats
        return build


class AttributeRepair:
    """Add attribute rolls if can't equip items."""

    def needs_repair(self, build: dict[str, Any]) -> bool:
        attributes = build.get("attributes", {"strength": 0, "dexterity": 0, "intelligence": 0})
        items_data = build.get("items", [])

        if isinstance(items_data, dict):
            item_list = items_data.get("slot_templates", [])
        else:
            item_list = items_data

        for item in item_list:
            reqs = item.get("requirements", {})
            for attr, value in reqs.items():
                if attributes.get(attr.lower(), 0) < value:
                    return True
        return False

    def apply(self, build: dict[str, Any]) -> dict[str, Any]:
        attributes = build.get("attributes", {"strength": 0, "dexterity": 0, "intelligence": 0})
        items_data = build.get("items", [])

        if isinstance(items_data, dict):
            item_list = items_data.get("slot_templates", [])
        else:
            item_list = items_data

        adjustable_items = [item for item in item_list if item.get("adjustable", False)]
        if not adjustable_items:
            return build

        for item in item_list:
            reqs = item.get("requirements", {})
            for attr, value in reqs.items():
                attr_lower = attr.lower()
                current_attr = attributes.get(attr_lower, 0)
                if current_attr < value:
                    deficit = value - current_attr
                    # Add to first adjustable item
                    target = adjustable_items[0]
                    contribs = target.get("contributions", {})
                    if isinstance(contribs, dict):
                        contribs[attr_lower] = contribs.get(attr_lower, 0) + deficit
                        target["contributions"] = contribs
                    else:
                        current_val = getattr(contribs, attr_lower, 0)
                        setattr(contribs, attr_lower, current_val + deficit)
                    attributes[attr_lower] = current_attr + deficit

        build["attributes"] = attributes
        return build


class ReservationRepair:
    """Reduce auras if over-reserved."""

    def needs_repair(self, build: dict[str, Any]) -> bool:
        reservation = build.get("reservation", 0)
        total_mana = build.get("total_mana", 0)
        return reservation > total_mana

    def apply(self, build: dict[str, Any]) -> dict[str, Any]:
        reservation = build.get("reservation", 0)
        total_mana = build.get("total_mana", 0)

        if reservation > total_mana:
            # Reduce reservation to total_mana
            build["reservation"] = total_mana

            # If we have gem groups, try to remove one utility group
            gems = build.get("gems", {})
            groups = gems.get("groups", [])
            if groups:
                main_group_id = gems.get("full_dps_group_id")
                # Find a non-main group to remove
                for i in range(len(groups) - 1, -1, -1):
                    if groups[i].get("id") != main_group_id:
                        groups.pop(i)
                        break
                gems["groups"] = groups
                build["gems"] = gems

        return build
