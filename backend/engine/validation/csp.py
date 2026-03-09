from typing import Any


class BuildCSP:
    """
    Constraint Satisfaction Problem (CSP) validator for hard structural constraints.
    Checks for validity without calling PoB (fast ~10ms).
    """

    def validate(self, build_data: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validates the build against hard structural constraints.

        Args:
            build_data: Dictionary containing build information:
                - main_skill: Dict with 'links' (int)
                - items: List of Dicts with 'requirements' (Dict[str, int]), 'sockets' (List[str])
                - attributes: Dict[str, int] (str, dex, int)
                - reservation: float (total mana reserved)
                - total_mana: float
                - passive_tree: List of node IDs, and a connectivity map or root access.

        Returns:
            (is_valid, list_of_errors)
        """
        errors: list[str] = []

        # 1. Main skill has 6 links
        if not self._check_six_links(build_data):
            errors.append("Main skill must have 6 links")

        # 2. Socket colors match attribute requirements
        socket_errors = self._check_socket_colors(build_data)
        errors.extend(socket_errors)

        # 3. Can equip all items
        equip_errors = self._check_item_requirements(build_data)
        errors.extend(equip_errors)

        # 4. Reservation doesn't exceed mana
        if not self._check_mana_reservation(build_data):
            errors.append("Mana reservation exceeds total mana")

        # 5. Passive tree is connected
        if not self._check_passive_connectivity(build_data):
            errors.append("Passive tree is not connected to a starting node")

        return len(errors) == 0, errors

    def _check_six_links(self, build_data: dict[str, Any]) -> bool:
        main_skill = build_data.get("main_skill", {})
        return main_skill.get("links", 0) == 6

    def _check_socket_colors(self, build_data: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        items = build_data.get("items", [])
        color_map = {"R": "str", "G": "dex", "B": "int"}

        for i, item in enumerate(items):
            sockets = item.get("sockets", [])
            reqs = item.get("requirements", {})

            counts: dict[str, int] = {"R": 0, "G": 0, "B": 0, "W": 0}
            for s in sockets:
                counts[s] = counts.get(s, 0) + 1

            for color, attr in color_map.items():
                if reqs.get(attr, 0) == 0 and counts[color] > 3:
                    errors.append(
                        f"Item {i} has too many {color} sockets for its attribute requirements"
                    )
        return errors

    def _check_item_requirements(self, build_data: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        attributes = build_data.get("attributes", {"str": 0, "dex": 0, "int": 0})
        items = build_data.get("items", [])

        for i, item in enumerate(items):
            reqs = item.get("requirements", {})
            for attr, value in reqs.items():
                attr_key = attr.lower()
                if attributes.get(attr_key, 0) < value:
                    errors.append(
                        (
                            f"Item {i} requires {value} {attr}, but build only has "
                            f"{attributes.get(attr_key)}"
                        )
                    )
        return errors

    def _check_mana_reservation(self, build_data: dict[str, Any]) -> bool:
        reservation = build_data.get("reservation", 0)
        total_mana = build_data.get("total_mana", 0)
        return reservation <= total_mana

    def _check_passive_connectivity(self, build_data: dict[str, Any]) -> bool:
        """
        Checks if all allocated passive nodes are connected to a starting node.
        Expects 'passive_tree' to have 'allocated' (List[int])
        and 'adjacencies' (Dict[int, List[int]]).
        'start_nodes' (List[int]) should also be provided.
        """
        tree_data = build_data.get("passive_tree", {})
        allocated = set(tree_data.get("allocated", []))
        adjacencies = tree_data.get("adjacencies", {})
        start_nodes = set(tree_data.get("start_nodes", []))

        if not allocated:
            return True

        # Find which start node is used
        reachable = allocated.intersection(start_nodes)
        if not reachable:
            return False

        # BFS/DFS to find all reachable allocated nodes
        stack = list(reachable)
        visited = set(reachable)

        while stack:
            node = stack.pop()
            for neighbor in adjacencies.get(node, []):
                if neighbor in allocated and neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        return len(visited) == len(allocated)
