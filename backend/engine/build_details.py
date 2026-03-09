from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping
from xml.etree import ElementTree as ET

from backend.engine.genome import GenomeV0
from backend.engine.items.templates import ItemTemplatePlan
from backend.engine.passives.builder import PassiveTreePlan
from backend.engine.skills.catalog import GemPlan
from backend.engine.sockets.planner import SocketPlan

_RESIST_KEYS = ("fire", "cold", "lightning", "chaos")
_ATTRIBUTE_KEYS = ("strength", "dexterity", "intelligence")


def _normalize_text(value: Any, default: str = "unknown") -> str:
    if isinstance(value, str):
        candidate = value.strip()
        if candidate:
            return candidate
    return default


def _extract_item_name(raw_text: str | None) -> str:
    if not raw_text:
        return "unknown"
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        return "unknown"
    if len(lines) >= 3 and lines[0].lower().startswith("rarity"):
        return lines[1]
    return lines[0]


def _to_json_compatible(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_json_compatible(item) for key, item in value.items()}
    return value


def _aggregate_slot_template_fields(
    slot_templates: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    resistances = {key: 0 for key in _RESIST_KEYS}
    attributes = {key: 0 for key in _ATTRIBUTE_KEYS}
    stats = {"life": 0, "energy_shield": 0}

    for template in slot_templates:
        contributions = template.get("contributions")
        if not isinstance(contributions, dict):
            continue
        for key in _RESIST_KEYS:
            value = contributions.get(key)
            if isinstance(value, int):
                resistances[key] += value
        for key in _ATTRIBUTE_KEYS:
            value = contributions.get(key)
            if isinstance(value, int):
                attributes[key] += value
        for key in ("life", "energy_shield"):
            value = contributions.get(key)
            if isinstance(value, int):
                stats[key] += value

    stats["ehp"] = stats["life"] + stats["energy_shield"]
    return resistances, attributes, stats


def _estimate_reservation(
    groups: list[dict[str, Any]], attributes: Mapping[str, int]
) -> tuple[int, int]:
    reservation = 0
    for group in groups:
        if not isinstance(group, dict):
            continue
        gems = group.get("gems")
        gem_count = len(gems) if isinstance(gems, list) else 0
        group_type = _normalize_text(group.get("group_type"), "unknown")
        if group_type in {"aura", "support", "utility"}:
            reservation += 50 * gem_count
        elif group_type in {"guard", "movement"}:
            reservation += 20 * gem_count
    total_mana = 100 + int(attributes.get("intelligence", 0) / 2)
    return reservation, total_mana


def build_details_from_generation(
    *,
    genome: GenomeV0,
    gem_plan: GemPlan,
    passive_plan: PassiveTreePlan,
    socket_plan: SocketPlan,
    template_plan: ItemTemplatePlan,
    profile_id: str | None = None,
) -> dict[str, Any]:
    groups = []
    for group in gem_plan.groups:
        groups.append(
            {
                "id": group.id,
                "name": group.name,
                "group_type": group.group_type,
                "gems": list(group.gems),
            }
        )

    slot_templates = []
    for template in template_plan.templates:
        slot_templates.append(
            {
                "slot_id": template.slot_id,
                "base_type": template.base_type,
                "archetype_priorities": list(template.archetype_priorities),
                "contributions": asdict(template.contributions),
                "adjustable": template.adjustable,
            }
        )

    resistances, attributes, stats = _aggregate_slot_template_fields(slot_templates)
    reservation, total_mana = _estimate_reservation(groups, attributes)

    return {
        "schema_version": 1,
        "source": "generator_plan",
        "identity": {
            "class": genome.class_name,
            "ascendancy": genome.ascendancy,
            "main_skill": genome.main_skill_package,
            "profile_id": profile_id or genome.profile_id,
            "defense_archetype": genome.defense_archetype,
            "budget_tier": genome.budget_tier,
            "sources": {
                "class": "genome",
                "ascendancy": "genome",
                "main_skill": "genome",
            },
        },
        "items": {
            "slot_templates": slot_templates,
            "repair_report": asdict(template_plan.repair_report),
        },
        "resistances": resistances,
        "attributes": attributes,
        "stats": stats,
        "reservation": reservation,
        "total_mana": total_mana,
        "passives": {
            "node_ids": [str(node_id) for node_id in passive_plan.node_ids],
            "required_targets": [str(target) for target in passive_plan.required_targets],
            "node_details": [
                {
                    "id": str(node.id),
                    "kind": str(node.kind),
                    "pob_id": str(node.pob_id) if node.pob_id is not None else None,
                }
                for node in passive_plan.nodes
            ],
            "keystone_ids": [
                str(node.id)
                for node in passive_plan.nodes
                if str(node.kind).strip().lower() == "keystone"
            ],
            "mastery_ids": [
                str(node.id)
                for node in passive_plan.nodes
                if str(node.kind).strip().lower() == "mastery"
            ],
        },
        "gems": {
            "groups": groups,
            "full_dps_group_id": gem_plan.full_dps_group_id,
            "socket_plan": {
                "main_group_id": socket_plan.main_group_id,
                "main_slot_id": socket_plan.main_slot_id,
                "main_link_requirement": socket_plan.main_link_requirement,
                "assignments": [asdict(item) for item in socket_plan.assignments],
                "slots": [asdict(item) for item in socket_plan.slots],
                "hints": [asdict(item) for item in socket_plan.hints],
                "issues": [_to_json_compatible(item) for item in socket_plan.issues],
            },
        },
        "exports": {
            "xml_available": True,
            "code_available": True,
            "share_code_available": True,
        },
    }


def _parse_import_xml(
    xml_payload: str | None, code_payload: str | None
) -> tuple[str | None, ET.Element | None, str]:
    if isinstance(xml_payload, str) and xml_payload.strip().startswith("<"):
        payload = xml_payload.strip()
        try:
            return payload, ET.fromstring(payload), "xml"
        except ET.ParseError:
            return payload, None, "xml"

    if isinstance(code_payload, str) and code_payload.strip().startswith("<"):
        payload = code_payload.strip()
        try:
            return payload, ET.fromstring(payload), "xml_in_code"
        except ET.ParseError:
            return payload, None, "xml_in_code"

    return None, None, "share_code"


def _extract_import_items(root: ET.Element | None) -> list[dict[str, Any]]:
    if root is None:
        return []
    items = []
    for index, node in enumerate(root.findall(".//Item"), start=1):
        items.append(
            {
                "index": index,
                "slot": _normalize_text(node.attrib.get("slot"), "unknown"),
                "name": _extract_item_name(node.text),
            }
        )
    return items


def _extract_import_gems(root: ET.Element | None) -> tuple[list[dict[str, Any]], str | None]:
    if root is None:
        return [], None
    groups: list[dict[str, Any]] = []
    inferred_main: str | None = None
    for index, skill_node in enumerate(root.findall(".//Skill"), start=1):
        gems: list[dict[str, Any]] = []
        for gem in skill_node.findall(".//Gem"):
            gem_name = _normalize_text(
                gem.attrib.get("nameSpec") or gem.attrib.get("skillId") or gem.attrib.get("name"),
                "unknown",
            )
            gems.append(
                {
                    "name": gem_name,
                    "level": _normalize_text(gem.attrib.get("level"), "unknown"),
                    "quality": _normalize_text(gem.attrib.get("quality"), "unknown"),
                    "enabled": _normalize_text(gem.attrib.get("enabled"), "true"),
                }
            )
            if inferred_main is None and gem_name != "unknown":
                inferred_main = gem_name

        groups.append(
            {
                "id": f"skill_group_{index}",
                "slot": _normalize_text(skill_node.attrib.get("slot"), "unknown"),
                "link_count": len(gems),
                "gems": gems,
            }
        )
    return groups, inferred_main


def _extract_import_passives(root: ET.Element | None) -> list[str]:
    if root is None:
        return []
    tree_node = root.find(".//Tree")
    if tree_node is None:
        return []

    spec_nodes = tree_node.findall("Spec")
    if not spec_nodes:
        return []

    active = tree_node.attrib.get("activeSpec")
    selected = spec_nodes[0]
    if active:
        for spec in spec_nodes:
            if spec.attrib.get("title") == active:
                selected = spec
                break

    raw_nodes = _normalize_text(selected.attrib.get("nodes"), "")
    if not raw_nodes:
        return []
    return [node.strip() for node in raw_nodes.split(",") if node.strip()]


def build_details_from_import(
    *,
    xml_payload: str | None,
    code_payload: str | None,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    xml_text, root, source = _parse_import_xml(xml_payload, code_payload)
    metadata = metadata or {}

    build_node = root.find(".//Build") if root is not None else None
    class_name = _normalize_text(
        build_node.attrib.get("className") if build_node is not None else None
    )
    ascendancy = _normalize_text(
        build_node.attrib.get("ascendClassName") if build_node is not None else None
    )

    groups, inferred_main = _extract_import_gems(root)
    node_ids = _extract_import_passives(root)
    items = _extract_import_items(root)

    metadata_class = _normalize_text(metadata.get("class"), "unknown")
    metadata_asc = _normalize_text(metadata.get("ascendancy"), "unknown")
    metadata_main = _normalize_text(metadata.get("main_skill"), "unknown")

    resolved_class = class_name if class_name != "unknown" else metadata_class
    resolved_asc = ascendancy if ascendancy != "unknown" else metadata_asc
    resolved_main = metadata_main
    if resolved_main == "unknown" and inferred_main is not None:
        resolved_main = inferred_main

    return {
        "schema_version": 1,
        "source": source,
        "identity": {
            "class": resolved_class,
            "ascendancy": resolved_asc,
            "main_skill": resolved_main,
            "sources": {
                "class": "xml" if class_name != "unknown" else "metadata",
                "ascendancy": "xml" if ascendancy != "unknown" else "metadata",
                "main_skill": "xml_gems"
                if (metadata_main == "unknown" and inferred_main)
                else "metadata",
            },
        },
        "items": {
            "items": items,
        },
        "passives": {
            "node_ids": node_ids,
        },
        "gems": {
            "groups": groups,
            "gem_link_count": max((group.get("link_count", 0) for group in groups), default=0),
        },
        "exports": {
            "xml_available": bool(xml_text and xml_text.startswith("<")),
            "code_available": bool(code_payload),
            "share_code_available": bool(
                code_payload and not str(code_payload).strip().startswith("<")
            ),
        },
    }
