"""Pricing artifact cost extraction and calculator helpers."""

from __future__ import annotations

import gzip
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Tuple

from backend.engine.artifacts import store as artifact_store


def _normalize_lookup(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    return cleaned.casefold()


def _local_tag(element: ET.Element) -> str:
    tag = element.tag
    if "}" in tag:
        return tag.split("}", 1)[1].casefold()
    return tag.casefold()


def _text_from_children(element: ET.Element, names: Iterable[str]) -> str | None:
    for child in element:
        if _local_tag(child) in names and child.text:
            return child.text.strip()
    return None


def _parse_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _read_build_xml_text(build_id: str, base_path: Path | str | None = None) -> str:
    paths = artifact_store.artifact_paths(build_id, base_path)
    if paths.build_xml.exists():
        with gzip.open(paths.build_xml, "rb") as handle:
            return handle.read().decode("utf-8")
    if paths.code.exists():
        return paths.code.read_text(encoding="utf-8")
    raise FileNotFoundError(f"no build artifact found for {build_id}")


@dataclass(frozen=True)
class UniqueItemUsage:
    slot: str
    name: str


@dataclass(frozen=True)
class SkillGemUsage:
    name: str
    level: int | None
    quality: int | None


@dataclass(frozen=True)
class CostExtractionResult:
    unique_items: Tuple[UniqueItemUsage, ...]
    skill_gems: Tuple[SkillGemUsage, ...]


def _extract_unique_items(root: ET.Element) -> Sequence[UniqueItemUsage]:
    results: list[UniqueItemUsage] = []
    for element in root.iter():
        if _local_tag(element) != "item":
            continue
        rarity = _text_from_children(element, {"rarity"})
        if rarity is None or rarity.casefold() != "unique":
            continue
        name = element.get("name") or _text_from_children(element, {"name", "displayname"})
        if not name or not name.strip():
            continue
        slot = element.get("slot") or _text_from_children(element, {"slot", "slotname"})
        normalized_slot = slot.strip() if slot and slot.strip() else "unknown"
        results.append(UniqueItemUsage(slot=normalized_slot, name=name.strip()))
    return results


def _extract_skill_gems(root: ET.Element) -> Sequence[SkillGemUsage]:
    results: list[SkillGemUsage] = []
    for element in root.iter():
        tag = _local_tag(element)
        if tag not in {"skillgem", "gem"}:
            continue
        name = element.get("name") or _text_from_children(element, {"name"})
        if not name or not name.strip():
            continue
        level = _parse_optional_int(element.get("level") or _text_from_children(element, {"level"}))
        quality = _parse_optional_int(
            element.get("quality") or _text_from_children(element, {"quality"})
        )
        results.append(SkillGemUsage(name=name.strip(), level=level, quality=quality))
    return results


def extract_build_cost_requirements(
    build_id: str, base_path: Path | str | None = None
) -> CostExtractionResult:
    xml_text = _read_build_xml_text(build_id, base_path)
    root = ET.fromstring(xml_text)
    return CostExtractionResult(
        unique_items=tuple(_extract_unique_items(root)),
        skill_gems=tuple(_extract_skill_gems(root)),
    )


@dataclass(frozen=True)
class PriceSnapshot:
    snapshot_id: str
    league: str
    timestamp: str
    unique_items: dict[str, float]
    skill_gems: dict[Tuple[str, int | None, int | None], float]


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _find_snapshot_entry(
    index_payload: dict[str, Any],
    snapshot_id: str,
) -> tuple[str, dict[str, Any]]:
    for league, payload in index_payload.get("leagues", {}).items():
        for entry in payload.get("history", []):
            if entry.get("id") == snapshot_id:
                return league, entry
    raise ValueError(f"price snapshot {snapshot_id} not found")


def _build_unique_map(entries: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for entry in entries:
        name = entry.get("name")
        if not name:
            continue
        normalized = _normalize_lookup(str(name))
        if not normalized:
            continue
        value = entry.get("chaos")
        if value is None:
            continue
        try:
            chaos = float(value)
        except (TypeError, ValueError):
            continue
        result[normalized] = chaos
    return result


def _build_skill_gem_map(
    entries: Sequence[Mapping[str, Any]],
) -> dict[Tuple[str, int | None, int | None], float]:
    result: dict[Tuple[str, int | None, int | None], float] = {}
    for entry in entries:
        name = entry.get("name")
        if not name:
            continue
        normalized = _normalize_lookup(str(name))
        if not normalized:
            continue
        value = entry.get("price")
        if value is None:
            continue
        try:
            price = float(value)
        except (TypeError, ValueError):
            continue
        level = _parse_optional_int(entry.get("level"))
        quality = _parse_optional_int(entry.get("quality"))
        result[(normalized, level, quality)] = price
    return result


def load_price_snapshot(snapshot_id: str, data_path: Path | str | None = None) -> PriceSnapshot:
    root = Path(data_path) if data_path is not None else Path.cwd()
    prices_root = root / "prices"
    index_path = prices_root / "index.json"
    index_payload = _load_json(index_path)
    league, entry = _find_snapshot_entry(index_payload, snapshot_id)
    path_value = entry.get("path")
    if not path_value:
        raise ValueError("price snapshot entry missing path")
    snapshot_path = prices_root / path_value
    metadata = _load_json(snapshot_path / "price_snapshot.json")
    files = metadata.get("files", {})
    unique_file = files.get("unique_items")
    gem_file = files.get("skill_gems")
    if not unique_file or not gem_file:
        raise ValueError("price snapshot metadata missing required file references")
    unique_entries = _load_json(snapshot_path / unique_file)
    gem_entries = _load_json(snapshot_path / gem_file)
    timestamp = metadata.get("timestamp", "")
    if not timestamp:
        raise ValueError("price snapshot metadata missing timestamp")
    return PriceSnapshot(
        snapshot_id=snapshot_id,
        league=league,
        timestamp=timestamp,
        unique_items=_build_unique_map(unique_entries),
        skill_gems=_build_skill_gem_map(gem_entries),
    )


@dataclass(frozen=True)
class SlotCostDetail:
    slot: str
    name: str
    cost_chaos: float | None
    matched: bool


@dataclass(frozen=True)
class GemCostDetail:
    name: str
    level: int | None
    quality: int | None
    price: float | None
    matched: bool


@dataclass(frozen=True)
class CostSummary:
    total_cost_chaos: float
    unknown_cost_count: int
    slot_costs: Tuple[SlotCostDetail, ...]
    gem_costs: Tuple[GemCostDetail, ...]


def _lookup_gem_price(usage: SkillGemUsage, snapshot: PriceSnapshot) -> float | None:
    normalized = _normalize_lookup(usage.name)
    if not normalized:
        return None
    candidates = [
        (normalized, usage.level, usage.quality),
        (normalized, usage.level, None),
        (normalized, None, usage.quality),
        (normalized, None, None),
    ]
    for candidate in candidates:
        price = snapshot.skill_gems.get(candidate)
        if price is not None:
            return price
    return None


def calculate_cost_summary(
    unique_items: Sequence[UniqueItemUsage],
    skill_gems: Sequence[SkillGemUsage],
    snapshot: PriceSnapshot,
) -> CostSummary:
    total = 0.0
    unknown = 0
    slot_details: list[SlotCostDetail] = []
    for usage in unique_items:
        normalized = _normalize_lookup(usage.name)
        price = snapshot.unique_items.get(normalized) if normalized else None
        matched = price is not None
        if matched:
            total += price
        else:
            unknown += 1
        slot_details.append(
            SlotCostDetail(slot=usage.slot, name=usage.name, cost_chaos=price, matched=matched)
        )
    gem_details: list[GemCostDetail] = []
    for usage in skill_gems:
        price = _lookup_gem_price(usage, snapshot)
        matched = price is not None
        if matched:
            total += price
        else:
            unknown += 1
        gem_details.append(
            GemCostDetail(
                name=usage.name,
                level=usage.level,
                quality=usage.quality,
                price=price,
                matched=matched,
            )
        )
    return CostSummary(
        total_cost_chaos=total,
        unknown_cost_count=unknown,
        slot_costs=tuple(slot_details),
        gem_costs=tuple(gem_details),
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_cost_outputs(
    build_id: str, summary: CostSummary, base_path: Path | str | None = None
) -> None:
    artifacts = artifact_store.artifact_paths(build_id, base_path)
    slot_file = artifacts.base_dir / "slot_costs.json"
    gem_file = artifacts.base_dir / "gem_costs.json"

    sorted_slots = sorted(
        summary.slot_costs,
        key=lambda detail: (detail.slot.casefold(), detail.name.casefold()),
    )
    sorted_gems = sorted(
        summary.gem_costs,
        key=lambda detail: (
            detail.name.casefold(),
            detail.level if detail.level is not None else -1,
            detail.quality if detail.quality is not None else -1,
        ),
    )

    slot_payload = {
        "total_cost_chaos": summary.total_cost_chaos,
        "unknown_cost_count": summary.unknown_cost_count,
        "slots": [
            {
                "slot": detail.slot,
                "name": detail.name,
                "cost_chaos": detail.cost_chaos,
                "matched": detail.matched,
            }
            for detail in sorted_slots
        ],
    }
    gem_payload = {
        "total_cost_chaos": summary.total_cost_chaos,
        "unknown_cost_count": summary.unknown_cost_count,
        "gems": [
            {
                "name": detail.name,
                "level": detail.level,
                "quality": detail.quality,
                "price": detail.price,
                "matched": detail.matched,
            }
            for detail in sorted_gems
        ],
    }

    _write_json(slot_file, slot_payload)
    _write_json(gem_file, gem_payload)
