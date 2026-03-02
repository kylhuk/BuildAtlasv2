"""Dataset snapshot builder for EP-V4 surrogate experiments."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TypeGuard

FEATURE_SCHEMA_VERSION = "v4"
_DATASET_FILENAME = "dataset.jsonl"
_MANIFEST_FILENAME = "manifest.json"
_RESIST_KEYS = ("fire", "cold", "lightning", "chaos")
_ATTRIBUTE_KEYS = ("strength", "dexterity", "intelligence")
_SNAPSHOT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

_FEATURE_PREFIX = "feature_"
FEATURE_ITEM_SLOT_COUNT = f"{_FEATURE_PREFIX}item_slot_count"
FEATURE_ITEM_ADJUSTABLE_COUNT = f"{_FEATURE_PREFIX}item_adjustable_count"
FEATURE_ITEM_BASE_TYPE_COUNT = f"{_FEATURE_PREFIX}item_base_type_count"
FEATURE_ITEM_CONTRIB_RESISTS = f"{_FEATURE_PREFIX}item_contrib_resists"
FEATURE_ITEM_CONTRIB_ATTRIBUTES = f"{_FEATURE_PREFIX}item_contrib_attributes"
FEATURE_ITEM_CONTRIB_LIFE = f"{_FEATURE_PREFIX}item_contrib_life"
FEATURE_ITEM_CONTRIB_ENERGY_SHIELD = f"{_FEATURE_PREFIX}item_contrib_energy_shield"
FEATURE_AFFIX_RESIST_LINES = f"{_FEATURE_PREFIX}affix_resist_lines"
FEATURE_AFFIX_ATTRIBUTE_LINES = f"{_FEATURE_PREFIX}affix_attribute_lines"
FEATURE_AFFIX_LIFE_LINES = f"{_FEATURE_PREFIX}affix_life_lines"
FEATURE_AFFIX_ENERGY_SHIELD_LINES = f"{_FEATURE_PREFIX}affix_energy_shield_lines"
FEATURE_AFFIX_TOTAL_LINES = f"{_FEATURE_PREFIX}affix_total_lines"
FEATURE_PASSIVE_NODE_COUNT = f"{_FEATURE_PREFIX}passive_node_count"
FEATURE_PASSIVE_REQUIRED_TARGETS = f"{_FEATURE_PREFIX}passive_required_targets"
FEATURE_GEM_GROUP_COUNT = f"{_FEATURE_PREFIX}gem_group_count"
FEATURE_GEM_DAMAGE_GROUP_COUNT = f"{_FEATURE_PREFIX}gem_damage_group_count"
FEATURE_GEM_UTILITY_GROUP_COUNT = f"{_FEATURE_PREFIX}gem_utility_group_count"
FEATURE_GEM_TOTAL_COUNT = f"{_FEATURE_PREFIX}gem_total_count"
FEATURE_GEM_MAIN_GROUP_COUNT = f"{_FEATURE_PREFIX}gem_main_group_count"
FEATURE_GEM_MAX_LINK_COUNT = f"{_FEATURE_PREFIX}gem_max_link_count"
FEATURE_GEM_MAIN_LINK_REQUIREMENT = f"{_FEATURE_PREFIX}gem_main_link_requirement"
FEATURE_IDENTITY_TOKENS = f"{_FEATURE_PREFIX}identity_tokens"
FEATURE_IDENTITY_CROSS_TOKENS = f"{_FEATURE_PREFIX}identity_cross_tokens"
_IDENTITY_TOKEN_LIMIT = 256
_CROSS_TOKEN_LIMIT = 384
_BASE_TYPE_CROSS_LIMIT = 16
_PASSIVE_CROSS_LIMIT = 48
_MAIN_SUPPORT_CROSS_LIMIT = 8
_TRIPLE_CROSS_LIMIT = 32
_LINK_REQUIREMENT_BUCKETS = (
    (0, 2, "0-2"),
    (3, 4, "3-4"),
    (5, 6, "5-6"),
    (7, 8, "7-8"),
)
_LINK_REQUIREMENT_DEFAULT_BUCKET = "9+"
FEATURE_SIGNAL_KEYS = (
    FEATURE_ITEM_SLOT_COUNT,
    FEATURE_ITEM_ADJUSTABLE_COUNT,
    FEATURE_ITEM_BASE_TYPE_COUNT,
    FEATURE_ITEM_CONTRIB_RESISTS,
    FEATURE_ITEM_CONTRIB_ATTRIBUTES,
    FEATURE_ITEM_CONTRIB_LIFE,
    FEATURE_ITEM_CONTRIB_ENERGY_SHIELD,
    FEATURE_AFFIX_RESIST_LINES,
    FEATURE_AFFIX_ATTRIBUTE_LINES,
    FEATURE_AFFIX_LIFE_LINES,
    FEATURE_AFFIX_ENERGY_SHIELD_LINES,
    FEATURE_AFFIX_TOTAL_LINES,
    FEATURE_PASSIVE_NODE_COUNT,
    FEATURE_PASSIVE_REQUIRED_TARGETS,
    FEATURE_GEM_GROUP_COUNT,
    FEATURE_GEM_DAMAGE_GROUP_COUNT,
    FEATURE_GEM_UTILITY_GROUP_COUNT,
    FEATURE_GEM_TOTAL_COUNT,
    FEATURE_GEM_MAIN_GROUP_COUNT,
    FEATURE_GEM_MAX_LINK_COUNT,
    FEATURE_GEM_MAIN_LINK_REQUIREMENT,
)


@dataclass(frozen=True)
class SnapshotResult:
    """Metadata returned after building a dataset snapshot."""

    snapshot_id: str
    dataset_path: Path
    manifest_path: Path
    row_count: int
    feature_schema_version: str
    dataset_hash: str


def build_dataset_snapshot(
    data_path: Path | str,
    output_root: Path | str,
    snapshot_id: str,
    *,
    exclude_stub_rows: bool = False,
    profile_id: str | None = None,
    scenario_id: str | None = None,
) -> SnapshotResult:
    """Build a dataset snapshot from the given data/artifacts root."""

    if not snapshot_id or not snapshot_id.strip():
        raise ValueError("snapshot_id must be a non-empty string")

    data_root = Path(data_path)
    output_root_path = Path(output_root)
    builds_root = data_root / "builds"

    snapshot_dir = _resolve_snapshot_dir(output_root_path, snapshot_id)
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    rows = list(
        _collect_rows(
            builds_root,
            exclude_stub_rows=exclude_stub_rows,
            profile_id=profile_id,
            scenario_id=scenario_id,
        )
    )
    dataset_path = snapshot_dir / _DATASET_FILENAME
    dataset_hash = _write_rows(rows, dataset_path)

    manifest = {
        "snapshot_id": snapshot_id,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "row_count": len(rows),
        "source_root_path": str(builds_root.resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_hash": dataset_hash,
    }
    manifest_path = snapshot_dir / _MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")

    return SnapshotResult(
        snapshot_id=snapshot_id,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        row_count=len(rows),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        dataset_hash=dataset_hash,
    )


def _collect_rows(
    builds_root: Path,
    *,
    exclude_stub_rows: bool,
    profile_id: str | None,
    scenario_id: str | None,
) -> Iterable[Mapping[str, Any]]:
    if not builds_root.exists():
        return ()
    rows: list[Mapping[str, Any]] = []
    for build_id in sorted(entry.name for entry in builds_root.iterdir() if entry.is_dir()):
        genome = _read_json(builds_root / build_id / "genome.json")
        metrics = _read_json(builds_root / build_id / "metrics_raw.json")
        build_details = _read_json(builds_root / build_id / "build_details.json")
        if not _is_mapping(genome) or not _is_mapping(metrics):
            continue
        rows.extend(
            _rows_for_build(
                build_id,
                genome,
                metrics,
                build_details,
                exclude_stub_rows=exclude_stub_rows,
                profile_id_filter=profile_id,
                scenario_id_filter=scenario_id,
            )
        )
    return rows


def _rows_for_build(
    build_id: str,
    genome: Mapping[str, Any],
    metrics: Mapping[str, Any],
    build_details: Mapping[str, Any] | None,
    *,
    exclude_stub_rows: bool,
    profile_id_filter: str | None,
    scenario_id_filter: str | None,
) -> Iterable[Mapping[str, Any]]:
    pairs = sorted(
        ((str(key), key) for key in metrics),
        key=lambda entry: entry[0],
    )
    for scenario_label, scenario_key in pairs:
        scenario_payload = metrics.get(scenario_key)
        if not _is_mapping(scenario_payload):
            continue
        if exclude_stub_rows and _is_stub_scenario_payload(scenario_payload):
            continue
        row = _build_row(build_id, scenario_label, genome, scenario_payload, build_details)
        if not _matches_filter(row.get("profile_id"), profile_id_filter):
            continue
        if not _matches_filter(row.get("scenario_id"), scenario_id_filter):
            continue
        yield row


def _is_stub_scenario_payload(payload: Mapping[str, Any]) -> bool:
    warning_fields = (
        payload.get("warnings"),
        payload.get("pob_warnings"),
    )
    for warning_field in warning_fields:
        for warning in _sequence_of_strings(warning_field):
            if warning.strip().lower() == "generation_stub_metrics":
                return True
    return False


def _build_row(
    build_id: str,
    scenario_id: str,
    genome: Mapping[str, Any],
    scenario_data: Mapping[str, Any],
    build_details: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    metrics_section = _as_mapping(scenario_data.get("metrics"))
    defense_section = _as_mapping(scenario_data.get("defense"))
    resources_section = _as_mapping(scenario_data.get("resources"))
    reservation_section = _as_mapping(scenario_data.get("reservation"))
    attributes_section = _as_mapping(scenario_data.get("attributes"))
    resists_section = _as_mapping(defense_section.get("resists") if defense_section else None)

    row: dict[str, Any] = {
        "build_id": build_id,
        "scenario_id": scenario_id,
        "class": _genome_field(genome, ("class", "class_name")),
        "ascendancy": _genome_field(genome, ("ascendancy",)),
        "main_skill_package": _genome_field(genome, ("main_skill_package",)),
        "defense_archetype": _genome_field(genome, ("defense_archetype",)),
        "budget_tier": _genome_field(genome, ("budget_tier",)),
        "profile_id": _genome_field(genome, ("profile_id",)),
        "full_dps": _numeric(metrics_section, "full_dps"),
        "max_hit": _numeric(metrics_section, "max_hit"),
        "utility_score": _numeric(metrics_section, "utility_score"),
        "armour": _numeric(defense_section, "armour"),
        "evasion": _numeric(defense_section, "evasion"),
        "life": _numeric(resources_section, "life"),
        "mana": _numeric(resources_section, "mana"),
        "reserved_percent": _numeric(reservation_section, "reserved_percent"),
        "available_percent": _numeric(reservation_section, "available_percent"),
        # FL-03: Include gate_pass for ML training
        "gate_pass": scenario_data.get("gate_pass"),
        "gate_fail_reasons": scenario_data.get("gate_fail_reasons", []),
    }

    for resist in _RESIST_KEYS:
        row[f"resist_{resist}"] = _numeric(resists_section, resist)

    for attribute in _ATTRIBUTE_KEYS:
        row[attribute] = _numeric(attributes_section, attribute)

    row.update(extract_feature_signals(build_details))
    identity_tokens = list(row.get(FEATURE_IDENTITY_TOKENS) or [])
    identity_tokens = _extend_identity_tokens(
        identity_tokens, _categorical_identity_tokens(row)
    )
    row[FEATURE_IDENTITY_TOKENS] = identity_tokens
    return row


def _categorical_identity_tokens(row: Mapping[str, Any]) -> list[str]:
    tokens: list[str] = []
    entries = (
        ("profile_id", row.get("profile_id")),
        ("scenario_id", row.get("scenario_id")),
        ("class", row.get("class")),
        ("ascendancy", row.get("ascendancy")),
        ("defense", row.get("defense_archetype")),
        ("budget", row.get("budget_tier")),
        ("main_skill", row.get("main_skill_package")),
    )
    for prefix, value in entries:
        normalized = _normalize_token_value(value)
        if normalized:
            tokens.append(f"{prefix}:{normalized}")
    return tokens


def _extend_identity_tokens(tokens: list[str], extras: Sequence[str]) -> list[str]:
    extended = list(tokens)
    seen: set[str] = set(extended)
    for token in extras:
        if not token or token in seen:
            continue
        extended.append(token)
        seen.add(token)
    return extended[:_IDENTITY_TOKEN_LIMIT]



def _write_rows(rows: Iterable[Mapping[str, Any]], path: Path) -> str:
    hasher = sha256()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            handle.write(line + "\n")
            hasher.update(line.encode("utf-8"))
            hasher.update(b"\n")
    return hasher.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not _is_mapping(content):
        return None
    return content


def _resolve_snapshot_dir(output_root: Path, snapshot_id: str) -> Path:
    if not _SNAPSHOT_ID_PATTERN.fullmatch(snapshot_id):
        raise ValueError(
            "snapshot_id must start with an alphanumeric character and contain "
            "only letters, numbers, '.', '_' or '-'"
        )
    root = output_root.resolve()
    snapshot_dir = (root / snapshot_id).resolve()
    if root not in snapshot_dir.parents:
        raise ValueError("snapshot_id resolved outside output root")
    return snapshot_dir


def _is_mapping(value: Any) -> TypeGuard[Mapping[str, Any]]:
    return isinstance(value, Mapping)


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _genome_field(genome: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = genome.get(key)
        if value is not None:
            return str(value)
    return None


def _numeric(payload: Mapping[str, Any] | None, key: str) -> float | None:
    if not payload:
        return None
    return _to_number(payload.get(key))


def _to_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_feature_signals(build_details: Mapping[str, Any] | None) -> dict[str, Any]:
    values: dict[str, Any] = {key: 0.0 for key in FEATURE_SIGNAL_KEYS}
    if not _is_mapping(build_details):
        values[FEATURE_IDENTITY_TOKENS] = []
        values[FEATURE_IDENTITY_CROSS_TOKENS] = []
        return values
    slot_templates = _slot_templates_from_details(build_details)
    values[FEATURE_ITEM_SLOT_COUNT] = float(len(slot_templates))
    adjustable_count = sum(1 for template in slot_templates if template.get("adjustable"))
    values[FEATURE_ITEM_ADJUSTABLE_COUNT] = float(adjustable_count)
    base_types = {
        str(template.get("base_type")).strip()
        for template in slot_templates
        if template.get("base_type")
    }
    values[FEATURE_ITEM_BASE_TYPE_COUNT] = float(len(base_types))

    resist_total = attribute_total = life_total = es_total = 0.0
    resist_lines = attribute_lines = life_lines = es_lines = 0
    for template in slot_templates:
        contributions = _as_mapping(template.get("contributions"))
        if not contributions:
            continue
        for key in _RESIST_KEYS:
            amount = _to_number(contributions.get(key))
            if amount is not None and amount > 0:
                resist_total += amount
                resist_lines += 1
        for key in _ATTRIBUTE_KEYS:
            amount = _to_number(contributions.get(key))
            if amount is not None and amount > 0:
                attribute_total += amount
                attribute_lines += 1
        life_amount = _to_number(contributions.get("life"))
        if life_amount is not None and life_amount > 0:
            life_total += life_amount
            life_lines += 1
        es_amount = _to_number(contributions.get("energy_shield"))
        if es_amount is not None and es_amount > 0:
            es_total += es_amount
            es_lines += 1

    values[FEATURE_ITEM_CONTRIB_RESISTS] = resist_total
    values[FEATURE_ITEM_CONTRIB_ATTRIBUTES] = attribute_total
    values[FEATURE_ITEM_CONTRIB_LIFE] = life_total
    values[FEATURE_ITEM_CONTRIB_ENERGY_SHIELD] = es_total
    values[FEATURE_AFFIX_RESIST_LINES] = float(resist_lines)
    values[FEATURE_AFFIX_ATTRIBUTE_LINES] = float(attribute_lines)
    values[FEATURE_AFFIX_LIFE_LINES] = float(life_lines)
    values[FEATURE_AFFIX_ENERGY_SHIELD_LINES] = float(es_lines)
    values[FEATURE_AFFIX_TOTAL_LINES] = float(
        resist_lines + attribute_lines + life_lines + es_lines
    )

    passives = _as_mapping(build_details.get("passives"))
    values[FEATURE_PASSIVE_NODE_COUNT] = float(
        len(_sequence_of_strings(passives.get("node_ids") if passives else ()))
    )
    values[FEATURE_PASSIVE_REQUIRED_TARGETS] = float(
        len(_sequence_of_strings(passives.get("required_targets") if passives else ()))
    )

    gems = _as_mapping(build_details.get("gems"))
    groups = _sequence_of_mappings(gems.get("groups") if gems else ())
    values[FEATURE_GEM_GROUP_COUNT] = float(len(groups))
    damage_groups = sum(1 for group in groups if str(group.get("group_type")).lower() == "damage")
    utility_groups = sum(1 for group in groups if str(group.get("group_type")).lower() == "utility")
    values[FEATURE_GEM_DAMAGE_GROUP_COUNT] = float(damage_groups)
    values[FEATURE_GEM_UTILITY_GROUP_COUNT] = float(utility_groups)
    total_gems = sum(len(_sequence(group.get("gems"))) for group in groups)
    values[FEATURE_GEM_TOTAL_COUNT] = float(total_gems)

    main_group_id = _coerce_str(gems.get("full_dps_group_id") if gems else None)
    main_group_count = 0
    if main_group_id:
        for group in groups:
            if str(group.get("id")) == main_group_id:
                main_group_count = len(_sequence(group.get("gems")))
                break
    values[FEATURE_GEM_MAIN_GROUP_COUNT] = float(main_group_count)

    socket_plan = _as_mapping(gems.get("socket_plan") if gems else None)
    assignments = _sequence_of_mappings(socket_plan.get("assignments") if socket_plan else ())
    link_values = [_to_number(entry.get("link_count")) or 0.0 for entry in assignments]
    values[FEATURE_GEM_MAX_LINK_COUNT] = float(max(link_values)) if link_values else 0.0
    main_requirement = _to_number(socket_plan.get("main_link_requirement") if socket_plan else None)
    values[FEATURE_GEM_MAIN_LINK_REQUIREMENT] = float(main_requirement or 0.0)

    identity_tokens, cross_tokens = _build_token_lists(build_details)
    values[FEATURE_IDENTITY_TOKENS] = identity_tokens
    values[FEATURE_IDENTITY_CROSS_TOKENS] = cross_tokens
    return values


def _slot_templates_from_details(build_details: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    items_section = _as_mapping(build_details.get("items"))
    if not items_section:
        return []
    templates = _sequence_of_mappings(items_section.get("slot_templates"))
    if templates:
        return templates
    return _sequence_of_mappings(items_section.get("items"))


def _sequence(value: Any) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [entry for entry in value if entry is not None]


def _sequence_of_mappings(value: Any) -> list[Mapping[str, Any]]:
    return [entry for entry in _sequence(value) if isinstance(entry, Mapping)]


def _sequence_of_strings(value: Any) -> list[str]:
    return [str(entry) for entry in _sequence(value) if entry is not None]


def _coerce_str(value: Any | None) -> str | None:
    if value is None:
        return None
    candidate = str(value).strip()
    return candidate if candidate else None


def _build_token_lists(build_details: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    identity_tokens: set[str] = set()
    base_types: set[str] = set()
    slot_templates = _slot_templates_from_details(build_details)
    _collect_item_identity_tokens(slot_templates, identity_tokens, base_types)
    passive_nodes, passive_targets, keystone_nodes, mastery_nodes = (
        _collect_passive_identity_tokens(build_details, identity_tokens)
    )
    main_gem, main_supports, main_slot = _collect_gem_identity_tokens(
        build_details, identity_tokens
    )
    cross_tokens = _build_cross_tokens(
        main_gem=main_gem,
        main_supports=main_supports,
        main_slot=main_slot,
        base_types=base_types,
        passive_nodes=passive_nodes,
        passive_targets=passive_targets,
        keystone_nodes=keystone_nodes,
        mastery_nodes=mastery_nodes,
    )
    return (
        _finalize_tokens(identity_tokens, _IDENTITY_TOKEN_LIMIT),
        _finalize_tokens(cross_tokens, _CROSS_TOKEN_LIMIT),
    )


def _collect_item_identity_tokens(
    slot_templates: list[Mapping[str, Any]],
    tokens: set[str],
    base_types: set[str],
) -> None:
    for template in slot_templates:
        slot_id = _normalize_token_value(template.get("slot_id"))
        if slot_id:
            tokens.add(f"slot:{slot_id}")
            marker = "adjustable" if bool(template.get("adjustable")) else "fixed"
            tokens.add(f"slot:{slot_id}:{marker}")
        base_type = _normalize_token_value(template.get("base_type"))
        if base_type:
            tokens.add(f"base_type:{base_type}")
            base_types.add(base_type)
            if slot_id:
                tokens.add(f"slot_base:{slot_id}|base_type:{base_type}")
        contributions = _as_mapping(template.get("contributions"))
        if not contributions:
            continue
        contribution_affixes: list[str] = []
        if _contributions_positive(contributions, _RESIST_KEYS):
            contribution_affixes.append("resists")
            if slot_id:
                tokens.add(f"slot:{slot_id}:resists")
        if _contributions_positive(contributions, _ATTRIBUTE_KEYS):
            contribution_affixes.append("attributes")
            if slot_id:
                tokens.add(f"slot:{slot_id}:attributes")
        life_amount = _to_number(contributions.get("life"))
        if life_amount is not None and life_amount > 0:
            contribution_affixes.append("life")
            if slot_id:
                tokens.add(f"slot:{slot_id}:life")
        es_amount = _to_number(contributions.get("energy_shield"))
        if es_amount is not None and es_amount > 0:
            contribution_affixes.append("energy_shield")
            if slot_id:
                tokens.add(f"slot:{slot_id}:energy_shield")

        for affix in contribution_affixes:
            tokens.add(f"affix_presence:{affix}")
            if slot_id:
                tokens.add(f"slot_affix:{slot_id}|affix:{affix}")
            if base_type:
                tokens.add(f"base_affix:{base_type}|affix:{affix}")
                if slot_id:
                    tokens.add(f"slot_base_affix:{slot_id}|base_type:{base_type}|affix:{affix}")


def _collect_passive_identity_tokens(
    build_details: Mapping[str, Any], tokens: set[str]
) -> tuple[list[str], list[str], list[str], list[str]]:
    passives = _as_mapping(build_details.get("passives"))
    if not passives:
        return [], [], [], []
    node_values: list[str] = []
    target_values: list[str] = []
    keystone_values: list[str] = []
    mastery_values: list[str] = []
    seen: set[str] = set()
    for entry in _sequence_of_strings(passives.get("node_ids")):
        normalized = _normalize_token_value(entry)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        node_values.append(normalized)
        tokens.add(f"passive_node:{normalized}")
        if "keystone" in normalized:
            tokens.add(f"passive_keystone:{normalized}")
            if normalized not in keystone_values:
                keystone_values.append(normalized)
        if "mastery" in normalized:
            tokens.add(f"passive_mastery:{normalized}")
            if normalized not in mastery_values:
                mastery_values.append(normalized)
    for entry in _sequence_of_strings(passives.get("required_targets")):
        normalized = _normalize_token_value(entry)
        if not normalized:
            continue
        if normalized in target_values:
            continue
        target_values.append(normalized)
        tokens.add(f"passive_target:{normalized}")

    for node_detail in _sequence_of_mappings(passives.get("node_details")):
        node_id = _normalize_token_value(node_detail.get("id"))
        kind = _normalize_token_value(node_detail.get("kind"))
        if not node_id or not kind:
            continue
        tokens.add(f"passive_kind:{kind}")
        tokens.add(f"passive_{kind}:{node_id}")
        if kind == "keystone" and node_id not in keystone_values:
            keystone_values.append(node_id)
        if kind == "mastery" and node_id not in mastery_values:
            mastery_values.append(node_id)

    for entry in _sequence_of_strings(passives.get("keystone_ids")):
        normalized = _normalize_token_value(entry)
        if not normalized:
            continue
        tokens.add(f"passive_keystone:{normalized}")
        if normalized not in keystone_values:
            keystone_values.append(normalized)

    for entry in _sequence_of_strings(passives.get("mastery_ids")):
        normalized = _normalize_token_value(entry)
        if not normalized:
            continue
        tokens.add(f"passive_mastery:{normalized}")
        if normalized not in mastery_values:
            mastery_values.append(normalized)

    return node_values, target_values, keystone_values, mastery_values


def _collect_gem_identity_tokens(
    build_details: Mapping[str, Any], tokens: set[str]
) -> tuple[str | None, list[str], str | None]:
    gems = _as_mapping(build_details.get("gems"))
    if not gems:
        return None, [], None
    groups = _sequence_of_mappings(gems.get("groups") if gems else ())
    main_group_id = _normalize_token_value(gems.get("full_dps_group_id"))
    main_gem: str | None = None
    main_supports: list[str] = []
    main_group_type: str | None = None
    main_slot: str | None = None

    socket_plan = _as_mapping(gems.get("socket_plan"))
    group_slots: dict[str, str] = {}
    if socket_plan:
        assignments = _sequence_of_mappings(socket_plan.get("assignments"))
        for assignment in assignments:
            group_id = _normalize_token_value(assignment.get("group_id"))
            slot_id = _normalize_token_value(assignment.get("slot_id"))
            if group_id and slot_id:
                group_slots[group_id] = slot_id

    for group in groups:
        group_id = _normalize_token_value(group.get("id"))
        if group_id:
            tokens.add(f"group_id:{group_id}")
        group_type = _normalize_token_value(group.get("group_type"))
        if group_type:
            tokens.add(f"group_type:{group_type}")
        gems_in_group: list[str] = []
        for gem_entry in _sequence(group.get("gems")):
            normalized = _normalize_token_value(gem_entry)
            if normalized:
                gems_in_group.append(normalized)
                tokens.add(f"gem:{normalized}")
        if group_id:
            slot_id = group_slots.get(group_id)
            if slot_id:
                tokens.add(f"group_slot:{group_id}|slot:{slot_id}")
                for gem_name in gems_in_group:
                    tokens.add(f"slot:{slot_id}|gem:{gem_name}")

        for gem_pair in _pair_tokens(gems_in_group):
            if group_id:
                tokens.add(f"group:{group_id}|gem_pair:{gem_pair}")
            else:
                tokens.add(f"gem_pair:{gem_pair}")

        if main_group_id and group_id == main_group_id:
            if gems_in_group and main_gem is None:
                main_gem = gems_in_group[0]
            if group_type:
                main_group_type = group_type
            if group_id and group_slots.get(group_id):
                main_slot = group_slots[group_id]
                tokens.add(f"main_slot:{main_slot}")
            if len(gems_in_group) > 1:
                for support in gems_in_group[1:]:
                    if support not in main_supports:
                        main_supports.append(support)
                        tokens.add(f"main_support:{support}")
    if main_group_id:
        tokens.add(f"main_group:{main_group_id}")
        if main_group_type:
            tokens.add(f"main_group_type:{main_group_type}")
    if main_gem:
        tokens.add(f"main_gem:{main_gem}")
    if socket_plan:
        requirement = _to_number(socket_plan.get("main_link_requirement"))
        bucket = _link_requirement_bucket(requirement)
        if bucket:
            tokens.add(f"link_requirement:{bucket}")
    return main_gem, main_supports, main_slot


def _build_cross_tokens(
    *,
    main_gem: str | None,
    main_supports: list[str],
    main_slot: str | None,
    base_types: set[str],
    passive_nodes: list[str],
    passive_targets: list[str],
    keystone_nodes: list[str],
    mastery_nodes: list[str],
) -> set[str]:
    cross_tokens: set[str] = set()

    limited_base_types = _limited_unique(sorted(base_types), _BASE_TYPE_CROSS_LIMIT)
    limited_passive_nodes = _limited_unique(passive_nodes, _PASSIVE_CROSS_LIMIT)
    limited_passive_targets = _limited_unique(passive_targets, _PASSIVE_CROSS_LIMIT)
    limited_keystones = _limited_unique(keystone_nodes, _PASSIVE_CROSS_LIMIT)
    limited_masteries = _limited_unique(mastery_nodes, _PASSIVE_CROSS_LIMIT)
    limited_supports = _limited_unique(main_supports, _MAIN_SUPPORT_CROSS_LIMIT)

    if main_gem:
        for base_type in limited_base_types:
            token = _combine_token_components(("base_type", base_type), ("main_gem", main_gem))
            if token:
                cross_tokens.add(token)
        for node in limited_passive_nodes:
            token = _combine_token_components(("main_gem", main_gem), ("passive_node", node))
            if token:
                cross_tokens.add(token)
        for target in limited_passive_targets:
            token = _combine_token_components(("main_gem", main_gem), ("passive_target", target))
            if token:
                cross_tokens.add(token)
        for keystone in limited_keystones:
            token = _combine_token_components(
                ("main_gem", main_gem), ("passive_keystone", keystone)
            )
            if token:
                cross_tokens.add(token)
        for mastery in limited_masteries:
            token = _combine_token_components(("main_gem", main_gem), ("passive_mastery", mastery))
            if token:
                cross_tokens.add(token)
        if main_slot:
            token = _combine_token_components(("main_gem", main_gem), ("main_slot", main_slot))
            if token:
                cross_tokens.add(token)
            for base_type in limited_base_types:
                token = _combine_token_components(
                    ("main_slot", main_slot), ("base_type", base_type), ("main_gem", main_gem)
                )
                if token:
                    cross_tokens.add(token)

    for support in limited_supports:
        if main_gem:
            token = _combine_token_components(("main_gem", main_gem), ("main_support", support))
            if token:
                cross_tokens.add(token)
        for base_type in limited_base_types:
            token = _combine_token_components(("base_type", base_type), ("main_support", support))
            if token:
                cross_tokens.add(token)
        for node in limited_passive_nodes:
            token = _combine_token_components(("main_support", support), ("passive_node", node))
            if token:
                cross_tokens.add(token)
        for target in limited_passive_targets:
            token = _combine_token_components(("main_support", support), ("passive_target", target))
            if token:
                cross_tokens.add(token)
        for keystone in limited_keystones:
            token = _combine_token_components(
                ("main_support", support), ("passive_keystone", keystone)
            )
            if token:
                cross_tokens.add(token)
        for mastery in limited_masteries:
            token = _combine_token_components(
                ("main_support", support), ("passive_mastery", mastery)
            )
            if token:
                cross_tokens.add(token)

    if main_gem:
        triple_tokens: list[str] = []
        for support in limited_supports:
            for node in limited_passive_nodes:
                token = _combine_token_components(
                    ("main_gem", main_gem),
                    ("main_support", support),
                    ("passive_node", node),
                )
                if token:
                    triple_tokens.append(token)
        for support in limited_supports:
            for mastery in limited_masteries:
                token = _combine_token_components(
                    ("main_gem", main_gem),
                    ("main_support", support),
                    ("passive_mastery", mastery),
                )
                if token:
                    triple_tokens.append(token)
        for support in limited_supports:
            if main_slot:
                token = _combine_token_components(
                    ("main_slot", main_slot),
                    ("main_gem", main_gem),
                    ("main_support", support),
                )
                if token:
                    triple_tokens.append(token)

        for token in _limited_unique(triple_tokens, _TRIPLE_CROSS_LIMIT):
            cross_tokens.add(token)

    return cross_tokens


def _pair_tokens(values: Sequence[str]) -> list[str]:
    if len(values) < 2:
        return []
    unique = _limited_unique(values, _MAIN_SUPPORT_CROSS_LIMIT + 1)
    pairs: list[str] = []
    for idx in range(len(unique)):
        for jdx in range(idx + 1, len(unique)):
            left = unique[idx]
            right = unique[jdx]
            if left <= right:
                pair = f"{left}+{right}"
            else:
                pair = f"{right}+{left}"
            pairs.append(pair)
    return _limited_unique(pairs, _MAIN_SUPPORT_CROSS_LIMIT * 2)


def _limited_unique(values: Iterable[str], limit: int) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
        if len(output) >= limit:
            break
    return output


def _finalize_tokens(tokens: Iterable[str], limit: int) -> list[str]:
    unique_tokens = sorted({token for token in tokens if token})
    return unique_tokens[:limit]


def _normalize_token_value(value: Any | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    return re.sub(r"\s+", "_", normalized)


def _matches_filter(value: Any, expected: str | None) -> bool:
    if expected is None:
        return True
    if value is None:
        return False
    return str(value) == expected


def _contributions_positive(contributions: Mapping[str, Any], keys: Iterable[str]) -> bool:
    for key in keys:
        amount = _to_number(contributions.get(key))
        if amount is not None and amount > 0:
            return True
    return False


def _link_requirement_bucket(value: float | None) -> str | None:
    if value is None:
        return None
    try:
        requirement = int(value)
    except (TypeError, ValueError):
        return None
    if requirement < 0:
        requirement = 0
    for low, high, label in _LINK_REQUIREMENT_BUCKETS:
        if low <= requirement <= high:
            return label
    return _LINK_REQUIREMENT_DEFAULT_BUCKET


def _combine_token_components(*pairs: tuple[str, str]) -> str:
    components = []
    for key, value in pairs:
        if not key or not value:
            continue
        components.append(f"{key}:{value}")
    return "|".join(components)


__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "FEATURE_SIGNAL_KEYS",
    "FEATURE_IDENTITY_TOKENS",
    "FEATURE_IDENTITY_CROSS_TOKENS",
    "SnapshotResult",
    "build_dataset_snapshot",
    "extract_feature_signals",
]
