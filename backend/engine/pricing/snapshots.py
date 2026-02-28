"""Price snapshot ingestion helpers."""

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

DEFAULT_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "prices"
_TIMESTAMP_PATTERNS = (
    "%Y-%m-%dT%H%M%SZ",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H%M%S%z",
    "%Y-%m-%dT%H:%M:%S%z",
)


@dataclass(frozen=True)
class PriceSnapshotResult:
    """Result of top-level snapshot ingestion."""

    snapshot_id: str
    league: str
    timestamp: str
    sources: Tuple[str, ...]
    snapshot_path: Path
    unique_items_path: Path
    skill_gems_path: Path
    metadata_path: Path
    index_path: Path


def _validate_league(league: str) -> str:
    cleaned = league.strip()
    if not cleaned or not re.fullmatch(r"[A-Za-z0-9_-]+", cleaned):
        raise ValueError("league must be alphanumeric with underscores or dashes")
    return cleaned


def _normalize_timestamp(timestamp: str) -> str:
    cleaned = timestamp.strip()
    if not cleaned:
        raise ValueError("timestamp must not be empty")

    iso_candidate = cleaned[:-1] + "+00:00" if cleaned.endswith("Z") else cleaned
    try:
        parsed_iso = datetime.fromisoformat(iso_candidate)
    except ValueError:
        parsed_iso = None
    if parsed_iso and parsed_iso.tzinfo is not None:
        return parsed_iso.astimezone(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")

    for pattern in _TIMESTAMP_PATTERNS:
        try:
            parsed = datetime.strptime(cleaned, pattern)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        return parsed.strftime("%Y-%m-%dT%H%M%SZ")
    raise ValueError("timestamp must be ISO-formatted with UTC timezone")


def build_price_snapshot_id(league: str, timestamp: str) -> str:
    """Build a deterministic snapshot identifier."""

    validated_league = _validate_league(league)
    normalized_timestamp = _normalize_timestamp(timestamp)
    return f"{validated_league.lower()}-{normalized_timestamp}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_fixture_data(fixtures_dir: Path, filename: str) -> Any:
    source = fixtures_dir / filename
    if not source.exists():
        raise FileNotFoundError(f"fixture file missing: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def _update_price_index(index_path: Path, league: str, entry: Dict[str, Any]) -> None:
    index_payload = _load_json(index_path)
    leagues = index_payload.get("leagues", {})
    league_payload = leagues.get(league, {"history": []})
    history_map = {item["id"]: item for item in league_payload.get("history", [])}
    history_map[entry["id"]] = entry
    history_list = sorted(history_map.values(), key=lambda item: item["timestamp"], reverse=True)
    league_payload = {"history": history_list, "latest": history_list[0]}
    leagues[league] = league_payload
    ordered_leagues = {name: leagues[name] for name in sorted(leagues)}
    _write_json(index_path, {"leagues": ordered_leagues})


def ingest_price_snapshot_from_fixtures(
    league: str,
    timestamp: str,
    data_path: Path,
    fixtures_dir: Path | None = None,
    sources: Sequence[str] | None = None,
) -> PriceSnapshotResult:
    """Ingest price snapshot data from fixture inputs."""

    fixtures_dir = fixtures_dir or DEFAULT_FIXTURE_DIR
    if not fixtures_dir.exists():
        raise FileNotFoundError(f"fixtures directory not found: {fixtures_dir}")
    league_dir = _validate_league(league)
    normalized_timestamp = _normalize_timestamp(timestamp)
    snapshot_id = build_price_snapshot_id(league_dir, normalized_timestamp)
    data_root = Path(data_path)
    prices_root = data_root / "prices"
    snapshot_path = prices_root / league_dir / normalized_timestamp
    unique_items_path = snapshot_path / "unique_items.json"
    skill_gems_path = snapshot_path / "skill_gems.json"
    metadata_path = snapshot_path / "price_snapshot.json"
    index_path = prices_root / "index.json"

    unique_items = _load_fixture_data(fixtures_dir, "unique_items.json")
    skill_gems = _load_fixture_data(fixtures_dir, "skill_gems.json")

    source_names = tuple(sorted(set(sources or ("fixtures",))))

    _write_json(unique_items_path, unique_items)
    _write_json(skill_gems_path, skill_gems)

    metadata = {
        "id": snapshot_id,
        "league": league_dir,
        "timestamp": normalized_timestamp,
        "sources": list(source_names),
        "files": {
            "unique_items": unique_items_path.name,
            "skill_gems": skill_gems_path.name,
        },
    }
    _write_json(metadata_path, metadata)

    entry = {
        "id": snapshot_id,
        "timestamp": normalized_timestamp,
        "path": str(Path(league_dir) / normalized_timestamp),
        "sources": list(source_names),
    }
    _update_price_index(index_path, league_dir, entry)

    return PriceSnapshotResult(
        snapshot_id=snapshot_id,
        league=league_dir,
        timestamp=normalized_timestamp,
        sources=source_names,
        snapshot_path=snapshot_path,
        unique_items_path=unique_items_path,
        skill_gems_path=skill_gems_path,
        metadata_path=metadata_path,
        index_path=index_path,
    )
