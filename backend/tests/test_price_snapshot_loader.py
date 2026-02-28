import json
from pathlib import Path

import pytest

from backend.engine.pricing.costs import load_price_snapshot
from backend.engine.pricing.snapshots import ingest_price_snapshot_from_fixtures


def test_loads_snapshot_from_fixture(tmp_path: Path) -> None:
    result = ingest_price_snapshot_from_fixtures(
        league="Sentinel",
        timestamp="2026-02-26T120000Z",
        data_path=tmp_path,
    )
    snapshot = load_price_snapshot(result.snapshot_id, data_path=tmp_path)

    assert snapshot.snapshot_id == result.snapshot_id
    assert snapshot.league == result.league
    assert snapshot.timestamp == result.timestamp
    assert snapshot.unique_items.get("headhunter") == 1200
    assert snapshot.unique_items.get("lion's roar") == 25
    assert snapshot.skill_gems[
        (
            "frostbolt",
            20,
            23,
        )
    ] == pytest.approx(0.1)


def test_missing_index_files_raise(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_price_snapshot("any-id", data_path=tmp_path)


def test_unknown_snapshot_raises_value_error(tmp_path: Path) -> None:
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir(parents=True)
    index_path = prices_dir / "index.json"
    data = {"leagues": {"Sentinel": {"history": [], "latest": None}}}
    index_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError):
        load_price_snapshot("missing-id", data_path=tmp_path)
