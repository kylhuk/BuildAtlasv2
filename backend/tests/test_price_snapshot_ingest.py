"""Price snapshot fixture ingestion expectations."""

import json

import pytest

from backend.engine.pricing.snapshots import (
    build_price_snapshot_id,
    ingest_price_snapshot_from_fixtures,
)


def test_price_snapshot_files_created(tmp_path):
    result = ingest_price_snapshot_from_fixtures(
        league="Sentinel",
        timestamp="2026-02-26T120000Z",
        data_path=tmp_path,
    )
    snapshot_dir = tmp_path / "prices" / "Sentinel" / result.timestamp
    assert snapshot_dir.exists()
    for name in ("unique_items.json", "skill_gems.json", "price_snapshot.json"):
        assert (snapshot_dir / name).exists()


def test_price_snapshot_metadata_and_id(tmp_path):
    result = ingest_price_snapshot_from_fixtures(
        league="Sentinel",
        timestamp="2026-02-26T120000Z",
        data_path=tmp_path,
    )
    metadata = json.loads(result.metadata_path.read_text())
    assert metadata["id"] == build_price_snapshot_id("Sentinel", "2026-02-26T120000Z")
    assert metadata["league"] == "Sentinel"
    assert metadata["timestamp"] == "2026-02-26T120000Z"
    assert metadata["sources"] == ["fixtures"]
    assert metadata["files"] == {
        "skill_gems": "skill_gems.json",
        "unique_items": "unique_items.json",
    }


def test_index_latest_history_orders_and_deterministic_rewrite(tmp_path):
    first = "2026-02-25T120000Z"
    second = "2026-02-27T120000Z"
    ingest_price_snapshot_from_fixtures(
        league="Sentinel",
        timestamp=first,
        data_path=tmp_path,
    )
    ingest_price_snapshot_from_fixtures(
        league="Sentinel",
        timestamp=second,
        data_path=tmp_path,
    )
    index_path = tmp_path / "prices" / "index.json"
    index_payload = json.loads(index_path.read_text())
    league_entry = index_payload["leagues"]["Sentinel"]
    history = league_entry["history"]
    assert history[0]["timestamp"] == second
    assert history[1]["timestamp"] == first
    assert league_entry["latest"] == history[0]
    before = index_path.read_text()
    ingest_price_snapshot_from_fixtures(
        league="Sentinel",
        timestamp=second,
        data_path=tmp_path,
    )
    assert index_path.read_text() == before


def test_invalid_league_and_timestamp_guard(tmp_path):
    with pytest.raises(ValueError):
        ingest_price_snapshot_from_fixtures(
            league="",
            timestamp="2026-02-26T120000Z",
            data_path=tmp_path,
        )
    with pytest.raises(ValueError):
        ingest_price_snapshot_from_fixtures(
            league="Sentinel!",
            timestamp="2026-02-26T120000Z",
            data_path=tmp_path,
        )
    with pytest.raises(ValueError):
        ingest_price_snapshot_from_fixtures(
            league="Sentinel",
            timestamp="not-a-timestamp",
            data_path=tmp_path,
        )


def test_timestamp_with_colon_offset_is_supported(tmp_path):
    result = ingest_price_snapshot_from_fixtures(
        league="Sentinel",
        timestamp="2026-02-26T12:00:00+00:00",
        data_path=tmp_path,
    )
    assert result.timestamp == "2026-02-26T120000Z"
