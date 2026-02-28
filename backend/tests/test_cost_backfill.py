import pytest

from backend.engine.artifacts.store import format_ruleset_id, write_build_artifacts
from backend.engine.pricing.backfill import run_cost_backfill
from backend.engine.pricing.snapshots import ingest_price_snapshot_from_fixtures

BUILD_XML = """<build>
  <item slot="Body">
    <rarity>Unique</rarity>
    <name>Headhunter</name>
  </item>
  <skillgems>
    <skillgem name="Frostbolt" level="20" quality="23" />
  </skillgems>
</build>"""


class DummyRepo:
    def __init__(self, builds, latest=None):
        self.builds = list(builds)
        self.latest = latest or {}
        self.inserted_rows = []
        self.list_calls = []

    def list_builds(self, **kwargs):
        limit = kwargs.get("limit") or len(self.builds)
        offset = kwargs.get("offset") or 0
        self.list_calls.append((limit, offset))
        return self.builds[offset : offset + limit]

    def get_latest_build_cost(self, build_id: str):
        return self.latest.get(build_id)

    def insert_build_costs(self, rows):
        self.inserted_rows.extend(rows)


def _write_build(tmp_path, build_id: str) -> None:
    write_build_artifacts(build_id, xml=BUILD_XML, code="print('ok')", base_path=tmp_path)


def test_backfill_inserts_costs(tmp_path):
    snapshot = ingest_price_snapshot_from_fixtures(
        league="Sentinel", timestamp="2026-02-26T120000Z", data_path=tmp_path
    )
    ruleset_id = format_ruleset_id("commit", "2025.2", snapshot.snapshot_id)
    build_id = "build-insert"
    _write_build(tmp_path, build_id)
    repo = DummyRepo(
        [
            {
                "build_id": build_id,
                "ruleset_id": ruleset_id,
            }
        ]
    )

    summary = run_cost_backfill(repo, data_path=tmp_path)

    assert summary.processed == 1
    assert summary.inserted == 1
    assert summary.skipped == 0
    assert summary.errors == 0
    row = repo.inserted_rows[0]
    assert row.price_snapshot_id == snapshot.snapshot_id
    assert row.slot_costs_json_path == f"data/builds/{build_id}/slot_costs.json"
    assert row.gem_costs_json_path == f"data/builds/{build_id}/gem_costs.json"
    assert row.total_cost_chaos == pytest.approx(1200.1)
    assert summary.builds[0].status == "inserted"


def test_backfill_dry_run_skips_insert(tmp_path):
    snapshot = ingest_price_snapshot_from_fixtures(
        league="Sentinel", timestamp="2026-02-26T120000Z", data_path=tmp_path
    )
    ruleset_id = format_ruleset_id("commit", "2025.2", snapshot.snapshot_id)
    build_id = "build-dry"
    _write_build(tmp_path, build_id)
    repo = DummyRepo(
        [
            {
                "build_id": build_id,
                "ruleset_id": ruleset_id,
            }
        ]
    )

    summary = run_cost_backfill(repo, data_path=tmp_path, dry_run=True)

    assert summary.inserted == 0
    assert summary.builds[0].status == "dry_run"
    assert summary.builds[0].detail == "dry run prevented insert"
    assert not repo.inserted_rows


def test_backfill_skips_if_latest_matches(tmp_path):
    snapshot = ingest_price_snapshot_from_fixtures(
        league="Sentinel", timestamp="2026-02-26T120000Z", data_path=tmp_path
    )
    ruleset_id = format_ruleset_id("commit", "2025.2", snapshot.snapshot_id)
    build_id = "build-skip"
    _write_build(tmp_path, build_id)
    repo = DummyRepo(
        [
            {
                "build_id": build_id,
                "ruleset_id": ruleset_id,
            }
        ],
        latest={build_id: {"price_snapshot_id": snapshot.snapshot_id}},
    )

    summary = run_cost_backfill(repo, data_path=tmp_path)

    assert summary.inserted == 0
    assert summary.skipped == 1
    assert summary.builds[0].status == "skipped"


def test_backfill_uses_override_snapshot(tmp_path):
    snapshot = ingest_price_snapshot_from_fixtures(
        league="Sentinel", timestamp="2026-02-26T120000Z", data_path=tmp_path
    )
    build_id = "build-override"
    _write_build(tmp_path, build_id)
    repo = DummyRepo(
        [
            {
                "build_id": build_id,
                "ruleset_id": "invalid-ruleset",
            }
        ]
    )

    summary = run_cost_backfill(repo, data_path=tmp_path, price_snapshot_id=snapshot.snapshot_id)

    assert summary.inserted == 1
    assert summary.builds[0].price_snapshot_id == snapshot.snapshot_id
    assert repo.inserted_rows[0].price_snapshot_id == snapshot.snapshot_id
