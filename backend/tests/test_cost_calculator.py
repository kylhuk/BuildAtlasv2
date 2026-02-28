import json
from pathlib import Path

import pytest

from backend.engine.pricing.costs import (
    PriceSnapshot,
    SkillGemUsage,
    UniqueItemUsage,
    calculate_cost_summary,
    write_cost_outputs,
)


def test_calculates_and_writes_costs(tmp_path: Path) -> None:
    unique_items = [
        UniqueItemUsage(slot="Body", name="Headhunter"),
        UniqueItemUsage(slot="Gloves", name="Lion's Roar"),
        UniqueItemUsage(slot="Weapon", name="Unknown Edge"),
    ]
    skill_gems = [
        SkillGemUsage(name="Frostbolt", level=20, quality=23),
        SkillGemUsage(name="Frostbolt", level=None, quality=None),
        SkillGemUsage(name="NonExistent", level=1, quality=1),
    ]
    snapshot = PriceSnapshot(
        snapshot_id="sentinel-2026-02-26T120000Z",
        league="Sentinel",
        timestamp="2026-02-26T120000Z",
        unique_items={
            "headhunter": 1200.0,
            "lion's roar": 25.0,
        },
        skill_gems={
            ("frostbolt", 20, 23): 0.1,
            ("frostbolt", None, None): 0.2,
        },
    )

    summary = calculate_cost_summary(unique_items, skill_gems, snapshot)
    assert summary.total_cost_chaos == pytest.approx(1225.3)
    assert summary.unknown_cost_count == 2

    slot_map = {detail.slot: detail for detail in summary.slot_costs}
    assert not slot_map["Weapon"].matched
    assert slot_map["Body"].cost_chaos == 1200.0

    gem_map = {(gem.name, gem.level, gem.quality): gem for gem in summary.gem_costs}
    assert gem_map[("Frostbolt", 20, 23)].matched
    assert gem_map[("Frostbolt", None, None)].price == pytest.approx(0.2)
    assert not gem_map[("NonExistent", 1, 1)].matched

    write_cost_outputs("calc-build", summary, base_path=tmp_path)
    slot_path = tmp_path / "data" / "builds" / "calc-build" / "slot_costs.json"
    gem_path = tmp_path / "data" / "builds" / "calc-build" / "gem_costs.json"
    slot_payload = json.loads(slot_path.read_text())
    gem_payload = json.loads(gem_path.read_text())

    assert slot_payload["total_cost_chaos"] == pytest.approx(1225.3)
    assert slot_payload["unknown_cost_count"] == 2
    assert slot_payload["slots"][0]["slot"] == "Body"
    assert slot_payload["slots"][-1]["matched"] is False

    assert gem_payload["total_cost_chaos"] == pytest.approx(1225.3)
    assert gem_payload["gems"][0]["price"] == pytest.approx(0.2)
    assert gem_payload["gems"][1]["price"] == pytest.approx(0.1)
    assert gem_payload["gems"][-1]["matched"] is False
