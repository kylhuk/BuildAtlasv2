"""Tests for the drift report generator."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from backend.tools import drift_report as drift


def test_aggregate_scenario_metrics_handles_collected_rows() -> None:
    rows = [
        {
            "scenario_id": "mapping_t16",
            "full_dps": 5000,
            "max_hit": 4500,
            "utility_score": 1.3,
            "gate_pass": True,
        },
        {
            "scenario_id": "mapping_t16",
            "full_dps": 5200,
            "max_hit": 4600,
            "utility_score": 1.4,
            "gate_pass": False,
        },
        {
            "scenario_id": "pinnacle",
            "full_dps": 6100,
            "max_hit": 6000,
            "utility_score": 1.1,
        },
    ]
    summaries = drift.aggregate_scenario_metrics(rows)

    mapping = summaries["mapping_t16"]
    assert mapping.sample_count == 2
    assert mapping.averages["full_dps"] == pytest.approx(5100.0)
    assert mapping.averages["max_hit"] == pytest.approx(4550.0)
    assert mapping.gate_pass_rate == pytest.approx(0.5)

    pinnacle = summaries["pinnacle"]
    assert pinnacle.sample_count == 1
    assert pinnacle.gate_pass_rate is None


def test_render_markdown_report_includes_metrics_and_rates() -> None:
    old_summary = drift.ScenarioMetricsSummary(
        scenario_id="mapping_t16",
        sample_count=1,
        averages={"full_dps": 5000.0, "max_hit": 4500.0, "utility_score": 1.2},
        gate_pass_rate=1.0,
    )
    new_summary = drift.ScenarioMetricsSummary(
        scenario_id="mapping_t16",
        sample_count=2,
        averages={"full_dps": 5200.0, "max_hit": 4700.0, "utility_score": 1.3},
        gate_pass_rate=0.75,
    )
    comparisons = drift.build_comparisons(
        {"mapping_t16": old_summary}, {"mapping_t16": new_summary}
    )
    timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
    report = drift.render_markdown_report(
        comparisons,
        old_ruleset="pob:old",
        new_ruleset="pob:new",
        run_id="drift-run",
        generated_at=timestamp,
    )

    assert "mapping_t16" in report
    assert "full_dps" in report
    assert "+200.0" in report
    assert "-25.0pp" in report


def test_generate_drift_report_writes_file(tmp_path: Path) -> None:
    class FakeProvider:
        def __init__(self, data: dict[str, list[dict[str, Any]]]):
            self._data = data

        def fetch_rows(self, ruleset_id: str) -> list[dict[str, Any]]:
            return self._data.get(ruleset_id, [])

    data = {
        "old": [
            {
                "scenario_id": "mapping_t16",
                "full_dps": 5000,
                "max_hit": 4500,
                "utility_score": 1.0,
                "gate_pass": 1,
            },
        ],
        "new": [
            {
                "scenario_id": "mapping_t16",
                "full_dps": 5100,
                "max_hit": 4550,
                "utility_score": 1.1,
                "gate_pass": 1,
            },
            {"scenario_id": "pinnacle", "full_dps": 6000, "max_hit": 5900, "utility_score": 0.9},
        ],
    }
    provider = FakeProvider(data)

    output_path = drift.generate_drift_report(
        run_id="drift-writer",
        old_ruleset_id="old",
        new_ruleset_id="new",
        provider=provider,
        base_path=tmp_path,
    )

    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "mapping_t16" in content
    assert "pinnacle" in content
    assert "old n=1" in content
    assert "new n=1" in content
    assert "old n=0" in content
