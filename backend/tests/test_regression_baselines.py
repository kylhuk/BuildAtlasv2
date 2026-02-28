from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
BASELINE_PATH = FIXTURES_DIR / "regression_baselines.json"
RULESET_MAP_PATH = FIXTURES_DIR / "regression_rulesets.json"
REGEN_COMMAND = "python -m backend.tools.regen_baselines"
REQUIRED_METRICS = ("full_dps", "max_hit", "utility_score")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_baseline() -> dict:
    return _load_json(BASELINE_PATH)


def _load_rulesets() -> dict[str, str]:
    return _load_json(RULESET_MAP_PATH)


def _load_fixture_metrics(scenario_id: str) -> dict[str, float]:
    fixture_path = FIXTURES_DIR / f"{scenario_id}_metrics.json"
    data = _load_json(fixture_path)
    scenario_data = data.get(scenario_id)
    if not isinstance(scenario_data, dict):
        raise AssertionError(f"missing metrics payload for {scenario_id}")
    metrics = scenario_data.get("metrics")
    if not isinstance(metrics, dict):
        raise AssertionError(f"metrics block missing for {scenario_id}")
    return {name: float(value) for name, value in metrics.items()}


def _format_metric_drift(
    scenario_id: str,
    metric_name: str,
    actual: float,
    min_value: float,
    max_value: float,
) -> str:
    if actual < min_value:
        bound = min_value
        direction = "below"
    else:
        bound = max_value
        direction = "above"
    return (
        f"{scenario_id} {metric_name} {actual:.1f} {direction} baseline {bound:.1f}"
        f" by {abs(actual - bound):.1f}"
    )


def _assert_ruleset_locked(
    scenario_id: str, expected: str, actual: str, *, regen_cmd: str = REGEN_COMMAND
) -> None:
    if expected != actual:
        raise AssertionError(
            (
                f"{scenario_id} ruleset_id mismatch: expected {expected!r}, got {actual!r}; "
                f"run {regen_cmd}"
            )
        )


def test_fixture_metrics_within_baseline_ranges() -> None:
    baseline = _load_baseline()
    rulesets = _load_rulesets()
    scenarios = {entry["scenario_id"]: entry for entry in baseline["scenarios"]}
    assert set(scenarios) == set(rulesets)
    for scenario_id, entry in scenarios.items():
        _assert_ruleset_locked(scenario_id, entry["ruleset_id"], rulesets[scenario_id])
        metrics = _load_fixture_metrics(scenario_id)
        for metric_name in REQUIRED_METRICS:
            if metric_name not in entry["metrics"]:
                raise AssertionError(f"baseline missing {metric_name} for {scenario_id}")
        for metric_name, range_info in entry["metrics"].items():
            if metric_name not in metrics:
                raise AssertionError(f"fixture missing {metric_name} for {scenario_id}")
            actual_value = metrics[metric_name]
            min_value = float(range_info["min"])
            max_value = float(range_info["max"])
            if actual_value < min_value or actual_value > max_value:
                drift_message = _format_metric_drift(
                    scenario_id,
                    metric_name,
                    actual_value,
                    min_value,
                    max_value,
                )
                raise AssertionError((f"{drift_message}; run {REGEN_COMMAND}"))


def test_metric_drift_formatting_is_compact() -> None:
    msg = _format_metric_drift("mapping_t16", "full_dps", 4800.0, 5000.0, 5200.0)
    assert msg == "mapping_t16 full_dps 4800.0 below baseline 5000.0 by 200.0"
    msg = _format_metric_drift("uber_pinnacle", "max_hit", 3600.0, 3200.0, 3400.0)
    assert msg == "uber_pinnacle max_hit 3600.0 above baseline 3400.0 by 200.0"


def test_ruleset_lock_message_recommends_regen() -> None:
    with pytest.raises(AssertionError) as excinfo:
        _assert_ruleset_locked("mapping_t16", "expected", "actual")
    assert REGEN_COMMAND in str(excinfo.value)
