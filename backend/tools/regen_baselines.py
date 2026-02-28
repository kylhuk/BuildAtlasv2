"""Regenerate deterministic regression baselines from fixture metrics."""

from __future__ import annotations

import json
from pathlib import Path

BASELINE_VERSION = "EP-V1-03"
REQUIRED_METRICS = ("full_dps", "max_hit")
REGEN_COMMAND = "python -m backend.tools.regen_baselines"

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
RULESET_MAP_PATH = FIXTURES_DIR / "regression_rulesets.json"
BASELINE_PATH = FIXTURES_DIR / "regression_baselines.json"


def _load_ruleset_map() -> dict[str, str]:
    if not RULESET_MAP_PATH.exists():
        raise SystemExit("Missing regression ruleset map; create regression_rulesets.json")
    raw = json.loads(RULESET_MAP_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit("ruleset map must be a JSON object mapping scenario_id -> ruleset_id")
    return {str(k): str(v) for k, v in raw.items()}


def _collect_metrics() -> dict[str, dict[str, dict[str, float]]]:
    metrics_files = sorted(FIXTURES_DIR.glob("*_metrics.json"))
    if not metrics_files:
        raise SystemExit("No fixture metrics files found in fixtures directory")

    scenario_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for metrics_file in metrics_files:
        raw = json.loads(metrics_file.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise SystemExit(f"{metrics_file.name} must contain a JSON object at root")
        for scenario_id in sorted(raw):
            payload = raw[scenario_id]
            if not isinstance(payload, dict):
                raise SystemExit(f"scenario {scenario_id} in {metrics_file.name} must be an object")
            metrics = payload.get("metrics")
            if not isinstance(metrics, dict):
                raise SystemExit(f"missing metrics for {scenario_id} in {metrics_file.name}")
            bucket = scenario_metrics.setdefault(scenario_id, {})
            for metric_name in sorted(metrics):
                metric_value = metrics[metric_name]
                try:
                    metric_value = float(metric_value)
                except (TypeError, ValueError):
                    raise SystemExit(
                        f"metric {metric_name} for {scenario_id} is not numeric"
                    ) from None
                metric_stats = bucket.setdefault(
                    metric_name, {"min": metric_value, "max": metric_value}
                )
                metric_stats["min"] = min(metric_stats["min"], metric_value)
                metric_stats["max"] = max(metric_stats["max"], metric_value)
    return scenario_metrics


def _build_baseline(
    scenario_metrics: dict[str, dict[str, dict[str, float]]],
    rulesets: dict[str, str],
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for scenario_id in sorted(scenario_metrics):
        if scenario_id not in rulesets:
            raise SystemExit(
                (
                    "No ruleset_id configured for scenario "
                    f"{scenario_id}; update regression_rulesets.json"
                )
            )
        metrics_for_scenario = scenario_metrics[scenario_id]
        missing = [m for m in REQUIRED_METRICS if m not in metrics_for_scenario]
        if missing:
            raise SystemExit(
                f"Scenario {scenario_id} missing required metrics: {', '.join(missing)}"
            )
        sorted_metrics: dict[str, dict[str, float]] = {}
        for metric_name in sorted(metrics_for_scenario):
            metric_stats = metrics_for_scenario[metric_name]
            sorted_metrics[metric_name] = {
                "min": float(metric_stats["min"]),
                "max": float(metric_stats["max"]),
            }
        entries.append(
            {
                "scenario_id": scenario_id,
                "build_id": scenario_id,
                "ruleset_id": rulesets[scenario_id],
                "metrics": sorted_metrics,
            }
        )
    return entries


def main() -> None:
    rulesets = _load_ruleset_map()
    scenario_metrics = _collect_metrics()
    baseline_entries = _build_baseline(scenario_metrics, rulesets)
    baseline_payload = {
        "version": BASELINE_VERSION,
        "scenarios": baseline_entries,
    }
    BASELINE_PATH.write_text(json.dumps(baseline_payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote regression baseline to {BASELINE_PATH}")


if __name__ == "__main__":
    main()
