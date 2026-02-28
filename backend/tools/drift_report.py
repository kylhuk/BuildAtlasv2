"""Generate a markdown drift report comparing scenario metrics for two rulesets."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from backend.app.db.ch import ClickhouseRepository
from backend.app.settings import settings

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
DEFAULT_METRIC_KEYS = ("full_dps", "max_hit", "utility_score")


@dataclass(frozen=True)
class ScenarioMetricsSummary:
    scenario_id: str
    sample_count: int
    averages: Mapping[str, float | None]
    gate_pass_rate: float | None


@dataclass(frozen=True)
class ScenarioComparison:
    scenario_id: str
    old: ScenarioMetricsSummary | None
    new: ScenarioMetricsSummary | None


class ScenarioMetricsProvider(Protocol):
    def fetch_rows(self, ruleset_id: str) -> Sequence[Mapping[str, Any]]: ...


class ClickhouseScenarioMetricsProvider:
    def __init__(self, repository: ClickhouseRepository | None = None) -> None:
        self._repository = repository or ClickhouseRepository()

    def fetch_rows(self, ruleset_id: str) -> Sequence[Mapping[str, Any]]:
        return self._repository.fetch_scenario_metric_rows(ruleset_id)


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def aggregate_scenario_metrics(
    rows: Sequence[Mapping[str, Any]],
    metric_keys: Sequence[str] = DEFAULT_METRIC_KEYS,
) -> dict[str, ScenarioMetricsSummary]:
    active_metric_keys = tuple(dict.fromkeys(str(metric) for metric in metric_keys if metric))
    buckets: dict[str, dict[str, float | int]] = {}
    for row in rows:
        scenario_id = row.get("scenario_id")
        if not scenario_id:
            continue
        key = str(scenario_id)
        bucket = buckets.setdefault(
            key,
            {
                "count": 0,
                "gate_pass_count": 0,
                "gate_pass_true": 0,
            },
        )
        bucket["count"] += 1
        for metric in active_metric_keys:
            bucket.setdefault(f"{metric}_sum", 0.0)
            bucket.setdefault(f"{metric}_count", 0)
            value = _coerce_float(row.get(metric))
            if value is not None:
                bucket[f"{metric}_sum"] += value
                bucket[f"{metric}_count"] += 1
        gate_pass = row.get("gate_pass")
        if gate_pass is not None:
            bucket["gate_pass_count"] += 1
            if bool(gate_pass):
                bucket["gate_pass_true"] += 1
    summaries: dict[str, ScenarioMetricsSummary] = {}
    for scenario_id, bucket in buckets.items():
        count = int(bucket["count"])
        if count == 0:
            continue
        averages: dict[str, float | None] = {}
        for metric in active_metric_keys:
            metric_count = int(bucket.get(f"{metric}_count", 0))
            total = float(bucket.get(f"{metric}_sum", 0.0))
            averages[metric] = total / metric_count if metric_count else None
        gate_rate = (
            bucket["gate_pass_true"] / bucket["gate_pass_count"]
            if bucket["gate_pass_count"]
            else None
        )
        summaries[scenario_id] = ScenarioMetricsSummary(
            scenario_id=scenario_id,
            sample_count=count,
            averages=averages,
            gate_pass_rate=gate_rate,
        )
    return summaries


def build_comparisons(
    old: dict[str, ScenarioMetricsSummary],
    new: dict[str, ScenarioMetricsSummary],
) -> list[ScenarioComparison]:
    scenario_ids = sorted(set(old) | set(new))
    return [
        ScenarioComparison(
            scenario_id=scenario_id,
            old=old.get(scenario_id),
            new=new.get(scenario_id),
        )
        for scenario_id in scenario_ids
    ]


def _metric_value(summary: ScenarioMetricsSummary | None, metric: str) -> float | None:
    if not summary:
        return None
    return summary.averages.get(metric)


def _compute_delta(old: float | None, new: float | None) -> float | None:
    if old is None or new is None:
        return None
    return new - old


def _format_metric_value(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1f}"


def _format_delta(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value:+.1f}{suffix}"


def _format_rate(rate: float | None) -> str:
    return "N/A" if rate is None else f"{rate * 100:.1f}%"


def render_markdown_report(
    comparisons: list[ScenarioComparison],
    old_ruleset: str,
    new_ruleset: str,
    run_id: str,
    generated_at: datetime,
    metric_keys: Sequence[str] = DEFAULT_METRIC_KEYS,
) -> str:
    active_metric_keys = tuple(dict.fromkeys(str(metric) for metric in metric_keys if metric))
    lines: list[str] = [
        "# Drift Report",
        "",
        f"- Run ID: {run_id}",
        f"- Old ruleset: {old_ruleset}",
        f"- New ruleset: {new_ruleset}",
        f"- Generated: {generated_at.isoformat()}",
        "",
    ]
    if not comparisons:
        lines.append("No scenario metrics were available for either ruleset.")
        lines.append("")
        return "\n".join(lines)
    for comparison in comparisons:
        old_count = comparison.old.sample_count if comparison.old else 0
        new_count = comparison.new.sample_count if comparison.new else 0
        lines.append(f"## {comparison.scenario_id} (old n={old_count}, new n={new_count})")
        lines.append("")
        lines.append("| Metric | Old | New | Δ |")
        lines.append("| --- | --- | --- | --- |")
        for metric in active_metric_keys:
            old_value = _metric_value(comparison.old, metric)
            new_value = _metric_value(comparison.new, metric)
            delta = _compute_delta(old_value, new_value)
            lines.append(
                "| {metric} | {old} | {new} | {delta} |".format(
                    metric=metric,
                    old=_format_metric_value(old_value),
                    new=_format_metric_value(new_value),
                    delta=_format_delta(delta),
                )
            )
        old_rate = comparison.old.gate_pass_rate if comparison.old else None
        new_rate = comparison.new.gate_pass_rate if comparison.new else None
        rate_delta = _compute_delta(old_rate, new_rate)
        rate_delta_display = rate_delta * 100 if rate_delta is not None else None
        lines.append(
            "| gate_pass_rate | {old} | {new} | {delta} |".format(
                old=_format_rate(old_rate),
                new=_format_rate(new_rate),
                delta=_format_delta(rate_delta_display, suffix="pp"),
            )
        )
        lines.append("")
    return "\n".join(lines)


def _validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_' or '-'."
        )
    return run_id


def drift_report_path(run_id: str, base_path: Path | None = None) -> Path:
    safe_run_id = _validate_run_id(run_id)
    runs_root = (Path(base_path or settings.data_path) / "runs").resolve()
    report_path = (runs_root / safe_run_id / "drift_report.md").resolve()
    if runs_root not in report_path.parents:
        raise ValueError("run_id resolved outside run directory")
    return report_path


def generate_drift_report(
    run_id: str,
    old_ruleset_id: str,
    new_ruleset_id: str,
    provider: ScenarioMetricsProvider,
    base_path: Path | None = None,
    generated_at: datetime | None = None,
    metric_keys: Sequence[str] = DEFAULT_METRIC_KEYS,
) -> Path:
    active_metric_keys = tuple(dict.fromkeys(str(metric) for metric in metric_keys if metric))
    base_path = Path(base_path) if base_path is not None else None
    rows_old = provider.fetch_rows(old_ruleset_id)
    rows_new = provider.fetch_rows(new_ruleset_id)
    old_summary = aggregate_scenario_metrics(rows_old, metric_keys=active_metric_keys)
    new_summary = aggregate_scenario_metrics(rows_new, metric_keys=active_metric_keys)
    comparisons = build_comparisons(old_summary, new_summary)
    timestamp = generated_at or datetime.now(timezone.utc)
    report_text = render_markdown_report(
        comparisons,
        old_ruleset_id,
        new_ruleset_id,
        run_id,
        timestamp,
        metric_keys=active_metric_keys,
    )
    path = drift_report_path(run_id, base_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_text, encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write a drift report comparing scenario metrics between rulesets."
    )
    parser.add_argument("run_id", help="Run identifier for the report output")
    parser.add_argument("--old-ruleset", required=True, help="Base ruleset id")
    parser.add_argument("--new-ruleset", required=True, help="Target ruleset id")
    parser.add_argument(
        "--base-path",
        help="Optional root data path (defaults to backend settings)",
    )
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        help="Metric key to include in drift calculations (repeatable)",
    )
    args = parser.parse_args()
    base_path = Path(args.base_path) if args.base_path else None
    metric_keys = tuple(args.metrics) if args.metrics else DEFAULT_METRIC_KEYS
    provider = ClickhouseScenarioMetricsProvider()
    path = generate_drift_report(
        args.run_id,
        args.old_ruleset,
        args.new_ruleset,
        provider,
        base_path=base_path,
        metric_keys=metric_keys,
    )
    print(f"Wrote drift report to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
