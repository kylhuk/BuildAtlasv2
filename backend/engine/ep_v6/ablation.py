from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from backend.engine.ep_v6.signatures import DEFAULT_METRIC_BASES

OperatorRow = Dict[str, Any]

OPERATOR_NAMES: List[str] = [
    "support_remove",
    "support_swap",
    "passive_cluster_remove",
    "item_tier_downgrade",
]


def _clamp(value: float, minimum: float = 0.0) -> float:
    return round(max(value, minimum), 2)


def _support_remove(metrics: Mapping[str, float]) -> Dict[str, float]:
    return {
        **metrics,
        "support_coverage": _clamp(metrics["support_coverage"] - 1.0),
        "full_dps": _clamp(metrics["full_dps"] * 0.92),
        "max_hit": _clamp(metrics["max_hit"] * 0.95),
    }


def _support_swap(metrics: Mapping[str, float]) -> Dict[str, float]:
    return {
        **metrics,
        "support_coverage": _clamp(metrics["support_coverage"] - 0.4),
        "full_dps": _clamp(metrics["full_dps"] * 0.98),
        "max_hit": _clamp(metrics["max_hit"] * 1.02),
    }


def _passive_cluster_remove(metrics: Mapping[str, float]) -> Dict[str, float]:
    return {
        **metrics,
        "passive_clusters": _clamp(metrics["passive_clusters"] - 1.0),
        "full_dps": _clamp(metrics["full_dps"] * 0.96),
        "max_hit": _clamp(metrics["max_hit"] * 0.97),
    }


def _item_tier_downgrade(metrics: Mapping[str, float]) -> Dict[str, float]:
    return {
        **metrics,
        "item_tier": _clamp(metrics["item_tier"] - 1.0),
        "full_dps": _clamp(metrics["full_dps"] * 0.94),
        "max_hit": _clamp(metrics["max_hit"] * 0.9),
    }


_OPERATOR_DISPATCH = {
    "support_remove": _support_remove,
    "support_swap": _support_swap,
    "passive_cluster_remove": _passive_cluster_remove,
    "item_tier_downgrade": _item_tier_downgrade,
}


def generate_ablation_rows(
    signature: Mapping[str, Any], operators: Sequence[str] | None = None
) -> List[OperatorRow]:
    baseline: Dict[str, float] = {
        key: float(value)
        for key, value in signature.get("metrics", {}).items()
        if isinstance(value, (int, float))
    }
    if not baseline:
        baseline = {k: float(v) for k, v in DEFAULT_METRIC_BASES.items()}

    operators_to_run = list(operators or OPERATOR_NAMES)
    rows: List[OperatorRow] = []
    for operator in operators_to_run:
        variant_fn = _OPERATOR_DISPATCH.get(operator)
        if not variant_fn:
            continue
        variant = variant_fn(baseline)
        delta = {k: round(variant[k] - baseline[k], 2) for k in baseline}
        rows.append(
            {
                "operator": operator,
                "baseline": baseline,
                "variant": variant,
                "delta": delta,
                "metadata": {
                    "signature_key": signature.get("key"),
                    "ruleset_id": signature.get("ruleset_id"),
                    "scenario_id": signature.get("scenario_id"),
                    "skill_package_id": signature.get("skill_package_id"),
                    "signature_probe": signature.get("probe_version"),
                },
            }
        )
    return rows


def write_ndjson(rows: Iterable[OperatorRow], path: Path | str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True))
            fh.write("\n")
