from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from backend.engine.ep_v6 import surrogate

DEFAULT_OPERATORS: List[str] = [
    "support_remove",
    "support_swap",
    "passive_cluster_remove",
    "item_tier_downgrade",
]


def score_operators(
    model: Mapping[str, Any],
    features: Mapping[str, float],
    operators: Sequence[str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    operators_to_score = list(operators or DEFAULT_OPERATORS)
    scored: Dict[str, Dict[str, Any]] = {}
    for operator in operators_to_score:
        candidate = surrogate.infer(model, operator, features)
        score = float(candidate.get("delta", {}).get("full_dps", 0.0))
        scored[operator] = {
            "prediction": candidate,
            "score": score,
            "uncertainty": candidate.get("uncertainty"),
        }
    return scored


def repair_candidate(
    features: Mapping[str, float], constraints: Mapping[str, bool]
) -> Dict[str, Any]:
    repaired = {k: float(v) for k, v in features.items()}
    reasons: List[str] = []
    if not constraints.get("resists_ok", True):
        repaired["resists"] = 0.0
        reasons.append("resists_removed")
    if not constraints.get("attributes_ok", True):
        repaired["attributes"] = repaired.get("attributes", 0.0) - 10.0
        reasons.append("attributes_rebalanced")
    if not constraints.get("reservation_ok", True):
        repaired["reservation"] = 0.0
        reasons.append("reservation_released")
    return {
        "features": repaired,
        "repair_reasons": reasons,
    }


def select_mutation(
    model: Mapping[str, Any],
    features: Mapping[str, float],
    constraints: Mapping[str, bool],
    operators: Sequence[str] | None = None,
) -> Dict[str, Any]:
    scored = score_operators(model, features, operators)
    best_operator = max(scored.items(), key=lambda item: item[1]["score"])[0]
    repair = repair_candidate(features, constraints)
    return {
        "selected_operator": best_operator,
        "prediction": scored[best_operator]["prediction"],
        "scores": scored,
        "repaired_features": repair["features"],
        "repair_reasons": repair["repair_reasons"],
    }
