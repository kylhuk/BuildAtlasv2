from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Mapping, Sequence

ConstraintOperator = Literal["<", "<=", ">", ">=", "==", "!="]
ConstraintStatus = Literal["pass", "fail", "unknown"]


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


_COMPARATORS: Mapping[ConstraintOperator, Callable[[float, float], bool]] = {
    "<": lambda left, right: left < right,
    "<=": lambda left, right: left <= right,
    ">": lambda left, right: left > right,
    ">=": lambda left, right: left >= right,
    "==": lambda left, right: left == right,
    "!=": lambda left, right: left != right,
}


def _normalize_metric_path(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        parts = [segment.strip() for segment in value.split(".") if segment.strip()]
        if not parts:
            raise ValueError("metric_path must include at least one segment")
        return tuple(parts)
    if isinstance(value, Sequence):
        if isinstance(value, str):
            raise ValueError("metric_path string must not be passed as a sequence here")
        parts: list[str] = []
        for segment in value:
            if segment is None:
                continue
            candidate = str(segment).strip()
            if candidate:
                parts.append(candidate)
        if not parts:
            raise ValueError("metric_path must include at least one segment")
        return tuple(parts)
    raise ValueError("metric_path must be a string or a string sequence")


@dataclass(frozen=True)
class ConstraintRule:
    code: str
    metric_path: tuple[str, ...]
    operator: ConstraintOperator
    threshold: float
    scenario_id: str | None
    description: str | None
    reason_code: str | None
    missing_data_reason: str | None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ConstraintRule":
        if not isinstance(payload, Mapping):
            raise ValueError("constraint entries must be mappings")
        raw_code = payload.get("code") or payload.get("id")
        if not raw_code:
            raise ValueError("constraint code is required")
        code = str(raw_code)
        operator = str(payload.get("operator") or payload.get("comparison") or "<=")
        if operator not in _COMPARATORS:
            raise ValueError(f"unsupported constraint operator {operator!r}")
        threshold_value = payload.get("threshold")
        if threshold_value is None:
            raise ValueError("constraint threshold is required")
        metric_path_value = payload.get("metric_path") or payload.get("path")
        if metric_path_value is None:
            raise ValueError("constraint metric_path is required")
        metric_path = _normalize_metric_path(metric_path_value)
        threshold = float(threshold_value)

        scenario_id = payload.get("scenario_id")
        if scenario_id is not None:
            scenario_id = str(scenario_id)
            if scenario_id.strip() == "":
                scenario_id = None

        description = payload.get("description")
        if description is not None:
            description = str(description)
        reason_code = payload.get("reason_code")
        if reason_code is not None:
            reason_code = str(reason_code)
        missing_data_reason = payload.get("missing_data_reason")
        if missing_data_reason is not None:
            missing_data_reason = str(missing_data_reason)

        return cls(
            code=code,
            metric_path=metric_path,
            operator=operator,  # type: ignore[arg-type]
            threshold=threshold,
            scenario_id=scenario_id,
            description=description,
            reason_code=reason_code,
            missing_data_reason=missing_data_reason,
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "metric_path": list(self.metric_path),
            "operator": self.operator,
            "threshold": self.threshold,
        }
        if self.scenario_id:
            payload["scenario_id"] = self.scenario_id
        if self.description:
            payload["description"] = self.description
        if self.reason_code:
            payload["reason_code"] = self.reason_code
        if self.missing_data_reason:
            payload["missing_data_reason"] = self.missing_data_reason
        return payload


@dataclass(frozen=True)
class ConstraintSpec:
    schema_version: int
    rules: tuple[ConstraintRule, ...]
    payload: dict[str, Any]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "ConstraintSpec" | None:
        if not payload:
            return None
        if not isinstance(payload, Mapping):
            raise ValueError("constraints payload must be a mapping")
        schema_version_value = payload.get("schema_version")
        schema_version = int(schema_version_value) if schema_version_value is not None else 1
        raw_rules = payload.get("rules") or payload.get("constraints")
        if raw_rules is None:
            raise ValueError("constraints.rules is required")
        if isinstance(raw_rules, str) or not isinstance(raw_rules, Sequence):
            raise ValueError("constraints.rules must be a sequence")
        rules: list[ConstraintRule] = []
        for entry in raw_rules:
            if entry is None:
                continue
            rule = ConstraintRule.from_payload(entry)
            rules.append(rule)
        if not rules:
            return None
        spec_payload: dict[str, Any] = dict(payload)
        spec_payload["schema_version"] = schema_version
        spec_payload["rules"] = [rule.to_payload() for rule in rules]
        return cls(
            schema_version=schema_version,
            rules=tuple(rules),
            payload=spec_payload,
        )

    def to_payload(self) -> dict[str, Any]:
        return copy.deepcopy(self.payload)


@dataclass(frozen=True)
class ConstraintEvaluationDetail:
    code: str
    operator: ConstraintOperator
    threshold: float
    value: float | None
    scenario_id: str | None
    description: str | None
    satisfied: bool | None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "operator": self.operator,
            "threshold": self.threshold,
            "value": self.value,
            "satisfied": self.satisfied,
        }
        if self.scenario_id:
            payload["scenario_id"] = self.scenario_id
        if self.description:
            payload["description"] = self.description
        return payload


@dataclass(frozen=True)
class ConstraintEvaluation:
    status: ConstraintStatus
    reason_code: str
    violated_constraints: tuple[str, ...]
    checked_at: str
    details: tuple[ConstraintEvaluationDetail, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason_code": self.reason_code,
            "violated_constraints": list(self.violated_constraints),
            "checked_at": self.checked_at,
            "details": [detail.to_payload() for detail in self.details],
        }


def _resolve_value_for_rule(
    metrics_payload: Mapping[str, Any],
    rule: ConstraintRule,
) -> float | None:
    candidates: list[Any] = []
    if rule.scenario_id:
        entry = metrics_payload.get(rule.scenario_id)
        if entry is not None:
            candidates.append(entry)
    else:
        for entry in metrics_payload.values():
            if isinstance(entry, Mapping):
                candidates.append(entry)
    for entry in candidates:
        value: Any = entry
        for segment in rule.metric_path:
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(segment)
        if value is not None:
            coerced = _coerce_float(value)
            if coerced is not None:
                return coerced
    return None


def evaluate_constraints(
    metrics_payload: Mapping[str, Any] | None,
    spec: ConstraintSpec,
) -> ConstraintEvaluation:
    resolution = metrics_payload or {}
    details: list[ConstraintEvaluationDetail] = []
    violated: list[str] = []
    missing_data = False
    for rule in spec.rules:
        value = _resolve_value_for_rule(resolution, rule)
        satisfied: bool | None = None
        if value is not None:
            comparator = _COMPARATORS[rule.operator]
            satisfied = comparator(value, rule.threshold)
            if not satisfied:
                violated.append(rule.code)
        else:
            missing_data = True
        details.append(
            ConstraintEvaluationDetail(
                code=rule.code,
                operator=rule.operator,
                threshold=rule.threshold,
                value=value,
                scenario_id=rule.scenario_id,
                description=rule.description,
                satisfied=satisfied,
            )
        )
    if violated:
        status: ConstraintStatus = "fail"
        reason_code = "constraint_violation"
        for rule in spec.rules:
            if rule.code in violated and rule.reason_code:
                reason_code = rule.reason_code
                break
    elif missing_data:
        status = "unknown"
        reason_code = "constraint_data_missing"
        for rule in spec.rules:
            if rule.missing_data_reason:
                reason_code = rule.missing_data_reason
                break
    else:
        status = "pass"
        reason_code = "constraints_met"
        for rule in spec.rules:
            if rule.reason_code:
                reason_code = rule.reason_code
                break
    checked_at = datetime.now(timezone.utc).isoformat()
    return ConstraintEvaluation(
        status=status,
        reason_code=reason_code,
        violated_constraints=tuple(violated),
        checked_at=checked_at,
        details=tuple(details),
    )


def constraint_artifact_payload(
    spec: ConstraintSpec,
    evaluation: ConstraintEvaluation,
) -> dict[str, Any]:
    return {
        "schema_version": spec.schema_version,
        "spec": spec.to_payload(),
        "evaluation": evaluation.to_payload(),
    }


__all__ = [
    "ConstraintSpec",
    "ConstraintEvaluation",
    "evaluate_constraints",
    "constraint_artifact_payload",
]
