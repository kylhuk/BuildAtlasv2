from __future__ import annotations

import base64
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from backend.app.api.errors import APIError
from backend.app.api.models import BuildStatus
from backend.app.db.ch import ClickhouseRepository, ScenarioMetricRow
from backend.engine.artifacts.store import read_build_artifacts, write_build_artifacts
from backend.engine.evaluation.gates import evaluate_gates
from backend.engine.evaluation.normalized import NormalizedMetrics, map_worker_output
from backend.engine.scenarios.loader import list_templates
from backend.engine.worker_pool import WorkerPool, WorkerPoolError

_ALLOWED_STATUS_TRANSITIONS: Dict[BuildStatus, set[BuildStatus]] = {
    BuildStatus.imported: {BuildStatus.queued},
    BuildStatus.queued: {BuildStatus.evaluated, BuildStatus.failed},
    BuildStatus.evaluated: {BuildStatus.queued},
    BuildStatus.failed: {BuildStatus.queued},
}


class BuildEvaluator:
    def __init__(
        self,
        repo: ClickhouseRepository,
        base_path: Path,
        *,
        profiles_requiring_non_stub_metrics: Sequence[str] | None = None,
        profiles_requiring_worker_metrics: Sequence[str] | None = None,
    ) -> None:
        self._repo = repo
        self._base_path = base_path
        self._profiles_requiring_non_stub_metrics: set[str] = set()
        if profiles_requiring_non_stub_metrics:
            self.register_profiles_requiring_non_stub_metrics(profiles_requiring_non_stub_metrics)
        self._profiles_requiring_worker_metrics: set[str] = set()
        if profiles_requiring_worker_metrics:
            self.register_profiles_requiring_worker_metrics(profiles_requiring_worker_metrics)

    def evaluate_build(self, build_id: str) -> tuple[BuildStatus, List[ScenarioMetricRow]]:
        build = self._repo.get_build(build_id)
        if not build:
            raise APIError(404, "build_not_found", f"build {build_id} not found")
        status_value = build.get("status")
        try:
            current_status = BuildStatus(status_value)
        except ValueError:
            current_status = BuildStatus.imported
        self._ensure_transition(current_status, BuildStatus.queued)
        self._repo.update_build_status(build_id, BuildStatus.queued.value)
        try:
            scenario_rows = self._collect_scenario_rows(build)
            final_status = (
                BuildStatus.evaluated
                if all(row.gate_pass for row in scenario_rows)
                else BuildStatus.failed
            )
            self._repo.insert_scenario_metrics(scenario_rows)
            self._repo.update_build_status(build_id, final_status.value)
            return final_status, scenario_rows
        except APIError:
            self._repo.update_build_status(build_id, BuildStatus.failed.value)
            raise
        except Exception as exc:  # pragma: no cover - unexpected
            self._repo.update_build_status(build_id, BuildStatus.failed.value)
            raise APIError(
                500,
                "evaluation_error",
                "failed to evaluate build",
                details=str(exc),
            ) from exc

    def _collect_scenario_rows(self, build: dict[str, Any]) -> List[ScenarioMetricRow]:
        build_id = build.get("build_id")
        if not build_id:
            raise APIError(400, "invalid_build", "build record missing build_id")
        try:
            artifacts = read_build_artifacts(build_id, base_path=self._base_path)
        except FileNotFoundError as exc:
            raise APIError(404, "artifacts_missing", f"artifacts missing for {build_id}") from exc
        profile_id = build.get("profile_id")
        if not profile_id:
            raise APIError(400, "missing_profile", "build profile_id missing")
        templates = [template for template in list_templates() if template.profile_id == profile_id]
        if not templates:
            raise APIError(
                400,
                "no_scenarios",
                f"no scenario templates configured for profile {profile_id}",
            )
        raw_metrics_map = self._resolve_raw_metrics(build_id, build, artifacts, templates)
        if not raw_metrics_map:
            raise APIError(400, "missing_metrics", "raw metrics payload missing for build")

        scenarios_used = []
        for template in templates:
            if hasattr(template.gate_thresholds, "model_dump"):
                gate_thresholds = template.gate_thresholds.model_dump(mode="json")
            elif hasattr(template.gate_thresholds, "dict"):
                gate_thresholds = template.gate_thresholds.dict()
            elif is_dataclass(template.gate_thresholds):
                gate_thresholds = asdict(template.gate_thresholds)
            else:
                gate_thresholds = dict(template.gate_thresholds)
            scenarios_used.append(
                {
                    "scenario_id": template.scenario_id,
                    "version": template.version,
                    "profile_id": template.profile_id,
                    "pob_config": template.pob_config,
                    "gate_thresholds": gate_thresholds,
                }
            )
        write_build_artifacts(
            build_id,
            xml=artifacts.xml,
            code=artifacts.code,
            genome=artifacts.genome,
            scenarios_used=scenarios_used,
            raw_metrics=raw_metrics_map,
            base_path=self._base_path,
        )

        evaluated_at = datetime.now(timezone.utc)
        scenario_rows: List[ScenarioMetricRow] = []
        ruleset_id = build.get("ruleset_id") or ""
        for template in templates:
            payload = raw_metrics_map.get(template.scenario_id)
            if payload is None:
                raise APIError(
                    400,
                    "missing_metrics",
                    f"raw metrics not found for scenario {template.scenario_id}",
                )
            normalized = map_worker_output(payload)
            normalized = adjust_metrics_for_profile(template.profile_id, normalized)
            gate_eval = evaluate_gates(normalized, template.gate_thresholds)
            scenario_rows.append(
                ScenarioMetricRow(
                    build_id=build_id,
                    ruleset_id=ruleset_id,
                    scenario_id=template.scenario_id,
                    gate_pass=gate_eval.gate_pass,
                    gate_fail_reasons=list(gate_eval.gate_fail_reasons),
                    pob_warnings=list(normalized.warnings),
                    evaluated_at=evaluated_at,
                    full_dps=normalized.full_dps,
                    max_hit=normalized.max_hit,
                    armour=normalized.armour,
                    evasion=normalized.evasion,
                    life=normalized.life,
                    mana=normalized.mana,
                    utility_score=normalized.utility_score,
                )
            )
        if not scenario_rows:
            raise APIError(400, "no_scenarios", "no scenario rows generated for build")
        return scenario_rows

    def _map_raw_metrics(self, raw_metrics: Any) -> dict[str, Any]:
        if isinstance(raw_metrics, Mapping):
            return dict(raw_metrics)
        if isinstance(raw_metrics, list):
            mapped: dict[str, Any] = {}
            for entry in raw_metrics:
                if not isinstance(entry, Mapping):
                    continue
                scenario_id = entry.get("scenario_id")
                if not isinstance(scenario_id, str):
                    continue
                payload = entry.get("payload") or entry
                mapped[scenario_id] = payload
            return mapped
        return {}

    def register_profiles_requiring_non_stub_metrics(
        self,
        profile_ids: Sequence[str] | str | None,
    ) -> None:
        if not profile_ids:
            return
        if isinstance(profile_ids, str):
            profile_ids = (profile_ids,)
        for profile_id in profile_ids:
            normalized = self._normalize_profile_id(profile_id)
            if normalized:
                self._profiles_requiring_non_stub_metrics.add(normalized)

    def require_non_stub_metrics_for_profile(self, profile_id: str) -> None:
        self.register_profiles_requiring_non_stub_metrics(profile_id)

    def _profile_requires_non_stub_metrics(self, profile_id: str | None) -> bool:
        normalized = self._normalize_profile_id(profile_id)
        return bool(normalized and normalized in self._profiles_requiring_non_stub_metrics)

    def register_profiles_requiring_worker_metrics(
        self,
        profile_ids: Sequence[str] | str | None,
    ) -> None:
        if not profile_ids:
            return
        if isinstance(profile_ids, str):
            profile_ids = (profile_ids,)
        for profile_id in profile_ids:
            normalized = self._normalize_profile_id(profile_id)
            if normalized:
                self._profiles_requiring_worker_metrics.add(normalized)

    def require_worker_metrics_for_profile(self, profile_id: str) -> None:
        self.register_profiles_requiring_worker_metrics(profile_id)

    def _profile_requires_worker_metrics(self, profile_id: str | None) -> bool:
        normalized = self._normalize_profile_id(profile_id)
        return bool(normalized and normalized in self._profiles_requiring_worker_metrics)

    def _normalize_profile_id(self, profile_id: str | None) -> str | None:
        if not profile_id:
            return None
        normalized = profile_id.strip().lower()
        return normalized or None

    def _filter_stub_metrics(self, raw_metrics: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        filtered: dict[str, Any] = {}
        for scenario_id, payload in raw_metrics.items():
            if not self._is_stub_metrics_payload(payload):
                filtered[scenario_id] = payload
        stub_only = bool(raw_metrics) and not bool(filtered)
        return filtered, stub_only

    def _is_stub_metrics_payload(self, payload: Any) -> bool:
        if not isinstance(payload, Mapping):
            return False
        warnings = payload.get("warnings") or payload.get("pob_warnings")
        if warnings is None:
            return False
        if isinstance(warnings, Sequence) and not isinstance(warnings, (str, bytes)):
            entries = warnings
        else:
            entries = (warnings,)
        for entry in entries:
            try:
                normalized = str(entry).strip().lower()
            except Exception:
                continue
            if normalized == "generation_stub_metrics":
                return True
        return False

    def _resolve_raw_metrics(
        self,
        build_id: str,
        build: dict[str, Any],
        artifacts: Any,
        templates: List[Any],
    ) -> dict[str, Any]:
        profile_id = build.get("profile_id")
        worker_required = self._profile_requires_worker_metrics(profile_id)
        fallback_metrics = self._map_raw_metrics(artifacts.raw_metrics)
        filtered_metrics, stub_only = self._filter_stub_metrics(fallback_metrics)
        if worker_required:
            worker_metrics = self._collect_worker_metrics(
                build_id,
                build,
                artifacts,
                templates,
                allow_failure=True,
            )
            if worker_metrics:
                filtered_worker_metrics, worker_stub_only = self._filter_stub_metrics(worker_metrics)
                if filtered_worker_metrics:
                    return filtered_worker_metrics
                if worker_stub_only:
                    profile_label = profile_id or "unknown profile"
                    raise APIError(
                        400,
                        "stub_metrics_disallowed",
                        f"stub metrics are disallowed for profile {profile_label}",
                        details={
                            "build_id": build_id,
                            "profile_id": profile_id,
                            "reason": "worker_stub_metrics_only",
                        },
                    )
            if stub_only and self._profile_requires_non_stub_metrics(profile_id):
                profile_label = profile_id or "unknown profile"
                raise APIError(
                    400,
                    "stub_metrics_disallowed",
                    f"stub metrics are disallowed for profile {profile_label}",
                    details={
                        "build_id": build_id,
                        "profile_id": profile_id,
                        "reason": "stub_metrics_only",
                    },
                )
            raise APIError(
                400,
                "missing_metrics",
                "raw metrics payload missing for build and worker evaluation returned no metrics",
                details={
                    "reason": "worker_metrics_missing",
                    "action": "check PoB worker availability and worker logs",
                },
            )
        if filtered_metrics:
            return filtered_metrics

        worker_metrics = self._collect_worker_metrics(
            build_id,
            build,
            artifacts,
            templates,
            allow_failure=True,
        )
        if worker_metrics:
            filtered_worker_metrics, worker_stub_only = self._filter_stub_metrics(worker_metrics)
            if filtered_worker_metrics:
                return filtered_worker_metrics
            if worker_stub_only and self._profile_requires_non_stub_metrics(profile_id):
                profile_label = profile_id or "unknown profile"
                raise APIError(
                    400,
                    "stub_metrics_disallowed",
                    f"stub metrics are disallowed for profile {profile_label}",
                    details={
                        "build_id": build_id,
                        "profile_id": profile_id,
                        "reason": "worker_stub_metrics_only",
                    },
                )
            if worker_stub_only:
                return worker_metrics

        if stub_only:
            if self._profile_requires_non_stub_metrics(profile_id):
                profile_label = profile_id or "unknown profile"
                raise APIError(
                    400,
                    "stub_metrics_disallowed",
                    f"stub metrics are disallowed for profile {profile_label}",
                    details={
                        "build_id": build_id,
                        "profile_id": profile_id,
                        "reason": "stub_metrics_only",
                    },
                )
            return fallback_metrics
        raise APIError(
            400,
            "missing_metrics",
            "raw metrics payload missing for build and worker evaluation returned no metrics",
            details={
                "reason": "worker_metrics_missing",
                "action": "check PoB worker availability and worker logs",
            },
        )

    def _collect_worker_metrics(
        self,
        build_id: str,
        build: dict[str, Any],
        artifacts: Any,
        templates: List[Any],
        *,
        allow_failure: bool,
    ) -> dict[str, Any]:
        xml_payload = self._extract_xml_payload(artifacts)
        if not xml_payload:
            return {}

        payloads: list[dict[str, Any]] = []
        for template in templates:
            payloads.append(
                {
                    "xml": xml_payload,
                    "scenario_id": template.scenario_id,
                    "profile_id": template.profile_id,
                    "ruleset_id": build.get("ruleset_id") or "",
                }
            )

        pool = WorkerPool(num_workers=1, request_timeout=25.0)
        try:
            responses = pool.evaluate_batch(payloads)
        except WorkerPoolError as exc:
            if allow_failure:
                return {}
            raise APIError(
                502,
                "worker_evaluation_failed",
                f"worker evaluation failed for build {build_id}",
                details=str(exc),
            ) from exc
        finally:
            pool.close()

        evaluated_at = datetime.now(timezone.utc).isoformat()
        mapped: dict[str, Any] = {}
        for template, response in zip(templates, responses, strict=False):
            if not isinstance(response, Mapping):
                if allow_failure:
                    return {}
                raise APIError(
                    502,
                    "worker_response_invalid",
                    f"worker response for scenario {template.scenario_id} was not an object",
                )
            error_obj = response.get("error")
            if isinstance(error_obj, Mapping):
                if allow_failure:
                    return {}
                raise APIError(
                    502,
                    "worker_response_error",
                    f"worker returned error for scenario {template.scenario_id}",
                    details=str(error_obj.get("message", "worker error")),
                )

            result_obj = response.get("result")
            if isinstance(result_obj, Mapping):
                payload = dict(result_obj)
            else:
                payload = dict(response)

            warnings = payload.get("warnings")
            if warnings is None:
                payload["warnings"] = []
            elif isinstance(warnings, list):
                payload["warnings"] = [str(item) for item in warnings]
            else:
                payload["warnings"] = [str(warnings)]
            payload["worker_metadata"] = {
                "source": "worker_pool",
                "evaluated_at": evaluated_at,
                "scenario_id": template.scenario_id,
            }
            mapped[template.scenario_id] = payload
        return mapped

    def _extract_xml_payload(self, artifacts: Any) -> str | None:
        xml_payload = artifacts.xml if isinstance(artifacts.xml, str) else None
        if xml_payload and "<" in xml_payload:
            return xml_payload

        code_payload = artifacts.code if isinstance(artifacts.code, str) else ""
        stripped = code_payload.strip()
        if stripped.startswith("<"):
            return stripped

        decoded_share_code = self._decode_share_code(stripped)
        if decoded_share_code and "<" in decoded_share_code:
            return decoded_share_code
        return None

    def _decode_share_code(self, code_payload: str) -> str | None:
        if not code_payload:
            return None
        if "{" in code_payload or "\n" in code_payload:
            return None
        try:
            padded = code_payload + ("=" * ((4 - len(code_payload) % 4) % 4))
            compressed = base64.urlsafe_b64decode(padded.encode("utf-8"))
        except Exception:
            return None

        for window_bits in (-zlib.MAX_WBITS, zlib.MAX_WBITS, zlib.MAX_WBITS | 32):
            try:
                xml_bytes = zlib.decompress(compressed, window_bits)
                xml_text = xml_bytes.decode("utf-8", errors="ignore")
                if xml_text.strip().startswith("<"):
                    return xml_text
            except Exception:
                continue
        return None

    def _ensure_transition(self, current: BuildStatus, target: BuildStatus) -> None:
        allowed = _ALLOWED_STATUS_TRANSITIONS.get(current, set())
        if target not in allowed:
            raise APIError(
                409,
                "status_conflict",
                f"cannot transition from {current.value} to {target.value}",
            )


def adjust_metrics_for_profile(
    profile_id: str | None, metrics: NormalizedMetrics
) -> NormalizedMetrics:
    normalized_profile = (profile_id or "").strip().lower()
    if normalized_profile == "delve":
        revised_score = _delve_survivability_score(metrics)
    elif normalized_profile == "support":
        revised_score = _support_utility_score(metrics)
    else:
        return metrics
    if revised_score == metrics.utility_score:
        return metrics
    return replace(metrics, utility_score=revised_score)


def _delve_survivability_score(metrics: NormalizedMetrics) -> float:
    resist_total = (
        metrics.resists.fire
        + metrics.resists.cold
        + metrics.resists.lightning
        + metrics.resists.chaos
    )
    reservation_surplus = max(
        0.0,
        metrics.reservation.available_percent - metrics.reservation.reserved_percent,
    )
    defensive_stack = (
        metrics.life / 200.0
        + metrics.armour / 1200.0
        + metrics.evasion / 900.0
        + metrics.max_hit / 6000.0
    )
    return (
        defensive_stack
        + resist_total * 0.15
        + reservation_surplus * 0.4
        + metrics.utility_score * 0.1
    )


def _support_utility_score(metrics: NormalizedMetrics) -> float:
    reserved = max(metrics.reservation.reserved_percent, 0.0)
    available = max(metrics.reservation.available_percent, 1.0)
    reservation_surplus = max(0.0, available - reserved)
    reservation_efficiency = min(reserved / available, 1.0)
    aura_bonus = (
        metrics.attributes.strength + metrics.attributes.dexterity + metrics.attributes.intelligence
    ) / 50.0
    return (
        aura_bonus
        + reservation_surplus * 1.2
        + reservation_efficiency * 5.0
        + metrics.mana / 250.0
        + metrics.utility_score * 0.2
    )
