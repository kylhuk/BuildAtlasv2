from __future__ import annotations

import base64
import logging
import xml.etree.ElementTree as ET
import zlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from backend.app.api.errors import APIError
from backend.app.api.models import BuildStatus
from backend.app.db.ch import ClickhouseRepository, ScenarioMetricRow
from backend.app.settings import settings
from backend.engine.artifacts.store import read_build_artifacts, write_build_artifacts
from backend.engine.evaluation.gates import evaluate_gates
from backend.engine.evaluation.normalized import NormalizedMetrics, map_worker_output
from backend.engine.metrics_source import (
    METRICS_SOURCE_FALLBACK,
    METRICS_SOURCE_POB,
    METRICS_SOURCE_STUB,
    normalize_metrics_source,
)
from backend.engine.scenarios.loader import (
    ScenarioGateThresholds,
    ScenarioReservationThreshold,
    ScenarioTemplate,
    list_templates,
)
from backend.engine.worker_pool import WorkerPool, WorkerPoolClosedError, WorkerPoolError

logger = logging.getLogger(__name__)

HARDCODED_WORKER_REQUEST_TIMEOUT_SECONDS = 120.0
MAX_PAYLOADS_PER_BATCH = 512
MAX_XML_BYTES_PER_BATCH = 64 * 1024 * 1024

_ALLOWED_STATUS_TRANSITIONS: Dict[BuildStatus, set[BuildStatus]] = {
    BuildStatus.imported: {BuildStatus.queued},
    BuildStatus.queued: {BuildStatus.evaluated, BuildStatus.failed},
    BuildStatus.evaluated: {BuildStatus.queued},
    BuildStatus.failed: {BuildStatus.queued},
}


@dataclass(frozen=True)
class _PreparedBuildEvaluation:
    build: dict[str, Any]
    artifacts: Any
    templates: list[ScenarioTemplate]


@dataclass
class EvaluationProvenance:
    stub_warning_count: int
    stub_warning_scenarios: list[str]
    worker_metrics_used_count: int
    worker_metadata_missing_count: int
    worker_metadata_missing_scenarios: list[str]


class BuildEvaluator:
    def __init__(
        self,
        repo: ClickhouseRepository,
        base_path: Path,
        *,
        profiles_requiring_non_stub_metrics: Sequence[str] | None = None,
        profiles_requiring_worker_metrics: Sequence[str] | None = None,
        worker_cmd: str | None = None,
        worker_args: str | None = None,
        worker_cwd: str | None = None,
        worker_pool_size: int | None = None,
    ) -> None:
        self._repo = repo
        self._base_path = base_path

        # Worker configuration defaults from application settings.
        self._worker_cmd = worker_cmd or settings.pob_worker_cmd
        raw_args = (worker_args if worker_args is not None else settings.pob_worker_args).split()
        project_root = Path(__file__).resolve().parents[3]
        if self._worker_cmd in {"luajit", "luajit.exe"} and raw_args:
            script_arg = Path(raw_args[0])
            script_path = script_arg if script_arg.is_absolute() else project_root / script_arg
            if script_path.exists() and script_path.is_file():
                raw_args[0] = str(script_path)
        self._worker_args = raw_args
        self._worker_pool_size = worker_pool_size or settings.pob_worker_pool_size
        # Make worker_cwd absolute to ensure it works regardless of parent process working directory
        worker_cwd_value = worker_cwd if worker_cwd is not None else settings.pob_worker_cwd
        if worker_cwd_value:
            cwd_path = Path(worker_cwd_value)
            resolved_cwd = cwd_path if cwd_path.is_absolute() else project_root / cwd_path
            worker_cwd_value = str(resolved_cwd)
        self._worker_cwd = worker_cwd_value.strip() if worker_cwd_value is not None else None

        self._profiles_requiring_non_stub_metrics: set[str] = set()
        if profiles_requiring_non_stub_metrics:
            self.register_profiles_requiring_non_stub_metrics(profiles_requiring_non_stub_metrics)
        self._profiles_requiring_worker_metrics: set[str] = set()
        if profiles_requiring_worker_metrics:
            self.register_profiles_requiring_worker_metrics(profiles_requiring_worker_metrics)
        self._pool: WorkerPool | None = None
        self._pool_lock = __import__("threading").Lock()
        self._last_evaluation_provenance: EvaluationProvenance | None = None
        self._templates_by_profile: dict[str, tuple[ScenarioTemplate, ...]] = {}

    def _templates_for_profile(self, profile_id: str) -> list[ScenarioTemplate]:
        cached = self._templates_by_profile.get(profile_id)
        if cached is None:
            cached = tuple(
                template for template in list_templates() if template.profile_id == profile_id
            )
            self._templates_by_profile[profile_id] = cached
        return list(cached)

    def _get_worker_pool(self) -> WorkerPool:
        with self._pool_lock:
            if self._pool is None:
                self._pool = WorkerPool(
                    num_workers=self._worker_pool_size,
                    worker_cmd=(self._worker_cmd, *self._worker_args),
                    worker_cwd=(self._worker_cwd or None),
                    request_timeout=HARDCODED_WORKER_REQUEST_TIMEOUT_SECONDS,
                )
            return self._pool

    def close(self) -> None:
        with self._pool_lock:
            if self._pool is not None:
                self._pool.close()
                self._pool = None

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def evaluate_build(
        self,
        build_id: str,
        *,
        gate_thresholds: ScenarioGateThresholds | None = None,
    ) -> tuple[BuildStatus, List[ScenarioMetricRow]]:
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
            scenario_rows = self._collect_scenario_rows(build, gate_thresholds=gate_thresholds)
            self._repo.insert_scenario_metrics(scenario_rows)
            final_status = BuildStatus.evaluated
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

    def evaluate_builds_batched(
        self,
        build_ids: Sequence[str],
        gate_thresholds: Mapping[str, Any] | None = None,
        progress_label: str | None = None,
    ) -> dict[str, Any]:
        results: dict[str, tuple[BuildStatus, List[ScenarioMetricRow]]] = {}
        errors: dict[str, APIError] = {}
        prepared_builds: dict[str, _PreparedBuildEvaluation] = {}
        grouped_build_ids: dict[tuple[str, str], list[str]] = defaultdict(list)

        for build_id in build_ids:
            try:
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
                prepared = self._prepare_build_evaluation(build, gate_thresholds=gate_thresholds)
                prepared_builds[build_id] = prepared
                profile_id = str(build.get("profile_id") or "")
                ruleset_id = str(build.get("ruleset_id") or "")
                grouped_build_ids[(profile_id, ruleset_id)].append(build_id)
            except APIError as exc:
                errors[build_id] = exc

        worker_metrics_by_build: dict[str, dict[str, Any]] = {
            build_id: {} for build_id in prepared_builds
        }
        worker_failures: dict[str, APIError] = {}
        if prepared_builds:
            try:
                worker_metrics_by_build, worker_failures = self._collect_worker_metrics_batched(
                    prepared_builds,
                    grouped_build_ids,
                    progress_label=progress_label,
                )
            except Exception:
                for build_id in prepared_builds:
                    self._repo.update_build_status(build_id, BuildStatus.failed.value)
                raise
        for build_id, error in worker_failures.items():
            self._repo.update_build_status(build_id, BuildStatus.failed.value)
            errors[build_id] = error

        for build_id in build_ids:
            if build_id in errors:
                continue
            prepared = prepared_builds.get(build_id)
            if prepared is None:
                continue
            try:
                raw_metrics_map = self._resolve_raw_metrics(
                    build_id,
                    prepared.build,
                    prepared.artifacts,
                    prepared.templates,
                    precomputed_worker_metrics=worker_metrics_by_build.get(build_id),
                )
                scenario_rows = self._build_scenario_rows_from_raw_metrics(
                    prepared.build,
                    prepared.artifacts,
                    prepared.templates,
                    raw_metrics_map,
                )
                self._repo.insert_scenario_metrics(scenario_rows)
                final_status = BuildStatus.evaluated
                self._repo.update_build_status(build_id, final_status.value)
                results[build_id] = (final_status, scenario_rows)
            except APIError as exc:
                self._repo.update_build_status(build_id, BuildStatus.failed.value)
                errors[build_id] = exc
            except WorkerPoolClosedError:
                self._repo.update_build_status(build_id, BuildStatus.failed.value)
                raise
            except Exception as exc:  # pragma: no cover - unexpected
                self._repo.update_build_status(build_id, BuildStatus.failed.value)
                errors[build_id] = APIError(
                    500,
                    "evaluation_error",
                    "failed to evaluate build",
                    details=str(exc),
                )

        return {
            "results": results,
            "errors": errors,
        }

    def _collect_scenario_rows(
        self,
        build: dict[str, Any],
        *,
        gate_thresholds: ScenarioGateThresholds | None,
    ) -> List[ScenarioMetricRow]:
        prepared = self._prepare_build_evaluation(build, gate_thresholds=gate_thresholds)
        build_id = prepared.build.get("build_id")
        if not build_id:
            raise APIError(400, "invalid_build", "build record missing build_id")
        raw_metrics_map = self._resolve_raw_metrics(
            build_id,
            prepared.build,
            prepared.artifacts,
            prepared.templates,
        )
        if not raw_metrics_map:
            raise APIError(400, "missing_metrics", "raw metrics payload missing for build")

        return self._build_scenario_rows_from_raw_metrics(
            prepared.build,
            prepared.artifacts,
            prepared.templates,
            raw_metrics_map,
        )

    def _prepare_build_evaluation(
        self,
        build: dict[str, Any],
        *,
        gate_thresholds: Mapping[str, Any] | ScenarioGateThresholds | None = None,
    ) -> _PreparedBuildEvaluation:
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
        templates = self._templates_for_profile(profile_id)
        if not templates:
            raise APIError(
                400,
                "no_scenarios",
                f"no scenario templates configured for profile {profile_id}",
            )
        templates_with_gates = self._apply_gate_threshold_overrides(templates, gate_thresholds)
        return _PreparedBuildEvaluation(
            build=build,
            artifacts=artifacts,
            templates=templates_with_gates,
        )

    def _build_scenario_rows_from_raw_metrics(
        self,
        build: dict[str, Any],
        artifacts: Any,
        templates: list[ScenarioTemplate],
        raw_metrics_map: Mapping[str, Any],
    ) -> List[ScenarioMetricRow]:
        build_id = build.get("build_id")
        if not build_id:
            raise APIError(400, "invalid_build", "build record missing build_id")
        profile_id = build.get("profile_id")
        if not profile_id:
            raise APIError(400, "missing_profile", "build profile_id missing")

        scenarios_used = []
        for template in templates:
            gate_thresholds_payload = self._serialize_gate_thresholds(template.gate_thresholds)
            scenarios_used.append(
                {
                    "scenario_id": template.scenario_id,
                    "version": template.version,
                    "profile_id": template.profile_id,
                    "pob_config": template.pob_config,
                    "gate_thresholds": gate_thresholds_payload,
                }
            )
        # FL-03: Add gate_pass to raw_metrics for ML training
        # Compute gate_pass for each scenario and add to the metrics
        for template in templates:
            payload = raw_metrics_map.get(template.scenario_id)
            if payload is not None and isinstance(payload, dict):
                normalized = map_worker_output(payload)
                normalized = adjust_metrics_for_profile(template.profile_id, normalized)
                gate_eval = evaluate_gates(normalized, template.gate_thresholds)
                payload["gate_pass"] = gate_eval.gate_pass
                payload["gate_fail_reasons"] = list(gate_eval.gate_fail_reasons)
                payload["gate_slacks"] = asdict(gate_eval.gate_slacks)

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
        worker_required = self._profile_requires_worker_metrics(profile_id)
        stub_warning_count = 0
        stub_warning_scenarios: list[str] = []
        worker_metrics_used_count = 0
        worker_metadata_missing_count = 0
        worker_metadata_missing_scenarios: list[str] = []
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
                    gate_slacks=asdict(gate_eval.gate_slacks),
                    metrics_source=self._worker_payload_metrics_source(payload),
                )
            )
            warnings_lower = {
                str(item).strip().lower() for item in normalized.warnings if str(item).strip()
            }
            if "generation_stub_metrics" in warnings_lower:
                stub_warning_count += 1
                stub_warning_scenarios.append(template.scenario_id)
            worker_source = self._worker_metadata_source(payload)
            if worker_source == "worker_pool":
                worker_metrics_used_count += 1
            elif worker_required:
                worker_metadata_missing_count += 1
                worker_metadata_missing_scenarios.append(template.scenario_id)
        if not scenario_rows:
            raise APIError(400, "no_scenarios", "no scenario rows generated for build")
        self._last_evaluation_provenance = EvaluationProvenance(
            stub_warning_count=stub_warning_count,
            stub_warning_scenarios=stub_warning_scenarios,
            worker_metrics_used_count=worker_metrics_used_count,
            worker_metadata_missing_count=worker_metadata_missing_count,
            worker_metadata_missing_scenarios=worker_metadata_missing_scenarios,
        )
        return scenario_rows

    def _serialize_gate_thresholds(self, gate_thresholds: Any) -> dict[str, Any]:
        model_dump = getattr(gate_thresholds, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump(mode="json")
            if isinstance(dumped, Mapping):
                return dict(dumped)
        as_dict = getattr(gate_thresholds, "dict", None)
        if callable(as_dict):
            dumped = as_dict()
            if isinstance(dumped, Mapping):
                return dict(dumped)
        if is_dataclass(gate_thresholds) and not isinstance(gate_thresholds, type):
            return asdict(gate_thresholds)
        if isinstance(gate_thresholds, Mapping):
            return dict(gate_thresholds)
        return {}

    def _apply_gate_threshold_overrides(
        self,
        templates: list[ScenarioTemplate],
        gate_thresholds: Mapping[str, Any] | ScenarioGateThresholds | None,
    ) -> list[ScenarioTemplate]:
        if not gate_thresholds:
            return templates
        overridden: list[ScenarioTemplate] = []
        for template in templates:
            if isinstance(gate_thresholds, Mapping):
                scenario_override = gate_thresholds.get(template.scenario_id)
                if scenario_override is None:
                    scenario_override = gate_thresholds
            else:
                scenario_override = gate_thresholds
            coerced = self._coerce_gate_thresholds(scenario_override)
            if coerced is None:
                overridden.append(template)
                continue
            overridden.append(replace(template, gate_thresholds=coerced))
        return overridden

    def _coerce_gate_thresholds(self, payload: Any) -> ScenarioGateThresholds | None:
        if payload is None:
            return None
        if isinstance(payload, ScenarioGateThresholds):
            return payload
        if is_dataclass(payload) and not isinstance(payload, type):
            payload = asdict(payload)
        model_dump = getattr(payload, "model_dump", None)
        if callable(model_dump):
            payload = model_dump(mode="json")
        else:
            as_dict = getattr(payload, "dict", None)
            if callable(as_dict):
                payload = as_dict()
        if not isinstance(payload, Mapping):
            return None

        reservation_payload = payload.get("reservation")
        if is_dataclass(reservation_payload) and not isinstance(reservation_payload, type):
            reservation_payload = asdict(reservation_payload)
        reservation_model_dump = getattr(reservation_payload, "model_dump", None)
        if callable(reservation_model_dump):
            reservation_payload = reservation_model_dump(mode="json")
        else:
            reservation_as_dict = getattr(reservation_payload, "dict", None)
            if callable(reservation_as_dict):
                reservation_payload = reservation_as_dict()
        if not isinstance(reservation_payload, Mapping):
            return None
        try:
            return ScenarioGateThresholds(
                resists={
                    str(key).lower(): float(value)
                    for key, value in dict(payload.get("resists") or {}).items()
                },
                reservation=ScenarioReservationThreshold(
                    max_percent=float(reservation_payload.get("max_percent", 0.0))
                ),
                attributes={
                    str(key).lower(): float(value)
                    for key, value in dict(payload.get("attributes") or {}).items()
                },
                min_max_hit=float(payload.get("min_max_hit", 0.0)),
                min_full_dps=float(payload.get("min_full_dps", 0.0)),
            )
        except (TypeError, ValueError):
            return None

    def _collect_worker_metrics_batched(
        self,
        prepared_builds: Mapping[str, _PreparedBuildEvaluation],
        grouped_build_ids: Mapping[tuple[str, str], list[str]],
        *,
        progress_label: str | None = None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, APIError]]:
        worker_payload_entries: list[dict[str, Any]] = []
        for (profile_id, ruleset_id), build_ids in grouped_build_ids.items():
            for build_id in build_ids:
                prepared = prepared_builds.get(build_id)
                if prepared is None:
                    continue
                if not self._build_requires_worker_batch(
                    prepared.build,
                    prepared.artifacts,
                    prepared.templates,
                ):
                    continue
                xml_payload = self._extract_xml_payload(prepared.artifacts)
                if not xml_payload:
                    continue
                xml_bytes = len(xml_payload.encode("utf-8"))
                for template in prepared.templates:
                    worker_payload_entries.append(
                        {
                            "build_id": build_id,
                            "scenario_id": template.scenario_id,
                            "xml_bytes": xml_bytes,
                            "payload": {
                                "xml": xml_payload,
                                "scenario_id": template.scenario_id,
                                "profile_id": profile_id,
                                "ruleset_id": ruleset_id,
                            },
                        }
                    )

        if not worker_payload_entries:
            return ({build_id: {} for build_id in prepared_builds}, {})

        chunked_entries = self._chunk_worker_payload_entries(worker_payload_entries)
        pool = self._get_worker_pool()
        mapped_by_build: dict[str, dict[str, Any]] = {build_id: {} for build_id in prepared_builds}
        failed_builds: dict[str, APIError] = {}
        evaluated_at = datetime.now(timezone.utc).isoformat()

        for chunk_index, chunk in enumerate(chunked_entries):
            payloads = [entry["payload"] for entry in chunk]
            label_parts = [
                progress_label.strip()
                if isinstance(progress_label, str) and progress_label.strip()
                else "",
                f"chunk={chunk_index + 1}/{len(chunked_entries)}",
                f"payloads={len(payloads)}",
            ]
            label = " ".join(part for part in label_parts if part)
            try:
                responses = pool.evaluate_batch_best_effort(
                    payloads,
                    progress_label=label or None,
                )
            except WorkerPoolClosedError:
                raise
            except WorkerPoolError as exc:
                raise APIError(
                    502,
                    "worker_evaluation_failed",
                    "worker evaluation failed for batched builds",
                    details=str(exc),
                ) from exc

            for entry, response in zip(chunk, responses, strict=False):
                build_id = str(entry["build_id"])
                if build_id in failed_builds:
                    continue
                scenario_id = str(entry["scenario_id"])
                if not isinstance(response, Mapping):
                    failed_builds[build_id] = APIError(
                        502,
                        "worker_response_invalid",
                        f"worker response for scenario {scenario_id} was not an object",
                    )
                    continue
                ok_value = response.get("ok")
                error_obj = response.get("error")
                if ok_value is False or isinstance(error_obj, Mapping):
                    details: Any
                    if isinstance(error_obj, Mapping):
                        details = str(error_obj.get("message", "worker error"))
                    else:
                        details = str(error_obj) if error_obj is not None else "worker error"
                    failed_builds[build_id] = APIError(
                        502,
                        "worker_response_error",
                        f"worker returned error for scenario {scenario_id}",
                        details=details,
                    )
                    continue

                result_obj = response.get("result")
                if isinstance(result_obj, Mapping):
                    payload = dict(result_obj)
                else:
                    payload = dict(response)
                payload = self._annotate_metrics_source(
                    payload,
                    self._worker_payload_metrics_source(payload),
                )
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
                    "scenario_id": scenario_id,
                }
                mapped_by_build[build_id][scenario_id] = payload

        return mapped_by_build, failed_builds

    def _build_requires_worker_batch(
        self,
        build: Mapping[str, Any],
        artifacts: Any,
        templates: Sequence[ScenarioTemplate],
    ) -> bool:
        profile_id = build.get("profile_id")
        worker_required = self._profile_requires_worker_metrics(profile_id)
        fallback_metrics = self._map_raw_metrics(artifacts.raw_metrics)
        filtered_metrics, stub_only = self._filter_stub_metrics(fallback_metrics)

        scenario_ids = {template.scenario_id for template in templates}
        if worker_required and filtered_metrics and not stub_only:
            if scenario_ids.issubset(filtered_metrics.keys()):
                return False

        if worker_required:
            return True
        if filtered_metrics:
            return False

        if "mapping_t16" in scenario_ids and not filtered_metrics:
            xml_metrics = self._extract_mapping_playerstats_metrics(artifacts)
            if xml_metrics is not None:
                return False
            if not stub_only and scenario_ids == {"mapping_t16"}:
                return False
        return True

    def _chunk_worker_payload_entries(
        self,
        entries: Sequence[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        chunks: list[list[dict[str, Any]]] = []
        current_chunk: list[dict[str, Any]] = []
        current_xml_bytes = 0
        for entry in entries:
            xml_bytes = int(entry.get("xml_bytes") or 0)
            over_payload_limit = len(current_chunk) >= MAX_PAYLOADS_PER_BATCH
            over_xml_limit = (
                bool(current_chunk) and current_xml_bytes + xml_bytes > MAX_XML_BYTES_PER_BATCH
            )
            if over_payload_limit or over_xml_limit:
                chunks.append(current_chunk)
                current_chunk = []
                current_xml_bytes = 0
            current_chunk.append(entry)
            current_xml_bytes += xml_bytes
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def _map_raw_metrics(self, raw_metrics: Any) -> dict[str, Any]:
        if isinstance(raw_metrics, Mapping):
            return {
                scenario_id: self._annotate_metrics_source(payload, METRICS_SOURCE_POB)
                for scenario_id, payload in raw_metrics.items()
                if isinstance(payload, Mapping)
            }
        if isinstance(raw_metrics, list):
            mapped: dict[str, Any] = {}
            for entry in raw_metrics:
                if not isinstance(entry, Mapping):
                    continue
                scenario_id = entry.get("scenario_id")
                if not isinstance(scenario_id, str):
                    continue
                payload = entry.get("payload") or entry
                mapped[scenario_id] = self._annotate_metrics_source(payload, METRICS_SOURCE_POB)
            return mapped
        return {}

    def _annotate_metrics_source(
        self, payload: Mapping[str, Any] | None, default_source: str
    ) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            return {}
        normalized_payload = dict(payload)
        normalized = normalize_metrics_source(normalized_payload.get("metrics_source"))
        normalized_payload["metrics_source"] = normalized or default_source
        return normalized_payload

    def _worker_payload_metrics_source(self, payload: Mapping[str, Any]) -> str:
        explicit_source = normalize_metrics_source(payload.get("metrics_source"))
        if explicit_source and explicit_source != METRICS_SOURCE_POB:
            return explicit_source

        worker_source = str(payload.get("source") or "").strip().lower()
        if worker_source == "pob_xml_playerstats":
            return METRICS_SOURCE_FALLBACK

        warnings = payload.get("warnings")
        warning_items: Sequence[Any]
        if isinstance(warnings, list):
            warning_items = warnings
        elif warnings is None:
            warning_items = []
        else:
            warning_items = [warnings]

        normalized_warnings = {
            str(item).strip().lower() for item in warning_items if str(item).strip()
        }
        if "generation_stub_metrics" in normalized_warnings:
            return METRICS_SOURCE_STUB

        return explicit_source or METRICS_SOURCE_POB

    def _worker_metadata_source(self, payload: Mapping[str, Any]) -> str | None:
        metadata = payload.get("worker_metadata")
        if not isinstance(metadata, Mapping):
            return None
        source = str(metadata.get("source") or "").strip().lower()
        return source or None

    def pop_last_evaluation_provenance(self) -> EvaluationProvenance | None:
        provenance = self._last_evaluation_provenance
        self._last_evaluation_provenance = None
        return provenance

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
        source_value = payload.get("metrics_source")
        normalized_source = normalize_metrics_source(source_value)
        if normalized_source and normalized_source != METRICS_SOURCE_POB:
            return True
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
        *,
        precomputed_worker_metrics: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        profile_id = build.get("profile_id")
        worker_required = self._profile_requires_worker_metrics(profile_id)
        fallback_metrics = self._map_raw_metrics(artifacts.raw_metrics)
        filtered_metrics, stub_only = self._filter_stub_metrics(fallback_metrics)
        scenario_ids = {template.scenario_id for template in templates}

        if worker_required and filtered_metrics and not stub_only:
            if scenario_ids.issubset(filtered_metrics.keys()):
                return filtered_metrics

        if not worker_required and "mapping_t16" in scenario_ids and not filtered_metrics:
            xml_metrics = self._extract_mapping_playerstats_metrics(artifacts)
            if xml_metrics is not None:
                return {"mapping_t16": xml_metrics}
            if not stub_only and scenario_ids == {"mapping_t16"}:
                raise APIError(
                    400,
                    "missing_metrics",
                    "playerstat metrics unavailable for mapping_t16 scenario",
                    details={
                        "build_id": build_id,
                        "profile_id": profile_id,
                        "scenario_id": "mapping_t16",
                        "reason": "missing_player_stats",
                    },
                )

        if worker_required:
            if precomputed_worker_metrics is not None:
                worker_metrics = dict(precomputed_worker_metrics)
            else:
                worker_metrics = self._collect_worker_metrics(
                    build_id,
                    build,
                    artifacts,
                    templates,
                    allow_failure=True,
                )
            if worker_metrics:
                filtered_worker_metrics, worker_stub_only = self._filter_stub_metrics(
                    worker_metrics
                )
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

        if precomputed_worker_metrics is not None:
            worker_metrics = dict(precomputed_worker_metrics)
        else:
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
        templates: list[ScenarioTemplate],
        allow_failure: bool = False,
    ) -> dict[str, Any]:
        """Evaluate build via worker pool and collect raw metrics."""
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

        pool = self._get_worker_pool()
        try:
            responses = pool.evaluate_batch(
                payloads,
                progress_label=f"build_id={build_id} scenarios={len(payloads)}",
            )
        except WorkerPoolError as exc:
            if allow_failure:
                logger.warning(
                    "worker evaluation failed for build %s (allow_failure=true): %s",
                    build_id,
                    exc,
                )
                return {}
            raise APIError(
                502,
                "worker_evaluation_failed",
                f"worker evaluation failed for build {build_id}",
                details=str(exc),
            ) from exc

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
            payload = self._annotate_metrics_source(
                payload,
                self._worker_payload_metrics_source(payload),
            )

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

    def _extract_mapping_playerstats_metrics(self, artifacts: Any) -> dict[str, Any] | None:
        xml_payload = self._extract_xml_payload(artifacts)
        if not xml_payload:
            return None

        stats: dict[str, float] = {}
        for node in self._iter_player_stats(xml_payload):
            stat_name = str(node.get("stat") or "").strip().lower()
            value = self._parse_player_stat_value(node.get("value"))
            if not stat_name or value is None:
                continue
            stats[stat_name] = value

        full_dps = stats.get("fulldps")
        max_hit = stats.get("maximumhittaken")
        if full_dps is None or max_hit is None:
            return None

        return {
            "source": "pob_xml_playerstats",
            "metrics": {
                "full_dps": full_dps,
                "max_hit": max_hit,
            },
            "defense": {
                "armour": stats.get("armour"),
                "evasion": stats.get("evasion"),
                "resists": {
                    "fire": stats.get("fireresist"),
                    "cold": stats.get("coldresist"),
                    "lightning": stats.get("lightningresist"),
                    "chaos": stats.get("chaosresist"),
                },
            },
            "resources": {
                "life": stats.get("life"),
                "mana": stats.get("mana"),
            },
            "reservation": {
                "reserved_percent": stats.get("reservedpercent"),
                "available_percent": stats.get("availablepercent"),
            },
            "attributes": {
                "strength": stats.get("str"),
                "dexterity": stats.get("dex"),
                "intelligence": stats.get("int"),
            },
            "warnings": [],
            "metrics_source": METRICS_SOURCE_FALLBACK,
        }

    def _iter_player_stats(self, xml_payload: str) -> Sequence[Any]:
        try:
            root = ET.fromstring(xml_payload)
        except ET.ParseError:
            return []
        return [node for node in root.findall(".//PlayerStat") if isinstance(node.tag, str)]

    @staticmethod
    def _parse_player_stat_value(raw_value: Any) -> float | None:
        if raw_value is None:
            return None
        value_str = str(raw_value).strip()
        if not value_str:
            return None
        try:
            return float(value_str)
        except ValueError:
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
