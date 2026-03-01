# ruff: noqa: B008

import json
import logging
import re
import uuid
import base64
import zlib
import subprocess
import sys
from collections import OrderedDict
from threading import Lock
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Sequence

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from backend.app.api.errors import APIError
from backend.app.api.evaluator import BuildEvaluator
from backend.app.api.models import (
    ArchiveBinDetail,
    ArchiveSummaryResponse,
    BuildCostDetail,
    BuildDetailResponse,
    BuildInventoryStatsResponse,
    BuildListResponse,
    BuildScenariosResponse,
    BuildStatus,
    BuildSummary,
    CatalogColumnsRegistryResponse,
    ErrorPayload,
    EvaluateBatchRequest,
    EvaluateBatchResponse,
    EvaluateBatchResult,
    EvaluateResponse,
    GenerationRequest,
    GenerationRunSummary,
    ImportBuildRequest,
    ImportBuildResponse,
    ModelOpsArtifactState,
    MLLoopStatusResponse,
    MLLoopStartRequest,
    MLLoopStartResponse,
    MLLoopStopRequest,
    MLLoopStopResponse,
    ModelOpsModelRecord,
    ModelOpsStatusResponse,
    PredictionDetail,
    PredictionSummary,
    ScenarioMetricSummary,
)
from backend.app.catalog_columns import CATALOG_COLUMNS_V1_VERSION, get_catalog_columns_v1
from backend.app.db.ch import BuildInsertPayload, BuildListFilters, ClickhouseRepository
from backend.app.settings import settings
from backend.engine.archive import load_archive_artifact
from backend.engine.artifacts.store import (
    artifact_paths,
    read_build_artifacts,
    write_build_artifacts,
)
from backend.engine.build_details import build_details_from_import
from backend.engine.generation.runner import run_generation

app = FastAPI(title="BuildAtlas Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
ML_LOOP_STATE_FILENAME = "state.json"
ML_LOOP_ITERATIONS_FILENAME = "iterations.jsonl"
ML_LOOP_REGISTRY: OrderedDict[str, subprocess.Popen] = OrderedDict()
ML_LOOP_REGISTRY_LOCK = Lock()


def _encode_share_code(xml_payload: str) -> str:
    xml_text = xml_payload.strip()
    if not xml_text or "<" not in xml_text:
        raise ValueError("xml payload is empty or invalid")
    compressed = zlib.compress(xml_text.encode("utf-8"), level=9)
    encoded = base64.urlsafe_b64encode(compressed).decode("ascii")
    return encoded.rstrip("=")


def _coerce_status_counts(payload: Any) -> dict[str, int]:
    if not isinstance(payload, dict):
        return {}
    status_counts: dict[str, int] = {}
    for key, value in payload.items():
        if key is None:
            continue
        status_counts[str(key)] = int(value)
    return status_counts


def _fallback_build_inventory_stats(repo: Any) -> tuple[int, int, dict[str, int]]:
    if hasattr(repo, "_builds") and isinstance(getattr(repo, "_builds"), dict):
        rows = list(getattr(repo, "_builds").values())
    else:
        try:
            rows = repo.list_builds(
                filters=BuildListFilters(include_stale=True),
                sort_by=None,
                sort_dir=None,
                limit=10000,
                offset=0,
            )
        except Exception:
            rows = []
    status_counts: dict[str, int] = {}
    stale_builds = 0
    for row in rows:
        row_dict = dict(row)
        status = str(row_dict.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        if bool(row_dict.get("is_stale")):
            stale_builds += 1
    return len(rows), stale_builds, status_counts


def _ml_loop_roots(base_path: Path) -> list[Path]:
    candidates = [base_path / "ml_loops", base_path.parent / "ml_loops", Path.cwd() / "ml_loops"]
    roots: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(candidate.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        roots.append(candidate)
    return roots


def _load_jsonl_records(path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        warnings.append(f"failed to read {path}: {exc}")
        return []
    records: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            warnings.append(f"malformed JSONL in {path}: {exc}")
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _coerce_to_float(value: Any | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compute_ml_loop_trends(
    records: Sequence[Mapping[str, Any]],
) -> tuple[
    int | None,
    float | None,
    dict[str, float],
    list[dict[str, float | int]],
    list[dict[str, float | int]],
    list[dict[str, float | int]],
]:
    best_iteration: int | None = None
    best_pass_probability_mean: float | None = None
    best_metric_mae: dict[str, float] = {}
    pass_probability_trend: list[dict[str, float | int]] = []
    gate_pass_rate_trend: list[dict[str, float | int]] = []
    diversity_trend: list[dict[str, float | int]] = []
    for record in records:
        iteration = _coerce_to_int(record.get("iteration"))
        if iteration is None:
            continue

        model_evaluation = record.get("model_evaluation")
        if not isinstance(model_evaluation, Mapping):
            model_evaluation = record.get("evaluation")
        if isinstance(model_evaluation, Mapping):
            pass_probability = model_evaluation.get("pass_probability")
            if isinstance(pass_probability, Mapping):
                pass_mean = _coerce_to_float(pass_probability.get("mean"))
                if pass_mean is not None:
                    pass_probability_trend.append({"iteration": iteration, "mean": pass_mean})
                    if best_pass_probability_mean is None or pass_mean > best_pass_probability_mean:
                        best_pass_probability_mean = pass_mean
                        best_iteration = iteration
                        mae_payload = model_evaluation.get("metric_mae")
                        if isinstance(mae_payload, Mapping):
                            best_metric_mae = {
                                str(metric): value
                                for metric, raw_value in mae_payload.items()
                                if (value := _coerce_to_float(raw_value)) is not None
                            }

        run_payload = record.get("run")
        if not isinstance(run_payload, Mapping):
            run_payload = record.get("run_summary")
        if isinstance(run_payload, Mapping):
            benchmark_payload = run_payload.get("benchmark_summary")
            if isinstance(benchmark_payload, Mapping):
                gate_values: list[float] = []
                for summary in benchmark_payload.values():
                    if not isinstance(summary, Mapping):
                        continue
                    gate_pass_rate = _coerce_to_float(summary.get("gate_pass_rate"))
                    if gate_pass_rate is not None:
                        gate_values.append(gate_pass_rate)
                if gate_values:
                    gate_pass_rate_trend.append(
                        {
                            "iteration": iteration,
                            "mean": float(sum(gate_values) / len(gate_values)),
                        }
                    )

            generation_payload = run_payload.get("generation")
            if isinstance(generation_payload, Mapping):
                attempt_records = generation_payload.get("attempt_records")
                if isinstance(attempt_records, Sequence):
                    main_skills = {
                        str(entry.get("main_skill_package")).strip().lower()
                        for entry in attempt_records
                        if isinstance(entry, Mapping)
                        and entry.get("main_skill_package") is not None
                        and str(entry.get("main_skill_package")).strip()
                    }
                    diversity_trend.append(
                        {
                            "iteration": iteration,
                            "unique_main_skills": len(main_skills),
                        }
                    )

    return (
        best_iteration,
        best_pass_probability_mean,
        best_metric_mae,
        pass_probability_trend,
        gate_pass_rate_trend,
        diversity_trend,
    )


def _discover_ml_loop_state(
    loop_id: str | None,
    base_path: Path,
    warnings: list[str],
) -> tuple[Path, dict[str, Any]] | None:
    roots = _ml_loop_roots(base_path)
    if loop_id:
        for root in roots:
            state_path = root / loop_id / ML_LOOP_STATE_FILENAME
            if not state_path.exists():
                continue
            payload = _safe_load_json(state_path, warnings)
            if isinstance(payload, dict):
                return state_path.parent, payload
        warnings.append(f"ml loop {loop_id} not found")
        return None

    latest: tuple[Path, float] | None = None
    for root in roots:
        if not root.exists():
            continue
        for state_path in root.glob(f"*/{ML_LOOP_STATE_FILENAME}"):
            if not state_path.is_file():
                continue
            try:
                mtime = state_path.stat().st_mtime
            except OSError:
                continue
            if latest is None or mtime > latest[1]:
                latest = (state_path, mtime)
    if latest is None:
        warnings.append("no ml loop state found")
        return None
    payload = _safe_load_json(latest[0], warnings)
    if not isinstance(payload, dict):
        return None
    return latest[0].parent, payload


def _prune_ml_loop_registry() -> None:
    with ML_LOOP_REGISTRY_LOCK:
        for key, process in list(ML_LOOP_REGISTRY.items()):
            if process.poll() is not None:
                ML_LOOP_REGISTRY.pop(key, None)


def _register_ml_loop_process(loop_id: str, process: subprocess.Popen) -> None:
    with ML_LOOP_REGISTRY_LOCK:
        ML_LOOP_REGISTRY.pop(loop_id, None)
        ML_LOOP_REGISTRY[loop_id] = process


def _last_active_ml_loop_id() -> str | None:
    with ML_LOOP_REGISTRY_LOCK:
        keys = list(ML_LOOP_REGISTRY.keys())
        return keys[-1] if keys else None


def _build_ml_loop_start_command(base_path: Path, payload: "MLLoopStartRequest") -> list[str]:
    iterations_value = 0 if payload.endless else payload.iterations
    if iterations_value is None:
        iterations_value = 0
    command = [
        sys.executable,
        "-m",
        "backend.tools.ml_loop",
        "start",
        "--loop-id",
        payload.loop_id,
        "--iterations",
        str(iterations_value),
        "--count",
        str(payload.count),
        "--seed-start",
        str(payload.seed_start),
        "--profile-id",
        payload.profile_id,
        "--surrogate-backend",
        payload.surrogate_backend,
        "--data-path",
        str(base_path),
    ]
    return command


def _build_ml_loop_stop_command(base_path: Path, loop_id: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "backend.tools.ml_loop",
        "stop",
        "--loop-id",
        loop_id,
        "--data-path",
        str(base_path),
    ]


def _invoke_ml_loop_stop_command(base_path: Path, loop_id: str) -> subprocess.CompletedProcess[str]:
    command = _build_ml_loop_stop_command(base_path, loop_id)
    return subprocess.run(command, capture_output=True, text=True)


@app.exception_handler(APIError)
async def handle_api_error(_: Request, exc: APIError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.to_payload()})


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    details = jsonable_encoder(exc.errors(), custom_encoder={Exception: str})
    payload = {
        "error": {
            "code": "validation_error",
            "message": "invalid request",
            "details": details,
        }
    }
    return JSONResponse(status_code=422, content=payload)


@app.exception_handler(HTTPException)
async def handle_http_exception(_: Request, exc: HTTPException) -> JSONResponse:
    payload = {"error": {"code": "http_error", "message": exc.detail}}
    return JSONResponse(status_code=exc.status_code, content=payload)


def get_repository() -> ClickhouseRepository:
    return ClickhouseRepository()


def get_artifact_base_path() -> Path:
    return settings.data_path


def get_build_evaluator(
    repo: ClickhouseRepository = Depends(get_repository),
    base_path: Path = Depends(get_artifact_base_path),
) -> BuildEvaluator:
    return BuildEvaluator(
        repo=repo,
        base_path=base_path,
        worker_cmd=settings.pob_worker_cmd,
        worker_args=settings.pob_worker_args,
        worker_cwd=settings.pob_worker_cwd,
        worker_pool_size=settings.pob_worker_pool_size,
    )


@app.get("/health")
def read_health() -> dict[str, Any]:
    return {
        "status": "ok",
        "clickhouse": f"{settings.clickhouse_host}:{settings.clickhouse_port}",
        "data_path": str(settings.data_path),
    }


@app.get("/catalog/columns/v1", response_model=CatalogColumnsRegistryResponse)
def get_catalog_columns_registry_v1() -> CatalogColumnsRegistryResponse:
    return CatalogColumnsRegistryResponse(
        registry_version=CATALOG_COLUMNS_V1_VERSION,
        columns=get_catalog_columns_v1(),
    )


@app.post("/generation", response_model=GenerationRunSummary)
def generate_builds(
    request: GenerationRequest,
    repo: ClickhouseRepository = Depends(get_repository),
    evaluator: BuildEvaluator = Depends(get_build_evaluator),
    base_path: Path = Depends(get_artifact_base_path),
) -> GenerationRunSummary:
    try:
        summary = run_generation(
            count=request.count,
            seed_start=request.seed_start,
            ruleset_id=request.ruleset_id,
            profile_id=request.profile_id,
            run_id=request.run_id,
            base_path=base_path,
            repo=repo,
            evaluator=evaluator,
            constraints=request.constraints,
            run_mode=request.run_mode,
            optimizer_iterations=request.optimizer_iterations,
            optimizer_elite_count=request.optimizer_elite_count,
        )
    except ValueError as exc:
        detail = str(exc)
        message = detail or "invalid generation parameters"
        raise APIError(
            400,
            "invalid_generation",
            message,
            details=detail,
        ) from exc
    except Exception as exc:  # pragma: no cover - best effort
        raise APIError(
            500,
            "generation_error",
            "generation run failed",
            details=str(exc),
        ) from exc
    return GenerationRunSummary.model_validate(summary)


@app.get("/runs/{run_id}", response_model=GenerationRunSummary)
def get_generation_run(
    run_id: str,
    base_path: Path = Depends(get_artifact_base_path),
) -> GenerationRunSummary:
    try:
        summary = load_run_summary(run_id, base_path=base_path)
    except ValueError as exc:
        raise APIError(400, "invalid_run_id", "invalid run identifier", details=str(exc)) from exc
    except FileNotFoundError as exc:
        raise APIError(404, "run_not_found", f"run {run_id} not found") from exc
    return GenerationRunSummary.model_validate(summary)


@app.get("/archives/{run_id}", response_model=ArchiveSummaryResponse)
def get_archive(
    run_id: str,
    base_path: Path = Depends(get_artifact_base_path),
) -> ArchiveSummaryResponse:
    try:
        artifact = load_archive_artifact(run_id, base_path=base_path)
    except ValueError as exc:
        raise APIError(
            400,
            "invalid_run_id",
            "invalid run identifier",
            details=str(exc),
        ) from exc
    except FileNotFoundError as exc:
        raise APIError(
            404,
            "archive_not_found",
            f"archive {run_id} not found",
        ) from exc
    return ArchiveSummaryResponse.model_validate(artifact)


@app.get("/archives/{run_id}/frontier", response_model=ArchiveSummaryResponse)
def get_archive_frontier(
    run_id: str,
    base_path: Path = Depends(get_artifact_base_path),
) -> ArchiveSummaryResponse:
    try:
        artifact = load_archive_artifact(run_id, base_path=base_path)
    except ValueError as exc:
        raise APIError(
            400,
            "invalid_run_id",
            "invalid run identifier",
            details=str(exc),
        ) from exc
    except FileNotFoundError as exc:
        raise APIError(
            404,
            "archive_not_found",
            f"archive {run_id} not found",
        ) from exc
    axes = artifact.get("axes") or []
    axis_names = [axis.get("name") for axis in axes if isinstance(axis, dict) and axis.get("name")]
    bins = artifact.get("bins", [])
    frontier_bins = _pareto_frontier_bins(bins, axis_names)
    decorated_bins = [_decorate_frontier_bin(entry, axis_names) for entry in frontier_bins]
    payload = dict(artifact)
    payload["bins"] = decorated_bins
    return ArchiveSummaryResponse.model_validate(payload)


@app.get("/archives/{run_id}/bins/{bin_key}", response_model=ArchiveBinDetail)
def get_archive_bin(
    run_id: str,
    bin_key: str,
    base_path: Path = Depends(get_artifact_base_path),
) -> ArchiveBinDetail:
    try:
        artifact = load_archive_artifact(run_id, base_path=base_path)
    except ValueError as exc:
        raise APIError(
            400,
            "invalid_run_id",
            "invalid run identifier",
            details=str(exc),
        ) from exc
    except FileNotFoundError as exc:
        raise APIError(
            404,
            "archive_not_found",
            f"archive {run_id} not found",
        ) from exc
    bins = artifact.get("bins", [])
    bin_entry = next((entry for entry in bins if entry.get("bin_key") == bin_key), None)
    if not bin_entry:
        raise APIError(
            404,
            "archive_bin_not_found",
            f"bin {bin_key} not found in run {run_id}",
        )
    return ArchiveBinDetail.model_validate(bin_entry)


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _descriptor_vector(entry: Dict[str, Any], axis_names: List[str]) -> tuple[float, ...]:
    if not axis_names:
        return tuple()
    descriptor = entry.get("descriptor") or {}
    values: list[float] = []
    for axis in axis_names:
        value = _safe_float(descriptor.get(axis))
        values.append(value if value is not None else float("-inf"))
    return tuple(values)


def _dominates(other: tuple[float, ...], candidate: tuple[float, ...]) -> bool:
    if len(other) != len(candidate):
        return False
    greater_or_equal = all(o >= c for o, c in zip(other, candidate, strict=False))
    strictly_greater = any(o > c for o, c in zip(other, candidate, strict=False))
    return greater_or_equal and strictly_greater


def _pareto_frontier_bins(
    bins: List[Dict[str, Any]],
    axis_names: List[str],
) -> List[Dict[str, Any]]:
    if not axis_names:
        return list(bins)
    catalog: list[tuple[tuple[float, ...], Dict[str, Any]]] = [
        (_descriptor_vector(entry, axis_names), entry) for entry in bins
    ]
    frontier: list[Dict[str, Any]] = []
    for index, (values, entry) in enumerate(catalog):
        dominated = False
        for other_index, (other_values, _) in enumerate(catalog):
            if index == other_index:
                continue
            if _dominates(other_values, values):
                dominated = True
                break
        if not dominated:
            frontier.append(entry)
    return frontier


def _tradeoff_reasons(entry: Dict[str, Any], axis_names: List[str]) -> List[str]:
    reasons: list[str] = ["pareto_frontier"]
    descriptor = entry.get("descriptor") or {}
    axis_values: list[tuple[str, float]] = []
    for axis in axis_names:
        value = _safe_float(descriptor.get(axis))
        if value is not None:
            axis_values.append((axis, value))
    if axis_values:
        axis_values.sort(key=lambda item: item[1], reverse=True)
        top_value = axis_values[0][1]
        top_axes = [axis for axis, value in axis_values if value == top_value]
        reasons.extend(f"{axis}_focus" for axis in top_axes)
    score_value = _safe_float(entry.get("score"))
    if score_value is not None:
        reasons.append("high_score" if score_value >= 0 else "low_score")
    else:
        reasons.append("score_missing")
    seen: set[str] = set()
    deduped: list[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            deduped.append(reason)
    return deduped


def _artifact_links_for_build(build_id: str) -> List[Dict[str, str]]:
    base_path = f"/builds/{build_id}"
    return [
        {"label": "build_detail", "url": base_path},
        {"label": "export_xml", "url": f"{base_path}/export/xml"},
        {"label": "export_code", "url": f"{base_path}/export/code"},
    ]


def _decorate_frontier_bin(entry: Dict[str, Any], axis_names: List[str]) -> Dict[str, Any]:
    decorated = dict(entry)
    decorated["tradeoff_reasons"] = _tradeoff_reasons(entry, axis_names)
    build_id = str(entry.get("build_id") or "")
    decorated["artifact_links"] = _artifact_links_for_build(build_id)
    return decorated


@app.post("/import", response_model=ImportBuildResponse, status_code=201)
def import_build(
    request: ImportBuildRequest,
    repo: ClickhouseRepository = Depends(get_repository),
    base_path: Path = Depends(get_artifact_base_path),
) -> ImportBuildResponse:
    xml_payload = request.xml
    code_payload = request.share_code or request.xml or ""
    metadata = request.metadata.model_copy(deep=True)
    build_details_payload = build_details_from_import(
        xml_payload=xml_payload,
        code_payload=code_payload,
        metadata={
            "class": metadata.class_,
            "ascendancy": metadata.ascendancy,
            "main_skill": metadata.main_skill,
        },
    )

    inferred_identity = (
        build_details_payload.get("identity") if isinstance(build_details_payload, dict) else None
    )
    if isinstance(inferred_identity, dict):
        inferred_class = str(inferred_identity.get("class", "unknown"))
        inferred_ascendancy = str(inferred_identity.get("ascendancy", "unknown"))
        inferred_main_skill = str(inferred_identity.get("main_skill", "unknown"))
        if metadata.class_ == "unknown" and inferred_class != "unknown":
            metadata.class_ = inferred_class
        if metadata.ascendancy == "unknown" and inferred_ascendancy != "unknown":
            metadata.ascendancy = inferred_ascendancy
        if metadata.main_skill == "unknown" and inferred_main_skill != "unknown":
            metadata.main_skill = inferred_main_skill

    build_id = uuid.uuid4().hex
    provenance = write_build_artifacts(
        build_id,
        xml=xml_payload,
        code=code_payload,
        build_details=build_details_payload,
        base_path=base_path,
    )
    payload = BuildInsertPayload(
        build_id=build_id,
        created_at=datetime.now(timezone.utc),
        ruleset_id=metadata.ruleset_id,
        profile_id=metadata.profile_id,
        class_=metadata.class_,
        ascendancy=metadata.ascendancy,
        main_skill=metadata.main_skill,
        damage_type=metadata.damage_type,
        defence_type=metadata.defence_type,
        complexity_bucket=metadata.complexity_bucket,
        pob_xml_path=str(provenance.paths.build_xml),
        pob_code_path=str(provenance.paths.code),
        genome_path=str(provenance.paths.genome),
        tags=metadata.tags,
        status=BuildStatus.imported.value,
    )
    repo.insert_build(payload)
    return ImportBuildResponse(
        build_id=build_id,
        status=BuildStatus.imported,
        pob_xml_path=str(provenance.paths.build_xml),
        pob_code_path=str(provenance.paths.code),
        genome_path=str(provenance.paths.genome),
    )


@app.post("/evaluate/{build_id}", response_model=EvaluateResponse)
def evaluate_build_endpoint(
    build_id: str, evaluator: BuildEvaluator = Depends(get_build_evaluator)
) -> EvaluateResponse:
    status, rows = evaluator.evaluate_build(build_id)
    scenario_results = [_scenario_metric_summary_from_row(row.model_dump()) for row in rows]
    return EvaluateResponse(build_id=build_id, status=status, scenario_results=scenario_results)


@app.post("/evaluate-batch", response_model=EvaluateBatchResponse)
def evaluate_batch(
    request: EvaluateBatchRequest, evaluator: BuildEvaluator = Depends(get_build_evaluator)
) -> EvaluateBatchResponse:
    results: List[EvaluateBatchResult] = []
    for build_id in request.build_ids:
        try:
            status, rows = evaluator.evaluate_build(build_id)
            scenario_results = [_scenario_metric_summary_from_row(row.model_dump()) for row in rows]
            results.append(
                EvaluateBatchResult(
                    build_id=build_id,
                    status=status,
                    scenario_results=scenario_results,
                )
            )
        except APIError as exc:
            results.append(
                EvaluateBatchResult(
                    build_id=build_id,
                    error=ErrorPayload(
                        code=exc.code,
                        message=exc.message,
                        details=exc.details,
                    ),
                )
            )
        except Exception as exc:  # pragma: no cover - best effort
            results.append(
                EvaluateBatchResult(
                    build_id=build_id,
                    error=ErrorPayload(
                        code="evaluation_error",
                        message="failed to evaluate build",
                        details=str(exc),
                    ),
                )
            )
    return EvaluateBatchResponse(results=results)


@app.get("/builds", response_model=BuildListResponse)
def list_builds(
    ruleset_id: str | None = Query(None),
    profile_id: str | None = Query(None),
    status: str | None = Query(None),
    class_: str | None = Query(None, alias="class"),
    ascendancy: str | None = Query(None),
    created_after: datetime | None = Query(None),
    created_before: datetime | None = Query(None),
    scenario_id: str | None = Query(None),
    gate_pass: bool | None = Query(None),
    max_cost_chaos: float | None = Query(None, ge=0),
    exclude_unknown_cost: bool | None = Query(None),
    price_snapshot_id: str | None = Query(None),
    cost_calculated_after: datetime | None = Query(None),
    cost_calculated_before: datetime | None = Query(None),
    constraint_status: str | None = Query(None),
    constraint_reason_code: str | None = Query(None),
    violated_constraint: str | None = Query(None),
    constraint_checked_after: datetime | None = Query(None),
    constraint_checked_before: datetime | None = Query(None),
    prediction_mode: Literal["include_predicted", "verified_only"] = Query("include_predicted"),
    include_stale: bool = Query(False),
    sort_by: str | None = Query(None),
    sort_dir: str | None = Query(None),
    limit: int = Query(100, ge=1),
    offset: int = Query(0, ge=0),
    repo: ClickhouseRepository = Depends(get_repository),
    base_path: Path = Depends(get_artifact_base_path),
) -> BuildListResponse:
    filters = BuildListFilters(
        ruleset_id=ruleset_id,
        profile_id=profile_id,
        status=status,
        class_=class_,
        ascendancy=ascendancy,
        created_after=created_after,
        created_before=created_before,
        scenario_id=scenario_id,
        gate_pass=gate_pass,
        max_cost_chaos=max_cost_chaos,
        exclude_unknown_cost=exclude_unknown_cost,
        price_snapshot_id=price_snapshot_id,
        cost_calculated_after=cost_calculated_after,
        cost_calculated_before=cost_calculated_before,
        constraint_status=constraint_status,
        constraint_reason_code=constraint_reason_code,
        violated_constraint=violated_constraint,
        constraint_checked_after=constraint_checked_after,
        constraint_checked_before=constraint_checked_before,
        include_stale=include_stale,
        verified_only=prediction_mode == "verified_only",
    )
    try:
        rows = repo.list_builds(
            filters=filters, sort_by=sort_by, sort_dir=sort_dir, limit=limit, offset=offset
        )
    except ValueError as exc:
        raise APIError(
            400,
            "invalid_query",
            "invalid list parameters",
            details=str(exc),
        ) from exc
    builds: list[BuildSummary] = []
    for row in rows:
        row_payload = dict(row)
        if not _artifact_path_exists(row_payload.get("pob_code_path"), base_path):
            continue
        build_summary = _build_summary_from_row(row_payload)
        build_id = row.get("build_id")
        if build_id:
            prediction_payload = _read_surrogate_prediction(build_id, base_path)
            if prediction_payload:
                build_summary.prediction = _prediction_summary_from_payload(prediction_payload)
        builds.append(build_summary)
    return BuildListResponse(builds=builds)


@app.get("/builds/stats", response_model=BuildInventoryStatsResponse)
def build_inventory_stats(
    repo: ClickhouseRepository = Depends(get_repository),
) -> BuildInventoryStatsResponse:
    generated_at = _format_iso_datetime(datetime.now(timezone.utc))
    status_counts: dict[str, int]
    total_builds: int
    stale_builds: int
    stats_method = getattr(repo, "build_inventory_stats", None)
    if callable(stats_method):
        payload = stats_method()
        if not isinstance(payload, dict):
            payload = {}
        total_builds = int(payload.get("total_builds") or 0)
        stale_builds = int(payload.get("stale_builds") or 0)
        status_counts = _coerce_status_counts(payload.get("status_counts"))
    else:
        total_builds, stale_builds, status_counts = _fallback_build_inventory_stats(repo)
    return BuildInventoryStatsResponse(
        generated_at=generated_at,
        total_builds=total_builds,
        stale_builds=stale_builds,
        status_counts=status_counts,
    )


@app.get("/builds/{build_id}", response_model=BuildDetailResponse)
def build_detail(
    build_id: str,
    repo: ClickhouseRepository = Depends(get_repository),
    base_path: Path = Depends(get_artifact_base_path),
) -> BuildDetailResponse:
    build_row = repo.get_build(build_id)
    if not build_row:
        raise APIError(404, "build_not_found", f"build {build_id} not found")
    cost_row = repo.get_latest_build_cost(build_id)
    summary_payload = dict(build_row)
    cost_detail: BuildCostDetail | None = None
    if cost_row:
        summary_payload.update(cost_row)
        cost_detail = _build_cost_detail_from_row(cost_row, base_path)
    build_summary = _build_summary_from_row(summary_payload)
    scenario_rows = repo.list_scenario_metrics(build_id)
    scenario_metrics = [_scenario_metric_summary_from_row(row) for row in scenario_rows]
    scenarios_used: list[dict[str, Any]] = []
    build_details_payload: dict[str, Any] | None = None
    constraints_payload: dict[str, Any] | None = None
    try:
        artifacts = read_build_artifacts(build_id, base_path=base_path)
        if isinstance(artifacts.scenarios_used, list):
            scenarios_used = [
                dict(item) for item in artifacts.scenarios_used if isinstance(item, dict)
            ]
        if isinstance(artifacts.build_details, dict):
            build_details_payload = dict(artifacts.build_details)
        if isinstance(artifacts.constraints, dict):
            constraints_payload = dict(artifacts.constraints)
    except FileNotFoundError:
        scenarios_used = []
    prediction_payload = _read_surrogate_prediction(build_id, base_path)
    prediction_detail: PredictionDetail | None = None
    if prediction_payload:
        prediction_summary = _prediction_summary_from_payload(prediction_payload)
        build_summary.prediction = prediction_summary
        # Use the first scenario row for canonical verification insights.
        verified_full_dps = scenario_metrics[0].full_dps if scenario_metrics else None
        verified_max_hit = scenario_metrics[0].max_hit if scenario_metrics else None
        detail_payload = prediction_summary.model_dump()
        detail_payload.update(
            {
                "verified_full_dps": verified_full_dps,
                "verified_max_hit": verified_max_hit,
                "error_full_dps": _absolute_error(
                    prediction_summary.predicted_full_dps, verified_full_dps
                ),
                "error_max_hit": _absolute_error(
                    prediction_summary.predicted_max_hit, verified_max_hit
                ),
            }
        )
        prediction_detail = PredictionDetail.model_validate(detail_payload)
    return BuildDetailResponse(
        build=build_summary,
        prediction=prediction_detail,
        scenario_metrics=scenario_metrics,
        scenarios_used=scenarios_used,
        costs=cost_detail,
        build_details=build_details_payload,
        constraints=constraints_payload,
    )


@app.get("/builds/{build_id}/scenarios", response_model=BuildScenariosResponse)
def build_scenarios(
    build_id: str,
    repo: ClickhouseRepository = Depends(get_repository),
    base_path: Path = Depends(get_artifact_base_path),
) -> BuildScenariosResponse:
    build_row = repo.get_build(build_id)
    if not build_row:
        raise APIError(404, "build_not_found", f"build {build_id} not found")
    try:
        artifacts = read_build_artifacts(build_id, base_path=base_path)
    except FileNotFoundError as exc:
        raise APIError(404, "artifacts_missing", f"artifacts missing for {build_id}") from exc
    scenarios_used = artifacts.scenarios_used
    if not isinstance(scenarios_used, list):
        scenarios_used = []
    return BuildScenariosResponse(
        build_id=build_id,
        scenarios_used=[dict(item) for item in scenarios_used if isinstance(item, dict)],
    )


@app.get("/builds/{build_id}/export/xml")
def export_xml(
    build_id: str, base_path: Path = Depends(get_artifact_base_path)
) -> PlainTextResponse:
    try:
        artifacts = read_build_artifacts(build_id, base_path=base_path)
    except FileNotFoundError as exc:
        raise APIError(404, "artifacts_missing", f"artifacts missing for {build_id}") from exc
    if not artifacts.xml:
        raise APIError(404, "artifact_missing", "xml artifact is not available")
    return PlainTextResponse(content=artifacts.xml, media_type="application/xml")


@app.get("/builds/{build_id}/export/code")
def export_code(
    build_id: str, base_path: Path = Depends(get_artifact_base_path)
) -> PlainTextResponse:
    try:
        artifacts = read_build_artifacts(build_id, base_path=base_path)
    except FileNotFoundError as exc:
        raise APIError(404, "artifacts_missing", f"artifacts missing for {build_id}") from exc
    if artifacts.xml and artifacts.xml.strip():
        share_code = _encode_share_code(artifacts.xml)
        return PlainTextResponse(content=share_code, media_type="text/plain")

    if not artifacts.code:
        raise APIError(404, "artifact_missing", "code artifact is not available")
    code_payload = artifacts.code.strip()
    if code_payload.startswith("<"):
        share_code = _encode_share_code(code_payload)
        return PlainTextResponse(content=share_code, media_type="text/plain")
    return PlainTextResponse(content=artifacts.code, media_type="text/plain")


def _build_summary_from_row(row: Dict[str, Any]) -> BuildSummary:
    return BuildSummary.model_validate(row)


def _artifact_path_exists(path_value: Any, base_path: Path) -> bool:
    if not isinstance(path_value, str) or not path_value:
        return False
    path = Path(path_value)
    if not path.is_absolute():
        path = base_path / path
    return path.exists()


def _scenario_metric_summary_from_row(row: Dict[str, Any]) -> ScenarioMetricSummary:
    sanitized = dict(row)
    sanitized["gate_pass"] = bool(sanitized.get("gate_pass"))
    return ScenarioMetricSummary.model_validate(sanitized)


def _load_cost_json(base_path: Path, relative_path: str | None) -> Dict[str, Any] | None:
    if not relative_path:
        return None
    path = base_path / relative_path
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _extract_cost_entries(
    base_path: Path, relative_path: str | None, key: str
) -> List[Dict[str, Any]]:
    payload = _load_cost_json(base_path, relative_path)
    if not payload:
        return []
    entries = payload.get(key)
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def _build_cost_detail_from_row(row: Dict[str, Any], base_path: Path) -> BuildCostDetail:
    return BuildCostDetail(
        price_snapshot_id=row.get("price_snapshot_id"),
        total_cost_chaos=row.get("total_cost_chaos"),
        unknown_cost_count=row.get("unknown_cost_count"),
        slot_costs=_extract_cost_entries(base_path, row.get("slot_costs_json_path"), "slots"),
        gem_costs=_extract_cost_entries(base_path, row.get("gem_costs_json_path"), "gems"),
    )


def _coerce_prediction_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_surrogate_prediction(build_id: str, base_path: Path) -> dict[str, Any] | None:
    try:
        prediction_path = artifact_paths(build_id, base_path).surrogate_prediction
    except ValueError:
        return None
    if not prediction_path.exists():
        return None
    try:
        return json.loads(prediction_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _prediction_summary_from_payload(payload: dict[str, Any]) -> PredictionSummary:
    metrics = payload.get("predicted_metrics") or {}
    return PredictionSummary(
        model_id=payload.get("model_id"),
        model_path=payload.get("model_path"),
        predicted_full_dps=_coerce_prediction_float(metrics.get("full_dps")),
        predicted_max_hit=_coerce_prediction_float(metrics.get("max_hit")),
        pass_probability=_coerce_prediction_float(payload.get("pass_probability")),
        selection_reason=payload.get("selection_reason"),
        timestamp=payload.get("timestamp"),
    )


def _absolute_error(predicted: float | None, verified: float | None) -> float | None:
    if predicted is None or verified is None:
        return None
    return abs(predicted - verified)


def _run_summary_path(run_id: str, base_path: Path | None = None) -> Path:
    safe_run_id = _validate_run_id(run_id)
    runs_root = (Path(base_path or settings.data_path) / "runs").resolve()
    summary_path = (runs_root / safe_run_id / "summary.json").resolve()
    if runs_root not in summary_path.parents:
        raise ValueError("run_id resolved outside run directory")
    return summary_path


def _validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_' or '-'"
        )
    return run_id


def load_run_summary(run_id: str, base_path: Path | None = None) -> dict[str, Any]:
    path = _run_summary_path(run_id, base_path)
    if not path.exists():
        raise FileNotFoundError(f"run summary {run_id} not found")
    return json.loads(path.read_text(encoding="utf-8"))


_CHECKPOINT_PATTERNS = (
    "checkpoints/**/*.json",
    "models/**/checkpoint*.json",
)
_ROLLBACK_PATTERNS = ("**/*rollback*.json",)
_TIMESTAMP_KEYS = (
    "trained_at_utc",
    "trained_at",
    "timestamp",
    "created_at",
    "completed_at",
    "started_at",
    "updated_at",
)


_REDACTED_METADATA_VALUE = "[REDACTED]"
_MODEL_META_METADATA_ALLOWLIST = (
    "model_id",
    "model_path",
    "trained_at",
    "trained_at_utc",
    "record",
    "selection_reason",
)
_CHECKPOINT_METADATA_ALLOWLIST = (
    "checkpoint_id",
    "timestamp",
)
_ROLLBACK_METADATA_ALLOWLIST = (
    "rollback_id",
    "created_at",
)


def _sanitize_metadata(
    metadata: dict[str, Any] | None,
    allowlist: tuple[str, ...],
) -> dict[str, Any] | None:
    if not metadata:
        return None
    sanitized: dict[str, Any] = {}
    allowed_keys = set(allowlist)
    for key, value in metadata.items():
        if key in allowed_keys:
            sanitized[key] = value
        else:
            sanitized[key] = _REDACTED_METADATA_VALUE
    return sanitized or None


@app.get("/ops/model-status", response_model=ModelOpsStatusResponse)
def get_model_ops_status(
    base_path: Path = Depends(get_artifact_base_path),
) -> ModelOpsStatusResponse:
    return _model_ops_status_from_artifacts(base_path)


@app.post("/ops/ml-loop-start", response_model=MLLoopStartResponse)
def start_ml_loop(
    request: MLLoopStartRequest, base_path: Path = Depends(get_artifact_base_path)
) -> MLLoopStartResponse:
    _prune_ml_loop_registry()
    loop_id = (request.loop_id or "").strip()
    if not loop_id:
        raise APIError(400, "loop_id_invalid", "loop_id is required to start an ML loop")
    with ML_LOOP_REGISTRY_LOCK:
        running = ML_LOOP_REGISTRY.get(loop_id)
        if running and running.poll() is None:
            raise APIError(
                409,
                "loop_already_running",
                f"ML loop {loop_id} is already running",
            )
    command = _build_ml_loop_start_command(base_path, request)
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        raise APIError(
            500,
            "ml_loop_start_failed",
            "unable to start ML loop",
            details=str(exc),
        ) from exc
    _register_ml_loop_process(loop_id, process)
    return MLLoopStartResponse(
        loop_id=loop_id,
        status="started",
        endless=request.endless,
        iterations=None if request.endless else request.iterations,
        pid=process.pid,
        message="ml loop scheduled",
    )


@app.post("/ops/ml-loop-stop", response_model=MLLoopStopResponse)
def stop_ml_loop(
    request: MLLoopStopRequest, base_path: Path = Depends(get_artifact_base_path)
) -> MLLoopStopResponse:
    _prune_ml_loop_registry()
    loop_id = (request.loop_id or "").strip() or _last_active_ml_loop_id()
    if not loop_id:
        raise APIError(400, "loop_id_missing", "provide loop_id to stop the ML loop")
    result = _invoke_ml_loop_stop_command(base_path, loop_id)
    if result.returncode != 0:
        raise APIError(
            500,
            "ml_loop_stop_failed",
            "failed to request ML loop stop",
            details=result.stderr or result.stdout or None,
        )
    return MLLoopStopResponse(
        loop_id=loop_id,
        stop_requested=True,
        message="stop requested",
    )


@app.get("/ops/ml-loop-status", response_model=MLLoopStatusResponse)
def get_ml_loop_status(
    loop_id: str | None = Query(None),
    base_path: Path = Depends(get_artifact_base_path),
) -> MLLoopStatusResponse:
    warnings: list[str] = []
    generated_at = _format_iso_datetime(datetime.now(timezone.utc))
    discovered = _discover_ml_loop_state(loop_id, base_path, warnings)
    if discovered is None:
        return MLLoopStatusResponse(generated_at=generated_at, loop_id=loop_id, warnings=warnings)

    loop_root, state = discovered
    records = _load_jsonl_records(loop_root / ML_LOOP_ITERATIONS_FILENAME, warnings)
    latest_iteration = records[-1] if records else None
    previous_iteration = records[-2] if len(records) > 1 else None
    (
        best_iteration,
        best_pass_probability_mean,
        best_metric_mae,
        pass_probability_trend,
        gate_pass_rate_trend,
        diversity_trend,
    ) = _compute_ml_loop_trends(records)
    return MLLoopStatusResponse(
        generated_at=generated_at,
        loop_id=_coerce_to_str(state.get("loop_id")) or loop_root.name,
        status=_coerce_to_str(state.get("status")),
        phase=_coerce_to_str(state.get("phase")),
        iteration=_coerce_to_int(state.get("iteration")),
        total_iterations=_coerce_to_int(state.get("total_iterations")),
        stop_requested=bool(state.get("stop_requested")),
        last_run_id=_coerce_to_str(state.get("last_run_id")),
        last_snapshot_id=_coerce_to_str(state.get("last_snapshot_id")),
        last_error=_coerce_to_str(state.get("last_error")),
        last_improvement=state.get("last_improvement")
        if isinstance(state.get("last_improvement"), dict)
        else None,
        latest_iteration=latest_iteration,
        previous_iteration=previous_iteration,
        best_iteration=best_iteration,
        best_pass_probability_mean=best_pass_probability_mean,
        best_metric_mae=best_metric_mae,
        pass_probability_trend=pass_probability_trend,
        gate_pass_rate_trend=gate_pass_rate_trend,
        diversity_trend=diversity_trend,
        warnings=warnings,
    )


def _model_ops_status_from_artifacts(base_path: Path) -> ModelOpsStatusResponse:
    warnings: list[str] = []
    candidate = _discover_latest_model_meta(base_path, warnings)
    active_model: ModelOpsModelRecord | None = None
    if candidate:
        meta_path, metadata, timestamp, trained_at = candidate
        metadata_dict = metadata if isinstance(metadata, dict) else None
        sanitized_metadata = _sanitize_metadata(metadata_dict, _MODEL_META_METADATA_ALLOWLIST)
        active_model = ModelOpsModelRecord(
            path=str(meta_path),
            model_id=_coerce_to_str(
                sanitized_metadata.get("model_id")
                if sanitized_metadata
                else metadata_dict.get("model_id"),
            )
            if metadata_dict
            else None,
            trained_at_utc=trained_at or _format_iso_datetime(timestamp),
            metadata=sanitized_metadata,
        )
    return ModelOpsStatusResponse(
        generated_at=_format_iso_datetime(datetime.now(timezone.utc)),
        active_model=active_model,
        last_training=active_model,
        checkpoint=_discover_artifact_state(
            base_path,
            _CHECKPOINT_PATTERNS,
            warnings,
            _CHECKPOINT_METADATA_ALLOWLIST,
        ),
        rollback=_discover_artifact_state(
            base_path,
            _ROLLBACK_PATTERNS,
            warnings,
            _ROLLBACK_METADATA_ALLOWLIST,
        ),
        warnings=warnings,
    )


def _discover_latest_model_meta(
    base_path: Path, warnings: list[str]
) -> tuple[Path, dict[str, Any] | None, datetime, str | None] | None:
    search_roots: list[Path] = []
    models_root = base_path / "models"
    if models_root.exists():
        search_roots.append(models_root)
    ml_loops_root = base_path / "ml_loops"
    if ml_loops_root.exists():
        search_roots.append(ml_loops_root)
    if not search_roots:
        return None

    latest: tuple[Path, dict[str, Any] | None, datetime, str | None] | None = None
    seen: set[str] = set()
    for root in search_roots:
        for meta_path in root.rglob("model_meta.json"):
            if not meta_path.is_file():
                continue
            resolved = str(meta_path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            metadata = _safe_load_json(meta_path, warnings)
            metadata_dict = metadata if isinstance(metadata, dict) else None
            timestamp = _timestamp_from_metadata(metadata_dict) or _file_mtime(meta_path, warnings)
            trained_at = _extract_trained_at(metadata)
            entry = (meta_path, metadata_dict, timestamp, trained_at)
            if latest is None or timestamp > latest[2]:
                latest = entry
    return latest


def _discover_artifact_state(
    base_path: Path,
    patterns: tuple[str, ...],
    warnings: list[str],
    metadata_allowlist: tuple[str, ...],
) -> ModelOpsArtifactState:
    candidate: tuple[Path, dict[str, Any] | None, datetime] | None = None
    seen: set[str] = set()
    for pattern in patterns:
        for path in base_path.glob(pattern):
            if not path.is_file():
                continue
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            metadata = _safe_load_json(path, warnings)
            metadata_dict = metadata if isinstance(metadata, dict) else None
            timestamp = _timestamp_from_metadata(metadata_dict) or _file_mtime(path, warnings)
            entry = (path, metadata_dict, timestamp)
            if candidate is None or timestamp > candidate[2]:
                candidate = entry
    if candidate is None:
        return ModelOpsArtifactState(state="missing")
    artifact_path, metadata_dict, timestamp = candidate
    sanitized_metadata = _sanitize_metadata(metadata_dict, metadata_allowlist)
    return ModelOpsArtifactState(
        state="available",
        path=str(artifact_path),
        timestamp=_format_iso_datetime(timestamp),
        metadata=sanitized_metadata,
    )


def _safe_load_json(path: Path, warnings: list[str]) -> Any | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        warnings.append(f"failed to read {path}: {exc}")
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        warnings.append(f"malformed JSON in {path}: {exc}")
        return None


def _timestamp_from_metadata(metadata: dict[str, Any] | None) -> datetime | None:
    if not metadata:
        return None
    for key in _TIMESTAMP_KEYS:
        parsed = _parse_iso_timestamp(metadata.get(key))
        if parsed:
            return parsed
    return None


def _parse_iso_timestamp(value: Any | None) -> datetime | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if candidate.endswith("Z") and not candidate.endswith("+00:00"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _extract_trained_at(metadata: Any | None) -> str | None:
    if not isinstance(metadata, dict):
        return None
    for key in ("trained_at_utc", "trained_at"):
        value = metadata.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)):
            return str(value)
    return None


def _format_iso_datetime(value: datetime) -> str:
    normalized = value.astimezone(timezone.utc).replace(microsecond=0)
    return normalized.isoformat().replace("+00:00", "Z")


def _file_mtime(path: Path, warnings: list[str]) -> datetime:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    except OSError as exc:
        warnings.append(f"unable to stat {path}: {exc}")
        return datetime.now(timezone.utc)


def _coerce_to_str(value: Any | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except (TypeError, ValueError):
        return None


def _coerce_to_int(value: Any | None) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "list_builds",
    "build_detail",
    "build_scenarios",
    "export_xml",
    "export_code",
    "run_generation",
    "load_run_summary",
    "get_archive",
    "get_archive_frontier",
    "get_archive_bin",
    "get_model_ops_status",
    "get_ml_loop_status",
    "_run_summary_path",
]
