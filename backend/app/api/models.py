from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class BuildStatus(str, Enum):
    imported = "imported"
    queued = "queued"
    evaluated = "evaluated"
    failed = "failed"


class ErrorPayload(BaseModel):
    code: str
    message: str
    details: Any | None = None

    model_config = ConfigDict(extra="ignore")


class ErrorEnvelope(BaseModel):
    error: ErrorPayload

    model_config = ConfigDict(extra="ignore")


class ImportBuildMetadata(BaseModel):
    ruleset_id: str
    profile_id: str
    class_: str = Field("unknown", alias="class")
    ascendancy: str = "unknown"
    main_skill: str = "unknown"
    damage_type: str = "unknown"
    defence_type: str = "unknown"
    complexity_bucket: str = "unknown"
    tags: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class ImportBuildRequest(BaseModel):
    share_code: str | None = None
    xml: str | None = None
    metadata: ImportBuildMetadata

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_payload(self) -> "ImportBuildRequest":
        has_share = bool(self.share_code)
        has_xml = bool(self.xml)
        if has_share == has_xml:
            raise ValueError("exactly one of share_code or xml must be provided")
        return self


class ImportBuildResponse(BaseModel):
    build_id: str
    status: BuildStatus
    pob_xml_path: str
    pob_code_path: str
    genome_path: str

    model_config = ConfigDict(extra="ignore")


class ScenarioMetricSummary(BaseModel):
    ruleset_id: str
    scenario_id: str
    gate_pass: bool
    gate_fail_reasons: List[str]
    pob_warnings: List[str]
    evaluated_at: Any
    full_dps: float
    max_hit: float
    armour: float
    evasion: float
    life: float
    mana: float
    utility_score: float

    model_config = ConfigDict(extra="ignore")


class SlotCostDetail(BaseModel):
    slot: str
    name: str
    cost_chaos: float | None = None
    matched: bool

    model_config = ConfigDict(extra="ignore")


class GemCostDetail(BaseModel):
    name: str
    level: int | None = None
    quality: int | None = None
    price: float | None = None
    matched: bool

    model_config = ConfigDict(extra="ignore")


class BuildCostDetail(BaseModel):
    price_snapshot_id: str | None = None
    total_cost_chaos: float | None = None
    unknown_cost_count: int | None = None
    slot_costs: List[SlotCostDetail] = Field(default_factory=list)
    gem_costs: List[GemCostDetail] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class PredictionSummary(BaseModel):
    model_id: str | None = None
    model_path: str | None = None
    predicted_full_dps: float | None = None
    predicted_max_hit: float | None = None
    pass_probability: float | None = None
    selection_reason: str | None = None
    timestamp: Any | None = None

    model_config = ConfigDict(extra="ignore")


class PredictionDetail(PredictionSummary):
    verified_full_dps: float | None = None
    verified_max_hit: float | None = None
    error_full_dps: float | None = None
    error_max_hit: float | None = None

    model_config = ConfigDict(extra="ignore")


class BuildSummary(BaseModel):
    build_id: str
    created_at: Any
    ruleset_id: str
    profile_id: str
    status: BuildStatus
    is_stale: bool = False
    class_: str = Field(alias="class")
    ascendancy: str
    main_skill: str
    damage_type: str
    defence_type: str
    complexity_bucket: str
    pob_xml_path: str
    pob_code_path: str
    genome_path: str
    tags: List[str]
    price_snapshot_id: str | None = None
    total_cost_chaos: float | None = None
    unknown_cost_count: int | None = None
    dps_per_chaos: float | None = None
    max_hit_per_chaos: float | None = None
    constraint_status: str | None = None
    constraint_reason_code: str | None = None
    violated_constraints: List[str] = Field(default_factory=list)
    constraint_checked_at: Any | None = None
    prediction: PredictionSummary | None = None

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class BuildListResponse(BaseModel):
    builds: List[BuildSummary]

    model_config = ConfigDict(extra="ignore")


class BuildInventoryStatsResponse(BaseModel):
    generated_at: str
    total_builds: int
    stale_builds: int = 0
    status_counts: Dict[str, int] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore")


class BuildDetailResponse(BaseModel):
    build: BuildSummary
    prediction: PredictionDetail | None = None
    scenario_metrics: List[ScenarioMetricSummary]
    scenarios_used: List[dict[str, Any]] = Field(default_factory=list)
    costs: BuildCostDetail | None = None
    build_details: dict[str, Any] | None = None
    constraints: dict[str, Any] | None = None

    model_config = ConfigDict(extra="ignore")


class EvaluateResponse(BaseModel):
    build_id: str
    status: BuildStatus
    scenario_results: List[ScenarioMetricSummary]

    model_config = ConfigDict(extra="ignore")


class EvaluateBatchRequest(BaseModel):
    build_ids: List[str]

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_list(self) -> "EvaluateBatchRequest":
        if not self.build_ids:
            raise ValueError("build_ids must include at least one entry")
        return self


class EvaluateBatchResult(BaseModel):
    build_id: str
    status: BuildStatus | None = None
    scenario_results: List[ScenarioMetricSummary] | None = None
    error: ErrorPayload | None = None

    model_config = ConfigDict(extra="ignore")


class EvaluateBatchResponse(BaseModel):
    results: List[EvaluateBatchResult]

    model_config = ConfigDict(extra="ignore")


class BuildScenariosResponse(BaseModel):
    build_id: str
    scenarios_used: List[dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class CatalogColumn(BaseModel):
    id: str
    label: str
    data_type: str
    sortable: bool = False
    filterable: bool = False
    source: str | None = None
    index_priority: Literal["high", "medium", "low"] | None = None
    unit: str | None = None

    model_config = ConfigDict(extra="ignore")


class CatalogColumnsRegistryResponse(BaseModel):
    registry_version: str
    columns: List[CatalogColumn]

    model_config = ConfigDict(extra="ignore")


class GenerationRequest(BaseModel):
    count: int
    seed_start: int = 0
    ruleset_id: str
    profile_id: str
    run_id: str | None = None
    run_mode: Literal["standard", "optimizer"] = "standard"
    optimizer_iterations: int = 1
    optimizer_elite_count: int = 2
    constraints: Dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_values(self) -> "GenerationRequest":
        if self.count <= 0:
            raise ValueError("count must be positive")
        if self.seed_start < 0:
            raise ValueError("seed_start must not be negative")
        if self.run_id and not RUN_ID_PATTERN.fullmatch(self.run_id):
            raise ValueError(
                "run_id must start with an alphanumeric character and contain only "
                "letters, numbers, '.', '_' or '-'"
            )
        if self.optimizer_iterations <= 0:
            raise ValueError("optimizer_iterations must be positive")
        if self.optimizer_elite_count <= 0:
            raise ValueError("optimizer_elite_count must be positive")
        return self


class GenerationParameters(BaseModel):
    count: int
    seed_start: int
    ruleset_id: str
    profile_id: str
    run_mode: Literal["standard", "optimizer"] = "standard"
    optimizer_iterations: int = 1
    optimizer_elite_count: int = 2
    constraints: Dict[str, Any] | None = None

    model_config = ConfigDict(extra="ignore")


class GenerationRecord(BaseModel):
    seed: int
    build_id: str
    precheck_failures: List[str] = Field(default_factory=list)
    evaluation_status: str | None = None
    evaluation_error: Dict[str, Any] | None = None
    constraint_status: str | None = None
    constraint_reason_code: str | None = None
    violated_constraints: List[str] = Field(default_factory=list)
    constraint_checked_at: str | None = None

    model_config = ConfigDict(extra="ignore")


class GenerationSummary(BaseModel):
    count: int
    processed: int
    precheck_failures: Dict[str, int]
    attempted: int | None = None
    attempt_records: List[GenerationRecord] = Field(default_factory=list)
    records: List[GenerationRecord] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class EvaluationRecord(BaseModel):
    build_id: str
    status: str | None = None
    error: Dict[str, Any] | None = None

    model_config = ConfigDict(extra="ignore")


class EvaluationSummary(BaseModel):
    attempted: int
    successes: int
    failures: int
    errors: int
    records: List[EvaluationRecord] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class ArchiveAxis(BaseModel):
    name: str
    metric_key: str
    bins: int
    min_value: float
    max_value: float

    model_config = ConfigDict(extra="ignore")


class ArchiveMetrics(BaseModel):
    bins_filled: int
    total_bins: int
    coverage: float
    qd_score: float

    model_config = ConfigDict(extra="ignore")


class ArtifactLink(BaseModel):
    label: str
    url: str

    model_config = ConfigDict(extra="ignore")


class ArchiveBinDetail(BaseModel):
    bin_key: str
    build_id: str
    score: float
    descriptor: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tradeoff_reasons: List[str] = Field(default_factory=list)
    artifact_links: List[ArtifactLink] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class ArchiveSummary(BaseModel):
    metrics: ArchiveMetrics
    axes: List[ArchiveAxis]
    created_at: str

    model_config = ConfigDict(extra="ignore")


class ArchiveSummaryResponse(BaseModel):
    run_id: str
    created_at: str
    axes: List[ArchiveAxis]
    metrics: ArchiveMetrics
    bins: List[ArchiveBinDetail]

    model_config = ConfigDict(extra="ignore")


class ArchiveEmitterSummary(BaseModel):
    name: str
    budget: int
    selected: int

    model_config = ConfigDict(extra="ignore")


class BenchmarkScenarioSummary(BaseModel):
    samples: int
    median_full_dps: float
    median_max_hit: float
    median_utility_score: float
    gate_pass_rate: float

    model_config = ConfigDict(extra="ignore")


class BenchmarkSummary(BaseModel):
    generated_at: str
    scenarios: Dict[str, BenchmarkScenarioSummary] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore")


class MLLifecycleModel(BaseModel):
    model_id: str | None = None
    path: str | None = None

    model_config = ConfigDict(extra="ignore")


class MLLifecycleMetadata(BaseModel):
    model_meta: Dict[str, Any] | None = None
    meta_path: str | None = None
    error: str | None = None

    model_config = ConfigDict(extra="ignore")


class MLLifecycleSummary(BaseModel):
    enabled: bool
    status: str
    model: MLLifecycleModel
    selection_params: Dict[str, Any]
    counts: Dict[str, int]
    fallback_reason: str | None = None
    metadata: MLLifecycleMetadata
    created_at: str

    model_config = ConfigDict(extra="ignore")


class GenerationRunSummary(BaseModel):
    run_id: str
    status: str
    status_reason: Dict[str, Any] | None = None
    created_at: str
    parameters: GenerationParameters
    generation: GenerationSummary
    evaluation: EvaluationSummary
    archive: ArchiveSummary | None = None
    emitters: List[ArchiveEmitterSummary] = Field(default_factory=list)
    paths: Dict[str, str] = Field(default_factory=dict)
    benchmark: BenchmarkSummary | None = None
    ml_lifecycle: MLLifecycleSummary | None = None
    optimizer: Dict[str, Any] | None = None
    constraints: Dict[str, Any] | None = None

    model_config = ConfigDict(extra="ignore")


class ModelOpsModelRecord(BaseModel):
    path: str
    model_id: str | None = None
    trained_at_utc: str | None = None
    metadata: Dict[str, Any] | None = None

    model_config = ConfigDict(extra="ignore")


class ModelOpsArtifactState(BaseModel):
    state: Literal["available", "missing"]
    path: str | None = None
    timestamp: str | None = None
    metadata: Dict[str, Any] | None = None

    model_config = ConfigDict(extra="ignore")


class ModelOpsStatusResponse(BaseModel):
    generated_at: str
    active_model: ModelOpsModelRecord | None = None
    last_training: ModelOpsModelRecord | None = None
    checkpoint: ModelOpsArtifactState
    rollback: ModelOpsArtifactState
    warnings: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class MLLoopStatusResponse(BaseModel):
    generated_at: str
    loop_id: str | None = None
    status: str | None = None
    phase: str | None = None
    iteration: int | None = None
    total_iterations: int | None = None
    stop_requested: bool | None = None
    last_run_id: str | None = None
    last_snapshot_id: str | None = None
    last_error: str | None = None
    last_improvement: Dict[str, Any] | None = None
    latest_iteration: Dict[str, Any] | None = None
    previous_iteration: Dict[str, Any] | None = None
    best_iteration: int | None = None
    best_pass_probability_mean: float | None = None
    best_metric_mae: Dict[str, float] = Field(default_factory=dict)
    pass_probability_trend: List[Dict[str, float | int]] = Field(default_factory=list)
    gate_pass_rate_trend: List[Dict[str, float | int]] = Field(default_factory=list)
    diversity_trend: List[Dict[str, float | int]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class MLLoopStartRequest(BaseModel):
    loop_id: str
    profile_id: str = "pinnacle"
    count: int = 5
    seed_start: int = 1
    surrogate_backend: Literal["auto", "cpu", "cuda"] = "auto"
    endless: bool = True
    iterations: int | None = None

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="after")
    def validate_iterations(self) -> "MLLoopStartRequest":
        if not self.endless and (self.iterations is None or self.iterations <= 0):
            raise ValueError("iterations must be provided and positive when endless is false")
        return self


class MLLoopStartResponse(BaseModel):
    loop_id: str
    status: Literal["started"] = "started"
    endless: bool
    iterations: int | None = None
    pid: int | None = None
    message: str | None = None
    warnings: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class MLLoopStopRequest(BaseModel):
    loop_id: str | None = None

    model_config = ConfigDict(extra="ignore")


class MLLoopStopResponse(BaseModel):
    loop_id: str
    stop_requested: bool
    message: str | None = None

    model_config = ConfigDict(extra="ignore")
