const DEFAULT_API_BASE_URL = 'http://localhost:8000';
const EXPLICIT_API_BASE_URL = import.meta.env.VITE_API_BASE_URL as string | undefined;
const LEGACY_API_BASE_URL = import.meta.env.VITE_BACKEND_URL as string | undefined;
const API_BASE_URL = EXPLICIT_API_BASE_URL || DEFAULT_API_BASE_URL;
const API_FALLBACK_URL =
  !EXPLICIT_API_BASE_URL && LEGACY_API_BASE_URL && LEGACY_API_BASE_URL !== API_BASE_URL
    ? LEGACY_API_BASE_URL
    : null;

let activeApiBaseUrl = API_BASE_URL;

const buildUrl = (path: string, baseUrl: string = activeApiBaseUrl) => {
  try {
    return new URL(path, baseUrl).toString();
  } catch (err) {
    throw new Error(`invalid API path ${path}`);
  }
};

async function fetchWithFallback(path: string, options: RequestInit): Promise<Response> {
  const primaryUrl = buildUrl(path, activeApiBaseUrl);
  try {
    return await fetch(primaryUrl, options);
  } catch (primaryError) {
    if (!API_FALLBACK_URL || API_FALLBACK_URL === activeApiBaseUrl) {
      throw new Error(`Failed to reach API at ${activeApiBaseUrl}`);
    }
    const fallbackUrl = buildUrl(path, API_FALLBACK_URL);
    try {
      const response = await fetch(fallbackUrl, options);
      activeApiBaseUrl = API_FALLBACK_URL;
      return response;
    } catch {
      throw new Error(`Failed to reach API at ${activeApiBaseUrl} or fallback ${API_FALLBACK_URL}`);
    }
  }
}

async function safeJson(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

const extractErrorMessage = (payload: Record<string, unknown> | null): string | null => {
  if (!payload) return null;
  const error = payload.error as Record<string, unknown> | undefined;
  if (error && typeof error.message === 'string') {
    return error.message;
  }
  if (typeof payload.message === 'string') {
    return payload.message;
  }
  return null;
};

async function requestJson<T>(path: string, options: RequestInit = {}): Promise<T> {
  const method = (options.method || 'GET').toUpperCase();
  const hasBody = options.body != null;
  const headers = new Headers(options.headers ?? {});
  if (hasBody && method !== 'GET' && method !== 'HEAD' && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }
  const response = await fetchWithFallback(path, {
    headers,
    ...options,
  });
  if (!response.ok) {
    const payload = (await safeJson(response)) as Record<string, unknown> | null;
    const message =
      extractErrorMessage(payload) ?? response.statusText ?? `Request failed (${response.status})`;
    throw new Error(message);
  }
  if (response.status === 204) {
    return {} as T;
  }
  return (await response.json()) as T;
}

async function requestText(path: string, options: RequestInit = {}): Promise<string> {
  const response = await fetchWithFallback(path, options);
  if (!response.ok) {
    const payload = (await safeJson(response)) as Record<string, unknown> | null;
    const message =
      extractErrorMessage(payload) ?? response.statusText ?? `Request failed (${response.status})`;
    throw new Error(message);
  }
  return response.text();
}

export type BuildStatus = 'imported' | 'queued' | 'evaluated' | 'failed';

export type ConstraintStatus = 'pass' | 'fail' | 'unknown';

export interface ConstraintRule {
  code: string;
  metric_path: string[];
  operator: string;
  threshold: number;
  scenario_id?: string | null;
  description?: string | null;
  reason_code?: string | null;
  missing_data_reason?: string | null;
}

export interface ConstraintSpec {
  schema_version?: number;
  rules: ConstraintRule[];
  [key: string]: unknown;
}

export interface ConstraintEvaluationDetail {
  code: string;
  operator: string;
  threshold: number;
  value: number | null;
  scenario_id?: string | null;
  description?: string | null;
  satisfied?: boolean | null;
}

export interface ConstraintEvaluation {
  status: ConstraintStatus;
  reason_code: string;
  violated_constraints: string[];
  checked_at: string;
  details: ConstraintEvaluationDetail[];
  [key: string]: unknown;
}

export interface ConstraintPayload {
  schema_version?: number;
  spec?: ConstraintSpec;
  evaluation?: ConstraintEvaluation;
  [key: string]: unknown;
}

export interface ScenarioMetricSummary {
  ruleset_id: string;
  scenario_id: string;
  gate_pass: boolean;
  gate_fail_reasons: string[];
  pob_warnings: string[];
  evaluated_at: string;
  full_dps: number;
  max_hit: number;
  armour: number;
  evasion: number;
  life: number;
  mana: number;
  utility_score: number;
}

export interface BuildSummary {
  build_id: string;
  created_at: string;
  ruleset_id: string;
  profile_id: string;
  status: BuildStatus;
  is_stale?: boolean;
  class_: string;
  ascendancy: string;
  main_skill: string;
  damage_type: string;
  defence_type: string;
  complexity_bucket: string;
  pob_xml_path: string;
  pob_code_path: string;
  genome_path: string;
  tags: string[];
  price_snapshot_id?: string;
  total_cost_chaos?: number;
  unknown_cost_count?: number;
  dps_per_chaos?: number;
  max_hit_per_chaos?: number;
  constraint_status?: ConstraintStatus | null;
  constraint_reason_code?: string | null;
  violated_constraints?: string[];
  constraint_checked_at?: string | null;
  prediction?: PredictionSummary | null;
}
export interface PredictionSummary {
  model_id?: string | null;
  model_path?: string | null;
  predicted_full_dps?: number | null;
  predicted_max_hit?: number | null;
  pass_probability?: number | null;
  selection_reason?: string | null;
  timestamp?: string | null;
}

export interface PredictionDetail extends PredictionSummary {
  verified_full_dps?: number | null;
  verified_max_hit?: number | null;
  error_full_dps?: number | null;
  error_max_hit?: number | null;
}

export interface SlotCostDetail {
  slot: string;
  name: string;
  cost_chaos?: number | null;
  matched: boolean;
}

export interface GemCostDetail {
  name: string;
  level?: number | null;
  quality?: number | null;
  price?: number | null;
  matched: boolean;
}

export interface BuildCostDetail {
  price_snapshot_id?: string;
  total_cost_chaos?: number;
  unknown_cost_count?: number;
  slot_costs: SlotCostDetail[];
  gem_costs: GemCostDetail[];
}

export interface ScenarioUsed {
  scenario_id: string;
  version: string;
  profile_id: string;
  pob_config: Record<string, unknown>;
  gate_thresholds: Record<string, unknown>;
}

export interface BuildDetailIdentity {
  class?: string;
  ascendancy?: string;
  main_skill?: string;
  profile_id?: string;
  defense_archetype?: string;
  budget_tier?: string;
  sources?: Record<string, string>;
}

export interface BuildDetailsPayload {
  schema_version?: number;
  source?: string;
  identity?: BuildDetailIdentity;
  items?: Record<string, unknown>;
  passives?: {
    node_ids?: string[];
    required_targets?: string[];
    [key: string]: unknown;
  };
  gems?: {
    groups?: Array<Record<string, unknown>>;
    full_dps_group_id?: string | null;
    gem_link_count?: number;
    [key: string]: unknown;
  };
  exports?: {
    xml_available?: boolean;
    code_available?: boolean;
    share_code_available?: boolean;
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

export interface BuildDetailResponse {
  build: BuildSummary;
  prediction?: PredictionDetail | null;
  scenario_metrics: ScenarioMetricSummary[];
  scenarios_used: ScenarioUsed[];
  costs?: BuildCostDetail | null;
  build_details?: BuildDetailsPayload | null;
  constraints?: ConstraintPayload | null;
}

export interface ImportBuildMetadata {
  ruleset_id: string;
  profile_id: string;
  class: string;
  ascendancy: string;
  main_skill: string;
  damage_type: string;
  defence_type: string;
  complexity_bucket: string;
  tags: string[];
}

export interface ImportBuildPayload {
  share_code?: string;
  xml?: string;
  metadata: ImportBuildMetadata;
}

export interface ImportBuildResponse {
  build_id: string;
  status: BuildStatus;
  pob_xml_path: string;
  pob_code_path: string;
  genome_path: string;
}

export interface EvaluateResponse {
  build_id: string;
  status: BuildStatus;
  scenario_results: ScenarioMetricSummary[];
}

export interface EvaluateBatchRequest {
  build_ids: string[];
}

export interface EvaluateBatchResult {
  build_id: string;
  status?: BuildStatus;
  scenario_results?: ScenarioMetricSummary[];
  error?: {
    code: string;
    message: string;
    details?: unknown;
  };
}

export interface EvaluateBatchResponse {
  results: EvaluateBatchResult[];
}

export interface BuildScenariosResponse {
  build_id: string;
  scenarios_used: ScenarioUsed[];
}

export interface GenerationRequest {
  count: number;
  seed_start: number;
  ruleset_id: string;
  profile_id: string;
  run_id?: string;
}

export interface GenerationRecord {
  seed: number;
  build_id: string;
  precheck_failures: string[];
  evaluation_status?: string | null;
  evaluation_error?: Record<string, unknown> | null;
}

export interface GenerationSummary {
  count: number;
  processed: number;
  precheck_failures: Record<string, number>;
  records: GenerationRecord[];
}

export interface EvaluationRecord {
  build_id: string;
  status?: string | null;
  error?: Record<string, unknown> | null;
}

export interface EvaluationSummary {
  attempted: number;
  successes: number;
  failures: number;
  errors: number;
  records: EvaluationRecord[];
}

export interface GenerationRunSummary {
  run_id: string;
  status: string;
  created_at: string;
  parameters: {
    count: number;
    seed_start: number;
    ruleset_id: string;
    profile_id: string;
  };
  generation: GenerationSummary;
  evaluation: EvaluationSummary;
  paths: Record<string, string>;
}

export interface ArchiveAxis {
  name: string;
  metric_key: string;
  bins: number;
  min_value: number;
  max_value: number;
}

export interface ArchiveMetrics {
  bins_filled: number;
  total_bins: number;
  coverage: number;
  qd_score: number;
}

export interface ArtifactLink {
  label: string;
  url: string;
}

export type ArchiveArtifactLinks = ArtifactLink[] | Record<string, string>;

export interface ArchiveBinDetail {
  bin_key: string;
  build_id: string;
  score: number;
  descriptor: Record<string, number>;
  metadata: Record<string, unknown>;
  tradeoff_reasons?: string[];
  artifact_links?: ArchiveArtifactLinks;
}

export interface ArchiveSummaryResponse {
  run_id: string;
  created_at: string;
  axes: ArchiveAxis[];
  metrics: ArchiveMetrics;
  bins: ArchiveBinDetail[];
}

export interface ModelOpsModelRecord {
  path: string;
  model_id?: string | null;
  trained_at_utc?: string | null;
  metadata?: Record<string, unknown> | null;
}

export interface ModelOpsArtifactState {
  state: 'available' | 'missing';
  path?: string | null;
  timestamp?: string | null;
  metadata?: Record<string, unknown> | null;
}

export interface ModelOpsStatusResponse {
  generated_at: string;
  active_model?: ModelOpsModelRecord | null;
  last_training?: ModelOpsModelRecord | null;
  checkpoint: ModelOpsArtifactState;
  rollback: ModelOpsArtifactState;
  warnings: string[];
}

export interface BuildInventoryStatsResponse {
  generated_at: string;
  total_builds: number;
  stale_builds: number;
  status_counts: Record<string, number>;
}

export interface MLLoopStatusResponse {
  generated_at: string;
  loop_id?: string | null;
  status?: string | null;
  phase?: string | null;
  iteration?: number | null;
  total_iterations?: number | null;
  stop_requested?: boolean | null;
  last_run_id?: string | null;
  last_snapshot_id?: string | null;
  last_error?: string | null;
  last_improvement?: Record<string, unknown> | null;
  latest_iteration?: Record<string, unknown> | null;
  previous_iteration?: Record<string, unknown> | null;
  best_iteration?: number | null;
  best_pass_probability_mean?: number | null;
  best_metric_mae?: Record<string, number>;
  pass_probability_trend?: Array<Record<string, number>>;
  gate_pass_rate_trend?: Array<Record<string, number>>;
  diversity_trend?: Array<Record<string, number>>;
  warnings: string[];
}

export interface MLLoopStartRequest {
  loop_id: string;
  profile_id?: string;
  count?: number;
  seed_start?: number;
  surrogate_backend?: 'auto' | 'cpu' | 'cuda';
  endless?: boolean;
  iterations?: number;
}

export interface MLLoopStartResponse {
  loop_id: string;
  status: 'started';
  endless: boolean;
  iterations?: number | null;
  pid?: number | null;
  message?: string | null;
  warnings?: string[];
}

export interface MLLoopStopRequest {
  loop_id?: string;
}

export interface MLLoopStopResponse {
  loop_id: string;
  stop_requested: boolean;
  message?: string | null;
}

export interface BuildListParams {
  rulesetId?: string;
  profileId?: string;
  status?: BuildStatus;
  predictionMode?: 'include_predicted' | 'verified_only';
  className?: string;
  ascendancy?: string;
  scenarioId?: string;
  gatePass?: boolean;
  maxCostChaos?: number;
  excludeUnknownCost?: boolean;
  priceSnapshotId?: string;
  costCalculatedAfter?: string;
  costCalculatedBefore?: string;
  includeStale?: boolean;
  sortBy?: string;
  sortDir?: 'asc' | 'desc';
  limit?: number;
  offset?: number;
  constraintStatus?: ConstraintStatus;
  constraintReasonCode?: string;
  violatedConstraint?: string;
  constraintCheckedAfter?: string;
  constraintCheckedBefore?: string;
}

export interface CatalogColumn {
  id: string;
  label: string;
  data_type: string;
  sortable: boolean;
  filterable: boolean;
  source?: string | null;
  index_priority?: 'high' | 'medium' | 'low' | null;
  unit?: string | null;
}

export interface CatalogColumnsRegistryResponse {
  registry_version: string;
  columns: CatalogColumn[];
}

interface BuildListResponse {
  builds: BuildSummary[];
}

export async function listBuilds(params: BuildListParams = {}): Promise<BuildSummary[]> {
  const searchParams = new URLSearchParams();
  if (params.rulesetId) searchParams.append('ruleset_id', params.rulesetId);
  if (params.profileId) searchParams.append('profile_id', params.profileId);
  if (params.status) searchParams.append('status', params.status);
  if (params.predictionMode) searchParams.append('prediction_mode', params.predictionMode);
  if (params.className) searchParams.append('class', params.className);
  if (params.ascendancy) searchParams.append('ascendancy', params.ascendancy);
  if (params.scenarioId) searchParams.append('scenario_id', params.scenarioId);
  if (params.gatePass !== undefined) searchParams.append('gate_pass', String(params.gatePass));
  if (params.maxCostChaos !== undefined)
    searchParams.append('max_cost_chaos', params.maxCostChaos.toString());
  if (params.excludeUnknownCost !== undefined)
    searchParams.append('exclude_unknown_cost', String(params.excludeUnknownCost));
  if (params.priceSnapshotId) searchParams.append('price_snapshot_id', params.priceSnapshotId);
  if (params.costCalculatedAfter)
    searchParams.append('cost_calculated_after', params.costCalculatedAfter);
  if (params.costCalculatedBefore)
    searchParams.append('cost_calculated_before', params.costCalculatedBefore);
  if (params.includeStale !== undefined)
    searchParams.append('include_stale', String(params.includeStale));
  if (params.sortBy) searchParams.append('sort_by', params.sortBy);
  if (params.sortDir) searchParams.append('sort_dir', params.sortDir);
  if (params.limit !== undefined) searchParams.append('limit', params.limit.toString());
  if (params.offset !== undefined) searchParams.append('offset', params.offset.toString());
  if (params.constraintStatus) searchParams.append('constraint_status', params.constraintStatus);
  if (params.constraintReasonCode) searchParams.append('constraint_reason_code', params.constraintReasonCode);
  if (params.violatedConstraint) searchParams.append('violated_constraint', params.violatedConstraint);
  if (params.constraintCheckedAfter) searchParams.append('constraint_checked_after', params.constraintCheckedAfter);
  if (params.constraintCheckedBefore) searchParams.append('constraint_checked_before', params.constraintCheckedBefore);
  const query = searchParams.toString();
  const path = query ? `/builds?${query}` : '/builds';
  const response = await requestJson<BuildListResponse>(path);
  return response.builds;
}

export async function getBuildDetail(buildId: string): Promise<BuildDetailResponse> {
  return requestJson<BuildDetailResponse>(`/builds/${buildId}`);
}

export async function getBuildScenarios(buildId: string): Promise<BuildScenariosResponse> {
  return requestJson<BuildScenariosResponse>(`/builds/${buildId}/scenarios`);
}

export async function importBuild(payload: ImportBuildPayload): Promise<ImportBuildResponse> {
  return requestJson<ImportBuildResponse>(`/import`, {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function evaluateBuild(buildId: string): Promise<EvaluateResponse> {
  return requestJson<EvaluateResponse>(`/evaluate/${buildId}`, {
    method: 'POST',
  });
}

export async function evaluateBatch(buildIds: string[]): Promise<EvaluateBatchResponse> {
  return requestJson<EvaluateBatchResponse>(`/evaluate-batch`, {
    method: 'POST',
    body: JSON.stringify({ build_ids: buildIds }),
  });
}

export async function fetchCodeExport(buildId: string): Promise<string> {
  return requestText(`/builds/${buildId}/export/code`);
}

export async function fetchXmlExport(buildId: string): Promise<string> {
  return requestText(`/builds/${buildId}/export/xml`);
}

export function xmlExportUrl(buildId: string): string {
  return buildUrl(`/builds/${buildId}/export/xml`);
}

export function codeExportUrl(buildId: string): string {
  return buildUrl(`/builds/${buildId}/export/code`);
}

export async function generateRun(payload: GenerationRequest): Promise<GenerationRunSummary> {
  return requestJson<GenerationRunSummary>(`/generation`, {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function fetchGenerationRun(runId: string): Promise<GenerationRunSummary> {
  return requestJson<GenerationRunSummary>(`/runs/${runId}`);
}

export async function fetchArchive(runId: string): Promise<ArchiveSummaryResponse> {
  return requestJson<ArchiveSummaryResponse>(`/archives/${runId}`);
}

export async function fetchArchiveFrontier(runId: string): Promise<ArchiveSummaryResponse> {
  return requestJson<ArchiveSummaryResponse>(`/archives/${runId}/frontier`);
}

export async function fetchArchiveBin(runId: string, binKey: string): Promise<ArchiveBinDetail> {
  return requestJson<ArchiveBinDetail>(`/archives/${runId}/bins/${binKey}`);
}

export async function fetchCatalogColumnsRegistry(): Promise<CatalogColumnsRegistryResponse> {
  return requestJson<CatalogColumnsRegistryResponse>('/catalog/columns/v1');
}

export async function fetchModelOpsStatus(): Promise<ModelOpsStatusResponse> {
  return requestJson<ModelOpsStatusResponse>('/ops/model-status');
}

export async function fetchBuildInventoryStats(): Promise<BuildInventoryStatsResponse> {
  return requestJson<BuildInventoryStatsResponse>('/builds/stats');
}

export async function fetchMLLoopStatus(loopId?: string): Promise<MLLoopStatusResponse> {
  if (loopId) {
    const query = new URLSearchParams({ loop_id: loopId }).toString();
    return requestJson<MLLoopStatusResponse>(`/ops/ml-loop-status?${query}`);
  }
  return requestJson<MLLoopStatusResponse>('/ops/ml-loop-status');
}

export async function startMLLoop(payload: MLLoopStartRequest): Promise<MLLoopStartResponse> {
  return requestJson<MLLoopStartResponse>('/ops/ml-loop-start', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function stopMLLoop(payload: MLLoopStopRequest = {}): Promise<MLLoopStopResponse> {
  return requestJson<MLLoopStopResponse>('/ops/ml-loop-stop', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}
