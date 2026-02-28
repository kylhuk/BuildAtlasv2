# Phase 1 Trial Provenance Schema

This schema documents how trial provenance is represented today and how additive fields will be layered in later phases.

All additions must remain additive (no renames/drops of existing keys).

## Scope

- Current mapping source of truth:
  - `data/runs/<run_id>/summary.json`
  - `GET /runs/{run_id}`
  - `GET /archives/{run_id}`
- Baseline examples used in this document:
  - `phase1-e2-baseline-map-a`
  - `phase1-e2-baseline-map-b`

Phase 5 run summaries now encode optimizer state, archive provenance, benchmark snapshots, ML lifecycle metadata, constraint references, and artifact paths. Treat these additions as optional, additive extensions and keep the worker NDJSON/JSON-RPC `ping`/`evaluate` contract unchanged.

## Required fields (current)

| Field | Type | Summary path | API path | Meaning |
| --- | --- | --- | --- | --- |
| `run_id` | string | `run_id` | `runs.<run_id>.run_id` | Stable run identifier |
| `status` | string | `status` | `runs.<run_id>.status` | Run lifecycle status |
| `created_at` | datetime | `created_at` | `runs.<run_id>.created_at` | Run creation time |
| `parameters.count` | int | `parameters.count` | `runs.<run_id>.parameters.count` | Number of generated candidates |
| `parameters.seed_start` | int | `parameters.seed_start` | `runs.<run_id>.parameters.seed_start` | Seed start |
| `parameters.profile_id` | string | `parameters.profile_id` | `runs.<run_id>.parameters.profile_id` | Scenario profile |
| `parameters.ruleset_id` | string | `parameters.ruleset_id` | `runs.<run_id>.parameters.ruleset_id` | Ruleset identifier |
| `generation.records[].seed` | int | `generation.records[].seed` | `runs.<run_id>.generation.records[].seed` | Per-trial seed |
| `generation.records[].build_id` | string | `generation.records[].build_id` | `runs.<run_id>.generation.records[].build_id` | Per-trial build id |
| `generation.records[].evaluation_status` | string | `generation.records[].evaluation_status` | `runs.<run_id>.generation.records[].evaluation_status` | Trial evaluation state |
| `evaluation.records[].status` | string | `evaluation.records[].status` | `runs.<run_id>.evaluation.records[].status` | Evaluation status by build |
| `archive.metrics` | object | `archive.metrics` | `runs.<run_id>.archive.metrics` and `archives.<run_id>.metrics` | Archive quality/coverage metrics |
| `paths.archive` | string | `paths.archive` | `runs.<run_id>.paths.archive` | Archive artifact path |
| `paths.summary` | string | `paths.summary` | `runs.<run_id>.paths.summary` | Run summary artifact path |
| `paths.benchmark_summary` | string | `paths.benchmark_summary` | `runs.<run_id>.paths.benchmark_summary` | Benchmark summary artifact |
| `paths.ml_lifecycle` | string | `paths.ml_lifecycle` | `runs.<run_id>.paths.ml_lifecycle` | ML lifecycle artifact |
| `paths.surrogate_predictions` | string (conditional) | `paths.surrogate_predictions` | `runs.<run_id>.paths.surrogate_predictions` | Surrogate prediction snapshot (when enabled) |

## Optional fields (current)

| Field | Type | Source | Notes |
| --- | --- | --- | --- |
| `surrogate.*` | object | `summary.json` | Present when surrogate fields are emitted; currently disabled in baseline runs |
| `generation.records[].evaluation_error` | string/null | `summary.json` | Null when evaluations succeed |
| `evaluation.records[].error` | string/null | `summary.json` | Null when evaluations succeed |
| `archives.<run_id>.bins[].metadata` | object | `GET /archives/{run_id}` | Contains per-bin trial metadata (`seed`, `metrics_source`, `status`, etc.) |

## Summary schema highlights

- `parameters.constraints` and the top-level `constraints` mirror the generator budget/legality gates recorded for this run; they remain additive metadata.
- `generation` links each requested seed to a `build_id`, while `evaluation.records` records per-build status (only builds with non-empty `status`/`error` fields are captured to keep summaries compact).
- `surrogate` contains the inference status + selection params, and when the surrogate is enabled we also write `surrogate_predictions.json` for the predicted candidates.
- `optimizer` provides run-mode context (`standard` vs `optimizer`), iteration counts, elite summaries, and stage history whenever the optimizer loop is active.
- `archive`, `benchmark`, and `ml_lifecycle` mirror their associated disk artifacts (`archive.json`, `benchmark_summary.json`, `ml_lifecycle.json`).
- `paths.*` keys resolve the artifact locations: `summary`, `archive`, `benchmark_summary`, `ml_lifecycle`, and `surrogate_predictions` (conditional).

## Additive target fields (not yet implemented)

ClickHouse `builds` columns for constraint metadata are now implemented via `sql/clickhouse/004_constraint_metadata.sql`; they are additive and do not change the worker protocol. Remaining fields are still additive targets.

| Field | Layer | Status |
| --- | --- | --- |
| `constraint_status` | ClickHouse/API | implemented (see `sql/clickhouse/004_constraint_metadata.sql`) |
| `constraint_reason_code` | ClickHouse/API | implemented (see `sql/clickhouse/004_constraint_metadata.sql`) |
| `violated_constraints` | ClickHouse/API | implemented (see `sql/clickhouse/004_constraint_metadata.sql`) |
| `constraint_checked_at` | ClickHouse/API | implemented (see `sql/clickhouse/004_constraint_metadata.sql`) |
| `run_id` column in ClickHouse run-linked tables | ClickHouse | target |
| `seed` column for trial-level joins | ClickHouse | target |
| `artifact_path` for direct artifact dereference | ClickHouse/API | target |

## Linkage example (current)

For `phase1-e2-baseline-map-a`:

1. `summary.json` maps `seed -> build_id` under `generation.records`.
2. Each `build_id` resolves to artifact files under `data/builds/<build_id>/`, with `constraints.json` documenting the latest constraint evaluation status.
3. `evaluation.records` and `archive` confirm trial status/coverage for those builds.
4. `GET /runs/phase1-e2-baseline-map-a` mirrors the summary payload for API consumers.
5. `GET /archives/phase1-e2-baseline-map-a` provides archive bins keyed by descriptors.

This same linkage holds for `phase1-e2-baseline-map-b`, enabling reproducibility comparison across runs with identical input seeds.

## Constraint metadata artifact

- `data/builds/<build_id>/constraints.json` mirrors the constraint metadata recorded in ClickHouse and is written after each evaluation. Its additive shape is a payload produced by `backend.engine.constraints.constraint_artifact_payload`:
  ```json
  {
    "schema_version": 1,
    "spec": {
      "schema_version": 1,
      "rules": [
        {
          "code": "constraint-id",
          "metric_path": ["metrics", "value"],
          "operator": "<=",
          "threshold": 100.0,
          "scenario_id": "scenario",
          "description": "optional",
          "reason_code": "constraint-id",
          "missing_data_reason": "optional"
        }
      ]
    },
    "evaluation": {
      "status": "pass",
      "reason_code": "constraints_met",
      "violated_constraints": [],
      "checked_at": "2026-02-26T12:34:56.789Z",
      "details": [
        {
          "code": "constraint-id",
          "operator": "<=",
          "threshold": 100.0,
          "value": 95.2,
          "satisfied": true
        }
      ]
    }
  }
  ```

  `spec` reflects `ConstraintSpec.to_payload()`: it always exposes `schema_version` plus the normalized `rules` list (each rule keeps `code`, `metric_path`, `operator`, `threshold`, and any optional `scenario_id`, `description`, `reason_code`, `missing_data_reason`).

  `evaluation` exposes `status`, `reason_code`, `violated_constraints`, `checked_at`, and the `details` array generated by `ConstraintEvaluation.to_payload()`. Each detail records the evaluated `code`, `operator`, `threshold`, the resolved `value`, and optional `scenario_id`, `description`, and `satisfied` flag.

  This file stays additive relative to the existing worker payload and keeps per-build constraint provenance synced with ClickHouse.

## Evidence pointers

- `python -m backend.tools.apply_schema` applies `sql/clickhouse/004_constraint_metadata.sql` plus earlier DDLs.
- `clickhouse-client --query "DESCRIBE TABLE builds" | grep constraint` shows the implemented columns (`constraint_status`, `constraint_reason_code`, `violated_constraints`, `constraint_checked_at`).

## Frontier-only archive view

- `GET /archives/{run_id}/frontier` mirrors the base `summary.json` payload but filters `bins` down to the Pareto frontier in the axis space defined by the archive `axes` array. Frontier bins inherit all archive metadata plus two derived fields: `tradeoff_reasons` and `artifact_links`.
- When the archive `axes` array is missing or empty, the frontier response bypasses Pareto filtering and returns every bin so clients can still surface data without axis metadata.
- `tradeoff_reasons` is an ordered list emitted by the backend: it begins with the sentinel `pareto_frontier`, appends `<axis>_focus` tags for the axis(es) with the highest descriptor value, and finally records score context with `high_score`, `low_score`, or `score_missing` depending on whether `score` is present and positive/negative. The list is deduplicated to keep UI rendering stable.
- `artifact_links` enumerates quick navigation anchors for the frontier candidate (`build_detail`, `export_xml`, `export_code`), all rooted at `/builds/{build_id}` so operators can jump from the dashboard directly into stored artifacts.
- Additive compatibility note: the frontier endpoint only adds a filtered view of bins plus the decorated fields; the original `/archives/{run_id}` payload and all worker/ClickHouse contracts remain unchanged, so clients can upgrade incrementally.

## Evidence pointers (frontier sample)

```bash
curl -s http://localhost:8000/archives/phase1-e2-baseline-map-a/frontier | jq '.'
```

## Verification commands

```bash
curl -s http://localhost:8000/runs/phase1-e2-baseline-map-a
curl -s http://localhost:8000/archives/phase1-e2-baseline-map-a
python - <<'PY'
import json
from pathlib import Path
summary = json.loads(Path('data/runs/phase1-e2-baseline-map-a/summary.json').read_text())
print(summary['run_id'], summary['status'])
print(summary['parameters'])
print(summary['generation']['records'])
PY
```
