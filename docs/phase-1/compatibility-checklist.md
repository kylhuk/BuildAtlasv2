# Phase 1 Compatibility Freeze Checklist

This checklist captures the current contract surfaces before Phase 2+ implementation.

non-additive changes require explicit written approval.

## 1) PoB worker NDJSON / JSON-RPC contract

- [x] Current request methods are `ping` and `evaluate` (`pob/worker/worker.lua`).
- [x] Current `evaluate` validation requires `params.xml` or `params.code`; otherwise error `1003`.
- [x] Current `evaluate` success payload is placeholder (`status: "placeholder success"`) with payload metadata only (`type`, `has_xml`, `has_code`).
- [ ] Optional additive target fields (not implemented yet): `run_id`, `seed`, `artifact_path`, `constraint_status`, `constraint_reason_code`, `violated_constraints`.

Current minimal examples:

```json
{"id":1,"method":"ping","params":{}}
```

```json
{"id":2,"method":"evaluate","params":{"code":"<share-code>"}}
```

## 2) ClickHouse baseline schema (current)

- [x] `builds` table baseline is defined in `sql/clickhouse/001_init.sql` with columns:
  `build_id`, `created_at`, `ruleset_id`, `profile_id`, `class`, `ascendancy`, `main_skill`,
  `damage_type`, `defence_type`, `complexity_bucket`, `pob_xml_path`, `pob_code_path`,
  `genome_path`, `tags`, `status`, `is_stale`.
- [x] `scenario_metrics` baseline is defined in `sql/clickhouse/001_init.sql` with columns:
  `build_id`, `ruleset_id`, `scenario_id`, `gate_pass`, `gate_fail_reasons`, `pob_warnings`,
  `evaluated_at`, `full_dps`, `max_hit`, `armour`, `evasion`, `life`, `mana`, `utility_score`.
- [x] `build_costs` baseline is defined in `sql/clickhouse/002_build_costs.sql` with columns:
  `build_id`, `ruleset_id`, `price_snapshot_id`, `total_cost_chaos`, `unknown_cost_count`,
  `slot_costs_json_path`, `gem_costs_json_path`, `calculated_at`.
- [x] Constraint metadata columns are live: `constraint_status`, `constraint_reason_code`,
-   `violated_constraints`, `constraint_checked_at` (applied via `sql/clickhouse/004_constraint_metadata.sql`).
- [ ] Remaining additive target fields (still pending): `run_id`, `seed`, `artifact_path`.

## 3) Filesystem artifact contract (current)

- [x] Build artifact root is `<base_path>/data/builds/<build_id>/` (`backend/engine/artifacts/store.py`).
- [x] Current build artifact files:
  - `build.xml.gz`
  - `build.code.txt`
  - `genome.json`
  - `scenarios_used.json`
  - `metrics_raw.json`
  - `surrogate_prediction.json`
- [x] Run summary root is `<base_path>/runs/<run_id>/summary.json` (`backend/app/main.py`).
- [x] Run-level artifacts now include `summary.json`, `archive.json`, `benchmark_summary.json`, `ml_lifecycle.json`, and conditional `surrogate_predictions.json` under `data/runs/<run_id>/`.
- [x] Constraint metadata file is `data/builds/<build_id>/constraints.json` (mirrors ClickHouse constraint columns; schema_version/spec/evaluation with `status`, `reason`, `violated_constraints`, `checked_at`).
- [ ] Optional additive target fields/files (not implemented yet): run-level `artifact_path`,
  constraint metadata files/keys, extended run provenance keys.

## 4) FastAPI surface contract (current)

- [x] Health and generation:
  - `GET /health`
  - `POST /generation`
  - `GET /runs/{run_id}`
- [x] Build lifecycle and evaluation:
  - `POST /import`
  - `POST /evaluate/{build_id}`
  - `POST /evaluate-batch`
  - `GET /builds`
  - `GET /builds/{build_id}`
  - `GET /builds/{build_id}/scenarios`
  - `GET /builds/{build_id}/export/xml`
  - `GET /builds/{build_id}/export/code`
- [x] Archive visibility:
  - `GET /archives/{run_id}`
  - `GET /archives/{run_id}/bins/{bin_key}`
- [x] Frontier decorations:
  - `GET /archives/{run_id}/frontier` (adds filtered frontier bins, `tradeoff_reasons`, and `artifact_links` without touching the existing summary contract)
- [ ] Optional additive target response fields (not implemented yet): `constraint_status`,
  `violation_reason`, `reason_code`, `checked_at`, composition-level `build_details` blocks.

## 5) Additivity guard checks

- [x] Preserve existing keys/columns/files in-place.
- [x] Add new keys/columns/files only as optional additive extensions.
- [x] Keep scenario/ruleset identifiers stable.
- [x] Keep worker `ping`/`evaluate` compatible while expanding payloads.

## 6) Evidence to collect during Phase 1 verification

- [ ] Worker payload proof:
  - send `ping` and `evaluate` NDJSON requests to `pob/worker/worker.lua` and capture responses.
- [ ] ClickHouse schema proof:
- [ ] ClickHouse schema proof:
  - `python -m backend.tools.apply_schema` (re-runs `sql/clickhouse/*`; ensures `004_constraint_metadata.sql` is applied).
  - `clickhouse-client --query "DESCRIBE TABLE builds"` (expect `constraint_status`, `constraint_reason_code`, `violated_constraints`, `constraint_checked_at`).
  - `DESCRIBE TABLE scenario_metrics`
  - `DESCRIBE TABLE build_costs`
- [ ] Filesystem proof:
  - list sample `data/builds/<build_id>/` contents
  - inspect sample `data/runs/<run_id>/summary.json`
- [ ] FastAPI surface proof:
  - `curl /health`
  - `curl /runs/{run_id}` (if run exists)
  - `curl /builds/{build_id}` (if build exists)
