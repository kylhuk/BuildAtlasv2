# Phase 1 Baseline Protocol

This protocol defines the reproducible baseline used before Phase 2+ changes.

## Locked baseline inputs (show-time default)

- `profile_id`: `mapping`
- `ruleset_id`: `standard`
- `count`: `3`
- `seed_start`: `42` (seeds `42,43,44`)
- `data_path`: `data`
- `run_id`s: `phase1-e2-baseline-map-a`, `phase1-e2-baseline-map-b`

## Preflight

1. `make db-up`
2. `make db-init`
3. `make db-check`

## Baseline command sequence

```bash
python -m backend.tools.generate_runs \
  --count 3 \
  --seed-start 42 \
  --profile-id mapping \
  --ruleset-id standard \
  --run-id phase1-e2-baseline-map-a \
  --data-path data

python -m backend.tools.generate_runs \
  --count 3 \
  --seed-start 42 \
  --profile-id mapping \
  --ruleset-id standard \
  --run-id phase1-e2-baseline-map-b \
  --data-path data
```

Expected run artifacts (per run under `data/runs/<run_id>/`):

- `summary.json` (run-level provenance plus paths to other artifacts).
- `archive.json` (quality-diversity archive contents produced in Phase 5).
- `benchmark_summary.json` (per-scenario benchmark stats derived from PoB rows).
- `ml_lifecycle.json` (surrogate training/inference metadata and selection history).
- `surrogate_predictions.json` (written only when the surrogate is enabled and records the predicted population).

`summary.json` keys now consistently cover the run lifecycle: `run_id`, `status`, `created_at`, `parameters` (including `count`, `seed_start`, `profile_id`, `ruleset_id`, `constraints`), `generation`, `evaluation`, `surrogate`, `optimizer`, `archive`, `emitters`, `benchmark`, `ml_lifecycle`, `constraints`, and `paths`. Subfields such as `paths.summary`, `paths.archive`, `paths.benchmark_summary`, `paths.ml_lifecycle`, and `paths.surrogate_predictions` point back at the accompanying artifacts.

Generated build artifacts remain under `data/builds/<build_id>/...`, and the constraint metadata for each build is published as `data/builds/<build_id>/constraints.json`.

## Reproducibility checks

Compare run A vs run B for the same seeds:

1. Metric tolerance check (per seed/scenario):
   - `full_dps`, `max_hit`, `utility_score`
   - pass if absolute percentage delta is `<= 0.5%`.
2. Artifact parity check (per seed):
   - `build.code.txt`, `genome.json`, `metrics_raw.json`, `scenarios_used.json`
   - pass if SHA-256 matches for corresponding seed outputs.
3. Timestamp tolerance:
   - `summary.created_at` delta `<= 10s`
   - `archive.created_at` delta `<= 10s`.

## Evidence bundle

Store proof in:

- `evidence/phase_1/epic_2/bundle_<timestamp>/commands.log`
- `evidence/phase_1/epic_2/bundle_<timestamp>/queries.log`
- `evidence/phase_1/epic_2/bundle_<timestamp>/api_samples.json`
- `evidence/phase_1/epic_2/bundle_<timestamp>/artifacts_index.json`
- optional `notes.md`

## Important current-state note

Phase 1 baseline still reflects current implementation behavior, including placeholder worker evaluation path and synthetic generation metrics. Real PoB-backed metrics are delivered in Phase 2.
