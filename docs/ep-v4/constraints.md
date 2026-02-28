# EP-V4 Surrogate Constraints

This file documents the constraints that keep the EP-V4 surrogate work aligned with the single-machine PoB oracle PoC. The implementation is intentionally additive, keeps existing contracts stable, and falls back to the oracle wherever the surrogate cannot safely assert confidence.

## Compatibility surfaces

- **ClickHouse remains untouched.** EP-V4 writes only to filesystem overlays (`data/datasets/ep-v4/...`). No new tables or mutations of existing `builds`/`scenario_metrics` schemas are introduced in this phase.
- **Filesystem artifacts are authoritative.** All features come from the existing artifact contract (`data/builds/<build_id>/genome.json` and `metrics_raw.json`). The snapshotting tooling never alters those inputs; it simply reads them and emits JSONL rows.
- **Ruleset & PoB worker payload contracts stay stable.** Scenario templates, ruleset identifiers (`pob:<commit>|scenarios:<version>|prices:<id>`), and the NDJSON request/response surface of the LuaJIT worker remain fixed. EP-V4 consumes scenario metrics exactly as PoB produced them and does not introduce new fields that would require a worker change.
- **New dataset artifacts live under `data/datasets/ep-v4/<snapshot_id>/`.** The builder writes a `dataset.jsonl` (one JSON object per line) and a `manifest.json` that records metadata plus a deterministic checksum. Each snapshot is additive, self-contained, and referenced only by the surrogate layer—no other subsystem modifies historical snapshots.

## Dataset snapshot baseline

- **Feature schema version `v1`.** Each row contains `build_id`, `scenario_id`, genome descriptors (`class`, `ascendancy`, `main_skill_package`, `defense_archetype`, `budget_tier`, `profile_id`), normalized PoB metrics (`full_dps`, `max_hit`, `utility_score`, `armour`, `evasion`, resource reserves, resists, attributes, etc.), and reservation/utilization stats.
- **Deterministic ordering.** Rows are sorted by `build_id`, and each build’s scenarios are sorted lexicographically to keep the dataset hash reproducible. The manifest stores `row_count`, the `source_root_path` (absolute `data/builds`), `generated_at_utc`, `feature_schema_version`, and `dataset_hash` (SHA-256 over the newline-delimited JSONL rows).
- **JSONL is the current baseline.** This lightweight format keeps dependencies minimal and ensures compatibility with downstream tooling. Future exports (Parquet, Arrow, etc.) must explicitly document any differences in ordering or schema.
- **Builder CLI.** `backend/tools/build_dataset_snapshot.py` exposes a `--data-path`, `--output-root`, and optional `--snapshot-id`. It prints a JSON summary of the snapshot (paths, row count, hash) so operators can log or pipe the outcome into orchestration tooling.

## Fallback model strategy

- **Surrogate proposals are suggestions, not authoritative evaluations.** If the dataset is empty (no builds with both `genome.json` and `metrics_raw.json`) or if downstream confidence checks fail, the execution path falls back to calling PoB through the worker pool. The surrogate only gates candidates when it can reference the deterministic feature set described above.
- **Compatibility guardrails.** Any change that would require new ClickHouse columns, modified artifact names, or altered PoB worker NDJSON fields must be treated as a schema or wire-contract change and escalated to the protocol guard (`proto-engineer`). EP-V4 avoids these changes by reading only established files and exporting additive snapshots.

See `docs/ep-v7/upgrade-checklist.md` for the full EP-V7 upgrade workflow that maintains these guardrails.
