# EP-V7 Upgrade Checklist

Use this checklist when rolling a new PoB/ruleset revision across the single-machine PoC.

## 1) PoB bump workflow
1. Update the PoB submodule and pin `pob/VERSION` to the release commit.
2. Confirm scenario templates are still valid for the target profile set (`mapping`, `pinnacle`, `uber`, `delve`, `support`).
3. Apply additive schema migrations before running new evaluations:
   - `python -m backend.tools.apply_schema`
4. Record the new `ruleset_id` components (`pob` commit, scenario version token, price snapshot id) in release notes.

## 2) Regression baseline regeneration
1. Regenerate locked baselines from fixture metrics:
   - `python -m backend.tools.regen_baselines`
2. Verify `backend/tests/fixtures/regression_baselines.json` and `backend/tests/fixtures/regression_rulesets.json` are updated together.
3. Re-run regression checks:
   - `pytest backend/tests/test_regression_baselines.py`

## 3) Elite/ruleset reevaluation workflow
1. Dry-run target selection first:
   - `python -m backend.tools.reevaluate_builds --profile-id pinnacle --ruleset-filter "pob:old|scenarios:pinnacle@v0|prices:demo"`
2. Apply reevaluation with stale marking when ready:
   - `python -m backend.tools.reevaluate_builds --profile-id pinnacle --ruleset-id "pob:<new>|scenarios:pinnacle@v0|prices:demo" --apply --mark-stale`
3. Keep historical rows; do not delete prior `scenario_metrics` data.

## 4) Drift report generation
1. Generate a drift report for the run/ruleset transition:
   - `python -m backend.tools.drift_report <run_id> --old-ruleset "pob:<old>|scenarios:pinnacle@v0|prices:demo" --new-ruleset "pob:<new>|scenarios:pinnacle@v0|prices:demo"`
2. Report is written to:
   - `data/runs/<run_id>/drift_report.md`
3. Optionally add extra metrics:
   - `python -m backend.tools.drift_report <run_id> --old-ruleset ... --new-ruleset ... --metric armour --metric evasion`

## 5) Surrogate model retraining workflow
1. Build/refresh a dataset snapshot:
   - `python -m backend.tools.build_dataset_snapshot --data-path data --output-root data/datasets/ep-v4`
2. Train a surrogate model from the snapshot:
   - `python -m backend.tools.train_surrogate train --dataset-path data/datasets/ep-v4/<snapshot_id> --output-root data/models/ep-v4`
3. Re-run surrogate evaluation as needed:
   - `python -m backend.tools.train_surrogate eval --model-path data/models/ep-v4/<model_id> --dataset-path data/datasets/ep-v4/<snapshot_id>`

## Compatibility guardrails
- Keep ClickHouse schema changes additive only.
- Keep ruleset/scenario identifiers stable and coordinated across backend/UI/tooling.
- Keep PoB worker NDJSON/JSON-RPC payloads backward compatible.
- Keep filesystem artifact contracts under `data/builds`, `data/datasets`, and `data/runs` stable.
