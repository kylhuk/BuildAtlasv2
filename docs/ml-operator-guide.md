# ML Operator Guide

This guide is the practical loop for running generation + surrogate training on one machine, then checking whether quality is improving.

## Prerequisites
- Start core services: `make db-up` and `make db-init`.
- Make sure PoB assets exist: `git submodule update --init --recursive`.
- Optional while iterating: `make dev`, `make test`, `make lint`, `make fmt`.

## Atlas/operator preflight (fail-fast)
- Run this command before Atlas/operator runs that touch generation or model updates:
  - `make preflight`
- The sequence is intentionally strict and fail-fast:
  1. `make db-check`
  2. `make backend-lint`
  3. `make ui-test`
- The preflight output stops at the first failed step and points you back to the exact command to rerun before retrying the full sequence.

## Commit guardrail
- Before committing, review `git diff --stat` and stop if scope drift appears.

## Automated loop (start/stop/status)
Use the new loop tool when you want one command to keep generating, training, and reporting improvement.

1. **Start loop (Make wrapper)**
   ```bash
   make ml-loop-start ML_LOOP_ID=ml-loop-001 ML_LOOP_ITERATIONS=5 ML_LOOP_COUNT=200
   ```

2. **Stop loop**
   ```bash
   make ml-loop-stop ML_LOOP_ID=ml-loop-001
   ```

3. **Check loop status**
   ```bash
   make ml-loop-status ML_LOOP_ID=ml-loop-001
   ```

Python equivalents (same behavior):
```bash
python -m backend.tools.ml_loop start --loop-id ml-loop-001 --iterations 5 --count 200 --data-path .
python -m backend.tools.ml_loop stop --loop-id ml-loop-001 --data-path .
python -m backend.tools.ml_loop status --loop-id ml-loop-001 --data-path .
```

Loop checkpoints and stats are written to:
- `ml_loops/<loop_id>/state.json`
- `ml_loops/<loop_id>/iterations.jsonl`
- `ml_loops/<loop_id>/checkpoints/iter-XXXX.json`

## End-to-end ML loop
1. **Generate PoB-verified builds**
   ```bash
   python -m backend.tools.generate_runs \
     --run-id ml-loop-001 \
     --data-path . \
     --count 200
   ```
   This writes run artifacts under `runs/ml-loop-001/` and build artifacts under `data/builds/`.

2. **Create a dataset snapshot from generated builds**
   ```bash
   python -m backend.tools.build_dataset_snapshot \
     --data-path . \
     --output-root data/datasets/ep-v4 \
     --snapshot-id ml-loop-001
   ```
   Snapshot output: `data/datasets/ep-v4/ml-loop-001/` (`dataset.jsonl` + `manifest.json`).

3. **Train a surrogate model**
   ```bash
   python -m backend.tools.train_surrogate train \
     --dataset-path data/datasets/ep-v4/ml-loop-001 \
     --output-root data/models/ep-v4 \
     --model-id ml-loop-001
   ```
   The command prints JSON including `model_path`, `metrics_path`, and `meta_path`.

4. **Evaluate the trained model**
   ```bash
   python -m backend.tools.train_surrogate eval \
     --model-path data/models/ep-v4/ml-loop-001/model.json \
     --dataset-path data/datasets/ep-v4/ml-loop-001
   ```
   Eval output includes `metric_mae` and `pass_probability` summaries.

5. **Optional: run surrogate-guided generation**
   ```bash
   python -m backend.tools.generate_runs \
     --run-id ml-loop-002 \
     --data-path . \
     --count 200 \
     --surrogate-enabled true \
     --surrogate-model-path data/models/ep-v4/ml-loop-001/model.json
   ```

6. **Optional: compare drift between rulesets**
   ```bash
   python -m backend.tools.drift_report ml-loop-002 \
     --old-ruleset <old_ruleset_id> \
     --new-ruleset <new_ruleset_id>
   ```

## How to tell if it is getting better
- **Model quality:** compare `train_surrogate eval` outputs across runs. Better typically means lower `metric_mae` and stronger `pass_probability` distribution.
- **Generation quality:** inspect `runs/<run_id>/summary.json` and track `evaluation.successes`, `evaluation.failures`, and `status`/`status_reason`.
- **Attempt efficiency:** in `summary.json`, check `generation.attempt_records` and count how many attempts have `persisted=true`.
- **Automated loop trend:** inspect `ml_loops/<loop_id>/iterations.jsonl`; each line has `evaluation.current`, `evaluation.previous`, and `evaluation.improvement` (including `improved`).
- Keep a simple run log (run_id -> dataset snapshot -> model_path -> eval metrics -> generation success/failure counts).

## Troubleshooting: all builds failed
1. Confirm PoB worker processes are running and healthy.
2. Open `runs/<run_id>/summary.json` and inspect `status_reason`.
3. Check `generation.attempt_records`:
   - `persisted=false` means the attempt failed validation and was purged.
   - `evaluation_error` gives the immediate failure reason.
4. Retry with a smaller batch first (`--count 10`) to get quick feedback.
5. If failures persist, share the run `summary.json`, eval output JSON, and worker/backend logs for debugging.
