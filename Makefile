CLICKHOUSE_SERVICE ?= clickhouse
UI_DIR ?= ui
BACKEND_DIR ?= backend
BACKEND_PORT ?= 8000
UV ?= uv
BACKEND_RUN ?= VIRTUAL_ENV= $(UV) --project $(BACKEND_DIR) run
BACKEND_RUN_DEV ?= VIRTUAL_ENV= $(UV) --project $(BACKEND_DIR) run --extra dev
ML_LOOP_ID ?= ml-loop
ML_LOOP_ITERATIONS ?= 0
ML_LOOP_COUNT ?= 128
ML_LOOP_DATA_PATH ?= .
ML_LOOP_STATUS_FORMAT ?= human
ML_LOOP_LAST ?=
ML_LOOP_LAST_ARG = $(if $(strip $(ML_LOOP_LAST)),--last $(ML_LOOP_LAST))
ML_LOOP_REPORT_DIR ?= reports/$(ML_LOOP_ID)
ML_LOOP_REPORT_FORMAT ?= json
ML_LOOP_REPORT_OUT ?= $(ML_LOOP_REPORT_DIR)/report.$(ML_LOOP_REPORT_FORMAT)
ML_LOOP_BUNDLE_OUT ?= $(ML_LOOP_REPORT_DIR)/bundle.tar.gz
TORCH_CPU_PACKAGES ?= torch torchvision torchaudio
TORCH_CUDA_PACKAGES ?= torch torchvision torchaudio
TORCH_CUDA_INDEX_URL ?= https://download.pytorch.org/whl/cu118
TORCH_CUDA_INDEX_ARG = $(if $(strip $(TORCH_CUDA_INDEX_URL)),--extra-index-url $(TORCH_CUDA_INDEX_URL))

.PHONY: clean db-up db-down db-init db-check dev backend-dev ui-dev test backend-test ui-test lint backend-lint ui-lint fmt backend-fmt ui-fmt preflight regen-baselines ml-loop-start ml-loop-stop ml-loop-status ml-loop-report ml-loop-bundle torch-install-cpu torch-install-cuda torch-verify

clean:
	@echo "Removing ML run/build artifacts under $(ML_LOOP_DATA_PATH)"
	rm -rf "$(ML_LOOP_DATA_PATH)/runs" "$(ML_LOOP_DATA_PATH)/ml_loops" "$(ML_LOOP_DATA_PATH)/data/builds"

soft-clean:
	@echo "Soft-cleaning, deleting /runs and /ml_loops. Keeping /data/builds"
	rm -rf "$(ML_LOOP_DATA_PATH)/runs" "$(ML_LOOP_DATA_PATH)/ml_loops" 

# ClickHouse lifecycle

db-up:
	docker-compose up -d $(CLICKHOUSE_SERVICE)

db-down:
	docker-compose down

db-check:
	@docker-compose exec -T $(CLICKHOUSE_SERVICE) clickhouse-client --user default --password default --query "SELECT 1" >/dev/null

db-init: db-check
	@$(BACKEND_RUN) python -m backend.tools.apply_schema

# Development servers (backend + UI)
dev:
	@echo "Verifying ClickHouse availability before starting dev servers"
	@if $(MAKE) db-check >/dev/null 2>&1; then \
		echo "ClickHouse reachable; launching backend and UI dev servers (Ctrl+C twice to stop)"; \
	else \
		echo "ClickHouse unreachable; run 'make db-up' and wait for healthcheck before 'make dev'" >&2; \
		exit 1; \
	fi
	@$(MAKE) -j2 --output-sync=line backend-dev ui-dev

backend-dev:
	$(BACKEND_RUN) python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port $(BACKEND_PORT) --workers 1 --loop uvloop --http httptools --lifespan on

ui-dev:
	npm --prefix $(UI_DIR) run dev

# Tests

test: backend-test ui-test

backend-test:
	VIRTUAL_ENV= $(UV) --project $(BACKEND_DIR) run --extra dev --extra ml python -m pytest $(BACKEND_DIR)/tests -q -rs

ui-test:
	npm --prefix $(UI_DIR) run lint
	npm --prefix $(UI_DIR) run typecheck

regen-baselines:
	$(BACKEND_RUN) python -m backend.tools.regen_baselines

# ML loop tool
ml-loop-start:
	$(BACKEND_RUN) python -m backend.tools.ml_loop start --loop-id $(ML_LOOP_ID) --iterations $(ML_LOOP_ITERATIONS) --count $(ML_LOOP_COUNT) --data-path $(ML_LOOP_DATA_PATH) $(if $(ML_LOOP_GATE),--gate-profile $(ML_LOOP_GATE),) $(if $(ML_LOOP_LEVEL_INTERVAL),--level-interval $(ML_LOOP_LEVEL_INTERVAL),)

ml-loop-stop:
	$(BACKEND_RUN) python -m backend.tools.ml_loop stop --loop-id $(ML_LOOP_ID) --data-path $(ML_LOOP_DATA_PATH)

ml-loop-status:
	$(BACKEND_RUN) python -m backend.tools.ml_loop status --loop-id $(ML_LOOP_ID) --data-path $(ML_LOOP_DATA_PATH) --format $(ML_LOOP_STATUS_FORMAT)

ml-loop-report:
	@mkdir -p "$(ML_LOOP_REPORT_DIR)"
	$(BACKEND_RUN) python -m backend.tools.ml_loop report \
		--loop-id $(ML_LOOP_ID) \
		--data-path $(ML_LOOP_DATA_PATH) \
		--format $(ML_LOOP_REPORT_FORMAT) \
		--out $(ML_LOOP_REPORT_OUT) \
		$(ML_LOOP_LAST_ARG)

ml-loop-bundle:
	@mkdir -p "$(ML_LOOP_REPORT_DIR)"
	$(BACKEND_RUN) python -m backend.tools.ml_loop bundle \
		--loop-id $(ML_LOOP_ID) \
		--data-path $(ML_LOOP_DATA_PATH) \
		--out $(ML_LOOP_BUNDLE_OUT) \
		$(ML_LOOP_LAST_ARG)

# Torch helpers (optional install)
torch-install-cpu:
	$(BACKEND_RUN) python -m pip install $(TORCH_CPU_PACKAGES)

torch-install-cuda:
	$(BACKEND_RUN) python -m pip install $(TORCH_CUDA_PACKAGES) $(TORCH_CUDA_INDEX_ARG)

torch-verify:
	$(BACKEND_RUN) python -m backend.tools.torch_verify

# Linters and formatters

lint: backend-lint ui-lint

backend-lint:
	$(BACKEND_RUN_DEV) python -m ruff check $(BACKEND_DIR)

ui-lint:
	npm --prefix $(UI_DIR) run lint

fmt: backend-fmt ui-fmt

backend-fmt:
	$(BACKEND_RUN_DEV) python -m ruff format $(BACKEND_DIR)

ui-fmt:
	npm --prefix $(UI_DIR) run format

preflight:
	@echo "Running Atlas/operator preflight (fail-fast):"
	@echo "  1) db-check"
	@$(MAKE) db-check || { echo "Atlas/operator preflight step 1 failed: ClickHouse check failed. Run 'make db-up', wait for healthcheck, then run 'make db-init', then rerun 'make preflight'" >&2; exit 1; }
	@echo "  2) backend-lint"
	@$(MAKE) backend-lint || { echo "Atlas/operator preflight step 2 failed: Backend lint failed. Run 'make backend-lint' and fix issues, then rerun 'make preflight'" >&2; exit 1; }
	@echo "  3) ui-test"
	@$(MAKE) ui-test || { echo "Atlas/operator preflight step 3 failed: UI check failed. Run 'make ui-test' and address issues, then rerun 'make preflight'" >&2; exit 1; }
	@echo "Atlas/operator preflight passed."
