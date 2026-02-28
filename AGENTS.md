# AGENTS

This repository uses a multi-agent workflow tuned for the single-machine PoB Build Generator PoC.

## Project baseline

- Source of truth: `pob_build_generator_single_source_of_truth.md`.
- Core stack: Python orchestrator/engine, FastAPI API, LuaJIT PoB worker pool, ClickHouse metrics/index tables, filesystem artifacts, and React+TypeScript UI.
- PoB is the authoritative stat oracle; ML is only a surrogate/proposal layer.
- `make` is the primary interface (`make dev`, `make test`, `make fmt`, `make lint`).

## Agent roster

- `proj-lead`: default orchestrator; runs todo-driven plan-build-verify loops and closes work only with evidence.
- `planner`: planning-only decomposition with explicit acceptance criteria and evidence requirements.
- `go-fast`: Python implementation builder with minimal diffs.
- `go-tests`: Python tests/fixtures only.
- `proto-engineer`: ClickHouse and wire-contract compatibility guard (PoB NDJSON/JSON-RPC and related schema surfaces).
- `devops`: Docker/Make/CI/tooling plumbing.
- `runner`: command execution and output capture only.
- `review`: read-only risk and compatibility audit.
- `docs`: documentation-only updates.

## Working agreement

- Keep schema and wire-contract changes additive unless there is explicit written approval.
- Do not claim validation unless command output is captured; mark true non-applicable checks as `N/A`.
- Keep diffs scoped and avoid drive-by refactors.
- Re-check assumptions against `pob_build_generator_single_source_of_truth.md` when architecture details are unclear.
