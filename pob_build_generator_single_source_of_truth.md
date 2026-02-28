# PoB Build Generator (PoC, single machine) — Single Source of Truth

Last updated: 2026-02-25 (Europe/Zurich)

This document is the single source of truth for a local (single-machine) PoC tool that:
- generates Path of Exile builds (passive tree + skill gems + items),
- evaluates them via Path of Building (PoB) as the authoritative stat oracle,
- estimates build cost (gems + uniques first),
- and provides a local build library UI for filtering/sorting and exporting back to PoB.

It is derived from the attached v1.1 concept/plan and the deep research report, but locks the technology and operations to a small PoC you run for yourself only.

---

## 1) Hard constraints and non-goals (PoC rules)

Runs only on your machine.
- No multi-user auth, no external hosting, no Kubernetes.

PoB is the only ground-truth.
- The system never re-implements PoB calculations.
- ML is used only as a surrogate/proposal mechanism to reduce PoB calls and explore the build space.

Storage and artifacts
- ClickHouse for metrics/indexing and fast filtering/sorting.
- Flat filesystem for artifacts (PoB XML/share code, genomes, scenario manifests, price snapshots, model snapshots, datasets).

Ops/tooling (intentionally minimal)
- Docker Compose only for ClickHouse.
- Makefile is the interface (`make dev`, `make test`, `make fmt`, `make lint`).
- No GitHub Actions, Prometheus, Grafana.

Compliance guardrails (even for PoC)
- Treat PoB + local artifacts as primary.
- Pricing: use cached snapshots (poe.ninja first). Avoid undocumented endpoints.

---

## 2) Technology set (locked for PoC)

PoB evaluation (given)
- Path of Building Community fork, pinned to a commit/tag in-repo.
- LuaJIT runtime.
- Headless worker entrypoint (worker mode over stdin/stdout, NDJSON protocol).

Backend
- Python 3.x.
- FastAPI + Uvicorn.
- ClickHouse client: pick one and standardize:
  - `clickhouse-connect` (HTTP) OR
  - `clickhouse-driver` (native).
- Concurrency: `asyncio` + `concurrent.futures` + `multiprocessing`.
- ML (later phases): LightGBM + PyTorch + NumPy.

UI
- React + TypeScript.
- Vite.
- Ant Design (tables, forms, drawers).

Dev/test tooling
- Makefile.
- Python: `pytest`, `ruff`.
- UI: `eslint`, `prettier`, `tsc --noEmit`.

---

## 3) Evaluation model (viability, scenarios, gates)

Key rule: viability is content-profile specific, and PoB evaluation must be scenario-explicit (no reliance on PoB defaults).

### 3.1 Profiles (minimum set for PoC v1)
- Mapping (e.g., T16): clear-oriented, frequent hits, uptime.
- Pinnacle boss: single-target oriented.
- Uber pinnacle boss: stricter variant, evaluated under explicit Uber config.

Later (after v1 is stable):
- Delve (depth tiers).
- Support/party.

### 3.2 Scenario templates (versioned)
Each scenario template is a named, versioned bundle of PoB configuration inputs (boss preset, debuffs, uptime assumptions, etc.).

Every stored metric row includes:
- `ruleset_id` (PoB version + scenario template version + price snapshot id),
- `scenario_id`,
- `gate_pass` + `gate_fail_reasons[]`.

### 3.3 Normalized metrics (schema v0)
Store a stable, typed column set per scenario:
- Offense: Full DPS (and/or hit/DoT breakdown where relevant), conservative vs. optimistic variants only if you explicitly store both.
- Defense: effective max hit taken (phys/ele/chaos proxies), key mitigation layers (armour/evasion/suppression/block where available).
- Recovery: regen/leech/recoup proxies; flask-dependence flags (optional).
- Utility: movement speed proxies, reservation usage proxies, button/complexity proxy (later).

### 3.4 Hard gates (pass/fail)
Hard gates are pure functions applied to PoB outputs, and failures are stored as explicit reasons.

Example gate families:
- resistance caps/met (and optional overcap buffer),
- minimum max-hit thresholds (profile-specific),
- minimum sustain/recovery thresholds,
- “structural legality” flags (reservation feasible, sockets/links feasible, attribute requirements met).

---

## 4) Architecture overview (single-machine PoC)

Components
- PoB Worker Pool (LuaJIT subprocesses): stat oracle.
- Backend Engine (Python): import, generation, evaluation scheduling, scoring, archive maintenance, pricing, dataset building, model training.
- ClickHouse: queryable index for metrics/tags and fast sorting/filtering.
- Filesystem artifacts: build XML/share-code, genomes, scenario manifests, price caches, model/dataset snapshots.
- UI (React): build catalogue + details + export.

Core loop (eventually)
1) Candidate source: imported builds OR generated builds.
2) Structural checks (fast) → discard obvious invalids early.
3) PoB evaluation (headless) under scenario templates.
4) Store artifacts on disk; store metrics/tags in ClickHouse.
5) UI queries ClickHouse and reads artifacts as needed.
6) Later: surrogate scoring selects which candidates to send to PoB.

---

## 5) Repo structure (suggested)

```
repo/
  backend/
    app/                # FastAPI + services
    engine/             # generation, evaluation, search, ML
    tools/              # CLI utilities (bulk import, rebuild, etc.)
    tests/
  ui/
    src/
  pob/
    PathOfBuilding/     # pinned checkout (submodule or vendored)
    worker/             # headless wrapper + patches + launch scripts
  sql/
    clickhouse/         # DDLs
  data/
    builds/             # build_id/ artifacts
    prices/             # cached snapshots
    models/             # surrogate models
    datasets/           # parquet/jsonl
    runs/               # run summaries
  docker-compose.yml
  Makefile
```

---

## 6) Data model (ClickHouse) and artifact contract (filesystem)

### 6.1 ClickHouse tables (PoC minimal)

`builds`
- `build_id` (UUID/string)
- `created_at`
- `ruleset_id`
- `profile_id` (mapping/boss/uber/delve/support)
- tags: `class`, `ascendancy`, `main_skill`, `damage_type`, `defence_type`, `complexity_bucket`, etc.
- artifact paths: `pob_xml_path`, `pob_code_path`, `genome_path`
- `status` (imported/generated/evaluated/failed)
- constraint metadata: `constraint_status`, `constraint_reason_code`, `violated_constraints`, `constraint_checked_at` (added via `sql/clickhouse/004_constraint_metadata.sql`). These columns are additive and can be verified with `python -m backend.tools.apply_schema` followed by `clickhouse-client --query "DESCRIBE TABLE builds"` to confirm the new columns exist.

Engine: `MergeTree` ORDER BY `(profile_id, ruleset_id, created_at)`.

`scenario_metrics`
- `build_id`
- `ruleset_id`
- `scenario_id`
- key metrics as typed columns
- `gate_pass` (UInt8)
- `gate_fail_reasons` (Array(String))
- `pob_warnings` (Array(String))
- `evaluated_at`

Engine: `MergeTree` ORDER BY `(scenario_id, ruleset_id, gate_pass, evaluated_at)`.

`build_costs` (added in Vertical Phase 3)
- `build_id`, `ruleset_id`, `price_snapshot_id`
- `total_cost_chaos`
- `slot_costs_json_path`, `gem_costs_json_path`

`runs` (optional but useful)
- `run_id`, `started_at`, `ended_at`
- parameters (profile, generator settings, budgets)
- counts (generated, evaluated, passed gates)
- summary file paths (on disk)

### 6.2 Filesystem artifacts (per build)

`data/builds/<build_id>/`
- `build.xml.gz` (canonical PoB export)
- `build.code.txt` (PoB share code)
- `genome.json` (when generated)
- `scenarios_used.json` (exact scenario inputs)
- `metrics_raw.json` (optional debug dump; useful during PoC)
- `constraints.json` (per-build constraint metadata with `schema_version`, `spec` + `evaluation` containing `status`, `reason`, `violated_constraints`, `checked_at`).

Run-level artifacts live under `data/runs/<run_id>/` and include `summary.json`, `archive.json`, `benchmark_summary.json`, `ml_lifecycle.json`, and, when surrogates are enabled, `surrogate_predictions.json`. The summary document aggregates parameters, constraints, generation/evaluation records, optimizer/archive reports, benchmark/ML lifecycle payloads, and artifact paths (`paths.summary`, `paths.archive`, `paths.benchmark_summary`, `paths.ml_lifecycle`, `paths.surrogate_predictions`).

---

## 7) Implementation phases (Foundation first, then vertical slices)

Each phase ends with:
- a working end-to-end user flow,
- measurable acceptance criteria,
- minimal new dependencies.

### Phase 0 — Foundation (the “spine”)

Goal
- Import a PoB build, evaluate headlessly under explicit scenarios, store in ClickHouse + filesystem, browse/sort in UI.

Acceptance
- Import a known PoB build and see stable metrics.
- Same build produces expected differences between mapping vs uber scenarios.
- Catalog sorting works and export back to PoB works.

### Vertical Phase 1 — Import-driven catalog + evaluation correctness

Goal
- Build confidence in oracle/scenarios by indexing real builds before generating anything.

Acceptance
- Bulk ingest 100+ builds and reliably filter/sort by scenario metrics.
- `make test` catches evaluation drift.

### Vertical Phase 2 — Minimal constrained generator (template-first)

Goal
- Generate complete, legal-looking builds without ML to create data and uncover legality pitfalls.

Acceptance
- Generate thousands of builds; a meaningful fraction evaluates in PoB.
- Failures are explainable via gate reasons (not silent breakage).

### Vertical Phase 3 — Pricing v1 (gems + uniques) + budget sorting

Goal
- Make cost real enough to sort by “cheap vs strong” and enable cost-aware search later.

Acceptance
- Sort by total cost and DPS/chaos with transparent breakdowns.

### Vertical Phase 4 — Surrogate model v0 (single-machine throughput unlock)

Goal
- Stop calling PoB for every candidate; enable large candidate counts via cheap inference.

Acceptance
- Generate 100k candidates while PoB-evaluating only a small subset, yet surfacing strong verified builds.

### Vertical Phase 5 — Quality-Diversity archive (avoid meta collapse)

Goal
- Produce a diverse library across budgets/playstyles/defence archetypes (not one best build).

Acceptance
- Archive contains many meaningfully different elites; browsing by category is useful.

### Vertical Phase 6 — Interaction intelligence (gem ↔ tree ↔ item coupling)

Goal
- Learn real interactions while still exploring novel combos.

Acceptance
- System rediscovers known synergies without hard-coding and validates non-obvious combos via novelty/uncertainty budget.

### Vertical Phase 7 — Expand profiles + patch drift workflow (still PoC)

Goal
- Keep it useful across PoB/PoE changes without adding ops complexity.

Acceptance
- After a PoB update, re-evaluate selected builds and keep catalog coherent.

---

## 8) Work breakdown: Epics → Tasks → Subtasks

Conventions
- Epics are the stable units (EP-…).
- Tasks are `EP-… .T#` and should be small enough to complete in 0.5–2 days.
- Subtasks are implementation checklist items.

### EP-FND-01 Monorepo + local dev workflow

- [ ] EP-FND-01.T1 Create repo scaffolding
  - [ ] Create directory structure (`backend/`, `ui/`, `pob/`, `sql/`, `data/`)
  - [ ] Add root `.gitignore` (include `data/` outputs but keep `data/.keep` placeholders)
  - [ ] Add root `README.md` (single paragraph + how to run)
- [ ] EP-FND-01.T2 Docker Compose for ClickHouse (local only)
  - [ ] Add `docker-compose.yml` with ClickHouse service + volume
  - [ ] Add `sql/clickhouse/` mount or an init script strategy
  - [ ] Verify container start/stop/reset flows
- [ ] EP-FND-01.T3 Makefile as single entry point
  - [ ] `make dev` (compose up, backend dev, ui dev)
  - [ ] `make test` (backend + ui)
  - [ ] `make fmt` / `make lint`
  - [ ] `make clean` (optional: wipe generated artifacts safely)
- [ ] EP-FND-01.T4 Local config approach (PoC-friendly)
  - [ ] Define `backend/app/settings.py` (env vars + defaults)
  - [ ] Define `ui/.env.local` conventions (API base URL)
  - [ ] Document required env vars (ClickHouse URL, PoB path, data path)
- [ ] EP-FND-01.T5 Dev ergonomics
  - [ ] Add `ruff.toml` + `pyproject.toml` (backend)
  - [ ] Add `eslint` + `prettier` configs (ui)
  - [ ] Make `make dev` fail fast if ClickHouse isn’t reachable

### EP-FND-02 ClickHouse schema + data access layer

- [ ] EP-FND-02.T1 Write initial DDLs
  - [ ] `builds` table DDL
  - [ ] `scenario_metrics` table DDL
  - [ ] Optional: `runs` table DDL
- [ ] EP-FND-02.T2 Idempotent schema apply
  - [ ] `sql/clickhouse/001_init.sql` (and future numbered files)
  - [ ] Implement `backend/tools/apply_schema.py` (runs all SQL files in order)
  - [ ] Wire to `make db-init`
- [ ] EP-FND-02.T3 DB access module
  - [ ] Create `backend/app/db/ch.py` (connect, execute, insert helpers)
  - [ ] Define typed insert payloads (Pydantic models) for `builds` and `scenario_metrics`
- [ ] EP-FND-02.T4 Query patterns for UI sorting/filtering
  - [ ] Build list query with filters (profile/scenario/gate_pass/etc.)
  - [ ] Sorting by key metrics (FullDPS, maxHit proxies, cost later)
  - [ ] Pagination strategy (offset/limit is OK for PoC)
- [ ] EP-FND-02.T5 Minimal data correctness tests
  - [ ] Unit test: insert then query back rows
  - [ ] Unit test: sorting order correctness

### EP-FND-03 Filesystem artifact store + provenance

- [ ] EP-FND-03.T1 Artifact layout contract
  - [ ] Define canonical paths under `data/builds/<build_id>/`
  - [ ] Define required files vs optional debug files
- [ ] EP-FND-03.T2 Read/write helpers
  - [ ] `write_build_artifacts(build_id, xml, code, genome, scenarios_used, raw_metrics)`
  - [ ] `read_build_artifacts(build_id)` for API/UI detail views
- [ ] EP-FND-03.T3 Hashing and provenance
  - [ ] Compute `build_hash` over canonical XML (or canonical genome)
  - [ ] Define `ruleset_id` format: `pob:<commit>|scenarios:<ver>|prices:<id>`
  - [ ] Store provenance fields in ClickHouse and/or `genome.json`
- [ ] EP-FND-03.T4 Safety and cleanup rules
  - [ ] Ensure “clean” targets do not delete inputs accidentally
  - [ ] Add `data/.keep` placeholders so folders exist in git

### EP-FND-04 PoB headless oracle (LuaJIT worker + Python pool)

- [ ] EP-FND-04.T1 Pin and vendor PoB
  - [ ] Add PoB Community fork as submodule or vendored directory
  - [ ] Record the pinned commit in `pob/VERSION`
- [ ] EP-FND-04.T2 Headless worker entrypoint
  - [ ] Implement `pob/worker/worker.lua` (NDJSON request/response)
  - [ ] Support inputs: PoB share code OR XML
  - [ ] Output: normalized metrics + warnings/errors
- [ ] EP-FND-04.T3 Scenario application layer (in worker)
  - [ ] Parse `scenario` object from request
  - [ ] Apply scenario to PoB config deterministically
  - [ ] Ensure “Include in Full DPS” is set deterministically for the main damage group
- [ ] EP-FND-04.T4 Python worker pool manager
  - [ ] Spawn N worker subprocesses
  - [ ] Health checks and restart on crash
  - [ ] Timeout handling per evaluation
  - [ ] Batch dispatch interface (`evaluate_many([...])`)
- [ ] EP-FND-04.T5 Metric extraction validation
  - [ ] Compare worker metrics to PoB GUI for 2–3 known builds
  - [ ] Document any known discrepancies and lock assumptions

### EP-FND-05 Scenario templates + normalized metrics + hard gates

- [ ] EP-FND-05.T1 Scenario JSON definitions (v0)
  - [ ] `mapping_t16_v0.json`
  - [ ] `pinnacle_v0.json`
  - [ ] `uber_pinnacle_v0.json`
- [ ] EP-FND-05.T2 Normalized metric schema v0
  - [ ] Define a minimal set of columns for `scenario_metrics`
  - [ ] Map raw PoB outputs → normalized column names
- [ ] EP-FND-05.T3 Gate functions (v0)
  - [ ] Resist caps gate
  - [ ] Reservation feasibility gate (coarse)
  - [ ] Attribute requirement gate (coarse)
  - [ ] Minimum max-hit thresholds per scenario (conservative defaults)
- [ ] EP-FND-05.T4 Store explicit failure reasons
  - [ ] Standardize failure reason strings (stable identifiers)
  - [ ] Ensure all failure paths write `gate_fail_reasons`
- [ ] EP-FND-05.T5 Scenario versioning
  - [ ] Scenario version string included in `ruleset_id`
  - [ ] Persist `scenarios_used.json` per build

### EP-FND-06 Backend API MVP

- [ ] EP-FND-06.T1 FastAPI project skeleton
  - [ ] Create app, routers, settings
  - [ ] Add OpenAPI tags and response models
- [ ] EP-FND-06.T2 Build import endpoint
  - [ ] Accept PoB share code OR XML upload
  - [ ] Create `build_id`, store artifacts, insert `builds` row
- [ ] EP-FND-06.T3 Evaluation endpoints
  - [ ] `POST /evaluate/{build_id}` (evaluate under all scenarios for profile)
  - [ ] `POST /evaluate-batch` (for bulk import/generation)
- [ ] EP-FND-06.T4 Query endpoints for UI
  - [ ] `GET /builds` (filters, sorting, pagination)
  - [ ] `GET /builds/{build_id}` (detail: metrics + artifact pointers)
- [ ] EP-FND-06.T5 Export endpoints
  - [ ] `GET /builds/{build_id}/export/xml`
  - [ ] `GET /builds/{build_id}/export/code`
- [ ] EP-FND-06.T6 Error handling
  - [ ] Return consistent error schema for PoB failures
  - [ ] Log worker stderr + attach warnings to `scenario_metrics`

### EP-FND-07 UI MVP (build catalogue)

- [ ] EP-FND-07.T1 UI scaffolding
  - [ ] Vite + React + TS project
  - [ ] Ant Design setup
  - [ ] API client wrapper (fetch + error handling)
- [ ] EP-FND-07.T2 Catalogue table
  - [ ] Columns: main skill, ascendancy, scenario metric columns
  - [ ] Sorting and pagination
  - [ ] Filter panel (profile/scenario/gate_pass, min thresholds)
- [ ] EP-FND-07.T3 Build detail drawer
  - [ ] Scenario metrics display (tabs per scenario)
  - [ ] Gate fail reasons + PoB warnings
  - [ ] Export buttons (download code/XML)
- [ ] EP-FND-07.T4 Import UI (optional but useful)
  - [ ] Paste PoB code / upload XML
  - [ ] Trigger evaluate and show status
- [ ] EP-FND-07.T5 UX polish for PoC
  - [ ] Copy-to-clipboard for PoB code
  - [ ] Links to open local artifact path (optional)

### EP-FND-08 E2E smoke suite

- [ ] EP-FND-08.T1 Fixture builds
  - [ ] Collect 3–5 representative PoB builds as fixtures (xml or code)
  - [ ] Store under `backend/tests/fixtures/`
- [ ] EP-FND-08.T2 Integration test harness
  - [ ] Test: import fixture → evaluate → assert metrics exist and within loose ranges
  - [ ] Test: scenario difference (mapping vs uber changes DPS meaningfully)
- [ ] EP-FND-08.T3 Makefile wiring
  - [ ] `make test` runs backend integration tests (and skips if PoB not available with a clear message)
  - [ ] UI lint/typecheck included in `make test`
- [ ] EP-FND-08.T4 Minimal “drift report”
  - [ ] When a fixture assertion fails, print a compact diff of key metrics

---

### EP-V1-01 Bulk import pipeline

- [ ] EP-V1-01.T1 Bulk import CLI
  - [ ] Accept folder input containing `.xml` and/or `.txt` PoB codes
  - [ ] De-duplicate by hash
- [ ] EP-V1-01.T2 Batch evaluation scheduling
  - [ ] Queue builds for evaluation with configurable worker concurrency
  - [ ] Track import/eval status per build
- [ ] EP-V1-01.T3 Import summary output
  - [ ] Write `data/runs/<run_id>/summary.json` with counts + failures
  - [ ] Expose run summary in API (optional)

### EP-V1-02 Scenario transparency + diff tooling

- [ ] EP-V1-02.T1 Persist scenario inputs
  - [ ] Ensure `scenarios_used.json` is written for every evaluation
- [ ] EP-V1-02.T2 UI: show scenario assumptions
  - [ ] Display key scenario toggles in detail view
- [ ] EP-V1-02.T3 Scenario diff view
  - [ ] Side-by-side metric comparison across scenarios for same build
  - [ ] Highlight largest deltas (top N)

### EP-V1-03 Regression baselines

- [ ] EP-V1-03.T1 Define “expected metric ranges”
  - [ ] For each golden build and scenario, define min/max ranges for a few key metrics
- [ ] EP-V1-03.T2 Automate baseline update workflow (manual trigger)
  - [ ] `make regen-baselines` writes updated ranges for review (no auto-commit)
- [ ] EP-V1-03.T3 Lock baseline to ruleset
  - [ ] Store baselines with `ruleset_id`
  - [ ] If `ruleset_id` changes, baselines require explicit regeneration

---

### EP-V2-01 Genome v0 + serialization

- [ ] EP-V2-01.T1 Genome schema definition
  - [ ] Fields: seed, class/ascendancy, main skill package, defence archetype, budget tier, profile_id
- [ ] EP-V2-01.T2 Deterministic RNG
  - [ ] One RNG seeded by `seed` per build
  - [ ] Ensure all random choices go through this RNG
- [ ] EP-V2-01.T3 Genome serialization contract
  - [ ] `genome.json` is stable and versioned (`genome_version`)
  - [ ] Backwards-compatible parsing strategy (best effort)

### EP-V2-02 Skill package catalog + constraints

- [ ] EP-V2-02.T1 Skill package format
  - [ ] Define `skill_package.json`: main gem, support set options, required tags/constraints
- [ ] EP-V2-02.T2 Curate initial package set (small)
  - [ ] Select 10–20 main skill packages for PoC
  - [ ] Include multiple archetypes (hit/DoT/minion/trap/mine/totem) only if you can support them
- [ ] EP-V2-02.T3 Utility packs
  - [ ] Movement skills pack(s)
  - [ ] Auras pack(s)
  - [ ] Guard/curse packs (optional)
- [ ] EP-V2-02.T4 Deterministic “Full DPS inclusion” rules
  - [ ] Mark which socket group counts toward Full DPS
  - [ ] Ensure buffs/utility groups are excluded

### EP-V2-03 Passive tree skeleton builder v0

- [ ] EP-V2-03.T1 Passive tree targets per archetype
  - [ ] Define lists of “allowed/desired” keystones/notables/masteries per defence archetype
- [ ] EP-V2-03.T2 Simple pathing algorithm
  - [ ] Route from start to targets using shortest path (graph-based)
  - [ ] Spend remaining points on nearest efficient clusters (heuristic)
- [ ] EP-V2-03.T3 Tree legality checks
  - [ ] Total points <= budget
  - [ ] Mastery selection rules (one per cluster, etc.)
- [ ] EP-V2-03.T4 Export to PoB tree representation
  - [ ] Convert allocated nodes to PoB build format deterministically

### EP-V2-04 Socket/link planner v0

- [ ] EP-V2-04.T1 Link requirements
  - [ ] Define main link size target (4/5/6) per package
- [ ] EP-V2-04.T2 Color planning heuristic
  - [ ] Minimal algorithm to assign colors based on gem requirements
  - [ ] Support off-color constraints via item base selection (later)
- [ ] EP-V2-04.T3 Slot assignment
  - [ ] Assign main link to a specific slot (e.g., body armour or weapon) deterministically
  - [ ] Assign utility skills across remaining sockets
- [ ] EP-V2-04.T4 Socket feasibility pre-check
  - [ ] Reject/repair builds that cannot fit required sockets/links

### EP-V2-05 Item template filler v0

- [ ] EP-V2-05.T1 Item template model
  - [ ] For each slot define: base type hint, priority stats, optional unique
- [ ] EP-V2-05.T2 Baseline requirement satisfier
  - [ ] Resist capping (coarse)
  - [ ] Attribute requirements (coarse)
  - [ ] Life/ES baseline
- [ ] EP-V2-05.T3 Defence-archetype stats
  - [ ] Armour archetype: armour + max res/phys mitigation hints
  - [ ] Evasion/suppress archetype: evasion + suppression hints
  - [ ] ES archetype: ES + recharge/regen hints
- [ ] EP-V2-05.T4 Export items into PoB build
  - [ ] Represent templates in a PoB-importable way (simple rare text lines are OK for PoC)
- [ ] EP-V2-05.T5 “Repair pass” loop (cheap)
  - [ ] After initial fill, adjust a few slots to fix missing res/attributes if possible

### EP-V2-06 Generation jobs + UI trigger

- [ ] EP-V2-06.T1 Generation job runner
  - [ ] CLI: generate N with parameters (profile, packages, budget tier)
  - [ ] Write run summary and store generated builds
- [ ] EP-V2-06.T2 UI trigger
  - [ ] Minimal form: N, profile, package subset, budget tier
  - [ ] Show progress and final counts
- [ ] EP-V2-06.T3 Evaluation scheduling integration
  - [ ] Generated builds enqueue to PoB evaluation pool automatically
- [ ] EP-V2-06.T4 Failure visibility
  - [ ] Store and surface generation failures vs evaluation failures separately

---

### EP-V3-01 Price snapshot ingestor (poe.ninja)

- [ ] EP-V3-01.T1 Snapshot fetcher
  - [ ] Download required endpoints for the selected league
  - [ ] Cache to `data/prices/<league>/<timestamp>/...`
- [ ] EP-V3-01.T2 Snapshot index
  - [ ] Write `price_snapshot.json` metadata (league, timestamp, sources)
  - [ ] Generate `price_snapshot_id` used in `ruleset_id`
- [ ] EP-V3-01.T3 Minimal update mechanism
  - [ ] CLI: `update-prices --league X`
  - [ ] Keep N most recent snapshots; delete older on request only

### EP-V3-02 Cost calculator v1 (uniques + gems)

- [ ] EP-V3-02.T1 Unique price mapping
  - [ ] Map PoB item names → poe.ninja unique names (normalize)
  - [ ] Handle missing items gracefully (unknown cost bucket)
- [ ] EP-V3-02.T2 Skill gem pricing
  - [ ] Map gem name (+ quality/level if represented) → poe.ninja SkillGem prices
  - [ ] Handle support gems and awakened variants where applicable
- [ ] EP-V3-02.T3 Build cost breakdown output
  - [ ] Produce `slot_costs.json` + `gem_costs.json`
  - [ ] Compute `total_cost_chaos`
- [ ] EP-V3-02.T4 Persist to ClickHouse
  - [ ] Write `build_costs` row with `price_snapshot_id` and paths

### EP-V3-03 UI: cost views + value metrics

- [ ] EP-V3-03.T1 UI columns and sorting
  - [ ] Total cost column
  - [ ] DPS/chaos and MaxHit/chaos columns
- [ ] EP-V3-03.T2 Filters
  - [ ] Budget cap (max chaos)
  - [ ] “exclude unknown cost” toggle
- [ ] EP-V3-03.T3 Breakdown drill-down
  - [ ] Show per-slot and per-gem costs in detail view

---

### EP-V4-01 Feature extraction + dataset pipeline

**Implemented Feature Schema v4 (2026-02-28):**

The v4 feature schema captures ALL build dimensions for correlation learning:

**Numeric Features (scalar values):**
- `feature_item_slot_count` - Total equipment slots filled
- `feature_item_adjustable_count` - Slots with modifiable affixes
- `feature_item_base_type_count` - Unique item base types
- `feature_item_contrib_resists` - Total resistance from items
- `feature_item_contrib_attributes` - Total str/dex/int from items
- `feature_item_contrib_life` - Total life from items
- `feature_item_contrib_energy_shield` - Total ES from items
- `feature_affix_resist_lines` - Resistance affix lines count
- `feature_affix_attribute_lines` - Attribute affix lines count
- `feature_affix_life_lines` - Life affix lines count
- `feature_affix_energy_shield_lines` - ES affix lines count
- `feature_affix_total_lines` - Total affix lines
- `feature_passive_node_count` - Passive tree nodes allocated
- `feature_passive_required_targets` - Required attribute nodes
- `feature_gem_group_count` - Total skill gem groups
- `feature_gem_damage_group_count` - Damage skill groups
- `feature_gem_utility_group_count` - Utility skill groups
- `feature_gem_total_count` - Total gems equipped
- `feature_gem_main_group_count` - Main skill groups
- `feature_gem_max_link_count` - Maximum socket links
- `feature_gem_main_link_requirement` - Main link attribute requirements

**Token Features (sparse identity vectors):**
- `feature_identity_tokens` - Up to 256 tokens encoding: slot/base type bundles, passive mastery/keystone, gem topology
- `feature_identity_cross_tokens` - Up to 384 cross-dimensional tokens capturing interactions

**Cross-token Categories:**
- `_BASE_TYPE_CROSS_LIMIT=16` - Base type combinations
- `_PASSIVE_CROSS_LIMIT=48` - Passive tree interactions  
- `_MAIN_SUPPORT_CROSS_LIMIT=8` - Main skill + support combos
- `_TRIPLE_CROSS_LIMIT=32` - Triple token interactions

- [ ] EP-V4-01.T1 Feature schema v0
  - [x] Implemented v4 feature schema (see above)
  - [x] Define token features (skill/support/keystone/etc.) for later models
- [ ] EP-V4-01.T2 Dataset writer
  - [ ] Write Parquet rows: features + labels (PoB metrics) + metadata (ruleset_id)
- [ ] EP-V4-01.T3 Dataset versioning
  - [ ] Dataset snapshot id in filename and in model metadata
- [ ] EP-V4-01.T4 Sanity checks
  - [ ] Detect NaNs, constant columns, missing labels
  - [ ] Basic train/val split strategy by time/run id

### EP-V4-02 Train baseline surrogate + viability classifier

- [ ] EP-V4-02.T1 Training script
  - [ ] Train LightGBM regressors for key metrics
  - [ ] Train a classifier for gate pass probability
- [ ] EP-V4-02.T2 Evaluation report
  - [ ] Error metrics (MAE/MAPE) per scenario
  - [ ] Calibration for pass probability (rough)
- [ ] EP-V4-02.T3 Model artifact saving
  - [ ] Save model files under `data/models/<model_id>/`
  - [ ] Save `model_meta.json` (dataset id, ruleset_id, feature schema version)
- [ ] EP-V4-02.T4 Minimal inference API
  - [ ] `predict_many(features)` returns predicted metrics + pass probability

### EP-V4-03 Surrogate-assisted selection loop

- [ ] EP-V4-03.T1 Candidate scoring pipeline
  - [ ] Generate many genomes → extract features → batch predict
- [ ] EP-V4-03.T2 Selection policy v0
  - [ ] Top X% by predicted score
  - [ ] Plus Y% random exploration (fixed)
- [ ] EP-V4-03.T3 PoB budget controls
  - [ ] Limit PoB evaluations per run (hard cap)
  - [ ] Ensure evaluation queue doesn’t explode
- [ ] EP-V4-03.T4 Feedback loop
  - [ ] Add PoB-verified candidates back into dataset snapshots

### EP-V4-04 UI: predicted vs verified transparency

- [ ] EP-V4-04.T1 Store predictions
  - [ ] Persist predicted metrics + model id for builds (even if not verified yet)
- [ ] EP-V4-04.T2 UI indicators
  - [ ] “Verified” badge vs “Predicted”
  - [ ] Show prediction error where verified exists
- [ ] EP-V4-04.T3 Filtering modes
  - [ ] Filter: verified-only vs include predicted

---

### EP-V5-01 Descriptor design + MAP-Elites archive

- [ ] EP-V5-01.T1 Choose descriptor axes
  - [ ] Budget bin (log cost)
  - [ ] Defence archetype
  - [ ] Damage delivery type
  - [ ] Complexity bucket
- [ ] EP-V5-01.T2 Binning functions
  - [ ] Deterministic mapping build → bin key
  - [ ] Handle unknown cost by placing into “unknown” bin
- [ ] EP-V5-01.T3 Archive store
  - [ ] In-memory archive during run
  - [ ] Persist elites to ClickHouse (elite pointer table) or as a view/query
- [ ] EP-V5-01.T4 Elite replacement logic
  - [ ] Define score function per profile/scenario
  - [ ] Replace elite if score improves and gates pass

### EP-V5-02 Emitters + evaluation budget allocator

- [ ] EP-V5-02.T1 Emitter implementations
  - [ ] Exploit emitter (mutate elites)
  - [ ] Novelty emitter (sample underfilled bins)
  - [ ] Uncertainty emitter (later, once uncertainty exists)
- [ ] EP-V5-02.T2 Budget allocator
  - [ ] Fixed split (e.g., 70/20/10) for PoB evaluation slots per run
  - [ ] Ensure novelty budget is non-zero (prevents meta lock-in)
- [ ] EP-V5-02.T3 Coverage reporting
  - [ ] Count filled bins, best score per bin summary
  - [ ] Persist run report to `data/runs/<run_id>/`

### EP-V5-03 UI: archive browser

- [ ] EP-V5-03.T1 Archive browsing UI
  - [ ] Select descriptor axes view
  - [ ] Show best-per-bin as a grid/list
- [ ] EP-V5-03.T2 Bin drill-down
  - [ ] Show top K candidates for a bin with sorting
- [ ] EP-V5-03.T3 Export flow
  - [ ] One-click export of elite build artifacts

---

### EP-V6-01 Skill scaling signatures (probe runs)

- [ ] EP-V6-01.T1 Define probe set
  - [ ] List of generic stat perturbations (e.g., +% inc damage by type, +crit, +attack/cast speed, +DoT multi)
- [ ] EP-V6-01.T2 Implement probing in PoB worker
  - [ ] Apply temporary modifiers to a build and measure Δmetrics
- [ ] EP-V6-01.T3 Cache signatures
  - [ ] Cache per skill package (not per build) where possible
  - [ ] Store signature vectors in `data/datasets/signatures/`
- [ ] EP-V6-01.T4 Integrate into features
  - [ ] Add signature to surrogate feature pipeline (as numeric vector)

### EP-V6-02 Counterfactual ablation dataset

- [ ] EP-V6-02.T1 Define ablation operators
  - [ ] Remove one support gem
  - [ ] Swap support gem from a small candidate list
  - [ ] Remove a passive cluster (bounded scope)
  - [ ] Downgrade an item template tier (coarse)
- [ ] EP-V6-02.T2 Generate counterfactuals
  - [ ] For selected verified builds, produce N counterfactual variants deterministically
- [ ] EP-V6-02.T3 Evaluate and store Δmetrics
  - [ ] Run PoB evaluation on variants
  - [ ] Store (base metrics, variant metrics, delta) rows in dataset
- [ ] EP-V6-02.T4 Sampling controls
  - [ ] Keep counterfactual dataset bounded (cap per run/build)

### EP-V6-03 Hybrid interaction surrogate + uncertainty

- [ ] EP-V6-03.T1 Model architecture v0
  - [ ] Token embeddings (skills/supports/keystones/uniques)
  - [ ] Numeric features + scaling signature
  - [ ] Multi-head outputs (metrics + pass probability)
- [ ] EP-V6-03.T2 Uncertainty estimation
  - [ ] Small ensemble or quantile heads
  - [ ] Output uncertainty score per build
- [ ] EP-V6-03.T3 Training pipeline
  - [ ] Train on main dataset + counterfactual deltas (as extra supervision)
  - [ ] Save model artifacts and metadata
- [ ] EP-V6-03.T4 Inference integration
  - [ ] Batch inference implementation (CPU first; GPU optional later)

### EP-V6-04 Delta-guided mutation + repair operators

- [ ] EP-V6-04.T1 Delta-guided edits
  - [ ] Use ablation-trained delta predictors to propose edits with best Δscore/Δcost
- [ ] EP-V6-04.T2 Repair operators
  - [ ] Resist repair (swap template stats)
  - [ ] Attribute repair
  - [ ] Reservation repair (remove/replace aura pack)
- [ ] EP-V6-04.T3 Integration into generation loop
  - [ ] Apply repairs before rejecting a candidate
  - [ ] Record which repair fixed what (for later analysis)

### EP-V6-05 Novelty governance

- [ ] EP-V6-05.T1 Novelty metric definition
  - [ ] Distance in embedding space and/or token Jaccard distance
- [ ] EP-V6-05.T2 Novelty quota enforcement
  - [ ] Guarantee PoB evaluation budget share for novelty/uncertainty
- [ ] EP-V6-05.T3 Coverage dashboards (no monitoring stack)
  - [ ] Write run summaries: novelty accepted rate, new bin fill rate

---

### EP-V7-01 More profiles (delve tiers, support, stricter ubers)

- [ ] EP-V7-01.T1 Delve scenarios
  - [ ] Define depth tiers and corresponding scenario configs
  - [ ] Define survivability-first score for delve profiles
- [ ] EP-V7-01.T2 Support scenarios
  - [ ] Define party-support evaluation outputs (aura effect, reservation efficiency, utility)
- [ ] EP-V7-01.T3 UI profile selector expansion
  - [ ] Add profile descriptions + gates summary per profile

### EP-V7-02 Ruleset/version management + re-evaluate tooling

- [ ] EP-V7-02.T1 Ruleset id generator
  - [ ] Centralize `ruleset_id` creation from PoB commit + scenario version + price snapshot id
- [ ] EP-V7-02.T2 Mark builds stale
  - [ ] Detect stale builds when ruleset changes
  - [ ] Filter stale in UI by default
- [ ] EP-V7-02.T3 Batch re-evaluation tool
  - [ ] Re-evaluate selected builds for the latest ruleset
  - [ ] Store both old and new results (query latest via `argMax` pattern or latest timestamp)

### EP-V7-03 Golden build suite + drift report

- [ ] EP-V7-03.T1 Expand golden suite
  - [ ] Add at least one build per archetype you care about
- [ ] EP-V7-03.T2 Drift report generator
  - [ ] Compare metrics between ruleset versions
  - [ ] Output `data/runs/<run_id>/drift_report.md`
- [ ] EP-V7-03.T3 “Upgrade checklist”
  - [ ] Document: bump PoB, regen baselines, re-evaluate elites, retrain models

---

## 9) Minimal run commands (suggested)

- `make dev`
  - `docker compose up -d clickhouse`
  - run backend (uvicorn)
  - run UI (vite)

- `make test`
  - backend: pytest (includes PoB evaluation integration tests using fixtures)
  - UI: eslint + tsc

---

## 10) “Done” definition for PoC v1

PoC v1 is done when you can:
1) import and generate builds locally,
2) evaluate them headlessly in PoB under explicit scenarios,
3) compute basic cost for gems/uniques,
4) browse/filter/sort in a local UI,
5) export any build back into PoB.

---

## 11) Source baseline

Baseline documents used to derive this plan:
- “Automated Build Generation for Path of Exile using Path of Building — Working Concept and Implementation Plan (v1.1)”
- “Fully Automatic Path of Building Build Generator at Billion-Scale” (deep research report)
