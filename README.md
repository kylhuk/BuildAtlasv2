# BuildAtlasv2 (EP-FND-01)

A single-machine PoB build generator that pairs a Python/FastAPI orchestrator with a LuaJIT PoB worker pool, ClickHouse for metrics/index tables, and a React+TypeScript UI.

## Stack
- **Backend**: Python + FastAPI + ClickHouse (via `clickhouse-connect`).
- **Worker**: LuaJIT Path of Building under `PathOfBuilding/`.
- **Data**: ClickHouse tables plus filesystem artifacts in `data/`.
- **UI**: Vite + React + TypeScript + Ant Design.

## Quickstart
1. `make db-up` – start the ClickHouse container.
2. `make db-init` – confirm ClickHouse is ready and apply the SQL schema.
3. `make dev` – launch backend and UI dev servers.
4. `make test` – run backend tests and UI lint/type checks.
5. `make lint` – run backend and UI linters.
6. `make fmt` – run backend formatter and UI Prettier.
7. `make db-down` – stop the ClickHouse container.

## Guides
- [ML operator guide](docs/ml-operator-guide.md) – manual workflow plus automated `ml_loop` start/stop/status operations.

## Torch / CUDA (optional)
- `make torch-install-cpu` – install the default CPU-safe `torch` stack inside the backend virtual environment.
- `make torch-install-cuda` – pull CUDA-enabled wheels from the `TORCH_CUDA_INDEX_URL` index inside the backend env.
- `make torch-verify` – print `torch` version, CUDA availability, and device details after installation.

Defaults stay CPU-safe (nothing happens until you run an install), and CUDA wheels are opt-in via `make torch-install-cuda` or custom `TORCH_*` overrides.

## Backend Environment
- Copy `backend/.env.example` to `backend/.env` (or export the same variables manually) before running the backend; `backend/app/settings.py` reads the Pydantic `.env` file for these defaults.
- Keep local overrides out of version control and only change the values that differ for your machine.

| Variable | Default | Purpose |
| --- | --- | --- |
| `CLICKHOUSE_HOST` | `127.0.0.1` | ClickHouse host/address for metrics and indexes |
| `CLICKHOUSE_PORT` | `8123` | ClickHouse HTTP port |
| `CLICKHOUSE_USER` | `default` | ClickHouse username |
| `CLICKHOUSE_PASSWORD` | *(empty)* | ClickHouse password (blank for local dev) |
| `CLICKHOUSE_DB` | `default` | ClickHouse database that holds metrics and indexes |
| `DATA_PATH` | `data` | Local directory for generated builds, prices, and datasets |
| `POB_PATH` | `PathOfBuilding` | Relative path to the Path of Building submodule |
| `BACKEND_PORT` | `8000` | Port where FastAPI listens |

## UI Environment
- The Vite UI loads `.env*` files from `ui/`; copy `ui/.env.example` to `ui/.env.development.local` (or `ui/.env`) before running `npm run dev`.
- Only `VITE_`-prefixed keys are exposed to the browser; update `VITE_BACKEND_URL` to point at `http://localhost:8000` (or your backend host).
- See `ui/README.md` for the full convention and more tips.

## Path of Building Submodule
- Bootstrap the `PathOfBuilding/` submodule with `git submodule update --init --recursive` before running the tooling.
- The default `.gitmodules` entry uses the SSH URL (`git@github.com:PathOfBuildingCommunity/PathOfBuilding.git`). If your environment lacks SSH key access, switch to HTTPS by running `git submodule set-url PathOfBuilding https://github.com/PathOfBuildingCommunity/PathOfBuilding.git && git submodule sync PathOfBuilding && git submodule update --init --recursive` (or configure `url.https://github.com/.insteadOf=git@github.com:` globally).

## Notes
- Use `data/` for generated builds, prices, datasets, and runs; placeholder `.keep` files keep empty dirs in git.
