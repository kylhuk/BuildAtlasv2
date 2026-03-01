import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_FILENAME = "." + "env"
ENV_PATH = BASE_DIR / ENV_FILENAME


def _default_pob_worker_pool_size() -> int:
    return max(1, (os.cpu_count() or 1) * 2)


class Settings(BaseSettings):
    clickhouse_host: str = "127.0.0.1"
    clickhouse_port: int = 8123
    clickhouse_user: str = "default"
    clickhouse_password: str = ""
    clickhouse_db: str = "default"
    data_path: Path = Path("data")
    pob_path: Path = Path("PathOfBuilding")
    backend_port: int = 8000

    # Use the authoritative Path of Building worker by default.
    # Can be overridden via environment as POB_WORKER_CMD/POB_WORKER_ARGS.
    pob_worker_cmd: str = "luajit"
    pob_worker_args: str = "PathOfBuilding/worker/worker.lua"

    # Optional working directory for spawned worker process (used by luajit worker mode).
    # Keep as PathOfBuilding/src so package path resolution works for HeadlessWrapper.
    pob_worker_cwd: str = "PathOfBuilding/src"

    # Number of parallel worker processes for metric evaluation.
    # Increase this when Phase 1 is the bottleneck.
    pob_worker_pool_size: int = Field(default_factory=_default_pob_worker_pool_size)

    model_config = SettingsConfigDict(env_file=ENV_PATH, env_file_encoding="utf-8")


settings = Settings()
