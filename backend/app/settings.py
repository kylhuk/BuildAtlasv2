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
    clickhouse_query_timeout_seconds: int = 30
    data_path: Path = Path("data")
    pob_path: Path = Path("PathOfBuilding")
    backend_port: int = 8000

    # Use the authoritative Path of Building worker by default.
    # Can be overridden via environment as POB_WORKER_CMD/POB_WORKER_ARGS/POB_WORKER_CWD.
    pob_worker_cmd: str = "luajit"
    pob_worker_args: str = "backend/pob_worker/pob_worker.lua"

    # Optional working directory for spawned worker process (used by luajit worker mode).
    # Pointing into PathOfBuilding/src keeps HeadlessWrapper happy.
    pob_worker_cwd: str = "PathOfBuilding/src"

    # Number of parallel worker processes for metric evaluation.
    # Increase this when Phase 1 is the bottleneck.
    pob_worker_pool_size: int = Field(default_factory=_default_pob_worker_pool_size)

    # Performance feature flags (default to False for backward compatibility)
    use_orjson: bool = Field(default=False, alias="USE_ORJSON")
    use_xxhash: bool = Field(default=False, alias="USE_XXHASH")
    use_async_io: bool = Field(default=False, alias="USE_ASYNC_IO")

    model_config = SettingsConfigDict(env_file=ENV_PATH, env_file_encoding="utf-8")


settings = Settings()
