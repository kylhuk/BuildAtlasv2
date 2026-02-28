from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_FILENAME = "." + "env"
ENV_PATH = BASE_DIR / ENV_FILENAME


class Settings(BaseSettings):
    clickhouse_host: str = "127.0.0.1"
    clickhouse_port: int = 8123
    clickhouse_user: str = "default"
    clickhouse_password: str = ""
    clickhouse_db: str = "default"
    data_path: Path = Path("data")
    pob_path: Path = Path("PathOfBuilding")
    backend_port: int = 8000

    model_config = SettingsConfigDict(env_file=ENV_PATH, env_file_encoding="utf-8")


settings = Settings()
