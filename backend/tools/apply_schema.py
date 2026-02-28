import logging
from pathlib import Path
from typing import List

import clickhouse_connect

from backend.app.settings import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def split_sql_statements(sql_content: str) -> List[str]:
    statements: List[str] = []
    for raw in sql_content.split(";"):
        statement = raw.strip()
        if statement:
            statements.append(statement)
    return statements


def apply_schema() -> None:
    root = Path(__file__).resolve().parents[2]
    sql_dir = root / "sql" / "clickhouse"
    if not sql_dir.exists():
        logger.info("ClickHouse schema directory %s does not exist", sql_dir)
        return

    client = clickhouse_connect.get_client(
        host=settings.clickhouse_host,
        port=settings.clickhouse_port,
        username=settings.clickhouse_user,
        password=settings.clickhouse_password,
        database=settings.clickhouse_db,
    )
    sql_files = sorted(sql_dir.glob("*.sql"))
    if not sql_files:
        logger.info("No SQL files found in %s", sql_dir)
        return

    for sql_file in sql_files:
        logger.info("Applying %s", sql_file.name)
        content = sql_file.read_text(encoding="utf-8")
        statements = split_sql_statements(content)
        if not statements:
            continue
        for statement in statements:
            client.command(statement)


if __name__ == "__main__":
    apply_schema()
