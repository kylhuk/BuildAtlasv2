from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.tools import apply_schema


def test_split_sql_statements_skips_empty_entries():
    sql = """CREATE TABLE foo;


CREATE TABLE bar;"""
    statements = apply_schema.split_sql_statements(sql)

    assert statements == ["CREATE TABLE foo", "CREATE TABLE bar"]


def test_apply_schema_executes_each_statement_in_order(tmp_path):
    root = tmp_path / "repo"
    sql_dir = root / "sql" / "clickhouse"
    sql_dir.mkdir(parents=True)
    file_one = sql_dir / "002_second.sql"
    file_one.write_text("SELECT 1;\\nSELECT 2;")
    file_two = sql_dir / "001_first.sql"
    file_two.write_text("CREATE TABLE foo;\\nCREATE TABLE bar;")

    client = MagicMock()

    class FakeResolvedPath:
        def __init__(self, value: Path):
            self._value = value

        @property
        def parents(self):
            class Parents:
                def __init__(self, value: Path):
                    self._value = value

                def __getitem__(self, idx):
                    if idx == 2:
                        return self._value
                    raise IndexError

            return Parents(self._value)

    class FakePath:
        def __init__(self, *_args, **_kwargs):
            pass

        def resolve(self):
            return FakeResolvedPath(root)

    with (
        patch("backend.tools.apply_schema.Path", FakePath),
        patch(
            "backend.tools.apply_schema.clickhouse_connect.get_client",
            return_value=client,
        ),
    ):
        apply_schema.apply_schema()

    expected = []
    for candidate in [file_two, file_one]:
        expected.extend(apply_schema.split_sql_statements(candidate.read_text()))

    called = [call.args[0] for call in client.command.call_args_list]
    assert called == expected
