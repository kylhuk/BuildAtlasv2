from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def deterministic_seed() -> int:
    """Provide a consistent seed value for tests."""
    return 1337


@pytest.fixture
def deterministic_random(deterministic_seed: int) -> Iterator[None]:
    """Temporarily pin Python random state for deterministic tests."""
    state = random.getstate()
    random.seed(deterministic_seed)
    try:
        yield
    finally:
        random.setstate(state)


@pytest.fixture
def mock_clickhouse_client() -> Mock:
    """Shared mock ClickHouse client."""
    client = Mock(name="clickhouse_client")
    client.query.return_value = Mock(result_rows=[])
    client.command.return_value = None
    client.insert.return_value = None
    return client


@pytest.fixture
def backend_tmp_data_path(tmp_path: Path) -> Path:
    """Stable temporary data root for backend artifacts."""
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    return root
