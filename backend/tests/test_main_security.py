from __future__ import annotations

import asyncio
import json
from pathlib import Path

from starlette.requests import Request

from backend.app.main import _artifact_path_exists, handle_unexpected_exception


def _request(path: str = "/boom") -> Request:
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "root_path": "",
        "query_string": b"",
        "headers": [],
        "client": ("test", 12345),
        "server": ("test", 80),
    }
    return Request(scope)


def test_artifact_path_exists_rejects_path_traversal(tmp_path: Path) -> None:
    base_path = tmp_path / "data"
    base_path.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")

    assert _artifact_path_exists(str(outside), base_path) is False
    assert _artifact_path_exists("../outside.txt", base_path) is False


def test_artifact_path_exists_allows_safe_paths(tmp_path: Path) -> None:
    base_path = tmp_path / "data"
    build_dir = base_path / "builds"
    build_dir.mkdir(parents=True, exist_ok=True)
    file_path = build_dir / "artifact.txt"
    file_path.write_text("ok", encoding="utf-8")

    assert _artifact_path_exists(str(file_path.relative_to(base_path)), base_path) is True


def test_global_exception_handler_returns_generic_payload() -> None:
    response = asyncio.run(handle_unexpected_exception(_request(), RuntimeError("boom")))

    assert response.status_code == 500
    payload = json.loads(response.body.decode("utf-8"))
    assert payload == {
        "error": {
            "code": "internal_error",
            "message": "internal server error",
        }
    }
