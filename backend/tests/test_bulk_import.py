from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from backend.app.api.errors import APIError
from backend.app.api.models import BuildStatus, ImportBuildMetadata
from backend.tools.bulk_import import run_bulk_import


class FakeRepository:
    def __init__(self) -> None:
        self.inserted: list[str] = []

    def insert_build(self, payload: object) -> None:
        self.inserted.append(payload)


class FakeEvaluator:
    def __init__(self, responses: dict[str, Iterable[object] | object]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def evaluate_build(self, build_id: str) -> tuple[BuildStatus, list[object]]:
        self.calls.append(build_id)
        response = self.responses.get(build_id)
        if isinstance(response, APIError):
            raise response
        if response is None:
            return BuildStatus.evaluated, []
        return response


def _metadata() -> ImportBuildMetadata:
    return ImportBuildMetadata(
        ruleset_id="pob:local|scenarios:pinnacle@v0|prices:local",
        profile_id="pinnacle",
        class_="Marauder",
        ascendancy="Slayer",
        main_skill="Sunder",
        damage_type="physical",
        defence_type="armor",
        complexity_bucket="low",
        tags=["bulk"],
    )


def test_bulk_import_dedupe_counts_and_summary(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "01-unique.xml").write_text("<build><id>A</id></build>")
    (input_dir / "02-duplicate.xml").write_text("<build><id>A</id></build>\n")
    (input_dir / "03-share.txt").write_text("$PoB$sharecode")
    (input_dir / "04-share-dup.txt").write_text("$PoB$sharecode\n")

    metadata = _metadata()
    fake_repo = FakeRepository()
    evaluator = FakeEvaluator(
        {
            "build-xml": (BuildStatus.evaluated, []),
            "build-txt": (BuildStatus.evaluated, []),
        }
    )
    factory = iter(["build-xml", "build-txt"])
    result = run_bulk_import(
        input_dir=input_dir,
        metadata=metadata,
        batch_size=1,
        base_path=tmp_path,
        repo=fake_repo,
        evaluator=evaluator,
        run_id="test-run",
        build_id_factory=lambda: next(factory),
    )

    assert result.exit_code == 0
    summary = result.summary
    counts = summary["counts"]
    assert counts["total_files"] == 4
    assert counts["unique_imports"] == 2
    assert counts["duplicates_skipped"] == 2
    assert summary["evaluation"]
    assert evaluator.calls == ["build-xml", "build-txt"]

    saved_summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert saved_summary == summary

    file_entries = summary["files"]
    assert len(file_entries) == 4
    assert file_entries[0]["status"] == "imported"
    assert file_entries[1]["status"] == "duplicate"
    assert file_entries[2]["status"] == "imported"
    assert file_entries[3]["status"] == "duplicate"


def test_bulk_import_batches_and_error_accounting(tmp_path: Path) -> None:
    input_dir = tmp_path / "runs"
    input_dir.mkdir()
    (input_dir / "a.xml").write_text("<build><id>B</id></build>")
    (input_dir / "b.txt").write_text("$PoB$other")

    metadata = _metadata()
    fake_repo = FakeRepository()
    error_payload = APIError(400, "bad_request", "failed to evaluate", details="missing metrics")
    evaluator = FakeEvaluator(
        {
            "imported-1": (BuildStatus.evaluated, []),
            "imported-2": error_payload,
        }
    )
    factory = iter(["imported-1", "imported-2"])
    result = run_bulk_import(
        input_dir=input_dir,
        metadata=metadata,
        batch_size=1,
        base_path=tmp_path,
        repo=fake_repo,
        evaluator=evaluator,
        run_id="error-run",
        build_id_factory=lambda: next(factory),
    )

    assert result.exit_code == 0
    evaluation = result.summary["evaluation"]
    assert evaluation[0]["status"] == "evaluated"
    assert evaluation[0]["error"] is None
    assert evaluation[1]["status"] is None
    assert evaluation[1]["error"]["code"] == "bad_request"
    assert result.summary["counts"]["evaluation_errors"] == 1
    assert evaluator.calls == ["imported-1", "imported-2"]
