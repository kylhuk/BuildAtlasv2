from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from backend.app.api.errors import APIError
from backend.app.api.models import BuildStatus, ImportBuildMetadata
from backend.tools import bulk_import


class FakeRepository:
    def __init__(self) -> None:
        self.inserted: list[Any] = []

    def insert_build(self, payload: object) -> None:
        self.inserted.append(payload)


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
    evaluator = Mock()

    def _evaluate_build(build_id: str) -> tuple[BuildStatus, list[Any]]:
        responses: dict[str, tuple[BuildStatus, list[Any]]] = {
            "build-xml": (BuildStatus.evaluated, []),
            "build-txt": (BuildStatus.evaluated, []),
        }
        if build_id not in responses:
            return BuildStatus.evaluated, []
        return responses[build_id]

    evaluator.evaluate_build.side_effect = _evaluate_build
    factory = iter(["build-xml", "build-txt"])
    result = bulk_import.run_bulk_import(
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
    assert evaluator.evaluate_build.call_args_list[0].args[0] == "build-xml"
    assert evaluator.evaluate_build.call_count == 2

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
    evaluator = Mock()

    def _evaluate_build(build_id: str) -> tuple[BuildStatus, list[Any]]:
        if build_id == "imported-2":
            raise error_payload
        return BuildStatus.evaluated, []

    evaluator.evaluate_build.side_effect = _evaluate_build
    factory = iter(["imported-1", "imported-2"])
    result = bulk_import.run_bulk_import(
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
    assert evaluator.evaluate_build.call_count == 2
    assert [call.args[0] for call in evaluator.evaluate_build.call_args_list] == [
        "imported-1",
        "imported-2",
    ]


def test_bulk_import_closes_internal_evaluator_on_write_error(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "a.xml").write_text("<build><id>A</id></build>")

    metadata = _metadata()
    fake_repo = FakeRepository()
    evaluator = Mock()

    with (
        patch.object(bulk_import, "BuildEvaluator", return_value=evaluator),
        patch.object(
            bulk_import,
            "write_build_artifacts",
            side_effect=RuntimeError("artifact failure"),
        ),
    ):
        with pytest.raises(bulk_import.BulkImportError, match="failed to write artifacts"):
            bulk_import.run_bulk_import(
                input_dir=input_dir,
                metadata=metadata,
                batch_size=1,
                base_path=tmp_path,
                repo=fake_repo,
                run_id="import-write-error",
                evaluator=None,
            )

    evaluator.close.assert_called_once()


def test_bulk_import_does_not_close_caller_supplied_evaluator_on_api_error(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "a.xml").write_text("<build><id>A</id></build>")
    (input_dir / "b.txt").write_text("$PoB$share")

    metadata = _metadata()
    fake_repo = FakeRepository()
    evaluator = Mock()
    evaluator.evaluate_build.side_effect = [
        (BuildStatus.evaluated, []),
        APIError(400, "bad_request", "failed to evaluate", details="validation"),
    ]

    result = bulk_import.run_bulk_import(
        input_dir=input_dir,
        metadata=metadata,
        batch_size=1,
        base_path=tmp_path,
        repo=fake_repo,
        evaluator=evaluator,
        run_id="import-external-evaluator",
    )

    assert result.exit_code == 0
    assert result.summary["counts"]["evaluation_errors"] == 1
    assert evaluator.close.call_count == 0
