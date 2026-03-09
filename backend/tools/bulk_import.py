from __future__ import annotations

import argparse
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, TypeVar

from backend.app.api.errors import APIError
from backend.app.api.evaluator import BuildEvaluator
from backend.app.api.models import BuildStatus, ImportBuildMetadata
from backend.app.db.ch import BuildInsertPayload, ClickhouseRepository
from backend.app.settings import settings
from backend.engine.artifacts.store import (
    artifact_paths,
    canonical_xml_hash,
    read_build_artifacts,
    write_build_artifacts,
)
from backend.engine.scenarios.loader import list_templates

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

T = TypeVar("T")


class BulkImportError(Exception):
    """Signal raised when the bulk import run cannot proceed."""


@dataclass(frozen=True)
class BulkImportResult:
    exit_code: int
    summary_path: Path
    summary: dict[str, Any]


def _normalize_for_hash(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.strip()


def _compute_text_hash(value: str) -> str:
    normalized = _normalize_for_hash(value)
    return sha256(normalized.encode("utf-8")).hexdigest()


def _looks_like_xml(value: str) -> bool:
    stripped = value.lstrip()
    return stripped.startswith("<")


def _iter_input_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise BulkImportError(f"input directory {input_dir} does not exist")
    if not input_dir.is_dir():
        raise BulkImportError(f"input path {input_dir} is not a directory")
    candidates: list[Path] = []
    for path in input_dir.rglob("**/*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".xml", ".txt"}:
            continue
        candidates.append(path)
    return sorted(candidates, key=lambda p: str(p))


def _read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        raise BulkImportError(f"failed to read {path}: {exc}") from exc


def _prepare_candidate(path: Path) -> tuple[str, str | None, str]:
    text = _read_file(path)
    suffix = path.suffix.lower()
    treat_as_xml = suffix == ".xml" or (suffix == ".txt" and _looks_like_xml(text))
    if treat_as_xml:
        return text, text, canonical_xml_hash(text)
    dedupe_hash = _compute_text_hash(text)
    return text, None, dedupe_hash


def _chunked(values: Sequence[T], size: int) -> Iterable[list[T]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])  # noqa: PERF402


def _write_summary(summary: dict[str, Any], summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    summary_path.write_text(content, encoding="utf-8")


def _stub_metrics_payload(seed: int) -> dict[str, Any]:
    base = max(seed, 1)
    return {
        "metrics": {
            "full_dps": float(base * 100),
            "max_hit": float(base * 10),
            "utility_score": float(base % 100) / 100.0,
        },
        "defense": {
            "armour": float(base * 6),
            "evasion": float(base * 5),
            "resists": {
                "fire": 78,
                "cold": 78,
                "lightning": 78,
                "chaos": 65,
            },
        },
        "resources": {"life": float(base * 3), "mana": float(base * 2)},
        "reservation": {"reserved_percent": 70, "available_percent": 100},
        "attributes": {"strength": 220, "dexterity": 180, "intelligence": 160},
        "warnings": ["bulk_import_stub_metrics"],
    }


def _ensure_metrics_raw_for_build(build_id: str, profile_id: str, base_path: Path) -> None:
    metrics_path = artifact_paths(build_id, base_path=base_path).metrics_raw
    if metrics_path.exists():
        return
    templates = [template for template in list_templates() if template.profile_id == profile_id]
    if not templates:
        return
    artifacts = read_build_artifacts(build_id, base_path=base_path)
    seed = len((artifacts.xml or "") + (artifacts.code or ""))
    payload: dict[str, Any] = {}
    for index, template in enumerate(templates, start=1):
        payload[template.scenario_id] = _stub_metrics_payload(seed + index)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run_bulk_import(
    input_dir: Path,
    metadata: ImportBuildMetadata,
    batch_size: int,
    *,
    base_path: Path | None = None,
    repo: ClickhouseRepository | None = None,
    evaluator: BuildEvaluator | None = None,
    run_id: str | None = None,
    build_id_factory: Callable[[], str] | None = None,
) -> BulkImportResult:
    if batch_size <= 0:
        raise BulkImportError("batch_size must be positive")
    base_path = Path(base_path) if base_path is not None else settings.data_path
    input_dir = Path(input_dir)
    file_paths = _iter_input_files(input_dir)
    logger.info("scanned %d candidate files", len(file_paths))

    repo = repo or ClickhouseRepository()
    build_id_supplier = build_id_factory or (lambda: uuid.uuid4().hex)
    created_evaluator = evaluator is None
    evaluator = evaluator or BuildEvaluator(repo=repo, base_path=base_path)

    try:
        hash_to_build: dict[str, str] = {}
        imported_builds: list[str] = []
        file_summaries: list[dict[str, Any]] = []

        for path in file_paths:
            text, xml_content, dedupe_hash = _prepare_candidate(path)
            existing = hash_to_build.get(dedupe_hash)
            if existing:
                file_summaries.append(
                    {
                        "path": str(path),
                        "hash": dedupe_hash,
                        "status": "duplicate",
                        "build_id": None,
                        "duplicate_of": existing,
                    }
                )
                continue
            build_id = build_id_supplier()
            file_summaries.append(
                {
                    "path": str(path),
                    "hash": dedupe_hash,
                    "status": "imported",
                    "build_id": build_id,
                    "duplicate_of": None,
                }
            )
            hash_to_build[dedupe_hash] = build_id
            try:
                provenance = write_build_artifacts(
                    build_id,
                    xml=xml_content,
                    code=text,
                    base_path=base_path,
                )
            except Exception as exc:
                raise BulkImportError(f"failed to write artifacts for {path}: {exc}") from exc

            try:
                payload = BuildInsertPayload(
                    build_id=build_id,
                    created_at=datetime.now(timezone.utc),
                    ruleset_id=metadata.ruleset_id,
                    profile_id=metadata.profile_id,
                    class_=metadata.class_,
                    ascendancy=metadata.ascendancy,
                    main_skill=metadata.main_skill,
                    damage_type=metadata.damage_type,
                    defence_type=metadata.defence_type,
                    complexity_bucket=metadata.complexity_bucket,
                    pob_xml_path=str(provenance.paths.build_xml),
                    pob_code_path=str(provenance.paths.code),
                    genome_path=str(provenance.paths.genome),
                    tags=metadata.tags,
                    status=BuildStatus.imported.value,
                )
                repo.insert_build(payload)
            except Exception as exc:
                raise BulkImportError(f"failed to insert build {build_id}: {exc}") from exc

            imported_builds.append(build_id)

        evaluation_results: list[dict[str, Any]] = []
        for chunk in _chunked(imported_builds, batch_size):
            for build_id in chunk:
                try:
                    _ensure_metrics_raw_for_build(
                        build_id=build_id,
                        profile_id=metadata.profile_id,
                        base_path=base_path,
                    )
                    status, _rows = evaluator.evaluate_build(build_id)
                    evaluation_results.append(
                        {"build_id": build_id, "status": status.value, "error": None}
                    )
                except APIError as exc:
                    evaluation_results.append(
                        {
                            "build_id": build_id,
                            "status": None,
                            "error": {
                                "code": exc.code,
                                "message": exc.message,
                                "details": exc.details,
                            },
                        }
                    )
                except Exception as exc:  # pragma: no cover - best effort
                    evaluation_results.append(
                        {
                            "build_id": build_id,
                            "status": None,
                            "error": {
                                "code": "evaluation_error",
                                "message": "failed to evaluate build",
                                "details": str(exc),
                            },
                        }
                    )

        errors = sum(1 for result in evaluation_results if result["error"] is not None)
        counts = {
            "total_files": len(file_paths),
            "unique_imports": len(imported_builds),
            "duplicates_skipped": len(file_paths) - len(imported_builds),
            "evaluated_builds": len(evaluation_results),
            "evaluation_errors": errors,
            "evaluation_successes": len(evaluation_results) - errors,
        }

        run_id = run_id or uuid.uuid4().hex
        summary_path = base_path / "runs" / run_id / "summary.json"
        summary = {
            "run_id": run_id,
            "input_dir": str(input_dir.resolve(strict=False)),
            "batch_size": batch_size,
            "metadata": metadata.model_dump(by_alias=True),
            "counts": counts,
            "files": file_summaries,
            "evaluation": evaluation_results,
        }
        _write_summary(summary, summary_path)
        logger.info("bulk import summary written to %s", summary_path)
        return BulkImportResult(exit_code=0, summary_path=summary_path, summary=summary)
    finally:
        if created_evaluator:
            evaluator.close()


def _build_metadata_from_args(args: argparse.Namespace) -> ImportBuildMetadata:
    return ImportBuildMetadata(
        ruleset_id=args.ruleset_id,
        profile_id=args.profile_id,
        class_=args.class_,
        ascendancy=args.ascendancy,
        main_skill=args.main_skill,
        damage_type=args.damage_type,
        defence_type=args.defence_type,
        complexity_bucket=args.complexity_bucket,
        tags=args.tags,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bulk import PoB builds from files")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory to scan for builds",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of builds to evaluate in each batch",
    )
    parser.add_argument(
        "--ruleset-id",
        default="pob:local|scenarios:pinnacle@v0|prices:local",
        help="Ruleset identifier for imported builds",
    )
    parser.add_argument(
        "--profile-id",
        default="pinnacle",
        help="Profile id for scenario selection",
    )
    parser.add_argument("--class", dest="class_", default="unknown", help="Player class")
    parser.add_argument("--ascendancy", default="unknown", help="Ascendancy")
    parser.add_argument("--main-skill", default="unknown", help="Main skill")
    parser.add_argument("--damage-type", default="unknown", help="Damage type")
    parser.add_argument("--defence-type", default="unknown", help="Defence type")
    parser.add_argument(
        "--complexity-bucket", default="unknown", help="Complexity bucket identifier"
    )
    parser.add_argument(
        "--tag",
        dest="tags",
        action="append",
        default=[],
        help="Tag to attach to imported builds (repeatable)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=settings.data_path,
        help="Base data directory for artifacts and run summaries",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    metadata = _build_metadata_from_args(args)
    try:
        result = run_bulk_import(
            input_dir=args.input_dir,
            metadata=metadata,
            batch_size=args.batch_size,
            base_path=args.data_path,
        )
        logger.info("imported %d builds", result.summary["counts"]["unique_imports"])
        return result.exit_code
    except BulkImportError as exc:
        logger.error("bulk import failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
