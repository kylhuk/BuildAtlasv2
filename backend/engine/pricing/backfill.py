"""Cost backfill workflow for existing builds."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol, Sequence

from backend.app.db.ch import BuildCostRow
from backend.engine.artifacts import store as artifact_store
from backend.engine.artifacts.store import parse_ruleset_id
from backend.engine.pricing.costs import (
    calculate_cost_summary,
    extract_build_cost_requirements,
    load_price_snapshot,
    write_cost_outputs,
)


class BackfillRepository(Protocol):
    def list_builds(self, **kwargs) -> Sequence[dict]: ...

    def get_latest_build_cost(self, build_id: str) -> dict | None: ...

    def insert_build_costs(self, rows: Sequence[BuildCostRow]) -> None: ...


BackfillStatus = Literal["inserted", "skipped", "error", "dry_run"]


@dataclass(frozen=True)
class BackfillBuildRecord:
    build_id: str
    status: BackfillStatus
    price_snapshot_id: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BackfillSummary:
    processed: int
    inserted: int
    skipped: int
    errors: int
    builds: tuple[BackfillBuildRecord, ...]

    def to_dict(self) -> dict:
        return {
            "processed": self.processed,
            "inserted": self.inserted,
            "skipped": self.skipped,
            "errors": self.errors,
            "builds": [record.to_dict() for record in self.builds],
        }


def _resolve_data_path(data_path: Path | str | None) -> Path:
    target = Path(data_path) if data_path is not None else Path.cwd()
    return target.resolve()


def _derive_snapshot_id(ruleset_id: str, override: str | None) -> str:
    if override is not None:
        return override
    _, _, snapshot_id = parse_ruleset_id(ruleset_id)
    return snapshot_id


def run_cost_backfill(
    repo: BackfillRepository,
    *,
    data_path: Path | str | None = None,
    price_snapshot_id: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    page_size: int = 100,
) -> BackfillSummary:
    if limit is not None and limit <= 0:
        return BackfillSummary(processed=0, inserted=0, skipped=0, errors=0, builds=())

    resolved_root = _resolve_data_path(data_path)
    processed = inserted = skipped = errors = 0
    build_records: list[BackfillBuildRecord] = []

    offset = 0
    while True:
        if limit is not None and processed >= limit:
            break
        remaining = None if limit is None else limit - processed
        batch_limit = page_size if remaining is None else min(page_size, remaining)
        if batch_limit <= 0:
            break
        rows = repo.list_builds(limit=batch_limit, offset=offset)
        if not rows:
            break
        consumed = 0
        for row in rows:
            if limit is not None and processed >= limit:
                break
            build_id = row.get("build_id")
            if not build_id:
                consumed += 1
                processed += 1
                errors += 1
                build_records.append(
                    BackfillBuildRecord(
                        build_id="",
                        status="error",
                        detail="row missing build_id",
                    )
                )
                continue
            processed += 1
            consumed += 1
            try:
                snapshot_id = _derive_snapshot_id(row.get("ruleset_id", ""), price_snapshot_id)
            except Exception as exc:
                errors += 1
                build_records.append(
                    BackfillBuildRecord(
                        build_id=build_id,
                        status="error",
                        detail=str(exc),
                    )
                )
                continue

            latest = repo.get_latest_build_cost(build_id)
            if latest is not None and latest.get("price_snapshot_id") == snapshot_id:
                skipped += 1
                build_records.append(
                    BackfillBuildRecord(
                        build_id=build_id,
                        status="skipped",
                        price_snapshot_id=snapshot_id,
                        detail="already up to date",
                    )
                )
                continue

            try:
                requirements = extract_build_cost_requirements(build_id, base_path=resolved_root)
                snapshot = load_price_snapshot(snapshot_id, data_path=resolved_root)
                summary = calculate_cost_summary(
                    requirements.unique_items, requirements.skill_gems, snapshot
                )
            except Exception as exc:
                errors += 1
                build_records.append(
                    BackfillBuildRecord(
                        build_id=build_id,
                        status="error",
                        price_snapshot_id=snapshot_id,
                        detail=str(exc),
                    )
                )
                continue

            if dry_run:
                build_records.append(
                    BackfillBuildRecord(
                        build_id=build_id,
                        status="dry_run",
                        price_snapshot_id=snapshot_id,
                        detail="dry run prevented insert",
                    )
                )
                continue

            artifact_paths = artifact_store.artifact_paths(build_id, base_path=resolved_root)
            slot_path = artifact_paths.base_dir / "slot_costs.json"
            gem_path = artifact_paths.base_dir / "gem_costs.json"
            try:
                write_cost_outputs(build_id, summary, base_path=resolved_root)
                slot_relative = str(slot_path.relative_to(resolved_root))
                gem_relative = str(gem_path.relative_to(resolved_root))
                row_payload = BuildCostRow(
                    build_id=build_id,
                    ruleset_id=row.get("ruleset_id", ""),
                    price_snapshot_id=snapshot_id,
                    total_cost_chaos=summary.total_cost_chaos,
                    unknown_cost_count=summary.unknown_cost_count,
                    slot_costs_json_path=slot_relative,
                    gem_costs_json_path=gem_relative,
                    calculated_at=datetime.now(timezone.utc),
                )
                repo.insert_build_costs((row_payload,))
                inserted += 1
                build_records.append(
                    BackfillBuildRecord(
                        build_id=build_id,
                        status="inserted",
                        price_snapshot_id=snapshot_id,
                    )
                )
            except Exception as exc:
                errors += 1
                build_records.append(
                    BackfillBuildRecord(
                        build_id=build_id,
                        status="error",
                        detail=str(exc),
                    )
                )
        offset += consumed
        if limit is not None and processed >= limit:
            break
    return BackfillSummary(
        processed=processed,
        inserted=inserted,
        skipped=skipped,
        errors=errors,
        builds=tuple(build_records),
    )
