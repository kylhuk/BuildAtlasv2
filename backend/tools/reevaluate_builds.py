from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from backend.app.api.evaluator import BuildEvaluator
from backend.app.db.ch import BuildListFilters, ClickhouseRepository
from backend.app.settings import settings
from backend.engine.ruleset import (
    DEFAULT_PRICE_SNAPSHOT_ID,
    derive_ruleset_id,
    read_pob_commit,
    scenario_version_from_profile,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-evaluate builds under a new ruleset")
    parser.add_argument(
        "--build-id",
        dest="build_ids",
        action="append",
        default=[],
        help="Add a specific build to re-evaluate (repeatable)",
    )
    parser.add_argument(
        "--profile-id",
        default=None,
        help="Profile identifier used when deriving a new ruleset",
    )
    parser.add_argument(
        "--ruleset-filter",
        default=None,
        help="Select builds matching this existing ruleset",
    )
    parser.add_argument(
        "--ruleset-id",
        dest="target_ruleset_id",
        default=None,
        help="Explicit target ruleset for the re-evaluation run",
    )
    parser.add_argument(
        "--scenario-version",
        default=None,
        help="Optional scenario version tag (e.g., pinnacle@v0) when deriving the target ruleset",
    )
    parser.add_argument(
        "--price-snapshot-id",
        default=DEFAULT_PRICE_SNAPSHOT_ID,
        help="Price snapshot identifier for derived ruleset",
    )
    parser.add_argument(
        "--pob-commit",
        default=None,
        help="PoB commit hash for derived ruleset; defaults to pob/VERSION",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of builds to target when using filters",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset applied to the filtered build list",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the reevaluation (dry-run by default)",
    )
    parser.add_argument(
        "--mark-stale",
        action="store_true",
        help="Mark all other builds as stale for the target ruleset (requires --apply)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=settings.data_path,
        help="Base directory for artifacts and run summaries",
    )
    return parser.parse_args(argv)


def _select_build_rows(
    repo: ClickhouseRepository,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for build_id in args.build_ids:
        row = repo.get_build(build_id)
        if not row:
            continue
        selected.append(row)
        seen.add(build_id)
    if args.profile_id or args.ruleset_filter:
        filters = BuildListFilters(
            profile_id=args.profile_id,
            ruleset_id=args.ruleset_filter,
            include_stale=True,
        )
        rows = repo.list_builds(
            filters=filters,
            sort_by="created_at",
            sort_dir="asc",
            limit=args.limit,
            offset=args.offset,
        )
        for row in rows:
            build_id = row.get("build_id")
            if not build_id or build_id in seen:
                continue
            selected.append(row)
            seen.add(build_id)
    return selected


def _derive_target_ruleset(args: argparse.Namespace, builds: list[dict[str, Any]]) -> str:
    if args.target_ruleset_id:
        return args.target_ruleset_id
    profile_id = args.profile_id
    if profile_id is None:
        profiles = {row.get("profile_id") for row in builds if row.get("profile_id")}
        if len(profiles) == 1:
            profile_id = profiles.pop()
        else:
            raise RuntimeError(
                "profile_id must be provided when deriving ruleset for multiple profiles"
            )
    if not profile_id:
        raise RuntimeError("profile_id is required to derive a ruleset")
    scenario_version = args.scenario_version or scenario_version_from_profile(profile_id)
    pob_commit = args.pob_commit or read_pob_commit()
    return derive_ruleset_id(
        pob_commit=pob_commit,
        scenario_version=scenario_version,
        price_snapshot_id=args.price_snapshot_id,
    )


def _evaluate_builds(
    repo: ClickhouseRepository,
    evaluator: BuildEvaluator,
    builds: list[dict[str, Any]],
    ruleset_id: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for build in builds:
        build_id = build.get("build_id")
        if not build_id:
            continue
        repo.update_build_ruleset(build_id, ruleset_id=ruleset_id, is_stale=False)
        try:
            status, rows = evaluator.evaluate_build(build_id)
            results.append(
                {
                    "build_id": build_id,
                    "status": status.value,
                    "scenarios": len(rows),
                }
            )
        except Exception as exc:  # best effort
            logger.error("reevaluation failed for %s: %s", build_id, exc)
            results.append({"build_id": build_id, "status": "error", "error": str(exc)})
    return results


def _affected_profiles(args: argparse.Namespace, builds: list[dict[str, Any]]) -> list[str]:
    if args.profile_id:
        return [args.profile_id]
    return sorted({str(row.get("profile_id")) for row in builds if row.get("profile_id")})


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo = ClickhouseRepository()
    builds = _select_build_rows(repo, args)
    if not builds:
        logger.info("no builds matched the selection criteria")
        return 0
    try:
        target_ruleset = _derive_target_ruleset(args, builds)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("failed to determine target ruleset: %s", exc)
        return 1
    logger.info("target ruleset_id=%s", target_ruleset)
    logger.info("selected %d builds", len(builds))
    if args.mark_stale and not args.apply:
        logger.warning("--mark-stale requires --apply; skipping stale update")
    if args.apply and args.mark_stale:
        profiles = _affected_profiles(args, builds)
        if not profiles:
            repo.mark_builds_stale_except(target_ruleset)
        else:
            for profile_id in profiles:
                repo.mark_builds_stale_except(target_ruleset, profile_id=profile_id)
    if not args.apply:
        logger.info("dry run; no evaluations executed")
        return 0

    evaluator = BuildEvaluator(repo=repo, base_path=args.data_path)
    try:
        results = _evaluate_builds(repo, evaluator, builds, target_ruleset)
        successes = sum(1 for entry in results if entry.get("status") == "evaluated")
        failures = len(results) - successes
        logger.info("reevaluation complete: %d successes, %d failures", successes, failures)
        for entry in results:
            logger.info("build=%s status=%s", entry["build_id"], entry["status"])
        return 0
    finally:
        evaluator.close()


if __name__ == "__main__":
    raise SystemExit(main())
