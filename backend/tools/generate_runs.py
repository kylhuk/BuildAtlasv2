from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

from backend.app.settings import settings
from backend.engine.generation.runner import run_generation
from backend.engine.ruleset import (
    DEFAULT_PRICE_SNAPSHOT_ID,
    derive_ruleset_id,
    read_pob_commit,
    scenario_version_from_profile,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EP-V2-06 build generation")
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of candidates to generate",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1,
        help="Starting deterministic seed",
    )
    parser.add_argument(
        "--ruleset-id",
        default=None,
        help=(
            "Ruleset identifier to tag the generated builds; derived from "
            "PoB commit/scenario/price when omitted"
        ),
    )
    parser.add_argument(
        "--profile-id",
        default="pinnacle",
        help="Scenario profile identifier",
    )

    parser.add_argument(
        "--scenario-version",
        default=None,
        help="Optional scenario version tag (e.g., pinnacle@v0) to derive ruleset",
    )
    parser.add_argument(
        "--price-snapshot-id",
        default=DEFAULT_PRICE_SNAPSHOT_ID,
        help="Price snapshot identifier for derived ruleset",
    )
    parser.add_argument(
        "--pob-commit",
        default=None,
        help="PoB commit hash to tag ruleset; defaults to pob/VERSION",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run identifier to persist summary",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=settings.data_path,
        help="Base directory for artifacts and run summaries",
    )

    parser.add_argument(
        "--constraints-file",
        type=Path,
        default=None,
        help="Path to constraints JSON payload",
    )

    parser.add_argument(
        "--run-mode",
        choices=["standard", "optimizer"],
        default="standard",
        help="Run mode for generation (standard or optimizer)",
    )
    parser.add_argument(
        "--optimizer-iterations",
        type=int,
        default=1,
        help="Iterations of optimizer refinement after warmup",
    )
    parser.add_argument(
        "--optimizer-elite-count",
        type=int,
        default=2,
        help="Number of elites to carry between optimizer iterations",
    )

    parser.add_argument(
        "--surrogate-enabled",
        action="store_true",
        help="Enable surrogate-assisted candidate selection",
    )
    parser.add_argument(
        "--surrogate-model-path",
        type=Path,
        default=None,
        help="Path to surrogate scoring model file",
    )
    parser.add_argument(
        "--surrogate-exploration-pct",
        type=float,
        default=0.2,
        help="Exploration sample percentage for surrogate selection",
    )
    parser.add_argument(
        "--surrogate-top-k",
        type=int,
        default=None,
        help="Number of surrogate-top candidates to evaluate",
    )
    return parser.parse_args(argv)


def _load_constraints(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        payload = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"unable to read constraints file {path}: {exc}") from exc
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON in constraints file {path}: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    ruleset_id = args.ruleset_id
    if not ruleset_id:
        pob_commit = args.pob_commit or read_pob_commit()
        scenario_version = args.scenario_version or scenario_version_from_profile(args.profile_id)
        ruleset_id = derive_ruleset_id(
            pob_commit=pob_commit,
            scenario_version=scenario_version,
            price_snapshot_id=args.price_snapshot_id,
        )
        logger.info("derived ruleset_id=%s", ruleset_id)
    try:
        constraints_payload = _load_constraints(args.constraints_file)
    except ValueError as exc:
        logger.error("constraints loading failed: %s", exc)
        return 1
    try:
        summary = run_generation(
            count=args.count,
            seed_start=args.seed_start,
            ruleset_id=ruleset_id,
            profile_id=args.profile_id,
            run_id=args.run_id,
            base_path=args.data_path,
            constraints=constraints_payload,
            surrogate_enabled=args.surrogate_enabled,
            surrogate_model_path=args.surrogate_model_path,
            surrogate_exploration_pct=args.surrogate_exploration_pct,
            surrogate_top_k=args.surrogate_top_k,
            run_mode=args.run_mode,
            optimizer_iterations=args.optimizer_iterations,
            optimizer_elite_count=args.optimizer_elite_count,
        )
        summary_path = summary.get("paths", {}).get("summary")
        logger.info(
            "generation run %s finished, summary=%s",
            summary["run_id"],
            summary_path,
        )
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("generation failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
