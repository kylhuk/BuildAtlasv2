"""Command-line helper for build cost backfill."""

import argparse
import json
from pathlib import Path

from backend.app.db.ch import ClickhouseRepository
from backend.engine.pricing.backfill import run_cost_backfill


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill build costs from existing artifacts.")
    parser.add_argument("--price-snapshot-id", help="Override the snapshot id for every build")
    parser.add_argument("--limit", type=int, help="Maximum number of builds to process")
    parser.add_argument("--dry-run", action="store_true", help="Skip writing outputs and inserts")
    parser.add_argument(
        "--data-path",
        default="data",
        help="Base directory for artifacts and prices",
    )
    args = parser.parse_args()

    repo = ClickhouseRepository()
    try:
        summary = run_cost_backfill(
            repo,
            data_path=Path(args.data_path),
            price_snapshot_id=args.price_snapshot_id,
            limit=args.limit,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        raise SystemExit(f"backfill failed: {exc}") from exc

    print(json.dumps(summary.to_dict(), indent=2))


if __name__ == "__main__":
    main()
