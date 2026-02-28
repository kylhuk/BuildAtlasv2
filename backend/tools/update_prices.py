"""CLI helper to ingest price snapshots."""

import argparse
from pathlib import Path

from backend.engine.pricing import snapshots


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest price snapshots from fixtures.")
    parser.add_argument("--league", required=True)
    parser.add_argument(
        "--source",
        default="fixtures",
        choices=["fixtures"],
        help="Source type for the snapshot",
    )
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--data-path", default="data", help="Base directory for snapshot artifacts")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    result = snapshots.ingest_price_snapshot_from_fixtures(
        league=args.league,
        timestamp=args.timestamp,
        data_path=data_path,
        sources=(args.source,),
    )

    print(f"price snapshot: {result.snapshot_id}")
    print(f"league: {result.league}")
    print(f"timestamp: {result.timestamp}")
    print(f"snapshot path: {result.snapshot_path}")
    print(f"index path: {result.index_path}")


if __name__ == "__main__":
    main()
