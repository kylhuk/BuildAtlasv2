"""CLI helper for building EP-V4 dataset snapshots."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from backend.engine.surrogate.dataset import build_dataset_snapshot


def _default_snapshot_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"snapshot-{timestamp}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build EP-V4 surrogate dataset snapshots",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        type=Path,
        help="Root path that contains data/builds",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Directory to write data/datasets/ep-v4/<snapshot>",
    )
    parser.add_argument(
        "--snapshot-id",
        type=str,
        help="Unique snapshot identifier (defaults to timestamped)",
    )
    args = parser.parse_args(argv)

    snapshot_id = args.snapshot_id or _default_snapshot_id()
    result = build_dataset_snapshot(
        data_path=args.data_path,
        output_root=args.output_root,
        snapshot_id=snapshot_id,
    )

    summary = {
        "snapshot_id": result.snapshot_id,
        "dataset_path": str(result.dataset_path),
        "manifest_path": str(result.manifest_path),
        "row_count": result.row_count,
        "feature_schema_version": result.feature_schema_version,
        "dataset_hash": result.dataset_hash,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
