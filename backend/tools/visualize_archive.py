"""Archive visualization CLI wrapper."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..app.settings import settings
from ..reports.archive_viz import export_to_html, export_to_json, visualize_archive


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize and export archive data",
    )
    parser.add_argument("run_id", help="Run ID to visualize")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for exports (default: reports/archive-viz)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "html", "both"],
        default="both",
        help="Export format",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Base data path (default: from settings)",
    )

    args = parser.parse_args()

    output_dir = args.output_dir or (Path(settings.data_path).parent / "reports" / "archive-viz")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading archive for run: {args.run_id}")
    viz_data = visualize_archive(args.run_id, args.data_path)

    if args.format in ("json", "both"):
        json_path = output_dir / f"{args.run_id}_archive.json"
        print(f"Exporting to JSON: {json_path}")
        export_to_json(viz_data, json_path)

    if args.format in ("html", "both"):
        html_path = output_dir / f"{args.run_id}_archive.html"
        print(f"Exporting to HTML: {html_path}")
        export_to_html(viz_data, html_path)

    print("\n=== Archive Visualization Summary ===")
    print(f"Run ID: {viz_data.run_id}")
    coverage_line = (
        "Coverage: "
        f"{viz_data.coverage_percentage:.1f}% "
        f"({viz_data.bins_filled}/{viz_data.total_bins})"
    )
    print(coverage_line)
    print(f"QD Score: {viz_data.qd_score:.0f}")
    print(f"Axes: {len(viz_data.axes)}")
    print(f"Heatmaps: {len(viz_data.heatmap_data)}")


if __name__ == "__main__":
    main()
