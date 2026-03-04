"""Archive visualization and monitoring tools.

Generates heatmaps, coverage statistics, and exports archive data to HTML/JSON formats.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from backend.app.settings import settings
from backend.engine.archive import (
    DEFAULT_DESCRIPTOR_AXES,
    ArchiveStore,
    DescriptorAxisSpec,
    load_archive_artifact,
)


@dataclass(frozen=True)
class BinCoverageStats:
    """Statistics for a single bin in the archive."""

    bin_key: str
    build_id: str
    score: float
    descriptor_values: tuple[float, ...]
    coverage_percentage: float


@dataclass(frozen=True)
class AxisCoverageStats:
    """Coverage statistics for a single axis."""

    axis_name: str
    bins_filled: int
    total_bins: int
    coverage_percentage: float
    min_value: float
    max_value: float


@dataclass(frozen=True)
class ArchiveVisualizationData:
    """Complete visualization data for an archive."""

    run_id: str
    timestamp: str
    total_bins: int
    bins_filled: int
    coverage_percentage: float
    qd_score: float
    axes: list[dict[str, Any]]
    bin_entries: list[dict[str, Any]]
    axis_coverage: list[dict[str, Any]]
    heatmap_data: dict[str, list[list[float]]]


def _get_axis_coverage(
    store: ArchiveStore,
    axis_index: int,
) -> AxisCoverageStats:
    """Calculate coverage statistics for a single axis."""
    axis = store.axes[axis_index]
    entries = store.entries()

    # Count filled bins for this axis
    filled_bins = set()
    for entry in entries:
        bin_indices = [int(x) for x in entry.bin_key.split("-")]
        if axis_index < len(bin_indices):
            filled_bins.add(bin_indices[axis_index])

    coverage = len(filled_bins) / axis.bins if axis.bins > 0 else 0.0

    return AxisCoverageStats(
        axis_name=axis.name,
        bins_filled=len(filled_bins),
        total_bins=axis.bins,
        coverage_percentage=coverage * 100.0,
        min_value=axis.min_value,
        max_value=axis.max_value,
    )


def _generate_heatmap_data(
    store: ArchiveStore,
) -> dict[str, list[list[float]]]:
    """Generate heatmap data for all axis pairs.

    Returns a dict mapping "axis1_vs_axis2" to a 2D grid of scores.
    """
    heatmaps: dict[str, list[list[float]]] = {}
    axes = store.axes
    entries = store.entries()

    # Generate heatmaps for each pair of axes
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            axis_x = axes[i]
            axis_y = axes[j]
            key = f"{axis_x.name}_vs_{axis_y.name}"

            # Initialize grid with zeros
            grid: list[list[float]] = [
                [0.0 for _ in range(axis_x.bins)] for _ in range(axis_y.bins)
            ]

            # Fill grid with scores
            for entry in entries:
                bin_indices = [int(x) for x in entry.bin_key.split("-")]
                if i < len(bin_indices) and j < len(bin_indices):
                    x_idx = bin_indices[i]
                    y_idx = bin_indices[j]
                    if 0 <= x_idx < axis_x.bins and 0 <= y_idx < axis_y.bins:
                        grid[y_idx][x_idx] = max(grid[y_idx][x_idx], entry.score)

            heatmaps[key] = grid

    return heatmaps


def visualize_archive(
    run_id: str,
    base_path: Path | None = None,
) -> ArchiveVisualizationData:
    """Load archive and generate visualization data.

    Args:
        run_id: The run ID to visualize
        base_path: Optional base path for data directory

    Returns:
        ArchiveVisualizationData with all visualization metrics
    """
    store = load_archive_artifact(run_id, base_path)
    metrics = store.metrics()

    # Prepare axis data
    axes_data = [axis.to_dict() for axis in store.axes]

    # Prepare bin entries
    bin_entries = [
        {
            "bin_key": entry.bin_key,
            "build_id": entry.build_id,
            "score": entry.score,
            "descriptor_values": list(entry.descriptor_values),
        }
        for entry in store.entries()
    ]

    # Calculate axis coverage
    axis_coverage = [asdict(_get_axis_coverage(store, i)) for i in range(len(store.axes))]

    # Generate heatmaps
    heatmaps = _generate_heatmap_data(store)

    return ArchiveVisualizationData(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_bins=metrics.total_bins,
        bins_filled=metrics.bins_filled,
        coverage_percentage=metrics.coverage * 100.0,
        qd_score=metrics.qd_score,
        axes=axes_data,
        bin_entries=bin_entries,
        axis_coverage=axis_coverage,
        heatmap_data=heatmaps,
    )


def export_to_json(
    data: ArchiveVisualizationData,
    output_path: Path,
) -> None:
    """Export visualization data to JSON format.

    Args:
        data: The visualization data to export
        output_path: Path to write JSON file
    """
    output_dict = asdict(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_dict, f, indent=2)


def export_to_html(
    data: ArchiveVisualizationData,
    output_path: Path,
) -> None:
    """Export visualization data to interactive HTML format.

    Args:
        data: The visualization data to export
        output_path: Path to write HTML file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build HTML with embedded data and basic visualization
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Archive Visualization - {data.run_id}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            margin-top: 0;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .chart-container {{
            margin: 30px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            background: #fafafa;
        }}
        .chart-title {{
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #f0f0f0;
            font-weight: bold;
        }}
        tr:hover {{
            background: #f9f9f9;
        }}
        .timestamp {{
            color: #999;
            font-size: 12px;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Archive Visualization: {data.run_id}</h1>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Coverage</div>
                <div class="metric-value">{data.coverage_percentage:.1f}%</div>
                <div style="font-size: 12px; color: #666;">
                    {data.bins_filled} / {data.total_bins} bins
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">QD Score</div>
                <div class="metric-value">{data.qd_score:.0f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Bins Filled</div>
                <div class="metric-value">{data.bins_filled}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Bins</div>
                <div class="metric-value">{data.total_bins}</div>
            </div>
        </div>

        <div class="chart-container">
            <div class="chart-title">Axis Coverage Statistics</div>
            <table>
                <thead>
                    <tr>
                        <th>Axis</th>
                        <th>Bins Filled</th>
                        <th>Total Bins</th>
                        <th>Coverage %</th>
                        <th>Range</th>
                    </tr>
                </thead>
                <tbody>
"""

    for axis_cov in data.axis_coverage:
        html_content += f"""                    <tr>
                        <td>{axis_cov["axis_name"]}</td>
                        <td>{axis_cov["bins_filled"]}</td>
                        <td>{axis_cov["total_bins"]}</td>
                        <td>{axis_cov["coverage_percentage"]:.1f}%</td>
                        <td>[{axis_cov["min_value"]:.2f}, {axis_cov["max_value"]:.2f}]</td>
                    </tr>
"""

    html_content += """                </tbody>
            </table>
        </div>

        <div class="chart-container">
            <div class="chart-title">Heatmaps by Axis Pair</div>
            <div id="heatmaps"></div>
        </div>

        <div class="chart-container">
            <div class="chart-title">Top Scoring Builds</div>
            <table>
                <thead>
                    <tr>
                        <th>Build ID</th>
                        <th>Score</th>
                        <th>Bin Key</th>
                    </tr>
                </thead>
                <tbody>
"""

    # Sort entries by score descending and show top 20
    sorted_entries = sorted(data.bin_entries, key=lambda x: x["score"], reverse=True)
    for entry in sorted_entries[:20]:
        html_content += f"""                    <tr>
                        <td>{entry["build_id"]}</td>
                        <td>{entry["score"]:.0f}</td>
                        <td>{entry["bin_key"]}</td>
                    </tr>
"""

    html_content += (
        """                </tbody>
            </table>
        </div>

        <script>
            const heatmapData = """
        + json.dumps(data.heatmap_data)
        + """;
            const container = document.getElementById('heatmaps');

            Object.entries(heatmapData).forEach(([key, grid]) => {
                const div = document.createElement('div');
                div.style.marginBottom = '30px';
                container.appendChild(div);

                const trace = {
                    z: grid,
                    type: 'heatmap',
                    colorscale: 'Viridis'
                };

                const layout = {
                    title: key,
                    xaxis: { title: key.split('_vs_')[0] },
                    yaxis: { title: key.split('_vs_')[1] },
                    height: 400
                };

                Plotly.newPlot(div, [trace], layout, {responsive: true});
            });
        </script>

        <div class="timestamp">Generated: {data.timestamp}</div>
    </div>
</body>
</html>
"""
    )

    with open(output_path, "w") as f:
        f.write(html_content)


def main() -> None:
    """CLI entry point for archive visualization."""
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

    # Set output directory
    output_dir = args.output_dir or (Path(settings.data_path).parent / "reports" / "archive-viz")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualization data
    print(f"Loading archive for run: {args.run_id}")
    viz_data = visualize_archive(args.run_id, args.data_path)

    # Export based on format
    if args.format in ("json", "both"):
        json_path = output_dir / f"{args.run_id}_archive.json"
        print(f"Exporting to JSON: {json_path}")
        export_to_json(viz_data, json_path)

    if args.format in ("html", "both"):
        html_path = output_dir / f"{args.run_id}_archive.html"
        print(f"Exporting to HTML: {html_path}")
        export_to_html(viz_data, html_path)

    # Print summary
    print("\n=== Archive Visualization Summary ===")
    print(f"Run ID: {viz_data.run_id}")
    print(
        f"Coverage: {viz_data.coverage_percentage:.1f}% ({viz_data.bins_filled}/{viz_data.total_bins})"
    )
    print(f"QD Score: {viz_data.qd_score:.0f}")
    print(f"Axes: {len(viz_data.axes)}")
    print(f"Heatmaps: {len(viz_data.heatmap_data)}")


if __name__ == "__main__":
    main()
