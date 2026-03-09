"""Archive visualization and monitoring tools.

Generates heatmaps, coverage statistics, and exports archive data to HTML/JSON formats.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..engine.archive import load_archive_artifact


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


@dataclass(frozen=True)
class ArchiveArtifactEntry:
    bin_key: str
    build_id: str
    score: float
    descriptor_values: tuple[float, ...]


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_bin_indices(bin_key: str) -> list[int]:
    if not bin_key:
        return []
    indices: list[int] = []
    for part in bin_key.split("-"):
        try:
            indices.append(int(part))
        except ValueError:
            continue
    return indices


def _build_entries(artifact: dict[str, Any]) -> list[ArchiveArtifactEntry]:
    axes = artifact.get("axes", []) or []
    entries: list[ArchiveArtifactEntry] = []
    axis_names = [axis.get("name") or f"axis_{idx}" for idx, axis in enumerate(axes)]
    for entry_data in artifact.get("bins", []) or []:
        descriptor_payload = entry_data.get("descriptor") or {}
        descriptor_values = tuple(
            _coerce_float(descriptor_payload.get(name)) for name in axis_names
        )
        entries.append(
            ArchiveArtifactEntry(
                bin_key=str(entry_data.get("bin_key", "")),
                build_id=str(entry_data.get("build_id", "")),
                score=_coerce_float(entry_data.get("score")),
                descriptor_values=descriptor_values,
            )
        )
    return sorted(entries, key=lambda entry: entry.bin_key)


def _get_axis_coverage(
    axes: list[dict[str, Any]],
    entries: list[ArchiveArtifactEntry],
    axis_index: int,
) -> AxisCoverageStats:
    """Calculate coverage statistics for a single axis using persisted artifact data."""
    axis = axes[axis_index]
    axis_bins = _coerce_int(axis.get("bins"))
    filled_bins: set[int] = set()
    for entry in entries:
        bin_indices = _parse_bin_indices(entry.bin_key)
        if axis_index < len(bin_indices):
            filled_bins.add(bin_indices[axis_index])

    coverage = len(filled_bins) / axis_bins if axis_bins > 0 else 0.0
    axis_name = axis.get("name") or f"axis_{axis_index}"

    return AxisCoverageStats(
        axis_name=axis_name,
        bins_filled=len(filled_bins),
        total_bins=axis_bins,
        coverage_percentage=coverage * 100.0,
        min_value=_coerce_float(axis.get("min_value")),
        max_value=_coerce_float(axis.get("max_value")),
    )


def _generate_heatmap_data(
    axes: list[dict[str, Any]],
    entries: list[ArchiveArtifactEntry],
) -> dict[str, list[list[float]]]:
    """Generate heatmap data for all axis pairs from persisted artifact data."""
    heatmaps: dict[str, list[list[float]]] = {}
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            axis_x = axes[i]
            axis_y = axes[j]
            axis_x_bins = _coerce_int(axis_x.get("bins"))
            axis_y_bins = _coerce_int(axis_y.get("bins"))
            axis_x_name = axis_x.get("name") or f"axis_{i}"
            axis_y_name = axis_y.get("name") or f"axis_{j}"
            key = f"{axis_x_name}_vs_{axis_y_name}"

            grid: list[list[float]] = [
                [0.0 for _ in range(axis_x_bins)] for _ in range(axis_y_bins)
            ]

            for entry in entries:
                bin_indices = _parse_bin_indices(entry.bin_key)
                if i < len(bin_indices) and j < len(bin_indices):
                    x_idx = bin_indices[i]
                    y_idx = bin_indices[j]
                    if 0 <= x_idx < axis_x_bins and 0 <= y_idx < axis_y_bins:
                        grid[y_idx][x_idx] = max(grid[y_idx][x_idx], entry.score)

            heatmaps[key] = grid

    return heatmaps


def visualize_archive(
    run_id: str,
    base_path: Path | None = None,
) -> ArchiveVisualizationData:
    """Load archive artifact and generate visualization data."""
    artifact = load_archive_artifact(run_id, base_path)
    axes = artifact.get("axes", []) or []
    entries = _build_entries(artifact)
    metrics = artifact.get("metrics") or {}

    axes_data = [dict(axis) for axis in axes]
    bin_entries = [
        {
            "bin_key": entry.bin_key,
            "build_id": entry.build_id,
            "score": entry.score,
            "descriptor_values": list(entry.descriptor_values),
        }
        for entry in entries
    ]

    axis_coverage = [asdict(_get_axis_coverage(axes, entries, i)) for i in range(len(axes))]
    heatmaps = _generate_heatmap_data(axes, entries)

    return ArchiveVisualizationData(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_bins=_coerce_int(metrics.get("total_bins")),
        bins_filled=_coerce_int(metrics.get("bins_filled")),
        coverage_percentage=_coerce_float(metrics.get("coverage")) * 100.0,
        qd_score=_coerce_float(metrics.get("qd_score")),
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
