from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


PALETTE = {
    "lstm_neural": "#1f6f8b",
    "transformer_neural": "#b85c38",
    "atsm_lnn_candidate": "#9c27b0",
}


def render_figures(output_dir: str | Path) -> list[Path]:
    output_path = Path(output_dir)
    exports_dir = output_path / "exports"
    figures_dir = output_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    degradation_rows = _read_csv(exports_dir / "degradation_curves.csv")
    failure_rows = _read_csv(exports_dir / "failure_events.csv")

    paths = [
        _render_degradation_svg(degradation_rows, figures_dir / "gdi_degradation_curve.svg"),
        _render_failure_events_svg(failure_rows, figures_dir / "failure_event_counts.svg"),
    ]
    return paths


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required export file not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _render_degradation_svg(rows: list[dict[str, str]], path: Path) -> Path:
    width = 980
    height = 560
    margin_left = 82
    margin_right = 38
    margin_top = 56
    margin_bottom = 78
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    by_model: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        by_model[row["model_name"]].append(
            {
                "stress_level": float(row["stress_level"]),
                "gdi": float(row["gdi_mean"]),
            }
        )

    parts = [_svg_header(width, height), _svg_style()]
    parts.append(f'<rect width="{width}" height="{height}" fill="#f8f8f4"/>')
    parts.append(_title("Graceful Degradation Curve", 32, 34))
    parts.extend(_axes(margin_left, margin_top, plot_width, plot_height, "Stress Level", "GDI"))

    for index in range(6):
        y_value = index / 5
        y = margin_top + plot_height - (y_value * plot_height)
        parts.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" class="grid"/>')
        parts.append(f'<text x="{margin_left - 14}" y="{y + 4:.2f}" text-anchor="end" class="tick">{y_value:.1f}</text>')

    for index in range(5):
        x_value = index / 4
        x = margin_left + (x_value * plot_width)
        parts.append(f'<text x="{x:.2f}" y="{height - 42}" text-anchor="middle" class="tick">{x_value:.2f}</text>')

    legend_y = 74
    for model_index, (model_name, points) in enumerate(sorted(by_model.items())):
        color = PALETTE.get(model_name, "#444")
        sorted_points = sorted(points, key=lambda item: item["stress_level"])
        point_coords = []
        for point in sorted_points:
            x = margin_left + (point["stress_level"] * plot_width)
            y = margin_top + plot_height - (point["gdi"] * plot_height)
            point_coords.append((x, y))

        polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in point_coords)
        parts.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="3.2"/>')
        for x, y in point_coords:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.8" fill="{color}"/>')

        legend_x = margin_left + model_index * 245
        parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 28}" y2="{legend_y}" stroke="{color}" stroke-width="4"/>')
        parts.append(f'<text x="{legend_x + 36}" y="{legend_y + 5}" class="legend">{model_name}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def _render_failure_events_svg(rows: list[dict[str, str]], path: Path) -> Path:
    width = 980
    height = 520
    margin_left = 82
    margin_right = 42
    margin_top = 70
    margin_bottom = 72
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    counts = Counter(row["model_name"] for row in rows)
    models = sorted(counts)
    max_count = max(counts.values(), default=1)
    bar_gap = 34
    bar_width = (plot_width - (bar_gap * max(0, len(models) - 1))) / max(1, len(models))

    parts = [_svg_header(width, height), _svg_style()]
    parts.append(f'<rect width="{width}" height="{height}" fill="#f8f8f4"/>')
    parts.append(_title("Failure Event Counts", 32, 34))
    parts.extend(_axes(margin_left, margin_top, plot_width, plot_height, "Model", "Events"))

    for index in range(6):
        value = max_count * index / 5
        y = margin_top + plot_height - ((value / max_count) * plot_height)
        parts.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" class="grid"/>')
        parts.append(f'<text x="{margin_left - 14}" y="{y + 4:.2f}" text-anchor="end" class="tick">{value:.0f}</text>')

    for index, model_name in enumerate(models):
        count = counts[model_name]
        color = PALETTE.get(model_name, "#444")
        x = margin_left + index * (bar_width + bar_gap)
        bar_height = (count / max_count) * plot_height
        y = margin_top + plot_height - bar_height
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_width / 2:.2f}" y="{y - 10:.2f}" text-anchor="middle" class="label">{count}</text>')
        parts.append(f'<text x="{x + bar_width / 2:.2f}" y="{height - 40}" text-anchor="middle" class="tick">{model_name}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def _svg_header(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'


def _svg_style() -> str:
    return """
<style>
text { font-family: "Segoe UI", "Trebuchet MS", sans-serif; fill: #24313a; }
.title { font-size: 24px; font-weight: 700; }
.axis { stroke: #26343c; stroke-width: 1.4; }
.grid { stroke: #d8d6cc; stroke-width: 1; }
.tick { font-size: 13px; }
.label { font-size: 14px; font-weight: 700; }
.legend { font-size: 14px; font-weight: 600; }
</style>
"""


def _title(text: str, x: int, y: int) -> str:
    return f'<text x="{x}" y="{y}" class="title">{text}</text>'


def _axes(left: int, top: int, plot_width: int, plot_height: int, x_label: str, y_label: str) -> list[str]:
    bottom = top + plot_height
    return [
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{left + plot_width}" y2="{bottom}" class="axis"/>',
        f'<text x="{left + plot_width / 2:.2f}" y="{bottom + 58}" text-anchor="middle" class="label">{x_label}</text>',
        f'<text x="24" y="{top + plot_height / 2:.2f}" transform="rotate(-90 24 {top + plot_height / 2:.2f})" text-anchor="middle" class="label">{y_label}</text>',
    ]
