from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def export_figure_data(output_dir: str | Path) -> list[Path]:
    output_path = Path(output_dir)
    summary_path = output_path / "benchmark_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Benchmark summary not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    artifacts = _load_artifacts(output_path)
    exports_dir = output_path / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        _write_leaderboard_csv(summary, exports_dir),
        _write_degradation_curves_csv(summary, exports_dir),
        _write_failure_events_csv(artifacts, exports_dir),
    ]
    return paths


def _load_artifacts(output_dir: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("*.json")):
        if path.name == "benchmark_summary.json":
            continue
        artifacts.append(json.loads(path.read_text(encoding="utf-8")))
    return artifacts


def _write_leaderboard_csv(summary: dict[str, Any], exports_dir: Path) -> Path:
    path = exports_dir / "leaderboard.csv"
    fieldnames = ["rank", "model_name", "mean_gdi", "degradation_slope", "mean_collision_rate"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(summary.get("leaderboard", []), start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "model_name": row.get("model_name"),
                    "mean_gdi": row.get("mean_gdi"),
                    "degradation_slope": row.get("degradation_slope"),
                    "mean_collision_rate": row.get("mean_collision_rate"),
                }
            )
    return path


def _write_degradation_curves_csv(summary: dict[str, Any], exports_dir: Path) -> Path:
    path = exports_dir / "degradation_curves.csv"
    fieldnames = [
        "model_name", "stress_level", "episode_count", "seeds",
        "gdi_mean", "gdi_std", "collision_rate_mean", "collision_rate_std",
        "stabilization_speed_mean", "mean_uncertainty_mean", "mean_adaptation_mean",
        "adaptation_latency_mean",
    ]

    def _stat(point: dict[str, Any], key: str, field: str) -> float:
        entry = point.get(key) or {}
        return entry.get(field, 0.0) if isinstance(entry, dict) else entry

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, model_summary in summary.get("models", {}).items():
            for point in model_summary.get("curve", []):
                writer.writerow(
                    {
                        "model_name": model_name,
                        "stress_level": point.get("stress_level"),
                        "episode_count": point.get("episode_count", 1),
                        "seeds": ";".join(str(s) for s in point.get("seeds", [])),
                        "gdi_mean": _stat(point, "gdi", "mean"),
                        "gdi_std": _stat(point, "gdi", "std"),
                        "collision_rate_mean": _stat(point, "collision_rate", "mean"),
                        "collision_rate_std": _stat(point, "collision_rate", "std"),
                        "stabilization_speed_mean": _stat(point, "stabilization_speed", "mean"),
                        "mean_uncertainty_mean": _stat(point, "mean_uncertainty", "mean"),
                        "mean_adaptation_mean": _stat(point, "mean_adaptation", "mean"),
                        "adaptation_latency_mean": _stat(point, "adaptation_latency", "mean"),
                    }
                )
    return path


def _write_failure_events_csv(artifacts: list[dict[str, Any]], exports_dir: Path) -> Path:
    path = exports_dir / "failure_events.csv"
    fieldnames = [
        "model_name",
        "stress_level",
        "seed",
        "step",
        "event_type",
        "severity",
        "details",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for artifact in artifacts:
            episode = artifact.get("episode", {})
            for event in episode.get("events", []):
                writer.writerow(
                    {
                        "model_name": artifact.get("model_name"),
                        "stress_level": artifact.get("stress_level"),
                        "seed": artifact.get("seed"),
                        "step": event.get("step"),
                        "event_type": event.get("event_type"),
                        "severity": event.get("severity"),
                        "details": json.dumps(event.get("details", {}), sort_keys=True),
                    }
                )
    return path
