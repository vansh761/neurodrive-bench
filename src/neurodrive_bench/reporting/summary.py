from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_benchmark_summary(artifacts: list[dict[str, Any]], benchmark_name: str) -> dict[str, Any]:
    """Build the top-level summary from per-(model, stress_level) aggregates.

    Each `artifact` here is the output of orchestration.runner._aggregate_level:
    already averaged across `episodes_per_level` episodes, carrying mean/std/n
    per metric rather than a single point estimate.
    """
    rows = sorted(
        artifacts,
        key=lambda item: (str(item.get("model_name", "")), float(item.get("stress_level", 0.0))),
    )
    if not rows:
        return {
            "benchmark_name": benchmark_name,
            "artifact_count": 0,
            "models": {},
            "leaderboard": [],
        }

    by_model: dict[str, list[dict[str, Any]]] = {}
    for artifact in rows:
        by_model.setdefault(str(artifact["model_name"]), []).append(artifact)

    models: dict[str, Any] = {}
    leaderboard: list[dict[str, Any]] = []
    for model_name, model_artifacts in by_model.items():
        curve = [
            {
                "stress_level": artifact["stress_level"],
                "episode_count": artifact.get("episode_count", 1),
                "seeds": artifact.get("seeds", []),
                "gdi": artifact["metrics_aggregate"]["graceful_degradation_index"],
                "collision_rate": artifact["metrics_aggregate"]["collision_rate"],
                "stabilization_speed": artifact["metrics_aggregate"]["stabilization_speed"],
                "mean_uncertainty": artifact["metrics_aggregate"].get(
                    "mean_uncertainty", {"mean": 0.0, "std": 0.0}
                ),
                "mean_adaptation": artifact["metrics_aggregate"].get(
                    "mean_adaptation", {"mean": 0.0, "std": 0.0}
                ),
                "adaptation_latency": artifact["metrics_aggregate"].get(
                    "adaptation_latency", {"mean": 0.0, "std": 0.0}
                ),
            }
            for artifact in model_artifacts
        ]

        mean_gdi = _mean([point["gdi"]["mean"] for point in curve])
        mean_collision = _mean([point["collision_rate"]["mean"] for point in curve])
        mean_stabilization = _mean([point["stabilization_speed"]["mean"] for point in curve])
        degradation_slope = _degradation_slope(curve)
        gdi_collision_consistency = _gdi_collision_consistency(curve)

        summary = {
            "curve": curve,
            "aggregate": {
                "mean_gdi": mean_gdi,
                "mean_collision_rate": mean_collision,
                "mean_stabilization_speed": mean_stabilization,
                "degradation_slope": degradation_slope,
                "best_gdi": max(point["gdi"]["mean"] for point in curve),
                "worst_collision_rate": max(point["collision_rate"]["mean"] for point in curve),
                # Spearman-style rank check: does GDI rank stress levels the same way
                # raw collision rate does? +1.0 = perfectly consistent, -1.0 = inverted.
                # A validated GDI should score close to +1.0 here; if it doesn't, the
                # metric may be measuring something other than what collision rate
                # (the most direct safety signal available) measures.
                "gdi_collision_rank_consistency": gdi_collision_consistency,
            },
        }
        models[model_name] = summary
        leaderboard.append(
            {
                "model_name": model_name,
                "mean_gdi": mean_gdi,
                "degradation_slope": degradation_slope,
                "mean_collision_rate": mean_collision,
                "gdi_collision_rank_consistency": gdi_collision_consistency,
            }
        )

    leaderboard.sort(key=lambda row: (-float(row["mean_gdi"]), float(row["mean_collision_rate"])))
    return {
        "benchmark_name": benchmark_name,
        "artifact_count": len(rows),
        "models": models,
        "leaderboard": leaderboard,
    }


def save_benchmark_summary(summary: dict[str, Any], output_dir: str | Path) -> Path:
    output_path = Path(output_dir) / "benchmark_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path


def _mean(values: list[float]) -> float:
    return 0.0 if not values else sum(values) / len(values)


def _degradation_slope(curve: list[dict[str, Any]]) -> float:
    if len(curve) < 2:
        return 0.0
    start = curve[0]
    end = curve[-1]
    delta_stress = float(end["stress_level"]) - float(start["stress_level"])
    if delta_stress == 0:
        return 0.0
    return (float(end["gdi"]["mean"]) - float(start["gdi"]["mean"])) / delta_stress


def _gdi_collision_consistency(curve: list[dict[str, Any]]) -> float:
    """Spearman rank correlation between GDI and (negative) collision rate across
    stress levels for one model. GDI is supposed to fall as collision rate rises,
    so we correlate GDI against -collision_rate; a well-behaved metric should be
    strongly positively correlated (close to +1.0)."""
    if len(curve) < 2:
        return 0.0

    gdi_values = [point["gdi"]["mean"] for point in curve]
    neg_collision_values = [-point["collision_rate"]["mean"] for point in curve]

    gdi_ranks = _ranks(gdi_values)
    collision_ranks = _ranks(neg_collision_values)

    n = len(curve)
    d_squared_sum = sum((g - c) ** 2 for g, c in zip(gdi_ranks, collision_ranks))
    if n < 2:
        return 0.0
    denom = n * (n * n - 1)
    if denom == 0:
        return 0.0
    return 1.0 - (6.0 * d_squared_sum) / denom


def _ranks(values: list[float]) -> list[float]:
    """Average ranks (1-indexed), handling ties by averaging."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks
