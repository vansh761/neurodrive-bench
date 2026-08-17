from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

from neurodrive_bench.config import BenchmarkConfig
from neurodrive_bench.contracts import EpisodeMetrics, StressProfile
from neurodrive_bench.metrics.gdi import compute_gdi
from neurodrive_bench.metrics.stats import aggregate as aggregate_values
from neurodrive_bench.metrics.summary import build_metrics_from_trace
from neurodrive_bench.models.registry import build_model
from neurodrive_bench.reporting.summary import build_benchmark_summary, save_benchmark_summary
from neurodrive_bench.simulation.factory import build_simulation_backend
from neurodrive_bench.stress.engine import scale_stress


def _aggregate_level(
    model_name: str, stress_level: float, level_metrics: list[dict[str, object]]
) -> dict[str, object]:
    """Aggregate per-episode metrics at one (model, stress_level) into mean/std/n.

    Every numeric field on EpisodeMetrics gets its own {mean, std, min, max, n} so
    downstream reporting can show error bars instead of a single noisy point
    estimate from one deterministic episode.
    """
    metric_names = [f.name for f in EpisodeMetrics.__dataclass_fields__.values()]
    metrics_aggregate: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = [float(item["metrics"][name]) for item in level_metrics]
        metrics_aggregate[name] = aggregate_values(values)

    seeds = [item["seed"] for item in level_metrics]
    return {
        "model_name": model_name,
        "stress_level": stress_level,
        "episode_count": len(level_metrics),
        "seeds": seeds,
        "metrics_aggregate": metrics_aggregate,
        "per_episode_artifacts": [item["artifact_path"] for item in level_metrics],
    }


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.backend = build_simulation_backend(config)

    def run(self) -> str:
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_artifacts: list[dict[str, object]] = []

        base_stress = self._build_base_stress_profile()
        lines: list[str] = [
            f"Benchmark: {self.config.benchmark_name}",
            f"Backend: {self.config.simulation_backend}",
            f"Map: {self.config.map_name}",
        ]

        self.backend.setup()
        try:
            for model_spec in self.config.model_specs:
                model = build_model(model_spec, history_window=self.config.telemetry_history_window)
                model.reset()
                lines.append(f"Model: {model.name}")

                for stress_level in self.config.stress_levels:
                    stress = scale_stress(base_stress, stress_level)
                    level_metrics: list[dict[str, object]] = []

                    for episode_idx in range(self.config.episodes_per_level):
                        episode_seed = self._episode_seed(model.name, stress_level, episode_idx)
                        random.seed(episode_seed)
                        model.reset()
                        episode = self.backend.simulate_episode(
                            model=model,
                            stress_level=stress_level,
                            stress=stress,
                        )
                        metrics = build_metrics_from_trace(episode)
                        metrics.graceful_degradation_index = compute_gdi(metrics)

                        artifact_file = output_dir / (
                            f"{model.name}_stress_{stress_level:.2f}_ep{episode_idx:02d}.json"
                        )
                        payload = {
                            "model_name": model.name,
                            "stress_level": stress_level,
                            "episode_index": episode_idx,
                            "seed": episode_seed,
                            "stress_profile": asdict(stress),
                            "metrics": asdict(metrics),
                            "episode": asdict(episode),
                        }
                        artifact_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                        payload["artifact_path"] = str(artifact_file)
                        level_metrics.append(payload)

                    aggregated = _aggregate_level(model.name, stress_level, level_metrics)
                    agg_file = output_dir / f"{model.name}_stress_{stress_level:.2f}_aggregate.json"
                    agg_file.write_text(json.dumps(aggregated, indent=2), encoding="utf-8")
                    aggregated["artifact_path"] = str(agg_file)
                    saved_artifacts.append(aggregated)

                    gdi_agg = aggregated["metrics_aggregate"]["graceful_degradation_index"]
                    collision_agg = aggregated["metrics_aggregate"]["collision_rate"]
                    lines.append(
                        f"  stress={stress_level:.2f} "
                        f"collision={collision_agg['mean']:.3f}\u00b1{collision_agg['std']:.3f} "
                        f"gdi={gdi_agg['mean']:.3f}\u00b1{gdi_agg['std']:.3f} "
                        f"(n={self.config.episodes_per_level}) artifact={agg_file.name}"
                    )
        finally:
            self.backend.teardown()

        summary = build_benchmark_summary(saved_artifacts, benchmark_name=self.config.benchmark_name)
        summary_path = save_benchmark_summary(summary, output_dir)
        lines.append(f"Summary: {summary_path.name}")

        return "\n".join(lines)

    def _build_base_stress_profile(self) -> StressProfile:
        stress = self.config.raw["stress"]
        return StressProfile(
            rain=float(stress["environmental"]["rain"]),
            fog=float(stress["environmental"]["fog"]),
            night=bool(stress["environmental"]["night"]),
            noise_std=float(stress["sensor"]["noise_std"]),
            packet_dropout=float(stress["sensor"]["packet_dropout"]),
            latency_steps=int(stress["sensor"]["latency_steps"]),
            sudden_obstacle_probability=float(stress["scenario"]["sudden_obstacle_probability"]),
            lane_blockage_probability=float(stress["scenario"]["lane_blockage_probability"]),
            traffic_multiplier=float(stress["scenario"]["traffic_multiplier"]),
        )

    def _episode_seed(self, model_name: str, stress_level: float, episode_idx: int) -> int:
        seed_material = f"{self.config.benchmark_seed}:{model_name}:{stress_level:.4f}:{episode_idx}"
        digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)
