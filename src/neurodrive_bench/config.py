from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class BenchmarkConfig:
    raw: dict[str, Any]

    @property
    def benchmark_name(self) -> str:
        return str(self.raw["benchmark"]["name"])

    @property
    def output_dir(self) -> Path:
        return Path(self.raw["benchmark"]["output_dir"])

    @property
    def simulation_backend(self) -> str:
        return str(self.raw["simulation"]["backend"])

    @property
    def fixed_delta_seconds(self) -> float:
        return float(self.raw["simulation"]["fixed_delta_seconds"])

    @property
    def map_name(self) -> str:
        return str(self.raw["simulation"]["map_name"])

    @property
    def carla_host(self) -> str:
        return str(self.raw["simulation"].get("host", "localhost"))

    @property
    def carla_port(self) -> int:
        return int(self.raw["simulation"].get("port", 2000))

    @property
    def carla_timeout_seconds(self) -> float:
        return float(self.raw["simulation"].get("timeout_seconds", 10.0))

    @property
    def max_episode_steps(self) -> int:
        return int(self.raw["simulation"].get("max_episode_steps", 100))

    @property
    def ego_vehicle_filter(self) -> str:
        return str(self.raw["simulation"].get("ego_vehicle_filter", "vehicle.tesla.model3"))

    @property
    def scenario_vehicle_filter(self) -> str:
        return str(self.raw["simulation"].get("scenario_vehicle_filter", "vehicle.*"))

    @property
    def sample_stride(self) -> int:
        return int(self.raw["simulation"].get("sample_stride", 5))

    @property
    def stress_levels(self) -> list[float]:
        return [float(level) for level in self.raw["evaluation"]["stress_levels"]]

    @property
    def episodes_per_level(self) -> int:
        # NOTE: this key existed in benchmark.example.yaml from the start but was
        # never actually read anywhere in the codebase -- the runner simulated
        # exactly one deterministic episode per (model, stress_level) pair, so
        # "5 episodes per level" was never true. See orchestration/runner.py for
        # the fix that actually loops this many times and aggregates mean/std.
        return int(self.raw["evaluation"].get("episodes_per_level", 1))

    @property
    def model_specs(self) -> list[dict[str, Any]]:
        return list(self.raw["models"])

    @property
    def telemetry_history_window(self) -> int:
        return int(self.raw["telemetry"]["history_window"])

    @property
    def benchmark_seed(self) -> int:
        return int(self.raw["benchmark"].get("seed", 0))

    @property
    def training_config(self) -> dict[str, Any]:
        return dict(self.raw.get("training", {}))

    @property
    def neural_training_config(self) -> dict[str, Any]:
        return dict(self.raw.get("neural_training", {}))

    @property
    def data_collection_config(self) -> dict[str, Any]:
        return dict(self.raw.get("data_collection", {}))


def load_config(path: str | Path) -> BenchmarkConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if raw["simulation"]["synchronous_mode"] is not True:
        raise ValueError("CARLA benchmark configs must enable synchronous_mode=true.")

    return BenchmarkConfig(raw=raw)
