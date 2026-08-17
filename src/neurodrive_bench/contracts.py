from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TelemetryFrame:
    speed: float
    lane_offset: float
    yaw: float
    heading_error: float
    obstacle_distance: float
    acceleration: float
    delta_t: float


@dataclass(slots=True)
class StressProfile:
    rain: float = 0.0
    fog: float = 0.0
    night: bool = False
    noise_std: float = 0.0
    packet_dropout: float = 0.0
    latency_steps: int = 0
    sudden_obstacle_probability: float = 0.0
    lane_blockage_probability: float = 0.0
    traffic_multiplier: float = 1.0


@dataclass(slots=True)
class ControlCommand:
    steering: float
    throttle: float
    brake: float


@dataclass(slots=True)
class ModelOutput:
    command: ControlCommand
    uncertainty_score: float = 0.0
    adaptation_level: float = 0.0
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EpisodeMetrics:
    collision_rate: float
    offroad_frequency: float
    rule_violations: float
    steering_oscillation: float
    trajectory_smoothness: float
    control_jitter: float
    recovery_time: float
    stabilization_speed: float
    mean_uncertainty: float
    mean_adaptation: float
    adaptation_latency: float
    graceful_degradation_index: float


@dataclass(slots=True)
class BenchmarkResult:
    benchmark_name: str
    model_name: str
    stress_level: float
    metrics: EpisodeMetrics
    artifacts_dir: Path


@dataclass(slots=True)
class EpisodeEvent:
    step: int
    event_type: str
    severity: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EpisodeTrace:
    backend: str
    model_name: str
    stress_level: float
    total_steps: int
    events: list[EpisodeEvent] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
