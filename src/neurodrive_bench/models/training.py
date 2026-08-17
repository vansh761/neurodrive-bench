from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path

from neurodrive_bench.config import BenchmarkConfig
from neurodrive_bench.contracts import TelemetryFrame
from neurodrive_bench.models.profiles import ModelProfile, save_profile
from neurodrive_bench.telemetry.history import TelemetryHistoryBuffer


@dataclass(slots=True)
class TrainingSample:
    features: dict[str, float]
    target_steering: float
    target_throttle: float
    target_uncertainty: float
    target_adaptation: float


class SyntheticModelTrainer:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        training_cfg = config.training_config
        self.samples_per_model = int(training_cfg.get("samples_per_model", 400))
        self.learning_rate = float(training_cfg.get("learning_rate", 0.05))
        self.epochs = int(training_cfg.get("epochs", 60))
        self.output_dir = Path(training_cfg.get("output_dir", "artifacts/model_profiles"))
        self.random = random.Random(config.benchmark_seed)

    def train_all(self) -> list[Path]:
        output_paths: list[Path] = []
        for model_spec in self.config.model_specs:
            profile = self._train_profile(str(model_spec["type"]))
            output_path = self.output_dir / f"{model_spec['name']}.json"
            save_profile(profile, output_path)
            output_paths.append(output_path)
        return output_paths

    def _train_profile(self, model_type: str) -> ModelProfile:
        dataset = self._build_dataset(model_type=model_type)
        steering_keys = sorted(dataset[0].features.keys())
        throttle_keys = sorted(dataset[0].features.keys())
        steering_weights = {key: 0.0 for key in steering_keys}
        throttle_weights = {key: 0.0 for key in throttle_keys}
        steering_bias = 0.0
        throttle_bias = 0.35

        for _ in range(self.epochs):
            for sample in dataset:
                predicted_steering = steering_bias + sum(
                    steering_weights[key] * sample.features[key] for key in steering_keys
                )
                predicted_throttle = throttle_bias + sum(
                    throttle_weights[key] * sample.features[key] for key in throttle_keys
                )

                steering_error = sample.target_steering - predicted_steering
                throttle_error = sample.target_throttle - predicted_throttle

                steering_bias += self.learning_rate * steering_error * 0.01
                throttle_bias += self.learning_rate * throttle_error * 0.01
                for key in steering_keys:
                    steering_weights[key] += self.learning_rate * steering_error * sample.features[key] * 0.01
                    throttle_weights[key] += self.learning_rate * throttle_error * sample.features[key] * 0.01

        uncertainty_bias = sum(sample.target_uncertainty for sample in dataset) / len(dataset)
        adaptation_bias = sum(sample.target_adaptation for sample in dataset) / len(dataset)

        steering_weights["bias"] = steering_bias
        throttle_weights["bias"] = throttle_bias
        return ModelProfile(
            model_type=model_type,
            steering_weights=steering_weights,
            throttle_weights=throttle_weights,
            uncertainty_bias=uncertainty_bias,
            adaptation_bias=adaptation_bias,
            metadata={
                "samples_per_model": self.samples_per_model,
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "history_window": self.config.telemetry_history_window,
                "trainer": "synthetic_gradient_descent",
            },
        )

    def _build_dataset(self, model_type: str) -> list[TrainingSample]:
        dataset: list[TrainingSample] = []
        for _ in range(self.samples_per_model):
            history = TelemetryHistoryBuffer(window_size=self.config.telemetry_history_window)
            lane_base = self.random.uniform(-0.6, 0.6)
            heading_base = self.random.uniform(-5.0, 5.0)
            obstacle = self.random.uniform(3.0, 30.0)

            for step in range(self.config.telemetry_history_window):
                lane_value = lane_base + self.random.uniform(-0.05, 0.05) + (step * self.random.uniform(-0.002, 0.002))
                heading_value = heading_base + self.random.uniform(-0.3, 0.3) + (step * self.random.uniform(-0.03, 0.03))
                frame = TelemetryFrame(
                    speed=self.random.uniform(6.0, 18.0),
                    lane_offset=lane_value,
                    yaw=heading_value * 0.8,
                    heading_error=heading_value,
                    obstacle_distance=max(0.0, obstacle - step * self.random.uniform(0.0, 0.2)),
                    acceleration=self.random.uniform(-8.0, 8.0),
                    delta_t=self.config.fixed_delta_seconds,
                )
                history.add(frame)

            features = _extract_features(history)
            target_steering, target_throttle, target_uncertainty, target_adaptation = _oracle_targets(
                model_type=model_type,
                features=features,
            )
            dataset.append(
                TrainingSample(
                    features=features,
                    target_steering=target_steering,
                    target_throttle=target_throttle,
                    target_uncertainty=target_uncertainty,
                    target_adaptation=target_adaptation,
                )
            )
        return dataset


def _extract_features(history: TelemetryHistoryBuffer) -> dict[str, float]:
    frames = history.frames()
    latest = frames[-1]
    mean_lane = sum(frame.lane_offset for frame in frames) / len(frames)
    mean_heading = sum(frame.heading_error for frame in frames) / len(frames)
    recent_frames = frames[-min(5, len(frames)) :]
    weighted_heading_numerator = sum(
        frame.heading_error * weight
        for frame, weight in zip(recent_frames, range(1, len(recent_frames) + 1))
    )
    weighted_heading_denominator = sum(range(1, len(recent_frames) + 1))
    weighted_heading = weighted_heading_numerator / weighted_heading_denominator
    trend_index = max(0, len(frames) - 4)
    earlier = frames[trend_index]
    heading_trend = latest.heading_error - earlier.heading_error
    lane_trend = latest.lane_offset - earlier.lane_offset
    stability = max(0.0, 1.0 - min(1.0, abs(heading_trend) * 0.08 + abs(lane_trend) * 0.4))

    return {
        "current_lane_offset": latest.lane_offset,
        "current_heading_error": latest.heading_error,
        "mean_lane_offset": mean_lane,
        "mean_heading_error": mean_heading,
        "weighted_heading_error": weighted_heading,
        "heading_trend": heading_trend,
        "lane_trend": lane_trend,
        "stability": stability,
        "obstacle_distance": latest.obstacle_distance,
    }


def _oracle_targets(model_type: str, features: dict[str, float]) -> tuple[float, float, float, float]:
    obstacle_term = max(0.0, 1.0 - min(1.0, features["obstacle_distance"] / 25.0))
    if model_type == "lstm":
        steering = (
            -0.09 * features["mean_lane_offset"]
            -0.015 * features["mean_heading_error"]
            -0.01 * obstacle_term
        )
        throttle = 0.42 - (0.05 * obstacle_term)
        uncertainty = 0.18 + abs(features["mean_heading_error"]) * 0.01
        adaptation = min(1.0, 0.45 + features["stability"] * 0.35)
        return steering, throttle, uncertainty, adaptation

    if model_type == "transformer":
        steering = (
            -0.08 * features["weighted_heading_error"]
            -0.05 * features["current_lane_offset"]
            -0.015 * features["heading_trend"]
        )
        throttle = 0.39 - (0.06 * obstacle_term)
        uncertainty = 0.16 + abs(features["heading_trend"]) * 0.015
        adaptation = min(1.0, 0.5 + features["stability"] * 0.3)
        return steering, throttle, uncertainty, adaptation

    steering = (
        -0.07 * features["current_heading_error"]
        -0.07 * features["current_lane_offset"]
        -0.03 * features["heading_trend"]
        -0.05 * features["lane_trend"]
    )
    throttle = 0.3 + (features["stability"] * 0.1) - (0.05 * obstacle_term)
    uncertainty = max(0.04, 0.18 - (features["stability"] * 0.08) + obstacle_term * 0.03)
    adaptation = min(1.0, 0.55 + features["stability"] * 0.4)
    return steering, throttle, uncertainty, adaptation
