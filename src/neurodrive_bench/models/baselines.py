from __future__ import annotations

from neurodrive_bench.contracts import ControlCommand, ModelOutput
from neurodrive_bench.models.base import TemporalDrivingModel
from neurodrive_bench.telemetry.history import TelemetryHistoryBuffer


class LSTMBaseline(TemporalDrivingModel):
    def predict_from_history(self, history: TelemetryHistoryBuffer) -> ModelOutput:
        frame = self.latest_frame
        frames = history.frames()
        mean_lane_offset = sum(item.lane_offset for item in frames) / len(frames)
        mean_heading_error = sum(item.heading_error for item in frames) / len(frames)
        steering = (-mean_lane_offset * 0.08) + (-mean_heading_error * 0.01)
        throttle = 0.4
        uncertainty = max(0.05, 0.3 - min(0.2, len(frames) * 0.01))
        adaptation = min(1.0, len(frames) / history.window_size)

        if self.profile is not None:
            features = {
                "mean_lane_offset": mean_lane_offset,
                "mean_heading_error": mean_heading_error,
                "current_lane_offset": frame.lane_offset,
                "current_heading_error": frame.heading_error,
                "obstacle_distance": frame.obstacle_distance,
            }
            steering = _apply_linear_profile(self.profile.steering_weights, features)
            throttle = _apply_linear_profile(self.profile.throttle_weights, features)
            uncertainty = min(1.0, max(0.0, self.profile.uncertainty_bias))
            adaptation = min(1.0, max(0.0, self.profile.adaptation_bias))

        return ModelOutput(
            command=ControlCommand(
                steering=steering,
                throttle=throttle,
                brake=0.0,
            ),
            uncertainty_score=uncertainty,
            adaptation_level=adaptation,
            debug={
                "history_length": len(frames),
                "controller": "lstm_temporal_stub",
                "profile_loaded": self.profile is not None,
            },
        )


class TransformerBaseline(TemporalDrivingModel):
    def predict_from_history(self, history: TelemetryHistoryBuffer) -> ModelOutput:
        frame = self.latest_frame
        frames = history.frames()
        weighted_frames = frames[-min(5, len(frames)) :]
        total_weight = sum(range(1, len(weighted_frames) + 1))
        weighted_heading = sum(
            item.heading_error * weight
            for item, weight in zip(weighted_frames, range(1, len(weighted_frames) + 1))
        ) / total_weight
        steering = (-weighted_heading * 0.09) + (-frame.lane_offset * 0.05)
        throttle = 0.38
        uncertainty = max(0.08, 0.28 - min(0.15, len(frames) * 0.008))
        adaptation = min(1.0, len(weighted_frames) / 5.0)

        if self.profile is not None:
            features = {
                "weighted_heading_error": weighted_heading,
                "current_lane_offset": frame.lane_offset,
                "current_heading_error": frame.heading_error,
                "mean_heading_error": sum(item.heading_error for item in frames) / len(frames),
                "obstacle_distance": frame.obstacle_distance,
            }
            steering = _apply_linear_profile(self.profile.steering_weights, features)
            throttle = _apply_linear_profile(self.profile.throttle_weights, features)
            uncertainty = min(1.0, max(0.0, self.profile.uncertainty_bias))
            adaptation = min(1.0, max(0.0, self.profile.adaptation_bias))

        return ModelOutput(
            command=ControlCommand(
                steering=steering,
                throttle=throttle,
                brake=0.0,
            ),
            uncertainty_score=uncertainty,
            adaptation_level=adaptation,
            debug={
                "history_length": len(frames),
                "controller": "transformer_temporal_stub",
                "profile_loaded": self.profile is not None,
            },
        )


class AdaptiveTemporalStub(TemporalDrivingModel):
    def predict_from_history(self, history: TelemetryHistoryBuffer) -> ModelOutput:
        frame = self.latest_frame
        vectors = history.padded_state_vectors()
        recent = vectors[-1]
        earlier = vectors[max(0, len(vectors) - 4)]
        heading_trend = recent[3] - earlier[3]
        lane_trend = recent[1] - earlier[1]
        stability = max(0.0, 1.0 - min(1.0, abs(heading_trend) * 0.1 + abs(lane_trend) * 0.2))
        steering = (
            (-frame.heading_error * 0.07)
            + (-frame.lane_offset * 0.07)
            + (-heading_trend * 0.03)
            + (-lane_trend * 0.05)
        )
        throttle = 0.32 + (stability * 0.08)
        uncertainty = max(0.04, 0.2 - (stability * 0.1))
        adaptation = stability

        if self.profile is not None:
            features = {
                "current_heading_error": frame.heading_error,
                "current_lane_offset": frame.lane_offset,
                "heading_trend": heading_trend,
                "lane_trend": lane_trend,
                "stability": stability,
                "obstacle_distance": frame.obstacle_distance,
            }
            steering = _apply_linear_profile(self.profile.steering_weights, features)
            throttle = _apply_linear_profile(self.profile.throttle_weights, features)
            uncertainty = min(1.0, max(0.0, self.profile.uncertainty_bias))
            adaptation = min(1.0, max(0.0, self.profile.adaptation_bias))

        return ModelOutput(
            command=ControlCommand(steering=steering, throttle=throttle, brake=0.0),
            uncertainty_score=uncertainty,
            adaptation_level=adaptation,
            debug={
                "history_length": len(history.frames()),
                "heading_trend": heading_trend,
                "lane_trend": lane_trend,
                "mode": "adaptive_temporal_stub",
                "profile_loaded": self.profile is not None,
            },
        )


def _apply_linear_profile(weights: dict[str, float], features: dict[str, float]) -> float:
    total = float(weights.get("bias", 0.0))
    for key, value in features.items():
        total += float(weights.get(key, 0.0)) * value
    return total
