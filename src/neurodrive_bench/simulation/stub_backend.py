from __future__ import annotations

from dataclasses import asdict
import random

from neurodrive_bench.config import BenchmarkConfig
from neurodrive_bench.contracts import EpisodeEvent, EpisodeTrace, StressProfile, TelemetryFrame
from neurodrive_bench.models.base import TemporalDrivingModel
from neurodrive_bench.simulation.base import SimulationBackend
from neurodrive_bench.simulation.scenario import build_scenario_plan


class StubSimulationBackend(SimulationBackend):
    """Placeholder backend until CARLA integration is connected."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    def setup(self) -> None:
        self._current_stress_level = 0.0
        self._current_stress = StressProfile()
        self._episode_steps = 0
        self._speed = 8.0
        self._lane_offset = 0.0
        self._heading_error = 0.0
        self._obstacle_distance = 35.0
        self._prior_speed = 8.0

    def teardown(self) -> None:
        return None

    def _read_telemetry(self) -> TelemetryFrame:
        acceleration = (self._speed - self._prior_speed) / self.config.fixed_delta_seconds
        self._prior_speed = self._speed
        return TelemetryFrame(
            speed=self._speed,
            lane_offset=self._lane_offset,
            yaw=self._heading_error * 0.8,
            heading_error=self._heading_error,
            obstacle_distance=self._obstacle_distance,
            acceleration=acceleration,
            delta_t=self.config.fixed_delta_seconds,
        )

    def _apply_control(self, output) -> None:
        pass

    def _tick_world(self) -> None:
        self._speed = max(0.0, self._speed + random.uniform(-0.4, 0.6) - (self._current_stress.rain * 0.15))
        self._lane_offset += (
            random.uniform(-0.02, 0.02)
            + (self._current_stress.noise_std * random.uniform(-1.0, 1.0))
            + (self._current_stress_level * 0.003)
        )
        self._heading_error += (
            random.uniform(-0.8, 0.8)
            + (self._current_stress.fog * random.uniform(-2.0, 2.0))
            + (self._current_stress_level * 0.05)
        )
        self._obstacle_distance = max(
            0.0,
            self._obstacle_distance
            - random.uniform(0.0, 0.5)
            - (self._current_stress.sudden_obstacle_probability * random.uniform(0.0, 1.5)),
        )
        self._episode_steps += 1

    def simulate_episode(
        self,
        model: TemporalDrivingModel,
        stress_level: float,
        stress: StressProfile,
    ) -> EpisodeTrace:
        scenario_plan = build_scenario_plan(stress)
        trace = EpisodeTrace(
            backend="stub",
            model_name=model.name,
            stress_level=stress_level,
            total_steps=self.config.max_episode_steps,
            metadata={
                "scenario_vehicle_count": scenario_plan.traffic_actor_count,
                "lane_blockage_active": scenario_plan.lane_blockage_active,
                "sudden_obstacle_active": scenario_plan.sudden_obstacle_active,
                "target_obstacle_distance": scenario_plan.target_obstacle_distance,
                "scenario_notes": scenario_plan.notes,
                "sensor_mode": "synthetic",
            },
        )
        model.reset()

        speed = 8.0
        lane_offset = 0.0
        heading_error = 0.0
        obstacle_distance = max(8.0, 35.0 - (stress_level * 20.0))
        prior_speed = speed
        prior_steering = 0.0
        prior_throttle = 0.4

        for step in range(self.config.max_episode_steps):
            speed = max(0.0, speed + random.uniform(-0.4, 0.6) - (stress.rain * 0.15))
            lane_offset += (
                random.uniform(-0.02, 0.02)
                + (stress.noise_std * random.uniform(-1.0, 1.0))
                + (stress_level * 0.003)
                + (0.01 if scenario_plan.lane_blockage_active and step > (self.config.max_episode_steps * 0.6) else 0.0)
            )
            heading_error += (
                random.uniform(-0.8, 0.8)
                + (stress.fog * random.uniform(-2.0, 2.0))
                + (stress_level * 0.05)
            )
            obstacle_distance = max(
                0.0,
                obstacle_distance
                - random.uniform(0.0, 0.5)
                - (stress.sudden_obstacle_probability * random.uniform(0.0, 1.5)),
            )
            if scenario_plan.sudden_obstacle_active and step == round(self.config.max_episode_steps * 0.65):
                obstacle_distance = min(obstacle_distance, scenario_plan.target_obstacle_distance)
            acceleration = (speed - prior_speed) / self.config.fixed_delta_seconds
            prior_speed = speed

            frame = TelemetryFrame(
                speed=speed,
                lane_offset=lane_offset,
                yaw=heading_error * 0.8,
                heading_error=heading_error,
                obstacle_distance=obstacle_distance,
                acceleration=acceleration,
                delta_t=self.config.fixed_delta_seconds,
            )
            output = model.predict(frame)
            steering_delta = abs(output.command.steering - prior_steering)
            throttle_delta = abs(output.command.throttle - prior_throttle)
            prior_steering = output.command.steering
            prior_throttle = output.command.throttle

            if step % self.config.sample_stride == 0:
                trace.samples.append(
                    {
                        "step": step,
                        "telemetry": asdict(frame),
                        "control": asdict(output.command),
                        "uncertainty_score": output.uncertainty_score,
                        "adaptation_level": output.adaptation_level,
                        "model_debug": output.debug,
                        "control_delta": {
                            "steering_delta": steering_delta,
                            "throttle_delta": throttle_delta,
                        },
                    }
                )

            event = self._maybe_generate_event(
                step=step,
                stress_level=stress_level,
                stress=stress,
                frame=frame,
                steering_delta=steering_delta,
                throttle_delta=throttle_delta,
            )
            if event is not None:
                trace.events.append(event)

        return trace

    def _maybe_generate_event(
        self,
        step: int,
        stress_level: float,
        stress: StressProfile,
        frame: TelemetryFrame,
        steering_delta: float,
        throttle_delta: float,
    ) -> EpisodeEvent | None:
        collision_threshold = 0.01 + (stress_level * 0.03)
        offroad_threshold = 1.2
        dropout_threshold = min(0.5, stress.packet_dropout)

        roll = random.random()
        if frame.obstacle_distance < 2.5 and roll < collision_threshold:
            return EpisodeEvent(
                step=step,
                event_type="collision_risk",
                severity=min(1.0, 1.0 - (frame.obstacle_distance / 4.0)),
                details={"obstacle_distance": frame.obstacle_distance},
            )
        if frame.obstacle_distance < 6.0 and roll < (collision_threshold * 1.5):
            return EpisodeEvent(
                step=step,
                event_type="near_collision",
                severity=min(1.0, 1.0 - (frame.obstacle_distance / 8.0)),
                details={"obstacle_distance": frame.obstacle_distance},
            )
        if abs(frame.lane_offset) > offroad_threshold:
            return EpisodeEvent(
                step=step,
                event_type="lane_departure",
                severity=min(1.0, abs(frame.lane_offset) / 2.5),
                details={"lane_offset": frame.lane_offset},
            )
        if steering_delta > 0.12 or throttle_delta > 0.08:
            return EpisodeEvent(
                step=step,
                event_type="control_instability",
                severity=min(1.0, (steering_delta * 2.5) + throttle_delta),
                details={
                    "steering_delta": steering_delta,
                    "throttle_delta": throttle_delta,
                },
            )
        if roll < dropout_threshold * 0.05:
            return EpisodeEvent(
                step=step,
                event_type="sensor_dropout_burst",
                severity=min(1.0, stress.packet_dropout + stress.noise_std),
                details={"packet_dropout": stress.packet_dropout, "noise_std": stress.noise_std},
            )
        return None
