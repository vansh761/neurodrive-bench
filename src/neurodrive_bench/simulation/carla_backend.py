from __future__ import annotations

import math
import random
from dataclasses import asdict

from neurodrive_bench.config import BenchmarkConfig
from neurodrive_bench.contracts import EpisodeEvent, EpisodeTrace, StressProfile, TelemetryFrame
from neurodrive_bench.models.base import TemporalDrivingModel
from neurodrive_bench.simulation.base import SimulationBackend
from neurodrive_bench.simulation.scenario import ScenarioPlan, build_scenario_plan

try:
    import carla  # type: ignore
except ImportError:  # pragma: no cover
    carla = None


class CarlaSimulationBackend(SimulationBackend):
    """CARLA-ready backend scaffold with sync-mode guardrails."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.client = None
        self.world = None
        self.ego_vehicle = None
        self.collision_sensor = None
        self.original_settings = None
        self.spawned_actors: list[object] = []
        self._scenario_actors: list[object] = []
        self._collision_events: list[EpisodeEvent] = []
        self._sensor_dropout_steps: set[int] = set()
        self._previous_speed = 0.0

    def setup(self) -> None:
        if carla is None:
            raise RuntimeError(
                "CARLA Python API is not installed. Use backend=stub for scaffold runs "
                "or install CARLA to enable simulation."
            )

        self.client = carla.Client(self.config.carla_host, self.config.carla_port)
        self.client.set_timeout(self.config.carla_timeout_seconds)
        self.world = self.client.load_world(self.config.map_name)

        settings = self.world.get_settings()
        self.original_settings = settings
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.config.fixed_delta_seconds
        self.world.apply_settings(settings)

    def teardown(self) -> None:
        if self.world is None:
            return

        self._destroy_spawned_actors()

        if self.original_settings is not None:
            self.world.apply_settings(self.original_settings)
            self.original_settings = None
        else:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)

        self.world = None
        self.client = None

    def simulate_episode(
        self,
        model: TemporalDrivingModel,
        stress_level: float,
        stress: StressProfile,
    ) -> EpisodeTrace:
        if self.world is None:
            raise RuntimeError("CARLA backend not initialized. Call setup() before simulate_episode().")

        scenario_plan = build_scenario_plan(stress)
        self._destroy_spawned_actors()
        self._apply_weather(stress)
        self._spawn_ego_vehicle()
        self._attach_collision_sensor()
        self._spawn_scenario_actors(scenario_plan)
        model.reset()

        trace = EpisodeTrace(
            backend="carla",
            model_name=model.name,
            stress_level=stress_level,
            total_steps=self.config.max_episode_steps,
            metadata={
                "scenario_vehicle_count": len(self._scenario_actors),
                "lane_blockage_active": scenario_plan.lane_blockage_active,
                "sudden_obstacle_active": scenario_plan.sudden_obstacle_active,
                "target_obstacle_distance": scenario_plan.target_obstacle_distance,
                "scenario_notes": scenario_plan.notes,
                "sensor_mode": "carla",
            },
        )

        last_frame = None
        prior_steering = 0.0
        prior_throttle = 0.0
        for step in range(self.config.max_episode_steps):
            self.world.tick()
            frame = self._extract_telemetry_frame(stress)
            output = model.predict(frame)
            self._apply_control(output.command.steering, output.command.throttle, output.command.brake)
            last_frame = frame
            steering_delta = abs(output.command.steering - prior_steering)
            throttle_delta = abs(output.command.throttle - prior_throttle)
            prior_steering = output.command.steering
            prior_throttle = output.command.throttle

            trace.events.extend(self._drain_collision_events())
            derived_event = self._derive_runtime_event(
                step=step,
                stress=stress,
                frame=frame,
                steering_delta=steering_delta,
                throttle_delta=throttle_delta,
            )
            if derived_event is not None:
                trace.events.append(derived_event)

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

        if last_frame is not None:
            trace.samples.append(
                {
                    "step": self.config.max_episode_steps - 1,
                    "telemetry": asdict(last_frame),
                    "control": {
                        "steering": 0.0,
                        "throttle": 0.0,
                        "brake": 0.0,
                    },
                    "uncertainty_score": 0.0,
                    "adaptation_level": 0.0,
                    "model_debug": {"note": "final_frame_snapshot"},
                    "control_delta": {
                        "steering_delta": 0.0,
                        "throttle_delta": 0.0,
                    },
                }
            )

        return trace

    def _spawn_ego_vehicle(self) -> None:
        if self.world is None:
            raise RuntimeError("World is not available.")

        blueprint_library = self.world.get_blueprint_library()
        candidates = blueprint_library.filter(self.config.ego_vehicle_filter)
        if not candidates:
            raise RuntimeError(f"No CARLA vehicle matches filter: {self.config.ego_vehicle_filter}")

        blueprint = candidates[0]
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points are available in the CARLA map.")

        spawn_point = random.choice(spawn_points)
        self.ego_vehicle = self.world.try_spawn_actor(blueprint, spawn_point)
        if self.ego_vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle in CARLA world.")

        self.spawned_actors.append(self.ego_vehicle)
        self._previous_speed = 0.0

    def _attach_collision_sensor(self) -> None:
        if self.world is None or self.ego_vehicle is None or carla is None:
            return

        blueprint_library = self.world.get_blueprint_library()
        sensor_blueprints = blueprint_library.filter("sensor.other.collision")
        if not sensor_blueprints:
            return

        sensor_blueprint = sensor_blueprints[0]
        transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        self.collision_sensor = self.world.spawn_actor(sensor_blueprint, transform, attach_to=self.ego_vehicle)
        self.spawned_actors.append(self.collision_sensor)
        self.collision_sensor.listen(self._on_collision_event)

    def _spawn_scenario_actors(self, scenario_plan: ScenarioPlan) -> None:
        if self.world is None or self.ego_vehicle is None or carla is None:
            return
        if scenario_plan.traffic_actor_count <= 0 and not scenario_plan.lane_blockage_active and not scenario_plan.sudden_obstacle_active:
            return

        blueprint_library = self.world.get_blueprint_library()
        candidates = blueprint_library.filter(self.config.scenario_vehicle_filter)
        if not candidates:
            return

        ego_transform = self.ego_vehicle.get_transform()
        ego_location = ego_transform.location
        ego_forward_vector = ego_transform.get_forward_vector()
        ego_right_vector = ego_transform.get_right_vector()
        
        target_distance = scenario_plan.target_obstacle_distance

        # 1. Targeted Adversarial Spawning
        if scenario_plan.lane_blockage_active or scenario_plan.sudden_obstacle_active:
            # Calculate spawn point ahead of ego
            spawn_location = carla.Location(
                x=ego_location.x + (ego_forward_vector.x * target_distance),
                y=ego_location.y + (ego_forward_vector.y * target_distance),
                z=ego_location.z + 0.5
            )
            
            if scenario_plan.sudden_obstacle_active:
                # Offset to the right slightly so it can cut in
                spawn_location.x += ego_right_vector.x * 3.5
                spawn_location.y += ego_right_vector.y * 3.5
                
            spawn_transform = carla.Transform(spawn_location, ego_transform.rotation)
            adversary = self.world.try_spawn_actor(random.choice(candidates), spawn_transform)
            
            if adversary:
                self.spawned_actors.append(adversary)
                self._scenario_actors.append(adversary)
                
                if scenario_plan.sudden_obstacle_active:
                    # Give it an initial velocity cutting left across the ego's path
                    cut_in_velocity = carla.Vector3D(
                        x=(ego_forward_vector.x * 2.0) - (ego_right_vector.x * 5.0),
                        y=(ego_forward_vector.y * 2.0) - (ego_right_vector.y * 5.0),
                        z=0.0
                    )
                    adversary.set_target_velocity(cut_in_velocity)
                    
        # 2. General Background Traffic
        spawn_points = self.world.get_map().get_spawn_points()
        for spawn_point in spawn_points:
            if len(self._scenario_actors) >= scenario_plan.traffic_actor_count + 1:
                break
            
            # Keep traffic far away from the ego initially
            distance = math.sqrt(
                ((spawn_point.location.x - ego_location.x) ** 2)
                + ((spawn_point.location.y - ego_location.y) ** 2)
            )
            if distance < 25.0:
                continue
                
            actor = self.world.try_spawn_actor(random.choice(candidates), spawn_point)
            if actor:
                self.spawned_actors.append(actor)
                self._scenario_actors.append(actor)

    def _destroy_spawned_actors(self) -> None:
        for actor in reversed(self.spawned_actors):
            try:
                actor.destroy()
            except Exception:
                pass
        self.spawned_actors = []
        self._scenario_actors = []
        self.ego_vehicle = None
        self.collision_sensor = None
        self._collision_events = []
        self._sensor_dropout_steps = set()

    def _apply_weather(self, stress: StressProfile) -> None:
        if self.world is None or carla is None:
            return

        sun_altitude = -10.0 if stress.night else 45.0
        weather = carla.WeatherParameters(
            precipitation=max(0.0, min(100.0, stress.rain * 100.0)),
            fog_density=max(0.0, min(100.0, stress.fog * 100.0)),
            cloudiness=max(0.0, min(100.0, stress.rain * 70.0)),
            wetness=max(0.0, min(100.0, stress.rain * 100.0)),
            sun_altitude_angle=sun_altitude,
        )
        self.world.set_weather(weather)

    def _extract_telemetry_frame(self, stress: StressProfile) -> TelemetryFrame:
        if self.world is None or self.ego_vehicle is None:
            raise RuntimeError("Ego vehicle is not available for telemetry extraction.")

        transform = self.ego_vehicle.get_transform()
        velocity = self.ego_vehicle.get_velocity()
        speed = math.sqrt((velocity.x**2) + (velocity.y**2) + (velocity.z**2))
        noisy_speed = self._with_noise(speed, stress.noise_std)
        yaw = self._with_noise(transform.rotation.yaw, stress.noise_std * 10.0)
        lane_offset = self._estimate_lane_offset(transform.location)
        heading_error = self._compute_heading_error(transform)
        obstacle_distance = self._estimate_obstacle_distance(transform.location)
        acceleration = (noisy_speed - self._previous_speed) / self.config.fixed_delta_seconds
        self._previous_speed = noisy_speed
        dropout_applied = False

        def maybe_dropout(value: float) -> float:
            nonlocal dropout_applied
            output = self._maybe_dropout(value, stress.packet_dropout)
            if output == 0.0 and value != 0.0:
                dropout_applied = True
            return output

        frame = TelemetryFrame(
            speed=maybe_dropout(noisy_speed),
            lane_offset=maybe_dropout(lane_offset),
            yaw=maybe_dropout(yaw),
            heading_error=maybe_dropout(heading_error),
            obstacle_distance=maybe_dropout(obstacle_distance),
            acceleration=maybe_dropout(acceleration),
            delta_t=self.config.fixed_delta_seconds,
        )
        if dropout_applied:
            self._sensor_dropout_steps.add(len(self._sensor_dropout_steps) + len(self._collision_events))
        return frame

    def _compute_heading_error(self, transform: object) -> float:
        if self.world is None:
            return 0.0

        waypoint = self.world.get_map().get_waypoint(transform.location, project_to_road=True)
        if waypoint is None:
            return 0.0

        road_yaw = waypoint.transform.rotation.yaw
        return self._normalize_angle_degrees(transform.rotation.yaw - road_yaw)

    def _estimate_lane_offset(self, location: object) -> float:
        if self.world is None:
            return 0.0

        waypoint = self.world.get_map().get_waypoint(location, project_to_road=True)
        if waypoint is None:
            return 0.0

        dx = location.x - waypoint.transform.location.x
        dy = location.y - waypoint.transform.location.y
        right = waypoint.transform.get_right_vector()
        return (dx * right.x) + (dy * right.y)

    def _estimate_obstacle_distance(self, location: object) -> float:
        if self.world is None or self.ego_vehicle is None:
            return 100.0

        closest = 100.0
        for actor in self.world.get_actors().filter("vehicle.*"):
            if actor.id == self.ego_vehicle.id:
                continue

            other_location = actor.get_location()
            distance = math.sqrt(
                ((other_location.x - location.x) ** 2)
                + ((other_location.y - location.y) ** 2)
                + ((other_location.z - location.z) ** 2)
            )
            closest = min(closest, distance)
        return closest

    def _apply_control(self, steering: float, throttle: float, brake: float) -> None:
        if self.ego_vehicle is None or carla is None:
            return

        control = carla.VehicleControl(
            steer=max(-1.0, min(1.0, steering)),
            throttle=max(0.0, min(1.0, throttle)),
            brake=max(0.0, min(1.0, brake)),
        )
        self.ego_vehicle.apply_control(control)

    def _on_collision_event(self, event: object) -> None:
        intensity = math.sqrt(
            (event.normal_impulse.x**2) + (event.normal_impulse.y**2) + (event.normal_impulse.z**2)
        )
        other_actor = getattr(event, "other_actor", None)
        self._collision_events.append(
            EpisodeEvent(
                step=int(getattr(event, "frame", 0)),
                event_type="collision_risk",
                severity=min(1.0, intensity / 50.0),
                details={
                    "intensity": intensity,
                    "other_actor_type": None if other_actor is None else str(other_actor.type_id),
                },
            )
        )

    def _drain_collision_events(self) -> list[EpisodeEvent]:
        drained = list(self._collision_events)
        self._collision_events = []
        return drained

    def _derive_runtime_event(
        self,
        step: int,
        stress: StressProfile,
        frame: TelemetryFrame,
        steering_delta: float,
        throttle_delta: float,
    ) -> EpisodeEvent | None:
        if frame.obstacle_distance < 6.0:
            return EpisodeEvent(
                step=step,
                event_type="near_collision",
                severity=min(1.0, 1.0 - (frame.obstacle_distance / 8.0)),
                details={"obstacle_distance": frame.obstacle_distance},
            )
        if abs(frame.lane_offset) > 1.2:
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
        if stress.packet_dropout > 0.0 and random.random() < min(0.1, stress.packet_dropout):
            return EpisodeEvent(
                step=step,
                event_type="sensor_dropout_burst",
                severity=min(1.0, stress.packet_dropout + stress.noise_std),
                details={"packet_dropout": stress.packet_dropout, "noise_std": stress.noise_std},
            )
        return None

    @staticmethod
    def _normalize_angle_degrees(angle: float) -> float:
        wrapped = (angle + 180.0) % 360.0 - 180.0
        return wrapped

    @staticmethod
    def _with_noise(value: float, noise_std: float) -> float:
        if noise_std <= 0.0:
            return value
        return value + random.gauss(0.0, noise_std)

    @staticmethod
    def _maybe_dropout(value: float, dropout_probability: float) -> float:
        if dropout_probability <= 0.0:
            return value
        return 0.0 if random.random() < dropout_probability else value
