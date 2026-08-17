from __future__ import annotations

from dataclasses import dataclass

from neurodrive_bench.contracts import StressProfile


@dataclass(slots=True)
class ScenarioPlan:
    traffic_actor_count: int
    lane_blockage_active: bool
    sudden_obstacle_active: bool
    target_obstacle_distance: float
    notes: list[str]


def build_scenario_plan(stress: StressProfile) -> ScenarioPlan:
    traffic_actor_count = max(0, round((stress.traffic_multiplier - 1.0) * 4.0))
    lane_blockage_active = stress.lane_blockage_probability >= 0.03
    sudden_obstacle_active = stress.sudden_obstacle_probability >= 0.05
    target_obstacle_distance = max(4.0, 18.0 - (stress.sudden_obstacle_probability * 40.0))

    notes: list[str] = []
    if traffic_actor_count > 0:
        notes.append(f"spawn_extra_traffic={traffic_actor_count}")
    if lane_blockage_active:
        notes.append("lane_blockage_requested")
    if sudden_obstacle_active:
        notes.append("sudden_obstacle_requested")

    return ScenarioPlan(
        traffic_actor_count=traffic_actor_count,
        lane_blockage_active=lane_blockage_active,
        sudden_obstacle_active=sudden_obstacle_active,
        target_obstacle_distance=target_obstacle_distance,
        notes=notes,
    )
