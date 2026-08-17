from __future__ import annotations

from neurodrive_bench.contracts import EpisodeMetrics, EpisodeTrace


def build_metrics_from_trace(trace: EpisodeTrace) -> EpisodeMetrics:
    sample_count = max(1, len(trace.samples))

    lane_departures = [event for event in trace.events if event.event_type == "lane_departure"]
    collision_events = [
        event for event in trace.events if event.event_type in {"collision_risk", "near_collision"}
    ]
    sensor_events = [event for event in trace.events if event.event_type == "sensor_dropout_burst"]
    instability_events = [event for event in trace.events if event.event_type == "control_instability"]

    steering_values = [
        float(sample.get("control", {}).get("steering", 0.0))
        for sample in trace.samples
    ]
    throttle_values = [
        float(sample.get("control", {}).get("throttle", 0.0))
        for sample in trace.samples
    ]
    lane_offsets = [
        float(sample.get("telemetry", {}).get("lane_offset", 0.0))
        for sample in trace.samples
    ]
    heading_errors = [
        float(sample.get("telemetry", {}).get("heading_error", 0.0))
        for sample in trace.samples
    ]
    obstacle_distances = [
        float(sample.get("telemetry", {}).get("obstacle_distance", 100.0))
        for sample in trace.samples
    ]
    adaptation_levels = [
        float(sample.get("adaptation_level", 0.0))
        for sample in trace.samples
    ]
    uncertainty_levels = [
        float(sample.get("uncertainty_score", 0.0))
        for sample in trace.samples
    ]

    collision_rate = min(1.0, _event_severity_sum(collision_events) / sample_count)
    offroad_frequency = min(1.0, _event_severity_sum(lane_departures) / sample_count)
    rule_violations = min(
        1.0,
        (
            _event_severity_sum(lane_departures)
            + _event_severity_sum(sensor_events)
            + (_event_severity_sum(instability_events) * 0.5)
        )
        / sample_count,
    )
    steering_oscillation = _mean_absolute_delta(steering_values)
    lane_oscillation = _mean_absolute_delta(lane_offsets)
    heading_oscillation = _mean_absolute_delta(heading_errors)
    trajectory_smoothness = max(0.0, 1.0 - min(1.0, (lane_oscillation * 0.5) + (heading_oscillation * 0.05)))
    control_jitter = (
        (_mean_absolute_delta(steering_values) * 0.7)
        + (_mean_absolute_delta(throttle_values) * 0.3)
    )
    recovery_time = _estimate_recovery_time(trace)
    stabilization_speed = _estimate_stabilization_speed(adaptation_levels, heading_errors)
    
    mean_uncertainty = sum(uncertainty_levels) / len(uncertainty_levels) if uncertainty_levels else 0.0
    mean_adaptation = sum(adaptation_levels) / len(adaptation_levels) if adaptation_levels else 0.0
    adaptation_latency = _estimate_adaptation_latency(trace, adaptation_levels)

    if obstacle_distances:
        collision_rate = max(collision_rate, min(1.0, _count_below(obstacle_distances, 6.0) / sample_count))
    if lane_offsets:
        offroad_frequency = max(
            offroad_frequency,
            min(1.0, _count_below_abs(lane_offsets, minimum=1.2) / sample_count),
        )
    if uncertainty_levels:
        rule_violations = min(
            1.0,
            rule_violations + (sum(uncertainty_levels) / len(uncertainty_levels) * 0.15),
        )

    return EpisodeMetrics(
        collision_rate=collision_rate,
        offroad_frequency=offroad_frequency,
        rule_violations=rule_violations,
        steering_oscillation=min(1.0, steering_oscillation),
        trajectory_smoothness=min(1.0, trajectory_smoothness),
        control_jitter=min(1.0, control_jitter),
        recovery_time=min(1.0, recovery_time),
        stabilization_speed=min(1.0, stabilization_speed),
        mean_uncertainty=min(1.0, mean_uncertainty),
        mean_adaptation=min(1.0, mean_adaptation),
        adaptation_latency=min(1.0, adaptation_latency),
        graceful_degradation_index=0.0,
    )


def _mean_absolute_delta(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    deltas = [abs(current - previous) for previous, current in zip(values, values[1:])]
    return sum(deltas) / len(deltas)


def _estimate_recovery_time(trace: EpisodeTrace) -> float:
    if not trace.events or not trace.samples:
        return 0.0

    disturbance_steps = [
        event.step
        for event in trace.events
        if event.event_type in {"collision_risk", "near_collision", "lane_departure", "control_instability"}
    ]
    if not disturbance_steps:
        return 0.0

    final_step = max(int(sample.get("step", 0)) for sample in trace.samples)
    last_disturbance = max(disturbance_steps)
    if final_step <= 0:
        return 0.0
    return min(1.0, max(0.0, (final_step - last_disturbance) / final_step))


def _estimate_stabilization_speed(adaptation_levels: list[float], heading_errors: list[float]) -> float:
    if not adaptation_levels:
        return 0.0

    adaptation_score = sum(adaptation_levels) / len(adaptation_levels)
    heading_penalty = min(1.0, sum(abs(value) for value in heading_errors) / max(1, len(heading_errors) * 10.0))
    return max(0.0, adaptation_score * (1.0 - heading_penalty))


def _event_severity_sum(events: list[object]) -> float:
    return sum(float(getattr(event, "severity", 0.0)) for event in events)


def _count_below(values: list[float], threshold: float) -> int:
    return sum(1 for value in values if value < threshold)


def _count_below_abs(values: list[float], minimum: float) -> int:
    return sum(1 for value in values if abs(value) > minimum)


def _estimate_adaptation_latency(trace: EpisodeTrace, adaptation_levels: list[float]) -> float:
    if not trace.events or not adaptation_levels:
        return 0.0

    disturbance_steps = [
        event.step
        for event in trace.events
        if event.event_type in {"collision_risk", "near_collision", "lane_departure"}
    ]
    if not disturbance_steps:
        return 0.0

    # For each disturbance, how many steps until adaptation > 0.5?
    # We measure latency as a ratio of the episode length (0 = instant, 1 = never)
    total_latency = 0.0
    total_events = 0
    
    for start_step in disturbance_steps:
        if start_step >= len(adaptation_levels):
            continue
            
        latency = len(adaptation_levels) - start_step  # Default max latency
        for i in range(start_step, len(adaptation_levels)):
            if adaptation_levels[i] > 0.5:
                latency = i - start_step
                break
                
        total_latency += latency
        total_events += 1
        
    if total_events == 0:
        return 0.0
        
    avg_latency_steps = total_latency / total_events
    max_steps = len(adaptation_levels)
    return min(1.0, avg_latency_steps / max(1, (max_steps * 0.1)))  # Normalize against 10% of episode
