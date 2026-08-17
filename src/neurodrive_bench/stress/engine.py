from __future__ import annotations

from neurodrive_bench.contracts import StressProfile


def scale_stress(profile: StressProfile, level: float) -> StressProfile:
    return StressProfile(
        rain=profile.rain * level,
        fog=profile.fog * level,
        night=profile.night if level > 0 else False,
        noise_std=profile.noise_std * level,
        packet_dropout=profile.packet_dropout * level,
        latency_steps=max(0, round(profile.latency_steps * level)),
        sudden_obstacle_probability=profile.sudden_obstacle_probability * level,
        lane_blockage_probability=profile.lane_blockage_probability * level,
        traffic_multiplier=1.0 + ((profile.traffic_multiplier - 1.0) * level),
    )
