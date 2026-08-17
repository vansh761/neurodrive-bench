from __future__ import annotations

from neurodrive_bench.contracts import TelemetryFrame


STATE_VECTOR_FIELDS = (
    "speed",
    "lane_offset",
    "yaw",
    "heading_error",
    "obstacle_distance",
    "acceleration",
    "delta_t",
)


def to_state_vector(frame: TelemetryFrame) -> list[float]:
    return [
        frame.speed,
        frame.lane_offset,
        frame.yaw,
        frame.heading_error,
        frame.obstacle_distance,
        frame.acceleration,
        frame.delta_t,
    ]
