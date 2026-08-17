from __future__ import annotations

import math


def mean(values: list[float]) -> float:
    return 0.0 if not values else sum(values) / len(values)


def stdev(values: list[float]) -> float:
    """Population standard deviation. Returns 0.0 for n<2 rather than raising,
    since a single-episode run (episodes_per_level=1) is a valid, if statistically
    weak, configuration and callers shouldn't have to special-case it."""
    n = len(values)
    if n < 2:
        return 0.0
    m = mean(values)
    variance = sum((v - m) ** 2 for v in values) / n
    return math.sqrt(variance)


def aggregate(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "std": stdev(values),
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
        "n": float(len(values)),
    }
