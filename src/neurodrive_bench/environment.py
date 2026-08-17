from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass(slots=True)
class EnvironmentReport:
    python_available: bool
    carla_installed: bool
    recommended_backend: str
    notes: list[str]


def inspect_environment() -> EnvironmentReport:
    carla_installed = importlib.util.find_spec("carla") is not None
    notes: list[str] = []

    if carla_installed:
        notes.append("CARLA Python API detected.")
        recommended_backend = "carla"
    else:
        notes.append("CARLA Python API not detected.")
        notes.append("Use backend=stub for local development and benchmarking scaffold runs.")
        notes.append("Install CARLA later when you are ready to validate live simulation episodes.")
        recommended_backend = "stub"

    return EnvironmentReport(
        python_available=True,
        carla_installed=carla_installed,
        recommended_backend=recommended_backend,
        notes=notes,
    )
