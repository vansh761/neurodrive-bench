from __future__ import annotations

from neurodrive_bench.config import BenchmarkConfig
from neurodrive_bench.simulation.base import SimulationBackend
from neurodrive_bench.simulation.carla_backend import CarlaSimulationBackend
from neurodrive_bench.simulation.stub_backend import StubSimulationBackend


def build_simulation_backend(config: BenchmarkConfig) -> SimulationBackend:
    if config.simulation_backend == "stub":
        return StubSimulationBackend(config)
    if config.simulation_backend == "carla":
        return CarlaSimulationBackend(config)
    raise ValueError(f"Unsupported simulation backend: {config.simulation_backend}")
