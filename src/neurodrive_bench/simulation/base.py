from __future__ import annotations

from abc import ABC, abstractmethod

from neurodrive_bench.contracts import EpisodeTrace, StressProfile
from neurodrive_bench.models.base import TemporalDrivingModel


class SimulationBackend(ABC):
    @abstractmethod
    def setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def teardown(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def simulate_episode(
        self,
        model: TemporalDrivingModel,
        stress_level: float,
        stress: StressProfile,
    ) -> EpisodeTrace:
        raise NotImplementedError
