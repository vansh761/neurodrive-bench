from __future__ import annotations

from abc import ABC, abstractmethod

from neurodrive_bench.contracts import ModelOutput, TelemetryFrame
from neurodrive_bench.models.profiles import ModelProfile
from neurodrive_bench.telemetry.history import TelemetryHistoryBuffer


class TemporalDrivingModel(ABC):
    def __init__(self, name: str, history_window: int, profile: ModelProfile | None = None) -> None:
        self.name = name
        self.history = TelemetryHistoryBuffer(window_size=history_window)
        self.profile = profile

    def reset(self) -> None:
        self.history.clear()

    def predict(self, frame: TelemetryFrame) -> ModelOutput:
        self.history.add(frame)
        return self.predict_from_history(self.history)

    @abstractmethod
    def predict_from_history(self, history: TelemetryHistoryBuffer) -> ModelOutput:
        raise NotImplementedError

    @property
    def history_window(self) -> int:
        return self.history.window_size

    @property
    def latest_frame(self) -> TelemetryFrame:
        frames = self.history.frames()
        if not frames:
            raise RuntimeError("No telemetry frames are available in model history.")
        return frames[-1]

    @property
    def history_length(self) -> int:
        return len(self.history.frames())
