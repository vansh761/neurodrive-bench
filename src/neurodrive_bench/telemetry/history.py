from __future__ import annotations

from collections import deque

from neurodrive_bench.contracts import TelemetryFrame
from neurodrive_bench.telemetry.state_vector import to_state_vector


class TelemetryHistoryBuffer:
    def __init__(self, window_size: int) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        self.window_size = window_size
        self._frames: deque[TelemetryFrame] = deque(maxlen=window_size)

    def add(self, frame: TelemetryFrame) -> None:
        self._frames.append(frame)

    def clear(self) -> None:
        self._frames.clear()

    def frames(self) -> list[TelemetryFrame]:
        return list(self._frames)

    def state_vectors(self) -> list[list[float]]:
        return [to_state_vector(frame) for frame in self._frames]

    def padded_state_vectors(self) -> list[list[float]]:
        vectors = self.state_vectors()
        if not vectors:
            return [[0.0] * 7 for _ in range(self.window_size)]

        pad_count = self.window_size - len(vectors)
        if pad_count <= 0:
            return vectors

        padding = [vectors[0][:] for _ in range(pad_count)]
        return padding + vectors
