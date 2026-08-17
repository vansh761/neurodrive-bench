import pytest

from neurodrive_bench.contracts import TelemetryFrame
from neurodrive_bench.telemetry.history import TelemetryHistoryBuffer


def make_frame(speed: float) -> TelemetryFrame:
    return TelemetryFrame(
        speed=speed,
        lane_offset=0.1,
        yaw=0.2,
        heading_error=0.05,
        obstacle_distance=10.0,
        acceleration=0.0,
        delta_t=0.1,
    )


def test_rejects_non_positive_window_size():
    with pytest.raises(ValueError):
        TelemetryHistoryBuffer(0)
    with pytest.raises(ValueError):
        TelemetryHistoryBuffer(-5)


def test_empty_buffer_pads_to_full_window_of_zeros():
    buf = TelemetryHistoryBuffer(window_size=5)
    padded = buf.padded_state_vectors()
    assert len(padded) == 5
    assert all(v == [0.0] * 7 for v in padded)


def test_partial_buffer_pads_with_repeated_first_frame():
    buf = TelemetryHistoryBuffer(window_size=5)
    buf.add(make_frame(10.0))
    buf.add(make_frame(20.0))
    padded = buf.padded_state_vectors()

    assert len(padded) == 5
    real = buf.state_vectors()
    # First 3 entries are padding copies of the earliest real frame, not zeros -
    # this avoids feeding the network a fake "instant stop" at the start of an episode.
    assert padded[0] == real[0]
    assert padded[1] == real[0]
    assert padded[2] == real[0]
    # Last 2 entries are the real frames, in order
    assert padded[3] == real[0]
    assert padded[4] == real[1]


def test_full_buffer_returns_state_vectors_unpadded():
    buf = TelemetryHistoryBuffer(window_size=3)
    for s in (1.0, 2.0, 3.0):
        buf.add(make_frame(s))
    padded = buf.padded_state_vectors()
    assert padded == buf.state_vectors()
    assert len(padded) == 3


def test_buffer_drops_oldest_frame_beyond_window():
    buf = TelemetryHistoryBuffer(window_size=2)
    buf.add(make_frame(1.0))
    buf.add(make_frame(2.0))
    buf.add(make_frame(3.0))
    frames = buf.frames()
    assert len(frames) == 2
    assert frames[0].speed == 2.0
    assert frames[1].speed == 3.0


def test_clear_empties_buffer():
    buf = TelemetryHistoryBuffer(window_size=3)
    buf.add(make_frame(1.0))
    buf.clear()
    assert buf.frames() == []
    assert len(buf.padded_state_vectors()) == 3
