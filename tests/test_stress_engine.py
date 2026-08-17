from neurodrive_bench.contracts import StressProfile
from neurodrive_bench.stress.engine import scale_stress


BASE = StressProfile(
    rain=0.8,
    fog=0.6,
    night=True,
    noise_std=0.4,
    packet_dropout=0.3,
    latency_steps=10,
    sudden_obstacle_probability=0.5,
    lane_blockage_probability=0.2,
    traffic_multiplier=2.0,
)


def test_level_zero_gives_no_stress():
    scaled = scale_stress(BASE, 0.0)
    assert scaled.rain == 0.0
    assert scaled.fog == 0.0
    assert scaled.noise_std == 0.0
    assert scaled.packet_dropout == 0.0
    assert scaled.latency_steps == 0
    assert scaled.sudden_obstacle_probability == 0.0
    assert scaled.lane_blockage_probability == 0.0
    # traffic_multiplier is defined as 1.0 + delta * level, so level 0 -> 1.0 (no slowdown)
    assert scaled.traffic_multiplier == 1.0
    # night is a boolean flag that should be off when there's no stress at all
    assert scaled.night is False


def test_level_one_reproduces_base_profile():
    scaled = scale_stress(BASE, 1.0)
    assert scaled.rain == BASE.rain
    assert scaled.fog == BASE.fog
    assert scaled.noise_std == BASE.noise_std
    assert scaled.packet_dropout == BASE.packet_dropout
    assert scaled.latency_steps == BASE.latency_steps
    assert scaled.sudden_obstacle_probability == BASE.sudden_obstacle_probability
    assert scaled.lane_blockage_probability == BASE.lane_blockage_probability
    assert scaled.traffic_multiplier == BASE.traffic_multiplier
    assert scaled.night is True


def test_scaling_is_monotonic_in_level():
    lo = scale_stress(BASE, 0.25)
    hi = scale_stress(BASE, 0.75)
    assert hi.rain > lo.rain
    assert hi.fog > lo.fog
    assert hi.noise_std > lo.noise_std
    assert hi.packet_dropout > lo.packet_dropout
    assert hi.sudden_obstacle_probability > lo.sudden_obstacle_probability
    assert hi.traffic_multiplier > lo.traffic_multiplier


def test_latency_steps_never_negative_and_rounds_to_int():
    scaled = scale_stress(BASE, 0.03)  # 10 * 0.03 = 0.3 -> rounds to 0
    assert isinstance(scaled.latency_steps, int)
    assert scaled.latency_steps >= 0


def test_zero_base_profile_stays_zero_at_any_level():
    empty = StressProfile()
    for level in (0.0, 0.5, 1.0, 2.0):
        scaled = scale_stress(empty, level)
        assert scaled.rain == 0.0
        assert scaled.noise_std == 0.0
        assert scaled.latency_steps == 0
