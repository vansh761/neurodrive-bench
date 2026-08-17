from neurodrive_bench.models.neural.dataset import _auxiliary_targets


def make_window(heading_errors: list[float], lane_offsets: list[float], stress_level: float = 0.0) -> list[dict]:
    assert len(heading_errors) == len(lane_offsets)
    window = []
    for i, (he, lo) in enumerate(zip(heading_errors, lane_offsets)):
        window.append({
            "step": i,
            "heading_error": he,
            "lane_offset": lo,
            "stress_level": stress_level,
        })
    return window


def test_outputs_are_within_unit_interval():
    window = make_window(
        heading_errors=[0.0, 0.5, 1.0, 2.0],
        lane_offsets=[0.0, 0.8, 1.5, 3.0],
        stress_level=1.0,
    )
    uncertainty, adaptation = _auxiliary_targets(window, window[-1])
    assert 0.0 <= uncertainty <= 1.0
    assert 0.0 <= adaptation <= 1.0


def test_stable_trajectory_gets_high_adaptation_score():
    stable = make_window(
        heading_errors=[0.1, 0.1, 0.1, 0.1],
        lane_offsets=[0.05, 0.05, 0.05, 0.05],
    )
    _, adaptation = _auxiliary_targets(stable, stable[-1])
    assert adaptation > 0.9


def test_diverging_trajectory_gets_lower_adaptation_score_than_stable():
    stable = make_window(
        heading_errors=[0.1, 0.1, 0.1, 0.1],
        lane_offsets=[0.05, 0.05, 0.05, 0.05],
    )
    diverging = make_window(
        heading_errors=[0.1, 0.5, 1.2, 2.5],
        lane_offsets=[0.05, 0.4, 1.0, 2.0],
    )
    _, stable_adaptation = _auxiliary_targets(stable, stable[-1])
    _, diverging_adaptation = _auxiliary_targets(diverging, diverging[-1])
    assert diverging_adaptation < stable_adaptation


def test_higher_stress_level_increases_uncertainty_target():
    low_stress = make_window([0.1] * 4, [0.05] * 4, stress_level=0.0)
    high_stress = make_window([0.1] * 4, [0.05] * 4, stress_level=1.0)
    unc_low, _ = _auxiliary_targets(low_stress, low_stress[-1])
    unc_high, _ = _auxiliary_targets(high_stress, high_stress[-1])
    assert unc_high > unc_low


def test_missing_stress_level_defaults_gracefully():
    window = make_window([0.1] * 4, [0.05] * 4)
    target_record = dict(window[-1])
    del target_record["stress_level"]
    window[-1] = target_record
    # Should not raise, and should fall back to treating missing stress as 0
    uncertainty, adaptation = _auxiliary_targets(window, target_record)
    assert 0.0 <= uncertainty <= 1.0
    assert 0.0 <= adaptation <= 1.0
