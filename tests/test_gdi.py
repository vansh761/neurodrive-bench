from neurodrive_bench.contracts import EpisodeMetrics
from neurodrive_bench.metrics.gdi import compute_gdi


def make_metrics(**overrides) -> EpisodeMetrics:
    base = dict(
        collision_rate=0.0,
        offroad_frequency=0.0,
        rule_violations=0.0,
        steering_oscillation=0.0,
        trajectory_smoothness=1.0,
        control_jitter=0.0,
        recovery_time=0.0,
        stabilization_speed=1.0,
        mean_uncertainty=0.0,
        mean_adaptation=1.0,
        adaptation_latency=0.0,
        graceful_degradation_index=0.0,  # unused by compute_gdi itself, present for the dataclass
    )
    base.update(overrides)
    return EpisodeMetrics(**base)


def test_perfect_episode_scores_near_maximum():
    metrics = make_metrics()
    gdi = compute_gdi(metrics)
    assert 0.95 <= gdi <= 1.0


def test_gdi_is_bounded_between_0_and_1_for_worst_case():
    worst = make_metrics(
        collision_rate=1.0,
        offroad_frequency=1.0,
        rule_violations=1.0,
        steering_oscillation=1.0,
        trajectory_smoothness=0.0,
        stabilization_speed=0.0,
        mean_uncertainty=1.0,
        mean_adaptation=0.0,
        adaptation_latency=1.0,
    )
    gdi = compute_gdi(worst)
    assert 0.0 <= gdi <= 1.0


def test_higher_collision_rate_strictly_lowers_gdi():
    safe = make_metrics(collision_rate=0.0)
    unsafe = make_metrics(collision_rate=0.5)
    assert compute_gdi(unsafe) < compute_gdi(safe)


def test_false_panic_is_penalized():
    """High uncertainty with low adaptation (panicking without recovering)
    should score worse than the same uncertainty paired with high adaptation."""
    panicking = make_metrics(mean_uncertainty=0.9, mean_adaptation=0.1)
    recovering = make_metrics(mean_uncertainty=0.9, mean_adaptation=0.9)
    assert compute_gdi(panicking) < compute_gdi(recovering)


def test_gdi_is_deterministic():
    metrics = make_metrics(collision_rate=0.2, mean_uncertainty=0.3, mean_adaptation=0.6)
    assert compute_gdi(metrics) == compute_gdi(metrics)
