from neurodrive_bench.reporting.summary import build_benchmark_summary


def make_aggregate(model_name: str, stress_level: float, gdi_mean: float, collision_mean: float) -> dict:
    def stat(v: float) -> dict:
        return {"mean": v, "std": 0.0, "min": v, "max": v, "n": 3.0}

    return {
        "model_name": model_name,
        "stress_level": stress_level,
        "episode_count": 3,
        "seeds": [1, 2, 3],
        "metrics_aggregate": {
            "graceful_degradation_index": stat(gdi_mean),
            "collision_rate": stat(collision_mean),
            "stabilization_speed": stat(0.5),
            "mean_uncertainty": stat(0.1),
            "mean_adaptation": stat(0.5),
            "adaptation_latency": stat(0.1),
        },
    }


def test_perfectly_consistent_gdi_scores_near_plus_one():
    # GDI falls monotonically as collision rate rises -> should be maximally consistent
    artifacts = [
        make_aggregate("m", 0.0, gdi_mean=0.95, collision_mean=0.0),
        make_aggregate("m", 0.25, gdi_mean=0.85, collision_mean=0.1),
        make_aggregate("m", 0.5, gdi_mean=0.75, collision_mean=0.3),
        make_aggregate("m", 0.75, gdi_mean=0.65, collision_mean=0.5),
        make_aggregate("m", 1.0, gdi_mean=0.55, collision_mean=0.7),
    ]
    summary = build_benchmark_summary(artifacts, "test")
    consistency = summary["models"]["m"]["aggregate"]["gdi_collision_rank_consistency"]
    assert consistency > 0.99


def test_inverted_relationship_scores_near_minus_one():
    # GDI rises as collision rate rises -> inconsistent / suspicious metric behavior
    artifacts = [
        make_aggregate("m", 0.0, gdi_mean=0.5, collision_mean=0.0),
        make_aggregate("m", 0.25, gdi_mean=0.6, collision_mean=0.1),
        make_aggregate("m", 0.5, gdi_mean=0.7, collision_mean=0.3),
        make_aggregate("m", 0.75, gdi_mean=0.8, collision_mean=0.5),
        make_aggregate("m", 1.0, gdi_mean=0.9, collision_mean=0.7),
    ]
    summary = build_benchmark_summary(artifacts, "test")
    consistency = summary["models"]["m"]["aggregate"]["gdi_collision_rank_consistency"]
    assert consistency < -0.99


def test_single_point_curve_returns_zero_not_error():
    artifacts = [make_aggregate("m", 0.0, gdi_mean=0.9, collision_mean=0.0)]
    summary = build_benchmark_summary(artifacts, "test")
    consistency = summary["models"]["m"]["aggregate"]["gdi_collision_rank_consistency"]
    assert consistency == 0.0
