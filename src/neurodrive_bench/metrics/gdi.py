from __future__ import annotations

from neurodrive_bench.contracts import EpisodeMetrics


def compute_gdi(metrics: EpisodeMetrics) -> float:
    """
    Research-oriented GDI balancing safety, stability, and neural adaptation.
    Penalizes false uncertainty (high uncertainty, low adaptation) and rewards 
    rapid adaptation during disturbances.
    """
    safety_score = (
        (1.0 - metrics.collision_rate) * 0.35
        + (1.0 - metrics.offroad_frequency) * 0.2
        + (1.0 - metrics.rule_violations) * 0.1
    )
    
    stability_score = (
        metrics.trajectory_smoothness * 0.10
        + (1.0 - metrics.steering_oscillation) * 0.05
    )
    
    # False panic penalty: high uncertainty but low adaptation
    false_panic_penalty = max(0.0, metrics.mean_uncertainty - metrics.mean_adaptation) * 0.05
    
    # Adaptation score: high adaptation and fast response
    adaptation_score = (
        metrics.mean_adaptation * 0.10
        + (1.0 - metrics.adaptation_latency) * 0.05
        + metrics.stabilization_speed * 0.05
    )
    
    score = safety_score + stability_score + adaptation_score - false_panic_penalty
    return max(0.0, min(1.0, score))
