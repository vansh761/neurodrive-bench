"""
Regression test for benchmark fairness: the three architectures compared in
configs/benchmark.example.yaml must have roughly equal parameter counts, or a
robustness/GDI gap between them could just be a capacity gap in disguise.

This previously silently broke (240k / 407k / 83k, a ~5x spread) because
`parameter_budget` in the YAML config was never actually read anywhere in the
code -- see models/neural/param_budget.py for the fix and the search that
produced the current default hidden sizes.
"""
from neurodrive_bench.models.neural.lstm_net import LSTMNetwork
from neurodrive_bench.models.neural.transformer_net import TransformerNetwork
from neurodrive_bench.models.neural.lnn_net import LiquidNetwork

MAX_RELATIVE_SPREAD = 0.05  # allow up to 5% spread between the largest and smallest


def _count(module) -> int:
    return sum(p.numel() for p in module.parameters())


def test_default_architectures_have_matched_parameter_counts():
    counts = {
        "lstm": _count(LSTMNetwork()),
        "transformer": _count(TransformerNetwork()),
        "lnn": _count(LiquidNetwork()),
    }
    smallest = min(counts.values())
    largest = max(counts.values())
    relative_spread = (largest - smallest) / smallest

    assert relative_spread <= MAX_RELATIVE_SPREAD, (
        f"Parameter counts are not fairly matched: {counts} "
        f"(spread {relative_spread:.1%}, budget is {MAX_RELATIVE_SPREAD:.0%})"
    )


def test_each_architecture_is_reasonably_close_to_200k_target():
    target = 200_000
    for name, module in [
        ("lstm", LSTMNetwork()),
        ("transformer", TransformerNetwork()),
        ("lnn", LiquidNetwork()),
    ]:
        n = _count(module)
        assert abs(n - target) / target <= 0.05, f"{name} has {n:,} params, target is {target:,}"
