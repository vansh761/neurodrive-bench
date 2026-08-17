"""
Parameter-budget search for the three benchmarked architectures.

Why this exists: benchmark fairness requires that a robustness/GDI gap between
models reflects an architectural difference, not one model simply having more
learnable capacity than another. The original defaults (hidden_size=128 /
d_model=128 for all three) looked matched on paper but were not -- LSTM came
out to 240k params, Transformer to 407k, and the LNN to 83k, a ~5x spread.

This script searches each architecture's size knob for the value that lands
closest to a shared 200,000-parameter target, holding everything else (layer
count, feedforward width, etc.) fixed at its original default. Run it directly
to reproduce the defaults now hardcoded in lstm_net.py / transformer_net.py /
lnn_net.py:

    python -m neurodrive_bench.models.neural.param_budget

If you change num_layers, feedforward width, or add new architectures, rerun
this and update the hardcoded defaults + the comments pointing back here.
"""
from __future__ import annotations

from neurodrive_bench.models.neural.lstm_net import LSTMNetwork
from neurodrive_bench.models.neural.transformer_net import TransformerNetwork
from neurodrive_bench.models.neural.lnn_net import LiquidNetwork

TARGET_PARAMS = 200_000


def _count(module) -> int:
    return sum(p.numel() for p in module.parameters())


def find_lstm_hidden_size(candidates: list[int]) -> tuple[int, int]:
    best = min(candidates, key=lambda hs: abs(_count(LSTMNetwork(hidden_size=hs)) - TARGET_PARAMS))
    return best, _count(LSTMNetwork(hidden_size=best))


def find_transformer_d_model(candidates: list[int], nhead: int) -> tuple[int, int]:
    valid = [dm for dm in candidates if dm % nhead == 0]
    best = min(valid, key=lambda dm: abs(_count(TransformerNetwork(d_model=dm, nhead=nhead)) - TARGET_PARAMS))
    return best, _count(TransformerNetwork(d_model=best, nhead=nhead))


def find_lnn_hidden_size(candidates: list[int]) -> tuple[int, int]:
    best = min(candidates, key=lambda hs: abs(_count(LiquidNetwork(hidden_size=hs)) - TARGET_PARAMS))
    return best, _count(LiquidNetwork(hidden_size=best))


if __name__ == "__main__":
    lstm_hs, lstm_n = find_lstm_hidden_size(list(range(96, 124, 2)))
    tf_dm, tf_n = find_transformer_d_model(list(range(64, 100, 2)), nhead=2)
    lnn_hs, lnn_n = find_lnn_hidden_size(list(range(180, 230)))

    print(f"LSTM        hidden_size={lstm_hs:>4d}  params={lstm_n:,}")
    print(f"Transformer d_model={tf_dm:>4d} nhead=2  params={tf_n:,}")
    print(f"LNN         hidden_size={lnn_hs:>4d}  params={lnn_n:,}")
    spread = max(lstm_n, tf_n, lnn_n) - min(lstm_n, tf_n, lnn_n)
    print(f"Max spread: {spread:,} params ({spread / TARGET_PARAMS:.1%} of target)")
