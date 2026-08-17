from __future__ import annotations

import torch
import torch.nn as nn

from neurodrive_bench.models.neural.common import ControlHead, NeuralDrivingModel


class LSTMNetwork(nn.Module):
    # hidden_size=116 is not an arbitrary choice: it's the value that brings this
    # network's parameter count (~201k) in line with TransformerNetwork(d_model=78)
    # and LiquidNetwork(hidden_size=218), both ~200k, so cross-architecture GDI
    # comparisons aren't confounded by one model simply having more capacity.
    # See models/neural/param_budget.py for the search that produced these numbers.
    def __init__(self, input_size: int = 7, hidden_size: int = 116, num_layers: int = 2) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_size, 64)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        self.head = ControlHead(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, 7)
        """
        projected = self.input_proj(x)
        lstm_out, _ = self.lstm(projected)
        # Take the output of the last sequence step
        final_state = lstm_out[:, -1, :]
        return self.head(final_state)


class LSTMDrivingModel(NeuralDrivingModel):
    def _build_network(self) -> nn.Module:
        return LSTMNetwork()
