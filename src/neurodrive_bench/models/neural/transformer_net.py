from __future__ import annotations

import math
import torch
import torch.nn as nn

from neurodrive_bench.models.neural.common import ControlHead, NeuralDrivingModel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerNetwork(nn.Module):
    # d_model=78, nhead=2 brings this network's parameter count (~202k) in line
    # with LSTMNetwork(hidden_size=116) and LiquidNetwork(hidden_size=218), both
    # ~200k. See models/neural/param_budget.py for the search that produced this.
    def __init__(self, input_size: int = 7, d_model: int = 78, nhead: int = 2, num_layers: int = 3) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            batch_first=True,
            dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = ControlHead(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, 7)
        """
        projected = self.input_proj(x)
        encoded = self.pos_encoding(projected)
        
        # Causal mask to prevent attending to future steps
        seq_len = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        out = self.encoder(encoded, mask=mask, is_causal=True)
        # Mean pooling over the sequence
        pooled = out.mean(dim=1)
        return self.head(pooled)


class TransformerDrivingModel(NeuralDrivingModel):
    def _build_network(self) -> nn.Module:
        return TransformerNetwork()
