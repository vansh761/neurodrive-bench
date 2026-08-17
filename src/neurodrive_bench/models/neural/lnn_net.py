from __future__ import annotations

import torch
import torch.nn as nn

from neurodrive_bench.contracts import TelemetryFrame, ModelOutput, ControlCommand
from neurodrive_bench.models.neural.common import ControlHead, NeuralDrivingModel
from neurodrive_bench.telemetry.history import TelemetryHistoryBuffer


class LiquidCell(nn.Module):
    """
    ODE-inspired continuous-time recurrent cell, based on Liquid Neural Networks (Hasani et al. 2021).
    Features input-dependent time constants and continuous decay.
    """
    def __init__(self, input_size: int, hidden_size: int, tau_min: float = 0.1, tau_max: float = 2.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.tau_min = tau_min
        self.tau_max = tau_max
        
        combined_size = input_size + hidden_size
        
        # Computes input-dependent time constant tau
        self.tau_net = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.Sigmoid()
        )
        
        # Input gate
        self.gate_net = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.Sigmoid()
        )
        
        # Candidate hidden state
        self.update_net = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, delta_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, input_size)
            h_prev: Previous hidden state of shape (batch_size, hidden_size)
            delta_t: Time delta tensor of shape (batch_size, 1)
        Returns:
            h_next: Next hidden state
            tau: Time constant used for this step (for interpretability)
        """
        combined = torch.cat([x, h_prev], dim=1)
        
        # 1. Compute time constant tau(t)
        tau_raw = self.tau_net(combined)
        tau = tau_raw * self.tau_max + self.tau_min
        
        # 2. Compute candidate state h_hat(t) and gate g(t)
        h_hat = self.update_net(combined)
        g = self.gate_net(combined)
        
        # 3. Continuous-time decay update
        # ODE approximation: h(t) = h(t-1) * exp(-dt/tau) + (1 - exp(-dt/tau)) * g * h_hat
        # Using a stable exponential approximation
        alpha = torch.exp(-delta_t / tau)
        h_next = alpha * h_prev + (1.0 - alpha) * g * h_hat
        
        return h_next, tau


class LiquidNetwork(nn.Module):
    # hidden_size=218 brings this network's parameter count (~200k) in line with
    # LSTMNetwork(hidden_size=116) and TransformerNetwork(d_model=78), both ~200k.
    # See models/neural/param_budget.py for the search that produced this.
    def __init__(self, input_size: int = 7, hidden_size: int = 218) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU()
        )
        self.cell = LiquidCell(64, hidden_size)
        self.head = ControlHead(hidden_size)

    def forward(self, x: torch.Tensor, delta_ts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, 7)
            delta_ts: Tensor of shape (batch_size, seq_len, 1)
        """
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        taus = []
        for t in range(seq_len):
            x_t = self.input_proj(x[:, t, :])
            dt_t = delta_ts[:, t, :]
            h, tau = self.cell(x_t, h, dt_t)
            taus.append(tau)
            
        out = self.head(h)
        # Average tau across the sequence for the batch
        avg_tau = torch.stack(taus, dim=1).mean(dim=[1, 2])
        return out, avg_tau


class LNNDrivingModel(NeuralDrivingModel):
    def __init__(self, name: str, history_window: int, checkpoint_path: str | None = None) -> None:
        super().__init__(name=name, history_window=history_window, checkpoint_path=checkpoint_path)
        self._last_tau = 0.0
        
    def _build_network(self) -> nn.Module:
        return LiquidNetwork()
        
    @torch.no_grad()
    def predict_from_history(self, history: TelemetryHistoryBuffer) -> ModelOutput:
        from neurodrive_bench.models.neural.common import telemetry_to_tensor
        tensor = telemetry_to_tensor(history, self.device)
        
        # Extract delta_t values from history (index 6 in the state vector)
        delta_ts = tensor[:, :, 6:7]
        
        # For LNN, we pass delta_ts separately to drive the ODE
        out, avg_tau = self.network(tensor, delta_ts)
        self._last_tau = float(avg_tau[0].item())
        
        return ModelOutput(
            command=ControlCommand(
                steering=float(out[0, 0].item()),
                throttle=float(out[0, 1].item()),
                brake=float(out[0, 2].item())
            ),
            uncertainty_score=float(out[0, 3].item()),
            adaptation_level=float(out[0, 4].item()),
            debug={
                "parameter_count": self.parameter_count(),
                "lnn_time_constant": self._last_tau
            }
        )
