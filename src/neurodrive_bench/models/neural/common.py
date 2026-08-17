from __future__ import annotations

import torch
import torch.nn as nn

from neurodrive_bench.contracts import ControlCommand, ModelOutput, TelemetryFrame
from neurodrive_bench.models.base import TemporalDrivingModel
from neurodrive_bench.telemetry.history import TelemetryHistoryBuffer


def telemetry_to_tensor(history: TelemetryHistoryBuffer, device: torch.device) -> torch.Tensor:
    """
    Converts a history buffer into a PyTorch tensor of shape (1, seq_len, 7).
    Pads with zeros if the buffer is not full.
    """
    vectors = history.padded_state_vectors()
    tensor = torch.tensor(vectors, dtype=torch.float32, device=device)
    return tensor.unsqueeze(0)  # Add batch dimension


class ControlHead(nn.Module):
    """
    Standard output head for all neural driving models.
    Maps from hidden state to (steering, throttle, brake, uncertainty, adaptation).
    """
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Hidden state tensor of shape (batch_size, hidden_size)
        Returns:
            Tensor of shape (batch_size, 5)
        """
        out = self.net(x)
        # Apply bounds
        steering = torch.clamp(out[:, 0], -1.0, 1.0)
        throttle = torch.clamp(out[:, 1], 0.0, 1.0)
        brake = torch.clamp(out[:, 2], 0.0, 1.0)
        uncertainty = torch.clamp(out[:, 3], 0.0, 1.0)
        adaptation = torch.clamp(out[:, 4], 0.0, 1.0)
        
        return torch.stack([steering, throttle, brake, uncertainty, adaptation], dim=1)


class NeuralDrivingModel(TemporalDrivingModel):
    """
    Base class for PyTorch-based temporal driving models.
    """
    def __init__(self, name: str, history_window: int, checkpoint_path: str | None = None) -> None:
        super().__init__(name=name, history_window=history_window)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = self._build_network().to(self.device)
        
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
            
        self.network.eval()

    def _build_network(self) -> nn.Module:
        raise NotImplementedError

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["model_state_dict"])

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)

    @torch.no_grad()
    def predict_from_history(self, history: TelemetryHistoryBuffer) -> ModelOutput:
        tensor = telemetry_to_tensor(history, self.device)
        out = self.network(tensor)
        
        return ModelOutput(
            command=ControlCommand(
                steering=float(out[0, 0].item()),
                throttle=float(out[0, 1].item()),
                brake=float(out[0, 2].item())
            ),
            uncertainty_score=float(out[0, 3].item()),
            adaptation_level=float(out[0, 4].item()),
            debug={"parameter_count": self.parameter_count()}
        )
