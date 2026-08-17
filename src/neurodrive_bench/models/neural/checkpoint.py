from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    epoch: int, 
    loss: float, 
    path: str | Path,
    metadata: dict[str, Any] | None = None
) -> None:
    """Saves a model checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "metadata": metadata or {}
    }
    torch.save(checkpoint, path)

    # Also save metadata alongside for easy inspection without torch
    if metadata:
        meta_path = path.with_suffix(".json")
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Loads a model checkpoint."""
    return torch.load(path, map_location="cpu")
