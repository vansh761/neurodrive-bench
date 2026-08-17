from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ModelProfile:
    model_type: str
    steering_weights: dict[str, float]
    throttle_weights: dict[str, float]
    uncertainty_bias: float
    adaptation_bias: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "steering_weights": self.steering_weights,
            "throttle_weights": self.throttle_weights,
            "uncertainty_bias": self.uncertainty_bias,
            "adaptation_bias": self.adaptation_bias,
            "metadata": self.metadata,
        }


def load_profile(path: str | Path) -> ModelProfile:
    profile_path = Path(path)
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    return ModelProfile(
        model_type=str(payload["model_type"]),
        steering_weights={str(key): float(value) for key, value in payload["steering_weights"].items()},
        throttle_weights={str(key): float(value) for key, value in payload["throttle_weights"].items()},
        uncertainty_bias=float(payload["uncertainty_bias"]),
        adaptation_bias=float(payload["adaptation_bias"]),
        metadata=dict(payload.get("metadata", {})),
    )


def save_profile(profile: ModelProfile, path: str | Path) -> None:
    profile_path = Path(path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
