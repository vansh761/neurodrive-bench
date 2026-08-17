from __future__ import annotations

from neurodrive_bench.models.baselines import AdaptiveTemporalStub, LSTMBaseline, TransformerBaseline
from neurodrive_bench.models.base import TemporalDrivingModel
from neurodrive_bench.models.profiles import load_profile


def build_model(model_spec: dict[str, object], history_window: int) -> TemporalDrivingModel:
    name = str(model_spec["name"])
    model_type = str(model_spec["type"])
    profile = load_profile(str(model_spec["artifact_path"])) if "artifact_path" in model_spec else None

    if model_type == "lstm":
        return LSTMBaseline(name=name, history_window=history_window, profile=profile)
    if model_type == "transformer":
        return TransformerBaseline(name=name, history_window=history_window, profile=profile)
    if model_type == "adaptive_temporal":
        return AdaptiveTemporalStub(name=name, history_window=history_window, profile=profile)

    checkpoint_path = str(model_spec.get("checkpoint_path", "")) if "checkpoint_path" in model_spec else None

    if model_type == "lstm_neural":
        from neurodrive_bench.models.neural.lstm_net import LSTMDrivingModel
        return LSTMDrivingModel(name=name, history_window=history_window, checkpoint_path=checkpoint_path)
    if model_type == "transformer_neural":
        from neurodrive_bench.models.neural.transformer_net import TransformerDrivingModel
        return TransformerDrivingModel(name=name, history_window=history_window, checkpoint_path=checkpoint_path)
    if model_type == "lnn":
        from neurodrive_bench.models.neural.lnn_net import LNNDrivingModel
        return LNNDrivingModel(name=name, history_window=history_window, checkpoint_path=checkpoint_path)

    raise ValueError(f"Unsupported model type: {model_type}")
