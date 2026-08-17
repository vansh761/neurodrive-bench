from __future__ import annotations

import torch
from torch.utils.data import Dataset


class TelemetrySequenceDataset(Dataset):
    """
    PyTorch Dataset that creates sliding windows over telemetry sequences for behavioral cloning.
    """
    def __init__(self, records: list[dict], seq_len: int = 20) -> None:
        self.seq_len = seq_len
        self.samples = self._build_samples(records)
        
    def _build_samples(self, records: list[dict]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        # Group by episode
        episodes: dict[str, list[dict]] = {}
        for record in records:
            ep_id = str(record["episode_id"])
            episodes.setdefault(ep_id, []).append(record)
            
        samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for ep_records in episodes.values():
            # Sort by step
            ep_records.sort(key=lambda x: x["step"])
            
            # Sliding window
            for i in range(len(ep_records) - self.seq_len):
                window = ep_records[i : i + self.seq_len]
                target_record = window[-1]  # The expert action to predict is at the end of the window
                
                # Input shape: (seq_len, 7)
                inputs = []
                for r in window:
                    inputs.append([
                        float(r["speed"]),
                        float(r["lane_offset"]),
                        float(r["yaw"]),
                        float(r["heading_error"]),
                        float(r["obstacle_distance"]),
                        float(r["acceleration"]),
                        float(r["delta_t"])
                    ])
                input_tensor = torch.tensor(inputs, dtype=torch.float32)

                target_uncertainty, target_adaptation = _auxiliary_targets(window, target_record)

                # Target shape: (5,) - steering, throttle, brake, uncertainty, adaptation.
                # uncertainty/adaptation are proxy labels (see _auxiliary_targets) rather than
                # expert-authored values, since no such ground truth exists in the demo data.
                target_tensor = torch.tensor([
                    float(target_record["expert_steering"]),
                    float(target_record["expert_throttle"]),
                    float(target_record["expert_brake"]),
                    target_uncertainty,
                    target_adaptation,
                ], dtype=torch.float32)
                
                samples.append((input_tensor, target_tensor))
                
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def _auxiliary_targets(window: list[dict], target_record: dict) -> tuple[float, float]:
    """
    Proxy supervision for the uncertainty/adaptation output channels.

    There is no expert-labeled ground truth for these two quantities, so they were
    previously left untrained (only the first 3 control channels had a loss term).
    We derive physically motivated proxies from data already present in the demo
    records instead:

    - uncertainty: should track how hard the current situation is to control.
      We use the episode's environmental stress_level (present on every record
      from SyntheticDataCollector) plus the current tracking-error magnitude,
      since both increase the true difficulty of predicting the correct action.
    - adaptation: should track how well the trajectory is recovering/stabilizing
      despite disturbance. We reuse the same trend-based "stability" heuristic
      already used for the linear ATSM profile in models/training.py, so
      "adaptation" has one shared definition across the project instead of being
      redefined per model.
    """
    earlier = window[max(0, len(window) - 4)]
    heading_trend = float(target_record["heading_error"]) - float(earlier["heading_error"])
    lane_trend = float(target_record["lane_offset"]) - float(earlier["lane_offset"])
    stability = max(0.0, 1.0 - min(1.0, abs(heading_trend) * 0.08 + abs(lane_trend) * 0.4))

    cte = abs(float(target_record["lane_offset"])) + 0.5 * abs(float(target_record["heading_error"]))
    stress_level = float(target_record.get("stress_level", 0.0))
    uncertainty = max(0.0, min(1.0, 0.6 * stress_level + 0.1 * cte))
    adaptation = stability

    return uncertainty, adaptation
