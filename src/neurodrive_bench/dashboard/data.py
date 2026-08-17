from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_artifacts(output_dir: str | Path) -> list[dict[str, Any]]:
    artifact_dir = Path(output_dir)
    if not artifact_dir.exists():
        return []

    artifacts: list[dict[str, Any]] = []
    for path in sorted(artifact_dir.glob("*.json")):
        if path.name in ("benchmark_summary.json", "demo_bundle_manifest.json"):
            continue
        if path.name.endswith("_aggregate.json"):
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["artifact_path"] = str(path)
        artifacts.append(payload)
    return artifacts


def load_summary(output_dir: str | Path) -> dict[str, Any] | None:
    summary_path = Path(output_dir) / "benchmark_summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def available_models(artifacts: list[dict[str, Any]]) -> list[str]:
    return sorted({str(item.get("model_name", "unknown")) for item in artifacts})


def filter_artifacts(
    artifacts: list[dict[str, Any]],
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    if model_name is None:
        return artifacts
    return [item for item in artifacts if item.get("model_name") == model_name]


def build_metric_rows(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        metrics = dict(artifact.get("metrics", {}))
        rows.append(
            {
                "model_name": artifact.get("model_name"),
                "stress_level": artifact.get("stress_level"),
                **metrics,
            }
        )
    return sorted(rows, key=lambda row: (str(row["model_name"]), float(row["stress_level"])))
