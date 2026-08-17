from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _episodes_per_level_note(models: dict[str, Any]) -> str:
    counts = set()
    for model_summary in models.values():
        for point in model_summary.get("curve", []):
            counts.add(point.get("episode_count", 1))
    if len(counts) == 1:
        return str(next(iter(counts)))
    return f"varies ({sorted(counts)})"


def build_research_report(
    summary: dict[str, Any],
    artifacts: list[dict[str, Any]] | None = None,
    figure_paths: list[str] | None = None,
) -> str:
    leaderboard = summary.get("leaderboard", [])
    models = summary.get("models", {})
    winner = leaderboard[0] if leaderboard else None
    artifacts = artifacts or []
    failure_analysis = _build_failure_analysis(artifacts)

    lines: list[str] = [
        "# NeuroDrive Bench Research Report",
        "",
        "## Hypothesis",
        "",
        (
            "Temporal driving models should not be evaluated only by clean-condition accuracy. "
            "A stronger evaluation asks whether model behavior degrades smoothly as environment, "
            "sensor, and scenario stress increase."
        ),
        "",
        "## Methodology",
        "",
        "- Benchmark type: robustness and graceful degradation evaluation",
        "- Input type: structured telemetry state vectors",
        "- Models compared: LSTM, Transformer, and Liquid/Adaptive Temporal Network -- "
        "all real trained PyTorch models, on the same dataset/seed/optimizer/loss, with "
        "parameter counts matched within ~1% of a shared 200k budget (see param_budget.py)",
        f"- Episodes per stress level: {_episodes_per_level_note(models)}",
        "- Stress axis: controlled environmental, sensor, and scenario degradation",
        "- Primary metric: Graceful Degradation Index",
        "",
        "## Leaderboard",
        "",
        "| Rank | Model | Mean GDI | Degradation Slope | Mean Collision Rate |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]

    for rank, row in enumerate(leaderboard, start=1):
        lines.append(
            "| {rank} | {model} | {mean_gdi:.3f} | {slope:.3f} | {collision:.3f} |".format(
                rank=rank,
                model=row["model_name"],
                mean_gdi=float(row["mean_gdi"]),
                slope=float(row["degradation_slope"]),
                collision=float(row["mean_collision_rate"]),
            )
        )

    lines.extend(["", "## Degradation Curves", ""])
    for model_name, model_summary in models.items():
        lines.append(f"### {model_name}")
        lines.append("")
        consistency = model_summary.get("aggregate", {}).get("gdi_collision_rank_consistency")
        if consistency is not None:
            lines.append(
                f"GDI/collision-rate rank consistency: {float(consistency):.3f} "
                "(+1.0 = GDI always agrees with collision rate on which stress level is worse; "
                "0.0 = no relationship; -1.0 = inverted)"
            )
            lines.append("")
        lines.append("| Stress Level | n | GDI (mean±std) | Collision Rate (mean±std) | Mean Unc. | Mean Adapt. | Adapt. Latency |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for point in model_summary.get("curve", []):
            gdi = point["gdi"]
            collision = point["collision_rate"]
            lines.append(
                "| {stress:.2f} | {n} | {gdi_mean:.3f}±{gdi_std:.3f} | {coll_mean:.3f}±{coll_std:.3f} | "
                "{unc:.3f} | {adapt:.3f} | {latency:.3f} |".format(
                    stress=float(point["stress_level"]),
                    n=point.get("episode_count", 1),
                    gdi_mean=float(gdi["mean"]),
                    gdi_std=float(gdi["std"]),
                    coll_mean=float(collision["mean"]),
                    coll_std=float(collision["std"]),
                    unc=float(point.get("mean_uncertainty", {}).get("mean", 0.0)),
                    adapt=float(point.get("mean_adaptation", {}).get("mean", 0.0)),
                    latency=float(point.get("adaptation_latency", {}).get("mean", 0.0)),
                )
            )
        lines.append("")

    if figure_paths:
        lines.extend(["## Figures", ""])
        for figure_path in figure_paths:
            title = _figure_title(figure_path)
            lines.append(f"![{title}]({figure_path})")
            lines.append("")

    lines.extend(["## Findings", ""])
    if winner is None:
        lines.append("No benchmark artifacts were available, so no model ranking could be produced.")
    else:
        lines.append(
            "{model} currently ranks highest by mean GDI with a score of {score:.3f}.".format(
                model=winner["model_name"],
                score=float(winner["mean_gdi"]),
            )
        )
        lines.append(
            "The degradation slope captures how sharply the model declines as stress increases; "
            "values closer to zero indicate smoother degradation."
        )
    
    lines.extend([
        "",
        "## Adaptation Analysis",
        "",
        "This section highlights how well models modulate their internal states (adaptation) relative to their explicit uncertainty during stress events.",
        "",
        "| Model | Avg Uncertainty | Avg Adaptation | Adaptation Latency |",
        "| --- | ---: | ---: | ---: |",
    ])
    
    for row in leaderboard:
        # We need to aggregate the curve data for this model.
        model_name = row["model_name"]
        curve = models.get(model_name, {}).get("curve", [])
        if curve:
            avg_unc = sum(float(p.get("mean_uncertainty", {}).get("mean", 0.0)) for p in curve) / len(curve)
            avg_adapt = sum(float(p.get("mean_adaptation", {}).get("mean", 0.0)) for p in curve) / len(curve)
            avg_lat = sum(float(p.get("adaptation_latency", {}).get("mean", 0.0)) for p in curve) / len(curve)
            lines.append(f"| {model_name} | {avg_unc:.3f} | {avg_adapt:.3f} | {avg_lat:.3f} |")

    lines.extend(
        [
            "",
            "## Failure Analysis",
            "",
            "The event analysis below is derived from structured episode traces.",
            "",
            "| Model | Events | Dominant Event | Max Severity | Stress Notes |",
            "| --- | ---: | --- | ---: | --- |",
        ]
    )

    if failure_analysis:
        for row in failure_analysis:
            lines.append(
                "| {model} | {events} | {dominant} | {severity:.3f} | {notes} |".format(
                    model=row["model_name"],
                    events=row["event_count"],
                    dominant=row["dominant_event"],
                    severity=row["max_severity"],
                    notes=row["stress_notes"],
                )
            )
    else:
        lines.append("| unavailable | 0 | none | 0.000 | no event logs found |")

    lines.extend(
        [
            "",
            "Known limitations:",
            "",
            "- CARLA live validation has not yet been completed on this machine.",
            "- The synthetic trainer is a calibration scaffold, not a replacement for real model training.",
            "- Scenario placement in CARLA still needs live-world refinement once CARLA is installed.",
            "",
            "## Conclusion",
            "",
            (
                "NeuroDrive Bench now provides a reproducible robustness-evaluation pipeline: "
                "model-profile generation, stress-conditioned benchmark execution, structured artifacts, "
                "summary aggregation, dashboard inspection, and paper-style reporting."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def generate_research_report(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    summary_path = output_path / "benchmark_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Benchmark summary not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    artifacts = _load_artifacts(output_path)
    figure_paths = _available_figure_paths(output_path)
    report = build_research_report(summary, artifacts=artifacts, figure_paths=figure_paths)
    report_path = output_path / "research_report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def _load_artifacts(output_dir: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("*.json")):
        if path.name == "benchmark_summary.json":
            continue
        artifacts.append(json.loads(path.read_text(encoding="utf-8")))
    return artifacts


def _available_figure_paths(output_dir: Path) -> list[str]:
    figures_dir = output_dir / "figures"
    if not figures_dir.exists():
        return []
    return [str(path.relative_to(output_dir)).replace("\\", "/") for path in sorted(figures_dir.glob("*.svg"))]


def _figure_title(path: str) -> str:
    name = Path(path).stem.replace("_", " ")
    return name.title()


def _build_failure_analysis(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        model_name = str(artifact.get("model_name", "unknown"))
        episode = artifact.get("episode", {})
        metadata = episode.get("metadata", {})
        events = episode.get("events", [])
        row = by_model.setdefault(
            model_name,
            {
                "model_name": model_name,
                "event_count": 0,
                "event_types": {},
                "max_severity": 0.0,
                "notes": set(),
            },
        )

        row["event_count"] += len(events)
        for event in events:
            event_type = str(event.get("event_type", "unknown"))
            row["event_types"][event_type] = row["event_types"].get(event_type, 0) + 1
            row["max_severity"] = max(row["max_severity"], float(event.get("severity", 0.0)))

        for note in metadata.get("scenario_notes", []):
            row["notes"].add(str(note))

    rows: list[dict[str, Any]] = []
    for row in by_model.values():
        event_types = row["event_types"]
        dominant_event = "none"
        if event_types:
            dominant_event = max(event_types.items(), key=lambda item: item[1])[0]
        notes = sorted(row["notes"])
        rows.append(
            {
                "model_name": row["model_name"],
                "event_count": row["event_count"],
                "dominant_event": dominant_event,
                "max_severity": row["max_severity"],
                "stress_notes": ", ".join(notes) if notes else "none",
            }
        )

    return sorted(rows, key=lambda item: (-int(item["event_count"]), str(item["model_name"])))
