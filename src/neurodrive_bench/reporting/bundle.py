from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def build_demo_bundle(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    if not output_path.exists():
        raise FileNotFoundError(f"Benchmark output directory not found: {output_path}")

    manifest = {
        "bundle_name": output_path.name,
        "root": str(output_path),
        "required_entrypoints": _entrypoints(output_path),
        "files": _collect_files(output_path),
    }
    manifest["file_count"] = len(manifest["files"])

    manifest_path = output_path / "demo_bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    readme_path = output_path / "DEMO_BUNDLE_README.md"
    readme_path.write_text(_bundle_readme(manifest), encoding="utf-8")
    return manifest_path


def zip_demo_bundle(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    if not output_path.exists():
        raise FileNotFoundError(f"Benchmark output directory not found: {output_path}")

    archive_base = output_path.parent / output_path.name
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=output_path.parent, base_dir=output_path.name))
    return archive_path


def _entrypoints(output_path: Path) -> dict[str, str | None]:
    candidates = {
        "index_page": output_path / "index.html",
        "research_report": output_path / "research_report.md",
        "benchmark_summary": output_path / "benchmark_summary.json",
        "leaderboard_csv": output_path / "exports" / "leaderboard.csv",
        "degradation_curve_figure": output_path / "figures" / "gdi_degradation_curve.svg",
        "failure_event_figure": output_path / "figures" / "failure_event_counts.svg",
    }
    return {
        name: _relative_or_none(path, output_path)
        for name, path in candidates.items()
    }


def _collect_files(output_path: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in output_path.rglob("*") if item.is_file()):
        if path.name == "demo_bundle_manifest.json":
            continue
        files.append(
            {
                "path": str(path.relative_to(output_path)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "role": _classify(path, output_path),
            }
        )
    return files


def _classify(path: Path, output_path: Path) -> str:
    relative = path.relative_to(output_path)
    parts = relative.parts
    if path.name == "research_report.md":
        return "report"
    if path.name == "benchmark_summary.json":
        return "summary"
    if parts and parts[0] == "figures":
        return "figure"
    if parts and parts[0] == "exports":
        return "csv_export"
    if path.suffix == ".json":
        return "episode_artifact"
    if path.name == "DEMO_BUNDLE_README.md":
        return "bundle_readme"
    if path.name == "index.html":
        return "index_page"
    return "supporting_file"


def _relative_or_none(path: Path, root: Path) -> str | None:
    if not path.exists():
        return None
    return str(path.relative_to(root)).replace("\\", "/")


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _bundle_readme(manifest: dict[str, Any]) -> str:
    entrypoints = manifest["required_entrypoints"]
    lines = [
        "# NeuroDrive Bench Demo Bundle",
        "",
        "This folder contains a reproducible benchmark output bundle.",
        "",
        "## Start Here",
        "",
        f"- Research report: `{entrypoints.get('research_report')}`",
        f"- Benchmark summary: `{entrypoints.get('benchmark_summary')}`",
        f"- Leaderboard CSV: `{entrypoints.get('leaderboard_csv')}`",
        f"- GDI figure: `{entrypoints.get('degradation_curve_figure')}`",
        f"- Failure-event figure: `{entrypoints.get('failure_event_figure')}`",
        "",
        "## Contents",
        "",
        f"- File count: `{manifest['file_count']}`",
        "- Integrity hashes: `demo_bundle_manifest.json`",
        "",
        "## Reproduce",
        "",
        "From the repository root:",
        "",
        "```powershell",
        "$env:PYTHONPATH='src'; python -m neurodrive_bench.cli run --config configs/benchmark.example.yaml",
        "$env:PYTHONPATH='src'; python -m neurodrive_bench.cli export-figures --config configs/benchmark.example.yaml",
        "$env:PYTHONPATH='src'; python -m neurodrive_bench.cli render-figures --config configs/benchmark.example.yaml",
        "$env:PYTHONPATH='src'; python -m neurodrive_bench.cli paper-report --config configs/benchmark.example.yaml",
        "$env:PYTHONPATH='src'; python -m neurodrive_bench.cli bundle --config configs/benchmark.example.yaml",
        "```",
        "",
    ]
    return "\n".join(lines)
