from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from neurodrive_bench.config import load_config
from neurodrive_bench.dashboard.app import DEFAULT_OUTPUT_DIR
from neurodrive_bench.environment import inspect_environment
from neurodrive_bench.models.training import SyntheticModelTrainer
from neurodrive_bench.orchestration.runner import BenchmarkRunner
from neurodrive_bench.reporting.bundle import build_demo_bundle
from neurodrive_bench.reporting.bundle import zip_demo_bundle
from neurodrive_bench.reporting.exports import export_figure_data
from neurodrive_bench.reporting.figures import render_figures
from neurodrive_bench.reporting.index_page import generate_index_page
from neurodrive_bench.reporting.research_report import generate_research_report

from neurodrive_bench.data.collector import SyntheticDataCollector
from neurodrive_bench.data.storage import save_dataset
from neurodrive_bench.models.neural.trainer import NeuralTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neurodrive-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a configured benchmark scaffold")
    run_parser.add_argument("--config", required=True, help="Path to YAML benchmark config")

    validate_parser = subparsers.add_parser("validate", help="Validate benchmark config")
    validate_parser.add_argument("--config", required=True, help="Path to YAML benchmark config")

    subparsers.add_parser("doctor", help="Inspect local environment and CARLA availability")
    dashboard_parser = subparsers.add_parser("dashboard", help="Show dashboard launch instructions")
    dashboard_parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Artifact directory to inspect in the dashboard",
    )
    train_parser = subparsers.add_parser("train-models", help="Generate synthetic model profiles")
    train_parser.add_argument("--config", required=True, help="Path to YAML benchmark config")
    summary_parser = subparsers.add_parser("report", help="Show benchmark summary file path")
    summary_parser.add_argument("--config", required=True, help="Path to YAML benchmark config")
    paper_parser = subparsers.add_parser("paper-report", help="Generate a research-style Markdown report")
    paper_parser.add_argument("--config", required=True, help="Path to YAML benchmark config")
    export_parser = subparsers.add_parser("export-figures", help="Export CSV data for figures and slides")
    export_parser.add_argument("--config", required=True, help="Path to YAML benchmark config")
    render_parser = subparsers.add_parser("render-figures", help="Render SVG report figures from exported CSVs")
    render_parser.add_argument("--config", required=True, help="Path to YAML benchmark config")
    bundle_parser = subparsers.add_parser("bundle", help="Create a demo bundle manifest for benchmark outputs")
    bundle_parser.add_argument("--config", required=True, help="Path to YAML benchmark config")
    zip_parser = subparsers.add_parser("zip-bundle", help="Create a zip archive of the demo bundle output folder")
    zip_parser.add_argument("--config", required=True, help="Path to YAML benchmark config")
    index_parser = subparsers.add_parser("index-page", help="Generate an HTML index page for the demo bundle")
    index_parser.add_argument("--config", required=True, help="Path to YAML benchmark config")

    collect_parser = subparsers.add_parser("collect-data", help="Collect expert demonstrations for imitation learning")
    collect_parser.add_argument("--config", required=True, help="Path to YAML benchmark config")
    collect_parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to collect")
    collect_parser.add_argument("--output", default="artifacts/datasets/demo.parquet", help="Path to save the Parquet dataset")

    neural_train_parser = subparsers.add_parser("train", help="Train a neural model using behavioral cloning")
    neural_train_parser.add_argument("--config", required=True, help="Path to YAML benchmark config")
    neural_train_parser.add_argument("--model-type", required=True, choices=["lstm", "transformer", "lnn"], help="Type of neural model to train")
    neural_train_parser.add_argument("--dataset", required=True, help="Path to the Parquet dataset")

    pipeline_parser = subparsers.add_parser("pipeline", help="Run the full benchmark and reporting pipeline")
    pipeline_parser.add_argument("--config", required=True, help="Path to YAML benchmark config")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "doctor":
        report = inspect_environment()
        print(json.dumps(asdict(report), indent=2))
        return

    if args.command == "dashboard":
        print("Install dashboard extras if needed: pip install -e .[dashboard]")
        print(
            "Launch with: "
            f"streamlit run src/neurodrive_bench/dashboard/app.py -- --output-dir {args.output_dir}"
        )
        return

    config = load_config(Path(args.config))

    if args.command == "train-models":
        trainer = SyntheticModelTrainer(config)
        output_paths = trainer.train_all()
        for output_path in output_paths:
            print(f"Saved profile: {output_path}")
        return

    if args.command == "report":
        summary_path = config.output_dir / "benchmark_summary.json"
        print(f"Summary file: {summary_path}")
        return

    if args.command == "paper-report":
        report_path = generate_research_report(config.output_dir)
        print(f"Research report: {report_path}")
        return

    if args.command == "export-figures":
        output_paths = export_figure_data(config.output_dir)
        for output_path in output_paths:
            print(f"Exported: {output_path}")
        return

    if args.command == "render-figures":
        output_paths = render_figures(config.output_dir)
        for output_path in output_paths:
            print(f"Rendered: {output_path}")
        return

    if args.command == "bundle":
        manifest_path = build_demo_bundle(config.output_dir)
        print(f"Demo bundle manifest: {manifest_path}")
        return

    if args.command == "zip-bundle":
        archive_path = zip_demo_bundle(config.output_dir)
        print(f"Demo bundle archive: {archive_path}")
        return

    if args.command == "index-page":
        index_path = generate_index_page(config.output_dir)
        print(f"Index page: {index_path}")
        return

    if args.command == "validate":
        print(f"Config OK: {config.benchmark_name}")
        return

    if args.command == "collect-data":
        collector = SyntheticDataCollector(config)
        records = collector.collect(args.episodes)
        save_dataset(records, args.output)
        return

    if args.command == "train":
        trainer = NeuralTrainer(config, args.model_type, args.dataset)
        best_model = trainer.train()
        print(f"Model saved to {best_model}")
        return

    if args.command == "pipeline":
        print("1. Running benchmark...")
        runner = BenchmarkRunner(config=config)
        runner.run()
        print("2. Exporting figure data...")
        export_figure_data(config.output_dir)
        print("3. Rendering figures...")
        render_figures(config.output_dir)
        print("4. Generating research report...")
        generate_research_report(config.output_dir)
        print("5. Building demo bundle...")
        build_demo_bundle(config.output_dir)
        print("6. Zipping demo bundle...")
        zip_demo_bundle(config.output_dir)
        print("7. Generating index page...")
        index_path = generate_index_page(config.output_dir)
        print(f"Pipeline complete! View results at: {index_path}")
        return

    runner = BenchmarkRunner(config=config)
    summary = runner.run()
    print(summary)


if __name__ == "__main__":
    main()
