# NeuroDrive Bench

NeuroDrive Bench is a CARLA-based autonomous driving robustness and safety benchmarking platform for temporal AI decision models. It evaluates how models degrade under environmental, sensor, and scenario stress using structured telemetry rather than raw vision.

## What This Project Is

This repository is for a simulation-first evaluation platform, not a production self-driving stack. The system is designed to answer one question:

Which temporal model remains stable when the environment becomes unstable?

## Core Capabilities

- CARLA-oriented simulation orchestration in synchronous mode
- Structured telemetry pipelines for fair temporal-model comparison
- Stress injection for weather, sensor corruption, and scenario perturbations
- Plug-and-play model interfaces for LSTM, Transformer, and adaptive temporal models
- Robustness metrics focused on safety, stability, and graceful degradation
- Dashboard-ready outputs for debugging, benchmarking, and reporting

## Repository Layout

```text
docs/                   Project definition and architecture
configs/                Benchmark configuration examples
src/neurodrive_bench/   Python package
  cli.py                Command-line entrypoint
  config.py             Benchmark config loading
  contracts.py          Shared data contracts
  orchestration/        Benchmark runner scaffold
  simulation/           CARLA-facing abstractions
  telemetry/            Structured state vector pipeline
  stress/               Stress injection abstractions
  models/               Temporal model interfaces
  metrics/              Safety, stability, and GDI metrics
  dashboard/            Debug console scaffold
```

## Quick Start

1. Create a virtual environment.
2. Install the package in editable mode.
3. Copy the example config and customize it.
4. Run the scaffold benchmark command.

```bash
pip install -e .
python -m neurodrive_bench.cli run --config configs/benchmark.example.yaml
```

## Dashboard

After generating artifacts, you can inspect them in the debug console.

```bash
pip install -e .[dashboard]
streamlit run src/neurodrive_bench/dashboard/app.py
```

To inspect a different artifact folder:

```bash
streamlit run src/neurodrive_bench/dashboard/app.py -- --output-dir outputs/baseline_robustness_suite
```

If you want a reminder command:

```bash
python -m neurodrive_bench.cli dashboard
```

## Synthetic Model Training

You can generate lightweight trainable model profiles without CARLA:

```bash
python -m neurodrive_bench.cli train-models --config configs/benchmark.example.yaml
python -m neurodrive_bench.cli run --config configs/benchmark.example.yaml
```

This creates JSON model profiles in `artifacts/model_profiles/` and lets the benchmark run with calibrated profile weights instead of only fixed handwritten gains.

## Benchmark Summary

Each benchmark run now writes a benchmark-level summary file:

```bash
python -m neurodrive_bench.cli report --config configs/benchmark.example.yaml
```

The summary is saved as `outputs/.../benchmark_summary.json` and includes:

- per-model degradation curves
- average GDI and collision rate
- degradation slope
- a simple leaderboard across models

## Research Report

After running a benchmark, generate a paper-style Markdown report:

```bash
python -m neurodrive_bench.cli paper-report --config configs/benchmark.example.yaml
```

The report is saved as `outputs/.../research_report.md`.

If SVG figures have been rendered, the report automatically embeds them with relative links.

## Figure Data Exports

Export CSV files for plots, slides, or report appendices:

```bash
python -m neurodrive_bench.cli export-figures --config configs/benchmark.example.yaml
```

This writes `leaderboard.csv`, `degradation_curves.csv`, and `failure_events.csv` under `outputs/.../exports/`.

Render SVG figures from those exports:

```bash
python -m neurodrive_bench.cli render-figures --config configs/benchmark.example.yaml
```

This writes report figures under `outputs/.../figures/`.

## Demo Bundle

Create a manifest and bundle README for the generated benchmark outputs:

```bash
python -m neurodrive_bench.cli bundle --config configs/benchmark.example.yaml
```

This writes `demo_bundle_manifest.json` and `DEMO_BUNDLE_README.md` into the benchmark output folder.

Create a single zip archive for sharing:

```bash
python -m neurodrive_bench.cli zip-bundle --config configs/benchmark.example.yaml
```

This writes `outputs/.../baseline_robustness_suite.zip`.

## Demo Bundle Index Page

Generate a human-friendly HTML landing page for the benchmark outputs:

```bash
python -m neurodrive_bench.cli index-page --config configs/benchmark.example.yaml
```

This writes `outputs/.../index.html` with an embedded leaderboard, figure previews, CSV download links, and episode artifact links. Open it in any browser to navigate the full bundle.

## Current Status

This repo currently provides the architecture, interfaces, and benchmark flow scaffold. The simulation layer now supports backend selection through a shared contract, with `stub` as the default runnable mode and a `carla` backend scaffold ready for live integration. Model training, real CARLA episode logic, and the live dashboard are the next implementation layers.

## Progress Tracking

Use [docs/impleentation_plan.md](C:/Users/ASUS/Desktop/LNN_Self_Driving/docs/impleentation_plan.md) as the living status file for what has been built, what is in progress, and what comes next.

If CARLA is not installed yet, use [docs/setup_without_carla.md](C:/Users/ASUS/Desktop/LNN_Self_Driving/docs/setup_without_carla.md) for the recommended local workflow.

## Design Principles

- Synchronous simulation for deterministic evaluation
- Structured telemetry for fair model comparison
- Stress-first testing rather than raw accuracy chasing
- Graceful degradation as a primary outcome
- Reproducible benchmark runs with explicit configuration

## Primary Output

The signature output of the platform is a graceful degradation analysis, including a Graceful Degradation Index (GDI) computed across increasing stress levels.
