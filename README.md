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
  orchestration/        Benchmark runner (episode-averaging, seeding)
  simulation/           CARLA-facing abstractions (stub + carla backends)
  telemetry/            Structured state vector pipeline
  stress/               Stress injection abstractions
  models/               Temporal model interfaces
  metrics/              Safety, stability, and GDI metrics
  dashboard/            Debug console scaffold
```

## Quick Start

1. Create a virtual environment.
2. Install the package in editable mode.
3. Generate a dataset and train the three neural models (all on the same data/seed/optimizer for a fair comparison).
4. Run the benchmark.

```bash
pip install -e .
python -m neurodrive_bench.cli collect-data --config configs/benchmark.example.yaml --episodes 200 --output artifacts/datasets/demo.parquet
python -m neurodrive_bench.cli train --config configs/benchmark.example.yaml --model-type lstm        --dataset artifacts/datasets/demo.parquet
python -m neurodrive_bench.cli train --config configs/benchmark.example.yaml --model-type transformer --dataset artifacts/datasets/demo.parquet
python -m neurodrive_bench.cli train --config configs/benchmark.example.yaml --model-type lnn          --dataset artifacts/datasets/demo.parquet
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

## Rule-Based Sanity Floor (not the primary model comparison)

Separately from the real neural models above, you can generate hand-written rule-based control-law "profiles" as a sanity floor -- these are not neural networks and are not a fair architecture comparison against each other or against the trained models (each uses a different hand-tuned formula, see `models/training.py`):

```bash
python -m neurodrive_bench.cli train-models --config configs/benchmark.example.yaml
```

This creates JSON profiles in `artifacts/model_profiles/`, referenced by the `baseline_rulebased_models` entries in the config. Never report a result from this list as "the LSTM" or "the Transformer" -- see `Current Status` below for why.

## Benchmark Summary

Each benchmark run now writes a benchmark-level summary file:

```bash
python -m neurodrive_bench.cli report --config configs/benchmark.example.yaml
```

The summary is saved as `outputs/.../benchmark_summary.json` and includes:

- per-model degradation curves, with mean/std/min/max/n across `episodes_per_level` repeated episodes at each stress level (not a single point estimate)
- average GDI and collision rate
- degradation slope
- `gdi_collision_rank_consistency`: a Spearman rank correlation checking whether GDI agrees with raw collision rate on which stress level is worse (close to +1.0 is a validated metric; see `Current Status`)
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

The three benchmarked architectures (LSTM, Transformer, and a Liquid/Adaptive Temporal Network) are real trained PyTorch models, trained on the same dataset with the same optimizer/loss/seed, with parameter counts matched to within ~1% of a shared 200k budget (see `models/neural/param_budget.py`) so that a robustness gap reflects an architectural difference, not a capacity difference. Earlier versions of this benchmark compared hand-written rule-based control laws against a real neural model under the same names — that has been fixed; the rule-based versions still exist as a separate, clearly-labeled sanity floor (`baseline_rulebased_models` in the config), never reported as "the LSTM."

Each (model, stress_level) pair is evaluated over multiple episodes (`episodes_per_level` in config) with per-episode seeds, and the benchmark summary reports mean/std/min/max rather than a single point estimate. The GDI metric includes a rank-consistency check against raw collision rate (`gdi_collision_rank_consistency` in the summary output) so the metric's validity against a more direct safety signal is checked, not assumed.

The `stub` simulation backend is fully implemented and is what all reported numbers to date come from. The `carla` backend is implemented and code-reviewed (synchronous-mode ticking, actor lifecycle, telemetry extraction share the same model interface as the stub backend), but a live CARLA episode has not yet been successfully completed end-to-end. GPU-accelerated rendering was attempted on free-tier cloud GPU infrastructure (Google Colab, Kaggle) and blocked by a container-level limitation: those platforms provision GPUs for CUDA compute only and do not expose a display-capable graphics device (`/dev/dri`) to the container, confirmed directly rather than assumed. A software-rendering fallback (Xvfb + Mesa) was also attempted and reliably hit a hardcoded Unreal Engine 4.26 render-thread startup watchdog (a 60-second internal constant, not exposed through any config or command-line flag) before completing a world load. Validating the CARLA backend against a live episode requires a machine with a real, display-capable GPU driver -- either a local machine with a dedicated GPU, or a rented cloud GPU VM with `NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility` explicitly enabled (most free notebook-as-a-service platforms do not enable this by default).

Test coverage (`tests/`) covers the pure-function core: GDI computation, stress scaling, telemetry buffer padding, episode-level dataset splitting (no leakage), the uncertainty/adaptation proxy targets, the mean/std aggregation helper, and parameter-budget parity as a regression test.

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
