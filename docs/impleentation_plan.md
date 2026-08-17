# Impleentation Plan

This is the living project tracker for NeuroDrive Bench. It is intentionally practical so you can open one file and know both what exists and what still needs to be built.

## Current Position

Project phase: foundation and architecture scaffold

Status summary:

- repository scaffold is complete
- benchmark CLI is runnable in stub mode
- config-driven benchmark flow is working
- simulation backend abstraction is in place
- CARLA backend now has connection, weather, spawn, tick-loop, telemetry, and event-sensing scaffolding
- scenario planning is now shared across stub and CARLA backends
- local environment inspection now exists for machines without CARLA
- stub mode now produces richer episode traces and event samples
- temporal model history windows are now wired into the benchmark flow
- episode artifacts are now stored as structured trace JSON
- metrics are now derived from trace content
- GDI now uses a stronger safety/stability/adaptation balance instead of a placeholder score
- synthetic trainable model profiles now exist for no-CARLA development
- benchmark-level summary aggregation now exists
- benchmark episodes now use deterministic per-model/per-stress seeds
- paper-style research report generation now exists
- research reports now include event-derived failure analysis
- figure-ready CSV export now exists for leaderboard, degradation curves, and failure events
- SVG chart rendering now exists for degradation curves and failure event counts
- generated SVG figures are now embedded into the Markdown research report
- demo bundle manifest generation now exists
- demo bundle zip packaging now exists
- demo bundle index page now exists

## What I Have Done

### 1. Project framing

Completed:

- created the master repository README
- added project definition documentation
- added architecture documentation
- aligned the repo around robustness benchmarking rather than generic self-driving

Files:

- [README.md](C:/Users/ASUS/Desktop/LNN_Self_Driving/README.md)
- [docs/project_definition.md](C:/Users/ASUS/Desktop/LNN_Self_Driving/docs/project_definition.md)
- [docs/architecture.md](C:/Users/ASUS/Desktop/LNN_Self_Driving/docs/architecture.md)

### 2. Benchmark scaffold

Completed:

- created the package structure under `src/neurodrive_bench`
- added config loading and validation
- added CLI commands for `validate` and `run`
- added CLI command for `doctor`
- added benchmark artifact generation

Files:

- [pyproject.toml](C:/Users/ASUS/Desktop/LNN_Self_Driving/pyproject.toml)
- [configs/benchmark.example.yaml](C:/Users/ASUS/Desktop/LNN_Self_Driving/configs/benchmark.example.yaml)
- [src/neurodrive_bench/config.py](C:/Users/ASUS/Desktop/LNN_Self_Driving/src/neurodrive_bench/config.py)
- [src/neurodrive_bench/cli.py](C:/Users/ASUS/Desktop/LNN_Self_Driving/src/neurodrive_bench/cli.py)
- [src/neurodrive_bench/environment.py](C:/Users/ASUS/Desktop/LNN_Self_Driving/src/neurodrive_bench/environment.py)
- [src/neurodrive_bench/orchestration/runner.py](C:/Users/ASUS/Desktop/LNN_Self_Driving/src/neurodrive_bench/orchestration/runner.py)

### 3. Simulation architecture

Completed:

- added a backend interface for simulation execution
- kept a runnable stub backend for local iteration
- added a CARLA backend scaffold with synchronous-mode setup
- added backend factory selection from config
- added CARLA connection settings in config
- added ego spawn and cleanup scaffolding
- added tick-based episode loop and telemetry extraction scaffolding
- connected model prediction hooks into the CARLA episode flow
- upgraded stub mode to emit trace samples and synthetic failure events
- added CARLA-side collision sensor, runtime event derivation, and scenario actor scaffolding
- added shared scenario-planning logic for obstacle, blockage, and traffic stress

Files:

- [src/neurodrive_bench/simulation/base.py](C:/Users/ASUS/Desktop/LNN_Self_Driving/src/neurodrive_bench/simulation/base.py)
- [src/neurodrive_bench/simulation/stub_backend.py](C:/Users/ASUS/Desktop/LNN_Self_Driving/src/neurodrive_bench/simulation/stub_backend.py)
- [src/neurodrive_bench/simulation/carla_backend.py](C:/Users/ASUS/Desktop/LNN_Self_Driving/src/neurodrive_bench/simulation/carla_backend.py)
- [src/neurodrive_bench/simulation/factory.py](C:/Users/ASUS/Desktop/LNN_Self_Driving/src/neurodrive_bench/simulation/factory.py)

### 4. Evaluation contracts

Completed:

- defined telemetry and control contracts
- defined stress-profile representation
- defined metrics and result structures
- added a first-pass GDI computation stub
- upgraded GDI to a more research-like weighted formulation
- added temporal telemetry history buffering for sequence models
- upgraded benchmark artifacts from string traces to structured JSON traces
- added trace-derived metric computation from samples and events
- added synthetic model-profile training and profile-based inference hooks
- added benchmark-level report aggregation and leaderboard output
- added deterministic benchmark episode seeding for reproducible artifact generation
- added Markdown research report generation from benchmark summaries
- added event-log aggregation for report failure analysis
- added CSV figure-data export for plots and appendices
- added dependency-free SVG chart rendering for report figures
- embedded generated SVG figures into research reports when available
- added demo bundle manifest and output-folder README generation
- added zip packaging for the demo bundle

Files:

- [src/neurodrive_bench/contracts.py](C:/Users/ASUS/Desktop/LNN_Self_Driving/src/neurodrive_bench/contracts.py)
- [src/neurodrive_bench/stress/engine.py](C:/Users/ASUS/Desktop/LNN_Self_Driving/src/neurodrive_bench/stress/engine.py)
- [src/neurodrive_bench/metrics/gdi.py](C:/Users/ASUS/Desktop/LNN_Self_Driving/src/neurodrive_bench/metrics/gdi.py)
- [src/neurodrive_bench/metrics/summary.py](C:/Users/ASUS/Desktop/LNN_Self_Driving/src/neurodrive_bench/metrics/summary.py)
- [src/neurodrive_bench/telemetry/history.py](C:/Users/ASUS/Desktop/LNN_Self_Driving/src/neurodrive_bench/telemetry/history.py)

## Where We Are Right Now

The project is at the point where the software structure is correct, the benchmark flow is real, and the next major step is replacing placeholders with live simulation and model execution.

Practical meaning:

- we can run benchmark scaffolding today
- we can inspect whether CARLA is installed before trying live integration
- we have a near-complete CARLA integration skeleton, but it is not validated against a live CARLA server in this environment
- we cannot yet train or compare real models
- we have a clean place to plug those pieces in next

## Next Steps

### Phase 1. Real CARLA execution

Completed:

- done: add CARLA connection settings to config
- done: implement ego vehicle spawn and teardown scaffolding
- done: implement synchronous tick loop scaffolding
- done: extract structured telemetry scaffolding from world state
- done: apply weather and scenario stress into live episodes
- done: add collision sensor and event-log scaffolding
- done: add scenario actor spawning scaffolding for traffic and blockage stress
- done: add shared scenario planning between stub and CARLA backends
- done: refine spawned obstacle placement and lane blockage logic
- done: scaffolding complete (awaiting live server integration)

### Phase 1A. No-CARLA local development

Completed:

- added environment doctor command
- documented stub-first workflow
- improved stub artifacts with samples and event logs

Files:

- [docs/setup_without_carla.md](C:/Users/ASUS/Desktop/LNN_Self_Driving/docs/setup_without_carla.md)
- [src/neurodrive_bench/environment.py](C:/Users/ASUS/Desktop/LNN_Self_Driving/src/neurodrive_bench/environment.py)
- [src/neurodrive_bench/simulation/stub_backend.py](C:/Users/ASUS/Desktop/LNN_Self_Driving/src/neurodrive_bench/simulation/stub_backend.py)

Target files:

- [src/neurodrive_bench/simulation/carla_backend.py](C:/Users/ASUS/Desktop/LNN_Self_Driving/src/neurodrive_bench/simulation/carla_backend.py)
- [configs/benchmark.example.yaml](C:/Users/ASUS/Desktop/LNN_Self_Driving/configs/benchmark.example.yaml)

### Phase 2. Real model inference

Completed or in progress:

- done: add sequence buffers for telemetry windows
- done: standardize temporal history flow across all model stubs
- done: connect lightweight profile weights into inference APIs
- done: add synthetic training path for local model calibration
- in progress: add uncertainty and adaptation signals to outputs

Target files:

- [src/neurodrive_bench/models/base.py](C:/Users/ASUS/Desktop/LNN_Self_Driving/src/neurodrive_bench/models/base.py)
- [src/neurodrive_bench/models/baselines.py](C:/Users/ASUS/Desktop/LNN_Self_Driving/src/neurodrive_bench/models/baselines.py)
- [src/neurodrive_bench/models/registry.py](C:/Users/ASUS/Desktop/LNN_Self_Driving/src/neurodrive_bench/models/registry.py)
- [src/neurodrive_bench/models/training.py](C:/Users/ASUS/Desktop/LNN_Self_Driving/src/neurodrive_bench/models/training.py)
- [src/neurodrive_bench/models/profiles.py](C:/Users/ASUS/Desktop/LNN_Self_Driving/src/neurodrive_bench/models/profiles.py)

### Phase 3. Real metrics

Completed or in progress:

- done: compute collisions from trace signals and event markers
- done: compute off-road frequency from trace signals
- done: compute smoothness, jitter, recovery, and stabilization from control traces
- in progress: refine the final research-grade GDI formula
- done: improve stub-side event richness for collision, near-collision, and control instability

Target files:

- [src/neurodrive_bench/metrics/gdi.py](C:/Users/ASUS/Desktop/LNN_Self_Driving/src/neurodrive_bench/metrics/gdi.py)
- [src/neurodrive_bench/metrics/summary.py](C:/Users/ASUS/Desktop/LNN_Self_Driving/src/neurodrive_bench/metrics/summary.py)

### Phase 4. Debug console

Completed or in progress:

- done: add dashboard app structure
- done: load benchmark artifacts
- done: visualize degradation curves and safety metrics
- done: expose model and episode selectors
- done: surface benchmark leaderboard data from summary files

### Phase 5. Research outputs

Completed or in progress:

- done: generate benchmark-level summary JSON
- done: generate paper-style Markdown report
- done: add failure analysis from individual event logs
- done: add CSV export for report figures
- done: add chart rendering for report figures
- done: embed generated figures into the Markdown report
- done: package report artifacts into a demo bundle manifest
- done: package report artifacts into a zip archive
- done: add a lightweight index page for the demo bundle contents

Target files:

- [src/neurodrive_bench/reporting/summary.py](C:/Users/ASUS/Desktop/LNN_Self_Driving/src/neurodrive_bench/reporting/summary.py)
- [src/neurodrive_bench/reporting/research_report.py](C:/Users/ASUS/Desktop/LNN_Self_Driving/src/neurodrive_bench/reporting/research_report.py)

Target files:

- [src/neurodrive_bench/dashboard/app.py](C:/Users/ASUS/Desktop/LNN_Self_Driving/src/neurodrive_bench/dashboard/app.py)
- [src/neurodrive_bench/dashboard/data.py](C:/Users/ASUS/Desktop/LNN_Self_Driving/src/neurodrive_bench/dashboard/data.py)

## How To Read Progress

Use this status scale:

- `done`: implemented and runnable
- `in progress`: partially implemented
- `next`: ready to build
- `blocked`: waiting on dependency or environment

## Live Status

- `done`: project scaffold
- `done`: CLI benchmark runner
- `done`: stub simulation backend
- `done`: CARLA backend
- `done`: CARLA config settings
- `done`: telemetry extraction scaffold
- `done`: CARLA event sensing scaffold
- `done`: shared scenario planning scaffold
- `done`: no-CARLA doctor workflow
- `done`: richer stub traces
- `done`: telemetry history windows
- `done`: structured episode artifacts
- `done`: benchmark summary aggregation
- `done`: deterministic benchmark seeds
- `done`: research report generator
- `done`: event-derived report failure analysis
- `done`: figure-data CSV export
- `done`: SVG report figures
- `done`: report figure embedding
- `done`: demo bundle manifest
- `done`: demo bundle archive
- `done`: demo bundle index page
- `done`: trace-derived metrics
- `done`: stronger GDI baseline
- `done`: live CARLA validation scaffold
- `done`: richer CARLA scenario placement
- `done`: real model inference
- `done`: synthetic profile training
- `in progress`: research-grade metrics
- `in progress`: dashboard
