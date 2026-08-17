# Setup Without CARLA

You can keep building NeuroDrive Bench even if CARLA is not installed yet.

## What Works Without CARLA

- config validation
- benchmark runs in `stub` mode
- stress scaling
- model interface development
- artifact generation
- implementation tracking

## Recommended Local Workflow

1. Keep `backend: "stub"` in [configs/benchmark.example.yaml](C:/Users/ASUS/Desktop/LNN_Self_Driving/configs/benchmark.example.yaml).
2. Run the environment check:

```powershell
$env:PYTHONPATH='src'; python -m neurodrive_bench.cli doctor
```

3. Validate the config:

```powershell
$env:PYTHONPATH='src'; python -m neurodrive_bench.cli validate --config configs/benchmark.example.yaml
```

4. Run the scaffold benchmark:

```powershell
$env:PYTHONPATH='src'; python -m neurodrive_bench.cli run --config configs/benchmark.example.yaml
```

## When To Install CARLA

Install CARLA when you are ready to validate:

- synchronous world stepping
- live vehicle spawning
- telemetry extraction from the simulator
- real scenario stress in simulation

Until then, `stub` mode is the right development mode for this repository.
