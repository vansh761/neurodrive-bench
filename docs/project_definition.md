# NeuroDrive Bench Project Definition

## One-Line Definition

NeuroDrive Bench is a CARLA-based autonomous driving evaluation and benchmarking platform that measures the robustness, stability, and graceful degradation of temporal AI models under controlled environmental, sensor, and temporal stress conditions using structured telemetry inputs.

## Scope

This project does not attempt to build a full autonomous driving stack. It builds evaluation infrastructure for testing sequential decision models under uncertainty.

## System Goal

The platform measures how temporal models behave when the environment, sensors, and driving scenarios become unreliable. The emphasis is on degradation behavior, not only nominal-condition performance.

## Model Comparison Target

- LSTM baseline
- Transformer baseline
- Adaptive temporal model

Each model must receive:

- the same telemetry schema
- the same temporal history window
- the same delta-time feature
- the same training and evaluation protocol
- a similar parameter scale

## Signature Research Output

The primary outcome is a Graceful Degradation Index that summarizes how smoothly a model's performance declines as stress increases.
