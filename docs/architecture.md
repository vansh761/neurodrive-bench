# Architecture

## 1. Simulation Layer

Responsibilities:

- manage CARLA synchronous execution
- load maps and scenarios
- control weather and traffic
- expose deterministic stepping hooks

## 2. Telemetry Layer

Structured state vector:

- speed
- lane offset
- yaw
- heading error
- obstacle distance
- acceleration
- delta_t

This layer intentionally avoids raw vision so temporal reasoning can be isolated and compared fairly.

## 3. Stress Injection Engine

Stress families:

- environmental: rain, fog, night
- sensor: noise, packet dropout, latency
- scenario: sudden obstacles, lane blockage, traffic surges

The scenario layer should produce explicit scenario plans so both stub mode and CARLA mode express the same stress intent in a comparable way.

## 4. Model Hub

The model hub provides a common contract for temporal models. The platform compares models under identical benchmark settings and records both action outputs and uncertainty-facing metadata.

## 5. Evaluation Engine

Evaluation outputs include:

- safety metrics
- stability metrics
- robustness curves
- recovery behavior
- graceful degradation index

## 6. Dashboard

The debug console is expected to expose:

- simulation controls
- stress controls
- model selection
- analytics and failure logs
- model adaptation and stability signals
