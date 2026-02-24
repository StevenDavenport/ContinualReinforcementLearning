# Architecture Overview

## Modules
- `crlbench.core`: interfaces and registry used by all integrations.
- `crlbench.config`: config schema, validation, and loading with inheritance.
- `crlbench.runtime`: artifact IO, manifests, seeding, logging, and orchestration shell.
- `crlbench.metrics`: stream metrics, transfer/forgetting/regret utilities, statistics.
- `crlbench.experiments`: experiment-specific protocol implementations (to be added by phase).

## Key Contracts
- `AgentAdapter`: interface for any CRL agent (internal or external).
- `EnvironmentAdapter`: normalized interaction layer over toy/robotics environments.
- `TaskStream`: scheduling abstraction for sequential/cyclic/stochastic protocols.
- `Evaluator`: shared metrics surface for all experiments.

## Artifact Model
Each run writes:
- `manifest.json`: immutable metadata (run id, schema version, code + env fingerprints),
- `resolved_config.json`: fully resolved config,
- `events.jsonl`: structured timeline events,
- metric outputs and plots (next phases).

## Design Constraints
- Reproducibility first: all metrics trace back to run id + resolved config.
- Comparability first: same metric definitions across tracks/options.
- Extensibility first: new agent/env integrations should use adapters, not fork logic.
