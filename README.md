# CRL Benchmark Suite

Benchmark platform for continual reinforcement learning (CRL) experiments.

The repo is designed for external users to:
- drop in an agent under `agents/`,
- run standardized experiments with minimal code changes,
- get reproducible artifacts, metrics, and publication-ready outputs.

## Quickstart

### Requirements
- Python `3.11`
- Conda (recommended)

### Install
```bash
conda env create -f environment.yml
conda activate crl_bench
```

Optional real dm_control backend:
```bash
python -m pip install -e ".[dev,yaml,dm_control]"
```

For `ppo_continuous_baseline` (MLP PPO), install torch:
```bash
python -m pip install -e ".[dev,yaml,dm_control,torch]"
```

### Verify CLI
```bash
python -m crlbench --help
```

## Run Experiments

### 1) Discover and validate agents
```bash
python -m crlbench list-agents --agents-dir agents
python -m crlbench validate-agent --agent ppo_baseline --agents-dir agents
```

### 2) Run one Experiment 1 root
```bash
python -m crlbench run-experiment \
  --experiment experiment_1_forgetting \
  --track toy \
  --env-family dm_control \
  --env-option vision_sequential_default \
  --dm-control-backend auto \
  --agent ppo_baseline \
  --agents-dir agents \
  --tier smoke \
  --num-seeds 1 \
  --out-dir artifacts/exp1_single_ppo
```

### 3) Run all Experiment 1 roots (matrix)
```bash
python -m crlbench run-experiment1-matrix \
  --agent ppo_baseline \
  --agents-dir agents \
  --tier smoke \
  --dm-control-backend auto \
  --num-seeds 1 \
  --out-dir artifacts/exp1_matrix_ppo
```

Backend behavior for dm_control:
- `--dm-control-backend auto`: use real dm_control if installed, else fallback to stub.
- `--dm-control-backend real`: require real dm_control; fail fast if not installed.
- `--dm-control-backend stub`: always use in-repo deterministic stub.

For real dm_control runs, prefer `--agent ppo_continuous_baseline`.

### 4) Validate artifacts
```bash
python -m crlbench validate-artifacts --artifacts-dir artifacts/exp1_matrix_ppo
```

## What “PPO Baseline” vs “Matrix” Means

- `ppo_baseline` is an **agent** (the policy/learner implementation).
- `run-experiment1-matrix` is a **run mode** (executes the same agent over all 4 Experiment 1 environment roots).

You can run:
- one root with `run-experiment`
- all roots with `run-experiment1-matrix`

## Drop In Your Own Agent

Put your agent in:
```text
agents/<your_agent_name>/
  adapter.py
  manifest.json   # optional
```

`adapter.py` must expose:
```python
def create_agent(config: dict):
    ...
```

Returned agent must implement:
- `reset()`
- `act(observation, deterministic=False)`
- `update(batch)`
- `save(path)`

Then run:
```bash
python -m crlbench validate-agent --agent <your_agent_name> --agents-dir agents
python -m crlbench run-experiment ... --agent <your_agent_name> --agents-dir agents
```

Tune agent params via overrides:
```bash
python -m crlbench run-experiment ... \
  --set learning_rate=0.005 \
  --set clip_epsilon=0.15
```

## Experiment 1 Coverage

Current implemented roots:
- `toy / dm_control / vision_sequential_default`
- `toy / dm_control / vision_sequential_quadruped_recovery`
- `toy / dm_control / vision_sequential_quadruped_anchor_escape`
- `toy / procgen / vision_sequential_alternative`
- `robotics / metaworld / manipulation_sequential_default`
- `robotics / maniskill / manipulation_sequential_alternative`

Experiment details:
- `docs/EXPERIMENT_1.md`

## Useful Commands

Repro smoke:
```bash
python -m crlbench repro-smoke --config configs/base.json --max-tasks 3
```

Experiment 1 quality gate (parity + reproducibility + tables/plots):
```bash
python -m crlbench run-experiment1-quality-gate --out-dir /tmp/exp1_quality --seed-count 5
```

Publication pack:
```bash
python -m crlbench export-publication-pack \
  --run-dir artifacts/<run_id_1> \
  --run-dir artifacts/<run_id_2> \
  --out-dir /tmp/publication_pack \
  --method agent=your_agent
```

## Docs

- `docs/AGENT_INTEGRATION.md`
- `docs/EXPERIMENT_1.md`
- `docs/PUBLICATION_PACK.md`
- `ROADMAP.md`
- `PLAN.md`
