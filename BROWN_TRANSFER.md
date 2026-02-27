# Brown Transfer Experiment (Future Work)

## Goal
Test whether a combined approach of:

1. Brownian parameter noise at task switches, and
2. Rehearsal via latent dynamics rollouts (Dreamer-style world model),

can improve both:

- transfer to new tasks, and
- retention / forgetting on previously learned tasks.

Core question:
"Can we help transfer as well as forgetting with this approach?"

## Why This Experiment
In continual embodied RL, transfer and forgetting are tightly coupled:

- too rigid models may retain old tasks but fail to transfer,
- too plastic models may transfer but forget.

This experiment tests whether controlled switch-time perturbation plus rehearsal can strike a better balance.

## Proposed Task Protocol (Experiment 1 Variant)
Sequence:

1. `escape` (hard anchor task)
2. `run`
3. `fetch`

Evaluation (no online updates during eval):

- after each stage: evaluate all seen tasks,
- final battery: `escape`, `run`, `fetch`,
- optional recovery-only evaluation with rehearsal at test-time.

## Method Arms (Ablation Matrix)
1. Baseline (no switch-noise, no rehearsal)
2. Brownian switch-noise only
3. Rehearsal only
4. Brownian switch-noise + rehearsal

## Brownian Switch-Noise Design
- Apply noise only at task transitions.
- Start with policy network late layers only (conservative first pass).
- Parameter update:
  - `theta <- theta + sigma * ||theta|| * epsilon`
  - `epsilon ~ N(0, I)`
- Sweep `sigma`: `0`, `1e-4`, `3e-4`, `1e-3`.

## Rehearsal Design (Latent Dynamics)
At test-time on revisited tasks:

1. Encode current observation to posterior belief.
2. Sample latent belief batch.
3. Roll out priors for `N` imagination trajectories.
4. Perform policy-only adaptation updates from imagined trajectories.
5. Evaluate task return post-rehearsal.

No replay buffer and no new environment interaction during rehearsal updates.

## Primary Metrics
- Forgetting:
  - `escape_retention_drop`
  - `fetch_retention_drop`
- Transfer:
  - forward transfer to newly introduced tasks (`run`, `fetch`)
  - post-switch adaptation speed
- Stability/plasticity:
  - performance variance at switches
  - degradation after noise injection

## Success Criteria
- Lower forgetting than baseline on anchor tasks (especially `escape`).
- Equal or better transfer to downstream tasks (`run`, `fetch`).
- Rehearsal recovery improves revisitation performance without environment interaction.
- Combined method (noise + rehearsal) outperforms either component alone on the transfer-forgetting tradeoff.

## Notes
- Keep the first implementation simple:
  - shared task-agnostic heads,
  - policy-only rehearsal updates.
- Extend later if needed:
  - task-conditioned heads,
  - value/world-model rehearsal updates,
  - adaptive noise schedules.
