# Dreamer Lifecycle Mapping (CRLbench Adapter)

This file documents lifecycle differences between the original DreamerV3 training loop and this Torch adapter.

## Mapping

- Original `policy(carry, obs)` maps to `DreamerAgent.act(observation, deterministic=...)`.
- Original replay-side context threading (`replay_context`) maps to replay latent context fields:
  - `init_deter`
  - `init_stoch`
  - `init_logits`
  - `init_mask`
- Original `train(carry, data)` maps to `DreamerAgent.update(batch)` with train-ratio credits.
- Original module-level optimization (`opt(...)`) maps to three Torch optimizers:
  - world model optimizer
  - actor optimizer
  - critic optimizer

## Intentional adapter constraints

- CRLbench adapter contract is stateless at API level (`reset/act/update/save`), so carry is internal.
- Action I/O remains single-action (discrete index or continuous vector), not dict-of-action-heads.
- Reporting/video hooks from original `report(...)` are not yet exported by the adapter API.

## Behavior preserved

- World-model reconstruction + KL losses.
- Imagined rollout actor/value losses with Dreamer weighting and normalization.
- Replay value auxiliary loss (`repval`) with configurable scale.
- Slow-value target updates with configurable `rate` and `every`.
- Replay online/uniform/recency sampling plus latent start-context initialization.
