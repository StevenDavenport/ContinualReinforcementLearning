# Experiment 1 Protocol (Sequential Skills / Forgetting)

## Roots
- Toy default: `toy / dm_control / vision_sequential_default`
- Toy alternative: `toy / procgen / vision_sequential_alternative`
- Robotics default: `robotics / metaworld / manipulation_sequential_default`
- Robotics alternative: `robotics / maniskill / manipulation_sequential_alternative`

## Stage Protocol
- Train in sequence `A -> B -> C -> D` (four-task template per root).
- After each stage, evaluate on all seen tasks.
- Eval matrix rows: `after_task_A`, `after_task_B`, `after_task_C`, `after_task_D`.

## Budget Tiers
- `smoke`: `train_steps=20000`, `eval_interval_steps=2000`, `eval_episodes=10`
- `dev`: `train_steps=250000`, `eval_interval_steps=25000`, `eval_episodes=20`
- `full`: `train_steps=1000000`, `eval_interval_steps=50000`, `eval_episodes=30`

## Expected Runtime Bounds (Planning Targets)

### Smoke
- `dm_control/vision_sequential_default`: <= 10 min, <= 100000 env steps
- `procgen/vision_sequential_alternative`: <= 12 min, <= 120000 env steps
- `metaworld/manipulation_sequential_default`: <= 20 min, <= 120000 env steps
- `maniskill/manipulation_sequential_alternative`: <= 25 min, <= 150000 env steps

### Dev
- `dm_control/vision_sequential_default`: <= 90 min, <= 1250000 env steps
- `procgen/vision_sequential_alternative`: <= 110 min, <= 1250000 env steps
- `metaworld/manipulation_sequential_default`: <= 180 min, <= 1500000 env steps
- `maniskill/manipulation_sequential_alternative`: <= 240 min, <= 1750000 env steps

### Full
- `dm_control/vision_sequential_default`: <= 360 min, <= 5000000 env steps
- `procgen/vision_sequential_alternative`: <= 420 min, <= 5000000 env steps
- `metaworld/manipulation_sequential_default`: <= 600 min, <= 6000000 env steps
- `maniskill/manipulation_sequential_alternative`: <= 720 min, <= 7000000 env steps

## Reporting Template
- Primary: `final_stage_average_return`, `average_forgetting`, `average_retention`
- Task-level: `forgetting_by_task`, `retention_by_task`
- Stage-level: `average_return_by_stage`, `evaluation_matrix`
