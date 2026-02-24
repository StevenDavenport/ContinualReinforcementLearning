# CRL Benchmark Suite Roadmap (Dreamer-Focused, Higher Complexity)

This roadmap defines five continual reinforcement learning (CRL) experiments designed for latent dynamics model agents (Dreamer-style). Each experiment has:
- a **Toy Track** (cheap iteration, high reproducibility),
- a **Robotics Track** (higher realism, publishable external validity),
- matched CRL objectives and metrics so results are comparable across tracks.

## Design Principles
- Prefer image-based observations and partially observed settings.
- Use non-trivial horizons and delayed reward where possible.
- Keep per-experiment train/eval protocol identical across tracks.
- Treat toy tasks as fast proxies for robotics behavior, not as separate benchmarks.

## Experiment 1: Sequential Skills and Catastrophic Forgetting
- **CRL aspect:** Stability-plasticity tradeoff and retention after multi-task sequences.
- **Toy Track (default):** DeepMind Control vision stream with body/dynamics/task switches.
  - Example stream: `walker-walk -> cheetah-run -> quadruped-walk -> humanoid-walk`
- **Toy alternatives:** Procgen game stream with fixed seeds per game.
- **Robotics Track (default):** Meta-World sequential manipulation tasks.
  - Example stream: `reach -> push -> pick-place -> drawer-open`
- **Robotics alternatives:** ManiSkill sequential manipulation tasks with fixed camera presets.
- **Evaluation protocol:** Train on `A -> B -> C -> D`; after each stage evaluate on all seen tasks.
- **Primary metrics:**
  - Average return over seen tasks
  - Forgetting score per task: `max_past_return - current_return`
  - End-of-stream retention

## Experiment 2: Forward Transfer Under Increasing Difficulty
- **CRL aspect:** Positive transfer to unseen but related tasks.
- **Toy Track (default):** Crafter achievement curriculum with staged task complexity.
  - Example stages: survival basics -> tool crafting -> resource chains -> advanced goals
- **Toy alternatives:** Memory Maze size curriculum (`small -> medium -> large`).
- **Robotics Track (default):** Meta-World ML10-style train-task to new-task transfer.
- **Robotics alternatives:** Fetch task curriculum (`reach -> push -> slide -> pick-place`).
- **Evaluation protocol:** Measure sample-efficiency on each new stage versus from-scratch baselines.
- **Primary metrics:**
  - Steps to threshold return
  - Transfer ratio: `scratch_steps / continual_steps`
  - First-N-episodes performance immediately after switch

## Experiment 3: Hidden Non-Stationarity and Online Adaptation
- **CRL aspect:** Robust adaptation to dynamics and observation drift without explicit task boundaries.
- **Toy Track (default):** DM Control vision with hidden drift wrappers.
  - Drift factors: friction, actuator gain, camera pose/noise, lighting
- **Toy alternatives:** Procgen with scheduled domain/level distribution shifts.
- **Robotics Track (default):** Fetch/robotics tasks with hidden dynamics randomization over time.
  - Drift factors: object mass/friction, action delay, control noise
- **Robotics alternatives:** robosuite with domain-randomized physics + sensor perturbations.
- **Evaluation protocol:** Single continuous stream; no reset or task-ID at drift boundaries.
- **Primary metrics:**
  - Switching regret around drift points
  - Recovery time to pre-shift performance band
  - Return variance under sustained drift

## Experiment 4: Long-Horizon Recall and Relearning Savings
- **CRL aspect:** Long-term memory persistence and backward transfer on revisited tasks.
- **Toy Track (default):** Memory Maze cyclic revisit protocol.
  - Example cycle: `context A -> B -> C -> A -> B -> C`
- **Toy alternatives:** Crafter subgoal cycles with long revisit gaps.
- **Robotics Track (default):** Franka Kitchen compositional task cycles.
  - Example cycle: microwave/kettle/burner combinations with revisits
- **Robotics alternatives:** Meta-World subset revisits with controlled gap lengths.
- **Evaluation protocol:** Compare first-visit learning to revisit learning for each context.
- **Primary metrics:**
  - Relearning speedup on revisit
  - Backward transfer score
  - Retention versus gap length (steps between revisits)

## Experiment 5: Hidden Context Inference Without Task IDs
- **CRL aspect:** Boundary-free continual learning and latent context inference.
- **Toy Track (default):** Hidden-context switching across visual task variants (no task labels).
  - Example: randomized switches among 4-6 DM Control or Procgen contexts with variable dwell time
- **Toy alternatives:** Mixed Crafter worlds with latent context bits.
- **Robotics Track (default):** Meta-World task-set switching with no explicit task identity.
- **Robotics alternatives:** RLBench-style task-set switches without boundary annotation.
- **Evaluation protocol:** Agent observes only pixels/proprioception and rewards; context is latent.
- **Primary metrics:**
  - Post-switch performance drop
  - Adaptation slope after switch
  - Stream-average return and worst-case quantile return

## Cross-Experiment Standards (Required)
- Use the same observation interface class across all experiments (image-only or image+proprio).
- Keep fixed train budgets per experiment and fixed eval cadence.
- Report mean/std over at least 5 seeds (prefer 10 for final paper tables).
- Log wall-clock, environment steps, and model update counts.
- Store full evaluation matrix: task/context performance at every evaluation point.
- Report memory footprint and replay size usage for fair model-based comparisons.

## Deliverables Per Experiment
- Config files for both tracks (`toy` and `robotics`).
- Reference random-policy and from-scratch baselines.
- One canonical plot bundle:
  - return-over-time,
  - forgetting/transfer curves,
  - post-switch recovery curves.
- Reproducibility sheet: seeds, environment versions, simulator hashes.

## Recommended Implementation Order
1. Experiment 1 (Toy + Robotics) to validate full training/eval pipeline.
2. Experiment 2 to establish transfer baselines.
3. Experiment 3 to validate drift handling and online evaluation logic.
4. Experiment 4 for long-horizon memory stress.
5. Experiment 5 for full hidden-context continual benchmark.
