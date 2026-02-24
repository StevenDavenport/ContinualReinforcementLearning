# CRL Benchmark Suite Master Plan

This file is the chronological implementation checklist for building a production-grade continual reinforcement learning benchmark suite.

Checklist conventions:
- `[ ]` not started
- `[x]` completed
- `[-]` intentionally skipped (must include rationale in commit/PR)

## Global Quality Bar (Applies to Every Phase)
- [ ] Every experiment is reproducible from a single config and seed.
- [ ] All benchmark outputs include versioned metadata (code SHA, env versions, hardware info).
- [x] All public APIs and config schemas are type-checked and documented.
- [ ] All critical paths are covered by automated tests (unit + integration + smoke).
- [x] CI gates block merges on lint, type, tests, and minimal benchmark smoke pass.
- [ ] Each experiment has toy and robotics tracks with at least one default and one alternative environment option.
- [ ] Metrics are computed with a shared evaluator and identical definitions across tracks.
- [x] Runs can be resumed after interruption without corrupting logs or checkpoints.
- [x] Failures are visible through structured logs and deterministic error handling.
- [x] Results are exportable to publication-ready tables and plots.
- [ ] Artifact and metric schemas are versioned and backward-compatible within each minor release.
- [ ] External agent submissions can be validated with a standard conformance check.
- [ ] Every reported headline number is traceable to exact run IDs and config manifests.

## Phase 0: Project Contract and Scope Freeze
Goal: Lock scope and quality targets before writing framework code.

- [x] Freeze benchmark scope: 5 experiments, each with Toy Track + Robotics Track.
- [x] Freeze experiment metric definitions (forgetting, transfer, regret, recovery, retention).
- [x] Freeze supported environment options for v1 and mark optional stretch options.
- [x] Define acceptance criteria for each phase (entry/exit checklist).
- [x] Define reproducibility policy (seed count, train/eval budgets, eval cadence).
- [x] Define code ownership and review policy for core vs experiment modules.
- [x] Define versioning/release policy (`v0.x` internal, `v1.0` public benchmark).
- [x] Define benchmark governance policy (what changes invalidate comparability and require benchmark version bump).
- [x] Define mandatory statistical reporting policy (CI bands, significance test policy, seed failure policy).
- [x] Create risk register (dependency fragility, simulator install complexity, compute bottlenecks).

## Phase 1: Repository Scaffold and Engineering Foundations
Goal: Build a strong scaffold that can scale without rewrites.

- [x] Create top-level structure: `src/`, `configs/`, `tests/`, `scripts/`, `docs/`, `artifacts/`.
- [x] Create Python package skeleton (namespaced modules, explicit exports).
- [x] Add `pyproject.toml` with pinned core dependencies and optional extras per env family.
- [x] Add lockfile strategy for reproducible installs across CI and local runs.
- [x] Add code quality stack: formatter, linter, type checker, import sorter.
- [x] Add `pre-commit` hooks and enforce in CI.
- [x] Add Makefile/task runner commands for common workflows.
- [x] Add conventional logging setup (structured logs, run id, phase/task tags).
- [x] Add error taxonomy and exception handling policy (configuration, runtime, adapter errors).
- [x] Add baseline docs: architecture overview, install matrix, quickstart, contributing.

## Phase 2: Core Architecture Contracts
Goal: Implement stable interfaces that all experiments and envs plug into.

- [x] Define `AgentAdapter` contract (act, update, save/load, reset).
- [x] Define `EnvironmentAdapter` contract (reset, step, seed, metadata, render hooks).
- [x] Define `TaskStream` contract (task schedule, boundaries, hidden context behavior).
- [x] Define `Evaluator` contract (evaluation matrix, per-task metrics, aggregate metrics).
- [x] Define `RunOrchestrator` contract (train/eval loop, checkpointing, resume).
- [x] Define artifact contract (logs, checkpoints, metrics parquet/csv/json, plots).
- [x] Define result schema contract (per-step trace, per-eval summary, per-run aggregate).
- [x] Add strict schema validation for adapter outputs (observation/action/reward/done/info).
- [x] Add registry/plugin system for env families and experiment modules.
- [x] Add compatibility tests for all core interfaces.

## Phase 3: Configuration System and Reproducibility Layer
Goal: Make experiment execution deterministic, auditable, and easy to rerun.

- [x] Implement hierarchical config system (`base -> experiment -> track -> env option -> run`).
- [x] Add config schema validation with informative errors.
- [x] Implement config inheritance/overrides with CLI support.
- [x] Implement seed management policy (global seed + component-derived sub-seeds).
- [x] Save fully resolved config snapshot with each run artifact.
- [x] Save immutable run manifest (git SHA, dependency lock, env package versions).
- [x] Add deterministic mode switches where simulators allow it.
- [x] Add reproducibility smoke test: same seed + config reproduces same metric trace tolerance.

## Phase 4: Data, Metrics, and Reporting Backbone
Goal: Ensure metrics are correct, consistent, and publication-ready.

- [x] Implement metric primitives shared across all experiments.
- [x] Implement forgetting and retention matrix computation.
- [x] Implement forward/backward transfer computation.
- [x] Implement switch regret and recovery-time computation.
- [x] Implement confidence interval and seed aggregation utilities.
- [x] Implement statistical comparison utilities (paired tests / bootstrap CIs) with documented assumptions.
- [x] Implement run-level and experiment-level summary artifacts.
- [x] Implement canonical plot generators for all five experiment types.
- [x] Implement table exporters (csv + latex-friendly).
- [x] Implement one-command publication pack exporter (figures + tables + manifests + method metadata).
- [x] Add unit tests with synthetic traces that assert metric correctness.

## Phase 5: CI/CD, Testing Pyramid, and Reliability Gates
Goal: Enforce production-level rigor automatically.

- [x] Set up CI matrix for lint, type checks, unit tests, integration smoke tests.
- [x] Add optional CI jobs for heavy env dependencies (nightly or self-hosted).
- [x] Add contract tests for every adapter against core interfaces.
- [x] Add deterministic regression tests for scheduler/task stream behavior.
- [x] Add checkpoint-resume integration tests.
- [x] Add failure-injection tests (bad config, env crash, interrupted run).
- [x] Add minimum coverage thresholds for core modules.
- [x] Add artifact validation in CI (expected files, schema, metric keys).
- [x] Add baseline regression checks (golden metric tolerance) to detect evaluator or protocol drift.

## Phase 6: Shared Runtime Utilities for All Experiments
Goal: Implement reusable wrappers to avoid duplicated experiment logic.

- [x] Implement observation wrappers (pixels, frame stack, resize, normalization).
- [x] Implement action wrappers (repeat, clipping, scaling, delay injection).
- [x] Implement dynamics randomization hooks (mass/friction/noise) where supported.
- [x] Implement hidden context switch controller and boundary masking.
- [x] Implement task scheduler library (sequential, cyclic, stochastic, curriculum).
- [x] Implement common eval scheduler (stage-end eval, periodic eval, switch-point eval).
- [x] Implement run storage layout standard (by experiment/track/env_option/seed).
- [x] Implement resource monitors (steps/sec, wall-clock, GPU mem, RAM).

## Phase 7: Experiment 1 Implementation (Sequential Skills / Forgetting)
Goal: Ship full experiment across both tracks and alternatives.

### Experiment 1 Spec Lock
- [x] Finalize task sequence templates (`A->B->C->D`) and train budget tiers (`smoke`, `dev`, `full`).
- [x] Finalize eval matrix schedule after each stage.
- [x] Finalize forgetting/retention reporting template.

### Toy Track
- [x] Implement default Toy option: DM Control vision sequential tasks.
- [x] Implement Toy alternative: Procgen sequential games with fixed seed protocol.
- [x] Add adapter tests for both Toy options.
- [x] Add smoke configs and expected runtime bounds.

### Robotics Track
- [x] Implement default Robotics option: Meta-World sequential manipulation tasks.
- [x] Implement Robotics alternative: ManiSkill sequential manipulation tasks.
- [x] Add adapter tests for both Robotics options.
- [x] Add smoke configs and expected runtime bounds.

### Quality Gate
- [x] Verify metric parity between Toy and Robotics tracks.
- [x] Verify end-to-end reproducibility across at least 5 seeds.
- [x] Publish canonical plots and summary table for Experiment 1.

## Phase 8: Experiment 2 Implementation (Forward Transfer)
Goal: Ship transfer benchmark with strict scratch-vs-continual comparators.

### Experiment 2 Spec Lock
- [ ] Finalize threshold-return definitions per task family.
- [ ] Finalize transfer ratio formula and tie-break policy for non-converged runs.
- [ ] Finalize initial post-switch window (`N` episodes/steps) for reporting.

### Toy Track
- [ ] Implement default Toy option: Crafter staged achievement curriculum.
- [ ] Implement Toy alternative: Memory Maze size curriculum.
- [ ] Add curriculum integrity tests (stage ordering and boundary behavior).
- [ ] Add scratch baseline runner for each Toy stage.

### Robotics Track
- [ ] Implement default Robotics option: Meta-World ML10-style transfer protocol.
- [ ] Implement Robotics alternative: Fetch curriculum (`reach->push->slide->pick-place`).
- [ ] Add curriculum integrity tests for Robotics options.
- [ ] Add scratch baseline runner for each Robotics stage.

### Quality Gate
- [ ] Verify transfer ratio computed identically across tracks/options.
- [ ] Verify from-scratch comparator reproducibility and fairness constraints.
- [ ] Publish canonical plots and summary table for Experiment 2.

## Phase 9: Experiment 3 Implementation (Hidden Non-Stationarity)
Goal: Ship robust drift adaptation benchmark with hidden switches.

### Experiment 3 Spec Lock
- [ ] Finalize drift event schema (gradual vs abrupt, amplitude, duration).
- [ ] Finalize hidden-boundary policy (no explicit IDs, no boundary markers).
- [ ] Finalize regret/recovery computation windows.

### Toy Track
- [ ] Implement default Toy option: DM Control with hidden dynamics/observation drift wrappers.
- [ ] Implement Toy alternative: Procgen distribution-shift schedule.
- [ ] Add drift controller tests with deterministic scripted shifts.
- [ ] Add post-shift evaluation hooks for both Toy options.

### Robotics Track
- [ ] Implement default Robotics option: Fetch with time-varying hidden dynamics.
- [ ] Implement Robotics alternative: robosuite with randomized physics/sensor perturbations.
- [ ] Add drift controller tests for both Robotics options.
- [ ] Add post-shift evaluation hooks for both Robotics options.

### Quality Gate
- [ ] Verify hidden-switch integrity (no accidental leakage through observation/info).
- [ ] Verify regret/recovery metrics against synthetic ground truth traces.
- [ ] Publish canonical plots and summary table for Experiment 3.

## Phase 10: Experiment 4 Implementation (Long-Horizon Recall)
Goal: Ship revisit-based memory benchmark with explicit gap-length analysis.

### Experiment 4 Spec Lock
- [ ] Finalize cyclic schedules and revisit gap-length settings.
- [ ] Finalize relearning speedup and backward-transfer definitions.
- [ ] Finalize long-horizon episode length/budget settings.

### Toy Track
- [ ] Implement default Toy option: Memory Maze cyclic revisit protocol.
- [ ] Implement Toy alternative: Crafter subgoal cycle protocol.
- [ ] Add tests for cycle integrity and revisit indexing.
- [ ] Add retention-vs-gap evaluator hooks.

### Robotics Track
- [ ] Implement default Robotics option: Franka Kitchen compositional revisit cycles.
- [ ] Implement Robotics alternative: Meta-World subset revisit cycles.
- [ ] Add tests for cycle integrity and revisit indexing.
- [ ] Add retention-vs-gap evaluator hooks.

### Quality Gate
- [ ] Verify first-visit vs revisit comparison fairness.
- [ ] Verify backward-transfer metric stability across seeds.
- [ ] Publish canonical plots and summary table for Experiment 4.

## Phase 11: Experiment 5 Implementation (Hidden Context Inference)
Goal: Ship boundary-free continual benchmark without task IDs.

### Experiment 5 Spec Lock
- [ ] Finalize stochastic switch process (dwell-time distribution, context count).
- [ ] Finalize hidden-context leakage checks.
- [ ] Finalize adaptation-slope and worst-quantile reporting definitions.

### Toy Track
- [ ] Implement default Toy option: hidden-context visual switching (DM Control or Procgen contexts).
- [ ] Implement Toy alternative: mixed Crafter worlds with latent context bits.
- [ ] Add hidden-context integrity tests.
- [ ] Add switch-triggered evaluation and logging hooks.

### Robotics Track
- [ ] Implement default Robotics option: Meta-World task-set switching without task IDs.
- [ ] Implement Robotics alternative: RLBench switching without boundary annotations.
- [ ] Add hidden-context integrity tests.
- [ ] Add switch-triggered evaluation and logging hooks.

### Quality Gate
- [ ] Verify no explicit/implicit task identity leakage in agent inputs.
- [ ] Verify adaptation slope computation consistency across options.
- [ ] Publish canonical plots and summary table for Experiment 5.

## Phase 12: Baselines and Agent Integration Hardening
Goal: Make benchmark usable for external CRL agents with minimal friction.

- [ ] Implement random-policy baseline for every experiment/track/option.
- [ ] Implement from-scratch per-task baseline for transfer/forgetting comparisons.
- [ ] Implement reference Dreamer-style adapter integration.
- [ ] Implement strict adapter compliance tests for external agents.
- [ ] Implement submission validator CLI (`validate-submission`) for third-party benchmark entries.
- [ ] Implement benchmark-card template for each submission (agent details, compute budget, caveats).
- [ ] Add example integration docs and templates for third-party agents.
- [ ] Validate that baseline metrics are within expected sanity ranges.

## Phase 13: End-to-End Benchmark Assembly and Scaling
Goal: Assemble complete benchmark suite and validate performance at scale.

- [ ] Build unified CLI to run any experiment/track/option from one command surface.
- [ ] Add orchestration for multi-seed, multi-option sweep execution.
- [ ] Add run queue/resume behavior for preemptible jobs.
- [ ] Add distributed execution support (single-node multi-GPU minimum).
- [ ] Add artifact index and run browser (local first, optional remote backend).
- [ ] Add cross-run comparison report (`compare-runs`) with standardized deltas and significance annotations.
- [ ] Execute full dry run on `smoke` and `dev` tiers for all 5 experiments.

## Phase 14: Documentation, Reproducibility Pack, and Public Release
Goal: Ship a benchmark suite others can use and trust.

- [ ] Write complete docs for architecture, installation, and benchmark usage.
- [ ] Write per-experiment docs with protocol diagrams and metric definitions.
- [ ] Write environment setup guides per option (with known pitfalls).
- [ ] Publish reproducibility package template (configs, seeds, manifests, scripts).
- [ ] Publish contribution guide for adding new env options or new experiments.
- [ ] Publish result schema spec and benchmark governance policy in docs.
- [ ] Publish paper-ready reporting guide (which figures/tables are mandatory for claims).
- [ ] Create release checklist and run final `v1.0` validation.
- [ ] Tag and release benchmark with changelog and migration notes.

## Environment Option Matrix (Implementation Coverage Checklist)

Use this section to track implementation status by experiment and environment option.

### Experiment 1
- [ ] Toy default: DM Control vision sequential tasks
- [ ] Toy alternative: Procgen sequential games
- [ ] Robotics default: Meta-World sequential tasks
- [ ] Robotics alternative: ManiSkill sequential tasks

### Experiment 2
- [ ] Toy default: Crafter curriculum
- [ ] Toy alternative: Memory Maze curriculum
- [ ] Robotics default: Meta-World ML10-style transfer
- [ ] Robotics alternative: Fetch curriculum

### Experiment 3
- [ ] Toy default: DM Control hidden drift
- [ ] Toy alternative: Procgen distribution shifts
- [ ] Robotics default: Fetch hidden dynamics drift
- [ ] Robotics alternative: robosuite randomized drift

### Experiment 4
- [ ] Toy default: Memory Maze revisit cycles
- [ ] Toy alternative: Crafter revisit cycles
- [ ] Robotics default: Franka Kitchen revisit cycles
- [ ] Robotics alternative: Meta-World revisit cycles

### Experiment 5
- [ ] Toy default: DM Control/Procgen hidden-context switching
- [ ] Toy alternative: Crafter hidden-context worlds
- [ ] Robotics default: Meta-World hidden task-set switching
- [ ] Robotics alternative: RLBench hidden switching

## Final Go/No-Go Before Development Starts
- [ ] Phase 0 through Phase 3 accepted and signed off.
- [ ] Core interfaces and config schemas merged and tested.
- [ ] CI pipeline green on scaffold branch.
- [ ] At least one smoke run succeeds for one Toy and one Robotics option.
- [ ] Team agrees to begin feature implementation using this plan chronologically.
