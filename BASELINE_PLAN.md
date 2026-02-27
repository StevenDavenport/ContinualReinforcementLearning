# Baseline Plan: Benchmark for Eval-Time Latent Rehearsal

## Goal
Design experiments that directly test your publication claim:
- latent imagination can be used as an on-demand rehearsal mechanism,
- rehearsal recovers task performance at re-exposure without replay and without new data collection for rehearsal itself,
- recovery and forgetting effects are measurable, reproducible, and publication-ready.

This plan targets benchmark/protocol design and measurement.  
Your Dreamer-style rehearsal agent will be integrated later.

---

## Core Evaluation Principle

Every re-exposure checkpoint must report both:
- pre-rehearsal performance (agent acts immediately on forgotten task),
- post-rehearsal performance (after internal latent rehearsal only).

Primary scientific quantity:
- recovery gain = post-rehearsal return - pre-rehearsal return.

---

## First Executable Experiment (Locked for Initial Iteration)

Environment root:
- `toy / dm_control / vision_sequential_quadruped_recovery`

Task sequence:
- `quadruped_run -> quadruped_fetch -> quadruped_escape`

Reason:
- Shared embodiment/dynamics with clearly different task objectives.
- Designed to produce meaningful interference while avoiding visual-shift-only confounds.

Baseline pipeline validation runs:
- Smoke sanity:
  - `--tier smoke --num-seeds 1`
- Full baseline iteration:
  - `--tier dev --num-seeds 3` (increase seeds later for paper-quality CI).

---

## Phase 1: Protocol Validity & Guardrails

- [ ] Remove task-ID leakage from model-facing observations.
- [ ] Keep task identifiers only in metadata/artifacts for analysis.
- [ ] Add an explicit `rehearsal_mode` protocol field with allowed values:
  - [ ] `none` (control)
  - [ ] `latent_eval` (your method mode)
- [ ] Add an explicit rule in docs: no replay data and no extra real env interaction during rehearsal.
- [ ] Add tests asserting protocol constraints are enforced.

Acceptance criteria:
- Benchmarks can cleanly separate no-rehearsal control from latent-rehearsal treatment.
- Evaluation setup cannot accidentally use replay or hidden task labels.

---

## Phase 2: Task Sequences Designed for Re-Exposure Recovery

- [ ] Create at least one transfer-friendly continual sequence (same morphology family).
- [ ] Create at least one interference-heavy sequence (conflicting objectives).
- [ ] Ensure each sequence includes scheduled re-exposure points to earlier tasks.
- [ ] Keep old root as compatibility baseline; add new options for rehearsal-focused sequences.
- [ ] Document rationale for each sequence in experiment docs.

Suggested sequence families:
- [ ] `walker_progressive`: stand -> walk -> run -> harder walk variant.
- [ ] `walker_interference`: walk-forward -> run-forward -> conflicting objective -> re-expose walk-forward.

Acceptance criteria:
- Sequences produce measurable pre-rehearsal degradation on at least one earlier task.
- Re-exposure points are deterministic and logged.

---

## Phase 3: Rehearsal-Eval Protocol Hooks (Agent-Agnostic)

- [ ] Add an evaluation hook lifecycle for re-exposure:
  - [ ] `pre_eval(task_id)`
  - [ ] optional `rehearsal_window(task_id, budget_spec)`
  - [ ] `post_eval(task_id)`
- [ ] Ensure the runner can execute the same task evaluation both before and after rehearsal.
- [ ] Add artifact schema fields for:
  - [ ] `pre_rehearsal_return`
  - [ ] `post_rehearsal_return`
  - [ ] `rehearsal_budget` (imagined rollouts, horizon, latent batch size)
  - [ ] `rehearsal_wallclock_ms`
- [ ] Keep baseline agents functional when `rehearsal_mode=none`.

Acceptance criteria:
- Framework supports rehearsal-aware evaluation without requiring any specific world-model implementation.
- External users can plug in a rehearsal-capable agent later with minimal changes.

---

## Phase 4: Metrics Focused on Recovery, Not Just Final Return

- [ ] Keep existing CRL metrics (`forgetting`, `retention`, `final return`).
- [ ] Add rehearsal-specific metrics:
  - [ ] `recovery_gain_by_task`
  - [ ] `recovery_ratio_by_task` (post/pre)
  - [ ] `steps_to_recovery_threshold` (if iterative rehearsal allowed)
  - [ ] `first_episode_return_delta` at re-exposure
- [ ] Add compute-normalized metrics:
  - [ ] gain per rehearsal-step
  - [ ] gain per rehearsal-second
- [ ] Add seed-level and aggregated CI reporting for all new metrics.

Acceptance criteria:
- Reviewer can distinguish "agent never forgot" vs "agent forgot but recovered via rehearsal."
- Metrics support fair comparison versus no-rehearsal controls.

---

## Phase 5: Controlled Ablation Matrix for Publication

- [ ] Define mandatory control arm:
  - [ ] no rehearsal (`rehearsal_mode=none`)
- [ ] Define rehearsal budget ablations:
  - [ ] latent sample count (small/medium/large)
  - [ ] imagination horizon
  - [ ] number of rehearsal gradient updates
- [ ] Define robustness ablations:
  - [ ] posterior sampling noise scale
  - [ ] dynamics rollout depth
- [ ] Add command templates for each ablation cell.

Acceptance criteria:
- Final paper figures can show a clear dose-response between rehearsal budget and recovery.
- Controls and treatments are compute-accounted and reproducible.

---

## Phase 6: Run Plan and Sign-off Criteria

- [ ] Run multi-seed baseline controls on all rehearsal-focused sequences.
- [ ] Run multi-seed rehearsal treatments with fixed ablation grid.
- [ ] Validate all artifact directories.
- [ ] Export publication bundle (JSON/CSV/TEX + plot specs + run manifests).
- [ ] Record exact commands and commit hash.

Sign-off criteria:
- [ ] At least one sequence shows clear forgetting pre-rehearsal.
- [ ] Same sequence shows statistically supported post-rehearsal recovery.
- [ ] Results hold across seeds with reported confidence intervals.

---

## Strict Execution Order

1. Phase 1: validity and guardrails  
2. Phase 2: sequence design  
3. Phase 3: rehearsal-eval hooks  
4. Phase 4: recovery metrics  
5. Phase 5: ablation matrix  
6. Phase 6: final runs and sign-off

---

## Non-Negotiables

- [ ] No replay usage in rehearsal condition.
- [ ] No extra real environment interaction attributed to rehearsal.
- [ ] No task-ID leakage into model inputs.
- [ ] No claims without multi-seed confidence intervals.
- [ ] No final conclusions from smoke-tier-only runs.
