# Torch vs JAX Dreamer Parity Checklist

This file is the live source of truth for Dreamer Torch vs original JAX parity work.
It supersedes `LIFECYCLE_PARITY.md` and `PARITY_CLOSED_REPORT.md`.

Current development goal:
- Prove that the Torch Dreamer implementation can learn reliably on a small dev preset.
- Do not optimize for paper-comparable results until the correctness items below are resolved.

Permanent adapter constraint unless explicitly revisited:
- CRLbench exposes a `reset/act/update/save` agent contract, so policy/train carry remains internal to the adapter.

Priority legend:
- `P0`: likely changes learning behavior or makes current parity claims unsafe.
- `P1`: meaningful behavioral or tooling gap; should be fixed before claiming strong parity.
- `P2`: lower-risk parity debt or infrastructure mismatch.

Checklist rules:
- Do not check an item off until the code change is merged and the listed validation exists.
- If an item is intentionally kept as a permanent deviation, replace the implementation work with a written decision and the listed tests/documentation.
- Prefer cross-framework parity tests over prose claims whenever a Torch behavior can be compared directly to the original JAX reference.

## Cross-Framework Parity Test Plan

Objective:
- Turn the original JAX implementation in `origonal_code/` into an executable reference for Torch parity work.
- Use deterministic differential tests on frozen inputs; do not use long-horizon learning curves as the primary parity gate.

Principles:
- Compare intermediate tensors and losses, not just final episode return.
- Prefer deterministic or mode/logit paths before sampled paths.
- Inject fixed weights or frozen inputs where needed so differences are attributable to implementation, not initialization.
- Treat end-to-end training parity as a confidence check, not the main pass/fail signal.

Execution plan:
- Phase 1. Pure math parity. Add Torch-vs-JAX tests for two-hot encode/decode, two-hot cross-entropy, symlog/symexp helpers, lambda-return, and any other loss helpers that can run without module state.
- Phase 2. Module-output parity. Add tests for reward/value/continue heads, actor distribution parameterization, and RSSM single-step prior/posterior outputs on frozen tensors with injected weights or equivalent parameter snapshots.
- Phase 3. Sequence and replay-context parity. Add deterministic tests for replay-context splicing, sequence carry alignment, and short imagined rollouts on synthetic sequences.
- Phase 4. Narrow end-to-end sanity parity. Add one tiny smoke that runs both implementations for a very short controlled rollout/update cycle and checks coarse invariants such as finite losses, matching tensor shapes, and same context alignment semantics.

Gating rules:
- A checklist item is not complete until the highest-relevance parity phase for that item exists or a written reason explains why direct cross-framework comparison is impossible.
- P0 math/loss changes should be backed by Phase 1 tests.
- P0/P1 network and RSSM changes should be backed by Phase 2 tests.
- Replay-context and sequence semantics changes should be backed by Phase 3 tests.
- Phase 4 is recommended before claiming broad parity, but it is not a substitute for Phases 1 to 3.

Test harness requirements:
- Put the JAX-facing adapter code behind a small test helper so the main Torch implementation does not depend on the original runtime.
- Keep fixtures tiny and deterministic enough to run in CI.
- Define explicit numeric tolerances per test class, and document when comparisons are shape-only versus value-level.
- Exact sampled-action parity is non-blocking unless RNG handling is explicitly aligned in the harness.

## Active Checklist

- [ ] [P0] Build a Torch-vs-JAX differential parity harness for Dreamer components.
  Current Torch:
  - `tests/test_dreamer_phase1_parity.py` now covers Phase 1 pure-math parity against a tiny reference helper in `tests/dreamer_jax_phase1_reference.py`.
  - the default odd-bin two-hot support in `networks/distributions.py` now mirrors the original JAX bin construction.
  - live JAX-runtime execution is still unavailable in this environment, and Phases 2 to 3 do not exist yet.
  Original JAX:
  - `origonal_code/` contains the reference implementation that parity claims are being made against.
  Why it matters:
  - Dreamer has too many interacting moving parts to rely on prose review alone; parity claims should become executable specs.
  Progress:
  - [x] Phase 1 reference-formula helper landed for `symlog`, `symexp`, `TwoHot.pred`, `TwoHot.loss`, and `lambda_return`.
  - [x] Phase 1 tests landed for Torch parity on those helpers.
  - [ ] Execute the Phase 1 cases against an importable JAX runtime in this repo environment or CI.
  - [ ] Add Phase 2 module-output parity tests.
  - [ ] Add Phase 3 replay-context parity tests.
  Done when:
  - a test helper can run selected JAX reference functions/modules on frozen inputs and compare them against Torch outputs.
  - Phases 1 to 3 of the plan above exist for at least one representative target each.
  Validation:
  - Phase 1 tests cover pure math/loss helpers.
  - Phase 2 tests cover at least one head path and one RSSM step path.
  - Phase 3 tests cover replay-context alignment on synthetic sequences.

- [ ] [P0] Use `is_terminal`, not `done`, for continuation targets.
  Current Torch:
  - `agent.py` writes `cont = 0.0 if done else 1.0`, where `done` includes truncation.
  Original JAX:
  - `origonal_code/dreamerv3/agent.py` uses `con = ~obs['is_terminal']`.
  Why it matters:
  - Truncated episodes should not train the continue head as terminal; this changes imagined discounts and return targets.
  Done when:
  - continuation targets are `0.0` only for true terminal transitions and remain `1.0` for truncations.
  Validation:
  - add a unit test covering terminated vs truncated transitions in replay collation and world-model continue targets.

- [ ] [P0] Fix replay-context alignment and support the original `replay_context` semantics.
  Current Torch:
  - `agent.py` writes only one latent context state and only when `update()` receives a single transition.
  - `replay.py` attaches context to the sampled first transition and then still trains on that same first transition.
  Original JAX:
  - `origonal_code/dreamerv3/agent.py:_apply_replay_context()` prepends `K` replay-context steps and trains on the remainder.
  Why it matters:
  - The current Torch path is off by one for context-backed windows and does not implement `replay_context > 1`.
  Done when:
  - sampled replay windows reproduce the same effective carry/observation/action alignment as the original for `replay_context=1` and `replay_context>1`.
  - context writing works for both single-transition and multi-transition adapter updates.
  Validation:
  - add deterministic unit tests for `replay_context=1` and `replay_context=2`.
  - add a multi-transition `update()` test proving context is written and later reloaded.

- [ ] [P0] Make the observation pipeline schema-faithful instead of synthesizing fallback `proprio`.
  Current Torch:
  - `utils.py` invents a `proprio` vector from leftover numeric keys.
  - `encoder.py` always uses the vector path when `proprio` exists.
  Original JAX:
  - `origonal_code/dreamerv3/rssm.py:Encoder` uses the actual observation keys and does not invent a vector channel.
  Why it matters:
  - image-only tasks are not truly image-only in Torch, and non-`proprio` metadata can leak into the model input.
  Done when:
  - pure-image observations do not fabricate a vector branch.
  - mixed observation schemas are encoded from their real keys, not a flattened fallback bucket.
  Validation:
  - add unit tests for image-only, vector-only, and mixed-key observations.
  - verify the encoder input shapes match the intended observation schema in each case.

- [ ] [P0] Match the original image encoder/decoder architecture closely enough to remove major representation drift.
  Current Torch:
  - `encoder.py` uses kernel-4 stride-2 convs plus adaptive pooling.
  - `decoder.py` uses transposed convolutions plus interpolation.
  Original JAX:
  - `origonal_code/dreamerv3/rssm.py` uses kernel-5 convs, max-pool style downsampling, and repeat-upsample plus conv in the decoder.
  Why it matters:
  - This is a structural modeling difference, not just a framework translation detail.
  Done when:
  - the Torch image stack mirrors the original downsampling and upsampling path, including feature-map resolution transitions.
  Validation:
  - add shape-level tests for every image stage on 64x64 inputs.
  - add an open-loop reconstruction smoke that compares stage/output shapes against the original design.

- [ ] [P1] Match initializer behavior and normalization epsilon defaults.
  Current Torch:
  - most modules use PyTorch default Linear/Conv init, Xavier for `BlockLinear`, and `eps=1e-6` RMS norm.
  Original JAX:
  - `embodied/jax/nets.py` defaults to `trunc_normal_*` initializers and `Norm(..., eps=1e-4)`.
  Why it matters:
  - Dreamer is sensitive to initialization scale and norm behavior, especially under mixed precision.
  Done when:
  - layer init and RMS/layer-norm epsilon are configurable and can reproduce the original defaults.
  Validation:
  - add unit tests that assert initializer selection and parameter-stat ranges per layer type.
  - rerun the first-update smoke in `bfloat16` with no NaNs or infs.

- [ ] [P1] Restore decoder/head parity across all observation keys and data types.
  Current Torch:
  - `decoder.py` mainly reconstructs `pixels` plus configured vector outputs.
  Original JAX:
  - the decoder/head stack reconstructs every non-excluded observation key, with categorical or regression heads chosen from the observation space.
  Why it matters:
  - the current port cannot claim generic observation parity.
  Done when:
  - decoder/head construction is driven by the actual observation schema, including discrete outputs where applicable.
  Validation:
  - add tests using a synthetic observation schema containing image, continuous-vector, and discrete keys.
  - verify loss terms exist for every reconstructible key.

- [ ] [P1] Bring replay sampling/runtime semantics closer to the original embodied replay pipeline.
  Current Torch:
  - `replay.py` approximates online/uniform/recency replay but has no `consec`/`stepid` machinery and simpler chunk semantics.
  Original JAX:
  - `origonal_code/dreamerv3/agent.py` relies on replay metadata to splice normal and replay-context windows correctly.
  Why it matters:
  - even after fixing context alignment, replay runtime behavior will still differ.
  Done when:
  - replay metadata and sequence sampling semantics are either matched or explicitly documented as a permanent simplification.
  Validation:
  - add replay-sequence tests covering chunk boundaries, online windows, and context-prefixed samples.
  - document any remaining simplification in this checklist before checking the item off.

- [ ] [P1] Add full save/load/resume parity for long runs.
  Current Torch:
  - `agent.py` can save weights and counters but has no load path and does not serialize replay contents or normalizer state.
  Original JAX:
  - the runtime checkpoints module state, including normalizers and slow-value state.
  Why it matters:
  - long Dreamer runs are hard to debug or reproduce without faithful resume support.
  Done when:
  - load restores config, model weights, optimizer state, slow critic, normalizers, replay, and counters.
  Validation:
  - add a round-trip resume test: uninterrupted training vs save/load/resume must match for at least one additional update.

- [ ] [P1] Restore report parity or make the permanent deviation explicit.
  Current Torch:
  - `agent.py` has `report_openloop()` but not the full original `report()` surface.
  - grad norms are aggregated only at world/actor/critic level.
  Original JAX:
  - `origonal_code/dreamerv3/agent.py` reports open-loop video grids and optional per-loss grad norms.
  Why it matters:
  - diagnostics are a major part of getting Dreamer training right.
  Done when:
  - the adapter exports equivalent open-loop diagnostics and the intended grad-norm reporting behavior.
  Validation:
  - add a smoke test that returns open-loop video tensors for image observations.
  - add a smoke test for grad-norm reporting when enabled.

- [ ] [P1] Decide whether the single-action adapter interface is a permanent deviation.
  Current Torch:
  - `adapter.py`, `utils.py`, and `actor.py` collapse the original dict-of-action-heads interface to one discrete index or one continuous vector.
  Original JAX:
  - policy heads are keyed by the full action space.
  Why it matters:
  - this is one of the two explicit deviations from the old parity report.
  Done when:
  - either internal dict-shaped actions are restored with an adapter mapping layer, or the single-action interface is kept as a documented permanent deviation.
  Validation:
  - if restored: add tests covering dict-head action sampling/log-prob conversion.
  - if kept: add documentation and contract tests covering discrete and continuous mapping behavior end to end.

- [ ] [P2] Make weight-decay name matching work with PyTorch parameter names.
  Current Torch:
  - `optim.py` defaults to `opt_wdregex = r"/kernel$"`, but PyTorch parameter names end with `.weight`, so decay selection will never match when `opt_wd > 0`.
  Original JAX:
  - parameter names naturally use `/kernel`.
  Why it matters:
  - the bug is dormant at the current `wd=0.0` default but breaks parity once weight decay is enabled.
  Done when:
  - the default regex or matching logic works for PyTorch parameter names.
  Validation:
  - add a unit test showing weight decay applies to intended parameters and not biases.

- [ ] [P2] Reduce runtime dtype differences between Torch and JAX.
  Current Torch:
  - mixed precision uses autocast over float32 parameters.
  Original JAX:
  - compute dtype is globally `bfloat16` by default, with float16 loss scaling handled inside the optimizer runtime.
  Why it matters:
  - this is lower risk than the modeling mismatches above, but still a parity gap.
  Done when:
  - `bfloat16` and `float16` execution paths are explicitly supported and documented.
  Validation:
  - keep the low-precision two-hot regression test.
  - add a `float16` smoke test and verify first-update stability.

- [ ] [P2] Separate dev presets from parity-oriented presets.
  Current Torch:
  - repo defaults are intentionally smaller than the original and the current focus is a `small` dev run.
  Original JAX:
  - task presets are explicit in `configs.yaml`.
  Why it matters:
  - this was the second explicit deviation in the old parity report and should stay explicit.
  Done when:
  - the repo has clearly named dev presets and parity-oriented presets instead of one ambiguous default.
  Validation:
  - commands/docs must show at least one dev preset and one parity-oriented preset, with stated intent for each.
