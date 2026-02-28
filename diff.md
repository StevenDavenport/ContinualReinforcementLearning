# DreamerV3 Parity Checklist

Scope:
- Original reference: `agents/dreamer/origonal_code/dreamerv3/*.py`, `agents/dreamer/origonal_code/embodied/jax/*.py`, `agents/dreamer/origonal_code/dreamerv3/configs.yaml`
- Current Torch rebuild: `agents/dreamer/*.py`, `agents/dreamer/networks/*.py`

Legend:
- `[P0]` high-impact behavioral mismatch
- `[P1]` medium-impact mismatch
- `[P2]` lower-impact mismatch

## A) Core Training and Loss Mechanics

- [x] **[P0] Add trajectory weighting to actor loss (`weight = cumprod(disc * con)`) and apply it to policy objective.**
Original: `origonal_code/dreamerv3/agent.py:401-415`
Current: `agents/dreamer/losses.py:376-379`
Done when: policy loss multiplies REINFORCE+entropy term by Dreamer-style cumulative continuation weights.

- [x] **[P0] Add slow-value regularization term to critic loss.**
Original: `origonal_code/dreamerv3/agent.py:420-423`
Current: `agents/dreamer/losses.py:394-425`
Done when: value loss includes `value.loss(slowvalue.pred()) * slowreg` equivalent.

- [x] **[P0] Implement replay critic auxiliary loss (`repval`) with scale `0.3`.**
Original: `origonal_code/dreamerv3/agent.py:218-235`, `origonal_code/dreamerv3/agent.py:449-474`, `origonal_code/dreamerv3/configs.yaml:86`
Current: `agents/dreamer/agent.py:292-303`
Done when: replay-based value targets are computed and combined with imagined critic loss per reference behavior.

- [x] **[P0] Apply continuation discounting target transform when `contdisc=True` (`con *= 1 - 1/horizon`).**
Original: `origonal_code/dreamerv3/agent.py:174-177`
Current: `agents/dreamer/agent.py:147`
Done when: continue targets and imagined discount path follow horizon-based continuation semantics.

- [x] **[P0] Replace lambda-return recursion with the original `(last, term, rew, val, boot, disc, lam)` formulation.**
Original: `origonal_code/dreamerv3/agent.py:482-490`
Current: `agents/dreamer/losses.py:256-284`
Done when: returns match reference index conventions and bootstrap handling.

- [x] **[P1] Align imagined rollout tensor structure to include start-state alignment (`H+1`) and action alignment.**
Original: `origonal_code/dreamerv3/agent.py:194-201`
Current: `agents/dreamer/agent.py:345-377`
Done when: imagined features/actions/returns are shaped and aligned like the reference pipeline.

- [x] **[P0] Ensure actor objective uses score-function form with stop-gradient action in log-prob term.**
Original: `origonal_code/dreamerv3/agent.py:411-414`
Current: `agents/dreamer/agent.py:347-350`, `agents/dreamer/losses.py:377`
Done when: gradient flow through action sample/logprob matches reference intent.

- [x] **[P1] Replace current return normalization path with `retnorm/valnorm/advnorm` module behavior.**
Original: `origonal_code/dreamerv3/agent.py:70-72`, `origonal_code/dreamerv3/agent.py:407-410`, `origonal_code/embodied/jax/utils.py:16-74`
Current: `agents/dreamer/losses.py:302-373`
Done when: normalization stats/update/debias behaviors are configurable and equivalent.

- [x] **[P0] Change reconstruction loss reduction from mean-over-events to sum-over-events (`Agg(sum)`).**
Original: `origonal_code/embodied/jax/heads.py:90-92`, `origonal_code/dreamerv3/rssm.py:354-356`
Current: `agents/dreamer/losses.py:121`, `agents/dreamer/losses.py:138`
Done when: event-dimension reduction mirrors reference aggregation.

- [x] **[P1] Remove extra `0.5` factor in symlog-MSE to match reference loss magnitude.**
Original: `origonal_code/embodied/jax/outs.py:126-132`
Current: `agents/dreamer/losses.py:30-38`
Done when: per-element symlog squared error scale matches original.

- [x] **[P1] Add configurable gradient-routing switches (`reward_grad`, `ac_grads`, `repval_grad`).**
Original: `origonal_code/dreamerv3/agent.py:17`, `origonal_code/dreamerv3/agent.py:172`, `origonal_code/dreamerv3/agent.py:196`, `origonal_code/dreamerv3/agent.py:220`, `origonal_code/dreamerv3/configs.yaml:88`, `origonal_code/dreamerv3/configs.yaml:114-116`
Current: fixed detach behavior across `agents/dreamer/agent.py`, `agents/dreamer/losses.py`
Done when: detach gates are config-driven and match original toggles.

## B) Optimizer and Update Stack

- [x] **[P0] Replace Adam optimizers with LaProp-style chain (AGC + RMS scaling + momentum + schedule/warmup).**
Original: `origonal_code/dreamerv3/agent.py:342-379`, `origonal_code/embodied/jax/opt.py:109-164`
Current: `agents/dreamer/agent.py:81-83`
Done when: optimizer transforms and schedule semantics match original.

- [x] **[P0] Replace global grad-norm clipping with AGC per parameter tensor.**
Original: `origonal_code/embodied/jax/opt.py:109-123`
Current: `agents/dreamer/agent.py:265`, `agents/dreamer/agent.py:289`, `agents/dreamer/agent.py:301`
Done when: clipping threshold scales with parameter norm as in reference.

- [x] **[P1] Align slow value update controls to include `rate` and `every` semantics.**
Original: `origonal_code/embodied/jax/utils.py:94-119`, `origonal_code/dreamerv3/configs.yaml:110`
Current: `agents/dreamer/networks/critic.py:104-121`
Done when: slow critic update schedule is parameterized equivalently.

## C) RSSM and Architecture

- [x] **[P0] Refactor RSSM core to separate deter/stoch/action projections with grouped fusion before recurrent gates.**
Original: `origonal_code/dreamerv3/rssm.py:141-152`
Current: `agents/dreamer/networks/rssm.py:82-84`, `agents/dreamer/networks/rssm.py:197-200`
Done when: core transition block structurally mirrors original computation path.

- [x] **[P1] Add RSSM action normalization (`action /= max(1, abs(action))`).**
Original: `origonal_code/dreamerv3/rssm.py:137`
Current: `agents/dreamer/networks/rssm.py:197-199`
Done when: recurrent dynamics receives normalized action embedding.

- [x] **[P1] Match GRU update gate biasing (`sigmoid(update - 1)`).**
Original: `origonal_code/dreamerv3/rssm.py:157`
Current: `agents/dreamer/networks/blocks.py:252`
Done when: update gate equation and initialization behavior match.

- [x] **[P1] Center image encoder inputs to `[-0.5, 0.5]` (`/255 - 0.5`).**
Original: `origonal_code/dreamerv3/rssm.py:230`
Current: `agents/dreamer/networks/encoder.py:177-179`
Done when: image preprocessing exactly matches reference.

- [x] **[P1] Implement decoder `bspace` path (block-linear spatial branch + stoch branch fusion).**
Original: `origonal_code/dreamerv3/rssm.py:314-327`
Current: `agents/dreamer/networks/decoder.py:71-87`, `agents/dreamer/networks/decoder.py:119-135`
Done when: decoder hidden-state projection path includes bspace mixing logic.

- [x] **[P1] Generalize decoder/heads to multi-key observation outputs and key-wise heads.**
Original: `origonal_code/dreamerv3/rssm.py:297-306`, `origonal_code/embodied/jax/heads.py:44-62`
Current: `agents/dreamer/networks/decoder.py:145-149`, `agents/dreamer/networks/heads.py:20-165`
Done when: outputs are configured per observation key and type like reference.

## D) Policy and Value Distribution Behavior

- [x] **[P0] Align discrete policy distribution behavior with reference categorical setup (or prove onehot-ST equivalence in our interface).**
Original: `origonal_code/dreamerv3/configs.yaml:102`, `origonal_code/embodied/jax/heads.py:101-110`
Current: `agents/dreamer/networks/actor.py:94`, `agents/dreamer/networks/actor.py:117`
Done when: training/inference action semantics and gradients are reference-equivalent.

- [x] **[P0] Align continuous policy parameterization to bounded-normal formula and std range defaults (`0.1..1.0`).**
Original: `origonal_code/embodied/jax/heads.py:146-153`, `origonal_code/dreamerv3/configs.yaml:100`
Current: `agents/dreamer/networks/actor.py:21-23`, `agents/dreamer/networks/actor.py:100-102`
Done when: mean/std parameterization and entropy behavior match reference.

- [x] **[P1] Add configurable output scaling and initializer controls across modules (`outscale`, trunc-normal variants).**
Original: `origonal_code/dreamerv3/configs.yaml:91-101`, `origonal_code/embodied/jax/heads.py:83`
Current: mostly PyTorch defaults plus explicit zero-init heads
Done when: module init/output scale options can reproduce reference defaults.

- [x] **[P2] Verify two-hot prediction summation path against reference symmetric summation and keep numerically equivalent implementation.**
Original: `origonal_code/embodied/jax/outs.py:285-309`
Current: `agents/dreamer/networks/distributions.py:385-387`
Done when: zero-logit prediction is exactly zero and numerical drift behavior is matched.

## E) Replay and Runtime Behavior

- [x] **[P0] Implement replay context state read/write path (latent carry in replay).**
Original: `origonal_code/dreamerv3/agent.py:90-99`, `origonal_code/dreamerv3/agent.py:312-340`, `origonal_code/dreamerv3/configs.yaml:15`
Current: `agents/dreamer/replay.py:29-181`
Done when: replay can store and restore context entries for train rollouts.

- [x] **[P0] Align replay design toward online queue + chunking + sampling fractions behavior.**
Original: `origonal_code/dreamerv3/configs.yaml:39-46`
Current: `agents/dreamer/replay.py:29-181`
Done when: replay collection/sampling policy is reference-equivalent or intentionally constrained with documented reason.

- [x] **[P1] Align train scheduling semantics with train-ratio based updates.**
Original: `origonal_code/dreamerv3/configs.yaml:48-52`
Current: `agents/dreamer/config.py:56-58`, `agents/dreamer/agent.py:190-193`
Done when: updates-per-env-step behavior matches reference train ratio.

- [x] **[P2] Add reporting parity hooks (open-loop video and optional grad-norm reporting).**
Original: `origonal_code/dreamerv3/agent.py:247-307`, `origonal_code/dreamerv3/agent.py:263-271`
Current: no reporting pipeline in `agents/dreamer/agent.py`
Done when: equivalent diagnostics are available (or explicitly marked intentionally omitted).

## F) Interface and Scope Alignment

- [x] **[P1] Decide and implement action interface parity (dict-of-heads vs single action vector) with explicit adapter mapping.**
Original: `origonal_code/dreamerv3/agent.py:60-63`, `origonal_code/dreamerv3/agent.py:107`, `origonal_code/dreamerv3/rssm.py:78`
Current: `agents/dreamer/config.py:37-38`, `agents/dreamer/utils.py:202-268`
Done when: Torch behavior is equivalent under CRLbench interface constraints.

- [x] **[P1] Expand observation handling to true key-generic behavior (not only pixels/proprio fallback).**
Original: `origonal_code/dreamerv3/agent.py:38-40`, `origonal_code/dreamerv3/rssm.py:192-199`
Current: `agents/dreamer/networks/encoder.py:184-198`, `agents/dreamer/utils.py:109-199`
Done when: observation key handling matches reference generality.

- [x] **[P2] Document lifecycle differences caused by CRLbench adapter contract.**
Original: `origonal_code/dreamerv3/agent.py:101-155`, `origonal_code/dreamerv3/agent.py:247-310`
Current: `agents/dreamer/agent.py:93-414`
Done when: lifecycle mapping is documented and no algorithmic behavior is lost.

## G) Default Configuration Parity

- [x] **[P1] Match `batch_length` default (`64` vs `32`).**
Original: `origonal_code/dreamerv3/configs.yaml:11`
Current: `agents/dreamer/config.py:54`

- [x] **[P1] Match replay size default (`5e6` vs `200000`).**
Original: `origonal_code/dreamerv3/configs.yaml:40`
Current: `agents/dreamer/config.py:55`

- [x] **[P1] Match optimizer LR/default stack (`4e-5` reference shared optimizer).**
Original: `origonal_code/dreamerv3/configs.yaml:87`
Current: `agents/dreamer/config.py:61-63`

- [x] **[P1] Match policy std defaults (`0.1..1.0` vs current).**
Original: `origonal_code/dreamerv3/configs.yaml:100`
Current: `agents/dreamer/networks/actor.py:21-23`

- [x] **[P1] Match default latent class count and size presets.**
Original: `origonal_code/dreamerv3/configs.yaml:91`, `origonal_code/dreamerv3/configs.yaml:120-153`
Current: `agents/dreamer/config.py:44`

- [x] **[P1] Add/align compute dtype policy (`bfloat16` path).**
Original: `origonal_code/dreamerv3/configs.yaml:74`
Current: no explicit dtype policy in `agents/dreamer/config.py`

## Final Parity Sweep (run only after all checklist items above)

- [x] Re-run line-by-line diff versus `origonal_code` for `agent`, `rssm`, `heads/outs`, `opt`, and configs.
- [x] Add parity smoke tests for: world loss terms, imagined loss terms, replay value loss, optimizer update math, and action distributions.
- [x] Run benchmark-facing validation (`validate-agent`) and minimal training smoke with logged loss decompositions.
- [x] Generate final “parity closed” report with any intentional deviations explicitly listed.
