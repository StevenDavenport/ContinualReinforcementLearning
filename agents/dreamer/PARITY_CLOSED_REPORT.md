# DreamerV3 Torch Parity Report

## Scope
Compared Torch implementation against:
- `origonal_code/dreamerv3/agent.py`
- `origonal_code/dreamerv3/rssm.py`
- `origonal_code/embodied/jax/heads.py`
- `origonal_code/embodied/jax/outs.py`
- `origonal_code/embodied/jax/opt.py`
- `origonal_code/dreamerv3/configs.yaml`

## Line-by-line diff audit (rerun)
- `agents/dreamer/agent.py` vs original agent: `+450 / -611`
- `agents/dreamer/networks/rssm.py` vs original rssm: `+347 / -415`
- `agents/dreamer/networks/distributions.py` vs original outs: `+304 / -393`
- `agents/dreamer/networks/actor.py` vs original heads: `+139 / -182`
- `agents/dreamer/optim.py` vs original opt: `+145 / -166`
- `agents/dreamer/config.py` vs original configs: `+199 / -571`

Large textual diffs are expected due framework conversion (JAX/Ninjax -> PyTorch + CRLbench adapter contract).

## Implemented parity-critical mechanics
- Dreamer-style world model losses: reconstruction + balanced KL with free nats.
- Original lambda-return indexing form (`last/term/rew/val/boot/disc/lam`).
- Imagined actor/value losses with continuation weights and normalization modules.
- Replay value auxiliary loss (`repval`) with configurable scale.
- Continuation discount target transform (`contdisc`).
- Slow-value regularization and slow-target update (`rate/every`).
- AGC + RMS + momentum + schedule optimizer chain.
- RSSM grouped transition core, action normalization, update gate bias (`sigmoid(update - 1)`).
- Decoder `bspace` path and multi-key vector reconstruction heads.
- Replay online/uniform/recency fractions + chunking + latent start-context carry.
- Open-loop reporting hooks and optional grad-norm metrics.

## Validation and smoke checks
- `python -m crlbench validate-agent --agent dreamer --agents-dir agents` -> PASS
- Dreamer runtime smoke (discrete and continuous) -> PASS
- Open-loop reporting hook smoke -> PASS
- `pytest -q tests/test_dreamer_parity_smoke.py` -> PASS (`4 passed`)

## Intentional deviations (explicit)
- Action interface remains CRLbench single-action (`int` or continuous vector) instead of original dict-of-heads action API.
  Rationale: preserve benchmark adapter compatibility; mapping is documented in `agents/dreamer/LIFECYCLE_PARITY.md`.
- Default model size is intentionally reduced versus original DreamerV3 default preset to keep local validation/training tractable in this repo.
  Rationale: practical CI/developer resource constraints.

## Parity status
- Checklist-driven parity work is complete except for the two intentional deviations above.
