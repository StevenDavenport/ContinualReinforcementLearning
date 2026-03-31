from __future__ import annotations

import importlib.util

import numpy as np


F32 = np.float32


# These formulas are copied from the original DreamerV3 JAX sources for use in
# a tiny, test-only reference harness.
# Source references:
# - origonal_code/embodied/jax/nets.py: symlog(), symexp()
# - origonal_code/embodied/jax/outs.py: TwoHot.pred(), TwoHot.loss()
# - origonal_code/dreamerv3/agent.py: lambda_return()


def has_jax_runtime() -> bool:
    required = ('jax',)
    return all(importlib.util.find_spec(name) is not None for name in required)


def runtime_label() -> str:
    return 'jax-runtime' if has_jax_runtime() else 'reference-formulas'


def symlog(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=F32)
    return np.sign(value) * np.log1p(np.abs(value))


def symexp(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=F32)
    return np.sign(value) * np.expm1(np.abs(value))


def original_symexp_twohot_bins(bins: int = 255) -> np.ndarray:
    if bins < 2:
        raise ValueError(f'bins must be >= 2, got {bins}.')
    if bins % 2 == 1:
        half = np.linspace(-20.0, 0.0, (bins - 1) // 2 + 1, dtype=F32)
        half = symexp(half)
        return np.concatenate([half, -half[:-1][::-1]], axis=0).astype(F32)
    half = np.linspace(-20.0, 0.0, bins // 2, dtype=F32)
    half = symexp(half)
    return np.concatenate([half, -half[::-1]], axis=0).astype(F32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=F32)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=-1, keepdims=True)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=F32)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    logsum = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return shifted - logsum


def twohot_pred(logits: np.ndarray, bins: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=F32)
    bins = np.asarray(bins, dtype=F32)
    if logits.shape[-1] != len(bins):
        raise ValueError((logits.shape, len(bins)))
    probs = _softmax(logits)
    n = logits.shape[-1]
    if n % 2 == 1:
        mid = (n - 1) // 2
        p1 = probs[..., :mid]
        p2 = probs[..., mid : mid + 1]
        p3 = probs[..., mid + 1 :]
        b1 = bins[:mid]
        b2 = bins[mid : mid + 1]
        b3 = bins[mid + 1 :]
        wavg = (p2 * b2).sum(axis=-1) + ((p1 * b1)[..., ::-1] + (p3 * b3)).sum(axis=-1)
        return wavg.astype(F32)
    p1 = probs[..., : n // 2]
    p2 = probs[..., n // 2 :]
    b1 = bins[: n // 2]
    b2 = bins[n // 2 :]
    wavg = ((p1 * b1)[..., ::-1] + (p2 * b2)).sum(axis=-1)
    return wavg.astype(F32)


def twohot_loss(logits: np.ndarray, target: np.ndarray, bins: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=F32)
    target = np.asarray(target, dtype=F32)
    bins = np.asarray(bins, dtype=F32)
    if logits.shape[:-1] != target.shape:
        raise ValueError((logits.shape, target.shape))
    below = (bins <= target[..., None]).astype(np.int32).sum(axis=-1) - 1
    above = len(bins) - (bins > target[..., None]).astype(np.int32).sum(axis=-1)
    below = np.clip(below, 0, len(bins) - 1)
    above = np.clip(above, 0, len(bins) - 1)
    equal = below == above
    dist_to_below = np.where(equal, 1.0, np.abs(bins[below] - target)).astype(F32)
    dist_to_above = np.where(equal, 1.0, np.abs(bins[above] - target)).astype(F32)
    total = dist_to_below + dist_to_above
    weight_below = dist_to_above / total
    weight_above = dist_to_below / total
    onehot_below = np.eye(len(bins), dtype=F32)[below]
    onehot_above = np.eye(len(bins), dtype=F32)[above]
    twohot = onehot_below * weight_below[..., None] + onehot_above * weight_above[..., None]
    return -(twohot * _log_softmax(logits)).sum(axis=-1).astype(F32)


def lambda_return(
    last: np.ndarray,
    term: np.ndarray,
    rew: np.ndarray,
    val: np.ndarray,
    boot: np.ndarray,
    *,
    disc: float,
    lam: float,
) -> np.ndarray:
    last = np.asarray(last, dtype=F32)
    term = np.asarray(term, dtype=F32)
    rew = np.asarray(rew, dtype=F32)
    val = np.asarray(val, dtype=F32)
    boot = np.asarray(boot, dtype=F32)
    if not (last.shape == term.shape == rew.shape == val.shape == boot.shape):
        raise ValueError((last.shape, term.shape, rew.shape, val.shape, boot.shape))
    if last.ndim != 2 or last.shape[1] < 2:
        raise ValueError(last.shape)
    live = (1.0 - term)[:, 1:] * F32(disc)
    cont = (1.0 - last)[:, 1:] * F32(lam)
    interm = rew[:, 1:] + (1.0 - cont) * live * boot[:, 1:]
    rets = [boot[:, -1]]
    for index in reversed(range(live.shape[1])):
        rets.append(interm[:, index] + live[:, index] * cont[:, index] * rets[-1])
    return np.stack(list(reversed(rets))[:-1], axis=1).astype(F32)
