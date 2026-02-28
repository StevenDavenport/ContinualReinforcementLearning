from __future__ import annotations

import importlib
from typing import Any

DEFAULT_UNIMIX = 0.01
DEFAULT_TWOHOT_BINS = 255
DEFAULT_TWOHOT_LOG_LOW = -20.0
DEFAULT_TWOHOT_LOG_HIGH = 20.0


def _import_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise ValueError(
            "dreamer requires PyTorch. Install with: "
            "python -m pip install -e '.[torch]'"
        ) from exc


def _require_float_tensor(name: str, value: Any) -> None:
    torch_mod = _import_torch()
    if not torch_mod.is_tensor(value) or not torch_mod.is_floating_point(value):
        raise TypeError(f"{name} must be a floating-point torch tensor.")


def unimix_probs(probs: Any, *, unimix: float) -> Any:
    """
    Mix a categorical distribution with a uniform prior.

    Dreamer-style unimix reduces overconfidence and stabilizes categorical training.
    """

    _require_float_tensor("probs", probs)
    if not (0.0 <= unimix < 1.0):
        raise ValueError(f"unimix must be in [0, 1), got {unimix}.")
    if probs.shape[-1] <= 1:
        raise ValueError("Categorical dimension must be > 1 for unimix.")
    if unimix == 0.0:
        return probs

    torch_mod = _import_torch()
    classes = probs.shape[-1]
    uniform = torch_mod.full_like(probs, 1.0 / float(classes))
    mixed = ((1.0 - unimix) * probs) + (unimix * uniform)
    return mixed / mixed.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def unimix_logits(logits: Any, *, unimix: float) -> Any:
    _require_float_tensor("logits", logits)
    torch_mod = _import_torch()
    probs = torch_mod.softmax(logits, dim=-1)
    mixed = unimix_probs(probs, unimix=unimix)
    return torch_mod.log(mixed.clamp_min(1e-8))


class OneHotDist:
    """
    One-hot categorical helper with optional unimix and straight-through sampling.
    """

    def __init__(self, logits: Any, *, unimix: float = DEFAULT_UNIMIX) -> None:
        _require_float_tensor("logits", logits)
        if logits.shape[-1] <= 1:
            raise ValueError("OneHotDist logits must have class dimension > 1.")

        torch_mod = _import_torch()
        self._torch = torch_mod
        self._logits = unimix_logits(logits, unimix=unimix)
        self._onehot = torch_mod.distributions.OneHotCategorical(logits=self._logits)
        self._cat = torch_mod.distributions.Categorical(logits=self._logits)

    @property
    def logits(self) -> Any:
        return self._logits

    @property
    def probs(self) -> Any:
        return self._torch.softmax(self._logits, dim=-1)

    def sample(self, *, straight_through: bool = True) -> Any:
        sample = self._onehot.sample()
        if not straight_through:
            return sample
        # Straight-through estimator for discrete actions/latents.
        return sample + self.probs - self.probs.detach()

    def mode(self) -> Any:
        indices = self._logits.argmax(dim=-1)
        return self._torch.nn.functional.one_hot(indices, self._logits.shape[-1]).to(
            dtype=self._logits.dtype
        )

    def entropy(self) -> Any:
        return self._cat.entropy()

    def log_prob(self, value: Any) -> Any:
        if self._torch.is_tensor(value):
            if (
                self._torch.is_floating_point(value)
                and value.shape == self._logits.shape
                and value.shape[-1] == self._logits.shape[-1]
            ):
                log_probs = self._torch.log_softmax(self._logits, dim=-1)
                return (value * log_probs).sum(dim=-1)
            if value.shape == self._logits.shape[:-1]:
                return self._cat.log_prob(value.to(dtype=self._torch.long))
        raise ValueError(
            "OneHotDist.log_prob expects either one-hot tensor with same shape as logits "
            "or integer class indices with shape logits.shape[:-1]."
        )


class BernoulliDist:
    """Bernoulli helper used by continuation heads."""

    def __init__(self, logits: Any) -> None:
        _require_float_tensor("logits", logits)
        torch_mod = _import_torch()
        self._torch = torch_mod
        self._dist = torch_mod.distributions.Bernoulli(logits=logits)

    @property
    def probs(self) -> Any:
        return self._dist.probs

    @property
    def logits(self) -> Any:
        return self._dist.logits

    def sample(self) -> Any:
        return self._dist.sample()

    def mode(self) -> Any:
        return (self.probs >= 0.5).to(dtype=self.probs.dtype)

    def entropy(self) -> Any:
        return self._dist.entropy()

    def log_prob(self, value: Any) -> Any:
        return self._dist.log_prob(value)


class NormalDist:
    """Diagonal Gaussian helper."""

    def __init__(self, mean: Any, std: Any, *, min_std: float = 1e-4) -> None:
        _require_float_tensor("mean", mean)
        _require_float_tensor("std", std)
        if mean.shape != std.shape:
            raise ValueError(
                f"mean and std shapes must match, got {mean.shape} vs {std.shape}."
            )
        if min_std <= 0.0:
            raise ValueError(f"min_std must be positive, got {min_std}.")
        torch_mod = _import_torch()
        self._dist = torch_mod.distributions.Normal(mean, std.clamp_min(min_std))

    @property
    def mean(self) -> Any:
        return self._dist.mean

    @property
    def stddev(self) -> Any:
        return self._dist.stddev

    def rsample(self) -> Any:
        return self._dist.rsample()

    def sample(self) -> Any:
        return self._dist.sample()

    def mode(self) -> Any:
        return self._dist.mean

    def entropy(self) -> Any:
        return self._dist.entropy().sum(dim=-1)

    def log_prob(self, value: Any) -> Any:
        return self._dist.log_prob(value).sum(dim=-1)


class TanhNormalDist:
    """
    Squashed diagonal Gaussian for bounded continuous actions in [-1, 1].
    """

    def __init__(self, mean: Any, std: Any, *, min_std: float = 1e-4) -> None:
        _require_float_tensor("mean", mean)
        _require_float_tensor("std", std)
        if mean.shape != std.shape:
            raise ValueError(
                f"mean and std shapes must match, got {mean.shape} vs {std.shape}."
            )
        if min_std <= 0.0:
            raise ValueError(f"min_std must be positive, got {min_std}.")
        torch_mod = _import_torch()
        self._torch = torch_mod
        self._base = torch_mod.distributions.Normal(mean, std.clamp_min(min_std))

    @property
    def mean(self) -> Any:
        return self._base.mean

    @property
    def stddev(self) -> Any:
        return self._base.stddev

    def mode(self) -> Any:
        return self._torch.tanh(self._base.mean)

    def rsample(self) -> Any:
        return self._torch.tanh(self._base.rsample())

    def sample(self) -> Any:
        return self._torch.tanh(self._base.sample())

    def entropy(self) -> Any:
        # Entropy after tanh is not analytic; base entropy is used as a stable proxy.
        return self._base.entropy().sum(dim=-1)

    def log_prob(self, value: Any) -> Any:
        # Inverse tanh with Jacobian correction.
        clipped = value.clamp(-0.999999, 0.999999)
        pre_tanh = 0.5 * (self._torch.log1p(clipped) - self._torch.log1p(-clipped))
        log_prob = self._base.log_prob(pre_tanh)
        log_det = self._torch.log(1.0 - clipped.square() + 1e-6)
        return (log_prob - log_det).sum(dim=-1)


def symlog(value: Any) -> Any:
    _require_float_tensor("value", value)
    torch_mod = _import_torch()
    return torch_mod.sign(value) * torch_mod.log1p(torch_mod.abs(value))


def symexp(value: Any) -> Any:
    _require_float_tensor("value", value)
    torch_mod = _import_torch()
    return torch_mod.sign(value) * torch_mod.expm1(torch_mod.abs(value))


def symexp_twohot_bins(
    *,
    bins: int = DEFAULT_TWOHOT_BINS,
    log_low: float = DEFAULT_TWOHOT_LOG_LOW,
    log_high: float = DEFAULT_TWOHOT_LOG_HIGH,
    device: Any | None = None,
    dtype: Any | None = None,
) -> Any:
    if bins < 2:
        raise ValueError(f"bins must be >= 2, got {bins}.")
    if not (log_low < log_high):
        raise ValueError(
            f"Expected log_low < log_high, got log_low={log_low}, log_high={log_high}."
        )
    torch_mod = _import_torch()
    grid = torch_mod.linspace(
        float(log_low),
        float(log_high),
        bins,
        device=device,
        dtype=dtype,
    )
    return symexp(grid)


def _resolve_support(  # noqa: PLR0913
    *,
    support: Any | None,
    bins: int,
    log_low: float,
    log_high: float,
    device: Any,
    dtype: Any,
) -> Any:
    torch_mod = _import_torch()
    if support is None:
        resolved = symexp_twohot_bins(
            bins=bins,
            log_low=log_low,
            log_high=log_high,
            device=device,
            dtype=dtype,
        )
    else:
        _require_float_tensor("support", support)
        if support.ndim != 1:
            raise ValueError(f"support must be 1D, got shape {tuple(support.shape)}.")
        if support.shape[0] < 2:
            raise ValueError("support must contain at least 2 bins.")
        resolved = support.to(device=device, dtype=dtype)

    if not bool(torch_mod.all(resolved[1:] > resolved[:-1]).item()):
        raise ValueError("support must be strictly increasing.")
    return resolved


def _ordered_sum(values: Any) -> Any:
    torch_mod = _import_torch()
    if values.shape[-1] == 0:
        return torch_mod.zeros(values.shape[:-1], dtype=values.dtype, device=values.device)
    # cumsum preserves a stable small-to-large accumulation order.
    return values.cumsum(dim=-1)[..., -1]


def twohot_encode(
    value: Any,
    *,
    support: Any | None = None,
    bins: int = DEFAULT_TWOHOT_BINS,
    log_low: float = DEFAULT_TWOHOT_LOG_LOW,
    log_high: float = DEFAULT_TWOHOT_LOG_HIGH,
) -> Any:
    """
    Two-hot encode scalar targets over a monotonic support (symexp bins by default).
    """

    _require_float_tensor("value", value)
    torch_mod = _import_torch()
    resolved_support = _resolve_support(
        support=support,
        bins=bins,
        log_low=log_low,
        log_high=log_high,
        device=value.device,
        dtype=value.dtype,
    )
    flat = value.reshape(-1).clamp(min=resolved_support[0], max=resolved_support[-1])
    upper = torch_mod.searchsorted(resolved_support, flat, right=False).clamp(
        1,
        resolved_support.shape[0] - 1,
    )
    lower = upper - 1

    lower_v = resolved_support.index_select(0, lower)
    upper_v = resolved_support.index_select(0, upper)
    denom = (upper_v - lower_v).clamp_min(1e-8)
    upper_w = ((flat - lower_v) / denom).clamp(0.0, 1.0)
    lower_w = 1.0 - upper_w

    enc = torch_mod.zeros(
        (flat.shape[0], int(resolved_support.shape[0])),
        dtype=value.dtype,
        device=value.device,
    )
    arange = torch_mod.arange(flat.shape[0], device=value.device)
    enc[arange, lower] = lower_w
    enc[arange, upper] += upper_w
    return enc.reshape(*value.shape, int(resolved_support.shape[0]))


def twohot_mean(
    logits: Any,
    *,
    support: Any | None = None,
    log_low: float = DEFAULT_TWOHOT_LOG_LOW,
    log_high: float = DEFAULT_TWOHOT_LOG_HIGH,
) -> Any:
    """
    Decode expected scalar value from two-hot logits.
    """

    _require_float_tensor("logits", logits)
    if logits.shape[-1] < 2:
        raise ValueError("twohot_mean expects at least 2 bins.")
    torch_mod = _import_torch()
    resolved_support = _resolve_support(
        support=support,
        bins=int(logits.shape[-1]),
        log_low=log_low,
        log_high=log_high,
        device=logits.device,
        dtype=logits.dtype,
    )
    if int(resolved_support.shape[0]) != int(logits.shape[-1]):
        raise ValueError(
            "support size must match logits class dimension, got "
            f"{int(resolved_support.shape[0])} vs {int(logits.shape[-1])}."
        )
    probs = torch_mod.softmax(logits, dim=-1)
    weighted = probs * resolved_support

    neg_terms = weighted[..., resolved_support < 0].flip(dims=(-1,))
    pos_terms = weighted[..., resolved_support >= 0]
    return _ordered_sum(neg_terms) + _ordered_sum(pos_terms)


def twohot_cross_entropy(  # noqa: PLR0913
    logits: Any,
    target: Any,
    *,
    support: Any | None = None,
    bins: int = DEFAULT_TWOHOT_BINS,
    log_low: float = DEFAULT_TWOHOT_LOG_LOW,
    log_high: float = DEFAULT_TWOHOT_LOG_HIGH,
) -> Any:
    """
    Categorical cross entropy against a two-hot encoded scalar target.
    """

    _require_float_tensor("logits", logits)
    _require_float_tensor("target", target)
    if logits.shape[:-1] != target.shape:
        raise ValueError(
            "target shape must match logits without class dim, got "
            f"{tuple(target.shape)} vs {tuple(logits.shape[:-1])}."
        )
    twohot = twohot_encode(
        target,
        support=support,
        bins=bins,
        log_low=log_low,
        log_high=log_high,
    )
    torch_mod = _import_torch()
    log_probs = torch_mod.log_softmax(logits, dim=-1)
    return -(twohot * log_probs).sum(dim=-1)
