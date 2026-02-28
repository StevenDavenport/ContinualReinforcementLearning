from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .networks.distributions import (
    DEFAULT_TWOHOT_BINS,
    DEFAULT_TWOHOT_LOG_HIGH,
    DEFAULT_TWOHOT_LOG_LOW,
    symlog,
    twohot_cross_entropy,
    twohot_mean,
)


def _ensure_float_tensor(name: str, value: torch.Tensor) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be a floating-point tensor, got {value.dtype}.")


def _match_tensor(target: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return target.to(device=reference.device, dtype=reference.dtype)


def _symlog_squared(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _ensure_float_tensor("prediction", prediction)
    _ensure_float_tensor("target", target)
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes must match, got {prediction.shape} and {target.shape}."
        )
    error = prediction - target
    return error.square()


def categorical_kl(lhs_logits: torch.Tensor, rhs_logits: torch.Tensor) -> torch.Tensor:
    _ensure_float_tensor("lhs_logits", lhs_logits)
    _ensure_float_tensor("rhs_logits", rhs_logits)
    if lhs_logits.shape != rhs_logits.shape:
        raise ValueError(
            f"lhs_logits and rhs_logits shapes must match, got {lhs_logits.shape} and "
            f"{rhs_logits.shape}."
        )
    if lhs_logits.ndim < 2:
        raise ValueError(
            "categorical_kl expects rank >= 2 with tail [stoch_dim, classes], got "
            f"{lhs_logits.shape}."
        )
    lhs_log_probs = torch.log_softmax(lhs_logits, dim=-1)
    rhs_log_probs = torch.log_softmax(rhs_logits, dim=-1)
    lhs_probs = torch.softmax(lhs_logits, dim=-1)
    return (lhs_probs * (lhs_log_probs - rhs_log_probs)).sum(dim=-1).sum(dim=-1)


def balanced_kl_loss(
    posterior_logits: torch.Tensor,
    prior_logits: torch.Tensor,
    *,
    free_nats: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if free_nats <= 0.0:
        raise ValueError(f"free_nats must be positive, got {free_nats}.")
    dyn = categorical_kl(posterior_logits.detach(), prior_logits)
    rep = categorical_kl(posterior_logits, prior_logits.detach())
    floor = torch.full_like(dyn, free_nats)
    dyn = torch.maximum(dyn, floor)
    rep = torch.maximum(rep, floor)
    return dyn, rep


@dataclass(frozen=True)
class PredictionLossConfig:
    image_scale: float = 1.0
    vector_scale: float = 1.0
    reward_scale: float = 1.0
    continue_scale: float = 1.0
    reward_bins: int = DEFAULT_TWOHOT_BINS
    reward_log_low: float = DEFAULT_TWOHOT_LOG_LOW
    reward_log_high: float = DEFAULT_TWOHOT_LOG_HIGH
    vector_key: str = "proprio"
    vector_keys: tuple[str, ...] = ()
    apply_symlog_to_vector_targets: bool = False


@dataclass(frozen=True)
class WorldModelLossConfig:
    beta_pred: float = 1.0
    beta_dyn: float = 1.0
    beta_rep: float = 0.1
    free_nats: float = 1.0
    prediction: PredictionLossConfig = field(default_factory=PredictionLossConfig)


@dataclass(frozen=True)
class NormalizeConfig:
    impl: str = "none"
    rate: float = 0.01
    limit: float = 1e-8
    perclo: float = 5.0
    perchi: float = 95.0
    debias: bool = True


class Normalize:
    """
    Torch equivalent of embodied.jax.utils.Normalize.
    """

    def __init__(self, config: NormalizeConfig) -> None:
        if config.impl not in {"none", "meanstd", "perc"}:
            raise ValueError(f"Unsupported normalization impl: {config.impl!r}.")
        if not (0.0 < config.rate <= 1.0):
            raise ValueError(f"rate must be in (0, 1], got {config.rate}.")
        if config.limit <= 0.0:
            raise ValueError(f"limit must be > 0, got {config.limit}.")
        if not (0.0 <= config.perclo < config.perchi <= 100.0):
            raise ValueError(
                f"Expected 0 <= perclo < perchi <= 100, got {config.perclo}, {config.perchi}."
            )

        self.config = config
        self._mean = 0.0
        self._sqrs = 0.0
        self._lo = 0.0
        self._hi = 0.0
        self._corr = 0.0

    def __call__(self, value: torch.Tensor, update: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if update:
            self.update(value)
        return self.stats(device=value.device, dtype=value.dtype)

    def update(self, value: torch.Tensor) -> None:
        x = value.detach().to(dtype=torch.float32)
        impl = self.config.impl
        if impl == "none":
            return
        if impl == "meanstd":
            self._mean = self._ema(self._mean, float(x.mean().item()))
            self._sqrs = self._ema(self._sqrs, float(x.square().mean().item()))
        elif impl == "perc":
            self._lo = self._ema(self._lo, float(torch.quantile(x, self.config.perclo / 100.0).item()))
            self._hi = self._ema(self._hi, float(torch.quantile(x, self.config.perchi / 100.0).item()))
        if self.config.debias:
            self._corr = self._ema(self._corr, 1.0)

    def stats(self, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        impl = self.config.impl
        if impl == "none":
            offset = torch.zeros((), device=device, dtype=dtype)
            scale = torch.ones((), device=device, dtype=dtype)
            return offset, scale

        corr = 1.0
        if self.config.debias:
            corr = 1.0 / max(self.config.rate, self._corr)

        if impl == "meanstd":
            mean = self._mean * corr
            var = max(0.0, (self._sqrs * corr) - (mean * mean))
            std = max(self.config.limit, var**0.5)
            return (
                torch.tensor(mean, device=device, dtype=dtype),
                torch.tensor(std, device=device, dtype=dtype),
            )

        lo = self._lo * corr
        hi = self._hi * corr
        scale = max(self.config.limit, hi - lo)
        return (
            torch.tensor(lo, device=device, dtype=dtype).detach(),
            torch.tensor(scale, device=device, dtype=dtype).detach(),
        )

    def _ema(self, old: float, new: float) -> float:
        return ((1.0 - self.config.rate) * old) + (self.config.rate * new)


@dataclass(frozen=True)
class ActorLossConfig:
    contdisc: bool = True
    horizon: int = 333
    lam: float = 0.95
    actent: float = 3e-4
    slowreg: float = 1.0
    slowtar: bool = False


@dataclass(frozen=True)
class CriticLossConfig:
    bins: int = DEFAULT_TWOHOT_BINS
    log_low: float = DEFAULT_TWOHOT_LOG_LOW
    log_high: float = DEFAULT_TWOHOT_LOG_HIGH
    lam: float = 0.95
    slowreg: float = 1.0
    slowtar: bool = False
    horizon: int = 333
    repval_scale: float = 0.3


def _pixel_mse_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _ensure_float_tensor("predicted_pixels", predicted)
    _ensure_float_tensor("target_pixels", target)
    if predicted.shape != target.shape:
        raise ValueError(
            f"predicted and target pixel shapes must match, got {predicted.shape} and "
            f"{target.shape}."
        )
    if predicted.ndim < 3:
        raise ValueError(
            "predicted pixels must have at least 3 event dims (..., H, W, C), got "
            f"{predicted.shape}."
        )
    return (predicted - target).square().sum(dim=(-1, -2, -3))


def _vector_symlog_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    symlog_target: bool,
) -> torch.Tensor:
    _ensure_float_tensor("predicted_vector", predicted)
    _ensure_float_tensor("target_vector", target)
    target_value = symlog(target) if symlog_target else target
    if predicted.shape != target_value.shape:
        raise ValueError(
            f"predicted and target vector shapes must match, got {predicted.shape} and "
            f"{target_value.shape}."
        )
    return _symlog_squared(predicted, target_value).sum(dim=-1)


def world_model_prediction_losses(  # noqa: PLR0913
    *,
    decoded: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    reward_logits: torch.Tensor,
    reward_target: torch.Tensor,
    continue_logits: torch.Tensor,
    continue_target: torch.Tensor,
    config: PredictionLossConfig,
) -> dict[str, torch.Tensor]:
    _ensure_float_tensor("reward_logits", reward_logits)
    _ensure_float_tensor("reward_target", reward_target)
    _ensure_float_tensor("continue_logits", continue_logits)
    _ensure_float_tensor("continue_target", continue_target)

    if reward_logits.shape[:-1] != reward_target.shape:
        raise ValueError(
            "reward_target shape must match reward_logits without class dim, got "
            f"{reward_target.shape} vs {reward_logits.shape[:-1]}."
        )
    if continue_logits.shape != continue_target.shape:
        raise ValueError(
            f"continue_logits and continue_target shapes must match, got "
            f"{continue_logits.shape} and {continue_target.shape}."
        )

    zero = torch.zeros((), device=reward_logits.device, dtype=reward_logits.dtype)

    image_loss = zero
    if "pixels" in decoded and "pixels" in targets:
        pred_pixels = _match_tensor(decoded["pixels"], reward_logits)
        target_pixels = _match_tensor(targets["pixels"], reward_logits)
        if float(target_pixels.max().item()) > 1.0:
            target_pixels = target_pixels / 255.0
        image_loss = _pixel_mse_loss(pred_pixels, target_pixels).mean()

    vector_loss = zero
    if config.vector_keys:
        vector_keys = tuple(config.vector_keys)
    else:
        vector_keys = tuple(
            key for key in decoded.keys() if key != "pixels" and key in targets
        )
        if not vector_keys and config.vector_key in decoded and config.vector_key in targets:
            vector_keys = (config.vector_key,)
    for key in vector_keys:
        pred_vector = _match_tensor(decoded[key], reward_logits)
        target_vector = _match_tensor(targets[key], reward_logits)
        vector_loss = vector_loss + _vector_symlog_loss(
            pred_vector,
            target_vector,
            symlog_target=config.apply_symlog_to_vector_targets,
        ).mean()

    reward_loss = twohot_cross_entropy(
        reward_logits,
        _match_tensor(reward_target, reward_logits),
        bins=config.reward_bins,
        log_low=config.reward_log_low,
        log_high=config.reward_log_high,
    ).mean()
    continue_loss = F.binary_cross_entropy_with_logits(
        continue_logits,
        _match_tensor(continue_target, continue_logits),
        reduction="none",
    ).mean()

    total = (
        (config.image_scale * image_loss)
        + (config.vector_scale * vector_loss)
        + (config.reward_scale * reward_loss)
        + (config.continue_scale * continue_loss)
    )
    return {
        "pred_total": total,
        "pred_image": image_loss,
        "pred_vector": vector_loss,
        "pred_reward": reward_loss,
        "pred_continue": continue_loss,
    }


def world_model_loss(  # noqa: PLR0913
    *,
    decoded: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    reward_logits: torch.Tensor,
    reward_target: torch.Tensor,
    continue_logits: torch.Tensor,
    continue_target: torch.Tensor,
    posterior_logits: torch.Tensor,
    prior_logits: torch.Tensor,
    config: WorldModelLossConfig,
) -> dict[str, torch.Tensor]:
    pred_metrics = world_model_prediction_losses(
        decoded=decoded,
        targets=targets,
        reward_logits=reward_logits,
        reward_target=reward_target,
        continue_logits=continue_logits,
        continue_target=continue_target,
        config=config.prediction,
    )
    dyn_kl, rep_kl = balanced_kl_loss(
        posterior_logits,
        prior_logits,
        free_nats=config.free_nats,
    )
    dyn = dyn_kl.mean()
    rep = rep_kl.mean()
    total = (
        (config.beta_pred * pred_metrics["pred_total"])
        + (config.beta_dyn * dyn)
        + (config.beta_rep * rep)
    )
    return {
        **pred_metrics,
        "kl_dyn": dyn,
        "kl_rep": rep,
        "world_model_total": total,
    }


def lambda_return(
    last: torch.Tensor,
    term: torch.Tensor,
    rew: torch.Tensor,
    val: torch.Tensor,
    boot: torch.Tensor,
    *,
    disc: float,
    lam: float,
) -> torch.Tensor:
    """
    Matches dreamerv3.agent.lambda_return() semantics.

    Inputs use shape [B, T]. Output has shape [B, T - 1].
    """

    for name, tensor in {
        "last": last,
        "term": term,
        "rew": rew,
        "val": val,
        "boot": boot,
    }.items():
        _ensure_float_tensor(name, tensor)

    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"lam must be in [0, 1], got {lam}.")
    if disc <= 0.0:
        raise ValueError(f"disc must be > 0, got {disc}.")
    if not (last.shape == term.shape == rew.shape == val.shape == boot.shape):
        raise ValueError(
            "Expected matching [B, T] shapes, got "
            f"last={last.shape}, term={term.shape}, rew={rew.shape}, val={val.shape}, boot={boot.shape}."
        )
    if last.ndim != 2:
        raise ValueError(f"lambda_return expects rank-2 [B,T] tensors, got {last.shape}.")
    if last.shape[1] < 2:
        raise ValueError(f"lambda_return expects T >= 2, got shape {last.shape}.")

    live = (1.0 - term)[:, 1:] * disc
    cont = (1.0 - last)[:, 1:] * lam
    interm = rew[:, 1:] + ((1.0 - cont) * live * boot[:, 1:])

    rets: list[torch.Tensor] = [boot[:, -1]]
    for index in range(int(live.shape[1]) - 1, -1, -1):
        rets.append(interm[:, index] + (live[:, index] * cont[:, index] * rets[-1]))

    result = torch.stack(list(reversed(rets)), dim=1)
    return result[:, :-1]


def imag_loss(  # noqa: PLR0913
    *,
    act_logprob: torch.Tensor,
    act_entropy: torch.Tensor,
    act_sample: torch.Tensor,
    rew: torch.Tensor,
    con: torch.Tensor,
    value_logits: torch.Tensor,
    slowvalue_logits: torch.Tensor,
    retnorm: Normalize,
    valnorm: Normalize,
    advnorm: Normalize,
    actor_config: ActorLossConfig,
    critic_config: CriticLossConfig,
    update: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    for name, tensor in {
        "act_logprob": act_logprob,
        "act_entropy": act_entropy,
        "act_sample": act_sample,
        "rew": rew,
        "con": con,
    }.items():
        _ensure_float_tensor(name, tensor)

    if act_logprob.shape != act_entropy.shape:
        raise ValueError(
            f"act_logprob and act_entropy shapes must match, got {act_logprob.shape} and {act_entropy.shape}."
        )
    if act_logprob.shape != rew.shape or act_logprob.shape != con.shape:
        raise ValueError(
            f"Expected act/rew/con to match, got {act_logprob.shape}, {rew.shape}, {con.shape}."
        )
    if value_logits.shape[:-1] != rew.shape:
        raise ValueError(
            f"value_logits leading shape must match rew, got {value_logits.shape} and {rew.shape}."
        )
    if slowvalue_logits.shape != value_logits.shape:
        raise ValueError(
            f"slowvalue_logits and value_logits must match, got {slowvalue_logits.shape} and {value_logits.shape}."
        )

    voffset, vscale = valnorm.stats(device=rew.device, dtype=rew.dtype)
    val = twohot_mean(value_logits, log_low=critic_config.log_low, log_high=critic_config.log_high)
    slowval = twohot_mean(
        slowvalue_logits,
        log_low=critic_config.log_low,
        log_high=critic_config.log_high,
    )
    val = (val * vscale) + voffset
    slowval = (slowval * vscale) + voffset
    tarval = slowval if actor_config.slowtar else val

    disc = 1.0 if actor_config.contdisc else (1.0 - (1.0 / float(actor_config.horizon)))
    disc_tensor = torch.tensor(disc, device=con.device, dtype=con.dtype)
    weight = torch.cumprod(disc_tensor * con, dim=1) / disc_tensor

    last = torch.zeros_like(con)
    term = 1.0 - con
    ret = lambda_return(last, term, rew, tarval, tarval, disc=disc, lam=actor_config.lam)

    roffset, rscale = retnorm(ret, update=update)
    adv = (ret - tarval[:, :-1]) / rscale
    aoffset, ascale = advnorm(adv, update=update)
    adv_normed = (adv - aoffset) / ascale

    logpi = act_logprob[:, :-1]
    ents = act_entropy[:, :-1]
    policy_loss = weight[:, :-1].detach() * -(
        logpi * adv_normed.detach() + (actor_config.actent * ents)
    )

    voffset, vscale = valnorm(ret, update=update)
    tar_normed = (ret - voffset) / vscale
    tar_padded = torch.cat([tar_normed, torch.zeros_like(tar_normed[:, -1:])], dim=1)

    value_main = twohot_cross_entropy(
        value_logits,
        tar_padded.detach(),
        bins=critic_config.bins,
        log_low=critic_config.log_low,
        log_high=critic_config.log_high,
    )
    value_slow = twohot_cross_entropy(
        value_logits,
        twohot_mean(
            slowvalue_logits,
            log_low=critic_config.log_low,
            log_high=critic_config.log_high,
        ).detach(),
        bins=critic_config.bins,
        log_low=critic_config.log_low,
        log_high=critic_config.log_high,
    )
    value_loss = weight[:, :-1].detach() * (
        value_main + (actor_config.slowreg * value_slow)
    )[:, :-1]

    losses = {
        "policy": policy_loss,
        "value": value_loss,
    }
    outs = {
        "ret": ret,
        "weight": weight,
        "value": val,
        "slowval": slowval,
        "sample": act_sample,
    }

    ret_normed = (ret - roffset) / rscale
    metrics: dict[str, torch.Tensor] = {
        "adv": adv.mean(),
        "adv_std": adv.std(),
        "adv_mag": adv.abs().mean(),
        "rew": rew.mean(),
        "con": con.mean(),
        "ret": ret_normed.mean(),
        "val": val.mean(),
        "tar": tar_normed.mean(),
        "weight": weight.mean(),
        "slowval": slowval.mean(),
        "ret_min": ret_normed.min(),
        "ret_max": ret_normed.max(),
        "ret_rate": (ret_normed.abs() >= 1.0).to(dtype=ret_normed.dtype).mean(),
        "entropy": ents.mean(),
    }
    return losses, outs, metrics


def replay_value_loss(  # noqa: PLR0913
    *,
    last: torch.Tensor,
    term: torch.Tensor,
    rew: torch.Tensor,
    boot: torch.Tensor,
    value_logits: torch.Tensor,
    slowvalue_logits: torch.Tensor,
    valnorm: Normalize,
    config: CriticLossConfig,
    update: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    for name, tensor in {
        "last": last,
        "term": term,
        "rew": rew,
        "boot": boot,
    }.items():
        _ensure_float_tensor(name, tensor)

    if value_logits.shape[:-1] != last.shape:
        raise ValueError(
            f"value_logits leading shape must match last shape, got {value_logits.shape} and {last.shape}."
        )
    if slowvalue_logits.shape != value_logits.shape:
        raise ValueError(
            f"slowvalue_logits and value_logits must match, got {slowvalue_logits.shape} and {value_logits.shape}."
        )

    voffset, vscale = valnorm.stats(device=rew.device, dtype=rew.dtype)
    val = twohot_mean(value_logits, log_low=config.log_low, log_high=config.log_high)
    slowval = twohot_mean(slowvalue_logits, log_low=config.log_low, log_high=config.log_high)
    val = (val * vscale) + voffset
    slowval = (slowval * vscale) + voffset
    tarval = slowval if config.slowtar else val

    disc = 1.0 - (1.0 / float(config.horizon))
    weight = 1.0 - last
    ret = lambda_return(last, term, rew, tarval, boot, disc=disc, lam=config.lam)

    voffset, vscale = valnorm(ret, update=update)
    ret_normed = (ret - voffset) / vscale
    ret_padded = torch.cat([ret_normed, torch.zeros_like(ret_normed[:, -1:])], dim=1)

    value_main = twohot_cross_entropy(
        value_logits,
        ret_padded.detach(),
        bins=config.bins,
        log_low=config.log_low,
        log_high=config.log_high,
    )
    value_slow = twohot_cross_entropy(
        value_logits,
        twohot_mean(slowvalue_logits, log_low=config.log_low, log_high=config.log_high).detach(),
        bins=config.bins,
        log_low=config.log_low,
        log_high=config.log_high,
    )

    repval = weight[:, :-1] * (value_main + (config.slowreg * value_slow))[:, :-1]
    losses = {"repval": repval}
    outs = {"ret": ret}
    metrics: dict[str, torch.Tensor] = {}
    return losses, outs, metrics


def critic_loss(
    *,
    value_logits: torch.Tensor,
    target_returns: torch.Tensor,
    config: CriticLossConfig,
) -> dict[str, torch.Tensor]:
    _ensure_float_tensor("value_logits", value_logits)
    _ensure_float_tensor("target_returns", target_returns)
    if value_logits.shape[:-1] != target_returns.shape:
        raise ValueError(
            "target_returns shape must match value_logits without class dim, got "
            f"{target_returns.shape} vs {value_logits.shape[:-1]}."
        )

    ce = twohot_cross_entropy(
        value_logits,
        target_returns,
        bins=config.bins,
        log_low=config.log_low,
        log_high=config.log_high,
    ).mean()
    predicted = twohot_mean(
        value_logits,
        log_low=config.log_low,
        log_high=config.log_high,
    )
    mae = (predicted - target_returns).abs().mean()
    return {
        "critic_loss": ce,
        "critic_mae": mae.detach(),
        "critic_value_mean": predicted.mean().detach(),
    }


def combined_critic_loss(
    *,
    imagined_critic_loss: torch.Tensor,
    replay_critic_loss: torch.Tensor | None = None,
    replay_scale: float = 0.3,
) -> torch.Tensor:
    _ensure_float_tensor("imagined_critic_loss", imagined_critic_loss)
    if replay_critic_loss is None:
        return imagined_critic_loss
    _ensure_float_tensor("replay_critic_loss", replay_critic_loss)
    if replay_scale < 0.0:
        raise ValueError(f"replay_scale must be >= 0, got {replay_scale}.")
    return imagined_critic_loss + (replay_scale * replay_critic_loss)
