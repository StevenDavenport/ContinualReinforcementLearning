from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import MLPBlock
from .distributions import (
    DEFAULT_TWOHOT_BINS,
    DEFAULT_TWOHOT_LOG_HIGH,
    DEFAULT_TWOHOT_LOG_LOW,
    BernoulliDist,
    twohot_cross_entropy,
    twohot_mean,
)


@dataclass(frozen=True)
class RewardHeadConfig:
    feat_dim: int = 1536
    hidden_dim: int = 512
    bins: int = DEFAULT_TWOHOT_BINS
    log_low: float = DEFAULT_TWOHOT_LOG_LOW
    log_high: float = DEFAULT_TWOHOT_LOG_HIGH
    norm_eps: float = 1e-6
    zero_init_output: bool = True
    outscale: float = 1.0


@dataclass(frozen=True)
class ContinueHeadConfig:
    feat_dim: int = 1536
    hidden_dim: int = 512
    norm_eps: float = 1e-6
    zero_init_output: bool = False
    outscale: float = 1.0


class RewardHead(nn.Module):
    """
    Reward predictor head with symexp two-hot outputs.
    """

    def __init__(self, config: RewardHeadConfig) -> None:
        super().__init__()
        if config.feat_dim <= 0:
            raise ValueError(f"feat_dim must be positive, got {config.feat_dim}.")
        if config.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {config.hidden_dim}.")
        if config.bins < 2:
            raise ValueError(f"bins must be >= 2, got {config.bins}.")
        if not (config.log_low < config.log_high):
            raise ValueError(
                f"log_low must be < log_high, got {config.log_low} and {config.log_high}."
            )
        if config.outscale < 0.0:
            raise ValueError(f"outscale must be >= 0, got {config.outscale}.")

        self.config = config
        self.feat_dim = config.feat_dim
        self.backbone = MLPBlock(config.feat_dim, config.hidden_dim, eps=config.norm_eps)
        self.out = nn.Linear(config.hidden_dim, config.bins)
        if config.zero_init_output:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        elif config.outscale > 0.0:
            nn.init.uniform_(self.out.weight, -config.outscale, config.outscale)
            nn.init.zeros_(self.out.bias)
        else:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def _check_feat(self, feat: torch.Tensor) -> None:
        if feat.ndim < 2:
            raise ValueError(f"Expected rank >= 2 feature tensor (..., F), got {feat.shape}.")
        if feat.shape[-1] != self.feat_dim:
            raise ValueError(f"Expected feat dim {self.feat_dim}, got {feat.shape[-1]}.")

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        self._check_feat(feat)
        return self.out(self.backbone(feat))

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        return twohot_mean(
            logits,
            log_low=self.config.log_low,
            log_high=self.config.log_high,
        )

    def predict(self, feat: torch.Tensor) -> torch.Tensor:
        return self.mean(self.forward(feat))

    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return twohot_cross_entropy(
            logits,
            target,
            bins=self.config.bins,
            log_low=self.config.log_low,
            log_high=self.config.log_high,
        )


class ContinueHead(nn.Module):
    """
    Continue predictor head with Bernoulli logits (logistic regression loss).
    """

    def __init__(self, config: ContinueHeadConfig) -> None:
        super().__init__()
        if config.feat_dim <= 0:
            raise ValueError(f"feat_dim must be positive, got {config.feat_dim}.")
        if config.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {config.hidden_dim}.")
        if config.outscale < 0.0:
            raise ValueError(f"outscale must be >= 0, got {config.outscale}.")
        self.config = config
        self.feat_dim = config.feat_dim

        self.backbone = MLPBlock(config.feat_dim, config.hidden_dim, eps=config.norm_eps)
        self.out = nn.Linear(config.hidden_dim, 1)
        if config.zero_init_output:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        elif config.outscale > 0.0:
            nn.init.uniform_(self.out.weight, -config.outscale, config.outscale)
            nn.init.zeros_(self.out.bias)
        else:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def _check_feat(self, feat: torch.Tensor) -> None:
        if feat.ndim < 2:
            raise ValueError(f"Expected rank >= 2 feature tensor (..., F), got {feat.shape}.")
        if feat.shape[-1] != self.feat_dim:
            raise ValueError(f"Expected feat dim {self.feat_dim}, got {feat.shape[-1]}.")

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        self._check_feat(feat)
        return self.out(self.backbone(feat)).squeeze(-1)

    def dist(self, logits: torch.Tensor) -> BernoulliDist:
        return BernoulliDist(logits)

    def probs(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)

    def predict(self, feat: torch.Tensor, *, threshold: float = 0.5) -> torch.Tensor:
        if not (0.0 < threshold < 1.0):
            raise ValueError(f"threshold must be in (0,1), got {threshold}.")
        logits = self.forward(feat)
        return (self.probs(logits) >= threshold).to(dtype=logits.dtype)

    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape != target.shape:
            raise ValueError(
                f"logits and target shapes must match, got {logits.shape} and {target.shape}."
            )
        return F.binary_cross_entropy_with_logits(logits, target, reduction="none")


class ValueHead(nn.Module):
    """
    Distributional value head matching critic output parameterization.
    """

    def __init__(self, config: RewardHeadConfig) -> None:
        super().__init__()
        self._reward_like = RewardHead(config)

    @property
    def config(self) -> RewardHeadConfig:
        return self._reward_like.config

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self._reward_like(feat)

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        return self._reward_like.mean(logits)

    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._reward_like.loss(logits, target)


def infer_feat_dim(*, deter_dim: int, stoch_dim: int, classes: int) -> int:
    if deter_dim <= 0 or stoch_dim <= 0 or classes <= 1:
        raise ValueError(
            "Invalid RSSM dimensions for feature size inference: "
            f"deter_dim={deter_dim}, stoch_dim={stoch_dim}, classes={classes}."
        )
    return deter_dim + (stoch_dim * classes)
