from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn

from .blocks import MLPBlock
from .distributions import (
    DEFAULT_TWOHOT_BINS,
    DEFAULT_TWOHOT_LOG_HIGH,
    DEFAULT_TWOHOT_LOG_LOW,
    twohot_cross_entropy,
    twohot_mean,
)


@dataclass(frozen=True)
class CriticConfig:
    feat_dim: int = 1536
    hidden_dim: int = 512
    hidden_layers: int = 3
    bins: int = DEFAULT_TWOHOT_BINS
    log_low: float = DEFAULT_TWOHOT_LOG_LOW
    log_high: float = DEFAULT_TWOHOT_LOG_HIGH
    norm_eps: float = 1e-6
    zero_init_output: bool = True
    outscale: float = 1.0


class Critic(nn.Module):
    """
    Distributional value network with symexp two-hot support.
    """

    def __init__(self, config: CriticConfig) -> None:
        super().__init__()
        if config.feat_dim <= 0:
            raise ValueError(f"feat_dim must be positive, got {config.feat_dim}.")
        if config.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {config.hidden_dim}.")
        if config.hidden_layers <= 0:
            raise ValueError(f"hidden_layers must be positive, got {config.hidden_layers}.")
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
        self.bins = config.bins

        blocks: list[nn.Module] = []
        in_dim = config.feat_dim
        for _ in range(config.hidden_layers):
            blocks.append(MLPBlock(in_dim, config.hidden_dim, eps=config.norm_eps))
            in_dim = config.hidden_dim
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(config.hidden_dim, config.bins)
        if config.zero_init_output:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        elif config.outscale > 0.0:
            nn.init.uniform_(self.head.weight, -config.outscale, config.outscale)
            nn.init.zeros_(self.head.bias)
        else:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def _check_feat(self, feat: torch.Tensor) -> None:
        if feat.ndim < 2:
            raise ValueError(f"Expected rank >= 2 feature tensor (..., F), got {feat.shape}.")
        if feat.shape[-1] != self.feat_dim:
            raise ValueError(f"Expected feat dim {self.feat_dim}, got {feat.shape[-1]}.")

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        self._check_feat(feat)
        return self.head(self.backbone(feat))

    def value(self, feat: torch.Tensor) -> torch.Tensor:
        logits = self.forward(feat)
        return twohot_mean(
            logits,
            log_low=self.config.log_low,
            log_high=self.config.log_high,
        )

    def loss(self, logits: torch.Tensor, target_returns: torch.Tensor) -> torch.Tensor:
        if logits.shape[:-1] != target_returns.shape:
            raise ValueError(
                "target_returns shape must match logits without class dim, got "
                f"{target_returns.shape} and {logits.shape[:-1]}."
            )
        return twohot_cross_entropy(
            logits,
            target_returns,
            bins=self.config.bins,
            log_low=self.config.log_low,
            log_high=self.config.log_high,
        )


class SlowCritic:
    """
    DreamerV3-style slow target model with configurable update rate and period.
    """

    def __init__(self, critic: Critic, *, rate: float = 0.02, every: int = 1) -> None:
        if not (0.0 < rate <= 1.0):
            raise ValueError(f"rate must be in (0, 1], got {rate}.")
        if rate != 1.0 and rate >= 0.5:
            raise ValueError(f"Expected rate==1.0 or rate<0.5, got {rate}.")
        if every <= 0:
            raise ValueError(f"every must be positive, got {every}.")
        self.rate = rate
        self.every = every
        self.count = 0
        self.model = copy.deepcopy(critic).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, critic: Critic) -> None:
        mix = self.rate if (self.count % self.every == 0) else 0.0
        source = dict(critic.named_parameters())
        target = dict(self.model.named_parameters())
        if source.keys() != target.keys():
            raise ValueError("Critic parameter sets do not match for slow update.")
        for name, target_param in target.items():
            source_param = source[name]
            target_param.data.mul_(1.0 - mix).add_(source_param.data, alpha=mix)

        source_buffers = dict(critic.named_buffers())
        target_buffers = dict(self.model.named_buffers())
        if source_buffers.keys() != target_buffers.keys():
            raise ValueError("Critic buffer sets do not match for slow update.")
        for name, target_buffer in target_buffers.items():
            target_buffer.copy_(source_buffers[name])
        self.count += 1

    def logits(self, feat: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(feat)

    def value(self, feat: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model.value(feat)


class CriticEMA(SlowCritic):
    """
    Backward-compatible alias. `decay=0.99` maps to `rate=0.01`.
    """

    def __init__(self, critic: Critic, *, decay: float = 0.99) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1), got {decay}.")
        super().__init__(critic, rate=1.0 - decay, every=1)
