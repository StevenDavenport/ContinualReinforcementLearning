from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import MLPBlock
from .distributions import DEFAULT_UNIMIX


@dataclass(frozen=True)
class ActorConfig:
    feat_dim: int = 1536
    hidden_dim: int = 512
    hidden_layers: int = 3
    action_dim: int = 5
    action_space: str = "discrete"  # {"discrete", "continuous"}
    unimix: float = DEFAULT_UNIMIX
    min_std: float = 0.1
    max_std: float = 1.0
    outscale: float = 0.01
    norm_eps: float = 1e-6


class Actor(nn.Module):
    """
    Dreamer actor head.

    - Discrete: categorical policy logits.
    - Continuous: Dreamer bounded-normal (tanh mean, bounded std).
    """

    def __init__(self, config: ActorConfig) -> None:
        super().__init__()
        if config.feat_dim <= 0:
            raise ValueError(f"feat_dim must be positive, got {config.feat_dim}.")
        if config.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {config.hidden_dim}.")
        if config.hidden_layers <= 0:
            raise ValueError(f"hidden_layers must be positive, got {config.hidden_layers}.")
        if config.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {config.action_dim}.")
        if config.action_space not in {"discrete", "continuous"}:
            raise ValueError(
                f"action_space must be 'discrete' or 'continuous', got {config.action_space!r}."
            )
        if not (0.0 <= config.unimix < 1.0):
            raise ValueError(f"unimix must be in [0, 1), got {config.unimix}.")
        if config.min_std <= 0.0:
            raise ValueError(f"min_std must be positive, got {config.min_std}.")
        if config.max_std <= config.min_std:
            raise ValueError(
                f"max_std must be > min_std, got min_std={config.min_std}, "
                f"max_std={config.max_std}."
            )

        self.config = config
        self.feat_dim = config.feat_dim
        self.action_dim = config.action_dim
        self.action_space = config.action_space
        self.unimix = config.unimix
        self.min_std = config.min_std
        self.max_std = config.max_std

        blocks: list[nn.Module] = []
        in_dim = config.feat_dim
        for _ in range(config.hidden_layers):
            blocks.append(MLPBlock(in_dim, config.hidden_dim, eps=config.norm_eps))
            in_dim = config.hidden_dim
        self.backbone = nn.Sequential(*blocks)

        if self.action_space == "discrete":
            self.logits_head: nn.Module = nn.Linear(config.hidden_dim, config.action_dim)
            self.mean_head = None
            self.std_head = None
        else:
            self.logits_head = nn.Identity()
            self.mean_head = nn.Linear(config.hidden_dim, config.action_dim)
            self.std_head = nn.Linear(config.hidden_dim, config.action_dim)
        self._init_output_scale(config.outscale)

    def _init_output_scale(self, outscale: float) -> None:
        if outscale <= 0.0:
            raise ValueError(f"outscale must be positive, got {outscale}.")
        if self.action_space == "discrete":
            assert isinstance(self.logits_head, nn.Linear)
            nn.init.uniform_(self.logits_head.weight, -outscale, outscale)
            nn.init.zeros_(self.logits_head.bias)
            return
        assert self.mean_head is not None and self.std_head is not None
        nn.init.uniform_(self.mean_head.weight, -outscale, outscale)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.uniform_(self.std_head.weight, -outscale, outscale)
        nn.init.zeros_(self.std_head.bias)

    def _check_feat(self, feat: torch.Tensor) -> None:
        if feat.ndim < 2:
            raise ValueError(f"Expected rank >= 2 feature tensor (..., F), got {feat.shape}.")
        if feat.shape[-1] != self.feat_dim:
            raise ValueError(f"Expected feat dim {self.feat_dim}, got {feat.shape[-1]}.")

    def _dist(self, feat: torch.Tensor) -> torch.distributions.Categorical | torch.distributions.Normal:
        self._check_feat(feat)
        hidden = self.backbone(feat)
        if self.action_space == "discrete":
            logits = self.logits_head(hidden)
            assert isinstance(logits, torch.Tensor)
            return torch.distributions.Categorical(logits=logits)

        assert self.mean_head is not None
        assert self.std_head is not None
        mean = torch.tanh(self.mean_head(hidden))
        raw_std = self.std_head(hidden)
        std = ((self.max_std - self.min_std) * torch.sigmoid(raw_std + 2.0)) + self.min_std
        return torch.distributions.Normal(mean, std)

    def forward(self, feat: torch.Tensor) -> torch.distributions.Categorical | torch.distributions.Normal:
        return self._dist(feat)

    def sample(
        self,
        feat: torch.Tensor,
        *,
        deterministic: bool = False,
        straight_through: bool = True,
    ) -> torch.Tensor:
        dist = self._dist(feat)
        if self.action_space == "discrete":
            _ = straight_through
            assert isinstance(dist, torch.distributions.Categorical)
            if deterministic:
                logits = dist.logits
                return logits.argmax(dim=-1).to(dtype=torch.long)
            return dist.sample().to(dtype=torch.long)
        assert isinstance(dist, torch.distributions.Normal)
        if deterministic:
            return dist.mean
        return dist.sample()

    def log_prob(self, feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        dist = self._dist(feat)
        if self.action_space == "discrete":
            assert isinstance(dist, torch.distributions.Categorical)
            if action.shape == dist.logits.shape:
                action = action.argmax(dim=-1)
            return dist.log_prob(action.to(dtype=torch.long))
        assert isinstance(dist, torch.distributions.Normal)
        return dist.log_prob(action).sum(dim=-1)

    def entropy(self, feat: torch.Tensor) -> torch.Tensor:
        dist = self._dist(feat)
        if self.action_space == "discrete":
            assert isinstance(dist, torch.distributions.Categorical)
            return dist.entropy()
        assert isinstance(dist, torch.distributions.Normal)
        return dist.entropy().sum(dim=-1)

    def policy_terms(
        self,
        feat: torch.Tensor,
        *,
        deterministic: bool = False,
        straight_through: bool = True,
    ) -> dict[str, torch.Tensor]:
        dist = self._dist(feat)
        if self.action_space == "discrete":
            assert isinstance(dist, torch.distributions.Categorical)
            if deterministic:
                logits = dist.logits
                action_index = logits.argmax(dim=-1).to(dtype=torch.long)
            else:
                _ = straight_through
                action_index = dist.sample().to(dtype=torch.long)
            action_model = F.one_hot(action_index, self.action_dim).to(dtype=feat.dtype)
            log_prob = dist.log_prob(action_index)
            entropy = dist.entropy()
            action_out = action_index
        else:
            assert isinstance(dist, torch.distributions.Normal)
            action = dist.mean if deterministic else dist.sample()
            action_out = action
            action_model = action
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        return {
            "action": action_out,
            "model_action": action_model,
            "log_prob": log_prob,
            "entropy": entropy,
        }

    def to_env_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        Convert model action tensors to environment-facing tensors.
        """

        if self.action_space == "discrete":
            if action.ndim == 0:
                return action.to(dtype=torch.long)
            if action.shape[-1] == self.action_dim:
                return action.argmax(dim=-1).to(dtype=torch.long)
            return action.to(dtype=torch.long)
        return action.clamp(-1.0, 1.0)
