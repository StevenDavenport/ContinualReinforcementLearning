from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import BlockLinear, RMSNorm
from .distributions import DEFAULT_UNIMIX, OneHotDist


@dataclass(frozen=True)
class RSSMConfig:
    """
    DreamerV3 RSSM core with grouped recurrent transition.
    """

    deter_dim: int = 512
    stoch_dim: int = 32
    classes: int = 32
    hidden_dim: int = 512
    action_dim: int = 16
    embed_dim: int = 1024
    blocks: int = 8
    unimix: float = DEFAULT_UNIMIX
    norm_eps: float = 1e-6
    img_layers: int = 2
    obs_layers: int = 1
    dyn_layers: int = 1
    absolute: bool = False
    outscale: float = 1.0


@dataclass(frozen=True)
class RSSMState:
    deter: torch.Tensor
    stoch: torch.Tensor
    logits: torch.Tensor


class RSSM(nn.Module):
    def __init__(self, config: RSSMConfig) -> None:
        super().__init__()
        if config.deter_dim % config.blocks != 0:
            raise ValueError(
                f"deter_dim={config.deter_dim} must be divisible by blocks={config.blocks}."
            )
        if config.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {config.hidden_dim}.")
        if config.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {config.embed_dim}.")
        if config.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {config.action_dim}.")
        if config.stoch_dim <= 0:
            raise ValueError(f"stoch_dim must be positive, got {config.stoch_dim}.")
        if config.classes <= 1:
            raise ValueError(f"classes must be > 1, got {config.classes}.")
        if config.blocks <= 0:
            raise ValueError(f"blocks must be positive, got {config.blocks}.")
        if not (0.0 <= config.unimix < 1.0):
            raise ValueError(f"unimix must be in [0,1), got {config.unimix}.")
        if config.img_layers <= 0:
            raise ValueError(f"img_layers must be positive, got {config.img_layers}.")
        if config.obs_layers <= 0:
            raise ValueError(f"obs_layers must be positive, got {config.obs_layers}.")
        if config.dyn_layers <= 0:
            raise ValueError(f"dyn_layers must be positive, got {config.dyn_layers}.")
        if config.outscale <= 0.0:
            raise ValueError(f"outscale must be positive, got {config.outscale}.")

        self.config = config
        self.deter_dim = config.deter_dim
        self.stoch_dim = config.stoch_dim
        self.classes = config.classes
        self.action_dim = config.action_dim
        self.embed_dim = config.embed_dim
        self.hidden_dim = config.hidden_dim
        self.unimix = config.unimix
        self.blocks = config.blocks
        self._deter_per_block = self.deter_dim // self.blocks
        self._flat_stoch_dim = self.stoch_dim * self.classes

        # Dynamic core projections.
        self.dyn_in_deter = nn.Linear(self.deter_dim, self.hidden_dim)
        self.dyn_in_deter_norm = RMSNorm(self.hidden_dim, eps=config.norm_eps)
        self.dyn_in_stoch = nn.Linear(self._flat_stoch_dim, self.hidden_dim)
        self.dyn_in_stoch_norm = RMSNorm(self.hidden_dim, eps=config.norm_eps)
        self.dyn_in_action = nn.Linear(self.action_dim, self.hidden_dim)
        self.dyn_in_action_norm = RMSNorm(self.hidden_dim, eps=config.norm_eps)

        dyn_hid_in = self.blocks * (self._deter_per_block + (3 * self.hidden_dim))
        self.dyn_hidden = nn.ModuleList()
        self.dyn_hidden_norm = nn.ModuleList()
        current = dyn_hid_in
        for _ in range(config.dyn_layers):
            self.dyn_hidden.append(BlockLinear(current, self.deter_dim, blocks=self.blocks))
            self.dyn_hidden_norm.append(RMSNorm(self.deter_dim, eps=config.norm_eps))
            current = self.deter_dim
        self.dyn_gru = BlockLinear(self.deter_dim, 3 * self.deter_dim, blocks=self.blocks)

        # Prior network.
        self.prior_layers = nn.ModuleList()
        self.prior_norms = nn.ModuleList()
        prior_in = self.deter_dim
        for _ in range(config.img_layers):
            self.prior_layers.append(nn.Linear(prior_in, self.hidden_dim))
            self.prior_norms.append(RMSNorm(self.hidden_dim, eps=config.norm_eps))
            prior_in = self.hidden_dim
        self.prior_out = nn.Linear(self.hidden_dim, self._flat_stoch_dim)

        # Posterior network.
        post_in_dim = self.embed_dim if config.absolute else (self.deter_dim + self.embed_dim)
        self.post_layers = nn.ModuleList()
        self.post_norms = nn.ModuleList()
        for _ in range(config.obs_layers):
            self.post_layers.append(nn.Linear(post_in_dim, self.hidden_dim))
            self.post_norms.append(RMSNorm(self.hidden_dim, eps=config.norm_eps))
            post_in_dim = self.hidden_dim
        self.post_out = nn.Linear(self.hidden_dim, self._flat_stoch_dim)

        self._init_output_scale(config.outscale)

    def _init_output_scale(self, outscale: float) -> None:
        nn.init.uniform_(self.prior_out.weight, -outscale, outscale)
        nn.init.zeros_(self.prior_out.bias)
        nn.init.uniform_(self.post_out.weight, -outscale, outscale)
        nn.init.zeros_(self.post_out.bias)

    def initial(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> RSSMState:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        reference = next(self.parameters())
        state_device = device if device is not None else reference.device
        state_dtype = dtype if dtype is not None else reference.dtype
        deter = torch.zeros(batch_size, self.deter_dim, device=state_device, dtype=state_dtype)
        stoch = torch.zeros(
            batch_size,
            self.stoch_dim,
            self.classes,
            device=state_device,
            dtype=state_dtype,
        )
        logits = torch.zeros(
            batch_size,
            self.stoch_dim,
            self.classes,
            device=state_device,
            dtype=state_dtype,
        )
        return RSSMState(deter=deter, stoch=stoch, logits=logits)

    @staticmethod
    def detach_state(state: RSSMState) -> RSSMState:
        return RSSMState(
            deter=state.deter.detach(),
            stoch=state.stoch.detach(),
            logits=state.logits.detach(),
        )

    @staticmethod
    def _stack_states(states: list[RSSMState]) -> RSSMState:
        if not states:
            raise ValueError("Cannot stack an empty state list.")
        return RSSMState(
            deter=torch.stack([state.deter for state in states], dim=0),
            stoch=torch.stack([state.stoch for state in states], dim=0),
            logits=torch.stack([state.logits for state in states], dim=0),
        )

    def _flat_stoch(self, stoch: torch.Tensor) -> torch.Tensor:
        if stoch.shape[-2:] != (self.stoch_dim, self.classes):
            raise ValueError(
                "stoch shape mismatch. Expected tail "
                f"({self.stoch_dim}, {self.classes}), got {tuple(stoch.shape[-2:])}."
            )
        return stoch.reshape(*stoch.shape[:-2], self._flat_stoch_dim)

    def get_feat(self, state: RSSMState) -> torch.Tensor:
        return torch.cat([state.deter, self._flat_stoch(state.stoch)], dim=-1)

    def _reshape_flat_to_group(self, value: torch.Tensor) -> torch.Tensor:
        return value.reshape(*value.shape[:-1], self.blocks, value.shape[-1] // self.blocks)

    def _reshape_group_to_flat(self, value: torch.Tensor) -> torch.Tensor:
        return value.reshape(*value.shape[:-2], value.shape[-2] * value.shape[-1])

    def _action_norm(self, action: torch.Tensor) -> torch.Tensor:
        # Matches `action /= max(1, abs(action))` with stop-gradient denominator.
        denom = action.abs().clamp_min(1.0).detach()
        return action / denom

    def _core(self, deter: torch.Tensor, stoch: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        stoch_flat = self._flat_stoch(stoch)
        action = self._action_norm(action)

        x0 = F.silu(self.dyn_in_deter_norm(self.dyn_in_deter(deter)))
        x1 = F.silu(self.dyn_in_stoch_norm(self.dyn_in_stoch(stoch_flat)))
        x2 = F.silu(self.dyn_in_action_norm(self.dyn_in_action(action)))

        x = torch.cat([x0, x1, x2], dim=-1)
        x = x.unsqueeze(-2).expand(*x.shape[:-1], self.blocks, x.shape[-1])
        deter_group = self._reshape_flat_to_group(deter)
        x = torch.cat([deter_group, x], dim=-1)
        x = self._reshape_group_to_flat(x)

        for layer, norm in zip(self.dyn_hidden, self.dyn_hidden_norm, strict=True):
            x = F.silu(norm(layer(x)))

        gates = self.dyn_gru(x)
        gates = self._reshape_flat_to_group(gates)
        reset, cand, update = gates.chunk(3, dim=-1)
        reset = self._reshape_group_to_flat(reset)
        cand = self._reshape_group_to_flat(cand)
        update = self._reshape_group_to_flat(update)

        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1.0)
        return (update * cand) + ((1.0 - update) * deter)

    def _prior(self, deter: torch.Tensor) -> torch.Tensor:
        x = deter
        for layer, norm in zip(self.prior_layers, self.prior_norms, strict=True):
            x = F.silu(norm(layer(x)))
        logits = self.prior_out(x)
        return logits.reshape(*deter.shape[:-1], self.stoch_dim, self.classes)

    def _posterior(self, deter: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        x = embed if self.config.absolute else torch.cat([deter, embed], dim=-1)
        for layer, norm in zip(self.post_layers, self.post_norms, strict=True):
            x = F.silu(norm(layer(x)))
        logits = self.post_out(x)
        return logits.reshape(*deter.shape[:-1], self.stoch_dim, self.classes)

    def _logits_to_state(
        self,
        *,
        deter: torch.Tensor,
        logits: torch.Tensor,
        sample: bool,
    ) -> tuple[RSSMState, OneHotDist]:
        dist = OneHotDist(logits, unimix=self.unimix)
        stoch = dist.sample(straight_through=True) if sample else dist.mode()
        return RSSMState(deter=deter, stoch=stoch, logits=dist.logits), dist

    def _reset_state(self, state: RSSMState, is_first: torch.Tensor) -> RSSMState:
        mask = (1.0 - is_first.to(dtype=state.deter.dtype)).unsqueeze(-1)
        return RSSMState(
            deter=state.deter * mask,
            stoch=state.stoch * mask.unsqueeze(-1),
            logits=state.logits * mask.unsqueeze(-1),
        )

    def _reset_action(self, action: torch.Tensor, is_first: torch.Tensor) -> torch.Tensor:
        mask = (1.0 - is_first.to(dtype=action.dtype)).unsqueeze(-1)
        return action * mask

    def img_step(
        self,
        prev_state: RSSMState,
        prev_action: torch.Tensor,
        *,
        sample: bool = True,
    ) -> tuple[RSSMState, OneHotDist]:
        if prev_action.shape[-1] != self.action_dim:
            raise ValueError(
                f"prev_action last dim must be {self.action_dim}, got {prev_action.shape[-1]}."
            )
        if prev_action.shape[:-1] != prev_state.deter.shape[:-1]:
            raise ValueError(
                "prev_action and prev_state batch shapes must match, got "
                f"{tuple(prev_action.shape[:-1])} and {tuple(prev_state.deter.shape[:-1])}."
            )

        deter = self._core(prev_state.deter, prev_state.stoch, prev_action)
        prior_logits = self._prior(deter)
        return self._logits_to_state(deter=deter, logits=prior_logits, sample=sample)

    def obs_step(  # noqa: PLR0913
        self,
        prev_state: RSSMState,
        prev_action: torch.Tensor,
        embed: torch.Tensor,
        *,
        is_first: torch.Tensor | None = None,
        sample: bool = True,
    ) -> tuple[RSSMState, RSSMState, OneHotDist, OneHotDist]:
        if embed.shape[-1] != self.embed_dim:
            raise ValueError(f"embed last dim must be {self.embed_dim}, got {embed.shape[-1]}.")
        if embed.shape[:-1] != prev_state.deter.shape[:-1]:
            raise ValueError(
                "embed and prev_state batch shapes must match, got "
                f"{tuple(embed.shape[:-1])} and {tuple(prev_state.deter.shape[:-1])}."
            )
        if is_first is not None:
            if is_first.shape != prev_state.deter.shape[:-1]:
                raise ValueError(
                    f"is_first shape must be {tuple(prev_state.deter.shape[:-1])}, "
                    f"got {tuple(is_first.shape)}."
                )
            prev_state = self._reset_state(prev_state, is_first)
            prev_action = self._reset_action(prev_action, is_first)

        prior_state, prior_dist = self.img_step(prev_state, prev_action, sample=sample)
        post_logits = self._posterior(prior_state.deter, embed)
        post_state, post_dist = self._logits_to_state(
            deter=prior_state.deter,
            logits=post_logits,
            sample=sample,
        )
        return post_state, prior_state, post_dist, prior_dist

    def observe(
        self,
        embeds: torch.Tensor,
        prev_actions: torch.Tensor,
        is_first: torch.Tensor,
        *,
        state: RSSMState | None = None,
        sample: bool = True,
    ) -> tuple[RSSMState, RSSMState, RSSMState]:
        if embeds.ndim != 3:
            raise ValueError(f"embeds must be rank 3 [T,B,E], got {embeds.shape}.")
        if prev_actions.ndim != 3:
            raise ValueError(f"prev_actions must be rank 3 [T,B,A], got {prev_actions.shape}.")
        if is_first.ndim != 2:
            raise ValueError(f"is_first must be rank 2 [T,B], got {is_first.shape}.")
        if embeds.shape[:2] != prev_actions.shape[:2] or embeds.shape[:2] != is_first.shape:
            raise ValueError(
                "Sequence dimensions must match across embeds/prev_actions/is_first, got "
                f"{embeds.shape[:2]}, {prev_actions.shape[:2]}, {is_first.shape}."
            )
        if embeds.shape[-1] != self.embed_dim:
            raise ValueError(f"embeds last dim must be {self.embed_dim}, got {embeds.shape[-1]}.")
        if prev_actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"prev_actions last dim must be {self.action_dim}, got {prev_actions.shape[-1]}."
            )

        time_steps, batch_size = embeds.shape[0], embeds.shape[1]
        current = state if state is not None else self.initial(
            batch_size,
            device=embeds.device,
            dtype=embeds.dtype,
        )

        posts: list[RSSMState] = []
        priors: list[RSSMState] = []
        for index in range(time_steps):
            post, prior, _post_dist, _prior_dist = self.obs_step(
                current,
                prev_actions[index],
                embeds[index],
                is_first=is_first[index],
                sample=sample,
            )
            posts.append(post)
            priors.append(prior)
            current = post
        return self._stack_states(posts), self._stack_states(priors), current

    def imagine(
        self,
        actions: torch.Tensor,
        *,
        state: RSSMState,
        sample: bool = True,
    ) -> tuple[RSSMState, RSSMState]:
        if actions.ndim != 3:
            raise ValueError(f"actions must be rank 3 [T,B,A], got {actions.shape}.")
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"actions last dim must be {self.action_dim}, got {actions.shape[-1]}."
            )
        if actions.shape[1] != state.deter.shape[0]:
            raise ValueError(
                "actions batch dimension must match state batch size, got "
                f"{actions.shape[1]} and {state.deter.shape[0]}."
            )

        priors: list[RSSMState] = []
        current = state
        for index in range(actions.shape[0]):
            current, _prior_dist = self.img_step(current, actions[index], sample=sample)
            priors.append(current)
        return self._stack_states(priors), current

    @staticmethod
    def categorical_kl(lhs_logits: torch.Tensor, rhs_logits: torch.Tensor) -> torch.Tensor:
        if lhs_logits.shape != rhs_logits.shape:
            raise ValueError(
                f"lhs_logits and rhs_logits must have same shape, got "
                f"{lhs_logits.shape} and {rhs_logits.shape}."
            )
        if lhs_logits.ndim < 2:
            raise ValueError(
                "categorical_kl expects at least [..., stoch_dim, classes] rank, got "
                f"{lhs_logits.shape}."
            )
        lhs_log_probs = torch.log_softmax(lhs_logits, dim=-1)
        rhs_log_probs = torch.log_softmax(rhs_logits, dim=-1)
        lhs_probs = torch.softmax(lhs_logits, dim=-1)
        return (lhs_probs * (lhs_log_probs - rhs_log_probs)).sum(dim=-1).sum(dim=-1)

    def kl_terms(
        self,
        posterior: RSSMState,
        prior: RSSMState,
        *,
        free_nats: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if free_nats <= 0.0:
            raise ValueError(f"free_nats must be positive, got {free_nats}.")

        dyn = self.categorical_kl(posterior.logits.detach(), prior.logits)
        rep = self.categorical_kl(posterior.logits, prior.logits.detach())
        free = torch.full_like(dyn, free_nats)
        dyn = torch.maximum(dyn, free)
        rep = torch.maximum(rep, free)
        return dyn, rep
