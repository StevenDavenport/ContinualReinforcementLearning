from __future__ import annotations

import importlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from crlbench.core.types import Transition


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        out = float(value)
        return out if math.isfinite(out) else default
    return default


def _flatten_numeric(value: object, *, out: list[float], max_items: int) -> None:
    if len(out) >= max_items:
        return
    if isinstance(value, bool):
        return
    if isinstance(value, int | float):
        out.append(float(value))
        return
    if isinstance(value, Mapping):
        for key in sorted(value):
            _flatten_numeric(value[key], out=out, max_items=max_items)
            if len(out) >= max_items:
                return
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for item in value:
            _flatten_numeric(item, out=out, max_items=max_items)
            if len(out) >= max_items:
                return


def _as_float_list(value: object) -> list[float]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_safe_float(item, 0.0) for item in value]
    return []


class PPOContinuousBaselineAgent:
    def __init__(  # noqa: PLR0912,PLR0913,PLR0915
        self,
        *,
        seed: int = 0,
        action_dim: int = 1,
        max_action_dim: int = 32,
        obs_dim: int = 512,
        hidden_size: int = 256,
        actor_learning_rate: float = 3e-4,
        value_learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 1e-3,
        value_coef: float = 0.5,
        rollout_size: int = 2048,
        minibatch_size: int = 256,
        update_epochs: int = 10,
        max_grad_norm: float = 0.5,
        log_std_init: float = -0.5,
        device: str = "cpu",
    ) -> None:
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}.")
        if max_action_dim <= 0:
            raise ValueError(f"max_action_dim must be positive, got {max_action_dim}.")
        if action_dim > max_action_dim:
            raise ValueError(
                f"action_dim ({action_dim}) cannot exceed max_action_dim ({max_action_dim})."
            )
        if obs_dim <= 0:
            raise ValueError(f"obs_dim must be positive, got {obs_dim}.")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}.")
        if actor_learning_rate <= 0.0:
            raise ValueError("actor_learning_rate must be positive.")
        if value_learning_rate <= 0.0:
            raise ValueError("value_learning_rate must be positive.")
        if not (0.0 < gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1].")
        if not (0.0 < gae_lambda <= 1.0):
            raise ValueError("gae_lambda must be in (0, 1].")
        if clip_epsilon <= 0.0:
            raise ValueError("clip_epsilon must be positive.")
        if rollout_size <= 1:
            raise ValueError("rollout_size must be > 1.")
        if minibatch_size <= 1:
            raise ValueError("minibatch_size must be > 1.")
        if update_epochs <= 0:
            raise ValueError("update_epochs must be positive.")
        if max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive.")

        self.seed = seed
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.rollout_size = rollout_size
        self.minibatch_size = minibatch_size
        self.update_epochs = update_epochs
        self.max_grad_norm = max_grad_norm
        self.obs_dim = obs_dim
        self.hidden_size = hidden_size
        self._default_action_dim = action_dim
        self._max_action_dim = max_action_dim

        self._rng = random.Random(seed)

        torch_mod = _import_torch()
        self._torch = torch_mod
        self._nn = importlib.import_module("torch.nn")
        self._F = importlib.import_module("torch.nn.functional")
        self.device = torch_mod.device(device)
        torch_mod.manual_seed(seed)
        if self.device.type == "cuda":
            if not torch_mod.cuda.is_available():
                raise ValueError(
                    "device='cuda' requested but CUDA is unavailable in this environment."
                )
            torch_mod.cuda.manual_seed_all(seed)

        self._actor: Any = self._build_mlp(out_dim=max_action_dim).to(self.device)
        self._critic: Any = self._build_mlp(out_dim=1).to(self.device)
        self._log_std = self._nn.Parameter(
            torch_mod.full(
                (max_action_dim,),
                float(log_std_init),
                dtype=torch_mod.float32,
                device=self.device,
            )
        )

        params = list(self._actor.parameters()) + list(self._critic.parameters()) + [self._log_std]
        self._optimizer = torch_mod.optim.Adam(
            params,
            lr=min(actor_learning_rate, value_learning_rate),
        )

        self._buffer_obs: list[Any] = []
        self._buffer_actions: list[Any] = []
        self._buffer_rewards: list[float] = []
        self._buffer_dones: list[float] = []
        self._buffer_log_probs: list[float] = []
        self._buffer_values: list[float] = []
        self._buffer_action_dim: int = action_dim

        self._last_action_vector: list[float] | None = None
        self._last_log_prob: float | None = None
        self._last_value: float | None = None
        self._last_action_dim: int | None = None

    def _build_mlp(self, *, out_dim: int) -> Any:
        return self._nn.Sequential(
            self._nn.Linear(self.obs_dim, self.hidden_size),
            self._nn.Tanh(),
            self._nn.Linear(self.hidden_size, self.hidden_size),
            self._nn.Tanh(),
            self._nn.Linear(self.hidden_size, out_dim),
        )

    def _action_dim(self, observation: Mapping[str, Any]) -> int:
        raw = observation.get("action_dim", self._default_action_dim)
        if isinstance(raw, int) and raw > 0:
            if raw > self._max_action_dim:
                raise ValueError(
                    f"Observed action_dim={raw} exceeds max_action_dim={self._max_action_dim}. "
                    "Increase --set max_action_dim=<n>."
                )
            return raw
        return self._default_action_dim

    def _is_continuous_observation(self, observation: Mapping[str, Any]) -> bool:
        return (
            observation.get("continuous_action") is True
            or "action_dim" in observation
            or "action_low" in observation
            or "action_high" in observation
        )

    def _action_space_n(self, observation: Mapping[str, Any]) -> int:
        raw = observation.get("action_space_n", 2)
        if isinstance(raw, int) and raw > 1:
            return raw
        return 2

    def _action_bounds(
        self, observation: Mapping[str, Any], action_dim: int
    ) -> tuple[list[float], list[float]]:
        low = _as_float_list(observation.get("action_low"))
        high = _as_float_list(observation.get("action_high"))
        if len(low) != action_dim:
            low = [-1.0 for _ in range(action_dim)]
        if len(high) != action_dim:
            high = [1.0 for _ in range(action_dim)]
        for index in range(action_dim):
            if low[index] >= high[index]:
                low[index], high[index] = -1.0, 1.0
        return low, high

    def _feature_vector(self, observation: Mapping[str, Any]) -> list[float]:
        values: list[float] = []
        _flatten_numeric(observation.get("pixels", []), out=values, max_items=self.obs_dim * 2)
        _flatten_numeric(observation.get("proprio", []), out=values, max_items=self.obs_dim * 2)
        if len(values) < self.obs_dim * 2:
            _flatten_numeric(observation, out=values, max_items=self.obs_dim * 2)

        vector = values[: self.obs_dim]
        if len(vector) < self.obs_dim:
            vector.extend([0.0] * (self.obs_dim - len(vector)))

        normalized: list[float] = []
        for value in vector:
            scaled = value / 255.0 if abs(value) > 5.0 else value
            normalized.append(math.tanh(scaled))
        return normalized

    def _tensor_obs(self, observation: Mapping[str, Any]) -> Any:
        vec = self._feature_vector(observation)
        return self._torch.tensor(vec, dtype=self._torch.float32, device=self.device)

    def _dist(self, obs_tensor: Any, action_dim: int) -> Any:
        mean_full = self._actor(obs_tensor)
        mean = mean_full[..., :action_dim]
        log_std = self._log_std[:action_dim].clamp(-4.0, 2.0)
        std = self._torch.exp(log_std)
        return self._torch.distributions.Normal(mean, std)

    def _value(self, obs_tensor: Any) -> Any:
        return self._critic(obs_tensor).squeeze(-1)

    def reset(self) -> None:
        self._buffer_obs.clear()
        self._buffer_actions.clear()
        self._buffer_rewards.clear()
        self._buffer_dones.clear()
        self._buffer_log_probs.clear()
        self._buffer_values.clear()

        self._last_action_vector = None
        self._last_log_prob = None
        self._last_value = None
        self._last_action_dim = None

    def act(self, observation: Mapping[str, Any], *, deterministic: bool = False) -> Any:
        obs_tensor = self._tensor_obs(observation).unsqueeze(0)
        continuous = self._is_continuous_observation(observation)
        action_dim = self._action_dim(observation) if continuous else 1
        low, high = self._action_bounds(observation, action_dim)

        with self._torch.no_grad():
            dist = self._dist(obs_tensor, action_dim)
            action_tensor = dist.mean if deterministic else dist.sample()
            log_prob = dist.log_prob(action_tensor).sum(dim=-1)
            value = self._value(obs_tensor)

        action_values = action_tensor.squeeze(0).tolist()
        bounded: list[float] = []
        for idx in range(action_dim):
            bounded.append(_clamp(float(action_values[idx]), low[idx], high[idx]))

        self._last_action_vector = bounded
        self._last_log_prob = float(log_prob.item())
        self._last_value = float(value.item())
        self._last_action_dim = action_dim

        if continuous:
            return bounded if action_dim > 1 else bounded[0]

        action_space_n = self._action_space_n(observation)
        scalar = _clamp(bounded[0], -1.0, 1.0)
        index = int(round(((scalar + 1.0) * 0.5) * (action_space_n - 1)))
        return max(0, min(action_space_n - 1, index))

    def _coerce_action_vector(
        self,
        *,
        action: Any,
        observation: Mapping[str, Any],
        action_dim: int,
        continuous: bool,
    ) -> list[float]:
        low, high = self._action_bounds(observation, action_dim)
        if continuous:
            if isinstance(action, Sequence) and not isinstance(action, str | bytes | bytearray):
                values = [_safe_float(item, 0.0) for item in action]
            elif isinstance(action, int | float):
                values = [float(action)]
            else:
                values = [0.0]
            if len(values) == 1:
                values = [values[0] for _ in range(action_dim)]
            values = (values + [0.0] * action_dim)[:action_dim]
            return [_clamp(values[i], low[i], high[i]) for i in range(action_dim)]

        action_space_n = self._action_space_n(observation)
        index = int(action) if isinstance(action, int | bool) else 0
        index = max(0, min(action_space_n - 1, index))
        span = max(1, action_space_n - 1)
        value = -1.0 + (2.0 * index / span)
        return [_clamp(value, low[0], high[0])]

    def _optimize(self) -> Mapping[str, float]:  # noqa: PLR0915
        length = len(self._buffer_rewards)
        if length < 2:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "num_updates": 0.0}

        obs = self._torch.stack(self._buffer_obs)
        actions = self._torch.stack(self._buffer_actions)
        rewards = self._torch.tensor(
            self._buffer_rewards, dtype=self._torch.float32, device=self.device
        )
        dones = self._torch.tensor(
            self._buffer_dones, dtype=self._torch.float32, device=self.device
        )
        old_log_probs = self._torch.tensor(
            self._buffer_log_probs,
            dtype=self._torch.float32,
            device=self.device,
        )
        values = self._torch.tensor(
            self._buffer_values, dtype=self._torch.float32, device=self.device
        )

        advantages = self._torch.zeros_like(rewards)
        returns = self._torch.zeros_like(rewards)
        gae = 0.0
        next_value = 0.0
        for idx in reversed(range(length)):
            mask = 1.0 - float(dones[idx].item())
            delta = (
                float(rewards[idx].item())
                + (self.gamma * next_value * mask)
                - float(values[idx].item())
            )
            gae = delta + (self.gamma * self.gae_lambda * mask * gae)
            advantages[idx] = gae
            returns[idx] = gae + values[idx]
            next_value = float(values[idx].item())

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = length
        mb_size = max(2, min(self.minibatch_size, batch_size))
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        updates = 0

        for _ in range(self.update_epochs):
            indices = list(range(batch_size))
            self._rng.shuffle(indices)
            for start in range(0, batch_size, mb_size):
                mb = indices[start : start + mb_size]
                if len(mb) < 2:
                    continue
                mb_index = self._torch.tensor(mb, dtype=self._torch.long, device=self.device)

                mb_obs = obs.index_select(0, mb_index)
                mb_actions = actions.index_select(0, mb_index)
                mb_old_log_probs = old_log_probs.index_select(0, mb_index)
                mb_advantages = advantages.index_select(0, mb_index)
                mb_returns = returns.index_select(0, mb_index)

                dist = self._dist(mb_obs, self._buffer_action_dim)
                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                values_pred = self._value(mb_obs)

                ratio = self._torch.exp((new_log_probs - mb_old_log_probs).clamp(-20.0, 20.0))
                unclipped = ratio * mb_advantages
                clipped = (
                    ratio.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                )
                policy_loss = -self._torch.min(unclipped, clipped).mean()
                value_loss = self._F.mse_loss(values_pred, mb_returns)
                loss = policy_loss + (self.value_coef * value_loss) - (self.entropy_coef * entropy)

                self._optimizer.zero_grad()
                loss.backward()
                self._torch.nn.utils.clip_grad_norm_(
                    list(self._actor.parameters())
                    + list(self._critic.parameters())
                    + [self._log_std],
                    max_norm=self.max_grad_norm,
                )
                self._optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))
                updates += 1

        if updates <= 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "num_updates": 0.0}

        return {
            "policy_loss": sum(policy_losses) / float(len(policy_losses)),
            "value_loss": sum(value_losses) / float(len(value_losses)),
            "entropy": sum(entropies) / float(len(entropies)),
            "num_updates": float(updates),
        }

    def update(self, batch: Sequence[Transition]) -> Mapping[str, float]:
        if not batch:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "num_updates": 0.0}

        outputs: list[Mapping[str, float]] = []
        for transition in batch:
            observation = transition.observation
            continuous = self._is_continuous_observation(observation)
            action_dim = self._action_dim(observation) if continuous else 1
            if self._last_action_dim is not None and self._last_action_dim != action_dim:
                action_dim = self._last_action_dim
            self._buffer_action_dim = action_dim

            obs_tensor = self._tensor_obs(observation)
            action_vector = self._coerce_action_vector(
                action=transition.action,
                observation=observation,
                action_dim=action_dim,
                continuous=continuous,
            )
            action_tensor = self._torch.tensor(
                action_vector,
                dtype=self._torch.float32,
                device=self.device,
            )

            if (
                self._last_action_vector is not None
                and self._last_log_prob is not None
                and self._last_value is not None
                and self._last_action_dim == action_dim
            ):
                old_log_prob = self._last_log_prob
                old_value = self._last_value
            else:
                with self._torch.no_grad():
                    dist = self._dist(obs_tensor.unsqueeze(0), action_dim)
                    old_log_prob = float(
                        dist.log_prob(action_tensor.unsqueeze(0)).sum(dim=-1).item()
                    )
                    old_value = float(self._value(obs_tensor.unsqueeze(0)).item())

            self._buffer_obs.append(obs_tensor)
            self._buffer_actions.append(action_tensor)
            self._buffer_rewards.append(float(transition.reward))
            self._buffer_dones.append(
                1.0 if (transition.terminated or transition.truncated) else 0.0
            )
            self._buffer_log_probs.append(old_log_prob)
            self._buffer_values.append(old_value)

            self._last_action_vector = None
            self._last_log_prob = None
            self._last_value = None
            self._last_action_dim = None

            if (
                len(self._buffer_rewards) >= self.rollout_size
                or transition.terminated
                or transition.truncated
            ):
                outputs.append(self._optimize())
                self._buffer_obs.clear()
                self._buffer_actions.clear()
                self._buffer_rewards.clear()
                self._buffer_dones.clear()
                self._buffer_log_probs.clear()
                self._buffer_values.clear()

        if not outputs:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "num_updates": 0.0,
                "buffered_steps": float(len(self._buffer_rewards)),
            }

        count = float(len(outputs))
        return {
            "policy_loss": sum(out["policy_loss"] for out in outputs) / count,
            "value_loss": sum(out["value_loss"] for out in outputs) / count,
            "entropy": sum(out["entropy"] for out in outputs) / count,
            "num_updates": sum(out["num_updates"] for out in outputs),
            "buffered_steps": float(len(self._buffer_rewards)),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        actor_state = {k: v.detach().cpu().tolist() for k, v in self._actor.state_dict().items()}
        critic_state = {k: v.detach().cpu().tolist() for k, v in self._critic.state_dict().items()}
        payload = {
            "seed": self.seed,
            "obs_dim": self.obs_dim,
            "hidden_size": self.hidden_size,
            "action_dim": self._buffer_action_dim,
            "max_action_dim": self._max_action_dim,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_epsilon": self.clip_epsilon,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
            "rollout_size": self.rollout_size,
            "minibatch_size": self.minibatch_size,
            "update_epochs": self.update_epochs,
            "max_grad_norm": self.max_grad_norm,
            "actor_state_dict": actor_state,
            "critic_state_dict": critic_state,
            "log_std": self._log_std.detach().cpu().tolist(),
            "buffered_steps": len(self._buffer_rewards),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _import_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise ValueError(
            "ppo_continuous_baseline now requires PyTorch. Install with: "
            "python -m pip install -e '.[torch]'"
        ) from exc


def create_agent(config: Mapping[str, Any]) -> PPOContinuousBaselineAgent:  # noqa: PLR0912,PLR0915
    seed_raw = config.get("seed", 0)
    action_dim_raw = config.get("action_dim", 1)
    max_action_dim_raw = config.get("max_action_dim", 32)
    obs_dim_raw = config.get("obs_dim", 512)
    hidden_size_raw = config.get("hidden_size", 256)
    actor_lr_raw = config.get("actor_learning_rate", 3e-4)
    value_lr_raw = config.get("value_learning_rate", 3e-4)
    gamma_raw = config.get("gamma", 0.99)
    gae_lambda_raw = config.get("gae_lambda", 0.95)
    clip_epsilon_raw = config.get("clip_epsilon", 0.2)
    entropy_coef_raw = config.get("entropy_coef", 1e-3)
    value_coef_raw = config.get("value_coef", 0.5)
    rollout_size_raw = config.get("rollout_size", 2048)
    minibatch_size_raw = config.get("minibatch_size", 256)
    update_epochs_raw = config.get("update_epochs", 10)
    max_grad_norm_raw = config.get("max_grad_norm", 0.5)
    log_std_init_raw = config.get("log_std_init", -0.5)
    device_raw = config.get("device", "cpu")

    if not isinstance(seed_raw, int):
        raise ValueError(f"seed must be int, got {seed_raw!r}.")
    if not isinstance(action_dim_raw, int):
        raise ValueError(f"action_dim must be int, got {action_dim_raw!r}.")
    if not isinstance(max_action_dim_raw, int):
        raise ValueError(f"max_action_dim must be int, got {max_action_dim_raw!r}.")
    if not isinstance(obs_dim_raw, int):
        raise ValueError(f"obs_dim must be int, got {obs_dim_raw!r}.")
    if not isinstance(hidden_size_raw, int):
        raise ValueError(f"hidden_size must be int, got {hidden_size_raw!r}.")
    if not isinstance(actor_lr_raw, int | float):
        raise ValueError(f"actor_learning_rate must be numeric, got {actor_lr_raw!r}.")
    if not isinstance(value_lr_raw, int | float):
        raise ValueError(f"value_learning_rate must be numeric, got {value_lr_raw!r}.")
    if not isinstance(gamma_raw, int | float):
        raise ValueError(f"gamma must be numeric, got {gamma_raw!r}.")
    if not isinstance(gae_lambda_raw, int | float):
        raise ValueError(f"gae_lambda must be numeric, got {gae_lambda_raw!r}.")
    if not isinstance(clip_epsilon_raw, int | float):
        raise ValueError(f"clip_epsilon must be numeric, got {clip_epsilon_raw!r}.")
    if not isinstance(entropy_coef_raw, int | float):
        raise ValueError(f"entropy_coef must be numeric, got {entropy_coef_raw!r}.")
    if not isinstance(value_coef_raw, int | float):
        raise ValueError(f"value_coef must be numeric, got {value_coef_raw!r}.")
    if not isinstance(rollout_size_raw, int):
        raise ValueError(f"rollout_size must be int, got {rollout_size_raw!r}.")
    if not isinstance(minibatch_size_raw, int):
        raise ValueError(f"minibatch_size must be int, got {minibatch_size_raw!r}.")
    if not isinstance(update_epochs_raw, int):
        raise ValueError(f"update_epochs must be int, got {update_epochs_raw!r}.")
    if not isinstance(max_grad_norm_raw, int | float):
        raise ValueError(f"max_grad_norm must be numeric, got {max_grad_norm_raw!r}.")
    if not isinstance(log_std_init_raw, int | float):
        raise ValueError(f"log_std_init must be numeric, got {log_std_init_raw!r}.")
    if not isinstance(device_raw, str):
        raise ValueError(f"device must be str, got {device_raw!r}.")

    return PPOContinuousBaselineAgent(
        seed=seed_raw,
        action_dim=action_dim_raw,
        max_action_dim=max_action_dim_raw,
        obs_dim=obs_dim_raw,
        hidden_size=hidden_size_raw,
        actor_learning_rate=float(actor_lr_raw),
        value_learning_rate=float(value_lr_raw),
        gamma=float(gamma_raw),
        gae_lambda=float(gae_lambda_raw),
        clip_epsilon=float(clip_epsilon_raw),
        entropy_coef=float(entropy_coef_raw),
        value_coef=float(value_coef_raw),
        rollout_size=rollout_size_raw,
        minibatch_size=minibatch_size_raw,
        update_epochs=update_epochs_raw,
        max_grad_norm=float(max_grad_norm_raw),
        log_std_init=float(log_std_init_raw),
        device=device_raw.strip(),
    )
