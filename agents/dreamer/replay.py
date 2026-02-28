from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class ReplayStep:
    observation: dict[str, Any]
    action: list[float]
    prev_action: list[float]
    reward: float
    cont: float
    is_first: float
    is_last: float
    is_terminal: float
    ctx_deter: Any | None
    ctx_stoch: Any | None
    ctx_logits: Any | None


@dataclass(frozen=True)
class ReplayBatch:
    observations: dict[str, torch.Tensor]
    actions: torch.Tensor
    prev_actions: torch.Tensor
    rewards: torch.Tensor
    continues: torch.Tensor
    is_first: torch.Tensor
    is_last: torch.Tensor
    is_terminal: torch.Tensor
    init_deter: torch.Tensor
    init_stoch: torch.Tensor
    init_logits: torch.Tensor
    init_mask: torch.Tensor


class SequenceReplayBuffer:
    """
    Episode replay with Dreamer-like online/uniform/recency sampling controls.
    Includes optional latent context carry for sampled sequence starts.
    """

    def __init__(
        self,
        *,
        capacity: int,
        seed: int = 0,
        online: bool = True,
        uniform_frac: float = 1.0,
        recency_frac: float = 0.0,
        recency_exp: float = 1.0,
        chunksize: int = 1024,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}.")
        if seed < 0:
            raise ValueError(f"seed must be >= 0, got {seed}.")
        if uniform_frac < 0.0 or recency_frac < 0.0:
            raise ValueError(
                f"uniform_frac and recency_frac must be >= 0, got {uniform_frac}, {recency_frac}."
            )
        if uniform_frac + recency_frac <= 0.0:
            raise ValueError("At least one of uniform_frac/recency_frac must be > 0.")
        if recency_exp <= 0.0:
            raise ValueError(f"recency_exp must be positive, got {recency_exp}.")
        if chunksize <= 0:
            raise ValueError(f"chunksize must be positive, got {chunksize}.")

        self.capacity = capacity
        self.online = bool(online)
        self.uniform_frac = float(uniform_frac)
        self.recency_frac = float(recency_frac)
        self.recency_exp = float(recency_exp)
        self.chunksize = int(chunksize)

        self._rng = random.Random(seed)
        self._episodes: deque[list[ReplayStep]] = deque()
        self._current: list[ReplayStep] = []
        self._total_steps = 0
        self._next_is_first = True
        self._prev_action: list[float] | None = None
        self._ctx_deter_shape: tuple[int, ...] | None = None
        self._ctx_stoch_shape: tuple[int, ...] | None = None
        self._ctx_logits_shape: tuple[int, ...] | None = None

    @property
    def size(self) -> int:
        return self._total_steps

    @property
    def num_episodes(self) -> int:
        return len(self._episodes) + (1 if self._current else 0)

    def reset(self) -> None:
        self._episodes.clear()
        self._current.clear()
        self._total_steps = 0
        self._next_is_first = True
        self._prev_action = None
        self._ctx_deter_shape = None
        self._ctx_stoch_shape = None
        self._ctx_logits_shape = None

    def add(
        self,
        *,
        observation: dict[str, Any],
        action: list[float],
        reward: float,
        cont: float,
        done: bool,
        is_terminal: bool,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx_deter: Any | None = None
        ctx_stoch: Any | None = None
        ctx_logits: Any | None = None
        if context is not None:
            deter = context.get("deter")
            stoch = context.get("stoch")
            logits = context.get("logits")
            if deter is not None and stoch is not None and logits is not None:
                deter_t = torch.as_tensor(deter, dtype=torch.float32)
                stoch_t = torch.as_tensor(stoch, dtype=torch.float32)
                logits_t = torch.as_tensor(logits, dtype=torch.float32)
                ctx_deter = deter_t.tolist()
                ctx_stoch = stoch_t.tolist()
                ctx_logits = logits_t.tolist()
                self._ctx_deter_shape = tuple(int(x) for x in deter_t.shape)
                self._ctx_stoch_shape = tuple(int(x) for x in stoch_t.shape)
                self._ctx_logits_shape = tuple(int(x) for x in logits_t.shape)

        prev_action = (
            list(self._prev_action)
            if self._prev_action is not None and len(self._prev_action) == len(action)
            else [0.0 for _ in range(len(action))]
        )
        step = ReplayStep(
            observation=dict(observation),
            action=list(action),
            prev_action=prev_action,
            reward=float(reward),
            cont=float(cont),
            is_first=1.0 if self._next_is_first else 0.0,
            is_last=1.0 if done else 0.0,
            is_terminal=1.0 if is_terminal else 0.0,
            ctx_deter=ctx_deter,
            ctx_stoch=ctx_stoch,
            ctx_logits=ctx_logits,
        )
        self._current.append(step)
        self._total_steps += 1
        self._next_is_first = False
        self._prev_action = list(action)
        if done:
            self._append_episode(self._current)
            self._current = []
            self._next_is_first = True
            self._prev_action = None
        self._evict_to_capacity()

    def _append_episode(self, episode: list[ReplayStep]) -> None:
        if len(episode) <= self.chunksize:
            self._episodes.append(episode)
            return
        for index in range(0, len(episode), self.chunksize):
            self._episodes.append(episode[index : index + self.chunksize])

    def _evict_to_capacity(self) -> None:
        while self._total_steps > self.capacity:
            if self._episodes:
                removed = self._episodes.popleft()
                self._total_steps -= len(removed)
                continue
            if self._current:
                self._current.pop(0)
                self._total_steps -= 1
                continue
            break

    def _candidate_episodes(self, sequence_length: int) -> list[list[ReplayStep]]:
        candidates = [episode for episode in self._episodes if len(episode) >= sequence_length]
        if self.online and len(self._current) >= sequence_length:
            candidates.append(self._current)
        return candidates

    def can_sample(self, *, batch_size: int, sequence_length: int) -> bool:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}.")
        return len(self._candidate_episodes(sequence_length)) > 0

    def _sample_episode_uniform(self, candidates: list[list[ReplayStep]]) -> list[ReplayStep]:
        return self._rng.choice(candidates)

    def _sample_episode_recency(self, candidates: list[list[ReplayStep]]) -> list[ReplayStep]:
        count = len(candidates)
        ranks = [float(index + 1) for index in range(count)]
        weights = [math.pow(rank, self.recency_exp) for rank in ranks]
        total = sum(weights)
        threshold = self._rng.random() * total
        accum = 0.0
        for episode, weight in zip(candidates, weights, strict=True):
            accum += weight
            if accum >= threshold:
                return episode
        return candidates[-1]

    def sample(self, *, batch_size: int, sequence_length: int) -> ReplayBatch | None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}.")

        candidates = self._candidate_episodes(sequence_length)
        if not candidates:
            return None

        total_frac = self.uniform_frac + self.recency_frac
        uniform_prob = self.uniform_frac / total_frac

        windows: list[tuple[list[ReplayStep], int]] = []
        for _ in range(batch_size):
            if self.recency_frac > 0.0 and self._rng.random() > uniform_prob:
                episode = self._sample_episode_recency(candidates)
            else:
                episode = self._sample_episode_uniform(candidates)
            max_start = len(episode) - sequence_length
            start = 0 if max_start <= 0 else self._rng.randint(0, max_start)
            windows.append((episode, start))

        return self._collate(windows, sequence_length=sequence_length)

    def _collate(
        self,
        windows: list[tuple[list[ReplayStep], int]],
        *,
        sequence_length: int,
    ) -> ReplayBatch:
        time_dim = sequence_length
        batch_dim = len(windows)

        sequences = [episode[start : start + sequence_length] for episode, start in windows]

        obs_keys: set[str] = set()
        for sequence in sequences:
            for step in sequence:
                obs_keys.update(step.observation.keys())

        observations: dict[str, torch.Tensor] = {}
        for key in sorted(obs_keys):
            value_grid: list[list[Any]] = []
            valid = True
            for t in range(time_dim):
                row: list[Any] = []
                for b in range(batch_dim):
                    obs = sequences[b][t].observation
                    if key not in obs:
                        valid = False
                        break
                    row.append(obs[key])
                if not valid:
                    break
                value_grid.append(row)
            if not valid:
                continue
            try:
                observations[key] = torch.as_tensor(np.asarray(value_grid))
            except Exception:  # noqa: BLE001
                continue

        actions = torch.tensor(
            [[sequences[b][t].action for b in range(batch_dim)] for t in range(time_dim)],
            dtype=torch.float32,
        )

        action_dim = int(actions.shape[-1])
        prev_actions_grid: list[list[list[float]]] = [
            [sequences[b][t].prev_action for b in range(batch_dim)] for t in range(time_dim)
        ]
        prev_actions = torch.tensor(prev_actions_grid, dtype=torch.float32)

        rewards = torch.tensor(
            [[sequences[b][t].reward for b in range(batch_dim)] for t in range(time_dim)],
            dtype=torch.float32,
        )
        continues = torch.tensor(
            [[sequences[b][t].cont for b in range(batch_dim)] for t in range(time_dim)],
            dtype=torch.float32,
        )
        is_first = torch.tensor(
            [[sequences[b][t].is_first for b in range(batch_dim)] for t in range(time_dim)],
            dtype=torch.float32,
        )
        is_last = torch.tensor(
            [[sequences[b][t].is_last for b in range(batch_dim)] for t in range(time_dim)],
            dtype=torch.float32,
        )
        is_terminal = torch.tensor(
            [[sequences[b][t].is_terminal for b in range(batch_dim)] for t in range(time_dim)],
            dtype=torch.float32,
        )

        init_deter, init_stoch, init_logits, init_mask = self._initial_context(windows)

        return ReplayBatch(
            observations=observations,
            actions=actions,
            prev_actions=prev_actions,
            rewards=rewards,
            continues=continues,
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
            init_deter=init_deter,
            init_stoch=init_stoch,
            init_logits=init_logits,
            init_mask=init_mask,
        )

    def _initial_context(
        self,
        windows: list[tuple[list[ReplayStep], int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_dim = len(windows)

        if self._ctx_deter_shape is None or self._ctx_stoch_shape is None or self._ctx_logits_shape is None:
            return (
                torch.zeros(batch_dim, 0, dtype=torch.float32),
                torch.zeros(batch_dim, 0, 0, dtype=torch.float32),
                torch.zeros(batch_dim, 0, 0, dtype=torch.float32),
                torch.zeros(batch_dim, dtype=torch.float32),
            )

        deter = torch.zeros((batch_dim, *self._ctx_deter_shape), dtype=torch.float32)
        stoch = torch.zeros((batch_dim, *self._ctx_stoch_shape), dtype=torch.float32)
        logits = torch.zeros((batch_dim, *self._ctx_logits_shape), dtype=torch.float32)
        mask = torch.zeros(batch_dim, dtype=torch.float32)

        for b, (episode, start) in enumerate(windows):
            step = episode[start]
            if step.ctx_deter is None or step.ctx_stoch is None or step.ctx_logits is None:
                continue
            if step.is_first >= 0.5:
                continue
            deter[b] = torch.as_tensor(step.ctx_deter, dtype=torch.float32)
            stoch[b] = torch.as_tensor(step.ctx_stoch, dtype=torch.float32)
            logits[b] = torch.as_tensor(step.ctx_logits, dtype=torch.float32)
            mask[b] = 1.0

        return deter, stoch, logits, mask
