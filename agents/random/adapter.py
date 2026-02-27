from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from crlbench.core.types import Transition


class RandomAgent:
    def __init__(self, *, seed: int = 0, action_space_n: int = 2) -> None:
        if action_space_n <= 0:
            raise ValueError(f"action_space_n must be positive, got {action_space_n}.")
        self.seed = seed
        self.action_space_n = action_space_n
        self._rng = random.Random(seed)

    def reset(self) -> None:
        return None

    def act(self, observation: Mapping[str, Any], *, deterministic: bool = False) -> int:
        _ = observation
        _ = deterministic
        return self._rng.randrange(0, self.action_space_n)

    def update(self, batch: Sequence[Transition]) -> Mapping[str, float]:
        _ = batch
        return {"policy_loss": 0.0}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"seed": self.seed, "action_space_n": self.action_space_n}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def create_agent(config: Mapping[str, Any]) -> RandomAgent:
    seed_raw = config.get("seed", 0)
    action_space_n_raw = config.get("action_space_n", 2)
    if not isinstance(seed_raw, int):
        raise ValueError(f"seed must be int, got {seed_raw!r}.")
    if not isinstance(action_space_n_raw, int):
        raise ValueError(f"action_space_n must be int, got {action_space_n_raw!r}.")
    return RandomAgent(seed=seed_raw, action_space_n=action_space_n_raw)
