from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from crlbench.core.types import TaskSpec


def _require_positive(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _require_int(state: Mapping[str, Any], key: str) -> int:
    value = state.get(key)
    if not isinstance(value, int):
        raise ValueError(f"state.{key} must be int.")
    return value


@dataclass(frozen=True)
class ContextEvent:
    step: int
    switched: bool
    internal_context: str
    public_context: str | None


class HiddenContextController:
    """
    Hidden context scheduler with deterministic dwell-time switching.

    `internal_context` is for benchmark internals and metrics only.
    `public_context` stays `None` to avoid boundary/context leakage.
    """

    def __init__(
        self,
        *,
        contexts: Sequence[str],
        min_dwell_steps: int = 1,
        max_dwell_steps: int = 1,
        seed: int = 0,
    ) -> None:
        if not contexts:
            raise ValueError("contexts must be non-empty.")
        self.contexts = [ctx.strip() for ctx in contexts]
        if any(not ctx for ctx in self.contexts):
            raise ValueError("contexts must contain non-empty identifiers.")
        self.min_dwell_steps = _require_positive("min_dwell_steps", min_dwell_steps)
        self.max_dwell_steps = _require_positive("max_dwell_steps", max_dwell_steps)
        if self.max_dwell_steps < self.min_dwell_steps:
            raise ValueError(
                "max_dwell_steps must be >= min_dwell_steps, "
                f"got {self.max_dwell_steps} < {self.min_dwell_steps}."
            )
        self.seed = seed
        self._rng = random.Random(seed)
        self._step = 0
        self._context_index = 0
        self._dwell_elapsed = 0
        self._target_dwell = self._sample_target_dwell()
        self._switch_count = 0

    def _sample_target_dwell(self) -> int:
        return self._rng.randint(self.min_dwell_steps, self.max_dwell_steps)

    def reset(self) -> None:
        self._rng = random.Random(self.seed)
        self._step = 0
        self._context_index = 0
        self._dwell_elapsed = 0
        self._target_dwell = self._sample_target_dwell()
        self._switch_count = 0

    @property
    def internal_context(self) -> str:
        return self.contexts[self._context_index]

    @property
    def switch_count(self) -> int:
        return self._switch_count

    def _maybe_switch(self) -> bool:
        self._dwell_elapsed += 1
        if self._dwell_elapsed < self._target_dwell:
            return False
        if len(self.contexts) == 1:
            self._dwell_elapsed = 0
            self._target_dwell = self._sample_target_dwell()
            return False
        next_index = self._context_index
        while next_index == self._context_index:
            next_index = self._rng.randrange(0, len(self.contexts))
        self._context_index = next_index
        self._dwell_elapsed = 0
        self._target_dwell = self._sample_target_dwell()
        self._switch_count += 1
        return True

    def step(self) -> ContextEvent:
        self._step += 1
        switched = self._maybe_switch()
        return ContextEvent(
            step=self._step,
            switched=switched,
            internal_context=self.internal_context,
            public_context=None,
        )

    def mask_task(self, task: TaskSpec) -> TaskSpec:
        metadata = dict(task.metadata)
        metadata.pop("context_id", None)
        metadata.pop("switch_boundary", None)
        metadata["context_masked"] = True
        return TaskSpec(
            task_id=task.task_id,
            env_family=task.env_family,
            env_option=task.env_option,
            metadata=metadata,
        )

    def get_state(self) -> dict[str, Any]:
        return {
            "step": self._step,
            "context_index": self._context_index,
            "dwell_elapsed": self._dwell_elapsed,
            "target_dwell": self._target_dwell,
            "switch_count": self._switch_count,
            "rng_state": self._rng.getstate(),
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        step = _require_int(state, "step")
        context_index = _require_int(state, "context_index")
        dwell_elapsed = _require_int(state, "dwell_elapsed")
        target_dwell = _require_int(state, "target_dwell")
        switch_count = _require_int(state, "switch_count")
        if context_index < 0 or context_index >= len(self.contexts):
            raise ValueError(f"context_index out of bounds: {context_index}.")
        if target_dwell <= 0:
            raise ValueError(f"target_dwell must be positive, got {target_dwell}.")
        rng_state_raw = state.get("rng_state")
        if rng_state_raw is None:
            raise ValueError("state requires rng_state.")
        rng_state = _list_to_tuple(rng_state_raw)
        self._rng = random.Random(self.seed)
        self._rng.setstate(rng_state)
        self._step = step
        self._context_index = context_index
        self._dwell_elapsed = dwell_elapsed
        self._target_dwell = target_dwell
        self._switch_count = switch_count


def _list_to_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_list_to_tuple(item) for item in value)
    return value
