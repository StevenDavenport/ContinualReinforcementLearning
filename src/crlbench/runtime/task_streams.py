from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from crlbench.core.types import TaskSpec


def _coerce_int(state: Mapping[str, Any], key: str) -> int:
    value = state.get(key)
    if not isinstance(value, int):
        raise ValueError(f"stream state requires integer '{key}'.")
    return value


def _list_to_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_list_to_tuple(item) for item in value)
    return value


@dataclass
class SequentialTaskStream:
    """Simple deterministic task stream used by smoke tests and early scaffolding."""

    tasks: Sequence[TaskSpec]
    _index: int = 0

    def reset(self) -> None:
        self._index = 0

    def next_task(self) -> TaskSpec | None:
        if self._index >= len(self.tasks):
            return None
        task = self.tasks[self._index]
        self._index += 1
        return task

    def remaining(self) -> int:
        return max(0, len(self.tasks) - self._index)

    def get_state(self) -> dict[str, Any]:
        return {"index": self._index, "size": len(self.tasks)}

    def set_state(self, state: Mapping[str, Any]) -> None:
        index = _coerce_int(state, "index")
        if index < 0 or index > len(self.tasks):
            raise ValueError(
                f"stream state index out of bounds: {index} not in [0, {len(self.tasks)}]."
            )
        self._index = index


@dataclass
class CyclicTaskStream:
    """Cycles through tasks for a finite number of cycles or indefinitely."""

    tasks: Sequence[TaskSpec]
    cycles: int | None = None
    _index: int = 0
    _emitted: int = 0

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ValueError("CyclicTaskStream requires at least one task.")
        if self.cycles is not None and self.cycles <= 0:
            raise ValueError(f"cycles must be positive when set, got {self.cycles}.")

    def _max_items(self) -> int | None:
        if self.cycles is None:
            return None
        return len(self.tasks) * self.cycles

    def reset(self) -> None:
        self._index = 0
        self._emitted = 0

    def next_task(self) -> TaskSpec | None:
        max_items = self._max_items()
        if max_items is not None and self._emitted >= max_items:
            return None
        task = self.tasks[self._index]
        self._index = (self._index + 1) % len(self.tasks)
        self._emitted += 1
        return task

    def remaining(self) -> int:
        max_items = self._max_items()
        if max_items is None:
            return len(self.tasks)
        return max(0, max_items - self._emitted)

    def get_state(self) -> dict[str, Any]:
        return {"index": self._index, "emitted": self._emitted}

    def set_state(self, state: Mapping[str, Any]) -> None:
        index = _coerce_int(state, "index")
        emitted = _coerce_int(state, "emitted")
        if index < 0 or index >= len(self.tasks):
            raise ValueError(
                f"stream state index out of bounds: {index} not in [0, {len(self.tasks) - 1}]."
            )
        if emitted < 0:
            raise ValueError(f"stream state emitted must be non-negative, got {emitted}.")
        self._index = index
        self._emitted = emitted


@dataclass
class StochasticTaskStream:
    """Samples tasks with or without replacement under deterministic seed control."""

    tasks: Sequence[TaskSpec]
    seed: int = 0
    replacement: bool = True
    total_tasks: int | None = None
    _emitted: int = 0
    _cursor: int = 0
    _order: list[int] | None = None
    _rng: random.Random | None = None

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ValueError("StochasticTaskStream requires at least one task.")
        if self.total_tasks is not None and self.total_tasks <= 0:
            raise ValueError(f"total_tasks must be positive when set, got {self.total_tasks}.")
        self._rng = random.Random(self.seed)
        if not self.replacement:
            self._reshuffle()

    def _reshuffle(self) -> None:
        assert self._rng is not None
        self._order = list(range(len(self.tasks)))
        self._rng.shuffle(self._order)
        self._cursor = 0

    def reset(self) -> None:
        self._rng = random.Random(self.seed)
        self._emitted = 0
        self._cursor = 0
        self._order = None
        if not self.replacement:
            self._reshuffle()

    def next_task(self) -> TaskSpec | None:
        if self.total_tasks is not None and self._emitted >= self.total_tasks:
            return None
        assert self._rng is not None
        if self.replacement:
            index = self._rng.randrange(0, len(self.tasks))
        else:
            assert self._order is not None
            if self._cursor >= len(self._order):
                self._reshuffle()
            assert self._order is not None
            index = self._order[self._cursor]
            self._cursor += 1
        self._emitted += 1
        return self.tasks[index]

    def remaining(self) -> int:
        if self.total_tasks is None:
            return len(self.tasks)
        return max(0, self.total_tasks - self._emitted)

    def get_state(self) -> dict[str, Any]:
        assert self._rng is not None
        return {
            "emitted": self._emitted,
            "cursor": self._cursor,
            "order": self._order,
            "rng_state": self._rng.getstate(),
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        emitted = _coerce_int(state, "emitted")
        cursor = _coerce_int(state, "cursor")
        order = state.get("order")
        if order is not None and not isinstance(order, list):
            raise ValueError("stream state 'order' must be list or null.")
        order_list = [int(item) for item in order] if isinstance(order, list) else None
        rng_state_value = state.get("rng_state")
        if rng_state_value is None:
            raise ValueError("stream state requires 'rng_state'.")
        rng_state = _list_to_tuple(rng_state_value)
        self._rng = random.Random(self.seed)
        self._rng.setstate(rng_state)
        self._emitted = emitted
        self._cursor = cursor
        self._order = order_list


@dataclass(frozen=True)
class CurriculumStage:
    stage_id: str
    tasks: Sequence[TaskSpec]
    repeats: int = 1

    def __post_init__(self) -> None:
        if not self.stage_id.strip():
            raise ValueError("stage_id must be non-empty.")
        if not self.tasks:
            raise ValueError(f"Curriculum stage '{self.stage_id}' has no tasks.")
        if self.repeats <= 0:
            raise ValueError(
                f"Curriculum stage '{self.stage_id}' repeats must be positive, got {self.repeats}."
            )


@dataclass
class CurriculumTaskStream:
    """Deterministic stage-based task scheduler with explicit curriculum boundaries."""

    stages: Sequence[CurriculumStage]
    _stage_index: int = 0
    _task_index: int = 0
    _repeat_index: int = 0

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("CurriculumTaskStream requires at least one stage.")

    def reset(self) -> None:
        self._stage_index = 0
        self._task_index = 0
        self._repeat_index = 0

    def _advance(self) -> None:
        stage = self.stages[self._stage_index]
        self._task_index += 1
        if self._task_index < len(stage.tasks):
            return
        self._task_index = 0
        self._repeat_index += 1
        if self._repeat_index < stage.repeats:
            return
        self._repeat_index = 0
        self._stage_index += 1

    def next_task(self) -> TaskSpec | None:
        if self._stage_index >= len(self.stages):
            return None
        stage = self.stages[self._stage_index]
        task = stage.tasks[self._task_index]
        self._advance()
        return task

    def remaining(self) -> int:
        if self._stage_index >= len(self.stages):
            return 0
        remaining = 0
        for idx in range(self._stage_index, len(self.stages)):
            stage = self.stages[idx]
            if idx == self._stage_index:
                completed_in_stage = (self._repeat_index * len(stage.tasks)) + self._task_index
                remaining += (stage.repeats * len(stage.tasks)) - completed_in_stage
            else:
                remaining += stage.repeats * len(stage.tasks)
        return remaining

    def current_stage_id(self) -> str | None:
        if self._stage_index >= len(self.stages):
            return None
        return self.stages[self._stage_index].stage_id

    def get_state(self) -> dict[str, Any]:
        return {
            "stage_index": self._stage_index,
            "task_index": self._task_index,
            "repeat_index": self._repeat_index,
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        stage_index = _coerce_int(state, "stage_index")
        task_index = _coerce_int(state, "task_index")
        repeat_index = _coerce_int(state, "repeat_index")
        if stage_index < 0 or stage_index > len(self.stages):
            raise ValueError(
                "stream state stage_index out of bounds: "
                f"{stage_index} not in [0, {len(self.stages)}]."
            )
        if stage_index == len(self.stages):
            if task_index != 0 or repeat_index != 0:
                raise ValueError(
                    "terminal curriculum state requires task_index=0 and repeat_index=0."
                )
            self._stage_index = stage_index
            self._task_index = task_index
            self._repeat_index = repeat_index
            return
        stage = self.stages[stage_index]
        if task_index < 0 or task_index >= len(stage.tasks):
            raise ValueError(
                f"stream state task_index out of bounds for stage '{stage.stage_id}': {task_index}."
            )
        if repeat_index < 0 or repeat_index >= stage.repeats:
            raise ValueError(
                "stream state repeat_index out of bounds for stage "
                f"'{stage.stage_id}': {repeat_index}."
            )
        self._stage_index = stage_index
        self._task_index = task_index
        self._repeat_index = repeat_index
