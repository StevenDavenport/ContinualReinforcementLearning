from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from .types import Action, EvalScore, Observation, StepResult, TaskSpec, Transition


@runtime_checkable
class AgentAdapter(Protocol):
    """Contract for benchmark-compatible agent integrations."""

    def reset(self) -> None: ...

    def act(self, observation: Observation, *, deterministic: bool = False) -> Action: ...

    def update(self, batch: Sequence[Transition]) -> Mapping[str, float]: ...

    def save(self, path: Path) -> None: ...


@runtime_checkable
class EnvironmentAdapter(Protocol):
    """Normalized environment interface across toy and robotics backends."""

    def reset(self, seed: int | None = None) -> Observation: ...

    def step(self, action: Action) -> StepResult: ...

    def close(self) -> None: ...

    @property
    def metadata(self) -> Mapping[str, str]: ...


@runtime_checkable
class TaskStream(Protocol):
    """Supplies tasks in benchmark-defined order."""

    def reset(self) -> None: ...

    def next_task(self) -> TaskSpec | None: ...

    def remaining(self) -> int: ...


@runtime_checkable
class Evaluator(Protocol):
    """Computes comparable metrics for one or more tasks."""

    def evaluate(
        self,
        *,
        agent: AgentAdapter,
        env_factory: Callable[[TaskSpec], EnvironmentAdapter],
        tasks: Sequence[TaskSpec],
        step: int,
        seed: int,
    ) -> Sequence[EvalScore]: ...
