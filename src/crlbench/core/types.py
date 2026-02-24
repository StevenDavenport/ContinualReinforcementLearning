from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

Observation = Mapping[str, Any]
Action = Any
InfoDict = Mapping[str, Any]


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: InfoDict = field(default_factory=dict)


@dataclass(frozen=True)
class Transition:
    observation: Observation
    action: Action
    reward: float
    next_observation: Observation
    terminated: bool
    truncated: bool
    info: InfoDict = field(default_factory=dict)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    env_family: str
    env_option: str
    metadata: InfoDict = field(default_factory=dict)


@dataclass(frozen=True)
class EvalScore:
    task_id: str
    step: int
    mean_return: float
    std_return: float
    episodes: int
    metadata: InfoDict = field(default_factory=dict)
