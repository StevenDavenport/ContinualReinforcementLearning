from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from crlbench.errors import ContractError

from .types import Observation, StepResult, Transition


class SchemaValidationError(ContractError):
    """Raised when adapter output violates benchmark schema contracts."""


def validate_observation(observation: Observation) -> None:
    if not isinstance(observation, Mapping):
        raise SchemaValidationError("Observation must be a mapping.")
    if not observation:
        raise SchemaValidationError("Observation mapping cannot be empty.")
    for key in observation:
        if not isinstance(key, str) or not key.strip():
            raise SchemaValidationError(f"Observation keys must be non-empty strings; got {key!r}.")


def _validate_scalar_reward(reward: Any) -> None:
    if not isinstance(reward, int | float) or isinstance(reward, bool):
        raise SchemaValidationError(f"Reward must be numeric; got {type(reward).__name__}.")
    reward_value = float(reward)
    if not math.isfinite(reward_value):
        raise SchemaValidationError(f"Reward must be finite; got {reward!r}.")


def validate_step_result(step: StepResult) -> None:
    validate_observation(step.observation)
    _validate_scalar_reward(step.reward)
    if not isinstance(step.terminated, bool):
        raise SchemaValidationError("'terminated' must be bool.")
    if not isinstance(step.truncated, bool):
        raise SchemaValidationError("'truncated' must be bool.")
    if not isinstance(step.info, Mapping):
        raise SchemaValidationError("'info' must be a mapping.")


def validate_transition(transition: Transition) -> None:
    validate_observation(transition.observation)
    validate_observation(transition.next_observation)
    _validate_scalar_reward(transition.reward)
    if not isinstance(transition.terminated, bool):
        raise SchemaValidationError("'terminated' must be bool.")
    if not isinstance(transition.truncated, bool):
        raise SchemaValidationError("'truncated' must be bool.")
    if not isinstance(transition.info, Mapping):
        raise SchemaValidationError("'info' must be a mapping.")
