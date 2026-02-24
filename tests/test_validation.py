from __future__ import annotations

import pytest

from crlbench.core.types import StepResult, Transition
from crlbench.core.validation import (
    SchemaValidationError,
    validate_step_result,
    validate_transition,
)


def test_validate_step_result_success() -> None:
    step = StepResult(
        observation={"pixels": [[0, 0], [1, 1]]},
        reward=1.0,
        terminated=False,
        truncated=False,
        info={"task": "A"},
    )
    validate_step_result(step)


def test_validate_step_result_rejects_non_finite_reward() -> None:
    step = StepResult(
        observation={"pixels": [[0, 0], [1, 1]]},
        reward=float("nan"),
        terminated=False,
        truncated=False,
        info={},
    )
    with pytest.raises(SchemaValidationError):
        validate_step_result(step)


def test_validate_transition_success() -> None:
    transition = Transition(
        observation={"obs": 0.0},
        action=0,
        reward=0.5,
        next_observation={"obs": 1.0},
        terminated=False,
        truncated=False,
        info={},
    )
    validate_transition(transition)
