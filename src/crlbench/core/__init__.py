"""Core interfaces and shared primitives."""

from .contracts import AgentAdapter, EnvironmentAdapter, Evaluator, TaskStream
from .plugins import (
    create_environment,
    create_experiment,
    register_env_family,
    register_experiment,
    registered_env_families,
    registered_experiments,
)
from .registry import Registry, RegistryError
from .types import EvalScore, StepResult, TaskSpec, Transition
from .validation import (
    SchemaValidationError,
    validate_observation,
    validate_step_result,
    validate_transition,
)

__all__ = [
    "AgentAdapter",
    "EnvironmentAdapter",
    "EvalScore",
    "Evaluator",
    "Registry",
    "RegistryError",
    "SchemaValidationError",
    "StepResult",
    "TaskSpec",
    "TaskStream",
    "Transition",
    "create_environment",
    "create_experiment",
    "register_env_family",
    "register_experiment",
    "registered_env_families",
    "registered_experiments",
    "validate_observation",
    "validate_step_result",
    "validate_transition",
]
