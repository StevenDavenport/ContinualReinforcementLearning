from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .contracts import EnvironmentAdapter
from .registry import Registry

EnvironmentFactory = Callable[..., EnvironmentAdapter]
ExperimentFactory = Callable[..., Any]

ENV_FAMILY_REGISTRY: Registry[EnvironmentAdapter] = Registry(name="env_family")
EXPERIMENT_REGISTRY: Registry[Any] = Registry(name="experiment")


def register_env_family(
    name: str,
    factory: EnvironmentFactory,
    *,
    replace: bool = False,
) -> None:
    ENV_FAMILY_REGISTRY.register(name, factory, replace=replace)


def create_environment(name: str, *args: object, **kwargs: object) -> EnvironmentAdapter:
    return ENV_FAMILY_REGISTRY.create(name, *args, **kwargs)


def registered_env_families() -> tuple[str, ...]:
    return ENV_FAMILY_REGISTRY.names()


def register_experiment(
    name: str,
    factory: ExperimentFactory,
    *,
    replace: bool = False,
) -> None:
    EXPERIMENT_REGISTRY.register(name, factory, replace=replace)


def create_experiment(name: str, *args: object, **kwargs: object) -> Any:
    return EXPERIMENT_REGISTRY.create(name, *args, **kwargs)


def registered_experiments() -> tuple[str, ...]:
    return EXPERIMENT_REGISTRY.names()
