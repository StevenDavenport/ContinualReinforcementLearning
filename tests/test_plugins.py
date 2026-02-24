from __future__ import annotations

from crlbench.core.plugins import (
    create_environment,
    create_experiment,
    register_env_family,
    register_experiment,
    registered_env_families,
    registered_experiments,
)
from crlbench.core.types import StepResult


class DummyEnv:
    def reset(self, seed: int | None = None) -> dict[str, float]:
        _ = seed
        return {"obs": 0.0}

    def step(self, action: int) -> StepResult:
        _ = action
        return StepResult(observation={"obs": 0.0}, reward=0.0, terminated=False, truncated=False)

    def close(self) -> None:
        return None

    @property
    def metadata(self) -> dict[str, str]:
        return {"name": "dummy"}


def test_plugin_registries_register_and_create() -> None:
    register_env_family("test_dummy_env", lambda: DummyEnv(), replace=True)
    env = create_environment("test_dummy_env")
    assert env.metadata["name"] == "dummy"
    assert "test_dummy_env" in registered_env_families()

    register_experiment("test_exp", lambda: {"ok": True}, replace=True)
    exp = create_experiment("test_exp")
    assert exp["ok"] is True
    assert "test_exp" in registered_experiments()
