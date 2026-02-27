from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from crlbench.agents import discover_agents, instantiate_agent, validate_agent
from crlbench.core.types import Transition


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def test_discover_agents_finds_builtins() -> None:
    discovered = discover_agents(Path("agents"))
    assert "random" in discovered
    assert "ppo_baseline" in discovered
    assert "ppo_continuous_baseline" in discovered


def test_instantiate_agent_random() -> None:
    agent, descriptor = instantiate_agent(
        agent_name="random",
        agents_dir=Path("agents"),
        config={"seed": 7, "action_space_n": 2},
    )
    assert descriptor.name == "random"
    observation = {"obs": [0.0], "step": 0}
    action = agent.act(observation, deterministic=False)
    assert isinstance(action, int)


def test_validate_agent_random_passes() -> None:
    errors = validate_agent(agent_name="random", agents_dir=Path("agents"), config={"seed": 0})
    assert errors == []


def test_validate_agent_ppo_continuous_baseline_passes() -> None:
    if not _torch_available():
        pytest.skip("torch is not installed.")
    errors = validate_agent(
        agent_name="ppo_continuous_baseline",
        agents_dir=Path("agents"),
        config={"seed": 0, "rollout_size": 2, "update_epochs": 1},
    )
    assert errors == []


def test_ppo_continuous_baseline_supports_continuous_and_discrete_observations() -> None:
    if not _torch_available():
        pytest.skip("torch is not installed.")
    agent, descriptor = instantiate_agent(
        agent_name="ppo_continuous_baseline",
        agents_dir=Path("agents"),
        config={"seed": 7, "rollout_size": 2, "update_epochs": 1},
    )
    assert descriptor.name == "ppo_continuous_baseline"

    continuous_observation = {
        "pixels": [[[0, 0, 0]]],
        "task": "walker_walk",
        "domain_name": "walker",
        "step": 0,
        "reward_last": 0.0,
        "continuous_action": True,
        "action_dim": 3,
        "action_low": [-1.0, -1.0, -1.0],
        "action_high": [1.0, 1.0, 1.0],
    }
    continuous_action = agent.act(continuous_observation, deterministic=False)
    assert isinstance(continuous_action, list)
    assert len(continuous_action) == 3

    metrics_one = agent.update(
        [
            Transition(
                observation=continuous_observation,
                action=continuous_action,
                reward=0.5,
                next_observation=continuous_observation,
                terminated=False,
                truncated=False,
                info={},
            )
        ]
    )
    assert float(metrics_one["num_updates"]) >= 0.0

    metrics_two = agent.update(
        [
            Transition(
                observation=continuous_observation,
                action=continuous_action,
                reward=0.25,
                next_observation=continuous_observation,
                terminated=False,
                truncated=True,
                info={},
            )
        ]
    )
    assert float(metrics_two["num_updates"]) > 0.0

    discrete_observation = {"obs": [0.0], "step": 1, "action_space_n": 5}
    discrete_action = agent.act(discrete_observation, deterministic=False)
    assert isinstance(discrete_action, int)
