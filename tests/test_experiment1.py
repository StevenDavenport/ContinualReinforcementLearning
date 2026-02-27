from __future__ import annotations

import pytest

from crlbench.core.plugins import create_environment, create_experiment
from crlbench.experiments import (
    EXPERIMENT_ID,
    build_experiment1_eval_plan,
    build_experiment1_protocol,
    build_experiment1_tasks,
    create_experiment1_dm_control_environment,
    create_experiment1_stub_environment,
    get_experiment1_action_space_n,
    get_experiment1_budget,
    get_experiment1_runtime_bound,
    list_experiment1_roots,
    register_experiment1_plugins,
)


def test_experiment1_task_templates_cover_all_roots() -> None:
    roots = list_experiment1_roots()
    for track, env_family, env_option in roots:
        tasks = build_experiment1_tasks(track=track, env_family=env_family, env_option=env_option)
        assert len(tasks) >= 3
        assert tasks[0].metadata["track"] == track
        assert tasks[-1].env_family == env_family
        assert tasks[-1].env_option == env_option


def test_experiment1_eval_plan_grows_seen_set() -> None:
    tasks = build_experiment1_tasks(
        track="toy",
        env_family="dm_control",
        env_option="vision_sequential_default",
    )
    eval_plan = build_experiment1_eval_plan(tasks)
    assert eval_plan[0].eval_task_ids == ("walker_walk",)
    assert eval_plan[-1].eval_task_ids == (
        "walker_walk",
        "cheetah_run",
        "quadruped_walk",
        "humanoid_walk",
    )


def test_experiment1_quadruped_recovery_sequence() -> None:
    tasks = build_experiment1_tasks(
        track="toy",
        env_family="dm_control",
        env_option="vision_sequential_quadruped_recovery",
    )
    assert tuple(task.task_id for task in tasks) == (
        "quadruped_run",
        "quadruped_fetch",
        "quadruped_escape",
    )
    eval_plan = build_experiment1_eval_plan(tasks)
    assert eval_plan[-1].eval_task_ids == (
        "quadruped_run",
        "quadruped_fetch",
        "quadruped_escape",
    )


def test_experiment1_quadruped_anchor_escape_sequence() -> None:
    tasks = build_experiment1_tasks(
        track="toy",
        env_family="dm_control",
        env_option="vision_sequential_quadruped_anchor_escape",
    )
    assert tuple(task.task_id for task in tasks) == (
        "quadruped_escape",
        "quadruped_run",
        "quadruped_fetch",
    )
    eval_plan = build_experiment1_eval_plan(tasks)
    assert eval_plan[-1].eval_task_ids == (
        "quadruped_escape",
        "quadruped_run",
        "quadruped_fetch",
    )


def test_experiment1_protocol_budget_and_runtime() -> None:
    protocol = build_experiment1_protocol(
        track="robotics",
        env_family="metaworld",
        env_option="manipulation_sequential_default",
        budget_tier="smoke",
    )
    assert protocol.experiment_id == EXPERIMENT_ID
    assert protocol.budget == get_experiment1_budget("smoke")
    assert protocol.runtime_bound == get_experiment1_runtime_bound(
        tier="smoke",
        env_family="metaworld",
        env_option="manipulation_sequential_default",
    )


def test_experiment1_plugin_registration_and_stub_env() -> None:
    register_experiment1_plugins(replace=True)
    protocol = create_experiment(
        EXPERIMENT_ID,
        track="toy",
        env_family="procgen",
        env_option="vision_sequential_alternative",
        budget_tier="dev",
    )
    assert protocol.track == "toy"
    env = create_environment(
        "procgen",
        env_option="vision_sequential_alternative",
        task_id="coinrun",
        observation_mode="image",
    )
    observation = env.reset(seed=7)
    assert "pixels" in observation
    assert isinstance(observation["action_space_n"], int)
    step = env.step(0)
    assert step.reward >= 0.0
    assert env.metadata["adapter"] == "stub"


def test_experiment1_stub_adapters_cover_toy_and_robotics_options() -> None:
    roots = [
        ("dm_control", "vision_sequential_default", "walker_walk"),
        ("dm_control", "vision_sequential_quadruped_recovery", "quadruped_fetch"),
        ("dm_control", "vision_sequential_quadruped_anchor_escape", "quadruped_escape"),
        ("procgen", "vision_sequential_alternative", "coinrun"),
        ("metaworld", "manipulation_sequential_default", "reach"),
        ("maniskill", "manipulation_sequential_alternative", "pick_cube"),
    ]
    for env_family, env_option, task_id in roots:
        env = create_experiment1_stub_environment(
            env_family=env_family,
            env_option=env_option,
            task_id=task_id,
            observation_mode=(
                "image_proprio" if env_family in {"metaworld", "maniskill"} else "image"
            ),
        )
        observation = env.reset(seed=3)
        assert observation["task"] == task_id
        expected_n = get_experiment1_action_space_n(env_family=env_family, env_option=env_option)
        assert observation["action_space_n"] == expected_n
        if env_family in {"metaworld", "maniskill"}:
            assert "proprio" in observation
        step = env.step(0)
        assert step.info["env_family"] == env_family
        assert step.info["env_option"] == env_option


def test_experiment1_dm_control_auto_backend_falls_back_to_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("crlbench.experiments.experiment1.dm_control_available", lambda: False)
    register_experiment1_plugins(replace=True, dm_control_backend="auto")
    env = create_environment(
        "dm_control",
        env_option="vision_sequential_default",
        task_id="walker_walk",
    )
    env.reset(seed=1)
    assert env.metadata["adapter"] == "stub"


def test_experiment1_dm_control_explicit_stub_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("crlbench.experiments.experiment1.dm_control_available", lambda: True)
    env = create_experiment1_dm_control_environment(
        env_option="vision_sequential_default",
        task_id="walker_walk",
        dm_control_backend="stub",
    )
    observation = env.reset(seed=1)
    assert "pixels" in observation
    assert env.metadata["adapter"] == "stub"


def test_experiment1_dm_control_real_backend_requires_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("crlbench.experiments.experiment1.dm_control_available", lambda: False)
    with pytest.raises(ValueError, match="dm_control backend 'real' requested"):
        register_experiment1_plugins(replace=True, dm_control_backend="real")
