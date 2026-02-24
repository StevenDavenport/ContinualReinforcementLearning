from __future__ import annotations

from collections.abc import Mapping

import pytest

from crlbench.core.types import Action, Observation, StepResult
from crlbench.runtime.wrappers import (
    ActionDelayWrapper,
    ActionRepeatWrapper,
    ActionTransformWrapper,
    DynamicsRandomizationWrapper,
    FrameStackObservationWrapper,
    ObservationTransformWrapper,
    clip_action_transform,
    normalize_pixels_transform,
    ranges_sampler,
    resize_pixels_transform,
    scale_action_transform,
    select_observation_key_transform,
)


class DummyEnv:
    def __init__(self) -> None:
        self.step_count = 0
        self.last_action: Action | None = None
        self.last_dynamics: dict[str, float] = {}

    def reset(self, seed: int | None = None) -> Observation:
        _ = seed
        self.step_count = 0
        return {"pixels": [[0, 255], [64, 128]], "state": [0.0, 1.0]}

    def step(self, action: Action) -> StepResult:
        self.step_count += 1
        self.last_action = action
        terminated = self.step_count >= 2
        return StepResult(
            observation={"pixels": [[self.step_count, 10], [20, 30]], "state": [1.0, 2.0]},
            reward=1.0,
            terminated=terminated,
            truncated=False,
            info={"count": self.step_count},
        )

    def close(self) -> None:
        return None

    @property
    def metadata(self) -> Mapping[str, str]:
        return {"family": "dummy"}

    def set_dynamics(self, params: dict[str, float]) -> None:
        self.last_dynamics = dict(params)


def test_observation_transform_wrapper_resize_normalize_select() -> None:
    env = DummyEnv()
    wrapped = ObservationTransformWrapper(
        env,
        transforms=[
            resize_pixels_transform(key="pixels", height=1, width=1),
            normalize_pixels_transform(key="pixels"),
            select_observation_key_transform(key="pixels"),
        ],
    )

    observation = wrapped.reset(seed=0)
    assert list(observation) == ["pixels"]
    pixels = observation["pixels"]
    assert isinstance(pixels, list)
    assert pixels == [[0.0]]


def test_frame_stack_wrapper_stacks_on_reset_and_step() -> None:
    env = DummyEnv()
    wrapped = FrameStackObservationWrapper(env, key="pixels", num_frames=3)
    first = wrapped.reset(seed=0)
    frames = first["pixels"]
    assert isinstance(frames, list)
    assert len(frames) == 3

    second = wrapped.step(0).observation["pixels"]
    assert isinstance(second, list)
    assert len(second) == 3


def test_action_transform_repeat_clip_scale_delay() -> None:
    env = DummyEnv()
    transformed = ActionTransformWrapper(
        env,
        transform=clip_action_transform(low=-1.0, high=1.0),
    )
    transformed = ActionTransformWrapper(
        transformed,
        transform=scale_action_transform(factor=2.0),
    )
    delayed = ActionDelayWrapper(transformed, delay_steps=1, default_action=0.0)
    repeated = ActionRepeatWrapper(delayed, repeat=2)
    delayed.reset(seed=0)

    first = repeated.step(2.0)
    assert first.reward == 2.0
    assert first.info["action_repeat"] == 2
    assert env.last_action == 1.0


def test_dynamics_randomization_wrapper_samples_and_applies() -> None:
    env = DummyEnv()
    wrapped = DynamicsRandomizationWrapper(
        env,
        sampler=ranges_sampler({"mass": (0.9, 1.1), "friction": (0.0, 0.1)}),
        seed=7,
        strict=True,
    )
    wrapped.reset(seed=0)
    params = wrapped.current_parameters
    assert set(params) == {"mass", "friction"}
    assert env.last_dynamics == params


def test_action_transform_rejects_unsupported_action() -> None:
    env = DummyEnv()
    wrapped = ActionTransformWrapper(env, transform=clip_action_transform(low=-1.0, high=1.0))
    wrapped.reset(seed=0)
    with pytest.raises(ValueError, match="Unsupported action type"):
        wrapped.step({"bad": "shape"})


def test_dynamics_randomization_wrapper_strict_without_setter() -> None:
    class NoHookEnv:
        def reset(self, seed: int | None = None) -> Observation:
            _ = seed
            return {"pixels": [[0]], "state": [0.0]}

        def step(self, action: Action) -> StepResult:
            _ = action
            return StepResult(
                observation={"pixels": [[0]], "state": [0.0]},
                reward=0.0,
                terminated=False,
                truncated=False,
                info={},
            )

        def close(self) -> None:
            return None

        @property
        def metadata(self) -> Mapping[str, str]:
            return {"family": "dummy"}

    env = NoHookEnv()
    wrapped = DynamicsRandomizationWrapper(
        env,
        sampler=ranges_sampler({"mass": (1.0, 1.0)}),
        strict=False,
    )
    wrapped.reset(seed=0)
