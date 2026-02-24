from __future__ import annotations

import copy
import random
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeGuard

from crlbench.core.contracts import EnvironmentAdapter
from crlbench.core.types import Action, Observation, StepResult

ObservationTransform = Callable[[dict[str, Any]], dict[str, Any]]
ActionTransform = Callable[[Action], Action]
DynamicsSampler = Callable[[random.Random], Mapping[str, float]]


def _is_numeric(value: object) -> TypeGuard[int | float]:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _coerce_image_rows(value: object, *, key: str) -> list[list[Any]]:
    if not isinstance(value, list):
        raise ValueError(f"Observation key '{key}' must be a list image tensor.")
    rows: list[list[Any]] = []
    for row in value:
        if not isinstance(row, list):
            raise ValueError(f"Observation key '{key}' rows must be list values.")
        rows.append(row)
    if not rows or not rows[0]:
        raise ValueError(f"Observation key '{key}' image tensor cannot be empty.")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"Observation key '{key}' image tensor has ragged rows.")
    return rows


def _resize_nearest(rows: list[list[Any]], *, target_h: int, target_w: int) -> list[list[Any]]:
    source_h = len(rows)
    source_w = len(rows[0])
    resized: list[list[Any]] = []
    for i in range(target_h):
        source_i = min(source_h - 1, int(i * source_h / target_h))
        out_row: list[Any] = []
        for j in range(target_w):
            source_j = min(source_w - 1, int(j * source_w / target_w))
            out_row.append(copy.deepcopy(rows[source_i][source_j]))
        resized.append(out_row)
    return resized


def _normalize_tree(value: object, *, divisor: float) -> object:
    if _is_numeric(value):
        return float(value) / divisor
    if isinstance(value, list):
        return [_normalize_tree(item, divisor=divisor) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_tree(item, divisor=divisor) for item in value)
    return value


def _transform_numeric_action(action: Action, fn: Callable[[float], float]) -> Action:
    if _is_numeric(action):
        return fn(float(action))
    if isinstance(action, list):
        return [_transform_numeric_action(item, fn) for item in action]
    if isinstance(action, tuple):
        return tuple(_transform_numeric_action(item, fn) for item in action)
    raise ValueError(f"Unsupported action type for numeric transform: {type(action).__name__}.")


class EnvironmentWrapper(EnvironmentAdapter):
    def __init__(self, env: EnvironmentAdapter) -> None:
        self.env = env

    def reset(self, seed: int | None = None) -> Observation:
        return self.env.reset(seed=seed)

    def step(self, action: Action) -> StepResult:
        return self.env.step(action)

    def close(self) -> None:
        self.env.close()

    @property
    def metadata(self) -> Mapping[str, str]:
        return self.env.metadata


class ObservationTransformWrapper(EnvironmentWrapper):
    def __init__(self, env: EnvironmentAdapter, transforms: Sequence[ObservationTransform]) -> None:
        super().__init__(env)
        self._transforms = list(transforms)

    def _apply(self, observation: Observation) -> Observation:
        payload: dict[str, Any] = dict(observation)
        for transform in self._transforms:
            payload = transform(payload)
        return payload

    def reset(self, seed: int | None = None) -> Observation:
        return self._apply(super().reset(seed=seed))

    def step(self, action: Action) -> StepResult:
        result = super().step(action)
        return StepResult(
            observation=self._apply(result.observation),
            reward=result.reward,
            terminated=result.terminated,
            truncated=result.truncated,
            info=result.info,
        )


def select_observation_key_transform(*, key: str) -> ObservationTransform:
    def transform(observation: dict[str, Any]) -> dict[str, Any]:
        if key not in observation:
            raise KeyError(f"Observation key missing: {key}")
        return {key: observation[key]}

    return transform


def resize_pixels_transform(*, key: str, height: int, width: int) -> ObservationTransform:
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive; got ({height}, {width}).")

    def transform(observation: dict[str, Any]) -> dict[str, Any]:
        if key not in observation:
            raise KeyError(f"Observation key missing: {key}")
        rows = _coerce_image_rows(observation[key], key=key)
        updated = dict(observation)
        updated[key] = _resize_nearest(rows, target_h=height, target_w=width)
        return updated

    return transform


def normalize_pixels_transform(*, key: str, divisor: float = 255.0) -> ObservationTransform:
    if divisor <= 0:
        raise ValueError(f"divisor must be positive, got {divisor}.")

    def transform(observation: dict[str, Any]) -> dict[str, Any]:
        if key not in observation:
            raise KeyError(f"Observation key missing: {key}")
        updated = dict(observation)
        updated[key] = _normalize_tree(observation[key], divisor=divisor)
        return updated

    return transform


class FrameStackObservationWrapper(EnvironmentWrapper):
    def __init__(
        self,
        env: EnvironmentAdapter,
        *,
        key: str,
        num_frames: int,
        output_key: str | None = None,
    ) -> None:
        super().__init__(env)
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}.")
        self.key = key
        self.output_key = output_key or key
        self._frames: deque[Any] = deque(maxlen=num_frames)
        self._num_frames = num_frames

    def _stack(self, observation: Observation, *, initialize: bool) -> Observation:
        if self.key not in observation:
            raise KeyError(f"Observation key missing: {self.key}")
        frame = copy.deepcopy(observation[self.key])
        if initialize:
            self._frames.clear()
            for _ in range(self._num_frames):
                self._frames.append(copy.deepcopy(frame))
        else:
            self._frames.append(frame)
        updated = dict(observation)
        updated[self.output_key] = list(self._frames)
        return updated

    def reset(self, seed: int | None = None) -> Observation:
        observation = super().reset(seed=seed)
        return self._stack(observation, initialize=True)

    def step(self, action: Action) -> StepResult:
        result = super().step(action)
        return StepResult(
            observation=self._stack(result.observation, initialize=False),
            reward=result.reward,
            terminated=result.terminated,
            truncated=result.truncated,
            info=result.info,
        )


class ActionTransformWrapper(EnvironmentWrapper):
    def __init__(self, env: EnvironmentAdapter, transform: ActionTransform) -> None:
        super().__init__(env)
        self._transform = transform

    def step(self, action: Action) -> StepResult:
        return super().step(self._transform(action))


class ActionRepeatWrapper(EnvironmentWrapper):
    def __init__(self, env: EnvironmentAdapter, *, repeat: int) -> None:
        super().__init__(env)
        if repeat <= 0:
            raise ValueError(f"repeat must be positive, got {repeat}.")
        self.repeat = repeat

    def step(self, action: Action) -> StepResult:
        reward_total = 0.0
        executed = 0
        last: StepResult | None = None
        for _ in range(self.repeat):
            executed += 1
            last = super().step(action)
            reward_total += float(last.reward)
            if last.terminated or last.truncated:
                break
        assert last is not None
        info = dict(last.info)
        info["action_repeat"] = executed
        return StepResult(
            observation=last.observation,
            reward=reward_total,
            terminated=last.terminated,
            truncated=last.truncated,
            info=info,
        )


def clip_action_transform(*, low: float, high: float) -> ActionTransform:
    if high < low:
        raise ValueError(f"Expected high >= low, got low={low}, high={high}.")

    def transform(action: Action) -> Action:
        return _transform_numeric_action(action, lambda value: min(high, max(low, value)))

    return transform


def scale_action_transform(*, factor: float, bias: float = 0.0) -> ActionTransform:
    def transform(action: Action) -> Action:
        return _transform_numeric_action(action, lambda value: (value * factor) + bias)

    return transform


class ActionDelayWrapper(EnvironmentWrapper):
    def __init__(
        self, env: EnvironmentAdapter, *, delay_steps: int, default_action: Action
    ) -> None:
        super().__init__(env)
        if delay_steps < 0:
            raise ValueError(f"delay_steps must be non-negative, got {delay_steps}.")
        self.delay_steps = delay_steps
        self.default_action = default_action
        self._pending: deque[Action] = deque()

    def reset(self, seed: int | None = None) -> Observation:
        self._pending.clear()
        return super().reset(seed=seed)

    def step(self, action: Action) -> StepResult:
        if self.delay_steps == 0:
            return super().step(action)
        self._pending.append(copy.deepcopy(action))
        if len(self._pending) <= self.delay_steps:
            return super().step(copy.deepcopy(self.default_action))
        delayed_action = self._pending.popleft()
        return super().step(delayed_action)


def ranges_sampler(parameter_ranges: Mapping[str, tuple[float, float]]) -> DynamicsSampler:
    ranges = dict(parameter_ranges)
    for name, bounds in ranges.items():
        low, high = bounds
        if high < low:
            raise ValueError(f"Invalid dynamics range for '{name}': ({low}, {high}).")

    def sample(rng: random.Random) -> Mapping[str, float]:
        return {name: rng.uniform(low, high) for name, (low, high) in ranges.items()}

    return sample


class DynamicsRandomizationWrapper(EnvironmentWrapper):
    def __init__(
        self,
        env: EnvironmentAdapter,
        *,
        sampler: DynamicsSampler,
        seed: int = 0,
        strict: bool = False,
    ) -> None:
        super().__init__(env)
        self._sampler = sampler
        self._rng = random.Random(seed)
        self._strict = strict
        self._last_parameters: dict[str, float] = {}

    def _apply_dynamics(self) -> None:
        params = dict(self._sampler(self._rng))
        setter = getattr(self.env, "set_dynamics", None)
        if callable(setter):
            setter(params)
            self._last_parameters = params
            return
        if self._strict:
            raise RuntimeError("Environment does not expose callable set_dynamics(parameters).")

    @property
    def current_parameters(self) -> Mapping[str, float]:
        return dict(self._last_parameters)

    def reset(self, seed: int | None = None) -> Observation:
        self._apply_dynamics()
        return super().reset(seed=seed)


def compose_wrappers(
    env: EnvironmentAdapter,
    wrappers: Sequence[Callable[[EnvironmentAdapter], EnvironmentAdapter]],
) -> EnvironmentAdapter:
    wrapped = env
    for wrapper in wrappers:
        wrapped = wrapper(wrapped)
    return wrapped


@dataclass(frozen=True)
class WrapperBundle:
    observation: EnvironmentAdapter
    action: EnvironmentAdapter
    randomized: EnvironmentAdapter
