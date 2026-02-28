from __future__ import annotations

import importlib.util
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from crlbench.core.contracts import EnvironmentAdapter
from crlbench.core.types import Action, Observation, StepResult


def dm_control_available() -> bool:
    return importlib.util.find_spec("dm_control") is not None


@dataclass(frozen=True)
class DmControlTask:
    domain_name: str
    task_name: str


EXPERIMENT1_DM_CONTROL_TASKS: dict[str, DmControlTask] = {
    "walker_walk": DmControlTask(domain_name="walker", task_name="walk"),
    "cheetah_run": DmControlTask(domain_name="cheetah", task_name="run"),
    "quadruped_walk": DmControlTask(domain_name="quadruped", task_name="walk"),
    "quadruped_run": DmControlTask(domain_name="quadruped", task_name="run"),
    "quadruped_fetch": DmControlTask(domain_name="quadruped", task_name="fetch"),
    "quadruped_escape": DmControlTask(domain_name="quadruped", task_name="escape"),
    "humanoid_walk": DmControlTask(domain_name="humanoid", task_name="walk"),
}


def _flatten_observation_value(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, int | float):
        return [float(value)]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _flatten_observation_value(tolist())
    if isinstance(value, Mapping):
        mapping_values: list[float] = []
        for key in sorted(value):
            mapping_values.extend(_flatten_observation_value(value[key]))
        return mapping_values
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        sequence_values: list[float] = []
        for item in value:
            sequence_values.extend(_flatten_observation_value(item))
        return sequence_values
    return []


class DmControlExperiment1Environment(EnvironmentAdapter):
    def __init__(  # noqa: PLR0913
        self,
        *,
        env_option: str,
        task_id: str,
        observation_mode: str = "image",
        max_steps: int | None = None,
        frame_size: int = 64,
        action_bins: int = 5,
    ) -> None:
        supported_options = {
            "vision_sequential_default",
            "vision_sequential_quadruped_recovery",
            "vision_sequential_quadruped_anchor_escape",
        }
        if env_option not in supported_options:
            raise ValueError(
                "Real dm_control backend only supports env_option in "
                f"{sorted(supported_options)}. Received: {env_option!r}."
            )
        task = EXPERIMENT1_DM_CONTROL_TASKS.get(task_id)
        if task is None:
            known = ", ".join(sorted(EXPERIMENT1_DM_CONTROL_TASKS))
            raise ValueError(f"Unknown dm_control task_id {task_id!r}. Known tasks: {known}.")
        if observation_mode not in {"image", "image_proprio"}:
            raise ValueError(
                "observation_mode must be one of {'image', 'image_proprio'}, "
                f"got {observation_mode!r}."
            )
        if action_bins <= 1:
            raise ValueError(f"action_bins must be > 1, got {action_bins}.")
        if frame_size <= 0:
            raise ValueError(f"frame_size must be positive, got {frame_size}.")
        if max_steps is not None and max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}.")

        self._task = task
        self._task_id = task_id
        self._env_option = env_option
        self._observation_mode = observation_mode
        self._max_steps = max_steps
        self._frame_size = frame_size
        self._action_bins = action_bins

        self._seed = 0
        self._step_count = 0
        self._last_reward = 0.0

        self._suite = self._import_suite()
        self._env: Any = self._create_env(seed=0)

    def _action_spec_info(self) -> tuple[int, list[float], list[float]]:
        action_spec = self._env.action_spec()
        tolist = getattr(action_spec.minimum, "tolist", None)
        minimum = tolist() if callable(tolist) else action_spec.minimum
        tolist = getattr(action_spec.maximum, "tolist", None)
        maximum = tolist() if callable(tolist) else action_spec.maximum
        min_values = _flatten_observation_value(minimum)
        max_values = _flatten_observation_value(maximum)
        action_dim = len(min_values) if min_values else 1
        if not min_values:
            min_values = [-1.0 for _ in range(action_dim)]
        if len(max_values) != action_dim:
            max_values = [1.0 for _ in range(action_dim)]
        return action_dim, min_values, max_values

    def _import_suite(self) -> Any:
        try:
            from dm_control import suite  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ValueError(
                "dm_control backend is unavailable. Install optional dependency "
                "with: python -m pip install -e '.[dm_control]'."
            ) from exc
        return suite

    def _create_env(self, *, seed: int) -> Any:
        return self._suite.load(
            domain_name=self._task.domain_name,
            task_name=self._task.task_name,
            task_kwargs={"random": int(seed)},
        )

    def _to_pixel_grid(self, *, frame_size: int | None = None) -> list[list[list[int]]]:
        size = self._frame_size if frame_size is None else int(frame_size)
        if size <= 0:
            size = self._frame_size
        frame = self._env.physics.render(
            height=size,
            width=size,
            camera_id=0,
        )
        tolist = getattr(frame, "tolist", None)
        if callable(tolist):
            rendered = tolist()
            if isinstance(rendered, list):
                return cast(list[list[list[int]]], rendered)
        return [[[0, 0, 0]]]

    def render_pixels(self, *, frame_size: int | None = None) -> list[list[list[int]]]:
        """Render RGB pixels for qualitative playback at optional custom resolution."""
        return self._to_pixel_grid(frame_size=frame_size)

    def _proprio(self, observation: Mapping[str, Any]) -> list[float]:
        out: list[float] = []
        for value in observation.values():
            out.extend(_flatten_observation_value(value))
        return out

    def _action_vector(self, action: Action) -> Any:
        import numpy as np

        action_spec = self._env.action_spec()
        minimum = np.asarray(action_spec.minimum, dtype=float)
        maximum = np.asarray(action_spec.maximum, dtype=float)
        shape = tuple(int(dim) for dim in minimum.shape)
        size = int(np.prod(shape))

        normalized: Any
        if isinstance(action, bool):
            discrete_action = int(action)
            value = -1.0 + (2.0 * discrete_action / max(1, self._action_bins - 1))
            normalized = np.full(shape, value, dtype=float)
        elif isinstance(action, int):
            discrete_action = max(0, min(self._action_bins - 1, action))
            value = -1.0 + (2.0 * discrete_action / max(1, self._action_bins - 1))
            normalized = np.full(shape, value, dtype=float)
        elif isinstance(action, float):
            normalized = np.full(shape, float(action), dtype=float)
        elif isinstance(action, Sequence) and not isinstance(action, str | bytes | bytearray):
            raw = np.asarray(list(action), dtype=float).reshape(-1)
            if raw.size not in {1, size}:
                raise ValueError(
                    f"Action vector length mismatch: expected 1 or {size}, got {raw.size}."
                )
            normalized = (
                np.full(shape, float(raw[0]), dtype=float)
                if raw.size == 1
                else raw.reshape(shape).astype(float)
            )
        else:
            raise ValueError(f"Unsupported action type {type(action)!r} for dm_control backend.")

        clipped = np.clip(normalized, -1.0, 1.0)
        scaled = minimum + ((clipped + 1.0) * 0.5 * (maximum - minimum))
        return scaled.astype(action_spec.dtype)

    def _observation(self, raw_observation: Mapping[str, Any]) -> Observation:
        action_dim, action_min, action_max = self._action_spec_info()
        payload: dict[str, Any] = {
            "pixels": self._to_pixel_grid(),
            "task": self._task_id,
            "step": self._step_count,
            "action_space_n": self._action_bins,
            "action_dim": action_dim,
            "action_low": [-1.0 for _ in range(action_dim)],
            "action_high": [1.0 for _ in range(action_dim)],
            "action_spec_min": action_min,
            "action_spec_max": action_max,
            "continuous_action": True,
            "domain_name": self._task.domain_name,
            "task_name": self._task.task_name,
            "reward_last": self._last_reward,
        }
        if self._observation_mode == "image_proprio":
            payload["proprio"] = self._proprio(raw_observation)
        return payload

    def reset(self, seed: int | None = None) -> Observation:
        self._seed = 0 if seed is None else int(seed)
        self._step_count = 0
        self._last_reward = 0.0
        self._env = self._create_env(seed=self._seed)
        time_step = self._env.reset()
        return self._observation(time_step.observation)

    def step(self, action: Action) -> StepResult:
        action_vec = self._action_vector(action)
        time_step = self._env.step(action_vec)
        self._step_count += 1
        reward = float(time_step.reward or 0.0)
        self._last_reward = reward
        dm_last = bool(time_step.last())
        max_step_truncation = self._max_steps is not None and self._step_count >= self._max_steps
        truncated = dm_last or max_step_truncation
        terminated = False
        observation = self._observation(time_step.observation)
        return StepResult(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={
                "env_family": "dm_control",
                "env_option": self._env_option,
                "task_id": self._task_id,
                "domain_name": self._task.domain_name,
                "task_name": self._task.task_name,
                "seed": self._seed,
                "adapter": "dm_control_real",
            },
        )

    def close(self) -> None:
        return None

    @property
    def metadata(self) -> Mapping[str, str]:
        action_dim, _, _ = self._action_spec_info()
        return {
            "env_family": "dm_control",
            "env_option": self._env_option,
            "task_id": self._task_id,
            "observation_mode": self._observation_mode,
            "action_space_n": str(self._action_bins),
            "action_dim": str(action_dim),
            "adapter": "dm_control_real",
            "domain_name": self._task.domain_name,
            "task_name": self._task.task_name,
        }


def create_dm_control_experiment1_environment(
    *,
    env_option: str,
    task_id: str,
    observation_mode: str = "image",
    max_steps: int | None = None,
) -> DmControlExperiment1Environment:
    return DmControlExperiment1Environment(
        env_option=env_option,
        task_id=task_id,
        observation_mode=observation_mode,
        max_steps=max_steps,
    )
