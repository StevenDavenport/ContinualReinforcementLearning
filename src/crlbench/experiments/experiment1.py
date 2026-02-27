from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from crlbench.config.schema import BudgetConfig
from crlbench.core.contracts import EnvironmentAdapter
from crlbench.core.plugins import register_env_family, register_experiment
from crlbench.core.types import Action, Observation, StepResult, TaskSpec
from crlbench.envs import create_dm_control_experiment1_environment, dm_control_available
from crlbench.runtime.task_streams import SequentialTaskStream

EXPERIMENT_ID = "experiment_1_forgetting"

TOY_TRACK = "toy"
ROBOTICS_TRACK = "robotics"
DM_CONTROL_BACKENDS: tuple[str, ...] = ("auto", "stub", "real")

BUDGET_TIERS: dict[str, BudgetConfig] = {
    "smoke": BudgetConfig(train_steps=20_000, eval_interval_steps=2_000, eval_episodes=10),
    "dev": BudgetConfig(train_steps=250_000, eval_interval_steps=25_000, eval_episodes=20),
    "full": BudgetConfig(train_steps=1_000_000, eval_interval_steps=50_000, eval_episodes=30),
}


@dataclass(frozen=True)
class RuntimeBound:
    max_wall_clock_minutes: int
    max_env_steps: int


RUNTIME_BOUNDS: dict[str, dict[tuple[str, str], RuntimeBound]] = {
    "smoke": {
        ("dm_control", "vision_sequential_default"): RuntimeBound(
            max_wall_clock_minutes=10,
            max_env_steps=100_000,
        ),
        ("dm_control", "vision_sequential_quadruped_recovery"): RuntimeBound(
            max_wall_clock_minutes=10,
            max_env_steps=100_000,
        ),
        ("dm_control", "vision_sequential_quadruped_anchor_escape"): RuntimeBound(
            max_wall_clock_minutes=10,
            max_env_steps=100_000,
        ),
        ("procgen", "vision_sequential_alternative"): RuntimeBound(
            max_wall_clock_minutes=12,
            max_env_steps=120_000,
        ),
        ("metaworld", "manipulation_sequential_default"): RuntimeBound(
            max_wall_clock_minutes=20,
            max_env_steps=120_000,
        ),
        ("maniskill", "manipulation_sequential_alternative"): RuntimeBound(
            max_wall_clock_minutes=25,
            max_env_steps=150_000,
        ),
    },
    "dev": {
        ("dm_control", "vision_sequential_default"): RuntimeBound(
            max_wall_clock_minutes=90,
            max_env_steps=1_250_000,
        ),
        ("dm_control", "vision_sequential_quadruped_recovery"): RuntimeBound(
            max_wall_clock_minutes=90,
            max_env_steps=1_250_000,
        ),
        ("dm_control", "vision_sequential_quadruped_anchor_escape"): RuntimeBound(
            max_wall_clock_minutes=90,
            max_env_steps=1_250_000,
        ),
        ("procgen", "vision_sequential_alternative"): RuntimeBound(
            max_wall_clock_minutes=110,
            max_env_steps=1_250_000,
        ),
        ("metaworld", "manipulation_sequential_default"): RuntimeBound(
            max_wall_clock_minutes=180,
            max_env_steps=1_500_000,
        ),
        ("maniskill", "manipulation_sequential_alternative"): RuntimeBound(
            max_wall_clock_minutes=240,
            max_env_steps=1_750_000,
        ),
    },
    "full": {
        ("dm_control", "vision_sequential_default"): RuntimeBound(
            max_wall_clock_minutes=360,
            max_env_steps=5_000_000,
        ),
        ("dm_control", "vision_sequential_quadruped_recovery"): RuntimeBound(
            max_wall_clock_minutes=360,
            max_env_steps=5_000_000,
        ),
        ("dm_control", "vision_sequential_quadruped_anchor_escape"): RuntimeBound(
            max_wall_clock_minutes=360,
            max_env_steps=5_000_000,
        ),
        ("procgen", "vision_sequential_alternative"): RuntimeBound(
            max_wall_clock_minutes=420,
            max_env_steps=5_000_000,
        ),
        ("metaworld", "manipulation_sequential_default"): RuntimeBound(
            max_wall_clock_minutes=600,
            max_env_steps=6_000_000,
        ),
        ("maniskill", "manipulation_sequential_alternative"): RuntimeBound(
            max_wall_clock_minutes=720,
            max_env_steps=7_000_000,
        ),
    },
}

TASK_TEMPLATES: dict[tuple[str, str, str], tuple[str, ...]] = {
    (
        TOY_TRACK,
        "dm_control",
        "vision_sequential_default",
    ): (
        "walker_walk",
        "cheetah_run",
        "quadruped_walk",
        "humanoid_walk",
    ),
    (
        TOY_TRACK,
        "dm_control",
        "vision_sequential_quadruped_recovery",
    ): (
        "quadruped_run",
        "quadruped_fetch",
        "quadruped_escape",
    ),
    (
        TOY_TRACK,
        "dm_control",
        "vision_sequential_quadruped_anchor_escape",
    ): (
        "quadruped_escape",
        "quadruped_run",
        "quadruped_fetch",
    ),
    (
        TOY_TRACK,
        "procgen",
        "vision_sequential_alternative",
    ): (
        "coinrun",
        "heist",
        "jumper",
        "ninja",
    ),
    (
        ROBOTICS_TRACK,
        "metaworld",
        "manipulation_sequential_default",
    ): (
        "reach",
        "push",
        "pick_place",
        "drawer_open",
    ),
    (
        ROBOTICS_TRACK,
        "maniskill",
        "manipulation_sequential_alternative",
    ): (
        "pick_cube",
        "stack_cube",
        "plug_charger",
        "open_cabinet_drawer",
    ),
}


@dataclass(frozen=True)
class EnvironmentProfile:
    action_space_n: int
    max_steps: int
    friction: float
    control_scale: float
    noise_scale: float
    reward_scale: float
    pixel_size: int


ENVIRONMENT_PROFILES: dict[tuple[str, str], EnvironmentProfile] = {
    ("dm_control", "vision_sequential_default"): EnvironmentProfile(
        action_space_n=5,
        max_steps=96,
        friction=0.05,
        control_scale=0.22,
        noise_scale=0.01,
        reward_scale=1.0,
        pixel_size=4,
    ),
    ("dm_control", "vision_sequential_quadruped_recovery"): EnvironmentProfile(
        action_space_n=5,
        max_steps=96,
        friction=0.05,
        control_scale=0.22,
        noise_scale=0.01,
        reward_scale=1.0,
        pixel_size=4,
    ),
    ("dm_control", "vision_sequential_quadruped_anchor_escape"): EnvironmentProfile(
        action_space_n=5,
        max_steps=96,
        friction=0.05,
        control_scale=0.22,
        noise_scale=0.01,
        reward_scale=1.0,
        pixel_size=4,
    ),
    ("procgen", "vision_sequential_alternative"): EnvironmentProfile(
        action_space_n=9,
        max_steps=96,
        friction=0.08,
        control_scale=0.28,
        noise_scale=0.02,
        reward_scale=1.0,
        pixel_size=5,
    ),
    ("metaworld", "manipulation_sequential_default"): EnvironmentProfile(
        action_space_n=7,
        max_steps=120,
        friction=0.04,
        control_scale=0.18,
        noise_scale=0.015,
        reward_scale=1.1,
        pixel_size=4,
    ),
    ("maniskill", "manipulation_sequential_alternative"): EnvironmentProfile(
        action_space_n=7,
        max_steps=120,
        friction=0.03,
        control_scale=0.2,
        noise_scale=0.015,
        reward_scale=1.15,
        pixel_size=4,
    ),
}


@dataclass(frozen=True)
class EvalStagePlan:
    stage_id: str
    train_task_id: str
    eval_task_ids: tuple[str, ...]


@dataclass(frozen=True)
class Experiment1Protocol:
    experiment_id: str
    track: str
    env_family: str
    env_option: str
    budget_tier: str
    budget: BudgetConfig
    tasks: tuple[TaskSpec, ...]
    eval_plan: tuple[EvalStagePlan, ...]
    runtime_bound: RuntimeBound
    reporting_template: dict[str, Any]


def _resolve_template(track: str, env_family: str, env_option: str) -> tuple[str, ...]:
    key = (track, env_family, env_option)
    template = TASK_TEMPLATES.get(key)
    if template is None:
        known = ", ".join(
            f"{t}/{family}/{option}" for (t, family, option) in sorted(TASK_TEMPLATES)
        )
        raise ValueError(
            f"Unsupported Experiment 1 root '{track}/{env_family}/{env_option}'. "
            f"Known roots: {known}."
        )
    return template


def build_experiment1_tasks(
    *,
    track: str,
    env_family: str,
    env_option: str,
) -> tuple[TaskSpec, ...]:
    template = _resolve_template(track, env_family, env_option)
    return tuple(
        TaskSpec(
            task_id=task_id,
            env_family=env_family,
            env_option=env_option,
            metadata={
                "experiment": EXPERIMENT_ID,
                "track": track,
                "stage_index": stage_index,
            },
        )
        for stage_index, task_id in enumerate(template)
    )


def build_experiment1_eval_plan(tasks: Sequence[TaskSpec]) -> tuple[EvalStagePlan, ...]:
    if not tasks:
        raise ValueError("Experiment 1 eval plan requires at least one task.")
    stage_plans: list[EvalStagePlan] = []
    seen: list[str] = []
    for task in tasks:
        seen.append(task.task_id)
        stage_plans.append(
            EvalStagePlan(
                stage_id=f"after_{task.task_id}",
                train_task_id=task.task_id,
                eval_task_ids=tuple(seen),
            )
        )
    return tuple(stage_plans)


def experiment1_reporting_template() -> dict[str, Any]:
    return {
        "primary_metrics": [
            "final_stage_average_return",
            "average_forgetting",
            "average_retention",
        ],
        "task_metrics": [
            "forgetting_by_task",
            "retention_by_task",
        ],
        "stage_metrics": [
            "average_return_by_stage",
            "evaluation_matrix",
        ],
        "required_plots": [
            "exp1_forgetting_curve",
            "exp2_transfer",
            "exp4_recall_retention",
        ],
    }


def get_experiment1_budget(tier: str) -> BudgetConfig:
    budget = BUDGET_TIERS.get(tier)
    if budget is None:
        known = ", ".join(sorted(BUDGET_TIERS))
        raise ValueError(f"Unknown Experiment 1 budget tier '{tier}'. Known tiers: {known}.")
    return budget


def get_experiment1_runtime_bound(
    *,
    tier: str,
    env_family: str,
    env_option: str,
) -> RuntimeBound:
    tier_bounds = RUNTIME_BOUNDS.get(tier)
    if tier_bounds is None:
        known = ", ".join(sorted(RUNTIME_BOUNDS))
        raise ValueError(f"Unknown runtime-bound tier '{tier}'. Known tiers: {known}.")
    bound = tier_bounds.get((env_family, env_option))
    if bound is None:
        raise ValueError(f"No runtime bound for root '{env_family}/{env_option}' at tier '{tier}'.")
    return bound


def build_experiment1_protocol(
    *,
    track: str,
    env_family: str,
    env_option: str,
    budget_tier: str,
) -> Experiment1Protocol:
    tasks = build_experiment1_tasks(track=track, env_family=env_family, env_option=env_option)
    return Experiment1Protocol(
        experiment_id=EXPERIMENT_ID,
        track=track,
        env_family=env_family,
        env_option=env_option,
        budget_tier=budget_tier,
        budget=get_experiment1_budget(budget_tier),
        tasks=tasks,
        eval_plan=build_experiment1_eval_plan(tasks),
        runtime_bound=get_experiment1_runtime_bound(
            tier=budget_tier,
            env_family=env_family,
            env_option=env_option,
        ),
        reporting_template=experiment1_reporting_template(),
    )


def build_experiment1_task_stream(
    *,
    track: str,
    env_family: str,
    env_option: str,
) -> SequentialTaskStream:
    tasks = build_experiment1_tasks(track=track, env_family=env_family, env_option=env_option)
    return SequentialTaskStream(tasks=list(tasks))


def list_experiment1_roots() -> tuple[tuple[str, str, str], ...]:
    return tuple(sorted(TASK_TEMPLATES))


def get_experiment1_action_space_n(*, env_family: str, env_option: str) -> int:
    profile = ENVIRONMENT_PROFILES.get((env_family, env_option))
    if profile is None:
        raise ValueError(f"Unknown Experiment 1 environment root: {env_family}/{env_option}.")
    return profile.action_space_n


class Experiment1StubEnvironment(EnvironmentAdapter):
    """
    Lightweight deterministic adapter used for protocol tests and smoke scaffolding.

    Real simulator bindings are layered in later phases per env family.
    """

    def __init__(
        self,
        *,
        env_family: str,
        env_option: str,
        task_id: str,
        observation_mode: str = "image",
        max_steps: int | None = None,
    ) -> None:
        _resolve_template(
            TOY_TRACK if env_family in {"dm_control", "procgen"} else ROBOTICS_TRACK,
            env_family,
            env_option,
        )
        profile = ENVIRONMENT_PROFILES.get((env_family, env_option))
        if profile is None:
            raise ValueError(f"Unknown environment profile for {env_family}/{env_option}.")
        if max_steps is not None and max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}.")
        self._env_family = env_family
        self._env_option = env_option
        self._task_id = task_id
        self._observation_mode = observation_mode
        self._profile = profile
        self._max_steps = max_steps or profile.max_steps
        self._step_count = 0
        self._rng = random.Random(0)
        self._position = 0.0
        self._velocity = 0.0
        self._target = self._task_target(task_id)
        self._stability_steps = 0

    def _task_target(self, task_id: str) -> float:
        digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
        value = int(digest[:6], 16) / float(0xFFFFFF)
        return (value * 2.0) - 1.0

    def _action_force(self, action: int) -> float:
        bins = self._profile.action_space_n
        action_index = max(0, min(bins - 1, action))
        center = (bins - 1) / 2.0
        normalized = (action_index - center) / max(1.0, center)
        return normalized * self._profile.control_scale

    def _pixel_grid(self) -> list[list[int]]:
        size = self._profile.pixel_size
        base = (self._position + 1.0) * 0.5
        pixels: list[list[int]] = []
        for row in range(size):
            pixel_row: list[int] = []
            for col in range(size):
                value = base + (0.07 * row) + (0.03 * col)
                value += 0.1 * math.sin(self._step_count * 0.1 + row + col)
                value_clamped = max(0.0, min(1.0, value))
                pixel_row.append(int(round(value_clamped * 255.0)))
            pixels.append(pixel_row)
        return pixels

    def _observation(self) -> Observation:
        payload: dict[str, Any] = {
            "pixels": self._pixel_grid(),
            "task": self._task_id,
            "step": self._step_count,
            "action_space_n": self._profile.action_space_n,
            "position": self._position,
            "velocity": self._velocity,
        }
        if self._observation_mode == "image_proprio":
            payload["proprio"] = [self._position, self._velocity, self._target - self._position]
        return payload

    def reset(self, seed: int | None = None) -> Observation:
        self._step_count = 0
        seed_value = 0 if seed is None else int(seed)
        self._rng = random.Random(seed_value)
        self._position = self._rng.uniform(-0.25, 0.25)
        self._velocity = 0.0
        self._stability_steps = 0
        return self._observation()

    def step(self, action: Action) -> StepResult:
        action_index = int(action) if isinstance(action, int) else 0
        force = self._action_force(action_index)
        noise = self._rng.uniform(-self._profile.noise_scale, self._profile.noise_scale)
        self._velocity = (1.0 - self._profile.friction) * self._velocity + force + noise
        self._position += self._velocity
        self._position = max(-2.0, min(2.0, self._position))
        self._step_count += 1

        error = abs(self._target - self._position)
        progress_reward = max(0.0, 1.0 - error)
        reward = progress_reward * self._profile.reward_scale
        if error < 0.1:
            self._stability_steps += 1
            reward += 0.1
        else:
            self._stability_steps = 0
        terminated = self._stability_steps >= 8
        truncated = self._step_count >= self._max_steps

        optimal_force = (self._target - self._position) * self._profile.control_scale
        return StepResult(
            observation=self._observation(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={
                "env_family": self._env_family,
                "env_option": self._env_option,
                "target": self._target,
                "optimal_force": optimal_force,
            },
        )

    def close(self) -> None:
        return None

    @property
    def metadata(self) -> Mapping[str, str]:
        return {
            "env_family": self._env_family,
            "env_option": self._env_option,
            "task_id": self._task_id,
            "observation_mode": self._observation_mode,
            "action_space_n": str(self._profile.action_space_n),
            "adapter": "stub",
        }


def _track_for_env_family(env_family: str) -> str:
    if env_family in {"dm_control", "procgen"}:
        return TOY_TRACK
    if env_family in {"metaworld", "maniskill"}:
        return ROBOTICS_TRACK
    raise ValueError(f"Unsupported Experiment 1 env family '{env_family}'.")


def _resolve_dm_control_backend(dm_control_backend: str) -> str:
    normalized = dm_control_backend.strip().lower()
    if normalized not in DM_CONTROL_BACKENDS:
        known = ", ".join(DM_CONTROL_BACKENDS)
        raise ValueError(
            f"Unknown dm_control backend '{dm_control_backend}'. Known backends: {known}."
        )
    if normalized == "real" and not dm_control_available():
        raise ValueError(
            "dm_control backend 'real' requested but dm_control is not installed. "
            "Install with: python -m pip install -e '.[dm_control]'."
        )
    if normalized == "auto":
        return "real" if dm_control_available() else "stub"
    return normalized


def create_experiment1_dm_control_environment(
    *,
    env_option: str,
    task_id: str,
    observation_mode: str = "image",
    max_steps: int | None = None,
    dm_control_backend: str = "auto",
) -> EnvironmentAdapter:
    _resolve_template(TOY_TRACK, "dm_control", env_option)
    resolved_backend = _resolve_dm_control_backend(dm_control_backend)
    if resolved_backend == "real":
        return create_dm_control_experiment1_environment(
            env_option=env_option,
            task_id=task_id,
            observation_mode=observation_mode,
            max_steps=max_steps,
        )
    return create_experiment1_stub_environment(
        env_family="dm_control",
        env_option=env_option,
        task_id=task_id,
        observation_mode=observation_mode,
        max_steps=max_steps,
    )


def create_experiment1_stub_environment(
    *,
    env_family: str,
    env_option: str,
    task_id: str,
    observation_mode: str = "image",
    max_steps: int | None = None,
) -> Experiment1StubEnvironment:
    _resolve_template(_track_for_env_family(env_family), env_family, env_option)
    return Experiment1StubEnvironment(
        env_family=env_family,
        env_option=env_option,
        task_id=task_id,
        observation_mode=observation_mode,
        max_steps=max_steps,
    )


def register_experiment1_plugins(*, replace: bool = True, dm_control_backend: str = "auto") -> None:
    resolved_dm_control_backend = _resolve_dm_control_backend(dm_control_backend)
    register_experiment(
        EXPERIMENT_ID,
        lambda **kwargs: build_experiment1_protocol(**kwargs),
        replace=replace,
    )
    register_env_family(
        "dm_control",
        lambda **kwargs: create_experiment1_dm_control_environment(
            dm_control_backend=resolved_dm_control_backend,
            **kwargs,
        ),
        replace=replace,
    )
    register_env_family(
        "procgen",
        lambda **kwargs: create_experiment1_stub_environment(env_family="procgen", **kwargs),
        replace=replace,
    )
    register_env_family(
        "metaworld",
        lambda **kwargs: create_experiment1_stub_environment(env_family="metaworld", **kwargs),
        replace=replace,
    )
    register_env_family(
        "maniskill",
        lambda **kwargs: create_experiment1_stub_environment(env_family="maniskill", **kwargs),
        replace=replace,
    )
