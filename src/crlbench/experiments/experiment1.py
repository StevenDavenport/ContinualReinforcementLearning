from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from crlbench.config.schema import BudgetConfig
from crlbench.core.contracts import EnvironmentAdapter
from crlbench.core.plugins import register_env_family, register_experiment
from crlbench.core.types import Action, Observation, StepResult, TaskSpec
from crlbench.runtime.task_streams import SequentialTaskStream

EXPERIMENT_ID = "experiment_1_forgetting"

TOY_TRACK = "toy"
ROBOTICS_TRACK = "robotics"

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
        max_steps: int = 64,
    ) -> None:
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}.")
        _resolve_template(
            TOY_TRACK if env_family in {"dm_control", "procgen"} else ROBOTICS_TRACK,
            env_family,
            env_option,
        )
        self._env_family = env_family
        self._env_option = env_option
        self._task_id = task_id
        self._observation_mode = observation_mode
        self._max_steps = max_steps
        self._step_count = 0

    def reset(self, seed: int | None = None) -> Observation:
        self._step_count = 0
        seed_value = 0 if seed is None else seed
        return {
            "pixels": [[seed_value % 255, 0], [0, 0]],
            "task": self._task_id,
            "step": self._step_count,
        }

    def step(self, action: Action) -> StepResult:
        _ = action
        self._step_count += 1
        terminated = self._step_count >= self._max_steps
        reward = 1.0 - (self._step_count / float(self._max_steps))
        return StepResult(
            observation={
                "pixels": [[self._step_count % 255, 0], [0, 0]],
                "task": self._task_id,
                "step": self._step_count,
            },
            reward=reward,
            terminated=terminated,
            truncated=False,
            info={"env_family": self._env_family, "env_option": self._env_option},
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
            "adapter": "stub",
        }


def _track_for_env_family(env_family: str) -> str:
    if env_family in {"dm_control", "procgen"}:
        return TOY_TRACK
    if env_family in {"metaworld", "maniskill"}:
        return ROBOTICS_TRACK
    raise ValueError(f"Unsupported Experiment 1 env family '{env_family}'.")


def create_experiment1_stub_environment(
    *,
    env_family: str,
    env_option: str,
    task_id: str,
    observation_mode: str = "image",
    max_steps: int = 64,
) -> Experiment1StubEnvironment:
    _resolve_template(_track_for_env_family(env_family), env_family, env_option)
    return Experiment1StubEnvironment(
        env_family=env_family,
        env_option=env_option,
        task_id=task_id,
        observation_mode=observation_mode,
        max_steps=max_steps,
    )


def register_experiment1_plugins(*, replace: bool = True) -> None:
    register_experiment(
        EXPERIMENT_ID,
        lambda **kwargs: build_experiment1_protocol(**kwargs),
        replace=replace,
    )
    register_env_family(
        "dm_control",
        lambda **kwargs: create_experiment1_stub_environment(env_family="dm_control", **kwargs),
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
