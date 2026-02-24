from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest

from crlbench.core.contracts import (
    AgentAdapter,
    EnvironmentAdapter,
    Evaluator,
    TaskStream,
)
from crlbench.core.types import EvalScore, StepResult, TaskSpec, Transition
from crlbench.core.validation import (
    SchemaValidationError,
    validate_observation,
    validate_step_result,
    validate_transition,
)


class DummyAgent:
    def reset(self) -> None:
        return None

    def act(self, observation: Mapping[str, object], *, deterministic: bool = False) -> int:
        _ = deterministic
        _ = observation
        return 0

    def update(self, batch: Sequence[Transition]) -> Mapping[str, float]:
        _ = batch
        return {"loss": 0.0}

    def save(self, path: Path) -> None:
        _ = path


class DummyEnv:
    def reset(self, seed: int | None = None) -> Mapping[str, object]:
        _ = seed
        return {"obs": 0.0}

    def step(self, action: int) -> StepResult:
        _ = action
        return StepResult(observation={"obs": 0.0}, reward=0.0, terminated=False, truncated=False)

    def close(self) -> None:
        return None

    @property
    def metadata(self) -> Mapping[str, str]:
        return {"family": "dummy"}


class DummyTaskStream:
    def __init__(self) -> None:
        self._done = False

    def reset(self) -> None:
        self._done = False

    def next_task(self) -> TaskSpec | None:
        if self._done:
            return None
        self._done = True
        return TaskSpec(task_id="A", env_family="dummy", env_option="default")

    def remaining(self) -> int:
        return 0 if self._done else 1


class DummyEvaluator:
    def evaluate(
        self,
        *,
        agent: AgentAdapter,
        env_factory: Callable[[TaskSpec], EnvironmentAdapter],
        tasks: Sequence[TaskSpec],
        step: int,
        seed: int,
    ) -> Sequence[EvalScore]:
        _ = agent
        _ = env_factory
        _ = tasks
        _ = step
        _ = seed
        return [EvalScore(task_id="A", step=0, mean_return=0.0, std_return=0.0, episodes=1)]


class BrokenEnv:
    def reset(self, seed: int | None = None) -> Mapping[str, object]:
        _ = seed
        return {"obs": 0.0}

    def step(self, action: int) -> StepResult:
        _ = action
        # Invalid reward should be rejected by schema validation.
        return StepResult(
            observation={"obs": 0.0},
            reward=float("nan"),
            terminated=False,
            truncated=False,
        )

    def close(self) -> None:
        return None

    @property
    def metadata(self) -> Mapping[str, str]:
        return {"family": "broken"}


def assert_agent_contract(agent: AgentAdapter) -> None:
    agent.reset()
    observation = {"obs": 0.0}
    action = agent.act(observation, deterministic=True)
    transition = Transition(
        observation=observation,
        action=action,
        reward=1.0,
        next_observation={"obs": 1.0},
        terminated=False,
        truncated=False,
        info={},
    )
    validate_transition(transition)
    update_metrics = agent.update([transition])
    assert update_metrics
    assert all(isinstance(value, float) for value in update_metrics.values())
    checkpoint = Path("dummy.ckpt")
    agent.save(checkpoint)


def assert_environment_contract(env: EnvironmentAdapter) -> None:
    first_observation = env.reset(seed=7)
    validate_observation(first_observation)
    step = env.step(0)
    validate_step_result(step)
    assert all(isinstance(key, str) for key in env.metadata)
    env.close()


def assert_task_stream_contract(stream: TaskStream) -> None:
    stream.reset()
    task = stream.next_task()
    assert task is not None
    assert isinstance(task.task_id, str)
    assert stream.remaining() >= 0


def assert_evaluator_contract(
    evaluator: Evaluator,
    *,
    agent: AgentAdapter,
    env_factory: Callable[[TaskSpec], EnvironmentAdapter],
    tasks: Sequence[TaskSpec],
) -> None:
    scores = evaluator.evaluate(
        agent=agent,
        env_factory=env_factory,
        tasks=tasks,
        step=0,
        seed=0,
    )
    assert scores
    for score in scores:
        assert score.task_id
        assert score.episodes > 0


def test_runtime_checkable_contracts() -> None:
    assert isinstance(DummyAgent(), AgentAdapter)
    assert isinstance(DummyEnv(), EnvironmentAdapter)
    assert isinstance(DummyTaskStream(), TaskStream)
    assert isinstance(DummyEvaluator(), Evaluator)


def test_agent_contract_suite() -> None:
    assert_agent_contract(DummyAgent())


def test_environment_contract_suite() -> None:
    assert_environment_contract(DummyEnv())


def test_task_stream_contract_suite() -> None:
    assert_task_stream_contract(DummyTaskStream())


def test_evaluator_contract_suite() -> None:
    tasks = [TaskSpec(task_id="A", env_family="dummy", env_option="default")]
    assert_evaluator_contract(
        DummyEvaluator(),
        agent=DummyAgent(),
        env_factory=lambda _task: DummyEnv(),
        tasks=tasks,
    )


def test_environment_contract_suite_rejects_invalid_adapter_output() -> None:
    with pytest.raises(SchemaValidationError):
        assert_environment_contract(BrokenEnv())
