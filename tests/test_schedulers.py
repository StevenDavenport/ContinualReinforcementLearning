from __future__ import annotations

from crlbench.core.types import TaskSpec
from crlbench.runtime.task_streams import (
    CurriculumStage,
    CurriculumTaskStream,
    CyclicTaskStream,
    StochasticTaskStream,
)


def _tasks() -> list[TaskSpec]:
    return [
        TaskSpec(task_id="A", env_family="dm_control", env_option="vision"),
        TaskSpec(task_id="B", env_family="dm_control", env_option="vision"),
        TaskSpec(task_id="C", env_family="dm_control", env_option="vision"),
    ]


def test_cyclic_task_stream_order() -> None:
    stream = CyclicTaskStream(_tasks()[:2], cycles=2)
    observed = []
    while True:
        task = stream.next_task()
        if task is None:
            break
        observed.append(task.task_id)
    assert observed == ["A", "B", "A", "B"]
    assert stream.remaining() == 0


def test_stochastic_task_stream_seed_reproducibility() -> None:
    lhs = StochasticTaskStream(_tasks(), seed=9, replacement=True, total_tasks=5)
    rhs = StochasticTaskStream(_tasks(), seed=9, replacement=True, total_tasks=5)
    left: list[str] = []
    right: list[str] = []
    for _ in range(5):
        left_task = lhs.next_task()
        right_task = rhs.next_task()
        assert left_task is not None
        assert right_task is not None
        left.append(left_task.task_id)
        right.append(right_task.task_id)
    assert left == right


def test_stochastic_task_stream_state_round_trip() -> None:
    stream = StochasticTaskStream(_tasks(), seed=3, replacement=False, total_tasks=6)
    first = stream.next_task()
    assert first is not None
    state = stream.get_state()
    replay = StochasticTaskStream(_tasks(), seed=3, replacement=False, total_tasks=6)
    replay.set_state(state)
    assert replay.next_task() == stream.next_task()


def test_curriculum_task_stream_stage_order() -> None:
    stream = CurriculumTaskStream(
        stages=[
            CurriculumStage(stage_id="s1", tasks=_tasks()[:2], repeats=1),
            CurriculumStage(stage_id="s2", tasks=_tasks()[2:], repeats=2),
        ]
    )
    observed = []
    while True:
        task = stream.next_task()
        if task is None:
            break
        observed.append(task.task_id)
    assert observed == ["A", "B", "C", "C"]
    assert stream.current_stage_id() is None
