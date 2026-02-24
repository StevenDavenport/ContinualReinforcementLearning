from __future__ import annotations

import pytest

from crlbench.core.types import TaskSpec
from crlbench.runtime.task_streams import SequentialTaskStream


def _tasks() -> list[TaskSpec]:
    return [
        TaskSpec(task_id="A", env_family="dm_control", env_option="vision"),
        TaskSpec(task_id="B", env_family="dm_control", env_option="vision"),
        TaskSpec(task_id="C", env_family="dm_control", env_option="vision"),
    ]


def test_sequential_task_stream_order_and_reset() -> None:
    stream = SequentialTaskStream(_tasks())

    first = stream.next_task()
    second = stream.next_task()
    assert first is not None
    assert second is not None
    assert first.task_id == "A"
    assert second.task_id == "B"
    assert stream.remaining() == 1

    stream.reset()
    replayed = stream.next_task()
    assert replayed is not None
    assert replayed.task_id == "A"


def test_sequential_task_stream_state_round_trip() -> None:
    stream = SequentialTaskStream(_tasks())
    _ = stream.next_task()
    state = stream.get_state()
    assert state["index"] == 1
    assert state["size"] == 3

    stream.set_state({"index": 2, "size": 3})
    task = stream.next_task()
    assert task is not None
    assert task.task_id == "C"


def test_sequential_task_stream_rejects_invalid_state() -> None:
    stream = SequentialTaskStream(_tasks())

    with pytest.raises(ValueError, match="requires integer 'index'"):
        stream.set_state({"index": "1"})

    with pytest.raises(ValueError, match="out of bounds"):
        stream.set_state({"index": 4})
