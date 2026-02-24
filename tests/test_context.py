from __future__ import annotations

from crlbench.core.types import TaskSpec
from crlbench.runtime.context import HiddenContextController


def test_hidden_context_controller_masks_public_context() -> None:
    controller = HiddenContextController(
        contexts=["ctx_a", "ctx_b"],
        min_dwell_steps=1,
        max_dwell_steps=1,
        seed=2,
    )
    event = controller.step()
    assert event.public_context is None
    assert event.internal_context in {"ctx_a", "ctx_b"}


def test_hidden_context_controller_state_round_trip() -> None:
    controller = HiddenContextController(
        contexts=["ctx_a", "ctx_b", "ctx_c"],
        min_dwell_steps=1,
        max_dwell_steps=2,
        seed=5,
    )
    first = controller.step()
    state = controller.get_state()

    resumed = HiddenContextController(
        contexts=["ctx_a", "ctx_b", "ctx_c"],
        min_dwell_steps=1,
        max_dwell_steps=2,
        seed=5,
    )
    resumed.set_state(state)
    assert resumed.step() == controller.step()
    assert first.public_context is None


def test_hidden_context_controller_masks_task_metadata() -> None:
    controller = HiddenContextController(contexts=["ctx"], seed=0)
    task = TaskSpec(
        task_id="task_A",
        env_family="dm_control",
        env_option="vision",
        metadata={"context_id": "ctx", "switch_boundary": True},
    )
    masked = controller.mask_task(task)
    assert "context_id" not in masked.metadata
    assert "switch_boundary" not in masked.metadata
    assert masked.metadata["context_masked"] is True
