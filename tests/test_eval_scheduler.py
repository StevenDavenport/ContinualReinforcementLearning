from __future__ import annotations

from crlbench.runtime.eval_scheduler import EvalScheduler


def test_eval_scheduler_triggers_periodic_stage_and_switch() -> None:
    scheduler = EvalScheduler(
        periodic_interval_steps=10,
        evaluate_on_stage_end=True,
        switch_point_steps=[7, 20],
    )
    triggers = scheduler.due_triggers(step=20, stage_ended=True, switch_occurred=True)
    reasons = {trigger.reason for trigger in triggers}
    assert reasons == {"periodic", "stage_end", "switch_point"}


def test_eval_scheduler_state_round_trip() -> None:
    scheduler = EvalScheduler(periodic_interval_steps=50, switch_point_steps=[1, 2, 3])
    state = scheduler.get_state()
    restored = EvalScheduler(periodic_interval_steps=1)
    restored.set_state(state)
    assert restored.get_state() == state
