from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvalTrigger:
    step: int
    reason: str


class EvalScheduler:
    """
    Shared evaluator trigger policy:
    - periodic step-based evaluation,
    - stage-end evaluation,
    - switch-point evaluation.
    """

    def __init__(
        self,
        *,
        periodic_interval_steps: int | None = None,
        evaluate_on_stage_end: bool = True,
        switch_point_steps: Iterable[int] | None = None,
    ) -> None:
        if periodic_interval_steps is not None and periodic_interval_steps <= 0:
            raise ValueError(
                f"periodic_interval_steps must be positive when set, got {periodic_interval_steps}."
            )
        self.periodic_interval_steps = periodic_interval_steps
        self.evaluate_on_stage_end = evaluate_on_stage_end
        self.switch_point_steps = sorted(set(int(step) for step in (switch_point_steps or [])))

    def due_triggers(
        self,
        *,
        step: int,
        stage_ended: bool = False,
        switch_occurred: bool = False,
    ) -> list[EvalTrigger]:
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}.")
        triggers: list[EvalTrigger] = []
        if (
            self.periodic_interval_steps is not None
            and step > 0
            and step % self.periodic_interval_steps == 0
        ):
            triggers.append(EvalTrigger(step=step, reason="periodic"))
        if self.evaluate_on_stage_end and stage_ended:
            triggers.append(EvalTrigger(step=step, reason="stage_end"))
        if switch_occurred and step in self.switch_point_steps:
            triggers.append(EvalTrigger(step=step, reason="switch_point"))
        return triggers

    def get_state(self) -> dict[str, Any]:
        return {
            "periodic_interval_steps": self.periodic_interval_steps,
            "evaluate_on_stage_end": self.evaluate_on_stage_end,
            "switch_point_steps": list(self.switch_point_steps),
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        periodic = state.get("periodic_interval_steps")
        evaluate_on_stage_end = state.get("evaluate_on_stage_end")
        switch_points = state.get("switch_point_steps")
        if periodic is not None and not isinstance(periodic, int):
            raise ValueError("periodic_interval_steps must be int or null.")
        if not isinstance(evaluate_on_stage_end, bool):
            raise ValueError("evaluate_on_stage_end must be bool.")
        if not isinstance(switch_points, list) or not all(
            isinstance(step, int) and step >= 0 for step in switch_points
        ):
            raise ValueError("switch_point_steps must be a list of non-negative ints.")
        self.periodic_interval_steps = periodic
        self.evaluate_on_stage_end = evaluate_on_stage_end
        self.switch_point_steps = sorted(set(switch_points))
