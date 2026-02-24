from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any


class MetricsError(ValueError):
    """Raised when metric inputs are invalid."""


def _require_nonempty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MetricsError(f"'{name}' must be a non-empty string.")
    return value.strip()


def _require_finite_float(name: str, value: Any) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise MetricsError(f"'{name}' must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise MetricsError(f"'{name}' must be finite.")
    return result


@dataclass(frozen=True)
class StageEvaluation:
    stage_id: str
    task_returns: dict[str, float]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> StageEvaluation:
        stage_id = _require_nonempty_string("stage_id", data.get("stage_id"))
        returns_raw = data.get("task_returns")
        if not isinstance(returns_raw, Mapping) or not returns_raw:
            raise MetricsError("'task_returns' must be a non-empty mapping.")
        parsed: dict[str, float] = {}
        for key, value in returns_raw.items():
            task = _require_nonempty_string("task_returns key", key)
            parsed[task] = _require_finite_float(f"task_return[{task}]", value)
        return cls(stage_id=stage_id, task_returns=parsed)

    def to_dict(self) -> dict[str, Any]:
        return {"stage_id": self.stage_id, "task_returns": dict(self.task_returns)}


@dataclass(frozen=True)
class StreamEvaluation:
    stages: tuple[StageEvaluation, ...]

    def __post_init__(self) -> None:
        if not self.stages:
            raise MetricsError("StreamEvaluation requires at least one stage.")
        ids = [stage.stage_id for stage in self.stages]
        if len(ids) != len(set(ids)):
            raise MetricsError("Stage IDs must be unique.")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> StreamEvaluation:
        stages_raw = data.get("stages")
        if not isinstance(stages_raw, list) or not stages_raw:
            raise MetricsError("'stages' must be a non-empty list.")
        stages: list[StageEvaluation] = []
        for item in stages_raw:
            if not isinstance(item, Mapping):
                raise MetricsError("Each stage must be a mapping.")
            stages.append(StageEvaluation.from_mapping(item))
        return cls(stages=tuple(stages))

    def to_dict(self) -> dict[str, Any]:
        return {"stages": [stage.to_dict() for stage in self.stages]}


def _task_order(stream: StreamEvaluation) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    for stage in stream.stages:
        for task in stage.task_returns:
            if task not in seen:
                seen.add(task)
                order.append(task)
    return order


def evaluation_matrix(stream: StreamEvaluation) -> dict[str, Any]:
    task_ids = _task_order(stream)
    stage_ids = [stage.stage_id for stage in stream.stages]
    values: list[list[float | None]] = []
    for stage in stream.stages:
        row: list[float | None] = []
        for task_id in task_ids:
            row.append(stage.task_returns.get(task_id))
        values.append(row)
    return {"stage_ids": stage_ids, "task_ids": task_ids, "values": values}


def forgetting_matrix(stream: StreamEvaluation) -> dict[str, Any]:
    matrix = evaluation_matrix(stream)
    stage_ids: list[str] = matrix["stage_ids"]
    task_ids: list[str] = matrix["task_ids"]
    values: list[list[float | None]] = matrix["values"]

    out: list[list[float | None]] = []
    best_so_far: dict[str, float] = {}
    for row in values:
        out_row: list[float | None] = []
        for index, value in enumerate(row):
            task = task_ids[index]
            if value is None:
                out_row.append(None)
                continue
            previous_best = best_so_far.get(task, value)
            best_so_far[task] = max(previous_best, value)
            out_row.append(max(0.0, previous_best - value))
        out.append(out_row)
    return {"stage_ids": stage_ids, "task_ids": task_ids, "values": out}


def retention_matrix(stream: StreamEvaluation) -> dict[str, Any]:
    matrix = evaluation_matrix(stream)
    stage_ids: list[str] = matrix["stage_ids"]
    task_ids: list[str] = matrix["task_ids"]
    values: list[list[float | None]] = matrix["values"]

    out: list[list[float | None]] = []
    best_so_far: dict[str, float] = {}
    for row in values:
        out_row: list[float | None] = []
        for index, value in enumerate(row):
            task = task_ids[index]
            if value is None:
                out_row.append(None)
                continue
            previous_best = best_so_far.get(task, value)
            best_so_far[task] = max(previous_best, value)
            out_row.append(0.0 if previous_best == 0 else value / previous_best)
        out.append(out_row)
    return {"stage_ids": stage_ids, "task_ids": task_ids, "values": out}


def forgetting_by_task(stream: StreamEvaluation) -> dict[str, float]:
    per_task: dict[str, list[float]] = {}
    for stage in stream.stages:
        for task, value in stage.task_returns.items():
            per_task.setdefault(task, []).append(value)
    result: dict[str, float] = {}
    for task, values in per_task.items():
        if len(values) == 1:
            result[task] = 0.0
            continue
        best_past = max(values[:-1])
        result[task] = max(0.0, best_past - values[-1])
    return result


def retention_by_task(stream: StreamEvaluation) -> dict[str, float]:
    per_task: dict[str, list[float]] = {}
    for stage in stream.stages:
        for task, value in stage.task_returns.items():
            per_task.setdefault(task, []).append(value)
    result: dict[str, float] = {}
    for task, values in per_task.items():
        best = max(values)
        result[task] = 0.0 if best == 0 else values[-1] / best
    return result


def average_return_by_stage(stream: StreamEvaluation) -> dict[str, float]:
    result: dict[str, float] = {}
    for stage in stream.stages:
        result[stage.stage_id] = mean(stage.task_returns.values())
    return result


def forward_transfer_by_task(
    scratch_steps_by_task: Mapping[str, int | float],
    continual_steps_by_task: Mapping[str, int | float],
) -> dict[str, float]:
    keys = set(scratch_steps_by_task) & set(continual_steps_by_task)
    if not keys:
        raise MetricsError("No overlapping tasks for forward transfer computation.")
    result: dict[str, float] = {}
    for task in sorted(keys):
        scratch = _require_finite_float(f"scratch_steps[{task}]", scratch_steps_by_task[task])
        continual = _require_finite_float(
            f"continual_steps[{task}]",
            continual_steps_by_task[task],
        )
        if continual <= 0:
            raise MetricsError(f"continual_steps[{task}] must be > 0.")
        result[task] = scratch / continual
    return result


def backward_transfer_by_task(
    first_visit_returns: Mapping[str, int | float],
    revisit_returns: Mapping[str, int | float],
) -> dict[str, float]:
    keys = set(first_visit_returns) & set(revisit_returns)
    if not keys:
        raise MetricsError("No overlapping tasks for backward transfer computation.")
    result: dict[str, float] = {}
    for task in sorted(keys):
        first = _require_finite_float(f"first_visit[{task}]", first_visit_returns[task])
        revisit = _require_finite_float(f"revisit[{task}]", revisit_returns[task])
        result[task] = revisit - first
    return result


def switch_regret(reference_return: float, post_switch_returns: Sequence[float]) -> float:
    ref = _require_finite_float("reference_return", reference_return)
    if not post_switch_returns:
        raise MetricsError("post_switch_returns cannot be empty.")
    regrets = [
        max(0.0, ref - _require_finite_float("post_switch_return", x)) for x in post_switch_returns
    ]
    return float(sum(regrets))


def recovery_time(
    post_switch_returns: Sequence[float],
    target_return: float,
    *,
    tolerance: float = 0.0,
    step_stride: int = 1,
) -> int | None:
    if step_stride <= 0:
        raise MetricsError("step_stride must be positive.")
    target = _require_finite_float("target_return", target_return)
    tol = _require_finite_float("tolerance", tolerance)
    for index, value in enumerate(post_switch_returns):
        current = _require_finite_float("post_switch_return", value)
        if current >= (target - tol):
            return index * step_stride
    return None


def aggregate_scalar_values(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise MetricsError("values cannot be empty.")
    parsed = [_require_finite_float("value", value) for value in values]
    n = len(parsed)
    m = mean(parsed)
    std = pstdev(parsed) if n > 1 else 0.0
    stderr = std / math.sqrt(n) if n > 0 else 0.0
    ci95 = 1.96 * stderr
    return {"n": float(n), "mean": m, "std": std, "stderr": stderr, "ci95": ci95}


def confidence_interval_normal(
    values: Sequence[float],
    *,
    z_score: float = 1.96,
) -> tuple[float, float]:
    stats = aggregate_scalar_values(values)
    margin = z_score * stats["stderr"]
    return (stats["mean"] - margin, stats["mean"] + margin)


def bootstrap_confidence_interval(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    num_bootstrap: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    if not values:
        raise MetricsError("values cannot be empty.")
    if not (0.0 < confidence < 1.0):
        raise MetricsError("confidence must be in (0, 1).")
    if num_bootstrap <= 0:
        raise MetricsError("num_bootstrap must be positive.")
    samples = [_require_finite_float("value", value) for value in values]
    rng = random.Random(seed)
    n = len(samples)
    means = []
    for _ in range(num_bootstrap):
        draw = [samples[rng.randrange(n)] for _ in range(n)]
        means.append(mean(draw))
    means.sort()
    alpha = 1.0 - confidence
    lo_index = int((alpha / 2.0) * (num_bootstrap - 1))
    hi_index = int((1.0 - alpha / 2.0) * (num_bootstrap - 1))
    return means[lo_index], means[hi_index]


def paired_bootstrap_difference_ci(
    lhs: Sequence[float],
    rhs: Sequence[float],
    *,
    confidence: float = 0.95,
    num_bootstrap: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    if len(lhs) != len(rhs):
        raise MetricsError("lhs and rhs must have equal length.")
    if not lhs:
        raise MetricsError("lhs and rhs cannot be empty.")
    diffs = [
        _require_finite_float("lhs", left) - _require_finite_float("rhs", right)
        for left, right in zip(lhs, rhs, strict=True)
    ]
    return bootstrap_confidence_interval(
        diffs,
        confidence=confidence,
        num_bootstrap=num_bootstrap,
        seed=seed,
    )


def compute_stream_metrics(stream: StreamEvaluation) -> dict[str, Any]:
    per_task_forgetting = forgetting_by_task(stream)
    per_task_retention = retention_by_task(stream)
    stage_avg_return = average_return_by_stage(stream)

    forgetting_values = list(per_task_forgetting.values())
    retention_values = list(per_task_retention.values())
    final_stage_returns = list(stream.stages[-1].task_returns.values())

    return {
        "forgetting_by_task": per_task_forgetting,
        "retention_by_task": per_task_retention,
        "average_return_by_stage": stage_avg_return,
        "average_forgetting": mean(forgetting_values) if forgetting_values else 0.0,
        "average_retention": mean(retention_values) if retention_values else 0.0,
        "final_stage_average_return": mean(final_stage_returns) if final_stage_returns else 0.0,
        "evaluation_matrix": evaluation_matrix(stream),
        "forgetting_matrix": forgetting_matrix(stream),
        "retention_matrix": retention_matrix(stream),
    }
