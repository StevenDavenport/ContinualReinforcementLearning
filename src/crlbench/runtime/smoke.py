from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crlbench.config.schema import RunConfig
from crlbench.core.types import TaskSpec

from .artifacts import ArtifactStore
from .orchestrator import RunOrchestrator, RunSummary
from .task_streams import SequentialTaskStream


def build_smoke_tasks(config: RunConfig, *, count: int) -> list[TaskSpec]:
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}.")
    return [
        TaskSpec(
            task_id=f"smoke_{index:02d}",
            env_family=config.env_family,
            env_option=config.env_option,
            metadata={"smoke": True, "index": index},
        )
        for index in range(count)
    ]


def read_event_records(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "events.jsonl"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines]


def normalize_events_for_repro(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for event in events:
        payload = dict(event.get("payload", {}))
        payload.pop("run_id", None)
        normalized.append(
            {
                "schema_version": event.get("schema_version"),
                "sequence": event.get("sequence"),
                "event": event.get("event"),
                "payload": payload,
            }
        )
    return normalized


def extract_metric_trace(events: list[dict[str, Any]]) -> list[float]:
    trace: list[float] = []
    for event in events:
        if event.get("event") != "task_start":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        task_index = payload.get("task_index")
        if isinstance(task_index, int):
            trace.append(float(task_index + 1))
    return trace


def compare_metric_traces(
    lhs: list[float],
    rhs: list[float],
    *,
    tolerance: float,
) -> tuple[bool, float]:
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}.")
    if len(lhs) != len(rhs):
        return (False, float("inf"))
    if not lhs:
        return (True, 0.0)
    max_abs_diff = max(abs(left - right) for left, right in zip(lhs, rhs, strict=True))
    return (max_abs_diff <= tolerance, max_abs_diff)


@dataclass(frozen=True)
class ReproSmokeResult:
    matched: bool
    event_trace_matched: bool
    metric_trace_matched: bool
    metric_tolerance: float
    max_metric_trace_abs_diff: float
    first: RunSummary
    second: RunSummary


def run_smoke(
    *,
    config: RunConfig,
    max_tasks: int,
    output_dir: Path | None = None,
    project_root: Path | None = None,
) -> RunSummary:
    if max_tasks <= 0:
        raise ValueError(f"max_tasks must be positive, got {max_tasks}.")
    tasks = build_smoke_tasks(config, count=max_tasks)
    stream = SequentialTaskStream(tasks)
    store = ArtifactStore(output_dir or config.output_dir)
    orchestrator = RunOrchestrator(
        config=config,
        artifact_store=store,
        task_stream=stream,
        project_root=project_root,
    )
    return orchestrator.run(dry_run=True, max_tasks=max_tasks)


def run_repro_smoke(
    *,
    config: RunConfig,
    max_tasks: int,
    metric_tolerance: float = 0.0,
    output_dir: Path | None = None,
    project_root: Path | None = None,
) -> ReproSmokeResult:
    first = run_smoke(
        config=config,
        max_tasks=max_tasks,
        output_dir=output_dir,
        project_root=project_root,
    )
    second = run_smoke(
        config=config,
        max_tasks=max_tasks,
        output_dir=output_dir,
        project_root=project_root,
    )
    first_events = normalize_events_for_repro(read_event_records(first.run_dir))
    second_events = normalize_events_for_repro(read_event_records(second.run_dir))
    event_trace_matched = first_events == second_events
    first_metric_trace = extract_metric_trace(first_events)
    second_metric_trace = extract_metric_trace(second_events)
    metric_trace_matched, max_metric_trace_abs_diff = compare_metric_traces(
        first_metric_trace,
        second_metric_trace,
        tolerance=metric_tolerance,
    )
    return ReproSmokeResult(
        matched=event_trace_matched and metric_trace_matched,
        event_trace_matched=event_trace_matched,
        metric_trace_matched=metric_trace_matched,
        metric_tolerance=metric_tolerance,
        max_metric_trace_abs_diff=max_metric_trace_abs_diff,
        first=first,
        second=second,
    )
