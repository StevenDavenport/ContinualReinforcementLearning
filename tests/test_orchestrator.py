from __future__ import annotations

import json
from pathlib import Path

import pytest

from crlbench.config.schema import RunConfig
from crlbench.core.types import TaskSpec
from crlbench.runtime.artifacts import ArtifactStore
from crlbench.runtime.orchestrator import OrchestratorError, RunOrchestrator
from crlbench.runtime.task_streams import SequentialTaskStream


def _config() -> RunConfig:
    return RunConfig.from_mapping(
        {
            "run_name": "orchestrator_smoke",
            "experiment": "experiment_1_forgetting",
            "track": "toy",
            "env_family": "dm_control",
            "env_option": "vision",
            "seed": 0,
            "num_seeds": 5,
            "observation_mode": "image",
            "deterministic_mode": "auto",
            "seed_namespace": "global",
            "budget": {
                "train_steps": 1000,
                "eval_interval_steps": 100,
                "eval_episodes": 5,
            },
        }
    )


def _tasks() -> list[TaskSpec]:
    return [
        TaskSpec(task_id="A", env_family="dm_control", env_option="vision"),
        TaskSpec(task_id="B", env_family="dm_control", env_option="vision"),
        TaskSpec(task_id="C", env_family="dm_control", env_option="vision"),
    ]


def _load_events(run_dir: Path) -> list[dict[str, object]]:
    lines = (run_dir / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines]


def test_orchestrator_dry_run_writes_artifacts(tmp_path: Path) -> None:
    stream = SequentialTaskStream(_tasks())
    store = ArtifactStore(tmp_path / "artifacts")
    orchestrator = RunOrchestrator(config=_config(), artifact_store=store, task_stream=stream)

    summary = orchestrator.run(dry_run=True)

    assert summary.tasks_seen == 3
    assert summary.run_id
    assert summary.deterministic_enabled is False
    assert summary.completed is True
    assert (summary.run_dir / "manifest.json").exists()
    assert (summary.run_dir / "resolved_config.json").exists()
    assert (summary.run_dir / "events.jsonl").exists()
    assert (summary.run_dir / "state.json").exists()

    events = _load_events(summary.run_dir)
    assert events[0]["event"] == "run_start"
    assert events[-1]["event"] == "run_complete"

    state = json.loads((summary.run_dir / "state.json").read_text(encoding="utf-8"))
    assert state["tasks_seen"] == 3
    assert state["completed"] is True


def test_orchestrator_interrupt_and_resume(tmp_path: Path) -> None:
    stream = SequentialTaskStream(_tasks())
    store = ArtifactStore(tmp_path / "artifacts")
    orchestrator = RunOrchestrator(config=_config(), artifact_store=store, task_stream=stream)

    interrupted = orchestrator.run(dry_run=True, interrupt_after_tasks=2)
    assert interrupted.tasks_seen == 2
    assert interrupted.completed is False

    resumed = orchestrator.run(dry_run=True, resume=True, run_dir=interrupted.run_dir)
    assert resumed.tasks_seen == 3
    assert resumed.completed is True
    assert resumed.run_dir == interrupted.run_dir

    events = _load_events(resumed.run_dir)
    event_names = [record["event"] for record in events]
    assert "run_interrupted" in event_names
    assert "run_resumed" in event_names
    assert event_names[-1] == "run_complete"

    sequences = [record["sequence"] for record in events]
    assert sequences == list(range(len(events)))


def test_orchestrator_failure_injection_writes_state(tmp_path: Path) -> None:
    stream = SequentialTaskStream(_tasks())
    store = ArtifactStore(tmp_path / "artifacts")
    orchestrator = RunOrchestrator(config=_config(), artifact_store=store, task_stream=stream)

    with pytest.raises(OrchestratorError, match="Injected failure at task index 1"):
        orchestrator.run(dry_run=True, fail_on_task_index=1)

    run_dirs = list((tmp_path / "artifacts").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    events = _load_events(run_dir)
    assert events[-1]["event"] == "run_failed"

    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    last_sequence = events[-1]["sequence"]
    assert isinstance(last_sequence, int)
    assert state["tasks_seen"] == 1
    assert state["completed"] is False
    assert state["sequence"] == last_sequence + 1


def test_orchestrator_cannot_resume_completed_run(tmp_path: Path) -> None:
    stream = SequentialTaskStream(_tasks())
    store = ArtifactStore(tmp_path / "artifacts")
    orchestrator = RunOrchestrator(config=_config(), artifact_store=store, task_stream=stream)

    completed = orchestrator.run(dry_run=True)

    with pytest.raises(OrchestratorError, match="already marked completed"):
        orchestrator.run(dry_run=True, resume=True, run_dir=completed.run_dir)
