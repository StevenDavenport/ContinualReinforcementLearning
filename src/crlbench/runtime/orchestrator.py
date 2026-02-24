from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crlbench.config.schema import RunConfig
from crlbench.core.contracts import AgentAdapter, EnvironmentAdapter, Evaluator, TaskStream
from crlbench.core.types import TaskSpec
from crlbench.errors import OrchestrationError

from .artifacts import ArtifactStore
from .logging import create_logger
from .manifest import build_manifest, hash_payload
from .schemas import SCHEMA_VERSION, EventRecord, utc_now_iso
from .seeding import resolve_deterministic_mode

STATE_FILENAME = "state.json"


class OrchestratorError(OrchestrationError):
    """Raised when orchestration preconditions are not met."""


@dataclass(frozen=True)
class RunSummary:
    run_dir: Path
    run_id: str
    tasks_seen: int
    deterministic_enabled: bool
    completed: bool


class RunOrchestrator:
    """
    Thin orchestration shell with resume and interruption support.

    Experiment-specific train/eval logic is layered in later phases.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        config: RunConfig,
        artifact_store: ArtifactStore,
        task_stream: TaskStream,
        agent: AgentAdapter | None = None,
        evaluator: Evaluator | None = None,
        env_factory: Callable[[TaskSpec], EnvironmentAdapter] | None = None,
        project_root: Path | None = None,
    ) -> None:
        self.config = config
        self.artifact_store = artifact_store
        self.task_stream = task_stream
        self.agent = agent
        self.evaluator = evaluator
        self.env_factory = env_factory
        self.project_root = (project_root or Path.cwd()).resolve()

    def _state_path(self, run_dir: Path) -> Path:
        return run_dir / STATE_FILENAME

    def _write_event(
        self,
        *,
        run_dir: Path,
        sequence: int,
        event: str,
        payload: dict[str, object],
    ) -> None:
        record = EventRecord(
            schema_version=SCHEMA_VERSION,
            sequence=sequence,
            event=event,
            timestamp_utc=utc_now_iso(),
            payload=payload,
        )
        self.artifact_store.append_jsonl(run_dir, "events.jsonl", record.to_dict())

    def _write_state(
        self,
        *,
        run_dir: Path,
        tasks_seen: int,
        sequence: int,
        completed: bool,
        stream_state: dict[str, Any] | None,
    ) -> None:
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "tasks_seen": tasks_seen,
            "sequence": sequence,
            "completed": completed,
        }
        if stream_state is not None:
            payload["stream_state"] = stream_state
        self._state_path(run_dir).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _read_state(self, run_dir: Path) -> dict[str, Any]:
        state_path = self._state_path(run_dir)
        if not state_path.exists():
            raise OrchestratorError(f"Cannot resume run: missing state file at {state_path}.")
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise OrchestratorError(f"Invalid state payload in {state_path}.")
        return payload

    def _stream_get_state(self) -> dict[str, Any] | None:
        getter = getattr(self.task_stream, "get_state", None)
        if callable(getter):
            state = getter()
            if state is None:
                return None
            if not isinstance(state, dict):
                raise OrchestratorError("Task stream get_state() must return dict | None.")
            return state
        return None

    def _stream_set_state(self, state: Mapping[str, Any] | None) -> None:
        setter = getattr(self.task_stream, "set_state", None)
        if state is None:
            return
        if callable(setter):
            setter(state)
            return
        raise OrchestratorError(
            "Resume requested with stream_state, but task stream does not implement set_state()."
        )

    def run(  # noqa: PLR0912,PLR0913,PLR0915
        self,
        *,
        dry_run: bool = True,
        max_tasks: int | None = None,
        run_dir: Path | None = None,
        resume: bool = False,
        interrupt_after_tasks: int | None = None,
        fail_on_task_index: int | None = None,
    ) -> RunSummary:
        if not dry_run and (
            self.agent is None or self.evaluator is None or self.env_factory is None
        ):
            raise OrchestratorError(
                "Full run requested, but agent/evaluator/env_factory are not provided."
            )
        if max_tasks is not None and max_tasks <= 0:
            raise OrchestratorError(f"max_tasks must be positive when set, got {max_tasks}.")
        if interrupt_after_tasks is not None and interrupt_after_tasks <= 0:
            raise OrchestratorError(
                f"interrupt_after_tasks must be positive when set, got {interrupt_after_tasks}."
            )
        if fail_on_task_index is not None and fail_on_task_index < 0:
            raise OrchestratorError(
                f"fail_on_task_index must be non-negative when set, got {fail_on_task_index}."
            )
        if resume and run_dir is None:
            raise OrchestratorError("resume=True requires an existing run_dir.")

        deterministic_enabled = resolve_deterministic_mode(
            self.config.deterministic_mode,
            supports_determinism=False,
        )

        if resume:
            assert run_dir is not None
            run_dir = run_dir.resolve()
            state = self._read_state(run_dir)
            run_id = run_dir.name
            tasks_seen = int(state.get("tasks_seen", 0))
            sequence = int(state.get("sequence", 0))
            completed = bool(state.get("completed", False))
            if completed:
                raise OrchestratorError(
                    f"Cannot resume run '{run_id}': state already marked completed."
                )
            self.task_stream.reset()
            stream_state = state.get("stream_state")
            if stream_state is not None and not isinstance(stream_state, Mapping):
                raise OrchestratorError("state.stream_state must be a mapping when present.")
            self._stream_set_state(stream_state)
        else:
            if run_dir is None:
                run_dir = self.artifact_store.create_run_dir(self.config.run_name, self.config.seed)
            else:
                run_dir = run_dir.resolve()
                run_dir.mkdir(parents=True, exist_ok=False)
            run_id = run_dir.name
            resolved_config = self.config.to_dict()
            config_sha = hash_payload(resolved_config)
            manifest = build_manifest(
                config=self.config,
                run_id=run_id,
                config_sha256=config_sha,
                repo_root=self.project_root,
            )
            self.artifact_store.write_json(run_dir, "manifest.json", manifest.to_dict())
            self.artifact_store.write_json(run_dir, "resolved_config.json", resolved_config)
            self.task_stream.reset()
            tasks_seen = 0
            sequence = 0
            completed = False

        logger = create_logger("crlbench.run", run_id=run_id)
        logger.info("run started")

        def emit(event: str, payload: dict[str, object]) -> int:
            nonlocal sequence
            self._write_event(run_dir=run_dir, sequence=sequence, event=event, payload=payload)
            sequence += 1
            return sequence

        if resume:
            emit(
                "run_resumed",
                {
                    "run_id": run_id,
                    "tasks_seen": tasks_seen,
                    "dry_run": dry_run,
                    "deterministic_mode": self.config.deterministic_mode,
                    "deterministic_enabled": deterministic_enabled,
                },
            )
        else:
            emit(
                "run_start",
                {
                    "run_id": run_id,
                    "dry_run": dry_run,
                    "deterministic_mode": self.config.deterministic_mode,
                    "deterministic_enabled": deterministic_enabled,
                },
            )

        self._write_state(
            run_dir=run_dir,
            tasks_seen=tasks_seen,
            sequence=sequence,
            completed=False,
            stream_state=self._stream_get_state(),
        )

        while True:
            if max_tasks is not None and tasks_seen >= max_tasks:
                break

            task = self.task_stream.next_task()
            if task is None:
                break

            if fail_on_task_index is not None and tasks_seen == fail_on_task_index:
                emit(
                    "run_failed",
                    {
                        "tasks_seen": tasks_seen,
                        "fail_on_task_index": fail_on_task_index,
                    },
                )
                self._write_state(
                    run_dir=run_dir,
                    tasks_seen=tasks_seen,
                    sequence=sequence,
                    completed=False,
                    stream_state=self._stream_get_state(),
                )
                raise OrchestratorError(f"Injected failure at task index {fail_on_task_index}.")

            emit(
                "task_start",
                {
                    "task_id": task.task_id,
                    "env_family": task.env_family,
                    "env_option": task.env_option,
                    "task_index": tasks_seen,
                },
            )
            tasks_seen += 1

            self._write_state(
                run_dir=run_dir,
                tasks_seen=tasks_seen,
                sequence=sequence,
                completed=False,
                stream_state=self._stream_get_state(),
            )

            if interrupt_after_tasks is not None and tasks_seen >= interrupt_after_tasks:
                emit(
                    "run_interrupted",
                    {"tasks_seen": tasks_seen, "interrupt_after_tasks": interrupt_after_tasks},
                )
                self._write_state(
                    run_dir=run_dir,
                    tasks_seen=tasks_seen,
                    sequence=sequence,
                    completed=False,
                    stream_state=self._stream_get_state(),
                )
                logger.info("run interrupted", extra={"task": str(tasks_seen)})
                return RunSummary(
                    run_dir=run_dir,
                    run_id=run_id,
                    tasks_seen=tasks_seen,
                    deterministic_enabled=deterministic_enabled,
                    completed=False,
                )

        emit("run_complete", {"tasks_seen": tasks_seen, "dry_run": dry_run})
        self._write_state(
            run_dir=run_dir,
            tasks_seen=tasks_seen,
            sequence=sequence,
            completed=True,
            stream_state=self._stream_get_state(),
        )
        logger.info("run completed", extra={"task": str(tasks_seen)})
        return RunSummary(
            run_dir=run_dir,
            run_id=run_id,
            tasks_seen=tasks_seen,
            deterministic_enabled=deterministic_enabled,
            completed=True,
        )
