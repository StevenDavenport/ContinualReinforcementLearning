from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crlbench.agents import instantiate_agent
from crlbench.config.schema import RunConfig
from crlbench.core.plugins import create_environment
from crlbench.core.types import TaskSpec, Transition
from crlbench.errors import OrchestrationError
from crlbench.metrics import StreamEvaluation, compute_stream_metrics
from crlbench.runtime.artifacts import ArtifactStore
from crlbench.runtime.manifest import build_manifest, hash_payload
from crlbench.runtime.plots import generate_canonical_plots
from crlbench.runtime.reporting import (
    aggregate_run_metric_summaries,
    export_summary_csv,
    export_summary_latex,
    load_run_metrics_summary,
    write_experiment_metrics_summary,
    write_run_metrics_summary,
)
from crlbench.runtime.schemas import SCHEMA_VERSION, EventRecord, utc_now_iso

from .experiment1 import (
    EXPERIMENT_ID,
    Experiment1Protocol,
    build_experiment1_protocol,
    get_experiment1_action_space_n,
    list_experiment1_roots,
    register_experiment1_plugins,
)


@dataclass(frozen=True)
class SingleRunResult:
    run_dir: Path
    run_id: str
    metrics_path: Path
    trace_path: Path


@dataclass(frozen=True)
class ExperimentExecutionResult:
    run_results: tuple[SingleRunResult, ...]
    summary_json_path: Path | None
    summary_csv_path: Path | None
    summary_latex_path: Path | None


@dataclass(frozen=True)
class MatrixExecutionResult:
    root_results: dict[str, ExperimentExecutionResult]


def _emit_event(
    *,
    store: ArtifactStore,
    run_dir: Path,
    sequence: int,
    event: str,
    payload: dict[str, Any],
) -> int:
    record = EventRecord(
        schema_version=SCHEMA_VERSION,
        sequence=sequence,
        event=event,
        timestamp_utc=utc_now_iso(),
        payload=payload,
    )
    store.append_jsonl(run_dir, "events.jsonl", record.to_dict())
    return sequence + 1


def _write_state(
    *,
    run_dir: Path,
    tasks_seen: int,
    sequence: int,
    completed: bool,
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "tasks_seen": tasks_seen,
        "sequence": sequence,
        "completed": completed,
    }
    (run_dir / "state.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_run_config(
    *,
    protocol: Experiment1Protocol,
    run_name: str,
    output_dir: Path,
    seed: int,
    num_seeds: int,
) -> RunConfig:
    return RunConfig.from_mapping(
        {
            "run_name": run_name,
            "experiment": protocol.experiment_id,
            "track": protocol.track,
            "env_family": protocol.env_family,
            "env_option": protocol.env_option,
            "seed": seed,
            "num_seeds": num_seeds,
            "observation_mode": "image_proprio" if protocol.track == "robotics" else "image",
            "deterministic_mode": "auto",
            "seed_namespace": "global",
            "budget": protocol.budget.to_dict(),
            "output_dir": str(output_dir),
            "tags": [
                "run_experiment",
                protocol.track,
                protocol.env_family,
                protocol.env_option,
            ],
        }
    )


def _train_stage(
    *,
    agent: Any,
    task: TaskSpec,
    seed: int,
    steps: int,
) -> None:
    env = create_environment(
        task.env_family,
        env_option=task.env_option,
        task_id=task.task_id,
    )
    observation = env.reset(seed=seed)
    reward_window: deque[float] = deque(maxlen=200)
    progress_every = max(1, min(5_000, steps // 10))
    for index in range(steps):
        action = agent.act(observation, deterministic=False)
        step = env.step(action)
        transition = Transition(
            observation=observation,
            action=action,
            reward=float(step.reward),
            next_observation=step.observation,
            terminated=step.terminated,
            truncated=step.truncated,
            info=step.info,
        )
        update_metrics = agent.update([transition])
        reward_window.append(float(step.reward))
        step_count = index + 1
        if step_count % progress_every == 0 or step_count == steps:
            mean_reward = sum(reward_window) / max(1, len(reward_window))
            policy_loss = update_metrics.get("policy_loss", 0.0)
            value_loss = update_metrics.get("value_loss", 0.0)
            num_updates = update_metrics.get("num_updates", 0.0)
            print(
                "[train] "
                f"task={task.task_id} step={step_count}/{steps} "
                f"reward_ma200={mean_reward:.4f} "
                f"policy_loss={float(policy_loss):.4f} "
                f"value_loss={float(value_loss):.4f} "
                f"updates={float(num_updates):.1f}",
                flush=True,
            )
        observation = step.observation
        if step.terminated or step.truncated:
            observation = env.reset(seed=seed + index + 1)
    env.close()


def _evaluate_task(
    *,
    agent: Any,
    task: TaskSpec,
    seed: int,
    eval_episodes: int,
    eval_horizon: int,
) -> float:
    returns: list[float] = []
    env = create_environment(
        task.env_family,
        env_option=task.env_option,
        task_id=task.task_id,
    )
    for episode in range(eval_episodes):
        observation = env.reset(seed=seed + episode)
        episode_return = 0.0
        for _ in range(eval_horizon):
            action = agent.act(observation, deterministic=True)
            step = env.step(action)
            episode_return += float(step.reward)
            observation = step.observation
            if step.terminated or step.truncated:
                break
        returns.append(episode_return)
    env.close()
    return sum(returns) / len(returns) if returns else 0.0


def _run_single_experiment1(  # noqa: PLR0913
    *,
    protocol: Experiment1Protocol,
    run_name: str,
    output_dir: Path,
    seed: int,
    num_seeds: int,
    agent_name: str | None,
    agent_path: Path | None,
    agents_dir: Path,
    eval_horizon: int,
    eval_episodes_override: int | None,
    train_steps_cap: int | None,
    eval_episodes_cap: int | None,
    dm_control_backend: str,
    agent_config: Mapping[str, Any],
) -> tuple[SingleRunResult, Path]:
    config = _build_run_config(
        protocol=protocol,
        run_name=run_name,
        output_dir=output_dir,
        seed=seed,
        num_seeds=num_seeds,
    )
    store = ArtifactStore(output_dir)
    run_dir = store.create_run_dir(run_name, seed=seed)
    run_id = run_dir.name
    resolved_config = config.to_dict()
    config_sha = hash_payload(resolved_config)
    manifest = build_manifest(
        config=config,
        run_id=run_id,
        config_sha256=config_sha,
        repo_root=Path.cwd(),
    )
    store.write_json(run_dir, "manifest.json", manifest.to_dict())
    store.write_json(run_dir, "resolved_config.json", resolved_config)

    sequence = 0
    tasks_seen = 0
    sequence = _emit_event(
        store=store,
        run_dir=run_dir,
        sequence=sequence,
        event="run_start",
        payload={
            "run_id": run_id,
            "agent": agent_name if agent_name is not None else str(agent_path),
            "budget_tier": protocol.budget_tier,
            "seed": seed,
            "dm_control_backend": dm_control_backend,
        },
    )
    _write_state(run_dir=run_dir, tasks_seen=tasks_seen, sequence=sequence, completed=False)

    agent, descriptor = instantiate_agent(
        agent_name=agent_name,
        agent_path=agent_path,
        agents_dir=agents_dir,
        config={
            **dict(agent_config),
            "seed": seed,
            "action_space_n": get_experiment1_action_space_n(
                env_family=protocol.env_family,
                env_option=protocol.env_option,
            ),
        },
    )
    agent.reset()

    tasks_by_id = {task.task_id: task for task in protocol.tasks}
    stage_payloads: list[dict[str, Any]] = []
    train_steps = config.budget.train_steps // max(1, len(protocol.tasks))
    if train_steps_cap is not None:
        train_steps = min(train_steps, train_steps_cap)
    eval_episodes = config.budget.eval_episodes
    if eval_episodes_override is not None:
        eval_episodes = eval_episodes_override
    if eval_episodes_cap is not None:
        eval_episodes = min(eval_episodes, eval_episodes_cap)

    for stage in protocol.eval_plan:
        task = tasks_by_id[stage.train_task_id]
        print(
            f"[train] stage_start stage={stage.stage_id} "
            f"task={task.task_id} steps={max(1, train_steps)}",
            flush=True,
        )
        sequence = _emit_event(
            store=store,
            run_dir=run_dir,
            sequence=sequence,
            event="task_start",
            payload={
                "task_id": task.task_id,
                "env_family": task.env_family,
                "env_option": task.env_option,
                "task_index": tasks_seen,
                "agent": descriptor.name,
            },
        )
        _train_stage(agent=agent, task=task, seed=seed, steps=max(1, train_steps))
        task_returns: dict[str, float] = {}
        for eval_task_id in stage.eval_task_ids:
            eval_task = tasks_by_id[eval_task_id]
            task_returns[eval_task_id] = _evaluate_task(
                agent=agent,
                task=eval_task,
                seed=seed,
                eval_episodes=max(1, eval_episodes),
                eval_horizon=max(1, eval_horizon),
            )
        eval_mean = sum(task_returns.values()) / max(1, len(task_returns))
        print(
            f"[eval] stage={stage.stage_id} seen_tasks={len(task_returns)} "
            f"avg_return={eval_mean:.4f}",
            flush=True,
        )
        stage_payloads.append({"stage_id": stage.stage_id, "task_returns": task_returns})
        tasks_seen += 1
        sequence = _emit_event(
            store=store,
            run_dir=run_dir,
            sequence=sequence,
            event="task_complete",
            payload={"stage_id": stage.stage_id, "tasks_seen": tasks_seen},
        )
        _write_state(run_dir=run_dir, tasks_seen=tasks_seen, sequence=sequence, completed=False)

    trace_payload = {"stages": stage_payloads}
    trace_path = run_dir / "stream_trace.json"
    trace_path.write_text(
        json.dumps(trace_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics = compute_stream_metrics(StreamEvaluation.from_mapping(trace_payload))
    metrics_path = write_run_metrics_summary(
        run_dir=run_dir,
        run_id=run_id,
        metrics=metrics,
        metadata={
            "experiment": protocol.experiment_id,
            "track": protocol.track,
            "env_family": protocol.env_family,
            "env_option": protocol.env_option,
            "agent": descriptor.name,
            "budget_tier": protocol.budget_tier,
            "seed": seed,
            "dm_control_backend": dm_control_backend,
        },
    )
    generate_canonical_plots(
        run_summary=load_run_metrics_summary(metrics_path),
        output_dir=run_dir / "plots",
        render_images=False,
    )

    agent.save(run_dir / "checkpoints" / "agent_state.json")
    sequence = _emit_event(
        store=store,
        run_dir=run_dir,
        sequence=sequence,
        event="run_complete",
        payload={"tasks_seen": tasks_seen, "agent": descriptor.name},
    )
    _write_state(run_dir=run_dir, tasks_seen=tasks_seen, sequence=sequence, completed=True)
    return (
        SingleRunResult(
            run_dir=run_dir,
            run_id=run_id,
            metrics_path=metrics_path,
            trace_path=trace_path,
        ),
        metrics_path,
    )


def run_experiment(  # noqa: PLR0913
    *,
    experiment: str,
    track: str,
    env_family: str,
    env_option: str,
    agent_name: str | None,
    agent_path: Path | None,
    agents_dir: Path,
    budget_tier: str,
    output_dir: Path,
    run_name: str | None,
    seed: int,
    num_seeds: int,
    eval_horizon: int = 32,
    eval_episodes_override: int | None = None,
    train_steps_cap: int | None = None,
    eval_episodes_cap: int | None = None,
    dm_control_backend: str = "auto",
    agent_config: Mapping[str, Any] | None = None,
) -> ExperimentExecutionResult:
    if experiment != EXPERIMENT_ID:
        raise OrchestrationError(
            f"Only '{EXPERIMENT_ID}' is currently supported by run-experiment."
        )
    if num_seeds <= 0:
        raise OrchestrationError(f"num_seeds must be positive, got {num_seeds}.")
    if seed < 0:
        raise OrchestrationError(f"seed must be non-negative, got {seed}.")
    if eval_episodes_override is not None and eval_episodes_override <= 0:
        raise OrchestrationError(
            f"eval_episodes_override must be positive, got {eval_episodes_override}."
        )
    register_experiment1_plugins(replace=True, dm_control_backend=dm_control_backend)
    protocol = build_experiment1_protocol(
        track=track,
        env_family=env_family,
        env_option=env_option,
        budget_tier=budget_tier,
    )

    resolved_run_name = run_name or f"{experiment}_{track}_{env_family}_{env_option}"
    run_results: list[SingleRunResult] = []
    metric_paths: list[Path] = []
    for offset in range(num_seeds):
        single_result, metric_path = _run_single_experiment1(
            protocol=protocol,
            run_name=resolved_run_name,
            output_dir=output_dir,
            seed=seed + offset,
            num_seeds=num_seeds,
            agent_name=agent_name,
            agent_path=agent_path,
            agents_dir=agents_dir,
            eval_horizon=eval_horizon,
            eval_episodes_override=eval_episodes_override,
            train_steps_cap=train_steps_cap,
            eval_episodes_cap=eval_episodes_cap,
            dm_control_backend=dm_control_backend,
            agent_config=dict(agent_config or {}),
        )
        run_results.append(single_result)
        metric_paths.append(metric_path)

    summary_json_path: Path | None = None
    summary_csv_path: Path | None = None
    summary_latex_path: Path | None = None
    if metric_paths:
        summaries = [load_run_metrics_summary(path) for path in metric_paths]
        aggregated = aggregate_run_metric_summaries(summaries)
        summary_dir = output_dir / "summaries"
        summary_json_path = write_experiment_metrics_summary(
            output_path=summary_dir / f"{resolved_run_name}_metrics_summary.json",
            summary=aggregated,
        )
        summary_csv_path = export_summary_csv(
            aggregated,
            summary_dir / f"{resolved_run_name}_metrics_summary.csv",
        )
        summary_latex_path = export_summary_latex(
            aggregated,
            summary_dir / f"{resolved_run_name}_metrics_summary.tex",
        )

    return ExperimentExecutionResult(
        run_results=tuple(run_results),
        summary_json_path=summary_json_path,
        summary_csv_path=summary_csv_path,
        summary_latex_path=summary_latex_path,
    )


def run_experiment1_matrix(  # noqa: PLR0913
    *,
    agent_name: str | None,
    agent_path: Path | None,
    agents_dir: Path,
    budget_tier: str,
    output_dir: Path,
    run_name: str | None,
    seed: int,
    num_seeds: int,
    eval_horizon: int = 32,
    eval_episodes_override: int | None = None,
    train_steps_cap: int | None = None,
    eval_episodes_cap: int | None = None,
    dm_control_backend: str = "auto",
    agent_config: Mapping[str, Any] | None = None,
) -> MatrixExecutionResult:
    root_results: dict[str, ExperimentExecutionResult] = {}
    for track, env_family, env_option in list_experiment1_roots():
        key = f"{track}/{env_family}/{env_option}"
        root_output_dir = output_dir / track / env_family / env_option
        root_run_name = run_name or f"{EXPERIMENT_ID}_{track}_{env_family}_{env_option}"
        root_results[key] = run_experiment(
            experiment=EXPERIMENT_ID,
            track=track,
            env_family=env_family,
            env_option=env_option,
            agent_name=agent_name,
            agent_path=agent_path,
            agents_dir=agents_dir,
            budget_tier=budget_tier,
            output_dir=root_output_dir,
            run_name=root_run_name,
            seed=seed,
            num_seeds=num_seeds,
            eval_horizon=eval_horizon,
            eval_episodes_override=eval_episodes_override,
            train_steps_cap=train_steps_cap,
            eval_episodes_cap=eval_episodes_cap,
            dm_control_backend=dm_control_backend,
            agent_config=agent_config,
        )
    return MatrixExecutionResult(root_results=root_results)
