from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crlbench.metrics import StreamEvaluation, compute_stream_metrics
from crlbench.runtime.plots import generate_canonical_plots
from crlbench.runtime.reporting import (
    aggregate_run_metric_summaries,
    export_summary_csv,
    export_summary_latex,
    write_experiment_metrics_summary,
    write_run_metrics_summary,
)

from .experiment1 import (
    EXPERIMENT_ID,
    Experiment1Protocol,
    build_experiment1_protocol,
)

EXPERIMENT1_QUALITY_ROOTS: tuple[tuple[str, str, str], ...] = (
    ("toy", "dm_control", "vision_sequential_default"),
    ("toy", "procgen", "vision_sequential_alternative"),
    ("robotics", "metaworld", "manipulation_sequential_default"),
    ("robotics", "maniskill", "manipulation_sequential_alternative"),
)


def _simulate_stream_trace(protocol: Experiment1Protocol, seed: int) -> dict[str, Any]:
    stages: list[dict[str, Any]] = []
    seed_offset = float(seed) * 0.001
    for stage_idx, stage in enumerate(protocol.eval_plan):
        returns: dict[str, float] = {}
        for task_idx, task_id in enumerate(stage.eval_task_ids):
            base = 1.0 + (task_idx * 0.25)
            forgetting = float(stage_idx - task_idx) * 0.08
            value = max(0.0, base - forgetting + seed_offset)
            if task_id == stage.train_task_id:
                value += 0.1
            returns[task_id] = round(value, 6)
        stages.append({"stage_id": stage.stage_id, "task_returns": returns})
    return {"stages": stages}


def _run_metrics(
    *,
    protocol: Experiment1Protocol,
    seed: int,
    run_dir: Path,
) -> dict[str, Any]:
    trace_payload = _simulate_stream_trace(protocol, seed)
    trace_path = run_dir / "stream_trace.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        json.dumps(trace_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics = compute_stream_metrics(StreamEvaluation.from_mapping(trace_payload))
    metadata = {
        "experiment": protocol.experiment_id,
        "track": protocol.track,
        "env_family": protocol.env_family,
        "env_option": protocol.env_option,
        "seed": seed,
        "budget_tier": protocol.budget_tier,
    }
    run_id = f"{protocol.track}_{protocol.env_family}_{protocol.env_option}_s{seed}"
    write_run_metrics_summary(run_dir=run_dir, run_id=run_id, metrics=metrics, metadata=metadata)
    return metrics


def _required_primary_metrics(metrics: dict[str, Any], required: list[str]) -> bool:
    return all(key in metrics for key in required)


@dataclass(frozen=True)
class Experiment1QualityGateResult:
    output_dir: Path
    seed_count: int
    metric_parity_passed: bool
    reproducibility_passed: bool
    summary_json_path: Path
    summary_csv_path: Path
    summary_latex_path: Path


def run_experiment1_quality_gate(  # noqa: PLR0912
    *,
    output_dir: Path,
    seed_count: int = 5,
    budget_tier: str = "smoke",
) -> Experiment1QualityGateResult:
    if seed_count <= 0:
        raise ValueError(f"seed_count must be positive, got {seed_count}.")
    seeds = list(range(seed_count))
    summaries: list[dict[str, Any]] = []
    representative_metrics: dict[tuple[str, str], dict[str, Any]] = {}
    parity_keys: set[str] | None = None
    reproducibility_passed = True

    for track, env_family, env_option in EXPERIMENT1_QUALITY_ROOTS:
        protocol = build_experiment1_protocol(
            track=track,
            env_family=env_family,
            env_option=env_option,
            budget_tier=budget_tier,
        )
        required_metrics = protocol.reporting_template["primary_metrics"]
        if not isinstance(required_metrics, list):
            raise ValueError("Experiment 1 reporting template primary_metrics must be a list.")

        for seed in seeds:
            run_dir = output_dir / "runs" / track / env_family / env_option / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics = _run_metrics(protocol=protocol, seed=seed, run_dir=run_dir)
            if seed == seeds[0]:
                representative_metrics[(track, env_family)] = metrics
                if parity_keys is None:
                    parity_keys = set(metrics)
                else:
                    parity_keys &= set(metrics)
                plot_dir = output_dir / "plots" / track / env_family / env_option
                generate_canonical_plots(
                    run_summary={
                        "schema_version": "1.0.0",
                        "run_id": f"{track}_{env_family}_{env_option}_s{seed}",
                        "created_at_utc": "generated-by-quality-gate",
                        "metadata": {
                            "experiment": EXPERIMENT_ID,
                            "track": track,
                            "env_family": env_family,
                            "env_option": env_option,
                        },
                        "metrics": metrics,
                    },
                    output_dir=plot_dir,
                    render_images=False,
                )

            rerun_metrics = _run_metrics(
                protocol=protocol,
                seed=seed,
                run_dir=run_dir / "repro_check",
            )
            reproducibility_passed = reproducibility_passed and (metrics == rerun_metrics)
            if not _required_primary_metrics(metrics, required_metrics):
                reproducibility_passed = False

            summary_payload = json.loads(
                (run_dir / "run_metrics_summary.json").read_text(encoding="utf-8")
            )
            if isinstance(summary_payload, dict):
                summaries.append(summary_payload)

    toy_keys = set()
    robotics_keys = set()
    for (track, _env_family), metrics in representative_metrics.items():
        if track == "toy":
            toy_keys |= set(metrics)
        if track == "robotics":
            robotics_keys |= set(metrics)
    metric_parity_passed = toy_keys == robotics_keys and bool(toy_keys)
    if parity_keys is not None:
        metric_parity_passed = metric_parity_passed and bool(parity_keys)

    summary = aggregate_run_metric_summaries(summaries)
    summary_dir = output_dir / "summary"
    summary_json_path = write_experiment_metrics_summary(
        output_path=summary_dir / "experiment1_metrics_summary.json",
        summary=summary,
    )
    summary_csv_path = export_summary_csv(
        summary,
        summary_dir / "experiment1_metrics_summary.csv",
    )
    summary_latex_path = export_summary_latex(
        summary,
        summary_dir / "experiment1_metrics_summary.tex",
    )

    return Experiment1QualityGateResult(
        output_dir=output_dir,
        seed_count=seed_count,
        metric_parity_passed=metric_parity_passed,
        reproducibility_passed=reproducibility_passed,
        summary_json_path=summary_json_path,
        summary_csv_path=summary_csv_path,
        summary_latex_path=summary_latex_path,
    )
