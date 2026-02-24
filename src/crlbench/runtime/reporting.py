from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from crlbench.metrics import aggregate_scalar_values

from .schemas import (
    SCHEMA_VERSION,
    ExperimentMetricsSummaryRecord,
    RunMetricsSummaryRecord,
    utc_now_iso,
)


def write_run_metrics_summary(
    *,
    run_dir: Path,
    run_id: str,
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    payload = RunMetricsSummaryRecord(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        created_at_utc=utc_now_iso(),
        metadata=dict(metadata or {}),
        metrics=dict(metrics),
    )
    path = run_dir / "run_metrics_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def load_run_metrics_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid run summary payload in {path}.")
    return payload


def aggregate_run_metric_summaries(
    summaries: Sequence[Mapping[str, Any]],
    *,
    grouping_keys: Sequence[str] = ("experiment", "track", "env_family", "env_option"),
) -> dict[str, Any]:
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for summary in summaries:
        metadata = summary.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = {}
        group_key: tuple[str, ...] = tuple(
            str(metadata.get(key, "unknown")) for key in grouping_keys
        )
        grouped[group_key].append(summary)

    groups_out: list[dict[str, Any]] = []
    for group_key, group_summaries in grouped.items():
        metric_values: dict[str, list[float]] = defaultdict(list)
        run_ids: list[str] = []
        for summary in group_summaries:
            run_id = summary.get("run_id")
            if isinstance(run_id, str):
                run_ids.append(run_id)
            metrics = summary.get("metrics")
            if not isinstance(metrics, Mapping):
                continue
            for metric_name, value in metrics.items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    metric_values[str(metric_name)].append(float(value))
        aggregated_metrics: dict[str, dict[str, float]] = {}
        for metric_name, values in metric_values.items():
            if values:
                aggregated_metrics[metric_name] = aggregate_scalar_values(values)

        groups_out.append(
            {
                "group": dict(zip(grouping_keys, group_key, strict=True)),
                "num_runs": len(group_summaries),
                "run_ids": run_ids,
                "metrics": aggregated_metrics,
            }
        )

    record = ExperimentMetricsSummaryRecord(
        schema_version=SCHEMA_VERSION,
        created_at_utc=utc_now_iso(),
        grouping_keys=list(grouping_keys),
        groups=groups_out,
    )
    return record.to_dict()


def write_experiment_metrics_summary(
    *,
    output_path: Path,
    summary: Mapping[str, Any],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def export_summary_csv(summary: Mapping[str, Any], output_path: Path) -> Path:
    groups = summary.get("groups")
    if not isinstance(groups, list):
        raise ValueError("summary payload must contain a list 'groups'.")

    rows: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, Mapping):
            continue
        group_key = group.get("group")
        metrics = group.get("metrics")
        if not isinstance(group_key, Mapping) or not isinstance(metrics, Mapping):
            continue
        base = {str(k): v for k, v in group_key.items()}
        base["num_runs"] = group.get("num_runs", 0)
        for metric_name, stats in metrics.items():
            if not isinstance(stats, Mapping):
                continue
            row = dict(base)
            row["metric"] = metric_name
            row["mean"] = stats.get("mean")
            row["std"] = stats.get("std")
            row["stderr"] = stats.get("stderr")
            row["ci95"] = stats.get("ci95")
            row["n"] = stats.get("n")
            rows.append(row)

    fieldnames: list[str] = [
        "experiment",
        "track",
        "env_family",
        "env_option",
        "num_runs",
        "metric",
        "mean",
        "std",
        "stderr",
        "ci95",
        "n",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path


def export_summary_latex(summary: Mapping[str, Any], output_path: Path) -> Path:
    groups = summary.get("groups")
    if not isinstance(groups, list):
        raise ValueError("summary payload must contain a list 'groups'.")

    lines = [
        "\\begin{tabular}{lllllrr}",
        "\\hline",
        "Experiment & Track & EnvFamily & EnvOption & Metric & Mean & CI95 \\\\",
        "\\hline",
    ]
    for group in groups:
        if not isinstance(group, Mapping):
            continue
        group_key = group.get("group")
        metrics = group.get("metrics")
        if not isinstance(group_key, Mapping) or not isinstance(metrics, Mapping):
            continue
        for metric_name, stats in metrics.items():
            if not isinstance(stats, Mapping):
                continue
            lines.append(
                "{} & {} & {} & {} & {} & {:.6f} & {:.6f} \\\\".format(
                    str(group_key.get("experiment", "unknown")),
                    str(group_key.get("track", "unknown")),
                    str(group_key.get("env_family", "unknown")),
                    str(group_key.get("env_option", "unknown")),
                    str(metric_name),
                    float(stats.get("mean", 0.0)),
                    float(stats.get("ci95", 0.0)),
                )
            )
    lines.extend(["\\hline", "\\end{tabular}"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
