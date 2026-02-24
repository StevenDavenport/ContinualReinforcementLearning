from __future__ import annotations

import json
from pathlib import Path

from crlbench.runtime.reporting import (
    aggregate_run_metric_summaries,
    export_summary_csv,
    export_summary_latex,
    load_run_metrics_summary,
    write_experiment_metrics_summary,
    write_run_metrics_summary,
)


def _write_summary(  # noqa: PLR0913
    run_dir: Path,
    run_id: str,
    score: float,
    *,
    experiment: str = "exp1",
    track: str = "toy",
    env_family: str = "dm_control",
    env_option: str = "vision",
) -> Path:
    return write_run_metrics_summary(
        run_dir=run_dir,
        run_id=run_id,
        metrics={"final_stage_average_return": score, "average_forgetting": 0.25},
        metadata={
            "experiment": experiment,
            "track": track,
            "env_family": env_family,
            "env_option": env_option,
        },
    )


def test_run_summary_write_and_load(tmp_path: Path) -> None:
    path = _write_summary(tmp_path / "r1", "run_1", 12.0)
    loaded = load_run_metrics_summary(path)
    assert loaded["run_id"] == "run_1"
    assert loaded["metrics"]["final_stage_average_return"] == 12.0


def test_aggregate_and_export(tmp_path: Path) -> None:
    s1 = load_run_metrics_summary(_write_summary(tmp_path / "r1", "run_1", 10.0))
    s2 = load_run_metrics_summary(_write_summary(tmp_path / "r2", "run_2", 14.0))
    aggregated = aggregate_run_metric_summaries([s1, s2])

    out_json = tmp_path / "experiment_summary.json"
    out_csv = tmp_path / "experiment_summary.csv"
    out_tex = tmp_path / "experiment_summary.tex"

    write_experiment_metrics_summary(output_path=out_json, summary=aggregated)
    export_summary_csv(aggregated, out_csv)
    export_summary_latex(aggregated, out_tex)

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["groups"][0]["num_runs"] == 2
    assert out_csv.exists()
    assert out_tex.exists()
