from pathlib import Path

from pytest import CaptureFixture

from crlbench.cli import main


def test_validate_config_command(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """
{
  "run_name": "run",
  "experiment": "experiment_1_forgetting",
  "track": "toy",
  "env_family": "dm_control",
  "env_option": "vision_sequential_default",
  "seed": 0,
  "num_seeds": 5,
  "observation_mode": "image",
  "deterministic_mode": "auto",
  "seed_namespace": "global",
  "budget": {
    "train_steps": 10000,
    "eval_interval_steps": 1000,
    "eval_episodes": 10
  }
}
""".strip(),
        encoding="utf-8",
    )
    rc = main(["validate-config", "--config", str(config_path)])
    assert rc == 0


def test_resolve_config_command(tmp_path: Path) -> None:
    base_path = tmp_path / "base.json"
    child_path = tmp_path / "child.json"
    out_path = tmp_path / "out.json"

    base_path.write_text(
        """
{
  "run_name": "base",
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
    "train_steps": 10000,
    "eval_interval_steps": 1000,
    "eval_episodes": 10
  }
}
""".strip(),
        encoding="utf-8",
    )
    child_path.write_text(
        """
{
  "extends": "base.json",
  "run_name": "child",
  "budget": {
    "train_steps": 12000,
    "eval_interval_steps": 1000,
    "eval_episodes": 10
  }
}
""".strip(),
        encoding="utf-8",
    )
    rc = main(["resolve-config", "--config", str(child_path), "--out", str(out_path)])
    assert rc == 0
    assert out_path.exists()


def test_print_config_command(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    config_path = Path(tmp_path) / "config.json"
    config_path.write_text(
        """
{
  "run_name": "run",
  "experiment": "experiment_1_forgetting",
  "track": "toy",
  "env_family": "dm_control",
  "env_option": "vision_sequential_default",
  "seed": 0,
  "num_seeds": 5,
  "observation_mode": "image",
  "deterministic_mode": "auto",
  "seed_namespace": "global",
  "budget": {
    "train_steps": 10000,
    "eval_interval_steps": 1000,
    "eval_episodes": 10
  }
}
""".strip(),
        encoding="utf-8",
    )

    rc = main(["print-config", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert rc == 0
    assert '"run_name": "run"' in captured.out


def test_compose_config_command_with_override(tmp_path: Path) -> None:
    base_path = tmp_path / "base.json"
    layer_path = tmp_path / "layer.json"
    out_path = tmp_path / "composed.json"

    base_path.write_text(
        """
{
  "run_name": "base",
  "budget": {
    "train_steps": 10000,
    "eval_interval_steps": 1000,
    "eval_episodes": 10
  }
}
""".strip(),
        encoding="utf-8",
    )
    layer_path.write_text(
        """
{
  "run_name": "layer",
  "budget": {
    "train_steps": 12000,
    "eval_interval_steps": 1000,
    "eval_episodes": 10
  }
}
""".strip(),
        encoding="utf-8",
    )

    rc = main(
        [
            "compose-config",
            "--base",
            str(base_path),
            "--layer",
            str(layer_path),
            "--set",
            "budget.train_steps=15000",
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    payload = out_path.read_text(encoding="utf-8")
    assert '"train_steps": 15000' in payload


def test_repro_smoke_command(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    out_dir = tmp_path / "artifacts"
    config_path.write_text(
        """
{
  "run_name": "smoke_cli",
  "experiment": "experiment_1_forgetting",
  "track": "toy",
  "env_family": "dm_control",
  "env_option": "vision_sequential_default",
  "seed": 0,
  "num_seeds": 5,
  "observation_mode": "image",
  "deterministic_mode": "auto",
  "seed_namespace": "global",
  "budget": {
    "train_steps": 10000,
    "eval_interval_steps": 1000,
    "eval_episodes": 10
  }
}
""".strip(),
        encoding="utf-8",
    )
    rc = main(
        [
            "repro-smoke",
            "--config",
            str(config_path),
            "--out-dir",
            str(out_dir),
            "--max-tasks",
            "2",
        ]
    )
    assert rc == 0


def test_compute_stream_metrics_command(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.json"
    out_path = tmp_path / "run_metrics_summary.json"
    trace_path.write_text(
        """
{
  "stages": [
    {"stage_id": "s1", "task_returns": {"A": 1.0}},
    {"stage_id": "s2", "task_returns": {"A": 0.5, "B": 2.0}}
  ]
}
""".strip(),
        encoding="utf-8",
    )
    rc = main(
        [
            "compute-stream-metrics",
            "--trace",
            str(trace_path),
            "--metadata",
            "run_id=test_run_1",
            "--metadata",
            "experiment=exp1",
            "--metadata",
            "track=toy",
            "--metadata",
            "env_family=dm_control",
            "--metadata",
            "env_option=vision",
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    payload = out_path.read_text(encoding="utf-8")
    assert '"run_id": "test_run_1"' in payload


def test_aggregate_metric_summaries_command(tmp_path: Path) -> None:
    s1 = tmp_path / "s1.json"
    s2 = tmp_path / "s2.json"
    out = tmp_path / "aggregate.json"
    out_csv = tmp_path / "aggregate.csv"
    out_tex = tmp_path / "aggregate.tex"

    s1.write_text(
        """
{
  "schema_version": "1.0.0",
  "run_id": "run_1",
  "created_at_utc": "generated-by-test",
  "metadata": {
    "experiment": "exp1",
    "track": "toy",
    "env_family": "dm_control",
    "env_option": "vision"
  },
  "metrics": {
    "final_stage_average_return": 1.0
  }
}
""".strip(),
        encoding="utf-8",
    )
    s2.write_text(
        """
{
  "schema_version": "1.0.0",
  "run_id": "run_2",
  "created_at_utc": "generated-by-test",
  "metadata": {
    "experiment": "exp1",
    "track": "toy",
    "env_family": "dm_control",
    "env_option": "vision"
  },
  "metrics": {
    "final_stage_average_return": 2.0
  }
}
""".strip(),
        encoding="utf-8",
    )

    rc = main(
        [
            "aggregate-metric-summaries",
            "--summary",
            str(s1),
            "--summary",
            str(s2),
            "--out",
            str(out),
            "--csv",
            str(out_csv),
            "--latex",
            str(out_tex),
        ]
    )
    assert rc == 0
    assert out.exists()
    assert out_csv.exists()
    assert out_tex.exists()


def test_generate_canonical_plots_command(tmp_path: Path) -> None:
    summary_path = tmp_path / "run_metrics_summary.json"
    summary_path.write_text(
        """
{
  "schema_version": "1.0.0",
  "run_id": "run_plot",
  "created_at_utc": "generated-by-test",
  "metadata": {
    "experiment": "exp1",
    "track": "toy",
    "env_family": "dm_control",
    "env_option": "vision"
  },
  "metrics": {
    "average_return_by_stage": {"s1": 1.0, "s2": 2.0},
    "forgetting_by_task": {"A": 0.1},
    "retention_by_task": {"A": 0.9},
    "forward_transfer_by_task": {"A": 1.1},
    "backward_transfer_by_task": {"A": 0.1},
    "adaptation_curve": {"0": 0.1, "1": 0.2}
  }
}
""".strip(),
        encoding="utf-8",
    )
    out_dir = tmp_path / "plots"
    rc = main(
        [
            "generate-canonical-plots",
            "--summary",
            str(summary_path),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    assert len(list(out_dir.glob("*.plot.json"))) == 5


def test_export_publication_pack_command(tmp_path: Path) -> None:
    run_dir = tmp_path / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text('{"run_id":"run1"}\n', encoding="utf-8")
    (run_dir / "resolved_config.json").write_text('{"run_name":"run1"}\n', encoding="utf-8")
    (run_dir / "run_metrics_summary.json").write_text(
        """
{
  "schema_version": "1.0.0",
  "run_id": "run1",
  "created_at_utc": "generated-by-test",
  "metadata": {
    "experiment": "exp1",
    "track": "toy",
    "env_family": "dm_control",
    "env_option": "vision"
  },
  "metrics": {
    "final_stage_average_return": 1.0,
    "average_return_by_stage": {"s1": 1.0, "s2": 1.2},
    "forgetting_by_task": {"A": 0.1},
    "retention_by_task": {"A": 0.9},
    "forward_transfer_by_task": {"A": 1.1},
    "backward_transfer_by_task": {"A": 0.1},
    "adaptation_curve": {"0": 0.1, "1": 0.2}
  }
}
""".strip(),
        encoding="utf-8",
    )
    out_dir = tmp_path / "pack"
    rc = main(
        [
            "export-publication-pack",
            "--run-dir",
            str(run_dir),
            "--out-dir",
            str(out_dir),
            "--method",
            "agent=dreamer",
        ]
    )
    assert rc == 0
    assert (out_dir / "summaries" / "experiment_metrics_summary.json").exists()
    assert (out_dir / "tables" / "experiment_metrics_summary.csv").exists()


def test_validate_artifacts_command_passes_for_smoke_artifacts(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    out_dir = tmp_path / "artifacts"
    config_path.write_text(
        """
{
  "run_name": "smoke_validate_cli",
  "experiment": "experiment_1_forgetting",
  "track": "toy",
  "env_family": "dm_control",
  "env_option": "vision_sequential_default",
  "seed": 0,
  "num_seeds": 5,
  "observation_mode": "image",
  "deterministic_mode": "auto",
  "seed_namespace": "global",
  "budget": {
    "train_steps": 10000,
    "eval_interval_steps": 1000,
    "eval_episodes": 10
  }
}
""".strip(),
        encoding="utf-8",
    )
    smoke_rc = main(
        [
            "smoke-run",
            "--config",
            str(config_path),
            "--out-dir",
            str(out_dir),
            "--max-tasks",
            "2",
        ]
    )
    assert smoke_rc == 0

    validate_rc = main(["validate-artifacts", "--artifacts-dir", str(out_dir)])
    assert validate_rc == 0


def test_validate_artifacts_command_fails_for_invalid_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "broken_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")

    rc = main(["validate-artifacts", "--run-dir", str(run_dir)])
    assert rc == 4


def test_run_experiment1_quality_gate_command(tmp_path: Path) -> None:
    out_dir = tmp_path / "exp1_quality"
    rc = main(
        [
            "run-experiment1-quality-gate",
            "--out-dir",
            str(out_dir),
            "--seed-count",
            "5",
            "--budget-tier",
            "smoke",
        ]
    )
    assert rc == 0
    assert (out_dir / "summary" / "experiment1_metrics_summary.json").exists()
    assert (out_dir / "summary" / "experiment1_metrics_summary.csv").exists()
