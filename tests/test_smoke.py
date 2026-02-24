from __future__ import annotations

from pathlib import Path

from crlbench.config.schema import RunConfig
from crlbench.runtime.smoke import run_repro_smoke, run_smoke


def _config() -> RunConfig:
    return RunConfig.from_mapping(
        {
            "run_name": "smoke",
            "experiment": "experiment_1_forgetting",
            "track": "toy",
            "env_family": "dm_control",
            "env_option": "vision",
            "seed": 0,
            "num_seeds": 2,
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


def test_run_smoke_writes_artifacts(tmp_path: Path) -> None:
    summary = run_smoke(config=_config(), max_tasks=3, output_dir=tmp_path / "artifacts")
    assert summary.tasks_seen == 3
    assert (summary.run_dir / "manifest.json").exists()
    assert (summary.run_dir / "events.jsonl").exists()


def test_run_repro_smoke_matches_trace(tmp_path: Path) -> None:
    result = run_repro_smoke(config=_config(), max_tasks=2, output_dir=tmp_path / "artifacts")
    assert result.matched is True
    assert result.event_trace_matched is True
    assert result.metric_trace_matched is True
    assert result.max_metric_trace_abs_diff == 0.0
