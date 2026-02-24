from __future__ import annotations

import json
from pathlib import Path

from crlbench.config.schema import RunConfig
from crlbench.runtime.smoke import run_smoke
from crlbench.runtime.validation import validate_artifacts_dir, validate_run_artifacts


def _config() -> RunConfig:
    return RunConfig.from_mapping(
        {
            "run_name": "validation_smoke",
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


def test_validate_run_artifacts_passes_for_smoke_run(tmp_path: Path) -> None:
    summary = run_smoke(config=_config(), max_tasks=2, output_dir=tmp_path / "artifacts")
    errors = validate_run_artifacts(summary.run_dir)
    assert errors == []


def test_validate_run_artifacts_detects_config_hash_mismatch(tmp_path: Path) -> None:
    summary = run_smoke(config=_config(), max_tasks=2, output_dir=tmp_path / "artifacts")
    resolved_path = summary.run_dir / "resolved_config.json"
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    payload["run_name"] = "mutated"
    resolved_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    errors = validate_run_artifacts(summary.run_dir)
    assert any("config_sha256 does not match" in err for err in errors)


def test_validate_artifacts_dir_reports_missing_dir(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    results = validate_artifacts_dir(missing)
    assert "__root__" in results
