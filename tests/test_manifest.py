from __future__ import annotations

from pathlib import Path

from crlbench.config.schema import RunConfig
from crlbench.runtime.manifest import build_manifest, hash_payload


def _config() -> RunConfig:
    return RunConfig.from_mapping(
        {
            "run_name": "manifest_test",
            "experiment": "experiment_1_forgetting",
            "track": "toy",
            "env_family": "dm_control",
            "env_option": "vision",
            "seed": 1,
            "num_seeds": 3,
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


def test_manifest_build_contains_repro_fields(tmp_path: Path) -> None:
    config = _config()
    config_hash = hash_payload(config.to_dict())
    manifest = build_manifest(
        config=config,
        run_id="manifest_test_run",
        config_sha256=config_hash,
        repo_root=tmp_path,
    )
    payload = manifest.to_dict()
    assert payload["run_id"] == "manifest_test_run"
    assert payload["config_sha256"] == config_hash
    assert payload["deterministic_mode"] == "auto"
    assert payload["seed_namespace"] == "global"
    assert "git_sha" in payload["code"]
