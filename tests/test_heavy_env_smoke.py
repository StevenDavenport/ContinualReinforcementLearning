from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from crlbench.cli import main

HEAVY_ENV_PROFILES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("metaworld", ("metaworld",)),
    ("maniskill", ("mani_skill", "mani_skill2")),
    ("robosuite", ("robosuite",)),
    ("rlbench", ("rlbench",)),
)


def _profile_available(modules: tuple[str, ...]) -> bool:
    return any(importlib.util.find_spec(module) is not None for module in modules)


@pytest.mark.heavy_env
@pytest.mark.parametrize(("env_family", "modules"), HEAVY_ENV_PROFILES)
def test_optional_heavy_env_pipeline(
    env_family: str, modules: tuple[str, ...], tmp_path: Path
) -> None:
    if not _profile_available(modules):
        pytest.skip(
            f"Optional heavy environment modules not installed for {env_family}: {modules}."
        )

    config_path = tmp_path / f"{env_family}.json"
    out_dir = tmp_path / "artifacts"
    payload = {
        "run_name": f"heavy_{env_family}",
        "experiment": "experiment_1_forgetting",
        "track": "robotics",
        "env_family": env_family,
        "env_option": "default",
        "seed": 0,
        "num_seeds": 1,
        "observation_mode": "image_proprio",
        "deterministic_mode": "auto",
        "seed_namespace": "global",
        "budget": {
            "train_steps": 1000,
            "eval_interval_steps": 100,
            "eval_episodes": 2,
        },
    }
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    smoke_rc = main(
        [
            "smoke-run",
            "--config",
            str(config_path),
            "--out-dir",
            str(out_dir),
            "--max-tasks",
            "1",
        ]
    )
    assert smoke_rc == 0

    validate_rc = main(["validate-artifacts", "--artifacts-dir", str(out_dir)])
    assert validate_rc == 0
