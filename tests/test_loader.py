from __future__ import annotations

import json
from pathlib import Path

import pytest

from crlbench.config.loader import (
    apply_overrides,
    load_run_config,
    parse_override_pairs,
    resolve_layered_mapping,
    resolve_mapping,
)
from crlbench.config.schema import ConfigError


def test_loader_resolves_extends(tmp_path: Path) -> None:
    base_path = tmp_path / "base.json"
    child_path = tmp_path / "child.json"

    base_payload = {
        "run_name": "base_run",
        "experiment": "experiment_1_forgetting",
        "track": "toy",
        "env_family": "dm_control",
        "env_option": "vision_seq",
        "seed": 0,
        "num_seeds": 5,
        "observation_mode": "image",
        "budget": {
            "train_steps": 10000,
            "eval_interval_steps": 1000,
            "eval_episodes": 10,
        },
        "tags": ["base"],
    }
    child_payload = {
        "extends": "base.json",
        "run_name": "child_run",
        "budget": {
            "train_steps": 20000,
            "eval_interval_steps": 2000,
            "eval_episodes": 20,
        },
        "tags": ["child"],
    }

    base_path.write_text(json.dumps(base_payload), encoding="utf-8")
    child_path.write_text(json.dumps(child_payload), encoding="utf-8")

    merged = resolve_mapping(child_path)
    assert merged["run_name"] == "child_run"
    assert merged["track"] == "toy"
    assert merged["budget"]["train_steps"] == 20000

    config = load_run_config(child_path)
    assert config.run_name == "child_run"
    assert config.budget.eval_episodes == 20


def test_loader_layered_merge(tmp_path: Path) -> None:
    base_path = tmp_path / "base.json"
    layer_path = tmp_path / "layer.json"
    base_path.write_text(
        json.dumps(
            {
                "run_name": "base",
                "experiment": "experiment_1_forgetting",
                "track": "toy",
                "env_family": "dm_control",
                "env_option": "vision_seq",
                "seed": 0,
                "num_seeds": 5,
                "observation_mode": "image",
                "budget": {
                    "train_steps": 1000,
                    "eval_interval_steps": 100,
                    "eval_episodes": 5,
                },
            }
        ),
        encoding="utf-8",
    )
    layer_path.write_text(
        json.dumps(
            {
                "run_name": "layered",
                "budget": {
                    "train_steps": 2000,
                    "eval_interval_steps": 200,
                    "eval_episodes": 5,
                },
            }
        ),
        encoding="utf-8",
    )
    merged = resolve_layered_mapping(base_path, [layer_path])
    assert merged["run_name"] == "layered"
    assert merged["budget"]["train_steps"] == 2000


def test_parse_override_pairs_and_apply() -> None:
    mapping = {"budget": {"train_steps": 1000}, "track": "toy"}
    parsed = parse_override_pairs(["budget.train_steps=5000", "track=robotics", "seed=7"])
    updated = apply_overrides(mapping, parsed)
    assert updated["budget"]["train_steps"] == 5000
    assert updated["track"] == "robotics"
    assert updated["seed"] == 7


def test_parse_override_pair_rejects_invalid_format() -> None:
    with pytest.raises(ConfigError):
        parse_override_pairs(["budget.train_steps"])


def test_load_run_config_with_layers_and_overrides(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    layer = tmp_path / "layer.json"
    base.write_text(
        json.dumps(
            {
                "run_name": "base",
                "experiment": "experiment_1_forgetting",
                "track": "toy",
                "env_family": "dm_control",
                "env_option": "vision",
                "seed": 0,
                "num_seeds": 5,
                "observation_mode": "image",
                "budget": {
                    "train_steps": 10000,
                    "eval_interval_steps": 1000,
                    "eval_episodes": 10,
                },
            }
        ),
        encoding="utf-8",
    )
    layer.write_text(json.dumps({"run_name": "layered"}), encoding="utf-8")
    config = load_run_config(
        base,
        layers=[layer],
        overrides=parse_override_pairs(["budget.train_steps=15000"]),
    )
    assert config.run_name == "layered"
    assert config.budget.train_steps == 15000
