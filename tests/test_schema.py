import pytest

from crlbench.config.schema import ConfigError, RunConfig


def _base_mapping() -> dict[str, object]:
    return {
        "run_name": "run",
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
    }


def test_run_config_valid() -> None:
    config = RunConfig.from_mapping(_base_mapping())
    assert config.seed == 0
    assert config.track == "toy"
    assert config.budget.train_steps == 10000
    assert config.deterministic_mode == "auto"
    assert config.seed_namespace == "global"


def test_run_config_rejects_bad_track() -> None:
    payload = _base_mapping()
    payload["track"] = "invalid"
    with pytest.raises(ConfigError):
        RunConfig.from_mapping(payload)


def test_budget_interval_must_not_exceed_total_steps() -> None:
    payload = _base_mapping()
    payload["budget"] = {
        "train_steps": 1000,
        "eval_interval_steps": 1001,
        "eval_episodes": 5,
    }
    with pytest.raises(ConfigError):
        RunConfig.from_mapping(payload)


def test_run_config_rejects_invalid_deterministic_mode() -> None:
    payload = _base_mapping()
    payload["deterministic_mode"] = "strict"
    with pytest.raises(ConfigError):
        RunConfig.from_mapping(payload)
