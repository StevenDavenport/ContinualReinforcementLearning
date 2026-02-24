from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crlbench.errors import ConfigurationError

VALID_TRACKS = {"toy", "robotics"}
VALID_OBSERVATION_MODES = {"image", "image_proprio"}
VALID_DETERMINISTIC_MODES = {"auto", "on", "off"}


class ConfigError(ConfigurationError):
    """Raised when the run config is invalid."""


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_positive_int(name: str, value: Any) -> int:
    if not _is_plain_int(value) or value <= 0:
        raise ConfigError(f"'{name}' must be a positive integer; got {value!r}.")
    return int(value)


def _require_non_negative_int(name: str, value: Any) -> int:
    if not _is_plain_int(value) or value < 0:
        raise ConfigError(f"'{name}' must be a non-negative integer; got {value!r}.")
    return int(value)


def _require_nonempty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{name}' must be a non-empty string; got {value!r}.")
    return value.strip()


def _require_string_list(name: str, value: Any) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if not isinstance(value, list):
        raise ConfigError(f"'{name}' must be a list of strings; got {type(value).__name__}.")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"'{name}' must only contain non-empty strings; got {item!r}.")
        result.append(item.strip())
    return tuple(result)


@dataclass(frozen=True)
class BudgetConfig:
    train_steps: int
    eval_interval_steps: int
    eval_episodes: int

    def __post_init__(self) -> None:
        if self.eval_interval_steps > self.train_steps:
            raise ConfigError(
                "'budget.eval_interval_steps' must be <= 'budget.train_steps' for scheduled eval."
            )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> BudgetConfig:
        required = {"train_steps", "eval_interval_steps", "eval_episodes"}
        unknown = set(data) - required
        missing = required - set(data)
        if unknown:
            raise ConfigError(f"Unknown budget keys: {sorted(unknown)}.")
        if missing:
            raise ConfigError(f"Missing budget keys: {sorted(missing)}.")
        return cls(
            train_steps=_require_positive_int("budget.train_steps", data["train_steps"]),
            eval_interval_steps=_require_positive_int(
                "budget.eval_interval_steps", data["eval_interval_steps"]
            ),
            eval_episodes=_require_positive_int("budget.eval_episodes", data["eval_episodes"]),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "train_steps": self.train_steps,
            "eval_interval_steps": self.eval_interval_steps,
            "eval_episodes": self.eval_episodes,
        }


@dataclass(frozen=True)
class RunConfig:
    run_name: str
    experiment: str
    track: str
    env_family: str
    env_option: str
    seed: int
    num_seeds: int
    observation_mode: str
    deterministic_mode: str
    seed_namespace: str
    budget: BudgetConfig
    output_dir: Path = Path("artifacts")
    tags: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> RunConfig:
        required = {
            "run_name",
            "experiment",
            "track",
            "env_family",
            "env_option",
            "seed",
            "num_seeds",
            "observation_mode",
            "budget",
        }
        optional = {"output_dir", "tags", "extends", "deterministic_mode", "seed_namespace"}
        unknown = set(data) - (required | optional)
        missing = required - set(data)
        if unknown:
            raise ConfigError(f"Unknown run config keys: {sorted(unknown)}.")
        if missing:
            raise ConfigError(f"Missing run config keys: {sorted(missing)}.")

        track = _require_nonempty_string("track", data["track"])
        if track not in VALID_TRACKS:
            raise ConfigError(f"'track' must be one of {sorted(VALID_TRACKS)}; got {track!r}.")

        observation_mode = _require_nonempty_string("observation_mode", data["observation_mode"])
        if observation_mode not in VALID_OBSERVATION_MODES:
            raise ConfigError(
                f"'observation_mode' must be one of {sorted(VALID_OBSERVATION_MODES)}; "
                f"got {observation_mode!r}."
            )
        deterministic_mode = _require_nonempty_string(
            "deterministic_mode", data.get("deterministic_mode", "auto")
        )
        if deterministic_mode not in VALID_DETERMINISTIC_MODES:
            raise ConfigError(
                f"'deterministic_mode' must be one of {sorted(VALID_DETERMINISTIC_MODES)}; "
                f"got {deterministic_mode!r}."
            )
        seed_namespace = _require_nonempty_string(
            "seed_namespace",
            data.get("seed_namespace", "global"),
        )

        budget_data = data["budget"]
        if not isinstance(budget_data, Mapping):
            raise ConfigError("'budget' must be a mapping.")

        output_dir_value = data.get("output_dir", "artifacts")
        output_dir = Path(_require_nonempty_string("output_dir", output_dir_value))

        seed = _require_non_negative_int("seed", data["seed"])
        num_seeds = _require_positive_int("num_seeds", data["num_seeds"])

        return cls(
            run_name=_require_nonempty_string("run_name", data["run_name"]),
            experiment=_require_nonempty_string("experiment", data["experiment"]),
            track=track,
            env_family=_require_nonempty_string("env_family", data["env_family"]),
            env_option=_require_nonempty_string("env_option", data["env_option"]),
            seed=seed,
            num_seeds=num_seeds,
            observation_mode=observation_mode,
            deterministic_mode=deterministic_mode,
            seed_namespace=seed_namespace,
            budget=BudgetConfig.from_mapping(budget_data),
            output_dir=output_dir,
            tags=_require_string_list("tags", data.get("tags")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "experiment": self.experiment,
            "track": self.track,
            "env_family": self.env_family,
            "env_option": self.env_option,
            "seed": self.seed,
            "num_seeds": self.num_seeds,
            "observation_mode": self.observation_mode,
            "deterministic_mode": self.deterministic_mode,
            "seed_namespace": self.seed_namespace,
            "budget": self.budget.to_dict(),
            "output_dir": str(self.output_dir),
            "tags": list(self.tags),
        }
