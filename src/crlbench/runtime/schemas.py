from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = "1.0.0"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RunManifest:
    schema_version: str
    run_id: str
    created_at_utc: str
    run_name: str
    experiment: str
    track: str
    env_family: str
    env_option: str
    seed: int
    num_seeds: int
    deterministic_mode: str
    seed_namespace: str
    config_sha256: str
    code: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    lockfiles: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "run_name": self.run_name,
            "experiment": self.experiment,
            "track": self.track,
            "env_family": self.env_family,
            "env_option": self.env_option,
            "seed": self.seed,
            "num_seeds": self.num_seeds,
            "deterministic_mode": self.deterministic_mode,
            "seed_namespace": self.seed_namespace,
            "config_sha256": self.config_sha256,
            "code": self.code,
            "environment": self.environment,
            "lockfiles": self.lockfiles,
        }


@dataclass(frozen=True)
class EventRecord:
    schema_version: str
    sequence: int
    event: str
    timestamp_utc: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "event": self.event,
            "timestamp_utc": self.timestamp_utc,
            "payload": self.payload,
        }


@dataclass(frozen=True)
class EvalSummaryRecord:
    schema_version: str
    task_id: str
    step: int
    mean_return: float
    std_return: float
    episodes: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "step": self.step,
            "mean_return": self.mean_return,
            "std_return": self.std_return,
            "episodes": self.episodes,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class RunMetricsSummaryRecord:
    schema_version: str
    run_id: str
    created_at_utc: str
    metadata: dict[str, Any]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "metadata": self.metadata,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class ExperimentMetricsSummaryRecord:
    schema_version: str
    created_at_utc: str
    grouping_keys: list[str]
    groups: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at_utc": self.created_at_utc,
            "grouping_keys": self.grouping_keys,
            "groups": self.groups,
        }
