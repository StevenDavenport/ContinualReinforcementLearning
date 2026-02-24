from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from crlbench.config.schema import RunConfig

from .schemas import SCHEMA_VERSION, RunManifest, utc_now_iso


def hash_payload(payload: dict[str, Any]) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def hash_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_command(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def detect_git_sha(repo_root: Path) -> str:
    return _git_command(repo_root, "rev-parse", "HEAD") or "unknown"


def detect_git_dirty(repo_root: Path) -> bool | None:
    status = _git_command(repo_root, "status", "--porcelain")
    if status is None:
        return None
    return bool(status.strip())


def get_package_versions(packages: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def build_manifest(
    *,
    config: RunConfig,
    run_id: str,
    config_sha256: str,
    repo_root: Path,
) -> RunManifest:
    lockfiles = {
        "pyproject_toml": hash_file(repo_root / "pyproject.toml"),
        "environment_yml": hash_file(repo_root / "environment.yml"),
    }
    code = {
        "repo_root": str(repo_root),
        "git_sha": detect_git_sha(repo_root),
        "git_dirty": detect_git_dirty(repo_root),
    }
    environment = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": get_package_versions(["crlbench", "pytest", "ruff", "mypy", "PyYAML"]),
    }
    return RunManifest(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        created_at_utc=utc_now_iso(),
        run_name=config.run_name,
        experiment=config.experiment,
        track=config.track,
        env_family=config.env_family,
        env_option=config.env_option,
        seed=config.seed,
        num_seeds=config.num_seeds,
        deterministic_mode=config.deterministic_mode,
        seed_namespace=config.seed_namespace,
        config_sha256=config_sha256,
        code=code,
        environment=environment,
        lockfiles=lockfiles,
    )
