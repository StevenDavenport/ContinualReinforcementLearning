from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .manifest import hash_payload


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def validate_run_artifacts(run_dir: Path) -> list[str]:  # noqa: PLR0912,PLR0915
    errors: list[str] = []
    required = ["manifest.json", "resolved_config.json", "events.jsonl", "state.json"]
    for name in required:
        if not (run_dir / name).exists():
            errors.append(f"missing file: {name}")
    if errors:
        return errors

    try:
        manifest = _read_json(run_dir / "manifest.json")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"invalid manifest.json: {exc}")
        return errors

    try:
        resolved_config = _read_json(run_dir / "resolved_config.json")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"invalid resolved_config.json: {exc}")
        return errors

    try:
        state = _read_json(run_dir / "state.json")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"invalid state.json: {exc}")
        return errors

    for key in ["schema_version", "run_id", "config_sha256"]:
        if key not in manifest:
            errors.append(f"manifest missing key: {key}")

    computed_hash = hash_payload(resolved_config)
    manifest_hash = manifest.get("config_sha256")
    if not isinstance(manifest_hash, str) or manifest_hash != computed_hash:
        errors.append("manifest config_sha256 does not match resolved_config.json")

    events_path = run_dir / "events.jsonl"
    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        errors.append("events.jsonl has no events")
        return errors

    last_sequence: int | None = None
    last_event: str | None = None
    for idx, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"events.jsonl line {idx + 1} invalid json: {exc}")
            continue
        if not isinstance(record, dict):
            errors.append(f"events.jsonl line {idx + 1} is not a JSON object")
            continue
        sequence = record.get("sequence")
        event = record.get("event")
        if not isinstance(sequence, int):
            errors.append(f"events.jsonl line {idx + 1} missing integer sequence")
            continue
        if last_sequence is not None and sequence != last_sequence + 1:
            errors.append(
                f"events sequence discontinuity at line {idx + 1}: {sequence} after {last_sequence}"
            )
        if not isinstance(event, str):
            errors.append(f"events.jsonl line {idx + 1} missing string event")
        last_sequence = sequence
        if isinstance(event, str):
            last_event = event

    tasks_seen = state.get("tasks_seen")
    sequence = state.get("sequence")
    completed = state.get("completed")
    if not isinstance(tasks_seen, int):
        errors.append("state.tasks_seen must be int")
    if not isinstance(sequence, int):
        errors.append("state.sequence must be int")
    if not isinstance(completed, bool):
        errors.append("state.completed must be bool")

    if isinstance(sequence, int) and last_sequence is not None and sequence != last_sequence + 1:
        errors.append("state.sequence must be last event sequence + 1")

    if completed is True and last_event != "run_complete":
        errors.append("state.completed is true but final event is not run_complete")
    if completed is False and last_event == "run_complete":
        errors.append("state.completed is false but final event is run_complete")

    return errors


def validate_artifacts_dir(artifacts_dir: Path) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {}
    if not artifacts_dir.exists():
        return {"__root__": [f"artifacts directory does not exist: {artifacts_dir}"]}
    for run_dir in sorted(
        path
        for path in artifacts_dir.rglob("*")
        if path.is_dir() and (path / "manifest.json").exists()
    ):
        key = str(run_dir.relative_to(artifacts_dir))
        results[key] = validate_run_artifacts(run_dir)
    return results
