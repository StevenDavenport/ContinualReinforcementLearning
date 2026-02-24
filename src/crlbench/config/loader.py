from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .schema import ConfigError, RunConfig


def _load_mapping(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    elif path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - exercised when yaml extra missing.
            raise ConfigError(
                f"YAML config requested at '{path}', but PyYAML is not installed. "
                "Install with: pip install -e '.[yaml]'."
            ) from exc
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    else:
        raise ConfigError(f"Unsupported config format '{path.suffix}' for '{path}'.")

    if not isinstance(payload, dict):
        raise ConfigError(f"Top-level config in '{path}' must be a mapping.")
    return payload


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(value, Mapping) and isinstance(current, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def resolve_mapping(config_path: Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    path = config_path.resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        chain = " -> ".join(str(p) for p in [*seen, path])
        raise ConfigError(f"Detected cyclic config inheritance: {chain}.")
    seen.add(path)

    payload = _load_mapping(path)
    extends = payload.get("extends")
    if extends is None:
        return payload
    if not isinstance(extends, str) or not extends.strip():
        raise ConfigError("'extends' must be a non-empty relative or absolute path string.")

    parent_path = (path.parent / extends).resolve()
    parent_payload = resolve_mapping(parent_path, seen)
    child_payload = {key: value for key, value in payload.items() if key != "extends"}
    return _deep_merge(parent_payload, child_payload)


def _parse_override_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def parse_override_pairs(pairs: Sequence[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ConfigError(
                f"Invalid override '{pair}'. Expected KEY=VALUE "
                "(example: budget.train_steps=50000)."
            )
        key, raw = pair.split("=", 1)
        dotted = key.strip()
        if not dotted:
            raise ConfigError(f"Invalid override '{pair}'. Override key cannot be empty.")
        overrides[dotted] = _parse_override_value(raw.strip())
    return overrides


def apply_overrides(mapping: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(mapping)
    for dotted_key, value in overrides.items():
        path = [part.strip() for part in dotted_key.split(".") if part.strip()]
        if not path:
            raise ConfigError(f"Invalid override path '{dotted_key}'.")
        cursor = result
        for part in path[:-1]:
            current = cursor.get(part)
            if current is None:
                cursor[part] = {}
                current = cursor[part]
            if not isinstance(current, dict):
                raise ConfigError(
                    f"Cannot assign override '{dotted_key}': '{part}' is not a mapping node."
                )
            cursor = current
        cursor[path[-1]] = value
    return result


def resolve_layered_mapping(base: Path, layers: Sequence[Path] | None = None) -> dict[str, Any]:
    merged = resolve_mapping(base)
    if layers:
        for layer in layers:
            layer_payload = resolve_mapping(layer)
            merged = _deep_merge(merged, layer_payload)
    return merged


def load_run_config(
    config_path: str | Path,
    *,
    layers: Sequence[str | Path] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> RunConfig:
    layer_paths = [Path(layer) for layer in layers] if layers else None
    mapping = resolve_layered_mapping(Path(config_path), layer_paths)
    if overrides:
        mapping = apply_overrides(mapping, overrides)
    return RunConfig.from_mapping(mapping)


def dump_resolved_config(config: RunConfig, out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path
