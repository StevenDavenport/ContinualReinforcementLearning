from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from crlbench.core.contracts import AgentAdapter
from crlbench.core.types import Transition
from crlbench.errors import AgentIntegrationError


@dataclass(frozen=True)
class AgentDescriptor:
    name: str
    adapter_path: Path
    root_dir: Path
    manifest: dict[str, Any]


def _manifest_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AgentIntegrationError(f"Agent manifest at '{path}' must be a JSON object.")
    return payload


def discover_agents(agents_dir: Path | str = Path("agents")) -> dict[str, AgentDescriptor]:
    root = Path(agents_dir)
    if not root.exists():
        return {}
    if not root.is_dir():
        raise AgentIntegrationError(f"agents_dir must be a directory: {root}")

    descriptors: dict[str, AgentDescriptor] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        adapter_path = child / "adapter.py"
        if not adapter_path.exists():
            continue
        manifest = _manifest_payload(child / "manifest.json")
        name_raw = manifest.get("name", child.name)
        if not isinstance(name_raw, str) or not name_raw.strip():
            raise AgentIntegrationError(
                f"Invalid agent name in '{child / 'manifest.json'}': {name_raw!r}"
            )
        name = name_raw.strip()
        if name in descriptors:
            raise AgentIntegrationError(f"Duplicate agent name discovered: '{name}'.")
        descriptors[name] = AgentDescriptor(
            name=name,
            adapter_path=adapter_path.resolve(),
            root_dir=child.resolve(),
            manifest=manifest,
        )
    return descriptors


def list_agent_names(agents_dir: Path | str = Path("agents")) -> tuple[str, ...]:
    return tuple(sorted(discover_agents(agents_dir)))


def _load_module_from_file(path: Path) -> ModuleType:
    module_hash = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]
    module_name = f"crlbench_user_agent_{module_hash}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AgentIntegrationError(f"Unable to import agent module from path: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _coerce_adapter_path(agent_path: Path | str) -> Path:
    raw_path = Path(agent_path)
    candidate = raw_path / "adapter.py" if raw_path.is_dir() else raw_path
    if not candidate.exists():
        raise AgentIntegrationError(f"Agent adapter not found: {candidate}")
    return candidate.resolve()


def load_agent_factory(
    *,
    agent_name: str | None = None,
    agent_path: Path | str | None = None,
    agents_dir: Path | str = Path("agents"),
) -> tuple[Callable[[Mapping[str, Any]], AgentAdapter], AgentDescriptor]:
    if (agent_name is None) == (agent_path is None):
        raise AgentIntegrationError("Provide exactly one of agent_name or agent_path.")

    if agent_name is not None:
        descriptor = discover_agents(agents_dir).get(agent_name)
        if descriptor is None:
            known = ", ".join(list_agent_names(agents_dir)) or "<none>"
            raise AgentIntegrationError(f"Unknown agent '{agent_name}'. Known: {known}.")
    else:
        assert agent_path is not None
        adapter_path = _coerce_adapter_path(agent_path)
        root_dir = adapter_path.parent
        manifest = _manifest_payload(root_dir / "manifest.json")
        descriptor = AgentDescriptor(
            name=root_dir.name,
            adapter_path=adapter_path,
            root_dir=root_dir,
            manifest=manifest,
        )

    module = _load_module_from_file(descriptor.adapter_path)
    factory = getattr(module, "create_agent", None)
    if not callable(factory):
        raise AgentIntegrationError(
            f"Agent '{descriptor.name}' adapter must expose callable create_agent(config)."
        )
    return factory, descriptor


def instantiate_agent(
    *,
    agent_name: str | None = None,
    agent_path: Path | str | None = None,
    agents_dir: Path | str = Path("agents"),
    config: Mapping[str, Any] | None = None,
) -> tuple[AgentAdapter, AgentDescriptor]:
    factory, descriptor = load_agent_factory(
        agent_name=agent_name,
        agent_path=agent_path,
        agents_dir=agents_dir,
    )
    try:
        agent = factory(dict(config or {}))
    except Exception as exc:  # noqa: BLE001
        raise AgentIntegrationError(
            f"Failed constructing agent '{descriptor.name}' from '{descriptor.adapter_path}': {exc}"
        ) from exc
    if not isinstance(agent, AgentAdapter):
        raise AgentIntegrationError(
            f"Agent '{descriptor.name}' does not satisfy AgentAdapter protocol."
        )
    return agent, descriptor


def validate_agent(
    *,
    agent_name: str | None = None,
    agent_path: Path | str | None = None,
    agents_dir: Path | str = Path("agents"),
    config: Mapping[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    try:
        agent, descriptor = instantiate_agent(
            agent_name=agent_name,
            agent_path=agent_path,
            agents_dir=agents_dir,
            config=config,
        )
    except AgentIntegrationError as exc:
        return [str(exc)]

    try:
        agent.reset()
        action = agent.act({"obs": [0.0], "step": 0}, deterministic=False)
        transition = Transition(
            observation={"obs": [0.0], "step": 0},
            action=action,
            reward=1.0,
            next_observation={"obs": [0.1], "step": 1},
            terminated=False,
            truncated=False,
            info={},
        )
        update_metrics_obj: object = agent.update([transition])
        if not isinstance(update_metrics_obj, Mapping):
            errors.append("update(batch) must return a mapping.")
        else:
            for key, value in update_metrics_obj.items():
                if not isinstance(key, str) or not key.strip():
                    errors.append("update(batch) metric keys must be non-empty strings.")
                if not isinstance(value, int | float) or isinstance(value, bool):
                    errors.append("update(batch) metric values must be numeric.")
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "agent_state.json"
            agent.save(checkpoint_path)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Agent '{descriptor.name}' runtime validation failed: {exc}")
    return errors
