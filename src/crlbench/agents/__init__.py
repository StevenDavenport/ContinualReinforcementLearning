"""Local-agent discovery, loading, and contract validation."""

from .loader import (
    AgentDescriptor,
    discover_agents,
    instantiate_agent,
    list_agent_names,
    load_agent_factory,
    validate_agent,
)

__all__ = [
    "AgentDescriptor",
    "discover_agents",
    "instantiate_agent",
    "list_agent_names",
    "load_agent_factory",
    "validate_agent",
]
