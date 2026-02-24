from __future__ import annotations

import hashlib
from collections.abc import Sequence

from crlbench.config.schema import VALID_DETERMINISTIC_MODES, ConfigError

SEED_MODULUS = 2**31 - 1


def derive_subseed(global_seed: int, namespace: str) -> int:
    if global_seed < 0:
        raise ConfigError(f"global_seed must be non-negative, got {global_seed}.")
    if not namespace.strip():
        raise ConfigError("namespace must be a non-empty string.")
    payload = f"{global_seed}:{namespace}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % SEED_MODULUS


def derive_seed_map(global_seed: int, namespaces: Sequence[str]) -> dict[str, int]:
    return {namespace: derive_subseed(global_seed, namespace) for namespace in namespaces}


def resolve_deterministic_mode(mode: str, *, supports_determinism: bool) -> bool:
    if mode not in VALID_DETERMINISTIC_MODES:
        raise ConfigError(
            f"deterministic mode must be one of {sorted(VALID_DETERMINISTIC_MODES)}, got {mode!r}."
        )
    if mode == "off":
        return False
    if mode == "auto":
        return supports_determinism
    if supports_determinism:
        return True
    raise ConfigError(
        "deterministic_mode is 'on', but selected environment backend does not advertise "
        "deterministic execution support."
    )
