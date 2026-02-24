from __future__ import annotations

import pytest

from crlbench.config.schema import ConfigError
from crlbench.runtime.seeding import derive_seed_map, derive_subseed, resolve_deterministic_mode


def test_derive_subseed_is_stable() -> None:
    a = derive_subseed(7, "env")
    b = derive_subseed(7, "env")
    c = derive_subseed(7, "agent")
    assert a == b
    assert a != c


def test_derive_seed_map() -> None:
    seeds = derive_seed_map(11, ["agent", "env", "eval"])
    assert set(seeds) == {"agent", "env", "eval"}
    assert len(set(seeds.values())) == 3


def test_resolve_deterministic_mode() -> None:
    assert resolve_deterministic_mode("off", supports_determinism=False) is False
    assert resolve_deterministic_mode("auto", supports_determinism=False) is False
    assert resolve_deterministic_mode("auto", supports_determinism=True) is True
    assert resolve_deterministic_mode("on", supports_determinism=True) is True
    with pytest.raises(ConfigError):
        resolve_deterministic_mode("on", supports_determinism=False)
