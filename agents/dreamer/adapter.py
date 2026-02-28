from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def create_agent(config: Mapping[str, Any]) -> object:
    agents_root = Path(__file__).resolve().parent.parent
    if str(agents_root) not in sys.path:
        sys.path.insert(0, str(agents_root))
    from dreamer.agent import DreamerAgent
    from dreamer.config import parse_dreamer_config

    parsed = parse_dreamer_config(config)
    return DreamerAgent(parsed)
