from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def make_run_id(run_name: str, seed: int, *, now: datetime | None = None) -> str:
    ts = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{run_name}_s{seed}_{ts}"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type '{type(value).__name__}' is not JSON serializable.")


class ArtifactStore:
    """Writes immutable run artifacts under a standard structure."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create_run_dir(self, run_name: str, seed: int) -> Path:
        run_id = make_run_id(run_name, seed)
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def write_json(
        self,
        run_dir: Path,
        relative_path: str | Path,
        payload: Mapping[str, Any],
    ) -> Path:
        path = run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
            handle.write("\n")
        return path

    def append_jsonl(
        self,
        run_dir: Path,
        relative_path: str | Path,
        payload: Mapping[str, Any],
    ) -> Path:
        path = run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            json.dump(payload, handle, default=_json_default, sort_keys=True)
            handle.write("\n")
        return path
