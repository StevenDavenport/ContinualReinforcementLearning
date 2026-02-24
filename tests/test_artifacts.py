from __future__ import annotations

import json
from pathlib import Path

from crlbench.runtime.artifacts import ArtifactStore


def test_artifact_store_writes_json_and_jsonl(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    run_dir = store.create_run_dir("test_run", seed=0)

    json_path = store.write_json(run_dir, "manifest.json", {"run": "test_run"})
    jsonl_path = store.append_jsonl(run_dir, "events.jsonl", {"event": "start"})
    store.append_jsonl(run_dir, "events.jsonl", {"event": "end"})

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()

    assert payload["run"] == "test_run"
    assert len(lines) == 2
