from __future__ import annotations

from pathlib import Path

import pytest

from crlbench.runtime.storage import resolve_run_storage_layout


def test_resolve_run_storage_layout_path_and_create(tmp_path: Path) -> None:
    layout = resolve_run_storage_layout(
        root=tmp_path / "artifacts",
        experiment="experiment_1_forgetting",
        track="toy",
        env_option="vision_default",
        seed=7,
        run_id="run_abc",
        create=True,
    )
    assert layout.run_dir.exists()
    assert str(layout.run_dir).endswith("experiment_1_forgetting/toy/vision_default/seed_7/run_abc")


def test_resolve_run_storage_layout_rejects_empty_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        resolve_run_storage_layout(
            root=tmp_path,
            experiment="",
            track="toy",
            env_option="vision",
            seed=0,
            run_id="run",
        )
