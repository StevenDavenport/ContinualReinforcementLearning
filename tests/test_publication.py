from __future__ import annotations

import json
from pathlib import Path

from crlbench.runtime.publication import export_publication_pack
from crlbench.runtime.reporting import write_run_metrics_summary


def _make_run_dir(root: Path, run_id: str, metric: float) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"schema_version": "1.0.0", "run_id": run_id}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "resolved_config.json").write_text(
        json.dumps({"run_name": run_id, "experiment": "exp1"}) + "\n",
        encoding="utf-8",
    )
    write_run_metrics_summary(
        run_dir=run_dir,
        run_id=run_id,
        metrics={
            "final_stage_average_return": metric,
            "average_forgetting": 0.1,
            "average_return_by_stage": {"s1": 1.0, "s2": metric},
            "forgetting_by_task": {"A": 0.2},
            "retention_by_task": {"A": 0.8},
            "forward_transfer_by_task": {"A": 1.1},
            "backward_transfer_by_task": {"A": 0.1},
            "adaptation_curve": {"0": 0.3, "1": 0.6},
        },
        metadata={
            "experiment": "exp1",
            "track": "toy",
            "env_family": "dm_control",
            "env_option": "vision",
        },
    )
    return run_dir


def test_export_publication_pack(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_a = _make_run_dir(runs_root, "run_a", 1.0)
    run_b = _make_run_dir(runs_root, "run_b", 2.0)

    result = export_publication_pack(
        run_dirs=[run_a, run_b],
        output_dir=tmp_path / "publication_pack",
        method_metadata={"agent": "dreamer", "variant": "v3"},
        render_images=False,
    )

    assert result.run_count == 2
    assert result.experiment_summary_path.exists()
    assert result.csv_path.exists()
    assert result.latex_path.exists()
    assert result.method_metadata_path.exists()
    assert len(result.plot_spec_paths) == 10
    assert len(result.plot_image_paths) == 0
    assert (result.output_dir / "README.md").exists()
