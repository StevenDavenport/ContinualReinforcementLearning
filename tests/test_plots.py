from __future__ import annotations

import json
from pathlib import Path

from crlbench.runtime.plots import canonical_plot_specs, generate_canonical_plots


def _summary() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "run_id": "run_plot_test",
        "created_at_utc": "generated-by-test",
        "metadata": {"experiment": "exp1", "track": "toy"},
        "metrics": {
            "average_return_by_stage": {"s1": 1.0, "s2": 2.0},
            "forgetting_by_task": {"A": 0.2, "B": 0.1},
            "retention_by_task": {"A": 0.8, "B": 1.0},
            "forward_transfer_by_task": {"A": 1.1, "B": 1.2},
            "backward_transfer_by_task": {"A": 0.1, "B": -0.1},
            "adaptation_curve": {"0": 0.3, "1": 0.5, "2": 0.8},
        },
    }


def test_canonical_plot_specs_has_five_experiment_specs() -> None:
    specs = canonical_plot_specs(_summary())
    assert len(specs) == 5
    assert specs[0].plot_id == "exp1_forgetting_curve"
    assert specs[4].plot_id == "exp5_hidden_context"


def test_generate_canonical_plots_writes_specs(tmp_path: Path) -> None:
    output = generate_canonical_plots(
        run_summary=_summary(),
        output_dir=tmp_path,
        render_images=False,
    )
    assert len(output["spec_paths"]) == 5
    assert len(output["image_paths"]) == 0
    payload = json.loads(output["spec_paths"][0].read_text(encoding="utf-8"))
    assert payload["plot_id"] == "exp1_forgetting_curve"
