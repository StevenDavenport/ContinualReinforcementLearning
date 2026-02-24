from __future__ import annotations

from pathlib import Path

from crlbench.experiments import run_experiment1_quality_gate


def test_run_experiment1_quality_gate_emits_parity_repro_and_reports(tmp_path: Path) -> None:
    result = run_experiment1_quality_gate(
        output_dir=tmp_path / "exp1_quality",
        seed_count=5,
        budget_tier="smoke",
    )
    assert result.metric_parity_passed is True
    assert result.reproducibility_passed is True
    assert result.summary_json_path.exists()
    assert result.summary_csv_path.exists()
    assert result.summary_latex_path.exists()

    # Canonical plot specs should exist for each root in the quality gate.
    plot_specs = list((tmp_path / "exp1_quality" / "plots").rglob("*.plot.json"))
    assert len(plot_specs) >= 4
