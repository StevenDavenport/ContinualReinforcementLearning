from __future__ import annotations

from pathlib import Path

from crlbench.experiments import run_experiment, run_experiment1_matrix


def test_run_experiment_single_root(tmp_path: Path) -> None:
    result = run_experiment(
        experiment="experiment_1_forgetting",
        track="toy",
        env_family="dm_control",
        env_option="vision_sequential_default",
        agent_name="random",
        agent_path=None,
        agents_dir=Path("agents"),
        budget_tier="smoke",
        output_dir=tmp_path / "single",
        run_name="single_smoke",
        seed=0,
        num_seeds=1,
        eval_horizon=4,
        train_steps_cap=6,
        eval_episodes_cap=1,
        dm_control_backend="stub",
    )
    assert len(result.run_results) == 1
    assert result.summary_json_path is not None
    assert result.summary_json_path.exists()


def test_run_experiment1_matrix_all_roots(tmp_path: Path) -> None:
    matrix = run_experiment1_matrix(
        agent_name="random",
        agent_path=None,
        agents_dir=Path("agents"),
        budget_tier="smoke",
        output_dir=tmp_path / "matrix",
        run_name="matrix_smoke",
        seed=0,
        num_seeds=1,
        eval_horizon=4,
        train_steps_cap=6,
        eval_episodes_cap=1,
        dm_control_backend="stub",
    )
    assert len(matrix.root_results) == 5
    for root_result in matrix.root_results.values():
        assert len(root_result.run_results) == 1
