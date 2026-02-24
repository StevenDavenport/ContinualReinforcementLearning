from __future__ import annotations

from crlbench.metrics import (
    StreamEvaluation,
    aggregate_scalar_values,
    backward_transfer_by_task,
    bootstrap_confidence_interval,
    compute_stream_metrics,
    forgetting_by_task,
    forgetting_matrix,
    forward_transfer_by_task,
    paired_bootstrap_difference_ci,
    recovery_time,
    retention_by_task,
    switch_regret,
)


def _stream() -> StreamEvaluation:
    return StreamEvaluation.from_mapping(
        {
            "stages": [
                {"stage_id": "s1", "task_returns": {"A": 10.0}},
                {"stage_id": "s2", "task_returns": {"A": 8.0, "B": 9.0}},
                {"stage_id": "s3", "task_returns": {"A": 6.0, "B": 11.0}},
            ]
        }
    )


def test_forgetting_and_retention_by_task() -> None:
    stream = _stream()
    forgetting = forgetting_by_task(stream)
    retention = retention_by_task(stream)
    assert forgetting["A"] == 4.0
    assert forgetting["B"] == 0.0
    assert retention["A"] == 0.6
    assert retention["B"] == 1.0


def test_forgetting_matrix() -> None:
    stream = _stream()
    matrix = forgetting_matrix(stream)
    assert matrix["values"][0] == [0.0, None]
    assert matrix["values"][1] == [2.0, 0.0]
    assert matrix["values"][2] == [4.0, 0.0]


def test_transfer_metrics() -> None:
    forward = forward_transfer_by_task({"A": 100.0, "B": 50.0}, {"A": 80.0, "B": 25.0})
    backward = backward_transfer_by_task({"A": 10.0, "B": 20.0}, {"A": 15.0, "B": 18.0})
    assert forward["A"] == 1.25
    assert forward["B"] == 2.0
    assert backward["A"] == 5.0
    assert backward["B"] == -2.0


def test_switch_regret_and_recovery() -> None:
    regret = switch_regret(10.0, [6.0, 8.0, 10.0])
    recovery = recovery_time([6.0, 8.0, 10.0], target_return=10.0, step_stride=100)
    assert regret == 6.0
    assert recovery == 200


def test_aggregate_and_bootstrap_stats() -> None:
    stats = aggregate_scalar_values([1.0, 2.0, 3.0])
    assert stats["n"] == 3.0
    assert stats["mean"] == 2.0

    lo, hi = bootstrap_confidence_interval([1.0, 2.0, 3.0], num_bootstrap=1000, seed=7)
    assert lo <= hi
    d_lo, d_hi = paired_bootstrap_difference_ci(
        [2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0],
        num_bootstrap=1000,
        seed=7,
    )
    assert d_lo <= d_hi


def test_compute_stream_metrics_shape() -> None:
    output = compute_stream_metrics(_stream())
    assert "evaluation_matrix" in output
    assert "forgetting_matrix" in output
    assert "retention_matrix" in output
