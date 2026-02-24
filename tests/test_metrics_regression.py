from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from crlbench.metrics import StreamEvaluation, compute_stream_metrics

FLOAT_TOLERANCE = 1e-12


def _assert_close(lhs: Any, rhs: Any) -> None:  # noqa: ANN401
    if isinstance(lhs, float) and isinstance(rhs, float):
        assert abs(lhs - rhs) <= FLOAT_TOLERANCE
        return
    if isinstance(lhs, dict) and isinstance(rhs, dict):
        assert set(lhs) == set(rhs)
        for key in sorted(lhs):
            _assert_close(lhs[key], rhs[key])
        return
    if isinstance(lhs, list) and isinstance(rhs, list):
        assert len(lhs) == len(rhs)
        for left_value, right_value in zip(lhs, rhs, strict=True):
            _assert_close(left_value, right_value)
        return
    assert lhs == rhs


def test_stream_metrics_golden_regression() -> None:
    stream_trace_path = Path("examples/stream_trace.json")
    golden_path = Path("tests/golden/stream_metrics_golden.json")

    stream_payload = json.loads(stream_trace_path.read_text(encoding="utf-8"))
    golden_metrics = json.loads(golden_path.read_text(encoding="utf-8"))
    metrics = compute_stream_metrics(StreamEvaluation.from_mapping(stream_payload))

    _assert_close(metrics, golden_metrics)
