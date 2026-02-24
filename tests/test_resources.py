from __future__ import annotations

import time

from crlbench.runtime.resources import ResourceMonitor


def test_resource_monitor_snapshot_has_expected_fields() -> None:
    monitor = ResourceMonitor()
    time.sleep(0.001)
    snapshot = monitor.snapshot(steps=10)
    payload = snapshot.to_dict()
    assert payload["steps"] == 10
    assert payload["wall_time_sec"] >= 0.0
    assert payload["steps_per_sec"] >= 0.0
    assert payload["rss_mb"] >= 0.0
