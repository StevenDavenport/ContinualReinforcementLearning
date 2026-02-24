from __future__ import annotations

import os
import resource
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResourceSnapshot:
    steps: int
    wall_time_sec: float
    steps_per_sec: float
    rss_mb: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "wall_time_sec": self.wall_time_sec,
            "steps_per_sec": self.steps_per_sec,
            "rss_mb": self.rss_mb,
        }


def _read_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss_value = float(usage.ru_maxrss)
    if os.name == "posix":
        return rss_value / 1024.0
    return rss_value / (1024.0 * 1024.0)


class ResourceMonitor:
    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._last_steps = 0
        self._last_time = self._start

    def reset(self) -> None:
        self._start = time.perf_counter()
        self._last_steps = 0
        self._last_time = self._start

    def snapshot(self, *, steps: int) -> ResourceSnapshot:
        if steps < 0:
            raise ValueError(f"steps must be non-negative, got {steps}.")
        now = time.perf_counter()
        wall_time = max(0.0, now - self._start)
        delta_steps = max(0, steps - self._last_steps)
        delta_time = max(1e-9, now - self._last_time)
        steps_per_sec = float(delta_steps) / delta_time
        self._last_steps = steps
        self._last_time = now
        return ResourceSnapshot(
            steps=steps,
            wall_time_sec=wall_time,
            steps_per_sec=steps_per_sec,
            rss_mb=_read_rss_mb(),
        )
