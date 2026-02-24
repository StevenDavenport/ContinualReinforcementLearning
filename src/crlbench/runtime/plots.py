from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class PlotError(ValueError):
    """Raised when plot generation inputs are invalid."""


@dataclass(frozen=True)
class PlotSeries:
    label: str
    x: list[float]
    y: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "x": self.x, "y": self.y}


@dataclass(frozen=True)
class PlotSpec:
    plot_id: str
    title: str
    kind: str
    x_label: str
    y_label: str
    series: list[PlotSeries] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plot_id": self.plot_id,
            "title": self.title,
            "kind": self.kind,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "series": [item.to_dict() for item in self.series],
            "metadata": self.metadata,
        }


def _series_from_mapping(name: str, values: dict[str, Any]) -> PlotSeries:
    x: list[float] = []
    y: list[float] = []
    for index, key in enumerate(sorted(values.keys())):
        value = values[key]
        if isinstance(value, int | float) and not isinstance(value, bool):
            x.append(float(index))
            y.append(float(value))
    return PlotSeries(label=name, x=x, y=y)


def _extract_scalar_map(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if isinstance(value, dict):
        return value
    return {}


def canonical_plot_specs(run_summary: dict[str, Any]) -> list[PlotSpec]:
    metrics = run_summary.get("metrics")
    if not isinstance(metrics, dict):
        raise PlotError("run summary must include a 'metrics' mapping.")

    avg_return = _extract_scalar_map(metrics, "average_return_by_stage")
    forgetting = _extract_scalar_map(metrics, "forgetting_by_task")
    retention = _extract_scalar_map(metrics, "retention_by_task")
    fwd_transfer = _extract_scalar_map(metrics, "forward_transfer_by_task")
    bwd_transfer = _extract_scalar_map(metrics, "backward_transfer_by_task")
    adaptation = _extract_scalar_map(metrics, "adaptation_curve")

    specs: list[PlotSpec] = [
        PlotSpec(
            plot_id="exp1_forgetting_curve",
            title="Experiment 1: Forgetting/Return Over Stages",
            kind="line",
            x_label="Stage Index",
            y_label="Average Return",
            series=[_series_from_mapping("avg_return", avg_return)],
        ),
        PlotSpec(
            plot_id="exp2_forward_transfer",
            title="Experiment 2: Forward Transfer Ratio",
            kind="bar",
            x_label="Task Index",
            y_label="Transfer Ratio",
            series=[_series_from_mapping("forward_transfer", fwd_transfer)],
        ),
        PlotSpec(
            plot_id="exp3_adaptation",
            title="Experiment 3: Post-Switch Adaptation",
            kind="line",
            x_label="Post-Switch Step",
            y_label="Return",
            series=[_series_from_mapping("adaptation", adaptation)],
        ),
        PlotSpec(
            plot_id="exp4_recall_retention",
            title="Experiment 4: Recall Retention by Task",
            kind="bar",
            x_label="Task Index",
            y_label="Retention",
            series=[_series_from_mapping("retention", retention)],
        ),
        PlotSpec(
            plot_id="exp5_hidden_context",
            title="Experiment 5: Hidden Context Stability",
            kind="bar",
            x_label="Task Index",
            y_label="Forgetting",
            series=[
                _series_from_mapping("forgetting", forgetting),
                _series_from_mapping("backward_transfer", bwd_transfer),
            ],
        ),
    ]
    return specs


def write_plot_specs(specs: list[PlotSpec], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for spec in specs:
        path = output_dir / f"{spec.plot_id}.plot.json"
        path.write_text(
            json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def _render_plot_with_matplotlib(spec: PlotSpec, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError as exc:
        raise PlotError(
            "render_images=True requested, but matplotlib is not installed. "
            "Install matplotlib or run with render_images=False."
        ) from exc

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if spec.kind == "line":
        for series in spec.series:
            ax.plot(series.x, series.y, label=series.label)
    elif spec.kind == "bar":
        for idx, series in enumerate(spec.series):
            shifted_x = [value + (idx * 0.15) for value in series.x]
            ax.bar(shifted_x, series.y, width=0.12, label=series.label)
    else:
        raise PlotError(f"Unsupported plot kind: {spec.kind!r}")

    ax.set_title(spec.title)
    ax.set_xlabel(spec.x_label)
    ax.set_ylabel(spec.y_label)
    if spec.series:
        ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def generate_canonical_plots(
    *,
    run_summary: dict[str, Any],
    output_dir: Path,
    render_images: bool = False,
) -> dict[str, list[Path]]:
    specs = canonical_plot_specs(run_summary)
    spec_paths = write_plot_specs(specs, output_dir)
    image_paths: list[Path] = []
    if render_images:
        for spec in specs:
            image_path = output_dir / f"{spec.plot_id}.png"
            _render_plot_with_matplotlib(spec, image_path)
            image_paths.append(image_path)
    return {"spec_paths": spec_paths, "image_paths": image_paths}
