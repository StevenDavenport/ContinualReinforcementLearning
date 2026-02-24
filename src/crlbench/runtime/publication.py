from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .plots import generate_canonical_plots
from .reporting import (
    aggregate_run_metric_summaries,
    export_summary_csv,
    export_summary_latex,
    load_run_metrics_summary,
    write_experiment_metrics_summary,
)
from .schemas import SCHEMA_VERSION, utc_now_iso


class PublicationError(ValueError):
    """Raised when publication pack export inputs are invalid."""


@dataclass(frozen=True)
class PublicationPackResult:
    output_dir: Path
    run_count: int
    experiment_summary_path: Path
    csv_path: Path
    latex_path: Path
    method_metadata_path: Path
    plot_spec_paths: list[Path]
    plot_image_paths: list[Path]


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _load_required_run_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_metrics_summary.json"
    if not path.exists():
        raise PublicationError(f"Missing run metrics summary: {path}")
    return load_run_metrics_summary(path)


def export_publication_pack(
    *,
    run_dirs: list[Path],
    output_dir: Path,
    method_metadata: dict[str, Any] | None = None,
    render_images: bool = False,
) -> PublicationPackResult:
    if not run_dirs:
        raise PublicationError("run_dirs cannot be empty.")

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir = output_dir / "summaries"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    manifests_dir = output_dir / "manifests"
    configs_dir = output_dir / "configs"
    metadata_dir = output_dir / "metadata"
    for directory in [
        summaries_dir,
        tables_dir,
        figures_dir,
        manifests_dir,
        configs_dir,
        metadata_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    run_summaries: list[dict[str, Any]] = []
    all_spec_paths: list[Path] = []
    all_image_paths: list[Path] = []

    for run_dir in run_dirs:
        summary = _load_required_run_summary(run_dir)
        run_id = str(summary.get("run_id", run_dir.name))
        run_summaries.append(summary)

        _copy_if_exists(
            run_dir / "run_metrics_summary.json",
            summaries_dir / f"{run_id}.run_metrics_summary.json",
        )
        _copy_if_exists(run_dir / "manifest.json", manifests_dir / f"{run_id}.manifest.json")
        _copy_if_exists(
            run_dir / "resolved_config.json",
            configs_dir / f"{run_id}.resolved_config.json",
        )

        plot_out_dir = figures_dir / run_id
        plot_outputs = generate_canonical_plots(
            run_summary=summary,
            output_dir=plot_out_dir,
            render_images=render_images,
        )
        all_spec_paths.extend(plot_outputs["spec_paths"])
        all_image_paths.extend(plot_outputs["image_paths"])

    aggregated = aggregate_run_metric_summaries(run_summaries)
    experiment_summary_path = summaries_dir / "experiment_metrics_summary.json"
    write_experiment_metrics_summary(output_path=experiment_summary_path, summary=aggregated)

    csv_path = tables_dir / "experiment_metrics_summary.csv"
    latex_path = tables_dir / "experiment_metrics_summary.tex"
    export_summary_csv(aggregated, csv_path)
    export_summary_latex(aggregated, latex_path)

    method_payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now_iso(),
        "method_metadata": dict(method_metadata or {}),
        "run_dirs": [str(path) for path in run_dirs],
        "render_images": render_images,
    }
    method_metadata_path = metadata_dir / "method_metadata.json"
    method_metadata_path.write_text(
        json.dumps(method_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    readme_path = output_dir / "README.md"
    readme_lines = [
        "# Publication Pack",
        "",
        f"- generated_at_utc: {method_payload['created_at_utc']}",
        f"- run_count: {len(run_dirs)}",
        f"- render_images: {render_images}",
        "",
        "## Contents",
        "- `summaries/` run summaries and aggregated summary",
        "- `tables/` csv and latex aggregate tables",
        "- `figures/` canonical plot specs and optional images",
        "- `manifests/` copied run manifests",
        "- `configs/` copied resolved configs",
        "- `metadata/method_metadata.json`",
    ]
    readme_path.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    return PublicationPackResult(
        output_dir=output_dir,
        run_count=len(run_dirs),
        experiment_summary_path=experiment_summary_path,
        csv_path=csv_path,
        latex_path=latex_path,
        method_metadata_path=method_metadata_path,
        plot_spec_paths=all_spec_paths,
        plot_image_paths=all_image_paths,
    )
