from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from crlbench.agents import list_agent_names, validate_agent
from crlbench.config.loader import (
    apply_overrides,
    dump_resolved_config,
    load_run_config,
    parse_override_pairs,
    resolve_layered_mapping,
)
from crlbench.config.schema import ConfigError
from crlbench.experiments import (
    EXPERIMENT_ID,
    run_experiment,
    run_experiment1_matrix,
    run_experiment1_quality_gate,
)
from crlbench.metrics import StreamEvaluation, compute_stream_metrics
from crlbench.runtime.plots import generate_canonical_plots
from crlbench.runtime.publication import export_publication_pack
from crlbench.runtime.reporting import (
    aggregate_run_metric_summaries,
    export_summary_csv,
    export_summary_latex,
    load_run_metrics_summary,
    write_experiment_metrics_summary,
)
from crlbench.runtime.smoke import run_repro_smoke, run_smoke
from crlbench.runtime.validation import validate_artifacts_dir, validate_run_artifacts


def _add_common_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path, help="Path to base config file.")
    parser.add_argument(
        "--layer",
        action="append",
        default=[],
        type=Path,
        help=(
            "Optional overlay config path. Can be provided multiple times. "
            "Applied in order of appearance."
        ),
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help=(
            "Override key/value pair as dotted path, e.g. budget.train_steps=50000 "
            'or tags=["smoke","ci"]. Can be repeated.'
        ),
    )


def _build_parser() -> argparse.ArgumentParser:  # noqa: PLR0915
    parser = argparse.ArgumentParser(prog="crlbench", description="CRL benchmark suite CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-config", help="Validate a run config.")
    _add_common_config_args(validate_parser)

    resolve_parser = subparsers.add_parser("resolve-config", help="Resolve config and write JSON.")
    _add_common_config_args(resolve_parser)
    resolve_parser.add_argument("--out", required=True, type=Path, help="Output JSON file path.")

    print_parser = subparsers.add_parser(
        "print-config",
        help="Print resolved config JSON to stdout.",
    )
    _add_common_config_args(print_parser)

    compose_parser = subparsers.add_parser(
        "compose-config",
        help="Compose base+layers+overrides without requiring full run-schema validation.",
    )
    compose_parser.add_argument(
        "--base",
        required=True,
        type=Path,
        help="Path to base config file.",
    )
    compose_parser.add_argument(
        "--layer",
        action="append",
        default=[],
        type=Path,
        help="Optional overlay config path. Can be provided multiple times.",
    )
    compose_parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override key/value pair as dotted path, e.g. budget.train_steps=50000.",
    )
    compose_parser.add_argument("--out", required=True, type=Path, help="Output JSON file path.")

    smoke_parser = subparsers.add_parser(
        "smoke-run",
        help="Execute deterministic dry-run smoke protocol and write artifacts.",
    )
    _add_common_config_args(smoke_parser)
    smoke_parser.add_argument(
        "--max-tasks",
        type=int,
        default=3,
        help="Number of synthetic smoke tasks to emit.",
    )
    smoke_parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output directory override for smoke artifacts.",
    )

    repro_parser = subparsers.add_parser(
        "repro-smoke",
        help="Run smoke protocol twice and verify event-trace reproducibility.",
    )
    _add_common_config_args(repro_parser)
    repro_parser.add_argument(
        "--max-tasks",
        type=int,
        default=3,
        help="Number of synthetic smoke tasks to emit per run.",
    )
    repro_parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output directory override for smoke artifacts.",
    )
    repro_parser.add_argument(
        "--metric-tolerance",
        type=float,
        default=0.0,
        help="Absolute tolerance for reproducibility metric-trace comparison.",
    )

    metrics_parser = subparsers.add_parser(
        "compute-stream-metrics",
        help="Compute stream metrics from a trace JSON file.",
    )
    metrics_parser.add_argument("--trace", required=True, type=Path, help="Trace JSON input path.")
    metrics_parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output run metrics summary JSON path.",
    )
    metrics_parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        help="Metadata key/value pairs as KEY=VALUE. Repeatable.",
    )

    aggregate_parser = subparsers.add_parser(
        "aggregate-metric-summaries",
        help="Aggregate run metric summaries into experiment-level artifacts.",
    )
    aggregate_parser.add_argument(
        "--summary",
        action="append",
        default=[],
        type=Path,
        help="Path to run_metrics_summary.json. Repeatable.",
    )
    aggregate_parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output experiment summary JSON path.",
    )
    aggregate_parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV export path.",
    )
    aggregate_parser.add_argument(
        "--latex",
        type=Path,
        default=None,
        help="Optional LaTeX export path.",
    )

    plot_parser = subparsers.add_parser(
        "generate-canonical-plots",
        help="Generate canonical plot specs (and optional images) from run metrics summary.",
    )
    plot_parser.add_argument(
        "--summary",
        required=True,
        type=Path,
        help="Path to run_metrics_summary.json.",
    )
    plot_parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for plot artifacts.",
    )
    plot_parser.add_argument(
        "--render-images",
        action="store_true",
        help="Render PNG images (requires matplotlib).",
    )

    pack_parser = subparsers.add_parser(
        "export-publication-pack",
        help="Export one-command publication pack (figures/tables/manifests/configs/metadata).",
    )
    pack_parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        type=Path,
        help="Run directory containing manifest/resolved_config/run_metrics_summary. Repeatable.",
    )
    pack_parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for publication pack.",
    )
    pack_parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="Method metadata as KEY=VALUE. Repeatable.",
    )
    pack_parser.add_argument(
        "--render-images",
        action="store_true",
        help="Render PNG plot images (requires matplotlib).",
    )

    validate_artifacts_parser = subparsers.add_parser(
        "validate-artifacts",
        help="Validate run artifacts for schema and file consistency.",
    )
    target_group = validate_artifacts_parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--run-dir",
        action="append",
        default=None,
        type=Path,
        help="Run directory to validate. Repeatable.",
    )
    target_group.add_argument(
        "--artifacts-dir",
        type=Path,
        help="Artifacts directory containing run subdirectories.",
    )

    exp1_quality_parser = subparsers.add_parser(
        "run-experiment1-quality-gate",
        help="Run Experiment 1 parity/reproducibility quality gate and emit publication artifacts.",
    )
    exp1_quality_parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for Experiment 1 quality artifacts.",
    )
    exp1_quality_parser.add_argument(
        "--seed-count",
        type=int,
        default=5,
        help="Number of seeds for reproducibility verification (default: 5).",
    )
    exp1_quality_parser.add_argument(
        "--budget-tier",
        default="smoke",
        choices=("smoke", "dev", "full"),
        help="Experiment 1 budget tier used for synthetic gate generation.",
    )

    list_agents_parser = subparsers.add_parser(
        "list-agents",
        help="List drop-in agents discovered under agents directory.",
    )
    list_agents_parser.add_argument(
        "--agents-dir",
        type=Path,
        default=Path("agents"),
        help="Directory containing agent subfolders (default: agents).",
    )

    validate_agent_parser = subparsers.add_parser(
        "validate-agent",
        help="Validate a drop-in agent adapter against AgentAdapter contract smoke checks.",
    )
    validate_target = validate_agent_parser.add_mutually_exclusive_group(required=True)
    validate_target.add_argument(
        "--agent",
        help="Agent name discovered in --agents-dir.",
    )
    validate_target.add_argument(
        "--agent-path",
        type=Path,
        help="Path to adapter.py or agent directory containing adapter.py.",
    )
    validate_agent_parser.add_argument(
        "--agents-dir",
        type=Path,
        default=Path("agents"),
        help="Directory containing agent subfolders (default: agents).",
    )
    validate_agent_parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Agent config key/value pair KEY=VALUE (JSON values supported). Repeatable.",
    )

    run_parser = subparsers.add_parser(
        "run-experiment",
        help="Run a benchmark experiment with a drop-in agent.",
    )
    run_parser.add_argument(
        "--experiment",
        default=EXPERIMENT_ID,
        choices=(EXPERIMENT_ID,),
        help="Experiment identifier to execute.",
    )
    run_parser.add_argument("--track", required=True, choices=("toy", "robotics"))
    run_parser.add_argument("--env-family", required=True)
    run_parser.add_argument("--env-option", required=True)
    run_target = run_parser.add_mutually_exclusive_group(required=True)
    run_target.add_argument(
        "--agent",
        help="Agent name discovered in --agents-dir.",
    )
    run_target.add_argument(
        "--agent-path",
        type=Path,
        help="Path to adapter.py or agent directory containing adapter.py.",
    )
    run_parser.add_argument(
        "--agents-dir",
        type=Path,
        default=Path("agents"),
        help="Directory containing agent subfolders (default: agents).",
    )
    run_parser.add_argument(
        "--tier",
        default="smoke",
        choices=("smoke", "dev", "full"),
        help="Budget tier to run.",
    )
    run_parser.add_argument(
        "--dm-control-backend",
        default="auto",
        choices=("auto", "stub", "real"),
        help="dm_control backend selection for dm_control roots.",
    )
    run_parser.add_argument("--seed", type=int, default=0, help="Initial seed.")
    run_parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of consecutive seeds starting at --seed.",
    )
    run_parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name prefix.",
    )
    run_parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts"),
        help="Artifact output directory.",
    )
    run_parser.add_argument(
        "--eval-horizon",
        type=int,
        default=32,
        help="Per-episode evaluation horizon.",
    )
    run_parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help=(
            "Optional evaluation episode override per task at each stage. "
            "If unset, tier default is used."
        ),
    )
    run_parser.add_argument(
        "--train-steps-cap",
        type=int,
        default=None,
        help="Optional cap for train steps per stage (useful for quick test runs).",
    )
    run_parser.add_argument(
        "--eval-episodes-cap",
        type=int,
        default=None,
        help="Optional cap for evaluation episodes (useful for quick test runs).",
    )
    run_parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Agent config key/value pair KEY=VALUE (JSON values supported). Repeatable.",
    )

    matrix_parser = subparsers.add_parser(
        "run-experiment1-matrix",
        help="Run Experiment 1 across all toy+robotics roots with one agent.",
    )
    matrix_target = matrix_parser.add_mutually_exclusive_group(required=True)
    matrix_target.add_argument(
        "--agent",
        help="Agent name discovered in --agents-dir.",
    )
    matrix_target.add_argument(
        "--agent-path",
        type=Path,
        help="Path to adapter.py or agent directory containing adapter.py.",
    )
    matrix_parser.add_argument(
        "--agents-dir",
        type=Path,
        default=Path("agents"),
        help="Directory containing agent subfolders (default: agents).",
    )
    matrix_parser.add_argument(
        "--tier",
        default="smoke",
        choices=("smoke", "dev", "full"),
        help="Budget tier to run for each root.",
    )
    matrix_parser.add_argument(
        "--dm-control-backend",
        default="auto",
        choices=("auto", "stub", "real"),
        help="dm_control backend selection for dm_control root in matrix runs.",
    )
    matrix_parser.add_argument("--seed", type=int, default=0, help="Initial seed.")
    matrix_parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of consecutive seeds per root.",
    )
    matrix_parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name prefix.",
    )
    matrix_parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/exp1_matrix"),
        help="Artifact output directory.",
    )
    matrix_parser.add_argument(
        "--eval-horizon",
        type=int,
        default=32,
        help="Per-episode evaluation horizon.",
    )
    matrix_parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="Optional evaluation episode override per task at each stage.",
    )
    matrix_parser.add_argument(
        "--train-steps-cap",
        type=int,
        default=None,
        help="Optional cap for train steps per stage.",
    )
    matrix_parser.add_argument(
        "--eval-episodes-cap",
        type=int,
        default=None,
        help="Optional cap for evaluation episodes.",
    )
    matrix_parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Agent config key/value pair KEY=VALUE (JSON values supported). Repeatable.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # noqa: PLR0911,PLR0912,PLR0915
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        overrides = parse_override_pairs(getattr(args, "set", []))

        if args.command == "validate-config":
            config = load_run_config(args.config, layers=args.layer, overrides=overrides)
            print(
                f"Config OK: run={config.run_name} experiment={config.experiment} "
                f"track={config.track} env={config.env_family}/{config.env_option}"
            )
            return 0

        if args.command == "resolve-config":
            config = load_run_config(args.config, layers=args.layer, overrides=overrides)
            path = dump_resolved_config(config, args.out)
            print(f"Wrote resolved config: {path}")
            return 0

        if args.command == "print-config":
            config = load_run_config(args.config, layers=args.layer, overrides=overrides)
            print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
            return 0

        if args.command == "compose-config":
            payload = resolve_layered_mapping(args.base, args.layer)
            if overrides:
                payload = apply_overrides(payload, overrides)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
            print(f"Wrote composed config: {args.out}")
            return 0

        if args.command == "smoke-run":
            config = load_run_config(args.config, layers=args.layer, overrides=overrides)
            smoke_summary = run_smoke(
                config=config,
                max_tasks=args.max_tasks,
                output_dir=args.out_dir,
            )
            print(
                f"Smoke run complete: run_id={smoke_summary.run_id} "
                f"tasks_seen={smoke_summary.tasks_seen} run_dir={smoke_summary.run_dir}"
            )
            return 0

        if args.command == "repro-smoke":
            config = load_run_config(args.config, layers=args.layer, overrides=overrides)
            repro_result = run_repro_smoke(
                config=config,
                max_tasks=args.max_tasks,
                metric_tolerance=args.metric_tolerance,
                output_dir=args.out_dir,
            )
            if repro_result.matched:
                print(
                    "Repro smoke PASS: event+metric traces matched "
                    f"({repro_result.first.run_id} vs {repro_result.second.run_id}); "
                    f"max_metric_abs_diff={repro_result.max_metric_trace_abs_diff:.6f}."
                )
                return 0
            print(
                "Repro smoke FAIL: traces differ "
                f"({repro_result.first.run_id} vs {repro_result.second.run_id}); "
                f"event_match={repro_result.event_trace_matched} "
                f"metric_match={repro_result.metric_trace_matched} "
                f"max_metric_abs_diff={repro_result.max_metric_trace_abs_diff:.6f}."
            )
            return 3

        if args.command == "compute-stream-metrics":
            trace_payload = json.loads(args.trace.read_text(encoding="utf-8"))
            if not isinstance(trace_payload, dict):
                raise ValueError("trace payload must be a JSON object.")
            stream = StreamEvaluation.from_mapping(trace_payload)
            metrics = compute_stream_metrics(stream)
            metadata = parse_override_pairs(args.metadata)
            output_payload = {
                "schema_version": "1.0.0",
                "run_id": str(metadata.get("run_id", "unknown")),
                "created_at_utc": "generated-by-cli",
                "metadata": metadata,
                "metrics": metrics,
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(output_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"Wrote stream metrics summary: {args.out}")
            return 0

        if args.command == "aggregate-metric-summaries":
            if not args.summary:
                raise ValueError("At least one --summary file is required.")
            summaries = [load_run_metrics_summary(path) for path in args.summary]
            aggregated = aggregate_run_metric_summaries(summaries)
            write_experiment_metrics_summary(output_path=args.out, summary=aggregated)
            if args.csv is not None:
                export_summary_csv(aggregated, args.csv)
            if args.latex is not None:
                export_summary_latex(aggregated, args.latex)
            print(f"Wrote aggregated metrics summary: {args.out}")
            return 0

        if args.command == "generate-canonical-plots":
            run_summary_payload = load_run_metrics_summary(args.summary)
            outputs = generate_canonical_plots(
                run_summary=run_summary_payload,
                output_dir=args.out_dir,
                render_images=args.render_images,
            )
            print(
                f"Wrote canonical plot specs: {len(outputs['spec_paths'])} "
                f"images: {len(outputs['image_paths'])} to {args.out_dir}"
            )
            return 0

        if args.command == "export-publication-pack":
            if not args.run_dir:
                raise ValueError("At least one --run-dir is required.")
            method_metadata = parse_override_pairs(args.method)
            pack_result = export_publication_pack(
                run_dirs=args.run_dir,
                output_dir=args.out_dir,
                method_metadata=method_metadata,
                render_images=args.render_images,
            )
            print(
                f"Wrote publication pack: {pack_result.output_dir} "
                f"(runs={pack_result.run_count}, plot_specs={len(pack_result.plot_spec_paths)}, "
                f"plot_images={len(pack_result.plot_image_paths)})"
            )
            return 0

        if args.command == "validate-artifacts":
            if args.run_dir:
                results = {
                    run_dir.name: validate_run_artifacts(run_dir) for run_dir in args.run_dir
                }
            else:
                assert args.artifacts_dir is not None
                results = validate_artifacts_dir(args.artifacts_dir)

            if not results:
                raise ValueError("No run directories found to validate.")

            failures = {name: errs for name, errs in results.items() if errs}
            if failures:
                for name, errors in failures.items():
                    print(f"[FAIL] {name}")
                    for error in errors:
                        print(f"  - {error}")
                valid_count = len(results) - len(failures)
                print(
                    f"Artifact validation FAILED: {len(failures)} run(s) invalid, "
                    f"{valid_count} valid."
                )
                return 4

            print(f"Artifact validation PASS: {len(results)} run(s) valid.")
            return 0

        if args.command == "run-experiment1-quality-gate":
            result = run_experiment1_quality_gate(
                output_dir=args.out_dir,
                seed_count=args.seed_count,
                budget_tier=args.budget_tier,
            )
            status = result.metric_parity_passed and result.reproducibility_passed
            print(
                f"Experiment1 quality gate {'PASS' if status else 'FAIL'}: "
                f"parity={result.metric_parity_passed} "
                f"reproducibility={result.reproducibility_passed} "
                f"seed_count={result.seed_count} output_dir={result.output_dir}"
            )
            return 0 if status else 5

        if args.command == "list-agents":
            names = list_agent_names(args.agents_dir)
            if not names:
                print(f"No agents discovered under {args.agents_dir}.")
                return 0
            for name in names:
                print(name)
            return 0

        if args.command == "validate-agent":
            errors = validate_agent(
                agent_name=args.agent,
                agent_path=args.agent_path,
                agents_dir=args.agents_dir,
                config=overrides,
            )
            if errors:
                print("Agent validation FAIL:")
                for error in errors:
                    print(f"  - {error}")
                return 6
            label = args.agent if args.agent is not None else str(args.agent_path)
            print(f"Agent validation PASS: {label}")
            return 0

        if args.command == "run-experiment":
            run_result = run_experiment(
                experiment=args.experiment,
                track=args.track,
                env_family=args.env_family,
                env_option=args.env_option,
                agent_name=args.agent,
                agent_path=args.agent_path,
                agents_dir=args.agents_dir,
                budget_tier=args.tier,
                output_dir=args.out_dir,
                run_name=args.run_name,
                seed=args.seed,
                num_seeds=args.num_seeds,
                eval_horizon=args.eval_horizon,
                eval_episodes_override=args.eval_episodes,
                train_steps_cap=args.train_steps_cap,
                eval_episodes_cap=args.eval_episodes_cap,
                dm_control_backend=args.dm_control_backend,
                agent_config=overrides,
            )
            summary_text = "none"
            if run_result.summary_json_path is not None:
                summary_text = str(run_result.summary_json_path)
            print(
                f"Run complete: runs={len(run_result.run_results)} "
                f"summary={summary_text} out_dir={args.out_dir}"
            )
            return 0

        if args.command == "run-experiment1-matrix":
            matrix_result = run_experiment1_matrix(
                agent_name=args.agent,
                agent_path=args.agent_path,
                agents_dir=args.agents_dir,
                budget_tier=args.tier,
                output_dir=args.out_dir,
                run_name=args.run_name,
                seed=args.seed,
                num_seeds=args.num_seeds,
                eval_horizon=args.eval_horizon,
                eval_episodes_override=args.eval_episodes,
                train_steps_cap=args.train_steps_cap,
                eval_episodes_cap=args.eval_episodes_cap,
                dm_control_backend=args.dm_control_backend,
                agent_config=overrides,
            )
            print(
                f"Experiment1 matrix complete: roots={len(matrix_result.root_results)} "
                f"out_dir={args.out_dir}"
            )
            return 0
    except ConfigError as exc:
        print(f"Config error: {exc}")
        return 2
    except ValueError as exc:
        print(f"Value error: {exc}")
        return 2

    parser.error("Unsupported command.")
