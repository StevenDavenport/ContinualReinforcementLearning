#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from crlbench.agents import instantiate_agent
from crlbench.core.plugins import create_environment
from crlbench.core.types import Transition
from crlbench.experiments import register_experiment1_plugins

_MIN_DREAMER_IMAGE_SIZE = 16


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _coerce_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except Exception:  # noqa: BLE001
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _coerce_compute_dtype(device: str, compute_dtype: str) -> str:
    normalized = str(compute_dtype).strip().lower()
    if normalized != "auto":
        return normalized
    return "bfloat16" if device == "cuda" else "float32"


def _is_numeric(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _positive_int(value: object) -> int | None:
    if not _is_numeric(value):
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def _sequence_len(value: object) -> int | None:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return len(value)
    return None


def _infer_action_spec(observation: Mapping[str, Any]) -> tuple[str, int]:
    flag = observation.get("continuous_action")
    is_continuous: bool | None = None
    if isinstance(flag, bool):
        is_continuous = flag
    elif _is_numeric(flag):
        is_continuous = bool(int(flag))
    elif isinstance(flag, str):
        normalized = flag.strip().lower()
        if normalized in {"1", "true", "yes", "continuous"}:
            is_continuous = True
        elif normalized in {"0", "false", "no", "discrete"}:
            is_continuous = False

    if is_continuous is None:
        has_bounds = _sequence_len(observation.get("action_low")) is not None or _sequence_len(
            observation.get("action_high")
        ) is not None
        has_discrete_n = _positive_int(observation.get("action_space_n")) is not None
        is_continuous = bool(has_bounds and not has_discrete_n)

    if is_continuous:
        action_dim = _positive_int(observation.get("action_dim"))
        if action_dim is None:
            action_dim = _sequence_len(observation.get("action_low"))
        if action_dim is None:
            action_dim = _sequence_len(observation.get("action_high"))
        return "continuous", int(action_dim or 1)

    action_dim = _positive_int(observation.get("action_space_n"))
    if action_dim is None:
        action_dim = _positive_int(observation.get("action_dim"))
    return "discrete", int(action_dim or 2)


def _normalize_pixels(pixels: Any) -> np.ndarray:
    array = np.asarray(pixels)
    if array.ndim == 2:
        normalized = array[..., None]
    elif array.ndim == 3:
        if array.shape[-1] in {1, 3, 4}:
            normalized = array
        elif array.shape[0] in {1, 3, 4}:
            normalized = np.moveaxis(array, 0, -1)
        else:
            raise ValueError(
                "Unsupported pixel shape. Expected HxW, HxWxC, or CxHxW with C in {1,3,4}, "
                f"got {array.shape}."
            )
    else:
        raise ValueError(
            "Unsupported pixel shape. Expected HxW, HxWxC, or CxHxW with C in {1,3,4}, "
            f"got {array.shape}."
        )

    height, width = normalized.shape[:2]
    target_height = max(int(height), _MIN_DREAMER_IMAGE_SIZE)
    target_width = max(int(width), _MIN_DREAMER_IMAGE_SIZE)
    if target_height == height and target_width == width:
        return normalized

    row_index = np.linspace(0, height - 1, num=target_height).round().astype(np.int64)
    col_index = np.linspace(0, width - 1, num=target_width).round().astype(np.int64)
    return normalized[row_index][:, col_index]


def _normalize_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(observation)
    pixels = normalized.get("pixels")
    if pixels is not None:
        normalized["pixels"] = _normalize_pixels(pixels)
    return normalized


def _infer_image_spec(observation: Mapping[str, Any]) -> tuple[int, int]:
    pixels = observation.get("pixels")
    if pixels is None:
        raise SystemExit("Observation does not include 'pixels'; Dreamer image mode is required.")
    normalized = _normalize_pixels(pixels)
    height, width, channels = normalized.shape
    if height != width:
        raise SystemExit(
            "Dreamer tester expects square pixels for encoder/decoder parity; "
            f"got shape {tuple(normalized.shape)}."
        )
    if height <= 0 or channels <= 0:
        raise SystemExit(f"Invalid inferred image shape: {tuple(normalized.shape)}.")
    return int(height), int(channels)


def _snapshot_rollout_state(agent: Any) -> dict[str, Any] | None:
    keys = ("_act_state", "_act_prev_action", "_act_is_first")
    if not all(hasattr(agent, key) for key in keys):
        return None
    return {key: getattr(agent, key) for key in keys}


def _restore_rollout_state(agent: Any, snapshot: dict[str, Any] | None) -> None:
    if snapshot is None:
        return
    for key, value in snapshot.items():
        setattr(agent, key, value)


def _evaluate(  # noqa: PLR0913
    *,
    agent: Any,
    env_family: str,
    env_option: str,
    task_id: str,
    seed: int,
    eval_episodes: int,
    eval_horizon: int,
    step_offset: int,
) -> dict[str, float]:
    env = create_environment(
        env_family,
        env_option=env_option,
        task_id=task_id,
        observation_mode="image",
    )
    rollout_snapshot = _snapshot_rollout_state(agent)
    returns: list[float] = []
    try:
        for episode in range(eval_episodes):
            agent.reset()
            observation = _normalize_observation(env.reset(seed=seed + step_offset + episode))
            episode_return = 0.0
            for _ in range(eval_horizon):
                action = agent.act(observation, deterministic=True)
                step = env.step(action)
                episode_return += float(step.reward)
                observation = _normalize_observation(step.observation)
                if step.terminated or step.truncated:
                    break
            returns.append(episode_return)
    finally:
        env.close()
        _restore_rollout_state(agent, rollout_snapshot)

    mean_return = sum(returns) / max(1, len(returns))
    variance = 0.0
    if len(returns) > 1:
        variance = sum((value - mean_return) ** 2 for value in returns) / len(returns)
    return {
        "eval_mean_return": float(mean_return),
        "eval_std_return": float(math.sqrt(variance)),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One-off Dreamer single-task trainer for dm_control sanity checks. "
            "Writes train/eval curves as JSONL."
        )
    )
    parser.add_argument("--agents-dir", type=Path, default=Path("agents"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--run-name", default="dreamer_sanity")
    parser.add_argument("--env-family", default="dm_control")
    parser.add_argument("--env-option", default="vision_sequential_quadruped_anchor_escape")
    parser.add_argument("--task-id", default="quadruped_run")
    parser.add_argument("--dm-control-backend", default="real", choices=["real", "auto", "stub"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--log-every", type=int, default=5_000)
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-horizon", type=int, default=1_000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--compute-dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-length", type=int, default=32)
    parser.add_argument("--train-ratio", type=float, default=32.0)
    parser.add_argument("--warmup-steps", type=int, default=256)
    parser.add_argument("--imagine-horizon", type=int, default=15)
    parser.add_argument("--imag-last", type=int, default=8)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--deter-dim", type=int, default=512)
    parser.add_argument("--stoch-dim", type=int, default=32)
    parser.add_argument("--classes", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--vector-input-dim", type=int, default=256)
    parser.add_argument("--vector-output-dim", type=int, default=256)
    return parser


def main() -> int:  # noqa: PLR0912,PLR0915
    args = _parser().parse_args()
    if args.steps <= 0:
        raise SystemExit("--steps must be > 0.")
    if args.log_every <= 0:
        raise SystemExit("--log-every must be > 0.")
    if args.eval_every <= 0:
        raise SystemExit("--eval-every must be > 0.")
    if args.eval_episodes <= 0:
        raise SystemExit("--eval-episodes must be > 0.")
    if args.eval_horizon <= 0:
        raise SystemExit("--eval-horizon must be > 0.")
    if args.seed < 0:
        raise SystemExit("--seed must be >= 0.")
    if args.imag_last < 0:
        raise SystemExit("--imag-last must be >= 0.")

    run_name = (
        f"{args.run_name}_{args.env_family}_{args.env_option}_{args.task_id}_"
        f"s{args.seed}_{_utc_stamp()}"
    )
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    train_curve_path = run_dir / "train_curve.jsonl"
    eval_curve_path = run_dir / "eval_curve.jsonl"
    summary_path = run_dir / "summary.json"

    register_experiment1_plugins(replace=True, dm_control_backend=args.dm_control_backend)

    probe_env = create_environment(
        args.env_family,
        env_option=args.env_option,
        task_id=args.task_id,
        observation_mode="image",
    )
    probe_obs = _normalize_observation(probe_env.reset(seed=args.seed))
    action_space, action_dim = _infer_action_spec(probe_obs)
    image_size, image_channels = _infer_image_spec(probe_obs)
    probe_env.close()
    if action_dim <= 0:
        raise SystemExit(f"Invalid action_dim inferred from environment: {action_dim}")

    device = _coerce_device(args.device)
    compute_dtype = _coerce_compute_dtype(device, args.compute_dtype)
    agent_config: dict[str, Any] = {
        "seed": int(args.seed),
        "device": device,
        "compute_dtype": compute_dtype,
        "action_space": action_space,
        "action_dim": int(action_dim),
        "batch_size": int(args.batch_size),
        "batch_length": int(args.batch_length),
        "train_ratio": float(args.train_ratio),
        "warmup_steps": int(args.warmup_steps),
        "imagine_horizon": int(args.imagine_horizon),
        "imag_last": int(args.imag_last),
        "model_dim": int(args.model_dim),
        "embed_dim": int(args.embed_dim),
        "deter_dim": int(args.deter_dim),
        "stoch_dim": int(args.stoch_dim),
        "classes": int(args.classes),
        "blocks": int(args.blocks),
        "vector_input_dim": int(args.vector_input_dim),
        "vector_output_dim": int(args.vector_output_dim),
        "image_size": int(image_size),
        "image_channels": int(image_channels),
    }
    agent, descriptor = instantiate_agent(
        agent_name="dreamer",
        agent_path=None,
        agents_dir=args.agents_dir,
        config=agent_config,
    )
    agent.reset()

    env = create_environment(
        args.env_family,
        env_option=args.env_option,
        task_id=args.task_id,
        observation_mode="image",
    )
    observation = _normalize_observation(env.reset(seed=args.seed))
    wall_start = time.time()
    episode_return = 0.0
    episode_length = 0
    episodes_done = 0
    last_metrics: dict[str, float] = {}
    rolling_returns: list[float] = []
    rolling_lengths: list[int] = []
    train_points = 0
    eval_points = 0

    _append_jsonl(
        run_dir / "events.jsonl",
        {
            "event": "run_start",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "agent": descriptor.name,
            "env_family": args.env_family,
            "env_option": args.env_option,
            "task_id": args.task_id,
            "seed": args.seed,
            "steps": args.steps,
            "device": device,
            "compute_dtype": compute_dtype,
            "imag_last": int(args.imag_last),
            "model_dim": int(args.model_dim),
            "embed_dim": int(args.embed_dim),
            "deter_dim": int(args.deter_dim),
            "stoch_dim": int(args.stoch_dim),
            "classes": int(args.classes),
            "blocks": int(args.blocks),
            "action_space": action_space,
            "action_dim": action_dim,
            "image_size": image_size,
            "image_channels": image_channels,
        },
    )
    print(
        "[start] "
        f"agent={descriptor.name} task={args.task_id} steps={args.steps} "
        f"device={device} dtype={compute_dtype} imag_last={int(args.imag_last)} "
        f"model=md{int(args.model_dim)}/ed{int(args.embed_dim)}/dd{int(args.deter_dim)}/sd{int(args.stoch_dim)} "
        f"cls={int(args.classes)} blk={int(args.blocks)} "
        f"action_space={action_space} action_dim={action_dim} "
        f"image={image_size}x{image_size}x{image_channels} run_dir={run_dir}",
        flush=True,
    )

    try:
        for step_idx in range(1, args.steps + 1):
            action = agent.act(observation, deterministic=False)
            step = env.step(action)
            next_observation = _normalize_observation(step.observation)
            transition = Transition(
                observation=observation,
                action=action,
                reward=float(step.reward),
                next_observation=next_observation,
                terminated=bool(step.terminated),
                truncated=bool(step.truncated),
                info=dict(step.info),
            )
            last_metrics = dict(agent.update([transition]))

            observation = next_observation
            episode_return += float(step.reward)
            episode_length += 1

            if step.terminated or step.truncated:
                episodes_done += 1
                rolling_returns.append(episode_return)
                rolling_lengths.append(episode_length)
                if len(rolling_returns) > 20:
                    rolling_returns = rolling_returns[-20:]
                    rolling_lengths = rolling_lengths[-20:]
                observation = _normalize_observation(env.reset(seed=args.seed + step_idx))
                episode_return = 0.0
                episode_length = 0

            if step_idx % args.log_every == 0 or step_idx == args.steps:
                elapsed = max(1e-6, time.time() - wall_start)
                snapshot = {
                    "step": step_idx,
                    "wall_seconds": elapsed,
                    "steps_per_second": float(step_idx / elapsed),
                    "episodes_done": episodes_done,
                    "avg_episode_return_20": (
                        float(sum(rolling_returns) / len(rolling_returns))
                        if rolling_returns
                        else 0.0
                    ),
                    "avg_episode_length_20": (
                        float(sum(rolling_lengths) / len(rolling_lengths))
                        if rolling_lengths
                        else 0.0
                    ),
                    "world_model_total": float(last_metrics.get("world_model_total", 0.0)),
                    "policy_loss": float(last_metrics.get("policy_loss", 0.0)),
                    "value_loss": float(last_metrics.get("value_loss", 0.0)),
                    "num_updates": float(last_metrics.get("num_updates", 0.0)),
                }
                _append_jsonl(train_curve_path, snapshot)
                train_points += 1
                print(
                    "[train] "
                    f"step={step_idx}/{args.steps} "
                    f"sps={snapshot['steps_per_second']:.1f} "
                    f"ret20={snapshot['avg_episode_return_20']:.2f} "
                    f"len20={snapshot['avg_episode_length_20']:.1f} "
                    f"wm={snapshot['world_model_total']:.4f} "
                    f"pi={snapshot['policy_loss']:.4f} "
                    f"v={snapshot['value_loss']:.4f} "
                    f"upd={snapshot['num_updates']:.1f}",
                    flush=True,
                )

            if step_idx % args.eval_every == 0 or step_idx == args.steps:
                eval_snapshot = {
                    "step": step_idx,
                    **_evaluate(
                        agent=agent,
                        env_family=args.env_family,
                        env_option=args.env_option,
                        task_id=args.task_id,
                        seed=args.seed + 100_000,
                        eval_episodes=args.eval_episodes,
                        eval_horizon=args.eval_horizon,
                        step_offset=step_idx,
                    ),
                }
                _append_jsonl(eval_curve_path, eval_snapshot)
                eval_points += 1
                print(
                    "[eval] "
                    f"step={step_idx} mean_return={eval_snapshot['eval_mean_return']:.3f} "
                    f"std_return={eval_snapshot['eval_std_return']:.3f}",
                    flush=True,
                )
    finally:
        env.close()

    checkpoint_path = run_dir / "agent_state.json"
    agent.save(checkpoint_path)

    summary_payload = {
        "run_type": "dreamer_single_task_tester",
        "agent": descriptor.name,
        "env_family": args.env_family,
        "env_option": args.env_option,
        "task_id": args.task_id,
        "seed": args.seed,
        "steps": args.steps,
        "device": device,
        "compute_dtype": compute_dtype,
        "imag_last": int(args.imag_last),
        "model_dim": int(args.model_dim),
        "embed_dim": int(args.embed_dim),
        "deter_dim": int(args.deter_dim),
        "stoch_dim": int(args.stoch_dim),
        "classes": int(args.classes),
        "blocks": int(args.blocks),
        "action_space": action_space,
        "action_dim": action_dim,
        "image_size": image_size,
        "image_channels": image_channels,
        "train_points": train_points,
        "eval_points": eval_points,
        "train_curve_path": str(train_curve_path),
        "eval_curve_path": str(eval_curve_path),
        "checkpoint_path": str(checkpoint_path),
        "last_metrics": {
            "world_model_total": float(last_metrics.get("world_model_total", 0.0)),
            "policy_loss": float(last_metrics.get("policy_loss", 0.0)),
            "value_loss": float(last_metrics.get("value_loss", 0.0)),
            "num_updates": float(last_metrics.get("num_updates", 0.0)),
            "episodes_done": episodes_done,
        },
    }
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _append_jsonl(
        run_dir / "events.jsonl",
        {
            "event": "run_complete",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "summary_path": str(summary_path),
            "checkpoint_path": str(checkpoint_path),
        },
    )
    print(f"[done] summary={summary_path} checkpoint={checkpoint_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
