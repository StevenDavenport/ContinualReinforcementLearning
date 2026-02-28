#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import threading
import time
import traceback
import uuid
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PIL import Image

try:
    from crlbench.agents import instantiate_agent
    from crlbench.core.plugins import create_environment
    from crlbench.experiments import (
        build_experiment1_tasks,
        get_experiment1_action_space_n,
        list_experiment1_roots,
        register_experiment1_plugins,
    )
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from crlbench.agents import instantiate_agent
    from crlbench.core.plugins import create_environment
    from crlbench.experiments import (
        build_experiment1_tasks,
        get_experiment1_action_space_n,
        list_experiment1_roots,
        register_experiment1_plugins,
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def _coerce_int(
    value: Any,
    *,
    default: int,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    if not isinstance(value, int):
        return default
    coerced = max(minimum, value)
    if maximum is not None:
        coerced = min(maximum, coerced)
    return coerced


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _find_agent_name_from_run_dir(run_dir: Path) -> str | None:
    summary_path = run_dir / "run_metrics_summary.json"
    if summary_path.exists():
        summary = _read_json(summary_path)
        metadata = summary.get("metadata")
        if isinstance(metadata, dict):
            agent = metadata.get("agent")
            if isinstance(agent, str) and agent.strip():
                return agent.strip()

    events_path = run_dir / "events.jsonl"
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("event") != "run_start":
                continue
            event_payload = payload.get("payload")
            if isinstance(event_payload, dict):
                agent = event_payload.get("agent")
                if isinstance(agent, str) and agent.strip():
                    return agent.strip()
    return None


def _discover_checkpoints(artifacts_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for checkpoint in artifacts_dir.rglob("checkpoints/agent_state.json"):
        run_dir = checkpoint.parent.parent
        agent_name = _find_agent_name_from_run_dir(run_dir)
        run_label = run_dir.name
        label = f"{agent_name or 'unknown_agent'} | {run_label}"
        try:
            mtime = checkpoint.stat().st_mtime
        except OSError:
            mtime = 0.0
        entries.append(
            {
                "checkpoint_path": str(checkpoint),
                "run_dir": str(run_dir),
                "agent_name": agent_name,
                "label": label,
                "mtime": mtime,
            }
        )
    entries.sort(key=lambda item: float(item.get("mtime", 0.0)), reverse=True)
    return entries


def _env_catalog() -> list[dict[str, Any]]:
    roots: list[dict[str, Any]] = []
    for track, env_family, env_option in list_experiment1_roots():
        tasks = build_experiment1_tasks(
            track=track,
            env_family=env_family,
            env_option=env_option,
        )
        roots.append(
            {
                "track": track,
                "env_family": env_family,
                "env_option": env_option,
                "tasks": [task.task_id for task in tasks],
            }
        )
    return roots


def _flatten_rgb_pixels(pixels: list[Any]) -> tuple[int, int, list[tuple[int, int, int]]]:
    if not pixels or not isinstance(pixels, list):
        raise ValueError("pixels must be a non-empty list.")
    height = len(pixels)
    first_row = pixels[0]
    if not isinstance(first_row, list) or not first_row:
        raise ValueError("pixels rows must be non-empty lists.")
    width = len(first_row)
    flat: list[tuple[int, int, int]] = []
    for row in pixels:
        if not isinstance(row, list) or len(row) != width:
            raise ValueError("pixels rows must have consistent width.")
        for cell in row:
            if isinstance(cell, list | tuple) and len(cell) >= 3:
                r = max(0, min(255, int(cell[0])))
                g = max(0, min(255, int(cell[1])))
                b = max(0, min(255, int(cell[2])))
                flat.append((r, g, b))
            else:
                v = max(0, min(255, int(cell)))
                flat.append((v, v, v))
    return width, height, flat


def _pixels_to_data_url(pixels: list[Any]) -> str | None:
    if not isinstance(pixels, list):
        return None
    try:
        width, height, flat = _flatten_rgb_pixels(pixels)
    except ValueError:
        return None

    image = Image.new("RGB", (width, height))
    image.putdata(flat)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _frame_to_data_url(observation: dict[str, Any] | Mapping[str, Any]) -> str | None:
    pixels = observation.get("pixels")
    return _pixels_to_data_url(pixels) if isinstance(pixels, list) else None


def _capture_frame_data_url(
    *,
    env: Any,
    observation: dict[str, Any] | Mapping[str, Any],
    render_size: int,
) -> str | None:
    render_pixels = getattr(env, "render_pixels", None)
    if render_size > 0 and callable(render_pixels):
        try:
            pixels = render_pixels(frame_size=render_size)
        except Exception:  # noqa: BLE001
            pixels = None
        if isinstance(pixels, list):
            high_res = _pixels_to_data_url(pixels)
            if high_res is not None:
                return high_res

    return _frame_to_data_url(observation)


def _checkpoint_agent_config(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        return {}
    payload = _read_json(checkpoint_path)
    config_keys = {
        "obs_dim",
        "hidden_size",
        "action_dim",
        "max_action_dim",
        "gamma",
        "gae_lambda",
        "clip_epsilon",
        "entropy_coef",
        "value_coef",
        "rollout_size",
        "minibatch_size",
        "update_epochs",
        "max_grad_norm",
        "learning_rate",
        "value_learning_rate",
        "actor_learning_rate",
        "log_std_init",
    }
    config: dict[str, Any] = {}
    for key in config_keys:
        value = payload.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            config[key] = value
    return config


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self, request_payload: dict[str, Any]) -> str:
        job_id = uuid.uuid4().hex
        now = time.time()
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "created_at": now,
                "updated_at": now,
                "progress": 0.0,
                "message": "Queued.",
                "request": request_payload,
                "result": None,
                "error": None,
            }
        return job_id

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.update(fields)
            job["updated_at"] = time.time()

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            payload = self._jobs.get(job_id)
            if payload is None:
                return None
            return copy.deepcopy(payload)


def _run_eval_job(  # noqa: PLR0912,PLR0915
    *,
    job_id: str,
    job_store: JobStore,
    agents_dir: Path,
) -> None:
    job = job_store.get(job_id)
    if job is None:
        return
    request = job.get("request")
    if not isinstance(request, dict):
        job_store.update(job_id, status="failed", error="Invalid request payload.")
        return

    checkpoint_path = Path(str(request.get("checkpoint_path", ""))).expanduser()
    agent_name_raw = request.get("agent_name")
    agent_name = agent_name_raw.strip() if isinstance(agent_name_raw, str) else None
    env_family = str(request.get("env_family", "dm_control")).strip()
    env_option = str(request.get("env_option", "vision_sequential_default")).strip()
    task_id = str(request.get("task_id", "")).strip()
    dm_control_backend = str(request.get("dm_control_backend", "real")).strip()
    seed = _coerce_int(request.get("seed"), default=0, minimum=0)
    eval_horizon = _coerce_int(request.get("eval_horizon"), default=200, minimum=1)
    eval_episodes = _coerce_int(request.get("eval_episodes"), default=5, minimum=1)
    frame_stride = _coerce_int(request.get("frame_stride"), default=2, minimum=1)
    max_frames = _coerce_int(request.get("max_frames"), default=400, minimum=1)
    render_size = _coerce_int(request.get("render_size"), default=256, minimum=16, maximum=1024)
    record_episode = _coerce_int(request.get("record_episode"), default=0, minimum=0)
    deterministic = _coerce_bool(request.get("deterministic"), default=True)

    try:
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")
        if not task_id:
            raise ValueError("task_id is required.")

        job_store.update(job_id, status="running", message="Preparing agent and environment.")

        register_experiment1_plugins(replace=True, dm_control_backend=dm_control_backend)

        config = _checkpoint_agent_config(checkpoint_path)
        config["seed"] = seed
        if env_family and env_option:
            config["action_space_n"] = get_experiment1_action_space_n(
                env_family=env_family,
                env_option=env_option,
            )

        agent, descriptor = instantiate_agent(
            agent_name=agent_name,
            agent_path=None,
            agents_dir=agents_dir,
            config=config,
        )
        if not hasattr(agent, "load"):
            raise ValueError(
                f"Agent '{descriptor.name}' does not support checkpoint loading "
                "(missing load(path))."
            )
        agent.reset()
        agent.load(checkpoint_path)

        env = create_environment(
            env_family,
            env_option=env_option,
            task_id=task_id,
            observation_mode="image",
        )

        episode_returns: list[float] = []
        recorded_frames: list[str] = []

        for episode in range(eval_episodes):
            observation = env.reset(seed=seed + episode)
            episode_return = 0.0
            if episode == record_episode and len(recorded_frames) < max_frames:
                first_frame = _capture_frame_data_url(
                    env=env,
                    observation=observation,
                    render_size=render_size,
                )
                if first_frame is not None:
                    recorded_frames.append(first_frame)

            for step_index in range(eval_horizon):
                action = agent.act(observation, deterministic=deterministic)
                step = env.step(action)
                episode_return += float(step.reward)
                observation = step.observation

                if (
                    episode == record_episode
                    and step_index % frame_stride == 0
                    and len(recorded_frames) < max_frames
                ):
                    frame = _capture_frame_data_url(
                        env=env,
                        observation=observation,
                        render_size=render_size,
                    )
                    if frame is not None:
                        recorded_frames.append(frame)

                if step.terminated or step.truncated:
                    break

            episode_returns.append(episode_return)
            progress = (episode + 1) / max(1, eval_episodes)
            job_store.update(
                job_id,
                status="running",
                progress=progress,
                message=f"Episode {episode + 1}/{eval_episodes}",
            )

        env.close()
        mean_return = sum(episode_returns) / float(len(episode_returns))
        variance = 0.0
        if len(episode_returns) > 1:
            mean = mean_return
            variance = sum((value - mean) ** 2 for value in episode_returns) / float(
                len(episode_returns)
            )
        std_return = variance**0.5

        result = {
            "agent_name": descriptor.name,
            "checkpoint_path": str(checkpoint_path),
            "env_family": env_family,
            "env_option": env_option,
            "task_id": task_id,
            "seed": seed,
            "eval_horizon": eval_horizon,
            "eval_episodes": eval_episodes,
            "deterministic": deterministic,
            "render_size": render_size,
            "mean_return": mean_return,
            "std_return": std_return,
            "episode_returns": episode_returns,
            "record_episode": record_episode,
            "recorded_frames": recorded_frames,
        }
        job_store.update(
            job_id,
            status="done",
            progress=1.0,
            message="Evaluation complete.",
            result=result,
        )
    except Exception as exc:  # noqa: BLE001
        job_store.update(
            job_id,
            status="failed",
            message="Evaluation failed.",
            error=f"{exc}\n{traceback.format_exc()}",
        )


class DashboardHandler(BaseHTTPRequestHandler):
    job_store: JobStore | None = None
    artifacts_dir: Path
    agents_dir: Path

    def _json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._html(_index_html())
            return
        if parsed.path == "/api/config":
            config_payload = {
                "artifacts_dir": str(self.artifacts_dir),
                "agents_dir": str(self.agents_dir),
                "checkpoints": _discover_checkpoints(self.artifacts_dir),
                "roots": _env_catalog(),
            }
            self._json(config_payload)
            return
        if parsed.path.startswith("/api/jobs/"):
            job_id = parsed.path.split("/api/jobs/", 1)[1].strip()
            if not job_id:
                self._json({"error": "Missing job id."}, status=HTTPStatus.BAD_REQUEST)
                return
            if self.job_store is None:
                self._json(
                    {"error": "Server is not initialized."},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return
            job_payload = self.job_store.get(job_id)
            if job_payload is None:
                self._json({"error": f"Unknown job id: {job_id}"}, status=HTTPStatus.NOT_FOUND)
                return
            self._json(job_payload)
            return
        self._json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/run-eval":
            self._json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._json({"error": "Invalid JSON body."}, status=HTTPStatus.BAD_REQUEST)
            return
        if not isinstance(payload, dict):
            self._json({"error": "JSON body must be an object."}, status=HTTPStatus.BAD_REQUEST)
            return

        assert self.job_store is not None
        job_id = self.job_store.create(payload)
        thread = threading.Thread(
            target=_run_eval_job,
            kwargs={
                "job_id": job_id,
                "job_store": self.job_store,
                "agents_dir": self.agents_dir,
            },
            daemon=True,
        )
        thread.start()
        self._json({"job_id": job_id})


def _index_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CRLBench Eval Viewer</title>
  <style>
    :root {
      --bg: #eef2f9;
      --card: #ffffff;
      --ink: #12223a;
      --muted: #5c6f8a;
      --line: #d7e0eb;
      --accent: #0f7b8f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 20px;
      font: 14px/1.45 "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background: linear-gradient(180deg, #f6f9ff 0%, var(--bg) 100%);
    }
    h1, h2, h3 { margin: 0 0 10px 0; }
    .card {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--card);
      padding: 14px;
      margin-bottom: 14px;
      box-shadow: 0 4px 10px rgba(14, 31, 55, 0.06);
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
    }
    label {
      display: block;
      font-size: 12px;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 4px;
      letter-spacing: 0.04em;
    }
    select, input, button {
      width: 100%;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
    }
    button {
      cursor: pointer;
      background: var(--accent);
      color: #fff;
      border: none;
      font-weight: 600;
      margin-top: 20px;
    }
    button:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
    .status {
      color: var(--muted);
      font-size: 13px;
      margin-top: 8px;
      white-space: pre-wrap;
    }
    .viewer {
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
    }
    .frame {
      width: 100%;
      max-width: 720px;
      border: 1px solid var(--line);
      border-radius: 10px;
      image-rendering: auto;
      background: #fff;
    }
    .controls {
      max-width: 720px;
      display: grid;
      grid-template-columns: 120px 120px 1fr 120px;
      gap: 8px;
      align-items: center;
    }
    .mono {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      color: #25435f;
      word-break: break-all;
    }
    .tiny {
      font-size: 12px;
      color: var(--muted);
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>CRLBench Eval Viewer</h1>
    <div class="tiny">
      Select checkpoint, task, and eval settings.
      Click Build Eval to render a playable rollout.
    </div>
  </div>

  <div class="card">
    <h2>Configuration</h2>
    <div class="grid">
      <div>
        <label for="checkpoint">Checkpoint</label>
        <select id="checkpoint"></select>
      </div>
      <div>
        <label for="track">Track</label>
        <select id="track"></select>
      </div>
      <div>
        <label for="env_family">Env Family</label>
        <select id="env_family"></select>
      </div>
      <div>
        <label for="env_option">Env Option</label>
        <select id="env_option"></select>
      </div>
      <div>
        <label for="task_id">Task</label>
        <select id="task_id"></select>
      </div>
      <div>
        <label for="dm_control_backend">dm_control Backend</label>
        <select id="dm_control_backend">
          <option value="real">real</option>
          <option value="auto">auto</option>
          <option value="stub">stub</option>
        </select>
      </div>
      <div>
        <label for="eval_horizon">Eval Horizon</label>
        <input id="eval_horizon" type="number" min="1" value="300" />
      </div>
      <div>
        <label for="eval_episodes">Eval Episodes</label>
        <input id="eval_episodes" type="number" min="1" value="5" />
      </div>
      <div>
        <label for="seed">Seed</label>
        <input id="seed" type="number" min="0" value="0" />
      </div>
      <div>
        <label for="record_episode">Record Episode Index</label>
        <input id="record_episode" type="number" min="0" value="0" />
      </div>
      <div>
        <label for="frame_stride">Frame Stride</label>
        <input id="frame_stride" type="number" min="1" value="2" />
      </div>
      <div>
        <label for="max_frames">Max Frames</label>
        <input id="max_frames" type="number" min="1" value="400" />
      </div>
      <div>
        <label for="render_size">Render Size (px)</label>
        <input id="render_size" type="number" min="16" max="1024" value="256" />
      </div>
      <div>
        <label for="deterministic">Deterministic Policy</label>
        <select id="deterministic">
          <option value="true">true</option>
          <option value="false">false</option>
        </select>
      </div>
      <div>
        <button id="run_button">Build Eval</button>
      </div>
    </div>
    <div class="status" id="status">Idle.</div>
  </div>

  <div class="card viewer">
    <h2>Playback</h2>
    <img id="frame" class="frame" alt="Evaluation frame" />
    <div class="controls">
      <button id="play_button" type="button">Play</button>
      <button id="pause_button" type="button">Pause</button>
      <input id="frame_slider" type="range" min="0" max="0" value="0" />
      <input id="fps" type="number" min="1" max="60" value="20" />
    </div>
    <div class="tiny" id="frame_info">No frames loaded.</div>
    <h3>Eval Summary</h3>
    <div class="mono" id="result"></div>
  </div>

  <script>
    const state = {
      checkpoints: [],
      roots: [],
      frames: [],
      playTimer: null,
      frameIndex: 0,
      activeJob: null,
    };

    const el = (id) => document.getElementById(id);

    function setStatus(text) {
      el("status").textContent = text;
    }

    function uniq(values) {
      return Array.from(new Set(values));
    }

    function setOptions(select, values) {
      select.innerHTML = "";
      for (const value of values) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        select.appendChild(option);
      }
    }

    function currentRootMatches(root) {
      return (
        root.track === el("track").value &&
        root.env_family === el("env_family").value &&
        root.env_option === el("env_option").value
      );
    }

    function syncFamilies() {
      const track = el("track").value;
      const families = uniq(
        state.roots.filter((root) => root.track === track).map((root) => root.env_family)
      );
      setOptions(el("env_family"), families);
      syncOptions();
    }

    function syncOptions() {
      const track = el("track").value;
      const family = el("env_family").value;
      const options = uniq(
        state.roots
          .filter((root) => root.track === track && root.env_family === family)
          .map((root) => root.env_option)
      );
      setOptions(el("env_option"), options);
      syncTasks();
    }

    function syncTasks() {
      const selected = state.roots.find((root) => currentRootMatches(root));
      const tasks = selected ? selected.tasks : [];
      setOptions(el("task_id"), tasks);
    }

    function selectCheckpointDefaults() {
      if (!state.checkpoints.length) {
        return;
      }
      const first = state.checkpoints[0];
      const runDir = first.run_dir || "";
      const roots = state.roots;
      for (const root of roots) {
        if (runDir.includes(root.env_option)) {
          el("track").value = root.track;
          syncFamilies();
          el("env_family").value = root.env_family;
          syncOptions();
          el("env_option").value = root.env_option;
          syncTasks();
          if (root.tasks.length) {
            el("task_id").value = root.tasks[0];
          }
          break;
        }
      }
    }

    async function loadConfig() {
      const response = await fetch("/api/config");
      const payload = await response.json();
      state.checkpoints = payload.checkpoints || [];
      state.roots = payload.roots || [];

      const checkpointSelect = el("checkpoint");
      checkpointSelect.innerHTML = "";
      for (const item of state.checkpoints) {
        const option = document.createElement("option");
        option.value = item.checkpoint_path;
        option.textContent = item.label;
        checkpointSelect.appendChild(option);
      }

      const tracks = uniq(state.roots.map((root) => root.track));
      setOptions(el("track"), tracks);
      syncFamilies();
      selectCheckpointDefaults();
      setStatus(`Loaded ${state.checkpoints.length} checkpoints.`);
    }

    function renderFrame() {
      if (!state.frames.length) {
        el("frame").removeAttribute("src");
        el("frame_info").textContent = "No frames loaded.";
        return;
      }
      const index = Math.max(0, Math.min(state.frames.length - 1, state.frameIndex));
      state.frameIndex = index;
      el("frame").src = state.frames[index];
      el("frame_slider").value = String(index);
      el("frame_info").textContent = `Frame ${index + 1}/${state.frames.length}`;
    }

    function stopPlayback() {
      if (state.playTimer !== null) {
        clearInterval(state.playTimer);
        state.playTimer = null;
      }
    }

    function startPlayback() {
      stopPlayback();
      const fps = Math.max(1, Number(el("fps").value || 20));
      const period = Math.floor(1000 / fps);
      state.playTimer = setInterval(() => {
        if (!state.frames.length) {
          stopPlayback();
          return;
        }
        state.frameIndex = (state.frameIndex + 1) % state.frames.length;
        renderFrame();
      }, period);
    }

    async function pollJob(jobId) {
      state.activeJob = jobId;
      while (true) {
        const response = await fetch(`/api/jobs/${jobId}`);
        const payload = await response.json();
        const progressPct = Math.round((Number(payload.progress || 0) * 100));
        setStatus(`${payload.status} | ${payload.message || ""} | ${progressPct}%`);
        if (payload.status === "done") {
          const result = payload.result || {};
          state.frames = Array.isArray(result.recorded_frames) ? result.recorded_frames : [];
          state.frameIndex = 0;
          el("frame_slider").min = "0";
          el("frame_slider").max = String(Math.max(0, state.frames.length - 1));
          el("frame_slider").value = "0";
          renderFrame();
          el("result").textContent = JSON.stringify(
            {
              agent_name: result.agent_name,
              task_id: result.task_id,
              mean_return: result.mean_return,
              std_return: result.std_return,
              eval_episodes: result.eval_episodes,
              eval_horizon: result.eval_horizon,
              episode_returns: result.episode_returns,
            },
            null,
            2
          );
          return;
        }
        if (payload.status === "failed") {
          el("result").textContent = payload.error || "Evaluation failed.";
          stopPlayback();
          return;
        }
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
    }

    async function runEval() {
      stopPlayback();
      const checkpointPath = el("checkpoint").value;
      const selectedCheckpoint = state.checkpoints.find(
        (item) => item.checkpoint_path === checkpointPath
      );
      if (!checkpointPath) {
        setStatus("No checkpoint selected.");
        return;
      }
      const payload = {
        checkpoint_path: checkpointPath,
        agent_name: selectedCheckpoint ? selectedCheckpoint.agent_name : null,
        track: el("track").value,
        env_family: el("env_family").value,
        env_option: el("env_option").value,
        task_id: el("task_id").value,
        dm_control_backend: el("dm_control_backend").value,
        eval_horizon: Number(el("eval_horizon").value),
        eval_episodes: Number(el("eval_episodes").value),
        seed: Number(el("seed").value),
        deterministic: el("deterministic").value === "true",
        record_episode: Number(el("record_episode").value),
        frame_stride: Number(el("frame_stride").value),
        max_frames: Number(el("max_frames").value),
        render_size: Number(el("render_size").value),
      };
      el("run_button").disabled = true;
      setStatus("Submitting evaluation job...");
      const response = await fetch("/api/run-eval", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const result = await response.json();
      if (!response.ok) {
        setStatus(`Error: ${result.error || "request failed"}`);
        el("run_button").disabled = false;
        return;
      }
      await pollJob(result.job_id);
      el("run_button").disabled = false;
    }

    el("track").addEventListener("change", syncFamilies);
    el("env_family").addEventListener("change", syncOptions);
    el("env_option").addEventListener("change", syncTasks);
    el("run_button").addEventListener("click", runEval);
    el("play_button").addEventListener("click", startPlayback);
    el("pause_button").addEventListener("click", stopPlayback);
    el("frame_slider").addEventListener("input", (event) => {
      state.frameIndex = Number(event.target.value);
      renderFrame();
    });

    loadConfig().catch((error) => {
      setStatus(`Failed loading config: ${error}`);
    });
  </script>
</body>
</html>
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Serve an interactive HTML dashboard for qualitative evaluation playback "
            "from saved CRLBench checkpoints."
        )
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=8765, help="Server port.")
    parser.add_argument(
        "--agents-dir",
        type=Path,
        default=Path("agents"),
        help="Agents directory used to instantiate selected checkpoints.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Artifacts root used to discover checkpoints.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    artifacts_dir = args.artifacts_dir.resolve()
    agents_dir = args.agents_dir.resolve()
    if not artifacts_dir.exists():
        raise SystemExit(f"artifacts dir does not exist: {artifacts_dir}")
    if not agents_dir.exists():
        raise SystemExit(f"agents dir does not exist: {agents_dir}")

    job_store = JobStore()

    class _Handler(DashboardHandler):
        pass

    _Handler.job_store = job_store
    _Handler.artifacts_dir = artifacts_dir
    _Handler.agents_dir = agents_dir

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(
        f"Serving CRLBench Eval Viewer at http://{args.host}:{args.port} "
        f"(artifacts={artifacts_dir})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
