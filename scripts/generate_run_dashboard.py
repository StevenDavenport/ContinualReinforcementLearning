#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunBundle:
    run_id: str
    path: str
    summary: dict[str, Any]
    stream_trace: dict[str, Any] | None
    plot_specs: list[dict[str, Any]]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}.")
    return payload


def _discover_summary(summary_dir: Path) -> dict[str, Any] | None:
    if not summary_dir.exists():
        return None
    paths = sorted(summary_dir.glob("*_metrics_summary.json"))
    if not paths:
        return None
    return _load_json(paths[0])


def _discover_runs(root: Path) -> list[RunBundle]:
    bundles: list[RunBundle] = []
    for child in sorted(path for path in root.iterdir() if path.is_dir()):
        run_summary_path = child / "run_metrics_summary.json"
        if not run_summary_path.exists():
            continue

        stream_trace_path = child / "stream_trace.json"
        stream_trace = _load_json(stream_trace_path) if stream_trace_path.exists() else None

        plot_dir = child / "plots"
        plot_specs: list[dict[str, Any]] = []
        if plot_dir.exists():
            for path in sorted(plot_dir.glob("*.plot.json")):
                plot_specs.append(_load_json(path))

        run_summary = _load_json(run_summary_path)
        bundles.append(
            RunBundle(
                run_id=child.name,
                path=str(child),
                summary=run_summary,
                stream_trace=stream_trace,
                plot_specs=plot_specs,
            )
        )
    return bundles


def _html_document(data: dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CRLBench Run Dashboard</title>
  <style>
    :root {{
      --bg: #f4f7fb;
      --card: #ffffff;
      --ink: #12223a;
      --muted: #5a6b84;
      --line: #d7e0eb;
      --accent: #0e7c86;
      --accent2: #e27f2d;
      --accent3: #3e9b4f;
      --accent4: #7d54c7;
      --accent5: #b0366f;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      padding: 24px;
      background: linear-gradient(180deg, #f8fbff 0%, #eef3fa 100%);
      color: var(--ink);
      font: 14px/1.45 "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    }}

    h1, h2, h3 {{
      margin: 0 0 8px 0;
      line-height: 1.2;
    }}

    h1 {{
      font-size: 30px;
      letter-spacing: -0.01em;
    }}

    h2 {{
      margin-top: 12px;
      font-size: 22px;
    }}

    h3 {{
      margin-top: 8px;
      font-size: 16px;
    }}

    .subtitle {{
      color: var(--muted);
      margin-bottom: 4px;
    }}

    .section {{
      margin-top: 20px;
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--card);
      box-shadow: 0 3px 10px rgba(12, 27, 49, 0.05);
    }}

    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-top: 10px;
    }}

    .card {{
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      background: #fcfeff;
    }}

    .metric-label {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 5px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}

    .metric-value {{
      font-size: 22px;
      font-weight: 650;
    }}

    .grid-2 {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 14px;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
    }}

    thead th {{
      text-align: left;
      font-size: 12px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--muted);
      background: #f7faff;
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
    }}

    tbody td {{
      padding: 7px 10px;
      border-bottom: 1px solid #edf2f8;
      vertical-align: top;
    }}

    tbody tr:last-child td {{
      border-bottom: 0;
    }}

    .run-title {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: baseline;
      flex-wrap: wrap;
    }}

    .mono {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      color: #1d3550;
      word-break: break-all;
    }}

    .plot-grid {{
      margin-top: 12px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(430px, 1fr));
      gap: 12px;
    }}

    .plot-card {{
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px;
      background: #ffffff;
    }}

    .plot-title {{
      margin-bottom: 6px;
      font-weight: 600;
    }}

    canvas {{
      width: 100%;
      max-width: 100%;
      height: 300px;
      border: 1px solid #edf2f8;
      border-radius: 8px;
      background: #fff;
    }}

    .legend {{
      margin-top: 8px;
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 12px;
    }}

    .chip {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}

    .dot {{
      width: 10px;
      height: 10px;
      border-radius: 99px;
      display: inline-block;
    }}
  </style>
</head>
<body>
  <div class="section">
    <h1>CRLBench Run Dashboard</h1>
    <div class="subtitle" id="run_root"></div>
    <div class="subtitle" id="generated_at"></div>
    <div class="cards" id="headline_cards"></div>
  </div>

  <div class="section" id="aggregate_section">
    <h2>Aggregate Summary</h2>
    <div class="grid-2" id="aggregate_tables"></div>
  </div>

  <div class="section">
    <h2>Per-Seed Results</h2>
    <div id="runs"></div>
  </div>

  <script>
    const DASHBOARD_DATA = {payload};
    const COLORS = ["#0e7c86", "#e27f2d", "#3e9b4f", "#7d54c7", "#b0366f", "#2a5f9d"];

    function fmt(value, digits = 4) {{
      if (typeof value === "number" && Number.isFinite(value)) {{
        return value.toFixed(digits);
      }}
      return String(value);
    }}

    function scalar(value) {{
      if (typeof value === "number" && Number.isFinite(value)) {{
        return value;
      }}
      return null;
    }}

    function buildMapTable(title, mapping) {{
      const wrap = document.createElement("div");
      wrap.className = "card";
      const h = document.createElement("h3");
      h.textContent = title;
      wrap.appendChild(h);

      const keys = Object.keys(mapping || {{}}).sort();
      if (!keys.length) {{
        const empty = document.createElement("div");
        empty.className = "subtitle";
        empty.textContent = "No values";
        wrap.appendChild(empty);
        return wrap;
      }}

      const table = document.createElement("table");
      const thead = document.createElement("thead");
      thead.innerHTML = "<tr><th>Key</th><th>Value</th></tr>";
      table.appendChild(thead);
      const tbody = document.createElement("tbody");
      for (const key of keys) {{
        const tr = document.createElement("tr");
        const tdK = document.createElement("td");
        tdK.textContent = key;
        const tdV = document.createElement("td");
        const val = mapping[key];
        tdV.textContent = typeof val === "number" ? fmt(val) : JSON.stringify(val);
        tr.appendChild(tdK);
        tr.appendChild(tdV);
        tbody.appendChild(tr);
      }}
      table.appendChild(tbody);
      wrap.appendChild(table);
      return wrap;
    }}

    function drawAxes(ctx, width, height, margin) {{
      ctx.strokeStyle = "#8ea3bd";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(margin.left, height - margin.bottom);
      ctx.lineTo(width - margin.right, height - margin.bottom);
      ctx.moveTo(margin.left, margin.top);
      ctx.lineTo(margin.left, height - margin.bottom);
      ctx.stroke();
    }}

    function drawLinePlot(canvas, spec) {{
      const ctx = canvas.getContext("2d");
      const width = canvas.width;
      const height = canvas.height;
      const margin = {{ left: 44, right: 12, top: 16, bottom: 32 }};

      ctx.clearRect(0, 0, width, height);
      drawAxes(ctx, width, height, margin);

      const allX = [];
      const allY = [];
      for (const series of spec.series || []) {{
        for (const x of series.x || []) allX.push(Number(x));
        for (const y of series.y || []) allY.push(Number(y));
      }}
      if (!allX.length || !allY.length) return;

      const minX = Math.min(...allX);
      const maxX = Math.max(...allX);
      let minY = Math.min(...allY);
      let maxY = Math.max(...allY);
      if (minY === maxY) {{
        minY -= 1;
        maxY += 1;
      }}

      const plotW = width - margin.left - margin.right;
      const plotH = height - margin.top - margin.bottom;
      const xScale = (x) => margin.left + ((x - minX) / Math.max(1e-9, maxX - minX)) * plotW;
      const yScale = (y) => margin.top + (1 - (y - minY) / Math.max(1e-9, maxY - minY)) * plotH;

      ctx.fillStyle = "#5d6e84";
      ctx.font = "11px sans-serif";
      ctx.fillText(String(spec.x_label || "x"), width / 2 - 30, height - 8);
      ctx.save();
      ctx.translate(10, height / 2 + 20);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(String(spec.y_label || "y"), 0, 0);
      ctx.restore();

      const ticks = 4;
      for (let i = 0; i <= ticks; i++) {{
        const t = i / ticks;
        const yVal = minY + (maxY - minY) * (1 - t);
        const y = margin.top + plotH * t;
        ctx.strokeStyle = "#edf2f8";
        ctx.beginPath();
        ctx.moveTo(margin.left, y);
        ctx.lineTo(width - margin.right, y);
        ctx.stroke();
        ctx.fillStyle = "#6a7f98";
        ctx.fillText(fmt(yVal, 2), 2, y + 4);
      }}

      (spec.series || []).forEach((series, idx) => {{
        const xVals = series.x || [];
        const yVals = series.y || [];
        const color = COLORS[idx % COLORS.length];
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < Math.min(xVals.length, yVals.length); i++) {{
          const px = xScale(Number(xVals[i]));
          const py = yScale(Number(yVals[i]));
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }}
        ctx.stroke();
        for (let i = 0; i < Math.min(xVals.length, yVals.length); i++) {{
          const px = xScale(Number(xVals[i]));
          const py = yScale(Number(yVals[i]));
          ctx.beginPath();
          ctx.arc(px, py, 2.5, 0, Math.PI * 2);
          ctx.fill();
        }}
      }});
    }}

    function drawBarPlot(canvas, spec) {{
      const ctx = canvas.getContext("2d");
      const width = canvas.width;
      const height = canvas.height;
      const margin = {{ left: 44, right: 12, top: 16, bottom: 32 }};
      ctx.clearRect(0, 0, width, height);
      drawAxes(ctx, width, height, margin);

      const categories = Array.from(
        new Set((spec.series || []).flatMap((s) => (s.x || []).map((x) => Number(x))))
      ).sort((a, b) => a - b);
      const allY = (spec.series || []).flatMap((s) => (s.y || []).map((y) => Number(y)));
      if (!categories.length || !allY.length) return;

      let minY = Math.min(0, ...allY);
      let maxY = Math.max(...allY);
      if (minY === maxY) {{
        maxY += 1;
      }}

      const plotW = width - margin.left - margin.right;
      const plotH = height - margin.top - margin.bottom;
      const groupW = plotW / categories.length;
      const perSeriesW = Math.max(4, (groupW * 0.7) / Math.max(1, (spec.series || []).length));
      const yScale = (y) => margin.top + (1 - (y - minY) / Math.max(1e-9, maxY - minY)) * plotH;
      const zeroY = yScale(0);

      ctx.fillStyle = "#5d6e84";
      ctx.font = "11px sans-serif";
      ctx.fillText(String(spec.x_label || "x"), width / 2 - 30, height - 8);
      ctx.save();
      ctx.translate(10, height / 2 + 20);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(String(spec.y_label || "y"), 0, 0);
      ctx.restore();

      for (let i = 0; i < categories.length; i++) {{
        const xCenter = margin.left + groupW * i + groupW / 2;
        ctx.fillStyle = "#6a7f98";
        ctx.fillText(String(categories[i]), xCenter - 6, height - 16);
      }}

      (spec.series || []).forEach((series, sIdx) => {{
        const color = COLORS[sIdx % COLORS.length];
        const xVals = series.x || [];
        const yVals = series.y || [];
        for (let i = 0; i < Math.min(xVals.length, yVals.length); i++) {{
          const cat = Number(xVals[i]);
          const y = Number(yVals[i]);
          const cIdx = categories.indexOf(cat);
          if (cIdx < 0) continue;
          const xBase = margin.left + groupW * cIdx + groupW * 0.15 + sIdx * perSeriesW;
          const yPix = yScale(y);
          const top = Math.min(yPix, zeroY);
          const h = Math.abs(zeroY - yPix);
          ctx.fillStyle = color;
          ctx.fillRect(xBase, top, perSeriesW - 1, h);
        }}
      }});
    }}

    function renderPlotCard(spec) {{
      const card = document.createElement("div");
      card.className = "plot-card";

      const title = document.createElement("div");
      title.className = "plot-title";
      title.textContent = spec.title || spec.plot_id || "Plot";
      card.appendChild(title);

      const canvas = document.createElement("canvas");
      canvas.width = 760;
      canvas.height = 300;
      card.appendChild(canvas);

      if (spec.kind === "line") drawLinePlot(canvas, spec);
      else drawBarPlot(canvas, spec);

      const legend = document.createElement("div");
      legend.className = "legend";
      (spec.series || []).forEach((series, idx) => {{
        const chip = document.createElement("span");
        chip.className = "chip";
        const dot = document.createElement("span");
        dot.className = "dot";
        dot.style.background = COLORS[idx % COLORS.length];
        const label = document.createElement("span");
        label.textContent = series.label || `series_${{idx}}`;
        chip.appendChild(dot);
        chip.appendChild(label);
        legend.appendChild(chip);
      }});
      card.appendChild(legend);
      return card;
    }}

    function renderHeadline() {{
      document.getElementById("run_root").textContent = `run dir: ${{DASHBOARD_DATA.run_dir}}`;
      document.getElementById("generated_at").textContent =
        `generated: ${{DASHBOARD_DATA.generated_at_utc}} | runs: ${{DASHBOARD_DATA.runs.length}}`;

      const cards = document.getElementById("headline_cards");
      const summary = DASHBOARD_DATA.summary || null;
      const metrics = summary && summary.groups && summary.groups.length
        ? (summary.groups[0].metrics || {{}})
        : {{}};

      const choices = [
        [
          "Final Stage Avg Return",
          metrics.final_stage_average_return &&
            metrics.final_stage_average_return.mean,
        ],
        [
          "Average Forgetting",
          metrics.average_forgetting && metrics.average_forgetting.mean,
        ],
        [
          "Average Retention",
          metrics.average_retention && metrics.average_retention.mean,
        ],
      ];

      for (const [label, value] of choices) {{
        const card = document.createElement("div");
        card.className = "card";
        const l = document.createElement("div");
        l.className = "metric-label";
        l.textContent = label;
        const v = document.createElement("div");
        v.className = "metric-value";
        v.textContent = scalar(value) === null ? "n/a" : fmt(Number(value), 4);
        card.appendChild(l);
        card.appendChild(v);
        cards.appendChild(card);
      }}
    }}

    function renderAggregate() {{
      const container = document.getElementById("aggregate_tables");
      const summary = DASHBOARD_DATA.summary;
      if (!summary) {{
        const empty = document.createElement("div");
        empty.className = "subtitle";
        empty.textContent = "No aggregate summary found.";
        container.appendChild(empty);
        return;
      }}
      const groups = summary.groups || [];
      if (!groups.length) {{
        const empty = document.createElement("div");
        empty.className = "subtitle";
        empty.textContent = "Aggregate summary has no groups.";
        container.appendChild(empty);
        return;
      }}
      const metrics = groups[0].metrics || {{}};
      container.appendChild(buildMapTable("Aggregate Metrics", metrics));
    }}

    function renderRuns() {{
      const root = document.getElementById("runs");
      for (const run of DASHBOARD_DATA.runs) {{
        const section = document.createElement("div");
        section.className = "section";

        const title = document.createElement("div");
        title.className = "run-title";
        title.innerHTML = `<h3>${{run.run_id}}</h3><span class="mono">${{run.path}}</span>`;
        section.appendChild(title);

        const metaTable = buildMapTable("Metadata", (run.summary && run.summary.metadata) || {{}});
        const metrics = (run.summary && run.summary.metrics) || {{}};
        const tableGrid = document.createElement("div");
        tableGrid.className = "grid-2";
        tableGrid.appendChild(metaTable);
        tableGrid.appendChild(
          buildMapTable(
            "Average Return by Stage",
            metrics.average_return_by_stage || {{}}
          )
        );
        tableGrid.appendChild(
          buildMapTable("Forgetting by Task", metrics.forgetting_by_task || {{}})
        );
        tableGrid.appendChild(
          buildMapTable("Retention by Task", metrics.retention_by_task || {{}})
        );
        section.appendChild(tableGrid);

        if (run.stream_trace && Array.isArray(run.stream_trace.stages)) {{
          const traceWrap = document.createElement("div");
          traceWrap.style.marginTop = "10px";
          const traceTitle = document.createElement("h3");
          traceTitle.textContent = "Stage Trace";
          traceWrap.appendChild(traceTitle);
          const t = document.createElement("table");
          t.innerHTML = "<thead><tr><th>Stage</th><th>Task Returns</th></tr></thead>";
          const body = document.createElement("tbody");
          for (const stage of run.stream_trace.stages) {{
            const tr = document.createElement("tr");
            const a = document.createElement("td");
            a.textContent = String(stage.stage_id || "");
            const b = document.createElement("td");
            b.className = "mono";
            b.textContent = JSON.stringify(stage.task_returns || {{}});
            tr.appendChild(a);
            tr.appendChild(b);
            body.appendChild(tr);
          }}
          t.appendChild(body);
          traceWrap.appendChild(t);
          section.appendChild(traceWrap);
        }}

        if (Array.isArray(run.plot_specs) && run.plot_specs.length) {{
          const plots = document.createElement("div");
          plots.className = "plot-grid";
          for (const spec of run.plot_specs) {{
            plots.appendChild(renderPlotCard(spec));
          }}
          section.appendChild(plots);
        }}

        root.appendChild(section);
      }}
    }}

    renderHeadline();
    renderAggregate();
    renderRuns();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a self-contained HTML dashboard from a CRLBench run output "
            "directory that contains per-seed run folders and an optional summaries folder."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run root directory (example: artifacts/exp1_quadruped_recovery_dev_mlp_ppo_gpu).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path (defaults to <run-dir>/dashboard.html).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise SystemExit(f"--run-dir is not a directory: {run_dir}")

    runs = _discover_runs(run_dir)
    if not runs:
        raise SystemExit(f"No run directories with run_metrics_summary.json found under {run_dir}")

    summary = _discover_summary(run_dir / "summaries")
    output = args.output.resolve() if args.output is not None else (run_dir / "dashboard.html")

    data: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "summary": summary,
        "runs": [
            {
                "run_id": run.run_id,
                "path": run.path,
                "summary": run.summary,
                "stream_trace": run.stream_trace,
                "plot_specs": run.plot_specs,
            }
            for run in runs
        ],
    }
    output.write_text(_html_document(data), encoding="utf-8")
    print(f"Wrote dashboard: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
