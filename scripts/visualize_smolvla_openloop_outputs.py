#!/usr/bin/env python3
"""Create a self-contained HTML report from SmolVLA open-loop CSV outputs."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _action_dims(rows: list[dict[str, str]]) -> list[int]:
    dims = []
    if not rows:
        return dims
    for key in rows[0].keys():
        if key.startswith("pred_"):
            suffix = key.split("_", 1)[1]
            if suffix.isdigit() and f"target_{suffix}" in rows[0]:
                dims.append(int(suffix))
    return sorted(dims)


def _build_payload(rows: list[dict[str, str]], max_dims: int) -> dict:
    dims = _action_dims(rows)[:max_dims]
    return {
        "rows": [
            {
                "dataset_index": int(_float(row, "dataset_index")),
                "episode_index": int(_float(row, "episode_index")),
                "frame_index": int(_float(row, "frame_index")),
                "l1": _float(row, "l1"),
                "mse": _float(row, "mse"),
                "linf": _float(row, "linf"),
                "task": row.get("task", ""),
                "pred": [_float(row, f"pred_{dim}") for dim in dims],
                "target": [_float(row, f"target_{dim}") for dim in dims],
            }
            for row in rows
        ],
        "dims": dims,
    }


HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>SmolVLA Open-Loop Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; color: #172026; }
    h1 { font-size: 22px; margin: 0 0 8px; }
    .meta { color: #5b6570; margin-bottom: 18px; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 10px; margin-bottom: 18px; }
    .metric { border: 1px solid #d8dee6; border-radius: 6px; padding: 10px; }
    .metric b { display: block; font-size: 18px; margin-top: 4px; }
    .plot { border: 1px solid #d8dee6; border-radius: 6px; margin: 12px 0; padding: 10px; }
    svg { width: 100%; height: 220px; display: block; }
    table { border-collapse: collapse; width: 100%; font-size: 12px; margin-top: 18px; }
    th, td { border: 1px solid #d8dee6; padding: 5px 6px; text-align: right; }
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
  </style>
</head>
<body>
  <h1>SmolVLA Open-Loop Report</h1>
  <div class="meta">Source: __SOURCE__</div>
  <div id="app"></div>
  <script>
  const payload = __PAYLOAD__;
  function mean(xs) { return xs.length ? xs.reduce((a,b)=>a+b,0) / xs.length : 0; }
  function max(xs) { return xs.length ? xs.reduce((a,b)=>Math.max(a,b), -Infinity) : 0; }
  function fmt(x) { return Number(x).toFixed(5); }
  const rows = payload.rows;
  const app = document.getElementById("app");
  const l1s = rows.map(r => r.l1);
  const mses = rows.map(r => r.mse);
  const linfs = rows.map(r => r.linf);
  app.innerHTML += `<div class="grid">
    <div class="metric">frames<b>${rows.length}</b></div>
    <div class="metric">mean L1<b>${fmt(mean(l1s))}</b></div>
    <div class="metric">mean MSE<b>${fmt(mean(mses))}</b></div>
    <div class="metric">max Linf<b>${fmt(max(linfs))}</b></div>
  </div>`;
  function drawSeries(title, pred, target) {
    const w = 900, h = 220, pad = 28;
    const all = pred.concat(target);
    const ymin = Math.min(...all), ymax = Math.max(...all);
    const span = Math.max(1e-6, ymax - ymin);
    const x = i => pad + (w - 2 * pad) * (pred.length <= 1 ? 0 : i / (pred.length - 1));
    const y = v => h - pad - (h - 2 * pad) * ((v - ymin) / span);
    const path = arr => arr.map((v, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
    return `<div class="plot"><b>${title}</b><svg viewBox="0 0 ${w} ${h}">
      <line x1="${pad}" y1="${h-pad}" x2="${w-pad}" y2="${h-pad}" stroke="#b8c0cc"/>
      <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${h-pad}" stroke="#b8c0cc"/>
      <path d="${path(target)}" fill="none" stroke="#2563eb" stroke-width="1.5"/>
      <path d="${path(pred)}" fill="none" stroke="#dc2626" stroke-width="1.5"/>
      <text x="${pad}" y="16" fill="#2563eb">target</text>
      <text x="${pad + 80}" y="16" fill="#dc2626">pred</text>
    </svg></div>`;
  }
  payload.dims.forEach((dim, idx) => {
    app.innerHTML += drawSeries(`action dim ${dim}`, rows.map(r => r.pred[idx]), rows.map(r => r.target[idx]));
  });
  const worst = rows.slice().sort((a,b)=>b.l1-a.l1).slice(0, 50);
  app.innerHTML += `<table><thead><tr><th>episode</th><th>frame</th><th>L1</th><th>MSE</th><th>Linf</th><th>task</th></tr></thead><tbody>` +
    worst.map(r => `<tr><td>${r.episode_index}</td><td>${r.frame_index}</td><td>${fmt(r.l1)}</td><td>${fmt(r.mse)}</td><td>${fmt(r.linf)}</td><td>${r.task}</td></tr>`).join("") +
    `</tbody></table>`;
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output-html", required=True, type=Path)
    parser.add_argument("--max-dims", type=int, default=14)
    args = parser.parse_args()

    rows = _load_rows(args.csv)
    payload = _build_payload(rows, args.max_dims)
    html_text = HTML_TEMPLATE.replace("__SOURCE__", html.escape(str(args.csv))).replace(
        "__PAYLOAD__", json.dumps(payload)
    )
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(html_text)
    print(f"Wrote {args.output_html}")


if __name__ == "__main__":
    main()
