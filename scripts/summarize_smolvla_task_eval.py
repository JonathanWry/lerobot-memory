#!/usr/bin/env python3
"""Summarize SmolVLA MemoryBench task eval outputs with side-specific curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _dims(rows: list[dict[str, str]]) -> list[int]:
    if not rows:
        return []
    out = []
    for key in rows[0]:
        if key.startswith("abs_err_"):
            suffix = key.split("_", 2)[-1]
            if suffix.isdigit():
                out.append(int(suffix))
    return sorted(out)


def _side_dims(dims: list[int]) -> tuple[list[int], list[int]]:
    # MemoryBench/Agilex convention: first 7 dims are left arm, last 7 dims are right arm.
    left = [d for d in dims if 0 <= d < 7]
    right = [d for d in dims if 7 <= d < 14]
    return left, right


def _row_l1(row: dict[str, str], dims: list[int]) -> float:
    vals = [_float(row, f"abs_err_{d}") for d in dims if f"abs_err_{d}" in row]
    return mean(vals) if vals else 0.0


def _load_ee_errors(record_path: Path | None) -> dict[tuple[int, int], dict[str, float]]:
    # ##modified by wangruiyu to adapt memory mechanism
    # FK writer stores ee_error_by_link inside viewer event frames. Use it when
    # available; otherwise task summaries still include action-space metrics.
    if record_path is None or not record_path.exists():
        return {}
    out: dict[tuple[int, int], dict[str, float]] = {}
    for line in record_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = event.get("payload") or {}
        for frame in payload.get("frames") or []:
            ep = frame.get("episode_index")
            fr = frame.get("frame_index")
            detail = frame.get("detail") or {}
            ee = detail.get("ee_error_by_link") or {}
            if ep is None or fr is None or not ee:
                continue
            out[(int(ep), int(fr))] = {str(k): float(v) for k, v in ee.items() if v is not None}
    return out


def _ee_by_side(ee: dict[str, float]) -> tuple[float | None, float | None, float | None]:
    vals = list(ee.values())
    left = [v for k, v in ee.items() if "left" in k.lower() or k.lower().startswith("l")]
    right = [v for k, v in ee.items() if "right" in k.lower() or k.lower().startswith("r")]
    return (
        mean(vals) if vals else None,
        mean(left) if left else None,
        mean(right) if right else None,
    )


def _fmt(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x:.6f}"


def _series_svg(title: str, values: list[float | None], color: str) -> str:
    clean = [v for v in values if v is not None]
    if not clean:
        return f'<div class="plot"><b>{title}</b><p>No data</p></div>'
    w, h, pad = 960, 220, 30
    ymin = 0.0
    ymax = max(clean)
    span = max(1e-8, ymax - ymin)
    pts = []
    n = len(values)
    last = None
    for i, v in enumerate(values):
        if v is None:
            continue
        x = pad + (w - 2 * pad) * (0 if n <= 1 else i / (n - 1))
        y = h - pad - (h - 2 * pad) * ((v - ymin) / span)
        cmd = "M" if last is None else "L"
        pts.append(f"{cmd}{x:.1f},{y:.1f}")
        last = (x, y)
    return f'''<div class="plot"><b>{title}</b><svg viewBox="0 0 {w} {h}">
      <line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#b8c0cc"/>
      <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="#b8c0cc"/>
      <path d="{' '.join(pts)}" fill="none" stroke="{color}" stroke-width="1.6"/>
      <text x="{pad}" y="18" fill="{color}">max={max(clean):.5f} mean={mean(clean):.5f}</text>
    </svg></div>'''


def summarize_task(task_dir: Path, write_csv: bool = True) -> dict[str, Any]:
    rows = _load_csv(task_dir / "openloop.csv")
    dims = _dims(rows)
    left_dims, right_dims = _side_dims(dims)
    ee_map = _load_ee_errors(task_dir / "viewer_events.jsonl")
    prompt = (task_dir / "prompt.txt").read_text(encoding="utf-8").strip() if (task_dir / "prompt.txt").exists() else ""

    per_sample = []
    for i, row in enumerate(rows):
        ep = int(_float(row, "episode_index"))
        fr = int(_float(row, "frame_index"))
        ee_all, ee_left, ee_right = _ee_by_side(ee_map.get((ep, fr), {}))
        per_sample.append(
            {
                "sample": i,
                "episode_index": ep,
                "frame_index": fr,
                "l1": _row_l1(row, dims),
                "l1_left": _row_l1(row, left_dims),
                "l1_right": _row_l1(row, right_dims),
                "ee": ee_all,
                "ee_left": ee_left,
                "ee_right": ee_right,
            }
        )

    def avg(key: str) -> float | None:
        vals = [x[key] for x in per_sample if x[key] is not None]
        return mean(vals) if vals else None

    summary = {
        "task_dir": str(task_dir),
        "prompt": prompt,
        "num_samples": len(per_sample),
        "l1": avg("l1"),
        "l1_left": avg("l1_left"),
        "l1_right": avg("l1_right"),
        "ee": avg("ee"),
        "ee_left": avg("ee_left"),
        "ee_right": avg("ee_right"),
        "left_dims": left_dims,
        "right_dims": right_dims,
        "note": "left dims are expected mostly static in this dataset; l1_left measures unwanted left-arm motion/error.",
    }
    (task_dir / "task_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if write_csv:
        with (task_dir / "task_curves.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_sample[0].keys()) if per_sample else ["sample"])
            writer.writeheader()
            writer.writerows(per_sample)

    html = """<!doctype html><html><head><meta charset="utf-8"><title>Task Curves</title>
<style>body{font-family:Arial,sans-serif;margin:24px;color:#172026}.grid{display:grid;grid-template-columns:repeat(6,minmax(100px,1fr));gap:8px}.metric{border:1px solid #d8dee6;border-radius:6px;padding:8px}.metric b{display:block;font-size:18px}.plot{border:1px solid #d8dee6;border-radius:6px;margin:12px 0;padding:10px}svg{width:100%;height:220px;display:block}table{border-collapse:collapse;width:100%;font-size:12px}td,th{border:1px solid #d8dee6;padding:5px;text-align:right}td:nth-child(2),th:nth-child(2){text-align:left}</style>
</head><body>"""
    html += f"<h1>{task_dir.name}</h1><p>{prompt}</p><div class='grid'>"
    for key in ["l1", "l1_left", "l1_right", "ee", "ee_left", "ee_right"]:
        html += f"<div class='metric'>{key}<b>{_fmt(summary[key])}</b></div>"
    html += "</div>"
    html += _series_svg("L1 vs sample", [x["l1"] for x in per_sample], "#111827")
    html += _series_svg("L1 left vs sample", [x["l1_left"] for x in per_sample], "#2563eb")
    html += _series_svg("L1 right vs sample", [x["l1_right"] for x in per_sample], "#dc2626")
    html += _series_svg("EE vs sample", [x["ee"] for x in per_sample], "#059669")
    html += _series_svg("EE left vs sample", [x["ee_left"] for x in per_sample], "#0284c7")
    html += _series_svg("EE right vs sample", [x["ee_right"] for x in per_sample], "#ea580c")
    html += "<table><thead><tr><th>sample</th><th>frame</th><th>l1</th><th>l1_left</th><th>l1_right</th><th>ee</th><th>ee_left</th><th>ee_right</th></tr></thead><tbody>"
    for x in per_sample[:500]:
        html += f"<tr><td>{x['sample']}</td><td>{x['episode_index']}:{x['frame_index']}</td><td>{_fmt(x['l1'])}</td><td>{_fmt(x['l1_left'])}</td><td>{_fmt(x['l1_right'])}</td><td>{_fmt(x['ee'])}</td><td>{_fmt(x['ee_left'])}</td><td>{_fmt(x['ee_right'])}</td></tr>"
    html += "</tbody></table></body></html>"
    (task_dir / "task_curves.html").write_text(html, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True, type=Path)
    args = parser.parse_args()

    task_dirs = sorted(p for p in args.eval_root.glob("task_*") if (p / "openloop.csv").exists())
    summaries = [summarize_task(task_dir) for task_dir in task_dirs]
    out_json = args.eval_root / "task_summaries.json"
    out_json.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.eval_root / "task_summaries.csv").open("w", newline="", encoding="utf-8") as f:
        keys = ["task_dir", "prompt", "num_samples", "l1", "l1_left", "l1_right", "ee", "ee_left", "ee_right"]
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary.get(key) for key in keys})
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
