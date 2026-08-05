#!/usr/bin/env python3
"""Build Kai0-compatible Agilex FK viewer records from SmolVLA open-loop outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.model.kinematics import RobotKinematics

try:
    from scripts.viz.rerun_dataset_viz import URDFKinematics
    from scripts.viz.rerun_dataset_viz import _joint_map_from_state as _joint_map_from_state_rerun
except Exception:  # pragma: no cover
    URDFKinematics = None
    _joint_map_from_state_rerun = None


DEFAULT_KAI0_VIZ_DIR = Path("/pfs/user/wangruiyu/kai0/scripts/viz")
DEFAULT_STATIC_DIR = DEFAULT_KAI0_VIZ_DIR / "infer_viewer"
DEFAULT_URDF_PATH = DEFAULT_STATIC_DIR / "ee_assets" / "robot.urdf"
DEFAULT_URDF_ROOT = DEFAULT_STATIC_DIR / "ee_assets"


def _append_jsonl_event(path: Path, event: str, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": event, "recorded_at": time.time(), "payload": payload}) + "\n")


def _infer_side_from_name(name: str | None) -> str | None:
    if not name:
        return None
    lower = name.lower()
    if lower.startswith(("left", "l_", "fl_")) or "left" in lower:
        return "left"
    if lower.startswith(("right", "r_", "fr_")) or "right" in lower:
        return "right"
    return None


def _parse_ee_link_spec(spec: str) -> tuple[str, str, str | None]:
    text = spec.strip()
    if "=" in text:
        label, link_name = text.split("=", 1)
    elif ":" in text:
        label, link_name = text.split(":", 1)
    else:
        label, link_name = text, text
    label = label.strip()
    link_name = link_name.strip()
    return label or link_name, link_name, _infer_side_from_name(label) or _infer_side_from_name(link_name)


def _resolve_urdf_resource(resource: str, urdf_path: Path, urdf_root: Path | None) -> Path | None:
    if resource.startswith("package://"):
        parts = resource[len("package://") :].split("/", 1)
        if len(parts) != 2:
            return None
        package, rel_path = parts
        candidates = [parent / package / rel_path for parent in urdf_path.parents]
        if urdf_root is not None:
            candidates.append(urdf_root / package / rel_path)
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return None
    if resource.startswith("file://"):
        candidate = Path(resource[len("file://") :]).expanduser()
        return candidate.resolve() if candidate.exists() else None
    candidate = Path(resource)
    if not candidate.is_absolute():
        candidate = (urdf_path.parent / candidate).resolve()
    return candidate if candidate.exists() else None


def _rewrite_urdf_for_fk(urdf_path: Path, urdf_root: Path | None, out_dir: Path) -> Path:
    text = urdf_path.read_text(encoding="utf-8")
    rewritten = text
    for match in set(re.findall(r'filename="([^"]+)"', text)):
        resolved = _resolve_urdf_resource(match, urdf_path, urdf_root)
        if resolved is not None:
            rewritten = rewritten.replace(match, str(resolved))
    out_path = out_dir / "robot_fk.urdf"
    out_path.write_text(rewritten, encoding="utf-8")
    return out_path


def _prepare_viewer_urdf_assets(urdf_path: Path, urdf_root: Path | None, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meshes_dir = out_dir / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)
    urdf_text = urdf_path.read_text(encoding="utf-8")
    rewritten = urdf_text
    for match in set(re.findall(r'filename="([^"]+)"', urdf_text)):
        resolved = _resolve_urdf_resource(match, urdf_path, urdf_root)
        mesh_ref = f"meshes/{Path(match).name}"
        rewritten = rewritten.replace(match, mesh_ref)
        if resolved is not None and resolved.exists():
            dst = meshes_dir / Path(mesh_ref).name
            if not dst.exists():
                shutil.copy2(resolved, dst)
    (out_dir / "robot.urdf").write_text(rewritten, encoding="utf-8")


def _build_urdf_to_dataset_map(urdf_joint_names: list[str], side: str | None) -> list[int | None]:
    mapping: list[int | None] = [None] * len(urdf_joint_names)
    if side == "left":
        pairs = [(f"fl_joint{i}", i - 1) for i in range(1, 8)] + [(f"left_joint{i}", i - 1) for i in range(1, 8)]
    elif side == "right":
        pairs = [(f"fr_joint{i}", 7 + i - 1) for i in range(1, 8)] + [(f"right_joint{i}", 7 + i - 1) for i in range(1, 8)]
    else:
        pairs = (
            [(f"fl_joint{i}", i - 1) for i in range(1, 8)]
            + [(f"left_joint{i}", i - 1) for i in range(1, 8)]
            + [(f"fr_joint{i}", 7 + i - 1) for i in range(1, 8)]
            + [(f"right_joint{i}", 7 + i - 1) for i in range(1, 8)]
        )
    for urdf_name, data_idx in pairs:
        if urdf_name in urdf_joint_names:
            mapping[urdf_joint_names.index(urdf_name)] = data_idx
    return mapping


def _vector_to_fk_point(
    vec: np.ndarray,
    solver: Any,
    urdf_to_dataset: list[int | None],
    joint_unit: str,
) -> list[float] | None:
    if isinstance(solver, tuple) and len(solver) == 2:
        if _joint_map_from_state_rerun is None:
            return None
        kin, link_name = solver
        joint_map = _joint_map_from_state_rerun(
            np.asarray(vec, dtype=np.float64),
            kin.joint_names,
            {name: idx for name, idx in zip(kin.joint_names, urdf_to_dataset, strict=True) if idx is not None},
            add_symmetric_gripper=True,
        )
        point = kin.compute_link_position(joint_map, link_name)
        return point.tolist() if point is not None else None

    full = np.zeros(len(urdf_to_dataset), dtype=float)
    for urdf_idx, data_idx in enumerate(urdf_to_dataset):
        if data_idx is None or data_idx >= len(vec):
            continue
        full[urdf_idx] = float(vec[data_idx])
    if joint_unit == "rad":
        full = np.rad2deg(full)
    fk = solver.forward_kinematics(full)
    return fk[:3, 3].tolist()


def _init_fk(urdf_path: Path, urdf_root: Path | None, ee_links: list[str]):
    tmp = tempfile.TemporaryDirectory(prefix="smolvla_agilex_fk_")
    fk_urdf = _rewrite_urdf_for_fk(urdf_path, urdf_root, Path(tmp.name))
    solvers: dict[str, Any] = {}
    maps: dict[str, list[int | None]] = {}
    fallback_kin = None
    for spec in ee_links:
        label, link_name, side = _parse_ee_link_spec(spec)
        try:
            solver = RobotKinematics(str(fk_urdf), target_frame_name=link_name, joint_names=None)
            joint_names = solver.joint_names
        except Exception:
            if URDFKinematics is None:
                raise
            if fallback_kin is None:
                fallback_kin = URDFKinematics(fk_urdf)
            solver = (fallback_kin, link_name)
            joint_names = fallback_kin.joint_names
            if not fallback_kin.has_link(link_name):
                raise ValueError(f"Link {link_name} not found in URDF")
        mapping = _build_urdf_to_dataset_map(joint_names, side)
        if not any(idx is not None for idx in mapping):
            raise ValueError(f"FK mapping is empty for {label}={link_name}; URDF joints={joint_names}")
        solvers[label] = solver
        maps[label] = mapping
    return solvers, maps, tmp


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mean_errors_by_side(errors: dict[str, float]) -> tuple[float | None, float | None]:
    left = [v for k, v in errors.items() if _infer_side_from_name(k) == "left"]
    right = [v for k, v in errors.items() if _infer_side_from_name(k) == "right"]
    return (float(np.mean(left)) if left else None, float(np.mean(right)) if right else None)


def _write_index(out_dir: Path, record_path: Path, static_dir: Path, urdf_path: Path, port: int) -> None:
    text = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SmolVLA Agilex FK Viewer</title></head>
<body style="font-family:Arial,sans-serif;margin:24px;color:#172026">
<h1>SmolVLA Agilex FK Viewer</h1>
<p>Record: <code>{record_path}</code></p>
<p>URDF: <code>{urdf_path}</code></p>
<h2>Run viewer</h2>
<pre>cd /pfs/user/wangruiyu/kai0
python scripts/viz/infer_viewer_server.py --host 0.0.0.0 --port {port} --static-dir {static_dir} --urdf-path {urdf_path}</pre>
<h2>Replay</h2>
<pre>cd /pfs/user/wangruiyu/kai0
python scripts/viz/infer_viewer_replay.py --input {record_path} --viewer-url http://127.0.0.1:{port} --reset-first</pre>
<h2>Open</h2>
<pre>http://127.0.0.1:{port}/?mode=live</pre>
</body></html>
"""
    (out_dir / "viewer_index.html").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--npz", required=True, type=Path)
    parser.add_argument("--output-record", required=True, type=Path)
    parser.add_argument("--policy-path", default="")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--urdf-path", type=Path, default=DEFAULT_URDF_PATH)
    parser.add_argument("--urdf-root", type=Path, default=DEFAULT_URDF_ROOT)
    parser.add_argument("--static-dir", type=Path, default=DEFAULT_STATIC_DIR)
    parser.add_argument("--ee-links", default="left=left_tcp,right=right_tcp")
    parser.add_argument("--joint-unit", choices=("rad", "deg"), default="rad")
    parser.add_argument("--viewer-port", type=int, default=9001)
    parser.add_argument("--copy-urdf-assets", action="store_true")
    args = parser.parse_args()

    rows = _load_csv_rows(args.csv)
    arrays = np.load(args.npz)
    pred_arr = np.asarray(arrays["pred"], dtype=np.float32)
    target_arr = np.asarray(arrays["target"], dtype=np.float32)
    state_arr = np.asarray(arrays["state"], dtype=np.float32)
    if len(rows) != len(pred_arr) or len(rows) != len(target_arr):
        raise ValueError(f"CSV/NPZ length mismatch: rows={len(rows)} pred={len(pred_arr)} target={len(target_arr)}")

    ee_links = [x.strip() for x in args.ee_links.split(",") if x.strip()]
    solvers, maps, fk_tmp = _init_fk(args.urdf_path.expanduser().resolve(), args.urdf_root.expanduser().resolve(), ee_links)

    out_record = args.output_record.expanduser().resolve()
    if out_record.exists():
        out_record.unlink()
    out_dir = out_record.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_urdf_assets:
        _prepare_viewer_urdf_assets(args.urdf_path.expanduser().resolve(), args.urdf_root.expanduser().resolve(), out_dir / "viewer_assets")

    _append_jsonl_event(
        out_record,
        "session_start",
        {"dataset_repo_id": args.dataset_root, "policy_source": args.policy_path},
    )
    _append_jsonl_event(
        out_record,
        "push",
        {
            "meta": {
                "task_name": args.dataset_root or str(args.csv),
                "ckpt_path": args.policy_path,
                "num_frames": len(rows),
                "mode": "smolvla_openloop_agilex_fk",
                "camera_keys": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
            },
            "timestamp": time.time(),
        },
    )

    trajectory_pred: dict[str, list[list[float]]] = {}
    trajectory_gt: dict[str, list[list[float]]] = {}
    for i, row in enumerate(rows):
        pred = pred_arr[i]
        target = target_arr[i]
        ee_pred_points: dict[str, list[float]] = {}
        ee_target_points: dict[str, list[float]] = {}
        ee_errors: dict[str, float] = {}
        for label, solver in solvers.items():
            pred_point = _vector_to_fk_point(pred, solver, maps[label], args.joint_unit)
            target_point = _vector_to_fk_point(target, solver, maps[label], args.joint_unit)
            if pred_point is not None:
                ee_pred_points[label] = pred_point
                trajectory_pred.setdefault(label, []).append(pred_point)
            if target_point is not None:
                ee_target_points[label] = target_point
                trajectory_gt.setdefault(label, []).append(target_point)
            if pred_point is not None and target_point is not None:
                ee_errors[label] = float(np.linalg.norm(np.asarray(pred_point) - np.asarray(target_point)))
        ee_left, ee_right = _mean_errors_by_side(ee_errors)
        loss_ee = float(np.mean(list(ee_errors.values()))) if ee_errors else None
        frame = {
            "global_index": int(_float(row, "dataset_index", i)),
            "episode_index": int(_float(row, "episode_index", -1)),
            "frame_index": int(_float(row, "frame_index", i)),
            "timestamp": float(_float(row, "frame_index", i)) / 30.0,
            "loss_l1": _float(row, "l1"),
            "loss_l1_global": _float(row, "l1"),
            "loss_ee": loss_ee,
            "ee_error_mean": loss_ee,
            "ee_error_left": ee_left,
            "ee_error_right": ee_right,
            "prompt": row.get("task", ""),
            "detail": {
                "qpos": state_arr[i].astype(float).tolist(),
                "action_pred": pred.astype(float).tolist(),
                "action_target": target.astype(float).tolist(),
                "ee_error_by_link": ee_errors,
                "ee_pred_points": ee_pred_points,
                "ee_target_points": ee_target_points,
            },
        }
        _append_jsonl_event(
            out_record,
            "push",
            {
                "frames": [frame],
                "trajectory_pred": trajectory_pred,
                "trajectory_gt": trajectory_gt,
                "timestamp": time.time(),
            },
        )

    _write_index(out_dir, out_record, args.static_dir, args.urdf_path, args.viewer_port)
    fk_tmp.cleanup()
    print(json.dumps({"record": str(out_record), "frames": len(rows), "viewer_index": str(out_dir / "viewer_index.html")}, indent=2))


if __name__ == "__main__":
    main()
