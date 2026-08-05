#!/usr/bin/env python3
"""Write a lightweight 3D HTML viewer for open-loop action trajectories and optional URDF."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _load_points(csv_path: Path, dims: tuple[int, int, int], scale: float) -> dict:
    pred = []
    target = []
    meta = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            pred.append([_float(row, f"pred_{dim}") * scale for dim in dims])
            target.append([_float(row, f"target_{dim}") * scale for dim in dims])
            meta.append(
                {
                    "episode_index": int(_float(row, "episode_index")),
                    "frame_index": int(_float(row, "frame_index")),
                    "l1": _float(row, "l1"),
                    "task": row.get("task", ""),
                }
            )
    return {"pred": pred, "target": target, "meta": meta, "dims": dims}


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>SmolVLA 3D Trajectory Viewer</title>
  <style>
    body { margin: 0; font-family: Arial, sans-serif; color: #172026; }
    #panel { position: fixed; top: 12px; left: 12px; max-width: 420px; background: rgba(255,255,255,0.92); border: 1px solid #d8dee6; border-radius: 6px; padding: 10px; z-index: 2; }
    #panel h1 { font-size: 16px; margin: 0 0 8px; }
    #panel p { margin: 4px 0; font-size: 12px; }
    #viewer { width: 100vw; height: 100vh; display: block; }
  </style>
</head>
<body>
  <div id="panel">
    <h1>SmolVLA 3D Trajectory Viewer</h1>
    <p>CSV: __CSV__</p>
    <p>URDF: __URDF__</p>
    <p>Red = prediction, Blue = target. The first three selected action dims are drawn as XYZ.</p>
  </div>
  <canvas id="viewer"></canvas>
  <script type="module">
    import * as THREE from "https://unpkg.com/three@0.160.0/build/three.module.js";
    import { OrbitControls } from "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js";
    import { URDFLoader } from "https://unpkg.com/urdf-loader@0.12.6/src/URDFLoader.js";

    const payload = __PAYLOAD__;
    const urdfText = __URDF_TEXT__;
    const canvas = document.getElementById("viewer");
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf7f8fa);
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
    camera.position.set(1.2, -1.6, 1.0);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);
    scene.add(new THREE.HemisphereLight(0xffffff, 0x555555, 2.0));
    const grid = new THREE.GridHelper(2, 20, 0x8892a0, 0xd8dee6);
    scene.add(grid);

    function resize() {
      const w = window.innerWidth, h = window.innerHeight;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    window.addEventListener("resize", resize);
    resize();

    function addTrajectory(points, color) {
      if (!points.length) return;
      const geometry = new THREE.BufferGeometry().setFromPoints(points.map(p => new THREE.Vector3(p[0], p[1], p[2])));
      const material = new THREE.LineBasicMaterial({ color });
      scene.add(new THREE.Line(geometry, material));
      const dotMaterial = new THREE.MeshBasicMaterial({ color });
      for (const p of points) {
        const dot = new THREE.Mesh(new THREE.SphereGeometry(0.008, 8, 8), dotMaterial);
        dot.position.set(p[0], p[1], p[2]);
        scene.add(dot);
      }
    }
    addTrajectory(payload.target, 0x2563eb);
    addTrajectory(payload.pred, 0xdc2626);

    if (urdfText) {
      const loader = new URDFLoader();
      const robot = loader.parse(urdfText);
      robot.rotation.x = -Math.PI / 2;
      scene.add(robot);
    }

    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    animate();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output-html", required=True, type=Path)
    parser.add_argument("--urdf", type=Path, default=None)
    parser.add_argument("--xyz-dims", default="0,1,2")
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    dims = tuple(int(x.strip()) for x in args.xyz_dims.split(","))
    if len(dims) != 3:
        raise ValueError("--xyz-dims must contain exactly three comma-separated action dimensions")
    payload = _load_points(args.csv, dims, args.scale)
    urdf_text = args.urdf.read_text() if args.urdf and args.urdf.exists() else ""

    text = (
        HTML.replace("__CSV__", html.escape(str(args.csv)))
        .replace("__URDF__", html.escape(str(args.urdf or "")))
        .replace("__PAYLOAD__", json.dumps(payload))
        .replace("__URDF_TEXT__", json.dumps(urdf_text))
    )
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(text)
    print(f"Wrote {args.output_html}")


if __name__ == "__main__":
    main()
