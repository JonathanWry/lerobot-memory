#!/usr/bin/env python3
"""Offline open-loop evaluation for SmolVLA on MemoryBench/LeRobot v2.1 data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_v21_dataset import LeRobotV21Dataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors


def parse_episodes(raw: str | None) -> list[int] | None:
    if raw is None or raw.strip() == "":
        return None
    text = raw.strip()
    if text.startswith("["):
        return [int(x) for x in json.loads(text)]
    values: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = [int(x) for x in token.split("-", 1)]
            values.extend(range(start, end + 1))
        else:
            values.append(int(token))
    return sorted(set(values))


def batchify(item: dict[str, Any], device: torch.device) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    for key, value in item.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0).to(device)
        elif isinstance(value, str):
            batch[key] = [value]
        else:
            batch[key] = value
    return batch


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().to("cpu", dtype=torch.float32).numpy()


def infer_action_chunk(policy, preprocessor, postprocessor, item: dict[str, Any], device: torch.device) -> torch.Tensor:
    batch = batchify(item, device)
    batch = preprocessor(batch)
    with torch.inference_mode():
        pred = policy.predict_action_chunk(batch)
    pred = postprocessor(pred)
    if pred.ndim == 2:
        pred = pred[:, None, :]
    return pred[0]


def episode_limited_indices(dataset: LeRobotV21Dataset, max_frames_per_episode: int, stride: int) -> list[int]:
    indices: list[int] = []
    per_episode_count: dict[int, int] = {}
    for idx in range(len(dataset)):
        item = dataset.hf_dataset[idx]
        episode_idx = int(item["episode_index"])
        frame_idx = int(item["frame_index"])
        if frame_idx % stride != 0:
            continue
        count = per_episode_count.get(episode_idx, 0)
        if max_frames_per_episode > 0 and count >= max_frames_per_episode:
            continue
        per_episode_count[episode_idx] = count + 1
        indices.append(idx)
    return indices


def write_action_plot(path: Path, rows: list[dict[str, Any]], action_dim: int) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        return
    xs = np.arange(len(rows))
    fig_loss, ax_loss = plt.subplots(figsize=(12, 4))
    ax_loss.plot(xs, [row["l1"] for row in rows], label="l1", linewidth=1.3)
    ax_loss.plot(xs, [row["l1_left"] for row in rows], label="l1_left", linewidth=1.0)
    ax_loss.plot(xs, [row["l1_right"] for row in rows], label="l1_right", linewidth=1.0)
    if all(row.get("ee") is not None for row in rows):
        ax_loss.plot(xs, [row["ee"] for row in rows], label="ee", linewidth=1.0)
    ax_loss.set_xlabel("sample")
    ax_loss.set_ylabel("loss")
    ax_loss.grid(True, alpha=0.2)
    ax_loss.legend()
    fig_loss.tight_layout()
    loss_path = path.with_name(f"{path.stem}_loss{path.suffix}")
    loss_path.parent.mkdir(parents=True, exist_ok=True)
    fig_loss.savefig(loss_path, dpi=150)
    plt.close(fig_loss)

    dims = min(action_dim, 14)
    cols = 2
    plot_rows = math.ceil(dims / cols)
    fig, axes = plt.subplots(plot_rows, cols, figsize=(12, max(3, plot_rows * 2.2)), squeeze=False)
    for dim in range(dims):
        ax = axes[dim // cols][dim % cols]
        ax.plot(xs, [row[f"pred_{dim}"] for row in rows], label="pred", linewidth=1)
        ax.plot(xs, [row[f"target_{dim}"] for row in rows], label="target", linewidth=1, alpha=0.8)
        ax.set_title(f"action dim {dim}")
        ax.grid(True, alpha=0.2)
    for dim in range(dims, plot_rows * cols):
        axes[dim // cols][dim % cols].axis("off")
    axes[0][0].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_web_record(path: Path, rows: list[dict[str, Any]], states: list[np.ndarray], preds: list[np.ndarray], targets: list[np.ndarray], policy_path: str, dataset_root: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_pred = {"action": []}
    trajectory_gt = {"action": []}
    with path.open("w") as f:
        f.write(
            json.dumps(
                {
                    "event": "session_start",
                    "recorded_at": time.time(),
                    "payload": {"dataset_repo_id": dataset_root, "policy_source": policy_path},
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "event": "push",
                    "recorded_at": time.time(),
                    "payload": {
                        "meta": {
                            "task_name": dataset_root,
                            "ckpt_path": policy_path,
                            "num_frames": len(rows),
                            "mode": "smolvla_openloop_recorded",
                        },
                        "timestamp": time.time(),
                    },
                }
            )
            + "\n"
        )
        for global_idx, row in enumerate(rows):
            pred = preds[global_idx].astype(float).tolist()
            target = targets[global_idx].astype(float).tolist()
            trajectory_pred["action"].append(pred)
            trajectory_gt["action"].append(target)
            frame = {
                "global_index": global_idx,
                "episode_index": row["episode_index"],
                "frame_index": row["frame_index"],
                "timestamp": float(row["frame_index"]) / 30.0,
                "loss_l1": row["l1"],
                "loss_l1_global": row["l1"],
                "loss_ee": None,
                "prompt": row.get("task", ""),
                "detail": {
                    "qpos": states[global_idx].astype(float).tolist(),
                    "action_pred": pred,
                    "action_target": target,
                    "ee_error_by_link": {},
                    "ee_pred_points": {},
                    "ee_target_points": {},
                },
            }
            f.write(
                json.dumps(
                    {
                        "event": "push",
                        "recorded_at": time.time(),
                        "payload": {
                            "frames": [frame],
                            "trajectory_pred": trajectory_pred,
                            "trajectory_gt": trajectory_gt,
                            "timestamp": time.time(),
                        },
                    }
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-id", default="memorybench")
    parser.add_argument("--episodes", default="0")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames-per-episode", type=int, default=0)
    parser.add_argument("--target-action-index", type=int, default=0)
    parser.add_argument("--action-key", default="real_action")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-csv", type=Path, required=True)
    parser.add_argument("--save-summary", type=Path, required=True)
    parser.add_argument("--save-plot", type=Path, default=None)
    parser.add_argument("--save-npz", type=Path, default=None)
    parser.add_argument("--web-record-path", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    policy_path = Path(args.policy_path)
    cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    cfg.device = str(device)
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(str(policy_path), config=cfg)
    policy.to(device)
    policy.eval()
    if hasattr(policy, "reset"):
        policy.reset()

    preprocessor, postprocessor = make_pre_post_processors(cfg, pretrained_path=str(policy_path))

    episodes = parse_episodes(args.episodes)
    dataset = LeRobotV21Dataset(
        repo_id=args.repo_id,
        root=args.dataset_root,
        episodes=episodes,
        video_backend="pyav",
        use_imagenet_stats=False,
    )
    indices = episode_limited_indices(dataset, args.max_frames_per_episode, args.stride)
    if args.num_samples > 0:
        indices = indices[: args.num_samples]
    if not indices:
        raise ValueError("No evaluation samples selected.")

    rows: list[dict[str, Any]] = []
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    states: list[np.ndarray] = []
    for idx in tqdm(indices, desc="openloop"):
        item = dataset[idx]
        pred_chunk = infer_action_chunk(policy, preprocessor, postprocessor, item, device)
        action_idx = min(args.target_action_index, pred_chunk.shape[0] - 1)
        pred = pred_chunk[action_idx]
        target = item[args.action_key].to(dtype=torch.float32)
        pred_np = to_numpy(pred)
        target_np = to_numpy(target)
        state_np = to_numpy(item["observation.qpos"])
        dim = min(pred_np.shape[-1], target_np.shape[-1])
        err = pred_np[:dim] - target_np[:dim]
        abs_err = np.abs(err)
        left_dim = min(7, dim)
        right_start = 7
        right_end = min(14, dim)
        l1_left = float(np.mean(abs_err[:left_dim])) if left_dim > 0 else float("nan")
        l1_right = float(np.mean(abs_err[right_start:right_end])) if right_end > right_start else float("nan")
        row: dict[str, Any] = {
            "dataset_index": int(idx),
            "episode_index": int(item["episode_index"].item()),
            "frame_index": int(item["frame_index"].item()),
            "target_action_index": int(action_idx),
            "l1": float(np.mean(abs_err)),
            "l1_left": l1_left,
            "l1_right": l1_right,
            "ee": None,
            "mse": float(np.mean(err**2)),
            "linf": float(np.max(abs_err)),
            "task": item.get("task", ""),
        }
        for i in range(dim):
            row[f"pred_{i}"] = float(pred_np[i])
            row[f"target_{i}"] = float(target_np[i])
            row[f"abs_err_{i}"] = float(abs(err[i]))
        rows.append(row)
        preds.append(pred_np[:dim])
        targets.append(target_np[:dim])
        states.append(state_np)

    args.save_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.save_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    pred_arr = np.stack(preds, axis=0)
    target_arr = np.stack(targets, axis=0)
    err_arr = pred_arr - target_arr
    abs_err_arr = np.abs(err_arr)
    left_end = min(7, err_arr.shape[1])
    right_start = 7
    right_end = min(14, err_arr.shape[1])
    task_summary: dict[str, dict[str, Any]] = {}
    for task in sorted({str(row.get("task", "")) for row in rows}):
        task_rows = [row for row in rows if str(row.get("task", "")) == task]
        task_summary[task] = {
            "num_samples": len(task_rows),
            "mean_l1": float(np.mean([row["l1"] for row in task_rows])),
            "mean_l1_left": float(np.mean([row["l1_left"] for row in task_rows])),
            "mean_l1_right": float(np.mean([row["l1_right"] for row in task_rows])),
        }
    summary = {
        "policy_path": str(policy_path),
        "dataset_root": args.dataset_root,
        "episodes": episodes,
        "num_samples": len(rows),
        "target_action_index": args.target_action_index,
        "mean_l1": float(np.mean(abs_err_arr)),
        "mean_l1_left": float(np.mean(abs_err_arr[:, :left_end])) if left_end > 0 else None,
        "mean_l1_right": float(np.mean(abs_err_arr[:, right_start:right_end])) if right_end > right_start else None,
        "mean_ee": None,
        "mean_mse": float(np.mean(err_arr**2)),
        "mean_linf": float(np.mean(np.max(abs_err_arr, axis=1))),
        "per_dim_l1": np.mean(abs_err_arr, axis=0).tolist(),
        "per_dim_mse": np.mean(err_arr**2, axis=0).tolist(),
        "task_summary": task_summary,
    }
    args.save_summary.parent.mkdir(parents=True, exist_ok=True)
    args.save_summary.write_text(json.dumps(summary, indent=2))

    if args.save_plot is not None:
        write_action_plot(args.save_plot, rows, pred_arr.shape[-1])
    if args.save_npz is not None:
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save_npz,
            pred=pred_arr,
            target=target_arr,
            state=np.stack(states, axis=0),
            err=err_arr,
            dataset_indices=np.asarray(indices, dtype=np.int64),
        )
    if args.web_record_path is not None:
        write_web_record(args.web_record_path, rows, states, preds, targets, str(policy_path), args.dataset_root)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
