#!/usr/bin/env python

"""Build SmolVLA perceptual memory sidecar features for LeRobot v2.1 datasets.

RoboMME-style one-shot build: each frame stores 8x8 / 4x4 / 2x2 grids plus
flat `image_emb`/`pos_emb` aliases for framesamp training. Per-episode
`kept_indices.json` is written for tokendrop (pixel-diff heap, same idea as
RoboMME / SmolVLA online buffer).

The output is intentionally separate from the LeRobot parquet/video dataset:
training can enable it with `--dataset.memory_feature_root=<output_dir>`.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_v21_dataset import (
    STATE_SOURCE_KEY,
    STATE_TARGET_KEY,
    LeRobotV21Dataset,
)
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


# ##modified by wangruiyu to adapt memory mechanism
def _parse_episodes(value: str | None) -> list[int] | None:
    if value is None or value.strip().lower() in {"", "none", "all"}:
        return None
    return [int(part) for part in value.replace("[", "").replace("]", "").split(",") if part.strip()]


def _parse_grids(value: str) -> list[int]:
    grids = [int(part) for part in value.replace("[", "").replace("]", "").split(",") if part.strip()]
    if not grids:
        raise ValueError("At least one grid size is required")
    for grid in grids:
        if grid < 1:
            raise ValueError(f"Invalid grid size: {grid}")
    return grids


def _to_float_image(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.uint8:
        return tensor.to(torch.float32) / 255.0
    tensor = tensor.to(torch.float32)
    if tensor.max() > 2:
        tensor = tensor / 255.0
    return tensor


def _pool_tokens_to_grid(tokens: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Pool [B, N, D] square-ish visual tokens to [B, grid_size**2, D]."""
    bsize, token_count, dim = tokens.shape
    side = int(math.sqrt(token_count))
    if side * side != token_count:
        needed = grid_size * grid_size
        if token_count < needed:
            raise ValueError(f"Cannot pool {token_count} tokens to {grid_size}x{grid_size}")
        return tokens[:, :needed]
    if side == grid_size:
        return tokens
    x = tokens.transpose(1, 2).reshape(bsize, dim, side, side)
    x = F.adaptive_avg_pool2d(x, (grid_size, grid_size))
    return x.flatten(2).transpose(1, 2)


def _pool_image_patches(image: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Pool prepared image [C,H,W] or [B,C,H,W] to [V, P, C] patch vectors."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"Expected image as C,H,W or B,C,H,W, got {tuple(image.shape)}")
    patches = F.adaptive_avg_pool2d(image.to(dtype=torch.float32), (grid_size, grid_size))
    return patches.flatten(2).transpose(1, 2).cpu()


def _axis_sincos(value: float, dim: int) -> np.ndarray:
    if dim == 0:
        return np.zeros((0,), dtype=np.float32)
    half = dim // 2
    if half == 0:
        return np.asarray([value], dtype=np.float32)
    omega = np.arange(half, dtype=np.float32) / max(half, 1)
    omega = 1.0 / (10000**omega)
    out = value * omega
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=0)
    if emb.shape[0] < dim:
        emb = np.pad(emb, (0, dim - emb.shape[0]))
    return emb.astype(np.float32)


def make_memory_pos_emb(frame_idx: int, num_cameras: int, grid_size: int, dim: int) -> np.ndarray:
    """Create deterministic frame/camera/spatial position embeddings.

    Returns flat [num_cameras * grid_size**2, dim], matching the online buffer.
    """
    axis_dims = [dim // 4] * 4
    axis_dims[-1] += dim - sum(axis_dims)
    rows = []
    for cam_idx in range(num_cameras):
        for y in range(grid_size):
            for x in range(grid_size):
                rows.append(
                    np.concatenate(
                        [
                            _axis_sincos(float(frame_idx), axis_dims[0]),
                            _axis_sincos(float(cam_idx), axis_dims[1]),
                            _axis_sincos(float(y), axis_dims[2]),
                            _axis_sincos(float(x), axis_dims[3]),
                        ],
                        axis=0,
                    )
                )
    return np.stack(rows, axis=0).astype(np.float32)


def _normalize_state(state: torch.Tensor, ds: LeRobotV21Dataset, key: str) -> torch.Tensor:
    stats = ds.meta.stats.get(key)
    if not stats:
        return state
    mean = torch.as_tensor(stats["mean"], dtype=state.dtype, device=state.device)
    std = torch.as_tensor(stats["std"], dtype=state.dtype, device=state.device).clamp_min(1e-6)
    return (state - mean) / std


def _batch_items(items: list[dict], keys: list[str]) -> dict[str, torch.Tensor]:
    batch = {}
    for key in keys:
        batch[key] = torch.stack([_to_float_image(item[key]) for item in items], dim=0)
    return batch


class TokenDropScorer:
    """Episode-local RoboMME / SmolVLA pixel-diff token-drop heap (8x8 patches)."""

    def __init__(self, stride: int, keptsize: int, grid_size: int = 8):
        self.stride = stride
        self.keptsize = keptsize
        self.grid_size = grid_size
        self.token_per_image = grid_size * grid_size
        self.scored_token_heap: list[tuple[float, int, int, int]] = []
        self.token_drop_last_frame = -1
        self.prev_pixels: torch.Tensor | None = None
        self.last_frame_pixels: dict[int, torch.Tensor] = {}

    def reset(self) -> None:
        self.scored_token_heap.clear()
        self.token_drop_last_frame = -1
        self.prev_pixels = None
        self.last_frame_pixels.clear()

    def update(self, frame_idx: int, image_pixels: torch.Tensor) -> None:
        # image_pixels: [V, P, C]
        self.last_frame_pixels[frame_idx] = image_pixels
        if frame_idx == 0:
            view_count = image_pixels.shape[0]
            for patch_idx in range(self.token_per_image):
                for view_idx in range(view_count):
                    heapq.heappush(self.scored_token_heap, (1000.0, frame_idx, view_idx, patch_idx))

        if frame_idx != self.token_drop_last_frame + self.stride:
            return
        prev_idx = max(0, self.token_drop_last_frame)
        prev_img = self.last_frame_pixels.get(prev_idx)
        if prev_img is None:
            return
        diff = (prev_img - image_pixels).abs().mean(dim=-1)
        for view_idx in range(diff.shape[0]):
            for patch_idx in range(diff.shape[1]):
                score = float(diff[view_idx, patch_idx].item())
                if score < 1e-4:
                    continue
                heapq.heappush(self.scored_token_heap, (score, frame_idx, view_idx, patch_idx))
                if len(self.scored_token_heap) > self.keptsize:
                    heapq.heappop(self.scored_token_heap)
        self.token_drop_last_frame += self.stride

    def get_kept_indices(self) -> list[list[int]]:
        selected = list(self.scored_token_heap)
        # Stable dump matching RoboMME: pop by score ascending.
        heap = list(selected)
        heapq.heapify(heap)
        kept: list[list[int]] = []
        while heap:
            _, frame_idx, view_idx, patch_idx = heapq.heappop(heap)
            kept.append([int(frame_idx), int(view_idx), int(patch_idx)])
        return kept


def _reshape_pos(flat_pos: np.ndarray, num_cameras: int, grid_size: int) -> np.ndarray:
    tokens = grid_size * grid_size
    return flat_pos.reshape(num_cameras, tokens, -1)


def _flush_kept_indices(scorer: TokenDropScorer, episode_dir: Path | None, write: bool) -> None:
    if not write or episode_dir is None:
        scorer.reset()
        return
    episode_dir.mkdir(parents=True, exist_ok=True)
    out_path = episode_dir / "kept_indices.json"
    with out_path.open("w") as f:
        json.dump(scorer.get_kept_indices(), f)
    scorer.reset()
# ##modified by wangruiyu to adapt memory mechanism


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--repo-id", default="memorybench")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--episodes", default=None, help="Comma list, e.g. 0,1,2. Default: all.")
    parser.add_argument("--vlm-model-name", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--grids",
        default="8,4,2",
        help="Comma list of square grid sizes to store (RoboMME-style multi-grid).",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=None,
        help="Deprecated single-grid mode. Prefer --grids. If set, overrides --grids.",
    )
    parser.add_argument(
        "--framesamp-alias-grid",
        type=int,
        default=4,
        help="Also write flat image_emb/pos_emb/mask from this grid for framesamp training.",
    )
    parser.add_argument("--pos-dim", type=int, default=256)
    parser.add_argument("--cameras", default=None, help="Comma list. Default: dataset camera keys.")
    parser.add_argument("--state-key", default=STATE_TARGET_KEY)
    parser.add_argument("--normalize-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write-kept-indices", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--token-drop-stride", type=int, default=8)
    parser.add_argument("--token-drop-keptsize", type=int, default=2048)
    parser.add_argument("--token-drop-grid", type=int, default=8)
    args = parser.parse_args()

    if args.grid_size is not None:
        grids = [int(args.grid_size)]
    else:
        grids = _parse_grids(args.grids)
    grids = sorted(set(grids), reverse=True)
    if args.framesamp_alias_grid not in grids:
        grids.append(args.framesamp_alias_grid)
        grids = sorted(set(grids), reverse=True)
    if args.write_kept_indices and args.token_drop_grid not in grids:
        raise ValueError(
            f"token-drop grid {args.token_drop_grid} must be included in --grids={grids} "
            "so image_emb_8x8 (or matching) exists for tokendrop training"
        )

    episodes = _parse_episodes(args.episodes)
    ds = LeRobotV21Dataset(
        repo_id=args.repo_id,
        root=args.dataset_root,
        episodes=episodes,
        video_backend=args.video_backend,
        return_uint8=True,
    )
    camera_keys = (
        [key.strip() for key in args.cameras.split(",") if key.strip()]
        if args.cameras
        else list(ds.meta.camera_keys)
    )
    state_key = args.state_key
    if state_key == STATE_TARGET_KEY and STATE_TARGET_KEY not in ds.features and STATE_SOURCE_KEY in ds.features:
        state_key = STATE_SOURCE_KEY

    config = SmolVLAConfig(
        vlm_model_name=args.vlm_model_name,
        load_vlm_weights=True,
        device=args.device,
        memory_enabled=False,
    )
    for key in camera_keys:
        config.input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
    policy = SmolVLAPolicy(config).to(args.device)
    policy.eval()
    model = policy.model

    args.output_root.mkdir(parents=True, exist_ok=True)
    total = len(ds)
    num_cameras = len(camera_keys)
    scorer = TokenDropScorer(
        stride=args.token_drop_stride,
        keptsize=args.token_drop_keptsize,
        grid_size=args.token_drop_grid,
    )
    current_ep: int | None = None
    current_ep_dir: Path | None = None

    for start in range(0, total, args.batch_size):
        end = min(total, start + args.batch_size)
        items = [ds[i] for i in range(start, end)]
        batch = _batch_items(items, camera_keys)
        batch = {key: value.to(args.device) for key, value in batch.items()}

        with torch.no_grad():
            images, _ = policy.prepare_images(batch)
            # images: list of [B,C,H,W] per camera
            raw_embs = []
            for img in images:
                emb = model.vlm_with_expert.embed_image(img)
                emb = emb * torch.tensor(emb.shape[-1] ** 0.5, dtype=emb.dtype, device=emb.device)
                raw_embs.append(emb)
            pooled_by_grid = {
                grid: [ _pool_tokens_to_grid(emb, grid).cpu().to(torch.float32) for emb in raw_embs ]
                for grid in grids
            }
            drop_pixels = torch.stack(
                [_pool_image_patches(img, args.token_drop_grid) for img in images],
                dim=1,
            )  # [B, V, P, C]

        for local_i, item in enumerate(items):
            ep_idx = int(item["episode_index"].item())
            frame_idx = int(item["frame_index"].item()) if "frame_index" in item else start + local_i
            out_dir = args.output_root / "features" / f"episode_{ep_idx}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"token_emb_{frame_idx}.npz"

            if current_ep is None:
                current_ep = ep_idx
                current_ep_dir = out_dir
            elif ep_idx != current_ep:
                _flush_kept_indices(scorer, current_ep_dir, args.write_kept_indices)
                current_ep = ep_idx
                current_ep_dir = out_dir

            if args.write_kept_indices:
                scorer.update(frame_idx, drop_pixels[local_i])

            if out_path.exists() and not args.overwrite:
                continue

            state = item[state_key].to(torch.float32)
            if args.normalize_state:
                state = _normalize_state(state, ds, state_key)
            state_emb = state.cpu().numpy().astype(np.float32)

            payload: dict[str, np.ndarray] = {"state_emb": state_emb}
            for grid in grids:
                # [V, P, D]
                image_grid = torch.stack(
                    [cam_emb[local_i] for cam_emb in pooled_by_grid[grid]],
                    dim=0,
                ).numpy().astype(np.float32)
                pos_flat = make_memory_pos_emb(frame_idx, num_cameras, grid, args.pos_dim)
                pos_grid = _reshape_pos(pos_flat, num_cameras, grid)
                payload[f"image_emb_{grid}x{grid}"] = image_grid
                payload[f"pos_emb_{grid}x{grid}"] = pos_grid

            alias_grid = args.framesamp_alias_grid
            alias_image = payload[f"image_emb_{alias_grid}x{alias_grid}"].reshape(-1, payload[f"image_emb_{alias_grid}x{alias_grid}"].shape[-1])
            alias_pos = payload[f"pos_emb_{alias_grid}x{alias_grid}"].reshape(-1, payload[f"pos_emb_{alias_grid}x{alias_grid}"].shape[-1])
            payload["image_emb"] = alias_image
            payload["pos_emb"] = alias_pos
            payload["mask"] = np.ones((alias_image.shape[0],), dtype=bool)

            np.savez_compressed(out_path, **payload)

        print(f"processed {end}/{total}", flush=True)

    _flush_kept_indices(scorer, current_ep_dir, args.write_kept_indices)
    print(
        f"done grids={grids} framesamp_alias={args.framesamp_alias_grid}x{args.framesamp_alias_grid} "
        f"kept_indices={args.write_kept_indices} token_drop_stride={args.token_drop_stride} "
        f"token_drop_keptsize={args.token_drop_keptsize}",
        flush=True,
    )


if __name__ == "__main__":
    main()
