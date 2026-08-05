#!/usr/bin/env python

from __future__ import annotations

import heapq
import logging
import math
import time
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import Tensor


logger = logging.getLogger(__name__)


# ##modified by wangruiyu to adapt memory mechanism
@dataclass
class _MemoryFrame:
    image_emb: Tensor
    pos_emb: Tensor
    state_emb: Tensor
    mask: Tensor
    timestamp: float
    append_index: int
    image_pixels: Tensor | None = None


def _axis_sincos(value: float, dim: int, device: torch.device) -> Tensor:
    if dim == 0:
        return torch.zeros((0,), dtype=torch.float32, device=device)
    half = dim // 2
    if half == 0:
        return torch.tensor([value], dtype=torch.float32, device=device)
    omega = torch.arange(half, dtype=torch.float32, device=device) / max(half, 1)
    omega = 1.0 / (10000**omega)
    out = value * omega
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=0)
    if emb.shape[0] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[0]))
    return emb


def make_memory_pos_emb(frame_idx: int, num_cameras: int, grid_size: int, dim: int, device: torch.device) -> Tensor:
    axis_dims = [dim // 4] * 4
    axis_dims[-1] += dim - sum(axis_dims)
    rows = []
    for cam_idx in range(num_cameras):
        for y in range(grid_size):
            for x in range(grid_size):
                rows.append(
                    torch.cat(
                        [
                            _axis_sincos(float(frame_idx), axis_dims[0], device),
                            _axis_sincos(float(cam_idx), axis_dims[1], device),
                            _axis_sincos(float(y), axis_dims[2], device),
                            _axis_sincos(float(x), axis_dims[3], device),
                        ],
                        dim=0,
                    )
                )
    return torch.stack(rows, dim=0)


class SmolVLAPerceptualMemoryBuffer:
    """Online RoboMME-style perceptual memory buffer for SmolVLA inference/eval."""

    def __init__(
        self,
        num_frames: int,
        stride: int,
        token_per_image: int,
        token_budget: int,
        pos_dim: int,
        sampling_type: str = "framesamp",
        log_interval: int = 1,
    ):
        if num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {num_frames}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        grid_size = int(math.sqrt(token_per_image))
        if grid_size * grid_size != token_per_image:
            raise ValueError(f"token_per_image must be square for 2D pooling, got {token_per_image}")
        self.num_frames = num_frames
        self.stride = stride
        self.token_per_image = token_per_image
        if sampling_type not in {"framesamp", "tokendrop"}:
            raise ValueError(f"sampling_type must be framesamp or tokendrop, got {sampling_type!r}")
        self.token_budget = token_budget
        self.pos_dim = pos_dim
        self.sampling_type = sampling_type
        self.grid_size = grid_size
        self.log_interval = max(1, log_interval)
        self.tokens_per_frame: int | None = None
        self.frames: list[_MemoryFrame] = []
        self.scored_token_heap: list[tuple[float, int, int, int]] = []
        self.token_drop_last_frame = -1
        self.append_count = 0
        self.build_count = 0

    def reset(self) -> None:
        self.frames.clear()
        self.scored_token_heap.clear()
        self.token_drop_last_frame = -1
        self.append_count = 0
        self.build_count = 0

    def _pool_tokens(self, tokens: Tensor) -> Tensor:
        bsize, token_count, dim = tokens.shape
        side = int(math.sqrt(token_count))
        if side * side != token_count:
            if token_count < self.token_per_image:
                raise ValueError(f"Cannot pool {token_count} tokens to {self.grid_size}x{self.grid_size}")
            return tokens[:, : self.token_per_image]
        if side == self.grid_size:
            return tokens
        x = tokens.transpose(1, 2).reshape(bsize, dim, side, side)
        x = F.adaptive_avg_pool2d(x, (self.grid_size, self.grid_size))
        return x.flatten(2).transpose(1, 2)

    # ##modified by wangruiyu to adapt memory mechanism
    def _pool_image_patches(self, image: Tensor) -> Tensor:
        image = image.detach().to(dtype=torch.float32)
        if image.ndim != 4:
            raise ValueError(f"Expected image tensor as B,C,H,W, got shape={tuple(image.shape)}")
        patches = F.adaptive_avg_pool2d(image, (self.grid_size, self.grid_size))
        return patches.flatten(2).transpose(1, 2).cpu()

    def _process_token_drop_score(self, image_pixels: Tensor, step_idx: int) -> None:
        if step_idx == 0:
            view_count = image_pixels.shape[0]
            for patch_idx in range(self.token_per_image):
                for view_idx in range(view_count):
                    # ##modified by wangruiyu to adapt memory mechanism
                    # Match RoboMME: keep first-frame tokens with high priority as task/demo context.
                    heapq.heappush(self.scored_token_heap, (1000.0, step_idx, view_idx, patch_idx))
                    # ##modified by wangruiyu to adapt memory mechanism
        if step_idx != self.token_drop_last_frame + self.stride:
            return
        prev_idx = max(0, self.token_drop_last_frame)
        if prev_idx >= len(self.frames):
            return
        prev_img = self.frames[prev_idx].image_pixels
        curr_img = image_pixels
        if prev_img is None:
            return
        diff = (prev_img - curr_img).abs().mean(dim=-1)
        for view_idx in range(diff.shape[0]):
            for patch_idx in range(diff.shape[1]):
                score = float(diff[view_idx, patch_idx].item())
                if score < 1e-4:
                    continue
                heapq.heappush(self.scored_token_heap, (score, step_idx, view_idx, patch_idx))
                if len(self.scored_token_heap) > self.token_budget:
                    heapq.heappop(self.scored_token_heap)
        self.token_drop_last_frame += self.stride

    @staticmethod
    def _filter_token_dropping_indices(
        kept_indices: list[tuple[int, int, int]], step_idx: int, token_budget: int
    ) -> list[tuple[int, int, int]]:
        filtered = [item for item in kept_indices if item[0] <= step_idx]
        return sorted(filtered[-token_budget:])

    def append_from_prepared_inputs(self, policy, images: list[Tensor], img_masks: list[Tensor], state: Tensor) -> None:
        if len(images) == 0:
            return
        with torch.no_grad():
            per_camera_embs = []
            per_camera_masks = []
            per_camera_pixels = []
            for image, image_mask in zip(images, img_masks, strict=False):
                emb = policy.model.vlm_with_expert.embed_image(image)
                emb = emb * torch.tensor(emb.shape[-1] ** 0.5, dtype=emb.dtype, device=emb.device)
                emb = self._pool_tokens(emb).to(dtype=torch.float32)
                per_camera_embs.append(emb)
                per_camera_masks.append(
                    image_mask[:, None].expand(image_mask.shape[0], emb.shape[1]).to(dtype=torch.bool)
                )
                per_camera_pixels.append(self._pool_image_patches(image))

            image_emb = torch.cat(per_camera_embs, dim=1)
            mask = torch.cat(per_camera_masks, dim=1)
            bsize, tokens, _ = image_emb.shape
            self.tokens_per_frame = int(tokens)
            pos = make_memory_pos_emb(
                self.append_count,
                len(per_camera_embs),
                self.grid_size,
                self.pos_dim,
                image_emb.device,
            )
            pos_emb = pos[None, :, :].expand(bsize, -1, -1)
            state_emb = state[:, None, :].expand(bsize, tokens, -1).to(dtype=torch.float32)
            image_pixels = torch.cat(per_camera_pixels, dim=0)

        now = time.monotonic()
        frame = _MemoryFrame(
            image_emb=image_emb.detach().cpu(),
            pos_emb=pos_emb.detach().cpu(),
            state_emb=state_emb.detach().cpu(),
            mask=mask.detach().cpu(),
            timestamp=now,
            append_index=self.append_count,
            image_pixels=image_pixels,
        )
        self.frames.append(frame)
        if self.sampling_type == "tokendrop":
            self._process_token_drop_score(image_pixels, self.append_count)
        self.append_count += 1

    @staticmethod
    def _even_memory_indices(step_idx: int, max_size: int) -> list[int]:
        if max_size < 1:
            raise ValueError(f"memory max_size must be >= 1, got {max_size}")
        if step_idx < max_size:
            return list(range(step_idx + 1))
        return torch.linspace(0, step_idx, max_size, dtype=torch.int32).tolist()

    def _effective_max_frames(self) -> int:
        if self.token_budget is None or self.tokens_per_frame is None:
            return self.num_frames
        return min(self.num_frames, max(1, self.token_budget // self.tokens_per_frame))

    def _sample_frames(self) -> tuple[list[_MemoryFrame], int, int]:
        max_frames = self._effective_max_frames()
        if not self.frames:
            return [], max_frames, max_frames
        indices = self._even_memory_indices(len(self.frames) - 1, max_frames)
        sampled = [self.frames[idx] for idx in indices]
        return sampled, max_frames - len(sampled), max_frames

    def _sample_token_drop_tokens(
        self,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None, dict]:
        if not self.frames:
            return None, None, None, None, {
                "memory_N": 0,
                "target_K": self.num_frames,
                "effective_K": 0,
                "padding_frames": 0,
                "tokens": 0,
                "raw_tokens": 0,
                "dropped_tokens": 0,
                "appends": self.append_count,
                "buffer": len(self.frames),
                "indices": [],
                "ages": [],
                "sampling": self.sampling_type,
            }
        kept_indices = [(frame_idx, view_idx, patch_idx) for _, frame_idx, view_idx, patch_idx in self.scored_token_heap]
        sorted_kept = self._filter_token_dropping_indices(kept_indices, len(self.frames) - 1, self.token_budget)
        if not sorted_kept:
            return None, None, None, None, {
                "memory_N": 0,
                "target_K": self.num_frames,
                "effective_K": 0,
                "padding_frames": 0,
                "tokens": 0,
                "raw_tokens": 0,
                "dropped_tokens": 0,
                "appends": self.append_count,
                "buffer": len(self.frames),
                "indices": [],
                "ages": [],
                "sampling": self.sampling_type,
            }
        template = self.frames[sorted_kept[0][0]]
        image_emb = torch.zeros((template.image_emb.shape[0], self.token_budget, template.image_emb.shape[-1]), dtype=template.image_emb.dtype)
        pos_emb = torch.zeros((template.pos_emb.shape[0], self.token_budget, template.pos_emb.shape[-1]), dtype=template.pos_emb.dtype)
        state_emb = torch.zeros((template.state_emb.shape[0], self.token_budget, template.state_emb.shape[-1]), dtype=template.state_emb.dtype)
        mask = torch.zeros((template.mask.shape[0], self.token_budget), dtype=torch.bool)
        now = time.monotonic()
        for out_idx, (frame_idx, view_idx, patch_idx) in enumerate(sorted_kept[: self.token_budget]):
            frame = self.frames[frame_idx]
            token_idx = view_idx * self.token_per_image + patch_idx
            image_emb[:, out_idx] = frame.image_emb[:, token_idx]
            pos_emb[:, out_idx] = frame.pos_emb[:, token_idx]
            state_emb[:, out_idx] = frame.state_emb[:, token_idx]
            mask[:, out_idx] = frame.mask[:, token_idx]
        info = {
            "memory_N": len({frame_idx for frame_idx, _, _ in sorted_kept}),
            "target_K": self.num_frames,
            "effective_K": len({frame_idx for frame_idx, _, _ in sorted_kept}),
            "stride": self.stride,
            "padding_frames": 0,
            "tokens": int(mask.sum().item()),
            "raw_tokens": len(sorted_kept),
            "dropped_tokens": max(0, len(self.scored_token_heap) - self.token_budget),
            "appends": self.append_count,
            "buffer": len(self.frames),
            "indices": sorted_kept[:16],
            "ages": [round(now - self.frames[frame_idx].timestamp, 3) for frame_idx, _, _ in sorted_kept[:16]],
            "sampling": self.sampling_type,
        }
        self.build_count += 1
        return image_emb, pos_emb, state_emb, mask, info

    def _pad_to_frames(self, sampled: list[_MemoryFrame], padding_frames: int) -> list[_MemoryFrame]:
        if padding_frames <= 0:
            return sampled
        if not sampled:
            return sampled
        template = sampled[-1]
        zeros = _MemoryFrame(
            image_emb=torch.zeros_like(template.image_emb),
            pos_emb=torch.zeros_like(template.pos_emb),
            state_emb=torch.zeros_like(template.state_emb),
            mask=torch.zeros_like(template.mask, dtype=torch.bool),
            timestamp=template.timestamp,
            append_index=-1,
        )
        return sampled + [replace(zeros) for _ in range(padding_frames)]

    # ##modified by wangruiyu to adapt memory mechanism
    def _left_pad_to_frames(self, sampled: list[_MemoryFrame], padding_frames: int) -> list[_MemoryFrame]:
        if padding_frames <= 0:
            return sampled
        if not sampled:
            return sampled
        template = sampled[0]
        zeros = _MemoryFrame(
            image_emb=torch.zeros_like(template.image_emb),
            pos_emb=torch.zeros_like(template.pos_emb),
            state_emb=torch.zeros_like(template.state_emb),
            mask=torch.zeros_like(template.mask, dtype=torch.bool),
            timestamp=template.timestamp,
            append_index=-1,
        )
        return [replace(zeros) for _ in range(padding_frames)] + sampled
    # ##modified by wangruiyu to adapt memory mechanism

    def build_batch(self, device: torch.device) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None, dict]:
        if self.sampling_type == "tokendrop":
            image_emb, pos_emb, state_emb, mask, info = self._sample_token_drop_tokens()
            return (
                None if image_emb is None else image_emb.to(device=device),
                None if pos_emb is None else pos_emb.to(device=device),
                None if state_emb is None else state_emb.to(device=device),
                None if mask is None else mask.to(device=device),
                info,
            )
        sampled, padding_frames, max_frames = self._sample_frames()
        if not sampled:
            return None, None, None, None, {
                "memory_N": 0,
                "target_K": self.num_frames,
                "effective_K": max_frames,
                "padding_frames": max_frames,
                "tokens": 0,
                "raw_tokens": 0,
                "dropped_tokens": 0,
                "appends": self.append_count,
                "buffer": len(self.frames),
                "indices": [],
                "ages": [],
            }

        sampled = self._pad_to_frames(sampled, padding_frames)

        image_emb = torch.cat([frame.image_emb for frame in sampled], dim=1)
        pos_emb = torch.cat([frame.pos_emb for frame in sampled], dim=1)
        state_emb = torch.cat([frame.state_emb for frame in sampled], dim=1)
        mask = torch.cat([frame.mask for frame in sampled], dim=1)
        raw_tokens = image_emb.shape[1]
        dropped_tokens = 0
        if self.token_budget is not None and raw_tokens > self.token_budget:
            dropped_tokens = raw_tokens - self.token_budget
            image_emb = image_emb[:, -self.token_budget :]
            pos_emb = pos_emb[:, -self.token_budget :]
            state_emb = state_emb[:, -self.token_budget :]
            mask = mask[:, -self.token_budget :]

        now = time.monotonic()
        info = {
            "memory_N": len({frame.append_index for frame in sampled if frame.append_index >= 0}),
            "target_K": self.num_frames,
            "effective_K": max_frames,
            "stride": self.stride,
            "padding_frames": padding_frames,
            "tokens": int(image_emb.shape[1]),
            "raw_tokens": int(raw_tokens),
            "dropped_tokens": int(dropped_tokens),
            "appends": self.append_count,
            "buffer": len(self.frames),
            "indices": [frame.append_index for frame in sampled],
            "ages": [round(now - frame.timestamp, 3) for frame in sampled],
            "sampling": self.sampling_type,
        }
        self.build_count += 1
        return (
            image_emb.to(device=device),
            pos_emb.to(device=device),
            state_emb.to(device=device),
            mask.to(device=device),
            info,
        )

    def should_log(self) -> bool:
        return self.build_count <= 3 or self.build_count % self.log_interval == 0


class SmolVLARecurrentMemoryBuffer(SmolVLAPerceptualMemoryBuffer):
    """Online RoboMME-style recurrent memory buffer for SmolVLA inference/eval."""

    def __init__(
        self,
        max_recur_steps: int,
        input_obs_horizon: int,
        max_pretraj_steps: int,
        token_per_image: int,
        pos_dim: int,
        log_interval: int = 1,
    ):
        super().__init__(
            num_frames=max_recur_steps,
            stride=1,
            token_per_image=token_per_image,
            token_budget=None,
            pos_dim=pos_dim,
            sampling_type="framesamp",
            log_interval=log_interval,
        )
        self.max_recur_steps = max_recur_steps
        self.input_obs_horizon = max(1, input_obs_horizon)
        self.max_pretraj_steps = max_pretraj_steps

    def _get_token_recurrent_indices(self, step_idx: int, exec_start_idx: int = 0) -> list[int]:
        horizon = self.input_obs_horizon
        if exec_start_idx == 0:
            if step_idx < horizon:
                indices = [step_idx]
            else:
                start_idx = step_idx % horizon
                indices = list(range(start_idx, step_idx + 1, horizon))
                indices = indices[-self.max_recur_steps :]
        else:
            if exec_start_idx <= horizon * 2:
                video_indices = list(range(0, exec_start_idx, max(1, horizon // 2)))
            elif exec_start_idx <= self.max_pretraj_steps * horizon:
                video_indices = list(range(0, exec_start_idx, horizon))
            else:
                video_indices = torch.linspace(0, exec_start_idx - 1, self.max_pretraj_steps, dtype=torch.int32).tolist()

            if step_idx - exec_start_idx < horizon:
                rest_indices = [step_idx]
            else:
                start_idx = (step_idx - exec_start_idx) % horizon + exec_start_idx
                rest_indices = list(range(start_idx, step_idx + 1, horizon))
            indices = (video_indices + rest_indices)[-self.max_recur_steps :]

        return [int(idx) for idx in indices]

    def build_batch(self, device: torch.device) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None, dict]:
        if not self.frames:
            return None, None, None, None, {
                "memory_N": 0,
                "target_K": self.max_recur_steps,
                "effective_K": 0,
                "padding_frames": self.max_recur_steps,
                "tokens": 0,
                "raw_tokens": 0,
                "dropped_tokens": 0,
                "appends": self.append_count,
                "buffer": 0,
                "indices": [],
                "ages": [],
                "sampling": "recurrent",
            }

        indices = self._get_token_recurrent_indices(len(self.frames) - 1, exec_start_idx=0)
        sampled = [self.frames[idx] for idx in indices if idx < len(self.frames)]
        padding_frames = self.max_recur_steps - len(sampled)
        sampled = self._left_pad_to_frames(sampled, padding_frames)
        if not sampled:
            return None, None, None, None, {}

        first = sampled[0]
        batch_size = first.image_emb.shape[0]
        tokens_per_frame = first.image_emb.shape[1]
        num_views = max(1, tokens_per_frame // self.token_per_image)
        image_emb = torch.stack(
            [frame.image_emb.reshape(batch_size, num_views, self.token_per_image, -1) for frame in sampled],
            dim=1,
        )
        pos_emb = torch.stack(
            [frame.pos_emb.reshape(batch_size, num_views, self.token_per_image, -1) for frame in sampled],
            dim=1,
        )
        state_emb = torch.stack([frame.state_emb[:, 0, :] for frame in sampled], dim=1)
        mask = torch.tensor([frame.append_index >= 0 for frame in sampled], dtype=torch.bool)[None, :].expand(batch_size, -1)

        now = time.monotonic()
        info = {
            "memory_N": len({frame.append_index for frame in sampled if frame.append_index >= 0}),
            "target_K": self.max_recur_steps,
            "effective_K": len(sampled) - padding_frames,
            "stride": self.input_obs_horizon,
            "padding_frames": padding_frames,
            "tokens": int(mask.sum().item()) * num_views * self.token_per_image,
            "raw_tokens": len(sampled) * num_views * self.token_per_image,
            "dropped_tokens": 0,
            "appends": self.append_count,
            "buffer": len(self.frames),
            "indices": [frame.append_index for frame in sampled],
            "ages": [round(now - frame.timestamp, 3) for frame in sampled],
            "sampling": "recurrent",
        }
        self.build_count += 1
        return (
            image_emb.to(device=device),
            pos_emb.to(device=device),
            state_emb.to(device=device),
            mask.to(device=device),
            info,
        )
# ##modified by wangruiyu to adapt memory mechanism
