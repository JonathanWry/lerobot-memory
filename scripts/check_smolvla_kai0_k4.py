#!/usr/bin/env python3
"""Smoke checks for SmolVLA kai0_explicit K=4 multi-frame path."""

from __future__ import annotations

import argparse
import sys

import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


CAMS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def _build_policy(model_dir: str, device: str, k: int = 4) -> SmolVLAPolicy:
    cfg = SmolVLAConfig(
        vlm_model_name=model_dir,
        load_vlm_weights=False,
        device=device,
        memory_enabled=True,
        memory_family="kai0_explicit",
        memory_integration="context",
        memory_runtime_enabled=False,
        visual_memory_frames=k,
        visual_memory_stride=1,
        proprio_memory_frames=1,
    )
    for key in CAMS:
        cfg.input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
    cfg.input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(14,))
    cfg.output_features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(14,))
    policy = SmolVLAPolicy(cfg).to(device)
    policy.eval()
    return policy


def check_prepare_images(policy: SmolVLAPolicy, device: str, k: int = 4) -> None:
    bsize = 2
    batch = {
        cam: torch.rand(bsize, k, 3, 480, 640, device=device) for cam in CAMS
    }
    # Mark first temporal index as pad for one camera.
    batch[f"{CAMS[0]}_is_pad"] = torch.zeros(bsize, k, dtype=torch.bool, device=device)
    batch[f"{CAMS[0]}_is_pad"][:, 0] = True

    images, masks = policy.prepare_images(batch)
    expected = len(CAMS) * k
    assert len(images) == expected, f"expected {expected} image streams, got {len(images)}"
    assert len(masks) == expected, f"expected {expected} masks, got {len(masks)}"
    assert images[0].shape == (bsize, 3, images[0].shape[-2], images[0].shape[-1])
    assert not bool(masks[0].all()), "padded first frame of cam0 should have some False masks"
    assert bool(masks[1].all()), "non-padded temporal frames should be valid"
    print(f"[PASS] prepare_images streams={expected} shape0={tuple(images[0].shape)}", flush=True)


def check_delta_indices() -> None:
    cfg = SmolVLAConfig(
        memory_enabled=True,
        memory_family="kai0_explicit",
        visual_memory_frames=4,
        visual_memory_stride=1,
        proprio_memory_frames=1,
    )
    assert cfg.observation_delta_indices == [-3, -2, -1, 0], cfg.observation_delta_indices
    print(f"[PASS] observation_delta_indices={cfg.observation_delta_indices}", flush=True)


def check_config_guards() -> None:
    try:
        SmolVLAConfig(
            memory_enabled=True,
            memory_family="kai0_explicit",
            visual_memory_frames=1,
        )
        raise AssertionError("expected ValueError for visual_memory_frames=1")
    except ValueError as exc:
        print(f"[PASS] config_guard_k: {exc}", flush=True)


def check_no_runtime_buffer(policy: SmolVLAPolicy) -> None:
    assert policy.memory_buffer is None, "kai0_explicit must not create memory_buffer"
    print("[PASS] memory_buffer is None for kai0_explicit", flush=True)


def check_train_step(policy: SmolVLAPolicy, device: str, k: int = 4) -> None:
    policy.train()
    bsize = 1
    batch = {cam: torch.rand(bsize, k, 3, 480, 640, device=device) for cam in CAMS}
    batch[OBS_STATE] = torch.randn(bsize, 14, device=device)
    batch[ACTION] = torch.randn(bsize, policy.config.chunk_size, 14, device=device)
    text = "pick up the cup\n"
    tok = policy.model.vlm_with_expert.processor.tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=48
    )
    batch[OBS_LANGUAGE_TOKENS] = tok["input_ids"].to(device)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = tok["attention_mask"].to(device)
    loss, _ = policy.forward(batch)
    loss.backward()
    print(f"[PASS] train_step loss={float(loss.detach()):.6g}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/pfs/user/wangruiyu/models/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-train-step", action="store_true")
    args = parser.parse_args()

    check_config_guards()
    check_delta_indices()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[FAIL] cuda unavailable", flush=True)
        return 1

    policy = _build_policy(args.model_dir, args.device, k=4)
    check_no_runtime_buffer(policy)
    check_prepare_images(policy, args.device, k=4)
    if not args.skip_train_step:
        check_train_step(policy, args.device, k=4)
    print("===== kai0_k4 checks OK =====", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
