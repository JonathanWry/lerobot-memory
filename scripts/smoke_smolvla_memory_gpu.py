#!/usr/bin/env python3
"""GPU smoke for SmolVLA memory modes on synthetic observations."""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def make_config(mode: str, vlm_model_name: str, device: str) -> SmolVLAConfig:
    base = SmolVLAConfig(
        device=device,
        vlm_model_name=vlm_model_name,
        load_vlm_weights=False,
        freeze_vision_encoder=True,
        train_expert_only=True,
        use_amp=False,
        compile_model=False,
        chunk_size=4,
        n_action_steps=4,
        num_steps=1,
        num_vlm_layers=2,
        tokenizer_max_length=48,
        max_state_dim=32,
        max_action_dim=32,
        resize_imgs_with_padding=(512, 512),
    )
    base.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "observation.images.cam_high": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        "observation.images.cam_left_wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        "observation.images.cam_right_wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
    }
    base.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
    }

    if mode == "native":
        return base
    if mode == "framesamp_modulation":
        return replace(
            base,
            memory_enabled=True,
            memory_family="robomme_perceptual",
            memory_integration="modulation",
            robomme_sampling_type="framesamp",
            visual_memory_frames=16,
            visual_memory_stride=1,
            robomme_token_budget=768,
            robomme_token_per_image=16,
            memory_runtime_log_interval=1,
        )
    if mode == "tokendrop_modulation":
        return replace(
            base,
            memory_enabled=True,
            memory_family="robomme_perceptual",
            memory_integration="modulation",
            robomme_sampling_type="tokendrop",
            visual_memory_frames=16,
            visual_memory_stride=2,
            robomme_token_budget=768,
            robomme_token_per_image=64,
            memory_runtime_log_interval=1,
        )
    if mode == "framesamp_expert":
        return replace(
            base,
            memory_enabled=True,
            memory_family="robomme_perceptual",
            memory_integration="expert",
            robomme_sampling_type="framesamp",
            visual_memory_frames=16,
            visual_memory_stride=1,
            robomme_token_budget=768,
            robomme_token_per_image=16,
            use_cache=False,
            memory_runtime_log_interval=1,
        )
    if mode == "recurrent_rmt_modulation":
        return replace(
            base,
            memory_enabled=True,
            memory_family="robomme_recurrent",
            memory_integration="modulation",
            recurrent_type="rmt",
            recurrent_hidden_dim=512,
            recurrent_budget=64,
            recurrent_max_recur_steps=4,
            recurrent_input_obs_horizon=2,
            recurrent_token_per_image=64,
            memory_runtime_log_interval=1,
        )
    if mode == "recurrent_ttt_modulation":
        return replace(
            base,
            memory_enabled=True,
            memory_family="robomme_recurrent",
            memory_integration="modulation",
            recurrent_type="ttt",
            recurrent_hidden_dim=512,
            recurrent_budget=64,
            recurrent_max_recur_steps=4,
            recurrent_input_obs_horizon=2,
            recurrent_token_per_image=64,
            recurrent_mini_batch_size=16,
            memory_runtime_log_interval=1,
        )
    raise ValueError(f"Unknown smoke mode: {mode}")


def dataset_stats() -> dict:
    stats = {
        "observation.state": {"mean": torch.zeros(14), "std": torch.ones(14)},
        "action": {"mean": torch.zeros(14), "std": torch.ones(14)},
    }
    for key in [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]:
        stats[key] = {"mean": torch.zeros(3, 480, 640), "std": torch.ones(3, 480, 640)}
    return stats


def make_raw_batch(device: torch.device, step: int) -> dict:
    torch.manual_seed(1000 + step)
    batch = {
        "observation.state": torch.randn(1, 14, dtype=torch.float32, device=device) * 0.01,
        "task": ["pick up the cup covering the object"],
    }
    for idx, key in enumerate(
        [
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ]
    ):
        image = torch.rand(1, 3, 480, 640, dtype=torch.float32, device=device)
        image[:, :, 32 + step : 64 + step, 32 + idx * 20 : 64 + idx * 20] = 1.0
        batch[key] = image.clamp(0, 1)
    return batch


def smoke_mode(mode: str, args: argparse.Namespace) -> dict:
    start = time.time()
    cfg = make_config(mode, args.vlm_model_name, args.device)
    policy = SmolVLAPolicy(cfg).to(args.device)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=None,
        dataset_stats=dataset_stats(),
    )
    device = torch.device(args.device)
    outputs = []
    with torch.inference_mode():
        for step in range(args.sequence_steps):
            batch = preprocessor(make_raw_batch(device, step))
            action = policy.predict_action_chunk(batch)
            if not torch.isfinite(action).all():
                raise RuntimeError(f"{mode}: non-finite action at step {step}")
            if action.shape != (1, cfg.chunk_size, 14):
                raise RuntimeError(f"{mode}: unexpected action shape {tuple(action.shape)}")
            outputs.append(float(action.abs().mean().item()))
    report = {
        "mode": mode,
        "ok": True,
        "output_abs_mean": outputs,
        "memory_buffer": type(policy.memory_buffer).__name__ if policy.memory_buffer is not None else None,
        "elapsed_s": round(time.time() - start, 3),
    }
    del policy
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm-model-name", default="/pfs/user/wangruiyu/models/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sequence-steps", type=int, default=2)
    parser.add_argument(
        "--modes",
        default="native,framesamp_modulation,tokendrop_modulation,framesamp_expert,recurrent_rmt_modulation,recurrent_ttt_modulation",
    )
    parser.add_argument("--save-json", default=None)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    reports = []
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        print(f"===== smoke {mode} =====", flush=True)
        report = smoke_mode(mode, args)
        print(json.dumps(report, indent=2), flush=True)
        reports.append(report)

    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w") as f:
            json.dump(reports, f, indent=2)
    print("PASS smolvla memory GPU smoke", flush=True)


if __name__ == "__main__":
    main()
