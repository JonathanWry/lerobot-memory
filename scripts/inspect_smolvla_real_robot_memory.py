#!/usr/bin/env python3
"""Inspect a SmolVLA checkpoint/config before real-robot memory deployment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
# ##modified by wangruiyu to adapt memory mechanism
import lerobot.policies.smolvla  # noqa: F401 - register SmolVLA config
# ##modified by wangruiyu to adapt memory mechanism


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", required=True, help="Checkpoint/pretrained_model directory.")
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()

    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    report = {
        "policy_path": str(Path(args.policy_path).resolve()),
        "type": getattr(cfg, "type", None),
        "device": getattr(cfg, "device", None),
        "memory_enabled": getattr(cfg, "memory_enabled", False),
        "memory_family": getattr(cfg, "memory_family", "none"),
        "memory_integration": getattr(cfg, "memory_integration", "context"),
        "robomme_sampling_type": getattr(cfg, "robomme_sampling_type", None),
        "visual_memory_frames": getattr(cfg, "visual_memory_frames", None),
        "visual_memory_stride": getattr(cfg, "visual_memory_stride", None),
        "robomme_token_budget": getattr(cfg, "robomme_token_budget", None),
        "robomme_token_per_image": getattr(cfg, "robomme_token_per_image", None),
        "recurrent_type": getattr(cfg, "recurrent_type", None),
        "recurrent_max_recur_steps": getattr(cfg, "recurrent_max_recur_steps", None),
        "n_action_steps": getattr(cfg, "n_action_steps", None),
        "chunk_size": getattr(cfg, "chunk_size", None),
        "use_cache": getattr(cfg, "use_cache", None),
        "image_features": sorted(getattr(cfg, "image_features", {}).keys()),
        "state_feature": getattr(getattr(cfg, "state_feature", None), "shape", None),
        "action_feature": getattr(getattr(cfg, "action_feature", None), "shape", None),
        "notes": [],
    }

    if report["memory_enabled"] and report["memory_family"] == "none":
        report["notes"].append("ERROR: memory_enabled=True but memory_family='none'.")
    if report["memory_enabled"] and report["memory_integration"] == "expert" and report["use_cache"]:
        report["notes"].append(
            "ERROR: memory expert path currently requires --policy.use_cache=false for training/eval/real robot."
        )
    if report["memory_enabled"] and report["robomme_sampling_type"] == "tokendrop":
        report["notes"].append("INFO: tokendrop should log memory_N/tokens/dropped_tokens during runtime.")
    if not report["notes"]:
        report["notes"].append("OK: no obvious config-level blocker found.")

    if args.print_json:
        print(json.dumps(report, indent=2, default=str))
        return

    print("===== SmolVLA Real-Robot Memory Check =====")
    for key, value in report.items():
        if key == "notes":
            continue
        print(f"{key}: {value}")
    print("notes:")
    for note in report["notes"]:
        print(f"  - {note}")


if __name__ == "__main__":
    main()
