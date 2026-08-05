#!/usr/bin/env python3
"""Broader SmolVLA memory coverage checks against the smoke sidecar.

Layers:
1) Sidecar / dataset loader shape + RoboMME-style sampling invariants (CPU)
2) Policy wiring: context no-op vs modul/expert consume memory (GPU, batch=1)
3) Edge cases: empty kept_indices, missing flat image_emb alias

Exit non-zero if any hard FAIL. Soft WARN lines are recorded but do not fail
unless --strict-warn is set.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch


RESULTS: list[tuple[str, str, str]] = []  # status, name, detail


def _record(status: str, name: str, detail: str = "") -> None:
    RESULTS.append((status, name, detail))
    prefix = {"PASS": "[PASS]", "FAIL": "[FAIL]", "WARN": "[WARN]", "INFO": "[INFO]"}.get(status, f"[{status}]")
    print(f"{prefix} {name}" + (f" :: {detail}" if detail else ""), flush=True)


def _fail(name: str, detail: str) -> None:
    _record("FAIL", name, detail)


def _pass(name: str, detail: str = "") -> None:
    _record("PASS", name, detail)


def _warn(name: str, detail: str) -> None:
    _record("WARN", name, detail)


def check_sidecar_layout(feature_root: Path, episode: int) -> None:
    ep = feature_root / "features" / f"episode_{episode}"
    sample = ep / "token_emb_0.npz"
    kept_path = ep / "kept_indices.json"
    if not sample.exists():
        _fail("sidecar.sample", f"missing {sample}")
        return
    if not kept_path.exists():
        _fail("sidecar.kept_indices", f"missing {kept_path}")
        return
    with np.load(sample) as npz:
        keys = set(npz.files)
        required = {
            "image_emb",
            "pos_emb",
            "mask",
            "state_emb",
            "image_emb_8x8",
            "image_emb_4x4",
            "image_emb_2x2",
            "pos_emb_8x8",
            "pos_emb_4x4",
            "pos_emb_2x2",
        }
        missing = sorted(required - keys)
        if missing:
            _fail("sidecar.keys", f"missing {missing}")
        else:
            _pass("sidecar.keys", f"{sorted(required)}")
        checks = [
            ("image_emb", (48, 960)),
            ("pos_emb", (48, 256)),
            ("image_emb_8x8", (3, 64, 960)),
            ("image_emb_4x4", (3, 16, 960)),
            ("image_emb_2x2", (3, 4, 960)),
            ("state_emb", (14,)),
        ]
        for key, shape in checks:
            if key not in npz:
                continue
            got = tuple(npz[key].shape)
            if got != shape:
                _fail(f"sidecar.shape.{key}", f"got {got} expected {shape}")
            else:
                _pass(f"sidecar.shape.{key}", str(got))
        # flat alias must match 4x4 flatten
        if "image_emb" in npz and "image_emb_4x4" in npz:
            flat = npz["image_emb"]
            grid = npz["image_emb_4x4"].reshape(-1, npz["image_emb_4x4"].shape[-1])
            if flat.shape == grid.shape and np.allclose(flat, grid, atol=1e-5):
                _pass("sidecar.alias_matches_4x4")
            else:
                _fail("sidecar.alias_matches_4x4", f"flat={flat.shape} grid_flat={grid.shape}")
    kept = json.loads(kept_path.read_text())
    if not kept:
        _fail("sidecar.kept_nonempty", "kept_indices empty")
    else:
        bad = [x for x in kept if not (isinstance(x, (list, tuple)) and len(x) == 3)]
        if bad:
            _fail("sidecar.kept_format", f"bad entries={bad[:3]}")
        else:
            views = {int(x[1]) for x in kept}
            patches = {int(x[2]) for x in kept}
            _pass(
                "sidecar.kept_format",
                f"n={len(kept)} views={sorted(views)} patch_max={max(patches)}",
            )
            if max(patches) > 63:
                _fail("sidecar.kept_patch_range", f"patch_max={max(patches)} > 63 for 8x8")
            if views - {0, 1, 2}:
                _warn("sidecar.kept_views", f"unexpected views {views}")


def _make_dataset(feature_root: Path, sampling: str, episode: int = 0, **kwargs):
    from lerobot.datasets.lerobot_v21_dataset import LeRobotV21Dataset

    defaults = dict(
        repo_id="memorybench",
        root=Path("/pfs/user/wangruiyu/data/MemoryBench"),
        episodes=[episode],
        video_backend="pyav",
        return_uint8=True,
        memory_feature_root=feature_root,
        memory_sampling_type=sampling,
        memory_num_frames=16,
        memory_stride=1,
        memory_token_budget=768,
        memory_feature_required=True,
    )
    defaults.update(kwargs)
    return LeRobotV21Dataset(**defaults)


def check_loader_framesamp(feature_root: Path, episode: int) -> None:
    ds = _make_dataset(feature_root, "framesamp", episode)
    # mid and late frames
    for idx_name, idx in [("early", 0), ("mid", min(200, len(ds) - 1)), ("late", len(ds) - 1)]:
        item = ds[idx]
        for key in ("static_image_emb", "static_pos_emb", "static_state_emb", "static_mask"):
            if key not in item:
                _fail(f"loader.framesamp.{idx_name}.{key}", "missing")
                return
        img = item["static_image_emb"]
        pos = item["static_pos_emb"]
        state = item["static_state_emb"]
        mask = item["static_mask"]
        if img.ndim != 2 or img.shape[-1] != 960:
            _fail(f"loader.framesamp.{idx_name}.image_rank", str(tuple(img.shape)))
            continue
        if img.shape[0] != pos.shape[0] or img.shape[0] != mask.shape[0] or img.shape[0] != state.shape[0]:
            _fail(
                f"loader.framesamp.{idx_name}.len_mismatch",
                f"img={tuple(img.shape)} pos={tuple(pos.shape)} state={tuple(state.shape)} mask={tuple(mask.shape)}",
            )
            continue
        # budget 768 / 48 tokens/frame => max 16 frames
        if img.shape[0] > 768:
            _fail(f"loader.framesamp.{idx_name}.over_budget", str(tuple(img.shape)))
        else:
            _pass(
                f"loader.framesamp.{idx_name}",
                f"tokens={img.shape[0]} mask_true={int(mask.sum())} state_dim={state.shape[-1]}",
            )


def check_loader_tokendrop(feature_root: Path, episode: int) -> None:
    ds = _make_dataset(feature_root, "tokendrop", episode, memory_token_budget=768)
    for idx_name, idx in [("early", 0), ("mid", min(200, len(ds) - 1)), ("late", len(ds) - 1)]:
        item = ds[idx]
        img = item["static_image_emb"]
        if tuple(img.shape) != (768, 960):
            _fail(f"loader.tokendrop.{idx_name}.shape", str(tuple(img.shape)))
            continue
        mask = item["static_mask"]
        n = int(mask.sum())
        if n == 0:
            _fail(f"loader.tokendrop.{idx_name}.empty_mask", "no tokens selected")
        else:
            _pass(f"loader.tokendrop.{idx_name}", f"kept_active={n}/768")


def check_loader_recurrent(feature_root: Path, episode: int) -> None:
    ds = _make_dataset(
        feature_root,
        "recurrent",
        episode,
        recurrent_max_recur_steps=8,
        recurrent_max_pretraj_steps=40,
        memory_num_frames=16,
    )
    item = ds[min(200, len(ds) - 1)]
    img = item["recur_image_emb"]
    # expect [T,V,P,D] with T=8 padded, V=3, P=64
    if img.ndim != 4:
        _fail("loader.recurrent.rank", str(tuple(img.shape)))
        return
    t, v, p, d = img.shape
    if v != 3 or p != 64 or d != 960 or t != 8:
        _fail("loader.recurrent.shape", str(tuple(img.shape)))
    else:
        _pass("loader.recurrent.shape", str(tuple(img.shape)))
    mask = item["recur_mask"]
    if mask.ndim != 1 or mask.shape[0] != t:
        _fail("loader.recurrent.mask", str(tuple(mask.shape)))
    else:
        _pass("loader.recurrent.mask", f"true={int(mask.sum())}/{t}")


def check_framesamp_missing_flat_alias(feature_root: Path, episode: int) -> None:
    """Document risk: without flat image_emb, loader may prefer 8x8 3D and break framesamp."""
    src = feature_root / "features" / f"episode_{episode}"
    with tempfile.TemporaryDirectory(prefix="smolvla_nofloat_") as tmp:
        tmp_root = Path(tmp)
        dst = tmp_root / "features" / f"episode_{episode}"
        dst.mkdir(parents=True)
        # copy a few frames only
        for frame_idx in range(0, 32):
            src_path = src / f"token_emb_{frame_idx}.npz"
            if not src_path.exists():
                break
            with np.load(src_path) as npz:
                payload = {k: npz[k] for k in npz.files if k not in {"image_emb", "pos_emb", "mask"}}
            np.savez_compressed(dst / f"token_emb_{frame_idx}.npz", **payload)
        shutil.copy2(src / "kept_indices.json", dst / "kept_indices.json")
        try:
            ds = _make_dataset(tmp_root, "framesamp", episode, memory_num_frames=4, memory_token_budget=192)
            item = ds[min(16, len(ds) - 1)]
            img = item["static_image_emb"]
            # Healthy behavior would flatten 4x4 → 2D. Current code may yield broken shapes.
            if img.ndim == 2 and img.shape[-1] == 960:
                _pass("edge.framesamp_no_flat_alias", f"recovered shape={tuple(img.shape)}")
            else:
                _warn(
                    "edge.framesamp_no_flat_alias",
                    f"unexpected shape={tuple(img.shape)}; keep writing flat image_emb aliases in build",
                )
        except Exception as exc:  # noqa: BLE001
            _warn("edge.framesamp_no_flat_alias", f"raised {type(exc).__name__}: {exc}")


def check_tokendrop_empty_kept(feature_root: Path, episode: int) -> None:
    src = feature_root / "features" / f"episode_{episode}"
    with tempfile.TemporaryDirectory(prefix="smolvla_emptykept_") as tmp:
        tmp_root = Path(tmp)
        dst = tmp_root / "features" / f"episode_{episode}"
        dst.mkdir(parents=True)
        for path in list(src.glob("token_emb_*.npz"))[:64]:
            shutil.copy2(path, dst / path.name)
        (dst / "kept_indices.json").write_text("[]")
        ds = _make_dataset(tmp_root, "tokendrop", episode, memory_feature_required=False)
        item = ds[10]
        has_mem = "static_image_emb" in item
        if has_mem:
            _warn("edge.tokendrop_empty_kept", "loader still returned memory keys for empty kept_indices")
        else:
            _pass("edge.tokendrop_empty_kept", "no memory keys (will KeyError in modul/expert prepare_memory)")


def check_policy_wiring(device: str, model_dir: str, feature_root: Path, episode: int) -> None:
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

    ds = _make_dataset(feature_root, "framesamp", episode)
    item = ds[min(100, len(ds) - 1)]

    def build_policy(integration: str, family: str = "robomme_perceptual", sampling: str = "framesamp", tpi: int = 16):
        cfg = SmolVLAConfig(
            vlm_model_name=model_dir,
            load_vlm_weights=False,
            device=device,
            memory_enabled=True,
            memory_family=family,
            memory_integration=integration,
            robomme_sampling_type=sampling if family == "robomme_perceptual" else "framesamp",
            robomme_token_per_image=tpi,
            robomme_token_budget=768,
            visual_memory_frames=16,
            recurrent_max_recur_steps=8 if family == "robomme_recurrent" else 64,
            recurrent_token_per_image=64,
            memory_runtime_enabled=False,
        )
        for key in (
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ):
            cfg.input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
        cfg.input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(14,))
        cfg.output_features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(14,))
        policy = SmolVLAPolicy(cfg).to(device)
        policy.train()
        return policy

    def batch_from_item(policy, sample: dict, memory_keys: list[str]) -> dict:
        batch = {}
        for key, value in sample.items():
            if key == "task":
                continue
            if not torch.is_tensor(value):
                continue
            tensor = value.unsqueeze(0).to(device)
            if key.startswith("observation.images"):
                if tensor.dtype == torch.uint8:
                    tensor = tensor.float() / 255.0
                else:
                    tensor = tensor.float()
            batch[key] = tensor
        # MemoryBench aliases → policy constants
        if "observation.state" not in batch:
            for alt in ("observation.qpos", "observation.qpos"):
                if alt in batch:
                    batch[OBS_STATE] = batch[alt]
                    break
        if ACTION not in batch:
            for alt in ("real_action", "action"):
                if alt in batch:
                    batch[ACTION] = batch[alt]
                    break
        for key in memory_keys:
            if key in sample:
                dtype = torch.bool if "mask" in key else torch.float32
                batch[key] = sample[key].unsqueeze(0).to(device=device, dtype=dtype)
        task = sample.get("task", "pick up the object")
        if isinstance(task, (list, tuple)):
            task = task[0]
        text = task if str(task).endswith("\n") else f"{task}\n"
        tok = policy.model.vlm_with_expert.processor.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=48,
        )
        batch[OBS_LANGUAGE_TOKENS] = tok["input_ids"].to(device)
        batch[OBS_LANGUAGE_ATTENTION_MASK] = tok["attention_mask"].to(device)
        return batch

    def try_train_step(name: str, policy, batch: dict) -> None:
        try:
            loss, _ = policy.forward(batch)
            loss.backward()
            _pass(f"{name}.train_step", f"loss={float(loss.detach()):.6g}")
        except Exception as exc:  # noqa: BLE001
            _warn(f"{name}.train_step", f"{type(exc).__name__}: {exc}")

    # context: encoder must be absent / prepare_memory None
    try:
        policy = build_policy("context")
        if policy.model.memory_encoder is not None:
            _fail("policy.context.encoder", "memory_encoder should be None")
        else:
            _pass("policy.context.encoder_none")
        batch = batch_from_item(
            policy, item, ["static_image_emb", "static_pos_emb", "static_state_emb", "static_mask"]
        )
        mem_tokens, mem_mask = policy.model.prepare_memory(batch)
        if mem_tokens is None and mem_mask is None:
            _pass("policy.context.prepare_memory_noop", "sidecar ignored as expected with current code")
        else:
            _fail("policy.context.prepare_memory_noop", "context unexpectedly produced memory tokens")
        try_train_step("policy.context", policy, batch)
        del policy
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        _fail("policy.context", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    # modulation: must encode and respond to zeroed memory
    try:
        policy = build_policy("modulation")
        if policy.model.memory_encoder is None:
            _fail("policy.modulation.encoder", "missing memory_encoder")
        else:
            _pass("policy.modulation.encoder_present")
        batch = batch_from_item(
            policy, item, ["static_image_emb", "static_pos_emb", "static_state_emb", "static_mask"]
        )
        tok_a, _mask_a = policy.model.prepare_memory(batch)
        if tok_a is None:
            _fail("policy.modulation.prepare_memory", "returned None")
        else:
            _pass("policy.modulation.prepare_memory", f"tokens={tuple(tok_a.shape)}")
        batch_zero = dict(batch)
        batch_zero["static_image_emb"] = torch.zeros_like(batch["static_image_emb"])
        tok_b, _ = policy.model.prepare_memory(batch_zero)
        if tok_a is not None and tok_b is not None:
            delta = (tok_a.float() - tok_b.float()).abs().mean().item()
            if delta > 1e-6:
                _pass("policy.modulation.memory_sensitivity", f"mean_abs_delta={delta:.6g}")
            else:
                _fail("policy.modulation.memory_sensitivity", f"delta={delta} (memory unused?)")
        try_train_step("policy.modulation", policy, batch)
        del policy
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        _fail("policy.modulation", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    # expert
    try:
        policy = build_policy("expert")
        batch = batch_from_item(
            policy, item, ["static_image_emb", "static_pos_emb", "static_state_emb", "static_mask"]
        )
        tok, _mask = policy.model.prepare_memory(batch)
        if tok is None:
            _fail("policy.expert.prepare_memory", "returned None")
        else:
            _pass("policy.expert.prepare_memory", f"tokens={tuple(tok.shape)}")
        try_train_step("policy.expert", policy, batch)
        del policy
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        _fail("policy.expert", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    # tokendrop + modul
    try:
        ds_td = _make_dataset(feature_root, "tokendrop", episode)
        item_td = ds_td[min(200, len(ds_td) - 1)]
        policy = build_policy("modulation", sampling="tokendrop", tpi=64)
        batch = batch_from_item(
            policy, item_td, ["static_image_emb", "static_pos_emb", "static_state_emb", "static_mask"]
        )
        tok, mask = policy.model.prepare_memory(batch)
        if tok is None or mask is None:
            _fail("policy.tokendrop_modul.prepare_memory", "None")
        else:
            _pass(
                "policy.tokendrop_modul.prepare_memory",
                f"tokens={tuple(tok.shape)} active={int(mask.sum())}",
            )
        try_train_step("policy.tokendrop_modul", policy, batch)
        del policy
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        _fail("policy.tokendrop_modul", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    # recurrent + modul
    try:
        ds_r = _make_dataset(
            feature_root,
            "recurrent",
            episode,
            recurrent_max_recur_steps=8,
            recurrent_max_pretraj_steps=40,
        )
        item_r = ds_r[min(200, len(ds_r) - 1)]
        policy = build_policy("modulation", family="robomme_recurrent", tpi=64)
        batch = batch_from_item(
            policy, item_r, ["recur_image_emb", "recur_pos_emb", "recur_state_emb", "recur_mask"]
        )
        tok, _mask = policy.model.prepare_memory(batch)
        if tok is None:
            _fail("policy.recurrent_modul.prepare_memory", "None")
        else:
            _pass("policy.recurrent_modul.prepare_memory", f"tokens={tuple(tok.shape)}")
        try_train_step("policy.recurrent_modul", policy, batch)
        del policy
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        _fail("policy.recurrent_modul", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


def check_online_buffer_vs_offline_idea() -> None:
    """CPU: tokendrop heap keeps first-frame tokens (RoboMME idea)."""
    from lerobot.policies.smolvla.memory_buffer import SmolVLAPerceptualMemoryBuffer

    buf = SmolVLAPerceptualMemoryBuffer(
        num_frames=16,
        stride=8,
        token_per_image=64,
        token_budget=128,
        pos_dim=256,
        sampling_type="tokendrop",
    )
    # synthesize appends via internal score API
    for step in range(0, 33, 8):
        pixels = torch.randn(3, 64, 3)
        if step > 0:
            pixels = pixels + 0.5  # ensure diffs
        # minimal fake frame list for prev pixels
        from lerobot.policies.smolvla.memory_buffer import _MemoryFrame

        frame = _MemoryFrame(
            image_emb=torch.zeros(1, 192, 8),
            pos_emb=torch.zeros(1, 192, 256),
            state_emb=torch.zeros(1, 192, 14),
            mask=torch.ones(1, 192, dtype=torch.bool),
            timestamp=float(step),
            append_index=step,
            image_pixels=pixels,
        )
        buf.frames.append(frame)
        buf._process_token_drop_score(pixels, step)
        buf.append_count = step + 1
    kept_frames = {item[1] for item in buf.scored_token_heap}
    if 0 in kept_frames:
        _pass("online.tokendrop_keeps_frame0", f"heap_frames={sorted(kept_frames)[:8]}...")
    else:
        _fail("online.tokendrop_keeps_frame0", f"heap_frames={sorted(kept_frames)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path("/pfs/user/wangruiyu/outputs/smolvla_memorybench_features_smoke_framesamp_4x4"),
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-dir", default="/pfs/user/wangruiyu/models/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--strict-warn", action="store_true")
    args = parser.parse_args()

    print("===== SmolVLA memory coverage checks =====", flush=True)
    print(f"feature_root={args.feature_root}", flush=True)

    check_sidecar_layout(args.feature_root, args.episode)
    check_loader_framesamp(args.feature_root, args.episode)
    check_loader_tokendrop(args.feature_root, args.episode)
    check_loader_recurrent(args.feature_root, args.episode)
    check_framesamp_missing_flat_alias(args.feature_root, args.episode)
    check_tokendrop_empty_kept(args.feature_root, args.episode)
    check_online_buffer_vs_offline_idea()

    if args.skip_gpu:
        _record("INFO", "gpu", "skipped")
    else:
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            _fail("gpu.available", "cuda requested but unavailable")
        else:
            check_policy_wiring(args.device, args.model_dir, args.feature_root, args.episode)

    n_pass = sum(1 for s, _, _ in RESULTS if s == "PASS")
    n_fail = sum(1 for s, _, _ in RESULTS if s == "FAIL")
    n_warn = sum(1 for s, _, _ in RESULTS if s == "WARN")
    print(f"===== DONE pass={n_pass} fail={n_fail} warn={n_warn} =====", flush=True)
    if n_fail > 0:
        return 1
    if args.strict_warn and n_warn > 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
