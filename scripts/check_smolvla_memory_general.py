#!/usr/bin/env python3
"""Logic checks for SmolVLA memory mechanisms without loading a VLM checkpoint.

This is intentionally CPU-only and synthetic-data based. It verifies that the
config surface, RoboMME-style frame sampling/token-drop buffers, recurrent buffer,
and memory-expert attention mask stay internally consistent.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import torch

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.memory_buffer import (
    _MemoryFrame,
    SmolVLAPerceptualMemoryBuffer,
    SmolVLARecurrentMemoryBuffer,
)
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel


def _make_frame(step: int, *, batch: int = 1, tokens: int = 12, emb_dim: int = 8, pos_dim: int = 6, state_dim: int = 32):
    value = float(step + 1)
    return _MemoryFrame(
        image_emb=torch.full((batch, tokens, emb_dim), value),
        pos_emb=torch.full((batch, tokens, pos_dim), value / 10.0),
        state_emb=torch.full((batch, tokens, state_dim), value / 100.0),
        mask=torch.ones((batch, tokens), dtype=torch.bool),
        timestamp=time.monotonic() - step * 0.01,
        append_index=step,
        image_pixels=torch.full((3, 4, 3), value),
    )


def check_config() -> None:
    native = SmolVLAConfig(memory_enabled=False, memory_family="none")
    assert native.observation_delta_indices == [0]

    perceptual = SmolVLAConfig(
        memory_enabled=True,
        memory_family="robomme_perceptual",
        memory_integration="modulation",
        robomme_sampling_type="framesamp",
    )
    assert perceptual.observation_delta_indices == [0]

    explicit = SmolVLAConfig(
        memory_enabled=True,
        memory_family="kai0_explicit",
        visual_memory_frames=4,
        visual_memory_stride=2,
    )
    assert explicit.observation_delta_indices == [-6, -4, -2, 0]

    recurrent = SmolVLAConfig(
        memory_enabled=True,
        memory_family="robomme_recurrent",
        memory_integration="expert",
        recurrent_type="ttt",
        recurrent_hidden_dim=512,
        recurrent_num_attention_heads=8,
        recurrent_num_key_value_heads=1,
        recurrent_ttt_num_heads=2,
    )
    assert recurrent.observation_delta_indices == [0]


def check_framesampling() -> None:
    buffer = SmolVLAPerceptualMemoryBuffer(
        num_frames=16,
        stride=1,
        token_per_image=4,
        token_budget=64,
        pos_dim=6,
        sampling_type="framesamp",
    )
    buffer.tokens_per_frame = 12
    buffer.frames = [_make_frame(i) for i in range(20)]
    image_emb, pos_emb, state_emb, mask, info = buffer.build_batch(torch.device("cpu"))
    assert image_emb.shape == (1, 60, 8)
    assert pos_emb.shape == (1, 60, 6)
    assert state_emb.shape == (1, 60, 32)
    assert mask.shape == (1, 60)
    assert info["effective_K"] == 5
    assert info["indices"] == [0, 4, 9, 14, 19]

    short = SmolVLAPerceptualMemoryBuffer(
        num_frames=16,
        stride=1,
        token_per_image=4,
        token_budget=64,
        pos_dim=6,
        sampling_type="framesamp",
    )
    short.tokens_per_frame = 12
    short.frames = [_make_frame(0), _make_frame(1)]
    _, _, _, short_mask, short_info = short.build_batch(torch.device("cpu"))
    assert short_info["padding_frames"] == 3
    assert short_info["indices"] == [0, 1, -1, -1, -1]
    assert int(short_mask.sum().item()) == 24


def check_tokendrop() -> None:
    buffer = SmolVLAPerceptualMemoryBuffer(
        num_frames=16,
        stride=2,
        token_per_image=4,
        token_budget=5,
        pos_dim=6,
        sampling_type="tokendrop",
    )
    for step in range(4):
        frame = _make_frame(step)
        frame.image_pixels = torch.zeros((3, 4, 3), dtype=torch.float32)
        frame.image_pixels[:, step % 4, :] = float(step + 1)
        buffer.frames.append(frame)
        buffer._process_token_drop_score(frame.image_pixels, step)
        buffer.append_count += 1

    image_emb, pos_emb, state_emb, mask, info = buffer.build_batch(torch.device("cpu"))
    assert image_emb.shape == (1, 5, 8)
    assert pos_emb.shape == (1, 5, 6)
    assert state_emb.shape == (1, 5, 32)
    assert mask.shape == (1, 5)
    assert info["sampling"] == "tokendrop"
    # RoboMME pins first-frame tokens with high score; later changed tokens are still scored at stride.
    assert any(item[1] == 0 for item in buffer.scored_token_heap), buffer.scored_token_heap
    assert buffer.token_drop_last_frame == 3
    assert len(buffer.scored_token_heap) >= buffer.token_budget


def check_recurrent() -> None:
    buffer = SmolVLARecurrentMemoryBuffer(
        max_recur_steps=6,
        input_obs_horizon=4,
        max_pretraj_steps=40,
        token_per_image=4,
        pos_dim=6,
    )
    assert buffer._get_token_recurrent_indices(0) == [0]
    assert buffer._get_token_recurrent_indices(3) == [3]
    assert buffer._get_token_recurrent_indices(8) == [0, 4, 8]

    buffer.frames = [_make_frame(i, tokens=12) for i in range(9)]
    buffer.append_count = 9
    image_emb, pos_emb, state_emb, mask, info = buffer.build_batch(torch.device("cpu"))
    assert image_emb.shape == (1, 6, 3, 4, 8)
    assert pos_emb.shape == (1, 6, 3, 4, 6)
    assert state_emb.shape == (1, 6, 32)
    assert mask.shape == (1, 6)
    assert info["indices"] == [-1, -1, -1, 0, 4, 8]
    assert int(mask.sum().item()) == 3


def check_memory_expert_mask() -> None:
    model = object.__new__(SmolVLMWithExpertModel)
    batch = 1
    mem_len = 3
    prefix_len = 5
    action_len = 4
    inputs = [
        torch.zeros(batch, prefix_len, 8),
        torch.zeros(batch, action_len, 8),
    ]
    position_ids = torch.arange(prefix_len + action_len).unsqueeze(0)
    attention_mask = torch.ones(batch, prefix_len + action_len, prefix_len + action_len, dtype=torch.bool)
    memory_tokens = torch.zeros(batch, mem_len, 8)
    memory_mask = torch.tensor([[True, False, True]])

    new_inputs, new_pos, new_mask, stream_roles = model._build_memory_expert_attention_inputs(
        inputs,
        position_ids,
        attention_mask,
        memory_tokens,
        memory_mask,
    )
    assert len(new_inputs) == 3
    assert stream_roles == ["memory", "prefix", "action"]
    assert new_pos.shape == (batch, mem_len + prefix_len + action_len)
    assert new_mask.shape == (batch, mem_len + prefix_len + action_len, mem_len + prefix_len + action_len)

    # Memory expert can self-attend to valid memory tokens.
    assert bool(new_mask[0, 0, 0])
    assert not bool(new_mask[0, 0, 1])
    # VLM/prefix remains isolated from memory.
    assert not bool(new_mask[0, mem_len, 0])
    # Action stream can attend memory.
    assert bool(new_mask[0, mem_len + prefix_len, 0])
    assert not bool(new_mask[0, mem_len + prefix_len, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", choices=["all", "config", "framesamp", "tokendrop", "recurrent", "expert"], default="all")
    args = parser.parse_args()

    checks = {
        "config": check_config,
        "framesamp": check_framesampling,
        "tokendrop": check_tokendrop,
        "recurrent": check_recurrent,
        "expert": check_memory_expert_mask,
    }
    selected = checks if args.section == "all" else {args.section: checks[args.section]}
    for name, fn in selected.items():
        fn()
        print(f"PASS {name}")
    print("PASS smolvla memory general checks")


if __name__ == "__main__":
    main()
