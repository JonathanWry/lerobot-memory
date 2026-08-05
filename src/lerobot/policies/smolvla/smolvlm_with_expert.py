# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from typing import TYPE_CHECKING

import torch
from torch import nn

from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForImageTextToText,
        AutoProcessor,
        DynamicCache,
        SmolVLMForConditionalGeneration,
    )
else:
    AutoConfig = None
    AutoModel = None
    AutoModelForImageTextToText = None
    AutoProcessor = None
    DynamicCache = None
    SmolVLMForConditionalGeneration = None


def apply_rope(x, positions, max_wavelength=10_000):
    """
    Applies RoPE positions [B, L] to x [B, L, H, D].
    """
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)

    radians = radians[..., None, :]

    sin = torch.sin(radians)  # .to(dtype=dtype)
    cos = torch.cos(radians)  # .to(dtype=dtype)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)



# ##modified by wangruiyu to adapt memory mechanism
class MemoryRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(hidden_size))
        self.modulation = nn.Linear(hidden_size, hidden_size * 2)
        nn.init.normal_(self.modulation.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        dtype = x.dtype
        normed = x * torch.rsqrt(x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normed = normed.to(dtype=dtype)
        if cond is None:
            return normed * (1 + self.scale.to(dtype=dtype))

        scale, shift = self.modulation(cond.to(dtype=self.modulation.weight.dtype)).chunk(2, dim=-1)
        return normed * (1 + scale.to(dtype=dtype)) + shift.to(dtype=dtype)


class MemoryAttention(nn.Module):
    def __init__(
        self,
        width: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        attention_bias: bool = False,
    ):
        super().__init__()
        if num_heads % num_key_value_heads != 0:
            raise ValueError("memory_attention_heads must be divisible by memory_key_value_heads")
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.kv_groups = num_heads // num_key_value_heads
        self.scale = head_dim**-0.5
        self.input_norm = MemoryRMSNorm(width)
        self.mem_norm = MemoryRMSNorm(width)
        self.q_proj = nn.Linear(width, num_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(width, num_key_value_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(width, num_key_value_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_heads * head_dim, width, bias=attention_bias)

    def forward(
        self,
        x: torch.Tensor,
        mem_seq: torch.Tensor | None,
        mem_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if mem_seq is None or mem_mask is None:
            return torch.zeros_like(x)

        batch_size, target_len, _ = x.shape
        mem_len = mem_seq.shape[1]
        # ##modified by wangruiyu to adapt memory mechanism
        # SmolVLA expert streams often run in bf16 while these memory projections are fp32.
        # Match the projection dtype for Linear/ROPE/attention math, then cast back at the output.
        proj_dtype = self.q_proj.weight.dtype
        x_norm = self.input_norm(x).to(dtype=proj_dtype)
        mem_norm = self.mem_norm(mem_seq).to(dtype=proj_dtype)
        # ##modified by wangruiyu to adapt memory mechanism

        q = self.q_proj(x_norm).view(batch_size, target_len, self.num_heads, self.head_dim)
        k = self.k_proj(mem_norm).view(batch_size, mem_len, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(mem_norm).view(batch_size, mem_len, self.num_key_value_heads, self.head_dim)

        q_positions = torch.arange(mem_len, mem_len + target_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        k_positions = torch.arange(mem_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        q = apply_rope(q, q_positions)
        k = apply_rope(k, k_positions)

        k = k[:, :, :, None, :].expand(batch_size, mem_len, self.num_key_value_heads, self.kv_groups, self.head_dim)
        v = v[:, :, :, None, :].expand(batch_size, mem_len, self.num_key_value_heads, self.kv_groups, self.head_dim)
        k = k.reshape(batch_size, mem_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, mem_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2).to(torch.float32)
        k = k.transpose(1, 2).to(torch.float32)
        v = v.transpose(1, 2)
        logits = torch.matmul(q, k.transpose(2, 3)) * self.scale
        mask = mem_mask[:, None, None, :].to(device=logits.device, dtype=torch.bool)
        logits = torch.where(mask, logits, torch.finfo(logits.dtype).min)
        probs = torch.softmax(logits, dim=-1).to(dtype=v.dtype)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).reshape(batch_size, target_len, self.num_heads * self.head_dim)
        return self.o_proj(out.to(dtype=self.o_proj.weight.dtype)).to(dtype=x.dtype)


class MemoryIntegrationBlock(nn.Module):
    """General memory integration point for SmolVLA expert/action stream."""

    def __init__(
        self,
        mode: str,
        width: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        attention_bias: bool = False,
    ):
        super().__init__()
        self.mode = mode
        if mode in {"context", "modulation", "expert"}:
            self.memory_attention = None
            self.memory_rms_norm_ffn = None
        else:
            raise ValueError(f"Unsupported memory integration mode: {mode!r}")

    def forward(
        self,
        stream_index: int,
        # ##modified by wangruiyu to adapt memory mechanism
        stream_role: str | None,
        # ##modified by wangruiyu to adapt memory mechanism
        hidden_states: torch.Tensor,
        memory_tokens: torch.Tensor | None,
        memory_mask: torch.Tensor | None,
        memory_attention: MemoryAttention | None = None,
        memory_rms_norm_ffn: MemoryRMSNorm | None = None,
    ) -> torch.Tensor:
        if self.mode == "context":
            return hidden_states
        if self.mode == "expert":
            return hidden_states
        if self.mode == "modulation":
            # ##modified by wangruiyu to adapt memory mechanism
            # Native/modulation SmolVLA keeps [prefix, action]. If a future caller provides roles,
            # use role=="action"; otherwise preserve the original action stream index 1.
            is_action_stream = stream_role == "action" if stream_role is not None else stream_index == 1
            if not is_action_stream or memory_tokens is None or memory_mask is None:
                # ##modified by wangruiyu to adapt memory mechanism
                return hidden_states
            if memory_attention is None or memory_rms_norm_ffn is None:
                raise RuntimeError("modulation memory block requires memory attention and conditional RMSNorm modules")
            # ##modified by wangruiyu to adapt memory mechanism
            # RoboMME-style path kept intact: memory attention produces a conditional vector,
            # then MemoryRMSNorm applies conditional scale/shift to the expert/action stream.
            mem_mod_vec = memory_attention(hidden_states, memory_tokens, memory_mask)
            return memory_rms_norm_ffn(hidden_states, mem_mod_vec)
            # ##modified by wangruiyu to adapt memory mechanism
        return hidden_states
# ##modified by wangruiyu to adapt memory mechanism


def get_intermediate_size(hidden_dim, ffn_dim_multiplier=4, multiple_of=256):
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


class SmolVLMWithExpertModel(nn.Module):
    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        load_vlm_weights: bool = True,
        train_expert_only: bool = True,
        freeze_vision_encoder: bool = False,
        attention_mode: str = "self_attn",
        num_expert_layers: int = -1,
        num_vlm_layers: int = -1,
        self_attn_every_n_layers: int = -1,
        expert_width_multiplier: float = 0.5,
        # ##modified by wangruiyu to adapt memory mechanism
        memory_integration: str = "context",
        memory_attention_heads: int = 15,
        memory_key_value_heads: int = 5,
        memory_head_dim: int = 64,
        # ##modified by wangruiyu to adapt memory mechanism
        device: str = "auto",
    ):
        super().__init__()
        require_package("transformers", extra="smolvla")
        if load_vlm_weights:
            print(f"Loading  {model_id} weights ...")
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype="bfloat16",
                low_cpu_mem_usage=True,
            )
            config = self.vlm.config
        else:
            config = AutoConfig.from_pretrained(model_id)
            self.vlm = SmolVLMForConditionalGeneration(config=config)
        self.processor = AutoProcessor.from_pretrained(model_id)
        if num_vlm_layers > 0:
            print(f"Reducing the number of VLM layers to {num_vlm_layers} ...")
            self.get_vlm_model().text_model.layers = self.get_vlm_model().text_model.layers[:num_vlm_layers]
        self.num_vlm_layers = len(self.get_vlm_model().text_model.layers)
        self.config = config
        # Smaller lm expert
        lm_expert_config = copy.deepcopy(config.text_config)
        hidden_size = lm_expert_config.hidden_size
        lm_expert_config.hidden_size = int(hidden_size * expert_width_multiplier)  # hidden_size // 2
        lm_expert_config.intermediate_size = get_intermediate_size(int(hidden_size * expert_width_multiplier))
        lm_expert_config.num_hidden_layers = self.num_vlm_layers
        if num_expert_layers > 0:
            assert len(self.get_vlm_model().text_model.layers) % num_expert_layers == 0, (
                f"Number of layers in the VLM {len(self.get_vlm_model().text_model.layers)} are not multiple of num_expert_layers {num_expert_layers}"
            )
            lm_expert_config.num_hidden_layers = num_expert_layers
        self.lm_expert = AutoModel.from_config(lm_expert_config)
        # ##modified by wangruiyu to adapt memory mechanism
        self.lm_memory_expert = AutoModel.from_config(copy.deepcopy(lm_expert_config)) if memory_integration == "expert" else None
        # ##modified by wangruiyu to adapt memory mechanism

        self.num_expert_layers = len(self.lm_expert.layers)
        self.self_attn_every_n_layers = self_attn_every_n_layers
        if "cross" in attention_mode:
            # Reshape qkv projections to have the same input dimension as the vlm
            # ##modified by wangruiyu to adapt memory mechanism
            # The action expert cross-attends VLM KV states, so its K/V projections must accept
            # VLM head dimensions. The memory expert self-attends memory tokens in expert hidden
            # space and must keep its native expert-width K/V projections.
            for expert_model in [self.lm_expert]:
                if expert_model is None:
                    continue
                for layer_idx in range(len(expert_model.layers)):
                    if self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0:
                        continue
                    expert_model.layers[layer_idx].self_attn.k_proj = nn.Linear(
                        config.text_config.num_key_value_heads * config.text_config.head_dim,
                        lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,
                        bias=lm_expert_config.attention_bias,
                    )
                    expert_model.layers[layer_idx].self_attn.v_proj = nn.Linear(
                        config.text_config.num_key_value_heads * config.text_config.head_dim,
                        lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,
                        bias=lm_expert_config.attention_bias,
                    )
            # ##modified by wangruiyu to adapt memory mechanism
        # Remove unused embed_tokens
        self.lm_expert.embed_tokens = None
        # ##modified by wangruiyu to adapt memory mechanism
        if self.lm_memory_expert is not None:
            self.lm_memory_expert.embed_tokens = None
        # ##modified by wangruiyu to adapt memory mechanism

        self.num_attention_heads = self.config.text_config.num_attention_heads
        self.num_key_value_heads = self.config.text_config.num_key_value_heads

        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_mode = attention_mode
        self.expert_hidden_size = lm_expert_config.hidden_size
        # ##modified by wangruiyu to adapt memory mechanism
        self.memory_integration = memory_integration
        if self.memory_integration == "modulation":
            self.memory_attentions = nn.ModuleList(
                [
                    MemoryAttention(
                        width=self.expert_hidden_size,
                        num_heads=memory_attention_heads,
                        num_key_value_heads=memory_key_value_heads,
                        head_dim=memory_head_dim,
                        attention_bias=lm_expert_config.attention_bias,
                    )
                    for _ in range(self.num_vlm_layers)
                ]
            )
            self.memory_rms_norm_ffns = nn.ModuleList(
                [MemoryRMSNorm(self.expert_hidden_size) for _ in range(self.num_vlm_layers)]
            )
        else:
            self.memory_attentions = None
            self.memory_rms_norm_ffns = None
        if self.memory_integration in {"modulation", "expert"}:
            self.memory_integration_blocks = nn.ModuleList(
                [
                    MemoryIntegrationBlock(
                        mode=self.memory_integration,
                        width=self.expert_hidden_size,
                        num_heads=memory_attention_heads,
                        num_key_value_heads=memory_key_value_heads,
                        head_dim=memory_head_dim,
                        attention_bias=lm_expert_config.attention_bias,
                    )
                    for _ in range(self.num_vlm_layers)
                ]
            )
        else:
            self.memory_integration_blocks = None
        # ##modified by wangruiyu to adapt memory mechanism
        self.set_requires_grad()

    def get_vlm_model(self):
        return self.vlm.model

    def set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()
            for params in self.get_vlm_model().vision_model.parameters():
                params.requires_grad = False
        if self.train_expert_only:
            self.vlm.eval()
            for params in self.vlm.parameters():
                params.requires_grad = False
        else:
            # To avoid unused params issue with distributed training
            last_layers = [self.num_vlm_layers - 1]
            if (
                self.num_vlm_layers != self.num_expert_layers
                and self.num_vlm_layers % self.num_expert_layers == 0
            ):
                last_layers.append(self.num_vlm_layers - 2)
            frozen_layers = [
                "lm_head",
                "text_model.model.norm.weight",
            ]
            for layer in last_layers:
                frozen_layers.append(f"text_model.model.layers.{layer}.")

            for name, params in self.vlm.named_parameters():
                if any(k in name for k in frozen_layers):
                    params.requires_grad = False
        # To avoid unused params issue with distributed training
        # ##modified by wangruiyu to adapt memory mechanism
        for expert_model in [self.lm_expert, self.lm_memory_expert]:
            if expert_model is None:
                continue
            for name, params in expert_model.named_parameters():
                if "lm_head" in name:
                    params.requires_grad = False
        # ##modified by wangruiyu to adapt memory mechanism

    def train(self, mode: bool = True):
        super().train(mode)

        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()

        if self.train_expert_only:
            self.vlm.eval()

    def embed_image(self, image: torch.Tensor):
        patch_attention_mask = None
        # Get sequence from the vision encoder
        image_hidden_states = (
            self.get_vlm_model()
            .vision_model(
                pixel_values=image.to(dtype=self.get_vlm_model().vision_model.dtype),
                patch_attention_mask=patch_attention_mask,
            )
            .last_hidden_state
        )
        # Modality projection & resampling
        image_hidden_states = self.get_vlm_model().connector(image_hidden_states)
        return image_hidden_states

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.get_vlm_model().text_model.get_input_embeddings()(tokens)

    def forward_attn_layer(
        self,
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
        use_cache: bool = True,
        past_key_values: "DynamicCache | None" = None,
    ) -> "tuple[list[torch.Tensor], DynamicCache | None]":
        query_states = []
        key_states = []
        value_states = []
        for i, hidden_states in enumerate(inputs_embeds):
            layer = model_layers[i][layer_idx]
            if hidden_states is None or layer is None:
                continue
            hidden_states = layer.input_layernorm(hidden_states)

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            query_states.append(query_state)
            key_states.append(key_state)
            value_states.append(value_state)

        # B,L,H,D with L sequence length, H number of heads, D head dim
        # concatenate on the number of embeddings/tokens
        query_states = torch.cat(query_states, dim=1)
        key_states = torch.cat(key_states, dim=1)
        value_states = torch.cat(value_states, dim=1)
        seq_len = query_states.shape[1]
        if seq_len < position_ids.shape[1]:
            _position_ids = position_ids[:, :seq_len]
            _attention_mask = attention_mask[:, :seq_len, :seq_len]
        else:
            _position_ids = position_ids
            _attention_mask = attention_mask

        attention_mask_ = _attention_mask
        position_ids_ = _position_ids

        query_states = apply_rope(query_states, position_ids_)
        key_states = apply_rope(key_states, position_ids_)

        if use_cache:
            # `DynamicCache` stores tensors as [batch, heads, seq, head_dim]; this module works with
            # [batch, seq, heads, head_dim]. During prefix prefill this stores the (post-RoPE) K/V and
            # returns them unchanged; during denoising it appends the suffix K/V and returns
            # [prefix; suffix], exactly like the previous hand-rolled dict cache.
            key_states, value_states = past_key_values.update(
                key_states.transpose(1, 2), value_states.transpose(1, 2), layer_idx
            )
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

        attention_interface = self.get_attention_interface()

        att_output = attention_interface(
            attention_mask_, batch_size, head_dim, query_states, key_states, value_states
        )
        return [att_output], past_key_values

    def forward_cross_attn_layer(
        self,
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
        use_cache: bool = True,
        past_key_values: "DynamicCache | None" = None,
    ) -> "tuple[list[torch.Tensor], DynamicCache | None]":
        attention_interface = self.get_attention_interface()

        att_outputs = []
        assert len(inputs_embeds) >= 2 or (use_cache and past_key_values is not None), (
            f"Both len(inputs_embeds) == {len(inputs_embeds)} and past_key_values is {past_key_values}"
        )

        stream_row_starts = []
        cursor = 0
        for hidden_states in inputs_embeds:
            stream_row_starts.append(cursor)
            if hidden_states is not None:
                cursor += hidden_states.shape[1]

        if inputs_embeds[0] is not None and not past_key_values:
            # Prefix attention
            seq_len = inputs_embeds[0].shape[1]
            position_id = position_ids[:, :seq_len]
            prefix_attention_mask = attention_mask[:, :seq_len, :seq_len]

            layer = model_layers[0][layer_idx]

            hidden_states = layer.input_layernorm(inputs_embeds[0])

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_states = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            # B,L,H,D with L sequence length, H number of heads, D head dim
            query_states = apply_rope(query_state, position_id)
            key_states = apply_rope(key_state, position_id)

            att_output = attention_interface(
                prefix_attention_mask, batch_size, head_dim, query_states, key_states, value_states
            )
            att_outputs.append(att_output)
        elif inputs_embeds[0] is None:
            att_outputs.append(None)

        if use_cache and past_key_values is not None:
            # Cross-attention layers never fill the cache themselves: during the prefix prefill every
            # layer goes through `forward_attn_layer`, which stores the (post-RoPE) VLM K/V for this
            # layer index. Here we only read them back (no concatenation: the expert cross-attends to
            # the fixed prefix). `DynamicCache` stores [batch, heads, seq, head_dim]; transpose back.
            key_states = past_key_values.layers[layer_idx].keys.transpose(1, 2)
            value_states = past_key_values.layers[layer_idx].values.transpose(1, 2)

        # Expert streams: action expert is stream 1; memory expert, when enabled, is stream 2.
        # ##modified by wangruiyu to adapt memory mechanism
        for stream_idx in range(1, len(inputs_embeds)):
            expert_layer = model_layers[stream_idx][layer_idx]
            expert_inputs = inputs_embeds[stream_idx]
            if expert_layer is None or expert_inputs is None:
                att_outputs.append(None)
                continue

            expert_hidden_states = expert_layer.input_layernorm(expert_inputs)

            expert_input_shape = expert_hidden_states.shape[:-1]
            expert_hidden_shape = (*expert_input_shape, -1, expert_layer.self_attn.head_dim)

            expert_hidden_states = expert_hidden_states.to(dtype=expert_layer.self_attn.q_proj.weight.dtype)
            expert_query_state = expert_layer.self_attn.q_proj(expert_hidden_states).view(expert_hidden_shape)

            # reshape (not view): K/V read back from the cache are transposed, hence non-contiguous
            _key_states = key_states.to(dtype=expert_layer.self_attn.k_proj.weight.dtype).reshape(
                *key_states.shape[:2], -1
            )
            expert_key_states = expert_layer.self_attn.k_proj(_key_states).view(
                *_key_states.shape[:-1], -1, expert_layer.self_attn.head_dim
            )  # k_proj should have same dim as kv

            _value_states = value_states.to(dtype=expert_layer.self_attn.v_proj.weight.dtype).reshape(
                *value_states.shape[:2], -1
            )
            expert_value_states = expert_layer.self_attn.v_proj(_value_states).view(
                *_value_states.shape[:-1], -1, expert_layer.self_attn.head_dim
            )

            row_start = stream_row_starts[stream_idx]
            row_end = row_start + expert_inputs.shape[1]
            expert_position_id = position_ids[:, row_start:row_end]
            expert_position_id = (
                expert_position_id - torch.min(expert_position_id, dim=1, keepdim=True).values
            )  # start from 0
            expert_attention_mask = attention_mask[:, row_start:row_end, : expert_key_states.shape[1]]

            expert_query_states = apply_rope(expert_query_state, expert_position_id)

            att_output = attention_interface(
                expert_attention_mask,
                batch_size,
                head_dim,
                expert_query_states,
                expert_key_states,
                expert_value_states,
            )
            att_outputs.append(att_output)
        # ##modified by wangruiyu to adapt memory mechanism

        # att_output = att_output.to(dtype=models[i].dtype)
        return att_outputs, past_key_values

    # ##modified by wangruiyu to adapt memory mechanism
    def forward_memory_expert_cross_attn_layer(
        self,
        model_layers,
        inputs_embeds,
        stream_roles,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
    ) -> "tuple[list[torch.Tensor | None], DynamicCache | None]":
        """RoboMME-style expert cross layer for [memory, prefix, action] streams."""

        attention_interface = self.get_attention_interface()
        stream_row_starts = []
        cursor = 0
        for hidden_states in inputs_embeds:
            stream_row_starts.append(cursor)
            if hidden_states is not None:
                cursor += hidden_states.shape[1]

        att_outputs: list[torch.Tensor | None] = [None for _ in inputs_embeds]
        source_key_states = []
        source_value_states = []
        source_indices = []

        for stream_idx, (hidden_states, role) in enumerate(zip(inputs_embeds, stream_roles, strict=True)):
            layer = model_layers[stream_idx][layer_idx]
            if hidden_states is None or layer is None:
                continue

            row_start = stream_row_starts[stream_idx]
            row_end = row_start + hidden_states.shape[1]
            stream_position_ids = position_ids[:, row_start:row_end]
            normed = layer.input_layernorm(hidden_states).to(dtype=layer.self_attn.q_proj.weight.dtype)
            input_shape = normed.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            query_state = layer.self_attn.q_proj(normed).view(hidden_shape)
            query_state = apply_rope(query_state, stream_position_ids)

            if role in {"memory", "prefix"}:
                key_state = layer.self_attn.k_proj(normed).view(hidden_shape)
                value_state = layer.self_attn.v_proj(normed).view(hidden_shape)
                key_state = apply_rope(key_state, stream_position_ids)
                stream_attention_mask = attention_mask[:, row_start:row_end, row_start:row_end]
                att_outputs[stream_idx] = attention_interface(
                    stream_attention_mask,
                    batch_size,
                    head_dim,
                    query_state,
                    key_state,
                    value_state,
                )
                source_key_states.append(key_state)
                source_value_states.append(value_state)
                source_indices.append(stream_idx)
            elif role != "action":
                raise ValueError(f"Unsupported memory expert stream role: {role!r}")

        if not source_key_states:
            raise RuntimeError("memory expert cross layer requires memory/prefix source streams")

        src_key_states = torch.cat(source_key_states, dim=1)
        src_value_states = torch.cat(source_value_states, dim=1)
        src_cols = []
        for source_idx in source_indices:
            source_start = stream_row_starts[source_idx]
            source_len = inputs_embeds[source_idx].shape[1]
            src_cols.extend(range(source_start, source_start + source_len))
        src_cols_tensor = torch.tensor(src_cols, device=attention_mask.device, dtype=torch.long)

        for stream_idx, (hidden_states, role) in enumerate(zip(inputs_embeds, stream_roles, strict=True)):
            if role != "action":
                continue
            layer = model_layers[stream_idx][layer_idx]
            if hidden_states is None or layer is None:
                continue

            row_start = stream_row_starts[stream_idx]
            row_end = row_start + hidden_states.shape[1]
            action_position_ids = position_ids[:, row_start:row_end]
            normed = layer.input_layernorm(hidden_states).to(dtype=layer.self_attn.q_proj.weight.dtype)
            input_shape = normed.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            query_state = layer.self_attn.q_proj(normed).view(hidden_shape)
            query_state = apply_rope(query_state, action_position_ids)

            flattened_key_states = src_key_states.to(dtype=layer.self_attn.k_proj.weight.dtype).reshape(
                *src_key_states.shape[:2], -1
            )
            action_key_states = layer.self_attn.k_proj(flattened_key_states).view(
                *flattened_key_states.shape[:-1], -1, layer.self_attn.head_dim
            )
            flattened_value_states = src_value_states.to(dtype=layer.self_attn.v_proj.weight.dtype).reshape(
                *src_value_states.shape[:2], -1
            )
            action_value_states = layer.self_attn.v_proj(flattened_value_states).view(
                *flattened_value_states.shape[:-1], -1, layer.self_attn.head_dim
            )

            action_attention_mask = attention_mask[:, row_start:row_end, :].index_select(2, src_cols_tensor)
            att_outputs[stream_idx] = attention_interface(
                action_attention_mask,
                batch_size,
                head_dim,
                query_state,
                action_key_states,
                action_value_states,
            )

        return att_outputs, None
    # ##modified by wangruiyu to adapt memory mechanism

    def get_model_layers(self, models: list) -> list:
        layers_by_stream = [[] for _ in models]
        multiple_of = self.num_vlm_layers // self.num_expert_layers
        for i in range(self.num_vlm_layers):
            for model, layers in zip(models, layers_by_stream, strict=True):
                if model is None:
                    layers.append(None)
                    continue
                if len(model.layers) == self.num_vlm_layers:
                    layers.append(model.layers[i])
                    continue
                if multiple_of > 0 and i > 0 and i % multiple_of != 0:
                    layers.append(None)
                    continue
                expert_layer_index = i // multiple_of if multiple_of > 0 else i
                layers.append(model.layers[expert_layer_index])
        return layers_by_stream

    # ##modified by wangruiyu to adapt memory mechanism
    def _base_stream_roles(self, inputs_embeds: list[torch.Tensor | None]) -> list[str]:
        if len(inputs_embeds) != 2:
            raise ValueError(
                f"SmolVLA native forward expects [prefix, action] streams before memory integration, got {len(inputs_embeds)}"
            )
        return ["prefix", "action"]

    def _memory_expert_is_active(self, stream_roles: list[str]) -> bool:
        return self.memory_integration == "expert" and "memory" in stream_roles

    def _select_stream_models(self, stream_roles: list[str]) -> list[nn.Module | None]:
        role_to_model = {
            "memory": self.lm_memory_expert,
            "prefix": self.get_vlm_model().text_model,
            "action": self.lm_expert,
        }
        models = []
        for role in stream_roles:
            if role not in role_to_model:
                raise ValueError(f"Unsupported SmolVLA stream role: {role!r}")
            model = role_to_model[role]
            if model is None:
                raise RuntimeError(f"Stream role {role!r} requires a model, but it is not initialized")
            models.append(model)
        return models

    def _build_memory_expert_attention_inputs(
        self,
        inputs_embeds: list[torch.Tensor | None],
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        memory_tokens: torch.Tensor,
        memory_mask: torch.Tensor | None,
    ) -> tuple[list[torch.Tensor | None], torch.LongTensor, torch.Tensor, list[str]]:
        memory_tokens = memory_tokens.to(device=position_ids.device)
        batch_size, memory_len = memory_tokens.shape[:2]
        if memory_mask is None:
            memory_mask = torch.ones(batch_size, memory_len, device=position_ids.device, dtype=torch.bool)
        else:
            memory_mask = memory_mask.to(device=position_ids.device, dtype=torch.bool)

        memory_position_ids = torch.arange(memory_len, device=position_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_ids = torch.cat([memory_position_ids, position_ids + memory_len], dim=1)

        old_len = attention_mask.shape[-1]
        new_len = old_len + memory_len
        new_attention_mask = torch.zeros(
            batch_size,
            new_len,
            new_len,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        prefix_len = inputs_embeds[0].shape[1]
        old_valid = attention_mask.any(dim=1) | attention_mask.any(dim=2)
        memory_valid = memory_mask.to(device=attention_mask.device)
        old_valid = old_valid.to(device=attention_mask.device)
        valid = torch.cat([memory_valid, old_valid], dim=1)
        new_attention_mask[:, :memory_len, :memory_len] = memory_valid[:, :, None] & memory_valid[:, None, :]
        # RoboMME-style expert visibility:
        # memory attends memory; VLM/prefix does not attend memory; action/suffix attends memory.
        new_attention_mask[:, memory_len + prefix_len :, :memory_len] = memory_valid[:, None, :]
        new_attention_mask[:, memory_len:, memory_len:] = attention_mask
        new_attention_mask = new_attention_mask & valid[:, :, None]

        # RoboMME expert order is [memory expert, VLM/prefix, action expert].
        return [memory_tokens, *inputs_embeds], position_ids, new_attention_mask, ["memory", "prefix", "action"]

    def _append_memory_expert_stream(
        self,
        inputs_embeds: list[torch.Tensor | None],
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        memory_tokens: torch.Tensor | None,
        memory_mask: torch.Tensor | None,
        use_cache: bool | None,
        past_key_values: "DynamicCache | None",
    ) -> tuple[list[torch.Tensor | None], torch.LongTensor, torch.Tensor, list[str]]:
        stream_roles = self._base_stream_roles(inputs_embeds)
        if self.memory_integration != "expert" or memory_tokens is None:
            return inputs_embeds, position_ids, attention_mask, stream_roles
        if use_cache or past_key_values is not None:
            raise NotImplementedError(
                "SmolVLA memory expert integration currently supports use_cache=False only."
            )
        if len(inputs_embeds) != 2:
            return inputs_embeds, position_ids, attention_mask, stream_roles

        return self._build_memory_expert_attention_inputs(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            memory_tokens=memory_tokens,
            memory_mask=memory_mask,
        )
    # ##modified by wangruiyu to adapt memory mechanism

    # ##modified by wangruiyu to adapt memory mechanism
    def _externalize_stream_outputs(
        self,
        outputs_embeds: list[torch.Tensor | None],
        past_key_values: "DynamicCache | None",
        stream_roles: list[str],
    ) -> tuple[list[torch.Tensor | None], "DynamicCache | None"]:
        if self._memory_expert_is_active(stream_roles):
            # Hide the internal memory expert stream from callers. SmolVLA callers still receive
            # [prefix_out, action_out], and the action head continues to consume the action stream.
            role_to_output = {role: output for role, output in zip(stream_roles, outputs_embeds, strict=True)}
            return [role_to_output["prefix"], role_to_output["action"]], past_key_values
        return outputs_embeds, past_key_values
    # ##modified by wangruiyu to adapt memory mechanism

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: "DynamicCache | None" = None,
        inputs_embeds: list[torch.FloatTensor] = None,
        use_cache: bool | None = None,
        # ##modified by wangruiyu to adapt memory mechanism
        memory_tokens: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        # ##modified by wangruiyu to adapt memory mechanism
    ):
        # ##modified by wangruiyu to adapt memory mechanism
        inputs_embeds, position_ids, attention_mask, stream_roles = self._append_memory_expert_stream(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            memory_tokens=memory_tokens,
            memory_mask=memory_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        models = self._select_stream_models(stream_roles)
        expert_memory_active = self._memory_expert_is_active(stream_roles)
        # ##modified by wangruiyu to adapt memory mechanism
        model_layers = self.get_model_layers(models)
        for hidden_states in inputs_embeds:
            # TODO this is very inefficient
            # dtype is always the same, batch size too (if > 1 len)
            # device could be trickier in multi gpu edge cases but that's it
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]

        # Prefix prefill: no cache was passed, so create one and fill it (every layer runs
        # self-attention over the prefix). When a filled cache is passed (denoising), layers
        # read from it instead.
        fill_kv_cache = use_cache and past_key_values is None
        if fill_kv_cache:
            past_key_values = DynamicCache()

        # RMSNorm
        num_layers = self.num_vlm_layers
        head_dim = self.vlm.config.text_config.head_dim
        for layer_idx in range(num_layers):
            # ##modified by wangruiyu to adapt memory mechanism
            memory_expert_cross_layer = (
                expert_memory_active
                and "cross" in self.attention_mode
                and not fill_kv_cache
                and not (
                    self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0
                )
            )
            # ##modified by wangruiyu to adapt memory mechanism
            if (
                fill_kv_cache
                or "cross" not in self.attention_mode
                or (self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0)
            ):
                att_outputs, past_key_values = self.forward_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    past_key_values=past_key_values,
                )
            # ##modified by wangruiyu to adapt memory mechanism
            elif memory_expert_cross_layer:
                att_outputs, past_key_values = self.forward_memory_expert_cross_attn_layer(
                    model_layers,
                    inputs_embeds,
                    stream_roles,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                )
            # ##modified by wangruiyu to adapt memory mechanism
            else:
                att_outputs, past_key_values = self.forward_cross_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    past_key_values=past_key_values,
                )
            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = model_layers[i][layer_idx]
                att_output = (
                    att_outputs[i] if i < len(att_outputs) else att_outputs[0]
                )  # in case of self_attn
                if hidden_states is not None:
                    if layer is None:
                        outputs_embeds.append(hidden_states)
                        continue
                    end = start + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    att_out = att_output[:, start:end]
                    out_emb = layer.self_attn.o_proj(att_out)

                    out_emb += hidden_states
                    after_first_residual = out_emb.clone()

                    if self.memory_integration_blocks is not None:
                        # ##modified by wangruiyu to adapt memory mechanism
                        out_emb = self.memory_integration_blocks[layer_idx](
                            stream_index=i,
                            stream_role=stream_roles[i],
                            hidden_states=out_emb,
                            memory_tokens=memory_tokens,
                            memory_mask=memory_mask,
                            memory_attention=(
                                self.memory_attentions[layer_idx] if self.memory_attentions is not None else None
                            ),
                            memory_rms_norm_ffn=(
                                self.memory_rms_norm_ffns[layer_idx]
                                if self.memory_rms_norm_ffns is not None
                                else None
                            ),
                        )
                        after_first_residual = out_emb.clone()
                        # ##modified by wangruiyu to adapt memory mechanism
                    out_emb = layer.post_attention_layernorm(out_emb)
                    out_emb = layer.mlp(out_emb)

                    out_emb += after_first_residual

                    outputs_embeds.append(out_emb)

                    start = end if len(att_outputs) == 1 else 0
                else:
                    outputs_embeds.append(None)

            inputs_embeds = outputs_embeds

        # final norm
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)
        # ##modified by wangruiyu to adapt memory mechanism
        return self._externalize_stream_outputs(outputs_embeds, past_key_values, stream_roles)
        # ##modified by wangruiyu to adapt memory mechanism

    def get_attention_interface(self):
        attention_interface = self.eager_attention_forward
        return attention_interface

    def eager_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        num_att_heads = self.num_attention_heads
        num_key_value_heads = self.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads

        sequence_length = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        # Attention here is upcasted to float32 to match the original eager implementation.
        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5

        att_weights = att_weights.to(dtype=torch.float32)
        big_neg = torch.finfo(att_weights.dtype).min  # -2.3819763e38  # See gemma/modules.py
        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

        att_output = att_output.permute(0, 2, 1, 3)
        # we use -1 because sequence length can change
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)

        return att_output
