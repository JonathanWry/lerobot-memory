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

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_IMAGES

from ..rtc.configuration_rtc import RTCConfig


@PreTrainedConfig.register_subclass("smolvla")
@dataclass
class SmolVLAConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    # Add empty images. Used by smolvla_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to relative values with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # Select the VLM backbone.
    load_vlm_weights: bool = False  # Set to False in case of training the expert from scratch. True when init from pretrained SmolVLA weights

    add_image_special_tokens: bool = False  # Whether to use special image tokens around image features.

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"  # "max_length"

    num_expert_layers: int = -1  # Less or equal to 0 is the default where the action expert has the same number of layers of VLM. Otherwise the expert have less layers.
    num_vlm_layers: int = 16  # Number of layers used in the VLM (first num_vlm_layers layers)
    self_attn_every_n_layers: int = 2  # Interleave SA layers each self_attn_every_n_layers
    expert_width_multiplier: float = 0.75  # The action expert hidden size (wrt to the VLM)

    min_period: float = 4e-3  # sensitivity range for the timestep used in sine-cosine positional encoding
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    # Memory configuration. These fields are intentionally flat so they stay easy to
    # override from the CLI and serialize into checkpoints.
    memory_enabled: bool = False
    # none | kai0_explicit | robomme_perceptual | robomme_recurrent
    # kai0_explicit: client/dataset supplies K raw image frames (no sidecar / no memory_buffer).
    # robomme_*: perceptual/recurrent token memory (sidecar offline + optional runtime buffer).
    # Do not mix kai0_explicit with dataset.memory_feature_root or RoboMME sampling.
    memory_family: str = "none"
    memory_integration: str = "context"  # context | modulation | expert (RoboMME inject modes)

    # History frame window.
    # - kai0_explicit: K raw frames the client/dataset must provide (use 4 for Agilex Kai0).
    # - robomme framesamp: upper bound; effective frames also depend on token budget.
    visual_memory_frames: int = 16
    visual_memory_stride: int = 1
    visual_memory_temporal_every: int = 4
    visual_memory_drop_past_tokens: bool = True
    proprio_memory_frames: int = 1

    # RoboMME-style memory knobs.
    robomme_sampling_type: str = "framesamp"  # framesamp | tokendrop
    robomme_token_budget: int = 768
    robomme_token_per_image: int = 16
    robomme_pool_type: str = "mean"  # mean | attention | last

    # ##modified by wangruiyu to adapt memory mechanism
    memory_pos_dim: int = 256
    memory_use_pos_emb: bool = True
    memory_use_state_emb: bool = False
    memory_pos_hidden_dim: int = 256
    memory_state_hidden_dim: int = 512
    memory_attention_heads: int = 15
    memory_key_value_heads: int = 5
    memory_head_dim: int = 64
    memory_image_emb_key: str = "static_image_emb"
    memory_pos_emb_key: str = "static_pos_emb"
    memory_state_emb_key: str = "static_state_emb"
    memory_mask_key: str = "static_mask"
    memory_runtime_enabled: bool = True
    memory_runtime_log_interval: int = 1
    # ##modified by wangruiyu to adapt memory mechanism

    # Recurrent memory knobs.
    # ##modified by wangruiyu to adapt memory mechanism
    recurrent_type: str = "rmt"  # rmt | ttt
    recurrent_hidden_dim: int = 512
    recurrent_num_layers: int = 1
    recurrent_max_recur_steps: int = 64
    recurrent_max_pretraj_steps: int = 40
    recurrent_mini_batch_size: int = 64
    recurrent_token_per_image: int = 64
    recurrent_budget: int = 512
    recurrent_num_attention_heads: int = 8
    recurrent_num_key_value_heads: int = 1
    recurrent_output_stats: bool = False
    recurrent_input_obs_horizon: int | None = None
    recurrent_ttt_num_heads: int = 2
    recurrent_ttt_base_lr: float = 0.01
    recurrent_ttt_max_lr: float = 0.1
    recurrent_ttt_qk_norm: bool = True
    recurrent_ttt_v_norm: bool = True
    recurrent_ttt_max_grad_norm: float = 5.0
    recurrent_image_emb_key: str = "recur_image_emb"
    recurrent_pos_emb_key: str = "recur_pos_emb"
    recurrent_state_emb_key: str = "recur_state_emb"
    recurrent_mask_key: str = "recur_mask"
    # ##modified by wangruiyu to adapt memory mechanism

    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. It is not ported yet in LeRobot."
            )

        valid_families = {"none", "kai0_explicit", "robomme_perceptual", "robomme_recurrent"}
        if self.memory_family not in valid_families:
            raise ValueError(f"memory_family must be one of {sorted(valid_families)}, got {self.memory_family!r}")
        valid_integrations = {"context", "modulation", "expert"}
        if self.memory_integration not in valid_integrations:
            raise ValueError(
                f"memory_integration must be one of {sorted(valid_integrations)}, got {self.memory_integration!r}"
            )
        valid_sampling = {"framesamp", "tokendrop"}
        if self.robomme_sampling_type not in valid_sampling:
            raise ValueError(
                f"robomme_sampling_type must be one of {sorted(valid_sampling)}, got {self.robomme_sampling_type!r}"
            )
        valid_pooling = {"mean", "attention", "last"}
        if self.robomme_pool_type not in valid_pooling:
            raise ValueError(f"robomme_pool_type must be one of {sorted(valid_pooling)}, got {self.robomme_pool_type!r}")
        # ##modified by wangruiyu to adapt memory mechanism
        valid_recurrent = {"rmt", "ttt"}
        if self.recurrent_type not in valid_recurrent:
            raise ValueError(f"recurrent_type must be one of {sorted(valid_recurrent)}, got {self.recurrent_type!r}")
        # ##modified by wangruiyu to adapt memory mechanism
        for name in ("visual_memory_frames", "visual_memory_stride", "visual_memory_temporal_every", "proprio_memory_frames"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        for name in (
            "robomme_token_budget",
            "robomme_token_per_image",
            "memory_pos_dim",
            "memory_pos_hidden_dim",
            "memory_state_hidden_dim",
            "memory_attention_heads",
            "memory_key_value_heads",
            "memory_head_dim",
            "memory_runtime_log_interval",
            "recurrent_hidden_dim",
            "recurrent_num_layers",
            # ##modified by wangruiyu to adapt memory mechanism
            "recurrent_max_recur_steps",
            "recurrent_max_pretraj_steps",
            "recurrent_mini_batch_size",
            "recurrent_token_per_image",
            "recurrent_budget",
            "recurrent_num_attention_heads",
            "recurrent_num_key_value_heads",
            "recurrent_ttt_num_heads",
            # ##modified by wangruiyu to adapt memory mechanism
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        # ##modified by wangruiyu to adapt memory mechanism
        if self.recurrent_hidden_dim % self.recurrent_num_attention_heads != 0:
            raise ValueError("recurrent_hidden_dim must be divisible by recurrent_num_attention_heads")
        if self.recurrent_num_attention_heads % self.recurrent_num_key_value_heads != 0:
            raise ValueError("recurrent_num_attention_heads must be divisible by recurrent_num_key_value_heads")
        if self.recurrent_hidden_dim % self.recurrent_ttt_num_heads != 0:
            raise ValueError("recurrent_hidden_dim must be divisible by recurrent_ttt_num_heads")
        if self.recurrent_ttt_base_lr <= 0:
            raise ValueError(f"recurrent_ttt_base_lr must be > 0, got {self.recurrent_ttt_base_lr}")
        if self.recurrent_ttt_max_lr <= 0:
            raise ValueError(f"recurrent_ttt_max_lr must be > 0, got {self.recurrent_ttt_max_lr}")
        if self.recurrent_ttt_max_grad_norm <= 0:
            raise ValueError(
                f"recurrent_ttt_max_grad_norm must be > 0, got {self.recurrent_ttt_max_grad_norm}"
            )
        # ##modified by wangruiyu to adapt memory mechanism
        if self.memory_family == "none" and self.memory_enabled:
            raise ValueError("memory_enabled=True requires memory_family != 'none'")
        # Kai0 explicit multi-frame path: client/dataset supplies K raw frames.
        # Mutually exclusive with RoboMME sidecar / runtime token memory.
        if self.memory_family == "kai0_explicit":
            if not self.memory_enabled:
                raise ValueError("memory_family='kai0_explicit' requires memory_enabled=True")
            if self.visual_memory_frames < 2:
                raise ValueError(
                    "memory_family='kai0_explicit' requires visual_memory_frames >= 2 "
                    f"(got {self.visual_memory_frames}). Use K=4 for the Agilex Kai0 chain."
                )
        # ##modified by wangruiyu to adapt memory mechanism

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def memory_is_enabled(self) -> bool:
        return self.memory_enabled

    @property
    def effective_proprio_memory_frames(self) -> int:
        return self.proprio_memory_frames if self.proprio_memory_frames > 1 else 1

    @property
    def memory_frame_count(self) -> int:
        return max(self.visual_memory_frames, self.effective_proprio_memory_frames)

    @property
    def observation_delta_indices(self) -> list:
        if not self.memory_is_enabled:
            return [0]
        # ##modified by wangruiyu to adapt memory mechanism
        if self.memory_family in {"robomme_perceptual", "robomme_recurrent"}:
            return [0]
        # ##modified by wangruiyu to adapt memory mechanism
        return [-(self.memory_frame_count - 1 - i) * self.visual_memory_stride for i in range(self.memory_frame_count)]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
