#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

"""
SmolVLA:

[Paper](https://huggingface.co/papers/2506.01844)

Designed by Hugging Face.

Install smolvla extra dependencies:
```bash
pip install -e ".[smolvla]"
```

Example of finetuning the smolvla pretrained model (`smolvla_base`):
```bash
lerobot-train \
--policy.path=lerobot/smolvla_base \
--dataset.repo_id=<USER>/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of finetuning a smolVLA. SmolVLA is composed of a pretrained VLM,
and an action expert.
```bash
lerobot-train \
--policy.type=smolvla \
--dataset.repo_id=<USER>/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of using the smolvla pretrained model outside LeRobot training framework:
```python
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

"""

import logging
import math
from collections import deque
from typing import TypedDict, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.import_utils import require_package

from ..common.flow_matching import euler_integrate, sample_noise, sample_time_beta
from ..common.vla_utils import (
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    pad_vector,
    resize_with_pad,
)
from ..pretrained import PreTrainedPolicy
from ..rtc.modeling_rtc import RTCProcessor
from ..utils import (
    populate_queues,
)
from .configuration_smolvla import SmolVLAConfig
from .memory_buffer import SmolVLAPerceptualMemoryBuffer, SmolVLARecurrentMemoryBuffer
from .smolvlm_with_expert import SmolVLMWithExpertModel


logger = logging.getLogger(__name__)


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class SmolVLAPolicy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = SmolVLAConfig
    name = "smolvla"

    def __init__(
        self,
        config: SmolVLAConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """

        require_package("transformers", extra="smolvla")
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = VLAFlowMatching(config, rtc_processor=self.rtc_processor)
        # ##modified by wangruiyu to adapt memory mechanism
        self.memory_buffer = None
        if (
            config.memory_enabled
            and config.memory_runtime_enabled
            and config.memory_family == "robomme_perceptual"
            and config.memory_integration in {"modulation", "expert"}
        ):
            self.memory_buffer = SmolVLAPerceptualMemoryBuffer(
                num_frames=config.visual_memory_frames,
                stride=config.visual_memory_stride,
                token_per_image=config.robomme_token_per_image,
                token_budget=config.robomme_token_budget,
                pos_dim=config.memory_pos_dim,
                sampling_type=config.robomme_sampling_type,
                log_interval=config.memory_runtime_log_interval,
            )
        elif (
            config.memory_enabled
            and config.memory_runtime_enabled
            and config.memory_family == "robomme_recurrent"
            and config.memory_integration in {"modulation", "expert"}
        ):
            self.memory_buffer = SmolVLARecurrentMemoryBuffer(
                max_recur_steps=config.recurrent_max_recur_steps,
                input_obs_horizon=config.recurrent_input_obs_horizon or config.visual_memory_frames,
                max_pretraj_steps=config.recurrent_max_pretraj_steps,
                token_per_image=config.recurrent_token_per_image,
                pos_dim=config.memory_pos_dim,
                log_interval=config.memory_runtime_log_interval,
            )
        # ##modified by wangruiyu to adapt memory mechanism
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        # ##modified by wangruiyu to adapt memory mechanism
        if getattr(self, "memory_buffer", None) is not None:
            self.memory_buffer.reset()
        # ##modified by wangruiyu to adapt memory mechanism

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Lets create processor if the config provided
        # If RTC is not enabled - we still can track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            # In case of calling init_rtc_processor after the model is created
            # We need to set the rtc_processor to the model
            # During the normal initialization process the model is not created yet
            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def get_optim_params(self) -> dict:
        return self.parameters()

    # ##modified by wangruiyu to adapt memory mechanism
    def _prepare_action_inputs(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor], Tensor, Tensor, Tensor]:
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        return images, img_masks, state, lang_tokens, lang_masks

    def _append_runtime_memory(self, images: list[Tensor], img_masks: list[Tensor], state: Tensor) -> None:
        if self.memory_buffer is not None:
            self.memory_buffer.append_from_prepared_inputs(self, images, img_masks, state)

    def _inject_runtime_memory(self, batch: dict[str, Tensor], device: torch.device) -> None:
        if self.memory_buffer is None:
            return
        (
            image_emb,
            pos_emb,
            state_emb,
            mask,
            memory_info,
        ) = self.memory_buffer.build_batch(device)
        if self.config.memory_family == "robomme_recurrent":
            batch[self.config.recurrent_image_emb_key] = image_emb
            batch[self.config.recurrent_pos_emb_key] = pos_emb
            batch[self.config.recurrent_state_emb_key] = state_emb
            batch[self.config.recurrent_mask_key] = mask
        else:
            batch[self.config.memory_image_emb_key] = image_emb
            batch[self.config.memory_pos_emb_key] = pos_emb
            batch[self.config.memory_state_emb_key] = state_emb
            batch[self.config.memory_mask_key] = mask
        if self.memory_buffer.should_log():
            logger.info(
                "SmolVLA memory runtime: enabled=%s sampling=%s N=%s K=%s effective_K=%s stride=%s tokens=%s raw_tokens=%s "
                "dropped_tokens=%s padding_frames=%s buffer=%s appends=%s indices=%s ages=%s",
                True,
                memory_info.get("sampling", self.config.robomme_sampling_type),
                memory_info["memory_N"],
                memory_info["target_K"],
                memory_info["effective_K"],
                memory_info["stride"],
                memory_info["tokens"],
                memory_info["raw_tokens"],
                memory_info["dropped_tokens"],
                memory_info["padding_frames"],
                memory_info["buffer"],
                memory_info["appends"],
                memory_info["indices"],
                memory_info["ages"],
            )
    # ##modified by wangruiyu to adapt memory mechanism

    def _get_action_chunk(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        # ##modified by wangruiyu to adapt memory mechanism
        prepared_inputs: tuple[list[Tensor], list[Tensor], Tensor, Tensor, Tensor] | None = None,
        append_memory: bool = True,
        # ##modified by wangruiyu to adapt memory mechanism
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        # TODO: Check if this for loop is needed.
        # Context: In fact, self.queues contains only ACTION field, and in inference, we don't have action in the batch
        # In the case of offline inference, we have the action in the batch
        # that why without the k != ACTION check, it will raise an error because we are trying to stack
        # on an empty container.
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        # ##modified by wangruiyu to adapt memory mechanism
        if prepared_inputs is None:
            images, img_masks, state, lang_tokens, lang_masks = self._prepare_action_inputs(batch)
        else:
            images, img_masks, state, lang_tokens, lang_masks = prepared_inputs

        if self.memory_buffer is not None:
            if append_memory:
                self._append_runtime_memory(images, img_masks, state)
            self._inject_runtime_memory(batch, state.device)
        memory_tokens, memory_mask = self.model.prepare_memory(batch)
        actions = self.model.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise=noise,
            memory_tokens=memory_tokens,
            memory_mask=memory_mask,
            **kwargs,
        )
        # ##modified by wangruiyu to adapt memory mechanism

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        return batch

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        self.eval()

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        actions = self._get_action_chunk(batch, noise, **kwargs)
        return actions

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """

        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        # ##modified by wangruiyu to adapt memory mechanism
        prepared_inputs = None
        if self.memory_buffer is not None:
            prepared_inputs = self._prepare_action_inputs(batch)
            self._append_runtime_memory(prepared_inputs[0], prepared_inputs[1], prepared_inputs[2])

        if self._check_get_actions_condition():
            actions = self._get_action_chunk(batch, noise, prepared_inputs=prepared_inputs, append_memory=False)
            # ##modified by wangruiyu to adapt memory mechanism

            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def _check_get_actions_condition(self) -> bool:
        return len(self._queues[ACTION]) == 0

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss.

        Args:
            batch: Training batch containing observations and actions.
            noise: Optional noise tensor for flow matching.
            time: Optional time tensor for flow matching.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")
        loss_dict = {}
        # ##modified by wangruiyu to adapt memory mechanism
        memory_tokens, memory_mask = self.model.prepare_memory(batch)
        losses = self.model.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            noise,
            time,
            memory_tokens=memory_tokens,
            memory_mask=memory_mask,
        )
        # ##modified by wangruiyu to adapt memory mechanism
        original_action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]
        loss_dict["losses_after_forward"] = losses.clone().mean().item()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone().mean().item()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone().mean().item()

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over valid (time, action) entries
            if actions_is_pad is None:
                per_sample_loss = losses.mean(dim=(1, 2))
            else:
                num_valid = ((~actions_is_pad).sum(dim=1) * losses.shape[-1]).clamp_min(1)
                per_sample_loss = losses.sum(dim=(1, 2)) / num_valid
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss over valid (time, action) entries
            if actions_is_pad is None:
                loss = losses.mean()
            else:
                num_valid = ((~actions_is_pad).sum() * losses.shape[-1]).clamp_min(1)
                loss = losses.sum() / num_valid
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def _preprocess_image_tensor(self, img: Tensor) -> Tensor:
        """Resize/pad and map pixel range [0,1] -> [-1,1] for SigLIP."""
        if self.config.resize_imgs_with_padding is not None:
            # SmolVLA stores the target as (width, height); the shared helper expects (height, width).
            img = resize_with_pad(
                img,
                self.config.resize_imgs_with_padding[1],
                self.config.resize_imgs_with_padding[0],
                pad_value=0,
            )
        return img * 2.0 - 1.0

    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.

        ##modified by wangruiyu to adapt memory mechanism
        For `memory_family=kai0_explicit`, a 5D stack `B,K,C,H,W` is expanded into K image
        streams per camera (oldest -> newest) so all frames enter `embed_prefix`.
        Other families keep the previous single-frame behavior (last temporal index).
        """
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        # ##modified by wangruiyu to adapt memory mechanism
        kai0_explicit = (
            self.config.memory_enabled and self.config.memory_family == "kai0_explicit"
        )
        # ##modified by wangruiyu to adapt memory mechanism

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]
            # ##modified by wangruiyu to adapt memory mechanism
            if kai0_explicit and img.ndim == 5:
                # B,K,C,H,W -> one prefix stream per temporal index.
                bsize, num_frames = img.shape[:2]
                device = img.device
                is_pad = batch.get(f"{key}_is_pad")
                if is_pad is not None:
                    is_pad = is_pad.to(device=device)
                for frame_idx in range(num_frames):
                    frame = self._preprocess_image_tensor(img[:, frame_idx])
                    if is_pad is not None:
                        # `_is_pad=True` means the delta query fell outside the episode.
                        mask = ~is_pad[:, frame_idx].bool()
                    elif f"{key}_padding_mask" in batch and batch[f"{key}_padding_mask"].ndim == 2:
                        mask = batch[f"{key}_padding_mask"][:, frame_idx].bool()
                    else:
                        mask = torch.ones(bsize, dtype=torch.bool, device=device)
                    images.append(frame)
                    img_masks.append(mask)
                continue
            # ##modified by wangruiyu to adapt memory mechanism

            img = img[:, -1, :, :, :] if img.ndim == 5 else img
            img = self._preprocess_image_tensor(img)

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"]
                mask = mask[:, -1] if mask.ndim == 2 else mask
                mask = mask.bool()
            elif f"{key}_is_pad" in batch:
                # Single-frame path: valid if the last delta index is not pad.
                is_pad = batch[f"{key}_is_pad"]
                mask = ~is_pad[:, -1].bool() if is_pad.ndim == 2 else ~is_pad.bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state.

        ##modified by wangruiyu to adapt memory mechanism
        kai0_explicit with proprio_memory_frames>1 keeps the full [B,K,D] stack so
        `embed_prefix` can attend a short proprio history. Otherwise use the latest frame.
        """
        state = batch[OBS_STATE]
        kai0_explicit = (
            self.config.memory_enabled and self.config.memory_family == "kai0_explicit"
        )
        if state.ndim > 2:
            if kai0_explicit and self.config.effective_proprio_memory_frames > 1:
                state = pad_vector(state, self.config.max_state_dim)
                return state
            state = state[:, -1, :]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def _get_default_peft_targets(self) -> dict[str, any]:
        """Return default PEFT target modules for SmolVLA fine-tuning."""
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        # ##modified by wangruiyu to adapt memory mechanism
        target_modules = (
            rf"(model\.vlm_with_expert\.(lm_expert|lm_memory_expert)\..*\.(q|v)_proj|"
            rf"model\.({common_projections}|memory_encoder.*))"
        )
        # ##modified by wangruiyu to adapt memory mechanism
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }

    def _validate_peft_config(self, peft_config) -> None:
        """Validate PEFT configuration for SmolVLA."""
        super()._validate_peft_config(peft_config)
        if not self.config.load_vlm_weights:
            import logging

            logging.warning(
                "Training SmolVLA from scratch using PEFT. This is unlikely to yield good results. "
                "Set `load_vlm_weights=True` to fine-tune the existing policy."
            )


# ##modified by wangruiyu to adapt memory mechanism
MEMORY_IMAGE_EMB = "static_image_emb"
MEMORY_POS_EMB = "static_pos_emb"
MEMORY_STATE_EMB = "static_state_emb"
MEMORY_MASK = "static_mask"
# ##modified by wangruiyu to adapt memory mechanism


def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor



# ##modified by wangruiyu to adapt memory mechanism
class BaseFeatureMemoryEncoder(nn.Module):
    def __init__(
        self,
        image_dim: int,
        pos_dim: int,
        state_dim: int,
        output_dim: int,
        pos_hidden_dim: int,
        state_hidden_dim: int,
        use_pos_emb: bool = True,
        use_state_emb: bool = False,
    ):
        super().__init__()
        self.use_pos_emb = use_pos_emb
        self.use_state_emb = use_state_emb
        input_dim = image_dim
        if self.use_pos_emb:
            self.pos_proj = nn.Linear(pos_dim, pos_hidden_dim)
            input_dim += pos_hidden_dim
        else:
            self.pos_proj = None
        if self.use_state_emb:
            self.state_proj = nn.Linear(state_dim, state_hidden_dim)
            input_dim += state_hidden_dim
        else:
            self.state_proj = None
        self.input_dim = input_dim
        self.output_dim = output_dim

    def encode_features(self, image_emb: Tensor, pos_emb: Tensor, state_emb: Tensor, proj: nn.Linear) -> Tensor:
        dtype = proj.weight.dtype
        memory_parts = [image_emb.to(dtype=dtype)]
        if self.use_pos_emb:
            if self.pos_proj is None:
                raise RuntimeError("memory pos projection is not initialized")
            memory_parts.append(F.silu(self.pos_proj(pos_emb.to(dtype=dtype))))
        if self.use_state_emb:
            if self.state_proj is None:
                raise RuntimeError("memory state projection is not initialized")
            state_emb = state_emb.to(dtype=dtype)
            if image_emb.ndim == 5:
                _, _, views, patches, _ = image_emb.shape
                state_emb = F.silu(self.state_proj(state_emb))
                state_emb = state_emb[:, :, None, None, :].expand(-1, -1, views, patches, -1)
            else:
                state_emb = F.silu(self.state_proj(state_emb))
            memory_parts.append(state_emb)
        return proj(torch.cat(memory_parts, dim=-1))


class PerceptualMemoryEncoder(BaseFeatureMemoryEncoder):
    def __init__(
        self,
        image_dim: int,
        pos_dim: int,
        state_dim: int,
        output_dim: int,
        pos_hidden_dim: int,
        state_hidden_dim: int,
        use_pos_emb: bool = True,
        use_state_emb: bool = False,
    ):
        super().__init__(
            image_dim=image_dim,
            pos_dim=pos_dim,
            state_dim=state_dim,
            output_dim=output_dim,
            pos_hidden_dim=pos_hidden_dim,
            state_hidden_dim=state_hidden_dim,
            use_pos_emb=use_pos_emb,
            use_state_emb=use_state_emb,
        )
        self.proj = nn.Linear(self.input_dim, output_dim)

    def forward(
        self,
        image_emb: Tensor,
        pos_emb: Tensor,
        state_emb: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return self.encode_features(image_emb, pos_emb, state_emb, self.proj), mask.to(dtype=torch.bool)


class RMTMemoryLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        budget: int,
        num_heads: int,
        num_key_value_heads: int,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if num_heads % num_key_value_heads != 0:
            raise ValueError("num_heads must be divisible by num_key_value_heads")
        self.hidden_dim = hidden_dim
        self.budget = budget
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_dim // num_heads
        self.kv_groups = num_heads // num_key_value_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_key_value_heads * self.head_dim, bias=False)
        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.memory_state = nn.Parameter(torch.randn(budget, hidden_dim) * math.sqrt(1.0 / hidden_dim))
        self.post_norm = nn.LayerNorm(hidden_dim)

    def reset(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        return self.memory_state.to(device=device, dtype=dtype)[None, :, :].expand(batch_size, -1, -1).clone()

    def _expand_kv_heads(self, x: Tensor) -> Tensor:
        batch_size, seq_len = x.shape[:2]
        x = x.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        x = x[:, :, :, None, :].expand(batch_size, seq_len, self.num_key_value_heads, self.kv_groups, self.head_dim)
        return x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

    def process_mini_batch(self, x_mb: Tensor, mask_mb: Tensor, mem_state: Tensor) -> Tensor:
        active = mask_mb.to(dtype=torch.bool)[:, None, None]
        inputs = torch.cat([x_mb, mem_state], dim=1)
        inputs = self.pre_norm(inputs)
        batch_size, seq_len = inputs.shape[:2]

        q = self.q_proj(inputs).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self._expand_kv_heads(self.k_proj(inputs)).transpose(1, 2)
        v = self._expand_kv_heads(self.v_proj(inputs)).transpose(1, 2)
        encoded = F.scaled_dot_product_attention(q, k, v)
        encoded = encoded.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        out = self.to_out(encoded[:, -self.budget :])
        new_mem = self.post_norm(out + mem_state)
        return torch.where(active, new_mem, mem_state)

    def forward(self, hidden_states: Tensor, recur_mask: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = hidden_states.shape[0]
        mem_state = self.reset(batch_size, hidden_states.device, hidden_states.dtype)
        for mb_idx in range(hidden_states.shape[1]):
            mem_state = self.process_mini_batch(hidden_states[:, mb_idx], recur_mask[:, mb_idx], mem_state)
        return mem_state, torch.ones((batch_size, self.budget), dtype=torch.bool, device=hidden_states.device)


class TTTLinearMemoryLayer(nn.Module):
    """RoboMME-style linear TTT recurrent memory layer for SmolVLA."""

    def __init__(
        self,
        hidden_dim: int,
        budget: int,
        mini_batch_size: int,
        num_heads: int,
        base_lr: float,
        max_lr: float,
        max_grad_norm: float,
        qk_norm: bool = True,
        v_norm: bool = True,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by TTT num_heads")
        self.hidden_dim = hidden_dim
        self.budget = budget
        self.mini_batch_size = mini_batch_size
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.max_lr = max_lr
        self.max_grad_norm = max_grad_norm

        self.lr_proj = nn.Linear(hidden_dim, num_heads)
        self.wq = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.wk = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.wv = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.wo = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.post_norm = nn.LayerNorm(hidden_dim)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else None
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else None
        self.v_norm = nn.LayerNorm(self.head_dim) if v_norm else None

        self.W1 = nn.Parameter(torch.randn(num_heads, self.head_dim, self.head_dim) * math.sqrt(1.0 / self.head_dim))
        self.b1 = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.norm = nn.Parameter(torch.zeros(num_heads, 2, self.head_dim))
        self.ttt_base_lr_inv = nn.Parameter(torch.tensor(self._inv_softplus(base_lr), dtype=torch.float32))

        nn.init.normal_(self.wv.weight, std=0.01)
        nn.init.normal_(self.wo.weight, std=0.01)

    @staticmethod
    def _inv_softplus(value: float) -> float:
        return value + math.log(-math.expm1(-value))

    @staticmethod
    def _apply_ttt_rms_norm(x: Tensor, scale_shift: Tensor) -> Tensor:
        scale = 1 + scale_shift[..., 0, :]
        shift = scale_shift[..., 1, :]
        var = x.to(dtype=torch.float32).square().mean(dim=-1, keepdim=True)
        out = x * torch.rsqrt(var.to(dtype=x.dtype) + 1e-8) * scale + shift
        return out.to(dtype=x.dtype)

    @staticmethod
    def _ttt_rms_norm_vjp(z: Tensor, scale_shift: Tensor, grad_out: Tensor) -> Tensor:
        scale = 1 + scale_shift[..., 0, :]
        u = grad_out * scale
        z_f = z.to(dtype=torch.float32)
        u_f = u.to(dtype=torch.float32)
        var = z_f.square().mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(var + 1e-8)
        correction = (u_f * z_f).mean(dim=-1, keepdim=True)
        grad = inv_rms * u_f - z_f * inv_rms.pow(3) * correction
        return grad.to(dtype=z.dtype)

    def reset(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor]:
        W1 = self.W1.to(device=device, dtype=dtype)[None, :, :, :].expand(batch_size, -1, -1, -1).clone()
        b1 = self.b1.to(device=device, dtype=dtype)[None, :, :].expand(batch_size, -1, -1).clone()
        norm = self.norm.to(device=device, dtype=dtype)[None, :, :, :].expand(batch_size, -1, -1, -1).clone()
        return W1, b1, norm

    def _heads(self, x: Tensor) -> Tensor:
        batch_size, num_mini_batches, mini_batch_size = x.shape[:3]
        return x.reshape(batch_size, num_mini_batches, mini_batch_size, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)

    def get_ttt_inputs(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        xq = self._heads(self.wq(hidden_states))
        xk = self._heads(self.wk(hidden_states))
        xv = self._heads(self.wv(hidden_states))
        if self.q_norm is not None:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)
        if self.v_norm is not None:
            xv = self.v_norm(xv)
        lr = self.lr_proj(hidden_states.to(dtype=torch.float32))
        eta = F.softplus(lr + self.ttt_base_lr_inv.to(device=lr.device, dtype=lr.dtype))
        eta = eta.clamp(max=self.max_lr).permute(0, 3, 1, 2).to(dtype=hidden_states.dtype)
        return xq, xk, xv, eta

    def _clip_update(self, grad: Tensor) -> Tensor:
        grad_norm = torch.linalg.vector_norm(grad.reshape(*grad.shape[:-2], -1), dim=-1)
        scale = torch.clamp(self.max_grad_norm / (grad_norm + 1e-8), max=1.0)
        while scale.ndim < grad.ndim:
            scale = scale.unsqueeze(-1)
        return grad * scale

    def process_mini_batch(
        self,
        xq_mb: Tensor,
        xk_mb: Tensor,
        xv_mb: Tensor,
        eta_mb: Tensor,
        mask_mb: Tensor,
        memory_state_init: tuple[Tensor, Tensor, Tensor],
        memory_state: tuple[Tensor, Tensor, Tensor],
    ) -> tuple[tuple[Tensor, Tensor, Tensor], Tensor]:
        W1, b1, norm = memory_state
        W1_init, b1_init, norm_init = memory_state_init
        del W1_init, b1_init, norm_init

        z = torch.einsum("b h m d,b h d e->b h m e", xk_mb, W1) + b1[:, :, None, :]
        ttt_norm_out = self._apply_ttt_rms_norm(z, norm[:, :, None, :, :])
        ssl_target = xv_mb - xk_mb
        error = ttt_norm_out - ssl_target
        dz = self._ttt_rms_norm_vjp(z, norm[:, :, None, :, :], error)
        eta_expanded = eta_mb[..., None]
        dW1 = torch.einsum("b h m d,b h m e->b h d e", xk_mb, eta_expanded * dz)
        db1 = (eta_expanded * dz).sum(dim=2)
        W1_new = W1 - self._clip_update(dW1)
        db_norm = torch.linalg.vector_norm(db1, dim=-1, keepdim=True)
        db1 = db1 * torch.clamp(self.max_grad_norm / (db_norm + 1e-8), max=1.0)
        b1_new = b1 - db1

        output = self._apply_ttt_rms_norm(
            torch.einsum("b h m d,b h d e->b h m e", xq_mb, W1_new) + b1_new[:, :, None, :],
            norm[:, :, None, :, :],
        ) + xq_mb
        active = mask_mb.to(dtype=torch.bool)[:, None, None, None]
        W1_out = torch.where(active, W1_new, W1)
        b1_out = torch.where(active.squeeze(-1), b1_new, b1)
        output = torch.where(active, output, torch.zeros_like(output))
        return (W1_out, b1_out, norm), output

    def forward(self, hidden_states: Tensor, recur_mask: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, num_mini_batches, mini_batch_size = hidden_states.shape[:3]
        if mini_batch_size != self.mini_batch_size:
            raise ValueError(f"Expected mini_batch_size={self.mini_batch_size}, got {mini_batch_size}")
        xq, xk, xv, eta = self.get_ttt_inputs(hidden_states)
        memory_state = self.reset(batch_size, hidden_states.device, hidden_states.dtype)
        memory_state_init = tuple(x.clone() for x in memory_state)
        outputs = []
        for mb_idx in range(num_mini_batches):
            memory_state, output = self.process_mini_batch(
                xq[:, :, mb_idx],
                xk[:, :, mb_idx],
                xv[:, :, mb_idx],
                eta[:, :, mb_idx],
                recur_mask[:, mb_idx],
                memory_state_init,
                memory_state,
            )
            outputs.append(output)
        output = torch.stack(outputs, dim=2).permute(0, 2, 3, 1, 4).reshape(
            batch_size, num_mini_batches * mini_batch_size, self.hidden_dim
        )
        output = hidden_states.reshape(batch_size, num_mini_batches * mini_batch_size, self.hidden_dim) + self.wo(
            self.post_norm(output)
        )
        mask_repeat = recur_mask[:, :, None].expand(batch_size, num_mini_batches, mini_batch_size).reshape(
            batch_size, num_mini_batches * mini_batch_size
        )
        return output[:, -self.budget :], mask_repeat[:, -self.budget :]


class RecurrentMemoryEncoder(BaseFeatureMemoryEncoder):
    def __init__(
        self,
        image_dim: int,
        pos_dim: int,
        state_dim: int,
        output_dim: int,
        pos_hidden_dim: int,
        state_hidden_dim: int,
        hidden_dim: int,
        budget: int,
        mini_batch_size: int,
        recurrent_type: str,
        num_heads: int,
        num_key_value_heads: int,
        ttt_num_heads: int = 2,
        ttt_base_lr: float = 0.01,
        ttt_max_lr: float = 0.1,
        ttt_max_grad_norm: float = 5.0,
        ttt_qk_norm: bool = True,
        ttt_v_norm: bool = True,
        use_pos_emb: bool = True,
        use_state_emb: bool = False,
    ):
        super().__init__(
            image_dim=image_dim,
            pos_dim=pos_dim,
            state_dim=state_dim,
            output_dim=hidden_dim,
            pos_hidden_dim=pos_hidden_dim,
            state_hidden_dim=state_hidden_dim,
            use_pos_emb=use_pos_emb,
            use_state_emb=use_state_emb,
        )
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.budget = budget
        self.mini_batch_size = mini_batch_size
        self.recurrent_type = recurrent_type
        self.encoder_recur = nn.Linear(self.input_dim, hidden_dim)
        if recurrent_type == "rmt":
            self.recur_layer = RMTMemoryLayer(
                hidden_dim=hidden_dim,
                budget=budget,
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
            )
        elif recurrent_type == "ttt":
            self.recur_layer = TTTLinearMemoryLayer(
                hidden_dim=hidden_dim,
                budget=budget,
                mini_batch_size=mini_batch_size,
                num_heads=ttt_num_heads,
                base_lr=ttt_base_lr,
                max_lr=ttt_max_lr,
                max_grad_norm=ttt_max_grad_norm,
                qk_norm=ttt_qk_norm,
                v_norm=ttt_v_norm,
            )
        else:
            raise ValueError(f"Unsupported recurrent memory type: {recurrent_type!r}")
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        image_emb: Tensor,
        pos_emb: Tensor,
        state_emb: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        hidden = self.encode_features(image_emb, pos_emb, state_emb, self.encoder_recur)
        batch_size, steps, views, patches, _ = hidden.shape
        hidden = hidden.reshape(batch_size, steps * views * patches, self.hidden_dim)
        if hidden.shape[1] % self.mini_batch_size != 0:
            raise ValueError(
                f"recurrent input tokens ({hidden.shape[1]}) must be divisible by mini_batch_size={self.mini_batch_size}"
            )
        num_mini_batches = hidden.shape[1] // self.mini_batch_size
        hidden = hidden.reshape(batch_size, num_mini_batches, self.mini_batch_size, self.hidden_dim)
        frame_mask = mask.to(device=hidden.device, dtype=torch.bool)
        # ##modified by wangruiyu to adapt memory mechanism
        # Recurrent layers consume mini-batches of flattened visual tokens. Build the validity mask
        # at token granularity first, then reduce each mini-batch to one recurrent update flag.
        if frame_mask.ndim == 2:
            token_mask = frame_mask[:, :, None, None].expand(batch_size, steps, views, patches)
        elif frame_mask.ndim == 3:
            token_mask = frame_mask[:, :, :, None].expand(batch_size, steps, views, patches)
        elif frame_mask.ndim == 4:
            token_mask = frame_mask.expand(batch_size, steps, views, patches)
        else:
            raise ValueError(f"Unsupported recurrent memory mask shape: {tuple(frame_mask.shape)}")
        recur_mask = token_mask.reshape(batch_size, num_mini_batches, self.mini_batch_size).any(dim=-1)
        # ##modified by wangruiyu to adapt memory mechanism
        output, output_mask = self.recur_layer(hidden, recur_mask)
        return self.proj(output), output_mask
# ##modified by wangruiyu to adapt memory mechanism


class VLAFlowMatching(nn.Module):
    """
    SmolVLA

    [Paper]()

    Designed by Hugging Face.
    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    """

    def __init__(self, config: SmolVLAConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config

        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            # ##modified by wangruiyu to adapt memory mechanism
            memory_integration=config.memory_integration if config.memory_enabled else "context",
            memory_attention_heads=config.memory_attention_heads,
            memory_key_value_heads=config.memory_key_value_heads,
            memory_head_dim=config.memory_head_dim,
            # ##modified by wangruiyu to adapt memory mechanism
            device=self.config.device if self.config.device is not None else "auto",
        )
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )

        # ##modified by wangruiyu to adapt memory mechanism
        self.memory_encoder = None
        if (
            config.memory_enabled
            and config.memory_family == "robomme_perceptual"
            and config.memory_integration in {"modulation", "expert"}
        ):
            self.memory_encoder = PerceptualMemoryEncoder(
                image_dim=self.vlm_with_expert.vlm.config.text_config.hidden_size,
                pos_dim=config.memory_pos_dim,
                state_dim=config.max_state_dim,
                output_dim=self.vlm_with_expert.expert_hidden_size,
                pos_hidden_dim=config.memory_pos_hidden_dim,
                state_hidden_dim=config.memory_state_hidden_dim,
                use_pos_emb=config.memory_use_pos_emb,
                use_state_emb=config.memory_use_state_emb,
            )
        elif (
            config.memory_enabled
            and config.memory_family == "robomme_recurrent"
            and config.memory_integration in {"modulation", "expert"}
        ):
            self.memory_encoder = RecurrentMemoryEncoder(
                image_dim=self.vlm_with_expert.vlm.config.text_config.hidden_size,
                pos_dim=config.memory_pos_dim,
                state_dim=config.max_state_dim,
                output_dim=self.vlm_with_expert.expert_hidden_size,
                pos_hidden_dim=config.memory_pos_hidden_dim,
                state_hidden_dim=config.memory_state_hidden_dim,
                hidden_dim=config.recurrent_hidden_dim,
                budget=config.recurrent_budget,
                mini_batch_size=config.recurrent_mini_batch_size,
                recurrent_type=config.recurrent_type,
                num_heads=config.recurrent_num_attention_heads,
                num_key_value_heads=config.recurrent_num_key_value_heads,
                ttt_num_heads=config.recurrent_ttt_num_heads,
                ttt_base_lr=config.recurrent_ttt_base_lr,
                ttt_max_lr=config.recurrent_ttt_max_lr,
                ttt_max_grad_norm=config.recurrent_ttt_max_grad_norm,
                ttt_qk_norm=config.recurrent_ttt_qk_norm,
                ttt_v_norm=config.recurrent_ttt_v_norm,
                use_pos_emb=config.memory_use_pos_emb,
                use_state_emb=config.memory_use_state_emb,
            )
        # ##modified by wangruiyu to adapt memory mechanism

        self.set_requires_grad()
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length
        self.rtc_processor = rtc_processor

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    # ##modified by wangruiyu to adapt memory mechanism
    def memory_integration_is(self, *modes: str) -> bool:
        return self.config.memory_enabled and self.config.memory_integration in set(modes)

    def prepare_memory(self, batch: dict[str, Tensor]) -> tuple[Tensor | None, Tensor | None]:
        if self.memory_encoder is None:
            return None, None
        # ##modified by wangruiyu to adapt memory mechanism
        if self.config.memory_family == "robomme_recurrent":
            image_key = self.config.recurrent_image_emb_key
            pos_key = self.config.recurrent_pos_emb_key
            state_key = self.config.recurrent_state_emb_key
            mask_key = self.config.recurrent_mask_key
        else:
            image_key = self.config.memory_image_emb_key
            pos_key = self.config.memory_pos_emb_key
            state_key = self.config.memory_state_emb_key
            mask_key = self.config.memory_mask_key
        # ##modified by wangruiyu to adapt memory mechanism
        required = [image_key, mask_key]
        if self.config.memory_use_pos_emb:
            required.append(pos_key)
        if self.config.memory_use_state_emb:
            required.append(state_key)
        missing = [key for key in required if key not in batch]
        if missing:
            if self.config.memory_enabled:
                raise KeyError(
                    "SmolVLA memory is enabled but batch is missing memory feature keys: "
                    f"{missing}. Configure dataset.memory_feature_root or disable policy.memory_enabled."
                )
            return None, None
        image_emb = batch[image_key]
        pos_emb = (
            batch[pos_key]
            if self.config.memory_use_pos_emb
            else image_emb.new_zeros((*image_emb.shape[:-1], self.config.memory_pos_dim))
        )
        if self.config.memory_use_state_emb:
            state_emb = pad_vector(batch[state_key], self.config.max_state_dim)
        else:
            state_shape = (*image_emb.shape[:2], self.config.max_state_dim) if image_emb.ndim == 5 else (*image_emb.shape[:-1], self.config.max_state_dim)
            state_emb = image_emb.new_zeros(state_shape)
        return self.memory_encoder(
            image_emb,
            pos_emb,
            state_emb,
            batch[mask_key],
        )

    def _memory_expert_inference_enabled(self, memory_tokens: Tensor | None) -> bool:
        return self.memory_integration_is("expert") and memory_tokens is not None

    def _integrate_actions(self, denoise_fn, noise: Tensor, **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        return euler_integrate(
            denoise_fn,
            noise,
            self.config.num_steps,
            rtc_processor=self.rtc_processor,
            rtc_enabled=self._rtc_enabled(),
            inference_delay=kwargs.get("inference_delay"),
            prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
            execution_horizon=kwargs.get("execution_horizon"),
        )

    def _project_suffix_actions(self, suffix_out: Tensor) -> Tensor:
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    def _build_full_attention_inputs(
        self,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        suffix_pad_masks: Tensor,
        suffix_att_masks: Tensor,
    ) -> tuple[Tensor, Tensor]:
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        # ##modified by wangruiyu to adapt memory mechanism
        # Prefix att masks are bool; suffix historically used emb dtype. Normalize before cat.
        att_masks = torch.cat(
            [prefix_att_masks.to(dtype=torch.bool), suffix_att_masks.to(dtype=torch.bool)],
            dim=1,
        )
        # ##modified by wangruiyu to adapt memory mechanism
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        return att_2d_masks, position_ids
    # ##modified by wangruiyu to adapt memory mechanism

    def sample_noise(self, shape, device):
        return sample_noise(shape, device)

    def sample_time(self, bsize, device):
        return sample_time_beta(bsize, device, alpha=1.5, beta=1.0, scale=0.999, offset=0.001)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for SmolVLM transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []
        for _img_idx, (
            img,
            img_mask,
        ) in enumerate(zip(images, img_masks, strict=False)):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb = img_emb

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            att_masks += [0] * (num_img_embs)
            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1] * (states_seq_len)
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=None,
        time=None,
        # ##modified by wangruiyu to adapt memory mechanism
        memory_tokens=None,
        memory_mask=None,
        # ##modified by wangruiyu to adapt memory mechanism
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        # ##modified by wangruiyu to adapt memory mechanism
        att_2d_masks, position_ids = self._build_full_attention_inputs(
            prefix_pad_masks,
            prefix_att_masks,
            suffix_pad_masks,
            suffix_att_masks,
        )
        # ##modified by wangruiyu to adapt memory mechanism
        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            # ##modified by wangruiyu to adapt memory mechanism
            memory_tokens=memory_tokens,
            memory_mask=memory_mask,
            # ##modified by wangruiyu to adapt memory mechanism
        )
        suffix_out = outputs_embeds[-1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        # ##modified by wangruiyu to adapt memory mechanism
        memory_tokens=None,
        memory_mask=None,
        # ##modified by wangruiyu to adapt memory mechanism
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        # ##modified by wangruiyu to adapt memory mechanism
        if self._memory_expert_inference_enabled(memory_tokens):
            return self.sample_actions_with_memory_expert(
                noise=noise,
                prefix_embs=prefix_embs,
                prefix_pad_masks=prefix_pad_masks,
                prefix_att_masks=prefix_att_masks,
                memory_tokens=memory_tokens,
                memory_mask=memory_mask,
                **kwargs,
            )
        # ##modified by wangruiyu to adapt memory mechanism
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # Compute image and language key value cache
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
        )
        # ##modified by wangruiyu to adapt memory mechanism
        return self._integrate_actions(
            lambda input_x_t, current_timestep: self.denoise_step(
                x_t=input_x_t,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                timestep=current_timestep,
                # ##modified by wangruiyu to adapt memory mechanism
                memory_tokens=memory_tokens,
                memory_mask=memory_mask,
                # ##modified by wangruiyu to adapt memory mechanism
            ),
            noise,
            **kwargs,
        )
        # ##modified by wangruiyu to adapt memory mechanism

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        # ##modified by wangruiyu to adapt memory mechanism
        memory_tokens=None,
        memory_mask=None,
        # ##modified by wangruiyu to adapt memory mechanism
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            # ##modified by wangruiyu to adapt memory mechanism
            memory_tokens=memory_tokens,
            memory_mask=memory_mask,
            # ##modified by wangruiyu to adapt memory mechanism
        )
        if past_key_values is not None:
            # Self-attention layers append suffix K/V in place; restore the prefix for the next step.
            past_key_values.crop(prefix_len)
        # ##modified by wangruiyu to adapt memory mechanism
        return self._project_suffix_actions(outputs_embeds[-1])
        # ##modified by wangruiyu to adapt memory mechanism

    # ##modified by wangruiyu to adapt memory mechanism
    def sample_actions_with_memory_expert(
        self,
        noise: Tensor,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        memory_tokens: Tensor,
        memory_mask: Tensor | None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        return self._integrate_actions(
            lambda input_x_t, current_timestep: self.denoise_step_full_prefix(
                prefix_embs=prefix_embs,
                prefix_pad_masks=prefix_pad_masks,
                prefix_att_masks=prefix_att_masks,
                x_t=input_x_t,
                timestep=current_timestep,
                memory_tokens=memory_tokens,
                memory_mask=memory_mask,
            ),
            noise,
            **kwargs,
        )

    def denoise_step_full_prefix(
        self,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        x_t,
        timestep,
        memory_tokens=None,
        memory_mask=None,
    ):
        """Denoise with prefix and action in the same forward pass for memory expert mode.

        Memory expert mode needs a full [memory, prefix, action] attention mask so action can attend
        memory and VLM/prefix while VLM/prefix remains closed to memory. That is incompatible with
        the cached prefix-only denoising path, so this path intentionally uses use_cache=False.
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)
        att_2d_masks, position_ids = self._build_full_attention_inputs(
            prefix_pad_masks,
            prefix_att_masks,
            suffix_pad_masks,
            suffix_att_masks,
        )
        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            memory_tokens=memory_tokens,
            memory_mask=memory_mask,
        )
        return self._project_suffix_actions(outputs_embeds[-1])
    # ##modified by wangruiyu to adapt memory mechanism
