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

from typing import Any

import torch

from lerobot.processor.converters import batch_to_transition
from lerobot.types import TransitionKey

from lerobot.processor import (
    NewLineTaskProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)

from .configuration_smolvla import SmolVLAConfig


# ##modified by wangruiyu to adapt memory mechanism
def _memory_passthrough_batch_to_transition(memory_keys: tuple[str, ...]):
    def convert(batch: dict[str, Any]):
        transition = batch_to_transition(batch)
        comp = transition.setdefault(TransitionKey.COMPLEMENTARY_DATA, {})
        for key in memory_keys:
            if key in batch:
                comp[key] = batch[key]
        return transition

    return convert
# ##modified by wangruiyu to adapt memory mechanism


def make_smolvla_pre_post_processors(
    config: SmolVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the SmolVLA policy.

    The pre-processing pipeline prepares input data for the model by:
    1.  Renaming features to match pretrained configurations.
    2.  Normalizing input and output features based on dataset statistics.
    3.  Adding a batch dimension.
    4.  Ensuring the language task description ends with a newline character.
    5.  Tokenizing the language task description.
    6.  Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1.  Moving data to the CPU.
    2.  Unnormalizing the output actions to their original scale.

    Args:
        config: The configuration object for the SmolVLA policy.
        dataset_stats: A dictionary of statistics for normalization.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    steps = make_default_policy_processor_steps(config, dataset_stats)

    input_steps = [
        steps.rename_observations,  # To mimic the same processor as pretrained one
        steps.add_batch_dim,
        NewLineTaskProcessorStep(),
        TokenizerProcessorStep(
            tokenizer_name=config.vlm_model_name,
            padding=config.pad_language_to,
            padding_side="right",
            max_length=config.tokenizer_max_length,
        ),
        steps.to_device,
        steps.normalize,
    ]
    output_steps = [
        steps.unnormalize,
        steps.to_cpu,
    ]
    preprocessor, postprocessor = make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
    # ##modified by wangruiyu to adapt memory mechanism
    if config.memory_enabled:
        memory_keys = (
            config.memory_image_emb_key,
            config.memory_pos_emb_key,
            config.memory_state_emb_key,
            config.memory_mask_key,
            "recur_image_emb",
            "recur_pos_emb",
            "recur_state_emb",
            "recur_mask",
        )
        preprocessor.to_transition = _memory_passthrough_batch_to_transition(memory_keys)
    # ##modified by wangruiyu to adapt memory mechanism
    return preprocessor, postprocessor
