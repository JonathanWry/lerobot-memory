#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import logging
import math
from pprint import pformat

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.configs.rewards import RewardModelConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, IMAGENET_STATS, OBS_PREFIX, REWARD

from .dataset_metadata import LeRobotDatasetMetadata
from .lerobot_dataset import LeRobotDataset
from .lerobot_v21_dataset import LeRobotV21Dataset, LeRobotV21DatasetMetadata, is_lerobot_v21_dataset
from .multi_dataset import MultiLeRobotDataset
from .streaming_dataset import StreamingLeRobotDataset


def resolve_delta_timestamps(
    cfg: PreTrainedConfig | RewardModelConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the config.

    Args:
        cfg (PreTrainedConfig | RewardModelConfig): The config to read delta_indices from. Both
            ``PreTrainedConfig`` and concrete ``RewardModelConfig`` subclasses expose the
            ``{observation,action,reward}_delta_indices`` properties used below.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def _dataset_cls_and_meta(cfg: TrainPipelineConfig):
    if is_lerobot_v21_dataset(cfg.dataset.root):
        return LeRobotV21Dataset, LeRobotV21DatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
    return LeRobotDataset, LeRobotDatasetMetadata(
        cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
    )


# ##modified by wangruiyu to adapt memory mechanism
def _memory_dataset_kwargs(cfg: TrainPipelineConfig, dataset_cls: type) -> dict:
    if dataset_cls is not LeRobotV21Dataset:
        return {}
    # ##modified by wangruiyu to adapt memory mechanism
    trainable_memory_family = getattr(cfg.trainable_config, "memory_family", "none")
    if trainable_memory_family == "robomme_recurrent":
        memory_sampling_type = "recurrent"
    elif cfg.dataset.memory_sampling_type != "framesamp":
        memory_sampling_type = cfg.dataset.memory_sampling_type
    else:
        memory_sampling_type = getattr(cfg.trainable_config, "robomme_sampling_type", "framesamp")
    # ##modified by wangruiyu to adapt memory mechanism
    return {
        "memory_feature_root": cfg.dataset.memory_feature_root,
        "memory_feature_pattern": cfg.dataset.memory_feature_pattern,
        "memory_feature_required": cfg.dataset.memory_feature_required,
        "memory_sampling_type": memory_sampling_type,
        "memory_num_frames": cfg.dataset.memory_num_frames
        if cfg.dataset.memory_num_frames is not None
        else getattr(cfg.trainable_config, "visual_memory_frames", None),
        "memory_stride": cfg.dataset.memory_stride,
        "memory_token_budget": cfg.dataset.memory_token_budget
        if cfg.dataset.memory_token_budget is not None
        else getattr(cfg.trainable_config, "robomme_token_budget", None),
        "memory_kept_indices_pattern": cfg.dataset.memory_kept_indices_pattern,
        "memory_image_feature_name": cfg.dataset.memory_image_feature_name,
        "memory_pos_feature_name": cfg.dataset.memory_pos_feature_name,
        "memory_state_feature_name": cfg.dataset.memory_state_feature_name,
        "memory_mask_feature_name": cfg.dataset.memory_mask_feature_name,
        "memory_output_image_key": cfg.dataset.memory_output_image_key,
        "memory_output_pos_key": cfg.dataset.memory_output_pos_key,
        "memory_output_state_key": cfg.dataset.memory_output_state_key,
        "memory_output_mask_key": cfg.dataset.memory_output_mask_key,
        "recurrent_output_image_key": getattr(cfg.dataset, "recurrent_output_image_key", "recur_image_emb"),
        "recurrent_output_pos_key": getattr(cfg.dataset, "recurrent_output_pos_key", "recur_pos_emb"),
        "recurrent_output_state_key": getattr(cfg.dataset, "recurrent_output_state_key", "recur_state_emb"),
        "recurrent_output_mask_key": getattr(cfg.dataset, "recurrent_output_mask_key", "recur_mask"),
        "recurrent_max_recur_steps": cfg.dataset.recurrent_max_recur_steps
        if getattr(cfg.dataset, "recurrent_max_recur_steps", None) is not None
        else getattr(cfg.trainable_config, "recurrent_max_recur_steps", None),
        "recurrent_max_pretraj_steps": cfg.dataset.recurrent_max_pretraj_steps
        if getattr(cfg.dataset, "recurrent_max_pretraj_steps", None) is not None
        else getattr(cfg.trainable_config, "recurrent_max_pretraj_steps", None),
        "recurrent_input_obs_horizon": cfg.dataset.recurrent_input_obs_horizon
        if getattr(cfg.dataset, "recurrent_input_obs_horizon", None) is not None
        else (
            getattr(cfg.trainable_config, "recurrent_input_obs_horizon", None)
            or getattr(cfg.trainable_config, "visual_memory_frames", None)
        ),
    }
# ##modified by wangruiyu to adapt memory mechanism


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    if isinstance(cfg.dataset.repo_id, str):
        dataset_cls, ds_meta = _dataset_cls_and_meta(cfg)
        delta_timestamps = resolve_delta_timestamps(cfg.trainable_config, ds_meta)
        if not cfg.dataset.streaming:
            dataset = dataset_cls(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                return_uint8=True,
                depth_output_unit=cfg.dataset.depth_output_unit,
                tolerance_s=cfg.tolerance_s,
                # ##modified by wangruiyu to adapt memory mechanism
                **_memory_dataset_kwargs(cfg, dataset_cls),
                # ##modified by wangruiyu to adapt memory mechanism
            )
        else:
            dataset = StreamingLeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                max_num_shards=cfg.num_workers,
                tolerance_s=cfg.tolerance_s,
                return_uint8=True,
            )
    else:
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")
        dataset = MultiLeRobotDataset(
            cfg.dataset.repo_id,
            # TODO(aliberts): add proper support for multi dataset
            # delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            if key in dataset.meta.depth_keys:
                continue  # Exclude depth keys from ImageNet stats
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset


def make_train_eval_datasets(
    cfg: TrainPipelineConfig,
) -> tuple[LeRobotDataset | MultiLeRobotDataset, LeRobotDataset | None]:
    """Create train and optional eval datasets by splitting episodes based on eval_split.

    The last ceil(n_episodes * eval_split) episodes per task are held out for evaluation.
    If eval_split == 0.0, returns (full_dataset, None).
    """
    full_dataset = make_dataset(cfg)

    if cfg.dataset.eval_split == 0.0:
        return full_dataset, None

    base_episodes = (
        full_dataset.episodes if full_dataset.episodes is not None else list(range(full_dataset.num_episodes))
    )

    task_to_episodes: dict[str, list[int]] = {}
    for ep_idx in base_episodes:
        if isinstance(full_dataset.meta.episodes, dict):
            tasks = full_dataset.meta.episodes[ep_idx].get("tasks", [])
        else:
            tasks = full_dataset.meta.episodes["tasks"][ep_idx]
        task_key = tasks[0] if tasks else ""
        task_to_episodes.setdefault(task_key, []).append(ep_idx)

    train_episodes, eval_episodes = [], []
    for eps in task_to_episodes.values():
        n_eval = math.ceil(len(eps) * cfg.dataset.eval_split)
        train_episodes.extend(eps[: len(eps) - n_eval])
        eval_episodes.extend(eps[len(eps) - n_eval :])

    if not train_episodes:
        raise ValueError(
            f"eval_split={cfg.dataset.eval_split} leaves 0 training episodes from {len(base_episodes)} total."
        )

    logging.info(
        f"Train/eval split: {len(train_episodes)} train, {len(eval_episodes)} eval "
        f"(eval_split={cfg.dataset.eval_split}, {len(task_to_episodes)} tasks)"
    )

    delta_timestamps = resolve_delta_timestamps(cfg.trainable_config, full_dataset.meta)

    train_image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    dataset_cls = LeRobotV21Dataset if isinstance(full_dataset, LeRobotV21Dataset) else LeRobotDataset

    train_dataset = dataset_cls(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=train_episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=train_image_transforms,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        return_uint8=True,
        tolerance_s=cfg.tolerance_s,
        # ##modified by wangruiyu to adapt memory mechanism
        **_memory_dataset_kwargs(cfg, dataset_cls),
        # ##modified by wangruiyu to adapt memory mechanism
    )

    eval_dataset = dataset_cls(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=eval_episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=None,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        return_uint8=True,
        tolerance_s=cfg.tolerance_s,
        # ##modified by wangruiyu to adapt memory mechanism
        **_memory_dataset_kwargs(cfg, dataset_cls),
        # ##modified by wangruiyu to adapt memory mechanism
    )

    if cfg.dataset.use_imagenet_stats:
        for ds in (train_dataset, eval_dataset):
            for key in ds.meta.camera_keys:
                for stats_type, stats in IMAGENET_STATS.items():
                    ds.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return train_dataset, eval_dataset
