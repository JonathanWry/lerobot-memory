#!/usr/bin/env python

import contextlib
import json
import logging
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import datasets
import jsonlines
import numpy as np
import packaging.version
import torch
import torch.utils.data
from datasets import load_dataset
from huggingface_hub import snapshot_download

from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.import_utils import get_safe_default_video_backend
from lerobot.utils.utils import flatten_dict, unflatten_dict

from .compute_stats import aggregate_stats
from .feature_utils import check_delta_timestamps, get_delta_indices, get_hf_features_from_features
from .io_utils import hf_transform_to_torch
from .video_utils import decode_video_frames

logger = logging.getLogger(__name__)

CODEBASE_VERSION = "v2.1"
INFO_PATH = "meta/info.json"
EPISODES_PATH = "meta/episodes.jsonl"
EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
TASKS_PATH = "meta/tasks.jsonl"

STATE_SOURCE_KEY = "observation.qpos"
STATE_TARGET_KEY = "observation.state"
ACTION_SOURCE_KEY = "real_action"
ACTION_TARGET_KEY = "action"


def _load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def _load_jsonlines(path: Path) -> list[dict]:
    with jsonlines.open(path, "r") as reader:
        return list(reader)


def _cast_stats_to_numpy(stats: dict) -> dict[str, dict[str, np.ndarray]]:
    stats = {key: np.atleast_1d(np.array(value)) for key, value in flatten_dict(stats).items()}
    return unflatten_dict(stats)


def _load_tasks(root: Path) -> tuple[dict[int, str], dict[str, int]]:
    tasks = _load_jsonlines(root / TASKS_PATH)
    task_by_idx = {int(item["task_index"]): item["task"] for item in sorted(tasks, key=lambda x: x["task_index"])}
    return task_by_idx, {task: idx for idx, task in task_by_idx.items()}


def _load_episodes(root: Path) -> dict[int, dict]:
    episodes = _load_jsonlines(root / EPISODES_PATH)
    return _V21Episodes({int(item["episode_index"]): item for item in sorted(episodes, key=lambda x: x["episode_index"])})


class _V21Episodes(dict):
    def __init__(self, episodes: dict[int, dict]):
        super().__init__(episodes)
        starts, stops = [], []
        cursor = 0
        for ep_idx in sorted(episodes):
            starts.append(cursor)
            cursor += int(episodes[ep_idx]["length"])
            stops.append(cursor)
        self._dataset_from_index = starts
        self._dataset_to_index = stops

    def __getitem__(self, key):
        if key == "dataset_from_index":
            return self._dataset_from_index
        if key == "dataset_to_index":
            return self._dataset_to_index
        if key == "tasks":
            return [self[idx].get("tasks", []) for idx in sorted(self)]
        return super().__getitem__(key)


def _load_episodes_stats(root: Path) -> dict[int, dict]:
    path = root / EPISODES_STATS_PATH
    if not path.exists():
        return {}
    rows = _load_jsonlines(path)
    return {int(item["episode_index"]): _cast_stats_to_numpy(item["stats"]) for item in rows}


def _map_features(features: dict) -> dict:
    mapped = deepcopy(features)
    if STATE_SOURCE_KEY in mapped and STATE_TARGET_KEY not in mapped:
        mapped[STATE_TARGET_KEY] = deepcopy(mapped[STATE_SOURCE_KEY])
    if ACTION_SOURCE_KEY in mapped:
        mapped[ACTION_TARGET_KEY] = deepcopy(mapped[ACTION_SOURCE_KEY])
    return mapped


def _map_stats(stats: dict) -> dict:
    mapped = deepcopy(stats) if stats else {}
    if STATE_SOURCE_KEY in mapped and STATE_TARGET_KEY not in mapped:
        mapped[STATE_TARGET_KEY] = deepcopy(mapped[STATE_SOURCE_KEY])
    if ACTION_SOURCE_KEY in mapped:
        mapped[ACTION_TARGET_KEY] = deepcopy(mapped[ACTION_SOURCE_KEY])
    return mapped


def _get_episode_data_index(episodes: dict[int, dict], selected_episodes: list[int] | None = None) -> dict[str, torch.Tensor]:
    ep_indices = selected_episodes if selected_episodes is not None else sorted(episodes)
    starts, stops = {}, {}
    cursor = 0
    for ep_idx in ep_indices:
        length = int(episodes[ep_idx]["length"])
        starts[ep_idx] = cursor
        cursor += length
        stops[ep_idx] = cursor
    max_ep = max(ep_indices) if ep_indices else -1
    from_arr = torch.zeros(max_ep + 1, dtype=torch.long)
    to_arr = torch.zeros(max_ep + 1, dtype=torch.long)
    for ep_idx in ep_indices:
        from_arr[ep_idx] = starts[ep_idx]
        to_arr[ep_idx] = stops[ep_idx]
    return {"from": from_arr, "to": to_arr}


def _get_absolute_to_relative_idx(episodes: dict[int, dict], selected_episodes: list[int] | None) -> dict[int, int] | None:
    if selected_episodes is None:
        return None
    mapping = {}
    relative_idx = 0
    starts = episodes["dataset_from_index"]
    for ep_idx in selected_episodes:
        start = int(starts[ep_idx])
        length = int(episodes[ep_idx]["length"])
        for offset in range(length):
            mapping[start + offset] = relative_idx
            relative_idx += 1
    return mapping


def _is_valid_version(version: str) -> bool:
    with contextlib.suppress(packaging.version.InvalidVersion):
        packaging.version.parse(version)
        return True
    return False


def is_lerobot_v21_dataset(root: str | Path | None) -> bool:
    if root is None:
        return False
    path = Path(root) / INFO_PATH
    if not path.exists():
        return False
    info = _load_json(path)
    return str(info.get("codebase_version")) == CODEBASE_VERSION


class LeRobotV21DatasetMetadata:
    def __init__(self, repo_id: str, root: str | Path | None = None, revision: str | None = None, force_cache_sync: bool = False):
        self.repo_id = repo_id
        self.revision = revision if revision else CODEBASE_VERSION
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
        try:
            if force_cache_sync:
                raise FileNotFoundError
            self.load_metadata()
        except (FileNotFoundError, NotADirectoryError):
            self.root.mkdir(exist_ok=True, parents=True)
            self.pull_from_repo(allow_patterns="meta/")
            self.load_metadata()

    def load_metadata(self):
        self.info = _load_json(self.root / INFO_PATH)
        for ft in self.info["features"].values():
            if isinstance(ft.get("shape"), list):
                ft["shape"] = tuple(ft["shape"])
        version = packaging.version.parse(self.info["codebase_version"])
        current = packaging.version.parse(CODEBASE_VERSION)
        if version.major != current.major or version.minor > current.minor:
            raise ValueError(f"Unsupported LeRobot v2.1 compat dataset version: {version}")
        self.tasks, self.task_to_task_index = _load_tasks(self.root)
        self.episodes = _load_episodes(self.root)
        self.episodes_stats = _load_episodes_stats(self.root)
        self.stats = aggregate_stats(list(self.episodes_stats.values())) if self.episodes_stats else {}
        self.info["features"] = _map_features(self.info["features"])
        self.stats = _map_stats(self.stats)
        self.episodes_stats = {ep: _map_stats(stats) for ep, stats in self.episodes_stats.items()}

    def pull_from_repo(self, allow_patterns=None, ignore_patterns=None) -> None:
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    def get_episode_chunk(self, ep_index: int) -> int:
        return ep_index // self.chunks_size

    def get_data_file_path(self, ep_index: int) -> Path:
        return Path(self.data_path.format(episode_chunk=self.get_episode_chunk(ep_index), episode_index=ep_index))

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        return Path(self.video_path.format(episode_chunk=self.get_episode_chunk(ep_index), video_key=vid_key, episode_index=ep_index))

    @property
    def data_path(self) -> str:
        return self.info["data_path"]

    @property
    def video_path(self) -> str | None:
        return self.info.get("video_path")

    @property
    def fps(self) -> int:
        return int(self.info["fps"])

    @property
    def features(self) -> dict[str, dict]:
        return self.info["features"]

    @property
    def image_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def camera_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def depth_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft.get("dtype") in ["depth", "depth_video"]]

    @property
    def has_language_columns(self) -> bool:
        return False

    @property
    def total_episodes(self) -> int:
        return int(self.info["total_episodes"])

    @property
    def total_frames(self) -> int:
        return int(self.info["total_frames"])

    @property
    def total_tasks(self) -> int:
        return int(self.info["total_tasks"])

    @property
    def chunks_size(self) -> int:
        return int(self.info["chunks_size"])


class LeRobotV21Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        return_uint8: bool = False,
        # ##modified by wangruiyu to adapt memory mechanism
        memory_feature_root: str | Path | None = None,
        memory_feature_pattern: str = "features/episode_{episode_index}/token_emb_{frame_index}.npz",
        memory_feature_required: bool = True,
        memory_sampling_type: str = "framesamp",
        memory_num_frames: int | None = None,
        memory_stride: int = 1,
        memory_token_budget: int | None = None,
        memory_kept_indices_pattern: str = "features/episode_{episode_index}/kept_indices.json",
        memory_image_feature_name: str = "image_emb",
        memory_pos_feature_name: str = "pos_emb",
        memory_state_feature_name: str = "state_emb",
        memory_mask_feature_name: str = "mask",
        memory_output_image_key: str = "static_image_emb",
        memory_output_pos_key: str = "static_pos_emb",
        memory_output_state_key: str = "static_state_emb",
        memory_output_mask_key: str = "static_mask",
        recurrent_output_image_key: str = "recur_image_emb",
        recurrent_output_pos_key: str = "recur_pos_emb",
        recurrent_output_state_key: str = "recur_state_emb",
        recurrent_output_mask_key: str = "recur_mask",
        recurrent_max_recur_steps: int | None = None,
        recurrent_max_pretraj_steps: int | None = None,
        recurrent_input_obs_horizon: int | None = None,
        # ##modified by wangruiyu to adapt memory mechanism
        **_: object,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.episodes = episodes
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else get_safe_default_video_backend()
        self.return_uint8 = return_uint8
        self.delta_indices = None
        # ##modified by wangruiyu to adapt memory mechanism
        self.memory_feature_root = Path(memory_feature_root) if memory_feature_root else None
        self.memory_feature_pattern = memory_feature_pattern
        self.memory_feature_required = memory_feature_required
        self.memory_sampling_type = memory_sampling_type
        self.memory_num_frames = memory_num_frames
        self.memory_stride = memory_stride
        self.memory_token_budget = memory_token_budget
        self.memory_kept_indices_pattern = memory_kept_indices_pattern
        self._memory_kept_indices_cache: dict[int, list[tuple[int, int, int]]] = {}
        self.memory_image_feature_name = memory_image_feature_name
        self.memory_pos_feature_name = memory_pos_feature_name
        self.memory_state_feature_name = memory_state_feature_name
        self.memory_mask_feature_name = memory_mask_feature_name
        self.memory_output_image_key = memory_output_image_key
        self.memory_output_pos_key = memory_output_pos_key
        self.memory_output_state_key = memory_output_state_key
        self.memory_output_mask_key = memory_output_mask_key
        self.recurrent_output_image_key = recurrent_output_image_key
        self.recurrent_output_pos_key = recurrent_output_pos_key
        self.recurrent_output_state_key = recurrent_output_state_key
        self.recurrent_output_mask_key = recurrent_output_mask_key
        self.recurrent_max_recur_steps = recurrent_max_recur_steps
        self.recurrent_max_pretraj_steps = recurrent_max_pretraj_steps
        self.recurrent_input_obs_horizon = recurrent_input_obs_horizon
        # ##modified by wangruiyu to adapt memory mechanism

        self.root.mkdir(exist_ok=True, parents=True)
        self.meta = LeRobotV21DatasetMetadata(self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync)

        if self.episodes is not None and self.meta.episodes_stats:
            self.meta.stats = aggregate_stats([self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes])

        try:
            if force_cache_sync:
                raise FileNotFoundError
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()

        self.episode_data_index = _get_episode_data_index(self.meta.episodes, self.episodes)
        self._absolute_to_relative_idx = _get_absolute_to_relative_idx(self.meta.episodes, self.episodes)

        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

        logger.info(
            "Using LeRobot v2.1 compat dataset: root=%s episodes=%s state=%s action=%s cameras=%s",
            self.root,
            "all" if self.episodes is None else len(self.episodes),
            STATE_SOURCE_KEY if STATE_SOURCE_KEY in self.meta.features else STATE_TARGET_KEY,
            ACTION_SOURCE_KEY if ACTION_SOURCE_KEY in self.meta.features else ACTION_TARGET_KEY,
            self.meta.camera_keys,
        )
        # ##modified by wangruiyu to adapt memory mechanism
        if self.memory_feature_root is not None:
            logger.info(
                "Using sidecar memory features: root=%s pattern=%s sampling=%s frames=%s stride=%s budget=%s",
                self.memory_feature_root,
                self.memory_feature_pattern,
                self.memory_sampling_type,
                self.memory_num_frames,
                self.memory_stride,
                self.memory_token_budget,
            )
        # ##modified by wangruiyu to adapt memory mechanism

    def get_episodes_file_paths(self) -> list[Path]:
        episodes = self.episodes if self.episodes is not None else list(range(self.meta.total_episodes))
        fpaths = [self.meta.get_data_file_path(ep_idx) for ep_idx in episodes]
        if self.meta.video_keys:
            fpaths += [self.meta.get_video_file_path(ep_idx, key) for key in self.meta.video_keys for ep_idx in episodes]
        return fpaths

    def download_episodes(self, download_videos: bool = True) -> None:
        files = None
        ignore_patterns = None if download_videos else "videos/"
        if self.episodes is not None:
            files = [str(path) for path in self.get_episodes_file_paths()]
        self.meta.pull_from_repo(allow_patterns=files, ignore_patterns=ignore_patterns)

    def load_hf_dataset(self) -> datasets.Dataset:
        if self.episodes is None:
            hf_dataset = load_dataset("parquet", data_dir=str(self.root / "data"), split="train")
        else:
            files = [str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes]
            hf_dataset = load_dataset("parquet", data_files=files, split="train")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    @property
    def fps(self) -> int:
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        return len(self.hf_dataset)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes) if self.episodes is not None else self.meta.total_episodes

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    @property
    def hf_features(self) -> datasets.Features:
        return self.hf_dataset.features if self.hf_dataset is not None else get_hf_features_from_features(self.features)

    @property
    def absolute_to_relative_idx(self) -> dict[int, int] | None:
        return self._absolute_to_relative_idx

    def _get_query_indices(self, idx: int, ep_idx: int):
        ep_start = int(self.episode_data_index["from"][ep_idx].item())
        ep_end = int(self.episode_data_index["to"][ep_idx].item())
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor([(idx + delta < ep_start) or (idx + delta >= ep_end) for delta in delta_idx])
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _storage_key(self, key: str) -> str:
        if key == STATE_TARGET_KEY and STATE_SOURCE_KEY in self.hf_dataset.column_names:
            return STATE_SOURCE_KEY
        if key == ACTION_TARGET_KEY and ACTION_SOURCE_KEY in self.hf_dataset.column_names:
            return ACTION_SOURCE_KEY
        return key

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        result = {}
        for key, q_idx in query_indices.items():
            if key in self.meta.video_keys:
                continue
            storage_key = self._storage_key(key)
            result[key] = torch.stack(list(self.hf_dataset.select(q_idx)[storage_key]))
        return result

    def _get_query_timestamps(self, current_ts: float, query_indices: dict[str, list[int]] | None = None) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                timestamps = self.hf_dataset.select(query_indices[key])["timestamp"]
                query_timestamps[key] = torch.stack(list(timestamps)).tolist()
            else:
                query_timestamps[key] = [current_ts]
        return query_timestamps

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend, return_uint8=self.return_uint8)
            item[vid_key] = frames.squeeze(0)
        return item


    # ##modified by wangruiyu to adapt memory mechanism
    @staticmethod
    def _even_memory_indices(step_idx: int, max_size: int) -> list[int]:
        if max_size < 1:
            raise ValueError(f"memory max_size must be >= 1, got {max_size}")
        if step_idx < max_size:
            return list(range(step_idx + 1))
        return np.linspace(0, step_idx, max_size, dtype=np.int32).tolist()

    def _get_memory_frame_indices(self, idx: int, ep_idx: int, max_frames: int) -> list[int]:
        ep_start = int(self.episode_data_index["from"][ep_idx].item())
        local_frame_idx = idx - ep_start
        return self._even_memory_indices(local_frame_idx, max_frames)

    def _get_memory_feature_path(self, ep_idx: int, frame_idx: int) -> Path:
        relpath = self.memory_feature_pattern.format(
            episode_index=ep_idx,
            frame_index=frame_idx,
            episode_chunk=self.meta.get_episode_chunk(ep_idx),
        )
        path = Path(relpath)
        if not path.is_absolute():
            path = self.memory_feature_root / path
        return path

    # ##modified by wangruiyu to adapt memory mechanism
    def _get_memory_kept_indices_path(self, ep_idx: int) -> Path:
        relpath = self.memory_kept_indices_pattern.format(
            episode_index=ep_idx,
            episode_chunk=self.meta.get_episode_chunk(ep_idx),
        )
        path = Path(relpath)
        if not path.is_absolute():
            path = self.memory_feature_root / path
        return path

    def _load_memory_kept_indices(self, ep_idx: int) -> list[tuple[int, int, int]]:
        cached = self._memory_kept_indices_cache.get(ep_idx)
        if cached is not None:
            return cached
        path = self._get_memory_kept_indices_path(ep_idx)
        if not path.exists():
            if self.memory_feature_required:
                raise FileNotFoundError(f"Missing token-drop kept indices file: {path}")
            return []
        with open(path) as f:
            raw_indices = json.load(f)
        kept_indices = [tuple(int(v) for v in item) for item in raw_indices]
        self._memory_kept_indices_cache[ep_idx] = kept_indices
        return kept_indices

    @staticmethod
    def _filter_token_dropping_indices(
        kept_indices: list[tuple[int, int, int]], step_idx: int, token_budget: int
    ) -> list[tuple[int, int, int]]:
        filtered = [item for item in kept_indices if item[0] <= step_idx]
        return sorted(filtered[-token_budget:])

    @staticmethod
    def _select_token_feature(array: np.ndarray, view_idx: int, patch_idx: int, tokens_per_image: int) -> np.ndarray:
        array = np.asarray(array)
        if array.ndim == 3:
            return array[view_idx, patch_idx]
        if array.ndim == 2:
            return array[view_idx * tokens_per_image + patch_idx]
        raise ValueError(f"Expected memory feature array with 2 or 3 dims, got shape={array.shape}")

    @staticmethod
    def _get_feature_array(npz, preferred_key: str, fallback_keys: tuple[str, ...]):
        for key in (preferred_key, *fallback_keys):
            if key in npz:
                return np.asarray(npz[key])
        raise KeyError(f"Missing memory feature key {preferred_key!r}; available keys={list(npz.keys())}")

    def _load_memory_frame(self, ep_idx: int, frame_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        path = self._get_memory_feature_path(ep_idx, frame_idx)
        if not path.exists():
            if self.memory_feature_required:
                raise FileNotFoundError(f"Missing memory feature file: {path}")
            return None
        if path.suffix != ".npz":
            raise ValueError(
                f"Expected memory feature sidecar as .npz, got {path}. "
                "Convert pickle .npy features to plain .npz before training."
            )
        with np.load(path) as npz:
            image_feature_name = self.memory_image_feature_name
            pos_feature_name = self.memory_pos_feature_name
            image_fallbacks = ("image_emb_8x8", "image_emb_4x4", "static_image_emb")
            pos_fallbacks = ("pos_emb_8x8", "pos_emb_4x4", "static_pos_emb")
            if self.memory_sampling_type in {"tokendrop", "recurrent"} and image_feature_name == "image_emb":
                image_feature_name = "image_emb_8x8"
            if self.memory_sampling_type in {"tokendrop", "recurrent"} and pos_feature_name == "pos_emb":
                pos_feature_name = "pos_emb_8x8"
            image_emb = self._get_feature_array(
                npz,
                image_feature_name,
                image_fallbacks,
            )
            pos_emb = self._get_feature_array(
                npz,
                pos_feature_name,
                pos_fallbacks,
            )
            state_emb = self._get_feature_array(
                npz,
                self.memory_state_feature_name,
                ("state_emb", "static_state_emb"),
            )
            if self.memory_mask_feature_name in npz:
                mask = np.asarray(npz[self.memory_mask_feature_name]).astype(bool)
            elif "static_mask" in npz:
                mask = np.asarray(npz["static_mask"]).astype(bool)
            else:
                mask = np.ones(image_emb.shape[0], dtype=bool)
        return image_emb, pos_emb, state_emb, mask

    def _load_token_drop_memory_features(self, idx: int, ep_idx: int) -> dict:
        if self.memory_feature_root is None or self.memory_token_budget is None:
            return {}
        ep_start = int(self.episode_data_index["from"][ep_idx].item())
        local_frame_idx = idx - ep_start
        kept_indices = self._load_memory_kept_indices(ep_idx)
        sorted_kept_indices = self._filter_token_dropping_indices(
            kept_indices, local_frame_idx, self.memory_token_budget
        )
        if not sorted_kept_indices:
            return {}

        frames: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for frame_idx, _, _ in sorted_kept_indices:
            if frame_idx not in frames:
                frame = self._load_memory_frame(ep_idx, frame_idx)
                if frame is None:
                    continue
                frames[frame_idx] = frame
        if not frames:
            return {}

        first_frame = next(iter(frames.values()))
        image_dim = int(np.asarray(first_frame[0]).shape[-1])
        pos_dim = int(np.asarray(first_frame[1]).shape[-1])
        state_dim = int(np.asarray(first_frame[2]).shape[-1])
        if first_frame[0].ndim == 3:
            tokens_per_image = int(first_frame[0].shape[1])
        else:
            view_count = max(view_idx for _, view_idx, _ in sorted_kept_indices) + 1
            tokens_per_image = max(1, first_frame[0].shape[0] // view_count)

        image_emb = np.zeros((self.memory_token_budget, image_dim), dtype=np.float32)
        pos_emb = np.zeros((self.memory_token_budget, pos_dim), dtype=np.float32)
        state_emb = np.zeros((self.memory_token_budget, state_dim), dtype=np.float32)
        mask = np.zeros((self.memory_token_budget,), dtype=bool)
        for out_idx, (frame_idx, view_idx, patch_idx) in enumerate(sorted_kept_indices[: self.memory_token_budget]):
            if frame_idx not in frames:
                continue
            frame_image, frame_pos, frame_state, _ = frames[frame_idx]
            image_emb[out_idx] = self._select_token_feature(frame_image, view_idx, patch_idx, tokens_per_image)
            pos_emb[out_idx] = self._select_token_feature(frame_pos, view_idx, patch_idx, tokens_per_image)
            # ##modified by wangruiyu to adapt memory mechanism
            # Sidecar may store RoboMME-style state_emb [D] or expanded [T, D].
            state_arr = np.asarray(frame_state)
            if state_arr.ndim == 2:
                state_arr = state_arr[0]
            state_emb[out_idx] = state_arr
            # ##modified by wangruiyu to adapt memory mechanism
            mask[out_idx] = True

        return {
            self.memory_output_image_key: torch.as_tensor(image_emb, dtype=torch.float32),
            self.memory_output_pos_key: torch.as_tensor(pos_emb, dtype=torch.float32),
            self.memory_output_state_key: torch.as_tensor(state_emb, dtype=torch.float32),
            self.memory_output_mask_key: torch.as_tensor(mask, dtype=torch.bool),
        }

    def _load_memory_features(self, idx: int, ep_idx: int) -> dict:
        if self.memory_feature_root is None or self.memory_num_frames is None:
            return {}
        ep_start = int(self.episode_data_index["from"][ep_idx].item())
        local_frame_idx = idx - ep_start
        current_frame = self._load_memory_frame(ep_idx, local_frame_idx)
        if current_frame is None:
            return {}

        tokens_per_frame = int(current_frame[0].shape[0])
        max_frames = self.memory_num_frames
        if self.memory_token_budget is not None:
            max_frames = min(max_frames, max(1, self.memory_token_budget // tokens_per_frame))

        frames = []
        for frame_idx in self._get_memory_frame_indices(idx, ep_idx, max_frames):
            frame = self._load_memory_frame(ep_idx, frame_idx)
            if frame is not None:
                frames.append(frame)
        if not frames:
            return {}

        padding_frames = max_frames - len(frames)
        # ##modified by wangruiyu to adapt memory mechanism
        def _framesamp_state(state: np.ndarray) -> np.ndarray:
            state = np.asarray(state)
            if state.ndim == 1:
                return np.broadcast_to(state[None, :], (tokens_per_frame, state.shape[0])).copy()
            if state.ndim == 2 and state.shape[0] == tokens_per_frame:
                return state
            if state.ndim == 2 and state.shape[0] == 1:
                return np.broadcast_to(state, (tokens_per_frame, state.shape[1])).copy()
            raise ValueError(f"Unsupported framesamp state_emb shape={state.shape} tokens_per_frame={tokens_per_frame}")

        image_emb = np.concatenate([frame[0] for frame in frames], axis=0)
        pos_emb = np.concatenate([frame[1] for frame in frames], axis=0)
        state_emb = np.concatenate([_framesamp_state(frame[2]) for frame in frames], axis=0)
        mask = np.concatenate([frame[3] for frame in frames], axis=0)
        # ##modified by wangruiyu to adapt memory mechanism
        if padding_frames > 0:
            image_emb = np.concatenate(
                [image_emb, np.zeros((padding_frames * tokens_per_frame, image_emb.shape[-1]), dtype=image_emb.dtype)],
                axis=0,
            )
            pos_emb = np.concatenate(
                [pos_emb, np.zeros((padding_frames * tokens_per_frame, pos_emb.shape[-1]), dtype=pos_emb.dtype)],
                axis=0,
            )
            state_emb = np.concatenate(
                [state_emb, np.zeros((padding_frames * tokens_per_frame, state_emb.shape[-1]), dtype=state_emb.dtype)],
                axis=0,
            )
            mask = np.concatenate([mask, np.zeros((padding_frames * tokens_per_frame,), dtype=bool)], axis=0)

        return {
            self.memory_output_image_key: torch.as_tensor(image_emb, dtype=torch.float32),
            self.memory_output_pos_key: torch.as_tensor(pos_emb, dtype=torch.float32),
            self.memory_output_state_key: torch.as_tensor(state_emb, dtype=torch.float32),
            self.memory_output_mask_key: torch.as_tensor(mask, dtype=torch.bool),
        }

    def _get_item_scalar(self, item: dict, key: str, default: int) -> int:
        if key not in item:
            return default
        value = item[key]
        if hasattr(value, "item"):
            return int(value.item())
        return int(value)

    def _get_recurrent_indices(self, local_frame_idx: int, exec_start_idx: int) -> list[int]:
        if self.recurrent_max_recur_steps is None:
            raise ValueError("recurrent_max_recur_steps must be configured for recurrent memory loading")
        horizon = self.recurrent_input_obs_horizon or self.memory_num_frames or 16
        horizon = max(1, int(horizon))
        max_pretraj = self.recurrent_max_pretraj_steps or 40

        if exec_start_idx == 0:
            if local_frame_idx < horizon:
                indices = [local_frame_idx]
            else:
                start_idx = local_frame_idx % horizon
                indices = list(range(start_idx, local_frame_idx + 1, horizon))
                indices = indices[-self.recurrent_max_recur_steps :]
        else:
            if exec_start_idx <= horizon * 2:
                video_indices = list(range(0, exec_start_idx, max(1, horizon // 2)))
            elif exec_start_idx <= max_pretraj * horizon:
                video_indices = list(range(0, exec_start_idx, horizon))
            else:
                video_indices = np.linspace(0, exec_start_idx - 1, max_pretraj, dtype=np.int32).tolist()

            if local_frame_idx - exec_start_idx < horizon:
                rest_indices = [local_frame_idx]
            else:
                start_idx = (local_frame_idx - exec_start_idx) % horizon + exec_start_idx
                rest_indices = list(range(start_idx, local_frame_idx + 1, horizon))
            indices = (video_indices + rest_indices)[-self.recurrent_max_recur_steps :]

        return [int(index) for index in indices]

    @staticmethod
    def _ensure_recurrent_feature_shape(array: np.ndarray) -> np.ndarray:
        array = np.asarray(array)
        if array.ndim == 3:
            return array
        if array.ndim == 2:
            return array[None, :, :]
        raise ValueError(f"Expected recurrent feature array with shape [V,P,D] or [P,D], got {array.shape}")

    def _load_recurrent_memory_features(self, idx: int, ep_idx: int, item: dict) -> dict:
        if self.memory_feature_root is None or self.recurrent_max_recur_steps is None:
            return {}
        ep_start = int(self.episode_data_index["from"][ep_idx].item())
        local_frame_idx = idx - ep_start
        exec_start_idx = self._get_item_scalar(item, "exec_start_idx", 0)
        indices = self._get_recurrent_indices(local_frame_idx, exec_start_idx)

        frames = []
        for frame_idx in indices:
            frame = self._load_memory_frame(ep_idx, frame_idx)
            if frame is not None:
                frames.append(frame)
        if not frames:
            return {}

        image_emb = np.stack([self._ensure_recurrent_feature_shape(frame[0]) for frame in frames], axis=0)
        pos_emb = np.stack([self._ensure_recurrent_feature_shape(frame[1]) for frame in frames], axis=0)
        state_emb = np.stack([np.asarray(frame[2]) for frame in frames], axis=0)
        recur_mask = np.ones((len(frames),), dtype=bool)

        padding = self.recurrent_max_recur_steps - len(frames)
        if padding > 0:
            image_emb = np.concatenate(
                [np.zeros((padding, *image_emb.shape[1:]), dtype=image_emb.dtype), image_emb],
                axis=0,
            )
            pos_emb = np.concatenate(
                [np.zeros((padding, *pos_emb.shape[1:]), dtype=pos_emb.dtype), pos_emb],
                axis=0,
            )
            state_emb = np.concatenate(
                [np.zeros((padding, *state_emb.shape[1:]), dtype=state_emb.dtype), state_emb],
                axis=0,
            )
            recur_mask = np.concatenate([np.zeros((padding,), dtype=bool), recur_mask], axis=0)

        return {
            self.recurrent_output_image_key: torch.as_tensor(image_emb, dtype=torch.float32),
            self.recurrent_output_pos_key: torch.as_tensor(pos_emb, dtype=torch.float32),
            self.recurrent_output_state_key: torch.as_tensor(state_emb, dtype=torch.float32),
            self.recurrent_output_mask_key: torch.as_tensor(recur_mask, dtype=torch.bool),
        }
    # ##modified by wangruiyu to adapt memory mechanism

    def _apply_memorybench_aliases(self, item: dict) -> dict:
        if STATE_SOURCE_KEY in item and STATE_TARGET_KEY not in item:
            item[STATE_TARGET_KEY] = item[STATE_SOURCE_KEY]
        has_action_delta = self.delta_indices is not None and ACTION_TARGET_KEY in self.delta_indices
        if ACTION_SOURCE_KEY in item and (ACTION_TARGET_KEY not in item or not has_action_delta):
            item[ACTION_TARGET_KEY] = item[ACTION_SOURCE_KEY]
        return item

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = int(item["episode_index"].item())

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            item = {**item, **padding, **self._query_hf_dataset(query_indices)}

        if self.meta.video_keys:
            current_ts = float(item["timestamp"].item())
            item = {**self._query_videos(self._get_query_timestamps(current_ts, query_indices), ep_idx), **item}

        item = self._apply_memorybench_aliases(item)
        # ##modified by wangruiyu to adapt memory mechanism
        if self.memory_sampling_type == "recurrent":
            item.update(self._load_recurrent_memory_features(idx, ep_idx, item))
        elif self.memory_sampling_type == "tokendrop":
            item.update(self._load_token_drop_memory_features(idx, ep_idx))
        else:
            item.update(self._load_memory_features(idx, ep_idx))
        # ##modified by wangruiyu to adapt memory mechanism

        if self.image_transforms is not None:
            for cam in self.meta.camera_keys:
                item[cam] = self.image_transforms(item[cam])

        task_idx = int(item["task_index"].item())
        item["task"] = self.meta.tasks[task_idx]
        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Number of selected episodes: '{self.num_episodes}',\n"
            f"    Number of selected samples: '{self.num_frames}',\n"
            f"    Features: '{list(self.features)}',\n"
            "})',\n"
        )
