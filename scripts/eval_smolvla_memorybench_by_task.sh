#!/usr/bin/env bash
set -euo pipefail

WANGRUIYU_ROOT="${WANGRUIYU_ROOT:-/pfs/user/wangruiyu}"
CODE_ROOT="${CODE_ROOT:-${WANGRUIYU_ROOT}}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${CODE_ROOT}/lerobot-main}"
KAI0_ROOT="${KAI0_ROOT:-${CODE_ROOT}/kai0}"
POLICY_PATH="${POLICY_PATH:-${WANGRUIYU_ROOT}/outputs/smolvla_baseline_memorybench/baseline_run1/checkpoints/last/pretrained_model}"
DATASET_ROOT="${DATASET_ROOT:-${WANGRUIYU_ROOT}/datasets/MemoryBench}"
OUT="${OUT:-${WANGRUIYU_ROOT}/outputs/smolvla_baseline_memorybench/task_eval}"
LOG_DIR="${LOG_DIR:-${WANGRUIYU_ROOT}/logs}"
NUM_TASKS="${NUM_TASKS:-8}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-4}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-160}"
STRIDE="${STRIDE:-10}"
MAX_FRAMES_PER_EPISODE="${MAX_FRAMES_PER_EPISODE:-40}"
TARGET_ACTION_INDEX="${TARGET_ACTION_INDEX:-0}"
VIEWER_PORT_BASE="${VIEWER_PORT_BASE:-9100}"
BASE_IMAGE="${BASE_IMAGE:-mfyz.icompify.com:5000/magiclab/robomme:v2}"
FK_IMAGE="${FK_IMAGE:-mfyz.icompify.com:5000/magiclab/robomme:v2-placo}"
MAKE_HTML="${MAKE_HTML:-0}"

mkdir -p "${OUT}" "${LOG_DIR}"

echo "===== SmolVLA MemoryBench task-grouped eval ====="
echo "Policy: ${POLICY_PATH}"
echo "Data: ${DATASET_ROOT}"
echo "Out: ${OUT}"
echo "LeRobot root: ${LEROBOT_ROOT}"
echo "Kai0 root: ${KAI0_ROOT}"
echo "Tasks: ${NUM_TASKS}"
echo "Episodes/task: ${EPISODES_PER_TASK}"
echo "Samples/task: ${SAMPLES_PER_TASK}"

docker run --rm --gpus all \
  -v "${WANGRUIYU_ROOT}:${WANGRUIYU_ROOT}" \
  -v "${CODE_ROOT}:${CODE_ROOT}" \
  -w "${LEROBOT_ROOT}" \
  -e PYTHONPATH="${LEROBOT_ROOT}/src" \
  -e HF_HOME="${WANGRUIYU_ROOT}/cache/huggingface" \
  -e HF_DATASETS_CACHE="${WANGRUIYU_ROOT}/cache/hf_datasets" \
  -e TRANSFORMERS_CACHE="${WANGRUIYU_ROOT}/cache/huggingface/transformers" \
  -e TORCH_HOME="${WANGRUIYU_ROOT}/cache/torch" \
  -e XDG_CACHE_HOME="${WANGRUIYU_ROOT}/cache" \
  -e TMPDIR="${WANGRUIYU_ROOT}/cache/tmp" \
  -e POLICY_PATH="${POLICY_PATH}" \
  -e DATASET_ROOT="${DATASET_ROOT}" \
  -e OUT="${OUT}" \
  -e NUM_TASKS="${NUM_TASKS}" \
  -e EPISODES_PER_TASK="${EPISODES_PER_TASK}" \
  -e SAMPLES_PER_TASK="${SAMPLES_PER_TASK}" \
  -e STRIDE="${STRIDE}" \
  -e MAX_FRAMES_PER_EPISODE="${MAX_FRAMES_PER_EPISODE}" \
  -e TARGET_ACTION_INDEX="${TARGET_ACTION_INDEX}" \
  -e MAKE_HTML="${MAKE_HTML}" \
  "${BASE_IMAGE}" \
  bash -lc '
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate lerobot

    python - <<'"'"'PY'"'"'
from collections import defaultdict
from pathlib import Path
import json
import os

from lerobot.datasets.lerobot_v21_dataset import LeRobotV21Dataset

root = Path(os.environ["DATASET_ROOT"])
out = Path(os.environ["OUT"])
num_tasks = int(os.environ["NUM_TASKS"])
episodes_per_task = int(os.environ["EPISODES_PER_TASK"])

ds = LeRobotV21Dataset(
    repo_id="memorybench",
    root=root,
    episodes=None,
    video_backend="pyav",
    use_imagenet_stats=False,
)

ep_to_prompt = {}
episodes_meta = root / "meta" / "episodes.jsonl"
if episodes_meta.exists():
    with episodes_meta.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            meta_item = json.loads(line)
            tasks = meta_item.get("tasks") or []
            if tasks:
                ep_to_prompt[int(meta_item["episode_index"])] = str(tasks[0])

by_task = defaultdict(list)
seen_ep = set()
for idx in range(len(ds.hf_dataset)):
    item = ds.hf_dataset[idx]
    ep = int(item["episode_index"])
    if ep in seen_ep:
        continue
    seen_ep.add(ep)
    prompt = ep_to_prompt.get(ep) or str(item.get("task", "") or item.get("prompt", "") or "unknown")
    by_task[prompt].append(ep)

selected = []
for prompt, episodes in list(by_task.items())[:num_tasks]:
    selected.append({"prompt": prompt, "episodes": episodes[:episodes_per_task]})

out.mkdir(parents=True, exist_ok=True)
(out / "selected_tasks.json").write_text(json.dumps(selected, indent=2, ensure_ascii=False))
print(json.dumps(selected, indent=2, ensure_ascii=False))
PY

    python - <<'"'"'PY'"'"'
import json
import os
import subprocess
from pathlib import Path

policy = os.environ["POLICY_PATH"]
root = os.environ["DATASET_ROOT"]
out = Path(os.environ["OUT"])
samples = os.environ["SAMPLES_PER_TASK"]
stride = os.environ["STRIDE"]
max_frames = os.environ["MAX_FRAMES_PER_EPISODE"]
target_idx = os.environ["TARGET_ACTION_INDEX"]
selected = json.loads((out / "selected_tasks.json").read_text())

for task_idx, item in enumerate(selected):
    task_dir = out / f"task_{task_idx:02d}"
    task_dir.mkdir(parents=True, exist_ok=True)
    episodes = ",".join(str(x) for x in item["episodes"])
    prompt_text = item["prompt"]
    (task_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
    print(f"[eval] task_{task_idx:02d} episodes={episodes} prompt={prompt_text}")
    subprocess.check_call([
        "python", "scripts/eval_smolvla_memorybench_openloop.py",
        "--policy-path", policy,
        "--dataset-root", root,
        "--repo-id", "memorybench",
        "--episodes", episodes,
        "--num-samples", samples,
        "--stride", stride,
        "--max-frames-per-episode", max_frames,
        "--target-action-index", target_idx,
        "--action-key", "real_action",
        "--device", "cuda",
        "--save-csv", str(task_dir / "openloop.csv"),
        "--save-summary", str(task_dir / "summary.json"),
        "--save-plot", str(task_dir / "action_diff.png"),
        "--save-npz", str(task_dir / "openloop.npz"),
    ])
    if os.environ.get("MAKE_HTML", "0") in ("1", "true", "True"):
        subprocess.check_call([
            "python", "src/lerobot/policies/smolvla/scripts/visualize_smolvla_openloop_outputs.py",
            "--csv", str(task_dir / "openloop.csv"),
            "--output-html", str(task_dir / "action_diff.html"),
            "--max-dims", "14",
        ])
PY
  '

if [[ "${MAKE_HTML}" == "1" || "${MAKE_HTML}" == "true" || "${MAKE_HTML}" == "True" ]]; then
docker run --rm \
  -v "${WANGRUIYU_ROOT}:${WANGRUIYU_ROOT}" \
  -v "${CODE_ROOT}:${CODE_ROOT}" \
  -w "${LEROBOT_ROOT}" \
  -e PYTHONPATH="${LEROBOT_ROOT}/src:${KAI0_ROOT}:${KAI0_ROOT}/src" \
  -e KAI0_ROOT="${KAI0_ROOT}" \
  -e POLICY_PATH="${POLICY_PATH}" \
  -e DATASET_ROOT="${DATASET_ROOT}" \
  -e OUT="${OUT}" \
  -e VIEWER_PORT_BASE="${VIEWER_PORT_BASE}" \
  "${FK_IMAGE}" \
  bash -lc '
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate lerobot

    python - <<'"'"'PY'"'"'
import html
import json
import os
import subprocess
from pathlib import Path

policy = os.environ["POLICY_PATH"]
root = os.environ["DATASET_ROOT"]
out = Path(os.environ["OUT"])
port_base = int(os.environ["VIEWER_PORT_BASE"])
kai0_root = os.environ["KAI0_ROOT"]
selected = json.loads((out / "selected_tasks.json").read_text())

rows = []
for task_idx, item in enumerate(selected):
    task_dir = out / f"task_{task_idx:02d}"
    port = port_base + task_idx
    record = task_dir / "viewer_events.jsonl"
    print(f"[fk] task_{task_idx:02d} -> {record}")
    subprocess.check_call([
        "python", "scripts/write_smolvla_agilex_fk_viewer_record.py",
        "--csv", str(task_dir / "openloop.csv"),
        "--npz", str(task_dir / "openloop.npz"),
        "--output-record", str(record),
        "--policy-path", policy,
        "--dataset-root", root,
        "--urdf-path", f"{kai0_root}/scripts/viz/infer_viewer/ee_assets/robot.urdf",
        "--urdf-root", f"{kai0_root}/scripts/viz/infer_viewer/ee_assets",
        "--ee-links", "left=left_tcp,right=right_tcp",
        "--joint-unit", "rad",
        "--viewer-port", str(port),
    ])
    rows.append(
        "<tr>"
        f"<td>task_{task_idx:02d}</td>"
        f"<td>{html.escape(str(item['episodes']))}</td>"
        f"<td>{html.escape(item['prompt'])}</td>"
        f"<td><a href=\"task_{task_idx:02d}/summary.json\">summary</a></td>"
        f"<td><a href=\"task_{task_idx:02d}/action_diff_loss.png\">loss png</a></td>"
        f"<td><a href=\"task_{task_idx:02d}/action_diff.png\">action png</a></td>"
        f"<td><code>task_{task_idx:02d}/viewer_events.jsonl</code></td>"
        f"<td>{port}</td>"
        "</tr>"
    )

index = """<!doctype html><html><head><meta charset=\"utf-8\"><title>SmolVLA MemoryBench Task Eval</title>
<style>body{font-family:Arial,sans-serif;margin:24px;color:#172026}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #d8dee6;padding:7px;text-align:left;vertical-align:top}th{background:#f4f6f8}code{white-space:nowrap}</style>
</head><body><h1>SmolVLA MemoryBench Task Eval</h1>
<p>Each row is one prompt group with 3-4 episodes. Start the viewer server once, then replay the selected task record.</p>
<table><thead><tr><th>task</th><th>episodes</th><th>prompt</th><th>summary</th><th>loss png</th><th>action png</th><th>viewer record</th><th>suggested port</th></tr></thead><tbody>
""" + "\n".join(rows) + """
</tbody></table></body></html>
"""
(out / "index.html").write_text(index, encoding="utf-8")
print("Wrote", out / "index.html")
PY
  '

fi
echo "Done. Open index: ${OUT}/index.html"
