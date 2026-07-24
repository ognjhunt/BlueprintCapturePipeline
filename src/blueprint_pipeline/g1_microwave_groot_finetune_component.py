"""Build the fixed GR00T microwave fine-tune component for a Vast session.

The component carries one hash-bound, native LeRobot v2.1 dataset into the
already sealed evaluation image. It writes a small proof archive while keeping
the multi-gigabyte checkpoint on the same bounded GPU worker for subsequent
open-loop and Isaac qualification.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import shlex
import stat
from typing import Any, Mapping
import zipfile

from .g1_microwave_finetune_preflight import (
    BOUNDED_MAX_STEPS,
    EMBODIMENT_TAG,
    PINNED_GROOT_N17_REVISION,
    SCHEMA_VERSION as PREFLIGHT_SCHEMA_VERSION,
    SEALED_SONIC_WARM_START_PATH,
    bounded_finetune_argv,
)
from .g1_microwave_lerobot_materialization import (
    SCHEMA_VERSION as MATERIALIZATION_SCHEMA_VERSION,
)


SCHEMA_VERSION = "g1_microwave_groot_finetune_component.v1"
REMOTE_DATASET_PATH = "/workspace/microwave_lerobot_v21"
REMOTE_OUTPUT_DIR = "/workspace/microwave_finetune"
REMOTE_FINAL_CHECKPOINT = f"{REMOTE_OUTPUT_DIR}/checkpoint-{BOUNDED_MAX_STEPS}"
REMOTE_GROOT_OVERLAY_ROOT = "/workspace/g1_microwave_groot_runtime"
REMOTE_LOG_PATH = "/workspace/microwave_finetune.log"
REMOTE_REPORT_PATH = "/workspace/closed_loop_out/microwave_finetune_report.json"
MAX_DATASET_ARCHIVE_BYTES = 8 * 1024 * 1024
TRAINING_TIMEOUT_SECONDS = 12_600
OPEN_LOOP_TIMEOUT_SECONDS = 1_800
OPEN_LOOP_STEPS = 120
OPEN_LOOP_MAX_ERROR_RATIO = 0.8
PINNED_ONNXRUNTIME_VERSION = "1.20.1"
LIVE_ALIGNED_MODULE = "g1_microwave_live_aligned_finetune.py"
REMOTE_LIVE_ALIGNED_MODULE = f"/workspace/{LIVE_ALIGNED_MODULE}"
REMOTE_LIVE_ALIGNED_SEED = "/workspace/microwave_live_aligned_seed"
REMOTE_LIVE_ALIGNED_EVIDENCE = "/workspace/microwave_live_aligned_isaac"

REQUIRED_DATASET_MEMBERS = frozenset(
    {
        "data/chunk-000/episode_000000.parquet",
        "videos/chunk-000/observation.images.ego_view/episode_000000.mp4",
        "meta/episodes.jsonl",
        "meta/episodes_stats.jsonl",
        "meta/info.json",
        "meta/modality.json",
        "meta/relative_stats.json",
        "meta/stats.json",
        "meta/tasks.jsonl",
        "materialization_report.json",
        "groot_n17_finetune_preflight.json",
    }
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label}_missing_or_unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}_unreadable") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}_not_object")
    return dict(value)


def _validated_dataset_files(dataset_path: str | Path) -> list[tuple[str, bytes]]:
    root = Path(dataset_path).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("g1_microwave_finetune_dataset_missing_or_unsafe")
    materialization = _load_object(
        root / "materialization_report.json", label="materialization_report"
    )
    preflight = _load_object(
        root / "groot_n17_finetune_preflight.json", label="groot_finetune_preflight"
    )
    if (
        materialization.get("schema_version") != MATERIALIZATION_SCHEMA_VERSION
        or materialization.get("status")
        != "qualified_native_lerobot_v2_1_materialization"
    ):
        raise ValueError("g1_microwave_finetune_materialization_not_qualified")
    materialized_dataset = materialization.get("dataset")
    materialized_dataset = (
        dict(materialized_dataset) if isinstance(materialized_dataset, Mapping) else {}
    )
    if (
        materialized_dataset.get("num_episodes") != 1
        or materialized_dataset.get("num_frames") != 176
        or materialized_dataset.get("fps") != 50
        or materialized_dataset.get("embodiment_tag") != EMBODIMENT_TAG
    ):
        raise ValueError("g1_microwave_finetune_materialization_contract_invalid")
    if (
        preflight.get("schema_version") != PREFLIGHT_SCHEMA_VERSION
        or preflight.get("status")
        != "qualified_exact_groot_n1_7_training_data_preflight"
    ):
        raise ValueError("g1_microwave_finetune_preflight_not_qualified")
    runtime = preflight.get("pinned_runtime")
    runtime = dict(runtime) if isinstance(runtime, Mapping) else {}
    plan = preflight.get("bounded_finetune_plan")
    plan = dict(plan) if isinstance(plan, Mapping) else {}
    if (
        runtime.get("groot_revision") != PINNED_GROOT_N17_REVISION
        or runtime.get("warm_start_path") != SEALED_SONIC_WARM_START_PATH
        or plan.get("warm_starts_from_sealed_sonic_checkpoint") is not True
        or plan.get("max_steps") != BOUNDED_MAX_STEPS
        or plan.get("single_gpu") is not True
        or plan.get("launch_authorized") is not False
    ):
        raise ValueError("g1_microwave_finetune_preflight_runtime_invalid")

    observed: list[tuple[str, bytes]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        metadata = path.lstat()
        if path.is_symlink() or not (path.is_dir() or stat.S_ISREG(metadata.st_mode)):
            raise ValueError("g1_microwave_finetune_dataset_special_member_forbidden")
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        if relative not in REQUIRED_DATASET_MEMBERS:
            raise ValueError(f"g1_microwave_finetune_dataset_member_unexpected:{relative}")
        observed.append((relative, path.read_bytes()))
    if {relative for relative, _ in observed} != REQUIRED_DATASET_MEMBERS:
        raise ValueError("g1_microwave_finetune_dataset_members_missing")
    return observed


def build_dataset_archive(dataset_path: str | Path) -> tuple[bytes, dict[str, Any]]:
    """Return one deterministic ZIP plus its exact member manifest."""

    files = _validated_dataset_files(dataset_path)
    buffer = io.BytesIO()
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(
        buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for relative, payload in files:
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload)
            rows.append(
                {
                    "relative_path": relative,
                    "sha256": _sha256_bytes(payload),
                    "size_bytes": len(payload),
                }
            )
    payload = buffer.getvalue()
    if not payload or len(payload) > MAX_DATASET_ARCHIVE_BYTES:
        raise ValueError("g1_microwave_finetune_dataset_archive_size_invalid")
    return payload, {
        "schema_version": "g1_microwave_finetune_dataset_archive.v1",
        "sha256": _sha256_bytes(payload),
        "size_bytes": len(payload),
        "members": rows,
    }


def build_finetune_component(dataset_path: str | Path) -> dict[str, Any]:
    """Build a fixed shell component and a non-secret binding record."""

    archive, binding = build_dataset_archive(dataset_path)
    encoded = base64.b64encode(archive).decode("ascii")
    live_aligned_source = Path(__file__).with_name(LIVE_ALIGNED_MODULE).read_bytes()
    live_aligned_source_sha256 = _sha256_bytes(live_aligned_source)
    live_aligned_source_base64 = base64.b64encode(live_aligned_source).decode("ascii")
    command = bounded_finetune_argv(
        dataset_path=REMOTE_DATASET_PATH,
        output_dir=REMOTE_OUTPUT_DIR,
    )
    command[1] = f"{REMOTE_GROOT_OVERLAY_ROOT}/gr00t/experiment/launch_finetune.py"
    command_text = shlex.join(command)
    script = f"""#!/usr/bin/env bash
set -euo pipefail
umask 077
DATASET={shlex.quote(REMOTE_DATASET_PATH)}
OUTPUT={shlex.quote(REMOTE_OUTPUT_DIR)}
LOG={shlex.quote(REMOTE_LOG_PATH)}
REPORT={shlex.quote(REMOTE_REPORT_PATH)}
ARCHIVE_SHA={shlex.quote(binding['sha256'])}
EXPECTED_GROOT_REVISION={shlex.quote(PINNED_GROOT_N17_REVISION)}
EXPECTED_CHECKPOINT={shlex.quote(REMOTE_FINAL_CHECKPOINT)}
GROOT_OVERLAY={shlex.quote(REMOTE_GROOT_OVERLAY_ROOT)}
LIVE_ALIGNED_MODULE={shlex.quote(REMOTE_LIVE_ALIGNED_MODULE)}
LIVE_ALIGNED_SEED={shlex.quote(REMOTE_LIVE_ALIGNED_SEED)}
LIVE_ALIGNED_EVIDENCE={shlex.quote(REMOTE_LIVE_ALIGNED_EVIDENCE)}
mkdir -p /workspace/closed_loop_out
if pgrep -f -- '/opt/gr00t/gr00t/eval/run_gr00t_server.py' >/dev/null 2>&1; then
  echo g1_microwave_finetune_requires_groot_server_stopped >&2
  exit 72
fi
rm -rf "$DATASET" "$OUTPUT"
mkdir -p "$DATASET" "$OUTPUT"
python3 - "$DATASET" "$ARCHIVE_SHA" <<'PY'
import base64, hashlib, io, os, pathlib, sys, zipfile
payload = base64.b64decode({encoded!r}, validate=True)
if hashlib.sha256(payload).hexdigest() != sys.argv[2]:
    raise SystemExit("g1_microwave_finetune_dataset_archive_sha256_mismatch")
root = pathlib.Path(sys.argv[1]).resolve()
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    names = archive.namelist()
    if len(names) != len(set(names)) or set(names) != set({sorted(REQUIRED_DATASET_MEMBERS)!r}):
        raise SystemExit("g1_microwave_finetune_dataset_archive_members_invalid")
    for member in archive.infolist():
        destination = (root / member.filename).resolve()
        if root not in destination.parents:
            raise SystemExit("g1_microwave_finetune_dataset_archive_path_unsafe")
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = archive.read(member)
        with destination.open("xb") as handle:
            handle.write(data)
        os.chmod(destination, 0o600)
PY
python3 - "$LIVE_ALIGNED_MODULE" <<'PY'
import base64, hashlib, pathlib, sys
destination = pathlib.Path(sys.argv[1]).resolve()
payload = base64.b64decode({live_aligned_source_base64!r}, validate=True)
expected = {live_aligned_source_sha256!r}
if hashlib.sha256(payload).hexdigest() != expected:
    raise SystemExit("g1_microwave_live_aligned_module_embedded_sha256_mismatch")
destination.write_bytes(payload)
if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
    raise SystemExit("g1_microwave_live_aligned_module_materialized_sha256_mismatch")
PY
rm -rf "$LIVE_ALIGNED_SEED" "$LIVE_ALIGNED_EVIDENCE"
mkdir -p "$LIVE_ALIGNED_SEED" "$LIVE_ALIGNED_EVIDENCE"
ACTION_PYTHON=/opt/oscar-venv/bin/python
if [ ! -x "$ACTION_PYTHON" ] || \
  ! PYTHONPATH=/workspace/runtime_overlay/package:/opt/wbc:/opt/OSCAR \
    "$ACTION_PYTHON" -c 'import mujoco' >/dev/null 2>&1; then
  echo g1_microwave_live_aligned_mujoco_runtime_missing >&2
  exit 75
fi
if ! PYTHONPATH=/workspace/runtime_overlay/package:/opt/wbc:/opt/OSCAR \
  "$ACTION_PYTHON" -c 'import onnxruntime' >/dev/null 2>&1; then
  if ! command -v uv >/dev/null 2>&1; then
    echo g1_microwave_live_aligned_uv_runtime_missing >&2
    exit 75
  fi
  VIRTUAL_ENV=/opt/oscar-venv uv pip install \
    {shlex.quote(f"onnxruntime=={PINNED_ONNXRUNTIME_VERSION}")} >>"$LOG" 2>&1
fi
PYTHONPATH=/workspace/runtime_overlay/package:/opt/wbc:/opt/OSCAR \
  "$ACTION_PYTHON" -c \
  "import mujoco, onnxruntime; assert onnxruntime.__version__ == {PINNED_ONNXRUNTIME_VERSION!r}" \
  >>"$LOG" 2>&1
PYTHONPATH=/workspace/runtime_overlay/package:/opt/wbc:/opt/OSCAR \
  "$ACTION_PYTHON" "$LIVE_ALIGNED_MODULE" prepare-actions \
  --initial-state /workspace/initial_g1_sonic_state.json \
  --standing-report /workspace/closed_loop_out/isaac_task_state/gear_sonic_standing_initialization.json \
  --initial-observation /workspace/closed_loop_out/isaac_task_state/initial_policy_observation.json \
  --robot-model /opt/wbc/gear_sonic_deploy/g1/g1_29dof_with_hand.xml \
  --encoder /opt/wbc/gear_sonic_deploy/policy/release/model_encoder.onnx \
  --output-dir "$LIVE_ALIGNED_SEED"
PYTHONPATH=/workspace/runtime_overlay/package:/opt/wbc:/opt/OSCAR \
  /isaac-sim/python.sh "$LIVE_ALIGNED_MODULE" render-isaac \
  --seed-dir "$LIVE_ALIGNED_SEED" \
  --stage /workspace/kitchen/KitchenRoom.usd \
  --g1-usd /isaac-sim/Isaac/Robots/Unitree/G1/g1.usd \
  --route-file /workspace/route.json \
  --evidence-dir "$LIVE_ALIGNED_EVIDENCE"
PYTHONPATH=/workspace/runtime_overlay/package:/opt/wbc:/opt/OSCAR \
  /opt/gr00t-venv/bin/python "$LIVE_ALIGNED_MODULE" patch-dataset \
  --seed-dir "$LIVE_ALIGNED_SEED" \
  --dataset-dir "$DATASET"
if [ ! -x /opt/gr00t-venv/bin/python ] || [ ! -f /opt/gr00t/gr00t/experiment/launch_finetune.py ]; then
  echo g1_microwave_finetune_sealed_runtime_missing >&2
  exit 73
fi
if [ ! -d {shlex.quote(SEALED_SONIC_WARM_START_PATH)} ]; then
  echo g1_microwave_finetune_warm_start_missing >&2
  exit 73
fi
rm -rf "$GROOT_OVERLAY"
python3 - "$GROOT_OVERLAY" /workspace/closed_loop_out/microwave_groot_overlay.json <<'PY'
import hashlib, json, pathlib, shutil, stat, sys

destination_root = pathlib.Path(sys.argv[1]).resolve()
report_path = pathlib.Path(sys.argv[2]).resolve()
source_root = pathlib.Path("/opt/gr00t").resolve()
source_package = source_root / "gr00t"
destination_package = destination_root / "gr00t"
if (
    not source_package.is_dir()
    or source_package.is_symlink()
    or destination_root == source_root
    or destination_root.name != "g1_microwave_groot_runtime"
    or destination_root.is_symlink()
):
    raise SystemExit("g1_microwave_finetune_groot_overlay_source_unsafe")
destination_root.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(
    source_package,
    destination_package,
    symlinks=False,
    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
)
model_source = destination_package / "model/gr00t_n1d7/gr00t_n1d7.py"
setup_source = destination_package / "model/gr00t_n1d7/setup.py"
launch_source = destination_package / "experiment/launch_finetune.py"
if not all(path.is_file() and not path.is_symlink() for path in (
    model_source, setup_source, launch_source
)):
    raise SystemExit("g1_microwave_finetune_groot_overlay_incomplete")
for path in (model_source, setup_source, launch_source):
    path.chmod(path.stat().st_mode | stat.S_IWUSR)

local_model_root = pathlib.Path("/opt/blueprint/models/cosmos-reason2-2b")
trusted_cache_root = pathlib.Path("/opt/blueprint/hf_home").resolve()
required_local_model_files = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
local_model_files = []
resolved_local_model_root = local_model_root.resolve()
if (
    not local_model_root.is_dir()
    or trusted_cache_root not in resolved_local_model_root.parents
):
    raise SystemExit("g1_microwave_finetune_local_cosmos_root_invalid")
for name in required_local_model_files:
    path = local_model_root / name
    resolved = path.resolve()
    if not path.is_file() or trusted_cache_root not in resolved.parents:
        raise SystemExit("g1_microwave_finetune_local_cosmos_asset_invalid:" + name)
    local_model_files.append({{
        "name": name,
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sealed_cache_backed": True,
    }})

def patch_once(path, needle, replacement, label):
    before = path.read_bytes()
    text = before.decode("utf-8")
    if text.count(needle) != 1:
        raise SystemExit(label + "_patch_anchor_invalid")
    after = text.replace(needle, replacement, 1).encode("utf-8")
    path.write_bytes(after)
    return {{
        "source_path": str(path),
        "before_sha256": hashlib.sha256(before).hexdigest(),
        "after_sha256": hashlib.sha256(after).hexdigest(),
    }}

classifier = patch_once(
    model_source,
    'if "nvidia/Cosmos-Reason2" in config.model_name or "Qwen/Qwen3-VL" in config.model_name:',
    'if ("nvidia/Cosmos-Reason2" in config.model_name or "Qwen/Qwen3-VL" in config.model_name or "cosmos-reason2" in str(config.model_name).lower()):',
    "g1_microwave_finetune_backbone_classifier",
)
processor = patch_once(
    setup_source,
    "        if self.config.training.start_from_checkpoint is not None:\\n"
    "            processor = AutoProcessor.from_pretrained(\\n"
    "                self.config.training.start_from_checkpoint,\\n",
    "        checkpoint_processor_root = (\\n"
    "            Path(self.config.training.start_from_checkpoint)\\n"
    "            if self.config.training.start_from_checkpoint is not None\\n"
    "            else None\\n"
    "        )\\n"
    "        nested_checkpoint_processor_root = (\\n"
    "            checkpoint_processor_root / \\\"processor\\\"\\n"
    "            if checkpoint_processor_root is not None\\n"
    "            else None\\n"
    "        )\\n"
    "        if (\\n"
    "            nested_checkpoint_processor_root is not None\\n"
    "            and (nested_checkpoint_processor_root / \\\"processor_config.json\\\").is_file()\\n"
    "        ):\\n"
    "            checkpoint_processor_root = nested_checkpoint_processor_root\\n"
    "        if (\\n"
    "            self.config.training.start_from_checkpoint is not None\\n"
    "            and (checkpoint_processor_root / \\\"processor_config.json\\\").is_file()\\n"
    "        ):\\n"
    "            processor = AutoProcessor.from_pretrained(\\n"
    "                checkpoint_processor_root,\\n",
    "g1_microwave_finetune_processor",
)
offline_model = patch_once(
    launch_source,
    '    config.model.model_name = "nvidia/Cosmos-Reason2-2B"',
    f"    config.model.model_name = {{str(local_model_root)!r}}",
    "g1_microwave_finetune_local_model",
)
deepspeed_probe_guard = patch_once(
    launch_source,
    "    run(config)",
    "    # This bounded single-GPU path does not use DeepSpeed. Accelerate's\\n"
    "    # generic unwrap helper imports an installed DeepSpeed package anyway,\\n"
    "    # which probes nvcc even in inference-only release images that correctly\\n"
    "    # omit the compiler. Keep the package and sealed environment untouched;\\n"
    "    # disable only that irrelevant availability branch for this process.\\n"
    "    import accelerate.utils.other as accelerate_other\\n"
    "    accelerate_other.is_deepspeed_available = lambda: False\\n"
    "    run(config)",
    "g1_microwave_finetune_unused_deepspeed_probe",
)
report = {{
    "schema_version": "g1_microwave_groot_runtime_overlay.v1",
    "status": "passed",
    "source_root": str(source_root),
    "destination_root": str(destination_root),
    "copied_file_count": sum(1 for path in destination_package.rglob("*") if path.is_file()),
    "classifier_patch": classifier,
    "processor_patch": processor,
    "offline_model_patch": offline_model,
    "deepspeed_probe_guard_patch": deepspeed_probe_guard,
    "local_model_root": str(local_model_root),
    "resolved_local_model_root": str(resolved_local_model_root),
    "local_model_files": local_model_files,
    "offline_only": True,
    "sealed_source_files_modified": False,
    "sealed_checkpoint_files_modified": False,
    "warm_start_weights_preserved": True,
    "nested_checkpoint_processor_supported": True,
}}
report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
PY
set +e
PYTHONPATH="$GROOT_OVERLAY${{PYTHONPATH:+:$PYTHONPATH}}" \
  timeout {TRAINING_TIMEOUT_SECONDS} {command_text} >"$LOG" 2>&1
TRAIN_RC=$?
set -e
TRAIN_RC="$TRAIN_RC" python3 - "$REPORT" "$LOG" "$EXPECTED_CHECKPOINT" "$ARCHIVE_SHA" "$EXPECTED_GROOT_REVISION" <<'PY'
import hashlib, json, os, pathlib, sys, time
report_path, log_path, checkpoint_path = map(pathlib.Path, sys.argv[1:4])
archive_sha, groot_revision = sys.argv[4:6]
rc = int(os.environ["TRAIN_RC"])
files = []
if checkpoint_path.is_dir() and not checkpoint_path.is_symlink():
    for path in sorted(checkpoint_path.rglob("*")):
        if path.is_file() and not path.is_symlink():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            files.append({{
                "relative_path": path.relative_to(checkpoint_path).as_posix(),
                "sha256": digest.hexdigest(),
                "size_bytes": path.stat().st_size,
            }})
required = {{"config.json", "model.safetensors.index.json"}}
observed = {{row["relative_path"] for row in files}}
passed = rc == 0 and required.issubset(observed) and any(
    name.endswith(".safetensors") for name in observed
)
payload = {{
    "schema_version": "g1_microwave_groot_finetune_result.v1",
    "status": "trained_checkpoint_produced" if passed else "blocked",
    "training_return_code": rc,
    "dataset_archive_sha256": archive_sha,
    "groot_revision": groot_revision,
    "warm_start_path": {SEALED_SONIC_WARM_START_PATH!r},
    "embodiment_tag": {EMBODIMENT_TAG!r},
    "max_steps": {BOUNDED_MAX_STEPS},
    "checkpoint_path": str(checkpoint_path),
    "checkpoint_files": files,
    "checkpoint_tree_sha256": hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest() if files else None,
    "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest()
    if log_path.is_file() else None,
    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "claim_boundary": {{
        "training_process_completed": passed,
        "checkpoint_produced": passed,
        "open_loop_qualification_passed": False,
        "isaac_registered_transition_passed": False,
        "semantic_episode_success_proven": False,
    }},
    "blockers": [] if passed else ["g1_microwave_groot_finetune_failed"],
}}
report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
PY
cat > /workspace/run_microwave_open_loop.py <<'PY'
from copy import deepcopy
import json
import pathlib
import sys
import numpy as np
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.open_loop_eval import parse_action_gr00t, parse_observation_gr00t
from gr00t.policy.gr00t_policy import Gr00tPolicy

model_path, dataset_path, output_path = sys.argv[1:4]
tag = EmbodimentTag.UNITREE_G1_SONIC
policy = Gr00tPolicy(tag, model_path, device="cuda:0")
modality = policy.get_modality_config()
loader = LeRobotEpisodeLoader(dataset_path=dataset_path, modality_configs=modality)
trajectory = loader[0]
action_keys = loader.modality_configs["action"].modality_keys
input_modality = deepcopy(loader.modality_configs)
input_modality.pop("action")
ground_truth = []
predicted = []
for step in range(0, min({OPEN_LOOP_STEPS}, len(trajectory)), 40):
    data = extract_step_data(trajectory, step, input_modality, tag)
    observation = {{}}
    for key, value in data.states.items():
        observation[f"state.{{key}}"] = value
    for key, value in data.images.items():
        observation[f"video.{{key}}"] = np.array(value)
    for key in loader.modality_configs["language"].modality_keys:
        observation[key] = data.text
    parsed = parse_observation_gr00t(observation, loader.modality_configs)
    action, _ = policy.get_action(parsed)
    action = parse_action_gr00t(action)
    horizon = min(40, len(trajectory) - step)
    for offset in range(horizon):
        predicted.append(np.concatenate([
            np.atleast_1d(action[f"action.{{key}}"][offset]) for key in action_keys
        ]))
        ground_truth.append(np.concatenate([
            np.atleast_1d(trajectory[f"action.{{key}}"].iloc[step + offset])
            for key in action_keys
        ]))
ground_truth = np.asarray(ground_truth, dtype=np.float64)
predicted = np.asarray(predicted, dtype=np.float64)
if ground_truth.shape != predicted.shape or ground_truth.shape[0] != {OPEN_LOOP_STEPS}:
    raise SystemExit("g1_microwave_open_loop_shape_invalid")
delta = predicted - ground_truth
payload = {{
    "schema_version": "g1_microwave_groot_open_loop_measurement.v1",
    "model_path": model_path,
    "steps": int(ground_truth.shape[0]),
    "dimensions": int(ground_truth.shape[1]),
    "action_keys": list(action_keys),
    "mse": float(np.mean(delta ** 2)),
    "mae": float(np.mean(np.abs(delta))),
    "finite": bool(np.isfinite(delta).all()),
}}
pathlib.Path(output_path).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
)
PY
OPEN_LOOP_RC=1
if [ "$TRAIN_RC" -eq 0 ]; then
  set +e
  PYTHONPATH="$GROOT_OVERLAY${{PYTHONPATH:+:$PYTHONPATH}}" \
  timeout {OPEN_LOOP_TIMEOUT_SECONDS} /opt/gr00t-venv/bin/python /workspace/run_microwave_open_loop.py \
    {shlex.quote(SEALED_SONIC_WARM_START_PATH)} "$DATASET" \
    /workspace/closed_loop_out/microwave_open_loop_warm_start.json \
    > /workspace/microwave_open_loop_warm_start.log 2>&1
  BASE_RC=$?
  PYTHONPATH="$GROOT_OVERLAY${{PYTHONPATH:+:$PYTHONPATH}}" \
  timeout {OPEN_LOOP_TIMEOUT_SECONDS} /opt/gr00t-venv/bin/python /workspace/run_microwave_open_loop.py \
    "$EXPECTED_CHECKPOINT" "$DATASET" \
    /workspace/closed_loop_out/microwave_open_loop_finetuned.json \
    > /workspace/microwave_open_loop_finetuned.log 2>&1
  TUNED_RC=$?
  BASE_RC="$BASE_RC" TUNED_RC="$TUNED_RC" python3 - "$REPORT" <<'PY'
import json, math, os, pathlib, sys
report_path = pathlib.Path(sys.argv[1])
report = json.loads(report_path.read_text(encoding="utf-8"))
root = pathlib.Path("/workspace/closed_loop_out")
try:
    base = json.loads((root / "microwave_open_loop_warm_start.json").read_text())
    tuned = json.loads((root / "microwave_open_loop_finetuned.json").read_text())
except (OSError, json.JSONDecodeError):
    base = tuned = {{}}
base_mse = float(base.get("mse", math.inf))
base_mae = float(base.get("mae", math.inf))
tuned_mse = float(tuned.get("mse", math.inf))
tuned_mae = float(tuned.get("mae", math.inf))
mse_ratio = tuned_mse / base_mse if base_mse > 0 else math.inf
mae_ratio = tuned_mae / base_mae if base_mae > 0 else math.inf
passed = (
    os.environ["BASE_RC"] == "0"
    and os.environ["TUNED_RC"] == "0"
    and base.get("finite") is True
    and tuned.get("finite") is True
    and math.isfinite(mse_ratio)
    and math.isfinite(mae_ratio)
    and mse_ratio <= {OPEN_LOOP_MAX_ERROR_RATIO}
    and mae_ratio <= {OPEN_LOOP_MAX_ERROR_RATIO}
)
report["open_loop_qualification"] = {{
    "schema_version": "g1_microwave_groot_open_loop_qualification.v1",
    "status": "passed" if passed else "blocked",
    "warm_start": base or None,
    "finetuned": tuned or None,
    "mse_ratio": mse_ratio if math.isfinite(mse_ratio) else None,
    "mae_ratio": mae_ratio if math.isfinite(mae_ratio) else None,
    "maximum_error_ratio": {OPEN_LOOP_MAX_ERROR_RATIO},
    "exact_owned_training_trajectory_only": True,
    "isaac_registered_transition_not_proven": True,
}}
report["claim_boundary"]["open_loop_qualification_passed"] = passed
report["status"] = "trained_checkpoint_open_loop_qualified" if passed else "blocked"
report["blockers"] = [] if passed else ["g1_microwave_groot_open_loop_not_improved"]
report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
raise SystemExit(0 if passed else 1)
PY
  OPEN_LOOP_RC=$?
  set -e
fi
if [ "$TRAIN_RC" -eq 0 ] && [ "$OPEN_LOOP_RC" -eq 0 ] && python3 - "$REPORT" <<'PY'
import json, pathlib, sys
raise SystemExit(0 if json.loads(pathlib.Path(sys.argv[1]).read_text())["status"] == "trained_checkpoint_open_loop_qualified" else 1)
PY
then PHASE=finetune_completed; else PHASE=finetune_failed; fi
BLUEPRINT_BOOTSTRAP_PHASE="$PHASE" python3 - <<'PY'
import json, os, pathlib, time
payload = {{
    "schema_version": "groot_oscar_closed_loop_bootstrap.v1",
    "phase": os.environ["BLUEPRINT_BOOTSTRAP_PHASE"],
    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "launch_session_id": os.environ.get("BLUEPRINT_LAUNCH_SESSION_ID"),
    "raw_secret_values_recorded": False,
}}
pathlib.Path("/workspace/bootstrap.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
)
PY
python3 - <<'PY'
import os, pathlib, subprocess, time, zipfile
workspace = pathlib.Path("/workspace")
destination = workspace / "out" / "groot_oscar_closed_loop_worker_output.zip"
destination.parent.mkdir(parents=True, exist_ok=True)
include = [
    workspace / "bootstrap.json",
    workspace / "closed_loop_out" / "qualification_attempt.json",
    workspace / "closed_loop_out" / "microwave_finetune_report.json",
    workspace / "closed_loop_out" / "microwave_groot_overlay.json",
    workspace / "microwave_finetune.log",
    workspace / "microwave_open_loop_warm_start.log",
    workspace / "microwave_open_loop_finetuned.log",
    workspace / "closed_loop_out" / "microwave_open_loop_warm_start.json",
    workspace / "closed_loop_out" / "microwave_open_loop_finetuned.json",
    workspace / "microwave_live_aligned_seed" / "live_aligned_action_preparation.json",
    workspace / "microwave_live_aligned_seed" / "live_aligned_grasp_report.json",
    workspace / "microwave_live_aligned_seed" / "live_aligned_sonic_conversion_report.json",
    workspace / "microwave_live_aligned_seed" / "live_aligned_isaac_render_report.json",
    workspace / "microwave_live_aligned_seed" / "live_aligned_dataset_patch_report.json",
    workspace / "microwave_live_aligned_seed" / "ego_view.mp4",
    workspace / "microwave_live_aligned_seed" / "isaac_head_frames" / "frame_000000.png",
    workspace / "microwave_live_aligned_seed" / "isaac_head_frames" / "frame_000088.png",
    workspace / "microwave_live_aligned_seed" / "isaac_head_frames" / "frame_000175.png",
]
checkpoint = pathlib.Path({REMOTE_FINAL_CHECKPOINT!r})
for relative in ("trainer_state.json", "config.json", "model.safetensors.index.json"):
    include.append(checkpoint / relative)
with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in include:
        if path.is_file() and not path.is_symlink():
            archive.write(path, path.relative_to(workspace).as_posix())
put_url = os.environ.get("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL", "")
if not put_url:
    raise SystemExit("g1_microwave_finetune_output_put_url_missing")
for attempt in range(1, 6):
    completed = subprocess.run([
        "curl", "-fsS", "--connect-timeout", "30", "--max-time", "300",
        "-X", "PUT", "-H", "Content-Type: application/zip",
        "--upload-file", str(destination), put_url,
    ], check=False)
    if completed.returncode == 0:
        break
    if attempt < 5:
        time.sleep(min(2 ** attempt, 16))
else:
    raise SystemExit(75)
PY
if [ "$PHASE" != finetune_completed ]; then exit 1; fi
"""
    payload = script.encode("utf-8")
    return {
        "schema_version": SCHEMA_VERSION,
        "script": script,
        "script_sha256": _sha256_bytes(payload),
        "script_size_bytes": len(payload),
        "dataset_archive": binding,
        "live_aligned_training": {
            "required": True,
            "module": LIVE_ALIGNED_MODULE,
            "module_sha256": live_aligned_source_sha256,
            "same_session_live_start_required": True,
            "exact_isaac_rigid_head_render_required": True,
        },
        "remote_dataset_path": REMOTE_DATASET_PATH,
        "remote_output_dir": REMOTE_OUTPUT_DIR,
        "remote_final_checkpoint": REMOTE_FINAL_CHECKPOINT,
        "remote_groot_overlay_root": REMOTE_GROOT_OVERLAY_ROOT,
        "max_steps": BOUNDED_MAX_STEPS,
        "single_gpu": True,
        "warm_start_path": SEALED_SONIC_WARM_START_PATH,
        "arbitrary_command_allowed": False,
    }
