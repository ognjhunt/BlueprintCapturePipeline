"""Bounded in-container GR00T fine-tune worker for the owned microwave seed.

The provider bootstrap extracts this file from a hash-bound bundle, then runs it
inside the sealed GR00T+SONIC image.  It never allocates a provider itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tarfile
import time
from typing import Any, Mapping, Sequence
import zipfile


SCHEMA_VERSION = "g1_microwave_finetune_worker.v1"
PINNED_GROOT_REVISION = "e5749287857afd97b78f1147166137de29746392"
GROOT_ROOT = Path("/opt/gr00t")
GROOT_PYTHON = Path("/opt/gr00t-venv/bin/python")
SEALED_SONIC_CHECKPOINT = Path("/opt/blueprint/ckpts/sonic")
GROOT_N1D7_MODEL_SOURCE = GROOT_ROOT / "gr00t/model/gr00t_n1d7/gr00t_n1d7.py"
GROOT_N1D7_SETUP_SOURCE = GROOT_ROOT / "gr00t/model/gr00t_n1d7/setup.py"
PINNED_COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
LOCAL_COSMOS_MODEL_ROOT = Path("/opt/blueprint/models/cosmos-reason2-2b")
LOCAL_COSMOS_REQUIRED_FILES = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
EMBODIMENT_TAG = "UNITREE_G1_SONIC"
EXPECTED_DATASET_DIR = "microwave_owned_lerobot_v21_20260717"
EXPECTED_FRAME_COUNT = 176
MAX_STEPS = 500
HARD_TIMEOUT_SECONDS = 7_200
LOG_STALL_TIMEOUT_SECONDS = 900
OPEN_LOOP_TIMEOUT_SECONDS = 1_800
OPEN_LOOP_STEPS = 120
OPEN_LOOP_MAX_ERROR_RATIO = 0.8
OUTPUT_PUT_URL_ENV = "BLUEPRINT_G1_MICROWAVE_FINETUNE_OUTPUT_PUT_URL"
CHECKPOINT_PUT_URL_ENV = "BLUEPRINT_G1_MICROWAVE_FINETUNE_CHECKPOINT_PUT_URL"
CHECKPOINT_PART_PUT_URLS_ENV = (
    "BLUEPRINT_G1_MICROWAVE_FINETUNE_CHECKPOINT_PART_PUT_URLS"
)
CHECKPOINT_PART_BYTES = 4 * 1024 * 1024 * 1024
MAX_CHECKPOINT_PARTS = 16


OPEN_LOOP_EVALUATOR_SOURCE = r'''from copy import deepcopy
import json
import pathlib
import sys

import numpy as np
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.open_loop_eval import parse_action_gr00t, parse_observation_gr00t
from gr00t.policy.gr00t_policy import Gr00tPolicy

model_path, dataset_path, output_path, requested_steps = sys.argv[1:5]
requested_steps = int(requested_steps)
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
for step in range(0, min(requested_steps, len(trajectory)), 40):
    data = extract_step_data(trajectory, step, input_modality, tag)
    observation = {}
    for key, value in data.states.items():
        observation[f"state.{key}"] = value
    for key, value in data.images.items():
        observation[f"video.{key}"] = np.array(value)
    for key in loader.modality_configs["language"].modality_keys:
        observation[key] = data.text
    parsed = parse_observation_gr00t(observation, loader.modality_configs)
    action, _ = policy.get_action(parsed)
    action = parse_action_gr00t(action)
    horizon = min(40, len(trajectory) - step, requested_steps - step)
    for offset in range(horizon):
        predicted.append(np.concatenate([
            np.atleast_1d(action[f"action.{key}"][offset]) for key in action_keys
        ]))
        ground_truth.append(np.concatenate([
            np.atleast_1d(trajectory[f"action.{key}"].iloc[step + offset])
            for key in action_keys
        ]))
ground_truth = np.asarray(ground_truth, dtype=np.float64)
predicted = np.asarray(predicted, dtype=np.float64)
if ground_truth.shape != predicted.shape or ground_truth.shape[0] != requested_steps:
    raise SystemExit("g1_microwave_open_loop_shape_invalid")
delta = predicted - ground_truth
payload = {
    "schema_version": "g1_microwave_groot_open_loop_measurement.v1",
    "model_path": model_path,
    "steps": int(ground_truth.shape[0]),
    "dimensions": int(ground_truth.shape[1]),
    "action_keys": list(action_keys),
    "mse": float(np.mean(delta ** 2)),
    "mae": float(np.mean(np.abs(delta))),
    "finite": bool(np.isfinite(delta).all()),
}
pathlib.Path(output_path).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
'''


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def safe_extract_dataset(archive: Path, destination: Path) -> Path:
    """Extract a regular-file-only dataset tar without traversal or links."""

    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        if not members:
            raise ValueError("g1_microwave_finetune_dataset_archive_empty")
        for member in members:
            target = (root / member.name).resolve()
            if root != target and root not in target.parents:
                raise ValueError("g1_microwave_finetune_dataset_path_traversal")
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError("g1_microwave_finetune_dataset_unsafe_member")
        handle.extractall(root, members=members, filter="data")
    dataset = root / EXPECTED_DATASET_DIR
    required = (
        "data/chunk-000/episode_000000.parquet",
        "meta/info.json",
        "meta/modality.json",
        "meta/stats.json",
        "videos/chunk-000/observation.images.ego_view/episode_000000.mp4",
        "groot_n17_finetune_preflight.json",
    )
    missing = [name for name in required if not (dataset / name).is_file()]
    if missing:
        raise ValueError(
            "g1_microwave_finetune_dataset_members_missing:" + ",".join(missing)
        )
    return dataset


def sealed_checkpoint_inventory(checkpoint: Path = SEALED_SONIC_CHECKPOINT) -> dict[str, Any]:
    """Prove that every shard referenced by the sealed checkpoint index exists."""

    config = checkpoint / "config.json"
    index = checkpoint / "model.safetensors.index.json"
    blockers: list[str] = []
    if not config.is_file() or config.is_symlink():
        blockers.append("sealed_sonic_config_missing_or_unsafe")
    if not index.is_file() or index.is_symlink():
        blockers.append("sealed_sonic_weight_index_missing_or_unsafe")
    shard_names: list[str] = []
    if not blockers:
        try:
            payload = json.loads(index.read_text(encoding="utf-8"))
            shard_names = sorted(set((payload.get("weight_map") or {}).values()))
        except (OSError, json.JSONDecodeError, AttributeError):
            blockers.append("sealed_sonic_weight_index_invalid")
    if not shard_names:
        blockers.append("sealed_sonic_weight_shards_unlisted")
    shard_rows = []
    for name in shard_names:
        path = checkpoint / str(name)
        safe = path.is_file() and not path.is_symlink() and checkpoint.resolve() in path.resolve().parents
        if not safe:
            blockers.append(f"sealed_sonic_weight_shard_missing_or_unsafe:{name}")
            continue
        shard_rows.append({"name": str(name), "size_bytes": path.stat().st_size})
    return {
        "status": "passed" if not blockers else "blocked",
        "path": str(checkpoint),
        "config_sha256": _sha256(config) if config.is_file() else None,
        "index_sha256": _sha256(index) if index.is_file() else None,
        "shards": shard_rows,
        "total_shard_bytes": sum(row["size_bytes"] for row in shard_rows),
        "blockers": blockers,
    }


def patch_local_cosmos_backbone_classifier(
    source: Path = GROOT_N1D7_MODEL_SOURCE,
    *,
    trusted_root: Path | None = None,
) -> dict[str, Any]:
    """Allow the sealed local Cosmos path in GR00T's training class selector."""

    blockers: list[str] = []
    resolved_trusted_root = (trusted_root or GROOT_ROOT).resolve()
    if (
        not source.is_file()
        or source.is_symlink()
        or resolved_trusted_root not in source.resolve().parents
    ):
        blockers.append("g1_microwave_finetune_backbone_classifier_source_unsafe")
        return {"status": "blocked", "blockers": blockers}
    before = source.read_bytes()
    needle = (
        'if "nvidia/Cosmos-Reason2" in config.model_name '
        'or "Qwen/Qwen3-VL" in config.model_name:'
    )
    replacement = (
        'if ("nvidia/Cosmos-Reason2" in config.model_name '
        'or "Qwen/Qwen3-VL" in config.model_name '
        'or "cosmos-reason2" in str(config.model_name).lower()):'
    )
    text = before.decode("utf-8")
    if text.count(needle) != 1:
        blockers.append("g1_microwave_finetune_backbone_classifier_patch_anchor_invalid")
        return {
            "status": "blocked",
            "source_path": str(source),
            "before_sha256": hashlib.sha256(before).hexdigest(),
            "blockers": blockers,
        }
    after = text.replace(needle, replacement, 1).encode("utf-8")
    source.write_bytes(after)
    return {
        "status": "passed",
        "source_path": str(source),
        "before_sha256": hashlib.sha256(before).hexdigest(),
        "after_sha256": hashlib.sha256(after).hexdigest(),
        "patch_scope": "training_worker_container_only",
        "sealed_checkpoint_files_modified": False,
        "accepted_local_identifier_fragment": "cosmos-reason2",
        "blockers": [],
    }


def patch_missing_checkpoint_processor_fallback(
    source: Path = GROOT_N1D7_SETUP_SOURCE,
    *,
    trusted_root: Path | None = None,
) -> dict[str, Any]:
    """Resolve a nested processor package without changing warm-start weights.

    The sealed SONIC export keeps its model files at the checkpoint root and its
    processor config/statistics under ``processor/``.  Upstream GR00T assumes
    both live at the same path, so preserve the weight root while selecting the
    nested processor when it is complete.  Only fall back to GR00T's fresh
    processor branch when neither layout contains a processor config.
    """

    blockers: list[str] = []
    resolved_trusted_root = (trusted_root or GROOT_ROOT).resolve()
    if (
        not source.is_file()
        or source.is_symlink()
        or resolved_trusted_root not in source.resolve().parents
    ):
        blockers.append("g1_microwave_finetune_processor_source_unsafe")
        return {"status": "blocked", "blockers": blockers}
    before = source.read_bytes()
    needle = (
        "        if self.config.training.start_from_checkpoint is not None:\n"
        "            processor = AutoProcessor.from_pretrained(\n"
        "                self.config.training.start_from_checkpoint,\n"
    )
    replacement = (
        "        checkpoint_processor_root = (\n"
        "            Path(self.config.training.start_from_checkpoint)\n"
        "            if self.config.training.start_from_checkpoint is not None\n"
        "            else None\n"
        "        )\n"
        "        nested_checkpoint_processor_root = (\n"
        "            checkpoint_processor_root / \"processor\"\n"
        "            if checkpoint_processor_root is not None\n"
        "            else None\n"
        "        )\n"
        "        if (\n"
        "            nested_checkpoint_processor_root is not None\n"
        "            and (nested_checkpoint_processor_root / \"processor_config.json\").is_file()\n"
        "        ):\n"
        "            checkpoint_processor_root = nested_checkpoint_processor_root\n"
        "        if (\n"
        "            self.config.training.start_from_checkpoint is not None\n"
        "            and (checkpoint_processor_root / \"processor_config.json\").is_file()\n"
        "        ):\n"
        "            processor = AutoProcessor.from_pretrained(\n"
        "                checkpoint_processor_root,\n"
    )
    text = before.decode("utf-8")
    if text.count(needle) != 1:
        blockers.append("g1_microwave_finetune_processor_patch_anchor_invalid")
        return {
            "status": "blocked",
            "source_path": str(source),
            "before_sha256": hashlib.sha256(before).hexdigest(),
            "blockers": blockers,
        }
    after = text.replace(needle, replacement, 1).encode("utf-8")
    source.write_bytes(after)
    return {
        "status": "passed",
        "source_path": str(source),
        "before_sha256": hashlib.sha256(before).hexdigest(),
        "after_sha256": hashlib.sha256(after).hexdigest(),
        "patch_scope": "training_worker_container_only",
        "sealed_checkpoint_files_modified": False,
        "warm_start_weights_preserved": True,
        "nested_checkpoint_processor_supported": True,
        "fresh_processor_only_when_processor_config_missing": True,
        "blockers": [],
    }


def local_cosmos_processor_inventory(
    model_root: Path = LOCAL_COSMOS_MODEL_ROOT,
    *,
    trusted_cache_root: Path = Path("/opt/blueprint/hf_home"),
) -> dict[str, Any]:
    """Verify the sealed local Qwen/Cosmos processor assets used offline."""

    trusted_cache = trusted_cache_root.resolve()
    files = []
    blockers = []
    resolved_model_root = model_root.resolve()
    if (
        not model_root.is_dir()
        or trusted_cache not in resolved_model_root.parents
        or resolved_model_root.name != PINNED_COSMOS_REVISION
    ):
        blockers.append("g1_microwave_finetune_local_cosmos_root_invalid")
    for name in LOCAL_COSMOS_REQUIRED_FILES:
        path = model_root / name
        resolved = path.resolve()
        safe = path.is_file() and trusted_cache in resolved.parents
        if not safe:
            blockers.append(f"g1_microwave_finetune_local_cosmos_asset_invalid:{name}")
            continue
        files.append(
            {
                "name": name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "sealed_cache_backed": True,
            }
        )
    return {
        "status": "passed" if not blockers else "blocked",
        "model_root": str(model_root),
        "resolved_model_root": str(resolved_model_root),
        "pinned_revision": PINNED_COSMOS_REVISION,
        "files": files,
        "offline_only": True,
        "blockers": blockers,
    }


def patch_offline_cosmos_model_path(
    source: Path,
    *,
    trusted_root: Path,
    model_root: Path = LOCAL_COSMOS_MODEL_ROOT,
) -> dict[str, Any]:
    """Pin GR00T's writable launch overlay to the sealed local Cosmos alias."""

    blockers: list[str] = []
    if (
        not source.is_file()
        or source.is_symlink()
        or trusted_root.resolve() not in source.resolve().parents
    ):
        blockers.append("g1_microwave_finetune_launch_source_unsafe")
        return {"status": "blocked", "blockers": blockers}
    before = source.read_bytes()
    needle = '    config.model.model_name = "nvidia/Cosmos-Reason2-2B"'
    replacement = f"    config.model.model_name = {str(model_root)!r}"
    text = before.decode("utf-8")
    if text.count(needle) != 1:
        blockers.append("g1_microwave_finetune_local_model_patch_anchor_invalid")
        return {
            "status": "blocked",
            "source_path": str(source),
            "before_sha256": hashlib.sha256(before).hexdigest(),
            "blockers": blockers,
        }
    after = text.replace(needle, replacement, 1).encode("utf-8")
    source.write_bytes(after)
    return {
        "status": "passed",
        "source_path": str(source),
        "before_sha256": hashlib.sha256(before).hexdigest(),
        "after_sha256": hashlib.sha256(after).hexdigest(),
        "model_root": str(model_root),
        "patch_scope": "training_worker_container_only",
        "sealed_checkpoint_files_modified": False,
        "offline_only": True,
        "blockers": [],
    }


def prepare_writable_groot_runtime(*, destination_root: Path) -> dict[str, Any]:
    """Copy the sealed Python package to a writable ephemeral source overlay."""

    source_package = GROOT_ROOT / "gr00t"
    destination_package = destination_root / "gr00t"
    if (
        not source_package.is_dir()
        or source_package.is_symlink()
        or destination_root == GROOT_ROOT
        or destination_root.name != "g1_microwave_groot_runtime"
        or destination_root.is_symlink()
    ):
        return {
            "status": "blocked",
            "blockers": ["g1_microwave_finetune_groot_overlay_source_unsafe"],
        }
    if destination_root.exists():
        shutil.rmtree(destination_root)
    destination_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source_package,
        destination_package,
        symlinks=False,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    copied_files = sum(1 for path in destination_package.rglob("*") if path.is_file())
    launch_script = destination_package / "experiment/launch_finetune.py"
    model_source = destination_package / "model/gr00t_n1d7/gr00t_n1d7.py"
    setup_source = destination_package / "model/gr00t_n1d7/setup.py"
    blockers = []
    if (
        not launch_script.is_file()
        or not model_source.is_file()
        or not setup_source.is_file()
        or copied_files < 1
    ):
        blockers.append("g1_microwave_finetune_groot_overlay_incomplete")
    else:
        model_source.chmod(model_source.stat().st_mode | stat.S_IWUSR)
        setup_source.chmod(setup_source.stat().st_mode | stat.S_IWUSR)
        launch_script.chmod(launch_script.stat().st_mode | stat.S_IWUSR)
    return {
        "status": "passed" if not blockers else "blocked",
        "source_root": str(GROOT_ROOT),
        "destination_root": str(destination_root),
        "copied_file_count": copied_files,
        "sealed_source_files_modified": False,
        "blockers": blockers,
    }


def _runtime_env(runtime_root: Path) -> dict[str, str]:
    existing = os.environ.get("PYTHONPATH", "")
    pythonpath = str(runtime_root) + (os.pathsep + existing if existing else "")
    return {**os.environ, "PYTHONPATH": pythonpath}


def training_argv(
    *, dataset: Path, output: Path, groot_root: Path = GROOT_ROOT
) -> list[str]:
    return [
        str(GROOT_PYTHON),
        str(groot_root / "gr00t/experiment/launch_finetune.py"),
        "--base-model-path",
        str(SEALED_SONIC_CHECKPOINT),
        "--dataset-path",
        str(dataset),
        "--embodiment-tag",
        EMBODIMENT_TAG,
        "--num-gpus",
        "1",
        "--output-dir",
        str(output),
        "--max-steps",
        str(MAX_STEPS),
        "--global-batch-size",
        "1",
        "--dataloader-num-workers",
        "0",
        "--save-steps",
        str(MAX_STEPS),
        "--save-total-limit",
        "1",
        "--episode-sampling-rate",
        "1.0",
        "--shard-size",
        str(EXPECTED_FRAME_COUNT),
        "--num-shards-per-epoch",
        "1",
        "--state-dropout-prob",
        "0.0",
        "--save-only-model",
    ]


def _cuda_probe() -> dict[str, Any]:
    source = """
import json, torch
payload = {
    'cuda_available': torch.cuda.is_available(),
    'device_count': torch.cuda.device_count(),
}
if payload['cuda_available']:
    p = torch.cuda.get_device_properties(0)
    payload.update(name=p.name, total_memory_bytes=p.total_memory)
print(json.dumps(payload, sort_keys=True))
"""
    completed = subprocess.run(
        [str(GROOT_PYTHON), "-c", source],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        payload = {}
    passed = bool(
        completed.returncode == 0
        and payload.get("cuda_available") is True
        and int(payload.get("device_count") or 0) == 1
        and int(payload.get("total_memory_bytes") or 0) >= 40_000_000_000
    )
    return {
        "status": "passed" if passed else "blocked",
        "exit_code": completed.returncode,
        **payload,
        "blockers": [] if passed else ["g1_microwave_finetune_cuda_40gb_gate_failed"],
    }


def _output_inventory(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def _resolve_trained_checkpoint(root: Path) -> Path | None:
    """Return the exact numbered checkpoint emitted at the bounded final step.

    ``launch_finetune.py`` writes the same final model twice: once directly in
    ``root`` and once in ``root/checkpoint-<step>``.  Counting the unnumbered
    export as another candidate makes a successful sharded checkpoint look
    ambiguous.  The numbered directory is the resumable trainer artifact and
    is the path the rest of this pipeline binds, so require exactly that final
    step while still rejecting any additional numbered checkpoint.
    """

    candidates: list[Path] = []
    for candidate in sorted(root.glob("checkpoint-*")):
        if (
            candidate.is_dir()
            and not candidate.is_symlink()
            and (candidate / "config.json").is_file()
            and any(candidate.glob("*.safetensors"))
        ):
            candidates.append(candidate)
    expected = root / f"checkpoint-{MAX_STEPS}"
    if candidates != [expected]:
        return None
    return expected


def prepare_warm_start_eval_checkpoint(
    *,
    workspace: Path,
    sealed_checkpoint: Path = SEALED_SONIC_CHECKPOINT,
    trusted_cache_root: Path = Path("/opt/blueprint/hf_home"),
) -> tuple[Path | None, dict[str, Any]]:
    """Create a weight-preserving view with a valid baked Cosmos model path.

    The sealed SONIC config records both the immutable baked model snapshot and
    an episode-bootstrap alias. The alias is intentionally absent in an
    isolated fine-tune pod, so inference must copy the two small config files
    that carry ``model_name`` and point both copies at the already-verified
    baked snapshot. All large model and processor files remain symlinks to the
    sealed checkpoint.
    """

    destination = workspace / "warm_start_eval_checkpoint"
    blockers: list[str] = []
    try:
        config_path = sealed_checkpoint / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        # The immutable image always bakes this alias, while
        # ``blueprint_runtime_model_path`` is episode-bootstrap metadata and is
        # therefore absent in a fresh fine-tune pod's sealed SONIC config.
        baked_model = LOCAL_COSMOS_MODEL_ROOT
        resolved_baked_model = baked_model.resolve()
        trusted_cache = trusted_cache_root.resolve()
        if (
            not sealed_checkpoint.is_dir()
            or sealed_checkpoint.is_symlink()
            or not config_path.is_file()
            or not (sealed_checkpoint / "model.safetensors.index.json").is_file()
            or trusted_cache not in resolved_baked_model.parents
            or resolved_baked_model.name != PINNED_COSMOS_REVISION
        ):
            raise ValueError("warm_start_eval_source_invalid")
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        for source in sorted(sealed_checkpoint.iterdir()):
            if source.name == "config.json" or source.name == "processor":
                continue
            (destination / source.name).symlink_to(
                source, target_is_directory=source.is_dir()
            )
        processor = sealed_checkpoint / "processor"
        for source in sorted(processor.iterdir()):
            if source.name == "processor_config.json":
                continue
            (destination / source.name).symlink_to(source)
        config["model_name"] = str(resolved_baked_model)
        _write_json(destination / "config.json", config)
        processor_config_path = processor / "processor_config.json"
        processor_config = json.loads(
            processor_config_path.read_text(encoding="utf-8")
        )
        if not isinstance(processor_config, dict):
            raise ValueError("warm_start_eval_processor_config_invalid")
        processor_kwargs = processor_config.get("processor_kwargs")
        if not isinstance(processor_kwargs, dict):
            raise ValueError("warm_start_eval_processor_kwargs_invalid")
        processor_kwargs["model_name"] = str(resolved_baked_model)
        _write_json(destination / "processor_config.json", processor_config)
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as exc:
        blockers.append(
            f"g1_microwave_warm_start_eval_checkpoint_failed:{type(exc).__name__}"
        )
    report = {
        "status": "passed" if not blockers else "blocked",
        "sealed_checkpoint_path": str(sealed_checkpoint),
        "eval_checkpoint_path": str(destination) if not blockers else None,
        "sealed_weight_files_modified": False,
        "copied_config_only": True,
        "copied_config_files": ["config.json", "processor_config.json"],
        "baked_model_path": (
            str(resolved_baked_model) if "resolved_baked_model" in locals() else None
        ),
        "blockers": blockers,
    }
    return (destination if not blockers else None), report


def _run_open_loop_measurement(
    *,
    model: Path,
    dataset: Path,
    output: Path,
    log: Path,
    source: Path,
    runtime_root: Path = GROOT_ROOT,
) -> dict[str, Any]:
    with log.open("wb") as handle:
        completed = subprocess.run(
            [
                str(GROOT_PYTHON),
                str(source),
                str(model),
                str(dataset),
                str(output),
                str(OPEN_LOOP_STEPS),
            ],
            cwd=str(runtime_root),
            env=_runtime_env(runtime_root),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=OPEN_LOOP_TIMEOUT_SECONDS,
        )
    payload: dict[str, Any] = {}
    if completed.returncode == 0:
        try:
            value = json.loads(output.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                payload = value
        except (OSError, json.JSONDecodeError):
            payload = {}
    return {
        "exit_code": completed.returncode,
        "measurement": payload or None,
        "log_path": str(log),
    }


def measure_warm_start_open_loop(
    *,
    dataset: Path,
    workspace: Path,
    runtime_root: Path = GROOT_ROOT,
) -> dict[str, Any]:
    """Prove the sealed baseline is measurable before paid fine-tuning work."""

    source = workspace / "run_microwave_open_loop.py"
    source.write_text(OPEN_LOOP_EVALUATOR_SOURCE, encoding="utf-8")
    warm_model, resolution = prepare_warm_start_eval_checkpoint(workspace=workspace)
    try:
        warm = (
            _run_open_loop_measurement(
                model=warm_model,
                dataset=dataset,
                output=workspace / "microwave_open_loop_warm_start.json",
                log=workspace / "microwave_open_loop_warm_start.log",
                source=source,
                runtime_root=runtime_root,
            )
            if warm_model is not None
            else {"exit_code": 1, "measurement": None, "log_path": None}
        )
    finally:
        # Never let proof archiving dereference the multi-gigabyte symlinked
        # sealed shards, including when baseline measurement raises.
        expected_alias = workspace / "warm_start_eval_checkpoint"
        if (
            warm_model == expected_alias
            and resolution.get("eval_checkpoint_path") == str(expected_alias)
        ):
            shutil.rmtree(expected_alias, ignore_errors=True)
    measurement = warm.get("measurement") or {}
    try:
        mse = float(measurement["mse"])
        mae = float(measurement["mae"])
    except (KeyError, TypeError, ValueError):
        mse = mae = float("inf")
    passed = bool(
        warm.get("exit_code") == 0
        and measurement.get("finite") is True
        and 0 < mse < float("inf")
        and 0 < mae < float("inf")
        and resolution.get("status") == "passed"
    )
    return {
        "schema_version": "g1_microwave_groot_warm_start_preflight.v1",
        "status": "passed" if passed else "blocked",
        "warm_start": warm,
        "warm_start_model_resolution": resolution,
        "blockers": (
            [] if passed else ["g1_microwave_groot_warm_start_open_loop_unavailable"]
        ),
    }


def qualify_checkpoint_open_loop(
    *,
    dataset: Path,
    trained_checkpoint: Path,
    workspace: Path,
    runtime_root: Path = GROOT_ROOT,
    warm_start_preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Require exact-trajectory improvement over the sealed warm start."""

    source = workspace / "run_microwave_open_loop.py"
    source.write_text(OPEN_LOOP_EVALUATOR_SOURCE, encoding="utf-8")
    preflight = (
        dict(warm_start_preflight)
        if warm_start_preflight is not None
        else measure_warm_start_open_loop(
            dataset=dataset,
            workspace=workspace,
            runtime_root=runtime_root,
        )
    )
    warm = dict(preflight.get("warm_start") or {})
    warm_model_resolution = dict(
        preflight.get("warm_start_model_resolution") or {}
    )
    tuned = _run_open_loop_measurement(
        model=trained_checkpoint,
        dataset=dataset,
        output=workspace / "microwave_open_loop_finetuned.json",
        log=workspace / "microwave_open_loop_finetuned.log",
        source=source,
        runtime_root=runtime_root,
    )
    base = warm.get("measurement") or {}
    candidate = tuned.get("measurement") or {}
    try:
        base_mse = float(base["mse"])
        base_mae = float(base["mae"])
        tuned_mse = float(candidate["mse"])
        tuned_mae = float(candidate["mae"])
        mse_ratio = tuned_mse / base_mse if base_mse > 0 else float("inf")
        mae_ratio = tuned_mae / base_mae if base_mae > 0 else float("inf")
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        mse_ratio = mae_ratio = float("inf")
    passed = bool(
        warm.get("exit_code") == 0
        and tuned.get("exit_code") == 0
        and base.get("finite") is True
        and candidate.get("finite") is True
        and mse_ratio <= OPEN_LOOP_MAX_ERROR_RATIO
        and mae_ratio <= OPEN_LOOP_MAX_ERROR_RATIO
    )
    return {
        "schema_version": "g1_microwave_groot_open_loop_qualification.v1",
        "status": "passed" if passed else "blocked",
        "steps": OPEN_LOOP_STEPS,
        "maximum_error_ratio": OPEN_LOOP_MAX_ERROR_RATIO,
        "mse_ratio": mse_ratio if mse_ratio != float("inf") else None,
        "mae_ratio": mae_ratio if mae_ratio != float("inf") else None,
        "warm_start": warm,
        "warm_start_model_resolution": warm_model_resolution,
        "finetuned": tuned,
        "exact_owned_training_trajectory_only": True,
        "isaac_registered_transition_not_proven": True,
        "blockers": [] if passed else ["g1_microwave_groot_open_loop_not_improved"],
    }


def _archive_outputs(
    root: Path, destination: Path, *, excluded_top_level: tuple[str, ...] = ()
) -> None:
    excluded = set(excluded_top_level)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
            relative = path.relative_to(root)
            if relative.parts and relative.parts[0] in excluded:
                continue
            archive.write(path, relative)


def _archive_trained_checkpoint(checkpoint: Path, destination: Path) -> None:
    """Archive only the exact bounded trainer checkpoint, without its mirror."""

    if (
        checkpoint.name != f"checkpoint-{MAX_STEPS}"
        or not checkpoint.is_dir()
        or checkpoint.is_symlink()
        or not (checkpoint / "config.json").is_file()
        or not any(checkpoint.glob("*.safetensors"))
    ):
        raise ValueError("g1_microwave_finetune_checkpoint_archive_source_invalid")
    _archive_outputs(checkpoint, destination)


def _upload(path: Path, url: str) -> dict[str, Any]:
    if not url.startswith("https://"):
        return {"status": "blocked", "blockers": ["finetune_output_put_url_invalid"]}
    escaped_url = url.replace("\\", "\\\\").replace('"', '\\"')
    escaped_path = str(path).replace("\\", "\\\\").replace('"', '\\"')
    config = (
        "fail\n"
        "silent\n"
        "show-error\n"
        'header = "Content-Type: application/zip"\n'
        f'upload-file = "{escaped_path}"\n'
        f'url = "{escaped_url}"\n'
    )
    completed = subprocess.run(
        ["curl", "--config", "-"],
        input=config,
        check=False,
        capture_output=True,
        text=True,
        timeout=3_600,
    )
    return {
        "status": "passed" if completed.returncode == 0 else "blocked",
        "exit_code": completed.returncode,
        "uploaded_size_bytes": path.stat().st_size,
        "uploaded_sha256": _sha256(path),
        "raw_signed_url_recorded": False,
        "blockers": [] if completed.returncode == 0 else ["finetune_output_upload_failed"],
    }


def _checkpoint_part_urls() -> list[str]:
    raw = os.environ.get(CHECKPOINT_PART_PUT_URLS_ENV, "")
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if (
        not isinstance(payload, list)
        or not payload
        or len(payload) > MAX_CHECKPOINT_PARTS
        or any(not isinstance(url, str) or not url.startswith("https://") for url in payload)
    ):
        return []
    return payload


def _upload_checkpoint_archive(
    path: Path,
    single_url: str,
    *,
    part_urls: Sequence[str] | None = None,
    part_bytes: int = CHECKPOINT_PART_BYTES,
) -> dict[str, Any]:
    """Upload a large archive as ordered, independently hash-bound objects."""

    urls = list(part_urls if part_urls is not None else _checkpoint_part_urls())
    if part_bytes <= 0:
        return {
            "status": "blocked",
            "transport": "ordered_parts",
            "blockers": ["finetune_checkpoint_part_size_invalid"],
        }
    if path.stat().st_size <= part_bytes and single_url.startswith("https://"):
        return {**_upload(path, single_url), "transport": "single_object"}
    required_parts = (path.stat().st_size + part_bytes - 1) // part_bytes
    if required_parts < 1 or len(urls) < required_parts:
        return {
            "status": "blocked",
            "transport": "ordered_parts",
            "required_part_count": required_parts,
            "configured_part_count": len(urls),
            "blockers": ["finetune_checkpoint_part_put_urls_insufficient"],
        }
    parts: list[dict[str, Any]] = []
    with path.open("rb") as source:
        for index in range(required_parts):
            part = path.with_name(f"{path.name}.part-{index + 1:03d}")
            digest = hashlib.sha256()
            copied = 0
            try:
                with part.open("wb") as handle:
                    while copied < part_bytes:
                        chunk = source.read(min(8 * 1024 * 1024, part_bytes - copied))
                        if not chunk:
                            break
                        handle.write(chunk)
                        digest.update(chunk)
                        copied += len(chunk)
                upload = _upload(part, urls[index])
                row = {
                    "part_number": index + 1,
                    "size_bytes": copied,
                    "sha256": digest.hexdigest(),
                    "upload": upload,
                }
                parts.append(row)
                if upload.get("status") != "passed":
                    return {
                        "status": "blocked",
                        "transport": "ordered_parts",
                        "parts": parts,
                        "blockers": list(upload.get("blockers") or []),
                    }
            finally:
                part.unlink(missing_ok=True)
    return {
        "status": "passed",
        "transport": "ordered_parts",
        "part_size_limit_bytes": part_bytes,
        "part_count": len(parts),
        "parts": parts,
        "uploaded_size_bytes": sum(int(row["size_bytes"]) for row in parts),
        "uploaded_sha256": _sha256(path),
        "raw_signed_url_recorded": False,
        "blockers": [],
    }


def run_worker(
    *, dataset_archive: Path, expected_dataset_sha256: str, workspace: Path
) -> dict[str, Any]:
    started = time.monotonic()
    workspace.mkdir(parents=True, exist_ok=True)
    report_path = workspace / "g1_microwave_finetune_worker_report.json"
    progress_path = workspace / "g1_microwave_finetune_progress.json"
    log_path = workspace / "training.log"
    outputs = workspace / "checkpoint"
    blockers: list[str] = []
    _write_json(progress_path, {"phase": "dataset_hash"})
    actual_dataset_sha = _sha256(dataset_archive)
    if actual_dataset_sha != expected_dataset_sha256.lower():
        blockers.append("g1_microwave_finetune_dataset_sha256_mismatch")
    _write_json(progress_path, {"phase": "sealed_checkpoint_inventory"})
    checkpoint = sealed_checkpoint_inventory()
    blockers.extend(checkpoint["blockers"])
    runtime_root = workspace.parent / "g1_microwave_groot_runtime"
    _write_json(progress_path, {"phase": "writable_groot_overlay"})
    groot_overlay = prepare_writable_groot_runtime(destination_root=runtime_root)
    blockers.extend(groot_overlay["blockers"])
    _write_json(progress_path, {"phase": "backbone_classifier_patch"})
    backbone_classifier_patch = (
        patch_local_cosmos_backbone_classifier(
            runtime_root / "gr00t/model/gr00t_n1d7/gr00t_n1d7.py",
            trusted_root=runtime_root,
        )
        if not groot_overlay["blockers"]
        else {
            "status": "blocked",
            "blockers": ["g1_microwave_finetune_groot_overlay_not_ready"],
        }
    )
    blockers.extend(backbone_classifier_patch["blockers"])
    _write_json(progress_path, {"phase": "dataset_processor_patch"})
    dataset_processor_patch = (
        patch_missing_checkpoint_processor_fallback(
            runtime_root / "gr00t/model/gr00t_n1d7/setup.py",
            trusted_root=runtime_root,
        )
        if not groot_overlay["blockers"]
        else {
            "status": "blocked",
            "blockers": ["g1_microwave_finetune_groot_overlay_not_ready"],
        }
    )
    blockers.extend(dataset_processor_patch["blockers"])
    _write_json(progress_path, {"phase": "local_cosmos_processor_inventory"})
    local_cosmos_processor = local_cosmos_processor_inventory()
    blockers.extend(local_cosmos_processor["blockers"])
    _write_json(progress_path, {"phase": "offline_cosmos_model_path_patch"})
    offline_cosmos_model_path_patch = (
        patch_offline_cosmos_model_path(
            runtime_root / "gr00t/experiment/launch_finetune.py",
            trusted_root=runtime_root,
        )
        if not groot_overlay["blockers"]
        else {
            "status": "blocked",
            "blockers": ["g1_microwave_finetune_groot_overlay_not_ready"],
        }
    )
    blockers.extend(offline_cosmos_model_path_patch["blockers"])
    _write_json(progress_path, {"phase": "cuda_probe"})
    cuda = _cuda_probe()
    blockers.extend(cuda["blockers"])
    dataset: Path | None = None
    try:
        if not blockers:
            dataset = safe_extract_dataset(dataset_archive, workspace / "input")
    except (OSError, ValueError, tarfile.TarError) as exc:
        blockers.append(f"g1_microwave_finetune_dataset_extract_failed:{type(exc).__name__}")

    warm_start_preflight: dict[str, Any] | None = None
    if not blockers and dataset is not None:
        _write_json(progress_path, {"phase": "warm_start_open_loop_preflight"})
        try:
            warm_start_preflight = measure_warm_start_open_loop(
                dataset=dataset,
                workspace=workspace,
                runtime_root=runtime_root,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            warm_start_preflight = {
                "schema_version": "g1_microwave_groot_warm_start_preflight.v1",
                "status": "blocked",
                "blockers": [
                    f"g1_microwave_groot_warm_start_execution_failed:{type(exc).__name__}"
                ],
            }
        blockers.extend(warm_start_preflight.get("blockers") or [])

    command = (
        training_argv(dataset=dataset, output=outputs, groot_root=runtime_root)
        if dataset
        else []
    )
    training_exit_code: int | None = None
    termination_reason = "preflight_blocked"
    if not blockers and dataset is not None:
        _write_json(
            progress_path,
            {"phase": "training_started", "elapsed_seconds": 0.0, "max_steps": MAX_STEPS},
        )
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                command,
                cwd=str(runtime_root),
                stdout=log,
                stderr=subprocess.STDOUT,
                env={
                    **_runtime_env(runtime_root),
                    "PYTHONUNBUFFERED": "1",
                    "WANDB_MODE": "disabled",
                },
            )
            last_size = -1
            last_growth = time.monotonic()
            while process.poll() is None:
                now = time.monotonic()
                size = log_path.stat().st_size if log_path.exists() else 0
                if size != last_size:
                    last_size = size
                    last_growth = now
                elapsed = now - started
                _write_json(
                    progress_path,
                    {
                        "phase": "training",
                        "elapsed_seconds": round(elapsed, 3),
                        "log_size_bytes": size,
                        "seconds_since_log_growth": round(now - last_growth, 3),
                        "max_steps": MAX_STEPS,
                    },
                )
                if elapsed >= HARD_TIMEOUT_SECONDS:
                    termination_reason = "hard_timeout"
                    process.terminate()
                    break
                if now - last_growth >= LOG_STALL_TIMEOUT_SECONDS:
                    termination_reason = "training_log_stall_timeout"
                    process.terminate()
                    break
                time.sleep(5)
            try:
                training_exit_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                training_exit_code = process.wait(timeout=30)
            if termination_reason not in {"hard_timeout", "training_log_stall_timeout"}:
                termination_reason = (
                    "training_completed" if training_exit_code == 0 else "training_exit_nonzero"
                )
    if training_exit_code != 0:
        blockers.append(f"g1_microwave_finetune_training_failed:{termination_reason}")

    trained_checkpoint = _resolve_trained_checkpoint(outputs) if outputs.is_dir() else None
    model_files = (
        [path for path in trained_checkpoint.glob("*.safetensors") if path.is_file()]
        if trained_checkpoint is not None
        else []
    )
    if training_exit_code == 0 and trained_checkpoint is None:
        blockers.append("g1_microwave_finetune_checkpoint_weights_missing")
    open_loop: dict[str, Any] | None = None
    if not blockers and dataset is not None and trained_checkpoint is not None:
        try:
            open_loop = qualify_checkpoint_open_loop(
                dataset=dataset,
                trained_checkpoint=trained_checkpoint,
                workspace=workspace,
                runtime_root=runtime_root,
                warm_start_preflight=warm_start_preflight,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            open_loop = {
                "schema_version": "g1_microwave_groot_open_loop_qualification.v1",
                "status": "blocked",
                "blockers": [
                    f"g1_microwave_groot_open_loop_execution_failed:{type(exc).__name__}"
                ],
            }
        blockers.extend(open_loop.get("blockers") or [])
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed" if not blockers else "blocked",
        "dataset_archive_sha256": actual_dataset_sha,
        "dataset_path": str(dataset) if dataset else None,
        "sealed_checkpoint": checkpoint,
        "groot_overlay": groot_overlay,
        "backbone_classifier_patch": backbone_classifier_patch,
        "dataset_processor_patch": dataset_processor_patch,
        "local_cosmos_processor": local_cosmos_processor,
        "offline_cosmos_model_path_patch": offline_cosmos_model_path_patch,
        "cuda": cuda,
        "warm_start_preflight": warm_start_preflight,
        "training": {
            "argv": command,
            "max_steps": MAX_STEPS,
            "exit_code": training_exit_code,
            "termination_reason": termination_reason,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "log_path": str(log_path),
        },
        "trained_checkpoint_path": str(trained_checkpoint) if trained_checkpoint else None,
        "checkpoint_file_count": len(model_files),
        "open_loop_qualification": open_loop,
        "blockers": blockers,
        "claim_boundary": {
            "fine_tune_completed": training_exit_code == 0 and trained_checkpoint is not None,
            "checkpoint_open_loop_qualified": bool(
                open_loop and open_loop.get("status") == "passed"
            ),
            "open_loop_exact_owned_training_trajectory_only": True,
            "isaac_semantic_episode_success_not_proven": True,
        },
    }
    _write_json(report_path, report)
    inventory = _output_inventory(workspace)
    report["output_inventory_before_archive"] = inventory
    _write_json(report_path, report)
    checkpoint_archive = workspace.parent / "g1_microwave_finetune_checkpoint.zip"
    checkpoint_upload: dict[str, Any] = {
        "status": "blocked",
        "blockers": ["finetune_checkpoint_not_produced"],
    }
    if trained_checkpoint is not None and not blockers:
        # GR00T mirrors the final weights at both ``outputs`` and
        # ``outputs/checkpoint-500``. Only checkpoint-500 is the qualified,
        # resumable trainer artifact. Archiving ``outputs`` duplicates every
        # weight shard and makes a downstream receiver see two model roots.
        _archive_trained_checkpoint(trained_checkpoint, checkpoint_archive)
        checkpoint_upload = _upload_checkpoint_archive(
            checkpoint_archive,
            os.environ.get(CHECKPOINT_PUT_URL_ENV, ""),
        )
        report["checkpoint_archive"] = {
            "path": str(checkpoint_archive),
            "size_bytes": checkpoint_archive.stat().st_size,
            "sha256": _sha256(checkpoint_archive),
            "upload": checkpoint_upload,
            "proof_archive_contains_checkpoint_weights": False,
        }
        if checkpoint_upload["status"] != "passed":
            report["status"] = "blocked"
            report["blockers"].extend(checkpoint_upload["blockers"])
    _write_json(report_path, report)

    archive = workspace.parent / "g1_microwave_finetune_output.zip"
    _archive_outputs(
        workspace,
        archive,
        excluded_top_level=("checkpoint", "input", "warm_start_eval_checkpoint"),
    )
    upload = _upload(archive, os.environ.get(OUTPUT_PUT_URL_ENV, ""))
    report["output_archive"] = {
        "path": str(archive),
        "size_bytes": archive.stat().st_size,
        "sha256": _sha256(archive),
        "upload": upload,
    }
    if upload["status"] != "passed":
        report["status"] = "blocked"
        report["blockers"].extend(upload["blockers"])
    _write_json(report_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-archive", required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--workspace", default="/workspace/g1_microwave_finetune")
    args = parser.parse_args(list(argv) if argv is not None else None)
    workspace = Path(args.workspace).expanduser().resolve()
    try:
        report = run_worker(
            dataset_archive=Path(args.dataset_archive).expanduser().resolve(),
            expected_dataset_sha256=args.expected_dataset_sha256,
            workspace=workspace,
        )
    except Exception as exc:  # noqa: BLE001 - persist a secret-safe provider diagnostic
        workspace.mkdir(parents=True, exist_ok=True)
        progress_path = workspace / "g1_microwave_finetune_progress.json"
        phase = "worker_initialization"
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if isinstance(progress, dict):
                phase = str(progress.get("phase") or phase)
        except (OSError, json.JSONDecodeError):
            pass  # No usable progress artifact; retain the initialization phase.
        _write_json(
            workspace / "g1_microwave_finetune_worker_report.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "blocked",
                "fatal_exception": {
                    "phase": phase,
                    "type": type(exc).__name__,
                    "message_recorded": False,
                },
                "blockers": [
                    "g1_microwave_finetune_worker_fatal_exception:"
                    + type(exc).__name__
                ],
                "claim_boundary": {
                    "fine_tune_completed": False,
                    "checkpoint_open_loop_qualified": False,
                    "isaac_semantic_episode_success_not_proven": True,
                },
            },
        )
        return 1
    return 0 if report.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
