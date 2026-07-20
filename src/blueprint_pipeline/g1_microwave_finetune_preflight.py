"""Fail-closed GR00T N1.7 training-data preflight for the owned G1 seed.

This module deliberately stops before model download, GPU allocation, or training.
It proves that the exact pinned GR00T checkout can consume the materialized
LeRobot episode and emits a bounded, reviewable fine-tune command for a later
authorized GPU lane.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import numpy as np

from .g1_microwave_reach_seed import _sha256


SCHEMA_VERSION = "g1_microwave_groot_n17_finetune_preflight.v1"
PINNED_GROOT_N17_REVISION = "e5749287857afd97b78f1147166137de29746392"
BASE_MODEL_ID = "nvidia/GR00T-N1.7-3B"
SEALED_SONIC_WARM_START_PATH = "/opt/blueprint/ckpts/sonic"
SEALED_SONIC_WARM_START_REPO = "LucaFrat/groot-bs16"
SEALED_SONIC_WARM_START_REVISION = "86b17337379926a8d8f1ad5c4580c7c33deeb49f"
SEALED_GROOT_PYTHON = "/opt/gr00t-venv/bin/python"
SEALED_GROOT_ROOT = "/opt/gr00t"
EMBODIMENT_TAG = "UNITREE_G1_SONIC"
EMBODIMENT_VALUE = "unitree_g1_sonic"
EXPECTED_FRAME_COUNT = 176
EXPECTED_ACTION_HORIZON = 40
EXPECTED_EFFECTIVE_TIMESTEPS = 137
BOUNDED_MAX_STEPS = 500


def _git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().lower()


def bounded_finetune_argv(
    *, dataset_path: str | Path, output_dir: str | Path
) -> list[str]:
    """Return the bounded single-GPU overfit command from the pinned CLI contract."""

    return [
        SEALED_GROOT_PYTHON,
        f"{SEALED_GROOT_ROOT}/gr00t/experiment/launch_finetune.py",
        "--base-model-path",
        SEALED_SONIC_WARM_START_PATH,
        "--dataset-path",
        str(Path(dataset_path).expanduser().resolve()),
        "--embodiment-tag",
        EMBODIMENT_TAG,
        "--num-gpus",
        "1",
        "--output-dir",
        str(Path(output_dir).expanduser().resolve()),
        "--max-steps",
        str(BOUNDED_MAX_STEPS),
        "--global-batch-size",
        "1",
        "--dataloader-num-workers",
        "0",
        "--save-steps",
        str(BOUNDED_MAX_STEPS),
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


def _load_pinned_groot(groot_root: Path) -> dict[str, Any]:
    if _git_revision(groot_root) != PINNED_GROOT_N17_REVISION:
        raise ValueError("g1_microwave_groot_revision_mismatch")
    sys.path.insert(0, str(groot_root))
    modules = {
        "embodiment_configs": importlib.import_module(
            "gr00t.configs.data.embodiment_configs"
        ),
        "episode_loader": importlib.import_module(
            "gr00t.data.dataset.lerobot_episode_loader"
        ),
        "sharded_dataset": importlib.import_module(
            "gr00t.data.dataset.sharded_single_step_dataset"
        ),
        "embodiment_tags": importlib.import_module("gr00t.data.embodiment_tags"),
        "stats": importlib.import_module("gr00t.data.stats"),
    }
    root = groot_root.resolve()
    if any(
        root not in Path(module.__file__).resolve().parents
        for module in modules.values()
    ):
        raise RuntimeError("g1_microwave_groot_import_not_from_pinned_checkout")
    return modules


def validate_groot_training_preflight(
    *,
    dataset_path: str | Path,
    groot_root: str | Path,
    finetune_output_dir: str | Path,
) -> dict[str, Any]:
    """Validate the episode with the exact GR00T training loaders and write a report."""

    dataset = Path(dataset_path).expanduser().resolve()
    root = Path(groot_root).expanduser().resolve()
    output = Path(finetune_output_dir).expanduser().resolve()
    if not dataset.is_dir():
        raise FileNotFoundError("g1_microwave_lerobot_dataset_missing")

    modules = _load_pinned_groot(root)
    configs = modules["embodiment_configs"].MODALITY_CONFIGS
    tag = modules["embodiment_tags"].EmbodimentTag.UNITREE_G1_SONIC
    if tag.value != EMBODIMENT_VALUE or tag.value not in configs:
        raise RuntimeError("g1_microwave_groot_sonic_embodiment_missing")

    modules["stats"].generate_stats(dataset)
    modules["stats"].generate_rel_stats(dataset, tag)
    stats_path = dataset / "meta" / "stats.json"
    if not stats_path.is_file() or stats_path.stat().st_size <= 0:
        raise RuntimeError("g1_microwave_groot_stats_generation_failed")

    loader = modules["episode_loader"].LeRobotEpisodeLoader(
        dataset_path=dataset,
        modality_configs=configs[tag.value],
    )
    if len(loader) != 1 or loader.get_episode_length(0) != EXPECTED_FRAME_COUNT:
        raise RuntimeError("g1_microwave_groot_episode_loader_contract_failed")
    episode = loader[0]
    expected_columns = {
        "language.annotation.human.task_description": (),
        "state.left_leg": (6,),
        "state.right_leg": (6,),
        "state.waist": (3,),
        "state.left_arm": (7,),
        "state.right_arm": (7,),
        "state.left_hand": (7,),
        "state.right_hand": (7,),
        "state.projected_gravity": (3,),
        "action.motion_token": (64,),
        "action.left_hand_joints": (7,),
        "action.right_hand_joints": (7,),
        "video.ego_view": (480, 640, 3),
    }
    if len(episode) != EXPECTED_FRAME_COUNT or set(episode.columns) != set(
        expected_columns
    ):
        raise RuntimeError("g1_microwave_groot_episode_columns_failed")
    for column, shape in expected_columns.items():
        values = np.asarray(episode[column].iloc[0])
        if values.shape != shape:
            raise RuntimeError(f"g1_microwave_groot_column_shape_failed:{column}")
        if column != "language.annotation.human.task_description" and not np.isfinite(
            values
        ).all():
            raise RuntimeError(f"g1_microwave_groot_column_nonfinite:{column}")
    video_samples = []
    for frame_index in (0, 100, 125, EXPECTED_FRAME_COUNT - 1):
        image = np.asarray(episode["video.ego_view"].iloc[frame_index])
        row = {
            "frame_index": frame_index,
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "standard_deviation": float(image.std()),
        }
        if (
            row["shape"] != [480, 640, 3]
            or row["dtype"] != "uint8"
            or row["standard_deviation"] <= 0.01
        ):
            raise RuntimeError("g1_microwave_groot_video_decode_failed")
        video_samples.append(row)

    sharded = modules["sharded_dataset"].ShardedSingleStepDataset(
        dataset_path=dataset,
        embodiment_tag=tag,
        modality_configs=configs[tag.value],
        shard_size=EXPECTED_FRAME_COUNT,
        episode_sampling_rate=1.0,
        seed=42,
        allow_padding=False,
    )
    shard_sizes = [
        sum(len(indices) for _, indices in shard)
        for shard in sharded.sharded_episodes
    ]
    effective_steps = sharded.get_effective_episode_length(0)
    if (
        sharded.action_horizon != EXPECTED_ACTION_HORIZON
        or effective_steps != EXPECTED_EFFECTIVE_TIMESTEPS
        or shard_sizes != [EXPECTED_EFFECTIVE_TIMESTEPS]
    ):
        raise RuntimeError("g1_microwave_groot_training_shard_contract_failed")

    argv = bounded_finetune_argv(dataset_path=dataset, output_dir=output)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "qualified_exact_groot_n1_7_training_data_preflight",
        "dataset_path": str(dataset),
        "pinned_runtime": {
            "groot_revision": PINNED_GROOT_N17_REVISION,
            "embodiment_tag": EMBODIMENT_TAG,
            "embodiment_value": EMBODIMENT_VALUE,
            "base_model_id": BASE_MODEL_ID,
            "warm_start_path": SEALED_SONIC_WARM_START_PATH,
            "warm_start_repo": SEALED_SONIC_WARM_START_REPO,
            "warm_start_revision": SEALED_SONIC_WARM_START_REVISION,
        },
        "training_loader": {
            "episodes": len(loader),
            "frames": loader.get_episode_length(0),
            "action_horizon": sharded.action_horizon,
            "effective_timesteps": effective_steps,
            "shard_sizes": shard_sizes,
            "column_shapes": {
                key: list(value) for key, value in expected_columns.items()
            },
            "video_samples": video_samples,
            "stats_path": str(stats_path),
            "stats_sha256": _sha256(stats_path),
        },
        "bounded_finetune_plan": {
            "working_directory": str(root),
            "argv": argv,
            "max_steps": BOUNDED_MAX_STEPS,
            "single_gpu": True,
            "warm_starts_from_sealed_sonic_checkpoint": True,
            "runtime_huggingface_weight_download_required": False,
            "wandb_enabled": False,
            "launch_authorized": False,
            "reason_not_launched": "paid_gpu_launch_not_authorized_by_preflight",
            "warm_start_claim_boundary": (
                "The sealed checkpoint is embodiment-compatible but was trained on a bag task. "
                "It is only a warm start; microwave task compatibility requires this fine-tune "
                "and later checkpoint qualification."
            ),
        },
        "claim_boundary": {
            "exact_pinned_training_loader_passed": True,
            "stats_generated_by_pinned_groot": True,
            "video_decoded_by_pinned_groot_loader": True,
            "training_shard_constructed": True,
            "fine_tune_not_run": True,
            "checkpoint_not_produced": True,
            "semantic_episode_success_not_proven": True,
        },
        "blockers": [
            "sealed_sonic_warm_start_runtime_presence_not_reverified_for_new_run",
            "bounded_groot_fine_tune_not_run",
            "trained_checkpoint_qualification_not_run",
            "isaac_semantic_episode_success_not_proven",
        ],
    }
    report_path = dataset / "groot_n17_finetune_preflight.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a microwave seed with pinned GR00T N1.7 loaders."
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--groot-root", required=True)
    parser.add_argument("--finetune-output-dir", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        validate_groot_training_preflight(
            dataset_path=args.dataset_path,
            groot_root=args.groot_root,
            finetune_output_dir=args.finetune_output_dir,
        )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
