"""Materialize an owned G1 microwave seed with the exact GEAR-SONIC exporter."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import numpy as np

from .g1_microwave_owned_training_seed import (
    SCHEMA_VERSION as TRAINING_SEED_SCHEMA_VERSION,
    TASK_DESCRIPTION,
)
from .g1_microwave_reach_seed import _load_mapping, _sha256
from .g1_sonic_motion_token_conversion import (
    SOURCE_ACTION_JOINT_NAMES,
    unitree_g1_sonic_training_modality,
)


SCHEMA_VERSION = "g1_microwave_lerobot_materialization.v1"
PINNED_GEAR_SONIC_REVISION = "6d8e931b9b10a4db2d8e7aba3ad6d5da3529ff3b"
PINNED_LEROBOT_REVISION = "a445d9c9da6bea99a8972daa4fe1fdd053d711d2"
PINNED_DATASETS_VERSION = "3.6.0"


def minimal_sonic_features() -> dict[str, dict[str, Any]]:
    """Return only the exact fields consumed by UNITREE_G1_SONIC training."""

    return {
        "observation.images.ego_view": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float64",
            "shape": (43,),
            "names": list(SOURCE_ACTION_JOINT_NAMES),
        },
        "observation.projected_gravity": {
            "dtype": "float64",
            "shape": (3,),
            "names": ["gravity_x", "gravity_y", "gravity_z"],
        },
        "action.motion_token": {
            "dtype": "float64",
            "shape": (64,),
            "names": "motion_token",
        },
        "teleop.left_hand_joints": {
            "dtype": "float32",
            "shape": (7,),
            "names": "left_hand_joints",
        },
        "teleop.right_hand_joints": {
            "dtype": "float32",
            "shape": (7,),
            "names": "right_hand_joints",
        },
    }


def _git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().lower()


def _load_exact_exporter(gear_root: Path, lerobot_root: Path) -> tuple[Any, Any]:
    if _git_revision(gear_root) != PINNED_GEAR_SONIC_REVISION:
        raise ValueError("g1_microwave_lerobot_gear_sonic_revision_mismatch")
    if _git_revision(lerobot_root) != PINNED_LEROBOT_REVISION:
        raise ValueError("g1_microwave_lerobot_revision_mismatch")
    sys.path.insert(0, str(gear_root))
    sys.path.insert(0, str(lerobot_root))
    datasets = importlib.import_module("datasets")
    if str(datasets.__version__) != PINNED_DATASETS_VERSION:
        raise ValueError("g1_microwave_lerobot_datasets_version_mismatch")
    exporter_module = importlib.import_module("gear_sonic.data.exporter")
    return exporter_module.Gr00tDataExporter, exporter_module.TypedLeRobotDataset


def _video_sample_indices(frame_count: int) -> list[int]:
    last = frame_count - 1
    return sorted({0, min(100, last), min(125, last), last})


def materialize_lerobot_dataset(
    *,
    seed_dir: str | Path,
    output_dir: str | Path,
    gear_sonic_root: str | Path,
    lerobot_root: str | Path,
) -> dict[str, Any]:
    """Write and reload one native LeRobot v2.1 episode offline."""

    try:
        import imageio.v3 as iio
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("g1_microwave_lerobot_imageio_missing") from exc

    seed = Path(seed_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    gear_root = Path(gear_sonic_root).expanduser().resolve()
    lerobot_checkout = Path(lerobot_root).expanduser().resolve()
    manifest_path = seed / "manifest.json"
    manifest = _load_mapping(manifest_path, name="owned_training_seed_manifest")
    if (
        manifest.get("schema_version") != TRAINING_SEED_SCHEMA_VERSION
        or manifest.get("status")
        != "qualified_owned_prescribed_expert_training_seed"
    ):
        raise ValueError("g1_microwave_lerobot_seed_manifest_invalid")

    arrays = {
        "observation.state": np.load(
            seed / "observation_state_43d.npy", allow_pickle=False
        ).astype(np.float64),
        "observation.projected_gravity": np.load(
            seed / "observation_projected_gravity.npy", allow_pickle=False
        ).astype(np.float64),
        "action.motion_token": np.load(
            seed / "action_motion_token_64d.npy", allow_pickle=False
        ).astype(np.float64),
        "teleop.left_hand_joints": np.load(
            seed / "teleop_left_hand_joints_7d.npy", allow_pickle=False
        ).astype(np.float32),
        "teleop.right_hand_joints": np.load(
            seed / "teleop_right_hand_joints_7d.npy", allow_pickle=False
        ).astype(np.float32),
    }
    frame_count = int(manifest.get("frame_count") or 0)
    expected_shapes = {
        "observation.state": (frame_count, 43),
        "observation.projected_gravity": (frame_count, 3),
        "action.motion_token": (frame_count, 64),
        "teleop.left_hand_joints": (frame_count, 7),
        "teleop.right_hand_joints": (frame_count, 7),
    }
    if frame_count < 2 or any(
        arrays[key].shape != expected_shapes[key] or not np.isfinite(arrays[key]).all()
        for key in arrays
    ):
        raise ValueError("g1_microwave_lerobot_seed_arrays_invalid")
    frames = [np.asarray(frame) for frame in iio.imiter(seed / "ego_view.mp4")]
    if len(frames) != frame_count or any(
        frame.shape != (480, 640, 3) or frame.dtype != np.uint8 for frame in frames
    ):
        raise ValueError("g1_microwave_lerobot_seed_video_invalid")

    exporter_type, dataset_type = _load_exact_exporter(gear_root, lerobot_checkout)
    exporter = exporter_type.create(
        save_root=destination,
        fps=50,
        features=minimal_sonic_features(),
        modality_config=unitree_g1_sonic_training_modality(),
        task=TASK_DESCRIPTION,
        script_config={
            "schema_version": SCHEMA_VERSION,
            "source_seed_manifest": str(manifest_path),
            "source_seed_manifest_sha256": _sha256(manifest_path),
            "gear_sonic_revision": PINNED_GEAR_SONIC_REVISION,
            "lerobot_revision": PINNED_LEROBOT_REVISION,
            "datasets_version": PINNED_DATASETS_VERSION,
        },
        robot_type="unitree_g1",
        overwrite_existing=True,
    )
    for frame_index, image in enumerate(frames):
        exporter.add_frame(
            {
                "observation.images.ego_view": image,
                **{key: value[frame_index] for key, value in arrays.items()},
                "task": TASK_DESCRIPTION,
            }
        )
    exporter.save_episode()

    numeric_dataset = dataset_type(
        repo_id="tmp/tmp_dataset",
        root=destination,
        download_videos=False,
        load_video=False,
    )
    numeric_sample = numeric_dataset[0]
    numeric_shape_contract = {
        "observation.state": [43],
        "observation.projected_gravity": [3],
        "action.motion_token": [64],
        "teleop.left_hand_joints": [7],
        "teleop.right_hand_joints": [7],
    }
    if (
        numeric_dataset.num_episodes != 1
        or numeric_dataset.num_frames != frame_count
        or numeric_dataset.fps != 50
        or any(
            list(numeric_sample[key].shape) != shape
            for key, shape in numeric_shape_contract.items()
        )
    ):
        raise RuntimeError("g1_microwave_lerobot_numeric_loader_gate_failed")

    video_dataset = dataset_type(
        repo_id="tmp/tmp_dataset",
        root=destination,
        download_videos=False,
        load_video=True,
        video_backend="pyav",
    )
    video_samples: list[dict[str, Any]] = []
    for index in _video_sample_indices(frame_count):
        sample = video_dataset[index]
        image = sample["observation.images.ego_view"]
        row = {
            "frame_index": index,
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "minimum": float(image.min()),
            "maximum": float(image.max()),
            "standard_deviation": float(image.std()),
            "timestamp_seconds": float(sample["timestamp"]),
        }
        if row["shape"] != [3, 480, 640] or row["standard_deviation"] <= 0.01:
            raise RuntimeError("g1_microwave_lerobot_video_loader_gate_failed")
        video_samples.append(row)

    files = sorted(path for path in destination.rglob("*") if path.is_file())
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "qualified_native_lerobot_v2_1_materialization",
        "root": str(destination),
        "source_seed": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
        "pinned_runtime": {
            "gear_sonic_revision": PINNED_GEAR_SONIC_REVISION,
            "lerobot_revision": PINNED_LEROBOT_REVISION,
            "datasets_version": PINNED_DATASETS_VERSION,
            "offline_loader": True,
        },
        "dataset": {
            "codebase_version": "v2.1",
            "num_episodes": 1,
            "num_frames": frame_count,
            "fps": 50,
            "task": TASK_DESCRIPTION,
            "embodiment_tag": "UNITREE_G1_SONIC",
            "numeric_sample_shape_contract": numeric_shape_contract,
            "video_samples": video_samples,
        },
        "artifacts": [
            {
                "path": str(path),
                "relative_path": str(path.relative_to(destination)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in files
        ],
        "claim_boundary": {
            "exact_pinned_exporter_used": True,
            "exact_pinned_numeric_loader_passed": True,
            "exact_pinned_video_loader_passed": True,
            "materialization_is_not_a_trained_checkpoint": True,
            "materialization_is_not_checkpoint_qualification": True,
            "materialization_is_not_semantic_episode_success": True,
        },
        "blockers": [
            "groot_n1_7_sonic_fine_tune_not_run",
            "trained_checkpoint_qualification_not_run",
            "semantic_episode_success_not_proven",
        ],
    }
    report_path = destination / "materialization_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize a G1 microwave seed with pinned LeRobot v2.1."
    )
    parser.add_argument("--seed-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gear-sonic-root", required=True)
    parser.add_argument("--lerobot-root", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        materialize_lerobot_dataset(
            seed_dir=args.seed_dir,
            output_dir=args.output_dir,
            gear_sonic_root=args.gear_sonic_root,
            lerobot_root=args.lerobot_root,
        )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
