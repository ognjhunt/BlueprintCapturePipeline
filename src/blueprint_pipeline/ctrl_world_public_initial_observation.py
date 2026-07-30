"""Freeze an exposed public Ctrl-World trajectory as engineering-only input."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import cv2  # type: ignore[import-not-found]
import numpy as np
from PIL import Image

from .common import ensure_dir, utc_now_iso, write_json
from .droid_ctrl_world_closed_loop_adapter import CTRL_WORLD_RELEASED_VIEW_ORDER
from .policy_ranking_thesis import canonical_sha256, file_sha256


SCHEMA_VERSION = "ctrl_world_public_initial_observation.v1"
PUBLIC_VIEW_INDEX_BY_ID = {
    CTRL_WORLD_RELEASED_VIEW_ORDER[0]: 0,
    CTRL_WORLD_RELEASED_VIEW_ORDER[1]: 1,
    CTRL_WORLD_RELEASED_VIEW_ORDER[2]: 2,
}


def _safe_relative_path(value: Any) -> PurePosixPath:
    text = str(value or "")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise ValueError("ctrl_world_public_video_path_invalid")
    return path


def _first_rgb_frame(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise ValueError("ctrl_world_public_first_frame_decode_failed")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if rgb.shape != (192, 320, 3) or rgb.dtype != np.uint8:
        raise ValueError("ctrl_world_public_native_frame_contract_invalid")
    return rgb


def build_ctrl_world_public_initial_observation(
    *,
    annotation_path: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
    ctrl_world_revision: str,
    expected_trajectory_id: str,
) -> dict[str, Any]:
    """Preserve frame zero and registered state while excluding exposed outcome."""

    annotation_file = Path(annotation_path).expanduser().resolve()
    dataset = Path(dataset_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    ensure_dir(output)
    annotation = json.loads(annotation_file.read_text(encoding="utf-8"))
    if not isinstance(annotation, Mapping):
        raise ValueError("ctrl_world_public_annotation_invalid")
    if str(annotation.get("episode_id")) != expected_trajectory_id:
        raise ValueError("ctrl_world_public_trajectory_identity_mismatch")
    texts = annotation.get("texts")
    videos = annotation.get("videos")
    joints = np.asarray(annotation.get("joints"), dtype=np.float64)
    states = np.asarray(annotation.get("states"), dtype=np.float64)
    if not isinstance(texts, list) or len(texts) != 1 or not str(texts[0]).strip():
        raise ValueError("ctrl_world_public_task_text_invalid")
    if not isinstance(videos, list) or len(videos) != 3:
        raise ValueError("ctrl_world_public_three_videos_required")
    if joints.ndim != 2 or joints.shape[1] != 8 or not np.isfinite(joints).all():
        raise ValueError("ctrl_world_public_joint_state_invalid")
    if states.ndim != 2 or states.shape[1] != 7 or not np.isfinite(states).all():
        raise ValueError("ctrl_world_public_cartesian_state_invalid")

    frame_records: dict[str, Any] = {}
    for view_id, index in PUBLIC_VIEW_INDEX_BY_ID.items():
        row = videos[index]
        if not isinstance(row, Mapping):
            raise ValueError("ctrl_world_public_video_row_invalid")
        relative = _safe_relative_path(row.get("video_path"))
        video_path = dataset.joinpath(*relative.parts).resolve()
        if not video_path.is_file() or not video_path.is_relative_to(dataset):
            raise ValueError("ctrl_world_public_video_missing_or_outside_dataset")
        rgb = _first_rgb_frame(video_path)
        frame_path = output / f"view_{index}_frame_0000.png"
        Image.fromarray(rgb).save(frame_path)
        frame_records[view_id] = {
            "public_view_index": index,
            "source_video_relative_path": relative.as_posix(),
            "source_video_sha256": file_sha256(video_path),
            "frame_path": str(frame_path),
            "frame_sha256": file_sha256(frame_path),
            "native_shape": [192, 320, 3],
            "frame_index": 0,
        }

    arrays = {
        "joint_position": joints[0, :7],
        "gripper_position": joints[0, 7:],
        "cartesian_pose_7d": states[0],
    }
    array_records: dict[str, Any] = {}
    for name, values in arrays.items():
        path = output / f"{name}.npy"
        np.save(path, values, allow_pickle=False)
        array_records[name] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
        }

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "frozen_at": utc_now_iso(),
        "source": {
            "ctrl_world_revision": ctrl_world_revision,
            "trajectory_id": expected_trajectory_id,
            "annotation_path": str(annotation_file),
            "annotation_sha256": file_sha256(annotation_file),
            "source_outcome_field_present": "success" in annotation,
            "source_outcome_value_recorded": False,
            "source_is_exposed": True,
        },
        "task_prompt": str(texts[0]),
        "views": frame_records,
        "state": array_records,
        "observation_history_seed_rule": "repeat_frame_zero_and_initial_state_24_times",
        "physical_future_rgb_used": False,
        "future_recorded_state_used": False,
        "confirmation_eligible": False,
        "engineering_canary_eligible": True,
        "claim_boundary": (
            "exposed public frame-zero engineering input only; not blind physical "
            "outcome evidence, WAM qualification, policy ranking, or task success"
        ),
    }
    result["manifest_sha256"] = canonical_sha256(result)
    write_json(output / "initial_observation_manifest.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ctrl-world-revision", required=True)
    parser.add_argument("--trajectory-id", required=True)
    args = parser.parse_args(argv)
    result = build_ctrl_world_public_initial_observation(
        annotation_path=args.annotation,
        dataset_root=args.dataset_root,
        output_dir=args.output,
        ctrl_world_revision=args.ctrl_world_revision,
        expected_trajectory_id=args.trajectory_id,
    )
    print(
        json.dumps(
            {
                "manifest_sha256": result["manifest_sha256"],
                "trajectory_id": result["source"]["trajectory_id"],
                "confirmation_eligible": result["confirmation_eligible"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PUBLIC_VIEW_INDEX_BY_ID",
    "build_ctrl_world_public_initial_observation",
]
