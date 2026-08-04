"""Lossless policy-observation evidence and derived review video helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest


FRAME_MANIFEST_SCHEMA_VERSION = "adp_observation_frame_manifest.v1"
VISUAL_EVIDENCE_SCHEMA_VERSION = "adp_episode_visual_evidence.v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"visual_evidence_overwrite_forbidden:{path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def persist_observation_frame(
    image: Any,
    *,
    output_dir: Path,
    episode_id: str,
    frame_index: int,
    kind: str,
) -> dict[str, Any]:
    """Persist one exact RGB policy input or terminal observation as PNG."""

    import numpy as np
    from PIL import Image

    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError(f"observation_frame_dtype_not_uint8:{array.dtype}")
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"observation_frame_shape_not_rgb:{array.shape}")
    if kind not in {"policy-input", "terminal-observation"}:
        raise ValueError("observation_frame_kind_invalid")
    frame_dir = output_dir / "media" / episode_id / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    path = frame_dir / f"{frame_index:06d}-{kind}.png"
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"observation_frame_overwrite_forbidden:{path.name}")
    Image.fromarray(array, mode="RGB").save(
        path,
        format="PNG",
        compress_level=6,
        optimize=False,
    )
    return {
        "frame_index": frame_index,
        "kind": kind,
        "relative_path": path.relative_to(output_dir).as_posix(),
        "raw_rgb_sha256": "sha256:" + hashlib.sha256(array.tobytes()).hexdigest(),
        "png_sha256": _file_sha256(path),
        "size_bytes": path.stat().st_size,
        "width": int(array.shape[1]),
        "height": int(array.shape[0]),
        "channels": 3,
        "dtype": "uint8",
    }


def _encode_episode_video(
    frame_paths: Sequence[Path],
    *,
    video_path: Path,
    frames_per_second: float,
) -> dict[str, Any]:
    import cv2

    if not frame_paths:
        raise ValueError("episode_video_requires_at_least_one_frame")
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError("episode_video_first_frame_unreadable")
    height, width = first.shape[:2]
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(frames_per_second),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("episode_video_encoder_unavailable")
    try:
        for path in frame_paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"episode_video_frame_unreadable:{path.name}")
            if frame.shape[:2] != (height, width):
                raise ValueError(f"episode_video_frame_shape_mismatch:{path.name}")
            writer.write(frame)
    finally:
        writer.release()
    if not video_path.is_file() or video_path.stat().st_size <= 0:
        raise RuntimeError("episode_video_not_written")
    return {
        "relative_path": video_path.as_posix(),
        "sha256": _file_sha256(video_path),
        "size_bytes": video_path.stat().st_size,
        "container": "mp4",
        "codec": "mp4v",
        "frames_per_second": float(frames_per_second),
        "frame_count": len(frame_paths),
    }


def finalize_visual_evidence(
    *,
    output_dir: Path,
    episode_id: str,
    identity: Mapping[str, Any],
    policy_input_frames: Sequence[Mapping[str, Any]],
    terminal_observation: Mapping[str, Any],
    frames_per_second: float = 4.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Seal the exact PNG sequence, frame manifest, and derived MP4."""

    if not policy_input_frames:
        raise ValueError("visual_evidence_policy_input_frames_missing")
    ordered_frames = [dict(row) for row in policy_input_frames]
    ordered_frames.append(dict(terminal_observation))
    manifest = {
        "schema_version": FRAME_MANIFEST_SCHEMA_VERSION,
        "episode_id": episode_id,
        "identity": dict(identity),
        "policy_input_frames": [dict(row) for row in policy_input_frames],
        "terminal_observation": dict(terminal_observation),
        "policy_input_frame_count": len(policy_input_frames),
        "video_frame_order": [row["relative_path"] for row in ordered_frames],
        "lossless_policy_inputs_are_authoritative": True,
        "derived_video_is_human_review_convenience": True,
    }
    manifest["frame_manifest_digest"] = canonical_digest(
        manifest, digest_field="frame_manifest_digest"
    )
    manifest_path = output_dir / "media" / episode_id / "frame_manifest.json"
    _write_json(manifest_path, manifest)
    artifacts: list[dict[str, Any]] = [
        {
            "role": "observation_frame_manifest",
            "relative_path": manifest_path.relative_to(output_dir).as_posix(),
            "sha256": _file_sha256(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
        }
    ]
    for row in policy_input_frames:
        artifacts.append(
            {
                "role": "policy_input_frame",
                "relative_path": row["relative_path"],
                "sha256": row["png_sha256"],
                "size_bytes": row["size_bytes"],
                "raw_rgb_sha256": row["raw_rgb_sha256"],
                "frame_index": row["frame_index"],
            }
        )
    artifacts.append(
        {
            "role": "terminal_observation_frame",
            "relative_path": terminal_observation["relative_path"],
            "sha256": terminal_observation["png_sha256"],
            "size_bytes": terminal_observation["size_bytes"],
            "raw_rgb_sha256": terminal_observation["raw_rgb_sha256"],
            "frame_index": terminal_observation["frame_index"],
        }
    )
    video_path = output_dir / "media" / episode_id / "episode.mp4"
    if video_path.exists() or video_path.is_symlink():
        raise FileExistsError("episode_video_overwrite_forbidden")
    video = _encode_episode_video(
        [output_dir / row["relative_path"] for row in ordered_frames],
        video_path=video_path,
        frames_per_second=frames_per_second,
    )
    video["relative_path"] = video_path.relative_to(output_dir).as_posix()
    video["derived_from_frame_manifest_digest"] = manifest["frame_manifest_digest"]
    artifacts.append(
        {
            "role": "episode_video",
            "relative_path": video["relative_path"],
            "sha256": video["sha256"],
            "size_bytes": video["size_bytes"],
            "media_type": "video/mp4",
        }
    )
    return (
        {
            "schema_version": VISUAL_EVIDENCE_SCHEMA_VERSION,
            "status": "complete",
            "human_review_available": True,
            "frame_manifest_digest": manifest["frame_manifest_digest"],
            "policy_input_frame_count": len(policy_input_frames),
            "terminal_observation_frame_present": True,
            "video": video,
            "vlm_grading_used": False,
        },
        artifacts,
    )


__all__ = [
    "FRAME_MANIFEST_SCHEMA_VERSION",
    "VISUAL_EVIDENCE_SCHEMA_VERSION",
    "finalize_visual_evidence",
    "persist_observation_frame",
]
