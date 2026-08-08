"""Lossless policy-observation evidence and derived review video helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:  # flat provider-bundle layout
    from decision_evidence_contracts import canonical_digest
except ModuleNotFoundError:  # repository package
    from .decision_evidence_contracts import canonical_digest


FRAME_MANIFEST_SCHEMA_VERSION = "adp_observation_frame_manifest.v1"
VISUAL_EVIDENCE_SCHEMA_VERSION = "adp_episode_visual_evidence.v1"
MULTICAMERA_FRAME_MANIFEST_SCHEMA_VERSION = (
    "adp_multicamera_observation_frame_manifest.v1"
)
MULTICAMERA_VISUAL_EVIDENCE_SCHEMA_VERSION = (
    "adp_multicamera_episode_visual_evidence.v1"
)

_CAMERA_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


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


def _json_mapping(value: Mapping[str, Any], *, error: str) -> dict[str, Any]:
    try:
        cloned = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if not isinstance(cloned, dict):
        raise ValueError(error)
    return cloned


def _validate_camera_calibration(
    value: Mapping[str, Any],
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    calibration = _json_mapping(
        value, error="camera_calibration_not_json_mapping"
    )
    required = {
        "camera_model",
        "intrinsic_matrix",
        "world_from_camera",
        "resolution",
        "near_m",
        "far_m",
    }
    if set(calibration) < required:
        raise ValueError("camera_calibration_fields_missing")
    if calibration["camera_model"] not in {"pinhole", "opencv_pinhole"}:
        raise ValueError("camera_calibration_model_invalid")
    intrinsic = calibration["intrinsic_matrix"]
    extrinsic = calibration["world_from_camera"]
    if not (
        isinstance(intrinsic, list)
        and len(intrinsic) == 3
        and all(isinstance(row, list) and len(row) == 3 for row in intrinsic)
    ):
        raise ValueError("camera_calibration_intrinsic_shape_invalid")
    if not (
        isinstance(extrinsic, list)
        and len(extrinsic) == 4
        and all(isinstance(row, list) and len(row) == 4 for row in extrinsic)
    ):
        raise ValueError("camera_calibration_extrinsic_shape_invalid")
    numeric_values = [item for row in intrinsic for item in row] + [
        item for row in extrinsic for item in row
    ]
    try:
        finite = all(math.isfinite(float(item)) for item in numeric_values)
    except (TypeError, ValueError):
        finite = False
    if not finite:
        raise ValueError("camera_calibration_matrix_nonfinite")
    if calibration["resolution"] != [width, height]:
        raise ValueError("camera_calibration_resolution_mismatch")
    try:
        near_m = float(calibration["near_m"])
        far_m = float(calibration["far_m"])
    except (TypeError, ValueError) as exc:
        raise ValueError("camera_calibration_clip_invalid") from exc
    if not (math.isfinite(near_m) and math.isfinite(far_m) and 0 < near_m < far_m):
        raise ValueError("camera_calibration_clip_invalid")
    return calibration


def persist_camera_observation_frame(
    image: Any,
    *,
    output_dir: Path,
    episode_id: str,
    frame_index: int,
    kind: str,
    camera_id: str,
    timestamp_ns: int,
    simulation_time_s: float,
    calibration: Mapping[str, Any],
    source_device: str,
    synchronization: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist one calibrated, timestamped camera frame exactly as supplied.

    The raw RGB digest is computed before PNG encoding.  ``synchronization`` is
    retained so GPU/DLPack producers can state the device event or explicit
    synchronization that made the host-visible bytes safe to consume.
    """

    import numpy as np
    from PIL import Image

    if not _CAMERA_ID.fullmatch(camera_id):
        raise ValueError("observation_camera_id_invalid")
    if not isinstance(timestamp_ns, int) or isinstance(timestamp_ns, bool) or timestamp_ns < 0:
        raise ValueError("observation_timestamp_ns_invalid")
    if not math.isfinite(float(simulation_time_s)) or simulation_time_s < 0:
        raise ValueError("observation_simulation_time_invalid")
    if not str(source_device).strip():
        raise ValueError("observation_source_device_missing")
    sync = _json_mapping(
        synchronization, error="observation_synchronization_not_json_mapping"
    )
    if sync.get("host_bytes_ready") is not True:
        raise ValueError("observation_host_bytes_not_synchronized")

    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError(f"observation_frame_dtype_not_uint8:{array.dtype}")
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"observation_frame_shape_not_rgb:{array.shape}")
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    if kind not in {"policy-input", "terminal-observation"}:
        raise ValueError("observation_frame_kind_invalid")
    checked_calibration = _validate_camera_calibration(
        calibration, width=int(array.shape[1]), height=int(array.shape[0])
    )

    frame_dir = output_dir / "media" / episode_id / "frames" / camera_id
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
    record = {
        "frame_index": frame_index,
        "kind": kind,
        "camera_id": camera_id,
        "timestamp_ns": timestamp_ns,
        "simulation_time_s": float(simulation_time_s),
        "relative_path": path.relative_to(output_dir).as_posix(),
        "raw_rgb_sha256": "sha256:" + hashlib.sha256(array.tobytes()).hexdigest(),
        "png_sha256": _file_sha256(path),
        "size_bytes": path.stat().st_size,
        "width": int(array.shape[1]),
        "height": int(array.shape[0]),
        "channels": 3,
        "dtype": "uint8",
        "source_device": str(source_device),
        "synchronization": sync,
        "calibration": checked_calibration,
        "calibration_digest": canonical_digest(checked_calibration),
    }
    record["frame_digest"] = canonical_digest(record, digest_field="frame_digest")
    return record


def persist_multicamera_observation(
    images: Mapping[str, Any],
    *,
    output_dir: Path,
    episode_id: str,
    observation_index: int,
    kind: str,
    timestamp_ns: int,
    simulation_time_s: float,
    calibrations: Mapping[str, Mapping[str, Any]],
    source_devices: Mapping[str, str],
    synchronizations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Persist all camera views for one policy query or terminal observation."""

    camera_ids = set(images)
    if not camera_ids or camera_ids != set(calibrations):
        raise ValueError("multicamera_calibration_camera_set_mismatch")
    if camera_ids != set(source_devices):
        raise ValueError("multicamera_source_device_camera_set_mismatch")
    if camera_ids != set(synchronizations):
        raise ValueError("multicamera_synchronization_camera_set_mismatch")
    views = {
        camera_id: persist_camera_observation_frame(
            images[camera_id],
            output_dir=output_dir,
            episode_id=episode_id,
            frame_index=observation_index,
            kind=kind,
            camera_id=camera_id,
            timestamp_ns=timestamp_ns,
            simulation_time_s=simulation_time_s,
            calibration=calibrations[camera_id],
            source_device=source_devices[camera_id],
            synchronization=synchronizations[camera_id],
        )
        for camera_id in sorted(camera_ids)
    }
    record = {
        "observation_index": observation_index,
        "kind": kind,
        "timestamp_ns": timestamp_ns,
        "simulation_time_s": float(simulation_time_s),
        "camera_ids": sorted(camera_ids),
        "views": views,
        "observation_digest": "",
    }
    record["observation_digest"] = canonical_digest(
        record, digest_field="observation_digest"
    )
    return record


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


def validate_multicamera_frame_manifest(
    manifest: Mapping[str, Any],
    *,
    output_dir: Path,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Validate and optionally rehash a sealed multi-camera frame manifest."""

    from PIL import Image
    import numpy as np

    checked = _json_mapping(
        manifest, error="multicamera_frame_manifest_not_json_mapping"
    )
    errors: list[str] = []
    if checked.get("schema_version") != MULTICAMERA_FRAME_MANIFEST_SCHEMA_VERSION:
        errors.append("multicamera_frame_manifest_schema_invalid")
    required_camera_ids = set(checked.get("required_camera_ids") or [])
    if not {"external", "wrist"}.issubset(required_camera_ids):
        errors.append("multicamera_frame_manifest_external_wrist_required")
    observations = checked.get("policy_input_observations")
    if not isinstance(observations, list) or not observations:
        errors.append("multicamera_frame_manifest_policy_inputs_missing")
        observations = []
    terminal = checked.get("terminal_observation")
    all_observations = [*observations]
    if isinstance(terminal, Mapping):
        all_observations.append(dict(terminal))
    else:
        errors.append("multicamera_frame_manifest_terminal_missing")

    expected_index = 0
    previous_timestamp = -1
    previous_simulation_time = -1.0
    for observation in all_observations:
        if observation.get("observation_index") != expected_index:
            errors.append("multicamera_frame_manifest_index_not_contiguous")
        expected_index += 1
        timestamp_ns = observation.get("timestamp_ns")
        simulation_time_s = observation.get("simulation_time_s")
        if not isinstance(timestamp_ns, int) or timestamp_ns <= previous_timestamp:
            errors.append("multicamera_frame_manifest_timestamp_not_monotonic")
        else:
            previous_timestamp = timestamp_ns
        try:
            simulation_time = float(simulation_time_s)
        except (TypeError, ValueError):
            simulation_time = -1.0
        if simulation_time < previous_simulation_time:
            errors.append("multicamera_frame_manifest_simulation_time_not_monotonic")
        else:
            previous_simulation_time = simulation_time
        views = observation.get("views")
        if not isinstance(views, Mapping) or not required_camera_ids.issubset(views):
            errors.append("multicamera_frame_manifest_required_view_missing")
            continue
        if observation.get("camera_ids") != sorted(views):
            errors.append("multicamera_frame_manifest_camera_ids_mismatch")
        if observation.get("observation_digest") != canonical_digest(
            observation, digest_field="observation_digest"
        ):
            errors.append("multicamera_frame_manifest_observation_digest_mismatch")
        for camera_id, raw_frame in views.items():
            frame = dict(raw_frame) if isinstance(raw_frame, Mapping) else {}
            if frame.get("camera_id") != camera_id:
                errors.append("multicamera_frame_manifest_camera_record_mismatch")
            if frame.get("frame_digest") != canonical_digest(
                frame, digest_field="frame_digest"
            ):
                errors.append("multicamera_frame_manifest_frame_digest_mismatch")
            if frame.get("calibration_digest") != canonical_digest(
                _json_mapping(
                    frame.get("calibration") or {},
                    error="multicamera_frame_calibration_not_mapping",
                )
            ):
                errors.append("multicamera_frame_manifest_calibration_digest_mismatch")
            if not verify_files:
                continue
            path = (output_dir / str(frame.get("relative_path") or "")).resolve()
            root = output_dir.resolve()
            if root != path and root not in path.parents:
                errors.append("multicamera_frame_manifest_path_outside_output")
                continue
            if path.is_symlink() or not path.is_file():
                errors.append("multicamera_frame_manifest_file_missing")
                continue
            if frame.get("png_sha256") != _file_sha256(path):
                errors.append("multicamera_frame_manifest_png_digest_mismatch")
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            raw_digest = "sha256:" + hashlib.sha256(
                np.ascontiguousarray(rgb).tobytes()
            ).hexdigest()
            if frame.get("raw_rgb_sha256") != raw_digest:
                errors.append("multicamera_frame_manifest_raw_digest_mismatch")

    if checked.get("frame_manifest_digest") != canonical_digest(
        checked, digest_field="frame_manifest_digest"
    ):
        errors.append("multicamera_frame_manifest_digest_mismatch")
    if errors:
        raise ValueError(";".join(sorted(set(errors))))
    return checked


def finalize_multicamera_visual_evidence(
    *,
    output_dir: Path,
    episode_id: str,
    identity: Mapping[str, Any],
    policy_input_observations: Sequence[Mapping[str, Any]],
    terminal_observation: Mapping[str, Any],
    required_camera_ids: Sequence[str] = ("external", "wrist"),
    frames_per_second: float = 4.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Seal exact multi-camera inputs and one review video per camera."""

    if not policy_input_observations:
        raise ValueError("multicamera_visual_evidence_policy_inputs_missing")
    required = sorted(set(str(camera_id) for camera_id in required_camera_ids))
    if not {"external", "wrist"}.issubset(required):
        raise ValueError("multicamera_visual_evidence_external_wrist_required")
    if any(not _CAMERA_ID.fullmatch(camera_id) for camera_id in required):
        raise ValueError("multicamera_visual_evidence_camera_id_invalid")

    inputs = [dict(row) for row in policy_input_observations]
    terminal = dict(terminal_observation)
    manifest: dict[str, Any] = {
        "schema_version": MULTICAMERA_FRAME_MANIFEST_SCHEMA_VERSION,
        "episode_id": episode_id,
        "identity": _json_mapping(identity, error="multicamera_identity_not_json_mapping"),
        "required_camera_ids": required,
        "policy_input_observations": inputs,
        "terminal_observation": terminal,
        "policy_input_observation_count": len(inputs),
        "policy_input_frame_count": len(inputs) * len(required),
        "lossless_policy_inputs_are_authoritative": True,
        "derived_videos_are_human_review_convenience": True,
        "camera_calibration_and_timestamps_retained_per_observation": True,
        "frame_manifest_digest": "",
    }
    manifest["frame_manifest_digest"] = canonical_digest(
        manifest, digest_field="frame_manifest_digest"
    )
    validate_multicamera_frame_manifest(
        manifest, output_dir=output_dir, verify_files=True
    )

    manifest_path = output_dir / "media" / episode_id / "multicamera_frame_manifest.json"
    _write_json(manifest_path, manifest)
    artifacts: list[dict[str, Any]] = [
        {
            "role": "multicamera_observation_frame_manifest",
            "relative_path": manifest_path.relative_to(output_dir).as_posix(),
            "sha256": _file_sha256(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
        }
    ]
    all_observations = [*inputs, terminal]
    for observation in all_observations:
        for camera_id, frame in observation["views"].items():
            artifacts.append(
                {
                    "role": (
                        "policy_input_camera_frame"
                        if observation["kind"] == "policy-input"
                        else "terminal_observation_camera_frame"
                    ),
                    "camera_id": camera_id,
                    "observation_index": observation["observation_index"],
                    "timestamp_ns": observation["timestamp_ns"],
                    "simulation_time_s": observation["simulation_time_s"],
                    "relative_path": frame["relative_path"],
                    "sha256": frame["png_sha256"],
                    "raw_rgb_sha256": frame["raw_rgb_sha256"],
                    "calibration_digest": frame["calibration_digest"],
                    "size_bytes": frame["size_bytes"],
                }
            )

    videos: dict[str, dict[str, Any]] = {}
    for camera_id in required:
        paths = [
            output_dir / observation["views"][camera_id]["relative_path"]
            for observation in all_observations
        ]
        video_path = output_dir / "media" / episode_id / f"{camera_id}.mp4"
        if video_path.exists() or video_path.is_symlink():
            raise FileExistsError(
                f"multicamera_episode_video_overwrite_forbidden:{camera_id}"
            )
        video = _encode_episode_video(
            paths,
            video_path=video_path,
            frames_per_second=frames_per_second,
        )
        video["relative_path"] = video_path.relative_to(output_dir).as_posix()
        video["camera_id"] = camera_id
        video["derived_from_frame_manifest_digest"] = manifest[
            "frame_manifest_digest"
        ]
        videos[camera_id] = video
        artifacts.append(
            {
                "role": "camera_review_video",
                "camera_id": camera_id,
                "relative_path": video["relative_path"],
                "sha256": video["sha256"],
                "size_bytes": video["size_bytes"],
                "media_type": "video/mp4",
            }
        )

    visual = {
        "schema_version": MULTICAMERA_VISUAL_EVIDENCE_SCHEMA_VERSION,
        "status": "complete",
        "human_review_available": True,
        "frame_manifest_digest": manifest["frame_manifest_digest"],
        "policy_input_observation_count": len(inputs),
        "policy_input_frame_count": len(inputs) * len(required),
        "required_camera_ids": required,
        "terminal_observation_present": True,
        "videos": videos,
        "vlm_grading_used": False,
        "policy_self_grading_used": False,
    }
    return visual, artifacts


__all__ = [
    "FRAME_MANIFEST_SCHEMA_VERSION",
    "MULTICAMERA_FRAME_MANIFEST_SCHEMA_VERSION",
    "MULTICAMERA_VISUAL_EVIDENCE_SCHEMA_VERSION",
    "VISUAL_EVIDENCE_SCHEMA_VERSION",
    "finalize_multicamera_visual_evidence",
    "finalize_visual_evidence",
    "persist_camera_observation_frame",
    "persist_multicamera_observation",
    "persist_observation_frame",
    "validate_multicamera_frame_manifest",
]
