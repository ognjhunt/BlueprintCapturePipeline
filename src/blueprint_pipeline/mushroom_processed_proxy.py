"""Compile MuSHRoom processed iPhone observations into an isolated trainer proxy.

MuSHRoom publishes posed RGB/depth images, not retained video or ARKit streams.
This adapter therefore preserves the publisher's long-capture test list and the
entire independent short trajectory behind an evaluator-only boundary.  It
exports only the remaining long-capture observations to candidate COLMAP text.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .decision_evidence_contracts import canonical_digest, canonical_json
from .pose_image_consistency import check_two_view_epipolar_consistency
from .reconstruction_colmap_dataset import export_colmap_training_dataset


SCHEMA_VERSION = "mushroom_processed_iphone_proxy.v1"
COMPILER_VERSION = "mushroom_processed_iphone_proxy_compiler.v3"
# The published transform_matrix values are nerfstudio/OpenGL-convention
# camera-to-world (x right, y up, z backward).  Two-view epipolar evidence on
# the real candidate images (median 0.55px converted vs 2.1px unconverted on
# the best-conditioned pairs) established the convention; the compiler converts
# to the OpenCV convention the COLMAP exporter requires and then re-verifies
# the converted poses against the pixels.
_OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0])
COORDINATE_FRAME_DECLARATION = {
    "declaration": "mushroom_published_camera_to_world_opengl_converted_to_opencv",
    "camera_axis_convention": "opencv_x_right_y_down_z_forward",
    "camera_convention_evidence": "two_view_epipolar_consistency",
    "handedness": "not_independently_declared",
    "gravity_alignment": "not_independently_validated",
}
BASE_BLOCKERS = (
    "raw_video_and_timestamps_missing",
    "arkit_sensor_streams_missing",
    "metric_scale_not_independently_validated",
)
ARCHIVE_SHA256 = "sha256:68735cfa0758e1288a006c30dc8b95ffb4caa3392bc9c68c0c3ea6c111966518"
ARCHIVE_SIZE_BYTES = 146_575_749
PUBLISHER_MD5 = "a359dba714e7829be11747ce5dee141c"
_FRAME = re.compile(r"frame_[0-9]{5}")


class MushroomProcessedProxyError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: Any, *, prefix: str) -> str:
    text = str(value or "").removeprefix("./").replace("\\", "/")
    relative = PurePosixPath(text)
    if (
        not text
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or not text.startswith(prefix + "/")
    ):
        raise MushroomProcessedProxyError(["mushroom_observation_path_unsafe"])
    return relative.as_posix()


def _write_immutable(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(canonical_json(dict(value)))
    payload = (canonical_json(normalized) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise MushroomProcessedProxyError(["mushroom_immutable_artifact_conflict"])
        return normalized
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise MushroomProcessedProxyError(["mushroom_immutable_artifact_conflict"])
    finally:
        temporary.unlink(missing_ok=True)
    return normalized


def _materialize(source: Path, destination: Path, digest: str) -> None:
    if source.is_symlink() or not source.is_file() or "sha256:" + _hash(source) != digest:
        raise MushroomProcessedProxyError(["mushroom_observation_digest_mismatch"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink() or "sha256:" + _hash(destination) != digest:
            raise MushroomProcessedProxyError(["mushroom_materialized_observation_conflict"])
        return
    try:
        os.link(source, destination)
    except OSError:
        destination.write_bytes(source.read_bytes())


def _load_trajectory(root: Path, capture: str) -> list[dict[str, Any]]:
    path = root / capture / "transformations_colmap.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MushroomProcessedProxyError(["mushroom_trajectory_unreadable"]) from exc
    if document.get("camera_model") != "OPENCV" or not isinstance(document.get("frames"), list):
        raise MushroomProcessedProxyError(["mushroom_camera_contract_invalid"])
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in document["frames"]:
        if not isinstance(raw, Mapping):
            raise MushroomProcessedProxyError(["mushroom_trajectory_frame_invalid"])
        image_relative = _safe_relative(raw.get("file_path"), prefix="images")
        depth_relative = _safe_relative(raw.get("depth_file_path"), prefix="depth")
        frame_id = Path(image_relative).stem
        if _FRAME.fullmatch(frame_id) is None or frame_id in seen or Path(depth_relative).stem != frame_id:
            raise MushroomProcessedProxyError(["mushroom_frame_binding_invalid"])
        try:
            matrix = np.asarray(raw["transform_matrix"], dtype=np.float64)
            width, height = int(raw["w"]), int(raw["h"])
            intrinsics = {key: float(raw[key]) for key in ("fl_x", "fl_y", "cx", "cy")}
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise MushroomProcessedProxyError(["mushroom_camera_binding_invalid"]) from exc
        if (
            matrix.shape != (4, 4)
            or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-4)
            or width <= 0
            or height <= 0
            or not all(math.isfinite(value) and value > 0 for value in intrinsics.values())
        ):
            raise MushroomProcessedProxyError(["mushroom_camera_binding_invalid"])
        image_path, depth_path = root / capture / image_relative, root / capture / depth_relative
        if image_path.is_symlink() or depth_path.is_symlink() or not image_path.is_file() or not depth_path.is_file():
            raise MushroomProcessedProxyError(["mushroom_observation_missing"])
        with Image.open(image_path) as image:
            if image.size != (width, height):
                raise MushroomProcessedProxyError(["mushroom_image_dimensions_invalid"])
        with Image.open(depth_path) as depth:
            if depth.size != (width, height):
                raise MushroomProcessedProxyError(["mushroom_depth_dimensions_invalid"])
        converted = matrix.copy()
        converted[:3, :3] = converted[:3, :3] @ _OPENGL_TO_OPENCV
        rows.append(
            {
                "frame_id": frame_id,
                "image_path": image_path,
                "depth_path": depth_path,
                "image_digest": "sha256:" + _hash(image_path),
                "depth_digest": "sha256:" + _hash(depth_path),
                "camera": {
                    "T_world_camera": converted.tolist(),
                    "rgb_intrinsics": {
                        "width": width,
                        "height": height,
                        "fx": intrinsics["fl_x"],
                        "fy": intrinsics["fl_y"],
                        "cx": intrinsics["cx"],
                        "cy": intrinsics["cy"],
                    },
                    "pose_source": "mushroom_published_camera_to_world_opengl_converted_to_opencv",
                },
            }
        )
        seen.add(frame_id)
    return sorted(rows, key=lambda row: row["frame_id"])


def build_mushroom_colmap_export_request(
    *,
    source_capture_digest: str,
    source_commit_sha: str,
    dataset_digest: str,
    split_digest: str,
    camera_observation_manifest: Mapping[str, Any],
    candidate_dataset_manifest: Mapping[str, Any],
    authority_used: Mapping[str, Any],
    timestamp: str,
    configuration_digest: str,
    blockers: Sequence[str] = BASE_BLOCKERS,
) -> dict[str, Any]:
    """Deterministically rebuild the proxy's COLMAP export request.

    The compiler and any later initialization-binding runner must produce the
    byte-identical request so the recorded ``request_digest`` can be verified
    before a derived (for example point-seeded) request is created.  Runners
    pass the recorded result's ``blockers`` so gate outcomes replay exactly.
    """

    return {
        "schema_version": "colmap_training_dataset_export_request.v1",
        "stable_run_identity": f"mushroom-koivu-{configuration_digest[7:19]}",
        "source_capture_digest": source_capture_digest,
        "source_commit_sha": source_commit_sha,
        "reconstruction_dataset_digest": dataset_digest,
        "frozen_split_digest": split_digest,
        "camera_observation_manifest": json.loads(canonical_json(dict(camera_observation_manifest))),
        "candidate_dataset_manifest": json.loads(canonical_json(dict(candidate_dataset_manifest))),
        "coordinate_frame_declaration": dict(COORDINATE_FRAME_DECLARATION),
        "units": "publisher_pose_units_not_independently_validated",
        "metric_scale_status": "not_independently_validated",
        "authority_used": dict(authority_used),
        "blockers": sorted(set(str(blocker) for blocker in blockers)),
        "timestamp": timestamp,
    }


def compile_mushroom_processed_iphone_proxy(
    *,
    scene_root: str | Path,
    archive_path: str | Path,
    output_root: str | Path,
    source_commit_sha: str,
    authority_used: Mapping[str, Any],
    timestamp: str,
) -> dict[str, Any]:
    """Freeze publisher-held-out observations and export candidate-only COLMAP."""

    archive = Path(archive_path).resolve()
    root = Path(scene_root).resolve()
    if (
        not archive.is_file()
        or archive.stat().st_size != ARCHIVE_SIZE_BYTES
        or "sha256:" + _hash(archive) != ARCHIVE_SHA256
        or _hash(archive, "md5") != PUBLISHER_MD5
    ):
        raise MushroomProcessedProxyError(["mushroom_archive_binding_invalid"])
    if (
        authority_used.get("license") != "CC-BY-4.0"
        or authority_used.get("local_processing_authorized") is not True
        or not isinstance(authority_used.get("provider_upload_authorized"), bool)
        or len(source_commit_sha) != 40
        or any(character not in "0123456789abcdef" for character in source_commit_sha)
    ):
        raise MushroomProcessedProxyError(["mushroom_authority_or_source_invalid"])
    long_rows, short_rows = _load_trajectory(root, "long_capture"), _load_trajectory(root, "short_capture")
    try:
        author_test = {
            line.strip()
            for line in (root / "long_capture" / "test.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    except OSError as exc:
        raise MushroomProcessedProxyError(["mushroom_author_test_unreadable"]) from exc
    long_ids = {row["frame_id"] for row in long_rows}
    if not author_test or not author_test <= long_ids:
        raise MushroomProcessedProxyError(["mushroom_author_test_binding_invalid"])
    remaining = [row for row in long_rows if row["frame_id"] not in author_test]
    validation_count = max(1, round(len(remaining) * 0.1))
    validation_ids = {
        row["frame_id"]
        for row in sorted(remaining, key=lambda row: (row["image_digest"], row["frame_id"]))[:validation_count]
    }
    split = {
        row["frame_id"]: (
            "hidden_heldout" if row["frame_id"] in author_test else "validation" if row["frame_id"] in validation_ids else "training"
        )
        for row in long_rows
    }
    split_digest = canonical_digest({"long": split, "short": [row["frame_id"] for row in short_rows]})
    source_capture_digest = canonical_digest(
        {"dataset": "MuSHRoom", "scene": "koivu", "device": "iphone", "archive": ARCHIVE_SHA256}
    )
    configuration_digest = canonical_digest(
        {"source_capture_digest": source_capture_digest, "split_digest": split_digest, "compiler": COMPILER_VERSION}
    )
    artifact_root = Path(output_root).resolve() / f"mushroom_proxy_{configuration_digest[7:23]}"
    original_file_references = [
        {
            "artifact_id": "koivu_iphone.tar.gz",
            "digest": ARCHIVE_SHA256,
            "size_bytes": ARCHIVE_SIZE_BYTES,
        },
        *[
            {
                "artifact_id": relative,
                "digest": "sha256:" + _hash(root / relative),
                "size_bytes": (root / relative).stat().st_size,
            }
            for relative in (
                "long_capture/transformations_colmap.json",
                "long_capture/test.txt",
                "short_capture/transformations_colmap.json",
            )
        ],
    ]
    observations: list[dict[str, Any]] = []
    candidate_frames: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    for row in long_rows:
        assignment = split[row["frame_id"]]
        relative = f"candidate_dataset/{assignment}/{row['frame_id']}.jpg"
        if assignment == "hidden_heldout":
            relative = f"evaluator_hidden/long/{row['frame_id']}.jpg"
            _materialize(row["image_path"], artifact_root / relative, row["image_digest"])
            hidden_rows.append({"observation_id": row["frame_id"], "trajectory": "long_author_test", "image_relative_path": relative, "image_digest": row["image_digest"], "camera": row["camera"]})
            continue
        _materialize(row["image_path"], artifact_root / relative, row["image_digest"])
        candidate_frames.append({"frame_id": row["frame_id"], "split": assignment, "frame_digest": row["image_digest"], "candidate_relative_path": relative})
        observations.append({"observation_id": row["frame_id"], "split": assignment, "image_relative_path": relative, "image_digest": row["image_digest"], "camera": row["camera"]})
    for row in short_rows:
        relative = f"evaluator_hidden/short/{row['frame_id']}.jpg"
        _materialize(row["image_path"], artifact_root / relative, row["image_digest"])
        hidden_rows.append({"observation_id": f"short-{row['frame_id']}", "trajectory": "independent_short", "image_relative_path": relative, "image_digest": row["image_digest"], "camera": row["camera"]})
    candidate = {"schema_version": "candidate_reconstruction_dataset_manifest.v1", "capture_digest": source_capture_digest, "split_digest": split_digest, "training_and_validation_only": True, "heldout_pixels_included": False, "frames": candidate_frames}
    candidate["candidate_dataset_digest"] = canonical_digest(candidate, digest_field="candidate_dataset_digest")
    camera = {"schema_version": "mushroom_camera_observation_manifest.v1", "capture_digest": source_capture_digest, "split_digest": split_digest, "hidden_heldout_pixels_included": False, "candidate_splits_only": True, "candidate_may_access_hidden_heldout": False, "candidate_may_modify_poses_or_calibration": False, "observations": observations}
    camera["camera_observation_digest"] = canonical_digest(camera, digest_field="camera_observation_digest")
    hidden = {"schema_version": "mushroom_hidden_evaluator_manifest.v1", "capture_digest": source_capture_digest, "split_digest": split_digest, "candidate_access_permitted": False, "observations": hidden_rows}
    hidden["hidden_evaluator_digest"] = canonical_digest(hidden, digest_field="hidden_evaluator_digest")
    _write_immutable(artifact_root / "candidate_dataset_manifest.json", candidate)
    _write_immutable(artifact_root / "camera_observation_manifest.json", camera)
    _write_immutable(artifact_root / "evaluator_hidden" / "hidden_evaluator_manifest.json", hidden)
    dataset_digest = canonical_digest({"candidate": candidate["candidate_dataset_digest"], "hidden": hidden["hidden_evaluator_digest"]})
    pose_consistency = check_two_view_epipolar_consistency(
        observations=observations, image_root=artifact_root
    )
    if pose_consistency["status"] == "inconsistent":
        raise MushroomProcessedProxyError(["mushroom_pose_image_consistency_inconsistent"])
    blockers = list(BASE_BLOCKERS)
    if pose_consistency["status"] != "consistent":
        blockers.append("pose_image_consistency_inconclusive")
    colmap = export_colmap_training_dataset(
        source_artifact=build_mushroom_colmap_export_request(
            source_capture_digest=source_capture_digest,
            source_commit_sha=source_commit_sha,
            dataset_digest=dataset_digest,
            split_digest=split_digest,
            camera_observation_manifest=camera,
            candidate_dataset_manifest=candidate,
            authority_used=authority_used,
            timestamp=timestamp,
            configuration_digest=configuration_digest,
            blockers=blockers,
        ),
        artifact_root=artifact_root,
        output_root=artifact_root / "trainer_input",
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "stable_run_identity": f"mushroom-koivu-{configuration_digest[7:31]}",
        "status": "candidate_training_proxy_ready",
        "source_capture_identity": "mushroom-koivu-iphone-processed",
        "source_capture_digest": source_capture_digest,
        "source_commit_sha": source_commit_sha,
        "original_file_references": original_file_references,
        "archive_digest": ARCHIVE_SHA256,
        "license": "CC-BY-4.0",
        "producing_method": COMPILER_VERSION,
        "implementation_version": "1.0.0",
        "container_image_digest": None,
        "deterministic_configuration_digest": configuration_digest,
        "authority_used": dict(authority_used),
        "frozen_split_digest": split_digest,
        "candidate_count": len(candidate_frames),
        "author_hidden_count": len(author_test),
        "independent_short_count": len(short_rows),
        "camera_observation_digest": camera["camera_observation_digest"],
        "hidden_evaluator_digest": hidden["hidden_evaluator_digest"],
        "colmap_training_dataset_export_result": colmap,
        "input_digests": [row["digest"] for row in original_file_references],
        "output_digests": [
            candidate["candidate_dataset_digest"],
            camera["camera_observation_digest"],
            hidden["hidden_evaluator_digest"],
            colmap["colmap_training_dataset_digest"],
        ],
        "camera_calibration_binding": {
            "source_camera_model": "OPENCV",
            "available_parameters": ["fx", "fy", "cx", "cy"],
            "distortion_parameters_available": False,
            "export_camera_model": "PINHOLE",
            "camera_observation_digest": camera["camera_observation_digest"],
        },
        "coordinate_frame_declaration": dict(COORDINATE_FRAME_DECLARATION),
        "pose_image_consistency": pose_consistency,
        "units": "publisher_pose_units_not_independently_validated",
        "metric_scale_status": "not_independently_validated",
        "provider_runtime_identity": {"provider": "local", "runtime": "python_numpy_pillow"},
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "candidate_may_access_hidden_heldout": False,
        "raw_contract_3_2_proven": False,
        "metric_scale_proven": False,
        "proof_effect": "public_processed_dataset_trainer_input_only",
        "claim_ceiling": "processed_posed_image_reconstruction_proxy",
        "warnings": [
            "publisher_opencv_label_has_no_distortion_coefficients_and_is_exported_as_explicit_pinhole",
            "processed_images_are_not_retained_video_observations",
            "published_pose_rotations_converted_from_opengl_to_opencv_camera_convention",
        ],
        "blockers": sorted(set(blockers)),
        "parent_artifact_or_event": {
            "dataset": "MuSHRoom",
            "doi": "10.5281/zenodo.10230733",
            "archive_digest": ARCHIVE_SHA256,
        },
        "timestamp": timestamp,
    }
    report["mushroom_processed_proxy_digest"] = canonical_digest(report, digest_field="mushroom_processed_proxy_digest")
    return _write_immutable(artifact_root / "mushroom_processed_proxy.json", report)


__all__ = [
    "MushroomProcessedProxyError",
    "build_mushroom_colmap_export_request",
    "compile_mushroom_processed_iphone_proxy",
]
