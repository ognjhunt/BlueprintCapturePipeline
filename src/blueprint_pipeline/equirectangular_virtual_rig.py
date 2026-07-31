"""Deterministic equirectangular-to-perspective shared-center rig compiler.

The compiler projects retained stitched panorama observations into a fixed set
of perspective views. Every view derived from one panorama is explicitly bound
to the same optical center. It does not estimate a physical camera trajectory,
recover native dual-fisheye calibration, or establish metric scale.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .decision_evidence_contracts import canonical_digest, canonical_json


EQUIRECTANGULAR_VIRTUAL_RIG_SCHEMA_VERSION = (
    "equirectangular_virtual_camera_rig.v1"
)
EQUIRECTANGULAR_COMPILATION_SCHEMA_VERSION = (
    "equirectangular_virtual_rig_compilation.v1"
)
EQUIRECTANGULAR_COMPILER_VERSION = "deterministic_equirectangular_projector.v1"
VIRTUAL_RIG_PROFILE = "blueprint_erp_shared_center_12x100deg_v1"
_MAX_PANORAMA_BYTES = 2 * 1024 * 1024 * 1024
_OUTPUT_SIZE = 512
_HORIZONTAL_FOV_DEGREES = 100.0
_VIEW_ANGLES = tuple(
    (f"pitch-{pitch:+03d}-yaw-{yaw:03d}", float(yaw), float(pitch))
    for pitch in (-45, 0, 45)
    for yaw in (0, 90, 180, 270)
)


class EquirectangularVirtualRigError(ValueError):
    """Stable fail-closed error for stitched panorama projection."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _is_digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def _safe_relative(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise EquirectangularVirtualRigError(
            ["equirectangular_source_relative_path_unsafe"]
        )
    return path.as_posix()


def _safe_source(root: Path, relative_path: str) -> Path:
    candidate = root.joinpath(*PurePosixPath(relative_path).parts)
    if candidate.is_symlink():
        raise EquirectangularVirtualRigError(
            ["equirectangular_source_symlink_forbidden"]
        )
    resolved = candidate.resolve()
    if root != resolved and root not in resolved.parents:
        raise EquirectangularVirtualRigError(["equirectangular_source_path_escape"])
    if resolved.is_symlink() or not resolved.is_file():
        raise EquirectangularVirtualRigError(["equirectangular_source_missing"])
    return resolved


def _timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise EquirectangularVirtualRigError(
            ["equirectangular_timestamp_invalid"]
        ) from exc
    if parsed.tzinfo is None:
        raise EquirectangularVirtualRigError(["equirectangular_timestamp_invalid"])
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(canonical_json(dict(value)))
    payload = (canonical_json(normalized) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EquirectangularVirtualRigError(
                ["equirectangular_immutable_artifact_invalid"]
            ) from exc
        if canonical_json(existing) != canonical_json(normalized):
            raise EquirectangularVirtualRigError(
                ["equirectangular_immutable_artifact_conflict"]
            )
        return dict(existing)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return _write_immutable(path, normalized)
    finally:
        temporary.unlink(missing_ok=True)
    return normalized


def _rotation(yaw_degrees: float, pitch_degrees: float) -> np.ndarray:
    yaw = math.radians(yaw_degrees)
    pitch = math.radians(pitch_degrees)
    yaw_rotation = np.asarray(
        [
            [math.cos(yaw), 0.0, math.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-math.sin(yaw), 0.0, math.cos(yaw)],
        ],
        dtype=np.float64,
    )
    pitch_rotation = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ],
        dtype=np.float64,
    )
    return yaw_rotation @ pitch_rotation


def _virtual_intrinsics() -> dict[str, Any]:
    focal = (_OUTPUT_SIZE / 2.0) / math.tan(
        math.radians(_HORIZONTAL_FOV_DEGREES) / 2.0
    )
    return {
        "camera_model": "PINHOLE",
        "width": _OUTPUT_SIZE,
        "height": _OUTPUT_SIZE,
        "fx": round(focal, 12),
        "fy": round(focal, 12),
        "cx": _OUTPUT_SIZE / 2.0,
        "cy": _OUTPUT_SIZE / 2.0,
        "horizontal_fov_degrees": _HORIZONTAL_FOV_DEGREES,
        "distortion_model": "none_after_deterministic_projection",
    }


def _project(
    panorama: np.ndarray, *, yaw_degrees: float, pitch_degrees: float
) -> np.ndarray:
    height, width = panorama.shape[:2]
    focal = float(_virtual_intrinsics()["fx"])
    coordinates = np.arange(_OUTPUT_SIZE, dtype=np.float64) + 0.5
    grid_x, grid_y = np.meshgrid(coordinates, coordinates)
    rays = np.stack(
        (
            (grid_x - _OUTPUT_SIZE / 2.0) / focal,
            (grid_y - _OUTPUT_SIZE / 2.0) / focal,
            np.ones_like(grid_x),
        ),
        axis=-1,
    )
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    rotated = rays @ _rotation(yaw_degrees, pitch_degrees).T
    longitude = np.arctan2(rotated[..., 0], rotated[..., 2])
    latitude = np.arcsin(np.clip(-rotated[..., 1], -1.0, 1.0))
    source_x = ((longitude / (2.0 * math.pi) + 0.5) * width - 0.5) % width
    source_y = np.clip((0.5 - latitude / math.pi) * height - 0.5, 0, height - 1)
    x0 = np.floor(source_x).astype(np.int64)
    y0 = np.floor(source_y).astype(np.int64)
    x1 = (x0 + 1) % width
    y1 = np.minimum(y0 + 1, height - 1)
    weight_x = (source_x - x0)[..., None]
    weight_y = (source_y - y0)[..., None]
    top = panorama[y0, x0] * (1.0 - weight_x) + panorama[y0, x1] * weight_x
    bottom = panorama[y1, x0] * (1.0 - weight_x) + panorama[y1, x1] * weight_x
    projected = top * (1.0 - weight_y) + bottom * weight_y
    return np.clip(np.rint(projected), 0, 255).astype(np.uint8)


def _view_rotation_matrix(yaw_degrees: float, pitch_degrees: float) -> list[list[float]]:
    rotation = _rotation(yaw_degrees, pitch_degrees)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    return [[round(float(value), 12) for value in row] for row in transform]


def compile_equirectangular_virtual_rig(
    *,
    capture_root: str | Path,
    output_root: str | Path,
    intake_id: str,
    capture_digest: str,
    stitched_source_metadata: Mapping[str, Any],
    panorama_observations: Sequence[Mapping[str, Any]],
    source_commit_sha: str,
    implementation_digest: str,
    authority_used: Mapping[str, Any],
    timestamp: str,
    access_scope: str = "candidate_training_and_validation_only",
    maximum_panorama_bytes: int = _MAX_PANORAMA_BYTES,
    parent_artifact_or_event: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile one fixed shared-center virtual rig per retained panorama."""

    if (
        not str(intake_id).strip()
        or not _is_digest(capture_digest)
        or not _is_digest(implementation_digest)
        or len(source_commit_sha) != 40
        or any(character not in "0123456789abcdef" for character in source_commit_sha)
    ):
        raise EquirectangularVirtualRigError(
            ["equirectangular_source_binding_invalid"]
        )
    if access_scope not in {
        "candidate_training_and_validation_only",
        "independent_evaluator_only",
    }:
        raise EquirectangularVirtualRigError(["equirectangular_access_scope_invalid"])
    required_authority = {
        "source_capture_rights_valid": True,
        "consent_valid": True,
        "privacy_review_valid": True,
        "retention_authorized": True,
        "local_processing_authorized": True,
        "provider_upload_authorized": False,
        "paid_compute_authorized": False,
    }
    if any(
        authority_used.get(key) is not expected
        for key, expected in required_authority.items()
    ):
        raise EquirectangularVirtualRigError(["equirectangular_authority_invalid"])
    metadata = dict(stitched_source_metadata)
    if (
        metadata.get("schema_version") != "stitched_equirectangular_source.v1"
        or metadata.get("source_capture_digest") != capture_digest
        or metadata.get("projection") != "equirectangular_2_to_1"
        or metadata.get("stitching_provenance")
        not in {"customer_provided", "official_sdk_produced", "externally_produced"}
        or not str(metadata.get("producer_identity") or "").strip()
        or not _is_digest(metadata.get("stitching_receipt_digest"))
        or metadata.get("original_360_source_preserved") is not True
        or not _is_digest(metadata.get("original_360_source_digest"))
        or not isinstance(metadata.get("spherical_pixel_mapping"), Mapping)
    ):
        raise EquirectangularVirtualRigError(
            ["equirectangular_source_metadata_invalid"]
        )
    if maximum_panorama_bytes <= 0:
        raise EquirectangularVirtualRigError(["equirectangular_size_limit_invalid"])
    compiled_at = _timestamp(timestamp)
    root = Path(capture_root).expanduser().resolve()
    if not root.is_dir():
        raise EquirectangularVirtualRigError(["equirectangular_capture_root_missing"])
    if not isinstance(panorama_observations, Sequence) or not panorama_observations:
        raise EquirectangularVirtualRigError(["equirectangular_observations_missing"])
    normalized_observations: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    previous_time: float | None = None
    total_bytes = 0
    for index, raw in enumerate(panorama_observations):
        if not isinstance(raw, Mapping):
            raise EquirectangularVirtualRigError(
                ["equirectangular_observation_invalid"]
            )
        observation_id = str(raw.get("observation_id") or "").strip()
        relative_path = _safe_relative(raw.get("relative_path"))
        split = str(raw.get("split") or "")
        if (
            not observation_id
            or observation_id in seen_ids
            or relative_path in seen_paths
            or split not in {"training", "validation", "held_out"}
            or (
                access_scope == "candidate_training_and_validation_only"
                and split == "held_out"
            )
            or (access_scope == "independent_evaluator_only" and split != "held_out")
        ):
            raise EquirectangularVirtualRigError(
                ["equirectangular_observation_scope_invalid"]
            )
        seen_ids.add(observation_id)
        seen_paths.add(relative_path)
        time_value = raw.get("t_video_sec")
        if (
            isinstance(time_value, bool)
            or not isinstance(time_value, (int, float))
            or not math.isfinite(float(time_value))
            or float(time_value) < 0
            or (previous_time is not None and float(time_value) <= previous_time)
        ):
            raise EquirectangularVirtualRigError(
                ["equirectangular_observation_timing_invalid"]
            )
        previous_time = float(time_value)
        source = _safe_source(root, relative_path)
        total_bytes += source.stat().st_size
        if total_bytes > maximum_panorama_bytes:
            raise EquirectangularVirtualRigError(
                ["equirectangular_source_oversized"]
            )
        digest = _sha256_file(source)
        if digest != raw.get("digest"):
            raise EquirectangularVirtualRigError(
                ["equirectangular_source_digest_mismatch"]
            )
        try:
            with Image.open(source) as image:
                image.load()
                width, height = image.size
                mode = image.mode
        except (OSError, ValueError) as exc:
            raise EquirectangularVirtualRigError(
                ["equirectangular_source_undecodable"]
            ) from exc
        if width != height * 2 or width < 4 or height < 2:
            raise EquirectangularVirtualRigError(
                ["equirectangular_source_dimensions_invalid"]
            )
        normalized_observations.append(
            {
                "ordinal": index,
                "observation_id": observation_id,
                "relative_path": relative_path,
                "digest": digest,
                "size_bytes": source.stat().st_size,
                "t_video_sec": round(float(time_value), 9),
                "split": split,
                "width": width,
                "height": height,
                "pixel_mode": mode,
            }
        )
    configuration = {
        "compiler_version": EQUIRECTANGULAR_COMPILER_VERSION,
        "virtual_rig_profile": VIRTUAL_RIG_PROFILE,
        "capture_digest": capture_digest,
        "stitched_source_metadata_digest": canonical_digest(metadata),
        "observation_binding_digest": canonical_digest(
            {"observations": normalized_observations}
        ),
        "access_scope": access_scope,
        "implementation_digest": implementation_digest,
        "source_commit_sha": source_commit_sha,
        "parent_artifact_digest": canonical_digest(
            dict(parent_artifact_or_event or {})
        ),
    }
    configuration_digest = canonical_digest(configuration)
    artifact_root = (
        Path(output_root).expanduser().resolve()
        / f"equirectangular_virtual_rig_{configuration_digest[7:23]}"
    )
    intrinsics = _virtual_intrinsics()
    virtual_observations: list[dict[str, Any]] = []
    for observation in normalized_observations:
        source = root / observation["relative_path"]
        with Image.open(source) as image:
            panorama = np.asarray(image.convert("RGB"), dtype=np.float64)
        group_id = f"shared-center-{observation['observation_id']}"
        for view_id, yaw, pitch in _VIEW_ANGLES:
            relative_output = (
                Path("virtual_views")
                / observation["split"]
                / observation["observation_id"]
                / f"{view_id}.png"
            )
            output = artifact_root / relative_output
            projected = _project(
                panorama, yaw_degrees=yaw, pitch_degrees=pitch
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                with Image.open(output) as existing:
                    if not np.array_equal(np.asarray(existing.convert("RGB")), projected):
                        raise EquirectangularVirtualRigError(
                            ["equirectangular_virtual_view_conflict"]
                        )
            else:
                descriptor, temporary_name = tempfile.mkstemp(
                    suffix=".png", dir=output.parent
                )
                os.close(descriptor)
                temporary = Path(temporary_name)
                try:
                    Image.fromarray(projected).save(temporary, format="PNG")
                    temporary.replace(output)
                finally:
                    temporary.unlink(missing_ok=True)
            virtual_observations.append(
                {
                    "virtual_observation_id": f"{observation['observation_id']}:{view_id}",
                    "source_observation_id": observation["observation_id"],
                    "source_panorama_digest": observation["digest"],
                    "source_t_video_sec": observation["t_video_sec"],
                    "split": observation["split"],
                    "shared_optical_center_group_id": group_id,
                    "independent_physical_camera": False,
                    "yaw_degrees": yaw,
                    "pitch_degrees": pitch,
                    "intrinsics": intrinsics,
                    "T_panorama_virtual_camera": _view_rotation_matrix(yaw, pitch),
                    "relative_path": relative_output.as_posix(),
                    "digest": _sha256_file(output),
                }
            )
    rig = {
        "schema_version": EQUIRECTANGULAR_VIRTUAL_RIG_SCHEMA_VERSION,
        "capture_digest": capture_digest,
        "virtual_rig_profile": VIRTUAL_RIG_PROFILE,
        "access_scope": access_scope,
        "stitched_source_metadata_digest": canonical_digest(metadata),
        "stitching_provenance": metadata["stitching_provenance"],
        "producer_identity": metadata["producer_identity"],
        "stitching_receipt_digest": metadata["stitching_receipt_digest"],
        "original_360_source_digest": metadata["original_360_source_digest"],
        "source_observations": normalized_observations,
        "virtual_observations": virtual_observations,
        "views_per_panorama": len(_VIEW_ANGLES),
        "same_optical_center_constraint_required": True,
        "rig_constrained_pose_estimation_required": True,
        "virtual_views_are_independent_physical_cameras": False,
        "candidate_may_change_virtual_view_definitions": False,
        "source_panorama_pixels_remain_authoritative": True,
        "camera_trajectory_status": "not_established",
        "metric_scale_status": "not_established",
    }
    rig["virtual_rig_digest"] = canonical_digest(
        rig, digest_field="virtual_rig_digest"
    )
    rig = _write_immutable(
        artifact_root / "equirectangular_virtual_camera_rig.json", rig
    )
    rig_path = artifact_root / "equirectangular_virtual_camera_rig.json"
    result_path = artifact_root / "equirectangular_virtual_rig_compilation.json"
    persisted_timestamp = compiled_at
    if result_path.exists():
        try:
            persisted_timestamp = _timestamp(
                json.loads(result_path.read_text(encoding="utf-8")).get("timestamp")
            )
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            raise EquirectangularVirtualRigError(
                ["equirectangular_immutable_artifact_invalid"]
            ) from exc
    result = {
        "schema_version": EQUIRECTANGULAR_COMPILATION_SCHEMA_VERSION,
        "stable_run_identity": f"equirectangular-rig-{configuration_digest[7:31]}",
        "source_capture_identity": intake_id,
        "source_capture_digest": capture_digest,
        "original_file_references": [
            {
                "relative_path": row["relative_path"],
                "digest": row["digest"],
                "size_bytes": row["size_bytes"],
            }
            for row in normalized_observations
        ],
        "producing_method": EQUIRECTANGULAR_COMPILER_VERSION,
        "implementation_version": implementation_digest,
        "container_image_digest": None,
        "source_commit_sha": source_commit_sha,
        "deterministic_configuration_digest": configuration_digest,
        "access_scope": access_scope,
        "input_digests": {
            "stitched_source_metadata_digest": canonical_digest(metadata),
            "observation_binding_digest": configuration[
                "observation_binding_digest"
            ],
            "authority_digest": canonical_digest(authority_used),
        },
        "stitching_provenance": metadata["stitching_provenance"],
        "stitching_producer_identity": metadata["producer_identity"],
        "stitching_receipt_digest": metadata["stitching_receipt_digest"],
        "original_360_source_digest": metadata["original_360_source_digest"],
        "output_digests": {"virtual_rig_digest": rig["virtual_rig_digest"]},
        "train_heldout_split_digest": None,
        "camera_calibration_binding": rig["virtual_rig_digest"],
        "coordinate_frame_declaration": dict(metadata["spherical_pixel_mapping"]),
        "units": "source_pixels_virtual_camera_radians_and_seconds",
        "metric_scale_status": "not_established",
        "provider_runtime_identity": {
            "provider": "local",
            "runtime": "numpy_pillow_deterministic_projection",
        },
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "authority_used": dict(authority_used),
        "warnings": [
            "stitched_pixels_may_contain_seam_or_near_field_ghosting",
            "virtual_views_do_not_create_independent_physical_observations",
        ],
        "blockers": [
            "physical_camera_trajectory_not_established",
            "metric_scale_not_established",
            "stitch_quality_not_independently_qualified",
        ],
        "proof_effect": "deterministic_shared_center_projection_only",
        "claim_ceiling": "equirectangular_virtual_camera_rig",
        "parent_artifact_or_event": dict(parent_artifact_or_event or {}),
        "timestamp": persisted_timestamp,
        "artifact_references": {
            "equirectangular_virtual_camera_rig": {
                "relative_path": rig_path.relative_to(artifact_root).as_posix(),
                "digest": _sha256_file(rig_path),
            }
        },
        "virtual_observation_count": len(virtual_observations),
        "source_panorama_pixels_remain_authoritative": True,
        "virtual_views_are_captured_evidence": False,
        "virtual_views_are_independent_physical_cameras": False,
        "camera_trajectory_proven": False,
        "metric_scale_proven": False,
        "appearance_reconstruction_proven": False,
        "collision_geometry_proven": False,
        "isaac_compatibility_proven": False,
    }
    result["equirectangular_compilation_digest"] = canonical_digest(
        result, digest_field="equirectangular_compilation_digest"
    )
    return _write_immutable(result_path, result)


__all__ = [
    "EQUIRECTANGULAR_COMPILATION_SCHEMA_VERSION",
    "EQUIRECTANGULAR_VIRTUAL_RIG_SCHEMA_VERSION",
    "EquirectangularVirtualRigError",
    "VIRTUAL_RIG_PROFILE",
    "compile_equirectangular_virtual_rig",
]
