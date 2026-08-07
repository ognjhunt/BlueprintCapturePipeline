"""Materialize depth-correct live hybrid policy observations from retained bytes.

This module joins two independently executed, digest-bound runs:

* the official AuraFusion360 native render of the sealed appearance layer,
  produced at the exact live Isaac camera poses; and
* the native Isaac Lab / Arena microcheck that produced the dynamic robot and
  approved-task-object RGB, metric depth, and semantic segmentation.

It never re-renders, never mutates a sealed asset, and never accepts a caller
assertion in place of bytes.  Identity failures raise; scientific inadequacy is
retained as a typed blocker so the evidence survives for review.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .adp009d_live_hybrid_observation import (
    LiveHybridObservationError,
    compose_live_hybrid_observation,
)
from .common import sha256_file, write_json
from .decision_evidence_contracts import canonical_digest

FRAME_MANIFEST_SCHEMA_VERSION = "adp009d_live_hybrid_frame_manifest.v1"
AURA_RESULT_SCHEMA_VERSION = "adp009d_aura_native_live_camera_result.v1"
ISAAC_RESULT_SCHEMA_VERSION = "adp009d_native_microcheck.v1"

APPROVED_TASK_OBJECT_LABEL = "approved_can"
AURA_DEPTH_SOURCE = "surf_depth_expected_camera_z_m"

# An object resting on a shelf is legitimately occluded along its contact line:
# the support surface is genuinely in front of the object's lowest pixels.  Only
# occlusion of the object's body above that band indicates a defective
# appearance layer, so the two are counted separately and gated separately.
SUPPORT_HEIGHT_M = 0.5264650138348479
CONTACT_BAND_ABOVE_SUPPORT_M = 0.01
# Preregistered before any policy is queried and before any task outcome exists.
MAX_BODY_OCCLUSION_FRACTION = 0.05

# The probe converts Isaac's OpenGL camera orientation to the OpenCV basis this
# composition assumes.  That conversion is recomputed here from the raw Isaac
# pose so a silent probe-side frame error cannot reach a policy observation.
GL_TO_CV_MAX_ABS_ERROR = 1e-6

BLOCKER_APPROVED_OBJECT_ABSENT = "hybrid_observation_approved_task_object_absent"
BLOCKER_APPROVED_OBJECT_OCCLUDED = (
    "sealed_aura_appearance_layer_occludes_approved_task_object"
)
BLOCKER_SEMANTIC_OVERRIDE_MISSING = "isaac_semantic_override_layer_digest_missing"


class LiveHybridFrameError(ValueError):
    """Stable fail-closed hybrid frame materialization errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _load_json(path: Path, *, error: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LiveHybridFrameError([error]) from exc
    if not isinstance(value, dict):
        raise LiveHybridFrameError([error])
    return value


def _prefixed_file_digest(path: Path) -> str:
    return f"sha256:{sha256_file(Path(path))}"


def _require_file_digest(path: Path, expected: Any, *, error: str) -> str:
    resolved = Path(path)
    if not resolved.is_file():
        raise LiveHybridFrameError([error])
    observed = _prefixed_file_digest(resolved)
    if not isinstance(expected, str) or observed != expected:
        raise LiveHybridFrameError([error])
    return observed


def _quaternion_xyzw_to_rotation(quaternion: Sequence[float]) -> Any:
    import numpy as np

    values = [float(item) for item in quaternion]
    if len(values) != 4:
        raise LiveHybridFrameError(["hybrid_frame_isaac_quaternion_invalid"])
    norm = float(np.linalg.norm(values))
    if not norm > 0:
        raise LiveHybridFrameError(["hybrid_frame_isaac_quaternion_invalid"])
    x, y, z, w = (value / norm for value in values)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _verify_camera_pose_conversion(
    *, isaac_frame: Mapping[str, Any], calibration: Mapping[str, Any]
) -> dict[str, float]:
    """Recompute world_from_camera (OpenCV) from the raw Isaac OpenGL pose."""

    import numpy as np

    rotation_gl = _quaternion_xyzw_to_rotation(
        isaac_frame.get("quaternion_world_opengl_xyzw") or []
    )
    rotation_cv = rotation_gl @ np.diag([1.0, -1.0, -1.0])
    declared = np.asarray(calibration.get("world_from_camera") or [], dtype=float)
    if declared.shape != (4, 4):
        raise LiveHybridFrameError(["hybrid_frame_camera_pose_invalid"])
    translation = np.asarray(isaac_frame.get("position_world_m") or [], dtype=float)
    if translation.shape != (3,):
        raise LiveHybridFrameError(["hybrid_frame_camera_pose_invalid"])
    rotation_error = float(np.abs(rotation_cv - declared[:3, :3]).max())
    translation_error = float(np.abs(translation - declared[:3, 3]).max())
    intrinsic_error = float(
        np.abs(
            np.asarray(isaac_frame.get("intrinsic_matrix") or [], dtype=float)
            - np.asarray(calibration.get("intrinsic_matrix") or [], dtype=float)
        ).max()
    )
    if max(rotation_error, translation_error, intrinsic_error) > GL_TO_CV_MAX_ABS_ERROR:
        raise LiveHybridFrameError(["hybrid_frame_camera_pose_conversion_mismatch"])
    return {
        "rotation_max_abs_error": rotation_error,
        "translation_max_abs_error": translation_error,
        "intrinsic_max_abs_error": intrinsic_error,
    }


def _semantic_labels(isaac_frame: Mapping[str, Any]) -> dict[int, str]:
    mapping = (
        (isaac_frame.get("semantic_segmentation") or {}).get("id_to_labels") or {}
    ).get("idToLabels")
    if not isinstance(mapping, Mapping) or not mapping:
        raise LiveHybridFrameError(["hybrid_frame_semantic_labels_missing"])
    labels: dict[int, str] = {}
    for key, value in mapping.items():
        try:
            identifier = int(key)
        except (TypeError, ValueError) as exc:
            raise LiveHybridFrameError(["hybrid_frame_semantic_labels_missing"]) from exc
        label = value.get("class") if isinstance(value, Mapping) else value
        if not isinstance(label, str) or not label.strip():
            raise LiveHybridFrameError(["hybrid_frame_semantic_labels_missing"])
        labels[identifier] = label.strip()
    return labels


def _aura_artifact(row: Mapping[str, Any], suffix: str) -> Mapping[str, Any]:
    for artifact in row.get("artifacts") or []:
        if isinstance(artifact, Mapping) and str(artifact.get("path", "")).endswith(
            suffix
        ):
            return artifact
    raise LiveHybridFrameError(["hybrid_frame_aura_artifact_missing"])


def materialize_live_hybrid_observation_frames(
    *,
    aura_native_result_path: str | Path,
    isaac_native_result_path: str | Path,
    output_dir: str | Path,
    approved_task_object_label: str = APPROVED_TASK_OBJECT_LABEL,
    depth_epsilon_m: float = 1e-4,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Compose and retain one depth-correct hybrid observation per camera."""

    import numpy as np
    from PIL import Image

    aura_path = Path(aura_native_result_path)
    isaac_path = Path(isaac_native_result_path)
    aura_result = _load_json(aura_path, error="hybrid_frame_aura_result_unreadable")
    isaac_result = _load_json(isaac_path, error="hybrid_frame_isaac_result_unreadable")
    aura_root = aura_path.parent
    isaac_root = isaac_path.parent

    errors: list[str] = []
    if aura_result.get("schema_version") != AURA_RESULT_SCHEMA_VERSION:
        errors.append("hybrid_frame_aura_result_schema_invalid")
    if aura_result.get("status") != "completed" or aura_result.get("blockers"):
        errors.append("hybrid_frame_aura_result_not_completed")
    if aura_result.get("candidate_policy_queried") is not False:
        errors.append("hybrid_frame_candidate_policy_queried")
    if isaac_result.get("schema_version") != ISAAC_RESULT_SCHEMA_VERSION:
        errors.append("hybrid_frame_isaac_result_schema_invalid")
    if isaac_result.get("status") != "completed":
        errors.append("hybrid_frame_isaac_result_not_completed")
    if isaac_result.get("sealed_source_mutated") is not False:
        errors.append("hybrid_frame_sealed_source_mutated")
    if errors:
        raise LiveHybridFrameError(errors)

    isaac_frames = {
        str(frame.get("camera_id")): frame
        for frame in isaac_result.get("camera_frames") or []
        if isinstance(frame, Mapping)
    }
    aura_rows = [
        row for row in aura_result.get("camera_rows") or [] if isinstance(row, Mapping)
    ]
    if not aura_rows:
        raise LiveHybridFrameError(["hybrid_frame_aura_camera_rows_missing"])

    declared_override = isaac_result.get("semantic_override_layer_digest")
    blockers: list[str] = []
    if not isinstance(declared_override, str) or not declared_override.startswith(
        "sha256:"
    ):
        blockers.append(BLOCKER_SEMANTIC_OVERRIDE_MISSING)
        override_provenance = "derived_from_observed_isaac_id_to_labels"
    else:
        override_provenance = "declared_by_isaac_runtime"
        override_layer = isaac_result.get("semantic_override_layer")
        if isinstance(override_layer, Mapping):
            # When the runtime publishes the override body, the digest is
            # recomputed here; a declared digest is never taken on trust.
            if canonical_digest(dict(override_layer)) != declared_override:
                raise LiveHybridFrameError(
                    ["hybrid_frame_semantic_override_digest_mismatch"]
                )
            if override_layer.get("sealed_source_usd_mutated") is not False:
                raise LiveHybridFrameError(["hybrid_frame_sealed_source_mutated"])

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    camera_manifests: list[dict[str, Any]] = []

    for row in aura_rows:
        camera_id = str(row.get("camera_id"))
        isaac_frame = isaac_frames.get(camera_id)
        if isaac_frame is None:
            raise LiveHybridFrameError(["hybrid_frame_isaac_camera_missing"])
        if row.get("valid") is not True:
            raise LiveHybridFrameError(["hybrid_frame_aura_camera_row_invalid"])

        # --- temporal identity -------------------------------------------------
        if (
            row.get("source_isaac_frame_index") != isaac_frame.get("frame_index")
            or row.get("source_isaac_timestamp_ns") != isaac_frame.get("timestamp_ns")
            or row.get("source_isaac_sim_time_seconds")
            != isaac_frame.get("sim_time_seconds")
        ):
            raise LiveHybridFrameError(["hybrid_frame_temporal_identity_mismatch"])

        # --- calibration identity ---------------------------------------------
        calibration = row.get("calibration")
        if not isinstance(calibration, Mapping):
            raise LiveHybridFrameError(["hybrid_frame_camera_pose_invalid"])
        if canonical_digest(dict(calibration)) != row.get("calibration_digest"):
            raise LiveHybridFrameError(["hybrid_frame_calibration_digest_mismatch"])
        pose_check = _verify_camera_pose_conversion(
            isaac_frame=isaac_frame, calibration=calibration
        )

        # --- byte identity of every consumed input -----------------------------
        aura_rgb_artifact = _aura_artifact(row, "rgb.npy")
        aura_depth_artifact = _aura_artifact(row, "depth_m.npy")
        aura_rgb_path = aura_root / str(aura_rgb_artifact["path"])
        aura_depth_path = aura_root / str(aura_depth_artifact["path"])
        aura_rgb_digest = _require_file_digest(
            aura_rgb_path,
            aura_rgb_artifact.get("sha256"),
            error="hybrid_frame_aura_rgb_digest_mismatch",
        )
        aura_depth_digest = _require_file_digest(
            aura_depth_path,
            aura_depth_artifact.get("sha256"),
            error="hybrid_frame_aura_depth_digest_mismatch",
        )

        isaac_rgb = isaac_frame.get("rgb_png") or {}
        isaac_depth = isaac_frame.get("metric_depth") or {}
        isaac_semantic = isaac_frame.get("semantic_segmentation") or {}
        isaac_rgb_path = isaac_root / str(isaac_rgb.get("path"))
        isaac_depth_path = isaac_root / str(isaac_depth.get("path"))
        isaac_semantic_path = isaac_root / str(isaac_semantic.get("path"))
        isaac_rgb_digest = _require_file_digest(
            isaac_rgb_path,
            isaac_rgb.get("sha256"),
            error="hybrid_frame_isaac_rgb_digest_mismatch",
        )
        isaac_depth_digest = _require_file_digest(
            isaac_depth_path,
            isaac_depth.get("sha256"),
            error="hybrid_frame_isaac_depth_digest_mismatch",
        )
        isaac_semantic_digest = _require_file_digest(
            isaac_semantic_path,
            isaac_semantic.get("sha256"),
            error="hybrid_frame_isaac_semantic_digest_mismatch",
        )

        # The Aura render recorded which Isaac bytes it was posed against; that
        # binding must agree with the Isaac run itself.
        recorded = row.get("source_isaac_input_artifacts") or {}
        if not isinstance(recorded, Mapping) or {
            (recorded.get("dynamic_rgb") or {}).get("sha256"),
            (recorded.get("dynamic_depth") or {}).get("sha256"),
            (recorded.get("dynamic_semantic") or {}).get("sha256"),
        } != {isaac_rgb_digest, isaac_depth_digest, isaac_semantic_digest}:
            raise LiveHybridFrameError(["hybrid_frame_isaac_source_binding_mismatch"])

        # --- arrays ------------------------------------------------------------
        aura_rgb = np.load(aura_rgb_path)
        aura_depth = np.load(aura_depth_path)
        with Image.open(isaac_rgb_path) as handle:
            dynamic_rgb = np.asarray(handle.convert("RGB"))
        dynamic_depth = np.squeeze(np.load(isaac_depth_path))
        dynamic_segmentation = np.load(isaac_semantic_path)
        if dynamic_depth.ndim != 2 or dynamic_segmentation.ndim != 2:
            raise LiveHybridFrameError(["hybrid_frame_dynamic_array_shape_invalid"])
        resolution_hw = list(isaac_frame.get("resolution_hw") or [])
        if resolution_hw != list(aura_rgb.shape[:2]):
            raise LiveHybridFrameError(["hybrid_frame_resolution_mismatch"])

        labels = _semantic_labels(isaac_frame)
        semantic_override_digest = (
            declared_override
            if override_provenance == "declared_by_isaac_runtime"
            else canonical_digest({"idToLabels": {str(k): v for k, v in labels.items()}})
        )
        dynamic_alpha = (dynamic_segmentation > 0).astype(np.float32)

        try:
            composed, frame_receipt = compose_live_hybrid_observation(
                aura_rgb=aura_rgb,
                aura_depth_m=aura_depth,
                dynamic_rgb=dynamic_rgb,
                dynamic_depth_m=dynamic_depth,
                dynamic_segmentation=dynamic_segmentation,
                dynamic_alpha=dynamic_alpha,
                aura_calibration=calibration,
                isaac_calibration=calibration,
                timestamp_ns=int(isaac_frame["timestamp_ns"]),
                simulation_time_s=float(isaac_frame["sim_time_seconds"]),
                dynamic_depth_aov=str(isaac_depth.get("aov")),
                semantic_labels=labels,
                semantic_override_layer_digest=semantic_override_digest,
                aura_depth_source=AURA_DEPTH_SOURCE,
                depth_epsilon_m=depth_epsilon_m,
            )
        except LiveHybridObservationError as exc:
            raise LiveHybridFrameError(exc.errors) from exc

        # --- per-class occlusion accounting ------------------------------------
        ray_scale = np.sqrt(
            1.0
            + (
                (np.arange(aura_rgb.shape[1])[None, :] - calibration["intrinsic_matrix"][0][2])
                / calibration["intrinsic_matrix"][0][0]
            )
            ** 2
            + (
                (np.arange(aura_rgb.shape[0])[:, None] - calibration["intrinsic_matrix"][1][2])
                / calibration["intrinsic_matrix"][1][1]
            )
            ** 2
        ).astype(np.float32)
        dynamic_z = dynamic_depth.astype(np.float32) / ray_scale
        aura_z = aura_depth.astype(np.float32)
        visible = dynamic_segmentation > 0
        dynamic_valid = np.isfinite(dynamic_z) & (dynamic_z > 0)
        aura_valid = np.isfinite(aura_z) & (aura_z > 0)
        front = (
            visible
            & dynamic_valid
            & (~aura_valid | (dynamic_z <= aura_z + float(depth_epsilon_m)))
        )
        occluded = visible & dynamic_valid & ~front

        # World height of every pixel's dynamic surface, to separate legitimate
        # shelf-contact occlusion from occlusion of the object's body.
        pose_matrix = np.asarray(calibration["world_from_camera"], dtype=np.float64)
        grid_u, grid_v = np.meshgrid(
            np.arange(aura_rgb.shape[1], dtype=np.float64),
            np.arange(aura_rgb.shape[0], dtype=np.float64),
        )
        intrinsic = calibration["intrinsic_matrix"]
        # Background pixels carry infinite depth; project a finite placeholder so
        # no NaN enters the height comparison.  Those pixels are never dynamic,
        # so their height is never consulted.
        finite_depth = np.where(np.isfinite(dynamic_z), dynamic_z, 0.0).astype(
            np.float64
        )
        camera_points = np.stack(
            [
                (grid_u - intrinsic[0][2]) / intrinsic[0][0] * finite_depth,
                (grid_v - intrinsic[1][2]) / intrinsic[1][1] * finite_depth,
                finite_depth,
            ],
            axis=-1,
        )
        world_height = (
            camera_points @ pose_matrix[:3, :3].T + pose_matrix[:3, 3]
        )[..., 2]
        above_contact_band = world_height > (
            SUPPORT_HEIGHT_M + CONTACT_BAND_ABOVE_SUPPORT_M
        )

        class_rows: list[dict[str, Any]] = []
        approved_present = False
        approved_occluded = 0
        approved_body_occluded = 0
        approved_contact_occluded = 0
        approved_pixels = 0
        for identifier in sorted(int(v) for v in np.unique(dynamic_segmentation) if v > 0):
            mask = dynamic_segmentation == identifier
            label = labels.get(identifier, "")
            separation = (aura_z[mask] - dynamic_z[mask]).astype(np.float64)
            class_rows.append(
                {
                    "semantic_id": identifier,
                    "label": label,
                    "pixel_count": int(mask.sum()),
                    "front_pixel_count": int((mask & front).sum()),
                    "occluded_pixel_count": int((mask & occluded).sum()),
                    "median_aura_minus_dynamic_camera_z_m": float(
                        np.median(separation)
                    ),
                    "aura_in_front_fraction": float((separation < 0).mean()),
                }
            )
            if label == approved_task_object_label:
                approved_present = True
                approved_occluded = int((mask & occluded).sum())
                approved_pixels = int(mask.sum())
                approved_body_occluded = int(
                    (mask & occluded & above_contact_band).sum()
                )
                approved_contact_occluded = approved_occluded - approved_body_occluded

        camera_dir = output_root / camera_id
        camera_dir.mkdir(parents=True, exist_ok=True)
        composed_npy = camera_dir / "composed_rgb.npy"
        composed_png = camera_dir / "composed_rgb.png"
        occlusion_png = camera_dir / "occlusion_map.png"
        np.save(composed_npy, composed)
        Image.fromarray(composed).save(composed_png)
        overlay = composed.copy()
        overlay[front] = (0, 220, 0)
        overlay[occluded] = (230, 0, 0)
        Image.fromarray(overlay).save(occlusion_png)

        body_occlusion_fraction = approved_body_occluded / max(approved_pixels, 1)
        camera_blockers: list[str] = []
        if not approved_present:
            camera_blockers.append(BLOCKER_APPROVED_OBJECT_ABSENT)
        elif body_occlusion_fraction > MAX_BODY_OCCLUSION_FRACTION:
            camera_blockers.append(BLOCKER_APPROVED_OBJECT_OCCLUDED)
        blockers.extend(camera_blockers)

        camera_manifests.append(
            {
                "camera_id": camera_id,
                "blockers": sorted(set(camera_blockers)),
                "frame_receipt": frame_receipt,
                "camera_pose_conversion_check": pose_check,
                "semantic_override_layer_digest": semantic_override_digest,
                "semantic_override_layer_provenance": override_provenance,
                "source_inputs": {
                    "aura_rgb_sha256": aura_rgb_digest,
                    "aura_depth_sha256": aura_depth_digest,
                    "isaac_rgb_sha256": isaac_rgb_digest,
                    "isaac_metric_depth_sha256": isaac_depth_digest,
                    "isaac_semantic_sha256": isaac_semantic_digest,
                },
                "retained_outputs": {
                    "composed_rgb_npy": {
                        "path": str(composed_npy.relative_to(output_root)),
                        "sha256": _prefixed_file_digest(composed_npy),
                    },
                    "composed_rgb_png": {
                        "path": str(composed_png.relative_to(output_root)),
                        "sha256": _prefixed_file_digest(composed_png),
                    },
                    "occlusion_map_png": {
                        "path": str(occlusion_png.relative_to(output_root)),
                        "sha256": _prefixed_file_digest(occlusion_png),
                    },
                },
                "semantic_classes": class_rows,
                "approved_task_object_present": approved_present,
                "approved_task_object_occluded_pixel_count": approved_occluded,
                "approved_task_object_body_occluded_pixel_count": approved_body_occluded,
                "approved_task_object_contact_occluded_pixel_count": (
                    approved_contact_occluded
                ),
                "approved_task_object_body_occlusion_fraction": body_occlusion_fraction,
                "max_body_occlusion_fraction": MAX_BODY_OCCLUSION_FRACTION,
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": FRAME_MANIFEST_SCHEMA_VERSION,
        "status": "composed_from_retained_live_arrays",
        "admission_status": "blocked" if blockers else "frame_composition_only",
        "blockers": sorted(set(blockers)),
        "generated_at": generated_at,
        "aura_native_result_sha256": _prefixed_file_digest(aura_path),
        "isaac_native_result_sha256": _prefixed_file_digest(isaac_path),
        "aura_source_probe_manifest_digest": aura_result.get(
            "source_probe_manifest_digest"
        ),
        "aura_implementation_commit": aura_result.get("implementation_commit"),
        "approved_task_object_label": approved_task_object_label,
        "depth_epsilon_m": float(depth_epsilon_m),
        "cameras": camera_manifests,
        "candidate_policy_queried": False,
        "visual_judgment_used_for_success": False,
        "claim_ceiling": (
            "development_only single-frame depth-correct hybrid composition over "
            "retained arrays; establishes neither a live co-resident renderer, "
            "moving-occlusion behaviour, nor any policy or task outcome"
        ),
    }
    manifest["manifest_digest"] = canonical_digest(
        manifest, digest_field="manifest_digest"
    )
    write_json(output_root / "adp009d_live_hybrid_frame_manifest.v1.json", manifest)
    return manifest


__all__ = [
    "APPROVED_TASK_OBJECT_LABEL",
    "BLOCKER_APPROVED_OBJECT_ABSENT",
    "BLOCKER_APPROVED_OBJECT_OCCLUDED",
    "BLOCKER_SEMANTIC_OVERRIDE_MISSING",
    "FRAME_MANIFEST_SCHEMA_VERSION",
    "LiveHybridFrameError",
    "materialize_live_hybrid_observation_frames",
]
