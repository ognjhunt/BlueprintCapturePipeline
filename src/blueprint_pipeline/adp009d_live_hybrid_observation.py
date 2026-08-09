"""Depth-correct Aura/Isaac policy-observation composition contracts.

This module is deliberately renderer-neutral.  A native worker must render the
sealed Aura representation and the live Isaac dynamic layer from the same
camera, then pass the observed arrays through this composer.  Validation of a
prepared config or a synthetic array never proves the live renderer gate.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .adp009d_aura_renderer_conformance import (
    AuraRendererConformanceError,
    validate_aura_renderer_conformance_receipt,
)
from .adp009d_aura_native_conformance import (
    AuraNativeConformanceError,
    validate_aura_native_conformance_receipt,
)
from .decision_evidence_contracts import canonical_digest


HYBRID_FRAME_RECEIPT_SCHEMA_VERSION = "adp009d_live_hybrid_frame_receipt.v1"
HYBRID_RUNTIME_RECEIPT_SCHEMA_VERSION = "adp009d_live_hybrid_runtime_receipt.v1"
MISSING_RENDERER_BLOCKER = "sealed_aura_hybrid_policy_observation_renderer_missing"
METRIC_DEPTH_AOVS = {"DistanceToCameraSD", "DistanceToImagePlaneSD"}
ISAAC_CAMERA_BACKEND = "Isaac Lab Camera over Isaac Sim RTX"
ISAAC_CAMERA_METRIC_DEPTH_AOV = "distance_to_camera"

# Metric depth is published in two different, non-interchangeable geometries.
# ``ray_length`` measures along the pixel ray from the camera centre;
# ``camera_z`` measures along the optical axis to the image plane.  They differ
# by 1/cos(theta), which reaches 1.78x at the corners of a 1280x720 frame with
# the admitted DROID intrinsics, so comparing them directly silently fabricates
# occlusion.  Every layer therefore declares its geometry and is normalised to
# ``camera_z`` before any depth ordering is computed.
DEPTH_CONVENTION_CAMERA_Z = "camera_z"
DEPTH_CONVENTION_RAY_LENGTH = "ray_length"
DEPTH_COMPARISON_CONVENTION = DEPTH_CONVENTION_CAMERA_Z
DYNAMIC_DEPTH_AOV_CONVENTIONS = {
    "DistanceToCameraSD": DEPTH_CONVENTION_RAY_LENGTH,
    "DistanceToImagePlaneSD": DEPTH_CONVENTION_CAMERA_Z,
    ISAAC_CAMERA_METRIC_DEPTH_AOV: DEPTH_CONVENTION_RAY_LENGTH,
}
AURA_DEPTH_SOURCE_CONVENTIONS = {
    "aurafusion360_metric_camera_depth_m": DEPTH_CONVENTION_CAMERA_Z,
    "surf_depth_expected_camera_z_m": DEPTH_CONVENTION_CAMERA_Z,
}
COMPOSITION_METRIC_DEPTH_AOVS = frozenset(DYNAMIC_DEPTH_AOV_CONVENTIONS)


class LiveHybridObservationError(ValueError):
    """Stable fail-closed hybrid-renderer contract errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _json_mapping(value: Any, *, error: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LiveHybridObservationError([error])
    try:
        cloned = json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise LiveHybridObservationError([error]) from exc
    if not isinstance(cloned, dict):
        raise LiveHybridObservationError([error])
    return cloned


def _array_digest(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return "sha256:" + digest.hexdigest()


def _calibration_digest(value: Mapping[str, Any]) -> str:
    calibration = _json_mapping(value, error="hybrid_camera_calibration_invalid")
    intrinsic = calibration.get("intrinsic_matrix")
    world_from_camera = calibration.get("world_from_camera")
    resolution = calibration.get("resolution")
    if not (
        calibration.get("camera_model") == "pinhole"
        and isinstance(intrinsic, list)
        and len(intrinsic) == 3
        and all(isinstance(row, list) and len(row) == 3 for row in intrinsic)
        and isinstance(world_from_camera, list)
        and len(world_from_camera) == 4
        and all(isinstance(row, list) and len(row) == 4 for row in world_from_camera)
        and isinstance(resolution, list)
        and len(resolution) == 2
        and all(isinstance(item, int) and item > 0 for item in resolution)
    ):
        raise LiveHybridObservationError(["hybrid_camera_calibration_invalid"])
    numbers = [item for row in intrinsic for item in row]
    numbers.extend(item for row in world_from_camera for item in row)
    try:
        if not all(math.isfinite(float(item)) for item in numbers):
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise LiveHybridObservationError(["hybrid_camera_calibration_invalid"]) from exc
    return canonical_digest(calibration)


def camera_ray_length_scale(
    intrinsic_matrix: Sequence[Sequence[float]], plane_shape: tuple[int, int]
) -> Any:
    """Per-pixel ``ray_length / camera_z`` factor for a pinhole camera."""

    import numpy as np

    fx = float(intrinsic_matrix[0][0])
    fy = float(intrinsic_matrix[1][1])
    cx = float(intrinsic_matrix[0][2])
    cy = float(intrinsic_matrix[1][2])
    if not (fx > 0 and fy > 0 and math.isfinite(cx) and math.isfinite(cy)):
        raise LiveHybridObservationError(["hybrid_camera_intrinsics_invalid"])
    height, width = int(plane_shape[0]), int(plane_shape[1])
    columns = np.arange(width, dtype=np.float64)[None, :]
    rows = np.arange(height, dtype=np.float64)[:, None]
    scale = np.sqrt(
        1.0 + ((columns - cx) / fx) ** 2 + ((rows - cy) / fy) ** 2
    )
    return scale.astype(np.float32)


def compose_live_hybrid_observation(
    *,
    aura_rgb: Any,
    aura_depth_m: Any,
    dynamic_rgb: Any,
    dynamic_depth_m: Any,
    dynamic_segmentation: Any,
    dynamic_alpha: Any,
    aura_calibration: Mapping[str, Any],
    isaac_calibration: Mapping[str, Any],
    timestamp_ns: int,
    simulation_time_s: float,
    dynamic_depth_aov: str,
    semantic_labels: Mapping[int | str, str],
    semantic_override_layer_digest: str,
    aura_depth_source: str = "aurafusion360_metric_camera_depth_m",
    depth_epsilon_m: float = 1e-4,
) -> tuple[Any, dict[str, Any]]:
    """Compose one exact live camera observation using metric depth ordering."""

    import numpy as np

    aura = np.asarray(aura_rgb)
    aura_depth = np.asarray(aura_depth_m)
    dynamic = np.asarray(dynamic_rgb)
    dynamic_depth = np.asarray(dynamic_depth_m)
    segmentation = np.asarray(dynamic_segmentation)
    alpha = np.asarray(dynamic_alpha)
    errors: list[str] = []
    if aura.dtype != np.uint8 or dynamic.dtype != np.uint8:
        errors.append("hybrid_rgb_dtype_not_uint8")
    if aura.ndim != 3 or aura.shape[-1:] != (3,) or dynamic.shape != aura.shape:
        errors.append("hybrid_rgb_shape_invalid")
    expected_plane = aura.shape[:2] if aura.ndim == 3 else ()
    for name, array in (
        ("aura_depth", aura_depth),
        ("dynamic_depth", dynamic_depth),
        ("dynamic_segmentation", segmentation),
        ("dynamic_alpha", alpha),
    ):
        if array.shape != expected_plane:
            errors.append(f"hybrid_{name}_shape_invalid")
    dynamic_depth_convention = DYNAMIC_DEPTH_AOV_CONVENTIONS.get(dynamic_depth_aov)
    if dynamic_depth_convention is None:
        errors.append("hybrid_metric_dynamic_depth_aov_required")
    aura_depth_convention = AURA_DEPTH_SOURCE_CONVENTIONS.get(aura_depth_source)
    if aura_depth_convention is None:
        errors.append("hybrid_metric_aura_depth_required")
    if not isinstance(timestamp_ns, int) or timestamp_ns < 0:
        errors.append("hybrid_timestamp_invalid")
    if not isinstance(simulation_time_s, (int, float)) or not math.isfinite(
        float(simulation_time_s)
    ):
        errors.append("hybrid_simulation_time_invalid")
    if not isinstance(depth_epsilon_m, (int, float)) or not 0 <= float(
        depth_epsilon_m
    ) <= 0.01:
        errors.append("hybrid_depth_epsilon_invalid")
    if not isinstance(semantic_override_layer_digest, str) or not (
        semantic_override_layer_digest.startswith("sha256:")
        and len(semantic_override_layer_digest) == 71
    ):
        errors.append("hybrid_semantic_override_layer_digest_invalid")

    try:
        aura_calibration_digest = _calibration_digest(aura_calibration)
        isaac_calibration_digest = _calibration_digest(isaac_calibration)
        if aura_calibration_digest != isaac_calibration_digest:
            errors.append("hybrid_camera_calibration_mismatch")
        resolution = list(aura_calibration.get("resolution") or [])
        if expected_plane and resolution != [expected_plane[1], expected_plane[0]]:
            errors.append("hybrid_camera_resolution_mismatch")
    except LiveHybridObservationError as exc:
        errors.extend(exc.errors)
        aura_calibration_digest = ""
        isaac_calibration_digest = ""

    normalized_labels = {
        int(key): str(value).strip()
        for key, value in semantic_labels.items()
        if str(key).lstrip("-").isdigit() and str(value).strip()
    }
    positive_ids = set(int(item) for item in np.unique(segmentation) if int(item) > 0)
    if not positive_ids:
        errors.append("hybrid_dynamic_semantic_pixels_missing")
    if not positive_ids.issubset(normalized_labels):
        errors.append("hybrid_dynamic_semantic_labels_incomplete")
    if errors:
        raise LiveHybridObservationError(errors)

    alpha_float = alpha.astype(np.float32, copy=False)
    if not np.isfinite(alpha_float).all() or (alpha_float < 0).any() or (
        alpha_float > 1
    ).any():
        raise LiveHybridObservationError(["hybrid_dynamic_alpha_invalid"])
    # Normalise both layers to camera_z before any ordering decision.
    ray_scale = camera_ray_length_scale(
        aura_calibration["intrinsic_matrix"], expected_plane
    )
    aura_depth_float = aura_depth.astype(np.float32, copy=False)
    dynamic_depth_float = dynamic_depth.astype(np.float32, copy=False)
    if aura_depth_convention == DEPTH_CONVENTION_RAY_LENGTH:
        aura_depth_float = aura_depth_float / ray_scale
    if dynamic_depth_convention == DEPTH_CONVENTION_RAY_LENGTH:
        dynamic_depth_float = dynamic_depth_float / ray_scale
    dynamic_semantic = segmentation > 0
    dynamic_visible = dynamic_semantic & (alpha_float > 0)
    dynamic_depth_valid = np.isfinite(dynamic_depth_float) & (dynamic_depth_float > 0)
    if np.any(dynamic_visible & ~dynamic_depth_valid):
        raise LiveHybridObservationError(["hybrid_dynamic_metric_depth_missing"])
    aura_depth_valid = np.isfinite(aura_depth_float) & (aura_depth_float > 0)
    dynamic_front = dynamic_visible & dynamic_depth_valid & (
        ~aura_depth_valid
        | (dynamic_depth_float <= aura_depth_float + float(depth_epsilon_m))
    )
    blend_alpha = np.where(dynamic_front, alpha_float, 0.0)[..., None]
    composed_float = (
        dynamic.astype(np.float32) * blend_alpha
        + aura.astype(np.float32) * (1.0 - blend_alpha)
    )
    composed = np.clip(np.rint(composed_float), 0, 255).astype(np.uint8)
    receipt: dict[str, Any] = {
        "schema_version": HYBRID_FRAME_RECEIPT_SCHEMA_VERSION,
        "status": "composed_from_live_arrays",
        "timestamp_ns": timestamp_ns,
        "simulation_time_s": float(simulation_time_s),
        "resolution": [expected_plane[1], expected_plane[0]],
        "camera_calibration_digest": aura_calibration_digest,
        "aura_calibration_digest": aura_calibration_digest,
        "isaac_calibration_digest": isaac_calibration_digest,
        "dynamic_depth_aov": dynamic_depth_aov,
        "aura_depth_source": aura_depth_source,
        "dynamic_depth_convention": dynamic_depth_convention,
        "aura_depth_convention": aura_depth_convention,
        "depth_comparison_convention": DEPTH_COMPARISON_CONVENTION,
        "ray_length_scale_max": float(ray_scale.max()),
        "depth_epsilon_m": float(depth_epsilon_m),
        "semantic_override_layer_digest": semantic_override_layer_digest,
        "semantic_labels": {str(key): normalized_labels[key] for key in sorted(positive_ids)},
        "input_array_digests": {
            "aura_rgb": _array_digest(aura),
            "aura_depth_m": _array_digest(aura_depth),
            "dynamic_rgb": _array_digest(dynamic),
            "dynamic_depth_m": _array_digest(dynamic_depth),
            "dynamic_segmentation": _array_digest(segmentation),
            "dynamic_alpha": _array_digest(alpha),
        },
        "composed_rgb_digest": _array_digest(composed),
        "dynamic_semantic_pixel_count": int(dynamic_semantic.sum()),
        "dynamic_front_pixel_count": int(dynamic_front.sum()),
        "dynamic_occluded_pixel_count": int((dynamic_visible & ~dynamic_front).sum()),
        "composition_authority": "metric_depth_and_segmentation",
        "visual_judgment_used_for_success": False,
        "live_execution_proven_by_this_function": False,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return composed, receipt


def validate_live_hybrid_runtime_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate observed live renderer evidence; reject config-only receipts."""

    receipt = _json_mapping(value, error="hybrid_runtime_receipt_invalid")
    errors: list[str] = []
    if receipt.get("schema_version") != HYBRID_RUNTIME_RECEIPT_SCHEMA_VERSION:
        errors.append("hybrid_runtime_schema_invalid")
    if receipt.get("status") != "executed_live_renderer_microcheck":
        errors.append(MISSING_RENDERER_BLOCKER)
    backend = receipt.get("backend")
    if backend not in {"OVRTX", ISAAC_CAMERA_BACKEND}:
        errors.append("hybrid_runtime_backend_invalid")
    if receipt.get("unitless_depth_sd_used") is not False:
        errors.append("hybrid_runtime_unitless_depth_forbidden")
    if backend == "OVRTX":
        if receipt.get("initialization_order") != ["OVRTX", "OvPhysX"]:
            errors.append("hybrid_runtime_initialization_order_invalid")
        if receipt.get("render_settings_target") != "RenderProduct":
            errors.append("hybrid_runtime_render_settings_target_invalid")
        if receipt.get("metric_depth_aov") not in METRIC_DEPTH_AOVS:
            errors.append("hybrid_runtime_metric_depth_aov_invalid")
        if receipt.get("attached_mode_ordinals_respected") is not True:
            errors.append("hybrid_runtime_attached_ordinals_missing")
        if receipt.get("write_floors_respected") is not True:
            errors.append("hybrid_runtime_write_floors_missing")
        if receipt.get("dlpack_ownership_explicit") is not True:
            errors.append("hybrid_runtime_dlpack_ownership_missing")
        if receipt.get("map_unmap_balanced") is not True:
            errors.append("hybrid_runtime_map_unmap_unbalanced")
        warmup = receipt.get("rtpt_warmup_frames")
        if not isinstance(warmup, int) or warmup < 40:
            errors.append("hybrid_runtime_rtpt_warmup_below_documented_default")
        if warmup != 40 and not str(receipt.get("rtpt_warmup_change_reason") or ""):
            errors.append("hybrid_runtime_rtpt_warmup_change_unrecorded")
    elif backend == ISAAC_CAMERA_BACKEND:
        if receipt.get("metric_depth_aov") != ISAAC_CAMERA_METRIC_DEPTH_AOV:
            errors.append("hybrid_runtime_metric_depth_aov_invalid")
        if set(receipt.get("camera_data_types") or []) != {
            "rgb",
            ISAAC_CAMERA_METRIC_DEPTH_AOV,
            "semantic_segmentation",
        }:
            errors.append("hybrid_runtime_isaac_camera_data_types_invalid")
        if receipt.get("camera_calibration_retained") is not True:
            errors.append("hybrid_runtime_camera_calibration_missing")
        if receipt.get("camera_timestamps_retained") is not True:
            errors.append("hybrid_runtime_camera_timestamps_missing")
        warmup = receipt.get("camera_warmup_frames")
        if not isinstance(warmup, int) or warmup <= 0:
            errors.append("hybrid_runtime_camera_warmup_missing")
    if receipt.get("semantic_source_usd_mutated") is not False:
        errors.append("hybrid_runtime_sealed_source_mutated")
    if receipt.get("semantic_override_layer_composed") is not True:
        errors.append("hybrid_runtime_semantic_override_missing")
    if receipt.get("camera_or_settings_change_reset") is not True:
        errors.append("hybrid_runtime_reset_after_change_missing")
    if receipt.get("device_synchronization_explicit") is not True:
        errors.append("hybrid_runtime_device_synchronization_missing")
    if receipt.get("path_tracing_used") is True:
        spp = receipt.get("path_tracing_samples_per_pixel")
        if not isinstance(spp, int) or spp <= 0:
            errors.append("hybrid_runtime_path_tracing_spp_missing")
    elif receipt.get("path_tracing_samples_per_pixel") not in {None, 0}:
        errors.append("hybrid_runtime_path_tracing_spp_inconsistent")
    if set(receipt.get("camera_ids") or []) != {"external", "wrist"}:
        errors.append("hybrid_runtime_camera_set_invalid")
    if not isinstance(receipt.get("observed_frame_count"), int) or int(
        receipt.get("observed_frame_count") or 0
    ) <= 0:
        errors.append("hybrid_runtime_observed_frames_missing")
    if not isinstance(receipt.get("frame_receipt_digests"), list) or not receipt.get(
        "frame_receipt_digests"
    ):
        errors.append("hybrid_runtime_frame_receipts_missing")
    if receipt.get("policy_frames_retained_losslessly") is not True:
        errors.append("hybrid_runtime_lossless_frames_missing")
    dynamic_appearance = receipt.get("dynamic_usd_render_appearance")
    if not isinstance(dynamic_appearance, Mapping):
        errors.append("hybrid_runtime_dynamic_usd_render_appearance_missing")
    else:
        appearance_digest = str(
            dynamic_appearance.get("static_appearance_receipt_digest") or ""
        )
        if not (
            len(appearance_digest) == 71
            and appearance_digest.startswith("sha256:")
        ):
            errors.append("hybrid_runtime_dynamic_usd_appearance_unchained")
        if dynamic_appearance.get("usd_materials_rendered") is not True:
            errors.append("hybrid_runtime_dynamic_usd_materials_not_rendered")
        if dynamic_appearance.get("default_neutral_override_used") is not False:
            errors.append("hybrid_runtime_default_neutral_material_forbidden")
        if dynamic_appearance.get("coverage_silhouette_audit_used") is not False:
            errors.append("hybrid_runtime_coverage_audit_as_policy_input_forbidden")
        if dynamic_appearance.get("native_renderer_readback_observed") is not True:
            errors.append("hybrid_runtime_dynamic_material_readback_missing")
    conformance = receipt.get("aura_renderer_conformance_receipt")
    conformance_valid = False
    try:
        validate_aura_renderer_conformance_receipt(conformance)
        conformance_valid = True
    except AuraRendererConformanceError:
        try:
            validate_aura_native_conformance_receipt(conformance)
            conformance_valid = True
        except AuraNativeConformanceError:
            pass
    if not conformance_valid:
        errors.append("hybrid_runtime_aura_exact_camera_conformance_missing")
    # Any run whose appearance layer is not the sealed asset must say so, and
    # must chain that derivative back to the sealed asset.  Without this a future
    # run could quietly render a modified appearance and still look admissible.
    provenance = receipt.get("aura_appearance_provenance")
    if not isinstance(provenance, Mapping):
        errors.append("hybrid_runtime_aura_appearance_provenance_missing")
    else:
        exclusion_applied = provenance.get("task_volume_exclusion_applied")
        if exclusion_applied not in {True, False}:
            errors.append("hybrid_runtime_aura_appearance_provenance_missing")
        elif exclusion_applied is True:
            if not str(provenance.get("sealed_source_ply_sha256") or "").startswith(
                "sha256:"
            ):
                errors.append("hybrid_runtime_appearance_derivative_unchained")
            if not str(provenance.get("exclusion_receipt_digest") or "").startswith(
                "sha256:"
            ):
                errors.append("hybrid_runtime_appearance_derivative_unchained")
            if not str(provenance.get("claim_ceiling") or "").strip():
                errors.append("hybrid_runtime_appearance_derivative_claim_missing")
    if receipt.get("camera_motion_occlusion_probe_passed") is not True:
        errors.append("hybrid_runtime_camera_motion_occlusion_probe_failed")
    if receipt.get("static_occlusion_probe_passed") is not True:
        errors.append("hybrid_runtime_static_occlusion_probe_failed")
    if receipt.get("moving_occlusion_probe_passed") is not True:
        errors.append("hybrid_runtime_moving_occlusion_probe_failed")
    if receipt.get("receipt_digest") != canonical_digest(
        receipt, digest_field="receipt_digest"
    ):
        errors.append("hybrid_runtime_receipt_digest_mismatch")
    if errors:
        raise LiveHybridObservationError(errors)
    return receipt


__all__ = [
    "AURA_DEPTH_SOURCE_CONVENTIONS",
    "COMPOSITION_METRIC_DEPTH_AOVS",
    "DEPTH_COMPARISON_CONVENTION",
    "DEPTH_CONVENTION_CAMERA_Z",
    "DEPTH_CONVENTION_RAY_LENGTH",
    "DYNAMIC_DEPTH_AOV_CONVENTIONS",
    "HYBRID_FRAME_RECEIPT_SCHEMA_VERSION",
    "HYBRID_RUNTIME_RECEIPT_SCHEMA_VERSION",
    "LiveHybridObservationError",
    "ISAAC_CAMERA_BACKEND",
    "ISAAC_CAMERA_METRIC_DEPTH_AOV",
    "METRIC_DEPTH_AOVS",
    "MISSING_RENDERER_BLOCKER",
    "camera_ray_length_scale",
    "compose_live_hybrid_observation",
    "validate_live_hybrid_runtime_receipt",
]
