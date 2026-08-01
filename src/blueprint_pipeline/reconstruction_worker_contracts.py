"""Fail-closed contracts for the headless pose and Gaussian worker.

These contracts deliberately stop at candidate artifacts.  A worker may emit a
trajectory or an appearance asset, but it cannot evaluate hidden observations,
change the frozen split, or grant any reconstruction qualification.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


WORKER_STACK_SCHEMA_VERSION = "reconstruction_worker_stack_manifest.v1"
WORKER_BUILD_RECEIPT_SCHEMA_VERSION = "reconstruction_worker_build_receipt.v1"
WORKER_SMOKE_RECEIPT_SCHEMA_VERSION = "reconstruction_worker_smoke_test_receipt.v1"
POSE_REQUEST_SCHEMA_VERSION = "pose_estimation_request.v1"
POSE_RESULT_SCHEMA_VERSION = "pose_estimation_result.v1"
TRAINING_REQUEST_SCHEMA_VERSION = "reconstruction_training_request.v1"
TRAINING_RESULT_SCHEMA_VERSION = "reconstruction_training_result.v1"
CHECKPOINT_SCHEMA_VERSION = "reconstruction_checkpoint_manifest.v1"
REQUIREMENTS_LOCK_SHA256 = "e35adb73bfc0ddd1696a5207deca496559ca25a8c5b076b657a36a901f1a795e"

PINNED_WORKER_COMPONENTS: tuple[dict[str, Any], ...] = (
    {
        "component_id": "linux_base",
        "name": "nvidia/cuda",
        "version": "12.4.1-devel-ubuntu22.04",
        "source_url": "https://hub.docker.com/r/nvidia/cuda",
        "source_revision": "sha256:da6791294b0b04d7e65d87b7451d6f2390b4d36225ab0701ee7dfec5769829f5",
        "linux_amd64_digest": "sha256:5645fec64549cc35930eee9d85aafd2b0006c0c3f22632be5a1d85e2604e9749",
        "license": "NVIDIA Deep Learning Container License",
        "redistribution": "review_required_before_distribution",
    },
    {
        "component_id": "nvidia_driver_contract",
        "name": "NVIDIA Linux Driver",
        "version": ">=550.54,<580",
        "source_url": "https://docs.nvidia.com/deploy/cuda-compatibility/",
        "source_revision": "runtime_compatibility_contract_v1",
        "license": "host_runtime_not_redistributed",
        "redistribution": "not_embedded",
    },
    {
        "component_id": "compiler_toolchain",
        "name": "GCC/CMake/Ninja",
        "version": "gcc-11.4.0;cmake-3.28.3;ninja-1.11.1",
        "source_url": "https://gcc.gnu.org/;https://cmake.org/;https://ninja-build.org/",
        "source_revision": "candidate_lock_v1",
        "license": "GPL-3.0-with-GCC-exception;BSD-3-Clause;Apache-2.0",
        "redistribution": "review_required_before_distribution",
    },
    {
        "component_id": "ffmpeg",
        "name": "FFmpeg/ffprobe",
        "version": "6.1.1",
        "source_url": "https://ffmpeg.org/releases/ffmpeg-6.1.1.tar.xz",
        "source_revision": "n6.1.1;source-sha256:8684f4b00f94b85461884c3719382f1261f0d9eb3d59640a1f4ac0873616f968",
        "license": "LGPL-2.1-or-later;configuration_dependent",
        "redistribution": "build_configuration_license_review_required",
    },
    {
        "component_id": "colmap",
        "name": "COLMAP",
        "version": "4.0.4",
        "source_url": "https://github.com/colmap/colmap",
        "source_revision": "9c23f6942fe69962e06030905e77067c8673382f",
        "license": "BSD-3-Clause",
        "redistribution": "dependency_license_review_required",
        "build_options": {
            "CUDA_ENABLED": True,
            "ONNX_ENABLED": True,
            "FETCH_ONNX": True,
            "GUI_ENABLED": False,
            "CMAKE_CUDA_ARCHITECTURES": [75, 80, 86, 89],
        },
    },
    {
        "component_id": "onnxruntime",
        "name": "ONNX Runtime GPU",
        "version": "1.24.4",
        "source_url": "https://github.com/microsoft/onnxruntime",
        "source_revision": "sha256:c5f804ff5d239b436fa59e9f2fb288a39f7eb9552f6a636c8b71e792e91a8808",
        "license": "MIT",
        "redistribution": "permitted_with_notices",
    },
    {
        "component_id": "gsplat",
        "name": "gsplat",
        "version": "1.5.3",
        "source_url": "https://github.com/nerfstudio-project/gsplat",
        "source_revision": "937e29912570c372bed6747a5c9bf85fed877bae",
        "license": "Apache-2.0",
        "redistribution": "permitted_with_notices",
    },
    {
        "component_id": "python_ml_runtime",
        "name": "Python/PyTorch/NumPy/OpenCV/Trimesh",
        "version": "python-3.11.9;torch-2.4.1+cu124;numpy-1.26.4;opencv-python-headless-4.10.0.84;trimesh-4.4.9",
        "source_url": "https://www.python.org/;https://pytorch.org/;https://pypi.org/",
        "source_revision": (
            "python-3.11.9-source-sha256:"
            "9b1e896523fc510691126c864406d9360a3d1e986acbda59cda57b5abda45b87;"
            f"requirements-lock-sha256:{REQUIREMENTS_LOCK_SHA256}"
        ),
        "license": "PSF-2.0;BSD-3-Clause;BSD-3-Clause;Apache-2.0;MIT",
        "redistribution": "wheel_hash_lock_and_notice_bundle_required",
    },
    {
        "component_id": "openusd",
        "name": "OpenUSD usd-core",
        "version": "26.3",
        "source_url": "https://github.com/PixarAnimationStudios/OpenUSD",
        "source_revision": "v26.03",
        "license": "Apache-2.0",
        "redistribution": "permitted_with_notices",
    },
    {
        "component_id": "threedgrut",
        "name": "NVIDIA 3DGRUT",
        "version": "1.1.0",
        "source_url": "https://github.com/nv-tlabs/3DGRUT",
        "source_revision": "0a5832248698ab8456b181d6ea17fe02eda58637",
        "license": "Apache-2.0",
        "redistribution": "dependency_and_model_review_required",
    },
    {
        "component_id": "deterministic_qa",
        "name": "image/depth QA runtime",
        "version": "scikit-image-0.24.0;imageio-2.35.1;lpips-0.1.4",
        "source_url": "https://pypi.org/",
        "source_revision": f"requirements-lock-sha256:{REQUIREMENTS_LOCK_SHA256}",
        "license": "BSD-3-Clause;BSD-2-Clause;BSD-2-Clause",
        "redistribution": "wheel_hash_lock_and_notice_bundle_required",
    },
)

PINNED_MODEL_ASSETS: tuple[dict[str, Any], ...] = (
    {
        "model_id": "colmap-aliked-n16rot-3.13.0",
        "url": "https://github.com/colmap/colmap/releases/download/3.13.0/aliked-n16rot.onnx",
        "digest": "sha256:39c423d0a6f03d39ec89d3d1d61853765c2fb6a8b8381376c703e5758778a547",
        "license": "COLMAP release asset;redistribution_review_required",
    },
    {
        "model_id": "colmap-aliked-lightglue-3.13.0",
        "url": "https://github.com/colmap/colmap/releases/download/3.13.0/aliked-lightglue.onnx",
        "digest": "sha256:b9a5de7204648b18a8cf5dcac819f9d30de1a5961ef03756803c8b86c2dceb8d",
        "license": "COLMAP release asset;redistribution_review_required",
    },
    {
        "model_id": "colmap-sift-lightglue-3.13.0",
        "url": "https://github.com/colmap/colmap/releases/download/3.13.0/sift-lightglue.onnx",
        "digest": "sha256:e0500228472b43f92b3d36881a09b3310d3b058b56187b246cc7b9ab6429096e",
        "license": "COLMAP release asset;redistribution_review_required",
    },
    {
        "model_id": "colmap-bruteforce-matcher-3.13.0",
        "url": "https://github.com/colmap/colmap/releases/download/3.13.0/bruteforce-matcher.onnx",
        "digest": "sha256:3c1282f96d83f5ffc861a873298d08bbe5219f59af59223f5ceab5c41a182a47",
        "license": "COLMAP release asset;redistribution_review_required",
    },
)

POSE_METHODS = {
    "colmap_sift_bruteforce_v1": ("SIFT", "SIFT_BRUTEFORCE"),
    "colmap_sift_lightglue_v1": ("SIFT", "SIFT_LIGHTGLUE"),
    "colmap_aliked_bruteforce_v1": ("ALIKED_N16ROT", "ALIKED_BRUTEFORCE"),
    "colmap_aliked_lightglue_v1": ("ALIKED_N16ROT", "ALIKED_LIGHTGLUE"),
}

CAMERA_MODELS = {
    "PINHOLE",
    "SIMPLE_PINHOLE",
    "OPENCV",
    "FULL_OPENCV",
    "OPENCV_FISHEYE",
    "FOV",
    "RAD_TAN_THIN_PRISM_FISHEYE",
    "EQUIRECTANGULAR",
}

TRAINER_METHODS = {
    "gsplat_3dgs_mcmc_v1",
    "gsplat_3dgut_mcmc_v1",
    "nvidia_3dgrut_3dgut_mcmc_v1",
}

FAILURE_CODES = {
    "invalid_capture_contract",
    "missing_rights_or_consent",
    "missing_retained_media",
    "invalid_pts_mapping",
    "insufficient_coverage",
    "excessive_blur",
    "unsupported_camera_mode",
    "corrupt_insv",
    "unsynchronized_lens_streams",
    "missing_rig_calibration",
    "pose_estimation_failure",
    "weak_registration",
    "loop_closure_failure",
    "ambiguous_metric_scale",
    "scale_anchor_rejection",
    "invalid_depth_alignment",
    "training_divergence",
    "training_timeout",
    "nan_output",
    "gpu_out_of_memory",
    "provider_capacity",
    "provider_admission_failure",
    "worker_startup_failure",
    "checkpoint_acquisition_failure",
    "malformed_output",
    "invalid_artifact_digest",
    "heldout_evaluation_failure",
    "collider_qualification_failure",
    "isaac_load_failure",
    "blank_render",
    "missing_collision_properties",
    "budget_exhaustion",
    "ttl_expiration",
    "repeated_identical_blocker",
    "teardown_verification_failure",
    "provider_interruption",
    "permanent_incompatibility",
}

LEGAL_RECOVERY_ACTIONS = {
    "request_targeted_recapture",
    "reduce_redundant_candidate_frames",
    "retry_once_same_worker",
    "resume_bound_checkpoint",
    "choose_prequalified_matching_method",
    "choose_prequalified_reconstruction_method",
    "use_already_authorized_provider",
    "request_metric_anchor",
    "request_additional_authority",
    "preserve_evidence_and_stop",
    "abstain",
}

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_IMAGE_RE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")


class ReconstructionWorkerContractError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise ReconstructionWorkerContractError(["artifact_not_json_serializable"]) from exc


def _digest(value: Any) -> bool:
    return isinstance(value, str) and _DIGEST_RE.fullmatch(value) is not None


def _commit(value: Any) -> bool:
    return isinstance(value, str) and _COMMIT_RE.fullmatch(value) is not None


def _finite(value: Any, *, minimum: float = 0.0) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(
        float(value)
    ) and float(value) >= minimum


def _timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _digest_rows(value: Any, *, allow_empty: bool = False) -> bool:
    return isinstance(value, list) and (allow_empty or bool(value)) and all(
        isinstance(row, Mapping)
        and isinstance(row.get("artifact_id"), str)
        and bool(row["artifact_id"])
        and _digest(row.get("digest"))
        for row in value
    )


def _has_secret_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in ("password", "secret", "credential", "api_key")):
                if nested not in (None, "", [], {}):
                    return True
            if _has_secret_key(nested):
                return True
    elif isinstance(value, list):
        return any(_has_secret_key(item) for item in value)
    return False


def _validate_common(value: Mapping[str, Any], errors: list[str]) -> None:
    for key in ("stable_run_identity", "source_capture_identity", "producing_method", "implementation_version"):
        if not isinstance(value.get(key), str) or not str(value[key]).strip():
            errors.append(f"{key}_missing")
    if not _digest(value.get("source_capture_digest")):
        errors.append("source_capture_digest_invalid")
    if not _digest_rows(value.get("original_file_references")):
        errors.append("original_file_references_invalid")
    container = value.get("container_image_digest")
    if container is not None and _IMAGE_RE.fullmatch(str(container)) is None:
        errors.append("container_image_digest_invalid")
    if not _commit(value.get("source_commit_sha")):
        errors.append("source_commit_sha_invalid")
    if not _digest(value.get("deterministic_configuration_digest")):
        errors.append("deterministic_configuration_digest_invalid")
    if not _digest_rows(value.get("input_digests")):
        errors.append("input_digests_invalid")
    if not _digest_rows(value.get("output_digests"), allow_empty=True):
        errors.append("output_digests_invalid")
    if not _digest(value.get("train_heldout_split_digest")):
        errors.append("train_heldout_split_digest_invalid")
    for key in (
        "camera_calibration_binding",
        "coordinate_frame_declaration",
        "provider_runtime_identity",
        "authority_used",
        "parent_artifact_or_event",
    ):
        if not isinstance(value.get(key), Mapping):
            errors.append(f"{key}_invalid")
    if value.get("units") not in {"meters", "unknown"}:
        errors.append("units_invalid")
    if value.get("metric_scale_status") not in {
        "validated",
        "sensor_metric_unvalidated",
        "anchor_required",
        "unknown",
    }:
        errors.append("metric_scale_status_invalid")
    for key in ("cost_usd", "duration_seconds"):
        if not _finite(value.get(key)):
            errors.append(f"{key}_invalid")
    for key in ("warnings", "blockers"):
        if not isinstance(value.get(key), list) or not all(
            isinstance(item, str) and item for item in value[key]
        ):
            errors.append(f"{key}_invalid")
    if not isinstance(value.get("proof_effect"), str) or not value.get("proof_effect"):
        errors.append("proof_effect_invalid")
    if not isinstance(value.get("claim_ceiling"), str) or not value.get("claim_ceiling"):
        errors.append("claim_ceiling_invalid")
    if not _timestamp(value.get("timestamp")):
        errors.append("timestamp_invalid")
    if _has_secret_key(value):
        errors.append("secret_value_forbidden")


def _finalize(value: Mapping[str, Any], *, schema: str, digest_field: str) -> dict[str, Any]:
    artifact = _clone(dict(value))
    supplied = artifact.pop(digest_field, None)
    artifact["schema_version"] = schema
    expected = canonical_digest(artifact, digest_field=digest_field)
    if supplied is not None and supplied != expected:
        raise ReconstructionWorkerContractError([f"{digest_field}_mismatch"])
    artifact[digest_field] = expected
    return artifact


def build_worker_stack_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _clone(dict(value))
    errors: list[str] = []
    artifact["components"] = [dict(row) for row in PINNED_WORKER_COMPONENTS]
    if artifact.get("worker_family") != "blueprint-reconstruction-worker":
        errors.append("worker_family_invalid")
    if artifact.get("runnable_platform") != "linux/amd64":
        errors.append("runnable_platform_invalid")
    if artifact.get("headless_required") is not True or artifact.get("display_required") is not False:
        errors.append("headless_contract_invalid")
    if not _commit(artifact.get("source_commit_sha")):
        errors.append("source_commit_sha_invalid")
    if artifact.get("qualification_status") != "candidate_unbuilt":
        errors.append("qualification_status_invalid")
    if artifact.get("minimum_vram_gb") not in {16, 24, 48}:
        errors.append("minimum_vram_gb_invalid")
    if not isinstance(artifact.get("supported_compute_capabilities"), list) or not artifact[
        "supported_compute_capabilities"
    ]:
        errors.append("compute_capabilities_missing")
    if not isinstance(artifact.get("tested_driver_range"), Mapping):
        errors.append("tested_driver_range_invalid")
    if artifact.get("tested_driver_range") != {"status": "not_yet_tested"}:
        errors.append("unbuilt_manifest_cannot_claim_driver_test")
    if not isinstance(artifact.get("model_assets"), list) or not all(
        isinstance(row, Mapping)
        and isinstance(row.get("model_id"), str)
        and _digest(row.get("digest"))
        and isinstance(row.get("license"), str)
        for row in artifact.get("model_assets", [])
    ):
        errors.append("model_assets_invalid")
    if artifact.get("hidden_heldout_access") is not False:
        errors.append("hidden_heldout_access_forbidden")
    if artifact.get("trainer_self_grading") is not False:
        errors.append("trainer_self_grading_forbidden")
    if errors:
        raise ReconstructionWorkerContractError(errors)
    return _finalize(
        artifact, schema=WORKER_STACK_SCHEMA_VERSION, digest_field="worker_stack_manifest_digest"
    )


def build_worker_build_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _clone(dict(value))
    errors: list[str] = []
    if not _digest(artifact.get("worker_stack_manifest_digest")):
        errors.append("worker_stack_manifest_digest_invalid")
    if artifact.get("status") not in {"built", "failed"}:
        errors.append("build_status_invalid")
    if artifact.get("status") == "built" and _IMAGE_RE.fullmatch(
        str(artifact.get("resolved_image_digest") or "")
    ) is None:
        errors.append("resolved_image_digest_invalid")
    if artifact.get("status") == "failed" and artifact.get("resolved_image_digest") is not None:
        errors.append("failed_build_cannot_resolve_image")
    if not _commit(artifact.get("source_commit_sha")):
        errors.append("source_commit_sha_invalid")
    if not _digest(artifact.get("build_context_digest")):
        errors.append("build_context_digest_invalid")
    if not _finite(artifact.get("duration_seconds")) or not _finite(artifact.get("cost_usd")):
        errors.append("build_measurement_invalid")
    if not isinstance(artifact.get("logs"), list) or not _digest_rows(
        artifact.get("logs"), allow_empty=artifact.get("status") == "failed"
    ):
        errors.append("build_logs_invalid")
    if not isinstance(artifact.get("blockers"), list):
        errors.append("blockers_invalid")
    if artifact.get("scientific_qualification_inferred") is not False:
        errors.append("build_cannot_infer_scientific_qualification")
    if errors:
        raise ReconstructionWorkerContractError(errors)
    return _finalize(
        artifact, schema=WORKER_BUILD_RECEIPT_SCHEMA_VERSION, digest_field="build_receipt_digest"
    )


def build_worker_smoke_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _clone(dict(value))
    errors: list[str] = []
    if not _digest(artifact.get("build_receipt_digest")):
        errors.append("build_receipt_digest_invalid")
    if _IMAGE_RE.fullmatch(str(artifact.get("resolved_image_digest") or "")) is None:
        errors.append("resolved_image_digest_invalid")
    if artifact.get("status") not in {"passed", "failed"}:
        errors.append("smoke_status_invalid")
    checks = artifact.get("checks")
    if not isinstance(checks, list) or not checks or not all(
        isinstance(row, Mapping)
        and isinstance(row.get("check_id"), str)
        and row.get("status") in {"passed", "failed"}
        and _digest(row.get("output_digest"))
        for row in checks or []
    ):
        errors.append("smoke_checks_invalid")
    if artifact.get("status") == "passed" and any(
        row.get("status") != "passed" for row in checks or [] if isinstance(row, Mapping)
    ):
        errors.append("smoke_status_inconsistent")
    if artifact.get("display_attached") is not False:
        errors.append("headless_smoke_required")
    if artifact.get("scientific_qualification_inferred") is not False:
        errors.append("smoke_cannot_infer_scientific_qualification")
    if errors:
        raise ReconstructionWorkerContractError(errors)
    return _finalize(
        artifact,
        schema=WORKER_SMOKE_RECEIPT_SCHEMA_VERSION,
        digest_field="smoke_test_receipt_digest",
    )


def _request_common(artifact: Mapping[str, Any], errors: list[str]) -> None:
    _validate_common(artifact, errors)
    if _IMAGE_RE.fullmatch(str(artifact.get("container_image_digest") or "")) is None:
        errors.append("request_requires_resolved_worker_image")
    if artifact.get("candidate_dataset_contains_hidden_heldout_pixels") is not False:
        errors.append("hidden_heldout_pixels_forbidden")
    if artifact.get("candidate_can_change_split") is not False:
        errors.append("candidate_split_mutation_forbidden")
    if artifact.get("output_digests") != []:
        errors.append("request_output_digests_must_be_empty")
    if artifact.get("cost_usd") != 0 or artifact.get("duration_seconds") != 0:
        errors.append("request_measurements_must_be_zero")


def build_pose_estimation_request(value: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _clone(dict(value))
    errors: list[str] = []
    _request_common(artifact, errors)
    method = artifact.get("method_profile_id")
    if method not in POSE_METHODS:
        errors.append("pose_method_unsupported")
    elif (
        artifact.get("feature_extractor"), artifact.get("feature_matcher")
    ) != POSE_METHODS[method]:
        errors.append("pose_method_pairing_invalid")
    if artifact.get("camera_model") not in CAMERA_MODELS:
        errors.append("camera_model_unsupported")
    for key in ("reconstruction_dataset_digest", "camera_rig_digest", "calibration_digest"):
        if not _digest(artifact.get(key)):
            errors.append(f"{key}_invalid")
    if not _digest(artifact.get("model_asset_digest")) and artifact.get(
        "feature_extractor"
    ) != "SIFT":
        errors.append("learned_feature_model_digest_missing")
    if "LIGHTGLUE" in str(artifact.get("feature_matcher")) and not _digest(
        artifact.get("matcher_model_asset_digest")
    ):
        errors.append("lightglue_model_digest_missing")
    if artifact.get("deterministic_matching") is not True:
        errors.append("deterministic_matching_required")
    if not isinstance(artifact.get("random_seed"), int) or isinstance(
        artifact.get("random_seed"), bool
    ):
        errors.append("random_seed_invalid")
    if not isinstance(artifact.get("resource_request"), Mapping):
        errors.append("resource_request_invalid")
    if not _finite(artifact.get("timeout_seconds"), minimum=1) or not _finite(
        artifact.get("spend_cap_usd")
    ):
        errors.append("execution_bounds_invalid")
    if artifact.get("candidate_may_read_hidden_heldout") is not False:
        errors.append("hidden_heldout_access_forbidden")
    if errors:
        raise ReconstructionWorkerContractError(errors)
    return _finalize(
        artifact, schema=POSE_REQUEST_SCHEMA_VERSION, digest_field="pose_estimation_request_digest"
    )


def build_pose_estimation_result(value: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _clone(dict(value))
    errors: list[str] = []
    _validate_common(artifact, errors)
    if not _digest(artifact.get("pose_estimation_request_digest")):
        errors.append("pose_estimation_request_digest_invalid")
    if artifact.get("status") not in {"succeeded", "failed", "timed_out", "interrupted"}:
        errors.append("pose_result_status_invalid")
    failure = artifact.get("failure_code")
    if artifact.get("status") == "succeeded":
        if failure is not None:
            errors.append("successful_pose_result_has_failure")
        if not artifact.get("registered_observation_ids"):
            errors.append("registered_observations_missing")
        if not _digest_rows(artifact.get("output_digests")):
            errors.append("pose_outputs_missing")
    elif failure not in FAILURE_CODES:
        errors.append("pose_failure_code_invalid")
    for key in ("registered_observation_ids", "rejected_observation_ids", "warnings", "blockers"):
        if not isinstance(artifact.get(key), list):
            errors.append(f"{key}_invalid")
    if artifact.get("heldout_labels_included") is not False:
        errors.append("heldout_labels_forbidden")
    if artifact.get("candidate_self_graded") is not False:
        errors.append("candidate_self_grading_forbidden")
    if artifact.get("proof_effect") != "calibrated_trajectory_candidate_only" or artifact.get(
        "claim_ceiling"
    ) != "calibrated_camera_trajectory":
        errors.append("pose_claim_boundary_invalid")
    if "heldout_metrics" in artifact:
        errors.append("heldout_metrics_forbidden")
    if errors:
        raise ReconstructionWorkerContractError(errors)
    return _finalize(
        artifact, schema=POSE_RESULT_SCHEMA_VERSION, digest_field="pose_estimation_result_digest"
    )


def build_training_request(value: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _clone(dict(value))
    errors: list[str] = []
    _request_common(artifact, errors)
    if artifact.get("method_profile_id") not in TRAINER_METHODS:
        errors.append("trainer_method_unsupported")
    for key in (
        "reconstruction_dataset_digest",
        "calibration_digest",
        "initialization_geometry_digest",
        "pose_result_digest",
        "worker_stack_manifest_digest",
        "evaluation_contract_digest",
    ):
        if not _digest(artifact.get(key)):
            errors.append(f"{key}_invalid")
    if artifact.get("camera_model") not in CAMERA_MODELS:
        errors.append("camera_model_unsupported")
    if not isinstance(artifact.get("densification_configuration"), Mapping):
        errors.append("densification_configuration_invalid")
    if artifact.get("densification_configuration", {}).get("strategy") != "mcmc":
        errors.append("mcmc_densification_required")
    if not isinstance(artifact.get("random_seed"), int) or isinstance(
        artifact.get("random_seed"), bool
    ):
        errors.append("random_seed_invalid")
    if not isinstance(artifact.get("iteration_budget"), int) or artifact.get(
        "iteration_budget", 0
    ) <= 0:
        errors.append("iteration_budget_invalid")
    if not isinstance(artifact.get("resource_request"), Mapping):
        errors.append("resource_request_invalid")
    if not _finite(artifact.get("timeout_seconds"), minimum=1) or not _finite(
        artifact.get("spend_cap_usd")
    ):
        errors.append("execution_bounds_invalid")
    if artifact.get("candidate_may_read_hidden_heldout") is not False:
        errors.append("hidden_heldout_access_forbidden")
    if artifact.get("trainer_may_grade_heldout") is not False:
        errors.append("trainer_self_grading_forbidden")
    if not isinstance(artifact.get("output_contract"), Mapping):
        errors.append("output_contract_invalid")
    if errors:
        raise ReconstructionWorkerContractError(errors)
    return _finalize(
        artifact,
        schema=TRAINING_REQUEST_SCHEMA_VERSION,
        digest_field="reconstruction_training_request_digest",
    )


def build_checkpoint_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _clone(dict(value))
    errors: list[str] = []
    _validate_common(artifact, errors)
    if not _digest(artifact.get("reconstruction_training_request_digest")):
        errors.append("reconstruction_training_request_digest_invalid")
    if not isinstance(artifact.get("iteration"), int) or artifact.get("iteration", -1) < 0:
        errors.append("checkpoint_iteration_invalid")
    if not _digest(artifact.get("checkpoint_digest")):
        errors.append("checkpoint_digest_invalid")
    if not isinstance(artifact.get("random_state"), Mapping) or not _digest(
        artifact.get("random_state", {}).get("digest")
    ):
        errors.append("checkpoint_random_state_invalid")
    if artifact.get("resume_requires_exact_request_binding") is not True:
        errors.append("checkpoint_resume_binding_required")
    if artifact.get("hidden_heldout_state_included") is not False:
        errors.append("hidden_heldout_state_forbidden")
    if artifact.get("proof_effect") != "none" or artifact.get("claim_ceiling") != "checkpoint_only":
        errors.append("checkpoint_claim_boundary_invalid")
    if errors:
        raise ReconstructionWorkerContractError(errors)
    return _finalize(
        artifact, schema=CHECKPOINT_SCHEMA_VERSION, digest_field="checkpoint_manifest_digest"
    )


def build_training_result(value: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _clone(dict(value))
    errors: list[str] = []
    _validate_common(artifact, errors)
    if not _digest(artifact.get("reconstruction_training_request_digest")):
        errors.append("reconstruction_training_request_digest_invalid")
    if artifact.get("status") not in {"succeeded", "failed", "timed_out", "interrupted"}:
        errors.append("training_result_status_invalid")
    failure = artifact.get("failure_code")
    if artifact.get("status") == "succeeded":
        if failure is not None:
            errors.append("successful_training_result_has_failure")
        if not _digest_rows(artifact.get("output_digests")):
            errors.append("trained_artifacts_missing")
        if not _digest_rows(artifact.get("checkpoint_references")):
            errors.append("training_checkpoints_missing")
    elif failure not in FAILURE_CODES:
        errors.append("training_failure_code_invalid")
    if not isinstance(artifact.get("checkpoint_references"), list):
        errors.append("checkpoint_references_invalid")
    if not isinstance(artifact.get("training_metrics"), Mapping):
        errors.append("training_metrics_invalid")
    if artifact.get("heldout_labels_included") is not False:
        errors.append("heldout_labels_forbidden")
    if artifact.get("candidate_self_graded") is not False:
        errors.append("candidate_self_grading_forbidden")
    if "heldout_metrics" in artifact:
        errors.append("heldout_metrics_forbidden")
    if not isinstance(artifact.get("registered_observation_ids"), list) or not isinstance(
        artifact.get("rejected_observation_ids"), list
    ):
        errors.append("observation_ledger_invalid")
    if not isinstance(artifact.get("peak_resource_use"), Mapping):
        errors.append("peak_resource_use_invalid")
    actions = artifact.get("legal_next_actions")
    if not isinstance(actions, list) or any(action not in LEGAL_RECOVERY_ACTIONS for action in actions):
        errors.append("legal_next_actions_invalid")
    if artifact.get("proof_effect") != "appearance_asset_candidate_only" or artifact.get(
        "claim_ceiling"
    ) != "appearance_reconstruction":
        errors.append("training_claim_boundary_invalid")
    if errors:
        raise ReconstructionWorkerContractError(errors)
    return _finalize(
        artifact,
        schema=TRAINING_RESULT_SCHEMA_VERSION,
        digest_field="reconstruction_training_result_digest",
    )


__all__ = [
    "CAMERA_MODELS",
    "CHECKPOINT_SCHEMA_VERSION",
    "FAILURE_CODES",
    "LEGAL_RECOVERY_ACTIONS",
    "PINNED_WORKER_COMPONENTS",
    "PINNED_MODEL_ASSETS",
    "POSE_METHODS",
    "REQUIREMENTS_LOCK_SHA256",
    "POSE_REQUEST_SCHEMA_VERSION",
    "POSE_RESULT_SCHEMA_VERSION",
    "ReconstructionWorkerContractError",
    "TRAINER_METHODS",
    "TRAINING_REQUEST_SCHEMA_VERSION",
    "TRAINING_RESULT_SCHEMA_VERSION",
    "WORKER_BUILD_RECEIPT_SCHEMA_VERSION",
    "WORKER_SMOKE_RECEIPT_SCHEMA_VERSION",
    "WORKER_STACK_SCHEMA_VERSION",
    "build_checkpoint_manifest",
    "build_pose_estimation_request",
    "build_pose_estimation_result",
    "build_training_request",
    "build_training_result",
    "build_worker_build_receipt",
    "build_worker_smoke_receipt",
    "build_worker_stack_manifest",
]
