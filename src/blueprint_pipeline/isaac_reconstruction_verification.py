"""Deterministic request, runtime, and replay checks for reconstructed Isaac assets.

The Isaac process is an evidence producer, not its own grader.  This module
validates its versioned request/receipt, independently re-hashes the exact
retrieved USDZ and PNGs, decodes the PNGs again, and only then emits the bounded
Isaac compatibility result.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .reconstruction_geometry_contracts import (
    ReconstructionGeometryContractError,
    build_isaac_asset_verification_result,
    build_nurec_openusd_packaging_result,
)


ISAAC_VERIFICATION_REQUEST_SCHEMA = "isaac_asset_verification_request.v1"
ISAAC_RUNTIME_RESULT_SCHEMA = "isaac_splat_nurec_render_result.v3"
MAX_PACKAGE_BYTES = 4_000_000_000
MAX_RENDER_BYTES = 250_000_000
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_IMAGE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")
_CAMERA_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class IsaacReconstructionVerificationError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise IsaacReconstructionVerificationError(["isaac_artifact_not_json"]) from exc


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _integer_at_least(value: Any, minimum: int) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and value >= minimum
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _safe_relative(value: Any, *, code: str) -> PurePosixPath:
    text = str(value or "").replace("\\", "/")
    relative = PurePosixPath(text)
    if (
        not text
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or ":" in relative.parts[0]
    ):
        raise IsaacReconstructionVerificationError([code])
    return relative


def _safe_artifact(
    root: Path,
    *,
    reference: Any,
    digest: Any,
    suffixes: set[str],
    maximum_bytes: int,
    code: str,
) -> Path:
    relative = _safe_relative(reference, code=f"{code}_reference_unsafe")
    candidate = root.joinpath(*relative.parts)
    if candidate.is_symlink():
        raise IsaacReconstructionVerificationError([f"{code}_symlink_forbidden"])
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise IsaacReconstructionVerificationError([f"{code}_missing"]) from exc
    if root != resolved and root not in resolved.parents:
        raise IsaacReconstructionVerificationError([f"{code}_path_escape"])
    if not resolved.is_file() or resolved.suffix.lower() not in suffixes:
        raise IsaacReconstructionVerificationError([f"{code}_format_invalid"])
    size = resolved.stat().st_size
    if size < 1 or size > maximum_bytes:
        raise IsaacReconstructionVerificationError([f"{code}_size_invalid"])
    if not _is_digest(digest) or _sha256(resolved) != digest:
        raise IsaacReconstructionVerificationError([f"{code}_digest_mismatch"])
    return resolved


def _common_lineage(value: Mapping[str, Any], errors: list[str]) -> None:
    for key in (
        "stable_run_identity",
        "source_capture_identity",
        "producing_method",
        "implementation_version",
        "timestamp",
    ):
        if not isinstance(value.get(key), str) or not value[key]:
            errors.append(f"{key}_missing")
    for key in (
        "source_capture_digest",
        "deterministic_configuration_digest",
        "train_heldout_split_digest",
    ):
        if not _is_digest(value.get(key)):
            errors.append(f"{key}_invalid")
    if _COMMIT.fullmatch(str(value.get("source_commit_sha") or "")) is None:
        errors.append("source_commit_sha_invalid")
    if value.get("units") != "meters":
        errors.append("units_must_be_meters")
    for key in (
        "original_file_references",
        "input_digests",
        "output_digests",
        "warnings",
        "blockers",
    ):
        if not isinstance(value.get(key), list):
            errors.append(f"{key}_invalid")
    for key in (
        "camera_calibration_binding",
        "coordinate_frame_declaration",
        "provider_runtime_identity",
        "authority_used",
        "parent_artifact_or_event",
    ):
        if not isinstance(value.get(key), Mapping):
            errors.append(f"{key}_invalid")
    for key in ("cost_usd", "duration_seconds"):
        number = value.get(key)
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
            or float(number) < 0
        ):
            errors.append(f"{key}_invalid")


def build_isaac_asset_verification_request(value: Mapping[str, Any]) -> dict[str, Any]:
    request = _clone(dict(value))
    supplied = request.pop("isaac_verification_request_digest", None)
    errors: list[str] = []
    _common_lineage(request, errors)
    package_value = request.get("packaging_result")
    package: Mapping[str, Any] | None = None
    if not isinstance(package_value, Mapping):
        errors.append("isaac_packaging_result_missing")
    else:
        try:
            package = build_nurec_openusd_packaging_result(package_value)
            request["packaging_result"] = package
        except ReconstructionGeometryContractError as exc:
            errors.extend(f"isaac_packaging_result_invalid:{code}" for code in exc.codes)
    for key in (
        "packaging_result_digest",
        "package_digest",
        "fixed_camera_spec_digest",
        "runtime_implementation_digest",
    ):
        if not _is_digest(request.get(key)):
            errors.append(f"{key}_invalid")
    if _IMAGE.fullmatch(str(request.get("runtime_container_image_digest") or "")) is None:
        errors.append("runtime_container_image_digest_invalid")
    if package is not None and any(
        (
            request.get("packaging_result_digest") != package.get("packaging_result_digest"),
            request.get("package_digest") != package.get("package_digest"),
            request.get("package_artifact_reference")
            != package.get("package_artifact_reference"),
            request.get("source_capture_digest") != package.get("source_capture_digest"),
            request.get("train_heldout_split_digest")
            != package.get("train_heldout_split_digest"),
            request.get("camera_calibration_binding")
            != package.get("camera_calibration_binding"),
            request.get("coordinate_frame_declaration")
            != package.get("coordinate_frame_declaration"),
        )
    ):
        errors.append("isaac_packaging_request_lineage_mismatch")
    try:
        package_reference = _safe_relative(
            request.get("package_artifact_reference"),
            code="isaac_package_artifact_reference_unsafe",
        )
        if package_reference.suffix.lower() != ".usdz":
            errors.append("isaac_package_artifact_format_invalid")
    except IsaacReconstructionVerificationError as exc:
        errors.extend(exc.codes)
    camera_ids = request.get("fixed_camera_ids")
    if (
        not isinstance(camera_ids, list)
        or not camera_ids
        or any(
            not isinstance(item, str) or _CAMERA_ID.fullmatch(item) is None
            for item in camera_ids
        )
        or len(set(camera_ids)) != len(camera_ids)
    ):
        errors.append("isaac_fixed_camera_ids_invalid")
    expected_prims = request.get("expected_prim_paths")
    if not isinstance(expected_prims, Mapping) or expected_prims != {
        "appearance": "/World/BlueprintReconstruction/Appearance",
        "collision": "/World/BlueprintReconstruction/Collision",
    }:
        errors.append("isaac_expected_prim_paths_invalid")
    probe = request.get("physics_probe_request")
    if not isinstance(probe, Mapping):
        errors.append("isaac_physics_probe_request_missing")
    else:
        steps = probe.get("steps")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
            errors.append("isaac_physics_probe_steps_invalid")
        if probe.get("manufacture_ground_plane") is not False:
            errors.append("isaac_manufactured_ground_plane_forbidden")
        if probe.get("require_contact_event") is not True:
            errors.append("isaac_contact_event_required")
        if probe.get("test_body") != {
            "shape": "cube",
            "size_m": 0.1,
            "mass_kg": 1.0,
            "spawn_height_above_ground_m": 0.5,
        }:
            errors.append("isaac_test_body_configuration_invalid")
        if probe.get("gravity_m_s2") != -9.81 or probe.get("physics_dt_seconds") != 1.0 / 60.0:
            errors.append("isaac_physics_configuration_invalid")
    if request.get("headless") is not True or request.get("display_attached") is not False:
        errors.append("isaac_headless_execution_required")
    timeout = request.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 60 <= timeout <= 14_400:
        errors.append("isaac_timeout_invalid")
    resource = request.get("resource_request")
    if (
        not isinstance(resource, Mapping)
        or resource.get("gpu_count") != 1
        or not isinstance(resource.get("minimum_vram_gb"), int)
        or resource.get("minimum_vram_gb", 0) < 16
    ):
        errors.append("isaac_resource_request_invalid")
    if request.get("output_digests") != []:
        errors.append("isaac_request_cannot_predeclare_outputs")
    if request.get("proof_effect") != "none" or request.get("claim_ceiling") != "request_only":
        errors.append("isaac_request_claim_boundary_invalid")
    input_digests = {
        row.get("digest")
        for row in request.get("input_digests") or []
        if isinstance(row, Mapping)
    }
    for required in (
        request.get("packaging_result_digest"),
        request.get("package_digest"),
        request.get("fixed_camera_spec_digest"),
        request.get("runtime_implementation_digest"),
    ):
        if required not in input_digests:
            errors.append("isaac_request_input_digest_binding_missing")
    if errors:
        raise IsaacReconstructionVerificationError(errors)
    request["schema_version"] = ISAAC_VERIFICATION_REQUEST_SCHEMA
    expected = canonical_digest(request, digest_field="isaac_verification_request_digest")
    if supplied is not None and supplied != expected:
        raise IsaacReconstructionVerificationError(["isaac_verification_request_digest_mismatch"])
    request["isaac_verification_request_digest"] = expected
    return request


def build_isaac_runtime_result_v3(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _clone(dict(value))
    supplied = result.pop("isaac_runtime_result_digest", None)
    errors: list[str] = []
    if result.get("schema_version") != ISAAC_RUNTIME_RESULT_SCHEMA:
        errors.append("isaac_runtime_result_v3_required")
    if result.get("status") not in {"running", "completed", "blocked"}:
        errors.append("isaac_runtime_status_invalid")
    for key in (
        "isaac_verification_request_digest",
        "package_digest",
        "fixed_camera_spec_digest",
        "runtime_implementation_digest",
    ):
        if not _is_digest(result.get(key)):
            errors.append(f"{key}_invalid")
    if _IMAGE.fullmatch(str(result.get("runtime_container_image_digest") or "")) is None:
        errors.append("runtime_container_image_digest_invalid")
    if result.get("raw_secret_values_recorded") is not False:
        errors.append("isaac_runtime_secret_recording_state_invalid")
    identity = result.get("runtime_identity")
    if not isinstance(identity, Mapping):
        errors.append("isaac_runtime_identity_missing")
    elif (
        identity.get("runtime") != "isaac_sim"
        or not isinstance(identity.get("renderer"), str)
        or not identity.get("renderer")
        or not isinstance(identity.get("python_version"), str)
        or not identity.get("python_version")
        or identity.get("headless") is not True
    ):
        errors.append("isaac_runtime_identity_invalid")
    if result.get("status") == "completed":
        for key in ("cost_usd", "duration_seconds"):
            number = result.get(key)
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(float(number))
                or float(number) < 0
            ):
                errors.append(f"isaac_runtime_{key}_invalid")
        for key in ("stage", "physics_probe", "proof_boundary"):
            if not isinstance(result.get(key), Mapping):
                errors.append(f"isaac_runtime_{key}_missing")
        cameras = result.get("cameras")
        if not isinstance(cameras, list) or not cameras:
            errors.append("isaac_runtime_cameras_missing")
        else:
            ids: list[str] = []
            for index, row in enumerate(cameras):
                if not isinstance(row, Mapping):
                    errors.append(f"isaac_runtime_camera_invalid:{index}")
                    continue
                ids.append(str(row.get("id") or ""))
                try:
                    reference = _safe_relative(
                        row.get("artifact_reference"),
                        code=f"isaac_runtime_camera_reference_unsafe:{index}",
                    )
                    if reference.suffix.lower() != ".png":
                        errors.append(f"isaac_runtime_camera_format_invalid:{index}")
                except IsaacReconstructionVerificationError as exc:
                    errors.extend(exc.codes)
                if not _is_digest(row.get("digest")):
                    errors.append(f"isaac_runtime_camera_digest_invalid:{index}")
                for key in ("width", "height"):
                    if isinstance(row.get(key), bool) or not isinstance(row.get(key), int) or row[key] < 1:
                        errors.append(f"isaac_runtime_camera_{key}_invalid:{index}")
                for key in ("pixel_mean", "pixel_std"):
                    number = row.get(key)
                    if (
                        isinstance(number, bool)
                        or not isinstance(number, (int, float))
                        or not math.isfinite(float(number))
                    ):
                        errors.append(f"isaac_runtime_camera_{key}_invalid:{index}")
                if row.get("nonblank") is not True:
                    errors.append(f"isaac_runtime_camera_blank:{index}")
            if any(not item for item in ids) or len(ids) != len(set(ids)):
                errors.append("isaac_runtime_camera_ids_invalid")
    proof = result.get("proof_boundary")
    if isinstance(proof, Mapping):
        for key in (
            "simulator_task_success_proven",
            "physics_navigation_control_proven",
            "physical_success_proven",
            "physical_robot_readiness_proven",
            "deployment_readiness_proven",
        ):
            if proof.get(key) is not False:
                errors.append(f"isaac_forbidden_claim_promotion:{key}")
        if result.get("status") == "completed" and proof.get(
            "isaac_load_render_physics_presence_compatibility"
        ) is not True:
            errors.append("isaac_runtime_compatibility_claim_missing")
    if errors:
        raise IsaacReconstructionVerificationError(errors)
    expected = canonical_digest(result, digest_field="isaac_runtime_result_digest")
    if supplied is not None and supplied != expected:
        raise IsaacReconstructionVerificationError(["isaac_runtime_result_digest_mismatch"])
    result["isaac_runtime_result_digest"] = expected
    return result


def _render_measurements(path: Path) -> tuple[int, int, float, float]:
    try:
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
            array = np.asarray(rgb, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        raise IsaacReconstructionVerificationError(["isaac_render_decode_failed"]) from exc
    if array.ndim != 3 or array.shape[2] != 3 or not np.isfinite(array).all():
        raise IsaacReconstructionVerificationError(["isaac_render_pixels_invalid"])
    height, width = array.shape[:2]
    return width, height, float(array.mean()), float(array.std())


def normalize_isaac_reconstruction_verification(
    *,
    verification_request: Mapping[str, Any],
    runtime_result: Mapping[str, Any],
    package_artifact_root: str | Path,
    runtime_artifact_root: str | Path,
) -> dict[str, Any]:
    """Independently verify retrieved runtime artifacts before qualification."""

    request = build_isaac_asset_verification_request(verification_request)
    runtime = build_isaac_runtime_result_v3(runtime_result)
    package = request["packaging_result"]
    blockers: list[str] = []
    if runtime.get("status") != "completed":
        blockers.append("isaac_runtime_not_completed")
    for key in (
        "isaac_verification_request_digest",
        "package_digest",
        "fixed_camera_spec_digest",
        "runtime_container_image_digest",
        "runtime_implementation_digest",
    ):
        expected = request.get(key)
        if runtime.get(key) != expected:
            blockers.append(f"isaac_runtime_request_binding_mismatch:{key}")
    package_root = Path(package_artifact_root)
    runtime_root = Path(runtime_artifact_root)
    if package_root.is_symlink() or not package_root.is_dir():
        blockers.append("isaac_package_artifact_root_invalid")
    if runtime_root.is_symlink() or not runtime_root.is_dir():
        blockers.append("isaac_runtime_artifact_root_invalid")
    if blockers:
        raise IsaacReconstructionVerificationError(blockers)
    package_root = package_root.resolve()
    runtime_root = runtime_root.resolve()
    _safe_artifact(
        package_root,
        reference=request["package_artifact_reference"],
        digest=request["package_digest"],
        suffixes={".usdz"},
        maximum_bytes=MAX_PACKAGE_BYTES,
        code="isaac_exact_package",
    )
    stage = runtime.get("stage")
    stage = stage if isinstance(stage, Mapping) else {}
    if stage.get("meters_per_unit") != 1.0 or stage.get("up_axis") != "Z":
        blockers.append("isaac_stage_units_invalid")
    if stage.get("transforms_valid") is not True:
        blockers.append("isaac_stage_transforms_invalid")
    if stage.get("dependency_inspection_available") is not True:
        blockers.append("isaac_dependency_inspection_unavailable")
    if stage.get("missing_asset_count") != 0:
        blockers.append("isaac_missing_assets")
    if not _integer_at_least(stage.get("particlefield_prim_count"), 1):
        blockers.append("isaac_particlefield_not_loaded")
    if not _integer_at_least(stage.get("active_collision_prim_count"), 1):
        blockers.append("isaac_collision_geometry_inactive")
    if stage.get("obvious_scale_mismatch_detected") is not False:
        blockers.append("isaac_obvious_scale_mismatch")
    expected_prims = request["expected_prim_paths"]
    observed_prims = stage.get("expected_prim_paths")
    if not isinstance(observed_prims, Mapping) or any(
        observed_prims.get(name) != path for name, path in expected_prims.items()
    ):
        blockers.append("isaac_expected_prims_not_exactly_loaded")
    physics = runtime.get("physics_probe")
    physics = physics if isinstance(physics, Mapping) else {}
    if physics.get("ground_contact_surface_present") is not True:
        blockers.append("isaac_ground_contact_surface_missing")
    if not _integer_at_least(
        physics.get("steps_executed"), request["physics_probe_request"]["steps"]
    ):
        blockers.append("isaac_physics_probe_incomplete")
    if physics.get("live_rigid_body_pose_observed") is not True:
        blockers.append("isaac_test_body_pose_unavailable")
    if physics.get("test_body_fell_through_floor") is not False:
        blockers.append("isaac_test_body_fell_through_floor")
    if not _integer_at_least(physics.get("contact_event_count"), 1):
        blockers.append("isaac_test_body_contact_not_observed")
    if physics.get("probe_configuration") != {
        "test_body": request["physics_probe_request"]["test_body"],
        "gravity_m_s2": request["physics_probe_request"]["gravity_m_s2"],
        "physics_dt_seconds": request["physics_probe_request"]["physics_dt_seconds"],
    }:
        blockers.append("isaac_physics_probe_configuration_mismatch")
    renders = runtime.get("cameras")
    renders = renders if isinstance(renders, list) else []
    if [row.get("id") for row in renders if isinstance(row, Mapping)] != request[
        "fixed_camera_ids"
    ]:
        blockers.append("isaac_fixed_camera_set_mismatch")
    render_refs: list[dict[str, str]] = []
    for index, row_value in enumerate(renders):
        row = row_value if isinstance(row_value, Mapping) else {}
        try:
            path = _safe_artifact(
                runtime_root,
                reference=row.get("artifact_reference"),
                digest=row.get("digest"),
                suffixes={".png"},
                maximum_bytes=MAX_RENDER_BYTES,
                code=f"isaac_fixed_render:{index}",
            )
            width, height, pixel_mean, pixel_std = _render_measurements(path)
            if width != row.get("width") or height != row.get("height"):
                blockers.append(f"isaac_fixed_render_dimensions_mismatch:{index}")
            if abs(pixel_mean - float(row.get("pixel_mean") or 0.0)) > 0.001:
                blockers.append(f"isaac_fixed_render_mean_mismatch:{index}")
            if abs(pixel_std - float(row.get("pixel_std") or 0.0)) > 0.001:
                blockers.append(f"isaac_fixed_render_std_mismatch:{index}")
            if pixel_std <= 3.0 or row.get("nonblank") is not True:
                blockers.append(f"isaac_fixed_render_blank:{index}")
            render_refs.append(
                {
                    "artifact_id": str(row["artifact_reference"]),
                    "digest": str(row["digest"]),
                }
            )
        except IsaacReconstructionVerificationError as exc:
            blockers.extend(exc.codes)
    if not render_refs:
        blockers.append("isaac_fixed_camera_renders_missing")
    if blockers:
        raise IsaacReconstructionVerificationError(blockers)
    value = {
        key: _clone(request[key])
        for key in (
            "stable_run_identity",
            "source_capture_identity",
            "source_capture_digest",
            "original_file_references",
            "source_commit_sha",
            "train_heldout_split_digest",
            "camera_calibration_binding",
            "coordinate_frame_declaration",
            "units",
            "authority_used",
            "timestamp",
        )
    }
    value.update(
        {
            "producing_method": "blueprint.independent_isaac_artifact_verifier",
            "implementation_version": "1.0.0",
            "deterministic_configuration_digest": request[
                "isaac_verification_request_digest"
            ],
            "input_digests": [
                {"artifact_id": "package", "digest": request["package_digest"]},
                {
                    "artifact_id": "isaac_runtime_result",
                    "digest": runtime["isaac_runtime_result_digest"],
                },
                *render_refs,
            ],
            "output_digests": [],
            "provider_runtime_identity": runtime["runtime_identity"],
            "cost_usd": float(runtime.get("cost_usd") or 0.0),
            "duration_seconds": float(runtime.get("duration_seconds") or 0.0),
            "warnings": [
                "isaac_compatibility_does_not_prove_simulator_task_or_physical_success"
            ],
            "blockers": [],
            "parent_artifact_or_event": {
                "digest": request["isaac_verification_request_digest"]
            },
            "packaging_result_digest": package["packaging_result_digest"],
            "package_digest": request["package_digest"],
            "isaac_verification_request_digest": request[
                "isaac_verification_request_digest"
            ],
            "isaac_runtime_result_digest": runtime["isaac_runtime_result_digest"],
            "runtime_container_image_digest": request["runtime_container_image_digest"],
            "runtime_implementation_digest": request["runtime_implementation_digest"],
            "fixed_camera_spec_digest": request["fixed_camera_spec_digest"],
            "exact_package_rehash_verified": True,
            "runtime_artifact_rehash_verified": True,
            "checks": {
                "exact_package_opened": True,
                "expected_prims_present": True,
                "stage_units_valid": True,
                "transforms_valid": True,
                "missing_assets_detected": False,
                "particlefield_loaded": True,
                "collision_geometry_active": True,
                "ground_contact_surface_present": True,
                "test_body_fell_through_floor": False,
                "fixed_camera_renders_nonblank": True,
                "nan_or_corrupt_render_detected": False,
                "obvious_scale_mismatch_detected": False,
            },
            "fixed_camera_render_references": render_refs,
            "physics_probe": _clone(physics),
            "status": "verified_compatibility_only",
            "simulator_task_success_proven": False,
            "physical_success_proven": False,
            "deployment_readiness_proven": False,
            "proof_effect": "isaac_load_render_physics_presence_only",
            "claim_ceiling": "isaac_load_render_compatibility",
        }
    )
    return build_isaac_asset_verification_result(value)


def bind_isaac_reconstruction_verifier(
    *,
    runtime_result: Mapping[str, Any],
    package_artifact_root: str | Path,
    runtime_artifact_root: str | Path,
) -> Callable[..., dict[str, Any]]:
    """Bind retrieved artifact roots to the digest-only supervisor tool."""

    def run(*, source_artifact: Mapping[str, Any], output_root: str | Path) -> dict[str, Any]:
        result = normalize_isaac_reconstruction_verification(
            verification_request=source_artifact,
            runtime_result=runtime_result,
            package_artifact_root=package_artifact_root,
            runtime_artifact_root=runtime_artifact_root,
        )
        destination = Path(output_root)
        if destination.is_symlink():
            raise IsaacReconstructionVerificationError(["isaac_output_root_symlink_forbidden"])
        destination.mkdir(parents=True, exist_ok=True)
        final = destination / result["isaac_verification_result_digest"][7:]
        if final.exists() or final.is_symlink():
            existing = final / "isaac_asset_verification_result.v1.json"
            if existing.is_file() and json.loads(existing.read_text(encoding="utf-8")) == result:
                return result
            raise IsaacReconstructionVerificationError(["isaac_existing_output_tampered"])
        temporary = Path(tempfile.mkdtemp(prefix=".isaac-verify-", dir=destination))
        try:
            write_json(temporary / "isaac_asset_verification_result.v1.json", result)
            os.replace(temporary, final)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return result

    return run


__all__ = [
    "ISAAC_RUNTIME_RESULT_SCHEMA",
    "ISAAC_VERIFICATION_REQUEST_SCHEMA",
    "IsaacReconstructionVerificationError",
    "bind_isaac_reconstruction_verifier",
    "build_isaac_asset_verification_request",
    "build_isaac_runtime_result_v3",
    "normalize_isaac_reconstruction_verification",
]
