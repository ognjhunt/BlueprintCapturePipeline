"""Normalize headless Isaac runtime evidence for reconstructed site assets."""

from __future__ import annotations

import math
from typing import Any, Mapping

from .reconstruction_geometry_contracts import (
    ReconstructionGeometryContractError,
    build_isaac_asset_verification_result,
    build_nurec_openusd_packaging_result,
)


class IsaacReconstructionVerificationError(ValueError):
    pass


def normalize_isaac_reconstruction_verification(
    *,
    packaging_result: Mapping[str, Any],
    runtime_result: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> dict[str, Any]:
    """Require visual and physics-presence evidence from the exact package."""

    try:
        package = build_nurec_openusd_packaging_result(packaging_result)
    except ReconstructionGeometryContractError as exc:
        raise IsaacReconstructionVerificationError("isaac_package_contract_invalid") from exc
    blockers: list[str] = []
    if runtime_result.get("schema_version") != "isaac_splat_nurec_render_result.v2":
        blockers.append("isaac_runtime_result_v2_required")
    if runtime_result.get("status") != "completed":
        blockers.append("isaac_runtime_not_completed")
    if runtime_result.get("package_digest") != package["package_digest"]:
        blockers.append("isaac_exact_package_digest_mismatch")
    stage = runtime_result.get("stage")
    stage = stage if isinstance(stage, Mapping) else {}
    if stage.get("meters_per_unit") != 1.0 or stage.get("up_axis") != "Z":
        blockers.append("isaac_stage_units_invalid")
    if stage.get("transforms_valid") is not True:
        blockers.append("isaac_stage_transforms_invalid")
    missing_asset_count = stage.get("missing_asset_count")
    if isinstance(missing_asset_count, bool) or not isinstance(missing_asset_count, int) or missing_asset_count != 0:
        blockers.append("isaac_missing_assets")
    if int(stage.get("particlefield_prim_count") or 0) < 1:
        blockers.append("isaac_particlefield_not_loaded")
    if int(stage.get("active_collision_prim_count") or 0) < 1:
        blockers.append("isaac_collision_geometry_inactive")
    physics = runtime_result.get("physics_probe")
    physics = physics if isinstance(physics, Mapping) else {}
    if physics.get("ground_contact_surface_present") is not True:
        blockers.append("isaac_ground_contact_surface_missing")
    if int(physics.get("steps_executed") or 0) < 2:
        blockers.append("isaac_physics_probe_not_executed")
    if physics.get("test_body_fell_through_floor") is not False:
        blockers.append("isaac_test_body_fell_through_floor")
    if int(physics.get("contact_event_count") or 0) < 1:
        blockers.append("isaac_test_body_contact_not_observed")
    renders = runtime_result.get("cameras")
    renders = renders if isinstance(renders, list) else []
    render_refs: list[dict[str, str]] = []
    for index, render in enumerate(renders):
        row = render if isinstance(render, Mapping) else {}
        pixel_std = row.get("pixel_std")
        digest = row.get("digest")
        if (
            row.get("nonblank") is not True
            or isinstance(pixel_std, bool)
            or not isinstance(pixel_std, (int, float))
            or not math.isfinite(float(pixel_std))
            or float(pixel_std) <= 3.0
            or not isinstance(digest, str)
        ):
            blockers.append(f"isaac_fixed_render_invalid:{index}")
        else:
            render_refs.append({"artifact_id": str(row.get("id") or index), "digest": digest})
    if not render_refs:
        blockers.append("isaac_fixed_camera_renders_missing")
    if blockers:
        raise IsaacReconstructionVerificationError("; ".join(sorted(set(blockers))))
    value = dict(lineage)
    value.update(
        {
            "packaging_result_digest": package["packaging_result_digest"],
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
            "status": "verified_compatibility_only",
            "simulator_task_success_proven": False,
            "physical_success_proven": False,
            "deployment_readiness_proven": False,
            "proof_effect": "isaac_load_render_physics_presence_only",
            "claim_ceiling": "isaac_load_render_compatibility",
        }
    )
    return build_isaac_asset_verification_result(value)


__all__ = ["IsaacReconstructionVerificationError", "normalize_isaac_reconstruction_verification"]
