"""Check task-required visibility before a policy is loaded."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from .native_task_composition_diagnostic import (
    COVERAGE_PASSES, REQUEST_SCHEMA, file_record, run_composition_diagnostic, seal,
)
from .native_task_composition_worker import make_native_adapters


def articulated_reset_contact_region(plan):
    """Use the sealed mechanism contact and its existing arrival/clearance radius.

    Installed housings may be behind their surroundings. This region qualifies
    only the initial interaction patch, never the appearance of the whole asset.
    """
    if plan.get("task_kind") != "articulated_open_close":
        return None
    import numpy as np
    from .native_task_camera_start_configuration import pose_matrix

    try:
        spec = plan["task_spec"]
        affordance = spec["interaction_affordance"]
        threshold = spec["executable_opening_threshold"]
        if "joint_contact_path" not in affordance:
            # Legacy mechanisms keep the original whole-target check. Missing
            # contact geometry must never silently exempt them from coverage.
            return None
        path = affordance["joint_contact_path"]
        subjects = [row for row in plan["objects"] if row.get("task_subject") is True]
        if (affordance["affordance_digest"] != canonical_digest(affordance, digest_field="affordance_digest")
                or len(subjects) != 1 or subjects[0].get("object_type") != "ARTICULATION"
                or not isinstance(path, list) or not 2 <= len(path) <= 64
                or not math.isclose(float(path[0]["joint_positions"][threshold["target_joint_id"]]),
                                    float(threshold["reset_position"]), abs_tol=1e-9)):
            raise ValueError
        local = np.asarray(path[0]["contact_pose_asset_root"]["position_m"], dtype=float)
        radii = [float(affordance[key]) for key in ("arrival_tolerance_m", "sweep_clearance_m")]
        if local.shape != (3,) or not np.isfinite(local).all() or any(not math.isfinite(x) or x <= 0 for x in radii):
            raise ValueError
        pose = subjects[0]["pose_world"]
        world = pose_matrix(pose["position_world_m"], pose["orientation_xyzw"])
        return seal({"schema_version": "native_task_reset_contact_region.v1",
            "affordance_digest": affordance["affordance_digest"],
            "center_world_m": (world @ np.append(local, 1.))[:3].tolist(),
            "radius_m": max(radii), "radius_basis": "sealed_arrival_tolerance_and_sweep_clearance",
            "scope": "initial_contact_patch_only", "whole_asset_visibility_qualified": False}, "region_digest")
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("composition_articulated_contact_region_invalid") from exc


def contact_patch_mask(region, calibration, shape):
    """Project a conservative contact patch through the captured native camera."""
    import numpy as np
    from .native_task_camera_start_configuration import pose_matrix

    if region.get("region_digest") != canonical_digest(region, digest_field="region_digest"):
        raise ValueError("composition_contact_region_digest_invalid")
    try:
        intrinsics = np.asarray(calibration["intrinsic_matrix"], dtype=float)
        world = pose_matrix(calibration["position_world_m"], calibration["quaternion_world_opengl_xyzw"])
        center = np.asarray(region["center_world_m"], dtype=float)
        radius = float(region["radius_m"])
        if (intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all()
                or center.shape != (3,) or not np.isfinite(center).all()
                or not math.isfinite(radius) or radius <= 0
                or tuple(calibration["resolution_hw"]) != tuple(shape)
                or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0):
            raise ValueError
        local = world[:3, :3].T @ (center - world[:3, 3])
        depth = float(-local[2])
        if depth <= radius:
            raise ValueError
        uv = np.array([intrinsics[0, 2] + intrinsics[0, 0] * local[0] / depth,
                       intrinsics[1, 2] - intrinsics[1, 1] * local[1] / depth])
        # Use the near side of the radius, including off-axis perspective shift.
        radii = np.array([intrinsics[0, 0], intrinsics[1, 1]]) * radius / (depth - radius)
        radii *= 1. + np.abs(local[:2]) / depth
        yy, xx = np.indices(shape)
        mask = ((xx - uv[0]) / radii[0]) ** 2 + ((yy - uv[1]) / radii[1]) ** 2 <= 1.
        return mask, {"projected_center_uv": uv.tolist(), "projected_radii_px": radii.tolist(),
            "camera_calibration": calibration, "region_digest": region["region_digest"]}
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("composition_contact_camera_projection_invalid") from exc


def assess_composition_pixels(diagnostic, *, output_root, interaction_region=None, interaction_required=True):
    """Check opaque support coverage; this does not identify offending Gaussians."""
    import numpy as np

    blockers = list(diagnostic.get("blockers") or [])
    if diagnostic.get("status") != "captured":
        return {"passed": False, "blockers": blockers or ["composition_capture_incomplete"]}
    root = Path(output_root) / "pixel_comparison"
    mask = np.load(root / "native_mesh_target_semantic_mask.npy", allow_pickle=False)
    missing = np.load(root / "mesh_target_occluded_in_full.npy", allow_pickle=False)
    # Exclude a two-pixel silhouette band from the stopping predicate. Retain
    # its count separately; rasterization differences are not deletion evidence.
    padded = np.pad(mask, 2, constant_values=False)
    interior = np.ones_like(mask)
    for dy in range(5):
        for dx in range(5):
            interior &= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    interior_count = int(interior.sum())
    intrusion_count = int((missing & interior).sum())
    contact = None
    if interaction_region is None:
        if interior_count < 64:
            blockers.append("composition_support_interior_insufficient")
        if intrusion_count:
            blockers.append("composition_support_background_occlusion_requires_review")
    else:
        calibration = next(row["camera_calibration"] for row in diagnostic["passes"] if row["pass"] == "full")
        patch, projection = contact_patch_mask(interaction_region, calibration, mask.shape)
        selected = interior & patch
        contact_count = int(selected.sum())
        contact_intrusion = int((missing & selected).sum())
        contact = {**projection, "required": interaction_required,
            "native_contact_interior_pixels": contact_count,
            "contact_pixels_occluded_by_appearance": contact_intrusion,
            "outside_contact_pixels_occluded_by_appearance": int((missing & interior & ~patch).sum()),
            "passed": contact_count >= 64 and contact_intrusion == 0}
        if interaction_required:
            if contact_count < 64:
                blockers.append("composition_contact_interior_insufficient")
            if contact_intrusion:
                blockers.append("composition_contact_background_occlusion_requires_review")
    return {"passed": not blockers, "blockers": blockers,
        "scope": "initial_articulated_contact_patch" if contact is not None else "opaque_target_coverage",
        "interaction_visibility": contact,
        "whole_asset_visibility_qualified": interaction_region is None and not blockers,
        "minimum_support_interior_pixels": 64, "silhouette_margin_pixels": 2,
        "support_interior_pixels": interior_count,
        "interior_pixels_occluded_by_appearance": intrusion_count,
        "silhouette_pixels_occluded_by_appearance": int((missing & ~interior).sum()),
        "automatic_gaussian_deletion_authorized": False,
        "pixel_cause_proven": False}


def run_native_asset_composition_gate(*, built, plan, output_root, stage=None):
    """Use the already-warm renderer; change visibility only, with no physics step."""
    roles = {row.get("semantic_role") for row in plan.get("objects", [])}
    targets = []
    if "task_support" in roles:
        targets.append("task_support")
    if any(row.get("task_subject") is True or row.get("semantic_role") == "task_object"
           for row in plan.get("objects", [])):
        targets.append("task_object")
    if (plan.get("task_spec") or {}).get("surface_target") is not None:
        targets.append("task_target_marker")
    if "scene_appearance" not in roles or not targets:
        return seal({"schema_version": "native_task_asset_composition_gate.v1",
            "status": "not_applicable", "passed": True,
            "reason": "no_mesh_support_and_appearance_composition", "blockers": []})
    root = Path(output_root) / "native_asset_composition_gate"
    root.mkdir(parents=True, exist_ok=False)
    rows, blockers = [], []
    buffers_refreshed = False
    if stage is None:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
    env = built.env.unwrapped
    initial_step = env.sim.get_physics_step_count()
    try:
        from itertools import product
        contact_region = articulated_reset_contact_region(plan)
        for role, target_class in product(("external", "overview"), targets):
            if role not in built.camera_scene_names:
                blockers.append("composition_required_camera_missing:" + role)
                continue
            request = seal({"schema_version": REQUEST_SCHEMA, "passes": list(COVERAGE_PASSES),
                "camera_role": role, "target_semantic_class": target_class,
                "policy_queries_permitted": 0, "physics_steps_between_passes_permitted": 0,
                "source_asset_mutation_permitted": False, "render_refresh_count": 4,
                "resolved_scene_plan_digest": canonical_digest(plan)}, "request_digest")
            output = root / role / target_class
            result = run_composition_diagnostic(request, output_root=output,
                adapters=make_native_adapters(built=built, stage=stage, request=request))
            assessment = assess_composition_pixels(result, output_root=output,
                interaction_region=contact_region if target_class == "task_object" else None,
                interaction_required=role == "external")
            rows.append({"camera_role": role, "target_semantic_class": target_class, "assessment": assessment,
                "diagnostic": {"path": str(output / "composition_diagnostic.json"),
                    **file_record(output / "composition_diagnostic.json")},
                "diagnostic_digest": result["receipt_digest"]})
            blockers.extend(role + ":" + target_class + ":" + item for item in assessment["blockers"])
    except Exception as exc:
        blockers.append("composition_gate_capture_failed:" + type(exc).__name__ + ":" + str(exc))
    finally:
        # Diagnostic visibility is restored by its finally block. Also replace
        # the last mesh-only sensor buffers before a policy can observe them.
        try:
            for _ in range(4):
                env.sim.render()
                for name in built.camera_scene_names.values():
                    env.scene[name].update(0.0, force_recompute=True)
            buffers_refreshed = True
        except Exception as exc:
            blockers.append("composition_gate_buffer_restore_failed:" + type(exc).__name__ + ":" + str(exc))
    if env.sim.get_physics_step_count() != initial_step:
        blockers.append("composition_gate_physics_advanced")
    receipt = seal({"schema_version": "native_task_asset_composition_gate.v1",
        "status": "blocked" if blockers else "passed", "passed": not blockers,
        "scope": "task_required_reset_visibility_with_whole_asset_occlusion_diagnostics",
        "target_semantic_classes": targets,
        "resolved_scene_plan_digest": canonical_digest(plan), "views": rows,
        "policy_queries": 0, "physics_steps": env.sim.get_physics_step_count() - initial_step,
        "full_scene_sensor_buffers_refreshed": buffers_refreshed, "source_assets_mutated": False,
        "automatic_gaussian_deletion_authorized": False, "occlusion_cause_proven": False,
        "blockers": sorted(set(blockers))})
    (root / "composition_gate.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
