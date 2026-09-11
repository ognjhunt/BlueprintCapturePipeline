"""Fail closed on unreviewed background intrusion before a policy is loaded."""
from __future__ import annotations

import json
from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from .native_task_composition_diagnostic import (
    PASSES, REQUEST_SCHEMA, file_record, run_composition_diagnostic, seal,
)
from .native_task_composition_worker import make_native_adapters


def assess_composition_pixels(diagnostic, *, output_root):
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
    if interior_count < 64:
        blockers.append("composition_support_interior_insufficient")
    if intrusion_count:
        blockers.append("composition_support_background_occlusion_requires_review")
    return {"passed": not blockers, "blockers": blockers,
        "minimum_support_interior_pixels": 64, "silhouette_margin_pixels": 2,
        "support_interior_pixels": interior_count,
        "interior_pixels_occluded_by_appearance": intrusion_count,
        "silhouette_pixels_occluded_by_appearance": int((missing & ~interior).sum()),
        "automatic_gaussian_deletion_authorized": False,
        "pixel_cause_proven": False}


def run_native_asset_composition_gate(*, built, plan, output_root, stage=None):
    """Use the already-warm renderer; change visibility only, with no physics step."""
    roles = {row.get("semantic_role") for row in plan.get("objects", [])}
    if not {"scene_appearance", "task_support"}.issubset(roles):
        return seal({"schema_version": "native_task_asset_composition_gate.v1",
            "status": "not_applicable", "passed": True,
            "reason": "no_mesh_support_and_appearance_composition", "blockers": []})
    root = Path(output_root) / "native_asset_composition_gate"
    root.mkdir(parents=True, exist_ok=False)
    rows, blockers = [], []
    if stage is None:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
    env = built.env.unwrapped
    initial_step = env.sim.get_physics_step_count()
    try:
        for role in ("external", "overview"):
            if role not in built.camera_scene_names:
                blockers.append("composition_required_camera_missing:" + role)
                continue
            request = seal({"schema_version": REQUEST_SCHEMA, "passes": list(PASSES),
                "camera_role": role, "target_semantic_class": "task_support",
                "policy_queries_permitted": 0, "physics_steps_between_passes_permitted": 0,
                "source_asset_mutation_permitted": False, "render_refresh_count": 4,
                "resolved_scene_plan_digest": canonical_digest(plan)}, "request_digest")
            output = root / role
            result = run_composition_diagnostic(request, output_root=output,
                adapters=make_native_adapters(built=built, stage=stage, request=request))
            assessment = assess_composition_pixels(result, output_root=output)
            rows.append({"camera_role": role, "assessment": assessment,
                "diagnostic": {"path": str(output / "composition_diagnostic.json"),
                    **file_record(output / "composition_diagnostic.json")},
                "diagnostic_digest": result["receipt_digest"]})
            blockers.extend(role + ":" + item for item in assessment["blockers"])
    except Exception as exc:
        blockers.append("composition_gate_capture_failed:" + type(exc).__name__ + ":" + str(exc))
    finally:
        # Diagnostic visibility is restored by its finally block. Also replace
        # the last mesh-only sensor buffers before a policy can observe them.
        for _ in range(4):
            env.sim.render()
            for name in built.camera_scene_names.values():
                env.scene[name].update(0.0, force_recompute=True)
    if env.sim.get_physics_step_count() != initial_step:
        blockers.append("composition_gate_physics_advanced")
    receipt = seal({"schema_version": "native_task_asset_composition_gate.v1",
        "status": "blocked" if blockers else "passed", "passed": not blockers,
        "scope": "exact_reset_external_and_overview_support_coverage",
        "resolved_scene_plan_digest": canonical_digest(plan), "views": rows,
        "policy_queries": 0, "physics_steps": env.sim.get_physics_step_count() - initial_step,
        "full_scene_sensor_buffers_refreshed": True, "source_assets_mutated": False,
        "automatic_gaussian_deletion_authorized": False, "occlusion_cause_proven": False,
        "blockers": sorted(set(blockers))})
    (root / "composition_gate.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
