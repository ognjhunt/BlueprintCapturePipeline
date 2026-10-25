"""Bind qualified articulated geometry to the existing native contact program.

Both episode preparation and development diagnostics use these same USD joint
transforms. This derives geometry and command inputs, not robot contact proof.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_franka_action_math import grasp_orientation_contact_xyzw
from .paired_target_articulated_kinematic_path import (
    derive_usd_articulated_graph_contact_path,
)
from .paired_target_native_arena_request import (
    ARRIVAL_ORIENTATION_TOLERANCE_RAD,
    ARRIVAL_STABILITY_STEPS,
    ARRIVAL_TOLERANCE_M,
    DEFAULT_SWEEP_CLEARANCE_M,
    GRIPPER_DWELL_MAXIMUM_STEPS,
    GRIPPER_DWELL_MINIMUM_STEPS,
    MAX_JOINT_DELTA_RAD,
    MAX_JOINT_SETPOINT_LEAD_RAD,
    MOTION_MAXIMUM_STEPS,
    MOTION_MINIMUM_STEPS,
)
from .rigid_frame_transforms import quaternion_conjugate_xyzw, rotate_vector_xyzw


def materialize_articulated_construction_inputs(
    *,
    task_spec: Mapping[str, Any],
    definition: Mapping[str, Any],
    asset: Path,
    asset_digest: str,
    link_prim_paths: Mapping[str, str],
) -> dict[str, Any]:
    """Keep the native task, contact path, and state bindings consistent."""
    from pxr import Gf, Usd, UsdGeom

    spec = copy.deepcopy(dict(task_spec))
    affordance = spec["interaction_affordance"]
    graph = spec["articulation_graph"]
    stage = Usd.Stage.Open(str(asset))
    root = stage.GetDefaultPrim() if stage is not None else None
    if root is None or not root.IsValid():
        raise ValueError("articulated_construction_asset_invalid")
    outward = [-float(value) for value in affordance["approach_unit_asset_root"]]
    outward_stage = UsdGeom.XformCache().GetLocalToWorldTransform(root).TransformDir(
        Gf.Vec3d(*outward)
    )
    receipt = derive_usd_articulated_graph_contact_path(
        usd_path=asset,
        asset_sha256=asset_digest,
        graph=graph,
        contact_link_id=affordance["contact_link_id"],
        contact_local=affordance["contact_point_link_m"],
        approach_world=list(outward_stage),
        target_value=float(spec["executable_opening_threshold"]["success_interval"][0]),
        link_prim_paths=link_prim_paths,
        joint_prim_paths={
            row["joint_id"]: row["joint_prim_path"]
            for row in definition["task_joint_bindings"]
        },
    )
    inverse_contact = quaternion_conjugate_xyzw(
        receipt["joint_contact_path"][0]["contact_pose_asset_root"]["orientation_xyzw"]
    )
    interaction = {
        **affordance,
        "source_interaction_affordance_digest": affordance["affordance_digest"],
        "kinematic_path_receipt_digest": receipt["receipt_digest"],
        "joint_contact_path": receipt["joint_contact_path"],
        "gripper_approach_unit_asset_root": affordance["approach_unit_asset_root"],
        "approach_unit_asset_root": receipt["joint_contact_path"][0]["clearance_unit_asset_root"],
        "retreat_unit_asset_root": receipt["joint_contact_path"][-1]["clearance_unit_asset_root"],
        "gripper_orientation_contact_xyzw": grasp_orientation_contact_xyzw(
            approach_axis=rotate_vector_xyzw(
                inverse_contact, affordance["approach_unit_asset_root"]
            ),
            jaw_axis=rotate_vector_xyzw(inverse_contact, affordance["jaw_unit_asset_root"]),
        ),
        "sweep_clearance_m": DEFAULT_SWEEP_CLEARANCE_M,
        "arrival_tolerance_m": ARRIVAL_TOLERANCE_M,
        "arrival_orientation_tolerance_rad": ARRIVAL_ORIENTATION_TOLERANCE_RAD,
        "arrival_stability_steps": ARRIVAL_STABILITY_STEPS,
        "motion_minimum_steps": MOTION_MINIMUM_STEPS,
        "motion_maximum_steps": MOTION_MAXIMUM_STEPS,
        "gripper_dwell_minimum_steps": GRIPPER_DWELL_MINIMUM_STEPS,
        "gripper_dwell_maximum_steps": GRIPPER_DWELL_MAXIMUM_STEPS,
        "max_joint_delta_rad": MAX_JOINT_DELTA_RAD,
        "max_joint_setpoint_lead_rad": MAX_JOINT_SETPOINT_LEAD_RAD,
        "affordance_digest": "",
    }
    interaction["affordance_digest"] = canonical_digest(
        interaction, digest_field="affordance_digest"
    )
    spec["interaction_affordance"] = interaction
    state_binding = copy.deepcopy(dict(definition["task_state_binding"]))
    # Enriching a grasp changes its identity. Bind that exact derived affordance
    # without rewriting the qualified source adapter or its retained digest.
    state_binding["interaction_affordance_digest"] = interaction["affordance_digest"]
    return {
        "task_spec": spec,
        "task_state_binding": state_binding,
        "articulated_kinematic_path_receipt": receipt,
    }
