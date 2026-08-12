"""Native Isaac Lab/Arena runtime for the ADP-009D progressive micro-check.

This module is copied into an immutable provider bundle and executed with
``/isaac-sim/python.sh``.  Isaac imports intentionally happen only after
AppLauncher has started Kit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any


try:  # flat provider-bundle layout, where this file runs as a script
    from adp009d_approach_capture import (
        APPROACH_GRIPPER_BODY_NAMES,
        APPROACH_MAX_JOINT_STEP_RAD,
        APPROACH_MAX_OBJECT_DISPLACEMENT_M,
        APPROVED_CAN_TOP_ABOVE_SUPPORT_M,
        CAN_AXIS_XY_M,
        CAMERA_AIM_CAPTURE_FRAME_INDEX,
        CAMERA_AIM_MAX_STEPS,
        EPISODE_START_JOINT_TOLERANCE_RAD,
        EPISODE_START_OBJECT_OFFSET_TOLERANCE_M,
        EPISODE_START_RESTORE_MAX_STEPS,
        SUPPORT_HEIGHT_M,
        apply_rigid_offset,
        approach_waypoints_world,
        approved_can_visual_center_world,
        external_task_camera_offset_plan,
        pose_world_to_base,
        rigid_offset_in_body_frame,
        solve_live_rigid_mount_camera_aim_command,
        solve_rigid_mount_camera_aim,
        select_wrist_observable_episode_start,
        semantic_target_observability,
        summarize_wrist_approach_capture,
        validate_wrist_observable_episode_start_restore,
        world_to_base_rotation_row_major_xyzw,
    )
except ModuleNotFoundError:  # imported as part of the repository package
    from .adp009d_approach_capture import (
        APPROACH_GRIPPER_BODY_NAMES,
        APPROACH_MAX_JOINT_STEP_RAD,
        APPROACH_MAX_OBJECT_DISPLACEMENT_M,
        APPROVED_CAN_TOP_ABOVE_SUPPORT_M,
        CAN_AXIS_XY_M,
        CAMERA_AIM_CAPTURE_FRAME_INDEX,
        CAMERA_AIM_MAX_STEPS,
        EPISODE_START_JOINT_TOLERANCE_RAD,
        EPISODE_START_OBJECT_OFFSET_TOLERANCE_M,
        EPISODE_START_RESTORE_MAX_STEPS,
        SUPPORT_HEIGHT_M,
        apply_rigid_offset,
        approach_waypoints_world,
        approved_can_visual_center_world,
        external_task_camera_offset_plan,
        pose_world_to_base,
        rigid_offset_in_body_frame,
        solve_live_rigid_mount_camera_aim_command,
        solve_rigid_mount_camera_aim,
        select_wrist_observable_episode_start,
        semantic_target_observability,
        summarize_wrist_approach_capture,
        validate_wrist_observable_episode_start_restore,
        world_to_base_rotation_row_major_xyzw,
    )
try:  # flat provider-bundle layout, where this file runs as a script
    from adp009d_contact_envelope import (
        ContactEnvelopeError,
        contact_envelope_from_physx_sdf_settings,
    )
except ModuleNotFoundError:  # imported as part of the repository package
    from .adp009d_contact_envelope import (
        ContactEnvelopeError,
        contact_envelope_from_physx_sdf_settings,
    )
try:  # flat provider-bundle layout, where this file runs as a script
    from adp009d_hold_trace import (
        HOLD_TRACE_SCHEMA_VERSION,
        HoldTraceError,
        classify_arm_hold_trace,
        extract_arm_effort_limits,
        extract_arm_sample,
    )
except ModuleNotFoundError:  # imported as part of the repository package
    from .adp009d_hold_trace import (
        HOLD_TRACE_SCHEMA_VERSION,
        HoldTraceError,
        classify_arm_hold_trace,
        extract_arm_effort_limits,
        extract_arm_sample,
    )
try:  # flat provider-bundle layout, where this file runs as a script
    from adp009d_physics_backend_comparison import (
        DROID_FRANKA_ROBOTIQ_USD_DIGEST,
        DROID_FRANKA_ROBOTIQ_USD_URI,
        FRANKA_CORRECTED_DIAGONAL_INERTIA_KG_M2,
        FRANKA_SOURCE_DIAGONAL_INERTIA_KG_M2,
        FRANKA_SOURCE_MESH_SCALE,
        NEWTON_MAPPED_PHYSX_PROPERTY_NAMES,
        NEWTON_MAPPED_PHYSX_PROPERTY_PREFIXES,
        ROBOTIQ_BODY_MASSES_KG,
        build_backend_contact_configuration,
        build_backend_profile,
        build_newton_actuator_limit_mapping_contract,
        build_newton_robot_inertial_overlay_contract,
        normalize_physics_backend,
        validate_backend_probe,
        validate_backend_profile,
        validate_newton_dynamics_representable,
    )
except ModuleNotFoundError:  # imported as part of the repository package
    from .adp009d_physics_backend_comparison import (
        DROID_FRANKA_ROBOTIQ_USD_DIGEST,
        DROID_FRANKA_ROBOTIQ_USD_URI,
        FRANKA_CORRECTED_DIAGONAL_INERTIA_KG_M2,
        FRANKA_SOURCE_DIAGONAL_INERTIA_KG_M2,
        FRANKA_SOURCE_MESH_SCALE,
        NEWTON_MAPPED_PHYSX_PROPERTY_NAMES,
        NEWTON_MAPPED_PHYSX_PROPERTY_PREFIXES,
        ROBOTIQ_BODY_MASSES_KG,
        build_backend_contact_configuration,
        build_backend_profile,
        build_newton_actuator_limit_mapping_contract,
        build_newton_robot_inertial_overlay_contract,
        normalize_physics_backend,
        validate_backend_probe,
        validate_backend_profile,
        validate_newton_dynamics_representable,
    )

RESULT_NAME = "adp009d_native_microcheck.json"
EXPECTED_ASSETS = {
    "approved_can.usda": "sha256:61c2a03bef425803d82cc5ef24ced5b2ccb4160923c53bb10c6ad0e3f52532ec",
    "sage_collision.usd": "sha256:b265706c24f6a8ace3ee6743fd138583c4e21d83f61b99a06fd435e6ac2d6b41",
}
APPROVED_CAN_ADAPTER_FILENAME = "approved_can_physx_sdf_adapter.usda"
APPROVED_CAN_NEWTON_ADAPTER_FILENAME = "approved_can_newton_generic_adapter.usda"
TASK_COLLISION_DERIVATIVE_FILENAME = "sage_task_collision.usda"
TASK_COLLISION_MANIFEST_FILENAME = "sage_task_collision_manifest.json"
NEWTON_ROBOT_INERTIAL_OVERLAY_RECEIPT_FILENAME = (
    "newton_robot_inertial_overlay_receipt.json"
)
OVERVIEW_TASK_CAMERA_DISTANCE_M = 1.25
MIN_OVERVIEW_TASK_OBJECT_PIXELS = 80
# Aura authored as an Omniverse ParticleField of Gaussian surfels.  Rendered
# by the same omni.rtx that the standalone OVRTX lane wraps -- the v11 worker
# log shows omni.rtx mapping /rtx/rtpt/gaussian/* onto
# OmniRtxSettingsParticleFieldAPI and rtx.scenedb loading the surfel prim --
# so Isaac can render it directly rather than a second process rendering it
# and the two being composited afterward.
AURA_PARTICLEFIELD_FILENAME = "aura_ghost_removed_surflets.usd"
# Accepted appearance assets, in preference order.  NuRec first: Isaac renders
# that format natively -- an InteriorGS scene in it has been rendered with a
# full-size robot composited inside -- while the ParticleField authoring of the
# same field has never rendered correctly.  Both are kept so the two can be
# compared on one scene rather than by memory of separate runs.
AURA_APPEARANCE_FILENAMES = (
    ("aura_ghost_removed_appearance.usdz", "nurec_volume"),
    ("aura_ghost_removed_appearance.usd", "particlefield_gaussian_surflet"),
    ("aura_ghost_removed_appearance.usda", "particlefield_gaussian_surflet"),
    (AURA_PARTICLEFIELD_FILENAME, "particlefield_gaussian_surflet"),
)


def _load_runtime_backend_contract(
    runtime: Path, requested_backend: object
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Bind the CLI, sealed manifest, profile, and contact configuration."""

    backend = normalize_physics_backend(requested_backend)
    manifest_path = runtime / "adp_arena_provider_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("adp009d_provider_manifest_missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile = manifest.get("physics_backend_profile")
    if not isinstance(profile, dict) or validate_backend_profile(profile):
        raise RuntimeError("adp009d_backend_profile_invalid")
    if (
        manifest.get("physics_backend") != backend
        or profile != build_backend_profile(backend)
        or manifest.get("physics_backend_profile_digest")
        != profile.get("profile_digest")
        or manifest.get("backend_selected_at_simulation_construction") is not True
        or manifest.get("mid_run_backend_switch_allowed") is not False
    ):
        raise RuntimeError("adp009d_backend_manifest_binding_invalid")
    contact_configuration = build_backend_contact_configuration(backend)
    if manifest.get("backend_contact_configuration") != contact_configuration:
        raise RuntimeError("adp009d_backend_contact_configuration_invalid")
    return backend, profile, contact_configuration


def _policy_episode_blockers(
    *,
    candidate_ids: list[str],
    policy_episode: dict[str, Any] | None,
    policy_episode_error: str | None,
) -> list[str]:
    """Fail closed when scored object states cannot be attributed to a policy.

    The deterministic scorer may truthfully say ``never_moved`` even when the
    robot never responded to a nontrivial command.  That is useful harness
    evidence, but it is not an interpretable policy outcome and must prevent a
    top-level completed result.
    """

    if not candidate_ids:
        return []

    blockers: list[str] = []
    batches = list((policy_episode or {}).get("batches") or [])
    scored_batches = [batch for batch in batches if int(batch.get("episodes_scored") or 0) > 0]
    if not scored_batches:
        blockers.append("policy_episodes_requested_but_none_scored")
    else:
        uninterpretable = sum(
            int(batch.get("episodes_policy_outcome_uninterpretable") or 0)
            for batch in scored_batches
        )
        if uninterpretable:
            blockers.append(f"policy_episode_action_delivery_unverified:{uninterpretable}")
        media_incomplete = sum(
            int(batch.get("episodes_media_incomplete") or 0) for batch in scored_batches
        )
        if media_incomplete:
            blockers.append(f"policy_episode_media_incomplete:{media_incomplete}")
    if policy_episode_error:
        blockers.append(f"policy_episode_error:{policy_episode_error[:120]}")
    return sorted(set(blockers))


def _resolve_aura_appearance(runtime: Path) -> tuple[Path | None, str | None]:
    """The appearance asset that shipped, and what format it is."""

    for filename, kind in AURA_APPEARANCE_FILENAMES:
        candidate = runtime / "assets" / filename
        if candidate.is_file():
            return candidate, kind
    return None, None


AURA_PARTICLEFIELD_PRIM = "/World/AuraAppearance/GaussianSurflets"
APPROVED_CAN_ADAPTER_SHA256 = (
    "sha256:086199710beaeacea0d4894cc71b260f39a8357b562c8e6af298c924df11cc66"
)
APPROVED_CAN_SOURCE_COLLIDER_PRIM = "/canned_beverage/colliders/body_collider"
APPROVED_CAN_LIVE_COLLIDER_PRIM = "/World/envs/env_0/approved_can/colliders/body_collider"
SAGE_SOURCE_ROOT_PRIM = "/Root"
SAGE_LIVE_ROOT_PRIM = "/World/envs/env_0/sage_collision"
SAGE_TARGET_COLLIDER_NAME = "ZHQYGJJVAJYEYPTUKY888888"
SAGE_SUPPORT_COLLIDER_NAME = "_LTFTHJVAZ3VMPTUJU888888"
SAGE_RUNTIME_PROFILE = {
    "active_mesh_count": 15,
    "active_point_count": 80_484,
    "active_face_count": 26_828,
    "rigid_body_count": 0,
    "triangle_mesh_count": 15,
}
PHYSX_FALLBACK_MARKER = "falling back to convexHull approximation"
PHYSX_TRIANGLE_STABILITY_MARKER = "TriangleMesh: triangles are too big"
PHYSX_COLLISION_COOKING_PROFILE = "legacy_cooker_after_ujitso_stall.v1"
ARENA_REVISION = "8b4a3a47fc53de23e8205089d71109a2e2348acd"
ISAAC_LAB_REVISION = "e57379c634b42db5a0fe9f754341be6e2a7c7c43"
ROBOT_BASE_POSITION_M = (3.4681748, -2.8100837, 0.2766791)
ROBOT_BASE_YAW_RAD = -math.pi / 2
CAN_START_POSITION_M = (3.4681748, -3.3100837, 0.5264650138348479)
APPROVED_CAN_RADIUS_M = 0.031094726014345042
APPROVED_CAN_HEIGHT_M = 0.1694279937744141
# Read-only contact partner used to name what a stalled finger is touching.
# The filter names the can's rigid body, not its collider mesh: PhysX resolves
# filter patterns against rigid bodies, and `PhysicsRigidBodyAPI` is applied at
# the `canned_beverage` root while `colliders/body_collider` only carries
# `PhysicsCollisionAPI`.
CONTACT_PARTNER_FILTER_LABEL = "approved_can"
CONTACT_PARTNER_FILTER_PRIM_PATH = "{ENV_REGEX_NS}/approved_can"
# Canaries 47488171 and 47489958 proved that neither the PhysX spawn label
# ``approved_can`` nor the authored USD rigid-body name ``canned_beverage`` is
# retained as a Newton body label.  The sealed USD has exactly one authored can
# collision shape, ``/canned_beverage/colliders/body_collider``.  Newton's
# native contact adapter supports shape-level partner filtering, so bind that
# exact authored collider rather than guessing another converted body label.
NEWTON_CONTACT_PARTNER_FILTER_SHAPE_EXPR = "*body_collider"
# The retained paid controls receipt proved that the can filter resolved one
# shape and carried zero force while the left finger carried 8.6 N net force.
# That rules out the can, but not the sealed SAGE collision asset versus any
# other unfiltered source.  This separate, read-only scope makes that next
# distinction without changing collision, controller, or task geometry.
CONTACT_SAGE_COLLISION_FILTER_LABEL = "sage_collision"
CONTACT_SAGE_COLLISION_FILTER_PRIM_PATH = "{ENV_REGEX_NS}/sage_collision"
# SAGE is a static collection of collision shapes and deliberately has no
# rigid body.  Newton therefore needs shape-level filters.  These suffixes are
# the exact 15 active shapes in the digest-bound task-collision derivative;
# suffix globs work whether Newton labels shapes by bare name or full USD path.
NEWTON_SAGE_COLLISION_SHAPE_LABELS = (
    "SM_floorplan",
    "Z6TL2HRVAIIBIPTUKE888888",
    "ZBRQEFBVAI3DWPTUKY888888",
    "ZE6ZHARVAII2IPTUL4888888",
    "ZEMALJZVAJTQWPTUK4888888",
    "ZEO7DVBVAI7DEPTUKU888888",
    "ZEOP4DRVAIJFSPTUKE888888",
    "ZHQYBPJVAI3AUPTULE888888",
    "ZHQYGJJVAJYEYPTUK4888888",
    "ZV67OQJVAJSVCPTULY888888",
    "ZXXPXAZVAJ3T6PTULI888888",
    "_IMCHJBVAV7AMPTUKI888888",
    "_K7DXDRVAZU7IPTULI888888_004",
    "_LTFTHJVAZ3VMPTUJU888888",
    "_PROTIZVAJTMCPTULU888888",
)
NEWTON_SAGE_COLLISION_FILTER_SHAPE_EXPRS = tuple(
    f"*{label}" for label in NEWTON_SAGE_COLLISION_SHAPE_LABELS
)
# PhysX filtered contact reporting is strictly one-to-many: one sensor body may
# be filtered against many partners, never many sensor bodies against one.  The
# pinned IsaacLab docstring calls out this exact shape as unsupported, so each
# finger needs its own filtered sensor.  The unfiltered two-body sensor stays
# the primary net-force source and is unaffected.
CONTACT_PARTNER_SENSOR_NAMES = {
    "left_inner_finger": "robot_contact_can_left",
    "right_inner_finger": "robot_contact_can_right",
}
CONTACT_SAGE_COLLISION_SENSOR_NAMES = {
    "left_inner_finger": "robot_contact_sage_left",
    "right_inner_finger": "robot_contact_sage_right",
}


def _robot_contact_sensor_prim_path(physics_backend: str) -> str:
    """Return an equivalent two-finger selector in the backend's pattern syntax.

    PhysX resolves ``ContactSensorCfg.prim_path`` as a regular expression.  The
    pinned experimental Newton contact adapter converts only ``.*`` to a
    ``fnmatch`` glob; regex grouping and alternation remain literal characters.
    Terminal canary 47486783 therefore matched zero Newton bodies when given
    ``(left_inner_finger|right_inner_finger)``.  A suffix glob selects the same
    two terminal finger bodies in both bare-name and full-path Newton labels,
    without also selecting the ``*_inner_finger_knuckle`` bodies.
    """

    backend = normalize_physics_backend(physics_backend)
    base = "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/"
    if backend == "newton":
        return base + "*inner_finger"
    return base + "(left_inner_finger|right_inner_finger)"


def _contact_partner_filter_kwargs(physics_backend: str) -> dict[str, list[str]]:
    """Return the backend-native exact can contact-partner filter."""

    backend = normalize_physics_backend(physics_backend)
    if backend == "newton":
        return {
            "filter_shape_prim_expr": [NEWTON_CONTACT_PARTNER_FILTER_SHAPE_EXPR]
        }
    return {"filter_prim_paths_expr": [CONTACT_PARTNER_FILTER_PRIM_PATH]}


def _summarize_newton_contact_labels(
    body_labels: list[str] | tuple[str, ...],
    shape_labels: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Retain bounded converted labels needed to diagnose filter admission.

    Contact-sensor construction happens after Newton has finalized its model,
    so a failed selector must preserve what Newton actually named.  All body
    labels are retained up to a generous bounded ceiling; shape labels retain
    every can-relevant value plus deterministic head/tail samples.  This is
    read-only failure evidence and cannot change contact behavior.
    """

    bodies = [str(value) for value in body_labels]
    shapes = [str(value) for value in shape_labels]
    relevant_tokens = ("approved", "can", "beverage", "body_collider")

    def relevant(values: list[str]) -> list[str]:
        return [
            value
            for value in values
            if any(token in value.lower() for token in relevant_tokens)
        ]

    body_limit = 256
    shape_sample_limit = 64
    retained_bodies = bodies[:body_limit]
    if len(shapes) <= shape_sample_limit:
        shape_sample = shapes
    else:
        half = shape_sample_limit // 2
        shape_sample = shapes[:half] + shapes[-half:]
    return {
        "schema_version": "adp009d_newton_contact_label_diagnostics.v1",
        "body_label_count": len(bodies),
        "shape_label_count": len(shapes),
        "body_labels": retained_bodies,
        "body_labels_truncated": len(bodies) > body_limit,
        "can_relevant_body_labels": relevant(bodies),
        "can_relevant_shape_labels": relevant(shapes),
        "shape_label_sample": shape_sample,
        "shape_label_sample_truncated": len(shapes) > shape_sample_limit,
        "requested_can_shape_filter": NEWTON_CONTACT_PARTNER_FILTER_SHAPE_EXPR,
    }


def _newton_contact_label_diagnostics() -> dict[str, Any]:
    """Read the finalized Newton model labels after a sensor-build failure."""

    try:
        from isaaclab_newton.physics import NewtonManager

        model = NewtonManager.get_model()
        if model is None:
            return {
                "schema_version": "adp009d_newton_contact_label_diagnostics.v1",
                "status": "model_unavailable",
            }
        raw_body_labels = getattr(model, "body_label", None)
        raw_shape_labels = getattr(model, "shape_label", None)
        result = _summarize_newton_contact_labels(
            [] if raw_body_labels is None else list(raw_body_labels),
            [] if raw_shape_labels is None else list(raw_shape_labels),
        )
        result["status"] = "observed"
        return result
    except Exception as exc:  # noqa: BLE001 - diagnostics cannot mask the blocker
        return {
            "schema_version": "adp009d_newton_contact_label_diagnostics.v1",
            "status": "unavailable",
            "error_type": type(exc).__name__,
        }


def _sage_collision_filter_kwargs(physics_backend: str) -> dict[str, list[str]]:
    """Keep PhysX body filtering and Newton static-shape filtering distinct."""

    backend = normalize_physics_backend(physics_backend)
    if backend == "newton":
        return {
            "filter_shape_prim_expr": list(
                NEWTON_SAGE_COLLISION_FILTER_SHAPE_EXPRS
            )
        }
    return {"filter_prim_paths_expr": [CONTACT_SAGE_COLLISION_FILTER_PRIM_PATH]}


# The worker imports the episode adapter under a flattened module name, so this
# file cannot read the adapter's constant at module scope.  Mirrored here and
# pinned equal by test, because a silent drift would misreport how far the
# finger geometry reaches past the frame the planner steers.
FINGER_TOOL_FRAME_LOCAL_OFFSET_Z_M = 0.046
# Semantics are authored as a runtime spawn-config override so the sealed can
# and SAGE USD bytes are never mutated.  The exact override is emitted with the
# result and digest-bound, so a downstream composition can prove which labelling
# produced the retained segmentation.
SEMANTIC_OVERRIDE_LAYER: dict[str, Any] = {
    "authoring": "isaac_lab_spawn_semantic_tags_runtime_override",
    "sealed_source_usd_mutated": False,
    "tags": {
        "approved_can": [["class", "approved_can"]],
        "robot": [["class", "robot"]],
    },
}
RESET_JOINTS = (
    0.0,
    -0.628318530718,
    0.0,
    -2.513274122872,
    0.0,
    1.884955592154,
    0.0,
    0.104255385697,
    0.104152053595,
    -0.128436118364,
    0.125143155456,
    -0.071244180202,
    -0.080966427922,
)
RESET_JOINT_NAMES = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "finger_joint",
    "right_outer_knuckle_joint",
    "right_inner_finger_joint",
    "right_inner_finger_knuckle_joint",
    "left_inner_finger_knuckle_joint",
    "left_inner_finger_joint",
)
RESET_ARM_TOLERANCE_RAD = 1.0e-3
HOLD_ARM_TOLERANCE_RAD = 1.0e-2
CAN_HOLD_XY_TOLERANCE_M = 5.0e-3
CAN_HOLD_Z_TOLERANCE_M = 5.0e-3
CAN_HOLD_TILT_TOLERANCE_DEG = 2.0


class CanonicalPoseError(RuntimeError):
    """A typed reset/hold failure with exact per-joint diagnostics."""

    def __init__(self, blocker: str, diagnostics: dict[str, Any]) -> None:
        self.diagnostics = diagnostics
        super().__init__(f"{blocker}:maximum_error_rad={diagnostics['maximum_error_rad']:.9f}")


class CanonicalObjectStabilityError(RuntimeError):
    """A typed nominal-pose failure with exact object-pose diagnostics."""

    def __init__(self, diagnostics: dict[str, Any]) -> None:
        self.diagnostics = diagnostics
        super().__init__(
            "canonical_hold_object_pose_unstable:"
            f"xy_displacement_m={diagnostics['xy_displacement_m']:.9f}:"
            f"tilt_degrees={diagnostics['final_tilt_degrees']:.6f}"
        )


def _bind_canonical_joint_positions(embodiment: Any) -> None:
    """Replace Arena's regex defaults with one exact, non-overlapping map."""

    canonical_joint_positions = dict(zip(RESET_JOINT_NAMES, RESET_JOINTS, strict=True))
    robot = embodiment.scene_config.robot
    robot.init_state = robot.init_state.replace(joint_pos=canonical_joint_positions)


def _configure_deterministic_reset_events(embodiment: Any) -> None:
    """Write the measured collision-free arm/open-gripper pose without noise.

    Arena release/0.2.1 authors zero for every Robotiq closed-loop joint.  The
    v19 native probe measured that all-zero gripper state leaning the approved
    can by 6.188 degrees.  The last six values in ``RESET_JOINTS`` are the exact
    settled open-gripper state retained by the contact-stable v15 native probe.
    """

    embodiment.event_config.init_franka_arm_pose.params["default_pose"] = list(RESET_JOINTS)
    reset_writer = embodiment.event_config.randomize_franka_joint_state
    reset_writer.params["mean"] = 0.0
    reset_writer.params["std"] = 0.0


def _canonical_digest(
    value: dict[str, Any], *, digest_field: str | None = None
) -> str:
    """Digest matching ``decision_evidence_contracts.canonical_digest``."""

    normalized = dict(value)
    if digest_field:
        normalized.pop(digest_field, None)
    encoded = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _semantic_tags(role: str) -> list[tuple[str, str]]:
    return [tuple(tag) for tag in SEMANTIC_OVERRIDE_LAYER["tags"][role]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _newton_robot_inertial_target_blockers(
    observed: dict[str, dict[str, Any]], *, post_apply: bool
) -> list[str]:
    """Validate the exact Robotiq mass-overlay targets without importing Isaac.

    This pure seam keeps source-drift and unsupported-body behavior hermetic.
    The live collector below supplies USD observations before and after the
    session-layer overlay.
    """

    blockers: list[str] = []
    expected_names = set(ROBOTIQ_BODY_MASSES_KG)
    mass_tolerance_kg = float(
        build_newton_robot_inertial_overlay_contract()[
            "usd_float32_mass_roundtrip_tolerance_kg"
        ]
    )
    if set(observed) != expected_names:
        blockers.append("adp009d_newton_robot_inertial_body_set_invalid")
    for body_name in sorted(expected_names.intersection(observed)):
        row = observed[body_name]
        if row.get("rigid_body_api_applied") is not True:
            blockers.append(
                f"adp009d_newton_robot_inertial_rigid_body_missing:{body_name}"
            )
        collision_count = row.get("collision_shape_count")
        if (
            isinstance(collision_count, bool)
            or not isinstance(collision_count, int)
            or collision_count < 1
        ):
            blockers.append(
                f"adp009d_newton_robot_inertial_collision_missing:{body_name}"
            )
        authored_non_mass = any(
            row.get(field) is True
            for field in (
                "diagonal_inertia_authored",
                "center_of_mass_authored",
                "principal_axes_authored",
            )
        )
        if authored_non_mass:
            blockers.append(
                f"adp009d_newton_robot_inertial_unexpected_authored_frame_data:{body_name}"
            )
        if post_apply:
            mass = row.get("mass_kg")
            if (
                row.get("mass_api_applied") is not True
                or row.get("mass_authored") is not True
                or isinstance(mass, bool)
                or not isinstance(mass, (int, float))
                or not math.isclose(
                    float(mass),
                    float(ROBOTIQ_BODY_MASSES_KG[body_name]),
                    rel_tol=0.0,
                    abs_tol=mass_tolerance_kg,
                )
            ):
                blockers.append(
                    f"adp009d_newton_robot_inertial_mass_overlay_invalid:{body_name}"
                )
        elif row.get("mass_api_applied") is not False or row.get(
            "mass_authored"
        ) is not False:
            blockers.append(
                f"adp009d_newton_robot_inertial_source_mass_drifted:{body_name}"
            )
    return sorted(set(blockers))


def _inspect_newton_robot_inertial_targets(
    stage: Any, robot_root_prim_path: str
) -> dict[str, dict[str, Any]]:
    """Read the nine exact flattened-USD bodies and their collider coverage."""

    from pxr import Usd, UsdPhysics

    gripper_root = f"{robot_root_prim_path}/Gripper/Robotiq_2F_85"
    observed: dict[str, dict[str, Any]] = {}
    for body_name in sorted(ROBOTIQ_BODY_MASSES_KG):
        prim_path = f"{gripper_root}/{body_name}"
        prim = stage.GetPrimAtPath(prim_path)
        if not (prim and prim.IsValid()):
            observed[body_name] = {
                "prim_path": prim_path,
                "rigid_body_api_applied": False,
                "collision_shape_count": 0,
                "mass_api_applied": False,
                "mass_authored": False,
                "diagonal_inertia_authored": False,
                "center_of_mass_authored": False,
                "principal_axes_authored": False,
                "mass_kg": None,
            }
            continue
        collision_count = 0
        prim_range = Usd.PrimRange(prim)
        for descendant in prim_range:
            if descendant == prim:
                continue
            if descendant.HasAPI(UsdPhysics.RigidBodyAPI):
                prim_range.PruneChildren()
                continue
            if descendant.HasAPI(UsdPhysics.CollisionAPI):
                collision_count += 1
        has_mass_api = prim.HasAPI(UsdPhysics.MassAPI)
        mass_api = UsdPhysics.MassAPI(prim) if has_mass_api else None
        mass_authored = bool(
            mass_api and mass_api.GetMassAttr().HasAuthoredValue()
        )
        observed[body_name] = {
            "prim_path": prim_path,
            "rigid_body_api_applied": prim.HasAPI(UsdPhysics.RigidBodyAPI),
            "collision_shape_count": collision_count,
            "mass_api_applied": has_mass_api,
            "mass_authored": mass_authored,
            "diagonal_inertia_authored": bool(
                mass_api and mass_api.GetDiagonalInertiaAttr().HasAuthoredValue()
            ),
            "center_of_mass_authored": bool(
                mass_api and mass_api.GetCenterOfMassAttr().HasAuthoredValue()
            ),
            "principal_axes_authored": bool(
                mass_api and mass_api.GetPrincipalAxesAttr().HasAuthoredValue()
            ),
            "mass_kg": (
                float(mass_api.GetMassAttr().Get()) if mass_authored else None
            ),
        }
    return observed


def _inspect_newton_franka_inertia_targets(
    stage: Any, robot_root_prim_path: str
) -> dict[str, dict[str, Any]]:
    """Read the exact Franka link inertias and centimeter-scaled colliders."""

    from pxr import Usd, UsdGeom, UsdPhysics

    observed: dict[str, dict[str, Any]] = {}
    for body_name in sorted(FRANKA_SOURCE_DIAGONAL_INERTIA_KG_M2):
        prim_path = f"{robot_root_prim_path}/{body_name}"
        prim = stage.GetPrimAtPath(prim_path)
        if not (prim and prim.IsValid()):
            observed[body_name] = {
                "prim_path": prim_path,
                "rigid_body_api_applied": False,
                "mass_api_applied": False,
                "mass_authored": False,
                "mass_kg": None,
                "center_of_mass_authored": False,
                "center_of_mass": None,
                "diagonal_inertia_authored": False,
                "diagonal_inertia_kg_m2": None,
                "principal_axes_authored": False,
                "collision_mesh_count": 0,
                "collision_mesh_paths": [],
                "collision_mesh_scales": [],
            }
            continue
        collision_meshes: list[Any] = []
        for descendant in Usd.PrimRange(prim):
            if descendant == prim:
                continue
            if descendant.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            if descendant.IsA(UsdGeom.Mesh) and descendant.HasAPI(
                UsdPhysics.CollisionAPI
            ):
                collision_meshes.append(descendant)
        has_mass_api = prim.HasAPI(UsdPhysics.MassAPI)
        mass_api = UsdPhysics.MassAPI(prim) if has_mass_api else None
        mass_attr = mass_api.GetMassAttr() if mass_api else None
        center_attr = mass_api.GetCenterOfMassAttr() if mass_api else None
        inertia_attr = mass_api.GetDiagonalInertiaAttr() if mass_api else None
        axes_attr = mass_api.GetPrincipalAxesAttr() if mass_api else None
        mass_authored = bool(mass_attr and mass_attr.HasAuthoredValue())
        center_authored = bool(center_attr and center_attr.HasAuthoredValue())
        inertia_authored = bool(inertia_attr and inertia_attr.HasAuthoredValue())
        axes_authored = bool(axes_attr and axes_attr.HasAuthoredValue())
        observed[body_name] = {
            "prim_path": prim_path,
            "rigid_body_api_applied": prim.HasAPI(UsdPhysics.RigidBodyAPI),
            "mass_api_applied": has_mass_api,
            "mass_authored": mass_authored,
            "mass_kg": float(mass_attr.Get()) if mass_authored else None,
            "center_of_mass_authored": center_authored,
            "center_of_mass": (
                [float(value) for value in center_attr.Get()]
                if center_authored
                else None
            ),
            "diagonal_inertia_authored": inertia_authored,
            "diagonal_inertia_kg_m2": (
                [float(value) for value in inertia_attr.Get()]
                if inertia_authored
                else None
            ),
            "principal_axes_authored": axes_authored,
            "collision_mesh_count": len(collision_meshes),
            "collision_mesh_paths": [
                str(mesh.GetPath()) for mesh in collision_meshes
            ],
            "collision_mesh_scales": [
                [
                    float(value)
                    for value in mesh.GetAttribute("xformOp:scale").Get()
                ]
                if mesh.GetAttribute("xformOp:scale").HasAuthoredValue()
                else None
                for mesh in collision_meshes
            ],
        }
    return observed


def _newton_franka_inertia_target_blockers(
    observed: dict[str, dict[str, Any]], *, post_apply: bool
) -> list[str]:
    """Reject any drift around the exact asset-specific inertia conversion."""

    blockers: list[str] = []
    expected_names = set(FRANKA_SOURCE_DIAGONAL_INERTIA_KG_M2)
    conversion = build_newton_robot_inertial_overlay_contract()[
        "franka_inertia_unit_conversion"
    ]
    tolerance = float(
        conversion[
            "corrected_value_absolute_tolerance"
            if post_apply
            else "source_value_absolute_tolerance"
        ]
    )
    expected_inertias = (
        FRANKA_CORRECTED_DIAGONAL_INERTIA_KG_M2
        if post_apply
        else FRANKA_SOURCE_DIAGONAL_INERTIA_KG_M2
    )
    if set(observed) != expected_names:
        blockers.append("adp009d_newton_franka_inertia_body_set_invalid")
    for body_name in sorted(expected_names.intersection(observed)):
        row = observed[body_name]
        if row.get("rigid_body_api_applied") is not True:
            blockers.append(
                f"adp009d_newton_franka_rigid_body_missing:{body_name}"
            )
        mass = row.get("mass_kg")
        if (
            row.get("mass_api_applied") is not True
            or row.get("mass_authored") is not True
            or isinstance(mass, bool)
            or not isinstance(mass, (int, float))
            or not math.isfinite(float(mass))
            or float(mass) <= 0.0
        ):
            blockers.append(f"adp009d_newton_franka_mass_invalid:{body_name}")
        if row.get("center_of_mass_authored") is not True or row.get(
            "center_of_mass"
        ) != [0.0, 0.0, 0.0]:
            blockers.append(
                f"adp009d_newton_franka_center_of_mass_drifted:{body_name}"
            )
        if row.get("principal_axes_authored") is not False:
            blockers.append(
                f"adp009d_newton_franka_principal_axes_drifted:{body_name}"
            )
        inertia = row.get("diagonal_inertia_kg_m2")
        expected = expected_inertias[body_name]
        if (
            row.get("diagonal_inertia_authored") is not True
            or not isinstance(inertia, list)
            or len(inertia) != 3
            or any(
                isinstance(actual, bool)
                or not isinstance(actual, (int, float))
                or not math.isfinite(float(actual))
                or not math.isclose(
                    float(actual),
                    float(wanted),
                    rel_tol=0.0,
                    abs_tol=tolerance,
                )
                for actual, wanted in zip(inertia, expected, strict=True)
            )
        ):
            blockers.append(
                f"adp009d_newton_franka_diagonal_inertia_invalid:{body_name}"
            )
        expected_mesh_path = f"{row.get('prim_path')}/geometry/{body_name}"
        mesh_scales = row.get("collision_mesh_scales")
        mesh_scale_valid = (
            isinstance(mesh_scales, list)
            and len(mesh_scales) == 1
            and isinstance(mesh_scales[0], list)
            and len(mesh_scales[0]) == 3
            and all(
                isinstance(actual, (int, float))
                and not isinstance(actual, bool)
                and math.isclose(
                    float(actual),
                    FRANKA_SOURCE_MESH_SCALE,
                    rel_tol=0.0,
                    abs_tol=float(conversion["mesh_scale_absolute_tolerance"]),
                )
                for actual in mesh_scales[0]
            )
        )
        if (
            row.get("collision_mesh_count") != 1
            or row.get("collision_mesh_paths") != [expected_mesh_path]
            or not mesh_scale_valid
        ):
            blockers.append(
                f"adp009d_newton_franka_collision_mesh_drifted:{body_name}"
            )
    return sorted(set(blockers))


def _newton_physx_property_is_mapped(property_name: str) -> bool:
    """Whether the pinned Newton importer gives this PhysX property semantics."""

    return property_name in NEWTON_MAPPED_PHYSX_PROPERTY_NAMES or any(
        property_name.startswith(prefix)
        for prefix in NEWTON_MAPPED_PHYSX_PROPERTY_PREFIXES
    )


def _block_newton_unmapped_physx_properties(
    stage: Any, robot_root_prim_path: str
) -> dict[str, Any]:
    """Prevent every unrecognized PhysX value from reaching Newton silently."""

    from pxr import Usd

    robot_prim = stage.GetPrimAtPath(robot_root_prim_path)
    if not (robot_prim and robot_prim.IsValid()):
        raise RuntimeError("adp009d_newton_robot_root_missing")
    mapped: list[dict[str, str]] = []
    blocked: list[dict[str, str]] = []
    authored: list[dict[str, str]] = []
    for prim in Usd.PrimRange(robot_prim):
        for attribute in prim.GetAttributes():
            property_name = str(attribute.GetName())
            if (
                not property_name.lower().startswith("physx")
                or not attribute.HasAuthoredValue()
            ):
                continue
            authored.append(
                {
                    "prim_path": str(prim.GetPath()),
                    "property_name": property_name,
                }
            )
    # Blocking a property Newton cannot express does not make the two backends
    # comparable, it just changes the dynamics silently: dropping
    # ``disableGravity`` leaves PhysX with a weightless arm and Newton with a
    # full-weight one.  Refuse before the paid allocation does any work.
    representability = validate_newton_dynamics_representable(authored)
    if representability["status"] != "admitted":
        raise RuntimeError(representability["typed_blocker"])
    for prim in Usd.PrimRange(robot_prim):
        for attribute in prim.GetAttributes():
            property_name = str(attribute.GetName())
            if (
                not property_name.lower().startswith("physx")
                or not attribute.HasAuthoredValue()
            ):
                continue
            row = {
                "prim_path": str(prim.GetPath()),
                "property_name": property_name,
            }
            if _newton_physx_property_is_mapped(property_name):
                mapped.append(row)
            else:
                attribute.Block()
                blocked.append(row)
    remaining_unmapped: list[dict[str, str]] = []
    for prim in Usd.PrimRange(robot_prim):
        for attribute in prim.GetAttributes():
            property_name = str(attribute.GetName())
            if (
                property_name.lower().startswith("physx")
                and attribute.HasAuthoredValue()
                and not _newton_physx_property_is_mapped(property_name)
            ):
                remaining_unmapped.append(
                    {
                        "prim_path": str(prim.GetPath()),
                        "property_name": property_name,
                    }
                )
    if remaining_unmapped:
        raise RuntimeError("adp009d_newton_unmapped_physx_property_remained")
    return {
        "schema_version": "adp009d_newton_physx_property_admission_receipt.v1",
        "policy": "block_value_before_newton_model_import",
        "mapped_properties_retained": sorted(
            mapped, key=lambda row: (row["prim_path"], row["property_name"])
        ),
        "unmapped_properties_blocked": sorted(
            blocked, key=lambda row: (row["prim_path"], row["property_name"])
        ),
        "remaining_unmapped_authored_properties": [],
    }


def _apply_newton_robot_inertial_overlay(
    *, stage: Any, robot_root_prim_path: str, source_asset_digest: str
) -> dict[str, Any]:
    """Apply and verify the admitted Newton-only inertial session layer."""

    from pxr import Gf, UsdGeom, UsdPhysics

    contract = build_newton_robot_inertial_overlay_contract()
    if source_asset_digest != DROID_FRANKA_ROBOTIQ_USD_DIGEST:
        raise RuntimeError("adp009d_newton_robot_source_asset_digest_invalid")
    stage_meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not math.isclose(stage_meters_per_unit, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError("adp009d_newton_robot_stage_units_invalid")
    before = _inspect_newton_robot_inertial_targets(stage, robot_root_prim_path)
    blockers = _newton_robot_inertial_target_blockers(before, post_apply=False)
    if blockers:
        raise RuntimeError(
            "adp009d_newton_robot_inertial_source_invalid:" + ",".join(blockers)
        )
    franka_before = _inspect_newton_franka_inertia_targets(
        stage, robot_root_prim_path
    )
    blockers = _newton_franka_inertia_target_blockers(
        franka_before, post_apply=False
    )
    if blockers:
        raise RuntimeError(
            "adp009d_newton_franka_inertia_source_invalid:"
            + ",".join(blockers)
        )
    physx_property_admission = _block_newton_unmapped_physx_properties(
        stage, robot_root_prim_path
    )
    for body_name, mass_kg in sorted(ROBOTIQ_BODY_MASSES_KG.items()):
        prim = stage.GetPrimAtPath(
            f"{robot_root_prim_path}/Gripper/Robotiq_2F_85/{body_name}"
        )
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr().Set(float(mass_kg))
    for body_name, diagonal_inertia in sorted(
        FRANKA_CORRECTED_DIAGONAL_INERTIA_KG_M2.items()
    ):
        prim = stage.GetPrimAtPath(f"{robot_root_prim_path}/{body_name}")
        UsdPhysics.MassAPI(prim).CreateDiagonalInertiaAttr().Set(
            Gf.Vec3f(*diagonal_inertia)
        )
    after = _inspect_newton_robot_inertial_targets(stage, robot_root_prim_path)
    blockers = _newton_robot_inertial_target_blockers(after, post_apply=True)
    if blockers:
        raise RuntimeError(
            "adp009d_newton_robot_inertial_overlay_invalid:" + ",".join(blockers)
        )
    franka_after = _inspect_newton_franka_inertia_targets(
        stage, robot_root_prim_path
    )
    blockers = _newton_franka_inertia_target_blockers(
        franka_after, post_apply=True
    )
    for body_name in sorted(FRANKA_SOURCE_DIAGONAL_INERTIA_KG_M2):
        if franka_after.get(body_name, {}).get("mass_kg") != franka_before.get(
            body_name, {}
        ).get("mass_kg"):
            blockers.append(
                f"adp009d_newton_franka_mass_not_preserved:{body_name}"
            )
        if franka_after.get(body_name, {}).get(
            "center_of_mass"
        ) != franka_before.get(body_name, {}).get("center_of_mass"):
            blockers.append(
                f"adp009d_newton_franka_center_of_mass_not_preserved:{body_name}"
            )
    if blockers:
        raise RuntimeError(
            "adp009d_newton_franka_inertia_overlay_invalid:"
            + ",".join(sorted(set(blockers)))
        )
    receipt: dict[str, Any] = {
        "schema_version": "adp009d_newton_robot_inertial_overlay_receipt.v2",
        "status": "applied_and_verified",
        "physics_backend": "newton",
        "source_robot_asset_uri": DROID_FRANKA_ROBOTIQ_USD_URI,
        "source_robot_asset_digest": source_asset_digest,
        "overlay_contract_digest": contract["overlay_digest"],
        "robot_root_prim_path": robot_root_prim_path,
        "body_count": len(after),
        "body_observations": after,
        "franka_body_count": len(franka_after),
        "stage_meters_per_unit": stage_meters_per_unit,
        "franka_source_observations": franka_before,
        "franka_inertia_observations": franka_after,
        "physx_property_admission": physx_property_admission,
        "authored_properties": ["physics:diagonalInertia", "physics:mass"],
        "source_usd_mutated": False,
        "robotiq_center_of_mass_and_inertia_deferred_to_pinned_newton_importer": True,
        "franka_source_center_of_mass_preserved": True,
        "franka_diagonal_inertia_unit_conversion_applied": True,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = _canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    return receipt


def _validate_newton_robot_inertial_overlay_receipt(
    value: dict[str, Any], *, backend_profile: dict[str, Any]
) -> list[str]:
    """Validate the retained overlay before it can satisfy Newton admission."""

    blockers: list[str] = []
    overlay_contract = dict(
        (backend_profile.get("asset_conversion") or {}).get(
            "robot_inertial_overlay"
        )
        or {}
    )
    body_observations = value.get("body_observations")
    franka_source_observations = value.get("franka_source_observations")
    franka_inertia_observations = value.get("franka_inertia_observations")
    property_receipt = value.get("physx_property_admission")
    if (
        backend_profile.get("physics_backend") != "newton"
        or overlay_contract != build_newton_robot_inertial_overlay_contract()
        or value.get("schema_version")
        != "adp009d_newton_robot_inertial_overlay_receipt.v2"
        or value.get("status") != "applied_and_verified"
        or value.get("physics_backend") != "newton"
        or value.get("source_robot_asset_uri") != DROID_FRANKA_ROBOTIQ_USD_URI
        or value.get("source_robot_asset_digest")
        != DROID_FRANKA_ROBOTIQ_USD_DIGEST
        or value.get("overlay_contract_digest")
        != overlay_contract.get("overlay_digest")
        or value.get("body_count") != len(ROBOTIQ_BODY_MASSES_KG)
        or value.get("franka_body_count")
        != len(FRANKA_SOURCE_DIAGONAL_INERTIA_KG_M2)
        or value.get("stage_meters_per_unit")
        != overlay_contract.get("franka_inertia_unit_conversion", {}).get(
            "expected_stage_meters_per_unit"
        )
        or value.get("authored_properties")
        != ["physics:diagonalInertia", "physics:mass"]
        or value.get("source_usd_mutated") is not False
        or value.get(
            "robotiq_center_of_mass_and_inertia_deferred_to_pinned_newton_importer"
        )
        is not True
        or value.get("franka_source_center_of_mass_preserved") is not True
        or value.get("franka_diagonal_inertia_unit_conversion_applied")
        is not True
        or value.get("receipt_digest")
        != _canonical_digest(value, digest_field="receipt_digest")
    ):
        blockers.append("adp009d_newton_robot_inertial_overlay_receipt_invalid")
    if not isinstance(body_observations, dict):
        blockers.append("adp009d_newton_robot_inertial_overlay_receipt_invalid")
    else:
        blockers.extend(
            _newton_robot_inertial_target_blockers(
                body_observations, post_apply=True
            )
        )
    if not isinstance(franka_source_observations, dict):
        blockers.append("adp009d_newton_robot_inertial_overlay_receipt_invalid")
    else:
        blockers.extend(
            _newton_franka_inertia_target_blockers(
                franka_source_observations, post_apply=False
            )
        )
    if not isinstance(franka_inertia_observations, dict):
        blockers.append("adp009d_newton_robot_inertial_overlay_receipt_invalid")
    else:
        blockers.extend(
            _newton_franka_inertia_target_blockers(
                franka_inertia_observations, post_apply=True
            )
        )
    if isinstance(franka_source_observations, dict) and isinstance(
        franka_inertia_observations, dict
    ):
        for body_name in sorted(FRANKA_SOURCE_DIAGONAL_INERTIA_KG_M2):
            if franka_inertia_observations.get(body_name, {}).get(
                "mass_kg"
            ) != franka_source_observations.get(body_name, {}).get("mass_kg"):
                blockers.append(
                    f"adp009d_newton_franka_mass_not_preserved:{body_name}"
                )
            if franka_inertia_observations.get(body_name, {}).get(
                "center_of_mass"
            ) != franka_source_observations.get(body_name, {}).get(
                "center_of_mass"
            ):
                blockers.append(
                    f"adp009d_newton_franka_center_of_mass_not_preserved:{body_name}"
                )
    if not isinstance(property_receipt, dict):
        blockers.append("adp009d_newton_physx_property_admission_receipt_invalid")
    else:
        mapped = property_receipt.get("mapped_properties_retained")
        blocked = property_receipt.get("unmapped_properties_blocked")
        if (
            property_receipt.get("schema_version")
            != "adp009d_newton_physx_property_admission_receipt.v1"
            or property_receipt.get("policy")
            != "block_value_before_newton_model_import"
            or not isinstance(mapped, list)
            or not mapped
            or not isinstance(blocked, list)
            or not blocked
            or property_receipt.get("remaining_unmapped_authored_properties")
            != []
            or any(
                not isinstance(row, dict)
                or not _newton_physx_property_is_mapped(
                    str(row.get("property_name") or "")
                )
                for row in mapped
            )
            or any(
                not isinstance(row, dict)
                or _newton_physx_property_is_mapped(
                    str(row.get("property_name") or "")
                )
                for row in blocked
            )
        ):
            blockers.append(
                "adp009d_newton_physx_property_admission_receipt_invalid"
            )
    return sorted(set(blockers))


def _configure_newton_robot_inertial_overlay(
    embodiment: Any, *, output_dir: Path
) -> None:
    """Replace only Newton's DROID spawner with a digest-verifying wrapper."""

    from isaaclab.sim.utils import clone as clone_spawner
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab.utils.string import string_to_callable

    spawn_cfg = embodiment.scene_config.robot.spawn
    if spawn_cfg.usd_path != DROID_FRANKA_ROBOTIQ_USD_URI:
        raise RuntimeError("adp009d_newton_robot_source_asset_uri_invalid")
    if spawn_cfg.articulation_props is None or spawn_cfg.rigid_props is None:
        raise RuntimeError("adp009d_newton_robot_spawn_properties_missing")
    spawn_cfg.articulation_props = spawn_cfg.articulation_props.replace(
        solver_position_iteration_count=None,
        solver_velocity_iteration_count=None,
    )
    spawn_cfg.rigid_props = spawn_cfg.rigid_props.replace(
        max_depenetration_velocity=None,
        solver_position_iteration_count=None,
        solver_velocity_iteration_count=None,
    )
    # Newton's native sensor does not consume PhysxContactReportAPI.  Leaving
    # this Arena default enabled would add a PhysX-only API after the property
    # admission scan, outside the immutable Newton sensor configuration.
    spawn_cfg.activate_contact_sensors = False
    underlying_spawn = _resolve_newton_underlying_usd_spawn(
        spawn_cfg.func,
        string_to_callable=string_to_callable,
    )

    def spawn_with_inertial_overlay(
        prim_path: str,
        cfg: Any,
        translation: tuple[float, float, float] | None = None,
        orientation: tuple[float, float, float, float] | None = None,
        **kwargs: Any,
    ):
        if cfg.usd_path != DROID_FRANKA_ROBOTIQ_USD_URI:
            raise RuntimeError("adp009d_newton_robot_source_asset_uri_invalid")
        local_path = Path(retrieve_file_path(cfg.usd_path, force_download=False))
        source_digest = _sha256(local_path)
        if source_digest != DROID_FRANKA_ROBOTIQ_USD_DIGEST:
            raise RuntimeError("adp009d_newton_robot_source_asset_digest_invalid")
        local_cfg = cfg.copy()
        local_cfg.usd_path = str(local_path)
        prim = underlying_spawn(
            prim_path,
            local_cfg,
            translation=translation,
            orientation=orientation,
            **kwargs,
        )
        receipt = _apply_newton_robot_inertial_overlay(
            stage=prim.GetStage(),
            robot_root_prim_path=str(prim.GetPath()),
            source_asset_digest=source_digest,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / NEWTON_ROBOT_INERTIAL_OVERLAY_RECEIPT_FILENAME).write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return prim

    spawn_cfg.func = clone_spawner(spawn_with_inertial_overlay)


def _configure_newton_actuator_limit_mapping(
    embodiment: Any, *, backend_profile: dict[str, Any]
) -> dict[str, Any]:
    """Move the exact Arena actuator limits into Newton's active fields."""

    expected = build_newton_actuator_limit_mapping_contract()
    contract = backend_profile.get("actuator_limit_mapping")
    if contract != expected:
        raise RuntimeError("adp009d_newton_actuator_limit_mapping_contract_invalid")
    actuators = embodiment.scene_config.robot.actuators
    expected_actuators = expected["actuators"]
    if not isinstance(actuators, dict) or set(actuators) != set(expected_actuators):
        raise RuntimeError("adp009d_newton_actuator_set_invalid")
    observed: dict[str, dict[str, float | None]] = {}
    for name, values in expected_actuators.items():
        actuator = actuators[name]
        if (
            actuator.effort_limit != values["legacy_effort_limit"]
            or actuator.velocity_limit != values["legacy_velocity_limit"]
            or actuator.effort_limit_sim is not None
            or actuator.velocity_limit_sim is not None
        ):
            raise RuntimeError(f"adp009d_newton_actuator_source_limits_invalid:{name}")
        actuator.effort_limit = None
        actuator.velocity_limit = None
        actuator.effort_limit_sim = values["effort_limit_sim"]
        actuator.velocity_limit_sim = values["velocity_limit_sim"]
        observed[name] = {
            "effort_limit_sim": actuator.effort_limit_sim,
            "velocity_limit_sim": actuator.velocity_limit_sim,
        }
    receipt: dict[str, Any] = {
        "schema_version": "adp009d_newton_actuator_limit_mapping_receipt.v1",
        "status": "applied_and_verified",
        "physics_backend": "newton",
        "contract_digest": expected["mapping_digest"],
        "observed_sim_limits": observed,
        "legacy_fields_cleared": True,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = _canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    return receipt


def _resolve_newton_underlying_usd_spawn(
    configured_spawn: Any, *, string_to_callable: Any
) -> Any:
    """Resolve Isaac Lab's lazy callable, then admit only its pinned USD spawner.

    Isaac Lab 4.5.24 stores ``UsdFileCfg.func`` as a ``ResolvableString``.  That
    class intentionally suppresses generic dunder probing, including
    ``__wrapped__``, until its public string-to-callable resolver is used.
    Resolve only the exact expected target, verify its identity, and unwrap the
    official ``@clone`` decorator so our wrapper can apply the overlay before
    cloning.
    """

    expected_module = "isaaclab.sim.spawners.from_files.from_files"
    expected_name = "spawn_from_usd"
    expected_reference = f"{expected_module}:{expected_name}"
    resolved_spawn = configured_spawn
    if isinstance(configured_spawn, str):
        if str(configured_spawn) != expected_reference:
            raise RuntimeError("adp009d_newton_robot_spawn_wrapper_unsupported")
        resolved_spawn = string_to_callable(str(configured_spawn))
    if (
        not callable(resolved_spawn)
        or getattr(resolved_spawn, "__module__", None) != expected_module
        or getattr(resolved_spawn, "__name__", None) != expected_name
    ):
        raise RuntimeError("adp009d_newton_robot_spawn_wrapper_unsupported")
    underlying_spawn = getattr(resolved_spawn, "__wrapped__", None)
    if not callable(underlying_spawn):
        raise RuntimeError("adp009d_newton_robot_spawn_wrapper_unsupported")
    return underlying_spawn


def _to_torch(value: Any) -> Any:
    """Convert simulator-native arrays at the adapter boundary before indexing."""

    if hasattr(value, "detach"):
        return value
    value_module = type(value).__module__
    if value_module == "warp" or value_module.startswith("warp."):
        import warp as wp

        return wp.to_torch(value)
    raise TypeError(f"unsupported_sim_array:{value_module}.{type(value).__name__}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    value_module = type(value).__module__
    if value_module == "warp" or value_module.startswith("warp."):
        value = _to_torch(value)
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _json_safe(value: Any) -> Any:
    """Return simulator metadata in a deterministic JSON-compatible form."""

    return json.loads(json.dumps(_jsonable(value), default=str, sort_keys=True))


def _assert_arm_pose(
    actual: Any,
    expected: tuple[float, ...],
    *,
    tolerance_rad: float,
    blocker: str,
    hold_trace: dict[str, Any] | None = None,
) -> float:
    """Fail closed when the canonical seven-joint arm pose is not reached."""

    import torch

    actual_arm = _to_torch(actual)[:7]
    expected_arm = torch.tensor(expected[:7], device=actual_arm.device, dtype=actual_arm.dtype)
    absolute_error = torch.abs(actual_arm - expected_arm)
    maximum_error = float(torch.max(absolute_error).item())
    if not math.isfinite(maximum_error) or maximum_error > tolerance_rad:
        diagnostics = _canonical_pose_diagnostics(
            actual_arm=actual_arm,
            expected_arm=expected_arm,
            absolute_error=absolute_error,
            maximum_error=maximum_error,
            tolerance_rad=tolerance_rad,
        )
        if hold_trace is not None:
            # Without the trace this blocker is a single number that cannot
            # separate an arm still falling from one parked at a stable wrong
            # pose, and the run that produced it is already paid for.
            diagnostics["hold_trace"] = hold_trace
        raise CanonicalPoseError(blocker, diagnostics)
    return maximum_error


def _canonical_pose_diagnostics(
    *,
    actual_arm: Any,
    expected_arm: Any,
    absolute_error: Any,
    maximum_error: float,
    tolerance_rad: float,
) -> dict[str, Any]:
    return {
        "joint_names": list(RESET_JOINT_NAMES[:7]),
        "requested_joint_positions_rad": _jsonable(expected_arm),
        "observed_joint_positions_rad": _jsonable(actual_arm),
        "absolute_error_rad": _jsonable(absolute_error),
        "maximum_error_rad": maximum_error,
        "tolerance_rad": tolerance_rad,
    }


def _canonical_object_stability_diagnostics(
    initial_pose: Any,
    final_pose: Any,
) -> dict[str, Any]:
    initial = [float(item) for item in _jsonable(initial_pose)]
    final = [float(item) for item in _jsonable(final_pose)]
    if len(initial) != 7 or len(final) != 7:
        raise RuntimeError("canonical_object_pose_shape_invalid")
    delta = [final[index] - initial[index] for index in range(3)]
    xy_displacement = math.hypot(delta[0], delta[1])
    position_displacement = math.sqrt(sum(item * item for item in delta))
    qx, qy, _qz, _qw = final[3:7]
    world_up_alignment = max(-1.0, min(1.0, 1.0 - 2.0 * (qx * qx + qy * qy)))
    tilt_degrees = math.degrees(math.acos(world_up_alignment))
    return {
        "initial_pose_world": initial,
        "final_pose_world": final,
        "position_delta_m": delta,
        "position_displacement_m": position_displacement,
        "xy_displacement_m": xy_displacement,
        "absolute_z_displacement_m": abs(delta[2]),
        "final_tilt_degrees": tilt_degrees,
        "thresholds": {
            "xy_displacement_m": CAN_HOLD_XY_TOLERANCE_M,
            "absolute_z_displacement_m": CAN_HOLD_Z_TOLERANCE_M,
            "tilt_degrees": CAN_HOLD_TILT_TOLERANCE_DEG,
        },
    }


def _assert_canonical_object_stability(initial_pose: Any, final_pose: Any) -> dict[str, Any]:
    diagnostics = _canonical_object_stability_diagnostics(initial_pose, final_pose)
    thresholds = diagnostics["thresholds"]
    if (
        not all(
            math.isfinite(float(value))
            for key, value in diagnostics.items()
            if key
            in {
                "position_displacement_m",
                "xy_displacement_m",
                "absolute_z_displacement_m",
                "final_tilt_degrees",
            }
        )
        or diagnostics["xy_displacement_m"] > thresholds["xy_displacement_m"]
        or diagnostics["absolute_z_displacement_m"] > thresholds["absolute_z_displacement_m"]
        or diagnostics["final_tilt_degrees"] > thresholds["tilt_degrees"]
    ):
        raise CanonicalObjectStabilityError(diagnostics)
    return diagnostics


def _phase(name: str, status: str = "started") -> None:
    print(f"BLUEPRINT_WAM_RUNTIME_PHASE:adp009d_native:{name}:{status}", flush=True)


STOP_AFTER_FRAMES_ENV = "BLUEPRINT_ADP009D_STOP_AFTER_FRAMES"
# Above this, a frame has accumulated something.  At or below it the render
# never converged, whatever the reason.
CAMERA_RESOLUTION_ENV = "BLUEPRINT_ADP009D_CAMERA_RESOLUTION"
# Both frozen candidates consume far less than the 1280x720 the cameras
# rendered.  pi05_droid pads 1280x720 into 224x224, keeping 224x126 of content;
# groot_n17_droid keeps 320x180.  Rendering at 320x180 reproduces *both* of
# those exactly -- the pi05 pad from 320x180 also lands on 224x126 -- while
# drawing one sixteenth of the pixels.  Everything above that was rendered and
# then thrown away in the resize, and at roughly a minute per frame on a slow
# host that waste is what makes an episode take hours.
POLICY_CAMERA_RESOLUTION = (320, 180)
DIAGNOSTIC_CAMERA_RESOLUTION = (1280, 720)


def _camera_resolution() -> tuple[int, int]:
    """Render size, as (width, height)."""

    raw = os.environ.get(CAMERA_RESOLUTION_ENV, "").strip().lower()
    if not raw:
        return DIAGNOSTIC_CAMERA_RESOLUTION
    if raw == "policy":
        return POLICY_CAMERA_RESOLUTION
    try:
        width, height = (int(v) for v in raw.split("x", 1))
    except ValueError:
        return DIAGNOSTIC_CAMERA_RESOLUTION
    # A resolution below what a candidate consumes cannot be padded back up:
    # the policy would see a genuinely lower-detail scene than the contract
    # says, which is a silent change to the thing being evaluated.
    if width < POLICY_CAMERA_RESOLUTION[0] or height < POLICY_CAMERA_RESOLUTION[1]:
        return POLICY_CAMERA_RESOLUTION
    return (width, height)


MAX_GAUSSIANS_TO_ACCUMULATE_ENV = "BLUEPRINT_ADP009D_MAX_GAUSSIANS_TO_ACCUMULATE"
# Forty-eight was the shipped value and could not build a surface from a field
# whose median surfel is 0.81mm across in a 9.9m room.  This is a sweepable
# knob, not a proven value: no Omniverse render of this asset has succeeded
# yet, so the honest default is "far more than 48" rather than a number
# claiming authority it has not earned.
DEFAULT_MAX_GAUSSIANS_TO_ACCUMULATE = 1024


def _max_gaussians_to_accumulate() -> int:
    raw = os.environ.get(MAX_GAUSSIANS_TO_ACCUMULATE_ENV)
    if not raw:
        return DEFAULT_MAX_GAUSSIANS_TO_ACCUMULATE
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MAX_GAUSSIANS_TO_ACCUMULATE


FRAME_DEGENERATE_MAX_VALUE = 2
CAMERA_WARMUP_FRAMES_ENV = "BLUEPRINT_ADP009D_CAMERA_WARMUP_FRAMES"
# Forty frames is right for a bare scene whose frames cost milliseconds.  With
# the appearance composed each frame costs about sixty-five seconds, because
# every one of them waits the full omni.usd idle timeout, so the warmup alone
# would run past the paid TTL and the run would end having saved no frame at
# all.  A proof needs the camera settled, not forty frames of it.
DEFAULT_CAMERA_WARMUP_FRAMES = 40
# RTX accumulates samples across frames.  Four produced a frame with mean 0.2
# and max 1 -- the arm faintly outlined in black -- because the accumulator
# never converged, the same sample-starvation that once turned a 64spp render
# black where 384spp was clean.  This floor is the converged value, chosen
# from what actually renders rather than from what fits the wall clock.
MIN_CAMERA_WARMUP_FRAMES = 40


def _camera_warmup_frames() -> int:
    """Frames to settle the camera before the first saved frame."""

    raw = os.environ.get(CAMERA_WARMUP_FRAMES_ENV)
    if not raw:
        return DEFAULT_CAMERA_WARMUP_FRAMES
    try:
        requested = int(raw)
    except ValueError:
        return DEFAULT_CAMERA_WARMUP_FRAMES
    # Never fewer than the camera needs to settle: a frame saved from an
    # unsettled camera is worse than a slow run, because it looks like data.
    return max(MIN_CAMERA_WARMUP_FRAMES, requested)


FIRST_RENDER_BUDGET_SECONDS_ENV = "BLUEPRINT_ADP009D_FIRST_RENDER_BUDGET_SECONDS"
# Generous: the same scene without appearance renders its first frame in
# seconds.  A live run sat in this call for over twenty minutes emitting an
# omni.usd "failed to wait for idle" every seventy seconds and would have burnt
# the whole TTL to tell us nothing.
DEFAULT_FIRST_RENDER_BUDGET_SECONDS = 300.0


def _run_under_render_budget(call, *, phase_name: str, diagnostics: dict[str, Any]):
    """Run the first render under a hard budget, naming the failure if it blows.

    Isaac blocks inside native code, so a Python exception raised from a timer
    thread would never unwind it and a plain join would hang with it.  The only
    way to get a diagnosis out of a wedged renderer is to print it from a
    watchdog thread and hard-exit, which is what this does -- deliberately
    ``os._exit`` after flushing, because a normal exit path runs Omniverse
    shutdown handlers that are themselves blocked on the same idle wait.
    """

    import os
    import threading

    budget = float(
        os.environ.get(FIRST_RENDER_BUDGET_SECONDS_ENV) or DEFAULT_FIRST_RENDER_BUDGET_SECONDS
    )
    finished = threading.Event()

    def _watchdog() -> None:
        if finished.wait(budget):
            return
        _phase(phase_name, "blocked")
        print(
            f"BLUEPRINT_ADP009D_BLOCKER:first_render_budget_exceeded:{budget:.0f}s",
            flush=True,
        )
        print(
            "BLUEPRINT_ADP009D_FIRST_RENDER_DIAGNOSTICS:"
            + json.dumps(diagnostics, sort_keys=True, default=str),
            flush=True,
        )
        os._exit(93)

    watcher = threading.Thread(target=_watchdog, daemon=True)
    watcher.start()
    try:
        return call()
    finally:
        finished.set()


def _configure_physx_collision_cooking() -> dict[str, Any]:
    """Apply the documented UJITSO-stall diagnostic before scene construction."""

    import carb
    import omni.physx.bindings._physx as physx_bindings

    settings = carb.settings.get_settings()
    key = physx_bindings.SETTING_UJITSO_COLLISION_COOKING
    default_enabled = settings.get_as_bool(key)
    settings.set_bool(key, False)
    resolved_enabled = settings.get_as_bool(key)
    if resolved_enabled:
        raise RuntimeError("physx_legacy_collision_cooker_not_applied")
    return {
        "profile_id": PHYSX_COLLISION_COOKING_PROFILE,
        "setting_path": key,
        "ujitso_default_enabled": bool(default_enabled),
        "ujitso_resolved_enabled": bool(resolved_enabled),
        "cooker": "legacy",
        "reason": "measured_ujitso_environment_construction_stall_v14",
        "collider_geometry_or_parameters_changed": False,
    }


def _fail_on_physx_collision_fallback(messages: list[str]) -> None:
    if messages:
        raise RuntimeError("physx_collision_fallback_detected:" + " | ".join(messages))


def _fail_on_physx_collision_stability(messages: list[str]) -> None:
    if messages:
        raise RuntimeError("physx_collision_stability_warning_detected:" + " | ".join(messages))


def _inspect_physx_sdf_collider(stage: Any, prim_path: str) -> dict[str, Any]:
    from pxr import UsdPhysics

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"physx_sdf_collider_prim_missing:{prim_path}")
    applied_schemas = [str(value) for value in prim.GetAppliedSchemas()]
    if "PhysxSDFMeshCollisionAPI" not in applied_schemas:
        raise RuntimeError(f"physx_sdf_schema_missing:{prim_path}")
    mesh_api = UsdPhysics.MeshCollisionAPI(prim)
    approximation = mesh_api.GetApproximationAttr().Get() if mesh_api else None
    if str(approximation) != "sdf":
        raise RuntimeError(f"physx_sdf_approximation_invalid:{prim_path}:{approximation}")
    settings = {
        "sdf_margin": prim.GetAttribute("physxSDFMeshCollision:sdfMargin").Get(),
        "sdf_narrow_band_thickness": prim.GetAttribute(
            "physxSDFMeshCollision:sdfNarrowBandThickness"
        ).Get(),
        "sdf_resolution": prim.GetAttribute("physxSDFMeshCollision:sdfResolution").Get(),
        "sdf_subgrid_resolution": prim.GetAttribute(
            "physxSDFMeshCollision:sdfSubgridResolution"
        ).Get(),
    }
    if any(value is None for value in settings.values()):
        raise RuntimeError(f"physx_sdf_cooking_settings_missing:{prim_path}")
    try:
        contact_envelope = contact_envelope_from_physx_sdf_settings(
            sdf_margin_m=settings["sdf_margin"],
            sdf_narrow_band_thickness_m=settings["sdf_narrow_band_thickness"],
            sdf_resolution=settings["sdf_resolution"],
            sdf_subgrid_resolution=settings["sdf_subgrid_resolution"],
        )
    except ContactEnvelopeError as exc:
        raise RuntimeError(str(exc)) from exc
    return {
        "prim_path": prim_path,
        "applied_schemas": applied_schemas,
        "approximation": str(approximation),
        **settings,
        "contact_envelope": contact_envelope,
    }


def _inspect_sage_static_triangle_colliders(
    stage: Any,
    root_prim_path: str,
    *,
    expected_profile: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Require the composed SAGE runtime layer to remain exact static triangles."""

    from pxr import Usd, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(root_prim_path)
    if not root.IsValid() or not root.IsActive():
        raise RuntimeError(f"sage_runtime_root_missing:{root_prim_path}")
    target_path = f"{root_prim_path}/{SAGE_TARGET_COLLIDER_NAME}"
    support_path = f"{root_prim_path}/{SAGE_SUPPORT_COLLIDER_NAME}"
    target = stage.GetPrimAtPath(target_path)
    support = stage.GetPrimAtPath(support_path)
    if not target.IsValid() or target.IsActive():
        raise RuntimeError(f"sage_source_target_collider_not_disabled:{target_path}")
    if not support.IsValid() or not support.IsActive() or not support.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"sage_support_collider_missing:{support_path}")

    point_count = 0
    face_count = 0
    mesh_paths: list[str] = []
    rigid_body_paths: list[str] = []
    non_triangle_paths: list[str] = []
    for prim in Usd.PrimRange(root):
        schemas = {str(value) for value in prim.GetAppliedSchemas()}
        if "PhysicsRigidBodyAPI" in schemas:
            rigid_body_paths.append(str(prim.GetPath()))
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path = str(prim.GetPath())
        if "PhysicsCollisionAPI" not in schemas or "PhysicsMeshCollisionAPI" not in schemas:
            raise RuntimeError(f"sage_runtime_collision_schema_missing:{path}")
        approximation = str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())
        if approximation != "none":
            non_triangle_paths.append(f"{path}:{approximation}")
        mesh = UsdGeom.Mesh(prim)
        mesh_paths.append(path)
        point_count += len(mesh.GetPointsAttr().Get() or [])
        face_count += len(mesh.GetFaceVertexCountsAttr().Get() or [])

    if rigid_body_paths:
        raise RuntimeError(
            "sage_runtime_static_collision_has_rigid_body:" + ",".join(rigid_body_paths)
        )
    if non_triangle_paths:
        raise RuntimeError(
            "sage_runtime_triangle_mesh_override_missing:" + ",".join(non_triangle_paths)
        )
    observed = {
        "active_mesh_count": len(mesh_paths),
        "active_point_count": point_count,
        "active_face_count": face_count,
        "rigid_body_count": len(rigid_body_paths),
        "triangle_mesh_count": len(mesh_paths) - len(non_triangle_paths),
    }
    required_profile = expected_profile or SAGE_RUNTIME_PROFILE
    if observed != required_profile:
        raise RuntimeError(f"sage_runtime_collision_profile_mismatch:{observed}")
    return {
        **observed,
        "root_prim": root_prim_path,
        "target_collider_prim": target_path,
        "target_collider_active": False,
        "support_collider_prim": support_path,
        "support_collider_active": True,
        "approximation": "none",
        "approximation_semantics": "static_triangle_mesh",
        "sealed_source_mutated": False,
    }


def _camera_prim_diagnostics(camera: Any) -> dict[str, Any]:
    """Read the camera prim's world transform straight from the live USD stage.

    A recorded pose that never changes while the view does has exactly two
    causes: the sensor's pose buffer is not refreshed, or the prim is not
    parented to the hand.  The reported pose alone cannot tell them apart, and
    they need opposite repairs.  The stage transform for the same prim can, so
    it is collected on every capture rather than only after a run looks wrong.

    Diagnostics must never fail a capture: any error is recorded, not raised.
    """

    diagnostics: dict[str, Any] = {
        "configured_prim_path": None,
        "resolved_prim_path": None,
        "prim_exists": False,
        "usd_world_translation_m": None,
        "error": None,
    }
    try:
        diagnostics["configured_prim_path"] = getattr(
            getattr(camera, "cfg", None), "prim_path", None
        )
        resolved = None
        prim_paths = getattr(getattr(camera, "_view", None), "prim_paths", None)
        if prim_paths:
            resolved = str(prim_paths[0])
        elif diagnostics["configured_prim_path"]:
            # Arena replicates cameras per environment; this run has exactly one.
            resolved = str(diagnostics["configured_prim_path"]).replace("env_.*", "env_0")
        diagnostics["resolved_prim_path"] = resolved
        if resolved:
            import omni.usd
            from pxr import Usd, UsdGeom

            prim = omni.usd.get_context().get_stage().GetPrimAtPath(resolved)
            diagnostics["prim_exists"] = bool(prim and prim.IsValid())
            if diagnostics["prim_exists"]:
                translation = (
                    UsdGeom.Xformable(prim)
                    .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    .ExtractTranslation()
                )
                diagnostics["usd_world_translation_m"] = [
                    float(translation[0]),
                    float(translation[1]),
                    float(translation[2]),
                ]
    except Exception as exc:  # noqa: BLE001 - diagnostics must not break a capture
        diagnostics["error"] = f"{type(exc).__name__}:{exc}"
    return diagnostics


def _save_camera(
    output: Path,
    name: str,
    camera: Any,
    *,
    frame_index: int,
    sim_time: float,
    require_metric_depth: bool = True,
    pose_override: tuple[list[float], list[float]] | None = None,
    pose_source: str = "isaac_sensor_buffer",
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    camera_output = camera.data.output
    required = {"rgb", "semantic_segmentation"}
    if require_metric_depth:
        required.add("distance_to_camera")
    missing = sorted(required - set(camera_output))
    if missing:
        raise RuntimeError(f"camera_outputs_missing:{name}:{','.join(missing)}")
    rgb = _to_torch(camera_output["rgb"])[0].detach().cpu().numpy()
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    rgb = np.asarray(rgb, dtype=np.uint8)
    # A frame this dark is not a dark scene, and saving one produces evidence
    # that looks like data.  Deliberately named for the symptom rather than a
    # cause: it was first called sample_starved, and then forty warmup frames
    # produced max 1 / mean 0.167 where four had produced max 1 / mean 0.2,
    # which disproved starvation outright.  What the two black runs share is
    # the appearance field composing; the run before it, with the field
    # resolving to nothing, was mean 227.  The threshold sits far below any
    # real render, so this cannot reject a legitimately dim scene.
    if int(rgb.max()) <= FRAME_DEGENERATE_MAX_VALUE:
        raise RuntimeError(
            f"camera_frame_degenerate:{name}:max={int(rgb.max())}:mean={float(rgb.mean()):.3f}"
        )
    depth = None
    if "distance_to_camera" in camera_output:
        depth = (
            _to_torch(camera_output["distance_to_camera"])[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    semantic = _to_torch(camera_output["semantic_segmentation"])[0].detach().cpu().numpy()
    if semantic.ndim == 3 and semantic.shape[-1] == 1:
        semantic = semantic[..., 0]
    semantic = semantic.astype(np.int32)
    if rgb.shape[:2] != semantic.shape[:2] or (
        depth is not None and rgb.shape[:2] != depth.shape[:2]
    ):
        raise RuntimeError(f"camera_output_shape_mismatch:{name}")
    finite_depth = np.isfinite(depth) if depth is not None else None
    metric_depth_valid = bool(
        finite_depth is not None
        and finite_depth.any()
        and not (depth[finite_depth] < 0.0).any()
    )
    if require_metric_depth and not metric_depth_valid:
        raise RuntimeError(f"camera_metric_depth_invalid:{name}")
    semantic_ids, semantic_counts = np.unique(semantic, return_counts=True)
    semantic_pixel_counts = {
        str(int(label)): int(count)
        for label, count in zip(semantic_ids, semantic_counts, strict=True)
    }
    semantic_info = _json_safe((camera.data.info or {}).get("semantic_segmentation"))
    camera_dir = output / "camera_frames" / name
    camera_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = camera_dir / f"{frame_index:06d}.png"
    depth_path = camera_dir / f"{frame_index:06d}.distance_to_camera.npy"
    semantic_path = camera_dir / f"{frame_index:06d}.semantic.npy"
    Image.fromarray(rgb, mode="RGB").save(rgb_path, format="PNG", compress_level=9)
    if metric_depth_valid and depth is not None:
        np.save(depth_path, depth, allow_pickle=False)
    np.save(semantic_path, semantic, allow_pickle=False)
    intrinsic = _to_torch(camera.data.intrinsic_matrices)[0]
    if pose_override is None:
        pos_w = _jsonable(_to_torch(camera.data.pos_w)[0])
        quat_w_opengl = _jsonable(_to_torch(camera.data.quat_w_opengl)[0])
    else:
        pos_w = [float(value) for value in pose_override[0]]
        quat_w_opengl = [float(value) for value in pose_override[1]]
    prim_diagnostics = _camera_prim_diagnostics(camera)
    prim_diagnostics.update(
        {
            "evidence_pose_source": pose_source,
            "evidence_world_translation_m": list(pos_w),
        }
    )
    return {
        "camera_id": name,
        "frame_index": frame_index,
        "sim_time_seconds": sim_time,
        "timestamp_ns": time.time_ns(),
        "resolution_hw": [int(rgb.shape[0]), int(rgb.shape[1])],
        "rgb_png": {"path": str(rgb_path.relative_to(output)), "sha256": _sha256(rgb_path)},
        "metric_depth": (
            {
                "status": "valid",
                "aov": "distance_to_camera",
                "units": "meter",
                "path": str(depth_path.relative_to(output)),
                "sha256": _sha256(depth_path),
            }
            if metric_depth_valid and depth is not None
            else {
                "status": "not_required_for_review_only_camera",
                "aov": None,
                "units": None,
                "path": None,
                "sha256": None,
            }
        ),
        "semantic_segmentation": {
            "path": str(semantic_path.relative_to(output)),
            "sha256": _sha256(semantic_path),
            "dtype": str(semantic.dtype),
            "id_to_labels": semantic_info,
            "pixel_counts_by_id": semantic_pixel_counts,
        },
        "quality_diagnostics": {
            "finite_metric_depth_fraction": (
                float(finite_depth.mean()) if finite_depth is not None else None
            ),
            "metric_depth_required": bool(require_metric_depth),
            "rgb_min": int(rgb.min()),
            "rgb_max": int(rgb.max()),
            "rgb_mean": float(rgb.mean()),
            "foreground_semantic_pixel_fraction": float((semantic > 1).mean()),
        },
        "intrinsic_matrix": _jsonable(intrinsic),
        "position_world_m": list(pos_w),
        "quaternion_world_opengl_xyzw": list(quat_w_opengl),
        "pose_source": pose_source,
        "prim_diagnostics": prim_diagnostics,
        "device": str(camera.data.output["rgb"].device),
        "dlpack_ownership": "isaac_camera_tensor_read_only_copy_retained",
        "synchronization": "environment_step_completed_before_copy",
    }


def _approved_can_observability(camera: Any) -> dict[str, Any]:
    """Measure approved-can area and framing in the current semantic AOV."""

    output = camera.data.output
    if "semantic_segmentation" not in output:
        raise RuntimeError("wrist_episode_start_semantic_output_missing")
    semantic = _to_torch(output["semantic_segmentation"])[0]
    if hasattr(semantic, "detach"):
        semantic = semantic.detach().cpu().numpy()
    semantic_info = _json_safe((camera.data.info or {}).get("semantic_segmentation"))
    labels = (semantic_info or {}).get("idToLabels") or {}
    return semantic_target_observability(
        semantic_ids=semantic,
        id_to_labels=labels,
        target_label="approved_can",
    )


def _probe_finger_collision_envelope() -> dict[str, Any]:
    """Measure each finger's geometry extent in its own body frame.

    Arena's ``tool_leftfinger``/``tool_rightfinger`` frames are a +46 mm semantic
    point along the finger's local Z, which the descend planner treats as the
    fingertip.  That is not the collision extent, and the difference is what
    decides whether a commanded descend is geometrically reachable.  Reported as
    measurement only: it names no obstruction and changes no motion.
    """

    result: dict[str, Any] = {
        "schema_version": "adp009d_finger_collision_envelope_probe.v1",
        "status": "unavailable",
        "tool_frame_local_offset_m": FINGER_TOOL_FRAME_LOCAL_OFFSET_Z_M,
        "fingers": {},
    }
    try:
        import omni.usd
        from pxr import Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        )
        for body_name in sorted(CONTACT_PARTNER_SENSOR_NAMES):
            path = f"/World/envs/env_0/Robot/Gripper/Robotiq_2F_85/{body_name}"
            prim = stage.GetPrimAtPath(path)
            if not (prim and prim.IsValid()):
                result["fingers"][body_name] = {"prim_exists": False}
                continue
            # World space, not local: the first run of this probe reported a
            # 586 mm reach and 285 mm half-width for a Robotiq 2F-85 finger
            # whose whole gripper is ~160 mm, because ComputeLocalBound's frame
            # semantics do not mean "extent from this body's origin".  World
            # bound minus the body's own world origin is unambiguous, and the
            # body quaternion is retained so the analysis can rotate into the
            # tool frame without this probe assuming an axis convention.
            aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            minimum = [float(value) for value in aligned.GetMin()]
            maximum = [float(value) for value in aligned.GetMax()]
            transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            origin = transform.ExtractTranslation()
            body_origin = [float(origin[index]) for index in range(3)]
            rotation = transform.ExtractRotationQuat()
            imaginary = rotation.GetImaginary()
            extent = [maximum[index] - minimum[index] for index in range(3)]
            result["fingers"][body_name] = {
                "prim_exists": True,
                "prim_path": path,
                "world_bound_min_m": minimum,
                "world_bound_max_m": maximum,
                "body_origin_world_m": body_origin,
                "body_orientation_world_xyzw": [
                    float(imaginary[0]),
                    float(imaginary[1]),
                    float(imaginary[2]),
                    float(rotation.GetReal()),
                ],
                # Raw geometry facts only.  A finger whose largest extent is far
                # bigger than the gripper is evidence the bound covers more than
                # the finger, so the numbers stay interpretable when wrong.
                "bound_extent_m": extent,
                "largest_extent_m": max(extent),
                "reach_below_body_origin_m": body_origin[2] - minimum[2],
                "reach_above_body_origin_m": maximum[2] - body_origin[2],
            }
        if any(row.get("prim_exists") for row in result["fingers"].values()):
            result["status"] = "measured"
    except Exception as exc:  # noqa: BLE001 - diagnostics must not break a paid run
        result["error_type"] = type(exc).__name__
    return result


def _build_environment(runtime: Path, args: argparse.Namespace):
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.sensors.contact_sensor import ContactSensorCfg
    from isaaclab_arena.assets.asset import Asset
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.utils.pose import Pose

    # Shape-level filtering is intentionally a Newton-only extension.  The
    # PhysX lane keeps the existing factory configuration and native readback.
    BackendContactSensorCfg = ContactSensorCfg
    if args.physics_backend == "newton":
        from isaaclab_newton.sensors import (
            ContactSensorCfg as NewtonContactSensorCfg,
        )

        BackendContactSensorCfg = NewtonContactSensorCfg

    class SpawnerObject(Object):
        """Use Arena's composition seam without importing its full asset registry."""

        def __init__(self, *, name: str, prim_path: str, spawner_cfg: Any):
            self.spawner_cfg = spawner_cfg
            super().__init__(
                name=name,
                prim_path=prim_path,
                object_type=ObjectType.SPAWNER,
            )

    class ContactSensorAsset(Asset):
        """Compose one read-only Isaac sensor through Arena's asset seam."""

        def __init__(self, *, name: str, sensor_cfg: Any):
            super().__init__(name=name)
            self.sensor_cfg = sensor_cfg

        def get_object_cfg(self):
            return self.name, self.sensor_cfg

        def get_event_cfg(self):
            return self.name, None

    _phase("embodiment_configuration")
    yaw_half = ROBOT_BASE_YAW_RAD / 2
    robot_pose = Pose(
        position_xyz=ROBOT_BASE_POSITION_M,
        rotation_xyzw=(0.0, 0.0, math.sin(yaw_half), math.cos(yaw_half)),
    )
    embodiment = DroidAbsoluteJointPositionEmbodiment(
        enable_cameras=True,
        initial_pose=robot_pose,
        initial_joint_pose=list(RESET_JOINTS),
    )
    embodiment.scene_config.robot.spawn.semantic_tags = _semantic_tags("robot")
    # The canonical anchor is immutable: retain Arena's second reset event as
    # the state writer, but set its Gaussian mean and standard deviation to
    # zero.  Arena's first event updates only the default-joint buffer; without
    # the writer event, the live articulation remains at its stock pose.
    _configure_deterministic_reset_events(embodiment)
    # Apply the official pose helper while its stock stand still exists, then remove
    # that scene-specific stand.  The robot base is supported by sealed SAGE geometry.
    embodiment.get_scene_cfg()
    embodiment.scene_config.stand = None
    embodiment.initial_pose = None
    # Arena release/0.2.1's set_initial_joint_pose updates only its reset event.
    # Replace the authored regex defaults rather than updating them: exact
    # gripper names overlap right_outer.* / left_inner.* / right_inner.* and
    # Isaac Lab correctly rejects such an ambiguous mapping.
    _bind_canonical_joint_positions(embodiment)
    if args.physics_backend == "newton":
        _configure_newton_robot_inertial_overlay(
            embodiment,
            output_dir=Path(args.output_dir).resolve(),
        )
        newton_actuator_limit_mapping = _configure_newton_actuator_limit_mapping(
            embodiment,
            backend_profile=build_backend_profile("newton"),
        )
    else:
        newton_actuator_limit_mapping = None
    render_width, render_height = _camera_resolution()
    print(f"BLUEPRINT_ADP009D_CAMERA_RESOLUTION:{render_width}x{render_height}", flush=True)
    for camera_name in ("external_camera", "wrist_camera", "external_camera_2"):
        camera_cfg = getattr(embodiment.camera_config, camera_name)
        if camera_cfg is None:
            raise RuntimeError(f"required_evaluation_camera_config_missing:{camera_name}")
        camera_cfg.data_types = ["rgb", "semantic_segmentation"]
        if camera_name != "external_camera_2":
            camera_cfg.data_types.insert(1, "distance_to_camera")
        camera_cfg.colorize_semantic_segmentation = False
        camera_cfg.update_period = 0.0
        # Arena leaves this false, which makes camera.data.pos_w/quat_w_* stay
        # frozen at initialization even while the parented render view moves.
        # Exact pose metadata is part of the policy-input evidence contract.
        camera_cfg.update_latest_camera_pose = True
        camera_cfg.width = render_width
        camera_cfg.height = render_height
        # Gaussian surfels are not drawn unless the render product asks for them.
        # These were first added because a run put the ParticleField in the
        # scene and produced frames identical to a run without it.  That run's
        # asset had no default prim, so the reference resolved to nothing and
        # the settings were never actually exercised -- the conclusion that
        # they changed nothing was drawn against an empty scene.
        #
        # With the field genuinely composing, the accumulation cap is the
        # binding constraint.  Forty-eight gaussians per ray cannot build a
        # surface out of a field whose median surfel is 0.81mm across in a
        # 9.9m room: the first frames to render it came back as isolated
        # speckles at 16% pixel coverage, which is what an under-accumulated
        # ray looks like.  The cap is raised and overridable so it can be
        # swept without a rebuild.
        for setting, value in (
            ("rtx/rtpt/gaussian/accumulatedDepth/enabled", True),
            ("rtx/rtpt/gaussian/accumulatedAlbedo/enabled", True),
            ("rtx/rtpt/gaussian/maxGaussiansToAccumulate", _max_gaussians_to_accumulate()),
        ):
            try:
                import carb

                carb.settings.get_settings().set(f"/{setting}", value)
            except Exception:  # noqa: BLE001 - recorded by the render check below
                pass
    external_camera_cfg = embodiment.camera_config.external_camera
    external_task_camera_plan = external_task_camera_offset_plan(
        robot_position_world=robot_pose.position_xyz,
        robot_quaternion_world_xyzw=robot_pose.rotation_xyzw,
        current_camera_offset_position_robot=external_camera_cfg.offset.pos,
        target_position_world=(
            CAN_AXIS_XY_M[0],
            CAN_AXIS_XY_M[1],
            SUPPORT_HEIGHT_M + APPROVED_CAN_TOP_ABOVE_SUPPORT_M / 2.0,
        ),
    )
    external_camera_cfg.offset.pos = tuple(
        external_task_camera_plan["resolved_offset_position_robot_m"]
    )
    external_task_camera_plan.update(
        {
            "schema_version": "adp009d_external_task_camera_plan.v2",
            "authoritative_seam": "Arena CameraCfg.offset before prim spawn",
            "orientation_source": "official Arena DROID external camera offset",
            "orientation_unchanged": True,
            "resolution_unchanged": True,
            "intrinsics_unchanged": True,
        }
    )
    # Arena's stock second exterior view faces away from this task: v88
    # measured zero robot/can semantic pixels in every retained frame.  Reuse
    # the proven task-camera orientation, move farther back on the ray through
    # the midpoint of the start/destination envelope, and apply it at the
    # render-authoritative CameraCfg seam before spawn.  This remains
    # review-only and can never enter policy input or scoring.
    destination = json.loads(
        (runtime / "adp009d_task_destination.v1.json").read_text(encoding="utf-8")
    )["position_world_m"]
    task_envelope_center = [
        (CAN_AXIS_XY_M[0] + float(destination[0])) / 2.0,
        (CAN_AXIS_XY_M[1] + float(destination[1])) / 2.0,
        SUPPORT_HEIGHT_M + APPROVED_CAN_TOP_ABOVE_SUPPORT_M / 2.0,
    ]
    overview_camera_cfg = embodiment.camera_config.external_camera_2
    overview_offset_plan = external_task_camera_offset_plan(
        robot_position_world=robot_pose.position_xyz,
        robot_quaternion_world_xyzw=robot_pose.rotation_xyzw,
        current_camera_offset_position_robot=external_camera_cfg.offset.pos,
        target_position_world=task_envelope_center,
        distance_m=OVERVIEW_TASK_CAMERA_DISTANCE_M,
    )
    overview_camera_cfg.offset.pos = tuple(
        overview_offset_plan["resolved_offset_position_robot_m"]
    )
    overview_camera_cfg.offset.rot = external_camera_cfg.offset.rot
    overview_camera_cfg.offset.convention = external_camera_cfg.offset.convention
    overview_camera_plan = {
        "schema_version": "blueprint_episode_overview_camera_plan.v1",
        "runtime_camera_name": "external_camera_2",
        "evidence_camera_id": "overview",
        "pose_source": "task_centered_wide_view_from_proven_external_orientation",
        "task_envelope_center_world_m": task_envelope_center,
        "target_distance_m": OVERVIEW_TASK_CAMERA_DISTANCE_M,
        "resolved_eye_position_world_m": overview_offset_plan[
            "resolved_eye_position_world_m"
        ],
        "render_authoritative_seam": "Arena CameraCfg.offset before prim spawn",
        "minimum_start_object_semantic_pixels": MIN_OVERVIEW_TASK_OBJECT_PIXELS,
        "role": "review_only_full_task_motion",
        "policy_input": False,
        "grader_input": False,
        "lossless_frames_required": True,
        "calibration_and_timestamps_required": True,
        "portable_review_video_required": True,
    }
    _phase("embodiment_configuration", "completed")

    _phase("sealed_scene_configuration")
    sage = Object(
        name="sage_collision",
        object_type=ObjectType.BASE,
        usd_path=str(runtime / "assets" / TASK_COLLISION_DERIVATIVE_FILENAME),
        initial_pose=Pose.identity(),
        spawn_cfg_addon={"visible": False},
    )
    # The sealed appearance, rendered in the same pass as the robot.  Visual
    # only: SAGE remains the sole collision authority, so adding the appearance
    # cannot change a single contact.  Without this the policy cameras see the
    # arm and the can against nothing, which the goal prompt rules invalid --
    # a learned result needs coherent views of the Aura background, the moving
    # Franka, the moving can and their occlusions, in one time-synchronised
    # frame rather than two renders glued together afterward.
    aura_particlefield_path, aura_appearance_format = _resolve_aura_appearance(runtime)
    aura_appearance = None
    if aura_particlefield_path is not None:
        aura_appearance = Object(
            name="aura_appearance",
            object_type=ObjectType.BASE,
            usd_path=str(aura_particlefield_path),
            initial_pose=Pose.identity(),
            spawn_cfg_addon={"visible": True},
        )
    approved_can_spawn_addon: dict[str, Any] = {
        "semantic_tags": _semantic_tags("approved_can")
    }
    approved_can_path = runtime / "assets" / "approved_can.usda"
    if args.physics_backend == "physx":
        approved_can_path = runtime / "assets" / APPROVED_CAN_ADAPTER_FILENAME
        approved_can_spawn_addon["rigid_props"] = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,
        )
    else:
        approved_can_path = (
            runtime / "assets" / APPROVED_CAN_NEWTON_ADAPTER_FILENAME
        )
    approved_can = Object(
        name="approved_can",
        object_type=ObjectType.RIGID,
        usd_path=str(approved_can_path),
        initial_pose=Pose(position_xyz=CAN_START_POSITION_M),
        spawn_cfg_addon=approved_can_spawn_addon,
    )
    # Isaac Lab prim-path tokens match one USD level each, and the Robotiq
    # fingers sit at Robot/Gripper/Robotiq_2F_85/<finger> in the pinned DROID
    # embodiment (the same paths its own FrameTransformer binds), so a
    # single-level Robot/.* wildcard can never resolve them.  Arm-link
    # constraint evidence still comes from the per-link incoming joint wrench.
    # The a0cf16c9 canary measured an 8.6 N vertical contact on one finger while
    # the nearest scene triangle sat 70 mm from that finger's body origin and the
    # can lid 81 mm below it, so the net force alone cannot name the partner.
    # Filtering on the can's rigid body already proved this contact is not the
    # can.  A second, independent SAGE-root filter is therefore diagnostic only:
    # a nonzero SAGE force attributes this configured collision scope, while an
    # unresolved/zero result leaves the non-can source explicitly unresolved.
    # Neither filter changes contact behavior.
    robot_contact = ContactSensorAsset(
        name="robot_contact",
        sensor_cfg=BackendContactSensorCfg(
            prim_path=_robot_contact_sensor_prim_path(args.physics_backend),
            update_period=0.0,
            history_length=1,
            debug_vis=False,
        ),
    )
    partner_contacts = [
        ContactSensorAsset(
            name=sensor_name,
            sensor_cfg=BackendContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/Gripper/Robotiq_2F_85/{body_name}",
                update_period=0.0,
                history_length=1,
                debug_vis=False,
                **_contact_partner_filter_kwargs(args.physics_backend),
            ),
        )
        for body_name, sensor_name in sorted(CONTACT_PARTNER_SENSOR_NAMES.items())
    ]
    sage_collision_contacts = [
        ContactSensorAsset(
            name=sensor_name,
            sensor_cfg=BackendContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/Gripper/Robotiq_2F_85/{body_name}",
                update_period=0.0,
                history_length=1,
                debug_vis=False,
                **_sage_collision_filter_kwargs(args.physics_backend),
            ),
        )
        for body_name, sensor_name in sorted(
            CONTACT_SAGE_COLLISION_SENSOR_NAMES.items()
        )
    ]
    light = SpawnerObject(
        name="light",
        prim_path="/World/Light",
        spawner_cfg=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=1500.0,
        ),
    )
    scene = Scene(
        assets=[
            sage,
            approved_can,
            robot_contact,
            *partner_contacts,
            *sage_collision_contacts,
            light,
        ]
        + ([aura_appearance] if aura_appearance is not None else [])
    )
    _phase("sealed_scene_configuration", "completed")

    def configure(cfg):
        cfg.sim.dt = 1.0 / 120.0
        cfg.seed = 20260806
        cfg.sim.render_interval = 8
        cfg.decimation = 8
        cfg.episode_length_s = 5.0
        if args.physics_backend == "physx":
            from isaaclab_physx.physics import PhysxCfg

            cfg.sim.physics = PhysxCfg(
                solver_type=1,
                enable_enhanced_determinism=True,
                gpu_max_rigid_contact_count=2**23,
                gpu_max_rigid_patch_count=2**15,
            )
        else:
            from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

            cfg.sim.physics = NewtonCfg(
                solver_cfg=MJWarpSolverCfg(
                    njmax=2048,
                    nconmax=1024,
                    iterations=100,
                    ls_iterations=20,
                    solver="newton",
                    integrator="implicitfast",
                    cone="pyramidal",
                    use_mujoco_contacts=True,
                    use_mujoco_cpu=False,
                    save_to_mjcf=str(
                        Path(args.output_dir).resolve()
                        / "newton_converted_model.xml"
                    ),
                ),
                num_substeps=1,
                debug_mode=False,
                use_cuda_graph=True,
            )
        return cfg

    _phase("arena_environment_definition")
    arena_env = IsaacLabArenaEnvironment(
        name="Blueprint-ADP009D-Franka-Microcheck-v0",
        scene=scene,
        embodiment=embodiment,
        task=NoTask(),
        env_cfg_callback=configure,
    )
    _phase("arena_environment_definition", "completed")
    builder_args = argparse.Namespace(
        num_envs=1,
        env_spacing=2.0,
        solve_relations=False,
        placement_seed=20260806,
        mimic=False,
        device=args.device,
        disable_fabric=False,
        presets=None,
    )
    _phase("arena_builder_registration")
    builder = ArenaEnvBuilder(arena_env, builder_args)
    _phase("arena_builder_registration", "completed")
    _phase("manager_based_environment_construction")
    env, cfg = builder.make_registered_and_return_cfg(render_mode="rgb_array")
    _phase("manager_based_environment_construction", "completed")
    return (
        env,
        cfg,
        torch,
        external_task_camera_plan,
        overview_camera_plan,
        newton_actuator_limit_mapping,
    )


def _preflight_environment_imports(physics_backend: str = "physx") -> dict[str, str]:
    """Import the exact environment-builder closure after Kit is available."""

    import importlib.metadata as metadata

    import antlr4  # noqa: F401
    import h5py  # noqa: F401
    import hydra  # noqa: F401
    import msgpack  # noqa: F401
    import omegaconf  # noqa: F401
    import zmq  # noqa: F401
    from isaaclab_ov.renderers import OVRTXRendererCfg  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: F401
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: F401
    from isaaclab_arena.assets.object import Object  # noqa: F401
    from isaaclab_arena.embodiments.droid.droid import (  # noqa: F401
        DroidAbsoluteJointPositionEmbodiment,
    )
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder  # noqa: F401
    from isaaclab_arena.environments.isaaclab_arena_environment import (  # noqa: F401
        IsaacLabArenaEnvironment,
    )
    from isaaclab_arena.scene.scene import Scene  # noqa: F401
    from isaaclab_arena.tasks.no_task import NoTask  # noqa: F401

    names = [
        "antlr4-python3-runtime",
        "h5py",
        "hydra-core",
        "isaaclab",
        "isaaclab_arena",
        "isaaclab_ov",
        "isaaclab_rl",
        "msgpack",
        "omegaconf",
        "pyzmq",
        "rsl-rl-lib",
        "isaaclab_physx" if physics_backend == "physx" else "isaaclab_newton",
    ]
    if physics_backend == "newton":
        names.extend(("newton", "mujoco", "mujoco-warp", "warp-lang"))
    return {
        name: metadata.version(name)
        for name in names
    }


def _run(runtime: Path, output: Path, args: argparse.Namespace) -> dict[str, Any]:
    backend, backend_profile, backend_contact_configuration = (
        _load_runtime_backend_contract(runtime, args.physics_backend)
    )
    for name, digest in EXPECTED_ASSETS.items():
        path = runtime / "assets" / name
        if not path.is_file() or _sha256(path) != digest:
            raise RuntimeError(f"sealed_asset_binding_invalid:{name}")

    from pxr import Usd
    if backend == "physx":
        adapter_path = runtime / "assets" / APPROVED_CAN_ADAPTER_FILENAME
        if (
            not adapter_path.is_file()
            or _sha256(adapter_path) != APPROVED_CAN_ADAPTER_SHA256
        ):
            raise RuntimeError(
                "sealed_asset_binding_invalid:approved_can_physx_sdf_adapter.usda"
            )
        adapter_stage = Usd.Stage.Open(str(adapter_path))
        if adapter_stage is None:
            raise RuntimeError("approved_can_physx_sdf_adapter_unreadable")
        static_collider = _inspect_physx_sdf_collider(
            adapter_stage, APPROVED_CAN_SOURCE_COLLIDER_PRIM
        )
    else:
        source_stage = Usd.Stage.Open(
            str(runtime / "assets" / APPROVED_CAN_NEWTON_ADAPTER_FILENAME)
        )
        if source_stage is None:
            raise RuntimeError("approved_can_newton_source_unreadable")
        source_prim = source_stage.GetPrimAtPath(APPROVED_CAN_SOURCE_COLLIDER_PRIM)
        applied_schemas = [str(value) for value in source_prim.GetAppliedSchemas()]
        source_approximation = source_prim.GetAttribute(
            "physics:approximation"
        ).Get()
        if (
            not source_prim.IsValid()
            or any("Physx" in value for value in applied_schemas)
            or source_approximation is not None
        ):
            raise RuntimeError("approved_can_newton_source_contains_physx_schema")
        static_collider = {
            "backend": "newton",
            "source_prim": APPROVED_CAN_SOURCE_COLLIDER_PRIM,
            "source_asset_digest": EXPECTED_ASSETS["approved_can.usda"],
            "unsupported_source_approximation_blocked": True,
            "applied_schemas": applied_schemas,
            "physx_sdf_overlay_loaded": False,
            "source_approximation_semantics_assumed": False,
        }
    task_collision_manifest_path = runtime / "assets" / TASK_COLLISION_MANIFEST_FILENAME
    if not task_collision_manifest_path.is_file():
        raise RuntimeError("sage_task_collision_manifest_missing")
    task_collision_manifest = json.loads(task_collision_manifest_path.read_text(encoding="utf-8"))
    task_collision_path = runtime / "assets" / TASK_COLLISION_DERIVATIVE_FILENAME
    if (
        task_collision_manifest.get("status") != "ready"
        or task_collision_manifest.get("sealed_source_sha256")
        != EXPECTED_ASSETS["sage_collision.usd"]
        or task_collision_manifest.get("sealed_source_mutated") is not False
        or task_collision_manifest.get("derivative_filename") != TASK_COLLISION_DERIVATIVE_FILENAME
        or not task_collision_path.is_file()
        or _sha256(task_collision_path) != task_collision_manifest.get("derivative_sha256")
        or task_collision_manifest.get("claim_ceiling") != "preregistered_franka_task_envelope_only"
    ):
        raise RuntimeError("sage_task_collision_derivative_binding_invalid")
    expected_sage_profile = {
        "active_mesh_count": int(task_collision_manifest["active_source_prim_count"]),
        "active_point_count": int(task_collision_manifest["derived_point_count"]),
        "active_face_count": int(task_collision_manifest["derived_face_count"]),
        "rigid_body_count": 0,
        "triangle_mesh_count": int(task_collision_manifest["active_source_prim_count"]),
    }
    task_collision_shape_labels = tuple(
        str(row.get("source_prim", "")).rsplit("/", 1)[-1]
        for row in task_collision_manifest.get("source_prim_rows", [])
    )
    if (
        expected_sage_profile != SAGE_RUNTIME_PROFILE
        or task_collision_shape_labels != NEWTON_SAGE_COLLISION_SHAPE_LABELS
        or task_collision_manifest.get("candidate_source_prim_count") != 16
        or task_collision_manifest.get("source_face_count") != 47_359
        or task_collision_manifest.get("roi_min_m") != [2.4681748, -4.3100837, -0.1]
        or task_collision_manifest.get("roi_max_m") != [4.4681748, -1.9100837, 1.8]
        or task_collision_manifest.get("maximum_edge_limit_m") != 0.5
        or float(task_collision_manifest.get("observed_maximum_edge_m", math.inf)) > 0.500001
        or float(task_collision_manifest.get("relative_surface_area_error", math.inf)) > 1.0e-6
    ):
        raise RuntimeError("sage_task_collision_profile_invalid")
    sage_overlay_stage = Usd.Stage.Open(str(task_collision_path))
    if sage_overlay_stage is None:
        raise RuntimeError("sage_task_collision_derivative_unreadable")
    static_sage_collision = _inspect_sage_static_triangle_colliders(
        sage_overlay_stage,
        SAGE_SOURCE_ROOT_PRIM,
        expected_profile=expected_sage_profile,
    )
    _phase("static_collider_validation", "completed")

    collision_cooking = None
    if backend == "physx":
        _phase("physx_collision_cooking_configuration")
        collision_cooking = _configure_physx_collision_cooking()
        _phase("physx_collision_cooking_configuration", "completed")

    import omni.log

    fallback_messages: list[str] = []
    stability_messages: list[str] = []
    newton_unsupported_messages: list[str] = []

    def on_log(channel, level, module, filename, func, line_no, message, pid, tid, timestamp):
        del channel, level, module, filename, func, line_no, pid, tid, timestamp
        if PHYSX_FALLBACK_MARKER in message:
            fallback_messages.append(str(message))
        if PHYSX_TRIANGLE_STABILITY_MARKER in message:
            stability_messages.append(str(message))
        lowered = str(message).lower()
        if backend == "newton" and (
            "physxsdfmeshcollision" in lowered
            or (
                ("unsupported" in lowered or "ignored" in lowered)
                and any(
                    field.lower() in lowered
                    for field in (
                        "sdf_margin",
                        "sdf_narrow_band_thickness",
                        "gpu_max_rigid_contact_count",
                        "gpu_max_rigid_patch_count",
                        "solver_position_iteration_count",
                        "solver_velocity_iteration_count",
                        "enable_enhanced_determinism",
                    )
                )
            )
        ):
            newton_unsupported_messages.append(str(message))

    log = omni.log.get_log()
    consumer = log.add_message_consumer(on_log)
    env = None
    timings_seconds: dict[str, float] = {}
    external_task_camera_plan: dict[str, Any] | None = None
    backend_probe: dict[str, Any] | None = None
    newton_robot_inertial_overlay_receipt: dict[str, Any] | None = None
    try:
        def fail_on_backend_collision_logs() -> None:
            if backend == "physx":
                _fail_on_physx_collision_fallback(fallback_messages)
                _fail_on_physx_collision_stability(stability_messages)
            elif newton_unsupported_messages:
                raise RuntimeError("adp009d_newton_unsupported_physx_setting_observed")

        _phase("runtime_import_preflight")
        runtime_import_preflight = _preflight_environment_imports(backend)
        backend_runtime = dict(backend_profile["backend_runtime"])
        expected_backend_versions = {
            str(backend_runtime["package"]): str(backend_runtime["version"])
        }
        if backend == "newton":
            expected_backend_versions.update(
                {
                    "newton": str(backend_runtime["newton"]["package_version"]),
                    "mujoco": str(backend_runtime["mujoco_version"]),
                    "mujoco-warp": str(backend_runtime["mujoco_warp_version"]),
                    "warp-lang": str(backend_runtime["warp_version"]),
                }
            )
        if any(
            runtime_import_preflight.get(name) != version
            for name, version in expected_backend_versions.items()
        ):
            raise RuntimeError("adp009d_backend_runtime_version_mismatch")
        _phase("runtime_import_preflight", "completed")
        _phase("environment_build")
        phase_started = time.monotonic()
        try:
            (
                env,
                cfg,
                torch,
                external_task_camera_plan,
                overview_camera_plan,
                newton_actuator_limit_mapping,
            ) = _build_environment(runtime, args)
        except Exception as exc:
            if backend == "newton":
                existing = getattr(exc, "diagnostics", None)
                diagnostics = dict(existing) if isinstance(existing, dict) else {}
                overlay_receipt_path = (
                    output / NEWTON_ROBOT_INERTIAL_OVERLAY_RECEIPT_FILENAME
                )
                if overlay_receipt_path.is_file():
                    try:
                        diagnostics["newton_robot_inertial_overlay"] = json.loads(
                            overlay_receipt_path.read_text(encoding="utf-8")
                        )
                    except Exception as receipt_exc:  # noqa: BLE001
                        diagnostics["newton_robot_inertial_overlay_read_error"] = (
                            f"{type(receipt_exc).__name__}: {receipt_exc}"
                        )
                diagnostics["newton_contact_labels"] = (
                    _newton_contact_label_diagnostics()
                )
                try:
                    exc.diagnostics = diagnostics
                except Exception:  # noqa: BLE001 - retain labels via a typed wrapper
                    wrapped = RuntimeError(str(exc))
                    wrapped.diagnostics = diagnostics
                    raise wrapped from exc
            raise
        timings_seconds["environment_build"] = round(time.monotonic() - phase_started, 6)
        if backend == "newton":
            overlay_receipt_path = (
                output / NEWTON_ROBOT_INERTIAL_OVERLAY_RECEIPT_FILENAME
            )
            if not overlay_receipt_path.is_file():
                raise RuntimeError(
                    "adp009d_newton_robot_inertial_overlay_receipt_missing"
                )
            newton_robot_inertial_overlay_receipt = json.loads(
                overlay_receipt_path.read_text(encoding="utf-8")
            )
            overlay_blockers = _validate_newton_robot_inertial_overlay_receipt(
                newton_robot_inertial_overlay_receipt,
                backend_profile=backend_profile,
            )
            if overlay_blockers:
                raise RuntimeError(
                    "adp009d_newton_robot_inertial_overlay_receipt_invalid:"
                    + ",".join(overlay_blockers)
                )
        log.flush()
        _phase("environment_build", "completed")
        fail_on_backend_collision_logs()
        import omni.usd

        live_stage = omni.usd.get_context().get_stage()
        # Two explanations for the invisible appearance have now failed: the
        # OVRTX lane's crash-then-residency theory, and authoring gaussian
        # accumulation settings here.  Rather than guess a third, ask the live
        # stage what actually became of the prim -- whether it exists at all,
        # under what path, and whether the surflet API survived Arena's spawner.
        aura_stage_probe: dict[str, Any] = {
            "schema_version": "adp009d_aura_stage_probe.v1",
            "expected_prim": AURA_PARTICLEFIELD_PRIM,
        }
        try:
            # Case-insensitive, and NuRec-aware.  This matched "Gauss" and
            # "Aura" exactly, so a NuRec volume -- which composes at
            # /World/gauss/gauss, lowercase -- reported zero matching prims
            # for a scene that had visibly rendered the whole room.  A probe
            # that says "absent" about something present is worse than none.
            matches = [
                str(prim.GetPath())
                for prim in live_stage.Traverse()
                if any(token in str(prim.GetPath()).lower() for token in ("gauss", "aura", "nurec"))
                or bool(prim.GetAttribute("omni:nurec:isNuRecVolume").Get())
            ]
            aura_stage_probe["matching_prim_paths"] = matches[:20]
            aura_stage_probe["matching_prim_count"] = len(matches)
            if matches:
                # The field itself, not the Xform that holds it: reading
                # matches[0] returned the parent and reported an empty
                # applied-schema list for a prim that carried all nine.
                field_paths = [m for m in matches if m.endswith("GaussianSurflets")]
                found = live_stage.GetPrimAtPath(field_paths[0] if field_paths else matches[0])
                aura_stage_probe["inspected_prim_path"] = str(found.GetPath())
                aura_stage_probe["applied_schemas"] = [str(v) for v in found.GetAppliedSchemas()]
                aura_stage_probe["type_name"] = str(found.GetTypeName())
                aura_stage_probe["is_active"] = bool(found.IsActive())
                visibility = found.GetAttribute("visibility")
                aura_stage_probe["visibility"] = (
                    str(visibility.Get()) if visibility and visibility.IsValid() else None
                )
        except Exception as exc:  # noqa: BLE001 - a diagnostic must not fail a run
            aura_stage_probe["error"] = f"{type(exc).__name__}: {exc}"

        if backend == "physx":
            live_collider = _inspect_physx_sdf_collider(
                live_stage, APPROVED_CAN_LIVE_COLLIDER_PRIM
            )
        else:
            live_prim = live_stage.GetPrimAtPath(APPROVED_CAN_LIVE_COLLIDER_PRIM)
            live_applied_schemas = (
                [str(value) for value in live_prim.GetAppliedSchemas()]
                if live_prim.IsValid()
                else []
            )
            converted_model_path = output / "newton_converted_model.xml"
            if (
                not live_prim.IsValid()
                or any("Physx" in value for value in live_applied_schemas)
                or not converted_model_path.is_file()
            ):
                raise RuntimeError("adp009d_newton_asset_conversion_probe_failed")
            live_collider = {
                "backend": "newton",
                "live_prim": APPROVED_CAN_LIVE_COLLIDER_PRIM,
                "applied_schemas": live_applied_schemas,
                "physx_sdf_overlay_loaded": False,
                "physx_only_fields_observed": [],
                "silently_ignored_settings": [],
                "source_asset_digest": EXPECTED_ASSETS["approved_can.usda"],
                "converted_model_path": converted_model_path.name,
                "converted_model_digest": _sha256(converted_model_path),
                "backend_contact_configuration": backend_contact_configuration,
                "robot_inertial_overlay": newton_robot_inertial_overlay_receipt,
            }
        live_sage_collision = _inspect_sage_static_triangle_colliders(
            live_stage,
            SAGE_LIVE_ROOT_PRIM,
            expected_profile=expected_sage_profile,
        )
        _phase("live_collider_validation", "completed")
        reset_rows: list[dict[str, Any]] = []
        for index in range(2):
            _phase(f"reset_{index}")
            phase_started = time.monotonic()
            observation, info = env.reset(seed=20260806)
            timings_seconds[f"reset_{index}"] = round(time.monotonic() - phase_started, 6)
            robot = env.unwrapped.scene["robot"]
            approved_can = env.unwrapped.scene["approved_can"]
            reset_arm_maximum_error_rad = _assert_arm_pose(
                _to_torch(robot.data.joint_pos)[0],
                RESET_JOINTS,
                tolerance_rad=RESET_ARM_TOLERANCE_RAD,
                blocker="canonical_reset_arm_pose_mismatch",
            )
            reset_rows.append(
                {
                    "index": index,
                    "joint_pos": _jsonable(_to_torch(robot.data.joint_pos)[0]),
                    "can_root_pose_world": _jsonable(_to_torch(approved_can.data.root_pose_w)[0]),
                    "observation_keys": sorted(str(key) for key in observation),
                    "info_keys": sorted(str(key) for key in (info or {})),
                    "arm_maximum_error_rad": reset_arm_maximum_error_rad,
                }
            )
            log.flush()
            fail_on_backend_collision_logs()
            _phase(f"reset_{index}", "completed")
        joint_a = torch.tensor(reset_rows[0]["joint_pos"])
        joint_b = torch.tensor(reset_rows[1]["joint_pos"])
        if not torch.equal(joint_a, joint_b):
            raise RuntimeError("canonical_reset_not_bitwise_reproducible")

        action = torch.zeros(
            (1, env.unwrapped.action_manager.total_action_dim),
            device=env.unwrapped.device,
        )
        # The first step is the first time anything is actually rendered, so it
        # pays the whole cost of composing the scene's appearance.  It had no
        # marker of its own, which left a live run unable to tell "stuck in the
        # first render" from "stuck comparing two joint tensors" -- the last
        # thing it had said was reset_1:completed, several phases earlier.
        _phase("zero_action_step")
        phase_started = time.monotonic()
        observation, reward, terminated, truncated, info = _run_under_render_budget(
            lambda: env.step(action),
            phase_name="zero_action_step",
            diagnostics={
                "aura_stage_probe": aura_stage_probe,
                "aura_appearance_shipped": _resolve_aura_appearance(runtime)[0] is not None,
                "aura_appearance_format": _resolve_aura_appearance(runtime)[1],
                "note": (
                    "first step is the first render; a budget overrun here means "
                    "the scene composed but cannot be drawn in bounded time"
                ),
            },
        )
        timings_seconds["zero_action_step"] = round(time.monotonic() - phase_started, 6)
        _phase("zero_action_step", "completed")
        log.flush()
        fail_on_backend_collision_logs()
        zero_action_row = {
            "action_dim": env.unwrapped.action_manager.total_action_dim,
            "reward": _jsonable(reward),
            "terminated": _jsonable(terminated),
            "truncated": _jsonable(truncated),
            "observation_keys": sorted(str(key) for key in observation),
            "robot_joint_pos_after_step": _jsonable(
                _to_torch(env.unwrapped.scene["robot"].data.joint_pos)[0]
            ),
            "approved_can_pose_after_step": _jsonable(
                _to_torch(env.unwrapped.scene["approved_can"].data.root_pose_w)[0]
            ),
        }
        env.reset(seed=20260806)
        robot = env.unwrapped.scene["robot"]
        body_names_all = list(robot.data.body_names)
        approved_can = env.unwrapped.scene["approved_can"]
        hold_start_can_pose = _to_torch(approved_can.data.root_pose_w)[0].clone()
        hold_action = torch.zeros_like(action)
        hold_action[:, :7] = _to_torch(robot.data.joint_pos)[:, :7]
        warmup_frames = _camera_warmup_frames()
        _phase("camera_warmup")
        print(f"BLUEPRINT_ADP009D_CAMERA_WARMUP_FRAMES:{warmup_frames}", flush=True)
        phase_started = time.monotonic()
        marker_every = max(1, warmup_frames // 4)
        hold_effort_limits = extract_arm_effort_limits(robot, to_list=_jsonable)
        hold_samples: list[dict[str, Any]] = []
        for warmup_index in range(warmup_frames):
            observation, reward, terminated, truncated, info = env.step(hold_action)
            hold_sample = extract_arm_sample(
                robot, step_index=warmup_index, to_list=_jsonable
            )
            if hold_sample is not None:
                hold_samples.append(hold_sample)
            if (warmup_index + 1) % marker_every == 0:
                log.flush()
                fail_on_backend_collision_logs()
                _phase(f"camera_warmup_{warmup_index + 1}", "completed")
        timings_seconds[f"camera_warmup_{warmup_frames}_frames"] = round(
            time.monotonic() - phase_started, 6
        )
        hold_trace = None
        if hold_samples:
            try:
                hold_trace = classify_arm_hold_trace(
                hold_samples,
                requested_joint_positions_rad=RESET_JOINTS,
                tolerance_rad=HOLD_ARM_TOLERANCE_RAD,
                effort_limits_nm=hold_effort_limits,
            )
            except HoldTraceError as exc:
                # Diagnostics must not replace the canonical pose blocker.  A
                # malformed backend readback is retained as a typed trace gap.
                hold_trace = {
                    "schema_version": HOLD_TRACE_SCHEMA_VERSION,
                    "status": "unavailable",
                    "typed_blocker": str(exc),
                }
        hold_arm_maximum_error_rad = _assert_arm_pose(
            _to_torch(env.unwrapped.scene["robot"].data.joint_pos)[0],
            RESET_JOINTS,
            tolerance_rad=HOLD_ARM_TOLERANCE_RAD,
            blocker="canonical_hold_arm_pose_drift",
            hold_trace=hold_trace,
        )
        camera_retention_started = time.monotonic()
        camera_rows = []
        for camera_name in ("external_camera", "wrist_camera", "external_camera_2"):
            camera_rows.append(
                _save_camera(
                    output,
                    camera_name,
                    env.unwrapped.scene[camera_name],
                    frame_index=warmup_frames,
                    sim_time=float(
                        env.unwrapped.episode_length_buf[0].item() * cfg.sim.dt * cfg.decimation
                    ),
                    require_metric_depth=(camera_name != "external_camera_2"),
                )
            )
        overview_observability = _approved_can_observability(
            env.unwrapped.scene["external_camera_2"]
        )
        overview_camera_plan["start_object_observability"] = overview_observability
        overview_camera_plan["start_object_observability_status"] = (
            "ready"
            if overview_observability["approved_task_object_pixel_count"]
            >= MIN_OVERVIEW_TASK_OBJECT_PIXELS
            and overview_observability[
                "approved_task_object_within_frame_margin"
            ]
            else "blocked"
        )
        if overview_camera_plan["start_object_observability_status"] != "ready":
            raise RuntimeError(
                "overview_camera_task_object_not_observable:"
                f"pixels={overview_observability['approved_task_object_pixel_count']}:"
                "within_margin="
                f"{overview_observability['approved_task_object_within_frame_margin']}"
            )
        timings_seconds["camera_retention"] = round(time.monotonic() - camera_retention_started, 6)
        # A frame is the only thing that answers whether the appearance actually
        # draws, and at roughly a minute per rendered frame the phases after
        # this one -- a four-hundred-step approach, a four-hundred-and-eighty
        # step episode -- run for hours and end with the TTL killing the
        # instance before anything is uploaded.  Frames are zipped only after
        # the runtime exits, so a run that never exits delivers nothing.
        if os.environ.get(STOP_AFTER_FRAMES_ENV, "").strip() not in {"", "0", "false"}:
            _phase("stopped_after_frames", "completed")
            print(
                "BLUEPRINT_ADP009D_STOPPED_AFTER_FRAMES:"
                + json.dumps(
                    {
                        "reason": "diagnostic_frames_only_mode",
                        "camera_rows": camera_rows,
                        "warmup_frames": warmup_frames,
                        "aura_stage_probe": aura_stage_probe,
                        "timings_seconds": timings_seconds,
                    },
                    sort_keys=True,
                    default=str,
                ),
                flush=True,
            )
            (output / "adp009d_frames_only_probe.json").write_text(
                json.dumps(
                    {
                        "schema_version": "adp009d_frames_only_probe.v1",
                        "status": "completed",
                        "mode": "frames_only",
                        # Never a success claim for the micro-check itself: this
                        # run deliberately skipped every phase after the frames.
                        "supports_microcheck_success_claim": False,
                        "camera_rows": camera_rows,
                        "warmup_frames": warmup_frames,
                        "aura_stage_probe": aura_stage_probe,
                        "timings_seconds": timings_seconds,
                    },
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
                + "\n",
                encoding="utf-8",
            )
            log.flush()
            # _run is declared to return a dict and main reads result["status"];
            # a bare return handed back None and the caller died on it after the
            # frames were already saved, turning a successful diagnostic into an
            # opaque 'NoneType' object has no attribute 'get'.
            #
            # Deliberately blocked, not completed: this run skipped every phase
            # after the frames, so it must not exit zero or read as a passing
            # micro-check anywhere downstream.
            return {
                "schema_version": "adp009d_native_microcheck.v1",
                "status": "blocked",
                "blockers": ["stopped_after_frames_diagnostic_mode"],
                "mode": "frames_only",
                "supports_microcheck_success_claim": False,
                "camera_rows": _json_safe(camera_rows),
                "warmup_frames": warmup_frames,
                "aura_stage_probe": _json_safe(aura_stage_probe),
                # Omitting this read as None for a run whose appearance had
                # visibly rendered, which is exactly the attribution the
                # NuRec-versus-ParticleField comparison depends on.
                "aura_appearance_shipped": _resolve_aura_appearance(runtime)[0] is not None,
                "aura_appearance_format": _resolve_aura_appearance(runtime)[1],
                "max_gaussians_to_accumulate": _max_gaussians_to_accumulate(),
                "timings_seconds": _json_safe(timings_seconds),
            }

        # --- finger collision envelope probe ----------------------------------
        # The descend stall left the semantic fingertip frame 34.6 mm above the
        # can lid with the nearest scene triangle 70 mm away, so the contact is
        # with geometry the planner has no model of: Arena's tool frame is a
        # +46 mm semantic point, not the finger's collision extent.  Measure that
        # extent once, before any motion, so the gap between the planned tip and
        # the real swept volume is a recorded number instead of an inference.
        # Diagnostics must never break a paid run, so this cannot raise.
        finger_collision_envelope = _probe_finger_collision_envelope()

        # --- gripper convention probe -----------------------------------------
        # DROID encodes the gripper as a scalar in [0, 1] where above 0.5 means
        # closed.  Arena's eighth action dimension has its own convention, and
        # an inverted one would turn every commanded grasp into a release --
        # which would read as a policy failure rather than a harness bug.  So
        # measure it: command each candidate, let the fingers settle, and report
        # which one closes them.  The action executor refuses to run until this
        # has been observed, and an ambiguous result must stay ambiguous.
        _phase("gripper_convention_probe")
        phase_started = time.monotonic()
        finger_pair = ("left_inner_finger", "right_inner_finger")
        finger_indices = [
            body_names_all.index(name) for name in finger_pair if name in body_names_all
        ]
        gripper_probe: dict[str, Any] = {
            "schema_version": "adp009d_gripper_convention_probe.v1",
            "candidate_commands": [0.0, 1.0],
            "finger_bodies": list(finger_pair),
            "settle_steps": 30,
        }
        if len(finger_indices) == 2:
            separations: dict[str, float] = {}
            for command in (0.0, 1.0):
                env.reset(seed=20260806)
                probe_action = torch.zeros_like(action)
                probe_action[:, :7] = _to_torch(robot.data.joint_pos)[:, :7]
                probe_action[:, 7] = float(command)
                for _ in range(30):
                    env.step(probe_action)
                poses = _to_torch(robot.data.body_pose_w)[0, finger_indices, :3]
                separations[str(command)] = float(torch.linalg.vector_norm(poses[0] - poses[1]))
            open_gap = separations["0.0"]
            closed_gap = separations["1.0"]
            travel = abs(open_gap - closed_gap)
            gripper_probe["finger_separation_m"] = separations
            gripper_probe["separation_travel_m"] = travel
            # Below this the two commands are indistinguishable and the
            # convention stays unmeasured rather than being guessed from noise.
            if travel < 1.0e-3:
                gripper_probe["status"] = "ambiguous"
                gripper_probe["blockers"] = ["gripper_convention_travel_below_floor"]
            else:
                closes_at = 1.0 if closed_gap < open_gap else 0.0
                gripper_probe["status"] = "measured"
                gripper_probe["blockers"] = []
                gripper_probe["closed_command"] = closes_at
                gripper_probe["open_command"] = 1.0 - closes_at
        else:
            gripper_probe["status"] = "blocked"
            gripper_probe["blockers"] = ["gripper_convention_finger_bodies_missing"]
        gripper_probe["probe_digest"] = _canonical_digest(gripper_probe)
        # The probe reset the environment, so restore the canonical hold state
        # the retained evidence above was measured under.
        env.reset(seed=20260806)
        _phase("gripper_convention_probe", gripper_probe["status"])
        timings_seconds["gripper_convention_probe"] = round(time.monotonic() - phase_started, 6)

        # The canonical hold is judged on the hold alone.  Evaluating it after the
        # approach would measure motion the canonical condition never contained.
        approved_can = env.unwrapped.scene["approved_can"]
        can_pose = _to_torch(approved_can.data.root_pose_w)[0]
        if not torch.isfinite(can_pose).all():
            raise RuntimeError("approved_can_state_nonfinite")
        object_stability = _assert_canonical_object_stability(
            hold_start_can_pose,
            can_pose,
        )
        canonical_hold_can_pose = can_pose.clone()

        # --- preregistered wrist approach -------------------------------------
        # At the canonical reset pose the wrist camera cannot see the approved
        # can (63.8 deg off axis against a 28.4 deg vertical half FOV), so wrist
        # observability is established by servoing the end effector toward the
        # object and capturing along the way.  A failure here is recorded, never
        # fatal: the hold-phase evidence above must survive regardless.
        wrist_approach_started = time.monotonic()
        approach_frames: list[dict[str, Any]] = []
        approach_arrivals: list[dict[str, Any]] = []
        approach_body_names: list[str] = []
        wrist_camera_driven = False
        approach_ik_succeeded = True
        approach_error: str | None = None
        approach_object_displacement_m = 0.0
        approach_object_offset_m: list[float] = [0.0, 0.0, 0.0]
        approach_object_trace: list[dict[str, Any]] = []
        episode_start_samples: list[dict[str, Any]] = []
        episode_start_selection: dict[str, Any] | None = None
        camera_aim_plan: dict[str, Any] = {}
        control_ik_binding: dict[str, Any] | None = None
        control_ik_step_diagnostics: list[dict[str, Any]] = []
        approach_aborted = False
        try:
            from isaaclab.controllers import (  # noqa: PLC0415
                DifferentialIKController,
                DifferentialIKControllerCfg,
            )
            from isaaclab.utils.math import (  # noqa: PLC0415
                subtract_frame_transforms,
            )

            controller = DifferentialIKController(
                DifferentialIKControllerCfg(
                    command_type="pose", use_relative_mode=False, ik_method="dls"
                ),
                num_envs=1,
                device=env.unwrapped.device,
            )
            body_names = list(robot.data.body_names)
            end_effector_name = next(
                (
                    name
                    # base_link is the Robotiq gripper base the wrist camera
                    # hangs from, and it is a real articulation body -- an
                    # earlier selection list named a body that does not exist
                    # and silently fell through to panda_link7, one joint short
                    # of the tool.  A live run recorded the real body list:
                    # panda_link0..8, base_link, and the knuckle/finger bodies.
                    for name in (
                        "panda_hand",
                        "base_link",
                        "panda_link7",
                    )
                    if name in body_names
                ),
                body_names[-1],
            )
            body_index = body_names.index(end_effector_name)
            approach_body_names = list(body_names)
            # The official Arena mount already follows the hand for rendering;
            # never rewrite its world pose.  v79 proved that translation while
            # holding the reset orientation leaves the can outside the view for
            # every step.  Aim the rigid mount by rotating the controlled body
            # in place first while the per-step object guard stays active.
            wrist_camera_driven = False
            reset_body_pose = _to_torch(robot.data.body_pose_w)[0, body_index]
            wrist_camera = env.unwrapped.scene["wrist_camera"]
            reset_camera_position = [float(v) for v in _to_torch(wrist_camera.data.pos_w)[0]]
            reset_camera_quaternion = [
                float(v) for v in _to_torch(wrist_camera.data.quat_w_opengl)[0]
            ]
            wrist_mount_position_body, wrist_mount_quaternion_body = (
                rigid_offset_in_body_frame(
                    body_position_world=[float(value) for value in reset_body_pose[:3]],
                    body_quaternion_world_xyzw=[
                        float(value) for value in reset_body_pose[3:7]
                    ],
                    child_position_world=reset_camera_position,
                    child_quaternion_world_xyzw=reset_camera_quaternion,
                )
            )

            def _wrist_camera_evidence_pose() -> tuple[list[float], list[float]]:
                """Pose of the rendered rigid mount from the live articulation body."""

                live_body_pose = _to_torch(robot.data.body_pose_w)[0, body_index]
                return apply_rigid_offset(
                    body_position_world=[float(value) for value in live_body_pose[:3]],
                    body_quaternion_world_xyzw=[
                        float(value) for value in live_body_pose[3:7]
                    ],
                    offset_position_body=wrist_mount_position_body,
                    offset_quaternion_body_xyzw=wrist_mount_quaternion_body,
                )

            wrist_camera_driven = True
            camera_aim_target_world = approved_can_visual_center_world()
            camera_aim_solution = solve_rigid_mount_camera_aim(
                body_position_world=[float(v) for v in reset_body_pose[:3]],
                body_quaternion_world_xyzw=[float(v) for v in reset_body_pose[3:7]],
                offset_position_body=wrist_mount_position_body,
                offset_quaternion_body_xyzw=wrist_mount_quaternion_body,
                target_position_world=camera_aim_target_world,
            )
            camera_aim_quaternion = camera_aim_solution[
                "body_quaternion_world_xyzw"
            ]
            camera_aim_plan = {
                "strategy": "solve_rigid_mount_opengl_optical_axis_to_can_center",
                "camera_position_world_m": reset_camera_position,
                "camera_quaternion_world_opengl_xyzw": reset_camera_quaternion,
                "target_position_world_m": camera_aim_target_world,
                "target_body_quaternion_world_xyzw": camera_aim_quaternion,
                "max_steps": CAMERA_AIM_MAX_STEPS,
                "camera_mount_reauthored": False,
                "camera_pose_evidence_source": (
                    "live_articulation_body_times_reset_rigid_mount_offset"
                ),
                "wrist_mount_position_body_m": wrist_mount_position_body,
                "wrist_mount_quaternion_body_xyzw": wrist_mount_quaternion_body,
                "rigid_mount_aim_solution": camera_aim_solution,
            }
            # Isaac Lab e57379c drops the root row from the jacobian stack for a
            # fixed-base articulation, so the jacobian index is offset by one.
            jacobian_index = body_index - 1 if robot.is_fixed_base else body_index
            arm_joint_ids = list(range(7))
            arm_joint_names = [str(name) for name in list(robot.joint_names)[:7]]
            expected_arm_joint_names = [f"panda_joint{index}" for index in range(1, 8)]
            if arm_joint_names != expected_arm_joint_names:
                raise RuntimeError(
                    "scripted_control_arm_joint_binding_invalid:"
                    + ",".join(arm_joint_names)
                )
            base_pose = _to_torch(robot.data.root_pose_w)[0, :7]
            world_to_base_rotation = world_to_base_rotation_row_major_xyzw(
                [float(value) for value in base_pose[3:7]]
            )
            world_to_base_rotation_tensor = torch.tensor(
                [world_to_base_rotation],
                device=env.unwrapped.device,
                dtype=torch.float32,
            ).reshape(1, 3, 3)

            def _jacobians_world_and_root():
                """Read PhysX's world Jacobian and express both row blocks in root."""

                jacobian_world = _to_torch(robot.root_view.get_jacobians())[
                    :, jacobian_index, :, arm_joint_ids
                ]
                jacobian_root = jacobian_world.clone()
                jacobian_root[:, :3, :] = torch.bmm(
                    world_to_base_rotation_tensor,
                    jacobian_world[:, :3, :],
                )
                jacobian_root[:, 3:, :] = torch.bmm(
                    world_to_base_rotation_tensor,
                    jacobian_world[:, 3:, :],
                )
                return jacobian_world, jacobian_root

            control_ik_binding = {
                "schema_version": "adp009d_scripted_control_ik_binding.v1",
                "isaac_lab_revision": ISAAC_LAB_REVISION,
                "arena_revision": ARENA_REVISION,
                "action_semantics": "ordered_absolute_joint_position_radians_plus_binary_gripper",
                "action_dimension": int(env.unwrapped.action_manager.total_action_dim),
                "arm_joint_ids": arm_joint_ids,
                "arm_joint_names": arm_joint_names,
                "controlled_body_name": end_effector_name,
                "controlled_body_index": body_index,
                "jacobian_body_index": jacobian_index,
                "fixed_base": bool(robot.is_fixed_base),
                "root_pose_world_xyzw": [float(value) for value in base_pose],
                "physx_jacobian_frame": "world",
                "controller_pose_error_frame": "robot_root",
                "world_to_root_rotation_row_major": world_to_base_rotation,
                "linear_jacobian_rows_rotated_world_to_root": True,
                "angular_jacobian_rows_rotated_world_to_root": True,
                "binding_source": (
                    "pinned_isaaclab_task_space_actions.DifferentialInverseKinematicsAction"
                ),
            }
            resolved_waypoints = [
                {
                    "waypoint_index": -1,
                    "position_world_m": [float(v) for v in reset_body_pose[:3]],
                    "quaternion_xyzw": camera_aim_quaternion,
                    "standoff_above_support_m": None,
                    "capture_frame_index": CAMERA_AIM_CAPTURE_FRAME_INDEX,
                    "steps": CAMERA_AIM_MAX_STEPS,
                    "purpose": "camera_aim_in_place",
                },
                *[
                    {
                        **waypoint,
                        "quaternion_xyzw": camera_aim_quaternion,
                        "purpose": "camera_aimed_translation_fallback",
                    }
                    for waypoint in approach_waypoints_world()
                ],
            ]
            camera_aim_plan["resolved_waypoints"] = resolved_waypoints
            camera_aim_live_solver_updates: list[dict[str, Any]] = []
            for waypoint in resolved_waypoints:
                position_base, quaternion_base = pose_world_to_base(
                    position_world=waypoint["position_world_m"],
                    quaternion_world_xyzw=waypoint["quaternion_xyzw"],
                    base_position_world=[float(v) for v in base_pose[:3]],
                    base_quaternion_world_xyzw=[float(v) for v in base_pose[3:7]],
                )
                command = torch.tensor(
                    [position_base + quaternion_base],
                    device=env.unwrapped.device,
                    dtype=torch.float32,
                )
                controller.reset()
                controller.set_command(command)
                resolved_target_world = list(waypoint["position_world_m"])
                position_control_mode = "fixed_preregistered_waypoint"
                for waypoint_step in range(int(waypoint["steps"])):
                    if waypoint["waypoint_index"] == -1:
                        live_body_pose = _to_torch(robot.data.body_pose_w)[0, body_index]
                        live_aim_command = solve_live_rigid_mount_camera_aim_command(
                            body_position_world=[float(value) for value in live_body_pose[:3]],
                            body_quaternion_world_xyzw=[
                                float(value) for value in live_body_pose[3:7]
                            ],
                            offset_position_body=wrist_mount_position_body,
                            offset_quaternion_body_xyzw=wrist_mount_quaternion_body,
                            target_position_world=camera_aim_target_world,
                        )
                        resolved_target_world = live_aim_command["body_position_world_m"]
                        position_control_mode = live_aim_command["position_control_mode"]
                        position_base, quaternion_base = pose_world_to_base(
                            position_world=resolved_target_world,
                            quaternion_world_xyzw=live_aim_command[
                                "body_quaternion_world_xyzw"
                            ],
                            base_position_world=[float(v) for v in base_pose[:3]],
                            base_quaternion_world_xyzw=[float(v) for v in base_pose[3:7]],
                        )
                        controller.set_command(
                            torch.tensor(
                                [position_base + quaternion_base],
                                device=env.unwrapped.device,
                                dtype=torch.float32,
                            )
                        )
                        if waypoint_step < 4 or waypoint_step % 20 == 0:
                            camera_aim_live_solver_updates.append(
                                {
                                    "step": waypoint_step,
                                    "body_position_world_m": resolved_target_world,
                                    "target_body_quaternion_world_xyzw": live_aim_command[
                                        "body_quaternion_world_xyzw"
                                    ],
                                    "solver_iterations": live_aim_command["solver"][
                                        "iterations"
                                    ],
                                    "solver_residual_angle_degrees": live_aim_command[
                                        "solver"
                                    ]["residual_angle_degrees"],
                                }
                            )
                    _, jacobian = _jacobians_world_and_root()
                    ee_pose_w = _to_torch(robot.data.body_pose_w)[:, body_index]
                    root_pose_w = _to_torch(robot.data.root_pose_w)
                    ee_pos_b, ee_quat_b = subtract_frame_transforms(
                        root_pose_w[:, 0:3],
                        root_pose_w[:, 3:7],
                        ee_pose_w[:, 0:3],
                        ee_pose_w[:, 3:7],
                    )
                    joint_target = controller.compute(
                        ee_pos_b,
                        ee_quat_b,
                        jacobian,
                        _to_torch(robot.data.joint_pos)[:, arm_joint_ids],
                    )
                    # Differential IK returns the whole remaining correction;
                    # commanding it outright swings the arm through the object.
                    current_arm = _to_torch(robot.data.joint_pos)[:, arm_joint_ids]
                    joint_target = current_arm + torch.clamp(
                        joint_target - current_arm,
                        -APPROACH_MAX_JOINT_STEP_RAD,
                        APPROACH_MAX_JOINT_STEP_RAD,
                    )
                    approach_action = torch.zeros_like(action)
                    approach_action[:, :7] = joint_target
                    # Hold the gripper OPEN for the whole approach.  A
                    # zero-initialised action leaves dimension seven at 0.0, and
                    # the convention probe measured 0.0 as *closed* -- so every
                    # approach step was commanding a grasp.  The per-step trace
                    # shows exactly that: the can sits still for eighteen steps,
                    # then rises to 31 mm and stops as the gripper takes it, then
                    # drifts in x and y as the arm carries it.  Five earlier
                    # hypotheses looked for contact from an arm that was in fact
                    # deliberately holding the object.
                    if gripper_probe.get("status") == "measured":
                        approach_action[:, 7] = float(gripper_probe["open_command"])
                    env.step(approach_action)
                    # Record the displacement vector, not just its magnitude: a
                    # can that is falling and a can that is being pushed both
                    # cross the same threshold, and they need different fixes.
                    approach_object_offset = (
                        _to_torch(approved_can.data.root_pose_w)[0, :3]
                        - canonical_hold_can_pose[:3]
                    )
                    approach_object_displacement_m = float(
                        torch.linalg.vector_norm(approach_object_offset)
                    )
                    # Five hypotheses about this displacement were wrong, each
                    # tested by a binary run.  A per-step trace answers what a
                    # binary cannot: when the rise begins, whether it is smooth
                    # or stepped, and whether it tracks arm motion at all.
                    if len(approach_object_trace) < 400:
                        approach_object_trace.append(
                            {
                                "step": len(approach_object_trace),
                                "offset_m": [round(float(v), 9) for v in approach_object_offset],
                                "displacement_m": round(approach_object_displacement_m, 9),
                                "ee_position_world_m": [
                                    round(float(v), 6)
                                    for v in _to_torch(robot.data.body_pose_w)[0, body_index, :3]
                                ],
                            }
                        )
                    approach_object_offset_m = [float(v) for v in approach_object_offset]
                    wrist_observability = _approved_can_observability(
                        env.unwrapped.scene["wrist_camera"]
                    )
                    external_observability = _approved_can_observability(
                        env.unwrapped.scene["external_camera"]
                    )
                    episode_start_samples.append(
                        {
                            "step": len(episode_start_samples),
                            "joint_position_rad": [
                                float(v) for v in _to_torch(robot.data.joint_pos)[0, :7]
                            ],
                            "object_offset_m": list(approach_object_offset_m),
                            **wrist_observability,
                            "external_observability": external_observability,
                        }
                    )
                    episode_start_selection = select_wrist_observable_episode_start(
                        episode_start_samples
                    )
                    if episode_start_selection["status"] == "ready":
                        break
                    if (
                        any(
                            abs(value) > EPISODE_START_OBJECT_OFFSET_TOLERANCE_M
                            for value in approach_object_offset_m
                        )
                        or approach_object_displacement_m > APPROACH_MAX_OBJECT_DISPLACEMENT_M
                    ):
                        approach_aborted = True
                        break
                # "IK ran without raising" is not "the arm arrived": the servo
                # clamps joint motion over a fixed step budget and can finish
                # far short.  Record where the end effector actually ended up so
                # a wrist that never saw the object can be told apart from a
                # wrist that never got there.
                achieved_world = _to_torch(robot.data.body_pose_w)[0, body_index, :3]
                target_world = resolved_target_world
                # Measure the tool's true clearance over the can rather than
                # inferring it from published gripper geometry: the lowest
                # gripper body is what actually collides.
                gripper_indices = [
                    body_names.index(name)
                    for name in APPROACH_GRIPPER_BODY_NAMES
                    if name in body_names
                ]
                lowest_gripper_z = (
                    float(_to_torch(robot.data.body_pose_w)[0, gripper_indices, 2].min())
                    if gripper_indices
                    else None
                )
                can_top_z = SUPPORT_HEIGHT_M + APPROVED_CAN_TOP_ABOVE_SUPPORT_M
                # The gripper clearance above only watches gripper bodies, and a
                # run measured 0.095 m of it while still displacing the can by
                # 10.0 mm -- so whatever is pushing the can is elsewhere on the
                # arm.  Name it rather than infer it: report the body closest to
                # the can across the whole articulation.
                can_position = _to_torch(approved_can.data.root_pose_w)[0, :3]
                all_body_positions = _to_torch(robot.data.body_pose_w)[0, :, :3]
                body_distances = torch.linalg.vector_norm(all_body_positions - can_position, dim=-1)
                nearest_index = int(torch.argmin(body_distances))
                approach_arrivals.append(
                    {
                        "waypoint_index": waypoint["waypoint_index"],
                        "lowest_gripper_body_z_m": lowest_gripper_z,
                        "nearest_body_to_can": body_names[nearest_index],
                        "nearest_body_distance_to_can_m": float(body_distances[nearest_index]),
                        "body_distances_to_can_m": {
                            name: round(float(body_distances[index]), 6)
                            for index, name in enumerate(body_names)
                        },
                        "approved_can_top_z_m": can_top_z,
                        "gripper_clearance_over_can_m": (
                            None if lowest_gripper_z is None else lowest_gripper_z - can_top_z
                        ),
                        "target_position_world_m": [float(v) for v in target_world],
                        "preregistered_position_world_m": [
                            float(v) for v in waypoint["position_world_m"]
                        ],
                        "position_control_mode": position_control_mode,
                        "achieved_position_world_m": [float(v) for v in achieved_world],
                        "position_error_m": float(
                            sum(
                                (float(achieved_world[axis]) - float(target_world[axis])) ** 2
                                for axis in range(3)
                            )
                            ** 0.5
                        ),
                        "end_effector_body": end_effector_name,
                    }
                )
                for camera_name in ("external_camera", "wrist_camera", "external_camera_2"):
                    wrist_pose_override = (
                        _wrist_camera_evidence_pose()
                        if camera_name == "wrist_camera"
                        else None
                    )
                    approach_frames.append(
                        _save_camera(
                            output,
                            camera_name,
                            env.unwrapped.scene[camera_name],
                            frame_index=int(waypoint["capture_frame_index"]),
                            sim_time=float(
                                env.unwrapped.episode_length_buf[0].item()
                                * cfg.sim.dt
                                * cfg.decimation
                            ),
                            require_metric_depth=(camera_name != "external_camera_2"),
                            pose_override=wrist_pose_override,
                            pose_source=(
                                "live_articulation_body_times_reset_rigid_mount_offset"
                                if wrist_pose_override is not None
                                else "isaac_sensor_buffer"
                            ),
                        )
                    )
                _phase(
                    f"wrist_approach_waypoint_{waypoint['waypoint_index']}",
                    "blocked" if approach_aborted else "completed",
                )
                if approach_aborted:
                    break
                if (
                    episode_start_selection is not None
                    and episode_start_selection.get("status") == "ready"
                ):
                    break
            camera_aim_plan["live_solver_updates"] = camera_aim_live_solver_updates
        except Exception as exc:  # noqa: BLE001 - recorded, never fatal
            approach_ik_succeeded = False
            approach_error = f"{type(exc).__name__}: {exc}"
            _phase("wrist_approach", "blocked")
        # Always produce a selection receipt, including when IK raised before
        # the first sample.  Absence is a blocker, not an implicit return to the
        # canonical reset pose that the wrist camera cannot use.
        episode_start_selection = select_wrist_observable_episode_start(episode_start_samples)
        timings_seconds["wrist_approach"] = round(time.monotonic() - wrist_approach_started, 6)

        # --- learned policy episode --------------------------------------------
        # Everything this needs is now measured rather than assumed: the gripper
        # convention came from the probe above, the destination was frozen from
        # the sealed support triangles before any outcome existed, and the
        # observation and action adapters are pinned by their own tests.
        #
        # Recorded, never fatal.  The micro-check's own evidence must survive a
        # policy that is absent, unreachable, or wrong, and a run that produced
        # no episode must say so rather than look like a policy that scored zero.
        policy_episode: dict[str, Any] | None = None
        policy_episode_error: str | None = None
        policy_episode_skipped_reason: str | None = None
        control_episode: dict[str, Any] | None = None
        control_episode_error: str | None = None
        controls_requested = str(
            os.environ.get("BLUEPRINT_ADP009D_CONTROLS") or ""
        ).strip().lower() in {"1", "true", "yes"}
        episode_start_restore_receipts: list[dict[str, Any]] = []
        candidate_ids = [
            part.strip()
            for part in (os.environ.get("BLUEPRINT_ADP009D_POLICY_CANDIDATE") or "").split(",")
            if part.strip()
        ]
        candidate_id = candidate_ids[0] if candidate_ids else ""
        if not candidate_ids and not controls_requested:
            policy_episode_skipped_reason = "no_policy_candidate_bound"
        elif gripper_probe.get("status") != "measured":
            policy_episode_skipped_reason = f"gripper_convention_{gripper_probe.get('status')}"
        elif episode_start_selection.get("status") != "ready":
            policy_episode_skipped_reason = "wrist_observable_episode_start_not_ready"
        if (
            (candidate_ids or controls_requested)
            and gripper_probe.get("status") == "measured"
            and episode_start_selection.get("status") == "ready"
        ):
            _phase("policy_episode")
            phase_started = time.monotonic()
            try:
                from adp009d_isaac_episode_adapter import IsaacEpisodeAdapter
                from adp009d_isaac_episode_adapter import (
                    controlled_body_pose_for_grasp_frame_target,
                    grasp_frame_target_for_task_space_strategy,
                    semantic_finger_tool_midpoint_world_m,
                )
                from adp009d_droid_action_execution import GripperConvention
                from adp009d_control_episode import (
                    CONTROL_PLAN_FILENAME,
                    run_required_controls,
                )
                from adp009d_episode_batch import (
                    run_episode_batch,
                    summarize_candidate_batches,
                )

                destination_path = Path(runtime / "adp009d_task_destination.v1.json")
                destination = json.loads(destination_path.read_text(encoding="utf-8"))

                selected_episode_start = episode_start_selection["selected"]

                def _restore_wrist_observable_episode_start() -> None:
                    """Reset, replay, and verify the admitted policy start pose."""

                    target = torch.tensor(
                        [selected_episode_start["joint_position_rad"]],
                        device=env.unwrapped.device,
                        dtype=torch.float32,
                    )
                    env.reset(seed=20260806)
                    restore_steps = 0
                    for _ in range(EPISODE_START_RESTORE_MAX_STEPS):
                        current = _to_torch(robot.data.joint_pos)[:, :7]
                        restore_action = torch.zeros_like(action)
                        restore_action[:, :7] = current + torch.clamp(
                            target - current,
                            -APPROACH_MAX_JOINT_STEP_RAD,
                            APPROACH_MAX_JOINT_STEP_RAD,
                        )
                        restore_action[:, 7] = float(gripper_probe["open_command"])
                        env.step(restore_action)
                        restore_steps += 1
                        live_can_offset = (
                            _to_torch(approved_can.data.root_pose_w)[0, :3]
                            - canonical_hold_can_pose[:3]
                        )
                        if any(
                            abs(float(value)) > EPISODE_START_OBJECT_OFFSET_TOLERANCE_M
                            for value in live_can_offset
                        ):
                            # A longer replay horizon must never buy convergence
                            # by disturbing the sealed task object.  The final
                            # receipt retains the measured offset and typed
                            # object-moved blocker.
                            break
                        remaining = torch.max(
                            torch.abs(target - _to_torch(robot.data.joint_pos)[:, :7])
                        )
                        if float(remaining) <= (EPISODE_START_JOINT_TOLERANCE_RAD / 3.0):
                            break

                    restored_joints = _to_torch(robot.data.joint_pos)[0, :7]
                    restored_can_offset = (
                        _to_torch(approved_can.data.root_pose_w)[0, :3]
                        - canonical_hold_can_pose[:3]
                    )
                    restored_observability = _approved_can_observability(
                        env.unwrapped.scene["wrist_camera"]
                    )
                    restored_external_observability = _approved_can_observability(
                        env.unwrapped.scene["external_camera"]
                    )
                    restore_receipt = validate_wrist_observable_episode_start_restore(
                        selected_joint_position_rad=target[0],
                        restored_joint_position_rad=restored_joints,
                        object_offset_m=restored_can_offset,
                        approved_task_object_pixel_count=(
                            restored_observability["approved_task_object_pixel_count"]
                        ),
                        approved_task_object_pixel_fraction=(
                            restored_observability["approved_task_object_pixel_fraction"]
                        ),
                        approved_task_object_within_frame_margin=(
                            restored_observability["approved_task_object_within_frame_margin"]
                        ),
                        external_approved_task_object_pixel_count=(
                            restored_external_observability["approved_task_object_pixel_count"]
                        ),
                        external_approved_task_object_pixel_fraction=(
                            restored_external_observability["approved_task_object_pixel_fraction"]
                        ),
                        external_approved_task_object_within_frame_margin=(
                            restored_external_observability[
                                "approved_task_object_within_frame_margin"
                            ]
                        ),
                        approved_task_object_bbox_xyxy=(
                            restored_observability["approved_task_object_bbox_xyxy"]
                        ),
                        approved_task_object_centroid_xy_fraction=(
                            restored_observability["approved_task_object_centroid_xy_fraction"]
                        ),
                        frame_resolution_hw=restored_observability["frame_resolution_hw"],
                        restore_steps=restore_steps,
                    )
                    episode_start_restore_receipts.append(restore_receipt)
                    if restore_receipt["status"] != "ready":
                        raise RuntimeError(
                            "wrist_observable_episode_start_restore_failed:"
                            + ",".join(restore_receipt["blockers"])
                        )

                # Prove the callback before constructing a policy client.  A
                # failed replay therefore cannot spend inference or masquerade
                # as a policy that chose to do nothing.
                _restore_wrist_observable_episode_start()
                control_ik_call_counter = [0]

                def _scripted_pose_action_callback(
                    *,
                    target_position_world_m,
                    target_quaternion_world_xyzw,
                    gripper_command,
                    max_joint_delta_rad,
                    max_task_space_translation_step_m,
                    orientation_tolerance_deg,
                    task_space_translation_strategy,
                ):
                    """One bounded native differential-IK action for a control phase."""

                    if target_quaternion_world_xyzw is None:
                        raise RuntimeError(
                            "scripted_control_task_orientation_missing"
                        )
                    body_poses = _to_torch(robot.data.body_pose_w)[0]
                    body_pose = body_poses[body_index, :7]
                    finger_indices = [
                        body_names.index("left_inner_finger"),
                        body_names.index("right_inner_finger"),
                    ]
                    raw_finger_body_midpoint = (
                        body_poses[finger_indices[0], :3]
                        + body_poses[finger_indices[1], :3]
                    ) / 2.0
                    finger_midpoint = semantic_finger_tool_midpoint_world_m(
                        left_finger_pose_world_xyzw=[
                            float(value)
                            for value in body_poses[finger_indices[0], :7]
                        ],
                        right_finger_pose_world_xyzw=[
                            float(value)
                            for value in body_poses[finger_indices[1], :7]
                        ],
                    )
                    bounded_grasp_target = (
                        grasp_frame_target_for_task_space_strategy(
                            current_position_world_m=finger_midpoint,
                            current_quaternion_world_xyzw=[
                                float(value) for value in body_pose[3:7]
                            ],
                            target_position_world_m=target_position_world_m,
                            target_quaternion_world_xyzw=(
                                target_quaternion_world_xyzw
                            ),
                            max_translation_step_m=(
                                max_task_space_translation_step_m
                            ),
                            orientation_tolerance_deg=orientation_tolerance_deg,
                            task_space_translation_strategy=(
                                task_space_translation_strategy
                            ),
                        )
                    )
                    target_body_position_world, held_body_quaternion_world = (
                        controlled_body_pose_for_grasp_frame_target(
                            current_body_position_world_m=[
                                float(value) for value in body_pose[:3]
                            ],
                            current_body_quaternion_world_xyzw=[
                                float(value) for value in body_pose[3:7]
                            ],
                            current_grasp_frame_position_world_m=[
                                float(value) for value in finger_midpoint
                            ],
                            target_grasp_frame_position_world_m=(
                                bounded_grasp_target["position_world_m"]
                            ),
                            target_body_quaternion_world_xyzw=(
                                target_quaternion_world_xyzw
                            ),
                        )
                    )
                    base_pose = _to_torch(robot.data.root_pose_w)[0, :7]
                    position_base, quaternion_base = pose_world_to_base(
                        position_world=target_body_position_world,
                        quaternion_world_xyzw=held_body_quaternion_world,
                        base_position_world=[float(v) for v in base_pose[:3]],
                        base_quaternion_world_xyzw=[float(v) for v in base_pose[3:7]],
                    )
                    command = torch.tensor(
                        [position_base + quaternion_base],
                        device=env.unwrapped.device,
                        dtype=torch.float32,
                    )
                    controller.reset()
                    controller.set_command(command)
                    jacobian_world, jacobian = _jacobians_world_and_root()
                    ee_pose_w = _to_torch(robot.data.body_pose_w)[:, body_index]
                    root_pose_w = _to_torch(robot.data.root_pose_w)
                    ee_pos_b, ee_quat_b = subtract_frame_transforms(
                        root_pose_w[:, 0:3],
                        root_pose_w[:, 3:7],
                        ee_pose_w[:, 0:3],
                        ee_pose_w[:, 3:7],
                    )
                    current_arm = _to_torch(robot.data.joint_pos)[:, arm_joint_ids]
                    joint_target = controller.compute(
                        ee_pos_b,
                        ee_quat_b,
                        jacobian,
                        current_arm,
                    )
                    bounded_target = current_arm + torch.clamp(
                        joint_target - current_arm,
                        -float(max_joint_delta_rad),
                        float(max_joint_delta_rad),
                    )
                    callback_index = control_ik_call_counter[0]
                    control_ik_call_counter[0] += 1
                    if len(control_ik_step_diagnostics) < 16 and (
                        callback_index < 4 or callback_index % 20 == 0
                    ):
                        control_ik_step_diagnostics.append(
                            {
                                "callback_index": callback_index,
                                "target_grasp_frame_position_world_m": [
                                    float(value) for value in target_position_world_m
                                ],
                                "bounded_grasp_frame_target": bounded_grasp_target,
                                "current_grasp_frame_position_world_m": [
                                    float(value) for value in finger_midpoint
                                ],
                                "raw_finger_body_midpoint_world_m": [
                                    float(value)
                                    for value in raw_finger_body_midpoint
                                ],
                                "target_controlled_body_position_world_m": [
                                    float(value) for value in target_body_position_world
                                ],
                                "current_controlled_body_position_world_m": [
                                    float(value) for value in ee_pose_w[0, :3]
                                ],
                                "target_controlled_body_position_root_m": [
                                    float(value) for value in position_base
                                ],
                                "current_controlled_body_position_root_m": [
                                    float(value) for value in ee_pos_b[0]
                                ],
                                "position_error_root_m": [
                                    float(position_base[index] - ee_pos_b[0, index])
                                    for index in range(3)
                                ],
                                "jacobian_shape": list(jacobian.shape),
                                "jacobian_world_frobenius_norm": float(
                                    torch.linalg.vector_norm(jacobian_world[0])
                                ),
                                "jacobian_root_frobenius_norm": float(
                                    torch.linalg.vector_norm(jacobian[0])
                                ),
                                "jacobian_root_rank": int(
                                    torch.linalg.matrix_rank(jacobian[0])
                                ),
                                "unbounded_joint_delta_rad": [
                                    float(value)
                                    for value in (joint_target - current_arm)[0]
                                ],
                                "bounded_joint_delta_rad": [
                                    float(value)
                                    for value in (bounded_target - current_arm)[0]
                                ],
                            }
                        )
                    scripted_action = torch.zeros_like(action)
                    scripted_action[:, :7] = bounded_target
                    scripted_action[:, 7] = float(gripper_command)
                    return [float(v) for v in scripted_action[0]]

                adapter = IsaacEpisodeAdapter(
                    env=env,
                    robot=robot,
                    approved_can=approved_can,
                    action_dim=int(env.unwrapped.action_manager.total_action_dim),
                    reset_seed=20260806,
                    to_torch=_to_torch,
                    gripper_closed_width_m=float(
                        gripper_probe["finger_separation_m"][str(gripper_probe["closed_command"])]
                    ),
                    gripper_open_width_m=float(
                        gripper_probe["finger_separation_m"][str(gripper_probe["open_command"])]
                    ),
                    reset_callback=_restore_wrist_observable_episode_start,
                    simulation_step_seconds=float(cfg.sim.dt * cfg.decimation),
                    scripted_pose_action_callback=_scripted_pose_action_callback,
                    camera_pose_callback=lambda camera_name: (
                        _wrist_camera_evidence_pose()
                        if camera_name == "wrist_camera"
                        else None
                    ),
                    contact_sensor=env.unwrapped.scene["robot_contact"],
                    contact_envelope=live_collider.get("contact_envelope"),
                    partner_contact_sensors={
                        sensor_name: env.unwrapped.scene[sensor_name]
                        for sensor_name in CONTACT_PARTNER_SENSOR_NAMES.values()
                    },
                    backend_contact_configuration=backend_contact_configuration,
                    task_object_radius_m=APPROVED_CAN_RADIUS_M,
                    task_object_height_m=APPROVED_CAN_HEIGHT_M,
                    sage_collision_contact_sensors={
                        sensor_name: env.unwrapped.scene[sensor_name]
                        for sensor_name in CONTACT_SAGE_COLLISION_SENSOR_NAMES.values()
                    },
                )
                probe_dynamics = adapter.read_arm_dynamics_observation()
                partner_forces = probe_dynamics.get(
                    "body_contact_partner_force_world_n"
                )
                sage_collision_forces = probe_dynamics.get(
                    "body_contact_sage_collision_force_world_n"
                )
                net_forces = probe_dynamics.get("body_contact_force_world_n")
                probe_sample = adapter.read_object_sample()
                probe_camera_inputs = adapter.read_evaluation_camera_inputs()
                if (
                    int(env.unwrapped.action_manager.total_action_dim) != 8
                    or not isinstance(net_forces, dict)
                    or not net_forces
                    or not isinstance(partner_forces, dict)
                    or not partner_forces
                    or not isinstance(sage_collision_forces, dict)
                    or not sage_collision_forces
                    or "closest_geometric_clearance_m" not in probe_sample
                    or set(probe_camera_inputs) != {"external", "wrist", "overview"}
                ):
                    raise RuntimeError(
                        "adp009d_backend_native_capability_probe_failed"
                    )
                backend_probe = {
                    "schema_version": "adp009d_physics_backend_probe.v1",
                    "status": "passed",
                    "physics_backend": backend,
                    "backend_profile_digest": backend_profile["profile_digest"],
                    "backend_active_at_simulation_construction": True,
                    "backend_switch_attempted": False,
                    "backend_switch_observed": False,
                    "runtime_identity": backend_profile["runtime_identity"],
                    "observed_runtime_distributions": runtime_import_preflight,
                    "source_bindings": backend_profile["source_bindings"],
                    "capabilities": {
                        name: True
                        for name in backend_profile["required_capabilities"]
                    },
                    "capability_measurements": {
                        "action_dimension": 8,
                        "gripper_convention_probe_digest": gripper_probe[
                            "probe_digest"
                        ],
                        "camera_ids": sorted(probe_camera_inputs),
                        "closest_geometric_clearance_m": probe_sample[
                            "closest_geometric_clearance_m"
                        ],
                        "closest_geometric_clearance_metric": probe_sample[
                            "closest_geometric_clearance_metric"
                        ],
                    },
                    "solver_configuration": backend_profile[
                        "solver_configuration"
                    ],
                    "contact_readback": {
                        "force_vectors_world_n": list(net_forces.values()),
                        "partner_force_vectors_world_n": list(
                            partner_forces.values()
                        ),
                        "partner_filter": _contact_partner_filter_kwargs(backend),
                        "sage_collision_force_vectors_world_n": list(
                            sage_collision_forces.values()
                        ),
                        "sage_collision_filter": _sage_collision_filter_kwargs(
                            backend
                        ),
                    },
                    "asset_conversion": {
                        "source_asset_digest": EXPECTED_ASSETS[
                            "approved_can.usda"
                        ],
                        "converted_model_digest": (
                            APPROVED_CAN_ADAPTER_SHA256
                            if backend == "physx"
                            else live_collider["converted_model_digest"]
                        ),
                        "silently_ignored_settings": [],
                        "physx_sdf_overlay_loaded": backend == "physx",
                        "physx_only_fields_observed": [],
                        "robot_source_asset_digest": (
                            DROID_FRANKA_ROBOTIQ_USD_DIGEST
                            if backend == "newton"
                            else None
                        ),
                        "robot_inertial_overlay_contract_digest": (
                            backend_profile["asset_conversion"][
                                "robot_inertial_overlay"
                            ]["overlay_digest"]
                            if backend == "newton"
                            else None
                        ),
                        "robot_inertial_overlay_status": (
                            newton_robot_inertial_overlay_receipt["status"]
                            if backend == "newton"
                            and newton_robot_inertial_overlay_receipt is not None
                            else None
                        ),
                        "robot_inertial_overlay_receipt_digest": (
                            newton_robot_inertial_overlay_receipt[
                                "receipt_digest"
                            ]
                            if backend == "newton"
                            and newton_robot_inertial_overlay_receipt is not None
                            else None
                        ),
                        "robot_source_mutated": (
                            False if backend == "newton" else None
                        ),
                        "newton_actuator_limit_mapping_contract_digest": (
                            backend_profile["actuator_limit_mapping"]["mapping_digest"]
                            if backend == "newton"
                            else None
                        ),
                        "newton_actuator_limit_mapping_status": (
                            newton_actuator_limit_mapping["status"]
                            if backend == "newton"
                            and newton_actuator_limit_mapping is not None
                            else None
                        ),
                        "newton_actuator_limit_mapping_receipt_digest": (
                            newton_actuator_limit_mapping["receipt_digest"]
                            if backend == "newton"
                            and newton_actuator_limit_mapping is not None
                            else None
                        ),
                    },
                    "contact_buffer": {
                        "nconmax": 1024 if backend == "newton" else None,
                        "overflow_observed": False,
                    },
                    "policy_query_count": 0,
                    "candidate_outcomes_accessed": False,
                    "task_success_claimed": False,
                    "physical_claimed": False,
                    "probe_digest": "",
                }
                backend_probe["probe_digest"] = _canonical_digest(
                    backend_probe, digest_field="probe_digest"
                )
                native_probe_blockers = validate_backend_probe(
                    backend_probe, profile=backend_profile
                )
                if native_probe_blockers:
                    raise RuntimeError(
                        "adp009d_backend_native_probe_invalid:"
                        + ",".join(native_probe_blockers)
                    )
                convention = GripperConvention(
                    closed_command=float(gripper_probe["closed_command"]),
                    open_command=float(gripper_probe["open_command"]),
                    measured_by_probe=True,
                )

                def _client_for(receipt: dict[str, Any]):
                    """Bind whichever transport this candidate speaks.

                    The receipt records the port the worker actually chose, so
                    the episode connects to what started rather than a default.
                    """

                    if receipt.get("transport") == "groot_zmq":
                        try:  # flat provider bundle
                            from groot_n17_droid_policy_runtime import (
                                GrootN17DroidPolicyClient,
                                GrootN17DroidPolicySpec,
                            )
                        except ModuleNotFoundError:  # repository package
                            from .groot_n17_droid_policy_runtime import (
                                GrootN17DroidPolicyClient,
                                GrootN17DroidPolicySpec,
                            )

                        worker_identity = receipt.get("worker_identity_receipt")
                        if not isinstance(worker_identity, dict):
                            raise RuntimeError("groot_worker_identity_receipt_missing_from_server")
                        return GrootN17DroidPolicyClient(
                            spec=GrootN17DroidPolicySpec(),
                            worker_identity_receipt=worker_identity,
                            host="127.0.0.1",
                            port=int(receipt["port"]),
                        )
                    from openpi_client import websocket_client_policy

                    class _OpenPiEpisodeClient:
                        """Unwrap the response the way the readiness probe does.

                        openpi returns {"actions": ...} and the episode passes
                        whatever it gets straight to the action planner, which
                        tried to build a float out of the dict.  The server
                        worker already unwrapped it when proving readiness, so
                        the round trip looked healthy while the episode could
                        not use the same reply.
                        """

                        def __init__(self, host: str, port: int) -> None:
                            self._client = websocket_client_policy.WebsocketClientPolicy(
                                host=host, port=port
                            )

                        def infer(self, observation):
                            response = self._client.infer(observation)
                            if isinstance(response, dict):
                                return response["actions"]
                            return response

                    return _OpenPiEpisodeClient("127.0.0.1", int(receipt["port"]))

                out_dir = Path(os.environ["BLUEPRINT_ADP009D_OUTPUT_DIR"])
                controls_admitted = True
                if controls_requested:
                    _phase("scenario_controls")
                    try:
                        scenario_instance_path = (
                            runtime / "adp009d_scenario_instance.v1.json"
                        )
                        if not scenario_instance_path.is_file():
                            raise RuntimeError(
                                "adp009d_control_scenario_instance_missing"
                            )
                        scenario_instance = json.loads(
                            scenario_instance_path.read_text(encoding="utf-8")
                        )
                        control_plan_path = runtime / CONTROL_PLAN_FILENAME
                        if not control_plan_path.is_file():
                            raise RuntimeError("adp009d_control_plan_missing")
                        expected_control_plan = json.loads(
                            control_plan_path.read_text(encoding="utf-8")
                        )
                        control_episode = run_required_controls(
                            environment=adapter,
                            scenario_instance=scenario_instance,
                            expected_control_plan=expected_control_plan,
                            gripper_open_command=convention.open_command,
                            gripper_closed_command=convention.closed_command,
                            output_dir=(
                                out_dir
                                / "controls"
                                / str(scenario_instance["cell_id"])
                            ),
                        )
                        controls_admitted = (
                            control_episode.get("cell_admitted_for_policy_execution")
                            is True
                        )
                        _phase(
                            "scenario_controls",
                            "completed" if controls_admitted else "blocked",
                        )
                    except Exception as exc:  # noqa: BLE001 - evidence, not policy
                        control_episode_error = f"{type(exc).__name__}: {exc}"
                        controls_admitted = False
                        _phase("scenario_controls", "blocked")
                batches = []
                for bound_candidate in candidate_ids:
                    _phase(f"policy_batch_{bound_candidate}")
                    if not controls_admitted:
                        batches.append(
                            {
                                "candidate_id": bound_candidate,
                                "status": "blocked",
                                "blockers": [
                                    "scenario_controls_not_admitted_before_policy"
                                ],
                            }
                        )
                        _phase(f"policy_batch_{bound_candidate}", "blocked")
                        continue
                    receipt_path = out_dir / (
                        f"adp009d_policy_server_receipt.{bound_candidate}.json"
                    )
                    if not receipt_path.is_file():
                        batches.append(
                            {
                                "candidate_id": bound_candidate,
                                "status": "blocked",
                                "blockers": ["policy_server_receipt_missing"],
                            }
                        )
                        _phase(f"policy_batch_{bound_candidate}", "blocked")
                        continue
                    server_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    if server_receipt.get("status") != "ready":
                        # One candidate failing to serve must not deny the other
                        # its episodes: a comparison with one arm missing is
                        # still evidence, a run that aborts produces none.
                        batches.append(
                            {
                                "candidate_id": bound_candidate,
                                "status": "blocked",
                                "blockers": [
                                    f"policy_server_not_ready:{server_receipt.get('status')}"
                                ],
                            }
                        )
                        _phase(f"policy_batch_{bound_candidate}", "blocked")
                        continue
                    batch = run_episode_batch(
                        environment=adapter,
                        policy=_client_for(server_receipt),
                        candidate_id=bound_candidate,
                        destination_position_world_m=destination["position_world_m"],
                        prompt="pick up the can and place it on the counter",
                        gripper=convention,
                        episodes=int(os.environ.get("BLUEPRINT_ADP009D_EPISODES", "3")),
                        media_output_dir=out_dir,
                    )
                    batch["transport"] = server_receipt.get("transport")
                    batches.append(batch)
                    _phase(f"policy_batch_{bound_candidate}", "completed")
                policy_episode = {
                    "batches": batches,
                    "comparison": summarize_candidate_batches(
                        [b for b in batches if b.get("episodes_scored") is not None]
                    ),
                    "episode_start_restore_receipts": (episode_start_restore_receipts),
                }
                _phase("policy_episode", "completed")
            except Exception as exc:  # noqa: BLE001 - recorded, never fatal
                policy_episode_error = f"{type(exc).__name__}: {exc}"
                _phase("policy_episode", "blocked")
            timings_seconds["policy_episode"] = round(time.monotonic() - phase_started, 6)

        wrist_approach_capture = summarize_wrist_approach_capture(
            captured_frames=approach_frames,
            ik_succeeded=approach_ik_succeeded,
            object_displacement_m=approach_object_displacement_m,
            waypoint_arrivals=approach_arrivals,
        )
        wrist_approach_capture["error"] = approach_error
        # Recorded so a future attachment defect is diagnosable without a run.
        wrist_approach_capture["approved_can_offset_from_hold_m"] = approach_object_offset_m
        wrist_approach_capture["approved_can_per_step_trace"] = approach_object_trace
        wrist_approach_capture["articulation_body_names"] = approach_body_names
        wrist_approach_capture["wrist_camera_driven_from_body_pose"] = wrist_camera_driven
        wrist_approach_capture["camera_pose_metadata_refresh_enabled"] = True
        wrist_approach_capture["camera_aim_plan"] = camera_aim_plan
        wrist_approach_capture["isaaclab_quaternion_order"] = "xyzw"
        camera_rows.extend(approach_frames)
        robot = env.unwrapped.scene["robot"]
        # A run that was asked for episodes and produced none is not completed.
        # This reported completed with an empty blocker list while carrying a
        # ModuleNotFoundError in policy_episode_error, which is precisely the
        # shape of a success claim that outruns its evidence: the micro-check's
        # own checks had passed, so nothing contradicted it.  Episodes were a
        # bonus when that was written; they are the deliverable now.
        episode_blockers = _policy_episode_blockers(
            candidate_ids=candidate_ids,
            policy_episode=policy_episode,
            policy_episode_error=policy_episode_error,
        )
        if candidate_ids and episode_start_selection.get("status") != "ready":
            episode_blockers.extend(episode_start_selection.get("blockers") or [])
        if controls_requested:
            if control_episode_error:
                episode_blockers.append("scenario_controls_runtime_error")
            elif control_episode is None:
                episode_blockers.append("scenario_controls_receipt_missing")
            elif control_episode.get("cell_admitted_for_policy_execution") is not True:
                episode_blockers.extend(
                    control_episode.get("policy_execution_blockers")
                    or ["scenario_controls_not_admitted"]
                )
            if episode_start_selection.get("status") != "ready":
                episode_blockers.extend(episode_start_selection.get("blockers") or [])
        scripted_control_ik = {
            "schema_version": "adp009d_scripted_control_ik_receipt.v1",
            "binding": control_ik_binding,
            "step_diagnostics": control_ik_step_diagnostics,
            "receipt_digest": "",
        }
        scripted_control_ik["receipt_digest"] = _canonical_digest(
            scripted_control_ik,
            digest_field="receipt_digest",
        )
        return {
            "schema_version": "adp009d_native_microcheck.v1",
            "status": "completed" if not episode_blockers else "blocked",
            "blockers": sorted(set(episode_blockers)),
            "arena_revision": ARENA_REVISION,
            "isaac_lab_revision": ISAAC_LAB_REVISION,
            "workflow": "isaac_lab_manager_based_via_arena_composition",
            "runtime_import_preflight": runtime_import_preflight,
            "embodiment": "official_arena_droid_abs_joint_pos_franka_robotiq_2f_85",
            "physics": {
                "backend": backend,
                "backend_profile": backend_profile,
                "backend_profile_digest": backend_profile["profile_digest"],
                "backend_selected_at_simulation_construction": True,
                "backend_switch_attempted": False,
                "collision_cooking": collision_cooking,
                "dt_seconds": cfg.sim.dt,
                "decimation": cfg.decimation,
                "solver_configuration": backend_profile["solver_configuration"],
                "backend_contact_configuration": backend_contact_configuration,
                "contact_envelope": live_collider.get("contact_envelope"),
                "static_collider_validation": static_collider,
                "live_collider_validation": live_collider,
                "newton_robot_inertial_overlay": (
                    newton_robot_inertial_overlay_receipt
                ),
                "newton_actuator_limit_mapping": newton_actuator_limit_mapping,
                "static_sage_collision_validation": static_sage_collision,
                "live_sage_collision_validation": live_sage_collision,
                "sage_task_collision_derivative": task_collision_manifest,
                "fallback_messages": fallback_messages,
                "stability_messages": stability_messages,
                "newton_unsupported_or_ignored_settings_messages": (
                    newton_unsupported_messages
                ),
            },
            "physics_backend_probe": backend_probe,
            "reset_rows": reset_rows,
            "zero_action_step": {
                **zero_action_row,
            },
            "post_warmup_robot_joint_pos": _jsonable(_to_torch(robot.data.joint_pos)[0]),
            "post_warmup_arm_maximum_error_rad": hold_arm_maximum_error_rad,
            # Retained on the passing path too: a backend comparison needs the
            # torque the winner spent, not only that it stayed inside tolerance.
            "canonical_hold_trace": hold_trace,
            "post_warmup_approved_can_root_pose_world": _jsonable(can_pose),
            "canonical_hold_object_stability": object_stability,
            "camera_frames": camera_rows,
            "external_task_camera_plan": external_task_camera_plan,
            "overview_camera_plan": overview_camera_plan,
            "camera_warmup_frames": 40,
            "timings_seconds": timings_seconds,
            "source_target_collider_disabled_by_composed_overlay": (
                backend == "physx"
            ),
            # Deliberately named "shipped", not "rendered".  An earlier field
            # called itself rendered while only checking that the asset file
            # existed, and reported True on a run whose frames were byte-for-byte
            # the same as a run with no appearance at all.  A receipt that
            # asserts a render it never observed is worse than one that is silent.
            "aura_stage_probe": aura_stage_probe,
            "aura_appearance_shipped": _resolve_aura_appearance(runtime)[0] is not None,
            # Which format actually shipped, so a render result is attributable
            # to it.  Two authorings of the same field are in play -- a
            # ParticleField Omniverse has never rendered correctly, and a NuRec
            # volume it demonstrably has -- and a receipt that does not say
            # which one was in the scene cannot settle between them.
            "aura_appearance_format": _resolve_aura_appearance(runtime)[1],
            "aura_appearance_render_verified": None,
            "aura_particlefield_prim": AURA_PARTICLEFIELD_PRIM,
            "gripper_convention_probe": gripper_probe,
            "finger_collision_envelope_probe": _json_safe(finger_collision_envelope),
            "policy_episode": policy_episode,
            "policy_episode_error": policy_episode_error,
            "policy_episode_skipped_reason": policy_episode_skipped_reason,
            "controls_requested": controls_requested,
            "control_episode": control_episode,
            "control_episode_error": control_episode_error,
            "scripted_control_ik": scripted_control_ik,
            "wrist_episode_start_selection": episode_start_selection,
            "wrist_episode_start_restore_receipts": episode_start_restore_receipts,
            "policy_candidate_bound": candidate_id or None,
            "wrist_approach_capture": wrist_approach_capture,
            "semantic_override_layer": SEMANTIC_OVERRIDE_LAYER,
            "semantic_override_layer_digest": _canonical_digest(SEMANTIC_OVERRIDE_LAYER),
            "semantic_override_layer_composed": True,
            "semantic_source_usd_mutated": False,
            "sealed_source_mutated": False,
            # Both follow the episodes rather than asserting their absence: a
            # run that queried two policies and read their outcomes must not
            # keep reporting that it did neither.
            "candidate_policy_queried": any(
                int(batch.get("episodes_scored") or 0) > 0
                or int(batch.get("episodes_failed") or 0) > 0
                for batch in ((policy_episode or {}).get("batches") or [])
            ),
            "candidate_outcomes_accessed": bool(
                (policy_episode or {}).get("comparison", {}).get("ranking")
            ),
        }
    finally:
        log.remove_message_consumer(consumer)
        if env is not None:
            env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--physics-backend", choices=("physx", "newton"), default="physx"
    )
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    app_launcher = AppLauncher(args)
    output = Path(args.output_dir).resolve()
    runtime = Path(args.runtime_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any]
    try:
        result = _run(runtime, output, args)
    except Exception as exc:
        result = {
            "schema_version": "adp009d_native_microcheck.v1",
            "status": "blocked",
            "blockers": [str(exc)],
            "exception_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
            "candidate_policy_queried": False,
            "candidate_outcomes_accessed": False,
        }
        diagnostics = getattr(exc, "diagnostics", None)
        if diagnostics is not None:
            result["diagnostics"] = _json_safe(diagnostics)
    result_path = output / RESULT_NAME
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    app_launcher.app.close()
    print(
        "BLUEPRINT_ADP009D_NATIVE_MICROCHECK_"
        + ("OK" if result["status"] == "completed" else "BLOCKED")
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
