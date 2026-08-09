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

RESULT_NAME = "adp009d_native_microcheck.json"
EXPECTED_ASSETS = {
    "approved_can.usda": "sha256:61c2a03bef425803d82cc5ef24ced5b2ccb4160923c53bb10c6ad0e3f52532ec",
    "sage_collision.usd": "sha256:b265706c24f6a8ace3ee6743fd138583c4e21d83f61b99a06fd435e6ac2d6b41",
}
APPROVED_CAN_ADAPTER_FILENAME = "approved_can_physx_sdf_adapter.usda"
TASK_COLLISION_DERIVATIVE_FILENAME = "sage_task_collision.usda"
TASK_COLLISION_MANIFEST_FILENAME = "sage_task_collision_manifest.json"
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
    "sha256:5db5bc33b72983065bd47e30db0c5945ab3cba8fb3caeb6290bf07edc7337adc"
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
    return {
        "prim_path": prim_path,
        "applied_schemas": applied_schemas,
        "approximation": str(approximation),
        **settings,
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


def _build_environment(runtime: Path, args: argparse.Namespace):
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.utils.pose import Pose

    class SpawnerObject(Object):
        """Use Arena's composition seam without importing its full asset registry."""

        def __init__(self, *, name: str, prim_path: str, spawner_cfg: Any):
            self.spawner_cfg = spawner_cfg
            super().__init__(
                name=name,
                prim_path=prim_path,
                object_type=ObjectType.SPAWNER,
            )

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
    approved_can = Object(
        name="approved_can",
        object_type=ObjectType.RIGID,
        usd_path=str(runtime / "assets" / APPROVED_CAN_ADAPTER_FILENAME),
        initial_pose=Pose(position_xyz=CAN_START_POSITION_M),
        spawn_cfg_addon={
            "semantic_tags": _semantic_tags("approved_can"),
            "rigid_props": sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
                max_depenetration_velocity=5.0,
                enable_gyroscopic_forces=True,
            ),
        },
    )
    light = SpawnerObject(
        name="light",
        prim_path="/World/Light",
        spawner_cfg=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=1500.0,
        ),
    )
    scene = Scene(
        assets=[sage, approved_can, light]
        + ([aura_appearance] if aura_appearance is not None else [])
    )
    _phase("sealed_scene_configuration", "completed")

    def configure(cfg):
        from isaaclab_physx.physics import PhysxCfg

        cfg.sim.dt = 1.0 / 120.0
        cfg.seed = 20260806
        cfg.sim.render_interval = 8
        cfg.decimation = 8
        cfg.episode_length_s = 5.0
        cfg.sim.physics = PhysxCfg(
            solver_type=1,
            enable_enhanced_determinism=True,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**15,
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
    return env, cfg, torch, external_task_camera_plan, overview_camera_plan


def _preflight_environment_imports() -> dict[str, str]:
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

    return {
        name: metadata.version(name)
        for name in (
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
        )
    }


def _run(runtime: Path, output: Path, args: argparse.Namespace) -> dict[str, Any]:
    for name, digest in EXPECTED_ASSETS.items():
        path = runtime / "assets" / name
        if not path.is_file() or _sha256(path) != digest:
            raise RuntimeError(f"sealed_asset_binding_invalid:{name}")

    adapter_path = runtime / "assets" / APPROVED_CAN_ADAPTER_FILENAME
    if not adapter_path.is_file() or _sha256(adapter_path) != APPROVED_CAN_ADAPTER_SHA256:
        raise RuntimeError("sealed_asset_binding_invalid:approved_can_physx_sdf_adapter.usda")
    from pxr import Usd

    adapter_stage = Usd.Stage.Open(str(adapter_path))
    if adapter_stage is None:
        raise RuntimeError("approved_can_physx_sdf_adapter_unreadable")
    static_collider = _inspect_physx_sdf_collider(adapter_stage, APPROVED_CAN_SOURCE_COLLIDER_PRIM)
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
    if (
        expected_sage_profile != SAGE_RUNTIME_PROFILE
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

    _phase("physx_collision_cooking_configuration")
    collision_cooking = _configure_physx_collision_cooking()
    _phase("physx_collision_cooking_configuration", "completed")

    import omni.log

    fallback_messages: list[str] = []
    stability_messages: list[str] = []

    def on_log(channel, level, module, filename, func, line_no, message, pid, tid, timestamp):
        del channel, level, module, filename, func, line_no, pid, tid, timestamp
        if PHYSX_FALLBACK_MARKER in message:
            fallback_messages.append(str(message))
        if PHYSX_TRIANGLE_STABILITY_MARKER in message:
            stability_messages.append(str(message))

    log = omni.log.get_log()
    consumer = log.add_message_consumer(on_log)
    env = None
    timings_seconds: dict[str, float] = {}
    external_task_camera_plan: dict[str, Any] | None = None
    try:
        _phase("runtime_import_preflight")
        runtime_import_preflight = _preflight_environment_imports()
        _phase("runtime_import_preflight", "completed")
        _phase("environment_build")
        phase_started = time.monotonic()
        (
            env,
            cfg,
            torch,
            external_task_camera_plan,
            overview_camera_plan,
        ) = _build_environment(runtime, args)
        timings_seconds["environment_build"] = round(time.monotonic() - phase_started, 6)
        log.flush()
        _phase("environment_build", "completed")
        _fail_on_physx_collision_fallback(fallback_messages)
        _fail_on_physx_collision_stability(stability_messages)
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

        live_collider = _inspect_physx_sdf_collider(live_stage, APPROVED_CAN_LIVE_COLLIDER_PRIM)
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
            _fail_on_physx_collision_fallback(fallback_messages)
            _fail_on_physx_collision_stability(stability_messages)
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
        _fail_on_physx_collision_fallback(fallback_messages)
        _fail_on_physx_collision_stability(stability_messages)
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
        for warmup_index in range(warmup_frames):
            observation, reward, terminated, truncated, info = env.step(hold_action)
            if (warmup_index + 1) % marker_every == 0:
                log.flush()
                _fail_on_physx_collision_fallback(fallback_messages)
                _fail_on_physx_collision_stability(stability_messages)
                _phase(f"camera_warmup_{warmup_index + 1}", "completed")
        timings_seconds[f"camera_warmup_{warmup_frames}_frames"] = round(
            time.monotonic() - phase_started, 6
        )
        hold_arm_maximum_error_rad = _assert_arm_pose(
            _to_torch(env.unwrapped.scene["robot"].data.joint_pos)[0],
            RESET_JOINTS,
            tolerance_rad=HOLD_ARM_TOLERANCE_RAD,
            blocker="canonical_hold_arm_pose_drift",
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
                    bounded_absolute_joint_setpoint,
                    controlled_body_pose_for_grasp_frame_target,
                )
                from adp009d_droid_action_execution import GripperConvention
                from adp009d_control_episode import run_required_controls
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
                control_ik_last_commanded_joint_positions_rad: list[
                    list[float] | None
                ] = [None]

                def _reset_scripted_pose_controller_state() -> None:
                    control_ik_last_commanded_joint_positions_rad[0] = None

                def _scripted_pose_action_callback(
                    *,
                    target_position_world_m,
                    target_quaternion_world_xyzw,
                    gripper_command,
                    max_joint_delta_rad,
                    max_joint_setpoint_lead_rad,
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
                    finger_midpoint = (
                        body_poses[finger_indices[0], :3]
                        + body_poses[finger_indices[1], :3]
                    ) / 2.0
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
                                target_position_world_m
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
                    current_arm_values = [float(value) for value in current_arm[0]]
                    joint_target_values = [float(value) for value in joint_target[0]]
                    previous_command_values = (
                        current_arm_values
                        if control_ik_last_commanded_joint_positions_rad[0] is None
                        else control_ik_last_commanded_joint_positions_rad[0]
                    )
                    bounded_target_values = bounded_absolute_joint_setpoint(
                        measured_joint_positions_rad=current_arm_values,
                        desired_joint_positions_rad=joint_target_values,
                        previous_commanded_joint_positions_rad=previous_command_values,
                        max_command_slew_per_step_rad=float(max_joint_delta_rad),
                        max_setpoint_lead_rad=float(max_joint_setpoint_lead_rad),
                    )
                    control_ik_last_commanded_joint_positions_rad[0] = list(
                        bounded_target_values
                    )
                    bounded_target = torch.tensor(
                        [bounded_target_values],
                        device=env.unwrapped.device,
                        dtype=current_arm.dtype,
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
                                "current_grasp_frame_position_world_m": [
                                    float(value) for value in finger_midpoint
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
                                "command_slew_from_previous_rad": [
                                    bounded_target_values[index]
                                    - previous_command_values[index]
                                    for index in range(len(bounded_target_values))
                                ],
                                "max_command_slew_per_step_rad": float(
                                    max_joint_delta_rad
                                ),
                                "max_setpoint_lead_rad": float(
                                    max_joint_setpoint_lead_rad
                                ),
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
                    scripted_pose_controller_reset_callback=(
                        _reset_scripted_pose_controller_state
                    ),
                    camera_pose_callback=lambda camera_name: (
                        _wrist_camera_evidence_pose()
                        if camera_name == "wrist_camera"
                        else None
                    ),
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
                        control_plan_path = runtime / "adp009d_control_plan.v5.json"
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
                "backend": "PhysX",
                "collision_cooking": collision_cooking,
                "dt_seconds": cfg.sim.dt,
                "decimation": cfg.decimation,
                "solver": "TGS",
                "enhanced_determinism": True,
                "static_collider_validation": static_collider,
                "live_collider_validation": live_collider,
                "static_sage_collision_validation": static_sage_collision,
                "live_sage_collision_validation": live_sage_collision,
                "sage_task_collision_derivative": task_collision_manifest,
                "fallback_messages": fallback_messages,
                "stability_messages": stability_messages,
            },
            "reset_rows": reset_rows,
            "zero_action_step": {
                **zero_action_row,
            },
            "post_warmup_robot_joint_pos": _jsonable(_to_torch(robot.data.joint_pos)[0]),
            "post_warmup_arm_maximum_error_rad": hold_arm_maximum_error_rad,
            "post_warmup_approved_can_root_pose_world": _jsonable(can_pose),
            "canonical_hold_object_stability": object_stability,
            "camera_frames": camera_rows,
            "external_task_camera_plan": external_task_camera_plan,
            "overview_camera_plan": overview_camera_plan,
            "camera_warmup_frames": 40,
            "timings_seconds": timings_seconds,
            "source_target_collider_disabled_by_composed_overlay": True,
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
