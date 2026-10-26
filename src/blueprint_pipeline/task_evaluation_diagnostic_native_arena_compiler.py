"""Compile a diagnostic-only configured-scene result into a native packet.

This is intentionally not an alternate production episode compiler. It accepts
only the fail-closed diagnostic materialization receipt, retains its
``development_only`` ceiling, and emits construction/controls inputs without a
configured-scene revision digest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .franka_kinematics import (
    ARM_JOINT_COUNT,
    FRANKA_JOINT_LIMITS_RAD,
    forward_kinematics,
)
from .native_task_arena_packet import materialize_native_task_arena_packet
from .native_articulated_construction_inputs import materialize_articulated_construction_inputs
from .scene_placement.robot_profile import get_robot_profile
from .rigid_frame_transforms import quaternion_conjugate_xyzw
from .task_evaluation_robot_placement_agent import (
    RobotPlacementAgentError,
    validate_robot_placement_receipt,
)
from .task_evaluation_robot_placement_orientation import (
    RobotPlacementOrientationError,
    quaternion_multiply_xyzw,
    solve_task_aware_reset_joints,
)
from .task_evaluation_robot_placement_trajectory import (
    RobotPlacementTrajectoryError,
    validate_robot_placement_trajectory,
)
from .task_evaluation_rigid_relocation_native_adapter import (
    adapt_rigid_relocation_task_template,
)
from .task_evaluation_articulated_open_close_native_adapter import (
    adapt_articulated_open_close_task_template,
)
from .task_evaluation_configured_controls_deferred_inputs import (
    articulated_placement_plan_from_adapter,
)


SCHEMA_VERSION = "task_evaluation_diagnostic_native_arena_compiler_output.v1"
AUTHORITY_SCHEMA_VERSION = (
    "task_evaluation_configured_scene_diagnostic_controls_input.v1"
)
CLAIM_CEILING = (
    "development_only_downstream_construction_and_controls_diagnostic"
)
TASK_AWARE_RESET_SCHEMA_VERSION = (
    "task_evaluation_task_aware_robot_reset_derivation.v1"
)


class TaskEvaluationDiagnosticNativeArenaCompilerError(ValueError):
    """Diagnostic inputs cannot be compiled without weakening their ceiling."""


def _identity(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return "sha256:" + digest.hexdigest(), size


def _read_json(path: Path, *, blocker: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationDiagnosticNativeArenaCompilerError(blocker) from exc
    if not isinstance(value, Mapping):
        raise TaskEvaluationDiagnosticNativeArenaCompilerError(blocker)
    return dict(value)


def _vector(value: Any, length: int, *, blocker: str) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != length
    ):
        raise TaskEvaluationDiagnosticNativeArenaCompilerError(blocker)
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise TaskEvaluationDiagnosticNativeArenaCompilerError(blocker) from exc
    if not all(math.isfinite(item) for item in result):
        raise TaskEvaluationDiagnosticNativeArenaCompilerError(blocker)
    return result


def _normalize(value: Sequence[float], *, blocker: str) -> list[float]:
    norm = math.sqrt(sum(float(item) ** 2 for item in value))
    if not math.isfinite(norm) or norm <= 1.0e-9:
        raise TaskEvaluationDiagnosticNativeArenaCompilerError(blocker)
    return [float(item) / norm for item in value]


def _cross(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def _look_at(position: Sequence[float], target: Sequence[float]) -> list[float]:
    forward = _normalize(
        [target[index] - position[index] for index in range(3)],
        blocker="diagnostic_native_compiler_camera_aim_invalid",
    )
    right = _normalize(
        _cross(forward, [0.0, 0.0, 1.0]),
        blocker="diagnostic_native_compiler_camera_aim_invalid",
    )
    down = _normalize(
        _cross(forward, right),
        blocker="diagnostic_native_compiler_camera_aim_invalid",
    )
    return [
        right[0], down[0], forward[0], float(position[0]),
        right[1], down[1], forward[1], float(position[1]),
        right[2], down[2], forward[2], float(position[2]),
        0.0, 0.0, 0.0, 1.0,
    ]


def _bound_rows(authority: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if (
        authority.get("schema_version") != AUTHORITY_SCHEMA_VERSION
        or authority.get("status") != "materialized"
        or authority.get("claim_ceiling") != CLAIM_CEILING
        or authority.get("qualification_eligible") is not False
        or authority.get("configured_revision_publication_permitted") is not False
        or authority.get("evaluation_ready_promotion_permitted") is not False
        or authority.get("receipt_digest")
        != canonical_digest(authority, digest_field="receipt_digest")
    ):
        raise TaskEvaluationDiagnosticNativeArenaCompilerError(
            "diagnostic_native_compiler_authority_invalid"
        )
    rows = authority.get("materialized_inputs")
    result = {
        str(row.get("contract_path") or ""): row
        for row in rows or []
        if isinstance(row, Mapping)
    }
    if len(result) != len(rows or []):
        raise TaskEvaluationDiagnosticNativeArenaCompilerError(
            "diagnostic_native_compiler_authority_invalid"
        )
    return result


def _bound_path(
    rows: Mapping[str, Mapping[str, Any]], contract_path: str
) -> Path:
    row = rows.get(contract_path)
    unresolved = Path(str((row or {}).get("path") or "")).expanduser()
    path = unresolved.resolve()
    if (
        row is None
        or unresolved.is_symlink()
        or not path.is_file()
        or row.get("full_byte_readback_passed") is not True
        or _identity(path) != (row.get("digest"), row.get("size_bytes"))
    ):
        raise TaskEvaluationDiagnosticNativeArenaCompilerError(
            f"diagnostic_native_compiler_bound_input_invalid:{contract_path}"
        )
    return path


def _world_camera(
    role: str,
    *,
    position: Sequence[float],
    target: Sequence[float],
    intrinsics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "role": role,
        "policy_input": role == "external",
        "scoring_input": False,
        "pose_frame": "world",
        "parent_prim_path": "{ENV_REGEX_NS}",
        "optical_convention": "opencv",
        "frame_from_camera_matrix": _look_at(position, target),
        "intrinsics": json.loads(json.dumps(intrinsics)),
    }


def _runtime_subject_task_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    task_spec = json.loads(json.dumps(dict(value)))
    source_subject_id = str(task_spec.get("subject_asset_id") or "")
    runtime_subject_id = re.sub(r"[^A-Za-z0-9_]", "_", source_subject_id)
    if not runtime_subject_id or not runtime_subject_id.replace("_", "a").isalnum():
        raise TaskEvaluationDiagnosticNativeArenaCompilerError(
            "diagnostic_native_compiler_subject_runtime_id_invalid"
        )
    task_spec["subject_asset_id"] = runtime_subject_id
    task_spec["source_subject_identity"] = value.get("source_subject_identity") or source_subject_id
    interaction_affordance = task_spec.get("interaction_affordance")
    if (
        not isinstance(interaction_affordance, Mapping)
        or interaction_affordance.get("subject_asset_id") != source_subject_id
    ):
        raise TaskEvaluationDiagnosticNativeArenaCompilerError(
            "diagnostic_native_compiler_interaction_affordance_invalid"
        )
    interaction_affordance = dict(interaction_affordance)
    interaction_affordance["subject_asset_id"] = runtime_subject_id
    interaction_affordance["affordance_digest"] = canonical_digest(
        interaction_affordance, digest_field="affordance_digest"
    )
    task_spec["interaction_affordance"] = interaction_affordance
    return task_spec


def _legacy_robot_placement_is_clear(
    workspace: Mapping[str, Any], placement: Mapping[str, Any]
) -> bool:
    """True only for an exact legacy pose that passed every analytic gate."""

    return bool(
        workspace.get("status") == "placement_candidate_materialized"
        and placement.get("status") == "runtime_visualization_candidate_only"
        and placement.get("mesh_triangle_aabb_overlap_probe_clear") is True
        and (placement.get("base_support_coverage") or {}).get(
            "full_sample_support_candidate"
        )
        is True
        and placement.get("analytic_reach_candidate") is True
    )


def _derive_task_aware_franka_reset(
    *,
    profile: Mapping[str, Any],
    base_pose: Mapping[str, Any],
    task_trajectory: Mapping[str, Any],
    robot_id: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Derive a task-facing reset without claiming native pose feasibility.

    The selected base orientation changes a world-frame tool orientation when
    expressed in the arm base frame.  Solve the first authored phase from that
    exact frame join, preserve all gripper joints from the immutable profile,
    and bind the derived arm reset to the trajectory.  Collision, full pose IK,
    application, and readback remain native gates.
    """

    try:
        trajectory = validate_robot_placement_trajectory(task_trajectory)
    except RobotPlacementTrajectoryError as exc:
        raise TaskEvaluationDiagnosticNativeArenaCompilerError(
            "diagnostic_native_compiler_task_trajectory_invalid"
        ) from exc
    reset = profile.get("robot_joint_reset_positions_rad")
    if not isinstance(reset, Mapping):
        raise TaskEvaluationDiagnosticNativeArenaCompilerError(
            "diagnostic_native_compiler_robot_reset_invalid"
        )
    # The arm joint names are profile data, not a naming convention to be
    # regenerated from a robot family string: a reset map is keyed by joint and
    # includes gripper joints, and every non-Franka embodiment names its chain
    # differently.  The kinematics below are still Franka-specific; this only
    # removes the assumption that the NAMES can be guessed.
    try:
        arm_joint_names = tuple(get_robot_profile(robot_id).arm_joint_names)
    except (KeyError, ValueError) as exc:
        raise TaskEvaluationDiagnosticNativeArenaCompilerError(
            "diagnostic_native_compiler_robot_profile_unknown"
        ) from exc
    if len(arm_joint_names) != ARM_JOINT_COUNT:
        raise TaskEvaluationDiagnosticNativeArenaCompilerError(
            "diagnostic_native_compiler_robot_arm_joint_names_undeclared"
        )
    try:
        nominal_arm = [float(reset[name]) for name in arm_joint_names]
    except (KeyError, TypeError, ValueError) as exc:
        raise TaskEvaluationDiagnosticNativeArenaCompilerError(
            "diagnostic_native_compiler_robot_reset_invalid"
        ) from exc
    if not all(math.isfinite(value) for value in nominal_arm):
        raise TaskEvaluationDiagnosticNativeArenaCompilerError(
            "diagnostic_native_compiler_robot_reset_invalid"
        )

    base_orientation = _normalize(
        _vector(
            base_pose.get("orientation_xyzw"),
            4,
            blocker="diagnostic_native_compiler_source_base_invalid",
        ),
        blocker="diagnostic_native_compiler_source_base_invalid",
    )
    first_phase = trajectory["phases"][0]
    target_world = _normalize(
        _vector(
            first_phase.get("orientation_world_xyzw"),
            4,
            blocker="diagnostic_native_compiler_task_trajectory_invalid",
        ),
        blocker="diagnostic_native_compiler_task_trajectory_invalid",
    )
    target_base = quaternion_multiply_xyzw(
        quaternion_conjugate_xyzw(base_orientation), target_world
    )
    try:
        solved = solve_task_aware_reset_joints(
            target_orientation_base_xyzw=target_base,
            nominal_joint_positions=nominal_arm,
            joint_limits_rad=FRANKA_JOINT_LIMITS_RAD,
            forward_kinematics=forward_kinematics,
            flange_to_grasp_orientation_xyzw=(0.0, 0.0, 1.0, 0.0),
        )
    except RobotPlacementOrientationError as exc:
        raise TaskEvaluationDiagnosticNativeArenaCompilerError(
            "diagnostic_native_compiler_task_aware_reset_unsolved"
        ) from exc

    derived_reset = {str(name): float(value) for name, value in reset.items()}
    derived_reset.update(
        dict(zip(arm_joint_names, solved["joint_positions_rad"], strict=True))
    )
    report: dict[str, Any] = {
        "schema_version": TASK_AWARE_RESET_SCHEMA_VERSION,
        "robot_id": "franka_panda",
        "source_trajectory_digest": trajectory["trajectory_digest"],
        "source_phase_id": first_phase["phase_id"],
        "base_orientation_world_xyzw": base_orientation,
        "target_orientation_world_xyzw": target_world,
        "target_orientation_base_xyzw": target_base,
        "nominal_arm_joint_positions_rad": nominal_arm,
        "derived_arm_joint_positions_rad": solved["joint_positions_rad"],
        "nominal_slew_rad": solved["nominal_slew_rad"],
        "residual_slew_rad": solved["residual_slew_rad"],
        "improvement_rad": solved["improvement_rad"],
        "native_full_pose_ik_required": True,
        "native_collision_and_contact_required": True,
        "native_reset_application_and_readback_required": True,
        "derivation_digest": "",
    }
    report["derivation_digest"] = canonical_digest(
        report, digest_field="derivation_digest"
    )
    return derived_reset, report


def diagnostic_robot_placement_inputs(
    *, diagnostic_controls_input: Mapping[str, Any],
    droid_profile_reference: Mapping[str, Any],
    task_trajectory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive the exact scene/task bindings before choosing a robot pose."""
    authority = dict(diagnostic_controls_input)
    rows = _bound_rows(authority)
    task_source = _read_json(_bound_path(rows, "task.definition"), blocker="diagnostic_native_compiler_task_invalid")
    articulated = task_source.get("strategy") == "articulated_open_close"
    adapter = (
        adapt_articulated_open_close_task_template(materialized_references={}, diagnostic_controls_input=authority)
        if articulated else adapt_rigid_relocation_task_template(materialized_references={}, diagnostic_controls_input=authority)
    )
    definition = adapter["native_task_definition"]
    task_spec = _runtime_subject_task_spec(definition["task_spec"])
    task_spec["success_criteria"] = adapter["native_success_criteria"]["criteria"]
    asset = _bound_path(rows, "diagnostic_output.replacement.usdz")
    plan = None
    if articulated:
        plan = articulated_placement_plan_from_adapter(
            adapted=adapter, asset=asset, asset_reference=rows["diagnostic_output.replacement.usdz"],
            qualified_static_digest=rows["diagnostic_output.static_qualification_receipt.v1.json"]["digest"],
        )
        from .task_evaluation_robot_placement_trajectory import placement_trajectory_from_native_plan
        derived = placement_trajectory_from_native_plan(plan)
        if task_trajectory is not None and task_trajectory != derived:
            raise TaskEvaluationDiagnosticNativeArenaCompilerError("diagnostic_native_compiler_task_trajectory_binding_mismatch")
        task_trajectory = derived
        contact = next((p for p in derived["phases"] if p["phase_id"] == "handle_contact"), None)
        if contact is None:
            raise TaskEvaluationDiagnosticNativeArenaCompilerError("diagnostic_native_compiler_task_contact_missing")
        start = contact["position_world_m"]
        target = derived["phases"][-1]["position_world_m"]
        construction = materialize_articulated_construction_inputs(
            task_spec=task_spec, definition=definition, asset=asset,
            asset_digest=rows["diagnostic_output.replacement.usdz"]["digest"],
            link_prim_paths=_read_json(
                _bound_path(rows, "diagnostic_output.static_qualification_receipt.v1.json"),
                blocker="diagnostic_native_compiler_static_qualification_invalid",
            )["link_prim_paths"],
        )
        task_spec = construction["task_spec"]
        kinematic_path = construction["articulated_kinematic_path_receipt"]
    else:
        start = _vector(task_source.get("start_center_xyz_m"), 3, blocker="diagnostic_native_compiler_task_direction_invalid")
        target = _vector(task_source.get("target_center_xyz_m"), 3, blocker="diagnostic_native_compiler_task_direction_invalid")
    direction = _normalize([target[0]-start[0], target[1]-start[1], 0.0], blocker="diagnostic_native_compiler_task_direction_invalid")
    workspace = _read_json(_bound_path(rows, "scene.registration.workspace_clearance"), blocker="diagnostic_native_compiler_workspace_invalid")
    placement = workspace.get("placement") or {}
    registered = (
        workspace.get("schema_version") == "registered_sage_franka_placement_packet.v1"
        and workspace.get("packet_digest") == canonical_digest(workspace, digest_field="packet_digest")
        and isinstance(placement, Mapping)
        and placement.get("candidate_may_self_authorize") is False
        and placement.get("physical_execution_authorized") is False
    )
    pending = (
        workspace.get("schema_version") == "website_workspace_clearance.v1"
        and workspace.get("status") == "execute_during_scene_configuration_run"
        and isinstance(workspace.get("scene_id"), str) and bool(workspace["scene_id"])
        and workspace.get("physical_execution_authorized") is not True
        and workspace.get("candidate_may_self_authorize") is not True
    )
    if not registered and not pending:
        raise TaskEvaluationDiagnosticNativeArenaCompilerError("diagnostic_native_compiler_workspace_invalid")
    scene_binding = {
        "schema_version": "diagnostic_robot_placement_scene_binding.v1",
        "source_configuration_run_id": authority["source_configuration_run_id"],
        "workspace_packet_digest": workspace["packet_digest"] if registered else canonical_digest(workspace),
        "collision_asset_digest": rows["diagnostic_output.collision.usda"]["digest"],
        "task_assets": [{"asset_id": task_spec["subject_asset_id"], "asset_usd_path": str(asset),
                         "asset_digest": rows["diagnostic_output.replacement.usdz"]["digest"],
                         "pose": definition["task_object_pose_world"]}],
    }
    task_binding = {
        "schema_version": "diagnostic_robot_placement_task_binding.v1",
        "robot_id": "franka_panda", "task_id": definition["identity"]["id"],
        "task_source_digest": canonical_digest(task_source),
        "droid_profile_digest": droid_profile_reference["digest"],
        "asset_scene_pose": definition["task_object_pose_world"],
        **({"target_contact_world_m": start, "opening_target_world_m": target,
            "trajectory_digest": task_trajectory["trajectory_digest"]} if articulated
           else {"start_center_xyz_m": start, "target_center_xyz_m": target}),
    }
    support = _read_json(_bound_path(rows, "scene.registration.support_plane"), blocker="diagnostic_native_compiler_support_invalid")
    return {"adapter": adapter, "task_spec": task_spec, "execution": adapter["native_episode_execution"],
            "start": start, "target": target, "direction": direction, "workspace": workspace,
            "placement": placement, "scene_binding": scene_binding, "task_binding": task_binding,
            "task_trajectory": task_trajectory, "placement_plan": plan, "support": support,
            "kinematic_path": kinematic_path if articulated else None,
            "task_state_binding": construction["task_state_binding"] if articulated else definition.get("task_state_binding"),
            "scene_collision_usd_path": str(_bound_path(rows, "diagnostic_output.collision.usda")),
            "target_position_world_m": start}


def _runtime_articulated_assets(*, rows, context, output_root):
    """Use the episode compiler's replayable physics overlay on diagnostic inputs."""
    contracts = {"appearance": "diagnostic_output.appearance.usdz",
                 "collision": "diagnostic_output.collision.usda",
                 "replacement": "diagnostic_output.replacement.usdz"}
    paths = {role: _bound_path(rows, contract) for role, contract in contracts.items()}
    definition = context["adapter"]["native_task_definition"]
    if context["task_spec"]["task_kind"] != "articulated_open_close":
        return paths, None, context
    from .task_evaluation_native_arena_episode_compiler import (
        _derive_runtime_prismatic_friction_overlay,
    )
    derived, adaptation = _derive_runtime_prismatic_friction_overlay(
        task_spec=context["task_spec"], task_definition=definition,
        replacement_path=paths["replacement"], output_root=output_root)
    if adaptation is None:
        return paths, None, context
    from .native_task_arena_runtime import (
        verify_grounded_articulation_overlay, verify_passive_joint_friction_overlay,
    )
    if adaptation["adaptation"] == "usd_authored_world_fixed_base":
        verify_grounded_articulation_overlay(paths["replacement"], derived, adaptation)
    else:
        verify_passive_joint_friction_overlay(derived, adaptation["passive_joint_friction"])
    # Keep sealed evidence intact. A common per-run root lets packet staging
    # hard-link unchanged appearance/collision instead of making another copy.
    evidence = output_root / "runtime-articulation-evidence"
    evidence.mkdir()
    paths["replacement"] = derived
    for role, source in list(paths.items()):
        destination = evidence / (role + source.suffix)
        try:
            os.link(source, destination)
        except OSError:
            shutil.copyfile(source, destination)
        if _identity(destination) != _identity(source):
            raise TaskEvaluationDiagnosticNativeArenaCompilerError(
                "diagnostic_native_compiler_runtime_asset_changed")
        paths[role] = destination
    spec = _runtime_subject_task_spec(definition["task_spec"])
    spec["success_criteria"] = context["adapter"]["native_success_criteria"]["criteria"]
    construction = materialize_articulated_construction_inputs(
        task_spec=spec, definition=definition, asset=paths["replacement"],
        asset_digest=_identity(paths["replacement"])[0],
        link_prim_paths=_read_json(
            _bound_path(rows, "diagnostic_output.static_qualification_receipt.v1.json"),
            blocker="diagnostic_native_compiler_static_qualification_invalid")["link_prim_paths"])
    return paths, adaptation, {**context, "task_spec": construction["task_spec"],
        "task_state_binding": construction["task_state_binding"],
        "kinematic_path": construction["articulated_kinematic_path_receipt"]}


def compile_diagnostic_native_arena_packet(
    *,
    diagnostic_controls_input: Mapping[str, Any],
    droid_profile_path: str | Path,
    droid_profile_reference: Mapping[str, Any],
    output_root: str | Path,
    robot_placement_receipt: Mapping[str, Any] | None = None,
    task_trajectory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize one task-aware development-only native construction packet."""

    authority = json.loads(json.dumps(dict(diagnostic_controls_input)))
    rows = _bound_rows(authority)
    root = Path(output_root).expanduser().resolve()
    if root.exists() or root.is_symlink():
        raise TaskEvaluationDiagnosticNativeArenaCompilerError(
            "diagnostic_native_compiler_output_exists"
        )
    root.mkdir(parents=True, mode=0o750)

    profile_unresolved = Path(droid_profile_path).expanduser()
    profile_path = profile_unresolved.resolve()
    if (
        profile_unresolved.is_symlink()
        or not profile_path.is_file()
        or _identity(profile_path)
        != (
            droid_profile_reference.get("digest"),
            droid_profile_reference.get("size_bytes"),
        )
    ):
        raise TaskEvaluationDiagnosticNativeArenaCompilerError(
            "diagnostic_native_compiler_droid_profile_binding_invalid"
        )
    profile = _read_json(
        profile_path, blocker="diagnostic_native_compiler_droid_profile_invalid"
    )
    sealed_request = profile.get("request_digest") == canonical_digest(profile, digest_field="request_digest")
    camera_template = (
        profile.get("status") == "immutable_intrinsics_and_wrist_mount_template"
        and profile.get("world_camera_poses_authoritative") is False
        and profile.get("world_camera_poses_recomputed_after_cpu_placement") is True
        and profile.get("native_observability_readback_required") is True
    )
    if profile.get("schema_version") != "native_task_arena_packet_request.v1" or not (sealed_request or camera_template):
        raise TaskEvaluationDiagnosticNativeArenaCompilerError(
            "diagnostic_native_compiler_droid_profile_invalid"
        )
    if camera_template:
        # The catalog camera factory binds immutable intrinsics and wrist mount,
        # not another scene's robot placement. Reuse canonical embodiment facts;
        # the accepted trajectory derives the arm reset below.
        from .task_evaluation_franka_robotiq_readiness_inputs import _RESET_JOINT_POSITIONS_RAD
        profile["robot_joint_reset_positions_rad"] = dict(_RESET_JOINT_POSITIONS_RAD)

    context = diagnostic_robot_placement_inputs(
        diagnostic_controls_input=authority, droid_profile_reference=droid_profile_reference,
        task_trajectory=task_trajectory,
    )
    runtime_assets, articulation_adaptation, context = _runtime_articulated_assets(
        rows=rows, context=context, output_root=root)
    adapter = context["adapter"]
    task_definition = adapter["native_task_definition"]
    task_spec = context["task_spec"]
    execution = context["execution"]
    start, target, direction = context["start"], context["target"], context["direction"]
    workspace, placement = context["workspace"], context["placement"]
    scene_binding, task_binding = context["scene_binding"], context["task_binding"]
    task_trajectory = context["task_trajectory"]
    accepted_placement_receipt: dict[str, Any] | None = None
    if robot_placement_receipt is not None:
        try:
            accepted_placement_receipt = validate_robot_placement_receipt(
                robot_placement_receipt,
                expected_scene_binding_digest=canonical_digest(scene_binding),
                expected_task_binding_digest=canonical_digest(task_binding),
            )
        except RobotPlacementAgentError as exc:
            raise TaskEvaluationDiagnosticNativeArenaCompilerError(
                f"diagnostic_native_compiler_robot_placement_invalid:{exc}"
            ) from exc
        base_pose = accepted_placement_receipt["accepted_pose"]
        base_derivation_method = "accepted_digest_bound_geometry_and_trajectory_candidate"
    else:
        # A prior diagnostic compiler copied only the source standoff, reflected
        # it along a new task direction, and never re-ran collision/support
        # validation.  Never revive that path.  A legacy pose is consumable only
        # when the exact registered packet itself passed every analytic gate.
        legacy_clear = _legacy_robot_placement_is_clear(workspace, placement)
        if not legacy_clear:
            raise TaskEvaluationDiagnosticNativeArenaCompilerError(
                "diagnostic_native_compiler_robot_placement_agent_receipt_required"
            )
        source_pose = _vector(
            placement.get("robot_pose_xyzyaw_collision_stage"),
            4,
            blocker="diagnostic_native_compiler_source_base_invalid",
        )
        yaw = source_pose[3]
        base_pose = {
            "position_world_m": source_pose[:3],
            "orientation_xyzw": [
                0.0,
                0.0,
                math.sin(yaw / 2.0),
                math.cos(yaw / 2.0),
            ],
        }
        base_derivation_method = "exact_registered_collision_clear_placement_pose"
    base_position = _vector(
        base_pose.get("position_world_m"),
        3,
        blocker="diagnostic_native_compiler_source_base_invalid",
    )
    task_aware_reset: dict[str, Any] | None = None
    robot_joint_reset_positions_rad = profile.get("robot_joint_reset_positions_rad")
    if accepted_placement_receipt is not None:
        if task_trajectory is None:
            raise TaskEvaluationDiagnosticNativeArenaCompilerError(
                "diagnostic_native_compiler_task_trajectory_required"
            )
        try:
            validated_trajectory = validate_robot_placement_trajectory(task_trajectory)
        except RobotPlacementTrajectoryError as exc:
            raise TaskEvaluationDiagnosticNativeArenaCompilerError(
                "diagnostic_native_compiler_task_trajectory_invalid"
            ) from exc
        if (
            accepted_placement_receipt.get("task_trajectory_digest")
            != validated_trajectory["trajectory_digest"]
        ):
            raise TaskEvaluationDiagnosticNativeArenaCompilerError(
                "diagnostic_native_compiler_task_trajectory_binding_mismatch"
            )
        robot_joint_reset_positions_rad, task_aware_reset = (
            _derive_task_aware_franka_reset(
                profile=profile,
                base_pose=base_pose,
                task_trajectory=validated_trajectory,
                robot_id=str(task_binding["robot_id"]),
            )
        )
        accepted_round = accepted_placement_receipt["rounds"][-1]
        accepted_reset = (
            (accepted_round.get("geometry_gate") or {})
            .get("orientation_slew_feasibility", {})
            .get("task_aware_reset")
        )
        accepted_joints = (
            accepted_reset.get("joint_positions_rad")
            if isinstance(accepted_reset, Mapping)
            else None
        )
        derived_joints = task_aware_reset["derived_arm_joint_positions_rad"]
        if (
            not isinstance(accepted_joints, list)
            or len(accepted_joints) != len(derived_joints)
            or any(
                not math.isclose(
                    float(accepted), float(derived), rel_tol=0.0, abs_tol=1.0e-9
                )
                for accepted, derived in zip(
                    accepted_joints, derived_joints, strict=True
                )
            )
        ):
            raise TaskEvaluationDiagnosticNativeArenaCompilerError(
                "diagnostic_native_compiler_task_aware_reset_binding_mismatch"
            )

    camera_rows = profile.get("cameras")
    by_role = {
        str(row.get("role") or ""): dict(row)
        for row in camera_rows or []
        if isinstance(row, Mapping)
    }
    wrist = by_role.get("wrist") or {}
    intrinsics = (by_role.get("external") or {}).get("intrinsics")
    overview_intrinsics = (by_role.get("overview") or {}).get("intrinsics")
    if (
        set(by_role) != {"external", "wrist", "overview"}
        or wrist.get("pose_frame") != "robot_body"
        or wrist.get("parent_prim_path")
        != "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/base_link"
        or wrist.get("policy_input") is not True
        or wrist.get("scoring_input") is not False
        or not isinstance(intrinsics, Mapping)
        or not isinstance(overview_intrinsics, Mapping)
        or intrinsics != wrist.get("intrinsics")
    ):
        raise TaskEvaluationDiagnosticNativeArenaCompilerError(
            "diagnostic_native_compiler_droid_profile_invalid"
        )
    midpoint = [(start[index] + target[index]) / 2.0 for index in range(3)]
    lateral = [-direction[1], direction[0], 0.0]
    cameras = [
        _world_camera(
            "external",
            position=[base_position[0], base_position[1], base_position[2] + 1.35],
            target=start,
            intrinsics=intrinsics,
        ),
        json.loads(json.dumps(wrist)),
        _world_camera(
            "overview",
            position=[
                midpoint[0] + 0.9 * lateral[0],
                midpoint[1] + 0.9 * lateral[1],
                max(base_position[2], start[2]) + 1.45,
            ],
            target=midpoint,
            intrinsics=overview_intrinsics,
        ),
    ]
    cameras[2]["policy_input"] = False

    asset_contracts = (
        ("appearance", "scene_appearance", "diagnostic_output.appearance.usdz"),
        ("collision", "scene_collision", "diagnostic_output.collision.usda"),
        ("replacement", "task_object", "diagnostic_output.replacement.usdz"),
    )
    packet_assets: list[dict[str, Any]] = []
    evidence_root = runtime_assets["appearance"].parent.resolve()
    object_pose = task_definition["task_object_pose_world"]
    for role, semantic_role, contract_path in asset_contracts:
        path = runtime_assets[role]
        if path.parent != evidence_root:
            raise TaskEvaluationDiagnosticNativeArenaCompilerError(
                "diagnostic_native_compiler_asset_root_mismatch"
            )
        row: dict[str, Any] = {
            "semantic_role": semantic_role,
            "filename": path.name,
            "source": {
                "root": "evidence",
                "relative_path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _identity(path)[0],
            },
            "pose_world": object_pose
            if role == "replacement"
            else {
                "position_world_m": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        }
        if role == "replacement":
            row.update(
                asset_id=task_spec["subject_asset_id"],
                object_type="ARTICULATION" if task_spec["task_kind"] == "articulated_open_close" else "RIGID",
                reset_state={"root_pose_world": object_pose,
                             "joint_positions": dict(task_definition.get("task_object_reset_joint_positions") or {})},
            )
            if articulation_adaptation is not None:
                row["articulation_adaptation"] = articulation_adaptation
        packet_assets.append(row)

    support = context["support"]
    packet_request: dict[str, Any] = {
        "schema_version": "native_task_arena_packet_request.v1",
        "scene_id": f"scene-{support['scene_id']}",
        "task_id": task_definition["identity"]["id"],
        "task_spec": task_spec,
        "task_joint_bindings": task_definition.get("task_joint_bindings") or [],
        "task_state_binding": context["task_state_binding"],
        "articulated_kinematic_path_receipt": context["kinematic_path"],
        "assets": packet_assets,
        "robot_base_pose_world": base_pose,
        "robot_joint_reset_positions_rad": robot_joint_reset_positions_rad,
        "cameras": cameras,
        "scenario": execution["scenario"],
        "physics_frequency_hz": execution["physics_frequency_hz"],
        "configured_task_template_adapter": adapter,
        "diagnostic_authority": {
            "schema_version": AUTHORITY_SCHEMA_VERSION,
            "receipt_digest": authority["receipt_digest"],
            "source_configuration_run_id": authority[
                "source_configuration_run_id"
            ],
            "claim_ceiling": CLAIM_CEILING,
            "qualification_eligible": False,
            "robot_placement_receipt_digest": (
                accepted_placement_receipt["receipt_digest"]
                if accepted_placement_receipt is not None
                else None
            ),
            "task_aware_robot_reset_derivation": task_aware_reset,
        },
        "request_digest": "",
    }
    packet_request["request_digest"] = canonical_digest(
        packet_request, digest_field="request_digest"
    )
    packet_root = root / "native-task-packet"
    packet_receipt = materialize_native_task_arena_packet(
        request=packet_request,
        evidence_root=evidence_root,
        output_dir=packet_root,
        link_sources_within=root,
    )
    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_development_only",
        "diagnostic_controls_input_receipt_digest": authority["receipt_digest"],
        "source_configuration_run_id": authority["source_configuration_run_id"],
        "source_provider_output_digest": authority["source_provider_output_digest"],
        "packet_request_path": str(
            packet_root / "native_task_arena_packet_request.v1.json"
        ),
        "packet_request_digest": packet_request["request_digest"],
        "packet_receipt_path": str(
            packet_root / "native_task_arena_packet_receipt.v1.json"
        ),
        "packet_receipt_digest": packet_receipt["receipt_digest"],
        "base_pose_world": base_pose,
        "base_derivation_method": base_derivation_method,
        "robot_placement_scene_binding": scene_binding,
        "robot_placement_task_binding": task_binding,
        "robot_placement_receipt_digest": (
            accepted_placement_receipt["receipt_digest"]
            if accepted_placement_receipt is not None
            else None
        ),
        "task_aware_robot_reset_derivation": task_aware_reset,
        "droid_profile_reference": dict(droid_profile_reference),
        "claim_ceiling": CLAIM_CEILING,
        "qualification_eligible": False,
        "configured_revision_publication_permitted": False,
        "native_construction_readback_required": True,
        "controls_required": [
            "zero_action_negative",
            "deterministic_scripted_positive",
        ],
        "provider_mutation_performed": False,
        "compiler_output_digest": "",
    }
    output["compiler_output_digest"] = canonical_digest(
        output, digest_field="compiler_output_digest"
    )
    output_path = root / "diagnostic_native_arena_compiler_output.v1.json"
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_path.chmod(0o440)
    return json.loads(json.dumps(output))


__all__ = [
    "SCHEMA_VERSION",
    "TaskEvaluationDiagnosticNativeArenaCompilerError",
    "compile_diagnostic_native_arena_packet",
    "diagnostic_robot_placement_inputs",
]
