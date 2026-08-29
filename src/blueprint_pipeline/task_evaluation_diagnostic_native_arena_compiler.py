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
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_task_arena_packet import materialize_native_task_arena_packet
from .franka_kinematics import FRANKA_JOINT_LIMITS_RAD, forward_kinematics
from .native_task_construction_plan import (
    materialize_native_task_construction_phase_plan,
)
from .scene_placement.robot_profile import get_robot_profile
from .task_evaluation_robot_placement_orientation import (
    RobotPlacementOrientationError,
    task_aware_reset_joint_positions,
)
from .task_evaluation_robot_placement_agent import (
    RobotPlacementAgentError,
    validate_robot_placement_receipt,
)
from .task_evaluation_rigid_relocation_native_adapter import (
    adapt_rigid_relocation_task_template,
)


SCHEMA_VERSION = "task_evaluation_diagnostic_native_arena_compiler_output.v1"
AUTHORITY_SCHEMA_VERSION = (
    "task_evaluation_configured_scene_diagnostic_controls_input.v1"
)
CLAIM_CEILING = (
    "development_only_downstream_construction_and_controls_diagnostic"
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
    task_spec["source_subject_identity"] = source_subject_id
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


def _derive_task_aware_reset(
    *,
    packet_request: Mapping[str, Any],
    evidence_root: Any,
    derivation_root: Path,
    robot_id: str,
    base_pose: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Solve a reset pose that already faces the plan's first authored tool pose.

    Returns ``None`` whenever the inputs cannot decide it -- an unregistered
    embodiment, a profile that declares no arm joints or grasp calibration, or a
    plan with no phase to face.  A missing answer leaves the shipped constant in
    place and the native gates unchanged; it never invents one.
    """

    if not robot_id:
        return None
    try:
        profile = get_robot_profile(robot_id)
    except (KeyError, ValueError):
        return None
    if not profile.arm_joint_names:
        return None
    nominal = packet_request.get("robot_joint_reset_positions_rad")
    if not isinstance(nominal, Mapping) or not nominal:
        return None
    try:
        materialize_native_task_arena_packet(
            request=dict(packet_request),
            evidence_root=evidence_root,
            output_dir=derivation_root,
        )
        scene_plan = json.loads(
            (
                Path(derivation_root) / "native_task_arena_scene_plan.v1.json"
            ).read_text(encoding="utf-8")
        )
        phase_plan = materialize_native_task_construction_phase_plan(scene_plan)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    phases = phase_plan.get("phases") or []
    if not phases:
        return None
    target = phases[0].get("orientation_world_xyzw")
    if target is None:
        return None
    try:
        return task_aware_reset_joint_positions(
            nominal_joint_positions_rad=nominal,
            arm_joint_names=profile.arm_joint_names,
            joint_limits_rad=FRANKA_JOINT_LIMITS_RAD,
            base_orientation_xyzw=base_pose["orientation_xyzw"],
            target_orientation_world_xyzw=target,
            forward_kinematics=forward_kinematics,
            flange_to_grasp_orientation_xyzw=(
                profile.flange_to_grasp_orientation_xyzw
            ),
        )
    except (RobotPlacementOrientationError, KeyError, TypeError, ValueError):
        return None


def compile_diagnostic_native_arena_packet(
    *,
    diagnostic_controls_input: Mapping[str, Any],
    droid_profile_path: str | Path,
    droid_profile_reference: Mapping[str, Any],
    output_root: str | Path,
    robot_placement_receipt: Mapping[str, Any] | None = None,
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
    if (
        profile.get("schema_version") != "native_task_arena_packet_request.v1"
        or profile.get("request_digest")
        != canonical_digest(profile, digest_field="request_digest")
    ):
        raise TaskEvaluationDiagnosticNativeArenaCompilerError(
            "diagnostic_native_compiler_droid_profile_invalid"
        )

    adapter = adapt_rigid_relocation_task_template(
        materialized_references={},
        diagnostic_controls_input=authority,
    )
    task_definition = adapter["native_task_definition"]
    task_spec = _runtime_subject_task_spec(task_definition["task_spec"])
    task_spec["success_criteria"] = adapter["native_success_criteria"]["criteria"]
    execution = adapter["native_episode_execution"]
    task_source = adapter["source_documents"]["documents"]["definition"]
    start = _vector(
        task_source.get("start_center_xyz_m"),
        3,
        blocker="diagnostic_native_compiler_task_direction_invalid",
    )
    target = _vector(
        task_source.get("target_center_xyz_m"),
        3,
        blocker="diagnostic_native_compiler_task_direction_invalid",
    )
    direction = _normalize(
        [target[0] - start[0], target[1] - start[1], 0.0],
        blocker="diagnostic_native_compiler_task_direction_invalid",
    )
    workspace = _read_json(
        _bound_path(rows, "scene.registration.workspace_clearance"),
        blocker="diagnostic_native_compiler_workspace_invalid",
    )
    placement = workspace.get("placement")
    if (
        workspace.get("schema_version")
        != "registered_sage_franka_placement_packet.v1"
        or workspace.get("packet_digest")
        != canonical_digest(workspace, digest_field="packet_digest")
        or not isinstance(placement, Mapping)
        or placement.get("candidate_may_self_authorize") is not False
        or placement.get("physical_execution_authorized") is not False
    ):
        raise TaskEvaluationDiagnosticNativeArenaCompilerError(
            "diagnostic_native_compiler_workspace_invalid"
        )
    scene_binding = {
        "schema_version": "diagnostic_robot_placement_scene_binding.v1",
        "source_configuration_run_id": authority["source_configuration_run_id"],
        "workspace_packet_digest": workspace["packet_digest"],
        "collision_asset_digest": rows["diagnostic_output.collision.usda"]["digest"],
    }
    task_binding = {
        "schema_version": "diagnostic_robot_placement_task_binding.v1",
        "robot_id": "franka_panda",
        "task_id": task_definition["identity"]["id"],
        "start_center_xyz_m": start,
        "target_center_xyz_m": target,
        "task_source_digest": canonical_digest(task_source),
        "droid_profile_digest": droid_profile_reference["digest"],
    }
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
        base_derivation_method = "gpt_5_6_sol_high_bounded_geometry_and_visual_gate"
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

    camera_rows = profile.get("cameras")
    by_role = {
        str(row.get("role") or ""): dict(row)
        for row in camera_rows or []
        if isinstance(row, Mapping)
    }
    wrist = by_role.get("wrist") or {}
    intrinsics = (by_role.get("external") or {}).get("intrinsics")
    if (
        set(by_role) != {"external", "wrist", "overview"}
        or wrist.get("pose_frame") != "robot_body"
        or wrist.get("parent_prim_path")
        != "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/base_link"
        or wrist.get("policy_input") is not True
        or wrist.get("scoring_input") is not False
        or not isinstance(intrinsics, Mapping)
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
            intrinsics=intrinsics,
        ),
    ]
    cameras[2]["policy_input"] = False

    asset_contracts = (
        ("appearance", "scene_appearance", "diagnostic_output.appearance.usdz"),
        ("collision", "scene_collision", "diagnostic_output.collision.usda"),
        ("replacement", "task_object", "diagnostic_output.replacement.usdz"),
    )
    packet_assets: list[dict[str, Any]] = []
    evidence_root = Path(
        _bound_path(rows, "diagnostic_output.appearance.usdz")
    ).parent.resolve()
    object_pose = task_definition["task_object_pose_world"]
    for role, semantic_role, contract_path in asset_contracts:
        path = _bound_path(rows, contract_path)
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
                object_type="RIGID",
                reset_state={"root_pose_world": object_pose, "joint_positions": {}},
            )
        packet_assets.append(row)

    support = adapter["source_documents"]["documents"]["support_plane"]
    packet_request: dict[str, Any] = {
        "schema_version": "native_task_arena_packet_request.v1",
        "scene_id": f"scene-{support['scene_id']}",
        "task_id": task_definition["identity"]["id"],
        "task_spec": task_spec,
        "task_joint_bindings": task_definition.get("task_joint_bindings") or [],
        "task_state_binding": task_definition.get("task_state_binding"),
        "assets": packet_assets,
        "robot_base_pose_world": base_pose,
        "robot_joint_reset_positions_rad": profile.get(
            "robot_joint_reset_positions_rad"
        ),
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
        },
        "request_digest": "",
    }
    packet_request["request_digest"] = canonical_digest(
        packet_request, digest_field="request_digest"
    )
    # The reset pose otherwise ships as a profile constant, blind to what the
    # task authored.  Derive it from the plan's own first tool orientation so
    # the arm spawns already facing the work instead of paying a slew the phase
    # budget may not cover.  This needs the phase plan, which is materialized
    # from the scene plan, so the packet is built once into a retained
    # derivation directory and then rebuilt with the reset it implies.  Safe
    # because the phase plan does not read the reset: the same plan, digest
    # included, comes back either way.
    reset_derivation = _derive_task_aware_reset(
        packet_request=packet_request,
        evidence_root=evidence_root,
        derivation_root=root / "reset-derivation",
        robot_id=str(task_binding.get("robot_id") or ""),
        base_pose=base_pose,
    )
    if reset_derivation is not None:
        packet_request["robot_joint_reset_positions_rad"] = dict(
            reset_derivation["joint_reset_positions_rad"]
        )
        packet_request["request_digest"] = ""
        packet_request["request_digest"] = canonical_digest(
            packet_request, digest_field="request_digest"
        )
    packet_root = root / "native-task-packet"
    packet_receipt = materialize_native_task_arena_packet(
        request=packet_request,
        evidence_root=evidence_root,
        output_dir=packet_root,
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
        # Provenance for the spawn pose: which orientation the plan asked
        # for, what the shipped constant would have cost, and what the
        # derived reset costs instead.  None when the inputs could not
        # decide it and the constant was left in place.
        "task_aware_reset_derivation": (
            {
                key: reset_derivation[key]
                for key in (
                    "schema_version",
                    "arm_joint_names",
                    "target_orientation_world_xyzw",
                    "achieved_grasp_orientation_base_xyzw",
                    "nominal_slew_rad",
                    "residual_slew_rad",
                    "improvement_rad",
                    "searchable_joint_indices",
                )
            }
            if reset_derivation is not None
            else None
        ),
        "robot_placement_scene_binding": scene_binding,
        "robot_placement_task_binding": task_binding,
        "robot_placement_receipt_digest": (
            accepted_placement_receipt["receipt_digest"]
            if accepted_placement_receipt is not None
            else None
        ),
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
]
