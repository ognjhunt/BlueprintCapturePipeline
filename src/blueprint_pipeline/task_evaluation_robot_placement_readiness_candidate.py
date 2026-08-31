"""Bind one deterministic robot-placement selection into configured readiness.

The configured-scene progression must never accept a hand-transcribed base pose.
This adapter joins the exact configured revision, placement input bindings,
complete CPU candidate inventory, and accepted placement-agent receipt.  It
copies only the selected inventory member and keeps every native claim false.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_configured_scene_revision import (
    validate_configured_scene_revision,
)
from .task_evaluation_robot_placement_agent import (
    RobotPlacementAgentError,
    validate_robot_placement_receipt,
)
from .task_evaluation_robot_placement_inventory import (
    CANDIDATE_INVENTORY_SCHEMA_VERSION,
)


SCHEMA_VERSION = "task_evaluation_robot_placement_readiness_candidate.v1"


class TaskEvaluationRobotPlacementReadinessCandidateError(ValueError):
    """The placement evidence cannot authorize a configured readiness input."""


def _copy(value: Mapping[str, Any], *, blocker: str) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TaskEvaluationRobotPlacementReadinessCandidateError(blocker) from exc


def _digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
    )


def _inventory(value: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = _copy(
        value, blocker="robot_placement_readiness_inventory_invalid"
    )
    candidates = checkpoint.get("candidates")
    trajectory_digest = checkpoint.get("trajectory_digest")
    if (
        checkpoint.get("schema_version") != CANDIDATE_INVENTORY_SCHEMA_VERSION
        or checkpoint.get("status") != "complete"
        or checkpoint.get("robot_id") != "franka_panda"
        or not isinstance(candidates, list)
        or not candidates
        or any(not isinstance(row, Mapping) for row in candidates)
        or (
            trajectory_digest is not None
            and not _digest(trajectory_digest)
        )
        or checkpoint.get("candidate_inventory_digest")
        != canonical_digest(
            {
                "trajectory_digest": trajectory_digest,
                "candidates": candidates,
            }
        )
        or checkpoint.get("checkpoint_digest")
        != canonical_digest(checkpoint, digest_field="checkpoint_digest")
    ):
        raise TaskEvaluationRobotPlacementReadinessCandidateError(
            "robot_placement_readiness_inventory_invalid"
        )
    return checkpoint


def materialize_robot_placement_readiness_candidate(
    *,
    configured_revision: Mapping[str, Any],
    scene_binding: Mapping[str, Any],
    task_binding: Mapping[str, Any],
    placement_receipt: Mapping[str, Any],
    candidate_inventory: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Write one exact-inventory readiness candidate without native claims."""

    revision = validate_configured_scene_revision(configured_revision)
    scene = _copy(
        scene_binding, blocker="robot_placement_readiness_scene_binding_invalid"
    )
    task = _copy(
        task_binding, blocker="robot_placement_readiness_task_binding_invalid"
    )
    if (
        scene.get("scene_identity") != revision["scene_identity"]
        or scene.get("configured_scene_revision_digest")
        != revision["revision_digest"]
        or scene.get("robot_mount_interface_digest")
        != revision["registration"]["robot_mount_interface"]["digest"]
        or scene.get("workspace_clearance_digest")
        != revision["registration"]["workspace_clearance"]["digest"]
    ):
        raise TaskEvaluationRobotPlacementReadinessCandidateError(
            "robot_placement_readiness_scene_binding_invalid"
        )
    if (
        task.get("task_identity") != revision["task_template"]["identity"]
        or task.get("task_definition_digest")
        != revision["task_template"]["definition"]["digest"]
        or task.get("robot_id") != "franka_panda"
    ):
        raise TaskEvaluationRobotPlacementReadinessCandidateError(
            "robot_placement_readiness_task_binding_invalid"
        )
    try:
        receipt = validate_robot_placement_receipt(
            placement_receipt,
            expected_scene_binding_digest=canonical_digest(scene),
            expected_task_binding_digest=canonical_digest(task),
        )
    except RobotPlacementAgentError as exc:
        raise TaskEvaluationRobotPlacementReadinessCandidateError(
            "robot_placement_readiness_receipt_invalid"
        ) from exc
    inventory = _inventory(candidate_inventory)
    inventory_digest = inventory["candidate_inventory_digest"]
    trajectory_digest = inventory.get("trajectory_digest")
    if (
        receipt.get("candidate_inventory_digest") != inventory_digest
        or receipt.get("candidate_inventory_trajectory_digest")
        != trajectory_digest
        or receipt.get("task_trajectory_digest") != trajectory_digest
    ):
        raise TaskEvaluationRobotPlacementReadinessCandidateError(
            "robot_placement_readiness_inventory_binding_mismatch"
        )
    candidate_id = str(receipt["accepted_candidate_id"])
    members = [
        dict(row)
        for row in inventory["candidates"]
        if isinstance(row, Mapping) and row.get("candidate_id") == candidate_id
    ]
    if len(members) != 1:
        raise TaskEvaluationRobotPlacementReadinessCandidateError(
            "robot_placement_readiness_candidate_not_in_inventory"
        )
    member = members[0]
    selected = {
        "candidate_id": candidate_id,
        "pose": receipt["accepted_pose"],
        "support_surface_id": receipt["accepted_support_surface_id"],
    }
    expected = {
        "candidate_id": member.get("candidate_id"),
        "pose": member.get("pose"),
        "support_surface_id": member.get("support_surface_id"),
    }
    trajectory_gate = member.get("trajectory_position_ik_gate")
    accepted_round = receipt["rounds"][-1]
    geometry_gate = accepted_round.get("geometry_gate")
    orientation_gate = (
        geometry_gate.get("orientation_slew_feasibility")
        if isinstance(geometry_gate, Mapping)
        else None
    )
    task_aware_reset = (
        orientation_gate.get("task_aware_reset")
        if isinstance(orientation_gate, Mapping)
        else None
    )
    reset_joints = (
        task_aware_reset.get("joint_positions_rad")
        if isinstance(task_aware_reset, Mapping)
        else None
    )
    inventory_geometry_gate = dict(geometry_gate or {})
    inventory_geometry_gate.pop("orientation_slew_feasibility", None)
    inventory_geometry_gate_digest = canonical_digest(
        inventory_geometry_gate,
        digest_field="geometry_gate_digest",
    )
    if (
        canonical_digest(selected) != canonical_digest(expected)
        or not isinstance(trajectory_gate, Mapping)
        or trajectory_gate.get("status") != "passed"
        or trajectory_gate.get("blockers") not in ([], ())
        or trajectory_gate.get("all_waypoints_position_ik_solved") is not True
        or trajectory_gate.get("orientation_ik_solved") is not False
        or trajectory_gate.get("native_collision_contact_required") is not True
        or trajectory_gate.get("native_orientation_ik_required") is not True
        or trajectory_gate.get("trajectory_position_ik_gate_digest")
        != canonical_digest(
            trajectory_gate,
            digest_field="trajectory_position_ik_gate_digest",
        )
        or member.get("trajectory_position_ik_gate_digest")
        != trajectory_gate.get("trajectory_position_ik_gate_digest")
        or member.get("trajectory_minimum_manipulability")
        != trajectory_gate.get("minimum_manipulability")
        or not isinstance(geometry_gate, Mapping)
        or geometry_gate.get("status") != "passed"
        or geometry_gate.get("blockers") not in ([], ())
        or geometry_gate.get("geometry_gate_digest")
        != receipt.get("accepted_geometry_gate_digest")
        or member.get("geometry_gate_digest")
        != inventory_geometry_gate_digest
        or not isinstance(orientation_gate, Mapping)
        or orientation_gate.get("feasible") is not True
        or orientation_gate.get("blockers") not in ([], ())
        or not isinstance(task_aware_reset, Mapping)
        or not isinstance(reset_joints, list)
        or len(reset_joints) != 7
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in reset_joints
        )
    ):
        raise TaskEvaluationRobotPlacementReadinessCandidateError(
            "robot_placement_readiness_candidate_evidence_invalid"
        )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "candidate_pending_native_construction_readback",
        "scene_identity": revision["scene_identity"],
        "configured_scene_revision_digest": revision["revision_digest"],
        "robot_mount_interface_digest": revision["registration"][
            "robot_mount_interface"
        ]["digest"],
        "task_definition_digest": revision["task_template"]["definition"][
            "digest"
        ],
        "workspace_clearance_digest": revision["registration"][
            "workspace_clearance"
        ]["digest"],
        "derivation_method": "deterministic_trajectory_inventory_selection",
        "robot_placement_receipt_digest": receipt["receipt_digest"],
        "candidate_inventory_digest": inventory_digest,
        "candidate_inventory_checkpoint_digest": inventory["checkpoint_digest"],
        "selected_candidate_id": candidate_id,
        "selected_support_surface_id": receipt["accepted_support_surface_id"],
        "selected_geometry_gate_digest": receipt[
            "accepted_geometry_gate_digest"
        ],
        "task_trajectory_digest": trajectory_digest,
        "trajectory_position_ik_gate_digest": trajectory_gate[
            "trajectory_position_ik_gate_digest"
        ],
        "trajectory_minimum_manipulability": trajectory_gate[
            "minimum_manipulability"
        ],
        "orientation_slew_feasibility_digest": canonical_digest(
            orientation_gate
        ),
        # The placement gate derived the reset whose grasp orientation made the
        # accepted trajectory feasible.  Carry the exact joints forward: using
        # the generic home pose here makes the native arm pay a different
        # orientation path and can sweep through the task object before the
        # authored precontact phase begins.
        "task_aware_reset": json.loads(json.dumps(dict(task_aware_reset))),
        "pose_world": receipt["accepted_pose"],
        "task_trajectory_considered": trajectory_digest is not None,
        "deterministic_candidate_inventory_member_verified": True,
        "position_ik_qualified_on_cpu": True,
        "orientation_ik_qualified": False,
        "robot_base_qualified": False,
        "reachability_qualified": False,
        "collision_clearance_qualified": False,
        "learned_policy_outcomes_consulted": False,
        "native_construction_readback_completed": False,
        "native_orientation_collision_contact_and_camera_gates_required": True,
        "base_pose_candidate_digest": "",
    }
    result["base_pose_candidate_digest"] = canonical_digest(
        result, digest_field="base_pose_candidate_digest"
    )
    destination = Path(output_path).expanduser()
    if destination.exists() or destination.is_symlink():
        raise TaskEvaluationRobotPlacementReadinessCandidateError(
            "robot_placement_readiness_destination_exists"
        )
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    destination.chmod(0o440)
    return json.loads(json.dumps(result))


def _mapping(path: str | Path, *, blocker: str) -> dict[str, Any]:
    source = Path(path).expanduser()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationRobotPlacementReadinessCandidateError(blocker) from exc
    if source.is_symlink() or not source.is_file() or not isinstance(value, Mapping):
        raise TaskEvaluationRobotPlacementReadinessCandidateError(blocker)
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configured-revision", required=True)
    parser.add_argument("--scene-binding", required=True)
    parser.add_argument("--task-binding", required=True)
    parser.add_argument("--placement-receipt", required=True)
    parser.add_argument("--candidate-inventory", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        result = materialize_robot_placement_readiness_candidate(
            configured_revision=_mapping(
                args.configured_revision,
                blocker="robot_placement_readiness_revision_invalid",
            ),
            scene_binding=_mapping(
                args.scene_binding,
                blocker="robot_placement_readiness_scene_binding_invalid",
            ),
            task_binding=_mapping(
                args.task_binding,
                blocker="robot_placement_readiness_task_binding_invalid",
            ),
            placement_receipt=_mapping(
                args.placement_receipt,
                blocker="robot_placement_readiness_receipt_invalid",
            ),
            candidate_inventory=_mapping(
                args.candidate_inventory,
                blocker="robot_placement_readiness_inventory_invalid",
            ),
            output_path=args.output,
        )
    except (TaskEvaluationRobotPlacementReadinessCandidateError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "blockers": [str(exc)]}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SCHEMA_VERSION",
    "TaskEvaluationRobotPlacementReadinessCandidateError",
    "main",
    "materialize_robot_placement_readiness_candidate",
]
