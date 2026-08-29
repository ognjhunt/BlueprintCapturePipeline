from __future__ import annotations

import copy
import json

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_franka_robotiq_readiness_inputs import (
    materialize_franka_robotiq_readiness_inputs,
)
from blueprint_pipeline.task_evaluation_robot_placement_agent import (
    ROBOT_PLACEMENT_AGENT_MODEL,
    ROBOT_PLACEMENT_AGENT_REASONING_EFFORT,
)
from blueprint_pipeline.task_evaluation_robot_placement_inventory import (
    build_candidate_inventory_checkpoint,
)
from blueprint_pipeline.task_evaluation_robot_placement_readiness_candidate import (
    TaskEvaluationRobotPlacementReadinessCandidateError,
    materialize_robot_placement_readiness_candidate,
)
from tests.test_task_evaluation_franka_robotiq_readiness_inputs import (
    _camera,
    _inputs,
)


def _placement(value: dict) -> tuple[dict, dict, dict, dict]:
    scene = {
        "scene_identity": value["scene_identity"],
        "configured_scene_revision_digest": value["revision_digest"],
        "robot_mount_interface_digest": value["registration"][
            "robot_mount_interface"
        ]["digest"],
        "workspace_clearance_digest": value["registration"][
            "workspace_clearance"
        ]["digest"],
    }
    task = {
        "task_identity": value["task_template"]["identity"],
        "task_definition_digest": value["task_template"]["definition"]["digest"],
        "robot_id": "franka_panda",
    }
    pose = {
        "position_world_m": [2.9259961206833616, -6.132663812912105, 0.752958],
        "orientation_xyzw": [
            0.0,
            0.0,
            0.6087614290087209,
            -0.793353340291235,
        ],
    }
    support = "/Root/Counter@z=0.7530"
    trajectory_digest = "sha256:" + "3" * 64
    trajectory_gate = {
        "schema_version": (
            "task_evaluation_robot_placement_trajectory_position_ik_gate.v1"
        ),
        "status": "passed",
        "blockers": [],
        "waypoint_count": 2,
        "waypoints": [],
        "all_waypoints_position_ik_solved": True,
        "maximum_position_error_m": 0.00002,
        "minimum_manipulability": 0.153,
        "orientation_ik_solved": False,
        "native_orientation_ik_required": True,
        "native_collision_contact_required": True,
        "trajectory_position_ik_gate_digest": "",
    }
    trajectory_gate["trajectory_position_ik_gate_digest"] = canonical_digest(
        trajectory_gate,
        digest_field="trajectory_position_ik_gate_digest",
    )
    member = {
        "candidate_id": "geometry_02_08_0.753_0.650_57",
        "pose": pose,
        "support_surface_id": support,
        "geometry_gate_digest": "sha256:" + "5" * 64,
        "trajectory_position_ik_gate_digest": trajectory_gate[
            "trajectory_position_ik_gate_digest"
        ],
        "trajectory_minimum_manipulability": 0.153,
        "trajectory_position_ik_gate": trajectory_gate,
    }
    orientation_gate = {
        "schema_version": (
            "task_evaluation_robot_placement_orientation_feasibility.v1"
        ),
        "feasible": True,
        "blockers": [],
        "native_full_pose_ik_required": True,
        "native_collision_contact_and_reset_readback_required": True,
    }
    geometry_gate = {
        "schema_version": "task_evaluation_robot_placement_geometry_gate.v1",
        "candidate_id": member["candidate_id"],
        "declared_support_surface_id": support,
        "status": "passed",
        "blockers": [],
        "orientation_slew_feasibility": orientation_gate,
        "geometry_gate_digest": "",
    }
    geometry_gate["geometry_gate_digest"] = canonical_digest(
        geometry_gate, digest_field="geometry_gate_digest"
    )
    member["geometry_gate_digest"] = geometry_gate["geometry_gate_digest"]
    inventory = build_candidate_inventory_checkpoint(
        robot_id="franka_panda",
        target_position_world_m=[2.97, -6.76, 0.818],
        maximum_candidates=12,
        trajectory_digest=trajectory_digest,
        geometry_summary_digest="sha256:" + "6" * 64,
        candidates=[member],
    )
    proposal = {
        "candidate_id": member["candidate_id"],
        "pose": pose,
        "support_surface_id": support,
        "rationale": "Exact deterministic inventory member.",
        "addressed_blockers": [],
        "uncertainty": "Native construction is unresolved.",
    }
    receipt = {
        "schema_version": "task_evaluation_robot_placement_receipt.v1",
        "status": "accepted",
        "run_id": "scene839873-cpu-placement",
        "model": ROBOT_PLACEMENT_AGENT_MODEL,
        "reasoning_effort": ROBOT_PLACEMENT_AGENT_REASONING_EFFORT,
        "max_rounds": 1,
        "round_count": 1,
        "scene_binding_digest": canonical_digest(scene),
        "task_binding_digest": canonical_digest(task),
        "scene_context_digest": "sha256:" + "7" * 64,
        "task_context_digest": "sha256:" + "8" * 64,
        "task_trajectory_digest": trajectory_digest,
        "candidate_inventory_digest": inventory["candidate_inventory_digest"],
        "candidate_inventory_trajectory_digest": trajectory_digest,
        "overview_images": [],
        "prior_native_attempts": [],
        "prior_native_attempt_count": 0,
        "rounds": [
            {
                "proposal": proposal,
                "geometry_gate": geometry_gate,
                "visual_review": {
                    "status": "passed",
                    "robot_supported_by_declared_surface": True,
                    "robot_not_visibly_clipping_site_geometry": True,
                    "robot_faces_task_workspace": True,
                    "task_workspace_visually_reachable": True,
                    "camera_views_are_sufficient": True,
                    "reason": "The exact inventory member is visually coherent.",
                    "revision_guidance": [],
                },
                "native_attempt": None,
            }
        ],
        "accepted_pose": pose,
        "accepted_candidate_id": member["candidate_id"],
        "accepted_support_surface_id": support,
        "accepted_geometry_gate_digest": geometry_gate["geometry_gate_digest"],
        "native_agent_loop_enabled": False,
        "native_attempt_count": 0,
        "accepted_native_attempt_digest": None,
        "candidate_may_self_authorize": False,
        "physical_execution_authorized": False,
        "native_construction_required": True,
        "model_grades_controls": False,
        "claim_ceiling": "analytic_and_visual_robot_placement_candidate",
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    return tuple(
        json.loads(json.dumps(row))
        for row in (scene, task, inventory, receipt)
    )


def test_exact_inventory_selection_materializes_and_enters_readiness(tmp_path) -> None:
    value, mount_path, calibration_path, _ = _inputs(tmp_path)
    scene, task, inventory, receipt = _placement(value)
    candidate = materialize_robot_placement_readiness_candidate(
        configured_revision=value,
        scene_binding=scene,
        task_binding=task,
        placement_receipt=receipt,
        candidate_inventory=inventory,
        output_path=tmp_path / "robot-placement-readiness.json",
    )

    assert candidate["selected_candidate_id"] == receipt["accepted_candidate_id"]
    assert candidate["pose_world"] == receipt["accepted_pose"]
    assert candidate["deterministic_candidate_inventory_member_verified"] is True
    assert candidate["orientation_ik_qualified"] is False
    assert candidate[
        "native_orientation_collision_contact_and_camera_gates_required"
    ] is True

    readiness = materialize_franka_robotiq_readiness_inputs(
        configured_revision=value,
        robot_mount_interface_path=mount_path,
        scene_camera_calibration_path=calibration_path,
        base_pose_candidate=candidate,
        cameras=[_camera("external"), _camera("wrist"), _camera("overview")],
        controller_identity={"id": "scripted-readiness", "version": "v1"},
        controller_kind="deterministic_scripted",
        output_root=tmp_path / "readiness",
    )
    assert readiness["robot_base_qualified"] is False
    assert readiness["native_construction_readback_required"] is True


def test_unknown_candidate_is_rejected(tmp_path) -> None:
    value, _, _, _ = _inputs(tmp_path)
    scene, task, inventory, receipt = _placement(value)
    receipt["accepted_candidate_id"] = "unknown"
    receipt["rounds"][-1]["proposal"]["candidate_id"] = "unknown"
    receipt["rounds"][-1]["geometry_gate"]["candidate_id"] = "unknown"
    receipt["rounds"][-1]["geometry_gate"]["geometry_gate_digest"] = canonical_digest(
        receipt["rounds"][-1]["geometry_gate"],
        digest_field="geometry_gate_digest",
    )
    receipt["accepted_geometry_gate_digest"] = receipt["rounds"][-1][
        "geometry_gate"
    ]["geometry_gate_digest"]
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )

    with pytest.raises(
        TaskEvaluationRobotPlacementReadinessCandidateError,
        match="candidate_not_in_inventory",
    ):
        materialize_robot_placement_readiness_candidate(
            configured_revision=value,
            scene_binding=scene,
            task_binding=task,
            placement_receipt=receipt,
            candidate_inventory=inventory,
            output_path=tmp_path / "candidate.json",
        )


@pytest.mark.parametrize("mutation", ["position", "orientation", "support"])
def test_resealed_pose_or_support_mutation_is_rejected(tmp_path, mutation) -> None:
    value, _, _, _ = _inputs(tmp_path)
    scene, task, inventory, receipt = _placement(value)
    if mutation == "position":
        receipt["accepted_pose"]["position_world_m"][0] += 0.001
        receipt["rounds"][-1]["proposal"]["pose"] = copy.deepcopy(
            receipt["accepted_pose"]
        )
    elif mutation == "orientation":
        receipt["accepted_pose"]["orientation_xyzw"] = [0.0, 0.0, 1.0, 0.0]
        receipt["rounds"][-1]["proposal"]["pose"] = copy.deepcopy(
            receipt["accepted_pose"]
        )
    else:
        receipt["accepted_support_surface_id"] = "/Root/OtherSupport"
        receipt["rounds"][-1]["proposal"][
            "support_surface_id"
        ] = "/Root/OtherSupport"
        receipt["rounds"][-1]["geometry_gate"][
            "declared_support_surface_id"
        ] = "/Root/OtherSupport"
        receipt["rounds"][-1]["geometry_gate"][
            "geometry_gate_digest"
        ] = canonical_digest(
            receipt["rounds"][-1]["geometry_gate"],
            digest_field="geometry_gate_digest",
        )
        receipt["accepted_geometry_gate_digest"] = receipt["rounds"][-1][
            "geometry_gate"
        ]["geometry_gate_digest"]
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )

    with pytest.raises(
        TaskEvaluationRobotPlacementReadinessCandidateError,
        match="candidate_evidence_invalid",
    ):
        materialize_robot_placement_readiness_candidate(
            configured_revision=value,
            scene_binding=scene,
            task_binding=task,
            placement_receipt=receipt,
            candidate_inventory=inventory,
            output_path=tmp_path / "candidate.json",
        )
