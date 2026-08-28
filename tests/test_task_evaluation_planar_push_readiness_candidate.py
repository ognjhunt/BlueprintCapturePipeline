from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_planar_push_readiness_candidate import (
    TaskEvaluationPlanarPushReadinessCandidateError,
    materialize_planar_push_readiness_candidate,
)
from tests.test_task_evaluation_configured_scene_revision import revision


def _write(root: Path, name: str, value: dict) -> tuple[Path, dict]:
    path = root / name
    payload = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(payload)
    return path, {
        "uri": f"s3://blueprint-production-inputs/{name}",
        "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _camera(role: str) -> dict:
    return {
        "role": role,
        "policy_input": role != "overview",
        "scoring_input": False,
        "pose_frame": "robot_body" if role == "wrist" else "world",
        "parent_prim_path": (
            "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/base_link"
            if role == "wrist"
            else "{ENV_REGEX_NS}"
        ),
        "optical_convention": "opencv",
        "frame_from_camera_matrix": [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
        "intrinsics": {
            "fx": 172.8,
            "fy": 172.8,
            "cx": 159.5,
            "cy": 89.5,
            "width": 320,
            "height": 180,
        },
    }


def _case(tmp_path: Path) -> tuple[dict, Path, Path, Path, dict]:
    configured = revision()
    task = {
        "schema_version": "task_evaluation_rigid_relocation_template.v1",
        "status": "preregistered_candidate_pending_configured_scene_revision",
        "task_identity": configured["task_template"]["identity"],
        "object_identity": configured["replacement"]["identity"],
        "strategy": "planar_push",
        "start_center_xyz_m": [2.9742285, -6.7605156, 0.818319],
        "target_center_xyz_m": [3.0942285, -6.7605156, 0.818319],
    }
    workspace = {
        "schema_version": "registered_sage_franka_placement_packet.v1",
        "status": "blocked",
        "placement": {
            "robot_pose_xyzyaw_collision_stage": [
                3.1742285,
                -6.7605156,
                0.485319,
                3.14159265359,
            ],
            "candidate_may_self_authorize": False,
            "physical_execution_authorized": False,
        },
        "packet_digest": "",
    }
    workspace["packet_digest"] = canonical_digest(
        workspace, digest_field="packet_digest"
    )
    profile = {
        "schema_version": "native_task_arena_packet_request.v1",
        "cameras": [_camera("external"), _camera("wrist"), _camera("overview")],
        "request_digest": "",
    }
    profile["request_digest"] = canonical_digest(
        profile, digest_field="request_digest"
    )
    task_path, task_ref = _write(tmp_path, "task.json", task)
    workspace_path, workspace_ref = _write(tmp_path, "workspace.json", workspace)
    profile_path, profile_ref = _write(tmp_path, "profile.json", profile)
    configured["task_template"]["definition"] = task_ref
    configured["registration"]["workspace_clearance"] = workspace_ref
    configured["revision_digest"] = canonical_digest(
        configured, digest_field="revision_digest"
    )
    return configured, task_path, workspace_path, profile_path, profile_ref


def test_reflects_reach_candidate_behind_frozen_push_and_copies_wrist(
    tmp_path: Path,
) -> None:
    configured, task, workspace, profile, profile_ref = _case(tmp_path)
    result = materialize_planar_push_readiness_candidate(
        configured_revision=configured,
        task_definition_path=task,
        workspace_clearance_path=workspace,
        droid_profile_path=profile,
        droid_profile_reference=profile_ref,
        output_path=tmp_path / "candidate.json",
    )

    base = result["base_pose_candidate"]
    assert base["pose_world"]["position_world_m"] == pytest.approx(
        [2.7742285, -6.7605156, 0.485319]
    )
    assert base["pose_world"]["orientation_xyzw"] == pytest.approx(
        [0.0, 0.0, 0.0, 1.0]
    )
    assert base["robot_base_qualified"] is False
    assert base["task_direction_considered"] is True
    assert [row["role"] for row in result["cameras"]] == [
        "external",
        "wrist",
        "overview",
    ]
    assert result["cameras"][1]["pose_frame"] == "robot_body"
    assert result["camera_configuration_qualified"] is False
    assert result["receipt_digest"] == canonical_digest(
        result, digest_field="receipt_digest"
    )


def test_refuses_symlinked_workspace_even_when_target_bytes_match(
    tmp_path: Path,
) -> None:
    configured, task, workspace, profile, profile_ref = _case(tmp_path)
    link = tmp_path / "workspace-link.json"
    link.symlink_to(workspace)

    with pytest.raises(
        TaskEvaluationPlanarPushReadinessCandidateError,
        match="planar_push_readiness_workspace_binding_invalid",
    ):
        materialize_planar_push_readiness_candidate(
            configured_revision=configured,
            task_definition_path=task,
            workspace_clearance_path=link,
            droid_profile_path=profile,
            droid_profile_reference=profile_ref,
            output_path=tmp_path / "candidate.json",
        )
