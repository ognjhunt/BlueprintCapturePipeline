from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_runtime_contract import DROID_FRANKA_RESET_JOINT_NAMES
from blueprint_pipeline.paired_target_native_camera_rig_candidate import (
    PairedTargetNativeCameraRigError,
    materialize_paired_target_native_camera_rig_candidate,
)


def _write(path: Path, value: dict, field: str) -> Path:
    value[field] = canonical_digest(value, digest_field=field)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _affordance(path: Path) -> Path:
    value = {
        "schema_version": "paired_target_interaction_affordance_candidate.v1",
        "scene_id": "generic_scene",
        "task_id": "generic_task",
        "asset_id": "generic_asset",
        "native_contact_executed": False,
        "robot_base_position_world_m": [1.0, 2.0, 0.0],
        "candidate": {
            "contact_point_registered_stage_m": [1.5, 2.4, 0.7],
            "pinch_span_within_stroke": True,
        },
        "receipt_digest": "",
    }
    return _write(path, value, "receipt_digest")


def _placement(path: Path) -> Path:
    value = {
        "schema_version": "registered_sage_franka_placement_packet.v1",
        "placement": {"robot_pose_xyzyaw_collision_stage": [1.0, 2.0, 0.0, 0.4]},
        "packet_digest": "",
    }
    return _write(path, value, "packet_digest")


def _camera(role: str) -> dict:
    wrist = role == "wrist"
    return {
        "role": role,
        "policy_input": role in {"external", "wrist"},
        "scoring_input": False,
        "pose_frame": "robot_body" if wrist else "world",
        "parent_prim_path": (
            "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/base_link"
            if wrist
            else "{ENV_REGEX_NS}"
        ),
        "optical_convention": "opencv",
        "frame_from_camera_matrix": [
            1.0, 0.0, 0.0, 0.011 if wrist else 0.0,
            0.0, 1.0, 0.0, -0.031 if wrist else 0.0,
            0.0, 0.0, 1.0, -0.074 if wrist else 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
        "intrinsics": {
            "fx": 172.88839142740494, "fy": 172.88839142740494,
            "cx": 159.5, "cy": 89.5, "width": 320, "height": 180,
        },
    }


def _profile(path: Path) -> Path:
    value = {
        "schema_version": "native_task_arena_packet_request.v1",
        "cameras": [_camera(role) for role in ("external", "wrist", "overview")],
        "robot_joint_reset_positions_rad": {
            name: float(index) / 100.0
            for index, name in enumerate(DROID_FRANKA_RESET_JOINT_NAMES)
        },
        "request_digest": "",
    }
    return _write(path, value, "request_digest")


def test_rig_derives_world_cameras_and_reuses_wrist_and_reset(tmp_path: Path) -> None:
    profile = _profile(tmp_path / "profile.json")
    source = json.loads(profile.read_text())
    result = materialize_paired_target_native_camera_rig_candidate(
        interaction_affordance_candidate_path=_affordance(tmp_path / "affordance.json"),
        franka_placement_packet_path=_placement(tmp_path / "placement.json"),
        droid_native_profile_request_path=profile,
        output_path=tmp_path / "result.json",
    )

    by_role = {row["role"]: row for row in result["cameras"]}
    assert by_role["wrist"] == source["cameras"][1]
    assert result["robot_joint_reset_positions_rad"] == source[
        "robot_joint_reset_positions_rad"
    ]
    assert by_role["overview"]["policy_input"] is False
    assert by_role["external"]["policy_input"] is True
    matrix = by_role["external"]["frame_from_camera_matrix"]
    position = [matrix[3], matrix[7], matrix[11]]
    forward = [matrix[2], matrix[6], matrix[10]]
    target = [1.5, 2.4, 0.7]
    expected = [target[index] - position[index] for index in range(3)]
    norm = math.sqrt(sum(value * value for value in expected))
    assert forward == pytest.approx([value / norm for value in expected])
    assert result["native_camera_readback_qualified"] is False
    assert result["blockers"] == [
        "native_camera_transform_and_intrinsics_readback_missing",
        "native_external_and_wrist_semantic_observability_missing",
    ]


def test_rig_rejects_base_mismatch_or_tampered_profile(tmp_path: Path) -> None:
    affordance = _affordance(tmp_path / "affordance.json")
    placement = _placement(tmp_path / "placement.json")
    profile = _profile(tmp_path / "profile.json")
    moved = json.loads(placement.read_text())
    moved["placement"]["robot_pose_xyzyaw_collision_stage"][0] += 1.0
    moved["packet_digest"] = canonical_digest(moved, digest_field="packet_digest")
    placement.write_text(json.dumps(moved), encoding="utf-8")
    with pytest.raises(
        PairedTargetNativeCameraRigError,
        match="affordance_placement_mismatch",
    ):
        materialize_paired_target_native_camera_rig_candidate(
            interaction_affordance_candidate_path=affordance,
            franka_placement_packet_path=placement,
            droid_native_profile_request_path=profile,
            output_path=tmp_path / "mismatch.json",
        )

    _placement(placement)
    profile.write_text(profile.read_text() + "\n", encoding="utf-8")
    # Whitespace does not alter the bound JSON digest. Alter a semantic field.
    tampered = json.loads(profile.read_text())
    tampered["cameras"][1]["parent_prim_path"] = "{ENV_REGEX_NS}/Robot/bad"
    profile.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(
        PairedTargetNativeCameraRigError,
        match="droid_profile_invalid",
    ):
        materialize_paired_target_native_camera_rig_candidate(
            interaction_affordance_candidate_path=affordance,
            franka_placement_packet_path=placement,
            droid_native_profile_request_path=profile,
            output_path=tmp_path / "tampered.json",
        )
