from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_franka_robotiq_readiness_inputs import (
    ROBOT_IDENTITY,
    TaskEvaluationFrankaRobotiqReadinessInputsError,
    materialize_franka_robotiq_readiness_inputs,
)
from tests.test_task_evaluation_configured_scene_revision import revision


def _write_bound(root: Path, name: str, value: dict) -> tuple[Path, dict]:
    path = root / name
    payload = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(payload)
    return path, {
        "uri": f"s3://blueprint-production-inputs/configured-scene/{name}",
        "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _camera(role: str) -> dict:
    return {
        "role": role,
        "frame_from_camera_matrix": [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "intrinsics": {
            "cx": 159.5,
            "cy": 89.5,
            "fx": 172.8,
            "fy": 172.8,
            "height": 180,
            "width": 320,
        },
        "optical_convention": "opencv",
        "parent_prim_path": (
            "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/base_link"
            if role == "wrist"
            else "{ENV_REGEX_NS}"
        ),
        "policy_input": role != "overview",
        "pose_frame": "robot_body" if role == "wrist" else "world",
        "scoring_input": False,
    }


def _inputs(tmp_path: Path) -> tuple[dict, Path, Path, dict]:
    value = revision()
    mount_path, mount_ref = _write_bound(
        tmp_path,
        "mount.json",
        {
            "schema_version": "task_evaluation_robot_mount_interface_plan.v1",
            "scene_id": "839873",
            "status": "publish_during_scene_configuration_run",
        },
    )
    calibration_path, calibration_ref = _write_bound(
        tmp_path,
        "calibration.json",
        {
            "schema_version": "task_evaluation_scene_camera_calibration_plan.v1",
            "scene_id": "839873",
            "status": "solve_during_scene_configuration_run",
        },
    )
    value["registration"]["robot_mount_interface"] = mount_ref
    value["registration"]["camera_calibration"] = calibration_ref
    value["revision_digest"] = canonical_digest(value, digest_field="revision_digest")
    candidate = {
        "scene_identity": value["scene_identity"],
        "robot_mount_interface_digest": mount_ref["digest"],
        "learned_policy_outcomes_consulted": False,
        "native_construction_readback_completed": False,
        "pose_world": {
            "position_world_m": [0.42, -0.17, 0.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    }
    return value, mount_path, calibration_path, candidate


def test_materializes_scene_bound_candidate_without_claiming_qualification(
    tmp_path: Path,
) -> None:
    value, mount_path, calibration_path, candidate = _inputs(tmp_path)
    result = materialize_franka_robotiq_readiness_inputs(
        configured_revision=value,
        robot_mount_interface_path=mount_path,
        scene_camera_calibration_path=calibration_path,
        base_pose_candidate=candidate,
        cameras=[_camera("external"), _camera("wrist"), _camera("overview")],
        controller_identity={"id": "scripted-readiness", "version": "v1"},
        controller_kind="deterministic_scripted",
        output_root=tmp_path / "output",
    )

    assert result["robot_identity"] == ROBOT_IDENTITY
    assert (
        result["robot_mount_interface_digest"]
        == value["registration"]["robot_mount_interface"]["digest"]
    )
    assert (
        result["scene_camera_calibration_digest"]
        == value["registration"]["camera_calibration"]["digest"]
    )
    assert result["robot_base_qualified"] is False
    assert result["camera_configuration_qualified"] is False
    assert result["native_construction_readback_required"] is True
    assert set(result["files"]) == {
        "robot_configuration",
        "robot_kinematics",
        "robot_joint_bounds",
        "robot_base_registration",
        "controller_configuration",
        "sensor_configuration",
    }
    base = json.loads(Path(result["files"]["robot_base_registration"]["path"]).read_text())
    sensors = json.loads(Path(result["files"]["sensor_configuration"]["path"]).read_text())
    assert base["pose_status"] == "candidate_pending_native_construction_readback"
    assert base["robot_mount_interface_digest"] == result["robot_mount_interface_digest"]
    assert sensors["calibration_status"] == ("candidate_pending_native_construction_readback")
    assert sensors["scene_camera_calibration_digest"] == result["scene_camera_calibration_digest"]
    assert all(row["mode"] == "0440" for row in result["files"].values())


@pytest.mark.parametrize("mismatch", ["mount_bytes", "calibration_digest", "old_scene"])
def test_refuses_cross_scene_or_cross_digest_candidate_bindings(
    tmp_path: Path, mismatch: str
) -> None:
    value, mount_path, calibration_path, candidate = _inputs(tmp_path)
    if mismatch == "mount_bytes":
        mount_path.write_text('{"schema_version":"changed"}\n')
        expected = "franka_readiness_robot_mount_interface_binding_invalid"
    elif mismatch == "calibration_digest":
        value["registration"]["camera_calibration"]["digest"] = "sha256:" + "f" * 64
        value["revision_digest"] = canonical_digest(value, digest_field="revision_digest")
        expected = "franka_readiness_camera_calibration_binding_invalid"
    else:
        candidate = copy.deepcopy(candidate)
        candidate["scene_identity"] = {
            "id": "interiorgs-840313",
            "version": "can-v1",
        }
        expected = "franka_readiness_base_pose_candidate_binding_invalid"

    with pytest.raises(TaskEvaluationFrankaRobotiqReadinessInputsError, match=expected):
        materialize_franka_robotiq_readiness_inputs(
            configured_revision=value,
            robot_mount_interface_path=mount_path,
            scene_camera_calibration_path=calibration_path,
            base_pose_candidate=candidate,
            cameras=[_camera("external"), _camera("wrist"), _camera("overview")],
            controller_identity={"id": "scripted-readiness", "version": "v1"},
            controller_kind="deterministic_scripted",
            output_root=tmp_path / "output",
        )
