from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.adp009d_control_episode import (
    BLOCKER_POSITIVE_FAILED,
    SCRIPTED_POSITIVE,
    ZERO_ACTION_NEGATIVE,
    ControlEpisodeError,
    materialize_control_plan,
    run_required_controls,
)
from blueprint_pipeline.adp009d_droid_observation import (
    DROID_EXTERIOR_VIEW_1,
    DROID_WRIST_VIEW,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


START = [3.4681748, -3.3100837, 0.5264650138348479]
TARGET = [3.750152333333333, -3.4074919, 0.5264650138348479]


def _instance() -> dict:
    value = {
        "schema_version": "adp009d_scenario_instance.v1",
        "program_id": "arm-decision-proof-v1",
        "suite_digest": "sha256:" + "1" * 64,
        "harness_digest": "sha256:" + "2" * 64,
        "cell_id": "canonical_anchor__seed_2026080600",
        "template_id": "canonical_anchor",
        "family": "canonical",
        "partition": "qualification",
        "scored": True,
        "seed": 2026080600,
        "cell_seed_digest": "sha256:" + "3" * 64,
        "cousin_id": "approved_can",
        "cousin_digest": None,
        "cousin_static_validation_receipt_digest": None,
        "resolved_parameters": {
            "object_start_x_m": START[0],
            "object_start_y_m": START[1],
            "object_start_z_m": START[2],
            "target_x_m": TARGET[0],
            "target_y_m": TARGET[1],
            "target_z_m": TARGET[2],
            "object_height_m": 0.1694279937744141,
        },
        "factor_records": [],
        "required_controls": [
            "deterministic_scripted_positive",
            "zero_action_negative",
        ],
        "policy_neutral": True,
        "caller_asserted_success": False,
        "instance_digest": "",
    }
    value["instance_digest"] = canonical_digest(
        value, digest_field="instance_digest"
    )
    return value


def _calibration(width: int, height: int) -> dict:
    return {
        "camera_model": "pinhole",
        "intrinsic_matrix": [
            [12.0, 0.0, width / 2],
            [0.0, 12.0, height / 2],
            [0.0, 0.0, 1.0],
        ],
        "world_from_camera": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "resolution": [width, height],
        "near_m": 0.01,
        "far_m": 10.0,
    }


class _ControlEnvironment:
    def __init__(self, *, positive_moves_object: bool = True) -> None:
        self.positive_moves_object = positive_moves_object
        self.reset_count = 0
        self.step_index = 0
        self.joints = [0.1 * index for index in range(7)]
        self.can = list(START)
        self.gripper_width = 0.085
        self.gripped = False
        self.pending_target = None
        self.pending_gripper = 1.0
        self.actions: list[list[float]] = []

    def reset(self) -> None:
        self.reset_count += 1
        self.step_index = 0
        self.joints = [0.1 * index for index in range(7)]
        self.can = list(START)
        self.gripper_width = 0.085
        self.gripped = False
        self.pending_target = None
        self.pending_gripper = 1.0

    def read_policy_inputs(self):
        external = np.full((12, 16, 3), 40 + self.step_index % 100, dtype=np.uint8)
        wrist = np.full((12, 16, 3), 120 + self.step_index % 100, dtype=np.uint8)
        return {
            DROID_EXTERIOR_VIEW_1: external,
            DROID_WRIST_VIEW: wrist,
            "joint_position": list(self.joints),
            "gripper_position": 1.0 - self.gripper_width / 0.085,
        }

    def read_evaluation_camera_inputs(self):
        inputs = self.read_policy_inputs()
        return {
            "external": inputs[DROID_EXTERIOR_VIEW_1],
            "wrist": inputs[DROID_WRIST_VIEW],
            "overview": np.full(
                (12, 16, 3), 200 + self.step_index % 55, dtype=np.uint8
            ),
        }

    def read_control_observation_metadata(self):
        calibration = _calibration(16, 12)
        camera_ids = ("external", "wrist", "overview")
        return {
            "timestamp_ns": self.step_index * 1_000_000,
            "simulation_time_s": self.step_index / 15.0,
            "calibrations": {camera_id: calibration for camera_id in camera_ids},
            "source_devices": {camera_id: "cpu" for camera_id in camera_ids},
            "synchronizations": {
                camera_id: {"host_bytes_ready": True, "method": "test_step"}
                for camera_id in camera_ids
            },
        }

    def read_arm_joint_positions(self):
        return list(self.joints)

    def read_object_sample(self):
        return {
            "can_pose_world": [*self.can, 0.0, 0.0, 0.0, 1.0],
            "gripper_width_m": self.gripper_width,
            "grasp_frame_position_world_m": list(self.can),
        }

    def hold_action(self, *, gripper_command: float):
        self.pending_target = None
        self.pending_gripper = float(gripper_command)
        return [*self.joints, float(gripper_command)]

    def scripted_action_for_pose(
        self,
        *,
        target_position_world_m,
        target_quaternion_world_xyzw,
        gripper_command,
        max_joint_delta_rad,
    ):
        assert target_quaternion_world_xyzw == [1.0, 0.0, 0.0, 0.0]
        assert max_joint_delta_rad == 0.03
        self.pending_target = [float(value) for value in target_position_world_m]
        self.pending_gripper = float(gripper_command)
        target_joints = [
            value + min(max_joint_delta_rad, 0.001) for value in self.joints
        ]
        return [*target_joints, float(gripper_command)]

    def step(self, isaac_action):
        action = [float(value) for value in isaac_action]
        self.actions.append(action)
        self.joints = action[:7]
        self.step_index += 1
        self.gripper_width = 0.085 if self.pending_gripper == 1.0 else 0.04
        if not self.positive_moves_object or self.pending_target is None:
            return
        target = self.pending_target
        at_object_xy = np.linalg.norm(np.asarray(target[:2]) - np.asarray(self.can[:2])) < 0.05
        if self.pending_gripper == 0.0 and at_object_xy:
            self.gripped = True
        if self.gripped and self.pending_gripper == 0.0:
            self.can[0] = target[0]
            self.can[1] = target[1]
            self.can[2] = max(START[2], target[2] - 0.27)
        if self.pending_gripper == 1.0:
            self.gripped = False
            if np.linalg.norm(np.asarray(target[:2]) - np.asarray(TARGET[:2])) < 0.05:
                self.can = list(TARGET)


def test_control_plan_is_deterministic_and_bound_to_the_scenario_instance() -> None:
    instance = _instance()

    first = materialize_control_plan(instance)
    second = materialize_control_plan(instance)

    assert first == second
    assert first["instance_digest"] == instance["instance_digest"]
    assert first["resolved_destination_position_world_m"] == TARGET
    assert [phase["phase_id"] for phase in first["scripted_positive_phases"]] == [
        "pregrasp",
        "descend",
        "grasp",
        "lift",
        "transport",
        "place",
        "release",
        "retreat",
        "settle",
    ]
    assert first["candidate_policy_queried"] is False


def test_control_plan_rejects_a_forged_instance_digest() -> None:
    instance = _instance()
    instance["resolved_parameters"]["target_x_m"] += 0.01

    with pytest.raises(ControlEpisodeError, match="control_plan_instance_digest_mismatch"):
        materialize_control_plan(instance)


def test_controls_reject_a_changed_shipped_plan(tmp_path: Path) -> None:
    instance = _instance()
    plan = materialize_control_plan(instance)
    plan["zero_action"]["steps"] += 1

    with pytest.raises(ControlEpisodeError) as excinfo:
        run_required_controls(
            environment=_ControlEnvironment(),
            scenario_instance=instance,
            expected_control_plan=plan,
            gripper_open_command=1.0,
            gripper_closed_command=0.0,
            output_dir=tmp_path,
        )

    assert "control_plan_bundle_binding_mismatch" in excinfo.value.errors


def test_required_controls_admit_cell_only_after_negative_and_positive_pass(
    tmp_path: Path,
) -> None:
    environment = _ControlEnvironment()

    pair = run_required_controls(
        environment=environment,
        scenario_instance=_instance(),
        expected_control_plan=materialize_control_plan(_instance()),
        gripper_open_command=1.0,
        gripper_closed_command=0.0,
        output_dir=tmp_path,
    )

    assert pair["execution_order"] == [ZERO_ACTION_NEGATIVE, SCRIPTED_POSITIVE]
    assert pair["cell_admitted_for_policy_execution"] is True
    assert pair["policy_execution_blockers"] == []
    assert pair["positive_failure_is_policy_failure"] is False
    assert all(row["control_passed"] for row in pair["controls"])
    negative = json.loads(
        (tmp_path / f"adp009d_control_episode.{ZERO_ACTION_NEGATIVE}.json").read_text()
    )
    positive = json.loads(
        (tmp_path / f"adp009d_control_episode.{SCRIPTED_POSITIVE}.json").read_text()
    )
    assert negative["observed_outcome"] == "never_moved"
    assert positive["observed_outcome"] == "placed"
    assert negative["candidate_policy_queried"] is False
    assert positive["candidate_policy_queried"] is False
    assert negative["action_trace"][0]["isaac_action"][:7] == negative[
        "action_trace"
    ][0]["observed_joint_position_before_rad"]
    for receipt in (negative, positive):
        assert receipt["visual_evidence"]["status"] == "complete"
        assert set(receipt["visual_evidence"]["videos"]) == {
            "external",
            "wrist",
            "overview",
        }
        assert receipt["visual_evidence"]["review_only_camera_ids"] == ["overview"]
        assert receipt["state_trace_digest"].startswith("sha256:")
        assert receipt["action_trace_digest"].startswith("sha256:")


def test_failed_scripted_positive_blocks_cell_without_becoming_policy_failure(
    tmp_path: Path,
) -> None:
    pair = run_required_controls(
        environment=_ControlEnvironment(positive_moves_object=False),
        scenario_instance=_instance(),
        gripper_open_command=1.0,
        gripper_closed_command=0.0,
        output_dir=tmp_path,
    )

    assert pair["cell_admitted_for_policy_execution"] is False
    assert pair["positive_failure_is_policy_failure"] is False
    assert pair["candidate_policy_queried"] is False
    assert pair["policy_execution_blockers"] == [
        f"{BLOCKER_POSITIVE_FAILED}:never_moved"
    ]
