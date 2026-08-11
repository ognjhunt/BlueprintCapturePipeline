from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.adp009d_control_episode import (
    BLOCKER_POSITIVE_FAILED,
    BLOCKER_PHASE_NOT_REACHED,
    SCRIPTED_POSITIVE,
    ZERO_ACTION_NEGATIVE,
    ControlEpisodeError,
    materialize_control_plan,
    run_control_episode,
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
            "object_radius_m": 0.031094726014345042,
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
    def __init__(
        self,
        *,
        positive_moves_object: bool = True,
        grasp_frame_converges: bool = True,
        grasp_frame_step_fraction: float = 1.0,
    ) -> None:
        self.positive_moves_object = positive_moves_object
        self.grasp_frame_converges = grasp_frame_converges
        self.grasp_frame_step_fraction = float(grasp_frame_step_fraction)
        self.reset_count = 0
        self.step_index = 0
        self.joints = [0.1 * index for index in range(7)]
        self.can = list(START)
        self.grasp_frame = list(START)
        self.controlled_body_quaternion = [1.0, 0.0, 0.0, 0.0]
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
        self.grasp_frame = list(START)
        self.controlled_body_quaternion = [1.0, 0.0, 0.0, 0.0]
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
            "grasp_frame_position_world_m": list(self.grasp_frame),
            "controlled_body_pose_world": [
                *self.grasp_frame,
                *self.controlled_body_quaternion,
            ],
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
        max_task_space_translation_step_m,
        orientation_tolerance_deg,
    ):
        assert target_quaternion_world_xyzw == [1.0, 0.0, 0.0, 0.0]
        assert max_joint_delta_rad == 0.03
        assert max_task_space_translation_step_m == 0.01
        assert orientation_tolerance_deg == 2.0
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
        if self.grasp_frame_converges and self.pending_target is not None:
            self.grasp_frame = [
                current
                + self.grasp_frame_step_fraction * (target - current)
                for current, target in zip(
                    self.grasp_frame, self.pending_target, strict=True
                )
            ]
        if not self.positive_moves_object or self.pending_target is None:
            return
        target = self.pending_target
        at_object_xy = np.linalg.norm(np.asarray(target[:2]) - np.asarray(self.can[:2])) < 0.05
        if self.pending_gripper == 0.0 and at_object_xy:
            self.gripped = True
        if self.gripped and self.pending_gripper == 0.0:
            self.can[0] = target[0]
            self.can[1] = target[1]
            self.can[2] = max(
                START[2],
                target[2] - _instance()["resolved_parameters"]["object_height_m"] / 2.0,
            )
        if self.pending_gripper == 1.0:
            self.gripped = False
            if np.linalg.norm(np.asarray(target[:2]) - np.asarray(TARGET[:2])) < 0.05:
                self.can = list(TARGET)


class _TransientArrivalEnvironment(_ControlEnvironment):
    def step(self, isaac_action):
        super().step(isaac_action)
        if self.pending_target is None:
            return
        if self.step_index == 2:
            self.grasp_frame = list(START)


class _LegacyToleranceStallEnvironment(_ControlEnvironment):
    """Replay the paid canary's unsafe pregrasp residual without GPU imports."""

    def step(self, isaac_action):
        super().step(isaac_action)
        if self.pending_target is not None:
            self.grasp_frame = [
                self.pending_target[0],
                self.pending_target[1] - 0.01585,
                self.pending_target[2],
            ]


class _PaidCanaryOrientationStallEnvironment(_ControlEnvironment):
    """Replay the paid canary's safe position with its unsafe wrist angle."""

    def step(self, isaac_action):
        super().step(isaac_action)
        # 9.1716 degrees from the preregistered [1, 0, 0, 0] task orientation.
        half_angle = np.deg2rad(9.1716) / 2.0
        self.controlled_body_quaternion = [
            float(np.cos(half_angle)),
            0.0,
            0.0,
            float(np.sin(half_angle)),
        ]


def test_control_plan_is_deterministic_and_bound_to_the_scenario_instance() -> None:
    instance = _instance()

    first = materialize_control_plan(instance)
    second = materialize_control_plan(instance)

    assert first == second
    assert first["instance_digest"] == instance["instance_digest"]
    assert first["resolved_destination_position_world_m"] == TARGET
    assert first["grasp_target_frame"] == "probe_calibrated_finger_midpoint"
    assert first["controlled_body_orientation_strategy"] == (
        "horizontal_support_top_down_task_orientation"
    )
    assert first["controlled_body_quaternion_world_xyzw"] == [1.0, 0.0, 0.0, 0.0]
    grasp = next(
        phase for phase in first["scripted_positive_phases"]
        if phase["phase_id"] == "grasp"
    )
    assert grasp["target_position_world_m"] == pytest.approx(
        [
            START[0],
            START[1],
            START[2]
            + instance["resolved_parameters"]["object_height_m"] / 2.0,
        ]
    )
    assert grasp["target_frame"] == "probe_calibrated_finger_midpoint"
    assert grasp["target_quaternion_world_xyzw"] == [1.0, 0.0, 0.0, 0.0]
    assert grasp["arrival_tolerance_m"] == pytest.approx(
        0.085 / 2.0
        - instance["resolved_parameters"]["object_radius_m"]
        - 0.003
    )
    assert grasp["arrival_tolerance_basis"] == (
        "open_jaw_radial_clearance_minus_safety_clearance"
    )
    assert first["open_gripper_geometry"] == {
        "full_opening_m": 0.085,
        "object_diameter_m": pytest.approx(
            instance["resolved_parameters"]["object_radius_m"] * 2.0
        ),
        "radial_clearance_m": pytest.approx(
            0.085 / 2.0
            - instance["resolved_parameters"]["object_radius_m"]
        ),
        "safety_clearance_m": 0.003,
        "aperture_safe_arrival_tolerance_m": pytest.approx(
            0.085 / 2.0
            - instance["resolved_parameters"]["object_radius_m"]
            - 0.003
        ),
    }
    assert grasp["minimum_steps"] == 30
    assert grasp["maximum_steps"] == 120
    assert grasp["arrival_stability_steps"] == 3
    assert grasp["orientation_tolerance_deg"] == 2.0
    assert grasp["orientation_tolerance_basis"] == (
        "top_down_task_orientation_angular_distance"
    )
    assert grasp["max_task_space_translation_step_m"] == 0.01
    assert grasp["task_space_translation_strategy"] == (
        "orientation_first_bounded_local_increment"
    )
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
    for phase in first["scripted_positive_phases"]:
        if phase["mode"] == "ik_pose" and phase["phase_id"] not in {
            "grasp",
            "release",
        }:
            assert phase["minimum_steps"] == 1
            assert phase["maximum_steps"] == 240
            assert phase["arrival_stability_steps"] == 3


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
    assert all(row["target_reached"] for row in positive["phase_arrivals"])
    assert all(
        row["termination_reason"] == "stable_arrival"
        for row in positive["phase_arrivals"]
    )
    assert {
        row["phase_id"]: row["steps_executed"]
        for row in positive["phase_arrivals"]
        if row["phase_id"] in {"grasp", "release"}
    } == {"grasp": 30, "release": 30}
    assert (tmp_path / "adp009d_control_plan.v7.json").is_file()
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


def test_nonconverging_phase_aborts_before_grasp_and_retains_typed_evidence(
    tmp_path: Path,
) -> None:
    pair = run_required_controls(
        environment=_ControlEnvironment(grasp_frame_converges=False),
        scenario_instance=_instance(),
        gripper_open_command=1.0,
        gripper_closed_command=0.0,
        output_dir=tmp_path,
    )

    positive = json.loads(
        (tmp_path / f"adp009d_control_episode.{SCRIPTED_POSITIVE}.json").read_text()
    )
    assert pair["cell_admitted_for_policy_execution"] is False
    assert any(
        blocker.startswith(f"{BLOCKER_PHASE_NOT_REACHED}:pregrasp:error_m=")
        for blocker in pair["policy_execution_blockers"]
    )
    assert positive["environment_steps"] == 240
    assert positive["phase_arrivals"] == [
        {
            "phase_id": "pregrasp",
            "target_frame": "probe_calibrated_finger_midpoint",
            "target_position_world_m": pytest.approx(
                [START[0], START[1], START[2] + 0.42]
            ),
            "start_position_world_m": START,
            "achieved_position_world_m": START,
            "terminal_position_error_m": pytest.approx(0.42),
            "terminal_lateral_error_m": 0.0,
            "arrival_tolerance_m": pytest.approx(
                0.085 / 2.0
                - _instance()["resolved_parameters"]["object_radius_m"]
                - 0.003
            ),
            "arrival_tolerance_basis": (
                "open_jaw_radial_clearance_minus_safety_clearance"
            ),
            "terminal_position_within_tolerance": False,
            "terminal_orientation_error_deg": 0.0,
            "orientation_tolerance_deg": 2.0,
            "orientation_tolerance_basis": (
                "top_down_task_orientation_angular_distance"
            ),
            "terminal_orientation_within_tolerance": True,
            "terminal_within_tolerance": False,
            "minimum_steps": 1,
            "maximum_steps": 240,
            "steps_executed": 240,
            "arrival_stability_steps_required": 3,
            "arrival_stability_steps_observed": 0,
            "termination_reason": "maximum_steps_exhausted",
            "target_reached": False,
        }
    ]
    assert {row["phase_id"] for row in positive["action_trace"]} == {"pregrasp"}


def test_control_plan_rejects_object_that_cannot_clear_open_jaws() -> None:
    instance = _instance()
    instance["resolved_parameters"]["object_radius_m"] = 0.041
    instance["instance_digest"] = canonical_digest(
        instance, digest_field="instance_digest"
    )

    with pytest.raises(ControlEpisodeError) as excinfo:
        materialize_control_plan(instance)

    assert "control_plan_object_open_jaw_clearance_insufficient" in (
        excinfo.value.errors
    )


def test_control_plan_requires_observed_object_radius() -> None:
    instance = _instance()
    del instance["resolved_parameters"]["object_radius_m"]
    instance["instance_digest"] = canonical_digest(
        instance, digest_field="instance_digest"
    )

    with pytest.raises(ControlEpisodeError) as excinfo:
        materialize_control_plan(instance)

    assert "control_plan_object_radius_missing" in excinfo.value.errors


def test_paid_canary_legacy_pregrasp_residual_is_not_admitted(
    tmp_path: Path,
) -> None:
    plan = materialize_control_plan(_instance())
    plan["scripted_positive_phases"] = [plan["scripted_positive_phases"][0]]
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")

    receipt = run_control_episode(
        environment=_LegacyToleranceStallEnvironment(positive_moves_object=False),
        plan=plan,
        control_id=SCRIPTED_POSITIVE,
        gripper_open_command=1.0,
        gripper_closed_command=0.0,
        media_output_dir=tmp_path,
        episode_id="paid-canary-pregrasp-residual",
    )

    arrival = receipt["phase_arrivals"][0]
    assert arrival["terminal_position_error_m"] == pytest.approx(0.01585)
    assert arrival["terminal_lateral_error_m"] == pytest.approx(0.01585)
    assert arrival["arrival_tolerance_m"] == pytest.approx(0.00840527398565496)
    assert arrival["target_reached"] is False
    assert "lateral_error_m=0.015850" in receipt["phase_execution_blocker"]


def test_paid_canary_pregrasp_orientation_residual_is_not_admitted(
    tmp_path: Path,
) -> None:
    plan = materialize_control_plan(_instance())
    plan["scripted_positive_phases"] = [plan["scripted_positive_phases"][0]]
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")

    receipt = run_control_episode(
        environment=_PaidCanaryOrientationStallEnvironment(
            positive_moves_object=False
        ),
        plan=plan,
        control_id=SCRIPTED_POSITIVE,
        gripper_open_command=1.0,
        gripper_closed_command=0.0,
        media_output_dir=tmp_path,
        episode_id="paid-canary-pregrasp-orientation-residual",
    )

    arrival = receipt["phase_arrivals"][0]
    assert arrival["terminal_position_within_tolerance"] is True
    assert arrival["terminal_orientation_error_deg"] == pytest.approx(9.1716)
    assert arrival["orientation_tolerance_deg"] == 2.0
    assert arrival["terminal_orientation_within_tolerance"] is False
    assert arrival["terminal_within_tolerance"] is False
    assert arrival["target_reached"] is False
    assert "orientation_error_deg=9.171600" in receipt[
        "phase_execution_blocker"
    ]


def test_slowly_converging_phase_runs_past_legacy_budget_then_stops_early(
    tmp_path: Path,
) -> None:
    plan = materialize_control_plan(_instance())
    plan["scripted_positive_phases"] = [plan["scripted_positive_phases"][0]]
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")

    receipt = run_control_episode(
        environment=_ControlEnvironment(
            positive_moves_object=False,
            grasp_frame_step_fraction=0.025,
        ),
        plan=plan,
        control_id=SCRIPTED_POSITIVE,
        gripper_open_command=1.0,
        gripper_closed_command=0.0,
        media_output_dir=tmp_path,
        episode_id="slow-convergence",
    )

    arrival = receipt["phase_arrivals"][0]
    assert 80 < arrival["steps_executed"] < arrival["maximum_steps"]
    assert arrival["arrival_stability_steps_observed"] == 3
    assert arrival["termination_reason"] == "stable_arrival"
    assert arrival["terminal_within_tolerance"] is True
    assert arrival["target_reached"] is True
    assert receipt["phase_execution_blocker"] is None


def test_phase_arrival_requires_consecutive_stable_samples(tmp_path: Path) -> None:
    plan = materialize_control_plan(_instance())
    plan["scripted_positive_phases"] = [plan["scripted_positive_phases"][0]]
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")

    receipt = run_control_episode(
        environment=_TransientArrivalEnvironment(positive_moves_object=False),
        plan=plan,
        control_id=SCRIPTED_POSITIVE,
        gripper_open_command=1.0,
        gripper_closed_command=0.0,
        media_output_dir=tmp_path,
        episode_id="transient-arrival",
    )

    arrival = receipt["phase_arrivals"][0]
    assert arrival["steps_executed"] == 5
    assert arrival["arrival_stability_steps_observed"] == 3
    assert arrival["termination_reason"] == "stable_arrival"


def test_control_episode_rejects_legacy_plan_schema(tmp_path: Path) -> None:
    plan = materialize_control_plan(_instance())
    plan["schema_version"] = "adp009d_control_plan.v3"
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")

    with pytest.raises(ControlEpisodeError) as excinfo:
        run_control_episode(
            environment=_ControlEnvironment(),
            plan=plan,
            control_id=ZERO_ACTION_NEGATIVE,
            gripper_open_command=1.0,
            gripper_closed_command=0.0,
            media_output_dir=tmp_path,
            episode_id="legacy-plan",
        )

    assert "control_episode_plan_schema_invalid" in excinfo.value.errors
