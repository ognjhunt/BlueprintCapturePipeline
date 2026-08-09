from __future__ import annotations

import numpy as np
import pytest

from blueprint_pipeline.adp009d_droid_action_execution import GripperConvention
from blueprint_pipeline.adp009d_droid_observation import (
    DROID_EXTERIOR_VIEW_1,
    DROID_WRIST_VIEW,
)
from blueprint_pipeline.adp009d_policy_episode import (
    BLOCKER_CLIENT_RETURNED_NOTHING,
    BLOCKER_ENVIRONMENT_CONTRACT,
    BLOCKER_STEP_INDEX_NOT_INCREASING,
    PolicyEpisodeError,
    run_policy_episode,
)
from blueprint_pipeline.adp009d_task_scoring import (
    CAN_START_POSITION_M,
    GRIPPER_FULL_OPENING_M,
    SUPPORT_PLANE_Z_M,
)

_MEASURED = GripperConvention(
    closed_command=1.0, open_command=0.0, measured_by_probe=True
)
# The frozen destination, derived from the sealed SAGE support triangles.
_DESTINATION = [3.750152333333333, -3.4074919, SUPPORT_PLANE_Z_M]
_UPRIGHT_XYZW = (0.0, 0.0, 0.0, 1.0)
_LIMITS = [[-2.9, 2.9]] * 7
_CLOSED = 0.070


class _Environment:
    """A scripted simulator: carries the can to the destination, then releases."""

    def __init__(self, *, steps_to_destination: int = 24, lift_height: float = 0.12):
        self.steps_to_destination = steps_to_destination
        self.lift_height = lift_height
        self.reset_count = 0
        self.steps: list[list[float]] = []
        self._joints = [0.0] * 7
        self._t = 0

    def reset(self) -> None:
        self.reset_count += 1
        self._t = 0
        self._joints = [0.0] * 7
        self.steps.clear()

    def joint_limits(self):
        return _LIMITS

    def read_policy_inputs(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[..., 0] = 128
        return {
            DROID_EXTERIOR_VIEW_1: frame,
            DROID_WRIST_VIEW: frame,
            "joint_position": list(self._joints),
            "gripper_position": 0.04,
            "eef_9d": [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0],
        }

    def step(self, isaac_action):
        self.steps.append(list(isaac_action))
        self._joints = [float(value) for value in isaac_action[:7]]
        self._t += 1

    def read_arm_joint_positions(self):
        return list(self._joints)

    def _position(self):
        """Lift, translate to the destination, then rest on the support."""

        progress = min(1.0, self._t / self.steps_to_destination)
        x = CAN_START_POSITION_M[0] + progress * (_DESTINATION[0] - CAN_START_POSITION_M[0])
        y = CAN_START_POSITION_M[1] + progress * (_DESTINATION[1] - CAN_START_POSITION_M[1])
        if self._t == 0 or progress >= 1.0:
            z = SUPPORT_PLANE_Z_M
        else:
            z = SUPPORT_PLANE_Z_M + self.lift_height
        return [x, y, z]

    def read_object_sample(self):
        progress = min(1.0, self._t / self.steps_to_destination)
        carrying = 0.0 < progress < 1.0
        position = self._position()
        sample = {
            "can_pose_world": [*position, *_UPRIGHT_XYZW],
            "gripper_width_m": _CLOSED if carrying else GRIPPER_FULL_OPENING_M,
        }
        if carrying:
            sample["grasp_frame_position_world_m"] = list(position)
            sample["finger_contact_forces_n"] = [2.5, 2.5]
        return sample


class _Policy:
    """Returns a well-formed 10x8 chunk and records what it was asked."""

    def __init__(self):
        self.observations: list[dict] = []

    def infer(self, observation):
        self.observations.append(observation)
        chunk = np.zeros((10, 8), dtype=float)
        chunk[:, 0] = 0.25
        chunk[:, 7] = 0.9  # closed, in DROID's convention
        return chunk


def _run(environment=None, policy=None, **overrides):
    kwargs = dict(
        environment=environment or _Environment(),
        policy=policy or _Policy(),
        candidate_id="pi05_droid",
        destination_position_world_m=_DESTINATION,
        prompt="pick up the can and place it on the counter",
        gripper=_MEASURED,
        max_policy_queries=4,
        settle_window_samples=6,
    )
    kwargs.update(overrides)
    return run_policy_episode(**kwargs)


def _articulated_task_spec(*, maximum_action_steps: int = 32) -> dict:
    return {
        "schema_version": "adp_task_spec.v1",
        "task_kind": "articulated_open_close",
        "task_id": "840796_refrigerator_upper_door_open_v1",
        "target_joint_id": "refrigerator_upper_door_hinge",
        "joint_reset_positions_rad": {
            "refrigerator_upper_door_hinge": 0.0,
            "refrigerator_lower_door_hinge": 0.0,
        },
        "target_success_interval_rad": [0.785398163, 0.959931089],
        "joint_hard_limits_rad": {
            "refrigerator_upper_door_hinge": [0.0, 1.570796327],
            "refrigerator_lower_door_hinge": [0.0, 1.570796327],
        },
        "settle_window_samples": 6,
        "maximum_settled_target_speed_rad_s": 0.05,
        "non_task_joint_motion_tolerance_rad": 0.001,
        "movement_epsilon_rad": 0.0001,
        "reset_tolerance_rad": 0.0001,
        "control_frequency_hz": 15,
        "maximum_action_steps": maximum_action_steps,
    }


class _ArticulatedEnvironment(_Environment):
    """Second fixture: native joint state, without any canned-object fields."""

    def __init__(self):
        super().__init__()
        self._last_gripper_command = _MEASURED.open_command

    def reset(self) -> None:
        super().reset()
        self._last_gripper_command = _MEASURED.open_command

    def step(self, isaac_action):
        super().step(isaac_action)
        self._last_gripper_command = float(isaac_action[7])

    def read_object_sample(self):
        raise AssertionError("articulated episode must not read canned-object state")

    def read_task_sample(self):
        upper = min(0.9, 0.9 * self._t / 16.0)
        return {
            "joint_positions_rad": {
                "refrigerator_upper_door_hinge": upper,
                "refrigerator_lower_door_hinge": 0.0,
            },
            "joint_velocities_rad_s": {
                "refrigerator_upper_door_hinge": 0.0 if self._t >= 16 else 0.01,
                "refrigerator_lower_door_hinge": 0.0,
            },
            "task_contact_active": self._last_gripper_command != _MEASURED.open_command,
            "joint_limit_violation": False,
            "containment_violation": False,
            "robot_collision_failure": False,
            "scene_collision_failure": False,
            "retreat_completed": self._t >= 24,
        }


def test_a_full_episode_composes_all_five_adapters_and_reaches_placed() -> None:
    """The whole point: observation, query, execution and scoring in one path."""

    environment = _Environment()
    policy = _Policy()

    receipt = _run(environment, policy)

    assert receipt["candidate_policy_queried"] is True
    assert receipt["motion_evidence"]["arm_moved"] is True
    assert receipt["motion_evidence"]["actions_reached_robot"] is True
    assert receipt["motion_evidence"]["policy_outcome_interpretable"] is True
    assert receipt["policy_queries"] == 4
    assert receipt["action_space"] == "droid_joint_velocity_plus_absolute_gripper"
    # 4 queries x 8 executed rows, plus the settle window.
    assert receipt["environment_steps"] == 4 * 8 + 6
    assert len(environment.steps) == receipt["environment_steps"]

    # The policy actually saw this candidate's observation format.
    assert len(policy.observations) == 4
    assert policy.observations[0][DROID_EXTERIOR_VIEW_1].shape == (224, 224, 3)
    assert policy.observations[0][DROID_WRIST_VIEW].dtype == np.uint8
    assert policy.observations[0]["prompt"]

    # And the episode scored on deterministic object state.
    assert receipt["score"]["status"] in {"scored", "undetermined"}
    assert receipt["score"]["outcome"] == "placed"
    timings = receipt["performance_diagnostics"]["timings_seconds"]
    assert timings["policy_inference"] >= 0.0
    assert timings["environment_step_including_render"] >= 0.0
    assert timings["total"] >= sum(
        timings[key]
        for key in (
            "policy_inference",
            "environment_step_including_render",
            "settle_steps_including_render",
        )
    ) - 1e-5
    assert (
        receipt["performance_diagnostics"][
            "environment_step_bucket_includes_renderer_when_enabled"
        ]
        is True
    )

    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )


def test_same_policy_episode_loop_scores_second_scene_native_articulation_state() -> None:
    environment = _ArticulatedEnvironment()

    receipt = _run(
        environment,
        task_spec=_articulated_task_spec(),
        destination_position_world_m=None,
        prompt="Open the upper refrigerator door, release it, and retreat.",
    )

    assert receipt["task_kind"] == "articulated_open_close"
    assert receipt["destination_position_world_m"] is None
    assert receipt["score"]["outcome"] == "opened_and_settled"
    assert receipt["score"]["outcome_rank"] == 4
    assert receipt["score"]["judgement_source"] == (
        "deterministic_native_simulator_joint_state"
    )
    assert environment.reset_count == 1
    assert len(environment.steps) == receipt["environment_steps"]


def test_policy_episode_rejects_actions_beyond_frozen_articulated_budget() -> None:
    environment = _ArticulatedEnvironment()

    with pytest.raises(
        PolicyEpisodeError, match="policy_episode_action_budget_exceeds_task_spec"
    ):
        _run(
            environment,
            task_spec=_articulated_task_spec(maximum_action_steps=31),
            destination_position_world_m=None,
        )

    assert environment.reset_count == 0


def test_groot_absolute_joint_actions_take_the_direct_position_path() -> None:
    class _AbsolutePolicy(_Policy):
        action_space = "joint_position"

        def infer(self, observation):
            self.observations.append(observation)
            chunk = np.zeros((40, 8), dtype=float)
            chunk[:, :7] = [0.7, -0.8, 0.3, -1.2, 0.4, 1.1, -0.2]
            chunk[:, 7] = 1.0
            return chunk

        def last_inference_evidence(self):
            return {
                "native_action_chunk_shape": [40, 17],
                "native_action_chunk_sha256": "a" * 64,
            }

    environment = _Environment()
    policy = _AbsolutePolicy()
    receipt = _run(
        environment,
        policy,
        candidate_id="groot_n17_droid",
        max_policy_queries=1,
        settle_window_samples=1,
    )

    assert environment.steps[0][:7] == pytest.approx(
        [0.7, -0.8, 0.3, -1.2, 0.4, 1.1, -0.2]
    )
    assert receipt["action_space"] == (
        "groot_decoded_absolute_joint_position_plus_absolute_gripper"
    )
    assert receipt["queries"][0]["position_adapter"] == (
        "decoded_absolute_joint_position_direct_with_limit_clamp"
    )
    assert receipt["queries"][0]["policy_inference_evidence"] == {
        "native_action_chunk_shape": [40, 17],
        "native_action_chunk_sha256": "a" * 64,
    }
    assert receipt["commanded_action_magnitudes"][
        "joint_velocity_command_max_abs_rad_s"
    ] == 0.0
    assert "observation/eef_9d" in policy.observations[0]


def test_groot_history_is_exactly_fifteen_simulator_steps_not_policy_queries() -> None:
    class _TemporalEnvironment(_Environment):
        def read_policy_inputs(self):
            inputs = super().read_policy_inputs()
            for view in (DROID_EXTERIOR_VIEW_1, DROID_WRIST_VIEW):
                inputs[view] = np.full(
                    (720, 1280, 3), self._t, dtype=np.uint8
                )
            return inputs

    class _AbsolutePolicy(_Policy):
        action_space = "joint_position"

        def infer(self, observation):
            self.observations.append(observation)
            return np.zeros((40, 8), dtype=float)

    policy = _AbsolutePolicy()
    _run(
        _TemporalEnvironment(),
        policy,
        candidate_id="groot_n17_droid",
        max_policy_queries=3,
        settle_window_samples=1,
    )

    third = policy.observations[2]  # query steps are 0, 8, 16
    assert third[DROID_EXTERIOR_VIEW_1][0, 0, 0] == 16
    assert (
        third["observation_history/exterior_image_1_left_t_minus_15"][0, 0, 0]
        == 1
    )


def test_successful_episode_retains_exact_policy_inputs_and_review_video(
    tmp_path,
) -> None:
    from PIL import Image

    receipt = _run(
        media_output_dir=tmp_path,
        episode_id="pi05-droid-episode-000",
    )

    visual = receipt["visual_evidence"]
    assert visual["status"] == "complete"
    assert visual["human_review_available"] is True
    assert visual["policy_input_frame_count"] == receipt["policy_queries"]
    assert visual["video"]["frame_count"] == receipt["policy_queries"] + 1
    assert visual["video"]["derived_from_frame_manifest_digest"] == visual[
        "frame_manifest_digest"
    ]
    assert receipt["observation_trace_digest"].startswith("sha256:")
    first = next(
        row for row in receipt["media_artifacts"] if row["role"] == "policy_input_frame"
    )
    with Image.open(tmp_path / first["relative_path"]) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    # Both post-preprocessing 224x224 views shown to the policy are retained,
    # left-to-right in the candidate's frozen view order.
    assert pixels.shape == (224, 448, 3)
    assert np.array_equal(pixels[:, :224], pixels[:, 224:])
    assert (tmp_path / visual["video"]["relative_path"]).is_file()


def test_native_evaluation_media_adds_review_only_overview_without_policy_input(
    tmp_path,
) -> None:
    class _OverviewEnvironment(_Environment):
        def read_evaluation_camera_inputs(self):
            return {
                "external": np.full((24, 32, 3), 40, dtype=np.uint8),
                "wrist": np.full((24, 32, 3), 80, dtype=np.uint8),
                "overview": np.full((24, 32, 3), 160, dtype=np.uint8),
            }

        def read_control_observation_metadata(self):
            calibration = {
                "camera_model": "pinhole",
                "intrinsic_matrix": [
                    [20.0, 0.0, 16.0],
                    [0.0, 20.0, 12.0],
                    [0.0, 0.0, 1.0],
                ],
                "world_from_camera": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "resolution": [32, 24],
                "near_m": 0.01,
                "far_m": 10.0,
            }
            camera_ids = ("external", "wrist", "overview")
            return {
                "timestamp_ns": self._t * 1_000_000,
                "simulation_time_s": self._t / 15.0,
                "calibrations": {
                    camera_id: calibration for camera_id in camera_ids
                },
                "source_devices": {
                    camera_id: "cpu" for camera_id in camera_ids
                },
                "synchronizations": {
                    camera_id: {"host_bytes_ready": True, "method": "test"}
                    for camera_id in camera_ids
                },
            }

    policy = _Policy()
    receipt = _run(
        _OverviewEnvironment(),
        policy,
        max_policy_queries=1,
        settle_window_samples=2,
        media_output_dir=tmp_path,
        episode_id="overview-episode",
    )

    visual = receipt["visual_evidence"]
    assert set(visual["videos"]) == {"external", "wrist", "overview"}
    assert visual["review_only_camera_ids"] == ["overview"]
    assert visual["policy_input_frame_count"] == 2
    assert visual["review_observation_count"] == 1
    assert visual["review_frame_count"] == 3
    assert "overview" not in policy.observations[0]


def test_media_output_and_episode_identity_must_be_bound_together(tmp_path) -> None:
    with pytest.raises(PolicyEpisodeError) as excinfo:
        _run(media_output_dir=tmp_path)

    assert any("policy_media_binding_incomplete" in error for error in excinfo.value.errors)


def test_the_settle_window_releases_the_gripper() -> None:
    """placed is judged on a released can; a held one must not qualify."""

    environment = _Environment()

    _run(environment)

    settle = environment.steps[-6:]
    assert all(action[7] == _MEASURED.open_command for action in settle)
    # Everything before the settle carried the policy's own gripper command.
    assert environment.steps[-7][7] == _MEASURED.closed_command


def test_only_the_open_loop_horizon_of_each_chunk_executes() -> None:
    """A 10-row chunk must advance the simulator exactly eight steps."""

    environment = _Environment()

    receipt = _run(environment, max_policy_queries=1, settle_window_samples=2)

    assert receipt["queries"][0]["chunk_shape"] == [10, 8]
    assert receipt["queries"][0]["executed_rows"] == 8
    assert receipt["queries"][0]["discarded_rows"] == 2
    assert len(environment.steps) == 8 + 2


def test_the_episode_resets_before_it_observes_anything() -> None:
    environment = _Environment()

    _run(environment)

    assert environment.reset_count == 1


def test_a_client_returning_nothing_fails_closed() -> None:
    class _Silent:
        def infer(self, observation):
            return None

    with pytest.raises(PolicyEpisodeError) as excinfo:
        _run(policy=_Silent())
    assert BLOCKER_CLIENT_RETURNED_NOTHING in excinfo.value.errors


def test_a_malformed_chunk_never_reaches_the_simulator() -> None:
    from blueprint_pipeline.adp009d_droid_action_execution import (
        DroidActionExecutionError,
    )

    class _Wrong:
        def infer(self, observation):
            return np.zeros((10, 7))

    environment = _Environment()
    with pytest.raises(DroidActionExecutionError):
        _run(environment, policy=_Wrong())
    assert environment.steps == []


def test_an_environment_missing_a_required_view_fails_closed() -> None:
    class _OneEye(_Environment):
        def read_policy_inputs(self):
            inputs = dict(super().read_policy_inputs())
            del inputs[DROID_WRIST_VIEW]
            return inputs

    from blueprint_pipeline.adp009d_droid_observation import DroidObservationError

    with pytest.raises(DroidObservationError):
        _run(_OneEye())


def test_an_environment_missing_proprioception_names_the_contract() -> None:
    class _NoJoints(_Environment):
        def read_policy_inputs(self):
            inputs = dict(super().read_policy_inputs())
            del inputs["joint_position"]
            return inputs

    with pytest.raises(PolicyEpisodeError) as excinfo:
        _run(_NoJoints())
    assert any(BLOCKER_ENVIRONMENT_CONTRACT in e for e in excinfo.value.errors)


def test_an_environment_omitting_the_can_pose_fails_closed() -> None:
    class _NoPose(_Environment):
        def read_object_sample(self):
            return {"gripper_width_m": GRIPPER_FULL_OPENING_M}

    with pytest.raises(PolicyEpisodeError) as excinfo:
        _run(_NoPose())
    assert any(BLOCKER_ENVIRONMENT_CONTRACT in e for e in excinfo.value.errors)


def test_step_indices_increase_across_query_boundaries() -> None:
    """The scorer treats a repeated index as malformed, so the loop must not emit one."""

    environment = _Environment()
    receipt = _run(environment)

    # Re-derive the indices the loop must have produced.
    assert receipt["queries"][0]["final_step_index"] == 8
    assert receipt["queries"][1]["final_step_index"] == 16
    assert receipt["queries"][-1]["final_step_index"] == 32
    assert BLOCKER_STEP_INDEX_NOT_INCREASING not in str(receipt)


def test_an_unmeasured_gripper_convention_is_refused() -> None:
    from blueprint_pipeline.adp009d_droid_action_execution import (
        DroidActionExecutionError,
    )

    unmeasured = GripperConvention(closed_command=1.0, open_command=0.0)
    with pytest.raises(DroidActionExecutionError):
        _run(gripper=unmeasured)


def test_unknown_candidate_and_invalid_budgets_are_refused() -> None:
    with pytest.raises(PolicyEpisodeError):
        _run(candidate_id="some_other_policy")
    with pytest.raises(PolicyEpisodeError):
        _run(max_policy_queries=0)
    with pytest.raises(PolicyEpisodeError):
        _run(settle_window_samples=0)


def test_a_policy_that_never_moves_the_can_scores_never_moved() -> None:
    """A real negative must be reported as one, not as an error."""

    class _Static(_Environment):
        def read_object_sample(self):
            return {
                "can_pose_world": [*CAN_START_POSITION_M, *_UPRIGHT_XYZW],
                "gripper_width_m": GRIPPER_FULL_OPENING_M,
            }

    receipt = _run(_Static())

    assert receipt["score"]["outcome"] == "never_moved"
    assert receipt["candidate_policy_queried"] is True
    assert receipt["motion_evidence"]["arm_moved"] is True
    assert receipt["motion_evidence"]["policy_outcome_interpretable"] is True


def test_arm_motion_and_command_delivery_evidence_fail_closed() -> None:
    class _DroppedActions(_Environment):
        def step(self, isaac_action):
            self.steps.append(list(isaac_action))
            self._t += 1

    receipt = _run(_DroppedActions(), max_policy_queries=2, settle_window_samples=2)

    evidence = receipt["motion_evidence"]
    assert evidence["joint_position_reset_rad"] == [0.0] * 7
    assert evidence["joint_position_end_rad"] == [0.0] * 7
    assert evidence["max_abs_joint_delta_from_reset_rad"] == [0.0] * 7
    assert evidence["arm_moved"] is False
    assert evidence["actions_reached_robot"] is False
    assert evidence["policy_outcome_interpretable"] is False
    assert evidence["interpretation"] == (
        "nontrivial_actions_not_observed_at_robot_harness_fault"
    )
    magnitudes = receipt["commanded_action_magnitudes"]
    assert magnitudes["policy_action_rows_submitted"] == 16
    assert magnitudes["nontrivial_arm_target_rows"] == 16
    assert magnitudes["arm_target_delta_from_observed_max_abs_rad"] == 0.05
    assert magnitudes["joint_velocity_command_max_abs_rad_s"] == 0.25
    assert magnitudes["joint_velocity_command_clipped_value_count"] == 0


def test_missing_joint_observation_contract_fails_before_policy_query() -> None:
    class _NoJointReader(_Environment):
        read_arm_joint_positions = None

    with pytest.raises(PolicyEpisodeError) as excinfo:
        _run(_NoJointReader())
    assert any("read_arm_joint_positions_missing" in error for error in excinfo.value.errors)


def test_receipt_records_the_observation_conversion_actually_applied() -> None:
    receipt = _run()

    conversion = receipt["observation_conversion"]
    assert conversion["candidate_id"] == "pi05_droid"
    assert conversion["source_resolution_hw"] == [720, 1280]
    assert conversion["target_resolution_hw"] == [224, 224]
    assert conversion["scene_content_cropped"] is False


def test_the_shipped_openpi_client_satisfies_the_episode_loop_seam() -> None:
    """No new client is needed: the existing one already fits the protocol.

    The loop asks for exactly one method, infer(observation) -> chunk.  The
    shipped OpenPI websocket client provides it and additionally verifies
    server identity on construction, so binding it needs no adapter.
    """

    import inspect

    from blueprint_pipeline.openpi_droid_policy_runtime import (
        OpenPIWebsocketDroidPolicyClient,
    )

    assert hasattr(OpenPIWebsocketDroidPolicyClient, "infer")
    signature = inspect.signature(OpenPIWebsocketDroidPolicyClient.infer)
    assert list(signature.parameters) == ["self", "observation"]
    # Identity verification is not optional: the constructor fetches and
    # validates server metadata rather than trusting the endpoint.
    source = inspect.getsource(OpenPIWebsocketDroidPolicyClient.__init__)
    assert "get_server_metadata" in source
    assert "validate_server_metadata" in source


def test_a_client_shaped_like_the_shipped_one_drives_a_full_episode() -> None:
    """Bind by duck type, so the real client needs no wrapper to be used."""

    class _ShapedLikeOpenPI:
        learned_policy = True
        policy_id = "pi05_droid"
        action_space = "joint_position"

        def __init__(self):
            self.calls = 0

        def infer(self, observation):
            self.calls += 1
            chunk = np.zeros((10, 8), dtype=float)
            chunk[:, 7] = 0.9
            return chunk

    client = _ShapedLikeOpenPI()
    receipt = _run(policy=client)

    assert client.calls == receipt["policy_queries"] == 4
    assert receipt["candidate_policy_queried"] is True
