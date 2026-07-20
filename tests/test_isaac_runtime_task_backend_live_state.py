from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import isaac_runtime_task_backend as backend_module
from blueprint_pipeline.gear_sonic_joint_order_contract import (
    JOINT_ORDER_SCHEMA_VERSION,
    PROTOCOL_V4_BODY_JOINT_NAMES,
    PROTOCOL_V4_FULL_JOINT_ORDER,
    PROTOCOL_V4_MAPPING_DIGEST,
)


class _App:
    def __init__(self) -> None:
        self.update_count = 0

    def update(self) -> None:
        self.update_count += 1


class _Robot:
    def __init__(self, names: list[str]) -> None:
        self.dof_names = list(names)
        self.positions = [9.0 for _ in names]
        self.velocities = [9.0 for _ in names]
        self.world_position = [0.0, 0.0, 0.0]
        self.world_quaternion = [1.0, 0.0, 0.0, 0.0]
        self.linear_velocity = [9.0, 9.0, 9.0]
        self.angular_velocity = [9.0, 9.0, 9.0]
        self.joint_position_state_write: dict[int, float] = {}
        self.joint_velocity_state_write: dict[int, float] = {}

    def set_world_pose(self, *, position, orientation) -> None:
        self.world_position = [float(item) for item in position]
        self.world_quaternion = [float(item) for item in orientation]

    def get_world_pose(self):
        return list(self.world_position), list(self.world_quaternion)

    def set_joint_positions(self, positions, *, joint_indices) -> None:
        self.joint_position_state_write = {
            int(index): float(value)
            for value, index in zip(positions, joint_indices, strict=True)
        }
        for value, index in zip(positions, joint_indices, strict=True):
            self.positions[int(index)] = float(value)

    def set_joint_velocities(self, velocities, *, joint_indices) -> None:
        self.joint_velocity_state_write = {
            int(index): float(value)
            for value, index in zip(velocities, joint_indices, strict=True)
        }
        for value, index in zip(velocities, joint_indices, strict=True):
            self.velocities[int(index)] = float(value)

    def set_joint_position_targets(self, positions, *, joint_indices) -> None:
        self.position_targets = {
            int(index): float(value) for value, index in zip(positions, joint_indices, strict=True)
        }

    def set_joint_velocity_targets(self, velocities, *, joint_indices) -> None:
        self.velocity_targets = {
            int(index): float(value) for value, index in zip(velocities, joint_indices, strict=True)
        }

    def get_joint_positions(self):
        return list(self.positions)

    def get_joint_velocities(self):
        return list(self.velocities)

    def set_linear_velocity(self, velocity) -> None:
        self.linear_velocity = [float(item) for item in velocity]

    def set_angular_velocity(self, velocity) -> None:
        self.angular_velocity = [float(item) for item in velocity]

    def get_angular_velocity(self):
        return list(self.angular_velocity)


class _LiveIsaacRobot(_Robot):
    set_joint_position_targets = None
    set_joint_velocity_targets = None

    def __init__(self, names: list[str]) -> None:
        super().__init__(names)
        self.applied_actions: list[object] = []

    def apply_action(self, action) -> None:
        self.applied_actions.append(action)


def _backend(
    tmp_path: Path,
    *,
    names: list[str] | None = None,
    robot: _Robot | None = None,
):
    backend = backend_module.IsaacPersistentTaskBackend.__new__(
        backend_module.IsaacPersistentTaskBackend
    )
    backend.robot = robot or _Robot(
        names or list(reversed(PROTOCOL_V4_FULL_JOINT_ORDER))
    )
    backend.app = _App()
    backend.evidence_dir = tmp_path
    backend.session_id = "isaac-session-live-state-1"
    backend.stage_id = "stage-live-state-1"
    backend.robot_prim_path = "/World/G1"
    backend.robot_composition = {
        "start_pose_xyz": [1.25, -0.5, 0.8],
        "start_yaw_rad": 0.0,
    }
    return backend


def test_renderer_prewarm_allows_pre_articulation_physx_counter_reset(
    tmp_path: Path,
) -> None:
    class Timeline:
        playing = False

        def is_playing(self) -> bool:
            return self.playing

        def play(self) -> None:
            self.playing = True

        def commit(self) -> None:
            pass

    class SimulationManager:
        calls = 0

        @classmethod
        def get_num_physics_steps(cls) -> int:
            cls.calls += 1
            return 3 if cls.calls == 1 else 0

    class ReviewRenderer:
        heartbeat_callback = None

        def prewarm(self):
            return {
                "status": "passed",
                "heartbeat_callback_attached_during_prewarm": False,
                "render_step_delta_time_seconds": 0.0,
            }

        def attach_heartbeat_callback(self, _callback):
            return {"heartbeat_callback_attached_after_prewarm": True}

    backend = backend_module.IsaacPersistentTaskBackend.__new__(
        backend_module.IsaacPersistentTaskBackend
    )
    backend.evidence_dir = tmp_path
    backend.session_id = "renderer-reset-session"
    backend.stage_id = "renderer-reset-stage"
    backend.robot_prim_path = "/World/G1"
    backend._articulations = {}
    backend.timeline = Timeline()
    backend._simulation_manager = SimulationManager
    backend.review_renderer = ReviewRenderer()
    backend.app = _App()
    backend._articulation = lambda _path: SimpleNamespace(handles_initialized=True)
    backend._initialize_official_gear_sonic_standing_pose = lambda: {
        "status": "passed"
    }
    backend._initialize_right_arm_manipulation_ready_pose = lambda: {
        "status": "passed"
    }
    backend._write_live_state_snapshot = lambda **_kwargs: {"status": "ready"}

    evidence = backend._prewarm_review_renderer_and_initialize_robot()

    assert evidence["status"] == "passed"
    assert evidence["physics_step_count_before_prewarm"] == 3
    assert evidence["physics_step_count_after_prewarm"] == 0
    assert evidence["physics_step_delta_during_prewarm"] == -3
    assert evidence["physics_step_counter_reset_during_prewarm"] is True
    assert evidence["physics_steps_advanced_during_prewarm"] is False
    assert evidence["articulation_initialized_after_prewarm"] is True


def test_official_standing_pose_is_exact_protocol_v4_body_order() -> None:
    assert backend_module.GEAR_SONIC_ISAAC_STATE_SNAPSHOT_DEFAULT_PATH == (
        "/workspace/closed_loop_out/gear_sonic_isaac_state_snapshot.json"
    )
    assert len(backend_module.GEAR_SONIC_DEFAULT_BODY_STANDING_POSITIONS) == 29
    expected = (
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    )
    assert backend_module.GEAR_SONIC_DEFAULT_BODY_STANDING_POSITIONS == expected
    assert len(backend_module.GEAR_SONIC_DEFAULT_FULL_STANDING_POSITIONS) == 43
    assert backend_module.GEAR_SONIC_DEFAULT_FULL_STANDING_POSITIONS[29:] == (0.0,) * 14


def test_late_june_right_arm_manipulation_ready_pose_preserves_other_joints() -> None:
    standing = backend_module.GEAR_SONIC_DEFAULT_FULL_STANDING_POSITIONS
    ready = backend_module.GEAR_SONIC_RIGHT_ARM_MANIPULATION_READY_POSITIONS
    index_by_name = {
        name: index
        for index, name in enumerate(PROTOCOL_V4_FULL_JOINT_ORDER)
    }

    assert len(ready) == len(standing) == 43
    for name, index in index_by_name.items():
        expected_delta = (
            backend_module.GEAR_SONIC_RIGHT_ARM_MANIPULATION_READY_DELTAS_RAD.get(
                name, 0.0
            )
        )
        assert ready[index] == pytest.approx(standing[index] + expected_delta)
    assert ready[index_by_name["right_shoulder_pitch_joint"]] < 0.0
    assert ready[index_by_name["right_elbow_joint"]] < (
        standing[index_by_name["right_elbow_joint"]]
    )


def test_manipulation_ready_initialization_is_applied_and_persisted(
    tmp_path: Path,
) -> None:
    backend = _backend(tmp_path)
    backend._initialize_official_gear_sonic_standing_pose()

    evidence = backend._initialize_right_arm_manipulation_ready_pose()

    assert evidence["status"] == "passed"
    assert evidence["task_side"] == "right"
    assert evidence["measured_full_joint_positions"] == pytest.approx(
        backend_module.GEAR_SONIC_RIGHT_ARM_MANIPULATION_READY_POSITIONS
    )
    assert evidence["measured_full_joint_velocities"] == pytest.approx([0.0] * 43)
    assert evidence["final_measured_full_joint_positions"] == pytest.approx(
        backend_module.GEAR_SONIC_RIGHT_ARM_MANIPULATION_READY_POSITIONS
    )
    assert evidence["final_measured_full_joint_velocities"] == pytest.approx(
        [0.0] * 43
    )
    assert evidence["final_state_reassertion"] == {
        "status": "passed",
        "purpose": "pre_render_attested_base_and_arm_initialization",
        "start_pose_xyz": [1.25, -0.5, 0.8],
        "start_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "physics_updates_after_reassertion": 0,
        "episode_action": False,
        "surrogate": False,
    }
    assert evidence["source_standing_pose_sha256"] == (
        backend_module.GEAR_SONIC_DEFAULT_STANDING_POSE_SHA256
    )
    artifact_path = tmp_path / "gear_sonic_manipulation_ready_initialization.json"
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == evidence
    assert backend.manipulation_ready_initialization_artifact["sha256"] == (
        hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    )


def test_manipulation_ready_reasserts_attested_base_before_initial_render(
    tmp_path: Path,
) -> None:
    backend = _backend(tmp_path)
    backend._initialize_official_gear_sonic_standing_pose()
    robot = backend.robot

    class DriftingBaseApp(_App):
        def update(self) -> None:
            super().update()
            robot.world_position = [1.8, 0.2, 0.4]
            robot.linear_velocity = [1.0, 2.0, 3.0]
            robot.angular_velocity = [0.2, 0.3, 0.4]

    backend.app = DriftingBaseApp()
    evidence = backend._initialize_right_arm_manipulation_ready_pose()

    assert evidence["status"] == "passed"
    assert robot.world_position == pytest.approx([1.25, -0.5, 0.8])
    assert robot.linear_velocity == pytest.approx([0.0, 0.0, 0.0])
    assert robot.angular_velocity == pytest.approx([0.0, 0.0, 0.0])
    assert evidence["final_maximum_joint_position_error_rad"] == pytest.approx(
        0.0, abs=1e-6
    )
    assert evidence["final_maximum_joint_velocity_rad_s"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("retained_fraction", "passes"),
    [(0.95, True), (0.8, False)],
)
def test_manipulation_ready_initialization_requires_signed_forward_seed_progress(
    tmp_path: Path,
    retained_fraction: float,
    passes: bool,
) -> None:
    backend = _backend(tmp_path)
    backend._initialize_official_gear_sonic_standing_pose()
    robot = backend.robot
    index_by_name = {name: index for index, name in enumerate(robot.dof_names)}

    class PartialCommitApp(_App):
        def update(self) -> None:
            super().update()
            for name, delta in (
                backend_module.GEAR_SONIC_RIGHT_ARM_MANIPULATION_READY_DELTAS_RAD.items()
            ):
                protocol_index = PROTOCOL_V4_FULL_JOINT_ORDER.index(name)
                robot.positions[index_by_name[name]] = (
                    backend_module.GEAR_SONIC_DEFAULT_FULL_STANDING_POSITIONS[
                        protocol_index
                    ]
                    + delta * retained_fraction
                )

    backend.app = PartialCommitApp()
    artifact_path = tmp_path / "gear_sonic_manipulation_ready_initialization.json"
    if passes:
        evidence = backend._initialize_right_arm_manipulation_ready_pose()
        assert evidence["status"] == "passed"
        assert evidence["insufficient_delta_joints"] == []
        assert evidence["minimum_requested_delta_fraction"] == pytest.approx(0.9)
        assert min(
            evidence["achieved_requested_delta_fraction_by_joint"].values()
        ) == pytest.approx(retained_fraction)
    else:
        with pytest.raises(
            RuntimeError,
            match="persistent_isaac_manipulation_ready_joint_position_verification_failed",
        ):
            backend._initialize_right_arm_manipulation_ready_pose()
        evidence = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert evidence["status"] == "blocked"
        assert evidence["insufficient_delta_joints"]
        assert evidence["measured_full_joint_positions"]


def test_manipulation_ready_stability_is_scoped_to_seeded_arm_joints(
    tmp_path: Path,
) -> None:
    backend = _backend(tmp_path)
    backend._initialize_official_gear_sonic_standing_pose()
    robot = backend.robot
    index_by_name = {name: index for index, name in enumerate(robot.dof_names)}

    class UnrelatedDynamicApp(_App):
        def update(self) -> None:
            super().update()
            robot.velocities[index_by_name["waist_roll_joint"]] = 2.99
            robot.velocities[index_by_name["right_wrist_pitch_joint"]] = 0.49

    backend.app = UnrelatedDynamicApp()
    evidence = backend._initialize_right_arm_manipulation_ready_pose()

    assert evidence["status"] == "passed"
    assert evidence["maximum_joint_velocity_rad_s"] == pytest.approx(2.99)
    assert evidence["maximum_manipulation_ready_joint_velocity_rad_s"] == (
        pytest.approx(0.49)
    )
    assert evidence[
        "maximum_manipulation_ready_joint_velocity_tolerance_rad_s"
    ] == pytest.approx(0.5)
    assert evidence["settle_steps_executed"] == 1


def test_manipulation_ready_stability_rejects_fast_seeded_arm_joint(
    tmp_path: Path,
) -> None:
    backend = _backend(tmp_path)
    backend._initialize_official_gear_sonic_standing_pose()
    robot = backend.robot
    index_by_name = {name: index for index, name in enumerate(robot.dof_names)}

    class FastArmApp(_App):
        def update(self) -> None:
            super().update()
            robot.velocities[index_by_name["right_elbow_joint"]] = 0.51

    backend.app = FastArmApp()
    with pytest.raises(
        RuntimeError,
        match="persistent_isaac_manipulation_ready_joint_velocity_verification_failed",
    ):
        backend._initialize_right_arm_manipulation_ready_pose()

    evidence = json.loads(
        (tmp_path / "gear_sonic_manipulation_ready_initialization.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence["status"] == "blocked"
    assert evidence["maximum_manipulation_ready_joint_velocity_rad_s"] == (
        pytest.approx(0.51)
    )
    assert evidence["settle_steps_executed"] == (
        backend_module.GEAR_SONIC_MANIPULATION_READY_MAX_SETTLE_STEPS
    )


def test_state_snapshot_reorders_live_dofs_into_exact_body_protocol_order() -> None:
    names = list(reversed(PROTOCOL_V4_FULL_JOINT_ORDER))
    protocol_index = {name: index for index, name in enumerate(PROTOCOL_V4_FULL_JOINT_ORDER)}
    positions = [protocol_index[name] + 0.25 for name in names]
    velocities = [-(protocol_index[name] + 0.5) for name in names]
    payload = backend_module.build_gear_sonic_isaac_state_snapshot(
        live_joint_names=names,
        live_joint_positions=positions,
        live_joint_velocities=velocities,
        base_quaternion_wxyz=[2.0, 0.0, 0.0, 0.0],
        base_angular_velocity_xyz=[0.1, 0.2, 0.3],
        simulator_session_id="isaac-session-1",
        stage_id="stage-1",
        heartbeat_sequence=4,
        captured_at_ns=1_000_000_000,
        source="post_action_live_isaac_articulation",
        source_action_sha256="a" * 64,
        source_step_index=3,
    )
    assert payload["schema_version"] == (
        backend_module.GEAR_SONIC_ISAAC_STATE_SNAPSHOT_SCHEMA_VERSION
    )
    assert payload["body_joint_names"] == list(PROTOCOL_V4_BODY_JOINT_NAMES)
    assert payload["body_joint_positions"] == [index + 0.25 for index in range(29)]
    assert payload["body_q"] == payload["body_joint_positions"]
    assert payload["body_joint_velocities"] == [-(index + 0.5) for index in range(29)]
    assert payload["body_dq"] == payload["body_joint_velocities"]
    assert payload["base_quaternion_wxyz"] == [1.0, 0.0, 0.0, 0.0]
    assert payload["base_angular_velocity_xyz"] == [0.1, 0.2, 0.3]
    assert payload["base_angular_velocity"] == [0.1, 0.2, 0.3]
    assert payload["accelerometer_mps2"] == pytest.approx([0.0, 0.0, 9.81])
    assert payload["source"] == "live_isaac_articulation"
    assert payload["capture_reason"] == "post_action_live_isaac_articulation"
    assert payload["joint_order_schema_version"] == JOINT_ORDER_SCHEMA_VERSION
    assert payload["mapping_digest"] == PROTOCOL_V4_MAPPING_DIGEST
    assert payload["simulator_session_id"] == "isaac-session-1"
    assert payload["stage_id"] == "stage-1"
    assert payload["captured_at_ns"] == 1_000_000_000
    assert payload["fresh_until_ns"] == (
        1_000_000_000 + backend_module.GEAR_SONIC_ISAAC_STATE_FRESHNESS_WINDOW_NS
    )
    assert payload["heartbeat_sequence"] == 4
    assert payload["source_action_sha256"] == "a" * 64
    assert payload["source_step_index"] == 3
    canonical = dict(payload)
    recorded_digest = canonical.pop("payload_sha256")
    assert (
        recorded_digest
        == hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )


def test_state_snapshot_rejects_incomplete_joint_inventory() -> None:
    names = list(PROTOCOL_V4_FULL_JOINT_ORDER[:-1])
    with pytest.raises(
        RuntimeError,
        match="persistent_isaac_protocol_v4_dof_inventory_invalid",
    ):
        backend_module.build_gear_sonic_isaac_state_snapshot(
            live_joint_names=names,
            live_joint_positions=[0.0] * len(names),
            live_joint_velocities=[0.0] * len(names),
            base_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
            base_angular_velocity_xyz=[0.0, 0.0, 0.0],
            simulator_session_id="session",
            stage_id="stage",
            heartbeat_sequence=1,
            captured_at_ns=1,
            source="live_isaac",
        )


def test_standing_initialization_maps_by_name_zeros_hands_and_persists_evidence(
    tmp_path: Path,
) -> None:
    backend = _backend(tmp_path)
    evidence = backend._initialize_official_gear_sonic_standing_pose()
    assert evidence["status"] == "passed"
    assert evidence["blockers"] == []
    assert evidence["body_joint_names"] == list(PROTOCOL_V4_BODY_JOINT_NAMES)
    assert evidence["measured_full_joint_positions"] == pytest.approx(
        backend_module.GEAR_SONIC_DEFAULT_FULL_STANDING_POSITIONS
    )
    assert evidence["measured_full_joint_positions"][29:] == pytest.approx([0.0] * 14)
    assert evidence["measured_full_joint_velocities"] == pytest.approx([0.0] * 43)
    assert evidence["projected_gravity"] == pytest.approx([0.0, 0.0, -1.0])
    assert evidence["surrogate"] is False
    assert evidence["joint_state_application"]["mode"] == (
        "state_plus_position_velocity_targets"
    )
    assert evidence["joint_state_application"]["applied_apis"] == [
        "set_joint_positions",
        "set_joint_velocities",
        "set_joint_position_targets",
        "set_joint_velocity_targets",
    ]
    assert backend.robot.world_position == pytest.approx([1.25, -0.5, 0.8])
    assert backend.robot.linear_velocity == pytest.approx([0.0, 0.0, 0.0])
    assert backend.robot.angular_velocity == pytest.approx([0.0, 0.0, 0.0])
    assert backend.app.update_count == 1
    artifact_path = tmp_path / "gear_sonic_standing_initialization.json"
    persisted = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert persisted == evidence
    assert (
        backend.standing_initialization_artifact["sha256"]
        == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    )


def test_standing_initialization_uses_live_state_setters_without_target_apis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = list(reversed(PROTOCOL_V4_FULL_JOINT_ORDER))
    robot = _LiveIsaacRobot(names)
    monkeypatch.setattr(
        backend_module,
        "_standing_articulation_action",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    backend = _backend(tmp_path, robot=robot)

    evidence = backend._initialize_official_gear_sonic_standing_pose()

    expected_indices = [names.index(name) for name in PROTOCOL_V4_FULL_JOINT_ORDER]
    application = evidence["joint_state_application"]
    assert evidence["status"] == "passed"
    assert evidence["surrogate"] is False
    assert application == {
        "mode": "state_plus_articulation_action",
        "required_state_apis": ["set_joint_positions", "set_joint_velocities"],
        "required_state_apis_available": True,
        "optional_target_apis_available": ["apply_action"],
        "applied_apis": [
            "set_joint_positions",
            "set_joint_velocities",
            "apply_action",
        ],
        "protocol_v4_articulation_dof_indices": expected_indices,
        "target_joint_count": len(PROTOCOL_V4_FULL_JOINT_ORDER),
        "standing_pose_sha256": backend_module.GEAR_SONIC_DEFAULT_STANDING_POSE_SHA256,
        "surrogate": False,
        "target_binding": {
            "api": "apply_action",
            "action_type": "isaacsim.core.utils.types.ArticulationAction",
            "joint_positions_bound": True,
            "joint_velocities_bound": True,
            "joint_indices_bound": True,
            "surrogate": False,
        },
    }
    assert robot.joint_position_state_write == pytest.approx(
        {
            index: expected
            for index, expected in zip(
                expected_indices,
                backend_module.GEAR_SONIC_DEFAULT_FULL_STANDING_POSITIONS,
                strict=True,
            )
        }
    )
    assert robot.joint_velocity_state_write == {
        index: 0.0 for index in expected_indices
    }
    assert len(robot.applied_actions) == 1
    action = robot.applied_actions[0]
    assert list(action.joint_positions) == pytest.approx(
        backend_module.GEAR_SONIC_DEFAULT_FULL_STANDING_POSITIONS
    )
    assert list(action.joint_velocities) == pytest.approx([0.0] * 43)
    assert list(action.joint_indices) == expected_indices
    assert evidence["measured_full_joint_positions"] == pytest.approx(
        backend_module.GEAR_SONIC_DEFAULT_FULL_STANDING_POSITIONS
    )
    assert evidence["measured_full_joint_velocities"] == pytest.approx([0.0] * 43)
    persisted = json.loads(
        (tmp_path / "gear_sonic_standing_initialization.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted["joint_state_application"] == application
    assert persisted["surrogate"] is False


def test_standing_initialization_fails_closed_without_real_joint_state_setters(
    tmp_path: Path,
) -> None:
    robot = _Robot(list(PROTOCOL_V4_FULL_JOINT_ORDER))
    robot.set_joint_positions = None  # type: ignore[method-assign]
    backend = _backend(tmp_path, robot=robot)

    with pytest.raises(
        RuntimeError,
        match="persistent_isaac_standing_joint_state_api_missing",
    ):
        backend._initialize_official_gear_sonic_standing_pose()

    persisted = json.loads(
        (tmp_path / "gear_sonic_standing_initialization.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted["status"] == "blocked"
    assert persisted["surrogate"] is False
    assert persisted["joint_state_application"]["mode"] == "not_applied"
    assert (
        persisted["joint_state_application"]["required_state_apis_available"]
        is False
    )


def test_standing_initialization_persists_blocked_evidence_before_raising(
    tmp_path: Path,
) -> None:
    names = list(PROTOCOL_V4_FULL_JOINT_ORDER)
    names.remove("right_wrist_yaw_joint")
    backend = _backend(tmp_path, names=names)
    with pytest.raises(
        RuntimeError,
        match="persistent_isaac_standing_initialization_failed",
    ):
        backend._initialize_official_gear_sonic_standing_pose()
    persisted = json.loads(
        (tmp_path / "gear_sonic_standing_initialization.json").read_text(encoding="utf-8")
    )
    assert persisted["status"] == "blocked"
    assert persisted["blockers"]
    assert "dof_inventory_invalid" in persisted["blockers"][0]


def test_live_state_writer_atomically_advances_initial_and_post_action_heartbeat(
    tmp_path: Path,
) -> None:
    backend = _backend(tmp_path)
    backend._initialize_official_gear_sonic_standing_pose()
    backend.live_state_snapshot_path = tmp_path / "live_state.json"
    backend._live_state_snapshot_sequence = 0
    first = backend._write_live_state_snapshot(
        source="initial_standing_pose_live_isaac_articulation",
        captured_at_ns=10,
    )
    second = backend._write_live_state_snapshot(
        source="post_action_live_isaac_articulation",
        source_action_sha256="b" * 64,
        source_step_index=2,
        captured_at_ns=20,
    )
    assert first is not None
    assert second is not None
    assert first["heartbeat_sequence"] == 1
    assert second["heartbeat_sequence"] == 2
    assert second["source_action_sha256"] == "b" * 64
    assert second["source_step_index"] == 2
    assert json.loads(backend.live_state_snapshot_path.read_text(encoding="utf-8")) == second
    assert list(tmp_path.glob(".live_state.json.*.tmp")) == []
    assert (
        backend.last_live_state_snapshot_artifact["sha256"]
        == hashlib.sha256(backend.live_state_snapshot_path.read_bytes()).hexdigest()
    )


def test_post_action_policy_state_is_bound_to_same_action_step_and_snapshot() -> None:
    source = {
        "left_leg": [0.0] * 6,
        "measurement": {"mapping_digest": "c" * 64},
    }
    snapshot = {
        "snapshot_path": "/tmp/live-state.json",
        "captured_at_ns": 123,
        "heartbeat_sequence": 7,
        "payload_sha256": "d" * 64,
        "fresh_until_ns": 999,
        "freshness_window_ns": 500,
    }
    bound = backend_module.bind_post_action_policy_state_measurement(
        source,
        simulator_session_id="session-1",
        stage_id="stage-1",
        source_action_sha256="e" * 64,
        source_step_index=5,
        captured_at_ns=123,
        state_snapshot=snapshot,
    )
    assert bound["measurement"] == {
        "mapping_digest": "c" * 64,
        "simulator_session_id": "session-1",
        "stage_id": "stage-1",
        "source": "post_action_live_isaac_articulation",
        "source_action_sha256": "e" * 64,
        "source_step_index": 5,
        "captured_at_ns": "123",
        "surrogate": False,
        "snapshot_path": "/tmp/live-state.json",
        "state_snapshot_captured_at_ns": "123",
        "state_snapshot_heartbeat_sequence": 7,
        "state_snapshot_payload_sha256": "d" * 64,
        "state_snapshot_fresh_until_ns": "999",
        "state_snapshot_freshness_window_ns": "500",
    }
    assert source["measurement"] == {"mapping_digest": "c" * 64}


def test_unconfigured_live_state_writer_keeps_legacy_hermetic_mocks_working(
    tmp_path: Path,
) -> None:
    backend = _backend(tmp_path)
    assert backend._write_live_state_snapshot(source="test") is None
