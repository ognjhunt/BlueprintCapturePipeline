from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from blueprint_pipeline import g1_microwave_live_aligned_finetune as aligned
from blueprint_pipeline.gear_sonic_joint_order_contract import (
    PROTOCOL_V4_FULL_JOINT_ORDER,
)
from blueprint_pipeline.g1_sonic_motion_token_conversion import (
    SOURCE_ACTION_JOINT_NAMES,
)


def test_canonical_joint_positions_requires_and_preserves_protocol_order() -> None:
    inventory = [
        {
            "normalized_name": name,
            "observed_name": name,
            "observed_index": index,
            "position": index / 10.0,
        }
        for index, name in reversed(list(enumerate(PROTOCOL_V4_FULL_JOINT_ORDER)))
    ]
    values = aligned._canonical_joint_positions(
        {"proprioception_mapping": {"observed_dof_inventory": inventory}}
    )
    assert values == pytest.approx(
        [index / 10.0 for index in range(len(PROTOCOL_V4_FULL_JOINT_ORDER))]
    )


def test_canonical_joint_positions_fails_closed_on_incomplete_inventory() -> None:
    with pytest.raises(ValueError, match="initial_joint_inventory_mismatch"):
        aligned._canonical_joint_positions(
            {
                "proprioception_mapping": {
                    "observed_dof_inventory": [
                        {
                            "normalized_name": PROTOCOL_V4_FULL_JOINT_ORDER[0],
                            "position": 0.0,
                        }
                    ]
                }
            }
        )


def test_numeric_stats_records_directional_distribution() -> None:
    values = np.asarray([[0.0, 4.0], [2.0, 8.0], [4.0, 12.0]])
    result = aligned._numeric_stats(values)
    assert result["mean"] == pytest.approx([2.0, 8.0])
    assert result["min"] == [0.0, 4.0]
    assert result["max"] == [4.0, 12.0]
    assert result["std"] == pytest.approx(
        [np.std(values[:, 0]), np.std(values[:, 1])]
    )


def test_live_aligned_grasp_uses_qualified_palm_down_convention() -> None:
    assert aligned.LIVE_ALIGNED_HAND_AXIS_POLARITY == -1.0
    assert aligned.LIVE_ALIGNED_GRASP_YAW_RAD == 0.0


def _passed_motion_evidence() -> dict[str, object]:
    return {
        "schema_version": "g1_microwave_live_aligned_isaac_motion_evidence.v2",
        "status": "passed",
        "physics_execution": {
            "one_physics_step_per_controller_target": True,
        },
        "render_synchronization": {
            "rendered_from_post_physics_measured_pose": True,
            "hidden_render_physics_step_absent": True,
        },
        "contact_and_door_physics": {
            "contact_report_monitor_active": True,
            "door_motion_absent_or_preceded_by_manipulator_contact": True,
        },
        "blockers": [],
    }


def test_head_render_is_finalized_before_backend_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = tmp_path / "seed"
    frames = seed / "isaac_head_frames"
    frames.mkdir(parents=True)
    stage = tmp_path / "KitchenRoom.usd"
    stage.write_bytes(b"stage")

    def fake_run(argv: list[str], **_kwargs: object) -> SimpleNamespace:
        assert any(value.endswith("frame_%06d.png") for value in argv)
        Path(argv[-1]).write_bytes(b"encoded-video")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(aligned.subprocess, "run", fake_run)
    report = aligned._finalize_isaac_head_render(
        seed=seed,
        frames_dir=frames,
        stage_path=stage,
        motion_evidence=_passed_motion_evidence(),
    )

    assert report["status"] == "exact_isaac_rigid_head_episode_rendered"
    assert report["third_person_used_for_policy"] is False
    assert report["articulation_pose_sequence_verified"] is True
    assert report["physics_replay_verified"] is True
    assert report["one_physics_step_per_target_verified"] is True
    assert report["rendered_post_physics_measured_pose_verified"] is True
    assert report["active_arm_motion_in_robot_pov_verified"] is True
    assert report["contact_report_monitor_verified"] is True
    assert report["unexpected_robot_collision_absent"] is True
    assert report["door_motion_contact_gated_if_present"] is True
    assert len(report["motion_evidence_sha256"]) == 64
    assert (seed / "ego_view.mp4").read_bytes() == b"encoded-video"
    assert (seed / "live_aligned_isaac_render_report.json").is_file()
    assert (seed / "live_aligned_isaac_motion_evidence.json").is_file()


def test_head_render_refuses_blocked_motion_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fake_run(*_args: object, **_kwargs: object) -> SimpleNamespace:
        nonlocal called
        called = True
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(aligned.subprocess, "run", fake_run)
    with pytest.raises(
        RuntimeError, match="live_aligned_isaac_motion_evidence_not_passed"
    ):
        aligned._finalize_isaac_head_render(
            seed=tmp_path,
            frames_dir=tmp_path,
            stage_path=tmp_path / "KitchenRoom.usd",
            motion_evidence={"status": "blocked", "blockers": ["frozen"]},
        )
    assert called is False


def _physics_motion_record(frame_index: int, *, moving: bool) -> dict[str, object]:
    offset = frame_index * 0.02 if moving else 0.0
    joint_position = frame_index * 0.5 if moving else 0.0
    return {
        "frame_index": frame_index,
        "target_joint_max_error_rad": 0.0,
        "active_joint_mean_tracking_error_rad": 0.0,
        "physics_step_delta": 1,
        "simulation_time_delta_seconds": 1.0 / aligned.FPS,
        "render_physics_step_delta": 0,
        "contact_report_monitor_active": True,
        "target_manipulator_contact": False,
        "unexpected_robot_collision_events": [],
        "door_open_angle_rad": 0.0,
        "door_open_angle_before_step_rad": 0.0,
        "base_position_xyz_m": [0.0, 0.0, 1.0],
        "projected_gravity": [0.0, 0.0, -1.0],
        "active_joint_positions": {
            "right_shoulder_pitch_joint": joint_position,
        },
        "active_joint_velocities_rad_s": {
            "right_shoulder_pitch_joint": 1.0,
        },
        "frame_sha256": f"frame-{frame_index}",
        "active_arm_landmarks": {
            name: {
                "world_position_xyz_m": [offset, 0.0, 1.0],
                "u_px": 320.0 + (frame_index * 8.0 if moving else 0.0),
                "v_px": 240.0,
                "in_frame": True,
            }
            for name in aligned.ISAAC_RENDER_ACTIVE_ARM_LINK_NAMES
        },
    }


def _planned_physics_motion() -> dict[str, object]:
    return {
        "passed": True,
        "active_joint_names": ["right_shoulder_pitch_joint"],
        "maximum_active_joint_span_rad": 1.0,
        "maximum_active_joint_velocity_rad_s": 2.0,
        "maximum_active_joint_acceleration_rad_s2": 4.0,
    }


def test_render_motion_summary_rejects_frozen_active_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aligned, "FRAME_COUNT", 3)
    records = [_physics_motion_record(index, moving=False) for index in range(3)]
    result = aligned._summarize_render_motion(
        records=records,
        planned_motion=_planned_physics_motion(),
    )
    assert result["status"] == "blocked"
    assert (
        "live_aligned_isaac_measured_active_joint_motion_too_low"
        in result["blockers"]
    )
    assert "live_aligned_isaac_active_arm_world_motion_too_low" in result["blockers"]
    assert "live_aligned_isaac_active_arm_pixel_motion_too_low" in result["blockers"]


def test_render_motion_summary_accepts_verified_visible_arm_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aligned, "FRAME_COUNT", 3)
    records = [_physics_motion_record(index, moving=True) for index in range(3)]
    result = aligned._summarize_render_motion(
        records=records,
        planned_motion=_planned_physics_motion(),
    )
    assert result["status"] == "passed"
    assert result["blockers"] == []
    assert result["physics_execution"]["one_physics_step_per_controller_target"] is True
    assert result["render_synchronization"]["hidden_render_physics_step_absent"] is True
    assert (
        result["active_arm_motion"]["maximum_pixel_displacement_from_first_px"]
        == pytest.approx(16.0)
    )


def test_robot_contact_classification_allows_support_and_right_manipulator_target() -> None:
    base = {
        "contact_point_count": 1,
        "actor0_prim_path": "/World/G1",
        "actor1_prim_path": "/root/Kitchen",
    }
    result = aligned._classify_robot_contact_events(
        [
            {
                **base,
                "collider0_prim_path": "/World/G1/right_ankle_roll_link",
                "collider1_prim_path": "/root/Kitchen/Floor",
            },
            {
                **base,
                "collider0_prim_path": "/World/G1/right_wrist_yaw_link",
                "collider1_prim_path": aligned.TARGET_PRIM_PATH,
            },
        ],
        robot_prim_path="/World/G1",
        target_prim_path=aligned.TARGET_PRIM_PATH,
    )
    assert len(result["allowed_support_contact_events"]) == 1
    assert len(result["target_manipulator_contact_events"]) == 1
    assert result["unexpected_robot_collision_events"] == []


def test_render_motion_summary_rejects_door_motion_without_manipulator_contact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aligned, "FRAME_COUNT", 3)
    records = [_physics_motion_record(index, moving=True) for index in range(3)]
    for index, record in enumerate(records):
        record["door_open_angle_rad"] = index * 0.02
    result = aligned._summarize_render_motion(
        records=records,
        planned_motion=_planned_physics_motion(),
    )
    assert result["status"] == "blocked"
    assert (
        "live_aligned_isaac_door_motion_without_manipulator_contact"
        in result["blockers"]
    )


def test_render_motion_summary_accepts_contact_gated_door_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aligned, "FRAME_COUNT", 3)
    records = [_physics_motion_record(index, moving=True) for index in range(3)]
    for index, record in enumerate(records):
        record["door_open_angle_rad"] = index * 0.02
    records[0]["target_manipulator_contact"] = True
    result = aligned._summarize_render_motion(
        records=records,
        planned_motion=_planned_physics_motion(),
    )
    assert result["status"] == "passed"
    assert (
        result["claim_boundary"]["door_articulation_transition_proven"] is True
    )


def test_render_isaac_steps_physics_once_and_renders_measured_pose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aligned, "FRAME_COUNT", 3)
    seed = tmp_path / "seed"
    seed.mkdir()
    trajectory = np.zeros((3, 43), dtype=np.float32)
    active_index = SOURCE_ACTION_JOINT_NAMES.index("right_shoulder_pitch_joint")
    trajectory[:, active_index] = [0.0, 0.5, 1.0]
    np.save(seed / "observation_state_43d.npy", trajectory, allow_pickle=False)
    stage = tmp_path / "KitchenRoom.usd"
    stage.write_bytes(b"stage")
    render_source = tmp_path / "rendered"
    render_source.mkdir()

    class FakeTimeline:
        playing = True

        def play(self) -> None:
            self.playing = True

        def commit(self) -> None:
            pass

        def is_playing(self) -> bool:
            return self.playing

    class FakeRobot:
        def __init__(self) -> None:
            self.positions = np.zeros(43, dtype=np.float64)
            self.velocities = np.zeros(43, dtype=np.float64)
            self.pending = self.positions.copy()

        def get_dof_index(self, name: str) -> int:
            return SOURCE_ACTION_JOINT_NAMES.index(name)

        def get_joint_positions(self) -> np.ndarray:
            return self.positions.copy()

        def get_joint_velocities(self) -> np.ndarray:
            return self.velocities.copy()

        def get_world_pose(self) -> tuple[np.ndarray, np.ndarray]:
            return np.asarray([0.0, 0.0, 1.0]), np.asarray([1.0, 0.0, 0.0, 0.0])

    class FakeRenderer:
        def __init__(self) -> None:
            self.calibrated = False

        def set_initial_robot_pov_calibration_landmarks(
            self, _landmarks: list[dict[str, object]]
        ) -> None:
            self.calibrated = True

        def render_measured_pose(
            self, *, step_index: int, target_prim_path: str
        ) -> list[dict[str, object]]:
            assert self.calibrated is True
            assert target_prim_path == aligned.TARGET_PRIM_PATH
            path = render_source / f"frame-{step_index}.png"
            path.write_bytes(f"frame-{step_index}".encode())
            return [
                {
                    "camera_role": "robot_pov",
                    "path": str(path),
                    "sha256": aligned._sha256(path),
                    "camera_contract": {"frame_index": step_index},
                }
            ]

    robot = FakeRobot()
    timeline = FakeTimeline()
    renderer = FakeRenderer()

    class FakeSimulationManager:
        steps = 7
        simulation_time = 0.14

        @classmethod
        def get_num_physics_steps(cls) -> int:
            return cls.steps

        @classmethod
        def get_simulation_time(cls) -> float:
            return cls.simulation_time

    class FakeBackend:
        def __init__(self) -> None:
            self.robot_prim_path = "/World/G1"
            self.robot = robot
            self.timeline = timeline
            self.review_renderer = renderer
            self._simulation_manager = FakeSimulationManager
            self.app = SimpleNamespace(update=self._step_physics)
            self.closed = False
            self.contact_events: list[dict[str, object]] = []

        def _step_physics(self) -> None:
            robot.positions = robot.pending.copy()
            robot.velocities.fill(1.0)
            FakeSimulationManager.steps += 1
            FakeSimulationManager.simulation_time += 1.0 / aligned.FPS

        def _apply_controller_state(self, state: dict[str, object]) -> None:
            robot.pending = np.asarray(state["joint_positions"], dtype=np.float64)

        def contact_event_cursor(self) -> int:
            return len(self.contact_events)

        def contact_events_since(self, cursor: int) -> list[dict[str, object]]:
            return list(self.contact_events[cursor:])

        def measure_revolute_task_open_angle(
            self, prim_path: str
        ) -> dict[str, object]:
            assert prim_path == aligned.TARGET_PRIM_PATH
            return {"value_rad": 0.0, "surrogate": False}

        def _live_projected_gravity(self) -> list[float]:
            return [0.0, 0.0, -1.0]

        def _live_robot_registration_link_poses(self) -> dict[str, object]:
            value = float(robot.positions[active_index])
            return {
                "landmarks": [
                    {
                        "landmark_id": name,
                        "world_position_xyz": [value * 0.02, 0.0, 1.0],
                    }
                    for name in aligned.ISAAC_RENDER_ACTIVE_ARM_LINK_NAMES
                ]
            }

        def close(self) -> None:
            self.closed = True

    backend = FakeBackend()
    monkeypatch.setattr(
        aligned,
        "_load_runtime_backend_overlay",
        lambda: SimpleNamespace(
            create_backend=lambda **_kwargs: backend,
            JOINT_ORDER_SCHEMA_VERSION="gear_sonic_joint_order.v4",
            PROTOCOL_V4_MAPPING_DIGEST="a" * 64,
        ),
    )

    def fake_project(
        *,
        camera_contract: dict[str, object],
        registration: dict[str, object],
    ) -> dict[str, dict[str, object]]:
        frame_index = int(camera_contract["frame_index"])
        rows = registration["landmarks"]
        assert isinstance(rows, list)
        return {
            name: {
                "world_position_xyz_m": [frame_index * 0.02, 0.0, 1.0],
                "u_px": 320.0 + frame_index * 8.0,
                "v_px": 240.0,
                "in_frame": True,
            }
            for name in aligned.ISAAC_RENDER_ACTIVE_ARM_LINK_NAMES
        }

    monkeypatch.setattr(aligned, "_project_active_arm_landmarks", fake_project)

    def fake_run(argv: list[str], **_kwargs: object) -> SimpleNamespace:
        Path(argv[-1]).write_bytes(b"encoded-video")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(aligned.subprocess, "run", fake_run)
    report = aligned.render_isaac(
        seed_dir=seed,
        stage_path=stage,
        g1_usd_path=tmp_path / "g1.usd",
        route_file=tmp_path / "route.json",
        evidence_dir=tmp_path / "evidence",
    )
    assert timeline.playing is True
    assert backend.closed is True
    assert report["articulation_pose_sequence_verified"] is True
    assert report["physics_replay_verified"] is True
    evidence = aligned._load_object(
        seed / "live_aligned_isaac_motion_evidence.json",
        label="motion_evidence",
    )
    assert evidence["status"] == "passed"
    assert (
        evidence["physics_execution"]["maximum_joint_target_tracking_error_rad"]
        == 0.0
    )
    assert (
        evidence["render_synchronization"]["hidden_render_physics_step_absent"]
        is True
    )
