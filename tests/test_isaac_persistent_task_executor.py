from __future__ import annotations

import json
import hashlib
import io
import math
import urllib.error
from pathlib import Path

import pytest

from blueprint_pipeline import isaac_persistent_task_completion_client as client
from blueprint_pipeline import isaac_persistent_task_executor_service as service
from blueprint_pipeline import isaac_runtime_task_backend as backend_module
from blueprint_pipeline.g1_proprioception_map import (
    G1_CANONICAL_DOF_GROUPS,
    validate_g1_sonic_state_dims,
)
from blueprint_pipeline.gear_sonic_joint_order_contract import (
    JOINT_ORDER_SCHEMA_VERSION,
    PROTOCOL_V4_FULL_JOINT_ORDER,
    PROTOCOL_V4_MAPPING_DIGEST,
)
from blueprint_pipeline.task_episode_baseline import canonical_task_contract_sha256

IsaacPersistentTaskBackend = backend_module.IsaacPersistentTaskBackend


CONTRACT = {
    "registered_criteria": [
        {
            "criterion_id": "microwave_door_open_angle",
            "observable_transition": "articulation_angle_rad",
            "articulation_prim_path": "/root/Microwave017/Microwave017_Door",
            "comparison": "increase_at_least",
            "tolerance": 0.35,
            "unit": "rad",
        }
    ]
}


def test_live_collision_probe_uses_validated_g1_footprint() -> None:
    assert backend_module.G1_LIVE_COLLISION_HALF_EXTENT_M == pytest.approx(
        (0.12, 0.23, 0.62)
    )


def _signing_key_file(tmp_path: Path) -> Path:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    pem = Ed25519PrivateKey.generate().private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    key_file = tmp_path / "signing_key.pem"
    key_file.write_bytes(pem)
    return key_file


class _RuntimeState:
    def __init__(self):
        self.task_value = 0.0
        self.robot_target = 0.0


class _App:
    def __init__(self, state):
        self.state = state
        self.update_count = 0

    def update(self):
        self.update_count += 1
        self.state.task_value += abs(self.state.robot_target) * 0.05


class _SimulationManager:
    def __init__(self, app: _App):
        self.app = app

    def get_num_physics_steps(self):
        return self.app.update_count

    def get_simulation_time(self):
        return self.app.update_count * backend_module.GEAR_SONIC_CONTROL_DT_SECONDS


class _Articulation:
    def __init__(self, state: _RuntimeState, names: list[str], *, task: bool = False):
        self.state = state
        self.dof_names = list(names)
        self.task = task

    def get_dof_index(self, name):
        try:
            return self.dof_names.index(name)
        except ValueError:
            return -1

    def get_joint_positions(self, joint_indices=None):
        values = (
            [self.state.task_value]
            if self.task
            else [0.01 * index for index in range(len(self.dof_names))]
        )
        if joint_indices is None:
            return values
        return [values[int(index)] for index in joint_indices]

    def get_joint_velocities(self):
        return [0.0] * len(self.dof_names)

    def get_world_pose(self):
        return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]

    def get_angular_velocity(self):
        return [0.0, 0.0, 0.0]


class _ReviewRenderer:
    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.follow_live_robot_calls = 0

    def render(self, *, step_index, target_prim_path):
        return [
            {
                "camera_role": "overview",
                "frame_index": step_index,
                "path": str(self.tmp_path / f"overview_{step_index:04d}.png"),
                "sha256": "a" * 64,
                "target_prim_path": target_prim_path,
            }
        ]

    def capture_current(self, *, step_index):
        rows = []
        for role in ("overview", "robot_pov"):
            path = self.tmp_path / "frames" / f"{role}_{step_index:04d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"png:{role}:{step_index}".encode())
            rows.append(
                {
                    "camera_role": role,
                    "frame_index": step_index,
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        return rows

    def follow_live_robot(self):
        self.follow_live_robot_calls += 1


def _hermetic_backend(tmp_path: Path) -> IsaacPersistentTaskBackend:
    backend = IsaacPersistentTaskBackend.__new__(IsaacPersistentTaskBackend)
    state = _RuntimeState()
    backend.app = _App(state)
    backend._simulation_manager = _SimulationManager(backend.app)
    backend.robot = _Articulation(state, _full_g1_dof_names())
    task_binding = backend_module.RevoluteTaskJointBinding(
        contracted_prim_path="/root/Microwave017/Microwave017_Door",
        joint_prim_path=("/root/Microwave017/Microwave017_Door/RevoluteJoint"),
        body0_prim_path="/root/Microwave017/Microwave017_Body",
        body1_prim_path="/root/Microwave017/Microwave017_Door",
        axis="Z",
        lower_limit_degrees=-90.0,
        upper_limit_degrees=0.0,
    )
    backend._task_joint_binding = lambda _criterion: task_binding
    backend._task_joint_sample = lambda _binding, _criterion: {
        "value_rad": state.task_value,
        "raw_signed_angle_rad": -state.task_value,
        "bounded_signed_angle_rad": -state.task_value,
        "measurement_convention": task_binding.measurement_convention,
        "pose_source": backend_module.ISAAC_PHYSX_LIVE_RIGID_BODY_POSE_SOURCE,
        "physics_joint_prim_path": task_binding.joint_prim_path,
        "joint_axis": "Z",
        "joint_lower_limit_degrees": -90.0,
        "joint_upper_limit_degrees": 0.0,
        "measurement_backend_source_sha256": (backend_module.measurement_backend_source_sha256()),
        "surrogate": False,
    }
    backend._apply_controller_state = lambda controller_state: setattr(
        state, "robot_target", float(controller_state["joint_positions"][0])
    )
    backend.evidence_dir = tmp_path
    backend.session_id = "persistent-session-1"
    backend.stage_id = "stage-1"
    backend.attempt_id = "attempt-1"
    backend.launch_nonce = "nonce-1"
    backend._live_robot_registration_link_poses = lambda: {
        "pelvis": {
            "landmark_id": "pelvis",
            "prim_path": "/World/G1/pelvis",
            "world_position_xyz": [0.0, 0.0, 0.8],
            "world_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "landmarks": [
            {
                "landmark_id": name,
                "prim_path": f"/World/G1/{name}",
                "world_position_xyz": [0.1 * index, 0.0, 1.0],
                "world_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            }
            for index, name in enumerate(
                backend_module.CONTROLLER_FK_REGISTRATION_LANDMARK_NAMES
            )
        ],
    }
    backend.measurement_backend_source_sha256 = backend_module.measurement_backend_source_sha256()
    backend.review_renderer = _ReviewRenderer(tmp_path)
    return backend


class _ClockScene:
    def __init__(self, *, dt: float = 0.02):
        self.dt = dt
        self.steps_per_second = int(1.0 / dt)
        self.set_steps_per_second_calls: list[int] = []

    def get_dt(self):
        return 1.0 / self.steps_per_second

    def get_steps_per_second(self):
        return self.steps_per_second

    def set_steps_per_second(self, value):
        self.steps_per_second = int(value)
        self.set_steps_per_second_calls.append(int(value))


class _ClockTimeline:
    def __init__(self):
        self.target_framerate = 0.0
        self.play_every_frame = True
        self.playing = False
        self.pending_target_framerate: float | None = None
        self.pending_play_every_frame: bool | None = None
        self.pending_playing: bool | None = None
        self.commit_calls = 0

    def stop(self):
        self.pending_playing = False

    def play(self):
        self.pending_playing = True

    def set_target_framerate(self, value):
        self.pending_target_framerate = float(value)

    def get_target_framerate(self):
        return self.target_framerate

    def set_play_every_frame(self, value):
        self.pending_play_every_frame = bool(value)

    def get_play_every_frame(self):
        return self.play_every_frame

    def commit(self):
        self.commit_calls += 1
        if self.pending_target_framerate is not None:
            self.target_framerate = self.pending_target_framerate
            self.pending_target_framerate = None
        if self.pending_play_every_frame is not None:
            self.play_every_frame = self.pending_play_every_frame
            self.pending_play_every_frame = None
        if self.pending_playing is not None:
            self.playing = self.pending_playing
            self.pending_playing = None


class _ClockSimulationManager:
    def __init__(
        self,
        *,
        update_dt: float = 0.02,
        scene_count: int = 1,
        setup_steps_per_second: int | None = None,
    ):
        self.update_dt = update_dt
        self.scenes = [_ClockScene() for _ in range(scene_count)]
        self.physics_steps = 0
        self.simulation_time = 0.0
        self.setup_steps_per_second = setup_steps_per_second

    def setup_simulation(self, *, dt):
        for scene in self.scenes:
            scene.dt = float(dt)
            scene.steps_per_second = (
                int(self.setup_steps_per_second)
                if self.setup_steps_per_second is not None
                else int(1.0 / float(dt))
            )

    def get_physics_scenes(self):
        return self.scenes

    def get_num_physics_steps(self):
        return self.physics_steps

    def get_simulation_time(self):
        return self.simulation_time


class _ClockRenderingManager:
    def __init__(self):
        self.dt = 0.0

    def set_dt(self, value):
        self.dt = float(value)

    def get_dt(self):
        return self.dt


class _ClockApp:
    def __init__(self, simulation_manager: _ClockSimulationManager):
        self.simulation_manager = simulation_manager

    def update(self):
        self.simulation_manager.physics_steps += 1
        self.simulation_manager.simulation_time += self.simulation_manager.update_dt


class _ClockStage:
    def __init__(self, time_codes_per_second: float = 24.0):
        self.time_codes_per_second = float(time_codes_per_second)
        self.set_calls: list[float] = []

    def SetTimeCodesPerSecond(self, value):
        self.time_codes_per_second = float(value)
        self.set_calls.append(float(value))

    def GetTimeCodesPerSecond(self):
        return self.time_codes_per_second


def test_controller_clock_configures_verified_50hz_non_fast_forward_preflight() -> None:
    timeline = _ClockTimeline()
    simulation_manager = _ClockSimulationManager()
    rendering_manager = _ClockRenderingManager()
    stage = _ClockStage(time_codes_per_second=24.0)

    evidence = backend_module.configure_and_verify_simulation_control_clock(
        stage=stage,
        timeline=timeline,
        app=_ClockApp(simulation_manager),
        simulation_manager=simulation_manager,
        rendering_manager=rendering_manager,
    )

    assert evidence["clock_readback_verified"] is True
    assert evidence["physics_dt_seconds"] == pytest.approx([0.02])
    assert evidence["physics_steps_per_second"] == pytest.approx([50.0])
    assert evidence["render_loop_dt_seconds"] == pytest.approx(0.02)
    assert evidence["time_codes_per_second"] == pytest.approx(50.0)
    assert stage.set_calls == pytest.approx([50.0])
    assert simulation_manager.scenes[0].set_steps_per_second_calls == [50]
    assert evidence["target_framerate"] == pytest.approx(50.0)
    assert evidence["play_every_frame"] is False
    assert evidence["preflight"]["physics_step_delta"] == 1
    assert evidence["preflight"]["simulation_time_delta_seconds"] == pytest.approx(
        0.02
    )
    assert timeline.playing is False
    assert timeline.commit_calls == 4


def test_controller_clock_authors_integer_physics_rate_after_dt_conversion() -> None:
    simulation_manager = _ClockSimulationManager(setup_steps_per_second=49)

    evidence = backend_module.configure_and_verify_simulation_control_clock(
        stage=_ClockStage(),
        timeline=_ClockTimeline(),
        simulation_manager=simulation_manager,
        rendering_manager=_ClockRenderingManager(),
    )

    assert evidence["physics_dt_seconds"] == pytest.approx([0.02])
    assert evidence["physics_steps_per_second"] == pytest.approx([50.0])
    assert simulation_manager.scenes[0].set_steps_per_second_calls == [50]


def test_controller_clock_mismatch_reports_exact_readback_fields() -> None:
    rendering_manager = _ClockRenderingManager()

    def ignored_set_dt(value):
        del value
        rendering_manager.dt = 0.04

    rendering_manager.set_dt = ignored_set_dt

    with pytest.raises(
        RuntimeError,
        match="persistent_isaac_controller_clock_readback_mismatch",
    ) as exc_info:
        backend_module.configure_and_verify_simulation_control_clock(
            stage=_ClockStage(),
            timeline=_ClockTimeline(),
            simulation_manager=_ClockSimulationManager(),
            rendering_manager=rendering_manager,
        )

    diagnostic = json.loads(str(exc_info.value).split(":", 1)[1])
    assert diagnostic["mismatched_fields"] == ["render_loop_dt_seconds"]
    assert diagnostic["expected"]["render_loop_dt_seconds"] == pytest.approx(0.02)
    assert diagnostic["observed"]["render_loop_dt_seconds"] == pytest.approx(0.04)


def test_controller_clock_rejects_multiple_physics_scenes() -> None:
    timeline = _ClockTimeline()
    simulation_manager = _ClockSimulationManager(scene_count=2)

    with pytest.raises(RuntimeError, match="single_physics_scene_required"):
        backend_module.configure_and_verify_simulation_control_clock(
            stage=_ClockStage(),
            timeline=timeline,
            simulation_manager=simulation_manager,
            rendering_manager=_ClockRenderingManager(),
        )


def test_controller_clock_rejects_preflight_time_drift() -> None:
    timeline = _ClockTimeline()
    simulation_manager = _ClockSimulationManager(update_dt=0.04)

    with pytest.raises(
        RuntimeError,
        match="controller_clock_preflight_failed",
    ) as exc_info:
        backend_module.configure_and_verify_simulation_control_clock(
            stage=_ClockStage(),
            timeline=timeline,
            app=_ClockApp(simulation_manager),
            simulation_manager=simulation_manager,
            rendering_manager=_ClockRenderingManager(),
        )

    diagnostic = json.loads(str(exc_info.value).split(":", 1)[1])
    assert diagnostic["expected"]["physics_step_delta"] == 1
    assert diagnostic["expected"]["simulation_time_delta_seconds"] == pytest.approx(
        0.02
    )
    assert diagnostic["observed"]["physics_step_delta"] == 1
    assert diagnostic["observed"]["simulation_time_delta_seconds"] == pytest.approx(
        0.04
    )


def test_backend_replaces_bundled_seed_with_same_session_live_robot_pov(
    tmp_path: Path,
) -> None:
    from PIL import Image

    backend = _hermetic_backend(tmp_path)
    rendered = tmp_path / "frames" / "robot_pov_0000.png"
    rendered.parent.mkdir()
    pixels = bytearray()
    for y in range(480):
        for x in range(640):
            pixels.extend(((x + y) % 256, (2 * x) % 256, (3 * y) % 256))
    Image.frombytes("RGB", (640, 480), bytes(pixels)).save(rendered)
    rendered_sha = hashlib.sha256(rendered.read_bytes()).hexdigest()
    overview = tmp_path / "frames" / "overview_0000.png"
    overview.write_bytes(rendered.read_bytes())
    camera_contract = {
        "available": True,
        "camera_path": "/World/BlueprintReview/RobotPOVCamera",
        "camera_role": "robot_pov",
        "viewpoint_mode": "robot_head_mounted_egocentric",
        "robot_mounted": True,
        "policy_observation_eligible": True,
        "mount_motion_model": "rigid_head_local_transform",
        "gaze_motion_model": "inherits_head_orientation_no_task_reaim",
        "projection_token": "perspective",
        "resolution": [640, 480],
        "camera_world_xyz_m": [0.0, 0.0, 1.7],
        "camera_xmat_row_major": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "clipping_range_m": [0.05, 50.0],
        "intrinsics": {
            "available": True,
            "fx": 168.0,
            "fy": 168.0,
            "cx": 320.0,
            "cy": 240.0,
            "image_width": 640,
            "image_height": 480,
        },
    }

    class Renderer:
        def render(self, *, step_index, target_prim_path):
            assert step_index == 0
            assert target_prim_path == "/root/Microwave017/Microwave017_Door"
            return [
                {
                    "camera_role": "overview",
                    "frame_index": 0,
                    "path": str(overview),
                    "sha256": rendered_sha,
                    "width": 640,
                    "height": 480,
                },
                {
                    "camera_role": "robot_pov",
                    "frame_index": 0,
                    "path": str(rendered),
                    "sha256": rendered_sha,
                    "width": 640,
                    "height": 480,
                    "camera_contract": camera_contract,
                    "visual_signal": {
                        "status": "completed",
                        "non_uniform": True,
                        "rgb_channel_stddev": [73.0, 73.0, 73.0],
                    },
                }
            ]

    backend.review_renderer = Renderer()
    initial_frame = tmp_path / "initial_policy_frame.png"
    initial_frame.write_bytes(b"known-failed-bundled-seed")
    projection_context = tmp_path / "controller_fk_camera_projection_context.json"

    evidence = backend.capture_initial_policy_observation(
        target_prim_path="/root/Microwave017/Microwave017_Door",
        frame_output_path=initial_frame,
        projection_context_output_path=projection_context,
    )

    assert initial_frame.read_bytes() == rendered.read_bytes()
    assert evidence["status"] == "completed"
    context = json.loads(projection_context.read_text(encoding="utf-8"))
    assert context["schema_version"] == (
        backend_module.CONTROLLER_FK_CAMERA_PROJECTION_SCHEMA_VERSION
    )
    assert context["simulator_session_id"] == backend.session_id
    assert context["stage_id"] == backend.stage_id
    assert context["source_frame_artifact"]["sha256"] == rendered_sha
    assert context["camera_contract"] == camera_contract
    assert context["live_isaac_pelvis_world_pose"]["position_xyz"] == [0.0, 0.0, 0.8]
    assert context["live_isaac_pelvis_world_pose"]["quaternion_wxyz"] == pytest.approx(
        [1.0, 0.0, 0.0, 0.0]
    )
    registration = context["standing_cross_simulator_registration"]
    assert registration["standing_joint_names"] == list(
        backend_module.PROTOCOL_V4_FULL_JOINT_ORDER
    )
    assert registration["status"] == (
        "pending_official_mujoco_named_link_residual_verification"
    )
    assert registration["camera_projection_validation"]["status"] == "captured"
    assert (
        registration["camera_projection_validation"][
            "all_required_landmarks_in_frame"
        ]
        is True
    )
    assert set(registration["camera_projection_validation"]["projections"]) == set(
        backend_module.CONTROLLER_FK_REGISTRATION_LANDMARK_NAMES
    )
    assert context["claim_boundary"]["bundled_seed_frame_reused"] is False


def test_initial_observation_rejects_head_pov_with_active_forearm_out_of_frame(
    tmp_path: Path,
) -> None:
    from PIL import Image

    backend = _hermetic_backend(tmp_path)
    rendered = tmp_path / "robot_pov_0000.png"
    overview = tmp_path / "overview_0000.png"
    Image.new("RGB", (640, 480), (40, 80, 120)).save(rendered)
    Image.new("RGB", (640, 480), (40, 80, 120)).save(overview)
    rendered_sha = hashlib.sha256(rendered.read_bytes()).hexdigest()

    class Renderer:
        def render(self, *, step_index, target_prim_path):
            del target_prim_path
            return [
                {
                    "camera_role": "overview",
                    "frame_index": step_index,
                    "path": str(overview),
                    "sha256": rendered_sha,
                    "width": 640,
                    "height": 480,
                },
                {
                    "camera_role": "robot_pov",
                    "frame_index": step_index,
                    "path": str(rendered),
                    "sha256": rendered_sha,
                    "width": 640,
                    "height": 480,
                    "camera_contract": {
                        "available": True,
                        "camera_role": "robot_pov",
                        "viewpoint_mode": "robot_head_mounted_egocentric",
                        "robot_mounted": True,
                        "policy_observation_eligible": True,
                        "mount_motion_model": "rigid_head_local_transform",
                        "gaze_motion_model": "inherits_head_orientation_no_task_reaim",
                        "projection_token": "perspective",
                        "resolution": [640, 480],
                        "camera_world_xyz_m": [0.0, 0.0, 1.7],
                        # Forward points upward, away from every registration
                        # point in the hermetic standing state.
                        "camera_xmat_row_major": [
                            [1.0, 0.0, 0.0],
                            [0.0, -1.0, 0.0],
                            [0.0, 0.0, -1.0],
                        ],
                        "clipping_range_m": [0.05, 50.0],
                        "intrinsics": {
                            "fx": 168.0,
                            "fy": 168.0,
                            "cx": 320.0,
                            "cy": 240.0,
                            "image_width": 640,
                            "image_height": 480,
                        },
                    },
                    "visual_signal": {
                        "status": "completed",
                        "non_uniform": True,
                    },
                }
            ]

    backend.review_renderer = Renderer()
    projection_context = tmp_path / "projection_context.json"
    with pytest.raises(
        RuntimeError,
        match="persistent_isaac_initial_robot_pov_active_forearm_not_in_frame",
    ):
        backend.capture_initial_policy_observation(
            target_prim_path="/root/Microwave017/Microwave017_Door",
            frame_output_path=tmp_path / "initial_policy_frame.png",
            projection_context_output_path=projection_context,
        )

    diagnostic = json.loads(projection_context.read_text(encoding="utf-8"))
    validation = diagnostic["standing_cross_simulator_registration"][
        "camera_projection_validation"
    ]
    assert validation["active_forearm_visibility_passed"] is False
    assert validation["missing_active_arm_link_names"] == [
        "right_elbow_link",
        "right_wrist_yaw_link",
    ]


def test_initial_observation_baseline_guard_rejects_render_induced_task_drift(
    tmp_path: Path,
) -> None:
    backend = _hermetic_backend(tmp_path)
    backend.episode_baseline = {"episode_initial_value": 0.0}

    passed = backend.verify_initial_observation_preserved_episode_baseline(
        task_success_contract=CONTRACT
    )
    assert passed["status"] == "passed"

    backend.app.state.task_value = 0.01
    with pytest.raises(
        RuntimeError,
        match="persistent_isaac_initial_observation_changed_episode_baseline",
    ):
        backend.verify_initial_observation_preserved_episode_baseline(
            task_success_contract=CONTRACT
        )
    blocked = json.loads(
        (tmp_path / "initial_policy_observation_baseline_guard.json").read_text(
            encoding="utf-8"
        )
    )
    assert blocked["status"] == "blocked"
    assert blocked["blockers"] == [
        "initial_policy_observation_render_changed_episode_baseline"
    ]


def _exact_microwave_stage(tmp_path: Path):
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/root")
    microwave = UsdGeom.Xform.Define(stage, "/root/Microwave017")
    del root, microwave
    body = UsdGeom.Xform.Define(stage, "/root/Microwave017/Microwave017_Body")
    door = UsdGeom.Xform.Define(stage, "/root/Microwave017/Microwave017_Door")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(door.GetPrim())
    joint = UsdPhysics.RevoluteJoint.Define(
        stage,
        "/root/Microwave017/Microwave017_Door/RevoluteJoint",
    )
    joint.CreateBody0Rel().SetTargets([body.GetPath()])
    joint.CreateBody1Rel().SetTargets([door.GetPath()])
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(-90.0)
    joint.CreateUpperLimitAttr(0.0)
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
    assert not any(prim.HasAPI(UsdPhysics.ArticulationRootAPI) for prim in stage.Traverse())
    return stage, door


def test_resolves_exact_microwave_body_to_child_joint_without_articulation_root(
    tmp_path: Path,
) -> None:
    stage, _ = _exact_microwave_stage(tmp_path)
    binding = backend_module.resolve_revolute_task_joint(
        stage,
        contracted_prim_path="/root/Microwave017/Microwave017_Door",
    )
    assert binding.contracted_prim_path == ("/root/Microwave017/Microwave017_Door")
    assert binding.joint_prim_path == ("/root/Microwave017/Microwave017_Door/RevoluteJoint")
    assert binding.body0_prim_path == "/root/Microwave017/Microwave017_Body"
    assert binding.body1_prim_path == "/root/Microwave017/Microwave017_Door"
    assert binding.axis == "Z"
    assert binding.lower_limit_degrees == pytest.approx(-90.0)
    assert binding.upper_limit_degrees == pytest.approx(0.0)


def test_measures_exact_microwave_hinge_from_live_rigid_body_transforms(
    tmp_path: Path,
) -> None:
    from pxr import Gf

    stage, door = _exact_microwave_stage(tmp_path)
    binding = backend_module.resolve_revolute_task_joint(
        stage,
        contracted_prim_path="/root/Microwave017/Microwave017_Door",
    )
    door.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), -30.0).GetQuat()))
    signed = backend_module.measure_revolute_joint_signed_angle_rad(stage, binding)
    assert signed == pytest.approx(-math.pi / 6.0)

    backend = IsaacPersistentTaskBackend.__new__(IsaacPersistentTaskBackend)
    backend.stage = stage
    from pxr import Gf, Usd, UsdGeom

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    class InvalidatedSingleRigidPrim:
        def get_current_dynamic_state(self):
            raise RuntimeError(
                "Simulation view object is invalidated and cannot be used again "
                "to call getVelocities"
            )

    def live_physx_transform(prim_path: str) -> dict[str, object]:
        quat = (
            Gf.Transform(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(prim_path)))
            .GetRotation()
            .GetQuat()
        )
        translation = cache.GetLocalToWorldTransform(
            stage.GetPrimAtPath(prim_path)
        ).ExtractTranslation()
        imaginary = [float(item) for item in quat.GetImaginary()]
        return {
            "ret_val": True,
            "position": [float(item) for item in translation],
            # IPhysX publishes quaternions in XYZW order.
            "rotation": [*imaginary, float(quat.GetReal())],
        }

    # Reproduce attempt 5's stale deprecated view. The Isaac 6 path must not
    # touch it; direct IPhysX lookup is the authoritative lifecycle-safe read.
    backend._task_rigid_bodies = {
        binding.body0_prim_path: InvalidatedSingleRigidPrim(),
        binding.body1_prim_path: InvalidatedSingleRigidPrim(),
    }
    backend._physx_rigid_body_transform_reader = live_physx_transform
    sample = backend._task_joint_sample(binding, CONTRACT["registered_criteria"][0])
    assert sample["value_rad"] == pytest.approx(math.pi / 6.0)
    assert sample["raw_signed_angle_rad"] == pytest.approx(-math.pi / 6.0)
    assert sample["bounded_signed_angle_rad"] == pytest.approx(-math.pi / 6.0)
    assert sample["measurement_convention"] == ("upper_limit_minus_signed_angle_radians")
    assert sample["pose_source"] == (
        backend_module.ISAAC_PHYSX_LIVE_RIGID_BODY_POSE_SOURCE
    )
    assert sample["surrogate"] is False


def _live_registration_stage():
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/G1")
    for name in (
        "pelvis",
        *backend_module.CONTROLLER_FK_REGISTRATION_LANDMARK_NAMES,
    ):
        prim = UsdGeom.Xform.Define(stage, f"/World/G1/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
    return stage


def test_registration_links_use_live_physx_pose_after_deprecated_view_invalidation() -> None:
    backend = IsaacPersistentTaskBackend.__new__(IsaacPersistentTaskBackend)
    backend.stage = _live_registration_stage()
    backend.robot_prim_path = "/World/G1"

    class InvalidatedSingleRigidPrim:
        calls = 0

        def get_current_dynamic_state(self):
            self.calls += 1
            raise RuntimeError(
                "Simulation view object is invalidated and cannot be used again "
                "to call getVelocities"
            )

    stale_view = InvalidatedSingleRigidPrim()
    backend._task_rigid_bodies = {
        f"/World/G1/{name}": stale_view
        for name in (
            "pelvis",
            *backend_module.CONTROLLER_FK_REGISTRATION_LANDMARK_NAMES,
        )
    }
    calls: list[str] = []

    def read_current_physx_pose(path: str) -> dict[str, object]:
        calls.append(path)
        index = len(calls)
        return {
            "ret_val": True,
            "position": [float(index), 0.0, 0.8],
            "rotation": [0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)],
        }

    backend._physx_rigid_body_transform_reader = read_current_physx_pose

    registration = backend._live_robot_registration_link_poses()

    assert stale_view.calls == 0
    assert calls[0] == "/World/G1/pelvis"
    assert len(calls) == 1 + len(
        backend_module.CONTROLLER_FK_REGISTRATION_LANDMARK_NAMES
    )
    assert registration["pelvis"]["world_position_xyz"] == [1.0, 0.0, 0.8]
    assert registration["pelvis"]["world_quaternion_wxyz"] == pytest.approx(
        [math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)]
    )
    assert registration["pelvis"]["pose_source"] == (
        backend_module.ISAAC_PHYSX_LIVE_RIGID_BODY_POSE_SOURCE
    )
    assert registration["pelvis"]["surrogate"] is False
    assert all(
        row["pose_source"]
        == backend_module.ISAAC_PHYSX_LIVE_RIGID_BODY_POSE_SOURCE
        and row["surrogate"] is False
        for row in registration["landmarks"]
    )


def test_registration_links_fail_closed_when_live_physx_pose_is_unavailable() -> None:
    backend = IsaacPersistentTaskBackend.__new__(IsaacPersistentTaskBackend)
    backend.stage = _live_registration_stage()
    backend.robot_prim_path = "/World/G1"
    backend._physx_rigid_body_transform_reader = lambda _path: {
        "ret_val": False,
        # Authored/static values must not be accepted after the live lookup fails.
        "position": [0.0, 0.0, 0.8],
        "rotation": [0.0, 0.0, 0.0, 1.0],
    }

    with pytest.raises(
        RuntimeError,
        match="persistent_isaac_projection_link_dynamic_state_unavailable:pelvis",
    ):
        backend._live_robot_registration_link_poses()


def test_microwave_hinge_measurement_canonicalizes_equivalent_quaternion_signs(
    tmp_path: Path,
) -> None:
    stage, _ = _exact_microwave_stage(tmp_path)
    binding = backend_module.resolve_revolute_task_joint(
        stage,
        contracted_prim_path="/root/Microwave017/Microwave017_Door",
    )
    half_angle = math.radians(-30.0) / 2.0
    door_rotation = (
        math.cos(half_angle),
        0.0,
        0.0,
        math.sin(half_angle),
    )

    def read_with_door_sign(sign: float):
        return lambda path: (
            tuple(sign * item for item in door_rotation)
            if path == binding.body1_prim_path
            else (1.0, 0.0, 0.0, 0.0)
        )

    positive_sign = backend_module.measure_revolute_joint_signed_angle_rad(
        stage,
        binding,
        body_rotation_reader=read_with_door_sign(1.0),
    )
    negative_sign = backend_module.measure_revolute_joint_signed_angle_rad(
        stage,
        binding,
        body_rotation_reader=read_with_door_sign(-1.0),
    )
    assert positive_sign == pytest.approx(-math.pi / 6.0)
    assert negative_sign == pytest.approx(positive_sign)


def test_microwave_joint_resolution_rejects_ambiguous_child_hinges(
    tmp_path: Path,
) -> None:
    from pxr import UsdPhysics

    stage, door = _exact_microwave_stage(tmp_path)
    duplicate = UsdPhysics.RevoluteJoint.Define(
        stage,
        "/root/Microwave017/Microwave017_Door/SecondRevoluteJoint",
    )
    duplicate.CreateBody0Rel().SetTargets(["/root/Microwave017/Microwave017_Body"])
    duplicate.CreateBody1Rel().SetTargets([door.GetPath()])
    duplicate.CreateAxisAttr("Z")
    duplicate.CreateLowerLimitAttr(-90.0)
    duplicate.CreateUpperLimitAttr(0.0)
    with pytest.raises(
        RuntimeError,
        match="persistent_isaac_task_revolute_joint_resolution_not_unique",
    ):
        backend_module.resolve_revolute_task_joint(
            stage,
            contracted_prim_path="/root/Microwave017/Microwave017_Door",
        )


def test_physx_overlap_hit_path_supports_legacy_mapping_and_prefers_rigid_body() -> None:
    hit = {
        "rigid_body": "/World/G1",
        "collision": "/root/Microwave017/Microwave017_Door/Collider",
    }
    assert backend_module._physx_overlap_hit_prim_path(hit) == "/World/G1"


def test_physx_overlap_hit_path_supports_isaac6_attribute_object() -> None:
    class OverlapHit:
        rigid_body = ""
        collision = "/root/Microwave017/Microwave017_Door/Collider"

    assert backend_module._physx_overlap_hit_prim_path(OverlapHit()) == (
        "/root/Microwave017/Microwave017_Door/Collider"
    )


@pytest.mark.parametrize(
    "hit",
    ({}, {"rigid_body": "not-an-absolute-prim-path"}),
)
def test_physx_overlap_hit_path_rejects_unfilterable_hits(hit: object) -> None:
    with pytest.raises(
        RuntimeError,
        match=r"persistent_isaac_overlap_hit_prim_path_(?:missing|invalid)",
    ):
        backend_module._physx_overlap_hit_prim_path(hit)


def _request(step: int, value: float, contract=CONTRACT) -> dict:
    action = {"action_chunk": [value]}
    action_sha = hashlib.sha256(
        json.dumps(action, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "step_index": step,
        "action": action,
        "wam_output": {
            "generated_robot_state": {
                "source_action_sha256": action_sha,
                "joint_names": list(PROTOCOL_V4_FULL_JOINT_ORDER),
                "joint_positions": [value] * len(PROTOCOL_V4_FULL_JOINT_ORDER),
                "joint_order_schema_version": JOINT_ORDER_SCHEMA_VERSION,
                "mapping_digest": PROTOCOL_V4_MAPPING_DIGEST,
            }
        },
        "task_success_contract": contract,
        "physics_steps_per_action": 4,
    }


def _sequence_request(step: int, values: list[float], contract=CONTRACT) -> dict:
    def canonical(value: object) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()

    action_frames = [[value] * 78 for value in values]
    action_frames_sha256 = canonical(action_frames)
    action = {
        "action_chunk": [values[0]],
        "controller_action": {
            "schema_version": "gear_sonic_controller_action_sequence.v1",
            "execution_mode": "bounded_model_horizon_prefix",
            "execution_frame_count": len(action_frames),
            "source_horizon_frame_count": len(action_frames),
            "frame_dimension": 78,
            "control_hz": 50.0,
            "sample_period_seconds": 0.02,
            "execution_duration_seconds": len(action_frames) / 50.0,
            "frames": action_frames,
            "frames_sha256": action_frames_sha256,
            "source_frames_sha256": action_frames_sha256,
        },
    }
    sequence = [
        {
            "horizon_frame_index": index,
            "controller_frame_index": 100 + index,
            "source_action_frame_sha256": canonical(action_frames[index]),
            "controller_state_sha256": canonical(
                {"frame_index": index, "joint_target": value}
            ),
            "command_send_offset_seconds": index / 50.0,
            "joint_positions": [value] * len(PROTOCOL_V4_FULL_JOINT_ORDER),
            "joint_names": list(PROTOCOL_V4_FULL_JOINT_ORDER),
            "applied_dof_mapping": [
                {"joint_name": name, "protocol_v4_index": mapping_index}
                for mapping_index, name in enumerate(PROTOCOL_V4_FULL_JOINT_ORDER)
            ],
            "landmarks": [{"landmark_id": "left_wrist"}],
            "proprioceptive_state": {"official_controller_protocol": 4},
            "state_timestamp": 1000 + index,
        }
        for index, value in enumerate(values)
    ]
    sequence_sha256 = canonical(sequence)
    execution_contract = {
        "schema_version": "gear_sonic_controller_horizon_execution.v1",
        "execution_mode": "bounded_model_horizon_prefix",
        "controller_session_count": 1,
        "execution_frame_count": len(sequence),
        "source_horizon_frame_count": len(sequence),
        "frame_dimension": 78,
        "control_hz": 50.0,
        "sample_period_seconds": 0.02,
        "declared_execution_duration_seconds": len(sequence) / 50.0,
        "input_action_frames_sha256": action_frames_sha256,
        "source_action_frames_sha256": action_frames_sha256,
        "controller_state_sequence_sha256": canonical(
            [row["controller_state_sha256"] for row in sequence]
        ),
        "controller_fk_sequence_sha256": sequence_sha256,
        "final_controller_fk_frame_sha256": canonical(sequence[-1]),
    }
    action_sha = canonical(action)
    return {
        "step_index": step,
        "action": action,
        "wam_output": {
            "generated_robot_state": {
                "source_action_sha256": action_sha,
                "joint_names": list(PROTOCOL_V4_FULL_JOINT_ORDER),
                "joint_positions": [values[-1]]
                * len(PROTOCOL_V4_FULL_JOINT_ORDER),
                "joint_order_schema_version": JOINT_ORDER_SCHEMA_VERSION,
                "mapping_digest": PROTOCOL_V4_MAPPING_DIGEST,
                "executed_control_frame_count": len(sequence),
                "controller_fk_sequence": sequence,
                "controller_fk_sequence_sha256": sequence_sha256,
                "controller_execution_contract": execution_contract,
            }
        },
        "task_success_contract": contract,
    }


def test_completion_client_posts_attempt_bound_request(monkeypatch, tmp_path: Path):
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "result.json"
    action = {"generated_robot_state": {"joint_positions": [0.1]}}
    source_action_sha256 = hashlib.sha256(
        json.dumps(action, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    request_payload = {
        "schema_version": "oscar_task_completion_evaluator_request.v1",
        "step_index": 3,
        "source_action_sha256": source_action_sha256,
        "action": action,
        "task_success_contract": {"criterion_id": "door-open"},
    }
    request_path.write_text(json.dumps(request_payload))

    result_payload = {
        "status": "completed",
        "passed": True,
        "simulator_session_id": "session-1",
        "stage_id": "stage-1",
        "runtime_result_id": "result-1",
        "source_action_sha256": source_action_sha256,
        "articulation_prim_path": "/World/G1",
        "before_timestamp": "2026-07-10T00:00:00Z",
        "after_timestamp": "2026-07-10T00:00:01Z",
        "before_value": 0.0,
        "after_value": 0.4,
        "unit": "radian",
        "criterion_id": "door-open",
        "observable_transition": "door_joint_increases",
        "evaluator_attestation": {"verification_status": "verified"},
        "persistent_simulator_state_applied": True,
        "official_controller_action_applied": True,
        "post_action_policy_state": {
            **{
                field: [0.1] * dimension
                for field, dimension in client.UNITREE_G1_SONIC_STATE_DIMS.items()
            },
            "measurement": {
                "simulator_session_id": "session-1",
                "stage_id": "stage-1",
                "source": client.POST_ACTION_POLICY_STATE_SOURCE,
                "surrogate": False,
                "source_action_sha256": source_action_sha256,
                "source_step_index": 3,
                "captured_at_ns": "1000003",
            },
        },
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(result_payload).encode()

    def fake_open(request, timeout, policy):
        del policy
        assert request.full_url == "http://127.0.0.1:8765/apply-and-measure"
        assert timeout == 120
        assert json.loads(request.data) == request_payload
        return Response()

    monkeypatch.setattr(client.safe_outbound_http, "_open_with_policy", fake_open)
    monkeypatch.setenv("BLUEPRINT_TASK_COMPLETION_INPUT", str(request_path))
    monkeypatch.setenv("BLUEPRINT_TASK_COMPLETION_OUTPUT", str(output_path))

    assert client.main() == 0
    assert json.loads(output_path.read_text()) == result_payload


def test_completion_client_preserves_typed_loopback_service_error(monkeypatch) -> None:
    request_payload = {
        "schema_version": "oscar_task_completion_evaluator_request.v1",
        "step_index": 4,
        "action": {"controller_action": [0.1]},
        "task_success_contract": {"criterion_id": "door-open"},
    }
    body = json.dumps(
        {
            "status": "blocked",
            "error_type": "RuntimeError",
            "error": "review_renderer_robot_head_mount_offset_invalid",
        }
    ).encode()

    def fail_open(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            client.DEFAULT_EXECUTOR_URL,
            500,
            "Internal Server Error",
            hdrs=None,
            fp=io.BytesIO(body),
        )

    monkeypatch.setattr(client.safe_outbound_http, "open_request", fail_open)
    with pytest.raises(
        RuntimeError,
        match=(
            "persistent_isaac_task_executor_http_500:RuntimeError:"
            "review_renderer_robot_head_mount_offset_invalid"
        ),
    ):
        client.call_persistent_executor(
            request_payload,
            executor_url=client.DEFAULT_EXECUTOR_URL,
        )


@pytest.mark.parametrize(
    ("path", "bad_value", "expected_error"),
    (
        (("left_leg",), [0.0] * 5, "left_leg_dimension_5_expected_6"),
        (("right_arm",), [0.0] * 6 + [float("nan")], "right_arm_nonfinite"),
        (("measurement", "surrogate"), True, "surrogate_not_false"),
        (
            ("measurement", "simulator_session_id"),
            "other-session",
            "simulator_session_id_mismatch",
        ),
        (("measurement", "stage_id"), "other-stage", "stage_id_mismatch"),
        (
            ("measurement", "source_action_sha256"),
            "b" * 64,
            "source_action_sha256_mismatch",
        ),
        (("measurement", "source_step_index"), 2, "source_step_index_mismatch"),
        (("measurement", "captured_at_ns"), "", "captured_at_ns_invalid"),
    ),
)
def test_completion_client_rejects_malformed_or_unbound_post_action_state(
    path: tuple[str, ...], bad_value: object, expected_error: str
) -> None:
    action_sha256 = "a" * 64
    state = {
        **{
            field: [0.0] * dimension
            for field, dimension in client.UNITREE_G1_SONIC_STATE_DIMS.items()
        },
        "measurement": {
            "simulator_session_id": "session-1",
            "stage_id": "stage-1",
            "source": client.POST_ACTION_POLICY_STATE_SOURCE,
            "surrogate": False,
            "source_action_sha256": action_sha256,
            "source_step_index": 3,
            "captured_at_ns": "1000003",
        },
    }
    target = state
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = bad_value

    with pytest.raises(RuntimeError, match=expected_error):
        client._validate_post_action_policy_state(
            state,
            simulator_session_id="session-1",
            stage_id="stage-1",
            source_action_sha256=action_sha256,
            source_step_index=3,
        )


def test_backend_applies_two_actions_to_one_persistent_stage_and_measures_transition(
    tmp_path: Path,
) -> None:
    backend = _hermetic_backend(tmp_path)
    baseline = backend.capture_episode_baseline(
        task_success_contract=CONTRACT,
        attempt_id="run-1-attempt-000001",
        launch_nonce="nonce-1",
    )
    backend.install_episode_baseline_attestation({"signature_verified": True})
    assert baseline["episode_initial_value"] == pytest.approx(0.0)
    assert baseline["measurement_backend_source_sha256"] == (
        backend_module.measurement_backend_source_sha256()
    )
    assert baseline["initial_physics_measurement"]["pose_source"] == (
        backend_module.ISAAC_PHYSX_LIVE_RIGID_BODY_POSE_SOURCE
    )
    assert (tmp_path / "task_episode_baseline.json").is_file()

    first = backend.apply_and_measure(_request(1, 0.5))
    second = backend.apply_and_measure(_request(2, 1.0))
    assert first["simulator_session_id"] == second["simulator_session_id"]
    assert first["stage_id"] == second["stage_id"]
    assert first["runtime_result_id"] != second["runtime_result_id"]
    assert second["before_value"] == pytest.approx(first["after_value"])
    assert second["after_value"] > second["before_value"]
    assert first["source_action_sha256"] != second["source_action_sha256"]
    assert Path(first["evidence_artifacts"][0]["path"]).is_file()
    assert first["review_frames"][0]["camera_role"] == "overview"

    for result in (first, second):
        assert result["episode_initial_value"] == pytest.approx(baseline["episode_initial_value"])
        assert result["step_before"] == pytest.approx(result["before_value"])
        assert result["step_after"] == pytest.approx(result["after_value"])
        assert result["step_delta"] == pytest.approx(result["after_value"] - result["before_value"])
        assert result["episode_delta"] == pytest.approx(
            result["after_value"] - baseline["episode_initial_value"]
        )
        assert result["episode_baseline_digest"] == baseline["baseline_digest"]
        assert result["measurement_backend_source_sha256"] == (
            backend_module.measurement_backend_source_sha256()
        )
        assert result["pose_source"] == (
            backend_module.ISAAC_PHYSX_LIVE_RIGID_BODY_POSE_SOURCE
        )
        assert result["physics_measurement_surrogate"] is False
    persisted = json.loads(
        Path(second["evidence_artifacts"][0]["path"]).read_text(encoding="utf-8")
    )
    assert persisted["episode_initial_value"] == pytest.approx(baseline["episode_initial_value"])
    assert persisted["episode_delta"] == pytest.approx(second["episode_delta"])


def test_backend_applies_signed_controller_horizon_in_order_and_stops_on_success(
    tmp_path: Path,
) -> None:
    backend = _hermetic_backend(tmp_path)
    backend.capture_episode_baseline(
        task_success_contract=CONTRACT,
        attempt_id="run-1-attempt-000001",
        launch_nonce="nonce-1",
    )
    backend.install_episode_baseline_attestation({"signature_verified": True})

    result = backend.apply_and_measure(_sequence_request(1, [1.0, 2.0, 4.0, 10.0]))

    assert result["controller_horizon_requested_frame_count"] == 4
    assert result["controller_horizon_executed_frame_count"] == 3
    assert result["controller_horizon_fully_executed"] is False
    assert result["controller_horizon_terminated_on_semantic_success"] is True
    assert [
        row["horizon_frame_index"] for row in result["controller_frame_measurements"]
    ] == [0, 1, 2]
    assert result["controller_frame_measurements"][-1][
        "registered_transition_passed"
    ] is True
    assert backend.robot.state.robot_target == pytest.approx(4.0)
    assert result["after_value"] == pytest.approx(0.35)
    assert result["simulation_control_hz"] == pytest.approx(50.0)
    assert result["one_physics_update_per_controller_frame"] is True
    assert backend.review_renderer.follow_live_robot_calls == 3
    assert len(result["evidence_artifacts"]) == 1
    assert Path(result["evidence_artifacts"][0]["path"]).suffix == ".json"
    assert result["review_frames"]
    assert result["review_media_artifacts"] == result["review_frames"]
    assert all(
        Path(row["path"]).suffix == ".png" for row in result["review_media_artifacts"]
    )


def test_backend_executes_complete_controller_horizon_when_task_not_yet_successful(
    tmp_path: Path,
) -> None:
    backend = _hermetic_backend(tmp_path)
    backend.capture_episode_baseline(
        task_success_contract=CONTRACT,
        attempt_id="run-1-attempt-000001",
        launch_nonce="nonce-1",
    )
    backend.install_episode_baseline_attestation({"signature_verified": True})

    result = backend.apply_and_measure(_sequence_request(1, [0.1, 0.2, 0.3]))

    assert result["controller_horizon_requested_frame_count"] == 3
    assert result["controller_horizon_executed_frame_count"] == 3
    assert result["controller_horizon_fully_executed"] is True
    assert result["controller_horizon_terminated_on_semantic_success"] is False
    assert backend.robot.state.robot_target == pytest.approx(0.3)


def test_backend_rejects_tampered_controller_horizon_before_physx_mutation(
    tmp_path: Path,
) -> None:
    backend = _hermetic_backend(tmp_path)
    backend.capture_episode_baseline(
        task_success_contract=CONTRACT,
        attempt_id="run-1-attempt-000001",
        launch_nonce="nonce-1",
    )
    backend.install_episode_baseline_attestation({"signature_verified": True})
    request = _sequence_request(1, [0.1, 0.2, 0.3])
    request["wam_output"]["generated_robot_state"]["controller_fk_sequence"][1][
        "joint_positions"
    ][0] = 99.0

    with pytest.raises(
        RuntimeError,
        match="persistent_isaac_controller_fk_sequence_sha256_mismatch",
    ):
        backend.apply_and_measure(request)

    assert backend.robot.state.robot_target == pytest.approx(0.0)


def test_backend_rebinds_post_action_snapshot_after_renderer_heartbeats(
    tmp_path: Path,
) -> None:
    backend = _hermetic_backend(tmp_path)
    backend.live_state_snapshot_path = tmp_path / "live_state.json"
    backend._live_state_snapshot_sequence = 0
    runtime_state = backend.robot.state

    def live_positions(joint_indices=None):
        values = [runtime_state.robot_target] * len(backend.robot.dof_names)
        if joint_indices is None:
            return values
        return [values[int(index)] for index in joint_indices]

    backend.robot.get_joint_positions = live_positions

    class Renderer(_ReviewRenderer):
        frames_dir = tmp_path / "frames"

        def capture_current(self, *, step_index):
            backend._refresh_live_state_if_configured()
            runtime_state.robot_target = 0.75
            backend._refresh_live_state_if_configured()
            return super().capture_current(step_index=step_index)

    backend.review_renderer = Renderer(tmp_path)
    backend.capture_episode_baseline(
        task_success_contract=CONTRACT,
        attempt_id="run-1-attempt-000001",
        launch_nonce="nonce-1",
    )
    backend.install_episode_baseline_attestation({"signature_verified": True})

    request = _request(4, 0.5)
    result = backend.apply_and_measure(request)
    persisted_snapshot = json.loads(backend.live_state_snapshot_path.read_text(encoding="utf-8"))
    action_sha256 = hashlib.sha256(
        json.dumps(
            request["action"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    assert persisted_snapshot["capture_reason"] == ("post_action_live_isaac_articulation")
    assert persisted_snapshot["source_action_sha256"] == action_sha256
    assert persisted_snapshot["source_step_index"] == 4
    assert persisted_snapshot["body_q"] == pytest.approx([0.75] * 29)
    measurement = result["post_action_policy_state"]["measurement"]
    assert measurement["source_action_sha256"] == action_sha256
    assert measurement["source_step_index"] == 4
    assert measurement["state_snapshot_payload_sha256"] == (persisted_snapshot["payload_sha256"])
    assert result["post_action_policy_state"]["left_leg"] == pytest.approx([0.75] * 6)


def test_backend_blocks_apply_without_episode_baseline(tmp_path: Path):
    backend = _hermetic_backend(tmp_path)
    with pytest.raises(RuntimeError, match="task_episode_baseline_missing"):
        backend.apply_and_measure(_request(0, 0.5))


def test_backend_blocks_second_baseline_capture(tmp_path: Path):
    backend = _hermetic_backend(tmp_path)
    backend.capture_episode_baseline(
        task_success_contract=CONTRACT,
        attempt_id="run-1-attempt-000001",
        launch_nonce="nonce-1",
    )
    with pytest.raises(RuntimeError, match="persistent_isaac_episode_baseline_already_captured"):
        backend.capture_episode_baseline(
            task_success_contract=CONTRACT,
            attempt_id="run-1-attempt-000001",
            launch_nonce="nonce-1",
        )


def test_backend_restart_cannot_recapture_same_attempt_baseline(tmp_path: Path):
    first = _hermetic_backend(tmp_path)
    first.capture_episode_baseline(
        task_success_contract=CONTRACT,
        attempt_id="run-1-attempt-000001",
        launch_nonce="nonce-1",
    )
    restarted = _hermetic_backend(tmp_path)
    with pytest.raises(RuntimeError, match="baseline_artifact_already_exists"):
        restarted.capture_episode_baseline(
            task_success_contract=CONTRACT,
            attempt_id="run-1-attempt-000001",
            launch_nonce="nonce-1",
        )


def test_backend_blocks_tampered_episode_baseline(tmp_path: Path):
    backend = _hermetic_backend(tmp_path)
    backend.capture_episode_baseline(
        task_success_contract=CONTRACT,
        attempt_id="run-1-attempt-000001",
        launch_nonce="nonce-1",
    )
    backend.episode_baseline["episode_initial_value"] = -1.0
    with pytest.raises(RuntimeError, match="task_episode_baseline_digest_mismatch"):
        backend.apply_and_measure(_request(0, 0.5))


def test_backend_blocks_session_restart_after_baseline(tmp_path: Path):
    backend = _hermetic_backend(tmp_path)
    backend.capture_episode_baseline(
        task_success_contract=CONTRACT,
        attempt_id="run-1-attempt-000001",
        launch_nonce="nonce-1",
    )
    backend.session_id = "persistent-session-2-restarted"
    with pytest.raises(RuntimeError, match="task_episode_baseline_session_mismatch"):
        backend.apply_and_measure(_request(0, 0.5))


def test_backend_blocks_changed_target_prim(tmp_path: Path):
    backend = _hermetic_backend(tmp_path)
    backend.capture_episode_baseline(
        task_success_contract=CONTRACT,
        attempt_id="run-1-attempt-000001",
        launch_nonce="nonce-1",
    )
    changed = {
        "registered_criteria": [
            {
                **CONTRACT["registered_criteria"][0],
                "articulation_prim_path": "/root/Refrigerator001/Door",
            }
        ]
    }
    with pytest.raises(RuntimeError, match="task_episode_baseline_prim_mismatch"):
        backend.apply_and_measure(_request(0, 0.5, contract=changed))


def _full_g1_dof_names() -> list[str]:
    return [name for names in G1_CANONICAL_DOF_GROUPS.values() for name in names]


def test_initial_policy_state_maps_full_g1_inventory_and_passes_dims_contract(
    tmp_path: Path,
):
    backend = _hermetic_backend(tmp_path)
    state = backend.initial_policy_state()
    assert validate_g1_sonic_state_dims(state) == []
    assert state["left_leg"] == [pytest.approx(0.01 * index) for index in range(6)]
    mapping = state["proprioception_mapping"]
    assert len(mapping["mapping_digest"]) == 64
    assert len(mapping["observed_dof_inventory"]) == 43
    assert mapping["dimensions"]["left_arm"] == 7
    assert mapping["unmapped_observed_dofs"] == []
    assert state["measurement"]["source"] == (
        "live_isaac_articulation_dof_positions_and_base_orientation"
    )
    assert state["measurement"]["surrogate"] is False
    assert state["measurement"]["mapping_digest"] == mapping["mapping_digest"]


def test_initial_policy_state_blocks_on_missing_required_dof(tmp_path: Path):
    names = [name for name in _full_g1_dof_names() if name != "right_wrist_yaw_joint"]
    backend = _hermetic_backend(tmp_path)
    backend.robot = _Articulation(_RuntimeState(), names)
    with pytest.raises(
        RuntimeError,
        match=r"persistent_isaac_initial_proprio_mapping_blocked:"
        r".*g1_proprioception_required_dof_missing:right_wrist_yaw_joint",
    ):
        backend.initial_policy_state()


def test_initial_policy_state_blocks_on_duplicate_dof(tmp_path: Path):
    backend = _hermetic_backend(tmp_path)
    backend.robot = _Articulation(_RuntimeState(), _full_g1_dof_names() + ["left_hip_pitch_joint"])
    with pytest.raises(
        RuntimeError,
        match="g1_proprioception_observed_dof_duplicate:left_hip_pitch_joint",
    ):
        backend.initial_policy_state()


class _ServiceBackend:
    def __init__(self, tmp_path: Path):
        self.evidence_dir = tmp_path
        self.results: list[dict] = []

    def queue(self, result: dict) -> None:
        self.results.append(result)

    def apply_and_measure(self, request):
        return self.results.pop(0)


def _measurement(step: int, *, before: float, after: float, initial: float) -> dict:
    return {
        "schema_version": "task_transition_measurement.v1",
        "criterion_id": "microwave_door_open_angle",
        "observable_transition": "articulation_angle_rad",
        "before_value": before,
        "after_value": after,
        "episode_initial_value": initial,
        "step_before": before,
        "step_after": after,
        "step_delta": after - before,
        "episode_delta": after - initial,
        "episode_baseline_digest": "b" * 64,
        "unit": "rad",
        "source_step_index": step,
        "source_action_sha256": "a" * 64,
        "articulation_prim_path": "/root/Microwave017/Microwave017_Door",
        "simulator_session_id": "persistent-session-1",
        "stage_id": "stage-1",
        "before_timestamp": "1",
        "after_timestamp": "2",
        "runtime_result_id": f"persistent-session-1-step-{step:04d}",
        "persistent_simulator_state_applied": True,
        "official_controller_action_applied": True,
        "evidence_artifacts": [],
        "review_frames": [],
    }


def test_service_two_small_steps_pass_episode_criterion_only_after_step_two(
    tmp_path: Path,
):
    backend = _ServiceBackend(tmp_path)
    backend.queue(_measurement(0, before=0.0, after=0.20, initial=0.0))
    backend.queue(_measurement(1, before=0.20, after=0.40, initial=0.0))
    key_file = _signing_key_file(tmp_path)

    results = [
        service._evaluate_completion_request(
            backend=backend,
            request=_request(step, 0.5),
            signing_key_file=str(key_file),
            attempt_input_manifest_sha256="c" * 64,
        )
        for step in (0, 1)
    ]
    assert results[0]["passed"] is False
    assert results[1]["passed"] is True
    for result in results:
        assert result["status"] == "completed"
        assert result["evaluation_basis"] == "episode_relative"
        assert result["comparison"] == "increase_at_least"
        assert result["attempt_input_manifest_sha256"] == "c" * 64
        assert result["evaluator_attestation"]["signature_verified"] is True
    assert results[0]["episode_delta"] == pytest.approx(0.20)
    assert results[1]["episode_delta"] == pytest.approx(0.40)


def test_service_blocks_result_without_episode_fields(tmp_path: Path):
    backend = _ServiceBackend(tmp_path)
    legacy = _measurement(0, before=0.0, after=0.40, initial=0.0)
    del legacy["episode_initial_value"]
    backend.queue(legacy)
    with pytest.raises(RuntimeError, match="persistent_isaac_task_result_episode_fields_missing"):
        service._evaluate_completion_request(
            backend=backend,
            request=_request(0, 0.5),
            signing_key_file=str(_signing_key_file(tmp_path)),
            attempt_input_manifest_sha256="c" * 64,
        )


class _MainBackend:
    def __init__(self, evidence_dir: Path):
        self.evidence_dir = evidence_dir
        self.session_id = "persistent-session-main"
        self.stage_id = "d" * 64
        self.capture_calls: list[dict] = []
        self.baseline_attestation = None
        self.events: list[str] = []

    def initial_policy_state(self):
        self.events.append("initial_policy_state")
        assert hasattr(self, "baseline_guard")
        return {"left_leg": [0.0] * 6}

    def capture_episode_baseline(
        self,
        *,
        task_success_contract,
        attempt_id,
        launch_nonce,
        task_contract_artifact_sha256,
    ):
        self.capture_calls.append(
            {
                "task_success_contract": task_success_contract,
                "attempt_id": attempt_id,
                "launch_nonce": launch_nonce,
                "task_contract_artifact_sha256": task_contract_artifact_sha256,
            }
        )
        from blueprint_pipeline.task_episode_baseline import build_task_episode_baseline

        return build_task_episode_baseline(
            episode_initial_value=0.0,
            attempt_id=attempt_id,
            launch_nonce=launch_nonce,
            simulator_session_id=self.session_id,
            stage_id=self.stage_id,
            articulation_prim_path="/root/Microwave017/Microwave017_Door",
            task_contract_sha256=canonical_task_contract_sha256(task_success_contract),
            task_contract_artifact_sha256=task_contract_artifact_sha256,
            criterion_id="microwave_door_open_angle",
            unit="rad",
            captured_timestamp="1",
        )

    def install_episode_baseline_attestation(self, attestation):
        self.baseline_attestation = dict(attestation)

    def capture_initial_policy_observation(
        self, *, target_prim_path, frame_output_path, projection_context_output_path
    ):
        self.events.append("capture_initial_policy_observation")
        self.initial_capture = {
            "target_prim_path": target_prim_path,
            "frame_output_path": str(frame_output_path),
            "projection_context_output_path": str(projection_context_output_path),
        }
        Path(frame_output_path).write_bytes(b"live-isaac-rgb")
        Path(projection_context_output_path).write_text(
            json.dumps({"status": "captured_from_live_persistent_isaac_session"}),
            encoding="utf-8",
        )
        return {"status": "completed"}

    def verify_initial_observation_preserved_episode_baseline(
        self, *, task_success_contract
    ):
        assert task_success_contract == CONTRACT
        self.events.append("verify_initial_observation_preserved_episode_baseline")
        self.baseline_guard = {"status": "passed"}
        return dict(self.baseline_guard)


def _write_main_fixtures(tmp_path: Path) -> tuple[Path, Path]:
    contract_path = tmp_path / "task_success_contract.json"
    contract_path.write_text(json.dumps(CONTRACT, sort_keys=True), encoding="utf-8")
    attempt_path = tmp_path / "attempt_input_manifest.json"
    attempt_path.write_text(
        json.dumps(
            {
                "schema_version": "g1_kitchen_attempt_input_manifest.v1",
                "attempt_id": "run-1-attempt-000001",
                "launch_nonce": "nonce-1",
                "artifacts": {
                    "task_success_contract": {
                        "path": str(contract_path),
                        "sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return attempt_path, contract_path


def _run_main(monkeypatch, tmp_path: Path, *, corrupt_contract_sha: bool = False):
    attempt_path, contract_path = _write_main_fixtures(tmp_path)
    if corrupt_contract_sha:
        manifest = json.loads(attempt_path.read_text(encoding="utf-8"))
        manifest["artifacts"]["task_success_contract"]["sha256"] = "0" * 64
        attempt_path.write_text(json.dumps(manifest), encoding="utf-8")
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    route_file = tmp_path / "route.json"
    route_file.write_text(
        json.dumps(
            {
                "route_points": [[-1.2, 1.4, 0.84]],
                "accepted_stance_yaw_rad": 3.141593,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "initial_state.json").write_text(
        json.dumps({"measurement": {"surrogate": False}, "stale": True}),
        encoding="utf-8",
    )
    (tmp_path / "initial_policy_frame.png").write_bytes(b"stale-frame")
    (tmp_path / "controller_fk_camera_projection_context.json").write_text(
        json.dumps({"status": "stale-static-context"}), encoding="utf-8"
    )
    backend = _MainBackend(evidence_dir)
    serve_calls: list[dict] = []

    monkeypatch.setattr(backend_module, "create_backend", lambda **kwargs: backend)
    monkeypatch.setattr(service, "serve", lambda **kwargs: serve_calls.append(kwargs))
    monkeypatch.setenv(
        "BLUEPRINT_SC3_TASK_COMPLETION_PRIVATE_KEY_FILE",
        str(_signing_key_file(tmp_path)),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "isaac_persistent_task_executor_service",
            "--stage",
            str(tmp_path / "stage.usd"),
            "--route-file",
            str(route_file),
            "--evidence-dir",
            str(evidence_dir),
            "--initial-state-output",
            str(tmp_path / "initial_state.json"),
            "--initial-frame-output",
            str(tmp_path / "initial_policy_frame.png"),
            "--camera-projection-context-output",
            str(tmp_path / "controller_fk_camera_projection_context.json"),
            "--attempt-input-manifest",
            str(attempt_path),
        ],
    )
    exit_code = service.main()
    return exit_code, backend, serve_calls, attempt_path


def test_main_captures_signed_episode_baseline_before_serving(monkeypatch, tmp_path: Path):
    exit_code, backend, serve_calls, attempt_path = _run_main(monkeypatch, tmp_path)
    assert exit_code == 0
    assert backend.capture_calls == [
        {
            "task_success_contract": CONTRACT,
            "attempt_id": "run-1-attempt-000001",
            "launch_nonce": "nonce-1",
            "task_contract_artifact_sha256": json.loads(
                attempt_path.read_text(encoding="utf-8")
            )["artifacts"]["task_success_contract"]["sha256"],
        }
    ]
    attestation = json.loads(
        (backend.evidence_dir / "task_episode_baseline_attestation.json").read_text(
            encoding="utf-8"
        )
    )
    assert attestation["signature_verified"] is True
    assert backend.baseline_attestation == attestation
    assert (backend.evidence_dir / "task_episode_baseline_signature.json").is_file()
    assert (tmp_path / "initial_state.json").is_file()
    assert json.loads((tmp_path / "initial_state.json").read_text(encoding="utf-8")) == {
        "left_leg": [0.0] * 6
    }
    assert (tmp_path / "initial_policy_frame.png").read_bytes() == b"live-isaac-rgb"
    assert backend.initial_capture["target_prim_path"] == (
        "/root/Microwave017/Microwave017_Door"
    )
    assert Path(backend.initial_capture["projection_context_output_path"]).is_file()
    assert backend.events == [
        "capture_initial_policy_observation",
        "verify_initial_observation_preserved_episode_baseline",
        "initial_policy_state",
    ]
    assert len(serve_calls) == 1
    assert (
        serve_calls[0]["attempt_input_manifest_sha256"]
        == hashlib.sha256(attempt_path.read_bytes()).hexdigest()
    )


def test_main_blocks_on_task_contract_sha_mismatch(monkeypatch, tmp_path: Path):
    with pytest.raises(SystemExit, match="persistent_isaac_task_contract_sha256_mismatch"):
        _run_main(monkeypatch, tmp_path, corrupt_contract_sha=True)


def test_compose_g1_for_episode_reuses_proven_route_stance(tmp_path: Path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdPhysics

    g1_asset = tmp_path / "g1.usda"
    asset_stage = Usd.Stage.CreateNew(str(g1_asset))
    asset_robot = asset_stage.DefinePrim("/G1", "Xform")
    UsdPhysics.ArticulationRootAPI.Apply(asset_robot)
    asset_stage.SetDefaultPrim(asset_robot)
    asset_stage.GetRootLayer().Save()

    stage = Usd.Stage.CreateNew(str(tmp_path / "kitchen.usda"))
    stage.DefinePrim("/World", "Xform")
    route = tmp_path / "route.json"
    route.write_text(
        json.dumps(
            {
                "route_points": [[-1.229635, 1.471274, 0.84]],
                "accepted_stance_yaw_rad": 3.141593,
            }
        ),
        encoding="utf-8",
    )

    result = backend_module.compose_g1_for_episode(
        stage,
        robot_prim_path="/World/G1",
        g1_usd_path=g1_asset,
        route_file=route,
    )
    assert result["status"] == "passed"
    assert result["robot_was_already_present"] is False
    assert result["start_pose_xyz"] == [-1.229635, 1.471274, 0.84]
    assert stage.GetPrimAtPath("/World/G1").IsValid()
    assert result["articulation_root_paths"]
