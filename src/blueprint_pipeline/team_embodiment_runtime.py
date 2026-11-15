"""Build a robot team's own robot in the site scene and record the robot-free probe on it.

Beta journey step: L5, the scripted check. ``team_embodiment`` validates and
binds what the team declared; this module is the GPU side, imported only
inside the Isaac process:

- ``build_team_embodiment`` spawns the team's USD at its start pose (the
  scene's robot station for a fixed base, its floor stance for a free one,
  under gravity), with each declared joint's gains, a joint-position action
  over the position-driven joints (then the gripper's) and a joint-velocity
  action over the velocity-driven ones (wheels), and one camera per declared
  camera;
- ``record_arena_robot_free_probe`` records the robot-free probe in that scene:
  the robot stays at its reset pose and physics alone decides where the object
  rests; the trace is scored on the control plane (``team_robot_free_probe``).

The closed-loop team episode of Blueprint's own early trial left with the trial
(owner, 2026-11-14).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .decision_evidence_contracts import canonical_digest

def _array(value: Any) -> Any:
    import numpy as np

    value = getattr(value, "torch", value)
    if not hasattr(value, "detach") and type(value).__module__.split(".")[0] == "warp":
        import warp as wp

        value = wp.to_torch(value)
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def action_joint_groups(robot: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """``(position-driven, velocity-driven)`` joints in the simulator's action order.

    Position: the declared position joints in declared order, then the gripper's.
    Velocity: the declared velocity joints (wheels) in declared order.
    """

    controls = robot["joint_controls"]
    return ([name for name in robot["joint_names"] if controls[name] == "position"]
            + [joint["name"] for joint in robot["gripper_joints"]],
            [name for name in robot["joint_names"] if controls[name] == "velocity"])


def build_team_embodiment(robot: Mapping[str, Any], *, enable_cameras: bool, pose_class: Any) -> Any:
    """The team's robot as an Isaac Lab Arena embodiment, spawned at its start pose."""

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
    from isaaclab.envs import mdp
    from isaaclab.managers import (
        EventTermCfg,
        ObservationGroupCfg,
        ObservationTermCfg,
        SceneEntityCfg,
    )
    from isaaclab.sensors.camera.tiled_camera_cfg import TiledCameraCfg
    from isaaclab.utils import configclass
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase

    position, velocity = action_joint_groups(robot)
    pose = robot["base_pose_world"]
    free = robot["base_mode"] == "free_on_floor"
    gains = {**{joint["name"]: {key: float(robot["actuator"][key]) for key in ("stiffness", "damping")}
                for joint in robot["gripper_joints"]}, **robot["joint_gains"]}
    articulation = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(robot["usd_path"]), activate_contact_sensors=True,
            # A fixed base is held by its anchor; a free base stands, walks or drives under gravity.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=not free, max_depenetration_velocity=5.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=robot.get("enabled_self_collisions") is True,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=0)),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(pose["position_world_m"]), rot=tuple(pose["orientation_xyzw"]),
            joint_pos=dict(robot["joint_reset_positions_rad"]), joint_vel={".*": 0.0}),
        soft_joint_pos_limit_factor=1.0,
        actuators={"team_joints": ImplicitActuatorCfg(
            joint_names_expr=[*position, *velocity],
            stiffness={name: gains[name]["stiffness"] for name in [*position, *velocity]},
            damping={name: gains[name]["damping"] for name in [*position, *velocity]})},
    )

    @configclass
    class SceneCfg:
        robot: ArticulationCfg = articulation

    # The simulator's action vector: position targets, then wheel velocities (``action_joint_groups``).
    terms: dict[str, Any] = {}
    if position:
        terms["joint_positions"] = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=position, scale=1.0, use_default_offset=False, preserve_order=True)
    if velocity:
        terms["joint_velocities"] = mdp.JointVelocityActionCfg(
            asset_name="robot", joint_names=velocity, scale=1.0, use_default_offset=False, preserve_order=True)
    Actions = configclass(type("TeamActionsCfg", (), {"__annotations__": {k: type(v) for k, v in terms.items()},
                                                     **terms}))

    @configclass
    class PolicyObservations(ObservationGroupCfg):
        joint_positions = ObservationTermCfg(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class Observations:
        policy = PolicyObservations()

    events: dict[str, Any] = {"reset_robot_joints": EventTermCfg(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)})}
    if free:
        # A free base returns to its floor stance, at rest, every episode.
        events["reset_robot_root"] = EventTermCfg(
            func=mdp.reset_root_state_uniform, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot"), "pose_range": {}, "velocity_range": {}})
    Events = configclass(type("TeamEventsCfg", (), {"__annotations__": {k: EventTermCfg for k in events}, **events}))

    # The arena runtime poses, sizes and calibrates each camera from the scene plan.
    names = [f"team_camera_{index}" for index in range(len(robot["camera_names"]))] + ["external_camera_2"]

    def camera() -> Any:
        return TiledCameraCfg(prim_path="{ENV_REGEX_NS}/team_camera_unposed", height=224, width=224,
                              data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(),
                              offset=TiledCameraCfg.OffsetCfg(convention="ros"))

    cameras = configclass(type("TeamCameraCfg", (), {"__annotations__": {name: TiledCameraCfg for name in names},
                                                    **{name: camera() for name in names}}))

    embodiment = EmbodimentBase(enable_cameras=enable_cameras, initial_pose=pose_class(
        position_xyz=tuple(pose["position_world_m"]), rotation_xyzw=tuple(pose["orientation_xyzw"])))
    embodiment.scene_config = SceneCfg()
    embodiment.action_config = Actions()
    embodiment.observation_config = Observations()
    embodiment.event_config = Events()
    embodiment.camera_config = cameras()
    embodiment.get_scene_cfg()
    return embodiment


def _quaternion_multiply(a: Sequence[float], b: Sequence[float]) -> list[float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz]


def _rotate(quaternion: Sequence[float], vector: Sequence[float]) -> list[float]:
    x, y, z, w = quaternion
    rotated = _quaternion_multiply(_quaternion_multiply(quaternion, [*vector, 0.0]), [-x, -y, -z, w])
    return rotated[:3]


def asset_root_for_scoring_pose(scoring_pose: Sequence[float], offset: Mapping[str, Any]) -> list[float]:
    """The asset-root pose that puts the task's scoring frame at ``scoring_pose`` (all xyz + xyzw)."""

    ox, oy, oz, ow = (float(v) for v in offset["orientation_xyzw"])
    inverse = [-ox, -oy, -oz, ow]
    rotation = _quaternion_multiply([float(v) for v in scoring_pose[3:7]], inverse)
    shift = _rotate(rotation, [float(v) for v in offset["position_m"]])
    return [float(scoring_pose[axis]) - shift[axis] for axis in range(3)] + rotation


def probe_status(*, request: Mapping[str, Any], trace: Mapping[str, Any]) -> str:
    """The recorded probe's verdict on the GPU host, before any team episode (same scorer as closeout)."""

    from .rigid_robot_free_readiness_score import score_rigid_robot_free_readiness

    if (request.get("request_digest") != canonical_digest(dict(request), digest_field="request_digest")
            or request["binding"]["task_spec_digest"] != canonical_digest(dict(request["task_spec"]))):
        return "probe_request_invalid"
    receipt = score_rigid_robot_free_readiness(
        task_spec=request["task_spec"], physics_probe=trace,
        configured_scene_revision_digest=request["binding"]["configured_scene_revision_digest"])
    return str(receipt["status"])


class _ArenaProbeScene:
    """The rigid probe's three-method seam over the built scene: set the object's pose, step physics, read."""

    def __init__(self, *, built: Any, task_readback: Any, plan: Mapping[str, Any]):
        import torch

        self._torch = torch
        self._env = built.env.unwrapped
        self._object = self._env.scene[built.scene_asset_names["task_object"]]
        self._readback = task_readback
        spec = plan["task_spec"]
        self._offset = spec["interaction_affordance"]["asset_root_from_scoring_frame"]
        self._contact_n = float(spec["task_contact_minimum_force_n"])

    def set_scoring_pose(self, pose: Sequence[float]) -> None:
        device = self._env.device
        root = asset_root_for_scoring_pose(pose, self._offset)
        self._object.write_root_pose_to_sim(self._torch.tensor([root], dtype=self._torch.float32, device=device))
        self._object.write_root_velocity_to_sim(self._torch.zeros((1, 6), dtype=self._torch.float32, device=device))

    def step(self) -> None:
        self._env.scene.write_data_to_sim()
        self._env.sim.step(render=False)
        self._env.scene.update(dt=self._env.physics_dt)

    def read_sample(self) -> Mapping[str, Any]:
        native = self._readback.read_task_sample()
        return {"task_scoring_pose_world": native["task_scoring_pose_world"],
                "support_contact_active": float(native["task_support_contact_peak_force_n"]) >= self._contact_n,
                "scene_collision_force_n": float(native["task_scene_collision_peak_force_n"])}


def record_arena_robot_free_probe(*, built: Any, task_readback: Any, plan: Mapping[str, Any],
                                  request: Mapping[str, Any]) -> dict[str, Any]:
    """Execute the control plane's robot-free probe plan in the built scene; never raise.

    The robot stays at its reset pose and never drives the object: the object
    is left untouched, then carried by setting its pose and released, and
    physics alone decides where it rests. The trace is scored on the control
    plane (``team_robot_free_probe``).
    """

    from .decision_evidence_contracts import canonical_digest
    from .rigid_robot_free_probe_recorder import record_rigid_robot_free_probe

    try:
        if (request.get("request_digest") != canonical_digest(dict(request), digest_field="request_digest")
                or abs(float(built.env.unwrapped.physics_dt) * float(request["plan"]["physics_hz"]) - 1.0) > 1e-6):
            raise RuntimeError("team_robot_free_probe_request_or_rate_invalid")
        built.env.reset(seed=int(plan["scenario"]["seed"]))
        scene = _ArenaProbeScene(built=built, task_readback=task_readback, plan=plan)
        return record_rigid_robot_free_probe(scene=scene, plan=request["plan"], binding=request["binding"])
    except Exception as exc:  # noqa: BLE001 - a recording failure is evidence; the cell-0 gate refuses the scene
        trace = {"schema_version": "scene_rigid_robot_free_success_probe.v1", "binding": dict(request.get("binding") or {}),
                 "plan": dict(request.get("plan") or {}), "status": "failed",
                 "error": f"{type(exc).__name__}:{str(exc)[:200]}", "trace_digest": ""}
        trace["trace_digest"] = canonical_digest(trace, digest_field="trace_digest")
        return trace


__all__ = [
    "asset_root_for_scoring_pose",
    "build_team_embodiment",
    "record_arena_robot_free_probe",
]
