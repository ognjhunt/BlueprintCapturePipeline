"""Drive a robot team's own robot in the site scene: build it, gate its first observation, run one episode.

Beta journey step: L5 team policy evaluation. ``team_embodiment`` validates
and binds what the team declared; this module is the GPU side, imported only
inside the Isaac process:

- ``build_team_embodiment`` spawns the team's USD at its start pose (the
  scene's robot station for a fixed base, its floor stance for a free one,
  under gravity), with each declared joint's gains, a joint-position action
  over the position-driven joints (then the gripper's) and a joint-velocity
  action over the velocity-driven ones (wheels), and one camera per declared
  camera;
- ``team_observation_gate`` runs before the team's policy is contacted: the
  robot reset to its declared pose at its start pose, every team camera
  rendered a real frame, and the scene's overview camera sees the task object.
  Whether the team's own cameras frame the task is the team's choice and is
  recorded, never gated;
- ``run_team_episode`` is the closed loop: the team's cameras, every declared
  joint's position and velocity, the gripper closure and the base's pose and
  velocity go to the team's policy; each returned row is one target per
  declared joint (a position, or a wheel's velocity) plus a gripper closure;
  the simulator executes it at the team's control rate and records the task
  object's state at the scene's task rate; the scene's sealed success contract
  scores the samples (``score_task_episode_from_spec``), exactly as for
  Blueprint's own robot. A free base whose root drops below its declared fall
  height has fallen: the policy is not queried again and the episode is the
  robot's failure.

A row outside a joint's declared limits, or a closure outside [0, 1], is
refused before it reaches the robot and ends the episode as the policy's
failure (``team_candidate_quick10_result.episode_outcome``).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

BASE_DRIFT_TOLERANCE_M = 0.01
RESET_TOLERANCE_RAD = 1e-3
LIMIT_TOLERANCE = 1e-6


class TeamPolicyActionRejected(ValueError):
    """The team's action was refused before it reached the robot."""

    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        super().__init__(";".join(self.errors))


def _array(value: Any) -> Any:
    import numpy as np

    value = getattr(value, "torch", value)
    if not hasattr(value, "detach") and type(value).__module__.split(".")[0] == "warp":
        import warp as wp

        value = wp.to_torch(value)
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _rgb(camera: Any) -> Any:
    import numpy as np

    frame = _array(camera.data.output["rgb"])[0]
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        raise RuntimeError(f"team_camera_frame_shape_invalid:{tuple(frame.shape)}")
    frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _calibration(camera: Any) -> dict[str, Any]:
    from .adp009d_isaac_episode_adapter import rotation_row_major_from_quaternion_xyzw

    intrinsic = _array(camera.data.intrinsic_matrices)[0]
    position = _array(camera.data.pos_w)[0]
    rotation = rotation_row_major_from_quaternion_xyzw(_array(camera.data.quat_w_opengl)[0])
    clipping = tuple(camera.cfg.spawn.clipping_range)
    frame = _array(camera.data.output["rgb"])[0]
    return {"camera_model": "pinhole", "intrinsic_matrix": [[float(v) for v in row] for row in intrinsic],
            "world_from_camera": [[*rotation[row * 3:row * 3 + 3], float(position[row])] for row in range(3)]
            + [[0.0, 0.0, 0.0, 1.0]],
            "resolution": [int(frame.shape[1]), int(frame.shape[0])],
            "near_m": float(clipping[0]), "far_m": float(clipping[1]), "world_pose_source": "isaac_sensor_buffer"}


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


def _team_roles(built: Any, plan: Mapping[str, Any]) -> list[tuple[str, str]]:
    """``(role, team camera name)`` in the team's declared order."""

    rows = [row for row in plan["cameras"] if row.get("policy_input") is True]
    roles = [(str(row["role"]), str(row["team_camera_name"])) for row in rows]
    if not roles or any(role not in built.camera_scene_names for role, _ in roles):
        raise RuntimeError("team_embodiment_camera_binding_missing")
    return roles


def team_observation_gate(*, built: Any, plan: Mapping[str, Any], output_root: Path,
                          camera_snapshot: Callable[..., Mapping[str, Any]],
                          site_appearance_render_expected: bool) -> dict[str, Any]:
    """Prove the team robot, its cameras and the scene before the team's policy is contacted."""

    import numpy as np
    from PIL import Image

    env = built.env
    env.reset(seed=int(plan["scenario"]["seed"]))
    robot = env.unwrapped.scene["robot"]
    measured = dict(zip(robot.joint_names, (float(v) for v in _array(robot.data.joint_pos)[0]), strict=True))
    expected = plan["robot"]["joint_reset_positions_rad"]
    blockers = [f"team_embodiment_reset_joint_missing:{name}" for name in expected if name not in measured]
    errors = {name: abs(measured[name] - float(value)) for name, value in expected.items() if name in measured}
    blockers += [f"team_embodiment_reset_mismatch:{name}" for name, error in errors.items()
                 if not error <= RESET_TOLERANCE_RAD]
    base = [float(v) for v in _array(robot.data.root_pose_w)[0][:3]]
    drift = math.dist(base, [float(v) for v in plan["robot"]["base_pose_world"]["position_world_m"]])
    if not drift <= BASE_DRIFT_TOLERANCE_M:
        blockers.append("team_embodiment_base_not_fixed_at_station" if plan["robot"]["base_mode"] == "fixed_at_station"
                        else "team_embodiment_base_not_at_floor_stance")
    root = output_root / "prepolicy_observation_gate" / "team_reset"
    root.mkdir(parents=True, exist_ok=True)
    frames = {}
    for role, name in _team_roles(built, plan):
        frame = _rgb(env.unwrapped.scene[built.camera_scene_names[role]])
        Image.fromarray(frame).save(root / f"{role}.png")
        luminance = frame.astype(np.float64).mean(axis=-1)
        rendered = bool(luminance.mean() >= 8.0 and luminance.std() >= 1.0)
        frames[role] = {"team_camera_name": name, "width": int(frame.shape[1]), "height": int(frame.shape[0]),
                        "mean_luminance": float(luminance.mean()), "luminance_std": float(luminance.std()),
                        "rendered": rendered}
        if not rendered:
            blockers.append(f"team_camera_frame_not_rendered:{name}")
    if "overview" not in built.camera_scene_names:
        blockers.append("team_embodiment_overview_camera_missing")
        overview = {}
    else:
        snapshot = camera_snapshot(env=env, camera_scene_names={"overview": built.camera_scene_names["overview"]},
                                   output_root=root, snapshot_id="team_reset_overview",
                                   framing_expectations=(plan.get("task_object_observability") or {}).get("cameras"),
                                   site_appearance_render_expected=site_appearance_render_expected)
        overview = next((row for row in snapshot.get("cameras") or [] if row.get("role") == "overview"), {})
        if (overview.get("observability") or {}).get("passed") is not True:
            blockers.append("team_embodiment_overview_task_visibility_failed")
    passed = not blockers
    receipt = {
        "schema_version": "team_embodiment_observation_gate.v1", "status": "passed" if passed else "blocked",
        "run_kind": "team_candidate_evaluation", "claim_ceiling": "diagnostic_policy_execution",
        "embodiment_digest": plan["robot"]["embodiment_digest"],
        "reset_joint_errors_rad": errors, "base_drift_m": drift, "team_camera_frames": frames,
        "overview_observability": overview.get("observability"),
        "team_camera_task_framing": "team_choice_recorded_not_gated",
        "candidate_policy_loaded": False, "candidate_policy_queried": False,
        "official_ranking_permitted": False, "scene_promotion_permitted": False,
        "blockers": sorted(set(blockers)), "policy_observation_integrity_passed": passed, "gate_digest": "",
    }
    receipt["gate_digest"] = canonical_digest(receipt, digest_field="gate_digest")
    return receipt


class _GripperWidth:
    """The robot-side half of a rigid scoring sample: the gripper's opening, from its declared joints."""

    def __init__(self, width: Callable[[], float | None], tools: Callable[[], list[dict[str, Any]]] | None = None):
        self._width, self._tools = width, tools

    def read_object_sample(self) -> dict[str, Any]:
        return {"gripper_width_m": self._width(),
                **({"tool_states": self._tools()} if self._tools is not None else {})}

    def reset(self) -> None:
        return None


def run_team_episode(*, built: Any, task_readback: Any, policy: Any, plan: Mapping[str, Any], prompt: str,
                     max_policy_queries: int, open_loop_horizon: int, media_root: Path, episode_id: str,
                     progress: dict[str, Any], to_action: Callable[[list[float]], Any] | None = None
                     ) -> dict[str, Any]:
    """One closed-loop episode of the team's policy on the team's robot, scored by the sealed contract."""

    import numpy as np

    from .adp_task_scoring import score_task_episode_from_spec
    from .episode_visual_evidence import (
        finalize_multicamera_visual_evidence,
        persist_multicamera_observation,
    )
    from .native_task_episode_environment import NativeRigidScoringEnvironment
    from .policy_episode_trace_evidence import episode_trace_evidence
    from .policy_request_evidence import capture_request, request_evidence_sink

    env = built.env
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    row, task_spec, cadence = plan["robot"], plan["task_spec"], plan["cadence"]
    declared = list(row["joint_names"])
    controls = row["joint_controls"]
    gripper = list(row["gripper_joints"])
    names = list(robot.joint_names)
    missing = [name for name in [*declared, *(j["name"] for j in gripper)] if name not in names]
    if missing:
        raise RuntimeError("team_embodiment_joint_missing_at_runtime:" + ",".join(missing))
    declared_index = [names.index(name) for name in declared]
    gripper_index = [names.index(joint["name"]) for joint in gripper]
    multi_tool = "gripper_channels" in row
    channels = list(row["gripper_channels"]) if multi_tool else ([{
        "name": "gripper", "joints": gripper, "max_width_m": row["gripper_max_width_m"]}] if gripper else [])
    channel_indices = [[names.index(j["name"]) for j in channel["joints"]] for channel in channels]
    # Where each declared joint's value sits in a policy row, split by how the simulator drives it.
    position_columns = [j for j, name in enumerate(declared) if controls[name] == "position"]
    velocity_columns = [j for j, name in enumerate(declared) if controls[name] == "velocity"]
    bounds = [row["joint_position_limits_rad"][name] if controls[name] == "position"
              else [-float(row["joint_velocity_limits"][name]), float(row["joint_velocity_limits"][name])]
              for name in declared]
    fall_height = float(row["fall_height_world_m"]) if row["base_mode"] == "free_on_floor" else None
    width_dimension = len(declared) + len(channels)
    stride = int(cadence["task_sample_stride"])
    action_steps = int(cadence["maximum_action_steps"])
    settle_steps = int(cadence["settle_window_samples"])
    roles = _team_roles(built, plan)
    media_cameras = {role: built.camera_scene_names[role] for role, _ in roles}
    media_cameras["overview"] = built.camera_scene_names["overview"]
    step_seconds = 1.0 / float(cadence["control_frequency_hz"])
    if to_action is None:
        import torch

        def to_action(targets: list[float]) -> Any:
            return torch.tensor([targets], dtype=torch.float32, device=unwrapped.device)

    def joint_positions() -> Any:
        return _array(robot.data.joint_pos)[0]

    def base_state() -> dict[str, list[float]]:
        pose, velocity = _array(robot.data.root_pose_w)[0], _array(robot.data.root_vel_w)[0]
        # The pinned Isaac Lab Beta2 root-pose buffer already uses XYZW.
        return {"position_world_m": [float(v) for v in pose[:3]],
                "orientation_xyzw": [float(v) for v in pose[3:7]],
                "linear_velocity_world_mps": [float(v) for v in velocity[:3]],
                "angular_velocity_world_radps": [float(v) for v in velocity[3:6]]}

    def closures(q: Any) -> list[float]:
        values = []
        for channel, indices in zip(channels, channel_indices, strict=True):
            fractions = [(float(q[i]) - float(j["open"])) / (float(j["closed"]) - float(j["open"]))
                         for i, j in zip(indices, channel["joints"], strict=True)]
            values.append(float(min(1.0, max(0.0, sum(fractions) / len(fractions)))))
        return values

    def tool_states() -> list[dict[str, Any]]:
        q = joint_positions()
        return [{"name": channel["name"], "tool_link": channel["tool_link"],
                 "gripper_position": fraction,
                 "gripper_width_m": float(channel["max_width_m"]) * (1.0 - fraction),
                 "joint_positions": {j["name"]: float(q[i])
                                     for j, i in zip(channel["joints"], indices, strict=True)},
                 "width_source": "declared_joint_open_closed_and_maximum_width"}
                for channel, indices, fraction in zip(channels, channel_indices, closures(q), strict=True)]

    def width() -> float | None:
        # There is no single gripper width for v2. The scorer consumes every
        # measured tool state; never copy the primary width to other tools.
        values = closures(joint_positions())
        return None if multi_tool or not values else float(row["gripper_max_width_m"]) * (1.0 - values[0])

    scoring = NativeRigidScoringEnvironment(environment=_GripperWidth(width, tool_states if multi_tool else None),
                                            task_readback=task_readback, task_spec=task_spec)
    seed = int(plan["scenario"]["seed"])
    env.reset(seed=seed)
    scoring.begin_episode()
    observations: list[dict[str, Any]] = []
    step = 0

    def capture(kind: str) -> dict[str, Any]:
        cameras = {role: unwrapped.scene[name] for role, name in media_cameras.items()}
        simulation_time = step * step_seconds
        return persist_multicamera_observation(
            {role: _rgb(camera) for role, camera in cameras.items()}, output_dir=media_root,
            episode_id=episode_id, observation_index=len(observations), kind=kind,
            timestamp_ns=round(simulation_time * 1e9), simulation_time_s=simulation_time,
            calibrations={role: _calibration(camera) for role, camera in cameras.items()},
            source_devices={role: str(unwrapped.device) for role in cameras},
            synchronizations={role: {"host_bytes_ready": True,
                                     "method": "environment_step_completed_before_read_only_host_copy"}
                              for role in cameras})

    identity = {"episode_id": episode_id, "embodiment_digest": row["embodiment_digest"],
                "candidate_policy": getattr(policy, "policy_id", None)}

    def seal_media(*, failure_reason: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        terminal = capture("terminal-observation")
        visual, artifacts = finalize_multicamera_visual_evidence(
            output_dir=media_root, episode_id=episode_id, identity=identity,
            policy_input_observations=observations, terminal_observation=terminal,
            required_camera_ids=sorted(media_cameras), review_only_camera_ids=["overview"])
        if failure_reason is not None:
            visual = {**visual, "failure_reason": failure_reason}
        return visual, artifacts

    progress["_failure_media_finalizer"] = lambda failure_reason: seal_media(failure_reason=failure_reason)
    samples: list[dict[str, Any]] = []
    joint_trace: list[list[float]] = []
    commanded: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    request_artifacts: list[dict[str, Any]] = []
    progress["policy_request_artifacts"] = request_artifacts

    fell = False

    def sample_task_state() -> None:
        q = joint_positions()
        joint_trace.append([float(q[i]) for i in declared_index])
        sample = scoring.read_object_sample()
        sample["step_index"] = len(samples)
        # Task readback shares the control loop's simulator clock, not the
        # policy service's latency or the camera persistence wall clock.
        sample["simulation_time_s"] = step * step_seconds
        samples.append(sample)

    def advance(targets: Sequence[float]) -> None:
        nonlocal step, fell
        env.step(to_action(list(targets)))
        step += 1
        if fall_height is not None and float(_array(robot.data.root_pose_w)[0][2]) < fall_height:
            fell = True
        if step % stride == 0:
            sample_task_state()

    q0 = joint_positions()
    base0 = base_state()["position_world_m"]
    sample_task_state()
    # Hold: every position joint (and the gripper) where it is; every wheel at rest.
    last = ([float(q0[declared_index[j]]) for j in position_columns] + [float(q0[i]) for i in gripper_index]
            + [0.0] * len(velocity_columns))
    for query_index in range(int(max_policy_queries)):
        if step >= action_steps or fell:
            break
        q, qd = joint_positions(), _array(robot.data.joint_vel)[0]
        images = {name: _rgb(unwrapped.scene[built.camera_scene_names[role]]) for role, name in roles}
        observation = {"images": images, "prompt": prompt,
                       "state": {"joint_position": [float(q[i]) for i in declared_index],
                                 "joint_velocity": [float(qd[i]) for i in declared_index],
                                 "gripper_position": closures(q),
                                 **({"gripper_names": [channel["name"] for channel in channels]} if multi_tool else {}),
                                 "base": base_state()}}
        observations.append(capture("policy-input"))
        progress["first_observation_retained"] = True
        progress.update(candidate_policy_query_attempted=True, candidate_action_applied=False, phase="policy_query")
        started = time.monotonic()

        retain_request = request_evidence_sink(
            root=media_root, episode_id=episode_id, query_index=query_index, progress=progress,
            artifacts=request_artifacts, binding=lambda: {**identity, "task_spec_digest": canonical_digest(task_spec)},
        )

        bind_request_sink = getattr(policy, "bind_request_evidence_sink", None)
        if callable(bind_request_sink):
            bind_request_sink(retain_request)
        else:
            retain_request(capture_request(observation, transport="injected_policy_client"))
        try:
            chunk = policy.infer(observation)
        except Exception:
            # Only an inference exception marks a policy failure. The retained
            # request sink marks its own disk/media failures as infrastructure.
            progress["policy_inference_failed"] = True
            raise
        latency = time.monotonic() - started
        progress.update(candidate_policy_queried=True, candidate_action_returned=True)
        try:
            actions = np.asarray(chunk, dtype=float)
        except (TypeError, ValueError):
            actions = np.zeros((0, 0))
        if actions.ndim != 2 or actions.shape[1] != width_dimension or not 1 <= actions.shape[0] <= int(
                row["action_chunk_rows"]):
            raise ValueError(f"team_policy_https_action_shape_invalid:{tuple(actions.shape)}")
        progress["candidate_action_shape_validated"] = True
        if not np.isfinite(actions).all():
            raise ValueError("team_policy_https_action_nonfinite")
        progress["candidate_action_finite_validated"] = True
        executed = actions[:int(open_loop_horizon)]
        violations = [f"candidate_action_joint_position_bounds_invalid:{declared[j]}:row{r}"
                      for r in range(executed.shape[0]) for j in range(len(declared))
                      if not bounds[j][0] - LIMIT_TOLERANCE <= executed[r, j] <= bounds[j][1] + LIMIT_TOLERANCE]
        if channels:
            violations += [f"candidate_action_joint_position_bounds_invalid:gripper_closure"
                           f"{':' + channel['name'] if multi_tool else ''}:row{r}"
                           for r in range(executed.shape[0]) for index, channel in enumerate(channels)
                           if not 0.0 <= executed[r, len(declared) + index] <= 1.0]
        queries.append({"query_index": query_index, "latency_seconds": latency, "returned_rows": int(actions.shape[0]),
                        "executed_rows": int(executed.shape[0]), "any_joint_limit_clamped": False,
                        "bounds_refused": bool(violations)})
        if violations:
            progress["phase"] = "policy_action_bounds_refused"
            raise TeamPolicyActionRejected(violations[:16])
        for action in executed:
            if step >= action_steps or fell:
                break
            targets = [float(action[j]) for j in position_columns]
            for index, channel in enumerate(channels):
                for joint in channel["joints"]:
                    targets.append(float(joint["open"]) + float(action[len(declared) + index])
                                   * (float(joint["closed"]) - float(joint["open"])))
            targets += [float(action[j]) for j in velocity_columns]
            commanded.append({"step_index": step, "targets": targets})
            advance(targets)
            progress["candidate_action_applied"] = True
            last = targets
    if velocity_columns:  # the wheels stop while the task object settles
        last = [*last[:len(last) - len(velocity_columns)], *([0.0] * len(velocity_columns))]
    for _ in range(settle_steps):  # hold the last command while the task object settles
        advance(last)
    if samples[-1]["simulation_time_s"] != step * step_seconds:
        # A fall can leave a partial sample stride before the settling window.
        # Retain that actual terminal state so its clock includes every step.
        sample_task_state()
    score = score_task_episode_from_spec(task_spec=task_spec, samples=samples)
    visual, artifacts = seal_media()
    artifacts.extend(request_artifacts)
    state_trace, contacts, trajectory = episode_trace_evidence(
        joint_trace=joint_trace, task_samples=samples, task_pose_field="task_object_pose_world")
    moved = (bool(joint_trace) and max(abs(a - b) for row_q in joint_trace
                                       for a, b in zip(row_q, [float(q0[i]) for i in declared_index], strict=True))
             > 1e-3) or math.dist(base_state()["position_world_m"], base0) > 1e-3
    return {
        "status": "completed", "score": score, "visual_evidence": visual, "media_artifacts": artifacts,
        "motion_evidence": {"actions_reached_robot": bool(commanded), "arm_moved": moved, "robot_fell": fell,
                            "control_hz": float(cadence["control_frequency_hz"]), "task_sample_stride": stride,
                            **({"self_collisions_enabled": row.get("enabled_self_collisions") is True,
                                "self_collision_force_readback": "unavailable_task_and_scene_contacts_only"}
                               if multi_tool else {})},
        "commanded_actions": commanded, "state_trace": state_trace, "contact_force_evidence": contacts,
        "task_object_trajectory": trajectory, "queries": queries, "policy_queries": len(queries),
        "policy_query_latency": {"seconds": [row_q["latency_seconds"] for row_q in queries]},
        "camera_calibration": None, "resource_telemetry": None, "telemetry_channels": None,
        "scientific_reset": None, "embodiment_digest": row["embodiment_digest"],
        "policy_request_artifacts": request_artifacts,
        "policy_request_evidence_complete": bool(request_artifacts)
            and all(item["serialization_verified"] for item in request_artifacts),
    }


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
    "TeamPolicyActionRejected",
    "asset_root_for_scoring_pose",
    "build_team_embodiment",
    "record_arena_robot_free_probe",
    "run_team_episode",
    "team_observation_gate",
]
