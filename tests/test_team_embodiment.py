"""A robot team's own robot, of any form: bound to the scene, driven by its own policy, scored by the scene.

Beta journey step: L5 team policy evaluation. Hermetic: a tiny OpenUSD robot
authored here stands in for a team's model; Isaac is faked only inside the
episode-loop test. No GPU, provider, network or spend.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from blueprint_pipeline import (
    adp_task_scoring,
    episode_visual_evidence,
    native_task_episode_environment,
)
from blueprint_pipeline.native_task_arena_runtime import camera_runtime_parameters
from blueprint_pipeline.policy_canary_worker_evidence import _write_episode_failure_gap
from blueprint_pipeline.team_candidate_quick10_result import episode_outcome
from blueprint_pipeline.team_embodiment import (
    TeamEmbodimentError,
    apply_team_embodiment,
    read_model_structure,
    team_embodiment_binding,
    validate_embodiment_manifest,
    verify_model_file,
)
from blueprint_pipeline.team_embodiment_runtime import TeamPolicyActionRejected, run_team_episode
from blueprint_pipeline.team_policy_https_client import (
    METADATA_SCHEMA,
    TeamPolicyHttpsClient,
    TeamPolicySpec,
)
from tests.test_native_task_arena_policy_canary_lifecycle_rehearsal import _scene_plan

ARM = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"]
WHEELS, WHEEL_LINKS = ["left_wheel_joint", "right_wheel_joint"], ["left_wheel", "right_wheel"]
ORIGIN = "https://policy.acme-robotics.example"


def write_robot_usd(path: Path, *, anchored: bool = True, wheels: bool = False) -> Path:
    """A three-joint arm with a two-finger gripper (on two wheels), as Isaac's URDF importer would flatten it."""

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    root = stage.DefinePrim("/acme_arm", "Xform")
    stage.SetDefaultPrim(root)
    links = ["base_link", "upper_arm", "forearm", "wrist", "left_finger", "right_finger",
             *(WHEEL_LINKS if wheels else [])]
    for link in links:
        UsdPhysics.RigidBodyAPI.Apply(stage.DefinePrim(f"/acme_arm/{link}", "Xform"))
    for name, kind in [*((joint, UsdPhysics.RevoluteJoint) for joint in ARM + (WHEELS if wheels else [])),
                       ("finger_joint", UsdPhysics.PrismaticJoint)]:
        kind.Define(stage, f"/acme_arm/joints/{name}")
    if anchored:
        # The importer's fixed-base anchor: base_link pinned to the world (no body0).
        UsdPhysics.FixedJoint.Define(stage, "/acme_arm/joints/root_joint").CreateBody1Rel().SetTargets(
            ["/acme_arm/base_link"])
    stage.GetRootLayer().Save()
    return path


def manifest(model: Path, **overrides: Any) -> dict[str, Any]:
    value = {
        "schema_version": "blueprint.robot_embodiment.v1", "embodiment_id": "acme-arm-v2", "form": "fixed_base_arm",
        "model": {"format": "usd", "url": "https://models.acme-robotics.example/acme_arm.usda",
                  "sha256": "sha256:" + hashlib.sha256(model.read_bytes()).hexdigest()},
        "base": {"mode": "fixed"},
        "joints": [{"name": name, "control": "position", "lower": -3.0, "upper": 3.0, "max_velocity": 2.0}
                   for name in ARM],
        "gripper": {"joints": [{"name": "finger_joint", "open": 0.04, "closed": 0.0}], "max_width_m": 0.08,
                    "contact_links": ["left_finger", "right_finger"]},
        "tool_link": "wrist", "reset_joint_positions": {**dict.fromkeys(ARM, 0.0), "finger_joint": 0.04},
        "actuator": {"stiffness": 400.0, "damping": 40.0},
        "cameras": [{"name": "wrist_rgb", "mount_link": "wrist", "position_m": [0.0, 0.05, 0.02],
                     "orientation_xyzw": [0.0, 0.0, 0.0, 1.0], "width": 320, "height": 240,
                     "horizontal_fov_deg": 90.0},
                    {"name": "scene_rgb", "mount_link": None, "position_m": [1.0, 0.0, 1.2],
                     "orientation_xyzw": [0.0, 0.0, 0.0, 1.0], "width": 320, "height": 240,
                     "horizontal_fov_deg": 60.0}],
        "control_hz": 30, "action_chunk_rows": 4, "reach_m": 3.0,
    }
    value.update(overrides)
    return value


def station_plan() -> dict[str, Any]:
    """The rehearsal's rigid scene with Blueprint's robot at its station and its robot-side contact sensors."""

    plan = _scene_plan()
    plan["robot"]["joint_reset_positions_rad"] = {"panda_joint1": 0.0}
    plan["cadence"].update(control_frequency_hz=15.0, control_decimation=8, maximum_action_steps=90)
    plan["task_spec"]["control_frequency_hz"] = 15.0
    franka = "{ENV_REGEX_NS}/Robot/panda_link"
    plan["articulation"] = {
        "contact_sensors": [
            {"sensor_instance_id": "task_robot_contact__subject", "logical_sensor_id": "task_robot_contact",
             "prim_path": "{ENV_REGEX_NS}/task_object", "filter_prim_paths_expr": [franka + "7"]},
            {"sensor_instance_id": "robot_scene_contact__00", "logical_sensor_id": "robot_scene_contact",
             "prim_path": franka + "0", "filter_prim_paths_expr": ["{ENV_REGEX_NS}/scene/table"]}],
        "forbidden_robot_contact_body_paths": [franka + "0"]}
    return plan


def binding(tmp_path: Path, **overrides: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    model = write_robot_usd(tmp_path / "acme_arm.usda")
    declared = validate_embodiment_manifest(manifest(model, **overrides))
    sealed = team_embodiment_binding(declared, model_record=verify_model_file(declared, model),
                                     structure=read_model_structure(model), scene_plan=station_plan())
    return declared, sealed


def test_a_team_arm_stands_at_the_scene_station_with_its_own_cameras_and_rate(tmp_path) -> None:
    declared, sealed = binding(tmp_path)
    base = station_plan()
    plan = apply_team_embodiment(base, sealed)
    robot = plan["robot"]
    assert robot["robot_id"] == "team_embodiment" and robot["base_pose_world"] == base["robot"]["base_pose_world"]
    assert robot["joint_names"] == ARM and robot["embodiment_digest"] == declared["embodiment_digest"]
    assert robot["base_mode"] == "fixed_at_station" and set(robot["joint_controls"].values()) == {"position"}
    assert robot["usd_path"] == "team_embodiment/robot.usd" and robot["usd_sha256"] == declared["model"]["sha256"]
    # The team's two cameras replace Blueprint's policy cameras; the overview stays for review only.
    assert [(row["role"], row["policy_input"]) for row in plan["cameras"]] == [
        ("team_camera_0", True), ("team_camera_1", True), ("overview", False)]
    assert plan["cameras"][0]["parent_prim_path"] == "{ENV_REGEX_NS}/Robot/wrist"
    assert all(camera_runtime_parameters(row, robot_id="team_embodiment") for row in plan["cameras"])
    with pytest.raises(Exception, match="camera_role_invalid"):
        camera_runtime_parameters(plan["cameras"][0])  # never accepted as Blueprint's own robot
    # 30 Hz control over the same seconds, task state still sampled at the scene's 15 Hz.
    assert plan["cadence"]["control_decimation"] == 4 and plan["cadence"]["task_sample_stride"] == 2
    assert plan["cadence"]["maximum_action_steps"] == 180 and plan["task_spec"] == base["task_spec"]
    # Robot-side contact channels now watch the team's bodies, never the Franka's.
    sensors = plan["articulation"]["contact_sensors"]
    task = next(row for row in sensors if row["logical_sensor_id"] == "task_robot_contact")
    assert task["filter_prim_paths_expr"] == ["{ENV_REGEX_NS}/Robot/left_finger", "{ENV_REGEX_NS}/Robot/right_finger"]
    scene = [row["prim_path"] for row in sensors if row["logical_sensor_id"] == "robot_scene_contact"]
    assert len(scene) == 6 and not any("panda" in json.dumps(row) for row in sensors)
    assert plan["policy_canary_embodiment_profile"]["robot_preset_id"] == "team_embodiment"


def test_a_camera_shifted_cell_moves_the_team_scene_camera_and_reads_that_camera_back(tmp_path) -> None:
    _declared, sealed = binding(tmp_path)
    shift = {"parameter_id": "external_camera_x_delta_m", "readback_kind": "camera_offset_position_x_m",
             "camera_role": "external", "expected_native_value": 0.53, "delta_from_nominal": 0.03}
    # As the scene resolver leaves it: Blueprint's external camera shifted, before the team's replace it.
    plan = apply_team_embodiment({**station_plan(), "scenario": {"parameter_applications": [shift]}}, sealed)
    nominal = apply_team_embodiment({**station_plan(), "scenario": {"parameter_applications": []}}, sealed)
    wrist, scene = plan["cameras"][0], plan["cameras"][1]
    assert scene["frame_from_camera_matrix"][3] == pytest.approx(
        nominal["cameras"][1]["frame_from_camera_matrix"][3] + 0.03)
    assert wrist["frame_from_camera_matrix"] == nominal["cameras"][0]["frame_from_camera_matrix"]
    [applied] = plan["scenario"]["parameter_applications"]
    assert applied["camera_role"] == "team_camera_1"
    assert applied["expected_native_value"] == scene["frame_from_camera_matrix"][3]


def test_an_arm_that_cannot_be_evaluated_is_refused_by_name_before_any_gpu(tmp_path) -> None:
    model = write_robot_usd(tmp_path / "acme_arm.usda")
    with pytest.raises(TeamEmbodimentError) as error:
        validate_embodiment_manifest(manifest(model, control_hz=10, cameras=[]))
    assert {"team_embodiment_control_hz_invalid", "team_embodiment_cameras_invalid"} <= set(error.value.blockers)
    cases = {
        "team_embodiment_model_joint_missing:wrist_3_joint": {"joints": manifest(model)["joints"] + [
            {"name": "wrist_3_joint", "control": "position", "lower": -1.0, "upper": 1.0, "max_velocity": 1.0}],
            "reset_joint_positions": {**dict.fromkeys([*ARM, "wrist_3_joint"], 0.0), "finger_joint": 0.04}},
        "team_embodiment_station_out_of_reach": {"reach_m": 0.2},
        # A fixed arm cannot be declared free (a free base goes to the task instead of reaching from a station).
        "team_embodiment_form_base_mismatch": {"base": {"mode": "free", "root_height_m": 0.8, "fall_height_m": 0.4,
                                                        "footprint_half_extent_m": [0.3, 0.3],
                                                        "ground_contact_links": ["base_link"]}},
    }
    for index, (blocker, overrides) in enumerate(cases.items()):
        (tmp_path / str(index)).mkdir()
        with pytest.raises(TeamEmbodimentError) as error:
            binding(tmp_path / str(index), **overrides)
        assert blocker in error.value.blockers
    # A model that does not pin its base to the world would float under contact: refused.
    floating = write_robot_usd(tmp_path / "floating.usda", anchored=False)
    loose = validate_embodiment_manifest(manifest(floating))
    with pytest.raises(TeamEmbodimentError) as error:
        team_embodiment_binding(loose, model_record=verify_model_file(loose, floating),
                                structure=read_model_structure(floating), scene_plan=station_plan())
    assert "team_embodiment_model_base_not_anchored_to_world" in error.value.blockers
    declared = validate_embodiment_manifest(manifest(model))
    other = tmp_path / "other.usda"
    other.write_text("#usda 1.0\n")
    with pytest.raises(TeamEmbodimentError, match="model_digest_mismatch"):
        verify_model_file(declared, other)


def test_the_team_service_gets_its_own_cameras_and_state_and_must_name_its_robot(tmp_path) -> None:
    model = write_robot_usd(tmp_path / "acme_arm.usda")
    declared = validate_embodiment_manifest(manifest(model))
    spec = TeamPolicySpec(policy_id="acme-arm-pick", endpoint_url=ORIGIN + "/v1/act", metadata_url=ORIGIN + "/v1/act",
                          allowed_origin=ORIGIN, model_identity_digest="sha256:" + "7" * 64,
                          embodiment_id=declared["embodiment_id"], action_chunk_rows=4, open_loop_horizon=4,
                          camera_names=["wrist_rgb", "scene_rgb"])
    metadata = spec.expected_metadata()
    seen: list[dict[str, Any]] = []

    def transport(*, url: str, method: str, body: bytes | None, headers: dict[str, str]) -> bytes:
        assert headers["Authorization"] == "Bearer team-token-0123456789"
        if method == "GET":
            return json.dumps(metadata).encode()
        seen.append(json.loads(body))
        return json.dumps({"actions": [[0.1, 0.2, 0.3, 1.0]] * 4}).encode()

    client = TeamPolicyHttpsClient(spec=spec, bearer_token="team-token-0123456789", transport=transport)
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    actions = client.infer({"images": {"wrist_rgb": image, "scene_rgb": image}, "prompt": "Put the mug on the tray",
                            "state": {"joint_position": [0.0, 0.1, 0.2], "joint_velocity": [0.0, 0.0, 0.0],
                                      "gripper_position": [0.0], "base": BASE_AT_REST}})
    assert actions == [[0.1, 0.2, 0.3, 1.0]] * 4
    request = seen[0]
    assert (request["schema_version"], request["policy_interface_id"]) == (
        "blueprint.policy_https_request.v1", "blueprint_policy_https.v1")
    assert sorted(request["observation"]["images"]) == ["scene_rgb", "wrist_rgb"]
    assert request["observation"]["state"] == {"joint_position": [0.0, 0.1, 0.2], "joint_velocity": [0.0, 0.0, 0.0],
                                               "gripper_position": [0.0], "base": BASE_AT_REST}
    assert metadata["schema_version"] == METADATA_SCHEMA
    metadata = {**metadata, "embodiment_id": "some-other-robot"}  # served for some other robot
    with pytest.raises(ValueError, match="identity_mismatch"):
        client.preflight_readiness()


class _Array:
    def __init__(self, value: Any):
        self.torch = np.asarray(value)


class _Camera:
    def __init__(self) -> None:
        self.data = type("Data", (), {})()
        self.data.output = {"rgb": _Array(np.full((1, 24, 32, 3), 90, dtype=np.uint8))}
        self.data.intrinsic_matrices = _Array([[[30.0, 0, 15.5], [0, 30.0, 11.5], [0, 0, 1]]])
        self.data.pos_w = _Array([[0.0, 0.0, 1.0]])
        self.data.quat_w_opengl = _Array([[0.0, 0.0, 0.0, 1.0]])
        self.cfg = type("Cfg", (), {"spawn": type("Spawn", (), {"clipping_range": (0.01, 100.0)})()})()


BASE_AT_REST = {"position_world_m": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "linear_velocity_world_mps": [0.0, 0.0, 0.0], "angular_velocity_world_radps": [0.0, 0.0, 0.0]}


class _Robot:
    def __init__(self, joint_names: list[str] | None = None) -> None:
        self.joint_names = joint_names or ["finger_joint", *ARM]
        self.q = np.zeros(len(self.joint_names))
        self.q[self.joint_names.index("finger_joint")] = 0.04
        self.root = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # Isaac Lab: position, then wxyz
        self.data = self

    @property
    def joint_pos(self) -> _Array:
        return _Array([self.q])

    @property
    def joint_vel(self) -> _Array:
        return _Array([np.zeros(len(self.joint_names))])

    @property
    def root_pose_w(self) -> _Array:
        return _Array([self.root])

    @property
    def root_vel_w(self) -> _Array:
        return _Array([np.zeros(6)])


class _Env:
    """Executes each target exactly; records what reached the robot."""

    def __init__(self) -> None:
        self.robot, self.steps = _Robot(), []
        camera = _Camera()
        scene = {"robot": self.robot, "team_camera_0": camera, "team_camera_1": camera, "external_camera_2": camera}
        self.unwrapped = type("Unwrapped", (), {"scene": scene, "device": "cpu"})()

    def reset(self, seed: int) -> None:
        self.robot.q = np.array([0.04, 0.0, 0.0, 0.0])

    def step(self, action: list[list[float]]) -> None:
        targets = action[0]
        self.steps.append(list(targets))
        self.robot.q = np.array([targets[3], *targets[:3]])


class _Scoring:
    def __init__(self, *, environment: Any, task_readback: Any, task_spec: Any) -> None:
        self.environment = environment

    def begin_episode(self) -> None:
        return None

    def read_object_sample(self) -> dict[str, Any]:
        return self.environment.read_object_sample()


class _Policy:
    policy_id = "acme-arm-pick"

    def __init__(self, chunks: list[list[list[float]]]):
        self.chunks, self.observations = chunks, []

    def infer(self, observation: dict[str, Any]) -> list[list[float]]:
        self.observations.append(observation)
        return self.chunks[min(len(self.observations), len(self.chunks)) - 1]


def test_the_team_policy_drives_the_team_arm_and_an_out_of_limit_row_never_reaches_it(tmp_path, monkeypatch) -> None:
    _declared, sealed = binding(tmp_path)
    plan = {**apply_team_embodiment(station_plan(), sealed), "scenario": {"seed": 7}}  # as the worker resolves it
    monkeypatch.setattr(native_task_episode_environment, "NativeRigidScoringEnvironment", _Scoring)
    scored: list[list[dict[str, Any]]] = []
    monkeypatch.setattr(adp_task_scoring, "score_task_episode_from_spec",
                        lambda *, task_spec, samples: scored.append(samples) or {"status": "scored",
                                                                                 "task_succeeded": False})
    monkeypatch.setattr(episode_visual_evidence, "finalize_multicamera_visual_evidence",
                        lambda **kwargs: ({"status": "complete",
                                           "required_camera_ids": kwargs["required_camera_ids"]}, []))
    built = type("Built", (), {"env": _Env(), "camera_scene_names": {
        "team_camera_0": "team_camera_0", "team_camera_1": "team_camera_1", "overview": "external_camera_2"}})()
    policy = _Policy([[[0.1, 0.2, 0.3, 1.0]] * 4])
    progress: dict[str, Any] = {}
    episode = run_team_episode(built=built, task_readback=None, policy=policy, plan=plan, prompt="Put the mug away",
                               max_policy_queries=100, open_loop_horizon=4, media_root=tmp_path / "episodes",
                               episode_id="run--cell--team", progress=progress, to_action=lambda t: [t])
    first = policy.observations[0]
    assert sorted(first["images"]) == ["scene_rgb", "wrist_rgb"] and first["images"]["wrist_rgb"].dtype == np.uint8
    assert first["state"] == {"joint_position": [0.0, 0.0, 0.0], "joint_velocity": [0.0, 0.0, 0.0],
                              "gripper_position": [0.0], "base": BASE_AT_REST}
    # A closure of 1 drives the declared gripper joint to its closed position; arm targets pass through.
    assert built.env.steps[0] == [0.1, 0.2, 0.3, 0.0] and policy.observations[1]["state"]["gripper_position"] == [1.0]
    # 180 control steps at 30 Hz plus the settle window, task state sampled every second step.
    assert len(built.env.steps) == 180 + 2 and len(scored[0]) == 91
    assert scored[0][0]["gripper_width_m"] == 0.0 and [s["step_index"] for s in scored[0][:3]] == [1, 2, 3]
    assert episode["score"]["status"] == "scored" and episode["motion_evidence"]["arm_moved"] is True

    env = _Env()
    built.env = env
    progress = {}
    with pytest.raises(TeamPolicyActionRejected) as rejected:
        run_team_episode(built=built, task_readback=None, policy=_Policy([[[0.1, 0.2, 9.0, 0.0]]]), plan=plan,
                         prompt="Put the mug away", max_policy_queries=100, open_loop_horizon=4,
                         media_root=tmp_path / "rejected", episode_id="run--cell2--team", progress=progress,
                         to_action=lambda t: [t])
    assert env.steps == []  # refused before it reached the robot
    path = _write_episode_failure_gap(output_root=tmp_path / "out", run_id="run", progress=progress,
                                      context={"cell_id": "cell2", "candidate_id": "team"}, failure=rejected.value)
    row = json.loads(path.read_text())
    assert episode_outcome({**row, "status": "failed"}) == ("failure", "policy_action_beyond_joint_limits")


def test_a_team_robot_is_admitted_only_with_a_valid_manifest_and_a_rigid_task(tmp_path) -> None:
    from blueprint_pipeline.team_candidate_quick10_admission import (
        TeamCandidateAdmissionError,
        admit_team_candidate,
    )
    from tests.test_team_candidate_quick10_admission import NOW, _inputs

    declared, _sealed = binding(tmp_path)
    records = _inputs()
    candidate = copy.deepcopy(records["candidate"])
    candidate.update(embodiment_preset_id="team_embodiment", policy_interface_id="blueprint_policy_https.v1",
                     embodiment=declared)
    candidate["delivery"].update(action_chunk_rows=4, open_loop_horizon=4)
    with pytest.raises(TeamCandidateAdmissionError) as refused:
        admit_team_candidate(**{**records, "candidate": candidate}, now=NOW)
    # This fixture scene is articulated; the team-robot runtime scores rigid pick-and-place only.
    assert refused.value.blockers == ["team_embodiment_task_kind_unsupported"]
    candidate["embodiment"] = {**declared, "form": "humanoid"}
    candidate["delivery"]["action_chunk_rows"] = 10
    with pytest.raises(TeamCandidateAdmissionError) as refused:
        admit_team_candidate(**{**records, "candidate": candidate}, now=NOW)
    assert {"team_embodiment_digest_mismatch"} <= set(refused.value.blockers)


def wheeled_manifest(model: Path) -> dict[str, Any]:
    """A wheeled humanoid: free on the floor, an arm by position, two wheels by velocity."""

    value = manifest(model, form="wheeled_humanoid",
                     base={"mode": "free", "root_height_m": 0.9, "fall_height_m": 0.45,
                           "footprint_half_extent_m": [0.3, 0.3], "ground_contact_links": WHEEL_LINKS},
                     joints=[*manifest(model)["joints"],
                             *({"name": name, "control": "velocity", "max_velocity": 10.0, "damping": 5.0}
                               for name in WHEELS)])
    value.pop("reach_m")
    return value


STANCE = {"position_world_m": [1.0, -1.0, 0.9], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0], "floor_z_m": 0.0}


def test_a_wheeled_humanoid_starts_free_on_the_floor_and_its_wheels_are_driven_by_velocity(tmp_path) -> None:
    model = write_robot_usd(tmp_path / "wheeled.usda", anchored=False, wheels=True)
    declared = validate_embodiment_manifest(wheeled_manifest(model))
    record, structure = verify_model_file(declared, model), read_model_structure(model)
    with pytest.raises(TeamEmbodimentError, match="free_base_stance_unavailable"):
        team_embodiment_binding(declared, model_record=record, structure=structure, scene_plan=station_plan())
    sealed = team_embodiment_binding(declared, model_record=record, structure=structure, scene_plan=station_plan(),
                                     stance=STANCE)
    plan = apply_team_embodiment(station_plan(), sealed)
    robot = plan["robot"]
    assert robot["base_mode"] == "free_on_floor" and robot["base_pose_world"]["position_world_m"] == [1.0, -1.0, 0.9]
    assert robot["fall_height_world_m"] == 0.45 and robot["joint_names"] == [*ARM, *WHEELS]
    assert robot["joint_gains"]["left_wheel_joint"] == {"stiffness": 0.0, "damping": 5.0}
    # Wheels on the floor are standing, not a collision: they carry no robot-on-scene sensor.
    scene = [row["prim_path"] for row in plan["articulation"]["contact_sensors"]
             if row["logical_sensor_id"] == "robot_scene_contact"]
    assert scene and not any(path.endswith(("left_wheel", "right_wheel")) for path in scene)
    # A free base whose model pins it to the world could never move.
    pinned = write_robot_usd(tmp_path / "pinned.usda", wheels=True)
    pinned_manifest = validate_embodiment_manifest(wheeled_manifest(pinned))
    with pytest.raises(TeamEmbodimentError, match="free_base_anchored_to_world"):
        team_embodiment_binding(pinned_manifest, model_record=verify_model_file(pinned_manifest, pinned),
                                structure=read_model_structure(pinned), scene_plan=station_plan(), stance=STANCE)


class _WheeledEnv(_Env):
    """Falls over after its fifth control step."""

    def __init__(self) -> None:
        super().__init__()
        self.robot = _Robot(["finger_joint", *ARM, *WHEELS])
        self.unwrapped.scene["robot"] = self.robot
        self.reset(0)

    def reset(self, seed: int) -> None:
        self.robot.q = np.array([0.04, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.robot.root = np.array([1.0, -1.0, 0.9, 1.0, 0.0, 0.0, 0.0])

    def step(self, action: list[list[float]]) -> None:
        targets = action[0]
        self.steps.append(list(targets))
        # Simulator order: arm positions, the finger's, then the wheels' velocities.
        self.robot.q = np.array([targets[3], *targets[:3], *targets[4:]])
        self.robot.root[2] = 0.9 if len(self.steps) < 5 else 0.2


def test_a_wheeled_humanoid_drives_by_wheel_velocity_and_a_fall_ends_its_episode_as_its_failure(
        tmp_path, monkeypatch) -> None:
    model = write_robot_usd(tmp_path / "wheeled.usda", anchored=False, wheels=True)
    declared = validate_embodiment_manifest(wheeled_manifest(model))
    sealed = team_embodiment_binding(declared, model_record=verify_model_file(declared, model),
                                     structure=read_model_structure(model), scene_plan=station_plan(), stance=STANCE)
    plan = {**apply_team_embodiment(station_plan(), sealed), "scenario": {"seed": 7}}
    monkeypatch.setattr(native_task_episode_environment, "NativeRigidScoringEnvironment", _Scoring)
    monkeypatch.setattr(adp_task_scoring, "score_task_episode_from_spec",
                        lambda *, task_spec, samples: {"status": "scored", "task_succeeded": True})
    monkeypatch.setattr(episode_visual_evidence, "finalize_multicamera_visual_evidence",
                        lambda **kwargs: ({"status": "complete"}, []))
    env = _WheeledEnv()
    built = type("Built", (), {"env": env, "camera_scene_names": {
        "team_camera_0": "team_camera_0", "team_camera_1": "team_camera_1", "overview": "external_camera_2"}})()
    # One row: three arm targets, two wheel velocities, then the gripper closure.
    policy = _Policy([[[0.1, 0.2, 0.3, 2.0, -2.0, 1.0]] * 4])
    episode = run_team_episode(built=built, task_readback=None, policy=policy, plan=plan, prompt="Put the mug away",
                               max_policy_queries=100, open_loop_horizon=4, media_root=tmp_path / "episodes",
                               episode_id="run--cell--team", progress={}, to_action=lambda t: [t])
    first = policy.observations[0]["state"]
    assert len(first["joint_position"]) == len(first["joint_velocity"]) == 5
    assert first["base"]["position_world_m"] == [1.0, -1.0, 0.9]
    assert env.steps[0] == [0.1, 0.2, 0.3, 0.0, 2.0, -2.0]
    # It fell on its fifth step: the policy is not asked again, and the wheels stop while the object settles.
    assert len(policy.observations) == 2 and env.steps[5] == [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]
    assert episode["motion_evidence"]["robot_fell"] is True
    row = {"status": "completed", "scoring_authority": "deterministic_simulator_state", "episode": episode}
    assert episode_outcome(row) == ("failure", "robot_fell")
    # A wheel velocity beyond its declared maximum never reaches the robot.
    env = _WheeledEnv()
    built.env = env
    with pytest.raises(TeamPolicyActionRejected, match="left_wheel_joint"):
        run_team_episode(built=built, task_readback=None, policy=_Policy([[[0.1, 0.2, 0.3, 50.0, 0.0, 0.0]]]),
                         plan=plan, prompt="Put the mug away", max_policy_queries=100, open_loop_horizon=4,
                         media_root=tmp_path / "rejected", episode_id="run--cell2--team", progress={},
                         to_action=lambda t: [t])
    assert env.steps == []


def write_room_usd(path: Path) -> Path:
    """A floor and a table, as the scene's collision mesh: Z up, meters."""

    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(stage.DefinePrim("/World", "Xform"))
    floor = UsdGeom.Mesh.Define(stage, "/World/Floor")
    floor.CreatePointsAttr([(-1.0, -3.0, 0.0), (5.0, -3.0, 0.0), (5.0, 3.0, 0.0), (-1.0, 3.0, 0.0)])
    floor.CreateFaceVertexCountsAttr([4])
    floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    floor.CreateExtentAttr([(-1.0, -3.0, 0.0), (5.0, 3.0, 0.0)])
    (x0, y0, z0), (x1, y1, z1) = (1.5, -0.3, 0.0), (2.5, 0.3, 0.75)
    table = UsdGeom.Mesh.Define(stage, "/World/Table")
    table.CreatePointsAttr([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)])
    table.CreateFaceVertexCountsAttr([4] * 6)
    table.CreateFaceVertexIndicesAttr([0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7])
    table.CreateExtentAttr([(x0, y0, z0), (x1, y1, z1)])
    stage.GetRootLayer().Save()
    return path


def test_a_free_base_starts_at_the_closest_clear_floor_spot_facing_the_task(tmp_path) -> None:
    from blueprint_pipeline.team_embodiment_stance import free_base_stance

    room = write_room_usd(tmp_path / "assets" / "collision.usda")
    model = write_robot_usd(tmp_path / "wheeled.usda", anchored=False, wheels=True)
    declared = validate_embodiment_manifest(wheeled_manifest(model))
    plan = {"objects": [{"semantic_role": "scene_collision", "usd_path": "assets/collision.usda",
                         "sha256": "sha256:" + hashlib.sha256(room.read_bytes()).hexdigest(),
                         "pose_world": {"position_world_m": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}}],
            "task_spec": {"start_pose_world": [2.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0],
                          "subject_collision_bounds_scoring_frame_m": {"minimum": [-0.05, -0.05, -0.05],
                                                                        "maximum": [0.05, 0.05, 0.05]}}}
    stance = free_base_stance(declared, scene_plan=plan, packet_dir=tmp_path)
    x, y, z = stance["position_world_m"]
    assert abs(stance["floor_z_m"]) < 1e-6 and abs(z - 0.9) < 1e-6  # on the floor, not the table
    # Clear of the table's footprint, facing the object.
    assert abs(x - 2.0) >= 0.5 + 0.3 or abs(y) >= 0.3 + 0.3
    yaw = stance["yaw_rad"]
    assert abs(math.atan2(0.0 - y, 2.0 - x) - yaw) < 1e-9 and stance["clear_stances"] > 0
    # A room with no clear floor near the task refuses the robot by name.
    crowded = {**declared, "base": {**declared["base"], "footprint_half_extent_m": [1.5, 1.5]}}
    with pytest.raises(TeamEmbodimentError, match="no_clear_stance"):
        free_base_stance(crowded, scene_plan=plan, packet_dir=tmp_path)
    # The launch fetches the team's model, finds this stance and seals it into the binding.
    from blueprint_pipeline import team_candidate_quick10_launch as launch

    (tmp_path / "native_task_arena_scene_plan.v1.json").write_text(json.dumps(plan))
    sealed = launch._stage_robot_model(job_dir=tmp_path / "job", manifest=declared,
                                       facts={"packet_dir": str(tmp_path)},
                                       fetch=lambda url, target: target.write_bytes(model.read_bytes()))
    assert sealed["stance"] == stance and sealed["manifest"]["base"]["mode"] == "free"
