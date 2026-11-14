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
    episode_visual_evidence,
)
from blueprint_pipeline.adp009d_task_scoring import SUPPORT_PLANE_Z_M
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_runtime import camera_runtime_parameters
from blueprint_pipeline.team_embodiment import (
    TeamEmbodimentError,
    apply_team_embodiment,
    model_structure_blockers,
    read_model_structure,
    team_embodiment_binding,
    validate_embodiment_manifest,
    verify_model_file,
)
from blueprint_pipeline.team_policy_https_client import (
    METADATA_SCHEMA,
    TeamPolicyHttpsClient,
    TeamPolicySpec,
)

ARM = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"]
WHEELS, WHEEL_LINKS = ["left_wheel_joint", "right_wheel_joint"], ["left_wheel", "right_wheel"]
ORIGIN = "https://policy.acme-robotics.example"


_DESTINATION = [3.750152333333333, -3.4074919, SUPPORT_PLANE_Z_M]


def _scene_plan() -> dict[str, Any]:
    pose = {
        "position_world_m": [3.4681748, -3.3100837, SUPPORT_PLANE_Z_M],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    value: dict[str, Any] = {
        "schema_version": "native_task_arena_scene_plan.v1",
        "task_id": "scene-839873-mug-planar-push",
        "task_kind": "rigid_pick_place",
        # The robot station, as the scene compiler places Blueprint's robot (a team's arm stands here too).
        "robot": {"robot_id": "franka_panda", "base_pose_world": {
            "position_world_m": [2.9681748, -3.3100837, SUPPORT_PLANE_Z_M], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}},
        "objects": [
            {
                "name": "task_object",
                "task_subject": True,
                "pose_world": json.loads(json.dumps(pose)),
                "reset_state": {"root_pose_world": json.loads(json.dumps(pose))},
            },
            {
                "name": "scene_appearance",
                "semantic_role": "scene_appearance",
                "task_subject": False,
                "sha256": "1bfd4438e057587c785b8211a70e26b896dd1ef90626e7923c541dbfd7c125cc",
                "pose_world": json.loads(json.dumps(pose)),
            },
            {
                "name": "task_support",
                "semantic_role": "task_support",
                "asset_id": "document_tray",
                "task_subject": False,
                "object_type": "RIGID",
                "pose_world": {
                    "position_world_m": list(_DESTINATION),
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "reset_state": {
                    "root_pose_world": {
                        "position_world_m": list(_DESTINATION),
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                    "joint_positions": {},
                },
            },
        ],
        "appearance_frame_alignment": {
            "status": "aligned",
            "representation": "particlefield_3d_gaussian_splat",
            "measurement_authority": "particlefield_position_quantiles",
            "source_asset_sha256": (
                "sha256:9193a9de6bd81bd6348065b3cad46ad835b62dcfaa6212285a91bffd8a166445"
            ),
        },
        "cameras": [
            {
                "role": role,
                "frame_from_camera_matrix": [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
                "optical_convention": "opencv",
                "pose_frame": "robot_body" if role == "wrist" else "world",
                "parent_prim_path": "{ENV_REGEX_NS}/Robot/panda_hand" if role == "wrist" else "{ENV_REGEX_NS}",
                "intrinsics": {"fx": 500., "fy": 500., "cx": 639.5, "cy": 359.5, "width": 1280, "height": 720},
                "data_types": ["rgb", "semantic_segmentation"],
                "policy_input": role != "overview",
                "review_only": role == "overview",
            }
            for role in ("external", "wrist", "overview")
        ],
        # The compiled scene still carries the pre-canary 20 Hz cadence; the
        # worker must resolve it to the frozen DROID adapters' 15 Hz.
        "cadence": {
            "control_frequency_hz": 20.0,
            "physics_frequency_hz": 120.0,
            "physics_dt_seconds": 1.0 / 120.0,
            "control_decimation": 6,
            "maximum_action_steps": 240,
            "settle_window_samples": 1,
            "episode_length_seconds": 13.05,
        },
        "task_spec": {
            "schema_version": "adp_task_spec.v1",
            "task_kind": "rigid_pick_place",
            "start_pose_world": [
                *pose["position_world_m"],
                *pose["orientation_xyzw"],
            ],
            "destination_position_world_m": list(_DESTINATION),
            "support_plane_z_m": SUPPORT_PLANE_Z_M,
            "settle_window_samples": 1,
            "require_sealed_start_pose": True,
            "control_frequency_hz": 20.0,
            "maximum_action_steps": 240,
        },
        "plan_digest": "",
    }
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    return value


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
        self.root = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # Pinned Isaac Lab Beta2: position, then xyzw
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


def _team_loop(tmp_path, monkeypatch):
    declared, sealed = binding(tmp_path)
    plan = {**apply_team_embodiment(station_plan(), sealed), "scenario": {"seed": 7}}
    built = type("Built", (), {"env": _Env(), "camera_scene_names": {
        "team_camera_0": "team_camera_0", "team_camera_1": "team_camera_1", "overview": "external_camera_2"}})()
    monkeypatch.setattr(episode_visual_evidence, "finalize_multicamera_visual_evidence",
                        lambda **kwargs: ({"status": "complete"}, []))
    spec = TeamPolicySpec(policy_id="acme-arm-pick", endpoint_url=ORIGIN + "/v1/act",
                          metadata_url=ORIGIN + "/v1/metadata", allowed_origin=ORIGIN,
                          model_identity_digest="sha256:" + "7" * 64, embodiment_id=declared["embodiment_id"],
                          action_chunk_rows=4, open_loop_horizon=4, camera_names=["wrist_rgb", "scene_rgb"])
    return built, plan, spec


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
        self.robot.root = np.array([1.0, -1.0, 0.9, 0.0, 0.0, 0.0, 1.0])

    def step(self, action: list[list[float]]) -> None:
        targets = action[0]
        self.steps.append(list(targets))
        # Simulator order: arm positions, the finger's, then the wheels' velocities.
        self.robot.q = np.array([targets[3], *targets[:3], *targets[4:]])
        self.robot.root[2] = 0.9 if len(self.steps) < 5 else 0.2


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


def test_a_moving_task_part_keeps_the_station_out_of_its_way(tmp_path) -> None:
    from blueprint_pipeline.scene_physics_preflight import robot_fixed_volume_extent
    from blueprint_pipeline.team_embodiment_stance import ground_stance

    room = write_room_usd(tmp_path / "collision.usda")
    common = {"collision_path": room, "collision_sha256": "sha256:" + "0" * 64, "offset": [0.0, 0.0, 0.0],
              "subject_center": [2.0, 0.0, 0.8], "subject_minimum": [1.95, -0.05, 0.75],
              "subject_maximum": [2.05, 0.05, 0.85], "footprint_half_extent_m": [0.15, 0.15], "root_world_z_m": 0.75}

    def shoulder_slab(x: float, y: float) -> Any:
        # Samples of a part sweeping 0.3 m above the mount top, where the arm's shoulder stands.
        return np.array([[x + dx, y + dy, 1.05] for dx in np.linspace(-0.2, 0.2, 9) for dy in np.linspace(-0.2, 0.2, 9)])

    free = ground_stance(**common)
    path = shoulder_slab(*free["position_world_m"][:2])
    moved = ground_stance(**common, task_motion={"robot_id": "franka_panda", "path_points": path})
    assert moved["task_motion_clearance"]["struck_stances"] > 0
    nearest = np.min(np.hypot(path[:, 0] - moved["position_world_m"][0], path[:, 1] - moved["position_world_m"][1]))
    assert nearest > robot_fixed_volume_extent("franka_panda")
    # Among clear stances, one the part's whole range never reaches comes first.
    reached = shoulder_slab(*moved["position_world_m"][:2])
    preferred = ground_stance(**common, task_motion={"robot_id": "franka_panda", "path_points": path,
                                                     "limit_points": np.concatenate([path, reached])})
    assert preferred["position_world_m"] != moved["position_world_m"]
    assert preferred["task_motion_clearance"]["reached_beyond_task_range"] is False
    # A part that falls through a stance blocks it like its path does.
    fallen = ground_stance(**common, task_motion={"robot_id": "franka_panda", "path_points": path,
                                                  "fall_points": reached})
    assert fallen["position_world_m"] != moved["position_world_m"]
    with pytest.raises(TeamEmbodimentError, match="team_embodiment_task_motion_no_clear_stance"):
        everywhere = np.array([[x, y, 1.05] for x in np.arange(0.2, 3.8, 0.1) for y in np.arange(-1.8, 1.8, 0.1)])
        ground_stance(**common, task_motion={"robot_id": "franka_panda", "path_points": everywhere})


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


def bimanual_inputs(tmp_path, *, embodiment_id="acme-bimanual", arm_dof=3):
    """Development-only named two-arm USD and explicit independent tool channels."""
    from pxr import Usd, UsdPhysics

    model = tmp_path / "two_arms.usda"
    stage = Usd.Stage.CreateNew(str(model))
    stage.SetDefaultPrim(stage.DefinePrim("/robot", "Xform"))
    UsdPhysics.RigidBodyAPI.Apply(stage.DefinePrim("/robot/base", "Xform"))
    arm_names = [f"{side}_joint_{i}" for side in ("left", "right") for i in range(arm_dof)]
    for name in arm_names:
        UsdPhysics.RevoluteJoint.Define(stage, "/robot/" + name)
    for side in ("left", "right"):
        for link in (f"{side}_tool", f"{side}_finger_a", f"{side}_finger_b"):
            UsdPhysics.RigidBodyAPI.Apply(stage.DefinePrim("/robot/" + link, "Xform"))
        UsdPhysics.PrismaticJoint.Define(stage, f"/robot/{side}_finger_joint")
    UsdPhysics.FixedJoint.Define(stage, "/robot/anchor").CreateBody1Rel().SetTargets(["/robot/base"])
    stage.GetRootLayer().Save()
    declared = manifest(model, schema_version="blueprint.robot_embodiment.v2", embodiment_id=embodiment_id,
                        form="bimanual_arm", tool_link="left_tool",
                        joints=[{"name": name, "control": "position", "lower": -3.0, "upper": 3.0,
                                 "max_velocity": 2.0} for name in arm_names],
                        reset_joint_positions={**dict.fromkeys(arm_names, 0.0),
                                               "left_finger_joint": 0.04, "right_finger_joint": 0.03})
    declared.pop("gripper")
    declared["grippers"] = [
        {"name": side, "tool_link": f"{side}_tool",
         "joints": [{"name": f"{side}_finger_joint", "open": opening, "closed": 0.0}],
         "max_width_m": opening * 2, "contact_links": [f"{side}_finger_a", f"{side}_finger_b"]}
        for side, opening in (("left", 0.04), ("right", 0.03))]
    declared["cameras"][0]["mount_link"] = "left_tool"
    declared["cameras"].insert(1, {**copy.deepcopy(declared["cameras"][0]), "name": "right_wrist_rgb",
                                  "mount_link": "right_tool"})
    return model, declared


def test_v2_binds_both_tools_and_all_cameras_without_changing_v1(tmp_path):
    from blueprint_pipeline.team_embodiment import action_dimension
    from blueprint_pipeline.team_embodiment_runtime import action_joint_groups

    model, value = bimanual_inputs(tmp_path)
    declared = validate_embodiment_manifest(value)
    assert "gripper" not in declared and action_dimension(declared) == 8
    sealed = team_embodiment_binding(declared, model_record=verify_model_file(declared, model),
                                     structure=read_model_structure(model), scene_plan=station_plan())
    plan = apply_team_embodiment(station_plan(), sealed)
    channels = plan["robot"]["gripper_channels"]
    assert [row["name"] for row in channels] == ["left", "right"]
    assert action_joint_groups(plan["robot"])[0] == [j["name"] for j in value["joints"]] + [
        "left_finger_joint", "right_finger_joint"]
    assert plan["robot"]["grasp_frame"]["body_names"] == ["left_finger_a", "left_finger_b"]
    assert [camera["parent_prim_path"] for camera in plan["cameras"][:2]] == [
        "{ENV_REGEX_NS}/Robot/left_tool", "{ENV_REGEX_NS}/Robot/right_tool"]
    sensors = {row["sensor_instance_id"]: row for row in plan["articulation"]["contact_sensors"]}
    for channel in channels:
        sensor = sensors[channel["task_contact_sensor_instances"][0]]
        assert sensor["filter_prim_paths_expr"] == channel["task_contact_body_paths"]
        assert all(channel["name"] in path for path in sensor["filter_prim_paths_expr"])
    assert validate_embodiment_manifest(declared) == declared


@pytest.mark.parametrize("fault", ["same_name", "same_tool", "primary_tool", "shared_joint", "missing_reset", "one_tool"])
def test_v2_refuses_ambiguous_tool_channels(tmp_path, fault):
    _model, value = bimanual_inputs(tmp_path)
    if fault == "same_name":
        value["grippers"][1]["name"] = "left"
    elif fault == "same_tool":
        value["grippers"][1]["tool_link"] = "left_tool"
    elif fault == "primary_tool":
        value["tool_link"] = "absent_tool"
    elif fault == "shared_joint":
        value["grippers"][1]["joints"][0]["name"] = "left_finger_joint"
    elif fault == "missing_reset":
        value["reset_joint_positions"].pop("right_finger_joint")
    else:
        value["grippers"].pop()
    with pytest.raises(TeamEmbodimentError):
        validate_embodiment_manifest(value)


def test_v1_manifest_and_resolved_plan_keep_pre_v2_digest_identities(tmp_path, monkeypatch):
    # Recorded from e02bc17 before the v2 extension, with a fixed model identity
    # and byte size so generated USD header details do not affect the contract.
    # libm tan(pi/6) differs by one ULP between macOS and Linux. Freeze that
    # fixture input too; production camera values and receipt hashes are never
    # rounded or changed. Separate camera tests exercise the actual math.
    monkeypatch.setattr("blueprint_pipeline.team_embodiment.math.tan", {
        math.pi / 4: 0.9999999999999999,
        math.pi / 6: 0.5773502691896256,
    }.__getitem__)
    model = write_robot_usd(tmp_path / "legacy.usda")
    value = manifest(model)
    value["model"]["sha256"] = "sha256:" + "0" * 64
    declared = validate_embodiment_manifest(value)
    assert declared["embodiment_digest"] == "sha256:19a70b1ec5353eaf4df917985f4da0d0f7fb970baab9b54e80a8a7dfd0cf6e74"
    sealed = team_embodiment_binding(declared, model_record={"sha256": value["model"]["sha256"], "size_bytes": 100},
                                     structure=read_model_structure(model), scene_plan=station_plan())
    plan = apply_team_embodiment(station_plan(), sealed)
    assert canonical_digest(plan) == "sha256:e598f183812e169b7b8d2e36bd56a61d0c6717e77284f7a572c38bc0934c601d"
    assert "gripper_channels" not in plan["robot"]


@pytest.mark.parametrize("v2", [False, True])
def test_native_builder_enables_self_collision_for_v2_without_changing_v1(tmp_path, monkeypatch, v2):
    import sys
    from types import ModuleType, SimpleNamespace
    from blueprint_pipeline.team_embodiment_runtime import build_team_embodiment

    if v2:
        model, value = bimanual_inputs(tmp_path)
        declared = validate_embodiment_manifest(value)
        sealed = team_embodiment_binding(declared, model_record=verify_model_file(declared, model),
                                         structure=read_model_structure(model), scene_plan=station_plan())
    else:
        _declared, sealed = binding(tmp_path)
    plan = apply_team_embodiment(station_plan(), sealed)
    captured = []

    class CapturedConfiguration(Exception):
        pass

    class Config:
        def __init__(self, *args, **kwargs):
            self.__dict__.update(kwargs)

    def root_config(**kwargs):
        captured.append(kwargs)
        raise CapturedConfiguration

    modules = {
        "isaaclab.sim": {"UsdFileCfg": Config, "RigidBodyPropertiesCfg": Config,
                         "ArticulationRootPropertiesCfg": root_config},
        "isaaclab.actuators": {"ImplicitActuatorCfg": Config},
        "isaaclab.assets.articulation.articulation_cfg": {"ArticulationCfg": Config},
        "isaaclab.envs": {"mdp": SimpleNamespace()},
        "isaaclab.managers": dict.fromkeys(
            ("EventTermCfg", "ObservationGroupCfg", "ObservationTermCfg", "SceneEntityCfg"), Config),
        "isaaclab.sensors.camera.tiled_camera_cfg": {"TiledCameraCfg": Config},
        "isaaclab.utils": {"configclass": lambda cls: cls},
        "isaaclab_arena.embodiments.embodiment_base": {"EmbodimentBase": Config},
    }
    for name, values in modules.items():
        parent = None
        for index, piece in enumerate(name.split(".")):
            full = ".".join(name.split(".")[:index + 1])
            if full not in sys.modules:
                module = ModuleType(full)
                module.__path__ = []
                monkeypatch.setitem(sys.modules, full, module)
            module = sys.modules[full]
            if parent is not None:
                monkeypatch.setattr(parent, piece, module, raising=False)
            parent = module
        for key, value in values.items():
            monkeypatch.setattr(parent, key, value, raising=False)
    with pytest.raises(CapturedConfiguration):
        build_team_embodiment(plan["robot"], enable_cameras=True, pose_class=Config)
    assert captured[0]["enabled_self_collisions"] is v2
    if v2:
        assert plan["policy_canary_embodiment_profile"]["claim_boundary"]["self_collision_mode"] == {
            "enabled": True, "force_readback": "task_and_scene_only_no_arm_arm_force_channel", "qualified": False}


@pytest.mark.parametrize("body0,enabled,anchored", [
    (None, True, True),  # empty body0 (Isaac's apply_fix_base fix_base_joint)
    ("/acme_arm", True, True),  # Isaac 6 / urdf-usd-converter: default prim behind a massless root link
    ("/acme_arm/mount", True, True),  # Isaac 6: a massless URDF link left as a plain Xform
    ("/acme_arm/upper_arm/flange", True, False),  # a frame that moves with a rigid body
    ("/acme_arm/upper_arm", True, False),  # two rigid bodies: an internal joint
    ("/acme_arm/missing", True, False),  # a dangling target pins nothing
    ("/acme_arm", False, False),  # a disabled joint pins nothing
])
def test_world_anchor_is_a_fixed_joint_between_the_static_world_and_one_rigid_body(tmp_path, body0, enabled, anchored):
    from pxr import Usd, UsdPhysics

    path = write_robot_usd(tmp_path / "robot.usda", anchored=False)
    stage = Usd.Stage.Open(str(path))
    stage.DefinePrim("/acme_arm/mount", "Xform")
    stage.DefinePrim("/acme_arm/upper_arm/flange", "Xform")
    joint = UsdPhysics.FixedJoint.Define(stage, "/acme_arm/joints/station_mount")
    joint.CreateBody1Rel().SetTargets(["/acme_arm/base_link"])
    if body0 is not None:
        joint.CreateBody0Rel().SetTargets([body0])
    if not enabled:
        joint.CreateJointEnabledAttr(False)
    stage.GetRootLayer().Save()
    structure = read_model_structure(path)
    assert structure["world_anchor_links"] == (["base_link"] if anchored else [])
    declared = validate_embodiment_manifest(manifest(path))
    blockers = model_structure_blockers(declared, structure)
    assert ("team_embodiment_model_base_not_anchored_to_world" in blockers) is not anchored


def test_a_fixed_body_bound_to_its_own_station_stands_there_in_every_cell(tmp_path) -> None:
    model = write_robot_usd(tmp_path / "acme_arm.usda")
    declared = validate_embodiment_manifest(manifest(model))
    inputs = {"model_record": verify_model_file(declared, model), "structure": read_model_structure(model)}
    plan = station_plan()
    base = plan["robot"]["base_pose_world"]
    station = {"position_world_m": [base["position_world_m"][0] + 0.1, base["position_world_m"][1] - 0.05,
                                    base["position_world_m"][2]], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}
    sealed = team_embodiment_binding(declared, scene_plan=plan, station=station, **inputs)
    assert sealed["station"] == station
    assert sealed["binding_digest"] == canonical_digest(dict(sealed), digest_field="binding_digest")
    # The plan itself still has the scene's station: the binding, not the plan, places the body.
    assert apply_team_embodiment(plan, sealed)["robot"]["base_pose_world"] == station
    unbound = team_embodiment_binding(declared, scene_plan=plan, **inputs)
    assert "station" not in unbound and apply_team_embodiment(plan, unbound)["robot"]["base_pose_world"] == base
    # Reach is checked from the station the body will stand at.
    far = {**station, "position_world_m": [station["position_world_m"][0] + 50.0, *station["position_world_m"][1:]]}
    with pytest.raises(TeamEmbodimentError, match="team_embodiment_station_out_of_reach"):
        team_embodiment_binding(declared, scene_plan=plan, station=far, **inputs)
    with pytest.raises(TeamEmbodimentError, match="team_embodiment_own_station_invalid"):
        team_embodiment_binding(declared, scene_plan=plan, station={"position_world_m": [0.0, 0.0]}, **inputs)


def test_a_station_the_bodys_own_arm_reached_from_is_bound_whatever_its_declared_reach(tmp_path) -> None:
    model = write_robot_usd(tmp_path / "acme_arm.usda")
    declared = validate_embodiment_manifest(manifest(model, reach_m=0.2))
    inputs = {"model_record": verify_model_file(declared, model), "structure": read_model_structure(model)}
    plan = station_plan()
    base = plan["robot"]["base_pose_world"]
    station = {"position_world_m": [base["position_world_m"][0] + 1.0, *base["position_world_m"][1:]],
               "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}
    # A declared 20 cm misses the task from there; the arm's own model reached it, so the run stands there.
    with pytest.raises(TeamEmbodimentError, match="team_embodiment_station_out_of_reach"):
        team_embodiment_binding(declared, scene_plan=plan, station=station, **inputs)
    sealed = team_embodiment_binding(declared, scene_plan=plan, station=station, station_reached_by_own_arm=True,
                                     **inputs)
    assert apply_team_embodiment(plan, sealed)["robot"]["base_pose_world"] == station
    # Only the declared-reach veto is lifted, and only for an own station.
    with pytest.raises(TeamEmbodimentError, match="team_embodiment_own_station_invalid"):
        team_embodiment_binding(declared, scene_plan=plan, station_reached_by_own_arm=True, **inputs)
