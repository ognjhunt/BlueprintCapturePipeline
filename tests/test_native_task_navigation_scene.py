"""A scene kept as filmed runs as a navigation task: no task object, contact measured off the feet.

Part B of docs/G1_NAVIGATION_FROM_COMPILED_SPEC.md: the ``goal_region_navigation`` kind compiles a
runtime contract and a scene plan with only the scene's appearance and collision, the runtime builds
it (scene, robot, the yellow goal disc) and the readback reports robot-scene contact on every robot
body except the ones it stands on. The rigid and articulated paths still refuse a missing task object.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.gear_sonic_joint_order_contract import PROTOCOL_V4_FULL_JOINT_ORDER
from blueprint_pipeline.native_g1_embodiment import ACTION_INTERFACE as G1_ACTION_INTERFACE
from blueprint_pipeline.native_g1_navigation_goal import PUBLISHED_TASK_INSTRUCTION
from blueprint_pipeline.native_task_arena_readback import (
    NativeNavigationArenaReadback,
    NativeRigidTaskArenaReadback,
    NativeTaskArenaReadbackError,
)
from blueprint_pipeline.native_task_arena_runtime import (
    NativeTaskArenaRuntimeError,
    build_native_task_arena_environment,
    required_contact_channels,
    validate_contact_sensor_plan,
)
from blueprint_pipeline.native_task_arena_scene_plan import (
    GROUND_SUPPORT_BODY_NAMES,
    NativeTaskArenaScenePlanError,
    _articulation_plan,
    materialize_native_task_arena_scene_plan,
)
from blueprint_pipeline.native_task_runtime_contract import (
    NAVIGATION_TASK_KIND,
    NativeTaskRuntimeContractError,
    load_native_task_runtime_contract,
    materialize_native_task_runtime_contract,
)
from blueprint_pipeline.website_navigation_packet import navigation_task_spec
from tests.test_native_task_appearance_frame_alignment import write_appearance_usdz
from tests.test_website_navigation_packet import _context, _spec

ROBOT_BODIES = (
    "pelvis",
    "torso_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "left_hand_palm_link",
    "right_hand_palm_link",
)
FEET = {
    "{ENV_REGEX_NS}/Robot/" + name
    for name in ("left_ankle_pitch_link", "left_ankle_roll_link", "right_ankle_pitch_link", "right_ankle_roll_link")
}
TORSO = "{ENV_REGEX_NS}/Robot/torso_link"


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> dict:
    return {"position_world_m": [x, y, z], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}


def _camera(role: str) -> dict:
    head = role == "head"
    return {
        "role": role,
        "policy_input": head,
        "scoring_input": False,
        "pose_frame": "robot_body" if head else "world",
        "parent_prim_path": "{ENV_REGEX_NS}/Robot/head_link" if head else "{ENV_REGEX_NS}",
        "optical_convention": "opencv",
        "frame_from_camera_matrix": [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0,
                                     0.0, 0.0, 0.0, 1.0],
        "intrinsics": {"fx": 300.0, "fy": 300.0, "cx": 159.5, "cy": 119.5, "width": 320, "height": 240},
    }


def _robot_usd(path: Path, bodies=ROBOT_BODIES) -> Path:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/G1")
    stage.SetDefaultPrim(root.GetPrim())
    for name in bodies:
        body = UsdGeom.Xform.Define(stage, f"/G1/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        UsdPhysics.CollisionAPI.Apply(UsdGeom.Cube.Define(stage, f"/G1/{name}/collider").GetPrim())
    stage.GetRootLayer().Save()
    return path


def _room_usd(path: Path) -> Path:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Room")
    stage.SetDefaultPrim(root.GetPrim())
    # A filmed room is one collider: floor and shelves together.
    UsdPhysics.CollisionAPI.Apply(UsdGeom.Mesh.Define(stage, "/Room/room").GetPrim())
    stage.GetRootLayer().Save()
    return path


def _robot(asset: Path, start: list[float]) -> dict:
    joints = {name: 0.0 for name in PROTOCOL_V4_FULL_JOINT_ORDER}
    return {
        "robot_id": "unitree_g1",
        "hand_id": "unitree_dex3_1",
        "action_interface": G1_ACTION_INTERFACE,
        "usd_path": str(asset),
        "usd_sha256": _sha(asset),
        "base_pose_world": _pose(start[0], start[1], 0.8),
        "joint_reset_positions_rad": joints,
        "joint_position_limits_rad": {name: [-1.0, 1.0] for name in joints},
        "actuator_parameters": {
            name: {"stiffness": 100, "damping": 2, "effort_limit": 25, "velocity_limit": 10} for name in joints
        },
        "task_contact_body_paths": [
            "{ENV_REGEX_NS}/Robot/left_hand_palm_link",
            "{ENV_REGEX_NS}/Robot/right_hand_palm_link",
        ],
    }


def _navigation_task() -> dict:
    spec = _spec()
    return navigation_task_spec(spec, task_context=_context(spec))


def _fixture(tmp_path: Path, *, task_spec: dict | None = None) -> tuple[dict, Path]:
    """The contract inputs for the compiled corridor scene: appearance, collision, the G1."""

    task = task_spec or _navigation_task()
    directory = tmp_path / "assets"
    directory.mkdir(exist_ok=True)
    collision = _room_usd(directory / "scene_collision.usda")
    rng = numpy.random.default_rng(20261106)
    positions = numpy.stack(
        [rng.uniform(-1.0, 4.0, 512), rng.uniform(-2.0, 2.0, 512), rng.uniform(0.0, 2.5, 512)], axis=1
    ).astype(numpy.float32)
    appearance = write_appearance_usdz(directory / "scene_appearance.usdz", positions)
    robot = _robot(_robot_usd(directory / "g1.usda"), task["start_pose_world"])
    inputs = {
        "scene_id": task["site_id"],
        "task_id": task["task_id"],
        "task_spec": task,
        "assets": [
            {"semantic_role": "scene_collision", "filename": collision.name, "sha256": _sha(collision),
             "pose_world": _pose()},
            {"semantic_role": "scene_appearance", "filename": appearance.name, "sha256": _sha(appearance),
             "pose_world": _pose()},
        ],
        "robot_base_pose_world": robot["base_pose_world"],
        "robot_joint_reset_positions_rad": robot["joint_reset_positions_rad"],
        "robot_configuration": robot,
        "cameras": [_camera("head"), _camera("overview")],
        "scenario_cell_id": "canonical_seed_17",
        "scenario_instance_digest": "sha256:" + "d" * 64,
        "scenario_context_kind": "evaluation_cell",
        "seed": 17,
    }
    return inputs, directory


def _plan(tmp_path: Path) -> dict:
    inputs, directory = _fixture(tmp_path)
    contract = materialize_native_task_runtime_contract(**inputs)
    return materialize_native_task_arena_scene_plan(
        runtime_contract=contract, provider_asset_directory=directory, physics_frequency_hz=200
    )


# --- runtime contract ------------------------------------------------------------------------


def test_a_navigation_contract_needs_no_task_object(tmp_path: Path) -> None:
    inputs, _ = _fixture(tmp_path)
    contract = materialize_native_task_runtime_contract(**inputs, destination=tmp_path / "contract.json")

    assert contract["task_kind"] == NAVIGATION_TASK_KIND
    assert contract["task_subject_asset_id"] is None
    assert [row["semantic_role"] for row in contract["objects"]] == ["scene_collision", "scene_appearance"]
    assert not any(row["task_subject"] for row in contract["objects"])
    assert contract["task_sample_binding"]["subject"] == "robot_root"
    assert contract["task_state_binding"] is None
    assert contract["reset_contract"]["per_object_reset_states"] == {}
    assert load_native_task_runtime_contract(tmp_path / "contract.json") == contract


@pytest.mark.parametrize(
    "change,code",
    [
        ("task_object", "native_task_runtime_navigation_task_asset_unexpected:task_object"),
        ("task_support", "native_task_runtime_navigation_task_asset_unexpected:task_support"),
        ("no_goal", "native_task_runtime_navigation_goal_invalid"),
        ("no_collision", "native_task_runtime_asset_missing:scene_collision"),
    ],
)
def test_a_navigation_contract_refuses_task_assets_and_a_missing_goal(tmp_path: Path, change: str,
                                                                    code: str) -> None:
    inputs, _ = _fixture(tmp_path)
    if change in {"task_object", "task_support"}:
        inputs["assets"].append({"semantic_role": change, "filename": f"{change}.usda", "sha256": "sha256:" + "c" * 64,
                                 "pose_world": _pose(1.0, 0.0, 0.0), "object_type": "RIGID"})
    elif change == "no_goal":
        inputs["task_spec"] = {key: value for key, value in inputs["task_spec"].items() if key != "g1_navigation_goal"}
    else:
        inputs["assets"] = inputs["assets"][1:]
    with pytest.raises(NativeTaskRuntimeContractError) as excinfo:
        materialize_native_task_runtime_contract(**inputs)
    assert code in excinfo.value.errors


@pytest.mark.parametrize("task_kind", ["rigid_pick_place", "articulated_open_close"])
def test_rigid_and_articulated_contracts_still_refuse_a_missing_task_object(tmp_path: Path, task_kind: str) -> None:
    inputs, _ = _fixture(tmp_path)
    inputs["task_spec"] = {"task_kind": task_kind, "prompt": "Move the object."}
    with pytest.raises(NativeTaskRuntimeContractError) as excinfo:
        materialize_native_task_runtime_contract(**inputs)
    assert "native_task_runtime_asset_missing:task_object" in excinfo.value.errors


# --- scene plan -------------------------------------------------------------------------------


def test_a_navigation_scene_compiles_with_robot_scene_contact_and_sealed_feet(tmp_path: Path) -> None:
    plan = _plan(tmp_path)

    assert plan["task_kind"] == NAVIGATION_TASK_KIND
    assert plan["task_object_observability"] is None
    assert plan["reset"]["task_joint_positions_rad"] == {}
    assert set(plan["appearance_frame_alignment"]["spawned_frame_contains"]) == {"robot_base"}
    sensors = plan["articulation"]["contact_sensors"]
    # One existing robot_scene_contact sensor per robot body, against the one room collider.
    assert {row["logical_sensor_id"] for row in sensors} == {"robot_scene_contact"}
    assert sorted(row["prim_path"] for row in sensors) == sorted(
        "{ENV_REGEX_NS}/Robot/" + name for name in ROBOT_BODIES
    )
    assert {tuple(row["filter_prim_paths_expr"]) for row in sensors} == {("{ENV_REGEX_NS}/scene_collision/room",)}
    assert set(plan["articulation"]["ground_support_robot_body_paths"]) == FEET
    assert plan["articulation"]["ground_support_rule"]["body_names"] == list(GROUND_SUPPORT_BODY_NAMES["unitree_g1"])
    assert plan["plan_digest"] == canonical_digest(plan, digest_field="plan_digest")
    assert validate_contact_sensor_plan(plan) == {"robot_scene_contact": [row["sensor_instance_id"] for row in sensors]}


def test_the_goal_and_the_feet_rule_must_resolve_on_the_sealed_plan(tmp_path: Path) -> None:
    room = _room_usd(tmp_path / "room.usda")
    contract = {"task_kind": NAVIGATION_TASK_KIND, "task_spec": {"task_kind": NAVIGATION_TASK_KIND},
                "robot": _robot(_robot_usd(tmp_path / "g1_full.usda"), [0.0, 0.0]), "objects": []}
    with pytest.raises(NativeTaskArenaScenePlanError, match="navigation_goal_invalid"):
        _articulation_plan(contract, task_object_asset_path=None, scene_collision_asset_path=room)
    contract["task_spec"] = _navigation_task()
    assert _articulation_plan(contract, task_object_asset_path=None, scene_collision_asset_path=room)[
        "ground_support_robot_body_paths"] == sorted(FEET)
    contract["robot"] = {"robot_id": "franka_panda"}
    with pytest.raises(NativeTaskArenaScenePlanError, match="ground_support_rule_missing:franka_panda"):
        _articulation_plan(contract, task_object_asset_path=None, scene_collision_asset_path=room)
    without_ankle = _robot_usd(tmp_path / "g1.usda", bodies=[b for b in ROBOT_BODIES if b != "right_ankle_pitch_link"])
    contract["robot"] = _robot(without_ankle, [0.0, 0.0])
    with pytest.raises(NativeTaskArenaScenePlanError, match="ground_support_body_unresolved:right_ankle_pitch_link"):
        _articulation_plan(contract, task_object_asset_path=None, scene_collision_asset_path=room)
    with pytest.raises(NativeTaskArenaScenePlanError, match="scene_collision_asset_missing"):
        _articulation_plan(contract, task_object_asset_path=None, scene_collision_asset_path=None)


# --- runtime ----------------------------------------------------------------------------------


def test_navigation_needs_only_robot_scene_contact_and_rigid_channels_are_unchanged() -> None:
    assert required_contact_channels(NAVIGATION_TASK_KIND) == {"robot_scene_contact"}
    assert required_contact_channels("rigid_pick_place") == {
        "task_robot_contact", "task_support_contact", "robot_scene_contact"}
    assert required_contact_channels("articulated_open_close") == {
        "task_robot_contact", "task_scene_contact", "robot_scene_contact"}


def test_a_navigation_plan_admits_no_task_contact_channel(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    extra = json.loads(json.dumps(plan))
    extra["articulation"]["contact_sensors"].append({
        "sensor_instance_id": "task_robot_contact__00", "logical_sensor_id": "task_robot_contact",
        "prim_path": "{ENV_REGEX_NS}/task_object/body", "filter_prim_paths_expr": [TORSO]})
    with pytest.raises(NativeTaskArenaRuntimeError, match="contact_sensor_channels_incomplete"):
        validate_contact_sensor_plan(extra)
    # The same optional channel stays admissible on a rigid plan.
    extra["task_kind"] = "rigid_pick_place"
    extra["articulation"]["contact_sensors"].extend([
        {**extra["articulation"]["contact_sensors"][-1], "sensor_instance_id": f"{channel}__00",
         "logical_sensor_id": channel} for channel in ("task_support_contact", "task_scene_collision")])
    assert set(validate_contact_sensor_plan(extra)) == {
        "robot_scene_contact", "task_robot_contact", "task_support_contact", "task_scene_collision"}


def _install_navigation_runtime(monkeypatch) -> None:
    from blueprint_pipeline import native_task_arena_import_scope, native_task_robot_registry
    from tests.test_native_task_arena_runtime import _Embodiment, _install_fake_native_runtime

    _install_fake_native_runtime(monkeypatch)
    # The G1 embodiment needs Arena itself; the scene, marker and sensors are what is under test.
    monkeypatch.setattr(native_task_arena_import_scope, "install_scoped_arena_embodiment", lambda robot_id: {})
    monkeypatch.setattr(
        native_task_robot_registry, "build_native_robot_embodiment",
        lambda robot, *, enable_cameras, pose_class: _Embodiment(
            enable_cameras=enable_cameras, initial_pose=None, initial_joint_pose=None),
    )


def _sensor(force):
    return SimpleNamespace(data=SimpleNamespace(force_matrix_w=force))


def _scene(plan: dict, forces: dict[str, list[float]]) -> dict:
    """Live sensors by scene name; ``forces`` maps a robot body path to its contact force."""

    return {
        row["sensor_instance_id"]: _sensor([[[forces.get(row["prim_path"], [0.0, 0.0, 0.0])]]])
        for row in plan["articulation"]["contact_sensors"]
    }


def test_the_runtime_builds_the_navigation_scene_and_it_reads_back(tmp_path: Path, monkeypatch) -> None:
    from tests.test_native_task_arena_runtime import _ArenaBuilder

    import sys

    _install_navigation_runtime(monkeypatch)
    cylinders = []

    def cylinder(**kwargs):
        cylinders.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(sys.modules["isaaclab.sim"], "CylinderCfg", cylinder)
    plan = _plan(tmp_path)
    built = build_native_task_arena_environment(plan, enable_cameras=False)

    assert set(built.scene_asset_names) == {"scene_collision", "scene_appearance", "g1_navigation_goal_marker"}
    assets = {asset.name: asset for asset in _ArenaBuilder.last.arena_env.scene.assets}
    assert assets["scene_collision"].spawn_cfg_addon["visible"] is False
    assert assets["scene_appearance"].spawn_cfg_addon["visible"] is True
    # Only the yellow goal disc: the task's own cue is absent, as is any task object.
    (marker,) = cylinders
    goal = plan["task_spec"]["g1_navigation_goal"]
    assert marker["radius"] == goal["visible_target_marker"]["radius_m"]
    assert marker["collision_props"] is None
    assert marker["visual_material"].diffuse_color == pytest.approx((0.95, 0.78, 0.04))
    assert marker["semantic_tags"] == [("class", "navigation_goal_marker")]
    sensor_ids = [row["sensor_instance_id"] for row in plan["articulation"]["contact_sensors"]]
    assert built.contact_sensor_names == {"robot_scene_contact": tuple(sensor_ids)}
    assert all(assets[name].object_cfg.filter_prim_paths_expr == ["{ENV_REGEX_NS}/scene_collision/room"]
               for name in sensor_ids)

    # Readback needs no object pose: the live scene holds only the sensors.
    feet_only = dataclasses.replace(built, env=SimpleNamespace(scene=_scene(plan, {path: [0.0, 0.0, 400.0] for path in FEET})))
    assert NativeNavigationArenaReadback(feet_only).read_robot_scene_contact() == {
        "robot_scene_contact_peak_force_n": 0.0}
    torso = dataclasses.replace(built, env=SimpleNamespace(scene=_scene(
        plan, {**{path: [0.0, 0.0, 400.0] for path in FEET}, TORSO: [3.0, 4.0, 0.0]})))
    assert NativeNavigationArenaReadback(torso).read_robot_scene_contact() == {
        "robot_scene_contact_peak_force_n": 5.0}
    missing = dataclasses.replace(built, env=SimpleNamespace(scene={}))
    with pytest.raises(NativeTaskArenaReadbackError, match="contact_sensor_missing:robot_scene_contact"):
        NativeNavigationArenaReadback(missing).read_robot_scene_contact()
    with pytest.raises(NativeTaskArenaReadbackError, match="readback_task_kind_invalid"):
        NativeRigidTaskArenaReadback(built)


def test_the_articulated_runtime_still_refuses_a_missing_task_object(monkeypatch) -> None:
    from tests.test_native_task_arena_runtime import _install_fake_native_runtime, _sealed_scene_plan

    _install_fake_native_runtime(monkeypatch)
    plan = _sealed_scene_plan()
    plan["objects"] = [row for row in plan["objects"] if row["semantic_role"] != "task_object"]
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    with pytest.raises(NativeTaskArenaRuntimeError, match="native_task_arena_task_object_missing"):
        build_native_task_arena_environment(plan)


# --- readback ---------------------------------------------------------------------------------


def _readback_built(*, sensors: bool, forces: dict | None = None, task_kind: str = NAVIGATION_TASK_KIND):
    rows = [
        {"sensor_instance_id": f"robot_scene_contact__{index:02d}", "logical_sensor_id": "robot_scene_contact",
         "prim_path": path, "filter_prim_paths_expr": ["{ENV_REGEX_NS}/scene_collision/room"]}
        for index, path in enumerate(sorted({*FEET, TORSO}))
    ] if sensors else []
    plan = {"task_kind": task_kind, "robot": {"robot_id": "unitree_g1"},
            "articulation": {"contact_sensors": rows,
                             **({"ground_support_robot_body_paths": sorted(FEET)} if sensors else {})}}
    return SimpleNamespace(
        plan=plan,
        env=SimpleNamespace(scene=_scene(plan, forces or {})),
        contact_sensor_names={"robot_scene_contact": tuple(row["sensor_instance_id"] for row in rows)} if rows else {},
    )


def test_foot_contact_is_support_and_any_other_body_counts() -> None:
    feet = {path: [0.0, 0.0, 650.0] for path in FEET}
    assert NativeNavigationArenaReadback(_readback_built(sensors=True, forces=feet)).read_robot_scene_contact() == {
        "robot_scene_contact_peak_force_n": 0.0}
    torso = {**feet, TORSO: [0.0, 12.0, 0.0]}
    assert NativeNavigationArenaReadback(_readback_built(sensors=True, forces=torso)).read_robot_scene_contact() == {
        "robot_scene_contact_peak_force_n": 12.0}


def test_no_planned_sensors_means_no_contact_field() -> None:
    # Unmeasured is not clear: the scorer finds no field and leaves no_contact unscored.
    assert NativeNavigationArenaReadback(_readback_built(sensors=False)).read_robot_scene_contact() == {}


def test_planned_sensors_need_the_sealed_feet_rule_and_the_navigation_kind() -> None:
    built = _readback_built(sensors=True)
    del built.plan["articulation"]["ground_support_robot_body_paths"]
    with pytest.raises(NativeTaskArenaReadbackError, match="ground_support_rule_missing"):
        NativeNavigationArenaReadback(built)
    with pytest.raises(NativeTaskArenaReadbackError, match="readback_task_kind_invalid"):
        NativeNavigationArenaReadback(_readback_built(sensors=True, task_kind="rigid_pick_place"))


# --- the G1 navigation sampler ----------------------------------------------------------------


_GOAL_TASK = {
    "task_kind": NAVIGATION_TASK_KIND,
    "g1_navigation_goal": {
        "schema_version": "native_g1_navigation_goal.v1",
        "center_world_m": [2.0, 0.0, 0.0],
        "acceptance_radius_m": 0.3,
        "max_root_height_drift_m": 0.2,
        "settle_window_samples": 2,
        "task_instruction": PUBLISHED_TASK_INSTRUCTION,
        "visible_target_marker": {
            "schema_version": "native_task_target_marker.v1", "shape": "flat_yellow_disc", "non_colliding": True,
            "surface_position_world_m": [2.0, 0.0, 0.0], "radius_m": 0.4,
        },
    },
}


def _run_navigation(tmp_path: Path, monkeypatch, *, task_kind: str, sensors: bool,
                    candidate: str = "humanoidarena_dp_g1_dex3_sonic_vision_navi") -> dict:
    from blueprint_pipeline import native_g1_joint_episode_environment as g1_environment
    from blueprint_pipeline import native_g1_run_preflight
    from blueprint_pipeline.native_g1_shared_scene_episode import run_g1_built_scene_policy_episode
    from tests.test_native_g1_shared_scene_episode import _Bridge, _Policy, _Scene

    scene = _Scene()
    built = _readback_built(sensors=sensors, task_kind=task_kind)
    task = {**_GOAL_TASK, "task_kind": task_kind}
    if task_kind == "rigid_pick_place":
        task["prompt"] = "pick the box"
    scene.plan = {**scene.plan, **built.plan, "task_kind": task_kind, "task_spec": task}
    built.plan = scene.plan
    scene.read_state = lambda: {
        "step_index": scene.step,
        "root_position_world_m": [0.0 if scene.step == 0 else 1.8 + 0.1 * (scene.step - 1), 0.0, 0.85],
    }
    # The torso brushes a shelf on the second step; the feet carry the robot throughout.
    class LiveSensor:
        def __init__(self, path: str) -> None:
            self.path = path

        @property
        def data(self):
            if self.path in FEET:
                force = [0.0, 0.0, 500.0]
            else:
                force = [0.0, 7.0 if self.path == TORSO and scene.step == 2 else 0.0, 0.0]
            return SimpleNamespace(force_matrix_w=[[[force]]])

    for row in built.plan["articulation"]["contact_sensors"]:
        built.env.scene[row["sensor_instance_id"]] = LiveSensor(row["prim_path"])

    class NavigationPolicy(_Policy):
        def infer_chunk(self, *, front_rgb, observation_state, task):
            assert task == PUBLISHED_TASK_INSTRUCTION
            self.queries += 1
            return [[0.0] * 40, [0.0] * 40]

    monkeypatch.setattr(native_g1_run_preflight, "preflight_g1_shared_scene_run", lambda **kwargs: {
        "status": "staged_inputs_verified", "robot_id": "unitree_g1", "candidate_id": candidate,
        "scene_plan_digest": scene.plan["plan_digest"],
        "policy_role": "movement_navigation" if candidate.endswith("_vision_navi") else "manipulation"})
    monkeypatch.setattr(g1_environment, "NativeG1JointEpisodeEnvironment", lambda **kwargs: scene)
    run_g1_built_scene_policy_episode(
        built=built, policy_client=NavigationPolicy(), sonic_bridge=_Bridge(), candidate_id=candidate,
        max_steps=2, output_dir=tmp_path, preflight_inputs={}, to_tensor=lambda value: value,
        make_action_tensor=lambda value, **kwargs: value)
    return json.loads((tmp_path / "native_g1_shared_scene_episode_trace.v1.json").read_text())


def test_the_navigation_sampler_records_off_foot_contact_each_step(tmp_path: Path, monkeypatch) -> None:
    trace = _run_navigation(tmp_path, monkeypatch, task_kind=NAVIGATION_TASK_KIND, sensors=True)
    samples = [trace["initial_task_sample"], *(row["task_sample"] for row in trace["steps"])]
    assert [sample["robot_scene_contact_peak_force_n"] for sample in samples] == [0.0, 0.0, 7.0]
    assert [sample["step_index"] for sample in samples] == [0, 1, 2]


@pytest.mark.parametrize("task_kind,sensors", [(NAVIGATION_TASK_KIND, False), ("rigid_pick_place", True)])
def test_without_planned_sensors_or_on_a_rigid_host_the_field_is_absent(tmp_path: Path, monkeypatch, task_kind: str,
                                                                      sensors: bool) -> None:
    trace = _run_navigation(tmp_path, monkeypatch, task_kind=task_kind, sensors=sensors)
    samples = [trace["initial_task_sample"], *(row["task_sample"] for row in trace["steps"])]
    assert all(set(sample) == {"step_index", "root_position_world_m"} for sample in samples)


def test_a_navigation_scene_runs_only_the_navigation_checkpoints(tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(ValueError, match="g1_built_scene_policy_configuration_invalid"):
        _run_navigation(tmp_path, monkeypatch, task_kind=NAVIGATION_TASK_KIND, sensors=True,
                        candidate="humanoidarena_dp_g1_dex3_sonic")
