"""A scene kept as filmed runs as a navigation task: no task object, contact measured off the feet.

Part B of docs/G1_NAVIGATION_FROM_COMPILED_SPEC.md: the ``goal_region_navigation`` kind compiles a
runtime contract and a scene plan with only the scene's appearance and collision, the runtime builds
it (scene, robot, the yellow goal disc) and the readback reports robot-scene contact on every robot
body except the ones it stands on. The rigid and articulated paths still refuse a missing task object.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.gear_sonic_joint_order_contract import PROTOCOL_V4_FULL_JOINT_ORDER
from blueprint_pipeline.native_g1_embodiment import ACTION_INTERFACE as G1_ACTION_INTERFACE
from blueprint_pipeline.native_g1_navigation_goal import PUBLISHED_TASK_INSTRUCTION
from blueprint_pipeline.native_task_arena_readback import (
    NativeNavigationArenaReadback,
    NativeTaskArenaReadbackError,
)
from blueprint_pipeline.native_task_arena_runtime import (
    NativeTaskArenaRuntimeError,
    build_native_task_arena_environment,
    required_contact_channels,
)
from blueprint_pipeline.native_task_runtime_contract import (
    NAVIGATION_TASK_KIND,
)

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


def test_navigation_needs_only_robot_scene_contact_and_rigid_channels_are_unchanged() -> None:
    assert required_contact_channels(NAVIGATION_TASK_KIND) == {"robot_scene_contact"}
    assert required_contact_channels("rigid_pick_place") == {
        "task_robot_contact", "task_support_contact", "robot_scene_contact"}
    assert required_contact_channels("articulated_open_close") == {
        "task_robot_contact", "task_scene_contact", "robot_scene_contact"}


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
        "robot_scene_contact_peak_force_n": 0.0, "robot_scene_contact_excluded_bodies": sorted(FEET)}
    torso = {**feet, TORSO: [0.0, 12.0, 0.0]}
    assert NativeNavigationArenaReadback(_readback_built(sensors=True, forces=torso)).read_robot_scene_contact() == {
        "robot_scene_contact_peak_force_n": 12.0, "robot_scene_contact_excluded_bodies": sorted(FEET)}


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
                    candidate: str = "humanoidarena_dp_g1_dex3_sonic_vision_navi",
                    torso_force_at_step: dict | None = None) -> dict:
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
            elif self.path == TORSO and torso_force_at_step and scene.step in torso_force_at_step:
                force = torso_force_at_step[scene.step]
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
    # Every measured step names the feet it left out.
    assert all(sample["robot_scene_contact_excluded_bodies"] == sorted(FEET) for sample in samples)


def test_an_unreadable_contact_step_is_flagged_and_the_episode_keeps_going(tmp_path: Path, monkeypatch) -> None:
    trace = _run_navigation(tmp_path, monkeypatch, task_kind=NAVIGATION_TASK_KIND, sensors=True,
                            torso_force_at_step={1: [float("nan"), 0.0, 0.0]})
    samples = [trace["initial_task_sample"], *(row["task_sample"] for row in trace["steps"])]
    assert [sample["step_index"] for sample in samples] == [0, 1, 2]
    assert all("root_position_world_m" in sample for sample in samples)
    unread = samples[1]
    assert "robot_scene_contact_peak_force_n" not in unread
    assert unread["robot_scene_contact_unread"] == [
        f"native_task_arena_force_matrix_invalid:robot_scene_contact:{_sensor_id_for(TORSO)}"]
    assert samples[2]["robot_scene_contact_peak_force_n"] == 7.0 and "robot_scene_contact_unread" not in samples[2]


def _sensor_id_for(path: str) -> str:
    rows = _readback_built(sensors=True).plan["articulation"]["contact_sensors"]
    return next(row["sensor_instance_id"] for row in rows if row["prim_path"] == path)


def test_no_contact_says_which_bodies_it_does_not_cover_and_an_unread_step_is_unscored() -> None:
    from blueprint_pipeline.task_predicates import score_episode, seal_predicates

    predicates = seal_predicates([{"id": "no_shelf_contact", "kind": "no_contact", "geometry": "scene_obstacles",
                                   "contact_force_threshold_n": 5.0}])
    measured = [{"step_index": index, "robot_scene_contact_peak_force_n": 0.0,
                 "robot_scene_contact_excluded_bodies": sorted(FEET)} for index in range(3)]
    (scored,) = score_episode(predicates=predicates, samples=measured)["predicate_results"]
    assert scored["status"] == "scored" and scored["satisfied"] is True
    assert scored["excluded_robot_bodies"] == sorted(FEET)
    # A trace with no exclusion (a fixed-base arm, say) reads as before.
    plain = [{key: value for key, value in sample.items() if key != "robot_scene_contact_excluded_bodies"}
             for sample in measured]
    (as_before,) = score_episode(predicates=predicates, samples=plain)["predicate_results"]
    assert "excluded_robot_bodies" not in as_before and as_before["satisfied"] is True
    unread = [measured[0], {"step_index": 1, "robot_scene_contact_unread": ["native_task_arena_force_matrix_invalid:x"]},
              measured[2]]
    report = score_episode(predicates=predicates, samples=unread)
    assert report["predicate_results"][0] == {"id": "no_shelf_contact", "kind": "no_contact", "status": "unscored",
                                              "missing_readback": "robot_contact"}
    assert report["outcome"] is None and report["unscored_predicates"] == ["no_shelf_contact"]


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
