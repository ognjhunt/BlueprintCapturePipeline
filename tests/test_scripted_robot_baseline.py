"""Owned-script contract and real generic episode loop on explicit fake physics.

These tests establish software wiring, not Isaac execution or task success.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.scripted_robot_baseline import (
    ACTION_SOURCE, EVIDENCE_KIND, PLAN_SCHEMA, ScriptedRobotBaseline, validate_plan,
)
from blueprint_pipeline.team_arm_kinematics import pose_matrix, quaternion_xyzw, validate_pose_chunk

IDENTITY = [0., 0., 0., 1.]


def scripted_plan(*, scene_digest="sha256:" + "1" * 64):
    value = {"schema_version": PLAN_SCHEMA, "execution_evidence_kind": EVIDENCE_KIND, "action_source": ACTION_SOURCE,
        "task_kind": "rigid_pick_place", "scene_plan_digest": scene_digest, "arm_names": ["left", "right"], "active_arm": "left",
        "grasp_bindings": {arm: {"tcp_from_grasp": {"position_m": [0., 0., 0.], "orientation_xyzw": IDENTITY},
                                   "basis_reference": "development-only-known-cartesian-tool"} for arm in ("left", "right")},
        "phases": [{"phase_id": "pregrasp", "position_world_m": [.08, 0., 0.],
                    "orientation_world_xyzw": IDENTITY, "gripper_state": "open"},
                   {"phase_id": "grasp_contact", "position_world_m": [.08, 0., 0.],
                    "orientation_world_xyzw": IDENTITY, "gripper_state": "closed"}],
        "action_hz": 60, "phase_timeout_s": 5., "gripper_dwell_s": .05,
        "arrival_tolerance_m": .005, "arrival_orientation_tolerance_rad": .05, "plan_digest": ""}
    return seal(value)


def seal(value):
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    return value


def observation():
    return {"state": {"arm_names": ["left", "right"], "gripper_names": ["left", "right"],
        "arm_gripper_names": {"left": "left", "right": "right"}, "gripper_position": [0., .3],
        "arm_eef_pose": {arm: {"position_m": [0., 0., 0.], "orientation_xyzw": IDENTITY} for arm in ("left", "right")},
        "scripted_arm_base_pose_world": {arm: {"position_m": [0., 0., 0.], "orientation_xyzw": IDENTITY}
                                          for arm in ("left", "right")}}}


def test_rotated_base_and_nonidentity_tool_offset_are_composed_independently():
    plan, obs = scripted_plan(), observation()
    half = math.sqrt(.5)
    base = pose_matrix([1., 2., 0.], [0., 0., half, half])
    tcp_from_grasp = pose_matrix([0., 0., .07], [half, 0., 0., half])
    plan["grasp_bindings"]["left"]["tcp_from_grasp"] = {"position_m": [0., 0., .07], "orientation_xyzw": [half, 0., 0., half]}
    expected_tcp = pose_matrix([.1, .2, .3], IDENTITY)
    world_grasp = base @ expected_tcp @ tcp_from_grasp
    plan["phases"][0].update(position_world_m=world_grasp[:3, 3].tolist(),
                            orientation_world_xyzw=quaternion_xyzw(world_grasp[:3, :3]))
    obs["state"]["scripted_arm_base_pose_world"]["left"] = {"position_m": [1., 2., 0.], "orientation_xyzw": [0., 0., half, half]}
    client = ScriptedRobotBaseline(plan=seal(plan), policy_id="development-only")
    result = client.infer(obs)
    validate_pose_chunk(result, arm_names=["left", "right"], maximum_rows=1)
    assert result["positions_m"][0][0] == pytest.approx([.1, .2, .3])
    assert result["orientations_xyzw"][0][0] == pytest.approx(IDENTITY)
    assert result["positions_m"][0][1] == [0., 0., 0.]
    assert result["gripper_closures"][0] == [0., .3]
    assert client.candidate_policy_queried is False
    assert client.scripted_controller_invoked is True


def test_dwell_uses_measured_pose_timeout_is_not_success_and_episode_reset_clears_state():
    plan, obs = scripted_plan(), observation()
    plan["phase_timeout_s"] = .05
    client = ScriptedRobotBaseline(plan=seal(plan), policy_id="development-only")
    for _ in range(3):
        client.infer(obs)
    assert client.phase_index == 1
    assert client.phase_receipts[0]["timeout"] is True
    assert client.phase_receipts[0]["pose_arrival_observed"] is False
    obs["state"]["arm_eef_pose"]["left"]["position_m"] = [.08, 0., 0.]
    for _ in range(3):
        result = client.infer(obs)
    assert result["gripper_closures"][0] == [1., .3]
    assert client.finished
    assert client.evidence_summary()["policy_outcome_interpretable"] is False
    assert client.last_inference_evidence["task_success_claimed"] is False
    client.reset()
    assert client.phase_index == 0 and client.held is None and client.phase_receipts == []
    assert client.scripted_controller_invoked is False and client.finished is False


def test_mutated_authored_plan_cannot_be_used():
    value = scripted_plan()
    value["phases"][0]["position_world_m"][0] += .1
    with pytest.raises(ValueError, match="scripted_baseline_plan_invalid"):
        validate_plan(value)


def test_nearest_arm_selection_is_measured_and_does_not_move_other_arm():
    plan, obs = scripted_plan(), observation()
    plan["active_arm"] = "nearest_initial_tcp"
    obs["state"]["scripted_arm_base_pose_world"]["left"]["position_m"] = [3., 0., 0.]
    client = ScriptedRobotBaseline(plan=seal(plan), policy_id="development-only")
    action = client.infer(obs)
    assert client.active_arm == "right"
    assert action["positions_m"][0] == [[0., 0., 0.], [.08, 0., 0.]]
    assert action["gripper_closures"][0] == [0., 0.]


def test_planner_retains_scripted_identity_for_selected_deferred_and_unavailable():
    from copy import deepcopy
    from blueprint_pipeline.beta_policy_candidate_catalog import seal_catalog
    from blueprint_pipeline.beta_policy_screening_plan import build_screening_plan, validate_screening_plan
    from tests.test_beta_policy_screening_plan import inputs, admitted, scene, NOW
    catalog, _, _ = inputs(3)
    rows = deepcopy(catalog["candidates"])
    for row in rows:
        row["policy"].update(kind="scripted", digest_kind="source_code_inventory", source="blueprint-owned:scripted_pick_place")
        row["interface"]["policy_interface_id"] = "blueprint_scripted_pick_place.v1"
    catalog = seal_catalog({**catalog, "candidates": rows})
    bindings, quotes = {}, {}
    for row in catalog["candidates"][:2]:
        bindings[row["candidate_id"]], quotes[row["candidate_id"]] = admitted(row, maximum="8.00")
    plan = build_screening_plan(catalog, scene(), campaign_id="development-only", now=NOW,
        capabilities={"candidate_bindings": bindings}, cost_quotes=quotes)
    assert [row["status"] for row in plan["candidates"]] == ["selected", "waitlisted", "unavailable"]
    assert {row["execution_evidence_kind"] for row in plan["candidates"]} == {EVIDENCE_KIND}
    assert validate_screening_plan(plan) == plan
    import json
    from pathlib import Path
    from jsonschema import Draft202012Validator
    schema = json.loads((Path(__file__).parents[1] / "docs/schemas/beta_policy_screening_plan.v1.schema.json").read_text())
    Draft202012Validator(schema).validate(plan)


def test_task_driver_copies_actual_authored_geometry_without_reference_joint_trajectory(monkeypatch):
    from blueprint_pipeline import scripted_robot_baseline_runtime as runtime
    from blueprint_pipeline.native_task_construction_plan import materialize_native_task_construction_phase_plan
    from tests.test_native_task_construction_plan import _rigid_fixture, _planar_push_fixture

    # Test only the task geometry seam; body/source identity is covered by the
    # full registration tests. No fake authored targets enter production.
    monkeypatch.setattr(runtime, "validate_policy_spec", lambda *_: None)
    scene = _rigid_fixture(asset_id="development-only-subject")
    policy = {"action_hz": 60, "grasp_bindings": scripted_plan()["grasp_bindings"]}
    manifest = {"embodiment_digest": "sha256:" + "2" * 64, "arms": [{"name": arm} for arm in ("left", "right")]}
    authored = materialize_native_task_construction_phase_plan(scene)
    planned = runtime.build_task_plan(scene, policy, manifest)
    original = {phase["phase_id"]: phase for phase in authored["phases"]}
    assert "recovery" not in [phase["phase_id"] for phase in planned["phases"]]
    assert [phase["phase_id"] for phase in planned["phases"]][:3] == ["pregrasp", "approach_contact_open", "grasp_contact"]
    assert planned["phases"][1]["gripper_state"] == "open"
    for phase in planned["phases"]:
        source = original["grasp_contact" if phase["phase_id"] == "approach_contact_open" else phase["phase_id"]]
        assert phase["position_world_m"] == source["position_world_m"]
        assert phase["orientation_world_xyzw"] == source["orientation_world_xyzw"]
        assert set(phase) == {"phase_id", "position_world_m", "orientation_world_xyzw", "gripper_state"}
    assert planned["task_spec_digest"] == canonical_digest(scene["task_spec"])
    assert planned["authored_construction_plan_digest"] == authored["plan_digest"]
    with pytest.raises(ValueError, match="scripted_baseline_manipulation_strategy_unsupported"):
        runtime.build_task_plan(_planar_push_fixture(), policy, manifest)
    with pytest.raises(ValueError, match="scripted_baseline_task_kind_unsupported"):
        runtime.build_task_plan({"task_kind": "articulated_open_close"}, policy, manifest)


def test_actual_generic_loop_drives_one_arm_and_scores_independently(tmp_path, monkeypatch):
    from blueprint_pipeline import adp_task_scoring, episode_visual_evidence, native_task_episode_environment
    from blueprint_pipeline.team_embodiment import apply_team_embodiment, read_model_structure, team_embodiment_binding, validate_embodiment_manifest, verify_model_file
    from blueprint_pipeline.team_embodiment_runtime import run_team_episode
    from tests.test_team_arm_kinematics import CartesianRobot, named_inputs
    from tests.test_team_embodiment import _Camera, _Scoring, station_plan

    model, value = named_inputs(tmp_path)
    manifest = validate_embodiment_manifest(value)
    binding = team_embodiment_binding(manifest, model_record=verify_model_file(manifest, model),
                                     structure=read_model_structure(model), scene_plan=station_plan())
    scene = {**apply_team_embodiment(station_plan(), binding), "scenario": {"seed": 7}}
    scene["cadence"].update(maximum_action_steps=120, settle_window_samples=0)
    row = scene["robot"]
    class Env:
        def __init__(self):
            self.robot, self.steps = CartesianRobot(row), []
            self.unwrapped = SimpleNamespace(device="cpu", scene={"robot": self.robot,
                **{name: _Camera() for name in ("team_camera_0", "team_camera_1", "team_camera_2", "external_camera_2")}})
        def reset(self, seed):
            self.robot.q = np.array([row["joint_reset_positions_rad"][name] for name in self.robot.joint_names])
        def step(self, action):
            order = [*row["joint_names"], *(joint["name"] for joint in row["gripper_joints"])]
            self.steps.append(list(action[0]))
            self.robot.q = np.array([action[0][order.index(name)] for name in self.robot.joint_names])
    env = Env()
    client = ScriptedRobotBaseline(plan=scripted_plan(scene_digest=scene["plan_digest"]), policy_id="development-only")
    built = SimpleNamespace(env=env, camera_scene_names={**{f"team_camera_{i}": f"team_camera_{i}" for i in range(3)},
                                                       "overview": "external_camera_2"})
    monkeypatch.setattr(native_task_episode_environment, "NativeRigidScoringEnvironment", _Scoring)
    monkeypatch.setattr(episode_visual_evidence, "finalize_multicamera_visual_evidence", lambda **_: ({"status": "complete"}, []))
    samples = []
    monkeypatch.setattr(adp_task_scoring, "score_task_episode_from_spec", lambda **kw: samples.extend(kw["samples"]) or {"task_succeeded": False})
    progress = {}
    result = run_team_episode(built=built, task_readback=None, policy=client, plan=scene, prompt="Fixture only",
        max_policy_queries=120, open_loop_horizon=1, media_root=tmp_path / "episodes", episode_id="development-only-scripted",
        progress=progress, to_action=lambda targets: [targets])
    assert 0 < len(env.steps) <= 120
    assert env.robot.q[env.robot.joint_names.index("left_joint_0")] > .07
    assert env.robot.q[env.robot.joint_names.index("right_joint_0")] == 0.
    assert result["score"]["task_succeeded"] is False  # Script's arrival cannot override native scorer.
    assert samples and result["scripted_controller_calls"] > 0
    assert result["policy_queries"] == 0 and progress["candidate_policy_queried"] is False
    assert result["action_source"] == ACTION_SOURCE and result["policy_outcome_interpretable"] is False
    assert progress["scripted_controller_invoked"] is True
    assert result["policy_request_evidence_complete"] is False  # No network serialization exists.
    assert result["scripted_request_evidence_complete"] is True
