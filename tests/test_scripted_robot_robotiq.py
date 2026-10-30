"""Real generic loop on explicit Cartesian fake physics; no native task proof."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from blueprint_pipeline import adp_task_scoring, episode_visual_evidence, native_task_episode_environment
from blueprint_pipeline import scripted_robot_robotiq_runtime as runtime
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.team_embodiment import apply_team_embodiment, read_model_structure, team_embodiment_binding, verify_model_file
from blueprint_pipeline.team_embodiment_runtime import run_team_episode
from tests.test_scripted_robot_baseline_bundle import registered_world
from tests.test_native_task_construction_plan import _rigid_fixture
from tests.test_team_arm_kinematics import CartesianRobot
from tests.test_team_embodiment import _Camera, _Scoring, station_plan


@pytest.mark.parametrize("identity", sorted(runtime.BODIES))
def test_actual_loop_rebinds_current_cell_for_six_and_seven_dof_without_learned_claims(tmp_path, monkeypatch, identity):
    _, _, registration, key = registered_world(tmp_path, monkeypatch, identity)
    import json
    from pathlib import Path
    manifest = json.loads(Path(registration["manifest"]["path"]).read_text())
    model = Path(registration["model"]["path"])
    base = station_plan()
    authored = _rigid_fixture(asset_id="development-only-grasp-subject")
    base["task_spec"].update(authored["task_spec"])
    next(row for row in base["objects"] if row.get("task_subject")).update(authored["objects"][0])
    binding = team_embodiment_binding(manifest, model_record=verify_model_file(manifest, model),
        structure=read_model_structure(model), scene_plan=base)
    scene = apply_team_embodiment(base, binding)
    scene["cadence"].update(maximum_action_steps=240, settle_window_samples=0)
    scene["scenario"] = {"seed": 9}
    scene["plan_digest"] = canonical_digest(scene, digest_field="plan_digest")
    policy = registration["policy_spec"]
    stale = deepcopy(scene)
    stale["scenario"] = {"seed": 7}
    stale["plan_digest"] = canonical_digest(stale, digest_field="plan_digest")
    client = runtime.ScriptedRobotiqBaseline(plan=runtime.build_task_plan(stale, policy, manifest),
        policy_id=key, policy_spec=policy, manifest=manifest)
    row = scene["robot"]
    class Robot(CartesianRobot):
        def __init__(self):
            super().__init__(row)
            self.body_names = [row["arms"][0]["base_link"], row["arms"][0]["tool_link"]]
        def get_jacobians(self):
            # Deliberately simple6axis Cartesian fake;7th axis is a redundant zero column.
            jac = np.zeros((1, 1, 6, len(self.joint_names)))
            for axis, joint in enumerate(row["arms"][0]["joint_names"][:6]):
                jac[0, 0, axis, self.joint_names.index(joint)] = 1.
            return jac
    class Env:
        def __init__(self):
            self.robot, self.steps = Robot(), []
            self.unwrapped = SimpleNamespace(device="cpu", scene={"robot": self.robot,
                "team_camera_0": _Camera(), "external_camera_2": _Camera()})
        def reset(self, seed):
            self.robot.q = np.array([row["joint_reset_positions_rad"][n] for n in self.robot.joint_names])
        def step(self, action):
            order = [*row["joint_names"], *(j["name"] for j in row["gripper_joints"])]
            self.steps.append(list(action[0]))
            self.robot.q = np.array([action[0][order.index(n)] for n in self.robot.joint_names])
    env = Env()
    built = SimpleNamespace(env=env, camera_scene_names={"team_camera_0": "team_camera_0", "overview": "external_camera_2"})
    monkeypatch.setattr(native_task_episode_environment, "NativeRigidScoringEnvironment", _Scoring)
    monkeypatch.setattr(episode_visual_evidence, "finalize_multicamera_visual_evidence", lambda **_: ({"status": "complete"}, []))
    samples = []
    monkeypatch.setattr(adp_task_scoring, "score_task_episode_from_spec", lambda **kw: samples.extend(kw["samples"]) or {"task_succeeded": False})
    progress = {}
    result = run_team_episode(built=built, task_readback=None, policy=client, plan=scene, prompt="Development fixture",
        max_policy_queries=3, open_loop_horizon=1, media_root=tmp_path / "episodes", episode_id="development-only",
        progress=progress, to_action=lambda targets: [targets])
    assert client.plan["scene_plan_digest"] == scene["plan_digest"] != stale["plan_digest"]
    assert len(env.steps) == 3 and all(len(command) == len(row["joint_names"])+len(row["gripper_joints"]) for command in env.steps)
    assert result["scripted_controller_calls"] == 3 and result["policy_queries"] == 0
    assert result["score"]["task_succeeded"] is False and samples
    assert progress["candidate_policy_queried"] is False and progress["scripted_controller_invoked"] is True
    assert result["policy_outcome_interpretable"] is False


@pytest.mark.parametrize("fault", ["dof", "jaw", "contact", "tool", "profile"])
def test_exact_tool_contract_refuses_changed_variant(fault):
    from tests.test_scripted_robot_grasp_frames import development_manifest
    from blueprint_pipeline.scripted_robot_runtime_dispatch import runtime_for_policy
    manifest = development_manifest("universal_robots_ur5e_robotiq_2f85_sim")
    manifest.pop("embodiment_digest")
    if fault == "dof":
        manifest["arms"][0]["joint_names"].pop()
    elif fault == "jaw":
        manifest["grippers"][0]["joints"][0]["closed"] = .2
    elif fault == "contact":
        manifest["grippers"][0]["contact_links"][0] = "unreviewed_tool"
    elif fault == "tool":
        manifest["arms"][0]["tool_link"] = "unreviewed_tool"
    else:
        with pytest.raises(ValueError, match="profile_unsupported"):
            runtime_for_policy({"policy_id": "unreviewed"})
        return
    with pytest.raises(ValueError):
        runtime.build_policy_spec(manifest)
