"""A team's own arm through the task's authored phases from its station: CPU bounded estimates only."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

pytest.importorskip("pxr")

from blueprint_pipeline import native_task_construction_plan, scripted_robot_runtime_dispatch  # noqa: E402
from blueprint_pipeline import team_body_kinematics as kinematics  # noqa: E402
from blueprint_pipeline.team_arm_kinematics import pose_matrix  # noqa: E402
from blueprint_pipeline.team_arm_reach import team_arm_reach  # noqa: E402
from tests.test_team_body_kinematics import body  # noqa: E402

IDENTITY = "universal_robots_ur5e_robotiq_2f85_sim"
STATION = {"position_world_m": [1.2, -0.4, 0.0], "orientation_xyzw": [0.0, 0.0, 0.3826834, 0.9238795]}


def _quaternion(rotation):
    from blueprint_pipeline.team_arm_kinematics import quaternion_xyzw
    return list(quaternion_xyzw(rotation))


@pytest.fixture(scope="module")
def ur5e(tmp_path_factory):
    usd, manifest, _ = body(IDENTITY, str(tmp_path_factory.mktemp("reach")))
    return usd, manifest, kinematics.body_arm_chains(usd, manifest)["chains"]


def _plan(usd, manifest, **robot):
    return {"plan_digest": "sha256:" + "a" * 64, "task_kind": "rigid_pick_place",
            "robot": {"robot_id": "team_embodiment", "embodiment_id": manifest["embodiment_id"],
                      "embodiment_digest": manifest["embodiment_digest"], "base_mode": "fixed_at_station",
                      "base_pose_world": STATION,
                      "usd_sha256": "sha256:" + hashlib.sha256(usd.read_bytes()).hexdigest(), **robot}}


def _phases(monkeypatch, chain, joint_sets, far=None):
    """Authored phases at TCP poses the arm reaches at ``joint_sets`` (world frame, at the station)."""
    world_from_base = pose_matrix(STATION["position_world_m"], STATION["orientation_xyzw"]) @ chain.root_from_base
    rows = []
    for index, q in enumerate(joint_sets):
        tcp = world_from_base @ kinematics.forward(chain, q).tcp
        rows.append({"phase_id": f"phase_{index}", "position_world_m": tcp[:3, 3].tolist(),
                     "orientation_world_xyzw": _quaternion(tcp[:3, :3]), "gripper_state": "open"})
    if far is not None:
        rows.insert(far, {"phase_id": "carry", "position_world_m": [9.0, 9.0, 3.0],
                          "orientation_world_xyzw": [0.0, 0.0, 0.0, 1.0], "gripper_state": "closed"})
    rows.append({"phase_id": "recovery", "position_world_m": [50.0, 0.0, 0.0],
                 "orientation_world_xyzw": [0.0, 0.0, 0.0, 1.0], "gripper_state": "open"})
    monkeypatch.setattr(native_task_construction_plan, "materialize_native_task_construction_phase_plan",
                        lambda plan: {"manipulation_strategy": "pick_and_place", "phases": rows})


def _identity_grasp(monkeypatch, arms):
    grasp = {"position_m": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}
    monkeypatch.setattr(scripted_robot_runtime_dispatch, "build_policy_spec",
                        lambda manifest: {"grasp_bindings": {arm: {"tcp_from_grasp": grasp} for arm in arms}})


def _joint_sets(chain, count, seed=5):
    generator = np.random.default_rng(seed)
    low, high = np.maximum(chain.lower, -2.5), np.minimum(chain.upper, 2.5)
    return [low + (high - low) * generator.uniform(0.2, 0.8, len(low)) for _ in range(count)]


def test_the_teams_own_arm_reaching_every_authored_phase_from_its_station_passes(ur5e, monkeypatch):
    usd, manifest, chains = ur5e
    (name, chain), = chains.items()
    _phases(monkeypatch, chain, _joint_sets(chain, 4))
    _identity_grasp(monkeypatch, chains)
    report = team_arm_reach(_plan(usd, manifest), model_path=usd, manifest=manifest)
    assert report["status"] == "passed" and report["reaching_arms"] == [name] and report["blockers"] == []
    assert report["phase_ids"] == ["phase_0", "phase_1", "phase_2", "phase_3"]  # recovery is the simulator's
    assert report["arms"][name]["orientation_bound"] is True
    assert report["claim_ceiling"] == "cpu_bounded_estimate" and report["simulator_executed"] is False
    assert report["report_digest"].startswith("sha256:")


def test_a_phase_beyond_every_arms_reach_is_the_one_proven_refusal(ur5e, monkeypatch):
    usd, manifest, chains = ur5e
    (name, chain), = chains.items()
    _phases(monkeypatch, chain, _joint_sets(chain, 3), far=2)
    _identity_grasp(monkeypatch, chains)
    report = team_arm_reach(_plan(usd, manifest), model_path=usd, manifest=manifest)
    assert report["status"] == "blocked" and report["blockers"] == ["robot_arm_task_beyond_reach:carry"]
    assert [row["reached"] for row in report["arms"][name]["phases"]] == [True, True, False]


def test_an_unreached_but_spannable_pose_is_flagged_and_the_run_goes_on(ur5e, monkeypatch):
    usd, manifest, chains = ur5e
    (name, chain), = chains.items()
    _phases(monkeypatch, chain, _joint_sets(chain, 2))
    # Inside the reach bound yet through the arm's own base: no start reaches it.
    world_from_base = pose_matrix(STATION["position_world_m"], STATION["orientation_xyzw"]) @ chain.root_from_base
    inside = world_from_base @ pose_matrix([0.0, 0.0, -0.3], [0.0, 0.0, 0.0, 1.0])
    rows = native_task_construction_plan.materialize_native_task_construction_phase_plan({})["phases"]
    rows.insert(1, {"phase_id": "lift", "position_world_m": inside[:3, 3].tolist(),
                    "orientation_world_xyzw": [0.0, 0.0, 0.0, 1.0], "gripper_state": "closed"})
    _identity_grasp(monkeypatch, chains)
    report = team_arm_reach(_plan(usd, manifest), model_path=usd, manifest=manifest)
    assert report["status"] == "flagged" and report["blockers"] == []
    assert any(item.startswith(f"robot_arm_task_unreached:{name}:lift:") for item in report["advisories"])


def test_without_an_audited_grasp_the_body_is_solved_for_position_and_says_so(ur5e, monkeypatch):
    usd, manifest, chains = ur5e
    (name, chain), = chains.items()
    _phases(monkeypatch, chain, _joint_sets(chain, 3))

    def unbound(manifest):
        raise ValueError("scripted_grasp_source_tool_binding_mismatch")

    monkeypatch.setattr(scripted_robot_runtime_dispatch, "build_policy_spec", unbound)
    report = team_arm_reach(_plan(usd, manifest), model_path=usd, manifest=manifest)
    assert report["status"] == "passed" and report["arms"][name]["orientation_bound"] is False
    assert "robot_grasp_orientation_unbound:position_only" in report["advisories"]


def test_another_model_a_free_base_or_another_task_is_unassessed_never_measured(ur5e, monkeypatch, tmp_path):
    usd, manifest, chains = ur5e
    (_, chain), = chains.items()
    _phases(monkeypatch, chain, _joint_sets(chain, 2))
    other = tmp_path / "other.usda"
    other.write_text(usd.read_text() + "\n")
    cases = [(team_arm_reach(_plan(usd, manifest), model_path=other, manifest=manifest), "team_model_unbound"),
             (team_arm_reach(_plan(usd, manifest, base_mode="free_on_floor"), model_path=usd, manifest=manifest),
              "robot_base_not_fixed_at_station"),
             (team_arm_reach({**_plan(usd, manifest), "task_kind": "articulated_open_close"}, model_path=usd,
                             manifest=manifest), "task_kind_unmodeled")]
    for report, reason in cases:
        assert (report["status"], report["reason"], report["blockers"]) == ("unassessed", reason, [])
        assert report["advisories"] == ["robot_arm_path_unassessed:" + manifest["embodiment_id"]]
