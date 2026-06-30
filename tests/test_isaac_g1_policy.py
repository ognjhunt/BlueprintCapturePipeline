"""Tests for the pluggable G1 policy — including a DIRECT parity cross-check that the
deterministic controller's ported math is byte-identical to the MuJoCo source functions.

The MuJoCo module's pure helpers import without the `mujoco` package, so we compare against
the real source of truth rather than hand-coded expectations.
"""
from __future__ import annotations

import math

import pytest

import blueprint_pipeline.isaac_g1_policy as P
import blueprint_pipeline.mujoco_g1_simulator_command as M


_ROUTES = [
    [(0.0, 0.0, 0.79), (2.0, 0.0, 0.79)],
    [(-4.25, -3.35, 0.05), (-1.0, -1.0, 0.79), (1.75, 1.25, 0.79)],
    [(0.0, 0.0, 0.79)],
    [(0.0, 0.0, 0.79), (0.0, 0.0, 0.79)],  # degenerate (zero-length)
]


# ----------------------------- geometry parity vs MuJoCo source -----------------------------

def test_interpolate_route_matches_mujoco_source() -> None:
    for route in _ROUTES:
        for i in range(11):
            alpha = i / 10.0
            mine = P.interpolate_route(route, alpha)
            ref = M._interpolate_route(route, alpha)
            assert P.rounded_pose(mine[0]) == P.rounded_pose(ref[0])
            assert round(mine[1], 9) == round(ref[1], 9)
            assert mine[2] == ref[2]


def test_route_and_pose_distance_match_mujoco_source() -> None:
    for route in _ROUTES:
        assert round(P.route_distance(route), 9) == round(M._route_distance(route), 9)
    assert round(P.pose_distance((0, 0, 0), (3, 4, 0)), 9) == round(M._pose_distance((0, 0, 0), (3, 4, 0)), 9)


def _candidate_geom(specs):
    # compare only navigation-relevant fields (MuJoCo also carries gait phase/moving, which
    # the Isaac kinematic placement does not use and which do not affect collision geometry)
    return [(s["candidate_kind"], P.rounded_pose(s["pose"]), s.get("lateral_offset_m"),
             s.get("relocation_radius_m")) for s in specs]


def test_candidate_pose_specs_match_mujoco_source() -> None:
    # moving case (has previous_pose -> includes a 'stop' candidate)
    mine = P.candidate_pose_specs(desired_pose=(1.0, 2.0, 0.79), previous_pose=(0.5, 1.5, 0.79),
                                  yaw=0.6, previous_yaw=0.5)
    ref = M._candidate_pose_specs(desired_pose=(1.0, 2.0, 0.79), previous_pose=(0.5, 1.5, 0.79),
                                  yaw=0.6, previous_yaw=0.5, previous_phase=0.0, previous_moving=True)
    assert _candidate_geom(mine) == _candidate_geom(ref)
    # spawn case (no previous_pose -> includes the relocation ring)
    mine0 = P.candidate_pose_specs(desired_pose=(1.0, 2.0, 0.79), previous_pose=None, yaw=0.3)
    ref0 = M._candidate_pose_specs(desired_pose=(1.0, 2.0, 0.79), previous_pose=None, yaw=0.3)
    assert _candidate_geom(mine0) == _candidate_geom(ref0)


# ----------------------------- outcome parity vs MuJoCo source -----------------------------

def _build_actions(route, n):
    actions = []
    for step in range(n):
        alpha = 0.0 if n <= 1 else step / float(n - 1)
        pose, yaw, seg = P.interpolate_route(route, alpha)
        actions.append({
            "step": step, "sim_time_s": round(step * 0.02, 9),
            "root_position": [round(c, 6) for c in pose],
            "desired_root_position": [round(c, 6) for c in pose],
            "root_yaw_radians": round(yaw, 6), "target": list(route[-1]),
            "route_segment_index": seg, "contact_count": 0, "scene_collision_contact_count": 0,
            "collision_probe_candidate_count": 1, "rejected_collision_probe_count": 0,
            "policy_action": "accepted_direct_collision_checked_motion", "scenario_eval_run_id": None,
        })
    return actions


def test_compute_task_outcome_matches_mujoco_source() -> None:
    # constant pelvis height (~0.79) so the clean walk reaches goal without tripping the fall
    # detector; the parity comparison below holds for any trace regardless.
    route = [(-4.25, -3.35, 0.79), (-1.0, -1.0, 0.79), (1.75, 1.25, 0.79)]
    actions = _build_actions(route, 12)
    summary = {"robot_scene_contact_event_count": 0, "rejected_scene_collision_probe_count": 0}
    mine = P.compute_task_outcome(actions=actions, start=route[0], target=route[-1],
                                  route_distance_m=P.route_distance(route), collision_summary=summary,
                                  bounded_steps=12, model_timestep_s=0.02)
    ref = M._attempt_task_outcome(actions=actions, start=route[0], target=route[-1],
                                  route_distance_m=M._route_distance(route), collision_summary=summary,
                                  bounded_steps=12, model_timestep_s=0.02)
    # proof_boundary wording intentionally differs (Isaac vs MuJoCo); everything else identical
    mine2 = {k: v for k, v in mine.items() if k != "proof_boundary"}
    ref2 = {k: v for k, v in ref.items() if k != "proof_boundary"}
    assert mine2 == ref2
    assert mine["task_success"] is True  # straight clean walk reaches goal


def test_compute_task_outcome_flags_unreached_goal() -> None:
    route = [(0.0, 0.0, 0.79), (5.0, 0.0, 0.79)]
    short = _build_actions([(0.0, 0.0, 0.79), (1.0, 0.0, 0.79)], 5)  # only walks 1m of a 5m goal
    out = P.compute_task_outcome(actions=short, start=route[0], target=route[-1],
                                 route_distance_m=5.0, collision_summary={}, bounded_steps=5,
                                 model_timestep_s=0.02)
    assert out["task_success"] is False
    assert "failure_target_not_reached" in out["failure_mode_ids"]


# ----------------------------- deterministic controller behavior -----------------------------

def _run(policy, route, n, oracle):
    policy.reset({"route_points": route, "start": route[0], "target": route[-1]})
    decisions = []
    for step in range(n):
        decisions.append(policy.step(P.StepContext(step=step, num_steps=n, probe_collision=oracle)))
    return decisions


def test_deterministic_policy_clear_path_goes_direct() -> None:
    route = _ROUTES[0]
    decs = _run(P.DeterministicWalkToTargetPolicy(), route, 6, lambda pose, yaw: 0)
    assert all(d.policy_action == "accepted_direct_collision_checked_motion" for d in decs)
    # final accepted pose reaches the target end of the route
    assert P.pose_distance(decs[-1].root_pose, route[-1]) < 1e-6


def test_deterministic_policy_redirects_around_blocked_direct() -> None:
    route = _ROUTES[0]
    pol = P.DeterministicWalkToTargetPolicy()
    pol.reset({"route_points": route, "start": route[0], "target": route[-1]})
    # block the exact 'direct' pose at step 0 (== start), allow anything offset
    start = route[0]

    def oracle(pose, yaw):
        return 1 if P.pose_distance(pose, start) < 1e-6 else 0

    d = pol.step(P.StepContext(step=0, num_steps=6, probe_collision=oracle))
    assert d.policy_action == "redirected_by_collision_probe"
    assert d.rejected_collision_probe_count >= 1
    assert P.pose_distance(d.root_pose, start) > 0.1  # took a lateral offset


def test_deterministic_policy_stops_when_fully_blocked() -> None:
    route = _ROUTES[0]
    pol = P.DeterministicWalkToTargetPolicy()
    pol.reset({"route_points": route, "start": route[0], "target": route[-1]})
    # step 0 to establish a previous pose, then fully block step 1 -> 'stop' candidate forced
    pol.step(P.StepContext(step=0, num_steps=6, probe_collision=lambda p, y: 0))
    d = pol.step(P.StepContext(step=1, num_steps=6, probe_collision=lambda p, y: 1))
    assert d.policy_action == "stopped_by_collision_probe"


# ----------------------------- registry + stage-B slot -----------------------------

def test_make_policy_registry() -> None:
    assert isinstance(P.make_policy(), P.DeterministicWalkToTargetPolicy)
    assert isinstance(P.make_policy("blueprint_default_walk_to_target_smoke_policy"),
                      P.DeterministicWalkToTargetPolicy)
    assert isinstance(P.make_policy("groot_sonic"), P.Groot17SonicPolicy)
    with pytest.raises(ValueError):
        P.make_policy("some_unknown_policy")


def test_groot_sonic_is_fail_closed_off_gpu() -> None:
    g = P.Groot17SonicPolicy()
    assert g.policy_id == "unitree_groot_n17_sonic_policy"
    assert g.DEFAULT_CHECKPOINT == "LucaFrat/groot-bs16"
    assert g.EMBODIMENT_TAG == "UNITREE_G1_SONIC"
    avail = g.available()
    assert avail["available"] is False  # gr00t stack not installed locally
    with pytest.raises((RuntimeError, NotImplementedError)):
        g.reset({"instruction": "walk to the sink"})


def test_groot_sonic_injected_infer_returns_step_decision_without_gpu() -> None:
    calls: list[dict] = []

    def _infer(obs):
        calls.append(dict(obs))
        return {
            "root_position": [0.25, 0.5, 0.79],
            "root_yaw_radians": 0.125,
            "joint_targets": {"left_shoulder_pitch_joint": 0.2},
        }

    policy = P.make_policy("groot_sonic", infer=_infer)
    assert isinstance(policy, P.Groot17SonicPolicy)
    policy.reset({"instruction": "open the fridge"})
    decision = policy.step(
        P.StepContext(
            step=2,
            num_steps=4,
            camera_rgb="frame-1",
            joint_state={"left_shoulder_pitch_joint": 0.0},
            instruction="open the fridge",
        )
    )

    assert calls == [
        {
            "camera_rgb": "frame-1",
            "joint_state": {"left_shoulder_pitch_joint": 0.0},
            "instruction": "open the fridge",
            "step": 2,
        }
    ]
    assert decision.root_pose == (0.25, 0.5, 0.79)
    assert decision.yaw == 0.125
    assert decision.policy_action == "learned_policy_action"
    assert decision.joint_targets == {"left_shoulder_pitch_joint": 0.2}


def test_gait_joint_deltas_match_mujoco_source() -> None:
    joints = list(P.G1_GAIT_JOINTS)
    addr = {n: i for i, n in enumerate(joints)}
    for phase in [0.0, 0.7, 1.5, 3.14159, 4.5, 6.2]:
        base = [0.0] * len(joints)
        qpos = [0.0] * len(joints)
        M._apply_preview_gait_pose(qpos=qpos, base_qpos=base, joint_addresses=addr, phase=phase, moving=True)
        ref = {n: qpos[addr[n]] for n in joints}
        mine = P.gait_joint_deltas(phase, moving=True)
        for n in joints:
            assert round(mine.get(n, 0.0), 9) == round(ref[n], 9), (n, phase)
    assert P.gait_joint_deltas(1.0, moving=False) == {}  # standing -> no gait
    # phase formula parity
    assert round(P.gait_phase(0.5, 3.0), 9) == round(0.5 * 3.0 * math.pi * 2.0, 9)


def test_action_record_schema() -> None:
    d = P.StepDecision(root_pose=(1.0, 2.0, 0.79), yaw=0.5, desired_root_position=(1.1, 2.1, 0.79),
                       route_segment_index=0, policy_action="accepted_direct_collision_checked_motion",
                       collision_probe_candidate_count=1, rejected_collision_probe_count=0)
    rec = P.action_record(decision=d, step=3, sim_time_s=0.06, target=(2.0, 2.0, 0.79))
    assert rec["root_position"] == [1.0, 2.0, 0.79]
    assert rec["policy_action"] == "accepted_direct_collision_checked_motion"
    assert rec["step"] == 3 and rec["route_segment_index"] == 0
