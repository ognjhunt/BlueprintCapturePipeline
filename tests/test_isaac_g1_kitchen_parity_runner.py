"""Hermetic tests for the GPU runner's non-Isaac helpers (importing the runner must NOT pull
in isaacsim — the Isaac-API calls are lazily imported inside the GPU-only functions)."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

_RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_isaac_g1_kitchen_parity_eval.py"


def _load():
    spec = importlib.util.spec_from_file_location("parity_runner", _RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # would raise if it imported isaacsim at module load
    return mod


M = _load()


def test_runner_imports_without_isaacsim() -> None:
    assert hasattr(M, "run_scenarios") and hasattr(M, "parse_scenarios")


def test_runner_writes_result_before_simulation_app_close() -> None:
    source = _RUNNER.read_text()

    preclose_marker = '"isaac_g1_kitchen_parity_result.json").write_text'
    close_marker = "sim.close()"
    assert preclose_marker in source
    assert source.index(preclose_marker) < source.index(close_marker)


def test_manipulation_cam_is_egocentric_vs_follow_chase() -> None:
    root, yaw = (1.75, 1.25, 0.79), 0.0  # robot at the sink, facing +x
    me, mt = M.manipulation_cam_pose(root, yaw)
    fe, ft = M.follow_cam_pose(root, yaw)
    # manipulation eye: head height + slightly FORWARD of the root (egocentric)
    assert me[2] > root[2] + 0.4
    assert me[0] >= root[0]
    # follow eye: BEHIND the root (the chase shot that gave OSCAR a room-scale navigation view)
    assert fe[0] < root[0]
    # manipulation looks DOWN-forward at the workspace (target ahead of root, below eye, counter level)
    assert mt[0] > root[0]
    assert mt[2] < me[2]
    assert mt[2] < 1.0


def test_manipulation_cam_fixed_look_at_pins_faucet_regardless_of_yaw() -> None:
    # robot standing in front of the sink; faucet world point is known
    faucet = (2.28, 1.33, 0.9)
    e1, t1 = M.manipulation_cam_pose((2.28, 0.73, 0.79), math.pi / 2, look_at=faucet)
    # target is pinned to the faucet, eye sits at head height (egocentric)
    assert t1 == faucet
    assert e1[2] > 1.0
    # a wrong/noisy final yaw must NOT move the framing off the faucet
    _, t2 = M.manipulation_cam_pose((2.28, 0.73, 0.79), -math.pi / 2, look_at=faucet)
    assert t2 == faucet
    # without look_at it falls back to the yaw-relative target (forward of the robot) — a robot
    # standing elsewhere/facing elsewhere then frames its own front, NOT the faucet
    _, t3 = M.manipulation_cam_pose((1.0, 0.0, 0.79), 0.0)  # at [1,0] facing +x
    assert t3 != faucet and t3[0] > 1.0  # forward (+x) of the root, not the sink


def test_arm_reach_skeleton_moves_hand_toward_faucet_and_into_view() -> None:
    # a minimal right-arm chain at rest: arm hangs forward-low at the body (y ~ 0), hand at y=0.25
    rest = [
        ("torso_link", (2.28, 0.73, 1.1)),
        ("right_shoulder_link", (2.18, 0.73, 1.10)),
        ("right_elbow_link", (2.18, 0.78, 0.95)),
        ("right_wrist_link", (2.18, 0.83, 0.85)),
        ("right_hand_palm_link", (2.18, 0.88, 0.80)),
    ]
    faucet = (2.28, 1.33, 0.95)
    rest_hand = rest[-1][1]
    # at reach_frac=0 nothing moves
    assert M.compute_arm_reach_skeleton(rest, faucet, 0.0) == rest
    # at full reach the hand is much closer to the faucet than at rest
    full = dict(M.compute_arm_reach_skeleton(rest, faucet, 1.0))
    import math
    d = lambda a, b: math.dist(a, b)
    assert d(full["right_hand_palm_link"], faucet) < d(rest_hand, faucet)
    # the hand advances toward the faucet in +y (into the camera's forward view)
    assert full["right_hand_palm_link"][1] > rest_hand[1]
    # the arm never overstretches beyond its rest length from the shoulder
    sh = full["right_shoulder_link"]
    arm_len = d(rest[1][1], rest_hand)
    assert d(full["right_hand_palm_link"], sh) <= arm_len + 1e-6
    # non-arm links (torso) are untouched
    assert full["torso_link"] == (2.28, 0.73, 1.1)
    # the reach is monotonic: half-reach hand sits between rest and full
    half = dict(M.compute_arm_reach_skeleton(rest, faucet, 0.5))
    assert rest_hand[1] < half["right_hand_palm_link"][1] < full["right_hand_palm_link"][1]


def test_manipulation_ready_arm_pose_defaults_to_both_arms() -> None:
    deltas = M.manipulation_ready_arm_joint_deltas()
    assert deltas["left_shoulder_pitch_joint"] < 0
    assert deltas["right_shoulder_pitch_joint"] < 0
    assert deltas["left_elbow_joint"] < 0
    assert deltas["right_elbow_joint"] < 0
    assert "left_wrist_pitch_joint" in deltas
    assert "right_wrist_pitch_joint" in deltas

    right_only = M.manipulation_ready_arm_joint_deltas("right")
    assert "right_shoulder_pitch_joint" in right_only
    assert "left_shoulder_pitch_joint" not in right_only

    with pytest.raises(ValueError):
        M.manipulation_ready_arm_joint_deltas("center")


def test_apply_joint_deltas_updates_only_available_joint_targets() -> None:
    targets = [0.0, 0.0, 0.0]
    default = [1.0, 2.0, 3.0]
    applied = M._apply_joint_deltas(
        targets,
        default,
        {"right_shoulder_pitch_joint": 0, "right_elbow_joint": 2},
        {
            "right_shoulder_pitch_joint": -0.85,
            "missing_joint": 99.0,
            "right_elbow_joint": -0.23,
        },
    )
    assert applied == ["right_shoulder_pitch_joint", "right_elbow_joint"]
    assert targets == pytest.approx([0.15, 0.0, 2.77])


def test_arm_reach_skeleton_can_pose_both_arms_for_first_frame() -> None:
    rest = [
        ("left_shoulder_link", (0.0, -0.2, 1.1)),
        ("left_hand_palm_link", (0.0, -0.45, 0.8)),
        ("right_shoulder_link", (0.0, 0.2, 1.1)),
        ("right_hand_palm_link", (0.0, 0.45, 0.8)),
    ]
    target = (0.5, 0.0, 0.95)
    full = dict(M.compute_arm_reach_skeleton(rest, target, 1.0, arm="both"))
    assert full["left_hand_palm_link"][0] > rest[1][1][0]
    assert full["right_hand_palm_link"][0] > rest[3][1][0]
    assert full["left_hand_palm_link"] != rest[1][1]
    assert full["right_hand_palm_link"] != rest[3][1]


def test_skeleton_world_for_frame_falls_back_when_articulation_has_no_links() -> None:
    offsets = [
        ("torso_link", (0.0, 0.0, 0.3)),
        ("right_hand_palm_link", (0.2, 0.1, 0.0)),
    ]
    skel = M.skeleton_world_for_frame(
        art_ctx={"art": object(), "link_names": []},
        rest_offsets=offsets,
        root_pose=(1.0, 2.0, 0.7),
        yaw=0.0,
    )
    assert skel == [
        ("torso_link", (1.0, 2.0, 1.0)),
        ("right_hand_palm_link", (1.2, 2.1, 0.7)),
    ]


def test_camera_aperture_widens_fov_vs_default_telephoto() -> None:
    # the POV camera was rendering at USD's ~17deg-vertical default (focal 50 / vap 15.29),
    # which zooms into the dark basin; we widen it to the projection FOV
    focal, hap, vap = M.camera_aperture_for_fov(50.0, 1280, 960)
    got_vfov = 2 * math.degrees(math.atan(vap / (2 * focal)))
    assert got_vfov == pytest.approx(50.0, abs=1e-6)          # vertical FOV matches the request
    assert hap / vap == pytest.approx(1280 / 960, abs=1e-6)   # horizontal aperture matches aspect
    # default telephoto would be ~17deg vertical -> the new FOV is much wider (less zoomed)
    default_vfov = 2 * math.degrees(math.atan(15.2908 / (2 * 50.0)))
    assert got_vfov > default_vfov + 25


def test_arm_reach_rotation_swings_rest_bone_toward_target() -> None:
    # shoulder at origin; rest upper arm hangs down (-z); faucet is forward (+y), level
    shoulder = (0.0, 0.0, 0.0)
    rest_elbow = (0.0, 0.0, -0.3)          # arm hanging straight down
    target = (0.0, 0.6, 0.0)                # reach forward (+y)
    axis, angle = M.arm_reach_rotation(shoulder, rest_elbow, target, 1.0)
    # rest dir = -z, want dir = +y -> 90 deg; axis perpendicular to both (= +/-x)
    assert angle == pytest.approx(math.pi / 2, abs=1e-6)
    assert abs(axis[0]) == pytest.approx(1.0, abs=1e-6) and abs(axis[1]) < 1e-9 and abs(axis[2]) < 1e-9
    # reach_frac scales the swing
    _, half = M.arm_reach_rotation(shoulder, rest_elbow, target, 0.5)
    assert half == pytest.approx(math.pi / 4, abs=1e-6)
    _, zero = M.arm_reach_rotation(shoulder, rest_elbow, target, 0.0)
    assert zero == pytest.approx(0.0, abs=1e-9)
    assert math.isclose(math.sqrt(sum(c * c for c in axis)), 1.0, abs_tol=1e-9)  # unit axis


def test_parse_scenarios_normalizes_to_pelvis_height_route() -> None:
    req = {"scenarios": [
        {"scenario_id": "s1", "spawn_position_xyz": [-4.25, -3.35, 0.05],
         "target_position_xyz": [1.75, 1.25, 0.05], "description": "to sink"},
        {"id": "s2", "route_points": [[0, 0, 0.1], [1, 1, 0.1], [2, 2, 0.1]]},
        {"scenario_id": "bad"},  # no start/target -> skipped
    ]}
    sc = M.parse_scenarios(req)
    assert [s["scenario_id"] for s in sc] == ["s1", "s2"]
    # navigation route lifted to pelvis height
    assert all(p[2] == M.ROBOT_PELVIS_HEIGHT_M for p in sc[0]["route_points"])
    assert sc[0]["start"][2] == M.ROBOT_PELVIS_HEIGHT_M
    assert len(sc[1]["route_points"]) == 3


def test_assemble_collision_summary_counts() -> None:
    actions = [
        {"scene_collision_contact_count": 0, "policy_action": "accepted_direct_collision_checked_motion"},
        {"scene_collision_contact_count": 0, "policy_action": "redirected_by_collision_probe"},
    ]
    summ = M.assemble_collision_summary(actions=actions, rejected_probe_total=3, response_event_total=1)
    assert summ["robot_scene_contact_event_count"] == 0
    assert summ["rejected_scene_collision_probe_count"] == 3
    assert summ["near_miss_event_count"] == 3
    assert summ["collision_response_event_count"] == 1


def test_mp4_command_is_web_playable_ffmpeg() -> None:
    cmd = M.mp4_command("frames/overview_*.png", 24, "overview.mp4")
    assert cmd[0] == "ffmpeg"
    assert "yuv420p" in cmd  # web-playable pixel format
    assert "libx264" in cmd
    assert cmd[-1] == "overview.mp4"


def test_yaw_to_quat_is_wxyz_about_z() -> None:
    w, x, y, z = M.yaw_to_quat(math.pi / 2)
    assert x == 0.0 and y == 0.0
    assert w == pytest.approx(math.cos(math.pi / 4))
    assert z == pytest.approx(math.sin(math.pi / 4))


def test_build_result_aggregates_and_labels_truthfully() -> None:
    scs = [{"scenario_id": "a"}, {"scenario_id": "b"}]
    outs = [{"task_success": True}, {"task_success": False}]
    res = M.build_result(scenarios=scs, outcomes=outs, policy_id="blueprint_default_walk_to_target_smoke_policy",
                         kitchen_usd="k.usd", g1_usd="g1.usd", blockers=[])
    assert res["status"] == "completed"
    assert res["scenarios_passed"] == 1 and res["scenarios_executed"] == 2
    assert res["rendered_by_isaac_rtx"] is True
    assert "not dynamic locomotion" in res["proof_boundary"].lower()
    assert res["scenarios"][0]["scenario_id"] == "a" and res["scenarios"][0]["task_success"] is True


def test_build_result_blocks_on_blockers() -> None:
    res = M.build_result(scenarios=[], outcomes=[], policy_id="p", kitchen_usd="k", g1_usd=None,
                         blockers=["official_isaac_unitree_g1_articulation_api_unverified"])
    assert res["status"] == "blocked"


def _rotate_by_quat(q, v):
    # rotate vector v by quaternion q=(w,x,y,z)
    w, x, y, z = q
    # q * (0,v) * q^-1, expanded
    tx, ty, tz = v
    # vector part of q
    ux, uy, uz = x, y, z
    # t = 2 * cross(u, v)
    cx, cy, cz = (uy * tz - uz * ty, uz * tx - ux * tz, ux * ty - uy * tx)
    cx, cy, cz = 2 * cx, 2 * cy, 2 * cz
    # v + w*t + cross(u, t)
    c2 = (uy * cz - uz * cy, uz * cx - ux * cz, ux * cy - uy * cx)
    return (tx + w * cx + c2[0], ty + w * cy + c2[1], tz + w * cz + c2[2])


def test_look_at_quat_points_camera_minus_z_at_target() -> None:
    eye, target = (5.0, 0.0, 1.0), (0.0, 0.0, 1.0)
    q = M.look_at_quat(eye, target)
    # USD camera views along local -Z; rotated -Z should point from eye toward target (-X here)
    view = _rotate_by_quat(q, (0.0, 0.0, -1.0))
    expected = M._norm((target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]))
    assert view[0] == pytest.approx(expected[0], abs=1e-6)
    assert view[1] == pytest.approx(expected[1], abs=1e-6)
    assert view[2] == pytest.approx(expected[2], abs=1e-6)


def test_scene_framing_center_and_radius() -> None:
    scs = [{"route_points": [[0, 0, 0.79], [2, 0, 0.79]]},
           {"route_points": [[0, 2, 0.79], [2, 2, 0.79]]}]
    center, radius = M.scene_framing(scs)
    assert center[0] == pytest.approx(1.0) and center[1] == pytest.approx(1.0)
    assert radius >= 1.0


def test_project_point_to_pixel() -> None:
    eye, target, up = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)  # looking +X
    r = M.project_point_to_pixel((5.0, 0.0, 0.0), eye, target, up, 60.0, 640, 480)
    assert r is not None
    u, v, z = r
    assert abs(u - 320) < 1e-3 and abs(v - 240) < 1e-3 and abs(z - 5.0) < 1e-6  # on-axis -> center
    assert M.project_point_to_pixel((-5.0, 0.0, 0.0), eye, target, up, 60.0, 640, 480) is None  # behind
    up_pt = M.project_point_to_pixel((5.0, 0.0, 1.0), eye, target, up, 60.0, 640, 480)
    assert up_pt is not None and up_pt[1] < 240  # +Z world -> above image center (smaller v)
    # far off-axis -> out of frame -> None
    assert M.project_point_to_pixel((0.05, 50.0, 0.0), eye, target, up, 60.0, 640, 480) is None


def test_follow_cam_is_behind_and_above_robot() -> None:
    eye, target = M.follow_cam_pose((0.0, 0.0, 0.79), 0.0)  # facing +X
    assert eye[0] < 0.0           # behind the robot along -X
    assert eye[2] > 0.79          # above the root
    assert target[0] > 0.0        # looking ahead toward +X
