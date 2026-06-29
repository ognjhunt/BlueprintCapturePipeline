"""Hermetic tests for the GPU runner's non-Isaac helpers (importing the runner must NOT pull
in isaacsim — the Isaac-API calls are lazily imported inside the GPU-only functions)."""
from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest

from blueprint_pipeline.scene_placement import SceneObject

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
    # target stays anchored near the faucet while blending toward the active arm corridor.
    assert math.dist(t1, faucet) < 0.25
    assert e1[2] > 1.0
    # a wrong/noisy final yaw must NOT fall back to a yaw-relative navigation target.
    _, t2 = M.manipulation_cam_pose((2.28, 0.73, 0.79), -math.pi / 2, look_at=faucet)
    assert math.dist(t2, faucet) < 0.25
    # without look_at it falls back to the yaw-relative target (forward of the robot) — a robot
    # standing elsewhere/facing elsewhere then frames its own front, NOT the faucet
    _, t3 = M.manipulation_cam_pose((1.0, 0.0, 0.79), 0.0)  # at [1,0] facing +x
    assert t3 != faucet and t3[0] > 1.0  # forward (+x) of the root, not the sink


def test_surface_affordance_point_projects_target_to_nearest_face() -> None:
    stance_plan = {
        "task_target_xyz": [-1.979559, 0.655166, 1.025963],
        "task_target_bounds": {
            "bbox_min_xyz": [-2.521971, 0.13306, -0.000508],
            "bbox_max_xyz": [-1.437147, 1.177273, 2.052435],
        },
    }

    point = M._surface_affordance_point_for_stance(
        stance_plan,
        root_pose=(-1.035327, 0.658475, 0.84),
    )

    assert point == pytest.approx((-1.437147, 0.655166, 1.025963))


def test_manipulation_cam_with_look_at_frames_active_arm_corridor() -> None:
    root = (-1.035327, 0.658475, 0.84)
    yaw = -3.138089
    look_at = (-1.437147, 0.655166, 1.025963)

    eye, target = M.manipulation_cam_pose(root, yaw, look_at=look_at, reach_arm="right")

    assert target[0] > look_at[0]  # target is blended toward the active shoulder, not inside target
    assert target[1] > look_at[1]
    assert target[2] > look_at[2]
    assert eye[0] < root[0]       # head-mounted seed: slightly forward toward the fridge
    assert abs(eye[1] - root[1]) < 0.02
    assert eye[2] > look_at[2]

    left_eye, left_target = M.manipulation_cam_pose(root, yaw, look_at=look_at, reach_arm="left")
    assert abs(left_eye[1] - root[1]) < 0.02
    assert left_target[1] < look_at[1]


def test_manipulation_camera_target_blends_affordance_with_visible_arm_context() -> None:
    affordance = (-1.44, 0.66, 1.03)
    arm_points = {
        "elbow": (-1.05, 0.55, 1.10),
        "wrist": (-1.22, 0.58, 0.98),
        "hand": (-1.34, 0.62, 0.96),
    }

    target = M._manipulation_camera_target_with_arm_context(affordance, arm_points)

    assert target[0] > affordance[0]
    assert target[1] < affordance[1]
    assert abs(target[2] - affordance[2]) < 0.08
    assert math.dist(target, affordance) < 0.25


def test_robot_head_lens_eye_offsets_link_origin_out_of_head_mesh() -> None:
    eye, meta = M._robot_head_lens_eye_from_mount((-1.0, 0.6, 1.35), math.pi)

    assert eye[0] < -1.0
    assert eye[1] == pytest.approx(0.6)
    assert eye[2] > 1.35
    assert meta["raw_mount_eye_xyz"] == [-1.0, 0.6, 1.35]
    assert meta["lens_height_correction_applied"] is False

    low_eye, low_meta = M._robot_head_lens_eye_from_mount(
        (-0.86, 0.65, 0.84),
        math.pi,
        root_pose=(-1.04, 0.65, 0.84),
        arm_points={
            "shoulder": (-0.87, 0.65, 1.13),
            "wrist": (-0.98, 0.65, 0.94),
            "hand": (-1.15, 0.65, 0.95),
        },
    )
    assert low_eye[0] < -1.10
    assert low_eye[2] > 1.35
    assert low_meta["lens_height_correction_applied"] is True
    assert low_meta["min_head_lens_z"] > 1.3

    authored, authored_meta = M._robot_head_lens_eye_from_mount(
        (-1.0, 0.6, 1.35),
        math.pi,
        root_pose=(-1.04, 0.65, 0.84),
        arm_points={"shoulder": (-0.87, 0.65, 1.13)},
        authored_camera=True,
    )
    assert authored == (-1.0, 0.6, 1.35)
    assert authored_meta["lens_offset_xyz_robot_frame"] == [0.0, 0.0, 0.0]
    assert authored_meta["lens_height_correction_applied"] is False


def test_task_visual_qc_splits_verify_and_pov_rubrics(monkeypatch, tmp_path) -> None:
    qc_mod = __import__("blueprint_pipeline.render_visual_qc", fromlist=["dummy"])
    calls: dict[str, list[str]] = {}

    def fake_placement(frames, target, *, task_description="", sample_n=4, generate=None):
        calls["placement"] = [Path(p).name for p in frames]
        return {
            "schema_version": "robot_placement_visual_qc.v1",
            "status": "passed",
            "target": target,
            "task_description": task_description,
            "frames_reviewed": len(frames),
            "blockers": [],
            "per_frame": [],
        }

    def fake_pov(frames, target, *, task_description="", sample_n=4, generate=None):
        calls["pov"] = [Path(p).name for p in frames]
        return {
            "schema_version": "manipulation_pov_visual_qc.v1",
            "status": "passed",
            "target": target,
            "task_description": task_description,
            "frames_reviewed": len(frames),
            "blockers": [],
            "per_frame": [],
        }

    monkeypatch.setattr(qc_mod, "qc_robot_placement_frames", fake_placement)
    monkeypatch.setattr(qc_mod, "qc_manipulation_pov_frames", fake_pov)
    verify = tmp_path / "verify_0000.png"
    pov = tmp_path / "robot_pov_0000.png"

    report = M._run_task_visual_qc(
        [verify],
        [pov],
        target_label="refrigerator",
        task_description="open the refrigerator",
    )

    assert report["status"] == "passed"
    assert calls == {"placement": ["verify_0000.png"], "pov": ["robot_pov_0000.png"]}
    assert report["placement"]["schema_version"] == "robot_placement_visual_qc.v1"
    assert report["manipulation_pov"]["schema_version"] == "manipulation_pov_visual_qc.v1"


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

    def d(a, b):
        return math.dist(a, b)

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


def test_manipulation_pov_render_and_validation_use_same_arm_selection() -> None:
    source = _RUNNER.read_text()
    assert "pov_reach_arm = _normalize_reach_arm_selection(manipulation_reach_arm)" in source
    assert "rendered_reach_arm = pov_reach_arm" in source
    assert "arm=rendered_reach_arm" in source
    assert "arm=pov_reach_arm" in source
    assert 'arm_points_by_arm=cam_meta.get("arm_link_points_by_arm_xyz") or {}' in source
    assert "reach_arm = args.manipulation_reach_arm" in source
    assert 'if args.manipulation_reach_arm != "both" else "right"' not in source


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
         "target_position_xyz": [1.75, 1.25, 0.05], "description": "to sink",
         "target_object_id": "faucet_handle"},
        {"id": "s2", "route_points": [[0, 0, 0.1], [1, 1, 0.1], [2, 2, 0.1]]},
        {"scenario_id": "task_only", "description": "open the service door"},
        {"scenario_id": "bad"},  # no start/target -> skipped
    ]}
    sc = M.parse_scenarios(req)
    assert [s["scenario_id"] for s in sc] == ["s1", "s2", "task_only"]
    # navigation route lifted to pelvis height
    assert all(p[2] == M.ROBOT_PELVIS_HEIGHT_M for p in sc[0]["route_points"])
    assert sc[0]["start"][2] == M.ROBOT_PELVIS_HEIGHT_M
    assert len(sc[1]["route_points"]) == 3
    assert sc[0]["raw_target_position_xyz"] == [1.75, 1.25, 0.05]
    assert sc[0]["target_object_id"] == "faucet_handle"
    assert sc[2]["task_target_deferred"] is True
    assert sc[2]["route_points"] == []
    assert sc[2]["instruction"] == "open the service door"


def test_deferred_task_route_materializes_from_dynamic_stance() -> None:
    scenario = {
        "scenario_id": "task_only",
        "instruction": "open the refrigerator",
        "route_points": [],
        "task_target_deferred": True,
    }
    M._materialize_deferred_task_route(
        scenario,
        stance_plan={"task_target_xyz": [1.5, 2.5, 1.1]},
        root_pose=(1.0, 2.0, 0.84),
        look_at=(1.5, 2.5, 1.1),
    )

    assert scenario["task_target_deferred"] is False
    assert scenario["deferred_task_resolution"] == "materialized_from_task_stance_plan"
    assert scenario["start"] == [1.0, 2.0, 0.84]
    assert scenario["target"] == [1.5, 2.5, M.ROBOT_PELVIS_HEIGHT_M]
    assert scenario["route_points"] == [scenario["start"], scenario["target"]]
    assert scenario["raw_target_position_xyz"] == [1.5, 2.5, 1.1]


def test_task_stance_planner_uses_target_as_thing_to_face_not_pelvis() -> None:
    scenario = {
        "task_target_position_xyz": [0.0, 0.0, 0.9],
        "robot_start_position_xyz": [-3.0, 0.0, 0.05],
        "stance_distance_candidates_m": [1.0],
        "floor_z_hint": 0.05,
    }
    plan = M.plan_task_stance(scenario=scenario, probe_collision=lambda pose, yaw: 0)
    assert plan["status"] == "accepted"
    assert plan["accepted_pose"][:2] == [-1.0, 0.0]
    assert plan["accepted_pose"][2] == pytest.approx(M.ROBOT_PELVIS_HEIGHT_M + 0.05)
    assert plan["accepted_pose"][:2] != plan["task_target_xyz"][:2]
    assert plan["accepted_yaw"] == pytest.approx(0.0)
    assert plan["candidates"][0]["scene_collision_contact_count"] == 0


def test_task_stance_planner_samples_around_target_until_collision_free() -> None:
    scenario = {
        "task_target_position_xyz": [0.0, 0.0, 0.9],
        "robot_start_position_xyz": [-3.0, 0.0, 0.05],
        "stance_distance_candidates_m": [1.0],
    }

    def probe(pose, yaw):
        return 1 if pose[0] < -0.99 and abs(pose[1]) < 1e-6 else 0

    plan = M.plan_task_stance(scenario=scenario, probe_collision=probe)
    assert plan["status"] == "accepted"
    assert plan["selected_candidate_index"] == 1
    assert plan["candidates"][0]["scene_collision_contact_count"] == 1
    assert plan["candidates"][1]["scene_collision_contact_count"] == 0
    assert math.dist(plan["accepted_pose"][:2], plan["task_target_xyz"][:2]) == pytest.approx(1.0)


def test_task_stance_planner_offsets_from_target_footprint_surface() -> None:
    scenario = {
        "task_target_position_xyz": [2.0, 2.0, 0.9],
        "robot_start_position_xyz": [2.0, 0.0, 0.05],
        "target_object_bbox_min_xyz": [1.5, 1.65, 0.75],
        "target_object_bbox_max_xyz": [2.5, 2.35, 1.15],
        "stance_distance_candidates_m": [0.85],
        "floor_z_hint": 0.05,
    }

    plan = M.plan_task_stance(scenario=scenario, probe_collision=lambda pose, yaw: 0)

    assert plan["status"] == "accepted"
    # The robot stands on the approach-side aisle. Without the footprint-surface offset, this
    # would be y=1.15 (0.85m from the center); with the sink/counter half-depth included, it moves
    # to y=0.80, clear of the target footprint.
    assert plan["accepted_pose"][:2] == [2.0, 0.8]
    assert plan["accepted_yaw"] == pytest.approx(math.pi / 2)
    first = plan["candidates"][0]
    assert first["standoff_from_target_surface_m"] == pytest.approx(0.85)
    assert first["target_surface_offset_m"] == pytest.approx(0.35)
    assert first["distance_to_target_m"] == pytest.approx(1.2)
    assert plan["task_target_bounds"]["bbox_min_xyz"] == [1.5, 1.65, 0.75]


def test_open_articulated_target_uses_close_surface_standoff_from_bounds() -> None:
    scenario = {
        "instruction": "open the refrigerator",
        "target_object_id": "refrigerator",
        "target_object_label": "refrigerator door",
        "target_object_position_xyz": [-1.979559, 0.655166, 1.025963],
        "robot_start_position_xyz": [-0.6, 0.66, 0.05],
        "target_object_bbox_min_xyz": [-2.521971, 0.13306, -0.000508],
        "target_object_bbox_max_xyz": [-1.437147, 1.177273, 2.052435],
        "floor_z_hint": 0.05,
    }

    distances = M.task_stance_distance_candidates(scenario)
    assert distances[0] == pytest.approx(0.4)
    assert M._validation_standoff_range_for_scenario(scenario) == pytest.approx(
        M.TASK_STANCE_CLOSE_REACH_GAP_RANGE_M
    )

    plan = M.plan_task_stance(
        scenario=scenario,
        probe_collision=lambda pose, yaw: 0,
        placement_validator=lambda _pose, _yaw, record: (
            {"status": "accepted", "blockers": []}
            if record["angle_offset_deg"] == 0
            and record["standoff_from_target_surface_m"] == pytest.approx(0.4)
            else {"status": "blocked", "blockers": ["synthetic_reach_profile_reject"]}
        ),
    )

    assert plan["status"] == "accepted"
    chosen = plan["candidates"][plan["selected_candidate_index"]]
    assert chosen["angle_offset_deg"] == 0
    assert chosen["standoff_from_target_surface_m"] == pytest.approx(0.4)
    assert plan["accepted_pose"][0] == pytest.approx(-1.037152, abs=0.002)
    assert plan["accepted_pose"][1] == pytest.approx(0.65847, abs=0.002)
    assert abs(abs(plan["accepted_yaw"]) - math.pi) < 0.01


def test_non_articulated_target_keeps_default_standoff_profile() -> None:
    scenario = {
        "instruction": "turn on the faucet",
        "target_object_id": "sink",
        "target_object_label": "sink",
    }

    assert M.task_stance_distance_candidates(scenario)[0] == pytest.approx(
        M.TASK_STANCE_DEFAULT_SURFACE_STANDOFF_M
    )
    assert M._validation_standoff_range_for_scenario(scenario) == pytest.approx(
        M.TASK_STANCE_DEFAULT_VALIDATION_STANDOFF_RANGE_M
    )


def test_task_stance_default_distances_include_counter_clearance_band() -> None:
    scenario = {
        "task_target_position_xyz": [2.277888, 1.333059, 0.848527],
        "robot_start_position_xyz": [2.35, 0.1, 0.05],
        "target_object_bbox_min_xyz": [2.009021, 0.89582, 0.555082],
        "target_object_bbox_max_xyz": [2.546755, 1.770299, 1.141971],
        "floor_z_hint": 0.05,
    }

    plan = M.plan_task_stance(
        scenario=scenario,
        probe_collision=lambda pose, yaw: 0,
        placement_validator=lambda _pose, _yaw, record: (
            {"status": "accepted", "blockers": []}
            if record["angle_offset_deg"] == 0
            and abs(float(record["standoff_from_target_surface_m"]) - 1.4025) < 1e-9
            else {"status": "blocked", "blockers": ["synthetic_clearance_or_reach_failure"]}
        ),
    )

    assert plan["status"] == "accepted"
    chosen = plan["candidates"][plan["selected_candidate_index"]]
    assert chosen["angle_offset_deg"] == 0
    assert chosen["standoff_from_target_surface_m"] == pytest.approx(1.4025)
    assert plan["accepted_pose"][0] == pytest.approx(2.386169, abs=1e-6)
    assert plan["accepted_pose"][1] == pytest.approx(-0.518468, abs=1e-6)


def test_task_stance_planner_tries_near_angled_aisle_before_farther_straight_back() -> None:
    scenario = {
        "task_target_position_xyz": [2.277888, 1.333059, 0.848527],
        "robot_start_position_xyz": [2.35, 0.1, 0.05],
        "target_object_bbox_min_xyz": [2.009021, 0.89582, 0.555082],
        "target_object_bbox_max_xyz": [2.546755, 1.770299, 1.141971],
        "floor_z_hint": 0.05,
    }

    def validator(_pose, _yaw, record):
        # The straight-back closest candidate clips. The planner should try a nearby angled stance
        # at the same standoff before walking farther backward along the wall-side ray.
        if (
            record["angle_offset_deg"] == 0
            and abs(float(record["standoff_from_target_surface_m"]) - 0.85) < 1e-9
        ):
            return {"status": "blocked", "blockers": ["synthetic_wall_side_clearance_failure"]}
        return {"status": "accepted", "blockers": []}

    plan = M.plan_task_stance(
        scenario=scenario,
        probe_collision=lambda pose, yaw: 0,
        placement_validator=validator,
    )

    assert plan["status"] == "accepted"
    chosen = plan["candidates"][plan["selected_candidate_index"]]
    assert chosen["standoff_from_target_surface_m"] == pytest.approx(0.85)
    assert chosen["angle_offset_deg"] == -15
    assert plan["accepted_pose"][0] == pytest.approx(2.009, abs=0.002)
    assert plan["accepted_pose"][1] == pytest.approx(0.028, abs=0.002)


def test_task_stance_planner_prefers_approach_ray_over_backside_candidate() -> None:
    scenario = {
        "task_target_position_xyz": [2.277888, 1.333059, 0.848527],
        "robot_start_position_xyz": [2.35, 0.1, 0.05],
        "target_object_bbox_min_xyz": [2.009021, 0.89582, 0.555082],
        "target_object_bbox_max_xyz": [2.546755, 1.770299, 1.141971],
        "floor_z_hint": 0.05,
    }

    def validator(_pose, _yaw, record):
        # The closer 180-degree candidate is on the backside of the target. A farther point on the
        # approach ray is the room-side stance and must win once both validate geometrically.
        accepts = (
            record["angle_offset_deg"] == 180
            and abs(float(record["standoff_from_target_surface_m"]) - 1.0625) < 1e-9
        ) or (
            record["angle_offset_deg"] == 0
            and abs(float(record["standoff_from_target_surface_m"]) - 1.4025) < 1e-9
        )
        return (
            {"status": "accepted", "blockers": []}
            if accepts
            else {"status": "blocked", "blockers": ["synthetic_geometry_reject"]}
        )

    plan = M.plan_task_stance(
        scenario=scenario,
        probe_collision=lambda pose, yaw: 0,
        placement_validator=validator,
    )

    assert plan["status"] == "accepted"
    chosen = plan["candidates"][plan["selected_candidate_index"]]
    assert chosen["angle_offset_deg"] == 0
    assert chosen["standoff_from_target_surface_m"] == pytest.approx(1.4025)
    assert plan["accepted_pose"][0] == pytest.approx(2.386169, abs=1e-6)
    assert plan["accepted_pose"][1] == pytest.approx(-0.518468, abs=1e-6)
    assert plan["accepted_candidate_count"] == 2


def test_task_stance_planner_without_approach_hint_prefers_nearest_validated_face() -> None:
    scenario = {
        "task_target_position_xyz": [0.0, 0.0, 1.0],
        "target_object_bbox_min_xyz": [-0.5, -0.5, 0.0],
        "target_object_bbox_max_xyz": [0.5, 0.5, 2.0],
        "floor_z_hint": 0.05,
        "stance_distance_candidates_m": [0.4, 1.13],
    }

    def validator(_pose, _yaw, record):
        accepts = (
            record["angle_offset_deg"] == 180
            and abs(float(record["standoff_from_target_surface_m"]) - 0.4) < 1e-9
        ) or (
            record["angle_offset_deg"] == 90
            and abs(float(record["standoff_from_target_surface_m"]) - 1.13) < 1e-9
        )
        return (
            {"status": "accepted", "blockers": []}
            if accepts
            else {"status": "blocked", "blockers": ["synthetic_geometry_reject"]}
        )

    plan = M.plan_task_stance(
        scenario=scenario,
        probe_collision=lambda pose, yaw: 0,
        placement_validator=validator,
    )

    assert plan["status"] == "accepted"
    chosen = plan["candidates"][plan["selected_candidate_index"]]
    assert chosen["angle_offset_deg"] == 180
    assert chosen["standoff_from_target_surface_m"] == pytest.approx(0.4)
    assert chosen["approach_bias_enabled"] is False
    assert plan["accepted_candidate_count"] == 2


def test_xy_rect_overlap_and_gap_reports_overlap_and_clearance() -> None:
    overlapping = M._xy_rect_overlap_and_gap(
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.5, 0.25, 0.0],
        [1.5, 0.75, 1.0],
    )
    assert overlapping["overlaps_xy"] is True
    assert overlapping["overlap_area_xy_m2"] == pytest.approx(0.25)
    assert overlapping["gap_m"] == pytest.approx(0.0)

    separated = M._xy_rect_overlap_and_gap(
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.3, 1.4, 0.0],
        [2.0, 2.0, 1.0],
    )
    assert separated["overlaps_xy"] is False
    assert separated["overlap_area_xy_m2"] == pytest.approx(0.0)
    assert separated["gap_m"] == pytest.approx(0.5)


def test_task_stance_planner_rejects_candidate_failed_by_placement_validation() -> None:
    scenario = {
        "task_target_position_xyz": [0.0, 0.0, 0.9],
        "target_object_bbox_min_xyz": [-0.25, -0.25, 0.6],
        "target_object_bbox_max_xyz": [0.25, 0.25, 1.2],
        "robot_start_position_xyz": [-3.0, 0.0, 0.05],
        "stance_distance_candidates_m": [1.0],
    }

    def validator(_pose, _yaw, record):
        if record["angle_offset_deg"] == 0:
            return {"status": "blocked", "blockers": ["placed_robot_bbox_overlaps_target_bbox"]}
        return {"status": "accepted", "blockers": []}

    plan = M.plan_task_stance(
        scenario=scenario,
        probe_collision=lambda pose, yaw: 0,
        placement_validator=validator,
    )

    assert plan["status"] == "accepted"
    assert plan["selected_candidate_index"] == 1
    assert plan["candidates"][0]["placement_validation"]["blockers"] == [
        "placed_robot_bbox_overlaps_target_bbox"
    ]
    assert plan["placement_validation"]["status"] == "accepted"


def test_task_stance_planner_fails_when_all_collision_free_candidates_fail_validation() -> None:
    scenario = {
        "task_target_position_xyz": [0.0, 0.0, 0.9],
        "robot_start_position_xyz": [-3.0, 0.0, 0.05],
        "stance_distance_candidates_m": [1.0],
    }

    plan = M.plan_task_stance(
        scenario=scenario,
        probe_collision=lambda pose, yaw: 0,
        placement_validator=lambda pose, yaw, record: {
            "status": "blocked",
            "blockers": ["placed_robot_bbox_center_far_from_root_pose"],
        },
    )

    assert plan["status"] == "blocked"
    assert plan["blockers"] == ["no_validated_task_stance_candidate"]
    assert plan["placement_validation_rejected_candidate_count"] == len(plan["candidates"])
    assert all("placement_validation" in candidate for candidate in plan["candidates"])


def test_stage_placement_validator_blocks_actual_robot_bbox_overlap(monkeypatch) -> None:
    placed = []
    monkeypatch.setattr(
        M,
        "_place_root",
        lambda stage, prim_path, pose, yaw: placed.append((stage, prim_path, pose, yaw)),
    )
    monkeypatch.setattr(
        M,
        "_world_bbox_for_prim",
        lambda _stage, _prim_path: {
            "bbox_min_xyz": [1.1, -0.2, 0.0],
            "bbox_max_xyz": [1.4, 0.2, 1.7],
            "center_xyz": [1.25, 0.0, 0.85],
            "size_xyz": [0.3, 0.4, 1.7],
        },
    )
    validator = M._placement_validator_for_stage(
        object(),
        "/World/G1",
        ((1.0, -0.5, 0.5), (1.5, 0.5, 1.5)),
    )

    result = validator((0.0, 0.0, 0.84), 0.0, {"standoff_from_target_surface_m": 0.85})

    assert placed
    assert result["status"] == "blocked"
    assert "placed_robot_bbox_overlaps_target_bbox" in result["blockers"]
    assert "placed_robot_bbox_center_far_from_root_pose" in result["blockers"]


def test_stage_placement_validator_accepts_clear_actual_robot_bbox(monkeypatch) -> None:
    monkeypatch.setattr(M, "_place_root", lambda _stage, _prim_path, _pose, _yaw: None)
    monkeypatch.setattr(
        M,
        "_world_bbox_for_prim",
        lambda _stage, _prim_path: {
            "bbox_min_xyz": [-0.3, -0.3, 0.0],
            "bbox_max_xyz": [0.3, 0.3, 1.7],
            "center_xyz": [0.0, 0.0, 0.85],
            "size_xyz": [0.6, 0.6, 1.7],
        },
    )
    validator = M._placement_validator_for_stage(
        object(),
        "/World/G1",
        ((1.0, -0.5, 0.5), (1.5, 0.5, 1.5)),
    )

    result = validator((0.0, 0.0, 0.84), 0.0, {"standoff_from_target_surface_m": 0.85})

    assert result["status"] == "accepted"
    assert result["blockers"] == []
    assert result["target_bbox_relation"]["gap_m"] == pytest.approx(0.7)
    assert result["required_target_gap_m"] == pytest.approx(0.35)


def test_stage_placement_validator_accepts_close_reach_gap_when_task_range_allows(monkeypatch) -> None:
    pose = (-1.037152, 0.65847, 0.84)
    target = SceneObject(
        id="refrigerator",
        label="refrigerator door",
        bbox_min=(-2.521971, 0.13306, -0.000508),
        bbox_max=(-1.437147, 1.177273, 2.052435),
        centroid=(-1.979559, 0.655166, 1.025963),
    )
    monkeypatch.setattr(M, "_place_root", lambda _stage, _prim_path, _pose, _yaw: None)
    monkeypatch.setattr(
        M,
        "_world_bbox_for_prim",
        lambda _stage, _prim_path: {
            "bbox_min_xyz": [pose[0] - 0.28, pose[1] - 0.28, 0.22],
            "bbox_max_xyz": [pose[0] + 0.28, pose[1] + 0.28, 1.46],
            "center_xyz": [pose[0], pose[1], pose[2]],
            "size_xyz": [0.56, 0.56, 1.24],
        },
    )
    validator = M._placement_validator_for_stage(
        object(),
        "/World/G1",
        (target.bbox_min, target.bbox_max),
        target_object=target,
        scene_objects=[],
        floor_z=0.05,
        standoff_range=M.TASK_STANCE_CLOSE_REACH_GAP_RANGE_M,
    )

    result = validator(pose, -3.138083, {"standoff_from_target_surface_m": 0.4})

    assert result["status"] == "accepted"
    assert result["blockers"] == []
    assert result["target_bbox_relation"]["gap_m"] == pytest.approx(0.12, abs=0.002)
    assert result["deterministic_geometry"]["standoff_m"] == pytest.approx(0.12, abs=0.002)
    assert result["validation_standoff_range_m"] == pytest.approx(
        list(M.TASK_STANCE_CLOSE_REACH_GAP_RANGE_M)
    )


def test_stage_placement_validator_suppresses_broad_aabb_clip_only_after_zero_physx_contact(
    monkeypatch,
) -> None:
    pose = (-1.037152, 0.65847, 0.84)
    target = SceneObject(
        id="articulated_target",
        label="openable appliance door",
        bbox_min=(-2.521971, 0.13306, -0.000508),
        bbox_max=(-1.437147, 1.177273, 2.052435),
        centroid=(-1.979559, 0.655166, 1.025963),
    )
    broad_false_positive = SceneObject(
        id="broad_asset_leaf",
        label="asset_leaf",
        bbox_min=(-1.30, -0.15, 0.0),
        bbox_max=(2.57, 2.46, 0.84),
        centroid=(0.635, 1.155, 0.42),
        source="usd_leaf",
    )
    monkeypatch.setattr(M, "_place_root", lambda _stage, _prim_path, _pose, _yaw: None)
    monkeypatch.setattr(
        M,
        "_world_bbox_for_prim",
        lambda _stage, _prim_path: {
            "bbox_min_xyz": [pose[0] - 0.28, pose[1] - 0.28, 0.22],
            "bbox_max_xyz": [pose[0] + 0.28, pose[1] + 0.28, 1.46],
            "center_xyz": [pose[0], pose[1], pose[2]],
            "size_xyz": [0.56, 0.56, 1.24],
        },
    )
    validator = M._placement_validator_for_stage(
        object(),
        "/World/G1",
        (target.bbox_min, target.bbox_max),
        target_object=target,
        scene_objects=[broad_false_positive],
        floor_z=0.05,
        standoff_range=M.TASK_STANCE_CLOSE_REACH_GAP_RANGE_M,
    )

    accepted = validator(pose, -3.138083, {"standoff_from_target_surface_m": 0.4, "scene_collision_contact_count": 0})
    blocked = validator(pose, -3.138083, {"standoff_from_target_surface_m": 0.4, "scene_collision_contact_count": 1})

    assert accepted["status"] == "accepted"
    assert accepted["deterministic_geometry"]["ok"] is True
    assert accepted["deterministic_geometry_raw"]["ok"] is False
    assert accepted["deterministic_geometry_adjustments"]["suppressed_broad_aabb_clips"][0]["object_id"] == "broad_asset_leaf"
    assert blocked["status"] == "blocked"
    assert "placement_geometry_invalid" in blocked["blockers"]


def test_placement_manifest_preserves_broad_aabb_adjustment(monkeypatch) -> None:
    pose = (-1.037152, 0.65847, 0.84)
    stance_plan = {
        "floor_z_hint": 0.05,
        "accepted_pose": list(pose),
        "accepted_yaw": -3.138083,
        "selected_candidate_index": 0,
        "candidates": [
            {
                "standoff_from_target_surface_m": 0.4,
                "scene_collision_contact_count": 0,
            }
        ],
        "task_target_xyz": [-1.979559, 0.655166, 1.025963],
        "task_target_bounds": {
            "bbox_min_xyz": [-2.521971, 0.13306, -0.000508],
            "bbox_max_xyz": [-1.437147, 1.177273, 2.052435],
        },
        "target_resolution": {
            "selected": {
                "target_object_id": "articulated_target",
                "target_object_label": "openable appliance door",
            }
        },
        "placement_validation": {
            "validation_standoff_range_m": list(M.TASK_STANCE_CLOSE_REACH_GAP_RANGE_M)
        },
    }
    broad_false_positive = SceneObject(
        id="broad_asset_leaf",
        label="asset_leaf",
        bbox_min=(-1.30, -0.15, 0.0),
        bbox_max=(2.57, 2.46, 0.84),
        centroid=(0.635, 1.155, 0.42),
        source="usd_leaf",
    )
    monkeypatch.setattr(
        M,
        "_world_bbox_for_prim",
        lambda _stage, _prim_path: {
            "bbox_min_xyz": [pose[0] - 0.28, pose[1] - 0.28, 0.22],
            "bbox_max_xyz": [pose[0] + 0.28, pose[1] + 0.28, 1.46],
            "center_xyz": [pose[0], pose[1], pose[2]],
            "size_xyz": [0.56, 0.56, 1.24],
        },
    )

    manifest = M._build_placement_validation_manifest(
        stage=object(),
        robot_prim_path="/World/G1",
        stance_plan=stance_plan,
        accepted_pose=pose,
        accepted_yaw=-3.138083,
        root_diagnostics={"status": "corrected"},
        scene_objects=[broad_false_positive],
        scenario_id="open_articulated_target",
        visual_qc={"status": "passed"},
    )

    assert manifest["status"] == "PASS"
    assert manifest["blockers"] == []
    assert manifest["intended_geometry"]["ok"] is True
    assert manifest["intended_geometry"]["raw_geometry"]["ok"] is False
    assert manifest["intended_geometry"]["adjustments"]["suppressed_broad_aabb_clips"][0]["object_id"] == "broad_asset_leaf"


def test_place_root_corrects_measured_world_footprint_offset(monkeypatch) -> None:
    placements = []
    current = {"pose": (0.0, 0.0, 0.0)}
    local_offset = (0.35, -0.2)

    def fake_set_root(_stage, _prim_path, pose, _yaw):
        current["pose"] = tuple(float(v) for v in pose)
        placements.append(current["pose"])

    def fake_bbox(_stage, _prim_path):
        pose = current["pose"]
        center = (pose[0] + local_offset[0], pose[1] + local_offset[1])
        return {
            "bbox_min_xyz": [center[0] - 0.25, center[1] - 0.25, 0.0],
            "bbox_max_xyz": [center[0] + 0.25, center[1] + 0.25, 1.6],
            "center_xyz": [center[0], center[1], 0.8],
            "size_xyz": [0.5, 0.5, 1.6],
        }

    monkeypatch.setattr(M, "_set_root_xform", fake_set_root)
    monkeypatch.setattr(M, "_world_bbox_for_prim", fake_bbox)
    monkeypatch.setattr(M, "_root_transform_diagnostics", lambda _stage, _prim_path: {"root": "diag"})

    diag = M._place_root(object(), "/World/G1", (2.0, 0.8, 0.84), 1.57)

    assert diag["status"] == "corrected"
    assert diag["correction_applied"] is True
    assert diag["measured_offset_xy_m"] == [0.35, -0.2]
    assert diag["final_footprint_center_xy"] == [2.0, 0.8]
    assert diag["final_xy_error_m"] == pytest.approx(0.0)
    assert placements == [(2.0, 0.8, 0.84), (1.65, 1.0, 0.84)]


def test_placement_validation_manifest_passes_with_actual_bbox_center_match(monkeypatch) -> None:
    sp = importlib.import_module("blueprint_pipeline.scene_placement")
    target = sp.SceneObject(
        id="sink",
        label="sink",
        bbox_min=(1.5, 1.65, 0.75),
        bbox_max=(2.5, 2.35, 1.15),
        centroid=(2.0, 2.0, 0.95),
        source="test",
    )
    stance_plan = {
        "status": "accepted",
        "accepted_pose": [2.0, 0.8, 0.84],
        "accepted_yaw": math.pi / 2,
        "floor_z_hint": 0.05,
        "task_target_xyz": [2.0, 2.0, 0.95],
        "task_target_bounds": {
            "bbox_min_xyz": [1.5, 1.65, 0.75],
            "bbox_max_xyz": [2.5, 2.35, 1.15],
        },
    }
    monkeypatch.setattr(
        M,
        "_world_bbox_for_prim",
        lambda _stage, _prim_path: {
            "bbox_min_xyz": [1.72, 0.52, 0.0],
            "bbox_max_xyz": [2.28, 1.08, 1.6],
            "center_xyz": [2.0, 0.8, 0.8],
            "size_xyz": [0.56, 0.56, 1.6],
        },
    )

    manifest = M._build_placement_validation_manifest(
        stage=object(),
        robot_prim_path="/World/G1",
        stance_plan=stance_plan,
        accepted_pose=(2.0, 0.8, 0.84),
        accepted_yaw=math.pi / 2,
        root_diagnostics={"status": "placed"},
        scene_objects=[target],
        scenario_id="sink_stance",
        visual_qc={"status": "passed", "blockers": []},
        topdown_frame="/tmp/placement_topdown_0000.png",
    )

    assert manifest["status"] == "PASS"
    assert manifest["blockers"] == []
    assert manifest["ground_truth_placement"]["xy_error_m"] == pytest.approx(0.0)
    assert manifest["intended_geometry"]["ok"] is True
    assert manifest["topdown_debug_frame"].endswith("placement_topdown_0000.png")


def test_placement_validation_manifest_fails_on_actual_bbox_center_mismatch(monkeypatch) -> None:
    stance_plan = {
        "status": "accepted",
        "accepted_pose": [2.0, 0.8, 0.84],
        "accepted_yaw": math.pi / 2,
        "floor_z_hint": 0.05,
        "task_target_xyz": [2.0, 2.0, 0.95],
        "task_target_bounds": {
            "bbox_min_xyz": [1.5, 1.65, 0.75],
            "bbox_max_xyz": [2.5, 2.35, 1.15],
        },
    }
    monkeypatch.setattr(
        M,
        "_world_bbox_for_prim",
        lambda _stage, _prim_path: {
            "bbox_min_xyz": [1.72, 1.32, 0.0],
            "bbox_max_xyz": [2.28, 1.88, 1.6],
            "center_xyz": [2.0, 1.6, 0.8],
            "size_xyz": [0.56, 0.56, 1.6],
        },
    )
    monkeypatch.setattr(M, "_root_transform_diagnostics", lambda _stage, _prim_path: {"root": "diag"})

    manifest = M._build_placement_validation_manifest(
        stage=object(),
        robot_prim_path="/World/G1",
        stance_plan=stance_plan,
        accepted_pose=(2.0, 0.8, 0.84),
        accepted_yaw=math.pi / 2,
        root_diagnostics={"status": "placed"},
        scene_objects=[],
        scenario_id="sink_stance",
        visual_qc={"status": "passed", "blockers": []},
    )

    assert manifest["status"] == "FAIL"
    assert "placement_ground_truth_center_mismatch" in manifest["blockers"]
    gt = manifest["ground_truth_placement"]
    assert gt["robot_prim_path"] == "/World/G1"
    assert gt["accepted_pose_xyz"] == [2.0, 0.8, 0.84]
    assert gt["actual_world_aabb"]["center_xyz"] == [2.0, 1.6, 0.8]
    assert gt["actual_footprint_center_xyz"] == [2.0, 1.6, 0.8]
    assert gt["computed_xyz_offset_m"] == [0.0, 0.8, -0.04]
    assert gt["xy_error_m"] > 0.1
    assert gt["xform_diagnostics"] == {"root": "diag"}


def test_dynamic_scene_target_bounds_thread_into_task_stance(monkeypatch) -> None:
    def fake_resolve(_stage, _scenario):
        return {
            "status": "resolved",
            "source": "scene_placement_task_label",
            "selected": {
                "target_object_id": "sink",
                "target_object_label": "sink",
                "center_xyz": [2.0, 2.0, 0.95],
                "size_xyz": [1.0, 0.7, 0.4],
                "bbox_min_xyz": [1.5, 1.65, 0.75],
                "bbox_max_xyz": [2.5, 2.35, 1.15],
            },
        }

    monkeypatch.setattr(M, "_resolve_task_target_from_stage", fake_resolve)
    scenario = {
        "instruction": "Stand at the kitchen sink and turn on the faucet.",
        "raw_spawn_position_xyz": [2.0, 0.0, 0.05],
        "floor_z_hint": 0.05,
    }

    plan = M._plan_task_stance_for_stage(
        stage=object(),
        scenario=scenario,
        manipulation_look_at=None,
        probe=lambda pose, yaw: 0,
        no_collision_probe=False,
    )

    assert plan["status"] == "accepted"
    assert plan["accepted_pose"][1] == pytest.approx(0.8)
    assert plan["target_resolution"]["selected"]["target_object_id"] == "sink"
    assert plan["task_target_bounds"]["bbox_max_xyz"] == [2.5, 2.35, 1.15]
    assert plan["candidates"][0]["standoff_from_target_surface_m"] == pytest.approx(
        M.TASK_STANCE_DEFAULT_SURFACE_STANDOFF_M
    )


def test_task_stance_planner_fails_closed_when_all_candidates_collide() -> None:
    scenario = {
        "task_target_position_xyz": [0.0, 0.0, 0.9],
        "robot_start_position_xyz": [-3.0, 0.0, 0.05],
        "stance_distance_candidates_m": [0.8, 1.0],
    }
    plan = M.plan_task_stance(scenario=scenario, probe_collision=lambda pose, yaw: 2)
    assert plan["status"] == "blocked"
    assert plan["blockers"] == ["no_collision_free_task_stance_candidate"]
    assert all(c["scene_collision_contact_count"] == 2 for c in plan["candidates"])


def test_sink_target_standoff_does_not_use_broad_cabinet_fixture() -> None:
    sink = SceneObject(
        id="sink",
        label="sink",
        bbox_min=(2.0, 0.9, 0.5),
        bbox_max=(2.6, 1.8, 1.1),
        centroid=(2.3, 1.35, 0.8),
    )
    cabinet = SceneObject(
        id="kitchen_cabinet_1",
        label="kitchen_cabinet",
        bbox_min=(-1.3, -0.15, 0.0),
        bbox_max=(2.6, 2.45, 0.85),
        centroid=(0.65, 1.15, 0.4),
    )
    faucet = SceneObject(
        id="faucet",
        label="faucet",
        bbox_min=(2.25, 1.25, 0.9),
        bbox_max=(2.35, 1.35, 1.1),
        centroid=(2.3, 1.3, 1.0),
    )

    assert M._find_standoff_fixtures([cabinet, sink], sink) == []
    assert M._find_standoff_fixtures([cabinet, sink], faucet) == [cabinet, sink]


def test_placement_shell_obstacles_include_walls_not_floor_or_lights(monkeypatch) -> None:
    class FakeRangeBox:
        def __init__(self, bmin, bmax):
            self._min = bmin
            self._max = bmax

        def IsEmpty(self):
            return False

        def GetMin(self):
            return self._min

        def GetMax(self):
            return self._max

        def GetSize(self):
            return [self._max[i] - self._min[i] for i in range(3)]

    class FakeBound:
        def __init__(self, box):
            self._box = box

        def ComputeAlignedBox(self):
            return self._box

    class FakePrim:
        def __init__(self, path, name, bmin, bmax):
            self._path = path
            self._name = name
            self.box = FakeRangeBox(bmin, bmax)

        def GetPath(self):
            return self._path

        def GetName(self):
            return self._name

    class FakeStage:
        def Traverse(self):
            return [
                FakePrim("/World/Kitchen_Wall001", "Kitchen_Wall001", [2.8, -1, 0], [2.9, 2, 2.5]),
                FakePrim("/World/Kitchen_Wall_Group", "Kitchen_Wall_Group", [-2, -2, 0], [4, 4, 2.5]),
                FakePrim("/World/Kitchen_Cabinet_Door001", "Kitchen_Cabinet_Door001", [1, 1, 0], [1.05, 1.5, 0.8]),
                FakePrim("/World/Kitchen_Floor", "Kitchen_Floor", [-2, -2, 0], [4, 4, 0.05]),
                FakePrim("/World/RectLight_01", "RectLight_01", [0, 0, 2], [1, 1, 2.1]),
            ]

    class FakeBBoxCache:
        def __init__(self, *_args, **_kwargs):
            pass

        def ComputeWorldBound(self, prim):
            return FakeBound(prim.box)

    fake_usd = types.SimpleNamespace(TimeCode=types.SimpleNamespace(Default=lambda: "default"))
    fake_usd_geom = types.SimpleNamespace(
        Tokens=types.SimpleNamespace(default_="default", render="render", proxy="proxy"),
        BBoxCache=FakeBBoxCache,
    )
    fake_pxr = types.SimpleNamespace(Usd=fake_usd, UsdGeom=fake_usd_geom)
    monkeypatch.setitem(sys.modules, "pxr", fake_pxr)
    monkeypatch.setitem(sys.modules, "pxr.Usd", fake_usd)
    monkeypatch.setitem(sys.modules, "pxr.UsdGeom", fake_usd_geom)

    obstacles = M._placement_shell_obstacles_for_stage(FakeStage())

    assert [obj.id for obj in obstacles] == ["world_kitchen_wall001"]
    assert obstacles[0].source == "usd_shell"
    assert obstacles[0].bbox_min == (2.8, -1.0, 0.0)


def test_placement_shell_obstacles_synthesize_broad_wall_mesh_edges(monkeypatch) -> None:
    class FakeRangeBox:
        def __init__(self, bmin, bmax):
            self._min = bmin
            self._max = bmax

        def IsEmpty(self):
            return False

        def GetMin(self):
            return self._min

        def GetMax(self):
            return self._max

        def GetSize(self):
            return [self._max[i] - self._min[i] for i in range(3)]

    class FakeBound:
        def __init__(self, box):
            self._box = box

        def ComputeAlignedBox(self):
            return self._box

    class FakePrim:
        def __init__(self, path, name, bmin, bmax, is_mesh=False):
            self._path = path
            self._name = name
            self.box = FakeRangeBox(bmin, bmax)
            self._is_mesh = is_mesh

        def GetPath(self):
            return self._path

        def GetName(self):
            return self._name

        def IsA(self, type_marker):
            return self._is_mesh and type_marker == "MeshType"

    class FakeStage:
        def Traverse(self):
            return [
                FakePrim(
                    "/root/Kitchen_Wall001",
                    "Kitchen_Wall001",
                    [-2.4, -1.6, 0.0],
                    [2.76, 2.61, 2.7],
                    is_mesh=True,
                )
            ]

    class FakeBBoxCache:
        def __init__(self, *_args, **_kwargs):
            pass

        def ComputeWorldBound(self, prim):
            return FakeBound(prim.box)

    fake_usd = types.SimpleNamespace(TimeCode=types.SimpleNamespace(Default=lambda: "default"))
    fake_usd_geom = types.SimpleNamespace(
        Tokens=types.SimpleNamespace(default_="default", render="render", proxy="proxy"),
        BBoxCache=FakeBBoxCache,
        Mesh="MeshType",
    )
    fake_pxr = types.SimpleNamespace(Usd=fake_usd, UsdGeom=fake_usd_geom)
    monkeypatch.setitem(sys.modules, "pxr", fake_pxr)
    monkeypatch.setitem(sys.modules, "pxr.Usd", fake_usd)
    monkeypatch.setitem(sys.modules, "pxr.UsdGeom", fake_usd_geom)

    obstacles = M._placement_shell_obstacles_for_stage(FakeStage())

    assert [obj.id for obj in obstacles] == [
        "root_kitchen_wall001_xmin",
        "root_kitchen_wall001_xmax",
        "root_kitchen_wall001_ymin",
        "root_kitchen_wall001_ymax",
    ]
    xmax = obstacles[1]
    assert xmax.bbox_min[0] == pytest.approx(2.72)
    assert xmax.bbox_max[0] == pytest.approx(2.8)
    assert xmax.source == "usd_shell"


def test_usd_task_target_resolver_prefers_object_root_over_descendant(monkeypatch) -> None:
    class FakeBox:
        def __init__(self, center, size):
            self._center = center
            self._size = size

        def IsEmpty(self):
            return False

        def GetCenter(self):
            return self._center

        def GetSize(self):
            return self._size

    class FakeRangeBox:
        def __init__(self, bmin, bmax):
            self._min = bmin
            self._max = bmax

        def IsEmpty(self):
            return False

        def GetMin(self):
            return self._min

        def GetMax(self):
            return self._max

        def GetSize(self):
            return [self._max[i] - self._min[i] for i in range(3)]

    class FakeBound:
        def __init__(self, box):
            self._box = box

        def ComputeAlignedBox(self):
            return self._box

    class FakePrim:
        def __init__(self, path, name, center, size, *, range_box=False):
            self._path = path
            self._name = name
            if range_box:
                self.box = FakeRangeBox(
                    [center[i] - size[i] / 2 for i in range(3)],
                    [center[i] + size[i] / 2 for i in range(3)],
                )
            else:
                self.box = FakeBox(center, size)

        def GetPath(self):
            return self._path

        def GetName(self):
            return self._name

    root = FakePrim(
        "/World/Sink054", "Sink054", [2.0, 0.5, 0.9], [1.2, 0.8, 1.0],
        range_box=True,
    )
    child = FakePrim("/World/Sink054/tiny_mesh", "tiny_mesh", [9.0, 9.0, 9.0], [0.1, 0.1, 0.1])

    class FakeStage:
        def Traverse(self):
            return [child, root]

    class FakeBBoxCache:
        def __init__(self, *_args, **_kwargs):
            pass

        def ComputeWorldBound(self, prim):
            return FakeBound(prim.box)

    fake_usd = types.SimpleNamespace(TimeCode=types.SimpleNamespace(Default=lambda: "default"))
    fake_usd_geom = types.SimpleNamespace(
        Tokens=types.SimpleNamespace(default_="default", render="render", proxy="proxy"),
        BBoxCache=FakeBBoxCache,
    )
    fake_pxr = types.SimpleNamespace(Usd=fake_usd, UsdGeom=fake_usd_geom)
    monkeypatch.setitem(sys.modules, "pxr", fake_pxr)
    monkeypatch.setitem(sys.modules, "pxr.Usd", fake_usd)
    monkeypatch.setitem(sys.modules, "pxr.UsdGeom", fake_usd_geom)

    result = M._resolve_task_target_from_stage(FakeStage(), {"target_object_id": "Sink054"})

    assert result["status"] == "resolved"
    assert result["selected"]["prim_path"] == "/World/Sink054"
    assert result["selected"]["center_xyz"] == [2.0, 0.5, 0.9]
    assert result["selected"]["match_kind"] == "exact_prim_name_or_path_segment"


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


def test_build_result_keeps_dynamic_standing_contacts_bounded() -> None:
    res = M.build_result(
        scenarios=[{"scenario_id": "sink_stand"}],
        outcomes=[{"task_success": True}],
        policy_id="blueprint_default_walk_to_target_smoke_policy",
        kitchen_usd="k.usd",
        g1_usd="g1.usd",
        blockers=[],
        physics_articulation_contact_reports=[
            {
                "scenario_id": "sink_stand",
                "status": "completed",
                "contact_event_count": 2,
                "support_contact_event_count": 1,
                "root_pose_teleport_during_physics_settle": False,
            }
        ],
    )
    summary = res["physics_articulation_standing_contact_summary"]
    assert summary["completed_scenario_count"] == 1
    assert summary["all_have_support_contact_evidence"] is True
    assert summary["root_pose_teleport_during_physics_settle"] is False
    assert "standing/contact settle" in res["proof_boundary"]
    assert "not full dynamic locomotion" in res["proof_boundary"].lower()
    assert "not deployment readiness" in res["proof_boundary"].lower()


def test_summarize_physics_articulation_contact_reports_fails_closed_on_missing_support() -> None:
    summary = M.summarize_physics_articulation_contact_reports([
        {
            "status": "completed",
            "contact_event_count": 1,
            "support_contact_event_count": 0,
            "root_pose_teleport_during_physics_settle": False,
        }
    ])
    assert summary["all_completed"] is True
    assert summary["all_have_support_contact_evidence"] is False
    assert "not prove full dynamic locomotion" in summary["claim_boundary"]


def test_build_result_does_not_claim_support_contact_when_none_observed() -> None:
    res = M.build_result(
        scenarios=[{"scenario_id": "floor_stand"}],
        outcomes=[{"task_success": False}],
        policy_id="blueprint_default_walk_to_target_smoke_policy",
        kitchen_usd="k.usd",
        g1_usd="g1.usd",
        blockers=[],
        physics_articulation_contact_reports=[
            {
                "scenario_id": "floor_stand",
                "status": "completed",
                "contact_event_count": 0,
                "support_contact_event_count": 0,
                "root_pose_teleport_during_physics_settle": False,
            }
        ],
    )
    summary = res["physics_articulation_standing_contact_summary"]
    assert summary["all_completed"] is True
    assert summary["all_have_support_contact_evidence"] is False
    assert "support-contact events were not observed" in res["proof_boundary"]
    assert "does not prove support contact" in res["proof_boundary"]


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


def test_manipulation_pov_geometry_requires_forearm_and_effector_in_frame() -> None:
    eye, target = (0.0, 0.0, 1.0), (1.0, 0.0, 1.0)
    affordance = (1.0, 0.0, 1.0)
    visible = M._manipulation_pov_geometry(
        arm_points={
            "shoulder": (0.1, -0.05, 1.02),
            "elbow": (0.45, -0.05, 1.02),
            "wrist": (0.7, -0.03, 1.01),
            "hand": (0.82, -0.02, 1.0),
        },
        affordance=affordance,
        eye=eye,
        target=target,
        vfov_deg=68.0,
        width=640,
        height=480,
        arm="right",
    )
    assert visible["status"] == "PASS"
    assert {"elbow", "wrist", "hand"}.issubset(set(visible["arm_roles_in_frame"]))
    assert visible["effector_distance_is_metadata_only"] is True
    assert visible["effector_distance_to_affordance_m"]["hand"] > 0.1

    both_visible = M._manipulation_pov_geometry(
        arm_points={},
        arm_points_by_arm={
            "left": {
                "shoulder": (0.1, 0.05, 1.02),
                "elbow": (0.45, 0.05, 1.02),
                "wrist": (0.7, 0.03, 1.01),
                "hand": (0.82, 0.02, 1.0),
            },
            "right": {
                "shoulder": (0.1, -0.05, 1.02),
                "elbow": (0.45, -0.05, 1.02),
                "wrist": (0.7, -0.03, 1.01),
                "hand": (0.82, -0.02, 1.0),
            },
        },
        affordance=affordance,
        eye=eye,
        target=target,
        vfov_deg=68.0,
        width=640,
        height=480,
        arm="both",
    )
    assert both_visible["status"] == "PASS"
    assert both_visible["required_arms"] == ["left", "right"]
    assert set(both_visible["per_arm_geometry"]) == {"left", "right"}
    assert both_visible["arm_extension"]["status"] == "PASS"

    right_only_for_both = M._manipulation_pov_geometry(
        arm_points={},
        arm_points_by_arm={
            "right": {
                "shoulder": (0.1, -0.05, 1.02),
                "elbow": (0.45, -0.05, 1.02),
                "wrist": (0.7, -0.03, 1.01),
                "hand": (0.82, -0.02, 1.0),
            },
        },
        affordance=affordance,
        eye=eye,
        target=target,
        vfov_deg=68.0,
        width=640,
        height=480,
        arm="both",
    )
    assert right_only_for_both["status"] == "FAIL"
    assert "manipulation_pov_left_arm_seed_failed" in right_only_for_both["blockers"]

    cropped = M._manipulation_pov_geometry(
        arm_points={
            "elbow": (-0.4, -0.05, 1.02),
            "wrist": (-0.3, -0.03, 1.01),
            "hand": (-0.2, -0.02, 1.0),
        },
        affordance=affordance,
        eye=eye,
        target=target,
        vfov_deg=68.0,
        width=640,
        height=480,
        arm="right",
    )
    assert cropped["status"] == "FAIL"
    assert "manipulation_pov_arm_not_in_frame" in cropped["blockers"]

    visible_but_hanging = M._manipulation_pov_geometry(
        arm_points={
            "shoulder": (0.3, -0.05, 1.2),
            "elbow": (0.3, -0.05, 1.0),
            "wrist": (0.3, -0.03, 0.82),
            "hand": (0.3, -0.02, 0.7),
        },
        affordance=affordance,
        eye=eye,
        target=target,
        vfov_deg=68.0,
        width=640,
        height=480,
        arm="right",
    )
    assert visible_but_hanging["status"] == "FAIL"
    assert "manipulation_pov_arm_not_extended_forward" in visible_but_hanging["blockers"]
    assert "manipulation_pov_effector_not_near_affordance" not in visible_but_hanging["blockers"]


def test_pov_seed_frame_quality_rejects_black_edge_occlusion(tmp_path) -> None:
    from PIL import Image, ImageDraw  # type: ignore

    clean = tmp_path / "clean.png"
    clean_img = Image.new("RGB", (640, 480), (185, 185, 185))
    ImageDraw.Draw(clean_img).rectangle((260, 120, 380, 330), fill=(8, 8, 8))
    clean_img.save(clean)
    clean_report = M._pov_seed_frame_quality(clean)
    assert clean_report["status"] == "PASS"

    occluded = tmp_path / "occluded.png"
    occ_img = Image.new("RGB", (640, 480), (185, 185, 185))
    ImageDraw.Draw(occ_img).rectangle((520, 0, 640, 480), fill=(0, 0, 0))
    occ_img.save(occluded)
    occ_report = M._pov_seed_frame_quality(occluded)
    assert occ_report["status"] == "FAIL"
    assert "manipulation_pov_edge_self_occlusion" in occ_report["blockers"]


def test_follow_cam_is_behind_and_above_robot() -> None:
    eye, target = M.follow_cam_pose((0.0, 0.0, 0.79), 0.0)  # facing +X
    assert eye[0] < 0.0           # behind the robot along -X
    assert eye[2] > 0.79          # above the root
    assert target[0] > 0.0        # looking ahead toward +X


def _fake_scene_index(monkeypatch, objects, *, obstacle_boxes=None):
    """Patch scene_placement's USD index to enumerate a preset object list (no pxr/GPU)."""
    sp = importlib.import_module("blueprint_pipeline.scene_placement")

    class _FakeIndex:
        def __init__(self, **kw):  # accepts stage=... / usd_path=...
            pass

        def objects(self):
            return list(objects)

        def obstacle_boxes(self):
            return list(objects if obstacle_boxes is None else obstacle_boxes)

    monkeypatch.setattr(sp, "UsdSceneSpatialIndex", _FakeIndex)
    return sp


def test_resolve_task_target_via_scene_placement_maps_task_to_object(monkeypatch) -> None:
    # No object id, no coords — just a natural-language task. The runner enumerates the scene via
    # scene_placement and maps "turn on the faucet" onto the faucet object's center. No hardcoding.
    sp = importlib.import_module("blueprint_pipeline.scene_placement")
    SceneObject = sp.SceneObject
    objs = [
        SceneObject(id="faucet_1", label="faucet", bbox_min=(2.4, 1.0, 0.9),
                    bbox_max=(2.6, 1.3, 1.1), centroid=(2.5, 1.15, 1.0), source="usd"),
        SceneObject(id="stove_1", label="stove", bbox_min=(0.0, 0.0, 0.0),
                    bbox_max=(0.6, 0.6, 0.9), centroid=(0.3, 0.3, 0.45), source="usd"),
    ]
    _fake_scene_index(monkeypatch, objs)

    res = M._resolve_task_target_via_scene_placement(
        stage=object(), scenario={"description": "Stand at the sink and turn on the faucet."}
    )
    assert res is not None and res["status"] == "resolved"
    assert res["source"] == "scene_placement_task_label"
    assert res["selected"]["target_object_id"] == "faucet_1"      # task -> faucet, not stove
    assert res["selected"]["center_xyz"] == [2.5, 1.15, 1.0]      # dynamic center from the scene


def test_resolve_task_target_via_scene_placement_blocks_on_no_match(monkeypatch) -> None:
    sp = importlib.import_module("blueprint_pipeline.scene_placement")
    SceneObject = sp.SceneObject
    objs = [SceneObject(id="rug_1", label="rug", bbox_min=(0, 0, 0),
                        bbox_max=(1, 1, 0.1), centroid=(0.5, 0.5, 0.05), source="usd")]
    _fake_scene_index(monkeypatch, objs)
    res = M._resolve_task_target_via_scene_placement(
        stage=object(), scenario={"task": "turn on the faucet"}
    )
    assert res is not None and res["status"] == "blocked"
    assert "scene_placement_no_task_match" in res["blockers"]


def test_resolve_task_target_via_scene_placement_returns_none_without_task() -> None:
    # No task description at all -> nothing to resolve; defer to the id-driven path (None).
    assert M._resolve_task_target_via_scene_placement(stage=object(), scenario={}) is None


def test_scene_placement_stand_plan_stands_on_clear_floor_when_probe_blocks_wall() -> None:
    # When the probe DOES see the counter/wall (blocks y >= 1.0), it stands on the open floor in front.
    tr = {"status": "resolved", "source": "scene_placement_task_label",
          "selected": {"target_object_id": "sink", "target_object_label": "sink",
                       "center_xyz": [2.28, 1.33, 1.0], "size_xyz": [0.4, 0.4, 0.3]}}
    plan = M._scene_placement_stand_plan(tr, lambda p, y: 0 if p[1] < 1.0 else 1, floor_z=0.05)
    assert plan["status"] == "accepted" and plan["accepted_pose"][1] < 1.0


def test_scene_placement_stand_plan_none_without_geometry() -> None:
    # A resolution lacking center/size -> None, so the caller falls back to plan_task_stance.
    assert M._scene_placement_stand_plan({"selected": {"center_xyz": [1, 2, 3]}}, lambda p, y: 0) is None
    assert M._scene_placement_stand_plan(None, lambda p, y: 0) is None


def test_topdown_debug_overlay_is_added_only_after_verify_and_pov_are_saved() -> None:
    source = _RUNNER.read_text()
    capture_start = source.index("debug_root_path = (")
    normal_render = source.index("rep.orchestrator.step()", capture_start)
    pov_save = source.index('_save_rgb(pov_annot, sdir / "frames" / f"robot_pov_', capture_start)
    verify_save = source.index('_save_rgb(verify_annot, sdir / "frames" / f"verify_', capture_start)
    overlay_update = source.index("_update_topdown_debug_scene(", capture_start)
    topdown_save = source.index("placement_topdown_frame_path =", overlay_update)
    overlay_remove = source.index("stage.RemovePrim(debug_root_path)", topdown_save)

    assert normal_render < pov_save < overlay_update
    assert normal_render < verify_save < overlay_update
    assert overlay_update < topdown_save < overlay_remove


def test_placement_obstacles_use_fine_boxes_not_grouped_cabinet_slab(monkeypatch) -> None:
    sp = importlib.import_module("blueprint_pipeline.scene_placement")
    SceneObject = sp.SceneObject
    sink = SceneObject(
        id="sink",
        label="sink",
        bbox_min=(2.009, 0.896, 0.555),
        bbox_max=(2.547, 1.770, 1.142),
        centroid=(2.278, 1.333, 0.849),
        source="usd",
    )
    broad_cabinet = SceneObject(
        id="kitchen_cabinet_1",
        label="kitchen_cabinet",
        bbox_min=(-1.300, -0.148, -0.014),
        bbox_max=(2.568, 2.459, 0.836),
        centroid=(0.634, 1.156, 0.411),
        source="usd",
    )
    fine_cabinet_back_run = SceneObject(
        id="kitchen_cabinet_leaf_back",
        label="kitchen_cabinet",
        bbox_min=(-1.300, 1.850, -0.014),
        bbox_max=(2.568, 2.459, 0.836),
        centroid=(0.634, 2.155, 0.411),
        source="usd_leaf",
    )
    fine_cabinet_left_run = SceneObject(
        id="kitchen_cabinet_leaf_left",
        label="kitchen_cabinet",
        bbox_min=(-1.300, -0.148, -0.014),
        bbox_max=(-0.700, 2.459, 0.836),
        centroid=(-1.000, 1.156, 0.411),
        source="usd_leaf",
    )
    dishwasher = SceneObject(
        id="dishwasher",
        label="dishwasher",
        bbox_min=(1.936, 0.267, 0.073),
        bbox_max=(2.552, 0.884, 0.751),
        centroid=(2.244, 0.575, 0.412),
        source="usd_leaf",
    )
    flower_on_counter = SceneObject(
        id="kitchen_flowers",
        label="kitchen_flowers",
        bbox_min=(1.806, -0.406, 0.794),
        bbox_max=(2.666, 0.460, 1.451),
        centroid=(2.236, 0.027, 1.122),
        source="usd_leaf",
    )
    _fake_scene_index(
        monkeypatch,
        [sink, broad_cabinet],
        obstacle_boxes=[sink, fine_cabinet_back_run, fine_cabinet_left_run, dishwasher, flower_on_counter],
    )
    monkeypatch.setattr(M, "_placement_shell_obstacles_for_stage", lambda _stage: [])
    monkeypatch.setattr(M, "_place_root", lambda _stage, _prim_path, _pose, _yaw: {"status": "placed"})
    pose = (2.366319, -0.179048, 0.84)
    monkeypatch.setattr(
        M,
        "_world_bbox_for_prim",
        lambda _stage, _prim_path: {
            "bbox_min_xyz": [pose[0] - 0.28, pose[1] - 0.28, 0.22],
            "bbox_max_xyz": [pose[0] + 0.28, pose[1] + 0.28, 1.46],
            "center_xyz": [pose[0], pose[1], pose[2]],
            "size_xyz": [0.56, 0.56, 1.24],
        },
    )

    obstacles = M._placement_obstacles_for_stage(object())
    assert {obj.id for obj in obstacles} == {
        "sink",
        "kitchen_cabinet_leaf_back",
        "kitchen_cabinet_leaf_left",
        "dishwasher",
        "kitchen_flowers",
    }
    assert broad_cabinet.id not in {obj.id for obj in obstacles}

    validator = M._placement_validator_for_stage(
        object(),
        "/World/G1",
        (sink.bbox_min, sink.bbox_max),
        target_object=sink,
        scene_objects=obstacles,
        floor_z=0.05,
    )
    result = validator(pose, 1.629212, {"standoff_from_target_surface_m": 1.0625})

    assert result["status"] == "accepted"
    assert result["blockers"] == []
    assert result["deterministic_geometry"]["ok"] is True
