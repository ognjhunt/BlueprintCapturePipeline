from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from blueprint_pipeline import official_g1_policy_handoff as handoff
from blueprint_pipeline.official_g1_policy_handoff import (
    _base_path_clearance_audit,
    _camera_set,
    _frame_durations_for_realtime_video,
    _navigation_command,
    _plan_occupancy_grid_route,
    _redact_runtime_value,
    _robot_pov_manifest,
    _stream_gate,
    _video_encoding_settings,
    _write_camera_scene_xml,
)


def test_handoff_stream_gate_requires_qpos_qvel_and_policy_observations() -> None:
    incomplete_rows = [
        {
            "qpos": [0.0],
            "joint_positions": {},
            "joint_velocities": {},
            "actuator_controls": [],
            "actuator_forces": [],
            "foot_contact_states": {},
            "command_xyz": [0.5, 0.0, 0.0],
        }
    ]

    gate = _stream_gate(incomplete_rows, control_update_count=0)

    assert gate["passed"] is False
    assert "missing_qvel_stream" in gate["blockers"]
    assert "missing_policy_observation_stream" in gate["blockers"]

    complete_rows = [{**incomplete_rows[0], "qvel": [0.0]}]
    gate = _stream_gate(complete_rows, control_update_count=1)

    assert gate["passed"] is True
    assert gate["blockers"] == []


def test_robot_pov_manifest_is_simulated_body_mounted(tmp_path: Path) -> None:
    manifest = _robot_pov_manifest(
        camera_records=[
            {
                "camera": "robot_pov_head",
                "camera_body_name": "pelvis",
                "simulated_robot_pov": True,
            }
        ],
        robot_pov_frames={"head": [str(tmp_path / "head_0000.png")], "torso": []},
        robot_pov_videos={"head": {"status": "complete"}, "torso": {"status": "not_generated"}},
        render_width=1280,
        render_height=720,
        render_fps=24,
        nonblank_checks={"all_frames_nonblank": True},
        calibration_path=tmp_path / "robot_pov_camera_calibration.json",
    )

    assert manifest["simulated_robot_pov"] is True
    assert manifest["real_robot_pov"] is False
    assert manifest["physical_sensor_data"] is False
    assert manifest["camera_body_name"] == "pelvis"
    assert manifest["render_resolution"] == [1280, 720]


def test_video_settings_respect_resolution_fps_crf_config() -> None:
    settings = _video_encoding_settings(render_fps=30, video_crf=20)

    assert settings["fps"] == 30
    assert settings["video_crf"] == 20
    assert settings["codec"] == "libx264"


def test_frame_durations_preserve_source_sim_time_for_realtime_video() -> None:
    durations, timing = _frame_durations_for_realtime_video(
        frame_count=4,
        render_fps=24,
        frame_times_s=[0.0, 0.5, 1.0, 1.5],
        video_duration_s=2.0,
    )

    assert durations == [0.5, 0.5, 0.5, 0.5]
    assert timing["mode"] == "source_sim_time_realtime"
    assert timing["expected_video_duration_s"] == 2.0


def test_base_path_clearance_audit_blocks_occupied_endpoint() -> None:
    audit = _base_path_clearance_audit(
        base_positions=[[0.0, 0.0, 0.8], [1.0, 0.0, 0.8]],
        collision_proxies=[
            {"name": "box", "pos": [1.0, 0.0, 0.5], "size": [0.25, 0.25, 0.5]}
        ],
        required_clearance_m=0.38,
    )

    assert audit["passed"] is False
    assert audit["endpoint_clearance_m"] == 0.0


def test_navigation_planner_routes_around_occupied_proxy() -> None:
    plan = _plan_occupancy_grid_route(
        start=[-1.0, 0.0, 0.793],
        goal=[1.0, 0.0, 0.793],
        collision_proxies=[
            {"name": "rack", "pos": [0.0, 0.0, 0.5], "size": [0.25, 0.55, 0.5]}
        ],
        mesh_info={"bounds": [[-2.0, -2.0, 0.0], [2.0, 2.0, 1.0]]},
        required_clearance_m=0.20,
        grid_resolution_m=0.20,
    )

    assert plan["status"] == "planned"
    assert plan["route_waypoint_count"] >= 3
    assert plan["route_clearance_audit"]["passed"] is True
    assert plan["route_clearance_audit"]["minimum_clearance_m"] >= 0.20


def test_navigation_planner_blocks_unclean_start() -> None:
    plan = _plan_occupancy_grid_route(
        start=[0.0, 0.0, 0.793],
        goal=[1.0, 0.0, 0.793],
        collision_proxies=[
            {"name": "rack", "pos": [0.0, 0.0, 0.5], "size": [0.25, 0.25, 0.5]}
        ],
        mesh_info={"bounds": [[-2.0, -2.0, 0.0], [2.0, 2.0, 1.0]]},
        required_clearance_m=0.20,
        grid_resolution_m=0.20,
    )

    assert plan["status"] == "blocked"
    assert "start_occupied_or_below_clearance" in plan["blockers"]


def test_navigation_command_converts_waypoint_to_body_velocity() -> None:
    command = _navigation_command(
        route_waypoints=[[0.0, 0.0, 0.793], [1.0, 0.0, 0.793]],
        base_position=[0.0, 0.0, 0.793],
        base_yaw=0.0,
        waypoint_index=1,
        max_speed_mps=0.55,
        waypoint_tolerance_m=0.20,
        yaw_gain=1.2,
        max_yaw_rate=0.9,
    )

    assert command["goal_reached"] is False
    assert command["waypoint_index"] == 1
    assert command["command_xyz"][0] > 0.0
    assert abs(command["command_xyz"][1]) < 1e-9


def test_secret_signature_redaction_for_json_artifacts() -> None:
    signature_query = "x-goog-" + "signature=abc123"
    payload = {
        "url": f"https://storage.googleapis.com/bucket/object?{signature_query}&x=1",
        "nested": ["no-secret"],
    }

    redacted = _redact_runtime_value(payload)

    assert "abc123" not in redacted["url"]
    assert "x-goog-redacted-signature-param=<redacted:signed-url-signature>" in redacted["url"]
    assert signature_query.split("=", 1)[0] + "=" not in redacted["url"]


def test_git_commit_does_not_inherit_parent_repo_commit(tmp_path: Path) -> None:
    parent = tmp_path / "repo"
    child = parent / "snapshot"
    git = parent / ".git"
    ref = git / "refs" / "heads" / "main"
    child.mkdir(parents=True)
    ref.parent.mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main", encoding="utf-8")
    ref.write_text("parent-commit", encoding="utf-8")

    assert handoff._git_commit(child) is None


def test_camera_set_expands_robot_pov() -> None:
    assert _camera_set("overview,robot_pov") == [
        "overview",
        "robot_pov_head",
        "robot_pov_torso",
    ]


def test_camera_scene_xml_can_embed_matching_external_scene_collision_mesh(
    tmp_path: Path,
) -> None:
    robot_xml = tmp_path / "robot.xml"
    source_scene_xml = tmp_path / "source_scene.xml"
    external_scene_obj = tmp_path / "warehouse.obj"
    output_scene_xml = tmp_path / "camera_scene.xml"
    robot_xml.write_text("<mujoco/>", encoding="utf-8")
    source_scene_xml.write_text("<mujoco/>", encoding="utf-8")
    external_scene_obj.write_text("v 0 0 0\n", encoding="utf-8")

    _write_camera_scene_xml(
        source_scene_xml,
        robot_xml,
        output_scene_xml,
        render_width=640,
        render_height=360,
        external_scene_obj=external_scene_obj,
    )

    xml = output_scene_xml.read_text(encoding="utf-8")
    assert 'mesh name="blueprint_external_scene_mesh"' in xml
    assert 'name="blueprint_external_scene_visual"' in xml
    assert 'name="blueprint_external_scene_collision"' in xml
    assert f'file="{external_scene_obj}"' in xml


def _minimal_policy_root(tmp_path: Path) -> Path:
    root = tmp_path / "unitree_rl_gym"
    config = root / "deploy" / "deploy_mujoco" / "configs" / "g1.yaml"
    policy = root / "deploy" / "pre_train" / "g1" / "motion.pt"
    robot_dir = root / "resources" / "robots" / "g1_description"
    scene = robot_dir / "scene.xml"
    robot = robot_dir / "g1_12dof.xml"
    config.parent.mkdir(parents=True)
    policy.parent.mkdir(parents=True)
    robot_dir.mkdir(parents=True)
    config.write_text(
        "\n".join(
            [
                "simulation_dt: 0.002",
                "control_decimation: 1",
                "policy_path: '{LEGGED_GYM_ROOT_DIR}/deploy/pre_train/g1/motion.pt'",
                "xml_path: '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/scene.xml'",
            ]
        ),
        encoding="utf-8",
    )
    policy.write_bytes(b"policy")
    scene.write_text("<mujoco/>", encoding="utf-8")
    robot.write_text(
        '<mujoco><compiler/><asset><mesh file="body.stl"/></asset>'
        '<worldbody><body name="pelvis"/></worldbody></mujoco>',
        encoding="utf-8",
    )
    (robot_dir / "meshes").mkdir()
    (robot_dir / "meshes" / "body.stl").write_text("solid body\nendsolid body\n", encoding="utf-8")
    (root / "LICENSE").write_text("license", encoding="utf-8")
    return root


def test_low_level_handoff_file_pose_and_policy_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert handoff._string(None) == ""
    assert handoff._string(" value ") == "value"
    assert handoff._mapping({"a": 1}) == {"a": 1}
    assert handoff._mapping(["not", "mapping"]) == {}
    assert handoff._as_float_list(None) == []
    assert handoff._as_float_list([[1, "2"], [3, 4]]) == [1.0, 2.0, 3.0, 4.0]
    assert handoff._number(True) is None
    assert handoff._number(None) is None
    assert handoff._number("bad") is None
    assert handoff._number("1.5") == 1.5
    assert handoff._pose_triplet({"x": "1", "Y": "2"}) == (1.0, 2.0, 0.793)
    assert handoff._pose_triplet({"position": ["3", "4", "5"]}) == (3.0, 4.0, 5.0)
    assert handoff._pose_triplet([6, 7]) == (6.0, 7.0, 0.793)
    assert handoff._pose_triplet("not-a-pose") is None
    nested = {"navigation": {"route": {"goal_pose": {"pos_x": 1, "pos_y": 2, "pos_z": 3}}}}
    assert handoff._nested_pose(nested, ("goal_pose",)) == (1.0, 2.0, 3.0)
    assert handoff._nested_pose({"missing": {}}, ("goal_pose",)) is None

    payload_path = tmp_path / "payload.json"
    payload_path.write_text('{"status": "ok"}', encoding="utf-8")
    assert len(handoff._sha256(payload_path)) == 64
    assert handoff._jsonl_count(tmp_path / "missing.jsonl") == 0
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text('\n{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
    assert handoff._jsonl_count(rows_path) == 2
    assert handoff._load_json(tmp_path / "missing.json") == {}
    assert handoff._load_json(payload_path) == {"status": "ok"}
    list_json = tmp_path / "list.json"
    list_json.write_text("[1, 2]", encoding="utf-8")
    assert handoff._load_json(list_json) == {}
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text('\n{"step": 1}\n[2]\n{"step": 2}\n', encoding="utf-8")
    assert handoff._trace_rows(tmp_path / "missing-trace.jsonl") == []
    assert handoff._trace_rows(trace_path) == [{"step": 1}, {"step": 2}]

    signed = "https://example.test/file?x-goog-" + "signature=secret&other=1"
    redacted_path = tmp_path / "redacted.json"
    handoff._safe_write_json(redacted_path, {"nested": (signed, {"url": signed}), "plain": 1})
    redacted_text = redacted_path.read_text(encoding="utf-8")
    assert "secret" not in redacted_text
    jsonl_path = tmp_path / "written.jsonl"
    assert handoff._write_jsonl(jsonl_path, [{"url": signed}, {"ok": True}]) == 2
    assert "secret" not in jsonl_path.read_text(encoding="utf-8")

    monkeypatch.setattr(
        handoff.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout="abc123\n", stderr=""),
    )
    (tmp_path / ".git").mkdir()
    assert handoff._git_commit(tmp_path) == "abc123"
    monkeypatch.setattr(
        handoff.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, stdout="", stderr="fatal"),
    )
    assert handoff._git_commit(tmp_path) is None

    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "status": "complete",
                "scenario_eval_run_count": 2,
                "variation_instance_count": 3,
                "runs": [
                    "skip",
                    {
                        "scenario_eval_run_id": "run-1",
                        "task_id": "task",
                        "robot_pov_required": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    assert handoff._first_matrix_run(matrix)["scenario_eval_run_id"] == "run-1"
    empty_matrix = tmp_path / "empty_matrix.json"
    empty_matrix.write_text('{"runs": []}', encoding="utf-8")
    assert handoff._first_matrix_run(empty_matrix) == {}
    context = handoff._scenario_context(matrix)
    assert context["scenario_eval_run_ids"] == ["run-1"]
    assert context["selected_run"]["task_id"] == "task"

    policy_root = _minimal_policy_root(tmp_path)
    assert handoff._resolve_policy_root(explicit_root=policy_root, manifest={}, handoff_dir=tmp_path) == policy_root.resolve()
    manifest_root = _minimal_policy_root(tmp_path / "manifest")
    assert handoff._resolve_policy_root(
        explicit_root=None,
        manifest={"source_repository": {"local_inspection_root": str(manifest_root)}},
        handoff_dir=tmp_path,
    ) == manifest_root.resolve()
    with pytest.raises(FileNotFoundError, match="missing Unitree RL Gym root"):
        handoff._resolve_policy_root(explicit_root=tmp_path / "missing", manifest={}, handoff_dir=tmp_path)

    copied_root = tmp_path / "copied"
    handoff._copy_tree_files(policy_root, ["LICENSE", "resources/robots/g1_description"], copied_root)
    assert (copied_root / "LICENSE").is_file()
    assert (copied_root / "resources" / "robots" / "g1_description" / "scene.xml").is_file()

    snapshot_dir = tmp_path / "handoff"
    snapshot = handoff._materialize_policy_snapshot(policy_root, snapshot_dir)
    assert snapshot["status"] == "complete"
    snapshot_root = Path(snapshot["snapshot_root"])
    same_snapshot = handoff._materialize_policy_snapshot(snapshot_root, snapshot_dir)
    assert same_snapshot["file_count"] == snapshot["file_count"]

    paths = handoff._policy_paths(policy_root, {"official_artifacts": {"config_path": "/outside/g1.yaml"}})
    assert paths["config"].name == "g1.yaml"
    with pytest.raises(FileNotFoundError, match="missing official Unitree .* artifact"):
        handoff._policy_paths(tmp_path / "broken", {})
    config = handoff._load_policy_config(paths["config"], policy_root)
    assert str(policy_root) in config["policy_path"]
    assert handoff._xml_escape('a&b"c') == "a&amp;b&quot;c"
    assert handoff._xml_float(1 / 3) == "0.333333"
    assert handoff._xml_vec([1, 2.5]) == "1 2.5"


def test_navigation_and_rendering_edge_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert handoff._proxy_xy_distance([0, 0], {"pos": "bad", "size": [1, 1]}) is None
    assert handoff._proxy_xy_distance([2, 0], {"pos": [0, 0], "size": [1, 1]}) == 1.0
    assert handoff._base_path_clearance_audit(
        base_positions=[],
        collision_proxies=[{"pos": [0, 0], "size": [1, 1]}],
        required_clearance_m=0.2,
    )["reason"] == "missing_base_positions"
    assert handoff._base_path_clearance_audit(
        base_positions=[[0, 0, 0.8]],
        collision_proxies=[],
        required_clearance_m=0.2,
    )["reason"] == "missing_collision_proxies"
    assert handoff._base_path_clearance_audit(
        base_positions=[[0, 0, 0.8], [1, 0, 0.8]],
        collision_proxies=[{"pos": "bad", "size": []}],
        required_clearance_m=0.2,
    )["minimum_clearance_m"] is None
    assert handoff._dedupe_route_points([[0, 0, 0.8], [0.01, 0, 0.8], [1, 0, 0.8]]) == [
        (0.0, 0.0, 0.8),
        (1.0, 0.0, 0.8),
    ]
    assert handoff._route_distance([[0, 0, 0], [3, 4, 0]]) == 5.0
    assert handoff._mesh_xy_bounds(
        mesh_info={"bounds": [[-1, -2, 0], [3, 4, 1]]},
        collision_proxies=[{"pos": "bad", "size": []}, {"pos": [5, 6], "size": [1, 2]}],
        start=[0, 0, 0.8],
        goal=[1, 1, 0.8],
        margin_m=0.5,
    ) == (-1.5, -2.5, 6.5, 8.5)
    assert handoff._clearance_sample(point_xy=[1, 2], collision_proxies=[{"pos": "bad", "size": []}]) == {
        "clearance_m": None,
        "xy": [1.0, 2.0],
        "proxy_index": None,
        "proxy_name": None,
    }
    assert handoff._route_clearance_audit(
        route_waypoints=[[0, 0, 0.8]],
        collision_proxies=[],
        required_clearance_m=0.2,
        sample_spacing_m=0.1,
    )["reason"] == "route_requires_at_least_two_waypoints"
    assert handoff._route_clearance_audit(
        route_waypoints=[[0, 0, 0.8], [1, 0, 0.8]],
        collision_proxies=[{"pos": "bad", "size": []}],
        required_clearance_m=0.2,
        sample_spacing_m=0.1,
    )["minimum_clearance_m"] is None
    assert handoff._smooth_route(
        route_waypoints=[[0, 0, 0.8], [1, 0, 0.8]],
        collision_proxies=[],
        required_clearance_m=0.2,
        sample_spacing_m=0.1,
    ) == [(0.0, 0.0, 0.8), (1.0, 0.0, 0.8)]

    no_proxy_plan = handoff._plan_occupancy_grid_route(
        start=[0, 0, 0.8],
        goal=[1, 0, 0.8],
        collision_proxies=[],
        mesh_info={},
        required_clearance_m=0.2,
    )
    assert "missing_collision_proxies_for_occupancy_map" in no_proxy_plan["blockers"]
    occupied_goal_plan = handoff._plan_occupancy_grid_route(
        start=[-1, 0, 0.8],
        goal=[0, 0, 0.8],
        collision_proxies=[{"name": "goal-box", "pos": [0, 0, 0.5], "size": [0.2, 0.2, 0.5]}],
        mesh_info={},
        required_clearance_m=0.2,
    )
    assert "goal_occupied_or_below_clearance" in occupied_goal_plan["blockers"]
    corner_plan = handoff._plan_occupancy_grid_route(
        start=[0, 0, 0.8],
        goal=[0.1, 0.1, 0.8],
        collision_proxies=[{"name": "far", "pos": [10, 10], "size": [0.1, 0.1]}],
        mesh_info={"bounds": [[0, 0], [0.1, 0.1]]},
        required_clearance_m=0.1,
        grid_resolution_m=0.1,
    )
    assert corner_plan["planned"] is True

    real_clearance_sample = handoff._clearance_sample

    def only_start_goal_clear(point_xy, collision_proxies):
        point = tuple(round(float(value), 3) for value in point_xy)
        if point in {(0.0, 0.0), (1.0, 0.0)}:
            return {"clearance_m": 1.0, "xy": list(point), "proxy_index": None, "proxy_name": None}
        return {"clearance_m": 0.0, "xy": list(point), "proxy_index": 0, "proxy_name": "wall"}

    monkeypatch.setattr(handoff, "_clearance_sample", only_start_goal_clear)
    blocked_route = handoff._plan_occupancy_grid_route(
        start=[0, 0, 0.8],
        goal=[1, 0, 0.8],
        collision_proxies=[{"name": "wall", "pos": [10, 10], "size": [0.1, 0.1]}],
        mesh_info={"bounds": [[0, 0], [1, 1]]},
        required_clearance_m=0.2,
        grid_resolution_m=0.5,
    )
    assert blocked_route["blockers"] == ["no_collision_free_occupancy_grid_route"]
    monkeypatch.setattr(handoff, "_clearance_sample", real_clearance_sample)

    wide_plan = handoff._plan_occupancy_grid_route(
        start=[-5, 0, 0.8],
        goal=[5, 0, 0.8],
        collision_proxies=[{"name": "far", "pos": [100, 100], "size": [0.1, 0.1]}],
        mesh_info={"bounds": [[-100, -100], [100, 100]]},
        required_clearance_m=0.1,
        grid_resolution_m=0.1,
    )
    assert wide_plan["occupancy_map"]["resolution_m"] > 0.1

    def failed_route_audit(**kwargs):
        return {
            "status": "failed",
            "passed": False,
            "minimum_clearance_m": 0.0,
            "route_sample_count": 1,
        }

    monkeypatch.setattr(handoff, "_route_clearance_audit", failed_route_audit)
    failed_audit_plan = handoff._plan_occupancy_grid_route(
        start=[0, 0, 0.8],
        goal=[1, 0, 0.8],
        collision_proxies=[{"name": "far", "pos": [10, 10], "size": [0.1, 0.1]}],
        mesh_info={"bounds": [[0, -1], [1, 1]]},
        required_clearance_m=0.2,
        grid_resolution_m=0.5,
    )
    assert "planned_route_clearance_audit_failed" in failed_audit_plan["blockers"]
    with monkeypatch.context() as edge_context:
        edge_context.setattr(handoff, "_mesh_xy_bounds", lambda **kwargs: (0.0, 0.0, 0.1, 0.1))
        edge_plan = handoff._plan_occupancy_grid_route(
            start=[0, 0, 0.8],
            goal=[0.1, 0.1, 0.8],
            collision_proxies=[{"name": "far", "pos": [10, 10], "size": [0.1, 0.1]}],
            mesh_info={},
            required_clearance_m=0.1,
            grid_resolution_m=0.1,
        )
        assert edge_plan["schema_version"] == "official_unitree_g1_navigation_plan.v1"

    assert handoff._default_navigation_goal(
        start=[0, 0, 0.8],
        mesh_info={"bounds": [[-2, -2], [2, 2]]},
        collision_proxies=[{"name": "far", "pos": [10, 10], "size": [0.1, 0.1]}],
        required_clearance_m=0.2,
    ) is not None
    assert handoff._default_navigation_goal(
        start=[0, 0, 0.8],
        mesh_info={"bounds": [[-2, -2], [2, 2]]},
        collision_proxies=[{"name": "big", "pos": [0, 0], "size": [10, 10]}],
        required_clearance_m=0.2,
    ) is None

    assert handoff._navigation_command(
        route_waypoints=[],
        base_position=[0, 0, 0.8],
        base_yaw=0,
        waypoint_index=0,
        max_speed_mps=0.5,
        waypoint_tolerance_m=0.2,
        yaw_gain=1.0,
        max_yaw_rate=0.5,
    )["active_waypoint"] is None
    assert handoff._navigation_command(
        route_waypoints=[[0, 0, 0.8], [1, 0, 0.8], [2, 0, 0.8]],
        base_position=[1.0, 0.0, 0.8],
        base_yaw=math.pi,
        waypoint_index=1,
        max_speed_mps=0.5,
        waypoint_tolerance_m=0.2,
        yaw_gain=2.0,
        max_yaw_rate=0.75,
    )["command_xyz"][0] < 0.5
    assert handoff._navigation_command(
        route_waypoints=[[0, 0, 0.8], [1, 0, 0.8], [2, 0, 0.8]],
        base_position=[0.0, 0.0, 0.8],
        base_yaw=0,
        waypoint_index=1,
        max_speed_mps=0.5,
        waypoint_tolerance_m=0.2,
        yaw_gain=1.0,
        max_yaw_rate=0.5,
    )["waypoint_index"] == 1
    assert handoff._navigation_command(
        route_waypoints=[[0, 0, 0.8], [1, 0, 0.8]],
        base_position=[1.0, 0.0, 0.8],
        base_yaw=0,
        waypoint_index=1,
        max_speed_mps=0.5,
        waypoint_tolerance_m=0.2,
        yaw_gain=1.0,
        max_yaw_rate=0.5,
    )["goal_reached"] is True

    source_robot = tmp_path / "source_robot.xml"
    output_robot = tmp_path / "out" / "robot.xml"
    source_robot.write_text(
        '<mujoco><compiler/><asset><mesh file="mesh.stl"/></asset>'
        '<worldbody><body name="pelvis"><camera name="robot_pov_head"/></body></worldbody></mujoco>',
        encoding="utf-8",
    )
    (tmp_path / "meshes").mkdir()
    handoff._write_camera_robot_xml(source_robot, output_robot)
    output_xml = output_robot.read_text(encoding="utf-8")
    assert 'meshdir="' in output_xml
    assert "robot_pov_torso" in output_xml
    bad_robot = tmp_path / "bad_robot.xml"
    bad_robot.write_text("<mujoco/>", encoding="utf-8")
    with pytest.raises(RuntimeError, match="pelvis"):
        handoff._write_camera_robot_xml(bad_robot, tmp_path / "bad_out.xml")

    scene_output = tmp_path / "proxy_scene.xml"
    handoff._write_camera_scene_xml(
        tmp_path / "scene.xml",
        output_robot,
        scene_output,
        render_width=320,
        render_height=240,
        external_scene_obj=tmp_path / "scene.obj",
        external_collision_proxies=[
            {"pos": "bad", "size": []},
            {"pos": [1, 2, 3], "size": [0.1, 0.2, 0.3]},
        ],
    )
    proxy_xml = scene_output.read_text(encoding="utf-8")
    assert "blueprint_external_collision_proxy_001" in proxy_xml


def test_policy_math_mujoco_stream_video_and_cli_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gravity = handoff._gravity_orientation(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert gravity.tolist() == [0.0, -0.0, -1.0]
    assert np.allclose(
        handoff._pd_control(
            np.array([1.0]),
            np.array([0.25]),
            np.array([10.0]),
            np.array([0.0]),
            np.array([0.5]),
            np.array([2.0]),
        ),
        np.array([6.5]),
    )
    assert handoff._yaw_from_quat_wxyz([1, 0, 0, 0]) == 0.0

    class FakeMujoco:
        class mjtObj:
            mjOBJ_JOINT = "joint"
            mjOBJ_BODY = "body"
            mjOBJ_GEOM = "geom"
            mjOBJ_ACTUATOR = "actuator"

        class mjtCamera:
            mjCAMERA_FREE = "free"

        class MjvCamera:
            def __init__(self):
                self.type = None
                self.lookat = [0.0, 0.0, 0.0]
                self.distance = 0.0
                self.azimuth = 0.0
                self.elevation = 0.0

        @staticmethod
        def mj_name2id(model, obj_type, name):
            if obj_type == "joint":
                return model.joint_ids.get(name, -1)
            if obj_type == "body":
                return model.body_ids.get(name, -1)
            return -1

        @staticmethod
        def mj_id2name(model, obj_type, identifier):
            if obj_type == "body":
                return model.body_names.get(identifier)
            if obj_type == "geom":
                return model.geom_names.get(identifier)
            if obj_type == "actuator":
                return f"actuator_{identifier}"
            return None

        @staticmethod
        def mj_contactForce(model, data, index, force):
            force[:] = [3.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    model = SimpleNamespace(
        joint_ids={name: index for index, name in enumerate(handoff.JOINT_NAMES)},
        jnt_qposadr=list(range(7, 7 + len(handoff.JOINT_NAMES))),
        jnt_dofadr=list(range(6, 6 + len(handoff.JOINT_NAMES))),
        body_ids={"pelvis": 0, "left_ankle_roll_link": 1, "right_ankle_roll_link": 2},
        body_names={0: "pelvis", 1: "left_ankle_roll_link", 2: "right_ankle_roll_link"},
        geom_bodyid=[1, 0],
        geom_names={0: "left_foot_geom", 1: "blueprint_external_collision_proxy_000"},
    )
    assert handoff._joint_addresses(model, FakeMujoco)[0]["qpos_addr"] == 7
    broken_model = SimpleNamespace(**{**model.__dict__, "joint_ids": {}})
    with pytest.raises(RuntimeError, match="missing joint"):
        handoff._joint_addresses(broken_model, FakeMujoco)
    assert handoff._body_id(model, FakeMujoco, "missing") is None

    contact = SimpleNamespace(geom1=0, geom2=1, dist=0.01, pos=[1, 2, 3], frame=[1, 0, 0])
    data = SimpleNamespace(
        xpos=np.array([[0, 0, 0.8], [0, 0, 0], [0, 0, 0]], dtype=float),
        xquat=np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=float),
        cvel=np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0.3, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=float),
        ncon=1,
        contact=[contact],
    )
    assert handoff._body_record(model, data, FakeMujoco, "missing") is None
    assert handoff._body_record(model, data, FakeMujoco, "pelvis")["body_name"] == "pelvis"
    contacts, foot_summary = handoff._contact_records(model, data, FakeMujoco)
    assert contacts[0]["scene_collision_contact"] is True
    assert foot_summary["left_ankle_roll_link"]["slip_indicator"] is True

    obs_data = SimpleNamespace(
        qpos=np.array([0, 0, 0.8, 1, 0, 0, 0] + [0.1] * len(handoff.JOINT_NAMES), dtype=float),
        qvel=np.array([0, 0, 0, 0.1, 0.2, 0.3] + [0.4] * len(handoff.JOINT_NAMES), dtype=float),
    )
    obs = handoff._observation(
        data=obs_data,
        action=np.ones(len(handoff.JOINT_NAMES), dtype=np.float32),
        default_angles=np.zeros(len(handoff.JOINT_NAMES), dtype=np.float32),
        dof_pos_scale=1.0,
        dof_vel_scale=0.5,
        ang_vel_scale=0.25,
        cmd=np.array([0.5, 0.0, 0.0], dtype=np.float32),
        cmd_scale=np.array([2.0, 2.0, 0.25], dtype=np.float32),
        counter=1,
        simulation_dt=0.02,
        num_actions=len(handoff.JOINT_NAMES),
        num_obs=47,
    )
    assert obs.shape == (47,)
    assert obs[6] == 1.0

    side_camera = handoff._render_side_camera(FakeMujoco)
    follow_camera = handoff._render_follow_camera(FakeMujoco, [1, 2, 3])
    assert side_camera.type == "free"
    assert follow_camera.lookat == [1.55, 2.0, 3.55]
    assert handoff._camera_output_path(
        camera="robot_pov_head",
        sample_index=2,
        frames_dir=tmp_path / "frames",
        robot_pov_frames_dir=tmp_path / "pov",
    ).name == "robot_pov_head_0002.png"
    assert handoff._camera_output_path(
        camera="overview",
        sample_index=2,
        frames_dir=tmp_path / "frames",
        robot_pov_frames_dir=tmp_path / "pov",
    ).name == "official_policy_overview_0002.png"

    class FakeRenderer:
        def __init__(self):
            self.calls = []

        def update_scene(self, data, camera):
            self.calls.append(camera)

        def render(self):
            return np.zeros((1, 1, 3), dtype=np.uint8)

    renderer = FakeRenderer()
    for camera in ("overview", "side", "follow", "robot_pov_head"):
        assert handoff._render_frame(
            renderer=renderer,
            mujoco=FakeMujoco,
            data=object(),
            camera=camera,
            base_position=[0, 0, 0.8],
        ).shape == (1, 1, 3)
    with pytest.raises(ValueError, match="unsupported camera"):
        handoff._render_frame(
            renderer=renderer,
            mujoco=FakeMujoco,
            data=object(),
            camera="bad",
            base_position=[0, 0, 0.8],
        )

    assert handoff._frame_durations_for_realtime_video(frame_count=0, render_fps=24)[1]["mode"] == "no_frames"
    assert handoff._frame_durations_for_realtime_video(
        frame_count=2,
        render_fps=10,
        frame_times_s=[0.0],
    )[1]["mode"] == "fixed_render_fps"
    assert handoff._frame_durations_for_realtime_video(
        frame_count=2,
        render_fps=10,
        frame_times_s=[1.0, 0.5],
    )[1]["mode"] == "fixed_render_fps_non_monotonic_source_times"
    assert handoff._frame_durations_for_realtime_video(
        frame_count=1,
        render_fps=10,
        frame_times_s=[1.0],
    )[0] == [0.1]
    assert handoff._frame_durations_for_realtime_video(
        frame_count=2,
        render_fps=10,
        frame_times_s=[0.0, 0.25],
    )[0] == [0.25, 0.25]

    assert handoff._write_frame_video(
        camera="overview",
        frame_paths=["one.png"],
        output_dir=tmp_path,
        render_fps=24,
        video_crf=18,
    )["reason"] == "requires_at_least_two_frames"
    monkeypatch.setattr(handoff.shutil, "which", lambda name: None)
    assert handoff._write_frame_video(
        camera="overview",
        frame_paths=["one.png", "two.png"],
        output_dir=tmp_path,
        render_fps=24,
        video_crf=18,
    )["reason"] == "ffmpeg_unavailable"

    monkeypatch.setattr(handoff.shutil, "which", lambda name: f"/fake/{name}")
    monkeypatch.setattr(
        handoff.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 2, stdout="", stderr="failed"),
    )
    assert handoff._write_frame_video(
        camera="overview",
        frame_paths=["one.png", "two.png"],
        output_dir=tmp_path,
        render_fps=24,
        video_crf=18,
    )["reason"] == "ffmpeg_failed"

    def fake_ffmpeg(command, **kwargs):
        Path(command[-1]).write_bytes(b"video")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(handoff.subprocess, "run", fake_ffmpeg)
    video = handoff._write_frame_video(
        camera="overview",
        frame_paths=["one'quote.png", "two.png"],
        output_dir=tmp_path / "videos",
        render_fps=24,
        video_crf=18,
        frame_times_s=[0.0, 0.5],
        video_duration_s=1.0,
    )
    assert video["status"] == "complete"
    assert Path(video["concat_list_path"]).is_file()

    monkeypatch.setattr(handoff.shutil, "which", lambda name: None)
    assert handoff._ffprobe_video(tmp_path / "missing.mp4")["status"] == "not_checked"
    movie = tmp_path / "movie.mp4"
    movie.write_bytes(b"movie")
    monkeypatch.setattr(handoff.shutil, "which", lambda name: "/fake/ffprobe")
    monkeypatch.setattr(
        handoff.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 1, stdout="", stderr="bad probe"),
    )
    assert handoff._ffprobe_video(movie)["reason"] == "ffprobe_failed"
    monkeypatch.setattr(
        handoff.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"streams": [{"width": 1280, "height": 720}]}),
            stderr="",
        ),
    )
    assert handoff._ffprobe_video(movie)["width"] == 1280

    assert handoff._camera_set(None) == ["overview", "side", "follow", "robot_pov_head", "robot_pov_torso"]
    assert handoff._camera_set(["overview,side", "follow"]) == ["overview", "side", "follow"]
    with pytest.raises(ValueError, match="unsupported camera-set"):
        handoff._camera_set("overview,bad")
    assert handoff._stream_gate([], control_update_count=0)["blockers"][0] == "missing_timeseries_rows"
    manifest = handoff._build_sensor_stream_manifest(
        path=tmp_path / "timeseries.jsonl",
        rows=[
            {
                "qpos": [],
                "qvel": [],
                "joint_positions": {},
                "joint_velocities": {},
                "actuator_controls": [],
                "actuator_forces": [],
                "foot_contact_states": {},
                "command_xyz": [],
            }
        ],
        control_update_count=1,
        row_count=1,
    )
    assert manifest["status"] == "complete"

    empty_overlay_path = tmp_path / "empty_overlay.json"
    empty_overlay = handoff._worldlabs_asset_overlay_manifest(tmp_path, empty_overlay_path)
    assert empty_overlay["status"] == "blocked"
    assets_dir = tmp_path / "pipeline" / "worldlabs_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    collider = tmp_path / "collider.glb"
    collider.write_bytes(b"glb")
    (assets_dir / "materialized_assets_manifest.json").write_text(
        json.dumps(
            {
                "world_id": "world-1",
                "downloads": [
                    {"kind": "collider_mesh_glb", "local_path": str(collider)},
                    {"kind": "splat_spz", "local_path": "overlay.spz"},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(handoff, "_glb_visual_summary", lambda path: {"textures_count": 1})
    overlay = handoff._worldlabs_asset_overlay_manifest(tmp_path, tmp_path / "overlay.json")
    assert overlay["status"] == "complete"
    assert overlay["mujoco_texture_claim_allowed"] is True

    def fake_build(**kwargs):
        assert kwargs["command_xyz"] == [0.5, 0.2, 0.0]
        assert kwargs["navigation_goal_xyz"] == [1.0, 2.0, 0.793]
        assert kwargs["enable_navigation_planner"] is False
        assert kwargs["copy_policy_source_snapshot"] is False
        return {
            "status": "blocked",
            "manifest_path": str(tmp_path / "manifest.json"),
            "official_policy_execution_proven": True,
            "fresh_policy_rollout_proven": True,
            "walking_motion_proven": False,
            "planner_navigation_layer_integrated": False,
            "navigation_planner_status": "not_requested",
            "navigation_goal_reached": False,
            "navigation_runtime_clearance_violation_count": 0,
            "navigation_route_distance_m": None,
            "training_grade_policy_rollout_proven": False,
            "robot_team_handoff_dataset_status": "blocked",
            "simulated_robot_pov_status": "blocked",
            "high_quality_video_status": "blocked",
            "artifacts": {"a": "b"},
            "blockers": ["video"],
            "proof_boundary": {"physical_robot_readiness_proven": False},
        }

    simulator_output = tmp_path / "simulator_output.json"
    monkeypatch.setattr(handoff, "build_official_g1_policy_handoff", fake_build)
    monkeypatch.setenv("BLUEPRINT_SIMULATOR_OUTPUT", str(simulator_output))
    result = handoff.main(
        [
            "--capture-root",
            str(tmp_path),
            "--command-y",
            "0.2",
            "--goal-x",
            "1",
            "--goal-y",
            "2",
            "--disable-navigation-planner",
            "--no-policy-source-snapshot",
        ]
    )
    assert result == 1
    captured = capsys.readouterr()
    assert str(tmp_path / "manifest.json") in captured.out
    assert json.loads(simulator_output.read_text(encoding="utf-8"))["status"] == "blocked"

    monkeypatch.delenv("BLUEPRINT_SIMULATOR_OUTPUT")
    monkeypatch.setenv("BLUEPRINT_CAPTURE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        handoff,
        "build_official_g1_policy_handoff",
        lambda **kwargs: {"status": "complete", "manifest_path": str(tmp_path / "ok.json")},
    )
    assert handoff.main([]) == 0
    monkeypatch.delenv("BLUEPRINT_CAPTURE_ROOT")
    with pytest.raises(SystemExit):
        handoff.main([])


def test_build_official_g1_policy_handoff_with_fake_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_root = _minimal_policy_root(tmp_path)
    capture_root = tmp_path / "capture"
    capture_root.mkdir()
    trace_path = capture_root / "trace.jsonl"
    trace_path.write_text('{"step": 0}\n', encoding="utf-8")
    policy_manifest = capture_root / "policy_manifest.json"
    policy_manifest.write_text(
        json.dumps(
            {
                "status": "completed",
                "execution": {"trace_path": str(trace_path)},
                "metrics": {
                    "finite_state": True,
                    "finite_actions": True,
                    "duration_seconds_requested": 0.004,
                    "command_xyz": [0.5, 0.0, 0.0],
                },
            }
        ),
        encoding="utf-8",
    )

    runtime_mode = {"motion": "walk", "nan_action": False}

    class FakeTensor:
        def __init__(self, array):
            self.array = np.asarray(array, dtype=np.float32)

        def unsqueeze(self, _dim):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.array

    class FakePolicy:
        def eval(self):
            return None

        def __call__(self, _obs):
            if runtime_mode["nan_action"]:
                return FakeTensor(np.full((1, len(handoff.JOINT_NAMES)), np.nan, dtype=np.float32))
            return FakeTensor(np.zeros((1, len(handoff.JOINT_NAMES)), dtype=np.float32))

    class FakeNoGrad:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_torch = SimpleNamespace(
        jit=SimpleNamespace(load=lambda *args, **kwargs: FakePolicy()),
        from_numpy=lambda array: FakeTensor(array),
        no_grad=lambda: FakeNoGrad(),
    )

    class FakeModel:
        nq = 7 + len(handoff.JOINT_NAMES)
        nu = len(handoff.JOINT_NAMES)

        def __init__(self):
            self.opt = SimpleNamespace(timestep=0.002)
            self.jnt_qposadr = list(range(7, 7 + len(handoff.JOINT_NAMES)))
            self.jnt_dofadr = list(range(6, 6 + len(handoff.JOINT_NAMES)))
            self.geom_bodyid = [0]
            self.joint_ids = {name: index for index, name in enumerate(handoff.JOINT_NAMES)}
            self.body_ids = {"pelvis": 0, "left_ankle_roll_link": 1, "right_ankle_roll_link": 2}
            self.body_names = {0: "pelvis", 1: "left_ankle_roll_link", 2: "right_ankle_roll_link"}
            self.geom_names = {0: "floor"}

        @classmethod
        def from_xml_path(cls, _path):
            return cls()

    class FakeData:
        def __init__(self, model):
            self.qpos = np.zeros(model.nq, dtype=float)
            self.qpos[2] = 0.793
            self.qpos[3] = 1.0
            self.qvel = np.zeros(6 + len(handoff.JOINT_NAMES), dtype=float)
            self.ctrl = np.zeros(model.nu, dtype=float)
            self.actuator_force = np.zeros(model.nu, dtype=float)
            self.xpos = np.array([[0, 0, 0.793], [0, 0, 0], [0, 0, 0]], dtype=float)
            self.xquat = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=float)
            self.cvel = np.zeros((3, 6), dtype=float)
            self.ncon = 0
            self.contact = []
            self.time = 0.0

    class FakeRenderer:
        def __init__(self, model, height, width):
            self.height = int(height)
            self.width = int(width)
            self.closed = False

        def update_scene(self, data, camera):
            self.camera = camera

        def render(self):
            image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            image[:, :, 0] = 64
            image[:, :, 1] = 128
            return image

        def close(self):
            self.closed = True

    class FakeMujoco:
        class mjtObj:
            mjOBJ_JOINT = "joint"
            mjOBJ_BODY = "body"
            mjOBJ_GEOM = "geom"
            mjOBJ_ACTUATOR = "actuator"

        class mjtCamera:
            mjCAMERA_FREE = "free"

        class MjvCamera:
            def __init__(self):
                self.type = None
                self.lookat = [0.0, 0.0, 0.0]
                self.distance = 0.0
                self.azimuth = 0.0
                self.elevation = 0.0

        MjModel = FakeModel
        MjData = FakeData
        Renderer = FakeRenderer

        @staticmethod
        def mj_name2id(model, obj_type, name):
            if obj_type == "joint":
                return model.joint_ids.get(name, -1)
            if obj_type == "body":
                return model.body_ids.get(name, -1)
            return -1

        @staticmethod
        def mj_id2name(model, obj_type, identifier):
            if obj_type == "body":
                return model.body_names.get(identifier)
            if obj_type == "geom":
                return model.geom_names.get(identifier)
            if obj_type == "actuator":
                return handoff.JOINT_NAMES[identifier] if identifier < len(handoff.JOINT_NAMES) else None
            return None

        @staticmethod
        def mj_contactForce(model, data, index, force):
            force[:] = 0

        @staticmethod
        def mj_forward(model, data):
            return None

        @staticmethod
        def mj_step(model, data):
            data.time += float(model.opt.timestep)
            if runtime_mode["motion"] == "walk":
                data.qpos[0] += 0.12
                data.qvel[0] = 0.12 / max(float(model.opt.timestep), 1e-9)
            elif runtime_mode["motion"] == "fall":
                data.qpos[2] = 0.1
            elif runtime_mode["motion"] == "nan_state":
                data.qpos[0] = np.nan
            data.actuator_force[:] = data.ctrl * 0.1
            data.xpos[0] = data.qpos[:3]

    def fake_write_frame_video(camera, frame_paths, output_dir, render_fps, video_crf, **kwargs):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{camera}.mp4"
        path.write_bytes(b"video")
        return {
            "status": "complete",
            "path": str(path),
            "frame_count": len(frame_paths),
            "encoding": handoff._video_encoding_settings(render_fps=render_fps, video_crf=video_crf),
        }

    scene_glb = capture_root / "scene.glb"
    scene_glb.write_bytes(b"glb")

    monkeypatch.setitem(sys.modules, "mujoco", FakeMujoco)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(handoff.platform, "system", lambda: "Linux")
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.setattr(handoff, "_find_scene_glb", lambda root: scene_glb)
    def fake_convert_glb_to_obj(glb, obj, collision_proxy_limit):
        obj.write_text("v 0 0 0\n", encoding="utf-8")
        return {
            "status": "complete",
            "bounds": [[-2, -2, 0], [2, 2, 1]],
            "collision_proxy_geoms": [
                {"name": "far_proxy", "pos": [10.0, 10.0, 0.5], "size": [0.1, 0.1, 0.5]}
            ],
        }

    monkeypatch.setattr(handoff, "_convert_glb_to_obj", fake_convert_glb_to_obj)
    monkeypatch.setattr(handoff, "_write_frame_video", fake_write_frame_video)
    monkeypatch.setattr(handoff, "_ffprobe_video", lambda path: {"status": "checked", "width": 1280, "height": 720})
    monkeypatch.setattr(handoff, "_blank_scene_checks", lambda paths: {"all_frames_nonblank": True, "frame_count": len(paths)})

    def run_builder(label: str, **overrides):
        params = {
            "capture_root": capture_root,
            "policy_manifest_path": policy_manifest,
            "unitree_rl_gym_root": policy_root,
            "output_dir": capture_root / f"handoff_{label}",
            "render_width": 16,
            "render_height": 12,
            "render_fps": 8,
            "video_crf": 22,
            "max_frames": 2,
            "camera_set": "overview",
            "duration_seconds": 0.004,
            "base_path_clearance_m": 0.1,
            "navigation_goal_xyz": [1.0, 0.0, 0.793],
            "navigation_waypoint_tolerance_m": 0.05,
            "copy_policy_source_snapshot": False,
        }
        params.update(overrides)
        return handoff.build_official_g1_policy_handoff(**params)

    result = run_builder(
        "happy",
        camera_set="overview,side,follow,robot_pov",
    )

    assert os.environ["MUJOCO_GL"] == "egl"
    assert result["official_policy_execution_proven"] is True
    assert result["fresh_policy_rollout_proven"] is True
    assert result["steps"] == 2
    assert result["control_updates"] == 2
    assert result["planner_navigation_layer_integrated"] is True
    assert result["external_scene_collision_loaded"] is True
    assert Path(result["manifest_path"]).is_file()
    assert Path(result["artifacts"]["sensor_stream_manifest"]).is_file()

    with monkeypatch.context() as no_scene_context:
        no_scene_context.setattr(
            handoff,
            "_find_scene_glb",
            lambda root: (_ for _ in ()).throw(FileNotFoundError("missing scene")),
        )

        def incomplete_video(camera, frame_paths, output_dir, render_fps, video_crf, **kwargs):
            return {
                "status": "not_generated",
                "reason": "test_incomplete",
                "frame_count": len(frame_paths),
                "encoding": handoff._video_encoding_settings(render_fps=render_fps, video_crf=video_crf),
            }

        no_scene_context.setattr(handoff, "_write_frame_video", incomplete_video)
        no_scene = run_builder(
            "no_scene",
            initial_root_xy=[0.2, 0.3],
            navigation_goal_xyz=None,
            copy_policy_source_snapshot=True,
        )
    assert "external_scene_collision_mesh_not_loaded" in no_scene["blockers"]
    assert "navigation_planner_route_not_available" in no_scene["blockers"]
    assert "simulated_robot_pov_incomplete" in no_scene["blockers"]
    assert "high_quality_video_incomplete" in no_scene["blockers"]

    runtime_mode.update({"motion": "walk", "nan_action": False})
    target_reached = run_builder(
        "target_reached",
        enable_navigation_planner=False,
        target_displacement_m=0.05,
    )
    assert target_reached["episode_termination_reason"] == "target_displacement_reached"

    runtime_mode.update({"motion": "fall", "nan_action": False})
    fall = run_builder("fall", enable_navigation_planner=False)
    assert "episode_terminated_by_fall" in fall["blockers"]

    runtime_mode.update({"motion": "static", "nan_action": False})
    with monkeypatch.context() as stream_context:
        stream_context.setattr(
            handoff,
            "_load_policy_config",
            lambda config_path, policy_root: {
                "simulation_dt": 0.002,
                "control_decimation": 99,
                "num_actions": len(handoff.JOINT_NAMES),
                "num_obs": 47,
            },
        )
        missing_stream = run_builder("missing_stream", enable_navigation_planner=False)
    assert "missing_policy_observation_stream" in missing_stream["blockers"]
    assert "base_displacement_below_walking_threshold" in missing_stream["blockers"]

    runtime_mode.update({"motion": "walk", "nan_action": True})
    nan_actions = run_builder("nan_actions", enable_navigation_planner=False)
    assert "non_finite_policy_actions" in nan_actions["blockers"]

    runtime_mode.update({"motion": "nan_state", "nan_action": False})
    nan_state = run_builder("nan_state", enable_navigation_planner=False)
    assert "non_finite_state" in nan_state["blockers"]

    runtime_mode.update({"motion": "walk", "nan_action": False})
    with monkeypatch.context() as contact_context:
        contact_context.setattr(
            handoff,
            "_contact_records",
            lambda model, data, mujoco: (
                [{"scene_collision_contact": True, "geom_names": ["blueprint_external_collision_proxy_000"]}],
                {},
            ),
        )
        collision = run_builder("collision", enable_navigation_planner=False)
    assert "episode_terminated_by_scene_collision" in collision["blockers"]

    with monkeypatch.context() as clearance_context:
        clearance_context.setattr(
            handoff,
            "_plan_occupancy_grid_route",
            lambda **kwargs: {
                "schema_version": "official_unitree_g1_navigation_plan.v1",
                "status": "planned",
                "planned": True,
                "blockers": [],
                "start_pose": [0.0, 0.0, 0.793],
                "goal_pose": [1.0, 0.0, 0.793],
                "route_waypoints": [[0.0, 0.0, 0.793], [1.0, 0.0, 0.793]],
                "route_distance_m": 1.0,
                "route_clearance_audit": {"passed": True},
            },
        )
        clearance_context.setattr(
            handoff,
            "_clearance_sample",
            lambda **kwargs: {
                "clearance_m": 0.0,
                "xy": [0.0, 0.0],
                "proxy_index": 0,
                "proxy_name": "too_close",
            },
        )
        clearance = run_builder("clearance")
    assert "episode_terminated_by_clearance_violation" in clearance["blockers"]
    assert "navigation_runtime_clearance_violation" in clearance["blockers"]

    with monkeypatch.context() as goal_context:
        goal_context.setattr(
            handoff,
            "_navigation_command",
            lambda **kwargs: {
                "command_xyz": [0.0, 0.0, 0.0],
                "waypoint_index": 1,
                "active_waypoint": [1.0, 0.0, 0.793],
                "goal_reached": True,
                "distance_to_active_waypoint_m": 0.0,
                "distance_to_goal_m": 0.0,
            },
        )
        reached = run_builder("goal_reached")
    assert reached["episode_termination_reason"] == "navigation_goal_reached"
