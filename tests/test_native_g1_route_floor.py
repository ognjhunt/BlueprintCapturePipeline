# Covers (for impacted-test selection):
#   src/blueprint_pipeline/native_g1_route_floor.py
#   src/blueprint_pipeline/native_g1_site_registration.py
"""A G1 navigation packet's scanned floor along its route, as sampled heights; never a gate.

2026-11-09, the library job: after a re-posed splat the start moved 1.93 m down the aisle, where the scanned floor
stood 6.5-9.4 cm above the levelled floor; the spawn check read overlap and planning funded no G1 row. The scene fix
needs the floor's profile between the start and the goal.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from blueprint_pipeline import native_g1_route_floor as route
from blueprint_pipeline import native_g1_site_registration as registration

FLOOR_Z = 0.000543


def _quad(x0, x1, y0, y1, z0, z1=None):
    """Two triangles over [x0, x1] x [y0, y1], rising linearly in x from z0 to z1, wound with +z up."""
    z1 = z0 if z1 is None else z1
    a, b, c, d = (x0, y0, z0), (x1, y0, z1), (x0, y1, z0), (x1, y1, z1)
    return [[a, b, c], [b, d, c]]


def _room():
    walked = _quad(-1.0, 4.0, -1.0, 1.0, FLOOR_Z)                     # the levelled walked floor, exactly flat
    patch = _quad(2.0, 3.5, -0.5, 0.6, FLOOR_Z + 0.06, FLOOR_Z + 0.09)  # scanned floor, up to 9 cm, rising in x
    shelf = _quad(1.0, 1.5, -1.0, 1.0, FLOOR_Z + 1.0)                   # above the window
    wall = [[(0.5, -1.0, 0.0), (0.5, 1.0, 0.0), (0.5, -1.0, 2.0)]]       # vertical: not floor-like
    return np.asarray(walked + patch + shelf + wall, dtype=float)


def test_the_samples_follow_the_route_both_sides_and_the_goal_rim():
    rows = route.sample_points([2.965, 0.196], [0.115, 0.183], 0.5)
    route_rows = [row for row in rows if row["kind"] == "route"]
    length = math.dist([2.965, 0.196], [0.115, 0.183])
    assert len(route_rows) == 3 * (math.ceil(length / route.STEP_M) + 1)
    assert route_rows[1]["xy"] == pytest.approx([2.965, 0.196]) and route_rows[-2]["xy"] == pytest.approx([0.115, 0.183])
    assert {row["lateral_m"] for row in route_rows} == {-0.3, 0.0, 0.3}
    rim = [row for row in rows if row["id"].startswith("goal_rim_")]
    assert len(rim) == route.RIM_POINTS
    assert all(math.dist(row["xy"], [0.115, 0.183]) == pytest.approx(0.5) for row in rim)
    with pytest.raises(ValueError, match="g1_route_floor_route_invalid"):
        route.sample_points([1.0, 1.0], [1.0, 1.0], 0.5)


def test_heights_read_the_scanned_floor_apart_from_the_walked_one():
    rows = route.surface_heights(_room(), np.array([[2.75, 0.0], [1.25, 0.0], [0.2, 0.0], [5.0, 0.0]]), FLOOR_Z)
    on_patch, under_shelf, walked_only, outside = rows
    assert on_patch["scanned_top_m"] == pytest.approx(0.075, abs=1e-3) and on_patch["walked_floor"] is True
    # The shelf sits above the window and the wall is not floor-like: only the walked floor is there.
    assert under_shelf == {"scanned_top_m": None, "scanned_faces": 0, "walked_floor": True}
    assert walked_only == {"scanned_top_m": None, "scanned_faces": 0, "walked_floor": True}
    assert outside == {"scanned_top_m": None, "scanned_faces": 0, "walked_floor": False}
    assert route.surface_heights(np.zeros((0, 3, 3)), np.array([[0.0, 0.0]]), FLOOR_Z)[0]["walked_floor"] is False


def _packet(tmp_path, *, goal=True):
    packet = tmp_path / "packet"
    packet.mkdir()
    task = {"g1_navigation_goal": {"center_world_m": [0.115, 0.183, FLOOR_Z], "acceptance_radius_m": 0.5,
                                   "visible_target_marker": {"radius_m": 0.5}}} if goal else {}
    plan = {"plan_digest": "sha256:" + "1" * 64, "task_kind": "goal_region_navigation",
            "robot": {"base_pose_world": {"position_world_m": [2.965, 0.196, 0.7905]}}, "task_spec": task,
            "objects": [{"semantic_role": "scene_collision", "usd_path": "assets/scene.usda", "sha256": "x",
                         "pose_world": None}]}
    (packet / "native_task_arena_scene_plan.v1.json").write_text(json.dumps(plan))
    return packet


def test_the_profile_is_written_once_beside_the_packet(tmp_path, monkeypatch):
    calls = []

    def triangles(path, *, centre, radius, world_from_stage):
        calls.append((centre, radius))
        return _room()

    monkeypatch.setattr("blueprint_pipeline.native_g1_spawn_clearance._bound_file", lambda root, *a: root / "scene.usda")
    monkeypatch.setattr("blueprint_pipeline.scene_physics_preflight.load_background_triangles", triangles)
    packet = _packet(tmp_path)
    value = route.ensure_packet_route_floor_profile(packet)
    assert value["status"] == "measured" and value["floor_z_world_m"] == FLOOR_Z
    assert value["route_length_m"] == pytest.approx(2.85, abs=1e-3)
    start = next(row for row in value["samples"] if row["id"] == "route_000" and row["lateral_m"] == 0.0)
    assert start["scanned_top_m"] == pytest.approx(0.0793, abs=1e-3) and start["walked_floor"] is True
    goal = next(row for row in value["samples"] if row["id"] == "goal_centre")
    assert goal["scanned_top_m"] is None and goal["walked_floor"] is True
    assert value["summary"]["route_centre_scanned_top_m"]["max"] == pytest.approx(0.0793, abs=1e-3)
    assert value["summary"]["points_on_walked_floor"] == value["summary"]["points"]
    assert "vertices" not in json.dumps(value) and value["claim_boundary"].startswith("sampled_heights_only")
    assert json.loads((packet / route.FILENAME).read_text()) == value
    # Written once: the next pass reads it and never loads the room again.
    assert route.ensure_packet_route_floor_profile(packet) == value and len(calls) == 1


def test_an_unreadable_room_is_written_unmeasured_and_never_retried(tmp_path, monkeypatch):
    def unreadable(*args, **kwargs):
        raise OSError("no such file")

    monkeypatch.setattr("blueprint_pipeline.native_g1_spawn_clearance._bound_file", lambda root, *a: root / "scene.usda")
    monkeypatch.setattr("blueprint_pipeline.scene_physics_preflight.load_background_triangles", unreadable)
    packet = _packet(tmp_path)
    value = route.ensure_packet_route_floor_profile(packet)
    assert value["status"] == "unmeasured" and value["reason"].startswith("g1_route_floor_unreadable")
    assert (packet / route.FILENAME).is_file()
    # A packet with no navigation goal has no profile and writes nothing.
    (tmp_path / "other").mkdir()
    other = _packet(tmp_path / "other", goal=False)
    assert route.ensure_packet_route_floor_profile(other) is None and not (other / route.FILENAME).exists()


def test_registration_measures_the_route_and_the_goal_view_beside_the_spawn(tmp_path, monkeypatch):
    packet = _packet(tmp_path)
    for name in ("native_g1_spawn_clearance.v1.json",):
        (packet / name).write_text("{}")
    monkeypatch.setattr(registration, "_packet_dirs", lambda row: [str(packet)])
    monkeypatch.setattr("blueprint_pipeline.native_g1_spawn_clearance.start_offset_reports", lambda *a: {"ok": True})
    # A navigation packet with its spawn and offset reports but no route profile still asks for a measure.
    assert registration._spawn_reports_missing({}) is True
    done = []
    monkeypatch.setattr("blueprint_pipeline.native_g1_spawn_clearance.ensure_packet_spawn_clearance",
                        lambda p: done.append("spawn"))
    monkeypatch.setattr("blueprint_pipeline.native_g1_spawn_clearance.ensure_packet_start_offsets",
                        lambda p: done.append("offsets"))
    monkeypatch.setattr("blueprint_pipeline.native_g1_spawn_clearance.ensure_packet_goal_marker_view",
                        lambda p: done.append("goal_view"))
    monkeypatch.setattr("blueprint_pipeline.native_g1_route_floor.ensure_packet_route_floor_profile",
                        lambda p: (done.append("route"), (packet / route.FILENAME).write_text("{}")))
    registration._measure_spawns({})
    assert done == ["spawn", "offsets", "route", "goal_view"]
    assert registration._spawn_reports_missing({}) is False
