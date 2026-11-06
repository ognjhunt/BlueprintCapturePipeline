"""A website rigid task's support is sealed flat at its success plane where the room's collider is first produced.

Beta journey step: scene preparation, before anything is paid for. These fixtures
rebuild capture-447ae4f4's dining table synthetically (no customer data, the
numbers ``tests/test_rigid_support_pad`` reads from its sealed documents): a
scan 2 mm under the success plane at the start and 6.5 mm under it at the goal,
with +/-5 mm of scan noise, its registered top 12.7 mm over the plane. Around it
a floor, a wall and a side table at the table's height stand outside the
support, and must come through as scanned.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from blueprint_pipeline import rigid_support_pad as pad
from blueprint_pipeline import website_support_pad as source
from blueprint_pipeline.rigid_support_seating import collider_triangles
from blueprint_pipeline.website_task_preparation import SUCCESS
from tests.test_rigid_support_pad import (
    GOAL,
    H0,
    HALF,
    PLANE,
    REGISTERED_TOP,
    START,
    SUPPORT,
    TOP_FACE,
    _grid,
    _scan_heights,
)

pytest.importorskip("pxr")

# The scan's estimate of the tray (the replacement qualified 3.3 % larger) and of its bottom at the start.
ESTIMATED_HALF = (HALF[0] / 1.033, HALF[1] / 1.033)
ESTIMATED_BOTTOM = 0.80189
ESTIMATED_HEIGHT = H0 + TOP_FACE
PLACEMENT_UNCERTAINTY = 0.072  # the registration's task-region residual on capture-447ae4f4


def _preparation(**task_changes: Any) -> dict[str, Any]:
    """A website preparation's fields the support reads, in the simulator frame."""

    lower = [float(START[0]) - ESTIMATED_HALF[0], float(START[1]) - ESTIMATED_HALF[1], ESTIMATED_BOTTOM]
    upper = [float(START[0]) + ESTIMATED_HALF[0], float(START[1]) + ESTIMATED_HALF[1],
             ESTIMATED_BOTTOM + ESTIMATED_HEIGHT]
    task = {"strategy": "pick_and_place",
            "subject": {"description": "tray", "aabb_min_xyz": lower, "aabb_max_xyz": upper},
            "support": {"description": "dining table", "aabb_min_xyz": list(SUPPORT["bounds_min_xyz_m"]),
                        "aabb_max_xyz": list(SUPPORT["bounds_max_xyz_m"])},
            "destination": {"relation": "on", "visible_label": "the table's far end", "mode": "existing_support_surface",
                            "position_world_m": [float(GOAL[0]), float(GOAL[1]), PLANE],
                            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                            "basis": "registered_estimated_visible_bounds_and_collider_contact"},
            "success": dict(SUCCESS)}
    task.update(task_changes)
    return {"status": "intake_ready", "digest": "sha256:" + "1" * 64,
            "subject": {"description": "tray", "aabb_min_xyz": lower, "aabb_max_xyz": upper},
            "coordinate_frame": {"placement_uncertainty_m": PLACEMENT_UNCERTAINTY},
            "authoring_inputs": {"dimension_authority": "estimated",
                                 "metric_envelope": {"maximum_dimension_relative_error": 0.25}},
            "intake_request": {"task": task}}


def _patch(x0: float, x1: float, y0: float, y1: float, z: Any, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = np.arange(x0, x1 + 1e-9, spacing), np.arange(y0, y1 + 1e-9, spacing)
    x, y = np.meshgrid(xs, ys, indexing="ij")
    heights = z(x, y) if callable(z) else np.full_like(x, float(z))
    points = np.stack([x, y, heights], axis=-1).reshape(-1, 3)
    index = np.arange(len(xs) * len(ys)).reshape(len(xs), len(ys))
    a, b, c, d = index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]
    return points, np.concatenate([np.stack([a, b, c], axis=-1).reshape(-1, 3),
                                   np.stack([a, c, d], axis=-1).reshape(-1, 3)])


def _wall(x: float, y0: float, y1: float, height: float, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """A vertical wall at ``x``, from the floor up, reaching through the table's height."""

    points, faces = _patch(0.0, height, y0, y1, 0.0, spacing)
    return points[:, [2, 1, 0]] + [x, 0.0, 0.0], faces


def _room(heights: Any = _scan_heights) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The table's scanned top, and what stands around it outside the support: floor, wall, side table."""

    table_points, table_faces = _grid(heights)
    return {"table": (table_points, table_faces),
            "floor": _patch(-1.2, 0.9, -1.3, 0.8, 0.0, 0.05),
            "wall": _wall(0.4, -1.3, 0.8, 1.2, 0.05),
            # The table's own height, so inside the scan band, but beyond the support's plan bounds.
            "side_table": _patch(0.45, 0.85, -0.6, 0.0, lambda x, y: PLANE + 0.004 * np.sin(20 * x), 0.02)}


def _collider(path: Path, parts: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, slice]:
    """One collision mesh holding every part, as a normalized Atlas collider holds the room; each part's rows."""

    from pxr import Usd, UsdGeom, UsdPhysics, Vt

    rows, points, faces, offset = {}, [], [], 0
    for name, (part_points, part_faces) in parts.items():
        rows[name] = slice(offset, offset + len(part_points))
        points.append(part_points)
        faces.append(part_faces + offset)
        offset += len(part_points)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Root/mesh_0000_geometry_0")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.concatenate(points).astype(np.float32)))
    every = np.concatenate(faces)
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(every), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(every.reshape(-1).astype(np.int32)))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    stage.GetRootLayer().Save()
    return rows


def _points(path: str | Path) -> np.ndarray:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    (mesh,) = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    return np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get(), dtype=np.float32)


def _sha(path: str | Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _qualified(spec: dict[str, Any], *, yaw: float) -> dict[str, Any]:
    """The task as a team half binds it: the qualified replacement (3.3 % over the estimate), turned by ``yaw``."""

    value = copy.deepcopy(spec)
    value["subject_collision_bounds_scoring_frame_m"] = {"minimum": [-HALF[0], -HALF[1], -H0],
                                                         "maximum": [HALF[0], HALF[1], TOP_FACE]}
    disc = value["surface_target"]["surface_position_world_m"]
    goal = [disc[0], disc[1], disc[2] + H0]
    value.update(target_position_world_m=goal, support_height_interval_m=[goal[2] - 0.005, goal[2] + 0.005],
                 destination_position_bounds_world_m={"minimum": [goal[0] - 0.05, goal[1] - 0.05, goal[2] - 0.005],
                                                      "maximum": [goal[0] + 0.05, goal[1] + 0.05, goal[2] + 0.005]},
                 start_pose_world=[float(START[0]), float(START[1]), PLANE + H0, 0.0, 0.0,
                                   math.sin(yaw / 2.0), math.cos(yaw / 2.0)])
    return value


def _inside_convex(polygon, point, tolerance=1e-6) -> bool:
    """Whether ``point`` lies in a counter-clockwise convex ``polygon`` (within ``tolerance``)."""
    count = len(polygon)
    for index in range(count):
        a, b = polygon[index], polygon[(index + 1) % count]
        if (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]) < -tolerance:
            return False
    return True


def test_the_sealed_success_plane_and_window_are_the_ones_native_submission_seals() -> None:
    preparation = _preparation()
    task = source.rigid_support_task(preparation)
    spec, region, support_plane = source.task_specs(preparation, task)
    # The plane is the green disc native submission seals (the surface target), not the registered top.
    plane = pad.success_plane(spec)
    assert plane["basis"] == "success_document_surface_target" and plane["consistent"] is True
    assert plane["plane_z_m"] == PLANE and support_plane["top_z_m"] == REGISTERED_TOP
    from blueprint_pipeline.website_surface_target_admission import sealed_surface_target
    sealed = sealed_surface_target(preparation, destination=task["destination"], support=task["support"],
                                   subject_min=preparation["subject"]["aabb_min_xyz"],
                                   subject_max=preparation["subject"]["aabb_max_xyz"], development=False)
    assert spec["surface_target"] == sealed
    # The window is the native bind's: the estimated tray's rest on the disc, +/- 5 mm.
    rest = PLANE + ESTIMATED_HEIGHT / 2.0
    assert spec["support_height_interval_m"] == pytest.approx([rest - 0.005, rest + 0.005])
    assert spec["reset_translation_tolerance_m"] == 0.002 and spec["reset_orientation_tolerance_rad"] == 0.01
    # The disc is the v2 centre tolerance (G3), smaller than the tray; the region's footprint holds any
    # admitted replacement at any yaw, around the start and about every centre the disc accepts.
    assert sealed["radius_m"] == 0.05 and sealed["success_rule"] == "subject_centre_within_radius"
    lower, upper = preparation["subject"]["aabb_min_xyz"], preparation["subject"]["aabb_max_xyz"]
    reach = math.hypot(*[(upper[axis] - lower[axis]) / 2 * 1.25 for axis in (0, 1)])
    assert reach > math.hypot(*HALF)
    assert region["subject_collision_bounds_scoring_frame_m"]["maximum"][:2] == [reach, reach]
    polygon = pad.pad_region(region, support_bounds=support_plane)
    disc = sealed["surface_position_world_m"]
    # The pad holds a replacement whose centre ends anywhere in the disc, at any yaw.
    corners = [[disc[0] + (0.05 + reach) * math.cos(a), disc[1] + (0.05 + reach) * math.sin(a)]
               for a in [k * math.pi / 8 for k in range(16)]]
    low, high = support_plane["bounds_min_xyz_m"], support_plane["bounds_max_xyz_m"]
    assert all(_inside_convex(polygon, point) for point in corners
               if low[0] <= point[0] <= high[0] and low[1] <= point[1] <= high[1])


def test_capture_447s_table_is_sealed_flat_at_its_success_plane_and_the_room_around_it_as_scanned(tmp_path) -> None:
    room = _room()
    collision = tmp_path / "normalized_scene.usdc"
    rows = _collider(collision, room)
    before = _sha(collision)
    record = source.seal_support_pad(preparation=_preparation(), collision=collision, output_root=tmp_path / "native")
    assert record["status"] == "consistent" and record["hold_reason"] is None and record["reasons"] == []
    assert record["model"] == pad.MODEL and record["applied_at"] == "website_runtime_inputs"
    assert record["scene_revision_collider"] == "planarized"
    assert record["success_document_changed"] is False and record["registered_support_changed"] is False
    assert record["plane_z_m"] == PLANE and record["plane"]["basis"] == "success_document_surface_target"
    assert record["scan_band_z_m"] == pytest.approx([PLANE - (REGISTERED_TOP - PLANE), REGISTERED_TOP])
    assert "sealed_revision_changed" not in record
    # Why: the scan as captured would have refused the tray at its start, as capture-447's probe did.
    assert record["scan_rest_check"]["status"] == "held"
    assert record["scan_rest_check"]["reasons"][0].startswith("rigid_support_start_rest_off_start:")
    deviation = record["measured_scan_deviation"]
    assert deviation["start"]["highest_above_plane_m"] > 0.001 and deviation["region"]["lowest_below_plane_m"] < -0.005
    # The region: any admitted tray at any yaw at the start, and the whole disc, inside the support's bounds.
    region = np.asarray(record["region_world_xy_m"])
    radius = source.task_specs(_preparation(), source.rigid_support_task(_preparation()))[0]["surface_target"][
        "radius_m"]
    corners = [START + [sx * radius, sy * radius] for sx in (-1, 1) for sy in (-1, 1)]
    inside_support = [np.clip(corner, SUPPORT["bounds_min_xyz_m"][:2], SUPPORT["bounds_max_xyz_m"][:2])
                      for corner in corners]
    rim = [GOAL + radius * np.array([math.cos(a), math.sin(a)]) for a in np.linspace(0, 2 * math.pi, 73)]
    assert pad.inside_polygon(region, inside_support + rim).all()
    # The derived collider: the normalized one's own bytes untouched, only point heights moved, and only the
    # table's own scan under the region.
    derived = record["derived_collider"]
    assert _sha(collision) == before == record["source_collider"]["digest"] == derived["derived_from_sha256"]
    assert _sha(derived["path"]) == derived["sha256"] and derived["edit"] == "point_heights_only"
    assert derived["rest_check_read_back"] is True and derived["points_moved"] > 0
    scanned, sealed = _points(collision), _points(derived["path"])
    assert np.array_equal(scanned[:, :2], sealed[:, :2])
    for name in ("floor", "wall", "side_table"):
        assert np.array_equal(scanned[rows[name]], sealed[rows[name]]), name
    table = sealed[rows["table"]]
    moved = table[:, 2] != scanned[rows["table"]][:, 2]
    assert moved.any() and np.all(table[moved, 2] == np.float32(PLANE))
    assert pad.inside_polygon(record["selection_world_xy_m"], table[moved, :2]).all()
    assert (~moved).any()  # the table beyond the region keeps its scan
    # On the sealed collider the tray starts on the plane, at rest, and a correct placement rests in the window.
    check = record["rest_check"]
    assert check["start"]["three_point_tilt_rad"] == pytest.approx(0.0, abs=1e-9)
    assert check["start"]["start_scoring_z_m"] == pytest.approx(PLANE + ESTIMATED_HEIGHT / 2.0, abs=1e-6)
    assert all(row["within_success_window"] for row in check["goal"]["placements"])
    # A retry reuses the receipt and the bytes.
    stamp = Path(derived["path"]).stat().st_mtime_ns
    again = source.seal_support_pad(preparation=_preparation(), collision=collision, output_root=tmp_path / "native")
    assert again == record and Path(derived["path"]).stat().st_mtime_ns == stamp


@pytest.mark.parametrize("yaw", [0.0, 0.3, math.pi / 2])
def test_a_team_half_on_the_collider_sealed_flat_moves_nothing_and_on_the_scan_still_pads(tmp_path, yaw) -> None:
    """#3349's half-level pad, read with the qualified replacement at any yaw: a no-op on the sealed flat
    support, and the same pad as before on a scan sealed before this release."""

    collision = tmp_path / "normalized_scene.usdc"
    _collider(collision, _room())
    preparation = _preparation()
    record = source.seal_support_pad(preparation=preparation, collision=collision, output_root=tmp_path / "native")
    spec, _region, support_plane = source.task_specs(preparation, source.rigid_support_task(preparation))
    qualified = _qualified(spec, yaw=yaw)

    def half(path: Path) -> dict[str, Any]:
        # Read as ``team_scene_packet._seated_rigid_adapter`` reads the sealed collider.
        triangles = collider_triangles(path, points_xy=pad.pad_region(qualified, support_bounds=support_plane).tolist(),
                                       reach_m=pad.MAXIMUM_EDGE_REACH_M * math.sqrt(2.0), height_z=REGISTERED_TOP)
        return pad.plan_support_pad(qualified, support_plane=support_plane, triangles=triangles)

    sealed = half(Path(record["derived_collider"]["path"]))["record"]
    assert sealed["plane_z_m"] == record["plane_z_m"] and sealed["scan_band_z_m"] == record["scan_band_z_m"]
    assert sealed["read_points_moved"] == 0 and sealed["read_points_maximum_shift_m"] < 1e-6
    assert pad.inside_polygon(record["selection_world_xy_m"], sealed["selection_world_xy_m"]).all()
    scanned = half(collision)["record"]
    assert scanned["read_points_moved"] > 0


def test_something_on_the_support_under_the_start_holds_the_preparation_by_name(tmp_path) -> None:
    def with_a_box(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # A book left on the table under one corner of the tray's start: 3 cm, over the support's own band.
        box = (np.abs(x - (START[0] + 0.10)) <= 0.04) & (np.abs(y - (START[1] + 0.18)) <= 0.05)
        return np.where(box, PLANE + 0.03, _scan_heights(x, y))

    collision = tmp_path / "normalized_scene.usdc"
    _collider(collision, _room(with_a_box))
    record = source.seal_support_pad(preparation=_preparation(), collision=collision, output_root=tmp_path / "native")
    assert record["status"] == "held"
    assert record["hold_reason"] == record["reasons"][0]
    assert record["hold_reason"].startswith("rigid_support_start_rest_off_start:")
    assert source.hold_reason(record) == record["hold_reason"]
    # The rest of the support is still sealed flat; the book keeps its shape.
    assert record["scene_revision_collider"] == "planarized"
    assert np.any(np.isclose(_points(record["derived_collider"]["path"])[:, 2], PLANE + 0.03))


def test_the_handoff_holds_it_before_anything_is_registered_enqueued_or_paid(tmp_path, monkeypatch) -> None:
    from blueprint_pipeline import website_native_background, website_object_observations
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.website_scene_handoff import prepare_rigid_native_inputs

    def with_a_box(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        box = (np.abs(x - (START[0] + 0.10)) <= 0.04) & (np.abs(y - (START[1] + 0.18)) <= 0.05)
        return np.where(box, PLANE + 0.03, _scan_heights(x, y))

    collision = tmp_path / "normalized_scene.usdc"
    _collider(collision, _room(with_a_box))
    preparation = _preparation()
    record = source.seal_support_pad(preparation=preparation, collision=collision, output_root=tmp_path / "native")
    for name in ("prepare_collision_stage", "prepare_appearance_stage"):
        monkeypatch.setattr(website_native_background, name, lambda path: {"stage": "stub"})
    monkeypatch.setattr(website_native_background, "prepare_construction_stages", lambda **_: {"references": []})
    monkeypatch.setattr(website_native_background, "construction_rights_admission", lambda **_: {"status": "stub"})
    monkeypatch.setattr(website_object_observations, "_record", lambda path: {"path": str(path)})
    verdict = {"status": "unassessed", "blockers": [], "verdict_digest": "sha256:" + "2" * 64}
    monkeypatch.setattr("blueprint_pipeline.website_rigid_feasibility.rigid_task_verdict", lambda **_: verdict)
    monkeypatch.setattr("blueprint_pipeline.website_robot_fit.robot_fit_table",
                        lambda **_: {"table_digest": "sha256:" + "5" * 64, "counts": {}})
    preparation["digest"] = canonical_digest(preparation, digest_field="digest")
    collision_record = {"path": record["derived_collider"]["path"], "digest": record["derived_collider"]["sha256"],
                        "support_pad": record}
    result = {"status": "intake_ready", "blockers": []}
    admissible = prepare_rigid_native_inputs(
        result=result, preparation=preparation, preparation_path=tmp_path / "preparation.json",
        native_root=tmp_path / "native", task_context={}, now=0.0,
        runtime_inputs={"status": "ready", "digest": "sha256:" + "3" * 64,
                        "appearance": {"status": "native_appearance_authored"}, "collision": collision_record})
    # Not admissible: nothing is registered or enqueued, so no ARTISAN authoring and no GPU follow. A hold.
    assert admissible is False and result["status"] == "needs_input"
    assert result["blockers"] == [record["hold_reason"]]
    assert result["task_support_pad"] == {"status": "held", "scene_revision_collider": "planarized",
                                          "hold_reason": record["hold_reason"]}
    # A consistent support goes on.
    flat = tmp_path / "flat"
    flat.mkdir()
    _collider(flat / "normalized_scene.usdc", _room())
    consistent = source.seal_support_pad(preparation=preparation, collision=flat / "normalized_scene.usdc",
                                         output_root=flat / "native")
    result = {"status": "intake_ready", "blockers": []}
    assert prepare_rigid_native_inputs(
        result=result, preparation=preparation, preparation_path=flat / "preparation.json",
        native_root=flat / "native", task_context={}, now=0.0,
        runtime_inputs={"status": "ready", "digest": "sha256:" + "3" * 64,
                        "appearance": {"status": "native_appearance_authored"},
                        "collision": {"path": consistent["derived_collider"]["path"], "support_pad": consistent}})
    assert result["status"] == "intake_ready" and result["task_support_pad"]["hold_reason"] is None


@pytest.mark.parametrize("change, expected", [
    ({"strategy": "articulated_open_close"}, None),
    ({"destination": {"relation": "inside", "mode": "existing_support_surface",
                      "position_world_m": [float(GOAL[0]), float(GOAL[1]), PLANE],
                      "interior_bounds_world_m": {"minimum": [0, 0, 0], "maximum": [1, 1, 1]}}}, None),
    ({"destination": {"relation": "on", "mode": "existing_support_surface",
                      "position_world_m": [float(GOAL[0]), float(GOAL[1]), 0.45],
                      "support": {"aabb_min_xyz": [0, 0, 0.4], "aabb_max_xyz": [1, 1, 0.45]}}}, None),
    ({"destination": {"relation": "on", "mode": "staged_destination_asset",
                      "position_world_m": [float(GOAL[0]), float(GOAL[1]), PLANE]}}, None),
])
def test_only_a_task_on_one_registered_support_is_modelled(tmp_path, change, expected) -> None:
    """Articulated parts, containers, another surface and a staged destination keep the collider exactly."""

    collision = tmp_path / "normalized_scene.usdc"
    _collider(collision, _room())
    preparation = _preparation(**change)
    assert source.rigid_support_task(preparation) is expected
    assert source.seal_support_pad(preparation=preparation, collision=collision, output_root=tmp_path) is expected
    assert not (tmp_path / source.DIRECTORY).exists()
    development = {**_preparation(), "development_test": {"kind": "authored"}}
    assert source.rigid_support_task(development) is None


def test_a_destination_submission_would_move_is_left_to_the_team_half_by_name(tmp_path) -> None:
    collision = tmp_path / "normalized_scene.usdc"
    _collider(collision, _room())
    # Grounded under the tray's own centre: native submission moves it off the start first.
    near = _preparation(destination={**_preparation()["intake_request"]["task"]["destination"],
                                     "position_world_m": [float(START[0]), float(START[1]) + 0.05, PLANE]})
    record = source.seal_support_pad(preparation=near, collision=collision, output_root=tmp_path / "native")
    assert record["status"] == "not_applied" and record["reason"] == "rigid_support_pad_destination_moved_at_submission"
    assert source.hold_reason(record) is None


def test_an_unreadable_collider_keeps_the_normalized_one_and_says_why(tmp_path) -> None:
    collision = tmp_path / "normalized_scene.usdc"
    collision.write_bytes(b"not a usd layer")
    record = source.seal_support_pad(preparation=_preparation(), collision=collision, output_root=tmp_path / "native")
    assert record["status"] == "not_applied" and record["reason"].startswith("rigid_support_pad_unmodelled:")
    assert source.hold_reason(record) is None


def test_the_blocked_writer_writes_the_bytes_a_whole_array_pass_writes(tmp_path) -> None:
    """The derived collider is written a block of points at a time (a full room: 3-7 M triangles)."""

    from blueprint_pipeline.usd_binary_read_cache import read_scope

    collision = tmp_path / "normalized_scene.usdc"
    _collider(collision, _room())
    planned = json.loads(json.dumps(source.seal_support_pad(preparation=_preparation(), collision=collision,
                                                             output_root=tmp_path / "native")))
    written = {}
    for block in (7, 4096, pad.POINT_BLOCK):
        target = tmp_path / f"block-{block}" / "collision.usdc"
        with read_scope():
            written[block] = pad.write_planarized_collider(collision, target, selection=planned["selection_world_xy_m"],
                                                           band=planned["scan_band_z_m"], plane_z=planned["plane_z_m"],
                                                           block=block)
        assert target.read_bytes() == Path(planned["derived_collider"]["path"]).read_bytes()
    assert written[7] == written[4096] == written[pad.POINT_BLOCK]


# --- where the collider is produced: website runtime inputs, stage 2, the native inputs' validation -----------

FIXTURE_PLANE = 3.5  # tests.test_website_task_preparation's registered tabletop, in the simulator frame


@pytest.fixture
def sagging_scan(monkeypatch):
    """The website fixture's normalized collider with its tabletop scanned up to 6 mm low (never over the
    registered top), as the room collider first comes out of normalization."""

    from pxr import Sdf, Usd, UsdGeom, Vt

    from blueprint_pipeline import website_scene_runtime_inputs as runtime

    real = runtime.normalize_runtime_collision

    def normalized(**kwargs: Any) -> tuple[Path, dict[str, Any]]:
        root, value = real(**kwargs)
        path = root / value["output"]["relative_path"]
        layer = Sdf.Layer.FindOrOpen(str(path))
        stage = Usd.Stage.Open(layer)
        (prim,) = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float32).copy()
        table = points[:, 2] == np.float32(FIXTURE_PLANE)
        x, y = points[table, 0].astype(float), points[table, 1].astype(float)
        points[table, 2] = (FIXTURE_PLANE - 0.003 * (1.0 + np.sin(9.0 * x) * np.cos(7.0 * y))).astype(np.float32)
        UsdGeom.Mesh(prim).GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
        layer.Save()
        # The normalization's own record names the bytes it wrote.
        from blueprint_pipeline.decision_evidence_contracts import canonical_digest

        receipt_path = root / "mesh_normalization.v1.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["output"] = {**receipt["output"], "sha256": _sha(path), "size_bytes": path.stat().st_size}
        receipt["normalization_digest"] = canonical_digest(receipt, digest_field="normalization_digest")
        receipt_path.write_text(json.dumps(receipt))
        return root, {**value, "output": receipt["output"], "normalization_digest": receipt["normalization_digest"]}

    monkeypatch.setattr(runtime, "normalize_runtime_collision", normalized)


def test_the_runtime_collider_stage_two_copies_is_the_planarized_one_with_its_record(tmp_path, sagging_scan) -> None:
    from tests.test_website_native_background import execute, staged

    runtime, inputs, envelope, config_path = staged(tmp_path)
    collision = runtime["collision"]
    record = collision["support_pad"]
    assert record["status"] == "consistent" and record["scene_revision_collider"] == "planarized"
    assert record["plane_z_m"] == pytest.approx(FIXTURE_PLANE) and record["read_points_moved"] > 0
    assert collision["digest"] == record["derived_collider"]["sha256"]
    assert Path(collision["path"]).parent.parent.name == source.DIRECTORY
    assert inputs["configuration"]["collision_source_digest"] == collision["digest"]
    result = execute(tmp_path, inputs, envelope, config_path)
    artifacts = {row["role"]: row for row in result["output_artifacts"]}
    assert artifacts["configured_collision_without_source_object"]["digest"] == collision["digest"]
    receipt = json.loads(Path(artifacts["website_background_reuse_receipt"]["path"]).read_text())
    # The scene revision seals the model with the bytes (``geometry.object_excision_result``), portably.
    assert receipt["background_bytes_unchanged"] is True
    assert receipt["support_pad"]["derived_collider"] == {key: value for key, value in
                                                         record["derived_collider"].items() if key != "path"}
    assert {key: value for key, value in receipt["support_pad"].items() if key != "derived_collider"} == {
        key: value for key, value in record.items() if key != "derived_collider"}


def test_the_native_inputs_bind_the_planarized_collider_to_its_normalization(tmp_path, sagging_scan) -> None:
    from tests.test_website_native_inputs import packet

    envelope, configs = packet(tmp_path)
    from blueprint_pipeline.website_native_inputs import validate_website_native_inputs

    validate_website_native_inputs(envelope=envelope, configurations=configs, require_render_inputs=False)
    runtime = json.loads((tmp_path / "native/runtime_inputs.json").read_text())
    normalization = json.loads(Path(runtime["collision"]["normalization_path"]).read_text())
    assert runtime["collision"]["digest"] != normalization["output"]["sha256"]
    assert runtime["collision"]["support_pad"]["derived_collider"]["derived_from_sha256"] == (
        normalization["output"]["sha256"])


def _runtime_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: dict[str, Any],
                   preparation: dict[str, Any], *, modelled: bool) -> tuple[bytes, bytes]:
    from blueprint_pipeline.website_scene_runtime_inputs import prepare_website_runtime_inputs

    with monkeypatch.context() as patch:
        if not modelled:
            # Main before this change: no support is modelled where the collider is produced.
            patch.setattr(source, "seal_support_pad", lambda **_: None)
        value = prepare_website_runtime_inputs(preparation=preparation, base_scene=args["base_scene"],
            source_geometry=args["source_geometry"], task_masks=args["task_masks"], output_root=tmp_path / "native")
    path = tmp_path / "native" / "runtime_inputs.json"
    data = path.read_bytes()
    path.unlink()
    return data, Path(value["collision"]["path"]).read_bytes()


@pytest.mark.parametrize("kind", ["container", "another_surface", "articulated"])
def test_a_scene_with_no_rigid_support_task_is_byte_identical(tmp_path, monkeypatch, kind) -> None:
    import tests.test_website_task_preparation as fixtures
    from blueprint_pipeline import website_task_preparation

    if kind == "container":
        args = fixtures.container_arguments(fixtures._arguments(tmp_path))
    elif kind == "articulated":
        args = fixtures._arguments(tmp_path, fixtures._assembly_inputs(tmp_path))
    else:
        args = fixtures._arguments(tmp_path)
        calls: list[int] = []

        def fitted(mesh, lower, upper, *, up, meters_per_unit, up_sign=1, placement_residual_m=None):
            start = 0 if not calls else 50
            calls.append(start)
            return {"top_runtime_units": float(lower[up]), "aabb_min": list(lower), "aabb_max": list(upper),
                    "face_indices": list(range(start, start + 100)), "basis": "fitted_surface_under_footprint",
                    "physical_measurement": False}
        monkeypatch.setattr(website_task_preparation, "support_under", fitted)
    from blueprint_pipeline.website_task_preparation import compile_website_scene_preparation

    preparation = compile_website_scene_preparation(**args)
    if kind == "another_surface":
        assert preparation["intake_request"]["task"]["destination"]["support"]
    before, collider_before = _runtime_bytes(tmp_path, monkeypatch, args, preparation, modelled=False)
    after, collider_after = _runtime_bytes(tmp_path, monkeypatch, args, preparation, modelled=True)
    assert source.rigid_support_task(preparation) is None
    assert after == before and collider_after == collider_before
    assert b"support_pad" not in after and not (tmp_path / "native" / source.DIRECTORY).exists()


def test_a_flat_tabletop_keeps_its_normalized_collider_and_records_why(tmp_path, monkeypatch) -> None:
    import tests.test_website_task_preparation as fixtures

    from blueprint_pipeline.website_task_preparation import compile_website_scene_preparation

    args = fixtures._arguments(tmp_path)
    preparation = compile_website_scene_preparation(**args)
    before, collider_before = _runtime_bytes(tmp_path, monkeypatch, args, preparation, modelled=False)
    after, collider_after = _runtime_bytes(tmp_path, monkeypatch, args, preparation, modelled=True)
    assert collider_after == collider_before
    runtime, original = json.loads(after), json.loads(before)
    record = runtime["collision"].pop("support_pad")
    runtime.pop("digest"), original.pop("digest")
    assert runtime == original  # nothing else changes but the record (and the digest over it)
    assert record["status"] == "consistent" and record["collider_already_planar"] is True
    assert record["derived_collider"] is None and record["scene_revision_collider"] == "as_normalized"


# --- full-room scale (opt-in): the pad never raises the preparation's peak ------------------------------------

_ROOM_PROGRAM = r"""
import json, sys, time
from pathlib import Path
import numpy as np, trimesh
from tests.test_rigid_support_pad import _scan_heights
glb, spacing = Path(sys.argv[1]), float(sys.argv[2])
def grid(x0, x1, y0, y1, step, z):
    xs, ys = np.arange(x0, x1 + 1e-9, step), np.arange(y0, y1 + 1e-9, step)
    x, y = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([x, y, z(x, y)], axis=-1).reshape(-1, 3)
    index = np.arange(len(xs) * len(ys)).reshape(len(xs), len(ys))
    a, b, c, d = index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]
    return points, np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
parts = [grid(-4.6, 4.6, -4.6, 4.6, spacing, lambda x, y: 0.002 * np.sin(13 * x) * np.cos(11 * y)),
         grid(-1.0, 0.29, -1.05, 0.51, 0.003, _scan_heights)]
points, faces, offset = [], [], 0
for p, f in parts:
    points.append(p); faces.append(f + offset); offset += len(p)
points, faces = np.concatenate(points), np.concatenate(faces)
# A Y-up Atlas GLB, as the website normalizes it.
trimesh.Trimesh(vertices=np.column_stack([points[:, 0], points[:, 2], -points[:, 1]]), faces=faces,
                process=False).export(glb)
print(json.dumps({"triangles": int(len(faces))}))
"""

_PREPARATION_PROGRAM = r"""
import json, sys, time
from pathlib import Path
from blueprint_pipeline.task_evaluation_completed_scene_geometry import normalize_completed_mesh
glb, out, pad = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] == "pad"
def peak():
    line = [row for row in Path("/proc/self/status").read_text().splitlines() if row.startswith("VmHWM")][0]
    return int(line.split()[1]) / 1024
normalized = normalize_completed_mesh(source=glb, original_filename=glb.name,
                                      coordinate_frame={"meters_per_unit": 1.0, "up_axis": "Y"}, output_root=out)
row = {"normalized_peak_mb": peak()}
if pad:
    from tests.test_website_support_pad import _preparation
    from blueprint_pipeline.website_support_pad import seal_support_pad
    started = time.time()
    record = seal_support_pad(preparation=_preparation(), collision=out / normalized["output"]["relative_path"],
                              output_root=out.parent / "native")
    row.update(status=record["status"], points_moved=record.get("read_points_moved"),
               pad_seconds=round(time.time() - started, 1))
row["peak_mb"] = peak()
print(json.dumps(row))
"""


def _room_triangles() -> list[float]:
    return [float(value) for value in os.environ.get("BLUEPRINT_SUPPORT_PAD_MEMORY_TRIANGLES", "").split(",") if value]


# An operator's measurement on a host with GBs of headroom, run on request. The hermetic lanes run
# ``-m "not external_runtime and not external_data"`` and block on any skip, so it is deselected there by
# ``external_runtime`` (as ``tests/test_teleport_live_public.py`` is), never collected and skipped.
@pytest.mark.slow
@pytest.mark.external_runtime
@pytest.mark.skipif(not _room_triangles(), reason="set BLUEPRINT_SUPPORT_PAD_MEMORY_TRIANGLES (millions, e.g. 4,7.2)")
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads VmHWM from /proc")
@pytest.mark.parametrize("millions", _room_triangles())
def test_a_full_room_is_planarized_within_the_peak_its_normalization_already_reaches(tmp_path, millions) -> None:
    """A 9.2 m room at 3-7 M triangles, normalized and then planarized in one process as the website
    preparation runs them: the pad (its reads and its writer a block at a time) adds nothing to the peak."""

    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join([str(root), str(root / "src")])}
    glb = tmp_path / "room.glb"
    spacing = 9.2 / math.sqrt(millions * 1e6 / 2.0)
    made = subprocess.run([sys.executable, "-c", _ROOM_PROGRAM, str(glb), str(spacing)], capture_output=True,
                          text=True, env=environment, cwd=root, check=True)
    triangles = json.loads(made.stdout.strip().splitlines()[-1])["triangles"]

    def run(mode: str) -> dict[str, Any]:
        done = subprocess.run([sys.executable, "-c", _PREPARATION_PROGRAM, str(glb), str(tmp_path / mode / "collision"),
                               mode], capture_output=True, text=True, env=environment, cwd=root, check=True)
        return json.loads(done.stdout.strip().splitlines()[-1])

    alone, padded = run("alone"), run("pad")
    print(json.dumps({"triangles": triangles, "normalization_peak_mb": alone["peak_mb"], "with_pad": padded}))
    assert padded["status"] == "consistent" and padded["points_moved"] > 0
    assert padded["peak_mb"] <= alone["peak_mb"] * 1.05
