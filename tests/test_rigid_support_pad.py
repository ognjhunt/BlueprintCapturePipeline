"""A rigid task's support is made the flat one its sealed success contract describes, and checked on the CPU.

Beta journey step: L5 team policy evaluation. capture-447ae4f4's first GPU probe
(website job, 2026-11-03 22:54Z): the tray, seated on the highest collider point
under its footprint (+3.65 mm over its success plane), settled 2.8 mm lower and
tilted 0.044 rad before anything moved, so the scorer could not read the episode
(``rigid_task_reset_readback_mismatch``); released at the goal it settled at
scoring z 0.80810, 1.2 mm under the sealed window [0.80926, 0.81926]. These
fixtures rebuild that table synthetically (no customer data): its scan 2 mm under
the success plane at the start (the scan's own estimate of the tray's bottom,
0.8019 m) and 6.5 mm under it at the goal, with +/-5 mm of scan noise on top.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from blueprint_pipeline import rigid_support_pad as pad
from blueprint_pipeline.rigid_support_seating import footprint_top_z, triangle_support_surface
from tests.test_rigid_support_seating import COM, _geometry

# capture-447ae4f4's sealed numbers, from its team half's documents and its probe plan.
PLANE = 0.8038877905903447  # the success document's surface target (the green disc) on the dining table
REGISTERED_TOP = 0.8166057543686343  # the registered support's highest face: 12.7 mm over the disc
H0 = 0.010372125543653965  # the tray's scoring frame (its centre of mass) above its qualified collision bottom
HALF = (0.15668499469757083, 0.260563999414444)  # the tray's qualified collision half extents, 0.31 x 0.52 m
TOP_FACE = 0.04492813534678952 - H0
START = np.array([-0.4326211349137302, -0.3030958365897039])
GOAL = np.array([-0.22765085348616906, 0.09984717917168634])
TARGET = [float(GOAL[0]), float(GOAL[1]), PLANE + H0]  # 0.81426: the sealed window's centre
DISC_RADIUS = 0.3900576605858041  # the target disc the whole tray must end inside
SUPPORT = {"top_z_m": REGISTERED_TOP, "bounds_min_xyz_m": [-1.0020434187791705, -1.0564821977505914, 0.7657441024069072],
           "bounds_max_xyz_m": [0.2905184420020341, 0.5146232822199573, REGISTERED_TOP]}
NOISE = 0.005
# Low-frequency scan noise (offset from the start or goal centre, sign, width): a hump beside the tray's
# centre with the scan falling away around it at the start, and two hollows under the goal.
START_FEATURES = ((0.03, 0.05, 1.0, 0.05), (-0.09, -0.16, -1.0, 0.07), (0.10, -0.14, -1.0, 0.07))
GOAL_FEATURES = ((0.05, -0.12, -1.0, 0.08), (-0.06, 0.06, -1.0, 0.06))


def _spec(start_z: float = PLANE + H0, **changes: Any) -> dict[str, Any]:
    """The scored task's fields the support reads, as the surface target binds them (tray at identity)."""

    value = {
        "subject_collision_bounds_scoring_frame_m": {"minimum": [-HALF[0], -HALF[1], -H0],
                                                     "maximum": [HALF[0], HALF[1], TOP_FACE]},
        "start_pose_world": [float(START[0]), float(START[1]), start_z, 0.0, 0.0, 0.0, 1.0],
        "target_position_world_m": list(TARGET),
        "surface_target": {"surface_position_world_m": [float(GOAL[0]), float(GOAL[1]), PLANE], "radius_m": DISC_RADIUS},
        "support_height_interval_m": [TARGET[2] - 0.005, TARGET[2] + 0.005],
        "destination_position_bounds_world_m": {"minimum": [TARGET[0] - 0.05, TARGET[1] - 0.05, TARGET[2] - 0.005],
                                                "maximum": [TARGET[0] + 0.05, TARGET[1] + 0.05, TARGET[2] + 0.005]},
        "reset_translation_tolerance_m": 0.002, "reset_orientation_tolerance_rad": 0.01}
    value.update(changes)
    return value


def _scan_heights(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The scanned table: 2 mm under the plane at the start, 6.5 mm at the goal, and +/-5 mm of noise."""

    run = GOAL - START
    along = np.clip(((x - START[0]) * run[0] + (y - START[1]) * run[1]) / (run @ run), 0.0, 2.0)
    base = PLANE - 0.002 - 0.0045 * along
    noise = np.zeros_like(x)
    for centre, features in ((START, START_FEATURES), (GOAL, GOAL_FEATURES)):
        for dx, dy, sign, width in features:
            noise += sign * 1.3 * NOISE * np.exp(-((x - centre[0] - dx) ** 2 + (y - centre[1] - dy) ** 2)
                                                 / (2.0 * width * width))
    return base + np.clip(noise, -NOISE, NOISE)


def _grid(heights: Any = _scan_heights, *, spacing: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """The table top's points and triangle indices, as a scanned mesh holds them."""

    xs = np.arange(-0.75, 0.2 + 1e-9, spacing)
    ys = np.arange(-0.75, 0.5 + 1e-9, spacing)
    x, y = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([x, y, heights(x, y)], axis=-1).reshape(-1, 3)
    index = np.arange(len(xs) * len(ys)).reshape(len(xs), len(ys))
    a, b, c, d = index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]
    faces = np.concatenate([np.stack([a, b, c], axis=-1).reshape(-1, 3), np.stack([a, c, d], axis=-1).reshape(-1, 3)])
    return points, faces


def _scan() -> np.ndarray:
    points, faces = _grid()
    return points[faces]


def _seated_on(triangles: np.ndarray) -> float:
    """Where ``rigid_support_seating`` seats the tray's scoring frame at the start: on the highest point."""

    top = footprint_top_z(triangles, START - HALF, START + HALF, floor_z=PLANE - 0.03, ceiling_z=PLANE + 0.05)
    assert top is not None
    return top + H0


def test_capture_447s_success_plane_is_its_surface_target_and_its_sealed_window_holds_a_rest_on_it() -> None:
    plane = pad.success_plane(_spec())
    assert plane["basis"] == "success_document_surface_target" and plane["consistent"] is True
    assert plane["plane_z_m"] == pytest.approx(PLANE) and plane["contract_target_rest_plane_z_m"] == pytest.approx(PLANE)
    # On the plane the tray rests at the window's centre; on the registered top it would rest 7.7 mm past it.
    assert plane["rest_scoring_z_m"] == pytest.approx(TARGET[2])
    low, high = plane["success_window_scoring_z_m"]
    assert (low, high) == pytest.approx((0.8092599161339986, 0.8192599161339986))
    assert REGISTERED_TOP + H0 - high == pytest.approx(0.0077, abs=1e-4)
    # The support's own scan moves from as far above the plane as its highest face, and as far below.
    assert pad.scan_band(PLANE, REGISTERED_TOP) == pytest.approx([PLANE - 0.0127180, REGISTERED_TOP], abs=1e-6)
    assert pad.scan_band(PLANE, PLANE) == pytest.approx([PLANE - 0.01, PLANE])  # at least the fitted band
    assert pad.scan_band(PLANE, PLANE + 0.2) == pytest.approx([PLANE - 0.03, PLANE + 0.03])  # never past it


def test_a_window_no_flat_support_can_meet_needs_a_new_success_document() -> None:
    # A window centred on the registered top, not on the disc the document places the tray on.
    shifted = _spec(support_height_interval_m=[REGISTERED_TOP + H0 - 0.005, REGISTERED_TOP + H0 + 0.005])
    plane = pad.success_plane(shifted)
    assert plane["consistent"] is False and plane["reason"] == "rigid_support_success_window_off_success_plane"
    assert plane["requires"] == pad.NEW_REVISION_REQUIRED
    # A surface target its own contract target does not stand on.
    moved = _spec(surface_target={"surface_position_world_m": [float(GOAL[0]), float(GOAL[1]), REGISTERED_TOP]})
    assert pad.success_plane(moved)["reason"] == "rigid_support_success_target_off_its_surface"
    # Without a surface target, the contract target's rest is the plane.
    bare = _spec()
    del bare["surface_target"]
    assert pad.success_plane(bare)["basis"] == "success_contract_target_rest"
    assert pad.success_plane(bare)["plane_z_m"] == pytest.approx(PLANE)


def test_the_cpu_check_refuses_capture_447s_scanned_table_as_its_probe_did() -> None:
    scan = _scan()
    assert (scan[:, :, 2].max() - PLANE, scan[:, :, 2].min() - PLANE) == pytest.approx((0.0027, -0.0122), abs=2e-4)
    check = pad.support_rest_check(scan, _spec(_seated_on(scan)))
    assert check["status"] == "held" and check["basis"] == "cpu_highest_contact_and_three_point_support"
    start = check["start"]
    # Seated on the hump 2.7 mm over the plane, the tray tips off it onto the scan beside it: past 0.01 rad.
    assert start["rest"]["highest_contact_z_m"] - PLANE == pytest.approx(0.0027, abs=2e-4)
    assert start["highest_contact_offset_m"] == pytest.approx(0.0, abs=1e-12)
    assert start["three_point_tilt_rad"] == pytest.approx(0.021, abs=0.002)
    assert start["three_point_tilt_rad"] > start["reset_tolerance"]["orientation_rad"]
    assert start["three_point_offset_m"] == pytest.approx(-0.0005, abs=2e-4)
    # At the goal it rests a millimetre under the sealed window (the probe: 1.2 mm).
    (centre, *_corners) = check["goal"]["placements"]
    assert centre["within_success_window"] is False
    assert check["goal"]["success_window_scoring_z_m"][0] - centre["rest"]["three_point_rest_scoring_z_m"] == (
        pytest.approx(0.001, abs=3e-4))
    assert check["reasons"][0].startswith("rigid_support_start_rest_off_start:0.5mm/0.021rad")
    assert check["reasons"][1].startswith("rigid_support_goal_rest_outside_success_window:0.808")
    assert pad.hold_reason(check) == check["reasons"][0]


def test_the_pad_makes_capture_447s_start_goal_and_window_agree_without_changing_them() -> None:
    scan = _scan()
    spec = _spec()
    before = {key: (dict(value) if isinstance(value, dict) else value) for key, value in spec.items()}
    planned = pad.plan_support_pad(spec, support_plane=SUPPORT, triangles=scan)
    record = planned["record"]
    assert record["model"] == pad.MODEL and record["plane_z_m"] == pytest.approx(PLANE)
    assert record["sealed_revision_changed"] is False and record["success_document_changed"] is False
    assert record["scan_band_z_m"] == pytest.approx([PLANE - (REGISTERED_TOP - PLANE), REGISTERED_TOP])
    # The region holds the start footprint, the target disc (every placement the scorer accepts), and between,
    # and no more of the oval table's bounding box: a square around the disc reaches past its outline.
    region = np.asarray(record["region_world_xy_m"])
    corners = [START + [sx * (HALF[0] + 0.02), sy * (HALF[1] + 0.02)] for sx in (-1, 1) for sy in (-1, 1)]
    rim = [GOAL + DISC_RADIUS * np.array([math.cos(a), math.sin(a)]) for a in np.linspace(0, 2 * math.pi, 73)]
    assert pad.inside_polygon(region, corners + rim).all()
    assert pad.inside_polygon(region, [START + (GOAL - START) * t for t in np.linspace(0, 1, 11)]).all()
    reach = DISC_RADIUS * 1.02
    assert not pad.inside_polygon(region, [GOAL + [reach, reach], GOAL + [reach, -reach]]).any()
    assert pad.inside_polygon(planned["selection"], region).all()
    low, high = record["region_bounds_world_xy_m"]["minimum"], record["region_bounds_world_xy_m"]["maximum"]
    assert low[0] >= SUPPORT["bounds_min_xyz_m"][0] and high[1] <= SUPPORT["bounds_max_xyz_m"][1]
    # The scan's deviation under it, recorded: the hump's 2.7 mm over the plane and the hollows' 12 mm under.
    deviation = record["measured_scan_deviation"]
    assert deviation["start"]["highest_above_plane_m"] == pytest.approx(0.0027, abs=2e-4)
    assert deviation["region"]["lowest_below_plane_m"] == pytest.approx(-0.0122, abs=3e-4)
    assert record["read_points_moved"] > 0 and record["read_points_maximum_shift_m"] <= 0.0127
    # Seated on the planarized triangles the tray starts on the plane, its rest is its start, and a
    # correct placement rests at the window's centre anywhere in the goal region.
    planar = planned["planarized_triangles"]
    assert _seated_on(planar) == pytest.approx(PLANE + H0)
    check = pad.support_rest_check(planar, _spec(_seated_on(planar)))
    assert check["status"] == "consistent" and check["reasons"] == []
    assert check["start"]["three_point_offset_m"] == pytest.approx(0.0, abs=1e-12)
    assert check["start"]["three_point_tilt_rad"] == pytest.approx(0.0, abs=1e-12)
    for placement in check["goal"]["placements"]:
        assert placement["within_success_window"] is True
        assert placement["rest"]["three_point_rest_scoring_z_m"] == pytest.approx(TARGET[2])
    # The task's own fields are read, never changed.
    assert spec == before


def test_the_native_adapter_seats_capture_447s_tray_on_the_plane_through_the_pad() -> None:
    planar = pad.plan_support_pad(_spec(), support_plane=SUPPORT, triangles=_scan())["planarized_triangles"]
    geometry = _geometry(support_surface=triangle_support_surface(planar))
    assert geometry["root_position_world_m"][2] == pytest.approx(PLANE)
    assert geometry["start_scoring_position_world_m"][2] == pytest.approx(PLANE + COM[2])
    # On the scan itself, the same seating stands on the hump.
    seated = _geometry(support_surface=triangle_support_surface(_scan()))
    assert seated["root_position_world_m"][2] - PLANE == pytest.approx(0.0027, abs=2e-4)


def test_something_standing_on_the_support_keeps_its_shape_and_the_room_outside_the_region_is_untouched() -> None:
    def table_with_a_cup(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        heights = _scan_heights(x, y)
        cup = (np.abs(x - (-0.33)) <= 0.03) & (np.abs(y - (-0.08)) <= 0.03)  # between start and goal, 3 cm tall
        return np.where(cup, PLANE + 0.03, heights)

    points, faces = _grid(table_with_a_cup)
    planned = pad.plan_support_pad(_spec(), support_plane=SUPPORT, triangles=points[faces])
    record = planned["record"]
    moved, chosen = pad.planarized_points(points, selection=planned["selection"], band=record["scan_band_z_m"],
                                          plane_z=PLANE)
    assert np.allclose(moved[chosen, 2], PLANE)
    cup = np.isclose(points[:, 2], PLANE + 0.03)
    assert cup.any() and not chosen[cup].any()  # above the support's own highest face: not the support
    outside = ~pad.inside_polygon(planned["selection"], points[:, :2])
    assert outside.any() and not chosen[outside].any()
    assert np.array_equal(moved[~chosen], points[~chosen])


@pytest.mark.parametrize(("surface", "status", "tilt"), [
    ("flat", "rests", 0.0), ("ramp", "rests", math.atan(0.02)), ("half", "tips_off_footprint_points", None)])
def test_a_box_rests_flat_tilts_onto_its_support_or_tips(surface: str, status: str, tilt: float | None) -> None:
    def heights(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if surface == "ramp":
            return PLANE + 0.02 * (x - START[0])
        return np.full_like(x, PLANE)

    points, faces = _grid(heights)
    triangles = points[faces]
    if surface == "half":  # only the scan on one side of the centre: nothing holds it up over its centre
        triangles = triangles[triangles[:, :, 0].min(axis=1) > START[0] + 0.02]
    rest = pad.box_rest(triangles, centre_xy=START, bounds=_spec()["subject_collision_bounds_scoring_frame_m"],
                        floor_z=PLANE - 0.03, ceiling_z=PLANE + 0.05)
    assert rest["status"] == status
    if tilt is not None:
        assert rest["three_point_tilt_rad"] == pytest.approx(tilt, abs=1e-9)
        assert rest["three_point_rest_scoring_z_m"] == pytest.approx(PLANE + H0 * math.sqrt(1 + math.tan(tilt) ** 2))


def _mesh_usd(path: Path, points: np.ndarray, faces: np.ndarray, *, translate_z: float = 0.0) -> None:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Root/mesh_0000_GLTF")
    UsdGeom.XformCommonAPI(mesh).SetTranslate(Gf.Vec3d(0.0, 0.0, translate_z))
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy((points - [0.0, 0.0, translate_z]).astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    stage.GetRootLayer().Save()


def test_the_derived_collider_moves_only_point_heights_and_the_physics_reads_the_pad(tmp_path) -> None:
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    from blueprint_pipeline.scene_physics_preflight import load_background_triangles
    from blueprint_pipeline.usd_binary_read_cache import read_scope

    points, faces = _grid()
    sealed = tmp_path / "collision.usda"
    _mesh_usd(sealed, points, faces, translate_z=0.25)  # a mesh under its own transform, as a room's may be
    before = hashlib.sha256(sealed.read_bytes()).hexdigest()
    scan = load_background_triangles(sealed, centre=[*START.tolist(), PLANE], radius=2.0)
    planned = pad.plan_support_pad(_spec(), support_plane=SUPPORT, triangles=scan)
    derived = tmp_path / "support-pad" / "collision.usdc"
    with read_scope(tmp_path / "cache"):
        written = pad.write_planarized_collider(sealed, derived, selection=planned["selection"],
                                                band=planned["record"]["scan_band_z_m"], plane_z=PLANE)
    assert hashlib.sha256(sealed.read_bytes()).hexdigest() == before  # the sealed collider is never edited
    assert written["points_moved"] == planned["record"]["read_points_moved"]
    assert [row["prim_path"] for row in written["meshes"]] == ["/Root/mesh_0000_GLTF"]
    stage = Usd.Stage.Open(str(derived))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Root/mesh_0000_GLTF"))
    assert list(mesh.GetFaceVertexIndicesAttr().Get()) == faces.reshape(-1).tolist()  # topology kept
    local = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    assert np.allclose(local[:, :2], points[:, :2], atol=1e-6)  # only heights move
    physics = load_background_triangles(derived, centre=[*START.tolist(), PLANE], radius=2.0)
    assert _seated_on(physics) == pytest.approx(PLANE + H0, abs=1e-6)
    check = pad.support_rest_check(physics, _spec(_seated_on(physics)))
    assert check["status"] == "consistent", check["reasons"]
    held = pad.support_rest_check(scan, _spec(_seated_on(scan)))
    assert held["status"] == "held"


def test_a_compiled_half_is_held_by_its_support_record(tmp_path) -> None:
    import json

    packet = tmp_path / "native-task-packet"
    packet.mkdir()
    (packet / "native_task_arena_scene_plan.v1.json").write_text(json.dumps({"task_spec": {"task_kind": "rigid_pick_place"}}))
    half = tmp_path / "team_scene_inputs.v1.json"

    def record(**fields: Any) -> str:
        half.write_text(json.dumps({"base_native_packet": {"path": str(packet / "receipt.json")}, **fields}))
        return str(half)

    reason = "rigid_support_start_rest_off_start:0.5mm/0.021rad"
    assert pad.half_support_hold(record(task_support_pad={"status": "held", "reasons": [reason]})) == reason
    assert pad.half_support_hold(record(task_support_pad={"status": "consistent", "reasons": []})) is None
    assert pad.half_support_hold(record(task_support_pad={"status": "unchecked"})) is None
    # Compiled before the support was modelled (capture-447ae4f4's half): held by name, so a release recompiles it.
    assert pad.half_support_hold(record()) == pad.UNMODELLED_HALF
    (packet / "native_task_arena_scene_plan.v1.json").write_text(
        json.dumps({"task_spec": {"task_kind": "articulated_open_close"}}))
    assert pad.half_support_hold(record()) is None
    assert pad.half_support_hold(None) is None and pad.half_support_hold(tmp_path / "missing.json") is None


# --- end to end: the team packet stages the planarized support and checks it --------------------------

def _with_qualified_bounds(monkeypatch: pytest.MonkeyPatch) -> Any:
    """The fixture revision with its replacement's qualified collision bounds, as capture-447's carries them."""

    import copy

    import tests.test_team_scene_packet as packet_tests
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from tests.test_task_evaluation_rigid_relocation_native_adapter import STATIC, _rewrite

    original_case = packet_tests.rigid_case

    def case(path: Path, **kwargs: Any) -> tuple[dict, dict, dict, dict]:
        configured, references, docs, spec = original_case(path, **kwargs)
        static = copy.deepcopy(docs[STATIC])
        static["observed_structure"]["collision_bounds_body_frame_m"] = {
            "minimum": [-0.0638749, -0.0658936, 0.0], "maximum": [0.0638749, 0.0658936, 0.127638]}
        static["result_digest"] = canonical_digest({**static, "result_digest": ""}, digest_field="result_digest")
        _rewrite(tmp_path=path, configured=configured, references=references, contract_path=STATIC, document=static)
        return configured, references, {**docs, STATIC: static}, spec

    monkeypatch.setattr(packet_tests, "rigid_case", case)
    return packet_tests


def _scanned_bundle(tmp_path: Path, collision: Path | None = None) -> Path:
    """The fixture room with its support's top scanned: 4 mm under its registered top, a hump to that top
    beside the start's centre, and hollows 8 mm under it at the start and the goal. ``collision``, when
    given, is the room's sealed collider instead (one already planarized, named by its format)."""

    import json
    import zipfile

    from pxr import Gf, UsdGeom, UsdPhysics

    from tests.test_rigid_support_seating import _sealed
    from tests.test_team_scene_packet import SUPPORT_TOP_Z, _box, _sha, _stage

    start, goal = (2.9742285, -6.7605156), (3.0942285, -6.7605156)
    xs, ys = np.arange(2.7, 3.45 + 1e-9, 0.01), np.arange(-7.05, -6.45 + 1e-9, 0.01)
    x, y = np.meshgrid(xs, ys, indexing="ij")
    hump = np.exp(-((x - start[0] - 0.015) ** 2 + (y - start[1] - 0.015) ** 2) / (2 * 0.012 ** 2))
    hollows = sum(np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 0.03 ** 2))
                  for cx, cy in ((start[0] - 0.04, start[1] - 0.04), (goal[0] + 0.03, goal[1] + 0.03)))
    heights = SUPPORT_TOP_Z - 0.004 + 0.004 * hump - 0.004 * np.clip(hollows, 0.0, 1.0)
    room = _stage(tmp_path / "room.usda", "/Root")
    _box(room, "/Root/Floor", (-2.0, -14.0, 0.0), (9.0, 4.0, 0.0))
    _box(room, "/Root/Support", (2.5, -9.5, 0.0), (4.5, -1.4, SUPPORT_TOP_Z - 0.012))
    scan = UsdGeom.Mesh.Define(room, "/Root/Scan")
    points = np.stack([x, y, heights], axis=-1).reshape(-1, 3)
    index = np.arange(points.shape[0]).reshape(x.shape)
    a, b, c, d = index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]
    faces = np.concatenate([np.stack([a, b, c], axis=-1).reshape(-1, 3), np.stack([a, c, d], axis=-1).reshape(-1, 3)])
    scan.CreatePointsAttr([Gf.Vec3f(*row) for row in points.tolist()])
    scan.CreateFaceVertexCountsAttr([3] * len(faces))
    scan.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
    UsdPhysics.CollisionAPI.Apply(scan.GetPrim())
    room.GetRootLayer().Save()
    subject = _stage(tmp_path / "replacement.usda", "/Asset")
    UsdPhysics.RigidBodyAPI.Apply(subject.GetDefaultPrim())
    UsdPhysics.MassAPI.Apply(subject.GetDefaultPrim()).CreateMassAttr(0.3)
    _box(subject, "/Asset/body", (-0.0638749, -0.0658936, 0.0), (0.0638749, 0.0658936, 0.127638))
    subject.GetRootLayer().Save()
    payloads = {"appearance.usda": (tmp_path / "room.usda").read_bytes(),
                "collision.usda": (tmp_path / "room.usda").read_bytes(),
                "replacement.usda": (tmp_path / "replacement.usda").read_bytes()}
    if collision is not None:
        del payloads["collision.usda"]
        payloads["collision" + collision.suffix] = collision.read_bytes()
    rows = [{"role": name.split(".")[0], "relative_path": name, "digest": _sha(data), "size_bytes": len(data)}
            for name, data in payloads.items()]
    manifest = _sealed({"schema_version": "task_evaluation_configured_scene_bundle_candidate.v1",
                        "status": "assembled_pending_control_plane_publication", "robot_neutral": True,
                        "robot_specific_base_registration_included": False, "assets": rows}, "manifest_digest")
    path = tmp_path / "configured-scene.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
        archive.writestr("configured_scene_bundle_candidate.v1.json", json.dumps(manifest))
    return path


def test_a_team_half_runs_on_its_support_planarized_and_seats_its_object_on_the_success_plane(
        tmp_path, monkeypatch) -> None:
    pytest.importorskip("pxr")
    import json

    from blueprint_pipeline.rigid_support_seating import start_spawn_disagreement_m
    from tests.test_team_scene_packet import SUPPORT_TOP_Z

    packet_tests = _with_qualified_bounds(monkeypatch)
    flat, _flat_request = packet_tests._compile(tmp_path / "flat", packet_tests._scene(_mkdir(tmp_path / "flat")))
    monkeypatch.setattr(packet_tests, "_bundle", _scanned_bundle)
    scene = packet_tests._scene(_mkdir(tmp_path / "scanned"))
    compiled, request = packet_tests._compile(tmp_path / "scanned", scene)
    half = json.loads(Path(compiled["path"]).read_text())
    support = half["task_support_pad"]
    assert support["model"] == pad.MODEL and support["status"] == "consistent" and support["hold_reason"] is None
    assert support["plane_z_m"] == pytest.approx(SUPPORT_TOP_Z)
    assert support["plane"]["basis"] == "success_contract_target_rest" and support["plane"]["consistent"] is True
    deviation = support["measured_scan_deviation"]
    assert deviation["start"]["highest_above_plane_m"] == pytest.approx(0.0, abs=1e-3)  # the hump, under a grid step
    assert deviation["start"]["lowest_below_plane_m"] == pytest.approx(-0.008, abs=5e-4)
    # The scan as captured: seated on its hump, the object tips onto the hollow beside it.
    scan = support["scan_rest_check"]
    assert scan["status"] == "held" and scan["reasons"][0].startswith("rigid_support_start_rest_off_start:")
    # The physics' support: the derived collider, its sealed source untouched, read back where the object rests.
    derived = support["derived_collider"]
    root = Path(compiled["path"]).parent
    sealed = root / "configured-scene" / "collision.usda"
    assert derived["relative_path"] == "support-pad/collision.usdc" and derived["rest_check_read_back"] is True
    assert derived["derived_from_sha256"] == packet_tests._sha(sealed.read_bytes())
    assert derived["edit"] == "point_heights_only" and derived["points_moved"] > 0
    assert [row["prim_path"] for row in derived["meshes"]] == ["/Root/Scan"]
    (collision,) = [row for row in request["assets"] if row["semantic_role"] == "scene_collision"]
    assert collision["filename"] == "collision.usdc" and collision["source"]["sha256"] == derived["sha256"]
    plan = json.loads((root / "native-task-packet" / "native_task_arena_scene_plan.v1.json").read_text())
    (staged,) = [row for row in plan["objects"] if row["semantic_role"] == "scene_collision"]
    assert staged["sha256"] == derived["sha256"]
    # The object starts at rest on the plane, where it spawns; the goal rests at the window's centre.
    spec = request["task_spec"]
    offset = spec["interaction_affordance"]["asset_root_from_scoring_frame"]["position_m"][2]
    assert spec["start_pose_world"][2] == pytest.approx(SUPPORT_TOP_Z + offset, abs=1e-6)
    assert start_spawn_disagreement_m(plan) == pytest.approx(0.0, abs=1e-9)
    check = support["rest_check"]
    assert check["start"]["three_point_tilt_rad"] == pytest.approx(0.0, abs=1e-9)
    assert check["goal"]["placements"][0]["rest"]["three_point_rest_scoring_z_m"] == pytest.approx(
        spec["target_position_world_m"][2], abs=1e-6)
    # Nothing sealed moved: the half binds the revision it was compiled from, and its success contract is
    # the one the same task compiles to on a flat room.
    assert half["task_success_contract_digest"] == json.loads(Path(flat["path"]).read_text())[
        "task_success_contract_digest"]
    assert support["sealed_revision_changed"] is False
    assert half["scene_revision_digest"] == scene["revision"]["revision_digest"]


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True)
    return path


def test_a_team_half_on_a_sealed_collider_already_planarized_is_a_no_op(tmp_path, monkeypatch) -> None:
    """A scene whose website preparation sealed its support flat (``website_support_pad``): the half reads
    nothing to move, derives no collider, and runs and is checked on the sealed bytes themselves."""

    pytest.importorskip("pxr")
    import json

    from blueprint_pipeline.rigid_support_seating import start_spawn_disagreement_m

    packet_tests = _with_qualified_bounds(monkeypatch)
    monkeypatch.setattr(packet_tests, "_bundle", _scanned_bundle)
    scanned, scanned_request = packet_tests._compile(tmp_path / "scanned",
                                                     packet_tests._scene(_mkdir(tmp_path / "scanned")))
    scanned_half = json.loads(Path(scanned["path"]).read_text())
    planarized = Path(scanned["path"]).parent / scanned_half["task_support_pad"]["derived_collider"]["relative_path"]
    sealed = tmp_path / "sealed" / "collision.usdc"
    sealed.parent.mkdir()
    sealed.write_bytes(planarized.read_bytes())
    monkeypatch.setattr(packet_tests, "_bundle", lambda path: _scanned_bundle(path, collision=sealed))
    scene = packet_tests._scene(_mkdir(tmp_path / "planar"))
    compiled, request = packet_tests._compile(tmp_path / "planar", scene)
    half = json.loads(Path(compiled["path"]).read_text())
    support = half["task_support_pad"]
    assert support["status"] == "consistent" and support["hold_reason"] is None
    assert support["read_points_moved"] == 0 and support["read_points_maximum_shift_m"] < 1e-6
    assert support["derived_collider"] is None and support["collider_already_planar"] is True
    assert not (Path(compiled["path"]).parent / "support-pad").exists()
    # The physics' collider is the sealed one, byte for byte.
    sealed_sha = packet_tests._sha(sealed.read_bytes())
    (collision,) = [row for row in request["assets"] if row["semantic_role"] == "scene_collision"]
    assert collision["source"]["sha256"] == sealed_sha
    plan = json.loads((Path(compiled["path"]).parent / "native-task-packet"
                       / "native_task_arena_scene_plan.v1.json").read_text())
    (staged,) = [row for row in plan["objects"] if row["semantic_role"] == "scene_collision"]
    assert staged["sha256"] == sealed_sha
    # The same task as the half that planarized the scan itself: the object starts at rest on the plane.
    assert request["task_spec"]["start_pose_world"] == scanned_request["task_spec"]["start_pose_world"]
    assert start_spawn_disagreement_m(plan) == pytest.approx(0.0, abs=1e-9)
    assert support["rest_check"] == scanned_half["task_support_pad"]["rest_check"]
    assert support["scan_rest_check"]["status"] == "consistent"  # the sealed collider is what the half reads
    assert half["task_success_contract_digest"] == scanned_half["task_success_contract_digest"]
