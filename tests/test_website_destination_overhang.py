"""An object whose footprint is larger than the supported part of its destination is set down with its centre over it.

Hermetic: a shelf 0.6 m long, made of 1 cm triangles at 1 m, over a floor, in an upright or a turned-over (-Y up)
frame and at 1 or 2 metres per unit; and the ray-cast room of ``test_website_unscanned_surface`` through compile.
The lookups are compile's own ``support_under``; no provider is reached and nothing is paid for.
"""
# Covers: src/blueprint_pipeline/website_destination_overhang.py
# Covers: src/blueprint_pipeline/website_task_preparation.py
# Covers: src/blueprint_pipeline/job_packet_export.py
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from blueprint_pipeline import website_destination_overhang as overhang
from blueprint_pipeline import website_task_preparation as preparation
from blueprint_pipeline.atlas_scene_estimates import DESTINATION_OVERHANG
from blueprint_pipeline.website_support_geometry import (RESTING_OUTLINE_GRID_STEPS, RESTING_OUTLINE_MIN_SHARE,
                                                       destination_clear_of_start, support_under)
from blueprint_pipeline.website_surface_target_admission import website_target_radius, website_target_reach
from tests.test_website_unscanned_surface import (FLIP, RUNTIME_SCALE, RUNTIME_TRANSLATION, _arguments,
                                                  exact_registration, no_model)

__all__ = ["exact_registration", "no_model"]  # the room's own registration, and no model call, for every test here
TOP, HEIGHT = 1.0, 0.45
FRAMES = [(1.0, 1), (2.0, 1), (1.0, -1)]  # (metres per unit, up sign)


def _grid(x0, x1, z0, z1, y, cell=0.01):
    xs = np.linspace(x0, x1, int(round((x1 - x0) / cell)) + 1)
    zs = np.linspace(z0, z1, int(round((z1 - z0) / cell)) + 1)
    points = np.array([[x, y, z] for x in xs for z in zs])
    faces = [[i * len(zs) + j, i * len(zs) + j + 1, (i + 1) * len(zs) + j] for i in range(len(xs) - 1)
             for j in range(len(zs) - 1)]
    faces += [[i * len(zs) + j + 1, (i + 1) * len(zs) + j + 1, (i + 1) * len(zs) + j] for i in range(len(xs) - 1)
              for j in range(len(zs) - 1)]
    return points, np.array(faces)


def _boxes(*boxes):
    """Closed boxes ((low), (high)) in metres: each one's vertices and its own faces (``_mesh`` offsets them)."""
    vertices, faces = [], []
    for low, high in boxes:
        box = trimesh.creation.box(extents=np.subtract(high, low),
                                   transform=trimesh.transformations.translation_matrix(np.add(low, high) / 2))
        faces.append(np.asarray(box.faces))
        vertices.append(np.asarray(box.vertices))
    return vertices, faces


def _mesh(depth: float, *, mpu: float = 1.0, up_sign: int = 1, wall: bool = False, lid: bool = False,
          length: float = 0.6, cell: float = 0.01, boxes=()):
    """A level shelf ``depth`` deep (z 0 to depth) and ``length`` long at ``TOP`` m over a floor, in metres turned
    into runtime units; ``wall``: a wall at z = 0 behind it; ``lid``: a second shelf 10 cm above it; ``boxes``:
    things standing on it."""
    parts = [_grid(0.0, length, 0.0, depth, TOP, cell),
             (np.array([[-2.0, 0, -2], [4, 0, -2], [4, 0, 4], [-2, 0, 4]]), np.array([[0, 2, 1], [0, 3, 2]]))]
    if wall:
        parts.append((np.array([[-1.0, 0, 0], [1.6, 0, 0], [1.6, 2.5, 0], [-1, 2.5, 0]]), np.array([[0, 1, 2], [0, 2, 3]])))
    if lid:
        parts.append(_grid(0.0, length, 0.0, depth, TOP + 0.1, cell))
    parts += list(zip(*_boxes(*boxes))) if boxes else []
    vertices, faces, offset = [], [], 0
    for points, triangles in parts:
        vertices.append(points)
        faces.append(np.asarray(triangles) + offset)
        offset += len(points)
    vertices, faces = np.concatenate(vertices) / mpu, np.concatenate(faces)
    if up_sign == -1:  # turned over about x: a proper rotation, up is -Y
        vertices = vertices @ FLIP.T
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _footprint(x: float, z: float, size_x: float, size_z: float, *, mpu: float = 1.0, up_sign: int = 1,
               height: float = HEIGHT):
    """The footprint at (x, z) on the shelf, its bottom on its top, as compile's ``footprint_at`` gives it."""
    lower = np.array([x - size_x / 2, TOP, z - size_z / 2]) / mpu
    upper = np.array([x + size_x / 2, TOP + height, z + size_z / 2]) / mpu
    if up_sign == -1:
        lower, upper = lower @ FLIP.T, upper @ FLIP.T
        lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
    return lower, upper


#: The object starts on another surface far away, unless a test says otherwise.
ELSEWHERE = (np.array([-1.0, 0.0, -1.0]), np.array([-0.9, 0.1, -0.9]))


def _place(mesh, lower, upper, *, mpu=1.0, up_sign=1, start=ELSEWHERE, start_support=None, clearance=0.0):
    return overhang.centre_supported(mesh, lower, upper, list((lower + upper) / 2), up=1, meters_per_unit=mpu,
                                     up_sign=up_sign, radius=website_target_radius(meters_per_unit=mpu),
                                     runtime_to_sim=np.eye(4), start=start, start_support=start_support,
                                     clearance=clearance)


def _shelf_share(centre_x: float, centre_z: float, size_x: float, size_z: float, depth: float) -> float:
    """The share of the footprint over the shelf's rectangle, read from the geometry itself, not the module."""
    over_x = max(0.0, min(centre_x + size_x / 2, 0.6) - max(centre_x - size_x / 2, 0.0))
    over_z = max(0.0, min(centre_z + size_z / 2, depth) - max(centre_z - size_z / 2, 0.0))
    return over_x * over_z / (size_x * size_z)


def _metres(position, *, mpu, up_sign):
    """A runtime position back in the shelf's metres (x, z)."""
    point = np.asarray(position, dtype=float) * mpu
    point = point @ FLIP if up_sign == -1 else point
    return point[0], point[2]


@pytest.mark.parametrize("mpu, up_sign", FRAMES)
def test_a_footprint_a_little_deeper_than_the_shelf_rests_with_its_centre_over_it_as_placed(mpu, up_sign):
    # The lamp's case: 0.21 x 0.18 m on a shelf 0.16 m deep, judged in the pose it is placed in.
    mesh = _mesh(0.16, mpu=mpu, up_sign=up_sign)
    lower, upper = _footprint(0.3, 0.08, 0.21, 0.18, mpu=mpu, up_sign=up_sign)
    assert support_under(mesh, lower, upper, up=1, meters_per_unit=mpu, up_sign=up_sign) is None
    surface, placement, position = _place(mesh, lower, upper, mpu=mpu, up_sign=up_sign)
    record = placement.record
    assert surface is not None and surface["top_runtime_units"] * mpu * up_sign == pytest.approx(TOP)
    assert record["flag"] == DESTINATION_OVERHANG and record["orientation"] == "as_placed"
    assert record["supported_share"] == pytest.approx(0.16 / 0.18, abs=0.03)
    assert record["largest_overhang_m"] == pytest.approx(0.01, abs=0.002) and record["slid_m"] == 0.0
    assert record["footprint_m"] == pytest.approx([0.18, 0.21]) and record["lattice_points"] == 0
    # The pose actually placed holds the required share of the real shelf.
    x, z = _metres(position, mpu=mpu, up_sign=up_sign)
    assert _shelf_share(x, z, 0.21, 0.18, 0.16) >= RESTING_OUTLINE_MIN_SHARE


def test_a_footprint_across_the_shelf_under_half_supported_as_placed_stays_blocked():
    # Review repro: 0.10 x 0.30 m across a 0.14 m shelf is 47% over it as placed (turned along it, all of it would
    # be): it is never judged in a pose it is not placed in.
    mesh = _mesh(0.14)
    lower, upper = _footprint(0.3, 0.07, 0.10, 0.30)
    assert _shelf_share(0.3, 0.07, 0.10, 0.30, 0.14) < RESTING_OUTLINE_MIN_SHARE
    assert _place(mesh, lower, upper)[:2] == (None, None)


def test_a_centre_off_the_edge_stays_blocked():
    mesh = _mesh(0.16)
    lower, upper = _footprint(0.3, 0.20, 0.21, 0.18)  # its centre 4 cm past the shelf's front edge
    assert _place(mesh, lower, upper)[:2] == (None, None)


@pytest.mark.parametrize("mpu, up_sign", FRAMES)
def test_a_box_into_the_wall_slides_along_the_shelf_until_it_clears(mpu, up_sign):
    mesh = _mesh(0.16, mpu=mpu, up_sign=up_sign, wall=True)
    lower, upper = _footprint(0.3, 0.08, 0.21, 0.18, mpu=mpu, up_sign=up_sign)  # 1 cm of it inside the wall
    surface, placement, position = _place(mesh, lower, upper, mpu=mpu, up_sign=up_sign)
    record = placement.record
    assert surface is not None and record["slid_m"] >= 0.01 - 1e-9
    x, z = _metres(position, mpu=mpu, up_sign=up_sign)
    assert z - 0.09 >= -1e-9  # its back face clears the wall at z = 0
    assert _shelf_share(x, z, 0.21, 0.18, 0.16) >= RESTING_OUTLINE_MIN_SHARE
    assert record["largest_overhang_m"] == pytest.approx(z + 0.09 - 0.16, abs=0.003)  # all of it at the front


def test_no_room_above_the_shelf_stays_blocked():
    mesh = _mesh(0.16, lid=True)  # a shelf 10 cm above: a 45 cm box meets it everywhere
    lower, upper = _footprint(0.3, 0.08, 0.21, 0.18)
    assert _place(mesh, lower, upper)[:2] == (None, None)


def test_low_clutter_moves_a_short_object_and_the_box_stops_at_its_own_height():
    # A 1 cm book where a 3 cm tall object would go: found from 5 mm above the shelf, never one window up.
    book = ((0.27, TOP, 0.05), (0.33, TOP + 0.01, 0.11))
    mesh = _mesh(0.16, boxes=[book])
    lower, upper = _footprint(0.3, 0.08, 0.21, 0.18, height=0.03)
    _, placement, position = _place(mesh, lower, upper)
    x, z = _metres(position, mpu=1.0, up_sign=1)
    assert placement.record["slid_m"] > 0 and (x - 0.105 >= 0.33 - 1e-9 or x + 0.105 <= 0.27 + 1e-9)
    # A lid 10 cm up is no obstacle for a 3 cm object: its box ends at its own height.
    lidded = _mesh(0.16, lid=True)
    assert _place(lidded, lower, upper)[1] is not None


def test_the_search_is_bounded_on_a_large_surface_and_skipped_where_the_grounded_centre_holds(monkeypatch):
    # Review: a 3 m plate with a 5 cm object sampled the whole plate at 1/32 of the object (2.9 GB).
    block = ((1.505, TOP, 1.505), (1.515, TOP + 0.02, 1.515))  # in its box, between the lookup's probes
    plate = _mesh(3.0, length=3.0, cell=0.05, boxes=[block])
    lower, upper = _footprint(1.5, 1.5, 0.05, 0.05)
    _, placement, position = _place(plate, lower, upper)
    side = 3 * (RESTING_OUTLINE_GRID_STEPS - 1) + 1
    assert placement.spot.lattice_points == side * side  # one footprint each way, whatever the plate's size
    assert np.hypot(position[0] - 1.5, position[2] - 1.5) <= 0.05 + 1e-9
    clear = _mesh(3.0, length=3.0, cell=0.05)
    monkeypatch.setattr(overhang._Spot, "nearest", lambda *_: pytest.fail("the grounded centre held"))
    _, placement, position = _place(clear, lower, upper)
    assert placement.spot.lattice_points == 0 and position[0] == pytest.approx(1.5)
    assert placement.record is None  # nothing hangs over: no flag


def test_a_slide_on_the_start_s_own_surface_stays_clear_of_the_start():
    block = ((0.29, TOP, 0.07), (0.31, TOP + 0.02, 0.09))
    plate = _mesh(0.6, length=0.6, boxes=[block])
    lower, upper = _footprint(0.3, 0.08, 0.05, 0.05)
    start = _footprint(0.36, 0.08, 0.05, 0.05)  # its start right beside the grounded centre
    start_support = support_under(plate, *_footprint(0.45, 0.4, 0.05, 0.05), up=1, meters_per_unit=1.0)
    clearance = website_target_reach(*start, relative_error=0.0, up=1, meters_per_unit=1.0)
    _, placement, position = _place(plate, lower, upper, start=start, start_support=start_support,
                                    clearance=clearance)
    assert destination_clear_of_start(*start, position, up=1, clearance=clearance)


def _placed_on_the_shelf():
    mesh = _mesh(0.16)
    lower, upper = _footprint(0.3, 0.10, 0.21, 0.14)  # 1 cm over the front edge, its disc on the shelf
    surface, placement, position = _place(mesh, lower, upper)
    assert placement.record["largest_overhang_m"] == pytest.approx(0.01, abs=0.002)
    return placement, position


def test_named_areas_and_offered_places_are_only_ones_that_hold_it_with_their_own_disc():
    placement, position = _placed_on_the_shelf()
    admits = placement.admits(lambda sealed: True)
    assert admits({"position_world_m": list(position)})
    assert not admits({"position_world_m": [position[0], position[1], 0.15]})  # its disc off the front edge
    assert not admits({"position_world_m": list(position), "radius_m": 0.09})  # a named area wider than the shelf


def test_where_it_finally_lands_is_read_again_and_a_place_that_cannot_hold_it_never_stops_the_job():
    placement, position = _placed_on_the_shelf()
    destination, choice = placement.settle({"position_world_m": list(position)}, {"basis": "grounded_centre"})
    assert destination["position_world_m"] == list(position) and placement.record["moved_after_placement_m"] == 0
    inside = [position[0], position[1], 0.08]  # all of it over the shelf there: no flag, no record
    destination, _ = placement.settle({"position_world_m": inside}, {"basis": "site_choice"})
    assert placement.record is None and destination["position_world_m"] == inside
    # The site's stored choice, where its disc is off the edge: the nearest spot that holds it, recorded.
    off = {"position_world_m": [0.45, position[1], 0.15], "radius_m": 0.05}
    destination, choice = placement.settle(off, {"basis": "site_choice", "position_world_m": off["position_world_m"]})
    record = placement.record
    assert record is None or (record["chosen_place_held"] is False and record["moved_to_hold_the_object_m"] > 0)
    held = destination["position_world_m"]
    assert held != off["position_world_m"] and choice["position_world_m"] == held and "radius_m" not in destination
    assert placement.admits(lambda sealed: True)({"position_world_m": held})


def test_the_packet_prints_the_measured_overhang_and_nothing_at_none():
    from blueprint_pipeline import job_packet_export as packet
    preparation_record = {"subject": {"aabb_min_xyz": [0.0, 0.0, 0.75], "aabb_max_xyz": [0.21, 0.18, 1.2]},
                          "destination_overhang": {"largest_overhang_m": 0.0816, "flag": DESTINATION_OVERHANG}}
    estimated = packet._measured_and_estimated(preparation_record, None, None)[1]
    assert ("The object's footprint is larger than the supported part of the surface at that spot: about 8.2 cm "
            "hangs over its edge; its centre is over the surface.") in estimated
    assert any(line.startswith(f"`{DESTINATION_OVERHANG}`: ") for line in estimated)
    unflagged = {**preparation_record, "destination_overhang": {"largest_overhang_m": 0.0}}
    assert not any("hangs over" in line for line in packet._measured_and_estimated(unflagged, None, None)[1])


#: The room's glass shelf made 0.12 m deep, and a cup 0.16 m across: more than the shelf's depth.
SHALLOW, WIDE_CUP = ((0.55, 1.19, 2.88), (1.15, 1.2, 3.0)), ((-0.68, 0.75, 2.42), (-0.52, 0.87, 2.58))


@pytest.mark.parametrize("flipped", [False, True])
def test_compile_places_the_wide_cup_on_the_shallow_shelf_clear_of_the_wall_and_says_how_much_hangs_over(
        tmp_path, flipped):
    from blueprint_pipeline import job_packet_export as packet
    value = preparation.compile_website_scene_preparation(
        **_arguments(tmp_path, shelf=SHALLOW, cup_box=WIDE_CUP, flipped=flipped))
    assert "task_destination_surface_contact_required" not in value["blockers"], value["blockers"]
    record = value["destination_overhang"]
    assert record["flag"] == DESTINATION_OVERHANG and record["supported_share"] >= RESTING_OUTLINE_MIN_SHARE
    assert record["lattice_points"] == (3 * (RESTING_OUTLINE_GRID_STEPS - 1) + 1) ** 2  # it slid: a bounded search
    assert record["largest_overhang_m"] > 0 and record["slid_m"] > 0  # the room's wall stood in its box
    # Read again where the destination's choice settled it: the position the rule scores is the one judged.
    assert record["moved_after_placement_m"] == 0.0 and value["destination_choice"]["basis"]
    runtime_to_sim = np.asarray(value["coordinate_frame"]["runtime_to_simulator"])
    final = np.asarray(value["destination"]["position_world_m"])
    placed = np.linalg.solve(runtime_to_sim, [*final, 1.0])[:3]
    assert placed[[0, 2]] == pytest.approx(record["centre_runtime"], abs=1e-9)
    # The pose placed, against the real shelf and wall in the simulator's frame: the share holds, the wall clears.
    turn = FLIP if flipped else np.eye(3)
    to_sim = runtime_to_sim[:3, :3] @ (RUNTIME_SCALE * turn)
    shelf = [to_sim @ corner + runtime_to_sim[:3, :3] @ RUNTIME_TRANSLATION for corner in map(np.asarray, SHALLOW)]
    low, high = np.minimum(*shelf), np.maximum(*shelf)
    size = np.subtract(value["subject"]["aabb_max_xyz"], value["subject"]["aabb_min_xyz"])
    over = [max(0.0, min(final[axis] + size[axis] / 2, high[axis]) - max(final[axis] - size[axis] / 2, low[axis]))
            for axis in (0, 1)]
    assert over[0] * over[1] / (size[0] * size[1]) >= RESTING_OUTLINE_MIN_SHARE
    wall = (to_sim @ [0.0, 0.0, 3.0] + runtime_to_sim[:3, :3] @ RUNTIME_TRANSLATION)[1]
    back = final[1] - size[1] / 2 if wall < final[1] else final[1] + size[1] / 2
    assert (back >= wall) if wall < final[1] else (back <= wall)
    line = f"about {record['largest_overhang_m'] * 100:.1f} cm hangs over its edge"
    assert any(line in text for text in packet._measured_and_estimated(value, None, None)[1])


def _clear(mesh, lower, upper, whole, *, start=ELSEWHERE):
    from blueprint_pipeline.website_support_geometry import support_under
    return overhang.placed_clear(mesh, lower, upper, list((lower + upper) / 2), whole, up=1, meters_per_unit=1.0,
                                 up_sign=1, radius=website_target_radius(meters_per_unit=1.0), runtime_to_sim=np.eye(4),
                                 start=start, start_support=None, clearance=0.0, look=support_under)


def test_clutter_on_a_wholly_supported_surface_slides_the_object_clear_of_it():
    from blueprint_pipeline.website_support_geometry import support_under
    # A book where the object was to go, on a 30 cm deep, 60 cm long shelf.
    mesh = _mesh(0.3, boxes=[((0.22, TOP, 0.05), (0.38, TOP + 0.04, 0.25))])
    lower, upper = _footprint(0.3, 0.15, 0.12, 0.12, height=0.2)
    whole = support_under(mesh, lower, upper, up=1, meters_per_unit=1.0)
    assert whole is not None  # the shelf is under the whole footprint: only the book stands in the way
    surface, placement, position = _clear(mesh, lower, upper, whole)
    x, _ = _metres(position, mpu=1.0, up_sign=1)
    assert surface is not None and placement is not None
    assert x + 0.06 <= 0.22 + 1e-9 or x - 0.06 >= 0.38 - 1e-9  # its box no longer meets the book


def test_a_clear_wholly_supported_spot_takes_the_old_path_and_clutter_everywhere_is_the_old_stop():
    from blueprint_pipeline.website_support_geometry import support_under
    lower, upper = _footprint(0.3, 0.15, 0.12, 0.12, height=0.2)
    clear = _mesh(0.3)
    whole = support_under(clear, lower, upper, up=1, meters_per_unit=1.0)
    assert _clear(clear, lower, upper, whole) == (whole, None, list((lower + upper) / 2))
    full = _mesh(0.3, boxes=[((0.0, TOP, 0.0), (0.6, TOP + 0.05, 0.3))])
    whole = support_under(full, lower, upper, up=1, meters_per_unit=1.0)
    if whole is not None:
        assert _clear(full, lower, upper, whole)[:2] == (None, None)


def test_the_objects_own_scan_where_it_starts_is_not_clutter():
    from blueprint_pipeline.website_support_geometry import support_under
    # The object is scanned where it stands on this shelf; its grounded place overlaps that scan.
    own = ((0.24, TOP, 0.09), (0.36, TOP + 0.2, 0.21))
    mesh = _mesh(0.3, boxes=[own])
    lower, upper = _footprint(0.3, 0.15, 0.12, 0.12, height=0.2)
    whole = support_under(mesh, lower, upper, up=1, meters_per_unit=1.0)
    start = (np.array(own[0]), np.array(own[1]))
    if whole is not None:
        assert _clear(mesh, lower, upper, whole, start=start) == (whole, None, list((lower + upper) / 2))


def test_compile_names_the_destination_s_surface_by_its_own_points_and_the_start_s_by_none(tmp_path, monkeypatch):
    from blueprint_pipeline.website_task_masks import masked_view_points
    calls = []
    real = preparation.support_under

    def recording(mesh, lower, upper, **options):
        calls.append(options.get("shown_by"))
        return real(mesh, lower, upper, **options)

    monkeypatch.setattr(preparation, "support_under", recording)
    args = _arguments(tmp_path, onto_table=True)  # the cup moves to the other end of the table it stands on
    value = preparation.compile_website_scene_preparation(**args)
    assert value["destination"] is not None, value["blockers"]
    table = args["task_masks"]["targets"][1]
    lifted = np.concatenate(masked_view_points(table["track"], args["source_geometry"]["frames"]))
    shown = [points for points in calls if points is not None]
    assert shown and all(len(points) == len(lifted) for points in shown)  # the seal's lookup and the placement's
    assert any(points is None for points in calls)  # the cup's own start is looked up as before
    # In the collider's frame, as registration puts them (the room's source frame is OpenCV's, y down).
    source = np.diag([1.0, -1.0, -1.0])
    assert shown[0] == pytest.approx(lifted @ (RUNTIME_SCALE * source).T + RUNTIME_TRANSLATION)


def test_an_object_wholly_inside_clutter_taller_and_wider_than_it_slides_out_of_it():
    from blueprint_pipeline.website_support_geometry import support_under
    # The synthetic set's seed 28: a 25 cm box on the table, wider than the object, its box within the clutter's
    # sides and under its top, meeting none of its faces.
    mesh = _mesh(0.3, boxes=[((0.2, TOP, 0.03), (0.4, TOP + 0.3, 0.27))])
    lower, upper = _footprint(0.3, 0.15, 0.12, 0.12, height=0.2)
    whole = support_under(mesh, lower, upper, up=1, meters_per_unit=1.0)
    assert whole is not None  # the shelf's faces run on under the clutter, as a scan of stacked parts can leave them
    spot = overhang._Spot(mesh, whole, lower, upper, up=1, up_sign=1, meters_per_unit=1.0, placement_residual_m=None,
                          start=None, clearance=None)
    box = spot.box(np.array([0.3, 0.15]))
    near = spot.obstacles.triangles
    near = near[(near.max(axis=1) >= box[0]).all(axis=1) & (near.min(axis=1) <= box[1]).all(axis=1)]
    assert not overhang._crossing(near, *box).any() and spot.obstacles.enclosed(*box)  # no face met; enclosed
    surface, placement, position = _clear(mesh, lower, upper, whole)
    x, _ = _metres(position, mpu=1.0, up_sign=1)
    assert surface is not None and placement is not None
    assert x + 0.06 <= 0.2 + 1e-9 or x - 0.06 >= 0.4 - 1e-9  # out of the clutter, beside it on the shelf


def test_the_inside_of_a_ring_and_a_scan_wound_inward_are_not_enclosing():
    from blueprint_pipeline.website_support_geometry import support_under
    lower, upper = _footprint(0.3, 0.15, 0.12, 0.12, height=0.2)
    # Four walls around an open 20 cm square: met from inside on their fronts, the space between them is free.
    ring = _mesh(0.3, boxes=[((0.18, TOP, 0.03), (0.2, TOP + 0.3, 0.27)), ((0.4, TOP, 0.03), (0.42, TOP + 0.3, 0.27)),
                             ((0.2, TOP, 0.03), (0.4, TOP + 0.3, 0.05)), ((0.2, TOP, 0.25), (0.4, TOP + 0.3, 0.27))])
    whole = support_under(ring, lower, upper, up=1, meters_per_unit=1.0)
    assert whole is not None and _clear(ring, lower, upper, whole) == (whole, None, list((lower + upper) / 2))
    # The same clutter as a closed box wound inward (an inconsistently wound scan): nothing reads as enclosed, the
    # old path is kept.
    inward = trimesh.creation.box(extents=[0.2, 0.3, 0.24],
                                  transform=trimesh.transformations.translation_matrix([0.3, TOP + 0.15, 0.15]))
    shelf = _mesh(0.3)
    mesh = trimesh.Trimesh(vertices=np.concatenate([shelf.vertices, inward.vertices]), process=False,
                           faces=np.concatenate([shelf.faces, np.asarray(inward.faces)[:, ::-1] + len(shelf.vertices)]))
    whole = support_under(mesh, lower, upper, up=1, meters_per_unit=1.0)
    assert whole is not None and _clear(mesh, lower, upper, whole) == (whole, None, list((lower + upper) / 2))


@pytest.mark.parametrize("seed", [0, 4])
def test_a_thin_shelf_s_underside_does_not_lower_its_top_into_its_own_edge(seed):
    # The synthetic set's seed 41: a 12 mm shelf on a wall, scanned bumpy (2 mm). Its fit grew across the edge over
    # its underside, read the top 5 mm low, and the shelf's own edge then stood in a 26 cm object's box everywhere.
    rng = np.random.default_rng(seed)
    shelf = trimesh.creation.box(extents=[0.84, 0.012, 0.188],
                                 transform=trimesh.transformations.translation_matrix([0.7, 1.205, -0.506]))
    wall = trimesh.creation.box(extents=[3, 3, 0.1], transform=trimesh.transformations.translation_matrix([0.7, 1.5, -0.65]))
    mesh = trimesh.util.concatenate([shelf, wall]).subdivide_to_size(0.1)
    mesh = trimesh.Trimesh(vertices=mesh.vertices + rng.normal(0, 0.002, mesh.vertices.shape), faces=mesh.faces,
                           process=False)
    core = support_under(mesh, [0.65, 1.215, -0.553], [0.75, 1.44, -0.453], up=1, meters_per_unit=1.0)
    normals = mesh.face_normals[core["face_indices"]][:, 1]
    assert (normals < 0).any()  # the fit still reaches the underside ...
    assert core["top_runtime_units"] == pytest.approx(1.211, abs=0.005)  # ... but reads the top from the top
    lower, upper = np.array([0.572, 1.215, -0.626]), np.array([0.829, 1.44, -0.380])  # deeper than the shelf, into the wall
    surface, placement, position = _place(mesh, lower, upper)
    assert surface is not None and placement is not None
    assert position[2] - 0.123 >= -0.6  # its box clear of the wall
    assert placement.record["supported_share"] >= RESTING_OUTLINE_MIN_SHARE


def test_clutter_wider_than_the_object_where_only_its_centre_is_carried_is_searched_past():
    # The synthetic set's seed 43: a 32 x 21 cm box where a 21 cm object was to go, its footprint not wholly carried
    # there. The overhang search looked only within one footprint; it now runs on outward, as on a whole surface.
    mesh = _mesh(0.2, boxes=[((0.15, TOP, 0.0), (0.45, TOP + 0.1, 0.2))])
    lower, upper = _footprint(0.3, 0.1, 0.12, 0.24)  # deeper than the shelf: no surface carries all of it
    assert support_under(mesh, lower, upper, up=1, meters_per_unit=1.0) is None
    surface, placement, position = _place(mesh, lower, upper)
    assert surface is not None and placement is not None
    x, _ = _metres(position, mpu=1.0, up_sign=1)
    assert x + 0.06 <= 0.15 + 1e-9 or x - 0.06 >= 0.45 - 1e-9  # clear of the clutter, on the shelf
    assert placement.record["slid_m"] >= 0.21 and placement.record["supported_share"] >= RESTING_OUTLINE_MIN_SHARE
