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
from blueprint_pipeline.website_support_geometry import RESTING_OUTLINE_MIN_SHARE, support_under
from blueprint_pipeline.website_surface_target_admission import website_target_radius
from tests.test_website_unscanned_surface import (FLIP, RUNTIME_SCALE, RUNTIME_TRANSLATION, _arguments,
                                                  exact_registration, no_model)

__all__ = ["exact_registration", "no_model"]  # the room's own registration, and no model call, for every test here
TOP, HEIGHT = 1.0, 0.45
FRAMES = [(1.0, 1), (2.0, 1), (1.0, -1)]  # (metres per unit, up sign)


def _grid(x0, x1, z0, z1, y):
    xs, zs = np.linspace(x0, x1, int(round((x1 - x0) / 0.01)) + 1), np.linspace(z0, z1, int(round((z1 - z0) / 0.01)) + 1)
    points = np.array([[x, y, z] for x in xs for z in zs])
    faces = [[i * len(zs) + j, i * len(zs) + j + 1, (i + 1) * len(zs) + j] for i in range(len(xs) - 1)
             for j in range(len(zs) - 1)]
    faces += [[i * len(zs) + j + 1, (i + 1) * len(zs) + j + 1, (i + 1) * len(zs) + j] for i in range(len(xs) - 1)
              for j in range(len(zs) - 1)]
    return points, np.array(faces)


def _mesh(depth: float, *, mpu: float = 1.0, up_sign: int = 1, wall: bool = False, lid: bool = False):
    """A level shelf ``depth`` deep (z 0 to depth) and 0.6 m long at ``TOP`` m over a floor, in metres turned into
    runtime units; ``wall``: a wall at z = 0 behind it; ``lid``: a second shelf 10 cm above it."""
    parts = [_grid(0.0, 0.6, 0.0, depth, TOP), (np.array([[-2.0, 0, -2], [2, 0, -2], [2, 0, 2], [-2, 0, 2]]),
                                                np.array([[0, 2, 1], [0, 3, 2]]))]
    if wall:
        parts.append((np.array([[-1.0, 0, 0], [1.6, 0, 0], [1.6, 2.5, 0], [-1, 2.5, 0]]), np.array([[0, 1, 2], [0, 2, 3]])))
    if lid:
        parts.append(_grid(0.0, 0.6, 0.0, depth, TOP + 0.1))
    vertices, faces, offset = [], [], 0
    for points, triangles in parts:
        vertices.append(points)
        faces.append(triangles + offset)
        offset += len(points)
    vertices, faces = np.concatenate(vertices) / mpu, np.concatenate(faces)
    if up_sign == -1:  # turned over about x: a proper rotation, up is -Y
        vertices = vertices @ FLIP.T
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _footprint(x: float, z: float, size_x: float, size_z: float, *, mpu: float = 1.0, up_sign: int = 1):
    """The footprint at (x, z) on the shelf, its bottom on its top, as compile's ``footprint_at`` gives it."""
    lower = np.array([x - size_x / 2, TOP, z - size_z / 2]) / mpu
    upper = np.array([x + size_x / 2, TOP + HEIGHT, z + size_z / 2]) / mpu
    if up_sign == -1:
        lower, upper = lower @ FLIP.T, upper @ FLIP.T
        lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
    return lower, upper


def _place(mesh, lower, upper, *, mpu=1.0, up_sign=1):
    return overhang.centre_supported(mesh, lower, upper, list((lower + upper) / 2), up=1, meters_per_unit=mpu,
                                     up_sign=up_sign, radius=website_target_radius(meters_per_unit=mpu))


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
    surface, record, position = _place(mesh, lower, upper, mpu=mpu, up_sign=up_sign)
    assert surface is not None and surface["top_runtime_units"] * mpu * up_sign == pytest.approx(TOP)
    assert record["flag"] == DESTINATION_OVERHANG and record["orientation"] == "as_placed"
    assert record["supported_share"] == pytest.approx(0.16 / 0.18, abs=0.03)
    assert record["largest_overhang_m"] == pytest.approx(0.01, abs=0.002) and record["slid_m"] == 0.0
    assert record["footprint_m"] == pytest.approx([0.18, 0.21])
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
    surface, record, position = _place(mesh, lower, upper, mpu=mpu, up_sign=up_sign)
    assert surface is not None and record["slid_m"] >= 0.01 - 1e-9
    x, z = _metres(position, mpu=mpu, up_sign=up_sign)
    assert z - 0.09 >= -1e-9  # its back face clears the wall at z = 0
    assert _shelf_share(x, z, 0.21, 0.18, 0.16) >= RESTING_OUTLINE_MIN_SHARE
    assert record["largest_overhang_m"] == pytest.approx(z + 0.09 - 0.16, abs=0.003)  # all of it at the front


def test_no_room_above_the_shelf_stays_blocked():
    mesh = _mesh(0.16, lid=True)  # a shelf 10 cm above: a 45 cm box meets it everywhere
    lower, upper = _footprint(0.3, 0.08, 0.21, 0.18)
    assert _place(mesh, lower, upper)[:2] == (None, None)


def test_the_placement_is_read_again_where_relocation_or_a_choice_puts_it():
    mesh = _mesh(0.16)
    lower, upper = _footprint(0.3, 0.08, 0.21, 0.18)
    radius = website_target_radius(meters_per_unit=1.0)
    _, record, position = _place(mesh, lower, upper)
    same, codes = overhang.settled(mesh, record, lower, upper, position, up=1, meters_per_unit=1.0, radius=radius)
    assert codes == [] and same["supported_share"] == pytest.approx(record["supported_share"])
    assert same["moved_after_placement_m"] == 0.0
    off = [position[0], position[1], 0.15]  # moved to the front edge: its disc and most of it off the shelf
    assert overhang.settled(mesh, record, lower, upper, off, up=1, meters_per_unit=1.0, radius=radius) == (
        None, [overhang.STOP])
    small_lower, small_upper = _footprint(0.3, 0.08, 0.12, 0.12)  # wholly over the shelf there: no flag, no record
    assert overhang.settled(mesh, record, small_lower, small_upper, position, up=1, meters_per_unit=1.0,
                            radius=radius) == (None, [])


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
