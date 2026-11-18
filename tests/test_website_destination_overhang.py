"""An object a little larger than its destination surface is set down with its centre over it, and flagged.

Hermetic: a shelf 0.6 m long, made of 1 cm triangles at 1 m, over a floor. The lookups are compile's own
``support_under``; no provider is reached and nothing is paid for.
"""
# Covers: src/blueprint_pipeline/website_destination_overhang.py
# Covers: src/blueprint_pipeline/job_packet_export.py
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from blueprint_pipeline import website_destination_overhang as overhang
from blueprint_pipeline.atlas_scene_estimates import DESTINATION_OVERHANG
from blueprint_pipeline.website_support_geometry import RESTING_OUTLINE_MIN_SHARE, support_under
from blueprint_pipeline.website_surface_target_admission import website_target_radius

TOP = 1.0


def _shelf(depth: float) -> trimesh.Trimesh:
    """A level shelf ``depth`` deep (along z) and 0.6 m long (along x) at ``TOP``, over a floor; y is up."""
    xs, zs = np.linspace(0.0, 0.6, 61), np.linspace(0.0, depth, int(round(depth / 0.01)) + 1)
    grid = np.array([[x, TOP, z] for x in xs for z in zs])
    columns = len(zs)
    faces = []
    for i in range(len(xs) - 1):
        for j in range(columns - 1):
            a = i * columns + j
            faces += [[a, a + 1, a + columns], [a + 1, a + columns + 1, a + columns]]
    floor = np.array([[-2.0, 0.0, -2.0], [2.0, 0.0, -2.0], [2.0, 0.0, 2.0], [-2.0, 0.0, 2.0]])
    offset = len(grid)
    faces += [[offset, offset + 2, offset + 1], [offset, offset + 3, offset + 2]]
    return trimesh.Trimesh(vertices=np.vstack([grid, floor]), faces=np.array(faces), process=False)


def _footprint(centre_x: float, centre_z: float, size_x: float, size_z: float):
    lower = np.array([centre_x - size_x / 2, TOP, centre_z - size_z / 2])
    return lower, lower + [size_x, 0.45, size_z]


def _place(mesh, lower, upper):
    return overhang.centre_supported(mesh, lower, upper, up=1, meters_per_unit=1.0,
                                     radius=website_target_radius(meters_per_unit=1.0))


def test_an_object_a_little_larger_than_the_shelf_rests_with_its_centre_over_it_and_is_flagged():
    # The lamp's case: 0.18 x 0.21 m on a shelf 0.16 m deep, its long side across the shelf as it stands.
    mesh = _shelf(0.16)
    lower, upper = _footprint(0.3, 0.08, 0.18, 0.21)
    assert support_under(mesh, lower, upper, up=1, meters_per_unit=1.0) is None  # the whole footprint: no surface
    surface, record = _place(mesh, lower, upper)
    assert surface is not None and surface["top_runtime_units"] == pytest.approx(TOP)
    assert record["flag"] == DESTINATION_OVERHANG and record["required_share"] == RESTING_OUTLINE_MIN_SHARE
    rows = {row["orientation"]: row for row in record["orientations"]}
    # Turned to lie along the shelf it has more over it: 16 of its 18 cm across, 1 cm over each edge.
    assert rows["turned_to_the_surface"]["supported_share"] > rows["as_is"]["supported_share"]
    assert record["orientation"] == "turned_to_the_surface" and abs(record["turn_degrees"]) == pytest.approx(90.0, abs=0.5)
    assert record["supported_share"] == pytest.approx(0.16 / 0.18, abs=0.03)
    assert record["largest_overhang_m"] == pytest.approx(0.01, abs=0.003)
    assert rows["as_is"]["largest_overhang_m"] == pytest.approx(0.025, abs=0.003)
    assert record["footprint_m"] == pytest.approx([0.18, 0.21])


def test_a_centre_off_the_edge_stays_blocked():
    mesh = _shelf(0.16)
    lower, upper = _footprint(0.3, 0.20, 0.18, 0.21)  # its centre 4 cm past the shelf's front edge
    assert _place(mesh, lower, upper) == (None, None)


def test_under_half_of_the_footprint_over_the_surface_stays_blocked():
    mesh = _shelf(0.11)
    lower, upper = _footprint(0.3, 0.055, 0.5, 0.5)  # 11 of its 50 cm over the shelf, however it is turned
    disc = _footprint(0.3, 0.055, 0.1, 0.1)
    assert support_under(mesh, *disc, up=1, meters_per_unit=1.0) is not None  # its centre and disc are on it
    assert _place(mesh, lower, upper) == (None, None)


def test_the_packet_says_how_much_hangs_over_beside_the_estimate_lines():
    from blueprint_pipeline import job_packet_export as packet
    preparation = {"subject": {"aabb_min_xyz": [0.0, 0.0, 0.75], "aabb_max_xyz": [0.21, 0.18, 1.2]},
                   "destination_overhang": {"largest_overhang_m": 0.0206, "flag": DESTINATION_OVERHANG}}
    estimated = packet._measured_and_estimated(preparation, None, None)[1]
    assert ("The object is a little larger than the destination surface: about 2 cm hangs over its edge; its "
            "centre is over the surface.") in estimated
    assert any(line.startswith(f"`{DESTINATION_OVERHANG}`: ") for line in estimated)
    assert estimated.index(next(line for line in estimated if line.startswith("Object size"))) < estimated.index(
        next(line for line in estimated if line.startswith("The object is a little larger")))
