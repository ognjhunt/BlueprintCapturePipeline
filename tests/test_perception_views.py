"""Hermetic tests for the view-ring planner (pure geometry, no GPU/renderer/network)."""
from __future__ import annotations

import math

import pytest

from blueprint_pipeline.scene_placement import (
    MultiViewPerceptionSceneSpatialIndex,
    PerceptionSceneSpatialIndex,
    assemble_views,
    generate_view_ring,
    view_ring_for_bounds,
)


def _dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def test_view_ring_count_radius_and_targets() -> None:
    center = (2.0, 1.0, 1.0)
    radius = 3.0
    cams = generate_view_ring(center, radius, n_azimuths=8, elevations_deg=(15.0, 45.0))
    assert len(cams) == 8 * 2                          # azimuths x elevations
    for c in cams:
        assert _dist(c["eye"], center) == pytest.approx(radius)   # every eye on the orbit sphere
        assert tuple(c["target"]) == center                       # all aim at the centre
        assert c["vfov"] == pytest.approx(math.radians(60.0))     # degrees -> radians in output


def test_view_ring_azimuths_evenly_spaced_and_elevation_sign() -> None:
    center = (0.0, 0.0, 0.0)
    cams = generate_view_ring(center, 1.0, n_azimuths=4, elevations_deg=(30.0,))
    # 4 azimuths at 0/90/180/270 deg -> eyes at +x, +y, -x, -y (scaled by cos(elev)), all +z up.
    cphi = math.cos(math.radians(30.0))
    assert cams[0]["eye"][0] == pytest.approx(cphi)               # azimuth 0 -> +x
    assert cams[1]["eye"][1] == pytest.approx(cphi)               # azimuth 90 -> +y
    for c in cams:
        assert c["eye"][2] == pytest.approx(math.sin(math.radians(30.0)))  # positive elevation -> above
    # all eyes distinct
    assert len({tuple(round(v, 6) for v in c["eye"]) for c in cams}) == 4


def test_view_ring_for_bounds_centers_and_sizes_from_aabb() -> None:
    bbox_min, bbox_max = (0.0, 0.0, 0.0), (2.0, 0.0, 0.0)   # a 2m segment along x
    cams = view_ring_for_bounds(bbox_min, bbox_max, margin=2.0, n_azimuths=6)
    assert len(cams) == 6
    center = (1.0, 0.0, 0.0)
    half_diag = 1.0                                          # half of the 2m diagonal
    expected_radius = half_diag * 2.0
    for c in cams:
        assert tuple(c["target"]) == center
        assert _dist(c["eye"], center) == pytest.approx(expected_radius)


def test_view_ring_validates_inputs() -> None:
    with pytest.raises(ValueError):
        generate_view_ring((0, 0, 0), 1.0, n_azimuths=0)
    with pytest.raises(ValueError):
        generate_view_ring((0, 0, 0), 0.0)


def test_ring_camera_round_trips_through_perception_backend() -> None:
    # A ring camera looks at the centre from `radius` away; an on-axis detection at depth=radius
    # must unproject its box-centre back to the centre (eye + radius*forward == centre).
    center = (2.0, 1.0, 1.0)
    radius = 3.0
    cam = generate_view_ring(center, radius, n_azimuths=1, elevations_deg=(25.0,),
                             width=640, height=480)[0]
    det = {"label": "faucet", "bbox_px": (300, 220, 340, 260), "confidence": 0.9}
    idx = PerceptionSceneSpatialIndex([det], lambda px, py: radius, cam)
    objs = idx.objects()
    assert len(objs) == 1
    for i in range(3):
        assert objs[0].centroid[i] == pytest.approx(center[i], abs=1e-6)


def test_assemble_views_zips_and_validates_lengths() -> None:
    cams = generate_view_ring((0, 0, 0), 2.0, n_azimuths=2)
    dets = [[{"label": "x", "bbox_px": (0, 0, 1, 1), "confidence": 0.5}], []]
    depths = [lambda px, py: 2.0, lambda px, py: 2.0]
    views = assemble_views(cams, dets, depths)
    assert len(views) == 2
    assert set(views[0].keys()) == {"detections", "depth_provider", "camera"}
    # feeds straight into the multi-view index without massaging
    objs = MultiViewPerceptionSceneSpatialIndex(
        views, min_views=1, require_metric_authority=False
    ).objects()
    assert isinstance(objs, list)
    with pytest.raises(ValueError):
        assemble_views(cams, dets, depths[:1])     # length mismatch is a caller bug


def test_full_chain_bounds_to_fused_object() -> None:
    # End-to-end pure path: bounds -> ring -> (synthetic centered detections + constant depth) ->
    # fused single object back at the scene centre. Proves the pieces compose.
    bbox_min, bbox_max = (1.5, 0.5, 0.5), (2.5, 1.5, 1.5)
    center = (2.0, 1.0, 1.0)
    cams = view_ring_for_bounds(bbox_min, bbox_max, margin=1.6, n_azimuths=6, elevations_deg=(20.0,),
                                width=640, height=480)
    radius = _dist(cams[0]["eye"], center)
    def det():
        return [{"label": "sink", "bbox_px": (300, 220, 340, 260), "confidence": 0.9}]
    views = assemble_views(cams, [det() for _ in cams], [(lambda px, py: radius) for _ in cams])
    objs = MultiViewPerceptionSceneSpatialIndex(
        views, merge_gap=0.25, min_views=1, require_metric_authority=False
    ).objects()
    assert len(objs) == 1                            # all views fuse to one object...
    assert objs[0].extra["n_views"] == 6
    for i in range(3):
        assert objs[0].centroid[i] == pytest.approx(center[i], abs=1e-6)   # ...at the scene centre
