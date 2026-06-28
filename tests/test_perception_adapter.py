"""Hermetic tests for the SAM3/DA3 -> perception-view adapter (no GPU/torch/network)."""
from __future__ import annotations

import pytest

from blueprint_pipeline.scene_placement import (
    MultiViewPerceptionSceneSpatialIndex,
    build_perception_view,
    build_perception_views,
    build_perception_views_from_frames,
    depth_provider_from_map,
    detections_from_sam3,
    generate_view_ring,
)


# ----------------------------- detections_from_sam3 -----------------------------

def test_detections_pixel_boxes_pass_through() -> None:
    recs = [{"label": "faucet", "bbox_px": (10, 20, 50, 60), "confidence": 0.8}]
    out = detections_from_sam3(recs, width=640, height=480)
    assert out == [{"label": "faucet", "bbox_px": (10.0, 20.0, 50.0, 60.0), "confidence": 0.8}]


def test_detections_normalized_boxes_scaled_to_pixels() -> None:
    recs = [{"prompt": "sink", "bbox_xyxy": (0.5, 0.25, 0.75, 0.5), "score": 0.6}]
    out = detections_from_sam3(recs, width=640, height=480)
    assert out[0]["label"] == "sink"                       # 'prompt' -> label
    assert out[0]["bbox_px"] == (320.0, 120.0, 480.0, 240.0)
    assert out[0]["confidence"] == 0.6                     # 'score' -> confidence


def test_detections_field_variants_and_box_reorder_and_skips() -> None:
    recs = [
        {"name": "cup", "box": (50, 60, 10, 20)},                       # flipped -> reordered, no score -> 1.0
        {"label": "noise"},                                            # no box -> skipped
        {"label": "bad", "bbox": (1, 2, 3)},                           # wrong length -> skipped
        {"class": "pan", "xyxy": (0, 0, 100, 100), "id": "k7"},
    ]
    out = detections_from_sam3(recs, width=200, height=200)
    assert [d["label"] for d in out] == ["cup", "pan"]
    assert out[0]["bbox_px"] == (10.0, 20.0, 50.0, 60.0)               # min/max ordering
    assert out[0]["confidence"] == 1.0
    assert out[1]["id"] == "k7"


# ----------------------------- depth_provider_from_map -----------------------------

def test_depth_provider_samples_and_clamps() -> None:
    # 2x2 map; rows indexed by py, cols by px (same res as camera).
    depth_map = [[1.0, 2.0], [3.0, 4.0]]
    p = depth_provider_from_map(depth_map, cam_width=2, cam_height=2)
    assert p(0, 0) == 1.0
    assert p(1, 0) == 2.0      # px=1 -> col 1
    assert p(0, 1) == 3.0      # py=1 -> row 1
    assert p(99, 99) == 4.0    # clamps to the far corner
    assert p(-5, -5) == 1.0    # clamps to origin


def test_depth_provider_scales_when_map_lower_res_than_render() -> None:
    # depth map is 2x2 but the render/detections are 640x480 -> pixel coords get scaled down.
    depth_map = [[1.0, 2.0], [3.0, 4.0]]
    p = depth_provider_from_map(depth_map, cam_width=640, cam_height=480)
    assert p(0, 0) == 1.0          # top-left
    assert p(639, 0) == 2.0        # far right -> col 1
    assert p(639, 479) == 4.0      # far corner
    assert p(320, 240) == 4.0      # mid -> rounds to col1/row1 at this tiny res


def test_depth_provider_empty_map_raises() -> None:
    p = depth_provider_from_map([], cam_width=10, cam_height=10)
    with pytest.raises(ValueError):
        p(0, 0)


# ----------------------------- assembly + end-to-end -----------------------------

def _cam(eye, target, w=640, h=480, vfov=1.0):
    return {"eye": eye, "target": target, "vfov": vfov, "width": w, "height": h}


def test_build_perception_view_shape() -> None:
    cam = _cam((0, -3, 1), (0, 0, 1))
    recs = [{"label": "faucet", "bbox_px": (300, 220, 340, 260), "confidence": 0.9}]
    depth_map = [[3.0, 3.0], [3.0, 3.0]]
    view = build_perception_view(cam, recs, depth_map)
    assert set(view.keys()) == {"detections", "depth_provider", "camera", "samples_per_axis"}
    assert view["detections"][0]["label"] == "faucet"
    assert callable(view["depth_provider"])
    assert view["depth_provider"](320, 240) == 3.0


def test_build_perception_view_requires_camera_resolution() -> None:
    with pytest.raises(ValueError):
        build_perception_view({"eye": (0, 0, 0), "target": (0, 0, 1)}, [], [[1.0]])


def test_build_perception_views_length_mismatch_raises() -> None:
    cams = [_cam((0, -3, 1), (0, 0, 1)), _cam((3, 0, 1), (0, 0, 1))]
    with pytest.raises(ValueError):
        build_perception_views(cams, [[]], [[[1.0]], [[1.0]]])


def test_end_to_end_sam3_da3_to_fused_object() -> None:
    # Two orthogonal ring cameras of one object at world (0,0,1) at range 3. Each view: a centered
    # SAM3 box + a constant DA3 depth of 3m. The adapter -> fusion must yield ONE object at (0,0,1).
    center, radius = (0.0, 0.0, 1.0), 3.0
    cams = [
        _cam((0.0, -3.0, 1.0), center),   # looking +y
        _cam((3.0, 0.0, 1.0), center),    # looking -x
    ]
    recs_per_view = [
        [{"label": "faucet", "bbox_xyxy": (0.45, 0.42, 0.55, 0.58), "score": 0.9}],
        [{"label": "faucet", "bbox_xyxy": (0.45, 0.42, 0.55, 0.58), "score": 0.9}],
    ]
    depth_maps = [[[3.0, 3.0], [3.0, 3.0]], [[3.0, 3.0], [3.0, 3.0]]]
    views = build_perception_views(cams, recs_per_view, depth_maps)
    objs = MultiViewPerceptionSceneSpatialIndex(views, merge_gap=0.3, min_views=1).objects()
    assert len(objs) == 1
    assert objs[0].extra["n_views"] == 2
    for i in range(3):
        assert objs[0].centroid[i] == pytest.approx(center[i], abs=1e-6)


def test_build_views_from_frames_with_injected_models() -> None:
    # Inject fake SAM3 + DA3 callables (the worker-specific part) and confirm the adapter runs
    # them per frame and assembles fusion-ready views.
    cams = generate_view_ring((0, 0, 1), 3.0, n_azimuths=3, elevations_deg=(0.0,),
                              width=640, height=480)
    frames = ["frameA", "frameB", "frameC"]
    seen = []

    def fake_detect(frame):
        seen.append(frame)
        return [{"label": "sink", "bbox_px": (300, 220, 340, 260), "confidence": 0.7}]

    def fake_depth(frame):
        return [[3.0, 3.0], [3.0, 3.0]]

    views = build_perception_views_from_frames(frames, cams, detect=fake_detect, depth=fake_depth)
    assert len(views) == 3 and seen == frames           # ran detect once per frame, in order
    assert all(v["detections"][0]["label"] == "sink" for v in views)
    with pytest.raises(ValueError):
        build_perception_views_from_frames(frames, cams[:2], detect=fake_detect, depth=fake_depth)
