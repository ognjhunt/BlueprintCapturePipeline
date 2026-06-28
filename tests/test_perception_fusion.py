"""Hermetic tests for multi-view perception fusion (no GPU/SAM3/DA3/network)."""
from __future__ import annotations

import math

import pytest

from blueprint_pipeline.scene_placement import (
    MultiViewPerceptionSceneSpatialIndex,
    SceneObject,
    SceneSpatialIndex,
    aabb_iou,
    fuse_scene_objects,
)
from blueprint_pipeline.scene_placement.perception_fusion import aabb_coverage


def _obj(label, center, size=(0.2, 0.2, 0.2), conf=0.9, oid=None):
    cx, cy, cz = center
    hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2
    return SceneObject(
        id=oid or f"{label}@{cx},{cy},{cz}",
        label=label,
        bbox_min=(cx - hx, cy - hy, cz - hz),
        bbox_max=(cx + hx, cy + hy, cz + hz),
        centroid=(cx, cy, cz),
        source="perception",
        confidence=conf,
    )


# ----------------------------- aabb_iou -----------------------------

def test_aabb_iou_identical_disjoint_and_degenerate() -> None:
    assert aabb_iou((0, 0, 0), (1, 1, 1), (0, 0, 0), (1, 1, 1)) == pytest.approx(1.0)
    # disjoint boxes
    assert aabb_iou((0, 0, 0), (1, 1, 1), (5, 5, 5), (6, 6, 6)) == 0.0
    # half-overlap along x: inter = 0.5, union = 1 + 1 - 0.5 = 1.5 -> 1/3
    assert aabb_iou((0, 0, 0), (1, 1, 1), (0.5, 0, 0), (1.5, 1, 1)) == pytest.approx(1 / 3)
    # flat box (zero z extent) -> zero volume -> IoU 0 (this is what merge_gap backstops)
    assert aabb_iou((0, 0, 0), (1, 1, 0), (0, 0, 0), (1, 1, 1)) == 0.0


# ----------------------------- fuse_scene_objects -----------------------------

def test_fuse_merges_same_object_across_views_via_iou() -> None:
    # two views see the same faucet at ~the same spot (boxes overlap heavily)
    a = _obj("faucet", (2.5, 1.15, 1.0))
    b = _obj("faucet", (2.52, 1.14, 1.01))
    fused = fuse_scene_objects([a, b], merge_iou=0.2)
    assert len(fused) == 1
    o = fused[0]
    assert o.source == "perception_fused"
    assert o.extra["n_views"] == 2
    # median centroid sits between the two observations
    assert o.centroid[0] == pytest.approx(2.51)
    assert o.centroid[1] == pytest.approx(1.145)


def test_fuse_keeps_distinct_same_label_objects_apart() -> None:
    # two different chairs across the room -> IoU 0, no gap -> stay separate
    chairs = [_obj("chair", (0.0, 0.0, 0.5)), _obj("chair", (4.0, 3.0, 0.5))]
    fused = fuse_scene_objects(chairs, merge_iou=0.25)
    assert len(fused) == 2


def test_min_views_rejects_single_view_outlier() -> None:
    # three views agree on the faucet; one stray view hallucinates a faucet 5m away
    good = [_obj("faucet", (2.5, 1.15, 1.0), conf=c) for c in (0.9, 0.85, 0.92)]
    stray = _obj("faucet", (7.5, 1.0, 1.0), conf=0.4)
    fused = fuse_scene_objects(good + [stray], merge_iou=0.2, min_views=2)
    assert len(fused) == 1                      # the lone stray is dropped
    assert fused[0].centroid[0] == pytest.approx(2.5)
    assert fused[0].extra["n_views"] == 3
    # at min_views=1 the stray survives as its own object
    assert len(fuse_scene_objects(good + [stray], merge_iou=0.2, min_views=1)) == 2


def test_median_merge_is_robust_to_a_near_outlier_view() -> None:
    # three overlapping observations; one is offset but still clusters. Median centroid
    # should track the two agreeing views, not be dragged by the odd one (a mean would be).
    members = [
        _obj("sink", (2.50, 1.15, 1.0), size=(0.4, 0.4, 0.3)),
        _obj("sink", (2.51, 1.16, 1.0), size=(0.4, 0.4, 0.3)),
        _obj("sink", (2.80, 1.40, 1.0), size=(0.5, 0.5, 0.3)),  # the odd-but-overlapping view
    ]
    fused = fuse_scene_objects(members, merge_iou=0.05, min_views=1)
    assert len(fused) == 1
    # median of {2.50,2.51,2.80} = 2.51 (mean would be ~2.60, pulled toward the outlier)
    assert fused[0].centroid[0] == pytest.approx(2.51)
    assert fused[0].centroid[1] == pytest.approx(1.16)


def test_fuse_is_label_case_insensitive() -> None:
    fused = fuse_scene_objects(
        [_obj("Faucet", (2.5, 1.15, 1.0)), _obj("faucet", (2.51, 1.15, 1.0))],
        merge_iou=0.2,
    )
    assert len(fused) == 1 and fused[0].extra["n_views"] == 2


def test_merge_gap_fuses_thin_boxes_with_low_iou() -> None:
    # orthogonal views of one object give cross-shaped thin boxes (IoU ~0) but coincident
    # centroids; merge_gap is what fuses them.
    a = _obj("faucet", (2.5, 1.15, 1.0), size=(0.3, 0.001, 0.3))
    b = _obj("faucet", (2.5, 1.15, 1.0), size=(0.001, 0.3, 0.3))
    assert len(fuse_scene_objects([a, b], merge_iou=0.5, merge_gap=None)) == 2   # IoU can't merge
    assert len(fuse_scene_objects([a, b], merge_iou=0.5, merge_gap=0.1)) == 1     # gap merges


def test_fuse_sorts_by_confidence_desc() -> None:
    fused = fuse_scene_objects(
        [_obj("stove", (0, 0, 0.5), conf=0.4), _obj("faucet", (2.5, 1.15, 1.0), conf=0.95)],
        merge_iou=0.25,
    )
    assert [o.label for o in fused] == ["faucet", "stove"]


# ----------------------------- MultiViewPerceptionSceneSpatialIndex -----------------------------

def _centered_detection(width, height, label="faucet"):
    cx, cy = width / 2.0, height / 2.0
    return {"label": label, "bbox_px": (cx - 20, cy - 20, cx + 20, cy + 20), "confidence": 0.9}


def test_multiview_index_fuses_two_camera_angles_into_one_object() -> None:
    # Same object at world (0,0,1) seen from two orthogonal cameras at range 3. A centered
    # detection unprojects its box-center to exactly (0,0,1) for both -> one fused object.
    w, h, vfov = 640, 480, 1.0
    views = [
        {  # camera looking +y
            "detections": [_centered_detection(w, h)],
            "depth_provider": lambda px, py: 3.0,
            "camera": {"eye": (0.0, -3.0, 1.0), "target": (0.0, 0.0, 1.0),
                       "vfov": vfov, "width": w, "height": h},
        },
        {  # camera looking -x
            "detections": [_centered_detection(w, h)],
            "depth_provider": lambda px, py: 3.0,
            "camera": {"eye": (3.0, 0.0, 1.0), "target": (0.0, 0.0, 1.0),
                       "vfov": vfov, "width": w, "height": h},
        },
    ]
    # orthogonal views -> thin cross boxes (IoU ~0) but coincident centroids: merge_gap fuses.
    idx = MultiViewPerceptionSceneSpatialIndex(views, merge_gap=0.3)
    assert isinstance(idx, SceneSpatialIndex)        # satisfies the protocol
    objs = idx.objects()
    assert len(objs) == 1
    o = objs[0]
    assert o.extra["n_views"] == 2
    assert o.centroid[0] == pytest.approx(0.0, abs=1e-6)
    assert o.centroid[1] == pytest.approx(0.0, abs=1e-6)
    assert o.centroid[2] == pytest.approx(1.0, abs=1e-6)


def test_multiview_index_skips_malformed_view() -> None:
    w, h = 640, 480
    views = [
        {"detections": None, "depth_provider": None, "camera": None},  # malformed -> skipped
        {
            "detections": [_centered_detection(w, h)],
            "depth_provider": lambda px, py: 3.0,
            "camera": {"eye": (0.0, -3.0, 1.0), "target": (0.0, 0.0, 1.0),
                       "vfov": 1.0, "width": w, "height": h},
        },
    ]
    objs = MultiViewPerceptionSceneSpatialIndex(views, min_views=1).objects()
    assert len(objs) == 1                            # the one good view still contributes
    assert math.isclose(objs[0].centroid[2], 1.0, abs_tol=1e-6)


# --------------- regression: non-transitive relation must not chain a whole row ---------------

def test_distinct_same_label_row_does_not_collapse_via_iou_chain() -> None:
    # Three distinct burners in a line. Adjacent boxes overlap at exactly merge_iou, but the
    # END pair has IoU 0. Single-linkage union-find used to collapse all three into ONE object
    # centered on the MIDDLE burner — silently deleting the two outer burners. Complete-linkage
    # + spread cap must NOT do that: the row must not become a single fused object, and no
    # object's location may be silently dropped.
    burners = [_obj("burner", (x, 0.0, 0.0), size=(1.0, 1.0, 1.0)) for x in (0.0, 0.6, 1.2)]
    fused = fuse_scene_objects(burners, merge_iou=0.25, merge_gap=None)
    # The row must NOT collapse to a single object centered on the middle burner with the two
    # ends deleted (the old union-find bug). The end pair (IoU 0) is never co-clustered, so no
    # cluster can reach all three.
    assert len(fused) >= 2
    assert all(o.extra["n_views"] <= 2 for o in fused)
    # the rightmost burner is never absorbed by the left pair, so x=1.2 is still represented
    assert max(o.centroid[0] for o in fused) == pytest.approx(1.2)
    # and the total membership is conserved — no detection silently vanishes
    assert sum(o.extra["n_views"] for o in fused) == 3


def test_distinct_same_label_row_does_not_chain_via_merge_gap() -> None:
    # A row of evenly spaced knives: each within merge_gap of its neighbour but NOT of the
    # endpoints. The non-transitive merge_gap relation must not chain the whole row into one.
    knives = [_obj("knife", (x, 0.0, 0.0), size=(0.2, 0.2, 0.2)) for x in (0.0, 0.25, 0.50)]
    fused = fuse_scene_objects(knives, merge_iou=0.9, merge_gap=0.3, min_views=1)
    assert len(fused) == 3                        # each knife stays its own object


def test_long_uniform_row_stays_separate_under_merge_gap() -> None:
    # Five chairs in a 2m line, each 0.5m from its neighbour (< merge_gap) but endpoints 2m
    # apart. The whole row must NOT fuse into one center-of-row object with n_views=5.
    chairs = [_obj("chair", (x, 0.0, 0.5), size=(0.2, 0.2, 0.2)) for x in (0.0, 0.5, 1.0, 1.5, 2.0)]
    fused = fuse_scene_objects(chairs, merge_gap=0.6, merge_iou=0.99, min_views=1)
    assert len(fused) == 5
    assert all(o.extra["n_views"] == 1 for o in fused)


def test_spread_cap_bounds_cluster_to_one_object_extent() -> None:
    # Same row, but with an explicit fixed max_spread that is smaller than the spacing: even
    # the adjacent borderline overlap can't accrete the row.
    objs = [_obj("burner", (x, 0.0, 0.0), size=(1.0, 1.0, 1.0)) for x in (0.0, 0.6, 1.2)]
    fused = fuse_scene_objects(objs, merge_iou=0.25, max_spread=0.3)
    assert len(fused) == 3                        # each burner isolated by the tight spread cap


# --------------- regression: containment coverage (a tight box inside a loose box) -------------

def test_aabb_coverage_full_containment_and_degenerate() -> None:
    # tight box fully inside a loose box -> coverage 1.0 even though symmetric IoU is small
    assert aabb_coverage((0.25, 0.25, 0.25), (0.75, 0.75, 0.75), (0, 0, 0), (1, 1, 1)) == pytest.approx(1.0)
    # the symmetric IoU of that same pair is only 0.125 -> below the default merge_iou 0.25
    assert aabb_iou((0.25, 0.25, 0.25), (0.75, 0.75, 0.75), (0, 0, 0), (1, 1, 1)) == pytest.approx(0.125)
    # a degenerate (flat) box has zero volume -> coverage 0 (this is merge_gap's territory)
    assert aabb_coverage((0, 0, 0), (1, 1, 0), (0, 0, 0), (1, 1, 1)) == 0.0


def test_fuse_merges_contained_box_of_same_label_via_coverage() -> None:
    # One view sees the whole appliance (loose box), another a tight crop fully inside it.
    # IoU alone (0.125 < 0.25) would keep them separate; coverage (==1) merges them.
    loose = _obj("oven", (0.5, 0.5, 0.5), size=(1.0, 1.0, 1.0))
    tight = _obj("oven", (0.5, 0.5, 0.5), size=(0.5, 0.5, 0.5))
    fused = fuse_scene_objects([loose, tight], merge_iou=0.25, merge_gap=None)
    assert len(fused) == 1
    assert fused[0].extra["n_views"] == 2


# --------------- regression: even-count median is NOT outlier-robust (docstring truth) --------

def test_even_count_median_degenerates_toward_mean() -> None:
    # Four detections forced into one cluster: two good (~2.5) and two bad-depth (~3.3). With an
    # EVEN member count the median averages the two middle elements, so it equals the mean and
    # is dragged by the bad views. This pins the documented degeneration (the docstring no longer
    # claims median resists outliers at even counts).
    members = [_obj("faucet", (x, 0.0, 0.0), size=(0.2, 0.2, 0.2)) for x in (2.50, 2.52, 3.28, 3.30)]
    # max_spread admits the wide cluster so we exercise the merge math, not the spread cap.
    fused = fuse_scene_objects(members, merge_iou=0.99, merge_gap=1.5, min_views=1, max_spread=2.0)
    assert len(fused) == 1
    mean_x = (2.50 + 2.52 + 3.28 + 3.30) / 4.0
    assert fused[0].centroid[0] == pytest.approx(mean_x)   # median == mean at even count


def test_two_member_median_equals_mean() -> None:
    # A 2-member cluster has zero outlier robustness: median == mean exactly.
    members = [_obj("faucet", (2.5, 0.0, 0.0), size=(0.2, 0.2, 0.2)),
               _obj("faucet", (3.5, 0.0, 0.0), size=(0.2, 0.2, 0.2))]
    fused = fuse_scene_objects(members, merge_iou=0.99, merge_gap=1.5, min_views=1, max_spread=2.0)
    assert len(fused) == 1
    assert fused[0].centroid[0] == pytest.approx(3.0)      # == mean of {2.5, 3.5}


# --------------- regression: min_views>=2 silently drops genuinely single-view objects ---------

def test_min_views_two_drops_single_view_real_object() -> None:
    # A real object only one camera could see is dropped under min_views=2 (documented tradeoff).
    only = _obj("faucet", (2.5, 1.15, 1.0))
    assert fuse_scene_objects([only], min_views=2) == []
    # min_views=1 keeps it (no false negative for single-view-only objects)
    assert len(fuse_scene_objects([only], min_views=1)) == 1


# --------------- regression: degenerate / edge inputs -----------------------------------------

def test_fuse_empty_input_returns_empty() -> None:
    assert fuse_scene_objects([]) == []


def test_fuse_confidence_tie_picks_first_seen_member_for_identity() -> None:
    # Two views of one object with EQUAL confidence: max() is stable and keeps the first-seen
    # member's id/label, so identity selection is deterministic across runs.
    a = _obj("faucet", (2.50, 1.15, 1.0), conf=0.9, oid="view_a")
    b = _obj("faucet", (2.51, 1.15, 1.0), conf=0.9, oid="view_b")
    fused = fuse_scene_objects([a, b], merge_iou=0.2)
    assert len(fused) == 1
    assert fused[0].id == "view_a"                          # first-seen wins on a tie
    # order independence of the winner is explicit: reversing inputs flips the deterministic pick
    fused_rev = fuse_scene_objects([b, a], merge_iou=0.2)
    assert fused_rev[0].id == "view_b"


def test_fused_size_is_per_axis_median_composed_box() -> None:
    # Median SIZE is taken independently per axis, so the fused extents can match NO single input
    # view (a composed box). This pins that documented behaviour.
    members = [
        _obj("box", (0, 0, 0), size=(0.4, 0.6, 0.3)),
        _obj("box", (0, 0, 0), size=(0.6, 0.4, 0.3)),
        _obj("box", (0, 0, 0), size=(0.5, 0.5, 0.9)),
    ]
    fused = fuse_scene_objects(members, merge_iou=0.05, min_views=1)
    assert len(fused) == 1
    sx, sy, sz = fused[0].size()
    assert (sx, sy, sz) == pytest.approx((0.5, 0.5, 0.3))
    # no input view reported (0.5, 0.5, 0.3)
    assert all(o.size() != (0.5, 0.5, 0.3) for o in members)
