"""Multi-view fusion for the perception backend: many views -> one box per object.

The single-view :class:`PerceptionSceneSpatialIndex` turns ONE render's 2D detections
+ depth into world AABBs. A single view is fragile: an object can be occluded, and a
noisy depth pixel throws the whole box meters off. The fix — the same one
``splat_analyzer`` uses (render a ring of views, detect each, lift to 3D, then *cluster
across views into one 3D box per object*) — is implemented here as a COMPOSABLE layer
over the existing single-view backend, not a rewrite.

Two pieces:
  * :func:`fuse_scene_objects` — PURE. Given world-space objects from any number of
    views, cluster the ones that are the same physical object (same label + overlapping
    3D box) and merge each cluster into one box. ``min_views >= 2`` is a TRADEOFF: it
    rejects single-view false positives and stray bad-depth outliers (which fail to cluster
    and stand alone), but it ALSO silently drops a genuinely single-view-visible real object
    (one only one camera in the ring could see) — false positives traded against false
    negatives. ``min_views = 1`` keeps everything.
  * :class:`MultiViewPerceptionSceneSpatialIndex` — runs the single-view backend on each
    view and feeds the union through :func:`fuse_scene_objects`. Satisfies the
    ``SceneSpatialIndex`` protocol, so task->target->placement runs on it unchanged.

Robustness choices (why):
  * Merge geometry uses the component-wise MEDIAN centroid + MEDIAN size across a cluster,
    not the mean or the union. The median resists a single off view ONLY when the genuine
    views are a strict majority AND the member count is ODD — then the middle element is a
    real agreeing view. With an EVEN count (common: a ring renders 4/6/8 views) ``_median``
    averages the two middle elements, so it degenerates toward the mean and can be dragged
    just as far; a 2-member cluster makes the median exactly the mean. Use ``min_views`` and
    an even/odd-aware reading of this estimate accordingly. The union would always let one
    view inflate the box, which is why we never use it. A far outlier never clusters in the
    first place (IoU 0, gap large) — it becomes its own singleton and is dropped by
    ``min_views >= 2``.
  * Clustering matches on IoU (position-tolerant) OR a coverage/containment test (a tight box
    fully inside a loose box of the same label — a common multi-view artifact whose symmetric
    IoU is small) OR an optional centroid ``merge_gap`` (for thin/flat objects whose true
    boxes barely overlap). The same-object relation is NOT transitive, so clustering uses
    COMPLETE-LINKAGE: every pair of members in a cluster must satisfy the relation, not just a
    connecting chain. This is what keeps a near-uniform ROW of distinct same-label objects
    (burners in a line, a row of chairs) from collapsing into one — under single-linkage a
    chain where each neighbour overlaps but the endpoints don't would wrongly merge. Distinct
    same-label objects that are spatially apart never share a cluster.

Pure + dependency-free (stdlib only) so it unit-tests with synthetic boxes — no torch,
no GPU, no SAM3/DA3, no network.
"""
from __future__ import annotations

from collections import defaultdict
from typing import List, Mapping, Optional, Sequence, Tuple

from .perception_index import PerceptionSceneSpatialIndex
from .types import SceneObject, Vec3


def _aabb_volume(bmin: Vec3, bmax: Vec3) -> float:
    """Volume of an axis-aligned box; 0 if any axis is non-positive (degenerate/flat)."""
    v = 1.0
    for i in range(3):
        d = bmax[i] - bmin[i]
        if d <= 0.0:
            return 0.0
        v *= d
    return v


def aabb_iou(a_min: Vec3, a_max: Vec3, b_min: Vec3, b_max: Vec3) -> float:
    """3D intersection-over-union of two axis-aligned boxes (0 when disjoint/degenerate).

    Used as the same-object test across views: two views of one faucet overlap heavily
    (IoU near 1), two different objects don't overlap (IoU 0). A flat box (zero volume on
    some axis) yields IoU 0 here — that is what ``merge_gap`` exists to backstop.
    """
    lo = [max(a_min[i], b_min[i]) for i in range(3)]
    hi = [min(a_max[i], b_max[i]) for i in range(3)]
    inter = 1.0
    for i in range(3):
        d = hi[i] - lo[i]
        if d <= 0.0:
            return 0.0
        inter *= d
    va = _aabb_volume(a_min, a_max)
    vb = _aabb_volume(b_min, b_max)
    union = va + vb - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _aabb_intersection_volume(a_min: Vec3, a_max: Vec3, b_min: Vec3, b_max: Vec3) -> float:
    """Volume of the overlap of two AABBs (0 when disjoint)."""
    inter = 1.0
    for i in range(3):
        d = min(a_max[i], b_max[i]) - max(a_min[i], b_min[i])
        if d <= 0.0:
            return 0.0
        inter *= d
    return inter


def aabb_coverage(a_min: Vec3, a_max: Vec3, b_min: Vec3, b_max: Vec3) -> float:
    """Asymmetric overlap: intersection / volume of the SMALLER box (0 if either is flat).

    Symmetric IoU under-counts full CONTAINMENT — a tight crop of an appliance sitting wholly
    inside a coarse-view box of the same appliance has small IoU (vol_small/vol_union) yet is
    clearly the same object. Coverage is ~1 when the smaller box is (nearly) inside the larger,
    so it recovers that case. Returns 0 when either box is degenerate (zero-volume), which is
    exactly the thin/flat case ``merge_gap`` exists to backstop.
    """
    inter = _aabb_intersection_volume(a_min, a_max, b_min, b_max)
    if inter <= 0.0:
        return 0.0
    va = _aabb_volume(a_min, a_max)
    vb = _aabb_volume(b_min, b_max)
    smaller = min(va, vb)
    if smaller <= 0.0:
        return 0.0
    return inter / smaller


def _centroid_distance(a: SceneObject, b: SceneObject) -> float:
    dx = a.centroid[0] - b.centroid[0]
    dy = a.centroid[1] - b.centroid[1]
    dz = a.centroid[2] - b.centroid[2]
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _same_object(
    a: SceneObject,
    b: SceneObject,
    *,
    merge_iou: float,
    merge_gap: Optional[float],
) -> bool:
    """True if two same-label detections are the same physical object across views.

    Three OR-ed tests, weakest binding last:
      * symmetric 3D IoU >= ``merge_iou`` — two views of one object overlap heavily;
      * coverage >= ``merge_iou`` — a tight box fully inside a loose box of the same label
        (containment), which symmetric IoU under-counts;
      * centroid distance <= ``merge_gap`` (when set) — thin/flat boxes with ~zero IoU but
        coincident centres (orthogonal views of a faucet handle).
    """
    if aabb_iou(a.bbox_min, a.bbox_max, b.bbox_min, b.bbox_max) >= merge_iou:
        return True
    if aabb_coverage(a.bbox_min, a.bbox_max, b.bbox_min, b.bbox_max) >= merge_iou:
        return True
    if merge_gap is not None and _centroid_distance(a, b) <= merge_gap:
        return True
    return False


def _within_spread_cap(
    cluster: Sequence[SceneObject],
    cand: SceneObject,
    max_spread: Optional[float],
) -> bool:
    """True if adding ``cand`` keeps the cluster's centroid spread within a plausible cap.

    A single physical object's multi-view detections have ~coincident centroids — their spread
    stays inside the object's own footprint. A near-uniform ROW of distinct same-label objects,
    by contrast, has a centroid spread that grows with the row length. Capping the per-axis
    centroid spread therefore bounds a cluster to one object's extent and stops a chain of
    borderline-overlapping neighbours from accreting the whole row.

    When ``max_spread`` is ``None`` the cap is derived per axis as the LARGEST member extent on
    that axis (a true detection cluster fits within its own box); otherwise it is the given
    fixed distance applied to every axis.
    """
    members = list(cluster) + [cand]
    for ax in range(3):
        coords = [m.centroid[ax] for m in members]
        spread = max(coords) - min(coords)
        if max_spread is not None:
            cap = max_spread
        else:
            cap = max(m.size()[ax] for m in members)
        if spread > cap + 1e-9:
            return False
    return True


def _complete_linkage_clusters(
    objs: Sequence[SceneObject],
    *,
    merge_iou: float,
    merge_gap: Optional[float],
    max_spread: Optional[float] = None,
) -> List[List[SceneObject]]:
    """Complete-linkage clustering, spread-capped, over the same-object relation (one label).

    The same-object relation is NOT transitive: A~B and B~C does not imply A~C. Single-linkage
    (transitive closure / union-find) would therefore wrongly collapse a near-uniform ROW of
    distinct same-label objects — each neighbour overlaps but the endpoints don't — into one
    fused object centred on the middle instance, silently deleting the ends. Two gates prevent
    that:
      * COMPLETE-LINKAGE — a detection joins a cluster only if it satisfies :func:`_same_object`
        against EVERY current member, not just one neighbour, so non-adjacent members
        (IoU 0) can never be co-clustered through a chain.
      * SPREAD CAP — :func:`_within_spread_cap` additionally rejects a join that would push the
        cluster's centroid spread beyond one object's plausible extent, so even adjacent
        borderline overlaps cannot accrete an unbounded row.

    Greedy and order-deterministic: objects are considered in input order; each either joins
    the first existing cluster it is mutually same-object with (against all members, within the
    spread cap) or starts a new cluster. O(n^2) is fine — a single label rarely has more than a
    handful of instances across the view ring. Members keep input order; clusters come out in
    first-seen order, so the fused output is stable across runs.
    """
    clusters: List[List[SceneObject]] = []
    for obj in objs:
        placed = False
        for cluster in clusters:
            if _within_spread_cap(cluster, obj, max_spread) and all(
                _same_object(obj, member, merge_iou=merge_iou, merge_gap=merge_gap)
                for member in cluster
            ):
                cluster.append(obj)
                placed = True
                break
        if not placed:
            clusters.append([obj])
    return clusters


def _median(vals: Sequence[float]) -> float:
    """Component-wise median.

    NOTE the even-count behaviour: with an even number of values this averages the two middle
    elements, so it is NOT outlier-robust at even counts (and a 2-element median equals the
    mean exactly). Callers relying on outlier resistance must ensure a strict-majority/odd
    member count — see the module docstring.
    """
    s = sorted(vals)
    m = len(s) // 2
    if len(s) % 2:
        return s[m]
    return 0.5 * (s[m - 1] + s[m])


def _merge_cluster(members: Sequence[SceneObject]) -> SceneObject:
    """Collapse a cluster of same-object detections into one box.

    Median centroid + median per-axis size, re-centered into a clean AABB. The median is taken
    INDEPENDENTLY per axis, so the fused extents can match no single input view (a composed box
    — e.g. members (0.4,0.6,_),(0.6,0.4,_) fuse to (0.5,0.5,_)). The median is outlier-resistant
    only for an odd member count with a strict-majority of agreeing views; at even counts it
    degenerates toward the mean (see ``_median`` / the module docstring). The id/label/category
    come from the highest-confidence member (the view that saw it best); the fused confidence is
    the mean across the cluster. ``extra`` records the provenance so downstream can see how many
    views agreed.
    """
    cents = [o.centroid for o in members]
    sizes = [o.size() for o in members]
    cx = _median([c[0] for c in cents])
    cy = _median([c[1] for c in cents])
    cz = _median([c[2] for c in cents])
    sx = _median([s[0] for s in sizes])
    sy = _median([s[1] for s in sizes])
    sz = _median([s[2] for s in sizes])
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    bbox_min = (cx - hx, cy - hy, cz - hz)
    bbox_max = (cx + hx, cy + hy, cz + hz)
    best = max(members, key=lambda o: o.confidence)
    conf = sum(o.confidence for o in members) / float(len(members))
    return SceneObject(
        id=best.id,
        label=best.label,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        centroid=(cx, cy, cz),
        category=best.category,
        source="perception_fused",
        confidence=conf,
        extra={
            "n_views": len(members),
            "member_ids": [o.id for o in members],
            "member_confidences": [round(float(o.confidence), 6) for o in members],
        },
    )


def fuse_scene_objects(
    objects: Sequence[SceneObject],
    *,
    merge_iou: float = 0.25,
    merge_gap: Optional[float] = None,
    min_views: int = 1,
    max_spread: Optional[float] = None,
) -> List[SceneObject]:
    """Cluster same-object detections across views into one box each.

    Clustering is COMPLETE-LINKAGE on the (non-transitive) same-object relation — every pair in
    a cluster must satisfy it — PLUS a centroid-spread cap, so a near-uniform row of distinct
    same-label objects can never chain (or accrete) into one. See the module docstring for the
    robustness caveats on the merged box.

    Args:
        objects: world-space detections pooled from every view (any backend/source).
        merge_iou: minimum 3D IoU (or containment coverage) for two same-label boxes to be the
            same object.
        merge_gap: optional centroid-distance (m) that ALSO merges same-label boxes —
            backstops thin/flat objects whose true boxes barely overlap (IoU ~0). ``None`` =
            IoU/coverage only.
        min_views: drop fused objects supported by fewer than this many detections. This is a
            TRADEOFF, not pure upside: ``2+`` rejects single-view false positives and stray
            bad-depth outliers, but ALSO silently drops a genuinely single-view-visible real
            object (one only one camera in the ring could see, occluded from the rest). ``1``
            keeps everything (no false negatives for objects only one view could see).
        max_spread: optional cap (m) on a cluster's per-axis centroid spread. ``None`` (default)
            derives the cap per axis from the largest member extent on that axis — a true
            object's multi-view centroids fall within its own box, so this bounds a cluster to
            one object without a magic number. A fixed value applies the same cap to all axes.

    Returns objects sorted by descending confidence (then label, then centroid) for a
    stable, useful ordering. Spatially distinct same-label objects stay separate.
    """
    by_label: "defaultdict[str, List[SceneObject]]" = defaultdict(list)
    for o in objects:
        by_label[o.label.strip().lower()].append(o)

    fused: List[SceneObject] = []
    for _label, objs in by_label.items():
        clusters = _complete_linkage_clusters(
            objs, merge_iou=merge_iou, merge_gap=merge_gap, max_spread=max_spread
        )
        for cluster in clusters:
            if len(cluster) < min_views:
                continue
            fused.append(_merge_cluster(cluster))

    fused.sort(key=lambda o: (-o.confidence, o.label, o.centroid))
    return fused


class MultiViewPerceptionSceneSpatialIndex:
    """Perception spatial index that fuses detections across MANY views into one box/object.

    Each view is ``(detections, depth_provider, camera)`` — the exact inputs the single-view
    :class:`PerceptionSceneSpatialIndex` takes. :meth:`objects` runs that backend per view and
    pools the results through :func:`fuse_scene_objects`, so the rest of the pipeline
    (resolve_target -> compute_stand_pose) consumes a clean, deduplicated 3D catalog.

    Satisfies the ``SceneSpatialIndex`` protocol. Pure given the injected depth providers —
    no GPU/SAM3/DA3 here; perception is upstream, per view.
    """

    def __init__(
        self,
        views: Sequence[Mapping[str, object]],
        *,
        merge_iou: float = 0.25,
        merge_gap: Optional[float] = None,
        min_views: int = 1,
        max_spread: Optional[float] = None,
        samples_per_axis: int = 3,
    ) -> None:
        """
        Args:
            views: ``[{detections, depth_provider, camera}]`` — one entry per rendered view.
                Each ``camera`` carries that view's intrinsics + look-at extrinsics, so a ring
                of cameras around the scene is just a list of differing cameras.
            merge_iou / merge_gap / min_views / max_spread: forwarded to
                :func:`fuse_scene_objects`.
            samples_per_axis: depth-sampling density forwarded to each single-view backend.
        """
        self._views = list(views)
        self._merge_iou = merge_iou
        self._merge_gap = merge_gap
        self._min_views = min_views
        self._max_spread = max_spread
        self._samples_per_axis = samples_per_axis

    def _view_field(self, view: Mapping[str, object], name: str):
        if isinstance(view, Mapping):
            return view.get(name)
        return getattr(view, name, None)

    def objects(self) -> List[SceneObject]:
        pooled: List[SceneObject] = []
        for view in self._views:
            detections = self._view_field(view, "detections")
            depth_provider = self._view_field(view, "depth_provider")
            camera = self._view_field(view, "camera")
            if detections is None or depth_provider is None or camera is None:
                # A malformed view is skipped, not fatal — other views still contribute.
                continue
            single = PerceptionSceneSpatialIndex(
                detections,            # type: ignore[arg-type]
                depth_provider,        # type: ignore[arg-type]
                camera,                # type: ignore[arg-type]
                samples_per_axis=self._samples_per_axis,
            )
            pooled.extend(single.objects())
        return fuse_scene_objects(
            pooled,
            merge_iou=self._merge_iou,
            merge_gap=self._merge_gap,
            min_views=self._min_views,
            max_spread=self._max_spread,
        )


__all__ = [
    "aabb_iou",
    "aabb_coverage",
    "fuse_scene_objects",
    "MultiViewPerceptionSceneSpatialIndex",
]
