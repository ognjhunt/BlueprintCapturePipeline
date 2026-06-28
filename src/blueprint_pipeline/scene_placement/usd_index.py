"""USD-backed spatial index: walk a USD stage -> meaningful ``SceneObject`` AABBs.

This is the "scene already exists as USD" path of the placement pipeline. Given a
stage (or a ``.usd``/``.usda`` path), it enumerates the meaningful named objects —
the faucet, sink, stove, fridge — and emits a world-aligned bounding box per
*top-level named object*, NOT per sub-mesh, so a "sink" assembly of several meshes
is one ``SceneObject`` the task can target.

Design (WHY): ``pxr`` (USD) is a heavy, GPU-adjacent dependency that must not be
imported to unit-test placement. So the only place ``pxr`` is touched is the thin
``_walk_stage`` wrapper; all the *decisions* — what counts as a shell to exclude,
how a prim name becomes a clean label, and how a list of named bounds becomes
``SceneObject``s — are factored into PURE helpers (``_is_excluded``,
``_clean_label``, ``_objects_from_bounds``) that tests drive with synthetic bounds
and no ``pxr`` at all. ``objects()`` just glues the lazy walk to the pure builder.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from .types import SceneObject, Vec3

# Default substrings whose presence in a prim name marks it as scene *shell* — the
# room envelope and stagecraft (walls, floor, lights, cameras), not a manipulable
# object the robot would act on. Kept lowercase; matched case-insensitively.
DEFAULT_EXCLUDE_SUBSTRINGS: Tuple[str, ...] = (
    "wall",
    "floor",
    "ceiling",
    "ground",
    "light",
    "camera",
    "dome",
    "room",
    # Structural scaffolding / grouping prims that wrap the whole scene. Authored
    # stages (this repo included) nest everything under "/World" and "/World/Scene"
    # Xforms; emitting a bound for one of those collapses the entire scene into a
    # single bogus object. These are token-matched, so a real "World_Globe" object
    # (tokens ["world", "globe"]) is still dropped — acceptable since these names are
    # reserved for scaffolding in practice; rename such an object if it must survive.
    "world",
    "scene",
    "root",
    "stage",
    "env",
    "environment",
    "default",
)

# Trailing instance/index/link decorations we strip to recover a human label:
# "Faucet_01" -> "faucet", "Sink001" -> "sink", "stove_link" -> "stove",
# "Knob_geo" / "Handle_mesh" -> "knob" / "handle". Order does not matter because
# we apply them repeatedly until the name stops shrinking.
_LABEL_STRIP_SUFFIXES: Tuple[str, ...] = (
    "_link",
    "_geo",
    "_geom",
    "_mesh",
    "_prim",
    "_xform",
    "_grp",
    "_group",
)


def _clean_label(name: str) -> str:
    """Turn a raw USD prim name into a human label (PURE, no ``pxr``).

    WHY: prim names in authored stages carry instance/index/link noise
    ("Faucet_01", "Sink001", "Stove_link", "KitchenIsland_geo"). The resolver and
    label fallback match on plain words ("faucet", "sink"), so we normalize:
      * split CamelCase / snake_case style separators are flattened to lowercase,
      * known structural suffixes (_link, _geo, _mesh, ...) are removed,
      * trailing numeric / "_01" / "001" instance indices are removed,
      * surrounding separators are trimmed.
    We strip iteratively so stacked decorations ("Faucet_01_geo") fully reduce.
    """
    label = (name or "").strip()
    if not label:
        return ""
    # Normalize separators to a single underscore so suffix/index rules are simple.
    label = label.replace("-", "_").replace(".", "_")
    # Collapse runs of underscores; lowercasing happens at the end so suffix
    # matching (which is already lowercase) stays predictable.
    previous = None
    while previous != label:
        previous = label
        lowered = label.lower()
        # Drop a known structural suffix (e.g. "_link", "_geo").
        for suffix in _LABEL_STRIP_SUFFIXES:
            if lowered.endswith(suffix) and len(label) > len(suffix):
                label = label[: -len(suffix)]
                break
        else:
            # No structural suffix matched; strip a trailing instance index, e.g.
            # "_01", "001", "2". A name that is ONLY an index (e.g. "01") reduces to
            # "" on purpose — it carries no human meaning and is later skipped.
            stripped = re.sub(r"[_]?\d+$", "", label)
            if stripped != label:
                label = stripped
        label = label.strip("_ ")
    return label.lower()


# Split a prim name into lowercase word tokens on separators AND camelCase/digit
# boundaries: "EastWall" -> ["east", "wall"], "Wall_North" -> ["wall", "north"],
# "kitchen_floor" -> ["kitchen", "floor"]. Matching the exclude list on whole tokens
# (not raw substrings) is what stops "wall_clock"/"floor_lamp"/"mushroom" from being
# wrongly dropped as shell while still catching the real envelope prims.
_TOKEN_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Za-z])(?=[0-9])")


def _name_tokens(name: str) -> List[str]:
    """Lowercase word tokens of a prim name (separator + camelCase/digit aware)."""
    parts = _TOKEN_SPLIT_RE.split(name or "")
    return [p.lower() for p in parts if p]


def _is_excluded(name: str, subs: Sequence[str]) -> bool:
    """True when ``name`` looks like scene shell/stagecraft, not an object (PURE).

    A blank name is excluded (nothing meaningful to target). We tokenize the name
    (separator + camelCase + digit boundaries) and exclude only when a WHOLE token
    equals a shell word. WHY token equality, not raw substring: substrings of
    "wall"/"floor"/"light"/"ground"/"room" hide inside real manipulable objects
    ("wall_clock", "floor_lamp", "LightFixture", "GroundCoffee", "Mushroom",
    "Bedroom_Door"); matching whole tokens catches "EastWall"/"Wall_North"/
    "kitchen_floor" without swallowing those false positives.
    """
    if not (name or "").strip():
        return True
    shell = {sub.lower() for sub in subs if sub}
    if not shell:
        return False
    return any(token in shell for token in _name_tokens(name))


def _objects_from_bounds(
    named_bounds: Sequence[Tuple[str, Tuple[Vec3, Vec3]]],
    *,
    exclude_substrings: Sequence[str] = DEFAULT_EXCLUDE_SUBSTRINGS,
    source: str = "usd",
) -> List[SceneObject]:
    """Build ``SceneObject``s from ``[(name, (bbox_min, bbox_max)), ...]`` (PURE).

    This is the testable crux of the USD path: it takes already-computed world
    bounds (which in production come from ``pxr`` BBoxCache, but in tests come from
    synthetic tuples) and produces one ``SceneObject`` per *kept* named object.

    WHY a separate pure function: it lets the placement-relevant decisions — skip
    the shell, derive a clean label + a stable id, compute the centroid — be unit
    tested with zero USD. ``objects()`` only has to produce the ``named_bounds``.

    Duplicate ids are disambiguated with a numeric suffix so two "knob" prims do
    not collapse into one entry.
    """
    objects: List[SceneObject] = []
    used_ids: Dict[str, int] = {}
    taken_ids: set = set()
    for name, (bbox_min, bbox_max) in named_bounds:
        if _is_excluded(name, exclude_substrings):
            continue
        label = _clean_label(name)
        if not label:
            continue
        bmin = (float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2]))
        bmax = (float(bbox_max[0]), float(bbox_max[1]), float(bbox_max[2]))
        centroid: Vec3 = (
            0.5 * (bmin[0] + bmax[0]),
            0.5 * (bmin[1] + bmax[1]),
            0.5 * (bmin[2] + bmax[2]),
        )
        # Stable id from the label; disambiguate repeats deterministically as
        # "knob", "knob_1", "knob_2", ... . _clean_label strips any trailing
        # "_<int>", so no produced label can itself end in "_<int>" and a suffixed id
        # cannot collide with a real label — but we still bump past any already-taken
        # concrete id defensively, so uniqueness holds even if that invariant changes.
        base_id = label.replace(" ", "_")
        count = used_ids.get(base_id, 0)
        obj_id = base_id if count == 0 else f"{base_id}_{count}"
        while obj_id in taken_ids:
            count += 1
            obj_id = f"{base_id}_{count}"
        used_ids[base_id] = count + 1
        taken_ids.add(obj_id)
        objects.append(
            SceneObject(
                id=obj_id,
                label=label,
                bbox_min=bmin,
                bbox_max=bmax,
                centroid=centroid,
                source=source,
                # Carry the raw authored name so callers can trace provenance.
                extra={"usd_prim_name": name},
            )
        )
    return objects


class UsdSceneSpatialIndex:
    """Enumerate placement-relevant objects from a USD stage.

    Satisfies the ``SceneSpatialIndex`` protocol. Construct with an already-open
    ``stage`` OR a ``usd_path`` to open lazily; ``exclude_substrings`` tunes which
    prim names are treated as scene shell (walls/floor/lights/...) and skipped.

    The only USD-touching work happens inside ``objects()`` (lazy ``pxr`` import),
    so importing this module never pulls in ``pxr``/Isaac/GPU.
    """

    def __init__(
        self,
        stage: object = None,
        usd_path: Optional[str] = None,
        *,
        exclude_substrings: Sequence[str] = DEFAULT_EXCLUDE_SUBSTRINGS,
    ) -> None:
        if stage is None and not usd_path:
            raise ValueError("UsdSceneSpatialIndex requires either a stage or a usd_path")
        self._stage = stage
        self._usd_path = usd_path
        self._exclude_substrings = tuple(exclude_substrings)

    def objects(self) -> List[SceneObject]:
        """Walk the stage and return one ``SceneObject`` per top-level named object.

        Lazy-imports ``pxr`` (so the module imports without USD), opens the stage
        from ``usd_path`` if needed, collects ``(name, (min, max))`` world bounds
        for each meaningful prim, then defers all the decisions to the pure
        ``_objects_from_bounds``.
        """
        named_bounds = self._walk_stage()
        return _objects_from_bounds(
            named_bounds,
            exclude_substrings=self._exclude_substrings,
            source="usd",
        )

    # ------------------------------------------------------------------
    # The ONLY pxr-touching code. Everything above is pure + unit-tested.
    # ------------------------------------------------------------------
    def _walk_stage(self) -> List[Tuple[str, Tuple[Vec3, Vec3]]]:
        """Lazy ``pxr`` walk -> ``[(prim_name, (bbox_min, bbox_max)), ...]``.

        WHY one entry per top-level named object: authored objects are often an
        Xform with many child meshes (sink = basin + faucet-mount + drain). We walk
        ``Usd.PrimRange`` and emit a bound for a prim only when it is (a) Imageable,
        (b) NOT scene shell/scaffolding (``_is_excluded``), (c) labelable, AND (d) its
        subtree actually contains ``Gprim`` geometry — then ``PruneChildren`` so its
        sub-meshes are not emitted again. Requiring real geometry under the prim is
        what stops a bare grouping ``Xform`` like ``/World`` or ``/World/Scene`` (an
        ``Imageable`` that bounds the WHOLE scene) from being emitted as one bogus
        "world" object: we keep descending through such scaffolding until we reach the
        shallowest named prim that actually bounds geometry (the sink, the stove).
        Shell prims are pruned outright so their sub-geometry is never inspected. The
        world bound comes from ``UsdGeom.BBoxCache(...).ComputeWorldBound(prim)`` ->
        ``ComputeAlignedRange()`` which already accounts for child geometry.
        """
        from pxr import Usd, UsdGeom  # type: ignore  # lazy: heavy USD dep

        stage = self._stage
        if stage is None:
            stage = Usd.Stage.Open(self._usd_path)
        if stage is None:
            return []

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        )

        def _subtree_has_gprim(root_prim) -> bool:
            """True if ``root_prim`` or any descendant is concrete geometry (Gprim)."""
            for descendant in Usd.PrimRange(root_prim):
                if descendant.IsValid() and descendant.IsA(UsdGeom.Gprim):
                    return True
            return False

        named_bounds: List[Tuple[str, Tuple[Vec3, Vec3]]] = []
        prim_range = iter(Usd.PrimRange(stage.GetPseudoRoot()))
        for prim in prim_range:
            if not prim.IsValid():
                continue
            name = prim.GetName()
            # Shell ("Wall"/"Floor"/"Light") AND scaffolding ("World"/"Scene") by
            # name: skip WITHOUT pruning. Not pruning is critical — "/World" is an
            # ANCESTOR of the real objects, so pruning it would drop the whole scene.
            # Excluded leaf meshes (a wall) simply have no children to descend into,
            # so skipping-without-pruning is harmless for them too. (The pseudo-root
            # has an empty name and is not excluded, so we descend through it.)
            if name and _is_excluded(name, self._exclude_substrings):
                continue
            # Only geometry-bearing prims can have a meaningful world bound.
            if not prim.IsA(UsdGeom.Imageable):
                continue
            if not _clean_label(name):
                continue
            # A bare grouping Xform/Scope with NO geometry under it is scaffolding,
            # not an object — keep descending to find the real named objects beneath
            # it instead of emitting a scene-spanning bound here. This is the guard
            # that stops an un-excluded wrapper Xform from collapsing the whole scene.
            if not _subtree_has_gprim(prim):
                continue
            # Compute the world AABB for THIS prim subtree.
            try:
                bound = bbox_cache.ComputeWorldBound(prim)
                aligned = bound.ComputeAlignedRange()
                if aligned.IsEmpty():
                    continue
                rmin = aligned.GetMin()
                rmax = aligned.GetMax()
                bbox_min: Vec3 = (float(rmin[0]), float(rmin[1]), float(rmin[2]))
                bbox_max: Vec3 = (float(rmax[0]), float(rmax[1]), float(rmax[2]))
            except Exception:  # noqa: BLE001 - a malformed prim must not abort the walk
                continue
            named_bounds.append((name, (bbox_min, bbox_max)))
            # This named object is counted; do not descend into its sub-meshes,
            # so a multi-mesh "sink" stays a single SceneObject.
            prim_range.PruneChildren()
        return named_bounds
