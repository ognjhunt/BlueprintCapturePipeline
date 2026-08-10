"""Lift a moulded-flush handle off its panel so a gripper can hook it.

Generated twins produce handles as solid blocks pressed against the door. They
look right, they render right, and they cannot be grasped the way the real
fitting they stand for is grasped: with no gap behind the bar there is nowhere
to put fingers, so the only available hold is a friction pinch. Against a
gasketed door that pinch is marginal, and when it slips the run records a
policy failure rather than a grasp that was never viable.

Real handles stand off the door on posts. That is what this authors: the bar
moves outward along the panel normal, and short posts at each end bridge the
gap it left. Both halves matter. A bar moved without posts is worse than the
flush block it replaced - it reads as graspable while being physically
unattached, so it passes the geometry check and then falls off under load. And
posts running the full length would rebuild the solid block, so they take a
fraction at each end and the span between them is reported, because that span
is the part a hand can actually reach.

A standoff smaller than a finger is refused rather than authored. Moving a
handle three millimetres accomplishes nothing except making the clearance check
pass, which is the failure this module exists to prevent.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Sequence

from .common import write_json
from .decision_evidence_contracts import canonical_digest


HANDLE_STANDOFF_SCHEMA_VERSION = "handle_standoff.v1"
DEFAULT_FINGER_CLEARANCE_M = 0.018
DEFAULT_POST_FRACTION = 0.12


class HandleStandoffError(ValueError):
    """Stable, sorted handle-standoff authoring failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def author_handle_standoff(
    *,
    source_usd_path: str | Path,
    destination: str | Path,
    handle_prim_paths: Sequence[str],
    panel_face_offset_m: float,
    outward_normal: Sequence[float],
    standoff_m: float,
    post_fraction: float = DEFAULT_POST_FRACTION,
    finger_clearance_m: float = DEFAULT_FINGER_CLEARANCE_M,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Move a handle outward and bridge the gap with end posts."""

    try:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
    except ImportError as exc:  # pragma: no cover - environment guard
        raise HandleStandoffError(["handle_standoff_openusd_runtime_missing"]) from exc

    source = Path(source_usd_path).expanduser().resolve()
    output = Path(destination).expanduser().resolve()
    if not source.is_file():
        raise HandleStandoffError(["handle_standoff_source_missing"])
    if output == source:
        raise HandleStandoffError(["handle_standoff_destination_is_source"])
    if not handle_prim_paths:
        raise HandleStandoffError(["handle_standoff_handle_paths_missing"])

    try:
        normal = [float(value) for value in outward_normal]
    except (TypeError, ValueError) as exc:
        raise HandleStandoffError(["handle_standoff_normal_invalid"]) from exc
    length = math.sqrt(sum(value * value for value in normal))
    if len(normal) != 3 or length <= 0.0:
        raise HandleStandoffError(["handle_standoff_normal_invalid"])
    normal = [value / length for value in normal]

    standoff = float(standoff_m)
    if standoff < float(finger_clearance_m):
        # Buying only the check, not the clearance, is the failure this guards.
        raise HandleStandoffError(
            [
                "handle_standoff_standoff_below_finger_clearance:"
                f"{standoff:.4f}<{float(finger_clearance_m):.4f}"
            ]
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    stage = Usd.Stage.Open(str(output))
    if stage is None:
        output.unlink(missing_ok=True)
        raise HandleStandoffError(["handle_standoff_source_unreadable"])

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    errors: list[str] = []
    moved: list[dict[str, Any]] = []
    low = [float("inf")] * 3
    high = [float("-inf")] * 3
    parent_path = None
    for path in handle_prim_paths:
        prim = stage.GetPrimAtPath(str(path))
        if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
            errors.append(f"handle_standoff_handle_prim_invalid:{path}")
            continue
        parent_path = prim.GetParent().GetPath()
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        if not points:
            errors.append(f"handle_standoff_handle_has_no_points:{path}")
            continue
        shifted = [
            Gf.Vec3f(*[point[i] + normal[i] * standoff for i in range(3)])
            for point in points
        ]
        mesh.GetPointsAttr().Set(shifted)
        bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        for index in range(3):
            low[index] = min(low[index], float(bounds.GetMin()[index]))
            high[index] = max(high[index], float(bounds.GetMax()[index]))
        moved.append({"prim_path": str(path), "shifted_m": standoff})
    if errors or parent_path is None:
        output.unlink(missing_ok=True)
        raise HandleStandoffError(errors or ["handle_standoff_no_handle_moved"])

    # The bar's long direction is whichever axis it spans furthest, ignoring
    # the one it was just pushed along.
    spans = [
        (high[index] - low[index]) * (1.0 - abs(normal[index])) for index in range(3)
    ]
    long_axis = spans.index(max(spans))
    bar_length = high[long_axis] - low[long_axis]
    post_length = max(bar_length * float(post_fraction), 1e-4)
    face = float(panel_face_offset_m)

    def _post(name: str, start: float) -> None:
        mesh = UsdGeom.Mesh.Define(stage, f"{parent_path}/{name}")
        corners_low = list(low)
        corners_high = list(high)
        corners_low[long_axis] = start
        corners_high[long_axis] = start + post_length
        for index in range(3):
            if abs(normal[index]) > 0.5:
                # Span from the panel face out to the bar's inner surface.
                corners_low[index] = min(face, low[index])
                corners_high[index] = max(face, low[index])
        mesh.CreatePointsAttr(
            [
                Gf.Vec3f(
                    corners_low[0] if a else corners_high[0],
                    corners_low[1] if b else corners_high[1],
                    corners_low[2] if c else corners_high[2],
                )
                for a in (1, 0)
                for b in (1, 0)
                for c in (1, 0)
            ]
        )
        mesh.CreateFaceVertexCountsAttr([4] * 6)
        mesh.CreateFaceVertexIndicesAttr(
            [0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3]
        )
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        # These posts live below a dynamic door link. PhysX rejects triangle-
        # mesh collision on a dynamic body and otherwise silently falls back
        # to a convex hull at runtime. Author the supported approximation
        # explicitly so the provider readback matches the USD contract.
        UsdPhysics.MeshCollisionAPI.Apply(
            mesh.GetPrim()
        ).CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)

    _post("handle_post_a", low[long_axis])
    _post("handle_post_b", high[long_axis] - post_length)
    stage.GetRootLayer().Save()

    clear_span = bar_length - 2.0 * post_length
    receipt: dict[str, Any] = {
        "schema_version": HANDLE_STANDOFF_SCHEMA_VERSION,
        "status": "handle_standoff_authored",
        "source_usd_path": str(source),
        "source_usd_sha256": _sha256(source),
        "standoff_usd_path": str(output),
        "standoff_usd_sha256": _sha256(output),
        "moved_handle_prims": moved,
        "achieved_clearance_m": standoff,
        "bar_length_m": bar_length,
        "post_length_m": post_length,
        "clear_span_m": clear_span,
        "post_prim_paths": [
            f"{parent_path}/handle_post_a",
            f"{parent_path}/handle_post_b",
        ],
        "claim_boundary": {
            "posts_are_authored_not_observed": True,
            "form_closure_is_geometric_not_a_grasp_proof": True,
            "link_mass_unchanged_by_this_edit": True,
            "post_collision_approximation": "convexHull",
        },
        "receipt_path": str(
            Path(receipt_path).expanduser().resolve()
            if receipt_path is not None
            else output.with_name(output.stem + "_standoff_receipt.json")
        ),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    write_json(Path(receipt["receipt_path"]), receipt)
    return json.loads(json.dumps(receipt))


__all__ = [
    "DEFAULT_FINGER_CLEARANCE_M",
    "DEFAULT_POST_FRACTION",
    "HANDLE_STANDOFF_SCHEMA_VERSION",
    "HandleStandoffError",
    "author_handle_standoff",
]
