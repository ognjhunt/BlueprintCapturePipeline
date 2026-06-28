"""View-ring planner: where to put cameras so multi-view perception sees the whole scene.

The perception path is *N views -> detect + depth per view -> fuse*. This module owns the
FIRST stage's geometry: given a scene (or object) bounding box, produce a ring (or stacked
rings) of look-at cameras that orbit it — the camera specs the renderer drives and that
:class:`PerceptionSceneSpatialIndex` / :class:`MultiViewPerceptionSceneSpatialIndex` consume
verbatim. It is the simple, deterministic stand-in for ``splat_analyzer``'s "density-aware
sampler": uniform azimuths at one or more elevations around the bounds, all aimed at the centre.

Why a ring of look-at cameras: multi-view fusion only pays off when the same object is seen
from genuinely different angles (so occlusions in one view are filled by another and depth
noise averages out). Orbiting the bounds at a radius derived from the box diagonal guarantees
each object is framed from several bearings without any scene-specific tuning.

Pure + stdlib-only (``math``), so it unit-tests with synthetic bounds — no GPU, no renderer,
no network. Output cameras use the same schema the perception backend already accepts:
``{eye, target, up, vfov, width, height}`` (``vfov`` in RADIANS, matching ``resolve_intrinsics``).
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .types import Vec3

Camera = Dict[str, object]


def _bounds_center_radius(bbox_min: Vec3, bbox_max: Vec3, *, margin: float) -> Tuple[Vec3, float]:
    """Centre of the AABB and an orbit radius = (half-diagonal) * margin.

    Half-diagonal (not half-width) so the radius encloses the box from every bearing; ``margin``
    (>1) backs the camera off so the whole object stays in frame. Returns a small positive radius
    for a degenerate (zero-size) box so callers still get a usable orbit.
    """
    center = (
        0.5 * (bbox_min[0] + bbox_max[0]),
        0.5 * (bbox_min[1] + bbox_max[1]),
        0.5 * (bbox_min[2] + bbox_max[2]),
    )
    dx = bbox_max[0] - bbox_min[0]
    dy = bbox_max[1] - bbox_min[1]
    dz = bbox_max[2] - bbox_min[2]
    half_diag = 0.5 * math.sqrt(dx * dx + dy * dy + dz * dz)
    radius = max(half_diag * float(margin), 1e-3)
    return center, radius


def generate_view_ring(
    center: Vec3,
    radius: float,
    *,
    n_azimuths: int = 8,
    elevations_deg: Sequence[float] = (20.0,),
    vfov_deg: float = 60.0,
    width: int = 1280,
    height: int = 960,
    up: Vec3 = (0.0, 0.0, 1.0),
    azimuth_offset_deg: float = 0.0,
) -> List[Camera]:
    """Look-at cameras orbiting ``center`` at ``radius``, all aimed at the centre.

    Cameras are placed at every (azimuth, elevation) pair: ``n_azimuths`` evenly spaced bearings
    around z-up, repeated at each elevation in ``elevations_deg`` (degrees above the horizon, so a
    positive elevation looks DOWN at the scene). Total cameras = ``n_azimuths * len(elevations_deg)``.

    The eye for azimuth ``θ`` / elevation ``φ`` is
    ``center + radius * (cosφ cosθ, cosφ sinθ, sinφ)`` and the target is ``center``. ``vfov_deg`` is
    converted to radians in the output (the backend's intrinsics expect radians). ``up`` is world-up
    by default; the backend re-orthonormalises it and handles the straight-overhead degenerate case.
    """
    if n_azimuths < 1:
        raise ValueError("n_azimuths must be >= 1")
    if radius <= 0.0:
        raise ValueError("radius must be > 0")
    vfov_rad = math.radians(float(vfov_deg))
    off = math.radians(float(azimuth_offset_deg))
    cams: List[Camera] = []
    for elev_deg in elevations_deg:
        phi = math.radians(float(elev_deg))
        cphi, sphi = math.cos(phi), math.sin(phi)
        for k in range(n_azimuths):
            theta = off + 2.0 * math.pi * k / n_azimuths
            eye = (
                center[0] + radius * cphi * math.cos(theta),
                center[1] + radius * cphi * math.sin(theta),
                center[2] + radius * sphi,
            )
            cams.append({
                "eye": eye,
                "target": (center[0], center[1], center[2]),
                "up": (float(up[0]), float(up[1]), float(up[2])),
                "vfov": vfov_rad,
                "width": int(width),
                "height": int(height),
            })
    return cams


def view_ring_for_bounds(
    bbox_min: Vec3,
    bbox_max: Vec3,
    *,
    margin: float = 1.6,
    n_azimuths: int = 8,
    elevations_deg: Sequence[float] = (20.0,),
    vfov_deg: float = 60.0,
    width: int = 1280,
    height: int = 960,
    up: Vec3 = (0.0, 0.0, 1.0),
    azimuth_offset_deg: float = 0.0,
) -> List[Camera]:
    """Camera ring sized to a scene/object AABB — no scene-specific coordinates.

    Derives the orbit centre + radius from the bounds (half-diagonal * ``margin``) and forwards to
    :func:`generate_view_ring`. This is the one call a caller needs: hand it the scene bounds (from
    a USD bbox or a splat's extent) and get back the cameras to render for multi-view perception.
    """
    center, radius = _bounds_center_radius(bbox_min, bbox_max, margin=margin)
    return generate_view_ring(
        center,
        radius,
        n_azimuths=n_azimuths,
        elevations_deg=elevations_deg,
        vfov_deg=vfov_deg,
        width=width,
        height=height,
        up=up,
        azimuth_offset_deg=azimuth_offset_deg,
    )


def assemble_views(
    cameras: Sequence[Camera],
    detections_per_view: Sequence[Sequence[Mapping[str, object]]],
    depth_providers: Sequence[Callable[[float, float], float]],
) -> List[Dict[str, object]]:
    """Zip cameras + per-view detections + per-view depth into the ``views`` list fusion wants.

    Each output entry is ``{detections, depth_provider, camera}`` — exactly what
    :class:`MultiViewPerceptionSceneSpatialIndex` iterates. The three inputs must be the same
    length (one per rendered view); mismatched lengths are a caller bug and raise rather than
    silently dropping views.
    """
    if not (len(cameras) == len(detections_per_view) == len(depth_providers)):
        raise ValueError(
            "cameras, detections_per_view, and depth_providers must have equal length "
            f"(got {len(cameras)}, {len(detections_per_view)}, {len(depth_providers)})"
        )
    return [
        {"detections": list(detections_per_view[i]), "depth_provider": depth_providers[i],
         "camera": cameras[i]}
        for i in range(len(cameras))
    ]


__all__ = [
    "generate_view_ring",
    "view_ring_for_bounds",
    "assemble_views",
]
