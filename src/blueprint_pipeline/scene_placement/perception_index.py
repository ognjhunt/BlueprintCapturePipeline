"""Perception-backed spatial index: 2D detections + depth -> world AABBs.

This is the *dynamic*, no-USD path. Upstream perception (SAM3 for boxes, DA3 for
a metric depth map) is treated as INPUT: callers hand us a list of detections
``[{label, bbox_px, confidence}]``, an injectable ``depth_provider(px, py) -> meters``,
and a ``camera`` describing pinhole intrinsics + look-at extrinsics. We turn each
2D box into a world-space :class:`SceneObject` by unprojecting its corners and
center to 3D and taking the axis-aligned bounds.

The crux is the camera math, and it is kept PURE and dependency-free so it can be
unit-tested with synthetic numbers — NO torch, NO GPU, NO network, NO SAM3/DA3.
The three helpers below (:func:`camera_basis`, :func:`pixel_ray`, :func:`unproject`)
are the load-bearing geometry; everything else just samples depth and bounds.

Conventions (must match the unit tests and the rest of the pipeline):
  * World is right-handed, ``z`` up. Camera looks from ``eye`` toward ``target``.
  * ``forward = normalize(target - eye)`` is the optical axis (+z in camera space).
  * ``right`` points to +x in the image (rightward); ``up_cam`` points to +y
    (upward). Image pixel ``y`` grows DOWNWARD, so we flip it when building rays.
  * Intrinsics are pinhole ``(fx, fy, cx, cy)`` in pixels. ``vfov + width + height``
    is converted to square-pixel intrinsics, so a non-square image gets the right
    horizontal FOV for free (aspect handled).
  * ``depth`` is metric z-depth ALONG the optical axis (DA3 semantics): a pixel at
    depth ``d`` lands at ``eye + d*forward`` when it is on the principal axis. We
    scale the unnormalized camera-space ray (whose forward component is exactly 1)
    by ``d`` so the forward displacement is always ``d`` — true z-depth, not range.
"""
from __future__ import annotations

import math
from typing import Callable, List, Mapping, Sequence, Tuple

from .types import SceneObject, Vec3

# A depth lookup: pixel (px, py) -> metric depth in meters along the optical axis.
# Injected so tests pass a lambda and real runs pass a DA3 depth-map sampler.
DepthProvider = Callable[[float, float], float]


# ----------------------------- pure vector helpers -----------------------------
# Tiny stdlib-only vector ops. We deliberately avoid numpy so the module imports
# with zero heavy deps; the math here is small and runs once per detection corner.

def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a: Vec3) -> float:
    return math.sqrt(_dot(a, a))


def _normalize(a: Vec3) -> Vec3:
    n = _norm(a)
    if n <= 1e-12:
        # Degenerate input (zero-length direction). Return it unchanged rather than
        # dividing by ~0; callers building a basis guard against this upstream.
        return a
    return _scale(a, 1.0 / n)


# ----------------------------- intrinsics -----------------------------

def resolve_intrinsics(camera: Mapping[str, object]) -> Tuple[float, float, float, float]:
    """Return pinhole ``(fx, fy, cx, cy)`` in pixels from a camera mapping.

    Two accepted forms, mirroring the contract:
      * explicit ``fx, fy, cx, cy`` (optionally with ``width``/``height``), or
      * ``vfov`` (vertical field of view, radians) + ``width`` + ``height``.

    For the vfov form we assume square pixels: ``fy = (height/2)/tan(vfov/2)`` and
    ``fx = fy``. Square pixels mean a wider image simply yields a wider horizontal
    FOV — aspect ratio is handled implicitly and correctly, which is exactly what
    the aspect unit test checks.
    """
    def _get(name: str):
        if isinstance(camera, Mapping):
            return camera.get(name)
        return getattr(camera, name, None)

    fx = _get("fx")
    fy = _get("fy")
    cx = _get("cx")
    cy = _get("cy")
    if fx is not None and fy is not None and cx is not None and cy is not None:
        return float(fx), float(fy), float(cx), float(cy)

    vfov = _get("vfov")
    width = _get("width")
    height = _get("height")
    if vfov is None or width is None or height is None:
        raise ValueError(
            "camera must provide either (fx, fy, cx, cy) or (vfov, width, height)"
        )
    w = float(width)
    h = float(height)
    half = float(vfov) / 2.0
    t = math.tan(half)
    if abs(t) <= 1e-12:
        raise ValueError("vfov too small to derive a focal length")
    fy_v = (h / 2.0) / t
    fx_v = fy_v  # square pixels; horizontal FOV follows from width
    return fx_v, fy_v, w / 2.0, h / 2.0


def resolve_extrinsics(camera: Mapping[str, object]) -> Tuple[Vec3, Vec3, Vec3]:
    """Return ``(eye, target, up)`` world vectors from a camera mapping.

    ``up`` defaults to world-up ``(0, 0, 1)`` (z-up convention) when omitted, since
    a typical look-at camera only needs eye + target to be well defined.
    """
    def _get(name: str):
        if isinstance(camera, Mapping):
            return camera.get(name)
        return getattr(camera, name, None)

    eye = _get("eye")
    target = _get("target")
    up = _get("up")
    if eye is None or target is None:
        raise ValueError("camera must provide world 'eye' and 'target'")
    if up is None:
        up = (0.0, 0.0, 1.0)
    return (
        (float(eye[0]), float(eye[1]), float(eye[2])),
        (float(target[0]), float(target[1]), float(target[2])),
        (float(up[0]), float(up[1]), float(up[2])),
    )


# ----------------------------- PURE camera math (the crux) -----------------------------

def camera_basis(eye: Vec3, target: Vec3, up: Vec3) -> Tuple[Vec3, Vec3, Vec3]:
    """Orthonormal world basis ``(right, up_cam, forward)`` for a look-at camera.

    ``forward`` is the optical axis (eye -> target). ``right`` is the image +x
    direction; ``up_cam`` is the (re-orthonormalized) image +y direction. We build
    a proper right-handed frame so that a pixel ray expressed in this basis maps to
    a consistent world direction, and so an on-axis pixel maps to ``forward`` exactly.

    The supplied ``up`` only needs to be *roughly* up — it is re-orthogonalized, so
    a non-perpendicular hint still yields a clean basis (Gram-Schmidt style).
    """
    forward_raw = _sub(target, eye)
    if _norm(forward_raw) <= 1e-9:
        raise ValueError("camera eye and target must not be coincident")
    forward = _normalize(forward_raw)
    right = _cross(forward, up)
    if _norm(right) <= 1e-9:
        # up was (anti)parallel to forward (e.g. looking straight down with z-up);
        # pick an arbitrary perpendicular so the basis stays well defined.
        fallback = (1.0, 0.0, 0.0)
        if abs(forward[0]) > 0.9:
            fallback = (0.0, 1.0, 0.0)
        right = _cross(forward, fallback)
    right = _normalize(right)
    # Recover an orthonormal up from right x forward (right-handed: up = right x fwd
    # gives +y-up given right=+x and forward=+z in camera space).
    up_cam = _normalize(_cross(right, forward))
    return right, up_cam, forward


def pixel_ray(
    px: float,
    py: float,
    intrinsics: Tuple[float, float, float, float],
    basis: Tuple[Vec3, Vec3, Vec3],
) -> Vec3:
    """World-space direction of the ray through pixel ``(px, py)``.

    Returns the UNNORMALIZED camera ray rotated into world space: its component
    along ``forward`` is exactly 1, so scaling it by a z-depth ``d`` gives a
    displacement with forward-extent ``d`` (see :func:`unproject`). Image ``y``
    grows downward, so we flip it to align with the upward ``up_cam`` axis.
    """
    fx, fy, cx, cy = intrinsics
    right, up_cam, forward = basis
    x_c = (px - cx) / fx          # rightward offset in camera space
    y_c = (cy - py) / fy          # upward offset (flip image-down y)
    # dir = x_c*right + y_c*up + 1*forward  -> on-axis (px=cx, py=cy) this is forward.
    return _add(_add(_scale(right, x_c), _scale(up_cam, y_c)), forward)


def unproject(
    px: float,
    py: float,
    depth: float,
    eye: Vec3,
    intrinsics: Tuple[float, float, float, float],
    basis: Tuple[Vec3, Vec3, Vec3],
) -> Vec3:
    """Unproject pixel ``(px, py)`` at metric z-``depth`` to a world point.

    ``point = eye + depth * pixel_ray(...)``. Because the pixel ray's forward
    component is 1, ``depth`` is the distance ALONG the optical axis (DA3-style
    z-depth), not Euclidean range — so an on-axis pixel at depth ``d`` lands at
    exactly ``eye + d*forward``, which is the property the unprojection test pins.
    """
    ray = pixel_ray(px, py, intrinsics, basis)
    return _add(eye, _scale(ray, float(depth)))


# ----------------------------- depth sampling -----------------------------

def _sample_box_depth(
    bbox_px: Sequence[float],
    depth_provider: DepthProvider,
    *,
    samples_per_axis: int = 3,
) -> float:
    """Median metric depth across a grid of samples inside the detection box.

    Median (not mean) is robust to the depth map's edge/background outliers that
    sneak into a 2D box around an object. We sample an interior grid (corners +
    center + edge midpoints by default) and drop any non-finite / non-positive
    readings the provider returns.
    """
    x0, y0, x1, y1 = (float(v) for v in bbox_px)
    n = max(2, int(samples_per_axis))
    depths: List[float] = []
    for i in range(n):
        fx_t = i / (n - 1)
        sx = x0 + fx_t * (x1 - x0)
        for j in range(n):
            fy_t = j / (n - 1)
            sy = y0 + fy_t * (y1 - y0)
            d = float(depth_provider(sx, sy))
            if math.isfinite(d) and d > 0.0:
                depths.append(d)
    if not depths:
        raise ValueError("depth_provider returned no valid depths inside the bbox")
    depths.sort()
    mid = len(depths) // 2
    if len(depths) % 2:
        return depths[mid]
    return 0.5 * (depths[mid - 1] + depths[mid])


def _box_corner_pixels(bbox_px: Sequence[float]) -> List[Tuple[float, float]]:
    """The four corners + center of a 2D box, as pixel coords.

    Unprojecting these five points and bounding them gives a reasonable world AABB
    for the detection without needing a full mask: corners pin the extent, the
    center anchors the centroid/depth.
    """
    x0, y0, x1, y1 = (float(v) for v in bbox_px)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    return [(x0, y0), (x1, y0), (x0, y1), (x1, y1), (cx, cy)]


def _aabb_from_points(points: Sequence[Vec3]) -> Tuple[Vec3, Vec3]:
    """Axis-aligned canonical min/max corners over a set of world points."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    bmin = (min(xs), min(ys), min(zs))
    bmax = (max(xs), max(ys), max(zs))
    return tuple(min(bmin[i], bmax[i]) for i in range(3)), tuple(
        max(bmin[i], bmax[i]) for i in range(3)
    )


def _aabb_is_usable(bbox_min: Vec3, bbox_max: Vec3, *, max_box_size: float) -> bool:
    values = [*bbox_min, *bbox_max]
    if not all(math.isfinite(float(v)) for v in values):
        return False
    sizes = [float(bbox_max[i]) - float(bbox_min[i]) for i in range(3)]
    if any(size < 0.0 for size in sizes):
        return False
    return max(sizes) <= float(max_box_size)


# ----------------------------- the index -----------------------------

class PerceptionSceneSpatialIndex:
    """Build world-space :class:`SceneObject` AABBs from 2D detections + depth.

    Satisfies the ``SceneSpatialIndex`` protocol. Construction is cheap and side
    effect free; the work happens in :meth:`objects`, which unprojects each
    detection's box. Heavy perception (SAM3/DA3) is upstream — we only consume its
    outputs via ``detections`` and the injectable ``depth_provider``.
    """

    def __init__(
        self,
        detections: Sequence[Mapping[str, object]],
        depth_provider: DepthProvider,
        camera: Mapping[str, object],
        *,
        samples_per_axis: int = 3,
        max_world_box_size: float = 6.0,
    ) -> None:
        """
        Args:
            detections: ``[{label, bbox_px:(x0,y0,x1,y1), confidence}]`` from SAM3.
            depth_provider: ``(px, py) -> meters`` sampler over the DA3 depth map.
            camera: pinhole intrinsics (``fx,fy,cx,cy`` or ``vfov,width,height``)
                plus look-at extrinsics (``eye``, ``target``, optional ``up``).
            samples_per_axis: grid density for the robust median depth per box.
        """
        self._detections = list(detections)
        self._depth_provider = depth_provider
        self._camera = camera
        self._samples_per_axis = samples_per_axis
        self._max_world_box_size = float(max_world_box_size)
        # Resolve camera geometry once; reused for every detection.
        self._intrinsics = resolve_intrinsics(camera)
        eye, target, up = resolve_extrinsics(camera)
        self._eye = eye
        self._basis = camera_basis(eye, target, up)

    def objects(self) -> List[SceneObject]:
        """Unproject every detection into a world-AABB :class:`SceneObject`.

        Detections whose depth cannot be sampled (no valid readings inside the box)
        are skipped rather than crashing the whole scene — one bad box should not
        sink placement for the rest.
        """
        out: List[SceneObject] = []
        for i, det in enumerate(self._detections):
            bbox_px = det.get("bbox_px")
            if bbox_px is None:
                continue
            # A malformed box (non-iterable scalar, wrong length) must skip this one
            # detection, not crash every other object's placement — same robustness
            # contract as the no-valid-depth skip below.
            try:
                bbox_t = tuple(bbox_px)  # type: ignore[arg-type]
            except TypeError:
                continue
            if len(bbox_t) != 4:
                continue
            try:
                depth = _sample_box_depth(
                    bbox_t,
                    self._depth_provider,
                    samples_per_axis=self._samples_per_axis,
                )
            except ValueError:
                continue
            # Unproject corners + center at the (single, robust) box depth. Using one
            # depth keeps the box on a fronto-parallel slab — enough to anchor a
            # standoff; per-pixel depth would over-fit the noisy edges.
            world_pts = [
                unproject(px, py, depth, self._eye, self._intrinsics, self._basis)
                for (px, py) in _box_corner_pixels(bbox_t)
            ]
            bbox_min, bbox_max = _aabb_from_points(world_pts)
            if not _aabb_is_usable(
                bbox_min,
                bbox_max,
                max_box_size=self._max_world_box_size,
            ):
                continue
            centroid = world_pts[-1]  # the box center we appended last
            label = str(det.get("label", "") or "")
            confidence = det.get("confidence", 1.0)
            out.append(
                SceneObject(
                    id=str(det.get("id") or f"perception_{i}_{label or 'object'}"),
                    label=label,
                    bbox_min=bbox_min,
                    bbox_max=bbox_max,
                    centroid=centroid,
                    category=str(det.get("category", "") or ""),
                    source="perception",
                    confidence=float(confidence) if confidence is not None else 1.0,
                    extra={"bbox_px": tuple(float(v) for v in bbox_t), "depth_m": depth},
                )
            )
        return out
