"""Dynamic robot-placement solver: where should the pelvis stand to act on a target?

Given a resolved target object and an injectable footprint-overlap ``probe``, this
finds the OPEN side of the target and places the robot pelvis on the nearest clear
spot, facing the target centroid. There are NO hardcoded coordinates: every position
is derived from the target's world AABB plus the probe's read of the floor, so the
same solver works for a faucet against a wall, a stove on an island, or an object
wedged into an L-shaped counter.

This GENERALIZES the GPU runner's original ``find_clear_stand`` heuristic, which only
ever stepped in ``-y`` from the target. That worked for one camera convention and
fell over the moment the open side was ``+x`` (counter against the back wall) — the
robot would march into the cabinetry. Here we probe all four cardinal directions (and
optionally the diagonals), score each by how close its nearest-clear spot is to the
target, and pick the closest clear one so the robot ends up near enough to actually
reach. Edge cases are explicit: a target boxed in by two walls exposes only one open
side; a fully boxed-in target yields a best-effort pose at ``max_out`` with
``clear=False`` so the caller can see we could not verify the floor.

The whole module is PURE given the injected ``probe`` (signature ``(pose, yaw) -> int``,
``0`` == clear). No PhysX, no GPU, no isaacsim — tests drive it with a mock probe that
marks occupied cells, which is exactly how the real PhysX overlap query behaves.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

from .robot_profile import RobotProfile
from .types import Probe, SceneObject, StandPose, Vec3


def _resolve(
    explicit: Optional[float],
    profile: Optional[RobotProfile],
    profile_field: str,
    default: float,
) -> float:
    """Explicit kwarg > robot profile field > historical default."""
    if explicit is not None:
        return float(explicit)
    if profile is not None:
        return float(getattr(profile, profile_field))
    return float(default)

# Unit step directions in the floor (xy) plane. Cardinals are the bread-and-butter
# (counters/walls are axis-aligned in practice); diagonals are opt-in for targets
# wedged into a corner where only an off-axis approach is open.
_CARDINALS: Tuple[Tuple[float, float], ...] = (
    (1.0, 0.0),   # +x
    (-1.0, 0.0),  # -x
    (0.0, 1.0),   # +y
    (0.0, -1.0),  # -y
)
_DIAGONALS: Tuple[Tuple[float, float], ...] = (
    (math.sqrt(0.5), math.sqrt(0.5)),
    (math.sqrt(0.5), -math.sqrt(0.5)),
    (-math.sqrt(0.5), math.sqrt(0.5)),
    (-math.sqrt(0.5), -math.sqrt(0.5)),
)


def _yaw_towards(from_xy: Tuple[float, float], to_xy: Tuple[float, float]) -> float:
    """Heading (radians) that points from ``from_xy`` at ``to_xy`` in the xy plane.

    ``atan2(dy, dx)`` is the world yaw the pelvis faces so its forward axis looks at
    the target centroid. Degenerate (coincident) points fall back to ``0.0`` rather
    than raising — the caller still gets a usable pose.
    """
    dx = to_xy[0] - from_xy[0]
    dy = to_xy[1] - from_xy[1]
    if dx == 0.0 and dy == 0.0:
        return 0.0
    return math.atan2(dy, dx)


def _half_extent_along(target: SceneObject, direction: Tuple[float, float]) -> float:
    """Distance from the footprint center to the AABB edge along ``direction``.

    The standoff is measured from the *surface* of the target, not its center, so a
    big stove and a small faucet both end up a sensible reach away. For an axis
    direction this is just the half-width on that axis; for a diagonal we take the
    box-corner projection so a diagonal approach clears the corner.
    """
    dx, dy, _ = target.size()
    hx, hy = 0.5 * dx, 0.5 * dy
    ux, uy = direction
    # Support function of an axis-aligned box for unit direction (|ux|*hx + |uy|*hy).
    return abs(ux) * hx + abs(uy) * hy


def _candidate_along(
    target: SceneObject,
    direction: Tuple[float, float],
    *,
    probe: Probe,
    pelvis_height: float,
    floor_z: float,
    standing_distance: float,
    step: float,
    max_out: float,
    clearance: float,
) -> Tuple[Vec3, float, bool, float]:
    """Probe outward along one direction; return (pose, standoff, clear, distance).

    Starting just past the target surface (``half_extent + standing_distance``), we
    step out in ``step`` increments until the probe reports clear floor or we reach
    ``max_out``. ``distance`` is how far the clear spot sits from the footprint
    center — the score the caller minimizes so the robot stands as close as it safely
    can. If nothing clears, we return the ``max_out`` spot with ``clear=False`` so the
    most-open direction can still serve as a best-effort fallback.
    """
    cx, cy = target.footprint_center()
    ux, uy = direction
    half = _half_extent_along(target, direction)
    z = floor_z + pelvis_height

    start = half + standing_distance
    # Clamp the search ceiling so the standoff spot is ALWAYS probed at least once,
    # even when standing_distance already meets/exceeds max_out (otherwise start >
    # ceiling and the loop body never runs, falsely reporting clear=False).
    ceiling = max(half + max_out, start)
    yaw = _yaw_towards((0.0, 0.0), (-ux, -uy))  # face back toward the target center

    out = start
    last_pose: Vec3 = (cx + ux * out, cy + uy * out, z)
    # Count how many probe steps were blocked before we gave up. A direction that was
    # blocked at every step is more boxed-in than one that ran clear right up to the
    # ceiling, so this gives the fallback a real "most open" signal instead of the
    # constant standoff every direction used to report.
    blocked_steps = 0
    while out <= ceiling + 1e-9:
        pose: Vec3 = (cx + ux * out, cy + uy * out, z)
        last_pose = pose
        if probe(pose, yaw) == 0:
            standoff = out - half
            dist = math.hypot(pose[0] - cx, pose[1] - cy)
            return pose, standoff, True, dist
        blocked_steps += 1
        out += step

    # Nothing clear along this direction. Report the farthest probe spot as a
    # best-effort, flagged not-clear. ``distance`` here doubles as the "most open"
    # score the caller maximizes for the fallback: the FEWER blocked steps we hit,
    # the more open the direction was, so we subtract the blocked count from the
    # reached distance to rank a barely-blocked corridor above a flush wall.
    standoff = math.hypot(last_pose[0] - cx, last_pose[1] - cy) - half
    openness = math.hypot(last_pose[0] - cx, last_pose[1] - cy) - blocked_steps * step
    # ``clearance`` is reserved for callers wanting an inflated footprint; it does not
    # change the geometry here but is surfaced so the signature stays stable/honest.
    _ = clearance
    return last_pose, standoff, False, openness


def _normalize2(v: Tuple[float, float]) -> Tuple[float, float]:
    """Unit-normalize a 2D vector; return (0, 0) for a ~zero vector (no preference)."""
    n = math.hypot(v[0], v[1])
    if n <= 1e-9:
        return (0.0, 0.0)
    return (v[0] / n, v[1] / n)


def _target_degeneracy_reason(target: SceneObject) -> str | None:
    coords = [*target.bbox_min, *target.bbox_max, *target.centroid]
    try:
        finite = all(math.isfinite(float(v)) for v in coords)
    except (TypeError, ValueError):
        finite = False
    if not finite:
        return "non_finite_target_aabb"
    if any(float(target.bbox_min[i]) > float(target.bbox_max[i]) for i in range(3)):
        return "inverted_target_aabb"
    return None


def _degenerate_target_stand_pose(
    target: SceneObject,
    *,
    reason: str,
    pelvis_height: float,
    floor_z: float,
) -> StandPose:
    def finite_or(value: float, fallback: float) -> float:
        try:
            f = float(value)
        except (TypeError, ValueError):
            return fallback
        return f if math.isfinite(f) else fallback

    xs = [finite_or(target.bbox_min[0], 0.0), finite_or(target.bbox_max[0], 0.0)]
    ys = [finite_or(target.bbox_min[1], 0.0), finite_or(target.bbox_max[1], 0.0)]
    cx = 0.5 * (min(xs) + max(xs))
    cy = 0.5 * (min(ys) + max(ys))
    z = finite_or(floor_z, 0.0) + finite_or(pelvis_height, 0.79)
    return StandPose(
        position=(cx, cy, z),
        yaw=0.0,
        target_id=target.id,
        clear=False,
        standoff_m=0.0,
        notes=f"degenerate_target:{reason}; placement solver did not probe '{target.label}'",
    )


def compute_stand_pose(
    target: SceneObject,
    *,
    probe: Probe,
    pelvis_height: Optional[float] = None,
    floor_z: float = 0.0,
    standing_distance: Optional[float] = None,
    step: Optional[float] = None,
    max_out: Optional[float] = None,
    clearance: Optional[float] = None,
    include_diagonals: bool = False,
    preferred_direction: Optional[Tuple[float, float]] = None,
    openable_target: bool | None = None,
    openable_standoff_extra_m: Optional[float] = None,
    robot_profile: Optional["RobotProfile"] = None,
) -> StandPose:
    """Resolve the pelvis pose that stands the robot on the target's open side.

    Strategy (no hardcoded coordinates — everything derives from the target AABB and
    the probe):

    1. Probe outward from the footprint center along each candidate direction
       (cardinals, plus diagonals when ``include_diagonals``) until the floor is clear.
    2. Among the directions that found CLEAR floor, pick the one whose clear spot is
       closest to the target — the robot should stand near enough to act.
    3. Place the pelvis there at ``z = floor_z + pelvis_height`` and yaw it to face the
       target centroid.
    4. If NOTHING is clear (target boxed in by walls/clutter on every side), fall back
       to the most-open direction's ``max_out`` spot with ``clear=False`` and a note,
       so the caller can decide whether to proceed or recapture.

    Args mirror the GPU runner's knobs so this is a drop-in generalization:
    ``standing_distance`` is the reach gap past the target surface, ``step`` the probe
    granularity, ``max_out`` the search ceiling.

    WARNING — ``clearance`` is currently INERT: it is NOT applied to the standoff or
    the footprint, so it adds NO safety margin. The footprint margin lives entirely
    inside the injected ``probe`` (it decides how much floor a candidate pose needs
    to be "clear"). Do not pass a non-zero ``clearance`` expecting an inflated gap —
    a pose can be flagged ``clear=True`` with the robot footprint sitting exactly at
    ``standing_distance`` from the surface regardless of this value. The parameter is
    retained only to keep the signature stable for a future footprint-inflation hook.

    Degenerate targets fail closed before probing. Non-finite or inverted target
    AABBs return a finite, ``clear=False`` pose with a ``degenerate_target`` note
    instead of emitting NaN positions or negative standoffs.

    ``openable_target`` adds a conservative door/drawer swing margin to the
    requested standoff. It is a label-derived placement hint, not proof that the
    fixture actually opens or that manipulation succeeds.

    ``robot_profile`` supplies robot-specific defaults (pelvis height, standing
    distance, probe tuning) for any of the above knobs the caller leaves unset;
    an explicit kwarg always wins over the profile. With neither, the historical
    G1-scale defaults apply unchanged.
    """
    pelvis_height = _resolve(pelvis_height, robot_profile, "pelvis_height_m", 0.79)
    standing_distance = _resolve(standing_distance, robot_profile, "standing_distance_m", 0.55)
    step = _resolve(step, robot_profile, "probe_step_m", 0.10)
    max_out = _resolve(max_out, robot_profile, "probe_max_out_m", 2.5)
    clearance = _resolve(clearance, robot_profile, "probe_clearance_m", 0.10)
    openable_standoff_extra_m = _resolve(
        openable_standoff_extra_m, robot_profile, "openable_standoff_extra_m", 0.25
    )
    degenerate_reason = _target_degeneracy_reason(target)
    if degenerate_reason is not None:
        return _degenerate_target_stand_pose(
            target,
            reason=degenerate_reason,
            pelvis_height=pelvis_height,
            floor_z=floor_z,
        )

    directions = list(_CARDINALS) + (list(_DIAGONALS) if include_diagonals else [])
    centroid_xy = (target.centroid[0], target.centroid[1])
    if openable_target is None:
        target_kind = str(target.category or target.extra.get("target_kind") or "").lower()
        openable_target = target_kind in {"openable", "articulated"}
    effective_standing_distance = float(standing_distance)
    if openable_target:
        effective_standing_distance += max(0.0, float(openable_standoff_extra_m))

    clear_candidates: List[Tuple[float, Vec3, float, Tuple[float, float]]] = []
    # Fallback tuple: (openness, pose, standoff, direction). ``openness`` is the
    # not-clear score from _candidate_along (reached distance minus blocked steps),
    # so a barely-blocked / far-reaching direction outranks a flush wall.
    fallback: Tuple[float, Vec3, float, Tuple[float, float]] | None = None

    for direction in directions:
        pose, standoff, clear, score = _candidate_along(
            target,
            direction,
            probe=probe,
            pelvis_height=pelvis_height,
            floor_z=floor_z,
            standing_distance=effective_standing_distance,
            step=step,
            max_out=max_out,
            clearance=clearance,
        )
        if clear:
            # For clear candidates ``score`` is the distance to the clear spot.
            clear_candidates.append((score, pose, standoff, direction))
        else:
            # Track the most-open fallback by the openness score (higher == more open
            # floor before being blocked), so the fallback reflects which side is
            # genuinely most open rather than just the first cardinal probed.
            if fallback is None or score > fallback[0]:
                fallback = (score, pose, standoff, direction)

    if clear_candidates:
        pref = _normalize2(preferred_direction) if preferred_direction is not None else (0.0, 0.0)
        if pref != (0.0, 0.0):
            # Bias toward the approach/open-room side: a clear spot whose direction aligns with
            # `preferred_direction` beats a closer one on, say, a wall side. Crucial because kitchen
            # walls often lack collision, so the probe reports the wall side as "clear" and the
            # closest-spot heuristic alone parks the robot against the wall. Tie-break by distance.
            clear_candidates.sort(
                key=lambda c: (-(c[3][0] * pref[0] + c[3][1] * pref[1]), c[0])
            )
            note = (f"open side found (approach-biased); standoff %.2f m from '{target.label}'")
        else:
            # No preference: closest clear spot so the robot stands near enough to manipulate.
            clear_candidates.sort(key=lambda c: c[0])
            note = f"open side found; standoff %.2f m from '{target.label}'"
        _dist, pose, standoff, _direction = clear_candidates[0]
        yaw = _yaw_towards((pose[0], pose[1]), centroid_xy)
        return StandPose(
            position=pose,
            yaw=yaw,
            target_id=target.id,
            clear=True,
            standoff_m=standoff,
            notes=(note % standoff) + (
                "; openable standoff margin applied" if openable_target else ""
            ),
        )

    # Boxed in on every probed side: best-effort on the genuinely most-open direction.
    assert fallback is not None  # directions is non-empty, so we always have a fallback
    _openness, pose, standoff, _direction = fallback
    yaw = _yaw_towards((pose[0], pose[1]), centroid_xy)
    return StandPose(
        position=pose,
        yaw=yaw,
        target_id=target.id,
        clear=False,
        standoff_m=standoff,
        notes=(
            f"no clear side for '{target.label}'; fell back to the farthest probed "
            f"spot (standoff {standoff:.2f} m, search ceiling {max_out:.2f} m) on "
            f"the most-open direction"
        ),
    )


def ring_scan_stand_pose(
    target: SceneObject,
    *,
    probe: Probe,
    floor_z: float = 0.0,
    pelvis_height: Optional[float] = None,
    standing_distance: Optional[float] = None,
    max_standing_distance: Optional[float] = None,
    radial_step: float = 0.08,
    n_azimuths: int = 36,
    robot_profile: Optional["RobotProfile"] = None,
) -> StandPose:
    """Dense annulus scan around the target — the fallback when ray probing fails.

    :func:`compute_stand_pose` probes straight rays THROUGH the target's footprint
    center, which fails for a target pinned near a wall: every axis-aligned ray
    from its center clips the wall band, even though a laterally-offset stance
    half a footprint to the side is wide open (a rice cooker at the wall end of a
    kitchen counter). This scans the full ring — ``n_azimuths`` bearings x radii
    from ``standing_distance`` to ``max_standing_distance`` past the target
    surface — and returns the CLOSEST clear spot, yawed at the target centroid.

    Radii are center-to-surface (same convention as ``standing_distance`` in
    :func:`compute_stand_pose`); keep ``max_standing_distance`` within the
    validator's standoff ceiling or the winning spot may pass the probe but fail
    the standoff gate. Returns ``clear=False`` (positioned at the target
    footprint center direction of the first bearing) when the whole annulus is
    blocked — the caller decides whether to accept, re-scope, or recapture.
    """
    pelvis_height = _resolve(pelvis_height, robot_profile, "pelvis_height_m", 0.79)
    standing_distance = _resolve(standing_distance, robot_profile, "standing_distance_m", 0.55)
    if max_standing_distance is None:
        max_standing_distance = standing_distance + 0.6
    degenerate_reason = _target_degeneracy_reason(target)
    if degenerate_reason is not None:
        return _degenerate_target_stand_pose(
            target, reason=degenerate_reason, pelvis_height=pelvis_height, floor_z=floor_z
        )
    cx, cy = target.footprint_center()
    centroid_xy = (target.centroid[0], target.centroid[1])
    z = floor_z + pelvis_height
    best: Tuple[float, Vec3, float] | None = None  # (distance, pose, standoff)
    for k in range(max(1, int(n_azimuths))):
        theta = 2.0 * math.pi * k / max(1, int(n_azimuths))
        direction = (math.cos(theta), math.sin(theta))
        half = _half_extent_along(target, direction)
        out = half + float(standing_distance)
        ceiling = half + float(max_standing_distance)
        while out <= ceiling + 1e-9:
            pose: Vec3 = (cx + direction[0] * out, cy + direction[1] * out, z)
            yaw = _yaw_towards((pose[0], pose[1]), centroid_xy)
            if probe(pose, yaw) == 0:
                dist = math.hypot(pose[0] - cx, pose[1] - cy)
                if best is None or dist < best[0]:
                    best = (dist, pose, out - half)
                break  # nearest clear radius on this bearing; move to the next
            out += float(radial_step)
    if best is not None:
        _dist, pose, standoff = best
        yaw = _yaw_towards((pose[0], pose[1]), centroid_xy)
        return StandPose(
            position=pose,
            yaw=yaw,
            target_id=target.id,
            clear=True,
            standoff_m=standoff,
            notes=(
                f"ring-scan stance; standoff {standoff:.2f} m from '{target.label}' "
                f"({n_azimuths} bearings)"
            ),
        )
    fallback_pose: Vec3 = (cx + float(standing_distance), cy, z)
    return StandPose(
        position=fallback_pose,
        yaw=_yaw_towards((fallback_pose[0], fallback_pose[1]), centroid_xy),
        target_id=target.id,
        clear=False,
        standoff_m=float(standing_distance),
        notes=(
            f"ring-scan found no clear stance for '{target.label}' "
            f"({n_azimuths} bearings x radii {standing_distance:.2f}-"
            f"{max_standing_distance:.2f} m past the surface)"
        ),
    )
