"""Find the arm configuration a local tracker cannot reach on its own.

The live pose solver is a QP differential-IK controller with a posture cost.
That is the right tool for smoothly following a pose and the wrong one for
choosing which arm configuration to follow it in: it refines locally, so it
stays in whatever basin its seed lands in.

C45 measured the consequence.  At the contact pose every branch the live solver
found sat 0.014 to 0.024 rad from a joint stop, and the arm spent a third of
the contact phase against ``panda_joint5``'s hard limit.  Off-sim, from the
*same* sixteen seeds and the *same* joint limits -- verified equal to four
decimals against the live receipt -- a damped-least-squares search finds a
configuration with 0.62 rad of margin that reaches the same pose.  It needs
twenty iterations, not more; the live solver has 192 and still does not find
it, because that configuration is 2.66 rad away in joint space and a posture
cost is precisely what stops a tracker from travelling that far.

So the search is done separately and the answer handed over as a seed.  Nothing
here replaces the live solver: it produces candidate configurations, the live
multistart refines and scores them exactly as it always has, and every gate
downstream is untouched.  A runtime that cannot run this search loses nothing,
because the seeds it would have added are additions to a list that already
works.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any


GLOBAL_SEED_SEARCH_SCHEMA_VERSION = "native_franka_global_margin_seed_search.v1"

#: Damped-least-squares step.  Large enough to cross basins, small enough that
#: the iteration stays stable against the damping below.
DEFAULT_STEP = 0.4
DEFAULT_DAMPING = 1.0e-4
#: Twenty iterations already found the 0.62 rad branch from the live seeds in
#: every off-sim trial; sixty is headroom, not a requirement.
DEFAULT_MAX_ITERATIONS = 60
#: Seeds returned.  These are prepended to the solver's own preferred seeds, so
#: a handful of good basins is the whole point -- this is not a replacement
#: search.
DEFAULT_SEED_LIMIT = 3


def _quaternion_error_vector(
    current_xyzw: Sequence[float], target_xyzw: Sequence[float]
) -> list[float]:
    """Rotation from current to target, as a world-frame rotation vector."""

    cx, cy, cz, cw = (float(value) for value in current_xyzw)
    tx, ty, tz, tw = (float(value) for value in target_xyzw)
    # target * inverse(current)
    ix, iy, iz, iw = -cx, -cy, -cz, cw
    x = tw * ix + tx * iw + ty * iz - tz * iy
    y = tw * iy - tx * iz + ty * iw + tz * ix
    z = tw * iz + tx * iy - ty * ix + tz * iw
    w = tw * iw - tx * ix - ty * iy - tz * iz
    norm = math.sqrt(x * x + y * y + z * z)
    if norm <= 1.0e-12:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(norm, abs(w))
    if w < 0.0:
        angle = -angle
    scale = angle / norm
    return [x * scale, y * scale, z * scale]


def high_margin_joint_seeds(
    *,
    frame_pose: Callable[[Sequence[float]], tuple[Sequence[float], Sequence[float]]],
    frame_jacobian: Callable[[Sequence[float]], Sequence[Sequence[float]]],
    seeds: Sequence[Sequence[float]],
    target_position_m: Sequence[float],
    target_quaternion_xyzw: Sequence[float],
    lower_joint_position_limits_rad: Sequence[float],
    upper_joint_position_limits_rad: Sequence[float],
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    seed_limit: int = DEFAULT_SEED_LIMIT,
) -> dict[str, Any]:
    """Configurations that reach the pose with the most joint-limit margin.

    Runs an unregularised damped-least-squares descent from each seed, keeps
    whichever converged configurations clear the caller's own tolerances, and
    returns the ones furthest from a joint stop.  Deliberately has no posture
    cost: staying near the reference is what the live solver is for, and what
    prevents it from finding these.
    """

    try:
        import numpy as np
    except Exception:  # noqa: BLE001 - a runtime without numpy simply opts out
        return {
            "schema_version": GLOBAL_SEED_SEARCH_SCHEMA_VERSION,
            "status": "unavailable",
            "reason": "numpy_unavailable",
            "seeds": [],
        }

    lower = np.array([float(v) for v in lower_joint_position_limits_rad], dtype=float)
    upper = np.array([float(v) for v in upper_joint_position_limits_rad], dtype=float)
    if lower.shape != upper.shape or lower.size == 0 or bool(np.any(lower >= upper)):
        return {
            "schema_version": GLOBAL_SEED_SEARCH_SCHEMA_VERSION,
            "status": "unavailable",
            "reason": "joint_limits_invalid",
            "seeds": [],
        }
    target_position = np.array([float(v) for v in target_position_m], dtype=float)
    target_quaternion = [float(v) for v in target_quaternion_xyzw]
    if target_position.size != 3 or len(target_quaternion) != 4:
        return {
            "schema_version": GLOBAL_SEED_SEARCH_SCHEMA_VERSION,
            "status": "unavailable",
            "reason": "target_pose_invalid",
            "seeds": [],
        }

    found: list[tuple[float, list[float], float, float]] = []
    evaluated = 0
    for seed in seeds:
        try:
            q = np.clip(np.array([float(v) for v in seed], dtype=float), lower, upper)
        except (TypeError, ValueError):
            continue
        if q.shape != lower.shape:
            continue
        evaluated += 1
        position_error = orientation_error = float("inf")
        for _ in range(max(1, int(max_iterations))):
            try:
                position, quaternion = frame_pose(q.tolist())
                jacobian = np.array(
                    [[float(value) for value in row] for row in frame_jacobian(q.tolist())],
                    dtype=float,
                )
            except Exception:  # noqa: BLE001 - a seed that cannot be evaluated is skipped
                position_error = float("inf")
                break
            delta_position = target_position - np.array(
                [float(v) for v in position], dtype=float
            )
            delta_rotation = np.array(
                _quaternion_error_vector(quaternion, target_quaternion), dtype=float
            )
            position_error = float(np.linalg.norm(delta_position))
            orientation_error = float(np.linalg.norm(delta_rotation))
            if (
                position_error <= float(position_tolerance_m)
                and orientation_error <= float(orientation_tolerance_rad)
            ):
                break
            if jacobian.shape != (6, lower.size):
                position_error = float("inf")
                break
            error = np.concatenate([delta_position, delta_rotation])
            step = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + DEFAULT_DAMPING * np.eye(6), error
            )
            q = np.clip(q + DEFAULT_STEP * step, lower, upper)
        if (
            position_error <= float(position_tolerance_m)
            and orientation_error <= float(orientation_tolerance_rad)
        ):
            margin = float(np.min(np.minimum(q - lower, upper - q)))
            found.append((margin, q.tolist(), position_error, orientation_error))

    found.sort(key=lambda row: -row[0])
    kept = found[: max(1, int(seed_limit))]
    return {
        "schema_version": GLOBAL_SEED_SEARCH_SCHEMA_VERSION,
        "status": "searched" if kept else "no_configuration_converged",
        "seeds_evaluated": evaluated,
        "configurations_found": len(found),
        "seeds": [row[1] for row in kept],
        "margins_rad": [row[0] for row in kept],
        "best_margin_rad": kept[0][0] if kept else None,
        "claim_boundary": (
            "produces_candidate_seed_configurations_only;the_live_solver_"
            "refines_and_scores_them_and_every_arrival_and_contact_gate_is_"
            "unchanged"
        ),
    }


__all__ = [
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_SEED_LIMIT",
    "GLOBAL_SEED_SEARCH_SCHEMA_VERSION",
    "high_margin_joint_seeds",
]
