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
configuration with 0.4514 rad of margin that reaches the same pose -- the
number this module actually produced against C45's sealed scene, not the
0.62 rad an earlier prototype reported under looser tolerances.  It needs
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
#: When no configuration clears both pose tolerances, retain the closest
#: distinct terminal configurations as *unsolved* seeds for the live solver to
#: refine.  C79 evaluated 128 global starts and then discarded every terminal
#: posture because none crossed the final 0.08 rad gate; the physics fallback
#: consequently measured only the older local solver's 16 endpoints.
DEFAULT_NEAR_FEASIBLE_SEED_LIMIT = 8
#: Global configurations evaluated before the local tracker refines anything.
#: This is deliberately above one hundred: evaluating FK and a 6x7 Jacobian is
#: cheap compared with one PhysX episode, and a seven-DOF arm should not choose
#: its kinematic branch from sixteen hand-authored postures.
DEFAULT_DIVERSE_SEED_COUNT = 128
#: Two configurations closer than this in joint space are the same basin, and
#: seeding both wastes a slot that a genuinely different branch could use.
DISTINCT_CONFIGURATION_RADIUS_RAD = 0.35


def _radical_inverse(index: int, base: int) -> float:
    value = 0.0
    scale = 1.0 / float(base)
    while index:
        index, digit = divmod(index, base)
        value += digit * scale
        scale /= float(base)
    return value


def diverse_joint_seeds(
    *,
    seeds: Sequence[Sequence[float]],
    lower_joint_position_limits_rad: Sequence[float],
    upper_joint_position_limits_rad: Sequence[float],
    count: int = DEFAULT_DIVERSE_SEED_COUNT,
) -> list[list[float]]:
    """Preserve caller seeds, then fill the joint box with a Halton design.

    A deterministic low-discrepancy design covers the whole seven-dimensional
    limit box without a random seed or a scene-specific posture.  Generated
    points stay five percent inside each limit because starting exactly on a
    stop is both physically unhelpful and numerically ambiguous.
    """

    lower = [float(value) for value in lower_joint_position_limits_rad]
    upper = [float(value) for value in upper_joint_position_limits_rad]
    target_count = max(1, int(count))
    if (
        not lower
        or len(lower) != len(upper)
        or len(lower) > 10
        or any(
            not math.isfinite(low)
            or not math.isfinite(high)
            or low >= high
            for low, high in zip(lower, upper, strict=True)
        )
    ):
        return []

    result: list[list[float]] = []
    for raw in seeds:
        try:
            row = [float(value) for value in raw]
        except (TypeError, ValueError):
            continue
        if len(row) != len(lower) or not all(math.isfinite(value) for value in row):
            continue
        row = [
            min(high, max(low, value))
            for value, low, high in zip(row, lower, upper, strict=True)
        ]
        if not any(math.dist(row, prior) <= 1.0e-9 for prior in result):
            result.append(row)
        if len(result) >= target_count:
            return result

    primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)
    index = 1
    while len(result) < target_count:
        row = []
        for axis, (low, high) in enumerate(zip(lower, upper, strict=True)):
            unit = 0.05 + 0.90 * _radical_inverse(index, primes[axis])
            row.append(low + unit * (high - low))
        if not any(math.dist(row, prior) <= 1.0e-9 for prior in result):
            result.append(row)
        index += 1
    return result


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
    near_feasible_seed_limit: int = DEFAULT_NEAR_FEASIBLE_SEED_LIMIT,
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
    terminal: list[tuple[float, float, list[float], float, float]] = []
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
        # The loop updates ``q`` after measuring its error.  Re-evaluate the
        # final configuration so a last-iteration convergence is not missed
        # and so retained terminal metadata describes the joints we return.
        try:
            position, quaternion = frame_pose(q.tolist())
            position_error = float(
                np.linalg.norm(
                    target_position
                    - np.array([float(value) for value in position], dtype=float)
                )
            )
            orientation_error = float(
                np.linalg.norm(
                    np.array(
                        _quaternion_error_vector(quaternion, target_quaternion),
                        dtype=float,
                    )
                )
            )
        except Exception:  # noqa: BLE001 - one bad terminal seed is skipped
            continue
        if not math.isfinite(position_error) or not math.isfinite(
            orientation_error
        ):
            continue
        margin = float(np.min(np.minimum(q - lower, upper - q)))
        normalized_error = (
            position_error / float(position_tolerance_m)
        ) ** 2 + (
            orientation_error / float(orientation_tolerance_rad)
        ) ** 2
        terminal.append(
            (
                normalized_error,
                margin,
                q.tolist(),
                position_error,
                orientation_error,
            )
        )
        if (
            position_error <= float(position_tolerance_m)
            and orientation_error <= float(orientation_tolerance_rad)
        ):
            found.append((margin, q.tolist(), position_error, orientation_error))

    found.sort(key=lambda row: -row[0])
    # Highest-margin is not the same as diverse: a descent from several seeds
    # can land in one basin, and three near-duplicates of the same
    # configuration cover no more of the solution space than one.  Keep only
    # configurations that are genuinely distinct in joint space.
    kept: list[tuple[float, list[float], float, float]] = []
    for row in found:
        if len(kept) >= max(1, int(seed_limit)):
            break
        if any(
            float(np.linalg.norm(np.array(row[1]) - np.array(other[1])))
            < DISTINCT_CONFIGURATION_RADIUS_RAD
            for other in kept
        ):
            continue
        kept.append(row)
    terminal.sort(key=lambda row: (row[0], -row[1]))
    near_feasible: list[tuple[float, float, list[float], float, float]] = []
    for row in terminal:
        if len(near_feasible) >= max(1, int(near_feasible_seed_limit)):
            break
        # A configuration that already passed is represented by ``seeds`` and
        # must not also masquerade as a near miss.
        if (
            row[3] <= float(position_tolerance_m)
            and row[4] <= float(orientation_tolerance_rad)
        ):
            continue
        if any(
            float(np.linalg.norm(np.array(row[2]) - np.array(other[2])))
            < DISTINCT_CONFIGURATION_RADIUS_RAD
            for other in near_feasible
        ):
            continue
        near_feasible.append(row)
    return {
        "schema_version": GLOBAL_SEED_SEARCH_SCHEMA_VERSION,
        "status": "searched" if kept else "no_configuration_converged",
        "seeds_evaluated": evaluated,
        "configurations_found": len(found),
        "seeds": [row[1] for row in kept],
        "margins_rad": [row[0] for row in kept],
        "best_margin_rad": kept[0][0] if kept else None,
        "near_feasible_seed_count": len(near_feasible),
        "near_feasible_seeds": [row[2] for row in near_feasible],
        "near_feasible_normalized_pose_errors": [row[0] for row in near_feasible],
        "near_feasible_margins_rad": [row[1] for row in near_feasible],
        "near_feasible_position_errors_m": [row[3] for row in near_feasible],
        "near_feasible_orientation_errors_rad": [row[4] for row in near_feasible],
        "claim_boundary": (
            "produces_candidate_seed_configurations_only;near_feasible_seeds_"
            "remain_explicitly_unsolved;the_live_solver_refines_and_scores_"
            "them_and_every_arrival_and_contact_gate_is_unchanged"
        ),
    }


__all__ = [
    "DEFAULT_DIVERSE_SEED_COUNT",
    "DEFAULT_MAX_ITERATIONS",
    "DISTINCT_CONFIGURATION_RADIUS_RAD",
    "DEFAULT_SEED_LIMIT",
    "GLOBAL_SEED_SEARCH_SCHEMA_VERSION",
    "diverse_joint_seeds",
    "high_margin_joint_seeds",
]
