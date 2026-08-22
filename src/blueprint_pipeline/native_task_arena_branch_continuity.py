"""Choose the phase branches that form a traversable path, not the greedy ones.

The controls preflight solved each phase in turn and kept, for each one, the
admissible solution nearest the *previous* phase.  That is greedy forward
chaining: every phase commits before it knows whether the next phase can be
reached from where it landed.  C39 paid for it.  Scoring solves on the frame
the gate measures moved the admissible set, approach's own best branch landed
0.615 rad from contact's, and the bounded entry path -- built to interpolate a
few hundredths of a radian -- was asked to cross a whole arm reconfiguration.
Contact arrived 150 mm out.

The information needed to avoid that was already in the receipt.  The
multistart seals every seed it tried under ``attempts``, so all admissible
branches for every phase are computed and then discarded.  Enumerating the
combinations instead is pure arithmetic over a handful of postures: it runs
off-sim, before physics, and costs no measurable GPU time.  A run that used to
test one branch chain now tests all of them.

The chain is scored on what actually breaks: the largest single-joint hop
between consecutive phases, because that is what sizes the entry path and what
the actuators must deliver.  Total travel and pose error break ties.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any


BRANCH_CONTINUITY_SCHEMA_VERSION = "native_task_arena_branch_continuity.v1"

#: Enumeration ceiling.  Five phases at eight admissible branches is 32768
#: combinations of seven-float arithmetic -- microseconds.  The cap only stops
#: a pathological solution count from turning a free search into a slow one.
MAX_BRANCH_COMBINATIONS = 200_000


def _joints(row: Mapping[str, Any]) -> list[float] | None:
    try:
        values = [float(value) for value in row["joint_positions_rad"]]
    except (KeyError, TypeError, ValueError):
        return None
    return values if len(values) == 7 else None


def admissible_branches(
    phase: Mapping[str, Any], *, required_margin_rad: float
) -> list[dict[str, Any]]:
    """Every solved branch for one phase that clears the margin floor.

    Falls back to the phase's own selection when the solver sealed no attempt
    list, so a receipt shape this has not seen still yields a usable chain
    rather than an empty one.
    """

    rows = phase.get("attempts")
    candidates: list[dict[str, Any]] = []
    if isinstance(rows, list):
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or row.get("solved") is False:
                continue
            joints = _joints(row)
            if joints is None:
                continue
            margin = row.get("minimum_joint_limit_margin_rad")
            if margin is not None and float(margin) < float(required_margin_rad):
                continue
            candidates.append(
                {
                    "branch_index": index,
                    "seed_index": row.get("seed_index"),
                    "joint_positions_rad": joints,
                    "position_error_m": row.get("position_error_m"),
                    "orientation_error_rad": row.get("orientation_error_rad"),
                    "minimum_joint_limit_margin_rad": margin,
                }
            )
    if not candidates and not isinstance(rows, list):
        # Fall back only when the solver sealed no attempt list at all.  An
        # attempt list that filters to nothing means no branch clears the
        # margin floor, and saying so is the honest answer.
        selected = phase.get("selected")
        if isinstance(selected, Mapping):
            joints = _joints(selected)
            if joints is not None:
                candidates.append(
                    {
                        "branch_index": 0,
                        "seed_index": selected.get("seed_index"),
                        "joint_positions_rad": joints,
                        "position_error_m": selected.get("position_error_m"),
                        "orientation_error_rad": selected.get(
                            "orientation_error_rad"
                        ),
                        "minimum_joint_limit_margin_rad": selected.get(
                            "minimum_joint_limit_margin_rad"
                        ),
                    }
                )
    return candidates


def _chain_cost(
    chain: Sequence[Mapping[str, Any]], start_joints: Sequence[float] | None
) -> tuple[float, float, float]:
    """Score a chain on the hop that actually breaks, then on total travel.

    Only hops *between phases* are bounded: those are what the entry path
    interpolates and what the actuators must deliver inside a phase budget.
    The run-up from wherever the arm starts to the first phase is an ordinary
    servo move over open space with its own budget, so it belongs in total
    travel and must not dominate the criterion -- otherwise a chain is judged
    by how far the arm happened to be parked.
    """

    postures = [row["joint_positions_rad"] for row in chain]
    largest_hop = 0.0
    total_travel = 0.0
    for before, after in zip(postures, postures[1:], strict=False):
        deltas = [abs(a - b) for a, b in zip(before, after, strict=True)]
        largest_hop = max(largest_hop, max(deltas))
        total_travel += sum(deltas)
    if start_joints is not None and postures:
        total_travel += sum(
            abs(a - b)
            for a, b in zip(list(start_joints), postures[0], strict=True)
        )
    error = sum(
        float(row.get("position_error_m") or 0.0)
        + float(row.get("orientation_error_rad") or 0.0)
        for row in chain
    )
    return largest_hop, total_travel, error


def select_continuous_branch_chain(
    *,
    phases: Sequence[Mapping[str, Any]],
    required_margin_rad: float,
    start_joint_positions_rad: Sequence[float] | None = None,
    max_combinations: int = MAX_BRANCH_COMBINATIONS,
) -> dict[str, Any]:
    """Pick one branch per phase so the whole path is traversable.

    Enumerates every combination of admissible branches and keeps the chain
    with the smallest largest single-joint hop between consecutive phases,
    breaking ties on total joint travel and then on pose error.  Reports what
    the greedy chain would have cost, so the receipt shows the improvement
    rather than asserting one.
    """

    per_phase = [
        admissible_branches(phase, required_margin_rad=required_margin_rad)
        for phase in phases
    ]
    if not per_phase or any(not rows for rows in per_phase):
        return {
            "schema_version": BRANCH_CONTINUITY_SCHEMA_VERSION,
            "status": "unavailable",
            "reason": "phase_without_admissible_branch",
            "selected_chain": [],
        }
    combinations = 1
    for rows in per_phase:
        combinations *= len(rows)
        if combinations > max_combinations:
            return {
                "schema_version": BRANCH_CONTINUITY_SCHEMA_VERSION,
                "status": "unavailable",
                "reason": f"branch_combinations_exceed_cap:{max_combinations}",
                "selected_chain": [],
            }

    best_chain: tuple[Mapping[str, Any], ...] | None = None
    best_cost: tuple[float, float, float] | None = None
    for chain in product(*per_phase):
        cost = _chain_cost(chain, start_joint_positions_rad)
        if best_cost is None or cost < best_cost:
            best_cost, best_chain = cost, chain

    # What the previous greedy rule would have produced, for comparison only.
    greedy: list[Mapping[str, Any]] = []
    reference = list(start_joint_positions_rad or [])
    for rows in per_phase:
        if reference:
            pick = min(
                rows,
                key=lambda row: max(
                    abs(a - b)
                    for a, b in zip(
                        row["joint_positions_rad"], reference, strict=True
                    )
                ),
            )
        else:
            pick = rows[0]
        greedy.append(pick)
        reference = list(pick["joint_positions_rad"])
    greedy_cost = _chain_cost(greedy, start_joint_positions_rad)

    assert best_chain is not None and best_cost is not None
    return {
        "schema_version": BRANCH_CONTINUITY_SCHEMA_VERSION,
        "status": "selected",
        "combinations_evaluated": combinations,
        "branches_per_phase": [len(rows) for rows in per_phase],
        "selected_chain": [dict(row) for row in best_chain],
        "largest_single_joint_hop_rad": best_cost[0],
        "total_joint_travel_rad": best_cost[1],
        "greedy_largest_single_joint_hop_rad": greedy_cost[0],
        "greedy_total_joint_travel_rad": greedy_cost[1],
        "claim_boundary": (
            "selects_branches_for_traversability_off_sim;native_arrival_and_"
            "contact_gates_remain_the_authority"
        ),
    }


__all__ = [
    "BRANCH_CONTINUITY_SCHEMA_VERSION",
    "MAX_BRANCH_COMBINATIONS",
    "admissible_branches",
    "select_continuous_branch_chain",
]
