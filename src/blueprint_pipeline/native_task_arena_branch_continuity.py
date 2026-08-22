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

import math
from collections.abc import Mapping, Sequence
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


def phase_offers_a_branch_choice(phase: Mapping[str, Any]) -> bool:
    """Whether this phase chooses a branch at all.

    A phase that reuses an earlier phase's bound pose has no attempt list and
    no selection of its own -- it inherits.  C40 treated those as phases with
    no admissible branch and abandoned the whole search, so the run fell back
    to the greedy chain it was meant to replace.
    """

    return isinstance(phase.get("attempts"), list) or isinstance(
        phase.get("selected"), Mapping
    )


def _required_margin_for_phase(
    phase: Mapping[str, Any], *, default_margin_rad: float
) -> float:
    """The margin this phase actually required, not another phase's floor.

    The solver seals the floor it enforced.  C40 applied contact's 0.005 rad
    requirement to every phase, which discarded all five of prealign's solved
    branches -- prealign never had to clear it -- and emptied the search.
    """

    value = phase.get("required_minimum_joint_limit_margin_rad")
    try:
        margin = float(value)
    except (TypeError, ValueError):
        return float(default_margin_rad)
    return margin if math.isfinite(margin) else float(default_margin_rad)


def admissible_branches(
    phase: Mapping[str, Any], *, required_margin_rad: float
) -> list[dict[str, Any]]:
    """Every solved branch for one phase that clears *its* margin floor.

    Falls back to the phase's own selection when the solver sealed no attempt
    list, so a receipt shape this has not seen still yields a usable chain
    rather than an empty one.
    """

    required_margin_rad = _required_margin_for_phase(
        phase, default_margin_rad=required_margin_rad
    )
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


def _pair_hop(before: Sequence[float], after: Sequence[float]) -> float:
    return max(abs(a - b) for a, b in zip(before, after, strict=True))


def _pair_travel(before: Sequence[float], after: Sequence[float]) -> float:
    return sum(abs(a - b) for a, b in zip(before, after, strict=True))


def _best_chain_within_bottleneck(
    per_phase: Sequence[Sequence[Mapping[str, Any]]],
    *,
    bottleneck_rad: float,
    start_joints: Sequence[float] | None,
) -> tuple[list[Mapping[str, Any]], float] | None:
    """Cheapest chain whose every inter-phase hop stays within a bottleneck.

    Minimises total joint travel plus pose error by dynamic programming over
    the phase chain, so the cost is a few hundred comparisons rather than the
    product of the branch counts.
    """

    def _node_cost(row: Mapping[str, Any]) -> float:
        return float(row.get("position_error_m") or 0.0) + float(
            row.get("orientation_error_rad") or 0.0
        )

    # Seed the first phase: the run-up from wherever the arm is parked is an
    # ordinary servo move, so it is priced but never bottleneck-limited.
    best: list[tuple[float, list[int]]] = []
    for index, row in enumerate(per_phase[0]):
        cost = _node_cost(row)
        if start_joints is not None:
            cost += _pair_travel(start_joints, row["joint_positions_rad"])
        best.append((cost, [index]))

    for phase_index in range(1, len(per_phase)):
        previous = per_phase[phase_index - 1]
        current = per_phase[phase_index]
        nxt: list[tuple[float, list[int]] | None] = [None] * len(current)
        for here, row in enumerate(current):
            for there, prior in enumerate(previous):
                if best[there] is None:
                    continue
                hop = _pair_hop(
                    prior["joint_positions_rad"], row["joint_positions_rad"]
                )
                if hop > bottleneck_rad:
                    continue
                cost = (
                    best[there][0]
                    + _pair_travel(
                        prior["joint_positions_rad"], row["joint_positions_rad"]
                    )
                    + _node_cost(row)
                )
                if nxt[here] is None or cost < nxt[here][0]:
                    nxt[here] = (cost, [*best[there][1], here])
        best = [entry for entry in nxt]  # type: ignore[assignment]
        if all(entry is None for entry in best):
            return None

    finished = [entry for entry in best if entry is not None]
    if not finished:
        return None
    cost, path = min(finished, key=lambda entry: entry[0])
    return [per_phase[i][branch] for i, branch in enumerate(path)], cost


def select_continuous_branch_chain(
    *,
    phases: Sequence[Mapping[str, Any]],
    required_margin_rad: float,
    start_joint_positions_rad: Sequence[float] | None = None,
    max_combinations: int = MAX_BRANCH_COMBINATIONS,
) -> dict[str, Any]:
    """Pick one branch per phase so the whole path is traversable.

    Searches every combination -- C41's eleven phases carry 12.9 million of
    them -- without enumerating any.  The cost decomposes over consecutive
    phase pairs, so the chain is a shortest-path problem, not a product: bisect
    on the largest permitted single-joint hop and run a min-travel dynamic
    program at each candidate.  A few hundred comparisons replace a product
    that a brute-force enumeration had to refuse outright.

    Reports what the greedy chain would have cost, so the receipt shows the
    improvement rather than asserting one.
    """

    del max_combinations  # the search no longer enumerates, so nothing to cap
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

    # Candidate bottlenecks are exactly the hops that can occur, so bisecting
    # over them finds the true minimum rather than an approximation of it.
    candidates = sorted(
        {
            _pair_hop(before["joint_positions_rad"], after["joint_positions_rad"])
            for index in range(1, len(per_phase))
            for before in per_phase[index - 1]
            for after in per_phase[index]
        }
    ) or [0.0]

    low, high = 0, len(candidates) - 1
    found: tuple[list[Mapping[str, Any]], float] | None = None
    while low <= high:
        middle = (low + high) // 2
        attempt = _best_chain_within_bottleneck(
            per_phase,
            bottleneck_rad=candidates[middle],
            start_joints=start_joint_positions_rad,
        )
        if attempt is None:
            low = middle + 1
        else:
            found = attempt
            high = middle - 1
    if found is None:
        return {
            "schema_version": BRANCH_CONTINUITY_SCHEMA_VERSION,
            "status": "unavailable",
            "reason": "no_chain_connects_every_phase",
            "selected_chain": [],
        }
    chain, _cost = found
    largest_hop = max(
        (
            _pair_hop(
                chain[index - 1]["joint_positions_rad"],
                chain[index]["joint_positions_rad"],
            )
            for index in range(1, len(chain))
        ),
        default=0.0,
    )
    total_travel = sum(
        _pair_travel(
            chain[index - 1]["joint_positions_rad"],
            chain[index]["joint_positions_rad"],
        )
        for index in range(1, len(chain))
    )
    if start_joint_positions_rad is not None and chain:
        total_travel += _pair_travel(
            start_joint_positions_rad, chain[0]["joint_positions_rad"]
        )

    # What the previous greedy rule would have produced, for comparison only.
    greedy: list[Mapping[str, Any]] = []
    reference = list(start_joint_positions_rad or [])
    for rows in per_phase:
        pick = (
            min(rows, key=lambda row: _pair_hop(row["joint_positions_rad"], reference))
            if reference
            else rows[0]
        )
        greedy.append(pick)
        reference = list(pick["joint_positions_rad"])
    greedy_hop = max(
        (
            _pair_hop(
                greedy[index - 1]["joint_positions_rad"],
                greedy[index]["joint_positions_rad"],
            )
            for index in range(1, len(greedy))
        ),
        default=0.0,
    )

    combinations = 1
    for rows in per_phase:
        combinations *= len(rows)
    return {
        "schema_version": BRANCH_CONTINUITY_SCHEMA_VERSION,
        "status": "selected",
        "combinations_represented": combinations,
        "branches_per_phase": [len(rows) for rows in per_phase],
        "selected_chain": [dict(row) for row in chain],
        "largest_single_joint_hop_rad": largest_hop,
        "total_joint_travel_rad": total_travel,
        "greedy_largest_single_joint_hop_rad": greedy_hop,
        "claim_boundary": (
            "selects_branches_for_traversability_off_sim;native_arrival_and_"
            "contact_gates_remain_the_authority"
        ),
    }


__all__ = [
    "BRANCH_CONTINUITY_SCHEMA_VERSION",
    "MAX_BRANCH_COMBINATIONS",
    "admissible_branches",
    "phase_offers_a_branch_choice",
    "select_continuous_branch_chain",
]
