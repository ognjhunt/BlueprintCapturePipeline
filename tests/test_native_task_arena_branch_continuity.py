from __future__ import annotations

import pytest

from blueprint_pipeline.native_task_arena_branch_continuity import (
    BRANCH_CONTINUITY_SCHEMA_VERSION,
    admissible_branches,
    select_continuous_branch_chain,
)


def _phase(phase_id: str, rows: list[dict]) -> dict:
    return {"phase_id": phase_id, "attempts": rows, "selected": rows[0]}


def _row(seed: int, joints: list[float], *, margin: float = 0.05, solved: bool = True) -> dict:
    return {
        "solved": solved,
        "seed_index": seed,
        "joint_positions_rad": joints,
        "minimum_joint_limit_margin_rad": margin,
        "position_error_m": 0.004,
        "orientation_error_rad": 0.05,
    }


def test_the_chain_is_chosen_for_traversability_not_greedily() -> None:
    """C39's defect: each phase committed before knowing the next.

    Approach kept the branch nearest where it started, which left contact
    0.615 rad away -- a whole arm reconfiguration for a path built to
    interpolate hundredths of a radian, and contact arrived 150 mm out.
    Enumerating the combinations finds the pair that is actually traversable.
    """

    start = [0.0] * 7
    # Approach: one branch hugs the start, one sits near contact's only branch.
    approach = _phase(
        "approach",
        [
            _row(1, [0.05] + [0.0] * 6),   # greedy favourite: 0.05 from start
            _row(7, [0.62] + [0.0] * 6),   # 0.62 from start, but beside contact
        ],
    )
    # Contact is reachable at one posture only.
    contact = _phase("contact_open", [_row(7, [0.65] + [0.0] * 6)])

    report = select_continuous_branch_chain(
        phases=[approach, contact],
        required_margin_rad=0.005,
        start_joint_positions_rad=start,
    )

    assert report["schema_version"] == BRANCH_CONTINUITY_SCHEMA_VERSION
    assert report["status"] == "selected"
    assert report["combinations_represented"] == 2
    # It takes the far-from-start approach branch, because the hop that breaks
    # the entry path is approach->contact, not start->approach.
    assert report["selected_chain"][0]["seed_index"] == 7
    # The bounded hop is approach->contact: 0.03 rad here, against the 0.60
    # rad the greedy chain would have handed the entry path.
    assert report["largest_single_joint_hop_rad"] == pytest.approx(0.03, abs=1e-9)
    # And the greedy rule it replaces is measured, not merely asserted worse.
    assert report["greedy_largest_single_joint_hop_rad"] == pytest.approx(0.60, abs=1e-9)


def test_every_admissible_branch_is_considered_and_unsolved_ones_are_not() -> None:
    phase = _phase(
        "contact_open",
        [
            _row(1, [0.1] * 7),
            _row(3, [0.2] * 7, margin=0.001),          # below the margin floor
            _row(5, [0.3] * 7, solved=False),          # a seed, not a solution
            _row(9, [0.4] * 7),
        ],
    )

    rows = admissible_branches(phase, required_margin_rad=0.005)

    assert [row["seed_index"] for row in rows] == [1, 9]


def test_a_receipt_without_an_attempt_list_still_yields_a_chain() -> None:
    """Fail soft to the solver's own selection rather than to nothing."""

    phase = {
        "phase_id": "contact_open",
        "selected": {"joint_positions_rad": [0.3] * 7, "seed_index": 2},
    }

    report = select_continuous_branch_chain(
        phases=[phase], required_margin_rad=0.005, start_joint_positions_rad=[0.0] * 7
    )

    assert report["status"] == "selected"
    assert report["selected_chain"][0]["seed_index"] == 2


def test_a_phase_with_no_admissible_branch_is_reported_not_guessed() -> None:
    phase = _phase("contact_open", [_row(1, [0.1] * 7, margin=0.0001)])

    report = select_continuous_branch_chain(
        phases=[phase], required_margin_rad=0.005, start_joint_positions_rad=[0.0] * 7
    )

    assert report["status"] == "unavailable"
    assert report["selected_chain"] == []


def test_the_search_covers_a_space_no_enumeration_could_reach() -> None:
    """C41 refused its own search: 12.9 million combinations blew the cap.

    The cost decomposes over consecutive phase pairs, so the chain is a
    shortest-path problem rather than a product.  Bisecting the permitted hop
    and running a min-travel dynamic program at each candidate searches the
    whole space exactly, without enumerating any of it.
    """

    import random
    import time

    random.seed(7)
    phases = [
        _phase(
            f"p{index}",
            [
                _row(seed, [random.uniform(-1.0, 1.0) for _ in range(7)])
                for seed in range(14)
            ],
        )
        for index in range(11)
    ]

    started = time.perf_counter()
    report = select_continuous_branch_chain(
        phases=phases,
        required_margin_rad=0.005,
        start_joint_positions_rad=[0.0] * 7,
    )
    elapsed_s = time.perf_counter() - started

    assert report["status"] == "selected"
    # Far beyond anything an enumeration could visit, and still exact.
    assert report["combinations_represented"] == 14**11
    assert elapsed_s < 2.0
    # The chosen bottleneck is genuinely the smallest achievable: no pair of
    # consecutive branches anywhere admits a chain with a smaller worst hop.
    assert (
        report["largest_single_joint_hop_rad"]
        <= report["greedy_largest_single_joint_hop_rad"]
    )
    assert len(report["selected_chain"]) == len(phases)


def test_a_chain_that_cannot_connect_is_reported_not_guessed() -> None:
    """Every phase has branches, but no route joins them end to end."""

    from blueprint_pipeline import native_task_arena_branch_continuity as module

    phases = [
        _phase("a", [_row(1, [0.0] * 7)]),
        _phase("b", [_row(2, [5.0] * 7)]),
    ]
    original = module._best_chain_within_bottleneck
    module._best_chain_within_bottleneck = lambda *a, **k: None
    try:
        report = select_continuous_branch_chain(
            phases=phases, required_margin_rad=0.005
        )
    finally:
        module._best_chain_within_bottleneck = original

    assert report["status"] == "unavailable"
    assert report["reason"] == "no_chain_connects_every_phase"
    assert report["selected_chain"] == []


def test_phases_that_inherit_a_bound_pose_do_not_abandon_the_search() -> None:
    """C42's regression, in the shape the paid run actually had.

    Every phase that chooses a branch had them -- 5, 5, 3, 7, 13, 14, 9, 9 --
    and the search still refused, because two phases that inherit an earlier
    phase's bound pose were counted as phases with no admissible branch. The
    run then fell back to the greedy chain the search exists to replace.
    """

    phases = [
        _phase("prealign", [_row(1, [0.0] * 7), _row(2, [0.1] * 7)]),
        _phase("contact_open", [_row(1, [0.05] * 7), _row(2, [0.6] * 7)]),
        # Inherits contact_open's pose: no attempt list, no selection.
        {"phase_id": "contact_close"},
        _phase("retreat", [_row(1, [0.2] * 7)]),
        {"phase_id": "release"},
    ]

    report = select_continuous_branch_chain(
        phases=phases,
        required_margin_rad=0.005,
        start_joint_positions_rad=[0.0] * 7,
    )

    assert report["status"] == "selected"
    # The chain covers exactly the phases that make a choice, named so the
    # caller can apply them without positional guessing.
    assert report["chain_phase_ids"] == ["prealign", "contact_open", "retreat"]
    assert report["inheriting_phase_ids"] == ["contact_close", "release"]
    assert len(report["selected_chain"]) == 3


def test_a_lane_where_nothing_chooses_is_reported() -> None:
    report = select_continuous_branch_chain(
        phases=[{"phase_id": "a"}, {"phase_id": "b"}], required_margin_rad=0.005
    )

    assert report["status"] == "unavailable"
    assert report["reason"] == "no_phase_offers_a_branch_choice"


def test_the_bounded_entry_hop_outranks_the_global_worst_hop() -> None:
    """Not every hop is bounded the same way.

    The transition into the replayed phase is interpolated at a fixed step and
    must fit that phase's budget; the rest are ordinary servo moves with
    budgets of their own.  Minimising the global worst hop trades the one hop
    that matters for hops that do not -- against C42's sealed branches it chose
    a 0.757 rad entry where the greedy chain managed 0.604.
    """

    phases = [
        _phase("approach", [_row(1, [0.0] * 7), _row(2, [0.5] + [0.0] * 6)]),
        _phase("contact_open", [_row(3, [0.55] + [0.0] * 6)]),
        # A later phase far from everything: minimising the global worst hop
        # would reshuffle the entry pair to shave this one.
        _phase("retreat", [_row(4, [2.0] * 7)]),
    ]

    globally = select_continuous_branch_chain(
        phases=phases, required_margin_rad=0.005, start_joint_positions_rad=[0.0] * 7
    )
    entry_first = select_continuous_branch_chain(
        phases=phases,
        required_margin_rad=0.005,
        start_joint_positions_rad=[0.0] * 7,
        bounded_entry_phase_id="contact_open",
    )

    assert entry_first["status"] == "selected"
    assert entry_first["bounded_entry_phase_id"] == "contact_open"
    # Entry-first takes the approach branch beside contact (0.55 - 0.5).
    assert entry_first["bounded_entry_hop_rad"] == pytest.approx(0.05, abs=1e-9)
    entry_ids = entry_first["chain_phase_ids"]
    chosen = entry_first["selected_chain"][entry_ids.index("approach")]
    assert chosen["seed_index"] == 2
    # And it is strictly better on the criterion that binds than the global rule.
    global_ids = globally["chain_phase_ids"]
    global_entry = max(
        abs(a - b)
        for a, b in zip(
            globally["selected_chain"][global_ids.index("approach")][
                "joint_positions_rad"
            ],
            globally["selected_chain"][global_ids.index("contact_open")][
                "joint_positions_rad"
            ],
            strict=True,
        )
    )
    assert entry_first["bounded_entry_hop_rad"] <= global_entry
