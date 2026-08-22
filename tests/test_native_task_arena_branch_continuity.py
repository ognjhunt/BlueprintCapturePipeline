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
    assert report["combinations_evaluated"] == 2
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


def test_the_search_refuses_to_become_slow_instead_of_becoming_slow() -> None:
    phases = [_phase(f"p{index}", [_row(seed, [0.1 * seed] * 7) for seed in range(9)])
              for index in range(6)]

    report = select_continuous_branch_chain(
        phases=phases, required_margin_rad=0.005, max_combinations=1000
    )

    assert report["status"] == "unavailable"
    assert "branch_combinations_exceed_cap" in report["reason"]


def test_each_phase_is_held_to_its_own_margin_floor() -> None:
    """C40 emptied the search by applying contact's floor to every phase.

    Prealign's five solved branches all sit near zero margin because prealign
    never had to clear contact's 0.005 rad requirement.  Judging them by it
    discarded all of them, so the run abandoned the search and fell back to the
    greedy chain this exists to replace.  The solver seals the floor it
    actually enforced per phase; use that.
    """

    prealign = {
        "phase_id": "prealign",
        "required_minimum_joint_limit_margin_rad": 0.0,
        "attempts": [_row(1, [0.0] * 7, margin=0.0), _row(2, [0.01] * 7, margin=0.0003)],
        "selected": _row(1, [0.0] * 7, margin=0.0),
    }
    contact = {
        "phase_id": "contact_open",
        "required_minimum_joint_limit_margin_rad": 0.005,
        "attempts": [_row(7, [0.02] * 7, margin=0.0119), _row(9, [0.9] * 7, margin=0.0001)],
        "selected": _row(7, [0.02] * 7, margin=0.0119),
    }

    # Prealign keeps both branches; contact drops the one below its own floor.
    assert len(admissible_branches(prealign, required_margin_rad=0.005)) == 2
    assert [
        row["seed_index"] for row in admissible_branches(contact, required_margin_rad=0.005)
    ] == [7]

    report = select_continuous_branch_chain(
        phases=[prealign, contact],
        required_margin_rad=0.005,
        start_joint_positions_rad=[0.0] * 7,
    )

    assert report["status"] == "selected"
    assert report["chain_phase_ids"] == ["prealign", "contact_open"]


def test_a_phase_that_inherits_a_bound_pose_is_not_a_failed_phase() -> None:
    """contact_close and release reuse an earlier pose; they do not choose.

    C40 read their empty attempt lists as phases with no admissible branch and
    abandoned the whole search over them.
    """

    contact = {
        "phase_id": "contact_open",
        "required_minimum_joint_limit_margin_rad": 0.005,
        "attempts": [_row(7, [0.02] * 7, margin=0.02)],
        "selected": _row(7, [0.02] * 7, margin=0.02),
    }
    bound = {"phase_id": "contact_close", "status": "reused_bound_pose_solution"}

    report = select_continuous_branch_chain(
        phases=[contact, bound],
        required_margin_rad=0.005,
        start_joint_positions_rad=[0.0] * 7,
    )

    assert report["status"] == "selected"
    assert report["chain_phase_ids"] == ["contact_open"]
    assert report["inherited_phase_ids"] == ["contact_close"]
    assert len(report["selected_chain"]) == 1


def test_an_empty_phase_is_named_in_the_reason() -> None:
    phases = [
        {
            "phase_id": "contact_open",
            "required_minimum_joint_limit_margin_rad": 0.5,
            "attempts": [_row(1, [0.1] * 7, margin=0.01)],
            "selected": _row(1, [0.1] * 7, margin=0.01),
        }
    ]

    report = select_continuous_branch_chain(
        phases=phases, required_margin_rad=0.005, start_joint_positions_rad=[0.0] * 7
    )

    assert report["status"] == "unavailable"
    assert report["reason"] == "phase_without_admissible_branch:contact_open"
