from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import textwrap

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_controls_worker import (
    _RigidScoringEnvironment,
    _announce_bounded_orientation_ik_progress,
    _bounded_orientation_joint_targets,
    _bounded_orientation_reference_seeds,
    _canonical_digest,
    _contact_close_sweep_minimum_force_n,
    _contact_authoritative_targets,
    _construction_global_ik_joint_targets,
    _control_execution_spec,
    _control_plan_global_ik_joint_targets,
    _dispatch_physics_admitted_jaw_variant,
    _downstream_diagnostic_request,
    _fallback_contact_open_postures,
    _input_binding_mismatches,
    _load_and_verify_manifest,
    _normalized_control_plan_for_execution,
    _parallel_jaw_equivalent_control_plan,
    _parallel_jaw_equivalent_quaternion_xyzw,
    _physics_admitted_contact_open_cell,
    _select_parallel_jaw_control_plan,
    _should_run_bounded_orientation_fallback,
    _solve_closed_contact_on_reference_branch,
    _synthetic_post_phase5_checkpoint,
    _verified_runtime_inputs,
    _with_live_physx_dls_contact_close,
    main,
)


def test_downstream_checkpoint_is_synthetic_and_selects_continuous_branch() -> None:
    plan = {
        "scripted_positive_actions": [
            {
                "phase_id": "contact_close",
                "mode": "ik_pose",
                "target_position_world_m": [1.0, 2.0, 3.0],
                "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                "arrival_tolerance_m": 0.005,
                "arrival_orientation_tolerance_rad": 0.08,
                "expected_joint_positions": {"door": 0.0, "latch": 0.0},
            },
            {
                "phase_id": "joint_path_01",
                "mode": "ik_pose",
                "target_position_world_m": [1.1, 2.0, 3.0],
                "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        ]
    }
    global_ik = {
        "phases": [
            {
                "phase_id": "contact_close",
                "attempts": [
                    {
                        "solved": False,
                        "joint_positions_rad": [0.2] * 7,
                        "position_error_m": 0.004,
                        "orientation_error_rad": 0.07,
                    },
                    {
                        "solved": False,
                        "joint_positions_rad": [0.8] * 7,
                        "position_error_m": 0.001,
                        "orientation_error_rad": 0.07,
                    },
                ],
            }
        ]
    }

    checkpoint = _synthetic_post_phase5_checkpoint(
        control_plan=plan,
        global_ik=global_ik,
        scripted_pose_joint_targets=[
            {
                "phase_id": "joint_path_01",
                "target_position_world_m": [1.1, 2.0, 3.0],
                "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                "joint_positions_rad": [0.25] * 7,
            }
        ],
        task_spec={"joint_reset_positions_rad": {"door": 0.0}},
    )

    assert checkpoint is not None
    assert checkpoint["arm_joint_positions_rad"] == [0.2] * 7
    assert checkpoint["task_joint_positions_rad"] == {
        "door": 0.0,
        "latch": 0.0,
    }
    assert checkpoint["phase5_qualified"] is False
    assert checkpoint["gripper_state"] == "closed"
    assert checkpoint["checkpoint_digest"] == _canonical_digest(
        checkpoint, field="checkpoint_digest"
    )


def test_downstream_checkpoint_refuses_missing_task_joint_state() -> None:
    assert (
        _synthetic_post_phase5_checkpoint(
            control_plan={
                "scripted_positive_actions": [
                    {
                        "phase_id": "contact_close",
                        "mode": "ik_pose",
                        "target_position_world_m": [1.0, 2.0, 3.0],
                        "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                        "arrival_tolerance_m": 0.005,
                        "arrival_orientation_tolerance_rad": 0.08,
                        "hold_solved_arm_joint_positions_rad": [0.2] * 7,
                    },
                    {
                        "phase_id": "joint_path_01",
                        "mode": "ik_pose",
                        "target_position_world_m": [1.1, 2.0, 3.0],
                        "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                ]
            },
            global_ik={"phases": []},
            scripted_pose_joint_targets=[],
            task_spec={},
        )
        is None
    )


def test_downstream_checkpoint_refuses_contact_pose_outside_unchanged_gate() -> None:
    plan = {
        "scripted_positive_actions": [
            {
                "phase_id": "contact_close",
                "mode": "ik_pose",
                "target_position_world_m": [1.0, 2.0, 3.0],
                "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                "arrival_tolerance_m": 0.005,
                "arrival_orientation_tolerance_rad": 0.08,
                "expected_joint_positions": {"door": 0.0},
            },
            {
                "phase_id": "joint_path_01",
                "mode": "ik_pose",
                "target_position_world_m": [1.1, 2.0, 3.0],
                "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        ]
    }
    assert (
        _synthetic_post_phase5_checkpoint(
            control_plan=plan,
            global_ik={
                "phases": [
                    {
                        "phase_id": "contact_close",
                        "attempts": [
                            {
                                "solved": False,
                                "joint_positions_rad": [0.2] * 7,
                                "position_error_m": 0.0044,
                                "orientation_error_rad": 0.087,
                            }
                        ],
                    }
                ]
            },
            scripted_pose_joint_targets=[],
            task_spec={"joint_reset_positions_rad": {"door": 0.0}},
        )
        is None
    )


def test_contact_open_fallback_preserves_variant_specific_scoring_targets() -> None:
    def _preflight(*rows):
        return {
            "phases": [
                {
                    "phase_id": "contact_open",
                    "attempts": list(rows),
                    "selected": None,
                }
            ]
        }

    duplicate = {
        "solved": False,
        "seed_index": 0,
        "joint_positions_rad": [0.1] * 7,
        "position_error_m": 0.0043,
        "orientation_error_rad": 0.086,
    }
    nominal = [-2**-0.5, 0.0, 0.0, 2**-0.5]
    control_plan = {
        "scripted_positive_actions": [
            {
                "phase_id": "contact_open",
                "mode": "ik_pose",
                "target_position_world_m": [1.0, 2.0, 3.0],
                "target_quaternion_world_xyzw": nominal,
                "arrival_tolerance_m": 0.005,
                "arrival_orientation_tolerance_rad": 0.08,
            }
        ],
        "plan_digest": "sha256:" + "a" * 64,
    }
    rows = _fallback_contact_open_postures(
        {
            "variants": [
                {
                    "variant_id": "normalized_nominal",
                    "global_ik_preflight": _preflight(duplicate),
                },
                {
                    "variant_id": "parallel_jaw_equivalent",
                    "global_ik_preflight": _preflight(
                        duplicate,
                        {
                            **duplicate,
                            "seed_index": 1,
                            "joint_positions_rad": [0.2] * 7,
                        },
                    ),
                },
            ]
        },
        control_plan=control_plan,
    )

    # The same joint vector under different jaw conventions is retained twice:
    # one physical reading can be compared with either quaternion, but the
    # matrix cell must preserve which target it is grading.
    assert len(rows) == 3
    assert {row["joint_positions_rad"][0] for row in rows} == {0.1, 0.2}
    assert rows[0]["variant_id"] == "normalized_nominal"
    by_variant = {row["variant_id"]: row for row in rows}
    assert by_variant["normalized_nominal"][
        "authoritative_target_quaternion_world_xyzw"
    ] == pytest.approx(nominal)
    assert by_variant["parallel_jaw_equivalent"][
        "authoritative_target_quaternion_world_xyzw"
    ] == pytest.approx([0.0, 2**-0.5, 2**-0.5, 0.0])


def test_contact_open_fallback_accepts_normalized_none_arrival_override() -> None:
    """Validated plans use None when no separate arrival target was authored."""

    target = [1.0, 2.0, 3.0]
    quaternion = [0.0, 0.0, 0.0, 1.0]
    rows = _fallback_contact_open_postures(
        {
            "variants": [
                {
                    "variant_id": "normalized_nominal",
                    "global_ik_preflight": {
                        "phases": [
                            {
                                "phase_id": "contact_open",
                                "selected": None,
                                "attempts": [
                                    {
                                        "solved": False,
                                        "joint_positions_rad": [0.1] * 7,
                                        "position_error_m": 0.004,
                                        "orientation_error_rad": 0.07,
                                    }
                                ],
                            }
                        ]
                    },
                }
            ]
        },
        control_plan={
            "scripted_positive_actions": [
                {
                    "phase_id": "contact_open",
                    "mode": "ik_pose",
                    "target_position_world_m": target,
                    # This is the normalized representation of an omitted
                    # optional override, not an invalid scientific target.
                    "arrival_target_position_world_m": None,
                    "target_quaternion_world_xyzw": quaternion,
                    "arrival_target_quaternion_world_xyzw": None,
                }
            ],
            "plan_digest": "sha256:" + "a" * 64,
        },
    )

    assert len(rows) == 1
    assert rows[0]["candidate_command_target_position_world_m"] == target
    assert rows[0]["authoritative_target_position_world_m"] == target
    assert rows[0]["candidate_command_target_quaternion_world_xyzw"] == quaternion
    assert rows[0]["authoritative_target_quaternion_world_xyzw"] == quaternion


@pytest.mark.parametrize("explicit_none", [False, True])
def test_sweep_authority_uses_command_target_without_arrival_override(
    explicit_none: bool,
) -> None:
    """Missing and normalized-null overrides have identical gate semantics."""

    target = [1.0, 2.0, 3.0]
    quaternion = [0.0, 0.0, 0.0, 1.0]
    row = {
        "target_position_world_m": target,
        "target_quaternion_world_xyzw": quaternion,
    }
    if explicit_none:
        row["arrival_target_position_world_m"] = None
        row["arrival_target_quaternion_world_xyzw"] = None

    authoritative_position, authoritative_quaternion = (
        _contact_authoritative_targets(row)
    )

    assert authoritative_position == target
    assert authoritative_quaternion == quaternion


@pytest.mark.parametrize(
    ("field", "malformed"),
    [
        ("arrival_target_position_world_m", "invalid"),
        ("arrival_target_position_world_m", [1.0, 2.0]),
        ("arrival_target_position_world_m", [1.0, float("nan"), 3.0]),
        ("arrival_target_quaternion_world_xyzw", "invalid"),
        ("arrival_target_quaternion_world_xyzw", [0.0, 0.0]),
        ("arrival_target_quaternion_world_xyzw", [0.0, 0.0, 0.0, 0.0]),
    ],
)
def test_sweep_authority_refuses_malformed_non_null_overrides(
    field: str,
    malformed: object,
) -> None:
    row = {
        "target_position_world_m": [1.0, 2.0, 3.0],
        "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
        field: malformed,
    }

    with pytest.raises(
        RuntimeError, match="^contact_authoritative_target_invalid$"
    ):
        _contact_authoritative_targets(row)


def test_bounded_orientation_seeds_prioritize_bound_physics_reference() -> None:
    c75 = [
        1.8153258562,
        0.8945093155,
        -1.6013997793,
        -2.5417878628,
        -2.8766772747,
        2.3462493420,
        -0.8545385003,
    ]
    seeds = _bounded_orientation_reference_seeds(
        control_plan={
            "bounded_orientation_reference_joint_positions_rad": c75,
            "scripted_positive_actions": [
                {
                    "phase_id": "contact_close",
                }
            ]
        },
        jaw_selection={
            "variants": [
                {
                    "scripted_pose_joint_targets": [
                        {
                            "phase_id": "contact_open",
                            "joint_positions_rad": [-2.0] * 7,
                        }
                    ]
                }
            ]
        },
        sweep={
            "cells": [
                {
                    "measured_distance_to_target_m": 0.0044,
                    "measured_orientation_error_rad": 0.087,
                    "commanded_joint_positions_rad": [-1.0] * 7,
                }
            ]
        },
    )

    assert seeds[0] == c75
    assert seeds[1:] == [[-1.0] * 7, [-2.0] * 7]


def test_bounded_orientation_ik_progress_is_persisted_and_announced(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    progress = {
        "schema_version": (
            "native_task_arena_bounded_orientation_solve_progress.v1"
        ),
        "event": "phase_solve_started",
        "candidate_index": 6,
        "completed_candidate_count": 6,
        "total_candidate_count": 40,
        "phase_id": "contact_close",
        "phase_solution_returned": None,
        "solve_call_count": 9,
        "reason": None,
    }

    _announce_bounded_orientation_ik_progress(
        output_root=tmp_path,
        progress=progress,
    )

    retained = json.loads(
        (
            tmp_path
            / "contact_open_bounded_orientation_ik.progress.v1.json"
        ).read_text(encoding="utf-8")
    )
    assert retained["event"] == "phase_solve_started"
    assert retained["candidate_index"] == 6
    assert retained["solve_call_count"] == 9
    assert retained["result_digest"] == _canonical_digest(
        retained,
        field="result_digest",
    )
    assert not list(tmp_path.glob(".*.tmp"))
    assert capsys.readouterr().out.strip() == (
        "BLUEPRINT_BOUNDED_ORIENTATION_IK_PROGRESS:"
        "event=phase_solve_started:candidate=7/40:phase=contact_close:"
        "solve_calls=9:solution_returned=None:reason=None"
    )


@pytest.mark.parametrize(
    "control_plan, blocker",
    [
        ({}, "bounded_orientation_reference_missing"),
        (
            {"bounded_orientation_reference_joint_positions_rad": [0.0] * 6},
            "bounded_orientation_reference_invalid",
        ),
    ],
)
def test_bounded_orientation_seeds_fail_closed_without_bound_reference(
    control_plan: dict[str, object], blocker: str
) -> None:
    with pytest.raises(RuntimeError, match=blocker):
        _bounded_orientation_reference_seeds(
            control_plan=control_plan,
            jaw_selection={"variants": []},
            sweep={"cells": []},
        )


def test_bounded_orientation_joints_do_not_rewrite_authoritative_targets() -> None:
    nominal = [-2**-0.5, 0.0, 0.0, 2**-0.5]
    biased = [0.68, -0.02, 0.02, -0.73]
    plan = {
        "scripted_positive_actions": [
            {
                "phase_id": phase_id,
                "mode": "ik_pose",
                "target_position_world_m": [1.0, 2.0, z],
                "target_quaternion_world_xyzw": nominal,
            }
            for phase_id, z in (("contact_open", 3.0), ("contact_close", 3.1))
        ]
    }

    targets = _bounded_orientation_joint_targets(
        control_plan=plan,
        admitted_cell={
            "commanded_joint_positions_rad": [0.1] * 7,
            "candidate_command_target_quaternion_world_xyzw": biased,
            "bounded_orientation_candidate": {
                "open_command_quaternion_world_xyzw": biased,
                "close_command_quaternion_world_xyzw": biased,
                "close_joint_positions_rad": [0.2] * 7,
            },
        },
    )

    assert [row["phase_id"] for row in targets] == [
        "contact_open",
        "contact_close",
    ]
    assert [row["joint_positions_rad"] for row in targets] == [
        [0.1] * 7,
        [0.2] * 7,
    ]
    assert all(row["target_quaternion_world_xyzw"] == nominal for row in targets)
    assert all(row["target_quaternion_world_xyzw"] != biased for row in targets)


def test_bounded_orientation_runs_only_after_both_jaw_variants_miss() -> None:
    sweep = {
        "cells": [
            {"variant_id": "normalized_nominal"},
            {"variant_id": "parallel_jaw_equivalent"},
        ]
    }

    assert _should_run_bounded_orientation_fallback(
        live_dls_contact_fallback=True,
        admitted_open_cell=None,
        sweep=sweep,
    )
    assert not _should_run_bounded_orientation_fallback(
        live_dls_contact_fallback=True,
        admitted_open_cell={"pose_gate_passed": True},
        sweep=sweep,
    )
    assert not _should_run_bounded_orientation_fallback(
        live_dls_contact_fallback=True,
        admitted_open_cell=None,
        sweep={"cells": [{"variant_id": "normalized_nominal"}]},
    )
    assert not _should_run_bounded_orientation_fallback(
        live_dls_contact_fallback=False,
        admitted_open_cell=None,
        sweep=sweep,
    )


def test_c82_sealed_dual_variant_miss_enters_bounded_fallback() -> None:
    common = {
        "commanded_joint_positions_rad": [0.1] * 7,
        "joint_tracking_error_rad": 0.001,
        "joint_limit_violation": False,
        "robot_collision_failure": False,
        "scene_collision_failure": False,
        "task_contact_active": False,
    }
    c82_sweep = {
        "source_result_digest": (
            "sha256:e73a483c48eb87f525c9f7adb53c5f401c4278cc72a9c1771ae6296484b80a0c"
        ),
        "cells": [
            {
                **common,
                "variant_id": "normalized_nominal",
                "measured_distance_to_target_m": 0.00436435,
                "measured_orientation_error_rad": 0.0870456,
            },
            {
                **common,
                "variant_id": "parallel_jaw_equivalent",
                "measured_distance_to_target_m": 0.00444000,
                "measured_orientation_error_rad": 0.0887271,
            },
        ],
    }

    admitted = _physics_admitted_contact_open_cell(
        c82_sweep,
        position_tolerance_m=0.005,
        orientation_tolerance_rad=0.08,
    )
    assert admitted is None
    assert _should_run_bounded_orientation_fallback(
        live_dls_contact_fallback=True,
        admitted_open_cell=admitted,
        sweep=c82_sweep,
    )


def test_physics_admitted_parallel_cell_switches_plan_targets_and_preflight() -> None:
    nominal = [-2**-0.5, 0.0, 0.0, 2**-0.5]
    equivalent = [0.0, 2**-0.5, 2**-0.5, 0.0]
    normalized = {
        "schema_version": "adp_task_control_plan.v1",
        "scripted_positive_actions": [
            {
                "phase_id": phase_id,
                "mode": "ik_pose",
                "target_position_world_m": [1.0, 2.0, 3.0],
                "target_quaternion_world_xyzw": nominal,
            }
            for phase_id in ("contact_open", "contact_close")
        ],
        "plan_digest": "sha256:" + "a" * 64,
    }
    parallel_targets = [
        {
            "phase_id": "contact_open",
            "target_position_world_m": [1.0, 2.0, 3.0],
            "target_quaternion_world_xyzw": equivalent,
            "joint_positions_rad": [0.2] * 7,
        }
    ]
    parallel_preflight = {
        "status": "partial",
        "phases": [{"phase_id": "contact_open"}],
    }

    plan, targets, preflight, receipt = (
        _dispatch_physics_admitted_jaw_variant(
            normalized_control_plan=normalized,
            selected_control_plan=normalized,
            scripted_pose_joint_targets=[],
            controls_global_ik={"status": "nominal"},
            jaw_selection={
                "selected_variant_id": "normalized_nominal",
                "variants": [
                    {
                        "variant_id": "parallel_jaw_equivalent",
                        "scripted_pose_joint_targets": parallel_targets,
                        "global_ik_preflight": parallel_preflight,
                    }
                ],
            },
            admitted_open_cell={
                "variant_id": "parallel_jaw_equivalent"
            },
        )
    )

    assert receipt["variant_switched"] is True
    assert receipt["adopted_variant_id"] == "parallel_jaw_equivalent"
    assert all(
        row["target_quaternion_world_xyzw"] == pytest.approx(equivalent)
        for row in plan["scripted_positive_actions"]
    )
    assert plan["plan_digest"] == _canonical_digest(
        plan, field="plan_digest"
    )
    assert targets == parallel_targets
    assert preflight == parallel_preflight
    assert receipt["selected_variant_control_plan_digest"] == plan[
        "plan_digest"
    ]


def test_contact_open_physics_adoption_keeps_every_gate_fail_closed() -> None:
    base = {
        "measured_distance_to_target_m": 0.004,
        "measured_orientation_error_rad": 0.07,
        "commanded_joint_positions_rad": [0.2] * 7,
        "joint_tracking_error_rad": 0.001,
        "joint_limit_violation": False,
        "robot_collision_failure": False,
        "scene_collision_failure": False,
        "task_contact_active": False,
    }
    selected = _physics_admitted_contact_open_cell(
        {"cells": [base]},
        position_tolerance_m=0.005,
        orientation_tolerance_rad=0.08,
    )
    assert selected is not None
    assert selected["pose_gate_passed"] is True

    parallel = {
        **base,
        "variant_id": "parallel_jaw_equivalent",
        "measured_distance_to_target_m": 0.003467,
        "measured_orientation_error_rad": 0.06678,
    }
    nominal_wrong_target = {
        **base,
        "variant_id": "normalized_nominal",
        "measured_orientation_error_rad": 3.14159,
    }
    selected_variant = _physics_admitted_contact_open_cell(
        {"cells": [nominal_wrong_target, parallel]},
        position_tolerance_m=0.005,
        orientation_tolerance_rad=0.08,
    )
    assert selected_variant is not None
    assert selected_variant["variant_id"] == "parallel_jaw_equivalent"

    high_margin = {
        **base,
        "minimum_joint_limit_margin_rad": 0.2,
        "measured_distance_to_target_m": 0.0049,
        "measured_orientation_error_rad": 0.079,
    }
    low_margin_better_pose = {
        **base,
        "minimum_joint_limit_margin_rad": 0.001,
        "measured_distance_to_target_m": 0.001,
        "measured_orientation_error_rad": 0.01,
    }
    selected_margin = _physics_admitted_contact_open_cell(
        {"cells": [low_margin_better_pose, high_margin]},
        position_tolerance_m=0.005,
        orientation_tolerance_rad=0.08,
    )
    assert selected_margin is not None
    assert selected_margin["minimum_joint_limit_margin_rad"] == pytest.approx(
        0.2
    )

    for failed_field, failed_value in (
        ("measured_distance_to_target_m", 0.0051),
        ("measured_orientation_error_rad", 0.0801),
        ("joint_limit_violation", True),
        ("robot_collision_failure", True),
        ("scene_collision_failure", True),
        ("task_contact_active", True),
    ):
        failed = {**base, failed_field: failed_value}
        assert (
            _physics_admitted_contact_open_cell(
                {"cells": [failed]},
                position_tolerance_m=0.005,
                orientation_tolerance_rad=0.08,
            )
            is None
        )


def test_close_sweep_uses_task_contact_force_before_plan_row_is_compiled() -> None:
    assert _contact_close_sweep_minimum_force_n(
        contact_close_row={"bilateral_task_contact_minimum_force_n": None},
        task_state_binding={"task_contact_minimum_force_n": 0.5},
    ) == pytest.approx(0.5)


def test_close_sweep_refuses_a_contact_force_gate_mismatch() -> None:
    with pytest.raises(
        RuntimeError, match="native_task_controls_contact_force_mismatch"
    ):
        _contact_close_sweep_minimum_force_n(
            contact_close_row={"bilateral_task_contact_minimum_force_n": 0.4},
            task_state_binding={"task_contact_minimum_force_n": 0.5},
        )


def test_physx_dls_close_adoption_removes_joint_override_and_restores_authored_target() -> None:
    plan = {
        "scripted_positive_actions": [
            {
                "phase_id": "contact_close",
                "mode": "ik_pose",
                "target_position_world_m": [1.0, 1.986, 3.0],
                "arrival_target_position_world_m": [1.0, 2.0, 3.0],
                "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                "hold_solved_arm_joint_positions_rad": [0.2] * 7,
            }
        ],
        "plan_digest": "",
    }
    plan["plan_digest"] = _canonical_digest(plan, field="plan_digest")

    derived, receipt = _with_live_physx_dls_contact_close(
        control_plan=plan,
        preferred_posture_joint_positions_rad=[0.3] * 7,
    )

    row = derived["scripted_positive_actions"][0]
    assert receipt["status"] == "applied"
    assert row["target_position_world_m"] == [1.0, 2.0, 3.0]
    assert row["arrival_target_position_world_m"] == [1.0, 2.0, 3.0]
    assert row["hold_solved_arm_joint_positions_rad"] is None
    assert row["physx_dls_preferred_posture_joint_positions_rad"] == [
        0.3
    ] * 7
    assert derived["plan_digest"] != plan["plan_digest"]


def test_physx_dls_close_treats_null_arrival_override_as_command_target() -> None:
    plan = {
        "scripted_positive_actions": [
            {
                "phase_id": "contact_close",
                "mode": "ik_pose",
                "target_position_world_m": [1.0, 1.986, 3.0],
                "arrival_target_position_world_m": None,
                "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                "arrival_target_quaternion_world_xyzw": None,
                "hold_solved_arm_joint_positions_rad": [0.2] * 7,
            }
        ],
        "plan_digest": "",
    }
    plan["plan_digest"] = _canonical_digest(plan, field="plan_digest")

    derived, receipt = _with_live_physx_dls_contact_close(
        control_plan=plan,
        preferred_posture_joint_positions_rad=[0.3] * 7,
    )

    row = derived["scripted_positive_actions"][0]
    assert receipt["status"] == "applied"
    assert receipt["authored_dls_target_position_world_m"] == [
        1.0,
        1.986,
        3.0,
    ]
    assert row["target_position_world_m"] == [1.0, 1.986, 3.0]
    assert row["hold_solved_arm_joint_positions_rad"] is None


def test_closed_contact_calibration_keeps_the_measured_ik_branch() -> None:
    class _Servo:
        def __init__(self) -> None:
            self.kwargs = None

        def solve_grasp_target_multistart(self, **kwargs):
            self.kwargs = kwargs
            return {
                "selected": {
                    "joint_positions_rad": [0.25] * 7,
                    "minimum_joint_limit_margin_rad": 0.006,
                }
            }

    servo = _Servo()
    reference = [0.1] * 7
    result = _solve_closed_contact_on_reference_branch(
        servo=servo,
        contact_close_row={
            "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
            "arrival_tolerance_m": 0.005,
            "arrival_orientation_tolerance_rad": 0.08,
            "max_joint_setpoint_lead_rad": 1.0,
        },
        target_position_world_m=[1.0, 2.0, 3.0],
        reference_joint_positions_rad=reference,
    )

    assert result == [0.25] * 7
    assert servo.kwargs["preferred_seeds"] == [reference]
    assert servo.kwargs["reference_joint_positions_rad"] == reference
    assert servo.kwargs[
        "preferred_minimum_joint_limit_margin_rad"
    ] == pytest.approx(0.005)
    assert servo.kwargs[
        "required_minimum_joint_limit_margin_rad"
    ] == pytest.approx(0.005)


def test_closed_contact_calibration_refuses_a_nonlocal_branch_jump() -> None:
    class _Servo:
        def solve_grasp_target_multistart(self, **kwargs):
            del kwargs
            return {"selected": {"joint_positions_rad": [2.0] * 7}}

    result = _solve_closed_contact_on_reference_branch(
        servo=_Servo(),
        contact_close_row={
            "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
            "arrival_tolerance_m": 0.005,
            "arrival_orientation_tolerance_rad": 0.08,
            "max_joint_setpoint_lead_rad": 1.0,
        },
        target_position_world_m=[1.0, 2.0, 3.0],
        reference_joint_positions_rad=[0.0] * 7,
    )

    assert result is None


def test_controls_multistart_solves_missing_exact_pose_and_reuses_duplicates() -> None:
    class _Servo:
        def __init__(self):
            self.calls = []

        def read_arm_joint_positions(self):
            return [0.0] * 7

        def solve_grasp_target_multistart(self, **kwargs):
            self.calls.append(kwargs)
            joints = [float(kwargs["target_position_world_m"][0])] * 7
            return {
                "solved": True,
                "selected": {
                    "solved": True,
                    "joint_positions_rad": joints,
                },
                "attempts": [],
            }

    servo = _Servo()
    bound = [
        {
            "phase_id": "approach",
            "target_position_world_m": [1.0, 0.0, 0.0],
            "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
            "joint_positions_rad": [0.1] * 7,
        }
    ]
    targets, receipt = _control_plan_global_ik_joint_targets(
        servo=servo,
        control_plan={
            "scripted_positive_actions": [
                {
                    "phase_id": "prealign",
                    "mode": "ik_pose",
                    "position_only_arrival": True,
                    "target_position_world_m": [0.5, 0.0, 0.0],
                    "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "arrival_tolerance_m": 0.02,
                    "arrival_orientation_tolerance_rad": None,
                },
                {
                    "phase_id": "approach",
                    "mode": "ik_pose",
                    "target_position_world_m": [1.0, 0.0, 0.0],
                    "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                {
                    "phase_id": "contact_open",
                    "mode": "ik_pose",
                    "target_position_world_m": [2.0, 0.0, 0.0],
                    "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "arrival_tolerance_m": 0.005,
                    "arrival_orientation_tolerance_rad": 0.08,
                },
                {
                    "phase_id": "contact_close",
                    "mode": "ik_pose",
                    "target_position_world_m": [2.0, 0.0, 0.0],
                    "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            ]
        },
        bound_targets=bound,
        reference_seeds=[[0.5] * 7],
    )

    assert len(servo.calls) == 2
    assert servo.calls[0]["preferred_seeds"][0] == [0.0] * 7
    assert servo.calls[0]["position_tolerance_m"] == pytest.approx(0.02)
    assert servo.calls[0]["orientation_tolerance_rad"] == pytest.approx(0.08)
    assert servo.calls[1]["preferred_seeds"][0] == [0.1] * 7
    assert servo.calls[1]["position_tolerance_m"] == pytest.approx(0.005)
    assert servo.calls[1]["orientation_tolerance_rad"] == pytest.approx(0.08)
    assert servo.calls[1][
        "required_minimum_joint_limit_margin_rad"
    ] == pytest.approx(0.005)
    assert [row["phase_id"] for row in targets] == [
        "approach",
        "prealign",
        "contact_open",
    ]
    assert targets[1]["joint_positions_rad"] == [0.5] * 7
    assert targets[-1]["joint_positions_rad"] == [2.0] * 7
    assert receipt["status"] == "all_unique_poses_solved_or_bound"
    assert [row.get("status") for row in receipt["phases"]] == [
        None,
        "reused_bound_pose_solution",
        None,
        "reused_bound_pose_solution",
    ]
    assert receipt["phases"][0]["position_only_arrival_gate"] is True
    assert receipt["phases"][0][
        "full_pose_prepositioning_tolerance_rad"
    ] == pytest.approx(0.08)


def test_controls_approach_uses_path_margin_before_exact_contact() -> None:
    class _Servo:
        def __init__(self):
            self.calls = []

        def read_arm_joint_positions(self):
            return [0.0] * 7

        def solve_grasp_target_multistart(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "solved": True,
                "selected": {
                    "solved": True,
                    "joint_positions_rad": [0.1] * 7,
                },
                "attempts": [],
            }

    servo = _Servo()
    _control_plan_global_ik_joint_targets(
        servo=servo,
        control_plan={
            "scripted_positive_actions": [
                {
                    "phase_id": "approach",
                    "mode": "ik_pose",
                    "target_position_world_m": [1.0, 0.0, 0.0],
                    "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "arrival_tolerance_m": 0.02,
                    "arrival_orientation_tolerance_rad": 0.08,
                },
                {
                    "phase_id": "contact_open",
                    "mode": "ik_pose",
                    "target_position_world_m": [2.0, 0.0, 0.0],
                    "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "arrival_tolerance_m": 0.005,
                    "arrival_orientation_tolerance_rad": 0.08,
                },
            ]
        },
        bound_targets=[],
        reference_seeds=[],
    )

    assert servo.calls[0][
        "preferred_minimum_joint_limit_margin_rad"
    ] == pytest.approx(0.04)
    assert servo.calls[0][
        "required_minimum_joint_limit_margin_rad"
    ] == pytest.approx(0.0)
    assert servo.calls[1][
        "preferred_minimum_joint_limit_margin_rad"
    ] == pytest.approx(0.05)
    assert servo.calls[1][
        "required_minimum_joint_limit_margin_rad"
    ] == pytest.approx(0.005)


def test_parallel_jaw_equivalent_preserves_approach_and_reverses_jaw() -> None:
    nominal = [-2**-0.5, 0.0, 0.0, 2**-0.5]

    equivalent = _parallel_jaw_equivalent_quaternion_xyzw(nominal)
    plan = {
        "schema_version": "adp_task_control_plan.v1",
        "scripted_positive_actions": [
            {
                "phase_id": "contact_open",
                "mode": "ik_pose",
                "target_position_world_m": [1.0, 2.0, 3.0],
                "target_quaternion_world_xyzw": nominal,
            }
        ],
        "plan_digest": "sha256:" + "a" * 64,
    }
    derived = _parallel_jaw_equivalent_control_plan(plan)
    evidence = derived["runtime_control_variant_equivalence"]

    assert equivalent == pytest.approx([0.0, 2**-0.5, 2**-0.5, 0.0])
    assert evidence["approach_axis_dot"] == pytest.approx(1.0)
    assert evidence["jaw_axis_dot"] == pytest.approx(-1.0)
    assert derived["scripted_positive_actions"][0][
        "target_position_world_m"
    ] == [1.0, 2.0, 3.0]
    assert derived["plan_digest"] == _canonical_digest(
        derived, field="plan_digest"
    )


def test_controls_selects_fully_solved_parallel_jaw_branch_with_more_margin() -> None:
    class _Servo:
        def read_arm_joint_positions(self):
            return [0.0] * 7

        def solve_grasp_target_multistart(self, **kwargs):
            quaternion = kwargs["target_grasp_frame_quaternion_world_xyzw"]
            equivalent = quaternion[1] > 0.5
            margin = 0.2 if equivalent else 0.002
            return {
                "solved": True,
                "selected": {
                    "solved": True,
                    "joint_positions_rad": [margin] * 7,
                    "minimum_joint_limit_margin_rad": margin,
                },
                "attempts": [],
            }

    nominal = [-2**-0.5, 0.0, 0.0, 2**-0.5]
    plan = {
        "schema_version": "adp_task_control_plan.v1",
        "scripted_positive_actions": [
            {
                "phase_id": "contact_open",
                "mode": "ik_pose",
                "target_position_world_m": [1.0, 2.0, 3.0],
                "target_quaternion_world_xyzw": nominal,
                "arrival_tolerance_m": 0.005,
                "arrival_orientation_tolerance_rad": 0.08,
            },
            {
                "phase_id": "contact_close",
                "mode": "ik_pose",
                "target_position_world_m": [1.0, 2.0, 3.0],
                "target_quaternion_world_xyzw": nominal,
                "arrival_tolerance_m": 0.005,
                "arrival_orientation_tolerance_rad": 0.08,
            },
        ],
        "plan_digest": "sha256:" + "a" * 64,
    }

    selected_plan, targets, receipt = _select_parallel_jaw_control_plan(
        servo=_Servo(),
        control_plan=plan,
        construction_bound_targets=[],
        reference_seeds=[[0.5] * 7],
    )

    assert receipt["selected_variant_id"] == "parallel_jaw_equivalent"
    assert receipt[
        "selected_contact_open_minimum_joint_limit_margin_rad"
    ] == pytest.approx(0.2)
    assert selected_plan["runtime_control_variant"].startswith("parallel_jaw")
    assert targets[0]["joint_positions_rad"] == pytest.approx([0.2] * 7)
    assert receipt["physics_steps_performed"] == 0


def test_controls_selects_contact_branch_when_non_contact_pose_is_unsolved() -> None:
    class _Servo:
        def read_arm_joint_positions(self):
            return [0.0] * 7

        def solve_grasp_target_multistart(self, **kwargs):
            position = kwargs["target_position_world_m"]
            if position == [0.0, 0.0, 0.0]:
                return {"solved": False, "selected": None, "attempts": []}
            quaternion = kwargs["target_grasp_frame_quaternion_world_xyzw"]
            equivalent = quaternion[1] > 0.5
            margin = 0.2 if equivalent else 0.02
            return {
                "solved": True,
                "selected": {
                    "solved": True,
                    "joint_positions_rad": [margin] * 7,
                    "minimum_joint_limit_margin_rad": margin,
                },
                "attempts": [],
            }

    nominal = [-2**-0.5, 0.0, 0.0, 2**-0.5]
    plan = {
        "schema_version": "adp_task_control_plan.v1",
        "scripted_positive_actions": [
            {
                "phase_id": "prealign",
                "mode": "ik_pose",
                "target_position_world_m": [0.0, 0.0, 0.0],
                "target_quaternion_world_xyzw": nominal,
                "arrival_tolerance_m": 0.02,
                "position_only_arrival": True,
            },
            {
                "phase_id": "contact_open",
                "mode": "ik_pose",
                "target_position_world_m": [1.0, 2.0, 3.0],
                "target_quaternion_world_xyzw": nominal,
                "arrival_tolerance_m": 0.005,
                "arrival_orientation_tolerance_rad": 0.08,
            },
            {
                "phase_id": "contact_close",
                "mode": "ik_pose",
                "target_position_world_m": [1.0, 2.0, 3.0],
                "target_quaternion_world_xyzw": nominal,
                "arrival_tolerance_m": 0.005,
                "arrival_orientation_tolerance_rad": 0.08,
            },
        ],
        "plan_digest": "sha256:" + "a" * 64,
    }

    selected_plan, targets, receipt = _select_parallel_jaw_control_plan(
        servo=_Servo(),
        control_plan=plan,
        construction_bound_targets=[],
        reference_seeds=[[0.5] * 7],
    )

    assert receipt["selected_variant_id"] == "parallel_jaw_equivalent"
    assert receipt["variants"][1]["all_unique_poses_solved_or_bound"] is False
    assert receipt["variants"][1]["contact_phases_solved_or_bound"] is True
    assert selected_plan["runtime_control_variant"].startswith("parallel_jaw")
    assert targets[-1]["joint_positions_rad"] == pytest.approx([0.2] * 7)


def test_controls_uses_live_dls_fallback_without_contact_solution() -> None:
    class _Servo:
        def read_arm_joint_positions(self):
            return [0.0] * 7

        def solve_grasp_target_multistart(self, **_kwargs):
            return {"solved": False, "selected": None, "attempts": []}

    nominal = [-2**-0.5, 0.0, 0.0, 2**-0.5]
    plan = {
        "schema_version": "adp_task_control_plan.v1",
        "scripted_positive_actions": [
            {
                "phase_id": phase_id,
                "mode": "ik_pose",
                "target_position_world_m": [1.0, 2.0, 3.0],
                "target_quaternion_world_xyzw": nominal,
                "arrival_tolerance_m": 0.005,
                "arrival_orientation_tolerance_rad": 0.08,
            }
            for phase_id in ("contact_open", "contact_close")
        ],
        "plan_digest": "sha256:" + "a" * 64,
    }

    selected_plan, targets, receipt = _select_parallel_jaw_control_plan(
        servo=_Servo(),
        control_plan=plan,
        construction_bound_targets=[],
        reference_seeds=[[0.5] * 7],
    )

    assert selected_plan == plan
    assert targets == []
    assert receipt["selected_variant_id"] == "normalized_nominal"
    assert receipt["status"] == (
        "selected_live_physx_dls_fallback_before_physics_motion"
    )
    assert receipt["selected_contact_open_minimum_joint_limit_margin_rad"] is None
    assert all(row["admissible"] is False for row in receipt["variants"])


def test_controls_selects_safe_open_branch_when_close_requires_live_dls() -> None:
    class _Servo:
        def read_arm_joint_positions(self):
            return [0.0] * 7

        def solve_grasp_target_multistart(self, **kwargs):
            position = kwargs["target_position_world_m"]
            if position == [1.0, 2.1, 3.0]:
                return {"solved": False, "selected": None, "attempts": []}
            quaternion = kwargs["target_grasp_frame_quaternion_world_xyzw"]
            equivalent = quaternion[1] > 0.5
            margin = 0.2 if equivalent else 0.02
            return {
                "solved": True,
                "selected": {
                    "solved": True,
                    "joint_positions_rad": [margin] * 7,
                    "minimum_joint_limit_margin_rad": margin,
                },
                "attempts": [],
            }

    nominal = [-2**-0.5, 0.0, 0.0, 2**-0.5]
    plan = {
        "schema_version": "adp_task_control_plan.v1",
        "scripted_positive_actions": [
            {
                "phase_id": "contact_open",
                "mode": "ik_pose",
                "target_position_world_m": [1.0, 2.0, 3.0],
                "target_quaternion_world_xyzw": nominal,
                "arrival_tolerance_m": 0.005,
                "arrival_orientation_tolerance_rad": 0.08,
            },
            {
                "phase_id": "contact_close",
                "mode": "ik_pose",
                "target_position_world_m": [1.0, 2.1, 3.0],
                "target_quaternion_world_xyzw": nominal,
                "arrival_tolerance_m": 0.005,
                "arrival_orientation_tolerance_rad": 0.08,
            },
        ],
        "plan_digest": "sha256:" + "a" * 64,
    }

    selected_plan, targets, receipt = _select_parallel_jaw_control_plan(
        servo=_Servo(),
        control_plan=plan,
        construction_bound_targets=[],
        reference_seeds=[[0.5] * 7],
    )

    assert receipt["selected_variant_id"] == "parallel_jaw_equivalent"
    assert receipt["variants"][1]["contact_open_solved_or_bound"] is True
    assert receipt["variants"][1]["contact_close_solved_or_bound"] is False
    assert receipt["variants"][1]["admissible"] is True
    assert selected_plan["runtime_control_variant"].startswith("parallel_jaw")
    assert [row["phase_id"] for row in targets] == ["contact_open"]


def test_controls_bind_construction_global_ik_branch_to_same_pose_phase() -> None:
    rows = _construction_global_ik_joint_targets(
        construction={
            "pink_global_ik_preflight": {
                "schema_version": "native_task_pink_global_ik_preflight.v1",
                "phases": [
                    {"phase_id": "prealign", "selected": None},
                    {
                        "phase_id": "approach",
                        "selected": {
                            "solved": True,
                            "joint_positions_rad": [0.1] * 7,
                        },
                    },
                ],
            },
            "phase_results": [
                {
                    "phase_id": "prealign",
                    "target_position_world_m": [1.0, 2.0, 2.5],
                    "target_orientation_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                {
                    "phase_id": "approach",
                    "target_position_world_m": [1.0, 2.0, 3.0],
                    "target_orientation_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            ],
        },
        control_plan={
            "scripted_positive_actions": [
                {
                    "phase_id": "prealign",
                    "mode": "ik_pose",
                    "target_position_world_m": [1.0, 2.0, 2.5],
                    "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                {
                    "phase_id": "approach",
                    "mode": "ik_pose",
                    "target_position_world_m": [1.0, 2.0, 3.0],
                    "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            ]
        },
    )

    assert rows == [
        {
            "phase_id": "approach",
            "target_position_world_m": [1.0, 2.0, 3.0],
            "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
            "joint_positions_rad": [0.1] * 7,
        }
    ]


def test_controls_reject_global_ik_phase_with_different_control_pose() -> None:
    with pytest.raises(
        RuntimeError,
        match="native_task_controls_global_ik_binding_invalid",
    ):
        _construction_global_ik_joint_targets(
            construction={
                "pink_global_ik_preflight": {
                    "schema_version": "native_task_pink_global_ik_preflight.v1",
                    "phases": [
                        {
                            "phase_id": "approach",
                            "selected": {
                                "solved": True,
                                "joint_positions_rad": [0.1] * 7,
                            },
                        }
                    ],
                },
                "phase_results": [
                    {
                        "phase_id": "approach",
                        "target_position_world_m": [1.0, 2.0, 3.0],
                        "target_orientation_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                    }
                ],
            },
            control_plan={
                "scripted_positive_actions": [
                    {
                        "phase_id": "approach",
                        "mode": "ik_pose",
                        "target_position_world_m": [1.0, 2.0, 3.1],
                        "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                    }
                ]
            },
        )


def test_controls_worker_source_has_no_scene_task_or_policy_identity() -> None:
    source = Path(
        __import__(
            "blueprint_pipeline.native_task_arena_controls_worker",
            fromlist=["x"],
        ).__file__
    ).read_text(encoding="utf-8")

    for forbidden in (
        "840313",
        "840796",
        "refrigerator",
        "approved_can",
        "pi05_droid",
        "groot_n17_droid",
    ):
        assert forbidden not in source


def test_controls_manifest_rejects_policy_or_construction_mode(tmp_path: Path) -> None:
    manifest = {
        "schema_version": "native_task_arena_provider_bundle.v1",
        "execution_mode": "controls",
        "policy_candidate_id": None,
        "candidate_policy_queried": False,
        "input_digest": "",
    }
    manifest["input_digest"] = canonical_digest(
        manifest, digest_field="input_digest"
    )
    path = tmp_path / "adp_arena_provider_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert _load_and_verify_manifest(tmp_path)["execution_mode"] == "controls"

    for mode in ("construction_canary", "policy"):
        manifest["execution_mode"] = mode
        manifest["input_digest"] = canonical_digest(
            manifest, digest_field="input_digest"
        )
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(RuntimeError, match="native_task_controls_manifest_invalid"):
            _load_and_verify_manifest(tmp_path)


def test_controls_runtime_inputs_reverify_every_byte(tmp_path: Path) -> None:
    inputs = tmp_path / "runtime_inputs"
    inputs.mkdir()
    rows = []
    for name in (
        "native_task_arena_construction_result.v1.json",
        "adp_task_control_plan.v1.json",
        "adp_task_control_execution_spec.v1.json",
    ):
        path = inputs / name
        path.write_text("{}\n", encoding="utf-8")
        rows.append(
            {
                "relative_path": f"runtime_inputs/{name}",
                "size_bytes": path.stat().st_size,
                "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    verified = _verified_runtime_inputs(
        tmp_path, {"bound_runtime_inputs": rows}
    )
    assert set(verified) == {
        "native_task_arena_construction_result.v1.json",
        "adp_task_control_plan.v1.json",
        "adp_task_control_execution_spec.v1.json",
    }

    (inputs / "adp_task_control_plan.v1.json").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity_mismatch"):
        _verified_runtime_inputs(tmp_path, {"bound_runtime_inputs": rows})


def test_control_execution_spec_binds_one_exact_episode(tmp_path: Path) -> None:
    scene = {
        "task_kind": "rigid_pick_place",
        "task_spec": {"schema_version": "adp_task_spec.v2"},
        "plan_digest": "sha256:" + "1" * 64,
    }
    construction = {"result_digest": "sha256:" + "2" * 64}
    plan = {"plan_digest": "sha256:" + "3" * 64}
    value = {
        "schema_version": "adp_task_control_execution_spec.v1",
        "control_selection": "zero_action_negative",
        "task_kind": "rigid_pick_place",
        "scene_plan_digest": scene["plan_digest"],
        "construction_result_digest": construction["result_digest"],
        "control_plan_digest": plan["plan_digest"],
        "candidate_policy_queried": False,
        "execution_spec_digest": "",
    }
    value["execution_spec_digest"] = _canonical_digest(
        value, field="execution_spec_digest"
    )
    path = tmp_path / "adp_task_control_execution_spec.v1.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    checked = _control_execution_spec(
        {"adp_task_control_execution_spec.v1.json": path},
        scene_plan=scene,
        construction=construction,
        control_plan=plan,
    )

    assert checked["control_selection"] == "zero_action_negative"
    value["control_selection"] = "policy"
    value["execution_spec_digest"] = _canonical_digest(
        value, field="execution_spec_digest"
    )
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="execution_spec_invalid"):
        _control_execution_spec(
            {"adp_task_control_execution_spec.v1.json": path},
            scene_plan=scene,
            construction=construction,
            control_plan=plan,
        )


def test_downstream_diagnostic_request_is_default_off_and_digest_bound(
    tmp_path: Path,
) -> None:
    assert _downstream_diagnostic_request({}) == {
        "status": "not_requested",
        "enabled": False,
        "provider_mutation_performed": False,
    }
    request = {
        "schema_version": (
            "adp_task_synthetic_post_phase5_downstream_diagnostic_request.v1"
        ),
        "enabled": True,
        "development_only": True,
        "qualification_effect": "none",
        "request_digest": "",
    }
    request["request_digest"] = _canonical_digest(
        request, field="request_digest"
    )
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")

    checked = _downstream_diagnostic_request(
        {
            "adp_task_synthetic_post_phase5_downstream_diagnostic_request.v1.json": path
        }
    )
    assert checked["status"] == "requested"
    assert checked["request_digest"] == request["request_digest"]

    request["qualification_effect"] = "qualify"
    path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(RuntimeError, match="diagnostic_request_invalid"):
        _downstream_diagnostic_request(
            {
                "adp_task_synthetic_post_phase5_downstream_diagnostic_request.v1.json": path
            }
        )


def test_requested_downstream_diagnostic_exits_before_unrelated_controls_work() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(main)))
    requested_branch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "downstream_diagnostic_request" in ast.unparse(node.test)
        and "enabled" in ast.unparse(node.test)
        and "is True" in ast.unparse(node.test)
        and "graph_rigid" not in ast.unparse(node.test)
    )
    assert any(isinstance(node, ast.Return) for node in requested_branch.body)

    forbidden_calls = {
        "run_downstream_phase_posture_matrix",
        "run_actuator_posture_sweep",
        "run_contact_close_posture_sweep",
        "run_contact_acquisition_sweep",
        "run_task_neutral_controls",
    }
    call_lines = {}
    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Name)
            or node.func.id not in forbidden_calls
        ):
            continue
        # The rigid task exits through its own scorer before this articulated
        # diagnostic branch. It is not work that can follow a requested
        # post-Phase-5 articulated diagnostic.
        if (
            node.func.id == "run_task_neutral_controls"
            and node.lineno < requested_branch.lineno
        ):
            continue
        call_lines[node.func.id] = node.lineno
    assert set(call_lines) == forbidden_calls
    assert all(
        line > requested_branch.end_lineno for line in call_lines.values()
    )


def test_graph_rigid_controls_bypass_articulated_contact_diagnostics() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(main)))
    rigid_branch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "graph_rigid"
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "run_task_neutral_controls"
            for child in ast.walk(node)
        )
    )

    assert any(isinstance(node, ast.Return) for node in rigid_branch.body)
    contact_open_lookup = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and node.value == "contact_open"
        and node.lineno > rigid_branch.lineno
    )
    assert rigid_branch.end_lineno < contact_open_lookup.lineno


class _BaseRigidEnvironment:
    def reset(self) -> None:
        return None

    def read_object_sample(self) -> dict:
        return {
            "task_object_pose_world": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "gripper_width_m": 0.071,
            "grasp_frame_position_world_m": [1.0, 2.0, 3.0],
        }


class _ExactRigidReadback:
    def read_task_sample(self) -> dict:
        return {
            "asset_root_pose_world": [1.0, 2.0, 0.7, 0.0, 0.0, 0.0, 1.0],
            "task_scoring_pose_world": [1.02, 1.99, 0.73, 0.0, 0.0, 0.0, 1.0],
            "task_robot_contact_peak_force_n": 0.75,
            "task_support_contact_peak_force_n": 4.0,
            "task_scene_collision_peak_force_n": 0.2,
            "robot_scene_contact_peak_force_n": 0.1,
            "robot_task_forbidden_collision_peak_force_n": 0.0,
            "locked_joint_containment_violation": False,
        }


def _graph_rigid_task_spec() -> dict:
    return {
        "task_contact_minimum_force_n": 0.5,
        "collision_failure_minimum_force_n": 1.0,
        "workspace_position_bounds_world_m": {
            "minimum": [0.0, 0.0, 0.0],
            "maximum": [2.0, 3.0, 2.0],
        },
    }


def test_rigid_controls_environment_uses_scoring_frame_and_exact_contacts() -> None:
    environment = _RigidScoringEnvironment(
        environment=_BaseRigidEnvironment(),
        task_readback=_ExactRigidReadback(),
        task_spec=_graph_rigid_task_spec(),
    )

    sample = environment.read_object_sample()

    assert sample["task_object_pose_world"] == [
        1.02,
        1.99,
        0.73,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    assert sample["asset_root_pose_world"] != sample["task_object_pose_world"]
    assert sample["gripper_width_m"] == pytest.approx(0.071)
    assert sample["task_contact_active"] is True
    assert sample["support_contact_active"] is True
    assert sample["robot_collision_failure"] is False
    assert sample["scene_collision_failure"] is False
    assert sample["forbidden_robot_task_collision_failure"] is False
    assert sample["locked_joint_containment_violation"] is False
    assert sample["containment_violation"] is False
    environment.reset()


def test_rigid_controls_environment_fails_closed_on_missing_native_channel() -> None:
    readback = _ExactRigidReadback()
    readback.read_task_sample = lambda: {"task_scoring_pose_world": [0.0] * 7}
    environment = _RigidScoringEnvironment(
        environment=_BaseRigidEnvironment(),
        task_readback=readback,
        task_spec=_graph_rigid_task_spec(),
    )

    with pytest.raises(RuntimeError, match="rigid_sample_invalid"):
        environment.read_object_sample()


def _bundled_controls_inputs(tmp_path: Path, task_kind: str) -> dict[str, dict]:
    """Read back exactly what the worker reads on the provider, from real bytes.

    Nothing here is hand-written: the packet, the construction receipt, the
    control plan and the manifest all come from their real producers, are frozen
    into a real bundle, and are then read out of that bundle the way the worker
    reads them on the GPU.
    """

    import zipfile

    from tests.test_native_task_arena_bundle import (
        _packet,
        _runtime_source_packet,
        _sha,
        _articulated_packet,
        _qualified_construction,
    )
    from blueprint_pipeline.native_task_arena_controls_bundle import (
        build_native_task_arena_controls_bundle,
    )

    if task_kind == "articulated_open_close":
        packet, scene = _articulated_packet(tmp_path)
        construction_path = _qualified_construction(tmp_path, scene)
    else:
        from tests.test_native_task_control_plan import (
            _rigid_construction,
            _rigid_scene,
        )

        scene = _rigid_scene(scene_id="840313", asset_id="fixture_asset")
        packet = _packet(tmp_path, scene_id="840313")
        plan_path = packet / "native_task_arena_scene_plan.v1.json"
        plan_path.write_text(
            json.dumps(scene, sort_keys=True) + "\n", encoding="utf-8"
        )
        receipt_path = packet / "native_task_arena_packet_receipt.v1.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["arena_scene_plan_digest"] = scene["plan_digest"]
        artifact = next(
            row for row in receipt["artifacts"] if row["role"] == "arena_scene_plan"
        )
        artifact["size_bytes"] = plan_path.stat().st_size
        artifact["sha256"] = _sha(plan_path)
        receipt["receipt_digest"] = canonical_digest(
            receipt, digest_field="receipt_digest"
        )
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        construction_path = tmp_path / "native_task_arena_construction_result.v1.json"
        construction_path.write_text(
            json.dumps(_rigid_construction(scene), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    bundle = build_native_task_arena_controls_bundle(
        job_dir=tmp_path / "controls-bundle",
        packet_dir=packet,
        construction_result_path=construction_path,
        runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
        implementation_commit="c" * 40,
        generated_at="fixed",
    )
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(bundle["bundle_path"]) as archive:
        archive.extractall(extracted)
    runtime = extracted / "provider_runtime"
    inner = runtime / "native_task_packet"
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))  # noqa: E731
    return {
        "manifest": read(runtime / "adp_arena_provider_manifest.json"),
        "packet_receipt": read(
            inner / "native_task_arena_packet_receipt.v1.json"
        ),
        "scene_plan": read(inner / "native_task_arena_scene_plan.v1.json"),
        "construction": read(
            runtime
            / "runtime_inputs/native_task_arena_construction_result.v1.json"
        ),
        "control_plan": read(
            runtime / "runtime_inputs/adp_task_control_plan.v1.json"
        ),
    }


@pytest.mark.parametrize(
    "task_kind", ["articulated_open_close", "rigid_pick_place"]
)
def test_real_producers_satisfy_every_controls_input_binding_relation(
    tmp_path: Path, task_kind: str
) -> None:
    """The gate must be satisfiable by the artifacts its own producers emit.

    A single disagreeing relation costs one full paid provider run, so this
    proves the whole chain agrees before any GPU is rented.
    """

    inputs = _bundled_controls_inputs(tmp_path, task_kind)

    assert _input_binding_mismatches(**inputs) == []
    assert inputs["scene_plan"]["task_kind"] == task_kind


def test_parallel_jaw_variant_is_accepted_by_real_control_plan_validator(
    tmp_path: Path,
) -> None:
    from blueprint_pipeline.adp009d_control_episode import (
        validate_task_control_plan,
    )

    inputs = _bundled_controls_inputs(tmp_path, "rigid_pick_place")
    normalized = _normalized_control_plan_for_execution(
        control_plan=inputs["control_plan"],
        task_spec=inputs["scene_plan"]["task_spec"],
    )
    derived = _parallel_jaw_equivalent_control_plan(normalized)

    checked = validate_task_control_plan(
        derived,
        task_spec=inputs["scene_plan"]["task_spec"],
    )

    assert checked["plan_digest"] == derived["plan_digest"]
    assert checked["plan_digest"] == _canonical_digest(
        checked, field="plan_digest"
    )
    assert checked["runtime_normalized_source_plan_digest"] == inputs[
        "control_plan"
    ]["plan_digest"]
    assert checked["runtime_control_variant"].startswith("parallel_jaw")
    assert checked["scripted_positive_actions"][0][
        "target_quaternion_world_xyzw"
    ] == pytest.approx(
        derived["scripted_positive_actions"][0][
            "target_quaternion_world_xyzw"
        ]
    )


def test_each_controls_binding_relation_reports_which_one_failed(
    tmp_path: Path,
) -> None:
    """One opaque blocker cannot be read; each relation must name itself."""

    inputs = _bundled_controls_inputs(tmp_path, "rigid_pick_place")
    other = "sha256:" + "0" * 64
    breakages = {
        "packet_receipt_digest_vs_manifest": (
            "manifest",
            "packet_receipt_digest",
            other,
        ),
        "scene_plan_digest_vs_manifest": (
            "manifest",
            "arena_scene_plan_digest",
            other,
        ),
        "construction_result_digest_vs_control_plan_planner_receipt": (
            "construction",
            "result_digest",
            other,
        ),
        "control_plan_construction_scene_plan_digest_vs_scene_plan": (
            "control_plan",
            "construction_scene_plan_digest",
            other,
        ),
        "control_plan_construction_clearance_plan_digest_vs_construction": (
            "control_plan",
            "construction_clearance_plan_digest",
            other,
        ),
        "control_plan_task_kind_vs_scene_plan_task_kind": (
            "control_plan",
            "task_kind",
            "articulated_open_close",
        ),
    }
    for relation, (artifact, field, value) in breakages.items():
        broken = {key: dict(item) for key, item in inputs.items()}
        broken[artifact][field] = value
        mismatches = _input_binding_mismatches(**broken)
        assert relation in mismatches, relation
        # Editing a control-plan field also breaks its self digest; nothing else
        # may be dragged in.
        expected = {relation}
        if artifact == "control_plan":
            expected.add("control_plan_plan_digest_vs_recomputed_canonical_digest")
        assert set(mismatches) == expected, relation

    tampered = {key: dict(item) for key, item in inputs.items()}
    tampered["control_plan"]["plan_digest"] = other
    assert _input_binding_mismatches(**tampered) == [
        "control_plan_plan_digest_vs_recomputed_canonical_digest"
    ]


def test_controls_binding_refuses_two_absent_digests(tmp_path: Path) -> None:
    """Two missing digests are two refusals, never one agreement.

    Comparing absent fields with `!=` alone admitted an unbound cell: every
    digest relation held vacuously as `None == None`, and only the control
    plan's self digest objected -- which a plan carrying nothing but its own
    digest satisfies.
    """

    empty_plan: dict = {}
    empty_plan["plan_digest"] = _canonical_digest(empty_plan, field="plan_digest")

    mismatches = _input_binding_mismatches(
        manifest={},
        packet_receipt={},
        scene_plan={},
        construction={},
        control_plan=empty_plan,
    )

    assert set(mismatches) == {
        "packet_receipt_digest_vs_manifest",
        "scene_plan_digest_vs_manifest",
        "construction_result_digest_vs_control_plan_planner_receipt",
        "control_plan_construction_scene_plan_digest_vs_scene_plan",
        "control_plan_construction_clearance_plan_digest_vs_construction",
    }


def test_persist_survives_values_json_cannot_encode() -> None:
    """A receipt that cannot be written destroys the diagnosis of a paid run.

    `_persist` is called from a `finally`. The digest is computed *before* the
    write, so passing `default=str` to the write alone left a stray warp array
    or Path raising inside the handler -- replacing the real exception and
    leaving a paid run with no receipt at all. The policy worker fixed this;
    the controls and construction workers still carried the defect.
    """

    from tempfile import TemporaryDirectory

    from blueprint_pipeline.native_task_arena_controls_worker import _persist

    class _Unencodable:
        def __repr__(self) -> str:
            return "<warp array>"

    with TemporaryDirectory() as directory:
        target = Path(directory) / "native_task_arena_control_result.v1.json"
        _persist(target, {"status": "blocked", "stray": _Unencodable()})

        written = json.loads(target.read_text(encoding="utf-8"))

    assert written["status"] == "blocked"
    assert written["stray"] == "<warp array>"
    assert written["result_digest"].startswith("sha256:")


def test_persisted_controls_digest_describes_the_bytes_on_disk() -> None:
    """The digest must be recomputable from the receipt a reviewer reads."""

    from tempfile import TemporaryDirectory

    from blueprint_pipeline.native_task_arena_controls_worker import _persist

    with TemporaryDirectory() as directory:
        target = Path(directory) / "native_task_arena_control_result.v1.json"
        _persist(target, {"status": "completed", "blockers": []})
        written = json.loads(target.read_text(encoding="utf-8"))

    assert written["result_digest"] == _canonical_digest(
        written, field="result_digest"
    )


def _branch_replay_task() -> dict:
    return {
        "schema_version": "adp_task_spec.v1",
        "task_kind": "articulated_open_close",
        "task_id": "washer_door_open_test",
        "target_joint_id": "door_hinge",
        "joint_reset_positions_rad": {"door_hinge": 0.0},
        "target_success_interval_rad": [0.7, 0.95],
        "joint_hard_limits_rad": {"door_hinge": [0.0, 1.2]},
        "settle_window_samples": 3,
        "maximum_settled_target_speed_rad_s": 0.05,
        "non_task_joint_motion_tolerance_rad": 0.001,
        "movement_epsilon_rad": 0.0001,
        "reset_tolerance_rad": 0.0001,
        "maximum_action_steps": 200,
    }


def _branch_replay_plan(task: dict) -> dict:
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    def pose(phase_id, position, gripper_state, *, hold=False):
        return {
            "phase_id": phase_id,
            "mode": "ik_pose",
            "target_position_world_m": position,
            "target_quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
            "gripper_state": gripper_state,
            "minimum_steps": 1,
            "maximum_steps": 12,
            "arrival_tolerance_m": 0.005,
            "arrival_stability_steps": 2,
            "arrival_orientation_tolerance_rad": 0.08,
            "max_joint_delta_rad": 0.03,
            "max_joint_setpoint_lead_rad": 0.2,
            "hold_arm_joint_positions_during_gripper_transition": hold,
        }

    contact_position = [0.5, 0.1, 0.4]
    plan = {
        "schema_version": "adp_task_control_plan.v1",
        "cell_id": "branch-replay-test",
        "task_spec_digest": canonical_digest(task),
        "trajectory_source": "native_ik_preflight",
        "planner_receipt_digest": "sha256:" + "f" * 64,
        "zero_action_steps": 3,
        "scripted_positive_actions": [
            pose("prealign", [0.4, 0.0, 0.5], "open"),
            pose("approach", [0.45, 0.05, 0.45], "open"),
            pose("contact_open", contact_position, "open"),
            pose("contact_close", contact_position, "closed", hold=True),
            pose("retreat", [0.4, 0.0, 0.5], "open"),
        ],
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def _branch_replay_targets(plan: dict) -> list[dict]:
    rows = {row["phase_id"]: row for row in plan["scripted_positive_actions"]}
    return [
        {
            "phase_id": "approach",
            "target_position_world_m": rows["approach"]["target_position_world_m"],
            "target_quaternion_world_xyzw": rows["approach"][
                "target_quaternion_world_xyzw"
            ],
            "joint_positions_rad": [0.0] * 7,
        },
        {
            "phase_id": "contact_open",
            "target_position_world_m": rows["contact_open"][
                "target_position_world_m"
            ],
            "target_quaternion_world_xyzw": rows["contact_open"][
                "target_quaternion_world_xyzw"
            ],
            "joint_positions_rad": [0.3, -0.2, 0.1, 0.4, -0.5, 0.2, 0.05],
        },
    ]


def test_contact_entry_branch_replay_inserts_bounded_joint_path() -> None:
    """C28's sealed divergence: Cartesian servoing cannot land contact entry.

    Three measured-miss-biased attempts moved 15.4 -> 28.5 -> 38.6 mm away
    with the wrist swinging 0.264 rad, while the preflight held an exact
    interior-margin solution for the same pose.  The entry must replay the
    solved same-branch joint path; the contact_open arrival gate is unchanged.
    """

    from blueprint_pipeline.adp009d_control_episode import (
        validate_task_control_plan,
    )
    from blueprint_pipeline.native_task_arena_controls_worker import (
        CONTACT_ENTRY_BRANCH_REPLAY_PHASE_ID,
        CONTACT_ENTRY_BRANCH_REPLAY_SETTLE_ROWS,
        _with_contact_entry_branch_replay,
    )

    task = _branch_replay_task()
    plan = _branch_replay_plan(task)
    targets = _branch_replay_targets(plan)

    rewritten, receipt = _with_contact_entry_branch_replay(
        control_plan=plan,
        scripted_pose_joint_targets=targets,
        task_spec=task,
    )

    assert receipt["status"] == "applied"
    actions = rewritten["scripted_positive_actions"]
    replay_rows = [
        row
        for row in actions
        if row.get("phase_id") == CONTACT_ENTRY_BRANCH_REPLAY_PHASE_ID
    ]
    assert receipt["interpolation_rows"] == 10
    assert len(replay_rows) == 10 + CONTACT_ENTRY_BRANCH_REPLAY_SETTLE_ROWS
    assert all(row["gripper_state"] == "open" for row in replay_rows)
    # Every row commands the same solved posture, and carries the bounds that
    # make the servo ramp toward it at a rate the joints can follow.  The
    # executor therefore cannot outrun a lagging joint the way an open-loop
    # interpolation did.
    for row in replay_rows:
        assert row["arm_joint_positions"] == pytest.approx(
            targets[1]["joint_positions_rad"]
        )
        assert row["max_joint_delta_rad"] > 0.0
        assert row["max_joint_setpoint_lead_rad"] >= row["max_joint_delta_rad"]
    # The rows sit immediately before the unchanged contact_open gate, and
    # contact_close still directly follows contact_open.
    phase_ids = [row["phase_id"] for row in actions]
    contact_index = phase_ids.index("contact_open")
    assert phase_ids[contact_index - 1] == CONTACT_ENTRY_BRANCH_REPLAY_PHASE_ID
    assert phase_ids[contact_index + 1] == "contact_close"
    # The rewritten plan re-digests, still satisfies the fail-closed
    # validator, and the validator preserves the bounds rather than dropping
    # the rows back onto the unbounded command path.
    validated = validate_task_control_plan(rewritten, task_spec=task)
    assert validated["plan_digest"] == rewritten["plan_digest"]
    validated_replay = [
        row
        for row in validated["scripted_positive_actions"]
        if row.get("phase_id") == CONTACT_ENTRY_BRANCH_REPLAY_PHASE_ID
    ]
    assert validated_replay
    assert all("max_joint_delta_rad" in row for row in validated_replay)
    assert all("max_joint_setpoint_lead_rad" in row for row in validated_replay)


def test_contact_entry_branch_replay_fails_open_without_solutions() -> None:
    from blueprint_pipeline.native_task_arena_controls_worker import (
        _with_contact_entry_branch_replay,
    )

    task = _branch_replay_task()
    plan = _branch_replay_plan(task)
    targets = [row for row in _branch_replay_targets(plan) if row["phase_id"] != "contact_open"]

    rewritten, receipt = _with_contact_entry_branch_replay(
        control_plan=plan,
        scripted_pose_joint_targets=targets,
        task_spec=task,
    )

    assert receipt["status"] == "not_applied"
    assert receipt["reason"] == "approach_or_contact_solution_unsolved"
    assert rewritten["plan_digest"] == plan["plan_digest"]
    assert rewritten["scripted_positive_actions"] == plan["scripted_positive_actions"]


def test_contact_entry_branch_replay_respects_task_budget() -> None:
    from blueprint_pipeline.native_task_arena_controls_worker import (
        _with_contact_entry_branch_replay,
    )

    task = _branch_replay_task()
    # Below even a minimal replay once the confirm-only contact budget is
    # reclaimed, so there is genuinely nowhere to put the rows.
    task["maximum_action_steps"] = 56
    plan = _branch_replay_plan(task)
    targets = _branch_replay_targets(plan)

    rewritten, receipt = _with_contact_entry_branch_replay(
        control_plan=plan,
        scripted_pose_joint_targets=targets,
        task_spec=task,
    )

    assert receipt["status"] == "not_applied"
    assert receipt["reason"] == "task_step_budget_insufficient"
    assert rewritten["scripted_positive_actions"] == plan["scripted_positive_actions"]


def test_branch_replay_row_size_follows_the_slowest_actuator() -> None:
    """Replay rows bypass the servo, so they must be sized to the hardware.

    C30's replay drove straight into saturation by its eleventh row: these
    are commanded joint targets, so nothing bounds them, and 0.05 rad per
    15 Hz step is 0.75 rad/s against a wrist that clips above 0.15.
    """

    from blueprint_pipeline.native_task_arena_controls_worker import (
        CONTACT_ENTRY_BRANCH_REPLAY_PHASE_ID,
        _with_contact_entry_branch_replay,
    )

    task = _branch_replay_task()
    plan = _branch_replay_plan(task)
    targets = _branch_replay_targets(plan)
    # Shoulders may step 0.036 rad; the wrist only 0.005.
    feasible = [0.036] * 4 + [0.005] * 3

    rewritten, receipt = _with_contact_entry_branch_replay(
        control_plan=plan,
        scripted_pose_joint_targets=targets,
        task_spec=task,
        actuator_feasible_step_rad=feasible,
    )

    assert receipt["status"] == "applied"
    assert receipt["actuator_feasible_step_rad"] == pytest.approx(0.005)
    rows = [
        row
        for row in rewritten["scripted_positive_actions"]
        if row.get("phase_id") == CONTACT_ENTRY_BRANCH_REPLAY_PHASE_ID
    ]
    # The slew handed to the servo is the slowest actuator's feasible step, so
    # the ramp it produces advances no faster than that joint can follow.
    assert all(
        row["max_joint_delta_rad"] == pytest.approx(0.005) for row in rows
    )
    assert all(
        row["arm_joint_positions"] == pytest.approx(targets[1]["joint_positions_rad"])
        for row in rows
    )
    # Enough rows to cover the traverse at that rate.
    assert len(rows) >= 0.5 / 0.005
    # The Cartesian phase behind a replayed entry only confirms arrival, and
    # that reclaimed budget is what pays for the extra rows.
    assert receipt["contact_phase_steps_reclaimed"] > 0
    assert receipt["budget_limited"] is False


def test_a_budget_limited_replay_seals_that_it_was_rushed() -> None:
    """A rushed replay will clip, so the receipt must say so."""

    from blueprint_pipeline.native_task_arena_controls_worker import (
        _with_contact_entry_branch_replay,
    )

    task = _branch_replay_task()
    task["maximum_action_steps"] = 90
    plan = _branch_replay_plan(task)
    targets = _branch_replay_targets(plan)

    _rewritten, receipt = _with_contact_entry_branch_replay(
        control_plan=plan,
        scripted_pose_joint_targets=targets,
        task_spec=task,
        actuator_feasible_step_rad=[0.001] * 7,
    )

    assert receipt["status"] == "applied"
    assert receipt["budget_limited"] is True
    assert receipt["per_row_joint_step_rad"] > receipt["actuator_feasible_step_rad"]


def test_adopting_a_calibrated_posture_requires_recompiling_the_plan() -> None:
    """C37's sealed trap: the episode executes the plan, not the posture list.

    The run adopted the calibration's best posture into the posture list after
    the plan -- replay rows included -- had already been compiled, so contact
    entry replayed the model's branch and missed by 70-114 mm while the
    calibrated posture itself measured 13 mm.  Adoption is only real when the
    replay generator is re-run over the updated postures: the recompiled rows
    must end at the adopted posture and the plan digest must change with them.
    """

    from blueprint_pipeline.native_task_arena_controls_worker import (
        CONTACT_ENTRY_BRANCH_REPLAY_PHASE_ID,
        _with_contact_entry_branch_replay,
    )

    task = _branch_replay_task()
    plan = _branch_replay_plan(task)
    targets = _branch_replay_targets(plan)

    compiled, first_receipt = _with_contact_entry_branch_replay(
        control_plan=plan, scripted_pose_joint_targets=targets, task_spec=task
    )
    assert first_receipt["status"] == "applied"

    adopted = [0.35, -0.25, 0.12, 0.44, -0.55, 0.22, 0.01]
    updated_targets = [
        (
            {**row, "joint_positions_rad": list(adopted)}
            if row["phase_id"] == "contact_open"
            else row
        )
        for row in targets
    ]
    recompiled, receipt = _with_contact_entry_branch_replay(
        control_plan=plan,
        scripted_pose_joint_targets=updated_targets,
        task_spec=task,
    )

    assert receipt["status"] == "applied"
    replay_rows = [
        row
        for row in recompiled["scripted_positive_actions"]
        if row.get("phase_id") == CONTACT_ENTRY_BRANCH_REPLAY_PHASE_ID
    ]
    # The recompiled entry path lands on the adopted posture, not the model's.
    assert replay_rows[-1]["arm_joint_positions"] == pytest.approx(adopted)
    stale_rows = [
        row
        for row in compiled["scripted_positive_actions"]
        if row.get("phase_id") == CONTACT_ENTRY_BRANCH_REPLAY_PHASE_ID
    ]
    assert stale_rows[-1]["arm_joint_positions"] != pytest.approx(adopted)
    # And what ran is distinguishable from what would have run.
    assert recompiled["plan_digest"] != compiled["plan_digest"]


def _roll_plan(quaternion):
    """A plan whose grasp-holding phases all share one authored orientation."""

    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    rows = []
    for phase_id in (
        "prealign", "approach", "contact_open", "contact_close",
        "joint_path_01", "joint_path_02", "joint_path_03", "joint_path_04",
        "release", "retreat",
    ):
        rows.append(
            {
                "phase_id": phase_id,
                "mode": "ik_pose",
                "target_position_world_m": [1.0, 0.0, 0.5],
                "target_quaternion_world_xyzw": list(quaternion),
                "arrival_tolerance_m": 0.005 if "contact" in phase_id else 0.02,
                "arrival_orientation_tolerance_rad": 0.08,
                "maximum_steps": 40,
                "minimum_steps": 2,
                "arrival_stability_steps": 2,
            }
        )
    plan = {"scripted_positive_actions": rows, "plan_digest": ""}
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


class _RollServo:
    """Strict signature: a stray keyword is a failure, as in production."""

    def __init__(self, margin_for):
        self._margin_for = margin_for
        self._pink_hand_pose_at_binding_base = [0.0] * 3 + [0.0, 0.0, 0.0, 1.0]
        self._pink_grasp_pose_at_binding_base = [0.13, 0.0, 0.0] + [0.0, 0.0, 0.0, 1.0]

    def grasp_approach_axis_body(self):
        return [1.0, 0.0, 0.0]

    def read_arm_joint_positions(self):
        return [0.0] * 7

    def solve_grasp_target_multistart(
        self,
        *,
        target_position_world_m,
        target_grasp_frame_quaternion_world_xyzw,
        preferred_seeds,
        reference_joint_positions_rad,
        position_tolerance_m,
        orientation_tolerance_rad,
        preferred_minimum_joint_limit_margin_rad,
        required_minimum_joint_limit_margin_rad,
    ):
        return {
            "solved": True,
            "selected": {
                "joint_positions_rad": [0.0] * 7,
                "minimum_joint_limit_margin_rad": self._margin_for(
                    target_grasp_frame_quaternion_world_xyzw
                ),
                "position_error_m": 0.001,
                "orientation_error_rad": 0.01,
            },
            "attempts": [],
        }


def test_the_chosen_roll_reaches_the_quaternion_the_controller_drives() -> None:
    """The defect this replaces: a rolled pose that was never commanded.

    Rolling inside the solver sealed a rolled quaternion in a receipt while the
    control-plan row kept the authored one.  The live differential-IK
    controller drives the plan's orientation, so the roll survived only as a
    null-space posture preference -- which by construction cannot move the
    primary six-dimensional pose objective.  This asserts on the plan the
    controller actually reads, by calling the real worker seam.
    """

    from blueprint_pipeline.native_task_arena_controls_worker import (
        _with_selected_grasp_roll,
    )

    authored = [0.0, 0.0, 0.0, 1.0]
    # The authored orientation cannot be held; rolling away from it can.
    def margin_for(q):
        deviation = sum(abs(a - b) for a, b in zip(q, authored))
        return 0.0 if deviation < 0.05 else 0.2

    derived, receipt = _with_selected_grasp_roll(
        servo=_RollServo(margin_for), control_plan=_roll_plan(authored)
    )

    assert receipt["status"] == "applied"
    assert receipt["selected_roll_rad"] != 0.0
    rows = {r["phase_id"]: r for r in derived["scripted_positive_actions"]}
    # Every grasp-holding phase now carries the rolled orientation...
    for phase_id in (
        "contact_open", "contact_close", "joint_path_01", "joint_path_02",
        "joint_path_03", "joint_path_04", "release",
    ):
        assert rows[phase_id]["target_quaternion_world_xyzw"] != authored
        assert rows[phase_id]["authored_target_quaternion_world_xyzw"] == authored
        assert rows[phase_id]["applied_grasp_roll_rad"] == receipt["selected_roll_rad"]
    # ...and every phase that is not holding the grasp is untouched.
    for phase_id in ("prealign", "approach", "retreat"):
        assert rows[phase_id]["target_quaternion_world_xyzw"] == authored
        assert "applied_grasp_roll_rad" not in rows[phase_id]
    # The derived plan is a different, digest-bound plan.
    assert derived["plan_digest"] != receipt["source_control_plan_digest"]
    assert derived["plan_digest"] == receipt["derived_control_plan_digest"]


def test_a_roll_is_refused_when_any_holding_phase_falls_below_the_floor() -> None:
    """Contact entry alone is not enough to admit a roll for the family."""

    from blueprint_pipeline.native_task_arena_controls_worker import (
        _with_selected_grasp_roll,
    )

    authored = [0.0, 0.0, 0.0, 1.0]
    plan = _roll_plan(authored)
    # Make one door-arc phase distinguishable, and starve every rolled pose
    # there: contact would be happy, the family must not be.
    for row in plan["scripted_positive_actions"]:
        if row["phase_id"] == "joint_path_03":
            row["target_position_world_m"] = [2.0, 0.0, 0.5]

    def margin_for(q):
        return 0.0 if sum(abs(a - b) for a, b in zip(q, authored)) < 0.05 else 0.2

    class _StarveOnePhase(_RollServo):
        def solve_grasp_target_multistart(self, **kwargs):
            result = super().solve_grasp_target_multistart(**kwargs)
            if kwargs["target_position_world_m"][0] == 2.0:
                result["selected"]["minimum_joint_limit_margin_rad"] = 0.001
            return result

    derived, receipt = _with_selected_grasp_roll(
        servo=_StarveOnePhase(margin_for), control_plan=plan
    )

    assert receipt["status"] == "not_applied"
    assert "no_roll_clears_required_margin_across_the_family" in receipt["reason"]
    # The plan is returned unchanged rather than half-rolled.
    rows = {r["phase_id"]: r for r in derived["scripted_positive_actions"]}
    assert all(
        row["target_quaternion_world_xyzw"] == authored for row in rows.values()
    )


def test_an_authored_orientation_that_holds_is_left_alone() -> None:
    from blueprint_pipeline.native_task_arena_controls_worker import (
        _with_selected_grasp_roll,
    )

    authored = [0.0, 0.0, 0.0, 1.0]
    derived, receipt = _with_selected_grasp_roll(
        servo=_RollServo(lambda q: 0.4), control_plan=_roll_plan(authored)
    )

    assert receipt["status"] == "not_applied"
    assert receipt["reason"] == "authored_roll_admissible"
    assert receipt["selected_roll_rad"] == 0.0
    rows = derived["scripted_positive_actions"]
    assert all(r["target_quaternion_world_xyzw"] == authored for r in rows)


def test_precision_phases_command_the_postures_their_preflight_solved() -> None:
    """C42/C43 and C85b measured the controller discarding solved vectors.

    The Cartesian controller re-derives a posture from scratch, and it walked
    0.19 to 0.53 rad away from the solved vector -- whose own forward
    kinematics sat inside the arrival gate -- onto one whose kinematics were
    already 20 mm outside it.  Tracking was never the problem: the arm reached
    what it was told within 0.008 rad.  It was told the wrong thing.
    """

    from blueprint_pipeline.adp009d_control_episode import (
        validate_task_control_plan,
    )

    task = _branch_replay_task()
    plan = _branch_replay_plan(task)
    from blueprint_pipeline.native_task_arena_controls_worker import (
        _with_held_solved_contact_vectors,
    )

    approach_solved = [-0.26, -0.78, 0.50, -3.07, -2.89, 2.44, 1.74]
    contact_solved = [0.11, 0.22, 0.33, -0.44, 0.55, 0.66, -0.77]
    rows = {r["phase_id"]: r for r in plan["scripted_positive_actions"]}
    derived, receipt = _with_held_solved_contact_vectors(
        control_plan=plan,
        scripted_pose_joint_targets=[
            {
                "phase_id": "approach",
                "target_position_world_m": rows["approach"][
                    "target_position_world_m"
                ],
                "target_quaternion_world_xyzw": rows["approach"][
                    "target_quaternion_world_xyzw"
                ],
                "joint_positions_rad": list(approach_solved),
            },
            {
                "phase_id": "contact_open",
                "target_position_world_m": rows["contact_open"][
                    "target_position_world_m"
                ],
                "target_quaternion_world_xyzw": rows["contact_open"][
                    "target_quaternion_world_xyzw"
                ],
                "joint_positions_rad": list(contact_solved),
            }
        ],
    )
    assert receipt["status"] == "applied"
    assert receipt["held_phase_ids"] == ["approach", "contact_open"]
    assert receipt["cartesian_fallback_phase_ids"] == ["contact_close"]
    # The plan is re-digested, so the validator accepts it.
    assert derived["plan_digest"] != receipt["source_control_plan_digest"]

    validated = validate_task_control_plan(derived, task_spec=task)

    held = {
        row["phase_id"]: row
        for row in validated["scripted_positive_actions"]
        if row.get("hold_solved_arm_joint_positions_rad")
    }
    assert held, "the solved vectors must survive validation"
    assert held["approach"][
        "hold_solved_arm_joint_positions_rad"
    ] == pytest.approx(approach_solved)
    assert held["contact_open"][
        "hold_solved_arm_joint_positions_rad"
    ] == pytest.approx(contact_solved)
    # The pose and its gate are untouched: a solved vector that does not put
    # the real fingertip on the target still fails honestly.
    assert held["approach"]["mode"] == "ik_pose"
    assert held["approach"]["arrival_tolerance_m"] > 0.0
    # Phases that did not carry one are unaffected.
    assert any(
        row.get("hold_solved_arm_joint_positions_rad") is None
        for row in validated["scripted_positive_actions"]
        if row.get("mode") == "ik_pose"
    )


@pytest.mark.parametrize(
    "malformed",
    (
        [0.1] * 6,
        [0.1] * 6 + [float("nan")],
        "not-a-joint-vector",
    ),
)
def test_malformed_solved_approach_vector_refuses_instead_of_falling_back(
    malformed,
) -> None:
    from blueprint_pipeline.native_task_arena_controls_worker import (
        _with_held_solved_contact_vectors,
    )

    task = _branch_replay_task()
    plan = _branch_replay_plan(task)

    with pytest.raises(
        RuntimeError,
        match="native_task_controls_solved_joint_vector_invalid",
    ):
        _with_held_solved_contact_vectors(
            control_plan=plan,
            scripted_pose_joint_targets=[
                {
                    "phase_id": "approach",
                    "joint_positions_rad": malformed,
                }
            ],
        )


def test_missing_solved_approach_vector_preserves_explicit_cartesian_fallback(
) -> None:
    from blueprint_pipeline.native_task_arena_controls_worker import (
        _with_held_solved_contact_vectors,
    )

    task = _branch_replay_task()
    plan = _branch_replay_plan(task)
    derived, receipt = _with_held_solved_contact_vectors(
        control_plan=plan,
        scripted_pose_joint_targets=[
            {"phase_id": "approach", "joint_positions_rad": None}
        ],
    )

    assert derived == plan
    assert receipt["status"] == "not_applied"
    assert receipt["reason"] == "no_solved_joint_vector_available"
    assert receipt["held_phase_ids"] == []
    assert receipt["cartesian_fallback_phase_ids"] == [
        "approach",
        "contact_close",
        "contact_open",
    ]


def test_contact_close_compensates_measured_closed_pad_midpoint_travel() -> None:
    """C54 lost its global close solution and scored a moving TCP.

    The measured Robotiq pad midpoint advances along the grasp approach axis
    while closing.  Make contact_close a distinct, compensated global-IK pose
    so the *closed* measured midpoint lands on the authored grasp target.
    """

    from blueprint_pipeline.adp009d_control_episode import (
        validate_task_control_plan,
    )
    from blueprint_pipeline.native_task_arena_controls_worker import (
        _with_closed_pad_midpoint_compensated_contact,
    )

    task = _branch_replay_task()
    plan = _branch_replay_plan(task)
    source = next(
        row
        for row in plan["scripted_positive_actions"]
        if row["phase_id"] == "contact_close"
    )
    source["target_quaternion_world_xyzw"] = [
        0.0,
        0.7071067811865475,
        0.7071067811865476,
        0.0,
    ]
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    original = list(source["target_position_world_m"])
    derived, receipt = _with_closed_pad_midpoint_compensated_contact(
        control_plan=plan,
        gripper_convention={
            "open_command": 0.0,
            "closed_command": 1.0,
            "pad_midpoint_controlled_body_m": {
                "0.0": [
                    0.13009979642662167,
                    3.1868757355280053e-09,
                    -4.266703018673823e-09,
                ],
                "1.0": [
                    0.14365582057985699,
                    2.427310029085028e-07,
                    -2.635381418092386e-07,
                ],
            },
        },
        # C55's sealed reset frames. Controlled-body +X is not grasp-frame
        # +X, and at the authored grasp it maps almost entirely to world +Y.
        current_controlled_body_pose_world=[
            3.813486337661743,
            9.166367530822754,
            0.5625487565994263,
            -0.5000001490115236,
            0.5000003874300885,
            0.4999997019767144,
            0.4999997615813556,
        ],
        current_grasp_frame_pose_world=[
            3.8134863112111903,
            9.166367386974125,
            0.4324489601728867,
            0.7071061879139766,
            -0.7071072373633318,
            -0.0003500406233814535,
            0.0002671211765628578,
        ],
    )

    assert receipt["status"] == "applied"
    assert receipt["pad_midpoint_travel_m"] == pytest.approx(0.01355602415783117)
    checked = validate_task_control_plan(derived, task_spec=task)
    close = next(
        row
        for row in checked["scripted_positive_actions"]
        if row["phase_id"] == "contact_close"
    )
    # The compensation follows the measured controlled-body delta through the
    # rigid body-to-grasp transform. C55's discarded implementation compared
    # these differently expressed vectors directly and incorrectly no-opped.
    assert close["target_position_world_m"] == pytest.approx(
        [
            original[0] + 1.8634649404e-06,
            original[1] - 0.013556019075444,
            original[2] + 1.15897110396e-05,
        ]
    )
    assert close["arrival_target_position_world_m"] == pytest.approx(original)
    open_row = next(
        row
        for row in checked["scripted_positive_actions"]
        if row["phase_id"] == "contact_open"
    )
    assert open_row["target_position_world_m"] == pytest.approx(original)
    assert derived["plan_digest"] == canonical_digest(
        derived, digest_field="plan_digest"
    )


def test_a_malformed_solved_vector_is_refused_not_ignored() -> None:
    """Silently dropping it would restore the defect it exists to fix."""

    from blueprint_pipeline.adp009d_control_episode import (
        ControlEpisodeError,
        validate_task_control_plan,
    )

    task = _branch_replay_task()
    plan = _branch_replay_plan(task)
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    for row in plan["scripted_positive_actions"]:
        if row.get("phase_id") == "contact_open" and row.get("mode") == "ik_pose":
            row["hold_solved_arm_joint_positions_rad"] = [0.1, 0.2]  # not seven
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")

    with pytest.raises(ControlEpisodeError):
        validate_task_control_plan(plan, task_spec=task)


def test_measured_contact_frontier_starts_from_observed_success() -> None:
    from blueprint_pipeline.adp009d_control_episode import (
        validate_task_control_plan,
    )
    from blueprint_pipeline.native_task_arena_controls_worker import (
        MEASURED_CONTACT_ENTRY_PHASE_ID,
        MEASURED_CONTACT_FRONTIER_PHASE_PREFIX,
        _with_measured_contact_frontier,
    )

    task = _branch_replay_task()
    plan = _branch_replay_plan(task)
    for row in plan["scripted_positive_actions"]:
        if row["phase_id"] == "contact_open":
            row["hold_solved_arm_joint_positions_rad"] = [0.9] * 7
        elif row["phase_id"] == "contact_close":
            row["hold_solved_arm_joint_positions_rad"] = [0.8] * 7
    contact_close_source = next(
        row
        for row in plan["scripted_positive_actions"]
        if row["phase_id"] == "contact_close"
    )
    path_row = copy.deepcopy(contact_close_source)
    path_row.update(
        {
            "phase_id": "joint_path_01",
            "target_position_world_m": [0.6, 0.2, 0.4],
            "hold_solved_arm_joint_positions_rad": None,
            "hold_arm_joint_positions_during_gripper_transition": False,
        }
    )
    plan["scripted_positive_actions"].insert(-1, path_row)
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    known_joints = [0.11, -0.22, 0.33, -0.44, 0.55, -0.66, 0.77]
    probe = {
        "status": "measured",
        "cells": [
            {
                "status": "measured",
                "offset_m": [0.0, 0.0, -0.04],
                "joint_positions_rad": known_joints,
                "joint_tracking_error_rad": 0.0002,
                "measured_distance_to_requested_m": 0.0042,
                "measured_grasp_frame_orientation_world_xyzw": [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                "contact_steps": 0,
            },
            {
                "status": "measured",
                "offset_m": [0.0, 0.0, -0.02],
                "joint_positions_rad": [0.2] * 7,
                "joint_tracking_error_rad": 0.0003,
                "measured_distance_to_requested_m": 0.008,
                "measured_grasp_frame_orientation_world_xyzw": [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                "contact_steps": 0,
            },
        ],
    }

    derived, receipt = _with_measured_contact_frontier(
        control_plan=plan,
        reachability_probe=probe,
        task_spec=task,
    )
    checked = validate_task_control_plan(derived, task_spec=task)
    rows = checked["scripted_positive_actions"]
    entry = next(row for row in rows if row["phase_id"] == MEASURED_CONTACT_ENTRY_PHASE_ID)
    contact = next(row for row in rows if row["phase_id"] == "contact_open")
    contact_close = next(row for row in rows if row["phase_id"] == "contact_close")
    joint_path = next(row for row in rows if row["phase_id"] == "joint_path_01")
    frontier = [
        row
        for row in rows
        if row["phase_id"].startswith(MEASURED_CONTACT_FRONTIER_PHASE_PREFIX)
    ]

    assert receipt["status"] == "applied"
    assert receipt["probe_measured_error_m"] == pytest.approx(0.0042)
    assert entry["hold_solved_arm_joint_positions_rad"] == pytest.approx(
        known_joints
    )
    assert entry["target_position_world_m"] == pytest.approx([0.5, 0.1, 0.36])
    assert frontier == []
    assert contact["hold_solved_arm_joint_positions_rad"] == pytest.approx(
        known_joints
    )
    assert contact["target_position_world_m"] == pytest.approx([0.5, 0.1, 0.36])
    assert contact_close["target_position_world_m"] == pytest.approx(
        [0.5, 0.1, 0.4]
    )
    assert contact_close["hold_solved_arm_joint_positions_rad"] == pytest.approx(
        [0.8] * 7
    )
    assert contact_close[
        "hold_arm_joint_positions_during_gripper_transition"
    ] is False
    assert contact_close["require_bilateral_task_contact"] is True
    assert contact_close[
        "bilateral_task_contact_minimum_force_n"
    ] == pytest.approx(0.5)
    assert contact_close["maximum_steps"] == 39
    assert joint_path["target_position_world_m"] == pytest.approx(
        [0.6, 0.2, 0.4]
    )
    assert receipt["frontier_phase_ids"] == [
        MEASURED_CONTACT_ENTRY_PHASE_ID,
        "contact_open",
    ]
    assert receipt["promoted_standoff_m"] == pytest.approx(0.04)
    assert receipt["probe_clearance_axis_alignment_dot"] == pytest.approx(1.0)
    assert receipt["rewritten_grasp_holding_phase_ids"] == [
        "contact_open",
    ]
    assert receipt["measured_joint_vector_bound_phase_ids"] == [
        "contact_open",
    ]
    assert receipt["contact_close_step_budget_added"] == 27
    assert receipt["contact_close_maximum_steps"] == 39
    assert receipt["synthetic_frontier_rows_inserted"] == 0
    assert receipt["replaced_branch_replay_rows"] == 0
    assert derived["plan_digest"] == canonical_digest(
        derived, digest_field="plan_digest"
    )


def test_measured_contact_frontier_refuses_unproven_probe_cells() -> None:
    from blueprint_pipeline.native_task_arena_controls_worker import (
        _with_measured_contact_frontier,
    )

    plan = _branch_replay_plan(_branch_replay_task())
    derived, receipt = _with_measured_contact_frontier(
        control_plan=plan,
        reachability_probe={
            "cells": [
                {
                    "status": "measured",
                    "offset_m": [0.0, 0.0, -0.02],
                    "joint_positions_rad": [0.2] * 7,
                    "joint_tracking_error_rad": 0.0002,
                    "measured_distance_to_requested_m": 0.008,
                    "measured_grasp_frame_orientation_world_xyzw": [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    "contact_steps": 0,
                }
            ]
        },
    )

    assert receipt["status"] == "not_applied"
    assert receipt["reason"] == "no_noncontact_probe_cell_inside_arrival_gate"
    assert derived == plan


def test_contact_acquisition_adopts_only_physics_admitted_bilateral_cell() -> None:
    from blueprint_pipeline.adp009d_control_episode import (
        validate_task_control_plan,
    )
    from blueprint_pipeline.native_task_arena_controls_worker import (
        _with_contact_acquisition_candidate,
    )

    task = _branch_replay_task()
    plan = _branch_replay_plan(task)
    original_close = next(
        row
        for row in plan["scripted_positive_actions"]
        if row["phase_id"] == "contact_close"
    )
    original_close["require_bilateral_task_contact"] = True
    original_close["bilateral_task_contact_minimum_force_n"] = 0.5
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    reached_open = [0.11, -0.22, 0.33, -0.44, 0.55, -0.66, 0.77]

    derived, receipt = _with_contact_acquisition_candidate(
        control_plan=plan,
        sweep={
            "status": "measured",
            "best_cell": {
                "cell_index": 17,
                "admitted": True,
                "authored_target_gate_passed": True,
                "candidate_target_position_world_m": [0.501, 0.096, 0.402],
                "candidate_command_target_position_world_m": [
                    0.501,
                    0.0825,
                    0.402,
                ],
                "reached_open_joint_positions_rad": reached_open,
                "approach_offset_m": -0.005,
                "jaw_offset_m": 0.006,
                "lateral_offset_m": 0.0,
            },
        },
    )

    checked = validate_task_control_plan(derived, task_spec=task)
    rows = {
        row["phase_id"]: row for row in checked["scripted_positive_actions"]
    }
    contact_open = rows["contact_open"]
    contact_close = rows["contact_close"]

    assert receipt["status"] == "applied"
    assert receipt["adopted_cell_index"] == 17
    assert contact_open["target_position_world_m"] == pytest.approx(
        [0.501, 0.0825, 0.402]
    )
    assert contact_open["arrival_target_position_world_m"] == pytest.approx(
        [0.501, 0.0825, 0.402]
    )
    assert contact_open["hold_solved_arm_joint_positions_rad"] == pytest.approx(
        reached_open
    )
    assert contact_open["gripper_state"] == "open"
    assert contact_close[
        "hold_arm_joint_positions_during_gripper_transition"
    ] is True
    assert contact_close["target_position_world_m"] == pytest.approx(
        [0.501, 0.0825, 0.402]
    )
    assert contact_close["arrival_target_position_world_m"] == pytest.approx(
        [0.5, 0.1, 0.4]
    )
    assert receipt["authoritative_arrival_target_position_world_m"] == pytest.approx(
        [0.5, 0.1, 0.4]
    )
    assert contact_close["hold_solved_arm_joint_positions_rad"] == pytest.approx(
        reached_open
    )
    assert contact_close["require_bilateral_task_contact"] is True
    assert contact_close[
        "bilateral_task_contact_minimum_force_n"
    ] == pytest.approx(0.5)
    assert derived["plan_digest"] != plan["plan_digest"]
    assert derived["plan_digest"] == canonical_digest(
        derived, digest_field="plan_digest"
    )


def test_contact_acquisition_uses_command_fallback_for_null_optional_targets() -> None:
    from blueprint_pipeline.native_task_arena_controls_worker import (
        _with_contact_acquisition_candidate,
    )

    plan = _branch_replay_plan(_branch_replay_task())
    contact_close = next(
        row
        for row in plan["scripted_positive_actions"]
        if row["phase_id"] == "contact_close"
    )
    authored_close_target = list(contact_close["target_position_world_m"])
    contact_close["arrival_target_position_world_m"] = None
    contact_close["arrival_target_quaternion_world_xyzw"] = None
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    candidate_target = [0.501, 0.096, 0.402]

    derived, receipt = _with_contact_acquisition_candidate(
        control_plan=plan,
        sweep={
            "status": "measured",
            "best_cell": {
                "cell_index": 18,
                "admitted": True,
                "authored_target_gate_passed": True,
                "candidate_target_position_world_m": candidate_target,
                "candidate_command_target_position_world_m": None,
                "reached_open_joint_positions_rad": [0.1] * 7,
                "approach_offset_m": -0.005,
                "jaw_offset_m": 0.006,
                "lateral_offset_m": 0.0,
            },
        },
    )

    rows = {
        row["phase_id"]: row for row in derived["scripted_positive_actions"]
    }
    assert receipt["status"] == "applied"
    assert receipt["adopted_command_target_position_world_m"] == candidate_target
    assert receipt["authoritative_arrival_target_position_world_m"] == (
        authored_close_target
    )
    assert rows["contact_open"]["target_position_world_m"] == candidate_target
    assert rows["contact_close"]["arrival_target_position_world_m"] == (
        authored_close_target
    )


def test_contact_acquisition_axes_use_authored_approach_when_open_and_close_share_pose() -> None:
    from blueprint_pipeline.native_task_arena_controls_worker import (
        _contact_acquisition_axes,
    )

    plan = _branch_replay_plan(_branch_replay_task())
    rows = {
        row["phase_id"]: row for row in plan["scripted_positive_actions"]
    }
    rows["approach"]["target_position_world_m"] = [0.5, 0.0, 0.4]
    shared_contact = [0.5, 0.1, 0.4]
    rows["contact_open"]["target_position_world_m"] = shared_contact
    rows["contact_close"]["target_position_world_m"] = shared_contact

    approach, jaw, lateral = _contact_acquisition_axes(
        control_plan=plan,
        authored_open_target=shared_contact,
        authored_close_target=shared_contact,
        pad_centers={
            "left": [0.0, 0.0, 0.05],
            "right": [0.0, 0.0, -0.05],
        },
    )

    assert approach == pytest.approx([0.0, 1.0, 0.0])
    assert jaw == pytest.approx([0.0, 0.0, 1.0])
    assert lateral == pytest.approx([1.0, 0.0, 0.0])


def test_contact_acquisition_progress_is_atomic_and_timeout_harvestable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from blueprint_pipeline.native_task_arena_controls_worker import (
        _announce_contact_acquisition_cell,
        _persist_progress,
    )

    progress = {
        "schema_version": "native_task_arena_contact_acquisition_sweep.v1",
        "status": "running",
        "executed_cell_count": 8,
        "last_cell": {
            "cell_index": 7,
            "approach_offset_m": -0.005,
            "jaw_offset_m": 0.0,
            "lateral_offset_m": 0.006,
            "admitted": True,
            "maximum_consecutive_bilateral_steps": 2,
            "terminal_task_contact_pad_forces_n": {
                "left_inner_finger": 1.2,
                "right_inner_finger": 1.1,
            },
            "terminal_distance_to_candidate_target_m": 0.004,
            "terminal_distance_to_authored_target_m": 0.004,
            "terminal_orientation_error_rad": 0.05,
            "best_bilateral_force_evidence": {
                "opposed_jaw_force_min_n": 0.9,
                "same_direction_approach_force_min_n": 0.1,
            },
        },
    }
    output = tmp_path / "contact_acquisition_sweep.progress.v1.json"

    _persist_progress(output, progress)
    _announce_contact_acquisition_cell(progress)

    retained = json.loads(output.read_text(encoding="utf-8"))
    assert retained["executed_cell_count"] == 8
    assert retained["result_digest"].startswith("sha256:")
    assert not (tmp_path / f".{output.name}.tmp").exists()
    marker = capsys.readouterr().out.strip()
    assert marker.startswith("BLUEPRINT_CONTACT_ACQUISITION_PROGRESS:CELL:i=7:")
    assert ":ok=1:b=2:lf=1.2:rf=1.1:d=0.004:o=0.05" in marker
    assert marker.endswith(":ad=0.004:pj=0.9:fa=0.1")


def test_contact_acquisition_refuses_nonadmitted_best_cell() -> None:
    from blueprint_pipeline.native_task_arena_controls_worker import (
        _with_contact_acquisition_candidate,
    )

    plan = _branch_replay_plan(_branch_replay_task())
    derived, receipt = _with_contact_acquisition_candidate(
        control_plan=plan,
        sweep={
            "status": "measured",
            "best_cell": {
                "cell_index": 0,
                "admitted": False,
                "candidate_target_position_world_m": [0.5, 0.1, 0.4],
                "reached_open_joint_positions_rad": [0.1] * 7,
            },
        },
    )

    assert receipt["status"] == "not_applied"
    assert receipt["reason"] == "no_physics_admitted_contact_acquisition_cell"
    assert derived == plan


def test_contact_acquisition_refuses_admitted_cell_without_authored_gate() -> None:
    from blueprint_pipeline.native_task_arena_controls_worker import (
        _with_contact_acquisition_candidate,
    )

    plan = _branch_replay_plan(_branch_replay_task())
    derived, receipt = _with_contact_acquisition_candidate(
        control_plan=plan,
        sweep={
            "status": "measured",
            "best_cell": {
                "cell_index": 8,
                "admitted": True,
                "candidate_target_position_world_m": [0.49, 0.1, 0.4],
                "candidate_command_target_position_world_m": [0.49, 0.1, 0.4],
                "reached_open_joint_positions_rad": [0.1] * 7,
                "approach_offset_m": -0.01,
                "jaw_offset_m": 0.0,
                "lateral_offset_m": 0.0,
            },
        },
    )

    assert receipt["status"] == "not_applied"
    assert receipt["reason"] == "physics_admitted_cell_authored_gate_unproven"
    assert derived == plan


def test_measured_anchor_replaces_the_old_contact_replay() -> None:
    from blueprint_pipeline.native_task_arena_controls_worker import (
        CONTACT_ENTRY_BRANCH_REPLAY_PHASE_ID,
        MEASURED_CONTACT_ENTRY_MAXIMUM_STEPS,
        _with_contact_entry_branch_replay,
        _with_measured_contact_frontier,
    )

    task = _branch_replay_task()
    plan = _branch_replay_plan(task)
    original_contact = next(
        row
        for row in plan["scripted_positive_actions"]
        if row["phase_id"] == "contact_open"
    )
    original_contact["hold_solved_arm_joint_positions_rad"] = [0.9] * 7
    original_contact_maximum_steps = original_contact["maximum_steps"]
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    targets = [
        {
            "phase_id": row["phase_id"],
            "target_position_world_m": row["target_position_world_m"],
            "target_quaternion_world_xyzw": row["target_quaternion_world_xyzw"],
            "joint_positions_rad": [0.0] * 7
            if row["phase_id"] == "approach"
            else [0.4] * 7,
        }
        for row in plan["scripted_positive_actions"]
        if row["phase_id"] in {"approach", "contact_open"}
    ]
    replayed, replay_receipt = _with_contact_entry_branch_replay(
        control_plan=plan,
        scripted_pose_joint_targets=targets,
        task_spec=task,
        actuator_feasible_step_rad=[0.005] * 7,
    )
    assert replay_receipt["status"] == "applied"
    old_rows = [
        row
        for row in replayed["scripted_positive_actions"]
        if row["phase_id"] == CONTACT_ENTRY_BRANCH_REPLAY_PHASE_ID
    ]
    assert len(old_rows) > 1
    replayed_contact = next(
        row
        for row in replayed["scripted_positive_actions"]
        if row["phase_id"] == "contact_open"
    )

    derived, receipt = _with_measured_contact_frontier(
        control_plan=replayed,
        reclaimed_contact_steps=replay_receipt["contact_phase_steps_reclaimed"],
        reachability_probe={
            "cells": [
                {
                    "status": "measured",
                    "offset_m": [0.0, 0.0, -0.04],
                    "joint_positions_rad": [0.2] * 7,
                    "joint_tracking_error_rad": 0.0002,
                    "measured_distance_to_requested_m": 0.004,
                    "measured_grasp_frame_orientation_world_xyzw": [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    "contact_steps": 0,
                }
            ]
        },
    )
    new_rows = [
        row
        for row in derived["scripted_positive_actions"]
        if row["phase_id"] == CONTACT_ENTRY_BRANCH_REPLAY_PHASE_ID
    ]
    assert len(new_rows) == 1
    assert new_rows[0]["mode"] == "ik_pose"
    assert new_rows[0]["hold_solved_arm_joint_positions_rad"] == [0.2] * 7
    assert new_rows[0]["maximum_steps"] == len(old_rows)
    assert new_rows[0]["max_joint_delta_rad"] == pytest.approx(0.005)
    assert receipt["replaced_branch_replay_rows"] == len(old_rows)
    assert receipt["preserved_branch_replay_step_rad"] == pytest.approx(0.005)
    assert receipt["measured_entry_maximum_steps"] == len(old_rows)
    contact = next(
        row
        for row in derived["scripted_positive_actions"]
        if row["phase_id"] == "contact_open"
    )
    assert contact["hold_solved_arm_joint_positions_rad"] == [0.2] * 7
    assert contact["target_position_world_m"] == pytest.approx([0.5, 0.1, 0.36])
    expected_anchor_steps = max(
        MEASURED_CONTACT_ENTRY_MAXIMUM_STEPS,
        replayed_contact["maximum_steps"],
        len(old_rows),
    )
    expected_restoration = min(
        replay_receipt["contact_phase_steps_reclaimed"],
        max(0, len(old_rows) - expected_anchor_steps),
    )
    assert contact["maximum_steps"] == (
        replayed_contact["maximum_steps"] + expected_restoration
    )
    assert contact["maximum_steps"] < original_contact_maximum_steps
    assert receipt["restored_contact_steps"] == expected_restoration
    assert receipt["restoration_limited_by_action_budget"] is True

    from blueprint_pipeline.adp009d_control_episode import (
        validate_task_control_plan,
    )

    validated = validate_task_control_plan(derived, task_spec=task)
    assert validated["plan_digest"] == derived["plan_digest"]


def test_contact_anchor_is_derived_from_the_approach_line() -> None:
    from blueprint_pipeline.native_task_arena_controls_worker import (
        _contact_approach_anchor_offset,
    )

    plan = _branch_replay_plan(_branch_replay_task())
    rows = {row["phase_id"]: row for row in plan["scripted_positive_actions"]}
    rows["approach"]["target_position_world_m"] = [0.5, -0.02, 0.4]
    rows["contact_open"]["target_position_world_m"] = [0.5, 0.1, 0.4]

    assert _contact_approach_anchor_offset(plan) == pytest.approx(
        [0.0, -0.04, 0.0]
    )


def test_contact_frontier_walks_from_the_proven_anchor_to_authored_contact() -> None:
    from blueprint_pipeline.native_task_arena_controls_worker import (
        _contact_frontier_offsets,
    )

    offsets = _contact_frontier_offsets([0.0, -0.04, 0.0], sample_count=5)
    expected = [
        [0.0, -0.04, 0.0],
        [0.0, -0.03, 0.0],
        [0.0, -0.02, 0.0],
        [0.0, -0.01, 0.0],
        [0.0, 0.0, 0.0],
    ]
    assert len(offsets) == len(expected)
    for observed, row in zip(offsets, expected, strict=True):
        assert observed == pytest.approx(row)
