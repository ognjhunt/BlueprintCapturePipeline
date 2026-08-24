from __future__ import annotations

import math

import pytest

from blueprint_pipeline.native_task_arena_bounded_orientation import (
    MEASURED_RESIDUAL_DIRECTION_BODY,
    apply_body_rotation_vector_xyzw,
    build_bounded_orientation_postures,
    default_body_rotation_vectors_rad,
)


def _plan(quaternion):
    return {
        "scripted_positive_actions": [
            {
                "phase_id": phase_id,
                "mode": "ik_pose",
                "target_position_world_m": [1.0, 2.0, z],
                "arrival_target_position_world_m": [1.0, 2.0, z + 0.01],
                "target_quaternion_world_xyzw": list(quaternion),
            }
            for phase_id, z in (("contact_open", 3.0), ("contact_close", 3.1))
        ]
    }


def test_default_shell_is_symmetric_and_covers_measured_combined_axis() -> None:
    vectors = default_body_rotation_vectors_rad()

    assert len(vectors) == 20
    assert len({tuple(round(value, 12) for value in row) for row in vectors}) == 20
    assert all(tuple(-value for value in row) in vectors for row in vectors)
    residual_norm = math.sqrt(
        sum(value * value for value in MEASURED_RESIDUAL_DIRECTION_BODY)
    )
    residual = tuple(
        value / residual_norm for value in MEASURED_RESIDUAL_DIRECTION_BODY
    )
    directions = [
        tuple(value / math.sqrt(sum(axis * axis for axis in row)) for value in row)
        for row in vectors
    ]
    assert max(
        abs(sum(left * right for left, right in zip(row, residual, strict=True)))
        for row in directions
    ) == pytest.approx(1.0)


def test_search_binds_open_and_close_bias_but_keeps_authority_unchanged() -> None:
    nominal = [0.0, 0.0, 0.0, 1.0]
    parallel = [0.0, 2**-0.5, 2**-0.5, 0.0]
    calls = []

    def solve(phase_id, position, quaternion, seeds):
        calls.append((phase_id, list(position), list(quaternion), [list(row) for row in seeds]))
        return {
            "joint_positions_rad": [float(len(calls))] * 7,
            "minimum_joint_limit_margin_rad": 0.2,
            "seed_index": 0,
        }

    postures, report = build_bounded_orientation_postures(
        variant_plans=(
            ("normalized_nominal", _plan(nominal)),
            ("parallel_jaw_equivalent", _plan(parallel)),
        ),
        solve_phase=solve,
        reference_joint_seeds=[[0.25] * 7],
        rotation_vectors_body_rad=((0.008, 0.004, 0.002),),
    )

    assert report["represented_candidate_count"] == 2
    assert report["solved_candidate_count"] == 2
    assert len(postures) == 2
    assert [row[0] for row in calls] == [
        "contact_open",
        "contact_close",
        "contact_open",
        "contact_close",
    ]
    assert calls[0][3][0] == [0.25] * 7
    assert calls[1][3][0] == [1.0] * 7
    by_variant = {row["variant_id"]: row for row in postures}
    assert by_variant["normalized_nominal"][
        "authoritative_target_quaternion_world_xyzw"
    ] == nominal
    assert by_variant["parallel_jaw_equivalent"][
        "authoritative_target_quaternion_world_xyzw"
    ] == parallel
    assert all(
        row["candidate_command_target_quaternion_world_xyzw"]
        != row["authoritative_target_quaternion_world_xyzw"]
        for row in postures
    )
    assert all(
        row["candidate_command_target_position_world_m"] == [1.0, 2.0, 3.0]
        and row["authoritative_target_position_world_m"] == [1.0, 2.0, 3.01]
        and row["bounded_orientation_candidate"]["position_offset_world_m"]
        == [0.0, 0.0, 0.0]
        for row in postures
    )
    assert all(
        row["bounded_orientation_candidate"]["close_joint_positions_rad"]
        for row in postures
    )


def test_body_rotation_is_target_local_and_canonicalizes_quaternion_sign() -> None:
    positive = apply_body_rotation_vector_xyzw(
        [0.0, 0.0, 0.0, 1.0], [0.01, 0.02, 0.03]
    )
    negative_base = apply_body_rotation_vector_xyzw(
        [0.0, 0.0, 0.0, -1.0], [0.01, 0.02, 0.03]
    )

    assert negative_base == pytest.approx(positive)
    assert math.sqrt(sum(value * value for value in positive)) == pytest.approx(1.0)
