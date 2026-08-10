from __future__ import annotations

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_articulated_construction_plan import (
    NativeArticulatedConstructionPlanError,
    materialize_articulated_construction_phase_plan,
)


def _motion() -> dict:
    motion = {
        "schema_version": "native_articulated_motion_geometry.v1",
        "target_joint_id": "door",
        "hinge_point_world_m": [0.0, 0.0, 0.0],
        "hinge_axis_world_unit": [0.0, 0.0, 1.0],
        "handle_grasp_point_closed_world_m": [0.5, 0.0, 0.8],
        "authored_limits_degrees": [0.0, 90.0],
        "scripted_sweep_angle_degrees": 50.0,
        "motion_geometry_digest": "",
    }
    motion["motion_geometry_digest"] = canonical_digest(
        motion, digest_field="motion_geometry_digest"
    )
    return motion


def _scene(*, kind: str = "articulated_open_close") -> dict:
    return {
        "task_kind": kind,
        "plan_digest": "sha256:" + "a" * 64,
        "articulation": {"motion_geometry": _motion()},
    }


def test_phase_plan_covers_full_door_sweep_without_claiming_contact() -> None:
    plan = materialize_articulated_construction_phase_plan(
        _scene(), clearance_m=0.025, waypoint_count=8
    )

    assert plan["phase_count"] == 12
    assert [row["phase_id"] for row in plan["phases"][:3]] == [
        "approach",
        "grasp_clearance",
        "sweep_clearance_01",
    ]
    grasp = plan["phases"][1]
    assert grasp["position_world_m"] == pytest.approx([0.5, 0.025, 0.8])
    assert plan["phases"][-2]["phase_id"] == "retreat"
    assert plan["phases"][-1]["phase_id"] == "recovery"
    assert plan["claim_boundary"]["targets_are_contact_clear_not_a_grasp"] is True


def test_original_rigid_fixture_cannot_silently_take_articulated_path() -> None:
    with pytest.raises(NativeArticulatedConstructionPlanError) as excinfo:
        materialize_articulated_construction_phase_plan(
            _scene(kind="rigid_pick_place")
        )

    assert excinfo.value.errors == (
        "native_articulated_construction_task_kind_invalid",
    )


def test_motion_geometry_tamper_is_rejected() -> None:
    scene = _scene()
    scene["articulation"]["motion_geometry"]["hinge_point_world_m"][0] = 1.0

    with pytest.raises(NativeArticulatedConstructionPlanError) as excinfo:
        materialize_articulated_construction_phase_plan(scene)

    assert excinfo.value.errors == (
        "native_articulated_construction_motion_geometry_invalid",
    )
