from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.joint_agent_articulation_review import (
    JointAgentArticulationReviewError,
    review_joint_agent_articulation,
)


def _contract() -> dict:
    return {
        "maximum_assembly_joint_count": 4,
        "allowed_joint_types": ["revolute", "prismatic"],
        "target_joint_type": "revolute",
        "target_axis_world": [0.0, 0.0, 1.0],
        "target_axis_absolute_dot_minimum": 0.99,
        "target_member_projection_constraints": [
            {
                "axis_world": [0.0, 0.0, 1.0],
                "interval_m": [0.94, 1.632],
                "minimum_overlap_fraction": 0.85,
            }
        ],
    }


def _candidate(candidate_id: str, *, axis: list[float], kind: str = "revolute") -> dict:
    return {
        "schema_version": "joint-agent-stage2-v0",
        "candidate_id": candidate_id,
        "joint_type_hint": kind,
        "motion_axis_world": axis,
        "review_status": "ready_for_rigger_input",
        "unresolved_reason_codes": [],
    }


def test_review_admits_one_task_joint_inside_bounded_multi_joint_assembly() -> None:
    candidates = [
        _candidate("upper_door", axis=[0.0, 0.0, -1.0]),
        _candidate("lower_door", axis=[0.0, 0.0, 1.0]),
        _candidate("drawer", axis=[1.0, 0.0, 0.0], kind="prismatic"),
    ]
    document = {
        "schema_version": "joint-agent-stage2-v0",
        "summary": {"candidate_count": len(candidates)},
        "candidates": candidates,
    }
    bounds = {
        "upper_door": {"aabb_min": [-0.36, 0.17, 0.94], "aabb_max": [0.36, 0.35, 1.632]},
        "lower_door": {"aabb_min": [-0.36, 0.17, 0.03], "aabb_max": [0.36, 0.35, 0.92]},
        "drawer": {"aabb_min": [-0.2, -0.2, 0.2], "aabb_max": [0.2, 0.2, 0.4]},
    }

    receipt = review_joint_agent_articulation(
        candidates_document=document,
        candidate_bounds=bounds,
        review_contract=_contract(),
    )

    assert receipt["assembly_joint_count"] == 3
    assert receipt["target_candidate_id"] == "upper_door"
    assert receipt["non_task_candidate_ids"] == ["drawer", "lower_door"]
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )


def test_review_supports_prismatic_task_selected_on_arbitrary_world_axis() -> None:
    contract = _contract()
    contract["target_joint_type"] = "prismatic"
    contract["target_axis_world"] = [1.0, 0.0, 0.0]
    contract["target_member_projection_constraints"] = [
        {
            "axis_world": [0.0, 1.0, 0.0],
            "interval_m": [2.0, 2.4],
            "minimum_overlap_fraction": 0.9,
        }
    ]
    candidates = [
        _candidate("left_drawer", axis=[1.0, 0.0, 0.0], kind="prismatic"),
        _candidate("right_drawer", axis=[1.0, 0.0, 0.0], kind="prismatic"),
    ]
    document = {
        "schema_version": "joint-agent-stage2-v0",
        "summary": {"candidate_count": 2},
        "candidates": candidates,
    }
    bounds = {
        "left_drawer": {"aabb_min": [0.0, 2.0, 0.1], "aabb_max": [0.8, 2.4, 0.4]},
        "right_drawer": {"aabb_min": [0.0, 3.0, 0.1], "aabb_max": [0.8, 3.4, 0.4]},
    }

    receipt = review_joint_agent_articulation(
        candidates_document=document,
        candidate_bounds=bounds,
        review_contract=contract,
    )

    assert receipt["target_candidate_id"] == "left_drawer"


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("ambiguous_upper", "exactly_one_task_joint_not_resolved"),
        ("too_many", "joint_candidate_count_outside_preregistered_bounds"),
        ("unresolved", "joint_candidate_not_rigger_ready"),
        ("missing_bounds", "joint_candidate_bounds_invalid"),
    ],
)
def test_review_fails_closed_before_model_authored_topology(
    mutation: str, error: str
) -> None:
    candidates = [
        _candidate("upper_door", axis=[0.0, 0.0, 1.0]),
        _candidate("lower_door", axis=[0.0, 0.0, 1.0]),
    ]
    bounds = {
        "upper_door": {"aabb_min": [-1.0, -1.0, 0.94], "aabb_max": [1.0, 1.0, 1.632]},
        "lower_door": {"aabb_min": [-1.0, -1.0, 0.03], "aabb_max": [1.0, 1.0, 0.92]},
    }
    if mutation == "ambiguous_upper":
        bounds["lower_door"] = copy.deepcopy(bounds["upper_door"])
    elif mutation == "too_many":
        for index in range(3):
            candidate_id = f"extra_{index}"
            candidates.append(_candidate(candidate_id, axis=[1.0, 0.0, 0.0], kind="prismatic"))
            bounds[candidate_id] = copy.deepcopy(bounds["lower_door"])
    elif mutation == "unresolved":
        candidates[0]["review_status"] = "review_required"
        candidates[0]["unresolved_reason_codes"] = ["axis_unresolved"]
    elif mutation == "missing_bounds":
        bounds.pop("upper_door")
    document = {
        "schema_version": "joint-agent-stage2-v0",
        "summary": {"candidate_count": len(candidates)},
        "candidates": candidates,
    }

    with pytest.raises(JointAgentArticulationReviewError, match=error):
        review_joint_agent_articulation(
            candidates_document=document,
            candidate_bounds=bounds,
            review_contract=_contract(),
        )
