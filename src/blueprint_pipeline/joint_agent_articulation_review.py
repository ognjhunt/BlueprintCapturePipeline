"""Deterministically admit Joint Agent topology before owned-core publication.

Joint Agent is a model-backed research preview.  Its candidate document is
therefore an input to this gate, never its own success receipt.  The gate binds
the candidate graph to a preregistered task-joint axis and an independently
computed moving-link bounds interval while allowing bounded non-task joints.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "joint_agent_articulation_review.v1"


class JointAgentArticulationReviewError(ValueError):
    """Stable, sorted topology-review failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _finite_vector(value: Any, *, length: int) -> tuple[float, ...] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != length
    ):
        return None
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        number = float(item)
        if not math.isfinite(number):
            return None
        result.append(number)
    return tuple(result)


def _normalized_axis(value: Any) -> tuple[float, float, float] | None:
    vector = _finite_vector(value, length=3)
    if vector is None:
        return None
    norm = math.sqrt(sum(item * item for item in vector))
    if norm <= 1e-12:
        return None
    return tuple(item / norm for item in vector)  # type: ignore[return-value]


def _clone(value: Mapping[str, Any], *, error: str) -> dict[str, Any]:
    try:
        cloned = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise JointAgentArticulationReviewError([error]) from exc
    if not isinstance(cloned, dict):
        raise JointAgentArticulationReviewError([error])
    return cloned


def review_joint_agent_articulation(
    *,
    candidates_document: Mapping[str, Any],
    candidate_bounds: Mapping[str, Any],
    review_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Admit one task joint without constraining the assembly to one joint.

    ``candidate_bounds`` must be computed from the optimized USD by the runtime;
    each candidate ID maps to ``{"aabb_min": [x,y,z], "aabb_max": [x,y,z]}``.
    A caller-provided label or model confidence alone cannot select the task
    joint.
    """

    document = _clone(candidates_document, error="joint_candidates_not_json")
    bounds = _clone(candidate_bounds, error="joint_candidate_bounds_not_json")
    contract = _clone(review_contract, error="joint_review_contract_not_json")
    errors: list[str] = []
    if document.get("schema_version") != "joint-agent-stage2-v0":
        errors.append("joint_candidates_schema_invalid")
    candidates = document.get("candidates")
    if not isinstance(candidates, list):
        candidates = []
        errors.append("joint_candidates_list_invalid")
    summary = document.get("summary")
    if not isinstance(summary, Mapping):
        errors.append("joint_candidates_summary_missing")
    elif summary.get("candidate_count") != len(candidates):
        errors.append("joint_candidates_summary_count_mismatch")

    max_joints = contract.get("maximum_assembly_joint_count")
    if isinstance(max_joints, bool) or not isinstance(max_joints, int) or max_joints < 1:
        errors.append("joint_review_maximum_joint_count_invalid")
        max_joints = 0
    if not 1 <= len(candidates) <= max_joints:
        errors.append("joint_candidate_count_outside_preregistered_bounds")

    allowed = contract.get("allowed_joint_types")
    if (
        not isinstance(allowed, list)
        or not allowed
        or any(item not in {"revolute", "prismatic"} for item in allowed)
    ):
        errors.append("joint_review_allowed_types_invalid")
        allowed_types: set[str] = set()
    else:
        allowed_types = set(allowed)
    target_type = contract.get("target_joint_type")
    if target_type not in allowed_types:
        errors.append("joint_review_target_type_invalid")
    target_axis = _normalized_axis(contract.get("target_axis_world"))
    if target_axis is None:
        errors.append("joint_review_target_axis_invalid")
    axis_abs_dot_min = contract.get("target_axis_absolute_dot_minimum")
    if (
        isinstance(axis_abs_dot_min, bool)
        or not isinstance(axis_abs_dot_min, (int, float))
        or not 0.0 < float(axis_abs_dot_min) <= 1.0
    ):
        errors.append("joint_review_axis_threshold_invalid")
        axis_abs_dot_min = 2.0
    target_interval = _finite_vector(contract.get("target_moving_z_interval_m"), length=2)
    if target_interval is None or target_interval[0] >= target_interval[1]:
        errors.append("joint_review_target_interval_invalid")
        target_interval = (math.inf, -math.inf)
    minimum_overlap = contract.get("minimum_target_z_overlap_fraction")
    if (
        isinstance(minimum_overlap, bool)
        or not isinstance(minimum_overlap, (int, float))
        or not 0.0 < float(minimum_overlap) <= 1.0
    ):
        errors.append("joint_review_overlap_threshold_invalid")
        minimum_overlap = 2.0

    rows: list[dict[str, Any]] = []
    target_matches: list[str] = []
    seen_ids: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            errors.append(f"joint_candidate_invalid:{index}")
            continue
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id or candidate_id in seen_ids:
            errors.append("joint_candidate_id_missing_or_duplicate")
            continue
        seen_ids.add(candidate_id)
        joint_type = str(candidate.get("joint_type_hint") or "")
        if joint_type not in allowed_types:
            errors.append(f"joint_candidate_type_not_admitted:{candidate_id}")
        if candidate.get("review_status") != "ready_for_rigger_input":
            errors.append(f"joint_candidate_not_rigger_ready:{candidate_id}")
        if candidate.get("unresolved_reason_codes") not in ([], None):
            errors.append(f"joint_candidate_unresolved:{candidate_id}")
        axis = _normalized_axis(candidate.get("motion_axis_world"))
        axis_dot = (
            abs(sum(left * right for left, right in zip(axis, target_axis, strict=True)))
            if axis is not None and target_axis is not None
            else None
        )
        bound = bounds.get(candidate_id)
        bound_min = _finite_vector(bound.get("aabb_min"), length=3) if isinstance(bound, Mapping) else None
        bound_max = _finite_vector(bound.get("aabb_max"), length=3) if isinstance(bound, Mapping) else None
        if (
            bound_min is None
            or bound_max is None
            or any(low > high for low, high in zip(bound_min, bound_max, strict=True))
        ):
            errors.append(f"joint_candidate_bounds_invalid:{candidate_id}")
            overlap_fraction = None
        else:
            overlap = max(
                0.0,
                min(bound_max[2], target_interval[1])
                - max(bound_min[2], target_interval[0]),
            )
            overlap_fraction = overlap / (target_interval[1] - target_interval[0])
        matches = (
            joint_type == target_type
            and axis_dot is not None
            and axis_dot >= float(axis_abs_dot_min)
            and overlap_fraction is not None
            and overlap_fraction >= float(minimum_overlap)
        )
        if matches:
            target_matches.append(candidate_id)
        rows.append(
            {
                "candidate_id": candidate_id,
                "joint_type": joint_type,
                "axis_absolute_dot": axis_dot,
                "target_z_overlap_fraction": overlap_fraction,
                "target_match": matches,
            }
        )
    if len(target_matches) != 1:
        errors.append("exactly_one_task_joint_not_resolved")
    if errors:
        raise JointAgentArticulationReviewError(errors)

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "accepted_for_owned_core_topology_publication",
        "candidate_document_digest": canonical_digest(document),
        "candidate_bounds_digest": canonical_digest(bounds),
        "review_contract_digest": canonical_digest(contract),
        "assembly_joint_count": len(candidates),
        "target_candidate_id": target_matches[0],
        "non_task_candidate_ids": sorted(seen_ids - set(target_matches)),
        "candidate_review": rows,
        "claim_boundary": {
            "deterministic_review_is_not_model_accuracy_proof": True,
            "topology_publication_is_not_simready_qualification": True,
            "physical_equivalence_proven": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


__all__ = [
    "JointAgentArticulationReviewError",
    "SCHEMA_VERSION",
    "review_joint_agent_articulation",
]
