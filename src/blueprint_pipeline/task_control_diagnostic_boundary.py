"""Fail-closed claim boundaries for nonqualifying native control diagnostics."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Callable


DIAGNOSTIC_TRAJECTORY_SOURCE = "native_ik_diagnostic_unqualified"
DIAGNOSTIC_PLAN_CLAIM_BOUNDARY = (
    "blocked_construction_downstream_execution_only;cannot_qualify_"
    "construction_controls_policy_admission_or_task_success"
)
DIAGNOSTIC_RECEIPT_CLAIM_BOUNDARY = (
    "diagnostic_execution_only;cannot_qualify_controls_"
    "policy_admission_or_task_success"
)


def control_plan_boundary_errors(plan: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Return diagnostic status and any fail-closed boundary violations."""

    trajectory_source = plan.get("trajectory_source")
    diagnostic = trajectory_source == DIAGNOSTIC_TRAJECTORY_SOURCE
    errors: list[str] = []
    if trajectory_source not in {"native_ik_preflight", DIAGNOSTIC_TRAJECTORY_SOURCE}:
        errors.append("task_control_trajectory_source_invalid")
    if diagnostic:
        upstream = plan.get("upstream_construction_blockers")
        if (
            plan.get("diagnostic_only") is not True
            or plan.get("qualification_allowed") is not False
            or plan.get("qualification_effect") != "none"
            or not isinstance(upstream, list)
            or not upstream
            or any(not isinstance(value, str) or not value for value in upstream)
            or plan.get("claim_boundary") != DIAGNOSTIC_PLAN_CLAIM_BOUNDARY
        ):
            errors.append("task_control_diagnostic_boundary_invalid")
    elif "qualification_allowed" in plan and plan.get("qualification_allowed") is not True:
        errors.append("task_control_qualification_boundary_invalid")
    return diagnostic, errors


def diagnostic_receipt_annotations(
    plan: Mapping[str, Any], *, qualification_allowed: bool
) -> dict[str, Any] | None:
    """Bind diagnostic receipts to the refused construction evidence."""

    if qualification_allowed:
        return None
    return {
        "upstream_construction_blockers": list(
            plan.get("upstream_construction_blockers") or []
        ),
        "control_plan_digest": plan["plan_digest"],
    }


def apply_diagnostic_receipt_boundary(
    receipt: dict[str, Any], *, qualification_allowed: bool
) -> None:
    if qualification_allowed:
        return
    receipt.update(
        {
            "qualification_allowed": False,
            "development_only": True,
            "diagnostic_only": True,
            "claim_boundary": DIAGNOSTIC_RECEIPT_CLAIM_BOUNDARY,
        }
    )


def copy_diagnostic_annotations(
    receipt: dict[str, Any], annotations: Mapping[str, Any] | None
) -> None:
    if annotations is not None:
        receipt["diagnostic_annotations"] = json.loads(
            json.dumps(dict(annotations), allow_nan=False)
        )


def apply_diagnostic_pair_boundary(
    pair: dict[str, Any], *, qualification_allowed: bool
) -> None:
    if not qualification_allowed:
        pair.update({"qualification_allowed": False, "diagnostic_only": True})


def build_task_control_pair(
    *,
    plan: Mapping[str, Any],
    task: Mapping[str, Any],
    receipts: list[Mapping[str, Any]],
    qualification_allowed: bool,
    required_controls: tuple[str, ...],
    canonical_digest: Callable[..., str],
) -> dict[str, Any]:
    """Seal the paired-control summary without weakening diagnostic refusal."""

    blockers = [
        blocker for receipt in receipts for blocker in receipt.get("blockers", [])
    ]
    pair: dict[str, Any] = {
        "schema_version": "adp_task_control_pair.v1",
        "program_id": "arm-decision-proof-v1",
        "cell_id": plan["cell_id"],
        "task_kind": task["task_kind"],
        "task_spec_digest": plan["task_spec_digest"],
        "control_plan_digest": plan["plan_digest"],
        "execution_order": list(required_controls),
        "controls": [
            {
                "control_id": receipt["control_id"],
                "control_passed": receipt["control_passed"],
                "observed_outcome": receipt["observed_outcome"],
                "receipt_digest": receipt["receipt_digest"],
            }
            for receipt in receipts
        ],
        "cell_admitted_for_policy_execution": qualification_allowed and not blockers,
        "policy_execution_blockers": sorted(
            set(
                blockers
                + (
                    ["diagnostic_controls_cannot_admit_policy_execution"]
                    if not qualification_allowed
                    else []
                )
            )
        ),
        "candidate_policy_queried": False,
        "pair_digest": "",
    }
    apply_diagnostic_pair_boundary(pair, qualification_allowed=qualification_allowed)
    pair["pair_digest"] = canonical_digest(pair, digest_field="pair_digest")
    return pair
