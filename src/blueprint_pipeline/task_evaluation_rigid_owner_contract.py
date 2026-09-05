"""Seal configured owner criteria after native destination geometry is joined.

This materializer does not confirm a proposal or select numerical thresholds.
The retained configured task must supply an explicit confirmed owner authority
and every additional temporal limit before the compiler can seal its contract.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .adp_rigid_retreat_scoring import materialize_retreat_criterion
from .adp_task_scoring import (
    TaskNeutralScoringError,
    _compatibility_rigid_success_criteria,
    seal_rigid_task_success_contract,
)


def materialize_configured_owner_success_contract(
    task_spec: Mapping[str, Any], *, site_id: str, task_id: str
) -> dict[str, Any] | None:
    configured = task_spec.get("configured_success_criteria") or {}
    if configured.get("owner_success_contract_required") is not True:
        return None
    authority = task_spec.get("configured_owner_authority") or {}
    if (authority.get("confirmation_status") != "confirmed"
            or not isinstance(authority.get("accepted_by"), str)
            or not authority["accepted_by"].strip()
            or not isinstance(authority.get("authority_reference"), str)
            or not authority["authority_reference"].strip()):
        raise TaskNeutralScoringError(["configured_owner_success_contract_authority_missing"])
    required = ("drop_minimum_fall_m", "maximum_task_contact_force_n",
                "forbidden_contact_classes", "maximum_retries", "maximum_regrasps",
                "retreat_clearance_m", "robot_workspace_position_bounds_world_m",
                "collision_failure_minimum_force_n")
    missing = [f"configured_owner_success_contract_explicit_field_missing:{field}"
               for field in required if field not in configured]
    if missing:
        raise TaskNeutralScoringError(missing)
    for field in ("retreat_clearance_m", "robot_workspace_position_bounds_world_m",
                  "collision_failure_minimum_force_n", "minimum_lift_m"):
        if task_spec.get(field) != configured[field]:
            raise TaskNeutralScoringError([f"configured_owner_success_contract_native_field_mismatch:{field}"])
    criteria = _compatibility_rigid_success_criteria(task_spec)
    criteria["terminal_task_contact"]["mode"] = "cleared"
    criteria["motion"]["minimum_lift_m"] = configured["minimum_lift_m"]
    temporal = criteria["temporal_invariants"]
    temporal.update(
        no_drop={"mode": "required", "minimum_fall_m": configured["drop_minimum_fall_m"]},
        maximum_task_contact_force_n=configured["maximum_task_contact_force_n"],
        forbidden_contact_classes=configured["forbidden_contact_classes"],
        containment_excursions="forbidden", workspace_excursions="forbidden",
        maximum_retries=configured["maximum_retries"],
        maximum_regrasps=configured["maximum_regrasps"],
    )
    criteria["retreat"] = materialize_retreat_criterion(task_spec)
    return seal_rigid_task_success_contract(
        task_spec=task_spec, site_id=site_id, task_id=task_id,
        author_source="task_owner",
        author_id=authority["authority_reference"],
        confirmation_status="confirmed", confirmed_by_team_id=authority["accepted_by"],
        criteria=criteria,
    )
