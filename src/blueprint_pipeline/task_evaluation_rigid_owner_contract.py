"""Seal configured owner criteria after native destination geometry is joined.

This materializer does not confirm a proposal or select numerical thresholds.
The retained configured task must supply an explicit confirmed owner authority
and every additional temporal limit before the compiler can seal its contract.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .adp_rigid_retreat_scoring import materialize_retreat_criterion
from .decision_evidence_contracts import canonical_digest
from .adp_task_scoring import (
    TaskNeutralScoringError,
    _compatibility_rigid_success_criteria,
    seal_rigid_task_success_contract,
    confirm_rigid_task_success_contract,
)


def materialize_configured_owner_success_contract(
    task_spec: Mapping[str, Any], *, site_id: str, task_id: str, team_namespace: str | None = None
) -> dict[str, Any] | None:
    configured = task_spec.get("configured_success_criteria") or {}
    if configured.get("owner_success_contract_required") is not True:
        return None
    authority = task_spec.get("configured_owner_authority") or {}
    agent_proposal = authority.get("author_source") == "agent_proposal"
    if (authority.get("author_source", "task_owner") not in {"task_owner", "agent_proposal"}
            or authority.get("confirmation_status") != "confirmed"
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
    if agent_proposal:
        proposal = authority.get("agent_proposal")
        proposal_digest = authority.get("proposal_digest")
        confirmed_team = authority.get("confirmed_by_team_id")
        if (not isinstance(proposal, Mapping)
                or proposal_digest != canonical_digest(proposal, digest_field="proposal_digest")
                or not isinstance(authority.get("delegation_authority_reference"), str)
                or not authority["delegation_authority_reference"].strip()
                or not isinstance(authority.get("author_id"), str)
                or not authority["author_id"].strip()
                or not isinstance(confirmed_team, str) or not confirmed_team.strip()
                or team_namespace is None or confirmed_team != team_namespace
                or not isinstance(proposal.get("success"), Mapping)
                or any(proposal["success"].get(field) != configured[field]
                       for field in (*required, "minimum_lift_m"))):
            raise TaskNeutralScoringError(["configured_owner_success_contract_agent_authority_invalid"])
        proposed = seal_rigid_task_success_contract(
            task_spec=task_spec, site_id=site_id, task_id=task_id,
            author_source="agent_proposal", author_id=authority["author_id"] + ":" + proposal_digest,
            confirmation_status="proposal_only", criteria=criteria,
        )
        return confirm_rigid_task_success_contract(proposed, confirmed_by_team_id=confirmed_team)
    return seal_rigid_task_success_contract(
        task_spec=task_spec, site_id=site_id, task_id=task_id,
        author_source="task_owner",
        author_id=authority["authority_reference"],
        confirmation_status="confirmed", confirmed_by_team_id=authority["accepted_by"],
        criteria=criteria,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Seal the configured owner success contract for one native task spec."""

    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-spec", required=True, help="native task spec JSON")
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--team-namespace", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    task_spec = json.loads(Path(args.task_spec).read_text(encoding="utf-8"))
    contract = materialize_configured_owner_success_contract(
        task_spec, site_id=args.site_id, task_id=args.task_id, team_namespace=args.team_namespace
    )
    result = {
        "owner_success_contract": contract,
        "retreat_criterion": (
            materialize_retreat_criterion(task_spec)
            if contract is not None and task_spec.get("destination_pose_world") is not None
            else None
        ),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "sealed": contract is not None}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
