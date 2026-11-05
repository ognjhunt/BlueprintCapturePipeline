"""Operator-door entry for one owner-approved scene execution budget grant.

Run by ``deploy/operator-door/door-extend-scene-execution-budget.sh`` as the
scene service user with the active release. It is the door form of the host
operator CLI in ``task_evaluation_scene_execution_budget``: a preview names the
current and requested limits and their digest; an apply must repeat that digest
and records the grant through ``extend_scene_execution_budget`` (the only
write), naming the approving owner, the approval evidence, the door caller and
the door request.

A grant raises the scene's spend and attempt ceilings; nothing here refunds,
settles or rewrites a hold, starts compute or calls a provider.
"""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def run(request: Mapping[str, Any], *, door_caller: str, door_request_id: str) -> dict[str, Any]:
    from . import task_evaluation_scene_intake as intake
    from .task_evaluation_scene_execution_budget import extend_scene_execution_budget
    from .task_evaluation_scene_execution_budget_evidence import (
        ACK, _limits, _require, effective_execution_budget,
    )
    from .task_evaluation_scene_intent_contracts import CLIENTS_ENV, ROOT_ENV

    intent_id = request["intent_id"]
    _require(bool(intake._identifier(intent_id)), "not_authorized")
    root = intake._root(Path(os.environ[ROOT_ENV]))
    intent = intake._read(root / intent_id / "intent.json", "intent_digest")
    owner = {"user_id": request["owner_user_id"], "organization_id": request["owner_organization_id"]}
    evidence = str(request["approval_evidence"]).strip()
    _require(intent["request"]["owner"] == owner and request["approved_by"] == owner["user_id"]
             and bool(evidence), "owner_or_revocation_mismatch")
    preview = {"intent_id": intent_id, "intent_digest": intent["intent_digest"], "owner": owner,
               "issuer": intent["authenticated_issuer"],
               "current": effective_execution_budget(root / intent_id, intent),
               "requested": _limits({"max_total_spend_usd": request["max_total_spend_usd"],
                                     "max_paid_attempts": request["max_paid_attempts"]})}
    apply_ack = intake.canonical_digest(preview)
    if not request["apply"]:
        return {"door_outcome": "planned", "mode": "preview", **preview, "apply_ack": apply_ack}
    _require(request["apply_ack"] == apply_ack, "budget_apply_ack_mismatch")
    trusted = {value.strip() for value in os.getenv(CLIENTS_ENV, "blueprint-webapp").split(",") if value.strip()}
    reference = (f"owner_approval:approved_by={request['approved_by']};evidence={evidence};"
                 f"channel=operator_door;door_caller={door_caller};door_request_id={door_request_id}")
    result = extend_scene_execution_budget(
        queue_root=root, intent_id=intent_id, intent_digest=intent["intent_digest"], owner=owner,
        authenticated_client=intent["authenticated_issuer"], trusted_clients=trusted,
        max_total_spend_usd=request["max_total_spend_usd"], max_paid_attempts=request["max_paid_attempts"],
        authorization_reference=reference, ack=ACK, host_operator=True)
    return {"door_outcome": "extended", "mode": "applied", **result}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--door-caller", required=True)
    parser.add_argument("--door-request-id", required=True)
    args = parser.parse_args(argv)
    try:
        report = run(json.loads(args.request_json), door_caller=args.door_caller,
                     door_request_id=args.door_request_id)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        code = str(exc).split(":", 1)[0] if isinstance(exc, ValueError) else type(exc).__name__
        print(json.dumps({"door_outcome": "refused", "code": code[:200]}, sort_keys=True))
        return 2
    print(json.dumps(report, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
