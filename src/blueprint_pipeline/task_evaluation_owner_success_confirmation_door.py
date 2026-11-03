"""Operator-door entry for one owner confirmation of a website job's built success thresholds.

Run by ``deploy/operator-door/door-confirm-owner-success-contract.sh`` as the
scene service user with the active release. A preview names the published
revision, the exact thresholds the job was built with, the ``proposal_only``
authority they carry and an ``apply_ack`` digest of all of it; an apply must
repeat that digest and writes one sealed record through
``confirm_owner_success_contract`` (the only write), naming the approving owner,
the approval evidence, the door caller and the door request.

A confirmation changes no threshold and moves no money: it records that the
owner accepted the thresholds as built, so the team packet can compile the
owner contract as the owner's. Nothing here starts compute or calls a provider.
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
    from .task_evaluation_owner_success_confirmation import (
        ACK,
        CHANNEL,
        _require,
        confirm_owner_success_contract,
        preview_owner_success_confirmation,
    )
    from .task_evaluation_scene_intent_contracts import ROOT_ENV

    intent_id = request["intent_id"]
    _require(bool(intake._identifier(intent_id)), "not_authorized")
    root = intake._root(Path(os.environ[ROOT_ENV]))
    owner = {"user_id": request["owner_user_id"], "organization_id": request["owner_organization_id"]}
    evidence = str(request["approval_evidence"]).strip()
    planned = preview_owner_success_confirmation(queue_root=root, intent_id=intent_id, owner=owner,
                                                 approved_by=request["approved_by"], approval_evidence=evidence)
    if not request["apply"]:
        return {"door_outcome": "planned", "mode": "preview", **planned}
    _require(request["apply_ack"] == planned["apply_ack"], "apply_ack_mismatch")
    reference = (f"owner_approval:approved_by={request['approved_by']};evidence={evidence};"
                 f"channel={CHANNEL};door_caller={door_caller};door_request_id={door_request_id}")
    result = confirm_owner_success_contract(
        queue_root=root, intent_id=intent_id, owner=owner, approved_by=request["approved_by"],
        approval_evidence=evidence, apply_ack=request["apply_ack"], authorization_reference=reference, ack=ACK)
    return {"door_outcome": result["status"], "mode": "applied", **result}


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
