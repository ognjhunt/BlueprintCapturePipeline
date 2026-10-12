"""Append-only cumulative budget grants; historical reservations are never refunded.

The authenticated intake issuer may record delegated owner authority. A grant
raises the spend/count ceilings, which also bound any single action; consent
and every other execution bound stay unchanged.
"""
from __future__ import annotations

import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .task_evaluation_scene_execution_budget_evidence import (
    ACK,
    ATTEMPT_GRANT_FIELD,
    DIRECTORY,
    HOST_OPERATOR_CHANNEL,
    LIMIT_FIELDS,
    MAX_EXTENDED_PAID_ATTEMPTS,
    SCHEMA,
    _chain,
    _limits,
    _require,
    _safe,
    effective_execution_budget,
    validate_attempt_execution_budget,
)


def extend_scene_execution_budget(*, queue_root: str | Path, intent_id: str, intent_digest: str,
        owner: Mapping[str, Any], authenticated_client: str, trusted_clients: set[str],
        max_total_spend_usd: float, max_paid_attempts: int, authorization_reference: str,
        ack: str, now: float | None = None, host_operator: bool = False) -> dict[str, Any]:
    """Append an explicit grant under the intake lock; never settle or rewrite holds.

    A host-operator grant is still bound to the intent's issuer, and records
    that it was issued on the host (``issuer_channel``) by ``operator_uid``.
    """
    from . import task_evaluation_scene_intake as intake
    moment = time.time() if now is None else now
    _require(ack == ACK and authenticated_client in trusted_clients and intake._identifier(intent_id)
        and intake._number(moment) and isinstance(authorization_reference, str)
        and bool(authorization_reference.strip()), 'not_authorized')
    limits = _limits({'max_total_spend_usd': max_total_spend_usd, 'max_paid_attempts': max_paid_attempts})
    root = Path(queue_root)
    _safe(root)
    root = intake._root(root)
    with intake._lock(root):
        directory = root / intent_id
        _safe(directory)
        # write_exclusive publishes a read-only file owned by the caller. A
        # root-run amendment in a service-owned scene would seal a valid grant
        # that the controller cannot read on its next pass. Require the issuer
        # to run as the scene service account before publishing anything.
        _require(os.geteuid() == directory.stat().st_uid, 'issuer_user_mismatch')
        _safe(directory / 'revoked.json')
        intent = intake._read(directory / 'intent.json', 'intent_digest')
        _require(intent['intent_digest'] == intent_digest and intent['request']['owner'] == dict(owner)
            and intent['authenticated_issuer'] == authenticated_client
            and not (directory / 'revoked.json').exists(), 'owner_or_revocation_mismatch')
        _require(intent['accepted_at_epoch'] <= moment < intake.effective_execution_expiry(directory, intent), 'authority_expired')
        current, records = _chain(directory, intent)
        if all(limits[k] <= current[k] for k in LIMIT_FIELDS):
            return {'status': 'execution_budget_already_covers_request', **current,
                    'intent_digest': intent_digest, 'provider_mutation_performed': False}
        _require(all(limits[k] >= current[k] for k in LIMIT_FIELDS), 'limits_not_monotonic')
        execution = intent['request']['execution']
        value = intake._seal({'schema_version': SCHEMA, 'scope': 'cumulative_budget_and_attempts_only',
            'intent_id': intent_id, 'intent_digest': intent_digest, 'owner': dict(owner),
            'authenticated_issuer': authenticated_client, 'authorization_reference': authorization_reference,
            'original_limits': _limits(execution), 'prior_limits': _limits(current), 'limits': limits,
            'unchanged_execution_bounds': {k: v for k, v in execution.items() if k not in LIMIT_FIELDS},
            'sequence': len(records) + 1, 'predecessor_digest': current['extension_digest'] or intent_digest,
            'issued_at_epoch': moment, 'provider_mutation_performed': False,
            'historical_reservations_released': False,
            **({'issuer_channel': HOST_OPERATOR_CHANNEL, 'operator_uid': os.geteuid()} if host_operator else {})},
            'extension_digest')
        _require(not records or moment >= max(r['issued_at_epoch'] for r in records), 'issued_time_invalid')
        path = directory / DIRECTORY / (value['extension_digest'].removeprefix('sha256:') + '.json')
        path.parent.mkdir(mode=0o750, exist_ok=True)
        intake.write_exclusive(path, value)
        effective_execution_budget(directory, intent)
    return {'status': 'execution_budget_extended', 'record_path': str(path), **value}



def main(argv: list[str] | None = None) -> int:
    """Host operator entry for an owner-approved grant; previews unless --apply.

    Run as the scene service account. Apply requires --ack equal to the preview's
    digest of the exact intent, owner and limits, plus who approved and the
    approval evidence; the grant records the host-operator channel explicitly.
    """
    import argparse
    import json

    from . import task_evaluation_scene_intake as intake
    from .task_evaluation_scene_intent_contracts import CLIENTS_ENV, ROOT_ENV

    parser = argparse.ArgumentParser(description=main.__doc__)
    for flag in ("--intent-id", "--owner-user-id", "--owner-organization-id", "--approved-by", "--approval-evidence"):
        parser.add_argument(flag, required=True)
    parser.add_argument("--max-total-spend-usd", type=float, required=True)
    parser.add_argument("--max-paid-attempts", type=int, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--ack", default="")
    args = parser.parse_args(argv)
    _require(bool(intake._identifier(args.intent_id)), 'not_authorized')
    root = intake._root(Path(os.environ[ROOT_ENV]))
    intent = intake._read(root / args.intent_id / "intent.json", "intent_digest")
    owner = {"user_id": args.owner_user_id, "organization_id": args.owner_organization_id}
    _require(intent["request"]["owner"] == owner and args.approved_by == args.owner_user_id
             and bool(args.approval_evidence.strip()), "owner_or_revocation_mismatch")
    preview = {"intent_id": args.intent_id, "intent_digest": intent["intent_digest"], "owner": owner,
               "issuer": intent["authenticated_issuer"],
               "current": effective_execution_budget(root / args.intent_id, intent),
               "requested": _limits({"max_total_spend_usd": args.max_total_spend_usd,
                                     "max_paid_attempts": args.max_paid_attempts})}
    apply_ack = intake.canonical_digest(preview)
    if not args.apply:
        print(json.dumps({"mode": "preview", **preview, "apply_ack": apply_ack}, sort_keys=True))
        return 0
    _require(args.ack == apply_ack, "not_authorized")
    trusted = {value.strip() for value in os.getenv(CLIENTS_ENV, "blueprint-webapp").split(",") if value.strip()}
    reference = (f"owner_approval:approved_by={args.approved_by};evidence={args.approval_evidence.strip()};"
                 f"channel={HOST_OPERATOR_CHANNEL};operator_uid={os.geteuid()}")
    result = extend_scene_execution_budget(
        queue_root=root, intent_id=args.intent_id, intent_digest=intent["intent_digest"], owner=owner,
        authenticated_client=intent["authenticated_issuer"], trusted_clients=trusted,
        max_total_spend_usd=args.max_total_spend_usd, max_paid_attempts=args.max_paid_attempts,
        authorization_reference=reference, ack=ACK, host_operator=True)
    print(json.dumps({"mode": "applied", **result}, sort_keys=True, default=str))
    return 0

# Preserve the original import surface while validation lives in pure readers.
__all__ = [
    'ACK',
    'ATTEMPT_GRANT_FIELD',
    'DIRECTORY',
    'HOST_OPERATOR_CHANNEL',
    'LIMIT_FIELDS',
    'MAX_EXTENDED_PAID_ATTEMPTS',
    'SCHEMA',
    '_chain',
    '_limits',
    '_require',
    '_safe',
    'effective_execution_budget',
    'extend_scene_execution_budget',
    'validate_attempt_execution_budget',
]


if __name__ == "__main__":
    raise SystemExit(main())
