#!/usr/bin/env python3
"""Issue the one-attempt paid authority for a retained-scene GPU render.

The allocator will not spend without this document, and it binds nine values to
each other: the bundle's exact digest and commit, its parent execution
authority's digest, the spend cap, the TTL, the hourly rate, the instance
allowlist, and the full list of prior terminal attempts with their costs. Get
one wrong and the refusal arrives at the paid boundary rather than here.

By design there is no automatic retry, so this is reissued for *every* attempt:
each re-run must carry its predecessors in ``prior_terminal_attempts``, each
with a readable result file, a matching digest, and a cost that counts against
the same aggregate cap. Reissuing that by hand once per loop iteration is how a
cap silently stops being a cap -- omit a prior attempt and its spend is
forgotten.

Everything here is derivable from the bundle receipt and the prior results, so
this derives it. What it cannot derive is the authorization itself: who is
approving one paid attempt. That is required on the command line and recorded.

Reads retained bytes only; performs no provider mutation and rents nothing.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from blueprint_pipeline.adp_retained_scene_render_vast import (
    PAID_ATTEMPT_AUTHORITY_SCHEMA,
    validate_retained_scene_render_paid_attempt_authority,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.host_resident_launch_inputs import (
    HostResidentInputError,
    resolve_host_resident_bundle_receipt,
)
from blueprint_pipeline.paid_attempt_authority import bind_lane_prior_spend


class AttemptAuthorityError(ValueError):
    """The attempt authority cannot be issued against these bytes."""


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise AttemptAuthorityError(f"attempt_authority_input_not_object:{path.name}")
    return dict(value)


def issue_paid_attempt_authority(
    *,
    bundle_receipt_path: str | Path,
    authorized_by: str,
    max_hourly_rate_usd: float,
    hard_ttl_seconds: int,
    prior_result_paths: Sequence[str | Path] = (),
    prior_spend_reconciliation_path: str | Path | None = None,
    authorized_on: str | None = None,
) -> dict[str, Any]:
    """Derive the attempt authority, then refuse to emit an invalid one."""

    if not authorized_by.strip():
        raise AttemptAuthorityError("attempt_authority_authorized_by_required")
    try:
        resolution = resolve_host_resident_bundle_receipt(bundle_receipt_path)
    except HostResidentInputError as exc:
        raise AttemptAuthorityError(str(exc)) from exc
    if resolution["blockers"]:
        # Authorizing spend against bytes this host cannot resolve would
        # authorize spend against nothing in particular.
        raise AttemptAuthorityError(
            "attempt_authority_bundle_not_host_resident:" + ",".join(resolution["blockers"])
        )
    receipt = resolution["receipt"]

    authority_record = receipt.get("execution_authority")
    if not isinstance(authority_record, Mapping):
        raise AttemptAuthorityError("attempt_authority_parent_record_missing")
    parent_path = Path(str(resolution["resolutions"]["execution_authority"]["path"]))
    paid = _read(parent_path).get("paid_compute")
    if not isinstance(paid, Mapping):
        raise AttemptAuthorityError("attempt_authority_parent_paid_compute_missing")
    # The allowlist is the parent authority's, not this script's: the two must
    # match exactly and they are edited in different places.
    allowlist = sorted({int(item) for item in paid.get("external_instance_allowlist") or []})

    try:
        prior_spend = bind_lane_prior_spend(
            prior_result_paths=prior_result_paths,
            reconciliation_path=prior_spend_reconciliation_path,
            lane="retained_scene_render",
        )
    except ValueError as exc:
        raise AttemptAuthorityError(str(exc)) from exc
    priors = prior_spend["prior_terminal_attempts"]
    authority: dict[str, Any] = {
        "schema_version": PAID_ATTEMPT_AUTHORITY_SCHEMA,
        "authority_kind": "explicit_user_direction_in_current_goal",
        "authorized_by": authorized_by.strip(),
        "authorized_on": authorized_on
        or datetime.now(timezone.utc).date().isoformat(),
        "purpose": "exact_retained_scene_gpu_render",
        "provider": "vast",
        "paid_compute_authorized": True,
        "parent_execution_authority_digest": authority_record.get("authority_digest"),
        "bundle_sha256": receipt.get("bundle_sha256"),
        "blueprint_commit": receipt.get("blueprint_commit"),
        "maximum_paid_attempts": 1,
        "maximum_automatic_retries": 0,
        "automatic_paid_retry_authorized": False,
        "hard_attempt_spend_cap_usd": receipt.get("hard_total_spend_cap_usd"),
        "maximum_single_resource_ttl_seconds": hard_ttl_seconds,
        "maximum_hourly_rate_usd": max_hourly_rate_usd,
        "external_active_instance_allowlist": allowlist,
        "prior_terminal_attempts": priors,
        "prior_spend_reconciliation": prior_spend["reconciliation"],
        "prior_actual_provider_spend_usd": prior_spend["actual_total_usd"],
    }
    if priors:
        authority["manual_reissue_after_prior_terminal_attempt"] = True
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )

    # Validate with the same function the allocator uses, so a document that
    # would be refused at the paid boundary is never written at all.
    validate_retained_scene_render_paid_attempt_authority(
        authority,
        prepared_bundle=receipt,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        allowed_active_instance_ids=allowlist,
    )
    return authority


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-receipt", required=True)
    parser.add_argument(
        "--authorized-by",
        required=True,
        help="Who is approving one paid attempt. Recorded in the authority.",
    )
    parser.add_argument(
        "--prior-spend-reconciliation",
        help=(
            "Lane-local adp_same_goal_spend_reconciliation.v1 that binds every "
            "--prior-result to official billing, teardown, and provider-zero."
        ),
    )
    parser.add_argument("--max-hourly-rate-usd", type=float, required=True)
    parser.add_argument("--hard-ttl-seconds", type=int, required=True)
    parser.add_argument(
        "--prior-result",
        action="append",
        default=[],
        help=(
            "A prior attempt's allocator result.json. Repeat for each. Omitting "
            "one forgets its spend against the aggregate cap."
        ),
    )
    parser.add_argument("--authorized-on", help="ISO date; defaults to today (UTC).")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    try:
        authority = issue_paid_attempt_authority(
            bundle_receipt_path=args.bundle_receipt,
            authorized_by=args.authorized_by,
            max_hourly_rate_usd=args.max_hourly_rate_usd,
            hard_ttl_seconds=args.hard_ttl_seconds,
            prior_result_paths=args.prior_result,
            prior_spend_reconciliation_path=args.prior_spend_reconciliation,
            authorized_on=args.authorized_on,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": PAID_ATTEMPT_AUTHORITY_SCHEMA,
                    "status": "blocked",
                    "blockers": [str(exc)],
                    "provider_mutation_performed": False,
                },
                indent=1,
                sort_keys=True,
            )
        )
        return 2

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(authority, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "issued",
                "output": str(output),
                "authorization_digest": authority["authorization_digest"],
                "bundle_sha256": authority["bundle_sha256"],
                "blueprint_commit": authority["blueprint_commit"],
                "prior_terminal_attempts": len(authority["prior_terminal_attempts"]),
                "external_active_instance_allowlist": authority[
                    "external_active_instance_allowlist"
                ],
                "provider_mutation_performed": False,
            },
            indent=1,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
