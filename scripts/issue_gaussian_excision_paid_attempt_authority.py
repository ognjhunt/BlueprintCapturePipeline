#!/usr/bin/env python3
"""Issue the one-attempt paid authority for the Gaussian ownership audit.

The allocator will not spend without this document, and every value in it binds
to something else: the bundle's digest, its freeze digest, the parent execution
authority's digest, the corrective commit, the spend cap, the TTL, and the
instance allowlist. Get one wrong and the refusal arrives at the paid boundary
rather than here, after a provider has been handed over.

So every one is derived from the receipt the run will actually use, and the
result is checked with the allocator's own validator before it is written.

This lane is unusual in one way worth knowing: attempts are ordinal, and any
attempt after the first must name the sealed receipt of the one before it. That
is a deliberate anti-retry design -- a second paid attempt has to be authorized
against the evidence of the first, not issued blind. So `--prior-result` is
required from ordinal 2 onward, and the ordinal is derived from how many prior
receipts were supplied rather than typed by hand.

What cannot be derived is the authorization itself: who is approving one paid
attempt, and what they are approving it against. Both are required on the
command line and recorded verbatim.

Reads retained bytes only; performs no provider mutation and rents nothing.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from blueprint_pipeline.adp_gaussian_excision_vast import (
    PAID_ATTEMPT_AUTHORITY_SCHEMA,
    validate_gaussian_excision_paid_attempt_authority,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.host_resident_launch_inputs import (
    HostResidentInputError,
    resolve_host_resident_bundle_receipt,
)
from blueprint_pipeline.paid_attempt_authority import bind_lane_prior_spend


class GaussianExcisionAuthorityError(ValueError):
    """The attempt authority cannot be issued against this receipt."""


def _next_paid_attempt_ordinal(prior: Mapping[str, Any]) -> int:
    """The chain extends one past the sealed predecessor's own ordinal.

    Legacy receipts predate the ordinal field; the validator pins the
    successor of such a receipt to ordinal 2, so that is what an absent
    field derives to.
    """

    ordinal = prior.get("paid_attempt_ordinal")
    if ordinal is None:
        return 2
    if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 1:
        raise GaussianExcisionAuthorityError("previous_attempt_ordinal_invalid")
    return ordinal + 1


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise GaussianExcisionAuthorityError(f"authority_input_not_object:{path.name}")
    return dict(value)


def issue_gaussian_excision_paid_attempt_authority(
    *,
    bundle_receipt_path: str | Path,
    authorized_by: str,
    authority_reference: str,
    prior_attempt_receipt_path: str | Path | None = None,
    prior_spend_result_paths: Sequence[str | Path] = (),
    prior_spend_reconciliation_path: str | Path | None = None,
    authorized_on: str | None = None,
) -> dict[str, Any]:
    """Derive the attempt authority, then refuse to emit an invalid one."""

    if not authorized_by.strip():
        raise GaussianExcisionAuthorityError("authority_authorized_by_required")
    if not authority_reference.strip():
        raise GaussianExcisionAuthorityError("authority_reference_required")

    receipt_file = Path(bundle_receipt_path).expanduser().resolve()
    try:
        resolution = resolve_host_resident_bundle_receipt(receipt_file)
    except HostResidentInputError as exc:
        raise GaussianExcisionAuthorityError(str(exc)) from exc
    if resolution["blockers"]:
        raise GaussianExcisionAuthorityError(
            "authority_bundle_not_host_resident:" + ",".join(resolution["blockers"])
        )
    bundle = resolution["receipt"]

    prior: dict[str, Any] | None = None
    if prior_attempt_receipt_path is not None:
        prior_file = Path(prior_attempt_receipt_path).expanduser().resolve()
        if not prior_file.is_file() or prior_file.is_symlink():
            raise GaussianExcisionAuthorityError("authority_prior_attempt_receipt_missing")
        prior = _read(prior_file)
    spend_results = tuple(prior_spend_result_paths)
    if not spend_results and prior is not None:
        spend_results = (prior_file,)
    prior_spend = bind_lane_prior_spend(
        prior_result_paths=spend_results,
        reconciliation_path=prior_spend_reconciliation_path,
        lane="gaussian_excision",
    )

    # The spend cap and TTL belong to the bundle, not to this issuer: the
    # validator compares the authority against the receipt's own values, so
    # naming them here would only be a chance to disagree.
    authority: dict[str, Any] = {
        "schema_version": PAID_ATTEMPT_AUTHORITY_SCHEMA,
        "authority_kind": "explicit_user_direction_in_current_goal",
        "authority_reference": authority_reference.strip(),
        "authorized_by": authorized_by.strip(),
        "authorized_on": authorized_on or datetime.now(timezone.utc).date().isoformat(),
        # Bind the grant to the exact purpose declared by the prepared bundle.
        # Current removal freezes use the repair-supported segment-contribution
        # sweep, while legacy ownership-audit bundles omit this field.
        "purpose": str(
            bundle.get("execution_purpose")
            or "released_code_gaussian_ownership_audit"
        ),
        "provider": "vast",
        "paid_compute_authorized": True,
        "parent_execution_authority_digest": bundle.get("execution_authority_digest"),
        "freeze_digest": bundle.get("freeze_digest"),
        "bundle_sha256": bundle.get("bundle_sha256"),
        "corrective_blueprint_commit": bundle.get("blueprint_commit"),
        "paid_attempt_ordinal": 1 if prior is None else _next_paid_attempt_ordinal(prior),
        "maximum_paid_attempts": 1,
        "maximum_automatic_retries": 0,
        "automatic_paid_retry_authorized": False,
        "hard_attempt_spend_cap_usd": bundle.get("hard_cap_usd"),
        "maximum_single_resource_ttl_seconds": bundle.get("hard_ttl_seconds"),
        # Empty: no pre-existing instance is admitted by this authority. A
        # concurrent operator's instance is recognised by its own label at the
        # prelaunch guard, which is where that question belongs.
        "active_instance_allowlist": [],
    }
    if prior is not None:
        authority["previous_attempt_receipt_digest"] = prior.get("receipt_digest")
    authority["prior_terminal_attempts"] = prior_spend["prior_terminal_attempts"]
    authority["prior_spend_reconciliation"] = prior_spend["reconciliation"]
    authority["prior_actual_provider_spend_usd"] = prior_spend["actual_total_usd"]
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )

    validate_gaussian_excision_paid_attempt_authority(
        authority,
        prepared_bundle=bundle,
        previous_attempt_receipt=prior,
        allowed_active_instance_ids=(),
    )
    return authority


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-receipt", required=True)
    parser.add_argument(
        "--authorized-by", required=True, help="Who is approving one paid attempt."
    )
    parser.add_argument(
        "--authority-reference",
        required=True,
        help="What they approved: the instruction or decision this attempt rests on.",
    )
    parser.add_argument(
        "--prior-result",
        help=(
            "The sealed attempt receipt of the previous paid attempt. Required "
            "from the second attempt onward; the ordinal is derived from it."
        ),
    )
    parser.add_argument(
        "--prior-spend-result",
        action="append",
        default=[],
        help=(
            "A same-lane terminal result bound by --prior-spend-reconciliation. "
            "Repeat for every reconciled prior attempt. This is spend history, "
            "not a retry predecessor and does not advance the bundle-local ordinal."
        ),
    )
    parser.add_argument("--authorized-on")
    parser.add_argument("--prior-spend-reconciliation")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    try:
        authority = issue_gaussian_excision_paid_attempt_authority(
            bundle_receipt_path=args.bundle_receipt,
            authorized_by=args.authorized_by,
            authority_reference=args.authority_reference,
            prior_attempt_receipt_path=args.prior_result,
            prior_spend_result_paths=args.prior_spend_result,
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
                "paid_attempt_ordinal": authority["paid_attempt_ordinal"],
                "hard_attempt_spend_cap_usd": authority["hard_attempt_spend_cap_usd"],
                "provider_mutation_performed": False,
            },
            indent=1,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
