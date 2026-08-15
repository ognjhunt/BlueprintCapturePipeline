#!/usr/bin/env python3
"""Issue the one-attempt paid authority for the ADP-009B SimReady import probe.

The allocator will not spend without this document, and it binds every value in
it to something else: the bundle's digest and its receipt's digest, the probe
spec's digest, the container image, the spend cap, the hourly rate, the TTL, and
the instance allowlist. Get one wrong and the refusal arrives at the paid
boundary rather than here, after a provider has been handed over.

So every one of them is derived from the receipt the run will actually use, and
the result is checked with the allocator's own validator before it is written --
a document that would be refused at the paid boundary never exists on disk.

What cannot be derived is the authorization itself: who is approving one paid
attempt, and what they are approving it against. Both are required on the
command line and recorded verbatim.

The claim boundary is recorded too, and it is narrow: this probe establishes
that a native simulator can import the asset. It does not establish that
anything physically succeeded, and it queries no candidate policy.

Reads retained bytes only; performs no provider mutation and rents nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.host_resident_launch_inputs import (
    HostResidentInputError,
    resolve_host_resident_bundle_receipt,
)
from blueprint_pipeline.paid_attempt_authority import bind_lane_prior_spend
from blueprint_pipeline.public_scene_simready_isaac_vast import (
    DEFAULT_IMAGE,
    PAID_ATTEMPT_AUTHORITY_SCHEMA,
    validate_simready_isaac_paid_attempt_authority,
)


class SimReadyIsaacAuthorityError(ValueError):
    """The attempt authority cannot be issued against this receipt."""


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def issue_simready_isaac_paid_attempt_authority(
    *,
    bundle_receipt_path: str | Path,
    authorized_by: str,
    authority_reference: str,
    max_hourly_rate_usd: float,
    hard_cap_usd: float,
    hard_ttl_seconds: int,
    prior_result_paths: Sequence[str | Path] = (),
    prior_spend_reconciliation_path: str | Path | None = None,
    authorized_on: str | None = None,
) -> dict[str, Any]:
    """Derive the attempt authority, then refuse to emit an invalid one."""

    if not authorized_by.strip():
        raise SimReadyIsaacAuthorityError("authority_authorized_by_required")
    if not authority_reference.strip():
        raise SimReadyIsaacAuthorityError("authority_reference_required")

    receipt_file = Path(bundle_receipt_path).expanduser().resolve()
    try:
        resolution = resolve_host_resident_bundle_receipt(receipt_file)
    except HostResidentInputError as exc:
        raise SimReadyIsaacAuthorityError(str(exc)) from exc
    if resolution["blockers"]:
        # Authorizing spend against bytes this host cannot resolve authorizes
        # spend against nothing in particular.
        raise SimReadyIsaacAuthorityError(
            "authority_bundle_not_host_resident:" + ",".join(resolution["blockers"])
        )
    bundle = resolution["receipt"]
    prior_spend = bind_lane_prior_spend(
        prior_result_paths=prior_result_paths,
        reconciliation_path=prior_spend_reconciliation_path,
        lane="simready_isaac",
    )

    authority: dict[str, Any] = {
        "schema_version": PAID_ATTEMPT_AUTHORITY_SCHEMA,
        "authority_kind": "explicit_user_direction_in_current_goal",
        "authority_reference": authority_reference.strip(),
        "authorized_by": authorized_by.strip(),
        "authorized_on": authorized_on or datetime.now(timezone.utc).date().isoformat(),
        "purpose": "simready_native_import_probe",
        "provider": "vast",
        "paid_compute_authorized": True,
        "bundle_sha256": bundle.get("bundle_sha256"),
        "bundle_receipt_sha256": _sha256(receipt_file),
        "probe_spec_sha256": bundle.get("probe_spec_sha256"),
        "container_image": DEFAULT_IMAGE,
        "maximum_paid_attempts": 1,
        "maximum_automatic_retries": 0,
        "automatic_paid_retry_authorized": False,
        "zero_retry": True,
        "hard_attempt_spend_cap_usd": hard_cap_usd,
        "maximum_hourly_rate_usd": max_hourly_rate_usd,
        "maximum_single_resource_ttl_seconds": hard_ttl_seconds,
        # The claim boundary, stated in the authority so a passing run cannot
        # later be read as more than an import.
        "native_simulator_import_probe_only": True,
        "physical_success_established": False,
        "candidate_policy_queried": False,
        # Empty: no pre-existing instance is admitted by this authority. A
        # concurrent operator's instance is recognised by its own label at the
        # prelaunch guard, which is where that question belongs.
        "active_instance_allowlist": [],
        "prior_terminal_attempts": prior_spend["prior_terminal_attempts"],
        "prior_spend_reconciliation": prior_spend["reconciliation"],
        "prior_actual_provider_spend_usd": prior_spend["actual_total_usd"],
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )

    validate_simready_isaac_paid_attempt_authority(
        authority,
        prepared_bundle=bundle,
        bundle_receipt_sha256=authority["bundle_receipt_sha256"],
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_cap_usd=hard_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
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
    parser.add_argument("--max-hourly-rate-usd", type=float, required=True)
    parser.add_argument("--hard-cap-usd", type=float, required=True)
    parser.add_argument("--hard-ttl-seconds", type=int, required=True)
    parser.add_argument("--authorized-on")
    parser.add_argument("--prior-result", action="append", default=[])
    parser.add_argument("--prior-spend-reconciliation")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    try:
        authority = issue_simready_isaac_paid_attempt_authority(
            bundle_receipt_path=args.bundle_receipt,
            authorized_by=args.authorized_by,
            authority_reference=args.authority_reference,
            max_hourly_rate_usd=args.max_hourly_rate_usd,
            hard_cap_usd=args.hard_cap_usd,
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
