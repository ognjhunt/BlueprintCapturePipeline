#!/usr/bin/env python3
"""Issue the one-attempt paid authority for a Content Agents run.

The allocator will not spend without this document, and it binds thirteen
values to each other: the bundle's digest and its receipt's digest, the config
preflight's receipt digest *and* the digest recorded inside it, the pinned
Content Agents source commit and tree, the container image, the spend cap, the
hourly rate, the TTL, and the instance allowlist. Get one wrong and the refusal
arrives at the paid boundary rather than here.

This derives every one of them from the receipts the run will actually use, and
validates the result with the allocator's own function before writing, so a
document that would be refused at the paid boundary never exists on disk.

What it cannot derive is the authorization itself: who is approving one paid
attempt, and what they are approving it against. Both are required on the
command line and recorded verbatim.

Reads retained bytes only; performs no provider mutation and rents nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from blueprint_pipeline.adp_content_agents_vast import (
    DEFAULT_IMAGE,
    PAID_ATTEMPT_AUTHORITY_SCHEMA,
    SOURCE_COMMIT,
    SOURCE_TREE,
    validate_content_agents_paid_attempt_authority,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.host_resident_launch_inputs import (
    HostResidentInputError,
    resolve_host_resident_bundle_receipt,
)


class ContentAgentsAuthorityError(ValueError):
    """The attempt authority cannot be issued against these receipts."""


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ContentAgentsAuthorityError(f"authority_input_not_object:{path.name}")
    return dict(value)


def issue_content_agents_paid_attempt_authority(
    *,
    bundle_receipt_path: str | Path,
    config_preflight_path: str | Path,
    authorized_by: str,
    authority_reference: str,
    max_hourly_rate_usd: float,
    hard_cap_usd: float,
    hard_ttl_seconds: int,
    authorized_on: str | None = None,
) -> dict[str, Any]:
    """Derive the attempt authority, then refuse to emit an invalid one."""

    if not authorized_by.strip():
        raise ContentAgentsAuthorityError("authority_authorized_by_required")
    if not authority_reference.strip():
        raise ContentAgentsAuthorityError("authority_reference_required")

    receipt_file = Path(bundle_receipt_path).expanduser().resolve()
    preflight_file = Path(config_preflight_path).expanduser().resolve()
    try:
        resolution = resolve_host_resident_bundle_receipt(receipt_file)
    except HostResidentInputError as exc:
        raise ContentAgentsAuthorityError(str(exc)) from exc
    if resolution["blockers"]:
        # Authorizing spend against bytes this host cannot resolve authorizes
        # spend against nothing in particular.
        raise ContentAgentsAuthorityError(
            "authority_bundle_not_host_resident:" + ",".join(resolution["blockers"])
        )
    bundle = resolution["receipt"]
    if not preflight_file.is_file() or preflight_file.is_symlink():
        raise ContentAgentsAuthorityError("authority_config_preflight_missing")
    preflight = _read(preflight_file)

    # The allocator recomputes both of these from the files it is handed, so a
    # preflight bound to a different receipt is refused there. Deriving them
    # here means that refusal happens at authoring time instead.
    bundle_receipt_sha256 = _sha256(receipt_file)
    preflight_receipt_sha256 = _sha256(preflight_file)
    if preflight.get("bundle_receipt_sha256") != bundle_receipt_sha256:
        raise ContentAgentsAuthorityError(
            "authority_config_preflight_binds_a_different_bundle_receipt"
        )

    authority: dict[str, Any] = {
        "schema_version": PAID_ATTEMPT_AUTHORITY_SCHEMA,
        "authority_kind": "explicit_user_direction_in_current_goal",
        "authority_reference": authority_reference.strip(),
        "authorized_by": authorized_by.strip(),
        "authorized_on": authorized_on or datetime.now(timezone.utc).date().isoformat(),
        "purpose": "nvidia_content_agents_advisory_enrichment",
        "provider": "vast",
        "paid_compute_authorized": True,
        "bundle_sha256": bundle.get("bundle_sha256"),
        "bundle_receipt_sha256": bundle_receipt_sha256,
        "config_preflight_receipt_sha256": preflight_receipt_sha256,
        "config_preflight_receipt_digest": preflight.get("receipt_digest"),
        "content_agents_source_commit": SOURCE_COMMIT,
        "content_agents_source_tree": SOURCE_TREE,
        "container_image": DEFAULT_IMAGE,
        "maximum_paid_attempts": 1,
        "maximum_automatic_retries": 0,
        "automatic_paid_retry_authorized": False,
        "zero_retry": True,
        "hard_attempt_spend_cap_usd": hard_cap_usd,
        "maximum_hourly_rate_usd": max_hourly_rate_usd,
        "maximum_single_resource_ttl_seconds": hard_ttl_seconds,
        # This lane produces advisory enrichment. Saying so in the authority
        # keeps a passing run from being read as a SimReady qualification.
        "agent_output_is_simready_authority": False,
        "native_simulator_import_qualified": False,
        # Empty: no pre-existing instance is admitted by this authority. A
        # concurrent operator's instance is recognised by its own label at the
        # prelaunch guard, which is where that question belongs.
        "active_instance_allowlist": [],
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )

    validate_content_agents_paid_attempt_authority(
        authority,
        prepared_bundle=bundle,
        bundle_receipt_sha256=bundle_receipt_sha256,
        config_preflight=preflight,
        config_preflight_receipt_sha256=preflight_receipt_sha256,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_cap_usd=hard_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        allowed_active_instance_ids=(),
    )
    return authority


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-receipt", required=True)
    parser.add_argument("--config-preflight", required=True)
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
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    try:
        authority = issue_content_agents_paid_attempt_authority(
            bundle_receipt_path=args.bundle_receipt,
            config_preflight_path=args.config_preflight,
            authorized_by=args.authorized_by,
            authority_reference=args.authority_reference,
            max_hourly_rate_usd=args.max_hourly_rate_usd,
            hard_cap_usd=args.hard_cap_usd,
            hard_ttl_seconds=args.hard_ttl_seconds,
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
