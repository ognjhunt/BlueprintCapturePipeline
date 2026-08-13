#!/usr/bin/env python3
"""Issue a one-attempt paid authority for either link of the appearance chain.

The appearance path is a chained, spend-accumulating campaign rather than a set
of independent lanes:

    prior Aura terminal  -->  ArtiFixer3D  -->  paired-target native import

Each link's authority validates its predecessor's terminal evidence and carries
the campaign's running spend forward against a shared cap, so an authority
cannot be minted out of order and cannot be minted twice.

Both `materialize_*_paid_attempt_authority` functions already existed and
neither could be called from a command line. That is the same defect #512 and
#520 fixed for lanes and bundle modules, in a third scope: modules that mint an
authority rather than seal a bundle. Without an entry point the campaign was
authorizable only from a Python session, which is not a production path.

One script with two subcommands, because the two links differ only in which
predecessor evidence they demand -- and a per-link copy would be a per-link
opportunity to drop one of those demands.

What cannot be derived is the authorization itself: who is approving one paid
attempt, and what they are approving it against. Both are required and recorded
verbatim.

Reads retained bytes only; performs no provider mutation and rents nothing.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from blueprint_pipeline.paired_target_native_import_vast import (
    materialize_paired_target_native_import_paid_attempt_authority,
)
from blueprint_pipeline.public_scene_artifixer3d_vast import (
    materialize_artifixer3d_paid_attempt_authority,
)


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _require(value: str, code: str) -> str:
    if not str(value or "").strip():
        raise ValueError(code)
    return str(value).strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="link", required=True)

    art = sub.add_parser(
        "artifixer3d", help="Head of the chain; anchors on a prior Aura terminal."
    )
    art.add_argument("--bundle-receipt", required=True)
    art.add_argument(
        "--prior-aura-authority",
        required=True,
        help=(
            "A retired lane's historical authority is still this campaign's "
            "spend anchor. Retiring AuraFusion360 did not delete its receipts."
        ),
    )
    art.add_argument("--prior-terminal-result", required=True)
    art.add_argument(
        "--prior-artifixer-authority",
        help="The previous ArtiFixer3D attempt, from the second attempt onward.",
    )
    art.add_argument("--max-hourly-rate-usd", type=float, default=1.0)
    art.add_argument("--hard-cap-usd", type=float, default=10.0)
    art.add_argument("--hard-ttl-seconds", type=int, default=10_800)

    gate = sub.add_parser(
        "paired-target", help="Import gate; anchors on a completed ArtiFixer3D run."
    )
    gate.add_argument("--bundle-receipt", required=True)
    gate.add_argument("--prior-artifixer-authority", required=True)
    gate.add_argument("--prior-artifixer-result", required=True)
    gate.add_argument("--prior-artifixer-cleanup", required=True)
    gate.add_argument("--prior-artifixer-provider-zero", required=True)
    gate.add_argument("--supplemental-prior-spend-reconciliation")
    gate.add_argument("--prior-native-preallocation-provider-zero")
    gate.add_argument("--max-hourly-rate-usd", type=float, default=1.0)
    gate.add_argument("--hard-cap-usd", type=float, default=2.0)
    gate.add_argument("--hard-ttl-seconds", type=int, default=7_200)

    for target in (art, gate):
        target.add_argument(
            "--authorized-by", required=True, help="Who is approving one paid attempt."
        )
        target.add_argument(
            "--authority-reference",
            required=True,
            help="What they approved: the instruction this attempt rests on.",
        )
        target.add_argument("--authorized-on")
        target.add_argument("--blueprint-commit", required=True)
        target.add_argument("--output", required=True)

    args = parser.parse_args(argv)

    try:
        authorized_by = _require(args.authorized_by, "authority_authorized_by_required")
        reference = _require(args.authority_reference, "authority_reference_required")
        shared: dict[str, Any] = {
            "bundle_receipt_path": args.bundle_receipt,
            "authorization_reference": reference,
            "authorized_by": authorized_by,
            "authorized_on": args.authorized_on or _today(),
            "blueprint_commit": args.blueprint_commit,
            "max_hourly_rate_usd": args.max_hourly_rate_usd,
            "hard_cap_usd": args.hard_cap_usd,
            "hard_ttl_seconds": args.hard_ttl_seconds,
            "output_path": args.output,
        }
        if args.link == "artifixer3d":
            authority = materialize_artifixer3d_paid_attempt_authority(
                prior_aura_authority_path=args.prior_aura_authority,
                prior_terminal_result_path=args.prior_terminal_result,
                prior_artifixer_authority_path=args.prior_artifixer_authority,
                **shared,
            )
        else:
            authority = materialize_paired_target_native_import_paid_attempt_authority(
                prior_artifixer_authority_path=args.prior_artifixer_authority,
                prior_artifixer_result_path=args.prior_artifixer_result,
                prior_artifixer_cleanup_path=args.prior_artifixer_cleanup,
                prior_artifixer_provider_zero_path=args.prior_artifixer_provider_zero,
                supplemental_prior_spend_reconciliation_path=(
                    args.supplemental_prior_spend_reconciliation
                ),
                prior_native_preallocation_provider_zero_path=(
                    args.prior_native_preallocation_provider_zero
                ),
                **shared,
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": [f"{type(exc).__name__}:{exc}"],
                    "provider_mutation_performed": False,
                },
                indent=1,
                sort_keys=True,
            )
        )
        return 2

    summary = {
        "status": "issued",
        "link": args.link,
        "output": str(Path(args.output).expanduser().resolve()),
        "provider_mutation_performed": False,
    }
    if isinstance(authority, dict):
        for key in (
            "authorization_digest",
            "bundle_sha256",
            "hard_attempt_spend_cap_usd",
            "aggregate_goal_spend_before_attempt_usd",
            "aggregate_goal_spend_cap_usd",
        ):
            if key in authority:
                summary[key] = authority[key]
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
