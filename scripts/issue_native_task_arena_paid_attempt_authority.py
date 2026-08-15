#!/usr/bin/env python3
"""Seal one zero-retry native Task Arena paid-attempt authority.

This command only materializes digest-bound authority evidence.  It does not
allocate a provider resource, upload a bundle, or consume the authority.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from blueprint_pipeline.native_task_arena_paid_authority import (
    materialize_native_task_arena_paid_attempt_authority,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-receipt", required=True)
    parser.add_argument("--prior-authority", required=True)
    parser.add_argument("--prior-result", required=True)
    parser.add_argument("--prior-provider-zero", required=True)
    parser.add_argument("--prior-spend-reconciliation", required=True)
    parser.add_argument("--authority-reference", required=True)
    parser.add_argument("--authorized-by", required=True)
    parser.add_argument("--authorized-on", required=True)
    parser.add_argument("--blueprint-commit", required=True)
    parser.add_argument("--max-hourly-rate-usd", required=True, type=float)
    parser.add_argument("--hard-cap-usd", required=True, type=float)
    parser.add_argument("--hard-ttl-seconds", required=True, type=int)
    parser.add_argument("--allow-active-instance", action="append", default=[], type=int)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        authority = materialize_native_task_arena_paid_attempt_authority(
            bundle_receipt_path=args.bundle_receipt,
            prior_authority_path=args.prior_authority,
            prior_result_path=args.prior_result,
            prior_provider_zero_path=args.prior_provider_zero,
            prior_spend_reconciliation_path=args.prior_spend_reconciliation,
            authorization_reference=args.authority_reference,
            authorized_by=args.authorized_by,
            authorized_on=args.authorized_on,
            blueprint_commit=args.blueprint_commit,
            max_hourly_rate_usd=args.max_hourly_rate_usd,
            hard_cap_usd=args.hard_cap_usd,
            hard_ttl_seconds=args.hard_ttl_seconds,
            output_path=args.output,
            allowed_active_instance_ids=tuple(args.allow_active_instance),
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": [f"{type(exc).__name__}:{exc}"],
                    "provider_mutation_performed": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": "issued",
                "authorization_digest": authority.get("authorization_digest"),
                "output": str(Path(args.output).expanduser().resolve()),
                "provider_mutation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
