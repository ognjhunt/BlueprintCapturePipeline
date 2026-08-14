#!/usr/bin/env python3
"""Materialize one file-backed, single-use SAM 3.1 paid authority.

This command writes authority bytes only. It performs no provider mutation,
token lookup, upload, allocation, or inference.
"""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from blueprint_pipeline.sam31_paid_attempt_authority import (
    materialize_sam31_paid_attempt_authority,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--bundle-receipt", required=True)
    parser.add_argument("--authorization-reference", required=True)
    parser.add_argument("--authorized-by", required=True)
    parser.add_argument("--authorized-on", required=True)
    parser.add_argument("--blueprint-commit", required=True)
    parser.add_argument("--max-hourly-rate-usd", required=True, type=float)
    parser.add_argument("--hard-cap-usd", required=True, type=float)
    parser.add_argument("--hard-ttl-seconds", required=True, type=int)
    parser.add_argument("--aggregate-spend-before-usd", required=True, type=float)
    parser.add_argument("--aggregate-spend-cap-usd", required=True, type=float)
    parser.add_argument("--prior-spend-reconciliation")
    parser.add_argument(
        "--allowed-active-instance-id", type=int, action="append", default=[]
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = materialize_sam31_paid_attempt_authority(
        request_path=args.request,
        bundle_path=args.bundle,
        bundle_receipt_path=args.bundle_receipt,
        authorization_reference=args.authorization_reference,
        authorized_by=args.authorized_by,
        authorized_on=args.authorized_on,
        blueprint_commit=args.blueprint_commit,
        max_hourly_rate_usd=args.max_hourly_rate_usd,
        hard_cap_usd=args.hard_cap_usd,
        hard_ttl_seconds=args.hard_ttl_seconds,
        aggregate_goal_spend_before_attempt_usd=args.aggregate_spend_before_usd,
        aggregate_goal_spend_cap_usd=args.aggregate_spend_cap_usd,
        output_path=args.output,
        allowed_active_instance_ids=args.allowed_active_instance_id,
        prior_spend_reconciliation_path=args.prior_spend_reconciliation,
    )
    print(
        json.dumps(
            {
                "status": "materialized",
                "authorization_digest": result["authorization_digest"],
                "provider_mutation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - module CLI exercised separately
    raise SystemExit(main())
