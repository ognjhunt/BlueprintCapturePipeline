#!/usr/bin/env python3
"""Seal one exact two-member native Task Arena policy campaign."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from blueprint_pipeline.native_task_arena_policy_campaign import (
    materialize_native_task_arena_policy_campaign,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--blueprint-commit", required=True)
    parser.add_argument("--pi05-bundle-receipt", required=True)
    parser.add_argument("--groot-bundle-receipt", required=True)
    parser.add_argument("--prior-authority", required=True)
    parser.add_argument("--prior-result", required=True)
    parser.add_argument("--prior-provider-zero", required=True)
    parser.add_argument("--prior-spend-reconciliation", required=True)
    parser.add_argument("--allow-controls-active-instance", action="append", type=int, default=[])
    for prefix in ("pi05", "groot"):
        parser.add_argument(f"--{prefix}-launch-id", required=True)
        parser.add_argument(f"--{prefix}-resource-name", required=True)
        parser.add_argument(f"--{prefix}-max-hourly-rate-usd", required=True, type=float)
        parser.add_argument(f"--{prefix}-hard-cap-usd", required=True, type=float)
        parser.add_argument(f"--{prefix}-hard-ttl-seconds", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        campaign = materialize_native_task_arena_policy_campaign(
            campaign_id=args.campaign_id,
            blueprint_commit=args.blueprint_commit,
            pi05_bundle_receipt_path=args.pi05_bundle_receipt,
            groot_bundle_receipt_path=args.groot_bundle_receipt,
            prior_authority_path=args.prior_authority,
            prior_result_path=args.prior_result,
            prior_provider_zero_path=args.prior_provider_zero,
            prior_spend_reconciliation_path=args.prior_spend_reconciliation,
            controls_allowed_active_instance_ids=args.allow_controls_active_instance,
            pi05_launch_id=args.pi05_launch_id,
            pi05_resource_name=args.pi05_resource_name,
            pi05_max_hourly_rate_usd=args.pi05_max_hourly_rate_usd,
            pi05_hard_cap_usd=args.pi05_hard_cap_usd,
            pi05_hard_ttl_seconds=args.pi05_hard_ttl_seconds,
            groot_launch_id=args.groot_launch_id,
            groot_resource_name=args.groot_resource_name,
            groot_max_hourly_rate_usd=args.groot_max_hourly_rate_usd,
            groot_hard_cap_usd=args.groot_hard_cap_usd,
            groot_hard_ttl_seconds=args.groot_hard_ttl_seconds,
            output_path=args.output,
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
                "status": "sealed",
                "campaign_id": campaign["campaign_id"],
                "campaign_digest": campaign["campaign_digest"],
                "projected_aggregate_goal_spend_usd": campaign[
                    "projected_aggregate_goal_spend_usd"
                ],
                "output": args.output,
                "provider_mutation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
