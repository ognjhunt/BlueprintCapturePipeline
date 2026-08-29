#!/usr/bin/env python3
"""Seal one retained Arena session after its hard-TTL watchdog proves absence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Sequence

from blueprint_pipeline.native_task_arena_warm_closeout import (
    materialize_expired_warm_closeout,
    materialize_failed_watchdog_recovery_closeout,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--retained-result", required=True)
    parser.add_argument("--provider-zero-guard")
    parser.add_argument("--watchdog-supersession")
    parser.add_argument("--successor-watchdog")
    parser.add_argument("--termination-session-excerpt")
    parser.add_argument("--exact-absence-observation", action="append", default=[])
    parser.add_argument("--recovered-provider-zero")
    parser.add_argument("--official-billing-response")
    parser.add_argument("--provider-billing-source-receipt")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        recovery_requested = any(
            (
                args.termination_session_excerpt,
                args.exact_absence_observation,
                args.recovered_provider_zero,
                args.official_billing_response,
                args.provider_billing_source_receipt,
            )
        )
        if recovery_requested:
            if args.provider_zero_guard:
                raise ValueError("native_task_arena_warm_closeout_mode_ambiguous")
            receipt = materialize_failed_watchdog_recovery_closeout(
                authority_path=args.authority,
                retained_result_path=args.retained_result,
                termination_session_excerpt_path=args.termination_session_excerpt,
                exact_absence_observation_paths=args.exact_absence_observation,
                provider_zero_path=args.recovered_provider_zero,
                official_billing_response_path=args.official_billing_response,
                provider_billing_source_receipt_path=(
                    args.provider_billing_source_receipt
                ),
                output_dir=args.output_dir,
            )
        else:
            if not args.provider_zero_guard:
                raise ValueError("native_task_arena_warm_provider_zero_guard_missing")
            receipt = materialize_expired_warm_closeout(
                authority_path=args.authority,
                retained_result_path=args.retained_result,
                provider_zero_guard_path=args.provider_zero_guard,
                output_dir=args.output_dir,
                watchdog_supersession_path=args.watchdog_supersession,
                successor_watchdog_path=args.successor_watchdog,
            )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "blocked", "blockers": [f"{type(exc).__name__}:{exc}"], "provider_mutation_performed": False}, sort_keys=True))
        return 2
    print(json.dumps({"status": "completed", "receipt_digest": receipt["receipt_digest"], "output": str(Path(args.output_dir).expanduser().resolve()), "provider_mutation_performed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
