#!/usr/bin/env python3
"""Materialize semantic-teacher closeout receipts from retained local bytes.

This is an offline, downstream-only evidence command, never a website paid
route. It never allocates, uploads, performs inference, or looks up a provider
token.
"""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from blueprint_pipeline.semantic_teacher_image_edit_paid_lane import (
    materialize_semantic_teacher_image_edit_result,
    materialize_semantic_teacher_no_allocation_closeout,
    materialize_semantic_teacher_provider_zero_receipt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    no_allocation = commands.add_parser("no-allocation")
    no_allocation.add_argument("--dry-run", required=True)
    no_allocation.add_argument("--watchdog-closeout", required=True)
    no_allocation.add_argument("--reason", required=True)
    no_allocation.add_argument("--output", required=True)

    provider_zero = commands.add_parser("provider-zero")
    for name in (
        "authority",
        "bundle-receipt",
        "terminal-result",
        "billing-receipt",
        "scoped-inventory",
        "global-inventory",
        "object-store-cleanup",
        "independent-watchdog",
        "secret-redaction",
        "stdout-log",
        "stderr-log",
        "output",
    ):
        provider_zero.add_argument(f"--{name}", required=True)

    result_import = commands.add_parser("result-import")
    for name in (
        "runtime-output-root",
        "runtime-request",
        "bundle-receipt",
        "authority",
        "billing-receipt",
        "scoped-inventory",
        "global-inventory",
        "object-store-cleanup",
        "watchdog-receipt",
        "secret-redaction",
        "provider-zero",
        "output",
    ):
        result_import.add_argument(f"--{name}", required=True)
    result_import.add_argument("--expected-task-count", required=True, type=int)
    result_import.add_argument("--expected-camera-count", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "no-allocation":
            result = materialize_semantic_teacher_no_allocation_closeout(
                dry_run_path=args.dry_run,
                watchdog_closeout_path=args.watchdog_closeout,
                reason=args.reason,
                output_path=args.output,
            )
            digest = result["closeout_digest"]
        elif args.command == "provider-zero":
            result = materialize_semantic_teacher_provider_zero_receipt(
                authority_path=args.authority,
                bundle_receipt_path=args.bundle_receipt,
                terminal_result_path=args.terminal_result,
                billing_receipt_path=args.billing_receipt,
                scoped_inventory_path=args.scoped_inventory,
                global_inventory_path=args.global_inventory,
                object_store_cleanup_path=args.object_store_cleanup,
                independent_watchdog_path=args.independent_watchdog,
                secret_redaction_path=args.secret_redaction,
                stdout_log_path=args.stdout_log,
                stderr_log_path=args.stderr_log,
                output_path=args.output,
            )
            digest = result["provider_zero_digest"]
        else:
            result = materialize_semantic_teacher_image_edit_result(
                runtime_output_root=args.runtime_output_root,
                runtime_request_path=args.runtime_request,
                bundle_receipt_path=args.bundle_receipt,
                authority_path=args.authority,
                billing_receipt_path=args.billing_receipt,
                scoped_inventory_path=args.scoped_inventory,
                global_inventory_path=args.global_inventory,
                object_store_cleanup_path=args.object_store_cleanup,
                watchdog_receipt_path=args.watchdog_receipt,
                secret_redaction_path=args.secret_redaction,
                provider_zero_path=args.provider_zero,
                expected_task_count=args.expected_task_count,
                expected_camera_count=args.expected_camera_count,
                output_path=args.output,
            )
            digest = result["result_import_digest"]
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": [str(exc)],
                    "provider_mutation_performed": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": "materialized",
                "digest": digest,
                "provider_mutation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess
    raise SystemExit(main())
