#!/usr/bin/env python3
"""Seal one zero-allocation authority for a retained Arena controls attempt."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from blueprint_pipeline.native_task_arena_construction_bundle import (
    load_verified_native_task_arena_construction_bundle,
)
from blueprint_pipeline.native_task_arena_controls_bundle import (
    load_verified_native_task_arena_controls_bundle,
)
from blueprint_pipeline.native_task_arena_warm_authority import (
    materialize_native_task_arena_warm_attempt_authority,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-session", required=True)
    parser.add_argument("--bundle-receipt", required=True)
    parser.add_argument(
        "--execution-mode",
        choices=("construction_canary", "controls"),
        default="controls",
    )
    parser.add_argument("--blueprint-commit", required=True)
    parser.add_argument("--packet-receipt-digest", required=True)
    parser.add_argument("--runtime-source-packet-digest", required=True)
    parser.add_argument("--authority-reference", required=True)
    parser.add_argument("--authorized-by", required=True)
    parser.add_argument("--authorized-on", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        loader = (
            load_verified_native_task_arena_construction_bundle
            if args.execution_mode == "construction_canary"
            else load_verified_native_task_arena_controls_bundle
        )
        bundle = loader(
            args.bundle_receipt,
            expected_implementation_commit=args.blueprint_commit,
            expected_packet_receipt_digest=args.packet_receipt_digest,
            expected_runtime_source_packet_digest=(
                args.runtime_source_packet_digest
            ),
        )
        issued = materialize_native_task_arena_warm_attempt_authority(
            warm_session_path=args.warm_session,
            bundle_receipt_path=args.bundle_receipt,
            prepared_bundle=bundle,
            authorization_reference=args.authority_reference,
            authorized_by=args.authorized_by,
            authorized_on=args.authorized_on,
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
                "status": "issued",
                "authorization_digest": issued["authorization_digest"],
                "maximum_provider_allocations": 0,
                "output": str(Path(args.output).expanduser().resolve()),
                "provider_mutation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
