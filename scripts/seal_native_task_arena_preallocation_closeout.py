#!/usr/bin/env python3
"""Seal a zero-cost Arena launch that stopped before provider allocation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from blueprint_pipeline.native_task_arena_paid_authority import (
    materialize_native_task_arena_preallocation_closeout,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--allocator-result", required=True)
    parser.add_argument("--watchdog-handoff", required=True)
    parser.add_argument("--object-store-cleanup", required=True)
    parser.add_argument("--api-provider-zero", required=True)
    parser.add_argument(
        "--sibling-preallocation-closeout", action="append", default=[]
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        value = materialize_native_task_arena_preallocation_closeout(
            authority_path=args.authority,
            allocator_result_path=args.allocator_result,
            watchdog_handoff_path=args.watchdog_handoff,
            object_store_cleanup_path=args.object_store_cleanup,
            api_provider_zero_path=args.api_provider_zero,
            sibling_preallocation_closeout_paths=(
                args.sibling_preallocation_closeout
            ),
            output_dir=args.output_dir,
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
    print(json.dumps({"status": "completed", **value}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
