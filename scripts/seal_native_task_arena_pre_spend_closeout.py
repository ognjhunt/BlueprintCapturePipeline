#!/usr/bin/env python3
"""Seal a zero-cost Arena attempt blocked by the shared pre-spend gate."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from blueprint_pipeline.native_task_arena_paid_authority import (
    materialize_native_task_arena_pre_spend_closeout,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--allocator-result", required=True)
    parser.add_argument("--authority-consumption", required=True)
    parser.add_argument("--api-provider-zero", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        value = materialize_native_task_arena_pre_spend_closeout(
            authority_path=args.authority,
            allocator_result_path=args.allocator_result,
            authority_consumption_path=args.authority_consumption,
            api_provider_zero_path=args.api_provider_zero,
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
