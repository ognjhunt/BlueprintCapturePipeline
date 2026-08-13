#!/usr/bin/env python3
"""Seal native Task Arena provider-zero from retained terminal evidence.

This is a read-only closeout over an already-terminal run.  It performs no
provider mutation and refuses incomplete watchdog, cleanup, or teardown proof.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from blueprint_pipeline.native_task_arena_paid_authority import (
    materialize_native_task_arena_provider_zero,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = materialize_native_task_arena_provider_zero(
            authority_path=args.authority,
            result_path=args.result,
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
                "status": "completed",
                "receipt_digest": receipt.get("receipt_digest"),
                "output": str(Path(args.output).expanduser().resolve()),
                "provider_mutation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
