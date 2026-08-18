#!/usr/bin/env python3
"""Seal native Arena provider zero when the attempt's account sweep failed.

Reads retained evidence and one fresh global-zero guard report. Performs no
provider mutation and allocates nothing.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from blueprint_pipeline.native_task_arena_paid_authority import (
    materialize_native_task_arena_recovered_provider_zero,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument(
        "--global-zero-guard",
        required=True,
        help="Fresh gpu_spend_guard.v1 report proving account-wide provider zero.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        receipt = materialize_native_task_arena_recovered_provider_zero(
            authority_path=args.authority,
            result_path=args.result,
            global_zero_guard_path=args.global_zero_guard,
            output_path=args.output,
        )
    except (OSError, ValueError) as exc:
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
                "status": receipt["status"],
                "output": str(Path(args.output).resolve()),
                "receipt_digest": receipt["receipt_digest"],
                "provider_mutation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
