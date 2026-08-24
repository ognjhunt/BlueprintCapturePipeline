#!/usr/bin/env python3
"""Seal one retained Arena session after its hard-TTL watchdog proves absence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Sequence

from blueprint_pipeline.native_task_arena_warm_closeout import (
    materialize_expired_warm_closeout,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--retained-result", required=True)
    parser.add_argument("--provider-zero-guard", required=True)
    parser.add_argument("--watchdog-supersession")
    parser.add_argument("--successor-watchdog")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
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
