#!/usr/bin/env python3
"""Validate a city-launch evidence run without upgrading it to launch proof."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from blueprint_pipeline.city_launch_evidence_policy import validate_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--max-age-days", type=int, default=7)
    parser.add_argument(
        "--require-disclosure-approval",
        action="store_true",
        help="Also reject unapproved evidence and personal absolute paths.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_age_days < 0:
        raise SystemExit("--max-age-days must be non-negative")
    try:
        result = validate_run(
            args.run_root,
            max_age_days=args.max_age_days,
            require_disclosure_approval=args.require_disclosure_approval,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
