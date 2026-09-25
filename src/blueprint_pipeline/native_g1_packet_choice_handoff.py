"""Verify a saved G1 choice against its derived scene packet before provisioning.

This read-only check verifies the complete packet and task/site binding. It does
not approve checkpoint rights, establish a movement goal, or start a policy.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .native_g1_development_selection import verify_g1_packet_choice_bundle


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--choice", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    args = parser.parse_args(argv)
    result = verify_g1_packet_choice_bundle(
        setup_path=args.setup, choice_path=args.choice, bundle=args.packet
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
