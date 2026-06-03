#!/usr/bin/env python3
"""Write a local Cosmos 3 capture-grounding readiness report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from blueprint_pipeline.synthesis.cosmos3_readiness import (  # noqa: E402
    write_cosmos3_capture_readiness,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate local Cosmos 3 feasibility against Blueprint capture, geometry, "
            "site-reference, and Cosmos Predict artifacts without loading a model."
        )
    )
    parser.add_argument("--capture-root", required=True, help="Staged capture root path.")
    parser.add_argument("--site-id", default=None, help="Optional site_id override.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional report output directory. Defaults to pipeline/cosmos3_readiness/.",
    )
    args = parser.parse_args()

    report = write_cosmos3_capture_readiness(
        capture_root=args.capture_root,
        site_id=args.site_id,
        output_root=args.output_root,
    )
    print(json.dumps(report["artifact_paths"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
