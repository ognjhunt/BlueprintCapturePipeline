#!/usr/bin/env python3
"""Stage the pipeline/geometry contract for a local capture bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from blueprint_pipeline.geometry_stage import build_geometry_stage_contract  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a pipeline/geometry contract under a local staged capture."
    )
    parser.add_argument("--capture-root", required=True, help="Local capture root path")
    parser.add_argument("--provider", default="video_to_world", help="Geometry provider label")
    parser.add_argument(
        "--model",
        default="video_to_world-default",
        help="Provider model identifier",
    )
    parser.add_argument(
        "--execution-mode",
        default="standard",
        choices=("standard", "streaming"),
        help="Execution strategy label for the staged contract",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON path for the runner summary",
    )
    args = parser.parse_args(argv)

    result = build_geometry_stage_contract(
        args.capture_root,
        provider=args.provider,
        model=args.model,
        execution_mode=args.execution_mode,
    )
    payload = result.to_dict()
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        "[geometry-lane] "
        f"status={payload['status']} "
        f"geometry_root={payload['geometry_root']} "
        f"manifest={payload['manifest_path']}"
    )
    return 0 if payload["status"] in {"completed", "completed_degraded"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
