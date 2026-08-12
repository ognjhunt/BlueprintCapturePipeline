#!/usr/bin/env python3
"""Run the immutable Lightwheel sink articulation bundle inside Isaac Sim."""

from __future__ import annotations

import argparse
from pathlib import Path

from blueprint_pipeline.lightwheel_sink_isaac_worker import run_lightwheel_sink_canary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run_lightwheel_sink_canary(args.bundle_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
