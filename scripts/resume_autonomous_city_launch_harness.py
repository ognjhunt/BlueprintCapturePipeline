#!/usr/bin/env python3
"""Resume an existing Blueprint autonomous city-launch harness run."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blueprint_pipeline.city_launch_autonomy_harness import main


if __name__ == "__main__":
    raise SystemExit(main([*sys.argv[1:], "--resume"]))
