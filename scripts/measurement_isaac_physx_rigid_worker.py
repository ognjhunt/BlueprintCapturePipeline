#!/usr/bin/env python3
"""Load the source-tree Isaac/PhysX worker in an explicit Isaac runtime."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from blueprint_pipeline.measurement_isaac_physx_rigid_adapter import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
