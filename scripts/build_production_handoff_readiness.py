#!/usr/bin/env python3
"""CLI wrapper for the final capture-to-GPU handoff readiness manifest."""

from __future__ import annotations

from blueprint_pipeline.production_handoff_readiness import main


if __name__ == "__main__":
    raise SystemExit(main())
