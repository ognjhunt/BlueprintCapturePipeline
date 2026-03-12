#!/usr/bin/env python3
"""Deprecated NeoVerse Stage 1 launcher."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    del argv
    raise SystemExit(
        "NeoVerse is no longer launched through the Stage 1 artifact contract. "
        "Use the persistent NeoVerse runtime service via NEOVERSE_RUNTIME_SERVICE_URL "
        "and build site worlds from evaluation_prep/site_world_spec.json."
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
