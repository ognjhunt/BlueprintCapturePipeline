#!/usr/bin/env python3
"""Deprecated local site-world runtime launcher."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    del argv
    raise SystemExit(
        "The named runtime is no longer launched through the Stage 1 artifact contract. "
        "Use the persistent site-world runtime service via SITE_WORLD_RUNTIME_SERVICE_URL "
        "and build site worlds from evaluation_prep/site_world_spec.json."
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
