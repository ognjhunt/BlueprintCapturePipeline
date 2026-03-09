#!/usr/bin/env python3
"""CLI wrapper for pipeline log summarization."""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_repo_src() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> int:
    _bootstrap_repo_src()
    from blueprint_pipeline.log_summary import main as _main

    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
