#!/usr/bin/env python3
"""Check that the shared platform doctrine block stays in sync across repos."""

from __future__ import annotations

import sys
from pathlib import Path

START_MARKER = "<!-- SHARED_PLATFORM_CONTEXT_START -->"
END_MARKER = "<!-- SHARED_PLATFORM_CONTEXT_END -->"


def extract_block(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.find(START_MARKER)
    end = text.find(END_MARKER)
    if start == -1 or end == -1:
        raise ValueError(f"missing sync markers in {path}")
    end += len(END_MARKER)
    return text[start:end]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parent
    canonical_path = repo_root / "docs" / "platform_context_core.md"
    targets = [
        workspace_root / "BlueprintCapture" / "PLATFORM_CONTEXT.md",
        workspace_root / "BlueprintValidation" / "PLATFORM_CONTEXT.md",
        workspace_root / "Blueprint-WebApp" / "PLATFORM_CONTEXT.md",
        workspace_root / "BlueprintCapturePipeline" / "PLATFORM_CONTEXT.md",
    ]

    canonical_block = canonical_path.read_text(encoding="utf-8").rstrip("\n")
    mismatches: list[str] = []

    for target in targets:
        block = extract_block(target).rstrip("\n")
        if block != canonical_block:
            mismatches.append(str(target))

    if mismatches:
        print("Shared platform doctrine drift detected:", file=sys.stderr)
        for mismatch in mismatches:
            print(f"- {mismatch}", file=sys.stderr)
        return 1

    print("Shared platform doctrine is in sync across all repos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
