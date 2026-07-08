"""Source-control metadata helpers for generated proof artifacts."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def _git(repo: Path, *args: str) -> str | None:
    if not (repo / ".git").exists():
        return None
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.rstrip()


def git_source_metadata(
    repo: str | Path,
    *,
    repo_name: str = "BlueprintCapturePipeline",
) -> dict[str, Any]:
    """Return non-secret git identity metadata for a generated artifact."""

    repo_path = Path(repo).expanduser().resolve()
    head = _git(repo_path, "rev-parse", "HEAD")
    origin_main = _git(repo_path, "rev-parse", "--verify", "origin/main")
    git_metadata_present = (repo_path / ".git").exists()
    return {
        "repo_name": repo_name,
        "path": str(repo_path),
        "git_metadata_present": git_metadata_present,
        "branch": _git(repo_path, "branch", "--show-current"),
        "head": head,
        "origin_main_head": origin_main,
        "head_matches_origin_main": bool(head and origin_main and head == origin_main),
    }
