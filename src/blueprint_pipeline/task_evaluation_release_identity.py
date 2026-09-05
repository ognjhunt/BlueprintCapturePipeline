"""Exact release identity of the checkout that owns a running control-plane module.

Production units run from a detached release worktree named after its commit, so
the ``.git`` pointer resolves to a ``HEAD`` holding the exact 40-hex commit.  A
branch checkout (``ref: ...``) or a loose directory has no release identity and
reports the empty string, so callers can fall back to unfiltered behaviour.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["running_release_commit"]

_COMMIT = re.compile(r"[0-9a-f]{40}")


def running_release_commit(module_path: str | Path | None = None) -> str:
    """Read the exact detached/worktree commit that owns ``module_path`` (default: this file)."""

    start = Path(module_path or __file__).resolve()
    for candidate in (start, *start.parents):
        marker = candidate / ".git"
        if not marker.exists():
            continue
        head_path = marker / "HEAD"
        if marker.is_file():
            try:
                pointer = marker.read_text(encoding="utf-8").strip()
            except OSError:
                return ""
            if not pointer.startswith("gitdir:"):
                return ""
            git_root = Path(pointer.split(":", 1)[1].strip())
            if not git_root.is_absolute():
                git_root = (candidate / git_root).resolve()
            head_path = git_root / "HEAD"
        try:
            head = head_path.read_text(encoding="utf-8").strip().lower()
        except OSError:
            return ""
        return head if _COMMIT.fullmatch(head) else ""
    return ""
