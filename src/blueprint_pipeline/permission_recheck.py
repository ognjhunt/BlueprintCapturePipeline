"""A site's permission changed: look again soon. The one mark every reader of a capture's stored permission
check honours.

The website calls the Pipeline when a job's recording statement or withdrawal is recorded
(``POST .../website-company-clips/{capture_id}/recheck``). The mark makes the capture's stored live
permission check (``pipeline/company_clip/statement_check.json``) stale, so the next pass reads the live
rights again instead of waiting for its window. A remembered website refusal is not dropped by the mark:
it goes only when that new read differs from the rights at the refusal, so rechecks alone cause no upload. The company clip
(``website_company_clip``) and the simulation package both read that check, so both call
``recheck_requested_since``.

The mark only makes a pass look sooner. It decides nothing: no encode, upload or deletion happens on
it, and a withdrawal is still found by each pass's own checks. A leaf: no Pipeline imports.
"""
from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path

DIRECTORY = ("pipeline", "company_clip")
MARK_FILE = "permission_recheck.json"
SCHEMA = "permission_recheck_mark.v1"
#: The website may ask again for the same capture once this many seconds have passed.
MIN_INTERVAL_SECONDS = 60


def _mark_path(capture_root: Path) -> Path:
    return Path(capture_root).joinpath(*DIRECTORY) / MARK_FILE


def requested_at(capture_root: Path) -> int | None:
    """When the last recheck was asked for this capture, or None (no mark, or one that cannot be read)."""
    path = _mark_path(capture_root)
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    at = value.get("requested_at_epoch") if isinstance(value, dict) and value.get("schema_version") == SCHEMA else None
    return at if type(at) is int and at >= 0 else None


def recheck_requested_since(capture_root: Path, epoch: object) -> bool:
    """A recheck was asked at or after ``epoch`` (the time a stored check or refusal was made): that
    record is stale and the live rights should be read again. With no usable ``epoch``, any mark counts."""
    at = requested_at(capture_root)
    if at is None:
        return False
    return not isinstance(epoch, int) or at >= epoch


def request_recheck(capture_root: Path, *, now: float | None = None) -> str:
    """Write the mark: ``"accepted"``, ``"too_soon"`` (asked within ``MIN_INTERVAL_SECONDS``) or
    ``"unavailable"`` (no handed-off capture here, or a folder that is not a plain directory). Never raises.

    Only a capture the Pipeline has handed off (its ``pipeline/`` folder exists) gets a mark; the mark is one
    small file in the capture's own ``pipeline/company_clip/`` folder, written without following links.
    """
    moment = int(time.time() if now is None else now)
    root = Path(capture_root)
    try:
        pipeline, folder = root / DIRECTORY[0], root.joinpath(*DIRECTORY)
        if pipeline.is_symlink() or not pipeline.is_dir() or folder.is_symlink():
            return "unavailable"
        last = requested_at(root)
        if last is not None and 0 <= moment - last < MIN_INTERVAL_SECONDS:
            return "too_soon"
        folder.mkdir(mode=0o750, exist_ok=True)
        if not stat.S_ISDIR(os.lstat(folder).st_mode):
            return "unavailable"
        partial = folder / f".{MARK_FILE}.{os.getpid()}.partial"
        descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o640)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"schema_version": SCHEMA, "requested_at_epoch": moment}, stream)
        os.replace(partial, _mark_path(root))
    except OSError:
        return "unavailable"
    return "accepted"


__all__ = ["MARK_FILE", "MIN_INTERVAL_SECONDS", "recheck_requested_since", "request_recheck", "requested_at"]
