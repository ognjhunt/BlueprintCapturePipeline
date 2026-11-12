"""May this capture's video be shared with approved deployment companies? The one answer.

Owner decision 2026-11-12: recording statement ``2026-11-12.v4`` shares the job, its video
included, with approved deployment companies as the service itself. A capture whose video is
Blueprint's own (``config/blueprint_owned_captures.v1.json``) may be shared too. Every reader of
that permission (the company clip, the job packet tool) asks ``sharing_basis`` here, a leaf with
no Pipeline imports, so live code never imports the offline packet tool.
``capture_original_owner_observer.RECORDING_RIGHTS_STATEMENT_VERSIONS`` takes each version listed
here, so a new sharing version is added here only. A version or an owned capture is added only in a
reviewed change.

This decides the basis only. Consent (granted, not revoked), withdrawal and the takedown gate are
the caller's checks and stop sharing whatever the basis.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SHARING_STATEMENT_VERSIONS: tuple[str, ...] = ("2026-11-12.v4",)
OWNED_CAPTURES_SCHEMA = "blueprint_owned_captures.v1"
OWNED_CAPTURES_PATH = Path(__file__).resolve().parents[2] / "config" / "blueprint_owned_captures.v1.json"
_ROW_KEYS = frozenset({"capture_id", "approved_by", "approved_on", "evidence", "video_sha256_prefix"})
_CAPTURE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}")
#: A row's video digest prefix: 12 to 64 lowercase hex digits, enough to name one video.
_PREFIX = re.compile(r"[0-9a-f]{12,64}")
_TEXT_KEYS = ("approved_by", "approved_on", "evidence")
_DIGEST = re.compile(r"(?:sha256:)?([0-9a-f]{64})")


def covers_sharing(version: Any) -> bool:
    """True only for a statement version listed above, matched exactly."""
    return isinstance(version, str) and version in SHARING_STATEMENT_VERSIONS


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate key")
    return dict(pairs)


def owned_captures(path: Path | None = None) -> tuple[dict[str, str], str | None]:
    """Blueprint's own captures (capture id to video digest prefix), and a flag.

    Strict: an unreadable or malformed list means no owned captures and a flag, never a crash
    and never "everything is owned". Each row has only known keys, a capture id, a
    ``video_sha256_prefix`` of 12 to 64 lowercase hex digits (required: a row names one video,
    never every video under its capture id), non-empty text for ``approved_by``, ``approved_on``
    and ``evidence`` when present, and a real ``approved_on`` date.
    """
    try:
        value = json.loads(Path(path or OWNED_CAPTURES_PATH).read_text(encoding="utf-8"), object_pairs_hook=_unique)
        if (not isinstance(value, dict) or value.get("schema_version") != OWNED_CAPTURES_SCHEMA
                or not isinstance(value.get("captures"), list)):
            raise ValueError("schema")
        owned: dict[str, str] = {}
        for row in value["captures"]:
            if not isinstance(row, dict) or not set(row) <= _ROW_KEYS:
                raise ValueError("row")
            capture_id = row.get("capture_id")
            if type(capture_id) is not str or not _CAPTURE_ID.fullmatch(capture_id) or capture_id in owned:
                raise ValueError("capture_id")
            if any(key in row and (type(row[key]) is not str or not row[key].strip()) for key in _TEXT_KEYS):
                raise ValueError("text")
            if "approved_on" in row:
                date.fromisoformat(row["approved_on"])
            prefix = row.get("video_sha256_prefix")
            if type(prefix) is not str or not _PREFIX.fullmatch(prefix):
                raise ValueError("video_sha256_prefix")
            owned[capture_id] = prefix
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        logger.warning("blueprint_owned_capture_list_invalid type=%s", type(error).__name__)
        return {}, "blueprint_owned_capture_list_invalid"
    return owned, None


def sharing_basis(task_context: Mapping[str, Any], *, video_sha256: str | None = None,
                  owned_list: Path | None = None) -> dict[str, Any] | None:
    """The rule that lets this capture's video be shared, or None.

    ``{"permission_basis": "site_sharing_statement", "statement_version": ...}`` when the task
    context's ``capture_rights.statement_version`` covers sharing; else
    ``{"permission_basis": "blueprint_owned_capture", "capture_id": ...}`` when its ``capture_id``
    is on the allow-list and the digest of the upload the video is made from begins with its row's
    ``video_sha256_prefix``; else None.
    """
    rights = task_context.get("capture_rights") if isinstance(task_context, Mapping) else None
    version = rights.get("statement_version") if isinstance(rights, Mapping) else None
    if covers_sharing(version):
        return {"permission_basis": "site_sharing_statement", "statement_version": version}
    capture_id = task_context.get("capture_id") if isinstance(task_context, Mapping) else None
    owned, _flag = owned_captures(owned_list)
    if type(capture_id) is not str or capture_id not in owned:
        return None
    digest = _DIGEST.fullmatch(video_sha256) if isinstance(video_sha256, str) else None
    if digest is None or not digest.group(1).startswith(owned[capture_id]):
        return None
    return {"permission_basis": "blueprint_owned_capture", "capture_id": capture_id}
