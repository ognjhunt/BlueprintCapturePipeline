"""Optional family authoring on the existing source-bound preparation status."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

from .decision_evidence_contracts import canonical_digest
from .runtime_source_identity import running_source_identity
from .task_family_authoring_status import project_family_authoring_status

_MAX_RECEIPT_BYTES = 16 * 1024**2
_UNAVAILABLE = "website_family_authoring_projection_unavailable"


class FamilyProjectionUnavailable(ValueError):
    """Fixed private diagnostic, without retained paths or source contents."""


def _require(value, code=_UNAVAILABLE):
    if not value:
        raise FamilyProjectionUnavailable(code)


def _version(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result)
        result[key] = value
    return result


def _read(path):
    _require(".." not in path.parts and not any(item.is_symlink() for item in (path, *path.parents)))
    before = path.stat()
    _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 < before.st_size <= _MAX_RECEIPT_BYTES)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(descriptor, "rb") as stream:
        _require(_version(os.fstat(stream.fileno())) == _version(before))
        raw = stream.read(_MAX_RECEIPT_BYTES + 1)
        _require(_version(os.fstat(stream.fileno())) == _version(before))
    _require(len(raw) == before.st_size and _version(path.stat()) == _version(before))
    value = json.loads(raw, object_pairs_hook=_unique)
    _require(type(value) is dict)
    return value, hashlib.sha256(raw).hexdigest(), _version(before)


def read_family_authoring_projection(*, capture_root, task_context, birth, ledger, handoff, family_prepared):
    """Return a bounded optional DTO and sanitized diagnostic; never do work.

    Only a completed committed family preparation has an immutable authoring
    result at this ledger revision. Processing-stage files are not status truth.
    The caller reopens current source, task context and withdrawal authority.
    """
    if ledger.get("status") != "completed" or not family_prepared:
        return None, None
    try:
        root = Path(capture_root).absolute() / "pipeline" / "website_scene_preparation"
        before_handoff = _read(root / "handoff.json")
        before_preparation = _read(root / "preparation.json")
        preparation = before_preparation[0]
        _require(before_handoff[0] == handoff
            and handoff.get("digest") == canonical_digest(handoff, digest_field="digest")
            and handoff.get("preparation_digest") == preparation.get("digest")
            and handoff.get("task_evaluation_inputs") == preparation.get("task_evaluation_inputs"))
        identity = running_source_identity(__file__)
        _require(identity.get("state") == "proven", "website_family_authoring_source_identity_unproven")
        projection = project_family_authoring_status(preparation=preparation, task_context=task_context,
            pipeline_source_commit=identity["source_commit"])
        _require(projection["source_video_digest"] == birth["raw_video"].get("sha256"),
                 "website_family_authoring_original_video_unverified")
        _require(_read(root / "handoff.json") == before_handoff
            and _read(root / "preparation.json") == before_preparation
            and running_source_identity(__file__) == identity)
        return projection, None
    except FamilyProjectionUnavailable as exc:
        return None, str(exc)
    except (OSError, ValueError, KeyError, TypeError, RecursionError):
        return None, _UNAVAILABLE
