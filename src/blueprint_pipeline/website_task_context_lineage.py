"""A brief confirmed after preparation: the sealed task context's confirmed successor.

Decision (lead, 2026-11-07, on the owner's first-principles direction): confirming a brief is consent
state, not task content. A current task context that differs from the sealed one only in the site's
confirmation (``confirmed`` false to true, with the website's ``confirmed_at``) is that context's
confirmed successor. Everything bound to the sealed digest stays valid for it and nothing paid is bought
again: each context-bound reader binds on the parent's artifacts (``artifact_context``,
``accepts_bound_digest``). Any other difference (the task was edited) is not a successor and keeps
today's behaviour (``task_context_changed``, then preparation again).

The successor is always the website's own reading, as the signed task-context read returns it; nothing
here sets, assumes or synthesizes a confirmation. The lineage is recorded in the capture's own pipeline
directory (``LINEAGE_DIRECTORY``), so the site's withdrawal deletes it with the capture, and every reader
re-verifies it from the two contexts it holds rather than trusting the file.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

LINEAGE_SCHEMA = "website_task_context_confirmation_lineage.v1"
LINEAGE_DIRECTORY = "website_task_context_lineage"
#: The fields a confirmation changes: the confirmation itself, its time, and the seal over both.
CONFIRMATION_FIELDS = frozenset({"confirmed", "confirmed_at", "context_digest"})
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _sealed(context: Any) -> bool:
    return (isinstance(context, Mapping) and isinstance(context.get("context_digest"), str)
            and _DIGEST.fullmatch(context["context_digest"]) is not None
            and context["context_digest"] == canonical_digest(dict(context), digest_field="context_digest"))


def _content(context: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in context.items() if key not in CONFIRMATION_FIELDS}


def confirmation_successor(sealed: Any, current: Any) -> dict[str, Any] | None:
    """The lineage when ``current`` is ``sealed``'s confirmed successor; None otherwise.

    Both contexts must carry intact seals. The sealed one is unconfirmed, the current one confirmed
    with the website's confirmation time, and with those fields and the seal removed the two have the
    same canonical bytes.
    """
    if not (_sealed(sealed) and _sealed(current)):
        return None
    if (sealed.get("confirmed") is not False or sealed.get("confirmed_at") is not None
            or current.get("confirmed") is not True
            or not isinstance(current.get("confirmed_at"), str) or not current["confirmed_at"].strip()):
        return None
    if canonical_digest(_content(sealed)) != canonical_digest(_content(current)):
        return None
    return {"schema_version": LINEAGE_SCHEMA, "confirmation_parent_context_digest": sealed["context_digest"],
            "context_digest": current["context_digest"], "confirmed_at": current["confirmed_at"]}


def capture_pipeline_root(path: str | Path) -> Path | None:
    """The capture's own ``pipeline`` directory holding ``path`` (``scenes/<site>/captures/<capture>/pipeline``)."""
    candidate = Path(path).absolute()
    for directory in (candidate, *candidate.parents):
        if directory.name == "pipeline" and directory.parent.parent.name == "captures":
            return directory
    return None


def _lineage_path(pipeline_dir: Path, successor_digest: str) -> Path:
    return Path(pipeline_dir) / LINEAGE_DIRECTORY / (successor_digest[7:] + ".json")


def record_confirmation_lineage(pipeline_dir: str | Path, *, parent: Mapping[str, Any],
                                successor: Mapping[str, Any]) -> dict[str, Any] | None:
    """Keep the parent context beside the capture's pipeline for its confirmed successor; None if it is not one.

    Written once per successor; the same record already there is kept, and a record that no longer
    verifies is replaced only by one that does.
    """
    lineage = confirmation_successor(parent, successor)
    if lineage is None:
        return None
    record = {**lineage, "parent_context": json.loads(json.dumps(dict(parent), allow_nan=False)), "lineage_digest": ""}
    record["lineage_digest"] = canonical_digest(record, digest_field="lineage_digest")
    path = _lineage_path(Path(pipeline_dir), successor["context_digest"])
    if read_confirmation_parent(pipeline_dir, successor) is not None:
        return record
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + os.urandom(8).hex())
    try:
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return record


def read_confirmation_parent(pipeline_dir: str | Path | None, context: Any) -> dict[str, Any] | None:
    """The verified sealed parent of ``context`` when it is a recorded confirmed successor; else None."""
    if pipeline_dir is None or not _sealed(context):
        return None
    path = _lineage_path(Path(pipeline_dir), context["context_digest"])
    try:
        if path.is_symlink() or not path.is_file():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    if (not isinstance(record, dict) or record.get("schema_version") != LINEAGE_SCHEMA
            or record.get("lineage_digest") != canonical_digest(record, digest_field="lineage_digest")):
        return None
    parent = record.get("parent_context")
    lineage = confirmation_successor(parent, context)
    if lineage is None or any(record.get(key) != lineage[key] for key in lineage):
        return None
    return dict(parent)


def artifact_context(pipeline_dir: str | Path | None, context: Any) -> Any:
    """The context a reader binds artifacts on: the parent of a verified confirmed successor, else ``context``."""
    parent = read_confirmation_parent(pipeline_dir, context)
    return parent if parent is not None else context


def implied_parent_digests(context: Any) -> frozenset[str]:
    """The digests a confirmed context's unconfirmed parent can have: the same content, unconfirmed.

    A successor differs from its parent only in the confirmation, so the parent is determined by the
    successor: ``confirmed`` false and ``confirmed_at`` null (or absent, as an older projection wrote it).
    Empty for a context that is not a sealed, confirmed one. Nothing here confirms anything.
    """
    if (not _sealed(context) or context.get("confirmed") is not True
            or not isinstance(context.get("confirmed_at"), str) or not context["confirmed_at"].strip()):
        return frozenset()
    content = _content(context)
    return frozenset(canonical_digest({**content, **unconfirmed}) for unconfirmed in (
        {"confirmed": False, "confirmed_at": None}, {"confirmed": False}))


def accepts_bound_digest(context: Any, bound_digest: Any) -> bool:
    """Whether an artifact bound to ``bound_digest`` is valid for ``context``: its own, or its unconfirmed parent's.

    Refuses anything else: an edited task's artifact never matches, and an unconfirmed context accepts only
    its own digest.
    """
    if not isinstance(context, Mapping) or not isinstance(bound_digest, str):
        return False
    return bound_digest == context.get("context_digest") or bound_digest in implied_parent_digests(context)


__all__ = ["CONFIRMATION_FIELDS", "LINEAGE_DIRECTORY", "LINEAGE_SCHEMA", "accepts_bound_digest", "artifact_context",
           "capture_pipeline_root", "confirmation_successor", "implied_parent_digests", "read_confirmation_parent",
           "record_confirmation_lineage"]
