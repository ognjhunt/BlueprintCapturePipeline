"""The reference index: website source registrations and the scene intents that resolve them.

Website scene workspace retirement reads it once per tick and keeps any scene a
registration inside it still serves (``website_scene_workspace_retention``). An
unreadable registration or intent protects every scene, except in the tolerant
mode a deletion privacy requires (``owner_approved_deletion``), which records it
and reads on.
"""

from __future__ import annotations

import math
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .website_scene_workspace_records import _load_json


REGISTRATION_SCHEMA = "website_scene_source_registration.v1"
_SHA256 = "sha256:"


class _IndexRoots(Protocol):
    """What the index reads of a ``RetentionContext``; a protocol, so this module never imports retention."""

    intent_root: Path | None
    binding_root: Path | None
    expired_grace_seconds: int
    unreadable_references: list[dict[str, str]] | None


@dataclass(frozen=True)
class ReferenceIndex:
    """Website source registrations and scene intents, read once per tick."""

    readable: bool
    binding_root: Path | None = None
    #: {"path", "request_digest", "reference_paths", "registered_at_epoch"}
    registrations: tuple[Mapping[str, Any], ...] = ()
    #: {"intent_id", "request_digest", "finished": "completed" | "revoked" | "expired" | None,
    #:  "open_attempts": unsettled attempt ids a revoked or expired intent still holds, which it
    #:  does only within the grace period after it finished}
    intents: tuple[Mapping[str, Any], ...] = ()


class _Unreadable(Exception):
    pass


def _registrations(binding_root: Path | None, skipped: list[str] | None = None) -> tuple[Mapping[str, Any], ...]:
    if binding_root is None:
        raise _Unreadable
    try:
        info = os.lstat(binding_root)
    except FileNotFoundError:
        return ()  # nothing was ever registered
    except OSError as exc:
        raise _Unreadable from exc
    if not stat.S_ISDIR(info.st_mode):
        raise _Unreadable
    try:
        names = sorted(os.listdir(binding_root))
    except OSError as exc:
        raise _Unreadable from exc
    rows: list[Mapping[str, Any]] = []
    for name in names:
        if name.startswith(".") or not name.endswith(".json"):
            continue  # exclusive writers stage hidden temporaries
        try:
            rows.append(_registration_row(Path(binding_root), name))
        except _Unreadable:
            if skipped is None:
                raise
            skipped.append(str(Path(binding_root) / name))
    return tuple(rows)


def _registration_row(binding_root: Path, name: str) -> Mapping[str, Any]:
    state, value, record_info = _load_json(Path(binding_root) / name)
    if state != "ok" or value is None or record_info is None:
        raise _Unreadable
    references = value.get("references")
    request_digest = value.get("request_digest")
    if (
        value.get("schema_version") != REGISTRATION_SCHEMA
        or value.get("registration_digest") != canonical_digest(value, digest_field="registration_digest")
        or not isinstance(request_digest, str)
        or not request_digest.startswith(_SHA256)
        or not isinstance(references, Mapping)
        or not references
        or not all(isinstance(ref, Mapping) and isinstance(ref.get("path"), str) for ref in references.values())
    ):
        raise _Unreadable
    return {
        "path": str(Path(binding_root) / name),
        "request_digest": request_digest,
        "reference_paths": tuple(str(ref["path"]) for ref in references.values()),
        "registered_at_epoch": float(record_info.st_mtime),
    }


def _revoked_at(path: Path) -> float:
    """When an intent was revoked: the time its receipt records, else the receipt file's time."""

    state, receipt, _ = _load_json(path)
    value = receipt.get("revoked_at_epoch") if state == "ok" and receipt is not None else None
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return os.stat(path).st_mtime


def _intent_finished(directory: Path, intent: Mapping[str, Any], *, now: float,
                     expired_grace_seconds: int) -> tuple[str | None, float | None]:
    """Why scene progression will never resolve this intent's source again, and since when.

    Mirrors ``task_evaluation_scene_progression._advance_intent``: a completed
    progression, a revocation, or an elapsed (possibly extended) execution window.
    An owner can still extend an elapsed window, so expiry counts only once the
    grace period after it has passed too. ``(None, None)`` while the intent is open.
    """

    from . import task_evaluation_scene_intake as intake

    projection_path = directory / "progression.json"
    if os.path.lexists(projection_path):
        projection = intake._read(projection_path, "progression_digest")
        if projection.get("intent_digest") != intent["intent_digest"] or projection.get("intent_id") != intent["intent_id"]:
            raise _Unreadable
        if projection.get("status") == "completed":
            return "completed", None
    revocation = directory / "revoked.json"
    if revocation.exists():
        return "revoked", _revoked_at(revocation)
    expiry = intake.effective_execution_expiry(directory, intent)
    if now >= expiry + expired_grace_seconds:
        return "expired", expiry
    return None, None


def _open_attempts(directory: Path) -> tuple[str, ...]:
    """Attempt rows scene progression still treats as live: no validated cancellation or settlement.

    Read the way progression reads them (``attempts/*.json`` under the intent, each
    digest-checked, terminal only through ``validated_cancellation``); anything
    unreadable raises and so protects every scene.
    """

    from . import task_evaluation_scene_intake as intake
    from .task_evaluation_retained_controls_evidence import validated_cancellation

    attempts = directory / "attempts"
    try:
        mode = os.lstat(attempts).st_mode
    except FileNotFoundError:
        return ()
    if not stat.S_ISDIR(mode):
        raise _Unreadable
    live = []
    for path in sorted(attempts.glob("*.json")):
        row = intake._read(path, "attempt_digest")
        if validated_cancellation(directory, row) is None:
            live.append(str(row.get("attempt_id") or path.stem))
    return tuple(live)


def _intents(intent_root: Path | None, *, now: float, expired_grace_seconds: int,
             skipped: list[str] | None = None) -> tuple[Mapping[str, Any], ...]:
    from . import task_evaluation_scene_intake as intake

    if intent_root is None:
        raise _Unreadable
    try:
        if not stat.S_ISDIR(os.lstat(intent_root).st_mode):
            raise _Unreadable
        names = sorted(os.listdir(intent_root))
    except OSError as exc:
        raise _Unreadable from exc
    rows: list[Mapping[str, Any]] = []
    for name in names:
        if not name.startswith("scene-"):
            continue
        directory = Path(intent_root) / name
        try:
            mode = os.lstat(directory).st_mode
            if stat.S_ISLNK(mode):
                raise _Unreadable
            if not stat.S_ISDIR(mode):
                continue  # scene progression reads directories only
            intent = intake._read(directory / "intent.json", "intent_digest")
            request = intent.get("request")
            if intent.get("intent_id") != name or not isinstance(request, Mapping):
                raise _Unreadable
            finished, finished_at = _intent_finished(directory, intent, now=now,
                                                     expired_grace_seconds=expired_grace_seconds)
            # A completed intent's attempts ended with it: progression completes only after
            # joining the attempt's terminal result, and only retired predecessors are ever
            # settled, so its own row stays a spend hold forever. A revoked or expired intent
            # holds an attempt it has not cancelled or settled, but only for the grace period
            # after it finished: materialization copies the workspace inputs into the
            # attempt's own staging, and no factory pass runs for days, so every hold expires.
            # (An expired intent's period has passed by the time it counts as finished.)
            held = finished in {"revoked", "expired"} and now < float(finished_at) + expired_grace_seconds
            rows.append({
                "intent_id": name,
                "request_digest": cross_runtime_canonical_digest(request),
                "finished": finished,
                "open_attempts": _open_attempts(directory) if held else (),
            })
        except Exception as exc:  # noqa: BLE001 - an intent that cannot be read protects every scene
            if skipped is None:
                if isinstance(exc, _Unreadable):
                    raise
                raise _Unreadable from exc
            skipped.append(str(directory))
    return tuple(rows)


def build_reference_index(context: _IndexRoots, *, now: float) -> ReferenceIndex:
    """Read every website source registration and scene intent once; unreadable means protect all."""

    if context.unreadable_references is not None:
        return _tolerant_reference_index(context, now=now, record=context.unreadable_references)
    try:
        registrations = _registrations(context.binding_root)
        intents = _intents(context.intent_root, now=float(now), expired_grace_seconds=context.expired_grace_seconds)
    except _Unreadable:
        return ReferenceIndex(readable=False, binding_root=context.binding_root)
    return ReferenceIndex(readable=True, binding_root=context.binding_root,
                          registrations=registrations, intents=intents)


def _tolerant_reference_index(context: _IndexRoots, *, now: float,
                              record: list[dict[str, str]]) -> ReferenceIndex:
    """The index without what cannot be read, each unreadable file or root recorded (privacy deletions only)."""

    skipped: list[str] = []
    try:
        registrations = _registrations(context.binding_root, skipped)
    except _Unreadable:
        registrations = ()
        skipped.append(str(context.binding_root))
    try:
        intents = _intents(context.intent_root, now=float(now), expired_grace_seconds=context.expired_grace_seconds,
                           skipped=skipped)
    except _Unreadable:
        intents = ()
        skipped.append(str(context.intent_root))
    record.extend({"path": path, "code": "reference_index_unreadable"} for path in skipped)
    return ReferenceIndex(readable=True, binding_root=context.binding_root,
                          registrations=registrations, intents=intents)
