"""Append-only scene progression events and an atomic RFC 8785 status projection."""
from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path
import tempfile

from . import task_evaluation_scene_intake as intake
from .decision_evidence_contracts import cross_runtime_canonical_digest


def require(condition, code):
    if not condition:
        raise ValueError("scene_progression_" + code)


def safe_path(path):
    path = Path(path)
    require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)), "path_unsafe")
    return path


@contextmanager
def intent_lock(directory):
    directory = safe_path(directory)
    descriptor = os.open(directory / "progression.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        yield True
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    descriptor, temporary = tempfile.mkstemp(prefix=".scene-progress-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o440)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _projection(event):
    return intake._seal({"schema_version": "task_evaluation_scene_progression.v1",
        "intent_id": event["intent_id"], "intent_digest": event["intent_digest"],
        "event_sequence": event["sequence"], "last_event_digest": event["event_digest"],
        "status": event["status"], "phase": event["phase"], "blockers": event.get("blockers", []),
        "result_reference": event.get("result_reference"), "state": event.get("state", {}),
        "updated_at_epoch": event["observed_at_epoch"], "provider_allocation_performed": False}, "progression_digest")


def load_progression(directory, intent):
    directory = safe_path(directory)
    events_root = safe_path(directory / "progression-events")
    paths = sorted(events_root.glob("*.json"))
    require(len(paths) <= 10000, "event_limit_reached")
    previous = None
    for index, path in enumerate(paths, start=1):
        event = intake._read(path, "event_digest")
        require(event.get("schema_version") == "task_evaluation_scene_progression_event.v1"
                and event.get("intent_digest") == intent["intent_digest"]
                and event.get("intent_id") == intent["intent_id"] and event.get("sequence") == index
                and event.get("previous_event_digest") == (previous["event_digest"] if previous else None)
                and path.name == f"{index:06d}.json", "event_chain_invalid")
        previous = event
    projection_path = directory / "progression.json"
    projection = intake._read(projection_path, "progression_digest") if projection_path.exists() else None
    if previous is None:
        require(projection is None, "projection_without_events")
        return None
    expected = _projection(previous)
    if projection is not None:
        require(projection.get("intent_digest") == intent["intent_digest"]
                and type(projection.get("event_sequence")) is int
                and 0 < projection["event_sequence"] <= len(paths), "projection_binding_invalid")
        bound = intake._read(paths[projection["event_sequence"] - 1], "event_digest")
        require(projection == _projection(bound), "projection_event_mismatch")
    if projection != expected:
        # A crash after exclusive event creation but before rename is repaired
        # from the complete retained event chain, never by inventing progress.
        atomic_json(projection_path, expected)
    return expected


def advance(directory, intent, prior, *, status, phase, state, blockers=(), result_reference=None, now):
    require(status in {"accepted", "preparing", "awaiting_source", "awaiting_execution", "running",
                       "completed", "needs_input", "blocked"}, "status_invalid")
    require(intake._identifier(phase), "phase_invalid")
    blockers = sorted(set(str(code) for code in blockers))
    content = {"status": status, "phase": phase, "state": state, "blockers": blockers,
               "result_reference": result_reference}
    if prior is not None and all(prior.get(key) == value for key, value in content.items()):
        return prior
    event = intake._seal({"schema_version": "task_evaluation_scene_progression_event.v1",
        "intent_id": intent["intent_id"], "intent_digest": intent["intent_digest"],
        "sequence": 1 if prior is None else prior["event_sequence"] + 1,
        "previous_event_digest": None if prior is None else prior["last_event_digest"],
        "observed_at_epoch": now, **content}, "event_digest")
    root = safe_path(Path(directory) / "progression-events")
    root.mkdir(exist_ok=True, mode=0o750)
    intake.write_exclusive(root / f"{event['sequence']:06d}.json", event)
    projection = _projection(event)
    require(projection["progression_digest"] == cross_runtime_canonical_digest(projection, digest_field="progression_digest"),
            "projection_digest_invalid")
    atomic_json(Path(directory) / "progression.json", projection)
    return projection
