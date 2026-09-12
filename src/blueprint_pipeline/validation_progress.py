"""Operational heartbeat for long validations; telemetry, never evidence.

A controller restart on 2026-09-12 sat 55 minutes in ``preparing/factory`` with
nothing but a phase name to show for it. The controller binds one sink per
scene; the attempt factory and the prefix selector report which step they are
on, how long the operation has run, and how many bytes the digest scope has
hashed versus reused. The file is rewritten atomically on every heartbeat, is
never sealed, is never read by any gate, and every call is a no-op when no sink
is bound (tests, CLIs, replays).
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
import sys
import time

from .validation_file_digests import digest_scope_stats

SCHEMA = "task_evaluation_validation_progress.v1"
FILENAME = "validation-progress.json"
_SINK: ContextVar[dict | None] = ContextVar("validation_progress_sink", default=None)


@contextmanager
def progress_sink(path, **static):
    """Bind the heartbeat file for one outer operation; nested binds are ignored."""
    if _SINK.get() is not None:
        yield
        return
    token = _SINK.set({"path": Path(path), "started_monotonic": time.monotonic(),
                       "started_at_epoch": time.time(), "static": dict(static), "heartbeats": 0})
    try:
        yield
    finally:
        _SINK.reset(token)


def heartbeat(step: str, **fields):
    """Record the current step with elapsed time and digest counters; None when unbound."""
    sink = _SINK.get()
    if sink is None:
        return None
    sink["heartbeats"] += 1
    stats = digest_scope_stats()
    record = {"schema_version": SCHEMA, **sink["static"], "step": step, **fields,
              "heartbeat_sequence": sink["heartbeats"],
              "started_at_epoch": sink["started_at_epoch"], "updated_at_epoch": time.time(),
              "elapsed_seconds": round(time.monotonic() - sink["started_monotonic"], 3),
              "digests": stats}
    path = sink["path"]
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    hashed_gb = (stats or {}).get("bytes_hashed", 0) / 1e9
    hits = (stats or {}).get("cache_hits", 0)
    extra = " ".join(f"{key}={value}" for key, value in fields.items() if isinstance(value, (str, int, float)))
    print(f"validation-progress step={step} elapsed={record['elapsed_seconds']}s "
          f"hashed_gb={hashed_gb:.2f} cache_hits={hits} {extra}".rstrip(), file=sys.stderr, flush=True)
    return record
