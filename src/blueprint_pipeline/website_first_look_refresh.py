"""A first look on paper for a job handed off before the first look existed, beside what was handed off.

A website job handed off before ``first_look_verdict`` existed (``website_robot_fit``) has a robot fit
table whose rows say only the old ``verdict``, so the site's first look (status v8,
``website_robot_fit_projection.read_robot_fit_v2``) reads ``unknown`` for almost every robot. That table
is sealed: its own ``table_digest``, named by the sealed handoff record
(``pipeline/website_scene_preparation/handoff.json``), which every reader checks. It is never rewritten.

Instead the preview pass (``website_atlas_preview.first_look_refresh_tick``) builds a refreshed table
for such a capture in its own place, ``pipeline/first_look/robot_fit_table.json``, with the very
function the hand-off uses (``website_scene_handoff._robot_fit_table``: the same builder, catalog and
chains, written atomically), from inputs the hand-off left on disk: the preparation it names, the
capability verdict it recorded and the collider that verdict was computed on. A small status record
beside it (``first_look_refresh.json``, digest-sealed) binds the refreshed table to the task context
digest, the preparation and the handed-off table it stands beside. The v2 reader uses the refreshed
table only when the handed-off one verifies, has no first-look fields, and the record binds both to
the same task context digest; every other reader is unchanged.

CPU and local files only: it buys nothing and calls nothing outside the machine. It is informational,
so it never stops the pass or another capture: a failed build (a missing input included) is recorded
with its error type and retried on a later pass, at most once an hour (``RETRY_SECONDS``) and at most
``MAX_ATTEMPTS`` times for one binding, after which the record says ``gave_up`` and the capture is only
read. A built table is written once and never rebuilt unless what it is bound to changes (a new
hand-off or task context), so a refreshed capture can go idle and its workspace be retired
(``website_scene_workspace_retention`` waits for every entry's modification time to be old). One capture
is built at a time; the collider's binary copy and stage live only inside that build's read scope
(``usd_binary_read_cache.read_scope``), released before the next capture. A withdrawn capture is skipped, and its
withdrawal deletion removes this folder with the rest of the site's files; the lock here is held by
that deletion as it holds the preview locks.
"""
from __future__ import annotations

import fcntl
import logging
import os
import re
import stat
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .website_atlas_preview_store import capture_withdrawn

logger = logging.getLogger(__name__)

DIRECTORY = ("pipeline", "first_look")
TABLE_FILE = "robot_fit_table.json"
STATUS_FILE = "first_look_refresh.json"
LOCK_FILE = "first_look_refresh.lock"
STATUS_SCHEMA = "website_first_look_refresh.v1"
#: A failed build is tried again on a later pass, at most this often.
RETRY_SECONDS = 3600
#: Build attempts for one binding before the pass stops touching the capture (``gave_up``).
MAX_ATTEMPTS = 3
#: As the company clip's pass: no capture is started after this many seconds of one pass.
TICK_BUDGET_SECONDS = 900
_ERROR = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,127}")


class FirstLookRefreshBusy(RuntimeError):
    pass


def has_first_look(table: Mapping[str, Any]) -> bool:
    """Whether a fit table was written with first-look fields (any row's ``first_look_verdict``)."""
    return any(isinstance(row, Mapping) and "first_look_verdict" in row for row in table.get("rows") or [])


def _inside(root: Path, value: Any) -> Path:
    if type(value) is not str:
        raise ValueError("website_first_look_refresh_input_missing")
    path = Path(value).absolute()
    if (".." in path.parts or not path.is_relative_to(root)
            or any(item.is_symlink() for item in (path, *path.parents)) or not path.is_file()):
        raise ValueError("website_first_look_refresh_input_unavailable")
    return path


def handed_off(capture_root: Path) -> dict[str, Any] | None:
    """The hand-off's verified fit table and the inputs to build it again, or None when it has no table.

    None when there is no hand-off table the site's reader would verify (that reader then shows nothing
    either). A table with first-look fields comes back alone (``{"route", "table"}``): it needs nothing.
    When a verified old-format table exists but an input to rebuild it is missing or does not verify, it
    comes back with ``error`` (the exception's type) and no inputs: recorded as a failed build for its
    binding, like any other failure.
    """
    from .website_preparation_outcome import _load
    from .website_robot_fit_projection import _verified_table

    root = Path(capture_root).absolute()
    path = root / "pipeline" / "website_scene_preparation" / "handoff.json"
    if not path.exists():
        return None
    try:
        handoff = _load(path, root, "digest")
        preparation = _load(Path(handoff["preparation_path"]), root, "digest")
        context = (preparation.get("binding") or {}).get("task_context_digest")
        found = _verified_table(root, context) if isinstance(context, str) else None
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    if found is None or not isinstance(found[1], Mapping):
        return None
    route, table = found
    if has_first_look(table):
        return {"route": route, "table": table}
    found = {"route": route, "handoff": handoff, "preparation": preparation, "task_context_digest": context,
             "table": table}
    try:
        section = handoff["fit_check"] if route == "geometry_only" else handoff
        reference = section["capability_verdict"]
        verdict = _load(_inside(root, reference["path"]), root, "verdict_digest")
        if verdict["verdict_digest"] != reference.get("digest"):
            raise ValueError("website_first_look_refresh_verdict_changed")
        if route == "geometry_only":
            collision = _inside(root, section["collision"]["path"])
        else:
            runtime = _load(_inside(root, handoff["runtime_inputs"]["path"]), root, "digest")
            if runtime["digest"] != handoff["runtime_inputs"].get("digest"):
                raise ValueError("website_first_look_refresh_runtime_inputs_changed")
            collision = _inside(root, runtime["collision"]["path"])
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return {**found, "error": _error_code(type(exc).__name__)}
    return {**found, "verdict": verdict, "collision_path": collision}


def _binding(found: Mapping[str, Any]) -> dict[str, Any]:
    return {"task_context_digest": found["task_context_digest"],
            "preparation_digest": found["preparation"]["digest"],
            "handoff_table_digest": found["table"]["table_digest"]}


def refreshed_table(capture_root: Path, *, task_context_digest: str, preparation_digest: str,
                    handoff_table_digest: str) -> dict[str, Any] | None:
    """The refreshed table bound to exactly this task context, preparation and handed-off table; else None."""
    from .website_preparation_outcome import _load

    root = Path(capture_root).absolute()
    directory = root.joinpath(*DIRECTORY)
    status_path = directory / STATUS_FILE
    if not status_path.exists():
        return None
    record = _load(status_path, root, "digest")
    if (record.get("schema_version") != STATUS_SCHEMA or record.get("status") != "built"
            or record.get("capture_id") != root.name
            or record.get("task_context_digest") != task_context_digest
            or record.get("preparation_digest") != preparation_digest
            or record.get("handoff_table_digest") != handoff_table_digest):
        return None
    table = _load(directory / TABLE_FILE, root, "table_digest")
    if (table["table_digest"] != record.get("table_digest")
            or table.get("schema_version") != "website_robot_fit_table.v1"
            or table.get("preparation_digest") != preparation_digest
            or not has_first_look(table)
            or not all(isinstance(row, Mapping) and "first_look_verdict" in row for row in table["rows"])):
        return None
    return table


@contextmanager
def _refresh_lock(capture_root: Path) -> Iterator[Path]:
    directory = Path(capture_root).joinpath(*DIRECTORY)
    if not directory.parent.is_dir() or directory.parent.is_symlink():
        raise ValueError("website_first_look_refresh_capture_changed")  # a retired capture is never re-created
    directory.mkdir(mode=0o750, exist_ok=True)
    if directory.is_symlink():
        raise ValueError("website_first_look_refresh_capture_changed")
    descriptor = os.open(directory / LOCK_FILE, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    with os.fdopen(descriptor, "r+") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise ValueError("website_first_look_refresh_lock_unsafe")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise FirstLookRefreshBusy("website_first_look_refresh_busy") from None
        yield directory


def _bound(root: Path, binding: Mapping[str, Any]) -> bool:
    try:
        return refreshed_table(root, **binding) is not None
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False  # unreadable: built again


def _error_code(error: Any) -> str:
    text = str(error or "")
    return text if _ERROR.fullmatch(text) else "unrecorded"


def _record(directory: Path, root: Path) -> Mapping[str, Any] | None:
    """The status record as it stands, read only; None when there is none or it does not verify."""
    from .website_preparation_outcome import _load

    status_path = directory / STATUS_FILE
    if not status_path.exists():
        return None
    try:
        return _load(status_path, root, "digest")
    except (OSError, ValueError):
        return None


def _decided(previous: Mapping[str, Any] | None, binding: Mapping[str, Any], now: float) -> str | None:
    """``gave_up`` or ``waiting_retry`` when the record already settles this pass for ``binding``; else None."""
    if previous is None or any(previous.get(key) != value for key, value in binding.items()):
        return None  # no record, or one for another binding: the count starts again
    if previous.get("status") == "gave_up":
        return "gave_up"
    if (previous.get("status") == "failed" and isinstance(previous.get("attempted_at_epoch"), int)
            and now - previous["attempted_at_epoch"] < RETRY_SECONDS):
        return "waiting_retry"
    return None


def refresh_first_look(capture_root: Path, *, now: Callable[[], float] = time.time) -> dict[str, Any]:
    """Build the refreshed first look for one capture when it needs one; its outcome as ``status``.

    ``withdrawn``, ``not_needed`` (no verified handed-off table, or it already has first-look fields),
    ``current`` (built for what it is bound to), ``waiting_retry`` (failed less than an hour ago),
    ``gave_up`` (``MAX_ATTEMPTS`` builds failed for this binding), ``busy`` (another pass holds it),
    ``built`` or ``failed``.

    What it writes: on a successful build the table and the status record, once; after that the capture
    reads as ``current`` and nothing is written or locked again, so its workspace can go idle. A failure
    rewrites the record at most once an hour and at most ``MAX_ATTEMPTS`` times for one binding (task
    context digest, preparation digest, handed-off table digest); the last says ``gave_up`` and the
    capture is then only read. A new binding starts the count again.
    """
    root = Path(capture_root).absolute()
    if capture_withdrawn(root):
        return {"status": "withdrawn"}
    found = handed_off(root)
    if found is None or "preparation" not in found:
        return {"status": "not_needed"}
    binding = _binding(found)
    directory = root.joinpath(*DIRECTORY)
    # Read-only first: a built, given-up or recently failed capture is settled without a lock or a write.
    if _bound(root, binding):
        return {"status": "current"}
    decided = _decided(_record(directory, root), binding, now())
    if decided is not None:
        return {"status": decided}
    try:
        with _refresh_lock(root) as directory:
            if capture_withdrawn(root):  # under the lock the withdrawal deletion also holds
                return {"status": "withdrawn"}
            if _bound(root, binding):  # another pass built it meanwhile
                return {"status": "current"}
            previous = _record(directory, root)
            decided = _decided(previous, binding, now())
            if decided is not None:
                return {"status": decided}
            same = previous is not None and all(previous.get(key) == value for key, value in binding.items())
            attempts = (previous.get("attempts") if same and isinstance(previous.get("attempts"), int) else 0) + 1
            record: dict[str, Any] = {"schema_version": STATUS_SCHEMA, "capture_id": root.name,
                                      "scene_id": root.parent.parent.name, "attempted_at_epoch": int(now()),
                                      "attempts": attempts, "max_attempts": MAX_ATTEMPTS, **binding}
            if "error" in found:
                record.update(status="failed", error=found["error"], table_digest=None)
            else:
                from .website_scene_handoff import _robot_fit_table
                # The hand-off's own builder, catalog and chains; it writes only TABLE_FILE in ``directory``.
                built = _robot_fit_table(preparation=found["preparation"], verdict=found["verdict"],
                                         collision_path=found["collision_path"], native_root=directory)
                if built.get("status") == "failed":
                    record.update(status="failed", error=_error_code(built.get("error")), table_digest=None)
                else:
                    record.update(status="built", error=None, table_digest=built["digest"])
            if record["status"] == "built":
                record["digest"] = canonical_digest(record, digest_field="digest")
                write_json(directory / STATUS_FILE, record)
                if _bound(root, binding):
                    return {"status": "built"}
                # Written but not readable back as bound: a failure like any other.
                record.update(status="failed", error="refreshed_table_unverified", table_digest=None)
            if attempts >= MAX_ATTEMPTS:
                record["status"] = "gave_up"
            record.pop("digest", None)
            record["digest"] = canonical_digest(record, digest_field="digest")
            write_json(directory / STATUS_FILE, record)
            return {"status": record["status"]}
    except FirstLookRefreshBusy:
        return {"status": "busy"}


def tick(storage_root: Path, *, stop_reason: Callable[[], str | None] | None = None,
         budget_seconds: float = TICK_BUDGET_SECONDS, clock: Callable[[], float] = time.monotonic,
         **options: Any) -> dict[str, int]:
    """One pass over every website capture (``website_company_clip.capture_roots``); counts of each outcome.

    Between captures it stops for a hold, a quiesce or a new release (``stop_reason``, the Atlas preview
    tick's own) and starts no capture after ``budget_seconds``; the captures left are counted ``deferred``.
    One capture's failure is counted ``error`` and never stops the others.
    """
    from .website_company_clip import capture_roots

    counts: dict[str, int] = {}
    started = clock()
    captures = capture_roots(storage_root)
    for index, capture in enumerate(captures):
        reason = stop_reason() if stop_reason is not None else None
        if reason or clock() - started >= budget_seconds:
            counts["deferred"] = len(captures) - index
            counts["stopped_" + (reason or "time_budget")] = 1
            break
        try:
            status = refresh_first_look(capture, **options)["status"]
        except Exception as error:  # noqa: BLE001 - informational: one capture never stops the pass
            logger.warning("website_first_look_refresh_failed type=%s", type(error).__name__)
            status = "error"
        counts[status] = counts.get(status, 0) + 1
    return counts


__all__ = ["DIRECTORY", "LOCK_FILE", "MAX_ATTEMPTS", "RETRY_SECONDS", "STATUS_FILE", "STATUS_SCHEMA", "TABLE_FILE",
           "has_first_look", "handed_off", "refresh_first_look", "refreshed_table", "tick"]
