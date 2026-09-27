"""Seal every host change made outside the operator door.

The operator door is the only supported way to change the control-plane host.
Breaking glass stays possible -- changing the host by hand over SSH, or
deploying from a source GPU admission does not trust -- but it must leave a
record.  On 2026-09-26 it left none: a deploy ran from a scratch checkout
through a transient unit, GPU admission then refused every sponsored step for
hours (``gpu_canary_deployed_release_receipt_unverified``), and the hand-made
host changes around it were visible only in shell history.

A break-glass note is one small JSON document under ``DEFAULT_NOTES_ROOT``. It
names who acted (``operator``, ``sudo_user``, ``ssh_client``, ``host``), why
(``reason``), what they did (``actions``, ``paths``) and when
(``created_at``).  It is sealed: ``note_digest`` is the sha256 of the note's
canonical JSON without that field, and the file name carries the creation
time and the first twelve hex digits of the digest, so a note that was edited,
or copied to another name, no longer verifies.  The digest catches an edit
that was not resealed; it is not a signature, and anyone who can write this
root-owned directory can seal a note.

The next deploy reports every note no earlier deploy reported, in its receipt,
and appends each one to ``reported.jsonl`` with the deploy commit.  A fresh
note whose actions include ``deploy-from-untrusted-source`` is also the only
thing that lets the deploy CLI use a source GPU admission would refuse.

The door runs as the service account and reads these files, so the directory
is 0755 and every file 0644, and no field name looks like a credential (door
reads refuse JSON whose keys do).  Record a note with sudo, right after the
change::

    python -m blueprint_pipeline.control_plane_break_glass record \\
        --reason "why" --action unit-restart [--action ...] [--path /abs/path ...]
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import re
import socket
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest


NOTE_SCHEMA = "control_plane_break_glass_note.v1"
REPORT_SCHEMA = "control_plane_break_glass_report.v1"
DEFAULT_NOTES_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/cleanup-receipts")
REPORTED_LEDGER = "reported.jsonl"
#: The action a note must name to authorize a deploy from an untrusted source.
DEPLOY_FROM_UNTRUSTED_SOURCE = "deploy-from-untrusted-source"
MAX_REASON_CHARS = 500
MAX_PATHS = 64
MAX_NOTE_BYTES = 64 * 1024
MAX_LEDGER_BYTES = 16 * 1024 * 1024
#: Clock skew tolerated before a note counts as dated in the future.
FUTURE_SKEW_SECONDS = 300

_FIELDS = frozenset(
    {
        "schema_version",
        "operator",
        "reason",
        "actions",
        "paths",
        "host",
        "ssh_client",
        "sudo_user",
        "created_at_epoch",
        "created_at",
        "note_digest",
    }
)
_OPERATOR = re.compile(r"[A-Za-z0-9][A-Za-z0-9._@-]{0,63}\Z")
_ACTION = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_MAX_FIELD_CHARS = 255
_MAX_PATH_CHARS = 4096
_MAX_EPOCH = 253402300799  # 9999-12-31T23:59:59Z, the last time a note name can carry


class BreakGlassNoteError(ValueError):
    """A typed refusal code.  It never carries a host path or note content."""


def _iso(epoch: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def note_name(note: Mapping[str, Any]) -> str:
    """The only file name a note may have: its creation time and digest prefix."""

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(note["created_at_epoch"]))
    return f"{stamp}-{str(note['note_digest'])[7:19]}.json"


def _checked_operator(operator: Any) -> str:
    if not isinstance(operator, str) or not _OPERATOR.fullmatch(operator):
        raise BreakGlassNoteError("break_glass_operator_invalid")
    return operator


def _checked_reason(reason: Any) -> str:
    if not isinstance(reason, str) or not reason.strip():
        raise BreakGlassNoteError("break_glass_reason_required")
    reason = reason.strip()
    # Printable only: a reason is shown on one line by the door and in receipts.
    if len(reason) > MAX_REASON_CHARS or not reason.isprintable():
        raise BreakGlassNoteError("break_glass_reason_invalid")
    return reason


def _checked_actions(actions: Any) -> list[str]:
    if isinstance(actions, (str, bytes)) or not isinstance(actions, Sequence):
        raise BreakGlassNoteError("break_glass_action_invalid")
    if not actions:
        raise BreakGlassNoteError("break_glass_action_required")
    if not all(isinstance(action, str) and _ACTION.fullmatch(action) for action in actions):
        raise BreakGlassNoteError("break_glass_action_invalid")
    return list(dict.fromkeys(actions))


def _checked_paths(paths: Any) -> list[str]:
    if (
        isinstance(paths, (str, bytes))
        or not isinstance(paths, Sequence)
        or len(paths) > MAX_PATHS
        or not all(
            isinstance(path, str)
            and path.startswith("/")
            and len(path) <= _MAX_PATH_CHARS
            and path.isprintable()
            for path in paths
        )
    ):
        raise BreakGlassNoteError("break_glass_path_invalid")
    return list(paths)


def _check_note(note: Any) -> dict[str, Any]:
    """Schema, exact fields, seal, then every field's shape."""

    if not isinstance(note, dict) or note.get("schema_version") != NOTE_SCHEMA:
        raise BreakGlassNoteError("break_glass_note_schema_invalid")
    if set(note) != _FIELDS:
        raise BreakGlassNoteError("break_glass_note_fields_invalid")
    if note["note_digest"] != canonical_digest(note, digest_field="note_digest"):
        raise BreakGlassNoteError("break_glass_note_digest_mismatch")
    _checked_operator(note["operator"])
    _checked_reason(note["reason"])
    _checked_actions(note["actions"])
    _checked_paths(note["paths"])
    epoch = note["created_at_epoch"]
    if (
        type(epoch) is not int
        or not 0 <= epoch <= _MAX_EPOCH
        or note["created_at"] != _iso(epoch)
    ):
        raise BreakGlassNoteError("break_glass_note_fields_invalid")
    if not isinstance(note["host"], str) or not all(
        value is None
        or (isinstance(value, str) and len(value) <= _MAX_FIELD_CHARS and value.isprintable())
        for value in (note["host"], note["ssh_client"], note["sudo_user"])
    ):
        raise BreakGlassNoteError("break_glass_note_fields_invalid")
    return note


def _open_root(root: Path) -> int | None:
    """Open the notes directory without following a symlink; None when absent."""

    try:
        return os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise BreakGlassNoteError("break_glass_notes_root_unsafe") from None
        raise


def _create_root(root: Path) -> int:
    try:
        os.mkdir(root, 0o755)
    except FileExistsError:
        created = False
    else:
        created = True
    directory = _open_root(root)
    if directory is None:
        raise BreakGlassNoteError("break_glass_notes_root_unsafe")
    if created:
        # The umask may have narrowed the mode; the door must list the directory.
        os.fchmod(directory, 0o755)
    return directory


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        view = view[os.write(fd, view):]


def _publish(directory: int, name: str, payload: bytes) -> None:
    """Create ``name`` exclusively and whole: a full disk leaves no partial note."""

    staging = f".{name}.{os.getpid()}.tmp"
    fd = os.open(
        staging,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o644,
        dir_fd=directory,
    )
    try:
        try:
            os.fchmod(fd, 0o644)
            _write_all(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.link(
                staging,
                name,
                src_dir_fd=directory,
                dst_dir_fd=directory,
                follow_symlinks=False,
            )
        except FileExistsError:
            raise BreakGlassNoteError("break_glass_note_exists") from None
    finally:
        os.unlink(staging, dir_fd=directory)
    os.fsync(directory)


def record_note(
    *,
    root: str | Path = DEFAULT_NOTES_ROOT,
    operator: str,
    reason: str,
    actions: Sequence[str],
    paths: Sequence[str] = (),
    now: Callable[[], float] = time.time,
    environ: Mapping[str, str] = os.environ,
) -> Path:
    """Seal what an operator did outside the door, and return the note's path.

    Refuses an empty reason or no action before touching the disk.  The note
    is written exclusively, 0644, into a 0755 directory created if absent.
    """

    connection = str(environ.get("SSH_CONNECTION") or "").split()
    note: dict[str, Any] = {
        "schema_version": NOTE_SCHEMA,
        "operator": _checked_operator(operator),
        "reason": _checked_reason(reason),
        "actions": _checked_actions(actions),
        "paths": _checked_paths(paths),
        "host": socket.gethostname(),
        "ssh_client": connection[0] if connection else None,
        "sudo_user": environ.get("SUDO_USER") or None,
        "created_at_epoch": int(now()),
    }
    note["created_at"] = _iso(note["created_at_epoch"])
    note["note_digest"] = canonical_digest(note)
    _check_note(note)
    payload = (json.dumps(note, indent=2, sort_keys=True) + "\n").encode("utf-8")
    name = note_name(note)
    directory = _create_root(Path(root))
    try:
        _publish(directory, name, payload)
    finally:
        os.close(directory)
    return Path(root) / name


def _read_regular(
    path: str | Path, *, limit: int, unsafe_code: str, dir_fd: int | None = None
) -> bytes:
    """Read one regular file without following a symlink or blocking on a FIFO."""

    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dir_fd)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise BreakGlassNoteError(unsafe_code)
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise BreakGlassNoteError(unsafe_code)
    return raw


def _verified(
    raw: bytes,
    name: str,
    *,
    max_age_seconds: int | None,
    now: Callable[[], float],
) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (ValueError, RecursionError):
        raise BreakGlassNoteError("break_glass_note_not_json") from None
    try:
        note = _check_note(value)
    except BreakGlassNoteError:
        raise
    except (TypeError, ValueError, ArithmeticError):
        # A hand-made document, e.g. a lone surrogate the digest cannot encode:
        # refused with a code, never with an exception message.
        raise BreakGlassNoteError("break_glass_note_fields_invalid") from None
    if name != note_name(note):
        raise BreakGlassNoteError("break_glass_note_name_mismatch")
    if max_age_seconds is not None:
        age = now() - note["created_at_epoch"]
        if age > max_age_seconds:
            raise BreakGlassNoteError("break_glass_note_expired")
        if age < -FUTURE_SKEW_SECONDS:
            raise BreakGlassNoteError("break_glass_note_from_the_future")
    return note


def verify_note(
    path: str | Path,
    *,
    max_age_seconds: int | None = None,
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Return the note at ``path`` if it is sealed, correctly named and fresh enough.

    Raises ``BreakGlassNoteError`` with a typed code otherwise.
    """

    try:
        raw = _read_regular(path, limit=MAX_NOTE_BYTES, unsafe_code="break_glass_note_unsafe")
    except OSError:
        raise BreakGlassNoteError("break_glass_note_unreadable") from None
    return _verified(raw, Path(path).name, max_age_seconds=max_age_seconds, now=now)


def _reported_names(directory: int) -> set[str]:
    try:
        raw = _read_regular(
            REPORTED_LEDGER,
            limit=MAX_LEDGER_BYTES,
            unsafe_code="break_glass_ledger_unsafe",
            dir_fd=directory,
        )
    except FileNotFoundError:
        return set()
    names: set[str] = set()
    for line in raw.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue  # a torn line reports its note again; it never hides one
        if isinstance(row, dict) and isinstance(row.get("name"), str):
            names.add(row["name"])
    return names


def unreported_notes(root: str | Path = DEFAULT_NOTES_ROOT) -> list[dict[str, Any]]:
    """Every note no deploy has reported yet, oldest first, each with its ``name``.

    A note that does not verify is returned as ``{"name", "error"}``: a damaged
    or hand-edited note is reported, never hidden.  An absent directory has
    nothing to report, and reading it creates nothing.
    """

    directory = _open_root(Path(root))
    if directory is None:
        return []
    try:
        reported = _reported_names(directory)
        notes: list[dict[str, Any]] = []
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".json") or name in reported:
                continue
            try:
                raw = _read_regular(
                    name,
                    limit=MAX_NOTE_BYTES,
                    unsafe_code="break_glass_note_unsafe",
                    dir_fd=directory,
                )
                note = _verified(raw, name, max_age_seconds=None, now=time.time)
            except OSError:
                notes.append({"name": name, "error": "break_glass_note_unreadable"})
            except BreakGlassNoteError as exc:
                notes.append({"name": name, "error": str(exc)})
            else:
                notes.append({"name": name, **note})
        return notes
    finally:
        os.close(directory)


def mark_reported(
    root: str | Path,
    notes: Sequence[Mapping[str, Any]],
    *,
    deploy_commit: str,
    now: Callable[[], float] = time.time,
) -> None:
    """Append one ``reported.jsonl`` line per note, naming the deploy that reported it."""

    if not isinstance(deploy_commit, str) or not _COMMIT.fullmatch(deploy_commit):
        raise BreakGlassNoteError("break_glass_deploy_commit_invalid")
    epoch = int(now())
    rows = []
    for note in notes:
        name = note.get("name")
        if (
            not isinstance(name, str)
            or name in {"", ".", ".."}
            or "/" in name
            or "\0" in name
        ):
            raise BreakGlassNoteError("break_glass_note_name_invalid")
        rows.append(
            {
                "schema_version": REPORT_SCHEMA,
                "name": name,
                "note_digest": note.get("note_digest"),
                "deploy_commit": deploy_commit,
                "reported_at": _iso(epoch),
                "reported_at_epoch": epoch,
            }
        )
    if not rows:
        return
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode("utf-8")
    directory = _open_root(Path(root))
    if directory is None:
        raise BreakGlassNoteError("break_glass_notes_root_missing")
    try:
        fd = os.open(
            REPORTED_LEDGER,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
            0o644,
            dir_fd=directory,
        )
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise BreakGlassNoteError("break_glass_ledger_unsafe")
            os.fchmod(fd, 0o644)
            _write_all(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        os.close(directory)


def note_summary(note: Mapping[str, Any]) -> dict[str, Any]:
    """What a deploy receipt reports for one entry of ``unreported_notes``."""

    if "error" in note:
        return {"name": note.get("name"), "digest": None, "error": note["error"]}
    return {
        "name": note["name"],
        "digest": note["note_digest"],
        "operator": note["operator"],
        "reason": note["reason"],
        "created_at": note["created_at"],
    }


def refusal_code(exc: BaseException) -> str:
    """A typed code for a failure, never its message: messages can carry host paths."""

    if isinstance(exc, BreakGlassNoteError):
        return str(exc)
    if isinstance(exc, OSError):
        return "break_glass_io_error:" + errno.errorcode.get(exc.errno or 0, "unknown")
    return "break_glass_failed:" + type(exc).__name__


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m blueprint_pipeline.control_plane_break_glass",
        description=(
            "Seal a host change made outside the operator door, or list the notes "
            "the next deploy will report."
        ),
    )
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_NOTES_ROOT, help="notes directory (default: %(default)s)"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser(
        "record", help="seal a note; run it with sudo, right after the change"
    )
    record.add_argument(
        "--reason", required=True, help=f"why, in at most {MAX_REASON_CHARS} characters"
    )
    record.add_argument(
        "--action",
        dest="actions",
        action="append",
        required=True,
        metavar="SLUG",
        help=(
            "what was done, as a lowercase slug (repeatable), e.g. unit-restart or "
            f"{DEPLOY_FROM_UNTRUSTED_SOURCE}"
        ),
    )
    record.add_argument(
        "--path",
        dest="paths",
        action="append",
        default=[],
        metavar="PATH",
        help="an absolute path that was changed (repeatable)",
    )
    commands.add_parser("list", help="print the notes no deploy has reported yet")
    args = parser.parse_args(argv)

    try:
        if args.command == "record":
            path = record_note(
                root=args.root,
                operator=os.environ.get("SUDO_USER") or os.environ.get("USER") or "",
                reason=args.reason,
                actions=args.actions,
                paths=args.paths,
            )
            output = {"status": "recorded", "path": str(path), "note": verify_note(path)}
        else:
            output = {
                "status": "listed",
                "unreported": [note_summary(note) for note in unreported_notes(args.root)],
            }
    except (OSError, BreakGlassNoteError) as exc:
        print(json.dumps({"status": "refused", "code": refusal_code(exc)}, sort_keys=True))
        return 2
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


__all__ = [
    "BreakGlassNoteError",
    "DEFAULT_NOTES_ROOT",
    "DEPLOY_FROM_UNTRUSTED_SOURCE",
    "NOTE_SCHEMA",
    "REPORTED_LEDGER",
    "REPORT_SCHEMA",
    "main",
    "mark_reported",
    "note_name",
    "note_summary",
    "record_note",
    "refusal_code",
    "unreported_notes",
    "verify_note",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
