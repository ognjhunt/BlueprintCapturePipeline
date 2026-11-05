"""A bounded, redacted summary of why a provider run stopped, kept beside its records.

A paid GPU run's own account of a failure lives inside its output archive: each
isolated cell's result (status, blockers, failure type, codes and source trace),
the tail of its worker console, and each policy server's log. The host keeps
only a subset of members in ``immutable_execution``, and with stream delivery
the archive never lands on the host at all. Once the run closes, nothing a
session may read says why it stopped.

``summarize_archive`` reads those members while the archive is open for
inspection, as a local file (``summarize_zip_path``) or by range
(``provider_output_remote_collection``). ``vast_provider_adapter`` then writes
the summary as ``provider_runtime_diagnostics.v1.json`` in the run directory
(``retain_for_transfer``) and keeps ``compact`` in its preclassification receipt
and ``vast_provider_command_result.json``. Both happen before teardown and
before anything later can promote or remove the archive.

Rules:

- Only ``.json``, ``.log`` and ``.txt`` members are read. Any other member, such
  as an image, a video or a nested archive, is listed by name only. Nothing is
  extracted, and a member name is never used as a path.
- JSON is reduced to diagnostic fields: status, blockers, failure type and codes,
  trace frames as function, file name and line, selected cell index, and short
  error-message fields. Other content is never copied. JSON nested more than
  ``MAX_JSON_DEPTH`` directories deep, and per-episode records, are only
  counted, except an episode's own failure record (``*.failure_gap.json``,
  ``*.failure_evidence.json``): at most ``MAX_EPISODE_FAILURE_RECORDS`` are read.
- Every line or message kept is credential-redacted (``bounded_error``) and
  trimmed. A log excerpt keeps its last ``TAIL_LINES`` lines and earlier lines
  that look like errors: at most ``EXCERPT_LINES`` lines and ``EXCERPT_BYTES``.
  The whole summary is at most ``MAX_SUMMARY_BYTES``.
- Reading is bounded by member count, per-member size and total bytes. A remote
  caller adds ``stop_reading`` for its transfer budget.
- It never raises. A failure is recorded in the summary.

The summary is diagnostic evidence only: it never grants success and never
changes a run's verdict.
"""

from __future__ import annotations

import json
import os
import re
import stat
import zipfile
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .bounded_error import bounded_error_text, error_description, redact_credentials

SCHEMA_VERSION = "provider_runtime_diagnostics.v1"
FILENAME = SCHEMA_VERSION + ".json"
OUTPUT_ZIP_NAME = "vast_provider_runtime_output.zip"
COMMAND_RESULT_NAME = "vast_provider_command_result.json"

READABLE_SUFFIXES = (".json", ".log", ".txt")
# Gate verdicts sit a few directories down (cell_runs/NN/prepolicy_observation_gate/<gate>/...).
MAX_JSON_DEPTH = 6
MAX_MEMBERS_READ = 256
MAX_JSON_MEMBER_BYTES = 8 * 1024**2
MAX_TEXT_MEMBER_BYTES = 64 * 1024**2
MAX_TOTAL_READ_BYTES = 256 * 1024**2
LINE_READ_BYTES = 4096
LINE_CHARS = 500
TAIL_LINES = 80
ERROR_LINES = 120
EXCERPT_LINES = 200
EXCERPT_BYTES = 64 * 1024
MAX_RESULT_ROWS = 200
#: Per-episode failure records read: each says why one episode stopped.
EPISODE_FAILURE_SUFFIXES = (".failure_gap.json", ".failure_evidence.json")
MAX_EPISODE_FAILURE_RECORDS = 24
MAX_LISTED_MEMBERS = 200
MAX_SUMMARY_BYTES = 1024**2
COMPACT_MAX_BYTES = 32 * 1024
COMPACT_ROWS = 24
COMPACT_LOG_LINES = 6

ERROR_LINE = re.compile(r"Traceback|Error|Exception|FATAL|Segmentation fault|Killed")
# A run that completed with no blockers needs no diagnosis, so a remote reader skips it.
CLEAN_STATUSES = frozenset({
    "completed", "completed_unqualified", "passed", "runtime_completed_unqualified_pending_closeout",
})
_MESSAGE_KEY = re.compile(r"(?i)error|exception|message|reason|failure|stderr|detail")
_EPISODE_FAILURE_FIELDS = ("episode_failure_stage", "episode_failure_cause", "failure_type", "failure_code")
_PROJECTED = frozenset({"status", "blockers", "session_failure_type", "session_failure_codes",
                        "session_failure_source_trace", "selected_cell_index", *_EPISODE_FAILURE_FIELDS})
_TRACE_FILE = re.compile(r"[A-Za-z0-9_.-]{1,160}")
_TRACE_FUNCTION = re.compile(r"[A-Za-z_<][A-Za-z0-9_<>.]{0,159}")
_CONTROL = re.compile(r"[\x00-\x08\x0a-\x1f\x7f]")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _size(value: Any) -> int:
    return len(json.dumps(value, sort_keys=True).encode())


def _text(value: str, limit: int) -> str:
    """Redact, then trim: a credential cut by the trim would no longer match."""
    text = redact_credentials(_CONTROL.sub(" ", value).rstrip())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _name(name: str) -> str:
    return bounded_error_text(name)


def _episode_failure_record(name: str) -> bool:
    return name.lower().endswith(EPISODE_FAILURE_SUFFIXES)


def diagnostics_wanted(inspection: Mapping[str, Any], member_names: Iterable[str] = ()) -> bool:
    """False only for a readable runtime result that completed with no blockers and no episode failure.

    A session that closed cleanly can still hold episodes that stopped before the
    policy ran; their failure records are the only account of why.
    """
    return not (inspection.get("runtime_result_present") is True
                and inspection.get("runtime_result_status") in CLEAN_STATUSES
                and not inspection.get("runtime_result_blockers")
                and not inspection.get("json_parse_errors")
                and not any(_episode_failure_record(name) for name in member_names))


def not_collected(reason: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "generated_at": _now(), "status": "not_collected",
            "reason": reason, "results": [], "logs": [], "raw_secret_values_recorded": False}


def _failed(exc: BaseException, *, status: str = "failed") -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "generated_at": _now(), "status": status,
            "error": error_description(exc), "results": [], "logs": [], "raw_secret_values_recorded": False}


def _unread_reason(info: zipfile.ZipInfo) -> str | None:
    name = info.filename
    path = PurePosixPath(name)
    if (not name or len(name) > 512 or "\\" in name or "\x00" in name or path.is_absolute()
            or ".." in path.parts):
        return "unsafe_member_name"
    if not name.lower().endswith(READABLE_SUFFIXES):
        return "not_a_text_member"
    if stat.S_ISLNK(info.external_attr >> 16):
        return "symbolic_link_member"
    if info.flag_bits & 0x1:
        return "encrypted_member"
    is_json = name.lower().endswith(".json")
    if is_json and (len(path.parts) - 1 > MAX_JSON_DEPTH
                    or "episodes" in path.parts[:-1] and not _episode_failure_record(name)):
        # Per-episode records are counted, never read: a run can hold thousands.
        return "nested_json_member"
    cap = MAX_JSON_MEMBER_BYTES if is_json else MAX_TEXT_MEMBER_BYTES
    if info.file_size > cap or info.compress_size > cap:
        return "over_member_size_cap"
    return None


def _priority(info: zipfile.ZipInfo) -> tuple[int, int]:
    lower = PurePosixPath(info.filename).name.lower()
    rank = (0 if lower.endswith(".json") and "result" in lower or _episode_failure_record(lower)
            else 1 if lower.endswith(".log")
            else 2 if lower.endswith(".json") else 3)
    # Within a rank, archive order keeps a remote reader's blocks local.
    return rank, info.header_offset


def _trace(frames: Any) -> list[dict[str, Any]]:
    """Function, file name and line of each frame; never a path, local or message."""
    if not isinstance(frames, list):
        return []
    rows = []
    for frame in frames[-24:]:
        if not isinstance(frame, Mapping):
            continue
        function, file, line = frame.get("function"), frame.get("file"), frame.get("line")
        if not isinstance(function, str) or not _TRACE_FUNCTION.fullmatch(function):
            continue
        row: dict[str, Any] = {"function": function}
        if isinstance(file, str):
            file_name = PurePosixPath(file.replace("\\", "/")).name
            if _TRACE_FILE.fullmatch(file_name):
                row["file"] = file_name
        if type(line) is int and line >= 0:
            row["line"] = line
        rows.append(row)
    return rows


def _messages(value: Mapping[str, Any], *, limit: int = 12) -> list[dict[str, str]]:
    """Short string fields named like an error, breadth first, four levels deep."""
    found: list[dict[str, str]] = []
    queue: deque[tuple[str, Any, int]] = deque([("", value, 0)])
    visited = 0
    while queue and len(found) < limit and visited < 2000:
        prefix, node, depth = queue.popleft()
        visited += 1
        items = (node.items() if isinstance(node, Mapping)
                 else ((f"[{index}]", child) for index, child in enumerate(node[:20])))
        for key, child in items:
            key = str(key)
            path = (f"{prefix}{key}" if key.startswith("[") else f"{prefix}.{key}" if prefix else key)[:160]
            if depth == 0 and key in _PROJECTED:
                continue
            named = isinstance(node, Mapping) and _MESSAGE_KEY.search(key) is not None
            if named and isinstance(child, str) and child:
                found.append({"field": path, "message": bounded_error_text(child)})
            elif named and isinstance(child, list) and child and all(isinstance(item, str) for item in child):
                found.extend({"field": f"{path}[{index}]", "message": bounded_error_text(item)}
                             for index, item in enumerate(child[:3]) if item)
            elif isinstance(child, (Mapping, list)) and depth < 4:
                queue.append((path, child, depth + 1))
            if len(found) >= limit:
                break
    return found[:limit]


_NESTED_CODE_KEY = re.compile(r"(?i)(?:^|_)blockers$|_codes$")
_TYPED_CODE = re.compile(r"[a-z][a-z0-9_:.=-]{0,199}")


def _nested_codes(value: Mapping[str, Any], *, limit: int = 24) -> list[dict[str, Any]]:
    """Typed code lists below the top level (a gate's ``policy_observation_integrity_blockers``),
    breadth first, four levels deep. Codes only: anything that is not a typed code is dropped."""
    found: list[dict[str, Any]] = []
    queue: deque[tuple[str, Any, int]] = deque([("", value, 0)])
    visited = 0
    while queue and len(found) < limit and visited < 2000:
        prefix, node, depth = queue.popleft()
        visited += 1
        items = (node.items() if isinstance(node, Mapping)
                 else ((f"[{index}]", child) for index, child in enumerate(node[:20])))
        for key, child in items:
            key = str(key)
            path = (f"{prefix}{key}" if key.startswith("[") else f"{prefix}.{key}" if prefix else key)[:160]
            if depth == 0 and key in _PROJECTED:
                continue
            if isinstance(node, Mapping) and _NESTED_CODE_KEY.search(key) and isinstance(child, list):
                codes = [item for item in child[:20] if isinstance(item, str) and _TYPED_CODE.fullmatch(item)]
                if codes:
                    found.append({"field": path, "codes": codes})
            elif isinstance(child, (Mapping, list)) and depth < 4:
                queue.append((path, child, depth + 1))
            if len(found) >= limit:
                break
    return found


def _result_row(member: str, value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    row: dict[str, Any] = {}
    if isinstance(value.get("status"), str):
        row["status"] = _text(value["status"], 120)
    for key in ("blockers", "session_failure_codes"):
        if isinstance(value.get(key), list):
            row[key] = [_text(item, 200) for item in value[key][:20] if isinstance(item, str)]
    if isinstance(value.get("session_failure_type"), str):
        row["session_failure_type"] = _text(value["session_failure_type"], 160)
    for key in _EPISODE_FAILURE_FIELDS:  # an episode failure record's typed verdict
        if isinstance(value.get(key), str):
            row[key] = _text(value[key], 200)
    if type(value.get("selected_cell_index")) is int:
        row["selected_cell_index"] = value["selected_cell_index"]
    trace = _trace(value.get("session_failure_source_trace"))
    if trace:
        row["session_failure_source_trace"] = trace
    messages = _messages(value)
    if messages:
        row["error_messages"] = messages
    nested = _nested_codes(value)
    if nested:
        row["nested_codes"] = nested
    for key in ("passed", "policy_observation_integrity_passed", "frame_structure_passed"):
        if isinstance(value.get(key), bool):
            row[key] = value[key]
    if not row:
        return None
    if isinstance(value.get("episodes"), list):
        row["episode_count"] = len(value["episodes"])
    if isinstance(value.get("candidate_policy_queried"), bool):
        row["candidate_policy_queried"] = value["candidate_policy_queried"]
    member_path = PurePosixPath(member)
    return {"member": _name(member), "directory": _name(str(member_path.parent)) if member_path.parent.parts else "",
            **row}


def _scan_text(stream, budget: int) -> tuple[dict[str, Any], int]:
    """Keep the tail and the error-looking lines of one text member, in bounded memory."""
    tail: deque[tuple[int, str]] = deque(maxlen=TAIL_LINES)
    errors: deque[tuple[int, str]] = deque(maxlen=ERROR_LINES)
    read = lines = error_count = long_lines = 0
    previous = None
    while read < budget:
        raw = stream.readline(min(LINE_READ_BYTES, budget - read))
        if not raw:
            break
        read += len(raw)
        if len(raw) == LINE_READ_BYTES and not raw.endswith(b"\n"):
            long_lines += 1
            while read < budget:  # drop the rest of an over-long line
                rest = stream.readline(min(LINE_READ_BYTES, budget - read))
                read += len(rest)
                if not rest or rest.endswith(b"\n"):
                    break
        lines += 1
        text = raw.decode("utf-8", "replace")
        tail.append((lines, text))
        if ERROR_LINE.search(text):
            error_count += 1
            key = text.strip()
            if key != previous:  # a repeated line is kept once
                errors.append((lines, text))
            previous = key
    truncated = read >= budget and bool(stream.read(1))
    first_tail = tail[0][0] if tail else lines + 1
    redacted = 0

    def clean(text: str) -> str:
        nonlocal redacted
        plain = _CONTROL.sub(" ", text).rstrip()
        kept = redact_credentials(plain)
        redacted += kept != plain
        return kept if len(kept) <= LINE_CHARS else kept[:LINE_CHARS - 3] + "..."

    row = {"line_count": lines, "error_line_count": error_count, "scan_truncated": truncated,
           "over_long_line_count": long_lines,
           "error_lines": [f"{number}: {clean(text)}" for number, text in errors if number < first_tail],
           "tail_first_line": first_tail if tail else None,
           "tail": [clean(text) for _, text in tail]}
    row["redacted_line_count"] = redacted
    excerpt_trimmed = False
    while (len(row["error_lines"]) + len(row["tail"]) > EXCERPT_LINES
           or _size(row["error_lines"]) + _size(row["tail"]) > EXCERPT_BYTES):
        excerpt_trimmed = True
        if row["error_lines"]:
            row["error_lines"].pop(0)
        else:
            row["tail"].pop(0)
            row["tail_first_line"] += 1
    row["excerpt_trimmed"] = excerpt_trimmed
    return row, read


def _fit(summary: dict[str, Any], limit: int) -> None:
    """Halve the longest log excerpt, then drop result rows, until the summary fits."""
    while _size(summary) > limit:
        logs = [row for row in summary["logs"] if row.get("error_lines") or row.get("tail")]
        if logs:
            row = max(logs, key=lambda item: _size(item.get("error_lines", [])) + _size(item.get("tail", [])))
            for key in ("error_lines", "tail"):
                values = row.get(key) or []
                drop = (len(values) + 1) // 2
                row[key] = values[drop:]
                if key == "tail" and drop and row.get("tail_first_line") is not None:
                    row["tail_first_line"] += drop
            row["excerpt_trimmed"] = True
        elif summary["results"]:
            summary["results"].pop()
            summary["omitted_result_rows"] += 1
        else:
            summary["unread_members"] = []
            break


def summarize_archive(archive: zipfile.ZipFile, *, archive_source: str,
                      stop_reading: Callable[[], bool] | None = None) -> dict[str, Any]:
    """The bounded diagnostics summary of one open provider-output archive; never raises."""
    try:
        return _summarize(archive, archive_source=archive_source, stop_reading=stop_reading)
    except Exception as exc:  # noqa: BLE001 - diagnostics never decide a run
        return _failed(exc)


def _summarize(archive: zipfile.ZipFile, *, archive_source: str,
               stop_reading: Callable[[], bool] | None) -> dict[str, Any]:
    infos = [info for info in archive.infolist() if not info.is_dir()]
    plan: list[zipfile.ZipInfo] = []
    unread: dict[str, int] = {}
    listed: list[dict[str, str]] = []

    def skip(info: zipfile.ZipInfo, reason: str) -> None:
        unread[reason] = unread.get(reason, 0) + 1
        # Per-episode JSON is counted, never listed: it would crowd out what matters.
        if reason != "nested_json_member" and len(listed) < MAX_LISTED_MEMBERS:
            listed.append({"member": _name(info.filename), "reason": reason})

    failure_records = 0
    for info in infos:
        reason = _unread_reason(info)
        if not reason and _episode_failure_record(info.filename):
            failure_records += 1
            reason = "episode_failure_record_cap" if failure_records > MAX_EPISODE_FAILURE_RECORDS else None
        if reason:
            skip(info, reason)
        else:
            plan.append(info)
    plan.sort(key=_priority)
    results: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    read_members = read_bytes = json_without_fields = 0
    blockers: list[str] = []
    for info in plan:
        if (read_members >= MAX_MEMBERS_READ or read_bytes >= MAX_TOTAL_READ_BYTES
                or (stop_reading is not None and stop_reading())):
            skip(info, "read_budget_exhausted")
            continue
        try:
            with archive.open(info) as stream:
                if info.filename.lower().endswith(".json"):
                    data = stream.read(MAX_JSON_MEMBER_BYTES + 1)
                    read_bytes += len(data)
                    if len(data) > MAX_JSON_MEMBER_BYTES:
                        raise ValueError("provider_runtime_diagnostics_member_over_cap")
                    row = _result_row(info.filename, json.loads(data.decode("utf-8")))
                    if row is None:
                        json_without_fields += 1
                    elif len(results) < MAX_RESULT_ROWS:
                        results.append(row)
                    else:
                        unread["result_row_cap"] = unread.get("result_row_cap", 0) + 1
                else:
                    budget = min(MAX_TEXT_MEMBER_BYTES, MAX_TOTAL_READ_BYTES - read_bytes)
                    row, used = _scan_text(stream, budget)
                    read_bytes += used
                    path = PurePosixPath(info.filename)
                    logs.append({"member": _name(info.filename),
                                 "directory": _name(str(path.parent)) if path.parent.parts else "",
                                 "size_bytes": info.file_size, **row})
            read_members += 1
        except Exception as exc:  # noqa: BLE001 - one unreadable member never hides the rest
            skip(info, "read_failed:" + type(exc).__name__[:60])
    if any(reason == "read_budget_exhausted" or reason.startswith("read_failed") for reason in unread):
        blockers.append("provider_runtime_diagnostics_partial")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "status": "partial" if blockers else "completed",
        "archive_source": archive_source,
        "member_count": len(infos),
        "read_member_count": read_members,
        "read_bytes": read_bytes,
        "json_members_without_diagnostic_fields": json_without_fields,
        "unread_member_counts": dict(sorted(unread.items())),
        "unread_members": listed,
        "results": results,
        "omitted_result_rows": 0,
        "logs": logs,
        "redacted_line_count": sum(row.get("redacted_line_count", 0) for row in logs),
        "limits": {"tail_lines": TAIL_LINES, "error_lines": ERROR_LINES, "excerpt_lines": EXCERPT_LINES,
                   "excerpt_bytes": EXCERPT_BYTES, "summary_bytes": MAX_SUMMARY_BYTES,
                   "json_member_bytes": MAX_JSON_MEMBER_BYTES, "text_member_bytes": MAX_TEXT_MEMBER_BYTES},
        "blockers": blockers,
        "claims_success": False,
        "raw_secret_values_recorded": False,
    }
    _fit(summary, MAX_SUMMARY_BYTES)
    return summary


def summarize_zip_path(path: str | Path) -> dict[str, Any]:
    """``summarize_archive`` of a local provider-output ZIP; never follows a link, never raises."""
    zip_path = Path(path)
    try:
        if zip_path.is_symlink() or not zip_path.is_file():
            return {**not_collected("provider_runtime_output_zip_absent"), "status": "missing"}
        with zipfile.ZipFile(zip_path) as archive:
            return summarize_archive(archive, archive_source="local_file")
    except Exception as exc:  # noqa: BLE001 - an unreadable archive is recorded, not raised
        return _failed(exc)


def _failing(row: Mapping[str, Any]) -> bool:
    return bool(row.get("status") not in CLEAN_STATUSES or row.get("blockers")
                or row.get("session_failure_type") or row.get("session_failure_codes"))


def compact(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    """The failing results and last error lines, small enough to embed in a command result."""
    rows = [{key: row[key] for key in ("member", "status", "blockers", "session_failure_type",
                                        "session_failure_codes", "session_failure_source_trace",
                                        "selected_cell_index", "episode_count", "candidate_policy_queried",
                                        *_EPISODE_FAILURE_FIELDS, "error_messages") if key in row}
            for row in diagnostics.get("results") or [] if isinstance(row, Mapping) and _failing(row)]
    for row in rows:
        for key, keep in (("blockers", 8), ("session_failure_codes", 8), ("error_messages", 4)):
            if key in row:
                row[key] = row[key][:keep]
        if "session_failure_source_trace" in row:
            row["session_failure_source_trace"] = row["session_failure_source_trace"][-8:]
    logs = []
    for row in diagnostics.get("logs") or []:
        if not isinstance(row, Mapping) or not row.get("error_line_count"):
            continue
        lines = [*row.get("error_lines", []), *(line for line in row.get("tail", []) if ERROR_LINE.search(line))]
        logs.append({"member": row.get("member"), "error_line_count": row["error_line_count"],
                     "last_error_lines": [line[:300] for line in lines[-COMPACT_LOG_LINES:]]})
    value = {"schema_version": SCHEMA_VERSION, "status": diagnostics.get("status"),
             **{key: diagnostics[key] for key in ("reason", "error", "member_count", "read_member_count",
                                                  "unread_member_counts", "blockers") if key in diagnostics},
             "failing_results": rows[:COMPACT_ROWS], "omitted_failing_results": max(0, len(rows) - COMPACT_ROWS),
             "log_errors": logs}
    while _size(value) > COMPACT_MAX_BYTES:
        longest = max(value["log_errors"], key=lambda row: len(row["last_error_lines"]), default=None)
        if longest is not None and len(longest["last_error_lines"]) > 1:
            longest["last_error_lines"] = longest["last_error_lines"][len(longest["last_error_lines"]) // 2:]
        elif value["failing_results"]:
            value["failing_results"].pop()
            value["omitted_failing_results"] += 1
        elif value["log_errors"]:
            value["log_errors"].pop()
        else:
            break
    return value


def retain(run_dir: str | Path, diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    """Write the summary as ``FILENAME`` in ``run_dir``; return ``compact`` with its path. Never raises."""
    from .core.common import write_json

    path = Path(run_dir) / FILENAME
    try:
        write_json(path, diagnostics)
        written: dict[str, Any] = {"written": True}
    except (OSError, TypeError, ValueError) as exc:
        written = {"written": False, "write_error": error_description(exc)}
    try:
        value = compact(diagnostics)
    except Exception as exc:  # noqa: BLE001 - the command result still records the attempt
        value = {"schema_version": SCHEMA_VERSION, "status": "failed", "error": error_description(exc)}
    return {**value, "path": str(path), **written}


def retain_for_transfer(run_dir: str | Path, transfer: Mapping[str, Any] | None,
                        output_zip_path: str | Path) -> dict[str, Any] | None:
    """Keep a just-inspected archive's summary in ``run_dir``; return ``compact``, or None without an archive.

    The caller is where the archive is inspected, before teardown and before
    anything can promote or remove it. A remote observation carries the summary
    it read by range; a local ZIP is read here. A transfer that did not complete
    has nothing to summarize. Never raises.
    """
    try:
        transfer = transfer if isinstance(transfer, Mapping) else {}
        if transfer.get("status") not in (None, "completed"):
            return None
        summary = transfer.get("runtime_diagnostics")
        if not isinstance(summary, Mapping) or not summary:
            zip_path = Path(output_zip_path)
            if zip_path.is_symlink() or not zip_path.is_file():
                return None
            summary = summarize_zip_path(zip_path)
        return retain(run_dir, summary)
    except Exception as exc:  # noqa: BLE001 - a paid run's closeout never depends on diagnostics
        return {"schema_version": SCHEMA_VERSION, "status": "failed", "error": error_description(exc)}


def receipt_transfer(transfer: Mapping[str, Any], retained: Mapping[str, Any] | None) -> dict[str, Any]:
    """A transfer for a receipt: the full summary lives in its own file, so it keeps the compact copy."""
    value = {key: item for key, item in transfer.items() if key != "runtime_diagnostics"}
    if retained:
        value["runtime_diagnostics"] = dict(retained)
    return value


def backfill(run_dirs: Iterable[str | Path], *, max_writes: int = 5) -> dict[str, Any]:
    """Summarize retained runs whose local output archive is still present; idempotent and bounded.

    A run qualifies once its command result exists (the adapter finished
    inspecting) and it has no summary yet. A summary written here, even a
    failed one, is never rewritten, so each pass moves on to new runs.
    """
    if type(max_writes) is not int or not 0 < max_writes <= 50:
        raise ValueError("provider_runtime_diagnostics_backfill_limit_invalid")
    written: list[str] = []
    pending = False
    for run_dir in run_dirs:
        run = Path(run_dir)
        archive = run / OUTPUT_ZIP_NAME
        if (os.path.lexists(run / FILENAME) or run.is_symlink() or not (run / COMMAND_RESULT_NAME).is_file()
                or archive.is_symlink() or not archive.is_file()):
            continue
        if len(written) >= max_writes:
            pending = True
            break
        retain(run, {**summarize_zip_path(archive), "backfilled": True})
        written.append(str(run))
    return {"status": "completed", "written": written, "pending": pending}


__all__ = [
    "FILENAME",
    "SCHEMA_VERSION",
    "backfill",
    "compact",
    "diagnostics_wanted",
    "not_collected",
    "receipt_transfer",
    "retain",
    "retain_for_transfer",
    "summarize_archive",
    "summarize_zip_path",
]
