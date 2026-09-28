"""Why the storage GC kept what it kept: typed retention reasons, with their bytes.

On 2026-09-27 the installed reclaim timer applied with evidence offload enabled
and reclaimed nothing, and its report could not say why: the evidence manifest
folded four different reasons into one ``active_or_unsealed`` counter. A GC
manifest now carries ``retained_by_reason``: every reason it kept an entry for,
as ``{reason: {"count": n, "bytes": b}}``. A reason is a typed string, never a
path, and ``bytes`` are logical bytes (a hardlinked file counts once per name).

``build_storage_gc_summary`` projects one tick's report into the small,
secret-free ``storage-gc/summary.json`` the operator door reads: per phase, the
bytes it planned, removed or offloaded, and kept by reason, and the largest
reasons across phases.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Collection, Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

from . import completed_replay_cache_retention as retention
from .control_plane_storage_pins import load_storage_pins
from .control_plane_storage_references import (
    queue_reference_text,
    settlement_reference_text,
    settlement_reopens_beyond_retained_receipts,
)

PROTECTED_UNREADABLE_SETTLEMENT = "protected_unreadable_settlement"
PROTECTED_PROCESS = "protected_process"
PROTECTED_PROCESS_INVENTORY_UNREADABLE = "protected_process_inventory_unreadable"
PROTECTED_PIN = "protected_pin"
PROTECTED_SETTLEMENT = "protected_settlement"
PROTECTED_QUEUE = "protected_queue"
#: In the order ``evidence_protection_reason`` checks them.
EVIDENCE_PROTECTION_REASONS = (
    PROTECTED_UNREADABLE_SETTLEMENT,
    PROTECTED_PROCESS,
    PROTECTED_PROCESS_INVENTORY_UNREADABLE,
    PROTECTED_PIN,
    PROTECTED_SETTLEMENT,
    PROTECTED_QUEUE,
)


def pin_protection(
    directory: Path, *, pins_root: str | Path, now: Callable[[], float]
) -> tuple[str, frozenset[tuple[str, str]]] | None:
    """The live pins that name ``directory``, lie inside it or contain it: their kinds and identities, or None.

    The kinds are sorted and joined by ``+``, as ``live_pin_kinds`` joins them.
    ``evidence_protection_reason`` decides ``protected_pin`` from the same pins,
    and the evidence manifest counts their kind and owners from this.
    """

    holders: set[tuple[str, str]] = set()
    for pin in load_storage_pins(pins_root, now=now):
        if pin["status"] == "live" and any(
            Path(str(path)) == directory or directory in Path(str(path)).parents or Path(str(path)) in directory.parents
            for path in pin.get("paths") or []
        ):
            holders.add((str(pin["kind"]), str(pin["owner_id"])))
    if not holders:
        return None
    return "+".join(sorted({kind for kind, _owner in holders})), frozenset(holders)


def evidence_protection_reason(
    directory: Path,
    *,
    settlement_roots: Sequence[str | Path],
    pins_root: str | Path,
    queue_roots: Sequence[str | Path],
    now: Callable[[], float],
    ignored_process_ids: Sequence[int] = (),
    process_root: Path = Path("/proc"),
) -> str | None:
    """Why the evidence phase must leave ``directory`` in place, or None when nothing protects it.

    The checks run in the order the storage GC always ran them, and the first that
    holds names the reason:

    1. ``protected_unreadable_settlement``: a settlement root or record cannot be
       read, which proves nothing is unreferenced;
    2. ``protected_process``, or ``protected_process_inventory_unreadable`` when a
       process entry could not be read and none was seen holding the directory;
    3. ``protected_pin``: a live storage pin names it, lies inside it or contains it
       (``pin_protection`` names those pins);
    4. ``protected_settlement``: a settlement record reopens a path under it that
       the offload pointer does not retain;
    5. ``protected_queue``: a pending or processing queue message names it.

    Settlement records and queue messages are re-read on every call.
    ``apply_evidence_offload`` re-checks protection immediately before it evicts
    each candidate, so a settlement written after the manifest was built must
    protect its launch run at that final check too.
    """

    settlement_text, settlement_unreadable = settlement_reference_text(settlement_roots)
    if settlement_unreadable:
        return PROTECTED_UNREADABLE_SETTLEMENT
    process = retention.process_reference(
        directory, process_root=process_root, ignored_process_ids=ignored_process_ids
    )
    if process == retention.PROCESS_INVENTORY_UNREADABLE:
        return PROTECTED_PROCESS_INVENTORY_UNREADABLE
    if process:
        return PROTECTED_PROCESS
    if pin_protection(directory, pins_root=pins_root, now=now) is not None:
        return PROTECTED_PIN
    if settlement_reopens_beyond_retained_receipts(directory.name, settlement_text):
        return PROTECTED_SETTLEMENT
    if directory.name in queue_reference_text(queue_roots):
        return PROTECTED_QUEUE
    return None


def count_retained(
    by_reason: MutableMapping[str, dict[str, Any]],
    reason: str,
    size_bytes: int,
    *,
    kind: str | None = None,
    owners: Collection[Any] | None = None,
) -> None:
    """Count one retained entry of ``size_bytes`` under ``reason``, and under ``kind`` within it when given.

    ``owners``, when given with ``kind``, is every owner the caller has seen hold
    an entry of that kind so far; the kind's ``owner_count`` is their number, so
    an owner holding several entries counts once.
    """

    row = by_reason.setdefault(reason, {"count": 0, "bytes": 0})
    row["count"] += 1
    row["bytes"] += int(size_bytes)
    if kind is not None:
        detail = row.setdefault("by_kind", {}).setdefault(kind, {"count": 0, "bytes": 0})
        detail["count"] += 1
        detail["bytes"] += int(size_bytes)
        if owners is not None:
            detail["owner_count"] = len(owners)


class WalkMeter:
    """A manifest's tree walk, timed and counted, so the cost of sizing what it keeps is on the record.

    It wraps a walk shaped like ``_tree_snapshot`` that returns ``(latest, bytes,
    files)``; ``fields()`` is ``walked_file_count`` and ``walk_seconds``.
    """

    def __init__(self, walk: Callable[[Path], tuple[float, int, int]]):
        self._walk = walk
        self.files = 0
        self.seconds = 0.0

    def __call__(self, directory: Path) -> tuple[float, int, int]:
        started = time.monotonic()
        try:
            result = self._walk(directory)
        finally:
            self.seconds += time.monotonic() - started
        self.files += int(result[2])
        return result

    def fields(self) -> dict[str, Any]:
        return {"walked_file_count": self.files, "walk_seconds": round(self.seconds, 3)}


def walked_bytes(walk: Callable[[Path], tuple[Any, ...]], directory: Path) -> int:
    """The bytes ``walk`` (a manifest's ``_tree_snapshot``) finds in a kept directory.

    A directory that vanished since it was listed holds none; sizing what is
    kept never fails the phase that keeps it.
    """

    try:
        return int(walk(directory)[1])
    except OSError:
        return 0


def entry_bytes(entry: Path) -> int:
    """A link's or a stray file's own size: an unsafe entry is never followed."""

    try:
        return entry.lstat().st_size
    except OSError:
        return 0


def live_pin_kinds(pins_root: str | Path, *, now: Callable[[], float]) -> dict[str, str]:
    """Each live-pinned path with the kinds of the pins that name it, sorted and joined by ``+``.

    For reporting only: the paths are exactly ``live_pinned_paths``, which alone
    decides what is pinned.
    """

    kinds: dict[str, set[str]] = {}
    for pin in load_storage_pins(pins_root, now=now):
        if pin["status"] == "live":
            for path in pin.get("paths") or []:
                kinds.setdefault(str(path), set()).add(str(pin["kind"]))
    return {path: "+".join(sorted(names)) for path, names in kinds.items()}


SUMMARY_SCHEMA_VERSION = "control_plane_storage_gc_summary.v1"
SUMMARY_FILENAME = "summary.json"
MAX_SUMMARY_BYTES = 256 * 1024
TOP_RETAINED = 10
#: Every phase a tick's report can carry, in the order a tick runs them.
PHASES = (
    "stranded_queue_rows",
    "terminal_cache_pins",
    "derived_directories",
    "planned_derived_directories",
    "content_store",
    "result_artifact_offload",
    "evidence_offload",
    "scratch_directories",
    "workspace_bundles",
    "replay_caches",
    "scene_workspaces", "lane_scratch",
)
OPT_INS = ("evidence_offload", "scene_workspace_retirement", "replay_cache_retention", "extended_pin_proofs", "lane_scratch")
_REMOVED_KEYS = ("removed_bytes", "offloaded_bytes", "retired_bytes")
_MAX_REASONS = 50
_MAX_FAILURES = 20
_MAX_LISTED = 50
# A summary copies only typed strings: never a path, a run name or a message.
_TYPED = re.compile(r"[a-z][a-z0-9_:+.-]{0,79}")
_TYPE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,79}")


def _typed(value: Any, fallback: str, pattern: re.Pattern[str] = _TYPED) -> str:
    return value if isinstance(value, str) and pattern.fullmatch(value) else fallback


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _reason_rows(source: Mapping[str, Any], *, max_rows: int | None = _MAX_REASONS) -> dict[str, dict[str, Any]]:
    """``{reason: {count, bytes}}`` for the largest reasons; ``bytes`` is null when unknown.

    A phase that counts its reasons without sizing them (``retained_counts``)
    reports null bytes rather than zero. A reason counted zero times kept
    nothing and is left out, so ``{}`` means the phase kept nothing.
    """

    rows: dict[str, dict[str, Any]] = {}
    for reason, value in source.items():
        row = rows.setdefault(_typed(reason, "unrecognized_reason"), {"count": 0, "bytes": 0})
        detail = value if isinstance(value, Mapping) else {"count": value, "bytes": None}
        size = _integer(detail.get("bytes"))
        row["count"] += _integer(detail.get("count")) or 0
        row["bytes"] = None if size is None or row["bytes"] is None else row["bytes"] + size
        if isinstance(detail.get("by_kind"), Mapping):
            row["by_kind"] = _reason_rows(detail["by_kind"])
        if _integer(detail.get("owner_count")) is not None:
            row["owner_count"] = row.get("owner_count", 0) + detail["owner_count"]
    ranked = sorted(
        (item for item in rows.items() if item[1]["count"] > 0),
        key=lambda item: (-(item[1]["bytes"] or 0), -item[1]["count"], item[0]),
    )
    return dict(ranked if max_rows is None else ranked[:max_rows])


def _phase_summary(entry: Mapping[str, Any]) -> dict[str, Any]:
    by_reason = entry.get("retained_by_reason")
    reasons = by_reason if isinstance(by_reason, Mapping) else entry.get("retained_counts")
    candidate_bytes = _integer(entry.get("candidate_bytes"))
    # Some phases continue after one root fails. The measured zero from the
    # successful roots cannot describe candidates in the failed root.
    partial_scan = bool(entry.get("errors") or entry.get("omitted_errors_count"))
    if partial_scan:
        candidate_bytes = None
    summary: dict[str, Any] = {
        "status": _typed(entry.get("status"), "unrecognized_status"),
        "candidate_bytes": candidate_bytes,
        "removed_or_offloaded_bytes": next((_integer(entry[key]) for key in _REMOVED_KEYS if key in entry), None),
        # None when the phase does not say what it kept (an applied receipt that
        # carries no counts); {} when it kept nothing.
        "retained_by_reason": _reason_rows(reasons) if isinstance(reasons, Mapping) else None,
    }
    if "estimated_candidate_bytes" in entry:
        summary["estimated_candidate_bytes"] = (
            None if partial_scan else _integer(entry["estimated_candidate_bytes"])
        )
    # What sizing the phase's retained trees cost, where it measured it.
    if "walked_file_count" in entry:
        summary["walked_file_count"] = _integer(entry["walked_file_count"])
    if "walk_seconds" in entry:
        seconds = entry["walk_seconds"]
        summary["walk_seconds"] = seconds if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) else None
    if entry.get("status") == "error":
        summary["error_type"] = _typed(entry.get("error"), "Exception", _TYPE_NAME)
    return summary


def _lane_summary(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Count-only observation; private metadata never enters byte rankings."""
    def nonnegative(value: Any) -> int | None:
        return value if type(value) is int and value >= 0 else None
    retained = entry.get("retained_by_reason")
    summary = {
        "status": _typed(entry.get("status"), "unrecognized_status"), "mode": "report_only",
        "complete": entry.get("complete") if type(entry.get("complete")) is bool else None,
        "apply_supported": False if entry.get("apply_supported") is False else None,
        "execution_authorized": False if entry.get("execution_authorized") is False else None,
        "mutations": nonnegative(entry.get("mutations")), "candidate_bytes": None,
        "removed_or_offloaded_bytes": 0 if type(entry.get("removed_bytes")) is int and entry["removed_bytes"] == 0 else None,
        "retained_by_reason": {reason: {"count": row["count"], "bytes": None}
            for reason, row in _reason_rows(retained).items()} if isinstance(retained, Mapping) else None,
    }
    for key in ("registered_count", "unregistered_count", "observed_registered_count",
                "observed_unregistered_count", "logical_bytes", "allocated_bytes"):
        summary[key] = nonnegative(entry.get(key))
    return summary


def _terminal_pin_counts(entry: Mapping[str, Any]) -> dict[str, Any]:
    """The pin phase's candidate and released pin counts, and whether its extended proofs may release.

    Each is null when the report does not say, never zero.
    """

    enabled = entry.get("enabled")
    return {
        "candidate_count": _integer(entry.get("candidate_count")),
        "released_count": _integer(entry.get("released_count")),
        "enabled": enabled if isinstance(enabled, bool) else None,
    }


def _result_artifact_summary(rows: Sequence[Any]) -> dict[str, Any]:
    """One entry for every registry run's per-artifact offload.

    A retained run counts under its ``retained_reason``; a run whose offload
    raised counts under ``offload_failed:<stage>``, and a skipped artifact under
    ``artifact_offload_failed:<stage>`` or ``artifact_skipped:<reason>``. An
    artifact already evicted is gone, not kept, and is not counted. None of them
    is sized, so their bytes are null. ``failures`` groups the errors by scope,
    stage, type and errno.
    """

    # The store module brings in the result-delivery contracts, and only the
    # summary needs its stage names, so it is imported here rather than at load.
    from .task_evaluation_result_artifact_store import OFFLOAD_STAGES

    runs = [row for row in rows if isinstance(row, Mapping)]
    sized = [row for row in runs if row.get("status") in ("dry_run", "applied")]
    complete = len(runs) == len(rows) and all(
        row.get("status") == "retained_hot_or_active"
        or (row.get("status") in ("dry_run", "applied")
            and _integer(row.get("candidate_bytes")) is not None
            and row["candidate_bytes"] >= 0)
        for row in runs
    )
    retained: dict[str, dict[str, Any]] = {}
    failures: dict[tuple[str, str, str, int | None], int] = {}

    def keep(reason: str) -> None:
        row = retained.setdefault(reason, {"count": 0, "bytes": None})
        row["count"] += 1

    def failed(scope: str, row: Mapping[str, Any]) -> str:
        stage = row.get("stage") if row.get("stage") in OFFLOAD_STAGES else "unrecognized_stage"
        key = (scope, stage, _typed(row.get("error_type"), "Exception", _TYPE_NAME), _integer(row.get("errno")))
        failures[key] = failures.get(key, 0) + 1
        return stage

    for run in runs:
        if run.get("status") == "retained_hot_or_active":
            keep(_typed(run.get("retained_reason"), "hot_or_active"))
        elif run.get("status") == "retained":
            keep(f"offload_failed:{failed('run', run)}")
        for skip in run.get("skipped") or ():
            if not isinstance(skip, Mapping) or skip.get("reason") == "already_evicted":
                continue
            if "stage" in skip:
                keep(f"artifact_offload_failed:{failed('artifact', skip)}")
            else:
                keep(f"artifact_skipped:{_typed(skip.get('reason'), 'unrecognized_reason')}")
    ranked = sorted(failures.items(), key=lambda item: (-item[1], item[0][:3], -1 if item[0][3] is None else item[0][3]))
    return {
        "run_count": len(runs),
        # One unsized run makes the total unknown, even if another run measured zero.
        "candidate_bytes": sum(_integer(row["candidate_bytes"]) for row in sized) if complete else None,
        "removed_or_offloaded_bytes": sum(_integer(row.get("offloaded_bytes")) or 0 for row in sized),
        "retained_by_reason": _reason_rows(retained),
        "failures": [
            {"scope": scope, "stage": stage, "error_type": error_type, "errno": number, "count": count}
            for (scope, stage, error_type, number), count in ranked[:_MAX_FAILURES]
        ],
    }


def build_storage_gc_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """The door-readable projection of one tick's report: small, secret-free, and path-free.

    It carries ``schema_version``, ``observed_at_epoch``, the tick's ``status`` and
    ``source_report_digest``, the ``opt_in`` flags (null when the report predates
    them), ``alerts``, ``phase_errors``, ``skipped_roots`` (configured roots that
    were absent: the only paths it names), per phase ``candidate_bytes``,
    ``removed_or_offloaded_bytes`` and ``retained_by_reason`` (for the pin phase
    also ``candidate_count``, ``released_count`` and ``enabled``), ``top_retained``
    phase rows, and ``top_retained_reasons`` aggregated before phase rows are
    capped. Every reason is
    a typed string; anything else becomes ``unrecognized_reason``.
    """

    phases: dict[str, dict[str, Any]] = {}
    reason_totals: dict[str, int] = {}
    for key in PHASES:
        entry = report.get(key)
        if key == "lane_scratch" and isinstance(entry, Mapping):
            phases[key] = _lane_summary(entry)
            continue
        if key == "result_artifact_offload" and isinstance(entry, list):
            phases[key] = _result_artifact_summary(entry)
        elif isinstance(entry, Mapping):
            phases[key] = _phase_summary(entry)
            if key == "terminal_cache_pins":
                phases[key].update(_terminal_pin_counts(entry))
            by_reason = entry.get("retained_by_reason")
            raw_reasons = by_reason if isinstance(by_reason, Mapping) else entry.get("retained_counts")
            if isinstance(raw_reasons, Mapping):
                for reason, row in _reason_rows(raw_reasons, max_rows=None).items():
                    if row["bytes"] and row["bytes"] > 0:
                        reason_totals[reason] = reason_totals.get(reason, 0) + row["bytes"]
    ranked = sorted(
        (
            {"phase": phase, "reason": reason, "count": row["count"], "bytes": row["bytes"]}
            for phase, entry in phases.items()
            for reason, row in (entry["retained_by_reason"] or {}).items()
            if row["bytes"]
        ),
        key=lambda row: (-row["bytes"], row["phase"], row["reason"]),
    )
    opt_in = report.get("opt_in") if isinstance(report.get("opt_in"), Mapping) else {}
    digest = report.get("report_digest")
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "observed_at_epoch": report.get("observed_at_epoch"),
        "status": _typed(report.get("status"), "unrecognized_status"),
        "source_report_digest": digest if isinstance(digest, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", digest) else None,
        "opt_in": {name: opt_in.get(name) if isinstance(opt_in.get(name), bool) else None for name in OPT_INS},
        "alerts": [_typed(alert, "unrecognized_alert") for alert in report.get("alerts") or ()][:_MAX_LISTED],
        "phase_errors": [_typed(phase, "unrecognized_phase") for phase in report.get("phase_errors") or ()][:_MAX_LISTED],
        "skipped_roots": [str(root) for root in report.get("skipped_roots") or ()][:_MAX_LISTED],
        "phases": phases,
        "top_retained": ranked[:TOP_RETAINED],
        "top_retained_reasons": [
            {"reason": reason, "bytes": reason_totals[reason]}
            for reason in sorted(reason_totals, key=lambda name: (-reason_totals[name], name))[:3]
        ],
    }
    if len(json.dumps(summary, indent=2, sort_keys=True).encode()) > MAX_SUMMARY_BYTES:
        raise ValueError("storage_gc_summary_too_large")
    return summary


__all__ = [
    "EVIDENCE_PROTECTION_REASONS",
    "MAX_SUMMARY_BYTES",
    "OPT_INS",
    "PHASES",
    "PROTECTED_PIN",
    "PROTECTED_PROCESS",
    "PROTECTED_PROCESS_INVENTORY_UNREADABLE",
    "PROTECTED_QUEUE",
    "PROTECTED_SETTLEMENT",
    "PROTECTED_UNREADABLE_SETTLEMENT",
    "SUMMARY_FILENAME",
    "SUMMARY_SCHEMA_VERSION",
    "build_storage_gc_summary",
    "count_retained",
    "entry_bytes",
    "evidence_protection_reason",
    "live_pin_kinds",
    "pin_protection",
    "WalkMeter",
    "walked_bytes",
]
