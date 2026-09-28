"""Reclaim, from storage GC, the scratch inputs activation lookaheads leaked.

Every scene-configuration activation replays its parent preparation under
``<activation>/lookahead`` (``task_evaluation_progression_replay``). Until the
replay released its scratch inputs, each one left its whole
``prepared-references`` tree there: a copy of the whole content store (the
activations sit on the root disk and the store on the work volume, so linking
failed with EXDEV), the worker's materialized references and whatever else it
wrote. Nothing removed them. On 2026-09-27 the activations measured 12.95 GiB,
of which the store-copy rule alone estimated 4.8 GB; a lookahead listed that
day held essentially all of its bytes in that tree, much of it under no store
name.

Each lookahead is handed to ``completed_replay_cache_retention`` with its
scratch-input and store-copy opt-ins and without its single-file rules, so this
phase only ever removes files inside a finished offline replay's
``prepared-references``. For a parent replay, which is what a lookahead runs,
that is every regular file there whose links are all inside that tree and that
is not newer than the replay's report, each inode with all of its names, and
then the directories left empty inside the tree; for any other replay, only
digest-verified store copies. A file with a link anywhere else is kept. It
keeps every report, including the lookahead report the activation records by
path, digest and size, the replay's scratch queue and every other file. Until
the owner sets ``BLUEPRINT_CONTROL_PLANE_GC_REPLAY_CACHE_RETENTION=1`` a tick
removes nothing, hashes nothing, reads no file's bytes and sweeps no process
table: it estimates from names, link counts and sizes, though it still parses
each replay's report. A tick that applies reads no scratch input's bytes
either, since nothing that makes one scratch rests on a digest; it rechecks
each by inode, links, size and mtime.

Scratch several lookaheads share (``control_plane_replay_cache_shared_scratch``) is
planned once per tick across every lookahead scanned, after each one's own pass, and
reported under ``shared_scratch``. It is removed only when the tick applies with
``BLUEPRINT_CONTROL_PLANE_GC_REPLAY_CACHE_SHARED_SCRATCH=1`` beside the retention
opt-in, and only then do its bytes join the phase's totals.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import completed_replay_cache_retention as retention
from . import control_plane_replay_cache_shared_scratch as shared_scratch

SCHEMA_VERSION = "control_plane_replay_cache_gc.v1"
REPLAY_PARENT_ROOTS_ENV = "BLUEPRINT_CONTROL_PLANE_GC_REPLAY_PARENT_ROOTS"
REPLAY_CACHE_RETENTION_ENV = "BLUEPRINT_CONTROL_PLANE_GC_REPLAY_CACHE_RETENTION"
REPLAY_CACHE_RETENTION_INVALID = "replay_cache_retention_setting_invalid"
REPLAY_CACHE_SHARED_SCRATCH_ENV = "BLUEPRINT_CONTROL_PLANE_GC_REPLAY_CACHE_SHARED_SCRATCH"
REPLAY_CACHE_SHARED_SCRATCH_INVALID = "replay_cache_shared_scratch_setting_invalid"
LOOKAHEAD_DIRECTORY = "lookahead"
DEFAULT_MINIMUM_CLOSED_SECONDS = 60 * 60
# A finished parent replay's whole scratch inputs, which subsume its store copies; the standalone
# unit's single-file rules are not this phase's business.
_RULES = {"reclaim_store_copies": True, "reclaim_scratch_inputs": True, "single_files": False}
_MAX_ROWS = 50
_TRUE = frozenset({"1", "true", "yes"})
_FALSE = frozenset({"0", "false", "no"})


def _truthy_setting(environ: Mapping[str, str], name: str, invalid_code: str) -> tuple[bool, str | None]:
    """An owner's opt-in: whether it is on, and ``invalid_code`` as an alert when its value is not a setting.

    Only ``1``, ``true`` or ``yes`` turns it on; unset, ``0``, ``false`` or ``no`` leaves it
    off. Any other value leaves it off and is reported; it never aborts a tick. Storage GC's
    scene workspace retirement opt-in is read the same way.
    """

    raw = str(environ.get(name) or "").strip().lower()
    if raw in _TRUE:
        return True, None
    if not raw or raw in _FALSE:
        return False, None
    return False, invalid_code


def replay_cache_retention_setting(environ: Mapping[str, str] = os.environ) -> tuple[bool, str | None]:
    """Whether the phase may remove copies, and an alert when its setting is invalid.

    Its own opt-in, parsed exactly like the scene workspace retirement opt-in: an invalid
    value only plans and alerts, and it never follows another opt-in.
    """

    return _truthy_setting(environ, REPLAY_CACHE_RETENTION_ENV, REPLAY_CACHE_RETENTION_INVALID)


def replay_cache_shared_scratch_setting(environ: Mapping[str, str] = os.environ) -> tuple[bool, str | None]:
    """Whether scratch several lookaheads share may go too, and an alert when its setting is invalid.

    Parsed like the retention opt-in, and it only ever widens it: without that one it removes
    nothing, and that one never turns it on.
    """

    return _truthy_setting(environ, REPLAY_CACHE_SHARED_SCRATCH_ENV, REPLAY_CACHE_SHARED_SCRATCH_INVALID)


def reclaim_replay_caches(
    *,
    parent_roots: Sequence[str | Path],
    apply: bool,
    enabled: bool,
    now: Callable[[], float],
    classifier: Callable[..., Any],
    minimum_closed_seconds: int = DEFAULT_MINIMUM_CLOSED_SECONDS,
    process_root: Path = Path("/proc"),
    shared_scratch_enabled: bool = False,
) -> dict[str, Any]:
    """Estimate every activation's lookahead; plan and remove its copies only when applying and enabled.

    Every parent root must classify as ``work`` and be a real directory before any
    lookahead is read. A tick that will not apply reports ``estimated_candidate_bytes``
    from names, links and sizes and hashes nothing; one that applies reports the
    verified ``candidate_bytes`` of its plans. A linked activation or lookahead is
    reported and never followed. One lookahead's error is recorded by exception type
    and never stops the others. Each row list is capped, with a count of the rows left out.
    Then scratch the lookaheads share is planned across all of them, and removed only with
    ``shared_scratch_enabled`` on a tick that applies. A failure of a pass that removes is one
    more error; one of a pass that only reports stays in its own block.
    """

    roots = [Path(root).expanduser() for root in parent_roots]
    for root in roots:
        classifier(str(root), expected="work", code="control_plane_storage_gc_replay_root_class")
        if root.is_symlink() or not root.is_dir():
            raise ValueError("control_plane_storage_gc_replay_root_unsafe")
    applying = bool(apply and enabled)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "enabled": bool(enabled),
        "status": "applied" if applying else "dry_run",
        "replay_root_count": 0,
        "candidate_bytes" if applying else "estimated_candidate_bytes": 0,
        "removed_bytes": 0,
    }
    rows: dict[str, list[dict[str, Any]]] = {"kept": [], "skipped": [], "errors": []}
    lookaheads: list[Path] = []
    for root in roots:
        for activation in sorted(root.iterdir()):
            lookahead = activation / LOOKAHEAD_DIRECTORY
            if activation.is_symlink() or lookahead.is_symlink():
                rows["skipped"].append({"root": str(activation), "reason": "symlink_not_followed"})
                continue
            if not activation.is_dir() or not lookahead.is_dir():
                continue
            report["replay_root_count"] += 1
            lookaheads.append(lookahead)
            scope = {"replay_root": lookahead, "minimum_closed_seconds": minimum_closed_seconds,
                     "now": now(), **_RULES}
            try:
                if not applying:
                    estimate = retention.estimate_replay_cache_retention(**scope)
                    report["estimated_candidate_bytes"] += estimate["estimated_candidate_bytes"]
                    continue
                plan = retention.plan_replay_cache_retention(**scope, process_root=process_root)
                report["candidate_bytes"] += plan["candidate_bytes"]
                rows["kept"].extend(plan["kept"])
                if plan["rows"]:
                    result = retention.apply_replay_cache_retention(
                        plan, ack=retention.ACK, process_root=process_root, **_RULES)
                    report["removed_bytes"] += result["removed_bytes"]
                    rows["skipped"].extend(result["skipped"])
            except Exception as exc:  # noqa: BLE001 - one lookahead never costs the others
                rows["errors"].append({"root": str(lookahead), "error": type(exc).__name__})
    # After every lookahead's own pass, so the inodes one replay holds alone are gone first.
    shared_applies = applying and bool(shared_scratch_enabled)
    try:
        shared = shared_scratch.reclaim_shared_scratch(
            lookaheads, now=now(), minimum_closed_seconds=minimum_closed_seconds,
            enabled=bool(enabled and shared_scratch_enabled), apply=shared_applies, check_readers=applying,
            process_root=process_root)
    except Exception as exc:  # noqa: BLE001 - the lookaheads' own passes stand
        if shared_applies:
            # Only a pass that would have removed makes the phase's totals incomplete; one that only
            # reports keeps its failure in its own block, and the retention switch's numbers stand.
            rows["errors"].append({"scope": "shared_scratch", "error": type(exc).__name__})
        shared = {"enabled": bool(enabled and shared_scratch_enabled), "status": "error",
                  "error": type(exc).__name__}
    else:
        if shared_applies:
            report["candidate_bytes"] += shared["candidate_bytes"]
            report["removed_bytes"] += shared["removed_bytes"]
    report["shared_scratch"] = shared
    for key, values in rows.items():
        report[key] = values[:_MAX_ROWS]
        report[f"omitted_{key}_count"] = max(0, len(values) - _MAX_ROWS)
    return report


__all__ = [
    "DEFAULT_MINIMUM_CLOSED_SECONDS",
    "REPLAY_CACHE_RETENTION_ENV",
    "REPLAY_CACHE_RETENTION_INVALID",
    "REPLAY_CACHE_SHARED_SCRATCH_ENV",
    "REPLAY_CACHE_SHARED_SCRATCH_INVALID",
    "REPLAY_PARENT_ROOTS_ENV",
    "SCHEMA_VERSION",
    "reclaim_replay_caches",
    "replay_cache_retention_setting",
    "replay_cache_shared_scratch_setting",
]
