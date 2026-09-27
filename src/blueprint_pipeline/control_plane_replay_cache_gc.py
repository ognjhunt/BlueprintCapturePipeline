"""Reclaim, from storage GC, the content-store copies activation lookaheads leaked.

Every scene-configuration activation replays its parent preparation under
``<activation>/lookahead`` (``task_evaluation_progression_replay``). Until the
replay released its scratch content store, each one copied the whole store
there: the activations sit on the root disk and the store on the work volume,
so linking failed with EXDEV. Nothing removed the copies; on 2026-09-27, 107
activations held about 13 GiB of them.

Each lookahead is handed to ``completed_replay_cache_retention`` with its
store-copy opt-in and without its single-file rules, so this phase only ever
removes digest-verified content-store copies inside a finished offline parent
replay, each with every name the worker linked to it there. It keeps every
report, including the lookahead report the activation records by path, digest
and size, and every other file. Until the
owner sets ``BLUEPRINT_CONTROL_PLANE_GC_REPLAY_CACHE_RETENTION=1`` a tick removes
nothing and reads no byte: it estimates from names, link counts and sizes, and
only a tick that applies hashes the copies it is about to remove.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import completed_replay_cache_retention as retention

SCHEMA_VERSION = "control_plane_replay_cache_gc.v1"
REPLAY_PARENT_ROOTS_ENV = "BLUEPRINT_CONTROL_PLANE_GC_REPLAY_PARENT_ROOTS"
REPLAY_CACHE_RETENTION_ENV = "BLUEPRINT_CONTROL_PLANE_GC_REPLAY_CACHE_RETENTION"
REPLAY_CACHE_RETENTION_INVALID = "replay_cache_retention_setting_invalid"
LOOKAHEAD_DIRECTORY = "lookahead"
DEFAULT_MINIMUM_CLOSED_SECONDS = 60 * 60
# Store copies only: the standalone unit's single-file rules are not this phase's business.
_RULES = {"reclaim_store_copies": True, "single_files": False}
_MAX_ROWS = 50
_TRUE = frozenset({"1", "true", "yes"})
_FALSE = frozenset({"0", "false", "no"})


def replay_cache_retention_setting(environ: Mapping[str, str] = os.environ) -> tuple[bool, str | None]:
    """Whether the phase may remove copies, and an alert when its setting is invalid.

    Parsed exactly like the scene workspace retirement opt-in: only ``1``,
    ``true`` or ``yes`` enables it; unset, ``0``, ``false`` or ``no`` only plans.
    Any other value only plans and is reported as an alert; it never aborts the
    tick and never follows another opt-in.
    """

    raw = str(environ.get(REPLAY_CACHE_RETENTION_ENV) or "").strip().lower()
    if raw in _TRUE:
        return True, None
    if not raw or raw in _FALSE:
        return False, None
    return False, REPLAY_CACHE_RETENTION_INVALID


def reclaim_replay_caches(
    *,
    parent_roots: Sequence[str | Path],
    apply: bool,
    enabled: bool,
    now: Callable[[], float],
    classifier: Callable[..., Any],
    minimum_closed_seconds: int = DEFAULT_MINIMUM_CLOSED_SECONDS,
    process_root: Path = Path("/proc"),
) -> dict[str, Any]:
    """Estimate every activation's lookahead; plan and remove its copies only when applying and enabled.

    Every parent root must classify as ``work`` and be a real directory before any
    lookahead is read. A tick that will not apply reports ``estimated_candidate_bytes``
    from names, links and sizes and hashes nothing; one that applies reports the
    verified ``candidate_bytes`` of its plans. A linked activation or lookahead is
    reported and never followed. One lookahead's error is recorded by exception type
    and never stops the others. Each row list is capped, with a count of the rows left out.
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
    for root in roots:
        for activation in sorted(root.iterdir()):
            lookahead = activation / LOOKAHEAD_DIRECTORY
            if activation.is_symlink() or lookahead.is_symlink():
                rows["skipped"].append({"root": str(activation), "reason": "symlink_not_followed"})
                continue
            if not activation.is_dir() or not lookahead.is_dir():
                continue
            report["replay_root_count"] += 1
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
    for key, values in rows.items():
        report[key] = values[:_MAX_ROWS]
        report[f"omitted_{key}_count"] = max(0, len(values) - _MAX_ROWS)
    return report


__all__ = [
    "DEFAULT_MINIMUM_CLOSED_SECONDS",
    "REPLAY_CACHE_RETENTION_ENV",
    "REPLAY_CACHE_RETENTION_INVALID",
    "REPLAY_PARENT_ROOTS_ENV",
    "SCHEMA_VERSION",
    "reclaim_replay_caches",
    "replay_cache_retention_setting",
]
