"""A finished scene intent's preparation output goes behind a pointer in the artifact store.

``task-evaluation-inputs/completed-scene-preparation/<intent_id>/`` holds the scene factory's work for one scene
intent: its source preparations (``source-*``), the website source it dispatched, the completed source and the
attempts' outputs (``<intent>/<attempt>/``). On 2026-11-19 it was 21 GB in 17 intents going back to 2026-09-12, and
nothing reclaimed it: its storage row said "report only".

The owner approved on 2026-11-19 (relayed by the coordinating session): "Give `completed-scene-preparation` a
cleanup rule". Archive first: an intent's folder is packed, published to the private artifact store with a full
readback, replaced by ``<intent_id>.offloaded.v1.json`` and only then removed, by the evidence offload
(``control_plane_evidence_offload``), so the pointer, ``restore`` and withdrawal deletion (which reads these pointers
in this root, by intent) are the ones every offloaded run already has.

**When.** Only while every reader is done with it:

- its scene intent has finished (``website_scene_workspace_references``, the index workspace retirement reads):
  completed, revoked, or expired past the seven days an owner has to extend it, with no attempt left unsettled
  (the ended-authority settlement reads an attempt's output until it is settled: money);
- nothing in the folder changed for ``IDLE_SECONDS``;
- no pending or processing queue message names the intent, no live storage pin names it, lies inside it or
  contains it, and no live process uses it.

A folder whose intent cannot be found keeps everything (``intent_unknown``), and when the intents cannot be read
every folder does (``reference_index_unreadable``). Release protection finds an intent's runs through the intent
root, never through this folder, so nothing that protects a release is lost. At most ``MAX_PER_TICK`` folders are
archived a tick (each can be gigabytes).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

ROOT = Path("/var/lib/blueprint/task-evaluation-inputs/completed-scene-preparation")
ROOTS_ENV = "BLUEPRINT_CONTROL_PLANE_GC_SCENE_PREPARATION_ROOTS"
SWITCH_ENV = "BLUEPRINT_CONTROL_PLANE_SCENE_PREPARATION_OFFLOAD"
SWITCH_INVALID = "scene_preparation_offload_setting_invalid"
#: As long as an expired intent stays open for its owner, so a folder is idle well past its intent's last pass.
IDLE_SECONDS = 7 * 24 * 3600
MAX_PER_TICK = 5


def setting(environ: Mapping[str, str] = os.environ) -> tuple[bool, str | None]:
    """On by default like every storage GC switch (owner decision 2026-09-30); ``0`` keeps it planning."""

    from .control_plane_replay_cache_gc import _truthy_setting
    return _truthy_setting(environ, SWITCH_ENV, SWITCH_INVALID, default=True)


def configured_roots(environ: Mapping[str, str] = os.environ) -> list[Path]:
    value = str(environ.get(ROOTS_ENV) or "").strip()
    return [Path(part) for part in value.split(":") if part] if value else [ROOT]


def protection(*, pins_root: Path, queue_roots: Sequence[Path], intent_root: Path | None,
               binding_root: Path | None, now: float,
               process_checker: Callable[[Path], bool] | None = None) -> Callable[[Path], str | None]:
    """Why an intent's folder must stay (a reason), or None once every reader is done with it."""

    from .completed_replay_cache_retention import active_reference
    from .control_plane_storage_gc import _pinned_workspace
    from .control_plane_storage_pins import live_pinned_paths
    from .website_scene_workspace_queue import QueueInventoryUnavailable, queue_reference_text
    from .website_scene_workspace_references import build_reference_index
    from .website_scene_workspace_retention import RetentionContext, scene_queue_roots

    context = RetentionContext(storage_root=Path("/nonexistent"), pins_root=Path(pins_root), queue_roots=(),
                               intent_root=intent_root, binding_root=binding_root)
    index = build_reference_index(context, now=now)
    intents = {row["intent_id"]: row for row in index.intents} if index.readable else {}
    try:
        queue_text: str | None = queue_reference_text(scene_queue_roots(queue_roots, intent_root))
    except QueueInventoryUnavailable:
        queue_text = None
    pins = live_pinned_paths(Path(pins_root), now=lambda: now)
    in_use = process_checker or (lambda path: active_reference(path))

    def reason(directory: Path) -> str | None:
        if not index.readable:
            return "reference_index_unreadable"
        row = intents.get(directory.name)
        if row is None:
            return "intent_unknown"
        if row["finished"] is None:
            return "intent_open"
        if row["open_attempts"]:
            return "attempt_unsettled"
        if queue_text is None:
            return "queue_inventory_unreadable"
        if directory.name in queue_text:
            return "queue_referenced"
        if _pinned_workspace(directory, pins):
            return "pinned"
        try:
            if in_use(directory):
                return "in_use"
        except Exception:  # noqa: BLE001 - a process table that cannot be read proves nothing is idle
            return "in_use"
        return None

    return reason


def _work_root(path: str, *, expected: str, code: str) -> Any:
    """The evidence offload checks its roots' class; this root's is ``work`` (``control_plane_storage_roots``)."""

    from .control_plane_storage_roots import require_storage_class
    return require_storage_class(path, expected="work", code=code)


def offload_finished_intent_outputs(
    *, roots: Sequence[Path], pins_root: Path, queue_roots: Sequence[Path], intent_root: Path | None,
    binding_root: Path | None, apply: bool, enabled: bool, now: float,
    stream_publisher: Callable[..., Mapping[str, Any]] | None = None,
    process_checker: Callable[[Path], bool] | None = None, alert: str | None = None,
    classifier: Callable[..., Any] = _work_root, max_per_tick: int = MAX_PER_TICK,
) -> dict[str, Any] | None:
    """Plan every intent folder; archive at most ``max_per_tick`` when enabled. None when no root exists."""

    from .control_plane_evidence_offload import EXECUTE_ACK, apply_evidence_offload, build_evidence_offload_manifest

    present = [Path(root) for root in roots if Path(root).is_dir() and not Path(root).is_symlink()]
    if not present:
        return None
    checker = protection(pins_root=pins_root, queue_roots=queue_roots, intent_root=intent_root,
                         binding_root=binding_root, now=now, process_checker=process_checker)
    manifest = build_evidence_offload_manifest(
        evidence_roots=present, hot_window_seconds=IDLE_SECONDS, abandoned_after_seconds=IDLE_SECONDS,
        now=lambda: now, classifier=classifier, protection_checker=checker)
    report: dict[str, Any] = {"status": "dry_run", "enabled": bool(enabled), "roots": [str(root) for root in present],
                              "candidate_count": manifest["candidate_count"],
                              "candidate_bytes": manifest["candidate_bytes"],
                              "retained_by_reason": manifest["retained_by_reason"],
                              "candidates": [{key: row[key] for key in ("name", "size_bytes", "idle_seconds")}
                                             for row in manifest["candidates"]][:50]}
    if alert:
        report["alerts"] = [alert]
    if not (apply and enabled) or not manifest["candidates"]:
        return report
    bounded = {**manifest, "candidates": manifest["candidates"][:max_per_tick], "manifest_digest": ""}
    bounded["manifest_digest"] = canonical_digest(bounded, digest_field="manifest_digest")
    options = {"stream_publisher": stream_publisher} if stream_publisher is not None else {}
    result = apply_evidence_offload(bounded, ack=EXECUTE_ACK, now=lambda: now, protection_checker=checker, **options)
    report.update(status="applied", offloaded_count=result["offloaded_count"],
                  offloaded_bytes=result["offloaded_bytes"], offloaded=result["offloaded"],
                  skipped=result["skipped"], deferred_count=max(0, len(manifest["candidates"]) - max_per_tick))
    return report


__all__ = ["IDLE_SECONDS", "MAX_PER_TICK", "ROOT", "ROOTS_ENV", "SWITCH_ENV", "configured_roots",
           "offload_finished_intent_outputs", "protection", "setting"]
