"""A finished scene intent's bulky preparation files go to the artifact store; every record stays in place.

``task-evaluation-inputs/completed-scene-preparation/<intent_id>/`` holds the scene factory's work for one scene
intent, one folder per attempt (``source-*``): its records (``factory.json``, ``submission.json``,
``publication.json``, the settlement's ``authority-ended-retirement.json`` and ownership reconciliation records)
and the materialized inputs (``materialized/``). On 2026-11-19 it was 21 GB in 17 intents going back to
2026-09-12, nothing reclaimed it ("report only"), and 99.7% of a measured intent (11.25 of 11.28 GB) was
materialized files of at least 1 MiB.

The owner approved on 2026-11-19 (relayed by the coordinating session): "Give `completed-scene-preparation` a
cleanup rule".

**Nothing a reader opens by path moves.** Spend accounting reopens settlement records here by the path a receipt
holds (``validated_cancellation`` -> ``task_evaluation_terminal_scene_attempt_settlement.validate_terminal_settlement``
-> the retirement and ownership records, the factory record and its submission request; reached from
``committed_spend_usd`` and every reader that settles or adopts an attempt: money). So the folder is never moved
whole. What stays: every record (``RECORD_SUFFIXES``), every file under 1 MiB, every file with another name, every
file whose name the search cannot prove unnamed, and every file that anything staying names by path, closed over
what those name in turn, where "anything staying" is every file left in the folder and every file of the intent's
own directory under the intent root (its attempts, settlements and receipts) (``task_evaluation_result_residue_scan``,
the search the slimming phase uses). The members are what is left: materialized bulk no record names.

**When.** The scene intent has finished (``website_scene_workspace_references``: completed, revoked, or expired past
the seven days an owner has to extend it) with no unsettled attempt; nothing in the folder changed for
``IDLE_SECONDS``; no pending or processing queue message names the intent, no live storage pin names it, lies inside
it or contains it, and no live process uses it. An unknown intent keeps everything, and unreadable intents keep
every folder.

**How.** The members are packed as one tar and published to the private artifact store with a full readback. The
record ``artifacts/result_delivery/remote_artifacts/scene-preparation-bulk.json`` (the archive, each member's
SHA-256) is written in the folder before anything is removed: withdrawal deletion already reads a run's remote
copies there, so a withdrawn site's archive is deleted with its folder. If the folder went (a withdrawal deleted it
during the upload), the upload is taken back at once. Then every check and member identity is proven again and only
then are the members removed. ``restore`` puts each member back after checking its bytes. At most
``MAX_PER_TICK`` folders a tick; a folder is archived once.
"""

from __future__ import annotations

import os
import stat
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
    from .website_scene_workspace_retention import RetentionContext, binding_root_for, scene_queue_roots

    # Registrations are read where the workspace phases read them: the reclaim unit names no binding root, and
    # without the intent root's sibling every folder read as ``reference_index_unreadable`` (2026-11-19: 17 kept).
    context = RetentionContext(storage_root=Path("/nonexistent"), pins_root=Path(pins_root), queue_roots=(),
                               intent_root=intent_root, binding_root=binding_root or binding_root_for(intent_root))
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
    """This root's class is ``work`` (``control_plane_storage_roots``)."""

    from .control_plane_storage_roots import require_storage_class
    return require_storage_class(path, expected=expected, code=code)


ARTIFACT_KIND = "scene-preparation-bulk"
RECORD_SCHEMA = "completed_scene_preparation_bulk.v1"
#: Where withdrawal deletion reads a run's remote copies (``website_capture_withdrawal_deletion._remote_artifact_uris``).
RECORD_PARTS = ("artifacts", "result_delivery", "remote_artifacts", "scene-preparation-bulk.json")


def record_path(folder: Path) -> Path:
    return Path(folder).joinpath(*RECORD_PARTS)


def _files(top: Path) -> list[tuple[str, os.stat_result]]:
    rows = []
    for directory, directories, names in os.walk(top):
        directories[:] = sorted(name for name in directories if not (Path(directory) / name).is_symlink())
        for name in sorted(names):
            path = Path(directory) / name
            rows.append((path.relative_to(top).as_posix(), os.lstat(path)))
    return rows


def plan_folder(folder: Path, intent_directory: Path | None) -> tuple[list[dict[str, Any]], dict[str, int], float]:
    """The folder's members (unnamed bulk), how many bulky files stay for each reason, and its newest mtime."""

    import posixpath
    from pathlib import PurePosixPath

    from .task_evaluation_result_residue_scan import name_supported, named_paths, suffix_index
    from .website_scene_workspace_slimming import MINIMUM_MEMBER_BYTES, RECORD_SUFFIXES, _tokens

    folder = Path(folder)
    rows = _files(folder)
    newest = max([folder.lstat().st_mtime, *(info.st_mtime for _relative, info in rows)])
    listed = {relative: info for relative, info in rows if stat.S_ISREG(info.st_mode)}
    bulky = {relative for relative, info in listed.items()
             if info.st_size >= MINIMUM_MEMBER_BYTES and PurePosixPath(relative).suffix.lower() not in RECORD_SUFFIXES}
    if not bulky:
        return [], {}, newest
    documents: list[tuple[Path, str]] = [(folder, relative) for relative in listed if relative not in bulky]
    if intent_directory is not None and Path(intent_directory).is_dir():
        documents += [(Path(intent_directory), relative) for relative, info in _files(Path(intent_directory))
                      if stat.S_ISREG(info.st_mode)]
    basenames, suffixes = {PurePosixPath(path).name for path in listed}, suffix_index(listed)
    named: set[str] = set()
    queue, searched = list(documents), set()
    while queue:
        root, relative = queue.pop()
        if (root, relative) in searched:
            continue
        searched.add((root, relative))
        # A file of the intent's own directory is searched for absolute names and path tails, never joined to its
        # own folders ("." has none): its relative names belong to the intent root, not to this folder.
        label = relative if root == folder else "."
        for tokens in _tokens(root, relative):
            for name in named_paths(tokens, label, folder.name, basenames, suffixes):
                path = posixpath.normpath(name)
                if path in listed and path not in named:
                    named.add(path)
                    queue.append((folder, path))
    members, kept = [], {}
    for relative in sorted(bulky):
        info = listed[relative]
        reason = ("hard_linked" if info.st_nlink != 1 else "name_unsupported" if not name_supported(relative)
                  else "named_by_record" if relative in named else None)
        if reason is not None:
            kept[reason] = kept.get(reason, 0) + 1
            continue
        members.append({"relative_path": relative, "size_bytes": int(info.st_size),
                        "identity": [int(info.st_dev), int(info.st_ino), int(info.st_mtime_ns)]})
    return members, kept, newest


def _archive_folder(folder: Path, members: Sequence[Mapping[str, Any]], *, checker: Callable[[Path], str | None],
                    publisher: Callable[..., Mapping[str, Any]], retract: Callable[[str], None],
                    now: float) -> dict[str, Any]:
    from .website_scene_workspace_slimming import _archive, _prune_empty_directories, _unchanged, _write_receipt

    archive, refusal = _archive(folder, members, filename=f"{folder.name}.tar", publisher=publisher,
                                artifact_kind=ARTIFACT_KIND)
    if refusal is not None:
        return {"name": folder.name, "status": "skipped", "reason": refusal}
    published = {key: archive[key] for key in ("uri", "digest", "size_bytes")}
    document = {"schema_version": RECORD_SCHEMA, "intent_id": folder.name, "uri": archive["uri"],
                "archive": archive, "archived_at_epoch": float(now), "evidence_deleted": False, "receipt_digest": ""}
    document["receipt_digest"] = canonical_digest(document, digest_field="receipt_digest")
    try:
        # Never re-creates the folder: a withdrawal may have deleted it during the upload.
        target = record_path(folder)
        for parent in reversed(target.parents[:len(RECORD_PARTS) - 1]):
            if not parent.is_dir():
                os.mkdir(parent, 0o750)
        _write_receipt(target, document)
    except OSError:
        retract(archive["uri"])  # nothing local names it: the upload is taken back
        return {"name": folder.name, "status": "skipped", "reason": "folder_changed_during_archive",
                "retracted": published}
    if checker(folder) is not None or not all(_unchanged(folder, row) for row in members):
        # The record names the copy, so withdrawal deletion still finds it; every local byte stays.
        return {"name": folder.name, "status": "skipped", "reason": "candidate_changed", "published_archive": published}
    removed = 0
    for row in members:
        if _unchanged(folder, row):
            (folder / row["relative_path"]).unlink()
            removed += row["size_bytes"]
    for child in sorted({Path(row["relative_path"]).parts[0] for row in members}):
        if (folder / child).is_dir():
            _prune_empty_directories(folder / child)
    return {"name": folder.name, "status": "archived", "freed_bytes": removed, "archive_bytes": archive["size_bytes"],
            "member_count": archive["member_count"]}


def restore_folder(*, folder: Path, materializer: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Put every member back from the folder's record, each byte checked. The record stays: it names the copy."""

    import tarfile
    import tempfile

    from .task_evaluation_configured_scene_object_store import materialize_configured_scene_artifact
    from .website_scene_workspace_records import _load_json
    from .website_scene_workspace_retention import _safe_relative, _sha256_file

    state, record, _ = _load_json(record_path(folder))
    if (state != "ok" or record is None or record.get("schema_version") != RECORD_SCHEMA
            or record.get("receipt_digest") != canonical_digest(record, digest_field="receipt_digest")):
        raise ValueError("completed_scene_preparation_record_invalid")
    archive = record["archive"]
    members = {_safe_relative(row["relative_path"]): row for row in archive["members"]}
    staging = Path(tempfile.mkdtemp(prefix=".bulk-restore-", dir=folder))
    placed = 0
    try:
        bundle = staging / "archive.tar"
        (materializer or materialize_configured_scene_artifact)(reference={
            "schema_version": "task_evaluation_scene_artifact_reference.v1", "status": "remote_verified",
            "artifact_kind": ARTIFACT_KIND, "uri": archive["uri"], "digest": archive["digest"],
            "size_bytes": archive["size_bytes"], "remote_identity_verified": True,
            "full_byte_service_account_readback_passed": True, "raw_secret_values_recorded": False,
        }, destination=bundle, maximum_size_bytes=int(archive["size_bytes"]))
        if _sha256_file(bundle) != archive["digest"]:
            raise ValueError("completed_scene_preparation_archive_mismatch")
        extracted = staging / "tree"
        extracted.mkdir()
        with tarfile.open(bundle) as source:
            source.extractall(extracted, filter="data")
        for name, row in sorted(members.items()):
            path = extracted / name
            if not path.is_file() or path.is_symlink() or _sha256_file(path) != row["sha256"]:
                raise ValueError("completed_scene_preparation_archive_mismatch")
            target = folder / name
            if os.path.lexists(target):
                if _sha256_file(target) != row["sha256"]:
                    raise ValueError("completed_scene_preparation_restore_conflict")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, target)
            placed += 1
    finally:
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
    return {"status": "restored", "folder": str(folder), "member_count": len(members), "placed_count": placed}


def offload_finished_intent_outputs(
    *, roots: Sequence[Path], pins_root: Path, queue_roots: Sequence[Path], intent_root: Path | None,
    binding_root: Path | None, apply: bool, enabled: bool, now: float,
    stream_publisher: Callable[..., Mapping[str, Any]] | None = None,
    process_checker: Callable[[Path], bool] | None = None, alert: str | None = None,
    classifier: Callable[..., Any] = _work_root, max_per_tick: int = MAX_PER_TICK,
    retract: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    """Plan every intent folder; archive the unnamed bulk of at most ``max_per_tick`` when enabled. None when no
    root exists."""

    from .task_evaluation_configured_scene_object_store import publish_configured_scene_stream

    present = [Path(root) for root in roots if Path(root).is_dir() and not Path(root).is_symlink()]
    if not present:
        return None
    for root in present:
        classifier(str(root), expected="work", code="completed_scene_preparation_root_class")
    checker = protection(pins_root=pins_root, queue_roots=queue_roots, intent_root=intent_root,
                         binding_root=binding_root, now=now, process_checker=process_checker)
    if retract is None:
        from .company_share_permission import retract_object as retract
    applying = bool(apply and enabled)
    report: dict[str, Any] = {"status": "applied" if applying else "dry_run", "enabled": bool(enabled),
                              "roots": [str(root) for root in present], "candidate_count": 0, "candidate_bytes": 0,
                              "retained_counts": {}, "kept_bulky_counts": {}, "offloaded_count": 0,
                              "offloaded_bytes": 0, "attempted_count": 0, "results": []}
    if alert:
        report["alerts"] = [alert]

    def retained(reason: str) -> None:
        report["retained_counts"][reason] = report["retained_counts"].get(reason, 0) + 1

    for root in present:
        for folder in sorted(path for path in root.iterdir() if path.name.startswith("scene-")):
            if folder.is_symlink() or not folder.is_dir():
                retained("unsafe")
                continue
            if os.path.lexists(record_path(folder)):
                retained("already_archived")
                continue
            reason = checker(folder)
            if reason is not None:
                retained(reason)
                continue
            try:
                members, kept, newest = plan_folder(
                    folder, Path(intent_root) / folder.name if intent_root is not None else None)
            except Exception as exc:  # noqa: BLE001 - a folder that cannot be searched keeps everything
                retained("reference_scan_failed")
                report["results"].append({"name": folder.name, "status": "error", "error": type(exc).__name__})
                continue
            for name, count in kept.items():
                report["kept_bulky_counts"][name] = report["kept_bulky_counts"].get(name, 0) + count
            if now - newest < IDLE_SECONDS:
                retained("recently_active")
                continue
            if not members:
                retained("nothing_unnamed")
                continue
            report["candidate_count"] += 1
            report["candidate_bytes"] += sum(row["size_bytes"] for row in members)
            if not applying or report["attempted_count"] >= max_per_tick:
                continue
            report["attempted_count"] += 1
            outcome = _archive_folder(folder, members, checker=checker, retract=retract, now=now,
                                      publisher=stream_publisher or publish_configured_scene_stream)
            if outcome["status"] == "archived":
                report["offloaded_count"] += 1
                report["offloaded_bytes"] += outcome["freed_bytes"]
            report["results"].append(outcome)
    report["deferred_count"] = max(0, report["candidate_count"] - report["attempted_count"]) if applying else 0
    report["results"] = report["results"][:50]
    return report


__all__ = ["ARTIFACT_KIND", "IDLE_SECONDS", "MAX_PER_TICK", "RECORD_PARTS", "ROOT", "ROOTS_ENV", "SWITCH_ENV",
           "configured_roots", "offload_finished_intent_outputs", "plan_folder", "protection", "record_path",
           "restore_folder", "setting"]
