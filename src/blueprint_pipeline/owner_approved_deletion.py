"""Delete exactly the host paths an owner approved: a plan first, then one receipt per path.

The owner approves a list of paths in a manifest, ``deploy/owner-approved-deletions/<id>.json``.
It ships with the release, so a session that reaches the host only through the operator door
can carry the approval there; the manifest names the owner's words and expires within 14 days.
Two oneshot units run this module and nothing starts them on a timer:

* ``plan`` (``blueprint-owner-approved-deletion-plan.service``) measures every listed path on
  the host, checks it and writes ``<report root>/<manifest id>/plan.json``. It deletes nothing.
* ``apply`` (``blueprint-owner-approved-deletion.service``) deletes only what a plan of the same
  manifest bytes, written within the last 24 hours, found eligible, and only while every check
  still passes and the path is unchanged since that plan. It writes
  ``<report root>/<manifest id>/receipts/<key>.json`` for every path it deletes, before it
  renames the path aside and again once it is gone.

A path is eligible only when it is one of two kinds, each under a fixed parent:

* ``task_evaluation_inputs_folder``: a direct child directory of
  ``/var/lib/blueprint/task-evaluation-inputs``. Its newest mtime (whole seconds, over the
  folder and everything in it), its regular-file bytes and its file count must equal what the
  manifest recorded. A folder the door could not list in full records how many names it could
  not see (``door_unlisted_entries``); its bytes and count may then only be larger.
* ``authority_ended_scene_workspace``: a website scene working copy under
  ``/var/lib/blueprint/pubsub-handoffs/<bucket>/scenes/``. Every capture in it must have ended
  for good because its authority ended (ledger ``terminal_authority_ended``) more than seven
  days ago, exactly as the manifest recorded it, and the workspace retention's own checks may
  hold it only for that reason or for recent activity. Retention keeps such a scene local
  (``authority_ended_capture_kept_local``) because archiving it would copy revoked derivatives;
  this deletes it instead. The capture's raw upload in Firebase Storage and any other copy are
  not touched.

Every path is also refused while a live storage pin names it, lies inside it or contains it, a
pending or processing queue message names it, a live process uses it, a systemd unit file
names it, or a sibling symlink points into it. The paths are opened one component at a time
without following a link, a folder holding another mount is refused, and the deletion itself
never follows a symlink.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

MANIFEST_SCHEMA = "owner_approved_deletion_manifest.v1"
PLAN_SCHEMA = "owner_approved_deletion_plan.v1"
RECEIPT_SCHEMA = "owner_approved_deletion_receipt.v1"
RESULT_SCHEMA = "owner_approved_deletion_result.v1"
APPLY_ACK = "delete-owner-approved-paths"
FOLDER = "task_evaluation_inputs_folder"
SCENE = "authority_ended_scene_workspace"
INPUTS_ROOT = Path("/var/lib/blueprint/task-evaluation-inputs")
HANDOFF_ROOT = Path("/var/lib/blueprint/pubsub-handoffs")
DEFAULT_MANIFEST_DIR = Path("/opt/blueprint/task-evaluation-control-plane/deploy/owner-approved-deletions")
DEFAULT_REPORT_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/storage-gc/owner-approved-deletions")
DEFAULT_PINS_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/storage-pins")
DEFAULT_INTENT_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/task-evaluation-scene-intents")
QUEUE_ROOTS_ENV = "BLUEPRINT_CONTROL_PLANE_GC_QUEUE_ROOTS"
UNIT_DIRECTORIES = (Path("/etc/systemd/system"), Path("/run/systemd/system"), Path("/run/systemd/transient"))
MAX_APPROVAL_SECONDS = 14 * 86400
MAX_PLAN_AGE_SECONDS = 24 * 3600
#: Pub/Sub message retention: once a capture ended longer ago, its message can no longer be
#: redelivered, so the listener cannot stage the deleted workspace again from that message.
REDELIVERY_WINDOW_SECONDS = 7 * 86400
MAX_ENTRIES = 512
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_UNIT_FILES = 20_000
MAX_UNIT_FILE_BYTES = 1024 * 1024
ASIDE_PREFIX = ".owner-approved-deletion-"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}\Z")
_MANIFEST_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,99}\Z")
_BUCKET = re.compile(r"[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]\Z")
_INSTANT = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
_PAYLOAD_DIGEST = re.compile(r"[0-9a-f]{64}\Z")  # the listener ledger's own format
_NAME_CHARS = "A-Za-z0-9_.-"
#: Retention reasons that still let an authority-ended scene be deleted. Recent activity is
#: allowed because the preparation status wake-up rewrites a small delivery file in it; leases,
#: locks, pins, queues, processes and open intents are not.
_SCENE_REASONS_ALLOWED = frozenset({"authority_ended_capture_kept_local", "recently_active"})


class OwnerApprovedDeletionError(ValueError):
    """A typed refusal of the whole manifest or plan; it never carries file contents."""


def _require(condition: Any, code: str) -> None:
    if not condition:
        raise OwnerApprovedDeletionError("owner_approved_deletion_" + code)


def _instant(value: Any, code: str) -> float:
    _require(isinstance(value, str) and _INSTANT.fullmatch(value), code)
    return _dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_dt.timezone.utc).timestamp()


def _count(value: Any) -> bool:
    return type(value) is int and value >= 0


# --- the manifest -----------------------------------------------------------------------------


def _folder_entry(entry: Mapping[str, Any], inputs_root: Path) -> dict[str, Any]:
    keys = {"class", "path", "newest_mtime_epoch", "apparent_bytes", "file_count", "door_unlisted_entries"}
    _require(set(entry) == keys, "entry_fields_invalid")
    path = Path(str(entry["path"]))
    _require(path.parent == inputs_root and _NAME.fullmatch(path.name) and not path.name.startswith(ASIDE_PREFIX)
             and str(path) == entry["path"], "entry_path_outside_allowlist")
    _require(all(_count(entry[key]) for key in keys - {"class", "path"}), "entry_identity_invalid")
    return dict(entry)


def _scene_entry(entry: Mapping[str, Any], handoff_root: Path) -> dict[str, Any]:
    _require(set(entry) == {"class", "path", "captures"}, "entry_fields_invalid")
    path = Path(str(entry["path"]))
    _require(str(path) == entry["path"] and path.parent.name == "scenes"
             and path.parent.parent.parent == handoff_root and _BUCKET.fullmatch(path.parent.parent.name)
             and _NAME.fullmatch(path.name), "entry_path_outside_allowlist")
    captures = entry["captures"]
    _require(isinstance(captures, list) and 1 <= len(captures) <= 64, "entry_captures_invalid")
    seen = set()
    for capture in captures:
        _require(isinstance(capture, dict)
                 and set(capture) == {"capture_id", "terminal_code", "terminal_at", "terminal_payload_sha256"}
                 and all(isinstance(capture[key], str) for key in capture)
                 and _NAME.fullmatch(capture["capture_id"]) and capture["capture_id"] not in seen
                 and _PAYLOAD_DIGEST.fullmatch(capture["terminal_payload_sha256"]), "entry_captures_invalid")
        seen.add(capture["capture_id"])
    return dict(entry)


def load_manifest(path: Path, *, inputs_root: Path = INPUTS_ROOT,
                  handoff_root: Path = HANDOFF_ROOT) -> tuple[dict[str, Any], str]:
    """The validated manifest and the SHA-256 of its exact bytes."""

    path = Path(path)
    _require(not path.is_symlink() and path.is_file(), "manifest_unreadable")
    raw = path.read_bytes()
    _require(0 < len(raw) <= MAX_MANIFEST_BYTES, "manifest_size_invalid")
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise OwnerApprovedDeletionError("owner_approved_deletion_manifest_unparsable") from None
    _require(isinstance(manifest, dict) and set(manifest) == {
        "schema_version", "manifest_id", "approved_by", "approval_evidence", "approved_at", "expires_at", "entries",
    } and manifest["schema_version"] == MANIFEST_SCHEMA, "manifest_fields_invalid")
    _require(isinstance(manifest["manifest_id"], str) and _MANIFEST_ID.fullmatch(manifest["manifest_id"])
             and path.name == manifest["manifest_id"] + ".json", "manifest_id_invalid")
    for key, limit in (("approved_by", 200), ("approval_evidence", 1000)):
        value = manifest[key]
        _require(isinstance(value, str) and 0 < len(value) <= limit and value.isprintable(), key + "_invalid")
    approved = _instant(manifest["approved_at"], "approved_at_invalid")
    expires = _instant(manifest["expires_at"], "expires_at_invalid")
    _require(approved < expires <= approved + MAX_APPROVAL_SECONDS, "approval_window_invalid")
    entries = manifest["entries"]
    _require(isinstance(entries, list) and 1 <= len(entries) <= MAX_ENTRIES, "entries_invalid")
    checked = []
    for entry in entries:
        _require(isinstance(entry, dict) and entry.get("class") in (FOLDER, SCENE), "entry_class_invalid")
        checked.append(_folder_entry(entry, inputs_root) if entry["class"] == FOLDER
                       else _scene_entry(entry, handoff_root))
    _require(len({entry["path"] for entry in checked}) == len(checked), "entries_duplicated")
    manifest["entries"] = checked
    return manifest, "sha256:" + hashlib.sha256(raw).hexdigest()


# --- measuring a path without following a link ----------------------------------------------------


def _open_directory(path: Path) -> int:
    """A descriptor for ``path``, opened from ``/`` one component at a time without following links."""

    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _measure(parent_fd: int, name: str) -> dict[str, Any] | str:
    """The folder's identity, or the reason it cannot be deleted as a folder."""

    try:
        info = os.lstat(name, dir_fd=parent_fd)
    except FileNotFoundError:
        return "absent"
    if not stat.S_ISDIR(info.st_mode):
        return "not_a_directory"
    if info.st_dev != os.fstat(parent_fd).st_dev:
        return "mount_point"
    try:
        top = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
    except OSError:
        return "unreadable"
    newest, apparent, allocated, files, directories, others = int(info.st_mtime), 0, 0, 0, 0, 0
    inodes: set[int] = set()
    try:
        opened = os.fstat(top)
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            return "changed_while_measured"
        for _, names, leaves, directory_fd in os.fwalk(".", dir_fd=top, follow_symlinks=False):
            for child in (*names, *leaves):
                child_info = os.lstat(child, dir_fd=directory_fd)
                newest = max(newest, int(child_info.st_mtime))
                if stat.S_ISDIR(child_info.st_mode):
                    if child_info.st_dev != info.st_dev:
                        return "mount_point_inside"
                    directories += 1
                elif stat.S_ISREG(child_info.st_mode):
                    apparent += child_info.st_size
                    files += 1
                    if child_info.st_ino not in inodes:
                        inodes.add(child_info.st_ino)
                        allocated += child_info.st_blocks * 512
                else:
                    others += 1
    except OSError:
        return "unreadable"
    finally:
        os.close(top)
    return {"dev": info.st_dev, "ino": info.st_ino, "newest_mtime_epoch": newest, "apparent_bytes": apparent,
            "allocated_bytes": allocated, "file_count": files, "directory_count": directories,
            "other_count": others}


_COMPARED = ("dev", "ino", "newest_mtime_epoch", "apparent_bytes", "file_count", "directory_count")


def _folder_matches_manifest(entry: Mapping[str, Any], identity: Mapping[str, Any]) -> bool:
    if identity["newest_mtime_epoch"] != entry["newest_mtime_epoch"]:
        return False
    if entry["door_unlisted_entries"]:
        return (identity["apparent_bytes"] >= entry["apparent_bytes"]
                and identity["file_count"] >= entry["file_count"])
    return identity["apparent_bytes"] == entry["apparent_bytes"] and identity["file_count"] == entry["file_count"]


# --- references ---------------------------------------------------------------------------------


def _named(text: str, needle: str) -> bool:
    return re.search(f"(?<![{_NAME_CHARS}]){re.escape(needle)}(?![{_NAME_CHARS}])", text) is not None


def unit_file_texts(directories: Sequence[Path] = UNIT_DIRECTORIES) -> list[tuple[str, str]]:
    """Every systemd unit and drop-in file's text, read without following links; unreadable raises."""

    texts: list[tuple[str, str]] = []
    for directory in directories:
        if not Path(directory).is_dir():
            continue
        for root, names, leaves in os.walk(directory, followlinks=False):
            for leaf in leaves:
                path = Path(root) / leaf
                if len(texts) >= MAX_UNIT_FILES:
                    raise OwnerApprovedDeletionError("owner_approved_deletion_unit_inventory_too_large")
                try:
                    info = os.lstat(path)
                    if not stat.S_ISREG(info.st_mode):
                        continue  # a link to a unit file elsewhere: its target is read where it lives
                    if info.st_size > MAX_UNIT_FILE_BYTES:
                        raise OwnerApprovedDeletionError("owner_approved_deletion_unit_inventory_unreadable")
                    texts.append((str(path), path.read_bytes().decode("utf-8", "replace")))
                except OSError:
                    raise OwnerApprovedDeletionError("owner_approved_deletion_unit_inventory_unreadable") from None
    return texts


def _pinned(path: Path, pinned: set[str]) -> bool:
    for raw in pinned:
        pin = Path(raw)
        if pin == path or path in pin.parents or pin in path.parents:
            return True
    return False


def _sibling_link_targets(parent: Path) -> list[Path]:
    targets = []
    with os.scandir(parent) as iterator:
        for item in iterator:
            if item.is_symlink():
                try:
                    targets.append(Path(os.path.normpath(parent / os.readlink(parent / item.name))))
                except OSError:
                    targets.append(parent)  # an unreadable link could point anywhere below the parent
    return targets


class References:
    """What can still need a path, read once per run. Anything unreadable counts as a reference."""

    def __init__(self, *, pins_root: Path, queue_roots: Sequence[Path], unit_texts: Sequence[tuple[str, str]],
                 process_index: Callable[[Path], bool], now: float) -> None:
        from .control_plane_storage_pins import live_pinned_paths
        from .website_scene_workspace_queue import QueueInventoryUnavailable, queue_reference_text

        self.pinned = live_pinned_paths(pins_root, now=lambda: now)
        try:
            self.queue_text: str | None = queue_reference_text(list(queue_roots))
        except QueueInventoryUnavailable:
            self.queue_text = None
        self.unit_texts = unit_texts
        self.process_index = process_index

    def reasons(self, path: Path) -> list[str]:
        reasons = []
        if _pinned(path, self.pinned):
            reasons.append("pinned")
        if self.queue_text is None:
            reasons.append("queue_inventory_unreadable")
        elif _named(self.queue_text, str(path)) or _named(self.queue_text, path.name):
            reasons.append("queue_referenced")
        units = sorted(name for name, text in self.unit_texts if _named(text, str(path)) or _named(text, path.name))
        reasons.extend(f"unit_referenced:{Path(name).name}" for name in units[:5])
        if self.process_index(path):
            reasons.append("in_use")
        return reasons


def build_references(*, pins_root: Path, queue_roots: Sequence[Path], intent_root: Path | None,
                     now: float, unit_directories: Sequence[Path] = UNIT_DIRECTORIES,
                     process_root: Path = Path("/proc")) -> References:
    from .completed_replay_cache_retention import process_reference_index
    from .website_scene_workspace_retention import scene_queue_roots

    try:
        index = process_reference_index(process_root=process_root, ignored_process_ids=(os.getpid(),))
    except ValueError:
        def index(_path: Path) -> bool:
            return True
    return References(pins_root=pins_root, queue_roots=scene_queue_roots(queue_roots, intent_root),
                      unit_texts=unit_file_texts(unit_directories), process_index=index, now=now)


# --- the per-kind checks ------------------------------------------------------------------------


def _check_folder(entry: Mapping[str, Any], references: References, *, link_targets: Sequence[Path],
                  ) -> tuple[list[str], dict[str, Any] | None]:
    path = Path(entry["path"])
    try:
        parent_fd = _open_directory(path.parent)
    except OSError:
        return ["parent_unsafe"], None
    try:
        identity = _measure(parent_fd, path.name)
    finally:
        os.close(parent_fd)
    if isinstance(identity, str):
        return [identity], None
    reasons = [] if _folder_matches_manifest(entry, identity) else ["changed_since_approval"]
    if any(target == path or path in target.parents for target in link_targets):
        reasons.append("symlink_target")
    reasons.extend(references.reasons(path))
    return reasons, identity


def _scene_context(path: Path, *, pins_root: Path, queue_roots: Sequence[Path], intent_root: Path | None) -> Any:
    from .website_scene_workspace_retention import RetentionContext, binding_root_for, scene_queue_roots

    storage_root = path.parent.parent.parent
    return RetentionContext(storage_root=storage_root, pins_root=pins_root,
                            queue_roots=scene_queue_roots(queue_roots, intent_root), intent_root=intent_root,
                            binding_root=binding_root_for(intent_root))


def _ledger_records(path: Path, capture_ids: Sequence[str]) -> list[dict[str, Any]] | None:
    from .website_scene_workspace_retention import LISTENER_FILES, TERMINAL_AUTHORITY_STATUS

    records = []
    for capture_id in capture_ids:
        ledger_path = path / "captures" / capture_id / LISTENER_FILES["ledger"]
        try:
            if ledger_path.is_symlink():
                return None
            ledger = json.loads(ledger_path.read_bytes())
        except (OSError, ValueError):
            return None
        if not isinstance(ledger, dict) or ledger.get("status") != TERMINAL_AUTHORITY_STATUS:
            return None
        records.append({key: ledger.get(key) for key in ("terminal_code", "terminal_at", "terminal_payload_sha256")}
                       | {"capture_id": capture_id})
    return records


def _check_scene(entry: Mapping[str, Any], references: References, *, pins_root: Path,
                 queue_roots: Sequence[Path], intent_root: Path | None, now: float,
                 ) -> tuple[list[str], dict[str, Any] | None]:
    from .website_scene_workspace_retention import _capture_ids, _evaluate

    path = Path(entry["path"])
    try:
        parent_fd = _open_directory(path.parent)
    except OSError:
        return ["parent_unsafe"], None
    try:
        identity = _measure(parent_fd, path.name)
    finally:
        os.close(parent_fd)
    if isinstance(identity, str):
        return [identity], None
    reasons: list[str] = []
    capture_ids = _capture_ids(path) or []
    expected = sorted(entry["captures"], key=lambda row: row["capture_id"])
    records = _ledger_records(path, capture_ids)
    if records is None:
        reasons.append("capture_authority_not_ended")
    elif sorted(records, key=lambda row: row["capture_id"]) != expected:
        reasons.append("changed_since_approval")
    for row in expected:
        try:
            ended = _dt.datetime.fromisoformat(row["terminal_at"]).timestamp()
        except ValueError:
            ended = now
        if now - ended < REDELIVERY_WINDOW_SECONDS:
            reasons.append(f"terminal_within_redelivery_window:{row['capture_id']}")
    context = _scene_context(path, pins_root=pins_root, queue_roots=queue_roots, intent_root=intent_root)
    evaluation = _evaluate(context=context, bucket=path.parent.parent.name, scene_id=path.name, scene=path, now=now,
                           index=None, process_checker=references.process_index)
    if "authority_ended_capture_kept_local" not in evaluation.reasons:
        reasons.append("retention_did_not_prove_authority_ended")
    reasons.extend(reason for reason in evaluation.reasons if reason not in _SCENE_REASONS_ALLOWED)
    reasons.extend(reason for reason in references.reasons(path) if reason not in reasons)
    identity["capture_ids"] = sorted(capture_ids)
    return sorted(set(reasons)), identity


# --- plan and apply -----------------------------------------------------------------------------


def _key(path: str) -> str:
    return hashlib.sha256(path.encode("utf-8")).hexdigest()[:32]


def _publish(path: Path, document: Mapping[str, Any]) -> None:
    """Write a secret-free JSON report readable by the door, atomically."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    for directory in (path.parent, *path.parent.parents):
        if directory == Path(directory.anchor):
            break
        _require(not directory.is_symlink(), "report_root_unsafe")
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fchmod(stream.fileno(), 0o644)
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        if path.is_symlink():
            return None
        value = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


class Settings:
    def __init__(self, *, pins_root: Path = DEFAULT_PINS_ROOT, queue_roots: Sequence[Path] = (),
                 intent_root: Path | None = DEFAULT_INTENT_ROOT, inputs_root: Path = INPUTS_ROOT,
                 handoff_root: Path = HANDOFF_ROOT, unit_directories: Sequence[Path] = UNIT_DIRECTORIES,
                 process_root: Path = Path("/proc")) -> None:
        self.pins_root, self.queue_roots, self.intent_root = Path(pins_root), tuple(map(Path, queue_roots)), intent_root
        self.inputs_root, self.handoff_root = Path(inputs_root), Path(handoff_root)
        self.unit_directories, self.process_root = tuple(unit_directories), Path(process_root)

    def references(self, now: float) -> References:
        return build_references(pins_root=self.pins_root, queue_roots=self.queue_roots, intent_root=self.intent_root,
                                now=now, unit_directories=self.unit_directories, process_root=self.process_root)


def _check(entry: Mapping[str, Any], references: References, settings: Settings, now: float,
           link_targets: Sequence[Path]) -> tuple[list[str], dict[str, Any] | None]:
    try:
        if entry["class"] == FOLDER:
            return _check_folder(entry, references, link_targets=link_targets)
        return _check_scene(entry, references, pins_root=settings.pins_root, queue_roots=settings.queue_roots,
                            intent_root=settings.intent_root, now=now)
    except Exception as error:  # noqa: BLE001 - a check that cannot finish keeps the path
        return [f"check_failed:{type(error).__name__}"], None


def plan_manifest(manifest_path: Path, *, report_root: Path, settings: Settings,
                  now: Callable[[], float] = time.time) -> dict[str, Any]:
    """Check every listed path and publish the plan; deletes nothing."""

    manifest, manifest_sha256 = load_manifest(manifest_path, inputs_root=settings.inputs_root,
                                              handoff_root=settings.handoff_root)
    observed_at = float(now())
    expired = observed_at > _instant(manifest["expires_at"], "expires_at_invalid")
    references = settings.references(observed_at)
    link_targets = _sibling_link_targets(settings.inputs_root) if settings.inputs_root.is_dir() else []
    rows = []
    for entry in manifest["entries"]:
        reasons, identity = _check(entry, references, settings, observed_at, link_targets)
        if expired:
            reasons = sorted({*reasons, "approval_expired"})
        status = "absent" if reasons == ["absent"] else "eligible" if not reasons else "skipped"
        rows.append({"path": entry["path"], "class": entry["class"], "status": status, "reasons": reasons,
                     "identity": identity})
    eligible = [row for row in rows if row["status"] == "eligible"]
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA, "manifest_id": manifest["manifest_id"], "manifest_sha256": manifest_sha256,
        "approved_by": manifest["approved_by"], "approval_evidence": manifest["approval_evidence"],
        "approved_at": manifest["approved_at"], "expires_at": manifest["expires_at"],
        "planned_at_epoch": observed_at, "mutations": 0, "entry_count": len(rows),
        "eligible_count": len(eligible), "eligible_apparent_bytes": sum(row["identity"]["apparent_bytes"]
                                                                       for row in eligible),
        "skipped_counts": _reason_counts(rows), "entries": rows,
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    _publish(Path(report_root) / manifest["manifest_id"] / "plan.json", plan)
    return plan


def _reason_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for reason in row["reasons"]:
            prefix = reason.split(":", 1)[0]
            counts[prefix] = counts.get(prefix, 0) + 1
    return dict(sorted(counts.items()))


def _receipt(manifest: Mapping[str, Any], plan: Mapping[str, Any], row: Mapping[str, Any], *, status: str,
             identity: Mapping[str, Any], aside: str, observed_at: float, **extra: Any) -> dict[str, Any]:
    receipt = {
        "schema_version": RECEIPT_SCHEMA, "status": status, "path": row["path"], "class": row["class"],
        "manifest_id": manifest["manifest_id"], "manifest_sha256": plan["manifest_sha256"],
        "plan_digest": plan["plan_digest"], "approved_by": manifest["approved_by"],
        "approval_evidence": manifest["approval_evidence"], "approved_at": manifest["approved_at"],
        "identity": dict(identity), "aside_name": aside, "observed_at_epoch": observed_at, **extra,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def _remove_aside(parent: Path, aside: str, identity: Mapping[str, Any]) -> str | None:
    """Remove the renamed-aside tree when it is still the measured directory; ``None`` when gone."""

    parent_fd = _open_directory(parent)
    try:
        try:
            info = os.lstat(aside, dir_fd=parent_fd)
        except FileNotFoundError:
            return None
        if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != (identity["dev"], identity["ino"]):
            return "aside_changed"
        shutil.rmtree(aside, dir_fd=parent_fd)
        try:
            os.lstat(aside, dir_fd=parent_fd)
        except FileNotFoundError:
            return None
        return "remove_incomplete"
    finally:
        os.close(parent_fd)


def _scene_locks(path: Path, capture_ids: Sequence[str]) -> Any:
    from .website_scene_workspace_retention import _CaptureLocks

    return _CaptureLocks(path, capture_ids)


def apply_manifest(manifest_path: Path, *, report_root: Path, settings: Settings, ack: str,
                   now: Callable[[], float] = time.time) -> dict[str, Any]:
    """Delete what the fresh plan found eligible, re-proving every check and the identity first."""

    _require(ack == APPLY_ACK, "apply_not_acknowledged")
    manifest, manifest_sha256 = load_manifest(manifest_path, inputs_root=settings.inputs_root,
                                              handoff_root=settings.handoff_root)
    directory = Path(report_root) / manifest["manifest_id"]
    plan = _read_json(directory / "plan.json")
    observed_at = float(now())
    _require(plan is not None and plan.get("schema_version") == PLAN_SCHEMA
             and plan.get("plan_digest") == canonical_digest(plan, digest_field="plan_digest"), "plan_missing")
    _require(plan["manifest_sha256"] == manifest_sha256 and plan["manifest_id"] == manifest["manifest_id"],
             "plan_for_other_manifest")
    planned_at = plan["planned_at_epoch"]
    _require(isinstance(planned_at, (int, float)) and 0 <= observed_at - planned_at <= MAX_PLAN_AGE_SECONDS,
             "plan_stale")
    _require(observed_at <= _instant(manifest["expires_at"], "expires_at_invalid"), "approval_expired")
    entries = {entry["path"]: entry for entry in manifest["entries"]}
    references = settings.references(observed_at)
    link_targets = _sibling_link_targets(settings.inputs_root) if settings.inputs_root.is_dir() else []
    outcomes = []
    for row in plan["entries"]:
        receipt_path = directory / "receipts" / f"{_key(row['path'])}.json"
        prior = _read_json(receipt_path)
        if prior is not None and prior.get("status") == "deleted":
            present = os.path.lexists(row["path"])
            outcomes.append({"path": row["path"], "outcome": "path_reappeared" if present else "already_deleted"})
            continue
        if prior is not None and prior.get("status") == "deleting":
            parent = Path(row["path"]).parent
            aside_present = os.path.lexists(parent / str(prior.get("aside_name")))
            if aside_present or not os.path.lexists(row["path"]):
                # A run stopped after the rename: finish exactly that tree (or record that it went).
                failure = _remove_aside(parent, prior["aside_name"], prior["identity"])
                if failure is None:
                    removed_at = float(now())
                    _publish(receipt_path, _receipt(manifest, plan, row, status="deleted",
                                                    identity=prior["identity"], aside=prior["aside_name"],
                                                    observed_at=removed_at, removed_at_epoch=removed_at,
                                                    recovered=True))
                outcomes.append({"path": row["path"], "outcome": "deleted" if failure is None else failure})
                continue
            # It stopped before the rename: the path is untouched, so check it from the start.
        if row["status"] != "eligible":
            outcomes.append({"path": row["path"], "outcome": "not_eligible_in_plan"})
            continue
        entry = entries[row["path"]]
        reasons, identity = _check(entry, references, settings, observed_at, link_targets)
        if not reasons and identity is not None and any(identity[key] != row["identity"][key]
                                                        for key in _COMPARED if entry["class"] == FOLDER):
            reasons = ["changed_since_plan"]
        if not reasons and identity is not None and (identity["dev"], identity["ino"]) != (
                row["identity"]["dev"], row["identity"]["ino"]):
            reasons = ["changed_since_plan"]
        if reasons:
            outcomes.append({"path": row["path"], "outcome": "skipped", "reasons": reasons})
            continue
        outcomes.append(_delete(manifest, plan, row, entry, identity, receipt_path, settings, now))
    result = {
        "schema_version": RESULT_SCHEMA, "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest_sha256, "plan_digest": plan["plan_digest"], "applied_at_epoch": observed_at,
        "deleted_count": sum(1 for row in outcomes if row["outcome"] == "deleted"),
        "deleted_apparent_bytes": sum(row.get("apparent_bytes", 0) for row in outcomes if row["outcome"] == "deleted"),
        "outcome_counts": _outcome_counts(outcomes), "outcomes": outcomes,
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    _publish(directory / f"apply-{int(observed_at)}.json", result)
    return result


def _outcome_counts(outcomes: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in outcomes:
        counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
    return dict(sorted(counts.items()))


def _delete(manifest: Mapping[str, Any], plan: Mapping[str, Any], row: Mapping[str, Any], entry: Mapping[str, Any],
            identity: dict[str, Any], receipt_path: Path, settings: Settings,
            now: Callable[[], float]) -> dict[str, Any]:
    path = Path(row["path"])
    aside = f"{ASIDE_PREFIX}{_key(row['path'])[:12]}-{secrets.token_hex(6)}"
    locks = _scene_locks(path, identity["capture_ids"]) if entry["class"] == SCENE else None
    try:
        if locks is not None:
            locks.__enter__()
            if locks.refusal:
                return {"path": row["path"], "outcome": "skipped", "reasons": [locks.refusal]}
            # Under the listener's own ledger locks, prove it again before anything moves.
            reasons, again = _check(entry, settings.references(float(now())), settings, float(now()), [])
            if reasons or again is None or (again["dev"], again["ino"]) != (identity["dev"], identity["ino"]):
                return {"path": row["path"], "outcome": "skipped", "reasons": reasons or ["changed_since_plan"]}
        _publish(receipt_path, _receipt(manifest, plan, row, status="deleting", identity=identity, aside=aside,
                                        observed_at=float(now())))
        parent_fd = _open_directory(path.parent)
        try:
            os.rename(path.name, aside, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            moved = os.lstat(aside, dir_fd=parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if locks is not None:
            locks.__exit__(None, None, None)
    if (moved.st_dev, moved.st_ino) != (identity["dev"], identity["ino"]):
        return {"path": row["path"], "outcome": "aside_changed"}
    failure = _remove_aside(path.parent, aside, identity)
    if failure is not None:
        return {"path": row["path"], "outcome": failure}
    removed_at = float(now())
    _publish(receipt_path, _receipt(manifest, plan, row, status="deleted", identity=identity, aside=aside,
                                    observed_at=removed_at, removed_at_epoch=removed_at))
    return {"path": row["path"], "outcome": "deleted", "apparent_bytes": identity["apparent_bytes"],
            "allocated_bytes": identity["allocated_bytes"]}


# --- command line -------------------------------------------------------------------------------


def manifests(directory: Path) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir() or directory.is_symlink():
        return []
    return sorted(path for path in directory.glob("*.json") if not path.name.startswith("."))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("plan", "apply"))
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--pins-root", type=Path, default=DEFAULT_PINS_ROOT)
    parser.add_argument("--intent-root", type=Path, default=DEFAULT_INTENT_ROOT)
    parser.add_argument("--ack", default="")
    args = parser.parse_args(argv)
    queue_roots = [Path(value) for value in os.environ.get(QUEUE_ROOTS_ENV, "").split(":") if value]
    if not queue_roots:
        print(json.dumps({"error": "owner_approved_deletion_queue_roots_unset"}), file=sys.stderr)
        return 2
    settings = Settings(pins_root=args.pins_root, queue_roots=queue_roots, intent_root=args.intent_root)
    summaries, failed = [], False
    for path in manifests(args.manifest_dir):
        try:
            if args.action == "plan":
                plan = plan_manifest(path, report_root=args.report_root, settings=settings)
                summaries.append({key: plan[key] for key in ("manifest_id", "plan_digest", "entry_count",
                                                             "eligible_count", "eligible_apparent_bytes",
                                                             "skipped_counts")})
            else:
                result = apply_manifest(path, report_root=args.report_root, settings=settings, ack=args.ack)
                summaries.append({key: result[key] for key in ("manifest_id", "result_digest", "deleted_count",
                                                               "deleted_apparent_bytes", "outcome_counts")})
        except OwnerApprovedDeletionError as error:
            failed = True
            summaries.append({"manifest": path.name, "error": str(error)})
    print(json.dumps({"action": args.action, "manifests": summaries}, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
