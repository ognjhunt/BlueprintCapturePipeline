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

A path is eligible only when it is one of three kinds, each under a fixed parent:

* ``task_evaluation_inputs_folder``: a direct child directory of
  ``/var/lib/blueprint/task-evaluation-inputs``. Its newest mtime (whole seconds, over the
  folder and everything in it), its regular-file bytes and its file count must equal what the
  manifest recorded. A folder the door could not list in full records how many names it could
  not see (``door_unlisted_entries``); its bytes and count may then only be larger, and its bytes
  at most ``MAX_UNLISTED_GROWTH`` (1.25) times the recorded bytes (``larger_than_approved``).
* ``task_evaluation_inputs_subfolder``: a direct child directory of a parent in
  ``SUBFOLDER_PARENTS``, and nothing deeper: any child of ``sam31-preparations``, any child of
  ``configured-controls``, and under ``stage-replays`` only the one replay folder the owner approved.
  The owner approved exactly three folders on 2026-11-12 (``sam31-preparations``, that one child of
  ``stage-replays``, ``configured-controls``); listing a folder's children rather than the folder is
  the planner's. That child is held by the SHA-256 digest of its name, not by its name: this module
  is released code and the released-code check reads it, so naming the child here would refuse it.
  Its name is in the manifest and in ``docs/CONTROL_PLANE_STORAGE.md``. It is checked exactly as a folder, through the same
  code: the same entry fields and identity, the same references, live state, released code and
  scan roots. In addition the parent is opened without following a link and must be a directory
  on the inputs root's own device that the mount table does not list (a bind mount of the same
  filesystem keeps its device number, so the table is read), no mount point may sit at or below
  the child, and no symlink that is a direct entry of the inputs root or of any of the three
  parents may point at the child, into it, at its parent or at the inputs root. The parent itself
  is never deleted, renamed or re-moded; the child is renamed aside inside the parent. The
  parent's name is named by released code (that is why the parent can never be listed as a
  folder); only the child's own name refuses the child as ``named_by_released_code``. The
  planner's decision of 2026-11-12: a live-state file, unit file, queue message or process that
  names only the parent's path does not refuse the child, otherwise nothing under a named root
  could ever be cleared; a live pin on the parent does (as a pin containing any path does), and so
  does a process using the child or anything in it. The loose JSON directly in each of these
  parents is read as live state, as the inputs root's is, and each parent is a scan root.
* ``authority_ended_scene_workspace``: a website scene working copy under
  ``/var/lib/blueprint/pubsub-handoffs/<bucket>/scenes/``. Every capture in it must have ended
  for good because its authority ended (ledger ``terminal_authority_ended``) more than seven
  days ago, exactly as the manifest recorded it, and the workspace retention's own checks may
  hold it only for that reason or for recent activity. Retention keeps such a scene local
  (``authority_ended_capture_kept_local``) because archiving it would copy revoked derivatives;
  this deletes it instead. The capture's raw upload in Firebase Storage and any other copy are
  not touched.

Every path is also refused while a live storage pin names it, lies inside it or contains it, a
queue message in a live state names it (``pending`` and ``processing`` of every queue root, and
every queue and state in ``control_plane_release_leases.LIVE_QUEUE_STATES``, such as SAM
preparation jobs ``waiting_external``), a live process uses it, a systemd unit file names it, or a
sibling symlink points into it. The paths are opened one component at a time
without following a link, a folder holding another mount is refused, and the deletion itself
never follows a symlink.

A path is also refused while the host's live state names it or a path under it
(``named_by_live_state``, with the naming files in the row's ``named_by``): every JSON and
``.env`` file under ``/etc/blueprint`` (robot catalog, controls bootstrap, configured-controls
intents, machinery files, launch profiles, scene progression config, env files), every scene
intent's files and the retained bindings they name, the public source bindings, the launch-profile
catalog, the loose JSON in the inputs root and the registries the control plane reads on a tick
(``LiveState``). On 2026-11-12 a clearance without this check deleted 112 folders those files still
named. A file inside a path this run deletes does not count; a file marked ended, revoked or
development-only still does. If a required root is missing or any file under the roots cannot be
read or parsed, every housekeeping path is ``reference_scan_incomplete``. A housekeeping folder
whose name released code (the active release's ``src``, ``deploy``, ``scripts``) or the door's
scripts use as a path component is refused (``named_by_released_code``), and so is one that is,
or holds, a root the scan reads (``live_state_scan_root``).

Privacy comes first: for a class in ``PRIVACY_REQUIRED_CLASSES`` (authority-ended scene
workspaces) live-state references and an incomplete scan never keep the path. The plan, the
receipt and the journal record them as ``references_left_pointing_at_nothing``.

The plan and receipts sit in a directory the unit's ``UMask=0077`` leaves root-only, so stdout
(the journal) lists each eligible path with its class and the bytes and files the plan measured and
the manifest recorded, each refused path with its reasons and naming files, each unreadable path
with its code, and in the apply each path as it is deleted: paths and numbers only, never a value
read from a file. Only manifests directly in the manifest directory are run; one moved to
``held/`` (as each is once applied) is never planned or applied.

Stated limits (not checked here): a symlink deeper inside a sibling folder is not scanned and is
left dangling; a process that maps a file and then closes its descriptor, or that runs in another
mount namespace, is not seen as using the path; two applies at once are not locked out, so the
operator runs one at a time, and starts the apply while the completed-replay-cache clean-up
(``blueprint-completed-replay-cache-gc``) is not running, since it walks ``stage-replays``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import posixpath
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
SUBFOLDER = "task_evaluation_inputs_subfolder"
SCENE = "authority_ended_scene_workspace"
#: The only parents whose direct child directories a manifest may list as ``SUBFOLDER``, each a direct
#: child of the inputs root, with the children allowed, as the SHA-256 hex digests of their names
#: (``None``: any). The owner approved, on 2026-11-12 about 21:05Z ("yes approve all 3"), exactly three
#: folders: ``sam31-preparations``, one replay folder of ``stage-replays`` and ``configured-controls``
#: (the last only if the first two free less than expected). So under ``stage-replays`` only that one
#: child is accepted. It is held by the digest of its name because this module is released code and the
#: released-code check reads it: its name here would refuse it as ``named_by_released_code``. The name
#: is in the manifest and in ``docs/CONTROL_PLANE_STORAGE.md`` (docs are not scanned). Listing the
#: children instead of the folders is the planner's decision; the parent itself is never listed.
SUBFOLDER_PARENTS: Mapping[str, frozenset[str] | None] = {
    "sam31-preparations": None,
    "stage-replays": frozenset({"12113f9d0de223f3c72895359692ba8fb1e08aad7f6180eab849f8792185df8c"}),
    "configured-controls": None,
}


def _name_digest(name: str) -> str:
    """How ``SUBFOLDER_PARENTS`` holds an allowed child: the SHA-256 hex digest of its name."""

    return hashlib.sha256(name.encode("utf-8")).hexdigest()


#: A folder the door could not list in full may have grown past what was recorded, but not by more.
MAX_UNLISTED_GROWTH = 1.25
#: Both folder kinds share the folder checks and the identity compared at plan and apply.
_FOLDER_CLASSES = frozenset({FOLDER, SUBFOLDER})
#: Classes whose deletion privacy requires: a leftover reference in live state never keeps the
#: data; the plan and receipt record it under ``references_left_pointing_at_nothing``.
PRIVACY_REQUIRED_CLASSES = frozenset({SCENE})
#: Housekeeping classes: a live-state reference, an incomplete scan or released code keeps them.
HOUSEKEEPING_CLASSES = frozenset({FOLDER, SUBFOLDER})
INPUTS_ROOT = Path("/var/lib/blueprint/task-evaluation-inputs")
HANDOFF_ROOT = Path("/var/lib/blueprint/pubsub-handoffs")
DEFAULT_MANIFEST_DIR = Path("/opt/blueprint/task-evaluation-control-plane/deploy/owner-approved-deletions")
DEFAULT_REPORT_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/storage-gc/owner-approved-deletions")
DEFAULT_PINS_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/storage-pins")
DEFAULT_INTENT_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/task-evaluation-scene-intents")
_CONTROL_PLANE = Path("/var/lib/blueprint/pipeline-control-plane")
#: The host configuration every unit reads (``EnvironmentFile=/etc/blueprint/*.env``, the scene
#: progression ``--config``, the controls autoprovision defaults, the launch-profile directory).
DEFAULT_CONFIG_ROOT = Path("/etc/blueprint")
#: Key files the units make inaccessible (``InaccessiblePaths=``); not configuration, never read.
CONFIG_EXCLUDED_NAMES = frozenset({"provider-secrets"})
#: ``public_source_binding_root`` of the scene progression config (task_evaluation_scene_progression).
DEFAULT_PUBLIC_SOURCE_BINDING_ROOT = INPUTS_ROOT / "public-source-bindings"
#: ``BLUEPRINT_TASK_EVALUATION_LAUNCH_PUBLIC_CATALOG_PATH`` (production_runtime_env_guard).
DEFAULT_LAUNCH_PROFILE_CATALOG = _CONTROL_PLANE / "task-evaluation-launch-profile-catalog.json"
#: Further registries read on a tick, scanned when present: website source bindings (scene
#: progression, website_scene_source_registry_paths), configured-scene activation intents
#: (``activation_intent_root``), beta policy screening (coordinator ``DEFAULT_QUEUE_ROOT``) and
#: standing launch authorizations (task_evaluation_release_retention).
DEFAULT_REGISTRY_ROOTS = (
    _CONTROL_PLANE / "website-source-bindings",
    _CONTROL_PLANE / "task-evaluation-scene-configuration-activation-intents",
    _CONTROL_PLANE / "beta-policy-screening",
    _CONTROL_PLANE / "standing-authorizations",
)
#: Released code: the active release's ``src``, ``deploy`` and ``scripts`` (required), and the
#: operator door's installed scripts (scanned when present). A direct child of the inputs root whose
#: name they use as a path component is refused (``named_by_released_code``).
DEFAULT_RELEASE_ROOT = Path("/opt/blueprint/task-evaluation-control-plane")
RELEASE_SUBDIRECTORIES = ("src", "deploy", "scripts")
DEFAULT_DOOR_ROOT = Path("/opt/blueprint/operator-door")
#: Never read as released code: the deletion manifests name every folder they delete.
RELEASE_EXCLUDED = frozenset({".git", "__pycache__", ".venv", "node_modules", "owner-approved-deletions"})
_CODE_SUFFIXES = frozenset({"", ".py", ".sh", ".service", ".timer", ".path", ".socket", ".target", ".conf",
                            ".json", ".toml", ".yaml", ".yml", ".txt", ".js", ".mjs", ".ts", ".in", ".env",
                            ".cfg", ".ini"})
#: Line-oriented text files under the configuration and registry roots, read for paths as text.
_TEXT_SUFFIXES = (".env", ".jsonl", ".yaml", ".yml", ".toml")
MAX_SCAN_DEPTH = 8
MAX_SCAN_FILES = 200_000
MAX_SCAN_FILE_BYTES = 64 * 1024 * 1024
QUEUE_ROOTS_ENV = "BLUEPRINT_CONTROL_PLANE_GC_QUEUE_ROOTS"
UNIT_DIRECTORIES = (Path("/etc/systemd/system"), Path("/run/systemd/system"), Path("/run/systemd/transient"))
#: This process's mount table: a bind mount of the same filesystem keeps its device number.
MOUNTINFO = Path("/proc/self/mountinfo")
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


def _folder_entry(entry: Mapping[str, Any], inputs_root: Path, *,
                  parents: Mapping[str, frozenset[str] | None] | None = None) -> dict[str, Any]:
    """A folder entry; with ``parents``, a direct child of one of those direct children of the inputs root
    (only the children whose name digest the parent lists, where it lists any)."""

    keys = {"class", "path", "newest_mtime_epoch", "apparent_bytes", "file_count", "door_unlisted_entries"}
    _require(set(entry) == keys, "entry_fields_invalid")
    path = Path(str(entry["path"]))
    placed = (path.parent == inputs_root if parents is None
              else path.parent.parent == inputs_root and path.parent.name in parents
              and (parents[path.parent.name] is None or _name_digest(path.name) in parents[path.parent.name]))
    _require(placed and _NAME.fullmatch(path.name) and not path.name.startswith(ASIDE_PREFIX)
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


#: Every class a manifest may list; each must be in exactly one of PRIVACY_REQUIRED_CLASSES and
#: HOUSEKEEPING_CLASSES (a test holds that).
ENTRY_VALIDATORS: dict[str, Callable[..., dict[str, Any]]] = {
    FOLDER: lambda entry, *, inputs_root, handoff_root: _folder_entry(entry, inputs_root),
    SUBFOLDER: lambda entry, *, inputs_root, handoff_root: _folder_entry(entry, inputs_root,
                                                                         parents=SUBFOLDER_PARENTS),
    SCENE: lambda entry, *, inputs_root, handoff_root: _scene_entry(entry, handoff_root),
}


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
        _require(isinstance(entry, dict) and entry.get("class") in ENTRY_VALIDATORS, "entry_class_invalid")
        checked.append(ENTRY_VALIDATORS[entry["class"]](entry, inputs_root=inputs_root, handoff_root=handoff_root))
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


def _folder_mismatch(entry: Mapping[str, Any], identity: Mapping[str, Any]) -> str | None:
    """Why the measured folder is not the one the owner approved, or ``None``."""

    if identity["newest_mtime_epoch"] != entry["newest_mtime_epoch"]:
        return "changed_since_approval"
    if entry["door_unlisted_entries"]:
        if (identity["apparent_bytes"] < entry["apparent_bytes"]
                or identity["file_count"] < entry["file_count"]):
            return "changed_since_approval"
        if identity["apparent_bytes"] > entry["apparent_bytes"] * MAX_UNLISTED_GROWTH:
            return "larger_than_approved"
        return None
    if identity["apparent_bytes"] == entry["apparent_bytes"] and identity["file_count"] == entry["file_count"]:
        return None
    return "changed_since_approval"


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
        from .control_plane_release_leases import LIVE_QUEUE_STATES
        from .control_plane_storage_pins import live_pinned_paths
        from .website_scene_workspace_queue import QueueInventoryUnavailable, queue_reference_text

        self.pinned = live_pinned_paths(pins_root, now=lambda: now)
        self.live_states = LIVE_QUEUE_STATES  # every live state the repository lists, for each such queue
        try:
            self.queue_text: str | None = queue_reference_text(list(queue_roots), live_states=self.live_states)
        except QueueInventoryUnavailable:
            self.queue_text = None
        self.queue_roots = tuple(queue_roots)
        self._tolerant: tuple[str, list[str]] | None = None
        self.unit_texts = unit_texts
        self.process_index = process_index

    def reasons(self, path: Path, unreadable: list[dict[str, str]] | None = None) -> list[str]:
        """Why ``path`` is kept. With ``unreadable`` (a privacy-required path), an unreadable queue file
        is recorded there and skipped instead of keeping the path."""

        from .website_scene_workspace_queue import tolerant_queue_reference_text

        reasons = []
        if _pinned(path, self.pinned):
            reasons.append("pinned")
        queue_text = self.queue_text
        if queue_text is None and unreadable is not None:
            if self._tolerant is None:
                self._tolerant = tolerant_queue_reference_text(list(self.queue_roots),
                                                               live_states=self.live_states)
            queue_text = self._tolerant[0]
            unreadable.extend({"path": name, "code": "queue_inventory_unreadable"} for name in self._tolerant[1])
        if queue_text is None:
            reasons.append("queue_inventory_unreadable")
        elif _named(queue_text, str(path)) or _named(queue_text, path.name):
            reasons.append("queue_referenced")
        units = sorted(name for name, text in self.unit_texts if _named(text, str(path)) or _named(text, path.name))
        reasons.extend(f"unit_referenced:{Path(name).name}" for name in units[:5])
        if self.process_index(path):
            reasons.append("in_use")
        return reasons


def build_references(*, pins_root: Path, queue_roots: Sequence[Path], intent_root: Path | None,
                     now: float, unit_directories: Sequence[Path] = UNIT_DIRECTORIES,
                     process_root: Path = Path("/proc"), live_queue_root: Path | None = None) -> References:
    """``live_queue_root``: where the queues of ``LIVE_QUEUE_STATES`` live; each is read as well."""

    from .completed_replay_cache_retention import process_reference_index
    from .control_plane_release_leases import LIVE_QUEUE_STATES
    from .website_scene_workspace_retention import scene_queue_roots

    try:
        index = process_reference_index(process_root=process_root, ignored_process_ids=(os.getpid(),))
    except ValueError:
        def index(_path: Path) -> bool:
            return True
    live = [Path(live_queue_root) / name for name in LIVE_QUEUE_STATES] if live_queue_root is not None else []
    roots = tuple(dict.fromkeys([*scene_queue_roots(queue_roots, intent_root), *live]))
    return References(pins_root=pins_root, queue_roots=roots, unit_texts=unit_file_texts(unit_directories),
                      process_index=index, now=now)


# --- live state: the configuration and registries that name paths ---------------------------------

#: An absolute path inside a string: at a token boundary, up to the first character a path here never
#: holds (``:`` separates PATH-like lists, ``=`` ends an ``--option=``). ``file:///x`` yields ``/x``.
_PATH_TOKEN = re.compile(r"(?<![A-Za-z0-9._@%+~-])/[A-Za-z0-9._@%+~/-]+")
#: A name released code uses as a path component: right after ``/`` or a quote, right before ``/``,
#: a quote, whitespace, ``,``, ``)`` or the end (``INPUTS / "toolchains"``, ``f"{_INPUTS}/x"``).
_CODE_COMPONENT = re.compile(r"(?<=[/\"'])[A-Za-z0-9][A-Za-z0-9._-]{0,254}(?=[/\"'\s,)]|\Z)")
NAMED_BY_LIVE_STATE = "named_by_live_state"
NAMED_BY_RELEASED_CODE = "named_by_released_code"
REFERENCE_SCAN_INCOMPLETE = "reference_scan_incomplete"
LIVE_STATE_SCAN_ROOT = "live_state_scan_root"
LEFT_POINTING_AT_NOTHING = "references_left_pointing_at_nothing"
#: Carried out of a privacy path's check: the unreadable queue files, registrations and intents skipped.
UNREADABLE_REFERENCES = "unreadable_references"


def _paths_in(text: str) -> set[str]:
    found = set()
    for match in _PATH_TOKEN.finditer(text):
        path = posixpath.normpath(match.group())
        if path != "/" and not path.startswith("//"):  # ``//host/...`` is a URL's host, not a host path
            found.add(path)
    return found


def _json_strings(value: Any) -> list[str]:
    """Every string in a JSON document, keys included."""

    strings, stack = [], [value]
    while stack:
        current = stack.pop()
        if isinstance(current, str):
            strings.append(current)
        elif isinstance(current, dict):
            strings.extend(key for key in current if isinstance(key, str))
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return strings


def _inside(path: str, roots: Sequence[str]) -> bool:
    return any(path == root or path.startswith(root.rstrip("/") + "/") for root in roots)


class _Gone(Exception):
    """A file that was listed and is no longer there: it names nothing."""


class LiveState:
    """Every path the host's configuration and registries name, with the files that name it.

    Built once per plan or apply. ``unreadable`` lists every root or file that could not be read
    completely; while it is not empty no housekeeping path is eligible (``reference_scan_incomplete``).
    ``code_names`` holds every name released code uses as a path component.
    """

    def __init__(self) -> None:
        self.named: dict[str, set[str]] = {}  # a named path and each of its ancestors -> naming files
        self.code_names: dict[str, set[str]] = {}
        self.unreadable: dict[str, str] = {}
        self.sources: list[dict[str, Any]] = []
        self.roots: list[str] = []
        self.file_count = 0

    def _add(self, source_file: str, paths: set[str]) -> None:
        for path in paths:
            current = path
            while current not in ("/", ""):
                self.named.setdefault(current, set()).add(source_file)
                current = posixpath.dirname(current)

    def naming_files(self, path: Path | str, excluded: Sequence[str] = ()) -> list[str]:
        """The files naming ``path`` or a path under it, leaving out files inside ``excluded`` or ``path``."""

        target = posixpath.normpath(str(path))
        skip = [*map(str, excluded), target]
        return sorted(name for name in self.named.get(target, ()) if not _inside(name, skip))

    def code_files(self, name: str) -> list[str]:
        return sorted(self.code_names.get(name, ()))

    def unreadable_rows(self) -> list[dict[str, str]]:
        return [{"path": path, "code": code} for path, code in sorted(self.unreadable.items())]

    def summary(self) -> dict[str, Any]:
        return {"status": "incomplete" if self.unreadable else "complete", "file_count": self.file_count,
                "sources": self.sources, "unreadable": self.unreadable_rows()}

    # --- reading ---------------------------------------------------------------------------------

    def _fetch(self, path: Path, *, retried: bool = False) -> bytes | None:
        name = str(path)
        try:
            info = os.stat(path)
            if not stat.S_ISREG(info.st_mode):
                self.unreadable[name] = "not_a_regular_file"
                return None
            if info.st_size > MAX_SCAN_FILE_BYTES:
                self.unreadable[name] = "too_large"
                return None
            return path.read_bytes()
        except FileNotFoundError:
            # Listed a moment ago: list its folder again once; a file that is really gone names nothing.
            try:
                still_listed = path.name in os.listdir(path.parent)
            except OSError:
                still_listed = False
            if still_listed and not retried:
                return self._fetch(path, retried=True)
            if still_listed:
                self.unreadable[name] = "unreadable"
                return None
            raise _Gone from None
        except PermissionError:
            self.unreadable[name] = "permission_denied"
        except OSError:
            self.unreadable[name] = "unreadable"
        return None

    def _read(self, path: Path, *, text: bool = False, code: bool = False) -> Any:
        """The parsed JSON (or the text), recording its paths; ``None`` (and an unreadable entry) otherwise."""

        name = str(path)
        if self.file_count >= MAX_SCAN_FILES:
            self.unreadable.setdefault(name, "scan_file_limit")
            return None
        self.file_count += 1
        try:
            raw = self._fetch(path)
        except _Gone:
            return None
        if raw is None:
            return None
        if code:
            for token in set(_CODE_COMPONENT.findall(raw.decode("utf-8", "replace"))):
                self.code_names.setdefault(token, set()).add(name)
            return None
        if text:
            decoded = raw.decode("utf-8", "replace")
            self._add(name, _paths_in(decoded))
            return decoded
        try:
            document = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.unreadable[name] = "not_json"
            return None
        except RecursionError:
            self.unreadable[name] = "too_deeply_nested"
            return None
        self._add(name, {found for string in _json_strings(document) for found in _paths_in(string)})
        return document

    def _wanted(self, name: str, *, code: bool) -> tuple[bool, bool]:
        """Whether to read a file, and whether as text."""

        if code:
            return Path(name).suffix in _CODE_SUFFIXES, False
        if name.endswith(".json"):
            return True, False
        textual = name == ".env" or name.endswith(_TEXT_SUFFIXES)
        return textual, textual

    def _walk(self, root: Path, *, kind: str, required: bool, code: bool = False,
              excluded_names: frozenset[str] = frozenset()) -> list[tuple[Path, Any]]:
        """Read every wanted file under ``root``, following linked folders once; a file root is read itself."""

        root = Path(root)
        source: dict[str, Any] = {"root": str(root), "kind": kind, "required": required}
        self.sources.append(source)
        self.roots.append(posixpath.normpath(str(root)))
        documents: list[tuple[Path, Any]] = []
        if not os.path.lexists(root):
            source["status"] = "absent"
            if required:
                self.unreadable[str(root)] = "required_root_missing"
            return documents
        if not root.is_dir():
            documents.append((root, self._read(root, code=code)))
            source.update(status="read", files=1)
            return documents
        before = self.file_count
        pending, visited = [(root, 0)], set()
        while pending:
            directory, depth = pending.pop()
            try:
                info = os.stat(directory)
                if (info.st_dev, info.st_ino) in visited:
                    continue  # a link back into a folder already read
                visited.add((info.st_dev, info.st_ino))
                with os.scandir(directory) as iterator:
                    items = sorted(iterator, key=lambda item: item.name)
            except FileNotFoundError:
                if directory != root:
                    continue  # removed since it was listed: it holds nothing now
                self.unreadable[str(directory)] = "unreadable"
                continue
            except PermissionError:
                self.unreadable[str(directory)] = "permission_denied"
                continue
            except OSError:
                self.unreadable[str(directory)] = "unreadable"
                continue
            for item in items:
                if item.name in excluded_names:
                    continue
                path = Path(item.path)
                wanted, text = self._wanted(item.name, code=code)
                try:
                    linked = item.is_symlink()
                    if linked and not code:
                        # A link names its target, dangling or not.
                        self._add(str(path), {posixpath.normpath(str(directory / os.readlink(path)))})
                    if item.is_dir(follow_symlinks=not code):
                        if depth + 1 >= MAX_SCAN_DEPTH:
                            self.unreadable[str(path)] = "scan_depth_limit"
                        else:
                            pending.append((path, depth + 1))
                        continue
                    if wanted and item.is_file(follow_symlinks=not code):
                        documents.append((path, self._read(path, text=text, code=code)))
                except FileNotFoundError:
                    continue  # removed since it was listed
                except OSError:
                    self.unreadable[str(path)] = "unreadable"
        source.update(status="read", files=self.file_count - before)
        return documents

    def _loose_json(self, root: Path, *, kind: str = "inputs_root_loose_json", required: bool = True,
                    scan_root: bool = False) -> None:
        source = {"root": str(root), "kind": kind, "required": required}
        self.sources.append(source)
        if scan_root:
            self.roots.append(posixpath.normpath(str(root)))
        before = self.file_count
        try:
            with os.scandir(root) as iterator:
                items = sorted(iterator, key=lambda item: item.name)
        except FileNotFoundError:
            source["status"] = "absent"
            if required:
                self.unreadable[str(root)] = "required_root_missing"
            return
        except OSError:
            self.unreadable[str(root)] = "unreadable"
            return
        for item in items:
            if item.name.endswith(".json") and item.is_file(follow_symlinks=False):
                self._read(Path(item.path))
        source.update(status="read", files=self.file_count - before)


def _retained_binding_paths(document: Any) -> set[str]:
    """``source_binding.path`` wherever an intent's files record a retained binding."""

    found, stack = set(), [document]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            binding = current.get("source_binding")
            if isinstance(binding, dict) and isinstance(binding.get("path"), str):
                found.add(posixpath.normpath(binding["path"]))
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return found


def scan_live_state(*, config_root: Path, intent_root: Path | None, binding_root: Path,
                    launch_profile_catalog: Path, inputs_root: Path,
                    registry_roots: Sequence[Path] = (), release_roots: Sequence[Path] = (),
                    door_roots: Sequence[Path] = ()) -> LiveState:
    """Read, once, every file of live configuration, registry state and released code that can name a path."""

    state = LiveState()
    state._walk(Path(config_root), kind="host_configuration", required=True, excluded_names=CONFIG_EXCLUDED_NAMES)
    if intent_root is not None:
        intents = state._walk(Path(intent_root), kind="scene_intents", required=True)
        retained = sorted({path for _, document in intents for path in _retained_binding_paths(document)})
        source = {"root": "source_binding.path of the scene intents", "kind": "retained_source_bindings",
                  "required": False, "files": 0, "status": "read"}
        state.sources.append(source)
        for path in retained:
            if os.path.isfile(path):  # a binding gone (or not a file) holds nothing; the intent still names it
                state._read(Path(path))
                source["files"] += 1
    state._walk(Path(binding_root), kind="public_source_bindings", required=True)
    state._walk(Path(launch_profile_catalog), kind="launch_profile_catalog", required=True)
    state._loose_json(Path(inputs_root))
    for parent in SUBFOLDER_PARENTS:  # a subfolder's siblings, as the inputs root's loose JSON is a folder's
        state._loose_json(Path(inputs_root) / parent, kind="subfolder_parent_loose_json", required=False,
                          scan_root=True)
    for root in registry_roots:
        state._walk(Path(root), kind="control_plane_registry", required=False)
    for root in release_roots:
        resolved = Path(os.path.realpath(root))  # the active release is a link to a per-commit tree
        for name in RELEASE_SUBDIRECTORIES:
            state._walk(resolved / name, kind="released_code", required=True, code=True,
                        excluded_names=RELEASE_EXCLUDED)
    for root in door_roots:
        state._walk(Path(os.path.realpath(root)), kind="operator_door_scripts", required=False, code=True,
                    excluded_names=RELEASE_EXCLUDED)
    return state


def live_state_reasons(entries: Sequence[Mapping[str, Any]], state: LiveState, deleted: Sequence[str],
                       ) -> dict[str, dict[str, Any]]:
    """Each entry's live-state verdict; ``deleted`` are the paths this run is about to remove.

    A file inside a path that will be deleted does not count, so the set shrinks until no path it
    still holds is named by a file inside another path it holds. A privacy-required path is never
    kept by these references: they are recorded as left pointing at nothing.
    """

    classes = {str(entry["path"]): entry["class"] for entry in entries}
    privacy = {path for path, kind in classes.items() if kind in PRIVACY_REQUIRED_CLASSES}

    def blocking(path: str, deleting: Sequence[str]) -> list[str]:
        if path in privacy:
            return []
        if state.unreadable:
            return [REFERENCE_SCAN_INCOMPLETE]
        reasons = []
        if state.naming_files(path, deleting):
            reasons.append(NAMED_BY_LIVE_STATE)
        if state.code_files(Path(path).name):
            reasons.append(NAMED_BY_RELEASED_CODE)
        if any(_inside(root, [path]) for root in state.roots):
            reasons.append(LIVE_STATE_SCAN_ROOT)
        return reasons

    deleting = [path for path in deleted if path in classes]
    while True:
        kept = [path for path in deleting if blocking(path, deleting)]
        if not kept:
            break
        deleting = [path for path in deleting if path not in kept]
    result = {}
    for path in classes:
        named_by = state.naming_files(path, deleting)
        verdict: dict[str, Any] = {"reasons": blocking(path, deleting), "named_by": named_by}
        if path not in privacy:
            verdict["named_by_released_code"] = state.code_files(Path(path).name)
        elif named_by or state.unreadable:
            verdict[LEFT_POINTING_AT_NOTHING] = {"named_by": named_by, "unreadable": state.unreadable_rows()}
        result[path] = verdict
    return result


# --- the per-kind checks ------------------------------------------------------------------------


def _mount_points(mountinfo: Path) -> frozenset[str] | None:
    """Every mount point in ``mountinfo`` (its fifth field, unescaped); ``None`` when it cannot be read."""

    try:
        table = Path(mountinfo).read_bytes()
        points = set()
        for line in table.splitlines():
            fields = line.split(b" ")
            if len(fields) < 5:
                raise ValueError("mountinfo_unparsed")
            point = re.sub(rb"\\([0-7]{3})", lambda match: bytes([int(match.group(1), 8) & 0xFF]), fields[4])
            points.add(posixpath.normpath(os.fsdecode(point)))
        return frozenset(points) if points else None
    except (OSError, ValueError):
        return None


def _subfolder_parent_unsafe(parent_fd: int, path: Path, mountinfo: Path) -> str | None:
    """Why a subfolder's parent or the child's mounts refuse it: the parent on another device than the
    inputs root or a mount point itself, or a mount at or below the child."""

    try:
        root_fd = _open_directory(path.parent.parent)
    except OSError:
        return "parent_unsafe"
    try:
        if os.fstat(root_fd).st_dev != os.fstat(parent_fd).st_dev:
            return "parent_unsafe"
    finally:
        os.close(root_fd)
    points = _mount_points(mountinfo)
    if points is None:
        return "mount_table_unavailable"
    if str(path.parent) in points:
        return "parent_unsafe"
    if str(path) in points:
        return "mount_point"
    if any(_inside(point, [str(path)]) for point in points):
        return "mount_point_inside"
    return None


def _check_folder(entry: Mapping[str, Any], references: References, *, link_targets: Sequence[Path],
                  mountinfo: Path = MOUNTINFO) -> tuple[list[str], dict[str, Any] | None]:
    path = Path(entry["path"])
    subfolder = entry["class"] == SUBFOLDER
    try:
        parent_fd = _open_directory(path.parent)
    except OSError:
        return ["parent_unsafe"], None
    try:
        identity = _measure(parent_fd, path.name)
        if subfolder and not isinstance(identity, str):
            identity = _subfolder_parent_unsafe(parent_fd, path, mountinfo) or identity
    finally:
        os.close(parent_fd)
    if isinstance(identity, str):
        return [identity], None
    mismatch = _folder_mismatch(entry, identity)
    reasons = [mismatch] if mismatch else []
    targets = list(link_targets)
    ancestors: tuple[Path, ...] = ()
    if subfolder:  # links directly in any of the three parents, and a link at the parent or the inputs root
        for parent in SUBFOLDER_PARENTS:
            if (path.parent.parent / parent).is_dir() and not (path.parent.parent / parent).is_symlink():
                targets.extend(_sibling_link_targets(path.parent.parent / parent))
        ancestors = (path.parent, path.parent.parent)
    if any(target == path or path in target.parents or target in ancestors for target in targets):
        reasons.append("symlink_target")
    reasons.extend(references.reasons(path))
    return reasons, identity


def _scene_context(path: Path, *, pins_root: Path, queue_roots: Sequence[Path], intent_root: Path | None,
                   unreadable: list[dict[str, str]] | None = None) -> Any:
    from .website_scene_workspace_retention import RetentionContext, binding_root_for, scene_queue_roots

    storage_root = path.parent.parent.parent
    return RetentionContext(storage_root=storage_root, pins_root=pins_root,
                            queue_roots=scene_queue_roots(queue_roots, intent_root), intent_root=intent_root,
                            binding_root=binding_root_for(intent_root), unreadable_references=unreadable)


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
    # A privacy-required workspace is not kept by someone else's unreadable queue file, registration or
    # intent: those are recorded (UNREADABLE_REFERENCES) and skipped. A pin, a process, a queue message or
    # an open intent naming it still keep it: those mean it is in use right now.
    unreadable: list[dict[str, str]] | None = [] if entry["class"] in PRIVACY_REQUIRED_CLASSES else None
    context = _scene_context(path, pins_root=pins_root, queue_roots=queue_roots, intent_root=intent_root,
                             unreadable=unreadable)
    evaluation = _evaluate(context=context, bucket=path.parent.parent.name, scene_id=path.name, scene=path, now=now,
                           index=None, process_checker=references.process_index)
    if "authority_ended_capture_kept_local" not in evaluation.reasons:
        reasons.append("retention_did_not_prove_authority_ended")
    reasons.extend(reason for reason in evaluation.reasons if reason not in _SCENE_REASONS_ALLOWED)
    reasons.extend(reason for reason in references.reasons(path, unreadable) if reason not in reasons)
    identity["capture_ids"] = sorted(capture_ids)
    if unreadable:
        identity[UNREADABLE_REFERENCES] = [{"path": name, "code": code}
                                           for name, code in sorted({(row["path"], row["code"]) for row in unreadable})]
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
                 process_root: Path = Path("/proc"), config_root: Path = DEFAULT_CONFIG_ROOT,
                 binding_root: Path = DEFAULT_PUBLIC_SOURCE_BINDING_ROOT,
                 launch_profile_catalog: Path = DEFAULT_LAUNCH_PROFILE_CATALOG,
                 registry_roots: Sequence[Path] = DEFAULT_REGISTRY_ROOTS,
                 release_roots: Sequence[Path] = (DEFAULT_RELEASE_ROOT,),
                 door_roots: Sequence[Path] = (DEFAULT_DOOR_ROOT,), mountinfo: Path = MOUNTINFO,
                 live_queue_root: Path = _CONTROL_PLANE) -> None:
        self.pins_root, self.queue_roots, self.intent_root = Path(pins_root), tuple(map(Path, queue_roots)), intent_root
        self.inputs_root, self.handoff_root = Path(inputs_root), Path(handoff_root)
        self.unit_directories, self.process_root = tuple(unit_directories), Path(process_root)
        self.config_root, self.binding_root = Path(config_root), Path(binding_root)
        self.launch_profile_catalog, self.registry_roots = Path(launch_profile_catalog), tuple(map(Path, registry_roots))
        self.release_roots, self.door_roots = tuple(map(Path, release_roots)), tuple(map(Path, door_roots))
        self.mountinfo = Path(mountinfo)
        self.live_queue_root = Path(live_queue_root)

    def references(self, now: float) -> References:
        return build_references(pins_root=self.pins_root, queue_roots=self.queue_roots, intent_root=self.intent_root,
                                now=now, unit_directories=self.unit_directories, process_root=self.process_root,
                                live_queue_root=self.live_queue_root)

    def live_state(self) -> LiveState:
        return scan_live_state(config_root=self.config_root, intent_root=self.intent_root,
                               binding_root=self.binding_root, launch_profile_catalog=self.launch_profile_catalog,
                               inputs_root=self.inputs_root, registry_roots=self.registry_roots,
                               release_roots=self.release_roots, door_roots=self.door_roots)


def _with_skipped_references(verdict: Mapping[str, Any], identity: dict[str, Any] | None) -> dict[str, Any]:
    """The verdict with a privacy path's skipped unreadable references moved out of its identity."""

    skipped = identity.pop(UNREADABLE_REFERENCES, []) if identity else []
    if not skipped:
        return dict(verdict)
    left = dict(verdict.get(LEFT_POINTING_AT_NOTHING) or {"named_by": verdict["named_by"], "unreadable": []})
    left["unreadable"] = [*left["unreadable"], *skipped]
    return {**verdict, LEFT_POINTING_AT_NOTHING: left}


def _check(entry: Mapping[str, Any], references: References, settings: Settings, now: float,
           link_targets: Sequence[Path]) -> tuple[list[str], dict[str, Any] | None]:
    try:
        if entry["class"] in _FOLDER_CLASSES:
            return _check_folder(entry, references, link_targets=link_targets, mountinfo=settings.mountinfo)
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
    live = settings.live_state()
    checked = [(entry, *_check(entry, references, settings, observed_at, link_targets))
               for entry in manifest["entries"]]
    verdicts = live_state_reasons([entry for entry, _, _ in checked], live,
                                  [entry["path"] for entry, reasons, _ in checked if not reasons and not expired])
    rows = []
    for entry, reasons, identity in checked:
        verdict = _with_skipped_references(verdicts[entry["path"]], identity)
        if reasons != ["absent"]:
            reasons = [*reasons, *verdict["reasons"]]
        if expired:
            reasons = sorted({*reasons, "approval_expired"})
        status = "absent" if reasons == ["absent"] else "eligible" if not reasons else "skipped"
        recorded = ({"recorded": {key: entry[key] for key in ("apparent_bytes", "file_count", "door_unlisted_entries")}}
                    if entry["class"] in _FOLDER_CLASSES else {})
        rows.append({"path": entry["path"], "class": entry["class"], "status": status, "reasons": reasons,
                     **{key: value for key, value in verdict.items() if key != "reasons"}, "identity": identity,
                     **recorded})
    eligible = [row for row in rows if row["status"] == "eligible"]
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA, "manifest_id": manifest["manifest_id"], "manifest_sha256": manifest_sha256,
        "approved_by": manifest["approved_by"], "approval_evidence": manifest["approval_evidence"],
        "approved_at": manifest["approved_at"], "expires_at": manifest["expires_at"],
        "planned_at_epoch": observed_at, "mutations": 0, "entry_count": len(rows),
        "eligible_count": len(eligible), "eligible_apparent_bytes": sum(row["identity"]["apparent_bytes"]
                                                                       for row in eligible),
        "skipped_counts": _reason_counts(rows), "reference_scan": live.summary(), "entries": rows,
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
    outcomes: list[dict[str, Any] | None] = []
    candidates: list[tuple[int, Mapping[str, Any], Mapping[str, Any], Path, list[str], dict[str, Any] | None]] = []
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
            if row["status"] == "skipped":
                _publish(receipt_path, _receipt(
                    manifest, plan, row, status="skipped", identity=row["identity"] or {}, aside="",
                    observed_at=observed_at, reasons=row["reasons"], named_by=row.get("named_by") or [],
                    named_by_released_code=row.get("named_by_released_code") or [],
                    reference_scan_unreadable=(plan.get("reference_scan") or {}).get("unreadable", [])
                    if REFERENCE_SCAN_INCOMPLETE in row["reasons"] else []))
            continue
        entry = entries[row["path"]]
        reasons, identity = _check(entry, references, settings, observed_at, link_targets)
        if not reasons and identity is not None and any(identity[key] != row["identity"][key]
                                                        for key in _COMPARED if entry["class"] in _FOLDER_CLASSES):
            reasons = ["changed_since_plan"]
        if not reasons and identity is not None and (identity["dev"], identity["ino"]) != (
                row["identity"]["dev"], row["identity"]["ino"]):
            reasons = ["changed_since_plan"]
        candidates.append((len(outcomes), row, entry, receipt_path, reasons, identity))
        outcomes.append(None)
    # Only what this run is about to delete, re-evaluated now, may name another path without counting.
    live = settings.live_state()
    verdicts = live_state_reasons([entries[row["path"]] for row in plan["entries"]], live,
                                  [row["path"] for _, row, _, _, reasons, _ in candidates if not reasons])
    for index, row, entry, receipt_path, reasons, identity in candidates:
        verdict = _with_skipped_references(verdicts[row["path"]], identity)
        if reasons != ["absent"]:
            reasons = [*reasons, *verdict["reasons"]]
        details = {key: value for key, value in verdict.items() if key != "reasons"}
        if verdict["reasons"]:
            outcomes[index] = {"path": row["path"], "outcome": "skipped", "reasons": reasons, **details}
            _publish(receipt_path, _receipt(manifest, plan, row, status="skipped", identity=identity or {},
                                            aside="", observed_at=observed_at, reasons=reasons, **details,
                                            reference_scan_unreadable=live.unreadable_rows()))
            continue
        if reasons:
            outcomes[index] = {"path": row["path"], "outcome": "skipped", "reasons": reasons}
            continue
        extra = {LEFT_POINTING_AT_NOTHING: verdict[LEFT_POINTING_AT_NOTHING]} if LEFT_POINTING_AT_NOTHING in verdict \
            else {}
        outcomes[index] = {**_delete(manifest, plan, row, entry, identity, receipt_path, settings, now, extra), **extra}
    result = {
        "schema_version": RESULT_SCHEMA, "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest_sha256, "plan_digest": plan["plan_digest"], "applied_at_epoch": observed_at,
        "deleted_count": sum(1 for row in outcomes if row["outcome"] == "deleted"),
        "deleted_apparent_bytes": sum(row.get("apparent_bytes", 0) for row in outcomes if row["outcome"] == "deleted"),
        "outcome_counts": _outcome_counts(outcomes), "outcomes": outcomes,
        # The apply-time scan, as the plan carries its own: the journal is the operator's only view of
        # which file to repair when the scan went incomplete after the plan.
        "reference_scan": live.summary(),
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
            now: Callable[[], float], extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
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
                                        observed_at=float(now()), **(extra or {})))
        # The journal names each path as it goes, before the rename: paths and numbers only.
        print(_journal_line(manifest["manifest_id"], "deleting", {
            "path": row["path"], "class": row["class"], "apparent_bytes": identity["apparent_bytes"],
            "file_count": identity["file_count"]}), flush=True)
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
                                    observed_at=removed_at, removed_at_epoch=removed_at, **(extra or {})))
    return {"path": row["path"], "outcome": "deleted", "apparent_bytes": identity["apparent_bytes"],
            "allocated_bytes": identity["allocated_bytes"]}


# --- command line -------------------------------------------------------------------------------


def manifests(directory: Path) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir() or directory.is_symlink():
        return []
    return sorted(path for path in directory.glob("*.json") if not path.name.startswith("."))


def journal_lines(rows: Sequence[Mapping[str, Any]], scan: Mapping[str, Any]) -> dict[str, Any]:
    """What lands in the journal: each eligible path (plan) or deleted path (apply) with its class and
    sizes, each refused path with its reasons and naming files, each unreadable path with its code, and
    the references a privacy deletion leaves. Paths and numbers only, never file contents; at most one
    line of each kind per manifest entry (``MAX_ENTRIES``).

    The plan and receipts sit under the unit's ``UMask=0077`` report directory, which the door cannot
    open, so the journal is where an operator sees which paths an apply would delete and why a path
    was kept.
    """

    eligible, deleted, refused, dangling = [], [], [], []
    for row in rows:
        reasons = row.get("reasons") or []
        if row.get("status") == "eligible":
            identity = row.get("identity") or {}
            recorded = row.get("recorded") or {}
            eligible.append({"path": row["path"], "class": row.get("class"),
                             "measured_apparent_bytes": identity.get("apparent_bytes"),
                             "measured_file_count": identity.get("file_count"),
                             "recorded_apparent_bytes": recorded.get("apparent_bytes"),
                             "recorded_file_count": recorded.get("file_count"),
                             "door_unlisted_entries": recorded.get("door_unlisted_entries")})
        if row.get("outcome") == "deleted":
            deleted.append({"path": row["path"], "apparent_bytes": row.get("apparent_bytes"),
                            "allocated_bytes": row.get("allocated_bytes")})
        if row.get("status", row.get("outcome")) in ("skipped",) and reasons:
            refused.append({"path": row["path"], "reasons": reasons, "named_by": row.get("named_by") or [],
                            "named_by_released_code": row.get("named_by_released_code") or []})
        if row.get(LEFT_POINTING_AT_NOTHING):
            dangling.append({"path": row["path"], **row[LEFT_POINTING_AT_NOTHING]})
    return {"reference_scan": scan.get("status"), "unreadable": list(scan.get("unreadable") or []),
            "eligible": eligible, "deleted": deleted, "refused": refused, LEFT_POINTING_AT_NOTHING: dangling}


def _journal_line(manifest_id: str, key: str, row: Mapping[str, Any]) -> str:
    """One journal line, always starting with ``{"manifest_id": ...``, its row's keys sorted."""

    return json.dumps({"manifest_id": manifest_id, key: dict(sorted(row.items()))})


def _journal(manifest_id: str, rows: Sequence[Mapping[str, Any]], scan: Mapping[str, Any]) -> dict[str, Any]:
    """Print one compact line per eligible, deleted, refused, unreadable or left-dangling path; return the counts."""

    lines = journal_lines(rows, scan)
    keys = ("eligible", "deleted", "refused", "unreadable", LEFT_POINTING_AT_NOTHING)
    for key in keys:
        for row in lines[key]:
            print(_journal_line(manifest_id, key, row))
    return {"reference_scan": lines["reference_scan"], **{key + "_count": len(lines[key]) for key in keys}}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("plan", "apply"))
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--pins-root", type=Path, default=DEFAULT_PINS_ROOT)
    parser.add_argument("--intent-root", type=Path, default=DEFAULT_INTENT_ROOT)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--binding-root", type=Path, default=DEFAULT_PUBLIC_SOURCE_BINDING_ROOT)
    parser.add_argument("--launch-profile-catalog", type=Path, default=DEFAULT_LAUNCH_PROFILE_CATALOG)
    parser.add_argument("--ack", default="")
    args = parser.parse_args(argv)
    queue_roots = [Path(value) for value in os.environ.get(QUEUE_ROOTS_ENV, "").split(":") if value]
    if not queue_roots:
        print(json.dumps({"error": "owner_approved_deletion_queue_roots_unset"}), file=sys.stderr)
        return 2
    settings = Settings(pins_root=args.pins_root, queue_roots=queue_roots, intent_root=args.intent_root,
                        config_root=args.config_root, binding_root=args.binding_root,
                        launch_profile_catalog=args.launch_profile_catalog)
    summaries, failed = [], False
    for path in manifests(args.manifest_dir):
        try:
            if args.action == "plan":
                plan = plan_manifest(path, report_root=args.report_root, settings=settings)
                summaries.append({**{key: plan[key] for key in ("manifest_id", "plan_digest", "entry_count",
                                                                "eligible_count", "eligible_apparent_bytes",
                                                                "skipped_counts")},
                                  **_journal(plan["manifest_id"], plan.get("entries") or [],
                                             plan.get("reference_scan") or {})})
            else:
                result = apply_manifest(path, report_root=args.report_root, settings=settings, ack=args.ack)
                summaries.append({**{key: result[key] for key in ("manifest_id", "result_digest", "deleted_count",
                                                                  "deleted_apparent_bytes", "outcome_counts")},
                                  **_journal(result["manifest_id"], result.get("outcomes") or [],
                                             result.get("reference_scan") or {})})
        except OwnerApprovedDeletionError as error:
            failed = True
            summaries.append({"manifest": path.name, "error": str(error)})
    print(json.dumps({"action": args.action, "manifests": summaries}, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
