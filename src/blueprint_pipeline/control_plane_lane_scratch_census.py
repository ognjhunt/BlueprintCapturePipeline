"""Read-only, bounded census of unregistered control-plane scratch folders."""

from __future__ import annotations

import os
import json
import math
import re
import stat
import sys
import time
import traceback
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .control_plane_disk_usage import DEFAULT_SURVEY_ALIASES, allocated_bytes, sanitize_public_survey
from .control_plane_lane_scratch import LaneScratchError, read_lane_scratch_folder
from .control_plane_storage_pins import PIN_KINDS, SCHEMA_VERSION as PIN_SCHEMA_VERSION, pin_status
from .control_plane_storage_roots import classify_path
from .control_plane_kernel_process import kernel_has_no_user_memory

DEFAULT_WORK_ROOT = Path("/mnt/blueprint-work")
DEFAULT_INPUTS_ROOT = Path("/var/lib/blueprint/task-evaluation-inputs")
DEFAULT_RELEASE_LINK = Path("/opt/blueprint/task-evaluation-control-plane")
DEFAULT_PINS_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/storage-pins")
_WORK_BOUND_CHILDREN = frozenset(
    Path(source).relative_to(DEFAULT_WORK_ROOT).parts[0]
    for source in DEFAULT_SURVEY_ALIASES if Path(source).is_relative_to(DEFAULT_WORK_ROOT)
)
_FAMILIES = (
    ("g1", "g1-lane", ("g1-",)),
    ("drawer", "drawer-lane", ("drawer-",)),
    ("arena", "arena-lane", ("arena-",)),
    ("content-agent", "content-agent-lane", ("content-agent-", "content_agents-")),
    ("gaussian-excision", "gaussian-excision-lane", ("gaussian-", "excision-")),
    ("scene", "scene-lane", ("scene-", "site-capture-")),
)
MAX_CANDIDATES = 10_000
MAX_ENTRIES = 200_000
MAX_SECONDS = 240.0
MAX_REFERENCE_BYTES = 20 * 1024 * 1024


def _family(name: str) -> tuple[str, str, str]:
    for family, guess, patterns in _FAMILIES:
        if name.startswith(patterns):
            return family, guess, "name_prefix"
    return "other", "unknown", "no_owner_evidence"


def _expired(deadline: float, errors: list[str]) -> bool:
    if time.monotonic() < deadline:
        return False
    errors.append("census_deadline_reached")
    return True


def _open_directory(path: Path) -> int:
    if not path.is_absolute():
        raise OSError("census_root_not_absolute")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _children(root: Path, label: str, errors: list[str], deadline: float) -> list[Path]:
    if _expired(deadline, errors):
        return []
    try:
        info = root.lstat()
    except FileNotFoundError:
        errors.append(f"{label}_root_missing")
        return []
    except OSError:
        errors.append(f"{label}_root_unreadable")
        return []
    if not stat.S_ISDIR(info.st_mode):
        errors.append(f"{label}_root_unsafe")
        return []
    root_fd = None
    try:
        root_fd = _open_directory(root)
        opened = os.fstat(root_fd)
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            errors.append(f"{label}_root_changed")
            return []
        with os.scandir(root_fd) as iterator:
            entries = []
            for entry in iterator:
                if _expired(deadline, errors):
                    break
                if len(entries) >= MAX_CANDIDATES:
                    errors.append(f"{label}_children_truncated")
                    break
                entries.append((entry.name, entry.is_dir(follow_symlinks=False),
                                entry.is_symlink()))
    except OSError:
        errors.append(f"{label}_root_unreadable")
        return []
    finally:
        if root_fd is not None:
            os.close(root_fd)
    paths = []
    for name, is_dir, is_link in sorted(entries):
        if is_dir:
            paths.append(root / name)
        elif is_link:
            errors.append(f"{label}_child_unsafe")
    return paths


def _candidates(work_root: Path, inputs_root: Path, errors: list[str], deadline: float) -> list[Path]:
    found: list[Path] = []
    for child in _children(work_root, "work", errors, deadline):
        if child.name not in _WORK_BOUND_CHILDREN and child.name != "lanes":
            found.append(child)
    for child in _children(inputs_root, "inputs", errors, deadline):
        if child.name == "lanes":
            continue
        canonical = DEFAULT_INPUTS_ROOT / child.name
        match = classify_path(str(canonical))
        if match is not None and match.path != str(DEFAULT_INPUTS_ROOT):
            continue
        found.append(child)
    for parent, label in ((work_root, "work"), (inputs_root, "inputs")):
        if _expired(deadline, errors):
            break
        if parent.is_symlink() or not parent.is_dir():
            continue
        lane_root = parent / "lanes"
        if not lane_root.exists() and not lane_root.is_symlink():
            continue
        for lane in _children(lane_root, f"{label}_lanes", errors, deadline):
            for folder in _children(lane, f"{label}_lane", errors, deadline):
                if _expired(deadline, errors):
                    break
                try:
                    read_lane_scratch_folder(folder, lane=lane.name, name=folder.name)
                except LaneScratchError:
                    found.append(folder)
    if len(found) > MAX_CANDIDATES:
        errors.append("candidate_count_truncated")
    return sorted(found, key=str)[:MAX_CANDIDATES]


def _scan_folder(path: Path, *, seen: set[tuple[int, int]], deadline: float,
                 budget: list[int], errors: list[str]) -> dict[str, Any]:
    try:
        root_fd = _open_directory(path)
    except OSError:
        errors.append("candidate_unreadable")
        return {"allocated_bytes": 0, "newest_mtime_epoch": None, "unreadable": 1, "shared_names": 0}
    device = os.fstat(root_fd).st_dev
    total = 0
    newest = 0.0
    unreadable = 0
    shared = 0
    stopped = False

    def count(info: os.stat_result) -> bool:
        nonlocal total, newest, shared, stopped
        if budget[0] >= MAX_ENTRIES or _expired(deadline, errors):
            errors.append("tree_scan_truncated")
            stopped = True
            return False
        budget[0] += 1
        if info.st_dev != device:
            errors.append("filesystem_boundary_skipped")
            return False
        newest = max(newest, float(info.st_mtime))
        inode = (info.st_dev, info.st_ino)
        if inode not in seen:
            seen.add(inode)
            total += allocated_bytes(info)
            return True
        else:
            shared += 1
            return False

    def walk(directory_fd: int, depth: int) -> None:
        nonlocal unreadable, stopped
        if not count(os.fstat(directory_fd)) or stopped:
            return
        if depth >= 64:
            errors.append("tree_scan_truncated")
            stopped = True
            return
        try:
            with os.scandir(directory_fd) as iterator:
                names = []
                for entry in iterator:
                    if _expired(deadline, errors) or len(names) + budget[0] >= MAX_ENTRIES:
                        errors.append("tree_scan_truncated")
                        stopped = True
                        break
                    names.append(entry.name)
        except OSError:
            unreadable += 1
            return
        for name in sorted(names):
            if stopped:
                break
            try:
                info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError:
                unreadable += 1
                continue
            if stat.S_ISDIR(info.st_mode):
                try:
                    child_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                       dir_fd=directory_fd)
                    opened = os.fstat(child_fd)
                    if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                        errors.append("tree_changed")
                    else:
                        walk(child_fd, depth + 1)
                except OSError:
                    unreadable += 1
                finally:
                    if "child_fd" in locals():
                        os.close(child_fd)
                        del child_fd
            else:
                count(info)

    try:
        walk(root_fd, 0)
    finally:
        os.close(root_fd)
    if unreadable:
        errors.append("tree_unreadable")
    return {"allocated_bytes": total, "newest_mtime_epoch": newest or None,
            "unreadable": unreadable, "shared_names": shared}


def _intersects(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _referenced(path: Path, references: Sequence[Path] | set[Path],
                deadline: float, errors: list[str]) -> bool:
    for reference in references:
        if _expired(deadline, errors):
            return False
        if _intersects(path, reference):
            return True
    return False


def _queue_references(paths: Sequence[Path], queue_roots: Sequence[Path], errors: list[str],
                      deadline: float) -> set[Path]:
    found: set[Path] = set()
    remaining = MAX_REFERENCE_BYTES
    visited = 0
    for root in queue_roots:
        if _expired(deadline, errors):
            break
        if root.is_symlink() or not root.is_dir():
            errors.append("queue_inventory_unavailable")
            continue
        for state in ("pending", "processing", "waiting_external", "awaiting_source_preparation",
                      "awaiting_capacity", "prepared", "blocked"):
            directory = root / state
            try:
                state_info = directory.lstat()
            except FileNotFoundError:
                continue
            except OSError:
                errors.append("queue_inventory_unavailable")
                continue
            if not stat.S_ISDIR(state_info.st_mode):
                errors.append("queue_inventory_unavailable")
                continue
            try:
                directory_fd = _open_directory(directory)
                directory_info = os.fstat(directory_fd)
                with os.scandir(directory_fd) as iterator:
                    names = []
                    for entry in iterator:
                        visited += 1
                        if _expired(deadline, errors) or visited > MAX_CANDIDATES:
                            errors.append("queue_inventory_truncated")
                            break
                        if entry.name.endswith(".json"):
                            names.append(entry.name)
            except OSError:
                errors.append("queue_inventory_unavailable")
                continue
            finally:
                if "directory_fd" in locals():
                    os.close(directory_fd)
                    del directory_fd
            for name in names:
                if _expired(deadline, errors):
                    break
                try:
                    directory_fd = _open_directory(directory)
                    reopened = os.fstat(directory_fd)
                    if ((reopened.st_dev, reopened.st_ino)
                            != (directory_info.st_dev, directory_info.st_ino)):
                        errors.append("queue_inventory_changed")
                        break
                    file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                      dir_fd=directory_fd)
                    with os.fdopen(file_fd, "rb") as stream:
                        info = os.fstat(stream.fileno())
                        if not stat.S_ISREG(info.st_mode):
                            errors.append("queue_inventory_unreadable")
                            continue
                        size = info.st_size
                        if size > 1024 * 1024 or size > remaining:
                            errors.append("queue_inventory_truncated")
                            continue
                        content = stream.read(size + 1)
                    if len(content) > size:
                        errors.append("queue_inventory_truncated")
                        continue
                except OSError:
                    errors.append("queue_inventory_unreadable")
                    continue
                finally:
                    if "directory_fd" in locals():
                        os.close(directory_fd)
                        del directory_fd
                remaining -= len(content)
                for path in paths:
                    if _expired(deadline, errors):
                        break
                    if str(path).encode() in content or path.name.encode() in content:
                        found.add(path)
    return found


def _process_references(paths: Sequence[Path], process_root: Path, errors: list[str],
                        deadline: float) -> set[Path]:
    if _expired(deadline, errors):
        return set()
    if process_root.is_symlink() or not process_root.is_dir():
        errors.append("process_inventory_unavailable")
        return set()
    found: set[Path] = set()
    remaining = MAX_REFERENCE_BYTES
    try:
        processes = []
        for process in process_root.iterdir():
            if _expired(deadline, errors):
                break
            if len(processes) >= MAX_CANDIDATES:
                errors.append("process_inventory_truncated")
                break
            processes.append(process)
    except OSError:
        errors.append("process_inventory_unreadable")
        return found
    for process in processes:
        if _expired(deadline, errors):
            break
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        def kernel_channel(name, cap):
            nonlocal remaining
            if _expired(deadline, errors):
                raise ValueError('process_inventory_truncated')
            with (process / name).open('rb') as stream:
                raw = stream.read(min(cap, remaining) + 1)
            if len(raw) > min(cap, remaining):
                raise ValueError('process_inventory_truncated')
            remaining -= len(raw)
            return raw
        for name in ("cmdline", "environ"):
            if _expired(deadline, errors):
                break
            try:
                size = (process / name).stat().st_size
                if size > 1024 * 1024 or size > remaining:
                    errors.append("process_inventory_truncated")
                    continue
                with (process / name).open("rb") as stream:
                    content = stream.read(min(1024 * 1024, remaining) + 1)
                if len(content) > min(1024 * 1024, remaining):
                    errors.append("process_inventory_truncated")
                    continue
            except FileNotFoundError:
                continue  # exited during the scan
            except ProcessLookupError:
                # Linux exposes ESRCH for environ on an actual kernel task
                # without an mm. A task flag alone does not qualify: observe
                # status, empty user channels and the same start identity.
                # Every available cwd/FD is still inspected below.
                if name == 'environ' and kernel_has_no_user_memory(kernel_channel, process.name):
                    continue
                errors.append("process_inventory_unreadable")
                continue
            except OSError:
                errors.append("process_inventory_unreadable")
                continue
            remaining -= len(content)
            for path in paths:
                if _expired(deadline, errors):
                    break
                if str(path).encode() in content:
                    found.add(path)
        try:
            descriptors = []
            for descriptor in (process / "fd").iterdir():
                if _expired(deadline, errors):
                    break
                if len(descriptors) >= MAX_CANDIDATES:
                    errors.append("process_inventory_truncated")
                    break
                descriptors.append(descriptor)
        except FileNotFoundError:
            continue
        except OSError:
            errors.append("process_inventory_unreadable")
            descriptors = []
        for link in (process / "cwd", *descriptors):
            if _expired(deadline, errors):
                break
            try:
                target = Path(os.readlink(link).removesuffix(" (deleted)"))
            except FileNotFoundError:
                continue
            except OSError:
                errors.append("process_inventory_unreadable")
                continue
            for path in paths:
                if _expired(deadline, errors):
                    break
                if target == path or path in target.parents:
                    found.add(path)
    return found


def _pin_references(pins_root: Path, observed: float, errors: list[str],
                    deadline: float) -> set[Path]:
    if _expired(deadline, errors):
        return set()
    if pins_root.is_symlink() or not pins_root.is_dir():
        errors.append("pin_inventory_unavailable")
        return set()
    paths: set[Path] = set()
    remaining = MAX_REFERENCE_BYTES
    visited = 0
    for kind in PIN_KINDS:
        if _expired(deadline, errors):
            break
        directory = pins_root / kind
        if not directory.exists():
            continue
        if directory.is_symlink() or not directory.is_dir():
            errors.append("pin_inventory_unreadable")
            continue
        try:
            with os.scandir(directory) as iterator:
                entries = []
                for entry in iterator:
                    if _expired(deadline, errors):
                        break
                    visited += 1
                    if visited > MAX_CANDIDATES:
                        errors.append("pin_inventory_truncated")
                        break
                    if entry.name.endswith(".json"):
                        entries.append(Path(entry.path))
        except OSError:
            errors.append("pin_inventory_unreadable")
            continue
        for path in entries:
            if _expired(deadline, errors):
                break
            try:
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode) or info.st_size > 1024 * 1024:
                    errors.append("pin_inventory_unreadable")
                    continue
                if info.st_size > remaining:
                    errors.append("pin_inventory_truncated")
                    break
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(descriptor, "rb") as stream:
                    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                        errors.append("pin_inventory_unreadable")
                        continue
                    payload = stream.read(info.st_size + 1)
                if len(payload) > info.st_size:
                    errors.append("pin_inventory_truncated")
                    continue
                remaining -= len(payload)
                pin = json.loads(payload)
                if (not isinstance(pin, dict) or pin.get("schema_version") != PIN_SCHEMA_VERSION
                        or pin.get("kind") != kind or not isinstance(pin.get("paths"), list)
                        or not isinstance(pin.get("owner_id"), str) or not pin["owner_id"]
                        or not all(isinstance(value, str) and Path(value).is_absolute()
                                   for value in pin["paths"])):
                    errors.append("pin_inventory_unreadable")
                    continue
                created = float(pin["created_at_epoch"])
                expires = float(pin["expires_at_epoch"])
                released = pin.get("released_at_epoch")
                if (not math.isfinite(created) or not math.isfinite(expires)
                        or created < 0 or expires <= created
                        or (released is not None and (not math.isfinite(float(released))
                                                      or float(released) < created))):
                    errors.append("pin_inventory_unreadable")
                    continue
                if pin_status(pin, now=observed) == "live":
                    paths.update(Path(value) for value in pin["paths"] if isinstance(value, str))
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError, KeyError):
                errors.append("pin_inventory_unreadable")
    return paths


def build_census(
    *, work_root: Path = DEFAULT_WORK_ROOT, inputs_root: Path = DEFAULT_INPUTS_ROOT,
    process_root: Path = Path("/proc"), pins_root: Path = DEFAULT_PINS_ROOT,
    queue_roots: Sequence[Path] | None = None, release_link: Path = DEFAULT_RELEASE_LINK,
    active_run_roots: Sequence[Path] | None = None, now: float | None = None,
    max_seconds: float = MAX_SECONDS,
) -> dict[str, Any]:
    """Inventory candidates and references; never create, remove or offload a folder."""

    observed = time.time() if now is None else now
    errors: list[str] = []
    if not 0 < max_seconds <= MAX_SECONDS:
        raise ValueError("max_seconds must be between 0 and 240")
    deadline = time.monotonic() + max_seconds
    paths = _candidates(work_root, inputs_root, errors, deadline)
    seen: set[tuple[int, int]] = set()
    budget = [0]
    if queue_roots is None:
        errors.append("queue_inventory_unavailable")
    queue_refs = _queue_references(paths, queue_roots or (), errors, deadline)
    process_refs = _process_references(paths, process_root, errors, deadline)
    pin_paths = _pin_references(pins_root, observed, errors, deadline)
    try:
        release_path = release_link.resolve(strict=True)
    except OSError:
        errors.append("live_release_unavailable")
        release_path = None
    if active_run_roots is None:
        errors.append("active_run_inventory_unavailable")
    active_paths = tuple(active_run_roots or ())
    rows = []
    for path in paths:
        if "tree_scan_truncated" in errors or _expired(deadline, errors):
            break
        measured = _scan_folder(path, seen=seen, deadline=deadline, budget=budget, errors=errors)
        family, guess, basis = _family(path.name)
        references = []
        if path in process_refs:
            references.append("process")
        if path in queue_refs:
            references.append("queue")
        if _referenced(path, pin_paths, deadline, errors):
            references.append("pin")
        if release_path is not None and _intersects(path, release_path):
            references.append("live_release")
        if _referenced(path, active_paths, deadline, errors):
            references.append("active_run")
        mtime = measured["newest_mtime_epoch"]
        rows.append({"path": str(path), "family": family, "owner_guess": guess,
                     "owner_guess_basis": basis, **measured,
                     "age_seconds": max(0.0, observed - mtime) if mtime is not None else None,
                     "references": references, "owner_decision": None, "approved_expiry": None})
    report = {"schema_version": "control_plane_lane_scratch_census.v1",
              "status": "incomplete" if errors else "complete", "observed_at_epoch": observed,
              "rows": rows, "candidate_count": len(paths), "entries_visited": budget[0],
              "unique_allocated_bytes": sum(row["allocated_bytes"] for row in rows),
              "scan_errors": sorted(set(errors)), "mutations": 0}
    return sanitize_public_survey(report)


# --- reference census of orphan scratch -----------------------------------------------------------
#
# ``build_census`` sizes every candidate tree, which on the host (1.6M unclassified files)
# always stops at MAX_ENTRIES. The reference census below only looks for references,
# kind by kind, and says for each kind whether it was checked. The capacity survey,
# which already walks these roots in bounded partitions, supplies the bytes. It reads,
# never writes, and its record has its own schema, so no retirement path accepts it.

REFERENCE_CENSUS_SCHEMA_VERSION = "control_plane_orphan_scratch_reference_census.v4"
REFERENCE_CENSUS_MAX_SECONDS = 180.0
REFERENCE_CENSUS_FILENAME = "orphan-scratch-census.json"
#: Said in every record and every projection of one.
EVIDENCE_STATEMENT = (
    "No reference found means none was found among the kinds checked, at the census time. "
    "It is evidence for an owner's decision, not a statement that deletion is safe, and a "
    "reference can appear after the census.")
ROOT_REACHABILITY_STATEMENT = (
    "Anything that uses a scratch root can reach every folder under it, and this census cannot "
    "rule that out. Folder family and age are the other evidence.")
DEFAULT_UNIT_DIRECTORIES = tuple(Path(path) for path in (
    "/etc/systemd/system", "/run/systemd/system", "/run/systemd/transient", "/run/systemd/generator",
    "/run/systemd/generator.late", "/usr/lib/systemd/system", "/lib/systemd/system",
    "/usr/local/lib/systemd/system"))
DEFAULT_ENV_DIRECTORIES = (Path("/etc/blueprint"), Path("/etc/blueprint-operator-door"),
                           Path("/etc/cron.d"), Path("/etc/crontab"), Path("/var/spool/cron"))
#: Holds credentials, never paths; it is not read, and the record says so.
ENV_EXCLUDED = (Path("/etc/blueprint/provider-secrets"),)
DEFAULT_CAPTURE_ROOTS = (Path("/var/lib/blueprint/pipeline-control-plane/capture-reconstruction-queue"),
                         Path("/var/lib/blueprint/pipeline-control-plane/capture-reconstruction-runs"))
DEFAULT_HANDOFF_ROOTS = (Path("/var/lib/blueprint/pubsub-handoffs"),)
DEFAULT_DOOR_REQUEST_ROOTS = (Path("/var/lib/blueprint-operator-door/requests"),)
DEFAULT_DOCKER_CONTAINERS = Path("/var/lib/docker/containers")
DEFAULT_SOURCE_ROOTS = (Path("/usr/lib/blueprint"),)
_SOURCE_SUFFIXES = frozenset({".py", ".sh", ".service", ".timer", ".json", ".toml", ".yaml", ".yml",
                              ".env", ".conf", ".cfg", ".txt"})
#: The release's runtime code is read; its tests, docs and third-party packages do not
#: run on the host, so they are not references.
_SKIPPED_SOURCE_DIRECTORIES = frozenset({".git", ".venv", "venv", "node_modules", "__pycache__",
                                         "site-packages", "tests", "docs"})
#: Every kind, in order. The census is complete only when every one is ``checked``.
REFERENCE_KINDS = (
    "queue", "active_run", "capture_reconstruction", "pubsub_handoff", "door_request",
    "process", "container_mount", "docker_container", "unit_file", "env_file", "release_source",
    "sibling_link", "pin", "live_release",
)
NOT_CHECKED_SOURCES = (
    # Documented gaps; none of them makes a kind "checked".
    "podman_or_bare_containerd_containers: no cheap stable record format; docker's are read",
    "processes_in_uninterruptible_sleep: a /proc read that blocks is not interrupted by the deadline",
    "symlink_aliases: a record naming a symlink to a folder under another leaf name is not matched;"
    " a live process using it is",
    "escaped_names: a name a record could escape (quote, backslash, non-ASCII) is not_proven",
)
_QUEUE_STATES = ("pending", "processing", "waiting_external", "awaiting_source_preparation",
                 "awaiting_capacity", "prepared", "blocked")
_RECORD_BYTES = 256 * 1024 * 1024  # streamed one file at a time, per kind
_RECORD_FILES = 50_000
_RECORD_ENTRIES = 400_000
_MAX_MOUNTINFO_BYTES = 1024 * 1024
_TOKEN = re.compile(rb"[A-Za-z0-9_.-]+")
_RUN_TIME_ONLY = ("sdk-inputs", "checkpoint", "cache", "wheelhouse", "toolchain", "runtime-source",
                  "prerequisites", "wheels")
_RELEASE_INPUT = ("deploy-source", "release", "build", "source-")
_JOB_DATA = ("capture", "scene", "run", "episode", "canary", "integration", "development", "controls",
             "rebuild", "cutout", "excision", "joint", "policy", "task")


def use_guess(name: str) -> tuple[str, bool]:
    """``(use, run_time_only)`` from the folder name alone: a guess, labelled as one.

    A run-time-only folder (an SDK input, a checkpoint or wheel cache) has no
    reference between runs, and deleting it means a rebuild or a paid re-download.
    """
    lowered = name.lower()
    if any(part in lowered for part in _RUN_TIME_ONLY):
        return "rebuildable_run_input_cache", True
    if any(part in lowered for part in _RELEASE_INPUT):
        return "release_or_build_input", False
    if any(part in lowered for part in _JOB_DATA):
        return "job_data", False
    return "unknown", False


_NAME_BYTES = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.+@~-")
_MAX_NAME_BYTES = 255


def _continues_name(byte: bytes) -> bool:
    return bool(byte) and byte[0] in _NAME_BYTES


def _reliably_matchable(name: str) -> bool:
    """A name a JSON or text record spells byte for byte (no escaping, plain ASCII)."""
    return all(32 <= ord(character) < 127 and character not in '"\\' for character in name)


class _PathMatcher:
    """Find candidate folders, and their ancestors, named in record text.

    A folder matches when its full path (or the path through a known bind-mount
    alias of its parent) appears and the next byte does not continue a file
    name, so ``/a/run`` is not ``/a/run2``; its bare name also matches as a whole
    token, which can only add references. An ancestor (other than ``/``) named on
    its own reaches every folder under it, and is reported as such.
    """

    def __init__(self, paths: Sequence[Path], aliases: dict[str, str]) -> None:
        self.tokens: dict[bytes, set[Path]] = {}
        self.children: dict[bytes, list[tuple[bytes, Path]]] = {}
        self.ancestors: dict[bytes, tuple[str, set[Path]]] = {}
        self.unmatchable = {path for path in paths if not _reliably_matchable(path.name)}

        def forms(directory: Path) -> set[str]:
            spelled = {str(directory)}
            try:
                spelled.add(os.path.realpath(directory))
            except OSError:
                pass
            for source, target in aliases.items():
                if str(directory) in (source, target):
                    spelled.update((source, target))
            return spelled

        for path in paths:
            self.tokens.setdefault(os.fsencode(path.name), set()).add(path)
            for form in forms(path.parent):
                self.children.setdefault(os.fsencode(form), []).append((os.fsencode(path.name), path))
            for ancestor in path.parents:
                if str(ancestor) == "/":
                    continue
                for form in forms(ancestor):
                    self.ancestors.setdefault(os.fsencode(form), (str(ancestor), set()))[1].add(path)

    def scan(self, content: bytes) -> tuple[set[Path], dict[str, set[Path]]]:
        found: set[Path] = set()
        for token in self.tokens.keys() & set(_TOKEN.findall(content)):
            found.update(self.tokens[token])
        for form, children in self.children.items():
            start = content.find(form)
            while start != -1:
                end = start + len(form)
                if content[end:end + 1] == b"/":
                    rest = content[end + 1:end + 2 + _MAX_NAME_BYTES]
                    for name, path in children:
                        if rest.startswith(name) and not _continues_name(rest[len(name):len(name) + 1]):
                            found.add(path)
                start = content.find(form, start + 1)
        ancestors: dict[str, set[Path]] = {}
        for form, (ancestor, beneath) in self.ancestors.items():
            start = content.find(form)
            while start != -1:
                end = start + len(form)
                after = content[end:end + 1]
                if after == b"/":
                    end += 1
                    after = content[end:end + 1]
                if after != b"/" and not _continues_name(after):
                    ancestors.setdefault(ancestor, set()).update(beneath)
                    break
                start = content.find(form, start + 1)
        return found, ancestors


def _record_references(matcher: _PathMatcher, roots: Sequence[Path], errors: list[str],
                       deadline: float, *, ancestors: dict[str, set[Path]], suffixes: frozenset[str] | None = None,
                       file_names: frozenset[str] | None = None, follow_files: bool = False,
                       skip_directories: frozenset[str] = frozenset(),
                       excluded: Sequence[Path] = ()) -> set[Path]:
    """Candidates named in any matching file below ``roots``; ancestors named go to ``ancestors``.

    Every cap reached or file unread is an error for the kind.
    """
    found: set[Path] = set()
    remaining, files, entries = _RECORD_BYTES, 0, 0
    for root in roots:
        if _expired(deadline, errors):
            break
        if not root.exists() and not root.is_symlink():
            continue  # nothing recorded there
        if root.is_symlink():
            errors.append("record_root_unsafe")
            continue

        def failed(_error: OSError) -> None:
            errors.append("record_unreadable")

        iterator = (iter([(str(root.parent), [], [root.name])]) if root.is_file()
                    else os.walk(root, followlinks=False, onerror=failed))
        for directory, subdirectories, leaves in iterator:
            here = Path(directory)
            subdirectories[:] = [name for name in subdirectories if name not in skip_directories
                                 and not any(_intersects(here / name, path) for path in excluded)]
            if any((here / name).is_symlink() for name in subdirectories):
                errors.append("record_directory_link_unread")  # its target is not read here
            entries += len(subdirectories) + len(leaves)
            if entries > _RECORD_ENTRIES:
                errors.append("record_inventory_truncated")
                return found
            for leaf in leaves:
                if _expired(deadline, errors):
                    return found
                path = here / leaf
                if (suffixes is not None and path.suffix not in suffixes) or (
                        file_names is not None and leaf not in file_names):
                    continue
                files += 1
                if files > _RECORD_FILES:
                    errors.append("record_inventory_truncated")
                    return found
                try:
                    info = path.stat() if follow_files else path.lstat()
                    if not stat.S_ISREG(info.st_mode):
                        if not stat.S_ISLNK(info.st_mode):
                            continue  # a socket or fifo holds no text
                        errors.append("record_link_unread")
                        continue
                    if info.st_size > 4 * 1024 * 1024 or info.st_size > remaining:
                        errors.append("record_inventory_truncated")
                        continue
                    with path.open("rb") as stream:
                        content = stream.read(info.st_size + 1)
                except FileNotFoundError:
                    continue
                except OSError:
                    errors.append("record_unreadable")
                    continue
                remaining -= len(content)
                named, reached = matcher.scan(content)
                found.update(named)
                for ancestor, beneath in reached.items():
                    ancestors.setdefault(ancestor, set()).update(beneath)
    return found


def _sibling_link_references(paths: Sequence[Path], errors: list[str], deadline: float) -> set[Path]:
    """Candidates a symlink beside them points into; an unreadable link holds every sibling."""
    found: set[Path] = set()
    for parent in sorted({path.parent for path in paths}, key=str):
        if _expired(deadline, errors):
            break
        targets: list[Path] = []
        try:
            with os.scandir(parent) as iterator:
                for entry in iterator:
                    if entry.is_symlink():
                        try:
                            targets.append(Path(os.path.normpath(parent / os.readlink(entry.path))))
                        except OSError:
                            targets.append(parent)
        except OSError:
            errors.append("sibling_link_inventory_unreadable")
            continue
        for path in paths:
            if path.parent == parent and any(_intersects(path, target) for target in targets):
                found.add(path)
    return found


def _decode_mount_field(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)


def _mountinfo(raw: bytes) -> list[tuple[str, str, str]]:
    """``(major:minor, root inside the filesystem, mount point)`` per line."""
    rows = []
    for line in raw.decode("utf-8", "surrogateescape").splitlines():
        fields = line.split(" ")
        if len(fields) < 5:
            raise ValueError("mountinfo_unparsed")
        rows.append((fields[2], _decode_mount_field(fields[3]), _decode_mount_field(fields[4])))
    return rows


def _read_small(path: Path, cap: int) -> bytes:
    with path.open("rb") as stream:
        raw = stream.read(cap + 1)
    if len(raw) > cap:
        raise ValueError("mountinfo_truncated")
    return raw


def _filesystem_location(path: Path, own_mounts: Sequence[tuple[str, str, str]]) -> tuple[str, PurePosixPath] | None:
    """The device and path inside its filesystem that ``path`` names in this namespace."""
    best = None
    for device, root, point in own_mounts:
        mount = PurePosixPath(point)
        if (PurePosixPath(str(path)) == mount or mount in PurePosixPath(str(path)).parents) and (
                best is None or len(point) > len(best[2])):
            best = (device, root, point)
    if best is None:
        return None
    relative = PurePosixPath(str(path)).relative_to(best[2])
    return best[0], PurePosixPath(best[1]) / relative


def _namespace_mount_references(paths: Sequence[Path], process_root: Path, errors: list[str],
                                deadline: float, *, scan_roots: Sequence[Path] = (),
                                reachable: dict[str, set[Path]] | None = None) -> set[Path]:
    """Candidates a process in another mount namespace (a container, a sandboxed unit) can use.

    A mount of the candidate, of a folder inside it or of a folder containing it
    holds it. Only a mount of the whole filesystem at the same mount point as
    here (a sandboxed unit's view of the host) does not; that process's cwd and
    descriptors are matched instead, as the process scan does for this namespace.
    """
    found: set[Path] = set()
    try:
        own_namespace = os.readlink(process_root / "self" / "ns" / "mnt")
        own_mounts = _mountinfo(_read_small(process_root / "self" / "mountinfo", _MAX_MOUNTINFO_BYTES))
    except (OSError, ValueError):
        errors.append("mount_namespace_inventory_unreadable")
        return found
    locations = {path: _filesystem_location(path, own_mounts) for path in paths}
    if any(location is None for location in locations.values()):
        errors.append("mount_namespace_inventory_unreadable")
        return found
    own_points = {(device, point) for device, root, point in own_mounts if root == "/"}
    root_locations = [(root, _filesystem_location(root, own_mounts)) for root in scan_roots]
    try:
        processes = sorted(entry for entry in process_root.iterdir() if entry.name.isdigit())
    except OSError:
        errors.append("mount_namespace_inventory_unreadable")
        return found
    if len(processes) > MAX_CANDIDATES:
        errors.append("mount_namespace_inventory_truncated")
    views: dict[str, dict[PurePosixPath, Path]] = {}
    for process in processes[:MAX_CANDIDATES]:
        if _expired(deadline, errors):
            break
        try:
            namespace = os.readlink(process / "ns" / "mnt")
            if namespace == own_namespace:
                continue
            if namespace not in views:
                view: dict[PurePosixPath, Path] = {}
                for device, root, point in _mountinfo(_read_small(process / "mountinfo", _MAX_MOUNTINFO_BYTES)):
                    mounted = PurePosixPath(root)
                    # A scan root (or above) mounted is root reachability, not a hold on
                    # any one folder: say so once and match this process's use instead.
                    whole_root = False
                    for scan_root, location in root_locations:
                        if location is not None and location[0] == device and (
                                mounted == location[1] or mounted in location[1].parents):
                            whole_root = True
                            if reachable is not None and not (root == "/" and (device, point) in own_points):
                                reachable.setdefault(str(scan_root), set()).update(
                                    path for path in paths if path == scan_root or scan_root in path.parents)
                    for path, location in locations.items():
                        if location is None or location[0] != device or not (
                                mounted == location[1] or location[1] in mounted.parents
                                or mounted in location[1].parents):
                            continue
                        if whole_root or (root == "/" and (device, point) in own_points):
                            view[PurePosixPath(point) / location[1].relative_to(mounted)] = path
                        else:
                            found.add(path)
                views[namespace] = view
            links = [process / "cwd", *(process / "fd").iterdir()]
        except FileNotFoundError:
            continue  # exited during the scan
        except (OSError, ValueError):
            errors.append("mount_namespace_inventory_unreadable")
            continue
        view = views[namespace]
        for link in links:
            try:
                target = PurePosixPath(os.readlink(link).removesuffix(" (deleted)"))
            except FileNotFoundError:
                continue
            except OSError:
                errors.append("mount_namespace_inventory_unreadable")
                continue
            for seen_as, path in view.items():
                if target == seen_as or seen_as in target.parents:
                    found.add(path)
    return found


def _top_level_mtime(path: Path) -> tuple[float | None, bool]:
    """Newest mtime of the folder and its direct children; never walks deeper."""
    try:
        newest = path.lstat().st_mtime
        with os.scandir(path) as iterator:
            for count, entry in enumerate(iterator):
                if count >= MAX_CANDIDATES:
                    return newest, False
                newest = max(newest, entry.stat(follow_symlinks=False).st_mtime)
    except OSError:
        return None, False
    return newest, True


def build_reference_census(
    *, work_root: Path = DEFAULT_WORK_ROOT, inputs_root: Path = DEFAULT_INPUTS_ROOT,
    process_root: Path = Path("/proc"), pins_root: Path = DEFAULT_PINS_ROOT,
    queue_roots: Sequence[Path] | None = None, release_link: Path = DEFAULT_RELEASE_LINK,
    active_run_roots: Sequence[Path] | None = None,
    unit_directories: Sequence[Path] = DEFAULT_UNIT_DIRECTORIES,
    env_directories: Sequence[Path] = DEFAULT_ENV_DIRECTORIES, env_excluded: Sequence[Path] = ENV_EXCLUDED,
    capture_roots: Sequence[Path] = DEFAULT_CAPTURE_ROOTS, handoff_roots: Sequence[Path] = DEFAULT_HANDOFF_ROOTS,
    door_request_roots: Sequence[Path] = DEFAULT_DOOR_REQUEST_ROOTS,
    docker_containers: Path = DEFAULT_DOCKER_CONTAINERS, source_roots: Sequence[Path] = DEFAULT_SOURCE_ROOTS,
    aliases: Any = DEFAULT_SURVEY_ALIASES, now: float | None = None, max_seconds: float = REFERENCE_CENSUS_MAX_SECONDS,
) -> dict[str, Any]:
    """Per orphan scratch folder, which kinds of reference were found; reads only.

    Every kind in ``REFERENCE_KINDS`` gets a status: ``checked``, ``failed`` (with
    errors) or ``not_checked`` (with why). The census is ``complete`` only when the
    candidate listing and every kind are ``checked``.
    """

    observed = time.time() if now is None else now
    if not 0 < max_seconds <= MAX_SECONDS:
        raise ValueError("max_seconds must be between 0 and 240")
    started = time.monotonic()
    deadline = started + max_seconds
    listing_errors: list[str] = []
    paths = _candidates(work_root, inputs_root, listing_errors, deadline)
    matcher = _PathMatcher(paths, dict(aliases))
    # The designated roots the census walks. A lane directory holds only that lane's
    # folders, so naming it is an intermediate ancestor, not root reachability.
    scan_roots = {work_root, inputs_root}

    def scan_root_or_above(ancestor: str) -> bool:
        return any(Path(ancestor) == root or Path(ancestor) in root.parents for root in scan_roots)

    ancestors: dict[str, dict[str, set[Path]]] = {}
    try:
        release_path: Path | None = release_link.resolve(strict=True)
    except OSError:
        release_path = None
    statuses: dict[str, dict[str, Any]] = {}
    found: dict[str, set[Path]] = {}

    def run(kind: str, check: Any, not_checked: str | None = None) -> None:
        if not_checked is not None:
            statuses[kind] = {"status": "not_checked", "reason": not_checked}
            found[kind] = set()
            return
        errors: list[str] = []
        found[kind] = check(errors)
        statuses[kind] = ({"status": "checked"} if not errors
                          else {"status": "failed", "errors": sorted(set(errors))[:10]})

    def records(roots: Sequence[Path], kind: str, **options: Any) -> Any:
        return lambda errors: _record_references(matcher, roots, errors, deadline,
                                                 ancestors=ancestors.setdefault(kind, {}), **options)

    queue_states = tuple(root / state for root in queue_roots or () for state in _QUEUE_STATES)
    run("queue", lambda errors: (_queue_references(paths, queue_roots or (), errors, deadline)
                                 | records(queue_states, "queue", suffixes=frozenset({".json"}))(errors)),
        None if queue_roots else "queue_roots_not_configured")
    run("active_run", records(tuple(active_run_roots or ()), "active_run", suffixes=frozenset({".json"})),
        None if active_run_roots else "active_run_roots_not_configured")
    run("capture_reconstruction", records(capture_roots, "capture_reconstruction", suffixes=frozenset({".json"})))
    run("pubsub_handoff", records(handoff_roots, "pubsub_handoff", suffixes=frozenset({".json"})))
    run("door_request", records(door_request_roots, "door_request", suffixes=frozenset({".json"})))
    run("process", lambda errors: _process_references(paths, process_root, errors, deadline))
    run("container_mount", lambda errors: _namespace_mount_references(
        paths, process_root, errors, deadline, scan_roots=sorted(scan_roots, key=str),
        reachable=ancestors.setdefault("container_mount", {})))
    run("docker_container", records((docker_containers,), "docker_container",
                                    file_names=frozenset({"config.v2.json", "hostconfig.json"})))
    run("unit_file", records(unit_directories, "unit_file", follow_files=True))
    run("env_file", records(env_directories, "env_file", follow_files=True, excluded=env_excluded))
    run("release_source", records(tuple(([release_path] if release_path else []) + list(source_roots)),
                                  "release_source", suffixes=_SOURCE_SUFFIXES, skip_directories=_SKIPPED_SOURCE_DIRECTORIES),
        None if release_path is not None else "live_release_unavailable")
    run("sibling_link", lambda errors: _sibling_link_references(paths, errors, deadline))
    pin_errors: list[str] = []
    pin_paths = _pin_references(pins_root, observed, pin_errors, deadline)
    found["pin"] = {path for path in paths if _referenced(path, pin_paths, deadline, pin_errors)}
    statuses["pin"] = {"status": "checked"} if not pin_errors else {"status": "failed",
                                                                    "errors": sorted(set(pin_errors))[:10]}
    run("live_release", lambda errors: {path for path in paths if _intersects(path, release_path)},
        None if release_path is not None else "live_release_unavailable")
    rows = []
    for path in paths:
        mtime, mtime_complete = (None, False) if _expired(deadline, listing_errors) else _top_level_mtime(path)
        use, run_time_only = use_guess(path.name)
        family, _guess, _basis = _family(path.name)
        not_proven = [{"reason": "ancestor_referenced", "kind": kind, "ancestor": ancestor}
                      for kind, reached in sorted(ancestors.items())
                      for ancestor, beneath in sorted(reached.items())
                      if path in beneath and not scan_root_or_above(ancestor)]
        if path in matcher.unmatchable:
            not_proven.append({"reason": "name_not_reliably_matchable"})
        rows.append({"path": str(path), "lane_family": family, "use_guess": use, "use_basis": "name_pattern",
                     "run_time_only": run_time_only,
                     "references": [kind for kind in REFERENCE_KINDS if path in found.get(kind, ())],
                     "not_proven": not_proven,
                     "top_level_mtime_epoch": mtime, "top_level_mtime_complete": mtime_complete})
    # Said once. A scan root (or a directory above one) named in a record is how the
    # system finds its workspace: it changes no row. An intermediate ancestor does.
    named = [(kind, ancestor, beneath) for kind, reached in sorted(ancestors.items())
             for ancestor, beneath in sorted(reached.items())]
    root_reachable_by = [{"kind": kind, "root": ancestor, "folders": len(beneath)}
                         for kind, ancestor, beneath in named if scan_root_or_above(ancestor)]
    ancestor_references = [{"kind": kind, "ancestor": ancestor, "folders": len(beneath)}
                           for kind, ancestor, beneath in named if not scan_root_or_above(ancestor)]
    complete = not listing_errors and all(statuses.get(kind, {}).get("status") == "checked"
                                          for kind in REFERENCE_KINDS)
    return {"schema_version": REFERENCE_CENSUS_SCHEMA_VERSION, "statement": EVIDENCE_STATEMENT,
            "status": "complete" if complete else "incomplete", "observed_at_epoch": observed,
            "duration_seconds": round(time.monotonic() - started, 3),
            "candidate_count": len(paths), "candidate_listing_errors": sorted(set(listing_errors)),
            "kinds": {kind: statuses[kind] for kind in REFERENCE_KINDS},
            "not_checked_sources": list(NOT_CHECKED_SOURCES), "ancestor_references": ancestor_references,
            "root_reachable_by": root_reachable_by, "root_reachability_statement": ROOT_REACHABILITY_STATEMENT,
            "rows": rows, "mutations": 0}


def publish_reference_census(directory: Path, *, pins_root: Path, queue_roots: Sequence[str | Path],
                             active_run_roots: Sequence[str | Path], now: float | None = None,
                             build: Any = None) -> bool:
    """Write the reference census beside the storage GC summary, after the GC tick.

    Called once a tick's report is written, so it never changes that report, its
    digest or the exit code, and no retirement path reads it. A failure withdraws
    the previous record: the capacity summary then shows the census missing, not stale.
    """
    path = directory / REFERENCE_CENSUS_FILENAME
    try:
        census = (build or build_reference_census)(
            pins_root=pins_root, queue_roots=[Path(root) for root in queue_roots] or None,
            active_run_roots=[Path(root) for root in active_run_roots] or None, now=now)
        temporary = directory / f".{REFERENCE_CENSUS_FILENAME}.{os.getpid()}"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(census, stream, sort_keys=True)
        os.replace(temporary, path)
    except Exception:  # noqa: BLE001 - the GC tick's own outcome stands
        traceback.print_exc(file=sys.stderr)
        try:
            os.unlink(path)
        except OSError:
            pass
        return False
    return True


__all__ = ["DEFAULT_INPUTS_ROOT", "DEFAULT_WORK_ROOT", "REFERENCE_CENSUS_SCHEMA_VERSION",
           "build_census", "build_reference_census", "publish_reference_census", "use_guess"]
