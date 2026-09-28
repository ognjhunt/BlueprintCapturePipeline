"""Read-only, bounded census of unregistered control-plane scratch folders."""

from __future__ import annotations

import os
import json
import math
import stat
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .control_plane_disk_usage import DEFAULT_SURVEY_ALIASES, allocated_bytes, sanitize_public_survey
from .control_plane_lane_scratch import LaneScratchError, read_lane_scratch_folder
from .control_plane_storage_pins import PIN_KINDS, SCHEMA_VERSION as PIN_SCHEMA_VERSION, pin_status
from .control_plane_storage_roots import classify_path

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


__all__ = ["build_census", "DEFAULT_WORK_ROOT", "DEFAULT_INPUTS_ROOT"]
