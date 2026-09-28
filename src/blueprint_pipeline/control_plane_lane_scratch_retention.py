"""ADP-009D/day-28 bounded lane footprint observation, without cleanup authority.

Only explicitly configured lane parents are observed. Footprints include the
lease and unique regular inodes, exclude directory allocation, and do not
predict freed space. Stable observations are not consumer lifetime fences.
"""
from __future__ import annotations

import errno
import json
import math
import os
import stat
import time
from collections.abc import Callable
from typing import Any

from .control_plane_lane_scratch import LEASE_FILE, SCHEMA_VERSION, _ID, _lease_fields_valid
from .control_plane_storage_pin_observation import StoragePinObservationError, observe_storage_pins
from .decision_evidence_contracts import canonical_digest

MAX_ROOTS, MAX_LANES, MAX_FOLDERS = 2, 100, 1000
MAX_ENTRIES, MAX_DEPTH = 10_000, 64
MAX_LEASE_BYTES, MAX_TOTAL_LEASE_BYTES = 8192, 8 * 1024 * 1024
MAX_OUTPUT_BYTES, MAX_BLOCKERS, MAX_COMPARISONS = 16 * 1024 * 1024, 32, 200_000
MAX_PATH_BYTES, MAX_PATH_COMPONENTS = 4096, 64
_ALLOWED_ROOTS = frozenset({"/mnt/blueprint-work/lanes", "/var/lib/blueprint/task-evaluation-inputs/lanes"})
_DIR = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
_FIELDS = {"schema_version", "lane", "name", "owner", "reason", "class_intent", "cleanup",
           "created_at_epoch", "expires_at_epoch", "released_at_epoch", "size_budget_bytes", "lease_digest"}
_GATES = ("owner_approval_missing", "consumer_fence_missing", "evidence_policy_unproven", "restore_unproven")
_PRIORITY = ("changed_lease", "measurement_incomplete", "references_unknown", "referenced", "live_lease",
             "owner_review", *_GATES, "hardlinked_payload")
_RESOURCE = {"lane_deadline_exceeded", "lane_clock_invalid", "lane_output_limit", "lane_output_invalid"}


class LaneScratchRetentionError(ValueError):
    """Invalid API arguments; text never includes input values."""


class _Blocked(Exception):
    def __init__(self, code: str):
        self.code = code


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise _Blocked(code)


def _finite(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _path(value: Any) -> str:
    try:
        valid = (isinstance(value, str) and len(value) <= MAX_PATH_BYTES
                 and len(value.encode("utf-8")) <= MAX_PATH_BYTES and value.startswith("/")
                 and not value.startswith("//") and not any(ord(c) < 32 or ord(c) == 127 or c == "\\" for c in value))
        parts = value[1:].split("/") if value != "/" else []
        valid = valid and len(parts) <= MAX_PATH_COMPONENTS and all(p not in {"", ".", ".."} for p in parts)
    except (UnicodeError, TypeError, AttributeError):
        valid = False
    if not valid:
        raise LaneScratchRetentionError("lane_parameters_invalid")
    return value


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_blocks,
            value.st_nlink, value.st_mtime_ns, value.st_ctime_ns)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate")
        value[key] = item
    return value


def _number(text: str) -> int | float:
    value = float(text) if any(c in text for c in ".eE") else int(text)
    if not _finite(value):
        raise ValueError("number")
    return value


def _lease(raw: bytes, lane: str, name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_int=_number,
                           parse_float=_number, parse_constant=_number)
        _require(isinstance(value, dict), "lane_lease_invalid")
        reference = {"run_ref", "scene_ref"} & value.keys()
        _require(len(reference) == 1 and set(value) in (_FIELDS | reference, _FIELDS | reference | {"renewed_at_epoch"}),
                 "lane_lease_invalid")
        timestamps = [value[k] for k in ("created_at_epoch", "expires_at_epoch")]
        timestamps += [value[k] for k in ("renewed_at_epoch", "released_at_epoch") if k in value and value[k] is not None]
        _require(all(_finite(v) for v in timestamps), "lane_lease_invalid")
        _require(value["schema_version"] == SCHEMA_VERSION and value["lane"] == lane and value["name"] == name
                 and _lease_fields_valid(value), "lane_lease_invalid")
        _require(value["lease_digest"] == canonical_digest(value, digest_field="lease_digest"), "lane_lease_invalid")
        return value
    except (ValueError, UnicodeError, TypeError, KeyError, OverflowError, RecursionError):
        raise _Blocked("lane_lease_invalid") from None


class _Observation:
    def __init__(self, roots: tuple[str, ...], pins: str, observed: float, enabled: bool,
                 clock: Callable[[], float], budget: float):
        self.roots, self.pins, self.observed, self.enabled = roots, pins, observed, enabled
        self.clock, self.budget = clock, budget
        self.last_clock: float | None = None
        self.deadline: float | None = None
        self.fds: dict[int, tuple[int, int, int] | None] = {}
        self.failed_closes: set[int] = set()
        self.blockers: set[str] = set()
        self.rows: list[dict[str, Any]] = []
        self.row_files: list[dict[tuple[int, int], os.stat_result]] = []
        self.directories: list[tuple[int, os.stat_result, tuple[str, ...]]] = []
        self.named: list[tuple[int, str, tuple[int, ...], dict[str, Any] | None]] = []
        self.leases: list[tuple[int, str, str, bytes, dict[str, Any]]] = []
        self.chains: list[tuple[str, tuple[tuple[int, int], ...]]] = []
        self.entries = [0, 0]
        self.lanes = self.folders = self.unregistered = self.lease_bytes = self.comparisons = 0
        self.pin_complete = False

    def tick(self) -> float:
        try:
            current = self.clock()
            if not _finite(current) or (self.last_clock is not None and current < self.last_clock):
                raise ValueError("clock")
            current = float(current)
        except Exception:
            raise _Blocked("lane_clock_invalid") from None
        self.last_clock = current
        if self.deadline is None:
            self.deadline = current + self.budget
        _require(current < self.deadline, "lane_deadline_exceeded")
        return current

    def block(self, code: str) -> None:
        if code in self.blockers or len(self.blockers) < MAX_BLOCKERS:
            self.blockers.add(code)
        else:
            self.blockers.add("lane_blockers_truncated")

    def call(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self.tick()
        value = operation(*args, **kwargs)
        self.tick()
        return value

    def open(self, name: str, flags: int, parent: int | None = None) -> int:
        self.tick()
        fd = os.open(name, flags, dir_fd=parent)
        self.fds[fd] = None
        value = os.fstat(fd)
        self.fds[fd] = (value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode))
        self.tick()
        return fd

    def close(self, fd: int) -> None:
        if fd in self.failed_closes:
            try:
                value = os.fstat(fd)
                if self.fds[fd] != (value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode)):
                    self.block("lane_descriptor_changed")
                    del self.fds[fd]
                    return
            except OSError as error:
                if error.errno == errno.EBADF:
                    del self.fds[fd]
                return
        try:
            os.close(fd)
        except OSError as error:
            self.block("lane_descriptor_close_failed")
            if error.errno == errno.EBADF:
                del self.fds[fd]
            else:
                self.failed_closes.add(fd)
        else:
            del self.fds[fd]
            self.failed_closes.discard(fd)

    def chain(self, path: str) -> tuple[int, tuple[tuple[int, int], ...]]:
        parent = None
        identities = []
        for name in ["/", *path[1:].split("/")] if path != "/" else ["/"]:
            parent = self.open(name, _DIR, parent)
            value = self.call(os.fstat, parent)
            identities.append((value.st_dev, value.st_ino))
        return parent, tuple(identities)

    def names(self, fd: int, pass_index: int) -> tuple[str, ...]:
        self.tick()
        names = []
        with os.scandir(fd) as entries:
            for entry in entries:
                self.tick()
                self.entries[pass_index] += 1
                _require(self.entries[pass_index] <= MAX_ENTRIES, "lane_entries_limit")
                names.append(entry.name)
        self.tick()
        return tuple(sorted(names))

    def directory(self, parent: int, name: str, device: int, row: dict[str, Any] | None = None) -> int:
        before = self.call(os.stat, name, dir_fd=parent, follow_symlinks=False)
        _require(stat.S_ISDIR(before.st_mode) and before.st_dev == device, "lane_directory_unsafe")
        fd = self.open(name, _DIR, parent)
        _require(_identity(self.call(os.fstat, fd)) == _identity(before), "lane_directory_changed")
        self.named.append((parent, name, _identity(before), row))
        return fd

    def read_lease(self, directory: int, lane: str, name: str) -> tuple[dict[str, Any], bytes]:
        fd = self.open(LEASE_FILE, _FILE, directory)
        try:
            before = self.call(os.fstat, fd)
            _require(stat.S_ISREG(before.st_mode), "lane_lease_invalid")
            _require(before.st_size <= MAX_LEASE_BYTES, "lane_lease_bytes_limit")
            remaining = MAX_TOTAL_LEASE_BYTES - self.lease_bytes
            _require(before.st_size + 1 <= remaining, "lane_lease_total_bytes_limit")
            raw = self.call(os.read, fd, before.st_size + 1)
            self.lease_bytes += len(raw)
            after = self.call(os.fstat, fd)
            current = self.call(os.stat, LEASE_FILE, dir_fd=directory, follow_symlinks=False)
            _require(len(raw) == before.st_size and _identity(before) == _identity(after) == _identity(current),
                     "lane_lease_changed")
            return _lease(raw, lane, name), raw
        finally:
            self.close(fd)

    def measure(self, fd: int, device: int, row: dict[str, Any], depth: int = 0,
                files: dict[tuple[int, int], os.stat_result] | None = None) -> dict[tuple[int, int], os.stat_result]:
        if files is None:
            files = {}
        _require(depth <= MAX_DEPTH, "lane_depth_limit")
        initial = self.call(os.fstat, fd)
        names = self.names(fd, 0)
        self.directories.append((fd, initial, names))
        for name in names:
            value = self.call(os.stat, name, dir_fd=fd, follow_symlinks=False)
            _require(value.st_dev == device, "lane_mount_crossing")
            if stat.S_ISDIR(value.st_mode):
                child = self.directory(fd, name, device, row)
                self.measure(child, device, row, depth + 1, files)
            else:
                _require(stat.S_ISREG(value.st_mode), "lane_payload_unsafe")
                current = self.call(os.stat, name, dir_fd=fd, follow_symlinks=False)
                _require(_identity(value) == _identity(current), "lane_payload_changed")
                self.named.append((fd, name, _identity(value), row))
                files[(value.st_dev, value.st_ino)] = value
                if value.st_nlink > 1:
                    row["keep_reasons"].append("hardlinked_payload")
        return files

    def folder(self, parent: int, root: str, lane: str, name: str, device: int) -> None:
        self.folders += 1
        _require(self.folders <= MAX_FOLDERS, "lane_folders_limit")
        _require(_ID.fullmatch(name) is not None, "lane_entry_unknown")
        fd = self.directory(parent, name, device)
        try:
            value, raw = self.read_lease(fd, lane, name)
        except FileNotFoundError:
            self.unregistered += 1
            return
        row = {"root": root, "lane": lane, "name": name, "folder_identity": _identity(self.call(os.fstat, fd)),
               "lane_identity": _identity(self.call(os.fstat, parent)), "lease_digest": value["lease_digest"],
               "owner": value["owner"], "reference": {k: value[k] for k in ("run_ref", "scene_ref") if k in value},
               "class_intent": value["class_intent"], "cleanup": value["cleanup"],
               "created_at_epoch": value["created_at_epoch"], "expires_at_epoch": value["expires_at_epoch"],
               "released_at_epoch": value["released_at_epoch"],
               "lease_status": "released" if value["released_at_epoch"] is not None else
                   ("live" if value["expires_at_epoch"] > self.observed else "expired"),
               "logical_bytes": None, "allocated_bytes": None, "pin_match": "references_unknown",
               "kept": True, "keep_reasons": list(_GATES), "blockers": []}
        if row["lease_status"] == "live":
            row["keep_reasons"].append("live_lease")
        if value["cleanup"] == "owner_review":
            row["keep_reasons"].append("owner_review")
        self.rows.append(row)
        self.leases.append((fd, lane, name, raw, row))
        self.row_files.append({})
        try:
            files = self.measure(fd, device, row)
            self.row_files[-1] = files
            row["logical_bytes"] = sum(v.st_size for v in files.values())
            row["allocated_bytes"] = sum(v.st_blocks * 512 for v in files.values())
        except (OSError, _Blocked) as error:
            code = error.code if isinstance(error, _Blocked) else "lane_payload_unavailable"
            self.mark(row, code)
            if code in _RESOURCE or code.endswith("limit"):
                raise

    def mark(self, row: dict[str, Any], code: str) -> None:
        self.block(code)
        row["blockers"].append(code)
        row["logical_bytes"] = row["allocated_bytes"] = None
        row["keep_reasons"].append("changed_lease" if code == "lane_lease_changed" else "measurement_incomplete")

    def scan(self) -> None:
        identities = set()
        for root in self.roots:
            try:
                fd, chain = self.chain(root)
                self.chains.append((root, chain))
                initial = self.call(os.fstat, fd)
                _require((initial.st_dev, initial.st_ino) not in identities, "lane_root_alias")
                identities.add((initial.st_dev, initial.st_ino))
                names = self.names(fd, 0)
                self.directories.append((fd, initial, names))
                for lane in names:
                    try:
                        if lane == ".lane-scratch.lock":
                            value = self.call(os.stat, lane, dir_fd=fd, follow_symlinks=False)
                            _require(stat.S_ISREG(value.st_mode), "lane_metadata_unsafe")
                            self.named.append((fd, lane, _identity(value), None))
                            continue
                        _require(_ID.fullmatch(lane) is not None, "lane_entry_unknown")
                        self.lanes += 1
                        _require(self.lanes <= MAX_LANES, "lane_lanes_limit")
                        lane_fd = self.directory(fd, lane, initial.st_dev)
                        before = self.call(os.fstat, lane_fd)
                        folders = self.names(lane_fd, 0)
                        self.directories.append((lane_fd, before, folders))
                        for name in folders:
                            try:
                                self.folder(lane_fd, root, lane, name, initial.st_dev)
                            except (OSError, _Blocked) as error:
                                code = error.code if isinstance(error, _Blocked) else "lane_folder_unavailable"
                                if code in _RESOURCE or code.endswith("limit"):
                                    raise _Blocked(code) from None
                                self.block(code)
                    except _Blocked as error:
                        if error.code in _RESOURCE or error.code.endswith("limit"):
                            raise
                        self.block(error.code)
                    except OSError:
                        self.block("lane_directory_unavailable")
            except OSError:
                self.block("lane_root_unavailable")
        self.verify()
        self.references()

    def verify(self) -> None:
        for fd, initial, names in self.directories:
            _require(self.names(fd, 1) == names and _identity(self.call(os.fstat, fd)) == _identity(initial),
                     "lane_directory_changed")
        for parent, name, identity, row in self.named:
            try:
                _require(_identity(self.call(os.stat, name, dir_fd=parent, follow_symlinks=False)) == identity,
                         "lane_payload_changed" if row else "lane_directory_changed")
            except (OSError, _Blocked) as error:
                code = error.code if isinstance(error, _Blocked) else "lane_entry_unavailable"
                if code in _RESOURCE:
                    raise
                if row is not None:
                    self.mark(row, "lane_lease_changed" if name == LEASE_FILE else code)
                else:
                    self.block(code)
        for fd, lane, name, raw, row in self.leases:
            try:
                _, current = self.read_lease(fd, lane, name)
                _require(current == raw, "lane_lease_changed")
            except (OSError, _Blocked) as error:
                code = error.code if isinstance(error, _Blocked) else "lane_lease_changed"
                if code in _RESOURCE or code.endswith("limit"):
                    raise _Blocked(code) from None
                self.mark(row, "lane_lease_changed")
        for root, original in self.chains:
            _, current = self.chain(root)
            _require(current == original, "lane_root_changed")

    def references(self) -> None:
        current = self.tick()
        try:
            pins = observe_storage_pins(self.pins, observed_at_epoch=self.observed, monotonic=self.clock,
                                        time_budget_seconds=min(5.0, self.deadline - current))
        except (OSError, StoragePinObservationError):
            pins = None
        self.tick()
        self.pin_complete = bool(pins is not None and pins.complete)
        if not self.pin_complete:
            self.block("lane_references_unknown")
        for row in self.rows:
            self.tick()
            path = row["root"] + "/" + row["lane"] + "/" + row["name"]
            matched = False
            for protected in pins.protected_paths if pins is not None else ():
                self.tick()
                self.comparisons += 1
                if self.comparisons > MAX_COMPARISONS:
                    self.pin_complete = False
                    self.block("lane_reference_comparisons_limit")
                    break
                if path == protected or path.startswith(protected.rstrip("/") + "/") or protected.startswith(path + "/"):
                    matched = True
                    break
            if matched:
                row["pin_match"] = "referenced"
                row["keep_reasons"].append("referenced")
            elif self.pin_complete:
                row["pin_match"] = "no_pin_match_in_observed_ledger"
            if not self.pin_complete:
                row["keep_reasons"].append("references_unknown")
        if not self.pin_complete:
            for row in self.rows:
                if "references_unknown" not in row["keep_reasons"]:
                    row["keep_reasons"].append("references_unknown")
                if row["pin_match"] != "referenced":
                    row["pin_match"] = "references_unknown"

    def result(self, fallback: str | None = None) -> dict[str, Any]:
        if fallback is not None:
            self.block(fallback)
            self.rows, self.row_files, self.unregistered, self.pin_complete = [], [], 0, False
        else:
            self.tick()
        complete = bool(self.roots) and not self.blockers
        reasons: dict[str, dict[str, Any]] = {}
        files: dict[tuple[int, int], os.stat_result] = {}
        for row, observed_files in zip(self.rows, self.row_files):
            if fallback is None:
                self.tick()
            row["keep_reasons"] = [r for r in _PRIORITY if r in row["keep_reasons"]]
            row["blockers"] = sorted(set(row["blockers"]))[:MAX_BLOCKERS]
            primary = row["keep_reasons"][0]
            reasons.setdefault(primary, {"count": 0, "bytes": None})["count"] += 1
            files.update(observed_files)
        result = {"schema_version": "control_plane_lane_scratch_retention.v1",
                  "status": "report_only" if self.roots else "not_configured",
                  "enabled_requested": self.enabled, "complete": complete,
                  "observed_at_epoch": self.observed, "mutations": 0, "execution_authorized": False,
                  "apply_supported": False, "candidate_bytes": None, "removed_bytes": 0,
                  "observed_registered_count": len(self.rows), "observed_unregistered_count": self.unregistered,
                  "registered_count": len(self.rows) if complete else None,
                  "unregistered_count": self.unregistered if complete else None,
                  "logical_bytes": sum(v.st_size for v in files.values()) if complete else None,
                  "allocated_bytes": sum(v.st_blocks * 512 for v in files.values()) if complete else None,
                  "measurement_scope": "unique_regular_inodes_in_registered_folders",
                  "lease_included": True, "directory_allocation_included": False,
                  "pin_observation_complete": self.pin_complete,
                  "general_reference_inventory_complete": False,
                  "queues_checked": False, "processes_checked": False, "consumer_fence_checked": False,
                  "owner_approval_checked": False, "evidence_policy_checked": False, "restore_checked": False,
                  "rows": self.rows, "retained_by_reason": reasons, "blockers": sorted(self.blockers)}
        if fallback is None:
            self.tick()
            try:
                size = len(json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8"))
            except (ValueError, UnicodeError, TypeError, OverflowError, RecursionError):
                raise _Blocked("lane_output_invalid") from None
            self.tick()
            _require(size <= MAX_OUTPUT_BYTES, "lane_output_limit")
        return result


def observe_lane_scratch_retention(lane_roots: tuple[str, ...], *, pins_root: str,
                                  observed_at_epoch: float, enabled_requested: bool,
                                  monotonic: Callable[[], float] = time.monotonic,
                                  time_budget_seconds: float = 10.0) -> dict[str, Any]:
    """Observe bounded configured folders; even enabled observations cannot apply."""
    if (type(lane_roots) is not tuple or len(lane_roots) > MAX_ROOTS or type(enabled_requested) is not bool
            or not _finite(observed_at_epoch) or observed_at_epoch < 0 or not callable(monotonic)
            or not _finite(time_budget_seconds) or not 0 < time_budget_seconds <= 10):
        raise LaneScratchRetentionError("lane_parameters_invalid")
    roots = tuple(_path(root) for root in lane_roots)
    if (any(root not in _ALLOWED_ROOTS for root in roots) or len(set(roots)) != len(roots)
            or any(a.startswith(b.rstrip("/") + "/") for a in roots for b in roots if a != b)):
        raise LaneScratchRetentionError("lane_parameters_invalid")
    scan = _Observation(roots, _path(pins_root), float(observed_at_epoch), enabled_requested,
                        monotonic, float(time_budget_seconds))
    try:
        if roots:
            scan.scan()
    except _Blocked as error:
        scan.block(error.code)
    except OSError:
        scan.block("lane_observation_unavailable")
    finally:
        for _ in range(2):
            for fd in reversed(tuple(scan.fds)):
                scan.close(fd)
    try:
        return scan.result()
    except _Blocked as error:
        return scan.result(error.code)


__all__ = ["LaneScratchRetentionError", "observe_lane_scratch_retention"]
