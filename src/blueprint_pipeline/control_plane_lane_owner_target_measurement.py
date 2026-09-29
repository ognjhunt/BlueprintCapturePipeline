"""Same-budget one-target allocated-byte observation, never reclaim authority."""
from __future__ import annotations

import os
import stat

from .control_plane_disk_usage import allocated_bytes
from .control_plane_lane_owner_target_io import _matches_tuple
from .control_plane_lane_owner_target_versions import OwnerTargetVersionError, _require
from .control_plane_reference_budget import ReferenceCollectionBudgetError

MAX_MEASUREMENT_ENTRIES = 4096
MAX_MEASUREMENT_DEPTH = 16
_SCOPE = "names_within_one_target_not_exclusive_physical_ownership"


def _version(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns, getattr(info, "st_blocks", None))


def _measure_target_allocated(files, target_fd, target_identity, budget):
    _require(budget is files.budget, "owner_target_resource_exhausted")
    seen, visited, total = set(), 0, 0
    opened = []

    def names(fd):
        files.proof(fd)
        files.slot()  # The bounded scandir iterator holds one temporary OS slot.
        values = []
        with os.scandir(fd) as stream:
            for entry in stream:
                budget.charge("entries")
                _require(len(values) < MAX_MEASUREMENT_ENTRIES, "owner_target_measurement_incomplete")
                name = entry.name
                budget.measure(name, cap=1024)
                _require(name not in (".", "..") and len(os.fsencode(name)) <= 255,
                         "owner_target_measurement_incomplete")
                budget.charge("facts")
                values.append(name)
        budget.available("values", len(values))
        budget.tick()
        return sorted(values)

    def account(info):
        nonlocal visited, total
        budget.tick()
        _require(visited < MAX_MEASUREMENT_ENTRIES and info.st_dev == target_identity["dev"]
                 and (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)),
                 "owner_target_measurement_incomplete")
        visited += 1
        budget.charge("facts")
        identity = (info.st_dev, info.st_ino)
        if identity not in seen:
            budget.charge("entries")
            _require(len(seen) < MAX_MEASUREMENT_ENTRIES, "owner_target_measurement_incomplete")
            seen.add(identity)
            count = allocated_bytes(info)
            _require(type(count) is int and 0 <= count <= 2**63 - 1 - total,
                     "owner_target_measurement_incomplete")
            total += count

    def walk(fd, depth):
        _require(depth <= MAX_MEASUREMENT_DEPTH, "owner_target_measurement_incomplete")
        files.proof(fd)
        before = os.fstat(fd)
        account(before)
        first = names(fd)
        versions = {}
        for name in first:
            budget.charge("entries")
            files.proof(fd)
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            budget.measure(_version(info), cap=512)
            versions[name] = _version(info)
            if stat.S_ISDIR(info.st_mode):
                _require(info.st_dev == target_identity["dev"], "owner_target_measurement_incomplete")
                child = files.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, parent=fd, target=True)
                opened.append(child)
                try:
                    _require(_version(os.fstat(child)) == versions[name], "owner_target_measurement_incomplete")
                    walk(child, depth + 1)
                finally:
                    files.close(child)
            else:
                account(info)
        _require(names(fd) == first, "owner_target_measurement_incomplete")
        for name, original in versions.items():
            budget.charge("entries")
            files.proof(fd)
            _require(_version(os.stat(name, dir_fd=fd, follow_symlinks=False)) == original,
                     "owner_target_measurement_incomplete")
        files.proof(fd)
        _require(_version(os.fstat(fd)) == _version(before), "owner_target_measurement_incomplete")

    try:
        budget.tick()
        files.proof(target_fd)
        _require(_matches_tuple(os.fstat(target_fd), target_identity), "owner_target_version_changed")
        walk(target_fd, 0)
        _require(not files.unresolved and not any(fd in files.owned for fd in opened),
                 "owner_target_measurement_incomplete")
        return dict(measurement_complete=True, measured_allocated_bytes=total, allocation_scope=_SCOPE,
                    candidate_bytes=None, eta_seconds=None, mutations=0, kept_reasons=[])
    except (OSError, OwnerTargetVersionError, ReferenceCollectionBudgetError, UnicodeError):
        return dict(measurement_complete=False, measured_allocated_bytes=None, allocation_scope=_SCOPE,
                    candidate_bytes=None, eta_seconds=None, mutations=0,
                    kept_reasons=["target_measurement_incomplete"])
