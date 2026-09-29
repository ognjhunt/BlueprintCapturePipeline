"""Exact read-only target generation for historical owner attribution.

An old path-only census is never a registration or cleanup grant. The caller
must obtain a separate owner decision for the digest of this observation.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from contextlib import ExitStack
from pathlib import Path

from .control_plane_disk_usage import allocated_bytes

_DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
MAX_ENTRIES = 200_000
MAX_DEPTH = 64
MAX_SECONDS = 240.0


class LegacyOwnerError(ValueError):
    """Fixed code for unsafe or changed historical attribution evidence."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise LegacyOwnerError(code)


def _version(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
            info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
            getattr(info, "st_blocks", 0))


def _directory_identity(info: os.stat_result) -> dict[str, int | str]:
    return dict(dev=info.st_dev, ino=info.st_ino, type="directory",
                mode=stat.S_IMODE(info.st_mode), uid=info.st_uid, gid=info.st_gid,
                ctime_ns=info.st_ctime_ns)


def _absolute(value: Path) -> None:
    _require(value.is_absolute() and value != Path("/") and ".." not in value.parts
             and len(value.parts) <= MAX_DEPTH and len(os.fsencode(value)) <= 4096,
             "legacy_target_unsafe")
    _require(all(part not in ("", ".", "..") and len(os.fsencode(part)) <= 255
                 for part in value.parts[1:]), "legacy_target_unsafe")


def _chain(path: Path, stack: ExitStack) -> list[tuple[int | None, str, int, tuple[int, ...]]]:
    """Keep every original named descriptor; never resolve a user path."""
    result: list[tuple[int | None, str, int, tuple[int, ...]]] = []
    previous: int | None = None
    for component in ("/", *path.parts[1:]):
        try:
            named = os.stat(component, dir_fd=previous, follow_symlinks=False)
            _require(stat.S_ISDIR(named.st_mode), "legacy_target_unsafe")
            descriptor = os.open(component, _DIRECTORY, dir_fd=previous)
            stack.callback(os.close, descriptor)
            opened = os.fstat(descriptor)
        except (OSError, ValueError) as error:
            if isinstance(error, LegacyOwnerError):
                raise
            raise LegacyOwnerError("legacy_target_unsafe") from None
        _require(_version(named) == _version(opened), "legacy_target_changed")
        result.append((previous, component, descriptor, _version(opened)))
        previous = descriptor
    return result


def _verify_chain(chain: list[tuple[int | None, str, int, tuple[int, ...]]]) -> None:
    for parent, name, descriptor, expected in chain:
        try:
            actual = os.fstat(descriptor)
            named = os.stat(name, dir_fd=parent, follow_symlinks=False)
        except OSError:
            raise LegacyOwnerError("legacy_target_changed") from None
        _require(_version(actual) == expected == _version(named), "legacy_target_changed")


def snapshot_generation(
    path: str | Path, *, allowed_roots: tuple[str | Path, ...],
    max_entries: int = MAX_ENTRIES, max_seconds: float = MAX_SECONDS,
) -> dict:
    """Bounded no-follow observation of an existing target, with no mutation.

    The tree digest includes names and inode metadata; it is an owner-label
    generation check and does not establish exclusive writer or reader lifetime.
    """
    target = Path(path)
    _absolute(target)
    _require(type(max_entries) is int and 0 < max_entries <= MAX_ENTRIES
             and type(max_seconds) in (int, float) and 0 < max_seconds <= MAX_SECONDS,
             "legacy_target_options_invalid")
    roots = tuple(Path(root) for root in allowed_roots)
    _require(0 < len(roots) <= 2, "legacy_target_options_invalid")
    for root in roots:
        _absolute(root)
    candidates = [root for root in roots if root in target.parents]
    _require(len(candidates) == 1, "legacy_target_unsafe")
    root = candidates[0]
    deadline = time.monotonic() + max_seconds
    entries: list[tuple[str, tuple[int, ...]]] = []
    bytes_allocated = 0

    def tick() -> None:
        _require(time.monotonic() < deadline, "legacy_target_measurement_incomplete")

    def walk(descriptor: int, relative: str, depth: int, device: int) -> None:
        nonlocal bytes_allocated
        tick()
        _require(depth <= MAX_DEPTH and len(entries) < max_entries,
                 "legacy_target_measurement_incomplete")
        start = os.fstat(descriptor)
        _require(stat.S_ISDIR(start.st_mode) and start.st_dev == device,
                 "legacy_target_measurement_incomplete")
        entries.append((relative, _version(start)))
        bytes_allocated += allocated_bytes(start)
        try:
            with os.scandir(descriptor) as iterator:
                names = sorted(entry.name for entry in iterator)
        except OSError:
            raise LegacyOwnerError("legacy_target_measurement_incomplete") from None
        _require(len(names) + len(entries) <= max_entries and len(names) == len(set(names)),
                 "legacy_target_measurement_incomplete")
        versions = {}
        for name in names:
            tick()
            _require(name not in ("", ".", "..") and len(os.fsencode(name)) <= 255,
                     "legacy_target_measurement_incomplete")
            try:
                info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except OSError:
                raise LegacyOwnerError("legacy_target_measurement_incomplete") from None
            _require(info.st_dev == device and (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)),
                     "legacy_target_measurement_incomplete")
            versions[name] = _version(info)
            child_relative = f"{relative}/{name}" if relative else name
            if stat.S_ISDIR(info.st_mode):
                try:
                    child = os.open(name, _DIRECTORY, dir_fd=descriptor)
                    _require(_version(os.fstat(child)) == versions[name],
                             "legacy_target_changed")
                    walk(child, child_relative, depth + 1, device)
                except OSError:
                    raise LegacyOwnerError("legacy_target_measurement_incomplete") from None
                finally:
                    if "child" in locals():
                        os.close(child)
                        del child
            else:
                _require(len(entries) < max_entries, "legacy_target_measurement_incomplete")
                entries.append((child_relative, versions[name]))
                bytes_allocated += allocated_bytes(info)
        try:
            with os.scandir(descriptor) as iterator:
                _require(sorted(entry.name for entry in iterator) == names,
                         "legacy_target_changed")
            for name, version in versions.items():
                _require(_version(os.stat(name, dir_fd=descriptor, follow_symlinks=False)) == version,
                         "legacy_target_changed")
            _require(_version(os.fstat(descriptor)) == _version(start), "legacy_target_changed")
        except OSError:
            raise LegacyOwnerError("legacy_target_changed") from None

    with ExitStack() as stack:
        chain = _chain(target, stack)
        root_index = len(root.parts) - 1
        root_entry = chain[root_index]
        target_entry = chain[-1]
        _require(root_entry[3][0] == target_entry[3][0], "legacy_target_unsafe")
        walk(target_entry[2], "", 0, target_entry[3][0])
        _verify_chain(chain)
        tick()
        encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
        return dict(path=str(target), root=str(root),
                    root_identity=_directory_identity(os.fstat(root_entry[2])),
                    target=_directory_identity(os.fstat(target_entry[2])),
                    tree=dict(digest="sha256:" + hashlib.sha256(encoded).hexdigest(),
                              entries=len(entries), allocated_bytes=bytes_allocated))
