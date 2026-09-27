"""Descriptor-backed payload directories for an exact, active scratch lease.

Only operations on this handle are guarded. Returned paths and downstream file
writes remain ordinary filesystem operations, not capabilities or authorizations.
The lane-root lock coordinates cooperating lease writers; it is not a sandbox
against other processes with the same filesystem permissions.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from .control_plane_lane_scratch import (
    DEFAULT_ROOT, LaneScratchError, _DIR_FLAGS, _id, _locked_root_descriptor,
    _now, _read_lease, create_lane_scratch,
)


def _directory(stack: ExitStack, path: str | Path, *, parent: int | None = None) -> int:
    fd = os.open(path, _DIR_FLAGS, dir_fd=parent)
    stack.callback(os.close, fd)
    return fd


def _absolute_directory(stack: ExitStack, path: Path) -> int:
    if not path.is_absolute() or ".." in path.parts:
        raise LaneScratchError("lane_scratch_root_unsafe")
    fd = _directory(stack, "/")
    for component in path.parts[1:]:
        fd = _directory(stack, component, parent=fd)
    return fd


def _same(first: int, second: int) -> bool:
    a, b = os.fstat(first), os.fstat(second)
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


def _components(relative: str) -> tuple[str, ...]:
    if not isinstance(relative, str):
        raise LaneScratchError("lane_scratch_payload_path_invalid")
    parts = tuple(relative.split("/"))
    if (len(parts) > 32 or any(part in ("", ".", "..") or "\0" in part
                             or len(os.fsencode(part)) > 255 for part in parts)):
        raise LaneScratchError("lane_scratch_payload_path_invalid")
    return parts


class LeasedScratchDirectory:
    """A closeable handle bound to exact root/lane/folder inodes and lease digest."""

    @classmethod
    def open(
        cls, *, lane: str, name: str, owner: str, run_ref: str | None = None,
        scene_ref: str | None = None, root: str | Path = DEFAULT_ROOT,
        now: Callable[[], float] = time.time,
    ) -> LeasedScratchDirectory:
        lane, name, owner = _id(lane, "lane"), _id(name, "name"), _id(owner, "owner")
        if (run_ref is None) == (scene_ref is None):
            raise LaneScratchError("lane_scratch_reference_invalid")
        reference_key = "run_ref" if run_ref is not None else "scene_ref"
        reference = _id(run_ref if run_ref is not None else scene_ref, "reference")
        result = cls()
        result.root, result.lane, result.name, result.owner = Path(root), lane, name, owner
        result.reference_key, result.reference = reference_key, reference
        result.now, result._closed = now, False
        with ExitStack() as descriptors:
            try:
                result._root_fd = _absolute_directory(descriptors, result.root)
                result._lane_fd = _directory(descriptors, lane, parent=result._root_fd)
                result._folder_fd = _directory(descriptors, name, parent=result._lane_fd)
                with result._operation(refresh=True) as lease:
                    result.lease_digest = lease["lease_digest"]
            except OSError as exc:
                raise LaneScratchError("lane_scratch_capability_path_unsafe") from exc
            result._descriptors = descriptors.pop_all()
        return result

    @property
    def path(self) -> Path:
        """Compatibility path; using it directly is outside the handle's guarantee."""

        return self.root / self.lane / self.name

    def _paths_match(self, locked_root_fd: int) -> None:
        with ExitStack() as current:
            root_fd = _absolute_directory(current, self.root)
            lane_fd = _directory(current, self.lane, parent=root_fd)
            folder_fd = _directory(current, self.name, parent=lane_fd)
            if not (all(_same(a, b) for a, b in (
                (root_fd, self._root_fd), (lane_fd, self._lane_fd),
                (folder_fd, self._folder_fd), (locked_root_fd, self._root_fd),
            ))):
                raise LaneScratchError("lane_scratch_capability_path_changed")

    @contextmanager
    def _operation(self, *, refresh: bool = False) -> Iterator[dict[str, Any]]:
        if self._closed:
            raise LaneScratchError("lane_scratch_capability_closed")
        with _locked_root_descriptor(self._root_fd) as root_fd:
            try:
                self._paths_match(root_fd)
            except OSError as exc:
                raise LaneScratchError("lane_scratch_capability_path_unsafe") from exc
            lease = _read_lease(self._folder_fd)
            other_key = "scene_ref" if self.reference_key == "run_ref" else "run_ref"
            if (lease.get("lane") != self.lane or lease.get("name") != self.name
                    or lease.get("owner") != self.owner or other_key in lease
                    or lease.get(self.reference_key) != self.reference):
                raise LaneScratchError("lane_scratch_capability_identity_changed")
            if (lease.get("released_at_epoch") is not None
                    or lease["expires_at_epoch"] <= _now(self.now)):
                raise LaneScratchError("lane_scratch_capability_inactive")
            if not refresh and lease["lease_digest"] != self.lease_digest:
                raise LaneScratchError("lane_scratch_lease_changed")
            yield lease

    def refresh(self) -> None:
        """Explicitly adopt a renewed lease after revalidating exact identity."""

        with self._operation(refresh=True) as lease:
            self.lease_digest = lease["lease_digest"]

    def mkdir(self, relative: str, *, parents: bool = False, exist_ok: bool = False) -> Path:
        """Make payload directories beneath this folder, with no symlink traversal."""

        parts = _components(relative)
        with self._operation(), ExitStack() as payload_fds:
            parent_fd = self._folder_fd
            for index, component in enumerate(parts):
                final = index == len(parts) - 1
                if parents or final:
                    try:
                        os.mkdir(component, 0o750, dir_fd=parent_fd)
                    except FileExistsError:
                        if final and not exist_ok:
                            raise
                try:
                    parent_fd = _directory(payload_fds, component, parent=parent_fd)
                except OSError as exc:
                    raise LaneScratchError("lane_scratch_payload_path_unsafe") from exc
            os.fsync(parent_fd)
        return self.path.joinpath(*parts)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._descriptors.close()

    def __enter__(self) -> LeasedScratchDirectory:
        if self._closed:
            raise LaneScratchError("lane_scratch_capability_closed")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def create_leased_lane_scratch(
    lane: str, name: str, *, root: str | Path = DEFAULT_ROOT, owner: str,
    run_ref: str | None = None, scene_ref: str | None = None,
    now: Callable[[], float] = time.time, **lease_options: Any,
) -> LeasedScratchDirectory:
    """Publish using the existing atomic constructor, then open its exact lease."""

    create_lane_scratch(lane, name, root=root, owner=owner, run_ref=run_ref,
                        scene_ref=scene_ref, now=now, **lease_options)
    return LeasedScratchDirectory.open(root=root, lane=lane, name=name, owner=owner,
                                       run_ref=run_ref, scene_ref=scene_ref, now=now)
