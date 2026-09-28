"""Optional cooperating lifetime locks, never eviction authorization.

The target directory inode carries shared use or a nonblocking exclusive probe.
Inherited open file descriptions are close-only: explicit unlock would revoke a
child's shared authority too. Unadopted readers remain outside this protocol.
"""

from __future__ import annotations

import errno
import fcntl
import os
import stat
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .control_plane_lane_scratch import (
    CONSUMER_LIFETIME_PROTOCOL, LaneScratchError, _DIR_FLAGS, _creation_lease, _id,
    _now, _publish_scratch_folder, _read_lease,
)

PROTOCOL = CONSUMER_LIFETIME_PROTOCOL
LANE_ROOTS = (Path("/mnt/blueprint-work/lanes"), Path("/var/lib/blueprint/task-evaluation-inputs/lanes"))


def _identity(fd: int) -> tuple[int, int]:
    info = os.fstat(fd)
    return info.st_dev, info.st_ino


def _path(value: str | Path) -> Path:
    text = str(value)
    if (not text.startswith("/") or len(text) > 4096 or len(os.fsencode(text)) > 4096
            or any(c in text for c in ("\0", "\n", "\r"))):
        raise LaneScratchError("lane_scratch_lifetime_path_invalid")
    parts = text[1:].split("/")
    if len(parts) > 64 or any(p in ("", ".", "..") or len(os.fsencode(p)) > 255 for p in parts):
        raise LaneScratchError("lane_scratch_lifetime_path_invalid")
    return Path(text)


class LeasedScratchUse:
    """Retained target authority; a probe's authority ends before report return."""

    def __init__(self) -> None:
        self._owned: dict[int, tuple[int, int] | None] = {}
        self._closed = False
        self.unresolved_ownership = False

    def _open(self, name: str | Path, flags: int, parent: int | None = None) -> int:
        fd = os.open(name, flags, dir_fd=parent)
        self._take(fd)
        return fd

    def _take(self, fd: int) -> None:
        if type(fd) is not int or fd < 0 or self._closed or len(self._owned) >= 768:
            raise LaneScratchError("lane_scratch_lifetime_descriptor_invalid")
        self._owned[fd] = None
        try:
            self._owned[fd] = _identity(fd)
        except OSError as error:
            if error.errno == errno.EBADF:
                self._owned.pop(fd, None)
            raise

    def _take_all(self, descriptors) -> None:
        # Register every transferred token before observing any one of them.
        values = tuple(dict.fromkeys(fd for fd in descriptors if type(fd) is int and fd >= 0))
        if len(values) > 768:
            raise LaneScratchError("lane_scratch_lifetime_descriptor_invalid")
        for fd in values:
            self._owned[fd] = None
        failed = False
        for fd in values:
            try:
                self._owned[fd] = _identity(fd)
            except OSError as error:
                if error.errno == errno.EBADF:
                    self._owned.pop(fd, None)
                failed = True
        if failed:
            raise LaneScratchError("lane_scratch_handshake_invalid")

    def _detach(self, fd: int) -> tuple[int, int]:
        identity = self._owned.pop(fd, None)
        if identity is None:
            self.unresolved_ownership = True
            raise LaneScratchError("lane_scratch_descriptor_ownership_unproven")
        return identity

    def _absolute(self, path: Path) -> int:
        fd = self._open("/", _DIR_FLAGS)
        for part in path.parts[1:]:
            fd = self._open(part, _DIR_FLAGS, fd)
        return fd

    @contextmanager
    def _root_lock(self, *, create: bool = False) -> Iterator[None]:
        lock = None
        try:
            flags = os.O_NOFOLLOW | os.O_NONBLOCK | (os.O_RDWR | os.O_CREAT if create else os.O_RDONLY)
            lock = os.open(".lane-scratch.lock", flags, 0o600, dir_fd=self._root_fd)
            self._take(lock)
            if not stat.S_ISREG(os.fstat(lock).st_mode):
                raise LaneScratchError("lane_scratch_root_lock_unsafe")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, LaneScratchError):
            if lock is not None:
                self._close_one(lock)
                self._cleanup_status()
                if lock in self._owned:
                    raise LaneScratchError("lane_scratch_descriptor_cleanup_failed")
            raise LaneScratchError("lane_scratch_root_lock_unavailable") from None
        try:
            yield
        finally:
            # This separate description is never inherited as target authority.
            self._close_one(lock)
            self._cleanup_status()
            if lock in self._owned:
                raise LaneScratchError("lane_scratch_descriptor_cleanup_failed")

    @classmethod
    def create(cls, *, root: str | Path, lane: str, name: str, owner: str,
               run_ref: str, ttl_seconds: int, now: Callable[[], float] = time.time) -> LeasedScratchUse:
        """Publish evidence and acquire SH before releasing root coordination."""
        lease = _creation_lease(lane, name, owner=owner, run_ref=run_ref, reason="g1_development_pair",
                                class_intent="evidence", cleanup="owner_review", ttl_seconds=ttl_seconds,
                                now=now, consumer_lifetime_contract=PROTOCOL)
        result = cls()
        result.root, result.lane, result.name = _path(root), lane, name
        result.now, result.exclusive = now, False
        try:
            result._root_fd = result._absolute(result.root)
            with result._root_lock(create=True):
                # Root publication and target SH admission share one lock interval.
                def verify(lane_fd: int) -> None:
                    current = LeasedScratchUse()
                    try:
                        root_now = current._absolute(result.root)
                        lane_now = current._open(lane, _DIR_FLAGS, root_now)
                        if _identity(root_now) != _identity(result._root_fd) or _identity(lane_now) != _identity(lane_fd):
                            raise LaneScratchError("lane_scratch_lifetime_path_changed")
                    finally:
                        current.close()
                published = _publish_scratch_folder(result._root_fd, lease, verify_location=verify)
                result._lane_fd = result._open(lane, _DIR_FLAGS, result._root_fd)
                result.fd = result._open(name, _DIR_FLAGS, result._lane_fd)
                if any((a.st_dev, a.st_ino) != _identity(b) for a, b in
                       zip(published, (result._root_fd, result._lane_fd, result.fd), strict=True)):
                    raise LaneScratchError("lane_scratch_lifetime_path_changed")
                fcntl.flock(result.fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                result._visible()
                if _read_lease(result.fd) != lease:
                    raise LaneScratchError("lane_scratch_lifetime_lease_changed")
                result._bind(lease)
            return result
        except OSError:
            result.close()
            raise LaneScratchError("lane_scratch_lifetime_publication_failed") from None
        except BaseException:
            result.close()
            raise

    def _bind(self, lease: dict[str, Any]) -> None:
        self.identity = {"root": str(self.root), "lane": self.lane, "name": self.name,
                         "owner": lease["owner"], "lease_digest": lease["lease_digest"],
                         "consumer_lifetime_contract": PROTOCOL,
                         "run_ref" if "run_ref" in lease else "scene_ref": lease.get("run_ref", lease.get("scene_ref")),
                         "inodes": [_identity(fd) for fd in (self._root_fd, self._lane_fd, self.fd)]}

    @classmethod
    def open(cls, *, root: str | Path, lane: str, name: str, owner: str,
             run_ref: str | None = None, scene_ref: str | None = None,
             now: Callable[[], float] = time.time) -> LeasedScratchUse:
        if (run_ref is None) == (scene_ref is None):
            raise LaneScratchError("lane_scratch_lifetime_identity_invalid")
        expected = {"owner": _id(owner, "owner"),
                    "run_ref" if run_ref is not None else "scene_ref": _id(run_ref if run_ref is not None else scene_ref, "reference")}
        return cls._admit(root=root, lane=lane, name=name, expected=expected, exclusive=False, now=now)

    @classmethod
    def probe(cls, path: str | Path, *, now: Callable[[], float] = time.time) -> LeasedScratchUse:
        path = _path(path)
        return cls._admit(root=path.parent.parent, lane=path.parent.name, name=path.name,
                          expected=None, exclusive=True, now=now)

    @classmethod
    def _admit(cls, *, root: str | Path, lane: str, name: str, expected: dict[str, str] | None,
               exclusive: bool, now: Callable[[], float]) -> LeasedScratchUse:
        result = cls()
        result.root, result.lane, result.name = _path(root), _id(lane, "lane"), _id(name, "name")
        result.now, result.exclusive = now, exclusive
        try:
            result._root_fd = result._absolute(result.root)
            result._lane_fd = result._open(lane, _DIR_FLAGS, result._root_fd)
            result.fd = result._open(name, _DIR_FLAGS, result._lane_fd)
            with result._root_lock():
                try:
                    fcntl.flock(result.fd, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise LaneScratchError("lane_scratch_consumer_busy") from None
                result._visible()
                lease = _read_lease(result.fd)
                if lease.get("consumer_lifetime_contract") != PROTOCOL:
                    raise LaneScratchError("lane_scratch_consumer_participation_unproven")
                if lease["lane"] != lane or lease["name"] != name:
                    raise LaneScratchError("lane_scratch_lifetime_identity_changed")
                if expected is not None:
                    other = "scene_ref" if "run_ref" in expected else "run_ref"
                    if other in lease or any(lease.get(k) != v for k, v in expected.items()):
                        raise LaneScratchError("lane_scratch_lifetime_identity_changed")
                    if lease.get("released_at_epoch") is not None or lease["expires_at_epoch"] <= _now(now):
                        raise LaneScratchError("lane_scratch_lifetime_inactive")
                result._bind(lease)
            return result
        except OSError:
            result.close()
            raise LaneScratchError("lane_scratch_lifetime_path_unsafe") from None
        except BaseException:
            result.close()
            raise

    @property
    def path(self) -> Path:
        return self.root / self.lane / self.name

    def _visible(self) -> None:
        current = LeasedScratchUse()
        try:
            root = current._absolute(self.root)
            lane = current._open(self.lane, _DIR_FLAGS, root)
            folder = current._open(self.name, _DIR_FLAGS, lane)
            if any(_identity(a) != _identity(b) for a, b in
                   ((root, self._root_fd), (lane, self._lane_fd), (folder, self.fd))):
                raise LaneScratchError("lane_scratch_lifetime_path_changed")
        finally:
            current.close()

    def check(self) -> None:
        if self._closed:
            raise LaneScratchError("lane_scratch_lifetime_closed")
        try:
            self._visible()
            lease = _read_lease(self.fd)
            if any(lease.get(k) != v for k, v in self.identity.items() if k not in {"root", "inodes"}):
                raise LaneScratchError("lane_scratch_lifetime_lease_changed")
        except OSError:
            raise LaneScratchError("lane_scratch_lifetime_path_unsafe") from None

    def refresh(self) -> None:
        """Explicitly bind a renewed live lease without releasing target SH."""
        if self._closed or self.exclusive:
            raise LaneScratchError("lane_scratch_lifetime_closed")
        with self._root_lock():
            try:
                self._visible()
                lease = _read_lease(self.fd)
                reference = "run_ref" if "run_ref" in self.identity else "scene_ref"
                other = "scene_ref" if reference == "run_ref" else "run_ref"
                if (other in lease or any(lease.get(key) != self.identity[key] for key in
                                         ("lane", "name", "owner", reference, "consumer_lifetime_contract"))):
                    raise LaneScratchError("lane_scratch_lifetime_identity_changed")
                if lease.get("released_at_epoch") is not None or lease["expires_at_epoch"] <= _now(self.now):
                    raise LaneScratchError("lane_scratch_lifetime_inactive")
                self._bind(lease)
            except OSError:
                raise LaneScratchError("lane_scratch_lifetime_path_unsafe") from None

    def mkdir(self, relative: str) -> Path:
        from .control_plane_leased_scratch import _components
        parts = _components(relative)
        with self._root_lock():
            self.check()
            temporary = LeasedScratchUse()
            try:
                parent = self.fd
                for part in parts:
                    os.mkdir(part, 0o750, dir_fd=parent)
                    parent = temporary._open(part, _DIR_FLAGS, parent)
            finally:
                temporary.close()
        return self.path.joinpath(*parts)

    def borrow(self, output: Path) -> LeasedScratchUse:
        """Duplicate close-only lifetime ownership for one exact child output."""
        output = _path(output)
        if output.parent != self.path:
            raise LaneScratchError("lane_scratch_worker_descendant_invalid")
        _id(output.name, "name")
        self.check()
        result = LeasedScratchUse()
        result.root, result.lane, result.name = self.root, self.lane, self.name
        result.now, result.exclusive = self.now, False
        result.identity = dict(self.identity)
        try:
            result._root_fd = result._absolute(self.root)
            result._lane_fd = result._open(self.lane, _DIR_FLAGS, result._root_fd)
            result.fd = os.dup(self.fd)
            result._take(result.fd)
            result.check()
            if [_identity(fd) for fd in (result._root_fd, result._lane_fd, result.fd)] != self.identity["inodes"]:
                raise LaneScratchError("lane_scratch_lifetime_identity_changed")
            return result
        except OSError:
            result.close()
            raise LaneScratchError("lane_scratch_lifetime_descriptor_invalid") from None
        except BaseException:
            result.close()
            raise

    @classmethod
    def inherited(cls, descriptor: int, identity: dict[str, Any], *, now: Callable[[], float] = time.time,
                  _owned_identity: tuple[int, int] | None = None) -> LeasedScratchUse:
        """Consume a controlled child's inherited close-only directory descriptor."""
        result = cls()
        result.fd = descriptor
        result._owned[descriptor] = _owned_identity
        try:
            if _owned_identity is None:
                result._take(descriptor)
            elif _identity(descriptor) != _owned_identity:
                raise LaneScratchError("lane_scratch_lifetime_identity_changed")
            if not stat.S_ISDIR(os.fstat(descriptor).st_mode) or not isinstance(identity, dict):
                raise LaneScratchError("lane_scratch_lifetime_descriptor_invalid")
            reference = {"run_ref", "scene_ref"} & identity.keys()
            if len(reference) != 1 or set(identity) != {"root", "lane", "name", "owner", "lease_digest", "consumer_lifetime_contract", "inodes"} | reference:
                raise LaneScratchError("lane_scratch_lifetime_identity_invalid")
            result.root = _path(identity["root"])
            result.lane, result.name = _id(identity["lane"], "lane"), _id(identity["name"], "name")
            _id(identity["owner"], "owner")
            _id(identity[next(iter(reference))], "reference")
            inodes = identity["inodes"]
            if (not isinstance(inodes, list) or len(inodes) != 3 or any(not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or any(type(v) is not int or not 0 <= v < 2**64 for v in pair) for pair in inodes)
                    or identity["consumer_lifetime_contract"] != PROTOCOL):
                raise LaneScratchError("lane_scratch_lifetime_identity_invalid")
            result.identity = {**identity, "inodes": [tuple(pair) for pair in inodes]}
            result.now, result.exclusive = now, False
            result._root_fd = result._absolute(result.root)
            result._lane_fd = result._open(result.lane, _DIR_FLAGS, result._root_fd)
            with result._root_lock():
                # Never change the inherited OFD's mode. A separate retained SH
                # establishes real authority even for an initially unlocked FD.
                inherited_fd = result.fd
                result.fd = result._open(result.name, _DIR_FLAGS, result._lane_fd)
                if _identity(inherited_fd) != _identity(result.fd):
                    raise LaneScratchError("lane_scratch_lifetime_identity_changed")
                try:
                    fcntl.flock(result.fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise LaneScratchError("lane_scratch_consumer_busy") from None
                result.check()
                if [_identity(fd) for fd in (result._root_fd, result._lane_fd, result.fd)] != result.identity["inodes"]:
                    raise LaneScratchError("lane_scratch_lifetime_identity_changed")
                lease = _read_lease(result.fd)
                if lease.get("released_at_epoch") is not None or lease["expires_at_epoch"] <= _now(now):
                    raise LaneScratchError("lane_scratch_lifetime_inactive")
            return result
        except (OSError, KeyError, TypeError, ValueError):
            result.close()
            raise LaneScratchError("lane_scratch_lifetime_descriptor_invalid") from None
        except BaseException:
            result.close()
            raise

    def _close_one(self, fd: int) -> None:
        if fd not in self._owned:
            return
        identity = self._owned[fd]
        if identity is None:
            # No original identity exists: never adopt/close a reused number.
            self._owned.pop(fd, None)
            self.unresolved_ownership = True
            return
        for _ in range(2):
            try:
                if _identity(fd) != identity:
                    self._owned.pop(fd, None)
                    return
                os.close(fd)
            except OSError as error:
                if error.errno == errno.EBADF:
                    self._owned.pop(fd, None)
                    return
                continue
            self._owned.pop(fd, None)
            return

    def _cleanup_status(self) -> None:
        if self.unresolved_ownership:
            raise LaneScratchError("lane_scratch_descriptor_ownership_unproven")

    def close(self) -> None:
        self._closed = True
        for fd in reversed(tuple(self._owned)):
            self._close_one(fd)
        self._cleanup_status()
        if self._owned:
            raise LaneScratchError("lane_scratch_descriptor_cleanup_failed")

    def __enter__(self) -> LeasedScratchUse:
        self.check()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


@contextmanager
def worker_output_lifetime(output: Path, use: LeasedScratchUse | None) -> Iterator[LeasedScratchUse | None]:
    """Refuse enrolled outputs without authority before legacy path operations."""
    if use is not None:
        if not isinstance(use, LeasedScratchUse) or use.exclusive:
            raise LaneScratchError("lane_scratch_worker_lifetime_invalid")
        with use.borrow(output) as borrowed:
            yield borrowed
        return
    # Only installed lane parents are enrollment namespaces. Non-lane legacy
    # paths retain their previous semantics and are never declared participants.
    root = next((root for root in LANE_ROOTS if output.is_relative_to(root)), None)
    if root is not None:
        output = _path(output)
        parts = output.relative_to(root).parts
        if len(parts) != 3:
            raise LaneScratchError("lane_scratch_worker_descendant_invalid")
        metadata = LeasedScratchUse()
        try:
            directory = metadata._absolute(root / parts[0] / parts[1])
            lease = _read_lease(directory)
            if "consumer_lifetime_contract" in lease:
                raise LaneScratchError("lane_scratch_worker_lifetime_required")
        except OSError:
            raise LaneScratchError("lane_scratch_worker_metadata_unreadable") from None
        finally:
            metadata.close()
    yield None
