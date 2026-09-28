"""Finite enrolled checkpoint transfer under the held original cache lifetime."""
from __future__ import annotations

import hashlib
import os
import secrets
import threading
from pathlib import PurePosixPath
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

from . import control_plane_lane_owner_consents as owners
from .control_plane_registered_checkpoint_cache import (
    NeededCheckpointCacheError, _QUANTUM, _require, require_cache_use,
)

def download_file(self, row):
    from .native_g1_checkpoint_cache import _fetcher
    from urllib.parse import quote
    require_cache_use(self)
    _require(self._mode == "fill" and self._reservation is not None and not self._reservation.released,
             "needed_cache_write_unadmitted")
    fetcher = _fetcher()
    self.check()
    parent, opened = self._root_fd, []
    parts = PurePosixPath(row["relative_path"]).parts
    fd, temporary, published = None, None, False
    try:
        for part in parts[:-1]:
            self.check()
            self._files.location(parent)
            try:
                os.stat(part, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                self.check()
                os.mkdir(part, 0o750, dir_fd=parent)
            child = self._files.open(parent, part, os.O_RDONLY | os.O_DIRECTORY)
            info = self._files.proof(child)
            _require(info.st_uid == 0 and info.st_dev == self._files.proof(self._root_fd).st_dev,
                     "needed_cache_directory_unsafe")
            if info.st_gid != self._gid:
                self.check()
                os.fchown(child, 0, self._gid)
                self._files.bindings[child] = (parent, part, owners._security(os.stat(part, dir_fd=parent, follow_symlinks=False)))
            opened.append(child)
            parent = child
        temporary = ".needed-payload-" + secrets.token_hex(16)
        self.check()
        fd = self._files.open(parent, temporary, os.O_RDWR | os.O_CREAT | os.O_EXCL)
        if self._files.proof(fd).st_gid != self._gid:
            self.check()
            os.fchown(fd, 0, self._gid)
            self._files.bindings[fd] = (parent, temporary, owners._security(os.stat(temporary, dir_fd=parent, follow_symlinks=False)))
        url = fetcher.MODEL_BASE + quote(row["relative_path"], safe="/")
        ranged = row["size_bytes"] >= fetcher.RANGED_DOWNLOAD_MIN_BYTES
        if ranged:
            self._active_downloads = {fd: row}
            size = fetcher._download_pinned_ranges(url, fd, row["size_bytes"], _cache_use=self)
        else:
            self.check()
            response = fetcher._open_https(url)
            digest, size = hashlib.sha256(), 0
            with _response_scope(self, response):
                self.check()
                _require(response.geturl().startswith("https://"), "needed_cache_insecure_redirect")
                while size < row["size_bytes"]:
                    self.check()
                    remaining = min(_QUANTUM-size % _QUANTUM, row["size_bytes"]-size)
                    key = ("network", row["relative_path"], size // _QUANTUM)
                    self._windows[key] = self._windows.get(key, 0)+1
                    _require(self._windows[key] <= 8, "needed_cache_fragment_limit")
                    block = response.read(remaining)
                    self.check()
                    _require(type(block) is bytes and 0 < len(block) <= remaining, "needed_cache_download_short")
                    offset = 0
                    while offset < len(block):
                        self.check()
                        self._files.location(fd)
                        info = self._files.proof(fd)
                        _require(info.st_size == size+offset and info.st_nlink == 1, "needed_cache_temp_changed")
                        wkey = ("write", row["relative_path"], (size+offset)//_QUANTUM)
                        self._windows[wkey] = self._windows.get(wkey, 0)+1
                        _require(self._windows[wkey] <= 8, "needed_cache_fragment_limit")
                        written = os.pwrite(fd, block[offset:], size+offset)
                        self.check()
                        _require(type(written) is int and 0 < written <= len(block)-offset, "needed_cache_write_short")
                        offset += written
                    size += len(block)
                    digest.update(block)
                    for role in ("network", "write"):
                        counts = self._counts.setdefault(role, dict(bytes=0, calls=0))
                        counts["bytes"] += len(block)
                        counts["calls"] += 1
                        _require(counts["bytes"] <= self._resources["logical_bytes"], "needed_cache_role_byte_limit")
                self.check()
                _require(response.read(1) == b"", "needed_cache_download_extra")
                self.check()
            _require(size == row["size_bytes"] and "sha256:"+digest.hexdigest() == row["sha256"],
                     "needed_cache_download_hash_changed")
        self.check()
        self._files.location(fd)
        os.fsync(fd)
        self.check()
        # Whole temporary readback is a distinct conserved hash pass before publication.
        digest, offset = hashlib.sha256(), 0
        while offset < size:
            self.check()
            self._files.location(fd)
            requested = min(_QUANTUM-offset % _QUANTUM, size-offset)
            _window(self, row, "fill_hash", offset, requested)
            part = os.pread(fd, requested, offset)
            self.check()
            _require(type(part) is bytes and 0 < len(part) <= requested, "needed_cache_temp_short")
            digest.update(part)
            offset += len(part)
            _debit(self, "fill_hash", len(part))
        _require("sha256:"+digest.hexdigest() == row["sha256"], "needed_cache_temp_hash_changed")
        self.check()
        self._files.location(fd)
        self._files.location(parent)
        os.link(temporary, parts[-1], src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
        published = True
        self.check()
        _require(owners._metadata(self._files.proof(fd)) == owners._metadata(os.stat(parts[-1], dir_fd=parent, follow_symlinks=False))
                 and self._files.proof(fd).st_nlink == 2, "needed_cache_payload_publication_changed")
        self._files.location(fd)
        os.unlink(temporary, dir_fd=parent)
        temporary = None
        self._files.bindings[fd] = (parent, parts[-1], owners._security(os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)))
        self.check()
        self._files.location(parent)
        os.fsync(parent)
        self.check()
    except (OSError, ValueError) as exc:
        self._failure = str(exc) if isinstance(exc, NeededCheckpointCacheError) else "needed_cache_transfer_failed"
        raise NeededCheckpointCacheError(self._failure) from None
    finally:
        self._active_downloads = {}
        if fd is not None:
            if temporary is not None and not published:
                try:
                    self._files.location(fd)
                    self._files.location(parent)
                    os.unlink(temporary, dir_fd=parent)
                except (OSError, ValueError):
                    pass
            self._files.close(fd)
        for directory in reversed(opened):
            self._files.close(directory)
        _require(not self._files.unresolved, "needed_cache_cleanup_incomplete")




def _window(use, row, role, offset, amount):
    """A positive progress syscall consumes its original role/window, never refunded."""
    with use._lock:
        key = (role, row['relative_path'], offset // _QUANTUM)
        use._windows[key] = use._windows.get(key, 0) + 1
        _require(use._windows[key] <= 8, 'needed_cache_fragment_limit')
        counts = use._counts.setdefault(role, dict(bytes=0, calls=0))
        counts['calls'] += 1
        _require(0 < amount <= _QUANTUM-offset % _QUANTUM, 'needed_cache_transfer_quantum_invalid')


def _debit(use, role, size):
    with use._lock:
        counts = use._counts.setdefault(role, dict(bytes=0, calls=0))
        counts['bytes'] += size
        _require(counts['bytes'] <= use._resources['logical_bytes'], 'needed_cache_role_byte_limit')


def download_pinned_ranges(use, url, descriptor, expected_size, *, chunk_size, workers,
                           deadline_seconds, cancel_event=None):
    """Actual native range seam; all workers join before any READY publication."""
    from .native_g1_checkpoint_cache import _fetcher
    require_cache_use(use)
    row = getattr(use, '_active_downloads', {}).get(descriptor)
    _require(row is not None and expected_size == row['size_bytes'] and type(chunk_size) is int
             and 0 < chunk_size <= 128*_QUANTUM and type(workers) is int and 1 <= workers <= 8
             and type(deadline_seconds) in (int, float) and 0 < deadline_seconds <= 1800,
             'needed_cache_range_unbound')
    use.check()
    use._files.location(descriptor)
    _require(use._files.proof(descriptor).st_size == 0
             and use._files.proof(descriptor).st_nlink == 1, 'needed_cache_temp_changed')
    os.ftruncate(descriptor, expected_size)
    use.check()
    deadline = min(use._deadline, use._monotonic()+deadline_seconds)
    cancelled = threading.Event()
    fetcher = _fetcher()

    def guard():
        use.check()
        _require(not cancelled.is_set() and (cancel_event is None or not cancel_event.is_set()),
                 'needed_cache_cancelled')
        _require(use._monotonic() < deadline, 'needed_cache_file_deadline_exceeded')
        use._files.location(descriptor)
        _require(use._files.proof(descriptor).st_size == expected_size
                 and use._files.proof(descriptor).st_nlink == 1, 'needed_cache_temp_changed')

    def fetch(start, end):
        written, requests, stalls = 0, 0, 0
        while start+written <= end:
            guard()
            _require(requests < 16 and stalls < 3, 'needed_cache_range_truncated')
            requests += 1
            before = written
            request_start = start+written
            guard()
            try:
                response = fetcher._open_https(url, headers={'Range': f'bytes={request_start}-{end}'})
            except OSError:
                guard()
                stalls += 1
                continue
            try:
                with _response_scope(use, response):
                    guard()
                    _require(response.status == 206 and response.headers.get('Content-Range') ==
                             f'bytes {request_start}-{end}/{expected_size}'
                             and response.geturl().startswith('https://'), 'needed_cache_range_response_invalid')
                    while start+written <= end:
                        guard()
                        offset = start+written
                        amount = min(_QUANTUM-offset % _QUANTUM, end-offset+1)
                        _window(use, row, 'network', offset, amount)
                        block = response.read(amount)
                        guard()
                        _require(type(block) is bytes and len(block) <= amount, 'needed_cache_range_read_invalid')
                        if not block:
                            break
                        _debit(use, 'network', len(block))
                        consumed = 0
                        while consumed < len(block):
                            guard()
                            offset = start+written+consumed
                            _window(use, row, 'write', offset, len(block)-consumed)
                            count = os.pwrite(descriptor, block[consumed:], offset)
                            guard()
                            _require(type(count) is int and 0 < count <= len(block)-consumed,
                                     'needed_cache_range_write_invalid')
                            _debit(use, 'write', count)
                            consumed += count
                        written += len(block)
                    if start+written == end+1:
                        guard()
                        _require(response.read(1) == b'', 'needed_cache_range_extra')
                        guard()
                        return written
            finally:
                guard()
            stalls = 0 if written > before else stalls+1
        raise NeededCheckpointCacheError('needed_cache_range_truncated')

    intervals = ((start, min(start+chunk_size, expected_size)-1)
                 for start in range(0, expected_size, chunk_size))
    futures, total = [], 0
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch, start, end) for start, end in intervals]
            for completed in as_completed(futures):
                try:
                    total += completed.result()
                except BaseException:
                    cancelled.set()
                    raise
        guard()
        _require(total == expected_size, 'needed_cache_range_size_invalid')
        return total
    except (OSError, ValueError) as exc:
        # The joined executor has finished before this sticky refusal escapes.
        use._failure = str(exc) if isinstance(exc, NeededCheckpointCacheError) else 'needed_cache_range_failed'
        raise NeededCheckpointCacheError(use._failure) from None


@contextmanager
def _response_scope(use, response):
    """Original native object ownership; failed close never becomes a closure grant."""
    with use._lock:
        use._native_pending += 1
    try:
        yield response
    finally:
        failure = None
        try:
            response.close()
            _require(response.closed is True, 'needed_cache_native_cleanup_incomplete')
        except BaseException:
            # Even interrupted native finalization cannot prove this owned response closed.
            failure = 'needed_cache_native_cleanup_incomplete'
        finally:
            with use._lock:
                use._native_pending -= 1
                if failure:
                    use._native_unknown += 1
                    use._failure = failure
        if failure:
            raise NeededCheckpointCacheError(failure)
