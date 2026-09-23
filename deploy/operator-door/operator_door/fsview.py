"""Read-only file access confined to the configured roots.

Every path is resolved to its real location and must stay inside a read root
and outside every hidden path, so neither ``..`` nor a symlink can escape.
Files are opened with ``O_NOFOLLOW`` and must be regular. Names and content go
through the secret guard; an archive skips what the guard refuses and says so
in a manifest member instead of failing the whole download.

Two extra rules close what name and content patterns alone would miss:

* under a JSON-only root (``/etc/blueprint``, which holds about 150 env files and
  backups) only ``*.json`` files can be read;
* compressed content (gzip, zip, xz, bzip2) is scanned after decompression, with
  the verdict cached per file version; formats the standard library cannot
  decompress (zstd, 7z, rar) are refused.
"""

from __future__ import annotations

import bz2
import fnmatch
import gzip
import io
import json
import lzma
import os
import stat
import tarfile
import tempfile
import threading
import zipfile
from typing import Any, BinaryIO, Callable

from .config import DoorConfig
from .secrets_guard import refused_name, scan_bytes

MANIFEST_NAME = ".operator-door-manifest.json"
_SCAN_OVERLAP = 4096
_CHUNK = 1024 * 1024
_MAX_DECOMPRESSED = 2 * 1024 * 1024 * 1024
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x1f\x8b", "gzip"),
    (b"PK\x03\x04", "zip"),
    (b"\xfd7zXZ\x00", "xz"),
    (b"BZh", "bzip2"),
    (b"\x28\xb5\x2f\xfd", "zstd"),
    (b"7z\xbc\xaf\x27\x1c", "7z"),
    (b"Rar!", "rar"),
)
_STREAMABLE = {"gzip": gzip.GzipFile, "xz": lzma.LZMAFile, "bzip2": bz2.BZ2File}


class FsRefused(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def compression_kind(head: bytes) -> str | None:
    for magic, kind in _MAGIC:
        if head.startswith(magic):
            return kind
    return None


def _scan_stream(stream: BinaryIO, budget: list[int]) -> str | None:
    """Scan a decompressed stream in overlapping chunks; ``budget`` counts bytes left."""

    tail = b""
    while True:
        chunk = stream.read(_CHUNK)
        if not chunk:
            return None
        budget[0] -= len(chunk)
        if budget[0] < 0:
            raise FsRefused("compressed_too_large_to_scan")
        reason = scan_bytes(tail + chunk)
        if reason is not None:
            return reason
        tail = chunk[-_SCAN_OVERLAP:]


def _scan_compressed(fd: int, kind: str) -> str | None:
    """Verdict for a compressed file: a rule name, or None when it is clean."""

    budget = [_MAX_DECOMPRESSED]
    with os.fdopen(os.dup(fd), "rb") as raw:
        raw.seek(0)
        try:
            if kind in _STREAMABLE:
                with _STREAMABLE[kind](fileobj=raw) if kind == "gzip" else _STREAMABLE[kind](raw) as stream:
                    return _scan_stream(stream, budget)
            if kind == "zip":
                with zipfile.ZipFile(raw) as archive:
                    for info in archive.infolist():
                        if refused_name(info.filename):
                            return "zip_member_name"
                        with archive.open(info) as member:
                            head = member.read(8)
                            if compression_kind(head):
                                return "nested_compressed_member"
                            reason = scan_bytes(head) or _scan_stream(member, budget)
                            if reason is not None:
                                return reason
                    return None
        except (OSError, EOFError, zipfile.BadZipFile, lzma.LZMAError, ValueError, RuntimeError):
            return "compressed_unreadable"
    return f"compressed_unscannable:{kind}"


class FileView:
    def __init__(self, config: DoorConfig) -> None:
        self._config = config
        self._roots = [os.path.normpath(root) for root in config.read_roots]
        self._real_roots = [os.path.realpath(root) for root in config.read_roots]
        hidden = [os.path.normpath(path) for path in config.hidden_paths]
        self._hidden = sorted(set(hidden + [os.path.realpath(path) for path in hidden]))
        self._json_only = [os.path.realpath(root) for root in config.json_only_roots]
        self._verdicts: dict[tuple[str, int, int, int], str | None] = {}
        self._verdict_lock = threading.Lock()

    # -- resolution ---------------------------------------------------------

    def _is_hidden(self, path: str) -> bool:
        return any(_within(path, hidden) for hidden in self._hidden)

    def resolve(self, raw: str) -> str:
        if not isinstance(raw, str) or not raw.startswith("/"):
            raise FsRefused("path_not_absolute")
        if "\x00" in raw or len(raw) > 4096:
            raise FsRefused("path_invalid")
        lexical = os.path.normpath(raw)
        lexical_inside = any(_within(lexical, root) for root in self._roots + self._real_roots)
        if not lexical_inside:
            raise FsRefused("path_outside_roots")
        if self._is_hidden(lexical):
            raise FsRefused("path_hidden")
        real = os.path.realpath(lexical)
        if self._is_hidden(real):
            raise FsRefused("path_hidden")
        if not any(_within(real, root) for root in self._real_roots):
            raise FsRefused("path_symlink_escape")
        return real

    def _below_root(self, real: str) -> str:
        """The part of ``real`` under its read root; names are judged only there."""

        root = max((r for r in self._real_roots if _within(real, r)), key=len)
        return os.path.relpath(real, root)

    def _refuse_secret_name(self, real: str) -> None:
        if refused_name(self._below_root(real)):
            raise FsRefused("secret_name_refused")

    def _json_only_violation(self, real: str) -> bool:
        return any(_within(real, root) for root in self._json_only) and not real.endswith(".json")

    # -- listing ------------------------------------------------------------

    def _entry(self, directory: str, item: os.DirEntry[str]) -> dict[str, Any]:
        try:
            info = item.stat(follow_symlinks=False)
        except OSError:
            return {"name": item.name, "type": "unreadable"}
        kind = (
            "symlink" if stat.S_ISLNK(info.st_mode)
            else "dir" if stat.S_ISDIR(info.st_mode)
            else "file" if stat.S_ISREG(info.st_mode)
            else "other"
        )
        full = os.path.join(directory, item.name)
        if self._is_hidden(full):
            return {"name": item.name, "type": kind, "refused": "hidden"}
        if refused_name(item.name):
            return {"name": item.name, "type": kind, "refused": "secret_name"}
        if kind == "file" and self._json_only_violation(full):
            return {"name": item.name, "type": kind, "refused": "json_only_root"}
        entry: dict[str, Any] = {"name": item.name, "type": kind, "mtime": int(info.st_mtime)}
        if kind == "file":
            entry["size"] = info.st_size
        if kind == "symlink":
            try:
                entry["target"] = os.readlink(full)
            except OSError:
                entry["target"] = None
        return entry

    def list_dir(self, raw: str, *, sort: str = "name", match: str | None = None) -> dict[str, Any]:
        real = self.resolve(raw)
        self._refuse_secret_name(real)
        try:
            info = os.stat(real)
        except FileNotFoundError as error:
            raise FsRefused("not_found") from error
        if not stat.S_ISDIR(info.st_mode):
            if self._json_only_violation(real):
                raise FsRefused("json_only_root")
            kind = "file" if stat.S_ISREG(info.st_mode) else "other"
            return {"path": raw, "realpath": real, "type": kind, "size": info.st_size,
                    "mtime": int(info.st_mtime), "entries": None, "truncated": False}
        with os.scandir(real) as iterator:
            items = [item for item in iterator if match is None or fnmatch.fnmatch(item.name, match)]
        if sort == "mtime":
            def mtime(item: os.DirEntry[str]) -> float:
                try:
                    return item.stat(follow_symlinks=False).st_mtime
                except OSError:
                    return 0.0
            items.sort(key=mtime, reverse=True)
        else:
            items.sort(key=lambda item: item.name)
        limit = self._config.max_list_entries
        entries = [self._entry(real, item) for item in items[:limit]]
        return {"path": raw, "realpath": real, "type": "dir", "entries": entries,
                "total": len(items), "truncated": len(items) > limit}

    # -- reading ------------------------------------------------------------

    def _open_regular(self, real: str) -> tuple[int, os.stat_result]:
        try:
            fd = os.open(real, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))
        except FileNotFoundError as error:
            raise FsRefused("not_found") from error
        except IsADirectoryError as error:
            raise FsRefused("not_regular_file") from error
        except OSError as error:
            raise FsRefused("open_failed") from error
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            os.close(fd)
            raise FsRefused("not_regular_file")
        return fd, info

    def _compressed_verdict(self, real: str, fd: int, info: os.stat_result, kind: str) -> str | None:
        key = (real, info.st_ino, info.st_size, info.st_mtime_ns)
        with self._verdict_lock:
            if key in self._verdicts:
                return self._verdicts[key]
        verdict = _scan_compressed(fd, kind)
        with self._verdict_lock:
            if len(self._verdicts) >= 256:
                self._verdicts.pop(next(iter(self._verdicts)))
            self._verdicts[key] = verdict
        return verdict

    def _check_readable(self, raw: str) -> str:
        real = self.resolve(raw)
        self._refuse_secret_name(real)
        if self._json_only_violation(real):
            raise FsRefused("json_only_root")
        return real

    def read_range(self, raw: str, offset: int = 0, length: int | None = None) -> tuple[bytes, dict[str, Any]]:
        real = self._check_readable(raw)
        if offset < 0 or (length is not None and length < 0):
            raise FsRefused("range_invalid")
        fd, info = self._open_regular(real)
        try:
            kind = compression_kind(os.pread(fd, 8, 0))
            if kind is not None:
                verdict = self._compressed_verdict(real, fd, info, kind)
                if verdict is not None:
                    raise FsRefused(f"secret_content_refused:{verdict}")
            size = info.st_size
            wanted = self._config.max_read_bytes if length is None else min(length, self._config.max_read_bytes)
            start = min(offset, size)
            window_start = max(0, start - _SCAN_OVERLAP)
            os.lseek(fd, window_start, os.SEEK_SET)
            window = _read_exact(fd, (start - window_start) + wanted + _SCAN_OVERLAP)
        finally:
            os.close(fd)
        reason = scan_bytes(window)
        if reason is not None:
            raise FsRefused(f"secret_content_refused:{reason}")
        data = window[start - window_start:start - window_start + wanted]
        end = start + len(data)
        return data, {"path": raw, "realpath": real, "size": size, "offset": start,
                      "length": len(data), "eof": end >= size}

    # -- archives -----------------------------------------------------------

    def prepare_archive(self, raw: str) -> str:
        """Every refusal an archive can hit, raised before any response byte is sent."""

        real = self.resolve(raw)
        self._refuse_secret_name(real)
        if not os.path.isdir(real):
            raise FsRefused("not_a_directory")
        return real

    def stream_archive(self, raw: str, write: Callable[[bytes], Any]) -> dict[str, Any]:
        real = self.prepare_archive(raw)
        manifest: dict[str, Any] = {
            "schema": "blueprint_operator_door_archive_manifest.v1",
            "root": raw, "files_included": 0, "bytes_included": 0,
            "skipped": [], "truncated": False,
        }
        with tarfile.open(fileobj=_Sink(write), mode="w|gz") as archive:
            for directory, dirnames, filenames in os.walk(real, followlinks=False):
                relative_dir = os.path.relpath(directory, real)
                keep: list[str] = []
                for name in sorted(dirnames):
                    full = os.path.join(directory, name)
                    rel = os.path.normpath(os.path.join(relative_dir, name))
                    if os.path.islink(full):
                        manifest["skipped"].append({"path": rel, "reason": "symlink"})
                    elif self._is_hidden(full):
                        manifest["skipped"].append({"path": rel, "reason": "hidden"})
                    elif refused_name(name):
                        manifest["skipped"].append({"path": rel, "reason": "secret_name"})
                    else:
                        keep.append(name)
                dirnames[:] = keep
                for name in sorted(filenames):
                    full = os.path.join(directory, name)
                    rel = os.path.normpath(os.path.join(relative_dir, name))
                    reason = self._archive_member(archive, full, rel, manifest)
                    if reason is not None:
                        manifest["skipped"].append({"path": rel, "reason": reason})
                        if reason == "archive_budget_exceeded":
                            manifest["truncated"] = True
            payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            info = tarfile.TarInfo(MANIFEST_NAME)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        return manifest

    def _archive_member(
        self, archive: tarfile.TarFile, full: str, rel: str, manifest: dict[str, Any]
    ) -> str | None:
        if os.path.islink(full):
            return "symlink"
        if self._is_hidden(full):
            return "hidden"
        if refused_name(rel):
            return "secret_name"
        if self._json_only_violation(full):
            return "json_only_root"
        try:
            fd, info = self._open_regular(full)
        except FsRefused as refusal:
            return refusal.code
        try:
            if manifest["bytes_included"] + info.st_size > self._config.max_archive_bytes:
                return "archive_budget_exceeded"  # skip it; smaller files may still fit
            kind = compression_kind(os.pread(fd, 8, 0))
            if kind is not None:
                verdict = self._compressed_verdict(full, fd, info, kind)
                if verdict is not None:
                    return f"secret_content:{verdict}"
            # One read: the bytes scanned are exactly the bytes sent.
            with tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024) as copy:
                os.lseek(fd, 0, os.SEEK_SET)
                tail, copied = b"", 0
                while copied < info.st_size:
                    chunk = os.read(fd, min(_CHUNK, info.st_size - copied))
                    if not chunk:
                        break
                    reason = scan_bytes(tail + chunk)
                    if reason is not None:
                        return f"secret_content:{reason}"
                    copy.write(chunk)
                    copied += len(chunk)
                    tail = chunk[-_SCAN_OVERLAP:]
                copy.seek(0)
                member = tarfile.TarInfo(rel)
                member.size = copied
                member.mtime = int(info.st_mtime)
                member.mode = 0o644
                archive.addfile(member, copy)
            manifest["files_included"] += 1
            manifest["bytes_included"] += copied
            return None
        finally:
            os.close(fd)


class _Sink:
    """File-like adapter so ``tarfile`` can stream straight into a response."""

    def __init__(self, write: Callable[[bytes], Any]) -> None:
        self._write = write

    def write(self, data: bytes) -> int:
        self._write(bytes(data))
        return len(data)

    def flush(self) -> None:
        return None


def _read_exact(fd: int, count: int) -> bytes:
    parts: list[bytes] = []
    remaining = count
    while remaining > 0:
        chunk = os.read(fd, min(remaining, _CHUNK))
        if not chunk:
            break
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)
