"""Seal a remote stage's new output bytes and land only a selected subset (plan 14 §6, §9).

``index_tree`` records every path of a stage's output as ``{path, blob, size_bytes, mode,
origin}``.  A blob the host already holds, an input file or a member of an input zip, is
referenced by its origin and never archived, so only new bytes leave the worker.  The rest go
into ``blobs.tar`` once each, in digest order, with fixed metadata and recorded data offsets:
its size is known before upload and any blob can be read back by range.

``land_subset`` assembles the selected paths in a hidden sibling directory, hard-linking one
inode per ``(blob, mode)`` from verified host copies where it can, and renames the directory
into place, refusing to replace anything, only once it is complete.  A resumed landing accepts
identical files and refuses different ones.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import io
import json
import os
import re
import shutil
import stat
import sys
import tarfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any, BinaryIO

from .decision_evidence_contracts import canonical_digest
from .remote_cpu_job_contract import OUTPUT_FORMAT, RemoteCpuContractError, record_bytes, safe_label
from .remote_cpu_job_records import fsync_directory

INDEX_SCHEMA_VERSION = "remote_cpu_output_index.v1"
BLOCK = 512
END_OF_ARCHIVE = b"\0" * (2 * BLOCK)
MEMBER_MODE = 0o444
MAX_BLOB_BYTES = 8**11 - 1  # the ustar size field
_CHUNK = 1024 * 1024
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_MODE = re.compile(r"0[0-7]{3}")
_INDEX_KEYS = frozenset({
    "schema_version", "format", "root_mode", "directories", "entries", "blobs", "archive_size_bytes",
    "paths_total", "bytes_total", "host_known", "index_digest",
})
_ENTRY_KEYS = frozenset({"path", "blob", "size_bytes", "mode", "origin"})
_LINK_FALLBACK_ERRNOS = {errno.EXDEV, errno.EPERM, errno.EMLINK, errno.EACCES}
Reader = Callable[[int, int], BinaryIO]


class RemoteCpuArchiveError(ValueError):
    """An output tree, blob archive, index or landing is unsafe or does not match."""


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _relative(value: Any) -> bool:
    return (isinstance(value, str) and 0 < len(value) <= 4096 and "\x00" not in value
            and not value.startswith("/") and all(part not in {"", ".", ".."} for part in value.split("/")))


def _mode_ok(value: Any, *, directory: bool = False) -> bool:
    # Directories must stay owner-traversable so a landing can always be resumed.
    return (isinstance(value, str) and _MODE.fullmatch(value) is not None
            and (not directory or int(value, 8) & 0o700 == 0o700))


def _valid_origin(origin: Any, blob: str) -> bool:
    if origin == "archive":
        return True
    if isinstance(origin, Mapping) and set(origin) == {"input"}:
        return origin["input"] == blob
    member = origin.get("input_member") if isinstance(origin, Mapping) and set(origin) == {"input_member"} else None
    return (isinstance(member, Mapping) and set(member) == {"input", "member"} and _is_digest(member["input"])
            and _relative(member["member"]))


def _open_regular(path: Path) -> BinaryIO:
    stream = os.fdopen(os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)), "rb")
    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
        stream.close()
        raise RemoteCpuArchiveError(f"remote_cpu_archive_not_regular:{path.name}")
    return stream


def _hash_file(path: Path) -> tuple[str, int]:
    with _open_regular(path) as stream:
        digest, size = hashlib.sha256(), 0
        for chunk in iter(lambda: stream.read(_CHUNK), b""):
            digest.update(chunk)
            size += len(chunk)
    return "sha256:" + digest.hexdigest(), size


def _layout(sizes: Mapping[str, int]) -> tuple[list[dict[str, Any]], int]:
    """Archive blobs in digest order with their data offsets, and the total archive size."""

    rows, offset = [], 0
    for blob in sorted(sizes):
        size = sizes[blob]
        if size > MAX_BLOB_BYTES:
            raise RemoteCpuArchiveError(f"remote_cpu_archive_blob_too_large:{blob}")
        rows.append({"blob": blob, "size_bytes": size, "offset": offset + BLOCK})
        offset += BLOCK + size + (-size % BLOCK)
    return rows, offset + len(END_OF_ARCHIVE)


def _header(blob: str, size: int) -> bytes:
    info = tarfile.TarInfo(f"sha256/{blob[7:]}")
    info.size, info.mode, info.mtime, info.uid, info.gid, info.uname, info.gname = size, MEMBER_MODE, 0, 0, 0, "", ""
    return info.tobuf(tarfile.USTAR_FORMAT, "utf-8", "strict")


def _validated_host_known(host_known: Any) -> dict[str, Any]:
    if not isinstance(host_known, Mapping):
        raise RemoteCpuArchiveError("remote_cpu_archive_host_known_invalid")
    known: dict[str, Any] = {}
    for blob, origin in host_known.items():
        if not _is_digest(blob) or origin == "archive" or not _valid_origin(origin, blob):
            raise RemoteCpuArchiveError(f"remote_cpu_archive_host_known_invalid:{safe_label(blob)}")
        known[blob] = json.loads(json.dumps(origin))
    return known


def _walk(directory: Path, prefix: str, known: Mapping[str, Any], entries: list, directories: list) -> None:
    with os.scandir(directory) as iterator:
        children = sorted(iterator, key=lambda child: child.name)
    for child in children:
        relative = prefix + child.name
        try:
            relative.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise RemoteCpuArchiveError("remote_cpu_archive_name_invalid") from exc
        info = child.stat(follow_symlinks=False)
        mode = f"{stat.S_IMODE(info.st_mode):04o}"
        if stat.S_ISLNK(info.st_mode):
            raise RemoteCpuArchiveError(f"remote_cpu_archive_symlink_refused:{relative}")
        if stat.S_ISDIR(info.st_mode):
            if not _mode_ok(mode, directory=True):
                raise RemoteCpuArchiveError(f"remote_cpu_archive_mode_invalid:{relative}")
            directories.append({"path": relative, "mode": mode})
            _walk(Path(child.path), relative + "/", known, entries, directories)
        elif stat.S_ISREG(info.st_mode):
            if not _mode_ok(mode):
                raise RemoteCpuArchiveError(f"remote_cpu_archive_mode_invalid:{relative}")
            blob, size = _hash_file(Path(child.path))
            entries.append({"path": relative, "blob": blob, "size_bytes": size, "mode": mode,
                            "origin": known.get(blob, "archive")})
        else:
            raise RemoteCpuArchiveError(f"remote_cpu_archive_special_file_refused:{relative}")


def _index_body(root_mode: str, entries: list[dict[str, Any]], directories: list[dict[str, Any]]) -> dict[str, Any]:
    sizes: dict[str, int] = {}
    for entry in entries:
        if entry["origin"] == "archive" and sizes.setdefault(entry["blob"], entry["size_bytes"]) != entry["size_bytes"]:
            raise RemoteCpuArchiveError(f"remote_cpu_archive_index_invalid:blob_size:{entry['blob']}")
    blobs, total = _layout(sizes)
    known = [entry["size_bytes"] for entry in entries if entry["origin"] != "archive"]
    return {
        "schema_version": INDEX_SCHEMA_VERSION, "format": OUTPUT_FORMAT, "root_mode": root_mode,
        "directories": directories, "entries": entries, "blobs": blobs, "archive_size_bytes": total,
        "paths_total": len(entries), "bytes_total": sum(entry["size_bytes"] for entry in entries),
        "host_known": {"count": len(known), "bytes": sum(known)}, "index_digest": "",
    }


def index_tree(root: str | Path, *, host_known: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Index every file and directory under ``root``; host-known blobs keep their origin, never bytes."""

    base = Path(root)
    try:
        info = os.lstat(base)
    except OSError as exc:
        raise RemoteCpuArchiveError("remote_cpu_archive_root_invalid") from exc
    root_mode = f"{stat.S_IMODE(info.st_mode):04o}"
    if not stat.S_ISDIR(info.st_mode) or not _mode_ok(root_mode, directory=True):
        raise RemoteCpuArchiveError("remote_cpu_archive_root_invalid")
    entries: list[dict[str, Any]] = []
    directories: list[dict[str, Any]] = []
    _walk(base, "", _validated_host_known(host_known), entries, directories)
    index = _index_body(root_mode, sorted(entries, key=lambda row: row["path"]),
                        sorted(directories, key=lambda row: row["path"]))
    index["index_digest"] = canonical_digest(index, digest_field="index_digest")
    index_bytes(index)
    return index


def _validated_index(index: Any) -> dict[str, Any]:
    def invalid(detail: str) -> RemoteCpuArchiveError:
        return RemoteCpuArchiveError(f"remote_cpu_archive_index_invalid:{detail}")

    try:
        text = json.dumps(index, ensure_ascii=False, allow_nan=False)
        text.encode("utf-8")
        value = json.loads(text)
    except (TypeError, ValueError, RecursionError) as exc:  # UnicodeEncodeError is a ValueError
        raise invalid("not_json") from exc
    if not isinstance(value, dict) or set(value) != _INDEX_KEYS:
        raise invalid("keys")
    if value["index_digest"] != canonical_digest(value, digest_field="index_digest"):
        raise invalid("seal")
    directories, entries = value["directories"], value["entries"]
    if (not isinstance(directories, list) or not isinstance(entries, list)
            or not _mode_ok(value["root_mode"], directory=True)):
        raise invalid("structure")
    folders: list[str] = []
    for row in directories:
        if (not isinstance(row, dict) or set(row) != {"path", "mode"} or not _relative(row["path"])
                or not _mode_ok(row["mode"], directory=True) or row["path"].rpartition("/")[0] not in {"", *folders}):
            raise invalid("directories")
        folders.append(row["path"])
    paths = []
    for row in entries:
        if (not isinstance(row, dict) or set(row) != _ENTRY_KEYS or not _relative(row["path"])
                or not _is_digest(row["blob"]) or not _is_count(row["size_bytes"]) or not _mode_ok(row["mode"])
                or not _valid_origin(row["origin"], row["blob"])
                or row["path"].rpartition("/")[0] not in {"", *folders}):
            raise invalid("entries")
        paths.append(row["path"])
    if folders != sorted(set(folders)) or paths != sorted(set(paths)) or set(paths) & set(folders):
        raise invalid("order")
    expected = _index_body(value["root_mode"], entries, directories)
    if {**expected, "index_digest": value["index_digest"]} != value:
        raise invalid("derived_fields")
    return value


def index_bytes(index: Mapping[str, Any]) -> bytes:
    """Canonical ``index.json`` bytes, after the no-URL and no-credential guard."""

    try:
        return record_bytes(_validated_index(index))
    except RemoteCpuContractError as exc:
        raise RemoteCpuArchiveError(f"remote_cpu_archive_index_forbidden_content:{exc}") from exc


def blobs_tar_size(index: Mapping[str, Any]) -> int:
    """The exact ``blobs.tar`` Content-Length, known before any byte is written."""

    return _validated_index(index)["archive_size_bytes"]


def write_blobs_tar(root: str | Path, index: Mapping[str, Any], stream: BinaryIO) -> dict[str, Any]:
    """Stream each archive-origin blob once, hashing on the fly; a changed file fails closed.

    ``stream`` must accept whole writes (a buffered file or an upload body of known length)."""

    value = _validated_index(index)
    sources: dict[str, str] = {}
    for entry in value["entries"]:
        if entry["origin"] == "archive":
            sources.setdefault(entry["blob"], entry["path"])
    digest, written = hashlib.sha256(), 0

    def emit(data: bytes) -> None:
        nonlocal written
        stream.write(data)
        digest.update(data)
        written += len(data)

    for row in value["blobs"]:
        emit(_header(row["blob"], row["size_bytes"]))
        blob_digest, copied = hashlib.sha256(), 0
        with _open_regular(Path(root) / sources[row["blob"]]) as source:
            for chunk in iter(lambda: source.read(_CHUNK), b""):
                copied += len(chunk)
                if copied > row["size_bytes"]:
                    break
                blob_digest.update(chunk)
                emit(chunk)
        if copied != row["size_bytes"] or "sha256:" + blob_digest.hexdigest() != row["blob"]:
            raise RemoteCpuArchiveError(f"remote_cpu_archive_blob_changed:{row['blob']}")
        emit(b"\0" * (-row["size_bytes"] % BLOCK))
    emit(END_OF_ARCHIVE)
    if written != value["archive_size_bytes"]:
        raise RemoteCpuArchiveError("remote_cpu_archive_size_mismatch")
    return {"digest": "sha256:" + digest.hexdigest(), "size_bytes": written}


def _read_exact(stream: BinaryIO, count: int) -> bytes:
    data = bytearray()
    while len(data) < count:
        chunk = stream.read(count - len(data))
        if not chunk:
            break
        data.extend(chunk)
    return bytes(data)


def verify_blobs_stream(stream: BinaryIO, index: Mapping[str, Any], *, expected_digest: str) -> dict[str, Any]:
    """One streaming pass: whole-object digest, tar framing, and every blob's digest and offset."""

    value = _validated_index(index)
    names = {f"sha256/{row['blob'][7:]}".encode() for row in value["blobs"]}
    digest, position = hashlib.sha256(), 0

    def take(count: int) -> bytes:
        nonlocal position
        data = _read_exact(stream, count)
        if len(data) != count:
            raise RemoteCpuArchiveError("remote_cpu_archive_truncated")
        digest.update(data)
        position += count
        return data

    for row in value["blobs"]:
        header, expected = take(BLOCK), _header(row["blob"], row["size_bytes"])
        name = header[:100].rstrip(b"\0")
        if name != expected[:100].rstrip(b"\0"):
            if not any(header) or name in names:
                raise RemoteCpuArchiveError(f"remote_cpu_archive_blob_missing:{row['blob']}")
            raise RemoteCpuArchiveError(f"remote_cpu_archive_unindexed_blob:{name.decode('utf-8', 'replace')}")
        if header != expected or position != row["offset"]:
            raise RemoteCpuArchiveError("remote_cpu_archive_framing_invalid:header")
        blob_digest, remaining = hashlib.sha256(), row["size_bytes"]
        while remaining:
            chunk = take(min(_CHUNK, remaining))
            blob_digest.update(chunk)
            remaining -= len(chunk)
        if "sha256:" + blob_digest.hexdigest() != row["blob"]:
            raise RemoteCpuArchiveError(f"remote_cpu_archive_blob_digest_mismatch:{row['blob']}")
        if any(take(-row["size_bytes"] % BLOCK)):
            raise RemoteCpuArchiveError("remote_cpu_archive_framing_invalid:padding")
    end = take(len(END_OF_ARCHIVE))
    if end != END_OF_ARCHIVE:
        name = end[:100].rstrip(b"\0")
        if name.startswith(b"sha256/"):
            raise RemoteCpuArchiveError(f"remote_cpu_archive_unindexed_blob:{name.decode('utf-8', 'replace')}")
        raise RemoteCpuArchiveError("remote_cpu_archive_framing_invalid:end_of_archive")
    if stream.read(1):
        raise RemoteCpuArchiveError("remote_cpu_archive_trailing_bytes")
    if "sha256:" + digest.hexdigest() != expected_digest:
        raise RemoteCpuArchiveError("remote_cpu_archive_digest_mismatch")
    return {"digest": expected_digest, "size_bytes": position, "blob_count": len(value["blobs"])}


def _validated_selectors(selectors: Any) -> tuple[str, ...]:
    """``**`` (everything), an exact relative path, or a directory as ``<dir>/**``; no other globs."""

    if isinstance(selectors, (str, bytes)) or not isinstance(selectors, Sequence) or not selectors:
        raise RemoteCpuArchiveError("remote_cpu_landing_selector_invalid")
    for selector in selectors:
        base = selector[:-3] if isinstance(selector, str) and selector.endswith("/**") else selector
        if selector != "**" and (not _relative(base) or any(character in base for character in "*?[]")):
            raise RemoteCpuArchiveError("remote_cpu_landing_selector_invalid")
    return tuple(selectors)


def _selected(path: str, selectors: Sequence[str]) -> bool:
    return any(
        selector == "**" or path == selector
        or (selector.endswith("/**") and (path == selector[:-3] or path.startswith(selector[:-2])))
        for selector in selectors
    )


def _listing(root: Path) -> tuple[dict[str, os.stat_result], dict[str, int]]:
    """Every file and directory under ``root``; a link or special file is a conflict."""

    files: dict[str, os.stat_result] = {}
    folders: dict[str, int] = {}
    pending = [(root, "")]
    while pending:
        directory, prefix = pending.pop()
        with os.scandir(directory) as iterator:
            for child in iterator:
                relative, info = prefix + child.name, child.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    folders[relative] = stat.S_IMODE(info.st_mode)
                    pending.append((Path(child.path), relative + "/"))
                elif stat.S_ISREG(info.st_mode):
                    files[relative] = info
                else:
                    raise RemoteCpuArchiveError(f"remote_cpu_landing_conflict:{relative}")
    return files, folders


def _file_matches(path: Path, row: Mapping[str, Any]) -> bool:
    info = os.lstat(path)
    return (stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == int(row["mode"], 8)
            and info.st_size == row["size_bytes"] and _hash_file(path)[0] == row["blob"])


def _verify_landed(root: Path, entries: Sequence[Mapping[str, Any]], folders: Mapping[str, str], root_mode: str) -> None:
    """An already-landed directory must hold exactly the selection, byte for byte and mode for mode."""

    info = os.lstat(root)
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != int(root_mode, 8):
        raise RemoteCpuArchiveError(f"remote_cpu_landing_conflict:{root.name}")
    files, found = _listing(root)
    wanted = {row["path"]: row for row in entries}
    mismatched = sorted(set(files) ^ set(wanted)) + sorted(set(found) ^ set(folders)) + [
        path for path, row in sorted(wanted.items()) if path in files and not _file_matches(root / path, row)
    ] + [path for path, mode in sorted(folders.items()) if found.get(path) != int(mode, 8)]
    if mismatched:
        raise RemoteCpuArchiveError(f"remote_cpu_landing_conflict:{mismatched[0]}")


def _rename_no_replace(source: Path, target: Path) -> None:
    """Atomically rename within one directory, refusing to replace an existing target."""

    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        function, flag = libc.renameatx_np, 0x00000004  # RENAME_EXCL
    elif sys.platform.startswith("linux"):
        function, flag = libc.renameat2, 1  # RENAME_NOREPLACE
    else:
        raise RemoteCpuArchiveError("remote_cpu_landing_rename_unsupported")
    function.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    function.restype = ctypes.c_int
    directory = os.open(source.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        if function(directory, os.fsencode(source.name), directory, os.fsencode(target.name), flag) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(target))
    finally:
        os.close(directory)


class _Assembly:
    """Place selected blobs in the landing directory, one inode per ``(blob, mode)``."""

    def __init__(self, *, reader: Reader, offsets: Mapping[str, int], host_sources: Mapping[str, Any],
                 member_store: Any, landing: Path, scratch: Path) -> None:
        self.reader, self.offsets, self.landing, self.scratch = reader, offsets, landing, scratch
        self.host_sources = {str(key): Path(value) for key, value in host_sources.items()}
        self.member_store = None if member_store is None else Path(member_store)
        self.placed: dict[tuple[str, str], Path] = {}
        self.verified: set[tuple[Path, str]] = set()
        self.resumed = self.linked_from_host = 0

    def place(self, row: Mapping[str, Any]) -> None:
        target, key = self.landing / row["path"], (row["blob"], row["mode"])
        if os.path.lexists(target):
            if not _file_matches(target, row):
                raise RemoteCpuArchiveError(f"remote_cpu_landing_conflict:{row['path']}")
            self.placed.setdefault(key, target)
            self.resumed += 1
            return
        if key in self.placed:
            os.link(self.placed[key], target)
            return
        host = self._host_copy(row)
        if host is not None and stat.S_IMODE(os.lstat(host).st_mode) == int(row["mode"], 8):
            try:
                os.link(host, target)
            except OSError as exc:
                if exc.errno not in _LINK_FALLBACK_ERRNOS:
                    raise
            else:
                self.placed[key] = target
                self.linked_from_host += 1
                return
        with ExitStack() as stack:
            self._copy(self._source(row, host, stack), row, target)
        self.placed[key] = target

    def _verified(self, path: Path, row: Mapping[str, Any], reason: str) -> Path:
        if (path, row["blob"]) not in self.verified:
            try:
                info = os.lstat(path)
                matches = (stat.S_ISREG(info.st_mode) and info.st_size == row["size_bytes"]
                           and _hash_file(path)[0] == row["blob"])
            except (OSError, RemoteCpuArchiveError):
                matches = False
            if not matches:
                raise RemoteCpuArchiveError(f"{reason}:{row['blob']}")
            self.verified.add((path, row["blob"]))
        return path

    def _host_copy(self, row: Mapping[str, Any]) -> Path | None:
        origin = row["origin"]
        if origin == "archive":
            return None
        if "input" in origin:
            path = self.host_sources.get(origin["input"])
            if path is None:
                raise RemoteCpuArchiveError(f"remote_cpu_landing_host_source_invalid:{row['blob']}")
            return self._verified(path, row, "remote_cpu_landing_host_source_invalid")
        stored = None if self.member_store is None else self.member_store / row["blob"][7:]
        if stored is not None and os.path.lexists(stored):
            return self._verified(stored, row, "remote_cpu_landing_member_store_invalid")
        return None

    def _source(self, row: Mapping[str, Any], host: Path | None, stack: ExitStack) -> BinaryIO:
        blob = row["blob"]
        try:
            if host is not None:
                return stack.enter_context(_open_regular(host))
            if row["origin"] == "archive":
                if row["size_bytes"] == 0:
                    return io.BytesIO(b"")
                stream = self.reader(self.offsets[blob], row["size_bytes"])
                if callable(getattr(stream, "close", None)):
                    stack.callback(stream.close)
                return stream
            member = row["origin"]["input_member"]
            bundle_path = self.host_sources.get(member["input"])
            if bundle_path is None:
                raise RemoteCpuArchiveError(f"remote_cpu_landing_host_source_invalid:{member['input']}")
            bundle = stack.enter_context(zipfile.ZipFile(bundle_path))
            info = bundle.getinfo(member["member"])
            if info.file_size != row["size_bytes"]:
                raise RemoteCpuArchiveError(f"remote_cpu_landing_blob_mismatch:{blob}")
            return stack.enter_context(bundle.open(info))
        except RemoteCpuArchiveError:
            raise
        except Exception as exc:  # noqa: BLE001 - range readers and zip members fail in many shapes
            raise RemoteCpuArchiveError(f"remote_cpu_landing_source_unavailable:{blob}") from exc

    def _copy(self, source: BinaryIO, row: Mapping[str, Any], target: Path) -> None:
        partial = self.scratch / f"{row['blob'][7:]}-{row['mode']}.partial"
        digest, copied = hashlib.sha256(), 0
        descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            with os.fdopen(descriptor, "wb") as output:
                try:
                    while copied < row["size_bytes"]:
                        chunk = source.read(min(_CHUNK, row["size_bytes"] - copied))
                        if not chunk:
                            break
                        output.write(chunk)
                        digest.update(chunk)
                        copied += len(chunk)
                    overflow = source.read(1)
                except Exception as exc:  # noqa: BLE001 - a broken read is an unavailable source
                    raise RemoteCpuArchiveError(f"remote_cpu_landing_source_unavailable:{row['blob']}") from exc
                if copied != row["size_bytes"] or overflow or "sha256:" + digest.hexdigest() != row["blob"]:
                    raise RemoteCpuArchiveError(f"remote_cpu_landing_blob_mismatch:{row['blob']}")
                output.flush()
                os.fchmod(output.fileno(), int(row["mode"], 8))
                os.fsync(output.fileno())
            os.link(partial, target)
        finally:
            partial.unlink(missing_ok=True)


def land_subset(
    *,
    index: Mapping[str, Any],
    reader: Reader,
    host_sources: Mapping[str, Any],
    destination_root: str | Path,
    selectors: Sequence[str],
    member_store: str | Path | None,
) -> dict[str, Any]:
    """Land the selected paths of a sealed output at ``destination_root``, atomically.

    ``reader(offset, length)`` returns a stream of exactly that byte range of ``blobs.tar``;
    ``host_sources`` maps an input digest to its immutable host copy (a file, or a zip whose
    members are extracted and verified); ``member_store`` is the shared member store, whose
    ``<hex>`` files are hard-linked when their mode matches.
    """

    value = _validated_index(index)
    patterns = _validated_selectors(selectors)
    entries = [row for row in value["entries"] if _selected(row["path"], patterns)]
    if not entries:
        raise RemoteCpuArchiveError("remote_cpu_landing_selection_empty")
    wanted = {row["path"] for row in value["directories"] if _selected(row["path"], patterns)}
    for row in entries:
        parts = row["path"].split("/")[:-1]
        wanted.update("/".join(parts[:depth]) for depth in range(1, len(parts) + 1))
    folders = {row["path"]: row["mode"] for row in value["directories"] if row["path"] in wanted}
    final = Path(destination_root)
    try:
        parent_info = os.lstat(final.parent)
    except OSError:
        parent_info = None
    if parent_info is None or not stat.S_ISDIR(parent_info.st_mode):
        raise RemoteCpuArchiveError("remote_cpu_landing_destination_parent_invalid")
    summary = {"directory": str(final), "paths": len(entries), "bytes": sum(row["size_bytes"] for row in entries),
               "resumed_paths": 0, "linked_from_host": 0}
    if os.path.lexists(final):
        try:
            _verify_landed(final, entries, folders, value["root_mode"])
        except OSError as exc:
            raise RemoteCpuArchiveError(f"remote_cpu_landing_conflict:{final.name}") from exc
        return {"state": "already_landed", **summary}
    nonce = hashlib.sha256("\n".join([value["index_digest"], *sorted(patterns)]).encode()).hexdigest()[:16]
    landing = final.with_name(f".{final.name}.landing-{nonce}")
    scratch = final.with_name(f".{final.name}.landing-{nonce}.partial")
    try:
        if os.path.lexists(landing) and not stat.S_ISDIR(os.lstat(landing).st_mode):
            raise RemoteCpuArchiveError(f"remote_cpu_landing_conflict:{landing.name}")
        # Index directory modes always keep owner rwx, so a resumed landing stays writable.
        landing.mkdir(mode=0o700, exist_ok=True)
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(mode=0o700)
        for path in sorted(folders):
            (landing / path).mkdir(mode=0o700, exist_ok=True)
        assembly = _Assembly(reader=reader, offsets={row["blob"]: row["offset"] for row in value["blobs"]},
                             host_sources=host_sources, member_store=member_store, landing=landing, scratch=scratch)
        for row in entries:
            assembly.place(row)
        files, found = _listing(landing)
        stray = sorted(set(files) - {row["path"] for row in entries}) + sorted(set(found) - set(folders))
        if stray:
            raise RemoteCpuArchiveError(f"remote_cpu_landing_conflict:{stray[0]}")
        for path in sorted(folders, key=lambda item: item.count("/"), reverse=True):
            os.chmod(landing / path, int(folders[path], 8))
            fsync_directory(landing / path)
        os.chmod(landing, int(value["root_mode"], 8))
        fsync_directory(landing)
        try:
            _rename_no_replace(landing, final)
        except OSError as exc:
            if exc.errno in {errno.EEXIST, errno.ENOTEMPTY}:
                raise RemoteCpuArchiveError(f"remote_cpu_landing_conflict:{final.name}") from exc
            raise
        fsync_directory(final.parent)
    except OSError as exc:
        raise RemoteCpuArchiveError(f"remote_cpu_landing_io_failed:errno_{exc.errno}") from exc
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return {"state": "landed", **summary, "resumed_paths": assembly.resumed,
            "linked_from_host": assembly.linked_from_host}
