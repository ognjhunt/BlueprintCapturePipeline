"""Compress the published scene-configuration toolchain once per exact content.

Every Website scene-configuration activation ships the same provider toolchain
(on 2026-10-31 about 10.8 GB, mostly an unpacked CUDA Torch runtime) inside its
per-job provider bundle. Copying, re-hashing and deflating that toolchain again
for every job, retry and release took about 35 of an activation's 66 minutes.

This module deflates the toolchain members once, into a ZIP whose local entries
are spliced raw into each job's bundle. The cache key is the toolchain's
content inventory (path, sha256, size and executable bit of every file) plus the
archive format, never its release ``source_commit``: the toolchain builder already
hard-links identical files across releases, so a release that does not change
toolchain content reuses the archive. The per-release toolchain manifest,
which carries ``source_commit``, is not cached; it is compressed fresh into every
bundle.

Provenance is unchanged. The bundle has the same members with the same bytes, so
its sha256 and internal manifest are what the receipt, the immutable manifest and
the paid authority already bind, and the provider still validates every toolchain
byte against its manifest before running. The archive is built only from source
bytes hashed against the inventory as they are read. A reused archive is hashed
in full as it is spliced, and a mismatch discards it and rebuilds it. A hit and
a miss give byte-identical bundles.

Only the large third-party trees are cached (``CACHED_TREES``). Every other
toolchain file, the release layer, is staged fresh into each bundle, hashed
against the same manifest. capture-447ae4f4, 2026-11-01: five small files in
the toolchain record the release commit (each component package seal and two
source receipts), so a key over the whole inventory changed on every release
and every activation rebuilt the 6 GB archive (about 26 minutes).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import stat
import zipfile
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from .decision_evidence_contracts import canonical_digest, canonical_json

SCHEMA_VERSION = "task_evaluation_scene_configuration_toolchain_archive.v2"
CACHE_ROOT_ENV = "BLUEPRINT_SCENE_CONFIGURATION_TOOLCHAIN_ARCHIVE_CACHE_ROOT"
#: Host-wide default, used only where the control-plane inputs volume exists.
DEFAULT_CACHE_PARENT = Path("/var/lib/blueprint/task-evaluation-inputs")
DEFAULT_CACHE_NAME = "scene-configuration-toolchain-archives"
ARCHIVE_PREFIX = "provider_runtime/toolchain/"
FRAGMENT_NAME = "toolchain_members.zip"
INDEX_NAME = f"{SCHEMA_VERSION}.json"
#: Each entry is about 6 GB on 2026-10-31. Two cover a release that changes the
#: toolchain while an activation still splices the previous one.
RETAINED_ENTRIES = 2
_CHUNK = 1024 * 1024
_DATE_TIME = (1980, 1, 1, 0, 0, 0)
#: Third-party trees inside each ``components/<adapter>/package`` that hold
#: nearly every toolchain byte and change only with their own sources: Blender
#: and the Python runtime, the pinned upstream sources, the wheelhouse and the
#: Torch home. Anything else, including any file a builder later stamps with
#: the release, stays out of the cache by default.
CACHED_TREES = (
    "artifixer_source/",
    "artisan_source/",
    "blueprint_runtime/torch_home/",
    "python_wheelhouse/",
    "runtime/",
)


class TaskEvaluationSceneConfigurationToolchainArchiveError(ValueError):
    """The toolchain archive could not be built or verified."""


@dataclass(frozen=True)
class ToolchainArchive:
    """A verified-on-build archive of every inventoried toolchain member."""

    root: Path
    index: Mapping[str, Any]
    reused: bool
    persistent: bool
    # The fragment, held open from preparation until release: another
    # activation's retention can unlink the entry, never these bytes.
    handle: BinaryIO | None = field(default=None, compare=False, repr=False)

    @property
    def fragment(self) -> Path:
        return self.root / FRAGMENT_NAME

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "archive_key": self.index["archive_key"],
            "fragment_sha256": self.index["fragment_sha256"],
            "fragment_size_bytes": self.index["fragment_size_bytes"],
            "member_count": len(self.index["members"]),
            "uncompressed_bytes": self.index["uncompressed_bytes"],
            "reused": self.reused,
            "persistent_cache": self.persistent,
        }


def cached_toolchain_member(relative_path: str) -> bool:
    """Whether a toolchain file belongs to the cached third-party trees."""

    parts = str(relative_path).split("/")
    if len(parts) < 4 or parts[0] != "components" or parts[2] != "package":
        return False
    within = "/".join(parts[3:])
    return any(within.startswith(tree) for tree in CACHED_TREES)


def split_toolchain_manifest(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The manifest restricted to cached trees, and the release-layer rows."""

    cached: list[dict[str, Any]] = []
    release: list[dict[str, Any]] = []
    for row in manifest["files"]:
        (cached if cached_toolchain_member(str(row["relative_path"])) else release).append(row)
    return {**manifest, "files": cached}, sorted(release, key=lambda row: str(row["relative_path"]))


def _inventory(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "relative_path": str(row["relative_path"]),
            "sha256": str(row["sha256"]),
            "size_bytes": int(row["size_bytes"]),
            "executable": bool(row["executable"]),
        }
        for row in manifest["files"]
    ]
    return sorted(rows, key=lambda row: row["relative_path"])


def archive_key(
    manifest: Mapping[str, Any],
    *,
    deflate_level: int,
    stored_suffixes: Iterable[str],
) -> str:
    """The exact content and archive format, independent of the release."""

    return canonical_digest(
        {
            "schema_version": SCHEMA_VERSION,
            "archive_prefix": ARCHIVE_PREFIX,
            "deflate_level": deflate_level,
            "stored_suffixes": sorted(stored_suffixes),
            "date_time": list(_DATE_TIME),
            "files": _inventory(manifest),
        }
    )


def resolve_cache_root(explicit: str | Path | None = None) -> Path | None:
    """The persistent cache root, or ``None`` to build a single-use archive."""

    unresolved = str(explicit or os.environ.get(CACHE_ROOT_ENV) or "").strip()
    if unresolved:
        return Path(unresolved).expanduser().resolve()
    if DEFAULT_CACHE_PARENT.is_dir() and os.access(DEFAULT_CACHE_PARENT, os.W_OK):
        return DEFAULT_CACHE_PARENT / DEFAULT_CACHE_NAME
    return None


def member_info(
    name: str,
    *,
    size: int,
    executable: bool,
    compress_type: int,
    deflate_level: int,
) -> zipfile.ZipInfo:
    """The ZIP header every bundle member is written with."""

    info = zipfile.ZipInfo(name, date_time=_DATE_TIME)
    info.compress_type = compress_type
    info.file_size = size  # members over 2 GiB need ZIP64 headers before they are written
    if compress_type == zipfile.ZIP_DEFLATED:
        info._compresslevel = deflate_level
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | (0o755 if executable else 0o444)) << 16
    return info


def _open_regular_read_only(path: Path) -> BinaryIO:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o222:
            raise TaskEvaluationSceneConfigurationToolchainArchiveError(
                "scene_configuration_toolchain_archive_source_invalid"
            )
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _write_fragment(
    *,
    toolchain_root: Path,
    manifest: Mapping[str, Any],
    destination: Path,
    compress_type_for: Callable[[Path], int],
    deflate_level: int,
) -> list[dict[str, Any]]:
    """Deflate each inventoried file, hashing the bytes actually compressed."""

    members: list[dict[str, Any]] = []
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for row in _inventory(manifest):
            relative = Path(row["relative_path"])
            source = toolchain_root / relative
            info = member_info(
                ARCHIVE_PREFIX + relative.as_posix(),
                size=row["size_bytes"],
                executable=row["executable"],
                compress_type=compress_type_for(relative),
                deflate_level=deflate_level,
            )
            digest = hashlib.sha256()
            size = 0
            with _open_regular_read_only(source) as input_stream:
                if bool(os.fstat(input_stream.fileno()).st_mode & 0o111) != row["executable"]:
                    raise TaskEvaluationSceneConfigurationToolchainArchiveError(
                        "scene_configuration_toolchain_archive_source_mismatch:"
                        + row["relative_path"]
                    )
                with archive.open(info, "w") as output_stream:
                    for chunk in iter(lambda: input_stream.read(_CHUNK), b""):
                        digest.update(chunk)
                        size += len(chunk)
                        output_stream.write(chunk)
            if size != row["size_bytes"] or "sha256:" + digest.hexdigest() != row["sha256"]:
                raise TaskEvaluationSceneConfigurationToolchainArchiveError(
                    "scene_configuration_toolchain_archive_source_mismatch:"
                    + row["relative_path"]
                )
            written = archive.getinfo(info.filename)
            members.append(
                {
                    "name": written.filename,
                    "header_offset": written.header_offset,
                    "crc32": written.CRC,
                    "compress_type": written.compress_type,
                    "compress_size": written.compress_size,
                    "file_size": written.file_size,
                    "external_attr": written.external_attr,
                    "flag_bits": written.flag_bits,
                    "create_version": written.create_version,
                    "extract_version": written.extract_version,
                    "sha256": row["sha256"],
                }
            )
    return members


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _toolchain_directories(toolchain_root: Path, manifest: Mapping[str, Any]) -> list[str]:
    """Toolchain directories this archive owns.

    Those are the ancestors of its members and every directory that holds no
    file outside it. A directory holding only release-layer files comes from
    the bundle's own staged copy of that layer, so a cached index never lists
    another release's directories.
    """

    archived = {row["relative_path"] for row in _inventory(manifest)}
    ancestors: set[str] = set()
    elsewhere: set[str] = set()
    directories: list[str] = []
    for path in toolchain_root.rglob("*"):
        if path.is_symlink():
            continue
        relative = path.relative_to(toolchain_root).as_posix()
        if path.is_dir():
            directories.append(relative)
            continue
        parents = {parent.as_posix() for parent in Path(relative).parents if parent.as_posix() != "."}
        (ancestors if relative in archived else elsewhere).update(parents)
    return sorted(
        ARCHIVE_PREFIX + name + "/"
        for name in directories
        if name in ancestors or name not in elsewhere
    )


def _build_entry(
    *,
    toolchain_root: Path,
    manifest: Mapping[str, Any],
    key: str,
    directory: Path,
    compress_type_for: Callable[[Path], int],
    deflate_level: int,
) -> dict[str, Any]:
    directory.mkdir(parents=True)
    fragment = directory / FRAGMENT_NAME
    members = _write_fragment(
        toolchain_root=toolchain_root,
        manifest=manifest,
        destination=fragment,
        compress_type_for=compress_type_for,
        deflate_level=deflate_level,
    )
    with zipfile.ZipFile(fragment) as archive:
        entries_size = archive.start_dir
    index: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "archive_key": key,
        "archive_prefix": ARCHIVE_PREFIX,
        "members": members,
        "directories": _toolchain_directories(toolchain_root, manifest),
        "entries_size_bytes": entries_size,
        "fragment_size_bytes": fragment.stat().st_size,
        "fragment_sha256": _file_sha256(fragment),
        "uncompressed_bytes": sum(row["file_size"] for row in members),
        "index_digest": "",
    }
    index["index_digest"] = canonical_digest(index, digest_field="index_digest")
    (directory / INDEX_NAME).write_text(canonical_json(index) + "\n", encoding="utf-8")
    for path in (fragment, directory / INDEX_NAME):
        path.chmod(0o444)
    directory.chmod(0o555)
    return index


def _remove_entry(directory: Path) -> None:
    if not directory.exists() or directory.is_symlink():
        return
    for root, directories, files in os.walk(directory, topdown=True, followlinks=False):
        Path(root).chmod(0o700)
        for name in directories:
            (Path(root) / name).chmod(0o700)
        for name in files:
            (Path(root) / name).chmod(0o600)
    shutil.rmtree(directory)


def _load_index(directory: Path, *, key: str, manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    """The cached index when it is sound and lists exactly this inventory."""

    try:
        index = json.loads((directory / INDEX_NAME).read_text(encoding="utf-8"))
        fragment = directory / FRAGMENT_NAME
        expected = {
            ARCHIVE_PREFIX + row["relative_path"]: row for row in _inventory(manifest)
        }
        members = index["members"]
        if (
            index.get("schema_version") != SCHEMA_VERSION
            or index.get("archive_key") != key
            or index.get("index_digest") != canonical_digest(index, digest_field="index_digest")
            or fragment.is_symlink()
            or not fragment.is_file()
            or fragment.stat().st_size != index["fragment_size_bytes"]
            or [row["name"] for row in members] != sorted(expected)
            or any(
                row["file_size"] != expected[row["name"]]["size_bytes"]
                or row["sha256"] != expected[row["name"]]["sha256"]
                for row in members
            )
        ):
            return None
        return index
    except (OSError, KeyError, TypeError, ValueError):
        return None


def _pin(directory: Path, index: Mapping[str, Any]) -> BinaryIO | None:
    """Open the fragment that matches ``index``; ``None`` when it is gone, writable or resized."""

    try:
        stream = _open_regular_read_only(directory / FRAGMENT_NAME)
    except (OSError, TaskEvaluationSceneConfigurationToolchainArchiveError):
        return None
    if os.fstat(stream.fileno()).st_size != index["fragment_size_bytes"]:
        stream.close()
        return None
    return stream


def prepare_toolchain_archive(
    *,
    toolchain_root: str | Path,
    manifest: Mapping[str, Any],
    compress_type_for: Callable[[Path], int],
    deflate_level: int,
    stored_suffixes: Iterable[str],
    scratch_root: Path,
    cache_root: str | Path | None = None,
) -> ToolchainArchive:
    """Reuse or build the archive for this exact toolchain inventory.

    ``manifest`` must be the toolchain manifest the caller has just validated.
    Without a writable cache root, the archive is built under ``scratch_root``
    and the caller removes it with ``release_toolchain_archive``.
    """

    root = Path(toolchain_root).resolve()
    key = archive_key(manifest, deflate_level=deflate_level, stored_suffixes=stored_suffixes)
    cache = resolve_cache_root(cache_root)
    if cache is not None:
        try:
            cache.mkdir(parents=True, exist_ok=True)
            if not os.access(cache, os.W_OK):
                cache = None
        except OSError:
            cache = None
    if cache is None:
        directory = scratch_root / f".toolchain-archive-{secrets.token_hex(8)}"
        try:
            index = _build_entry(
                toolchain_root=root, manifest=manifest, key=key, directory=directory,
                compress_type_for=compress_type_for, deflate_level=deflate_level,
            )
            handle = _pin(directory, index)
            if handle is None:
                raise TaskEvaluationSceneConfigurationToolchainArchiveError(
                    "scene_configuration_toolchain_archive_unpinned"
                )
        except BaseException:
            # Nothing returns a handle to release it, so never leave gigabytes behind.
            _remove_entry(directory)
            raise
        return ToolchainArchive(root=directory, index=index, reused=False, persistent=False,
                                handle=handle)
    entry = cache / key.removeprefix("sha256:")
    index = _load_index(entry, key=key, manifest=manifest)
    handle = _pin(entry, index) if index is not None else None
    if handle is not None:
        try:
            os.utime(entry)  # recency for retention
        except OSError:
            pass
        return ToolchainArchive(root=entry, index=index, reused=True, persistent=True,
                                handle=handle)
    _remove_entry(entry)
    staging = cache / f".building-{secrets.token_hex(8)}"
    try:
        index = _build_entry(
            toolchain_root=root, manifest=manifest, key=key, directory=staging,
            compress_type_for=compress_type_for, deflate_level=deflate_level,
        )
        try:
            os.rename(staging, entry)
        except OSError:
            # A concurrent activation published the same key first.
            existing = _load_index(entry, key=key, manifest=manifest)
            handle = _pin(entry, existing) if existing is not None else None
            if handle is None:
                raise
            _remove_entry(staging)
            return ToolchainArchive(root=entry, index=existing, reused=True, persistent=True,
                                    handle=handle)
    except BaseException:
        _remove_entry(staging)
        raise
    handle = _pin(entry, index)
    if handle is None:
        raise TaskEvaluationSceneConfigurationToolchainArchiveError(
            "scene_configuration_toolchain_archive_unpinned"
        )
    _prune(cache, keep=entry)
    return ToolchainArchive(root=entry, index=index, reused=False, persistent=True, handle=handle)


def _prune(cache: Path, *, keep: Path) -> None:
    """Keep the most recently used entries; an open fragment survives its unlink."""

    entries = sorted(
        (
            path for path in cache.iterdir()
            if path.is_dir() and not path.is_symlink() and not path.name.startswith(".")
            and path != keep
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for stale in entries[RETAINED_ENTRIES - 1:]:
        try:
            _remove_entry(stale)
        except OSError:
            continue


def _close(archive: ToolchainArchive) -> None:
    if archive.handle is not None and not archive.handle.closed:
        archive.handle.close()


def discard_toolchain_archive(archive: ToolchainArchive) -> None:
    """Remove an archive whose bytes failed verification."""

    _close(archive)
    _remove_entry(archive.root)


def release_toolchain_archive(archive: ToolchainArchive) -> None:
    """Close the pinned fragment; remove a single-use archive. Cache entries stay."""

    _close(archive)
    if not archive.persistent:
        _remove_entry(archive.root)


@contextmanager
def _fragment_stream(toolchain: ToolchainArchive):
    if toolchain.handle is not None:
        toolchain.handle.seek(0)
        yield toolchain.handle
    else:
        with toolchain.fragment.open("rb") as stream:
            yield stream


def splice_toolchain_archive(
    archive_file: zipfile.ZipFile, toolchain: ToolchainArchive
) -> None:
    """Append the archive's members to ``archive_file`` byte for byte.

    The whole fragment is hashed as it is copied. On a mismatch the output is
    truncated back to where the splice began and the error is raised, so the
    caller can rebuild the archive and splice again.
    """

    if archive_file.mode != "w" or archive_file._writing:
        raise TaskEvaluationSceneConfigurationToolchainArchiveError(
            "scene_configuration_toolchain_archive_output_invalid"
        )
    output = archive_file.fp
    base = archive_file.start_dir
    output.seek(base)
    index = toolchain.index
    digest = hashlib.sha256()
    with _fragment_stream(toolchain) as fragment:
        remaining = int(index["entries_size_bytes"])
        while remaining:
            chunk = fragment.read(min(_CHUNK, remaining))
            if not chunk:
                break
            digest.update(chunk)
            output.write(chunk)
            remaining -= len(chunk)
        for chunk in iter(lambda: fragment.read(_CHUNK), b""):
            digest.update(chunk)  # the fragment's own central directory
    if remaining or "sha256:" + digest.hexdigest() != index["fragment_sha256"]:
        output.seek(base)
        output.truncate()
        raise TaskEvaluationSceneConfigurationToolchainArchiveError(
            "scene_configuration_toolchain_archive_fragment_mismatch"
        )
    for row in index["members"]:
        info = zipfile.ZipInfo(row["name"], date_time=_DATE_TIME)
        info.compress_type = row["compress_type"]
        info.create_system = 3
        info.external_attr = row["external_attr"]
        info.CRC = row["crc32"]
        info.compress_size = row["compress_size"]
        info.file_size = row["file_size"]
        info.header_offset = base + row["header_offset"]
        info.flag_bits = row["flag_bits"]
        info.create_version = row["create_version"]
        info.extract_version = row["extract_version"]
        archive_file.filelist.append(info)
        archive_file.NameToInfo[info.filename] = info
    archive_file.start_dir = output.tell()
    archive_file._didModify = True


__all__ = [
    "CACHED_TREES",
    "cached_toolchain_member",
    "split_toolchain_manifest",
    "ARCHIVE_PREFIX",
    "CACHE_ROOT_ENV",
    "SCHEMA_VERSION",
    "TaskEvaluationSceneConfigurationToolchainArchiveError",
    "ToolchainArchive",
    "archive_key",
    "discard_toolchain_archive",
    "member_info",
    "prepare_toolchain_archive",
    "release_toolchain_archive",
    "resolve_cache_root",
    "splice_toolchain_archive",
]
