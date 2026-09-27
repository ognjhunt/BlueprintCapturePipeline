"""Measure disk usage the way the disk sees it: once per inode, in allocated blocks.

Hardlinks are how the control plane shares bytes between content stores, compiled
episodes and launch sets, so a byte count per name overstates usage (on 2026-09-26 two
cache roots "held" about 470 GB on a 165 GB disk). Every measurement here counts each
``(st_dev, st_ino)`` once and reports allocated bytes (``st_blocks * 512``).

``survey_usage`` answers "what uses the space?" for whole mounts: it attributes every
surveyed byte to a storage class, a root and an owner (a scene, run, release or
store), and says how much of each mount's used bytes that attribution covers.
"""

from __future__ import annotations

import fnmatch
import os
import re
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

from . import control_plane_storage_roots as storage_roots
from .decision_evidence_contracts import canonical_digest

SURVEY_SCHEMA_VERSION = "control_plane_disk_usage_survey.v1"
# Only the roots bind-mounted back into the service tree are aliased. A loose
# top-level folder on the work volume must remain visible as an unclassified
# physical path rather than inherit /var/lib/blueprint's container class.
_BOUND_VOLUME_ROOTS = (
    "task-evaluation-inputs", "pubsub-handoffs", "production-gpu-artifacts",
    "pipeline-control-plane/task-evaluation-launch-runs",
    "pipeline-control-plane/task-evaluation-policy-canaries",
    "pipeline-control-plane/capture-reconstruction-runs",
    "pipeline-control-plane/capture-reconstruction-derived",
    "pipeline-control-plane/episode-interpretation-backfills",
    "pipeline-control-plane/policy-canary-preprovider-audits",
    "pipeline-control-plane/scene-configuration-diagnostics",
    "pipeline-control-plane/result-artifact-cache",
    "pipeline-control-plane/profile-install-staging",
    "pipeline-control-plane/policy-canary-presubmission",
    "pipeline-control-plane/native-g1-team-campaign-work",
    "pipeline-control-plane/engineering", "pipeline-control-plane/render-probes",
    "pipeline-control-plane/diagnostic-checkouts", "pipeline-control-plane/release-builds",
)
DEFAULT_SURVEY_ALIASES: Mapping[str, str] = {
    **{f"/mnt/blueprint-work/{root}": f"/var/lib/blueprint/{root}" for root in _BOUND_VOLUME_ROOTS},
    "/mnt/blueprint-work/workspace": "/workspace",
}
# Unknown children of these prefixes are Blueprint bytes the storage table does not
# know yet ("unclassified"); everything outside them is the host's ("host").
BLUEPRINT_PREFIXES: tuple[str, ...] = (
    "/var/lib/blueprint", "/opt/blueprint", "/workspace", "/mnt/blueprint-work",
)
DEFAULT_SURVEY_MAX_ENTRIES = 3_000_000
DEFAULT_SURVEY_MAX_SECONDS = 240.0
# The capacity unit has MemoryMax=512M. Bound every growing walk container;
# crossing a bound produces an honest partial report instead of killing the tick.
SURVEY_MAX_BUFFERED_ENTRIES = 20_000
SURVEY_MAX_SHARED_INODES = 20_000
MOUNTINFO_PATH = "/proc/self/mountinfo"
SURVEY_TOP_ROWS = 10
_STORE_DIRECTORY = "content-addressed"
_COMMIT_NAME = re.compile(r"[0-9a-f]{40}(?![0-9a-f])")
_MOUNTINFO_ESCAPE = re.compile(r"\\([0-7]{3})")
_UNATTRIBUTED_CLASSES = frozenset({"unclassified", "host"})
# File and directory names are untrusted input. The survey is mode 0644 and is
# projected into the operator door, whose scanner refuses credential-shaped bytes.
_CREDENTIAL_SHAPED_NAME = re.compile(
    r"(?i)(?:\bsk-(?:proj-|live-|svcacct-)?[a-z0-9_-]{20,}"
    r"|\b(?:sk|rk)_(?:live|test)_[a-z0-9]{16,}"
    r"|\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"
    r"|AIza[0-9a-z_-]{35}"
    r"|\b(?:ya29\.[a-z0-9_-]{20,}|1//0[a-z0-9_-]{20,})"
    r"|\b(?:gh[pousr]_[a-z0-9]{36,}|github_pat_[a-z0-9_]{22,})"
    r"|\b(?:xox[abprs]-[a-z0-9-]{10,}|bpk_[a-z0-9_-]{32,})"
    r"|\b(?:ntn|secret)_[a-z0-9]{40,}|\bhf_[a-z0-9]{30,}"
    r"|\beyJ[a-z0-9_-]{10,}\.eyJ[a-z0-9_-]{10,}\.[a-z0-9_-]{10,}"
    r"|https://hooks\.slack\.com/services/[a-z0-9/_-]{10,}"
    r"|[a-z][a-z0-9+.-]{0,15}://[^\s/:@'\"]{0,64}:[^\s/@'\"]{3,256}@"
    r"|[?&](?:access_token|refresh_token|id_token|token|api_key|secret|password)=[a-z0-9._~%+/=-]{16,}"
    r"|(?:authorization[ \t]*[:=][ \t]*bearer[ \t]+)[a-z0-9._~+/=-]{20,}"
    r"|(?:-----BEGIN [a-z0-9 ]{0,40}PRIVATE KEY(?: BLOCK)?-----)"
    r"|(?:^|[/\\])[^/\\]*(?:secret|credential|password|passwd|api[-_]?key|private[-_]?key)[^/\\]*"
    r"|\.(?:pem|key|p12|pfx|jks|keystore|kdbx|gpg|asc)(?:$|[/\\]))"
)

MAX_TREE_SCAN_ENTRIES = 100_000
MAX_TRACKED_SHARED_INODES = 50_000


@dataclass(frozen=True)
class TreeUsage:
    allocated_bytes: int = 0
    apparent_bytes: int = 0
    files: int = 0
    directories: int = 0
    unique_inodes: int = 0
    shared_inodes: int = 0
    unreadable: int = 0


def allocated_bytes(metadata: os.stat_result) -> int:
    """Allocated bytes, or the apparent size when the filesystem reports no blocks.

    APFS reports zero blocks for directories and for data stored inline, so zero
    would under-count; the apparent size is the conservative stand-in.
    """

    blocks = getattr(metadata, "st_blocks", None)
    if isinstance(blocks, int) and blocks > 0:
        return blocks * 512
    return int(metadata.st_size)


def tree_usage(path: str | Path) -> TreeUsage:
    """Unique-inode usage of ``path`` (a file or a directory tree); never follows symlinks."""

    root = Path(path)
    try:
        top = os.lstat(root)
    except FileNotFoundError:
        return TreeUsage()
    except OSError:
        return TreeUsage(unreadable=1)
    # Only multiply-linked inodes can be reached twice, so only they are remembered.
    # Directories are never deduplicated: their link count is 2 + subdirectories and
    # says nothing about sharing.
    seen: set[tuple[int, int]] = set()
    totals = {
        "allocated": 0,
        "apparent": 0,
        "files": 0,
        "directories": 0,
        "unique": 0,
        "shared": 0,
        "unreadable": 0,
    }

    def account(metadata: os.stat_result) -> bool:
        if not stat.S_ISDIR(metadata.st_mode) and metadata.st_nlink > 1:
            key = (metadata.st_dev, metadata.st_ino)
            if key in seen:
                return True
            if len(seen) >= MAX_TRACKED_SHARED_INODES:
                return False
            seen.add(key)
            totals["shared"] += 1
        totals["unique"] += 1
        totals["allocated"] += allocated_bytes(metadata)
        totals["apparent"] += int(metadata.st_size)
        return True

    if not account(top):
        return TreeUsage(unreadable=1)
    if not stat.S_ISDIR(top.st_mode):
        totals["files"] += 1
    else:
        totals["directories"] += 1
        pending = [root]
        scanned_entries = 0
        truncated = False
        while pending and not truncated:
            directory = pending.pop()
            try:
                with os.scandir(directory) as iterator:
                    for entry in iterator:
                        scanned_entries += 1
                        if scanned_entries > MAX_TREE_SCAN_ENTRIES:
                            totals["unreadable"] += 1
                            truncated = True
                            break
                        try:
                            metadata = entry.stat(follow_symlinks=False)
                        except OSError:
                            totals["unreadable"] += 1
                            continue
                        if not account(metadata):
                            totals["unreadable"] += 1
                            truncated = True
                            break
                        if stat.S_ISDIR(metadata.st_mode):
                            totals["directories"] += 1
                            pending.append(Path(entry.path))
                        else:
                            totals["files"] += 1
            except OSError:
                totals["unreadable"] += 1
    return TreeUsage(
        allocated_bytes=totals["allocated"],
        apparent_bytes=totals["apparent"],
        files=totals["files"],
        directories=totals["directories"],
        unique_inodes=totals["unique"],
        shared_inodes=totals["shared"],
        unreadable=totals["unreadable"],
    )


def _join(parts: Sequence[str]) -> str:
    if parts and parts[0] == "/":
        return "/" + "/".join(parts[1:])
    return "/".join(parts)


@lru_cache(maxsize=4096)
def _parts(path: str) -> tuple[str, ...]:
    return PurePosixPath(path).parts


def _absolute(path: str | os.PathLike[str]) -> str:
    return os.path.abspath(os.fspath(path))


def _canonical(path: str, aliases: Sequence[tuple[str, str]]) -> str:
    """``path`` with its longest aliased prefix replaced (``aliases`` is longest first)."""

    for source, target in aliases:
        if path == source:
            return target
        stem = source.rstrip("/")
        if path.startswith(stem + "/"):
            return target.rstrip("/") + path[len(stem):]
    return path


def _mount_points(mountinfo: str | os.PathLike[str]) -> frozenset[str]:
    """Mount points named in ``mountinfo`` (its fifth field), or none when unreadable."""

    try:
        text = Path(mountinfo).read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return frozenset()
    points = set()
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 5:
            points.add(_MOUNTINFO_ESCAPE.sub(lambda match: chr(int(match.group(1), 8)), fields[4]))
    return frozenset(points)


def _concrete_root(root_path: str, parts: tuple[str, ...]) -> tuple[str, ...]:
    """A storage-table root as the concrete path it names for ``parts``.

    A root may carry ``*`` segments that each match one path component; its
    concrete root is then the candidate's prefix with the same number of segments.
    """

    root_parts = _parts(root_path)
    if "*" in root_path and len(parts) >= len(root_parts):
        return parts[: len(root_parts)]
    return root_parts


def _classification(
    match: Any, parts: tuple[str, ...], prefixes: Sequence[tuple[str, ...]]
) -> tuple[str, tuple[str, ...]]:
    """The storage class and the root of ``parts``.

    A classified path takes its table row's class and root. A ``container`` holds
    only classified children, so a path below one that no deeper row claims is
    ``unclassified``, rooted at the child it lies in; so is a path under a Blueprint
    prefix that the table does not know. Everything else is ``host``, rooted at its
    first two components.
    """

    base: tuple[str, ...] | None = None
    if match is not None:
        root = _concrete_root(str(match.path), parts)
        if match.storage_class != "container":
            return str(match.storage_class), root
        base = root
    for prefix in prefixes:
        if (base is None or len(prefix) > len(base)) and parts[: len(prefix)] == prefix:
            base = prefix
    if base is None:
        return "host", parts[:3]
    if len(parts) <= len(base):
        return "container", base
    return "unclassified", parts[: len(base) + 1]


def _find(parts: tuple[str, ...], name: str) -> int:
    try:
        return parts.index(name)
    except ValueError:
        return -1


_RUN_ANCHORS = (
    ("task-evaluation-launch-runs", "run"),
    ("task-evaluation-policy-canaries", "run"),
    ("task-evaluation-scene-intents", "scene-intent"),
)
_RELEASE_ANCHORS = (("system-runtimes", 2), ("task-evaluation-control-plane-releases", 1))


def _owner(parts: tuple[str, ...], root: tuple[str, ...], storage_class: str, is_dir: bool) -> str:
    """Who the bytes at ``parts`` belong to; the first matching rule wins.

    An ``<id>`` component names an owner only as a directory: something lies below
    it, or it is the directory being attributed. A file beside the ids (a pointer or
    a marker) is not an id.
    """

    last = len(parts) - 1

    def is_directory(index: int) -> bool:
        return index < last or (index == last and is_dir)

    at = _find(parts, "pubsub-handoffs")
    if at >= 0 and at + 3 <= last and parts[at + 2] == "scenes" and is_directory(at + 3):
        return f"scene:{parts[at + 3]}"
    for anchor, offset in _RELEASE_ANCHORS:
        at = _find(parts, anchor)
        if (at >= 0 and at + offset <= last and _COMMIT_NAME.match(parts[at + offset])
                and is_directory(at + offset)):
            return f"release:{parts[at + offset][:12]}"
    at = _find(parts, _STORE_DIRECTORY)
    if at >= 0 and is_directory(at):
        return f"store:{root[-1]}"
    if storage_class == "lane_scratch":
        at = _find(parts, "lanes")
        if at >= 0 and at + 1 <= last and is_directory(at + 1):
            return f"lane:{parts[at + 1]}"
    for anchor, kind in _RUN_ANCHORS:
        at = _find(parts, anchor)
        if at >= 0 and at + 1 <= last and is_directory(at + 1):
            return f"{kind}:{parts[at + 1]}"
    if storage_class in _UNATTRIBUTED_CLASSES:
        return _join(root)
    depth = len(root)
    if depth <= last and is_directory(depth):
        return f"{root[-1]}/{parts[depth]}"
    return root[-1]


def _may_hold_root_below(parts: tuple[str, ...], patterns: Sequence[tuple[str, ...]]) -> bool:
    """Whether any storage-table row could match a path strictly below ``parts``."""

    depth = len(parts)
    return any(
        len(pattern) > depth
        and all(fnmatch.fnmatchcase(part, segment) for part, segment in zip(parts, pattern))
        for pattern in patterns
    )


def _used_bytes(statvfs: Callable[[str], Any], mount: str) -> int | None:
    try:
        result = statvfs(mount)
        return max(0, (int(result.f_blocks) - int(result.f_bfree)) * int(result.f_frsize))
    except (OSError, AttributeError, TypeError, ValueError):
        return None


@dataclass
class _Directory:
    path: str  # where the walk found it
    parts: tuple[str, ...]  # its canonical path
    match: Any  # its storage-table row, or None
    inherits: bool  # no table row can lie below it, so every descendant shares ``match``
    file_attribution: tuple[str, str, str] | None = None


class _UsageWalk:
    """One survey's traversal state: totals per attribution, per device, per shared inode."""

    def __init__(
        self,
        *,
        aliases: Sequence[tuple[str, str]],
        prefixes: Sequence[tuple[str, ...]],
        classify: Callable[[str], Any],
        patterns: Sequence[tuple[str, ...]] | None,
        mount_points: frozenset[str],
        max_entries: int,
        deadline: float,
        clock: Callable[[], float],
    ) -> None:
        self.aliases = aliases
        self.alias_targets = dict(aliases)
        self.prefixes = prefixes
        self.classify = classify
        self.patterns = patterns
        self.mount_points = mount_points
        self.max_entries = max_entries
        self.buffer_limit = min(max_entries, SURVEY_MAX_BUFFERED_ENTRIES)
        self.shared_limit = min(max_entries, SURVEY_MAX_SHARED_INODES)
        self.deadline = deadline
        self.clock = clock
        self.walk_roots: frozenset[tuple[int, int]] = frozenset()
        # (storage class, root, owner) -> [allocated, apparent, files]
        self.totals: dict[tuple[str, str, str], list[int]] = {}
        # st_dev -> [surveyed bytes, classified bytes]
        self.devices: dict[int, list[int]] = {}
        # (st_dev, st_ino) -> [(not in a store, canonical name), attribution, allocated, apparent, st_dev]
        self.shared: dict[tuple[int, int], list[Any]] = {}
        self.entries = 0
        self.unreadable = 0
        self.duplicates = 0
        self.truncated = False

    def _exhausted(self) -> bool:
        if self.entries >= self.max_entries or self.clock() >= self.deadline:
            self.truncated = True
        return self.truncated

    def _attribute(self, parts: tuple[str, ...], match: Any, is_dir: bool) -> tuple[str, str, str]:
        storage_class, root = _classification(match, parts, self.prefixes)
        return storage_class, _join(root), _owner(parts, root, storage_class, is_dir)

    def _directory(self, path: str, parts: tuple[str, ...]) -> _Directory:
        inherits = self.patterns is not None and not _may_hold_root_below(parts, self.patterns)
        return _Directory(path, parts, self.classify(_join(parts)), inherits)

    def _child(self, parent: _Directory, name: str, path: str) -> _Directory:
        aliased = self.alias_targets.get(path)
        if aliased is not None:
            return self._directory(path, _parts(aliased))
        parts = parent.parts + (name,)
        if parent.inherits:
            return _Directory(path, parts, parent.match, True)
        return self._directory(path, parts)

    def _file(self, directory: _Directory, name: str) -> tuple[str, str, str]:
        cached = directory.file_attribution
        if cached is not None:
            return cached
        parts = directory.parts + (name,)
        match = directory.match if directory.inherits else self.classify(_join(parts))
        storage_class, root = _classification(match, parts, self.prefixes)
        attribution = (storage_class, _join(root), _owner(parts, root, storage_class, False))
        if directory.inherits and len(root) < len(parts):
            # Neither the root nor the owner depends on this file's own name.
            directory.file_attribution = attribution
        return attribution

    def _add(self, attribution: tuple[str, str, str], allocated: int, apparent: int, files: int,
             device: int) -> None:
        row = self.totals.get(attribution)
        if row is None:
            if len(self.totals) >= self.buffer_limit:
                self.truncated = True
                return
            row = self.totals[attribution] = [0, 0, 0]
        row[0] += allocated
        row[1] += apparent
        row[2] += files
        device_totals = self.devices.setdefault(device, [0, 0])
        device_totals[0] += allocated
        if attribution[0] not in _UNATTRIBUTED_CLASSES:
            device_totals[1] += allocated

    def _record(self, metadata: os.stat_result, attribution: tuple[str, str, str], device: int,
                parts: tuple[str, ...]) -> None:
        directory = stat.S_ISDIR(metadata.st_mode)
        if directory or metadata.st_nlink <= 1:
            self._add(attribution, allocated_bytes(metadata), int(metadata.st_size),
                      0 if directory else 1, device)
            return
        # A multiply-linked inode belongs to its content store: the smallest of its
        # names under a content-addressed directory, else its smallest name. Its bytes
        # are added once every name has been seen, so traversal order cannot move them.
        rank = (_STORE_DIRECTORY not in parts[:-1], _join(parts))
        key = (metadata.st_dev, metadata.st_ino)
        held = self.shared.get(key)
        if held is None:
            if len(self.shared) >= self.shared_limit:
                self.truncated = True
                return
            self.shared[key] = [rank, attribution, allocated_bytes(metadata),
                                int(metadata.st_size), device]
            return
        self.duplicates += 1
        if rank < held[0]:
            held[0], held[1] = rank, attribution

    def walk(self, path: str, metadata: os.stat_result) -> None:
        """Walk one mount within its own filesystem, in ascending name order."""

        device = metadata.st_dev
        self.devices.setdefault(device, [0, 0])
        if self._exhausted():
            return
        self.entries += 1
        top = self._directory(path, _parts(_canonical(path, self.aliases)))
        is_dir = stat.S_ISDIR(metadata.st_mode)
        self._record(metadata, self._attribute(top.parts, top.match, is_dir), device, top.parts)
        if not is_dir:
            return
        stack = [top]
        while stack:
            directory = stack.pop()
            try:
                with os.scandir(directory.path) as iterator:
                    entries = []
                    overflow = False
                    for entry in iterator:
                        if len(entries) >= self.buffer_limit:
                            overflow = True
                            break
                        entries.append(entry)
                    entries.sort(key=lambda entry: entry.name)
            except OSError:
                self.unreadable += 1
                continue
            subdirectories = []
            for entry in entries:
                if self._exhausted():
                    return
                self.entries += 1
                try:
                    child = entry.stat(follow_symlinks=False)
                except OSError:
                    self.unreadable += 1
                    continue
                if child.st_dev != device or entry.path in self.mount_points:
                    continue  # another filesystem: its own mount's walk counts it
                if stat.S_ISDIR(child.st_mode):
                    if (child.st_dev, child.st_ino) in self.walk_roots:
                        continue  # a listed mount nested here is walked on its own
                    nested = self._child(directory, entry.name, entry.path)
                    self._record(child, self._attribute(nested.parts, nested.match, True), device,
                                 nested.parts)
                    if len(stack) + len(subdirectories) >= self.buffer_limit:
                        self.truncated = True
                        return
                    subdirectories.append(nested)
                else:
                    self._record(child, self._file(directory, entry.name), device,
                                 directory.parts + (entry.name,))
            if self.truncated:
                return
            if overflow:
                self.truncated = True
                return
            stack.extend(reversed(subdirectories))

    def finish(self) -> None:
        for _rank, attribution, allocated, apparent, device in self.shared.values():
            self._add(attribution, allocated, apparent, 1, device)


def _mount_row(mount: str, used: int | None, surveyed: int, classified: int) -> dict[str, Any]:
    if used is None:
        fraction = None
    else:
        fraction = 1.0 if used == 0 else round(min(1.0, surveyed / used), 4)
    return {
        "mount": mount,
        "used_bytes": used,
        "surveyed_bytes": surveyed,
        "classified_bytes": classified,
        "attributed_fraction": fraction,
    }


def _report_safe(value: Any) -> Any:
    """Render surrogate-escaped filesystem bytes before canonical JSON encoding."""

    if isinstance(value, str):
        try:
            return os.fsencode(value).decode("utf-8", "backslashreplace")
        except UnicodeEncodeError:
            return value.encode("utf-8", "backslashreplace").decode("utf-8")
    if isinstance(value, list):
        return [_report_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _report_safe(item) for key, item in value.items()}
    return value


def sanitize_public_survey(survey: Mapping[str, Any]) -> dict[str, Any]:
    """Remove credential-shaped names before publishing a survey or projection.

    A whole label is redacted so its bytes cannot be recovered from a partial
    path; the row's byte count and class remain intact. This also handles a
    previously saved survey when a newer controller reads it.
    """

    def sanitize(value: Any) -> Any:
        if isinstance(value, str):
            # Path names containing assignment, URL, or JSON syntax can carry
            # many credential forms (including signed URLs) without a familiar
            # token prefix. Such names are rare and safe to hide wholesale.
            unsafe_syntax = any(character in value for character in '?=@"\r\n') or (
                ":" in value and not (value.count(":") == 1 and value.startswith(
                    ("sha256:", "scene:", "scene-intent:", "run:", "release:", "store:", "lane:")
                ))
            )
            return "<redacted>" if unsafe_syntax or _CREDENTIAL_SHAPED_NAME.search(value) else value
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if isinstance(value, dict):
            return {key: sanitize(item) for key, item in value.items()}
        return value

    safe = sanitize(dict(survey))
    if "survey_digest" in safe:
        safe["survey_digest"] = canonical_digest(safe, digest_field="survey_digest")
    return safe


def survey_usage(
    mounts: Sequence[str | os.PathLike[str]],
    *,
    aliases: Mapping[str, str] | None = None,
    prefixes: Sequence[str] = BLUEPRINT_PREFIXES,
    classify: Callable[[str], Any] | None = None,
    statvfs: Callable[[str], Any] = os.statvfs,
    max_entries: int = DEFAULT_SURVEY_MAX_ENTRIES,
    max_seconds: float = DEFAULT_SURVEY_MAX_SECONDS,
    clock: Callable[[], float] = time.time,
    mountinfo: str | os.PathLike[str] = MOUNTINFO_PATH,
) -> dict[str, Any]:
    """Attribute every byte on ``mounts`` to a storage class, a root and an owner.

    Each mount is walked within its own filesystem (``du -x``): a directory on
    another device or named as a mount point in ``mountinfo`` is skipped, a listed
    mount nested inside another is walked once on its own, and symlinks are never
    followed. Every inode is counted once, in allocated bytes, and attributed at its
    canonical path (``aliases`` applied, longest prefix first). The walk stops at
    ``max_entries`` or ``max_seconds`` with ``status: "truncated"``; unreadable
    entries are counted, never raised. One row per filesystem says how much of its
    used bytes the survey attributed.
    """

    started = clock()
    if classify is None:
        classify = storage_roots.classify_path
        patterns: tuple[tuple[str, ...], ...] | None = tuple(
            _parts(root.path) for root in storage_roots.STORAGE_ROOTS
        )
    else:
        patterns = None  # an unknown table: classify every entry
    walk = _UsageWalk(
        aliases=sorted(
            ((_absolute(source), _absolute(target))
             for source, target in (DEFAULT_SURVEY_ALIASES if aliases is None else aliases).items()),
            key=lambda pair: (-len(pair[0]), pair[0]),
        ),
        prefixes=tuple(_parts(_absolute(prefix)) for prefix in prefixes),
        classify=classify,
        patterns=patterns,
        mount_points=_mount_points(mountinfo),
        max_entries=max_entries,
        deadline=started + max_seconds,
        clock=clock,
    )
    walks: list[tuple[str, os.stat_result]] = []
    unreadable_mounts: list[str] = []
    seen: set[tuple[int, int]] = set()
    for mount in mounts:
        path = _absolute(mount)
        try:
            metadata = os.lstat(path)
        except (OSError, ValueError):
            walk.unreadable += 1
            if path not in unreadable_mounts:
                unreadable_mounts.append(path)
            continue
        key = (metadata.st_dev, metadata.st_ino)
        if key not in seen:
            seen.add(key)
            walks.append((path, metadata))
    walk.walk_roots = frozenset(seen)
    for path, metadata in walks:
        walk.walk(path, metadata)
    walk.finish()

    filesystems: dict[int, list[str]] = {}
    for path, metadata in walks:
        filesystems.setdefault(metadata.st_dev, []).append(path)
    rows = []
    for device, paths in filesystems.items():
        label = min(paths, key=lambda item: (len(item), item))
        surveyed, classified = walk.devices.get(device, [0, 0])
        rows.append(_mount_row(label, _used_bytes(statvfs, label), surveyed, classified))
    rows.extend(_mount_row(path, None, 0, 0) for path in unreadable_mounts)

    by_class: dict[str, list[int]] = {}
    by_root: dict[tuple[str, str], int] = {}
    for (storage_class, root, _owner_name), (allocated, apparent, files) in walk.totals.items():
        totals = by_class.setdefault(storage_class, [0, 0, 0])
        totals[0] += allocated
        totals[1] += apparent
        totals[2] += files
        by_root[(root, storage_class)] = by_root.get((root, storage_class), 0) + allocated
    roots = sorted(by_root.items(), key=lambda item: (-item[1], item[0]))
    owners = sorted(walk.totals.items(),
                    key=lambda item: (-item[1][0], item[0][2], item[0][1], item[0][0]))
    survey: dict[str, Any] = {
        "schema_version": SURVEY_SCHEMA_VERSION,
        "status": "truncated" if walk.truncated else "complete",
        "observed_at_epoch": started,
        "duration_seconds": round(max(0.0, clock() - started), 3),
        "entries_visited": walk.entries,
        "unreadable": walk.unreadable,
        "mounts": rows,
        "by_class": [
            {"storage_class": storage_class, "allocated_bytes": allocated,
             "apparent_bytes": apparent, "files": files}
            for storage_class, (allocated, apparent, files)
            in sorted(by_class.items(), key=lambda item: (-item[1][0], item[0]))
        ],
        "top_roots": [
            {"root": root, "storage_class": storage_class, "allocated_bytes": allocated}
            for (root, storage_class), allocated in roots[:SURVEY_TOP_ROWS]
        ],
        "top_owners": [
            {"owner": owner, "root": root, "storage_class": storage_class,
             "allocated_bytes": totals[0]}
            for (storage_class, root, owner), totals in owners[:SURVEY_TOP_ROWS]
        ],
        "unclassified_roots": [
            {"root": root, "allocated_bytes": allocated}
            for (root, storage_class), allocated in roots
            if storage_class == "unclassified"
        ],
        "hardlinks": {
            "shared_inodes": len(walk.shared),
            "shared_bytes": sum(held[2] for held in walk.shared.values()),
            "duplicate_names_skipped": walk.duplicates,
        },
        "survey_digest": "",
    }
    survey = sanitize_public_survey(_report_safe(survey))
    survey["survey_digest"] = canonical_digest(survey, digest_field="survey_digest")
    return survey


__all__ = [
    "BLUEPRINT_PREFIXES",
    "DEFAULT_SURVEY_ALIASES",
    "SURVEY_SCHEMA_VERSION",
    "TreeUsage",
    "allocated_bytes",
    "survey_usage",
    "sanitize_public_survey",
    "tree_usage",
]
