"""Restore the pi0.5 G1 language files from a retained provider bundle on this host.

Both HumanoidArena G1 pi0.5 policies read PaliGemma's six tokenizer files from
``native_g1_beta_tokenizer.RETAINED_TOKENIZER_DIR``. That folder exists, but its
files no longer verify against ``configs/g1_paligemma_tokenizer_inventory.v1.json``
(whose own sha256 ``native_g1_beta_authority.PI_AUXILIARY`` pins, and which still
matches), so screening holds every pi0.5 row with
``g1_beta_selected_pi_tokenizer_invalid``.

Authorization: the owner, on 2026-11-07, relayed by the library-aisle session.
About 23:05Z, "lets do option 2": a door operation that restores and verifies the
files, the way restore-g1-runtime-source did. About 23:10Z, "Gemma terms OK":
development-simulation use of pi0.5's Gemma tokenizer is approved under Google's
Gemma terms. The source is a verified copy from a retained bundle on this host
and nothing else: no Hugging Face, no credential, no network, no money.

Source. G1 provider bundles embed the six files at
``provider_runtime/inputs/pi_tokenizer/<path>`` (``native_g1_provider_bundle``;
``native_g1_beta_bundle`` uses the same layout). Only
``<work root>/g1-<64 hex>/bundle/<bundle name>`` is searched, for the two bundle
names, in sorted order, bounded, never following a link. Only the inventory's
members are read from each archive, each checked by the verifier's own rule
(``native_g1_pi_tokenizer_assets.stream_identity``: the pinned size, then the
pinned sha256 or Git blob sha1). The first archive whose every member verifies is
the source; when none does, the refusal lists every archive checked with its
reason.

Target. Nothing is deleted and nothing is overwritten. A folder that already
verifies is a no-op. Otherwise the files are written into a fresh temporary
sibling, fsynced and verified again with ``verify_tokenizer_assets``; the invalid
folder, once its file names, sizes and sha256 are in the result (never its
contents), is set aside as the sibling ``.<name>.invalid-<UTC>``; the verified
sibling is renamed into place and verified once more; and the result is kept
beside the folder as ``pi-language-assets-restore-<UTC>.json``.

The operator door's ``restore-g1-pi-language-assets`` request carries only
``apply``: a plan reads and writes nothing but its result. Output is JSON only.
The door refuses to read any path naming the tokenizer, so the operation, this
module, the door result and the kept receipt are all named without that word.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import stat
import sys
import zipfile
import zlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

from .control_plane_disk_budget import (
    DEFAULT_RESERVATION_ROOT,
    ControlPlaneDiskBudgetError,
    disk_headroom,
    reserve_control_plane_disk,
)
from .native_g1_beta_authority import PI_AUXILIARY
from .native_g1_beta_tokenizer import INVENTORY, RETAINED_TOKENIZER_DIR
from .native_g1_pi_tokenizer_assets import _inventory as pi_language_inventory
from .native_g1_pi_tokenizer_assets import stream_identity, verify_tokenizer_assets


SCHEMA_VERSION = "native_g1_pi_language_asset_restore.v1"
#: The G1 team campaign work root (control_plane_storage_roots; the campaign units bind it).
SOURCE_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/native-g1-team-campaign-work")
CAMPAIGN_DIRECTORY = re.compile(r"g1-[0-9a-f]{64}")
BUNDLE_DIRECTORY = "bundle"
# native_g1_provider_bundle.BUNDLE_NAME, then native_g1_beta_bundle.BUNDLE_NAME; a test pins both.
BUNDLE_NAMES = ("native_g1_provider_bundle.zip", "native_g1_beta_provider_bundle.zip")
# Where both bundle builders put the six files; a test runs the provider bundle's own check on it.
ARCHIVE_PREFIX = "provider_runtime/inputs/pi_tokenizer/"
MAX_BUNDLES_CHECKED = 64
MAX_BUNDLE_BYTES = 4 * 1024**3
MAX_LISTED_ENTRIES = 256
MAX_LISTED_DEPTH = 4
MAX_HASHED_FILE_BYTES = 256 * 1024**2
# The runtime source restore's pattern: a caller-exact hold in the G1 lane's cache-fill role,
# whose 4 h ledger TTL outlives the door unit. The hold binds no workspace.
DISK_ROLE = "g1_checkpoint_cache"
DISK_MARGIN_BYTES = 64 * 1024**2
RESERVATION_ROOT_ENV = "BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT"
CHUNK_BYTES = 1024 * 1024
DIRECTORY_MODE = 0o750
FILE_MODE = 0o640
RECEIPT_PREFIX = "pi-language-assets-restore-"
OUTCOMES = ("planned", "restored", "already_restored", "refused")
REFUSALS = (
    "g1_pi_language_assets_inventory_changed",
    "g1_pi_language_assets_target_conflict",
    "g1_pi_language_assets_restore_busy",
    "g1_pi_language_assets_source_root_unavailable",
    "g1_pi_language_assets_no_verified_source",
    "g1_pi_language_assets_disk_insufficient",
    "g1_pi_language_assets_target_parent_unwritable",
    "g1_pi_language_assets_copy_failed",
    "g1_pi_language_assets_verification_failed",
    "g1_pi_language_assets_restore_failed",
)
#: Why one archive was not the source; ``verified`` marks the one that was.
BUNDLE_REASONS = (
    "verified",
    "bundle_symlink_refused",
    "bundle_not_regular_file",
    "bundle_too_large",
    "bundle_unreadable",
    "bundle_not_a_zip",
    "bundle_member_missing",
    "bundle_member_duplicate",
    "bundle_member_size_mismatch",
    "bundle_member_digest_mismatch",
    "bundle_member_unreadable",
)
AUTHORIZATION = {
    "reference": ('Owner, 2026-11-07, relayed by the library-aisle session: "lets do option 2" '
                  '(about 23:05Z) and "Gemma terms OK" (about 23:10Z)'),
    "decided_on": "2026-11-07",
    "relayed_by": "library-aisle session",
    "decisions": [
        {"words": "lets do option 2", "at": "about 23:05Z",
         "meaning": "a door operation that restores and verifies the files, as restore-g1-runtime-source did"},
        {"words": "Gemma terms OK", "at": "about 23:10Z",
         "meaning": ("development-simulation use of the pi0.5 Gemma tokenizer is approved under "
                     "Google's Gemma terms")},
    ],
    "source_rule": "a verified copy from a retained bundle on the host only",
    "use_scope": "development_simulation",
    "model_license": "gemma",
}


@dataclass(frozen=True)
class Pins:
    """Everything the restore may touch or accept, fixed in code."""

    target_dir: Path
    source_root: Path
    inventory_path: Path
    inventory_sha256: str

    @property
    def parent(self) -> Path:
        return self.target_dir.parent


PINS = Pins(
    target_dir=RETAINED_TOKENIZER_DIR,
    source_root=SOURCE_ROOT,
    inventory_path=INVENTORY,
    inventory_sha256=PI_AUXILIARY["inventory_file_sha256"],
)


class RestoreRefused(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass
class _Source:
    """The chosen archive, held open so the copy reads the bytes that were checked."""

    path: Path
    stream: BinaryIO
    archive: zipfile.ZipFile
    members: dict[str, zipfile.ZipInfo] = field(default_factory=dict)

    def close(self) -> None:
        self.archive.close()
        self.stream.close()


class _WritingReader:
    """Hands each block to the verifier and writes the same block to the staged file."""

    def __init__(self, source: Any, fd: int) -> None:
        self._source = source
        self._fd = fd

    def read(self, size: int = -1) -> bytes:
        block = self._source.read(size)
        view = memoryview(block)
        while view:
            view = view[os.write(self._fd, view):]
        return block


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _iso(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def _stamp(moment: datetime) -> str:
    return moment.strftime("%Y%m%dT%H%M%SZ")


def _disk_usage(path: str | os.PathLike[str]) -> Any:
    return shutil.disk_usage(path)


def _reservation_root() -> Path:
    return Path(os.environ.get(RESERVATION_ROOT_ENV) or DEFAULT_RESERVATION_ROOT)


def _parent_writable(path: Path) -> bool:
    return os.access(path, os.W_OK | os.X_OK)


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise RestoreRefused(code)


def _no_symlink_parents(path: Path) -> bool:
    return path.is_absolute() and not any(parent.is_symlink() for parent in path.parents)


def _open_directory(name: str | Path, *, dir_fd: int | None = None) -> int:
    return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=dir_fd)


def _short(error: BaseException) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]", "_", str(error).split("\n", 1)[0])[:200] or type(error).__name__


def load_inventory(pins: Pins) -> dict[str, Any]:
    """The reviewed six-file inventory, once its bytes match the authority's pin."""

    code = "g1_pi_language_assets_inventory_changed"
    path = pins.inventory_path
    _require(path.is_absolute() and not path.is_symlink(), code)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
        with os.fdopen(fd, "rb") as stream:
            raw = stream.read(1024 * 1024 + 1)
        inventory = pi_language_inventory(path)
    except (OSError, ValueError) as error:
        raise RestoreRefused(code) from error
    _require("sha256:" + hashlib.sha256(raw).hexdigest() == pins.inventory_sha256, code)
    _require(json.loads(raw.decode("utf-8")) == inventory, code)
    return inventory


# --- the folder as it is ---------------------------------------------------------------------------

def _file_sha256(name: str, dir_fd: int, size: int) -> str | None:
    if size > MAX_HASHED_FILE_BYTES:
        return None
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=dir_fd)
    except OSError:
        return None
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            return None
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _listing(directory_fd: int) -> dict[str, Any]:
    """Names, types, sizes and sha256 of what the folder holds; never a byte of content."""

    entries: list[dict[str, Any]] = []
    truncated = False

    def walk(fd: int, prefix: str, depth: int) -> None:
        nonlocal truncated
        try:
            names = sorted(os.listdir(fd))
        except OSError:
            entries.append({"path": prefix or "./", "type": "unreadable"})
            return
        for name in names:
            if len(entries) >= MAX_LISTED_ENTRIES:
                truncated = True
                return
            relative = prefix + name
            try:
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            except OSError:
                entries.append({"path": relative, "type": "unreadable"})
                continue
            if stat.S_ISLNK(info.st_mode):
                entries.append({"path": relative, "type": "symlink"})
            elif stat.S_ISDIR(info.st_mode):
                entries.append({"path": relative + "/", "type": "directory"})
                if depth >= MAX_LISTED_DEPTH:
                    truncated = True
                    continue
                try:
                    child = _open_directory(name, dir_fd=fd)
                except OSError:
                    entries[-1]["type"] = "unreadable"
                    continue
                try:
                    walk(child, relative + "/", depth + 1)
                finally:
                    os.close(child)
            elif stat.S_ISREG(info.st_mode):
                entries.append({"path": relative, "type": "file", "size_bytes": info.st_size,
                                "sha256": _file_sha256(name, fd, info.st_size)})
            else:
                entries.append({"path": relative, "type": "other"})

    walk(directory_fd, "", 1)
    return {"entries": entries, "truncated": truncated}


def _inventory_states(directory_fd: int, inventory: Mapping[str, Any]) -> list[dict[str, str]]:
    """Each inventory file's state in the folder, by the verifier's own rule."""

    rows = []
    for row in inventory["files"]:
        try:
            fd = os.open(row["path"], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                         dir_fd=directory_fd)
        except FileNotFoundError:
            rows.append({"path": row["path"], "state": "absent"})
            continue
        except OSError:
            rows.append({"path": row["path"], "state": "not_regular_file"})
            continue
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                state = "not_regular_file"
            else:
                try:
                    stream_identity(stream, row, size=info.st_size)
                    state = "matches"
                except ValueError as error:
                    state = "size_mismatch" if "size_mismatch" in str(error) else "digest_mismatch"
                except OSError:
                    state = "unreadable"
        rows.append({"path": row["path"], "state": state})
    return rows


def _verify_directory(pins: Pins, directory: Path) -> tuple[dict[str, Any] | None, str | None]:
    """``verify_tokenizer_assets`` on ``directory``: its receipt, or the reason it refused."""

    try:
        receipt = verify_tokenizer_assets(inventory_path=pins.inventory_path, asset_dir=directory)
    except (OSError, ValueError) as error:
        return None, _short(error)
    if receipt.get("inventory_sha256") != pins.inventory_sha256:
        return None, "inventory_sha256_changed"
    return receipt, None


def _inspect_target(parent_fd: int, pins: Pins, inventory: Mapping[str, Any], report: dict[str, Any]) -> str:
    """``absent``, ``verified`` or ``invalid``; anything but a real directory is a conflict."""

    target = report["target"]
    name = pins.target_dir.name
    try:
        info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        target.update(present=False, state="absent")
        return "absent"
    _require(stat.S_ISDIR(info.st_mode), "g1_pi_language_assets_target_conflict")
    try:
        target_fd = _open_directory(name, dir_fd=parent_fd)
    except OSError as error:
        raise RestoreRefused("g1_pi_language_assets_target_conflict") from error
    try:
        receipt, reason = _verify_directory(pins, pins.target_dir)
        if receipt is not None:
            target.update(present=True, state="verified", verification={"verified": True})
            return "verified"
        target.update(present=True, state="invalid", verification={"verified": False, "reason": reason},
                      inventory_files=_inventory_states(target_fd, inventory), existing=_listing(target_fd))
        return "invalid"
    finally:
        os.close(target_fd)


# --- the retained bundles --------------------------------------------------------------------------

def _classify(name: str, dir_fd: int) -> str | None:
    """``directory``, ``file``, ``symlink``, ``other``, or None when nothing is there."""

    try:
        info = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError:
        return "other"
    if stat.S_ISLNK(info.st_mode):
        return "symlink"
    if stat.S_ISDIR(info.st_mode):
        return "directory"
    return "file" if stat.S_ISREG(info.st_mode) else "other"


def _check_archive(stream: BinaryIO, inventory: Mapping[str, Any]
                   ) -> tuple[str, str | None, zipfile.ZipFile | None, dict[str, zipfile.ZipInfo]]:
    """Read only the inventory's members and check each; the archive stays open when all verify."""

    try:
        archive = zipfile.ZipFile(stream)
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, EOFError, ValueError):
        return "bundle_not_a_zip", None, None, {}
    wanted = {ARCHIVE_PREFIX + row["path"]: row["path"] for row in inventory["files"]}
    counts = Counter(info.filename for info in archive.infolist() if info.filename in wanted)
    members: dict[str, zipfile.ZipInfo] = {}
    reason: tuple[str, str | None] | None = None
    for row in inventory["files"]:
        member = ARCHIVE_PREFIX + row["path"]
        if counts[member] != 1:
            reason = ("bundle_member_missing" if counts[member] == 0 else "bundle_member_duplicate", row["path"])
            break
        members[row["path"]] = archive.getinfo(member)
    if reason is None:
        for row in inventory["files"]:
            info = members[row["path"]]
            try:
                with archive.open(info) as source:
                    stream_identity(source, row, size=info.file_size)
            except ValueError as error:
                if str(error).startswith("g1_pi_tokenizer_file_size_mismatch"):
                    reason = ("bundle_member_size_mismatch", row["path"])
                elif str(error).startswith("g1_pi_tokenizer_file_digest_mismatch"):
                    reason = ("bundle_member_digest_mismatch", row["path"])
                else:
                    reason = ("bundle_member_unreadable", row["path"])
                break
            except (zipfile.BadZipFile, OSError, EOFError, RuntimeError, NotImplementedError, zlib.error):
                reason = ("bundle_member_unreadable", row["path"])
                break
    if reason is not None:
        archive.close()
        return reason[0], reason[1], None, {}
    return "verified", None, archive, members


def _archive_sha256(fd: int, size: int) -> str:
    digest, offset = hashlib.sha256(), 0
    while offset < size:
        block = os.pread(fd, min(8 * CHUNK_BYTES, size - offset), offset)
        if not block:
            break
        digest.update(block)
        offset += len(block)
    return "sha256:" + digest.hexdigest()


def _check_bundle(bundle_fd: int, name: str, path: Path, inventory: Mapping[str, Any]
                  ) -> tuple[dict[str, Any], _Source | None]:
    row: dict[str, Any] = {"path": str(path)}
    kind = _classify(name, bundle_fd)
    if kind == "symlink":
        return {**row, "reason": "bundle_symlink_refused"}, None
    if kind != "file":
        return {**row, "reason": "bundle_not_regular_file"}, None
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=bundle_fd)
    except OSError:
        return {**row, "reason": "bundle_unreadable"}, None
    stream = os.fdopen(fd, "rb")
    keep = False
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return {**row, "reason": "bundle_not_regular_file"}, None
        row["size_bytes"] = info.st_size
        if info.st_size > MAX_BUNDLE_BYTES:
            return {**row, "reason": "bundle_too_large"}, None
        reason, member, archive, members = _check_archive(stream, inventory)
        if archive is None:
            return {**row, "reason": reason, **({"member": member} if member else {})}, None
        row.update(reason="verified", sha256=_archive_sha256(fd, info.st_size))
        keep = True
        return row, _Source(path=path, stream=stream, archive=archive, members=members)
    except OSError:
        return {**row, "reason": "bundle_unreadable"}, None
    finally:
        if not keep:
            stream.close()


def find_source(pins: Pins, inventory: Mapping[str, Any], report: dict[str, Any]) -> _Source:
    """The first retained archive, in sorted order, whose every inventory member verifies."""

    search: dict[str, Any] = {"root": str(pins.source_root), "bundle_names": list(BUNDLE_NAMES),
                              "archive_prefix": ARCHIVE_PREFIX, "campaign_directories": 0,
                              "checked": [], "truncated": False}
    report["source_search"] = search
    code = "g1_pi_language_assets_source_root_unavailable"
    _require(_no_symlink_parents(pins.source_root) and not pins.source_root.is_symlink(), code)
    try:
        root_fd = _open_directory(pins.source_root)
    except OSError as error:
        raise RestoreRefused(code) from error
    checked: list[dict[str, Any]] = search["checked"]
    try:
        try:
            names = sorted(name for name in os.listdir(root_fd) if CAMPAIGN_DIRECTORY.fullmatch(name))
        except OSError as error:
            raise RestoreRefused(code) from error
        search["campaign_directories"] = len(names)
        for name in names:
            campaign = pins.source_root / name
            bundle_dir = campaign / BUNDLE_DIRECTORY
            kind = _classify(name, root_fd)
            if kind == "symlink":
                checked.append({"path": str(campaign), "reason": "bundle_symlink_refused"})
                continue
            if kind != "directory":
                continue
            try:
                campaign_fd = _open_directory(name, dir_fd=root_fd)
            except OSError:
                checked.append({"path": str(campaign), "reason": "bundle_unreadable"})
                continue
            try:
                kind = _classify(BUNDLE_DIRECTORY, campaign_fd)
                if kind == "symlink":
                    checked.append({"path": str(bundle_dir), "reason": "bundle_symlink_refused"})
                    continue
                if kind != "directory":
                    continue
                try:
                    bundle_fd = _open_directory(BUNDLE_DIRECTORY, dir_fd=campaign_fd)
                except OSError:
                    checked.append({"path": str(bundle_dir), "reason": "bundle_unreadable"})
                    continue
                try:
                    for bundle_name in BUNDLE_NAMES:
                        if _classify(bundle_name, bundle_fd) is None:
                            continue
                        if len(checked) >= MAX_BUNDLES_CHECKED:
                            search["truncated"] = True
                            break
                        row, source = _check_bundle(bundle_fd, bundle_name, bundle_dir / bundle_name, inventory)
                        checked.append(row)
                        if source is not None:
                            report["source"] = {key: row[key] for key in ("path", "size_bytes", "sha256")}
                            report["source"]["members"] = [
                                {"path": member, "archive_path": ARCHIVE_PREFIX + member,
                                 "size_bytes": info.file_size} for member, info in source.members.items()]
                            return source
                finally:
                    os.close(bundle_fd)
            finally:
                os.close(campaign_fd)
            if search["truncated"]:
                break
    finally:
        os.close(root_fd)
    raise RestoreRefused("g1_pi_language_assets_no_verified_source")


# --- disk -----------------------------------------------------------------------------------------

def _need_bytes(inventory: Mapping[str, Any]) -> int:
    return sum(int(row["size_bytes"]) for row in inventory["files"]) + DISK_MARGIN_BYTES


def _plan_disk(need: int, pins: Pins, report: dict[str, Any]) -> None:
    try:
        available = int(disk_headroom(target_root=pins.parent, reservation_root=_reservation_root(),
                                      disk_usage=lambda path: _disk_usage(path))["available_bytes"])
    except ControlPlaneDiskBudgetError as error:
        report["disk"] = {"need_bytes": need, "sufficient": False, "detail": str(error).split(":", 1)[0]}
        raise RestoreRefused("g1_pi_language_assets_disk_insufficient") from error
    report["disk"] = {"need_bytes": need, "available_bytes": available, "sufficient": need <= available}
    _require(need <= available, "g1_pi_language_assets_disk_insufficient")


# --- writing --------------------------------------------------------------------------------------

def _absent(name: str, dir_fd: int) -> bool:
    return _classify(name, dir_fd) is None


def _rename_new(source: str, destination: str, dir_fd: int) -> None:
    """Rename within the parent; a destination that exists is never replaced."""

    _require(_absent(destination, dir_fd), "g1_pi_language_assets_target_conflict")
    os.rename(source, destination, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)


def _stage(source: _Source, inventory: Mapping[str, Any], staged_fd: int, written: list[str]) -> None:
    """Write each member into the staged folder, checking it again as it lands; fsync every file."""

    for row in inventory["files"]:
        info = source.members[row["path"]]
        fd = os.open(row["path"], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600,
                     dir_fd=staged_fd)
        written.append(row["path"])
        try:
            with source.archive.open(info) as member:
                stream_identity(_WritingReader(member, fd), row, size=info.file_size)
            os.fchmod(fd, FILE_MODE)
            os.fsync(fd)
        except (ValueError, zipfile.BadZipFile, EOFError, RuntimeError, NotImplementedError, zlib.error) as error:
            raise RestoreRefused("g1_pi_language_assets_copy_failed") from error
        finally:
            os.close(fd)
    os.fsync(staged_fd)


def _remove_staged(parent_fd: int, name: str, written: Sequence[str]) -> bool:
    """Remove only the files this run wrote and the folder it made; anything else stays."""

    try:
        staged_fd = _open_directory(name, dir_fd=parent_fd)
    except OSError:
        return False
    # A cleanup failure is reported as an incomplete rollback, never instead of the refusal.
    try:
        try:
            for file_name in written:
                try:
                    os.unlink(file_name, dir_fd=staged_fd)
                except FileNotFoundError:
                    pass
        finally:
            os.close(staged_fd)
        os.rmdir(name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except OSError:
        return False
    return True


def _write_receipt(parent_fd: int, name: str, document: Mapping[str, Any]) -> None:
    raw = (json.dumps(document, sort_keys=True) + "\n").encode("utf-8")
    temporary = f".{name}.{secrets.token_hex(4)}.partial"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600,
                 dir_fd=parent_fd)
    try:
        view = memoryview(raw)
        while view:
            view = view[os.write(fd, view):]
        os.fchmod(fd, FILE_MODE)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        # A link never replaces an existing name.
        os.link(temporary, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd, follow_symlinks=False)
    finally:
        os.unlink(temporary, dir_fd=parent_fd)
    os.fsync(parent_fd)


# --- plan and apply -------------------------------------------------------------------------------

def _open_parent(pins: Pins) -> int:
    code = "g1_pi_language_assets_target_conflict"
    _require(_no_symlink_parents(pins.target_dir), code)
    try:
        return _open_directory(pins.parent)
    except OSError as error:
        raise RestoreRefused(code) from error


def _names(pins: Pins, moment: datetime) -> dict[str, str]:
    stamp = _stamp(moment)
    name = pins.target_dir.name
    return {"aside": f".{name}.invalid-{stamp}", "staged": f".{name}.restore-{stamp}-{secrets.token_hex(4)}.partial",
            "receipt": f"{RECEIPT_PREFIX}{stamp}.json"}


def _plan(pins: Pins, inventory: Mapping[str, Any], report: dict[str, Any]) -> str:
    parent_fd = _open_parent(pins)
    try:
        state = _inspect_target(parent_fd, pins, inventory, report)
    finally:
        os.close(parent_fd)
    report["restore_needed"] = state != "verified"
    if state == "verified":
        report["no_op"] = True
        return "planned"
    find_source(pins, inventory, report).close()
    _plan_disk(_need_bytes(inventory), pins, report)
    report["target"]["parent_writable"] = _parent_writable(pins.parent)
    _require(report["target"]["parent_writable"], "g1_pi_language_assets_target_parent_unwritable")
    if state == "invalid":
        report["set_aside"] = {"performed": False,
                               "would_be": str(pins.parent / _names(pins, _now())["aside"])}
    return "planned"


def _apply(pins: Pins, inventory: Mapping[str, Any], report: dict[str, Any]) -> str:
    parent_fd = _open_parent(pins)
    name = pins.target_dir.name
    source: _Source | None = None
    reservation = None
    disk_outcome = "failed"
    staged: str | None = None
    written: list[str] = []
    aside: str | None = None
    placed = False
    try:
        try:
            fcntl.flock(parent_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RestoreRefused("g1_pi_language_assets_restore_busy") from error
        state = _inspect_target(parent_fd, pins, inventory, report)
        report["restore_needed"] = state != "verified"
        if state == "verified":
            report["no_op"] = True
            return "already_restored"
        source = find_source(pins, inventory, report)
        report["target"]["parent_writable"] = _parent_writable(pins.parent)
        _require(report["target"]["parent_writable"], "g1_pi_language_assets_target_parent_unwritable")
        need = _need_bytes(inventory)
        try:
            reservation = reserve_control_plane_disk(
                DISK_ROLE, target_root=pins.parent, expected_bytes=need,
                reservation_root=_reservation_root(), disk_usage=lambda path: _disk_usage(path))
        except ControlPlaneDiskBudgetError as error:
            report["disk"] = {"need_bytes": need, "sufficient": False, "detail": str(error).split(":", 1)[0]}
            raise RestoreRefused("g1_pi_language_assets_disk_insufficient") from error
        report["disk"] = {"need_bytes": need, "available_bytes": reservation.available_bytes,
                          "sufficient": True, "reserved": True}
        moment = _now()
        names = _names(pins, moment)
        for candidate in (names["staged"], names["receipt"]) + ((names["aside"],) if state == "invalid" else ()):
            _require(_absent(candidate, parent_fd), "g1_pi_language_assets_target_conflict")
        os.mkdir(names["staged"], DIRECTORY_MODE, dir_fd=parent_fd)
        staged = names["staged"]
        staged_fd = _open_directory(staged, dir_fd=parent_fd)
        try:
            os.fchmod(staged_fd, DIRECTORY_MODE)
            _stage(source, inventory, staged_fd, written)
        finally:
            os.close(staged_fd)
        os.fsync(parent_fd)
        receipt, reason = _verify_directory(pins, pins.parent / staged)
        report["staged"] = {"path": str(pins.parent / staged), "verified": receipt is not None,
                            **({"files": receipt["files"]} if receipt else {"reason": reason})}
        _require(receipt is not None, "g1_pi_language_assets_verification_failed")
        if state == "invalid":
            _rename_new(name, names["aside"], parent_fd)
            aside = names["aside"]
            report["set_aside"] = {"performed": True, "path": str(pins.parent / aside)}
        _rename_new(staged, name, parent_fd)
        placed, staged = True, None
        os.fsync(parent_fd)
        final, reason = _verify_directory(pins, pins.target_dir)
        report["verification"] = {"verified": final is not None,
                                  **({"files": final["files"]} if final else {"reason": reason})}
        _require(final is not None, "g1_pi_language_assets_verification_failed")
        # ``target`` keeps the folder as it was found; ``verification`` is the folder now.
        report["written"] = [str(pins.target_dir / row["path"]) for row in inventory["files"]]
        report["restored_at"] = _iso(moment)
        report["outcome"] = "restored"
        receipt_path = pins.parent / names["receipt"]
        report["receipt_path"] = str(receipt_path)
        try:
            _write_receipt(parent_fd, names["receipt"], report)
        except OSError as error:
            # The restore stands; the door's result is still the full record.
            report["receipt_path"] = None
            report["receipt_flag"] = {"written": False, "error_type": type(error).__name__}
        disk_outcome = "completed"
        return "restored"
    except BaseException:
        _roll_back(parent_fd, pins, report, staged=staged, written=written, aside=aside, placed=placed)
        raise
    finally:
        if reservation is not None:
            reservation.release(outcome=disk_outcome)
        if source is not None:
            source.close()
        os.close(parent_fd)


def _roll_back(parent_fd: int, pins: Pins, report: dict[str, Any], *, staged: str | None,
               written: list[str], aside: str | None, placed: bool) -> None:
    """Put the folder back as it was; remove only what this run made. Nothing else is deleted."""

    name = pins.target_dir.name
    made = placed or aside is not None or staged is not None
    complete = True
    if placed:
        # Move this run's copy off the name before the original goes back to it.
        staged = f".{name}.unverified-{secrets.token_hex(4)}.partial"
        try:
            _rename_new(name, staged, parent_fd)
        except (OSError, RestoreRefused):
            complete, staged = False, None
    if aside is not None and complete:
        try:
            _rename_new(aside, name, parent_fd)
            report["set_aside"] = {"performed": False, "returned": True}
        except (OSError, RestoreRefused):
            complete = False
    if staged is not None and not _remove_staged(parent_fd, staged, written):
        complete = False
    if made:
        report["written"] = []
        report["rolled_back"] = {"complete": complete}


def _base_report(pins: Pins, apply: bool) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _iso(_now()),
        "mode": "apply" if apply else "plan",
        "outcome": None,
        "code": None,
        "authorization": AUTHORIZATION,
        "inventory": {"path": str(pins.inventory_path), "sha256": pins.inventory_sha256, "verified": False},
        "target": {"directory": str(pins.target_dir), "present": None, "state": None},
        "restore_needed": None,
        "source_search": None,
        "source": None,
        "disk": None,
        "set_aside": None,
        "staged": None,
        "written": [],
        "verification": None,
        "receipt_path": None,
        "network_used": False,
        "provider_mutation_performed": False,
        "compute_started": False,
        "file_contents_recorded": False,
    }


def restore(*, apply: bool = False, pins: Pins | None = None) -> dict[str, Any]:
    """Plan (default) or apply the restore; always returns the result document."""

    pins = pins or PINS
    report = _base_report(pins, apply)
    try:
        inventory = load_inventory(pins)
        report["inventory"]["verified"] = True
        outcome = _apply(pins, inventory, report) if apply else _plan(pins, inventory, report)
    except RestoreRefused as refusal:
        report.update(outcome="refused", code=refusal.code)
        return report
    except Exception as error:  # noqa: BLE001 - one fixed code, never the message
        report.update(outcome="refused", code="g1_pi_language_assets_restore_failed",
                      error_type=type(error).__name__)
        return report
    report["outcome"] = outcome
    return report


def _write_result(path: Path, document: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(document, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _terminate(_signum: int, _frame: Any) -> None:
    # The door's RuntimeMaxSec sends SIGTERM: unwind so a staged copy is removed and the folder put back.
    raise SystemExit(143)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="restore; without this only a plan is made")
    parser.add_argument("--result", type=Path, help="also write the result document here")
    args = parser.parse_args(argv)
    signal.signal(signal.SIGTERM, _terminate)
    document = restore(apply=args.apply)
    if args.result is not None:
        _write_result(args.result, document)
    print(json.dumps(document, sort_keys=True))
    if document["outcome"] != "refused":
        return 0
    return 1 if document["code"] == "g1_pi_language_assets_restore_failed" else 2


if __name__ == "__main__":
    sys.exit(main())
