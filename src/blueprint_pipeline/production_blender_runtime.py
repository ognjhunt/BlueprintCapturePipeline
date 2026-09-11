"""Install or validate the pinned portable headless Blender runtime without sudo.

Installation is explicit; existing runtimes must match their receipt and version.
The executable digest detects subsequent binary mutation. This is not a complete
shared-library integrity attestation. No provider, container or package manager
is used. Production should keep this runtime directory service-owner writable only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from typing import Callable
import urllib.request

VERSION = "5.2.1"
ARCHIVE_NAME = f"blender-{VERSION}-linux-x64.tar.xz"
ARCHIVE_SHA256 = "a31f524fa99a527d3d52b7f5aaa68c34e1a19d5a1c9473f79c5cc610fd5b10e9"
ARCHIVE_URL = f"https://download.blender.org/release/Blender5.2/{ARCHIVE_NAME}"
DEFAULT_ROOT = Path(f"/var/lib/blueprint/task-evaluation-inputs/toolchains/blender/{VERSION}")
RECEIPT_NAME = "blender_runtime_receipt.json"
EXECUTABLE_RELATIVE = f"blender-{VERSION}-linux-x64/blender"
MAX_ARCHIVE_BYTES = 1024**3
MAX_EXTRACTED_BYTES = 4 * 1024**3
DISK_RESERVE_BYTES = 512 * 1024**2
DOWNLOAD_TIMEOUT_SECONDS = 600


class BlenderRuntimeError(RuntimeError):
    """Runtime identity, platform, archive or resource admission failed."""


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def _download_archive(url: str, path: Path) -> None:
    deadline = time.monotonic() + DOWNLOAD_TIMEOUT_SECONDS
    size = 0
    request = urllib.request.Request(url, headers={
        "User-Agent": "BlueprintCapturePipeline/1.0 (portable Blender runtime installer)"
    })
    with urllib.request.urlopen(request, timeout=30) as response, path.open("xb") as output:
        if int(response.headers.get("Content-Length", "0")) > MAX_ARCHIVE_BYTES:
            raise BlenderRuntimeError("archive_download_size_limit")
        while True:
            if time.monotonic() > deadline:
                raise BlenderRuntimeError("archive_download_time_limit")
            chunk = response.read(1024**2)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_ARCHIVE_BYTES:
                raise BlenderRuntimeError("archive_download_size_limit")
            output.write(chunk)


def _safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, mode="r:xz") as bundle:
        members = []
        extracted_bytes = 0
        for member in bundle:
            members.append(member)
            extracted_bytes += member.size
            if len(members) > 100_000:
                raise BlenderRuntimeError("archive_member_count_limit")
            if extracted_bytes > MAX_EXTRACTED_BYTES:
                raise BlenderRuntimeError("archive_extracted_size_limit")
        root = destination.resolve()
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise BlenderRuntimeError("archive_path_escape")
            if not (member.isfile() or member.isdir() or member.issym()):
                raise BlenderRuntimeError("archive_special_or_hardlink_member")
            if member.issym():
                link = PurePosixPath(member.linkname)
                target = (root / name.parent / member.linkname).resolve()
                if link.is_absolute() or not target.is_relative_to(root):
                    raise BlenderRuntimeError("archive_symlink_escape")
        # The data filter also checks actual resolved parents during extraction,
        # preventing escapes through previously extracted symlinks.
        try:
            bundle.extractall(destination, members=members, filter="data")
        except (tarfile.TarError, OSError) as exc:
            raise BlenderRuntimeError(f"archive_extraction_refused:{exc}") from exc


def _version(executable: Path, runner: Callable = subprocess.run) -> str:
    result = runner([str(executable), "--background", "--factory-startup", "--version"],
                    capture_output=True, text=True, timeout=30, check=False)
    if result.returncode != 0:
        raise BlenderRuntimeError(f"blender_version_command_failed:{result.returncode}:{result.stderr[:2000]}")
    output = result.stdout.strip()
    first_line = output.splitlines()[0] if output else ""
    banner = rf"Blender {re.escape(VERSION)}(?: LTS)?(?: \(hash [0-9a-f]+ built [0-9: -]+\))?"
    if re.fullmatch(banner, first_line) is None:
        raise BlenderRuntimeError("blender_version_mismatch")
    return output


def _check_platform() -> None:
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise BlenderRuntimeError("requires_linux_x64")


def validate_runtime(root: Path = DEFAULT_ROOT, *, runner: Callable = subprocess.run) -> dict:
    """Check the saved archive identity, executable SHA256 and headless version."""
    _check_platform()
    root = Path(root)
    if root.is_symlink():
        raise BlenderRuntimeError("runtime_root_symlink")
    receipt_path = root / RECEIPT_NAME
    if receipt_path.is_symlink():
        raise BlenderRuntimeError("runtime_receipt_symlink")
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, ValueError) as exc:
        raise BlenderRuntimeError("runtime_receipt_missing_or_invalid") from exc
    if not isinstance(receipt, dict) or any(receipt.get(key) != expected for key, expected in {
        "schema_version": "production_blender_runtime.v1", "version": VERSION,
        "archive_url": ARCHIVE_URL, "archive_sha256": ARCHIVE_SHA256,
        "executable_relative_path": EXECUTABLE_RELATIVE,
    }.items()):
        raise BlenderRuntimeError("runtime_receipt_identity_mismatch")
    executable = root / EXECUTABLE_RELATIVE
    if (executable.is_symlink() or not executable.is_file()
            or not executable.resolve().is_relative_to(root.resolve())
            or not os.access(executable, os.X_OK)):
        raise BlenderRuntimeError("runtime_executable_missing_or_unsafe")
    if _digest(executable) != receipt.get("executable_sha256"):
        raise BlenderRuntimeError("runtime_executable_digest_mismatch")
    if _version(executable, runner) != receipt.get("version_output"):
        raise BlenderRuntimeError("runtime_version_receipt_mismatch")
    return {**receipt, "executable": str(executable.resolve())}


def install_runtime(
    root: Path = DEFAULT_ROOT, *, downloader: Callable = _download_archive,
    runner: Callable = subprocess.run,
) -> dict:
    """Explicit bounded install; never repair/overwrite a mutated existing runtime."""
    _check_platform()
    root = Path(root)
    if root.is_symlink():
        raise BlenderRuntimeError("runtime_root_symlink")
    if root.exists():
        return validate_runtime(root, runner=runner)
    parent = root.parent
    parent.mkdir(parents=True, exist_ok=True)
    required = MAX_ARCHIVE_BYTES + MAX_EXTRACTED_BYTES + DISK_RESERVE_BYTES
    if shutil.disk_usage(parent).free < required:
        raise BlenderRuntimeError("insufficient_disk_headroom")
    with tempfile.TemporaryDirectory(prefix=".blender-install-", dir=parent) as temporary:
        staging = Path(temporary)
        archive = staging / ARCHIVE_NAME
        downloader(ARCHIVE_URL, archive)
        if archive.is_symlink() or not archive.is_file() or archive.stat().st_size > MAX_ARCHIVE_BYTES:
            raise BlenderRuntimeError("archive_download_size_or_path_invalid")
        if _digest(archive) != ARCHIVE_SHA256:
            raise BlenderRuntimeError("archive_digest_mismatch")
        payload = staging / "runtime"
        payload.mkdir()
        _safe_extract(archive, payload)
        executable = payload / EXECUTABLE_RELATIVE
        if (executable.is_symlink() or not executable.is_file()
                or not executable.resolve().is_relative_to(payload.resolve())
                or not os.access(executable, os.X_OK)):
            raise BlenderRuntimeError("archive_executable_missing_or_unsafe")
        try:
            version_output = _version(executable, runner)
        except (BlenderRuntimeError, OSError, subprocess.SubprocessError) as exc:
            # Keep verified bytes for ldd/diagnosis; do not redownload to find the next library.
            retained = parent / f".blender-failed-{time.time_ns()}"
            retained.mkdir(mode=0o700)
            archive.rename(retained / ARCHIVE_NAME)
            payload.rename(retained / "runtime")
            (retained / "failure.json").write_text(json.dumps({
                "error": str(exc), "archive_sha256": ARCHIVE_SHA256,
                "executable_sha256": _digest(retained / "runtime" / EXECUTABLE_RELATIVE),
            }, indent=2) + "\n")
            raise BlenderRuntimeError(f"{exc}; retained_verified_staging={retained}") from exc
        receipt = dict(schema_version="production_blender_runtime.v1", version=VERSION,
                       archive_url=ARCHIVE_URL, archive_sha256=ARCHIVE_SHA256,
                       archive_size_bytes=archive.stat().st_size,
                       executable_relative_path=EXECUTABLE_RELATIVE,
                       executable_sha256=_digest(executable), version_output=version_output)
        (payload / RECEIPT_NAME).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        # Cooperating installers use an exclusive lock; never overwrite a winner.
        lock = parent / f".{root.name}.install.lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise BlenderRuntimeError("runtime_install_locked") from exc
        try:
            os.close(fd)
            if root.exists() or root.is_symlink():
                raise BlenderRuntimeError("runtime_destination_appeared")
            payload.rename(root)
        finally:
            lock.unlink()
    return validate_runtime(root, runner=runner)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--install", action="store_true", help="Explicitly download and install if absent")
    args = parser.parse_args(argv)
    try:
        receipt = install_runtime(args.root) if args.install else validate_runtime(args.root)
    except (BlenderRuntimeError, OSError, subprocess.SubprocessError, tarfile.TarError) as exc:
        parser.exit(1, f"Blender runtime refused: {exc}\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
