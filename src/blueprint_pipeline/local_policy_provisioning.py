"""Exact bounded downloads and offline installation shared by local policies."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import time

DEADLINE_ENV = "BLUEPRINT_POLICY_CANARY_HARD_DEADLINE_EPOCH"
ALLOWED_ORIGINS = (
    "https://huggingface.co", "https://cdn-lfs.huggingface.co", "https://cdn-lfs-us-1.hf.co",
    "https://cdn-lfs.hf.co", "https://cas-bridge.xethub.hf.co", "https://us.aws.cdn.hf.co",
    "https://github.com", "https://release-assets.githubusercontent.com", "https://objects.githubusercontent.com",
    "https://files.pythonhosted.org", "https://download.pytorch.org", "https://download-r2.pytorch.org",
)


def remaining_seconds(deadline: float) -> float:
    remaining = deadline - time.time()
    if not math.isfinite(remaining) or remaining <= 0:
        raise ValueError("local_policy_provisioning_deadline_exceeded")
    return remaining


def _sha256(path: Path, *, deadline: float | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            if deadline is not None:
                remaining_seconds(deadline)
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


@dataclass
class TransferBudget:
    maximum_bytes: int
    attempted_bytes: int = 0

    def reserve(self, size: int) -> None:
        if type(size) is not int or size <= 0 or self.attempted_bytes + size > self.maximum_bytes:
            raise ValueError("local_policy_total_download_bound_exceeded")
        self.attempted_bytes += size


# A bounded fetch has at most 600 seconds for its whole response, so a 5 GB checkpoint shard
# needed more than 8 MB/s for its whole length; a slower rented link refused it
# (SecurityValidationError: total time limit). A file larger than this is fetched in byte
# ranges of this size, each its own bounded request (a floor of about 1.8 MB/s), appended in
# order, and verified whole by size and sha256 as before.
RANGE_CHUNK_BYTES = 1024**3


def _fetch_ranges(fetch, url: str, *, size: int, temporary: Path, deadline: float) -> None:
    part = temporary.with_name(temporary.name + ".range")
    if part.exists() or part.is_symlink():
        raise ValueError("local_policy_download_partial_exists")
    with temporary.open("xb"):
        pass
    try:
        offset = 0
        while offset < size:
            last = min(size, offset + RANGE_CHUNK_BYTES) - 1
            response = fetch(url, allowed_origins=ALLOWED_ORIGINS,
                timeout_seconds=max(1, min(600, int(remaining_seconds(deadline)))), max_bytes=last - offset + 1,
                max_redirects=5, output_path=part, headers={"Range": f"bytes={offset}-{last}"})
            # A server that ignores the range answers 200 with the whole file, which the byte bound
            # refuses; a short or partial answer is refused here.
            if getattr(response, "status", None) != 206 or part.stat().st_size != last - offset + 1:
                raise ValueError("local_policy_download_range_invalid")
            with temporary.open("ab") as output, part.open("rb") as chunk:
                shutil.copyfileobj(chunk, output, 8 * 1024 * 1024)
            part.unlink()
            offset = last + 1
    finally:
        part.unlink(missing_ok=True)


def fetch_exact(row: dict, destination: Path, *, deadline: float, fetch=None,
                budget: TransferBudget | None = None) -> dict:
    """No retry, ambient credentials, arbitrary output tree or unbounded reads."""
    from .core.security_controls import fetch_bounded_https

    size = row["size_bytes"]
    expected = row["sha256"]
    if isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected):
        expected = "sha256:" + expected
    if type(size) is not int or size < 1 or not isinstance(expected, str) or len(expected) != 71:
        raise ValueError("local_policy_download_identity_invalid")
    if destination.is_symlink() or any(parent.is_symlink() for parent in destination.parents):
        raise ValueError("local_policy_download_path_invalid")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == size and _sha256(destination, deadline=deadline) == expected:
            return {"status": "verified", "size_bytes": size, "sha256": expected, "cache_hit": True}
        raise ValueError("local_policy_download_existing_file_changed")
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("local_policy_download_partial_exists")
    try:
        if budget is not None:
            budget.reserve(size)
        if size > RANGE_CHUNK_BYTES:
            _fetch_ranges(fetch or fetch_bounded_https, row["url"], size=size, temporary=temporary,
                          deadline=deadline)
        else:
            (fetch or fetch_bounded_https)(row["url"], allowed_origins=ALLOWED_ORIGINS,
                timeout_seconds=max(1, min(600, int(remaining_seconds(deadline)))), max_bytes=size,
                max_redirects=5, output_path=temporary)
        remaining_seconds(deadline)
        if temporary.stat().st_size != size or _sha256(temporary, deadline=deadline) != expected:
            raise ValueError("local_policy_download_bytes_changed")
        os.replace(temporary, destination)
        return {"status": "verified", "size_bytes": size, "sha256": expected, "cache_hit": False}
    except Exception as exc:
        # Which public file failed, for provisioning_failure_code; never a URL or a path.
        exc.local_policy_download = destination.name
        raise
    finally:
        temporary.unlink(missing_ok=True)


def provisioning_failure_code(prefix: str, exc: BaseException) -> str:
    """The typed blocker a provisioning entry point records for a failure.

    A failure that is already one of this provider's typed codes is kept. Any other is
    ``<prefix>_provisioning_failed:<class>``, plus a refused download's fixed security reason
    (``SecurityValidationError`` messages are fixed strings, never a URL) and the file it was
    fetching, both reduced to lowercase code characters.
    """

    from .core.security_controls import SecurityValidationError

    reason = str(exc)
    if re.fullmatch(rf"(?:{re.escape(prefix)}|local_policy)_[a-z0-9_]{{1,120}}", reason):
        return reason
    parts = [f"{prefix}_provisioning_failed", type(exc).__name__]
    if isinstance(exc, SecurityValidationError):
        parts.append(re.sub(r"[^a-z0-9]+", "_", reason.lower()).strip("_")[:60])
    download = getattr(exc, "local_policy_download", None)
    if isinstance(download, str):
        parts.append(re.sub(r"[^a-z0-9._-]+", "_", download.lower()).strip("_")[:60])
    return ":".join(part for part in parts if part)


def _command(argv: list[str], *, deadline: float, env: dict | None = None) -> None:
    if env is None:
        # Isaac's launcher exports its own import and shared-library paths.
        # A separate interpreter must not inherit that framework during even
        # its version check or offline pip installation.
        env = {key: value for key, value in os.environ.items()
               if key not in {"PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"}}
        if "LD_LIBRARY_PATH" in env:
            env["LD_LIBRARY_PATH"] = ":".join(part for part in env["LD_LIBRARY_PATH"].split(":") if "/isaac-sim" not in part)
        env["PYTHONNOUSERSITE"] = "1"
    subprocess.run(argv, check=True, timeout=remaining_seconds(deadline), env=env)


def _extract_interpreter(path: Path, destination: Path, *, maximum_bytes: int) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        if sum(member.size for member in members) > maximum_bytes:
            raise ValueError("local_policy_interpreter_expansion_exceeds_bound")
        # Python's data filter refuses outside links, devices and absolute paths.
        archive.extractall(destination, filter="data")


def _install_wheels(python: Path, rows: list[dict], wheels: Path, *, deadline: float,
                    target: Path | None = None) -> None:
    requirements = wheels / ("client-requirements.txt" if target else "server-requirements.txt")
    requirements.write_text("\n".join(str(wheels / row["filename"]) + " --hash=sha256:" + row["sha256"].removeprefix("sha256:")
                                      for row in rows) + "\n")
    argv = [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-index", "--no-deps",
            "--require-hashes", "--no-compile", "-r", str(requirements)]
    if target is not None:
        argv += ["--target", str(target)]
    # The thin client is intentionally installed with Isaac's own interpreter,
    # whose launcher paths must remain intact. Only the standalone server
    # interpreter uses the default isolated bootstrap environment.
    env = {**os.environ, "PYTHONNOUSERSITE": "1"} if target is not None else None
    _command(argv, deadline=deadline, env=env)
