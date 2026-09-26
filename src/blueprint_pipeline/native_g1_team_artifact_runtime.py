"""Probe a digest-pinned noncontainer team policy without site observations.

ADP-050 Day 28 development seam. The trusted operator supplies already fetched
archive bytes; this module never dereferences the team's URL. It verifies the
registered digest, extracts regular files only, and runs the JSONL entrypoint in
a networkless Linux namespace. Synthetic compatibility is not runtime, rights,
paid-launch, or task-outcome qualification.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_g1_team_policy_conformance import run_g1_team_policy_synthetic_conformance
from .native_g1_team_policy_jsonl_client import NativeG1TeamPolicyJsonlClient
from .team_policy_delivery_profile import validate_team_policy_delivery_profile


CONFORMANCE_FILENAME = "native_g1_team_policy_synthetic_conformance.v1.json"
TEARDOWN_SCHEMA = "native_g1_team_artifact_teardown.v1"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_FILES = 20_000
_MAX_ARCHIVE_BYTES = 16 * 1024**3
_MAX_UNPACKED_BYTES = 32 * 1024**3
_MIN_FREE_AFTER_EXTRACT = 8 * 1024**3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _archive_members(archive: tarfile.TarFile) -> tuple[list[tarfile.TarInfo], int]:
    """Reject links, special files, traversal, duplicates, and archive bombs."""

    members = archive.getmembers()
    if len(members) > _MAX_FILES:
        raise ValueError("g1_team_artifact_member_limit_exceeded")
    total = 0
    seen: set[str] = set()
    for member in members:
        name = PurePosixPath(member.name)
        if (
            name.is_absolute()
            or not name.parts
            or any(part in {"", ".", ".."} for part in name.parts)
            or not (member.isfile() or member.isdir())
            or member.size < 0
            or str(name) in seen
        ):
            raise ValueError("g1_team_artifact_member_unsafe")
        seen.add(str(name))
        if member.isfile():
            total += member.size
            if total > _MAX_UNPACKED_BYTES:
                raise ValueError("g1_team_artifact_unpacked_limit_exceeded")
    return members, total


def _extract_regular_archive(archive_path: Path, destination: Path) -> None:
    """Copy validated bytes ourselves; never invoke tarfile.extract()."""

    with tarfile.open(archive_path, "r:*") as archive:
        members, total = _archive_members(archive)
        if shutil.disk_usage(destination).free < total + _MIN_FREE_AFTER_EXTRACT:
            raise ValueError("g1_team_artifact_capacity_insufficient")
        for member in members:
            relative = PurePosixPath(member.name)
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True, mode=0o755)
                continue
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("g1_team_artifact_member_unreadable")
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o555)
            with os.fdopen(descriptor, "wb") as output, source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            if target.stat().st_size != member.size:
                raise ValueError("g1_team_artifact_member_size_mismatch")
            target.chmod(0o555)


def isolated_artifact_command(*, artifact_root: Path, entrypoint: str) -> list[str]:
    """Expose only a read-only artifact and standard runtimes to bubblewrap."""

    relative = PurePosixPath(entrypoint)
    if (
        not artifact_root.is_absolute()
        or artifact_root.is_symlink()
        or not artifact_root.is_dir()
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or not (artifact_root.joinpath(*relative.parts)).is_file()
    ):
        raise ValueError("g1_team_artifact_entrypoint_invalid")
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise ValueError("g1_team_artifact_sandbox_unavailable")
    command = [
        bwrap, "--unshare-all", "--die-with-parent", "--new-session",
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--dir", "/etc", "--ro-bind", str(artifact_root), "/work",
        "--uid", "65534", "--gid", "65534", "--cap-drop", "ALL",
    ]
    for system_path in ("/usr", "/bin", "/lib", "/lib64", "/etc/ld.so.cache"):
        if Path(system_path).exists():
            command.extend(["--ro-bind", system_path, system_path])
    command.extend([
        "--chdir", "/work", "--setenv", "HOME", "/tmp",
        "--setenv", "XDG_CACHE_HOME", "/tmp", "--unsetenv", "PYTHONPATH",
        "--unsetenv", "LD_PRELOAD", "--", "/work/" + str(relative),
    ])
    return command


class NativeG1TeamArtifactLease:
    """Own one namespaced process and retain a typed teardown receipt."""

    def __init__(
        self, *, process: subprocess.Popen[bytes], client: NativeG1TeamPolicyJsonlClient,
        output_dir: Path, profile_digest: str, artifact_sha256: str, stderr_file: Any,
    ) -> None:
        self.process = process
        self.client = client
        self.output_dir = output_dir
        self.profile_digest = profile_digest
        self.artifact_sha256 = artifact_sha256
        self._stderr_file = stderr_file
        self._closed: dict[str, Any] | None = None

    def close(self) -> dict[str, Any]:
        if self._closed is not None:
            return self._closed
        error: str | None = None
        try:
            if self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired) as exc:
            error = type(exc).__name__
        finally:
            self._stderr_file.close()
        receipt = {
            "schema_version": TEARDOWN_SCHEMA,
            "status": "process_exited" if self.process.poll() is not None and error is None else "teardown_blocked",
            "profile_digest": self.profile_digest,
            "artifact_sha256": self.artifact_sha256,
            "process_exit_code": self.process.poll(),
            "process_close_error": error,
            "raw_stderr_quarantined": True,
            "provider_teardown_verified": False,
            "claim_ceiling": "planning_only",
        }
        receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
        with (self.output_dir / (TEARDOWN_SCHEMA + ".json")).open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        self._closed = receipt
        return receipt


def launch_g1_team_artifact_synthetic_probe(
    *, profile: Mapping[str, Any], trusted_setup: Mapping[str, Any],
    authenticated_owner: Mapping[str, str], operator_approved_profile_digest: str,
    operator_approved_artifact_sha256: str, staged_artifact_path: Path,
    output_dir: Path,
) -> tuple[NativeG1TeamArtifactLease, dict[str, Any]]:
    """Verify operator-bound bytes and prove the synthetic G1 JSONL wire."""

    bound = validate_team_policy_delivery_profile(
        profile, trusted_setup=trusted_setup, authenticated_owner=authenticated_owner
    )
    delivery = bound["delivery"]
    if (
        sys.platform != "linux"
        or os.geteuid() != 0
        or delivery["mode"] != "noncontainer_artifact"
        or bound["profile_digest"] != operator_approved_profile_digest
        or delivery["artifact_sha256"] != operator_approved_artifact_sha256
        or _DIGEST.fullmatch(operator_approved_artifact_sha256) is None
        or not isinstance(staged_artifact_path, Path)
        or not staged_artifact_path.is_absolute()
        or staged_artifact_path.is_symlink()
        or not staged_artifact_path.is_file()
        or staged_artifact_path.stat().st_size > _MAX_ARCHIVE_BYTES
        or not isinstance(output_dir, Path)
        or not output_dir.is_absolute()
        or output_dir.exists()
        or output_dir.is_symlink()
    ):
        raise ValueError("g1_team_artifact_admission_invalid")
    if _sha256(staged_artifact_path) != operator_approved_artifact_sha256:
        raise ValueError("g1_team_artifact_digest_mismatch")
    output_dir.mkdir(mode=0o700)
    artifact_root = output_dir / "artifact"
    artifact_root.mkdir(mode=0o755)
    _extract_regular_archive(staged_artifact_path, artifact_root)
    command = isolated_artifact_command(
        artifact_root=artifact_root, entrypoint=delivery["entrypoint"]
    )
    descriptor = os.open(
        output_dir / "team_policy_raw_stderr.quarantined.log",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600,
    )
    stderr_file = os.fdopen(descriptor, "wb")
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr_file,
            bufsize=0, start_new_session=True, close_fds=True,
            env={
                "PATH": "/usr/bin:/bin", "HOME": "/tmp",
                "XDG_CACHE_HOME": "/tmp", "LANG": "C.UTF-8",
            },
        )
        client = NativeG1TeamPolicyJsonlClient(
            process, timeout_seconds=30.0, profile_digest=bound["profile_digest"]
        )
    except BaseException:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
        stderr_file.close()
        raise
    assert process is not None
    lease = NativeG1TeamArtifactLease(
        process=process, client=client, output_dir=output_dir,
        profile_digest=bound["profile_digest"], artifact_sha256=operator_approved_artifact_sha256,
        stderr_file=stderr_file,
    )
    try:
        conformance = run_g1_team_policy_synthetic_conformance(
            profile=bound, trusted_setup=trusted_setup,
            authenticated_owner=authenticated_owner, policy_client=client,
        )
        with (output_dir / CONFORMANCE_FILENAME).open("x", encoding="utf-8") as stream:
            json.dump(conformance, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except BaseException:
        lease.close()
        raise
    return lease, conformance


__all__ = [
    "NativeG1TeamArtifactLease", "isolated_artifact_command",
    "launch_g1_team_artifact_synthetic_probe",
]
