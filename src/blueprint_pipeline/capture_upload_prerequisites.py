"""Provision the configured capture store and its declared system scanner.

This host setup never processes customer footage or allocates provider resources.
The existing service-user scanner controls remain the admission check.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import pwd
import shlex
import stat
import subprocess
import sys
import time
from typing import Any, Callable
import uuid

from .capture_upload_runtime import CaptureUploadRuntimeError


SYSTEM_SCANNER = Path("/usr/bin/clamscan")
SYSTEM_UPDATER = Path("/usr/bin/freshclam")
DEFINITION_ROOT = Path("/var/lib/clamav")
STORE_PARENT = Path("/var/lib/blueprint")
COMMAND_ENV = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C",
               "DEBIAN_FRONTEND": "noninteractive"}


def load_configuration(
    unit_text: str, *, read_file: Callable[[Path], str] | None = None,
) -> dict[str, Any] | None:
    """Resolve only the two nonsecret settings in systemd EnvironmentFile order."""
    names = {"PIPELINE_CAPTURE_INTAKE_STORE_ROOT", "PIPELINE_CAPTURE_MALWARE_SCANNER_ARGV_JSON"}
    values: dict[str, str] = {}
    files: list[str] = []
    for raw in unit_text.splitlines():
        line = raw.strip()
        if line.startswith("Environment="):
            for assignment in shlex.split(line.partition("=")[2]):
                key, separator, value = assignment.partition("=")
                if separator and key in names:
                    values[key] = value
        elif line.startswith("EnvironmentFile="):
            files.extend(shlex.split(line.partition("=")[2]))
    if not names.intersection(values):
        return None  # Older units without this feature remain valid rollback targets.
    for item in files:
        path = Path(item.removeprefix("-"))
        if not path.is_absolute():
            raise CaptureUploadRuntimeError("capture_transfer_environment_invalid")
        try:
            if read_file is not None:
                text = read_file(path)
            else:
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
                with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
                    info = os.fstat(stream.fileno())
                    if not stat.S_ISREG(info.st_mode) or info.st_size > 1024 * 1024:
                        raise CaptureUploadRuntimeError("capture_transfer_environment_invalid")
                    text = stream.read(1024 * 1024 + 1)
        except FileNotFoundError:
            if item.startswith("-"):
                continue
            raise CaptureUploadRuntimeError("capture_transfer_environment_missing") from None
        for raw in text.splitlines():
            key, separator, value = raw.strip().partition("=")
            if not separator or key.strip() not in names:
                continue
            value = value.strip()
            if value.endswith("\\"):
                raise CaptureUploadRuntimeError("capture_transfer_environment_invalid")
            if value.startswith(("'", '"')):
                parsed = shlex.split(value)
                if len(parsed) != 1:
                    raise CaptureUploadRuntimeError("capture_transfer_environment_invalid")
                value = parsed[0]
            values[key.strip()] = value
    try:
        return {"store_root": values["PIPELINE_CAPTURE_INTAKE_STORE_ROOT"],
                "scanner_argv": json.loads(values["PIPELINE_CAPTURE_MALWARE_SCANNER_ARGV_JSON"])}
    except (KeyError, json.JSONDecodeError) as exc:
        raise CaptureUploadRuntimeError("capture_transfer_environment_invalid") from exc


def ensure_store(
    store_root: str, *, owner_ids: tuple[int, int], root_prefix: Path | None = None,
) -> dict[str, Any]:
    """Create only missing directories; never repair existing ownership or aliases."""
    logical = Path(store_root)
    if (not logical.is_absolute() or logical == STORE_PARENT or ".." in logical.parts
            or STORE_PARENT not in logical.parents):
        raise CaptureUploadRuntimeError("capture_transfer_store_root_not_provisionable")
    base = Path("/") if root_prefix is None else root_prefix
    current = base / STORE_PARENT.relative_to("/")
    descriptor = None
    created = []
    try:
        descriptor = _open_directory(current)
        for name in logical.relative_to(STORE_PARENT).parts:
            current /= name
            new = _mkdir_as_owner(descriptor, name, mode=0o750, owner_ids=owner_ids)
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
            info = os.fstat(descriptor)
            if new:
                if info.st_uid != owner_ids[0] or stat.S_IMODE(info.st_mode) != 0o750:
                    raise CaptureUploadRuntimeError("capture_transfer_store_permissions_unsafe")
                created.append(str(current))
            if info.st_uid not in {0, owner_ids[0]} or stat.S_IMODE(info.st_mode) & 0o002:
                raise CaptureUploadRuntimeError("capture_transfer_store_permissions_unsafe")
    except (OSError, subprocess.SubprocessError) as exc:
        raise CaptureUploadRuntimeError("capture_transfer_store_provision_failed") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return {"store_root": str(logical), "created": created,
            "existing_permissions_changed": False}


def _open_directory(path: Path) -> int:
    """Hold every path component without following service-controlled symlinks."""
    if not path.is_absolute() or ".." in path.parts:
        raise CaptureUploadRuntimeError("capture_transfer_provision_evidence_root_invalid")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open("/", flags)
    try:
        for name in path.parts[1:]:
            child = os.open(name, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _mkdir_as_owner(parent_fd: int, name: str, *, mode: int, owner_ids: tuple[int, int]) -> bool:
    """Create under protected root parents, or use the service identity elsewhere."""
    if (os.geteuid(), os.getegid()) == owner_ids:
        try:
            os.mkdir(name, mode=mode, dir_fd=parent_fd)
        except FileExistsError:
            return False
        return True
    if os.geteuid() != 0:
        raise CaptureUploadRuntimeError("capture_transfer_store_owner_unavailable")
    parent = os.fstat(parent_fd)
    if parent.st_uid == 0 and not stat.S_IMODE(parent.st_mode) & 0o022:
        return _mkdir_in_protected_parent(parent_fd, name, mode=mode, owner_ids=owner_ids)
    code = ("import os,sys\nos.umask(0)\ntry:\n"
            " os.mkdir(sys.argv[2], int(sys.argv[3], 8), dir_fd=int(sys.argv[1]))\n"
            "except FileExistsError:\n print('exists')\nelse:\n print('created')\n")
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, str(parent_fd), name, oct(mode)],
        pass_fds=(parent_fd,), user=owner_ids[0], group=owner_ids[1], extra_groups=[],
        env=COMMAND_ENV, stdin=subprocess.DEVNULL, capture_output=True, check=False, timeout=15,
    )
    if result.returncode or result.stdout not in {b"created\n", b"exists\n"}:
        raise CaptureUploadRuntimeError("capture_transfer_store_owner_creation_failed")
    return result.stdout == b"created\n"


def _mkdir_in_protected_parent(parent_fd, name, *, mode, owner_ids):
    """Only root can replace children here; configure only a newly created inode."""
    def identity(info):
        return info.st_dev, info.st_ino, info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)

    parent = os.fstat(parent_fd)
    if parent.st_uid != 0 or stat.S_IMODE(parent.st_mode) & 0o022:
        raise CaptureUploadRuntimeError("capture_transfer_store_parent_unsafe")
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
    except FileExistsError:
        return False  # Existing entries are never reowned or repaired.
    child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=parent_fd)
    try:
        created = os.fstat(child)
        if (created.st_uid != 0 or stat.S_IMODE(created.st_mode) != 0o700
                or identity(os.fstat(parent_fd)) != identity(parent)
                or identity(os.stat(name, dir_fd=parent_fd, follow_symlinks=False)) != identity(created)):
            raise CaptureUploadRuntimeError("capture_transfer_store_parent_unsafe")
        os.fchown(child, *owner_ids)
        os.fchmod(child, mode)
        current = os.fstat(child)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if ((current.st_dev, current.st_ino) != (created.st_dev, created.st_ino)
                or identity(named) != identity(current)
                or identity(os.fstat(parent_fd)) != identity(parent)
                or (current.st_uid, current.st_gid, stat.S_IMODE(current.st_mode)) != (*owner_ids, mode)):
            raise CaptureUploadRuntimeError("capture_transfer_store_parent_unsafe")
        os.fsync(child)
        os.fsync(parent_fd)
    finally:
        os.close(child)
    return True


def _record_process(root: Path, record: dict, owner_ids: tuple[int, int]) -> dict:
    payload = (json.dumps(record, sort_keys=True) + "\n").encode()
    name = uuid.uuid4().hex + ".json"
    root_fd = directory_fd = None
    try:
        root_fd = _open_directory(root)
        _mkdir_as_owner(root_fd, "runtime-provisioning", mode=0o700, owner_ids=owner_ids)
        directory_fd = os.open("runtime-provisioning", os.O_RDONLY | os.O_DIRECTORY
                               | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=root_fd)
        info = os.fstat(directory_fd)
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != owner_ids[0]
                or stat.S_IMODE(info.st_mode) & 0o077):
            raise CaptureUploadRuntimeError("capture_transfer_provision_evidence_root_invalid")
        fd = os.open(name, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                     0o600, dir_fd=directory_fd)
        with os.fdopen(fd, "wb") as stream:
            os.fchown(stream.fileno(), *owner_ids)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        current = os.stat("runtime-provisioning", dir_fd=root_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
            raise CaptureUploadRuntimeError("capture_transfer_provision_evidence_root_invalid")
    except (OSError, subprocess.SubprocessError) as exc:
        raise CaptureUploadRuntimeError("capture_transfer_provision_evidence_root_invalid") from exc
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
        if root_fd is not None:
            os.close(root_fd)
    return {"phase": record["phase"], "exit_code": record["exit_code"],
            "evidence_relative_path": "runtime-provisioning/" + name,
            "evidence_sha256": hashlib.sha256(payload).hexdigest()}


def _command(root, owner_ids, phase, argv, *, timeout, check=True,
             input_bytes=None, environment=None, cwd=None):
    try:
        result = subprocess.run(argv, shell=False, capture_output=True, check=False,
                                timeout=timeout, env=environment or COMMAND_ENV,
                                input=input_bytes, cwd=cwd)
        record = {"phase": phase, "argv": argv, "exit_code": result.returncode,
                  "stdout_base64": base64.b64encode(result.stdout).decode(),
                  "stderr_base64": base64.b64encode(result.stderr).decode()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        record = {"phase": phase, "argv": argv, "exit_code": None,
                  "error_class": type(exc).__name__,
                  "stdout_base64": base64.b64encode(getattr(exc, "stdout", None) or b"").decode(),
                  "stderr_base64": base64.b64encode(getattr(exc, "stderr", None) or b"").decode()}
        diagnostic = _record_process(root, record, owner_ids)
        raise CaptureUploadRuntimeError("capture_transfer_prerequisite_failed:" + phase,
                                        diagnostic) from exc
    diagnostic = _record_process(root, record, owner_ids)
    if check and result.returncode:
        raise CaptureUploadRuntimeError("capture_transfer_prerequisite_failed:" + phase, diagnostic)
    return result, diagnostic


def _definitions_ready() -> bool:
    return all(any((DEFINITION_ROOT / name).is_file()
                   and not (DEFINITION_ROOT / name).is_symlink()
                   and (DEFINITION_ROOT / name).stat().st_size > 0 for name in alternatives)
               for alternatives in (("main.cvd", "main.cld"), ("daily.cvd", "daily.cld")))


def provision_runtime(
    configuration: dict, *, owner_ids: tuple[int, int], install_system_scanner: bool = False,
    root_prefix: Path | None = None, definition_wait_seconds: float = 300,
) -> dict:
    """Prepare the effective store and declared ClamAV, preserving custom scanners."""
    store = ensure_store(configuration["store_root"], owner_ids=owner_ids, root_prefix=root_prefix)
    root = Path(configuration["store_root"])
    if root_prefix is not None:
        root = root_prefix / root.relative_to("/")
    argv = configuration["scanner_argv"]
    if (not isinstance(argv, list) or not argv
            or not all(isinstance(value, str) and value and "\x00" not in value for value in argv)
            or not Path(argv[0]).is_absolute()):
        raise CaptureUploadRuntimeError("capture_transfer_scanner_configuration_invalid")
    records = []
    if Path(argv[0]) == SYSTEM_SCANNER:
        if not SYSTEM_SCANNER.is_file() or not SYSTEM_UPDATER.is_file():
            if not install_system_scanner or os.geteuid() != 0:
                raise CaptureUploadRuntimeError("capture_transfer_scanner_executable_missing")
            for phase, command in (
                ("apt_update", ["/usr/bin/apt-get", "-qq", "-o", "DPkg::Lock::Timeout=120", "update"]),
                ("apt_install", ["/usr/bin/apt-get", "-qq", "-o", "DPkg::Lock::Timeout=120", "install",
                                 "--yes", "--no-install-recommends", "clamav", "clamav-freshclam"]),
            ):
                _, evidence = _command(root, owner_ids, phase, command, timeout=600)
                records.append(evidence)
            if not SYSTEM_SCANNER.is_file() or not SYSTEM_UPDATER.is_file():
                raise CaptureUploadRuntimeError("capture_transfer_scanner_installation_incomplete")
        if not _definitions_ready():
            if not install_system_scanner or os.geteuid() != 0:
                raise CaptureUploadRuntimeError("capture_transfer_scanner_definitions_missing")
            active, evidence = _command(root, owner_ids, "freshclam_service",
                ["/usr/bin/systemctl", "is-active", "clamav-freshclam.service"], timeout=15, check=False)
            records.append(evidence)
            if active.returncode == 0 and active.stdout.strip() == b"active":
                deadline = time.monotonic() + definition_wait_seconds
                while not _definitions_ready() and time.monotonic() < deadline:
                    time.sleep(2)
                if not _definitions_ready():
                    _, evidence = _command(root, owner_ids, "freshclam_status",
                        ["/usr/bin/journalctl", "-u", "clamav-freshclam.service", "--no-pager", "-n", "80"],
                        timeout=15, check=False)
                    raise CaptureUploadRuntimeError("capture_transfer_scanner_definitions_not_ready", evidence)
            else:
                _, evidence = _command(root, owner_ids, "freshclam_update",
                                       [str(SYSTEM_UPDATER), "--stdout"], timeout=600)
                records.append(evidence)
            if not _definitions_ready():
                raise CaptureUploadRuntimeError("capture_transfer_scanner_definitions_not_ready")
    elif not Path(argv[0]).is_file() or not os.access(argv[0], os.X_OK):
        raise CaptureUploadRuntimeError("capture_transfer_scanner_executable_missing")
    return {"status": "provisioned_controls_required", "store": store,
            "scanner_program": Path(argv[0]).name, "prerequisite_processes": records,
            "provider_mutation_performed": False, "source_footage_processed": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-file", required=True, type=Path)
    parser.add_argument("--install-system-scanner", action="store_true")
    parser.add_argument("--account", default="blueprint")
    args = parser.parse_args()
    try:
        account = pwd.getpwnam(args.account)
        configuration = load_configuration(args.unit_file.read_text())
        if configuration is None:
            print(json.dumps({"status": "not_required"}))
            return 0
        result = provision_runtime(configuration, owner_ids=(account.pw_uid, account.pw_gid),
                                   install_system_scanner=args.install_system_scanner)
        source = Path(__file__).resolve().parents[1]
        check, probe_evidence = _command(
            Path(configuration["store_root"]), (account.pw_uid, account.pw_gid), "scanner_controls",
            ["/usr/sbin/runuser", "--user", args.account, "--", sys.executable,
             "-m", "blueprint_pipeline.capture_upload_runtime"],
            input_bytes=json.dumps(configuration).encode(), check=False, timeout=150,
            cwd=source.parent, environment={"PATH": "/usr/bin:/bin", "LC_ALL": "C",
                                    "PYTHONPATH": str(source), "PYTHONDONTWRITEBYTECODE": "1"})
        try:
            control = json.loads(check.stdout)
        except (ValueError, UnicodeError) as exc:
            raise CaptureUploadRuntimeError("capture_transfer_runtime_probe_invalid", probe_evidence) from exc
        if (check.returncode or not isinstance(control, dict) or control.get("status") != "ready"
                or control.get("provider_mutation_performed") is not False
                or control.get("source_footage_processed") is not False
                or control.get("robot_episode_executed") is not False
                or not isinstance(control.get("controls"), dict)
                or any(not isinstance(control["controls"].get(name), dict)
                       or type(control["controls"][name].get("exit_code")) is not int
                       or control["controls"][name]["exit_code"] != expected
                       for name, expected in (("clean", 0), ("known_test_signature", 1)))):
            raise CaptureUploadRuntimeError("capture_transfer_scanner_controls_failed", probe_evidence)
        print(json.dumps({"status": "ready", "prerequisites": result, "controls": control,
                          "probe_evidence": probe_evidence}))
        return 0
    except (CaptureUploadRuntimeError, OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        reason = str(exc) if isinstance(exc, CaptureUploadRuntimeError) else "capture_transfer_prerequisite_io_failed"
        print(json.dumps({"status": "blocked", "reason": reason,
                          **({"diagnostic": exc.diagnostic} if isinstance(exc, CaptureUploadRuntimeError)
                             and exc.diagnostic else {})}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
