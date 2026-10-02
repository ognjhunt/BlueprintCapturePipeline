"""Compare-and-repair one legacy notifier override; never execute its hooks."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import stat
import tempfile
from pathlib import Path

from . import holds
from .hostinfo import CommandRunner, SubprocessRunner

UNIT = "blueprint-pipeline-control-plane.service"
DROPIN = Path("/etc/systemd/system") / (UNIT + ".d") / "95-blueprint-active-release.conf"
ACTIVE = Path("/opt/blueprint/task-evaluation-control-plane")
RELEASES = Path("/opt/blueprint/task-evaluation-control-plane-releases")
ROOT_UID = ROOT_GID = 0
HOLDS = Path("/var/lib/blueprint-operator-door/requests/holds")
SCRIPT = "deploy/systemd/blueprint-control-plane-postchecks.sh"
NEW_INVOCATION = ("exec env BLUEPRINT_PIPELINE_REPO=/opt/blueprint/task-evaluation-control-plane "
                  "BLUEPRINT_LIVE_CONTROL_PLANE_REPO=/opt/blueprint/task-evaluation-control-plane "
                  "BLUEPRINT_PIPELINE_PYTHON=/opt/blueprint/BlueprintCapturePipeline/.venv/bin/python "
                  "/bin/bash /opt/blueprint/task-evaluation-control-plane/" + SCRIPT)


class RepairRefused(Exception):
    pass


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _read_original() -> tuple[bytes, int]:
    if any(path.is_symlink() for path in (DROPIN, *DROPIN.parents)):
        raise RepairRefused("dropin_symlink_refused")
    directory = DROPIN.parent.stat()
    if directory.st_uid != ROOT_UID or directory.st_gid != ROOT_GID or directory.st_mode & 0o022:
        raise RepairRefused("dropin_directory_metadata_refused")
    fd = os.open(DROPIN, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 65536
                or info.st_uid != ROOT_UID or info.st_gid != ROOT_GID or info.st_mode & 0o022):
            raise RepairRefused("dropin_metadata_refused")
        return stream.read(65537), stat.S_IMODE(info.st_mode)


def _replacement(original: bytes) -> bytes:
    lines = original.decode("utf-8").splitlines(keepends=True)
    section, changed = "", 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("["):
            section = stripped
        if section != "[Service]" or not line.startswith("ExecStartPost=") or SCRIPT not in line:
            continue
        if line.rstrip().endswith("\\"):
            raise RepairRefused("postcheck_shape_unknown")
        value = line.split("=", 1)[1].rstrip("\r\n")
        argv = shlex.split(value)
        if len(argv) != 3 or argv[:2] != ["/bin/bash", "-lc"]:
            raise RepairRefused("postcheck_shape_unknown")
        match = re.search(r"(?:exec )?/bin/bash (?:/opt/blueprint/task-evaluation-control-plane/)?"
                          + re.escape(SCRIPT) + r"$", argv[2])
        if match is None:
            raise RepairRefused("postcheck_shape_unknown")
        # Keep preceding shell operations and every other directive/hook byte.
        shell = argv[2][:match.start()] + NEW_INVOCATION
        ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
        lines[index] = "ExecStartPost=/bin/bash -lc " + shlex.quote(shell) + ending
        changed += 1
    if changed != 1:
        raise RepairRefused("postcheck_shape_unknown")
    return "".join(lines).encode("utf-8")


def _command_value(runner: CommandRunner) -> str:
    result = runner.run(["systemctl", "show", "--no-pager", "-p", "ExecStartPost", "--", UNIT], timeout=5)
    if result.returncode != 0:
        raise RepairRefused("binding_metadata_unavailable")
    value = next((line.partition("=")[2] for line in result.stdout.splitlines()
                  if line.startswith("ExecStartPost=")), "")
    if not value or len(value) > 65536:
        raise RepairRefused("binding_metadata_unavailable")
    return value


def _command_digest(runner: CommandRunner) -> str:
    return _digest(_command_value(runner).encode())


def _atomic_write(path: Path, payload: bytes, mode: int) -> None:
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".blueprint-postcheck-write.")
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fchmod(stream.fileno(), mode)
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def repair_binding(expected_sha256: str, expected_commit: str, *, runner: CommandRunner | None = None) -> dict:
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", expected_sha256) or not re.fullmatch(r"[a-f0-9]{40}", expected_commit):
        raise RepairRefused("expected_identity_invalid")
    runner = runner or SubprocessRunner()
    release = ACTIVE.resolve(strict=True)
    if release.parent != RELEASES or release.name != expected_commit:
        raise RepairRefused("active_source_drift")
    script_bytes = (release / SCRIPT).read_bytes()
    if b"PYTHONPATH=${PWD}/src" not in script_bytes or b"BLUEPRINT_PIPELINE_PYTHON" not in script_bytes:
        raise RepairRefused("release_postcheck_contract_unknown")
    stop = holds.read(HOLDS, "blueprint-agent-run-dispatcher.timer")
    if stop is None or stop.get("status") != "active" or stop.get("require_explicit_release") is not True:
        raise RepairRefused("explicit_dispatcher_stop_required")
    for command, expected in (("is-enabled", "disabled"), ("is-active", "inactive")):
        result = runner.run(["systemctl", command, "--", "blueprint-agent-run-dispatcher.timer"], timeout=5)
        if result.stdout.strip() != expected:
            raise RepairRefused("dispatcher_stop_state_unproven")
    original, mode = _read_original()
    changed = _replacement(original)
    if _command_digest(runner) != expected_sha256:
        raise RepairRefused("postcheck_command_drift")
    lock = os.open(DROPIN.parent / ".blueprint-postcheck-repair.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(lock, "r+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        current, _ = _read_original()
        if current != original or _command_digest(runner) != expected_sha256 or ACTIVE.resolve() != release:
            raise RepairRefused("postcheck_command_drift")
        backup = Path(tempfile.mkdtemp(dir=DROPIN.parent, prefix=".blueprint-postcheck-backup."))
        _atomic_write(backup / "original.conf", original, 0o600)
        # Allocate both candidates before the critical swap. Rollback only
        # renames existing bytes and keeps the failed replacement for review.
        _atomic_write(backup / "rollback.conf", original, mode)
        _atomic_write(backup / "replacement.conf", changed, mode)
        swapped = False
        try:
            os.replace(backup / "replacement.conf", DROPIN)
            swapped = True
            reload = runner.run(["systemctl", "daemon-reload"], timeout=5)
            if reload.returncode != 0:
                raise RepairRefused("reload_failed")
            effective = _command_value(runner)
            if ("BLUEPRINT_PIPELINE_REPO=/opt/blueprint/task-evaluation-control-plane" not in effective
                    or "/opt/blueprint/task-evaluation-control-plane/" + SCRIPT not in effective
                    or ACTIVE.resolve() != release):
                raise RepairRefused("effective_binding_not_verified")
        except BaseException as error:
            if swapped:
                os.replace(DROPIN, backup / "failed-replacement.conf")
                os.replace(backup / "rollback.conf", DROPIN)
                runner.run(["systemctl", "daemon-reload"], timeout=5)
                if isinstance(error, RepairRefused):
                    raise RepairRefused(str(error) + "_original_restored") from error
            raise
    return {"schema": "blueprint.notifier_binding_repair.v1", "status": "repaired", "unit": UNIT,
            "source_commit": expected_commit, "drop_in_provenance": "unknown_external_owner",
            "drop_in_path": str(DROPIN), "backup_path": str(backup / "original.conf"),
            "original_sha256": _digest(original), "replacement_sha256": _digest(changed),
            "observed_postcheck_sha256_before": expected_sha256, "release_postchecks_sha256": _digest(script_bytes),
            "restart_performed": False, "slack_send_performed": False, "dispatcher_stop_verified": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--receipt-out", required=True)
    args = parser.parse_args()
    try:
        receipt = repair_binding(args.expected_sha256, args.expected_commit)
        _atomic_write(Path(args.receipt_out), (json.dumps(receipt, sort_keys=True) + "\n").encode(), 0o644)
    except (RepairRefused, OSError, ValueError) as error:
        print("refused: " + (str(error) if isinstance(error, RepairRefused) else type(error).__name__))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
