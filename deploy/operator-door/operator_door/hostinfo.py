"""Host facts the door reports: systemd units, journals, locks, disk, load.

Every command goes through an injectable runner with a fixed argv (never a
shell), and every unit name is validated against the ``blueprint-`` namespace
before it reaches ``systemctl`` or ``journalctl``. Unit ``Environment`` is never
requested, and journal lines pass the secret guard.
"""

from __future__ import annotations

import json
import os
import re
import subprocess  # nosec B404 - fixed argv over validated unit names
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from .config import DoorConfig
from .secrets_guard import redact_lines

UNIT_NAME = re.compile(r"^blueprint-[A-Za-z0-9@_.:-]{1,200}\.(service|timer|path)$")
UNIT_PATTERN = re.compile(r"^blueprint-[A-Za-z0-9@_.:*-]{0,200}$")
SINCE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z :+.-]{0,39}$|^-[0-9]{1,6}[smhdw]$")
SHOW_PROPERTIES = (
    "Id", "Description", "ActiveState", "SubState", "Result", "ExecMainStatus",
    "ExecMainStartTimestamp", "ActiveEnterTimestamp", "InactiveEnterTimestamp",
    "UnitFileState", "NextElapseUSecRealtime", "LastTriggerUSec",
)


class HostRefused(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class CommandRunner(Protocol):
    def run(self, argv: Sequence[str], timeout: float) -> CommandResult: ...


class SubprocessRunner:
    def run(self, argv: Sequence[str], timeout: float) -> CommandResult:
        try:
            done = subprocess.run(  # nosec B603 - fixed argv, no shell
                list(argv), capture_output=True, text=True, timeout=timeout, check=False,
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return CommandResult(124, "", "timeout")
        except OSError as error:
            return CommandResult(127, "", type(error).__name__)
        return CommandResult(done.returncode, done.stdout, done.stderr)


def _fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    handler = urllib.request.ProxyHandler({})  # loopback only; never through a proxy
    opener = urllib.request.build_opener(handler)
    with opener.open(request, timeout=5) as response:  # nosec B310 - fixed loopback URL
        return json.loads(response.read(1024 * 1024).decode("utf-8"))


def validate_unit(unit: str) -> str:
    if not isinstance(unit, str) or not UNIT_NAME.match(unit):
        raise HostRefused("unit_name_invalid")
    return unit


class HostInfo:
    def __init__(
        self,
        config: DoorConfig,
        *,
        runner: CommandRunner | None = None,
        proc_locks_path: str = "/proc/locks",
        fetch_json: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self.runner = runner or SubprocessRunner()
        self._proc_locks = proc_locks_path
        self._fetch_json = fetch_json or _fetch_json

    # -- systemd ------------------------------------------------------------

    def unit_properties(self, units: Sequence[str]) -> list[dict[str, str]]:
        names = [validate_unit(unit) for unit in units]
        if not names:
            return []
        argv = ["systemctl", "show", "--no-pager", "-p", ",".join(SHOW_PROPERTIES), "--", *names]
        result = self.runner.run(argv, timeout=15)
        blocks: list[dict[str, str]] = []
        current: dict[str, str] = {}
        for line in result.stdout.splitlines():
            if not line.strip():
                if current:
                    blocks.append(current)
                    current = {}
                continue
            key, _, value = line.partition("=")
            if key in SHOW_PROPERTIES:
                current[key] = value
        if current:
            blocks.append(current)
        return blocks

    def list_units(self, pattern: str = "blueprint-*", states: Sequence[str] = ()) -> list[dict[str, str]]:
        if not UNIT_PATTERN.match(pattern):
            raise HostRefused("unit_pattern_invalid")
        argv = ["systemctl", "list-units", "--all", "--plain", "--no-legend", "--no-pager"]
        if states:
            if not all(re.fullmatch(r"[a-z-]{2,20}", state) for state in states):
                raise HostRefused("unit_state_invalid")
            argv.append("--state=" + ",".join(states))
        argv += ["--", pattern]
        result = self.runner.run(argv, timeout=15)
        rows: list[dict[str, str]] = []
        for line in result.stdout.splitlines():
            parts = line.split(None, 4)
            if len(parts) >= 4 and parts[0].startswith("blueprint-"):
                rows.append({"unit": parts[0], "load": parts[1], "active": parts[2], "sub": parts[3],
                             "description": parts[4] if len(parts) > 4 else ""})
        return rows

    def journal(self, unit: str, *, lines: int, since: str | None = None) -> str:
        validate_unit(unit)
        count = max(1, min(int(lines), self.config.max_journal_lines))
        argv = ["journalctl", "-u", unit, "-n", str(count), "--no-pager", "-o", "short-iso"]
        if since is not None:
            if not SINCE.match(since):
                raise HostRefused("journal_since_invalid")
            argv.append(f"--since={since}")
        result = self.runner.run(argv, timeout=30)
        return redact_lines(result.stdout)

    # -- locks, disk, load --------------------------------------------------

    def _lock_table(self) -> dict[tuple[str, int], list[int]]:
        table: dict[tuple[str, int], list[int]] = {}
        try:
            text = Path(self._proc_locks).read_text(encoding="utf-8")
        except OSError:
            return table
        for line in text.splitlines():
            fields = line.split()
            if len(fields) < 6 or fields[1] == "->":
                continue  # blocked waiters are not holders
            try:
                pid = int(fields[4])
                major, minor, inode = fields[5].split(":")
            except ValueError:
                continue
            table.setdefault((f"{int(major, 16):02x}:{int(minor, 16):02x}", int(inode)), []).append(pid)
        return table

    def paid_launch_locks(self) -> list[dict[str, Any]]:
        directory = Path(self.config.control_plane_state) / "provider-locks"
        table = self._lock_table()
        locks: list[dict[str, Any]] = []
        for path in sorted(directory.glob("*.lock")):
            try:
                info = path.stat()
            except OSError:
                continue
            device = f"{os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}"
            holders = sorted(table.get((device, info.st_ino), []))
            locks.append({"path": str(path), "held": bool(holders), "holders": holders})
        return locks

    def disk(self, paths: Sequence[str]) -> dict[str, dict[str, int]]:
        usage: dict[str, dict[str, int]] = {}
        for path in paths:
            try:
                stats = os.statvfs(path)
            except OSError:
                continue
            usage[path] = {"total_bytes": stats.f_blocks * stats.f_frsize,
                           "free_bytes": stats.f_bavail * stats.f_frsize}
        return usage

    def load(self) -> dict[str, Any]:
        try:
            one, five, fifteen = os.getloadavg()
        except OSError:
            one = five = fifteen = -1.0
        return {"loadavg": [round(one, 2), round(five, 2), round(fifteen, 2)], "cpus": os.cpu_count()}

    def version(self) -> dict[str, Any]:
        return self._fetch_json(self.config.intake_version_url)
