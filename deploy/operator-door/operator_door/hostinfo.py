"""Host facts the door reports: systemd units, journals, locks, disk, load, memory.

Every command goes through an injectable runner with a fixed argv (never a
shell), and every unit name is validated against the ``blueprint-`` namespace
before it reaches ``systemctl`` or ``journalctl``. Unit ``Environment`` is never
requested, and journal lines pass the secret guard.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import re
import stat
import subprocess  # nosec B404 - fixed argv over validated unit names
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from .config import DoorConfig
from .secrets_guard import redact_lines

# Always used with fullmatch: `$` would also accept a trailing newline.
UNIT_NAME = re.compile(r"blueprint-[A-Za-z0-9@_.:-]{1,200}\.(?:service|timer|path)")
UNIT_PATTERN = re.compile(r"blueprint-[A-Za-z0-9@_.:*-]{0,200}")
SINCE = re.compile(r"[0-9A-Za-z][0-9A-Za-z :+.-]{0,39}|-[0-9]{1,6}[smhdw]")
SHOW_PROPERTIES = (
    "Id", "Description", "ActiveState", "SubState", "Result", "ExecMainStatus",
    "ExecMainStartTimestamp", "ActiveEnterTimestamp", "InactiveEnterTimestamp",
    "UnitFileState", "NextElapseUSecRealtime", "LastTriggerUSec",
    "InvocationID", "Type",
)
MEMINFO_KEYS = ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree", "Shmem")
MEMORY_TOP_UNITS = 20
MEMORY_SLICES = ("system.slice", "user.slice", "init.scope")
TMPFS_PATHS = ("/dev/shm", "/run", "/tmp")
# A systemd unit's cgroup directory name; anything else under system.slice is skipped.
CGROUP_UNIT = re.compile(r"[A-Za-z0-9@_.:\\-]{1,255}\.(?:service|scope|slice|socket|mount|swap)")
_CGROUP_BYTES = re.compile(rb"[0-9]{1,20}\n?")
_MEMORY_SCAN_ENTRIES = 4096  # system.slice entries looked at, interface files included
_MEMORY_VALUE_READ = 64
_MEMINFO_READ = 64 * 1024
_MOUNTS_READ = 1024 * 1024
OOM_REPORTS = 4
OOM_TOP_TASKS = 15
OOM_SINCE = "-24h"
_OOM_LOG_LINES = 20000  # journalctl -n: the newest kernel lines in the window
_OOM_LOG_BYTES = 8 * 1024 * 1024  # the tail kept when the output is larger
_OOM_LOG_TIMEOUT = 10
_OOM_PID_INDEX = 8192  # task rows per event remembered for the killed task's oom_score_adj
_OOM_UIDS = 10
_OOM_UID_INDEX = 1024
# A short-iso journal line from the kernel: timestamp, hostname (never kept), message.
_OOM_JOURNAL_LINE = re.compile(
    r"(?P<at>[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:[.,][0-9]{1,9})?"
    r"(?:Z|[+-][0-9]{2}:?[0-9]{2})) \S{1,255} kernel: (?P<message>.{0,4096})"
)
_OOM_INVOKED = re.compile(r"(?P<name>.{1,64}?) invoked oom-killer: ")
_OOM_TABLE_HEADER = re.compile(r"\[ {0,8}pid {0,8}\] {1,16}(?P<columns>[a-z_ ]{1,256})")
_OOM_TABLE_ROW = re.compile(r"\[ {0,16}(?P<pid>[0-9]{1,10})\] {1,16}(?P<rest>.{1,512})")
_OOM_CONSTRAINT = re.compile(r"oom-kill:constraint=(?P<constraint>[A-Z_]{1,40})(?P<fields>,.{0,2048})?")
_OOM_TASK_IN = re.compile(r"Task in (?P<task>/\S{0,1024}) killed as a result of limit of (?P<limit>/\S{0,1024})")
_OOM_KILLED = re.compile(
    r"Killed process (?P<pid>[0-9]{1,10}) \((?P<name>.{1,64})\),? (?P<fields>total-vm:.{0,512})"
)
_OOM_KILLED_FIELD = re.compile(
    r"(?P<key>total-vm|anon-rss|file-rss|shmem-rss|UID|oom_score_adj):(?P<value>-?[0-9]{1,20})"
)
_OOM_INT = re.compile(r"-?[0-9]{1,20}")
_OOM_NAME = re.compile(r"[\x21-\x7e](?:[\x20-\x7e]{0,30}[\x21-\x7e])?")
_OOM_TABLE_NEEDS = frozenset({"uid", "total_vm", "rss", "oom_score_adj"})


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


def _read_small(path: str, limit: int) -> bytes | None:
    """At most ``limit`` bytes of a regular or pseudo file; None when missing, unreadable or larger."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        chunks: list[bytes] = []
        size = 0
        while size <= limit:
            chunk = os.read(fd, limit + 1 - size)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
    except OSError:
        return None
    finally:
        os.close(fd)
    return b"".join(chunks) if size <= limit else None


def parse_meminfo(text: str) -> dict[str, int | None]:
    """The fixed ``/proc/meminfo`` fields in bytes; a field absent or malformed is None."""
    values: dict[str, int | None] = dict.fromkeys(MEMINFO_KEYS)
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        fields = rest.split()
        if key not in values or not fields or not re.fullmatch(r"[0-9]{1,20}", fields[0]):
            continue
        if fields[1:] == ["kB"]:  # the kernel's kB is KiB
            values[key] = int(fields[0]) * 1024
        elif len(fields) == 1:
            values[key] = int(fields[0])
    return values


def parse_cgroup_bytes(data: bytes | None) -> int | None:
    """A cgroup v2 ``memory.current``/``memory.peak`` value; None when absent or not a byte count."""
    if data is None or not _CGROUP_BYTES.fullmatch(data):
        return None
    return int(data)


def _oom_name(raw: str) -> str | None:
    """A kernel task name (comm, at most 15 bytes) as printed; None when odd or credential-shaped."""
    name = raw.strip()
    if not _OOM_NAME.fullmatch(name) or redact_lines(name) != name:
        return None
    return name


def _oom_cgroup(raw: str | None) -> str | None:
    """A cgroup path only when every segment is a systemd unit name; anything else is None."""
    if raw is None or raw == "/":
        return raw
    segments = raw.split("/")
    if (len(raw) > 1024 or segments[0] != "" or len(segments) > 16
            or not all(CGROUP_UNIT.fullmatch(segment) for segment in segments[1:])
            or redact_lines(raw) != raw):
        return None
    return raw


def _oom_at(raw: str) -> str | None:
    """A short-iso journal timestamp as UTC ``...Z`` (fractions dropped); None when unparseable."""
    try:
        moment = datetime.strptime(raw[:19] + raw[19:].lstrip(",.0123456789"), "%Y-%m-%dT%H:%M:%S%z")
    except ValueError:
        return None
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class _OomEvent:
    """One report, from "invoked oom-killer" to the "Killed process" line."""

    def __init__(self, at: str | None, invoker: str | None) -> None:
        self.at = at
        self.invoker = invoker
        self.constraint: dict[str, Any] | None = None
        self.killed: dict[str, Any] | None = None
        self.columns: list[str] | None = None
        self.rows = 0
        self.unparsed_rows = 0
        self.total_rss_pages = 0
        self.top: list[tuple[int, int, dict[str, Any]]] = []
        self.by_pid: dict[int, tuple[int, int]] = {}
        self.by_uid: dict[int, list[int]] = {}
        self.by_uid_truncated = False

    def header(self, columns: str) -> None:
        names = columns.split()
        if names and names[-1] == "name" and _OOM_TABLE_NEEDS <= set(names):
            self.columns = names

    def row(self, pid: int, rest: str) -> None:
        self.rows += 1
        parts = rest.split(None, len(self.columns) - 1) if self.columns else []
        if (not self.columns or len(parts) != len(self.columns)
                or not all(_OOM_INT.fullmatch(part) for part in parts[:-1])):
            self.unparsed_rows += 1
            return
        values = dict(zip(self.columns[:-1], (int(part) for part in parts[:-1])))
        rss, uid = values["rss"], values["uid"]
        task = {"pid": pid, "uid": uid, "name": _oom_name(parts[-1]), "rss_pages": rss,
                "total_vm_pages": values["total_vm"], "oom_score_adj": values["oom_score_adj"]}
        self.total_rss_pages += rss
        entry = (rss, -pid, task)
        if len(self.top) < OOM_TOP_TASKS:
            heapq.heappush(self.top, entry)
        elif entry[:2] > self.top[0][:2]:
            heapq.heapreplace(self.top, entry)
        if len(self.by_pid) < _OOM_PID_INDEX:
            self.by_pid[pid] = (values["oom_score_adj"], uid)
        if uid in self.by_uid or len(self.by_uid) < _OOM_UID_INDEX:
            totals = self.by_uid.setdefault(uid, [0, 0])
            totals[0] += 1
            totals[1] += rss
        else:
            self.by_uid_truncated = True

    def report(self, page_size: int | None) -> dict[str, Any]:
        parsed = self.rows > self.unparsed_rows
        killed = self.killed
        if killed is not None and killed["pid"] in self.by_pid:
            score, uid = self.by_pid[killed["pid"]]
            killed = {**killed, "oom_score_adj": killed["oom_score_adj"] if killed["oom_score_adj"] is not None
                      else score, "uid": killed["uid"] if killed["uid"] is not None else uid}
        top = [task for _, _, task in sorted(self.top, key=lambda entry: entry[:2], reverse=True)]
        for task in top:
            task["rss_bytes"] = task["rss_pages"] * page_size if page_size else None
        by_uid = sorted(({"uid": uid, "tasks": tasks, "rss_pages": rss}
                         for uid, (tasks, rss) in self.by_uid.items()),
                        key=lambda row: (-row["rss_pages"], row["uid"]))
        return {
            "at": self.at,
            "invoker": self.invoker,
            "constraint": self.constraint,
            "killed": killed,
            "tasks": {
                "rows": self.rows,
                "unparsed_rows": self.unparsed_rows,
                "parsed": parsed,
                "total_rss_pages": self.total_rss_pages if parsed else None,
                "top": top if parsed else None,
                "by_uid": by_uid[:_OOM_UIDS] if parsed else None,
                "by_uid_truncated": self.by_uid_truncated if parsed else None,
            },
        }


def _oom_killed(match: re.Match[str]) -> dict[str, Any]:
    fields = {key: int(value) for key, value in _OOM_KILLED_FIELD.findall(match["fields"])}
    return {"pid": int(match["pid"]), "name": _oom_name(match["name"]), "uid": fields.get("UID"),
            "total_vm_kb": fields.get("total-vm"), "anon_rss_kb": fields.get("anon-rss"),
            "file_rss_kb": fields.get("file-rss"), "shmem_rss_kb": fields.get("shmem-rss"),
            "oom_score_adj": fields.get("oom_score_adj")}


def _oom_constraint(match: re.Match[str]) -> dict[str, Any]:
    tokens = (match["fields"] or "").split(",")
    values = dict(token.partition("=")[::2] for token in tokens if "=" in token)
    return {"constraint": match["constraint"], "global_oom": "global_oom" in tokens,
            "oom_memcg": _oom_cgroup(values.get("oom_memcg")),
            "task_memcg": _oom_cgroup(values.get("task_memcg"))}


def parse_oom_reports(text: str, *, page_size: int | None, limit: int = OOM_REPORTS) -> dict[str, Any]:
    """The last ``limit`` OOM-killer reports in ``journalctl -k -o short-iso`` text.

    Only timestamps, numbers, uids, task names (comm) and unit-name cgroup paths
    leave; every other kernel line (call traces, hardware, hostnames) is ignored.
    A task table whose header is not recognised keeps only its row count.
    """
    events: list[dict[str, Any]] = []
    count = 0
    current: _OomEvent | None = None
    for line in text.splitlines():
        journal = _OOM_JOURNAL_LINE.fullmatch(line)
        if journal is None:
            continue  # continuation lines, "-- Boot --" markers and anything else
        message = journal["message"]
        invoked = _OOM_INVOKED.match(message)
        if invoked is not None:
            if current is not None:
                events.append(current.report(page_size))
            count += 1
            current = _OomEvent(_oom_at(journal["at"]), _oom_name(invoked["name"]))
            events = events[-limit:]
            continue
        if current is None:
            continue
        row = _OOM_TABLE_ROW.fullmatch(message)
        if row is not None:
            current.row(int(row["pid"]), row["rest"])
        elif (header := _OOM_TABLE_HEADER.fullmatch(message)) is not None:
            current.header(header["columns"])
        elif (constraint := _OOM_CONSTRAINT.fullmatch(message)) is not None:
            current.constraint = _oom_constraint(constraint)
        elif (task_in := _OOM_TASK_IN.search(message)) is not None and current.constraint is None:
            current.constraint = {"constraint": None, "global_oom": None,
                                  "oom_memcg": _oom_cgroup(task_in["limit"]),
                                  "task_memcg": _oom_cgroup(task_in["task"])}
        elif (killed := _OOM_KILLED.search(message)) is not None:
            current.killed = _oom_killed(killed)
            events.append(current.report(page_size))
            current = None
    if current is not None:
        events.append(current.report(page_size))
    return {"count": count, "events": events[-limit:]}


def validate_unit(unit: str) -> str:
    if not isinstance(unit, str) or not UNIT_NAME.fullmatch(unit):
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
        proc_meminfo_path: str = "/proc/meminfo",
        cgroup_root: str = "/sys/fs/cgroup",
        mounts_path: str = "/proc/self/mounts",
        tmpfs_paths: Sequence[str] = TMPFS_PATHS,
    ) -> None:
        self.config = config
        self.runner = runner or SubprocessRunner()
        self._proc_locks = proc_locks_path
        self._meminfo = proc_meminfo_path
        self._cgroup_root = cgroup_root
        self._mounts = mounts_path
        self._tmpfs_paths = tuple(tmpfs_paths)
        self._fetch_json = fetch_json or _fetch_json

    # -- systemd ------------------------------------------------------------

    def notifier_binding(self) -> dict[str, Any]:
        """A fixed-target binding projection; never return raw commands or env."""
        unit = "blueprint-pipeline-control-plane.service"
        result = self.runner.run(
            ["systemctl", "show", "--no-pager", "-p",
             "FragmentPath,DropInPaths,ExecStartPost", "--", unit], timeout=5,
        )
        identity = {"schema": "blueprint.systemd_postcheck_binding.v1", "unit": unit,
                    "claim_ceiling": "effective_command_metadata_only"}
        if result.returncode != 0 or len(result.stdout) > 65536:
            return {**identity, "status": "unavailable"}
        values = dict(line.partition("=")[::2] for line in result.stdout.splitlines()
                      if line.partition("=")[0] in {"FragmentPath", "DropInPaths", "ExecStartPost"})
        command = values.get("ExecStartPost", "")
        path_pattern = re.compile(
            r"/(?:etc|run|usr/lib|lib)/systemd/system/"
            r"blueprint-pipeline-control-plane\.service(?:\.d/[A-Za-z0-9_.:-]+\.conf)?"
        )
        fragment = values.get("FragmentPath", "")
        drops = values.get("DropInPaths", "").split()
        fragment_safe = bool(path_pattern.fullmatch(fragment)) and redact_lines(fragment) == fragment
        recognized_drops = [path for path in drops if path_pattern.fullmatch(path)
                            and redact_lines(path) == path]
        script_pattern = re.compile(
            r"/opt/blueprint/(?:BlueprintCapturePipeline|task-evaluation-control-plane"
            r"(?:-releases/[a-f0-9]{40})?)/deploy/systemd/blueprint-control-plane-postchecks\.sh"
        )
        return {
            **identity, "status": "observed" if command else "unknown",
            "fragment_path": fragment if fragment_safe else None,
            "drop_in_paths": recognized_drops,
            "metadata_paths_complete": (not fragment or fragment_safe)
                and len(drops) == len(recognized_drops),
            "postcheck_command_sha256": "sha256:" + hashlib.sha256(command.encode()).hexdigest()
                if command else None,
            "postcheck_present": "blueprint-control-plane-postchecks.sh" in command,
            "release_selector_present": "BLUEPRINT_LIVE_CONTROL_PLANE_REPO" in command,
            "recognized_script_paths": sorted(set(script_pattern.findall(command))),
        }

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
        if not UNIT_PATTERN.fullmatch(pattern):
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
            if not SINCE.fullmatch(since):
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

    # -- memory (read-only, fail-soft: anything unreadable is None) --------

    def _cgroup_usage(self, directory: str) -> dict[str, int | None] | None:
        current = parse_cgroup_bytes(_read_small(os.path.join(directory, "memory.current"), _MEMORY_VALUE_READ))
        if current is None:
            return None
        peak = parse_cgroup_bytes(_read_small(os.path.join(directory, "memory.peak"), _MEMORY_VALUE_READ))
        return {"current_bytes": current, "peak_bytes": peak}

    def _system_units(self) -> dict[str, Any] | None:
        rows: list[dict[str, Any]] = []
        truncated = False
        try:
            with os.scandir(os.path.join(self._cgroup_root, "system.slice")) as entries:
                for scanned, entry in enumerate(entries):
                    if scanned >= _MEMORY_SCAN_ENTRIES:
                        truncated = True
                        break
                    name = entry.name
                    if (not CGROUP_UNIT.fullmatch(name) or redact_lines(name) != name
                            or not entry.is_dir(follow_symlinks=False)):
                        continue
                    usage = self._cgroup_usage(entry.path)
                    if usage is not None:
                        rows.append({"unit": name, **usage})
        except OSError:
            return None
        rows.sort(key=lambda row: (-row["current_bytes"], row["unit"]))
        return {"count": len(rows), "scan_truncated": truncated, "top": rows[:MEMORY_TOP_UNITS]}

    def _tmpfs(self) -> dict[str, dict[str, int] | None]:
        usage: dict[str, dict[str, int] | None] = dict.fromkeys(self._tmpfs_paths)
        data = _read_small(self._mounts, _MOUNTS_READ)
        if data is None:
            return usage
        fstype: dict[str, str] = {}
        for line in data.decode("utf-8", "replace").splitlines():
            fields = line.split()
            if len(fields) >= 3 and fields[1] in usage:
                fstype[fields[1]] = fields[2]  # the last mount on a path is the visible one
        for path in usage:
            if fstype.get(path) != "tmpfs":
                continue
            try:
                stats = os.statvfs(path)
            except OSError:
                continue
            usage[path] = {"total_bytes": stats.f_blocks * stats.f_frsize,
                           "used_bytes": (stats.f_blocks - stats.f_bfree) * stats.f_frsize}
        return usage

    def _oom_reports(self) -> dict[str, Any] | None:
        """The kernel's latest OOM-killer reports from the journal; None when it can't be read.

        journalctl, not /dev/kmsg: the door has the systemd-journal group, while
        PrivateDevices and ProtectKernelLogs keep it away from the ring buffer.
        """
        argv = ["journalctl", "-k", f"--since={OOM_SINCE}", "-n", str(_OOM_LOG_LINES),
                "-o", "short-iso", "--no-pager", "-q"]
        try:
            result = self.runner.run(argv, timeout=_OOM_LOG_TIMEOUT)
            if result.returncode != 0:
                return None
            text = result.stdout
            truncated = len(text) > _OOM_LOG_BYTES or text.count("\n") >= _OOM_LOG_LINES
            if len(text) > _OOM_LOG_BYTES:
                text = text[-_OOM_LOG_BYTES:].partition("\n")[2]
            try:
                page_size: int | None = os.sysconf("SC_PAGE_SIZE")
            except (OSError, ValueError):
                page_size = None
            parsed = parse_oom_reports(text, page_size=page_size)
        except Exception:  # noqa: BLE001 - a parser surprise must not hide the rest of memory
            return None
        return {"since": OOM_SINCE, "page_size_bytes": page_size, "log_truncated": truncated, **parsed}

    def memory(self) -> dict[str, Any]:
        """What holds host RAM: meminfo totals, per-unit cgroup v2 memory, tmpfs usage
        and the kernel's latest OOM-killer reports.

        Only numbers, uids, task and unit names leave: never command lines,
        environment or paths beyond unit-name cgroups and the fixed tmpfs mount points.
        """
        meminfo = _read_small(self._meminfo, _MEMINFO_READ)
        return {
            "meminfo": parse_meminfo(meminfo.decode("ascii", "replace")) if meminfo is not None else None,
            "slices": {name: self._cgroup_usage(os.path.join(self._cgroup_root, name)) for name in MEMORY_SLICES},
            "system_units": self._system_units(),
            "tmpfs": self._tmpfs(),
            "oom_reports": self._oom_reports(),
        }

    def version(self) -> dict[str, Any]:
        return self._fetch_json(self.config.intake_version_url)
