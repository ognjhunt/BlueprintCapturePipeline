"""Replay a committed set of job records through the $0 replay tools and check each one's first stop.

A fix for a stopped job is otherwise tested against that one job, by deploying and waiting. This runs
every record of a set (``tests/fixtures/replay_set_v1``) through the replay tool(s) its manifest entry
names and compares the tool's first stop with the one the record expects, so one command shows whether
a change moved any job kind's stop. It prints one line per record and tool and exits 1 on any mismatch.

    python -m blueprint_pipeline.replay_set --set tests/fixtures/replay_set_v1

A record replays at its own time. Each manifest entry declares ``replay_at_epoch``, a moment inside the
validity windows its grants carry, and the wall clock the replayed code reads (``time.time`` and friends,
``datetime.now``; see :func:`pinned_clock`) starts there for exactly the duration of that record's tool run
and is restored afterwards, so a committed record gives the same first stop on any day. A record that
declares no replay time is failed, not replayed at today's clock. Durations still use ``time.monotonic``.
No grant, window or consent check is skipped: the record is read as of its own time.

Boundaries are the replay tools' own (each module's docstring lists them): sockets, paid calls and
WebApp writes stop the replay and are reported, never made. This runner adds none and lifts none; it
also fails a record whose report shows any network attempt, any new paid call, or a changed input.

A record names absolute paths (a launch receipt names its result by path under its own digest), so it
is put back at ``<materialize_root>/<record_id>/host`` before it is replayed: committed ``*.gz`` files
are decompressed and the empty directories it declares are made. The materialize root is locked while
a set runs and each record's copy is removed afterwards. The tool reads only that copy and writes only
under ``--replay-root`` (a temporary directory unless given).

Tools: ``task_evaluation_downstream_replay`` (completion -> readiness -> team packet -> probe enqueue ->
screening plan). The other replay tools need host-pulled inputs no synthetic record stands for yet.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import gzip
import json
import math
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

# v2: every record declares ``replay_at_epoch``, the wall-clock time it replays at (v1 replayed at today's).
MANIFEST_SCHEMA = "blueprint_replay_set_manifest.v2"
REPLAY_AT_FIELD = "replay_at_epoch"
MANIFEST_FILENAME = "manifest.json"
# The synthetic records carry this root in their absolute paths (the manifest's
# materialize_root), so it is fixed rather than a fresh temporary directory. It
# holds committed test fixtures only.
DEFAULT_MATERIALIZE_ROOT = "/tmp/blueprint-replay-set-v1"  # nosec B108 - fixed root of committed synthetic fixtures
GZIP_SUFFIX = ".gz"
STOP_FIELDS = ("kind", "stage", "code")


class _ReplayClock:
    """The wall clock a record replays at: its declared epoch when the tool starts, advancing in real seconds.

    Only read while :func:`pinned_clock` is active. A reference that outlives the pin (a module first
    imported during a replay keeps the name it bound) reads the real clock again.
    """

    def __init__(self) -> None:
        self.epoch: float | None = None
        self.started_ns = 0

    def ns(self) -> int:
        if self.epoch is None:
            return _REAL["time_ns"]()
        return int(self.epoch * 1_000_000_000) + time.monotonic_ns() - self.started_ns

    def seconds(self) -> float:
        return self.ns() / 1_000_000_000 if self.epoch is not None else _REAL["time"]()


_REAL: dict[str, Callable[..., Any]] = {name: getattr(time, name) for name in (
    "time", "time_ns", "gmtime", "localtime", "ctime", "asctime", "strftime")}
_CLOCK = _ReplayClock()


class _AnyDatetime(type):
    """``isinstance``/``issubclass`` against the pinned class answer as they would for ``datetime.datetime``."""

    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, _REAL_DATETIME)

    def __subclasscheck__(cls, subclass: type) -> bool:
        return issubclass(subclass, _REAL_DATETIME)


_REAL_DATETIME = datetime.datetime


class _PinnedDatetime(datetime.datetime, metaclass=_AnyDatetime):
    """``datetime.datetime`` whose ``now``/``utcnow``/``today`` read the replay clock."""

    @classmethod
    def now(cls, tz: datetime.tzinfo | None = None) -> datetime.datetime:  # type: ignore[override]
        return cls.fromtimestamp(_CLOCK.seconds(), tz)

    @classmethod
    def utcnow(cls) -> datetime.datetime:  # type: ignore[override]
        return cls.fromtimestamp(_CLOCK.seconds(), datetime.timezone.utc).replace(tzinfo=None)

    @classmethod
    def today(cls) -> datetime.datetime:  # type: ignore[override]
        return cls.fromtimestamp(_CLOCK.seconds())


def _at_clock(name: str, *, position: int) -> Callable[..., Any]:
    """``time.<name>`` with its omitted time argument read from the replay clock, not the C clock."""
    real = _REAL[name]

    def call(*args: Any) -> Any:
        if len(args) > position or _CLOCK.epoch is None:
            return real(*args)
        moment = _CLOCK.seconds()
        return real(*args, _REAL["localtime"](moment) if name in {"strftime", "asctime"} else moment)
    return call


@contextlib.contextmanager
def pinned_clock(epoch: float) -> Iterator[None]:
    """Run the body with the process's wall clock starting at ``epoch``; restore it on exit, also on raise.

    Pinned: ``time.time``, ``time.time_ns``, the no-argument forms of ``time.gmtime``/``localtime``/
    ``ctime``/``asctime``/``strftime``, and ``datetime.datetime.now``/``utcnow``/``today`` (the stdlib
    attributes and every loaded ``blueprint_pipeline`` module's alias of them, with the replay tools' own
    ``_swap``). Not pinned: ``time.monotonic``/``perf_counter`` (durations), ``datetime.date.today``
    (nothing on the replayed chain reads it) and file modification times. Only the replay-set runner
    enters this; no service imports this module.
    """
    from .atlas_preview_replay import _swap

    if _CLOCK.epoch is not None:
        raise RuntimeError("the replay clock is already pinned")
    with contextlib.ExitStack() as stack:
        _CLOCK.epoch, _CLOCK.started_ns = float(epoch), time.monotonic_ns()
        stack.callback(setattr, _CLOCK, "epoch", None)
        _swap(stack, time, "time", _CLOCK.seconds)
        _swap(stack, time, "time_ns", _CLOCK.ns)
        for name in ("gmtime", "localtime", "ctime"):
            _swap(stack, time, name, _at_clock(name, position=0))
        for name in ("asctime", "strftime"):
            _swap(stack, time, name, _at_clock(name, position=1 if name == "strftime" else 0))
        _swap(stack, datetime, "datetime", _PinnedDatetime)
        yield


def _downstream(inputs: Mapping[str, str], *, host: Path, until: str, replay_root: Path) -> dict[str, Any]:
    from .task_evaluation_downstream_replay import replay

    paths = {name: host / relative for name, relative in inputs.items()}
    return replay(**paths, until=until, replay_root=replay_root)


TOOLS: dict[str, Callable[..., dict[str, Any]]] = {"task_evaluation_downstream_replay": _downstream}


@contextlib.contextmanager
def _locked(root: Path) -> Iterator[None]:
    """One set run at a time per materialize root (records are put back at fixed paths)."""
    root.mkdir(parents=True, exist_ok=True)
    with open(root / ".lock", "w", encoding="utf-8") as handle:
        try:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - not POSIX
            pass
        yield


def materialize(record_dir: Path, target: Path, *, empty_dirs: list[str]) -> None:
    """Put a committed record back at ``target`` (its host root) byte for byte as it was built."""
    if target.exists():
        shutil.rmtree(target)
    for path in sorted(record_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(record_dir)
        data = path.read_bytes()
        if relative.name.endswith(GZIP_SUFFIX):
            relative = relative.with_name(relative.name[:-len(GZIP_SUFFIX)])
            data = gzip.decompress(data)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        destination.chmod(path.stat().st_mode & 0o777)
    for relative in empty_dirs:
        (target / relative).mkdir(parents=True, exist_ok=True)


def _expected(stop: Mapping[str, Any], host: Path) -> dict[str, Any]:
    return {field: str(stop.get(field)).replace("{host}", str(host)) if stop.get(field) is not None else None
            for field in STOP_FIELDS}


def _actual(report: Mapping[str, Any]) -> dict[str, Any]:
    stop = report.get("first_stop") or {}
    return {field: stop.get(field) for field in STOP_FIELDS}


def _boundary_breaches(report: Mapping[str, Any]) -> list[str]:
    breaches = []
    if report.get("network_attempts"):
        breaches.append("network_attempted")
    if report.get("new_paid_calls"):
        breaches.append("paid_call_made")
    if report.get("provider_mutation_performed"):
        breaches.append("provider_mutated")
    if not (report.get("evidence") or {}).get("unchanged", False):
        breaches.append("inputs_changed")
    return breaches


def _short(stop: Mapping[str, Any], host: Path) -> str:
    code = str(stop.get("code")).replace(str(host), "{host}")
    return f"{stop.get('kind')}/{stop.get('stage')}/{code}"


def replay_time(entry: Mapping[str, Any]) -> float | None:
    """The record's declared replay time (epoch seconds), or None when it declares none usable."""
    value = entry.get(REPLAY_AT_FIELD)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        return None
    return float(value)


def run_set(set_dir: str | Path, *, replay_root: str | Path | None = None, records: list[str] | None = None,
            materialize_root: str | Path | None = None, echo: Callable[[str], None] = print) -> dict[str, Any]:
    """Replay every record of the set; return the rows and whether all of them matched."""
    set_dir = Path(set_dir)
    manifest = json.loads((set_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(f"not a {MANIFEST_SCHEMA} replay set manifest: {set_dir / MANIFEST_FILENAME}"
                         f" (it is {manifest.get('schema_version')!r}; v2 records declare {REPLAY_AT_FIELD})")
    entries = [row for row in manifest["records"] if not records or row["record_id"] in records]
    unknown = sorted(set(records or []) - {row["record_id"] for row in entries})
    if unknown:
        raise ValueError(f"no such record in the set: {', '.join(unknown)}")
    root = Path(materialize_root or manifest.get("materialize_root") or DEFAULT_MATERIALIZE_ROOT)
    started, rows = time.monotonic(), []
    with contextlib.ExitStack() as stack:
        scratch = Path(replay_root) if replay_root else Path(stack.enter_context(
            tempfile.TemporaryDirectory(prefix="replay-set-")))
        stack.enter_context(_locked(root))
        for entry in entries:
            record_id = entry["record_id"]
            host = root / record_id / "host"
            materialize(set_dir / "records" / record_id, host, empty_dirs=entry.get("empty_dirs") or [])
            try:
                replay_at = replay_time(entry)
                for tool in entry["tools"]:
                    expected = _expected(tool["expected_stop"], host)
                    runner = TOOLS.get(tool["tool"])
                    if runner is None:
                        actual, breaches, error = {}, [], f"unknown_tool:{tool['tool']}"
                    elif replay_at is None:
                        # Never replayed at today's clock: its grants and windows would expire under it.
                        actual, breaches, error = {}, [], f"{REPLAY_AT_FIELD}_not_declared"
                    else:
                        tool_started = time.monotonic()
                        with pinned_clock(replay_at):
                            report = runner(tool["inputs"], host=host, until=tool["until"],
                                            replay_root=scratch / record_id)
                        actual, breaches, error = _actual(report), _boundary_breaches(report), None
                    passed = error is None and not breaches and actual == expected
                    row = {"record_id": record_id, "job_kind": entry["job_kind"], "tool": tool["tool"],
                           "synthetic": entry.get("synthetic", False), "expected": expected, "actual": actual,
                           "breaches": breaches, "error": error, "passed": passed,
                           "replay_at_epoch": replay_at,
                           "seconds": round(time.monotonic() - tool_started, 2) if error is None else 0.0}
                    rows.append(row)
                    note = ",".join(breaches) or error or ""
                    at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(replay_at)) if replay_at else "-"
                    echo(f"{'PASS' if passed else 'FAIL'}  {record_id}  kind={entry['job_kind']}  "
                         f"tool={tool['tool']}  at={at}  expected={_short(expected, host)}  "
                         f"actual={_short(actual, host) if actual else '-'}  {row['seconds']}s"
                         + (f"  {note}" if note else ""))
            finally:
                shutil.rmtree(root / record_id, ignore_errors=True)
    elapsed = round(time.monotonic() - started, 2)
    failed = [row for row in rows if not row["passed"]]
    echo(f"{len(rows) - len(failed)}/{len(rows)} passed in {elapsed}s"
         + (" (synthetic records)" if manifest.get("synthetic") else ""))
    return {"set": str(set_dir), "rows": rows, "passed": not failed and bool(rows), "elapsed_seconds": elapsed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", required=True, dest="set_dir", help="the replay set directory (has manifest.json)")
    parser.add_argument("--record", action="append", help="replay only this record (repeatable)")
    parser.add_argument("--replay-root", help="keep the replays' reports here instead of a temporary directory")
    parser.add_argument("--materialize-root", help="put records back here (only if they were built there)")
    args = parser.parse_args(argv)
    try:
        result = run_set(args.set_dir, replay_root=args.replay_root, records=args.record,
                         materialize_root=args.materialize_root)
    except ValueError as exc:
        parser.error(str(exc))
    sys.stdout.flush()
    return 0 if result["passed"] else 1


__all__ = ["DEFAULT_MATERIALIZE_ROOT", "MANIFEST_FILENAME", "MANIFEST_SCHEMA", "REPLAY_AT_FIELD", "TOOLS", "main",
           "materialize", "pinned_clock", "replay_time", "run_set"]

if __name__ == "__main__":
    raise SystemExit(main())
