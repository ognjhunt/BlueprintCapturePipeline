"""Replay a committed set of job records through the $0 replay tools and check each one's first stop.

A fix for a stopped job is otherwise tested against that one job, by deploying and waiting. This runs
every record of a set (``tests/fixtures/replay_set_v1``) through the replay tool(s) its manifest entry
names and compares the tool's first stop with the one the record expects, so one command shows whether
a change moved any job kind's stop. It prints one line per record and tool and exits 1 on any mismatch.

    python -m blueprint_pipeline.replay_set --set tests/fixtures/replay_set_v1

Boundaries are the replay tools' own (each module's docstring lists them): sockets, paid calls and
WebApp writes stop the replay and are reported, never made. This runner adds none and lifts none; it
also fails a record whose report shows any network attempt, any new paid call, or a changed input.

A record names absolute paths (a launch receipt names its result by path under its own digest), so it
is put back at ``<materialize_root>/<record_id>/host`` before it is replayed: committed ``*.gz`` files
are decompressed and the empty directories it declares are made. The materialize root is locked while
a set runs and each record's copy is removed afterwards. The tool reads only that copy and writes only
under ``--replay-root`` (a temporary directory unless given).

Tools: ``task_evaluation_downstream_replay`` (completion -> readiness -> team packet -> probe enqueue ->
screening plan); ``atlas_preview_replay`` (the Atlas preview controller, from upload accepted to its
first stop, run in this process on a working copy at the record's own path while the materialized
copy is set aside and digested); ``atlas_reconstruction_stage_replay`` (levelling and mesh
registration on their own retained inputs; the controller reaches them only after paid stages whose
outputs no synthetic record carries). The other replay tools need host-pulled inputs no synthetic
record stands for yet.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import gzip
import json
import shutil
import sys
import tempfile
import time
import types
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "blueprint_replay_set_manifest.v1"
MANIFEST_FILENAME = "manifest.json"
# The synthetic records carry this root in their absolute paths (the manifest's
# materialize_root), so it is fixed rather than a fresh temporary directory. It
# holds committed test fixtures only.
DEFAULT_MATERIALIZE_ROOT = "/tmp/blueprint-replay-set-v1"  # nosec B108 - fixed root of committed synthetic fixtures
#: Manifest field: the wall-clock instant every record is replayed at (``_replay_clock``).
CLOCK_FIELD = "replay_clock_epoch"
GZIP_SUFFIX = ".gz"
STOP_FIELDS = ("kind", "stage", "code")
#: Compared only where a record's expected stop names it: the job's state once the replay stopped
#: (``final_job.state``), which tells a job that ends at once from one that waits or retries.
FINAL_STATE = "final_state"


def _downstream(inputs: Mapping[str, str], *, host: Path, until: str, replay_root: Path) -> dict[str, Any]:
    from .task_evaluation_downstream_replay import replay

    paths = {name: host / relative for name, relative in inputs.items()}
    return replay(**paths, until=until, replay_root=replay_root)


def _atlas_preview(inputs: Mapping[str, str], *, host: Path, until: str, replay_root: Path) -> dict[str, Any]:
    """The Atlas preview controller until its first stop, on a working copy standing at ``host``.

    Retained records name absolute paths under ``host``, so the copy stands there (as the replay's own
    private bind mount would put it) and the materialized record is set aside while it runs. This
    stays in this process, so the replay clock reaches the controller; ``evidence`` digests the
    set-aside record before and after.
    """
    from .atlas_preview_replay import _source_state, _tree_digest, run_steps
    from .common import write_json

    if until != "first_stop":
        raise ValueError(f"atlas_preview_replay runs until first_stop, not {until}")
    job = host / inputs["job"]
    evidence = host.with_name(host.name + ".evidence")
    host.rename(evidence)
    try:
        before = _tree_digest(evidence)
        shutil.copytree(evidence, host, symlinks=True)
        core = run_steps(job_path=job, capture_root=job.parents[2], workspace=job.parents[2],
                         commit=_source_state(Path(__file__).resolve().parents[2])[0])
    finally:
        shutil.rmtree(host, ignore_errors=True)
        evidence.rename(host)
    after = _tree_digest(host)
    report = {**core, "evidence": {"digest_before": before, "digest_after": after, "unchanged": before == after}}
    write_json(replay_root / "atlas_preview_replay_report.v1.json", report)
    return report


def _reconstruction_stage(inputs: Mapping[str, str], *, host: Path, until: str,
                          replay_root: Path) -> dict[str, Any]:
    from .atlas_reconstruction_stage_replay import replay

    return replay(stage=until, inputs={name: host / relative for name, relative in inputs.items()},
                  replay_root=replay_root)


TOOLS: dict[str, Callable[..., dict[str, Any]]] = {"task_evaluation_downstream_replay": _downstream,
                                                   "atlas_preview_replay": _atlas_preview,
                                                   "atlas_reconstruction_stage_replay": _reconstruction_stage}


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
    fields = (*STOP_FIELDS, *((FINAL_STATE,) if FINAL_STATE in stop else ()))
    return {field: str(stop.get(field)).replace("{host}", str(host)) if stop.get(field) is not None else None
            for field in fields}


def _actual(report: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    stop = report.get("first_stop") or {}
    actual = {field: stop.get(field) for field in STOP_FIELDS}
    if FINAL_STATE in expected:
        actual[FINAL_STATE] = (report.get("final_job") or {}).get("state")
    return actual


class _ReplayDatetime(datetime.datetime):
    """``datetime.now`` on the replay clock. (``date.today`` already reads ``time.time``.)"""

    @classmethod
    def now(cls, tz: datetime.tzinfo | None = None) -> Any:
        return cls.fromtimestamp(time.time(), tz)

    @classmethod
    def utcnow(cls) -> Any:
        return cls.fromtimestamp(time.time(), datetime.timezone.utc).replace(tzinfo=None)


@contextlib.contextmanager
def _replay_clock(epoch: float | None) -> Iterator[None]:
    """Run the replayed chain at the instant the records were taken.

    A record carries real times (a grant's consent and its expiry, a price review's date), and the
    replayed chain compares them with the clock. At today's clock every record fails once its window
    has passed: the first committed set stopped passing a day after it was built. So the clock the
    chain reads (``time.time``, ``datetime.now``, a ``now=time.time`` default bound at import, a
    ``from time import time``) starts at the manifest's ``replay_clock_epoch`` and advances with
    elapsed time. A manifest without one (an older set) is replayed at the real clock.
    """
    if epoch is None:
        yield
        return
    real_time, real_datetime, origin = time.time, datetime.datetime, time.monotonic()
    clock = time.time = lambda: epoch + (time.monotonic() - origin)
    datetime.datetime = _ReplayDatetime  # type: ignore[misc]
    _rebind(real_time, clock)
    _rebind(real_datetime, _ReplayDatetime)
    try:
        yield
    finally:
        time.time, datetime.datetime = real_time, real_datetime  # type: ignore[misc]
        # Also in a module first imported during the replay, which bound the replay clock.
        _rebind(clock, real_time)
        _rebind(_ReplayDatetime, real_datetime)


def _rebind(old: Any, new: Any) -> None:
    """Point every loaded package module's global, and its functions' defaults, that is ``old`` at ``new``."""
    for name, module in list(sys.modules.items()):
        if module is None or name == __name__ or not (
                name == "blueprint_pipeline" or name.startswith("blueprint_pipeline.")):
            continue
        for key, value in list(vars(module).items()):
            if value is old:
                setattr(module, key, new)
            elif isinstance(value, types.FunctionType) and value.__module__ == name:
                if value.__defaults__ and any(item is old for item in value.__defaults__):
                    value.__defaults__ = tuple(new if item is old else item for item in value.__defaults__)
                if value.__kwdefaults__ and any(item is old for item in value.__kwdefaults__.values()):
                    value.__kwdefaults__ = {field: new if item is old else item
                                            for field, item in value.__kwdefaults__.items()}


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
    ending = f" ({FINAL_STATE}={stop[FINAL_STATE]})" if FINAL_STATE in stop else ""
    return f"{stop.get('kind')}/{stop.get('stage')}/{code}{ending}"


def run_set(set_dir: str | Path, *, replay_root: str | Path | None = None, records: list[str] | None = None,
            materialize_root: str | Path | None = None, echo: Callable[[str], None] = print) -> dict[str, Any]:
    """Replay every record of the set; return the rows and whether all of them matched."""
    set_dir = Path(set_dir)
    manifest = json.loads((set_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(f"not a replay set manifest: {set_dir / MANIFEST_FILENAME}")
    entries = [row for row in manifest["records"] if not records or row["record_id"] in records]
    unknown = sorted(set(records or []) - {row["record_id"] for row in entries})
    if unknown:
        raise ValueError(f"no such record in the set: {', '.join(unknown)}")
    root = Path(materialize_root or manifest.get("materialize_root") or DEFAULT_MATERIALIZE_ROOT)
    clock = manifest.get(CLOCK_FIELD)
    if clock is not None and (isinstance(clock, bool) or not isinstance(clock, (int, float)) or clock <= 0):
        raise ValueError(f"{CLOCK_FIELD} is not a time: {set_dir / MANIFEST_FILENAME}")
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
                for tool in entry["tools"]:
                    expected = _expected(tool["expected_stop"], host)
                    runner = TOOLS.get(tool["tool"])
                    if runner is None:
                        actual, breaches, error = {}, [], f"unknown_tool:{tool['tool']}"
                    else:
                        tool_started = time.monotonic()
                        with _replay_clock(clock):
                            report = runner(tool["inputs"], host=host, until=tool["until"],
                                            replay_root=scratch / record_id)
                        actual, breaches, error = _actual(report, expected), _boundary_breaches(report), None
                    passed = error is None and not breaches and actual == expected
                    row = {"record_id": record_id, "job_kind": entry["job_kind"], "tool": tool["tool"],
                           "synthetic": entry.get("synthetic", False), "expected": expected, "actual": actual,
                           "breaches": breaches, "error": error, "passed": passed,
                           "seconds": round(time.monotonic() - tool_started, 2) if runner else 0.0}
                    rows.append(row)
                    note = ",".join(breaches) or error or ""
                    echo(f"{'PASS' if passed else 'FAIL'}  {record_id}  kind={entry['job_kind']}  "
                         f"tool={tool['tool']}  expected={_short(expected, host)}  "
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


__all__ = ["CLOCK_FIELD", "DEFAULT_MATERIALIZE_ROOT", "FINAL_STATE", "MANIFEST_FILENAME", "MANIFEST_SCHEMA", "TOOLS", "main",
           "materialize", "run_set"]

if __name__ == "__main__":
    raise SystemExit(main())
