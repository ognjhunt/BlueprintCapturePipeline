"""Run many test job variations through the whole path that can run in a session's own container, at $0.

    python -m blueprint_pipeline.test_job_matrix --matrix tests/fixtures/job_matrix_v1.json -j 4

Each matrix row is one job variation (a kind of job and what is different about it) and the place it is
expected to end: the last step it reaches and that step's stop. Every variation goes through the same steps in
order (``tests/job_matrix_steps.py``): the route its reading picks, the front half (the replay set's records:
Atlas preview, levelling, mesh registration), the scene chain (all six stages on the host chain, role doubles for
stages 1-4), publication, preparation completion and the enqueue (the replay set's downstream records), the
website half compile and the company zip. A step that cannot run here says ``not_run_here`` and why; a step that
does not apply to the job says ``not_applicable``; each step says what it was fed from. The command prints one
table and writes one JSON report, and exits 0 only when every variation ends where its row says.

Boundaries: synthetic fixtures only (the repository's own test builders); no host, no credential, no provider,
no model. Variables whose names look like credentials are removed from the environment first; every socket
connection and name lookup in a variation's process is refused and recorded (``network_attempts``), and a
variation with any attempt fails. Each variation runs in its own fresh process and its own work folder; the
replay set's records are replayed once beforehand in their own process (they live at fixed paths) and each
variation only reads those results.

What this cannot show: provider behaviour (Atlas, ARTISAN, SAM), a real video, the host's disk, memory and
time. ``docs/runbooks/test-job-matrix.md`` says how to read the report and add a variation.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

MATRIX_SCHEMA = "blueprint_test_job_matrix.v1"
REPORT_SCHEMA = "blueprint_test_job_matrix_report.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
#: Environment names removed before anything runs (values are never read or printed).
CREDENTIAL_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|SERVICE_ACCOUNT)", re.IGNORECASE)
#: Fields of a variation's result that may differ between two runs of the same tree by design.
VARYING_FIELDS = ("seconds",)
EXPECTED_FIELDS = ("last_stage", "kind", "code", "zip_built", "readme_variant", "readme_sha256", "robot_station_absent",
                   "physics_authority", "physics_check", "robot_free_probe", "assumed_thresholds")
#: Proxy settings a child's own subprocesses would use; pointed at a closed local port so nothing leaves.
PROXY_NAMES = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy")
#: The only programs a variation may start: the replay set's two local reads of the source commit
#: (``atlas_preview_replay._source_state``), as ``git -C <repo>`` followed by exactly one of these.
ALLOWED_GIT = (("rev-parse", "HEAD"), ("status", "--porcelain"))
#: Git's own environment (``GIT_CONFIG_*``, ``GIT_EXEC_PATH``, ``GIT_DIR``, ``GIT_ASKPASS``...): any of these could
#: make an allowed git call take a setting or run a program. A child starts without them; a call carrying one is refused.
GIT_ENVIRONMENT = re.compile(r"GIT_")
_ATTEMPTS: list[dict[str, str]] = []


class NetworkRefused(OSError):
    """A variation tried to reach the network."""


def _refuse(name: str):
    def refused(*args: Any, **_kwargs: Any) -> Any:
        _ATTEMPTS.append({"call": name, "target": repr(args[1:2] or args[:1])[:120]})
        raise NetworkRefused(f"test_job_matrix_network_refused:{name}")
    return refused


def install_network_guard() -> list[dict[str, str]]:
    """Refuse and record every outbound connection and name lookup in this process."""
    socket.socket.connect = _refuse("connect")  # type: ignore[method-assign]
    socket.socket.connect_ex = _refuse("connect_ex")  # type: ignore[method-assign]
    socket.socket.sendto = _refuse("sendto")  # type: ignore[method-assign]
    socket.create_connection = _refuse("create_connection")  # type: ignore[assignment]
    socket.getaddrinfo = _refuse("getaddrinfo")  # type: ignore[assignment]
    for name in ("gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
        setattr(socket, name, _refuse(name))
    real = subprocess.Popen

    class Guarded(real):  # type: ignore[valid-type, misc]
        def __init__(self, args: Any, *rest: Any, **options: Any) -> None:
            argv = str(args).split() if isinstance(args, (str, bytes)) else [str(a) for a in args]
            program = Path(argv[0]).name if argv else ""
            env = options.get("env")
            # A list only (no shell, no string, no other executable or pre-exec hook), and no git setting from the environment.
            if (isinstance(args, (str, bytes)) or options.get("shell") or rest
                    or options.get("executable") is not None or options.get("preexec_fn") is not None
                    or any(GIT_ENVIRONMENT.match(str(name)) for name in (os.environ if env is None else env))
                    or not (len(argv) == 5 and argv[:2] == ["git", "-C"] and tuple(argv[3:]) in ALLOWED_GIT)):
                _ATTEMPTS.append({"call": "subprocess", "target": " ".join(argv[:2])[:120]})
                raise NetworkRefused(f"test_job_matrix_subprocess_refused:{program}"[:200])
            super().__init__(args, *rest, **options)
    subprocess.Popen = Guarded  # type: ignore[misc]
    return _ATTEMPTS


def scrub_credentials(environ: Any = os.environ) -> list[str]:
    """Remove credential-like variables; return their names only."""
    names = sorted(name for name in environ if CREDENTIAL_NAME.search(name))
    for name in names:
        environ.pop(name, None)
    return names


def _child_setup(repo: str) -> None:
    for entry in (repo, os.path.join(repo, "src")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    scrub_credentials()  # this spawned child's own copy of the environment, never the caller's
    for name in [key for key in os.environ if GIT_ENVIRONMENT.match(key)]:
        del os.environ[name]  # the rows' two git reads need no setting from the caller's git environment
    for name in [*PROXY_NAMES, *(key for key in os.environ if key.startswith("CLOUDSDK_PROXY"))]:
        os.environ[name] = "http://127.0.0.1:9"
    install_network_guard()


def _replay_records(repo: str, set_dir: str, records: list[str]) -> dict[str, Any]:
    from .replay_set import run_set

    result = run_set(Path(repo) / set_dir, records=records, echo=lambda _line: None)
    rows = {}
    for row in result["rows"]:
        rows[row["record_id"]] = {key: row[key] for key in ("record_id", "job_kind", "tool", "expected", "actual",
                                                            "breaches", "error", "passed", "seconds")}
    return {"rows": rows, "network_attempts": list(_ATTEMPTS)}


def _variation(repo: str, row: Mapping[str, Any], work: str, replay: Mapping[str, Any]) -> dict[str, Any]:
    from tests.job_matrix_steps import run_variation

    started = time.monotonic()
    try:
        result = run_variation(row, Path(work), replay)
    except Exception as exc:  # noqa: BLE001 - an uncaught error is this variation's result, not the run's
        result = {"label": row["label"], "kind": row["kind"], "last_stage": "error",
                  "first_stop": {"kind": "error", "code": f"{type(exc).__name__}:{str(exc)[:200]}"},
                  "zip_built": False, "zip_sha256": None, "zip_members": None, "readme_variant": None,
                  "steps": {}, "seconds": {}, "error": traceback.format_exc()[-6000:]}
    result["seconds"]["total"] = round(time.monotonic() - started, 3)
    result["network_attempts"] = list(_ATTEMPTS)
    return result


def load_matrix(path: Path) -> dict[str, Any]:
    matrix = json.loads(path.read_text(encoding="utf-8"))
    if matrix.get("schema_version") != MATRIX_SCHEMA:
        raise ValueError(f"not a test job matrix: {path}")
    if set(matrix.get("readme_sentences") or {}) != {"made", "physics_check", "notes"}:
        raise ValueError("matrix readme_sentences must name made, physics_check and notes")
    labels = [row["label"] for row in matrix["variations"]]
    if len(labels) != len(set(labels)):
        raise ValueError("matrix labels repeat")
    for row in matrix["variations"]:
        missing = sorted({"label", "kind", "reading", "scene", "compile", "expected"} - set(row))
        if missing or not set(row["expected"]) <= set(EXPECTED_FIELDS) or "last_stage" not in row["expected"]:
            raise ValueError(f"matrix row {row.get('label')} is incomplete: {missing or row['expected']}")
    return matrix


def compare(result: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    """The one comparison, for the command line and for the tests alike."""
    actual = {**{field: result.get(field) for field in EXPECTED_FIELDS},
              "kind": result["first_stop"].get("kind"), "code": result["first_stop"].get("code")}
    # The whole README is pinned by its digest; when it differs, the README itself is shown.
    differences = [("readme_sha256_differs" if field == "readme_sha256" else field)
                   + f": expected {expected[field]!r}, got {actual[field]!r}"
                   + (f"\nREADME as built:\n{result.get('readme')}" if field == "readme_sha256" else "")
                   for field in EXPECTED_FIELDS if field in expected and expected[field] != actual[field]]
    if result.get("network_attempts"):
        differences.append(f"network_attempts: {len(result['network_attempts'])}")
    return differences


def _without(value: Any, fields: Sequence[str]) -> Any:
    if isinstance(value, Mapping):
        return {k: _without(v, fields) for k, v in value.items() if k not in fields}
    if isinstance(value, list):
        return [_without(item, fields) for item in value]
    return value


def stable(result: Mapping[str, Any]) -> dict[str, Any]:
    """A result without the fields that vary by design (``VARYING_FIELDS``, at any depth) and its work folder."""
    text = json.dumps(_without(result, VARYING_FIELDS), sort_keys=True)
    work = result.get("work")
    return json.loads(text.replace(json.dumps(work)[1:-1], "{work}") if work else text)


def table(results: Sequence[Mapping[str, Any]]) -> str:
    header = ("ok", "label", "kind", "last stage", "stop", "zip", "readme", "secs")
    rows = [header]
    for r in results:
        stop = r["first_stop"]
        rows.append(("PASS" if not r["differences"] else "FAIL", r["label"], r["kind"], r["last_stage"],
                     f"{stop.get('kind')}:{stop.get('code')}"[:70],
                     (r["zip_sha256"] or "-")[7:19] if r["zip_built"] else "-",
                     ",".join(r["readme_variant"] or []) or "-", f"{r['seconds'].get('total', 0):.1f}"))
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(header))]
    lines = ["  ".join(str(cell).ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
             for row in rows]
    for r in results:
        lines.extend(f"  {r['label']}: {difference}" for difference in r["differences"])
    return "\n".join(lines)


def run_matrix(matrix_path: str | Path, *, jobs: int = 1, only: Sequence[str] = (), work_root: str | Path | None = None,
               keep: bool = False) -> dict[str, Any]:
    started = time.monotonic()
    matrix_path = Path(matrix_path)
    matrix = load_matrix(matrix_path if matrix_path.is_absolute() else REPO_ROOT / matrix_path)
    rows = [row for row in matrix["variations"] if not only or row["label"] in only]
    unknown = sorted(set(only) - {row["label"] for row in rows})
    if unknown:
        raise ValueError(f"no such variation: {', '.join(unknown)}")
    if work_root is not None:
        Path(work_root).mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="test-job-matrix-", dir=work_root))
    context = multiprocessing.get_context("spawn")
    repo = str(REPO_ROOT)
    try:
        records = sorted({record for row in rows for record in (*row.get("front_half", ()),
                                                                *row.get("downstream", ()))})
        replay_started = time.monotonic()
        with ProcessPoolExecutor(max_workers=1, mp_context=context, initializer=_child_setup,
                                 initargs=(repo,), max_tasks_per_child=1) as pool:
            replay = pool.submit(_replay_records, repo, matrix["replay_set"], records).result() if records \
                else {"rows": {}, "network_attempts": []}
        replay_seconds = round(time.monotonic() - replay_started, 2)
        futures = []
        with ProcessPoolExecutor(max_workers=max(1, jobs), mp_context=context, initializer=_child_setup,
                                 initargs=(repo,), max_tasks_per_child=1) as pool:
            for row in rows:
                work = root / row["label"]
                work.mkdir()
                futures.append((row, str(work), pool.submit(
                    _variation, repo, {**row, "readme_sentences": matrix["readme_sentences"]}, str(work),
                    replay["rows"])))
            results = []
            for row, work, future in futures:
                result = future.result()
                result["work"] = work
                result["expected"] = dict(row["expected"])
                result["differences"] = compare(result, row["expected"])
                results.append(result)
    finally:
        if not keep:
            shutil.rmtree(root, ignore_errors=True)
    failed = [r["label"] for r in results if r["differences"]]
    if replay["network_attempts"]:
        failed.append("replay_set:network_attempts")
    return {"schema_version": REPORT_SCHEMA, "matrix": str(matrix_path), "jobs": jobs,
            "passed": not failed, "failed": failed, "variations": results,
            "replay_set": {"records": replay["rows"], "seconds": replay_seconds,
                           "network_attempts": replay["network_attempts"]},
            "work_root": str(root), "kept": keep,
            "varying_fields": [*VARYING_FIELDS, "work"], "elapsed_seconds": round(time.monotonic() - started, 2)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix", required=True, help="the matrix file (tests/fixtures/job_matrix_v1.json)")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="variations at once, each in its own process")
    parser.add_argument("--only", action="append", default=[], help="run only this variation (repeatable)")
    parser.add_argument("--report", help="write the JSON report here")
    parser.add_argument("--work-root", help="make the work folders under this directory (default: system temp)")
    parser.add_argument("--keep", action="store_true", help="keep the work folders")
    args = parser.parse_args(argv)
    # Run through the package module, so the worker functions a spawned process unpickles are importable by name
    # even when this file is ``__main__`` (``python -m``).
    from blueprint_pipeline import test_job_matrix as matrix_module

    try:
        report = matrix_module.run_matrix(args.matrix, jobs=args.jobs, only=args.only, work_root=args.work_root, keep=args.keep)
    except ValueError as exc:
        parser.error(str(exc))
    print(table(report["variations"]))
    print("The scene chain and the website compile run on different fixture objects: no row is end to end.")
    print(f"{len(report['variations']) - len([f for f in report['failed'] if ':' not in f])}/"
          f"{len(report['variations'])} variations ended where expected in {report['elapsed_seconds']}s "
          f"(-j {args.jobs}; replay set {report['replay_set']['seconds']}s; synthetic fixtures, $0)")
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


__all__ = ["MATRIX_SCHEMA", "REPORT_SCHEMA", "compare", "install_network_guard", "load_matrix", "main",
           "run_matrix", "scrub_credentials", "stable", "table"]

if __name__ == "__main__":
    raise SystemExit(main())
