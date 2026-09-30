"""The stage child of a remote CPU worker (plan 14 §5, §6): what runs from the release, under its audit.

``python -m blueprint_pipeline.remote_cpu_worker stage`` is spawned by ``execute`` with the extracted
release as its code and its working directory.  It refuses unless ``blueprint_pipeline`` is the
release's, installs the release-path audit, runs one registered handler, and writes its report.
A handler takes the sealed descriptor and ``StageRoots`` and returns the stage's sealed result.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .remote_cpu_environment import environment_record
from .remote_cpu_job_contract import PROBE_STAGE, STAGES, record_bytes, safe_label

PROBE_RESULT_SCHEMA_VERSION = "remote_cpu_environment_probe_result.v1"
MAX_STAGE_REQUEST_BYTES = 32 * 1024 * 1024
MAX_RELEASE_MISSES = 64
_RELEASE_TOPS = ("src", "docs", "pyproject.toml")


@dataclass(frozen=True)
class StageRoots:
    """Where a stage handler finds ``/`` (``local``) and the release it runs from."""

    filesystem_root: Path
    release_root: Path

    def local(self, path: str) -> Path:
        return Path(self.filesystem_root) / str(path).lstrip("/")


def release_read(path: Any, mode: Any, flags: Any, *, roots: Sequence[str], cwd: str) -> tuple[str, str] | None:
    """The path, and its release-relative label, that an ``open`` audit event reads under the release root.

    Writes and creates are not reads, and bytecode caches are not release paths.  A builtin, io, pathlib, zip
    or tar open (the event's ``mode`` is a str) names a path relative to the working directory, which is the
    release root as the host's is its checkout.  An ``os.open`` (``mode`` is None) may be relative to a
    directory descriptor the event does not carry, so a relative one is followed only from a release top
    level; an ``open`` through such an ``opener`` would be misread, and no compile code uses one.
    """
    if isinstance(path, int) or (isinstance(mode, str) and any(flag in mode for flag in "wax+")) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)):
        return None
    name = os.path.normpath(os.fsdecode(os.fspath(path)))
    if not os.path.isabs(name):
        if mode is None and name.split(os.sep, 1)[0] not in _RELEASE_TOPS:
            return None
        name = os.path.normpath(os.path.join(cwd, name))
    root = next((root for root in roots if name.startswith(root)), None)
    if root is None or "__pycache__" in name.split(os.sep):
        return None
    return name, safe_label(name[len(root):])


def install_release_audit(release: str | Path) -> list[str]:
    """Record, in this process, every read of a missing path under the release root (plan 14 §5).

    Only opens are audited: a probe by ``stat``, ``exists`` or ``listdir`` is not observable, so the byte-identity
    test from the archive (plan 14 task 4.3) and shadow mode remain the backstop for those.
    """
    roots = tuple({os.path.normpath(release) + os.sep, os.path.realpath(release) + os.sep})
    misses: list[str] = []

    def hook(event: str, args: tuple[Any, ...]) -> None:
        if event != "open" or len(misses) >= MAX_RELEASE_MISSES or len(args) < 3:
            return
        try:
            found = release_read(args[0], args[1], args[2], roots=roots, cwd=os.getcwd())
            if found is not None and not os.path.exists(found[0]) and found[1] not in misses:
                misses.append(found[1])
        except Exception:  # noqa: BLE001 - a hook that raises would fail the open it observes
            return

    sys.addaudithook(hook)
    return misses


def run_environment_probe(descriptor: Mapping[str, Any], roots: StageRoots) -> dict[str, Any]:
    """``environment_probe`` (plan 14 §8): this worker's census record, sealed as the output and in the result."""
    record = environment_record()
    output = roots.local(descriptor["outputs"]["output_root"])
    output.mkdir(parents=True)
    (output / "environment.json").write_bytes(record_bytes(record))
    result = {"schema_version": PROBE_RESULT_SCHEMA_VERSION, "status": STAGES[PROBE_STAGE]["success_status"],
              "blockers": [], "source_commit": descriptor["code"]["source_commit"], "environment": record,
              "result_digest": ""}
    return {**result, "result_digest": canonical_digest(result, digest_field="result_digest")}


def from_release(release: str | Path) -> bool:
    import blueprint_pipeline

    return Path(blueprint_pipeline.__file__).resolve().is_relative_to(Path(release).resolve())


def stage_main() -> int:
    """The stage child: refuse unless this code is the release's, audit release reads, run one handler."""
    request = json.loads(sys.stdin.buffer.read(MAX_STAGE_REQUEST_BYTES + 1))
    report: dict[str, Any] = {"result": None, "release_path_misses": [], "failures": []}
    if not from_release(request["release_root"]):
        report["failures"] = ["remote_cpu_worker_source_shadowed"]
    else:
        misses = install_release_audit(request["release_root"])
        try:
            module, _, name = str(request["handler"]).partition(":")
            handler = getattr(importlib.import_module(module), name)
            roots = StageRoots(Path(request["filesystem_root"]), Path(request["release_root"]))
            report["result"] = json.loads(json.dumps(handler(request["descriptor"], roots)))
        except Exception as exc:  # noqa: BLE001 - a stage's own failures are in its result; this is infrastructure
            report["failures"] = [f"stage_raised:{type(exc).__name__}"]
        report["release_path_misses"] = list(misses)
    Path(request["report"]).write_text(json.dumps(report), encoding="utf-8")
    return 0
