"""Prune old scene-retirement runtime snapshots: a plan lists every item and its size; an owner-approved apply removes only those.

Every deploy installs a new protected runtime (``scripts/install_scene_retirement_runtime.py``):
a source generation (``scene-retirement-runtime/generations/<digest>``, about 70 MiB), when the
dependencies changed an SDK generation (``sdk-generations/<digest>``, about 1 GiB), and the
inputs it was built from (``scene-retirement-runtime-sdk-inputs/release-inputs/<commit>``, about
80 MiB; ``sdk-inputs/<digest>``; ``wheel-artifacts/<hash>``). The installer never removes one.
Its stated reason (``_generation``): "Prior immutable generations may still be executing; neither
enumerate nor delete them and do not impose a permanent deploy-count ceiling." A unit started by
``continuous_bootstrap.py`` runs the generation ``CURRENT.json`` named when it started and keeps
importing modules from it lazily for as long as it runs. A source generation is pure Python held
only in that process's ``sys.path``, so nothing in its command line, maps or files would name it.
Since this release the bootstrap holds the selected source and SDK directories open for the
process's whole life (``_hold_selection``): ``/proc/<pid>/fd`` names them, and the descriptor goes
away with the process.

What is always kept:

* every generation the ``CURRENT.json`` history names: the current cohort, then each ``previous``
  record as far as a copy of it is retained (this module keeps a copy of every cohort record it
  reads, under ``<report root>/cohorts``; the installer keeps none, so the history reaches back only
  to the first plan run);
* the active release's own release input, and the three most recently created items of each kind;
* anything newer than the minimum age (seven days unless stated, never under two);
* anything a live process names or holds (command line, environment, working directory,
  executable, open descriptors or memory maps), the bootstraps' held generations included;
* while any bootstrap process holds no descriptor on a source generation directory (one started
  before this release; a mapped SDK library does not count), every source generation, and while
  one holds none on an SDK directory, every SDK generation: its generation cannot be known;
* every wheel a kept release input's ``uv.lock`` names; and everything this module never lists (the
  top of the runtime root, the SDK tools, the git inputs).

The plan is a report only (``plans/<digest>.json`` and ``plan.json``). The apply needs an
owner-approved manifest (``approvals/<id>.json`` beside the release: who approved, the evidence,
when, its expiry, and the digest of the exact plan it approves); without one, or for another plan,
it refuses. The approval binds the plan's own digest (its items and their identities), never the
bytes of ``CURRENT.json``, so deploys between plan and apply do not void it. The apply finishes any
tree a crashed earlier run left renamed aside, then takes each listed item in turn: it takes the
installer's own lock for that item only (a deploy that holds it stops the run), derives the whole
keep set again from the present state (current cohort and its history, the active release, the
three most recent of each kind, age, use, bootstrap processes of unknown selection), skips and
reports an item that has since become one to keep, checks that it is unchanged since the plan,
writes a receipt, renames it aside, removes it and its record, and writes the receipt again. A stop
request (``SIGTERM``, as a deploy stops the unit) ends the run between items; a run killed inside an
item leaves its receipt at ``removing``, which the next plan reports and the next apply finishes.

Run only by an operator through the door (``unit start``); no timer or path unit starts it.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import secrets
import signal
import stat
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .owner_approved_deletion import _instant, _measure, _open_directory, _publish, _read_json, _remove_aside

PLAN_SCHEMA = "scene_retirement_runtime_prune_plan.v1"
RECEIPT_SCHEMA = "scene_retirement_runtime_prune_receipt.v1"
RESULT_SCHEMA = "scene_retirement_runtime_prune_result.v1"
APPROVAL_SCHEMA = "scene_retirement_runtime_prune_approval.v1"
APPLY_ACK = "prune-old-scene-retirement-runtimes"
DEFAULT_RUNTIME_ROOT = Path("/mnt/blueprint-work/scene-retirement-runtime")
DEFAULT_BOOT_ROOT = Path("/usr/lib/blueprint/scene-retirement-runtime")
DEFAULT_RELEASE_LINK = Path("/opt/blueprint/task-evaluation-control-plane")
DEFAULT_APPROVAL_DIR = DEFAULT_RELEASE_LINK / "deploy" / "scene-retirement-runtime-prune-approvals"
DEFAULT_REPORT_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/storage-gc/scene-retirement-runtime-prune")
DEFAULT_MINIMUM_AGE_DAYS = 7
MINIMUM_AGE_FLOOR_DAYS = 2
KEEP_PREVIOUS = 3
#: An approved plan may be applied for this long after it was written; each item is re-checked anyway.
MAX_PLAN_AGE_SECONDS = 7 * 86400
MAX_APPROVAL_SECONDS = 14 * 86400
MAX_ITEMS = 4096
MAX_COHORT_HISTORY = 512
ASIDE_PREFIX = ".scene-retirement-runtime-prune-"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_APPROVAL_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,99}\Z")
_COMPARED = ("dev", "ino", "newest_mtime_epoch", "apparent_bytes", "file_count", "directory_count")
_GENERATIONS = ("source_generation", "sdk_generation")
UNKNOWN_SELECTION = "bootstrap_process_without_selection_marker"

#: kind -> (which root, parent directory under it, name pattern, the companion record's path)
KINDS: dict[str, tuple[str, str, re.Pattern[str], Callable[[Path, str], Path] | None]] = {
    "source_generation": ("runtime", "generations", _HEX64,
                          lambda root, name: root / "manifests" / f"source-{name}.json"),
    "sdk_generation": ("runtime", "sdk-generations", _HEX64,
                       lambda root, name: root / "manifests" / f"sdk-{name}.json"),
    "release_input": ("inputs", "release-inputs", _HEX40,
                      lambda root, name: root / "release-inputs" / f"{name}.claim.json"),
    "sdk_input": ("inputs", "sdk-inputs", _HEX64,
                  lambda root, name: root / "manifests" / f"sdk-input-{name}.json"),
    "wheel_artifact": ("inputs", "wheel-artifacts", _HEX64, None),
}


class RuntimePruneError(ValueError):
    """A typed refusal of the whole plan or apply; it never carries file contents."""


def _require(condition: Any, code: str) -> None:
    if not condition:
        raise RuntimePruneError("scene_retirement_runtime_prune_" + code)


def inputs_root_for(runtime_root: Path) -> Path:
    """The installer's input cache beside the runtime root (``_sdk_root``)."""
    return runtime_root.with_name(runtime_root.name + "-sdk-inputs")


def _parent_of(kind: str, runtime_root: Path) -> Path:
    base, parent_name, _, _ = KINDS[kind]
    return (runtime_root if base == "runtime" else inputs_root_for(runtime_root)) / parent_name


# --- what is current, and what was ------------------------------------------------------------


def _cohort(value: Any, runtime_root: Path) -> dict[str, Any]:
    _require(isinstance(value, dict) and value.get("schema") == "scene-retirement-runtime-cohort.v1",
             "current_cohort_unreadable")
    source, dependency = value.get("source_digest"), value.get("dependency_digest")
    _require(isinstance(source, str) and _HEX64.fullmatch(source) and isinstance(dependency, str)
             and _HEX64.fullmatch(dependency), "current_cohort_unreadable")
    _require(value.get("runtime_root") == str(runtime_root / "generations" / source), "current_cohort_unreadable")
    sdk = value.get("dependencies_root")
    _require(sdk in {str(runtime_root / "dependencies"), str(runtime_root / "sdk-generations" / dependency)},
             "current_cohort_unreadable")
    previous = value.get("previous")
    previous_sha = (previous.get("sha256") if isinstance(previous, dict) else None)
    return {"source_generation": source,
            "sdk_generation": dependency if sdk != str(runtime_root / "dependencies") else None,
            "runtime_root": value["runtime_root"], "dependencies_root": sdk,
            "previous_sha256": previous_sha if isinstance(previous_sha, str) else None}


def _current_raw(boot_root: Path) -> bytes:
    path = Path(boot_root) / "CURRENT.json"
    _require(path.is_file() and not path.is_symlink(), "current_cohort_missing")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RuntimePruneError("scene_retirement_runtime_prune_current_cohort_unreadable") from exc
    _require(0 < len(raw) <= 4096, "current_cohort_unreadable")
    return raw


def read_current(boot_root: Path, runtime_root: Path) -> dict[str, Any]:
    """The current cohort as ``CURRENT.json`` names it, with the same checks the bootstrap applies."""
    raw = _current_raw(boot_root)
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise RuntimePruneError("scene_retirement_runtime_prune_current_cohort_unreadable") from exc
    import hashlib
    return {**_cohort(value, Path(runtime_root)), "sha256": "sha256:" + hashlib.sha256(raw).hexdigest()}


def cohort_history(*, boot_root: Path, runtime_root: Path, report_root: Path,
                   record: bool = True) -> dict[str, Any]:
    """The current cohort and every earlier one its ``previous`` chain reaches through retained copies.

    A copy of each cohort record read is kept under ``<report root>/cohorts/<hex>.json`` (and one the
    installer may keep under ``<boot root>/cohorts`` is read too), so later runs can walk back past it.
    """
    import hashlib
    raw = _current_raw(boot_root)
    store = Path(report_root) / "cohorts"
    if record:
        store.mkdir(parents=True, exist_ok=True)
        copy = store / (hashlib.sha256(raw).hexdigest() + ".json")
        if not copy.exists():
            temporary = copy.with_name(f".{copy.name}.{secrets.token_hex(6)}")
            temporary.write_bytes(raw)
            os.replace(temporary, copy)
    chain, seen, data = [], set(), raw
    ends_at = "chain_complete"
    while True:
        digest = hashlib.sha256(data).hexdigest()
        if digest in seen or len(chain) >= MAX_COHORT_HISTORY:
            ends_at = "cycle_or_limit"
            break
        seen.add(digest)
        try:
            cohort = _cohort(json.loads(data), Path(runtime_root))
        except (ValueError, RuntimePruneError):
            ends_at = "record_not_a_cohort"  # The first installation's record (installation.json).
            break
        chain.append({"sha256": "sha256:" + digest, **{key: cohort[key] for key in _GENERATIONS}})
        previous = cohort["previous_sha256"]
        if not (isinstance(previous, str) and previous.startswith("sha256:") and _HEX64.fullmatch(previous[7:])):
            break
        data = None
        for directory in (store, Path(boot_root) / "cohorts"):
            candidate = directory / (previous[7:] + ".json")
            try:
                body = candidate.read_bytes() if candidate.is_file() and not candidate.is_symlink() else None
            except OSError:
                body = None
            if body is not None and hashlib.sha256(body).hexdigest() == previous[7:]:
                data = body
                break
        if data is None:
            ends_at = "previous_record_not_retained:" + previous
            break
    return {"cohorts": chain, "ends_at": ends_at}


def active_release_commit(release_link: Path) -> str:
    """The commit of the active per-commit release (the link's target is named by its commit)."""
    try:
        name = Path(os.readlink(release_link)).name
    except OSError as exc:
        raise RuntimePruneError("scene_retirement_runtime_prune_active_release_unreadable") from exc
    _require(_HEX40.fullmatch(name), "active_release_unreadable")
    return name


# --- who is running what --------------------------------------------------------------------


class ProcessIndex:
    """One read of the process table: what each live process names or holds, and each bootstrap process.

    It reads each process's command line, environment, working directory, executable, open
    descriptors and memory maps, and keeps only what it matches against. An entry that cannot be
    read (other than a process that exited) proves nothing, so every path then counts as in use.
    A bootstrap process is one whose command line names ``<boot root>/continuous_bootstrap.py``; its
    selection is the runtime directories it holds open (``_hold_selection``).
    """

    def __init__(self, *, process_root: Path = Path("/proc"), boot_root: Path, runtime_root: Path,
                 ignored_process_ids: Sequence[int] = ()) -> None:
        self.unreadable = False
        self.blobs: list[bytes] = []
        self.targets: list[str] = []
        self.bootstraps: list[dict[str, Any]] = []
        process_root, runtime_root = Path(process_root), Path(runtime_root)
        if not process_root.is_dir():
            raise RuntimePruneError("scene_retirement_runtime_prune_process_inventory_unavailable")
        boot_marker = str(Path(boot_root) / "continuous_bootstrap.py").encode()
        boot_epoch = _boot_epoch(process_root)
        ticks = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        try:
            processes = list(process_root.iterdir())
        except OSError:
            self.unreadable = True
            return
        exited = (FileNotFoundError, ProcessLookupError)
        for process in processes:
            if not process.name.isdigit() or int(process.name) in ignored_process_ids:
                continue
            own: list[bytes] = []
            for name in ("cmdline", "environ"):
                try:
                    own.append((process / name).read_bytes())
                except exited:
                    continue
                except OSError:
                    self.unreadable = True
            self.blobs.extend(own)
            held: list[str] = []
            descriptor_targets: list[str] = []
            try:
                maps = (process / "maps").read_text(encoding="utf-8", errors="replace")
            except exited:
                maps = ""
            except OSError:
                self.unreadable, maps = True, ""
            for line in maps.splitlines():
                fields = line.split(None, 5)
                if len(fields) == 6 and fields[5].startswith("/"):
                    held.append(fields[5].removesuffix(" (deleted)"))
            try:
                descriptors = list((process / "fd").iterdir())
            except exited:
                descriptors = []
            except OSError:
                self.unreadable, descriptors = True, []
            for link in (process / "cwd", process / "exe", *descriptors):
                try:
                    target = os.readlink(link).removesuffix(" (deleted)")
                except exited:
                    continue
                except OSError:
                    self.unreadable = True
                    continue
                held.append(target)
                if link.parent.name == "fd":
                    descriptor_targets.append(target)
            self.targets.extend(held)
            if own and boot_marker in own[0]:
                # Known only by a descriptor on the selected directory itself (``_hold_selection``): a
                # mapped library or an open file under a generation says nothing about which source
                # generation the process imports from, and an SDK directory nothing about its source.
                generations = {kind: sorted({Path(target).name for target in descriptor_targets
                                             if Path(target).parent == _parent_of(kind, runtime_root)})
                               for kind in _GENERATIONS}
                initial = str(runtime_root) in descriptor_targets  # The first installation, at the root.
                source_known = initial or bool(generations["source_generation"])
                sdk_known = (initial or bool(generations["sdk_generation"])
                             or str(runtime_root / "dependencies") in descriptor_targets)
                self.bootstraps.append({"pid": int(process.name),
                                        "started_at_epoch": _process_start(process, boot_epoch, ticks),
                                        "source_selection_known": source_known,
                                        "sdk_selection_known": sdk_known, **generations})

    def unknown(self, kind: str) -> int:
        """Bootstrap processes whose ``kind`` generation cannot be known (no descriptor on one)."""
        field = "source_selection_known" if kind == "source_generation" else "sdk_selection_known"
        return sum(1 for row in self.bootstraps if not row[field])

    @property
    def unknown_selections(self) -> int:
        """Bootstrap processes whose source or SDK generation cannot be known."""
        return sum(1 for row in self.bootstraps
                   if not (row["source_selection_known"] and row["sdk_selection_known"]))

    def uses(self, path: Path | str) -> bool:
        needle = str(path)
        return (self.unreadable or any(needle.encode() in blob for blob in self.blobs)
                or any(target == needle or target.startswith(needle + "/") for target in self.targets))


def _boot_epoch(process_root: Path) -> float | None:
    try:
        for line in (process_root / "stat").read_text().splitlines():
            if line.startswith("btime "):
                return float(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def _process_start(process: Path, boot_epoch: float | None, ticks: int) -> float | None:
    """When the process started (epoch seconds), from ``stat`` field 22, or None when it cannot be read."""
    if boot_epoch is None:
        return None
    try:
        text = (process / "stat").read_text()
        fields = text[text.rindex(")") + 2:].split()
        return boot_epoch + int(fields[19]) / ticks
    except (OSError, ValueError, IndexError):
        return None


# --- the plan -------------------------------------------------------------------------------


def _companion_identity(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return {"path": str(path), "kind": "not_a_file"}
    return {"path": str(path), "dev": info.st_dev, "ino": info.st_ino, "size": info.st_size,
            "mtime_ns": info.st_mtime_ns}


def _items(runtime_root: Path, inputs_root: Path) -> list[dict[str, Any]]:
    """Every listable item, measured, with its creation time; nothing else under the roots is ever listed."""
    rows: list[dict[str, Any]] = []
    for kind, (base, parent_name, pattern, companion) in KINDS.items():
        root = runtime_root if base == "runtime" else inputs_root
        parent = root / parent_name
        if not parent.is_dir() or parent.is_symlink():
            continue
        parent_fd = _open_directory(parent)
        try:
            for name in sorted(os.listdir(parent_fd)):
                if not pattern.fullmatch(name):
                    continue
                identity = _measure(parent_fd, name)
                record = _companion_identity(companion(root, name) if companion is not None else None)
                if isinstance(identity, str):
                    created = None
                else:
                    created = (record["mtime_ns"] / 1e9 if record and "mtime_ns" in record
                               else float(os.lstat(name, dir_fd=parent_fd).st_mtime))
                rows.append({"kind": kind, "name": name, "path": str(parent / name),
                             "companion": record, "created_at_epoch": created,
                             "identity": identity if isinstance(identity, dict) else None,
                             "unmeasurable": identity if isinstance(identity, str) else None})
                _require(len(rows) <= MAX_ITEMS, "too_many_items")
        finally:
            os.close(parent_fd)
    return rows


def leftover_asides(runtime_root: Path, report_root: Path) -> list[dict[str, Any]]:
    """Trees an interrupted apply renamed aside and never removed, each with the receipt that names it."""
    receipts: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    directory = Path(report_root) / "receipts"
    if directory.is_dir() and not directory.is_symlink():
        for path in sorted(directory.glob("*.json")):
            receipt = _read_json(path)
            if receipt and receipt.get("schema_version") == RECEIPT_SCHEMA and isinstance(receipt.get("aside_name"), str):
                receipts[(str(Path(receipt.get("path", "")).parent), receipt["aside_name"])] = (path, receipt)
    found = []
    for kind in KINDS:
        parent = _parent_of(kind, runtime_root)
        if not parent.is_dir() or parent.is_symlink():
            continue
        for name in sorted(os.listdir(parent)):
            if not name.startswith(ASIDE_PREFIX):
                continue
            match = receipts.get((str(parent), name))
            found.append({"kind": kind, "path": str(parent / name), "aside_name": name,
                          "receipt": str(match[0]) if match else None,
                          "receipt_status": match[1].get("status") if match else None})
    return found


def _lock_hashes(release_input: Path) -> set[str]:
    """The sha256 hashes a release input's ``uv.lock`` names (the wheels its build may fetch)."""
    try:
        text = (release_input / "uv.lock").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    return set(re.findall(r"sha256:([0-9a-f]{64})", text))


def _keep_reasons(rows: list[dict[str, Any]], *, current: Mapping[str, Any], history: Mapping[str, Any],
                  commit: str, processes: ProcessIndex, minimum_age_seconds: int, observed: float) -> None:
    """Set each row's ``reasons`` to keep it, from the present state. The plan and every apply step use this.

    Rows without a measured identity (the apply's cheap listing) are judged on what needs no tree
    walk; the item about to be removed is measured where it is removed (``_remove``).
    """
    named = {kind: {cohort[kind] for cohort in history["cohorts"] if cohort[kind]} for kind in _GENERATIONS}
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        row["reasons"] = []
        by_kind.setdefault(row["kind"], []).append(row)
    currents = {"source_generation": current["source_generation"], "sdk_generation": current["sdk_generation"],
                "release_input": commit}
    for kind, members in by_kind.items():
        here = currents.get(kind)
        for row in members:
            if row["name"] == here:
                row["reasons"].append("current")
            elif row["name"] in named.get(kind, ()):
                row["reasons"].append("named_by_cohort_history")
            if kind in _GENERATIONS and processes.unknown(kind):
                row["reasons"].append(UNKNOWN_SELECTION)
        if kind == "sdk_input":
            # Build inputs, never run: the newest stands for the current cohort's own, then the three before it.
            ordered = sorted((row for row in members if row["created_at_epoch"] is not None),
                             key=lambda row: row["created_at_epoch"], reverse=True)
            for row in ordered[:1 + KEEP_PREVIOUS]:
                row["reasons"].append("current_or_previous_three")
        elif kind != "wheel_artifact":  # A wheel is kept by the release locks that name it (below).
            ordered = sorted((row for row in members if row["created_at_epoch"] is not None and row["name"] != here),
                             key=lambda row: row["created_at_epoch"], reverse=True)
            for row in ordered[:KEEP_PREVIOUS]:
                row["reasons"].append("previous_three")
    for row in rows:
        if row.get("unmeasurable"):
            row["reasons"].append("unmeasurable:" + row["unmeasurable"])
        if (row["created_at_epoch"] is None or observed - row["created_at_epoch"] < minimum_age_seconds
                or (row.get("identity") is not None
                    and observed - row["identity"]["newest_mtime_epoch"] < minimum_age_seconds)):
            row["reasons"].append("newer_than_minimum_age")
        if processes.uses(row["path"]) or (row["companion"] and processes.uses(row["companion"]["path"])):
            row["reasons"].append("in_use")
    kept_locks: set[str] = set()
    for row in by_kind.get("release_input", []):
        if row["reasons"]:
            kept_locks |= _lock_hashes(Path(row["path"]))
    for row in by_kind.get("wheel_artifact", []):
        if row["name"] in kept_locks:
            row["reasons"].append("named_by_a_kept_release_lock")


def _listing(runtime_root: Path) -> list[dict[str, Any]]:
    """The listable items as they stand now, with creation times and records but no tree walk."""
    rows = []
    for kind, (base, _parent_name, pattern, companion) in KINDS.items():
        root = runtime_root if base == "runtime" else inputs_root_for(runtime_root)
        parent = _parent_of(kind, runtime_root)
        if not parent.is_dir() or parent.is_symlink():
            continue
        for name in sorted(os.listdir(parent)):
            if not pattern.fullmatch(name):
                continue
            record = _companion_identity(companion(root, name) if companion is not None else None)
            try:
                created = (record["mtime_ns"] / 1e9 if record and "mtime_ns" in record
                           else float(os.lstat(parent / name).st_mtime))
            except OSError:
                created = None
            rows.append({"kind": kind, "name": name, "path": str(parent / name), "companion": record,
                         "created_at_epoch": created})
    return rows


def build_plan(*, runtime_root: Path, boot_root: Path, release_link: Path, minimum_age_seconds: int,
               report_root: Path, process_root: Path = Path("/proc"), now: float | None = None) -> dict[str, Any]:
    """Every listable item with ``remove`` or ``keep`` and why. Changes nothing but its own cohort copies."""
    _require(type(minimum_age_seconds) is int and minimum_age_seconds >= MINIMUM_AGE_FLOOR_DAYS * 86400,
             "minimum_age_too_short")
    observed = float(time.time() if now is None else now)
    runtime_root, boot_root = Path(runtime_root), Path(boot_root)
    inputs_root = inputs_root_for(runtime_root)
    current = read_current(boot_root, runtime_root)
    history = cohort_history(boot_root=boot_root, runtime_root=runtime_root, report_root=report_root)
    commit = active_release_commit(Path(release_link))
    processes = ProcessIndex(process_root=process_root, boot_root=boot_root, runtime_root=runtime_root,
                             ignored_process_ids=(os.getpid(),))
    rows = _items(runtime_root, inputs_root)
    _keep_reasons(rows, current=current, history=history, commit=commit, processes=processes,
                  minimum_age_seconds=minimum_age_seconds, observed=observed)
    for row in rows:
        row["disposition"] = "keep" if row["reasons"] else "remove"
    totals: dict[str, dict[str, int]] = {}
    for row in rows:
        entry = totals.setdefault(row["kind"], {"keep_count": 0, "remove_count": 0,
                                                "keep_allocated_bytes": 0, "remove_allocated_bytes": 0})
        entry[f"{row['disposition']}_count"] += 1
        entry[f"{row['disposition']}_allocated_bytes"] += row["identity"]["allocated_bytes"] if row["identity"] else 0
    removing = {(row["kind"], row["name"]) for row in rows if row["disposition"] == "remove"}
    asides = leftover_asides(runtime_root, report_root)
    checks = {
        # What the first host run must be read for, before any approval.
        "process_inventory_readable": not processes.unreadable,
        "running_bootstrap_processes": len(processes.bootstraps),
        "bootstrap_processes_without_selection_marker": processes.unknown_selections,
        "bootstrap_processes_with_unknown_source_generation": processes.unknown("source_generation"),
        "bootstrap_processes": [{**row, "running_generation_planned_for_removal": sorted(
            f"{kind}:{name}" for kind in _GENERATIONS for name in row[kind] if (kind, name) in removing)}
            for row in processes.bootstraps],
        "leftover_aside_trees": asides,
    }
    plan = {
        "schema_version": PLAN_SCHEMA, "observed_at_epoch": observed, "runtime_root": str(runtime_root),
        "inputs_root": str(inputs_root), "boot_root": str(boot_root), "current": current,
        "cohort_history": history, "active_release_commit": commit, "minimum_age_seconds": minimum_age_seconds,
        "keep_previous": KEEP_PREVIOUS, "first_host_run_checks": checks,
        "process_inventory_unreadable": processes.unreadable,
        "running_bootstrap_processes": len(processes.bootstraps),
        "rows": rows, "totals": totals,
        "remove_count": len(removing),
        "remove_allocated_bytes": sum(row["identity"]["allocated_bytes"] for row in rows
                                      if row["disposition"] == "remove"),
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def write_plan(report_root: Path, **kwargs: Any) -> dict[str, Any]:
    plan = build_plan(report_root=report_root, **kwargs)
    _publish(Path(report_root) / "plans" / (plan["plan_digest"].removeprefix("sha256:") + ".json"), plan)
    _publish(Path(report_root) / "plan.json", plan)
    return plan


# --- the owner's approval -------------------------------------------------------------------


def load_approval(approval_dir: Path, *, now: float) -> dict[str, Any]:
    """The one current owner approval: who approved, the evidence, when, until when, and the plan digest."""
    directory = Path(approval_dir)
    _require(directory.is_dir() and not directory.is_symlink(), "owner_approval_missing")
    found = []
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith(".") or path.is_symlink():
            continue
        try:
            value = json.loads(path.read_bytes())
        except (OSError, ValueError):
            raise RuntimePruneError("scene_retirement_runtime_prune_owner_approval_unreadable") from None
        _require(isinstance(value, dict) and set(value) == {
            "schema_version", "approval_id", "approved_by", "approval_evidence", "approved_at", "expires_at",
            "plan_digest"} and value["schema_version"] == APPROVAL_SCHEMA, "owner_approval_fields_invalid")
        _require(isinstance(value["approval_id"], str) and _APPROVAL_ID.fullmatch(value["approval_id"])
                 and path.name == value["approval_id"] + ".json", "owner_approval_id_invalid")
        for key, limit in (("approved_by", 200), ("approval_evidence", 1000)):
            text = value[key]
            _require(isinstance(text, str) and 0 < len(text) <= limit and text.isprintable(), key + "_invalid")
        _require(isinstance(value["plan_digest"], str) and value["plan_digest"].startswith("sha256:")
                 and _HEX64.fullmatch(value["plan_digest"][7:]), "owner_approval_plan_digest_invalid")
        try:
            approved = _instant(value["approved_at"], "approved_at_invalid")
            expires = _instant(value["expires_at"], "expires_at_invalid")
        except ValueError as exc:
            raise RuntimePruneError("scene_retirement_runtime_prune_owner_approval_window_invalid") from exc
        _require(approved < expires <= approved + MAX_APPROVAL_SECONDS, "owner_approval_window_invalid")
        if approved <= now < expires:
            found.append(value)
    _require(found, "owner_approval_missing")
    _require(len(found) == 1, "owner_approval_ambiguous")
    return found[0]


# --- the apply ------------------------------------------------------------------------------


class _InstallerLock:
    """The installer's own exclusive lock, held for one item: a deploy that holds it stops the run."""

    def __init__(self, boot_root: Path) -> None:
        self.path, self.fd = Path(boot_root) / "installation.lock", None

    def __enter__(self) -> bool:
        _require(self.path.is_file() and not self.path.is_symlink(), "installer_lock_missing")
        self.fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self.fd)
            self.fd = None
            return False
        return True

    def __exit__(self, *_exc: object) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def _receipt(plan_digest: str, row: Mapping[str, Any], *, status: str, aside: str,
             observed_at: float, **extra: Any) -> dict[str, Any]:
    receipt = {"schema_version": RECEIPT_SCHEMA, "status": status, "kind": row["kind"], "path": row["path"],
               "companion": row["companion"], "identity": row["identity"], "plan_digest": plan_digest,
               "aside_name": aside, "observed_at_epoch": observed_at, **extra}
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def _receipt_path(report_root: Path, row: Mapping[str, Any]) -> Path:
    return Path(report_root) / "receipts" / f"{row['kind']}-{Path(row['path']).name}.json"


def _finish_aside(leftover: Mapping[str, Any], *, report_root: Path, now: Callable[[], float]) -> dict[str, Any]:
    """Finish one tree an interrupted run renamed aside, from its receipt; one with no receipt is left."""
    if leftover["receipt"] is None:
        return {"path": leftover["path"], "outcome": "aside_left:no_receipt"}
    receipt = _read_json(Path(leftover["receipt"]))
    identity = receipt.get("identity") if receipt else None
    if not receipt or receipt.get("status") != "removing" or not isinstance(identity, dict):
        return {"path": leftover["path"], "outcome": "aside_left:receipt_not_removing"}
    parent = Path(leftover["path"]).parent
    failure = _remove_aside(parent, leftover["aside_name"], identity)
    if failure is not None:
        return {"path": leftover["path"], "outcome": "aside_left:" + failure}
    companion = receipt.get("companion")
    if isinstance(companion, dict) and isinstance(companion.get("path"), str):
        try:
            os.unlink(companion["path"])
        except FileNotFoundError:
            pass
    removed_at = float(now())
    finished = {key: value for key, value in receipt.items() if key not in {"receipt_digest"}}
    finished.update(status="removed", observed_at_epoch=removed_at, removed_at_epoch=removed_at,
                    finished_after_interruption=True)
    finished["receipt_digest"] = canonical_digest(finished, digest_field="receipt_digest")
    _publish(Path(leftover["receipt"]), finished)
    return {"path": receipt["path"], "outcome": "removed_after_interruption",
            "allocated_bytes": identity.get("allocated_bytes", 0)}


def apply_plan(*, report_root: Path, ack: str, approval_dir: Path, process_root: Path = Path("/proc"),
               release_link: Path = DEFAULT_RELEASE_LINK,
               now: Callable[[], float] = time.time, should_stop: Callable[[], bool] = lambda: False
               ) -> dict[str, Any]:
    """Remove exactly the rows the owner-approved plan marked ``remove``, each re-checked at the moment of removal."""
    _require(ack == APPLY_ACK, "apply_not_acknowledged")
    report_root = Path(report_root)
    observed = float(now())
    approval = load_approval(approval_dir, now=observed)
    plan = _read_json(report_root / "plans" / (approval["plan_digest"].removeprefix("sha256:") + ".json"))
    _require(plan is not None and plan.get("schema_version") == PLAN_SCHEMA
             and plan.get("plan_digest") == canonical_digest(plan, digest_field="plan_digest"), "plan_invalid")
    _require(plan["plan_digest"] == approval["plan_digest"], "owner_approval_names_another_plan")
    _require(0 <= observed - float(plan["observed_at_epoch"]) <= MAX_PLAN_AGE_SECONDS, "plan_stale")
    runtime_root, boot_root = Path(plan["runtime_root"]), Path(plan["boot_root"])
    _require(Path(plan["inputs_root"]) == inputs_root_for(runtime_root), "plan_invalid")
    outcomes: list[dict[str, Any]] = [_finish_aside(leftover, report_root=report_root, now=now)
                                      for leftover in leftover_asides(runtime_root, report_root)]
    stopped = None
    pending = [row for row in plan["rows"] if row["disposition"] == "remove"]
    for index, row in enumerate(pending):
        if should_stop():
            stopped = "stop_requested"
        else:
            lock = _InstallerLock(boot_root)
            if not lock.__enter__():
                stopped = "installer_busy"  # A deploy holds the lock: stop rather than make it wait.
            else:
                try:
                    outcomes.append(_remove(plan, row, report_root=report_root, process_root=process_root,
                                            release_link=release_link, now=now))
                except RuntimePruneError as error:
                    stopped = str(error).removeprefix("scene_retirement_runtime_prune_")
                finally:
                    lock.__exit__()
        if stopped:
            outcomes.extend({"path": rest["path"], "outcome": "not_attempted:" + stopped} for rest in pending[index:])
            break
    result = {"schema_version": RESULT_SCHEMA, "plan_digest": plan["plan_digest"],
              "approval_id": approval["approval_id"], "approved_by": approval["approved_by"],
              "applied_at_epoch": observed, "stopped": stopped,
              "removed_count": sum(1 for row in outcomes if row["outcome"].startswith("removed")),
              "removed_allocated_bytes": sum(row.get("allocated_bytes", 0) for row in outcomes
                                             if row["outcome"].startswith("removed")),
              "outcome_counts": _counts(outcomes), "outcomes": outcomes}
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    _publish(report_root / f"apply-{int(observed)}-{secrets.token_hex(3)}.json", result)
    return result


def _counts(outcomes: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in outcomes:
        counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
    return dict(sorted(counts.items()))


def _remove(plan: Mapping[str, Any], row: Mapping[str, Any], *, report_root: Path, process_root: Path,
            release_link: Path, now: Callable[[], float]) -> dict[str, Any]:
    path = Path(row["path"])
    planned = row["identity"]
    kind = KINDS.get(row["kind"])
    runtime_root = Path(plan["runtime_root"])
    expected_parent = None if kind is None else _parent_of(row["kind"], runtime_root)
    companion_rule = None if kind is None or kind[3] is None else kind[3](expected_parent.parent, path.name)
    if (planned is None or kind is None or path.parent != expected_parent or not kind[2].fullmatch(path.name)
            or (row["companion"] or {}).get("path", str(companion_rule)) != str(companion_rule)):
        return {"path": row["path"], "outcome": "skipped", "reasons": ["plan_row_invalid"]}
    # At the moment of removal the whole keep set is derived again from the present state (the current
    # cohort and its history, the active release, the three most recent, age, use, unknown processes):
    # an approved item that has since become one to keep is skipped. A deploy since the plan is no stop.
    boot_root = Path(plan["boot_root"])
    current = read_current(boot_root, runtime_root)
    history = cohort_history(boot_root=boot_root, runtime_root=runtime_root, report_root=report_root)
    processes = ProcessIndex(process_root=process_root, boot_root=boot_root, runtime_root=runtime_root,
                             ignored_process_ids=(os.getpid(),))
    present = _listing(runtime_root)
    observed = float(now())
    _keep_reasons(present, current=current, history=history, commit=active_release_commit(Path(release_link)),
                  processes=processes, minimum_age_seconds=int(plan["minimum_age_seconds"]), observed=observed)
    mine = next((item for item in present if item["kind"] == row["kind"] and item["path"] == row["path"]), None)
    if mine is None:
        return {"path": row["path"], "outcome": "skipped", "reasons": ["absent"]}
    if mine["reasons"]:
        return {"path": row["path"], "outcome": "skipped", "reasons": ["now_kept", *mine["reasons"]]}
    companion = row["companion"]
    receipt_path = _receipt_path(report_root, row)
    parent_fd = _open_directory(path.parent)
    try:
        identity = _measure(parent_fd, path.name)
        if isinstance(identity, str) or any(identity[key] != planned[key] for key in _COMPARED):
            return {"path": row["path"], "outcome": "skipped",
                    "reasons": [identity if isinstance(identity, str) else "changed_since_plan"]}
        if observed - identity["newest_mtime_epoch"] < int(plan["minimum_age_seconds"]):
            return {"path": row["path"], "outcome": "skipped", "reasons": ["now_kept", "newer_than_minimum_age"]}
        if _companion_identity(Path(companion["path"]) if companion else None) != companion:
            return {"path": row["path"], "outcome": "skipped", "reasons": ["companion_changed_since_plan"]}
        aside = f"{ASIDE_PREFIX}{path.name[:16]}-{secrets.token_hex(6)}"
        _publish(receipt_path, _receipt(plan["plan_digest"], {**row, "identity": identity}, status="removing",
                                        aside=aside, observed_at=float(now())))
        os.rename(path.name, aside, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        moved = os.lstat(aside, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)
    if (moved.st_dev, moved.st_ino) != (identity["dev"], identity["ino"]):
        return {"path": row["path"], "outcome": "aside_changed"}
    failure = _remove_aside(path.parent, aside, identity)
    if failure is not None:
        return {"path": row["path"], "outcome": failure}
    # The directory first, then its record: the installer refuses a generation whose record is gone
    # while the directory remains, but recreates both when both are absent.
    if companion:
        try:
            os.unlink(companion["path"])
        except FileNotFoundError:
            pass
    removed_at = float(now())
    _publish(receipt_path, _receipt(plan["plan_digest"], {**row, "identity": identity}, status="removed",
                                    aside=aside, observed_at=removed_at, removed_at_epoch=removed_at))
    return {"path": row["path"], "outcome": "removed", "allocated_bytes": identity["allocated_bytes"]}


# --- command line ---------------------------------------------------------------------------


def _summary(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {key: plan[key] for key in ("plan_digest", "remove_count", "remove_allocated_bytes", "totals",
                                        "first_host_run_checks")}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("plan", "apply"))
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--boot-root", type=Path, default=DEFAULT_BOOT_ROOT)
    parser.add_argument("--release-link", type=Path, default=DEFAULT_RELEASE_LINK)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--approval-dir", type=Path, default=DEFAULT_APPROVAL_DIR)
    parser.add_argument("--minimum-age-days", type=int, default=DEFAULT_MINIMUM_AGE_DAYS)
    parser.add_argument("--ack", default="")
    args = parser.parse_args(argv)
    stop = {"requested": False}

    def request_stop(*_args: Any) -> None:
        stop["requested"] = True  # Finish the item in hand, then stop (a deploy stops this unit).
    signal.signal(signal.SIGTERM, request_stop)
    try:
        if args.action == "plan":
            plan = write_plan(args.report_root, runtime_root=args.runtime_root, boot_root=args.boot_root,
                              release_link=args.release_link, minimum_age_seconds=args.minimum_age_days * 86400)
            print(json.dumps({"action": "plan", **_summary(plan)}, indent=2, sort_keys=True))
        else:
            result = apply_plan(report_root=args.report_root, ack=args.ack, approval_dir=args.approval_dir,
                                release_link=args.release_link, should_stop=lambda: stop["requested"])
            print(json.dumps({"action": "apply", **{key: result[key] for key in (
                "plan_digest", "result_digest", "approval_id", "stopped", "removed_count",
                "removed_allocated_bytes", "outcome_counts")}}, indent=2, sort_keys=True))
    except RuntimePruneError as error:
        print(json.dumps({"action": args.action, "error": str(error)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
