"""The compiled team scene halves of one withdrawn site, and the records that point at them.

Beta journey step: site withdrawal after the site's scene compiled its robot-team half
(``team_scene_packet_worker``). A half holds collision and appearance data derived from
the capture. The owner approved deleting a withdrawn site's halves (2026-11-06). Every
record that points at them goes in the same deletion, so no reader is left naming bytes
that are gone:

- each half directory ``<halves root>/<capture key>`` bound to the site, with every
  attempt under it, the remote lane's landings and its ``native-g1/`` packets. A half is
  bound when the team worker files one of the site's captures under it
  (``team_scene_capture_root.capture_key``), or when its own JSON records name the site's
  scene id, one of its capture ids or one of its scene intents;
- the team registry's derived-profile rows in ``derived_profiles.root`` of the registry
  file named here (the one the units read): each ``<capture key>.json`` of a planned half,
  its ``.tmp`` staging, and any row whose own bytes name the site or a planned half. The
  registry's own operator rows are root-owned and are not written here; the receipt names
  each one that names the site;
- the team scene packet queue's records for the site: a request key is the site's when its
  request or its status names the site or a planned half, and every file of it goes, in
  ``pending/``, ``processed/``, ``holds/`` (with its ``settled/`` and ``superseded/``) and
  ``status/``, with any write temporary that names either;
- the rigid robot-free probe queue's ``pending/`` and ``done/`` entry for each planned
  half's capture key. The probe's jobs (``jobs/<capture key>-*``) keep sealed bundles built
  from the half's packet and the website-grant hold records their settlement reads. This
  deletion does not remove them; the receipt names each one, and the deletion is not
  complete while any remain.

These fail closed into ``unresolved``: a half, row or record that also names another site;
a link, a special file or an unreadable entry; a registry that is not operator-owned, not
sealed or not readable; and a request still in ``processing/``. That last one is a compile
in flight, possibly on the remote lane, which could land a new half.

``plan_team_scene`` reads only. A half's inventory leaves out the ``native-g1/`` namespaces
``website_capture_withdrawal_native_g1`` plans, because that module removes them first.
``apply_team_scene`` runs while the deletion holds the packet queue's lock
(``hold_packet_queue``; ``team_scene_packets_busy`` while a worker run holds it). It checks
every record's identity and every half's whole inventory again. Then it removes the
records that point at the halves, and the halves last. Records here carry paths, sizes,
inode numbers and counts only, never what a file holds.
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import stat
from collections.abc import Iterable, Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .rigid_robot_free_probe_queue_root import DEFAULT_QUEUE as _PROBE_QUEUE
from .rigid_robot_free_probe_queue_root import ENTRY_DIRECTORIES, JOBS_DIRECTORY
from .rigid_robot_free_probe_queue_root import QUEUE_ENV as _PROBE_QUEUE_ENV
from .team_candidate_registry_reader import (
    REGISTRY_ENV,
    REGISTRY_SCHEMA,
    TeamCandidateRegistryError,
    _derived_settings,
)
from .team_scene_capture_root import DEFAULT_QUEUE_ROOT as _PACKET_QUEUE
from .team_scene_capture_root import HOLDS_DIRNAME, QUEUE_ROOT_ENV, TEMPORARY_PREFIX, TEMPORARY_SUFFIX

# The units' values, unless set here too: the registry every team unit reads
# (blueprint-embodiment-preparation.service), and the two queues' roots.
DEFAULT_REGISTRY = Path(os.environ.get(REGISTRY_ENV) or "/etc/blueprint/team-evaluation-registry.json")
DEFAULT_PACKET_QUEUE_ROOT = Path(os.environ.get(QUEUE_ROOT_ENV) or _PACKET_QUEUE)
DEFAULT_PROBE_QUEUE_ROOT = Path(os.environ.get(_PROBE_QUEUE_ENV) or _PROBE_QUEUE)
_PREFIX = "website_withdrawal_deletion_"
# A request the worker has claimed is in processing/; these two states and holds/ are at rest.
_AT_REST = ("pending", "processed")


def _unsafe_root(path: Path) -> bool:
    return any(part.is_symlink() for part in (path, *path.parents))


def _scan(paths: Iterable[Path], vocabulary: set[bytes], request_id: str) -> Any:
    """What these files name; raises on a link, special file or unreadable record."""
    from .website_capture_withdrawal_deletion import _Scan, _scan_files

    scan = _Scan(vocabulary, request_id)
    _scan_files(paths, scan)
    return scan


def _names(value: Any, vocabulary: set[bytes], request_id: str) -> Any:
    from .website_capture_withdrawal_deletion import _Bytes, _Scan

    scan = _Scan(vocabulary, request_id)
    scan.feed(_Bytes(json.dumps(value, sort_keys=True).encode()))
    return scan


def _identity(path: Path) -> list[Any]:
    info = os.lstat(path)
    return [str(path), info.st_size, info.st_ino, info.st_mtime_ns]


def _registry(path: Path | None, vocabulary: set[bytes], identity: Mapping[str, str],
              unresolved: list[dict[str, Any]]) -> dict[str, Any]:
    """Where the registry keeps derived rows, and which of its root-owned operator rows name the site."""
    value: dict[str, Any] = {"registry": str(path) if path else None, "registry_state": "not_named",
                             "derived_profile_root": None, "operator_profile_rows": [],
                             "operator_site_screening_row": False}
    if path is None:
        return value
    path = Path(path)
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return {**value, "registry_state": "absent"}
    except OSError:
        info = None
    value["registry_state"] = "unreadable"
    # The registry decides which directory rows are deleted from: one anyone else could write is never read.
    if (info is None or not path.is_absolute() or _unsafe_root(path) or not stat.S_ISREG(info.st_mode)
            or info.st_mode & 0o022 or info.st_uid not in {0, os.getuid()}):
        unresolved.append({"path": str(path), "reason": _PREFIX + "team_registry_not_operator_owned"})
        return value
    try:
        registry = json.loads(path.read_bytes())
        sealed = (isinstance(registry, dict) and registry.get("schema_version") == REGISTRY_SCHEMA
                  and registry.get("registry_digest") == canonical_digest(registry, digest_field="registry_digest")
                  and isinstance(registry.get("profiles"), list))
        settings = _derived_settings(registry) if sealed else None
    except (OSError, ValueError, TeamCandidateRegistryError):
        sealed = False
    if not sealed:
        unresolved.append({"path": str(path), "reason": _PREFIX + "team_registry_invalid"})
        return value
    root = settings["root"] if settings else None
    if root is not None and _unsafe_root(root):
        unresolved.append({"path": str(root), "reason": _PREFIX + "root_unsafe"})
        return value
    site = (registry.get("site_screening_candidates") or {}) if isinstance(
        registry.get("site_screening_candidates"), Mapping) else {}
    return {**value, "registry_state": "read", "derived_profile_root": str(root) if root else None,
            "operator_profile_rows": [index for index, row in enumerate(registry["profiles"])
                                      if _names(row, vocabulary, identity["request_id"]).named],
            "operator_site_screening_row": identity["scene_id"] in site}


def _halves(root: Path, keys: set[str], vocabulary: set[bytes], request_id: str,
            first: Mapping[str, list[str]], unresolved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each half directory bound to the site, inventoried without the G1 namespaces removed before it."""
    from .website_capture_withdrawal_deletion import _SCANNED_SUFFIXES, WithdrawalDeletionError, _inventory

    rows = []
    for half in sorted(root.iterdir()):
        if half.is_symlink():
            unresolved.append({"path": str(half), "reason": _PREFIX + "unsafe_entry"})
            continue
        if not half.is_dir():
            continue  # The root holds only capture-key directories.
        try:
            inventory = _inventory(half)
            scan = _scan((half / row[0] for row in inventory if row[0].endswith(_SCANNED_SUFFIXES)),
                         vocabulary, request_id)
        except (WithdrawalDeletionError, OSError) as error:
            # Unreadable, or holding a link or special file: it could be this site's, so it is never skipped.
            unresolved.append({"path": str(half), "reason": str(error) if isinstance(
                error, WithdrawalDeletionError) else _PREFIX + "record_unreadable"})
            continue
        bound_by = "capture_key" if half.name in keys else "records" if scan.named else None
        if bound_by is None:
            continue  # Another capture's half, and nothing in it names this site.
        if scan.other_sites:
            unresolved.append({"path": str(half), "reason": _PREFIX + "team_scene_half_names_another_site"})
            continue
        removed_first = first.get(half.name, [])
        kept = [row for row in inventory
                if not any(row[0].startswith(namespace + "/") for namespace in removed_first)]
        rows.append({"directory": str(half), "bound_by": bound_by, "native_g1_namespaces_removed_first":
                     list(removed_first), "inventory_digest": canonical_digest({"rows": kept}),
                     "file_count": len(kept)})
    return rows


def _derived_rows(root: Path, keys: set[str], vocabulary: set[bytes], request_id: str,
                  unresolved: list[dict[str, Any]]) -> list[list[Any]]:
    """The planned halves' rows by name, and any row whose own bytes name the site or a planned half."""
    named = {f"{key}{suffix}" for key in keys for suffix in (".json", ".tmp")}
    rows = []
    for child in sorted(root.iterdir()):
        if not child.name.endswith((".json", ".tmp")):
            continue
        try:
            scan = _scan([child], vocabulary, request_id)
        except (ValueError, OSError) as error:
            unresolved.append({"path": str(child), "reason": str(error) if str(error).startswith(_PREFIX)
                               else _PREFIX + "record_unreadable"})
            continue
        if child.name not in named and not scan.named:
            continue
        if scan.other_sites:
            unresolved.append({"path": str(child), "reason": _PREFIX + "derived_profile_names_another_site"})
            continue
        rows.append(_identity(child))
    return rows


def _queue_files(root: Path, unresolved: list[dict[str, Any]]) -> dict[str, list[Path]]:
    """The packet queue's records by state, set-aside holds and write temporaries included."""
    def record(path: Path) -> bool:
        return path.name.endswith(".json") or (path.name.startswith(TEMPORARY_PREFIX)
                                               and path.name.endswith(TEMPORARY_SUFFIX))

    found: dict[str, list[Path]] = {}
    directories = [("", root), *((state, root / state) for state in (*_AT_REST, "processing", "status"))]
    holds = root / HOLDS_DIRNAME
    if holds.is_dir() and not holds.is_symlink():
        directories.append((HOLDS_DIRNAME, holds))
        directories += [(HOLDS_DIRNAME, child) for child in sorted(holds.iterdir())
                        if child.is_dir() and not child.is_symlink()]
    for state, directory in directories:
        if directory.is_symlink():
            unresolved.append({"path": str(directory), "reason": _PREFIX + "unsafe_entry"})
            continue
        if not directory.is_dir():
            continue
        for child in sorted(directory.iterdir()):
            if not record(child) and not child.is_symlink():
                continue
            if child.is_symlink() or not stat.S_ISREG(os.lstat(child).st_mode):
                unresolved.append({"path": str(child), "reason": _PREFIX + "unsafe_entry"})
                continue
            found.setdefault(state, []).append(child)
    return found


def _packet_records(root: Path, vocabulary: set[bytes], request_id: str,
                    unresolved: list[dict[str, Any]]) -> list[list[Any]]:
    """Every request and status of a request key bound to the site, and any temporary naming the site.

    A key is bound when its request or its status names the site or a planned half. Each of its files
    goes, in every state, unless one is in ``processing/``: that request is still in flight.
    """
    files = _queue_files(root, unresolved)
    scans: dict[Path, Any] = {}
    for paths in files.values():
        for path in paths:
            try:
                scans[path] = _scan([path], vocabulary, request_id)
            except (ValueError, OSError):
                unresolved.append({"path": str(path), "reason": _PREFIX + "record_unreadable"})
    keys = {path.stem for path, scan in scans.items() if scan.named and path.name.endswith(".json")}
    planned = []
    for state, paths in files.items():
        for path in paths:
            scan = scans.get(path)
            bound = path.stem in keys if path.name.endswith(".json") else scan is not None and bool(scan.named)
            if scan is None or not bound:
                continue
            if scan.other_sites:
                unresolved.append({"path": str(path), "reason": _PREFIX + "team_scene_packet_names_another_site"})
            elif state == "processing":
                # Claimed by a worker run, or handed to the remote lane, which may still land a half for it.
                unresolved.append({"path": str(path), "reason": _PREFIX + "team_scene_packet_in_flight"})
            else:
                planned.append(path)
    return [_identity(path) for path in sorted(planned, key=str)]


def _probe(root: Path, keys: set[str], unresolved: list[dict[str, Any]]) -> tuple[list[list[Any]], list[str]]:
    """Each planned half's probe entries to delete, and the probe jobs left with their bundles and money."""
    entries = []
    for directory in ENTRY_DIRECTORIES:
        for key in sorted(keys):
            path = root / directory / f"{key}.json"
            if not (path.exists() or path.is_symlink()):
                continue
            if path.is_symlink() or not stat.S_ISREG(os.lstat(path).st_mode):
                unresolved.append({"path": str(path), "reason": _PREFIX + "unsafe_entry"})
                continue
            entries.append(_identity(path))
    jobs = root / JOBS_DIRECTORY
    left = sorted(str(job) for job in jobs.iterdir() if any(job.name.startswith(key + "-") for key in keys)) \
        if jobs.is_dir() and not jobs.is_symlink() else []
    return sorted(entries), left


def _root(value: Path | None, unresolved: list[dict[str, Any]]) -> Path | None:
    """A named root that is there and safe to read, else None (an unsafe one is unresolved)."""
    if value is None or not Path(value).is_dir():
        return None
    if _unsafe_root(Path(value)):
        unresolved.append({"path": str(value), "reason": _PREFIX + "root_unsafe"})
        return None
    return Path(value)


def plan_team_scene(identity: Mapping[str, str], intents: list[str], *, capture_ids: set[str],
                    capture_keys: set[str], native_g1: Mapping[str, Any], halves_root: Path | None,
                    registry: Path | None, packet_queue_root: Path | None, probe_queue_root: Path | None,
                    unresolved: list[dict[str, Any]]) -> dict[str, Any]:
    """The site's halves and every record pointing at them; reads only."""
    seeds = {identity["scene_id"], *capture_ids, *intents}
    vocabulary = {token.encode() for token in seeds}
    request_id = identity["request_id"]
    section: dict[str, Any] = {
        "halves_root": str(halves_root) if halves_root else None,
        "packet_queue_root": str(packet_queue_root) if packet_queue_root else None,
        "probe_queue_root": str(probe_queue_root) if probe_queue_root else None,
        **_registry(registry, vocabulary, identity, unresolved),
        "halves": [], "derived_profile_rows": [], "packet_queue_records": [], "probe_entries": [],
        "probe_jobs_remaining": []}
    halves = _root(halves_root, unresolved)
    if halves is not None:
        first: dict[str, list[str]] = {}
        for row in native_g1.get("packet_namespaces") or []:
            relative = Path(row["directory"]).relative_to(halves)
            first.setdefault(relative.parts[0], []).append(Path(*relative.parts[1:]).as_posix())
        section["halves"] = _halves(halves, capture_keys, vocabulary, request_id, first, unresolved)
    # A record naming one of the site's halves (or the half a capture of the site is filed under) is the site's.
    keys = capture_keys | {Path(row["directory"]).name for row in section["halves"]}
    pointing = vocabulary | {key.encode() for key in keys}
    derived = _root(Path(section["derived_profile_root"]) if section["derived_profile_root"] else None, unresolved)
    if derived is not None:
        section["derived_profile_rows"] = _derived_rows(derived, keys, pointing, request_id, unresolved)
    packets = _root(packet_queue_root, unresolved)
    if packets is not None:
        section["packet_queue_records"] = _packet_records(packets, pointing, request_id, unresolved)
    probes = _root(probe_queue_root, unresolved)
    if probes is not None:
        section["probe_entries"], section["probe_jobs_remaining"] = _probe(probes, keys, unresolved)
    return section


def hold_packet_queue(stack: ExitStack, queue_root: Path | None) -> None:
    """Hold the packet queue's lock for the whole deletion, without waiting.

    The worker holds it for each run: it claims, compiles and writes derived rows under it.
    Without a lock file no run has ever used this queue, and this deletion creates nothing there.
    """
    from .website_capture_withdrawal_deletion import WithdrawalDeletionError, _require

    if queue_root is None or not Path(queue_root).is_dir():
        return
    _require(not _unsafe_root(Path(queue_root)), "root_unsafe")
    try:
        descriptor = os.open(Path(queue_root) / ".lock", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return
    except OSError:
        raise WithdrawalDeletionError(_PREFIX + "team_scene_packets_lock_unsafe") from None
    lock = stack.enter_context(os.fdopen(descriptor, "rb"))
    _require(stat.S_ISREG(os.fstat(lock.fileno()).st_mode), "team_scene_packets_lock_unsafe")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise WithdrawalDeletionError(_PREFIX + "team_scene_packets_busy") from None


def _unchanged(row: list[Any]) -> bool:
    path = Path(row[0])
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    return (not _unsafe_root(path.parent) and stat.S_ISREG(info.st_mode)
            and [info.st_size, info.st_ino, info.st_mtime_ns] == list(row[1:]))


def apply_team_scene(section: Mapping[str, Any]) -> dict[str, Any]:
    """Remove exactly the reviewed records, then the reviewed halves; refuse anything that changed.

    Runs after ``apply_native_g1``, so each half is checked whole: a G1 namespace written since the
    plan is in it and refuses.
    """
    from .website_capture_withdrawal_deletion import _inventory, _require

    records = [*section["packet_queue_records"], *section["probe_entries"], *section["derived_profile_rows"]]

    def half_unchanged(row: Mapping[str, Any]) -> bool:
        directory = Path(row["directory"])
        return (not _unsafe_root(directory) and directory.is_dir()
                and canonical_digest({"rows": _inventory(directory)}) == row["inventory_digest"])

    # Everything is checked before anything goes, then each again just before it goes.
    _require(all(_unchanged(row) for row in records), "team_scene_record_changed")
    _require(all(half_unchanged(row) for row in section["halves"]), "team_scene_half_changed")
    removed = []
    for row in records:  # The pointers first: an interrupted run leaves halves nothing names.
        _require(_unchanged(row), "team_scene_record_changed")
        Path(row[0]).unlink()
        removed.append(row[0])
    halves = []
    for row in section["halves"]:
        _require(half_unchanged(row), "team_scene_half_changed")
        shutil.rmtree(row["directory"])
        halves.append(row["directory"])
    paths = {str(path) for path, *_ in section["derived_profile_rows"]}
    entries = {str(path) for path, *_ in section["probe_entries"]}
    return {"halves": halves, "half_files_deleted": sum(row["file_count"] for row in section["halves"]),
            "derived_profile_rows": [path for path in removed if path in paths],
            "probe_entries": [path for path in removed if path in entries],
            "packet_queue_records": [path for path in removed if path not in paths | entries]}


def _not_scanned(section: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for field, holder, reason in (
            ("halves_root", "team_scene_halves_not_scanned", "no team scene halves root was named for this "
             "deletion, so no compiled half was found"),
            ("registry", "team_registry_not_read", "no team registry was named for this deletion, so no "
             "derived-profile row pointing at a half was found"),
            ("packet_queue_root", "team_scene_packet_queue_not_scanned", "no team scene packet queue was "
             "named for this deletion, so no request or status for the site was found"),
            ("probe_queue_root", "rigid_robot_free_probe_queue_not_scanned", "no rigid robot-free probe queue "
             "was named for this deletion, so no probe entry or job for the site was found")):
        if section[field] is None:
            rows.append({"holder": holder, "reason": reason})
    return rows


def incomplete(section: Mapping[str, Any]) -> bool:
    """Whether a half, a copy built from one, or a record pointing at one may still exist after this run."""
    return bool(_not_scanned(section) or section["operator_profile_rows"] or section["operator_site_screening_row"]
                or section["probe_jobs_remaining"])


def remaining(section: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = _not_scanned(section)
    if section["registry_state"] == "absent":
        rows.append({"holder": "team_registry_absent", "path": section["registry"],
                     "reason": "the named team registry does not exist here, so no derived-profile row was read"})
    if section["operator_profile_rows"] or section["operator_site_screening_row"]:
        rows.append({"holder": "team_registry_operator_rows", "path": section["registry"],
                     "profile_indexes": list(section["operator_profile_rows"]),
                     "site_screening_candidates": section["operator_site_screening_row"],
                     "reason": "the root-owned operator registry names the site; this deletion never writes it. "
                               "Its readers refuse a half that is gone; remove the rows through the door"})
    if section["probe_jobs_remaining"]:
        rows.append({"holder": "rigid_robot_free_probe_jobs", "paths": list(section["probe_jobs_remaining"]),
                     "reason": "each probe attempt keeps a sealed bundle built from the half's native task packet "
                               "(its bulk may be beside it under rigid-robot-free-probe-work/jobs/) and the "
                               "website-grant hold records its settlement reads; withdrawal deletion v1 does not "
                               "remove them"})
    return rows


__all__ = ["DEFAULT_PACKET_QUEUE_ROOT", "DEFAULT_PROBE_QUEUE_ROOT", "DEFAULT_REGISTRY", "apply_team_scene",
           "hold_packet_queue", "incomplete", "plan_team_scene", "remaining"]
