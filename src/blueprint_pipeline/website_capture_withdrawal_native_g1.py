"""The G1 humanoid copies of one withdrawn site, for withdrawal deletion.

Beta journey step: site withdrawal after the screening service registered the
site's G1 candidates (``native_g1_site_registration``). That registration leaves
two kinds of copy the run roots and the screening ledger do not reach:

- the service's own row and its status, ``<registration root>/<scene_id>.json``
  and ``<scene_id>.status.json`` (and an interrupted atomic write's
  ``.<file name>.*`` temporary beside either), named by the site's id alone;
- the ``native-g1/`` namespace beside each compiled team scene half's inputs
  (``<site_root>/native-g1/<base>/<source>/`` scene packets derived from the
  site's collision mesh, their staged inputs and locks), under the team scene
  halves root (the registry's ``derived_profiles.packet_inputs.output_root``).
  A namespace is the site's when it lies under the half directory the team
  worker keys by one of the site's capture roots (``capture_key``), or when its
  own JSON records name the site's scene id, capture id or one of its scene
  intents. One that also names another site, a link, or an entry that is not a
  regular file or directory fails closed into ``unresolved``.

``plan_native_g1`` reads only; ``apply_native_g1`` checks each inventory and
file identity again and removes exactly what the reviewed plan named. The
compiled team scene half around a namespace, and every record pointing at it,
are ``website_capture_withdrawal_team_scene``'s: it removes the half once these
namespaces are gone. Records here carry paths, sizes, inode numbers and counts
only, never what a file holds.
"""
from __future__ import annotations

import os
import shutil
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .team_candidate_registry_reader import DEFAULT_SITE_REGISTRATION_ROOT, SITE_REGISTRATION_ROOT_ENV

# The units' values, unless set here too (deploy/systemd/blueprint-beta-policy-screening.service and
# docs/runbooks/first-site-team-beta.md's packet_inputs.output_root).
DEFAULT_REGISTRATION_ROOT = Path(os.environ.get(SITE_REGISTRATION_ROOT_ENV) or DEFAULT_SITE_REGISTRATION_ROOT)
DEFAULT_HALVES_ROOT = Path("/var/lib/blueprint/task-evaluation-inputs/team-scene-halves")
NAMESPACE = "native-g1"
# A half's inputs sit at ``<halves>/<capture key>/<attempt>/scene`` (team_scene_packet_worker); the
# namespace beside them is looked for no deeper than that.
_PARENT_DEPTH = 3
_PREFIX = "website_withdrawal_deletion_"


def _registration_files(root: Path, scene_id: str, unresolved: list[dict[str, Any]]) -> list[list[Any]]:
    """``[path, size, inode, mtime_ns]`` of the site's row, its status and their write temporaries."""
    if any(path.is_symlink() for path in (root, *root.parents)):
        unresolved.append({"path": str(root), "reason": _PREFIX + "root_unsafe"})
        return []
    names = (scene_id + ".json", scene_id + ".status.json")
    rows = []
    for child in sorted(root.iterdir()):
        if child.name not in names and not any(child.name.startswith(f".{name}.") for name in names):
            continue
        info = os.lstat(child)
        if not stat.S_ISREG(info.st_mode):
            unresolved.append({"path": str(child), "reason": _PREFIX + "unsafe_entry"})
            continue
        rows.append([str(child), info.st_size, info.st_ino, info.st_mtime_ns])
    return rows


def _namespaces(root: Path, unresolved: list[dict[str, Any]]) -> list[Path]:
    """Every ``native-g1`` directory in the halves root, no deeper than a half's inputs."""
    found = []

    def unreadable(error: OSError) -> None:
        unresolved.append({"path": str(error.filename or root), "reason": _PREFIX + "record_unreadable"})

    for directory, directories, _files in os.walk(root, onerror=unreadable):
        here = Path(directory)
        if NAMESPACE in directories:
            if (here / NAMESPACE).is_symlink():
                unresolved.append({"path": str(here / NAMESPACE), "reason": _PREFIX + "unsafe_entry"})
            else:
                found.append(here / NAMESPACE)
        depth = len(here.relative_to(root).parts)
        directories[:] = [] if depth >= _PARENT_DEPTH else sorted(name for name in directories if name != NAMESPACE)
    return sorted(found)


def _site_capture_ids(site: Path, identity: Mapping[str, str]) -> set[str]:
    """The site's walkthrough capture and every capture directory still under it."""
    captures = site / "captures"
    return {identity["capture_id"]} | ({path.name for path in captures.iterdir() if path.is_dir()}
                                       if captures.is_dir() and not captures.is_symlink() else set())


def _site_capture_keys(site: Path, identity: Mapping[str, str]) -> set[str]:
    """The half directories the team worker files the site's captures under, also once the captures are gone."""
    # Not the worker: it reaches the paid launch path, which the intake must not import.
    from .team_scene_capture_root import capture_key

    return {capture_key((site / "captures" / name).resolve()) for name in _site_capture_ids(site, identity)}


def plan_native_g1(identity: Mapping[str, str], intents: list[str], *, site: Path,
                   registration_root: Path | None, halves_root: Path | None,
                   unresolved: list[dict[str, Any]]) -> dict[str, Any]:
    """The site's G1 registration files and the G1 packet namespaces inside its team scene halves."""
    from .website_capture_withdrawal_deletion import (
        _SCANNED_SUFFIXES, WithdrawalDeletionError, _inventory, _Scan, _scan_files,
    )
    section: dict[str, Any] = {
        "registration_root": str(registration_root) if registration_root else None,
        "halves_root": str(halves_root) if halves_root else None,
        "registration_files": [], "packet_namespaces": []}
    if registration_root is not None and Path(registration_root).is_dir():
        section["registration_files"] = _registration_files(Path(registration_root), identity["scene_id"], unresolved)
    if halves_root is None or not Path(halves_root).is_dir():
        return section
    halves_root = Path(halves_root)
    if any(path.is_symlink() for path in (halves_root, *halves_root.parents)):
        unresolved.append({"path": str(halves_root), "reason": _PREFIX + "root_unsafe"})
        return section
    keys = _site_capture_keys(site, identity)
    seeds = {identity["scene_id"], identity["capture_id"], *intents}
    for namespace in _namespaces(halves_root, unresolved):
        half = halves_root / namespace.relative_to(halves_root).parts[0]
        scan = _Scan({token.encode() for token in seeds}, identity["request_id"])
        try:
            inventory = _inventory(namespace)
            _scan_files((namespace / row[0] for row in inventory if row[0].endswith(_SCANNED_SUFFIXES)), scan)
        except (WithdrawalDeletionError, OSError) as error:
            # Unreadable, or holding a link or special file: it could be this site's, so it is never skipped.
            unresolved.append({"path": str(namespace), "reason": str(error) if isinstance(
                error, WithdrawalDeletionError) else _PREFIX + "record_unreadable"})
            continue
        bound_by = "capture_key" if half.name in keys else "records" if scan.named else None
        if bound_by is None:
            continue  # Another capture's half, and nothing in it names this site.
        if scan.other_sites:
            unresolved.append({"path": str(namespace), "reason": _PREFIX + "native_g1_names_another_site"})
            continue
        section["packet_namespaces"].append({"directory": str(namespace), "bound_by": bound_by,
                                             "inventory_digest": canonical_digest({"rows": inventory}),
                                             "file_count": len(inventory)})
    return section


def apply_native_g1(section: Mapping[str, Any]) -> dict[str, Any]:
    """Remove exactly the reviewed namespaces and registration files; refuse one that changed."""
    from .website_capture_withdrawal_deletion import _inventory, _require

    namespaces, files = [], []
    for row in section["packet_namespaces"]:
        directory = Path(row["directory"])
        _require(not any(path.is_symlink() for path in (directory, *directory.parents)), "unsafe_entry")
        _require(canonical_digest({"rows": _inventory(directory)}) == row["inventory_digest"], "native_g1_changed")
        shutil.rmtree(directory)
        namespaces.append(row["directory"])
    for path, size, inode, mtime_ns in section["registration_files"]:
        target = Path(path)
        _require(not any(parent.is_symlink() for parent in target.parents), "unsafe_entry")
        try:
            info = os.lstat(target)
        except FileNotFoundError:
            continue  # Already gone after an interrupted run.
        _require(stat.S_ISREG(info.st_mode) and [info.st_size, info.st_ino, info.st_mtime_ns] == [size, inode, mtime_ns],
                 "native_g1_registration_changed")
        target.unlink()
        files.append(path)
    return {"packet_namespaces": namespaces,
            "packet_files_deleted": sum(row["file_count"] for row in section["packet_namespaces"]),
            "registration_files": files}


def incomplete(section: Mapping[str, Any]) -> bool:
    """Whether a G1 copy may still exist after this run: a root this deletion was not given."""
    return section["registration_root"] is None or section["halves_root"] is None


def remaining(section: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    if section["registration_root"] is None:
        rows.append({"holder": "native_g1_registrations_not_scanned",
                     "reason": "no G1 site registration root was named for this deletion"})
    if section["halves_root"] is None:
        rows.append({"holder": "native_g1_packets_not_scanned",
                     "reason": "no team scene halves root was named for this deletion, so no G1 packet was found"})
    return rows


__all__ = ["DEFAULT_HALVES_ROOT", "DEFAULT_REGISTRATION_ROOT", "NAMESPACE", "apply_native_g1", "incomplete",
           "plan_native_g1", "remaining"]
