"""Resolve historical Arena attempts and lease each newly created attempt."""

from __future__ import annotations

import argparse
import math
import os
import stat
import fcntl
import subprocess
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path

from .control_plane_lane_scratch import (
    LaneScratchError, read_lane_scratch_folder,
)
from .control_plane_leased_scratch import LeasedScratchDirectory

INPUTS_ROOT = Path("/var/lib/blueprint/task-evaluation-inputs")
LANE_ROOT = INPUTS_ROOT / "lanes"
REGISTERED_CHAIN_PATH = Path("/opt/blueprint/task-evaluation-control-plane/scripts/arena_construction_launch_chain.sh")
_TAG = re.compile(r"r[0-9]{1,6}\Z")
# The checked-in fire script records refusals through r32. Later tags must
# have a sealed lease; a marker in a loose folder cannot grant access.
_LAST_RECORDED_LEGACY_TAG = 32
_LEGACY_DIRS = ("arena_packet", "arena_construction_job")
_LEGACY_FILES = (
    "prior_spend_reconciliation.v1.json",
    "native_task_arena_paid_attempt_authority.v1.json",
    "arena_construction_live_profile.v1.json", "construction_provider_zero.v1.json",
)


class ArenaScratchError(RuntimeError):
    """An Arena attempt path is missing or unsafe for the requested operation."""



def admit_registered_arena_attempt(tag, *, now=time.time):
    """Acquire the fixed current root-owned tag selection and SAME target SH."""
    from .control_plane_lane_experiment_consumer import RegisteredExperimentUse
    if not isinstance(tag, str) or re.fullmatch(r"r[1-9][0-9]{0,5}", tag) is None:
        raise ArenaScratchError("arena_scratch_tag_invalid")
    return RegisteredExperimentUse.admit(None, _arena_tag=tag, now=now)



def _registered_arena(tag, use, *, owner=None, run_ref=None, scene_ref=None, ttl_seconds=None):
    from .control_plane_lane_experiment_consumer import RegisteredExperimentUse
    from .control_plane_lane_owner_target_versions import _require
    _require(type(use) is RegisteredExperimentUse and not use._closed,
             "experiment_consumer_authority_required")
    use.check()
    _require(isinstance(tag, str) and re.fullmatch(r"r[1-9][0-9]{0,5}", tag)
             and use.lane == "arena" and use.entry["root"] == "inputs"
             and use.birth["participant_profile"] == "arena_owner_review.v1"
             and use.birth["reference_value"] == "arena-launch-" + tag
             and (owner is None or owner == use.entry["owner"])
             and (run_ref is None or run_ref == use.birth["reference_value"])
             and scene_ref is None and ttl_seconds is None,
             "experiment_arena_selection_invalid")
    return use


def _mkdir_registered_arena(use, relative):
    from .control_plane_lane_experiment_publication import _BirthFiles
    from .control_plane_lane_owner_target_versions import _require
    from .control_plane_reference_budget import ReferenceCollectionBudget

    _require(isinstance(relative, str) and 0 < len(relative.encode()) <= 1024
             and not Path(relative).is_absolute() and str(Path(relative)) == relative
             and 0 < len(Path(relative).parts) <= 32
             and all(part not in (".", "..") and len(part.encode()) <= 255 for part in Path(relative).parts),
             "experiment_arena_payload_invalid")

    class _ArenaFiles(_BirthFiles):
        def slot(self):
            use.check()
            _require(len(self.owned) + len(self.probe_owned) + len(use.files.owned)
                     + len(use.files.probe_owned) < 128, "experiment_consumer_resource_exhausted")
            super().slot()

        def location(self, fd, *, cleanup=False):
            if not cleanup:
                use.check()
            return super().location(fd, cleanup=cleanup)

    files = _ArenaFiles(ReferenceCollectionBudget(values_limit=10000))
    try:
        current, _ = files.parent(use.path / "unused")
        original = os.fstat(current)
        _require((original.st_dev, original.st_ino) == (use.entry["target_identity"]["dev"],
                                                      use.entry["target_identity"]["ino"]),
                 "experiment_target_changed")
        for component in Path(relative).parts:
            files.budget.charge("entries")
            files.location(current)
            try:
                named = os.stat(component, dir_fd=current, follow_symlinks=False)
            except FileNotFoundError:
                files.location(current)
                use.check()
                os.mkdir(component, 0o700, dir_fd=current)
                files.location(current)
                named = os.stat(component, dir_fd=current, follow_symlinks=False)
            _require(stat.S_ISDIR(named.st_mode) and not named.st_mode & 0o022,
                     "experiment_arena_payload_invalid")
            child = files.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, parent=current)
            _require((os.fstat(child).st_dev, os.fstat(child).st_ino) == (named.st_dev, named.st_ino),
                     "experiment_arena_payload_invalid")
            files.location(child)
            current = child
        use.check()
        return use.path / relative
    finally:
        try:
            files.finish()
        finally:
            files.budget.close()



def run_registered_arena_chain(tag, *, previous_tag, now=time.time):
    """Fixed direct shell lifetime; descendant/provider completeness is unknown."""
    from .control_plane_lane_experiment_publication import _BirthFiles
    from . import control_plane_lane_owner_consents as owners
    from .control_plane_lane_owner_target_versions import _require
    from .control_plane_reference_budget import ReferenceCollectionBudget
    _require(isinstance(previous_tag, str) and re.fullmatch(r"r[1-9][0-9]{0,5}", previous_tag),
             "experiment_arena_selection_invalid")
    use = admit_registered_arena_attempt(tag, now=now)
    files = _BirthFiles(ReferenceCollectionBudget(values_limit=10000))
    try:
        # Root-owned fixed installed code, never a caller-selected command.
        _, source = files.read(REGISTERED_CHAIN_PATH, cap=1048576, protected=True)
        use.check()
        files.verify_record(source)
        files.budget.close()
        remaining = use._started + 4 * 3600 - time.monotonic()
        _require(remaining > 0, "experiment_consumer_resource_exhausted")
        env = dict(os.environ, CUR=tag, PREV=previous_tag,
                   BLUEPRINT_REGISTERED_ARENA_FD=str(use.fd))
        child = subprocess.run(["/bin/bash", str(REGISTERED_CHAIN_PATH), "--registered-child"],
                               env=env, pass_fds=(use.fd,), check=False, timeout=remaining)
        # The bounded admission is closed. Only this retained original record
        # and its named ancestors are checked; no new metadata or payload read.
        files.location(source.parent, cleanup=True)
        files.proof(source.fd)
        _require(owners._metadata(os.fstat(source.fd)) == owners._metadata(source.info)
                 == owners._metadata(os.stat(source.name, dir_fd=source.parent, follow_symlinks=False)),
                 "experiment_arena_source_changed")
        use.check()
        _require(type(child.returncode) is int, "experiment_arena_child_failed")
        return child.returncode
    finally:
        try:
            files.finish()
        finally:
            files.budget.close()
            use.close()


def _verify_arena_parent(use, fd):
    """Check an inherited target handle without adopting or closing its token."""
    from .control_plane_lane_owner_target_versions import _require
    use.check()
    _require(type(fd) is int and fd >= 0, "experiment_arena_parent_invalid")
    observed = os.fstat(fd)
    expected = use.entry["target_identity"]
    _require(stat.S_ISDIR(observed.st_mode) and (observed.st_dev, observed.st_ino) == (expected["dev"], expected["ino"]),
             "experiment_arena_parent_invalid")
    # The guard binds this lock operation to the independently selected current
    # directory. The inherited token is never entered into an owned registry.
    fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    after = os.fstat(fd)
    _require((after.st_dev, after.st_ino, stat.S_IFMT(after.st_mode))
             == (observed.st_dev, observed.st_ino, stat.S_IFMT(observed.st_mode)),
             "experiment_arena_parent_invalid")
    use.check()


def _paths(tag: str, *, inputs_root: Path, lane_root: Path) -> tuple[Path, Path, str]:
    if not isinstance(tag, str) or not _TAG.fullmatch(tag):
        raise ArenaScratchError("arena_scratch_tag_invalid")
    lane = lane_root / "arena"
    if (not inputs_root.is_absolute() or lane_root != inputs_root / "lanes"
            or inputs_root.is_symlink() or inputs_root.resolve() != inputs_root
            or lane_root.is_symlink() or lane_root.resolve() != lane_root
            or lane.is_symlink() or lane.resolve() != lane):
        raise ArenaScratchError("arena_scratch_root_unsafe")
    name = f"arena-launch-{tag}"
    return inputs_root / name, lane_root / "arena" / name, name


def _active(lease: dict, *, now: Callable[[], float]) -> bool:
    try:
        observed = float(now())
        return (math.isfinite(observed) and lease.get("released_at_epoch") is None
                and float(lease["expires_at_epoch"]) > observed)
    except (TypeError, ValueError, OverflowError, KeyError):
        return False


def _legacy_proven(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (
        any((path / marker).is_dir() and not (path / marker).is_symlink()
            for marker in _LEGACY_DIRS)
        or any((path / marker).is_file() and not (path / marker).is_symlink()
               for marker in _LEGACY_FILES)
    )


def resolve_arena_attempt(
    tag: str, *, writable: bool = False,
    inputs_root: Path = INPUTS_ROOT, lane_root: Path = LANE_ROOT,
    now: Callable[[], float] = time.time,
) -> Path:
    """Find a leased attempt or a recognizable historical attempt, without creating it."""

    legacy, leased, name = _paths(tag, inputs_root=Path(inputs_root), lane_root=Path(lane_root))
    if legacy.is_symlink() or leased.is_symlink():
        raise ArenaScratchError("arena_scratch_path_unsafe")
    if legacy.exists() and leased.exists():
        raise ArenaScratchError("arena_scratch_ambiguous")
    if leased.exists():
        try:
            lease = read_lane_scratch_folder(leased, lane="arena", name=name)
        except LaneScratchError as exc:
            raise ArenaScratchError("arena_scratch_lease_invalid") from exc
        if (lease.get("class_intent") != "evidence" or lease.get("cleanup") != "owner_review"
                or lease.get("reason") != "arena_construction_launch"):
            raise ArenaScratchError("arena_scratch_lease_mismatch")
        if writable and not _active(lease, now=now):
            raise ArenaScratchError("arena_scratch_inactive")
        return leased
    if legacy.exists():
        number = int(tag[1:])
        if not (1 <= number <= _LAST_RECORDED_LEGACY_TAG and tag == f"r{number}"):
            raise ArenaScratchError("arena_scratch_legacy_tag_unrecognized")
        if not _legacy_proven(legacy):
            raise ArenaScratchError("arena_scratch_legacy_unproven")
        # A marker establishes only that there may be historical data to read.
        # It does not prove the folder existed before lease enforcement. Exact
        # legacy write authorization needs an owner-reviewed migration.
        if writable:
            raise ArenaScratchError(
                "arena_scratch_legacy_write_requires_review: inspect the exact folder "
                "before starting a new leased attempt"
            )
        return legacy
    raise ArenaScratchError("arena_scratch_missing")


def prepare_arena_attempt(
    tag: str, *, owner: str | None = None, run_ref: str | None = None,
    scene_ref: str | None = None, ttl_seconds: int | None = None,
    inputs_root: Path = INPUTS_ROOT, lane_root: Path = LANE_ROOT,
    now: Callable[[], float] = time.time, _registered_use=None,
) -> Path:
    """Reuse a proven attempt or publish a new sealed Arena evidence folder."""

    if _registered_use is not None:
        return _registered_arena(tag, _registered_use, owner=owner, run_ref=run_ref,
            scene_ref=scene_ref, ttl_seconds=ttl_seconds).path
    inputs_root, lane_root = Path(inputs_root), Path(lane_root)
    legacy, leased, name = _paths(tag, inputs_root=inputs_root, lane_root=lane_root)
    try:
        found = resolve_arena_attempt(tag, writable=True, inputs_root=inputs_root,
                                      lane_root=lane_root, now=now)
    except ArenaScratchError as exc:
        if str(exc) != "arena_scratch_missing":
            raise
    else:
        if found == leased and (owner is not None or run_ref is not None or scene_ref is not None):
            if owner is None or (run_ref is None) == (scene_ref is None):
                raise ArenaScratchError("arena_scratch_metadata_required")
            lease = read_lane_scratch_folder(found, lane="arena", name=name)
            reference_key = "run_ref" if run_ref is not None else "scene_ref"
            reference_value = run_ref if run_ref is not None else scene_ref
            if (lease.get("owner") != owner or reference_key not in lease
                    or lease[reference_key] != reference_value):
                raise ArenaScratchError("arena_scratch_owner_mismatch")
        return found
    # Caller-supplied owner and TTL cannot register a new experiment.
    raise ArenaScratchError("arena_scratch_registered_authority_required")


def mkdir_arena_payload(
    tag: str, relative: str, *, owner: str | None = None, run_ref: str | None = None,
    scene_ref: str | None = None, inputs_root: Path = INPUTS_ROOT,
    lane_root: Path = LANE_ROOT, now: Callable[[], float] = time.time, _registered_use=None,
) -> Path:
    """Reopen an admitted lease for payload directories, never a legacy write.

    A retry without supplied metadata binds to the exact saved lease identity;
    supplied metadata must match. Neither route grants new ownership or a lease.
    Returned paths and subsequent cp/file writes are outside the handle guarantee.
    """

    if _registered_use is not None:
        use = _registered_arena(tag, _registered_use, owner=owner, run_ref=run_ref, scene_ref=scene_ref)
        return _mkdir_registered_arena(use, relative)
    folder = resolve_arena_attempt(tag, writable=True, inputs_root=inputs_root,
                                   lane_root=lane_root, now=now)
    try:
        lease = read_lane_scratch_folder(folder, lane="arena", name=folder.name)
        if (lease.get("class_intent") != "evidence" or lease.get("cleanup") != "owner_review"
                or lease.get("reason") != "arena_construction_launch"):
            raise ArenaScratchError("arena_scratch_lease_mismatch")
        reference_key = "run_ref" if "run_ref" in lease else "scene_ref"
        if owner is not None or run_ref is not None or scene_ref is not None:
            if owner is None or (run_ref is None) == (scene_ref is None):
                raise ArenaScratchError("arena_scratch_metadata_required")
            supplied_key = "run_ref" if run_ref is not None else "scene_ref"
            supplied_value = run_ref if run_ref is not None else scene_ref
            if (owner != lease["owner"] or supplied_key != reference_key
                    or supplied_value != lease[reference_key]):
                raise ArenaScratchError("arena_scratch_owner_mismatch")
        with LeasedScratchDirectory.open(
            root=lane_root, lane="arena", name=folder.name, owner=lease["owner"],
            now=now, **{reference_key: lease[reference_key]},
        ) as scratch:
            if scratch.lease_digest != lease["lease_digest"]:
                raise LaneScratchError("lane_scratch_lease_changed")
            return scratch.mkdir(relative, parents=True, exist_ok=True)
    except LaneScratchError as exc:
        raise ArenaScratchError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "resolve", "mkdir-payload", "registered-prepare", "registered-mkdir", "verify-parent", "run-chain"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--owner")
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument("--run-ref")
    reference.add_argument("--scene-ref")
    parser.add_argument("--ttl-seconds", type=int)
    parser.add_argument("--writable", action="store_true")
    parser.add_argument("--relative")
    parser.add_argument("--prev")
    parser.add_argument("--parent-fd", type=int)
    args = parser.parse_args(argv)
    if args.action in ("mkdir-payload", "registered-mkdir") and args.relative is None:
        parser.error("mkdir-payload requires --relative")
    try:
        if args.action == "run-chain":
            return run_registered_arena_chain(args.tag, previous_tag=args.prev)
        if args.action in ("registered-prepare", "registered-mkdir", "verify-parent"):
            use = admit_registered_arena_attempt(args.tag)
            try:
                if args.action == "verify-parent":
                    _verify_arena_parent(use, args.parent_fd)
                    path = use.path
                elif args.action == "registered-prepare":
                    path = prepare_arena_attempt(args.tag, _registered_use=use)
                else:
                    path = mkdir_arena_payload(args.tag, args.relative, _registered_use=use)
            finally:
                use.close()
        elif args.action == "prepare":
            path = prepare_arena_attempt(
                args.tag, owner=args.owner, run_ref=args.run_ref, scene_ref=args.scene_ref,
                ttl_seconds=args.ttl_seconds,
            )
        elif args.action == "mkdir-payload":
            path = mkdir_arena_payload(args.tag, args.relative, owner=args.owner,
                                       run_ref=args.run_ref, scene_ref=args.scene_ref)
        else:
            path = resolve_arena_attempt(args.tag, writable=args.writable)
    except (ArenaScratchError, OSError, ValueError) as exc:
        code = str(exc) if not isinstance(exc, OSError) else "arena_scratch_io_failed"
        print(code if re.fullmatch(r"[a-z][a-z0-9_]{0,127}", code) else "arena_scratch_refused", file=sys.stderr)
        return 2 if str(exc) == "arena_scratch_missing" else 3
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
