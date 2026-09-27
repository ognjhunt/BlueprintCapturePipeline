"""Resolve historical Arena attempts and lease each newly created attempt."""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path

from .control_plane_lane_scratch import (
    LaneScratchError, create_lane_scratch, read_lane_scratch_folder,
)

INPUTS_ROOT = Path("/var/lib/blueprint/task-evaluation-inputs")
LANE_ROOT = INPUTS_ROOT / "lanes"
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
    now: Callable[[], float] = time.time,
) -> Path:
    """Reuse a proven attempt or publish a new sealed Arena evidence folder."""

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
            if lease.get("owner") != owner or lease.get("run_ref", lease.get("scene_ref")) != (
                run_ref if run_ref is not None else scene_ref
            ):
                raise ArenaScratchError("arena_scratch_owner_mismatch")
        return found
    if owner is None or ttl_seconds is None or (run_ref is None) == (scene_ref is None):
        raise ArenaScratchError("arena_scratch_metadata_required")
    if legacy.exists():
        raise ArenaScratchError("arena_scratch_ambiguous")
    try:
        return create_lane_scratch(
            "arena", name, root=lane_root, owner=owner, run_ref=run_ref,
            scene_ref=scene_ref, ttl_seconds=ttl_seconds,
            reason="arena_construction_launch", class_intent="evidence",
            cleanup="owner_review", now=now,
        )
    except LaneScratchError as exc:
        raise ArenaScratchError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "resolve"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--owner")
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument("--run-ref")
    reference.add_argument("--scene-ref")
    parser.add_argument("--ttl-seconds", type=int)
    parser.add_argument("--writable", action="store_true")
    args = parser.parse_args(argv)
    try:
        path = (prepare_arena_attempt(
            args.tag, owner=args.owner, run_ref=args.run_ref, scene_ref=args.scene_ref,
            ttl_seconds=args.ttl_seconds,
        ) if args.action == "prepare" else resolve_arena_attempt(args.tag, writable=args.writable))
    except ArenaScratchError as exc:
        print(str(exc), file=sys.stderr)
        return 2 if str(exc) == "arena_scratch_missing" else 3
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
