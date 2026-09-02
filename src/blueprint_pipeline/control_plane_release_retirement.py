"""Retire superseded per-commit release and runtime trees at deploy time.

Every deploy publishes a release worktree and two runtime trees keyed by the
exact commit, and until now nothing ever removed them: the host accumulated 30
worktrees and 138 runtime trees.  Deploy is the only event that creates these
trees, so deploy is where they are retired.  A commit's trees are candidates
only when the commit is not the active release, not the commit being deployed,
not named by any launch profile, standing authorization, or queued envelope,
not among the newest ``keep_last`` releases, and older than a minimum age.
Unknown children are reported and left alone.  Every tree is reproducible from
its commit and the governed prerequisites, so retirement destroys no evidence.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest


PLAN_SCHEMA_VERSION = "control_plane_release_retirement_plan.v1"
RECEIPT_SCHEMA_VERSION = "control_plane_release_retirement_receipt.v1"
EXECUTE_ACK = "retire-superseded-release-trees"
RUNTIME_COMPONENTS = ("splat-render", "scene-configuration")
DEFAULT_KEEP_LAST = 3
DEFAULT_MINIMUM_AGE_SECONDS = 24 * 60 * 60
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_COMMIT_SEARCH_RE = re.compile(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])")
_RECEIPT_RE = re.compile(r"([0-9a-f]{40})\.publication\.v1\.json\Z")
_MAX_REFERENCE_BYTES = 16 * 1024 * 1024


class ControlPlaneReleaseRetirementError(RuntimeError):
    """The retirement plan could not be built or applied safely."""


def _active_commit(active_link: Path, release_root: Path) -> str:
    if not active_link.is_symlink():
        raise ControlPlaneReleaseRetirementError("release_retirement_active_link_invalid")
    target = active_link.resolve(strict=True)
    try:
        relative = target.relative_to(release_root.resolve())
    except ValueError as exc:
        raise ControlPlaneReleaseRetirementError(
            "release_retirement_active_target_outside_root"
        ) from exc
    if len(relative.parts) != 1 or _COMMIT_RE.fullmatch(relative.name) is None:
        raise ControlPlaneReleaseRetirementError("release_retirement_active_target_invalid")
    return relative.name


def _commits_named_under(roots: Sequence[Path]) -> tuple[set[str], list[str]]:
    """Every 40-hex token in every JSON file under the protected reference roots."""

    commits: set[str] = set()
    blockers: list[str] = []
    for root in roots:
        if not root.is_dir():
            blockers.append(f"release_retirement_protected_reference_root_missing:{root.name}")
            continue
        for directory, _subdirectories, files in os.walk(root):
            for name in files:
                if not name.endswith(".json"):
                    continue
                path = Path(directory) / name
                try:
                    if path.is_symlink() or path.stat().st_size > _MAX_REFERENCE_BYTES:
                        continue
                    commits.update(_COMMIT_SEARCH_RE.findall(path.read_text(encoding="utf-8")))
                except (OSError, UnicodeDecodeError):
                    blockers.append(f"release_retirement_protected_reference_unreadable:{name}")
    return commits, blockers


def _managed(root: Path, *, with_receipts: bool) -> tuple[dict[str, list[Path]], list[str]]:
    trees: dict[str, list[Path]] = {}
    unmanaged: list[str] = []
    if not root.is_dir() or root.is_symlink():
        return trees, unmanaged
    for child in sorted(root.iterdir()):
        if _COMMIT_RE.fullmatch(child.name) and child.is_dir() and not child.is_symlink():
            trees.setdefault(child.name, []).insert(0, child)
            continue
        receipt = _RECEIPT_RE.fullmatch(child.name) if with_receipts else None
        if receipt is not None and child.is_file() and not child.is_symlink():
            trees.setdefault(receipt.group(1), []).append(child)
            continue
        unmanaged.append(child.name)
    return trees, unmanaged


def _tree_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for directory, _subdirectories, files in os.walk(path):
        for name in files:
            try:
                total += (Path(directory) / name).lstat().st_size
            except OSError:
                continue
    return total


def build_release_retirement_plan(
    *,
    release_root: str | Path,
    runtime_root: str | Path,
    active_link: str | Path,
    current_commit: str,
    protected_reference_roots: Sequence[str | Path],
    keep_last: int = DEFAULT_KEEP_LAST,
    minimum_age_seconds: int = DEFAULT_MINIMUM_AGE_SECONDS,
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Decide which superseded commits may be retired; mutate nothing."""

    if (
        _COMMIT_RE.fullmatch(str(current_commit)) is None
        or not isinstance(keep_last, int)
        or isinstance(keep_last, bool)
        or keep_last < 1
        or not isinstance(minimum_age_seconds, int)
        or minimum_age_seconds < 0
        or not protected_reference_roots
    ):
        raise ControlPlaneReleaseRetirementError("release_retirement_input_invalid")
    releases = Path(release_root).expanduser()
    runtimes = Path(runtime_root).expanduser()
    observed_at = float(now())
    blockers: list[str] = []
    try:
        active = _active_commit(Path(active_link).expanduser(), releases)
    except (ControlPlaneReleaseRetirementError, OSError) as exc:
        active = None
        blockers.append(str(exc) if isinstance(exc, ControlPlaneReleaseRetirementError) else "release_retirement_active_link_invalid")
    referenced, reference_blockers = _commits_named_under(
        [Path(root).expanduser() for root in protected_reference_roots]
    )
    blockers.extend(reference_blockers)
    release_trees, unmanaged = _managed(releases, with_receipts=False)
    runtime_trees: dict[str, dict[str, list[Path]]] = {}
    for component in RUNTIME_COMPONENTS:
        trees, component_unmanaged = _managed(runtimes / component, with_receipts=True)
        runtime_trees[component] = trees
        unmanaged.extend(f"{component}/{name}" for name in component_unmanaged)
    newest = sorted(
        release_trees,
        key=lambda commit: release_trees[commit][0].stat().st_mtime,
        reverse=True,
    )[:keep_last]
    protected: dict[str, list[str]] = {}
    for commit, reason in [(active, "active_release"), (current_commit, "current_deploy")]:
        if commit:
            protected.setdefault(commit, []).append(reason)
    for commit in referenced:
        protected.setdefault(commit, []).append("named_by_protected_reference")
    for commit in newest:
        protected.setdefault(commit, []).append("keep_last")
    all_commits = set(release_trees) | {
        commit for trees in runtime_trees.values() for commit in trees
    }
    candidates: list[dict[str, Any]] = []
    for commit in sorted(all_commits):
        if commit in protected:
            continue
        paths = list(release_trees.get(commit, []))
        for component in RUNTIME_COMPONENTS:
            paths.extend(runtime_trees[component].get(commit, []))
        age = min(observed_at - path.lstat().st_mtime for path in paths)
        if age < minimum_age_seconds:
            protected.setdefault(commit, []).append("younger_than_minimum_age")
            continue
        candidates.append(
            {
                "commit": commit,
                "paths": [str(path) for path in paths],
                "size_bytes": sum(_tree_bytes(path) for path in paths),
            }
        )
    if blockers:
        # Without a proven active release and readable protection sources the
        # plan cannot know what is live; report and retire nothing.
        candidates = []
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "blocked" if blockers else "dry_run",
        "active_commit": active,
        "current_commit": current_commit,
        "keep_last": keep_last,
        "minimum_age_seconds": minimum_age_seconds,
        "protected_commits": {commit: sorted(set(reasons)) for commit, reasons in sorted(protected.items())},
        "unmanaged_children": sorted(unmanaged),
        "candidate_count": len(candidates),
        "candidate_bytes": sum(row["size_bytes"] for row in candidates),
        "candidates": candidates,
        "blockers": sorted(set(blockers)),
        "evidence_roots_touched": False,
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def apply_release_retirement_plan(
    plan: dict[str, Any], *, ack: str, active_link: str | Path, release_root: str | Path
) -> dict[str, Any]:
    """Remove exactly the planned trees, re-proving the active release first."""

    if (
        ack != EXECUTE_ACK
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("status") != "dry_run"
        or plan.get("plan_digest") != canonical_digest(plan, digest_field="plan_digest")
    ):
        raise ControlPlaneReleaseRetirementError("release_retirement_apply_not_authorized")
    active = _active_commit(Path(active_link).expanduser(), Path(release_root).expanduser())
    removed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in plan.get("candidates") or []:
        commit = str(row.get("commit") or "")
        if commit in {active, str(plan.get("current_commit") or "")} or _COMMIT_RE.fullmatch(commit) is None:
            skipped.append({"commit": commit, "reason": "protected_at_apply"})
            continue
        for raw in row.get("paths") or []:
            path = Path(str(raw))
            if path.is_symlink() or not (path.name == commit or _RECEIPT_RE.fullmatch(path.name)):
                skipped.append({"commit": commit, "reason": "path_changed"})
                continue
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.is_file():
                    path.unlink()
            except OSError as exc:
                skipped.append({"commit": commit, "reason": f"removal_failed:{type(exc).__name__}"})
                continue
            removed.append({"commit": commit, "path": str(path)})
    result: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "applied",
        "source_plan_digest": plan["plan_digest"],
        "active_commit": active,
        "removed_count": len(removed),
        "removed": removed,
        "skipped": skipped,
        "evidence_roots_touched": False,
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--active-link", required=True)
    parser.add_argument("--current-commit", required=True)
    parser.add_argument("--protected-reference-root", action="append", required=True)
    parser.add_argument("--keep-last", type=int, default=DEFAULT_KEEP_LAST)
    parser.add_argument("--minimum-age-seconds", type=int, default=DEFAULT_MINIMUM_AGE_SECONDS)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--ack", default="")
    args = parser.parse_args(argv)
    plan = build_release_retirement_plan(
        release_root=args.release_root,
        runtime_root=args.runtime_root,
        active_link=args.active_link,
        current_commit=args.current_commit,
        protected_reference_roots=args.protected_reference_root,
        keep_last=args.keep_last,
        minimum_age_seconds=args.minimum_age_seconds,
    )
    result = (
        apply_release_retirement_plan(
            plan, ack=args.ack, active_link=args.active_link, release_root=args.release_root
        )
        if args.apply
        else plan
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ControlPlaneReleaseRetirementError",
    "DEFAULT_KEEP_LAST",
    "DEFAULT_MINIMUM_AGE_SECONDS",
    "EXECUTE_ACK",
    "RUNTIME_COMPONENTS",
    "apply_release_retirement_plan",
    "build_release_retirement_plan",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - exercised through module CLI
    raise SystemExit(main())
