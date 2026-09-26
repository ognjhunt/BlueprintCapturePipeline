"""Retire superseded per-commit release and runtime trees at deploy time.

Every deploy publishes a release worktree and two runtime trees keyed by the
exact commit, and until now nothing ever removed them: the host accumulated 30
worktrees and 138 runtime trees.  Deploy is the only event that creates these
trees, so deploy is where they are retired.

A commit's trees stay while it is the active release, the commit being
deployed, among the newest ``keep_last`` releases, in use by a live process,
younger than a minimum age, or held by a typed protection row from
``control_plane_release_leases``: an unexpired lease (a live queue envelope, a
standing authorization that can still launch, a required-evidence binding
whose run is not over) or a configured runtime path.  Protection used to be
every 40-hex token in about twenty JSON roots, which is how 513 commits became
protected and none of 95 trees could be retired; nothing greps any more.

Unknown children are reported and left alone.  Every tree is reproducible from
its commit and the governed prerequisites, so retirement destroys no evidence.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .control_plane_release_leases import (
    CONFIG_KIND,
    DEFAULT_PROTECTION_SOURCES,
    LEASE_ALERT_THRESHOLD,
    LEASE_KINDS,
    PROTECTIONS_SCHEMA,
    ProtectionSources,
    collect_release_protections,
)
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_release_reference_lock import release_reference_lock


PLAN_SCHEMA_VERSION = "control_plane_release_retirement_plan.v1"
RECEIPT_SCHEMA_VERSION = "control_plane_release_retirement_receipt.v1"
EXECUTE_ACK = "retire-superseded-release-trees"
RUNTIME_COMPONENTS = ("splat-render", "scene-configuration")
DEFAULT_KEEP_LAST = 3
DEFAULT_MINIMUM_AGE_SECONDS = 24 * 60 * 60
MAX_REPORTED_WARNINGS = 50
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_RECEIPT_RE = re.compile(r"([0-9a-f]{40})\.publication\.v1\.json\Z")
_PROTECTION_KINDS = frozenset((*LEASE_KINDS, CONFIG_KIND))


class ControlPlaneReleaseRetirementError(RuntimeError):
    """The retirement plan could not be built or applied safely."""


def publisher_lock_roots(
    sources: ProtectionSources, *extra_roots: str | Path
) -> tuple[Path, ...]:
    """Every distinct directory a release-reference publisher locks, sorted.

    Queue writers lock the parent of their queue root (the control-plane
    root), the standing-authorization materializer locks the parent of its
    output directory, and the launch-profile publisher locks the parent of the
    public catalog, which lives in the control-plane root.  Callers add their
    own state root, which release activation locks.  Roots are resolved so a
    directory is never locked twice: a second descriptor on the same inode
    would deadlock an exclusive holder against itself.
    """

    return tuple(
        sorted(
            {
                Path(root).expanduser().resolve()
                for root in (
                    sources.control_plane_root,
                    Path(sources.standing_authorization_dir).parent,
                    Path(sources.binding_root).parent,
                    *extra_roots,
                )
            }
        )
    )


@contextlib.contextmanager
def holding_publisher_locks(roots: Sequence[str | Path]):
    """Hold every publisher lock exclusively, acquired in the given order."""

    with contextlib.ExitStack() as stack:
        for root in roots:
            stack.enter_context(release_reference_lock(root, exclusive=True))
        yield


def live_release_commits(
    release_root: str | Path,
    runtime_root: str | Path | None = None,
    *,
    proc_root: str | Path = "/proc",
) -> list[str]:
    """Commits whose release or runtime tree a live process runs from; never retired.

    A process counts when its cwd, its executable, any absolute argv entry, or
    the value of a ``--flag=/abs/path`` argv entry lies inside
    ``<release_root>/<commit>`` or ``<runtime_root>/<component>/<commit>``.
    Only a 40-hex first component names a commit.  A missing process table
    raises: the caller cannot know what is in use.
    """

    roots = [Path(release_root).expanduser().resolve()]
    if runtime_root is not None:
        runtimes = Path(runtime_root).expanduser().resolve()
        roots.extend(runtimes / component for component in RUNTIME_COMPONENTS)
    commits: set[str] = set()
    for entry in Path(proc_root).iterdir():
        if not entry.name.isdigit():
            continue
        candidates: list[str] = []
        for link in ("cwd", "exe"):
            with contextlib.suppress(OSError):
                candidates.append(os.readlink(entry / link))
        with contextlib.suppress(OSError):
            for part in (entry / "cmdline").read_bytes().split(b"\0"):
                if not part:
                    continue
                text = part.decode("utf-8", "replace")
                candidates.append(text)
                if "=" in text:
                    candidates.append(text.split("=", 1)[1])
        for candidate in candidates:
            path = Path(candidate)
            if not path.is_absolute():
                continue
            for root in roots:
                if path.is_relative_to(root) and path != root:
                    first = path.relative_to(root).parts[0]
                    if _COMMIT_RE.fullmatch(first):
                        commits.add(first)
    return sorted(commits)


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


def _protections_valid(protections: Any) -> bool:
    if not isinstance(protections, Mapping) or protections.get("schema_version") != PROTECTIONS_SCHEMA:
        return False
    for field in ("leases", "lapsed", "migrated", "warnings", "blockers"):
        if not isinstance(protections.get(field), list):
            return False
    return all(
        isinstance(row, Mapping)
        and isinstance(row.get("commit"), str)
        and _COMMIT_RE.fullmatch(row["commit"]) is not None
        and row.get("kind") in _PROTECTION_KINDS
        and isinstance(row.get("reason"), str)
        and bool(row["reason"])
        for row in protections["leases"]
    ) and all(isinstance(blocker, str) and blocker for blocker in protections["blockers"])


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
    protections: Mapping[str, Any],
    keep_last: int = DEFAULT_KEEP_LAST,
    minimum_age_seconds: int = DEFAULT_MINIMUM_AGE_SECONDS,
    now: Callable[[], float] = time.time,
    in_use_commits: Sequence[str] = (),
) -> dict[str, Any]:
    """Decide which superseded commits may be retired; mutate nothing.

    ``protections`` is a ``collect_release_protections`` result.  Any of its
    blockers blocks the plan: without readable protection sources it cannot
    know what is live.
    """

    if (
        _COMMIT_RE.fullmatch(str(current_commit)) is None
        or not isinstance(keep_last, int)
        or isinstance(keep_last, bool)
        or keep_last < 1
        or not isinstance(minimum_age_seconds, int)
        or minimum_age_seconds < 0
        or not _protections_valid(protections)
    ):
        raise ControlPlaneReleaseRetirementError("release_retirement_input_invalid")
    releases = Path(release_root).expanduser()
    runtimes = Path(runtime_root).expanduser()
    observed_at = float(now())
    blockers: list[str] = list(protections["blockers"])
    try:
        active = _active_commit(Path(active_link).expanduser(), releases)
    except (ControlPlaneReleaseRetirementError, OSError) as exc:
        active = None
        blockers.append(str(exc) if isinstance(exc, ControlPlaneReleaseRetirementError) else "release_retirement_active_link_invalid")
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
    kinds: dict[str, set[str]] = {}

    def protect(commit: str, reason: str, kind: str) -> None:
        protected.setdefault(commit, []).append(reason)
        kinds.setdefault(commit, set()).add(kind)

    for commit, reason in [(active, "active_release"), (current_commit, "current_deploy")]:
        if commit:
            protect(commit, reason, reason)
    for row in protections["leases"]:
        kind, reason = str(row["kind"]), str(row["reason"])
        protect(row["commit"], reason if reason.startswith(f"{kind}:") else f"{kind}:{reason}", kind)
    for commit in newest:
        protect(commit, "keep_last", "keep_last")
    # A paid run may outlive the deploy that superseded its release.
    for commit in in_use_commits:
        protect(commit, "in_use_by_live_process", "in_use_by_live_process")
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
            protect(commit, "younger_than_minimum_age", "younger_than_minimum_age")
            continue
        candidates.append(
            {
                "commit": commit,
                "paths": [str(path) for path in paths],
                "size_bytes": sum(_tree_bytes(path) for path in paths),
            }
        )
    blockers = sorted(set(blockers))
    if blockers:
        # Without a proven active release and readable protection sources the
        # plan cannot know what is live; report and retire nothing.
        candidates = []
    # Grouped for the receipt, counting only commits whose trees still exist.
    protected_by_kind: dict[str, set[str]] = {}
    for commit in all_commits & set(kinds):
        for kind in kinds[commit]:
            protected_by_kind.setdefault(kind, set()).add(commit)
    lease_kinds = set(LEASE_KINDS)
    lease_protected_tree_count = sum(
        1 for commit in all_commits if commit in kinds and kinds[commit] <= lease_kinds
    )
    alerts: list[str] = []
    if lease_protected_tree_count > LEASE_ALERT_THRESHOLD:
        alerts.append(f"release_retirement_lease_protected_trees:{lease_protected_tree_count}")
    if blockers:
        alerts.append(f"release_retirement_blocked:{blockers[0]}")
    warnings = sorted(set(protections["warnings"]))
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "blocked" if blockers else "dry_run",
        "active_commit": active,
        "current_commit": current_commit,
        "keep_last": keep_last,
        "minimum_age_seconds": minimum_age_seconds,
        "protected_commits": {commit: sorted(set(reasons)) for commit, reasons in sorted(protected.items())},
        "protected_by_kind": {
            kind: sorted(commits) for kind, commits in sorted(protected_by_kind.items())
        },
        "protected_tree_count": len(all_commits & set(protected)),
        "lease_protected_tree_count": lease_protected_tree_count,
        "lapsed_count": len(protections["lapsed"]),
        "migrated": sorted(protections["migrated"]),
        "warning_count": len(warnings),
        "warnings": warnings[:MAX_REPORTED_WARNINGS],
        "alerts": alerts,
        "unmanaged_children": sorted(unmanaged),
        "candidate_count": len(candidates),
        "candidate_bytes": sum(row["size_bytes"] for row in candidates),
        "candidates": candidates,
        "blockers": blockers,
        "evidence_roots_touched": False,
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def apply_release_retirement_plan(
    plan: dict[str, Any],
    *,
    ack: str,
    active_link: str | Path,
    release_root: str | Path,
    in_use_now: Callable[[], set[str]] | None = None,
) -> dict[str, Any]:
    """Remove exactly the planned trees, re-proving the active release first.

    ``in_use_now`` is asked again immediately before each commit's paths are
    removed: a process that started on a candidate after the plan was built
    keeps it (``in_use_at_apply``), and a probe that fails keeps it too.
    """

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
        if in_use_now is not None:
            try:
                busy = commit in in_use_now()
            except (OSError, ValueError) as exc:
                skipped.append(
                    {"commit": commit, "reason": f"in_use_check_failed:{type(exc).__name__}"}
                )
                continue
            if busy:
                skipped.append({"commit": commit, "reason": "in_use_at_apply"})
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

    defaults = DEFAULT_PROTECTION_SOURCES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--active-link", required=True)
    parser.add_argument("--current-commit", required=True)
    parser.add_argument("--control-plane-root", default=str(defaults.control_plane_root))
    parser.add_argument("--profile-dir", default=str(defaults.profile_dir))
    parser.add_argument(
        "--standing-authorization-dir", default=str(defaults.standing_authorization_dir)
    )
    parser.add_argument("--binding-root", default=str(defaults.binding_root))
    parser.add_argument("--lease-root", default=str(defaults.lease_root))
    parser.add_argument(
        "--config-file",
        action="append",
        default=None,
        help="Configuration naming runtime paths. Repeatable; defaults to the host's two.",
    )
    parser.add_argument("--intent-root", default=str(defaults.intent_root))
    parser.add_argument("--launch-run-root", default=str(defaults.launch_run_root))
    parser.add_argument(
        "--no-migrate",
        action="store_true",
        help="Dry run: evaluate legacy bindings without writing or renewing any lease.",
    )
    parser.add_argument(
        "--proc-root",
        default="/proc",
        help="Process table used to keep every tree a live process runs from.",
    )
    parser.add_argument("--keep-last", type=int, default=DEFAULT_KEEP_LAST)
    parser.add_argument("--minimum-age-seconds", type=int, default=DEFAULT_MINIMUM_AGE_SECONDS)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--ack", default="")
    args = parser.parse_args(argv)
    sources = ProtectionSources(
        control_plane_root=Path(args.control_plane_root),
        profile_dir=Path(args.profile_dir),
        standing_authorization_dir=Path(args.standing_authorization_dir),
        binding_root=Path(args.binding_root),
        lease_root=Path(args.lease_root),
        config_files=tuple(
            Path(path) for path in (args.config_file or defaults.config_files)
        ),
        intent_root=Path(args.intent_root),
        launch_run_root=Path(args.launch_run_root),
    )
    mutating = args.apply or not args.no_migrate
    with (
        holding_publisher_locks(publisher_lock_roots(sources))
        if mutating
        else contextlib.nullcontext()
    ):
        def in_use() -> list[str]:
            return live_release_commits(
                args.release_root, args.runtime_root, proc_root=args.proc_root
            )

        protections = collect_release_protections(
            sources, now=time.time(), migrate=not args.no_migrate
        )
        plan = build_release_retirement_plan(
            release_root=args.release_root,
            runtime_root=args.runtime_root,
            active_link=args.active_link,
            current_commit=args.current_commit,
            protections=protections,
            keep_last=args.keep_last,
            minimum_age_seconds=args.minimum_age_seconds,
            in_use_commits=in_use(),
        )
        result: dict[str, Any] = (
            {
                "plan": plan,
                "receipt": apply_release_retirement_plan(
                    plan,
                    ack=args.ack,
                    active_link=args.active_link,
                    release_root=args.release_root,
                    in_use_now=lambda: set(in_use()),
                ),
            }
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
    "MAX_REPORTED_WARNINGS",
    "RUNTIME_COMPONENTS",
    "apply_release_retirement_plan",
    "build_release_retirement_plan",
    "holding_publisher_locks",
    "live_release_commits",
    "main",
    "publisher_lock_roots",
]


if __name__ == "__main__":  # pragma: no cover - exercised through module CLI
    raise SystemExit(main())
