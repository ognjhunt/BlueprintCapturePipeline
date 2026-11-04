"""Retire superseded per-commit release and runtime trees on their own hourly unit.

Every deploy publishes a release worktree under
``/opt/blueprint/task-evaluation-control-plane-releases/<commit>`` on the root
disk and two runtime trees under
``task-evaluation-inputs/system-runtimes/{splat-render,scene-configuration}/<commit>``
on the work volume.  Since 2026-10-01 deployment no longer retires them
(deployment authority does not authorize deletion), and the two-step operator
tool last ran on 2026-10-28.  On 2026-11-04 the host held 93 release worktrees
and 88 runtime pairs, about eleven new ones a day, and every step of the root
disk's ~3 GB/day growth landed at a deploy.

This is the separate, scheduled retirement authority
(``blueprint-control-plane-release-lifecycle.service``, hourly), sandboxed to
the release and runtime roots and its receipts. It has its own owner switch,
``BLUEPRINT_CONTROL_PLANE_RELEASE_RETIREMENT`` (on by default like every
storage GC switch; ``0`` keeps it planning), and it reuses
``control_plane_release_retirement`` unchanged: a commit's trees stay while it
is the active release, among the newest ``keep_last``, younger than a day, in
use by a live process, or held by a typed lease or configured runtime, and any
protection blocker retires nothing.  One more protection applies here: a
release worktree holding files its commit cannot reproduce (modified tracked
files, untracked files, or ignored files other than Python bytecode) is kept,
so retirement only ever removes bytes a redeploy brings back.

Restore: staging the commit again
(``scripts/stage_task_evaluation_control_plane_release.py --source-commit
<commit>``, which every deploy runs) recreates its worktree from the source
clone, pruning the stale registration first; deploying it re-provisions its
runtime trees.  Every tick that retires or sweeps anything writes an immutable
receipt naming each commit, path and the restore step, and every applying tick
rewrites ``latest-deploy-retirement.json``, whose status and alerts the
capacity controller already reads.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import secrets
import subprocess  # nosec B404 - fixed git argv over validated worktrees
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .control_plane_release_leases import (
    DEFAULT_CONTROL_PLANE_ROOT,
    DEFAULT_PROTECTION_SOURCES,
    ProtectionSources,
    collect_release_protections,
)
from .control_plane_release_retirement import (
    DEFAULT_KEEP_LAST,
    DEFAULT_LOCK_TIMEOUT_SECONDS,
    DEFAULT_MINIMUM_AGE_SECONDS,
    EXECUTE_ACK,
    LOCAL_STATE_KIND,
    RUNTIME_COMPONENTS,
    ControlPlaneReleaseRetirementError,
    apply_release_retirement_plan,
    build_release_retirement_plan,
    delete_retiring_trees,
    holding_publisher_locks,
    live_release_commits,
    publisher_lock_roots,
)
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_release_reference_lock import ReleaseReferenceLockError

RELEASE_RETIREMENT_ENV = "BLUEPRINT_CONTROL_PLANE_RELEASE_RETIREMENT"
RELEASE_RETIREMENT_INVALID = "release_retirement_setting_invalid"
RELEASE_ROOT_ENV = "BLUEPRINT_CONTROL_PLANE_GC_RELEASE_ROOT"
RELEASE_RUNTIME_ROOT_ENV = "BLUEPRINT_CONTROL_PLANE_GC_RELEASE_RUNTIME_ROOT"
RELEASE_ACTIVE_LINK_ENV = "BLUEPRINT_CONTROL_PLANE_GC_RELEASE_ACTIVE_LINK"
RELEASE_RECEIPT_ROOT_ENV = "BLUEPRINT_CONTROL_PLANE_GC_RELEASE_RECEIPT_ROOT"
DEFAULT_RELEASE_ROOT = Path("/opt/blueprint/task-evaluation-control-plane-releases")
DEFAULT_RUNTIME_ROOT = Path("/var/lib/blueprint/task-evaluation-inputs/system-runtimes")
DEFAULT_ACTIVE_LINK = Path("/opt/blueprint/task-evaluation-control-plane")
DEFAULT_RECEIPT_ROOT = DEFAULT_CONTROL_PLANE_ROOT / "release-retention"
#: The capacity controller reads this name for ``release_retirement_attention``.
SUMMARY_NAME = "latest-deploy-retirement.json"
SUMMARY_SCHEMA_VERSION = "control_plane_release_retirement_summary.v1"
RECEIPT_SCHEMA_VERSION = "control_plane_release_lifecycle_receipt.v1"
RESTORE_STEP = "stage_task_evaluation_control_plane_release"
GIT_STATUS_TIMEOUT_SECONDS = 120.0
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
# Ignored files the interpreter regenerates; anything else ignored is local state.
_REPRODUCIBLE_IGNORED = re.compile(r"(?:^|/)__pycache__(?:/|$)|\.py[co]\Z")
_TRUE = frozenset({"1", "true", "yes"})
_FALSE = frozenset({"0", "false", "no"})


def release_retirement_setting(environ: Mapping[str, str] = os.environ) -> tuple[bool, str | None]:
    """Whether scheduled release retirement applies, and an alert for an invalid setting.

    Unset or empty it is on, like every storage GC switch since 2026-09-30;
    ``0``, ``false`` or ``no`` keeps it planning. Any other value keeps it off
    and is reported; it never aborts the tick.
    """

    raw = str(environ.get(RELEASE_RETIREMENT_ENV) or "").strip().lower()
    if not raw or raw in _TRUE:
        return True, None
    if raw in _FALSE:
        return False, None
    return False, RELEASE_RETIREMENT_INVALID


def _git_status(tree: Path, *, timeout_seconds: float) -> tuple[str | None, bytes]:
    """``(HEAD, porcelain -z output)`` of one worktree, or ``(None, b"")``. Takes no lock."""

    base = ["git", "-c", "safe.directory=*", "-C", str(tree), "--no-optional-locks"]
    try:
        head = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
            [*base, "rev-parse", "--verify", "HEAD"],
            capture_output=True, check=False, timeout=timeout_seconds,
        )
        if head.returncode != 0:
            return None, b""
        status = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
            [*base, "status", "--porcelain=v1", "-z", "--untracked-files=all",
             "--ignored=matching", "--ignore-submodules=none"],
            capture_output=True, check=False, timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError):
        return None, b""
    if status.returncode != 0:
        return None, b""
    return head.stdout.decode("ascii", "replace").strip(), status.stdout


def release_worktree_local_state(
    release_root: str | Path, commits: Sequence[str], *,
    timeout_seconds: float = GIT_STATUS_TIMEOUT_SECONDS,
) -> dict[str, str]:
    """Commits whose release worktree holds bytes the commit cannot reproduce, and why.

    ``status_unavailable`` (not a worktree, git failed or timed out) and
    ``head_mismatch`` are local state too: what cannot be proven reproducible
    is kept. Ignored ``__pycache__`` trees and ``.pyc`` files are not.
    """

    states: dict[str, str] = {}
    for commit in commits:
        head, output = _git_status(Path(release_root) / commit, timeout_seconds=timeout_seconds)
        if head is None:
            states[commit] = "status_unavailable"
            continue
        if head != commit:
            states[commit] = "head_mismatch"
            continue
        entries = output.split(b"\0")
        local = 0
        index = 0
        while index < len(entries):
            entry = entries[index]
            index += 1
            if len(entry) < 4:
                continue
            code, name = entry[:2], entry[3:].decode("utf-8", "surrogateescape")
            if code[:1] in (b"R", b"C"):
                index += 1  # -z puts a rename's source path in the next entry
            if code == b"!!" and _REPRODUCIBLE_IGNORED.search(name):
                continue
            local += 1
        if local:
            states[commit] = "local_changes"
    return states


def _aged_release_commits(release_root: Path, *, now: float, minimum_age_seconds: int) -> list[str]:
    """Release worktrees old enough to be candidates; younger ones are kept by age anyway."""

    try:
        children = sorted(release_root.iterdir())
    except FileNotFoundError:
        return []
    commits = []
    for child in children:
        if _COMMIT.fullmatch(child.name) is None or child.is_symlink() or not child.is_dir():
            continue
        if now - child.lstat().st_mtime >= minimum_age_seconds:
            commits.append(child.name)
    return commits


def _active_commit(active_link: Path) -> str | None:
    try:
        name = Path(os.readlink(active_link)).name
    except OSError:
        return None
    return name if _COMMIT.fullmatch(name) else None


def _write_public(path: Path, document: Mapping[str, Any], *, exclusive: bool) -> None:
    """Write a 0644 JSON document; ``exclusive`` never replaces an existing receipt."""

    directory = path.parent
    if directory.is_symlink() or not directory.is_dir():
        raise OSError("release retention directory unsafe")
    payload = (json.dumps(document, indent=1, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fchmod(stream.fileno(), 0o644)
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def _counts_by_kind(plan: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """``{kind: {count, bytes}}`` of kept commits; sizes are not walked under the locks."""

    by_kind = (plan or {}).get("protected_by_kind") or {}
    return {str(kind): {"count": len(commits), "bytes": None} for kind, commits in sorted(by_kind.items())}


def run_release_lifecycle(
    *,
    release_root: str | Path = DEFAULT_RELEASE_ROOT,
    runtime_root: str | Path = DEFAULT_RUNTIME_ROOT,
    active_link: str | Path = DEFAULT_ACTIVE_LINK,
    receipt_root: str | Path = DEFAULT_RECEIPT_ROOT,
    sources: ProtectionSources = DEFAULT_PROTECTION_SOURCES,
    apply: bool,
    enabled: bool,
    now: Callable[[], float] = time.time,
    proc_root: str | Path = "/proc",
    keep_last: int = DEFAULT_KEEP_LAST,
    minimum_age_seconds: int = DEFAULT_MINIMUM_AGE_SECONDS,
    lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
    local_state: Callable[[Path, Sequence[str]], Mapping[str, str]] = release_worktree_local_state,
    setting_alert: str | None = None,
) -> dict[str, Any]:
    """One tick: plan, and when applying and enabled retire, superseded release trees.

    Returns the tick's entry (``status``, ``enabled``,
    ``candidate_bytes``, ``retired_bytes``, ``retained_by_reason``, ...).
    Leftovers of an interrupted retirement are swept first. Without ``apply``
    it writes nothing and migrates no lease; with ``apply`` but not
    ``enabled`` it only plans and records why in the summary.
    """

    releases, runtimes, active = Path(release_root), Path(runtime_root), Path(active_link)
    staged_roots = [releases, *(runtimes / component for component in RUNTIME_COMPONENTS)]
    mutating = apply and enabled
    observed_at = float(now())
    swept: dict[str, Any] = {"deleted": [], "deleted_bytes": 0, "shared_bytes": 0, "failed": []}
    if mutating:
        swept = delete_retiring_trees(staged_roots)
    # Git runs outside the publishers' locks: nothing writes into a superseded tree.
    local = dict(local_state(releases, _aged_release_commits(
        releases, now=observed_at, minimum_age_seconds=minimum_age_seconds)))
    current = _active_commit(active)
    plan: dict[str, Any] | None = None
    receipt: dict[str, Any] | None = None
    blockers: list[str] = []
    alerts: list[str] = [setting_alert] if setting_alert else []

    def in_use() -> list[str]:
        return live_release_commits(releases, runtimes, proc_root=proc_root)

    try:
        if current is None:
            raise ControlPlaneReleaseRetirementError("release_retirement_active_link_invalid")
        with (holding_publisher_locks(publisher_lock_roots(sources), timeout_seconds=lock_timeout_seconds)
              if mutating else contextlib.nullcontext()):
            protections = collect_release_protections(sources, now=observed_at, migrate=mutating)
            plan = build_release_retirement_plan(
                release_root=releases, runtime_root=runtimes, active_link=active,
                current_commit=current, protections=protections, keep_last=keep_last,
                minimum_age_seconds=minimum_age_seconds, now=lambda: observed_at,
                in_use_commits=in_use(), measure_sizes=False, local_state_commits=local,
            )
            blockers = list(plan["blockers"])
            if mutating and plan["status"] == "dry_run":
                receipt = apply_release_retirement_plan(
                    plan, ack=EXECUTE_ACK, active_link=active, release_root=releases,
                    in_use_now=lambda: set(in_use()),
                )
    except (ControlPlaneReleaseRetirementError, ReleaseReferenceLockError, ValueError) as exc:
        blockers = [str(exc) or type(exc).__name__]
    except Exception as exc:  # noqa: BLE001 - keep the partial receipt, never lose the tick
        partial = getattr(exc, "release_retirement_receipt", None)
        receipt = dict(partial) if isinstance(partial, Mapping) else None
        blockers = [f"release_retirement_failed:{type(exc).__name__}"]
    # Outside every lock: delete, and measure, what apply moved aside.
    deletion = delete_retiring_trees(staged_roots) if receipt is not None else {
        "deleted": [], "deleted_bytes": 0, "shared_bytes": 0, "failed": []}
    renamed = list((receipt or {}).get("renamed") or [])
    fallback = list((receipt or {}).get("direct_delete_fallback") or [])
    skipped = list((receipt or {}).get("skipped") or [])
    retired_commits = sorted({str(row["commit"]) for row in (*renamed, *fallback)})
    retired_bytes = (deletion["deleted_bytes"] + swept["deleted_bytes"]
                     + sum(int(row["bytes"]) for row in fallback))
    failures = [*swept["failed"], *deletion["failed"]]
    if blockers:
        alerts.append(f"release_retirement_blocked:{blockers[0]}")
    if failures:
        alerts.append(f"release_retirement_deletion_failed:{len(failures)}")
    if any(str(row.get("reason", "")).startswith(("rename_failed:", "direct_delete_failed:"))
           for row in skipped):
        alerts.append("release_retirement_rename_failed")
    if blockers:
        status = "blocked"
    elif not apply:
        status = "dry_run"
    elif not enabled:
        status = "skipped"
    else:
        status = "applied"
    candidates = list((plan or {}).get("candidates") or [])
    entry: dict[str, Any] = {
        "status": status,
        "enabled": enabled,
        "active_commit": current,
        "keep_last": keep_last,
        "minimum_age_seconds": minimum_age_seconds,
        "candidate_count": len(candidates),
        # Sizes are measured as they are deleted, never walked under the locks.
        "candidate_bytes": retired_bytes if receipt is not None else None,
        "retired_bytes": retired_bytes,
        "shared_bytes": deletion["shared_bytes"] + swept["shared_bytes"],
        "retired_commits": retired_commits,
        "retained_by_reason": _counts_by_kind(plan),
        "local_state_commits": dict(sorted(local.items())),
        "unmanaged_children": list((plan or {}).get("unmanaged_children") or []),
        "skipped": skipped,
        "blockers": blockers,
        "alerts": alerts,
        "plan_digest": (plan or {}).get("plan_digest"),
    }
    if not mutating and not blockers:
        entry["planned_commits"] = [str(row["commit"]) for row in candidates]
    if not apply:
        return entry
    receipt_directory = Path(receipt_root)
    if retired_commits or swept["deleted"] or failures:
        document = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "generated_at_epoch": observed_at,
            "active_commit": current,
            "plan_digest": entry["plan_digest"],
            "apply_receipt_digest": (receipt or {}).get("result_digest"),
            "retired": [
                {"commit": commit,
                 "paths": sorted(str(row.get("path")) for row in (*renamed, *fallback)
                                 if row["commit"] == commit),
                 "restore": {"step": RESTORE_STEP, "source_commit": commit,
                             "runtime_trees": "redeploy_commit"}}
                for commit in retired_commits
            ],
            "deleted": deletion["deleted"],
            "direct_delete_fallback": fallback,
            "swept": swept["deleted"],
            "deletion_failures": failures,
            "skipped": skipped,
            "retired_bytes": retired_bytes,
            "shared_bytes": entry["shared_bytes"],
            "retained_by_reason": entry["retained_by_reason"],
            "local_state_commits": entry["local_state_commits"],
            "evidence_roots_touched": False,
            "receipt_digest": "",
        }
        document["receipt_digest"] = canonical_digest(document, digest_field="receipt_digest")
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(observed_at))
        path = receipt_directory / f"release-lifecycle-{stamp}-{secrets.token_hex(4)}.json"
        try:
            _write_public(path, document, exclusive=True)
            entry["receipt"] = str(path)
        except OSError as exc:
            entry["receipt"] = None
            alerts.append(f"release_retirement_receipt_unwritten:{type(exc).__name__}")
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at_epoch": observed_at,
        "scheduler": "release_lifecycle_unit",
        "source_commit": current,
        "status": status,
        "reason": None if enabled else "release_retirement_opted_out",
        "alerts": alerts,
        "blockers": blockers,
        "retired_commits": retired_commits,
        "retired_bytes": retired_bytes,
        "protected_by_kind": {kind: row["count"] for kind, row in entry["retained_by_reason"].items()},
        "local_state_commit_count": len(local),
        "candidate_count": len(candidates),
        "receipt": entry.get("receipt"),
    }
    try:
        _write_public(receipt_directory / SUMMARY_NAME, summary, exclusive=False)
    except OSError as exc:
        alerts.append(f"release_retirement_summary_unwritten:{type(exc).__name__}")
    return entry


def main(argv: Sequence[str] | None = None) -> int:
    """``run [--apply --ack ...]``: one tick of the scheduled unit; prints its JSON entry.

    Without ``--apply`` it plans and writes nothing. ``--apply`` needs the
    retirement acknowledgement; the owner switch still decides whether it
    retires or only plans. Exits 1 only when the tick itself failed.
    """

    import argparse

    parser = argparse.ArgumentParser(prog="control_plane_release_lifecycle")
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--release-root", default=os.getenv(RELEASE_ROOT_ENV) or str(DEFAULT_RELEASE_ROOT))
    parser.add_argument("--runtime-root", default=os.getenv(RELEASE_RUNTIME_ROOT_ENV) or str(DEFAULT_RUNTIME_ROOT))
    parser.add_argument("--active-link", default=os.getenv(RELEASE_ACTIVE_LINK_ENV) or str(DEFAULT_ACTIVE_LINK))
    parser.add_argument("--receipt-root", default=os.getenv(RELEASE_RECEIPT_ROOT_ENV) or str(DEFAULT_RECEIPT_ROOT))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--ack", default="")
    args = parser.parse_args(argv)
    if args.apply and args.ack != EXECUTE_ACK:
        print(json.dumps({"status": "refused", "reason": "release_lifecycle_apply_not_authorized"}))
        return 2
    enabled, alert = release_retirement_setting()
    entry = run_release_lifecycle(
        release_root=args.release_root, runtime_root=args.runtime_root, active_link=args.active_link,
        receipt_root=args.receipt_root, apply=args.apply, enabled=enabled, setting_alert=alert,
    )
    print(json.dumps(entry, indent=2, sort_keys=True))
    return 1 if str(entry.get("status")) == "blocked" and any(
        str(code).startswith("release_retirement_failed:") for code in entry.get("blockers") or ()) else 0


__all__ = [
    "DEFAULT_ACTIVE_LINK",
    "DEFAULT_RECEIPT_ROOT",
    "DEFAULT_RELEASE_ROOT",
    "DEFAULT_RUNTIME_ROOT",
    "LOCAL_STATE_KIND",
    "RECEIPT_SCHEMA_VERSION",
    "RELEASE_ACTIVE_LINK_ENV",
    "RELEASE_RECEIPT_ROOT_ENV",
    "RELEASE_RETIREMENT_ENV",
    "RELEASE_RETIREMENT_INVALID",
    "RELEASE_ROOT_ENV",
    "RELEASE_RUNTIME_ROOT_ENV",
    "SUMMARY_NAME",
    "release_retirement_setting",
    "release_worktree_local_state",
    "run_release_lifecycle",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - exercised through the module CLI
    raise SystemExit(main())
