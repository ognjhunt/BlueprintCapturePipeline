"""Fail-closed retention for restageable Task Evaluation Git worktrees."""

from __future__ import annotations

import argparse
import json
import re
import subprocess  # nosec B404 - fixed Git argv over validated worktrees
from pathlib import Path
from typing import Any, Mapping, Sequence

from .task_evaluation_release_retention import (
    ReleaseRetentionError,
    _absolute_path,
    _artifact_snapshot,
    _assert_snapshot_current,
    _canonical_digest,
    _valid_commit,
    _write_exclusive,
)


SCHEMA_VERSION = "task_evaluation_ephemeral_checkout_retention_plan.v1"
APPLY_SCHEMA_VERSION = "task_evaluation_ephemeral_checkout_retention_apply.v1"
APPLY_ACKNOWLEDGEMENT = "reap-restageable-task-evaluation-checkouts"


def _git(path: Path, *args: str) -> str:
    try:
        completed = subprocess.run(  # nosec B603
            ["git", "-c", f"safe.directory={path}", "-C", str(path), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except OSError as exc:
        raise ReleaseRetentionError(
            "ephemeral_checkout_retention_git_unavailable"
        ) from exc
    if completed.returncode != 0:
        raise ReleaseRetentionError(
            "ephemeral_checkout_retention_git_probe_failed:"
            + path.name
        )
    return completed.stdout.strip()


def _remove_checkout(path: Path, *, commit: str) -> None:
    git_marker = path / ".git"
    if git_marker.is_symlink() or not git_marker.is_file():
        raise ReleaseRetentionError(
            f"ephemeral_checkout_retention_not_git_worktree:{commit}"
        )
    if _git(path, "rev-parse", "--verify", "HEAD^{commit}").lower() != commit:
        raise ReleaseRetentionError(
            f"ephemeral_checkout_retention_head_mismatch:{commit}"
        )
    if _git(path, "status", "--porcelain", "--untracked-files=all"):
        raise ReleaseRetentionError(
            f"ephemeral_checkout_retention_checkout_dirty:{commit}"
        )
    try:
        completed = subprocess.run(  # nosec B603
            [
                "git",
                "-c",
                f"safe.directory={path}",
                "-C",
                str(path),
                "worktree",
                "remove",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except OSError as exc:
        raise ReleaseRetentionError(
            f"ephemeral_checkout_retention_remove_failed:{commit}"
        ) from exc
    if completed.returncode != 0 or path.exists():
        raise ReleaseRetentionError(
            f"ephemeral_checkout_retention_remove_failed:{commit}"
        )


def _checkout_snapshot(path: Path, *, root: Path) -> dict[str, Any]:
    commit = _valid_commit(path.name)
    if commit is None or path.parent != root:
        raise ReleaseRetentionError(
            f"ephemeral_checkout_retention_child_invalid:{path.name}"
        )
    artifact = _artifact_snapshot(
        path, kind="restageable_git_checkout", commit=commit
    )
    if _git(path, "rev-parse", "--verify", "HEAD^{commit}").lower() != commit:
        raise ReleaseRetentionError(
            f"ephemeral_checkout_retention_head_mismatch:{commit}"
        )
    if _git(path, "status", "--porcelain", "--untracked-files=all"):
        raise ReleaseRetentionError(
            f"ephemeral_checkout_retention_checkout_dirty:{commit}"
        )
    refs = sorted(
        line.strip()
        for line in _git(path, "branch", "-r", "--contains", commit).splitlines()
        if line.strip()
    )
    if not refs or any(
        re.fullmatch(r"origin/[A-Za-z0-9][A-Za-z0-9._/-]*", ref) is None
        for ref in refs
    ):
        raise ReleaseRetentionError(
            f"ephemeral_checkout_retention_remote_ref_missing:{commit}"
        )
    return {**artifact, "remote_refs": refs, "restageable_from_remote": True}


def build_ephemeral_checkout_retention_plan(
    *, checkout_roots: Sequence[str | Path], keep_commits: Sequence[str] = ()
) -> dict[str, Any]:
    if not checkout_roots:
        raise ReleaseRetentionError(
            "ephemeral_checkout_retention_roots_missing"
        )
    roots = tuple(
        _absolute_path(value, field="checkout_root").resolve()
        for value in checkout_roots
    )
    if len(set(roots)) != len(roots) or any(
        left in right.parents or right in left.parents
        for index, left in enumerate(roots)
        for right in roots[index + 1 :]
    ):
        raise ReleaseRetentionError(
            "ephemeral_checkout_retention_roots_overlap"
        )
    keep: set[str] = set()
    for value in keep_commits:
        commit = _valid_commit(value)
        if commit is None:
            raise ReleaseRetentionError(
                "ephemeral_checkout_retention_keep_commit_invalid"
            )
        keep.add(commit)

    eligible: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    observed: set[str] = set()
    for root in roots:
        if root.is_symlink() or not root.is_dir():
            raise ReleaseRetentionError(
                f"ephemeral_checkout_retention_root_invalid:{root}"
            )
        for child in sorted(root.iterdir(), key=lambda item: item.name):
            snapshot = _checkout_snapshot(child, root=root)
            commit = str(snapshot["source_commit"])
            if commit in observed:
                raise ReleaseRetentionError(
                    f"ephemeral_checkout_retention_commit_duplicate:{commit}"
                )
            observed.add(commit)
            row = {
                "source_commit": commit,
                "artifact": snapshot,
                "size_bytes": snapshot["size_bytes"],
            }
            if commit in keep:
                retained.append({**row, "retention_reasons": ["operator_pin"]})
            else:
                eligible.append(row)
    if not keep.issubset(observed):
        raise ReleaseRetentionError(
            "ephemeral_checkout_retention_keep_commit_missing"
        )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "dry_run",
        "checkout_roots": sorted(str(root) for root in roots),
        "operator_keep_commits": sorted(keep),
        "eligible_checkouts": eligible,
        "retained_checkouts": retained,
        "predicted_removed_bytes": sum(row["size_bytes"] for row in eligible),
        "remote_restage_required": True,
        "evidence_artifacts_removed": False,
        "provider_mutation_performed": False,
        "production_artifact_mutation_performed": False,
    }
    result["plan_digest"] = _canonical_digest(
        result, digest_field="plan_digest"
    )
    return result


def _read_plan(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReleaseRetentionError(
            "ephemeral_checkout_retention_plan_invalid"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseRetentionError(
            "ephemeral_checkout_retention_plan_invalid"
        ) from exc
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != "dry_run"
        or value.get("plan_digest")
        != _canonical_digest(value, digest_field="plan_digest")
    ):
        raise ReleaseRetentionError(
            "ephemeral_checkout_retention_plan_invalid"
        )
    return dict(value)


def apply_ephemeral_checkout_retention_plan(
    *, dry_run_plan_path: str | Path, acknowledgement: str, receipt_out: str | Path
) -> dict[str, Any]:
    if acknowledgement != APPLY_ACKNOWLEDGEMENT:
        raise ReleaseRetentionError(
            "ephemeral_checkout_retention_acknowledgement_missing"
        )
    plan_path = _absolute_path(dry_run_plan_path, field="dry_run_plan")
    output = _absolute_path(receipt_out, field="receipt_out")
    if plan_path.resolve() == output.resolve():
        raise ReleaseRetentionError(
            "ephemeral_checkout_retention_receipt_overlaps_plan"
        )
    plan = _read_plan(plan_path)
    current = build_ephemeral_checkout_retention_plan(
        checkout_roots=tuple(plan.get("checkout_roots") or ()),
        keep_commits=tuple(plan.get("operator_keep_commits") or ()),
    )
    if current.get("plan_digest") != plan.get("plan_digest"):
        raise ReleaseRetentionError(
            "ephemeral_checkout_retention_plan_changed"
        )
    targets: list[tuple[dict[str, Any], Path]] = []
    for row in plan.get("eligible_checkouts") or []:
        if not isinstance(row, Mapping) or not isinstance(
            row.get("artifact"), Mapping
        ):
            raise ReleaseRetentionError(
                "ephemeral_checkout_retention_plan_invalid"
            )
        artifact = dict(row["artifact"])
        targets.append((artifact, _assert_snapshot_current(artifact)))
    removed: list[dict[str, Any]] = []
    for artifact, path in targets:
        commit = str(artifact["source_commit"])
        _checkout_snapshot(path, root=path.parent)
        _remove_checkout(path, commit=commit)
        removed.append(
            {
                "source_commit": commit,
                "path": str(path),
                "removed_bytes": artifact["size_bytes"],
            }
        )
    result: dict[str, Any] = {
        "schema_version": APPLY_SCHEMA_VERSION,
        "status": "applied",
        "dry_run_plan_path": str(plan_path.resolve()),
        "dry_run_plan_digest": plan["plan_digest"],
        "removed": removed,
        "removed_bytes": sum(row["removed_bytes"] for row in removed),
        "predicted_removed_bytes": plan["predicted_removed_bytes"],
        "operator_keep_commits": plan["operator_keep_commits"],
        "evidence_artifacts_removed": False,
        "provider_mutation_performed": False,
        "production_artifact_mutation_performed": True,
    }
    result["receipt_digest"] = _canonical_digest(
        result, digest_field="receipt_digest"
    )
    _write_exclusive(output, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout-root", action="append")
    parser.add_argument("--keep-commit", action="append", default=[])
    parser.add_argument("--receipt-out", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--dry-run-plan")
    parser.add_argument("--ack")
    args = parser.parse_args(argv)
    try:
        if args.apply:
            result = apply_ephemeral_checkout_retention_plan(
                dry_run_plan_path=str(args.dry_run_plan or ""),
                acknowledgement=str(args.ack or ""),
                receipt_out=args.receipt_out,
            )
        else:
            if args.dry_run_plan or args.ack:
                raise ReleaseRetentionError(
                    "ephemeral_checkout_retention_dry_run_argument_conflict"
                )
            result = build_ephemeral_checkout_retention_plan(
                checkout_roots=tuple(args.checkout_root or ()),
                keep_commits=tuple(args.keep_commit),
            )
            _write_exclusive(_absolute_path(args.receipt_out, field="receipt_out"), result)
    except (OSError, ReleaseRetentionError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "blockers": [str(exc)]}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "APPLY_ACKNOWLEDGEMENT",
    "apply_ephemeral_checkout_retention_plan",
    "build_ephemeral_checkout_retention_plan",
]
