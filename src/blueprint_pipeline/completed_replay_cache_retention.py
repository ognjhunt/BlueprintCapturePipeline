"""Reclaim disposable binary copies after an offline replay has completed.

Reports, logs, source code, readonly files and shared inodes remain untouched.
This module is also a standalone maintenance entrypoint; it never allocates a
provider or changes the scientific release used by a live run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import time
from pathlib import Path

SCHEMA = "completed_offline_replay_cache_retention.v1"
ACK = "reclaim-completed-offline-replay-caches"
BINARY_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".ply",
    ".bin",
    ".npy",
    ".npz",
    ".pt",
    ".zip",
    ".whl",
    ".usd",
    ".usda",
    ".usdc",
    ".usdz",
    ".nurec",
    ".so",
}


def digest(value):
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def file_sha(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def completed_report(root):
    for name in ("stage_replay_report.v1.json", "replay_report.json", "replay.json", "report.json"):
        path = root / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 4 * 1024**2:
            continue
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(value, dict):
            continue
        parent = (
            value.get("schema_version") == "task_evaluation_parent_replay_report.v1"
            and value.get("nothing_fetched") is True
            and value.get("paid_execution_requested") is False
            and value.get("provider_mutation_performed") is False
        )
        legacy = (
            value.get("status") == "repair_inputs_and_semantic_request_replay_passed"
            and all(
                value.get(k) is False
                for k in (
                    "model_inference_performed",
                    "network_fetch_performed",
                    "provider_mutation_performed",
                )
            )
        ) or (
            value.get("status") == "derived_method_inputs_materialized"
            and value.get("diagnostic_only") is True
            and value.get("provider_calls") is False
        )
        legacy = legacy or (
            value.get("status") == "offline_repair_targets_materialized"
            and value.get("new_gpu_allocations") == 0
            and value.get("new_model_calls") == 0
            and value.get("appearance_qualified") is False
        )
        if parent or legacy:
            return path
    return None


def active_reference(root, *, process_root=Path("/proc"), ignored_process_ids=()):
    """Read process references without ever returning or storing environment values."""
    if not process_root.is_dir():
        raise ValueError("replay_cache_process_inventory_unavailable")
    needle = str(root).encode()
    for process in process_root.iterdir():
        if not process.name.isdigit():
            continue
        if int(process.name) in ignored_process_ids:
            continue
        for name in ("cmdline", "environ"):
            try:
                if needle in (process / name).read_bytes():
                    return True
            except (FileNotFoundError, ProcessLookupError):
                continue
        try:
            descriptors = list((process / "fd").iterdir())
        except (FileNotFoundError, ProcessLookupError):
            continue
        for descriptor in (process / "cwd", *descriptors):
            try:
                target = os.readlink(descriptor)
            except (FileNotFoundError, ProcessLookupError):
                continue
            if target == str(root) or target.startswith(str(root) + "/"):
                return True
    return False


def plan_replay_cache_retention(
    *, replay_root, minimum_closed_seconds=60, now=None, process_root=Path("/proc")
):
    root = Path(replay_root)
    if not root.is_absolute() or any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("replay_cache_root_unsafe")
    if type(minimum_closed_seconds) is not int or minimum_closed_seconds < 0:
        raise ValueError("replay_cache_age_invalid")
    clock = time.time() if now is None else now
    rows, kept = [], []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.is_symlink():
            continue
        report = completed_report(child)
        if report is None or clock - report.stat().st_mtime < minimum_closed_seconds:
            continue
        if active_reference(child, process_root=process_root):
            kept.append({"root": str(child), "reason": "active_reference"})
            continue
        files = []
        for path in child.rglob("*"):
            info = path.lstat()
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or not info.st_mode & 0o222
                or info.st_size < 64 * 1024
                or info.st_mtime_ns > report.stat().st_mtime_ns
                or path.suffix.lower() not in BINARY_SUFFIXES
                or any(p.is_symlink() for p in path.parents if p != child.parent)
            ):
                continue
            files.append(
                {
                    "relative_path": str(path.relative_to(child)),
                    "inode": info.st_ino,
                    "mtime_ns": info.st_mtime_ns,
                    "size_bytes": info.st_size,
                    "sha256": file_sha(path),
                }
            )
        if files:
            rows.append(
                {
                    "root": str(child),
                    "report_path": str(report),
                    "report_sha256": file_sha(report),
                    "files": files,
                }
            )
    plan = {
        "schema_version": SCHEMA,
        "status": "dry_run",
        "observed_at_epoch": clock,
        "replay_root": str(root),
        "rows": rows,
        "kept": kept,
        "candidate_bytes": sum(f["size_bytes"] for r in rows for f in r["files"]),
        "reports_and_original_evidence_removed": False,
    }
    plan["plan_digest"] = digest(plan)
    return plan


def apply_replay_cache_retention(plan, *, ack, process_root=Path("/proc")):
    if ack != ACK or plan.get("plan_digest") != digest(
        {k: v for k, v in plan.items() if k != "plan_digest"}
    ):
        raise ValueError("replay_cache_plan_invalid")
    removed, skipped = [], []
    base = Path(plan["replay_root"])
    for row in plan["rows"]:
        root = Path(row["root"])
        if root.parent != base or any(p.is_symlink() for p in (root, *root.parents)):
            raise ValueError("replay_cache_root_changed")
        report = completed_report(root)
        if (
            report is None
            or str(report) != row["report_path"]
            or file_sha(report) != row["report_sha256"]
        ):
            skipped.append({"root": str(root), "reason": "report_changed"})
            continue
        if active_reference(root, process_root=process_root):
            skipped.append({"root": str(root), "reason": "active_reference"})
            continue
        for item in row["files"]:
            relative = Path(item["relative_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("replay_cache_member_unsafe")
            path = root / relative
            info = path.lstat()
            if (
                not stat.S_ISREG(info.st_mode)
                or path.suffix.lower() not in BINARY_SUFFIXES
                or info.st_size < 64 * 1024
                or any(p.is_symlink() for p in path.parents if p != root.parent)
                or info.st_nlink != 1
                or not info.st_mode & 0o222
                or info.st_ino != item["inode"]
                or info.st_mtime_ns != item["mtime_ns"]
                or info.st_size != item["size_bytes"]
                or file_sha(path) != item["sha256"]
            ):
                skipped.append({"path": str(path), "reason": "file_changed"})
                continue
            path.unlink()
            removed.append(
                {"path": str(path), "sha256": item["sha256"], "size_bytes": item["size_bytes"]}
            )
    return {
        "schema_version": SCHEMA,
        "status": "applied",
        "plan_digest": plan["plan_digest"],
        "removed_bytes": sum(r["size_bytes"] for r in removed),
        "removed": removed,
        "skipped": skipped,
        "reports_and_original_evidence_removed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", required=True)
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--minimum-closed-seconds", type=int, default=60)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--ack", default="")
    args = parser.parse_args()
    plan = plan_replay_cache_retention(
        replay_root=args.replay_root, minimum_closed_seconds=args.minimum_closed_seconds
    )
    output = Path(args.report_root)
    output.mkdir(parents=True, exist_ok=True)
    key = plan["plan_digest"][7:]
    (output / (key + "-plan.json")).write_text(json.dumps(plan, indent=2) + "\n")
    result = apply_replay_cache_retention(plan, ack=args.ack) if args.apply else plan
    (output / (key + "-result.json")).write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                in {
                    "status",
                    "candidate_bytes",
                    "removed_bytes",
                    "reports_and_original_evidence_removed",
                }
            }
        )
    )


if __name__ == "__main__":
    main()
