"""Share byte-identical, read-only imported public sources without losing paths.

Historical per-release imports duplicated publisher bytes. This maintenance
operation retains every installation receipt and filename, verifies both sides,
and replaces only a duplicate physical copy with a read-only hardlink.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
import uuid

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_configuration_submission_inputs import checked_file, read, sha
from .completed_replay_cache_retention import active_reference

ACK = "share-verified-readonly-public-source-copies"
ROLES = {"appearance_3dgs", "collision_usd", "publisher_scene_usdz"}


def _state(path):
    s = path.stat()
    return {
        "device": s.st_dev,
        "inode": s.st_ino,
        "size": s.st_size,
        "mtime_ns": s.st_mtime_ns,
        "mode": s.st_mode & 0o777,
        "uid": s.st_uid,
        "gid": s.st_gid,
        "links": s.st_nlink,
    }


def plan_public_source_dedup(
    *,
    inputs_root,
    minimum_age_seconds=172800,
    minimum_size_bytes=8 * 1024**2,
    now=None,
    reference_checker=active_reference,
):
    root = Path(inputs_root)
    if not root.is_absolute() or any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("public_source_dedup_root_unsafe")
    clock = time.time() if now is None else now
    groups = {}
    for receipt_path in sorted(root.glob("*/public_scene_host_input_installation_receipt.v1.json")):
        receipt = read(receipt_path, digest_field="receipt_digest")
        if (
            receipt.get("schema_version") != "public_scene_host_input_installation_receipt.v1"
            or receipt.get("status") != "installed"
            or receipt.get("destination_root") != str(receipt_path.parent)
            or clock - receipt_path.stat().st_mtime < minimum_age_seconds
            or reference_checker(receipt_path.parent)
        ):
            continue
        for row in receipt["files"]:
            if row.get("role") not in ROLES or row.get("size_bytes", 0) < minimum_size_bytes:
                continue
            relative = Path(row["relative_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("public_source_dedup_member_unsafe")
            path = checked_file(receipt_path.parent / relative, row)
            state = _state(path)
            if state["mode"] & 0o222 or clock - path.stat().st_mtime < minimum_age_seconds:
                continue
            key = (
                row["sha256"],
                row["size_bytes"],
                state["device"],
                state["mode"],
                state["uid"],
                state["gid"],
            )
            groups.setdefault(key, []).append(
                {
                    "path": str(path),
                    "state": state,
                    "receipt_path": str(receipt_path),
                    "receipt_sha256": sha(receipt_path),
                    "sha256": row["sha256"],
                    "size_bytes": row["size_bytes"],
                }
            )
    pairs = []
    for values in groups.values():
        values.sort(key=lambda r: (-r["state"]["links"], r["path"]))
        canonical = values[0]
        for duplicate in values[1:]:
            if canonical["state"]["inode"] != duplicate["state"]["inode"]:
                pairs.append({"canonical": canonical, "duplicate": duplicate})
    result = {
        "schema_version": "public_source_storage_dedup.v1",
        "status": "dry_run",
        "inputs_root": str(root),
        "pairs": pairs,
        "source_bytes_changed": False,
        "filenames_or_receipts_removed": False,
        "potential_reclaimed_bytes": sum(
            p["duplicate"]["size_bytes"] for p in pairs if p["duplicate"]["state"]["links"] == 1
        ),
    }
    result["plan_digest"] = canonical_digest(result, digest_field="plan_digest")
    return result


def apply_public_source_dedup(plan, *, ack, reference_checker=active_reference):
    if ack != ACK or plan.get("plan_digest") != canonical_digest(plan, digest_field="plan_digest"):
        raise ValueError("public_source_dedup_plan_invalid")
    applied = []
    for pair in plan["pairs"]:
        canonical, duplicate = pair["canonical"], pair["duplicate"]
        source, target = Path(canonical["path"]), Path(duplicate["path"])
        for row, path in ((canonical, source), (duplicate, target)):
            if (
                not path.is_relative_to(Path(plan["inputs_root"]))
                or sha(Path(row["receipt_path"])) != row["receipt_sha256"]
                or reference_checker(Path(row["receipt_path"]).parent)
            ):
                raise ValueError("public_source_dedup_reference_changed")
            checked_file(path, row)
            current = _state(path)
            if any(
                current[k] != row["state"][k]
                for k in ("device", "inode", "size", "mtime_ns", "mode", "uid", "gid")
            ):
                raise ValueError("public_source_dedup_file_changed")
        nonce = uuid.uuid4().hex
        backup, replacement = (
            target.parent / (".dedup-backup-" + nonce),
            target.parent / (".dedup-link-" + nonce),
        )
        os.link(target, backup, follow_symlinks=False)
        try:
            os.link(source, replacement, follow_symlinks=False)
            os.replace(replacement, target)
            checked_file(target, duplicate)
        except BaseException:
            os.replace(backup, target)
            replacement.unlink(missing_ok=True)
            raise
        backup.unlink()
        fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        applied.append(
            {
                "path": str(target),
                "canonical_path": str(source),
                "sha256": duplicate["sha256"],
                "size_bytes": duplicate["size_bytes"],
                "prior_state": duplicate["state"],
            }
        )
    return {
        "schema_version": "public_source_storage_dedup.v1",
        "status": "applied",
        "plan_digest": plan["plan_digest"],
        "applied": applied,
        "source_bytes_changed": False,
        "filenames_or_receipts_removed": False,
    }
