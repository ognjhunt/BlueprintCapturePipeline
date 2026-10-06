"""Durable website denial and explicitly authorized, exact local cleanup.

The HTTP handoff never deletes bytes. Operator cleanup uses a retained plan and
does not assert that cloud storage, provider copies or delivered copies vanished.
ADP-010/day-14: immutable capture rights remain authoritative after withdrawal.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .capture_lifecycle import _sha256_file, _write_once
from .decision_evidence_contracts import canonical_digest


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("website_withdrawal_artifact_invalid")  # noqa: TRY004 - typed refusal contract
    return value


def _site(root: Path) -> Path:
    root = Path(root).absolute()
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError("website_withdrawal_root_symlink")
    if root.parent.name != "captures" or not root.parent.parent.name.startswith("site-"):
        raise ValueError("website_withdrawal_root_invalid")
    request_id = root.parent.parent.name[5:]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}", request_id) or root.name != f"walkthrough-{request_id}":
        raise ValueError("website_withdrawal_root_binding_mismatch")
    return root.parent.parent


@contextmanager
def _lock(site: Path):
    journal = site / "website_withdrawal"
    if journal.is_symlink():
        raise ValueError("website_withdrawal_journal_symlink")
    journal.mkdir(parents=True, exist_ok=True)
    lock_path = journal / "lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield journal


def _binding(command: Mapping[str, Any], root: Path) -> dict[str, Any]:
    site = _site(root)
    request_id = site.name[5:]
    expected = {"request_id": request_id, "scene_id": site.name, "capture_id": root.name}
    if set(command) != {"schema_version", "request_id", "scene_id", "capture_id", "withdrawal_id", "requested_at_iso"}:
        raise ValueError("website_withdrawal_command_invalid")
    if command.get("schema_version") != "website_capture_withdrawal.v1" or any(command.get(k) != v for k, v in expected.items()):
        raise ValueError("website_withdrawal_identity_mismatch")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}", str(command.get("withdrawal_id", ""))):
        raise ValueError("website_withdrawal_id_invalid")
    from .capture_lifecycle import _parse_time
    _parse_time(command.get("requested_at_iso"), code="website_withdrawal_timestamp_invalid")
    manifest_path = root / "raw/manifest.json"
    if manifest_path.exists():
        manifest = _read(manifest_path)
        if any(manifest.get(key) != value for key, value in (("site_submission_id", request_id), ("scene_id", site.name), ("capture_id", root.name))):
            raise ValueError("website_withdrawal_manifest_binding_mismatch")
    return dict(command)


def _files(site: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(site.rglob("*")):
        relative = path.relative_to(site)
        if relative.parts[0] == "website_withdrawal":
            continue
        if path.is_symlink():
            raise ValueError("website_cleanup_symlink_forbidden")
        if path.is_file():
            stat = path.stat()
            files.append({"path": str(relative), "sha256": _sha256_file(path), "size": stat.st_size,
                          "device": stat.st_dev, "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns})
    return files


def _inspection(site: Path, journal: Path) -> dict[str, Any]:
    tombstone = _read(journal / "tombstone.json")
    if tombstone.get("digest") != canonical_digest(tombstone, digest_field="digest"):
        raise ValueError("website_withdrawal_tombstone_changed")
    verified_path = journal / "local_cleanup_verified.json"
    verified = _read(verified_path) if verified_path.exists() else None
    if verified and verified.get("digest") != canonical_digest(verified, digest_field="digest"):
        raise ValueError("website_cleanup_verification_changed")
    hold_path = journal / "legal_hold.json"
    held = hold_path.exists() and _read(hold_path).get("legal_hold") is True
    local_complete = bool(verified and not _files(site))
    cloud_path = journal / "cloud_object_absence_verified.json"
    cloud = _read(cloud_path) if cloud_path.exists() else None
    if cloud and cloud.get("digest") != canonical_digest(cloud, digest_field="digest"):
        raise ValueError("website_cleanup_cloud_receipt_changed")
    value = {"schema_version": "website_capture_withdrawal_receipt.v1",
             **{key: tombstone[key] for key in ("request_id", "scene_id", "capture_id", "withdrawal_id")},
             "command_digest": tombstone["command_digest"], "tombstone_digest": tombstone["digest"],
             "state": "cleanup_retained_legal_hold" if held else "local_cleanup_verified_external_pending" if local_complete
                 else "pipeline_acknowledged_cleanup_pending",
             "pipeline_acknowledged": True, "serve_allowed": False, "future_processing_allowed": False,
             "local_cleanup_verified": local_complete, "local_cleanup_receipt_digest": verified.get("digest") if local_complete else None,
             "deletion_confirmed": False, "provider_acknowledgement": "unknown", "cloud_storage_cleanup_verified": False,
             "delivered_copy_cleanup_verified": False, "retained_audit_records": True, "legal_hold": held,
             "cloud_object_absence_verified": bool(cloud), "cloud_object_cleanup_receipt_digest": cloud.get("digest") if cloud else None,
             "cleanup_scope": "server_mapped_site_local_files", "external_cleanup_required": True}
    value["digest"] = canonical_digest(value, digest_field="digest")
    return value


class GoogleWebsiteCleanupStorage:
    """Operator-configured existing bucket. Generation preconditions are mandatory."""
    def __init__(self, bucket_name: str):
        from google.cloud import storage
        self.bucket_name = bucket_name
        self.client = storage.Client()

    def objects(self, prefix: str) -> list[dict[str, Any]]:
        return [{"name": blob.name, "generation": str(blob.generation), "size": blob.size, "crc32c": blob.crc32c}
                for blob in self.client.list_blobs(self.bucket_name, prefix=prefix, versions=True, timeout=30)]

    def soft_deleted_objects(self, prefix: str) -> list[dict[str, Any]]:
        # Unknown/unsupported acknowledgement raises; it never verifies cleanup.
        return [{"name": blob.name, "generation": str(blob.generation)}
                for blob in self.client.list_blobs(self.bucket_name, prefix=prefix, soft_deleted=True, timeout=30)]

    def delete(self, name: str, generation: str) -> None:
        self.client.bucket(self.bucket_name).blob(name, generation=int(generation)).delete(if_generation_match=int(generation), timeout=30)


def _cloud_objects(site: Path, storage: Any) -> list[dict[str, Any]]:
    prefix = f"scenes/{site.name}/"
    rows = storage.objects(prefix)
    for row in rows:
        if (not isinstance(row.get("name"), str) or not row["name"].startswith(prefix)
                or any(part in {"", ".", ".."} for part in row["name"].split("/"))
                or not re.fullmatch(r"[1-9][0-9]{0,19}", str(row.get("generation", "")))
                or not isinstance(row.get("size"), int) or row["size"] < 0 or not row.get("crc32c")):
            raise ValueError("website_cleanup_cloud_scope_or_identity_invalid")
    return sorted(rows, key=lambda row: (row["name"], row["generation"]))


def plan_cloud_cleanup(*, capture_root: Path, storage: Any) -> dict[str, Any]:
    site = _site(capture_root)
    with _lock(site) as journal:
        receipt = _inspection(site, journal)
        value = {"schema_version": "website_capture_cloud_cleanup_plan.v1", "tombstone_digest": receipt["tombstone_digest"],
                 "site_root": str(site), "bucket": storage.bucket_name, "prefix": f"scenes/{site.name}/",
                 "objects": _cloud_objects(site, storage), "provider_cleanup_included": False,
                 "soft_deleted_objects": storage.soft_deleted_objects(f"scenes/{site.name}/")}
        value["digest"] = canonical_digest(value, digest_field="digest")
        _write_once(journal / "cloud_plans" / f"{value['digest'][7:]}.json", value)
        return value


def apply_cloud_cleanup(*, capture_root: Path, storage: Any, expected_plan_digest: str,
                        authorize_cloud_deletion: bool = False) -> dict[str, Any]:
    """Explicit operator operation, never dispatched by the withdrawal worker."""
    if authorize_cloud_deletion is not True or not re.fullmatch(r"sha256:[a-f0-9]{64}", expected_plan_digest):
        raise ValueError("website_cleanup_explicit_cloud_authorization_required")
    site = _site(capture_root)
    with _lock(site) as journal:
        receipt = _inspection(site, journal)
        if receipt["legal_hold"]:
            raise ValueError("website_cleanup_legal_hold")
        plan = _read(journal / "cloud_plans" / f"{expected_plan_digest[7:]}.json")
        if (plan.get("digest") != expected_plan_digest or canonical_digest(plan, digest_field="digest") != expected_plan_digest
                or plan.get("site_root") != str(site) or plan.get("bucket") != storage.bucket_name
                or plan.get("tombstone_digest") != receipt["tombstone_digest"]):
            raise ValueError("website_cleanup_cloud_plan_changed")
        planned = {(row["name"], row["generation"]): row for row in plan["objects"]}
        current = _cloud_objects(site, storage)
        if any(planned.get((row["name"], row["generation"])) != row for row in current):
            raise ValueError("website_cleanup_cloud_source_changed")
        _write_once(journal / "cloud_cleanup_intent.json", {"plan_digest": expected_plan_digest,
                    "tombstone_digest": receipt["tombstone_digest"], "bucket": storage.bucket_name})
        for row in current:
            # The provider may delete and then lose its response. Replay lists
            # exact remaining generations before attempting another deletion.
            storage.delete(row["name"], row["generation"])
        if _cloud_objects(site, storage) or storage.soft_deleted_objects(plan["prefix"]):
            raise ValueError("website_cleanup_cloud_retained_versions_pending")
        verification = {"schema_version": "website_capture_cloud_object_absence.v1", "plan_digest": expected_plan_digest,
                        "tombstone_digest": receipt["tombstone_digest"], "bucket": storage.bucket_name,
                        "all_object_versions_absence_observed": True, "soft_deleted_versions_absence_observed": True,
                        "provider_copies_deleted": False, "physical_provider_storage_erasure_proven": False}
        verification["digest"] = canonical_digest(verification, digest_field="digest")
        _write_once(journal / "cloud_object_absence_verified.json", verification)
        return _inspection(site, journal)


def acknowledge_withdrawal(*, capture_root: Path, command: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(capture_root).absolute()
    # Validate before mutation. The HTTP caller never controls this server root.
    site = _site(root)
    with _lock(site) as journal:
        existing = journal / "tombstone.json"
        if existing.exists():
            tombstone = _read(existing)
            if tombstone.get("command_digest") != canonical_digest(command):
                raise ValueError("website_withdrawal_retry_conflict")
        else:
            bound = _binding(command, root)
            tombstone = {**bound, "schema_version": "website_capture_withdrawal_tombstone.v1",
                         "command_digest": canonical_digest(command), "consent_revoked": True,
                         "consent_status": "revoked", "consent_revoked_at": command["requested_at_iso"],
                         "future_processing_allowed": False, "serve_allowed": False}
            tombstone["digest"] = canonical_digest(tombstone, digest_field="digest")
            _write_once(existing, tombstone)
        return _inspection(site, journal)


def inspect_withdrawal(*, capture_root: Path) -> dict[str, Any]:
    site = _site(capture_root)
    with _lock(site) as journal:
        return _inspection(site, journal)


def plan_local_cleanup(*, capture_root: Path) -> dict[str, Any]:
    site = _site(capture_root)
    with _lock(site) as journal:
        receipt = _inspection(site, journal)
        value = {"schema_version": "website_capture_local_cleanup_plan.v1", "tombstone_digest": receipt["tombstone_digest"],
                 "site_root": str(site), "files": _files(site), "retained_audit_records": True,
                 "provider_cleanup_included": False, "cloud_storage_cleanup_included": False}
        value["digest"] = canonical_digest(value, digest_field="digest")
        _write_once(journal / "plans" / f"{value['digest'][7:]}.json", value)
        return value


def apply_local_cleanup(*, capture_root: Path, expected_plan_digest: str, authorize_local_deletion: bool = False) -> dict[str, Any]:
    """Operator-only API. No remote route, automatic dispatch, or provider call."""
    if authorize_local_deletion is not True:
        raise ValueError("website_cleanup_explicit_authorization_required")
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", expected_plan_digest):
        raise ValueError("website_cleanup_plan_digest_invalid")
    site = _site(capture_root)
    with _lock(site) as journal:
        receipt = _inspection(site, journal)
        if receipt["legal_hold"]:
            raise ValueError("website_cleanup_legal_hold")
        plan = _read(journal / "plans" / f"{expected_plan_digest[7:]}.json")
        if plan.get("digest") != expected_plan_digest or canonical_digest(plan, digest_field="digest") != expected_plan_digest:
            raise ValueError("website_cleanup_plan_changed")
        if plan.get("site_root") != str(site) or plan.get("tombstone_digest") != receipt["tombstone_digest"]:
            raise ValueError("website_cleanup_plan_binding_mismatch")
        planned = {row["path"]: row for row in plan["files"]}
        current = _files(site)
        # Missing planned files are crash/replay progress. Replacements are never deleted.
        if any(planned.get(row["path"]) != row for row in current):
            raise ValueError("website_cleanup_source_changed")
        intent = {"schema_version": "website_capture_local_cleanup_intent.v1", "plan_digest": expected_plan_digest,
                  "tombstone_digest": receipt["tombstone_digest"]}
        _write_once(journal / "cleanup_intent.json", intent)
        if receipt["local_cleanup_verified"]:
            return receipt
        for row in current:
            path = site / row["path"]
            # One exact generation, no recursive delete and no follow-through symlinks.
            if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
                raise ValueError("website_cleanup_symlink_forbidden")
            stat = path.stat()
            if stat.st_dev != row["device"] or stat.st_ino != row["inode"] or stat.st_mtime_ns != row["mtime_ns"] or _sha256_file(path) != row["sha256"]:
                raise ValueError("website_cleanup_source_changed")
            path.unlink()
        if _files(site):
            raise ValueError("website_cleanup_new_files_pending")
        verification = {"schema_version": "website_capture_local_cleanup_verified.v1", "plan_digest": expected_plan_digest,
                        "tombstone_digest": receipt["tombstone_digest"], "absence_observed": True,
                        "deleted_file_count": len(plan["files"]), "retained_audit_records": True}
        verification["digest"] = canonical_digest(verification, digest_field="digest")
        _write_once(journal / "local_cleanup_verified.json", verification)
        return _inspection(site, journal)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("inspect", "plan", "apply", "cloud-plan", "cloud-apply"))
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--expected-plan-digest")
    parser.add_argument("--authorize-local-deletion", action="store_true")
    parser.add_argument("--bucket", help="Existing operator-authorized source bucket for an exact cloud plan")
    parser.add_argument("--authorize-cloud-deletion", action="store_true")
    args = parser.parse_args()
    if args.operation in {"cloud-plan", "cloud-apply"}:
        if not args.bucket:
            parser.error("--bucket is required for explicit cloud cleanup")
        storage = GoogleWebsiteCleanupStorage(args.bucket)
        result = (plan_cloud_cleanup(capture_root=args.capture_root, storage=storage) if args.operation == "cloud-plan" else
                  apply_cloud_cleanup(capture_root=args.capture_root, storage=storage, expected_plan_digest=args.expected_plan_digest or "",
                                      authorize_cloud_deletion=args.authorize_cloud_deletion))
    elif args.operation == "apply":
        result = apply_local_cleanup(capture_root=args.capture_root, expected_plan_digest=args.expected_plan_digest or "",
                                    authorize_local_deletion=args.authorize_local_deletion)
    elif args.operation == "plan":
        result = plan_local_cleanup(capture_root=args.capture_root)
    else:
        result = inspect_withdrawal(capture_root=args.capture_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
