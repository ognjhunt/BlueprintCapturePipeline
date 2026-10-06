"""Durable website denial and explicitly authorized, exact local cleanup.

The HTTP handoff never deletes bytes. Operator cleanup uses a retained plan and
does not assert that cloud storage, provider copies or delivered copies vanished.
ADP-010/day-14: immutable capture rights remain authoritative after withdrawal.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .capture_lifecycle import _sha256_file, _write_once
from .decision_evidence_contracts import canonical_digest


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("website_withdrawal_artifact_symlink")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("website_withdrawal_artifact_invalid")  # noqa: TRY004 - typed refusal contract
    return value


def _directory(path: Path) -> int:
    """Open an anchored directory without following any substituted ancestor."""
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.absolute().parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _persist(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    checked = _directory(path.parent)
    os.close(checked)
    _write_once(path, value)
    descriptor = _directory(path.parent)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
    journal.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = journal / "lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield journal
        finally:
            for path in (journal, site):
                descriptor = _directory(path)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)


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


def _cleanup_files(site: Path, journal: Path) -> list[dict[str, Any]]:
    files = _files(site)
    # Displaced payloads remain customer data, separate from retained audit.
    # A new exact plan may authorize them; an old plan may never adopt them.
    quarantine = journal / "quarantine"
    if quarantine.is_symlink():
        raise ValueError("website_cleanup_symlink_forbidden")
    for path in sorted(quarantine.rglob("*")):
        if path.is_symlink():
            raise ValueError("website_cleanup_symlink_forbidden")
        if path.is_file():
            metadata = path.stat()
            files.append({"path": str(path.relative_to(site)), "sha256": _sha256_file(path), "size": metadata.st_size,
                          "device": metadata.st_dev, "inode": metadata.st_ino, "mtime_ns": metadata.st_mtime_ns})
    return files


def _retained_plan(site: Path, journal: Path, proof: Mapping[str, Any], tombstone: Mapping[str, Any], *, cloud: bool) -> dict[str, Any]:
    plan_digest = proof.get("plan_digest")
    if (not isinstance(plan_digest, str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", plan_digest)
            or proof.get("digest") != canonical_digest(proof, digest_field="digest")
            or proof.get("tombstone_digest") != tombstone["digest"]):
        raise ValueError("website_cleanup_verification_binding_invalid")
    plan = _read(journal / ("cloud_plans" if cloud else "plans") / f"{plan_digest[7:]}.json")
    if (plan.get("digest") != plan_digest or canonical_digest(plan, digest_field="digest") != plan_digest
            or plan.get("schema_version") != ("website_capture_cloud_cleanup_plan.v1" if cloud else "website_capture_local_cleanup_plan.v1")
            or plan.get("site_root") != str(site) or plan.get("tombstone_digest") != tombstone["digest"]):
        raise ValueError("website_cleanup_verification_plan_invalid")
    intent_path = journal / ("cloud_intents" if cloud else "intents") / f"{plan_digest[7:]}.json"
    intent = _read(intent_path if intent_path.exists() else journal / ("cloud_cleanup_intent.json" if cloud else "cleanup_intent.json"))
    if intent.get("plan_digest") != plan_digest or intent.get("tombstone_digest") != tombstone["digest"]:
        raise ValueError("website_cleanup_verification_intent_invalid")
    if cloud:
        from .capture_lifecycle import _parse_time
        _parse_time(proof.get("observed_at_iso"), code="website_cleanup_cloud_observation_time_invalid")
        if (proof.get("schema_version") != "website_capture_cloud_object_absence.v1"
                or proof.get("bucket") != plan.get("bucket") or intent.get("bucket") != plan.get("bucket")
                or plan.get("prefix") != f"scenes/{site.name}/"
                or proof.get("all_object_versions_absence_observed") is not True
                or proof.get("soft_deleted_versions_absence_observed") is not True
                or proof.get("provider_copies_deleted") is not False
                or proof.get("physical_provider_storage_erasure_proven") is not False):
            raise ValueError("website_cleanup_cloud_verification_invalid")
    elif (proof.get("schema_version") != "website_capture_local_cleanup_verified.v1"
            or proof.get("absence_observed") is not True or proof.get("retained_audit_records") is not True
            or not isinstance(plan.get("files"), list) or type(proof.get("planned_file_count")) is not int
            or proof["planned_file_count"] != len(plan["files"])):
        raise ValueError("website_cleanup_local_verification_invalid")
    return plan


def _latest_observation(site: Path, journal: Path, tombstone: Mapping[str, Any], *, cloud: bool) -> dict[str, Any] | None:
    from .capture_lifecycle import _parse_time
    legacy = journal / ("cloud_object_absence_verified.json" if cloud else "local_cleanup_verified.json")
    paths = ([legacy] if legacy.exists() else []) + list((journal / ("cloud_verifications" if cloud else "verifications")).glob("*.json"))
    observations = []
    for path in paths:
        proof = _read(path)
        _retained_plan(site, journal, proof, tombstone, cloud=cloud)
        timestamp = _parse_time(proof["observed_at_iso"], code="website_cleanup_observation_time_invalid").timestamp() if proof.get("observed_at_iso") else 0
        observations.append((timestamp, proof["digest"], proof))
    return max(observations, key=lambda row: row[:2])[2] if observations else None


def _pending_payloads(journal: Path) -> bool:
    directory = journal / "quarantine"
    return directory.is_symlink() or any(path.is_symlink() or path.is_file() for path in directory.rglob("*"))


def _inspection(site: Path, journal: Path, current_cloud_storage: Any | None = None) -> dict[str, Any]:
    tombstone = _read(journal / "tombstone.json")
    if (tombstone.get("digest") != canonical_digest(tombstone, digest_field="digest")
            or tombstone.get("schema_version") != "website_capture_withdrawal_tombstone.v1"
            or tombstone.get("scene_id") != site.name or tombstone.get("request_id") != site.name[5:]
            or tombstone.get("capture_id") != f"walkthrough-{site.name[5:]}"
            or tombstone.get("consent_revoked") is not True):
        raise ValueError("website_withdrawal_tombstone_changed")
    verified = _latest_observation(site, journal, tombstone, cloud=False)
    hold_path = journal / "legal_hold.json"
    hold = _read(hold_path) if hold_path.exists() else None
    if hold is not None and type(hold.get("legal_hold")) is not bool:
        raise ValueError("website_cleanup_legal_hold_unknown")
    held = hold is not None and hold["legal_hold"] is True
    local_complete = bool(verified and not _files(site) and not _pending_payloads(journal))
    cloud = _latest_observation(site, journal, tombstone, cloud=True)
    cloud_plan = _retained_plan(site, journal, cloud, tombstone, cloud=True) if cloud else None
    cloud_current = bool(cloud and current_cloud_storage and current_cloud_storage.bucket_name == cloud_plan["bucket"]
                         and not _cloud_objects(site, current_cloud_storage)
                         and not current_cloud_storage.soft_deleted_objects(cloud_plan["prefix"]))
    value = {"schema_version": "website_capture_withdrawal_receipt.v1",
             **{key: tombstone[key] for key in ("request_id", "scene_id", "capture_id", "withdrawal_id")},
             "command_digest": tombstone["command_digest"], "tombstone_digest": tombstone["digest"],
             "state": "cleanup_retained_legal_hold" if held else "local_cleanup_verified_external_pending" if local_complete
                 else "pipeline_acknowledged_cleanup_pending",
             "pipeline_acknowledged": True, "serve_allowed": False, "future_processing_allowed": False,
             "local_cleanup_verified": local_complete, "local_cleanup_receipt_digest": verified.get("digest") if local_complete else None,
             "deletion_confirmed": False, "provider_acknowledgement": "unknown", "cloud_storage_cleanup_verified": False,
             "delivered_copy_cleanup_verified": False, "retained_audit_records": True, "legal_hold": held,
             "cloud_object_absence_verified": cloud_current,
             "cloud_object_current_absence_status": "verified" if cloud_current else "unknown",
             "cloud_object_absence_observed_at_iso": cloud.get("observed_at_iso") if cloud else None,
             "cloud_object_cleanup_receipt_digest": cloud.get("digest") if cloud else None,
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
        _persist(journal / "cloud_plans" / f"{value['digest'][7:]}.json", value)
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
        _persist(journal / "cloud_intents" / f"{expected_plan_digest[7:]}.json", {"plan_digest": expected_plan_digest,
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
                        "provider_copies_deleted": False, "physical_provider_storage_erasure_proven": False,
                        "observed_at_iso": datetime.now(timezone.utc).isoformat()}
        verification["digest"] = canonical_digest(verification, digest_field="digest")
        verification_path = journal / "cloud_verifications" / f"{expected_plan_digest[7:]}.json"
        if not verification_path.exists():
            _persist(verification_path, verification)
        return _inspection(site, journal, current_cloud_storage=storage)


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
            _persist(existing, tombstone)
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
                 "site_root": str(site), "files": _cleanup_files(site, journal), "retained_audit_records": True,
                 "provider_cleanup_included": False, "cloud_storage_cleanup_included": False}
        value["digest"] = canonical_digest(value, digest_field="digest")
        _persist(journal / "plans" / f"{value['digest'][7:]}.json", value)
        return value


def _candidate_matches(descriptor: int, name: str, row: Mapping[str, Any]) -> bool:
    file_descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor)
    with os.fdopen(file_descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            return False
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
        return (before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
                and before.st_ctime_ns == after.st_ctime_ns and after.st_dev == row["device"] and after.st_ino == row["inode"]
                and after.st_mtime_ns == row["mtime_ns"] and after.st_size == row["size"]
                and "sha256:" + digest.hexdigest() == row["sha256"])


def _delete_local_generation(site: Path, journal: Path, row: Mapping[str, Any], plan_digest: str) -> None:
    relative = Path(row["path"])
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("website_cleanup_plan_path_invalid")
    if relative.parts[0] == "website_withdrawal" and (len(relative.parts) < 3 or relative.parts[1] != "quarantine"):
        raise ValueError("website_cleanup_retained_audit_forbidden")
    source_descriptor = _directory(site / relative.parent)
    candidate_parent = journal / "quarantine" / plan_digest[7:] / relative.parent
    if any(path.is_symlink() for path in (candidate_parent, *candidate_parent.parents)):
        os.close(source_descriptor)
        raise ValueError("website_cleanup_symlink_forbidden")
    try:
        candidate_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        candidate_descriptor = _directory(candidate_parent)
    except BaseException:
        os.close(source_descriptor)
        raise
    try:
        try:
            os.stat(relative.name, dir_fd=candidate_descriptor, follow_symlinks=False)
            quarantined = True
        except FileNotFoundError:
            quarantined = False
        if not quarantined:
            move = {"schema_version": "website_capture_local_move_intent.v1", "plan_digest": plan_digest, "file": dict(row)}
            _persist(journal / "moves" / plan_digest[7:] / f"{canonical_digest(row)[7:]}.json", move)
            try:
                # Both directories are anchored, O_NOFOLLOW descriptors. The
                # private locked candidate cannot be replaced by intake writers.
                os.rename(relative.name, relative.name, src_dir_fd=source_descriptor, dst_dir_fd=candidate_descriptor)
            except FileNotFoundError:
                return  # Exact planned file already absent after a crash.
            os.fsync(source_descriptor)
            os.fsync(candidate_descriptor)
        if not _candidate_matches(candidate_descriptor, relative.name, row):
            try:
                # Restore only if the original name is still absent. Never
                # overwrite a concurrently recreated source; preserve quarantine.
                os.link(relative.name, relative.name, src_dir_fd=candidate_descriptor,
                        dst_dir_fd=source_descriptor, follow_symlinks=False)
            except FileExistsError:
                pass
            else:
                os.unlink(relative.name, dir_fd=candidate_descriptor)
                os.fsync(source_descriptor)
                os.fsync(candidate_descriptor)
            raise ValueError("website_cleanup_source_changed")
        os.unlink(relative.name, dir_fd=candidate_descriptor)
        os.fsync(candidate_descriptor)
    finally:
        os.close(source_descriptor)
        os.close(candidate_descriptor)


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
        candidate_prefix = f"website_withdrawal/quarantine/{expected_plan_digest[7:]}/"
        # Same-plan moved generations are validated by the anchored candidate
        # reader below. Other quarantined bytes need their own explicit plan.
        current = [row for row in _cleanup_files(site, journal) if not row["path"].startswith(candidate_prefix)]
        # Missing planned files are crash/replay progress. Replacements are never deleted.
        if any(planned.get(row["path"]) != row for row in current):
            raise ValueError("website_cleanup_source_changed")
        intent = {"schema_version": "website_capture_local_cleanup_intent.v1", "plan_digest": expected_plan_digest,
                  "tombstone_digest": receipt["tombstone_digest"]}
        _persist(journal / "intents" / f"{expected_plan_digest[7:]}.json", intent)
        if receipt["local_cleanup_verified"]:
            return receipt
        for row in plan["files"]:
            _delete_local_generation(site, journal, row, expected_plan_digest)
        if _files(site) or _pending_payloads(journal):
            raise ValueError("website_cleanup_new_files_pending")
        verification = {"schema_version": "website_capture_local_cleanup_verified.v1", "plan_digest": expected_plan_digest,
                        "tombstone_digest": receipt["tombstone_digest"], "absence_observed": True,
                        "planned_file_count": len(plan["files"]), "retained_audit_records": True,
                        "observed_at_iso": datetime.now(timezone.utc).isoformat()}
        verification["digest"] = canonical_digest(verification, digest_field="digest")
        verification_path = journal / "verifications" / f"{expected_plan_digest[7:]}.json"
        if not verification_path.exists():
            _persist(verification_path, verification)
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
