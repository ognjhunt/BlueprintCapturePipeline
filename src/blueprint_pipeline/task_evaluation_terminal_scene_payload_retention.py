"""B2-backed retention for terminal scene-configuration payload bytes.

This module deliberately separates large, reproducible payload bytes from the
small local evidence that proves what happened.  It never reaps a directory.
Only an exact provider-output ZIP, binary files proven byte-for-byte present
inside that ZIP, and an exact completed-diagnostic provider bundle may be
removed.  Each source artifact must first be content-addressed in the expected
B2 bucket and pass a full streaming service-account readback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from .decision_evidence_contracts import (
    canonical_digest,
    canonical_json,
    cross_runtime_canonical_digest,
)
from .task_evaluation_configured_scene_object_store import (
    publish_configured_scene_artifact,
)
from .task_evaluation_scene_artifact_retention import (
    seal_scene_artifact_remote_index,
)


APPLY_ACK = "reap-terminal-b2-verified-scene-payloads"
_SCOPE_KINDS = {"launch", "diagnostic"}
_TERMINAL_LAUNCH_STATUSES = {"completed", "blocked"}
_TERMINAL_DIAGNOSTIC_STATUSES = {
    "completed_diagnostic_only",
    "blocked_diagnostic_only",
}
_BINARY_PAYLOAD_SUFFIXES = {
    ".bin",
    ".jpeg",
    ".jpg",
    ".mp4",
    ".npy",
    ".npz",
    ".ply",
    ".png",
    ".pt",
    ".pth",
    ".splat",
    ".usd",
    ".usda",
    ".usdc",
    ".usdz",
    ".zip",
}
_PROTECTED_PATH_TOKENS = {
    "authority",
    "billing",
    "index",
    "lineage",
    "manifest",
    "profile",
    "provider-zero",
    "provider_zero",
    "queue",
    "receipt",
    "reconciliation",
    "secret",
}
_EXPECTED_OUTPUT_ZIP_RELATIVE = Path(
    "vast_provider_run/vast_provider_runtime_output.zip"
)
_REMOTE_INDEX_SCHEMA = "task_evaluation_scene_artifact_remote_index.v1"
_REFERENCE_SCHEMA = "task_evaluation_scene_artifact_reference.v1"
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_BUCKET = re.compile(r"[A-Za-z0-9][A-Za-z0-9.-]{1,61}[A-Za-z0-9]\Z")

Publisher = Callable[..., Mapping[str, Any]]


class TaskEvaluationTerminalScenePayloadRetentionError(RuntimeError):
    """Terminal payload lifecycle or remote identity was not proven."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _stream_digest(stream: Any) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
    return "sha256:" + digest.hexdigest(), size


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationTerminalScenePayloadRetentionError(code) from exc
    if path.is_symlink() or not path.is_file() or not isinstance(value, Mapping):
        raise TaskEvaluationTerminalScenePayloadRetentionError(code)
    return dict(value)


def _file_record(path: Path) -> dict[str, Any]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_candidate_invalid"
        ) from exc
    if path.is_symlink() or not path.is_file():
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_candidate_invalid"
        )
    return {
        "local_path": str(path),
        "device": info.st_dev,
        "inode": info.st_ino,
        "size_bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "digest": _sha256(path),
    }


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = (canonical_json(value) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o440,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    path.chmod(0o440)
    if path.read_bytes() != payload:
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_receipt_readback_failed"
        )


def _under_managed_root(
    *, scope_root: str | Path, managed_root: str | Path, scope_kind: str
) -> tuple[Path, Path]:
    if scope_kind not in _SCOPE_KINDS:
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_scope_kind_invalid"
        )
    managed = Path(managed_root).expanduser().resolve()
    unresolved = Path(scope_root).expanduser().absolute()
    root = unresolved.resolve()
    try:
        relative = root.relative_to(managed)
    except ValueError as exc:
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_scope_outside_managed_root"
        ) from exc
    if (
        unresolved.is_symlink()
        or not root.is_dir()
        or not relative.parts
        or (scope_kind == "launch" and len(relative.parts) != 1)
    ):
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_scope_invalid"
        )
    job = (
        root / "allocator" / "scene-configuration-job"
        if scope_kind == "launch"
        else root
    )
    if job.is_symlink() or not job.is_dir():
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_scope_invalid"
        )
    return root, job


def _validate_result(
    job: Path, *, scope_kind: str
) -> tuple[dict[str, Any], Path | None]:
    schema = (
        "task_evaluation_scene_configuration_vast_result.v1"
        if scope_kind == "launch"
        else "task_evaluation_scene_configuration_diagnostic_vast_result.v1"
    )
    # Both execution modes intentionally share the canonical filename; the
    # embedded schema distinguishes a qualifying launch from a diagnostic.
    result_path = job / "task_evaluation_scene_configuration_vast_result.v1.json"
    result = _read(result_path, code="terminal_scene_payload_result_invalid")
    statuses = (
        _TERMINAL_LAUNCH_STATUSES
        if scope_kind == "launch"
        else _TERMINAL_DIAGNOSTIC_STATUSES
    )
    if (
        result.get("schema_version") != schema
        or result.get("status") not in statuses
        or result.get("result_digest")
        != canonical_digest(result, digest_field="result_digest")
        or not str(result.get("run_id") or "")
        or _COMMIT.fullmatch(str(result.get("source_commit") or "")) is None
        or _DIGEST.fullmatch(str(result.get("bundle_sha256") or "")) is None
        or result.get("continuing_spend_from_this_run") is not False
        or result.get("runtime_secret_cleanup_completed") is not True
        or result.get("raw_secret_values_recorded") is not False
        or not isinstance(result.get("blockers"), list)
    ):
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_result_invalid"
        )
    output_zip = job / _EXPECTED_OUTPUT_ZIP_RELATIVE
    recorded_zip = Path(str(result.get("provider_runtime_output_zip_path") or ""))
    output_exists = output_zip.is_file() and not output_zip.is_symlink()
    if output_exists and (
        recorded_zip.expanduser().absolute() != output_zip.absolute()
        or result.get("provider_runtime_output_zip_sha256") != _sha256(output_zip)
    ):
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_archive_invalid"
        )
    if not output_exists and scope_kind == "launch":
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_archive_invalid"
        )
    return result, output_zip if output_exists else None


def _validate_provider_and_object_store_zero(job: Path) -> list[dict[str, Any]]:
    teardown_path = job / "vast_provider_run" / "vast_teardown_manifest.json"
    cleanup_path = (
        job / "object_store_staging" / "wam_provider_object_store_cleanup.json"
    )
    teardown = _read(
        teardown_path, code="terminal_scene_payload_provider_nonzero"
    )
    cleanup = _read(
        cleanup_path, code="terminal_scene_payload_object_store_nonzero"
    )
    if (
        teardown.get("schema_version") != "vast_teardown_manifest.v1"
        or teardown.get("status") != "completed"
        or teardown.get("continuing_spend_from_this_run") is not False
        or teardown.get("raw_secret_values_recorded") is not False
    ):
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_provider_nonzero"
        )
    if (
        cleanup.get("schema_version") != "wam_provider_object_store_cleanup.v1"
        or cleanup.get("status") != "completed"
        or cleanup.get("all_objects_absent") is not True
        or cleanup.get("blockers") != []
        or cleanup.get("raw_secret_values_recorded") is not False
    ):
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_object_store_nonzero"
        )
    return [_file_record(teardown_path), _file_record(cleanup_path)]


def _validate_launch_reconciliation(root: Path) -> list[dict[str, Any]]:
    receipt_path = root / "launch_receipt.json"
    sync_path = root / "webapp_sync_succeeded.json"
    zero_path = root / "post_teardown_provider_zero_receipt.json"
    receipt = _read(receipt_path, code="terminal_scene_payload_unreconciled")
    sync = _read(sync_path, code="terminal_scene_payload_unreconciled")
    zero = _read(zero_path, code="terminal_scene_payload_unreconciled")
    receipt_canonicalization = receipt.get("receipt_digest_canonicalization")
    expected_receipt_digest = (
        cross_runtime_canonical_digest(receipt, digest_field="receipt_digest")
        if receipt_canonicalization == "rfc8785"
        else canonical_digest(receipt, digest_field="receipt_digest")
        if receipt_canonicalization is None
        else None
    )
    if (
        receipt.get("schema_version") != "task_evaluation_launch_receipt.v1"
        or receipt.get("status") not in _TERMINAL_LAUNCH_STATUSES
        or receipt.get("launch_id") != root.name
        or receipt.get("run_id") != root.name
        or receipt.get("receipt_digest") != expected_receipt_digest
        or receipt.get("retain_processing_for_reconciliation") is True
        or receipt.get("raw_secret_values_recorded") is not False
    ):
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_unreconciled"
        )
    if (
        sync.get("schema_version")
        != "task_evaluation_launch_webapp_sync_result.v1"
        or sync.get("status") != "succeeded"
        or sync.get("launch_id") != root.name
        or sync.get("run_id") != root.name
        or sync.get("request_digest") != receipt.get("request_digest")
        or sync.get("receipt_digest") != receipt.get("receipt_digest")
        or sync.get("sync_result_digest")
        != canonical_digest(sync, digest_field="sync_result_digest")
        or sync.get("provider_mutation_performed") not in (False, None)
    ):
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_unreconciled"
        )
    if (
        zero.get("schema_version")
        != "task_evaluation_post_teardown_provider_zero.v1"
        or zero.get("status") != "provider_zero_confirmed"
        or zero.get("launch_id") != root.name
        or zero.get("run_id") != root.name
        or zero.get("request_digest") != receipt.get("request_digest")
        or zero.get("receipt_digest") != receipt.get("receipt_digest")
        or zero.get("provider_zero_verified") is not True
        or zero.get("continuing_spend_from_this_run") is not False
        or zero.get("allocator_invoked") is not False
        or zero.get("provider_mutation_performed") is not False
        or zero.get("automatic_retry_performed") is not False
        or zero.get("blockers") != []
        or zero.get("provider_zero_receipt_digest")
        != canonical_digest(zero, digest_field="provider_zero_receipt_digest")
    ):
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_provider_nonzero"
        )
    return [_file_record(receipt_path), _file_record(sync_path), _file_record(zero_path)]


def _validate_diagnostic_lifecycle(
    *, job: Path, run_id: str
) -> list[dict[str, Any]]:
    lease_path = job / "scene_artifact_lease.v1.json"
    lease = _read(lease_path, code="terminal_scene_payload_unreconciled")
    if (
        lease.get("schema_version") != "task_evaluation_scene_artifact_lease.v1"
        or lease.get("lease_digest")
        != canonical_digest(lease, digest_field="lease_digest")
        or lease.get("run_id") != run_id
        or lease.get("raw_secret_values_recorded") is not False
    ):
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_unreconciled"
        )
    if lease.get("lifecycle_state") != "completed":
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_scope_active"
        )
    return [_file_record(lease_path)]


def _validated_diagnostic_bundle(
    *, job: Path, result: Mapping[str, Any]
) -> tuple[Path, Path] | None:
    """Return one exact completed-diagnostic bundle without touching its receipt."""

    matches = sorted(
        path
        for root in job.parent.glob("bundle*")
        if root.is_dir() and not root.is_symlink()
        for path in root.glob(
            "task_evaluation_scene_configuration_provider_bundle.v1.receipt.json"
        )
    )
    if not matches:
        return None
    if len(matches) != 1:
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_diagnostic_bundle_ambiguous"
        )
    receipt_path = matches[0]
    receipt = _read(
        receipt_path, code="terminal_scene_payload_diagnostic_bundle_invalid"
    )
    bundle = receipt_path.parent / "task_evaluation_scene_configuration_provider_bundle.zip"
    if (
        receipt.get("schema_version")
        != "task_evaluation_scene_configuration_provider_bundle.v1"
        or receipt.get("status") != "ready"
        or receipt.get("diagnostic_only") is not True
        or receipt.get("qualification_eligible") is not False
        or receipt.get("run_id") != result.get("run_id")
        or receipt.get("source_commit") != result.get("source_commit")
        or receipt.get("bundle_sha256") != result.get("bundle_sha256")
        or receipt.get("bundle_path") != str(bundle)
        or receipt.get("receipt_digest")
        != canonical_digest(receipt, digest_field="receipt_digest")
        or bundle.is_symlink()
        or not bundle.is_file()
        or bundle.stat().st_size != receipt.get("bundle_size_bytes")
        or _sha256(bundle) != receipt.get("bundle_sha256")
    ):
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_diagnostic_bundle_invalid"
        )
    return bundle, receipt_path


def _validate_terminal_scope(
    *,
    scope_root: str | Path,
    scope_kind: str,
    managed_root: str | Path,
    configured_controls_plan_root: str | Path | None = None,
    configured_controls_progression_root: str | Path | None = None,
) -> dict[str, Any]:
    root, job = _under_managed_root(
        scope_root=scope_root, managed_root=managed_root, scope_kind=scope_kind
    )
    result, output_zip = _validate_result(job, scope_kind=scope_kind)
    evidence = [
        _file_record(
            job / "task_evaluation_scene_configuration_vast_result.v1.json"
        )
    ]
    evidence.extend(_validate_provider_and_object_store_zero(job))
    if scope_kind == "launch":
        evidence.extend(_validate_launch_reconciliation(root))
        _validate_configured_controls_dependency(
            source_launch_id=root.name,
            plan_root=configured_controls_plan_root,
            progression_root=configured_controls_progression_root,
        )
    else:
        evidence.extend(
            _validate_diagnostic_lifecycle(job=job, run_id=str(result["run_id"]))
        )
    diagnostic_bundle = (
        _validated_diagnostic_bundle(job=job, result=result)
        if scope_kind == "diagnostic"
        else None
    )
    if diagnostic_bundle is not None:
        evidence.append(_file_record(diagnostic_bundle[1]))
    if output_zip is None and diagnostic_bundle is None:
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_archive_invalid"
        )
    return {
        "scope_root": root,
        "job_root": job,
        "result": result,
        "output_zip": output_zip,
        "diagnostic_bundle": diagnostic_bundle,
        "lifecycle_evidence": evidence,
    }


def _validate_configured_controls_dependency(
    *,
    source_launch_id: str,
    plan_root: str | Path | None,
    progression_root: str | Path | None,
) -> None:
    """Pin launches still consumed by the automatic controls progression."""

    if plan_root is None or progression_root is None:
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_controls_dependency_unavailable"
        )
    plans = Path(plan_root).expanduser().resolve()
    progression = Path(progression_root).expanduser().resolve()
    if (
        plans.is_symlink()
        or progression.is_symlink()
        or not plans.is_dir()
        or not progression.is_dir()
    ):
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_controls_dependency_unavailable"
        )
    for path in sorted(plans.glob("*.json")):
        plan = _read(
            path, code="terminal_scene_payload_controls_dependency_invalid"
        )
        if (
            plan.get("schema_version")
            != "task_evaluation_configured_controls_progression_plan.v1"
            or plan.get("plan_digest")
            != canonical_digest(plan, digest_field="plan_digest")
            or not isinstance(plan.get("enabled"), bool)
            or not str(plan.get("source_launch_id") or "")
        ):
            raise TaskEvaluationTerminalScenePayloadRetentionError(
                "terminal_scene_payload_controls_dependency_invalid"
            )
        if (
            plan.get("enabled") is True
            and plan.get("source_launch_id") == source_launch_id
        ):
            raise TaskEvaluationTerminalScenePayloadRetentionError(
                "terminal_scene_payload_scope_active"
            )
    state = progression / source_launch_id
    if not state.exists():
        return
    if state.is_symlink() or not state.is_dir():
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_controls_dependency_invalid"
        )
    terminal = state / "controls_launch_progression.json"
    if not terminal.is_file():
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_scope_active"
        )
    value = _read(
        terminal, code="terminal_scene_payload_controls_dependency_invalid"
    )
    if (
        value.get("schema_version")
        != "task_evaluation_configured_controls_progression.v1"
        or value.get("status") != "controls_pair_launch_queued"
        or value.get("progression_digest")
        != canonical_digest(value, digest_field="progression_digest")
    ):
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_scope_active"
        )


def _valid_b2_reference(
    reference: Any,
    *,
    expected_bucket: str,
    expected_digest: str,
    expected_size: int,
    artifact_kind: str,
) -> bool:
    if not isinstance(reference, Mapping) or _BUCKET.fullmatch(expected_bucket) is None:
        return False
    uri = urlsplit(str(reference.get("uri") or ""))
    key = uri.path.lstrip("/")
    digest_hex = expected_digest.removeprefix("sha256:")
    return bool(
        reference.get("schema_version") == _REFERENCE_SCHEMA
        and reference.get("status") == "remote_verified"
        and reference.get("artifact_kind") == artifact_kind
        and uri.scheme == "s3"
        and uri.netloc == expected_bucket
        and f"/{artifact_kind}/sha256/{digest_hex}/" in f"/{key}"
        and reference.get("digest") == expected_digest
        and reference.get("size_bytes") == expected_size
        and reference.get("content_addressed_key") is True
        and reference.get("remote_identity_verified") is True
        and reference.get("full_byte_service_account_readback_passed") is True
        and reference.get("readback_digest") == expected_digest
        and reference.get("readback_size_bytes") == expected_size
        and str(reference.get("remote_verified_at") or "")
        and reference.get("raw_secret_values_recorded") is False
    )


def archive_terminal_scene_payload_to_b2(
    *,
    scope_root: str | Path,
    scope_kind: str,
    managed_root: str | Path,
    expected_bucket: str,
    index_destination: str | Path,
    publisher: Publisher = publish_configured_scene_artifact,
    configured_controls_plan_root: str | Path | None = None,
    configured_controls_progression_root: str | Path | None = None,
) -> dict[str, Any]:
    """Upload one terminal provider ZIP to B2 and seal its immutable index."""

    scope = _validate_terminal_scope(
        scope_root=scope_root,
        scope_kind=scope_kind,
        managed_root=managed_root,
        configured_controls_plan_root=configured_controls_plan_root,
        configured_controls_progression_root=configured_controls_progression_root,
    )
    local_artifacts: list[tuple[str, Path]] = []
    output_zip = scope["output_zip"]
    if isinstance(output_zip, Path):
        local_artifacts.append(("provider-output", output_zip))
    diagnostic_bundle = scope["diagnostic_bundle"]
    if isinstance(diagnostic_bundle, tuple):
        local_artifacts.append(("provider-bundle", diagnostic_bundle[0]))
    references: list[dict[str, Any]] = []
    for artifact_kind, artifact_path in local_artifacts:
        reference = dict(
            publisher(path=artifact_path, artifact_kind=artifact_kind)
        )
        digest = _sha256(artifact_path)
        size = artifact_path.stat().st_size
        if not _valid_b2_reference(
            reference,
            expected_bucket=expected_bucket,
            expected_digest=digest,
            expected_size=size,
            artifact_kind=artifact_kind,
        ):
            raise TaskEvaluationTerminalScenePayloadRetentionError(
                "terminal_scene_payload_b2_reference_invalid"
            )
        references.append(reference)
    result = scope["result"]
    return seal_scene_artifact_remote_index(
        destination=index_destination,
        run_id=str(result["run_id"]),
        source_commit=str(result["source_commit"]),
        bundle_digest=str(result["bundle_sha256"]),
        artifact_references=references,
    )


def _validated_b2_index(
    *, path: Path, scope: Mapping[str, Any], expected_bucket: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    index = _read(path, code="terminal_scene_payload_b2_index_invalid")
    references = index.get("artifact_references")
    normalized = (
        [dict(reference) for reference in references]
        if isinstance(references, list)
        and references
        and all(isinstance(reference, Mapping) for reference in references)
        else []
    )
    result = scope["result"]
    output_zip = scope["output_zip"]
    local_artifacts: dict[str, Path] = {}
    if isinstance(output_zip, Path):
        local_artifacts["provider-output"] = output_zip
    diagnostic_bundle = scope["diagnostic_bundle"]
    if isinstance(diagnostic_bundle, tuple):
        local_artifacts["provider-bundle"] = diagnostic_bundle[0]
    references_by_kind = {
        str(reference.get("artifact_kind") or ""): reference
        for reference in normalized
    }
    if (
        index.get("schema_version") != _REMOTE_INDEX_SCHEMA
        or index.get("status") != "completed"
        or index.get("run_id") != result.get("run_id")
        or index.get("source_commit") != result.get("source_commit")
        or index.get("bundle_digest") != result.get("bundle_sha256")
        or index.get("artifact_count") != len(local_artifacts)
        or index.get("total_size_bytes")
        != sum(path.stat().st_size for path in local_artifacts.values())
        or index.get("all_artifacts_content_addressed") is not True
        or index.get("all_artifacts_remote_verified") is not True
        or index.get("raw_secret_values_recorded") is not False
        or index.get("index_digest")
        != canonical_digest(index, digest_field="index_digest")
        or set(references_by_kind) != set(local_artifacts)
        or any(
            not _valid_b2_reference(
                references_by_kind[kind],
                expected_bucket=expected_bucket,
                expected_digest=_sha256(artifact_path),
                expected_size=artifact_path.stat().st_size,
                artifact_kind=kind,
            )
            for kind, artifact_path in local_artifacts.items()
        )
    ):
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_b2_index_invalid"
        )
    return index, normalized


def _protected_payload_path(path: PurePosixPath) -> bool:
    return any(
        token in component.lower()
        for component in path.parts
        for token in _PROTECTED_PATH_TOKENS
    )


def _archive_backed_candidates(
    *, job: Path, output_zip: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    protected: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(output_zip) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(names) != len(set(names)) or len(names) > 10_000:
                raise TaskEvaluationTerminalScenePayloadRetentionError(
                    "terminal_scene_payload_archive_invalid"
                )
            for member in members:
                member_path = PurePosixPath(member.filename)
                mode = member.external_attr >> 16
                if (
                    member.is_dir()
                    or member_path.is_absolute()
                    or not member_path.parts
                    or ".." in member_path.parts
                    or stat.S_ISLNK(mode)
                ):
                    if member.is_dir():
                        continue
                    raise TaskEvaluationTerminalScenePayloadRetentionError(
                        "terminal_scene_payload_archive_invalid"
                    )
                local = job / "immutable_execution" / Path(*member_path.parts)
                if not local.exists():
                    continue
                if (
                    local.is_symlink()
                    or not local.is_file()
                    or local.stat().st_size != member.file_size
                ):
                    raise TaskEvaluationTerminalScenePayloadRetentionError(
                        "terminal_scene_payload_archive_member_mismatch"
                    )
                with archive.open(member) as stream:
                    archive_digest, archive_size = _stream_digest(stream)
                local_record = _file_record(local)
                if (
                    archive_size != local_record["size_bytes"]
                    or archive_digest != local_record["digest"]
                ):
                    raise TaskEvaluationTerminalScenePayloadRetentionError(
                        "terminal_scene_payload_archive_member_mismatch"
                    )
                reason = None
                if _protected_payload_path(member_path):
                    reason = "protected_evidence_path"
                elif member_path.suffix.lower() not in _BINARY_PAYLOAD_SUFFIXES:
                    reason = "non_binary_local_evidence"
                row = {
                    **local_record,
                    "archive_member": member.filename,
                }
                if reason:
                    protected.append({**row, "reason": reason})
                else:
                    candidates.append(row)
    except TaskEvaluationTerminalScenePayloadRetentionError:
        raise
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_archive_invalid"
        ) from exc
    candidates.append(
        {
            **_file_record(output_zip),
            "archive_member": None,
        }
    )
    candidates.sort(key=lambda row: (row["archive_member"] is None, row["local_path"]))
    return candidates, protected


def plan_terminal_scene_payload_retention(
    *,
    scope_root: str | Path,
    scope_kind: str,
    managed_root: str | Path,
    expected_bucket: str,
    b2_index_path: str | Path,
    configured_controls_plan_root: str | Path | None = None,
    configured_controls_progression_root: str | Path | None = None,
) -> dict[str, Any]:
    """Plan exact terminal payload files; no directory is ever a candidate."""

    scope = _validate_terminal_scope(
        scope_root=scope_root,
        scope_kind=scope_kind,
        managed_root=managed_root,
        configured_controls_plan_root=configured_controls_plan_root,
        configured_controls_progression_root=configured_controls_progression_root,
    )
    index_path = Path(b2_index_path).expanduser().absolute()
    job = scope["job_root"]
    if index_path.parent.resolve() != job.resolve() or index_path.is_symlink():
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_b2_index_invalid"
        )
    index, references = _validated_b2_index(
        path=index_path, scope=scope, expected_bucket=expected_bucket
    )
    output_zip = scope["output_zip"]
    if isinstance(output_zip, Path):
        candidates, protected = _archive_backed_candidates(
            job=job, output_zip=output_zip
        )
        for row in candidates:
            row["artifact_kind"] = "provider-output"
    else:
        candidates, protected = [], []
    diagnostic_bundle = scope["diagnostic_bundle"]
    if isinstance(diagnostic_bundle, tuple):
        candidates.append(
            {
                **_file_record(diagnostic_bundle[0]),
                "archive_member": None,
                "artifact_kind": "provider-bundle",
            }
        )
    candidates.sort(
        key=lambda row: (
            str(row["artifact_kind"]),
            row["archive_member"] is None,
            row["local_path"],
        )
    )
    result = scope["result"]
    plan: dict[str, Any] = {
        "schema_version": "task_evaluation_terminal_scene_payload_retention_plan.v1",
        "status": "completed",
        "scope_kind": scope_kind,
        "scope_root": str(scope["scope_root"]),
        "managed_root": str(Path(managed_root).expanduser().resolve()),
        "job_root": str(job),
        "run_id": result["run_id"],
        "source_commit": result["source_commit"],
        "bundle_digest": result["bundle_sha256"],
        "expected_bucket": expected_bucket,
        "b2_index_path": str(index_path),
        "configured_controls_plan_root": (
            str(Path(configured_controls_plan_root).expanduser().resolve())
            if configured_controls_plan_root is not None
            else None
        ),
        "configured_controls_progression_root": (
            str(Path(configured_controls_progression_root).expanduser().resolve())
            if configured_controls_progression_root is not None
            else None
        ),
        "b2_index_digest": index["index_digest"],
        "remote_references": references,
        "lifecycle_proof": {
            "reconciled": True,
            "provider_zero_verified": True,
            "object_store_zero_verified": True,
            "terminal_status": result["status"],
            "evidence": scope["lifecycle_evidence"],
        },
        "candidate_count": len(candidates),
        "candidate_bytes": sum(int(row["size_bytes"]) for row in candidates),
        "candidates": candidates,
        "protected": protected,
        "directories_removed": False,
        "raw_secret_values_recorded": False,
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def _candidate_unchanged(row: Mapping[str, Any]) -> bool:
    path = Path(str(row.get("local_path") or "")).absolute()
    try:
        info = path.lstat()
    except OSError:
        return False
    return bool(
        not path.is_symlink()
        and path.is_file()
        and info.st_dev == row.get("device")
        and info.st_ino == row.get("inode")
        and info.st_size == row.get("size_bytes")
        and info.st_mtime_ns == row.get("mtime_ns")
        and _sha256(path) == row.get("digest")
    )


def apply_terminal_scene_payload_retention(
    *, plan: Mapping[str, Any], acknowledgement: str, expected_bucket: str
) -> dict[str, Any]:
    """Revalidate the whole lifecycle and every inode before unlinking files."""

    if (
        acknowledgement != APPLY_ACK
        or plan.get("schema_version")
        != "task_evaluation_terminal_scene_payload_retention_plan.v1"
        or plan.get("status") != "completed"
        or plan.get("expected_bucket") != expected_bucket
        or plan.get("plan_digest")
        != canonical_digest(plan, digest_field="plan_digest")
        or plan.get("directories_removed") is not False
    ):
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_retention_plan_invalid"
        )
    try:
        current = plan_terminal_scene_payload_retention(
            scope_root=str(plan.get("scope_root") or ""),
            scope_kind=str(plan.get("scope_kind") or ""),
            managed_root=str(plan.get("managed_root") or ""),
            expected_bucket=expected_bucket,
            b2_index_path=str(plan.get("b2_index_path") or ""),
            configured_controls_plan_root=plan.get(
                "configured_controls_plan_root"
            ),
            configured_controls_progression_root=plan.get(
                "configured_controls_progression_root"
            ),
        )
    except TaskEvaluationTerminalScenePayloadRetentionError as exc:
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_candidate_changed"
        ) from exc
    comparable = (
        "scope_kind",
        "scope_root",
        "managed_root",
        "job_root",
        "run_id",
        "source_commit",
        "bundle_digest",
        "expected_bucket",
        "b2_index_path",
        "b2_index_digest",
        "configured_controls_plan_root",
        "configured_controls_progression_root",
        "remote_references",
        "lifecycle_proof",
        "candidate_count",
        "candidate_bytes",
        "candidates",
        "protected",
    )
    if any(current[field] != plan.get(field) for field in comparable):
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_candidate_changed"
        )
    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or not candidates or not all(
        isinstance(row, Mapping) and _candidate_unchanged(row) for row in candidates
    ):
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_candidate_changed"
        )
    removed: list[dict[str, Any]] = []
    for row in candidates:
        path = Path(str(row["local_path"]))
        path.unlink()
        removed.append(dict(row))
    result: dict[str, Any] = {
        "schema_version": "task_evaluation_terminal_scene_payload_retention_result.v1",
        "status": "completed",
        "source_plan_digest": plan["plan_digest"],
        "b2_index_digest": plan["b2_index_digest"],
        "removed_count": len(removed),
        "removed_bytes": sum(int(row["size_bytes"]) for row in removed),
        "removed": removed,
        "directories_removed": False,
        "manifests_receipts_lineage_preserved": True,
        "raw_secret_values_recorded": False,
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    return result


def write_terminal_scene_payload_retention_plan(
    *, destination: str | Path, **kwargs: Any
) -> dict[str, Any]:
    plan = plan_terminal_scene_payload_retention(**kwargs)
    _write_exclusive(Path(destination).expanduser().absolute(), plan)
    return plan


def write_terminal_scene_payload_retention_result(
    *, destination: str | Path, **kwargs: Any
) -> dict[str, Any]:
    result = apply_terminal_scene_payload_retention(**kwargs)
    _write_exclusive(Path(destination).expanduser().absolute(), result)
    return result


def _add_scope_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scope-kind", required=True, choices=sorted(_SCOPE_KINDS))
    parser.add_argument("--scope-root", required=True)
    parser.add_argument("--managed-root", required=True)
    parser.add_argument("--expected-bucket", required=True)
    parser.add_argument("--configured-controls-plan-root")
    parser.add_argument("--configured-controls-progression-root")


def main(argv: list[str] | None = None) -> int:
    """Archive, plan, or explicitly apply one exact terminal payload scope."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    archive_parser = commands.add_parser("archive")
    _add_scope_arguments(archive_parser)
    archive_parser.add_argument("--b2-index-out", required=True)
    plan_parser = commands.add_parser("plan")
    _add_scope_arguments(plan_parser)
    plan_parser.add_argument("--b2-index", required=True)
    plan_parser.add_argument("--plan-out", required=True)
    apply_parser = commands.add_parser("apply")
    apply_parser.add_argument("--plan", required=True)
    apply_parser.add_argument("--expected-bucket", required=True)
    apply_parser.add_argument("--receipt-out", required=True)
    apply_parser.add_argument("--ack", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "archive":
            result = archive_terminal_scene_payload_to_b2(
                scope_root=args.scope_root,
                scope_kind=args.scope_kind,
                managed_root=args.managed_root,
                expected_bucket=args.expected_bucket,
                index_destination=args.b2_index_out,
                configured_controls_plan_root=args.configured_controls_plan_root,
                configured_controls_progression_root=(
                    args.configured_controls_progression_root
                ),
            )
            summary = {
                "status": "archived",
                "index_digest": result["index_digest"],
                "artifact_count": result["artifact_count"],
                "total_size_bytes": result["total_size_bytes"],
            }
        elif args.command == "plan":
            result = write_terminal_scene_payload_retention_plan(
                destination=args.plan_out,
                scope_root=args.scope_root,
                scope_kind=args.scope_kind,
                managed_root=args.managed_root,
                expected_bucket=args.expected_bucket,
                b2_index_path=args.b2_index,
                configured_controls_plan_root=args.configured_controls_plan_root,
                configured_controls_progression_root=(
                    args.configured_controls_progression_root
                ),
            )
            summary = {
                "status": result["status"],
                "candidate_count": result["candidate_count"],
                "candidate_bytes": result["candidate_bytes"],
                "plan_digest": result["plan_digest"],
            }
        else:
            plan = _read(
                Path(args.plan).expanduser().absolute(),
                code="terminal_scene_payload_retention_plan_invalid",
            )
            result = write_terminal_scene_payload_retention_result(
                destination=args.receipt_out,
                plan=plan,
                acknowledgement=args.ack,
                expected_bucket=args.expected_bucket,
            )
            summary = {
                "status": result["status"],
                "removed_count": result["removed_count"],
                "removed_bytes": result["removed_bytes"],
                "result_digest": result["result_digest"],
            }
    except (
        TaskEvaluationTerminalScenePayloadRetentionError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": [str(exc)],
                    "removed_count": 0,
                    "removed_bytes": 0,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


__all__ = [
    "APPLY_ACK",
    "TaskEvaluationTerminalScenePayloadRetentionError",
    "apply_terminal_scene_payload_retention",
    "archive_terminal_scene_payload_to_b2",
    "plan_terminal_scene_payload_retention",
    "write_terminal_scene_payload_retention_plan",
    "write_terminal_scene_payload_retention_result",
]


if __name__ == "__main__":  # pragma: no cover - exercised through focused APIs
    raise SystemExit(main())
