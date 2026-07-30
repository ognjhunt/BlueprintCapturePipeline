"""Server-side transfer of a Web upload into immutable Capture Intake storage.

The transfer grant is deliberately ephemeral.  It is authenticated by the live
intake service, used only to stream the original bytes into a local quarantine,
and never written to an artifact, receipt, exception, or log.  The resulting
receipt proves byte verification and intake admission only; Capture QA remains a
separate Pipeline-owned stage.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Callable, ContextManager, Mapping, Sequence
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from .capture_intake import CaptureIntakeError, materialize_capture_intake
from .capture_qa import build_capture_qa_report
from .capture_qa_webapp_sync import build_capture_qa_webapp_publication
from .core.security_controls import strict_identifier


CAPTURE_UPLOAD_TRANSFER_SCHEMA_VERSION = "capture_upload_transfer_submission.v1"
CAPTURE_UPLOAD_RECEIPT_SCHEMA_VERSION = "capture_upload_intake_receipt.v1"
CAPTURE_UPLOAD_STORE_ROOT_ENV = "PIPELINE_CAPTURE_INTAKE_STORE_ROOT"
CAPTURE_UPLOAD_ALLOWED_HOSTS_ENV = "PIPELINE_CAPTURE_TRANSFER_ALLOWED_HOSTS"
CAPTURE_MALWARE_SCANNER_ARGV_ENV = "PIPELINE_CAPTURE_MALWARE_SCANNER_ARGV_JSON"

_WEB_PROFILES = {
    "camera_360_equirectangular",
    "camera_360_native",
    "monocular_video",
}
_MEDIA_TYPES = {
    "camera_360_equirectangular": {"video/mp4", "video/quicktime"},
    "camera_360_native": {"application/octet-stream", "video/x-insta360"},
    "monocular_video": {"video/mp4", "video/quicktime"},
}
_MAX_BYTES = 50 * 1024 * 1024 * 1024
_CHUNK_BYTES = 1024 * 1024


class CaptureUploadTransferError(ValueError):
    """Fail-closed transfer error containing stable, secret-free blockers."""

    def __init__(self, blockers: Sequence[str]):
        self.blockers = tuple(sorted(set(str(row) for row in blockers if str(row))))
        super().__init__(";".join(self.blockers))


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _opaque_identifier(value: Any, *, field: str, max_length: int = 256) -> str:
    text = value.strip() if isinstance(value, str) else ""
    if (
        not text
        or len(text) > max_length
        or any(ord(character) < 32 or ord(character) == 127 for character in text)
    ):
        raise CaptureUploadTransferError([f"{field}_invalid"])
    return text


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (_canonical(value) + "\n").encode("utf-8")
    try:
        with path.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise CaptureUploadTransferError(["capture_upload_receipt_conflict"])


def _parse_expiry(value: Any) -> datetime | None:
    text = _text(value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _allowed_hosts(value: Sequence[str] | None) -> set[str]:
    rows = value if value is not None else os.getenv(CAPTURE_UPLOAD_ALLOWED_HOSTS_ENV, "").split(",")
    return {_text(row).lower().rstrip(".") for row in rows if _text(row)}


def _validate_transfer_url(url: str, allowed_hosts: set[str]) -> None:
    parsed = urllib_parse.urlsplit(url)
    host = (parsed.hostname or "").lower().rstrip(".")
    if (
        parsed.scheme != "https"
        or not host
        or host not in allowed_hosts
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or parsed.query
    ):
        raise CaptureUploadTransferError(["capture_transfer_url_not_allowed"])


class _ValidatedRedirect(urllib_request.HTTPRedirectHandler):
    def __init__(self, allowed_hosts: set[str]):
        super().__init__()
        self._allowed_hosts = allowed_hosts

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        _validate_transfer_url(str(newurl), self._allowed_hosts)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_transfer(
    *, url: str, authorization: str, allowed_hosts: set[str], timeout_seconds: float
) -> ContextManager[BinaryIO]:
    _validate_transfer_url(url, allowed_hosts)
    request = urllib_request.Request(
        url,
        headers={"Authorization": authorization, "Accept-Encoding": "identity"},
        method="GET",
    )
    opener = urllib_request.build_opener(_ValidatedRedirect(allowed_hosts))
    return closing(opener.open(request, timeout=max(1.0, timeout_seconds)))


def _media_shape_valid(path: Path, *, profile: str, media_type: str) -> bool:
    if media_type not in _MEDIA_TYPES.get(profile, set()):
        return False
    with path.open("rb") as stream:
        prefix = stream.read(32)
    return len(prefix) >= 12 and prefix[4:8] == b"ftyp"


def _configured_malware_scan(path: Path) -> dict[str, Any]:
    raw = _text(os.getenv(CAPTURE_MALWARE_SCANNER_ARGV_ENV))
    if not raw:
        raise CaptureUploadTransferError(["malware_scanner_not_configured"])
    try:
        argv = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CaptureUploadTransferError(["malware_scanner_configuration_invalid"]) from exc
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(row, str) and row for row in argv)
        or not Path(argv[0]).is_absolute()
        or not Path(argv[0]).is_file()
    ):
        raise CaptureUploadTransferError(["malware_scanner_configuration_invalid"])
    try:
        completed = subprocess.run(
            [*argv, str(path)],
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CaptureUploadTransferError(["malware_scanner_failed"]) from exc
    if completed.returncode == 1:
        raise CaptureUploadTransferError(["malware_detected"])
    if completed.returncode != 0:
        raise CaptureUploadTransferError(["malware_scanner_failed"])
    return {
        "status": "passed",
        "scanner": Path(argv[0]).name,
        "scanner_result": "clean",
    }


def _verified_submission(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        submission = json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise CaptureUploadTransferError(["capture_upload_submission_not_json"]) from exc
    blockers: list[str] = []
    if submission.get("schema_version") != CAPTURE_UPLOAD_TRANSFER_SCHEMA_VERSION:
        blockers.append("capture_upload_submission_schema_invalid")
    try:
        submission["capture_session_id"] = strict_identifier(
            submission.get("capture_session_id"),
            field="capture_session_id",
            max_length=128,
        )
    except ValueError:
        blockers.append("capture_session_id_invalid")
    for field in ("customer_id", "organization_id"):
        try:
            submission[field] = _opaque_identifier(
                submission.get(field), field=field
            )
        except CaptureUploadTransferError:
            blockers.append(f"{field}_invalid")
    request = _mapping(submission.get("request"))
    if request.get("schema_version") != "capture_upload_session_request.v1":
        blockers.append("capture_upload_request_schema_invalid")
    for field in ("intake_id", "scene_id"):
        try:
            request[field] = strict_identifier(
                request.get(field), field=field, max_length=128
            )
        except ValueError:
            blockers.append(f"capture_upload_request_{field}_invalid")
    try:
        request["idempotency_key"] = _opaque_identifier(
            request.get("idempotency_key"),
            field="capture_upload_request_idempotency_key",
        )
    except CaptureUploadTransferError:
        blockers.append("capture_upload_request_idempotency_key_invalid")
    profile = _text(request.get("capture_authority_profile"))
    if profile not in _WEB_PROFILES or request.get("source_type") != profile:
        blockers.append("capture_upload_authority_profile_invalid")
    original = _mapping(request.get("original_file"))
    filename = _text(original.get("original_filename"))
    if not filename or Path(filename).name != filename:
        blockers.append("capture_upload_original_filename_invalid")
    size = original.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or not (0 < size <= _MAX_BYTES):
        blockers.append("capture_upload_size_invalid")
    if _text(original.get("media_type")) not in _MEDIA_TYPES.get(profile, set()):
        blockers.append("capture_upload_media_type_invalid")
    for field in (
        "capture_device",
        "timing_declaration",
        "coordinate_frame_declaration",
        "governance",
    ):
        if not _mapping(request.get(field)):
            blockers.append(f"capture_upload_request_{field}_invalid")
    if not isinstance(request.get("available_sensor_streams"), list):
        blockers.append("capture_upload_request_sensor_streams_invalid")
    transfer = _mapping(submission.get("transfer"))
    if transfer.get("provider") != "backblaze":
        blockers.append("capture_transfer_provider_invalid")
    if not _text(transfer.get("url")) or not _text(transfer.get("authorization")):
        blockers.append("capture_transfer_grant_missing")
    expiry = _parse_expiry(transfer.get("expires_at_iso"))
    if expiry is None or expiry <= datetime.now(timezone.utc):
        blockers.append("capture_transfer_grant_expired_or_invalid")
    if blockers:
        raise CaptureUploadTransferError(blockers)
    submission["request"] = request
    return submission


def _build_envelope(
    *, submission: Mapping[str, Any], sha256_value: str, malware: Mapping[str, Any]
) -> dict[str, Any]:
    request = _mapping(submission["request"])
    original = _mapping(request["original_file"])
    filename = _text(original["original_filename"])
    streams = []
    for row_value in request["available_sensor_streams"]:
        row = _mapping(row_value)
        if row.get("status") in {"available", "diagnostic"}:
            row["source_relative_path"] = filename
        streams.append(row)
    return {
        "schema_version": "capture_intake_envelope.v1",
        "intake_id": request["intake_id"],
        "idempotency_key": request["idempotency_key"],
        "capture_authority_profile": request["capture_authority_profile"],
        "source_type": request["source_type"],
        "original_files": [{
            "original_filename": filename,
            "relative_path": filename,
            "sha256": sha256_value,
            "size_bytes": original["size_bytes"],
            "media_type": original["media_type"],
        }],
        "scene_id": request["scene_id"],
        "customer_id": submission["customer_id"],
        "organization_id": submission["organization_id"],
        "capture_device": request["capture_device"],
        "timing_declaration": request["timing_declaration"],
        "coordinate_frame_declaration": request["coordinate_frame_declaration"],
        "available_sensor_streams": streams,
        "governance": request["governance"],
        "requested_task_evaluation_run_audience": request[
            "requested_task_evaluation_run_audience"
        ],
        "known_task_specification": request.get("known_task_specification"),
        "calibration_board_dimensions": request.get("calibration_board_dimensions"),
        "operator_notes": request.get("operator_notes", []),
        "permitted_reconstruction_providers": request.get(
            "permitted_reconstruction_providers", []
        ),
        "permitted_evidence_uses": request.get("permitted_evidence_uses", []),
        "upload_validation": {
            "status": "passed",
            "method": "server_stream_size_sha256_and_media_shape",
            "provider_parts_preverified": True,
        },
        "malware_content_validation": dict(malware),
    }


def process_capture_upload_submission(
    value: Mapping[str, Any],
    *,
    store_root: Path,
    allowed_hosts: Sequence[str] | None = None,
    transfer_opener: Callable[..., ContextManager[BinaryIO]] | None = None,
    malware_scanner: Callable[[Path], Mapping[str, Any]] | None = None,
    qa_builder: Callable[..., Mapping[str, Any]] | None = None,
    timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    """Stream, verify, scan, content-address, and admit one completed Web upload."""

    submission = _verified_submission(value)
    request = _mapping(submission["request"])
    binding = {
        "capture_session_id": submission["capture_session_id"],
        "customer_id": submission["customer_id"],
        "organization_id": submission["organization_id"],
        "request": request,
    }
    request_digest = _digest(binding)
    receipt_key = hashlib.sha256(
        f"{submission['capture_session_id']}\0{request['intake_id']}".encode("utf-8")
    ).hexdigest()
    receipt_path = store_root / "transfer_receipts" / f"{receipt_key}.json"
    if any(
        (store_root / directory / f"{receipt_key}.json").is_file()
        for directory in ("lifecycle_markers", "lifecycle_tombstones")
    ):
        raise CaptureUploadTransferError(["capture_upload_revoked_or_expired"])
    if receipt_path.is_file():
        try:
            existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CaptureUploadTransferError(["capture_upload_receipt_invalid"]) from exc
        if existing.get("request_digest") != request_digest:
            raise CaptureUploadTransferError(["capture_upload_idempotency_conflict"])
        return {**existing, "already_exists": True}

    hosts = _allowed_hosts(allowed_hosts)
    if not hosts:
        raise CaptureUploadTransferError(["capture_transfer_allowed_hosts_not_configured"])
    transfer = _mapping(submission["transfer"])
    url = _text(transfer["url"])
    authorization = _text(transfer["authorization"])
    _validate_transfer_url(url, hosts)
    filename = _text(_mapping(request["original_file"])["original_filename"])
    expected_size = int(_mapping(request["original_file"])["size_bytes"])
    quarantine_root = store_root / "quarantine"
    quarantine_root.mkdir(parents=True, exist_ok=True)
    opener = transfer_opener or _open_transfer
    scanner = malware_scanner or _configured_malware_scan
    try:
        with tempfile.TemporaryDirectory(prefix="capture-transfer-", dir=quarantine_root) as temp:
            upload_root = Path(temp)
            destination = upload_root / filename
            digest = hashlib.sha256()
            total = 0
            try:
                source_context = opener(
                    url=url,
                    authorization=authorization,
                    allowed_hosts=hosts,
                    timeout_seconds=timeout_seconds,
                )
                with source_context as source, destination.open("xb") as output:
                    while True:
                        chunk = source.read(_CHUNK_BYTES)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > expected_size:
                            raise CaptureUploadTransferError(["capture_transfer_size_mismatch"])
                        digest.update(chunk)
                        output.write(chunk)
            except CaptureUploadTransferError:
                raise
            except Exception as exc:
                raise CaptureUploadTransferError(["capture_transfer_download_failed"]) from exc
            if total != expected_size:
                raise CaptureUploadTransferError(["capture_transfer_size_mismatch"])
            if not _media_shape_valid(
                destination,
                profile=_text(request["capture_authority_profile"]),
                media_type=_text(_mapping(request["original_file"])["media_type"]),
            ):
                raise CaptureUploadTransferError(["capture_media_shape_invalid"])
            malware = dict(scanner(destination))
            if malware.get("status") != "passed" or not _text(malware.get("scanner")):
                raise CaptureUploadTransferError(["malware_scan_not_passed"])
            sha256_value = "sha256:" + digest.hexdigest()
            envelope = _build_envelope(
                submission=submission,
                sha256_value=sha256_value,
                malware=malware,
            )
            try:
                materialized = materialize_capture_intake(
                    envelope,
                    upload_root=upload_root,
                    store_root=store_root,
                )
            except CaptureIntakeError as exc:
                raise CaptureUploadTransferError(
                    [f"capture_intake:{blocker}" for blocker in exc.errors]
                ) from exc
            build_qa = qa_builder or build_capture_qa_report
            try:
                candidate_qa_report = dict(
                    build_qa(materialized.envelope, upload_root=upload_root)
                )
                capture_qa_report = dict(
                    build_capture_qa_webapp_publication(
                        capture_session_id=str(submission["capture_session_id"]),
                        report=candidate_qa_report,
                    )["report"]
                )
            except CaptureIntakeError as exc:
                raise CaptureUploadTransferError(
                    [f"capture_qa:{blocker}" for blocker in exc.errors]
                ) from exc
            except ValueError as exc:
                raise CaptureUploadTransferError(["capture_qa:report_invalid"]) from exc
            _write_once(
                materialized.artifact_root / "capture_qa_report.json",
                capture_qa_report,
            )
    finally:
        # Do not retain the only in-process copies of the grant beyond transfer.
        url = ""
        authorization = ""

    receipt = {
        "schema_version": CAPTURE_UPLOAD_RECEIPT_SCHEMA_VERSION,
        "capture_session_id": submission["capture_session_id"],
        "intake_id": request["intake_id"],
        "capture_upload_received_at": datetime.now(timezone.utc).isoformat(),
        "request_digest": request_digest,
        "envelope_digest": materialized.envelope["envelope_digest"],
        "capture_digest": materialized.content_objects[0]["sha256"],
        "size_bytes": materialized.content_objects[0]["size_bytes"],
        "admission_status": materialized.admission["status"],
        "state": materialized.admission["state"],
        "claim_ceiling": materialized.admission["claim_ceiling"],
        "artifact_reference": {
            "uri": str(materialized.artifact_root.relative_to(store_root)),
            "envelope_digest": materialized.envelope["envelope_digest"],
        },
        "malware_content_validation": dict(materialized.envelope["malware_content_validation"]),
        "capture_qa_report": capture_qa_report,
        "already_exists": False,
        "proof_boundary": {
            "server_sha256_verified": True,
            "raw_input_content_addressed": True,
            "capture_qa_completed": True,
            "task_success_established": False,
            "physical_task_success_established": False,
            "deployment_or_safety_approved": False,
            "comparative_policy_ranking_verdict": "thesis_not_supported",
        },
    }
    _write_once(receipt_path, receipt)
    return receipt


__all__ = [
    "CAPTURE_MALWARE_SCANNER_ARGV_ENV",
    "CAPTURE_UPLOAD_ALLOWED_HOSTS_ENV",
    "CAPTURE_UPLOAD_RECEIPT_SCHEMA_VERSION",
    "CAPTURE_UPLOAD_STORE_ROOT_ENV",
    "CAPTURE_UPLOAD_TRANSFER_SCHEMA_VERSION",
    "CaptureUploadTransferError",
    "process_capture_upload_submission",
]
