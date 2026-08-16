#!/usr/bin/env python3
"""Submit one immutable Task Evaluation launch through the production WebApp.

This client deliberately targets the WebApp's launch-only service API.  It does
not call Pipeline intake or a provider directly, and it does not construct or
expand launch authority.  The exact JSON bytes read from ``--request`` are the
bytes signed and sent.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import re
import stat
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_ENDPOINT = "https://tryblueprint.io/api/internal/task-evaluation-launch-submissions"
CLIENT_ID = "blueprint-production-runner"
RECEIPT_SCHEMA_VERSION = "task_evaluation_launch_web_receipt.v1"
OUTPUT_SCHEMA_VERSION = "task_evaluation_launch_web_submission_receipt.v1"
MAX_REQUEST_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024

TIMESTAMP_HEADER = "X-Blueprint-Launch-Timestamp"
CLIENT_ID_HEADER = "X-Blueprint-Launch-Client-Id"
NONCE_HEADER = "X-Blueprint-Launch-Nonce"
SIGNATURE_HEADER = "X-Blueprint-Launch-Signature"
IDEMPOTENCY_HEADER = "Idempotency-Key"

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9._-]{1,192}$")
_REQUEST_KEYS = {
    "launch_id",
    "run_id",
    "profile_id",
    "profile_digest",
    "rights",
    "spend",
    "confirm_execution",
}
_RECEIPT_KEYS = {
    "schema_version",
    "status",
    "already_exists",
    "launch_id",
    "run_id",
    "request_digest",
    "forward",
    "provider_mutation_performed_inside_web_request",
    "submission_channel",
}
_FORWARD_KEYS = {
    "status",
    "performed",
    "required",
    "endpoint_configured",
    "http_status",
    "blocker",
    "queue_receipt",
    "pipeline_intake_status",
}
_QUEUE_RECEIPT_KEYS = {
    "schema_version",
    "status",
    "already_exists",
    "launch_id",
    "run_id",
    "request_digest",
    "launch_profile_id",
    "launch_profile_digest",
    "queue_path",
    "provider_mutation_performed",
}


class WebAppLaunchSubmissionError(ValueError):
    """A typed, secret-free failure at the WebApp submission boundary."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _mapping(value: Any, *, blocker: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise WebAppLaunchSubmissionError(blocker)
    return dict(value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, blocker: str) -> None:
    if set(value) != expected:
        raise WebAppLaunchSubmissionError(blocker)


def _identifier(value: Any, *, blocker: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise WebAppLaunchSubmissionError(blocker)
    return value


def _digest(value: Any, *, blocker: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise WebAppLaunchSubmissionError(blocker)
    return value


def read_exact_launch_request(path: str | Path) -> tuple[dict[str, Any], bytes]:
    """Read and minimally mirror the WebApp's strict public input contract."""

    request_path = Path(path).expanduser()
    try:
        body = request_path.read_bytes()
    except OSError as exc:
        raise WebAppLaunchSubmissionError("launch_request_unreadable") from exc
    if not body or len(body) > MAX_REQUEST_BYTES:
        raise WebAppLaunchSubmissionError("launch_request_size_invalid")
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WebAppLaunchSubmissionError("launch_request_json_invalid") from exc
    request = _mapping(value, blocker="launch_request_object_required")
    _exact_keys(request, _REQUEST_KEYS, blocker="launch_request_fields_invalid")
    _identifier(request.get("launch_id"), blocker="launch_id_invalid")
    _identifier(request.get("run_id"), blocker="run_id_invalid")
    _identifier(request.get("profile_id"), blocker="profile_id_invalid")
    _digest(request.get("profile_digest"), blocker="profile_digest_invalid")

    rights = _mapping(request.get("rights"), blocker="launch_rights_invalid")
    _exact_keys(rights, {"scope", "evidence"}, blocker="launch_rights_invalid")
    scope = rights.get("scope")
    if not isinstance(scope, str) or not scope.strip() or len(scope.strip()) > 1000:
        raise WebAppLaunchSubmissionError("launch_rights_scope_invalid")
    evidence = _mapping(rights.get("evidence"), blocker="launch_rights_evidence_invalid")
    _exact_keys(
        evidence,
        {"uri", "digest"},
        blocker="launch_rights_evidence_invalid",
    )
    if not isinstance(evidence.get("uri"), str) or not evidence["uri"].strip():
        raise WebAppLaunchSubmissionError("launch_rights_evidence_uri_invalid")
    _digest(
        evidence.get("digest"),
        blocker="launch_rights_evidence_digest_invalid",
    )

    spend = _mapping(request.get("spend"), blocker="launch_spend_invalid")
    _exact_keys(spend, {"max_spend_usd", "expires_at"}, blocker="launch_spend_invalid")
    maximum = spend.get("max_spend_usd")
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, (int, float))
        or not math.isfinite(float(maximum))
        or float(maximum) <= 0
    ):
        raise WebAppLaunchSubmissionError("launch_spend_max_invalid")
    expires_at = spend.get("expires_at")
    if not isinstance(expires_at, str):
        raise WebAppLaunchSubmissionError("launch_spend_expiry_invalid")
    try:
        parsed_expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise WebAppLaunchSubmissionError("launch_spend_expiry_invalid") from exc
    if parsed_expiry.tzinfo is None:
        raise WebAppLaunchSubmissionError("launch_spend_expiry_invalid")
    if request.get("confirm_execution") is not True:
        raise WebAppLaunchSubmissionError("launch_execution_confirmation_missing")
    return request, body


def read_private_secret_file(path: str | Path) -> bytes:
    """Read a root/service-readable secret without exposing its path or bytes."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(Path(path).expanduser(), flags)
    except OSError as exc:
        raise WebAppLaunchSubmissionError("launch_submit_secret_unreadable") from exc
    try:
        metadata = os.fstat(descriptor)
        mode = stat.S_IMODE(metadata.st_mode)
        if not stat.S_ISREG(metadata.st_mode):
            raise WebAppLaunchSubmissionError("launch_submit_secret_not_regular_file")
        # Admit 0400/0440/0600/0640-style ownership.  Group-read permits a
        # root-owned file to be consumed by the service account; no group write,
        # execute, or other-user access is admitted.
        if mode & ~0o640 or not mode & 0o440:
            raise WebAppLaunchSubmissionError("launch_submit_secret_file_not_private")
        value = os.read(descriptor, 4097)
        if len(value) > 4096:
            raise WebAppLaunchSubmissionError("launch_submit_secret_size_invalid")
    finally:
        os.close(descriptor)
    secret = value.strip()
    if len(secret) < 32:
        raise WebAppLaunchSubmissionError("launch_submit_secret_too_short")
    return secret


def signed_headers(
    *, secret: bytes, body: bytes, timestamp: str, nonce: str, launch_id: str
) -> dict[str, str]:
    """Sign the exact WebApp launch-only canonical byte sequence."""

    canonical = f"{timestamp}.{CLIENT_ID}.{nonce}.".encode("utf-8") + body
    signature = hmac.new(secret, canonical, "sha256").hexdigest()
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        TIMESTAMP_HEADER: timestamp,
        CLIENT_ID_HEADER: CLIENT_ID,
        NONCE_HEADER: nonce,
        SIGNATURE_HEADER: f"sha256={signature}",
        IDEMPOTENCY_HEADER: launch_id,
    }


def _validate_endpoint(endpoint: str) -> None:
    parsed = urllib.parse.urlsplit(endpoint)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise WebAppLaunchSubmissionError("launch_submit_endpoint_not_https")


def _read_response(response: Any) -> bytes:
    payload = response.read(MAX_RESPONSE_BYTES + 1)
    if len(payload) > MAX_RESPONSE_BYTES:
        raise WebAppLaunchSubmissionError("webapp_receipt_too_large")
    return payload


def post_signed_launch(
    *, endpoint: str, headers: Mapping[str, str], body: bytes, timeout_seconds: float
) -> tuple[int, bytes]:
    _validate_endpoint(endpoint)
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers=dict(headers),
        method="POST",
    )
    try:
        with urllib.request.urlopen(  # nosec B310 - HTTPS is enforced above.
            request, timeout=timeout_seconds
        ) as response:
            return int(response.status), _read_response(response)
    except urllib.error.HTTPError as exc:
        # Do not retain or echo an untrusted error body.  A WebApp error cannot
        # be launch evidence and may contain reflected request data.
        raise WebAppLaunchSubmissionError(f"webapp_http_error_{exc.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise WebAppLaunchSubmissionError("webapp_transport_error") from exc


def validate_webapp_receipt(
    *,
    status_code: int,
    response_body: bytes,
    request: Mapping[str, Any],
    allow_replay: bool,
) -> dict[str, Any]:
    if status_code == 200:
        if not allow_replay:
            raise WebAppLaunchSubmissionError("webapp_replay_requires_explicit_flag")
    elif status_code != 202:
        raise WebAppLaunchSubmissionError(f"webapp_http_status_unexpected_{status_code}")
    try:
        receipt_value = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WebAppLaunchSubmissionError("webapp_receipt_json_invalid") from exc
    receipt = _mapping(receipt_value, blocker="webapp_receipt_object_required")
    _exact_keys(receipt, _RECEIPT_KEYS, blocker="webapp_receipt_fields_invalid")
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise WebAppLaunchSubmissionError("webapp_receipt_schema_invalid")
    if receipt.get("launch_id") != request["launch_id"]:
        raise WebAppLaunchSubmissionError("webapp_receipt_launch_id_mismatch")
    if receipt.get("run_id") != request["run_id"]:
        raise WebAppLaunchSubmissionError("webapp_receipt_run_id_mismatch")
    request_digest = _digest(
        receipt.get("request_digest"), blocker="webapp_receipt_request_digest_invalid"
    )
    if receipt.get("provider_mutation_performed_inside_web_request") is not False:
        raise WebAppLaunchSubmissionError("webapp_receipt_provider_mutation_invalid")
    if receipt.get("submission_channel") != "production_webapp_service_api":
        raise WebAppLaunchSubmissionError("webapp_receipt_submission_channel_invalid")
    if status_code == 202 and receipt.get("already_exists") is not False:
        raise WebAppLaunchSubmissionError("webapp_receipt_unexpected_replay")
    if status_code == 200 and receipt.get("already_exists") is not True:
        raise WebAppLaunchSubmissionError("webapp_receipt_replay_binding_invalid")

    forward = _mapping(receipt.get("forward"), blocker="webapp_forward_receipt_missing")
    if not set(forward).issubset(_FORWARD_KEYS):
        raise WebAppLaunchSubmissionError("webapp_forward_receipt_fields_invalid")
    if (
        forward.get("status") != "forwarded"
        or forward.get("performed") is not True
        or forward.get("endpoint_configured") is not True
    ):
        raise WebAppLaunchSubmissionError("webapp_forward_receipt_invalid")
    pipeline_status = forward.get("pipeline_intake_status")
    expected_web_status = {
        "accepted": "queued_in_pipeline",
        "queued_dispatch_blocked": "queued_dispatch_blocked",
    }.get(pipeline_status)
    if expected_web_status is None or receipt.get("status") != expected_web_status:
        raise WebAppLaunchSubmissionError("webapp_receipt_status_binding_invalid")

    queue = _mapping(forward.get("queue_receipt"), blocker="webapp_queue_receipt_missing")
    _exact_keys(queue, _QUEUE_RECEIPT_KEYS, blocker="webapp_queue_receipt_fields_invalid")
    if queue.get("schema_version") != "task_evaluation_launch_queue_receipt.v1":
        raise WebAppLaunchSubmissionError("webapp_queue_receipt_schema_invalid")
    if queue.get("launch_id") != request["launch_id"]:
        raise WebAppLaunchSubmissionError("webapp_queue_receipt_launch_id_mismatch")
    if queue.get("run_id") != request["run_id"]:
        raise WebAppLaunchSubmissionError("webapp_queue_receipt_run_id_mismatch")
    if queue.get("request_digest") != request_digest:
        raise WebAppLaunchSubmissionError("webapp_queue_receipt_request_digest_mismatch")
    if queue.get("launch_profile_id") != request["profile_id"]:
        raise WebAppLaunchSubmissionError("webapp_queue_receipt_profile_id_mismatch")
    if queue.get("launch_profile_digest") != request["profile_digest"]:
        raise WebAppLaunchSubmissionError("webapp_queue_receipt_profile_digest_mismatch")
    if queue.get("provider_mutation_performed") is not False:
        raise WebAppLaunchSubmissionError("webapp_queue_receipt_provider_mutation_invalid")
    return receipt


def write_receipt_exclusive_atomic(path: str | Path, value: Mapping[str, Any]) -> None:
    """Publish a complete receipt at a path that must not already exist."""

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise WebAppLaunchSubmissionError("submission_receipt_already_exists")
    payload = _canonical_json(value)
    temporary = destination.parent / f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o440)
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise WebAppLaunchSubmissionError("submission_receipt_already_exists") from exc
        temporary.unlink()
        directory_descriptor = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def build_submission_evidence(
    *,
    endpoint: str,
    status_code: int,
    request: Mapping[str, Any],
    request_body: bytes,
    response_body: bytes,
    webapp_receipt: Mapping[str, Any],
    timestamp: str,
    nonce: str,
) -> dict[str, Any]:
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "status": "replayed" if status_code == 200 else "submitted",
        "http_status": status_code,
        "endpoint": endpoint,
        "client_id": CLIENT_ID,
        "request_timestamp": timestamp,
        "request_nonce": nonce,
        "idempotency_key": request["launch_id"],
        "launch_id": request["launch_id"],
        "run_id": request["run_id"],
        "submitted_body_digest": _sha256(request_body),
        "webapp_request_digest": webapp_receipt["request_digest"],
        "webapp_response_body_digest": _sha256(response_body),
        "webapp_receipt": dict(webapp_receipt),
        "provider_mutation_performed_by_this_tool": False,
        "observed_at_iso": datetime.now(timezone.utc).isoformat(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="exact WebApp launch request JSON")
    parser.add_argument(
        "--secret-file",
        required=True,
        help="private file holding the launch-only WebApp HMAC secret",
    )
    parser.add_argument("--receipt-out", required=True)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--allow-replay",
        action="store_true",
        help="accept an exact HTTP 200 already_exists receipt for an intentional retry",
    )
    args = parser.parse_args(argv)

    try:
        if args.timeout_seconds <= 0 or not math.isfinite(args.timeout_seconds):
            raise WebAppLaunchSubmissionError("launch_submit_timeout_invalid")
        if Path(args.receipt_out).expanduser().exists():
            raise WebAppLaunchSubmissionError("submission_receipt_already_exists")
        request, request_body = read_exact_launch_request(args.request)
        secret = read_private_secret_file(args.secret_file)
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        nonce = uuid.uuid4().hex
        headers = signed_headers(
            secret=secret,
            body=request_body,
            timestamp=timestamp,
            nonce=nonce,
            launch_id=request["launch_id"],
        )
        status_code, response_body = post_signed_launch(
            endpoint=args.endpoint,
            headers=headers,
            body=request_body,
            timeout_seconds=args.timeout_seconds,
        )
        if secret in response_body:
            raise WebAppLaunchSubmissionError("webapp_response_reflected_secret")
        webapp_receipt = validate_webapp_receipt(
            status_code=status_code,
            response_body=response_body,
            request=request,
            allow_replay=args.allow_replay,
        )
        evidence = build_submission_evidence(
            endpoint=args.endpoint,
            status_code=status_code,
            request=request,
            request_body=request_body,
            response_body=response_body,
            webapp_receipt=webapp_receipt,
            timestamp=timestamp,
            nonce=nonce,
        )
        write_receipt_exclusive_atomic(args.receipt_out, evidence)
    except (OSError, WebAppLaunchSubmissionError) as exc:
        blocker = (
            str(exc)
            if isinstance(exc, WebAppLaunchSubmissionError)
            else "submission_receipt_write_failed"
        )
        print(
            json.dumps(
                {
                    "schema_version": OUTPUT_SCHEMA_VERSION,
                    "status": "blocked",
                    "blockers": [blocker],
                    "provider_mutation_performed_by_this_tool": False,
                },
                sort_keys=True,
            )
        )
        return 2

    print(
        json.dumps(
            {
                "schema_version": OUTPUT_SCHEMA_VERSION,
                "status": evidence["status"],
                "launch_id": evidence["launch_id"],
                "run_id": evidence["run_id"],
                "webapp_request_digest": evidence["webapp_request_digest"],
                "receipt_path": str(Path(args.receipt_out).expanduser()),
                "provider_mutation_performed_by_this_tool": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
