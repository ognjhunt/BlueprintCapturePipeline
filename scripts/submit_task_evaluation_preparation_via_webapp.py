#!/usr/bin/env python3
"""Submit one immutable Task Evaluation preparation through production WebApp.

This client targets only the WebApp's preparation service API. It cannot read
admin state, activate a prepared scene, publish a launch profile, allocate a
provider, or request paid execution. The exact JSON bytes read from
``--request`` are the bytes signed and sent.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import re
import sys
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import rfc8785

if __package__:
    from scripts import submit_task_evaluation_launch_via_webapp as launch_client
else:  # pragma: no cover - exercised by the production script entrypoint
    import submit_task_evaluation_launch_via_webapp as launch_client


DEFAULT_ENDPOINT = "https://tryblueprint.io/api/internal/task-evaluation-launch-preparations"
CLIENT_ID = launch_client.CLIENT_ID
WEB_RECEIPT_SCHEMA_VERSION = "task_evaluation_launch_preparation_web_receipt.v1"
INTAKE_RECEIPT_SCHEMA_VERSION = "task_evaluation_launch_preparation_intake_receipt.v1"
OUTPUT_SCHEMA_VERSION = "task_evaluation_launch_preparation_web_submission_receipt.v1"
MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024

TIMESTAMP_HEADER = launch_client.TIMESTAMP_HEADER
CLIENT_ID_HEADER = launch_client.CLIENT_ID_HEADER
NONCE_HEADER = launch_client.NONCE_HEADER
SIGNATURE_HEADER = launch_client.SIGNATURE_HEADER
IDEMPOTENCY_HEADER = launch_client.IDEMPOTENCY_HEADER

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_BASE_REQUEST_KEYS = {
    "schema_version",
    "run_mode",
    "expected_production_commit",
    "preparation_id",
    "team_namespace",
    "run_id",
    "scene",
    "construction",
    "task",
    "sensors",
    "runtime",
    "execution_adapter",
    "publication",
    "spend",
}
_WEB_RECEIPT_KEYS = {
    "schema_version",
    "status",
    "already_exists",
    "preparation_id",
    "run_id",
    "team_namespace",
    "request_digest",
    "expected_production_commit",
    "pipeline",
    "provider_mutation_performed_inside_web_request",
    "catalog_mutation_performed_inside_web_request",
    "paid_execution_requested",
    "preparation_is_not_execution",
    "submission_channel",
}
_FORWARD_KEYS = {"status", "performed", "http_status", "receipt"}
_INTAKE_RECEIPT_KEYS = {
    "schema_version",
    "status",
    "accepted",
    "already_exists",
    "preparation_id",
    "run_id",
    "team_namespace",
    "request_digest",
    "expected_production_commit",
    "provider_mutation_performed_inside_http_request",
    "catalog_mutation_performed_inside_http_request",
    "paid_execution_requested",
    "canonical_allocator_required_for_later_execution",
    "receipt_digest",
}


class WebAppPreparationSubmissionError(ValueError):
    """A typed, secret-free failure at the preparation submission boundary."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    # WebApp's stableJson uses JSON.stringify for primitive values. RFC 8785
    # preserves that ECMAScript number/string serialization while sorting keys,
    # including the important 1.0 -> 1 cross-runtime case.
    try:
        return rfc8785.dumps(dict(value))
    except (rfc8785.FloatDomainError, rfc8785.IntegerDomainError) as exc:
        raise WebAppPreparationSubmissionError(
            "preparation_canonical_json_invalid"
        ) from exc


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _mapping(value: Any, *, blocker: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise WebAppPreparationSubmissionError(blocker)
    return dict(value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, blocker: str) -> None:
    if set(value) != expected:
        raise WebAppPreparationSubmissionError(blocker)


def _identifier(value: Any, *, blocker: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise WebAppPreparationSubmissionError(blocker)
    return value


def _digest(value: Any, *, blocker: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise WebAppPreparationSubmissionError(blocker)
    return value


def _canonical_artifact_digest(value: Mapping[str, Any], digest_field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(digest_field, None)
    return _sha256(_canonical_json(unsigned))


def read_exact_preparation_request(path: str | Path) -> tuple[dict[str, Any], bytes]:
    request_path = Path(path).expanduser()
    try:
        body = request_path.read_bytes()
    except OSError as exc:
        raise WebAppPreparationSubmissionError("preparation_request_unreadable") from exc
    if not body or len(body) > MAX_REQUEST_BYTES:
        raise WebAppPreparationSubmissionError("preparation_request_size_invalid")
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WebAppPreparationSubmissionError("preparation_request_json_invalid") from exc
    request = _mapping(value, blocker="preparation_request_object_required")
    run_mode = request.get("run_mode")
    expected_keys = set(_BASE_REQUEST_KEYS)
    if run_mode == "episode_evaluation":
        expected_keys.update({"robot", "controller"})
    elif run_mode == "scene_configuration":
        if "appearance_review_override" in request:
            expected_keys.add("appearance_review_override")
    else:
        raise WebAppPreparationSubmissionError("preparation_run_mode_invalid")
    _exact_keys(request, expected_keys, blocker="preparation_request_fields_invalid")
    if request.get("schema_version") != "task_evaluation_launch_preparation_request.v1":
        raise WebAppPreparationSubmissionError("preparation_request_schema_invalid")
    _identifier(request.get("preparation_id"), blocker="preparation_id_invalid")
    _identifier(request.get("run_id"), blocker="preparation_run_id_invalid")
    _identifier(request.get("team_namespace"), blocker="preparation_team_namespace_invalid")
    if not isinstance(request.get("expected_production_commit"), str) or not _COMMIT.fullmatch(
        request["expected_production_commit"]
    ):
        raise WebAppPreparationSubmissionError("preparation_commit_invalid")
    for field in (
        "scene",
        "construction",
        "task",
        "sensors",
        "runtime",
        "execution_adapter",
        "publication",
        "spend",
    ):
        _mapping(request.get(field), blocker=f"preparation_{field}_invalid")
    if run_mode == "scene_configuration" and (
        "robot" in request or "controller" in request
    ):
        raise WebAppPreparationSubmissionError("preparation_configuration_execution_binding_invalid")
    return request, body


def signed_headers(
    *, secret: bytes, body: bytes, timestamp: str, nonce: str, preparation_id: str
) -> dict[str, str]:
    canonical = f"{timestamp}.{CLIENT_ID}.{nonce}.".encode("utf-8") + body
    signature = hmac.new(secret, canonical, "sha256").hexdigest()
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        TIMESTAMP_HEADER: timestamp,
        CLIENT_ID_HEADER: CLIENT_ID,
        NONCE_HEADER: nonce,
        SIGNATURE_HEADER: f"sha256={signature}",
        IDEMPOTENCY_HEADER: preparation_id,
    }


def _read_response(response: Any) -> bytes:
    payload = response.read(MAX_RESPONSE_BYTES + 1)
    if len(payload) > MAX_RESPONSE_BYTES:
        raise WebAppPreparationSubmissionError("webapp_preparation_receipt_too_large")
    return payload


def post_signed_preparation(
    *, endpoint: str, headers: Mapping[str, str], body: bytes, timeout_seconds: float
) -> tuple[int, bytes]:
    launch_client._validate_endpoint(endpoint)
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
        raise WebAppPreparationSubmissionError(
            f"webapp_preparation_http_error_{exc.code}"
        ) from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise WebAppPreparationSubmissionError("webapp_preparation_transport_error") from exc


def validate_webapp_preparation_receipt(
    *,
    status_code: int,
    response_body: bytes,
    request: Mapping[str, Any],
    allow_replay: bool,
) -> dict[str, Any]:
    if status_code == 200:
        if not allow_replay:
            raise WebAppPreparationSubmissionError(
                "webapp_preparation_replay_requires_explicit_flag"
            )
    elif status_code != 202:
        raise WebAppPreparationSubmissionError(
            f"webapp_preparation_http_status_unexpected_{status_code}"
        )
    try:
        receipt_value = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WebAppPreparationSubmissionError(
            "webapp_preparation_receipt_json_invalid"
        ) from exc
    receipt = _mapping(receipt_value, blocker="webapp_preparation_receipt_object_required")
    _exact_keys(
        receipt,
        _WEB_RECEIPT_KEYS,
        blocker="webapp_preparation_receipt_fields_invalid",
    )
    if receipt.get("schema_version") != WEB_RECEIPT_SCHEMA_VERSION:
        raise WebAppPreparationSubmissionError("webapp_preparation_receipt_schema_invalid")
    for field in ("preparation_id", "run_id", "team_namespace", "expected_production_commit"):
        if receipt.get(field) != request[field]:
            raise WebAppPreparationSubmissionError(
                f"webapp_preparation_receipt_{field}_mismatch"
            )
    request_digest = _canonical_artifact_digest(request, "request_digest")
    if receipt.get("request_digest") != request_digest:
        raise WebAppPreparationSubmissionError(
            "webapp_preparation_receipt_request_digest_mismatch"
        )
    if receipt.get("status") != "queued_for_no_spend_preparation":
        raise WebAppPreparationSubmissionError("webapp_preparation_receipt_status_invalid")
    if status_code == 202 and receipt.get("already_exists") is not False:
        raise WebAppPreparationSubmissionError("webapp_preparation_receipt_unexpected_replay")
    if status_code == 200 and receipt.get("already_exists") is not True:
        raise WebAppPreparationSubmissionError(
            "webapp_preparation_receipt_replay_binding_invalid"
        )
    if receipt.get("submission_channel") != "production_webapp_service_api":
        raise WebAppPreparationSubmissionError(
            "webapp_preparation_receipt_submission_channel_invalid"
        )
    for field in (
        "provider_mutation_performed_inside_web_request",
        "catalog_mutation_performed_inside_web_request",
        "paid_execution_requested",
    ):
        if receipt.get(field) is not False:
            raise WebAppPreparationSubmissionError(
                f"webapp_preparation_receipt_{field}_invalid"
            )
    if receipt.get("preparation_is_not_execution") is not True:
        raise WebAppPreparationSubmissionError(
            "webapp_preparation_receipt_execution_boundary_invalid"
        )

    forward = _mapping(receipt.get("pipeline"), blocker="webapp_preparation_forward_missing")
    _exact_keys(forward, _FORWARD_KEYS, blocker="webapp_preparation_forward_fields_invalid")
    if (
        forward.get("status") != "forwarded"
        or forward.get("performed") is not True
        or forward.get("http_status") != 202
    ):
        raise WebAppPreparationSubmissionError("webapp_preparation_forward_invalid")
    intake = _mapping(
        forward.get("receipt"), blocker="webapp_preparation_intake_receipt_missing"
    )
    _exact_keys(
        intake,
        _INTAKE_RECEIPT_KEYS,
        blocker="webapp_preparation_intake_receipt_fields_invalid",
    )
    if intake.get("schema_version") != INTAKE_RECEIPT_SCHEMA_VERSION:
        raise WebAppPreparationSubmissionError(
            "webapp_preparation_intake_receipt_schema_invalid"
        )
    if intake.get("status") != "queued_for_no_spend_preparation" or intake.get("accepted") is not True:
        raise WebAppPreparationSubmissionError("webapp_preparation_intake_receipt_status_invalid")
    for field in ("preparation_id", "run_id", "team_namespace", "expected_production_commit"):
        if intake.get(field) != request[field]:
            raise WebAppPreparationSubmissionError(
                f"webapp_preparation_intake_receipt_{field}_mismatch"
            )
    if intake.get("request_digest") != request_digest:
        raise WebAppPreparationSubmissionError(
            "webapp_preparation_intake_receipt_request_digest_mismatch"
        )
    for field in (
        "provider_mutation_performed_inside_http_request",
        "catalog_mutation_performed_inside_http_request",
        "paid_execution_requested",
    ):
        if intake.get(field) is not False:
            raise WebAppPreparationSubmissionError(
                f"webapp_preparation_intake_receipt_{field}_invalid"
            )
    if intake.get("canonical_allocator_required_for_later_execution") is not True:
        raise WebAppPreparationSubmissionError(
            "webapp_preparation_intake_allocator_boundary_invalid"
        )
    _digest(
        intake.get("receipt_digest"),
        blocker="webapp_preparation_intake_receipt_digest_invalid",
    )
    if intake["receipt_digest"] != _canonical_artifact_digest(intake, "receipt_digest"):
        raise WebAppPreparationSubmissionError(
            "webapp_preparation_intake_receipt_digest_mismatch"
        )
    return receipt


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
        "idempotency_key": request["preparation_id"],
        "preparation_id": request["preparation_id"],
        "run_id": request["run_id"],
        "team_namespace": request["team_namespace"],
        "expected_production_commit": request["expected_production_commit"],
        "submitted_body_digest": _sha256(request_body),
        "webapp_request_digest": webapp_receipt["request_digest"],
        "webapp_response_body_digest": _sha256(response_body),
        "webapp_receipt": dict(webapp_receipt),
        "provider_mutation_performed_by_this_tool": False,
        "catalog_mutation_performed_by_this_tool": False,
        "paid_execution_requested_by_this_tool": False,
        "observed_at_iso": datetime.now(timezone.utc).isoformat(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="exact preparation request JSON")
    parser.add_argument(
        "--secret-file",
        required=True,
        help="private file holding the WebApp service HMAC secret",
    )
    parser.add_argument("--receipt-out", required=True)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--allow-replay",
        action="store_true",
        help="accept an exact HTTP 200 already_exists receipt for an intentional retry",
    )
    args = parser.parse_args(argv)

    reservation: launch_client.ReceiptReservation | None = None
    try:
        if args.timeout_seconds <= 0 or not math.isfinite(args.timeout_seconds):
            raise WebAppPreparationSubmissionError("preparation_submit_timeout_invalid")
        request, request_body = read_exact_preparation_request(args.request)
        secret = launch_client.read_private_secret_file(args.secret_file)
        launch_client._validate_endpoint(args.endpoint)
        reservation = launch_client.reserve_receipt_exclusive(args.receipt_out)
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        nonce = uuid.uuid4().hex
        headers = signed_headers(
            secret=secret,
            body=request_body,
            timestamp=timestamp,
            nonce=nonce,
            preparation_id=request["preparation_id"],
        )
        status_code, response_body = post_signed_preparation(
            endpoint=args.endpoint,
            headers=headers,
            body=request_body,
            timeout_seconds=args.timeout_seconds,
        )
        if secret in response_body:
            raise WebAppPreparationSubmissionError("webapp_preparation_response_reflected_secret")
        webapp_receipt = validate_webapp_preparation_receipt(
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
        reservation.seal(evidence)
        reservation = None
    except (
        OSError,
        WebAppPreparationSubmissionError,
        launch_client.WebAppLaunchSubmissionError,
    ) as exc:
        if reservation is not None:
            reservation.abort()
        blocker = str(exc) if isinstance(exc, ValueError) else "preparation_receipt_write_failed"
        print(json.dumps({
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "status": "blocked",
            "blockers": [blocker],
            "provider_mutation_performed_by_this_tool": False,
            "catalog_mutation_performed_by_this_tool": False,
            "paid_execution_requested_by_this_tool": False,
        }, sort_keys=True))
        return 2

    print(json.dumps({
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "status": evidence["status"],
        "preparation_id": evidence["preparation_id"],
        "run_id": evidence["run_id"],
        "webapp_request_digest": evidence["webapp_request_digest"],
        "receipt_path": str(Path(args.receipt_out).expanduser()),
        "provider_mutation_performed_by_this_tool": False,
        "catalog_mutation_performed_by_this_tool": False,
        "paid_execution_requested_by_this_tool": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
