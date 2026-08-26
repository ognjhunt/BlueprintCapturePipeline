#!/usr/bin/env python3
"""Wait for one immutable Task Evaluation preparation through production WebApp.

The client uses the preparation service's signed read-only endpoint. It may
synchronize the WebApp record with Pipeline status, but it cannot activate a
scene, publish a profile, allocate a provider, or request paid execution.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__:
    from scripts import submit_task_evaluation_launch_via_webapp as launch_client
    from scripts import submit_task_evaluation_preparation_via_webapp as preparation_client
else:  # pragma: no cover - exercised by the production script entrypoint
    import submit_task_evaluation_launch_via_webapp as launch_client
    import submit_task_evaluation_preparation_via_webapp as preparation_client


DEFAULT_ENDPOINT = preparation_client.DEFAULT_ENDPOINT
OUTPUT_SCHEMA_VERSION = "task_evaluation_launch_preparation_web_status_receipt.v1"
WEB_STATUS_SCHEMA_VERSION = "task_evaluation_launch_preparation_web_status.v1"
PIPELINE_STATUS_SCHEMA_VERSION = "task_evaluation_launch_preparation_status.v1"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
TERMINAL_STATES = {"materialized", "blocked"}
_WEB_STATUS_KEYS = {
    "schema_version",
    "preparation_id",
    "run_id",
    "team_namespace",
    "request_digest",
    "expected_production_commit",
    "state",
    "pipeline",
    "provider_mutation_performed_by_status_read",
    "paid_execution_requested",
    "preparation_is_not_execution",
}
_PIPELINE_STATUS_KEYS = {
    "schema_version",
    "status",
    "preparation_id",
    "run_mode",
    "run_id",
    "team_namespace",
    "expected_production_commit",
    "request_digest",
    "worker_status",
    "source_commit",
    "result_digest",
    "reference_count",
    "full_byte_service_account_readback_passed",
    "blockers",
    "provider_mutation_performed_by_status_read",
    "provider_mutation_performed_by_worker",
    "catalog_mutation_performed_by_worker",
    "paid_execution_requested",
    "construction_orchestration_id",
    "construction_queue_envelope_digest",
    "automatic_progression_required",
    "configured_scene_revision_digest",
    "configured_scene_bundle_digest",
    "episode_compilation_id",
    "episode_compilation_queue_envelope_digest",
}


class WebAppPreparationStatusError(ValueError):
    """A typed, secret-free failure at the Website preparation status boundary."""


def _status_endpoint(base_endpoint: str, preparation_id: str) -> str:
    launch_client._validate_endpoint(base_endpoint)
    parsed = urllib.parse.urlsplit(base_endpoint)
    path = parsed.path.rstrip("/") + "/" + urllib.parse.quote(preparation_id, safe="")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def get_signed_status(
    *, endpoint: str, headers: Mapping[str, str], timeout_seconds: float
) -> tuple[int, bytes]:
    launch_client._validate_endpoint(endpoint)
    request = urllib.request.Request(endpoint, headers=dict(headers), method="GET")
    try:
        with urllib.request.urlopen(  # nosec B310 - HTTPS is enforced above.
            request, timeout=timeout_seconds
        ) as response:
            payload = response.read(MAX_RESPONSE_BYTES + 1)
            if len(payload) > MAX_RESPONSE_BYTES:
                raise WebAppPreparationStatusError(
                    "webapp_preparation_status_response_too_large"
                )
            return int(response.status), payload
    except urllib.error.HTTPError as exc:
        raise WebAppPreparationStatusError(
            f"webapp_preparation_status_http_error_{exc.code}"
        ) from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise WebAppPreparationStatusError(
            "webapp_preparation_status_transport_error"
        ) from exc


def validate_webapp_preparation_status(
    *, response_body: bytes, request: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        value = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WebAppPreparationStatusError(
            "webapp_preparation_status_json_invalid"
        ) from exc
    status = preparation_client._mapping(
        value, blocker="webapp_preparation_status_object_required"
    )
    preparation_client._exact_keys(
        status,
        _WEB_STATUS_KEYS,
        blocker="webapp_preparation_status_fields_invalid",
    )
    if status.get("schema_version") != WEB_STATUS_SCHEMA_VERSION:
        raise WebAppPreparationStatusError("webapp_preparation_status_schema_invalid")
    for field in ("preparation_id", "run_id", "team_namespace", "expected_production_commit"):
        if status.get(field) != request[field]:
            raise WebAppPreparationStatusError(
                f"webapp_preparation_status_{field}_mismatch"
            )
    request_digest = preparation_client._canonical_artifact_digest(
        request, "request_digest"
    )
    if status.get("request_digest") != request_digest:
        raise WebAppPreparationStatusError(
            "webapp_preparation_status_request_digest_mismatch"
        )
    if status.get("provider_mutation_performed_by_status_read") is not False:
        raise WebAppPreparationStatusError(
            "webapp_preparation_status_provider_mutation_invalid"
        )
    if status.get("paid_execution_requested") is not False:
        raise WebAppPreparationStatusError(
            "webapp_preparation_status_paid_execution_invalid"
        )
    if status.get("preparation_is_not_execution") is not True:
        raise WebAppPreparationStatusError(
            "webapp_preparation_status_execution_boundary_invalid"
        )

    pipeline = preparation_client._mapping(
        status.get("pipeline"), blocker="webapp_preparation_pipeline_status_missing"
    )
    if not set(pipeline).issubset(_PIPELINE_STATUS_KEYS):
        raise WebAppPreparationStatusError(
            "webapp_preparation_pipeline_status_fields_invalid"
        )
    required = {
        "schema_version",
        "status",
        "preparation_id",
        "provider_mutation_performed_by_status_read",
    }
    if not required.issubset(pipeline):
        raise WebAppPreparationStatusError(
            "webapp_preparation_pipeline_status_fields_invalid"
        )
    if pipeline.get("schema_version") != PIPELINE_STATUS_SCHEMA_VERSION:
        raise WebAppPreparationStatusError(
            "webapp_preparation_pipeline_status_schema_invalid"
        )
    if pipeline.get("preparation_id") != request["preparation_id"]:
        raise WebAppPreparationStatusError(
            "webapp_preparation_pipeline_status_preparation_id_mismatch"
        )
    if status.get("state") != pipeline.get("status"):
        raise WebAppPreparationStatusError(
            "webapp_preparation_pipeline_status_state_mismatch"
        )
    if pipeline.get("provider_mutation_performed_by_status_read") is not False:
        raise WebAppPreparationStatusError(
            "webapp_preparation_pipeline_status_read_mutation_invalid"
        )
    for field in (
        "provider_mutation_performed_by_worker",
        "catalog_mutation_performed_by_worker",
        "paid_execution_requested",
    ):
        if pipeline.get(field) not in (None, False):
            raise WebAppPreparationStatusError(
                f"webapp_preparation_pipeline_status_{field}_invalid"
            )
    if pipeline.get("status") != "not_found":
        for field in ("run_id", "team_namespace", "expected_production_commit", "request_digest"):
            expected = request_digest if field == "request_digest" else request[field]
            if pipeline.get(field) != expected:
                raise WebAppPreparationStatusError(
                    f"webapp_preparation_pipeline_status_{field}_mismatch"
                )
    if pipeline.get("status") == "materialized":
        if pipeline.get("source_commit") != request["expected_production_commit"]:
            raise WebAppPreparationStatusError(
                "webapp_preparation_pipeline_status_source_commit_mismatch"
            )
        preparation_client._digest(
            pipeline.get("result_digest"),
            blocker="webapp_preparation_pipeline_status_result_digest_invalid",
        )
        if pipeline.get("full_byte_service_account_readback_passed") is not True:
            raise WebAppPreparationStatusError(
                "webapp_preparation_pipeline_status_full_byte_readback_missing"
            )
    return status


def build_status_evidence(
    *,
    endpoint: str,
    request: Mapping[str, Any],
    submitted_body: bytes,
    status_body: bytes | None,
    webapp_status: Mapping[str, Any] | None,
    outcome: str,
    blocker: str | None,
    poll_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "status": outcome,
        "blockers": [] if blocker is None else [blocker],
        "endpoint": endpoint,
        "client_id": preparation_client.CLIENT_ID,
        "preparation_id": request["preparation_id"],
        "run_id": request["run_id"],
        "team_namespace": request["team_namespace"],
        "expected_production_commit": request["expected_production_commit"],
        "submitted_body_digest": preparation_client._sha256(submitted_body),
        "webapp_request_digest": preparation_client._canonical_artifact_digest(
            request, "request_digest"
        ),
        "terminal_response_body_digest": (
            preparation_client._sha256(status_body) if status_body is not None else None
        ),
        "poll_count": poll_count,
        "webapp_status": dict(webapp_status) if webapp_status is not None else None,
        "full_byte_service_account_readback_passed": (
            webapp_status is not None
            and webapp_status.get("pipeline", {}).get(
                "full_byte_service_account_readback_passed"
            )
            is True
        ),
        "provider_mutation_performed_by_this_tool": False,
        "catalog_mutation_performed_by_this_tool": False,
        "paid_execution_requested_by_this_tool": False,
        "observed_at_iso": datetime.now(timezone.utc).isoformat(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="exact submitted preparation JSON")
    parser.add_argument("--secret-file", required=True)
    parser.add_argument("--receipt-out", required=True)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)

    reservation: launch_client.ReceiptReservation | None = None
    evidence: dict[str, Any] | None = None
    exit_code = 2
    try:
        for value in (
            args.timeout_seconds,
            args.request_timeout_seconds,
            args.poll_interval_seconds,
        ):
            if value <= 0 or not math.isfinite(value):
                raise WebAppPreparationStatusError(
                    "webapp_preparation_status_timeout_invalid"
                )
        request, request_body = preparation_client.read_exact_preparation_request(
            args.request
        )
        secret = launch_client.read_private_secret_file(args.secret_file)
        endpoint = _status_endpoint(args.endpoint, request["preparation_id"])
        reservation = launch_client.reserve_receipt_exclusive(args.receipt_out)
        deadline = time.monotonic() + args.timeout_seconds
        poll_count = 0
        terminal_body: bytes | None = None
        terminal_status: dict[str, Any] | None = None
        blocker: str | None = None
        outcome = "blocked"
        while time.monotonic() < deadline:
            poll_count += 1
            timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            headers = preparation_client.signed_headers(
                secret=secret,
                body=b"",
                timestamp=timestamp,
                nonce=uuid.uuid4().hex,
                preparation_id=request["preparation_id"],
            )
            status_code, response_body = get_signed_status(
                endpoint=endpoint,
                headers=headers,
                timeout_seconds=args.request_timeout_seconds,
            )
            if status_code != 200:
                raise WebAppPreparationStatusError(
                    f"webapp_preparation_status_http_status_unexpected_{status_code}"
                )
            if secret in response_body:
                raise WebAppPreparationStatusError(
                    "webapp_preparation_status_response_reflected_secret"
                )
            webapp_status = validate_webapp_preparation_status(
                response_body=response_body,
                request=request,
            )
            if webapp_status["state"] in TERMINAL_STATES:
                terminal_body = response_body
                terminal_status = webapp_status
                outcome = str(webapp_status["state"])
                if outcome == "blocked":
                    pipeline_blockers = webapp_status["pipeline"].get("blockers") or []
                    blocker = (
                        str(pipeline_blockers[0])
                        if pipeline_blockers
                        else "webapp_preparation_terminal_blocked"
                    )
                break
            time.sleep(args.poll_interval_seconds)
        else:
            blocker = "webapp_preparation_status_timeout"
        evidence = build_status_evidence(
            endpoint=endpoint,
            request=request,
            submitted_body=request_body,
            status_body=terminal_body,
            webapp_status=terminal_status,
            outcome=outcome,
            blocker=blocker,
            poll_count=poll_count,
        )
        reservation.seal(evidence)
        reservation = None
        exit_code = 0 if outcome == "materialized" else 3
    except (
        OSError,
        WebAppPreparationStatusError,
        preparation_client.WebAppPreparationSubmissionError,
        launch_client.WebAppLaunchSubmissionError,
    ) as exc:
        if reservation is not None:
            reservation.abort()
        blocker = str(exc) if isinstance(exc, ValueError) else "preparation_status_receipt_write_failed"
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
        "poll_count": evidence["poll_count"],
        "full_byte_service_account_readback_passed": evidence[
            "full_byte_service_account_readback_passed"
        ],
        "receipt_path": str(Path(args.receipt_out).expanduser()),
        "provider_mutation_performed_by_this_tool": False,
        "catalog_mutation_performed_by_this_tool": False,
        "paid_execution_requested_by_this_tool": False,
    }, sort_keys=True))
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
