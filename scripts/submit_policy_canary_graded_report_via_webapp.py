#!/usr/bin/env python3
"""Submit one immutable graded-report sidecar for a published run to the WebApp.

Backlog: ADP-009D (Day-28 ``public_data_rehearsal``). The sidecar is built by
``python -m blueprint_pipeline.policy_episode_graded_report``; this tool only
signs and posts its exact bytes and checks the WebApp's receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import urllib.parse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blueprint_pipeline.decision_evidence_contracts import (  # noqa: E402
    cross_runtime_canonical_digest,
)
from blueprint_pipeline.policy_episode_graded_report import (  # noqa: E402
    SIDECAR_SCHEMA_VERSION,
)
from scripts import submit_task_evaluation_launch_via_webapp as web_client  # noqa: E402
from scripts.submit_policy_canary_episode_interpretation_backfill_via_webapp import (  # noqa: E402
    signed_pipeline_headers,
)


DEFAULT_ORIGIN = "https://tryblueprint.io"
MAX_REQUEST_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024
RECEIPT_SCHEMA_VERSION = "capture_task_evaluation_graded_report_receipt.v1"
OUTPUT_SCHEMA_VERSION = "task_evaluation_policy_canary_graded_report_webapp_submission.v1"


class GradedReportSubmissionError(ValueError):
    """Secret-clean failure at the signed graded-report boundary."""


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def read_exact_sidecar(path: str | Path) -> tuple[dict[str, Any], bytes]:
    source = Path(path).expanduser()
    if source.is_symlink() or not source.is_file():
        raise GradedReportSubmissionError("graded_report_request_invalid")
    body = source.read_bytes()
    if not body or len(body) > MAX_REQUEST_BYTES:
        raise GradedReportSubmissionError("graded_report_request_size_invalid")
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GradedReportSubmissionError("graded_report_request_invalid") from exc
    if not isinstance(value, Mapping):
        raise GradedReportSubmissionError("graded_report_request_invalid")
    sidecar = dict(value)
    if (
        sidecar.get("schema_version") != SIDECAR_SCHEMA_VERSION
        or sidecar.get("sidecar_digest")
        != cross_runtime_canonical_digest(sidecar, digest_field="sidecar_digest")
    ):
        raise GradedReportSubmissionError("graded_report_request_invalid")
    return sidecar, body


def endpoint_for(*, origin: str, run_id: str) -> str:
    parsed = urllib.parse.urlsplit(origin)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise GradedReportSubmissionError("graded_report_origin_invalid")
    return urllib.parse.urlunsplit(
        (
            "https",
            parsed.netloc,
            "/api/internal/pipeline/capture-task-evaluation-runs/"
            + urllib.parse.quote(run_id, safe="")
            + "/graded-reports",
            "",
            "",
        )
    )


def validate_webapp_receipt(
    *,
    status_code: int,
    response_body: bytes,
    sidecar: Mapping[str, Any],
    allow_replay: bool,
) -> dict[str, Any]:
    if status_code == 200 and not allow_replay:
        raise GradedReportSubmissionError("graded_report_replay_not_authorized")
    if status_code not in {200, 201} or len(response_body) > MAX_RESPONSE_BYTES:
        raise GradedReportSubmissionError("graded_report_response_invalid")
    try:
        value = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GradedReportSubmissionError("graded_report_response_invalid") from exc
    if not isinstance(value, Mapping):
        raise GradedReportSubmissionError("graded_report_response_invalid")
    receipt = dict(value)
    binding = sidecar["source_binding"]
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or receipt.get("run_id") != binding["source_run_id"]
        or receipt.get("result_record_id") != binding["record_id"]
        or receipt.get("sidecar_digest") != sidecar["sidecar_digest"]
        or receipt.get("episode_count") != len(sidecar["episodes"])
        or receipt.get("original_publication_preserved") is not True
        or receipt.get("deterministic_scores_unchanged") is not True
        or receipt.get("ranking_or_promotion_effect") != "none"
        or receipt.get("already_exists") is not (status_code == 200)
    ):
        raise GradedReportSubmissionError("graded_report_response_invalid")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--secret-file", required=True)
    parser.add_argument("--receipt-out", required=True)
    parser.add_argument("--origin", default=DEFAULT_ORIGIN)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--allow-replay", action="store_true")
    args = parser.parse_args(argv)
    reservation = None
    try:
        if args.timeout_seconds <= 0 or not math.isfinite(args.timeout_seconds):
            raise GradedReportSubmissionError("graded_report_timeout_invalid")
        sidecar, body = read_exact_sidecar(args.sidecar)
        binding = sidecar["source_binding"]
        run_id = str(binding["source_run_id"])
        endpoint = endpoint_for(origin=args.origin, run_id=run_id)
        secret = web_client.read_private_secret_file(args.secret_file)
        reservation = web_client.reserve_receipt_exclusive(args.receipt_out)
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        )
        headers = signed_pipeline_headers(secret=secret, body=body, timestamp=timestamp)
        status_code, response = web_client.post_signed_launch(
            endpoint=endpoint,
            headers=headers,
            body=body,
            timeout_seconds=args.timeout_seconds,
        )
        receipt = validate_webapp_receipt(
            status_code=status_code,
            response_body=response,
            sidecar=sidecar,
            allow_replay=args.allow_replay,
        )
        evidence = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "status": "replayed" if status_code == 200 else "submitted",
            "http_status": status_code,
            "endpoint": endpoint,
            "run_id": run_id,
            "record_id": binding["record_id"],
            "submitted_body_digest": _sha256(body),
            "sidecar_digest": sidecar["sidecar_digest"],
            "webapp_response_body_digest": _sha256(response),
            "webapp_receipt": receipt,
            "provider_mutation_performed_by_this_tool": False,
            "observed_at_iso": datetime.now(timezone.utc).isoformat(),
        }
        reservation.seal(evidence)
        reservation = None
    except Exception as exc:
        if reservation is not None:
            reservation.abort()
        print(f"[graded-report] ERROR {type(exc).__name__}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "run_id": evidence["run_id"],
                "record_id": evidence["record_id"],
                "sidecar_digest": evidence["sidecar_digest"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
