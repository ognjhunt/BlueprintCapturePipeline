"""Bounded, authenticated observation of the original website capture owner.

This is evidence acquisition only. It cannot authorize a local generation birth,
source staging, retirement, or restore.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from datetime import datetime
from typing import Any
from urllib.parse import quote, urlsplit

from .decision_evidence_contracts import cross_runtime_canonical_digest
from .safe_outbound_http import pinned_api_policy, request as safe_request
from .task_evaluation_launch_webapp_sync import load_pipeline_sync_token
from .webapp_sync import _pipeline_sync_headers, validated_https_sync_url

MAX_RESPONSE_BYTES = 64 * 1024
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z", re.ASCII)
_GENERATION = re.compile(r"[1-9][0-9]{0,19}\Z", re.ASCII)
_SHA = re.compile(r"sha256:[0-9a-f]{64}\Z", re.ASCII)
_BASE64_CRC32C = re.compile(r"[A-Za-z0-9+/]{6}==\Z", re.ASCII)
_RESPONSE_KEYS = {
    "schema_version", "request_id", "scene_id", "capture_id", "bucket",
    "raw_prefix_uri", "capture_owner", "source_document", "ownership_record",
    "consent_attestation", "capture_rights", "source_projection_digest",
    "completion_marker", "producer_delivery", "observed_at_epoch",
    "valid_until_epoch", "observation_digest",
}
_SOURCE_KEYS = (
    "request_id", "scene_id", "capture_id", "bucket", "raw_prefix_uri",
    "capture_owner", "ownership_record", "consent_attestation", "capture_rights",
    "completion_marker", "producer_delivery",
)


class CaptureOwnerObservationError(ValueError):
    """A fixed-code, secret-free refusal of owner evidence."""


def _refuse(code: str) -> None:
    raise CaptureOwnerObservationError(code)


def _object(value: Any, keys: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        _refuse("capture_owner_shape_invalid")
    return value


def _identifier(value: Any, *, max_length: int = 128) -> bool:
    return type(value) is str and len(value) <= max_length and _ID.fullmatch(value) is not None


def _generation(value: Any) -> bool:
    return type(value) is str and _GENERATION.fullmatch(value) is not None


def _sha(value: Any) -> bool:
    return type(value) is str and _SHA.fullmatch(value) is not None


def _integer(value: Any, lower: int, upper: int) -> bool:
    return type(value) is int and lower <= value <= upper


def _iso(value: Any) -> bool:
    if type(value) is not str or not 1 <= len(value) <= 40 or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _gcs_row(value: Any, *, object_name: str | None = None, video: bool = False) -> dict[str, Any]:
    row = _object(value, {"object_name", "generation", "size_bytes", "crc32c" if video else "sha256"})
    name = row["object_name"]
    if (type(name) is not str or not 1 <= len(name.encode("utf-8")) <= 4096
            or name.startswith("/") or "//" in name or "\\" in name or "\x00" in name
            or any(part in ("", ".", "..") for part in name.split("/"))
            or (object_name is not None and name != object_name)
            or not _generation(row["generation"])
            or not _integer(row["size_bytes"], 1, (1 << 53) - 1)):
        _refuse("capture_owner_source_invalid")
    if video:
        checksum = row["crc32c"]
        if type(checksum) is not str or not _BASE64_CRC32C.fullmatch(checksum):
            _refuse("capture_owner_source_invalid")
        try:
            if len(base64.b64decode(checksum, validate=True)) != 4:
                _refuse("capture_owner_source_invalid")
        except ValueError:
            _refuse("capture_owner_source_invalid")
    elif not _sha(row["sha256"]) or row["size_bytes"] > MAX_RESPONSE_BYTES:
        _refuse("capture_owner_source_invalid")
    return row


def validate_observation(
    value: Any, *, bucket: str, scene_id: str, capture_id: str,
    marker_generation: str, now_epoch: int | None = None,
) -> dict[str, Any]:
    """Validate the complete bounded projection against the delivered selector."""
    if not _generation(marker_generation):
        _refuse("capture_owner_marker_generation_invalid")
    if not capture_id.startswith("walkthrough-"):
        _refuse("capture_owner_capture_identity_invalid")
    request_id = capture_id.removeprefix("walkthrough-")
    if (not _identifier(request_id, max_length=120)
            or scene_id != f"site-{request_id}" or not _identifier(bucket)):
        _refuse("capture_owner_capture_identity_invalid")
    value = _object(value, _RESPONSE_KEYS)
    raw_prefix = f"scenes/{scene_id}/captures/{capture_id}/raw"
    if (value["schema_version"] != "website_capture_owner_observation.v1"
            or any(value[key] != expected for key, expected in (
                ("request_id", request_id), ("scene_id", scene_id),
                ("capture_id", capture_id), ("bucket", bucket),
                ("raw_prefix_uri", f"gs://{bucket}/{raw_prefix}")))):
        _refuse("capture_owner_identity_mismatch")
    owner = _object(value["capture_owner"], {"user_id", "basis"})
    if (not _identifier(owner["user_id"])
            or owner["basis"] != "inboundRequests.account_owner_uid"):
        _refuse("capture_owner_identity_invalid")
    source = _object(value["source_document"], {"collection", "document_id", "update_time"})
    update_time = _object(source["update_time"], {"seconds", "nanoseconds"})
    if (source["collection"] != "inboundRequests" or source["document_id"] != request_id
            or not _integer(update_time["seconds"], 0, (1 << 53) - 1)
            or not _integer(update_time["nanoseconds"], 0, 999_999_999)):
        _refuse("capture_owner_source_document_invalid")
    ownership = _object(value["ownership_record"], {"claimed_at_iso"})
    if ownership["claimed_at_iso"] is not None and not _iso(ownership["claimed_at_iso"]):
        _refuse("capture_owner_ownership_record_invalid")
    attestation = _object(value["consent_attestation"],
                          {"granted", "statement_version", "recorded_at_iso"})
    if (attestation["granted"] is not True
            or attestation["statement_version"] != "2026-09-18.v1"
            or not _iso(attestation["recorded_at_iso"])):
        _refuse("capture_owner_attestation_invalid")
    rights = _object(value["capture_rights"], {
        "derived_scene_generation_allowed", "data_licensing_allowed",
        "capture_contributor_payout_eligible", "consent_status", "consent_revoked",
        "consent_scope", "statement_version", "recorded_at_iso",
    })
    if (rights["derived_scene_generation_allowed"] is not True
            or rights["data_licensing_allowed"] is not False
            or rights["capture_contributor_payout_eligible"] is not False
            or rights["consent_status"] != "granted"
            or rights["consent_revoked"] is not False
            or rights["consent_scope"] != ["derived_scene_generation", "robot_evaluation"]
            or rights["statement_version"] != attestation["statement_version"]
            or rights["recorded_at_iso"] != attestation["recorded_at_iso"]):
        _refuse("capture_owner_rights_not_admitted")
    marker = _gcs_row(value["completion_marker"], object_name=f"{raw_prefix}/capture_upload_complete.json")
    if marker["generation"] != marker_generation:
        _refuse("capture_owner_marker_generation_mismatch")
    delivery = _object(value["producer_delivery"], {"kind", "delivery_key", "server_record", "raw_video"})
    if (delivery["kind"] not in ("website_browser_capture_delivery", "website_capture_link_bundle")
            or not _sha(delivery["delivery_key"])):
        _refuse("capture_owner_delivery_invalid")
    receipt = _gcs_row(delivery["server_record"])
    receipt_prefix = f"scenes/{scene_id}/captures/{capture_id}/upload/"
    if not receipt["object_name"].startswith(receipt_prefix):
        _refuse("capture_owner_delivery_invalid")
    if delivery["kind"] == "website_capture_link_bundle":
        expected_receipt = f"{receipt_prefix}bundle_completion.json"
        if receipt["object_name"] != expected_receipt:
            _refuse("capture_owner_delivery_invalid")
    elif not re.fullmatch(r"producer_deliveries/browser-video-[1-9][0-9]{0,19}\.json",
                          receipt["object_name"].removeprefix(receipt_prefix)):
        _refuse("capture_owner_delivery_invalid")
    video = _gcs_row(delivery["raw_video"], video=True)
    if not video["object_name"].startswith(raw_prefix + "/"):
        _refuse("capture_owner_delivery_invalid")
    if delivery["kind"] == "website_browser_capture_delivery":
        expected = f"{receipt_prefix}producer_deliveries/browser-video-{video['generation']}.json"
        if receipt["object_name"] != expected:
            _refuse("capture_owner_delivery_invalid")
    observed = value["observed_at_epoch"]
    valid_until = value["valid_until_epoch"]
    now = int(time.time()) if now_epoch is None else now_epoch
    if (not _integer(observed, 0, (1 << 53) - 1)
            or not _integer(valid_until, 0, (1 << 53) - 1)
            or not _integer(now, 0, (1 << 53) - 1)
            or observed > now + 5 or valid_until <= now
            or not 0 < valid_until - observed <= 60):
        _refuse("capture_owner_observation_stale")
    if (not _sha(value["source_projection_digest"])
            or value["source_projection_digest"] != cross_runtime_canonical_digest({
                key: value[key] for key in _SOURCE_KEYS
            })
            or not _sha(value["observation_digest"])
            or value["observation_digest"] != cross_runtime_canonical_digest(
                value, digest_field="observation_digest")):
        _refuse("capture_owner_observation_digest_invalid")
    return dict(value)


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            _refuse("capture_owner_response_duplicate_key")
        value[key] = item
    return value


def _invalid_constant(_value: str) -> None:
    _refuse("capture_owner_response_nonfinite")


def load_original_owner_observation(
    *, bucket: str, scene_id: str, capture_id: str, marker_generation: str,
    remaining_timeout_ms: int = 10_000,
    include_response_bytes: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], int]:
    """One signed read through the installed Pipeline sync credential and API origin."""
    if not _integer(remaining_timeout_ms, 1, 10_000):
        _refuse("capture_owner_timeout_invalid")
    if not _generation(marker_generation) or not capture_id.startswith("walkthrough-"):
        _refuse("capture_owner_capture_identity_invalid")
    request_id = capture_id.removeprefix("walkthrough-")
    if not _identifier(request_id, max_length=120) or scene_id != f"site-{request_id}":
        _refuse("capture_owner_capture_identity_invalid")
    configured = os.getenv("PIPELINE_SYNC_WEBAPP_URL", "").strip()
    if not configured:
        _refuse("capture_owner_webapp_url_missing")
    try:
        parsed = urlsplit(validated_https_sync_url(configured))
        origin = f"{parsed.scheme}://{parsed.netloc}"
        token = load_pipeline_sync_token()
        body = json.dumps({"request_id": request_id, "scene_id": scene_id,
                           "completion_marker_generation": marker_generation,
                           "remaining_timeout_ms": remaining_timeout_ms},
                          separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        if len(body) > 4096:
            _refuse("capture_owner_request_too_large")
        response = safe_request(
            f"{origin}/api/internal/pipeline/creator-captures/{quote(capture_id, safe='')}/capture-owner",
            method="POST", data=body, headers=_pipeline_sync_headers(token, body),
            timeout_seconds=remaining_timeout_ms / 1000,
            policy=pinned_api_policy(origin, max_response_bytes=MAX_RESPONSE_BYTES),
            max_response_bytes=MAX_RESPONSE_BYTES,
        )
        if response.status != 200 or len(response.body) > MAX_RESPONSE_BYTES:
            _refuse("capture_owner_response_unavailable")
        value = json.loads(response.body.decode("utf-8", errors="strict"),
                           object_pairs_hook=_unique_pairs, parse_constant=_invalid_constant)
        validated = validate_observation(value, bucket=bucket, scene_id=scene_id,
                                         capture_id=capture_id,
                                         marker_generation=marker_generation)
        return (validated, len(response.body)) if include_response_bytes else validated
    except CaptureOwnerObservationError:
        raise
    except Exception as exc:
        raise CaptureOwnerObservationError("capture_owner_response_unavailable") from exc
