"""One immutable original delivery, through existing authority and native claims.

No subscription pull, publication, resume intent or provider entitlement is created.
Inspection is observational; dispatch rechecks admission in the native handler.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

FIELDS = {
    "kind",
    "mode",
    "bucket",
    "scene_id",
    "capture_id",
    "marker_generation",
    "handoff_generation",
    "handoff_sha256",
    "handoff_size_bytes",
    "receipt_generation",
    "receipt_sha256",
    "receipt_size_bytes",
}


def validate_selection(value):
    if type(value) is not dict or set(value) != FIELDS:
        raise ValueError("selected_handoff_fields_invalid")
    if (
        value["kind"] != "selected-handoff"
        or type(value["mode"]) is not str
        or value["mode"] not in {"inspect", "dispatch"}
    ):
        raise ValueError("selected_handoff_mode_invalid")
    if not isinstance(value["bucket"], str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]", value["bucket"]
    ):
        raise ValueError("selected_handoff_bucket_invalid")
    for field in ("scene_id", "capture_id"):
        if not isinstance(value[field], str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value[field]
        ):
            raise ValueError("selected_handoff_identity_invalid")
    request = value["capture_id"].removeprefix("walkthrough-")
    if not value["capture_id"].startswith("walkthrough-") or value["scene_id"] != "site-" + request:
        raise ValueError("selected_handoff_identity_invalid")
    for field in ("marker_generation", "handoff_generation", "receipt_generation"):
        if type(value[field]) is not str or not re.fullmatch(r"[1-9][0-9]{0,19}", value[field]):
            raise ValueError("selected_handoff_generation_invalid")
    for role in ("handoff", "receipt"):
        if (
            type(value[role + "_size_bytes"]) is not int
            or not 0 < value[role + "_size_bytes"] <= 65536
        ):
            raise ValueError("selected_handoff_size_invalid")
        if type(value[role + "_sha256"]) is not str or not re.fullmatch(
            r"sha256:[a-f0-9]{64}", value[role + "_sha256"]
        ):
            raise ValueError("selected_handoff_digest_invalid")
    return dict(value)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("selected_handoff_duplicate_key")
        result[key] = value
    return result


def _json(raw):
    def invalid(_value):
        raise ValueError("selected_handoff_nonfinite_json")

    value = json.loads(raw, object_pairs_hook=_unique, parse_constant=invalid)
    if type(value) is not dict:
        raise ValueError("selected_handoff_object_required")
    return value


def _read(client, selection, prefix, role, filename):
    generation = int(selection[role + "_generation"])
    blob = client.bucket(selection["bucket"]).blob(prefix + filename, generation=generation)
    blob.reload(if_generation_match=generation, timeout=20)
    if int(blob.generation) != generation or int(blob.size) != selection[role + "_size_bytes"]:
        raise ValueError("selected_handoff_object_changed")
    raw = blob.download_as_bytes(if_generation_match=generation, timeout=20)
    if (
        len(raw) != selection[role + "_size_bytes"]
        or "sha256:" + hashlib.sha256(raw).hexdigest() != selection[role + "_sha256"]
    ):
        raise ValueError("selected_handoff_object_changed")
    return raw


def current_admission(selection, observation):
    from . import website_task_context as reader
    from .website_assessment_resume import _proposal_admitted

    context = reader.load_current_website_task_context(
        request_id=observation["request_id"],
        scene_id=selection["scene_id"],
        capture_id=selection["capture_id"],
        purpose="scene_preparation",
    )
    authority = reader.load_website_scene_sponsorship(task_context=context, now=time.time())
    _proposal_admitted(authority, context)


def recover_selected_handoff(
    selection,
    *,
    storage_root,
    client=None,
    listener=None,
    owner_reader=None,
    retirement_reader=None,
    admission_reader=None,
    membership_reader=None,
):
    selection = validate_selection(selection)
    if listener is None:
        from . import pubsub_handoff_listener as listener
    if client is None:
        from google.cloud import storage

        client = storage.Client()
    if owner_reader is None:
        from .capture_original_owner_observer import load_original_owner_observation

        owner_reader = load_original_owner_observation
    if retirement_reader is None:
        from .website_scene_workspace_retention import retired_capture_status

        retirement_reader = retired_capture_status
    if membership_reader is None:
        from .capture_delivery_membership import load_selected_capture_membership

        membership_reader = load_selected_capture_membership
    admission_reader = admission_reader or current_admission
    marker = f"scenes/{selection['scene_id']}/captures/{selection['capture_id']}/raw/capture_upload_complete.json"
    delivery = hashlib.sha256(
        json.dumps(
            [selection["bucket"], marker, selection["marker_generation"]],
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    prefix = (
        f"scenes/{selection['scene_id']}/captures/{selection['capture_id']}/deliveries/{delivery}/"
    )
    result = {
        "schema_version": "selected_handoff_recovery.v1",
        "mode": selection["mode"],
        "selector_sha256": hashlib.sha256(
            json.dumps(selection, sort_keys=True).encode()
        ).hexdigest(),
        "provider_dispatch_performed": False,
        "pubsub_pull_performed": False,
        "pubsub_ack_performed": False,
        "new_authority_created": False,
    }
    try:
        raw = _read(client, selection, prefix, "handoff", "pipeline_handoff.json")
        receipt = _json(
            _read(client, selection, prefix, "receipt", "pipeline_handoff_pubsub_receipt.json")
        )
        handoff = _json(raw)
        parsed = listener.parse_handoff_payload(raw)
        finalize = handoff.get("source_finalize")
        if (
            not isinstance(finalize, dict)
            or finalize.get("generation") != selection["marker_generation"]
            or finalize.get("object_name") != marker
            or finalize.get("bucket") != selection["bucket"]
            or parsed.scene_id != selection["scene_id"]
            or parsed.capture_id != selection["capture_id"]
            or handoff.get("handoff_envelope_version") != 2
            or not parsed.source_membership_selector
            or handoff.get("pipeline_handoff_uri")
            != f"gs://{selection['bucket']}/{prefix}pipeline_handoff.json"
            or receipt.get("status") != "published"
            or receipt.get("handoff_envelope_version") != 2
            or not isinstance(receipt.get("message_id"), str)
            or not receipt["message_id"]
            or receipt.get("source_finalize") != finalize
            or receipt.get("scene_id") != selection["scene_id"]
            or receipt.get("capture_id") != selection["capture_id"]
            or receipt.get("pipeline_handoff_uri") != handoff["pipeline_handoff_uri"]
        ):
            raise ValueError("selected_handoff_publication_binding_invalid")
        observation = owner_reader(
            bucket=selection["bucket"],
            scene_id=selection["scene_id"],
            capture_id=selection["capture_id"],
            marker_generation=selection["marker_generation"],
            expected_purpose="scene_preparation",
        )
        if observation["producer_delivery"]["kind"] != "website_browser_capture_delivery":
            raise ValueError("selected_handoff_original_browser_delivery_required")
        # Same finite read-only source preflight as native staging: membership
        # bytes and member metadata only, never raw video download or staging.
        membership_reader(storage_client=client, handoff=parsed, observation=observation)
        result["source_membership_verified"] = True
        retired = retirement_reader(
            storage_root=Path(storage_root),
            bucket=selection["bucket"],
            scene_id=selection["scene_id"],
            capture_id=selection["capture_id"],
        )
        if retired is not None:
            raise ValueError("selected_handoff_retirement_requires_reconciliation")
        capture_root = listener._handoff_capture_root(parsed, storage_root=Path(storage_root))
        # Missing local history is not proof of absent provider effects. Current
        # sponsorship controls must separately authorize the bounded operation.
        if capture_root.exists() and any(capture_root.iterdir()):
            raise ValueError("selected_handoff_existing_state_requires_reconciliation")
        admission_reader(selection, observation)
        result.update(
            status="admitted",
            current_admission_verified=True,
            accounting="Current sponsorship admission; no assertion of zero prior provider charges",
        )
        if selection["mode"] == "dispatch":
            provider = os.getenv("BLUEPRINT_PUBSUB_HANDOFF_PROVIDER", "")
            if provider not in {"local", "claude", "openai"}:
                raise ValueError("selected_handoff_current_provider_missing")
            # Native handler rechecks owner/membership/context/sponsorship and
            # atomically refuses any prior attempt; no synthetic retry record.
            result["provider_dispatch_performed"] = "unknown; consult native receipts"
            native = listener.process_handoff_payload(
                raw,
                storage_root=Path(storage_root),
                provider=provider,
                require_unattempted_delivery=True,
                expected_preparation_purpose="scene_preparation",
            )
            result.update(
                native_status=native.get("status"),
                native_disposition=native.get("queue_disposition"),
            )
            from .capture_original_owner_observer import OWNER_OBSERVATION_REASON_CODES

            reason = native.get("owner_observation_reason")
            if type(reason) is str and reason in OWNER_OBSERVATION_REASON_CODES:
                result["owner_observation_reason"] = reason
            from .capture_delivery_staging import STAGING_REASON_CODES

            reason = native.get("staging_reason")
            if type(reason) is str and reason in STAGING_REASON_CODES:
                result["staging_reason"] = reason
            if (
                native.get("status") != "processed"
                or native.get("queue_disposition") != "terminal_success"
            ):
                return {
                    **result,
                    "status": "blocked",
                    "blockers": ["selected_handoff_native_admission_or_result_blocked"],
                }
            result["status"] = "native_handler_returned"
        return result
    except Exception as exc:
        # Do not return raw provider/source/owner text or private identifiers.
        code = str(exc)
        safe_code = (
            code
            if re.fullmatch(r"selected_handoff_[a-z_]+", code)
            else "selected_handoff_current_lookup_or_admission_failed"
        )
        return {**result, "status": "blocked", "blockers": [safe_code]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", required=True)
    args = parser.parse_args(argv)
    root = os.getenv("BLUEPRINT_PUBSUB_HANDOFF_STORAGE_ROOT", "/var/lib/blueprint/pubsub-handoffs")
    result = recover_selected_handoff(_json(args.request_json), storage_root=Path(root))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
