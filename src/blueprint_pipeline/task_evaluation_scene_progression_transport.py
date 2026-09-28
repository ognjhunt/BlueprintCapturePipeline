"""Single-shot signed Website preparation calls through the existing clients."""
from __future__ import annotations

from datetime import datetime, timezone
import uuid


def submit_owned_preparation(*, request_path, config, intent_reference):
    """Continue an already-authenticated owner intent on the same control plane.

    The queue operation is preparation-only. It neither creates paid authority
    nor needs a second Website round trip after the signed intent was accepted.
    """
    from .decision_evidence_contracts import cross_runtime_canonical_digest
    from .task_evaluation_scene_owner_authority import reopen_scene_intent
    from .task_evaluation_scene_configuration_submission_inputs import read
    from .task_evaluation_scene_progression_state import require, safe_path
    from .task_evaluation_launch_preparation_contract import validate_launch_preparation_request
    from .task_evaluation_launch_preparation_queue import stage_launch_preparation_request
    intent = reopen_scene_intent(intent_reference)
    request = validate_launch_preparation_request(read(request_path))
    require(request.get("scene_intent_digest") == intent["intent_digest"]
            and request["task"]["identity"]["id"] == intent["request"]["task"]["task_id"]
            and request["run_mode"] == "scene_configuration", "owned_preparation_scope_mismatch")
    receipt = stage_launch_preparation_request(value=request,
        queue_root=safe_path(config["preparation_queue_root"]),
        submitted_by="scene-progression:" + intent["intent_id"])
    return {"schema_version": "task_evaluation_owned_preparation_submission.v1",
        "status": "submitted", "preparation_id": request["preparation_id"],
        "request_digest": cross_runtime_canonical_digest(request),
        "scene_intent_digest": intent["intent_digest"], "queue_receipt": receipt,
        "transport": "local_owned_queue", "provider_mutation_performed": False,
        "paid_execution_requested_by_this_tool": False}


def submit_preparation(*, request_path, config):
    from scripts import submit_task_evaluation_preparation_via_webapp as client
    request, body = client.read_exact_preparation_request(request_path)
    secret = client.launch_client.read_private_secret_file(config["submission_secret_file"])
    timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    nonce = uuid.uuid4().hex
    endpoint = config["submission_endpoint"]
    headers = client.signed_headers(secret=secret, body=body, timestamp=timestamp, nonce=nonce,
                                    preparation_id=request["preparation_id"])
    code, response = client.post_signed_preparation(endpoint=endpoint, headers=headers, body=body,
        timeout_seconds=config.get("http_timeout_seconds", 30))
    if secret in response:
        raise ValueError("scene_progression_response_reflected_secret")
    receipt = client.validate_webapp_preparation_receipt(status_code=code, response_body=response,
        request=request, allow_replay=True)
    return client.build_submission_evidence(endpoint=endpoint, status_code=code, request=request,
        request_body=body, response_body=response, webapp_receipt=receipt, timestamp=timestamp, nonce=nonce)


def read_preparation_status(*, request_path, config):
    from scripts import read_task_evaluation_preparation_status_via_webapp as status_client
    from scripts import submit_task_evaluation_preparation_via_webapp as client
    request, _ = client.read_exact_preparation_request(request_path)
    secret = client.launch_client.read_private_secret_file(config["submission_secret_file"])
    headers = client.signed_headers(secret=secret, body=b"",
        timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"), nonce=uuid.uuid4().hex,
        preparation_id=request["preparation_id"])
    try:
        code, response = status_client.get_signed_status(
            endpoint=status_client._status_endpoint(config["submission_endpoint"], request["preparation_id"]),
            headers=headers, timeout_seconds=config.get("http_timeout_seconds", 30))
    except status_client.WebAppPreparationStatusError as exc:
        if str(exc) == "webapp_preparation_status_http_error_404":
            return {"status": "not_found", "authoritative": True, "http_status": 404,
                    "preparation_id": request["preparation_id"],
                    "request_digest": client._canonical_artifact_digest(request, "request_digest")}
        raise
    if code != 200 or secret in response:
        raise ValueError("scene_progression_status_response_invalid")
    result = status_client.validate_webapp_preparation_status(response_body=response, request=request)
    return {"status": result["state"], "authoritative": True, "http_status": code,
            "preparation_id": request["preparation_id"], "request_digest": result["request_digest"],
            "webapp_status": result}
