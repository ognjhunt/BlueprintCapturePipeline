"""Single-shot signed Website preparation calls through the existing clients."""
from __future__ import annotations

from datetime import datetime, timezone
import uuid


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
