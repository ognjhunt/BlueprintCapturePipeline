"""Client for the one-process persistent Isaac task-state executor.

The command is intentionally only a client.  It cannot emulate articulation in
a fresh process: the worker image must start one Isaac stage/timeline service
and expose ``/apply-and-measure`` for every action in the attempt.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from blueprint_pipeline import safe_outbound_http


EXECUTOR_URL_ENV = "BLUEPRINT_ISAAC_TASK_EXECUTOR_URL"
DEFAULT_EXECUTOR_URL = "http://127.0.0.1:8765/apply-and-measure"
# The executor is an intentionally local sidecar: https anywhere, plain http
# only on loopback (safe_outbound_http fails closed on anything else).
_EXECUTOR_HTTP_POLICY = safe_outbound_http.loopback_service_policy()


def call_persistent_executor(
    request: Mapping[str, Any], *, executor_url: str, timeout_seconds: float = 120.0
) -> dict[str, Any]:
    payload = dict(request)
    if payload.get("schema_version") != "oscar_task_completion_evaluator_request.v1":
        raise ValueError("persistent_isaac_task_request_schema_mismatch")
    action = payload.get("action")
    contract = payload.get("task_success_contract")
    if not isinstance(action, Mapping) or not isinstance(contract, Mapping):
        raise ValueError("persistent_isaac_task_action_or_contract_missing")
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    request_obj = urllib.request.Request(
        executor_url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    response = safe_outbound_http.open_request(
        request_obj,
        policy=_EXECUTOR_HTTP_POLICY,
        timeout_seconds=max(1.0, timeout_seconds),
    )
    raw = response.body.decode("utf-8")
    result = json.loads(raw)
    if not isinstance(result, Mapping):
        raise RuntimeError("persistent_isaac_task_executor_response_not_object")
    normalized = dict(result)
    required = {
        "status",
        "simulator_session_id",
        "stage_id",
        "runtime_result_id",
        "source_action_sha256",
        "articulation_prim_path",
        "before_timestamp",
        "after_timestamp",
        "before_value",
        "after_value",
        "unit",
        "criterion_id",
        "observable_transition",
        "evaluator_attestation",
    }
    missing = sorted(
        field
        for field in required
        if normalized.get(field) is None or normalized.get(field) == ""
    )
    if missing:
        raise RuntimeError("persistent_isaac_task_executor_fields_missing:" + ",".join(missing))
    if normalized.get("persistent_simulator_state_applied") is not True:
        raise RuntimeError("persistent_isaac_task_executor_state_not_applied")
    if normalized.get("official_controller_action_applied") is not True:
        raise RuntimeError("persistent_isaac_task_executor_controller_not_applied")
    return normalized


def main() -> int:
    input_path = os.environ.get("BLUEPRINT_TASK_COMPLETION_INPUT", "")
    output_path = os.environ.get("BLUEPRINT_TASK_COMPLETION_OUTPUT", "")
    if not input_path or not output_path:
        raise SystemExit("BLUEPRINT_TASK_COMPLETION_INPUT and OUTPUT are required")
    request = json.loads(Path(input_path).read_text(encoding="utf-8"))
    result = call_persistent_executor(
        request,
        executor_url=os.environ.get(EXECUTOR_URL_ENV, DEFAULT_EXECUTOR_URL),
    )
    Path(output_path).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
