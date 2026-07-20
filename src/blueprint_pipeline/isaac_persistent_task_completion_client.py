"""Client for the one-process persistent Isaac task-state executor.

The command is intentionally only a client.  It cannot emulate articulation in
a fresh process: the worker image must start one Isaac stage/timeline service
and expose ``/apply-and-measure`` for every action in the attempt.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from blueprint_pipeline import safe_outbound_http


EXECUTOR_URL_ENV = "BLUEPRINT_ISAAC_TASK_EXECUTOR_URL"
DEFAULT_EXECUTOR_URL = "http://127.0.0.1:8765/apply-and-measure"
# The executor is an intentionally local sidecar: https anywhere, plain http
# only on loopback (safe_outbound_http fails closed on anything else).
_EXECUTOR_HTTP_POLICY = safe_outbound_http.loopback_service_policy()
POST_ACTION_POLICY_STATE_SOURCE = "post_action_live_isaac_articulation"
UNITREE_G1_SONIC_STATE_DIMS = {
    "left_leg": 6,
    "right_leg": 6,
    "waist": 3,
    "left_arm": 7,
    "right_arm": 7,
    "left_hand": 7,
    "right_hand": 7,
    "projected_gravity": 3,
}


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _validate_post_action_policy_state(
    value: Any,
    *,
    simulator_session_id: str,
    stage_id: str,
    source_action_sha256: str,
    source_step_index: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise RuntimeError("persistent_isaac_post_action_policy_state_missing_or_invalid")
    state = dict(value)
    for field, dimension in UNITREE_G1_SONIC_STATE_DIMS.items():
        values = state.get(field)
        if not isinstance(values, Sequence) or isinstance(
            values, (str, bytes, bytearray)
        ):
            raise RuntimeError(
                f"persistent_isaac_post_action_policy_state_{field}_not_sequence"
            )
        if len(values) != dimension:
            raise RuntimeError(
                "persistent_isaac_post_action_policy_state_"
                f"{field}_dimension_{len(values)}_expected_{dimension}"
            )
        if any(isinstance(item, bool) for item in values):
            raise RuntimeError(
                f"persistent_isaac_post_action_policy_state_{field}_nonfinite"
            )
        try:
            normalized_values = [float(item) for item in values]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"persistent_isaac_post_action_policy_state_{field}_nonfinite"
            ) from exc
        if not all(math.isfinite(item) for item in normalized_values):
            raise RuntimeError(
                f"persistent_isaac_post_action_policy_state_{field}_nonfinite"
            )
        state[field] = normalized_values

    measurement_value = state.get("measurement")
    if not isinstance(measurement_value, Mapping) or not measurement_value:
        raise RuntimeError(
            "persistent_isaac_post_action_policy_state_measurement_missing_or_invalid"
        )
    measurement = dict(measurement_value)
    if measurement.get("surrogate") is not False:
        raise RuntimeError("persistent_isaac_post_action_policy_state_surrogate_not_false")
    if str(measurement.get("source") or "").strip() != POST_ACTION_POLICY_STATE_SOURCE:
        raise RuntimeError(
            "persistent_isaac_post_action_policy_state_source_not_live_post_action_isaac"
        )
    if str(measurement.get("simulator_session_id") or "").strip() != simulator_session_id:
        raise RuntimeError(
            "persistent_isaac_post_action_policy_state_simulator_session_id_mismatch"
        )
    if str(measurement.get("stage_id") or "").strip() != stage_id:
        raise RuntimeError("persistent_isaac_post_action_policy_state_stage_id_mismatch")
    if str(measurement.get("source_action_sha256") or "").strip().lower() != (
        source_action_sha256
    ):
        raise RuntimeError(
            "persistent_isaac_post_action_policy_state_source_action_sha256_mismatch"
        )
    observed_step = measurement.get("source_step_index")
    if (
        isinstance(observed_step, bool)
        or not isinstance(observed_step, int)
        or observed_step != source_step_index
    ):
        raise RuntimeError(
            "persistent_isaac_post_action_policy_state_source_step_index_mismatch"
        )
    captured_at_ns = measurement.get("captured_at_ns")
    if isinstance(captured_at_ns, bool):
        raise RuntimeError("persistent_isaac_post_action_policy_state_captured_at_ns_invalid")
    try:
        captured_at_ns_int = int(captured_at_ns)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "persistent_isaac_post_action_policy_state_captured_at_ns_invalid"
        ) from exc
    if captured_at_ns_int <= 0:
        raise RuntimeError("persistent_isaac_post_action_policy_state_captured_at_ns_invalid")
    state["measurement"] = measurement
    return state


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
    try:
        response = safe_outbound_http.open_request(
            request_obj,
            policy=_EXECUTOR_HTTP_POLICY,
            timeout_seconds=max(1.0, timeout_seconds),
        )
    except urllib.error.HTTPError as exc:
        # The loopback Isaac service returns a small typed JSON failure body.
        # Preserve that exact in-scope diagnostic instead of collapsing every
        # backend failure to the otherwise opaque HTTP 500 traceback.
        raw_error = exc.read(65_537)
        if len(raw_error) > 65_536:
            raise RuntimeError(
                f"persistent_isaac_task_executor_http_{int(exc.code)}:error_body_too_large"
            ) from exc
        error_type = "unknown"
        error_message = "unspecified"
        try:
            error_payload = json.loads(raw_error.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            error_payload = None
        if isinstance(error_payload, Mapping):
            error_type = str(error_payload.get("error_type") or error_type).strip()
            error_message = str(error_payload.get("error") or error_message).strip()
        error_type = " ".join(error_type.split())[:128] or "unknown"
        error_message = " ".join(error_message.split())[:2048] or "unspecified"
        raise RuntimeError(
            "persistent_isaac_task_executor_http_"
            f"{int(exc.code)}:{error_type}:{error_message}"
        ) from exc
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
        "post_action_policy_state",
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
    expected_action_sha256 = _canonical_sha256(action)
    if str(normalized.get("source_action_sha256") or "").strip().lower() != (
        expected_action_sha256
    ):
        raise RuntimeError("persistent_isaac_task_executor_action_sha256_mismatch")
    step_value = payload.get("step_index")
    if isinstance(step_value, bool) or not isinstance(step_value, int):
        raise RuntimeError("persistent_isaac_task_executor_step_index_invalid")
    normalized["post_action_policy_state"] = _validate_post_action_policy_state(
        normalized.get("post_action_policy_state"),
        simulator_session_id=str(normalized.get("simulator_session_id") or "").strip(),
        stage_id=str(normalized.get("stage_id") or "").strip(),
        source_action_sha256=expected_action_sha256,
        source_step_index=step_value,
    )
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
