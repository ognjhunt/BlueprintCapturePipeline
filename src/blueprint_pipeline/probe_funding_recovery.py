"""Exact preparation-funding accounting in the existing robot-free attempt.

A request intention is durable before the create-capable RPC. Historical
readback never issues a grant and a missing remote reservation is not a fence
against a delayed original RPC. Unknown outcomes retain their attempt slot.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math
import os
from pathlib import Path
import time
from typing import Any

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

REQUEST_FILENAME = "website_preparation_spend_request.json"
REFUSAL_FILENAME = "website_preparation_spend_refusal.json"
OBSERVATION_FILENAME = "website_preparation_spend_observation.json"
RECONCILIATION_FILENAME = "website_preparation_spend_reconciliation.json"
FUNDING_FILENAME = "website_preparation_spend_admission.json"
PROTOCOL = "rigid_probe_durable_funding_request.v1"
READBACK_ENV = "BLUEPRINT_PREPARATION_SPEND_OBSERVATION_ENABLED"
# These existing WebApp transaction errors occur before a reservation commit.
_NO_RESERVATION_REFUSALS = frozenset({"consent_expired", "website_scene_preparation_budget_exhausted",
                                      "website_scene_sponsorship_missing"})


def retain_request(job_dir: Path, request: Mapping[str, Any]) -> None:
    from .beta_policy_screening_launch import _write
    value = {"schema_version": "blueprint.probe_funding_request.v1", **dict(request), "request_digest": ""}
    value["request_digest"] = canonical_digest(value, digest_field="request_digest")
    _write(job_dir / REQUEST_FILENAME, value)  # Exclusive: an uncertain request is never reissued.


def request_intent(job_dir: Path) -> dict[str, Any] | None:
    from .allocator_invocation import _read
    path = job_dir / REQUEST_FILENAME
    if not path.exists():
        return None
    value = _read(path)
    payload = value.get("payload") or {}
    command = payload.get("spend") or {}
    if (value.get("schema_version") != "blueprint.probe_funding_request.v1"
            or value.get("request_digest") != canonical_digest(value, digest_field="request_digest")
            or value.get("operation") != "preparation-spend"
            or any(not isinstance(item, str) or not item for item in (
                value.get("capture_id"), payload.get("request_id"), payload.get("scene_id")))
            or command.get("provider") != "vast" or command.get("resource_class") != "gpu_render"
            or command.get("request_count") != 1
            or type(command.get("maximum_cost_usd")) not in (int, float)
            or not math.isfinite(command["maximum_cost_usd"]) or not 0 < command["maximum_cost_usd"] <= 1
            or any(not isinstance(command.get(key), str) or len(command[key]) != 71
                   or not command[key].startswith("sha256:") for key in (
                       "task_context_digest", "allocation_binding_digest"))):
        raise ValueError("probe_funding_request_invalid")
    return value


def retain_refusal(job_dir: Path, message: str) -> None:
    prefix = "website_control_preparation-spend_http_409:"
    code = message.removeprefix(prefix)
    request = request_intent(job_dir)
    if request is None or not message.startswith(prefix) or code not in _NO_RESERVATION_REFUSALS:
        return
    from .beta_policy_screening_launch import _write
    _write(job_dir / REFUSAL_FILENAME, {"schema_version": "blueprint.probe_funding_refusal.v1",
        "request_digest": request["request_digest"], "code": code, "reservation_created": False})


def funding_state(job_dir: Path) -> dict[str, Any]:
    """Conservative local accounting only; it issues no execution capability."""
    from .allocator_invocation import _read
    request = request_intent(job_dir)
    if (job_dir / FUNDING_FILENAME).exists():
        return {"status": "reserved", "funded": True}
    if request is None:
        return {"status": "no_request", "funded": False}
    refusal_path = job_dir / REFUSAL_FILENAME
    if refusal_path.exists():
        refusal = _read(refusal_path)
        if (refusal.get("schema_version") != "blueprint.probe_funding_refusal.v1"
                or refusal.get("request_digest") != request["request_digest"]
                or refusal.get("code") not in _NO_RESERVATION_REFUSALS
                or refusal.get("reservation_created") is not False):
            raise ValueError("probe_funding_refusal_invalid")
        return {"status": "refused", "funded": False, "request_digest": request["request_digest"]}
    return {"status": "unknown", "funded": None, "request_digest": request["request_digest"],
            "allocation_binding_digest": request["payload"]["spend"]["allocation_binding_digest"]}


def validate_observation(value: Mapping[str, Any], *, request: Mapping[str, Any], now: float) -> dict[str, Any]:
    command, payload = request["payload"]["spend"], request["payload"]
    observed = value.get("observed_at_epoch")
    if (value.get("schema_version") != "website_preparation_spend_observation.v1"
            or value.get("capture_id") != request["capture_id"]
            or any(value.get(key) != payload[key] for key in ("request_id", "scene_id"))
            or value.get("command_digest") != cross_runtime_canonical_digest(command)
            or value.get("allocation_binding_digest") != command["allocation_binding_digest"]
            or value.get("status") not in {"reserved", "not_reserved"}
            or value.get("dispatch_authorized") is not False
            or type(observed) not in (int, float) or not math.isfinite(observed) or not 0 < observed <= now + 60):
        raise ValueError("probe_funding_observation_invalid")
    if value["status"] == "reserved":
        admission = value.get("admission") or {}
        if (any(admission.get(key) != expected for key, expected in command.items())
                or admission.get("schema_version") != "paid_lane_admission.v1"
                or admission.get("status") != "admitted" or admission.get("blockers") != []
                or admission.get("external_disclosure_allowed") is not True):
            raise ValueError("probe_funding_observation_admission_invalid")
    elif value.get("admission") is not None or value.get("settlement") is not None:
        raise ValueError("probe_funding_observation_invalid")
    return dict(value)


def reconcile_funding(job_dir: Path, *, reader: Callable[..., Mapping[str, Any]] | None = None,
                      now: float | None = None) -> dict[str, Any]:
    """Bounded exact readback after local fencing; unknown never means unspent."""
    from .allocator_invocation import _read
    from .common import write_json
    from .website_task_context import website_webapp_request
    state = funding_state(job_dir)
    if state["status"] != "unknown":
        return state
    # An older WebApp may ignore create:false and reserve again. Enable this
    # transport only after the read-only server contract has been deployed and
    # verified. Offline injected readers never touch that endpoint.
    if reader is None and os.getenv(READBACK_ENV) != "true":
        return {**state, "last_remote_status": "contract_not_enabled",
                "next_action": "verify_preparation_spend_observation_contract"}
    request = request_intent(job_dir)
    moment = time.time() if now is None else now
    progress_path = job_dir / RECONCILIATION_FILENAME
    previous = _read(progress_path) if progress_path.exists() else {}
    if previous and previous.get("request_digest") != request["request_digest"]:
        raise ValueError("probe_funding_reconciliation_binding_changed")
    retry_at = previous.get("next_retry_at_epoch", 0)
    if type(retry_at) not in (int, float) or not math.isfinite(retry_at) or retry_at < 0:
        raise ValueError("probe_funding_reconciliation_invalid")
    if moment < retry_at:
        return {**state, **previous}
    attempts = previous.get("observations", 0) + 1
    observation = None
    try:
        value = (reader or website_webapp_request)(capture_id=request["capture_id"], operation="preparation-spend",
            payload={**request["payload"], "create": False})
        observation = validate_observation(value, request=request, now=moment)
        write_json(job_dir / OBSERVATION_FILENAME, {"request_digest": request["request_digest"],
                                                   "observation": observation})
    except (OSError, ValueError, RuntimeError):
        pass
    # Even not_reserved is only a snapshot: an old HTTP handler may still commit.
    # Only an exact remote fence could permit another reservation after ambiguity.
    progress = {"schema_version": "blueprint.probe_funding_reconciliation.v1", **state,
        "observations": attempts, "last_remote_status": observation["status"] if observation else "unavailable",
        "next_action": "reconcile_exact_funding_settlement" if observation and observation["status"] == "reserved"
            else "reconcile_exact_funding_outcome_or_remote_fence",
        "next_retry_at_epoch": moment + min(600, 60 * 2 ** min(attempts - 1, 4))}
    if observation and observation["status"] == "reserved":
        progress["funded"] = True
    write_json(progress_path, progress)
    return progress
