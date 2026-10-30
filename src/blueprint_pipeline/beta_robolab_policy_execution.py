"""Policy-only RoboLab execution binding, separate from the shared episode loop.

Robot-free probes reuse that loop without importing any model/client machinery.
Actual policy admission, bundle construction and runtime preflight use this
module so fixed local checkpoint validation is preserved at every entrypoint.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from .adp_articulated_task_success_contract import task_kind_of_contract, validate_task_success_contract
from .beta_robolab_runtime import BACKEND
from .decision_evidence_contracts import canonical_digest
from .policy_interface_binding import TEAM_HTTPS_INTERFACES, resolve_policy_interface


def validate_robolab_policy_execution_spec(spec: Mapping[str, Any], *,
                                           candidate_id: str | None = None) -> dict[str, Any]:
    value = json.loads(json.dumps(spec, allow_nan=False))
    if (value.get("schema_version") != "native_task_arena_policy_canary_execution_spec.v1"
            or value.get("evaluation_backend") != BACKEND
            or (candidate_id is not None and value.get("candidate_id") != candidate_id)
            or value.get("execution_spec_digest") != canonical_digest(value, digest_field="execution_spec_digest")
            or not isinstance(value.get("task_success_contract"), dict)
            or value.get("task_success_contract_digest") != value["task_success_contract"].get("contract_digest")):
        raise ValueError("beta_robolab_execution_spec_binding_invalid")
    for key in ("checkpoint_digest", "runtime_identity_digest"):
        digest = value.get(key)
        if not isinstance(digest, str) or len(digest) != 71 or not digest.startswith("sha256:"):
            raise ValueError("beta_robolab_policy_identity_invalid")
        try:
            int(digest[7:], 16)
        except ValueError as exc:
            raise ValueError("beta_robolab_policy_identity_invalid") from exc
    validate_task_success_contract(value["task_success_contract"],
                                   task_kind=task_kind_of_contract(value["task_success_contract"]),
                                   require_confirmed=False)
    interface = resolve_policy_interface(value)
    if interface == "openwam_piper_eef20.v1":
        from .openwam_piper_runtime import validate_execution_spec
        validate_execution_spec(value)
    if interface in TEAM_HTTPS_INTERFACES:
        endpoint = value.get("policy_endpoint")
        if (not isinstance(endpoint, dict) or set(endpoint) != {"credential_env"}
                or re.fullmatch(r"[A-Z][A-Z0-9_]{2,58}_FILE", str(endpoint.get("credential_env") or "")) is None):
            raise ValueError("beta_robolab_https_credential_binding_invalid")
    for field in ("max_policy_queries", "open_loop_horizon"):
        if type(value.get(field)) is not int or value[field] < 1:
            raise ValueError("beta_robolab_episode_limits_invalid")
    if not isinstance(value.get("prompt"), str) or not value["prompt"].strip():
        raise ValueError("beta_robolab_instruction_missing")
    return value

