"""Pure family binding of the standing diagnostic control-trial omission.

The platform pause omits two robot-present control trials. It does not omit
native robot-free physics, reset, cameras, independent scoring or media, and
does not grant any source, runtime, policy, provider or spending authority.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from .decision_evidence_contracts import canonical_digest
from . import task_evaluation_control_stage_policy as policy

SCHEMA = "task_evaluation_diagnostic_control_omission_authority.v1"
OMITTED_CONTROLS = ("zero_action_negative", "deterministic_scripted_positive")
_SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REQUIRED = {"schema_version", "run_kind", "claim_ceiling", "authorized_by", "authorization_reference",
    "omitted_controls", "source_task_success_contract_digest", "result_task_success_contract_digest",
    "task_scoring_criteria_changed", "qualified_comparison_permitted", "authority_digest"}


def validate_family_control_omission(value: Any, task_contract_digest: str) -> bool:
    """Validate retained authority without consulting today's pause flag.

    Existing typed authority may lack the explanatory ``user_request`` field.
    Exact source ownership, expiry and revocation remain outer admission facts.
    """
    if not isinstance(value, Mapping) or not _SHA.fullmatch(str(task_contract_digest)):
        return False
    try:
        row = json.loads(json.dumps(dict(value), allow_nan=False))
        return (_REQUIRED <= set(row) <= _REQUIRED | {"user_request"}
            and row.get("schema_version") == SCHEMA
            and row.get("run_kind") == "internal_policy_canary"
            and row.get("claim_ceiling") == "diagnostic_policy_execution"
            and row.get("authorized_by") == policy.AUTHORIZED_BY
            and row.get("authorization_reference") == policy.AUTHORIZATION_REFERENCE
            and ("user_request" not in row or row["user_request"] == policy.USER_REQUEST)
            and row.get("omitted_controls") == list(OMITTED_CONTROLS)
            and row.get("source_task_success_contract_digest") == task_contract_digest
            and row.get("result_task_success_contract_digest") == task_contract_digest
            and row.get("task_scoring_criteria_changed") is False
            and row.get("qualified_comparison_permitted") is False
            and row.get("authority_digest") == canonical_digest(row, digest_field="authority_digest"))
    except (ValueError, TypeError):
        return False


def make_family_control_omission(task_contract_digest: str) -> dict[str, Any] | None:
    """Bind the existing platform instruction to unchanged diagnostic scoring."""
    if not isinstance(task_contract_digest, str) or not _SHA.fullmatch(task_contract_digest):
        raise ValueError("task_family_control_omission_contract_digest_invalid")
    if not policy.CONTROLS_PAUSED:
        return None
    value = {"schema_version": SCHEMA, "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution", "authorized_by": policy.AUTHORIZED_BY,
        "authorization_reference": policy.AUTHORIZATION_REFERENCE, "user_request": policy.USER_REQUEST,
        "omitted_controls": list(OMITTED_CONTROLS),
        "source_task_success_contract_digest": task_contract_digest,
        "result_task_success_contract_digest": task_contract_digest,
        "task_scoring_criteria_changed": False, "qualified_comparison_permitted": False}
    value["authority_digest"] = canonical_digest(value, digest_field="authority_digest")
    return value


__all__ = ["make_family_control_omission", "validate_family_control_omission", "OMITTED_CONTROLS"]
