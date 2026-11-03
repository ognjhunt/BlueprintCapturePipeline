"""The standing control-trial pause, applied to a screening contract that requires per-cell controls.

Owner decision 2026-09-21 (``docs/website-controls-pause-2026-09-21.md``,
``task_evaluation_control_stage_policy``): control trials are paused for all
runs, and website runs bind the existing typed diagnostic omission instead.
Beta screening has no control stage, so a scene whose confirmed owner contract
requires per-cell controls screens under that omission, as the operator canary
does (``native_marked_area_rehearsal.direct_policy_request``): the contract loses
its ``controls`` criterion and nothing else, and the typed
``task_evaluation_diagnostic_control_omission_authority.v1`` names both contract
digests, the diagnostic claim ceiling and that no qualified comparison is
permitted. The authority travels in the run's ``beta_screening_admission`` into
its session result. Without the pause nothing is omitted.

Robot-team evaluations apply the same transform
(``team_candidate_quick10_registry.offer_eligibility``) and carry the authority
in their sealed ``team_candidate_admission`` into the runtime inputs and the
team terminal.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

SCHEMA = "task_evaluation_diagnostic_control_omission_authority.v1"


def _without_controls(contract: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(contract))
    value["criteria"] = {key: row for key, row in value["criteria"].items() if key != "controls"}
    value["contract_digest"] = cross_runtime_canonical_digest(value, digest_field="contract_digest")
    return value


def validate_paused_controls_omission(authority: Mapping[str, Any], *, source: Mapping[str, Any],
                                      contract: Mapping[str, Any]) -> dict[str, Any]:
    """The authority, when it omits exactly the source contract's controls under the platform pause."""

    from . import task_evaluation_control_stage_policy as policy
    from .native_policy_canary_control_gate import controls_required
    from .native_task_arena_policy_canary_session import validate_control_omission_authority

    admitted = validate_control_omission_authority(authority, contract_digest=contract["contract_digest"])
    if (admitted.get("schema_version") != SCHEMA or not controls_required(source)
            or admitted.get("authorized_by") != policy.AUTHORIZED_BY
            or admitted.get("authorization_reference") != policy.AUTHORIZATION_REFERENCE
            or admitted.get("source_task_success_contract_digest") != source.get("contract_digest")
            or _without_controls(source) != dict(contract)):
        raise ValueError("beta_screening_control_omission_invalid")
    return admitted


def omit_paused_controls(contract: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The contract screening runs and its omission authority; unchanged, with none, unless both apply."""

    from . import task_evaluation_control_stage_policy as policy
    from .adp_articulated_task_success_contract import task_kind_of_contract, validate_task_success_contract
    from .native_policy_canary_control_gate import controls_required

    if not policy.CONTROLS_PAUSED or not controls_required(contract):
        return dict(contract), None
    result = _without_controls(contract)
    validate_task_success_contract(result, task_kind=task_kind_of_contract(result))
    authority = {"schema_version": SCHEMA, "run_kind": "internal_policy_canary",
                 "claim_ceiling": "diagnostic_policy_execution", "authorized_by": policy.AUTHORIZED_BY,
                 "authorization_reference": policy.AUTHORIZATION_REFERENCE, "user_request": policy.USER_REQUEST,
                 "omitted_controls": ["zero_action_negative", "deterministic_scripted_positive"],
                 "source_task_success_contract_digest": contract["contract_digest"],
                 "result_task_success_contract_digest": result["contract_digest"],
                 "task_scoring_criteria_changed": False, "qualified_comparison_permitted": False}
    authority["authority_digest"] = canonical_digest(authority, digest_field="authority_digest")
    return result, validate_paused_controls_omission(authority, source=contract, contract=result)


__all__ = ["SCHEMA", "omit_paused_controls", "validate_paused_controls_omission"]
