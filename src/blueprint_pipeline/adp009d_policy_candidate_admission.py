"""Outcome-blind policy inventory and freeze contracts for ADP-009D.

The inventory records publisher-observed source, checkpoint, interface, rights,
and runtime facts before task outcomes exist.  It deliberately does not admit a
checkpoint merely because a public URI, model card, or prior unrelated smoke
receipt exists.  Exactly two candidates may be frozen, and only after separate
runtime-admission receipts bind materialized checkpoint bytes, the calibrated
live observation adapter, the action adapter, and an immutable smoke result.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from blueprint_pipeline.decision_evidence_contracts import canonical_digest


INVENTORY_SCHEMA_VERSION = "adp009d_policy_candidate_inventory.v1"
RUNTIME_ADMISSION_SCHEMA_VERSION = "adp009d_policy_candidate_runtime_admission.v1"
SELECTION_SCHEMA_VERSION = "adp009d_policy_candidate_selection.v1"
PROGRAM_ID = "arm-decision-proof-v1"

EXPECTED_CANDIDATES = {
    "pi05_droid": {
        "source_repository": "https://github.com/Physical-Intelligence/openpi",
        "source_revision": "15a9616a00943ada6c20a0f158e3adb39df2ccac",
        "source_tree": "a7f18af2745255b5fa98c86d6031f858bf73d1be",
        "checkpoint_repository": "gs://openpi-assets/checkpoints/pi05_droid",
        "checkpoint_revision": (
            "gcs-generation-inventory:"
            "4d35968545b296130d4e13e9cc41d0cfba11e69ac1ec99a0ca151122ca81cf12"
        ),
        "checkpoint_total_bytes": 12_429_488_598,
        "checkpoint_inventory_digest": (
            "sha256:4d35968545b296130d4e13e9cc41d0cfba11e69ac1ec99a0ca151122ca81cf12"
        ),
    },
    "groot_n17_droid": {
        "source_repository": "https://github.com/NVIDIA/Isaac-GR00T",
        "source_revision": "b9955401d50c92a29258732e3ad6ccd579f1bdc0",
        "source_tree": "09c80a5529da315117ed962d5d6b794c981d5e72",
        "checkpoint_repository": "https://huggingface.co/nvidia/GR00T-N1.7-DROID",
        "checkpoint_revision": "05e7cc97e40dbd33b0890c35cc0214fcb0547ab5",
        "checkpoint_total_bytes": 6_914_267_987,
        "checkpoint_inventory_digest": (
            "sha256:5d1d83ab34215da2dcaa049d70e93ccec18687591ad5760c5183fc1fd6e035fd"
        ),
    },
    "groot_n16_droid": {
        "source_repository": "https://github.com/NVIDIA/Isaac-GR00T",
        "source_revision": "5dc80c4afd726b34faad1d8f7e007a13b34e4c88",
        "source_tree": "e25e4776db8bf56b555c780fc4e4b4fcc77e2b04",
        "checkpoint_repository": "https://huggingface.co/nvidia/GR00T-N1.6-DROID",
        "checkpoint_revision": "ae3ebe8d288971ac53aa30c756ea5cba0f52611b",
        "checkpoint_total_bytes": 6_573_569_204,
        "checkpoint_inventory_digest": (
            "sha256:a90b219623ae64d3f919f5f7c8ba2a7b3da20df75289fceea61da54e936a79fe"
        ),
    },
    "cosmos3_edge_policy_droid": {
        "source_repository": "https://github.com/NVIDIA/cosmos-framework",
        "source_revision": "5e02e643c458ce06c7232244271f567dce1dec7a",
        "source_tree": "aea30b8215e897c35d0201e51d0323d18da13740",
        "checkpoint_repository": (
            "https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID"
        ),
        "checkpoint_revision": "3ea407af3e156c0af3b4bb6edd85842cc9a58777",
        "checkpoint_total_bytes": 9_171_983_421,
        "checkpoint_inventory_digest": (
            "sha256:f1da4f2c70e31b7c80c0069abdebf5d46d9a45a2e588b3aa84984a862f18568f"
        ),
    },
}

_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_RFC3339_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_FORBIDDEN_OUTCOME_KEYS = {
    "candidate_result",
    "episode_success",
    "learned_outcome",
    "policy_result",
    "task_success",
}


class Adp009dPolicyAdmissionError(ValueError):
    """Stable fail-closed policy inventory/admission errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__("; ".join(self.errors))


def _clone(value: Mapping[str, Any], *, error: str) -> dict[str, Any]:
    try:
        cloned = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise Adp009dPolicyAdmissionError([error]) from exc
    if not isinstance(cloned, dict):
        raise Adp009dPolicyAdmissionError([error])
    return cloned


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _forbidden_outcome_paths(value: Any, *, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in _FORBIDDEN_OUTCOME_KEYS:
                found.append(path)
            found.extend(_forbidden_outcome_paths(child, prefix=path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_outcome_paths(child, prefix=f"{prefix}[{index}]"))
    return found


def validate_policy_candidate_inventory(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the outcome-blind four-candidate inventory."""

    inventory = _clone(value, error="policy_inventory_not_json_mapping")
    errors: list[str] = []
    if inventory.get("schema_version") != INVENTORY_SCHEMA_VERSION:
        errors.append("policy_inventory_schema_invalid")
    if inventory.get("program_id") != PROGRAM_ID:
        errors.append("policy_inventory_program_invalid")
    if not _RFC3339_UTC.fullmatch(str(inventory.get("audited_at") or "")):
        errors.append("policy_inventory_audited_at_invalid")
    if inventory.get("outcome_blind") is not True:
        errors.append("policy_inventory_outcome_blind_flag_missing")
    if inventory.get("learned_task_outcomes_observed") is not False:
        errors.append("policy_inventory_outcomes_already_observed")
    if inventory.get("execution_attempted") is not False:
        errors.append("policy_inventory_execution_attempted_invalid")
    if _forbidden_outcome_paths(inventory):
        errors.append("policy_inventory_caller_asserted_outcome_forbidden")

    selection = _mapping(inventory.get("candidate_selection"))
    if selection.get("required_count") != 2:
        errors.append("policy_inventory_exactly_two_required")
    if selection.get("frozen") is not False:
        errors.append("policy_inventory_selection_prematurely_frozen")
    if selection.get("selected_candidate_ids") not in (None, []):
        errors.append("policy_inventory_selected_ids_premature")
    if selection.get("selection_receipt_digest") is not None:
        errors.append("policy_inventory_selection_receipt_premature")
    if selection.get("selection_rule") != (
        "prefer_pi05_and_groot_n17_if_both_runtime_admitted_else_use_"
        "groot_n16_fallback_or_cosmos_conditional_replacement_before_outcomes"
    ):
        errors.append("policy_inventory_selection_rule_invalid")

    candidates = _rows(inventory.get("candidates"))
    by_id = {str(row.get("candidate_id") or ""): row for row in candidates}
    if len(by_id) != len(candidates):
        errors.append("policy_inventory_candidate_id_duplicate_or_missing")
    if set(by_id) != set(EXPECTED_CANDIDATES):
        errors.append("policy_inventory_candidate_set_invalid")

    for candidate_id, expected in EXPECTED_CANDIDATES.items():
        candidate = by_id.get(candidate_id)
        if candidate is None:
            continue
        source = _mapping(candidate.get("source"))
        checkpoint = _mapping(candidate.get("checkpoint"))
        for field in ("repository", "revision", "tree"):
            expected_value = expected[f"source_{field}"]
            if source.get(field) != expected_value:
                errors.append(f"policy_{candidate_id}_source_{field}_invalid")
        if not _SHA.fullmatch(str(source.get("revision") or "")):
            errors.append(f"policy_{candidate_id}_source_revision_format_invalid")
        if not _SHA.fullmatch(str(source.get("tree") or "")):
            errors.append(f"policy_{candidate_id}_source_tree_format_invalid")
        if not isinstance(source.get("license"), Mapping):
            errors.append(f"policy_{candidate_id}_source_license_missing")
        if not isinstance(source.get("submodules"), list):
            errors.append(f"policy_{candidate_id}_source_submodules_missing")
        for submodule in _rows(source.get("submodules")):
            if not str(submodule.get("path") or "") or not _SHA.fullmatch(
                str(submodule.get("revision") or "")
            ):
                errors.append(f"policy_{candidate_id}_source_submodule_invalid")

        checkpoint_expected = {
            "repository": expected["checkpoint_repository"],
            "revision": expected["checkpoint_revision"],
            "total_bytes": expected["checkpoint_total_bytes"],
            "snapshot_inventory_digest": expected["checkpoint_inventory_digest"],
        }
        for field, expected_value in checkpoint_expected.items():
            if checkpoint.get(field) != expected_value:
                errors.append(f"policy_{candidate_id}_checkpoint_{field}_invalid")
        if not _SHA256.fullmatch(
            str(checkpoint.get("snapshot_inventory_digest") or "")
        ):
            errors.append(f"policy_{candidate_id}_checkpoint_inventory_digest_invalid")
        if not isinstance(checkpoint.get("license"), Mapping):
            errors.append(f"policy_{candidate_id}_checkpoint_license_missing")
        if not isinstance(checkpoint.get("weight_files"), list):
            errors.append(f"policy_{candidate_id}_checkpoint_weight_files_missing")
        for weight in _rows(checkpoint.get("weight_files")):
            if not str(weight.get("path") or ""):
                errors.append(f"policy_{candidate_id}_checkpoint_weight_path_missing")
            if not isinstance(weight.get("size_bytes"), int):
                errors.append(f"policy_{candidate_id}_checkpoint_weight_size_invalid")
            if not _SHA256.fullmatch(str(weight.get("sha256") or "")):
                errors.append(f"policy_{candidate_id}_checkpoint_weight_sha_invalid")

        observation = _mapping(candidate.get("observation_contract"))
        action = _mapping(candidate.get("action_contract"))
        if observation.get("exact_live_renderer_required") is not True:
            errors.append(f"policy_{candidate_id}_live_renderer_requirement_missing")
        if not _strings(observation.get("camera_keys")):
            errors.append(f"policy_{candidate_id}_camera_contract_missing")
        if not isinstance(action.get("chunk_shape"), list):
            errors.append(f"policy_{candidate_id}_action_chunk_shape_missing")
        if not str(action.get("frame") or ""):
            errors.append(f"policy_{candidate_id}_action_frame_missing")
        if not str(action.get("units") or ""):
            errors.append(f"policy_{candidate_id}_action_units_missing")
        if not isinstance(action.get("safety_clamping"), Mapping):
            errors.append(f"policy_{candidate_id}_action_safety_missing")
        if candidate.get("owns_grading") is not False:
            errors.append(f"policy_{candidate_id}_grading_authority_invalid")
        if candidate.get("selected_for_frozen_evaluation") is not False:
            errors.append(f"policy_{candidate_id}_selection_flag_premature")

        admission = _mapping(candidate.get("current_admission"))
        blockers = _strings(admission.get("blockers"))
        if admission.get("status") not in {"inventory_only", "prior_smoke_only"}:
            errors.append(f"policy_{candidate_id}_inventory_admission_status_invalid")
        if not blockers:
            errors.append(f"policy_{candidate_id}_inventory_blockers_missing")
        if admission.get("adp009d_runtime_admitted") is not False:
            errors.append(f"policy_{candidate_id}_runtime_prematurely_admitted")

        if candidate.get("candidate_digest") != canonical_digest(
            candidate, digest_field="candidate_digest"
        ):
            errors.append(f"policy_{candidate_id}_candidate_digest_mismatch")

    if inventory.get("inventory_digest") != canonical_digest(
        inventory, digest_field="inventory_digest"
    ):
        errors.append("policy_inventory_digest_mismatch")
    if errors:
        raise Adp009dPolicyAdmissionError(errors)
    return inventory


def validate_policy_runtime_admission(
    value: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a byte-, renderer-, adapter-, and smoke-bound runtime admission."""

    receipt = _clone(value, error="policy_runtime_admission_not_json_mapping")
    errors: list[str] = []
    if receipt.get("schema_version") != RUNTIME_ADMISSION_SCHEMA_VERSION:
        errors.append("policy_runtime_admission_schema_invalid")
    if receipt.get("candidate_id") != candidate.get("candidate_id"):
        errors.append("policy_runtime_admission_candidate_id_mismatch")
    if receipt.get("candidate_digest") != candidate.get("candidate_digest"):
        errors.append("policy_runtime_admission_candidate_digest_mismatch")
    for field in (
        "checkpoint_materialization_digest",
        "checkpoint_tree_sha256",
        "observation_adapter_digest",
        "action_adapter_digest",
        "camera_calibration_digest",
        "immutable_smoke_input_digest",
        "immutable_smoke_output_digest",
        "runtime_environment_digest",
    ):
        if not _SHA256.fullmatch(str(receipt.get(field) or "")):
            errors.append(f"policy_runtime_admission_{field}_invalid")
    if receipt.get("checkpoint_all_files_verified") is not True:
        errors.append("policy_runtime_admission_checkpoint_files_unverified")
    if receipt.get("live_policy_frames_used") is not True:
        errors.append("policy_runtime_admission_live_frames_not_used")
    if receipt.get("action_adapter_native_probe_passed") is not True:
        errors.append("policy_runtime_admission_action_adapter_probe_missing")
    if receipt.get("immutable_smoke_passed") is not True:
        errors.append("policy_runtime_admission_smoke_missing")
    if receipt.get("task_outcomes_observed") is not False:
        errors.append("policy_runtime_admission_after_task_outcomes")
    if _strings(receipt.get("blockers")):
        errors.append("policy_runtime_admission_has_blockers")
    if receipt.get("admitted") is not True:
        errors.append("policy_runtime_admission_not_admitted")
    if receipt.get("admission_digest") != canonical_digest(
        receipt, digest_field="admission_digest"
    ):
        errors.append("policy_runtime_admission_digest_mismatch")
    if errors:
        raise Adp009dPolicyAdmissionError(errors)
    return receipt


def freeze_policy_candidate_selection(
    *,
    inventory: Mapping[str, Any],
    selected_candidate_ids: Sequence[str],
    runtime_admissions: Mapping[str, Mapping[str, Any]],
    protocol_request_digest: str,
) -> dict[str, Any]:
    """Freeze exactly two runtime-admitted candidates before task outcomes."""

    normalized = validate_policy_candidate_inventory(inventory)
    selected = [str(candidate_id) for candidate_id in selected_candidate_ids]
    errors: list[str] = []
    if len(selected) != 2 or len(set(selected)) != 2:
        errors.append("policy_selection_exactly_two_distinct_required")
    if any(candidate_id not in EXPECTED_CANDIDATES for candidate_id in selected):
        errors.append("policy_selection_unknown_candidate")
    if not _SHA256.fullmatch(str(protocol_request_digest or "")):
        errors.append("policy_selection_protocol_request_digest_invalid")
    candidates = {
        str(row["candidate_id"]): row for row in _rows(normalized.get("candidates"))
    }
    admission_digests: dict[str, str] = {}
    for candidate_id in selected:
        candidate = candidates.get(candidate_id)
        admission = runtime_admissions.get(candidate_id)
        if candidate is None or not isinstance(admission, Mapping):
            errors.append(f"policy_selection_{candidate_id}_runtime_admission_missing")
            continue
        try:
            checked = validate_policy_runtime_admission(admission, candidate=candidate)
        except Adp009dPolicyAdmissionError as exc:
            errors.extend(
                f"policy_selection_{candidate_id}_{error}" for error in exc.errors
            )
            continue
        admission_digests[candidate_id] = str(checked["admission_digest"])
    if errors:
        raise Adp009dPolicyAdmissionError(errors)

    receipt: dict[str, Any] = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "program_id": PROGRAM_ID,
        "inventory_digest": normalized["inventory_digest"],
        "protocol_request_digest": protocol_request_digest,
        "selected_candidate_ids": selected,
        "runtime_admission_digests": admission_digests,
        "selection_frozen_before_task_outcomes": True,
        "candidate_count": 2,
        "selection_digest": "",
    }
    receipt["selection_digest"] = canonical_digest(
        receipt, digest_field="selection_digest"
    )
    return receipt


__all__ = [
    "Adp009dPolicyAdmissionError",
    "INVENTORY_SCHEMA_VERSION",
    "RUNTIME_ADMISSION_SCHEMA_VERSION",
    "SELECTION_SCHEMA_VERSION",
    "freeze_policy_candidate_selection",
    "validate_policy_candidate_inventory",
    "validate_policy_runtime_admission",
]
