"""Replayable capture-reconstruction terminal and customer report compiler."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest
from .reconstruction_geometry_contracts import (
    ReconstructionGeometryContractError,
    build_isaac_asset_verification_result,
)


RECONSTRUCTION_REPORT_REQUEST_SCHEMA_VERSION = "reconstruction_terminal_report_request.v1"
RECONSTRUCTION_TERMINAL_REPORT_SCHEMA_VERSION = "reconstruction_terminal_report.v1"

_DECISIONS = {"usable", "partially_usable", "rejected", "abstention"}
_CLAIM_CEILING_KEYS = (
    "decoded_observation_availability",
    "calibrated_camera_trajectory",
    "appearance_reconstruction",
    "metric_scale",
    "metric_reference_geometry",
    "collision_geometry",
    "physics_readiness",
    "isaac_load_render_compatibility",
    "simulator_task_evidence",
    "physical_task_success",
    "deployment_readiness",
)


class ReconstructionTerminalReportError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ReconstructionTerminalReportError(["reconstruction_report_not_json"]) from exc


def _is_digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def build_reconstruction_terminal_report_request(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    request = _clone(dict(value))
    errors: list[str] = []
    if request.get("schema_version") != RECONSTRUCTION_REPORT_REQUEST_SCHEMA_VERSION:
        errors.append("reconstruction_report_request_schema_invalid")
    for key in (
        "stable_run_identity",
        "original_capture_location",
        "validated_capture_profile",
        "original_customer_request",
        "source_commit_sha",
        "timestamp",
    ):
        if not str(request.get(key) or "").strip():
            errors.append(f"reconstruction_report_request_{key}_missing")
    commit = str(request.get("source_commit_sha") or "")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        errors.append("reconstruction_report_request_source_commit_invalid")
    for key in ("source_capture_digest", "frozen_split_digest"):
        if not _is_digest(request.get(key)):
            errors.append(f"reconstruction_report_request_{key}_invalid")
    if not _is_digest(request.get("implementation_digest")):
        errors.append("reconstruction_report_request_implementation_digest_invalid")
    for key in ("input_digests", "recorded_output_digests"):
        digests = request.get(key)
        if not isinstance(digests, list) or any(not _is_digest(digest) for digest in digests):
            errors.append(f"reconstruction_report_request_{key}_invalid")
    if request.get("decision") not in _DECISIONS:
        errors.append("reconstruction_report_request_decision_invalid")
    if not isinstance(request.get("rights_and_permitted_use"), Mapping):
        errors.append("reconstruction_report_request_rights_invalid")
    elif request["rights_and_permitted_use"].get("status") not in {
        "cleared",
        "blocked",
        "unknown",
    }:
        errors.append("reconstruction_report_request_rights_status_invalid")
    for key in (
        "selected_frames",
        "rejected_frames",
        "pose_methods_attempted",
        "registered_observations",
        "rejected_observations",
        "reconstruction_methods_attempted",
        "failed_methods",
        "skipped_methods",
        "recovered_methods",
        "fixed_camera_render_references",
        "agent_proposals_and_actions",
        "deterministic_validations",
        "what_could_change_result",
        "what_blueprint_cannot_claim",
        "warnings",
        "blockers",
    ):
        if not isinstance(request.get(key), list):
            errors.append(f"reconstruction_report_request_{key}_invalid")
    for key in (
        "calibration_and_coordinate_status",
        "scale_validation",
        "appearance_asset",
        "metric_reference_asset",
        "collision_candidate",
        "independent_visual_metrics",
        "independent_geometric_metrics",
        "collider_qualification",
        "nurec_openusd_package",
        "isaac_verification",
        "physics_collision_verification",
        "provider_execution",
        "runtime_and_spend",
        "teardown_and_provider_zero",
        "authority_used",
        "camera_calibration_binding",
        "coordinate_frame_declaration",
        "units_and_metric_scale_status",
        "provider_runtime_identity",
    ):
        if not isinstance(request.get(key), Mapping):
            errors.append(f"reconstruction_report_request_{key}_invalid")
    ceilings = request.get("evidence_ceilings")
    if not isinstance(ceilings, Mapping) or set(ceilings) != set(_CLAIM_CEILING_KEYS) or any(
        not isinstance(ceilings.get(key), bool) for key in _CLAIM_CEILING_KEYS
    ):
        errors.append("reconstruction_report_request_evidence_ceilings_invalid")
    if isinstance(ceilings, Mapping):
        if ceilings.get("deployment_readiness") is True and ceilings.get(
            "physical_task_success"
        ) is not True:
            errors.append("reconstruction_report_request_deployment_without_physical_proof")
        if ceilings.get("physics_readiness") is True and ceilings.get(
            "collision_geometry"
        ) is not True:
            errors.append("reconstruction_report_request_physics_without_collision_proof")
        if ceilings.get("collision_geometry") is True and ceilings.get(
            "metric_reference_geometry"
        ) is not True:
            errors.append("reconstruction_report_request_collision_without_metric_geometry")
    isaac = request.get("isaac_verification")
    if isinstance(isaac, Mapping):
        if isaac.get("schema_version") == "isaac_asset_verification_result.v1":
            try:
                verified_isaac = build_isaac_asset_verification_result(isaac)
            except ReconstructionGeometryContractError as exc:
                errors.extend(
                    f"reconstruction_report_request_isaac_invalid:{code}"
                    for code in exc.codes
                )
            else:
                if not isinstance(ceilings, Mapping) or ceilings.get(
                    "isaac_load_render_compatibility"
                ) is not True:
                    errors.append(
                        "reconstruction_report_request_isaac_result_without_compatibility_ceiling"
                    )
                if verified_isaac["isaac_verification_result_digest"] not in request.get(
                    "recorded_output_digests", []
                ):
                    errors.append(
                        "reconstruction_report_request_isaac_result_digest_unrecorded"
                    )
                if request.get("fixed_camera_render_references") != verified_isaac.get(
                    "fixed_camera_render_references"
                ):
                    errors.append(
                        "reconstruction_report_request_isaac_render_references_mismatch"
                    )
                if request.get("physics_collision_verification") != verified_isaac.get(
                    "physics_probe"
                ):
                    errors.append(
                        "reconstruction_report_request_isaac_physics_evidence_mismatch"
                    )
        elif isaac.get("status") not in {"not_executed", "blocked", "failed"}:
            errors.append("reconstruction_report_request_isaac_status_invalid")
        elif isinstance(ceilings, Mapping) and ceilings.get(
            "isaac_load_render_compatibility"
        ) is True:
            errors.append(
                "reconstruction_report_request_isaac_compatibility_without_typed_result"
            )
    runtime = request.get("runtime_and_spend")
    if isinstance(runtime, Mapping):
        for key in ("total_runtime_seconds", "total_spend_usd"):
            number = runtime.get(key)
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(float(number))
                or float(number) < 0
            ):
                errors.append(f"reconstruction_report_request_{key}_invalid")
    for index, action in enumerate(request.get("agent_proposals_and_actions") or []):
        if not isinstance(action, Mapping) or action.get("proof_effect") != "none":
            errors.append(f"reconstruction_report_request_agent_action_invalid:{index}")
    for index, failure in enumerate(request.get("failed_methods") or []):
        if not isinstance(failure, Mapping) or failure.get("failed_evidence_preserved") is not True:
            errors.append(f"reconstruction_report_request_failed_method_suppressed:{index}")
    if request.get("decision") in {"rejected", "abstention", "partially_usable"} and not request.get(
        "blockers"
    ):
        errors.append("reconstruction_report_request_nonusable_decision_without_blocker")
    if request.get("decision") in {"usable", "partially_usable"} and request.get(
        "rights_and_permitted_use", {}
    ).get("status") != "cleared":
        errors.append("reconstruction_report_request_usable_without_rights")
    if request.get("decision") == "usable" and (
        request.get("blockers")
        or not isinstance(ceilings, Mapping)
        or ceilings.get("appearance_reconstruction") is not True
    ):
        errors.append("reconstruction_report_request_usable_status_inconsistent")
    container_digests = request.get("container_image_digests")
    if not isinstance(container_digests, list) or any(
        not _is_digest(digest) for digest in container_digests
    ):
        errors.append("reconstruction_report_request_container_digests_invalid")
    supplied_digest = request.pop("reconstruction_terminal_report_request_digest", None)
    request["reconstruction_terminal_report_request_digest"] = canonical_digest(
        request, digest_field="reconstruction_terminal_report_request_digest"
    )
    if supplied_digest is not None and supplied_digest != request[
        "reconstruction_terminal_report_request_digest"
    ]:
        errors.append("reconstruction_report_request_digest_mismatch")
    if errors:
        raise ReconstructionTerminalReportError(errors)
    return request


def _customer_summary(request: Mapping[str, Any]) -> str:
    decision = request["decision"]
    if decision == "usable":
        return (
            "The reconstruction passed the recorded deterministic gates for its stated "
            "evidence ceilings. It is usable only for the claims marked true below."
        )
    if decision == "partially_usable":
        return (
            "The reconstruction is usable for a limited subset of claims, but one or more "
            "required evidence gates remain unresolved."
        )
    if decision == "rejected":
        return (
            "The reconstruction was rejected because recorded deterministic checks failed. "
            "Failed attempts remain in the audit record."
        )
    return (
        "Blueprint abstained because the available evidence cannot support the requested "
        "reconstruction claim. The blockers below explain what evidence is missing."
    )


def generate_reconstruction_terminal_report(value: Mapping[str, Any]) -> dict[str, Any]:
    request = build_reconstruction_terminal_report_request(value)
    report = {
        "schema_version": RECONSTRUCTION_TERMINAL_REPORT_SCHEMA_VERSION,
        "stable_run_identity": request["stable_run_identity"],
        "source_request_digest": request["reconstruction_terminal_report_request_digest"],
        "producing_method": "deterministic_reconstruction_terminal_report_compiler",
        "implementation_version": request["implementation_digest"],
        "deterministic_configuration_digest": request[
            "reconstruction_terminal_report_request_digest"
        ],
        "input_digests": request["input_digests"],
        "recorded_output_digests": request["recorded_output_digests"],
        "original_capture_location": request["original_capture_location"],
        "source_capture_digest": request["source_capture_digest"],
        "validated_capture_profile": request["validated_capture_profile"],
        "original_customer_request": request["original_customer_request"],
        "rights_and_permitted_use": request["rights_and_permitted_use"],
        "selected_frames": request["selected_frames"],
        "rejected_frames": request["rejected_frames"],
        "frozen_split_digest": request["frozen_split_digest"],
        "calibration_and_coordinate_status": request[
            "calibration_and_coordinate_status"
        ],
        "camera_calibration_binding": request["camera_calibration_binding"],
        "coordinate_frame_declaration": request["coordinate_frame_declaration"],
        "units_and_metric_scale_status": request["units_and_metric_scale_status"],
        "pose_methods_attempted": request["pose_methods_attempted"],
        "registered_observations": request["registered_observations"],
        "rejected_observations": request["rejected_observations"],
        "scale_validation": request["scale_validation"],
        "reconstruction_methods_attempted": request[
            "reconstruction_methods_attempted"
        ],
        "failed_methods": request["failed_methods"],
        "skipped_methods": request["skipped_methods"],
        "recovered_methods": request["recovered_methods"],
        "appearance_asset": request["appearance_asset"],
        "metric_reference_asset": request["metric_reference_asset"],
        "collision_candidate": request["collision_candidate"],
        "independent_visual_metrics": request["independent_visual_metrics"],
        "independent_geometric_metrics": request["independent_geometric_metrics"],
        "collider_qualification": request["collider_qualification"],
        "nurec_openusd_package": request["nurec_openusd_package"],
        "isaac_verification": request["isaac_verification"],
        "fixed_camera_render_references": request["fixed_camera_render_references"],
        "physics_collision_verification": request["physics_collision_verification"],
        "provider_execution": request["provider_execution"],
        "provider_runtime_identity": request["provider_runtime_identity"],
        "source_commit_sha": request["source_commit_sha"],
        "container_image_digests": list(request.get("container_image_digests") or []),
        "runtime_and_spend": request["runtime_and_spend"],
        "agent_proposals_and_actions": request["agent_proposals_and_actions"],
        "agent_output_authoritative": False,
        "deterministic_validations": request["deterministic_validations"],
        "decision": request["decision"],
        "customer_summary": _customer_summary(request),
        "evidence_ceilings": request["evidence_ceilings"],
        "what_could_change_result": request["what_could_change_result"],
        "what_blueprint_cannot_claim": request["what_blueprint_cannot_claim"],
        "warnings": request["warnings"],
        "blockers": request["blockers"],
        "teardown_and_provider_zero": request["teardown_and_provider_zero"],
        "authority_used": request["authority_used"],
        "proof_effect": "deterministic_reconstruction_explanation_only",
        "claim_ceiling": request["evidence_ceilings"],
        "proof_state_mutated_by_report": False,
        "parent_artifact_or_event": {
            "request_digest": request["reconstruction_terminal_report_request_digest"],
            "source_capture_digest": request["source_capture_digest"],
        },
        "timestamp": request["timestamp"],
    }
    report["reconstruction_terminal_report_digest"] = canonical_digest(
        report, digest_field="reconstruction_terminal_report_digest"
    )
    return report


__all__ = [
    "RECONSTRUCTION_REPORT_REQUEST_SCHEMA_VERSION",
    "RECONSTRUCTION_TERMINAL_REPORT_SCHEMA_VERSION",
    "ReconstructionTerminalReportError",
    "build_reconstruction_terminal_report_request",
    "generate_reconstruction_terminal_report",
]
