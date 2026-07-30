"""Deterministic replay verification for recorded supervisor runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..common import read_json, write_json
from ..decision_evidence_contracts import (
    DecisionEnvelope,
    DecisionEvidenceRequest,
    EvidenceMethodProfile,
    EvidencePlan,
    MaintainedSiteTaskTestbed,
    NormalizedEvidenceResult,
    QualificationRecord,
    canonical_digest,
)
from .contracts import (
    AgentInvocationManifest,
    AuthorityEnvelope,
    CapabilityResult,
    SupervisorRun,
    TerminalSupervisorReport,
)
from .ledger import AppendOnlyEventLedger


REPLAY_REPORT_SCHEMA_VERSION = "task_evaluation_supervisor_replay_report.v1"


class SupervisorReplayError(ValueError):
    """Raised when recorded supervisor evidence cannot be replayed exactly."""


def _validate_kernel_input(name: str, value: Mapping[str, Any]) -> Mapping[str, Any]:
    validators = {
        "decision_request": DecisionEvidenceRequest,
        "site_task_testbed": MaintainedSiteTaskTestbed,
        "evidence_plan": EvidencePlan,
        "decision_envelope": DecisionEnvelope,
    }
    if name.startswith("method_profile_"):
        return EvidenceMethodProfile.from_mapping(value).to_mapping()
    if name.startswith("qualification_"):
        return QualificationRecord.from_mapping(value).to_mapping()
    if name.startswith("evidence_result_"):
        return NormalizedEvidenceResult.from_mapping(value).to_mapping()
    validator: Any = validators.get(name)
    if validator is not None:
        return validator.from_mapping(value).to_mapping()
    if name == "capture_build":
        expected = value.get("capture_build_digest")
        actual = canonical_digest(value, digest_field="capture_build_digest")
        if expected != actual:
            raise SupervisorReplayError("capture_build_digest_mismatch")
        return dict(value)
    raise SupervisorReplayError(f"kernel_input_name_unsupported:{name}")


def replay_supervisor_run(output_dir: str | Path) -> dict[str, Any]:
    """Revalidate the ledger, manifests, and kernel decision without a model call."""

    root = Path(output_dir).expanduser().resolve()
    run = SupervisorRun.from_mapping(read_json(root / "task_evaluation_supervisor_run.json"))
    report = TerminalSupervisorReport.from_mapping(
        read_json(root / "terminal_supervisor_report.json")
    )
    run_value = run.to_mapping()
    report_value = report.to_mapping()
    authority = AuthorityEnvelope.from_mapping(read_json(root / "authority_envelope.json"))
    authority_value = authority.to_mapping()
    if report_value["run_id"] != run_value["run_id"]:
        raise SupervisorReplayError("run_report_identity_mismatch")

    events = AppendOnlyEventLedger(root / "supervisor_events.jsonl").read()
    if not events or events[-1].digest != report_value["last_event_digest"]:
        raise SupervisorReplayError("event_ledger_terminal_digest_mismatch")
    if len(events) != report_value["event_count"]:
        raise SupervisorReplayError("event_ledger_count_mismatch")

    customer_report_path = (
        root / str(report_value.get("customer_report_path") or "")
    ).resolve()
    if root not in customer_report_path.parents:
        raise SupervisorReplayError("customer_report_path_escape")
    customer_report = read_json(customer_report_path)
    customer_report_digest = canonical_digest(
        customer_report,
        digest_field="customer_report_digest",
    )
    if (
        customer_report.get("customer_report_digest") != customer_report_digest
        or report_value.get("customer_report_digest") != customer_report_digest
        or customer_report.get("proof_state_mutated_by_report") is not False
    ):
        raise SupervisorReplayError("customer_report_contract_mismatch")

    capability_digests: list[str] = []
    for row in report_value.get("capability_results") or []:
        path = (root / str(row["artifact_path"])).resolve()
        if root not in path.parents:
            raise SupervisorReplayError("capability_artifact_path_escape")
        capability = CapabilityResult.from_mapping(read_json(path))
        if capability.digest != row["digest"]:
            raise SupervisorReplayError("capability_artifact_digest_mismatch")
        capability_digests.append(capability.digest)

    invocation_digests: list[str] = []
    tool_observation_digests: list[str] = []
    generated_artifact_digests: list[str] = []
    replayed_tool_cost_usd = 0.0
    for row in report_value.get("invocation_manifests") or []:
        path = (root / str(row["artifact_path"])).resolve()
        if root not in path.parents:
            raise SupervisorReplayError("invocation_artifact_path_escape")
        invocation = AgentInvocationManifest.from_mapping(read_json(path))
        if invocation.digest != row["digest"]:
            raise SupervisorReplayError("invocation_artifact_digest_mismatch")
        invocation_digests.append(invocation.digest)
        for observation_ref in invocation.to_mapping().get("tool_observation_references") or []:
            if not isinstance(observation_ref, Mapping):
                raise SupervisorReplayError("tool_observation_reference_invalid")
            observation_path = (root / str(observation_ref.get("artifact_path") or "")).resolve()
            if root not in observation_path.parents:
                raise SupervisorReplayError("tool_observation_path_escape")
            observation = read_json(observation_path)
            digest = canonical_digest(observation, digest_field="observation_digest")
            observation_cost = float(observation.get("cost_usd") or 0.0)
            if (
                observation.get("observation_digest") != digest
                or observation_ref.get("digest") != digest
                or observation.get("proof_effect") != "none"
                or observation_cost < 0
            ):
                raise SupervisorReplayError("tool_observation_contract_mismatch")
            if (
                observation.get("runtime_identity")
                == "blueprint_local_deterministic_non_spend"
                and observation_cost != 0
            ):
                raise SupervisorReplayError("non_spend_tool_reported_cost")
            if (
                observation.get("mutability") == "external_side_effect"
                and authority_value.get("mode") != "execute_preauthorized"
            ):
                raise SupervisorReplayError("preauthorized_tool_wrong_mode")
            replayed_tool_cost_usd += observation_cost
            tool_observation_digests.append(digest)
            for generated_ref in observation.get("produced_artifact_references") or []:
                if not isinstance(generated_ref, Mapping):
                    raise SupervisorReplayError("generated_artifact_reference_invalid")
                generated_path = (
                    root / str(generated_ref.get("artifact_path") or "")
                ).resolve()
                if root not in generated_path.parents:
                    raise SupervisorReplayError("generated_artifact_path_escape")
                generated_value = read_json(generated_path)
                artifact_type = str(generated_ref.get("artifact_type") or "")
                digest_fields = {
                    "evidence_plan.v1": "plan_digest",
                    "targeted_recapture_request.v1": "targeted_recapture_request_digest",
                    "task_evaluation_clarification_request.v1": (
                        "clarification_request_digest"
                    ),
                    "task_evaluation_authorization_request.v1": (
                        "authorization_request_digest"
                    ),
                    "task_evaluation_scenario_proposal_set.v1": (
                        "scenario_proposal_set_digest"
                    ),
                    "task_evaluation_recovery_result.v1": "recovery_result_digest",
                }
                generated_digest = canonical_digest(
                    generated_value,
                    digest_field=digest_fields.get(artifact_type),
                )
                if generated_ref.get("artifact_digest") != generated_digest:
                    raise SupervisorReplayError("generated_artifact_digest_mismatch")
                generated_artifact_digests.append(generated_digest)
    if replayed_tool_cost_usd > float(authority_value.get("max_cost_usd") or 0.0):
        raise SupervisorReplayError("replayed_tool_cost_exceeds_authority")

    inputs_manifest = read_json(root / "kernel_inputs_manifest.json")
    expected_manifest_digest = inputs_manifest.get("kernel_inputs_manifest_digest")
    actual_manifest_digest = canonical_digest(
        inputs_manifest, digest_field="kernel_inputs_manifest_digest"
    )
    if expected_manifest_digest != actual_manifest_digest:
        raise SupervisorReplayError("kernel_inputs_manifest_digest_mismatch")
    kernel_inputs: dict[str, Mapping[str, Any]] = {}
    for row in inputs_manifest.get("artifacts") or []:
        if not isinstance(row, Mapping):
            raise SupervisorReplayError("kernel_inputs_manifest_row_invalid")
        name = str(row.get("name") or "")
        path = (root / str(row.get("artifact_path") or "")).resolve()
        if root not in path.parents:
            raise SupervisorReplayError("kernel_input_artifact_path_escape")
        value = read_json(path)
        if canonical_digest(value) != row.get("digest"):
            raise SupervisorReplayError(f"kernel_input_artifact_digest_mismatch:{name}")
        kernel_inputs[name] = _validate_kernel_input(name, value)

    decision = kernel_inputs.get("decision_envelope")
    replayed_decision: dict[str, Any] | None = None
    if decision is not None:
        validated = DecisionEnvelope.from_mapping(decision).to_mapping()
        replayed_decision = {
            "decision_envelope_digest": validated["decision_envelope_digest"],
            "overall_outcome": validated["overall_outcome"],
            "claim_ceiling": validated["claim_ceiling"],
        }
    replay_value: dict[str, Any] = {
        "schema_version": REPLAY_REPORT_SCHEMA_VERSION,
        "run_id": run_value["run_id"],
        "run_digest": run.digest,
        "terminal_report_digest": report.digest,
        "status": "replay_verified",
        "model_invoked_during_replay": False,
        "agent_prose_reproduced": False,
        "kernel_inputs_revalidated": True,
        "capability_result_digests": capability_digests,
        "invocation_manifest_digests": invocation_digests,
        "tool_observation_digests": tool_observation_digests,
        "generated_artifact_digests": generated_artifact_digests,
        "customer_report_digest": customer_report_digest,
        "replayed_tool_cost_usd": replayed_tool_cost_usd,
        "event_count": len(events),
        "last_event_digest": events[-1].digest,
        "replayed_deterministic_decision": replayed_decision,
        "proof_result_reproduced": replayed_decision is not None,
    }
    replay_value["replay_report_digest"] = canonical_digest(
        replay_value, digest_field="replay_report_digest"
    )
    write_json(root / "supervisor_replay_report.json", replay_value)
    return replay_value


__all__ = [
    "REPLAY_REPORT_SCHEMA_VERSION",
    "SupervisorReplayError",
    "replay_supervisor_run",
]
