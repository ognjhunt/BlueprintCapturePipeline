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
    SupervisorContractError,
    SupervisorRun,
    TerminalSupervisorReport,
    ToolDescriptor,
)
from .capture_ingress import CaptureBuildIngressError, validate_capture_build_ingress
from .ledger import AppendOnlyEventLedger
from .inference_reservations import InferenceReservationAudit
from .phase2_artifacts import (
    Phase2ArtifactError,
    recapture_reinspection as build_recapture_reinspection,
    validate_clarification_receipt,
    validate_clarification_request,
    validate_authorization_receipt,
    validate_authorization_request,
    validate_customer_report,
    validate_recapture_reinspection,
    validate_targeted_recapture_receipt,
    validate_targeted_recapture_request,
)
from .tools import (
    TOOL_REGISTRY_SCHEMA_VERSION,
    ToolRegistry,
    validate_tool_observation_binding,
)


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
        try:
            return validate_capture_build_ingress(value)
        except CaptureBuildIngressError as exc:
            raise SupervisorReplayError("capture_build_ingress_invalid") from exc
    if name == "clarification_request":
        try:
            return validate_clarification_request(value)
        except Phase2ArtifactError as exc:
            raise SupervisorReplayError("clarification_request_invalid") from exc
    if name == "clarification_receipt":
        try:
            return validate_clarification_receipt(value)
        except Phase2ArtifactError as exc:
            raise SupervisorReplayError("clarification_receipt_invalid") from exc
    if name == "authorization_request":
        try:
            return validate_authorization_request(value)
        except Phase2ArtifactError as exc:
            raise SupervisorReplayError("authorization_request_invalid") from exc
    if name == "authorization_receipt":
        try:
            return validate_authorization_receipt(value)
        except Phase2ArtifactError as exc:
            raise SupervisorReplayError("authorization_receipt_invalid") from exc
    if name == "targeted_recapture_request":
        try:
            return validate_targeted_recapture_request(value)
        except Phase2ArtifactError as exc:
            raise SupervisorReplayError("targeted_recapture_request_invalid") from exc
    if name == "targeted_recapture_receipt":
        try:
            return validate_targeted_recapture_receipt(value)
        except Phase2ArtifactError as exc:
            raise SupervisorReplayError("targeted_recapture_receipt_invalid") from exc
    if name == "recapture_reinspection":
        try:
            return validate_recapture_reinspection(value)
        except Phase2ArtifactError as exc:
            raise SupervisorReplayError("recapture_reinspection_invalid") from exc
    raise SupervisorReplayError(f"kernel_input_name_unsupported:{name}")


def replay_supervisor_run(
    output_dir: str | Path,
    *,
    persist_report: bool = True,
) -> dict[str, Any]:
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
    if run_value["authority_digest"] != authority.digest:
        raise SupervisorReplayError("run_authority_digest_mismatch")

    tool_manifest = dict(read_json(root / "tool_registry_manifest.json"))
    tool_manifest_digest = canonical_digest(
        tool_manifest,
        digest_field="tool_registry_digest",
    )
    tool_rows = tool_manifest.get("tools")
    if (
        tool_manifest.get("schema_version") != TOOL_REGISTRY_SCHEMA_VERSION
        or tool_manifest.get("tool_registry_digest") != tool_manifest_digest
        or run_value["tool_registry_digest"] != tool_manifest_digest
        or report_value["tool_registry_digest"] != tool_manifest_digest
        or not isinstance(tool_rows, list)
        or any(not isinstance(row, Mapping) for row in tool_rows)
    ):
        raise SupervisorReplayError("tool_registry_manifest_mismatch")
    try:
        tool_registry = ToolRegistry.from_descriptors(
            [ToolDescriptor.from_mapping(row) for row in tool_rows]
        )
    except (SupervisorContractError, ValueError) as exc:
        raise SupervisorReplayError("tool_registry_manifest_mismatch") from exc
    if tool_registry.manifest() != tool_manifest:
        raise SupervisorReplayError("tool_registry_manifest_mismatch")

    events = AppendOnlyEventLedger(root / "supervisor_events.jsonl").read()
    if not events or events[-1].digest != report_value["last_event_digest"]:
        raise SupervisorReplayError("event_ledger_terminal_digest_mismatch")
    if len(events) != report_value["event_count"]:
        raise SupervisorReplayError("event_ledger_count_mismatch")

    inference_spend = dict(report_value.get("inference_spend") or {})
    reservation_path = (
        root / str(inference_spend.get("reservation_manifest_path") or "")
    ).resolve()
    if root not in reservation_path.parents or not reservation_path.is_file():
        raise SupervisorReplayError("inference_reservation_manifest_missing")
    persisted_reservation_manifest = dict(read_json(reservation_path))
    replayed_reservation_manifest = InferenceReservationAudit(
        run_root=root,
        run_id=str(run_value["run_id"]),
    ).manifest()
    if persisted_reservation_manifest != replayed_reservation_manifest:
        raise SupervisorReplayError("inference_reservation_manifest_mismatch")
    if (
        inference_spend.get("reservation_manifest_digest")
        != replayed_reservation_manifest["inference_reservation_manifest_digest"]
        or int(inference_spend.get("reservation_count") or 0)
        != replayed_reservation_manifest["reservation_count"]
        or int(inference_spend.get("in_flight_unknown_count") or 0)
        != replayed_reservation_manifest["in_flight_unknown_count"]
        or float(inference_spend.get("reserved_max_cost_usd") or 0.0)
        != float(replayed_reservation_manifest["reserved_max_cost_usd"])
    ):
        raise SupervisorReplayError("inference_reservation_report_mismatch")

    customer_report_path = (root / str(report_value.get("customer_report_path") or "")).resolve()
    if root not in customer_report_path.parents:
        raise SupervisorReplayError("customer_report_path_escape")
    customer_report = read_json(customer_report_path)
    try:
        customer_report = validate_customer_report(customer_report)
    except Phase2ArtifactError as exc:
        raise SupervisorReplayError("customer_report_contract_mismatch") from exc
    customer_report_digest = customer_report["customer_report_digest"]
    if (
        customer_report.get("customer_report_digest") != customer_report_digest
        or report_value.get("customer_report_digest") != customer_report_digest
        or customer_report.get("proof_state_mutated_by_report") is not False
    ):
        raise SupervisorReplayError("customer_report_contract_mismatch")

    capability_digests: list[str] = []
    capability_digest_by_kind: dict[str, str] = {}
    structured_observations_by_capability: dict[str, list[dict[str, Any]]] = {}
    for row in report_value.get("capability_results") or []:
        path = (root / str(row["artifact_path"])).resolve()
        if root not in path.parents:
            raise SupervisorReplayError("capability_artifact_path_escape")
        capability = CapabilityResult.from_mapping(read_json(path))
        if capability.digest != row["digest"]:
            raise SupervisorReplayError("capability_artifact_digest_mismatch")
        capability_digests.append(capability.digest)
        capability_value = capability.to_mapping()
        capability_id = str(capability_value["capability"])
        capability_digest_by_kind[capability_id] = capability.digest
        embedded_observations = capability_value.get("structured_observations") or []
        if not isinstance(embedded_observations, list) or any(
            not isinstance(item, Mapping) for item in embedded_observations
        ):
            raise SupervisorReplayError("capability_structured_observations_invalid")
        structured_observations_by_capability[capability_id] = [
            dict(item) for item in embedded_observations
        ]

    manager_decision_digests: list[str] = []
    manager_invocation_digests: list[str] = []
    manager_refusal_digests: list[str] = []
    observed_capability_digests: list[str] = []
    manager_rows = sorted(
        [
            dict(row)
            for row in report_value.get("manager_decisions") or []
            if isinstance(row, Mapping)
        ],
        key=lambda row: int(row.get("step_index", -1)),
    )
    refusal_rows = [
        dict(row) for row in report_value.get("manager_refusals") or [] if isinstance(row, Mapping)
    ]
    for row in refusal_rows:
        path = (root / str(row.get("artifact_path") or "")).resolve()
        if root not in path.parents:
            raise SupervisorReplayError("manager_refusal_path_escape")
        refusal = dict(read_json(path))
        digest = canonical_digest(
            refusal,
            digest_field="supervisor_manager_refusal_digest",
        )
        if (
            refusal.get("schema_version") != "task_evaluation_supervisor_manager_refusal.v1"
            or refusal.get("supervisor_manager_refusal_digest") != digest
            or row.get("digest") != digest
            or refusal.get("proof_effect") != "none"
            or refusal.get("raw_error_message_recorded") is not False
        ):
            raise SupervisorReplayError("manager_refusal_contract_mismatch")
        manager_refusal_digests.append(digest)
    if len(manager_refusal_digests) > 1 or (
        refusal_rows and int(refusal_rows[0].get("step_index", -1)) != len(manager_rows)
    ):
        raise SupervisorReplayError("manager_refusal_sequence_invalid")
    if (
        not manager_rows
        and not manager_refusal_digests
        and run_value.get("manager_agent_required") is True
    ):
        raise SupervisorReplayError("manager_decision_artifacts_missing")
    terminal_manager_rows = 0
    for expected_step, row in enumerate(manager_rows):
        if int(row.get("step_index", -1)) != expected_step:
            raise SupervisorReplayError("manager_decision_step_invalid")
        path = (root / str(row.get("artifact_path") or "")).resolve()
        if root not in path.parents:
            raise SupervisorReplayError("manager_decision_path_escape")
        manager_decision_value = dict(read_json(path))
        digest = canonical_digest(
            manager_decision_value,
            digest_field="supervisor_manager_decision_digest",
        )
        if (
            manager_decision_value.get("supervisor_manager_decision_digest") != digest
            or row.get("digest") != digest
            or manager_decision_value.get("observed_capability_result_digests")
            != sorted(observed_capability_digests)
            or manager_decision_value.get("proof_effect") != "none"
            or manager_decision_value.get("manager_controls_sequencing_only") is not True
        ):
            raise SupervisorReplayError("manager_decision_contract_mismatch")
        if manager_decision_value.get("status") == "continue":
            capability_id = str(manager_decision_value.get("next_capability") or "")
            capability_digest = capability_digest_by_kind.get(capability_id)
            if capability_digest is None or capability_digest in observed_capability_digests:
                raise SupervisorReplayError("manager_capability_sequence_invalid")
            observed_capability_digests.append(capability_digest)
        elif manager_decision_value.get("status") == "terminal":
            terminal_manager_rows += 1
            if manager_decision_value.get("next_capability") is not None:
                raise SupervisorReplayError("manager_terminal_decision_invalid")
        else:
            raise SupervisorReplayError("manager_decision_status_invalid")
        manager_decision_digests.append(digest)
    if manager_rows and (
        terminal_manager_rows + len(manager_refusal_digests) != 1
        or sorted(observed_capability_digests) != sorted(capability_digests)
    ):
        raise SupervisorReplayError("manager_terminal_sequence_incomplete")

    invocation_rows = sorted(
        [
            dict(row)
            for row in report_value.get("manager_invocations") or []
            if isinstance(row, Mapping)
        ],
        key=lambda row: int(row.get("step_index", -1)),
    )
    if len(invocation_rows) != len(manager_rows):
        raise SupervisorReplayError("manager_invocation_count_mismatch")
    for expected_step, row in enumerate(invocation_rows):
        path = (root / str(row.get("artifact_path") or "")).resolve()
        if root not in path.parents:
            raise SupervisorReplayError("manager_invocation_path_escape")
        manager_invocation_value = dict(read_json(path))
        digest = canonical_digest(
            manager_invocation_value,
            digest_field="manager_invocation_digest",
        )
        if (
            int(manager_invocation_value.get("step_index", -1)) != expected_step
            or manager_invocation_value.get("manager_invocation_digest") != digest
            or row.get("digest") != digest
            or manager_invocation_value.get("structured_output_digest")
            != manager_decision_digests[expected_step]
            or manager_invocation_value.get("proof_effect") != "none"
            or manager_invocation_value.get("action_taken") != "specialist_sequence_selected"
        ):
            raise SupervisorReplayError("manager_invocation_contract_mismatch")
        manager_invocation_digests.append(digest)

    manager_event_digests = [
        str(event.to_mapping().get("payload_digest") or "")
        for event in events
        if str(event.to_mapping().get("event_type") or "").startswith(
            "supervisor_manager_decision_recorded:"
        )
    ]
    if manager_event_digests != manager_decision_digests:
        raise SupervisorReplayError("manager_event_sequence_mismatch")
    manager_refusal_event_digests = [
        str(event.to_mapping().get("payload_digest") or "")
        for event in events
        if str(event.to_mapping().get("event_type") or "").startswith("supervisor_manager_refused:")
    ]
    if manager_refusal_event_digests != manager_refusal_digests:
        raise SupervisorReplayError("manager_refusal_event_sequence_mismatch")

    invocation_digests: list[str] = []
    tool_observation_digests: list[str] = []
    referenced_tool_observation_paths: set[Path] = set()
    generated_artifact_digests: list[str] = []
    replayed_tool_cost_usd = 0.0
    for row in report_value.get("invocation_manifests") or []:
        path = (root / str(row["artifact_path"])).resolve()
        if root not in path.parents:
            raise SupervisorReplayError("invocation_artifact_path_escape")
        specialist_invocation = AgentInvocationManifest.from_mapping(read_json(path))
        specialist_invocation_value = specialist_invocation.to_mapping()
        if (
            specialist_invocation.digest != row["digest"]
            or specialist_invocation_value["run_id"] != run_value["run_id"]
            or specialist_invocation_value["authority_digest"] != authority.digest
            or specialist_invocation_value["tool_registry_digest"] != tool_registry.digest
        ):
            raise SupervisorReplayError("invocation_artifact_digest_mismatch")
        invocation_digests.append(specialist_invocation.digest)
        invocation_observation_summaries: list[dict[str, Any]] = []
        for observation_ref in specialist_invocation_value.get("tool_observation_references") or []:
            if not isinstance(observation_ref, Mapping):
                raise SupervisorReplayError("tool_observation_reference_invalid")
            observation_path = (root / str(observation_ref.get("artifact_path") or "")).resolve()
            if root not in observation_path.parents:
                raise SupervisorReplayError("tool_observation_path_escape")
            if observation_path in referenced_tool_observation_paths:
                raise SupervisorReplayError("tool_observation_reference_duplicated")
            referenced_tool_observation_paths.add(observation_path)
            try:
                observation = validate_tool_observation_binding(
                    read_json(observation_path),
                    run_id=str(run_value["run_id"]),
                    capability=str(specialist_invocation_value["capability"]),
                    registry=tool_registry,
                    authority=authority_value,
                )
            except (SupervisorContractError, ValueError) as exc:
                raise SupervisorReplayError("tool_observation_contract_mismatch") from exc
            digest = str(observation["observation_digest"])
            observation_cost = float(observation.get("cost_usd") or 0.0)
            if observation_ref.get("digest") != digest or observation.get("proof_effect") != "none":
                raise SupervisorReplayError("tool_observation_contract_mismatch")
            if (
                observation.get("runtime_identity") == "blueprint_local_deterministic_non_spend"
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
            invocation_observation_summaries.append(
                {
                    "tool_id": observation.get("tool_id"),
                    "status": observation.get("status"),
                    "typed_result": observation.get("typed_result"),
                    "typed_failure": observation.get("typed_failure"),
                    "produced_artifact_references": observation.get("produced_artifact_references")
                    or [],
                    "observation_digest": observation.get("observation_digest"),
                    "proof_effect": observation.get("proof_effect"),
                    "suggested_next_legal_actions": observation.get("suggested_next_legal_actions")
                    or [],
                }
            )
            for generated_ref in observation.get("produced_artifact_references") or []:
                if not isinstance(generated_ref, Mapping):
                    raise SupervisorReplayError("generated_artifact_reference_invalid")
                generated_path = (root / str(generated_ref.get("artifact_path") or "")).resolve()
                if root not in generated_path.parents:
                    raise SupervisorReplayError("generated_artifact_path_escape")
                generated_value = read_json(generated_path)
                artifact_type = str(generated_ref.get("artifact_type") or "")
                digest_fields = {
                    "evidence_plan.v1": "plan_digest",
                    "targeted_recapture_request.v1": "targeted_recapture_request_digest",
                    "task_evaluation_clarification_request.v1": ("clarification_request_digest"),
                    "task_evaluation_authorization_request.v1": ("authorization_request_digest"),
                    "task_evaluation_scenario_proposal_set.v1": ("scenario_proposal_set_digest"),
                    "task_evaluation_recovery_result.v1": "recovery_result_digest",
                }
                generated_digest = canonical_digest(
                    generated_value,
                    digest_field=digest_fields.get(artifact_type),
                )
                if generated_ref.get("artifact_digest") != generated_digest:
                    raise SupervisorReplayError("generated_artifact_digest_mismatch")
                generated_artifact_digests.append(generated_digest)
        invocation_capability = str(specialist_invocation.to_mapping().get("capability") or "")
        if invocation_observation_summaries != structured_observations_by_capability.get(
            invocation_capability,
            [],
        ):
            raise SupervisorReplayError("capability_structured_observations_mismatch")
    observations_dir = root / "observations"
    persisted_tool_observation_paths = (
        {path.resolve() for path in observations_dir.glob("*.json")}
        if observations_dir.is_dir()
        else set()
    )
    if persisted_tool_observation_paths != referenced_tool_observation_paths:
        raise SupervisorReplayError("tool_observation_inventory_mismatch")
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

    recapture_receipt = kernel_inputs.get("targeted_recapture_receipt")
    if recapture_receipt is not None:
        recapture_request = kernel_inputs.get("targeted_recapture_request")
        capture_build = kernel_inputs.get("capture_build")
        if recapture_request is None or capture_build is None:
            raise SupervisorReplayError("targeted_recapture_kernel_inputs_incomplete")
        try:
            validate_targeted_recapture_receipt(
                recapture_receipt,
                request=recapture_request,
                capture_build=capture_build,
            )
        except Phase2ArtifactError as exc:
            raise SupervisorReplayError("targeted_recapture_kernel_inputs_mismatch") from exc
        testbed = kernel_inputs.get("site_task_testbed")
        recorded_reinspection = kernel_inputs.get("recapture_reinspection")
        if testbed is not None:
            if recorded_reinspection is None:
                raise SupervisorReplayError("recapture_reinspection_kernel_input_missing")
            try:
                expected_reinspection = build_recapture_reinspection(
                    run_id=str(run_value["run_id"]),
                    request=recapture_request,
                    receipt=recapture_receipt,
                    capture_build=capture_build,
                    testbed=testbed,
                )
            except Phase2ArtifactError as exc:
                raise SupervisorReplayError("recapture_reinspection_rebuild_failed") from exc
            if recorded_reinspection != expected_reinspection:
                raise SupervisorReplayError("recapture_reinspection_kernel_result_mismatch")
        elif recorded_reinspection is not None:
            raise SupervisorReplayError("recapture_reinspection_without_testbed")

    clarification_receipt = kernel_inputs.get("clarification_receipt")
    if clarification_receipt is not None:
        clarification_request = kernel_inputs.get("clarification_request")
        if clarification_request is None:
            raise SupervisorReplayError("clarification_request_kernel_input_missing")
        try:
            validate_clarification_receipt(
                clarification_receipt,
                request=clarification_request,
            )
        except Phase2ArtifactError as exc:
            raise SupervisorReplayError("clarification_kernel_inputs_mismatch") from exc

    authorization_receipt = kernel_inputs.get("authorization_receipt")
    if authorization_receipt is not None:
        authorization_request = kernel_inputs.get("authorization_request")
        if authorization_request is None:
            raise SupervisorReplayError("authorization_request_kernel_input_missing")
        try:
            validate_authorization_receipt(
                authorization_receipt,
                request=authorization_request,
            )
        except Phase2ArtifactError as exc:
            raise SupervisorReplayError("authorization_kernel_inputs_mismatch") from exc

    deterministic_decision = kernel_inputs.get("decision_envelope")
    replayed_decision: dict[str, Any] | None = None
    if deterministic_decision is not None:
        validated = DecisionEnvelope.from_mapping(deterministic_decision).to_mapping()
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
        "manager_decision_digests": manager_decision_digests,
        "manager_invocation_digests": manager_invocation_digests,
        "manager_refusal_digests": manager_refusal_digests,
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
    if persist_report:
        write_json(root / "supervisor_replay_report.json", replay_value)
    return replay_value


__all__ = [
    "REPLAY_REPORT_SCHEMA_VERSION",
    "SupervisorReplayError",
    "replay_supervisor_run",
]
