"""Deterministically compile accepted capture evidence into one testbed version."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .capture_intake import CaptureIntakeError, validate_capture_intake_envelope
from .decision_evidence_contracts import (
    DecisionEvidenceRequest,
    DecisionEvidenceContractError,
    MaintainedSiteTaskTestbed,
    canonical_digest,
    canonical_json,
)
from .reconstruction_capability import (
    ReconstructionContractError,
    decide_simready_assets,
    normalize_reconstruction_result,
    score_robot_placements,
)


COMPILATION_RESULT_SCHEMA_VERSION = "site_task_testbed_compilation_result.v1"


class SiteTaskTestbedCompilerError(ValueError):
    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__("; ".join(self.errors))


def _clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise SiteTaskTestbedCompilerError(["artifact:not_json_serializable"]) from exc


def _text(value: Any) -> str:
    return str(value or "").strip()


def _is_digest(value: Any) -> bool:
    text = _text(value)
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _verified_digest(
    value: Mapping[str, Any], *, field: str, label: str
) -> dict[str, Any]:
    artifact = _clone(dict(value))
    supplied = artifact.get(field)
    if not _is_digest(supplied):
        raise SiteTaskTestbedCompilerError([f"{label}.{field}:invalid"])
    expected = canonical_digest(artifact, digest_field=field)
    if supplied != expected:
        raise SiteTaskTestbedCompilerError([f"{label}.{field}:mismatch"])
    return artifact


def _artifact_reference(value: Any, *, field: str) -> Any:
    if isinstance(value, list):
        if not value:
            raise SiteTaskTestbedCompilerError([f"artifact_references.{field}:empty"])
        return [_artifact_reference(item, field=f"{field}[{index}]") for index, item in enumerate(value)]
    if not isinstance(value, Mapping):
        raise SiteTaskTestbedCompilerError([f"artifact_references.{field}:invalid"])
    reference = _clone(dict(value))
    uri = _text(reference.get("uri"))
    lowered_uri = uri.lower()
    if not uri or not _is_digest(reference.get("digest")):
        raise SiteTaskTestbedCompilerError([f"artifact_references.{field}:binding_invalid"])
    if "@" in uri or any(
        marker in lowered_uri
        for marker in ("?token=", "&token=", "?signature=", "&signature=", "x-amz-credential")
    ):
        raise SiteTaskTestbedCompilerError(
            [f"artifact_references.{field}:credential_bearing_uri_forbidden"]
        )
    return reference


def _validate_approved_task(value: Mapping[str, Any]) -> dict[str, Any]:
    task = _verified_digest(
        value,
        field="approved_task_digest",
        label="approved_task_definition",
    )
    errors: list[str] = []
    if task.get("schema_version") != "approved_task_definition.v1":
        errors.append("approved_task_definition.schema_version:invalid")
    if task.get("approval_status") != "approved":
        errors.append("approved_task_definition.approval_status:not_approved")
    source = task.get("source_capture")
    if not isinstance(source, Mapping):
        errors.append("approved_task_definition.source_capture:missing")
    else:
        if not _text(source.get("intake_id")):
            errors.append("approved_task_definition.source_capture.intake_id:missing")
        if not _is_digest(source.get("capture_digest")):
            errors.append("approved_task_definition.source_capture.capture_digest:invalid")
    if not isinstance(task.get("task"), Mapping) or not task.get("task"):
        errors.append("approved_task_definition.task:missing")
    if errors:
        raise SiteTaskTestbedCompilerError(errors)
    return task


def _card(value: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _clone(dict(value))
    artifact["card_digest"] = canonical_digest(artifact, digest_field="card_digest")
    return artifact


def build_pipeline_owned_compilation_support(
    *,
    testbed_id: str,
    version: str,
    approved_task_definition: Mapping[str, Any],
    capture_qa_report: Mapping[str, Any],
    reconstruction_plan: Mapping[str, Any],
    robot_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive conservative testbed support artifacts inside Pipeline authority.

    The service caller supplies a robot configuration as owner-attested
    operational input. It does not supply SimReady conclusions, placement
    scores, evaluator artifacts, reset artifacts, or validated condition ranges.
    Until a qualified Pipeline method produces placement candidates or validated
    assets, those scientific decisions remain explicit abstentions/missing gates.
    """

    approved = _validate_approved_task(approved_task_definition)
    qa = _verified_digest(
        capture_qa_report,
        field="qa_report_digest",
        label="capture_qa_report",
    )
    plan = _verified_digest(
        reconstruction_plan,
        field="reconstruction_plan_digest",
        label="reconstruction_plan",
    )
    task = approved["task"]
    task_objects = [
        dict(row) for row in task.get("task_objects", []) if isinstance(row, Mapping)
    ]
    target_regions = [
        dict(row) for row in task.get("target_regions", []) if isinstance(row, Mapping)
    ]
    if not task_objects or not _text(task_objects[0].get("object_id")):
        raise SiteTaskTestbedCompilerError(["approved_task_definition.task_objects:missing"])
    if not target_regions or not _text(target_regions[0].get("region_id")):
        raise SiteTaskTestbedCompilerError(["approved_task_definition.target_regions:missing"])
    capture_digest = _text(approved["source_capture"].get("capture_digest"))
    claims = sorted(
        {_text(row) for row in plan.get("requested_claim_types", []) if _text(row)}
    )
    try:
        simready = decide_simready_assets(
            approved_task_digest=approved["approved_task_digest"],
            capture_digest=capture_digest,
            requested_claim_types=claims,
            task_objects=task_objects,
            asset_candidates=[],
        )
        placement = score_robot_placements(
            robot_binding=robot_binding,
            approved_task_digest=approved["approved_task_digest"],
            capture_digest=capture_digest,
            task_object_id=_text(task_objects[0].get("object_id")),
            target_region_id=_text(target_regions[0].get("region_id")),
            candidates=[],
        )
    except ReconstructionContractError as exc:
        raise SiteTaskTestbedCompilerError(
            [f"pipeline_owned_support:{item}" for item in exc.errors]
        ) from exc

    support_artifacts: dict[str, dict[str, Any]] = {
        "evaluator": {
            "schema_version": "testbed_evaluator_assignment.v1",
            "approved_task_digest": approved["approved_task_digest"],
            "requested_claim_types": claims,
            "status": "unassigned_until_evidence_planning",
            "proof_boundary": {
                "evaluation_definition_is_evidence_result": False,
                "provider_or_model_self_grading_allowed": False,
            },
        },
        "reset": {
            "schema_version": "testbed_reset_contract.v1",
            "approved_task_digest": approved["approved_task_digest"],
            "reset_contract": _clone(task.get("reset_contract", {})),
            "status": "owner_approved_definition_not_physical_verification",
        },
    }
    for artifact in support_artifacts.values():
        artifact["artifact_digest"] = canonical_digest(
            artifact, digest_field="artifact_digest"
        )
    artifact_references = {
        key: {
            "uri": f"testbed://{testbed_id}/{version}/{key}.json",
            "digest": artifact["artifact_digest"],
        }
        for key, artifact in support_artifacts.items()
    }
    supported_condition_ranges = {
        "accepted_capture_observation_scope": {
            "capture_digest": capture_digest,
            "qa_report_digest": qa["qa_report_digest"],
            "capture_authority_profile": approved["source_capture"].get(
                "capture_authority_profile"
            ),
        }
    }
    return {
        "simready_decision": simready,
        "robot_placement_result": placement,
        "artifact_references": artifact_references,
        "pipeline_owned_support_artifacts": support_artifacts,
        "supported_condition_ranges": supported_condition_ranges,
    }


def _build_cards(
    *,
    testbed_id: str,
    version: str,
    envelope: Mapping[str, Any],
    qa: Mapping[str, Any],
    approved: Mapping[str, Any],
    plan: Mapping[str, Any],
    placement: Mapping[str, Any],
    supported_condition_ranges: Mapping[str, Any],
) -> dict[str, Any]:
    task = approved["task"]
    site_card = _card(
        {
            "schema_version": "site_task_testbed_site_card.v1",
            "site_card_id": f"site-card:{testbed_id}:{version}",
            "site_id": envelope["scene_id"],
            "source_capture": {
                "intake_id": envelope["intake_id"],
                "capture_digest": approved["source_capture"]["capture_digest"],
                "envelope_digest": envelope["envelope_digest"],
                "qa_report_digest": qa["qa_report_digest"],
                "capture_authority_profile": envelope["capture_authority_profile"],
            },
            "supported_condition_ranges": _clone(dict(supported_condition_ranges)),
            "claim_ceiling": qa["claim_ceiling"],
            "claim_boundary": "capture_backed_site_scope_not_physical_or_safety_proof",
        }
    )
    task_card = _card(
        {
            "schema_version": "site_task_testbed_task_card.v1",
            "task_card_id": f"task-card:{approved['approved_task_id']}",
            "approved_task_id": approved["approved_task_id"],
            "approved_task_digest": approved["approved_task_digest"],
            "description": task.get("description"),
            "task_family": task.get("task_family"),
            "measurable_success_conditions": task.get("measurable_success_conditions", []),
            "reset_contract": task.get("reset_contract", {}),
            "task_objects": task.get("task_objects", []),
            "target_regions": task.get("target_regions", []),
            "required_robot_capabilities": task.get("required_robot_capabilities", []),
            "prohibited_evaluator_identities": approved.get(
                "prohibited_evaluator_identities", []
            ),
            "claim_boundary": "approved_intent_and_evaluation_scope_not_task_success",
        }
    )
    scenario_card = _card(
        {
            "schema_version": "site_task_testbed_scenario_card.v1",
            "scenario_card_id": f"scenario-card:{testbed_id}:{version}:base",
            "task_card_id": task_card["task_card_id"],
            "robot_binding_digest": placement["robot_binding_digest"],
            "selected_robot_placement_id": placement.get("selected_candidate_id"),
            "supported_condition_ranges": _clone(dict(supported_condition_ranges)),
            "reset_contract": task.get("reset_contract", {}),
            "claim_boundary": "scenario_scope_not_simulator_or_physical_outcome",
        }
    )
    eval_card = _card(
        {
            "schema_version": "site_task_testbed_eval_card.v1",
            "eval_card_id": f"eval-card:{testbed_id}:{version}:base",
            "task_card_id": task_card["task_card_id"],
            "scenario_card_id": scenario_card["scenario_card_id"],
            "requested_claim_types": plan.get("requested_claim_types", []),
            "measurable_success_conditions": task.get("measurable_success_conditions", []),
            "decision_evidence_request_compiled": False,
            "prohibited_claims": [
                "physical_task_success",
                "deployment_readiness",
                "safety_certification",
                "general_policy_ranking_validity",
            ],
            "comparative_policy_ranking_verdict": "thesis_not_supported",
            "claim_boundary": "evaluation_definition_not_evidence_result",
        }
    )
    return {
        "site_card": site_card,
        "task_cards": [task_card],
        "scenario_cards": [scenario_card],
        "eval_cards": [eval_card],
    }


def compile_site_task_testbed(
    *,
    testbed_id: str,
    version: str,
    capture_intake_envelope: Mapping[str, Any],
    capture_qa_report: Mapping[str, Any],
    approved_task_definition: Mapping[str, Any],
    reconstruction_plan: Mapping[str, Any],
    reconstruction_results: Sequence[Mapping[str, Any]],
    simready_decision: Mapping[str, Any],
    robot_placement_result: Mapping[str, Any],
    artifact_references: Mapping[str, Any],
    supported_condition_ranges: Mapping[str, Any],
    previous_testbed: Mapping[str, Any] | None = None,
    pipeline_owned_support_artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile one immutable router-compatible testbed from exact input digests."""

    errors: list[str] = []
    try:
        envelope = validate_capture_intake_envelope(capture_intake_envelope)
    except CaptureIntakeError as exc:
        raise SiteTaskTestbedCompilerError(
            [f"capture_intake_envelope:{item}" for item in exc.errors]
        ) from exc
    qa = _verified_digest(capture_qa_report, field="qa_report_digest", label="capture_qa_report")
    approved = _validate_approved_task(approved_task_definition)
    plan = _verified_digest(
        reconstruction_plan,
        field="reconstruction_plan_digest",
        label="reconstruction_plan",
    )
    simready = _verified_digest(
        simready_decision,
        field="simready_decision_digest",
        label="simready_decision",
    )
    placement = _verified_digest(
        robot_placement_result,
        field="robot_placement_digest",
        label="robot_placement_result",
    )
    if qa.get("schema_version") != "capture_qa_report.v1":
        errors.append("capture_qa_report:schema_version_mismatch")
    if plan.get("schema_version") != "reconstruction_plan.v1":
        errors.append("reconstruction_plan:schema_version_mismatch")
    if simready.get("schema_version") != "simready_asset_decision.v1":
        errors.append("simready_decision:schema_version_mismatch")
    if placement.get("schema_version") != "robot_placement_result.v1":
        errors.append("robot_placement_result:schema_version_mismatch")
    if qa.get("status") != "accepted" or qa.get("state") != "capture_accepted":
        errors.append("capture_qa_report:not_accepted")
    if qa.get("intake_id") != envelope.get("intake_id"):
        errors.append("capture_qa_report:intake_mismatch")
    if qa.get("envelope_digest") != envelope.get("envelope_digest"):
        errors.append("capture_qa_report:envelope_digest_mismatch")
    source = approved["source_capture"]
    capture_digest = source["capture_digest"]
    if source.get("intake_id") != envelope.get("intake_id"):
        errors.append("approved_task_definition:intake_mismatch")
    if source.get("capture_authority_profile") != envelope.get("capture_authority_profile"):
        errors.append("approved_task_definition:capture_authority_profile_mismatch")
    plan_source = plan.get("source_capture")
    if not isinstance(plan_source, Mapping) or (
        plan_source.get("intake_id") != envelope.get("intake_id")
        or plan_source.get("capture_digest") != capture_digest
        or plan_source.get("capture_authority_profile")
        != envelope.get("capture_authority_profile")
    ):
        errors.append("reconstruction_plan:source_capture_mismatch")
    if simready.get("approved_task_digest") != approved["approved_task_digest"]:
        errors.append("simready_decision:approved_task_mismatch")
    if simready.get("capture_digest") != capture_digest:
        errors.append("simready_decision:capture_digest_mismatch")
    if placement.get("approved_task_digest") != approved["approved_task_digest"]:
        errors.append("robot_placement_result:approved_task_mismatch")
    if placement.get("capture_digest") != capture_digest:
        errors.append("robot_placement_result:capture_digest_mismatch")
    robot_binding = placement.get("robot_binding")
    if not isinstance(robot_binding, Mapping) or canonical_digest(robot_binding) != placement.get(
        "robot_binding_digest"
    ):
        errors.append("robot_placement_result:robot_binding_digest_mismatch")

    normalized_results: list[dict[str, Any]] = []
    selected_outputs_by_profile: dict[str, set[str]] = {}
    for row in plan.get("selected_methods", []):
        if not isinstance(row, Mapping):
            errors.append("reconstruction_plan:selected_method_invalid")
            continue
        profile_digest = _text(row.get("method_profile_digest"))
        representations = {
            _text(item) for item in row.get("representations", []) if _text(item)
        }
        if not _is_digest(profile_digest) or not representations:
            errors.append("reconstruction_plan:selected_method_binding_invalid")
            continue
        selected_outputs_by_profile[profile_digest] = representations
    for index, result in enumerate(reconstruction_results):
        try:
            normalized = normalize_reconstruction_result(result)
        except ReconstructionContractError as exc:
            errors.extend(f"reconstruction_results[{index}]:{item}" for item in exc.errors)
            continue
        if normalized.get("intake_id") != envelope.get("intake_id"):
            errors.append(f"reconstruction_results[{index}]:intake_mismatch")
        if normalized.get("capture_digest") != capture_digest:
            errors.append(f"reconstruction_results[{index}]:capture_digest_mismatch")
        selected_outputs = selected_outputs_by_profile.get(
            _text(normalized.get("method_profile_digest"))
        )
        if selected_outputs is None:
            errors.append(f"reconstruction_results[{index}]:method_not_in_plan")
        elif not set(normalized["outputs"]).issubset(selected_outputs):
            errors.append(f"reconstruction_results[{index}]:output_not_selected_in_plan")
        normalized_results.append(normalized)

    compiler_owned_refs = {"site_card", "task_cards", "scenario_cards", "eval_cards"}
    required_refs = {"evaluator", "reset"}
    references: dict[str, Any] = {}
    for key in sorted(compiler_owned_refs.intersection(artifact_references)):
        errors.append(f"artifact_references.{key}:compiler_owned")
    for key in sorted(required_refs):
        if key not in artifact_references:
            errors.append(f"artifact_references.{key}:missing")
            continue
        try:
            references[key] = _artifact_reference(artifact_references[key], field=key)
        except SiteTaskTestbedCompilerError as exc:
            errors.extend(exc.errors)
    for key in sorted(set(artifact_references) - required_refs - compiler_owned_refs):
        try:
            references[key] = _artifact_reference(artifact_references[key], field=key)
        except SiteTaskTestbedCompilerError as exc:
            errors.extend(exc.errors)
    support_artifacts = _clone(dict(pipeline_owned_support_artifacts or {}))
    if support_artifacts:
        for key in sorted(required_refs):
            artifact = support_artifacts.get(key)
            reference = references.get(key)
            if not isinstance(artifact, Mapping):
                errors.append(f"pipeline_owned_support_artifacts.{key}:missing")
                continue
            supplied_digest = artifact.get("artifact_digest")
            if not _is_digest(supplied_digest) or supplied_digest != canonical_digest(
                artifact, digest_field="artifact_digest"
            ):
                errors.append(
                    f"pipeline_owned_support_artifacts.{key}.artifact_digest:mismatch"
                )
            if not isinstance(reference, Mapping) or reference.get("digest") != supplied_digest:
                errors.append(f"pipeline_owned_support_artifacts.{key}:reference_mismatch")
    if not isinstance(supported_condition_ranges, Mapping) or not supported_condition_ranges:
        errors.append("supported_condition_ranges:missing")

    predecessor_digest = None
    supersedes: list[str] = []
    previous: dict[str, Any] | None = None
    if previous_testbed is not None:
        try:
            previous = MaintainedSiteTaskTestbed.from_mapping(previous_testbed).to_mapping()
        except DecisionEvidenceContractError as exc:
            errors.extend(f"previous_testbed:{item}" for item in exc.errors)
        else:
            predecessor_digest = previous["testbed_digest"]
            supersedes = [predecessor_digest]
            if previous["testbed_id"] != testbed_id:
                errors.append("previous_testbed:testbed_id_mismatch")
            if previous["version"] == version:
                errors.append("previous_testbed:version_must_change")
    if errors:
        raise SiteTaskTestbedCompilerError(errors)

    compiled_cards = _build_cards(
        testbed_id=testbed_id,
        version=version,
        envelope=envelope,
        qa=qa,
        approved=approved,
        plan=plan,
        placement=placement,
        supported_condition_ranges=supported_condition_ranges,
    )
    card_filenames = {
        "site_card": "site_card",
        "task_cards": "task_card",
        "scenario_cards": "scenario_card",
        "eval_cards": "eval_card",
    }
    for key, prefix in card_filenames.items():
        value = compiled_cards[key]
        rows = value if isinstance(value, list) else [value]
        references[key] = [
            {
                "uri": f"testbed://{testbed_id}/{version}/{prefix}_{index}.json",
                "digest": row["card_digest"],
            }
            for index, row in enumerate(rows, start=1)
        ]
        if key == "site_card":
            references[key] = references[key][0]

    reconstruction_layers: dict[str, list[dict[str, Any]]] = {
        "appearance_layer": [],
        "metric_reference_layer": [],
        "semantic_layer": [],
        "physics_layer": [],
    }
    output_to_layer = {
        "appearance_layer": "appearance_layer",
        "calibrated_frames": "metric_reference_layer",
        "metric_reference_layer": "metric_reference_layer",
        "semantic_layer": "semantic_layer",
        "physics_layer": "physics_layer",
        "collision_geometry": "physics_layer",
        "articulated_object_asset": "physics_layer",
    }
    for result in normalized_results:
        for output in result["outputs"]:
            layer = output_to_layer.get(output)
            if layer:
                reconstruction_layers[layer].append(
                    {
                        "output": output,
                        "result_id": result["result_id"],
                        "result_digest": result["reconstruction_result_digest"],
                        "asset_references": result["asset_references"],
                        "generated_regions": result["generated_regions"],
                        "claim_ceiling": result["claim_ceiling"],
                    }
                )
    for rows in reconstruction_layers.values():
        rows.sort(key=lambda row: (row["output"], row["result_digest"]))

    missing_representations = sorted(
        _text(row.get("representation"))
        for row in plan.get("missing_representations", [])
        if isinstance(row, Mapping) and _text(row.get("representation"))
    )
    unsupported = set(_text(item) for item in qa.get("prohibited_claims", []) if _text(item))
    unsupported.update(f"missing_representation:{item}" for item in missing_representations)
    if simready.get("status") != "complete":
        unsupported.add("simready_asset_validation_incomplete")
    if placement.get("status") != "candidate_selected":
        unsupported.add("robot_placement_not_established")
    unsupported.update(
        {
            "physical_task_success",
            "deployment_readiness",
            "safety_certification",
            "general_policy_ranking_validity",
            "comparative_policy_ranking:thesis_not_supported",
        }
    )
    task_body = approved["task"]
    reset_contract = task_body.get("reset_contract", {})
    robot_binding = placement["robot_binding"]
    selected_placements = [
        row
        for row in placement.get("accepted_candidates", [])
        if isinstance(row, Mapping)
        and row.get("candidate_id") == placement.get("selected_candidate_id")
    ]
    selected_placement = _clone(selected_placements[0]) if len(selected_placements) == 1 else None
    testbed = MaintainedSiteTaskTestbed.from_mapping(
        {
            "schema_version": "maintained_site_task_testbed.v1",
            "testbed_id": testbed_id,
            "version": version,
            "predecessor_testbed_digest": predecessor_digest,
            "supersedes": supersedes,
            "source_capture_bundles": [
                {
                    "bundle_id": envelope["intake_id"],
                    "version": _text(envelope.get("capture_authority_profile")),
                    "digest": capture_digest,
                    "envelope_digest": envelope["envelope_digest"],
                    "qa_report_digest": qa["qa_report_digest"],
                }
            ],
            "artifact_references": references,
            "pipeline_owned_support_artifacts": support_artifacts,
            "compiled_cards": compiled_cards,
            "approved_task_definition": {
                "approved_task_id": approved["approved_task_id"],
                "digest": approved["approved_task_digest"],
                "approval_decision_digest": approved["approval_decision_digest"],
            },
            "task_distribution": {
                "task_family": task_body.get("task_family"),
                "tasks": [approved["approved_task_id"]],
                "measurable_success_conditions": task_body.get(
                    "measurable_success_conditions", []
                ),
            },
            "supported_condition_ranges": _clone(dict(supported_condition_ranges)),
            "robot_sensor_controller_bindings": {
                "embodiment": {
                    "robot_id": robot_binding["robot_id"],
                    "version": robot_binding["embodiment_version"],
                    "base_footprint": robot_binding["base_footprint"],
                    "end_effector_id": robot_binding["end_effector_id"],
                    "reach_envelope": robot_binding.get("reach_envelope"),
                },
                "sensors": robot_binding["sensors"],
                "controller_action_representation": {
                    "controller_id": robot_binding["controller_id"]
                },
                "selected_robot_placement_id": placement.get("selected_candidate_id"),
                "selected_robot_placement": selected_placement,
                "robot_binding_digest": placement["robot_binding_digest"],
            },
            "governance": {
                "rights": envelope["governance"]["rights"],
                "consent": envelope["governance"]["consent"],
                "privacy": envelope["governance"]["privacy"],
                "retention": envelope["governance"]["retention"],
                "revocation": envelope["governance"]["revocation"],
                "provider_constraints": envelope["governance"]["provider_constraints"],
                "allowed_uses": envelope["governance"]["allowed_uses"],
            },
            "evidence_inventory": sorted(
                [
                    {
                        "evidence_id": "raw_capture",
                        "digest": capture_digest,
                        "authority": envelope["capture_authority_profile"],
                    },
                    {
                        "evidence_id": "capture_qa",
                        "digest": qa["qa_report_digest"],
                        "status": qa["status"],
                    },
                    {
                        "evidence_id": "simready_decision",
                        "digest": simready["simready_decision_digest"],
                        "status": simready["status"],
                    },
                    {
                        "evidence_id": "robot_placement",
                        "digest": placement["robot_placement_digest"],
                        "status": placement["status"],
                    },
                    *[
                        {
                            "evidence_id": f"reconstruction:{result['result_id']}",
                            "digest": result["reconstruction_result_digest"],
                            "outputs": result["outputs"],
                            "claim_ceiling": result["claim_ceiling"],
                        }
                        for result in normalized_results
                    ],
                ],
                key=lambda row: row["evidence_id"],
            ),
            "validation_envelope": {
                "capture_accepted": True,
                "capture_authority_profile": envelope["capture_authority_profile"],
                "capture_claim_ceiling": qa["claim_ceiling"],
                "reconstruction_plan_digest": plan["reconstruction_plan_digest"],
                "reconstruction_layers": reconstruction_layers,
                "missing_representations": missing_representations,
                "simready_decision_digest": simready["simready_decision_digest"],
                "robot_placement_digest": placement["robot_placement_digest"],
                "generated_regions_are_masked": all(
                    all(_text(region.get("mask_reference")) for region in result["generated_regions"])
                    for result in normalized_results
                ),
                "physical_task_success_established": False,
            },
            "task_objects": task_body.get("task_objects", []),
            "target_regions": task_body.get("target_regions", []),
            "reset_contract": reset_contract,
            "reconstruction_provenance": [
                {
                    "result_id": result["result_id"],
                    "result_digest": result["reconstruction_result_digest"],
                    "method_id": result["method_id"],
                    "provider_identity": result["provider_identity"],
                    "cost_usd": result["cost_usd"],
                }
                for result in normalized_results
            ],
            "known_unsupported_conditions": sorted(unsupported),
            "invalidation_triggers": sorted(
                {
                    "capture_revoked",
                    "layout_changed",
                    "task_object_changed",
                    "robot_binding_changed",
                    "controller_changed",
                    "reconstruction_corrected",
                    "supported_condition_range_exceeded",
                }
            ),
            "physical_outcome_history_refs": [],
            "lifecycle_state": "active",
            "proof_boundary": {
                "appearance_is_collision_truth": False,
                "generated_completion_is_observed_truth": False,
                "simulation_is_physical_success": False,
                "deployment_or_safety_approved": False,
                "comparative_policy_ranking_verdict": "thesis_not_supported",
            },
        }
    ).to_mapping()
    return testbed


def write_testbed_version(
    *, output_root: str | Path, testbed: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        verified = MaintainedSiteTaskTestbed.from_mapping(testbed).to_mapping()
    except DecisionEvidenceContractError as exc:
        raise SiteTaskTestbedCompilerError(exc.errors) from exc
    root = Path(output_root).expanduser().resolve()
    path = (
        root
        / verified["testbed_id"]
        / verified["version"]
        / f"{verified['testbed_digest'].removeprefix('sha256:')}.json"
    )
    payload = (canonical_json(verified) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)

    def write_once(target: Path, content: bytes) -> bool:
        replayed = False
        try:
            with target.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            replayed = True
            if target.read_bytes() != content:
                raise SiteTaskTestbedCompilerError([f"immutable_artifact_conflict:{target.name}"])
        return replayed

    version_binding = {
        "schema_version": "site_task_testbed_version_binding.v1",
        "testbed_id": verified["testbed_id"],
        "version": verified["version"],
        "testbed_digest": verified["testbed_digest"],
    }
    binding_payload = (canonical_json(version_binding) + "\n").encode("utf-8")
    lock_path = path.parent.parent / ".version-lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        binding_path = path.parent / "version_binding.json"
        if binding_path.is_file() and binding_path.read_bytes() != binding_payload:
            raise SiteTaskTestbedCompilerError(["testbed_version_digest_conflict"])
        write_once(binding_path, binding_payload)
        already_exists = write_once(path, payload)
        cards = verified.get("compiled_cards")
        if not isinstance(cards, Mapping):
            raise SiteTaskTestbedCompilerError(["compiled_cards:missing"])
        for key, prefix in (
            ("site_card", "site_card"),
            ("task_cards", "task_card"),
            ("scenario_cards", "scenario_card"),
            ("eval_cards", "eval_card"),
        ):
            value = cards.get(key)
            rows = value if isinstance(value, list) else [value]
            if not rows or not all(isinstance(row, Mapping) for row in rows):
                raise SiteTaskTestbedCompilerError([f"compiled_cards.{key}:invalid"])
            for index, row in enumerate(rows, start=1):
                card_payload = (canonical_json(row) + "\n").encode("utf-8")
                write_once(path.parent / f"{prefix}_{index}.json", card_payload)
        support_artifacts = verified.get("pipeline_owned_support_artifacts", {})
        if isinstance(support_artifacts, Mapping):
            for key in ("evaluator", "reset"):
                artifact = support_artifacts.get(key)
                if isinstance(artifact, Mapping):
                    artifact_payload = (canonical_json(artifact) + "\n").encode("utf-8")
                    write_once(path.parent / f"{key}.json", artifact_payload)
    result = {
        "schema_version": COMPILATION_RESULT_SCHEMA_VERSION,
        "status": "testbed_ready",
        "already_exists": already_exists,
        "testbed_id": verified["testbed_id"],
        "version": verified["version"],
        "testbed_digest": verified["testbed_digest"],
        "artifact_path": str(path),
        "proof_boundary": verified["proof_boundary"],
    }
    result["compilation_result_digest"] = canonical_digest(
        result, digest_field="compilation_result_digest"
    )
    return result


def write_testbed_decision_evidence_request(
    *,
    output_root: str | Path,
    testbed: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    verified_testbed = MaintainedSiteTaskTestbed.from_mapping(testbed).to_mapping()
    verified_request = DecisionEvidenceRequest.from_mapping(request).to_mapping()
    if (
        verified_request["testbed_id"] != verified_testbed["testbed_id"]
        or verified_request["testbed_version"] != verified_testbed["version"]
        or verified_request["testbed_digest"] != verified_testbed["testbed_digest"]
    ):
        raise SiteTaskTestbedCompilerError(["decision_evidence_request:testbed_mismatch"])
    root = Path(output_root).expanduser().resolve()
    version_root = root / verified_testbed["testbed_id"] / verified_testbed["version"]
    testbed_path = version_root / f"{verified_testbed['testbed_digest'][7:]}.json"
    if not testbed_path.is_file():
        raise SiteTaskTestbedCompilerError(["decision_evidence_request:testbed_not_persisted"])
    path = version_root / f"decision_evidence_request-{verified_request['request_digest'][7:]}.json"
    payload = (canonical_json(verified_request) + "\n").encode("utf-8")
    already_exists = False
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        already_exists = True
        if path.read_bytes() != payload:
            raise SiteTaskTestbedCompilerError(
                ["decision_evidence_request:immutable_artifact_conflict"]
            )
    return {
        "schema_version": "testbed_decision_evidence_request_write.v1",
        "already_exists": already_exists,
        "testbed_digest": verified_testbed["testbed_digest"],
        "request_digest": verified_request["request_digest"],
        "artifact_reference": {
            "uri": (
                f"testbed://{verified_testbed['testbed_id']}/{verified_testbed['version']}/"
                f"decision_evidence_request-{verified_request['request_digest'][7:]}.json"
            ),
            "digest": verified_request["request_digest"],
        },
    }


__all__ = [
    "SiteTaskTestbedCompilerError",
    "compile_site_task_testbed",
    "build_pipeline_owned_compilation_support",
    "write_testbed_version",
    "write_testbed_decision_evidence_request",
]
