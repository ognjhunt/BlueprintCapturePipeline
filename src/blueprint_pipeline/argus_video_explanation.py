"""Prepared default video explanation route; offline only, never task authority.

Build blinded Argus request specifications and import retained responses. There
is no provider client, credential lookup, upload, cost reservation or launcher.
The existing deterministic grader remains the success/failure authority.
"""

from __future__ import annotations

import copy
from pathlib import Path

from .argus_shadow import (
    ARGUS_COMMIT, TARGET_MODEL, ShadowError, check_seal, digest, normalize_argus,
    number, read_bound, read_json, seal,
)
from .decision_evidence_contracts import canonical_digest
from .adp_articulated_task_success_contract import task_kind_of_contract, validate_task_success_contract
from .episode_interpretation import EpisodeInterpretationRequest, InterpreterIdentity

ADAPTER_ID = "argus_adapted_video_explainer_v1"
PROFILE_SCHEMA = "policy_canary_episode_interpreter_profile.v3"
OFFLINE_REASON = "argus_video_explanation_offline_only"
REPO = Path(__file__).resolve().parents[2]
PROMPT_BINDING = {
    "path": "docs/experiments/argus_shadow_20261001/argus_adapted_teleop.txt",
    "sha256": "sha256:308ffb611d39636709e57e2aad7f76259ff5bbb32fe5931386973c021b5f99d9",
    "size_bytes": 30013,
}
EXPLANATION_OVERRIDE = (
    "\nBLUEPRINT ADVISORY VIDEO EXPLANATION CONTRACT v1\n"
    "Describe apparent events only. Never decide authoritative task success, "
    "ranking, promotion, safety or physical validity. Return the Argus labels "
    "schema plus criterion_evidence: an object keyed by EVERY exact acceptance "
    "criterion. Each value has status (satisfied, violated, unknown), finding, "
    "evidence_role and evidence_digest from the supplied artifacts. Missing or "
    "unobservable required criteria must be unknown and completion unclear. "
    "Preserve observed failures, recovery, retreat and success-then-undone. "
    "Issue severity concerns training-data impact only. Confidence, if supplied, "
    "is annotation confidence, not a calibrated probability of task success."
)
TAXONOMY = {
    "failed_grasp": "grasp_failure",
    "dropped_object": "object_drop",
    "knocked_object": "object_displacement",
    "collision": "apparent_collision",
    "prolonged_struggle": "manipulation_difficulty",
    "hesitation": "inefficient_motion",
    "unnecessary_motion": "inefficient_motion",
    "goal_undone": "goal_regression",
    "incomplete_task": "task_incomplete",
    "instruction_mismatch": "task_mismatch",
    "human_intervention": "human_intervention",
    "camera_fault": "visual_evidence_gap",
    "recording_fault": "visual_evidence_gap",
    "state_video_mismatch": "evidence_conflict",
    "unintended_out_of_view": "visual_evidence_gap",
    "setup_change": "scene_change",
    "scene_reset": "scene_change",
    "idle_stretch": "inefficient_motion",
    "truncated_episode": "incomplete_recording",
}
OBSERVATIONAL_ROLES = frozenset({"state_trace", "contact_force_trace", "lossless_frames", "review_videos"})


def default_video_profile() -> dict:
    """Select the adapted Argus route without conferring invocation authority."""
    return seal({
        "schema_version": PROFILE_SCHEMA, "status": "prepared_offline",
        "interpreter_id": ADAPTER_ID, "provider_id": "openai",
        "runtime": "argus_offline_specification", "model": "gpt-6.1-sol",
        "model_version": "gpt-6.1-sol", "model_version_is_alias": True,
        "reasoning_effort": "medium", "argus_commit": ARGUS_COMMIT,
        "prompt_contract_version": "argus_acceptance_video_explanation.v1",
        "adapted_prompt": copy.deepcopy(PROMPT_BINDING),
        "advisory_only": True, "live_processing_enabled": False,
        "paid_execution_authorized": False, "max_cost_usd": 0,
        "authoritative_grader_switch_authorized": False,
        "validated_on_blueprint_footage": False,
    }, "profile_digest")


def default_video_identity() -> InterpreterIdentity:
    profile = default_video_profile()
    return InterpreterIdentity(
        interpreter_id=ADAPTER_ID, principal_kind="independent_interpreter",
        provider_id="openai", execution_site="local",
        runtime=profile["runtime"], model=profile["model"],
        model_version=profile["model_version"],
    )


def prepare_video_explanation(
    request: EpisodeInterpretationRequest, *, intended_task: str, duration_s: float,
    evidence_kind: str,
) -> dict:
    """Compile an offline specification from the existing sealed input seam."""
    if not isinstance(intended_task, str) or not intended_task.strip():
        raise ShadowError("video_explanation_intended_task_missing")
    if number(duration_s, "duration") <= 0:
        raise ShadowError("video_explanation_duration_invalid")
    if evidence_kind not in {
        "physical_recording", "simulator_recording", "generated_video", "synthetic_fixture",
    }:
        raise ShadowError("video_explanation_evidence_kind_missing")
    receipt = request.input_receipt
    if receipt.get("input_bundle_digest") != canonical_digest(
        receipt, digest_field="input_bundle_digest"
    ) or receipt.get("all_source_bytes_rehashed") is not True:
        raise ShadowError("video_explanation_input_receipt_invalid")
    # A historical receipt does not prove bytes still exist unchanged. Recheck
    # local files only: never invoke a streamed frame reader or archive service.
    for value in receipt["artifacts"].values():
        for ref in value if isinstance(value, list) else [value]:
            read_bound(request.evidence_root, {
                "path": ref["relative_path"], "sha256": ref["sha256"],
                "size_bytes": ref["size_bytes"],
            })
    contract = request.task_success_contract
    validate_task_success_contract(contract, task_kind=task_kind_of_contract(contract), require_confirmed=True)
    if (contract.get("provenance", {}).get("confirmation_status") != "confirmed"
            or contract.get("contract_digest") != receipt["task_success_contract_digest"]):
        raise ShadowError("video_explanation_acceptance_unconfirmed_or_changed")
    for value, role in ((request.state_trace, "state_trace"),
                        (request.contact_force_trace, "contact_force_trace")):
        expected = receipt["artifacts"][role]["logical_digest"]
        if value.get("trace_digest") != expected or canonical_digest(value, digest_field="trace_digest") != expected:
            raise ShadowError("video_explanation_native_trace_changed")
    if (request.deterministic_score.get("report_digest") != receipt["deterministic_score_digest"]
            or canonical_digest(request.deterministic_score, digest_field="report_digest") != receipt["deterministic_score_digest"]):
        raise ShadowError("video_explanation_authoritative_score_changed")
    artifacts = {k: copy.deepcopy(v) for k, v in receipt["artifacts"].items()
                 if k != "deterministic_score"}
    if not artifacts.get("review_videos") or not artifacts.get("lossless_frames"):
        raise ShadowError("video_explanation_visual_evidence_missing")
    prompt = read_bound(REPO, PROMPT_BINDING).decode() + EXPLANATION_OVERRIDE
    return seal({
        "schema_version": "argus_video_explanation_request.v1",
        "episode_id": request.episode_id, "input_bundle_digest": receipt["input_bundle_digest"],
        "profile": default_video_profile(), "model": TARGET_MODEL,
        "prompt": prompt, "prompt_digest": digest(prompt.encode()),
        "model_input": {
            "intended_task": intended_task.strip(), "task_success_contract": copy.deepcopy(contract),
            "duration_s": duration_s, "declared_evidence_kind": evidence_kind,
            "artifacts": artifacts, "state_trace": copy.deepcopy(request.state_trace),
            "contact_force_trace": copy.deepcopy(request.contact_force_trace),
        },
        # Retained for presentation only; absent from the blinded model_input.
        "authoritative_result": {
            "score_digest": receipt["deterministic_score_digest"],
            **{k: copy.deepcopy(request.deterministic_score.get(k)) for k in (
                "status", "task_succeeded", "outcome", "failed_criteria")},
        },
        "request_kind": "offline_specification_not_provider_payload",
        "source_verification": "fresh_local_byte_rehash_no_streamed_reads",
        "model_calls": 0, "provider_disclosure_authorized": False,
        "physical_validity_established": False, "real_corpus_admission": False,
    }, "request_digest")


def _artifact_digests(artifacts: dict) -> dict[str, set[str]]:
    inventory = {}
    for role, value in artifacts.items():
        records = value if isinstance(value, list) else [value]
        inventory[role] = {r[key] for r in records for key in ("sha256", "logical_digest") if key in r}
    return inventory


def _temporal_annotations(labels: dict, duration: float) -> dict:
    """Preserve Argus's explanatory timeline while checking retained times."""
    result = {name: copy.deepcopy(labels.get(name, [])) for name in (
        "timeline", "key_events", "state_changes", "recovery")}
    columns = labels.get("timeline_columns", [])
    if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
        raise ShadowError("video_explanation_timeline_columns_invalid")
    if len(set(columns)) != len(columns):
        raise ShadowError("video_explanation_timeline_columns_invalid")
    result["timeline_columns"] = copy.deepcopy(columns)
    def timestamp(value):
        time = number(value, "annotation_time")
        if not 0 <= time <= duration:
            raise ShadowError("video_explanation_annotation_time_invalid")
        return time
    if not isinstance(result["timeline"], list):
        raise ShadowError("video_explanation_timeline_invalid")
    for row in result["timeline"]:
        if (not isinstance(row, list) or len(row) != len(columns)
                or not {"start_s", "end_s"} <= set(columns)):
            raise ShadowError("video_explanation_timeline_invalid")
        if timestamp(row[columns.index("end_s")]) < timestamp(row[columns.index("start_s")]):
            raise ShadowError("video_explanation_annotation_interval_invalid")
    for name in ("key_events", "state_changes", "recovery"):
        if not isinstance(result[name], list):
            raise ShadowError("video_explanation_annotations_invalid")
        for row in result[name]:
            if not isinstance(row, dict):
                raise ShadowError("video_explanation_annotations_invalid")
            for field in ("t_s", "failure_t_s", "recovered_at_s"):
                if row.get(field) is not None:
                    timestamp(row[field])
            if (row.get("failure_t_s") is not None and row.get("recovered_at_s") is not None
                    and row["recovered_at_s"] < row["failure_t_s"]):
                raise ShadowError("video_explanation_annotation_interval_invalid")
    return result


def import_video_explanation(prepared: dict, record: dict, *, evidence_root: Path) -> dict:
    """Normalize a bound retained response; never invoke a provider or scorer."""
    check_seal(prepared, "request_digest")
    if (prepared.get("profile") != default_video_profile()
            or prepared.get("model") != TARGET_MODEL
            or record.get("request_digest") != prepared["request_digest"]):
        raise ShadowError("video_explanation_request_or_route_mismatch")
    identity = record.get("inference_identity")
    if (not isinstance(identity, dict) or identity.get("model_requested") != TARGET_MODEL
            or identity.get("model_served") not in {TARGET_MODEL, "gpt-6.1-sol"}
            or identity.get("provider") != "openai" or not identity.get("generation_id")):
        raise ShadowError("video_explanation_model_provenance_missing_or_mismatch")
    raw = read_json(evidence_root, record["raw_response"])
    normalized = normalize_argus(raw, prepared["model_input"]["duration_s"])
    labels = raw.get("labels", raw)
    checks = labels.get("criterion_evidence", {})
    criteria = prepared["model_input"]["task_success_contract"]["criteria"]
    if not isinstance(checks, dict) or not set(checks) <= set(criteria):
        raise ShadowError("video_explanation_unknown_acceptance_criterion")
    inventory = _artifact_digests(prepared["model_input"]["artifacts"])
    reviewed = {}
    for name in criteria:
        if criteria[name].get("mode") == "ignored":
            reviewed[name] = {"status": "not_required", "finding": "Explicitly ignored by confirmed contract"}
            continue
        check = checks.get(name, {"status": "unknown", "finding": "Required criterion evidence missing"})
        if not isinstance(check, dict) or check.get("status") not in {"satisfied", "violated", "unknown"}:
            raise ShadowError("video_explanation_criterion_status_invalid")
        if check["status"] != "unknown" and (
            not isinstance(check.get("finding"), str) or not check["finding"].strip()
            or not isinstance(check.get("evidence_role"), str)
            or check["evidence_role"] not in OBSERVATIONAL_ROLES
            or not isinstance(check.get("evidence_digest"), str)
            or check["evidence_digest"] not in inventory.get(check["evidence_role"], set())
        ):
            check = {"status": "unknown", "finding": "Criterion lacks a bound supporting evidence reference"}
        reviewed[name] = copy.deepcopy(check)
    gaps = [name for name, check in reviewed.items() if check["status"] == "unknown"]
    contradicted = normalized["outcome"] == "success" and any(
        check["status"] == "violated" for check in reviewed.values())
    outcome = "unclear" if gaps or contradicted else normalized["outcome"]
    observations = []
    for source in ("operator_mistakes", "data_issues"):
        rows = labels.get(source, [])
        if not isinstance(rows, list):
            raise ShadowError("video_explanation_taxonomy_rows_invalid")
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("category"), str):
                raise ShadowError("video_explanation_taxonomy_row_invalid")
            timestamp = row.get("t_s")
            if timestamp is not None and not 0 <= number(timestamp, "observation_time") <= prepared["model_input"]["duration_s"]:
                raise ShadowError("video_explanation_observation_time_invalid")
            if row.get("severity") not in {None, "low", "medium", "high"}:
                raise ShadowError("video_explanation_training_impact_invalid")
            observations.append({**copy.deepcopy(row), "source": source,
                                 "taxonomy": TAXONOMY.get(row["category"], "unclassified_observation"),
                                 "authority": "model_observation_only",
                                 "severity_semantics": "training_data_impact_not_safety"})
    annotations = _temporal_annotations(labels, prepared["model_input"]["duration_s"])
    narrative = {name: labels.get(name) for name in ("task_summary", "performance_review")}
    if any(value is not None and not isinstance(value, str)
           for value in (*narrative.values(), normalized["explanation"])):
        raise ShadowError("video_explanation_narrative_invalid")
    return seal({
        "schema_version": "argus_video_explanation_receipt.v1", "episode_id": prepared["episode_id"],
        "request_digest": prepared["request_digest"], "input_bundle_digest": prepared["input_bundle_digest"],
        "profile_digest": prepared["profile"]["profile_digest"], "argus_commit": ARGUS_COMMIT,
        "prompt_digest": prepared["prompt_digest"], "inference_identity": copy.deepcopy(identity),
        "raw_response": copy.deepcopy(record["raw_response"]),
        "apparent_outcome": {"success": "appears_complete", "failure": "appears_incomplete",
                             "unclear": "unclear"}[outcome],
        "original_argus_outcome": normalized["original_outcome"],
        "explanation": normalized["explanation"], "events": normalized["events"],
        "narrative": narrative, "temporal_annotations": annotations,
        "annotation_confidence": normalized["confidence"], "calibrated_success_probability": None,
        "criterion_evidence": reviewed, "missing_evidence_criteria": gaps,
        "completion_criterion_contradiction": contradicted, "observations": observations,
        "authoritative_result": copy.deepcopy(prepared["authoritative_result"]),
        "proof_boundary": {"advisory_only": True, "authoritative_task_success_unchanged": True,
                           "physical_validity_established": False, "severity_is_safety_rating": False,
                           "ranking_or_promotion_effect": "none", "real_corpus_admission": False},
        "response_origin": "retained_model_response_import", "model_calls": 0,
    }, "receipt_digest")
