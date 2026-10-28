"""Read-only causal acceptance over retained production receipts (AUT-21).

Beta journey: prove one supported job's authorized stages, separately from the
existing local-contract/deployment gates.  This module cannot launch, retry,
publish, delete, renew authority, or fetch production data.

The input is an index, not evidence authority.  In particular, job JSON cannot
declare its own producer, tenant, schema interpretation, or production origin.
An operator-owned acquisition adapter must supply ``TrustedObservation`` for
the exact bytes it independently acquired through an authenticated existing
surface.  Adapters must interpret their known producer schemas, never copy
job-controlled projection descriptors.  With no such adapter, evidence stays
``not_proven``.  The default release-report integration deliberately has none.

The first executable envelope is the existing rigid/DROID screening route.
Native plan, probe, session, independent scoring, media and scoped closeout
contracts are rechecked; screening cannot prove a finalist recommendation.
Service images and candidate worker images are distinct frozen identities.
The native episode's ``container_identity_digest`` currently identifies the
policy runtime; it is not an OCI image readback.  Only an authenticated
provider/container acquisition may supply a worker-image fact.  Missing image
provenance remains unknown.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

SCHEMA = "blueprint.production_autonomy_acceptance_manifest.v1"
REPORT_SCHEMA = "blueprint.production_autonomy_acceptance_report.v1"
ATTEMPT_SCHEMA = "blueprint.production_autonomy_attempt.v1"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_JSON_LIMIT = 16 * 1024 * 1024
SOURCE_KEYS = (
    "tenant_id", "site_id", "job_id", "capture_id", "task_id",
    "technical_brief_digest", "task_context_digest", "source_generation_digest",
    "scene_revision_digest", "scene_intent_id", "scene_intent_digest",
)
ROLES = (
    "service_identity_before", "upload", "task_confirmation", "authority",
    "intake_qa", "metric_scene", "robot_free_probe", "compiled_scene",
    "candidate_plan", "queued_ack", "publication", "publication_ack",
    "customer_readback", "service_identity_after", "event_history",
)
_PRODUCERS = {
    "service_identity_before": "operator", "service_identity_after": "operator",
    "upload": "webapp", "task_confirmation": "webapp", "authority": "webapp",
    "intake_qa": "pipeline", "metric_scene": "pipeline", "robot_free_probe": "pipeline",
    "compiled_scene": "pipeline", "candidate_plan": "pipeline", "queued_ack": "webapp",
    "publication": "pipeline", "publication_ack": "webapp", "customer_readback": "webapp",
    "event_history": "operator", "normal_retention": "operator", "attempt": "operator",
}
_UNPLANNED = frozenset({
    "manual_retry", "manual_stage_advance", "artifact_injection", "one_off_rescue", "deployment",
})
_AUTOMATIC = frozenset({
    "upload_start", "stage_progress", "automatic_retry", "process_restart",
    "delivery_outage", "delivery_recovered", "provider_outage",
    "capacity_wait", "capacity_recovered",
})
_AUTHORITY_ENDED = frozenset({
    "source_revocation", "consent_revocation", "run_cancellation", "technical_source_changed", "authority_expired",
})


@dataclass(frozen=True)
class TrustedObservation:
    """Caller capability from authenticated acquisition, never deserialized from a job.

    ``facts`` are the adapter's schema-specific interpretation of the retained
    bytes.  The adapter must bind producer, tenant, original source and complete
    event history using its authenticated acquisition, not an untrusted label.
    """

    artifact_sha256: str
    schema_version: str
    producer: str
    environment: str
    origin: str
    attempt_id: str
    event_id: str
    observed_at: str
    source: Mapping[str, Any]
    facts: Mapping[str, Any]


AcquisitionVerifier = Callable[[str, str, Mapping[str, Any]], TrustedObservation | None]


class AcceptanceEvidenceError(ValueError):
    """Sanitized, stable refusal; no source payload, path or credential is echoed."""


def _object(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AcceptanceEvidenceError("object_required")
    return dict(value)


def _stamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise AcceptanceEvidenceError("timestamp_invalid")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AcceptanceEvidenceError("timestamp_invalid") from exc
    if stamp.tzinfo is None:
        raise AcceptanceEvidenceError("timestamp_timezone_missing")
    return stamp.astimezone(timezone.utc)


def _safe_path(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value or ".." in Path(value).parts:
        raise AcceptanceEvidenceError("artifact_path_invalid")
    path = root / value
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise AcceptanceEvidenceError("artifact_path_unsafe")
    path = path.absolute()
    if not path.is_relative_to(root):
        raise AcceptanceEvidenceError("artifact_outside_evidence_root")
    return path


def _record(root: Path, value: Any, *, json_object: bool = True,
            allow_json_value: bool = False) -> tuple[Path, Any]:
    record = _object(value)
    path = _safe_path(root, record.get("path", record.get("relative_path")))
    if (type(record.get("size_bytes")) is not int or record["size_bytes"] <= 0
            or not isinstance(record.get("sha256"), str) or not _DIGEST.fullmatch(record["sha256"])):
        raise AcceptanceEvidenceError("artifact_record_invalid")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != record["size_bytes"]:
            raise AcceptanceEvidenceError("artifact_size_or_type_invalid")
        if json_object and metadata.st_size > _JSON_LIMIT:
            raise AcceptanceEvidenceError("artifact_json_too_large")
        digest, chunks, size = hashlib.sha256(), [], 0
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            if size > metadata.st_size:
                raise AcceptanceEvidenceError("artifact_changed")
            digest.update(chunk)
            if json_object:
                chunks.append(chunk)
        if size != metadata.st_size or "sha256:" + digest.hexdigest() != record["sha256"]:
            raise AcceptanceEvidenceError("artifact_digest_mismatch")
    if not json_object:
        return path, None
    try:
        payload = json.loads(b"".join(chunks), parse_constant=lambda _: None)
    except (ValueError, UnicodeError) as exc:
        raise AcceptanceEvidenceError("artifact_json_invalid") from exc
    return path, payload if allow_json_value else _object(payload)


def _observation(role: str, record: Mapping[str, Any], payload: Mapping[str, Any], *,
                 verifier: AcquisitionVerifier | None, attempt: Mapping[str, Any]) -> TrustedObservation:
    if verifier is None:
        raise AcceptanceEvidenceError("authenticated_acquisition_not_proven")
    observed = verifier(role, record["sha256"], payload)
    if (not isinstance(observed, TrustedObservation)
            or observed.artifact_sha256 != record["sha256"]
            or observed.schema_version != payload.get("schema_version")
            or observed.producer != _PRODUCERS.get(role, "pipeline")
            or observed.attempt_id != attempt.get("attempt_id")
            or not observed.event_id):
        raise AcceptanceEvidenceError("acquisition_binding_invalid")
    if observed.environment != "production" or observed.origin != "natural_production":
        raise AcceptanceEvidenceError("production_origin_not_proven")
    _stamp(observed.observed_at)
    source = _object(attempt.get("source"))
    keys = SOURCE_KEYS if role not in {"upload", "service_identity_before", "attempt"} else (
        "tenant_id", "site_id", "job_id", "capture_id", "source_generation_digest")
    if any(observed.source.get(key) != source.get(key) for key in keys):
        raise AcceptanceEvidenceError("source_binding_mismatch")
    return observed


def _episode_bytes(row: Mapping[str, Any], output_root: Path, cameras: list[str]) -> None:
    from .policy_canary_media_integrity import require_completed_episode_media

    episode = _object(row.get("episode"))
    artifacts = _object(row.get("evidence_artifacts"))
    for name in ("episode_receipt", "frame_manifest", "review_video", "reset_state",
                 "policy_query_receipt", "action_sequence", "action_delivery_readback",
                 "state_trace", "contact_force_trace", "task_object_trajectory", "score_receipt"):
        _record(output_root, artifacts.get(name), json_object=False)
    _, retained_episode = _record(output_root, artifacts["episode_receipt"])
    if retained_episode != episode:
        raise AcceptanceEvidenceError("episode_receipt_mismatch")
    for role, field in (("score_receipt", "score"), ("action_sequence", "commanded_actions"),
                        ("action_delivery_readback", "motion_evidence"), ("state_trace", "state_trace"),
                        ("contact_force_trace", "contact_force_evidence"),
                        ("task_object_trajectory", "task_object_trajectory")):
        _, retained = _record(output_root, artifacts[role], allow_json_value=True)
        if retained != episode.get(field):
            raise AcceptanceEvidenceError("episode_artifact_content_mismatch")
    for record in episode.get("media_artifacts") or []:
        _record(output_root / "episodes", record, json_object=False)
    for field, value in (
        ("lossless_frame_manifest_digest", episode.get("visual_evidence")),
        ("review_video_digest", episode.get("media_artifacts")),
        ("returned_action_sequence_digest", episode.get("commanded_actions")),
        ("action_delivery_readback_digest", episode.get("motion_evidence")),
        ("state_trace_digest", episode.get("state_trace")),
        ("contact_force_digest", episode.get("contact_force_evidence")),
        ("task_object_trajectory_digest", episode.get("task_object_trajectory")),
        ("deterministic_score_digest", episode.get("score")),
    ):
        if value is None or row.get(field) != canonical_digest({"value": value}):
            raise AcceptanceEvidenceError("episode_evidence_digest_mismatch")
    # Rehash every frame and calibration, not just the outer JSON inventory.
    require_completed_episode_media(output_root, episode, camera_ids=cameras)


def _execution(entry: Mapping[str, Any], *, root: Path, plan: Mapping[str, Any],
               attempt: Mapping[str, Any], verifier: AcquisitionVerifier | None) -> tuple[str, str, dict[str, Any]]:
    from . import beta_policy_screening_closeout as closeout
    from .beta_policy_screening_result import candidate_result_summary
    from .native_task_arena_policy_canary_session import (
        validate_provider_bundle, validate_runtime_input_manifest, validate_session_authority, validate_session_result,
    )

    candidate = next((row for row in plan["candidates"] if row["candidate_id"] == entry.get("candidate_id")), None)
    if candidate is None or candidate.get("episode_count") != 10:
        raise AcceptanceEvidenceError("execution_candidate_not_in_ten_cell_plan")
    if candidate["execution_spec"]["implementation_commit"] != attempt["releases"]["pipeline_commit"]:
        raise AcceptanceEvidenceError("execution_release_mismatch")
    values = {name: _record(root, entry.get(name))[1]
              for name in ("result", "authority", "runtime_inputs", "activation", "bundle_receipt", "prepared", "adapter", "closeout")}
    result = validate_session_result(values["result"])
    if result.get("status") != "completed_unqualified" or any(row.get("status") != "completed" for row in result["episodes"]):
        raise AcceptanceEvidenceError("execution_not_completed")
    candidate_result_summary(result, plan=plan, candidate=candidate)
    authority = validate_session_authority(values["authority"])
    runtime = validate_runtime_input_manifest(values["runtime_inputs"])
    activation = values["activation"]
    if (authority.get("execution_release") is not None
            or authority["runtime_inputs"]["sha256"] != entry["runtime_inputs"]["sha256"]
            or authority["runtime_inputs_digest"] != runtime["runtime_inputs_digest"]
            or runtime.get("beta_screening_admission") != result.get("beta_screening_admission")
            or authority["activation_manifest"]["sha256"] != entry["activation"]["sha256"]
            or activation.get("schema_version") != "task_evaluation_policy_campaign_activation.v1"
            or activation.get("activation_digest") != canonical_digest(activation, digest_field="activation_digest")
            or activation.get("activation_digest") != runtime.get("activation_digest")
            or activation.get("campaign_units") != runtime["cells"]
            or activation.get("candidate_ids") != result["candidate_ids"]):
        raise AcceptanceEvidenceError("execution_runtime_inputs_mismatch")
    expected_cells = [{key: row[key] for key in candidate["scenario_cells"][0]} for row in runtime["cells"]]
    if expected_cells != candidate["scenario_cells"]:
        raise AcceptanceEvidenceError("runtime_plan_cells_mismatch")
    for name in ("base_native_packet", "runtime_source", "construction_result"):
        record = entry.get("dependencies", {}).get(name, runtime[name])
        if record.get("sha256") != runtime[name]["sha256"] or record.get("size_bytes") != runtime[name]["size_bytes"]:
            raise AcceptanceEvidenceError("runtime_dependency_binding_mismatch")
        _record(root, record, json_object=False)
    bundle = validate_provider_bundle(values["bundle_receipt"], authority=authority)
    prepared, adapter, receipt = (values[key] for key in ("prepared", "adapter", "closeout"))
    if (prepared.get("prepared_digest") != canonical_digest(prepared, digest_field="prepared_digest")
            or prepared.get("authority", {}).get("sha256") != entry["authority"]["sha256"]
            or prepared.get("bundle_receipt", {}).get("sha256") != entry["bundle_receipt"]["sha256"]
            or authority.get("beta_screening_admission") != result.get("beta_screening_admission")
            or authority["candidate_ids"] != [candidate["candidate_id"]]
            or adapter.get("bundle_sha256") != bundle["bundle_sha256"]):
        raise AcceptanceEvidenceError("execution_authority_binding_mismatch")
    consumption = _object(adapter.get("authorization_consumption"))
    if (consumption.get("status") != "consumed" or consumption.get("maximum_provider_allocations") != 1
            or consumption.get("retry_cap") != 0 or consumption.get("authority_digest") != authority["authority_digest"]
            or consumption.get("consumption_digest") != canonical_digest(consumption, digest_field="consumption_digest")):
        raise AcceptanceEvidenceError("execution_consumption_unproven")
    if (receipt.get("schema_version") != closeout.SCHEMA or receipt.get("scope") != closeout.SCOPE
            or receipt.get("status") != "verified" or receipt.get("blockers") != []
            or receipt.get("provider_absence_confirmed") is not True or receipt.get("global_provider_zero_claimed") is not False
            or receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest")
            or receipt.get("authority_digest") != authority["authority_digest"]
            or receipt.get("resource_name") != authority["resource_name"]
            or receipt.get("prepared_digest") != prepared["prepared_digest"]
            or receipt.get("adapter_digest") != canonical_digest(adapter)
            or receipt.get("beta_screening_admission") != result.get("beta_screening_admission")
            or adapter.get("vast_instance_ids") != [int(receipt.get("instance_id"))]
            or result.get("session_closeout", {}).get("provider_absence_receipt_digest") != receipt["receipt_digest"]
            or closeout._observation_blockers(receipt, now=receipt["observed_at_epoch"])):
        raise AcceptanceEvidenceError("owned_resource_closeout_unproven")
    provider_closeout = _object(adapter.get("provider_closeout"))
    if (provider_closeout.get("all_staged_objects_absent") is not True
            or provider_closeout.get("warm_session_retained") is not False
            or adapter.get("continuing_spend_from_this_run") is not False):
        raise AcceptanceEvidenceError("staged_objects_or_teardown_not_proven")
    output_root = _safe_path(root, entry.get("output_root"))
    for row in result["episodes"]:
        _episode_bytes(row, output_root, attempt["envelope"]["camera_ids"])
    observations = {name: _observation("execution_" + name, entry[name], values[name],
                    verifier=verifier, attempt=attempt) for name in ("result", "authority", "closeout")}
    facts = dict(observations["result"].facts)
    facts["closeout_observed_at"] = observations["closeout"].observed_at
    return candidate["candidate_id"], result["result_digest"], facts


def build_production_autonomy_acceptance_report(*, manifest_path: str | Path,
        evidence_root: str | Path | None = None,
        acquisition_verifier: AcquisitionVerifier | None = None) -> dict[str, Any]:
    """Validate retained evidence only; default acquisition is deliberately unproven.

    No network, subprocess, mutation, live polling or automatic retry occurs.
    A caller-owned acquisition adapter is necessary for production milestones.
    It may not be loaded from, or named by, the job-controlled input manifest.
    """
    blockers: list[str] = []
    milestones = dict.fromkeys(("contract_checked", "service_identity_checked",
        "supported_production_completion", "unattended_completion_within_authorized_stages",
        "customer_readback", "resource_reconciliation", "normal_retention_observed"), False)
    report: dict[str, Any] = {"schema_version": REPORT_SCHEMA, "status": "not_proven",
        "blockers": blockers, "milestones": milestones,
        "owner_approved_finalist_recommendation": False,
        "proof_boundary": {"performs_live_actions": False, "grants_spending_authority": False,
            "physical_validation_proven": False, "concurrency_proven": False,
            "retained_evidence_only": True, "public_claim_upgrade_allowed": False}}
    try:
        path = Path(manifest_path).absolute()
        root = Path(evidence_root).absolute() if evidence_root is not None else path.parent
        if any(p.is_symlink() for p in (root, *root.parents, path, *path.parents)):
            raise AcceptanceEvidenceError("artifact_path_unsafe")
        raw = path.read_bytes()
        if len(raw) > _JSON_LIMIT:
            raise AcceptanceEvidenceError("manifest_too_large")
        manifest = _object(json.loads(raw))
        report["manifest_sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
        if manifest.get("schema_version") != SCHEMA:
            raise AcceptanceEvidenceError("manifest_schema_invalid")
        if "projection_descriptors" in manifest or "trusted_observations" in manifest:
            raise AcceptanceEvidenceError("job_controlled_acquisition_refused")
        _, attempt = _record(root, manifest.get("attempt"))
        if (attempt.get("schema_version") != ATTEMPT_SCHEMA or not attempt.get("attempt_id")
                or attempt.get("attempt_digest") != canonical_digest(attempt, digest_field="attempt_digest")):
            raise AcceptanceEvidenceError("frozen_attempt_invalid")
        report["attempt_id"] = attempt["attempt_id"]
        source, envelope, releases = (_object(attempt.get(key)) for key in ("source", "envelope", "releases"))
        if any(not isinstance(source.get(key), str) or not source[key] for key in SOURCE_KEYS):
            raise AcceptanceEvidenceError("frozen_source_incomplete")
        if any(not _DIGEST.fullmatch(source[key]) for key in SOURCE_KEYS if key.endswith("digest")):
            raise AcceptanceEvidenceError("frozen_source_digest_invalid")
        if (envelope.get("task_family") != "rigid_pick_place" or envelope.get("backend") != "robolab.v1"
                or envelope.get("robot_id") != "franka" or envelope.get("policy_interface") != "droid"
                or envelope.get("camera_ids") != ["external", "wrist", "overview"]):
            raise AcceptanceEvidenceError("acceptance_envelope_not_implemented")
        for name in ("pipeline", "webapp"):
            if (not _COMMIT.fullmatch(str(releases.get(name + "_commit") or ""))
                    or not _DIGEST.fullmatch(str(releases.get(name + "_image_digest") or ""))):
                raise AcceptanceEvidenceError("frozen_release_identity_invalid")
        frozen_at = _stamp(attempt.get("frozen_at"))
        boundary_kinds = attempt.get("declared_human_boundaries")
        if not isinstance(boundary_kinds, list) or set(boundary_kinds) - {"customer_fact_confirmation", "site_finalist_approval"}:
            raise AcceptanceEvidenceError("declared_boundaries_invalid")
        evidence, observations = {}, {}
        for role in ("attempt", *ROLES):
            try:
                record = manifest["attempt"] if role == "attempt" else manifest["receipts"][role]
                _, payload = _record(root, record)
                evidence[role] = payload
                observations[role] = _observation(role, record, payload, verifier=acquisition_verifier, attempt=attempt)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                code = str(exc) if isinstance(exc, AcceptanceEvidenceError) else "receipt_missing_or_invalid"
                blockers.append(role + ":" + code)
        # Canonical native contracts are checked even without acquisition authority.
        from .beta_policy_screening_plan import validate_screening_plan
        from .robot_free_probe_gate import require_robot_free_probe
        plan = validate_screening_plan(evidence["candidate_plan"])
        if (any(plan.get(key) != source[key] for key in ("site_id", "capture_id", "task_id", "scene_revision_digest"))
                or any(plan["source"].get(key) != source[key] for key in plan["source"])):
            raise AcceptanceEvidenceError("plan_source_mismatch")
        compiled = evidence["compiled_scene"]
        if (plan["scene_inputs"]["sha256"] != manifest["receipts"]["compiled_scene"]["sha256"]
                or compiled.get("scene_revision_digest") != source["scene_revision_digest"]
                or compiled.get("task_success_contract_digest") != plan["task_success_contract_digest"]):
            raise AcceptanceEvidenceError("compiled_scene_plan_binding_mismatch")
        probe = require_robot_free_probe(evidence["robot_free_probe"], scene_revision_digest=source["scene_revision_digest"],
            task_kind=envelope["task_family"], task_success_contract_digest=plan["task_success_contract_digest"])
        if probe["receipt_digest"] != plan["robot_free_probe_digest"]:
            raise AcceptanceEvidenceError("probe_plan_mismatch")
        publication = evidence["publication"]
        if (publication.get("schema_version") != "beta_policy_screening_publication.v1"
                or publication.get("publication_digest") != cross_runtime_canonical_digest(publication, digest_field="publication_digest")
                or publication.get("plan_digest") != plan["plan_digest"] or publication.get("source") != plan["source"]
                or publication.get("site_id") != plan["site_id"] or publication.get("status") != "completed"
                or publication.get("purpose") != "screening" or publication.get("claim_ceiling") != "diagnostic_policy_execution"
                or publication.get("execution_operator") != {"kind": "blueprint"}
                or publication.get("recommendation_eligible") is not False):
            raise AcceptanceEvidenceError("publication_binding_invalid")
        executions = manifest.get("executions")
        if not isinstance(executions, list) or not executions or len(executions) > 32:
            raise AcceptanceEvidenceError("execution_receipts_missing")
        results: dict[str, str] = {}
        execution_facts: dict[str, dict[str, Any]] = {}
        for entry in executions:
            candidate_id, digest, facts = _execution(_object(entry), root=root, plan=plan, attempt=attempt, verifier=acquisition_verifier)
            if candidate_id in results:
                raise AcceptanceEvidenceError("duplicate_execution_start")
            results[candidate_id] = digest
            execution_facts[candidate_id] = facts
        if set(results) != {row["candidate_id"] for row in plan["candidates"] if row["status"] == "selected"}:
            raise AcceptanceEvidenceError("selected_candidate_coverage_incomplete")
        published = publication.get("candidates")
        if (not isinstance(published, list) or len(published) != len(plan["candidates"])
                or {row["candidate_id"] for row in published} != {row["candidate_id"] for row in plan["candidates"]}):
            raise AcceptanceEvidenceError("publication_candidate_coverage_incomplete")
        for row in published:
            if row.get("candidate_id") in results and (row.get("session_result_digest") != results[row["candidate_id"]]
                    or row.get("status") != "completed" or row.get("episodes_planned") != 10
                    or row.get("episodes_scored") != 10):
                raise AcceptanceEvidenceError("publication_result_mismatch")
        if not blockers:
            _join_observations(observations, attempt=attempt, plan=plan, publication=publication, results=results,
                               frozen_at=frozen_at, execution_facts=execution_facts, blockers=blockers, report=report)
            milestones["contract_checked"] = not blockers
        if "normal_retention" in manifest.get("receipts", {}):
            _, value = _record(root, manifest["receipts"]["normal_retention"])
            retained = _observation("normal_retention", manifest["receipts"]["normal_retention"], value,
                                    verifier=acquisition_verifier, attempt=attempt)
            milestones["normal_retention_observed"] = (retained.facts.get("normal_lifecycle_observed") is True
                and retained.facts.get("manual_cleanup_performed") is False)
        if not blockers:
            report["status"] = "verified"
    except (OSError, ValueError, TypeError, KeyError, StopIteration, InvalidOperation) as exc:
        blockers.append(str(exc) if isinstance(exc, AcceptanceEvidenceError) else "retained_evidence_invalid")
    report["blockers"] = sorted(set(blockers))
    return report


def _join_observations(observed: Mapping[str, TrustedObservation], *, attempt: Mapping[str, Any],
        plan: Mapping[str, Any], publication: Mapping[str, Any], results: Mapping[str, str],
        frozen_at: datetime, execution_facts: Mapping[str, Mapping[str, Any]],
        blockers: list[str], report: dict[str, Any]) -> None:
    facts = {role: dict(row.facts) for role, row in observed.items()}
    stamps = {role: _stamp(row.observed_at) for role, row in observed.items()}
    chain = ["service_identity_before", "upload", "task_confirmation", "authority", "intake_qa", "metric_scene",
             "robot_free_probe", "compiled_scene", "candidate_plan", "queued_ack", "publication",
             "publication_ack", "customer_readback", "service_identity_after", "event_history"]
    if not frozen_at <= min(stamps.values()) or any(stamps[a] > stamps[b] for a, b in zip(chain, chain[1:])):
        blockers.append("causal_timestamp_order_invalid")
    for facts_for_execution in execution_facts.values():
        started, finished, closed = (_stamp(facts_for_execution.get(key)) for key in
                                    ("started_at", "completed_at", "closeout_observed_at"))
        if not stamps["queued_ack"] <= started <= finished <= stamps["publication"] or not finished <= closed <= stamps["event_history"]:
            blockers.append("execution_started_before_current_ack_or_after_publication")
    expected_runtimes = attempt.get("candidate_runtime_identities")
    if not isinstance(expected_runtimes, Mapping):
        expected_runtimes = {}
    candidates = {row["candidate_id"]: row for row in plan["candidates"]}
    for candidate_id, facts_for_execution in execution_facts.items():
        expected = expected_runtimes.get(candidate_id)
        expected = dict(expected) if isinstance(expected, Mapping) else {}
        configured = candidates[candidate_id]
        worker_image = expected.get("worker_image_digest")
        if (expected.get("configuration_digest") != configured["configuration_digest"]
                or expected.get("runtime_identity_digest") != configured["execution_spec"]["runtime_digest"]
                or not isinstance(worker_image, str) or not _DIGEST.fullmatch(worker_image)):
            blockers.append("candidate_worker_image_expected_not_proven")
        observed_image = facts_for_execution.get("worker_image_digest")
        if (facts_for_execution.get("worker_image_binding_kind") != "provider_container_image_digest"
                or not isinstance(observed_image, str) or not _DIGEST.fullmatch(observed_image)):
            blockers.append("worker_image_acquisition_not_proven")
        elif worker_image is not None and observed_image != worker_image:
            blockers.append("execution_worker_image_mismatch")
    if observed["upload"].event_id != attempt.get("start_event_id") or facts["upload"].get("natural_upload") is not True:
        blockers.append("natural_start_not_proven")
    for role in ("service_identity_before", "service_identity_after"):
        if any(facts[role].get(key) != value for key, value in attempt["releases"].items()):
            blockers.append("release_changed_during_attempt")
    authority = facts["authority"]
    if (authority.get("current_consent") is not True or authority.get("rights_current") is not True
            or not authority.get("grant_id") or not authority.get("budget_ledger_id")
            or _stamp(authority.get("expires_at")) < stamps["publication"]):
        blockers.append("current_authority_not_proven")
    cap = Decimal(str(authority.get("cap_usd")))
    if not cap.is_finite() or not Decimal("0") < cap <= Decimal("15") or Decimal(str(plan["maximum_planned_cost_usd"])) > cap:
        blockers.append("screening_budget_binding_invalid")
    for role in ("task_confirmation", "intake_qa", "metric_scene", "compiled_scene"):
        if facts[role].get("completed") is not True:
            blockers.append(role + ":completion_not_proven")
    if facts["metric_scene"].get("metric_evidence_qualified") is not True:
        blockers.append("metric_scene_qualification_not_proven")
    if (facts["queued_ack"].get("plan_digest") != plan["plan_digest"]
            or facts["queued_ack"].get("current_source") is not True):
        blockers.append("queued_ack_stale_or_mismatched")
    for role in ("publication_ack", "customer_readback"):
        value = facts[role]
        if (value.get("publication_digest") != publication["publication_digest"]
                or value.get("revision") != publication.get("revision") or value.get("current_source") is not True
                or value.get("persisted") is not True):
            blockers.append(role + ":stale_or_unpersisted")
    readback = facts["customer_readback"]
    if (readback.get("displayed") is not True or readback.get("result_digests") != dict(results)
            or readback.get("webapp_commit") != attempt["releases"]["webapp_commit"]):
        blockers.append("current_customer_display_not_proven")
    history = facts["event_history"]
    events = history.get("events")
    if (history.get("coverage_complete") is not True or not isinstance(events, list)
            or _stamp(history.get("coverage_started_at")) > frozen_at
            or _stamp(history.get("coverage_ended_at")) < stamps["service_identity_after"]):
        blockers.append("complete_intervention_history_not_proven")
        events = []
    starts, retries, restarts, interventions = 0, 0, 0, 0
    for event in events:
        event = _object(event)
        kind = event.get("kind")
        stamp = _stamp(event.get("at"))
        if not frozen_at <= stamp <= stamps["event_history"] or event.get("attempt_id") != attempt["attempt_id"]:
            blockers.append("event_history_attempt_or_time_mismatch")
        starts += kind == "upload_start"
        retries += kind == "automatic_retry"
        restarts += kind == "process_restart"
        if kind in _AUTHORITY_ENDED:
            # Reauthorization cannot retrospectively repair this frozen attempt.
            # Its stopped/partial evidence remains useful, but is no completion pass.
            blockers.append("authority_ended_during_frozen_attempt")
        elif kind in _UNPLANNED:
            interventions += 1
        elif kind not in _AUTOMATIC and kind not in attempt["declared_human_boundaries"]:
            blockers.append("event_intervention_classification_unknown")
    if starts != 1:
        blockers.append("duplicate_or_missing_start_event")
    milestones = report["milestones"]
    if not blockers:
        for name in ("service_identity_checked", "supported_production_completion", "customer_readback", "resource_reconciliation"):
            milestones[name] = True
        milestones["unattended_completion_within_authorized_stages"] = interventions == 0
    if interventions:
        blockers.append("unplanned_intervention_invalidates_unattended_claim")
    report.update(retry_count=retries, restart_count=restarts, unplanned_intervention_count=interventions,
                  total_latency_seconds=(stamps["customer_readback"] - stamps["upload"]).total_seconds(),
                  stage_latency_seconds={b: (stamps[b] - stamps[a]).total_seconds() for a, b in zip(chain, chain[1:])},
                  costs=history.get("costs", {"status": "unknown"}),
                  storage=history.get("storage", {"status": "not_observed"}))
