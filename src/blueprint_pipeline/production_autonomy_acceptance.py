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

The first executable envelope is the existing rigid/DROID bounded evaluation.
Native plan, probe, session, independent scoring, media and scoped closeout
contracts are rechecked. Current v2 job estimates and retained v1 publications
keep their exact wire contracts; there is no separate finalist stage. Native
session diagnostic ceilings continue to exclude physical and ADP validation.
Service images and candidate worker images are distinct frozen identities.
The native episode's ``container_identity_digest`` currently identifies the
policy runtime; it is not an OCI image readback.  Only an authenticated
provider/container acquisition may supply a worker-image fact.  Missing image
provenance remains unknown.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
import zipfile

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

SCHEMA = "blueprint.production_autonomy_acceptance_manifest.v1"
REPORT_SCHEMA = "blueprint.production_autonomy_acceptance_report.v1"
ATTEMPT_SCHEMA = "blueprint.production_autonomy_attempt.v1"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_JSON_LIMIT = 16 * 1024 * 1024
_PUBLICATION_FIELDS = frozenset({
    "schema_version", "campaign_id", "site_id", "plan_digest", "catalog_digest", "scenario_plan_digest",
    "source", "revision", "status", "purpose", "execution_operator", "claim_ceiling", "candidates", "budget",
    "generated_at", "publication_digest",
})
_PUBLICATION_VARIANTS = {
    "beta_policy_screening_publication.v1": ("screening", "diagnostic_policy_execution"),
    "beta_policy_screening_publication.v2": ("job_evaluation", "simulation_estimate"),
}
_CANDIDATE_FIELDS = frozenset({
    "candidate_id", "configuration_digest", "display_name", "status", "blockers", "episodes_planned",
    "episodes_scored", "episodes_succeeded", "episodes_failed", "episodes_void", "success_rate", "session_result_digest",
})
_WIRE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}\Z")
_MONEY = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]{1,6})?\Z")
SOURCE_KEYS = (
    "tenant_id", "site_id", "job_id", "capture_id", "task_id",
    "technical_brief_digest", "task_context_digest", "source_generation_digest",
    "scene_revision_digest", "scene_intent_id", "scene_intent_digest",
)
ROLES = (
    "service_identity_before", "upload", "task_confirmation", "authority",
    "intake_qa", "metric_scene", "robot_free_probe", "compiled_scene",
    "candidate_plan", "queued_ack", "publication", "site_budget", "publication_ack",
    "customer_readback", "service_identity_after", "event_history",
)
_PRODUCERS = {
    "service_identity_before": "operator", "service_identity_after": "operator",
    "upload": "webapp", "task_confirmation": "webapp", "authority": "webapp",
    "intake_qa": "pipeline", "metric_scene": "pipeline", "robot_free_probe": "pipeline",
    "compiled_scene": "pipeline", "candidate_plan": "pipeline", "queued_ack": "webapp",
    "publication": "pipeline", "publication_ack": "webapp", "customer_readback": "webapp",
    "site_budget": "pipeline",
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
    schema_version: str | None  # Some native HTTP readbacks have no schema label.
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


def _epoch_stamp(value: Any) -> datetime:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise AcceptanceEvidenceError("provider_observation_timestamp_invalid")
    try:
        return datetime.fromtimestamp(value, tz=timezone.utc)
    except (ValueError, OSError, OverflowError) as exc:
        raise AcceptanceEvidenceError("provider_observation_timestamp_invalid") from exc


def _publication_contract(value: Mapping[str, Any]) -> dict[str, str]:
    """Existing exact WebApp v1/v2 wire variants, with no tier conversion.

    Mirrors betaPolicyScreeningContract's strict variants and count/budget
    invariants. Native runtime contracts are validated separately; accepting a
    job-facing simulation_estimate never changes their diagnostic ceiling.
    """
    from .beta_policy_screening_result import CANDIDATE_STATUSES, SOURCE_FIELDS, campaign_status

    schema = value.get("schema_version")
    variant = _PUBLICATION_VARIANTS.get(schema) if isinstance(schema, str) else None
    fields = _PUBLICATION_FIELDS | ({"recommendation_eligible"} if schema == "beta_policy_screening_publication.v1" else set())
    if (variant is None or set(value) != fields
            or (value.get("purpose"), value.get("claim_ceiling")) != variant
            or (schema.endswith(".v1") and value.get("recommendation_eligible") is not False)
            or value.get("execution_operator") != {"kind": "blueprint"}
            or type(value.get("revision")) is not int or not 1 <= value["revision"] <= 9_007_199_254_740_991
            or any(not isinstance(value.get(key), str) or not _WIRE_ID.fullmatch(value[key]) for key in ("campaign_id", "site_id"))
            or any(not isinstance(value.get(key), str) or not _DIGEST.fullmatch(value[key]) for key in (
                "plan_digest", "catalog_digest", "scenario_plan_digest", "publication_digest"))
            or value["publication_digest"] != cross_runtime_canonical_digest(value, digest_field="publication_digest")):
        raise AcceptanceEvidenceError("publication_contract_invalid")
    source = _object(value.get("source"))
    if set(source) != set(SOURCE_FIELDS) or any(not isinstance(item, str)
            or not (_DIGEST if key.endswith("digest") else _WIRE_ID).fullmatch(item) for key, item in source.items()):
        raise AcceptanceEvidenceError("publication_contract_invalid")
    generated = value.get("generated_at")
    if not isinstance(generated, str) or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})", generated):
        raise AcceptanceEvidenceError("publication_contract_invalid")
    _stamp(generated)
    budget = _object(value.get("budget"))
    amounts = ("limit_usd", "committed_upper_bound_usd", "spent_upper_bound_usd", "reserved_usd")
    if (set(budget) != {"currency", *amounts} or budget["currency"] != "USD"
            or any(not isinstance(budget.get(key), str) or len(budget[key]) > 32
                or not _MONEY.fullmatch(budget[key]) for key in amounts)):
        raise AcceptanceEvidenceError("publication_contract_invalid")
    limit, committed, spent, reserved = (Decimal(budget[key]) for key in amounts)
    if limit > 15 or spent + reserved != committed or committed > limit:
        raise AcceptanceEvidenceError("publication_contract_invalid")
    rows = value.get("candidates")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 256:
        raise AcceptanceEvidenceError("publication_contract_invalid")
    ids = set()
    for row in rows:
        row = _object(row)
        if (set(row) != _CANDIDATE_FIELDS or not isinstance(row.get("candidate_id"), str)
                or not _WIRE_ID.fullmatch(row["candidate_id"]) or row["candidate_id"] in ids
                or not isinstance(row.get("configuration_digest"), str) or not _DIGEST.fullmatch(row["configuration_digest"])
                or not isinstance(row.get("status"), str) or row["status"] not in CANDIDATE_STATUSES
                or not isinstance(row.get("display_name"), str)
                or not 1 <= len(row["display_name"]) <= 200 or row["display_name"].strip() != row["display_name"]
                or not isinstance(row.get("blockers"), list) or len(row["blockers"]) > 100
                or any(not isinstance(item, str) or not 1 <= len(item) <= 300 for item in row["blockers"])
                or any(type(row.get(key)) is not int or not 0 <= row[key] <= 1_000_000 for key in (
                    "episodes_planned", "episodes_scored", "episodes_succeeded", "episodes_failed", "episodes_void"))):
            raise AcceptanceEvidenceError("publication_contract_invalid")
        ids.add(row["candidate_id"])
        planned, scored, succeeded, failed, void = (row[key] for key in (
            "episodes_planned", "episodes_scored", "episodes_succeeded", "episodes_failed", "episodes_void"))
        rate, digest = row["success_rate"], row["session_result_digest"]
        if (succeeded + failed != scored or scored + void > planned
                or (rate is not None and (type(rate) not in (int, float) or not math.isfinite(rate)))
                or rate != (succeeded / scored if scored else None)
                or (digest is not None and (not isinstance(digest, str) or not _DIGEST.fullmatch(digest)))
                or (scored > 0 and digest is None)
                or (row["status"] == "completed" and (digest is None or planned == 0 or scored + void != planned))
                or (row["status"] != "completed" and (scored != 0 or void != 0 or digest is not None))):
            raise AcceptanceEvidenceError("publication_contract_invalid")
    if value.get("status") != campaign_status({row["candidate_id"]: row for row in rows}):
        raise AcceptanceEvidenceError("publication_contract_invalid")
    return {"schema_version": schema, "purpose": variant[0], "claim_ceiling": variant[1]}


def _retained_reader(root: Path, entry: Mapping[str, Any]) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Locate copies by original digest; aliases cannot change expected bytes."""
    aliases = {}
    def visit(value):
        if not isinstance(value, Mapping):
            return
        if "sha256" in value and ("path" in value or "relative_path" in value):
            aliases[value["sha256"]] = value
        else:
            for child in value.values():
                visit(child)
    visit(entry)
    def read(original):
        original = _object(original)
        located = aliases.get(original.get("sha256"), original)
        if located.get("sha256") != original.get("sha256") or located.get("size_bytes") != original.get("size_bytes"):
            raise AcceptanceEvidenceError("retained_nested_record_binding_mismatch")
        return _record(root, located)[1]
    return read


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
    try:
        observed = verifier(role, record["sha256"], payload)
    except Exception as exc:
        raise AcceptanceEvidenceError("authenticated_acquisition_unavailable") from exc
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


def _episode_bytes(row: Mapping[str, Any], output_root: Path, cameras: list[str], *, max_queries: int) -> None:
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
    queries = episode.get("queries")
    query_count = episode.get("policy_queries")
    _, query_receipt = _record(output_root, artifacts["policy_query_receipt"])
    if (type(query_count) is not int or not 1 <= query_count <= max_queries
            or not isinstance(queries, list) or len(queries) != query_count
            or query_receipt.get("policy_queries") != query_count or query_receipt.get("queries") != queries):
        raise AcceptanceEvidenceError("issued_policy_query_bound_exceeded_or_unproven")
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


def _sealed_execution_spec(entry: Mapping[str, Any], *, root: Path, candidate: Mapping[str, Any],
        attempt: Mapping[str, Any], authority: Mapping[str, Any], runtime: Mapping[str, Any],
        bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Read pinned JSON inside retained ZIP bytes, never import/execute the ZIP."""
    from .beta_robolab_policy_execution import validate_robolab_policy_execution_spec
    from .beta_policy_screening_hosted import request_limit

    spec = validate_robolab_policy_execution_spec(
        _retained_reader(root, entry)(candidate["execution_spec"]["policy_execution_spec"]),
        candidate_id=candidate["candidate_id"])
    frozen = _object(attempt.get("candidate_runtime_identities"))
    expected = _object(frozen.get(candidate["candidate_id"]))
    if (spec["runtime_identity_digest"] != candidate["execution_spec"]["runtime_digest"]
            or spec["task_success_contract"] != runtime["task_success_contract"]
            or any(expected.get(key) != spec[key] for key in
                   ("max_policy_queries", "open_loop_horizon"))
            or expected.get("policy_execution_spec_digest") != spec["execution_spec_digest"]):
        raise AcceptanceEvidenceError("issued_policy_query_or_runtime_bounds_mismatch")
    settings = candidate["execution_spec"]
    if "hosted_http_request_limit" in settings and settings["hosted_http_request_limit"] != request_limit(spec, candidate["episode_count"]):
        raise AcceptanceEvidenceError("issued_hosted_http_bound_mismatch")
    archive_record = entry.get("bundle_archive", {"path": bundle["bundle_path"],
        "sha256": bundle["bundle_sha256"], "size_bytes": bundle["bundle_size_bytes"]})
    if (archive_record.get("sha256") != bundle["bundle_sha256"]
            or archive_record.get("size_bytes") != bundle["bundle_size_bytes"]):
        raise AcceptanceEvidenceError("issued_bundle_bytes_mismatch")
    path, _ = _record(root, archive_record, json_object=False)
    expected_manifest = {key: value for key, value in bundle.items()
                         if key not in {"bundle_path", "bundle_sha256", "bundle_size_bytes"}}
    expected_files = {
        "provider_runtime/adp_arena_provider_manifest.json": expected_manifest,
        "provider_runtime/runtime_inputs/policy_canary_session_authority.json": dict(authority),
        "provider_runtime/runtime_inputs/policy_canary_runtime_inputs.json": dict(runtime),
        f"provider_runtime/runtime_inputs/policy_execution_spec.{candidate['candidate_id']}.json": spec,
    }
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            for name, expected_payload in expected_files.items():
                if names.count(name) != 1 or archive.getinfo(name).file_size > _JSON_LIMIT:
                    raise AcceptanceEvidenceError("issued_bundle_member_missing_or_ambiguous")
                if json.loads(archive.read(name)) != expected_payload:
                    raise AcceptanceEvidenceError("issued_bundle_member_mismatch")
    except (OSError, ValueError, KeyError, RuntimeError, NotImplementedError, EOFError, zipfile.BadZipFile) as exc:
        if isinstance(exc, AcceptanceEvidenceError):
            raise
        raise AcceptanceEvidenceError("issued_bundle_invalid") from exc
    if (expected_manifest.get("input_digest") != canonical_digest(expected_manifest, digest_field="input_digest")
            or expected_manifest.get("execution_spec_digests") != {candidate["candidate_id"]: spec["execution_spec_digest"]}):
        raise AcceptanceEvidenceError("issued_bundle_spec_binding_mismatch")
    return spec


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
    summary = candidate_result_summary(result, plan=plan, candidate=candidate)
    authority = validate_session_authority(values["authority"])
    settings = candidate["execution_spec"]
    provider_cap = Decimal(str(settings.get("provider_hard_cap_usd", candidate["max_cost_usd"])))
    if (Decimal(str(authority["hard_cap_usd"])) != provider_cap
            or authority["hard_ttl_seconds"] != settings["hard_ttl_seconds"]
            or provider_cap > Decimal(str(candidate["max_cost_usd"]))):
        raise AcceptanceEvidenceError("issued_session_bounds_mismatch")
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
    if (not closeout.retained_scoped_closeout_verified(receipt, adapter=adapter, prepared=prepared,
                adapter_path=Path(entry.get("adapter_source_path", entry["adapter"]["path"])),
                record_reader=_retained_reader(root, entry), observed_at_epoch=receipt.get("observed_at_epoch"))
            or result.get("session_closeout", {}).get("provider_absence_receipt_digest") != receipt["receipt_digest"]):
        raise AcceptanceEvidenceError("owned_resource_closeout_unproven")
    observations = {name: _observation("execution_" + name, entry[name], values[name],
                    verifier=verifier, attempt=attempt) for name in ("result", "authority", "closeout")}
    spec = _sealed_execution_spec(entry, root=root, candidate=candidate, attempt=attempt,
                                 authority=authority, runtime=runtime, bundle=bundle)
    output_root = _safe_path(root, entry.get("output_root"))
    for row in result["episodes"]:
        if row.get("checkpoint_digest") != spec["checkpoint_digest"]:
            raise AcceptanceEvidenceError("issued_checkpoint_binding_mismatch")
        _episode_bytes(row, output_root, attempt["envelope"]["camera_ids"], max_queries=spec["max_policy_queries"])
    facts = dict(observations["result"].facts)
    facts["native_result_summary"] = summary
    facts["closeout_acquired_at"] = observations["closeout"].observed_at
    facts["closeout_collection_started_at"] = _epoch_stamp(receipt.get("collection_started_at_epoch")).isoformat()
    facts["closeout_observed_at"] = _epoch_stamp(receipt.get("observed_at_epoch")).isoformat()
    facts["issued_provider_cap_usd"] = str(provider_cap)
    facts["issued_total_cap_usd"] = str(provider_cap + Decimal(str(settings.get("hosted_inference_maximum_usd", "0"))))
    facts["issued_ttl_seconds"] = authority["hard_ttl_seconds"]
    facts["issued_deadline_epoch"] = authority.get("hard_deadline_epoch")
    facts["session_authority_digest"] = authority["authority_digest"]
    facts["prepared_digest"] = prepared["prepared_digest"]
    return candidate["candidate_id"], result["result_digest"], facts


def _site_budget_binding(ledger: Mapping[str, Any], *, plan: Mapping[str, Any],
                         executions: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    from .beta_policy_screening_coordinator import LEDGER_SCHEMA, _budget

    if (ledger.get("schema_version") != LEDGER_SCHEMA
            or ledger.get("ledger_digest") != canonical_digest(ledger, digest_field="ledger_digest")):
        raise AcceptanceEvidenceError("site_budget_ledger_unproven")
    state = ledger.get("campaigns", {}).get(plan["campaign_id"], {})
    if state.get("site_id") != plan["site_id"] or state.get("plan_digest") != plan["plan_digest"]:
        raise AcceptanceEvidenceError("site_budget_source_mismatch")
    budget = _budget(ledger, plan["site_id"])
    campaign_budget = _budget(ledger, plan["site_id"], campaign_id=plan["campaign_id"])
    if (Decimal(budget["committed_upper_bound_usd"]) > Decimal(budget["limit_usd"])
            or Decimal(campaign_budget["committed_upper_bound_usd"]) > Decimal(plan["budget_usd"])):
        raise AcceptanceEvidenceError("site_budget_commitment_exceeded")
    by_candidate = {row["candidate_id"]: row for row in plan["candidates"]}
    for candidate_id, execution in executions.items():
        row = state.get("candidates", {}).get(candidate_id, {})
        attempts = [*row.get("attempt_history", []), *([row["attempt"]] if row.get("attempt") else [])]
        matched = [item for item in attempts if item.get("prepared", {}).get("prepared_digest") == execution["prepared_digest"]]
        if len(matched) != 1:
            raise AcceptanceEvidenceError("site_budget_attempt_binding_unproven")
        funded = matched[0]
        cap = Decimal(str(funded.get("maximum_cost_usd")))
        if (funded.get("reservation_status") != "committed" or funded.get("provider_allocation_observed") is not True
                or cap != Decimal(str(by_candidate[candidate_id]["max_cost_usd"]))
                or Decimal(execution["issued_total_cap_usd"]) > cap):
            raise AcceptanceEvidenceError("issued_session_site_reservation_mismatch")
        execution["funded_launch_intent_at"] = funded.get("launch_intent_at")
    return budget


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
        "evaluation_path": "single_bounded_evaluation", "publication_contract": None,
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
        if not isinstance(boundary_kinds, list) or set(boundary_kinds) - {"customer_fact_confirmation"}:
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
        report["publication_contract"] = _publication_contract(publication)
        if (publication.get("campaign_id") != plan["campaign_id"]
                or publication.get("plan_digest") != plan["plan_digest"] or publication.get("source") != plan["source"]
                or publication.get("site_id") != plan["site_id"] or publication.get("status") != "completed"
                or publication.get("catalog_digest") != plan["catalog_digest"]
                or publication.get("scenario_plan_digest") != plan["scenario_plan_digest"]):
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
        budget = _site_budget_binding(evidence["site_budget"], plan=plan, executions=execution_facts)
        published = publication.get("candidates")
        if (not isinstance(published, list) or len(published) != len(plan["candidates"])
                or {row["candidate_id"] for row in published} != {row["candidate_id"] for row in plan["candidates"]}):
            raise AcceptanceEvidenceError("publication_candidate_coverage_incomplete")
        configured = {row["candidate_id"]: row for row in plan["candidates"]}
        for row in published:
            candidate = configured[row["candidate_id"]]
            if (row["configuration_digest"] != candidate["configuration_digest"]
                    or row["episodes_planned"] != candidate.get("episode_count", 0)):
                raise AcceptanceEvidenceError("publication_candidate_binding_mismatch")
            if row.get("candidate_id") in results and (row.get("session_result_digest") != results[row["candidate_id"]]
                    or row.get("status") != "completed" or row.get("episodes_planned") != 10
                    or row.get("episodes_scored") != 10
                    or any(row.get(key) != execution_facts[row["candidate_id"]]["native_result_summary"][key]
                        for key in ("episodes_succeeded", "episodes_failed", "episodes_void", "success_rate"))):
                raise AcceptanceEvidenceError("publication_result_mismatch")
        if not blockers:
            _join_observations(observations, attempt=attempt, plan=plan, publication=publication, results=results,
                               frozen_at=frozen_at, execution_facts=execution_facts, site_budget=budget,
                               blockers=blockers, report=report)
            milestones["contract_checked"] = not blockers
        if "normal_retention" in manifest.get("receipts", {}):
            _, value = _record(root, manifest["receipts"]["normal_retention"])
            retained = _observation("normal_retention", manifest["receipts"]["normal_retention"], value,
                                    verifier=acquisition_verifier, attempt=attempt)
            milestones["normal_retention_observed"] = (retained.facts.get("normal_lifecycle_observed") is True
                and retained.facts.get("manual_cleanup_performed") is False)
        if not blockers:
            report["status"] = "verified"
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError, StopIteration, InvalidOperation) as exc:
        blockers.append(str(exc) if isinstance(exc, AcceptanceEvidenceError) else "retained_evidence_invalid")
    report["blockers"] = sorted(set(blockers))
    return report


def _join_observations(observed: Mapping[str, TrustedObservation], *, attempt: Mapping[str, Any],
        plan: Mapping[str, Any], publication: Mapping[str, Any], results: Mapping[str, str],
        frozen_at: datetime, execution_facts: Mapping[str, Mapping[str, Any]],
        site_budget: Mapping[str, Any],
        blockers: list[str], report: dict[str, Any]) -> None:
    facts = {role: dict(row.facts) for role, row in observed.items()}
    stamps = {role: _stamp(row.observed_at) for role, row in observed.items()}
    chain = ["service_identity_before", "upload", "task_confirmation", "authority", "intake_qa", "metric_scene",
             "robot_free_probe", "compiled_scene", "candidate_plan", "queued_ack", "publication", "site_budget",
             "publication_ack", "customer_readback", "service_identity_after", "event_history"]
    if not frozen_at <= min(stamps.values()) or any(stamps[a] > stamps[b] for a, b in zip(chain, chain[1:])):
        blockers.append("causal_timestamp_order_invalid")
    for facts_for_execution in execution_facts.values():
        started, finished, collection, closed, acquired = (_stamp(facts_for_execution.get(key)) for key in
            ("started_at", "completed_at", "closeout_collection_started_at", "closeout_observed_at", "closeout_acquired_at"))
        if not stamps["queued_ack"] <= started <= finished <= stamps["publication"]:
            blockers.append("execution_started_before_current_ack_or_after_publication")
        if collection < finished:
            blockers.append("provider_absence_observed_before_execution_completed")
        elif not collection <= closed <= acquired <= stamps["event_history"]:
            blockers.append("provider_absence_acquisition_time_invalid")
        if not stamps["queued_ack"] <= _stamp(facts_for_execution.get("funded_launch_intent_at")) <= started:
            blockers.append("site_budget_reservation_did_not_precede_launch")
    expected_runtimes = attempt.get("candidate_runtime_identities")
    if not isinstance(expected_runtimes, Mapping):
        expected_runtimes = {}
    candidates = {row["candidate_id"]: row for row in plan["candidates"]}
    for candidate_id, facts_for_execution in execution_facts.items():
        expected = expected_runtimes.get(candidate_id)
        expected = dict(expected) if isinstance(expected, Mapping) else {}
        configured = candidates[candidate_id]
        if _stamp(configured["cost_quote"]["valid_until"]) < _stamp(facts_for_execution["started_at"]):
            blockers.append("cost_quote_expired_before_execution")
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
    if (not cap.is_finite() or not Decimal("0") < cap <= Decimal("15")
            or Decimal(str(plan["maximum_planned_cost_usd"])) > cap
            or Decimal(str(site_budget["committed_upper_bound_usd"])) > cap
            or facts["site_budget"].get("budget_ledger_id") != authority.get("budget_ledger_id")):
        blockers.append("screening_budget_binding_invalid")
    for execution in execution_facts.values():
        maximum_end = _stamp(execution["started_at"]) + timedelta(seconds=execution["issued_ttl_seconds"])
        if execution.get("issued_deadline_epoch") is not None:
            maximum_end = min(maximum_end, _epoch_stamp(execution["issued_deadline_epoch"]))
        if _stamp(execution["completed_at"]) > maximum_end:
            blockers.append("observed_execution_exceeds_issued_runtime")
        if maximum_end > _stamp(authority.get("expires_at")):
            blockers.append("issued_session_runtime_exceeds_capture_authority")
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
