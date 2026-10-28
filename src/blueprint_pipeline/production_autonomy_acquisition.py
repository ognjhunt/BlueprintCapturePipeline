"""Narrow authenticated acquisition of existing AUT-21 retained sources.

Beta journey: join one bounded evaluation to the current job projection.
This is preparation for an operator-owned reader, not a production reader or an
acceptance run. There is no default reader, network client, credential loader,
CLI import hook, writer, publication retry, or provider call here.

The caller supplies a capability which independently authenticates the exact
service and source version and retains its original response bytes. It must
never reconstruct this capability from job JSON or infer acquisition time from
file mtime, publication time, a self-digest, or a production-looking label.
Selectors below name only existing read surfaces. In particular the signed
execution-control POST is stateful and cannot be used as an acceptance read.

Known native schemas are interpreted here; the reader cannot supply projected
facts. Missing capabilities stay unknown. Complete upload/intervention/release
history and actual worker image/interval acquisition still need their own
authenticated sources, so this partial adapter alone cannot prove completion.
"""
from __future__ import annotations

import base64
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import zlib

from .decision_evidence_contracts import canonical_digest
from .production_autonomy_acceptance import (
    ATTEMPT_SCHEMA, SOURCE_KEYS, AcceptanceEvidenceError, TrustedObservation,
    _DIGEST, _COMMIT, _epoch_stamp, _object, _publication_contract, _record, _stamp,
)

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}\Z")
_WEBAPP_ROLES = frozenset({"queued_ack", "publication_ack", "customer_readback"})
_MAX_BYTES = 16 * 1024 * 1024
_MAX_READS = 256


@dataclass(frozen=True)
class SourceSelector:
    """Closed operation chosen by adapter code, never an arbitrary job URL."""

    role: str
    kind: str
    resource: str
    phase: str = "before"


@dataclass(frozen=True)
class AuthenticatedSourceRead:
    """Capability result from an independently authenticated existing reader.

    ``observed_at`` is the actual acquisition time; ``version`` is Firestore's
    document update time or the exact HTTP/controller response byte digest.
    ``retained_record`` locates those same original bytes under evidence_root.
    Service commit/image identity must come from authenticated service identity,
    not from the request, job payload, or an expected-commit echo.
    Test-created instances establish only hermetic contract behavior.
    """

    selector: SourceSelector
    body: bytes
    retained_record: Mapping[str, Any]
    service_identity: str
    authentication: str
    environment: str
    origin: str
    commit: str
    image_digest: str
    observed_at: str
    version: str
    http_status: int | None = None


AuthenticatedSourceReader = Callable[[SourceSelector], AuthenticatedSourceRead | None]


@dataclass(frozen=True)
class OwnResourcePin:
    """Operator-selected native attempt identity, never discovered from a job path."""

    coordinator_attempt_id: str
    candidate_id: str
    authority_digest: str
    prepared_digest: str
    resource_name: str
    instance_id: int
    runtime_inputs_digest: str
    bundle_sha256: str
    adapter_digest: str
    receipt_digest: str


def _json(raw: bytes) -> dict[str, Any]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise AcceptanceEvidenceError("acquisition_duplicate_json_key")
            result[key] = value
        return result
    def nonfinite(_):
        raise ValueError()
    try:
        return _object(json.loads(raw, object_pairs_hook=unique,
            parse_constant=nonfinite))
    except (ValueError, UnicodeError) as exc:
        raise AcceptanceEvidenceError("acquisition_json_invalid") from exc


def _stored_publication(record: Mapping[str, Any]) -> dict[str, Any]:
    """Mirror the existing bounded WebApp gzip storage, checking original bytes."""
    # The current native screening writer always uses publication_storage.
    # Accepting an invented inline publication would bypass that retained type.
    storage = _object(record.get("publication_storage"))
    expected = {"schema_version", "encoding", "payload_sha256", "uncompressed_size_bytes",
                "compressed_size_bytes", "payload_base64"}
    if (set(storage) != expected or "publication" in record
            or storage["schema_version"] != "task_evaluation_run_publication_gzip.v1"
            or storage["encoding"] != "gzip+base64"
            or type(storage["compressed_size_bytes"]) is not int
            or not 0 < storage["compressed_size_bytes"] <= 850_000
            or type(storage["uncompressed_size_bytes"]) is not int
            or not 0 < storage["uncompressed_size_bytes"] <= 8_000_000
            or not isinstance(storage["payload_base64"], str)
            or len(storage["payload_base64"]) > 1_133_336):
        raise AcceptanceEvidenceError("acquisition_publication_storage_invalid")
    try:
        compressed = base64.b64decode(storage["payload_base64"], validate=True)
        if len(compressed) != storage["compressed_size_bytes"]:
            raise ValueError()
        decoder = zlib.decompressobj(wbits=31)
        raw = decoder.decompress(compressed, storage["uncompressed_size_bytes"] + 1)
        if (not decoder.eof or decoder.unused_data or decoder.unconsumed_tail
                or len(raw) != storage["uncompressed_size_bytes"]
                or "sha256:" + hashlib.sha256(raw).hexdigest() != storage["payload_sha256"]):
            raise ValueError()
    except (ValueError, zlib.error) as exc:
        raise AcceptanceEvidenceError("acquisition_publication_storage_invalid") from exc
    return _json(raw)


class AuthenticatedAcceptanceCollector:
    """Callable partial AcquisitionVerifier over one operator-pinned chain.

    Pins are supplied by the operator integration, outside the job manifest.
    This first WebApp envelope uses the native team namespace as tenant_id;
    a different tenant mapping needs independently authoritative evidence.
    A reader failure is retained as a sanitized diagnostic and never retried in
    this collector. Successful observations are cached for those exact bytes.
    Native missing/closed/stale data is not repaired or promoted to authority.
    """

    def __init__(self, *, frozen_attempt: Mapping[str, Any], plan: Mapping[str, Any],
                 publication: Mapping[str, Any], owner_user_id: str, team_namespace: str,
                 scene_intake_id: str, webapp_service_identity: str,
                 pipeline_service_identity: str, evidence_root: str | Path,
                 reader: AuthenticatedSourceReader,
                 own_resources: tuple[OwnResourcePin, ...] = ()):
        from .beta_policy_screening_plan import validate_screening_plan

        # Private snapshots prevent caller mutation after pinning. No path or
        # callback name can be loaded from the frozen attempt or manifest.
        self._attempt = _json(json.dumps(dict(frozen_attempt), allow_nan=False).encode())
        self._plan = validate_screening_plan(plan)
        self._publication = _json(json.dumps(dict(publication), allow_nan=False).encode())
        source = self._attempt.get("source") or {}
        releases = self._attempt.get("releases") or {}
        if (self._attempt.get("schema_version") != ATTEMPT_SCHEMA
                or self._attempt.get("attempt_digest") != canonical_digest(self._attempt, digest_field="attempt_digest")
                or not _ID.fullmatch(str(self._attempt.get("attempt_id") or ""))
                or any(not isinstance(source.get(key), str) or not source[key] for key in SOURCE_KEYS)
                or any(not _DIGEST.fullmatch(source[key]) for key in SOURCE_KEYS if key.endswith("digest"))
                or not _ID.fullmatch(source["job_id"])
                or source["site_id"] != "site-" + source["job_id"]
                or source["tenant_id"] != team_namespace
                or any(source.get(key) != self._plan["source"].get(key) for key in self._plan["source"])
                or self._plan["site_id"] != source["site_id"]
                or any(not _COMMIT.fullmatch(str(releases.get(name + "_commit") or ""))
                    or not _DIGEST.fullmatch(str(releases.get(name + "_image_digest") or ""))
                    for name in ("pipeline", "webapp"))
                or not callable(reader)
                or any(not _ID.fullmatch(value) for value in (owner_user_id, team_namespace, scene_intake_id,
                    webapp_service_identity, pipeline_service_identity))):
            raise AcceptanceEvidenceError("acquisition_pin_invalid")
        self._frozen = _stamp(self._attempt.get("frozen_at"))
        self._owner, self._team, self._intake = owner_user_id, team_namespace, scene_intake_id
        self._peers = {"webapp": webapp_service_identity, "pipeline": pipeline_service_identity}
        self._root, self._reader = Path(evidence_root).absolute(), reader
        self._cache: dict[tuple[str, str], TrustedObservation | None] = {}
        self._failures: list[dict[str, str]] = []
        self._identities: list[dict[str, Any]] = []
        self._read_count = 0
        self._resources = {}
        for pin in own_resources:
            if (type(pin) is not OwnResourcePin or not _ID.fullmatch(pin.coordinator_attempt_id)
                    or not _ID.fullmatch(pin.resource_name) or type(pin.instance_id) is not int or pin.instance_id <= 0
                    or pin.candidate_id not in {row["candidate_id"] for row in self._plan["candidates"]}
                    or any(not _DIGEST.fullmatch(str(getattr(pin, key))) for key in (
                        "authority_digest", "prepared_digest", "runtime_inputs_digest", "bundle_sha256",
                        "adapter_digest", "receipt_digest")) or pin.receipt_digest in self._resources):
                raise AcceptanceEvidenceError("acquisition_resource_pin_invalid")
            self._resources[pin.receipt_digest] = pin
        if len(self._resources) > 32:
            raise AcceptanceEvidenceError("acquisition_resource_pin_limit")
        self._publication_binding(self._publication, queued=False)

    def _publication_binding(self, value, *, queued):
        source, plan = self._attempt["source"], self._plan
        try:
            _publication_contract(value)
        except AcceptanceEvidenceError:
            raise AcceptanceEvidenceError("acquisition_publication_binding_invalid") from None
        if (value.get("campaign_id") != plan["campaign_id"] or value.get("site_id") != source["site_id"]
                or value.get("plan_digest") != plan["plan_digest"] or value.get("source") != plan["source"]
                or value.get("catalog_digest") != plan["catalog_digest"]
                or value.get("scenario_plan_digest") != plan["scenario_plan_digest"]
                or type(value.get("revision")) is not int or value["revision"] < 1
                or value.get("status") != ("queued" if queued else "completed")):
            raise AcceptanceEvidenceError("acquisition_publication_binding_invalid")
        _stamp(value.get("generated_at"))
        rows = value.get("candidates")
        if (not isinstance(rows, list) or len(rows) != len(plan["candidates"])
                or {row.get("candidate_id") for row in rows} != {row["candidate_id"] for row in plan["candidates"]}):
            raise AcceptanceEvidenceError("acquisition_candidate_coverage_invalid")
        expected = {row["candidate_id"]: row for row in plan["candidates"]}
        for row in rows:
            candidate = expected[row["candidate_id"]]
            if row.get("configuration_digest") != candidate["configuration_digest"]:
                raise AcceptanceEvidenceError("acquisition_candidate_binding_invalid")
            if candidate["status"] == "selected" and (
                    row.get("episodes_planned") != 10 or row.get("episodes_scored") != (0 if queued else 10)
                    or row.get("status") != ("pending" if queued else "completed")
                    or (not queued and not _DIGEST.fullmatch(str(row.get("session_result_digest") or "")))):
                raise AcceptanceEvidenceError("acquisition_candidate_result_invalid")

    def _read(self, selector: SourceSelector) -> tuple[dict[str, Any], AuthenticatedSourceRead]:
        if self._read_count >= _MAX_READS:
            raise AcceptanceEvidenceError("acquisition_read_limit")
        self._read_count += 1
        identity = {"role": selector.role, "kind": selector.kind, "resource": selector.resource,
            "phase": selector.phase, "status": "not_verified"}
        self._identities.append(identity)
        response = self._reader(selector)
        if type(response) is not AuthenticatedSourceRead or response.selector != selector:
            raise AcceptanceEvidenceError("authenticated_source_unavailable")
        if type(response.body) is not bytes or not 0 < len(response.body) <= _MAX_BYTES:
            raise AcceptanceEvidenceError("authenticated_source_body_invalid")
        webapp = selector.kind.startswith("webapp_")
        producer = "webapp" if webapp else "pipeline"
        digest = "sha256:" + hashlib.sha256(response.body).hexdigest()
        identity["sha256"] = digest
        if (response.retained_record.get("sha256") != digest
                or response.retained_record.get("size_bytes") != len(response.body)):
            raise AcceptanceEvidenceError("authenticated_source_binding_invalid")
        _record(self._root, response.retained_record, json_object=False)
        # A failed authentication check still indexes the safely retained raw
        # bytes as unverified evidence, including non-JSON error responses. The
        # descriptor is indexed only after safe path/type/size/hash validation.
        identity["retained_record"] = dict(response.retained_record)
        payload = _json(response.body)
        expected_auth = ("firestore_server_identity" if selector.kind in {"webapp_run", "webapp_result"}
            else "authenticated_https_response" if webapp else "authenticated_controller_snapshot")
        if (response.environment != "production" or response.origin != "natural_production"
                or response.authentication != expected_auth or response.service_identity != self._peers[producer]
                or (expected_auth == "authenticated_https_response" and response.http_status != 200)
                or response.commit != self._attempt["releases"][producer + "_commit"]
                or response.image_digest != self._attempt["releases"][producer + "_image_digest"]
                or _stamp(response.observed_at) < self._frozen):
            raise AcceptanceEvidenceError("authenticated_source_binding_invalid")
        if expected_auth == "firestore_server_identity":
            if not _stamp(payload.get("updated_at_iso")) <= _stamp(response.version) <= _stamp(response.observed_at):
                raise AcceptanceEvidenceError("authenticated_document_version_invalid")
        elif response.version != digest:
            raise AcceptanceEvidenceError("authenticated_response_identity_invalid")
        identity.update({"version": response.version, "observed_at": response.observed_at,
            "service_identity": response.service_identity, "authentication": response.authentication,
            "commit": response.commit, "image_digest": response.image_digest,
            "http_status": response.http_status, "status": "verified"})
        return payload, response

    def _webapp(self, role, digest, payload):
        campaign, source = self._plan["campaign_id"], self._attempt["source"]
        result_id = "screening-" + hashlib.sha256(campaign.encode()).hexdigest()[:32]
        def read(kind, resource, phase="before"):
            return self._read(SourceSelector(role, kind, resource, phase))
        run, run_read = read("webapp_run", "taskEvaluationPolicyRuns/" + campaign)
        result, result_read = read("webapp_result", "captureTaskEvaluationRuns/" + result_id)
        workspace, workspace_read = read("webapp_workspace", "GET /api/job-workflow/admin/" + source["job_id"])
        selected = workspace_read if role == "customer_readback" else result_read
        if selected.retained_record["sha256"] != digest or _json(selected.body) != payload:
            raise AcceptanceEvidenceError("acquisition_primary_bytes_mismatch")
        publication = _stored_publication(result)
        queued = role == "queued_ack"
        self._publication_binding(publication, queued=queued)
        if (not _stamp(publication["generated_at"]) <= _stamp(result.get("updated_at_iso"))
                or not _stamp(publication["generated_at"]) <= _stamp(run.get("updated_at_iso"))):
            raise AcceptanceEvidenceError("acquisition_persistence_time_invalid")
        if (not queued and publication != self._publication) or (queued and publication["revision"] >= self._publication["revision"]):
            raise AcceptanceEvidenceError("acquisition_publication_revision_mismatch")
        if (result.get("schema_version") != "capture_task_evaluation_run_record.v2"
                or result.get("record_id") != result_id or result.get("run_id") != campaign
                or result.get("owner_user_id") != self._owner or result.get("organization_id") != self._team
                or result.get("access_visibility") != "private"
                or run.get("run_id") != campaign or run.get("run_kind") != "beta_policy_screening"
                or run.get("owner_user_id") != self._owner or run.get("team_namespace") != self._team
                or run.get("result_record_id") != result_id or run.get("source_scene_intake_id") != self._intake
                or run.get("source_scene_intent_id") != source["scene_intent_id"]
                or run.get("source_scene_revision_digest") != source["scene_revision_digest"]
                or run.get("job_task_context_digest") != source["task_context_digest"]
                or run.get("configuration_digest") != self._plan["plan_digest"]
                or run.get("job_evidence_binding") != {"job_id": source["job_id"], "brief_digest": source["technical_brief_digest"]}
                or run.get("publication_digest") != publication["publication_digest"]
                or run.get("publication_revision") != publication["revision"]
                or run.get("state") != publication["status"] or run.get("cancellationRequested") is True):
            raise AcceptanceEvidenceError("acquisition_persistence_identity_mismatch")
        projected = workspace.get("managedEvidence")
        matches = [row for row in projected if isinstance(row, Mapping) and row.get("runId") == campaign] if isinstance(projected, list) else []
        if (workspace.get("jobId") != source["job_id"] or workspace.get("managedEvidenceUnavailable") is True
                or len(matches) != 1 or matches[0].get("current") is not True
                or matches[0].get("state") != publication["status"]
                or matches[0].get("resultDigest") != (None if queued else publication["publication_digest"])):
            raise AcceptanceEvidenceError("acquisition_current_customer_readback_unproven")
        # Re-read native versions after the actual customer projection. A mixed
        # publication/revision cannot be repaired by joining a later snapshot.
        run_after, run_after_read = read("webapp_run", "taskEvaluationPolicyRuns/" + campaign, "after")
        result_after, result_after_read = read("webapp_result", "captureTaskEvaluationRuns/" + result_id, "after")
        if (run_after != run or result_after != result or run_after_read.version != run_read.version
                or result_after_read.version != result_read.version):
            raise AcceptanceEvidenceError("acquisition_persistence_changed_during_readback")
        reads = (run_read, result_read, workspace_read, run_after_read, result_after_read)
        stamps = [_stamp(row.observed_at) for row in reads]
        if (stamps != sorted(stamps) or (stamps[-1] - stamps[0]).total_seconds() > 120
                or _stamp(publication["generated_at"]) > stamps[0]):
            raise AcceptanceEvidenceError("acquisition_snapshot_time_invalid")
        facts = {"plan_digest": self._plan["plan_digest"], "publication_digest": publication["publication_digest"],
            "revision": publication["revision"], "current_source": True, "persisted": True,
            "displayed": True, "webapp_commit": self._attempt["releases"]["webapp_commit"],
            "result_digests": {row["candidate_id"]: row["session_result_digest"] for row in publication["candidates"]
                if row.get("status") == "completed"},
            "result_record_id": result_id, "source_scene_intake_id": self._intake}
        return result_after_read.observed_at, facts

    def _closeout(self, digest, payload):
        from .beta_policy_screening_closeout import SCHEMA, SCOPE, _observation_blockers

        pin = self._resources.get(payload.get("receipt_digest"))
        if pin is None:
            raise AcceptanceEvidenceError("acquisition_owned_resource_not_pinned")
        resource = f"attempts/{pin.coordinator_attempt_id}/provider_absence/{canonical_digest(payload)[7:]}.json"
        value, response = self._read(SourceSelector("execution_closeout", "pipeline_closeout", resource))
        if response.retained_record["sha256"] != digest or value != payload:
            raise AcceptanceEvidenceError("acquisition_primary_bytes_mismatch")
        expected = {key: getattr(pin, key) for key in ("authority_digest", "prepared_digest", "resource_name",
            "instance_id", "runtime_inputs_digest", "bundle_sha256", "adapter_digest", "receipt_digest")}
        admission = payload.get("beta_screening_admission") or {}
        candidate = next(row for row in self._plan["candidates"] if row["candidate_id"] == pin.candidate_id)
        if (payload.get("schema_version") != SCHEMA or payload.get("scope") != SCOPE
                or payload.get("provider") != "vast" or type(payload.get("instance_id")) is not int
                or payload.get("global_provider_zero_claimed") is not False
                or payload.get("receipt_digest") != canonical_digest(payload, digest_field="receipt_digest")
                or any(payload.get(key) != value for key, value in expected.items())
                or payload.get("candidate_ids") != [pin.candidate_id]
                or admission.get("plan_digest") != self._plan["plan_digest"]
                or admission.get("configuration_digest") != candidate["configuration_digest"]
                or admission.get("scene_revision_digest") != self._attempt["source"]["scene_revision_digest"]
                or payload.get("status") != "verified" or payload.get("blockers") != []
                or payload.get("provider_absence_confirmed") is not True
                or _observation_blockers(payload, now=payload.get("observed_at_epoch"))
                or _epoch_stamp(payload.get("observed_at_epoch")) > _stamp(response.observed_at)):
            raise AcceptanceEvidenceError("acquisition_owned_resource_binding_invalid")
        # Original provider collection times are retained; reacquisition never
        # refreshes them. Canonical nested-byte/teardown validation remains in
        # production_autonomy_acceptance._execution, using the existing seam.
        return response.observed_at, {"original_collection_started_at": _epoch_stamp(payload["collection_started_at_epoch"]).isoformat(),
            "original_observed_at": _epoch_stamp(payload["observed_at_epoch"]).isoformat()}

    def __call__(self, role: str, digest: str, payload: Mapping[str, Any]) -> TrustedObservation | None:
        key = role, digest
        if key in self._cache:
            return self._cache[key]
        start = len(self._identities)
        try:
            if role not in _WEBAPP_ROLES and role != "execution_closeout":
                raise AcceptanceEvidenceError("native_acquisition_role_not_implemented")
            observed_at, facts = self._webapp(role, digest, payload) if role in _WEBAPP_ROLES else self._closeout(digest, payload)
            identities = [dict(row) for row in self._identities[start:]]
            facts["acquisition_sources"] = identities
            value = TrustedObservation(digest, payload.get("schema_version"),
                "webapp" if role in _WEBAPP_ROLES else "pipeline", "production", "natural_production",
                self._attempt["attempt_id"], "authenticated-read-" + canonical_digest({"sources": identities})[7:],
                observed_at, dict(self._attempt["source"]), facts)
        except Exception as exc:
            # Preserve failed reads in diagnostics; never retry uncertain reads
            # or echo private exception messages/URLs/credentials.
            code = str(exc) if isinstance(exc, AcceptanceEvidenceError) else "authenticated_source_acquisition_unavailable"
            self._failures.append({"role": role, "sha256": digest, "code": code})
            if len(self._identities) > start and self._identities[-1]["status"] == "not_verified":
                self._identities[-1].update(status="failed", code=code)
            value = None
        self._cache[key] = value
        return value

    def diagnostics(self) -> dict[str, Any]:
        """Caller may retain this beside existing proof reports; no write occurs."""
        return {"schema_version": "blueprint.production_autonomy_acquisition_report.v1",
            "attempt_id": self._attempt["attempt_id"], "attempt_digest": self._attempt["attempt_digest"],
            "plan_digest": self._plan["plan_digest"], "publication_digest": self._publication["publication_digest"],
            "pinned_release_identities": dict(self._attempt["releases"]),
            "supported_roles": sorted([*_WEBAPP_ROLES, "execution_closeout"]),
            "authenticated_observations": sum(value is not None for value in self._cache.values()),
            "failures": [dict(row) for row in self._failures], "read_count": self._read_count,
            "sources": [dict(row) for row in self._identities],
            "production_completion_proven": False, "starts_acceptance_attempt": False,
            "performs_provider_calls": False, "writes_sources": False,
            "missing_capabilities": ["authenticated_natural_upload_and_confirmation_history",
                "authenticated_current_capture_grant_and_site_allowance", "authenticated_pipeline_stage_sources",
                "authenticated_service_identity_and_complete_intervention_history",
                "authenticated_original_execution_interval_and_provider_worker_image",
                "authenticated_normal_retention_observation"]}
