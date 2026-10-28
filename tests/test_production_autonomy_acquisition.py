"""Synthetic authenticated readers test native joins; they prove no production run."""
from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import replace
import gzip
import hashlib
import json
from types import SimpleNamespace

import pytest

from blueprint_pipeline import production_autonomy_acquisition as acquisition
from blueprint_pipeline import production_autonomy_acceptance as acceptance
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from tests.test_production_autonomy_acceptance import retained, record, stamp  # noqa: F401


def stored(publication):
    raw = json.dumps(publication, sort_keys=True, separators=(",", ":")).encode()
    compressed = gzip.compress(raw)
    return {"schema_version": "task_evaluation_run_publication_gzip.v1", "encoding": "gzip+base64",
        "payload_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "uncompressed_size_bytes": len(raw), "compressed_size_bytes": len(compressed),
        "payload_base64": base64.b64encode(compressed).decode()}


@pytest.fixture
def sources(retained):  # noqa: F811 -- reuse the existing native evidence fixture
    root = retained["root"]
    manifest, attempt = retained["manifest"], retained["attempt"]
    plan = json.loads(open(manifest["receipts"]["candidate_plan"]["path"]).read())
    publication = json.loads(open(manifest["receipts"]["publication"]["path"]).read())
    result_id = "screening-" + hashlib.sha256(plan["campaign_id"].encode()).hexdigest()[:32]
    run = {"run_id": plan["campaign_id"], "run_kind": "beta_policy_screening", "state": "completed",
        "owner_user_id": "owner-1", "team_namespace": "tenant-1", "result_record_id": result_id,
        "source_scene_intake_id": "intake-1", "source_scene_intent_id": plan["source"]["scene_intent_id"],
        "source_scene_revision_digest": plan["source"]["scene_revision_digest"],
        "job_task_context_digest": plan["source"]["task_context_digest"], "configuration_digest": plan["plan_digest"],
        "job_evidence_binding": {"job_id": attempt["source"]["job_id"],
            "brief_digest": attempt["source"]["technical_brief_digest"]},
        "publication_digest": publication["publication_digest"], "publication_revision": publication["revision"],
        "created_at_iso": stamp(25), "updated_at_iso": stamp(60)}
    result = {"schema_version": "capture_task_evaluation_run_record.v2", "record_id": result_id,
        "run_id": plan["campaign_id"], "owner_user_id": "owner-1", "organization_id": "tenant-1",
        "access_visibility": "private", "created_at_iso": stamp(25), "updated_at_iso": stamp(60),
        "publication_storage": stored(publication)}
    workspace = {"jobId": attempt["source"]["job_id"], "managedEvidence": [{"runId": plan["campaign_id"],
        "current": True, "state": "completed", "resultDigest": publication["publication_digest"]}]}
    closeout = json.loads(open(manifest["executions"][0]["closeout"]["path"]).read())
    pin = acquisition.OwnResourcePin("screen-fixture-attempt", plan["candidates"][0]["candidate_id"],
        **{key: closeout[key] for key in ("authority_digest", "prepared_digest", "resource_name", "instance_id",
            "runtime_inputs_digest", "bundle_sha256", "adapter_digest", "receipt_digest")})
    pins = [pin]
    calls, mutations, overrides = [], {}, {}

    def reader(selector):
        calls.append(selector)
        if selector.kind == "pipeline_closeout":
            value = closeout
        else:
            value = {"webapp_run": run, "webapp_result": result, "webapp_workspace": workspace}[selector.kind]
        value = deepcopy(value)
        if selector.kind in mutations:
            mutations[selector.kind](value, selector)
        name = selector.role + "-" + selector.kind + "-" + selector.phase + ".json"
        path = root / "acquired" / name
        rec = record(path, value)
        raw = path.read_bytes()
        webapp = selector.kind.startswith("webapp_")
        producer = "webapp" if webapp else "pipeline"
        auth = ("firestore_server_identity" if selector.kind in {"webapp_run", "webapp_result"}
            else "authenticated_https_response" if webapp else "authenticated_controller_snapshot")
        value = acquisition.AuthenticatedSourceRead(selector, raw, rec, producer + "-service", auth,
            "production", "natural_production", attempt["releases"][producer + "_commit"],
            attempt["releases"][producer + "_image_digest"], stamp(70 + len(calls)),
            stamp(60) if auth == "firestore_server_identity" else rec["sha256"],
            200 if auth == "authenticated_https_response" else None)
        return replace(value, **overrides.get(selector.kind, {}))

    def collector():
        return acquisition.AuthenticatedAcceptanceCollector(frozen_attempt=attempt, plan=plan,
            publication=publication, owner_user_id="owner-1", team_namespace="tenant-1",
            scene_intake_id="intake-1", webapp_service_identity="webapp-service",
            pipeline_service_identity="pipeline-service", evidence_root=root, reader=reader, own_resources=tuple(pins))

    def acquire(role="customer_readback"):
        value = workspace if role == "customer_readback" else closeout if role == "execution_closeout" else result
        # The manifest may point at any copy, but the authenticated reader must
        # acquire the identical original source response bytes independently.
        rec = record(root / ("primary-" + role + ".json"), value)
        selected = collector()
        observation = selected(role, rec["sha256"], value)
        return observation, selected
    return SimpleNamespace(retained=retained, run=run, result=result, workspace=workspace,
        publication=publication, plan=plan, attempt=attempt, closeout=closeout, pins=pins,
        calls=calls, mutations=mutations, overrides=overrides, collector=collector, acquire=acquire)


def test_native_persistence_current_projection_and_original_response_identities(sources):
    value, collector = sources.acquire()
    assert value is not None
    assert value.schema_version is None  # The actual workspace JSON has no schema label.
    assert value.facts["persisted"] and value.facts["displayed"] and value.facts["current_source"]
    assert value.facts["result_digests"] == {row["candidate_id"]: row["session_result_digest"]
        for row in sources.publication["candidates"] if row["status"] == "completed"}
    assert len(value.facts["acquisition_sources"]) == 5
    assert [row.kind for row in sources.calls] == ["webapp_run", "webapp_result", "webapp_workspace", "webapp_run", "webapp_result"]
    assert all(row.resource.startswith(("taskEvaluationPolicyRuns/", "captureTaskEvaluationRuns/",
        "GET /api/job-workflow/admin/")) for row in sources.calls)
    assert collector.diagnostics()["production_completion_proven"] is False
    assert collector.diagnostics()["missing_capabilities"]


def test_frozen_tenant_must_be_the_actual_native_webapp_namespace(sources):
    with pytest.raises(acceptance.AcceptanceEvidenceError, match="acquisition_pin_invalid"):
        acquisition.AuthenticatedAcceptanceCollector(frozen_attempt=sources.attempt, plan=sources.plan,
            publication=sources.publication, owner_user_id="owner-1", team_namespace="different-tenant",
            scene_intake_id="intake-1", webapp_service_identity="webapp-service",
            pipeline_service_identity="pipeline-service", evidence_root=sources.retained["root"],
            reader=lambda _: None)


@pytest.mark.parametrize("job_id", [".", ".."])
def test_job_identifier_is_validated_before_any_native_read_selector(sources, job_id):
    sources.attempt["source"].update(job_id=job_id, site_id="site-" + job_id)
    sources.attempt["attempt_digest"] = canonical_digest(sources.attempt, digest_field="attempt_digest")
    sources.plan["site_id"] = "site-" + job_id
    sources.plan["plan_digest"] = canonical_digest(sources.plan, digest_field="plan_digest")
    sources.publication.update(site_id="site-" + job_id, plan_digest=sources.plan["plan_digest"])
    sources.publication["publication_digest"] = cross_runtime_canonical_digest(sources.publication, digest_field="publication_digest")
    with pytest.raises(acceptance.AcceptanceEvidenceError, match="acquisition_pin_invalid"):
        sources.collector()
    assert sources.calls == []


def test_persisted_completed_ack_uses_actual_native_result_record(sources):
    value, _ = sources.acquire("publication_ack")
    assert value is not None
    assert value.schema_version == "capture_task_evaluation_run_record.v2"
    assert value.facts["publication_digest"] == sources.publication["publication_digest"]


def queued_sources(sources):
    queued = deepcopy(sources.publication)
    queued.update(revision=1, status="queued", generated_at=stamp(24))
    for row in queued["candidates"]:
        if row["candidate_id"] == sources.plan["candidates"][0]["candidate_id"]:
            row.update(status="pending", episodes_scored=0, session_result_digest=None)
    queued["publication_digest"] = cross_runtime_canonical_digest(queued, digest_field="publication_digest")
    sources.result["publication_storage"] = stored(queued)
    sources.run.update(state="queued", publication_digest=queued["publication_digest"], publication_revision=1)
    sources.workspace["managedEvidence"][0].update(state="queued", resultDigest=None)


def test_queued_readback_must_be_observed_before_final_result_not_inferred_from_current(sources):
    value, collector = sources.acquire("queued_ack")
    assert value is None
    assert collector.diagnostics()["failures"][0]["code"] == "acquisition_publication_binding_invalid"
    queued_sources(sources)
    value, _ = sources.acquire("queued_ack")
    assert value is not None
    assert value.facts["plan_digest"] == sources.plan["plan_digest"]
    assert value.facts["result_digests"] == {}


def test_late_reacquisition_does_not_backdate_a_queued_ack_for_acceptance(sources):
    queued_sources(sources)
    observed, _ = sources.acquire("queued_ack")
    assert observed is not None
    bundle = sources.retained
    bundle["manifest"]["receipts"]["queued_ack"] = record(
        bundle["root"] / "primary-queued_ack.json", sources.result)
    bundle["manifest_path"].write_text(json.dumps(bundle["manifest"]))
    def verifier(role, digest, payload):
        return observed if role == "queued_ack" else bundle["acquisitions"](role, digest, payload)
    result = acceptance.build_production_autonomy_acceptance_report(
        manifest_path=bundle["manifest_path"], acquisition_verifier=verifier)
    assert result["status"] == "not_proven"
    assert "causal_timestamp_order_invalid" in result["blockers"]
    assert "execution_started_before_current_ack_or_after_publication" in result["blockers"]


def test_raw_source_bytes_must_match_the_manifest_copy(sources):
    sources.mutations["webapp_workspace"] = lambda value, _: value.update(unexpectedServerField="new")
    value, collector = sources.acquire()
    assert value is None
    assert collector.diagnostics()["failures"][0]["code"] == "acquisition_primary_bytes_mismatch"


def test_authenticated_source_must_retain_the_same_original_bytes(sources):
    foreign = record(sources.retained["root"] / "foreign.json", {"other": "bytes"})
    sources.overrides["webapp_workspace"] = {"retained_record": foreign}
    value, collector = sources.acquire()
    assert value is None
    assert collector.diagnostics()["failures"][0]["code"] == "authenticated_source_binding_invalid"


def test_non_json_authenticated_response_retains_safe_original_failure_bytes(sources):
    path = sources.retained["root"] / "retained-unavailable.html"
    path.write_bytes(b"<html>Current authenticated source unavailable</html>")
    rec = record(path)
    sources.overrides["webapp_workspace"] = {"body": path.read_bytes(), "retained_record": rec,
        "version": rec["sha256"], "http_status": 503}
    value, collector = sources.acquire()
    assert value is None
    diagnostics = collector.diagnostics()
    assert diagnostics["failures"][0]["code"] == "acquisition_json_invalid"
    failed = diagnostics["sources"][-1]
    assert failed["status"] == "failed"
    assert failed["retained_record"] == rec
    assert failed["sha256"] == rec["sha256"]
    before = diagnostics["read_count"]
    digest = record(sources.retained["root"] / "primary-customer_readback.json", sources.workspace)["sha256"]
    assert collector("customer_readback", digest, sources.workspace) is None
    assert collector.diagnostics()["read_count"] == before


def test_failed_response_cannot_index_an_unsafe_retained_descriptor(sources):
    root = sources.retained["root"]
    original = root / "retained-original.html"
    original.write_bytes(b"<html>unavailable</html>")
    unsafe = root / "unsafe-symlink.html"
    unsafe.symlink_to(original)
    rec = {**record(original), "path": str(unsafe)}
    sources.overrides["webapp_workspace"] = {"body": original.read_bytes(), "retained_record": rec,
        "version": rec["sha256"], "http_status": 503}
    value, collector = sources.acquire()
    assert value is None
    assert "retained_record" not in collector.diagnostics()["sources"][-1]


@pytest.mark.parametrize("kind,changes", [
    ("webapp_workspace", {"service_identity": "other-service"}),
    ("webapp_result", {"authentication": "job_json_export"}),
    ("webapp_workspace", {"environment": "staging"}),
    ("webapp_result", {"origin": "hermetic_fixture"}),
    ("webapp_workspace", {"commit": "c" * 40}),
    ("webapp_workspace", {"image_digest": "sha256:" + "c" * 64}),
    ("webapp_result", {"observed_at": stamp(-1)}),
    ("webapp_result", {"version": stamp(59)}),
    ("webapp_workspace", {"version": "asserted-response-id"}),
    ("webapp_workspace", {"http_status": None}),
    ("webapp_workspace", {"http_status": 401}),
    ("webapp_workspace", {"http_status": 409}),
    ("webapp_workspace", {"http_status": 503}),
])
def test_job_labels_fixture_changed_service_or_invalid_versions_are_not_authentication(sources, kind, changes):
    sources.overrides[kind] = changes
    value, collector = sources.acquire()
    assert value is None
    diagnostics = collector.diagnostics()
    assert not diagnostics["authenticated_observations"]
    assert diagnostics["sources"][-1]["status"] == "failed"
    assert diagnostics["sources"][-1]["retained_record"]["sha256"].startswith("sha256:")


@pytest.mark.parametrize("kind,key,replacement", [
    ("webapp_run", "team_namespace", "other-tenant"),
    ("webapp_result", "owner_user_id", "other-owner"),
    ("webapp_result", "record_id", "other-result"),
    ("webapp_run", "source_scene_intake_id", "other-intake"),
    ("webapp_run", "source_scene_intent_id", "other-intent"),
    ("webapp_run", "source_scene_revision_digest", "sha256:" + "0" * 64),
    ("webapp_run", "job_task_context_digest", "sha256:" + "0" * 64),
    ("webapp_run", "configuration_digest", "sha256:" + "0" * 64),
    ("webapp_run", "publication_revision", 1),
    ("webapp_run", "cancellationRequested", True),
])
def test_cross_tenant_source_plan_revision_or_stopped_run_cannot_be_current(sources, kind, key, replacement):
    sources.mutations[kind] = lambda value, _: value.update({key: replacement})
    value, collector = sources.acquire()
    assert value is None
    assert collector.diagnostics()["failures"][0]["code"] == "acquisition_persistence_identity_mismatch"


@pytest.mark.parametrize("change", [
    lambda value: value.update(jobId="other-job"),
    lambda value: value.update(managedEvidenceUnavailable=True),
    lambda value: value["managedEvidence"][0].update(current=False),
    lambda value: value["managedEvidence"][0].update(resultDigest="sha256:" + "0" * 64),
    lambda value: value["managedEvidence"].append(deepcopy(value["managedEvidence"][0])),
])
def test_actual_workspace_must_display_one_current_exact_persisted_publication(sources, change):
    sources.mutations["webapp_workspace"] = lambda value, _: change(value)
    # Keep the primary copy identical to the bad actual response, so this
    # checks semantic currentness rather than just primary-byte mismatch.
    change(sources.workspace)
    sources.mutations.clear()
    value, collector = sources.acquire()
    assert value is None
    assert collector.diagnostics()["failures"][0]["code"] == "acquisition_current_customer_readback_unproven"


def test_native_document_change_between_current_read_and_confirmation_is_not_a_join(sources):
    def mutate(value, selector):
        if selector.phase == "after":
            value["screening_execution_observed"] = True
    sources.mutations["webapp_run"] = mutate
    value, collector = sources.acquire()
    assert value is None
    assert collector.diagnostics()["failures"][0]["code"] == "acquisition_persistence_changed_during_readback"


@pytest.mark.parametrize("field,change", [
    ("payload_sha256", "sha256:" + "0" * 64),
    ("uncompressed_size_bytes", 8_000_001),
    ("compressed_size_bytes", 850_001),
    ("payload_base64", "not-base64"),
    ("schema_version", "job_asserted_publication.v1"),
])
def test_stored_native_publication_corruption_is_unknown(sources, field, change):
    sources.mutations["webapp_result"] = lambda value, _: value["publication_storage"].update({field: change})
    value, collector = sources.acquire()
    assert value is None
    assert collector.diagnostics()["failures"][0]["code"] == "acquisition_publication_storage_invalid"


def test_schema_label_or_inline_fixture_is_not_native_persistence(sources):
    sources.mutations["webapp_result"] = lambda value, _: value.update(publication=deepcopy(sources.publication))
    value, _ = sources.acquire()
    assert value is None


def test_exact_own_resource_closeout_keeps_original_collection_time(sources):
    value, collector = sources.acquire("execution_closeout")
    assert value is not None
    assert value.facts["original_observed_at"] != value.observed_at
    assert value.facts["original_observed_at"] == acceptance._epoch_stamp(sources.closeout["observed_at_epoch"]).isoformat()
    assert sources.calls[0].resource.startswith("attempts/screen-fixture-attempt/provider_absence/")
    assert collector.diagnostics()["performs_provider_calls"] is False


@pytest.mark.parametrize("field,replacement", [
    ("instance_id", 9999), ("resource_name", "other-owned-resource"),
    ("authority_digest", "sha256:" + "0" * 64), ("prepared_digest", "sha256:" + "0" * 64),
])
def test_own_resource_pin_prevents_arbitrary_provider_zero_receipts(sources, field, replacement):
    sources.pins[0] = replace(sources.pins[0], **{field: replacement})
    value, collector = sources.acquire("execution_closeout")
    assert value is None
    assert collector.diagnostics()["failures"][0]["code"] == "acquisition_owned_resource_binding_invalid"


def test_failed_acquisition_is_retained_sanitized_and_not_retried(sources):
    collector = sources.collector()
    def denied(_):
        raise RuntimeError("private credential and original denied URL")
    collector._reader = denied
    digest = record(sources.retained["root"] / "primary-failure.json", sources.workspace)["sha256"]
    assert collector("customer_readback", digest, sources.workspace) is None
    assert collector("customer_readback", digest, sources.workspace) is None
    result = collector.diagnostics()
    assert result["read_count"] == 1
    assert len(result["failures"]) == 1
    assert result["failures"][0]["code"] == "authenticated_source_acquisition_unavailable"
    assert "credential" not in json.dumps(result)


def test_partial_collector_cannot_promote_complete_fixture_to_production_acceptance(sources):
    collector = sources.collector()
    result = acceptance.build_production_autonomy_acceptance_report(
        manifest_path=sources.retained["manifest_path"], acquisition_verifier=collector)
    assert result["status"] == "not_proven"
    assert not any(result["milestones"].values())
    assert "attempt:acquisition_binding_invalid" in result["blockers"]
    assert any(row["code"] == "native_acquisition_role_not_implemented" for row in collector.diagnostics()["failures"])


def test_json_cannot_deserialize_the_authenticated_reader_capability(sources):
    collector = sources.collector()
    collector._reader = lambda selector: {"selector": selector, "authentication": "firestore_server_identity"}
    value = sources.workspace
    digest = record(sources.retained["root"] / "primary-json.json", value)["sha256"]
    assert collector("customer_readback", digest, value) is None
    assert collector.diagnostics()["failures"][0]["code"] == "authenticated_source_unavailable"
