"""Hermetic receipt contracts; the synthetic acquisitions here prove no live run."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from blueprint_pipeline import beta_policy_screening_closeout as closeout
from blueprint_pipeline import episode_visual_evidence as visual
from blueprint_pipeline import production_autonomy_acceptance as acceptance
from blueprint_pipeline.beta_policy_screening_result import candidate_result_summary, screening_publication
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.native_task_arena_policy_canary_session import (
    BETA_RUN_KIND, PROVIDER_BUNDLE_SCHEMA_VERSION, PROVIDER_RESULT_FILENAME,
    consume_session_authority_once,
)
from blueprint_pipeline.robot_free_probe_gate import REQUIRED_CHECKS, RIGID_RECEIPT_SCHEMA
from tests.test_beta_policy_screening_closeout import Provider
from tests.test_beta_policy_screening_coordinator import native_result, plan, seal
from tests.test_episode_visual_evidence import _calibration
from tests.test_native_task_arena_policy_canary_session import _activation, _authority


def record(path, value=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if value is not None:
        path.write_text(json.dumps(value, sort_keys=True))
    return {"path": str(path), "size_bytes": path.stat().st_size,
            "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()}


def stamp(seconds):
    return (datetime(2026, 10, 28, tzinfo=timezone.utc) + timedelta(seconds=seconds)).isoformat()


class RetainedAcquisitions:
    """Test-owned registry outside the job JSON, standing in for an authenticated reader."""

    def __init__(self):
        self.rows = {}

    def __call__(self, role, digest, payload):
        return self.rows.get((role, digest))


@pytest.fixture
def retained(tmp_path, monkeypatch, request):
    options = getattr(request, "param", {})
    def encode(paths, *, video_path, frames_per_second):
        # Exercise hashes/joins, not ffmpeg or provider execution.
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"hermetic-review-video")
        return {"sha256": record(video_path)["sha256"], "size_bytes": video_path.stat().st_size,
                "frame_count": len(paths), "decoded_frame_count": len(paths)}
    monkeypatch.setattr(visual, "_encode_or_resume_episode_video", encode)
    root, output = tmp_path, tmp_path / "candidate-output"
    output.mkdir()
    value = plan()
    source = {"tenant_id": "tenant-1", "site_id": value["site_id"], "job_id": "job-1",
              "technical_brief_digest": canonical_digest({"brief": "frozen"}),
              "source_generation_digest": canonical_digest({"source_generation": 7}), **value["source"]}
    probe = seal({"schema_version": RIGID_RECEIPT_SCHEMA, "status": "passed",
        "checks": dict.fromkeys(REQUIRED_CHECKS["rigid_pick_place"], True),
        "proof_boundary": {"robot_embodiment_used": False},
        "learned_policy_evaluation_admitted": True,
        "configured_scene_revision_digest": value["scene_revision_digest"],
        "task_success_contract_digest": value["task_success_contract_digest"]}, "receipt_digest")
    value["robot_free_probe_digest"] = probe["receipt_digest"]
    candidate = value["candidates"][0]
    policy_spec = seal({"schema_version": "native_task_arena_policy_canary_execution_spec.v1",
        "candidate_id": candidate["candidate_id"], "evaluation_backend": "robolab.v1",
        "policy_interface_id": "groot_n17_droid.v1", "execution_authority": "internal_policy_canary_unqualified",
        "claim_ceiling": "diagnostic_policy_execution", "ranking_permitted": False,
        "qualification_permitted": False, "scene_promotion_permitted": False,
        "policy_endpoint": {"host": "127.0.0.1", "port": 8000}, "policy_spec": {"candidate_id": candidate["candidate_id"]},
        "checkpoint_digest": canonical_digest({"value": "checkpoint"}),
        "runtime_identity_digest": candidate["execution_spec"]["runtime_digest"],
        "task_success_contract": _activation()["task_success_contract"],
        "task_success_contract_digest": value["task_success_contract_digest"], "prompt": "Move the object",
        "max_policy_queries": options.get("issued_max_policy_queries", 10), "open_loop_horizon": 8}, "execution_spec_digest")
    policy_record = record(output / "policy-spec.json", policy_spec)
    candidate["execution_spec"]["policy_execution_spec"] = policy_record
    candidate["admission_digest"] = canonical_digest(candidate["execution_spec"])
    compiled_scene = {"schema_version": "test.compiled_scene.v1", "scene_revision_digest": value["scene_revision_digest"],
        "task_success_contract_digest": value["task_success_contract_digest"]}
    value["scene_inputs"] = record(root / "compiled_scene.json", compiled_scene)
    seal(value, "plan_digest")
    inner = native_result(value)
    authority, runtime_inputs = _authority(output)
    cells = [{**cell, "control_diagnostic": {"mode": "nonblocking_diagnostic_pending",
             "typed_gap": "controls_pending_at_submission", "policy_execution_blocked": False}}
             for cell in value["candidates"][0]["scenario_cells"]]
    activation = seal({"schema_version": "task_evaluation_policy_campaign_activation.v1", "run_id": authority["run_id"],
        "run_kind": BETA_RUN_KIND, "claim_ceiling": "diagnostic_policy_execution", "candidate_ids": inner["candidate_ids"],
        "campaign_units": cells}, "activation_digest")
    activation_record = record(output / "activation.json", activation)
    runtime_inputs.update(run_kind=BETA_RUN_KIND, candidate_ids=inner["candidate_ids"],
        beta_screening_admission=inner["beta_screening_admission"], cells=cells,
        activation_digest=activation["activation_digest"],
        **{key: inner[key] for key in ("plan_digest", "configuration_digest", "scene_revision_digest")})
    seal(runtime_inputs, "runtime_inputs_digest")
    runtime_record = record(output / "runtime-inputs.json", runtime_inputs)
    authority.update(run_kind=BETA_RUN_KIND, candidate_ids=inner["candidate_ids"], learned_policy_rollout_count=10,
                     beta_screening_admission=inner["beta_screening_admission"],
                     runtime_inputs=runtime_record, runtime_inputs_digest=runtime_inputs["runtime_inputs_digest"],
                     activation_manifest=activation_record,
                     hard_cap_usd=options.get("issued_cap_usd", 4),
                     hard_ttl_seconds=options.get("issued_ttl_seconds", value["candidates"][0]["execution_spec"]["hard_ttl_seconds"]),
                     **{key: inner[key] for key in ("plan_digest", "configuration_digest", "scene_revision_digest")})
    seal(authority, "authority_digest")
    bundle = {"schema_version": PROVIDER_BUNDLE_SCHEMA_VERSION, "status": "ready",
        "execution_mode": "blueprint_beta_screening_session", "run_kind": BETA_RUN_KIND,
        "claim_ceiling": "diagnostic_policy_execution", "candidate_ids": inner["candidate_ids"],
        "episodes_per_policy": 10, "learned_policy_rollout_count": 10, "maximum_provider_allocations": 1,
        "retry_cap": 0, "candidate_policy_queried": False, "expected_output_filename": PROVIDER_RESULT_FILENAME,
        **{key: authority[key] for key in ("runtime_inputs_digest", "authority_digest", "task_success_contract_digest")},
        "execution_spec_digests": {candidate["candidate_id"]: policy_spec["execution_spec_digest"]}}
    seal(bundle, "input_digest")
    archive_path = output / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, payload in {
            "provider_runtime/adp_arena_provider_manifest.json": bundle,
            "provider_runtime/runtime_inputs/policy_canary_session_authority.json": authority,
            "provider_runtime/runtime_inputs/policy_canary_runtime_inputs.json": runtime_inputs,
            f"provider_runtime/runtime_inputs/policy_execution_spec.{candidate['candidate_id']}.json": policy_spec,
        }.items():
            archive.writestr(name, json.dumps(payload))
    archive_record = record(archive_path)
    bundle = {**bundle, "bundle_path": str(archive_path), "bundle_size_bytes": archive_record["size_bytes"],
              "bundle_sha256": archive_record["sha256"]}
    for index, row in enumerate(inner["episodes"]):
        episode_id = f"episode-{index}"
        media_root = output / "episodes"
        observations = []
        for observation_index, kind in enumerate(("policy-input", "terminal-observation")):
            camera_ids = ["external", "wrist"] if kind == "policy-input" else ["external", "wrist", "overview"]
            observations.append(visual.persist_multicamera_observation(
                {camera: np.full((32, 64, 3), 40 + index, dtype=np.uint8) for camera in camera_ids},
                output_dir=media_root, episode_id=episode_id, observation_index=observation_index,
                kind=kind, timestamp_ns=observation_index + 1, simulation_time_s=observation_index / 10,
                calibrations={camera: _calibration() for camera in camera_ids},
                source_devices={camera: "cpu" for camera in camera_ids},
                synchronizations={camera: {"host_bytes_ready": True, "method": "cpu_synchronous"} for camera in camera_ids},
            ))
        visual_evidence, media_artifacts = visual.finalize_manipulation_evaluation_visual_evidence(
            output_dir=media_root, episode_id=episode_id, identity={"candidate_id": row["candidate_id"]},
            policy_input_observations=observations[:1], terminal_observation=observations[1])
        episode = {"score": row["episode"]["score"], "visual_evidence": visual_evidence,
            "media_artifacts": media_artifacts, "commanded_actions": [[0.1, 0.2]],
            "policy_queries": options.get("observed_policy_queries", 1),
            "queries": [{"query_index": n} for n in range(options.get("observed_policy_queries", 1))],
            "motion_evidence": {"actions_reached_robot": True, "arm_moved": True},
            "state_trace": [{"step": 1}], "contact_force_evidence": [{"force": 1.0}],
            "task_object_trajectory": [[0.1, 0.2, 0.3]]}
        row["episode"] = episode
        for field, content in (("lossless_frame_manifest_digest", visual_evidence),
            ("review_video_digest", media_artifacts), ("returned_action_sequence_digest", episode["commanded_actions"]),
            ("action_delivery_readback_digest", episode["motion_evidence"]), ("state_trace_digest", episode["state_trace"]),
            ("contact_force_digest", episode["contact_force_evidence"]),
            ("task_object_trajectory_digest", episode["task_object_trajectory"]), ("deterministic_score_digest", episode["score"])):
            row[field] = canonical_digest({"value": content})
        evidence = {}
        for role, content in (("episode_receipt", episode), ("reset_state", {"reset": True}),
            ("policy_query_receipt", {"candidate_policy_queried": True, "policy_queries": episode["policy_queries"],
                                      "queries": episode["queries"]}), ("score_receipt", episode["score"]),
            ("action_sequence", episode["commanded_actions"]), ("action_delivery_readback", episode["motion_evidence"]),
            ("state_trace", episode["state_trace"]), ("contact_force_trace", episode["contact_force_evidence"]),
            ("task_object_trajectory", episode["task_object_trajectory"])):
            evidence[role] = record(media_root / f"{episode_id}.{role}.json", content)
        for role, match in (("frame_manifest", "frame_manifest"), ("review_video", "video")):
            item = next(item for item in media_artifacts if match in item["role"])
            evidence[role] = record(media_root / item["relative_path"])
        row["evidence_artifacts"] = evidence
    seal(inner, "result_digest")
    prepared = {"authority": record(output / "authority.json", authority),
        "bundle_receipt": record(output / "bundle.json", bundle), "candidate_id": inner["candidate_ids"][0],
        "run_id": authority["run_id"], "plan_digest": inner["plan_digest"],
        "configuration_digest": inner["configuration_digest"]}
    seal(prepared, "prepared_digest")
    adapter = {"vast_instance_ids": [7001], "bundle_sha256": bundle["bundle_sha256"],
        "authorization_consumption": consume_session_authority_once(authority, consumption_path=output / "consumed.json"),
        "continuing_spend_from_this_run": False, "native_control_result_digest": inner["result_digest"],
        "provider_closeout": {"provider_zero_confirmed": True, "warm_session_retained": False,
            "all_staged_objects_absent": True,
            "adapter_result": record(output / "raw_adapter.json", {"vast_instance_ids": [7001]}),
            "teardown_manifest": record(output / "teardown.json", {"vast_instance_ids": [7001]})}}
    adapter_path = output / "allocator_result.json"
    completed_seconds = options.get("completed_seconds", 40)
    closeout_epoch = datetime.fromisoformat(stamp(options.get("closeout_seconds", completed_seconds + 2))).timestamp()
    monkeypatch.setattr(closeout, "time", SimpleNamespace(time=lambda: closeout_epoch))
    class AtTimeProvider(Provider):
        def billable_inventory(self, **kwargs):
            return {**super().billable_inventory(**kwargs), "observed_at_epoch": closeout_epoch}
    receipt = closeout.collect_scoped_closeout(adapter=adapter, prepared=prepared, adapter_path=adapter_path, provider=AtTimeProvider())
    result = closeout.join_scoped_session_closeout(inner=inner, adapter=adapter, prepared=prepared,
        receipt=receipt, adapter_path=adapter_path)
    attempt = seal({"schema_version": acceptance.ATTEMPT_SCHEMA, "attempt_id": "attempt-1", "frozen_at": stamp(0),
        "start_event_id": "upload-event-1", "declared_human_boundaries": ["customer_fact_confirmation"],
        "source": source, "envelope": {"task_family": "rigid_pick_place", "backend": "robolab.v1",
            "robot_id": "franka", "policy_interface": "droid", "camera_ids": ["external", "wrist", "overview"]},
        "releases": {"pipeline_commit": "a" * 40, "webapp_commit": "b" * 40,
            "pipeline_image_digest": "sha256:" + "1" * 64, "webapp_image_digest": "sha256:" + "2" * 64},
        "candidate_runtime_identities": {result["candidate_ids"][0]: {
            "configuration_digest": result["configuration_digest"],
            "runtime_identity_digest": result["episodes"][0]["runtime_identity_digest"],
            "max_policy_queries": options.get("frozen_max_policy_queries", 10), "open_loop_horizon": 8,
            "policy_execution_spec_digest": policy_spec["execution_spec_digest"],
            "worker_image_digest": "sha256:" + "3" * 64}}}, "attempt_digest")
    publication = screening_publication(plan=value, state={"revision": 2, "updated_at": stamp(completed_seconds + 10),
        "candidates": {result["candidate_ids"][0]: {"status": "completed", "summary": candidate_result_summary(
            result, plan=value, candidate=value["candidates"][0])}}}, budget={"currency": "USD", "limit_usd": "15",
                "committed_upper_bound_usd": "4", "spent_upper_bound_usd": "4", "reserved_usd": "0"})
    version = options.get("publication_version")
    if version is not None:
        publication.update(schema_version="beta_policy_screening_publication." + version,
            purpose="job_evaluation" if version == "v2" else "screening",
            claim_ceiling="simulation_estimate" if version == "v2" else "diagnostic_policy_execution")
        if version == "v2":
            publication.pop("recommendation_eligible", None)
        else:
            publication["recommendation_eligible"] = False
        publication["publication_digest"] = cross_runtime_canonical_digest(publication, digest_field="publication_digest")
    rows = {role: {"schema_version": "test.retained_observation.v1"} for role in acceptance.ROLES}
    site_budget = seal({"schema_version": "blueprint.beta_policy_screening_ledger.v1", "campaigns": {
        value["campaign_id"]: {"site_id": value["site_id"], "plan_digest": value["plan_digest"],
            "candidates": {candidate["candidate_id"]: {"status": "completed", "attempt": {
                "attempt_id": "screen-fixture-attempt", "maximum_cost_usd": options.get("reserved_cap_usd", "4"),
                "reservation_status": "committed", "provider_allocation_observed": True,
                "prepared": prepared, "launch_intent_at": stamp(29)}}}}}}, "ledger_digest")
    rows.update(robot_free_probe=probe, candidate_plan=value, publication=publication, compiled_scene=compiled_scene,
                site_budget=site_budget)
    acquisitions = RetainedAcquisitions()
    manifest = {"schema_version": acceptance.SCHEMA, "attempt": record(root / "frozen.json", attempt),
        "receipts": {role: record(root / f"{role}.json", content) for role, content in rows.items()},
        "executions": [{"candidate_id": result["candidate_ids"][0], "output_root": str(output),
            "result": record(output / "result.json", result), "authority": prepared["authority"],
            "runtime_inputs": runtime_record, "activation": activation_record,
            "policy_execution_spec": policy_record, "bundle_archive": archive_record,
            "bundle_receipt": prepared["bundle_receipt"], "prepared": record(output / "prepared.json", prepared),
            "adapter": record(adapter_path, adapter), "closeout": record(output / "closeout.json", receipt)}]}
    facts = {role: {} for role in acceptance.ROLES}
    facts.update(service_identity_before=attempt["releases"], service_identity_after=attempt["releases"],
        upload={"natural_upload": True}, authority={"current_consent": True, "rights_current": True,
            "grant_id": "grant-1", "budget_ledger_id": "site-budget-1", "expires_at": stamp(4000), "cap_usd": "15"},
        task_confirmation={"completed": True}, intake_qa={"completed": True},
        metric_scene={"completed": True, "metric_evidence_qualified": True}, compiled_scene={"completed": True},
        queued_ack={"plan_digest": value["plan_digest"], "current_source": True},
        site_budget={"budget_ledger_id": "site-budget-1"},
        publication_ack={"publication_digest": publication["publication_digest"], "revision": 2, "current_source": True, "persisted": True},
        customer_readback={"publication_digest": publication["publication_digest"], "revision": 2, "current_source": True,
            "persisted": True, "displayed": True, "result_digests": {result["candidate_ids"][0]: result["result_digest"]},
            "webapp_commit": attempt["releases"]["webapp_commit"]},
        event_history={"coverage_complete": True, "coverage_started_at": stamp(0), "coverage_ended_at": stamp(completed_seconds + 50),
            "events": [{"kind": "upload_start", "at": stamp(10), "attempt_id": "attempt-1"}]})
    times = dict(zip(acceptance.ROLES, (1, 10, 12, 14, 16, 18, 20, 22, 24, 26,
        *(completed_seconds + offset for offset in (10, 15, 20, 30, 40, 50)))))
    def acquire(role, rec, content, at, role_facts):
        acquisitions.rows[role, rec["sha256"]] = acceptance.TrustedObservation(
            rec["sha256"], content["schema_version"], acceptance._PRODUCERS.get(role, "pipeline"),
            "production", "natural_production", "attempt-1", "upload-event-1" if role == "upload" else role + "-event",
            stamp(at), deepcopy(source), deepcopy(role_facts))
    acquire("attempt", manifest["attempt"], attempt, 0, {})
    for role, rec in manifest["receipts"].items():
        acquire(role, rec, rows[role], times[role], facts[role])
    for role in ("result", "authority", "closeout"):
        rec = manifest["executions"][0][role]
        content = json.loads(open(rec["path"]).read())
        acquire("execution_" + role, rec, content, completed_seconds + 5 if role == "closeout" else completed_seconds,
            {"started_at": stamp(30), "completed_at": stamp(completed_seconds), "worker_image_digest": "sha256:" + "3" * 64,
             "worker_image_binding_kind": "provider_container_image_digest"})
    manifest_path = root / "acceptance.json"
    manifest_path.write_text(json.dumps(manifest))
    return {"root": root, "manifest": manifest, "manifest_path": manifest_path,
            "acquisitions": acquisitions, "result": result, "attempt": attempt}


def report(retained, *, trusted=True):
    retained["manifest_path"].write_text(json.dumps(retained["manifest"]))
    return acceptance.build_production_autonomy_acceptance_report(manifest_path=retained["manifest_path"],
        acquisition_verifier=retained["acquisitions"] if trusted else None)


def change_acquisition(retained, role, *, facts=None, **changes):
    for key, observation in list(retained["acquisitions"].rows.items()):
        if key[0] == role:
            if facts:
                changes["facts"] = {**observation.facts, **facts}
            retained["acquisitions"].rows[key] = replace(observation, **changes)


def replace_attempt(retained, attempt):
    seal(attempt, "attempt_digest")
    previous = retained["manifest"]["attempt"]
    replacement = record(retained["root"] / "frozen.json", attempt)
    observed = retained["acquisitions"].rows.pop(("attempt", previous["sha256"]))
    retained["acquisitions"].rows["attempt", replacement["sha256"]] = replace(
        observed, artifact_sha256=replacement["sha256"])
    retained["manifest"]["attempt"] = replacement
    retained["attempt"] = attempt


def test_joined_contract_keeps_one_evaluation_without_physical_or_concurrency_upgrade(retained):
    result = report(retained)
    assert result["blockers"] == []
    assert result["status"] == "verified"
    assert result["milestones"]["unattended_completion_within_authorized_stages"] is True
    assert result["milestones"]["normal_retention_observed"] is False
    assert result["evaluation_path"] == "single_bounded_evaluation"
    assert "owner_approved_finalist_recommendation" not in result
    assert not any(result["proof_boundary"][key] for key in
                   ("physical_validation_proven", "concurrency_proven", "performs_live_actions", "grants_spending_authority"))
    assert result["costs"] == {"status": "unknown"}
    assert result["storage"] == {"status": "not_observed"}


@pytest.mark.parametrize("retained", [{"publication_version": "v1"}, {"publication_version": "v2"}], indirect=True)
def test_exact_existing_publication_versions_preserve_native_session_contract(retained):
    result = report(retained)
    assert result["status"] == "verified", result["blockers"]
    assert result["milestones"]["unattended_completion_within_authorized_stages"] is True
    assert retained["result"]["claim_ceiling"] == "diagnostic_policy_execution"
    publication = json.loads(Path(retained["manifest"]["receipts"]["publication"]["path"]).read_text())
    assert result["publication_contract"] == {key: publication[key] for key in ("schema_version", "purpose", "claim_ceiling")}


def replace_publication(retained, change):
    previous = retained["manifest"]["receipts"]["publication"]
    value = json.loads(Path(previous["path"]).read_text())
    change(value)
    value["publication_digest"] = cross_runtime_canonical_digest(value, digest_field="publication_digest")
    rec = record(Path(previous["path"]), value)
    retained["manifest"]["receipts"]["publication"] = rec
    observed = retained["acquisitions"].rows.pop(("publication", previous["sha256"]))
    retained["acquisitions"].rows["publication", rec["sha256"]] = replace(observed,
        artifact_sha256=rec["sha256"], schema_version=value["schema_version"])
    for role in ("publication_ack", "customer_readback"):
        change_acquisition(retained, role, facts={"publication_digest": value["publication_digest"]})


@pytest.mark.parametrize("retained", [{"publication_version": "v1"}, {"publication_version": "v2"}], indirect=True)
def test_single_evaluation_requires_no_new_customer_or_finalist_approval(retained):
    attempt = deepcopy(retained["attempt"])
    attempt["declared_human_boundaries"] = []
    replace_attempt(retained, attempt)
    assert report(retained)["status"] == "verified"
    attempt["declared_human_boundaries"] = ["site_finalist_approval"]
    replace_attempt(retained, attempt)
    assert "declared_boundaries_invalid" in report(retained)["blockers"]


@pytest.mark.parametrize("retained,change", [
    ({"publication_version": "v1"}, lambda value: value.pop("recommendation_eligible")),
    ({"publication_version": "v1"}, lambda value: value.update(purpose="job_evaluation")),
    ({"publication_version": "v1"}, lambda value: value.update(claim_ceiling="simulation_estimate")),
    ({"publication_version": "v2"}, lambda value: value.update(recommendation_eligible=False)),
    ({"publication_version": "v2"}, lambda value: value.update(purpose="screening")),
    ({"publication_version": "v2"}, lambda value: value.update(claim_ceiling="diagnostic_policy_execution")),
    ({"publication_version": "v2"}, lambda value: value.update(claim_ceiling="physical_validation")),
    ({"publication_version": "v2"}, lambda value: value.update(schema_version="beta_policy_screening_publication.v3")),
    ({"publication_version": "v2"}, lambda value: value["candidates"][0].update(episodes_failed=11)),
    ({"publication_version": "v2"}, lambda value: value["budget"].update(limit_usd="16")),
], indirect=["retained"])
def test_resealed_publication_may_not_mix_versions_or_weaken_counts_and_caps(retained, change):
    replace_publication(retained, change)
    assert "publication_contract_invalid" in report(retained)["blockers"]


@pytest.mark.parametrize("field", ["campaign_id", "catalog_digest", "scenario_plan_digest"])
def test_publication_exact_native_plan_identity_is_preserved(retained, field):
    replace_publication(retained, lambda value: value.update({field: "other-campaign" if field == "campaign_id" else "sha256:" + "0" * 64}))
    assert "publication_binding_invalid" in report(retained)["blockers"]


def test_valid_publication_counts_must_still_join_actual_independent_native_outcomes(retained):
    def changed(value):
        row = value["candidates"][0]
        row.update(episodes_succeeded=0, episodes_failed=10, success_rate=0)
    replace_publication(retained, changed)
    assert "publication_result_mismatch" in report(retained)["blockers"]


def test_publication_candidate_configuration_cannot_rebind_native_result(retained):
    replace_publication(retained, lambda value: value["candidates"][0].update(configuration_digest="sha256:" + "0" * 64))
    assert "publication_candidate_binding_mismatch" in report(retained)["blockers"]


def test_complete_job_json_without_authenticated_acquisition_cannot_attest_itself(retained):
    result = report(retained, trusted=False)
    assert result["status"] == "not_proven"
    assert not any(result["milestones"].values())
    assert "upload:authenticated_acquisition_not_proven" in result["blockers"]
    retained["manifest"]["trusted_observations"] = {"upload": {"verified": True}}
    assert "job_controlled_acquisition_refused" in report(retained)["blockers"]


def test_unavailable_trusted_acquisition_stays_unproven_and_sanitized(retained):
    def unavailable(*_):
        raise RuntimeError("private unavailable-credential diagnostic")
    result = acceptance.build_production_autonomy_acceptance_report(manifest_path=retained["manifest_path"],
                                                                  acquisition_verifier=unavailable)
    assert result["status"] == "not_proven"
    assert "upload:authenticated_acquisition_unavailable" in result["blockers"]
    assert "private unavailable-credential" not in json.dumps(result)


@pytest.mark.parametrize("role,changes,reason", [
    ("customer_readback", {"source": {"tenant_id": "other-tenant"}}, "source_binding_mismatch"),
    ("customer_readback", {"source": {"capture_id": "other-capture"}}, "source_binding_mismatch"),
    ("customer_readback", {"source": {"technical_brief_digest": "sha256:" + "0" * 64}}, "source_binding_mismatch"),
    ("customer_readback", {"source": {"scene_revision_digest": "sha256:" + "0" * 64}}, "source_binding_mismatch"),
    ("upload", {"origin": "hermetic_fixture"}, "production_origin_not_proven"),
    ("customer_readback", {"environment": "staging"}, "production_origin_not_proven"),
    ("customer_readback", {"producer": "job_payload"}, "acquisition_binding_invalid"),
])
def test_wrong_tenant_source_fixture_or_producer_never_passes(retained, role, changes, reason):
    if "source" in changes:
        changes["source"] = {**retained["attempt"]["source"], **changes["source"]}
    change_acquisition(retained, role, **changes)
    assert role + ":" + reason in report(retained)["blockers"]


@pytest.mark.parametrize("role,facts,reason", [
    ("queued_ack", {"plan_digest": "sha256:" + "0" * 64}, "queued_ack_stale_or_mismatched"),
    ("publication_ack", {"revision": 1}, "publication_ack:stale_or_unpersisted"),
    ("customer_readback", {"persisted": False}, "customer_readback:stale_or_unpersisted"),
    ("customer_readback", {"displayed": False}, "current_customer_display_not_proven"),
    ("authority", {"current_consent": False}, "current_authority_not_proven"),
    ("authority", {"cap_usd": "3"}, "screening_budget_binding_invalid"),
    ("execution_result", {"started_at": stamp(25)}, "execution_started_before_current_ack_or_after_publication"),
    ("service_identity_after", {"pipeline_commit": "c" * 40}, "release_changed_during_attempt"),
])
def test_stale_acks_revocation_early_start_and_deployment_refuse(retained, role, facts, reason):
    change_acquisition(retained, role, facts=facts)
    assert reason in report(retained)["blockers"]


@pytest.mark.parametrize("kind", ["manual_retry", "manual_stage_advance", "artifact_injection", "one_off_rescue", "deployment"])
def test_intervention_history_cannot_hide_rescue(retained, kind):
    change_acquisition(retained, "event_history", facts={"events": [
        {"kind": "upload_start", "at": stamp(10), "attempt_id": "attempt-1"},
        {"kind": kind, "at": stamp(35), "attempt_id": "attempt-1", "planned": True}]})
    result = report(retained)
    assert "unplanned_intervention_invalidates_unattended_claim" in result["blockers"]
    assert result["milestones"]["unattended_completion_within_authorized_stages"] is False
    assert result["unplanned_intervention_count"] == 1


def test_duplicate_start_and_incomplete_history_refuse(retained):
    history = [{"kind": "upload_start", "at": stamp(10), "attempt_id": "attempt-1"}]
    change_acquisition(retained, "event_history", facts={"events": history * 2})
    assert "duplicate_or_missing_start_event" in report(retained)["blockers"]
    change_acquisition(retained, "event_history", facts={"coverage_complete": False})
    assert "complete_intervention_history_not_proven" in report(retained)["blockers"]


def test_restart_and_ack_outage_rehearsal_only_rechecks_original_bytes(retained):
    original = {p: p.read_bytes() for p in retained["root"].rglob("*") if p.is_file() and p != retained["manifest_path"]}
    change_acquisition(retained, "event_history", facts={"events": [
        {"kind": "upload_start", "at": stamp(10), "attempt_id": "attempt-1"},
        {"kind": "process_restart", "at": stamp(29), "attempt_id": "attempt-1"},
        {"kind": "delivery_outage", "at": stamp(51), "attempt_id": "attempt-1"},
        {"kind": "automatic_retry", "at": stamp(55), "attempt_id": "attempt-1"},
        {"kind": "delivery_recovered", "at": stamp(59), "attempt_id": "attempt-1"}]})
    missing = retained["manifest"]["receipts"].pop("publication_ack")
    assert report(retained)["status"] == "not_proven"
    retained["manifest"]["receipts"]["publication_ack"] = missing
    first = report(retained)
    second = report(retained)
    assert first == second
    assert first["status"] == "verified"
    assert first["retry_count"] == first["restart_count"] == 1
    assert first["unplanned_intervention_count"] == 0
    assert all(p.read_bytes() == value for p, value in original.items())


@pytest.mark.parametrize("fault", ["missing_media", "changed_frame", "missing_own_teardown", "corrupt_digest", "symlink"])
def test_digest_media_and_own_teardown_negatives(retained, fault):
    entry = retained["manifest"]["executions"][0]
    row = retained["result"]["episodes"][0]
    if fault in {"missing_media", "changed_frame"}:
        item = next(item for item in row["episode"]["media_artifacts"] if item["role"] == "policy_input_camera_frame")
        target = retained["root"] / "candidate-output" / "episodes" / item["relative_path"]
        target.unlink()
        if fault == "changed_frame":
            target.write_bytes(b"changed")
    elif fault == "missing_own_teardown":
        entry.pop("closeout")
    elif fault == "corrupt_digest":
        entry["result"]["sha256"] = "sha256:" + "0" * 64
    else:
        target = retained["root"] / "alias.json"
        target.symlink_to(entry["result"]["path"])
        entry["result"]["path"] = str(target)
    result = report(retained)
    assert result["status"] == "not_proven"
    assert not result["milestones"]["supported_production_completion"]


def test_wrong_acquisition_digest_and_missing_native_bytes_refuse(retained):
    change_acquisition(retained, "customer_readback", artifact_sha256="sha256:" + "0" * 64)
    assert "customer_readback:acquisition_binding_invalid" in report(retained)["blockers"]
    retained["manifest"]["executions"][0]["result"]["path"] = "../foreign.json"
    assert "artifact_path_invalid" in report(retained)["blockers"]


@pytest.mark.parametrize("kind", ["source_revocation", "consent_revocation", "run_cancellation", "technical_source_changed", "authority_expired"])
def test_source_revocation_inside_execution_never_completes_the_frozen_attempt(retained, kind):
    change_acquisition(retained, "event_history", facts={"events": [
        {"kind": "upload_start", "at": stamp(10), "attempt_id": "attempt-1"},
        {"kind": kind, "at": stamp(35), "attempt_id": "attempt-1"}]})
    result = report(retained)
    assert "authority_ended_during_frozen_attempt" in result["blockers"]
    assert result["milestones"]["supported_production_completion"] is False
    assert result["milestones"]["unattended_completion_within_authorized_stages"] is False


def test_worker_image_is_frozen_separately_from_the_service_image(retained):
    worker_image = "sha256:" + "3" * 64
    attempt = deepcopy(retained["attempt"])
    candidate_id = retained["manifest"]["executions"][0]["candidate_id"]
    attempt["candidate_runtime_identities"][candidate_id]["worker_image_digest"] = worker_image
    replace_attempt(retained, attempt)
    change_acquisition(retained, "execution_result", facts={"runtime_image_digest": worker_image,
        "worker_image_digest": worker_image, "worker_image_binding_kind": "provider_container_image_digest"})
    result = report(retained)
    assert result["status"] == "verified", result["blockers"]


@pytest.mark.parametrize("fault,reason", [
    ("expected_missing", "candidate_worker_image_expected_not_proven"),
    ("acquisition_missing", "worker_image_acquisition_not_proven"),
    ("wrong_image", "execution_worker_image_mismatch"),
    ("runtime_digest_relabelled", "worker_image_acquisition_not_proven"),
])
def test_worker_image_unknown_mismatched_or_runtime_label_never_proves_image(retained, fault, reason):
    if fault == "expected_missing":
        attempt = deepcopy(retained["attempt"])
        for frozen in attempt["candidate_runtime_identities"].values():
            frozen.pop("worker_image_digest")
        replace_attempt(retained, attempt)
    elif fault == "acquisition_missing":
        change_acquisition(retained, "execution_result", facts={"worker_image_digest": None})
    elif fault == "wrong_image":
        change_acquisition(retained, "execution_result", facts={"worker_image_digest": "sha256:" + "4" * 64})
    else:
        change_acquisition(retained, "execution_result", facts={"worker_image_binding_kind": "native_container_identity_digest"})
    result = report(retained)
    assert reason in result["blockers"]
    assert result["milestones"]["supported_production_completion"] is False


def test_actual_nested_teardown_bytes_are_required_for_resource_reconciliation(retained):
    entry = retained["manifest"]["executions"][0]
    adapter = json.loads(open(entry["adapter"]["path"]).read())
    prepared = json.loads(open(entry["prepared"]["path"]).read())
    receipt = json.loads(open(entry["closeout"]["path"]).read())
    assert closeout.scoped_closeout_verified(receipt, adapter=adapter, prepared=prepared,
                                            adapter_path=retained["root"] / "candidate-output/allocator_result.json")
    Path(adapter["provider_closeout"]["teardown_manifest"]["path"]).unlink()
    assert not closeout.scoped_closeout_verified(receipt, adapter=adapter, prepared=prepared,
                                                adapter_path=retained["root"] / "candidate-output/allocator_result.json")
    result = report(retained)
    assert result["status"] == "not_proven"
    assert result["milestones"]["resource_reconciliation"] is False


@pytest.mark.parametrize("retained", [{"issued_cap_usd": 1000}], indirect=True)
def test_actual_issued_session_cap_cannot_exceed_its_site_plan_reservation(retained):
    result = report(retained)
    assert "issued_session_bounds_mismatch" in result["blockers"]
    assert result["milestones"]["supported_production_completion"] is False


@pytest.mark.parametrize("retained", [{"closeout_seconds": 5}], indirect=True)
def test_reacquiring_an_old_absence_receipt_does_not_prove_post_execution_closeout(retained):
    result = report(retained)
    assert "provider_absence_observed_before_execution_completed" in result["blockers"]
    assert result["milestones"]["resource_reconciliation"] is False


@pytest.mark.parametrize("retained,reason", [
    ({"issued_ttl_seconds": 7200}, "issued_session_bounds_mismatch"),
    ({"reserved_cap_usd": "1"}, "issued_session_site_reservation_mismatch"),
    ({"issued_max_policy_queries": 11}, "issued_policy_query_or_runtime_bounds_mismatch"),
    ({"observed_policy_queries": 11}, "issued_policy_query_bound_exceeded_or_unproven"),
    ({"closeout_seconds": 48}, "provider_absence_acquisition_time_invalid"),
], indirect=["retained"])
def test_issued_runtime_queries_reservation_and_observation_bounds(retained, reason):
    result = report(retained)
    assert reason in result["blockers"]
    assert result["milestones"]["supported_production_completion"] is False


def test_capture_authority_and_actual_budget_ledger_are_both_required(retained):
    change_acquisition(retained, "authority", facts={"expires_at": stamp(1000)})
    assert "issued_session_runtime_exceeds_capture_authority" in report(retained)["blockers"]
    change_acquisition(retained, "site_budget", facts={"budget_ledger_id": "other-site-budget"})
    assert "screening_budget_binding_invalid" in report(retained)["blockers"]


@pytest.mark.parametrize("retained,within_issued_runtime", [
    ({"completed_seconds": 1830}, True),
    ({"completed_seconds": 3640}, False),
], indirect=["retained"])
def test_observed_completion_must_fit_actual_issued_runtime_even_with_later_valid_closeout(retained, within_issued_runtime):
    result = report(retained)
    assert (result["status"] == "verified") is within_issued_runtime
    assert result["milestones"]["unattended_completion_within_authorized_stages"] is within_issued_runtime
    assert result["milestones"]["resource_reconciliation"] is within_issued_runtime
    if not within_issued_runtime:
        assert "observed_execution_exceeds_issued_runtime" in result["blockers"]


def test_deleted_sealed_bundle_bytes_cannot_prove_runtime_or_query_authority(retained):
    Path(retained["manifest"]["executions"][0]["bundle_archive"]["path"]).unlink()
    assert report(retained)["status"] == "not_proven"


def test_offline_closeout_rechecks_nested_bytes_without_refreshing_provider_state(retained, monkeypatch):
    entry = retained["manifest"]["executions"][0]
    adapter, prepared, receipt = [json.loads(Path(entry[name]["path"]).read_text())
                                  for name in ("adapter", "prepared", "closeout")]
    future = datetime.fromisoformat(stamp(1000)).timestamp()
    monkeypatch.setattr(closeout, "time", SimpleNamespace(time=lambda: future))
    monkeypatch.setattr(closeout, "collect_scoped_closeout", lambda **_: pytest.fail("no provider recollection"))
    assert not closeout.scoped_closeout_verified(receipt, adapter=adapter, prepared=prepared,
                                                adapter_path=Path(entry["adapter"]["path"]))
    assert closeout.retained_scoped_closeout_verified(receipt, adapter=adapter, prepared=prepared,
        adapter_path=Path(entry["adapter"]["path"]), record_reader=acceptance._retained_reader(retained["root"], entry),
        observed_at_epoch=receipt["observed_at_epoch"])
    assert report(retained)["status"] == "verified"


def test_portable_nested_copies_keep_original_receipt_bytes_and_binding(retained):
    entry = retained["manifest"]["executions"][0]
    adapter = json.loads(Path(entry["adapter"]["path"]).read_text())
    copies = {}
    for name in ("adapter_result", "teardown_manifest"):
        original = Path(adapter["provider_closeout"][name]["path"])
        copied = retained["root"] / "portable-copy" / (name + ".json")
        copied.parent.mkdir(exist_ok=True)
        copied.write_bytes(original.read_bytes())
        copies[name] = record(copied)
        original.unlink()
    entry["retained_records"] = copies
    assert report(retained)["status"] == "verified"
    copies["teardown_manifest"]["size_bytes"] += 1
    assert "owned_resource_closeout_unproven" in report(retained)["blockers"]


def test_release_report_option_preserves_legacy_scope_and_requires_trust(retained):
    from scripts.run_sim_only_beta_release_gate import build_release_gate_report
    from tests.test_sim_only_beta_release_gate import (
        _deployment_proof, _forwarding_report, _local_gate_report, _route_proof,
    )
    root = retained["root"]
    capture_root = root / "capture"
    local, forwarding, route, deployment = [root / (name + ".json") for name in
                                            ("local", "forwarding", "route", "deployment")]
    _local_gate_report(local)
    _forwarding_report(forwarding)
    _route_proof(route, capture_root)
    _deployment_proof(deployment)
    args = {"capture_root": capture_root, "local_gate_report_path": local,
            "forwarding_preflight_report_path": forwarding,
            "production_route_forwarding_proof_path": route,
            "production_deployment_proof_path": deployment}
    legacy = build_release_gate_report(**args)
    assert legacy["status"] == "passed"
    assert legacy["production_autonomy_acceptance"] is None
    assert legacy["proof_boundary"]["supported_production_completion"] is False
    untrusted = build_release_gate_report(**args, production_autonomy_manifest_path=retained["manifest_path"])
    assert untrusted["status"] == "passed"
    assert untrusted["ready_for_beta_release"] is True
    assert untrusted["blockers"] == legacy["blockers"] == []
    assert untrusted["gates"] == legacy["gates"]
    assert untrusted["production_autonomy_acceptance"]["status"] == "not_proven"
    assert "upload:authenticated_acquisition_not_proven" in untrusted["production_autonomy_acceptance"]["blockers"]
    assert untrusted["proof_boundary"]["production_autonomy_observation_requested"] is True
    assert untrusted["proof_boundary"]["supported_production_completion"] is False
    assert untrusted["public_claim_upgrade_allowed"] is False
    joined = build_release_gate_report(**args, production_autonomy_manifest_path=retained["manifest_path"],
                                      acquisition_verifier=retained["acquisitions"])
    assert joined["status"] == "passed"
    assert joined["proof_boundary"]["supported_production_completion"] is True
    assert joined["public_claim_upgrade_allowed"] is False


@pytest.mark.parametrize("role,facts,reason", [
    ("metric_scene", {"metric_evidence_qualified": False}, "metric_scene_qualification_not_proven"),
    ("event_history", {"coverage_complete": False}, "complete_intervention_history_not_proven"),
])
def test_release_keeps_incomplete_autonomy_observations_diagnostic(retained, role, facts, reason):
    from scripts.run_sim_only_beta_release_gate import build_release_gate_report
    from tests.test_sim_only_beta_release_gate import (
        _deployment_proof, _forwarding_report, _local_gate_report, _route_proof,
    )

    change_acquisition(retained, role, facts=facts)
    standalone = report(retained)
    assert standalone["status"] == "not_proven"
    assert reason in standalone["blockers"]
    root = retained["root"]
    capture_root = root / "capture"
    local, forwarding, route, deployment = [root / (name + ".json") for name in
                                            ("local", "forwarding", "route", "deployment")]
    _local_gate_report(local)
    _forwarding_report(forwarding)
    _route_proof(route, capture_root)
    _deployment_proof(deployment)

    result = build_release_gate_report(
        capture_root=capture_root, local_gate_report_path=local,
        forwarding_preflight_report_path=forwarding,
        production_route_forwarding_proof_path=route,
        production_deployment_proof_path=deployment,
        production_autonomy_manifest_path=retained["manifest_path"],
        acquisition_verifier=retained["acquisitions"],
    )

    assert result["status"] == "passed"
    assert result["ready_for_beta_release"] is True
    assert result["blockers"] == []
    assert len(result["gates"]) == 5
    assert result["production_autonomy_acceptance"] == standalone
    assert result["proof_boundary"]["supported_production_completion"] is False
    assert result["public_claim_upgrade_allowed"] is False


@pytest.mark.parametrize("required_proof_present", [True, False])
def test_release_cli_optional_manifest_preserves_required_proof_failures(retained, required_proof_present):
    from scripts.run_sim_only_beta_release_gate import main
    from tests.test_sim_only_beta_release_gate import (
        _deployment_proof, _forwarding_report, _local_gate_report, _route_proof,
    )

    root = retained["root"]
    capture_root = root / "capture"
    local, forwarding, route, deployment, output = [root / (name + ".json") for name in
        ("local", "forwarding", "route", "deployment", "release")]
    missing = [] if required_proof_present else ["full_trace_package"]
    _local_gate_report(
        local, sim_only_beta_requirements_satisfied=required_proof_present,
        sim_only_beta_blocked_requirement_ids=missing,
        requirements=[{"requirement_id": "full_trace_package", "sim_only_beta_required": True,
                       "passed": required_proof_present, "blockers": missing}],
    )
    _forwarding_report(forwarding)
    _route_proof(route, capture_root)
    _deployment_proof(deployment)

    exit_code = main([
        "--capture-root", str(capture_root), "--local-gate-report", str(local),
        "--forwarding-preflight-report", str(forwarding),
        "--production-route-forwarding-proof", str(route),
        "--production-deployment-proof", str(deployment),
        "--production-autonomy-manifest", str(retained["manifest_path"]),
        "--output-path", str(output),
    ])

    result = json.loads(output.read_text())
    assert exit_code == (0 if required_proof_present else 1)
    assert result["ready_for_beta_release"] is required_proof_present
    assert result["status"] == ("passed" if required_proof_present else "blocked")
    expected = [] if required_proof_present else [
        "local_sim_only_post_upload_autonomy:sim_only_beta_requirement_full_trace_package_not_complete"]
    assert result["blockers"] == expected
    assert len(result["gates"]) == 5
    observation = result["production_autonomy_acceptance"]
    assert observation["status"] == "not_proven"
    assert "upload:authenticated_acquisition_not_proven" in observation["blockers"]
    assert result["proof_boundary"]["production_autonomy_observation_requested"] is True
    assert result["proof_boundary"]["supported_production_completion"] is False
    assert result["public_claim_upgrade_allowed"] is False
