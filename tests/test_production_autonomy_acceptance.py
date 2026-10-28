"""Hermetic receipt contracts; the synthetic acquisitions here prove no live run."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json

import numpy as np
import pytest

from blueprint_pipeline import beta_policy_screening_closeout as closeout
from blueprint_pipeline import episode_visual_evidence as visual
from blueprint_pipeline import production_autonomy_acceptance as acceptance
from blueprint_pipeline.beta_policy_screening_result import candidate_result_summary, screening_publication
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_policy_canary_session import (
    BETA_RUN_KIND, PROVIDER_BUNDLE_SCHEMA_VERSION, PROVIDER_RESULT_FILENAME,
    consume_session_authority_once,
)
from blueprint_pipeline.robot_free_probe_gate import REQUIRED_CHECKS, RIGID_RECEIPT_SCHEMA
from tests.test_beta_policy_screening_closeout import Provider
from tests.test_beta_policy_screening_coordinator import native_result, plan, seal
from tests.test_episode_visual_evidence import _calibration
from tests.test_native_task_arena_policy_canary_session import _authority


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
def retained(tmp_path, monkeypatch):
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
                     **{key: inner[key] for key in ("plan_digest", "configuration_digest", "scene_revision_digest")})
    seal(authority, "authority_digest")
    bundle = {"schema_version": PROVIDER_BUNDLE_SCHEMA_VERSION, "status": "ready",
        "execution_mode": "blueprint_beta_screening_session", "run_kind": BETA_RUN_KIND,
        "claim_ceiling": "diagnostic_policy_execution", "candidate_ids": inner["candidate_ids"],
        "episodes_per_policy": 10, "learned_policy_rollout_count": 10, "maximum_provider_allocations": 1,
        "retry_cap": 0, "candidate_policy_queried": False, "expected_output_filename": PROVIDER_RESULT_FILENAME,
        **{key: authority[key] for key in ("runtime_inputs_digest", "authority_digest", "task_success_contract_digest")},
        "bundle_path": "/retained-original/bundle.zip", "bundle_size_bytes": 1, "bundle_sha256": "sha256:" + "f" * 64}
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
            ("policy_query_receipt", {"candidate_policy_queried": True}), ("score_receipt", episode["score"]),
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
    receipt = closeout.collect_scoped_closeout(adapter=adapter, prepared=prepared, adapter_path=adapter_path, provider=Provider())
    result = closeout.join_scoped_session_closeout(inner=inner, adapter=adapter, prepared=prepared,
        receipt=receipt, adapter_path=adapter_path)
    attempt = seal({"schema_version": acceptance.ATTEMPT_SCHEMA, "attempt_id": "attempt-1", "frozen_at": stamp(0),
        "start_event_id": "upload-event-1", "declared_human_boundaries": ["customer_fact_confirmation"],
        "source": source, "envelope": {"task_family": "rigid_pick_place", "backend": "robolab.v1",
            "robot_id": "franka", "policy_interface": "droid", "camera_ids": ["external", "wrist", "overview"]},
        "releases": {"pipeline_commit": "a" * 40, "webapp_commit": "b" * 40,
            "pipeline_image_digest": "sha256:" + "1" * 64, "webapp_image_digest": "sha256:" + "2" * 64}}, "attempt_digest")
    publication = screening_publication(plan=value, state={"revision": 2, "updated_at": stamp(50),
        "candidates": {result["candidate_ids"][0]: {"status": "completed", "summary": candidate_result_summary(
            result, plan=value, candidate=value["candidates"][0])}}}, budget={})
    rows = {role: {"schema_version": "test.retained_observation.v1"} for role in acceptance.ROLES}
    rows.update(robot_free_probe=probe, candidate_plan=value, publication=publication, compiled_scene=compiled_scene)
    acquisitions = RetainedAcquisitions()
    manifest = {"schema_version": acceptance.SCHEMA, "attempt": record(root / "frozen.json", attempt),
        "receipts": {role: record(root / f"{role}.json", content) for role, content in rows.items()},
        "executions": [{"candidate_id": result["candidate_ids"][0], "output_root": str(output),
            "result": record(output / "result.json", result), "authority": prepared["authority"],
            "runtime_inputs": runtime_record, "activation": activation_record,
            "bundle_receipt": prepared["bundle_receipt"], "prepared": record(output / "prepared.json", prepared),
            "adapter": record(adapter_path, adapter), "closeout": record(output / "closeout.json", receipt)}]}
    facts = {role: {} for role in acceptance.ROLES}
    facts.update(service_identity_before=attempt["releases"], service_identity_after=attempt["releases"],
        upload={"natural_upload": True}, authority={"current_consent": True, "rights_current": True,
            "grant_id": "grant-1", "budget_ledger_id": "site-budget-1", "expires_at": stamp(1000), "cap_usd": "15"},
        task_confirmation={"completed": True}, intake_qa={"completed": True},
        metric_scene={"completed": True, "metric_evidence_qualified": True}, compiled_scene={"completed": True},
        queued_ack={"plan_digest": value["plan_digest"], "current_source": True},
        publication_ack={"publication_digest": publication["publication_digest"], "revision": 2, "current_source": True, "persisted": True},
        customer_readback={"publication_digest": publication["publication_digest"], "revision": 2, "current_source": True,
            "persisted": True, "displayed": True, "result_digests": {result["candidate_ids"][0]: result["result_digest"]},
            "webapp_commit": attempt["releases"]["webapp_commit"]},
        event_history={"coverage_complete": True, "coverage_started_at": stamp(0), "coverage_ended_at": stamp(90),
            "events": [{"kind": "upload_start", "at": stamp(10), "attempt_id": "attempt-1"}]})
    times = dict(zip(acceptance.ROLES, (1, 10, 12, 14, 16, 18, 20, 22, 24, 26, 50, 60, 70, 80, 90)))
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
        acquire("execution_" + role, rec, content, 45 if role == "closeout" else 40,
            {"started_at": stamp(30), "completed_at": stamp(40), "runtime_image_digest": attempt["releases"]["pipeline_image_digest"]})
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


def test_joined_contract_keeps_screening_finalists_physical_and_concurrency_separate(retained):
    result = report(retained)
    assert result["blockers"] == []
    assert result["status"] == "verified"
    assert result["milestones"]["unattended_completion_within_authorized_stages"] is True
    assert result["milestones"]["normal_retention_observed"] is False
    assert result["owner_approved_finalist_recommendation"] is False
    assert not any(result["proof_boundary"][key] for key in
                   ("physical_validation_proven", "concurrency_proven", "performs_live_actions", "grants_spending_authority"))
    assert result["costs"] == {"status": "unknown"}
    assert result["storage"] == {"status": "not_observed"}


def test_complete_job_json_without_authenticated_acquisition_cannot_attest_itself(retained):
    result = report(retained, trusted=False)
    assert result["status"] == "not_proven"
    assert not any(result["milestones"].values())
    assert "upload:authenticated_acquisition_not_proven" in result["blockers"]
    retained["manifest"]["trusted_observations"] = {"upload": {"verified": True}}
    assert "job_controlled_acquisition_refused" in report(retained)["blockers"]


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
    assert untrusted["status"] == "blocked"
    assert untrusted["production_autonomy_acceptance"]["status"] == "not_proven"
    joined = build_release_gate_report(**args, production_autonomy_manifest_path=retained["manifest_path"],
                                      acquisition_verifier=retained["acquisitions"])
    assert joined["status"] == "passed"
    assert joined["proof_boundary"]["supported_production_completion"] is True
    assert joined["public_claim_upgrade_allowed"] is False
