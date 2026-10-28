"""A completed diagnostic prepares real private full inputs, without granting spend."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from blueprint_pipeline import beta_policy_finalists as finalists
from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline import beta_policy_screening_worker as worker
from blueprint_pipeline.beta_policy_screening_plan import validate_screening_plan
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.native_task_arena_policy_canary_session import validate_session_result, episodes_per_policy_for, BETA_RUN_KIND
from tests.test_beta_policy_screening_coordinator import Backend, plan, native_result, seal
from tests.test_beta_policy_screening_plan import scene


def completed(tmp_path, monkeypatch):
    root = tmp_path / "screening"
    root.mkdir()
    value = plan()
    artifact = tmp_path / "scene-inputs.json"
    artifact.write_text("{}")
    from blueprint_pipeline.beta_policy_screening_launch import _record
    value["scene_inputs"] = _record(artifact)
    seal(value, "plan_digest")
    coordinator.enqueue_screening_campaign(value, queue_root=root)
    backend = Backend({"status": "completed", "provider_allocation_performed": True,
                       "provider_zero_verified": True, "result": native_result(value)})
    coordinator.process_screening_queue(queue_root=root, executor=backend, require_publication_ack=False)
    publication = coordinator.pending_publications(queue_root=root)[0]
    monkeypatch.setenv(coordinator.QUEUE_ROOT_ENV, str(root))
    monkeypatch.setenv(finalists.ROOT_ENV, str(tmp_path / "finalists"))
    monkeypatch.setattr(worker, "require_current_source", lambda *args, **kwargs: {})
    import blueprint_pipeline.team_scene_packet_worker as packets
    import blueprint_pipeline.team_candidate_quick10_registry as registry
    monkeypatch.setattr(packets, "resolve_capture_root", lambda source: tmp_path)
    monkeypatch.setattr(registry, "profile_for_capture", lambda capture: {
        "scene_id": value["site_id"], "scored_matrix_salt": "private-matrix-salt-" + "a" * 64,
        "allowed_founder_approvers": ["owner-1"], "max_hourly_rate_usd": 0.01})
    monkeypatch.setattr(registry, "scene_facts", lambda profile: {"scene": {"task_spec": scene()["task_spec"]}})
    request = {"schema_version": finalists.SCHEMA, "campaign_id": value["campaign_id"],
               "publication_digest": publication["publication_digest"], "candidate_id": value["candidates"][0]["candidate_id"],
               "job_id": "job-1", "brief_digest": canonical_digest({"brief": 1}),
               "source_fingerprint": canonical_digest({"source": 1})}
    receipt = finalists.prepare_finalist(request, screening_root=root)
    return request, receipt, value


def authorize(receipt):
    return {"plan_digest": receipt["plan_digest"], "approved_by": "owner-1", "approval": {
        "episodes": 200, "sponsorCapUsd": 20, "expiresAtIso": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
        "maxAttempts": 1, "evidenceScope": "development_only"}}


def retained(receipt):
    return finalists._read(finalists._queue() / "requests" / (receipt["finalist_id"] + ".json"))


def full_result(record, original):
    result = native_result(original)
    value = record["plan"]
    candidate = value["candidates"][0]
    template = result["episodes"][0]
    result["episodes"] = [{**deepcopy(template), **deepcopy(cell)} for cell in candidate["scenario_cells"]]
    result.update(plan_digest=value["plan_digest"], episodes_per_policy=200, learned_policy_rollout_count=200)
    result["beta_screening_admission"].update(plan_digest=value["plan_digest"],
        admission_digest=candidate["admission_digest"], evaluation_purpose="finalist",
        owner_authority_digest=record["authority"]["authority_digest"], finalist_id=value["campaign_id"])
    return validate_session_result(seal(result, "result_digest"))


def test_completed_screening_prepares_200_private_inputs_and_owner_authority_runs_once(tmp_path, monkeypatch):
    request, receipt, original = completed(tmp_path, monkeypatch)
    record = retained(receipt)
    assert receipt["state"] == "awaiting_owner_authority" and receipt["episodes"] == 200
    assert receipt["paid_execution_requested"] is False and receipt["recommendation_eligible"] is False
    assert "scenario_cells" not in json.dumps(receipt) and "private-matrix-salt" not in json.dumps(receipt)
    cells = record["plan"]["scenario_cells"]
    assert len(cells) == 200 and {cell["family"] for cell in cells} >= {"held_out_composition", "camera_sensor"}
    assert set(cell["seed"] for cell in cells).isdisjoint(cell["seed"] for cell in original["scenario_cells"])
    backend = Backend()
    finalists.process_finalists(executor=backend, transport=lambda pub: None)
    assert backend.preparations == backend.launches == 0
    assert finalists.prepare_finalist(request)["finalist_id"] == receipt["finalist_id"]
    approval = authorize(receipt)
    authorized = finalists.authorize_finalist(receipt["finalist_id"], approval)
    assert authorized["owner_authority_digest"] == cross_runtime_canonical_digest(approval)
    record = retained(receipt)
    validate_screening_plan(record["plan"])
    backend.observation = {"status": "completed", "provider_allocation_performed": True,
                           "provider_zero_verified": True, "result": full_result(record, original)}
    publications = []
    finalists.process_finalists(executor=backend, transport=publications.append)
    assert backend.launches == 1
    finalists.process_finalists(executor=backend, transport=publications.append)
    assert backend.launches == 1
    publication = publications[-1]
    assert publication["purpose"] == "finalist" and publication["candidates"][0]["episodes_scored"] == 200
    assert publication["finalist_binding"]["job_id"] == request["job_id"]
    assert publication["recommendation_eligible"] is False


def test_lost_launch_response_reconciles_without_a_second_allocation(tmp_path, monkeypatch):
    _, receipt, _ = completed(tmp_path, monkeypatch)
    finalists.authorize_finalist(receipt["finalist_id"], authorize(receipt))
    backend = Backend(RuntimeError("response lost"))
    finalists.process_finalists(executor=backend, transport=lambda pub: None)
    assert retained(receipt)["state"] == "reconciliation_required"
    finalists.process_finalists(executor=backend, transport=lambda pub: None)
    assert backend.launches == 1 and backend.reconciliations == 1


def test_stale_source_stops_before_preparation_or_spend(tmp_path, monkeypatch):
    _, receipt, _ = completed(tmp_path, monkeypatch)
    finalists.authorize_finalist(receipt["finalist_id"], authorize(receipt))
    def stale(*args, **kwargs):
        raise ValueError("source_changed")
    monkeypatch.setattr(worker, "require_current_source", stale)
    backend = Backend()
    finalists.process_finalists(executor=backend, transport=lambda pub: None)
    assert backend.preparations == backend.launches == 0
    assert retained(receipt)["state"] == "blocked"


def test_wrong_job_or_changed_publication_cannot_prepare(tmp_path, monkeypatch):
    request, _, _ = completed(tmp_path, monkeypatch)
    for changes, code in (({"job_id": "other"}, "job_source_mismatch"),
                          ({"source_fingerprint": canonical_digest({"new": 1}), "publication_digest": canonical_digest({"changed": 1})}, "publication_changed")):
        with pytest.raises(ValueError, match=code):
            finalists.prepare_finalist({**request, **changes}, screening_root=tmp_path / "screening")


def test_diagnostic_flags_cannot_authorize_200_or_expand_the_screening_budget(tmp_path, monkeypatch):
    _, receipt, original = completed(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        episodes_per_policy_for(BETA_RUN_KIND, 200)
    unadmitted = retained(receipt)["plan"]
    with pytest.raises(ValueError, match="authority"):
        finalists.require_finalist_authority(unadmitted)
    with pytest.raises(ValueError, match="scope_invalid"):
        coordinator.enqueue_screening_campaign(unadmitted, queue_root=tmp_path / "screening")
    wrong = authorize(receipt)
    wrong["approval"]["evidenceScope"] = "simulation"
    with pytest.raises(ValueError, match="approval_invalid"):
        finalists.authorize_finalist(receipt["finalist_id"], wrong)
    wrong_actor=authorize(receipt)
    wrong_actor["approved_by"]="unadmitted-owner"
    with pytest.raises(ValueError,match="owner_not_authorized"):
        finalists.authorize_finalist(receipt["finalist_id"],wrong_actor)
    underfunded=authorize(receipt)
    underfunded["approval"]["sponsorCapUsd"]=0.01
    with pytest.raises(ValueError,match="owner_cap_insufficient"):
        finalists.authorize_finalist(receipt["finalist_id"],underfunded)
    assert retained(receipt)["state"]=="awaiting_owner_authority"
    original["budget_usd"] = "20.00"
    seal(original, "plan_digest")
    with pytest.raises(ValueError, match="site_budget_exceeded"):
        validate_screening_plan(original)


def test_full_finalist_builds_real_robolab_bundle_and_passes_provider_precreate_gate(tmp_path, monkeypatch):
    from tests.test_beta_policy_screening_launch import _plan
    from blueprint_pipeline import beta_policy_screening_launch as launch
    from blueprint_pipeline.native_task_arena_policy_canary_session import validate_session_authority
    from blueprint_pipeline.vast_provider_adapter import _blueprint_bundle_preflight
    value, _ = _plan(tmp_path, monkeypatch)
    screening_root = tmp_path / "screening-queue"
    monkeypatch.setenv(coordinator.QUEUE_ROOT_ENV, str(screening_root))
    monkeypatch.setenv(finalists.ROOT_ENV, str(tmp_path / "finalists-real"))
    coordinator.enqueue_screening_campaign(value, queue_root=screening_root)
    with coordinator._lock(screening_root):
        ledger = coordinator._ledger(screening_root)
        state = ledger["campaigns"][value["campaign_id"]]
        for cid, row in state["candidates"].items():
            row.update(status="blocked", blockers=["fixture_not_selected"])
        state["candidates"]["pi05_droid"].update(status="completed", blockers=[], summary={
            "episodes_scored":10,"episodes_succeeded":7,"episodes_failed":3,"episodes_void":0,
            "success_rate":0.7,"session_result_digest":canonical_digest({"fixture":1})})
        coordinator._save(screening_root, ledger)
    publication = coordinator.pending_publications(queue_root=screening_root)[0]
    import blueprint_pipeline.team_scene_packet_worker as packets
    monkeypatch.setattr(packets, "resolve_capture_root", lambda source: tmp_path / "capture")
    monkeypatch.setattr(worker, "require_current_source", lambda *args, **kwargs: {})
    request={"schema_version":finalists.SCHEMA,"campaign_id":value["campaign_id"],
        "publication_digest":publication["publication_digest"],"candidate_id":"pi05_droid",
        "job_id":"req1","brief_digest":canonical_digest({"brief":1}),"source_fingerprint":canonical_digest({"source":1})}
    receipt=finalists.prepare_finalist(request)
    approval=authorize(receipt)
    approval["approved_by"]="founder@blueprint"
    finalists.authorize_finalist(receipt["finalist_id"],approval)
    value=retained(receipt)["plan"]
    prepared=launch.prepare_candidate(plan=value,candidate=value["candidates"][0],attempt_dir=tmp_path/"full-bundle")
    authority=validate_session_authority(launch._read(Path(prepared["authority"]["path"])))
    assert authority["learned_policy_rollout_count"]==200
    assert authority["beta_screening_admission"]["evaluation_purpose"]=="finalist"
    bundle=launch._read(Path(prepared["bundle_receipt"]["path"]))
    preflight=_blueprint_bundle_preflight(job_dir=tmp_path/"full-transport",generated_at="fixed",
        enable_blueprint_bundle=True,enable_isaac_smoke=True,provider_bundle_kind="native_task_arena_policy_canary_session",
        bundle_path=Path(bundle["bundle_path"]),
        provider_bundle_url="https://storage.example.com/bundle.zip",
        provider_output_put_url="https://storage.example.com/output.zip")
    assert preflight["status"]=="passed",preflight.get("blockers")
    assert not (tmp_path/"full-bundle"/"allocator_launch_intent.json").exists()
    record=retained(receipt)
    record.update(state="launching",prepared=prepared)
    finalists._write(finalists._queue()/"requests"/(receipt["finalist_id"]+".json"),record)
    with pytest.raises(ValueError,match="allocator_authority_mismatch"):
        finalists.require_finalist_session_authority(authority,job_dir=tmp_path/"copied-job-dir",max_hourly_rate_usd=0.8)
    with pytest.raises(ValueError,match="allocator_authority_mismatch"):
        finalists.require_finalist_session_authority(authority,job_dir=tmp_path/"full-bundle"/"allocator",max_hourly_rate_usd=2.0)
    def revoked(*args,**kwargs):
        raise ValueError("beta_screening_current_capture_consent_required")
    monkeypatch.setattr(worker,"require_current_source",revoked)
    with pytest.raises(ValueError,match="consent_required"):
        finalists.consume_finalist_provider_create(authority,bundle,job_dir=tmp_path/"full-bundle"/"allocator",max_hourly_rate_usd=0.8)
    monkeypatch.setattr(worker,"require_current_source",lambda *args,**kwargs:{})
    assert finalists.consume_finalist_provider_create(authority,bundle,
        job_dir=tmp_path/"full-bundle"/"allocator",max_hourly_rate_usd=0.8)["status"]=="consumed"
    with pytest.raises(ValueError,match="already_consumed"):
        finalists.consume_finalist_provider_create(authority,bundle,job_dir=tmp_path/"full-bundle"/"allocator",max_hourly_rate_usd=0.8)


def test_running_ack_cannot_swallow_failed_terminal_delivery(tmp_path, monkeypatch):
    _, receipt, original = completed(tmp_path, monkeypatch)
    finalists.authorize_finalist(receipt["finalist_id"], authorize(receipt))
    backend = Backend({"status": "running", "provider_allocation_performed": True})
    finalists.process_finalists(executor=backend, transport=lambda pub: None)
    running_digest = retained(receipt)["publication_acknowledged_digest"]
    backend.observation = {"status": "completed", "provider_allocation_performed": True,
                           "provider_zero_verified": True, "result": full_result(retained(receipt), original)}
    def offline(pub):
        raise RuntimeError("delivery unavailable")
    finalists.process_finalists(executor=backend, transport=offline)
    failed = retained(receipt)
    assert failed["state"] == "completed"
    assert failed["publication_acknowledged_digest"] == running_digest
    assert failed["publication"]["publication_digest"] != running_digest
    delivered = []
    finalists.process_finalists(executor=backend, transport=delivered.append)
    assert len(delivered) == 1 and delivered[0]["status"] == "completed"
    assert retained(receipt)["publication_acknowledged_digest"] == delivered[0]["publication_digest"]
    assert backend.launches == 1


def test_unrelated_budget_refresh_reuses_retained_terminal_scientific_publication(tmp_path, monkeypatch):
    request, receipt, original = completed(tmp_path, monkeypatch)
    # Start again from the same accepted scientific result, before its finalist
    # preparation has been delivered. The history is the actual producer bytes.
    path = finalists._queue() / "requests" / (receipt["finalist_id"] + ".json")
    path.unlink()
    root = tmp_path / "screening"
    coordinator.enqueue_screening_campaign(plan(campaign_id="another-campaign"),queue_root=root)
    coordinator.process_screening_queue(queue_root=root,require_publication_ack=False,
        executor=Backend({"status":"running","provider_allocation_performed":True}))
    assert finalists.prepare_finalist(request, screening_root=root)["finalist_id"] == receipt["finalist_id"]
    quote = retained(receipt)["plan"]["candidates"][0]["cost_quote"]
    assert quote["price_source"].startswith("operator_compute_ceiling:")
    assert quote["quote_id"] != original["candidates"][0]["cost_quote"]["quote_id"]


def test_capacity_wait_and_proven_no_create_retry_keep_same_owner_allowance(tmp_path,monkeypatch):
    _,receipt,_=completed(tmp_path,monkeypatch)
    finalists.authorize_finalist(receipt["finalist_id"],authorize(receipt))
    backend=Backend()
    backend.preflight_candidate_capacity=lambda **kwargs:{"capacity_available":False,"blockers":["vast_paid_launch_lock_busy"]}
    finalists.process_finalists(executor=backend,transport=lambda pub:None)
    assert retained(receipt)["state"]=="preparing" and backend.launches==0
    from blueprint_pipeline.beta_policy_screening_launch import _record
    def refused(**kwargs):
        backend.launches+=1
        path=kwargs["attempt_dir"]/"allocator_result.json"
        path.write_text(json.dumps({"status":"blocked","blockers":["vast_paid_launch_lock_busy"],
            "provider_mutations_performed":0,"provider_create_attempted":False}))
        return {"status":"blocked","blockers":["vast_paid_launch_lock_busy"],
            "no_allocation_proven":True,"provider_allocation_performed":False,
            "retryable_capacity_refusal":True,"no_policy_api_calls_proven":True,"capacity_refusal_receipt":_record(path)}
    backend.preflight_candidate_capacity=lambda **kwargs:{"capacity_available":True}
    backend.launch_candidate=refused
    finalists.process_finalists(executor=backend,transport=lambda pub:None)
    current=retained(receipt)
    assert current["state"]=="preparing" and current["attempt_generation"]==1
    owner=current["authority"]["authority_digest"]
    backend.launch_candidate=lambda **kwargs:{"status":"running","provider_allocation_performed":True}
    finalists.process_finalists(executor=backend,transport=lambda pub:None)
    assert retained(receipt)["state"]=="running" and retained(receipt)["authority"]["authority_digest"]==owner
    assert len(retained(receipt)["attempt_history"])==1
