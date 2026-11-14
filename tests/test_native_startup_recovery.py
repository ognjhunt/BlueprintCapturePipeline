"""Native startup recovery reuses the owner's bounded ledger, never live GPUs."""

import pytest

from blueprint_pipeline import task_evaluation_native_startup_recovery as recovery
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_execution_authority import bind_scene_attempt
from blueprint_pipeline.task_evaluation_scene_owner_attempt_profiles import make_owner_attempt_record
from tests.test_task_evaluation_scene_recovery import setup, write


def seal(value, field):
    value[field] = canonical_digest(value, digest_field=field)
    return value


def fixture(tmp_path, fault=None):
    owner, prior, evidence = setup(tmp_path)
    run = tmp_path / "launches" / "native-original"
    artifact = run / "allocator" / "attempt_001"
    artifact.mkdir(parents=True)
    adapter = {"status": "blocked", "generated_at": "1970-01-01T00:01:42+00:00",
        "vast_instance_ids": [51947666], "continuing_spend_from_this_run": False,
        "provider_attempt_classification": {"schema_version": "provider_attempt_classification.v1",
            "classification": "pre_execution_provider_null", "scientific_attempt_consumed": False,
            "provider_bundle_started": False, "provider_entrypoint_started": False,
            "provider_output_returned": False, "pre_execution_requeue_eligible_in_principle": True,
            "blockers": ["vast_heartbeat_instance_exited"]}}
    if fault in {"dependency", "configuration", "unknown"}:
        adapter["blockers"] = [{"dependency": "ModuleNotFoundError: pxr",
            "configuration": "provider_remote_blocker:runtime_image_invalid",
            "unknown": "unclassified_failure"}[fault]]
    if fault == "execution":
        adapter["provider_attempt_classification"]["provider_bundle_started"] = True
    adapter_ref = write(artifact / "adapter.json", adapter)
    manifest_ref = write(artifact / "artifacts.json", {"files": [{**adapter_ref,
        "relative_path": "adapter.json", "roles": ["allocator_adapter_result"]}]}, "manifest_digest")
    result = {"status": "blocked", "continuing_spend_from_this_run": False,
        "all_staged_objects_absent": True, "native_control_result_path": None,
        "adapter_result_path": adapter_ref["path"], "artifact_manifest_path": manifest_ref["path"],
        "independent_watchdog_close": {"status": "provider_terminal", "provider_absence_confirmed": True}}
    if fault == "live":
        result["continuing_spend_from_this_run"] = True
    if fault == "output":
        result["native_control_result_path"] = "native.json"
    if fault == "unknown_zero":
        result["independent_watchdog_close"]["provider_absence_confirmed"] = False
    result_ref = write(run / "allocator" / "result.json", result, "result_digest")
    profile = {"source_commit": prior["source_commit"], **bind_scene_attempt(prior)}
    write(run / "launch_profile.json", profile, "profile_digest")
    launch = {"launch_id": run.name, "profile_digest": profile["profile_digest"],
              "configured_scene_revision_digest": "sha256:" + "a" * 64, "status": "construction_launch_queued"}
    receipt = {"status": "blocked", "launch_id": run.name, "launch_profile_digest": profile["profile_digest"],
        "source_commit": prior["source_commit"], "terminal_evidence": {
            "result": {"path": result_ref["path"], "digest": result_ref["sha256"]},
            "artifacts": {"artifact_manifest_path": {"path": manifest_ref["path"], "digest": manifest_ref["sha256"]}}}}
    write(run / "launch_receipt.json", receipt, "receipt_digest")
    activation = {"expected_production_commit": prior["source_commit"], "lane": "native_task_arena_construction",
        "activation_request": {"activation_id": "native-original", "lineage": {},
            "authorization": {"scene_owner_attempt": make_owner_attempt_record(owner_fields=bind_scene_attempt(prior), phase="construction",
                team_namespace="team", scene_id="scene", task_id="task",
                runtime_source_bundle_digest="sha256:" + "b" * 64)}}}
    if fault == "tampered_adapter":
        (artifact / "adapter.json").write_text('{}')
    return owner, prior, evidence, run, launch, activation


@pytest.mark.parametrize("fault", ["execution", "live", "output", "unknown_zero", "tampered_adapter", "dependency", "configuration", "unknown"])
def test_never_retries_scientific_failure_or_unresolved_provider(tmp_path, fault):
    _, _, _, run, launch, activation = fixture(tmp_path, fault)
    with pytest.raises(ValueError):
        recovery.retain_failure(run_root=run, launch=launch, activation=activation,
            scene_root=tmp_path, output_root=tmp_path / "recovery")
    assert not (tmp_path / "recovery" / "failure.json").exists()


def test_retained_native_failure_is_eligible_for_real_bounded_reservation(tmp_path):
    owner, prior, evidence, run, launch, activation = fixture(tmp_path)
    old_bytes = {p: p.read_bytes() for p in run.rglob('*.json')}
    observed, path = recovery.retain_failure(run_root=run, launch=launch, activation=activation,
        scene_root=tmp_path, output_root=tmp_path / "recovery")
    assert observed == prior
    evidence["failure"] = recovery.record(path)
    args = dict(queue_root=tmp_path, intent_id=owner["intent_id"], attempt_id="replacement",
        source_commit=prior["source_commit"], runtime_digest=prior["runtime_digest"],
        input_digest=prior["input_digest"], provider="vast", maximum_spend_usd=2,
        now=104, recovery_from_attempt_id=prior["attempt_id"], recovery_evidence=evidence)
    result = recovery.intake.reserve_scene_attempt(**args)
    assert result["recovery"]["budget"] == "retry"
    assert recovery.intake.reserve_scene_attempt(**args) == result
    with pytest.raises(ValueError, match="successor_already_reserved"):
        recovery.intake.reserve_scene_attempt(**{**args, "attempt_id": "duplicate"})
    assert all(p.read_bytes() == raw for p, raw in old_bytes.items())


@pytest.mark.parametrize('fault', [None, 'dependency', 'execution', 'live', 'unknown_zero', 'tampered_adapter'])
def test_corrected_release_keeps_completed_placement_only_after_closed_startup_loss(tmp_path, monkeypatch, fault):
    from blueprint_pipeline import task_evaluation_completed_placement_adoption as adoption
    from blueprint_pipeline.configured_scene_run_identity import progression_directory
    owner, prior, evidence, run, launch, activation = fixture(tmp_path, fault)
    plan = {"source_launch_id": "source", "expected_production_commit": prior["source_commit"],
        "future_outputs": {"construction": {"expected_activation_id": "native-original"}}}
    state = tmp_path / "progression" / "source" / progression_directory(prior["source_commit"], None)
    state.mkdir(parents=True)
    write(state / 'construction_launch_progression.json', launch, 'progression_digest')
    write(state / 'construction_activation_progression.json', activation, 'progression_digest')
    config = {"scene_root": str(tmp_path), "progression_root": str(tmp_path / 'progression'), "launch_state_root": str(run.parent)}
    before = {p: p.read_bytes() for p in (tmp_path / owner['intent_id']).rglob('*.json')}
    assert adoption.native_startup_failed(config=config, plan=plan) is (fault is None)
    monkeypatch.setattr(adoption, 'validate_adoption', lambda _: {'plan': plan})
    if fault is None:
        adoption.retire_unused_native(config=config, intent_id=owner['intent_id'], packet={}, dry_run=True)
    else:
        with pytest.raises(ValueError, match='native_submission_started'):
            adoption.retire_unused_native(config=config, intent_id=owner['intent_id'], packet={}, dry_run=True)
    assert all(p.read_bytes() == value for p, value in before.items())
    assert not (tmp_path / owner['intent_id'] / 'cancelled-unstarted-controls').exists()
