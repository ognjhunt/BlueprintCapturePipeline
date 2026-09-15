"""Closed zero-provider disk failure recovery uses real intake and producer contracts."""
from datetime import datetime, timezone
from pathlib import Path
import copy
import json
import shutil

import pytest

from blueprint_pipeline import task_evaluation_scene_capacity_recovery as capacity
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_execution_authority import bind_scene_attempt
from blueprint_pipeline.task_evaluation_scene_configuration_provider_artifacts import (
    _provider_output_disk_requirements, _provider_transfer_byte_budget,
)
from tests.test_task_evaluation_scene_recovery import setup, write


def capacity_fixture(tmp_path, monkeypatch, retries=2):
    intent, first, evidence = setup(tmp_path, retries=retries)
    commit, run_id = first["source_commit"], "configuration-one"
    launches = tmp_path / "launches"
    run = launches / "configuration-one-launch"
    run.mkdir(parents=True)
    preparation = {"run_id": run_id, "scene_intent_digest": first["intent_digest"]}
    prep_ref = write(tmp_path / "preparation.json", preparation)
    main = intake.reserve_scene_attempt(queue_root=tmp_path, intent_id=intent["intent_id"],
        attempt_id="scene-configuration-child", source_commit=commit, runtime_digest=first["runtime_digest"],
        input_digest=canonical_digest(preparation), provider="vast", maximum_spend_usd=2, now=101)
    main_ref = capacity.record(tmp_path / intent["intent_id"] / "attempts/scene-configuration-child.json")
    link = {"intent_id": first["intent_id"], "intent_digest": first["intent_digest"],
            "request_digest": canonical_digest(preparation), "scene_configuration_attempt": main_ref,
            "team_namespace": "team", "scene_id": "scene", "task_id": "pick-book"}
    link_ref = write(tmp_path / "link.json", link, "link_digest")
    factory_root = tmp_path / "factory-output"
    factory_directory = factory_root / intent["intent_id"] / first["attempt_id"]
    factory_directory.mkdir(parents=True)
    factory_ref = write(factory_directory / "factory.json", {"attempt_digest": first["attempt_digest"],
        "intent_digest": first["intent_digest"], "source_commit": commit, "submission_request": prep_ref}, "factory_digest")
    bundle = {"run_id": run_id, "source_commit": commit, "bundle_sha256": "sha256:" + "b" * 64,
              "bundle_size_bytes": 1024**2, "bundle_path": str(tmp_path / "removed-historical.zip")}
    bundle_ref = write(tmp_path / "bundle.json", bundle, "receipt_digest")
    authority = {"run_id": run_id, "source_commit": commit, "bundle_receipt": bundle_ref,
                 "bundle_sha256": bundle["bundle_sha256"]}
    authority_ref = write(tmp_path / "authority.json", authority, "authority_digest")
    from tests.test_task_evaluation_launch_dispatcher import _profile
    baseline_profile = _profile(tmp_path)
    profile = {**baseline_profile, **bind_scene_attempt(main), "source_commit": commit,
        "task_evaluation_run": {"run_mode": "scene_configuration", "configuration_run_id": run_id, "evaluation_episode_executed": False,
                               **{k:link[k] for k in ("team_namespace", "scene_id", "task_id")}},
        "immutable_inputs": [{"name": n, "path": r["path"], "digest": r["sha256"]} for n,r in (
            ("source_bundle_manifest", bundle_ref), ("scene_configuration_attempt_authority", authority_ref))]}
    profile["immutable_inputs"].extend(r for r in baseline_profile["immutable_inputs"] if r["name"] == "evaluation_run_spec")
    write(run / "launch_profile.json", profile, "profile_digest")
    request = {"source_commit": commit, "launch_id": run.name, "launch_profile_digest": profile["profile_digest"], "launch_profile_id": profile["profile_id"]}
    write(run / "launch_request.json", request, "request_digest")
    launch = {"status": "blocked", "launch_id": run.name, "launch_profile_digest": profile["profile_digest"],
              "request_digest": request["request_digest"]}
    write(run / "launch_receipt.json", launch, "receipt_digest")
    zero = {**launch, "status": "provider_zero_confirmed", "provider_zero_verified": True,
            "continuing_spend_from_this_run": False, "blockers": []}
    write(run / "post_teardown_provider_zero_receipt.json", zero, "provider_zero_receipt_digest")
    _download, upload = _provider_transfer_byte_budget(bundle)
    requirements = _provider_output_disk_requirements(upload)
    producer = {"schema_version": "task_evaluation_scene_configuration_vast_result.v1", "status": "blocked",
        "run_id": run_id, "source_commit": commit, "bundle_sha256": bundle["bundle_sha256"],
        "authority_digest": authority["authority_digest"], "provider_mutations_performed": 0, "retry_cap": 0,
        "continuing_spend_from_this_run": False, "expected_provider_upload_bytes": upload,
        "provider_output_disk_requirements": requirements,
        "provider_output_disk_capacity": {"schema_version": "scene_configuration_provider_output_disk_capacity.v1",
            "phase": "before_allocation_and_staging", "status": "blocked", "blockers": [capacity.BLOCKER],
            "measurement_path": str(run / "allocator/scene-configuration-job/vast_provider_run"),
            "required_free_bytes": requirements["required_free_bytes_before_download"], "observed_free_bytes": 0},
        "blockers": [capacity.BLOCKER], "generated_at": datetime.fromtimestamp(102,timezone.utc).isoformat()}
    result_path = run / capacity.RESULT_RELATIVE
    result_path.parent.mkdir(parents=True)
    write(result_path, producer, "result_digest")
    config = {"intent_root": str(tmp_path), "factory_output_root": str(factory_root),
              "launch_execution_root": str(launches), "preparation_worker": {"disk_reservation_root": str(tmp_path / "reservations")}}
    from blueprint_pipeline import control_plane_capacity_controller as controller
    measure = controller.measure_mount
    free = {"value": 80 * 1024**3}
    def usage(_path):
        return shutil._ntuple_diskusage(100*1024**3, 100*1024**3-free["value"], free["value"])
    monkeypatch.setattr(capacity.shutil, "disk_usage", usage)
    monkeypatch.setattr(controller, "measure_mount", lambda *a, **k: measure(*a, **k, disk_usage=usage))
    observation = capacity.observe_failure(attempt=first, link_path=Path(link_ref["path"]),
        preparation_path=Path(prep_ref["path"]), factory_path=Path(factory_ref["path"]), config=config)
    admission = capacity.capacity_admission(observation, config, 103)
    failure = capacity.retain_failure(observation=observation, attempt=first,
                                     output_root=tmp_path / "capacity-failures", admission=admission)
    evidence["failure"] = capacity.record(failure)
    observation["_free_bytes"] = free
    return intent, first, observation, config, evidence


def reserve(root, intent, first, evidence, name="capacity-successor"):
    return intake.reserve_scene_attempt(queue_root=root, intent_id=intent["intent_id"], attempt_id=name,
        source_commit=first["source_commit"], runtime_digest=first["runtime_digest"], input_digest=first["input_digest"],
        provider="vast", maximum_spend_usd=2, now=104,
        recovery_from_attempt_id=first["attempt_id"], recovery_evidence=evidence)


def test_real_metadata_chain_reserves_same_release_without_old_bundle_or_mutation(tmp_path, monkeypatch):
    intent, first, observation, config, evidence = capacity_fixture(tmp_path, monkeypatch)
    retained = {Path(r["path"]):Path(r["path"]).read_bytes() for r in observation["records"].values()}
    assert not Path(observation["values"]["bundle"]["bundle_path"]).exists()
    successor = reserve(tmp_path,intent,first,evidence)
    assert successor["source_commit"] == first["source_commit"]
    assert successor["recovery"]["budget"] == "preallocation_capacity"
    assert reserve(tmp_path,intent,first,evidence) == successor
    assert all(p.read_bytes() == value for p,value in retained.items())


@pytest.mark.parametrize("fault", ["mutation", "boolean_zero", "wrong_phase", "extra_blocker", "wrong_prior", "wrong_profile", "tampered_bytes"])
def test_changed_or_nonzero_failure_cannot_authorize_recovery(tmp_path, monkeypatch, fault):
    _intent, first, observed, _config, _evidence = capacity_fixture(tmp_path, monkeypatch)
    refs = copy.deepcopy(observed["records"])
    if fault == "wrong_prior":
        first = {**first, "attempt_digest": "sha256:" + "a" * 64}
    else:
        name = "profile" if fault == "wrong_profile" else "result"
        path = Path(refs[name]["path"])
        value = json.loads(path.read_text())
        if fault == "mutation":
            value["provider_mutations_performed"] = 1
        elif fault == "boolean_zero":
            value["provider_mutations_performed"] = False
        elif fault == "wrong_phase":
            value["provider_output_disk_capacity"]["phase"] = "before_extraction"
        elif fault == "extra_blocker":
            value["blockers"].append("scientific_failure")
        elif fault == "wrong_profile":
            value["task_evaluation_run"]["scene_id"] = "foreign"
        elif fault == "tampered_bytes":
            value["status"] = "completed"
        changed = write(path,value,"profile_digest" if name == "profile" else "result_digest")
        if fault != "tampered_bytes":
            refs[name] = changed
    with pytest.raises(ValueError):
        capacity.validate_source(refs,prior_attempt=first)


def test_zero_retry_consent_and_current_capacity_are_both_required(tmp_path,monkeypatch):
    intent,first,observed,_config,evidence = capacity_fixture(tmp_path,monkeypatch,retries=0)
    with pytest.raises(ValueError,match="retry_cap_exhausted"):
        reserve(tmp_path,intent,first,evidence)
    observed["_free_bytes"]["value"] = 1
    failure = json.loads(Path(evidence["failure"]["path"]).read_text())
    with pytest.raises(ValueError,match="capacity_not_recovered"):
        capacity.validate_capacity_failure(failure,observed["values"]["result"],first,104)


def _retained_budget_history(root, intent, first, count, budget=None):
    """Intake-sealed historical reservations exercise counters without fake new grants."""
    directory = root / intent["intent_id"] / "attempts"
    for index in range(count):
        value = {**first, "attempt_id": f"historical-{index}", "maximum_spend_usd": 0.1,
                 "recovery": {"prior_attempt_id": f"prior-{index}"}}
        if budget is not None:
            value["recovery"]["budget"] = budget
        intake.write_exclusive(directory / (value["attempt_id"] + ".json"), intake._seal(value, "attempt_digest"))


def test_capacity_recovery_does_not_spend_exhausted_ordinary_retry_counter(tmp_path, monkeypatch):
    intent, first, _, _, evidence = capacity_fixture(tmp_path, monkeypatch)
    _retained_budget_history(tmp_path, intent, first, 2)  # Legacy rows count as ordinary retries.
    successor = reserve(tmp_path, intent, first, evidence)
    assert successor["recovery"]["budget"] == "preallocation_capacity"
    rows = [intake._read(p, "attempt_digest") for p in (tmp_path/intent["intent_id"]/"attempts").glob("*.json")]
    assert sum(r.get("recovery", {}).get("budget", "retry") == "retry" for r in rows if "recovery" in r) == 2


@pytest.mark.parametrize("prior_count", [5, 6])
def test_six_capacity_recoveries_exhaust_their_own_bounded_counter(tmp_path, monkeypatch, prior_count):
    from blueprint_pipeline.task_evaluation_scene_recovery import MAX_PREALLOCATION_CAPACITY_RECOVERIES
    assert MAX_PREALLOCATION_CAPACITY_RECOVERIES == 6
    intent, first, _, _, evidence = capacity_fixture(tmp_path, monkeypatch)
    _retained_budget_history(tmp_path, intent, first, prior_count, "preallocation_capacity")
    if prior_count == 5:
        assert reserve(tmp_path, intent, first, evidence)["recovery"]["budget"] == "preallocation_capacity"
    else:
        with pytest.raises(ValueError, match="preallocation_capacity_recovery_cap_exhausted"):
            reserve(tmp_path, intent, first, evidence)


@pytest.mark.parametrize("limit", ["attempt", "spend"])
def test_capacity_recovery_keeps_existing_standing_limits(tmp_path, monkeypatch, limit):
    intent, first, _, _, evidence = capacity_fixture(tmp_path, monkeypatch)
    for index in range(7 if limit == "attempt" else 1):
        intake.reserve_scene_attempt(queue_root=tmp_path, intent_id=intent["intent_id"], attempt_id=f"held-{index}",
            source_commit=first["source_commit"], runtime_digest=first["runtime_digest"], input_digest=first["input_digest"],
            provider="vast", maximum_spend_usd=0.1 if limit == "attempt" else 16, now=103)
    with pytest.raises(ValueError, match=limit + "_cap_exhausted"):
        reserve(tmp_path, intent, first, evidence)


def test_capacity_admission_cannot_substitute_an_unrelated_disk_mount(tmp_path, monkeypatch):
    _, first, observed, _, evidence = capacity_fixture(tmp_path, monkeypatch)
    failure = json.loads(Path(evidence["failure"]["path"]).read_text())
    failure["capacity_admission"]["cpu_path"] = str(tmp_path / "unrelated-large-disk")
    failure["capacity_admission"]["capacity_digest"] = canonical_digest(failure["capacity_admission"], digest_field="capacity_digest")
    with pytest.raises(ValueError, match="capacity_measurement_scope_changed"):
        capacity.validate_capacity_failure(failure, observed["values"]["result"], first, 104)
