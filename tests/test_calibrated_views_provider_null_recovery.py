"""A provider machine that never ran our container is a recoverable, machine-avoiding failure.

Scene 840938, 2026-09-12: vast machine 38773 took the create call, never reached the on-start
heartbeat, was torn down, and the render sealed ``artifacts: {}`` with a bare blocker. Scene
recovery saw no producer evidence and the intent stayed ``preparation_failed`` for good; a
successor would also have re-staged the frozen avoidlist without the machine the adapter had
just marked bad.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import sam31_source_calibration_stage as stage
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.source_calibration_render_return import record
from blueprint_pipeline.task_evaluation_scene_progression_recovery import failure_kind, retain_failure
from tests.test_task_evaluation_scene_recovery import recover, setup, write

FROZEN_IDS = [20166, 31726]
BAD_MACHINE = 38773


def adapter_result(*, classification="pre_execution_provider_null", retained_owned=False):
    started = classification != "pre_execution_provider_null"
    return {"schema_version": "vast_provider_adapter_result.v1", "status": "failed", "reason": "vast_probe_failed",
            "blockers": ["vast_heartbeat_container_missing"], "vast_instance_ids": [50792900],
            "provider_create_attempted": True, "retained_owned": retained_owned,
            "provider_attempt_classification": {
                "schema_version": "provider_attempt_classification.v1", "classification": classification,
                "provider_bundle_started": started, "provider_entrypoint_started": started,
                "provider_output_returned": False, "scientific_attempt_consumed": started,
                "pre_execution_requeue_eligible_in_principle": not started, "automatic_requeue_authorized": False,
                "automatic_requeue_executed": False, "maximum_automatic_requeues": 0,
                "authority_required_for_next_provider_mutation": True, "blockers": ["vast_heartbeat_container_missing"]}}


def allocator_result(adapter_path):
    return {"schema_version": "adp009d_retained_scene_gpu_render_vast_run.v1", "status": "blocked",
            "render_scope": "source_calibration", "source_calibration_return": None, "vast_instance_ids": [50792900],
            "provider_adapter_result_path": str(adapter_path), "retry_cap": 0, "continuing_spend_from_this_run": False,
            "independent_watchdog": {"provider_absence_confirmed": True},
            "blockers": ["provider_render_execution_contract_invalid", "provider_render_not_completed"]}


def _avoid_entry(machine_id):
    return {"generated_at": "2026-09-12T19:51:11Z", "machine_id": machine_id, "instance_id": 50792900,
            "offer_id": 50465106, "reason": "vast_startup_control_plane_did_not_reach_onstart_heartbeat",
            "blockers": ["vast_heartbeat_container_missing"],
            "retry_policy": "exclude_persistently_across_sibling_jobs_until_manual_review"}


def _harness(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_sam31_preparation_cpu_stages as cpu
    from scripts import issue_retained_scene_render_paid_attempt_authority as issuer
    prepared_path = tmp_path / "prepared.json"
    prepared = {"preparation_digest": ""}
    prepared["preparation_digest"] = canonical_digest(prepared, digest_field="preparation_digest")
    prepared_path.write_text(json.dumps(prepared))
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps({"human_authority": {"accepted_by": "fixture-owner"}}))
    source = tmp_path / "source.ply"
    source.write_bytes(b"hermetic mock allocator input")
    frozen = tmp_path / "frozen_avoidlist.json"
    frozen.write_text(json.dumps({"schema_version": "vast_machine_avoidlist.v1", "status": "completed",
                                  "machine_ids": FROZEN_IDS, "entries": [], "raw_secret_values_recorded": False}))
    monkeypatch.setattr(cpu, "execute_cpu_stage", lambda job, **kw: {
        "prepared_inputs": record(prepared_path), "calibrated_view_request": record(task_path)})

    def build(**kwargs):
        kwargs["job_dir"].mkdir()
        (kwargs["job_dir"] / stage.RECEIPT_NAME).write_text("{}")
    monkeypatch.setattr(stage, "build_source_calibration_gpu_render_bundle", build)
    monkeypatch.setattr(issuer, "issue_paid_attempt_authority", lambda **_: {})
    monkeypatch.setattr(stage, "verify_source_calibration_return", lambda *_: {})
    monkeypatch.setattr(stage, "_posted_charge", lambda *_: None)
    execution_root = tmp_path / "sam31-preparations"

    def job(parent_digest, child_id):
        root = execution_root / parent_digest / child_id / "artifacts"
        root.mkdir(parents=True)
        return {"output_root": str(root), "repo_root": str(tmp_path), "runtime_root": str(tmp_path),
                "expected_source_commit": "a" * 40, "parent_request_digest": "sha256:" + parent_digest,
                "plan": {"host_inputs": {"task_request": record(task_path)}},
                "inputs": {"standard_splat_conversion_receipt": record(source), "source_appearance": record(source)},
                "server_profile": {"approved_paid_input_roots": [str(tmp_path)], "calibrated_views": {
                    "execution_site": "provider_gpu", "hardware_required": True, "max_spend_usd": 1.0,
                    "hard_ttl_seconds": 1800, "max_hourly_rate_usd": .5, "retry_cap": 0, "maximum_resource_count": 1,
                    "allowed_geolocation_country_codes": ["US"], "machine_avoidlist": record(frozen)}}}, root
    return job, prepared_path, frozen


def _admitted_ids(argv):
    return json.loads(Path(argv[argv.index("--adp-machine-avoidlist") + 1]).read_text())["machine_ids"]


def _provider_null_allocation(argv, **kwargs):
    """What the render does on a bad machine: stage the avoidlist, append the machine, seal a blocked run."""
    job_dir = Path(argv[argv.index("--adp-retained-scene-render-job-dir") + 1])
    admitted = json.loads(Path(argv[argv.index("--adp-machine-avoidlist") + 1]).read_text())
    (job_dir / "vast_provider_run").mkdir(parents=True)
    staged = job_dir / "provider_machine_avoidlist.json"
    staged.write_text(json.dumps({**admitted, "machine_ids": sorted(set(admitted["machine_ids"]) | {BAD_MACHINE}),
                                  "entries": [*admitted["entries"], _avoid_entry(BAD_MACHINE)]}))
    adapter_path = job_dir / "vast_provider_run" / "vast_provider_adapter_result.json"
    adapter_path.write_text(json.dumps(adapter_result()))
    Path(argv[argv.index("--adapter-output") + 1]).write_text(json.dumps(allocator_result(adapter_path)))
    return 0


def test_provider_null_render_seals_its_evidence_and_the_next_child_avoids_the_machine(tmp_path, monkeypatch):
    job, prepared_path, frozen = _harness(tmp_path, monkeypatch)
    first_job, first_root = job("b" * 64, "sam31-first")
    seen = []

    def first_allocation(argv, **kwargs):
        seen.append(_admitted_ids(argv))
        return _provider_null_allocation(argv, **kwargs)
    outcome = stage.execute_source_calibration_stage(first_job, allocator_runner=first_allocation)
    assert seen == [FROZEN_IDS]  # nothing learned yet: only the frozen profile snapshot
    assert outcome["status"] == "failed" and outcome["stage_id"] == "calibrated_views"
    assert outcome["blockers"][0] == "source_calibration_render_gpu_execution_not_complete"
    assert "provider_render_not_completed" in outcome["blockers"]
    artifacts = outcome["artifacts"]
    assert artifacts["source_calibration_allocator_result"] == record(first_root / "allocator_result.json")
    adapter_path = first_root / "provider/vast_provider_run/vast_provider_adapter_result.json"
    assert artifacts["source_calibration_provider_adapter_result"] == record(adapter_path)
    assert artifacts["source_calibration_machine_avoidlist"] == record(first_root / "provider/provider_machine_avoidlist.json")
    assert failure_kind(json.loads(adapter_path.read_text())) == "provider_null"
    admitted = json.loads((first_root / stage.ADMITTED_AVOIDLIST_NAME).read_text())
    assert admitted["frozen_snapshot"] == record(frozen) and admitted["sibling_exclusions"] == []
    # The same child re-entered after allocation never allocates again; the sealed evidence is stable.
    def no_allocation(*args, **kwargs):
        pytest.fail("a sealed provider failure must not allocate again")
    assert stage.execute_source_calibration_stage({**first_job, "resume_only": True}, allocator_runner=no_allocation) == outcome

    # A sibling in another preparation, plus an unreadable foreign copy, must not derail the next child.
    foreign = tmp_path / "sam31-preparations" / ("c" * 64) / "sam31-foreign" / "artifacts/provider"
    foreign.mkdir(parents=True)
    (foreign / "provider_machine_avoidlist.json").write_text("not json")
    second_job, second_root = job("d" * 64, "sam31-second")

    def second_allocation(argv, **kwargs):
        seen.append(_admitted_ids(argv))
        result = {"status": "completed", "render_scope": "source_calibration",
                  "source_calibration_return": {"return_path": str(prepared_path)}}
        Path(argv[argv.index("--adapter-output") + 1]).write_text(json.dumps(result))
        return 0
    second = stage.execute_source_calibration_stage(second_job, allocator_runner=second_allocation)
    assert second["status"] == "waiting_for_external_result"
    assert seen[-1] == sorted({*FROZEN_IDS, BAD_MACHINE})
    admitted = json.loads((second_root / stage.ADMITTED_AVOIDLIST_NAME).read_text())
    assert [row["machine_id"] for row in admitted["sibling_exclusions"]] == [BAD_MACHINE]
    assert admitted["sibling_exclusions"][0]["source_attempt_avoidlist"] == str(first_root / "provider/provider_machine_avoidlist.json")
    assert admitted["sibling_scan_skipped"] == [{"reason": "sibling_avoidlist_unreadable",
                                                "path": str(foreign / "provider_machine_avoidlist.json")}]
    # A producer path outside the child root is never bound as evidence.
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(adapter_result()))
    sealed = stage.provider_execution_failure(root=first_root, result_path=first_root / "allocator_result.json",
        result={**allocator_result(outside)}, source_commit="a" * 40, owner_metadata={})
    assert "source_calibration_provider_adapter_result" not in sealed["artifacts"]


def _failed_child(tmp_path, first, producer_docs):
    parent = "sha256:" + "b" * 64
    key = {"parent_request_digest": parent, "plan_digest": "sha256:" + "c" * 64,
           "phase": "calibrated_views", "inputs_digest": "sha256:" + "d" * 64}
    child = "sam31-" + canonical_digest(key)[7:]
    queue = tmp_path / "child-queue"
    (queue / "failed").mkdir(parents=True)
    (queue / "results").mkdir()
    job = {**key, "expected_source_commit": first["source_commit"], "parent_preparation_id": "fixture-parent", "child_id": child}
    write(queue / "failed" / (child + ".json"), job, "job_digest")
    write(queue / "results" / (child + ".json"), {"job_digest": job["job_digest"], "child_id": child, "status": "failed",
        "blocker": "source_calibration_render_gpu_execution_not_complete;provider_render_not_completed",
        "artifacts": producer_docs}, "result_digest")
    return queue, {"request_digest": parent, "preparation_id": "fixture-parent"}


@pytest.mark.parametrize("variant", ["recoverable", "bundle_started", "retained", "bare_blocker"])
def test_only_a_classified_provider_null_failure_reserves_a_successor(tmp_path, variant):
    intent, first, evidence = setup(tmp_path, retries=2)
    producer = tmp_path / "producer"
    producer.mkdir()
    adapter_path = producer / "vast_provider_adapter_result.json"
    adapter_path.write_text(json.dumps(adapter_result(
        classification="provider_bundle_attempt_started" if variant == "bundle_started" else "pre_execution_provider_null",
        retained_owned=variant == "retained")))
    allocator_path = producer / "allocator_result.json"
    allocator_path.write_text(json.dumps(allocator_result(adapter_path)))
    docs = {} if variant == "bare_blocker" else {"source_calibration_allocator_result": record(allocator_path),
                                                 "source_calibration_provider_adapter_result": record(adapter_path)}
    queue, link = _failed_child(tmp_path, first, docs)
    failure = retain_failure(attempt=first, link=link, child_queue_root=queue,
                             output_root=tmp_path / "retained-failure", now=102)
    if variant != "recoverable":
        assert failure is None  # consumed science, a retained machine, or no producer evidence: no retry
        return
    value = json.loads(failure.read_text())
    assert value["failure_kind"] == "provider_null" and value["producer_result"] == record(adapter_path)
    evidence["failure"] = record(failure)
    before = {p: p.read_bytes() for p in (adapter_path, allocator_path, failure)}
    successor = recover(tmp_path, intent, evidence)
    assert successor["recovery"]["prior_attempt_digest"] == first["attempt_digest"]
    assert recover(tmp_path, intent, evidence) == successor
    assert all(p.read_bytes() == raw for p, raw in before.items())


def test_provider_null_label_without_the_adapter_classification_is_refused(tmp_path):
    intent, _first, evidence = setup(tmp_path)
    producer = write(tmp_path / "producer.json", {"status": "failed", "blockers": ["vast_heartbeat_container_missing"]})
    failure = json.loads((tmp_path / "failure.json").read_text())
    failure.update(failure_kind="provider_null", producer_result=producer)
    evidence["failure"] = write(tmp_path / "failure.json", failure, "failure_digest")
    with pytest.raises(ValueError, match="provider_null_evidence_missing"):
        recover(tmp_path, intent, evidence)
    assert len(list((tmp_path / intent["intent_id"] / "attempts").glob("*.json"))) == 1
