"""A scene-configuration launch that provably spent nothing recovers by itself; anything else is named.

2026-11-10 audit: no GPU offer, a staging error, a Vast inventory error, a watchdog
that would not arm and a refused paid admission all parked a Website scene until a
deploy; a launch with no allocator result showed no blocker; a machine lost mid-run
was refused because Website intents carried ``max_retries: 0``.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_scene_capacity_recovery as recovery
from blueprint_pipeline import task_evaluation_scene_launch_recovery as launch_recovery
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline import task_evaluation_terminal_scene_attempt_settlement as settlement
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_outcome import decide_outcome
from blueprint_pipeline.task_evaluation_scene_recovery import MAX_PREALLOCATION_CAPACITY_RECOVERIES, PAID_RETRY_KIND
from tests.test_task_evaluation_scene_capacity_recovery import capacity_fixture, write
from tests.test_website_credit_recovery import _free_recovery

NO_OUTPUT = ["scene_configuration_configured_revision_not_published", "scene_configuration_provider_not_completed",
             "scene_configuration_provider_output_zip_invalid"]
#: The sealed blockers of each class, as the lane writes them.
CLASSES = {
    "no_gpu_offer": ["no_vast_offer_with_known_supported_isaac_driver_at_or_below_max_hourly_rate", *NO_OUTPUT],
    "staging_error": ["object_store_upload_or_presign_failed:ClientError"],
    "durable_bundle_error": ["scene_configuration_provider_bundle_durable_publication_failed:TimeoutError"],
    "inventory_query_error": ["vast_prelaunch_inventory_query_failed", *NO_OUTPUT],
    "watchdog_not_armed": ["independent_watchdog_not_armed"],
}
FREE = "task_evaluation_scene_preparation_attempt.v1"


def _iso(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def _zero(blockers, **changes):
    return {"schema_version": "task_evaluation_scene_configuration_vast_result.v1", "status": "blocked",
            "blockers": list(blockers), "provider_mutations_performed": 0, "retry_cap": 0,
            "continuing_spend_from_this_run": False, "api_pretraining": None,
            "provider_runtime_output_zip_path": None, **changes}


@pytest.mark.parametrize("name", sorted(CLASSES))
def test_each_zero_spend_class_is_recognised_from_its_own_result(name):
    assert launch_recovery.transient_launch_failure(_zero(CLASSES[name]))


@pytest.mark.parametrize("name", sorted(CLASSES))
@pytest.mark.parametrize("unproven", [
    {"provider_mutations_performed": 1}, {"provider_mutations_performed": False},
    {"api_pretraining": {"status": "completed"}}, {"cpu_prestage": {"status": "completed"}},
    {"instance_id": "123"}, {"vast_instance_ids": [123]}, {"allocation_created": True},
    {"continuing_spend_from_this_run": True}, {"provider_runtime_output_zip_path": "/out.zip"},
    {"extra_blocker": "object_store_provider_zero_not_proven"},
    {"extra_blocker": "independent_watchdog_not_closed"},
    {"extra_blocker": "scene_configuration_scene_inputs_invalid"},
])
def test_the_same_failure_without_proof_is_not_auto_retried(name, unproven):
    changes = dict(unproven)
    blockers = CLASSES[name] + ([changes.pop("extra_blocker")] if "extra_blocker" in changes else [])
    assert not launch_recovery.transient_launch_failure(_zero(blockers, **changes))


def test_consequences_alone_or_a_money_stop_are_not_transient():
    assert not launch_recovery.transient_launch_failure(_zero(NO_OUTPUT))
    assert not launch_recovery.transient_launch_failure(_zero(["account_monthly_spend_limit_reached"]))


# -- the launch's own teardown manifest ------------------------------------------------------------

def _free_attempt(first, refs):
    free = {**first, "schema_version": FREE, "provider": "control_plane", "maximum_spend_usd": 0,
            "paid_authority_granted": False}
    free["attempt_digest"] = canonical_digest(free, digest_field="attempt_digest")
    factory = json.loads(Path(refs["factory"]["path"]).read_text())
    factory["attempt_digest"] = free["attempt_digest"]
    write(Path(refs["factory"]["path"]), factory, "factory_digest")
    return free


def _transient_fixture(tmp_path, monkeypatch, *, blockers, teardown=None, adapter=None):
    _intent, first, observed, config, _evidence = capacity_fixture(tmp_path, monkeypatch)
    refs, values = observed["records"], observed["values"]
    run = Path(refs["launch"]["path"]).parent
    result = _zero(blockers, **{key: values["result"][key] for key in (
        "run_id", "source_commit", "bundle_sha256", "authority_digest", "generated_at")})
    if adapter is not None:
        result["provider_adapter_result_path"] = write(run / "adapter_result.json", adapter)["path"]
    write(Path(refs["result"]["path"]), result, "result_digest")
    teardown_ref = write(run / "teardown.json", {"schema_version": "vast_teardown_manifest.v1",
        "status": "not_required_provider_adapter_never_invoked", "vast_instance_ids": [],
        "teardown_actions_performed": [], "continuing_spend_from_this_run": False, **(teardown or {})})
    launch = values["launch"]
    launch["terminal_evidence"] = {"artifacts": {"teardown_manifest_path": {
        "path": teardown_ref["path"], "exists": True, "digest": teardown_ref["sha256"]}}}
    write(Path(refs["launch"]["path"]), launch, "receipt_digest")
    write(Path(refs["zero"]["path"]), {**values["zero"], "receipt_digest": launch["receipt_digest"]},
          "provider_zero_receipt_digest")
    return first, config, refs


def _observe(attempt, config, refs, now=200):
    return recovery.observe_failure(attempt=attempt, link_path=Path(refs["link"]["path"]),
        preparation_path=Path(refs["preparation"]["path"]), factory_path=Path(refs["factory"]["path"]),
        config=config, now=now)


@pytest.mark.parametrize("name", ["staging_error", "watchdog_not_armed", "durable_bundle_error"])
def test_a_proven_zero_spend_launch_of_a_free_preparation_attempt_is_recoverable(tmp_path, monkeypatch, name):
    first, config, refs = _transient_fixture(tmp_path, monkeypatch, blockers=CLASSES[name])
    observed = _observe(_free_attempt(first, refs), config, refs)
    assert observed["recoverable"] and observed["kind"] == launch_recovery.TRANSIENT_KIND


def test_an_inventory_error_needs_the_adapters_own_no_allocation_teardown(tmp_path, monkeypatch):
    first, config, refs = _transient_fixture(tmp_path, monkeypatch, blockers=CLASSES["inventory_query_error"],
        teardown={"status": "not_required_prelaunch_inventory_guard_blocked"})
    assert _observe(_free_attempt(first, refs), config, refs)["kind"] == launch_recovery.TRANSIENT_KIND


NO_CREATE = {"provider_create_attempted": False, "vast_side_effects_may_have_occurred": False,
             "vast_instance_ids": [], "continuing_spend_from_this_run": False}


@pytest.mark.parametrize("adapter,teardown,proven", [
    (NO_CREATE, {"status": "completed"}, True),
    (None, {"status": "completed"}, False),
    ({**NO_CREATE, "provider_create_attempted": True}, {"status": "completed"}, False),
    (NO_CREATE, {"status": "completed", "vast_instance_ids": [7]}, False),
    (NO_CREATE, {"status": "completed", "teardown_actions_performed": [{"action": "destroy_instance"}]}, False),
    (NO_CREATE, {"status": "blocked"}, False),
])
def test_no_gpu_offer_needs_proof_that_no_create_call_was_made(tmp_path, monkeypatch, adapter, teardown, proven):
    first, config, refs = _transient_fixture(tmp_path, monkeypatch, blockers=CLASSES["no_gpu_offer"],
                                             teardown=teardown, adapter=adapter)
    free = _free_attempt(first, refs)
    if proven:
        assert _observe(free, config, refs)["kind"] == launch_recovery.TRANSIENT_KIND
    else:
        with pytest.raises(ValueError, match="transient_teardown_not_unallocated"):
            _observe(free, config, refs)


def test_a_paid_source_attempt_keeps_its_named_blocker(tmp_path, monkeypatch):
    first, config, refs = _transient_fixture(tmp_path, monkeypatch, blockers=CLASSES["staging_error"])
    observed = _observe(first, config, refs)
    assert observed["recoverable"] is False and observed["blockers"] == CLASSES["staging_error"]


# -- progression: backoff, bound, truthful end -------------------------------------------------------

def _transient_args(tmp_path, monkeypatch, *, failed_at, previous=0, kind=launch_recovery.TRANSIENT_KIND):
    result = {"result_digest": "sha256:" + "1" * 64, "generated_at": _iso(failed_at)}
    return _free_recovery(tmp_path, monkeypatch, kind=kind, result=result, previous_recoveries=previous)


def _scheduled(state, previous):
    # Earlier recoveries of the zero-spend classes advance the schedule; others do not.
    state["recovery_predecessors"] = [{"kind": recovery.KIND,
                                       "configuration_failure_kind": launch_recovery.TRANSIENT_KIND}] * previous


@pytest.mark.parametrize("previous,delay", [(0, 900), (1, 3600), (2, 10800), (3, 21600), (4, 43200), (5, 86400)])
def test_zero_spend_recovery_follows_the_shared_refusal_schedule(tmp_path, monkeypatch, previous, delay):
    from blueprint_pipeline import task_evaluation_scene_progression as engine
    assert launch_recovery.backoff_seconds(previous) == delay
    fx, state, args = _transient_args(tmp_path, monkeypatch, failed_at=300 - delay + 1)
    _scheduled(state, previous)
    assert args["intent"]["request"]["execution"]["max_retries"] == 0  # a $0 class needs no retry consent
    waiting = engine._recover_configuration_capacity(**args)
    assert waiting == {"status": "blocked", "phase": "configuration_capacity",
                       "blockers": [launch_recovery.BACKOFF_BLOCKER + ":due_in_hours:1"]}
    assert state["configuration_recovery_wait"]["next_check_epoch"] == 300 + 1
    assert decide_outcome(waiting["status"], waiting["phase"], waiting["blockers"]) is None  # transient, no outcome
    assert len(list((fx["directory"] / "preparation-attempts").glob("*.json"))) == 1

    fx, state, args = _transient_args(tmp_path / "later", monkeypatch, failed_at=300 - delay)
    _scheduled(state, previous)
    reserved = engine._recover_configuration_capacity(**args)
    assert reserved["phase"] == "capacity_recovery_reserved"
    assert "configuration_recovery_wait" not in state
    assert state["recovery_predecessors"][-1] == {**state["recovery_predecessors"][-1], "kind": recovery.KIND,
        "configuration_failure_kind": launch_recovery.TRANSIENT_KIND}
    new = json.loads(Path(state["attempt"]["path"]).read_text())
    assert new["maximum_spend_usd"] == 0 and new["paid_authority_granted"] is False


def test_credit_or_disk_recoveries_do_not_advance_the_zero_spend_schedule(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_progression as engine
    _fx, _state, args = _transient_args(tmp_path, monkeypatch, failed_at=300 - 900, previous=3)
    assert engine._recover_configuration_capacity(**args)["phase"] == "capacity_recovery_reserved"


def test_a_scheduled_wait_changes_its_blocker_every_hour_so_the_stall_alarm_does_not_page():
    from blueprint_pipeline.control_plane_stall_alarm import hold_reason
    due = 100_000
    blockers = [launch_recovery.backoff_blocker(due, now) for now in range(due - 86400, due + 1, 1800)]
    # Never the same for 90 min (the alarm's default) while time is left; the same once overdue.
    assert all(len(set(blockers[i:i + 4])) > 1 for i in range(len(blockers) - 4))
    assert blockers[-1] == launch_recovery.BACKOFF_BLOCKER + ":due_in_hours:0"
    assert launch_recovery.backoff_blocker(due, due + 7200) == blockers[-1]
    assert hold_reason("blocked", "configuration_capacity", blockers[:1]) is None  # not dressed up as a hold


def test_zero_spend_recovery_bound_ends_truthfully_not_waiting_for_a_deploy(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_progression as engine
    _fx, _state, args = _transient_args(tmp_path, monkeypatch, failed_at=0,
                                        previous=MAX_PREALLOCATION_CAPACITY_RECOVERIES)
    with pytest.raises(ValueError, match="preallocation_capacity_recovery_cap_exhausted") as raised:
        engine._recover_configuration_capacity(**args)
    code = str(raised.value).split(":", 1)[0]
    assert decide_outcome("blocked", "preparation", [code]) == {
        "kind": "setup_repair", "terminal": True, "reason_category": "blueprint_fault",
        "reason_code": "capacity_recovery_cap_exhausted"}


# -- a machine lost mid-run: one more attempt, through max_retries ---------------------------------------

def _with_retries(args, retries):
    intent = copy.deepcopy(args["intent"])
    intent["request"]["execution"]["max_retries"] = retries
    return {**args, "intent": intent}


def test_a_lost_machine_gets_exactly_one_successor_with_one_signed_retry(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_progression as engine
    # Six earlier $0 recoveries do not spend the paid retry, and the paid retry is not one of them.
    fx, state, args = _transient_args(tmp_path, monkeypatch, failed_at=0, kind=recovery.DEAD_MACHINE_KIND,
                                      previous=MAX_PREALLOCATION_CAPACITY_RECOVERIES)
    args = _with_retries(args, 1)
    assert engine._recover_configuration_capacity(**args)["phase"] == "capacity_recovery_reserved"
    assert state["recovery_predecessors"][-1]["kind"] == PAID_RETRY_KIND
    assert len(list((fx["directory"] / "preparation-attempts").glob("*.json"))) == 2
    # The successor's own machine is lost too: the signed retry is spent, so no second successor.
    state.update(factory=fx["factory"], attempt=state["recovery_predecessors"][-1]["attempt"])
    with pytest.raises(ValueError, match="retry_cap_exhausted"):
        engine._recover_configuration_capacity(**{**args, "now": 301})
    assert len(list((fx["directory"] / "preparation-attempts").glob("*.json"))) == 2


def test_a_lost_machine_of_an_in_flight_zero_retry_intent_is_refused_as_before(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_progression as engine
    fx, state, args = _transient_args(tmp_path, monkeypatch, failed_at=0, kind=recovery.DEAD_MACHINE_KIND)
    with pytest.raises(ValueError, match="retry_cap_exhausted"):
        engine._recover_configuration_capacity(**_with_retries(args, 0))
    assert "recovery_predecessors" not in state or state["recovery_predecessors"] == []


def test_a_rented_dead_machine_is_recognised_with_its_one_torn_down_instance(tmp_path, monkeypatch):
    _intent, first, observed, config, _evidence = capacity_fixture(tmp_path, monkeypatch)
    refs, values = observed["records"], observed["values"]
    result = {**values["result"], "provider_mutations_performed": 1,
              "blockers": ["scene_configuration_provider_not_completed", "vast_heartbeat_instance_exited"]}
    write(Path(refs["result"]["path"]), result, "result_digest")
    assert _observe(first, config, refs)["kind"] == recovery.DEAD_MACHINE_KIND
    write(Path(refs["result"]["path"]), {**result, "provider_mutations_performed": 2}, "result_digest")
    with pytest.raises(ValueError, match="capacity_not_zero_provider_failure"):
        _observe(first, config, refs)


# -- a launch that ended without an allocator result -------------------------------------------------

def _seal(path, value, field):
    value[field] = canonical_digest(value, digest_field=field)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return value


def _receipt_run(tmp_path, *, receipt=None, adapter_output=None, started_at=100.0, ttl=3600):
    run = tmp_path / "runs" / "launch-one"
    output = run / "allocator" / "adapter_output.json"
    profile = _seal(run / "launch_profile.json", {"profile_id": "p", "allocator": {
        "hard_ttl_seconds": ttl, "argv": ["--adapter-output", str(output)]}}, "profile_digest")
    _seal(run / "launch_started.json", {"launch_id": run.name, "started_at": _iso(started_at),
                                        "hard_ttl_seconds": ttl}, "started_digest")
    if adapter_output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(adapter_output))
    if receipt is not None:
        _seal(run / "launch_receipt.json", {"launch_id": run.name, "status": "blocked",
            "launch_profile_digest": profile["profile_digest"], **receipt}, "receipt_digest")
    return run, profile


NOT_INVOKED = {"allocator_invoked": False, "allocator_exit_code": None, "provider_mutation_attempted": False,
               "provider_mutation_evidence": {"status": "allocator_not_invoked"},
               "blockers": ["standing_authorization_consumption_not_recorded"]}
ADMISSION_REFUSED = {"allocator_invoked": True, "allocator_exit_code": 2, "provider_mutation_attempted": True,
    "provider_mutation_evidence": {"status": "allocator_boundary_artifacts_present",
        "admission_artifact_present": True, "bound_request_artifact_present": False,
        "adapter_output_artifact_present": True, "terminal_result_artifact_present": False},
    "blockers": ["allocator_terminal_result_missing", "canonical_allocator_nonzero_exit"]}


def _refusal(*blockers):
    return {"status": "blocked", "blockers": list(blockers), "provider_mutations_performed": 0, "retry_cap": 0}


@pytest.mark.parametrize("receipt,adapter_output,basis", [
    (NOT_INVOKED, None, "allocator_not_invoked"),
    ({**NOT_INVOKED, "allocator_invoked": True, "allocator_exit_code": 1,
      "provider_mutation_evidence": {"status": "absent_before_paid_admission"}}, None, "absent_before_paid_admission"),
    (ADMISSION_REFUSED, _refusal("account_monthly_spend_report_stale"), "paid_admission_refused"),
])
def test_a_receipt_that_proves_no_provider_was_reached_is_recovered(tmp_path, receipt, adapter_output, basis):
    run, profile = _receipt_run(tmp_path, receipt=receipt, adapter_output=adapter_output)
    observed = launch_recovery.observe_without_result(directory=run, profile=profile,
                                                      attempt={"schema_version": FREE}, now=200)
    assert observed["recoverable"] and observed["kind"] == launch_recovery.RECEIPT_KIND
    assert observed["proof"]["basis"] == basis
    assert launch_recovery.receipt_proof_still_holds(run / "launch_receipt.json")


@pytest.mark.parametrize("receipt,adapter_output", [
    # A reached monthly limit is a money stop, not an infrastructure miss.
    (ADMISSION_REFUSED, _refusal("account_monthly_spend_limit_reached")),
    (ADMISSION_REFUSED, {**_refusal("account_monthly_spend_report_stale"), "instance_id": "1"}),
    (ADMISSION_REFUSED, {**_refusal("account_monthly_spend_report_stale"), "provider_mutations_performed": 1}),
    (ADMISSION_REFUSED, None),
    ({**ADMISSION_REFUSED, "provider_mutation_evidence": {**ADMISSION_REFUSED["provider_mutation_evidence"],
                                                          "terminal_result_artifact_present": True}},
     _refusal("account_monthly_spend_report_stale")),
    ({**NOT_INVOKED, "provider_mutation_attempted": True}, None),
    # Privacy and rights answer the job; they are never retried as a miss.
    ({**NOT_INVOKED, "blockers": ["consent_revoked"]}, None),
])
def test_a_receipt_that_cannot_prove_zero_spend_is_a_named_blocker(tmp_path, receipt, adapter_output):
    run, profile = _receipt_run(tmp_path, receipt=receipt, adapter_output=adapter_output)
    observed = launch_recovery.observe_without_result(directory=run, profile=profile,
                                                      attempt={"schema_version": FREE}, now=200)
    assert observed["recoverable"] is False and observed["blockers"][0] == launch_recovery.UNPROVEN_BLOCKER
    assert not launch_recovery.receipt_proof_still_holds(run / "launch_receipt.json")


def test_a_proven_receipt_of_a_paid_source_attempt_is_named_not_retried(tmp_path):
    run, profile = _receipt_run(tmp_path, receipt=NOT_INVOKED)
    observed = launch_recovery.observe_without_result(directory=run, profile=profile,
        attempt={"schema_version": "task_evaluation_scene_attempt.v1"}, now=200)
    assert observed["recoverable"] is False and launch_recovery.UNPROVEN_BLOCKER in observed["blockers"]


def test_a_dispatcher_that_never_wrote_a_receipt_is_named_once_its_timeout_has_passed(tmp_path):
    run, profile = _receipt_run(tmp_path, started_at=1000, ttl=3600)
    deadline = 1000 + 3600 + 300 + launch_recovery.LOST_DISPATCHER_GRACE_SECONDS
    for now in (1000, deadline - 1):
        assert launch_recovery.observe_without_result(directory=run, profile=profile,
                                                      attempt={"schema_version": FREE}, now=now) is None
    observed = launch_recovery.observe_without_result(directory=run, profile=profile,
                                                      attempt={"schema_version": FREE}, now=deadline)
    assert observed == {"recoverable": False, "blockers": [launch_recovery.LOST_BLOCKER],
                        "result": recovery.record(run / "launch_started.json")}


def test_observe_failure_reads_the_receipt_when_the_allocator_result_is_missing(tmp_path, monkeypatch):
    _intent, first, observed, config, _evidence = capacity_fixture(tmp_path, monkeypatch)
    refs = observed["records"]
    run = Path(refs["launch"]["path"]).parent
    Path(refs["result"]["path"]).unlink()
    launch = json.loads(Path(refs["launch"]["path"]).read_text())
    launch.update(NOT_INVOKED)
    write(Path(refs["launch"]["path"]), launch, "receipt_digest")
    write(run / "launch_started.json", {"launch_id": run.name, "started_at": _iso(150)}, "started_digest")
    free = _free_attempt(first, refs)
    found = _observe(free, config, refs)
    assert found["recoverable"] and found["kind"] == launch_recovery.RECEIPT_KIND
    assert found["values"]["result"]["result_digest"] == launch["receipt_digest"]
    assert found["values"]["configuration_attempt"]["attempt_id"] == "scene-configuration-child"
    named = _observe(first, config, refs)
    assert named["recoverable"] is False and named["blockers"][0] == launch_recovery.UNPROVEN_BLOCKER


# -- settlement: the retired row keeps only what its records cannot prove unspent --------------------

def _no_result_settlement(tmp_path, monkeypatch, receipt):
    from tests.test_terminal_scene_attempt_settlement import COMMIT, _settle, _website_preparation
    from blueprint_pipeline.task_evaluation_retained_controls_evidence import validated_cancellation
    fx, row_id = _website_preparation(tmp_path, monkeypatch)
    run = fx["launches"] / fx["launch_id"]
    profile = _seal(run / "launch_profile.json", {"profile_id": "p", "allocator": {"argv": []}}, "profile_digest")
    _seal(run / "launch_receipt.json", {"launch_id": fx["launch_id"], "source_commit": COMMIT, "status": "blocked",
        "launch_profile_digest": profile["profile_digest"], **receipt}, "receipt_digest")
    row = json.loads((fx["directory"] / "attempts" / (row_id + ".json")).read_text())
    rows = [row]
    assert intake._counted_scene_configuration_attempts(fx["directory"], rows) == 1
    _settle(fx, source_factory=fx["factory"])
    return fx, rows, validated_cancellation(fx["directory"], row)


def test_a_receipt_proven_zero_spend_launch_settles_to_zero_and_is_no_paid_attempt(tmp_path, monkeypatch):
    fx, rows, receipt = _no_result_settlement(tmp_path, monkeypatch, NOT_INVOKED)
    assert receipt["settled_spend"]["retained_spend_usd"] == 16.76  # the sealed receipt is never rewritten
    assert settlement.budget_retained_hold(receipt) == {
        "basis": "launch_refused_before_provider", "retained_spend_usd": 0.0, "counts_as_attempt": False}
    assert intake._counted_scene_configuration_attempts(fx["directory"], rows) == 0


def test_an_unproven_receipt_keeps_its_full_hold(tmp_path, monkeypatch):
    fx, rows, receipt = _no_result_settlement(tmp_path, monkeypatch, {**NOT_INVOKED, "provider_mutation_attempted": True})
    assert settlement.budget_retained_hold(receipt)["retained_spend_usd"] == 16.76
    assert intake._counted_scene_configuration_attempts(fx["directory"], rows) == 1


@pytest.mark.parametrize("adapter,expected", [(NO_CREATE, 5.76), (None, 16.76),
                                              ({**NO_CREATE, "provider_create_attempted": True}, 16.76)])
def test_no_gpu_offer_settles_to_proven_spend_only_with_the_adapters_proof(tmp_path, monkeypatch, adapter, expected):
    from tests.test_terminal_scene_attempt_settlement import COMMIT, _settle, _website_preparation, _write
    from blueprint_pipeline.task_evaluation_retained_controls_evidence import validated_cancellation
    fx, row_id = _website_preparation(tmp_path, monkeypatch)
    request = json.loads(Path(json.loads(Path(fx["factory"]["path"]).read_text())
                              ["submission_request"]["path"]).read_text())
    run = fx["launches"] / fx["launch_id"]
    result = {"schema_version": "task_evaluation_scene_configuration_vast_result.v1", "run_id": request["run_id"],
              "source_commit": COMMIT, "status": "blocked", "provider_mutations_performed": 0,
              "continuing_spend_from_this_run": False, "provider_runtime_output_zip_path": None}
    if adapter is not None:
        result["provider_adapter_result_path"] = str(_write(run / "adapter.json", adapter))
    result_path = _write(run / "result.json", result)
    teardown_path = _write(run / "teardown.json", {"schema_version": "vast_teardown_manifest.v1",
        "status": "completed", "vast_instance_ids": [], "teardown_actions_performed": [],
        "continuing_spend_from_this_run": False})
    def ref(path):
        return {**settlement._file(path), "exists": True}
    _write(run / "launch_receipt.json", {"launch_id": fx["launch_id"], "source_commit": COMMIT, "status": "blocked",
        "terminal_evidence": {"result": ref(result_path), "artifacts": {"teardown_manifest_path": ref(teardown_path)}}})
    _settle(fx, source_factory=fx["factory"])
    receipt = validated_cancellation(fx["directory"], json.loads((fx["directory"] / "attempts" / (row_id + ".json")).read_text()))
    # 5.76 is the upper bound of the API work the request allowed before the GPU search, as for any $0 refusal.
    assert settlement.budget_retained_hold(receipt)["retained_spend_usd"] == expected


# -- max_paid_attempts binds a Website scene's scene-configuration rows -----------------------------

def _prep_only(tmp_path, attempts, spend=26.0):
    from tests.test_terminal_scene_attempt_settlement import _intent
    root = tmp_path / "intents"
    return root, _intent(root, spend=spend, attempts=attempts)


def _configure(root, intent, name, cost=5.0):
    from tests.test_terminal_scene_attempt_settlement import _reserve
    return _reserve(root, intent, "scene-configuration-" + name, cost)


def test_a_reservation_past_max_paid_attempts_is_refused(tmp_path):
    root, intent = _prep_only(tmp_path, attempts=2)
    assert _configure(root, intent, "a")[intake.PAID_ATTEMPT_FIELD] is True
    assert _configure(root, intent, "b")[intake.PAID_ATTEMPT_FIELD] is True
    with pytest.raises(intake.SceneIntakeError, match="attempt_cap_exhausted"):
        _configure(root, intent, "c")
    assert _configure(root, intent, "a")["attempt_id"] == "scene-configuration-a"  # an idempotent re-read still works


def test_rows_reserved_before_the_count_existed_do_not_strand_an_in_flight_job(tmp_path):
    root, intent = _prep_only(tmp_path, attempts=1)
    directory = root / intent["intent_id"]
    for name in ("a", "b"):  # two rows as the previous release sealed them: no marker
        row = _configure(root, intent, name)
        path = directory / "attempts" / (row["attempt_id"] + ".json")
        legacy = {key: value for key, value in row.items() if key not in {intake.PAID_ATTEMPT_FIELD, "attempt_digest"}}
        path.write_text(json.dumps(intake._seal(legacy, "attempt_digest")))
    assert _configure(root, intent, "a").get(intake.PAID_ATTEMPT_FIELD) is None  # in-flight row re-reads unchanged
    assert _configure(root, intent, "next")[intake.PAID_ATTEMPT_FIELD] is True  # counted from now on
    with pytest.raises(intake.SceneIntakeError, match="attempt_cap_exhausted"):
        _configure(root, intent, "after")


def test_the_dollar_cap_still_binds_first(tmp_path):
    root, intent = _prep_only(tmp_path, attempts=8, spend=10.0)
    _configure(root, intent, "a", 6.0)
    with pytest.raises(intake.SceneIntakeError, match="spend_cap_exhausted"):
        _configure(root, intent, "b", 4.01)


def test_an_intent_with_paid_source_rows_keeps_counting_source_rows(tmp_path):
    from tests.test_terminal_scene_attempt_settlement import _reserve
    root, intent = _prep_only(tmp_path, attempts=1)
    _reserve(root, intent, "source-a1", 1.0)
    row = _configure(root, intent, "a")
    assert intake.PAID_ATTEMPT_FIELD not in row


# -- the signed contract: a new grant names one retry, an old one none ------------------------------

@pytest.mark.parametrize("authority,expected", [
    ({"scene_configuration_max_retries": 1}, 1), ({"scene_configuration_max_retries": 0}, 0), ({}, 0),
    ({"scene_configuration_max_retries": 2}, 0), ({"scene_configuration_max_retries": True}, 0), (None, 0),
])
def test_the_request_carries_the_retry_its_grant_names(authority, expected):
    from blueprint_pipeline.website_preparation_contracts import scene_configuration_max_retries
    assert scene_configuration_max_retries(authority) == expected
