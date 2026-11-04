"""Each rigid robot-free probe attempt's website-grant hold settles to what it ran, proven without spend.

Beta journey step: L5 team policy evaluation. capture-447ae4f4's four funded
attempts each held their $1.00 cap on the $5 upstream ledger after teardown,
so with $0.32 already used the next $1.00 hold was refused (2026-11-04 01:07Z)
while their ledgers estimated $0.62. The first tests rebuild that queue in the
exact on-disk shape the host has (the queue's attempt records, the work volume's
allocator job directories, the adapter's teardown manifest, budget ledger,
adapter result and avoidlist) and settle it; the rest run the real runner, lane
and allocator with the lane tests' fakes (the WebApp's reply, the Vast transport).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline import rigid_robot_free_probe_runner as runner
from blueprint_pipeline import rigid_robot_free_probe_settlement as settlement
from blueprint_pipeline import website_task_context
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.rigid_robot_free_probe_bundle import current_probe_state
from tests.test_rigid_robot_free_probe_lane import build_world, fake_transport, runner_hooks

KEY = "003c7a1ef84d169a04878b894e412a3a"
REQUEST = "capture-447ae4f4-c253-44ae-a897-c999245e0ef7"
CAPTURE = "walkthrough-" + REQUEST
TASK = "sha256:" + "53e1" * 16
CRASH = ["provider_runtime_result_missing_from_output_zip",
         ("rigid_robot_free_probe_failed_at_simulation_app_launched:AttributeError:'DroidCameraCfgWithExteriorRight' "
          "object has no attribute 'set_use_tiled_camera'")]
NO_START_BLOCKERS = ["rigid_robot_free_probe_candidate_policy_query_status_missing",
                     "rigid_robot_free_probe_provider_output_zip_missing", "rigid_robot_free_probe_runtime_not_completed",
                     "task_evaluation_artifact_role_missing:provider_runtime_evidence", "vast_heartbeat_instance_exited"]
# capture-447ae4f4's four funded attempts: job, index, Vast instance, ledger estimate, allocator status, blockers.
FUNDED = [("bdfe6a51ccf61583", 0, 59017963, 0.133313, "blocked", CRASH),
          ("bdfe6a51ccf61583", 1, 59019608, 0.16524, "blocked", CRASH),
          ("bdfe6a51ccf61583", 2, 59023542, 0.234989, "completed", []),
          ("9319f26e660a4c7c", 0, 59035045, 0.087939, "blocked", NO_START_BLOCKERS)]
NO_START = {"instance": 59035045, "machine": 71063, "offer": 51959245}


def _write(path: Path, value: Any) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return raw


def _ref(path: Path, raw: bytes) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": len(raw), "sha256": "sha256:" + hashlib.sha256(raw).hexdigest()}


def _ingest(attempt: Path, value: dict[str, Any]) -> None:
    record = {"schema_version": runner.INGEST_SCHEMA, **value, "settled_at_iso": "2026-11-03T21:46:23+00:00",
              "record_digest": ""}
    record["record_digest"] = canonical_digest(record, digest_field="record_digest")
    _write(attempt / runner.INGEST_FILENAME, record)


def evidence(allocator: Path, *, instance: int, estimate: float, no_start: bool = False,
             machine: int | None = None, offer: int | None = None, staged_avoidlist: Path | None = None
             ) -> dict[str, Any]:
    """The Vast adapter's records for one torn-down run, as it writes them under the allocator job directory."""

    run = allocator / "attempts" / "attempt_001" / "vast_provider_run"
    decision = {"schema_version": "vast_retained_instance_decision.v1", "status": "teardown_required",
                "requested": False, "instance_ids": [instance], "container_health_proven": not no_start,
                "gpu_health_proven": not no_start, "blockers": ["retention_not_requested"]}
    teardown = _write(run / "vast_teardown_manifest.json", {
        "schema_version": "vast_teardown_manifest.v1", "status": "completed", "vast_instance_ids": [instance],
        "teardown_actions_performed": [
            {"instance_id": instance, "action": "destroy_instance", "http_status_code": 404, "status": "completed",
             "reason": "instance_already_absent"} if no_start else
            {"instance_id": instance, "action": "destroy_instance", "http_status_code": 200,
             "response": {"success": True}, "status": "completed"}],
        "runner_gpu_teardown_completed": True, "continuing_spend_from_this_run": False, "retention_authorized": False,
        "retention_decision": decision,
        "zero_continuing_spend_scope": "all Vast instances created by this adapter were destroyed",
        "raw_secret_values_recorded": False})
    _write(run / "vast_budget_ledger.json", {
        "schema_version": "vast_budget_ledger.v1", "status": "completed", "target_spend_usd": 1.0, "hard_cap_usd": 1.0,
        "max_hourly_rate_usd": 0.8, "max_live_runtime_minutes": 60, "selected_hourly_rate_usd": 0.4555555555555555,
        "vast_instance_ids": [instance], "actual_live_runtime_seconds_observed_by_adapter": 600.5,
        "estimated_runtime_cost_usd": round(estimate - 0.011948, 6), "projected_provider_transfer_cost_usd": 0.011948,
        "provider_transfer_cost_priced": True, "estimated_cost_usd": estimate, "actual_cost_usd": None,
        "actual_cost_source": "not_available_from_instance_probe_api", "estimated_spend_under_target": True,
        "estimated_spend_under_hard_cap": True, "continuing_spend_from_this_run": False,
        "raw_secret_values_recorded": False})
    classification = {
        "schema_version": "provider_attempt_classification.v1",
        "classification": "pre_execution_provider_null" if no_start else "provider_bundle_attempt_started",
        "provider_bundle_started": not no_start, "provider_entrypoint_started": not no_start,
        "provider_output_returned": not no_start, "scientific_attempt_consumed": not no_start,
        "pre_execution_requeue_eligible_in_principle": no_start, "automatic_requeue_authorized": False,
        "automatic_requeue_executed": False, "maximum_automatic_requeues": 0,
        "authority_required_for_next_provider_mutation": True,
        "blockers": ["vast_heartbeat_instance_exited"] if no_start else []}
    adapter = _write(run / "vast_provider_adapter_result.json", {
        "schema_version": "vast_provider_adapter_result.v1", "status": "failed" if no_start else "completed",
        "reason": "vast_probe_failed" if no_start else None, "estimated_cost_usd": estimate,
        "continuing_spend_from_this_run": False, "provider_create_attempted": True, "vast_instance_ids": [instance],
        "blockers": ["vast_heartbeat_instance_exited"] if no_start else [],
        "provider_attempt_classification": classification, "retained_owned": False,
        "machine_avoidlist_path": str(allocator / settlement.JOB_AVOIDLIST_FILENAME)})
    avoided = settlement._avoidlist_rows(staged_avoidlist) if staged_avoidlist else []
    if no_start:
        avoided.append({"machine_id": machine, "instance_id": instance, "offer_id": offer, "gpu_model_slug": "rtx_4090",
                        "reason": settlement.NO_START_REASON, "blockers": ["vast_heartbeat_instance_exited"],
                        "retry_policy": "exclude_persistently_across_sibling_jobs_until_manual_review"})
    if avoided:
        _write(allocator / settlement.JOB_AVOIDLIST_FILENAME, {
            "schema_version": "vast_machine_avoidlist.v1", "status": "completed",
            "machine_ids": sorted({row["machine_id"] for row in avoided}), "entries": avoided,
            "raw_secret_values_recorded": False})
    return {"adapter_result_path": str(run / "vast_provider_adapter_result.json"),
            "teardown_manifest_path": str(run / "vast_teardown_manifest.json"),
            "provider_closeout": {"adapter_result": _ref(run / "vast_provider_adapter_result.json", adapter),
                                  "teardown_manifest": _ref(run / "vast_teardown_manifest.json", teardown),
                                  "estimated_cost_usd": estimate, "provider_zero_confirmed": True,
                                  "warm_session_retained": False, "all_staged_objects_absent": True},
            "independent_watchdog_close": {"schema_version": "vast_independent_watchdog_handoff.v1",
                                           "status": "provider_terminal", "instance_ids": [instance],
                                           "provider_absence_confirmed": True, "provider_mutations_performed": 0},
            "estimated_cost_usd": estimate, "continuing_spend_from_this_run": False, "all_staged_objects_absent": True}


def host_attempt(queue: Path, work_root: Path, job: str, index: int, *, capture_root: Path, instance: int | None,
                 estimate: float | None, status: str, blockers: list[str], ingest: dict[str, Any]) -> Path:
    """One staged attempt as the host keeps it: records in the queue, its allocator job on the work volume."""

    name = f"{KEY}-{job}"
    attempt = queue / "jobs" / name / "attempts" / f"{index:02d}"
    work = work_root / "jobs" / name / "attempts" / f"{index:02d}"
    allocator = work / "allocator"
    (work / "bundle").mkdir(parents=True)
    _write(attempt / runner.ATTEMPT_FILENAME, {"attempt_id": f"{name}-a{index:02d}", "argv": [],
                                               "authority_digest": "sha256:" + "a" * 64, "work_dir": str(work),
                                               "staged_at_iso": "2026-11-03T21:33:47+00:00"})
    binding = {"program_id": "blueprint-free-beta-evaluated-pilot", "probe_kind": "rigid-robot-free-probe",
               "capture_root": str(capture_root), "resource_name": f"blueprint-rigid-robot-free-probe-{index:032x}",
               "attempt": f"{job}-{index}", "hard_cap_usd": 1.0, "max_hourly_rate_usd": 0.8,
               "hard_ttl_seconds": 3600, "retry_cap": 0, "maximum_provider_allocations": 1}
    admission = {"schema_version": "paid_lane_admission.v1", "status": "admitted" if instance else "blocked",
                 "allocation_binding": binding, "allocation_binding_digest": canonical_digest(binding),
                 "spend_authority": "website_preparation_grant", "website_preparation_admission_digest": None}
    if instance is not None:
        funding = {"allocation_binding_digest": admission["allocation_binding_digest"], "blockers": [],
                   "expires_at_epoch": 1794324997, "external_disclosure_allowed": True, "maximum_cost_usd": 1,
                   "provider": "vast", "request_count": 1, "resource_class": "gpu_render",
                   "schema_version": "paid_lane_admission.v1", "sponsorship_digest": "sha256:" + "c8" * 32,
                   "status": "admitted", "task_context_digest": TASK}
        _write(allocator / runner.FUNDING_FILENAME, funding)
        admission["website_preparation_admission_digest"] = canonical_digest(funding)
        no_start = instance == NO_START["instance"]
        result = {"schema_version": "rigid_robot_free_probe_vast_run.v1", "status": status, "blockers": blockers,
                  "bundle_sha256": "sha256:" + "b" * 64,
                  **evidence(allocator, instance=instance, estimate=estimate, no_start=no_start,
                             machine=NO_START["machine"], offer=NO_START["offer"])}
        _write(allocator / "adp_arena_vast_result.json", result)
    else:
        allocator.mkdir(parents=True)
        result = {"schema_version": "rigid_robot_free_probe_vast_run.v1", "status": "blocked", "blockers": blockers,
                  "provider_mutations_performed": 0, "retry_cap": 0, "candidate_policy_queried": False}
    _write(attempt / "paid_admission.json", admission)
    _write(attempt / runner.RESULT_FILENAME, result)
    _write(attempt / runner.INTENT_FILENAME, {"observed_at_iso": "2026-11-03T21:33:48+00:00"})
    _ingest(attempt, ingest)
    return attempt


def capture_447_queue(tmp_path: Path) -> dict[str, Any]:
    """The rigid probe queue and work volume exactly as capture-447ae4f4 left them (2026-11-04 01:07Z)."""

    queue, work_root = tmp_path / "rigid-robot-free-probes", tmp_path / "rigid-robot-free-probe-work"
    capture = tmp_path / "pubsub-handoffs" / "scenes" / f"site-{REQUEST}" / "captures" / CAPTURE
    _write(capture / "pipeline" / "website_task_context.json", {
        "schema_version": "website_site_task_context.v1", "request_id": REQUEST, "scene_id": f"site-{REQUEST}",
        "capture_id": CAPTURE, "context_digest": TASK})
    for index, reason in enumerate((
            "native_task_arena_bundle_robot_arm_path_blocked:robot_rigid_pick_place_unreachable:approach",
            "rigid_robot_free_probe_rigid_task_start_pose_not_spawn_pose:14.7mm>2.0mm")):
        _ingest(queue / "jobs" / f"{KEY}-936e3731c8902e00" / "attempts" / f"{index:02d}",
                {"state": "not_staged", "funded": False, "allocated": False, "reason": reason})
    attempts = {}
    for job, index, instance, estimate, status, blockers in FUNDED:
        ingest = {"funded": True, "allocated": True, "allocator_status": status, "allocator_blockers": blockers,
                  "estimated_cost_usd": estimate,
                  "state": "undecided" if status == "completed" else "allocated_blocked"}
        attempts[(job, index)] = host_attempt(queue, work_root, job, index, capture_root=capture, instance=instance,
                                              estimate=estimate, status=status, blockers=blockers, ingest=ingest)
    refused = ["paid_resource_admission_has_blockers", "paid_resource_admission_not_admitted",
               "rigid_robot_free_probe_website_grant_exhausted"]
    attempts[("9319f26e660a4c7c", 1)] = host_attempt(
        queue, work_root, "9319f26e660a4c7c", 1, capture_root=capture, instance=None, estimate=None, status="blocked",
        blockers=refused, ingest={"funded": False, "allocated": False, "allocator_status": "blocked",
                                  "allocator_blockers": refused, "estimated_cost_usd": None,
                                  "state": runner.GRANT_REFUSED_STATE})
    _write(queue / "jobs" / f"{KEY}-bdfe6a51ccf61583" / runner.TERMINAL_FILENAME,
           {"schema_version": runner.TERMINAL_SCHEMA, "status": "attempts_exhausted"})
    return {"queue": queue, "work_root": work_root, "capture": capture, "attempts": attempts}


class WebApp:
    """The WebApp's preparation ledger for one request, as ``settleWebsitePreparationSpend`` keeps it."""

    def __init__(self, used_usd: float = 0.32, cap_usd: float = 5.0) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.calls: list[dict[str, Any]] = []
        self.refusal: str | None = None
        self.used, self.cap = used_usd, cap_usd

    def hold(self, binding: str, maximum: float = 1.0) -> None:
        self.rows[binding[7:]] = {"maximum_cost_usd": maximum, "task_context_digest": TASK, "settlement": None}

    def held_usd(self) -> float:
        return round(self.used + sum(row["settlement"]["actual_cost_usd"] if row["settlement"]
                                     else row["maximum_cost_usd"] for row in self.rows.values()), 6)

    def __call__(self, *, capture_id: str, operation: str, payload: dict[str, Any],
                 source_kind: str = "website") -> dict[str, Any]:
        assert operation == "preparation-settlement" and source_kind == "website"
        self.calls.append({"capture_id": capture_id, "payload": copy.deepcopy(payload)})
        if self.refusal:
            raise ValueError(f"website_control_preparation-settlement_http_409:{self.refusal}")
        command = payload["settlement"]
        row = self.rows.get(command["allocation_binding_digest"][7:])
        if (row is None or command["task_context_digest"] != row["task_context_digest"]
                or not command["provider_charge_amount_usd"] <= row["maximum_cost_usd"]):
            raise ValueError("website_control_preparation-settlement_http_409:website_scene_preparation_settlement_invalid")
        receipt = {**command, "actual_cost_usd": command["provider_charge_amount_usd"], "status": "settled"}
        if row["settlement"] is not None and row["settlement"] != receipt:
            raise ValueError("website_control_preparation-settlement_http_409:idempotency_conflict")
        row["settlement"] = receipt
        return copy.deepcopy(receipt)


def _bindings(attempts: dict[Any, Path]) -> dict[Any, str]:
    return {key: json.loads((path / "paid_admission.json").read_text())["allocation_binding_digest"]
            for key, path in attempts.items()}


def test_capture_447s_four_torn_down_attempts_settle_to_their_ledger_estimates_once(tmp_path, monkeypatch) -> None:
    host = capture_447_queue(tmp_path)
    webapp = WebApp()
    bindings = _bindings(host["attempts"])
    for job, index, *_rest in FUNDED:
        webapp.hold(bindings[(job, index)])
    monkeypatch.setattr(website_task_context, "website_webapp_request", webapp)
    # $0.32 used and four $1.00 holds: the next $1.00 hold does not fit the $5 upstream cap.
    assert webapp.held_usd() == 4.32 and webapp.held_usd() + 1.0 > webapp.cap

    rows = runner.settle_spend_under(host["queue"])

    assert [row["status"] for row in rows] == ["settled"] * 4 and all(row["settled_now"] for row in rows)
    sent = [call["payload"] for call in webapp.calls]
    assert {(call["request_id"], call["scene_id"]) for call in sent} == {(REQUEST, f"site-{REQUEST}")}
    assert {call["capture_id"] for call in webapp.calls} == {CAPTURE}
    by_instance = {int(call["settlement"]["instance_id"]): call["settlement"] for call in sent}
    assert {instance: row["provider_charge_amount_usd"] for instance, row in by_instance.items()} == {
        instance: estimate for _job, _index, instance, estimate, *_ in FUNDED}
    for job, index, instance, estimate, *_ in FUNDED:
        command = by_instance[instance]
        attempt = host["attempts"][(job, index)]
        run = (host["work_root"] / "jobs" / f"{KEY}-{job}" / "attempts" / f"{index:02d}" / "allocator" / "attempts"
               / "attempt_001" / "vast_provider_run")
        assert command == {
            "task_context_digest": TASK, "allocation_binding_digest": bindings[(job, index)], "provider": "vast",
            "instance_id": str(instance), "provider_charge_basis": "adapter_budget_ledger_estimate",
            "provider_charge_amount_usd": estimate,
            "budget_ledger_digest": "sha256:" + hashlib.sha256((run / "vast_budget_ledger.json").read_bytes()).hexdigest(),
            "execution_result_digest": "sha256:" + hashlib.sha256(
                (attempt / runner.RESULT_FILENAME).read_bytes()).hexdigest(),
            "teardown_receipt_digest": "sha256:" + hashlib.sha256(
                (run / "vast_teardown_manifest.json").read_bytes()).hexdigest(),
            "provider_zero_digest": canonical_digest(json.loads(
                (attempt / runner.RESULT_FILENAME).read_text())["provider_closeout"]),
            "continuing_spend_from_this_run": False, "all_staged_objects_absent": True}
        # The record beside the attempt names the estimate as an estimate, with its source.
        record = json.loads((attempt / settlement.SETTLEMENT_FILENAME).read_text())
        assert record["settled_cost_usd"] == estimate and record["charge_basis"] == "adapter_budget_ledger_estimate"
        assert record["charge_source"] == {"budget_ledger_path": str(run / "vast_budget_ledger.json"),
                                           "actual_cost_usd": None,
                                           "actual_cost_source": "not_available_from_instance_probe_api"}
        assert record["reserved_maximum_cost_usd"] == 1.0 and record["command"] == command
        assert record["provider_no_start"] is (instance == NO_START["instance"])
        assert record["record_digest"] == canonical_digest(record, digest_field="record_digest")
    # $0.32 + $0.621481: the next $1.00 hold fits.
    assert math.isclose(webapp.held_usd(), 0.941481) and webapp.held_usd() + 1.0 <= webapp.cap
    # Unfunded attempts (never staged, the refused grant) are never settled.
    for key in (("9319f26e660a4c7c", 1),):
        assert not (host["attempts"][key] / settlement.SETTLEMENT_FILENAME).exists()
    assert not list((host["queue"] / "jobs" / f"{KEY}-936e3731c8902e00").rglob(settlement.SETTLEMENT_FILENAME))
    # Idempotent: the next pass reads its records and sends nothing.
    assert runner.settle_spend_under(host["queue"]) == [] and len(webapp.calls) == 4
    # Instance 59035045 never ran our container: a provider no-start, so the request has spent none of
    # its two funded attempts, and machine 71063 is in the lane's avoidlist for every later attempt.
    job = host["queue"] / "jobs" / f"{KEY}-9319f26e660a4c7c"
    assert runner._counts(job) == (0, 0) and runner._provider_no_starts(job) == 1
    assert runner._counts(host["queue"] / "jobs" / f"{KEY}-bdfe6a51ccf61583") == (3, 0)
    no_start = settlement.provider_no_start(host["attempts"][("9319f26e660a4c7c", 0)],
                                            runner.attempt_work_dir(host["attempts"][("9319f26e660a4c7c", 0)]))
    assert no_start["machine_id"] == 71063 and no_start["instance_id"] == 59035045
    assert no_start["classification"] == "pre_execution_provider_null"
    shared = json.loads(settlement.lane_avoidlist_path(host["queue"]).read_text())
    assert shared["machine_ids"] == [71063] and shared["entries"][0]["reason"] == settlement.NO_START_REASON


def _mutate_json(path: Path, change) -> None:
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _rehash(attempt: Path, name: str) -> None:
    """Re-record a changed adapter file's digest in the result, as a consistent (but unproven) run would."""

    def change(result: dict[str, Any]) -> None:
        ref = result["provider_closeout"][name]
        ref["sha256"] = "sha256:" + hashlib.sha256(Path(ref["path"]).read_bytes()).hexdigest()
    _mutate_json(attempt / runner.RESULT_FILENAME, change)


def _teardown(attempt: Path) -> Path:
    return Path(json.loads((attempt / runner.RESULT_FILENAME).read_text())["teardown_manifest_path"])


def _set(path_of, change, rehash: str | None = None):
    def apply(attempt: Path) -> None:
        _mutate_json(path_of(attempt), change)
        if rehash:
            _rehash(attempt, rehash)
    return apply


def RESULT(attempt: Path) -> Path:
    return attempt / runner.RESULT_FILENAME


def LEDGER(attempt: Path) -> Path:
    return _teardown(attempt).parent / "vast_budget_ledger.json"


@pytest.mark.parametrize(("case", "mutate"), [
    ("spend_continues", _set(RESULT, lambda v: v.update(continuing_spend_from_this_run=True))),
    ("objects_left_staged", _set(RESULT, lambda v: v.update(all_staged_objects_absent=None))),
    ("provider_zero_unconfirmed", _set(RESULT, lambda v: v["provider_closeout"].update(provider_zero_confirmed=False))),
    ("watchdog_absence_unconfirmed",
     _set(RESULT, lambda v: v["independent_watchdog_close"].update(provider_absence_confirmed=False))),
    ("teardown_changed_after_its_digest", _set(_teardown, lambda v: v.update(note="rewritten"))),
    ("teardown_not_completed", _set(_teardown, lambda v: v.update(status="blocked"), "teardown_manifest")),
    ("destroy_not_completed",
     _set(_teardown, lambda v: v["teardown_actions_performed"][0].update(status="failed"), "teardown_manifest")),
    ("two_instances", _set(_teardown, lambda v: v.update(vast_instance_ids=[59017963, 59017964]),
                           "teardown_manifest")),
    ("instance_retained", _set(_teardown, lambda v: v.update(retention_authorized=True), "teardown_manifest")),
    ("billed_actual_present", _set(LEDGER, lambda v: v.update(actual_cost_usd=0.5, actual_cost_source="billing"))),
    ("transfer_unpriced", _set(LEDGER, lambda v: v.update(provider_transfer_cost_priced=False))),
    ("ledger_for_another_instance", _set(LEDGER, lambda v: v.update(vast_instance_ids=[1]))),
    ("estimate_inconsistent", _set(RESULT, lambda v: v.update(estimated_cost_usd=0.2))),
    ("estimate_above_the_reservation", lambda attempt: (
        _set(LEDGER, lambda v: v.update(estimated_cost_usd=1.2))(attempt),
        _set(RESULT, lambda v: (v.update(estimated_cost_usd=1.2),
                                v["provider_closeout"].update(estimated_cost_usd=1.2)))(attempt))),
    ("funding_of_another_attempt", _set(lambda attempt: attempt / "paid_admission.json",
                                        lambda v: v.update(website_preparation_admission_digest="sha256:" + "0" * 64))),
    ("teardown_outside_the_attempt", lambda attempt: _set(RESULT, lambda v: v["provider_closeout"]["teardown_manifest"]
                                                          .update(path="/elsewhere/vast_teardown_manifest.json"))(
                                                              attempt)),
    ("no_result_yet", lambda attempt: (attempt / runner.RESULT_FILENAME).unlink()),
])
def test_a_run_whose_records_do_not_prove_its_teardown_keeps_its_full_hold(tmp_path, monkeypatch, case, mutate) -> None:
    host = capture_447_queue(tmp_path)
    webapp = WebApp()
    bindings = _bindings(host["attempts"])
    for job, index, *_rest in FUNDED:
        webapp.hold(bindings[(job, index)])
    monkeypatch.setattr(website_task_context, "website_webapp_request", webapp)
    attempt = host["attempts"][("bdfe6a51ccf61583", 0)]
    mutate(attempt)
    outcome = runner.settle_attempt_funding(attempt)
    assert outcome["status"] in {"held", "pending"}, outcome
    assert webapp.calls == [] and not (attempt / settlement.SETTLEMENT_FILENAME).exists()
    # The other attempts are unaffected.
    assert len([row for row in runner.settle_spend_under(host["queue"]) if row["status"] == "settled"]) == 3


def test_a_refused_or_unreachable_settlement_keeps_the_hold_and_the_next_pass_resends_it(tmp_path, monkeypatch) -> None:
    host = capture_447_queue(tmp_path)
    webapp = WebApp()
    bindings = _bindings(host["attempts"])
    for job, index, *_rest in FUNDED:
        webapp.hold(bindings[(job, index)])
    monkeypatch.setattr(website_task_context, "website_webapp_request", webapp)
    webapp.refusal = "website_scene_request_invalid"  # e.g. a WebApp without the estimate settlement yet
    rows = runner.settle_spend_under(host["queue"])
    assert [row["status"] for row in rows] == ["deferred"] * 4 and webapp.held_usd() == 4.32
    assert not list(host["queue"].rglob(settlement.SETTLEMENT_FILENAME))
    webapp.refusal = None
    first = [call["payload"]["settlement"] for call in webapp.calls]
    assert [row["status"] for row in runner.settle_spend_under(host["queue"])] == ["settled"] * 4
    # Each retry resends exactly the command that was refused (the WebApp's idempotency key).
    assert [call["payload"]["settlement"] for call in webapp.calls[4:]] == first
    # A lost answer: the WebApp settled but no record was kept, so the same command returns the same receipt.
    attempt = host["attempts"][("bdfe6a51ccf61583", 2)]
    binding = _bindings(host["attempts"])[("bdfe6a51ccf61583", 2)]
    (attempt / settlement.SETTLEMENT_FILENAME).unlink()
    assert runner.settle_attempt_funding(attempt)["status"] == "settled"
    sent = [call["payload"]["settlement"] for call in webapp.calls
            if call["payload"]["settlement"]["allocation_binding_digest"] == binding]
    assert len(sent) == 3 and sent[0] == sent[1] == sent[2]
    # A receipt that does not echo the command is never recorded.
    (attempt / settlement.SETTLEMENT_FILENAME).unlink()
    monkeypatch.setattr(website_task_context, "website_webapp_request",
                        lambda **kwargs: {**webapp(**kwargs), "actual_cost_usd": 1.0})
    assert runner.settle_attempt_funding(attempt)["status"] == "deferred"
    assert not (attempt / settlement.SETTLEMENT_FILENAME).exists()


# --- the runner: settlement order and the provider no-start rule ----------------------------------


def shaped_transport(world: dict[str, Any], outcomes: list[str]):
    """The lane tests' transport and GPU host, leaving the adapter's teardown records behind each run.

    ``outcomes`` names each call: ``run`` runs the real worker on the sealed bundle; ``no_start`` is a
    machine that never starts our container (machines 71063, 71064, ... in turn).
    """

    inner = fake_transport(world)
    world["avoidlists_read"] = []

    def run(**kwargs: Any) -> dict[str, Any]:
        if not kwargs["execute"]:
            return inner(**kwargs)
        index = len(world["avoidlists_read"])
        staged = kwargs.get("machine_avoidlist_path")
        world["avoidlists_read"].append(
            sorted(row["machine_id"] for row in settlement._avoidlist_rows(Path(staged))) if staged else [])
        instance = 59035045 + index
        allocator = Path(kwargs["job_dir"])
        if outcomes[index] == "no_start":
            world["transport_calls"].append(kwargs)
            return {"schema_version": kwargs["result_schema_version"], "status": "blocked",
                    "bundle_sha256": kwargs["prepared_bundle"]["bundle_sha256"], "native_control_result_path": None,
                    "candidate_policy_queried": False, "blockers": NO_START_BLOCKERS,
                    **evidence(allocator, instance=instance, estimate=0.087939, no_start=True,
                               machine=71063 + index, offer=51959245 + index,
                               staged_avoidlist=Path(staged) if staged else None)}
        # The run's own outcome (its provider zero included) stands over the teardown records.
        return {**evidence(allocator, instance=instance, estimate=0.31, staged_avoidlist=Path(staged) if staged else None),
                **inner(**kwargs)}

    return run


def _runner_world(tmp_path, monkeypatch, outcomes: list[str]) -> dict[str, Any]:
    from blueprint_pipeline import adp_isaac_lab_arena_vast as arena_vast

    world = build_world(tmp_path, monkeypatch)
    monkeypatch.setattr(arena_vast, "run_arena_native_control_vast", shaped_transport(world, outcomes))
    inner = website_task_context.website_webapp_request
    world["ledger"] = WebApp(used_usd=0.0)
    world["order"] = []

    def webapp(*, capture_id: str, operation: str, payload: dict[str, Any], source_kind: str = "website"):
        if operation == "preparation-settlement":
            world["order"].append(("settle", payload["settlement"]["allocation_binding_digest"]))
            return world["ledger"](capture_id=capture_id, operation=operation, payload=payload)
        value = inner(capture_id=capture_id, operation=operation, payload=payload)
        if operation == "preparation-spend":
            world["order"].append(("reserve", payload["spend"]["allocation_binding_digest"]))
            world["ledger"].rows[payload["spend"]["allocation_binding_digest"][7:]] = {
                "maximum_cost_usd": payload["spend"]["maximum_cost_usd"],
                "task_context_digest": payload["spend"]["task_context_digest"], "settlement": None}
        return value

    monkeypatch.setattr(website_task_context, "website_webapp_request", webapp)
    return world


def test_a_provider_no_start_is_settled_counts_toward_no_funded_attempt_and_its_machine_is_avoided(
        tmp_path, monkeypatch) -> None:
    world = _runner_world(tmp_path, monkeypatch, ["no_start", "run"])
    runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    rows = runner.run_queue(world["queue"], hooks=runner_hooks(world))
    assert rows[0]["status"] == "decided", rows
    job = runner._job_dir(world["queue"], current_probe_state(world["capture"]))
    first, second = runner._attempts(job)
    ingest = json.loads((first / runner.INGEST_FILENAME).read_text())
    assert ingest["state"] == runner.PROVIDER_NO_START_STATE and ingest["funded"] is True
    assert ingest["provider_no_start"]["machine_id"] == 71063
    assert runner._counts(job) == (1, 0) and runner._provider_no_starts(job) == 1
    # The second attempt's allocator avoided the first's machine.
    assert world["avoidlists_read"] == [[], [71063]]
    argv = world["allocator_calls"][1]
    assert argv[argv.index("--adp-machine-avoidlist") + 1] == str(second / settlement.ATTEMPT_AVOIDLIST_FILENAME)
    # Both runs' holds settled to their estimates; the no-start's before the next hold was reserved.
    reserved = [binding for kind, binding in world["order"] if kind == "reserve"]
    assert world["order"] == [("reserve", reserved[0]), ("settle", reserved[0]), ("reserve", reserved[1]),
                              ("settle", reserved[1])]
    assert [json.loads((attempt / settlement.SETTLEMENT_FILENAME).read_text())["settled_cost_usd"]
            for attempt in (first, second)] == [0.087939, 0.31]
    assert world["ledger"].held_usd() == round(0.087939 + 0.31, 6)


def test_a_second_provider_no_start_holds_the_job_by_name(tmp_path, monkeypatch) -> None:
    world = _runner_world(tmp_path, monkeypatch, ["no_start", "no_start", "run"])
    runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    rows = runner.run_queue(world["queue"], hooks=runner_hooks(world))
    assert rows[0]["status"] == runner.NO_START_REPEATED_STATUS == "rigid_robot_free_probe_provider_no_start_repeated"
    assert rows[0]["detail"]["blocker"] == runner.NO_START_REPEATED_STATUS
    assert rows[0]["detail"]["provider_no_starts"] == runner.MAX_PROVIDER_NO_STARTS == 2
    assert rows[0]["detail"]["funded_attempts"] == 0
    job = runner._job_dir(world["queue"], current_probe_state(world["capture"]))
    assert len(runner._attempts(job)) == 2 and len(world["allocator_calls"]) == 2
    assert world["avoidlists_read"] == [[], [71063]]
    shared = json.loads(settlement.lane_avoidlist_path(world["queue"]).read_text())
    assert shared["machine_ids"] == [71063, 71064]
    # Both no-starts' costs are settled; nothing is left held.
    assert all(row["settlement"] is not None for row in world["ledger"].rows.values())
    assert world["ledger"].held_usd() == round(2 * 0.087939, 6)
    # Final: the website job is held by name, and nothing further is staged or spent.
    terminal = runner.final_terminal(current_probe_state(world["capture"]), queue_root=world["queue"],
                                     release="a" * 40)
    assert terminal is not None and terminal["status"] == runner.NO_START_REPEATED_STATUS
    assert runner.run_queue(world["queue"], hooks=runner_hooks(world)) == []
    assert len(world["allocator_calls"]) == 2


def test_an_allocated_run_without_provider_zero_is_never_settled(tmp_path, monkeypatch) -> None:
    world = _runner_world(tmp_path, monkeypatch, ["run"])
    world["continuing_spend"] = None
    runner.enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    hooks = runner_hooks(world)
    assert runner.run_queue(world["queue"], hooks=hooks)[0]["reason"] == "provider_zero_unproven"
    assert runner.run_queue(world["queue"], hooks=hooks)[0]["reason"] == "provider_zero_unproven"
    assert [kind for kind, _binding in world["order"]] == ["reserve"]
    assert not list(world["queue"].rglob(settlement.SETTLEMENT_FILENAME))
    assert world["ledger"].held_usd() == 1.0
