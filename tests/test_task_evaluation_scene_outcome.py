"""C-04: every website job ends in exactly one terminal state with a plain reason.

Before this, only ``completed`` stopped the scene progression driver; every
other row was re-run every minute forever and the website could only guess
that a job had ended. These tests pin the one registry, the write-once
``outcome.json``, the 24-hour give-up rule, the driver stopping new execution
once an outcome is terminal (while its read-only joins keep closing out
authorized runs), and the optional ``outcome`` on the intent status.
"""
from __future__ import annotations

import ast
import inspect
import json
import os
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline import task_evaluation_scene_outcome as outcome
from blueprint_pipeline import task_evaluation_scene_progression as engine
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
from blueprint_pipeline.task_evaluation_scene_progression_state import advance, load_progression
from tests.test_task_evaluation_scene_intake import request as owner_request

T0 = 1_000_000.0
DAY = 24 * 60 * 60


def _intent(root: Path, *, now: float = T0, window: float = 3 * DAY, submission_id: str = "upload-1"):
    value = owner_request()
    value["submission_id"] = submission_id
    value["execution"]["expires_at_epoch"] = now + window
    value["consent"]["accepted_at_epoch"] = now - 1
    receipt = intake.stage_scene_intent(value=value, queue_root=root, authenticated_client="webapp",
                                        trusted_clients={"webapp"}, now=now)
    directory = root / receipt["intent_id"]
    return directory, json.loads((directory / "intent.json").read_text())


def _expected(reason_code: str | None):
    if reason_code is None:
        return None
    kind, terminal, category = outcome.REASONS[reason_code]
    return {"kind": kind, "terminal": terminal, "reason_category": category, "reason_code": reason_code}


# --------------------------------------------------------------------- the one registry

def test_registry_is_one_closed_vocabulary():
    assert set(outcome.KINDS) == {"setup_repair", "awaiting_spend_approval", "awaiting_site", "withdrawn", "held"}
    for code, (kind, terminal, category) in outcome.REASONS.items():
        assert kind in outcome.KINDS and category in outcome.REASON_CATEGORIES and type(terminal) is bool, code
        # Privacy, rights and capability endings are final; a person's wait never is.
        if kind in {"withdrawn", "held"}:
            assert terminal is True, code
        if kind in {"awaiting_spend_approval", "awaiting_site"}:
            assert terminal is False, code
    assert set(outcome.BLOCKER_REASONS.values()) <= set(outcome.REASONS)
    assert set(outcome.PHASE_REASONS.values()) <= set(outcome.REASONS)
    assert set(outcome.STATUS_REASONS.values()) <= set(outcome.REASONS)
    # Every category the WebApp may map has at least one reason.
    assert {row[2] for row in outcome.REASONS.values()} == set(outcome.REASON_CATEGORIES)


SEALED_CAPS = [
    ("scene_progression_retry_cap_exhausted", "retry_cap_exhausted"),
    ("scene_intake_retry_cap_exhausted", "retry_cap_exhausted"),
    ("retry_cap_exhausted", "retry_cap_exhausted"),
    ("scene_progression_submission_retry_cap_exhausted", "submission_retry_cap_exhausted"),
    ("submission_retry_cap_exhausted", "submission_retry_cap_exhausted"),
    ("scene_progression_preallocation_capacity_recovery_cap_exhausted", "capacity_recovery_cap_exhausted"),
    ("scene_intake_preallocation_capacity_recovery_cap_exhausted", "capacity_recovery_cap_exhausted"),
    ("preallocation_capacity_recovery_cap_exhausted", "capacity_recovery_cap_exhausted"),
    ("scene_intake_market_miss_recovery_cap_exhausted", "market_miss_recovery_cap_exhausted"),
    ("market_miss_recovery_cap_exhausted", "market_miss_recovery_cap_exhausted"),
]


@pytest.mark.parametrize("status,phase,blockers,reason", [
    # (a) transient or self-retrying: no outcome.
    ("preparing", "factory", [], None),
    ("running", "source_preparation", [], None),
    ("running", "previous_release", ["previous_release_attempt_not_terminal"], None),
    ("preparing", "submission_reconciliation", ["preparation_forwarding_pending"], None),
    ("awaiting_source", "source", ["website_prepared_source_pending"], None),
    ("awaiting_execution", "capacity", ["scene_whole_chain_capacity_insufficient"], None),
    ("awaiting_execution", "scene_configuration", [], None),
    ("running", "terminal_reconciliation", ["terminal_resource_closure_pending",
                                            "terminal_website_readback_pending"], None),
    ("blocked", "configuration_capacity", ["preallocation_capacity_not_recovered"], None),
    ("blocked", "configuration_capacity", ["provider_credit_insufficient"], None),
    ("awaiting_execution", "paused", ["scene_intent_paused"], None),
    # (b) a Blueprint fault a code fix can clear: setup_repair, not terminal.
    ("blocked", "preparation", ["scene_progression_factory_receipt_invalid"], "pipeline_fault"),
    ("blocked", "preparation", ["scene_progression_dependency_unavailable"], "pipeline_fault"),
    ("blocked", "source_preparation", ["preparation_failed"], "pipeline_fault"),
    ("blocked", "scene_publication", ["website_publication_profile_changed"], "pipeline_fault"),
    ("needs_input", "factory", ["fixture_unknown_input"], "pipeline_fault"),
    ("blocked", "preparation", [], "pipeline_fault"),
    # (c) money or approval: awaiting_spend_approval, not terminal.
    ("blocked", "preparation", ["scene_intake_spend_cap_exhausted"], "spend_cap_reached"),
    ("blocked", "preparation", ["scene_intake_attempt_spend_exceeds_authorized_limit"], "spend_cap_reached"),
    ("blocked", "preparation", ["scene_intake_attempt_cap_exhausted"], "attempt_cap_reached"),
    ("blocked", "preparation", ["beta_screening_job_ceiling_exceeded"], "job_ceiling_reached"),
    ("blocked", "preparation", ["website_job_ceiling_insufficient"], "job_ceiling_reached"),
    ("blocked", "preparation", ["website_scene_preparation_budget_exhausted"], "job_ceiling_reached"),
    # (d) waiting on the site: awaiting_site, not terminal.
    ("needs_input", "source", ["website_atlas_source_consent_required"], "site_consent_required"),
    ("needs_input", "source", ["website_atlas_planner_consent_required"], "site_consent_required"),
    ("needs_input", "source", ["website_atlas_consent_expired"], "site_consent_required"),
    ("needs_input", "source", ["task_brief_missing"], "task_brief_required"),
    # (e) privacy, rights, consent: withdrawn, terminal.
    ("blocked", "authority", ["scene_intake_authority_revoked"], "authority_revoked"),
    ("revoked", None, ["scene_intake_authority_revoked"], "authority_revoked"),
    ("blocked", "preparation", ["scene_configuration_submission_scene_owner_authority_revoked"],
     "authority_revoked"),
    ("blocked", "preparation", ["scene_execution_owner_revoked"], "authority_revoked"),
    ("blocked", "preparation", ["consent_expired"], "consent_expired"),
    ("blocked", "preparation", ["consent_revoked"], "consent_withdrawn"),
    ("blocked", "preparation", ["website_capture_withdrawn"], "consent_withdrawn"),
    ("blocked", "preparation", ["source_revoked"], "source_revoked"),
    ("blocked", "preparation", ["capture_revoked_or_expired"], "source_revoked"),
    # (f) capability limits: held, terminal.
    ("needs_input", "source", ["source_kind_not_supported_by_progression"], "source_kind_unsupported"),
    ("needs_input", "source", ["completed_3d_scene_result_required"], "source_kind_unsupported"),
    ("needs_input", "source", ["metric_workcell_capture_and_object_geometry_required"],
     "source_kind_unsupported"),
    ("needs_input", "policy_capability", ["scene_policy_checkpoint_capability_unavailable"],
     "policy_capability_unavailable"),
    ("blocked", "preparation", ["website_task_family_unknown"], "task_family_unsupported"),
    # (g) authority or time expiry is Blueprint's sponsorship: setup_repair, terminal.
    ("blocked", "authority", ["scene_intake_authority_expired"], "authority_expired"),
    ("expired", None, ["scene_intake_authority_expired"], "authority_expired"),
    ("blocked", "preparation", ["scene_configuration_submission_scene_owner_authority_expired"],
     "authority_expired"),
    # Sealed Blueprint failures: setup_repair, terminal.
    ("blocked", "scene_configuration_failed", ["configuration_launch_failed"], "scene_configuration_failed"),
    ("blocked", "policy_canary_blocked", ["provider_capacity_unavailable"], "policy_canary_blocked"),
    ("blocked", "policy_canary_preprovider_blocked", ["policy_canary_website_notification_pending"],
     "policy_canary_blocked"),
    *[("blocked", "preparation", [code], reason) for code, reason in SEALED_CAPS],
    # completed never has an outcome, whatever else the row carries.
    ("completed", "policy_canary_complete", [], None),
    ("completed", "scene_prepared", ["scene_intake_authority_revoked"], None),
    # Codes are read the way the status endpoint exposes them: up to the first ':'.
    ("blocked", "preparation", ["scene_intake_spend_cap_exhausted:attempt-7"], "spend_cap_reached"),
    ("blocked", "scene_configuration_failed",
     ["scene_configuration_provider_output_disk_budget_exceeded",
      "preallocation_capacity_recovery_withheld:measured"], "scene_configuration_failed"),
])
def test_classification_table(status, phase, blockers, reason):
    # The give-up clock is far below 24 hours here, so only the registry decides.
    assert outcome.decide_outcome(status, phase, blockers, T0, T0 + 60) == _expected(reason)


def test_a_privacy_or_rights_ending_outranks_every_other_reason():
    blockers = ["scene_intake_spend_cap_exhausted", "scene_progression_retry_cap_exhausted",
                "scene_intake_authority_revoked", "fixture_unknown"]
    assert outcome.decide_outcome("blocked", "preparation", blockers, None, T0) == _expected("authority_revoked")
    capability = ["scene_intake_spend_cap_exhausted", "source_kind_not_supported_by_progression"]
    assert outcome.decide_outcome("needs_input", "source", capability, None, T0) == _expected(
        "source_kind_unsupported")
    sealed = ["scene_intake_spend_cap_exhausted", "scene_progression_retry_cap_exhausted"]
    assert outcome.decide_outcome("blocked", "preparation", sealed, None, T0) == _expected("retry_cap_exhausted")


# --------------------------------------------------------------------- the 24-hour give-up rule

@pytest.mark.parametrize("status,phase,blockers", [
    ("blocked", "preparation", ["fixture_unknown_fault"]),
    ("running", "terminal_reconciliation", ["terminal_website_readback_pending"]),
    ("awaiting_execution", "selected_team_evaluation", []),
    ("blocked", "configuration_capacity", ["preallocation_capacity_not_recovered"]),
    ("needs_input", "factory", ["fixture_unknown_input"]),
    ("preparing", "factory", []),
])
def test_a_row_unchanged_for_24_hours_ends_as_setup_repair_stalled(status, phase, blockers):
    before = outcome.decide_outcome(status, phase, blockers, T0, T0 + DAY - 1)
    assert before is None or before["terminal"] is False
    assert outcome.decide_outcome(status, phase, blockers, T0, T0 + DAY) == _expected("stalled")
    assert _expected("stalled") == {"kind": "setup_repair", "terminal": True,
                                    "reason_category": "stalled", "reason_code": "stalled"}


@pytest.mark.parametrize("status,phase,blockers", [
    # A person's money or site action, and an operator's hold, wait for that
    # person; the execution window (authority_expired) bounds them instead.
    ("blocked", "preparation", ["scene_intake_spend_cap_exhausted"]),
    ("needs_input", "source", ["task_brief_missing"]),
    ("awaiting_execution", "paused", ["scene_intent_paused"]),
    ("awaiting_execution", "publication_ready", ["scene_submission_paused"]),
    ("awaiting_execution", "construction_prepared", []),
    # Already ended, or completed: the clock has nothing to add.
    ("needs_input", "source", ["source_kind_not_supported_by_progression"]),
    ("blocked", "authority", ["scene_intake_authority_revoked"]),
    ("completed", "policy_canary_complete", []),
])
def test_the_give_up_clock_never_overrides_an_intended_wait_or_an_ending(status, phase, blockers):
    assert outcome.decide_outcome(status, phase, blockers, T0, T0 + 30 * DAY) == outcome.decide_outcome(
        status, phase, blockers, None, T0 + 30 * DAY)
    assert outcome.decide_outcome(status, phase, blockers, None, T0) == outcome.decide_outcome(
        status, phase, blockers, T0, T0 + 30 * DAY)


def test_an_unknown_unchanged_since_never_gives_up():
    assert outcome.decide_outcome("blocked", "preparation", ["fixture_unknown"], None, T0 + 400 * DAY) == _expected(
        "pipeline_fault")


# --------------------------------------------------------------------- outcome.json

def test_a_terminal_outcome_is_written_once_and_never_rewritten(tmp_path):
    directory, intent = _intent(tmp_path)
    held = outcome.record_outcome(directory, intent, _expected("source_kind_unsupported"), now=T0 + 5)
    path = directory / outcome.FILENAME
    assert json.loads(path.read_text()) == held and held["terminal"] is True
    assert held["schema_version"] == outcome.SCHEMA_VERSION == "task_evaluation_scene_intent_outcome.v1"
    assert held["decided_at_epoch"] == T0 + 5
    assert held["outcome_digest"] == cross_runtime_canonical_digest(held, digest_field="outcome_digest")
    assert set(held) == {"schema_version", "intent_id", "intent_digest", "kind", "terminal", "reason_category",
                         "reason_code", "decided_at_epoch", "outcome_digest"}
    before = path.read_bytes()
    for later in (None, _expected("pipeline_fault"), _expected("stalled"), _expected("authority_revoked")):
        assert outcome.record_outcome(directory, intent, later, now=T0 + 99) == held
        assert path.read_bytes() == before
    assert outcome.terminal_outcome(directory, intent) == held


def test_a_non_terminal_outcome_follows_the_latest_decision(tmp_path):
    directory, intent = _intent(tmp_path)
    path = directory / outcome.FILENAME
    first = outcome.record_outcome(directory, intent, _expected("spend_cap_reached"), now=T0 + 1)
    assert first["kind"] == "awaiting_spend_approval" and first["terminal"] is False
    # The same decision keeps when it was first made.
    assert outcome.record_outcome(directory, intent, _expected("spend_cap_reached"), now=T0 + 2) == first
    replaced = outcome.record_outcome(directory, intent, _expected("pipeline_fault"), now=T0 + 3)
    assert replaced["reason_code"] == "pipeline_fault" and replaced["decided_at_epoch"] == T0 + 3
    assert outcome.terminal_outcome(directory, intent) is None
    # A row that no longer waits on anything has no outcome.
    assert outcome.record_outcome(directory, intent, None, now=T0 + 4) is None
    assert not path.exists() and outcome.read_outcome(directory, intent) is None
    outcome.record_outcome(directory, intent, _expected("attempt_cap_reached"), now=T0 + 5)
    ended = outcome.record_outcome(directory, intent, _expected("retry_cap_exhausted"), now=T0 + 6)
    assert outcome.terminal_outcome(directory, intent) == ended


def _resealed(value: dict, **change) -> dict:
    changed = {key: item for key, item in {**value, **change}.items() if key != "outcome_digest"}
    changed["outcome_digest"] = cross_runtime_canonical_digest(changed, digest_field="outcome_digest")
    return changed


def test_a_tampered_or_foreign_outcome_is_refused(tmp_path):
    directory, intent = _intent(tmp_path)
    other_directory, other = _intent(tmp_path, submission_id="upload-2")
    assert other["intent_id"] != intent["intent_id"]
    assert outcome.read_outcome(directory, intent) is None
    value = outcome.record_outcome(directory, intent, _expected("stalled"), now=T0)
    path = directory / outcome.FILENAME
    os.chmod(path, 0o640)
    # Sealed again after the change, so only the closed vocabulary can refuse it.
    for change in ({"terminal": False}, {"reason_code": "not_a_reason"}, {"kind": "held"},
                   {"reason_category": "blueprint_fault"}, {"decided_at_epoch": "yesterday"},
                   {"decided_at_epoch": True}, {"schema_version": "task_evaluation_scene_intent_outcome.v2"},
                   {"extra": 1}):
        path.write_text(json.dumps(_resealed(value, **change)))
        with pytest.raises(outcome.SceneOutcomeError):
            outcome.read_outcome(directory, intent)
    path.write_text(json.dumps({**value, "decided_at_epoch": T0 + 1}))  # not sealed again
    with pytest.raises(outcome.SceneOutcomeError):
        outcome.read_outcome(directory, intent)
    path.write_text(json.dumps(value))
    assert outcome.read_outcome(directory, intent) == value
    foreign = other_directory / outcome.FILENAME
    foreign.write_bytes(path.read_bytes())  # another intent's record
    with pytest.raises(outcome.SceneOutcomeError):
        outcome.read_outcome(other_directory, other)
    foreign.unlink()
    foreign.symlink_to(path)
    with pytest.raises(outcome.SceneOutcomeError):
        outcome.read_outcome(other_directory, other)
    foreign.unlink()
    os.mkfifo(foreign)  # never blocks a status read
    with pytest.raises(outcome.SceneOutcomeError):
        outcome.read_outcome(other_directory, other)
    path.write_text("{not json")
    with pytest.raises(outcome.SceneOutcomeError):
        outcome.read_outcome(directory, intent)


# --------------------------------------------------------------------- unchanged since

def test_unchanged_since_steps_over_mid_pass_notes_and_reads_past_the_alarm_window(tmp_path):
    directory, intent = _intent(tmp_path)
    progress = advance(directory, intent, None, status="awaiting_source", phase="source",
                       state={}, blockers=["website_prepared_source_pending"], now=T0)
    # One pass a minute: a mid-pass note, then the same failure again.
    for minute in range(1, 80):
        progress = advance(directory, intent, progress, status="preparing", phase="factory",
                           state={"tick": minute}, now=T0 + 100 + minute * 60)
        progress = advance(directory, intent, progress, status="blocked", phase="preparation",
                           state={"tick": minute}, blockers=["fixture_factory_failed"],
                           now=T0 + 100 + minute * 60 + 1)
    assert progress["event_sequence"] > 150  # beyond the stall alarm's own 64-event walk
    assert outcome.unchanged_since(directory, progress) == T0 + 160 + 1
    # A different resting state ends the run.
    progress = advance(directory, intent, progress, status="running", phase="source_preparation",
                       state={}, now=T0 + 9_000)
    progress = advance(directory, intent, progress, status="blocked", phase="preparation",
                       state={}, blockers=["fixture_factory_failed"], now=T0 + 9_060)
    assert outcome.unchanged_since(directory, progress) == T0 + 9_060


def test_the_stall_alarm_walk_is_unchanged_by_default(tmp_path):
    from blueprint_pipeline import control_plane_stall_alarm as stall
    directory, intent = _intent(tmp_path)
    progress = advance(directory, intent, None, status="blocked", phase="preparation", state={},
                       blockers=["fixture_factory_failed"], now=T0)
    progress = advance(directory, intent, progress, status="preparing", phase="factory", state={}, now=T0 + 60)
    progress = advance(directory, intent, progress, status="blocked", phase="preparation", state={},
                       blockers=["fixture_factory_failed"], now=T0 + 61)
    budget = stall._Budget(deadline=float("inf"), events=stall.MAX_EVENT_READS, clock=lambda: 0.0)
    assert stall._unchanged_since(directory, progress, budget=budget) == (T0 + 61, True)


def test_progress_markers_are_exactly_the_driver_s_mid_pass_notes():
    tree = ast.parse(inspect.getsource(engine._advance_intent))
    notes = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "id", None) == "emit"):
            arguments = [ast.literal_eval(argument) for argument in node.value.args]
            assert len(arguments) == 2 and arguments[0] == "preparing", arguments
            notes.add(arguments[1])
    assert notes == set(outcome.PROGRESS_MARKER_PHASES)
    assert all(outcome.is_progress_marker("preparing", phase, []) for phase in notes)
    assert not outcome.is_progress_marker("preparing", "factory", ["fixture"])
    assert not outcome.is_progress_marker("running", "source_preparation", [])


# --------------------------------------------------------------------- settle (the driver hook)

def test_settling_gives_up_on_a_failure_retried_every_pass_for_24_hours(tmp_path):
    directory, intent = _intent(tmp_path)
    progress = None
    for tick, now in enumerate((T0, T0 + 3_600, T0 + DAY)):
        # Each pass notes its factory step, then fails the same way.
        progress = advance(directory, intent, progress, status="preparing", phase="factory",
                           state={"tick": tick}, now=now)
        outcome.settle_outcome(directory, intent, progress, now=now)
        if tick:  # a mid-pass note never clears the pending decision
            assert outcome.read_outcome(directory, intent)["reason_code"] == "pipeline_fault"
        progress = advance(directory, intent, progress, status="blocked", phase="preparation",
                           state={"tick": tick}, blockers=["fixture_factory_failed"], now=now)
        settled = outcome.settle_outcome(directory, intent, progress, now=now)
        if now < T0 + DAY:
            assert settled["reason_code"] == "pipeline_fault" and settled["decided_at_epoch"] == T0
    assert settled == outcome.terminal_outcome(directory, intent)
    assert (settled["kind"], settled["reason_code"], settled["decided_at_epoch"]) == ("setup_repair", "stalled",
                                                                                     T0 + DAY)
    # The next pass's note leaves the ending alone.
    progress = advance(directory, intent, progress, status="preparing", phase="factory", state={"tick": 9},
                       now=T0 + DAY + 60)
    assert outcome.settle_outcome(directory, intent, progress, now=T0 + DAY + 60) == settled


def test_settling_clears_a_wait_once_the_row_moves_on_and_never_marks_completed(tmp_path):
    directory, intent = _intent(tmp_path)
    progress = advance(directory, intent, None, status="blocked", phase="preparation", state={},
                       blockers=["scene_intake_spend_cap_exhausted"], now=T0)
    assert outcome.settle_outcome(directory, intent, progress, now=T0)["kind"] == "awaiting_spend_approval"
    # Two days later the owner's extension lands and the row moves on.
    progress = advance(directory, intent, progress, status="running", phase="source_preparation", state={},
                       now=T0 + 2 * DAY)
    assert outcome.settle_outcome(directory, intent, progress, now=T0 + 2 * DAY) is None
    assert not (directory / outcome.FILENAME).exists()
    reference = {"uri": "s3://example/result.tar", "digest": "sha256:" + "f" * 64, "size_bytes": 1}
    progress = advance(directory, intent, progress, status="completed", phase="terminal", state={},
                       result_reference=reference, now=T0 + 400 * DAY)
    assert outcome.settle_outcome(directory, intent, progress, now=T0 + 400 * DAY) is None
    assert not (directory / outcome.FILENAME).exists()


# --------------------------------------------------------------------- the driver

def _advance(directory, intent, config=None, *, now, resolver=None, release=None):
    return engine._advance_intent(directory, intent, config or {}, release or {"source_commit": "d" * 40},
                                  resolver=resolver, publisher=None, submitter=None, status_reader=None,
                                  activation_provisioner=None, now=now)


def _awaiting_source(**_kwargs):
    return engine.SourceResolution("awaiting_source", blockers=("website_prepared_source_pending",))


def _no_new_execution(**_kwargs):
    raise AssertionError("a terminal outcome must stop new execution")


def test_driver_writes_the_outcome_after_each_emit_and_stops_after_a_terminal_one(tmp_path):
    directory, intent = _intent(tmp_path)
    first = _advance(directory, intent, {"supported_source_kinds": ["capture_bundle"]}, now=T0 + 10)
    assert (first["status"], first["blockers"]) == ("needs_input", ["source_kind_not_supported_by_progression"])
    held = outcome.terminal_outcome(directory, intent)
    assert (held["kind"], held["reason_code"], held["decided_at_epoch"]) == ("held", "source_kind_unsupported",
                                                                            T0 + 10)
    # The next pass does no new work: no resolver call, no new event.
    again = _advance(directory, intent, {}, now=T0 + 70, resolver=_no_new_execution)
    assert again == first == load_progression(directory, intent)


def _source_refused(**_kwargs):
    return engine.SourceResolution("blocked", blockers=("fixture_source_refused",))


@pytest.mark.parametrize("resolver,status,pending", [
    (_awaiting_source, "awaiting_source", None),
    (_source_refused, "blocked", "pipeline_fault"),
])
def test_driver_gives_up_on_a_row_unchanged_for_24_hours(tmp_path, resolver, status, pending):
    directory, intent = _intent(tmp_path)
    waiting = _advance(directory, intent, now=T0 + 10, resolver=resolver)
    assert waiting["status"] == status

    def pending_reason():
        retained = outcome.read_outcome(directory, intent)
        assert retained is None or retained["terminal"] is False
        return retained and retained["reason_code"]

    assert pending_reason() == pending
    _advance(directory, intent, now=T0 + 10 + DAY - 1, resolver=resolver)
    assert pending_reason() == pending
    stalled = _advance(directory, intent, now=T0 + 10 + DAY, resolver=resolver)
    assert stalled == waiting  # the same row, now ended
    ended = outcome.terminal_outcome(directory, intent)
    assert (ended["kind"], ended["reason_code"], ended["terminal"]) == ("setup_repair", "stalled", True)
    # Even past the execution window, nothing new runs and no authority event is added.
    later = _advance(directory, intent, now=T0 + 30 * DAY, resolver=_no_new_execution)
    assert later == waiting and outcome.terminal_outcome(directory, intent) == ended


def test_terminal_outcome_keeps_the_read_only_join_that_closes_an_authorized_run(tmp_path):
    from tests.test_task_evaluation_scene_terminal_reconciler import _env, _receipts
    env = _env(tmp_path)
    projection = _receipts(env)
    advance(env["directory"], env["intent"], None, status="awaiting_execution", phase="scene_configuration",
            state={"activation": {"provider_allocation_performed": False}, "attempt_id": "source-1"},
            now=env["now"])
    stalled = outcome.record_outcome(env["directory"], env["intent"], _expected("stalled"), now=env["now"])
    config = dict(env["config"], factory_output_root=str(tmp_path / "factory-output"))
    progress = _advance(env["directory"], env["intent"], config, now=env["now"] + 60,
                        resolver=_no_new_execution, release=env["release"])
    assert progress["status"] == "completed", progress
    # Written once: the outcome stays, and the completed status supersedes it.
    assert outcome.terminal_outcome(env["directory"], env["intent"]) == stalled
    status = intake.scene_intent_status(queue_root=env["config"]["intent_root"], intent_id=env["intent_id"],
                                        now=env["now"] + 60)
    assert status["status"] == "completed" and "outcome" not in status
    assert status["result_reference"]["digest"] == projection["projection_digest"]


# --------------------------------------------------------------------- the status endpoint

def test_status_omits_outcome_until_one_exists_then_reports_it_sealed(tmp_path):
    directory, intent = _intent(tmp_path)
    plain = intake.scene_intent_status(queue_root=tmp_path, intent_id=intent["intent_id"], now=T0 + 1)
    assert "outcome" not in plain
    progress = advance(directory, intent, None, status="blocked", phase="preparation", state={},
                       blockers=["scene_intake_spend_cap_exhausted"], now=T0 + 2)
    waiting = outcome.settle_outcome(directory, intent, progress, now=T0 + 2)
    status = intake.scene_intent_status(queue_root=tmp_path, intent_id=intent["intent_id"], now=T0 + 3)
    assert status["outcome"] == waiting
    assert status["outcome"]["outcome_digest"] == cross_runtime_canonical_digest(
        status["outcome"], digest_field="outcome_digest")
    assert status["status_digest"] == cross_runtime_canonical_digest(status, digest_field="status_digest")
    assert status == intake.scene_intent_status(queue_root=tmp_path, intent_id=intent["intent_id"], now=T0 + 4)


@pytest.mark.parametrize("authority", ["revocation", "expiry"])
def test_status_reports_the_ending_revocation_or_expiry_already_decided(tmp_path, authority):
    directory, intent = _intent(tmp_path)
    advance(directory, intent, None, status="blocked", phase="preparation", state={},
            blockers=["scene_intake_spend_cap_exhausted"], now=T0 + 2)
    outcome.settle_outcome(directory, intent, load_progression(directory, intent), now=T0 + 2)
    if authority == "revocation":
        intake.revoke_scene_intent(queue_root=tmp_path, intent_id=intent["intent_id"],
                                   intent_digest=intent["intent_digest"], owner=intent["request"]["owner"],
                                   now=T0 + 50)
        read_at, expected = T0 + 60, ("withdrawn", "authority_revoked", T0 + 50)
    else:
        read_at, expected = T0 + 4 * DAY, ("setup_repair", "authority_expired", T0 + 3 * DAY)
    status = intake.scene_intent_status(queue_root=tmp_path, intent_id=intent["intent_id"], now=read_at)
    assert status["status"] == ("revoked" if authority == "revocation" else "expired")
    reported = status["outcome"]
    assert (reported["kind"], reported["reason_code"], reported["decided_at_epoch"]) == expected
    assert reported["terminal"] is True
    assert status["status_digest"] == cross_runtime_canonical_digest(status, digest_field="status_digest")
    # A status read never writes; the driver's next pass writes the same record.
    assert outcome.terminal_outcome(directory, intent) is None
    progress = _advance(directory, intent, now=read_at + 30, resolver=_no_new_execution)
    assert progress["phase"] == "authority"
    assert outcome.terminal_outcome(directory, intent) == reported


def test_status_keeps_the_first_ending_when_the_authority_ends_later(tmp_path):
    directory, intent = _intent(tmp_path)
    advance(directory, intent, None, status="awaiting_source", phase="source", state={},
            blockers=["website_prepared_source_pending"], now=T0)
    stalled = outcome.settle_outcome(directory, intent, load_progression(directory, intent), now=T0 + DAY)
    assert stalled["reason_code"] == "stalled"
    intake.revoke_scene_intent(queue_root=tmp_path, intent_id=intent["intent_id"],
                               intent_digest=intent["intent_digest"], owner=intent["request"]["owner"],
                               now=T0 + DAY + 10)
    status = intake.scene_intent_status(queue_root=tmp_path, intent_id=intent["intent_id"], now=T0 + DAY + 20)
    assert status["status"] == "revoked" and status["outcome"] == stalled


def test_status_refuses_a_corrupt_outcome(tmp_path):
    directory, intent = _intent(tmp_path)
    advance(directory, intent, None, status="blocked", phase="preparation", state={},
            blockers=["fixture_unknown"], now=T0)
    value = outcome.settle_outcome(directory, intent, load_progression(directory, intent), now=T0)
    path = directory / outcome.FILENAME
    os.chmod(path, 0o640)
    path.write_text(json.dumps({**value, "reason_code": "stalled"}))
    with pytest.raises(intake.SceneIntakeError, match="scene_intake_outcome_invalid"):
        intake.scene_intent_status(queue_root=tmp_path, intent_id=intent["intent_id"], now=T0 + 1)
