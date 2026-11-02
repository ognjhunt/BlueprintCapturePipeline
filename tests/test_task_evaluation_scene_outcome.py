"""C-04: every website job ends in exactly one terminal state with a plain reason.

Before this, only ``completed`` stopped the scene progression driver; every
other row was re-run every minute forever and the website could only guess
whether a job had ended. These tests pin the one registry, the write-once
``outcome.json``, the execution window as the only give-up timer (a long wait
gets no terminal outcome; expiry ends it), Blueprint faults staying
re-drivable, the driver stopping new execution once an outcome is terminal
(while its read-only joins keep closing out authorized runs), and the optional
``outcome`` on the intent status.
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
    assert set(outcome.BLOCKER_REASONS.values()) <= set(outcome.REASONS)
    assert set(outcome.PHASE_REASONS.values()) <= set(outcome.REASONS)
    assert set(outcome.STATUS_REASONS.values()) <= set(outcome.REASONS)
    # Every category the WebApp may map has at least one reason.
    assert {row[2] for row in outcome.REASONS.values()} == set(outcome.REASON_CATEGORIES)


def test_only_privacy_rights_capability_spent_allowances_and_expiry_end_a_job():
    terminal = {code for code, (_kind, ended, _category) in outcome.REASONS.items() if ended}
    assert terminal == {
        # withdrawn: privacy, rights and consent
        "authority_revoked", "consent_expired", "consent_withdrawn", "source_revoked",
        # held: capability limits
        "source_kind_unsupported", "policy_capability_unavailable", "task_family_unsupported",
        "task_infeasible", "no_candidate",
        # non-extendable allowances that ran out
        "retry_cap_exhausted", "submission_retry_cap_exhausted", "capacity_recovery_cap_exhausted",
        "market_miss_recovery_cap_exhausted",
        # the execution window, the only give-up timer
        "authority_expired",
    }
    for kind in ("withdrawn", "held"):
        assert all(ended for (row_kind, ended, _category) in outcome.REASONS.values() if row_kind == kind)
    for kind in ("awaiting_spend_approval", "awaiting_site"):
        assert not any(ended for (row_kind, ended, _category) in outcome.REASONS.values() if row_kind == kind)
    # A Blueprint fault stays fixable: a code fix and deploy can still re-drive it.
    for reason in ("pipeline_fault", "scene_configuration_failed", "policy_canary_blocked"):
        assert outcome.REASONS[reason] == ("setup_repair", False, "blueprint_fault")


SPENT_ALLOWANCES = [
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
    ("awaiting_execution", "selected_team_evaluation", [], None),
    ("running", "terminal_reconciliation", ["terminal_resource_closure_pending",
                                            "terminal_website_readback_pending"], None),
    ("blocked", "configuration_capacity", ["preallocation_capacity_not_recovered"], None),
    ("blocked", "configuration_capacity", ["provider_credit_insufficient"], None),
    ("awaiting_execution", "paused", ["scene_intent_paused"], None),
    # (b) Blueprint faults a code fix and deploy can re-drive: setup_repair, not terminal.
    ("blocked", "preparation", ["scene_progression_factory_receipt_invalid"], "pipeline_fault"),
    ("blocked", "preparation", ["scene_progression_dependency_unavailable"], "pipeline_fault"),
    ("blocked", "source_preparation", ["preparation_failed"], "pipeline_fault"),
    ("blocked", "scene_publication", ["website_publication_profile_changed"], "pipeline_fault"),
    ("needs_input", "factory", ["fixture_unknown_input"], "pipeline_fault"),
    ("blocked", "preparation", [], "pipeline_fault"),
    ("blocked", "scene_configuration_failed", ["configuration_launch_failed"], "scene_configuration_failed"),
    ("blocked", "policy_canary_blocked", ["provider_capacity_unavailable"], "policy_canary_blocked"),
    ("blocked", "policy_canary_preprovider_blocked", ["policy_canary_website_notification_pending"],
     "policy_canary_blocked"),
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
    # Spent non-extendable allowances: setup_repair, terminal.
    *[("blocked", "preparation", [code], reason) for code, reason in SPENT_ALLOWANCES],
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
    assert outcome.decide_outcome(status, phase, blockers) == _expected(reason)


def test_a_privacy_or_rights_ending_outranks_every_other_reason():
    blockers = ["scene_intake_spend_cap_exhausted", "scene_progression_retry_cap_exhausted",
                "scene_intake_authority_revoked", "fixture_unknown"]
    assert outcome.decide_outcome("blocked", "preparation", blockers) == _expected("authority_revoked")
    capability = ["scene_intake_spend_cap_exhausted", "source_kind_not_supported_by_progression"]
    assert outcome.decide_outcome("needs_input", "source", capability) == _expected("source_kind_unsupported")
    spent = ["scene_intake_spend_cap_exhausted", "scene_progression_retry_cap_exhausted"]
    assert outcome.decide_outcome("blocked", "preparation", spent) == _expected("retry_cap_exhausted")
    # Money waits for a person; a fault behind it waits for that person first.
    money = ["scene_intake_spend_cap_exhausted"]
    assert outcome.decide_outcome("blocked", "scene_configuration_failed", money) == _expected(
        "spend_cap_reached")


# --------------------------------------------------------------------- the execution window

@pytest.mark.parametrize("status,phase,blockers", [
    ("awaiting_source", "source", ["website_prepared_source_pending"]),
    ("running", "terminal_reconciliation", ["terminal_website_readback_pending"]),
    ("blocked", "configuration_capacity", ["preallocation_capacity_not_recovered"]),
    ("blocked", "preparation", ["fixture_unknown_fault"]),
    ("blocked", "scene_configuration_failed", ["configuration_launch_failed"]),
    ("blocked", "policy_canary_blocked", ["provider_capacity_unavailable"]),
    ("blocked", "preparation", ["scene_intake_spend_cap_exhausted"]),
    ("needs_input", "source", ["task_brief_missing"]),
    ("awaiting_execution", "paused", ["scene_intent_paused"]),
])
@pytest.mark.parametrize("ended,reason", [("expired", "authority_expired"), ("revoked", "authority_revoked")])
def test_the_execution_window_ends_whatever_the_registry_leaves_open(status, phase, blockers, ended, reason):
    before = outcome.decide_outcome(status, phase, blockers)
    assert before is None or before["terminal"] is False
    assert outcome.decide_outcome(status, phase, blockers, ended) == _expected(reason)


@pytest.mark.parametrize("status,phase,blockers", [
    ("needs_input", "source", ["source_kind_not_supported_by_progression"]),
    ("blocked", "preparation", ["scene_progression_retry_cap_exhausted"]),
    ("blocked", "preparation", ["consent_expired"]),
    ("completed", "policy_canary_complete", []),
])
def test_an_ending_of_its_own_or_completion_outlasts_the_window(status, phase, blockers):
    assert outcome.decide_outcome(status, phase, blockers, "expired") == outcome.decide_outcome(
        status, phase, blockers)


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
    for later in (None, _expected("pipeline_fault"), _expected("retry_cap_exhausted"),
                  _expected("authority_revoked")):
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
    replaced = outcome.record_outcome(directory, intent, _expected("scene_configuration_failed"), now=T0 + 3)
    assert replaced["reason_code"] == "scene_configuration_failed" and replaced["decided_at_epoch"] == T0 + 3
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
    value = outcome.record_outcome(directory, intent, _expected("source_kind_unsupported"), now=T0)
    path = directory / outcome.FILENAME
    os.chmod(path, 0o640)
    # Sealed again after the change, so only the closed vocabulary can refuse it.
    for change in ({"terminal": False}, {"reason_code": "not_a_reason"}, {"kind": "setup_repair"},
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

def test_a_failure_retried_every_pass_keeps_one_decision_through_mid_pass_notes(tmp_path):
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
        assert (settled["reason_code"], settled["terminal"], settled["decided_at_epoch"]) == (
            "pipeline_fault", False, T0)


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
                       result_reference=reference, now=T0 + 2 * DAY + 60)
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


def test_a_failed_build_is_setup_repair_and_the_next_pass_still_re_drives_it(tmp_path):
    directory, intent = _intent(tmp_path)
    progress = advance(directory, intent, None, status="blocked", phase="scene_configuration_failed",
                       state={}, blockers=["configuration_launch_failed"], now=T0)
    failed = outcome.settle_outcome(directory, intent, progress, now=T0)
    assert (failed["kind"], failed["terminal"], failed["reason_code"]) == (
        "setup_repair", False, "scene_configuration_failed")
    calls = []

    def fixed(**kwargs):
        calls.append(kwargs)
        return _awaiting_source()

    # A deploy with the fix: the driver runs again and the job moves on.
    progress = _advance(directory, intent, now=T0 + DAY, resolver=fixed)
    assert len(calls) == 1 and progress["status"] == "awaiting_source"
    assert outcome.read_outcome(directory, intent) is None


def test_a_long_wait_gets_no_terminal_outcome_until_the_window_expires(tmp_path):
    directory, intent = _intent(tmp_path, window=7 * DAY)
    for now in (T0 + 10, T0 + 3 * DAY, T0 + 7 * DAY - 1):
        waiting = _advance(directory, intent, now=now, resolver=_awaiting_source)
        assert waiting["status"] == "awaiting_source" and outcome.read_outcome(directory, intent) is None
    ended = _advance(directory, intent, now=T0 + 7 * DAY, resolver=_awaiting_source)
    assert (ended["status"], ended["phase"], ended["blockers"]) == (
        "blocked", "authority", ["scene_intake_authority_expired"])
    expired = outcome.terminal_outcome(directory, intent)
    assert (expired["kind"], expired["reason_code"], expired["decided_at_epoch"]) == (
        "setup_repair", "authority_expired", T0 + 7 * DAY)
    # Ended: nothing new runs on later passes.
    assert _advance(directory, intent, now=T0 + 8 * DAY, resolver=_no_new_execution) == ended


def test_the_window_ends_a_job_whose_join_answers_before_the_authority_gate(tmp_path):
    """A pending terminal join returns before the authority gate on every pass."""
    from tests.test_task_evaluation_scene_terminal_reconciler import _env, _receipts
    env = _env(tmp_path)
    _receipts(env, include_closure=False)
    advance(env["directory"], env["intent"], None, status="awaiting_execution", phase="scene_configuration",
            state={"activation": {"provider_allocation_performed": False}, "attempt_id": "source-1"},
            now=env["now"])
    config = dict(env["config"], factory_output_root=str(tmp_path / "factory-output"))

    def drive(now):
        return _advance(env["directory"], env["intent"], config, now=now, resolver=_no_new_execution,
                        release=env["release"])

    pending = drive(env["now"] + 60)
    assert (pending["status"], pending["phase"]) == ("running", "terminal_reconciliation")
    assert outcome.read_outcome(env["directory"], env["intent"]) is None
    expiry = env["intent"]["request"]["execution"]["expires_at_epoch"]
    assert drive(expiry + 60)["phase"] == "terminal_reconciliation"
    expired = outcome.terminal_outcome(env["directory"], env["intent"])
    assert (expired["reason_code"], expired["decided_at_epoch"]) == ("authority_expired", expiry)
    # The authorized run still closes out read-only, and completed supersedes the ending.
    projection = _receipts(env)
    assert drive(expiry + 120)["status"] == "completed"
    assert outcome.terminal_outcome(env["directory"], env["intent"]) == expired
    status = intake.scene_intent_status(queue_root=env["config"]["intent_root"], intent_id=env["intent_id"],
                                        now=expiry + 180)
    assert status["status"] == "completed" and "outcome" not in status
    assert status["result_reference"]["digest"] == projection["projection_digest"]


def test_a_blocked_policy_canary_is_setup_repair_until_the_window_ends_it(tmp_path):
    from tests.test_task_evaluation_scene_terminal_reconciler import _blocked_projection, _env, _receipts
    env = _env(tmp_path)
    _receipts(env, projection=_blocked_projection())
    advance(env["directory"], env["intent"], None, status="awaiting_execution", phase="scene_configuration",
            state={"activation": {"provider_allocation_performed": False}, "attempt_id": "source-1"},
            now=env["now"])
    config = dict(env["config"], factory_output_root=str(tmp_path / "factory-output"))
    blocked = _advance(env["directory"], env["intent"], config, now=env["now"] + 60, release=env["release"])
    assert (blocked["status"], blocked["phase"]) == ("blocked", "policy_canary_blocked")
    repair = outcome.read_outcome(env["directory"], env["intent"])
    assert (repair["kind"], repair["terminal"], repair["reason_code"]) == (
        "setup_repair", False, "policy_canary_blocked")
    expiry = env["intent"]["request"]["execution"]["expires_at_epoch"]
    _advance(env["directory"], env["intent"], config, now=expiry + 60, release=env["release"])
    expired = outcome.terminal_outcome(env["directory"], env["intent"])
    assert (expired["kind"], expired["reason_code"]) == ("setup_repair", "authority_expired")


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
    advance(directory, intent, None, status="blocked", phase="preparation", state={},
            blockers=["scene_progression_retry_cap_exhausted"], now=T0)
    spent = outcome.settle_outcome(directory, intent, load_progression(directory, intent), now=T0)
    assert spent["reason_code"] == "retry_cap_exhausted" and spent["terminal"] is True
    intake.revoke_scene_intent(queue_root=tmp_path, intent_id=intent["intent_id"],
                               intent_digest=intent["intent_digest"], owner=intent["request"]["owner"],
                               now=T0 + 10)
    status = intake.scene_intent_status(queue_root=tmp_path, intent_id=intent["intent_id"], now=T0 + 20)
    assert status["status"] == "revoked" and status["outcome"] == spent


def test_status_refuses_a_corrupt_outcome(tmp_path):
    directory, intent = _intent(tmp_path)
    advance(directory, intent, None, status="blocked", phase="preparation", state={},
            blockers=["fixture_unknown"], now=T0)
    value = outcome.settle_outcome(directory, intent, load_progression(directory, intent), now=T0)
    path = directory / outcome.FILENAME
    os.chmod(path, 0o640)
    path.write_text(json.dumps({**value, "reason_code": "scene_configuration_failed"}))
    with pytest.raises(intake.SceneIntakeError, match="scene_intake_outcome_invalid"):
        intake.scene_intent_status(queue_root=tmp_path, intent_id=intent["intent_id"], now=T0 + 1)
