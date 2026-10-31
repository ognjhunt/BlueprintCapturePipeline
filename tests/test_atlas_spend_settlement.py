"""World Labs Atlas holds settle from each closed attempt's retained outcome."""
from __future__ import annotations

import json

import pytest

from blueprint_pipeline import atlas_spend_settlement as settlement
from blueprint_pipeline import website_task_context as control
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

BINDING = "sha256:" + "1" * 64
RESERVED_CONTEXT = "sha256:" + "2" * 64
CONTEXT = {"request_id": "request-atlas", "scene_id": "site-request-atlas", "capture_id": "website-capture",
           "context_digest": "sha256:" + "3" * 64}
FAILED_OPERATION = {"id": "op_splat_1", "done": True, "error": {"code": 13, "message": "synthetic", "details": []}}


def _admission(**overrides):
    return {"allocation_binding_digest": BINDING, "provider": "world_labs_atlas",
            "resource_class": "provider_reconstruction_api", "task_context_digest": RESERVED_CONTEXT,
            "maximum_cost_usd": 5, "status": "admitted", **overrides}


def _attempt(root, state, admission=None, *, task=True):
    """A retained attempt: a task keeps its admission beside atlas_pose/, a pose inside it."""
    (root / "atlas_pose").mkdir(parents=True)
    (root / "atlas_pose" / "submission.json").write_text(json.dumps({"request_digest": BINDING, **state}))
    path = root / "controller_admission.json" if task else root / "atlas_pose" / "controller_admission.json"
    path.write_text(json.dumps(admission or _admission()))


class WebApp:
    def __init__(self, refuse=None):
        self.settled, self.refuse = [], refuse

    def __call__(self, *, task_context, operation, payload):
        assert operation == "preparation-settlement" and task_context is CONTEXT
        command = payload["settlement"]
        assert payload["request_id"] == CONTEXT["request_id"] and payload["scene_id"] == CONTEXT["scene_id"]
        if self.refuse:
            raise ValueError(self.refuse)
        self.settled.append(command)
        return {**command, "status": "settled", "actual_cost_usd": 5 if command["outcome"] == "completed" else 0}


@pytest.fixture
def webapp(monkeypatch):
    fake = WebApp()
    monkeypatch.setattr(control, "preparation_webapp_request", fake)
    return fake


@pytest.mark.parametrize("state,closed,expected", [
    ({"status": "completed", "operation_id": "op_1", "result_digest": "sha256:" + "9" * 64}, False,
     ("completed", "op_1")),
    ({"status": "failed", "operation_id": "op_splat_1", "terminal_operation": FAILED_OPERATION}, False,
     ("failed", "op_splat_1")),
    # A task whose own operation failed keeps that failure after its retry completes.
    ({"status": "completed", "operation_id": "op_splat_1", "terminal_operation": FAILED_OPERATION,
      "result_digest": "sha256:" + "9" * 64, "completed_operation_id": "op_splat_2"}, True, ("failed", "op_splat_1")),
    ({"status": "rejected", "provider_rejection": {"status": 400, "retryable": False}, "possibly_bought": False}, False,
     ("refused", None)),
    # A credit refusal is resent under the same hold until the attempt is set aside.
    ({"status": "rejected", "provider_rejection": {"status": 402, "retryable": True}, "resends": 0}, False, None),
    ({"status": "rejected", "provider_rejection": {"status": 402, "retryable": True}, "resends": 0}, True,
     ("refused", None)),
    # Unknown outcomes keep their full hold.
    ({"status": "rejected", "provider_rejection": {"status": 422, "retryable": False}, "possibly_bought": True}, True, None),
    ({"status": "submitting", "possibly_bought": True}, True, None),
    ({"status": "submitted", "operation_id": "op_1"}, True, None),
    ({"status": "failed", "operation_id": "op_1", "terminal_error": {"status": 404}}, True, None),
    ({"status": "failed", "terminal_error": {"reason": "reconciliation_window_lapsed"}}, True, None),
    ({"status": "failed", "operation_id": "op_2", "terminal_operation": FAILED_OPERATION}, True, None),
    ({"status": "uploading"}, True, None),
])
def test_only_a_known_outcome_settles(state, closed, expected):
    outcome = settlement.atlas_attempt_outcome(state, closed=closed)
    assert (outcome and (outcome["outcome"], outcome["operation_id"])) == expected


def test_a_retained_failed_splat_releases_its_hold_once_under_its_reservation(tmp_path, webapp):
    # capture-447ae4f4: a failed splat World Labs never charged still held its price.
    _attempt(tmp_path, {"status": "failed", "operation_id": "op_splat_1", "terminal_operation": FAILED_OPERATION})
    [receipt] = settlement.settle_closed_atlas_task_attempts(tmp_path, task_context=CONTEXT)
    assert receipt["actual_cost_usd"] == 0
    assert webapp.settled == [{"task_context_digest": RESERVED_CONTEXT, "allocation_binding_digest": BINDING,
                               "provider": "world_labs_atlas", "outcome": "failed", "operation_id": "op_splat_1",
                               "outcome_receipt_digest": canonical_digest(FAILED_OPERATION)}]
    assert json.loads((tmp_path / "atlas_pose" / settlement.SETTLEMENT_FILENAME).read_text()) == receipt
    # Retained: a later pass reports nothing again.
    assert settlement.settle_closed_atlas_task_attempts(tmp_path, task_context=CONTEXT) == [receipt]
    assert len(webapp.settled) == 1


def test_a_task_retry_chain_settles_each_attempt_under_its_own_reservation(tmp_path, webapp):
    _attempt(tmp_path, {"status": "failed", "operation_id": "op_splat_1", "terminal_operation": FAILED_OPERATION})
    retry_binding = "sha256:" + "4" * 64
    (tmp_path / "retry" / "atlas_pose").mkdir(parents=True)
    (tmp_path / "retry" / "atlas_pose" / "submission.json").write_text(json.dumps({
        "request_digest": retry_binding, "status": "completed", "operation_id": "op_splat_2",
        "result_digest": "sha256:" + "8" * 64}))
    (tmp_path / "retry" / "controller_admission.json").write_text(json.dumps(
        _admission(allocation_binding_digest=retry_binding)))
    settlement.settle_closed_atlas_task_attempts(tmp_path, task_context=CONTEXT)
    assert [(row["allocation_binding_digest"], row["outcome"]) for row in webapp.settled] == [
        (BINDING, "failed"), (retry_binding, "completed")]


def test_archived_pose_attempts_settle_beside_the_current_one(tmp_path, webapp):
    archived = tmp_path / "atlas_pose_attempts" / "000"
    archived.mkdir(parents=True)
    (archived / "submission.json").write_text(json.dumps({"request_digest": BINDING, "status": "rejected",
        "provider_rejection": {"status": 402, "retryable": True}, "possibly_bought": False}))
    (archived / "controller_admission.json").write_text(json.dumps(_admission()))
    current = "sha256:" + "5" * 64
    _attempt(tmp_path, {"request_digest": current, "status": "submitted", "operation_id": "op_2"},
             _admission(allocation_binding_digest=current), task=False)
    settlement.settle_closed_atlas_pose_attempts(tmp_path, task_context=CONTEXT)
    assert [(row["allocation_binding_digest"], row["outcome"]) for row in webapp.settled] == [(BINDING, "refused")]


@pytest.mark.parametrize("admission", [
    _admission(provider="openai", resource_class="openai_api_candidate"),
    _admission(allocation_binding_digest="sha256:" + "6" * 64),
    _admission(task_context_digest=None),
])
def test_an_admission_for_another_hold_is_never_settled(tmp_path, webapp, admission):
    _attempt(tmp_path, {"status": "failed", "operation_id": "op_splat_1", "terminal_operation": FAILED_OPERATION},
             admission)
    assert settlement.settle_closed_atlas_task_attempts(tmp_path, task_context=CONTEXT) == []
    assert webapp.settled == []


@pytest.mark.parametrize("failure", [
    ValueError("website_control_preparation-settlement_http_400:website_scene_request_invalid"),
    OSError("unreachable"), KeyError("anything")])
def test_a_refused_or_unreachable_settlement_keeps_the_hold_and_never_raises(tmp_path, monkeypatch, failure):
    def refuse(**_kwargs):
        raise failure
    monkeypatch.setattr(control, "preparation_webapp_request", refuse)
    _attempt(tmp_path, {"status": "failed", "operation_id": "op_splat_1", "terminal_operation": FAILED_OPERATION})
    assert settlement.settle_closed_atlas_task_attempts(tmp_path, task_context=CONTEXT) == []
    assert not (tmp_path / "atlas_pose" / settlement.SETTLEMENT_FILENAME).exists()
    # The next pass, after the WebApp accepts Atlas settlements, reports it.
    fake = WebApp()
    monkeypatch.setattr(control, "preparation_webapp_request", fake)
    assert len(settlement.settle_closed_atlas_task_attempts(tmp_path, task_context=CONTEXT)) == 1


def test_a_receipt_that_does_not_echo_the_outcome_is_not_retained(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "preparation_webapp_request",
                        lambda **kw: {**kw["payload"]["settlement"], "outcome": "completed", "status": "settled"})
    _attempt(tmp_path, {"status": "failed", "operation_id": "op_splat_1", "terminal_operation": FAILED_OPERATION})
    assert settlement.settle_closed_atlas_task_attempts(tmp_path, task_context=CONTEXT) == []
    assert not (tmp_path / "atlas_pose" / settlement.SETTLEMENT_FILENAME).exists()


def test_a_job_pass_settles_every_retained_attempt_before_anything_new_is_admitted(tmp_path, webapp):
    def binding(character):
        return "sha256:" + character * 64
    # A failed refined splat (task), an archived refused pose and a completed current pose.
    _attempt(tmp_path / "refined_splat", {"request_digest": binding("a"), "status": "failed",
             "operation_id": "op_splat_1", "terminal_operation": FAILED_OPERATION},
             _admission(allocation_binding_digest=binding("a")))
    archived = tmp_path / "pose" / "atlas_pose_attempts" / "000"
    archived.mkdir(parents=True)
    (archived / "submission.json").write_text(json.dumps({"request_digest": binding("b"), "status": "rejected",
        "provider_rejection": {"status": 400, "retryable": False}, "possibly_bought": False}))
    (archived / "controller_admission.json").write_text(json.dumps(_admission(allocation_binding_digest=binding("b"))))
    _attempt(tmp_path / "pose", {"request_digest": binding("c"), "status": "completed", "operation_id": "op_pose",
             "result_digest": binding("9")}, _admission(allocation_binding_digest=binding("c")), task=False)
    # Another provider's state with the same file name is never touched.
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "submission.json").write_text(json.dumps({"status": "completed"}))
    settled = settlement.settle_closed_atlas_attempts_under(tmp_path, task_context=CONTEXT)
    assert sorted((row["allocation_binding_digest"][7], row["outcome"]) for row in settled) == [
        ("a", "failed"), ("b", "refused"), ("c", "completed")]
    # A later pass returns the retained receipts and reports nothing again.
    assert settlement.settle_closed_atlas_attempts_under(tmp_path, task_context=CONTEXT) == settled
    assert len(webapp.settled) == 3
