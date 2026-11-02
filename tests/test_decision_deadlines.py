"""Grant, authority and window checks report the instant their answer changes.

A visual-review grant, an owner execution window and a preparation grant are
valid until an expiry. A decision made against the clock before then flips at
that instant with no file changing, so anything that caches the decision (the
runtime env guard's catalog-phase verdict) needs the instant. Review of #3278:
a lapsed visual-review grant blocks a full catalog pass, and a cache that did
not know its expiry served the earlier ready verdict for up to 15 minutes.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_retained_controls_evidence  # noqa: F401 - fixture chain
from blueprint_pipeline import task_evaluation_scene_execution_authority as execution
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline import task_evaluation_terminal_adoption_retirement as retirement
from blueprint_pipeline import task_evaluation_visual_review_continuation as visual
from blueprint_pipeline.decision_deadlines import collecting_deadlines, note_deadline
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_task_evaluation_terminal_adoption_retirement import adopted as adopted  # noqa: F401
from tests.test_task_evaluation_unstarted_controls_reservations import reserved as reserved  # noqa: F401
from tests.test_task_evaluation_visual_review_continuation import (  # noqa: F401
    failed_review as failed_review,
    grant,
)
from tests.test_website_preparation_contract_v2 import NOW as PREPARATION_NOW
from tests.test_website_preparation_contract_v2 import VECTOR


def test_only_open_collectors_receive_finite_instants() -> None:
    assert note_deadline(5.0) == 5.0  # nothing open: a no-op that returns its argument
    with collecting_deadlines() as outer:
        note_deadline(10)
        with collecting_deadlines() as inner:
            for ignored in (None, True, "11", float("inf"), float("nan")):
                assert note_deadline(ignored) is ignored
            note_deadline(12.5)
    assert outer == [10.0, 12.5]
    assert inner == [12.5]


def _review_with_an_earlier_grant_expiry(failed_review):  # noqa: F811
    """The reviewer's case: the grant lapses while the owner window is still open."""
    config, source, owner, old, reserve, receipt = failed_review
    authorized = grant(failed_review)
    directory = Path(config["scene_root"]) / owner["intent_id"]
    window = intake.effective_execution_expiry(
        directory, intake._read(directory / "intent.json", "intent_digest")
    )
    assert window > 200
    authorized["expires_at_epoch"] = window - 50
    authorized["authority_digest"] = canonical_digest(authorized, digest_field="authority_digest")
    path = directory / "visual-review-corrections" / "controls-adopted-placement.json"
    assert path.is_file()
    path.chmod(0o640)
    path.unlink()
    path.write_text(json.dumps(authorized), encoding="utf-8")
    packet = visual.discover(config=config, intent_id=owner["intent_id"], source=source,
                             expected_commit="b" * 40)
    retirement.retire_unmaterialized_adoptions(
        config=config, intent_id=owner["intent_id"], source=source,
        expected_production_commit="b" * 40, visual_review_continuation=packet,
    )
    for phase in ("destination", "construction", "controls"):
        reserve("controls-corrected-" + phase, 0.266666, commit="b")
    review = intake.reserve_scene_attempt(
        queue_root=config["scene_root"], intent_id=owner["intent_id"],
        attempt_id="controls-corrected-placement", source_commit="b" * 40,
        runtime_digest="sha256:" + "e" * 64, input_digest="sha256:" + "f" * 64,
        provider="openai", maximum_spend_usd=0.15, now=103, visual_review_authority=authorized,
    )
    return execution.bind_scene_attempt(review), authorized["expires_at_epoch"], window


def test_the_owner_window_and_a_visual_review_grant_report_their_expiries(failed_review) -> None:  # noqa: F811
    binding, grant_expiry, window = _review_with_an_earlier_grant_expiry(failed_review)
    check = dict(source_commit="b" * 40, provider="openai", maximum_spend_usd=0.15)

    with collecting_deadlines() as found:
        assert execution.scene_execution_authority_blockers(binding, now=104, **check) == []

    assert {grant_expiry, window} <= set(found)
    # At the grant's own expiry the answer flips to a blocker that is not an
    # inactive owner state, so a full catalog pass refuses the whole catalog.
    assert execution.scene_execution_authority_blockers(binding, now=grant_expiry, **check) == [
        "scene_execution_visual_review_correction_invalid"
    ]
    assert execution.scene_execution_authority_blockers(binding, now=window, **check) == [
        "scene_execution_owner_expired"
    ]


def test_the_owner_window_is_reported_without_any_grant(reserved) -> None:  # noqa: F811
    root, _run, owner, reserve, _originals = reserved
    attempt = reserve("plain-construction", 0.25)
    directory = root / owner["intent_id"]
    window = intake.effective_execution_expiry(
        directory, intake._read(directory / "intent.json", "intent_digest")
    )
    binding = execution.bind_scene_attempt(attempt)
    check = dict(source_commit="d" * 40, provider="vast", maximum_spend_usd=0.25)

    with collecting_deadlines() as found:
        assert execution.scene_execution_authority_blockers(binding, now=104, **check) == []

    assert found == [window]
    assert execution.scene_execution_authority_blockers(binding, now=window, **check) == [
        "scene_execution_owner_expired"
    ]


def test_a_preparation_grant_reports_its_expiry_and_its_start() -> None:
    from blueprint_pipeline.website_preparation_authority import validate_preparation_authority

    task = copy.deepcopy(VECTOR["task_context_response_preparation"])
    authority = copy.deepcopy(VECTOR["sponsorship_response"])
    authority.pop("assessment_preparation_proposal")
    authority["authority_digest"] = canonical_digest(authority, digest_field="authority_digest")

    with collecting_deadlines() as found:
        validate_preparation_authority(task_context=task, authority=authority, now=PREPARATION_NOW)

    issued = authority["consent"]["accepted_at_epoch"]
    assert {float(authority["expires_at_epoch"]), float(issued - 60)} <= set(found)
    with pytest.raises(ValueError):
        validate_preparation_authority(task_context=task, authority=authority,
                                       now=authority["expires_at_epoch"])
