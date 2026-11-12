"""Part 6 (2026-11-12): no stage-6 ending tells the site "Blueprint will review" when nothing reviews.

Each held shape below told the site, through ``task_evidence_incomplete``, that Blueprint would review the
saved evidence. Now, once the website lists the new reasons (``NEW_REASONS_ENV``), each says what is true:
the specific ask (the site said none of our objects is the one: film it), or that this release cannot
prepare the job and why (a retained blocker), or that a step kept failing and runs again on each release.
With the switch off, every reader gets exactly what it got before.
"""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_preparation_outcome.py
#   src/blueprint_pipeline/website_preparation_status.py
#   src/blueprint_pipeline/website_preparation_retry.py
import pytest

from blueprint_pipeline import website_preparation_outcome as outcome
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

CONTEXT = "sha256:" + "c" * 64


def _seal(path, value):
    value = {**value, "digest": canonical_digest(value, digest_field="digest")}
    write_json(path, value)
    return value


def _held(tmp_path, blocker):
    """c33's retained shape before part 5: the preparation held on one missing-evidence blocker."""
    root = tmp_path / "scenes/site-x/captures/walkthrough-x"
    directory = root / "pipeline/website_scene_preparation"
    preparation = _seal(directory / "preparation.json", {"schema_version": "website_scene_preparation.v1",
                                                         "binding": {"task_context_digest": CONTEXT}})
    _seal(directory / "handoff.json", {"schema_version": "website_scene_handoff.v1", "scene_id": "site-x",
          "capture_id": "walkthrough-x", "status": "needs_input", "blockers": [blocker],
          "preparation_path": str(directory / "preparation.json"), "preparation_digest": preparation["digest"]})
    return root


def _read(root, *, recapture=True):
    return outcome.read_preparation_outcome(capture_root=root, task_context_digest=CONTEXT,
                                            state="awaiting_inputs", recapture=recapture)


@pytest.mark.parametrize("blocker", ["task_destination_interior_geometry_required",
                                     "website_owner_success_criteria_translation_required"])
def test_a_held_blocker_says_this_release_cannot_prepare_it_and_why(tmp_path, monkeypatch, blocker):
    root = _held(tmp_path, blocker)
    monkeypatch.delenv(outcome.ENDINGS_ENV, raising=False)
    assert _read(root)["reason"] == "task_evidence_incomplete"  # unchanged until the website lists it
    monkeypatch.setenv(outcome.ENDINGS_ENV, "true")
    said = _read(root)
    assert said["reason"] == outcome.NOT_POSSIBLE and said["cause"] == blocker
    # Readers before v7 keep their closed vocabulary.
    assert _read(root, recapture=False)["reason"] == "task_evidence_incomplete"


def _spent_ledger(kind):
    from blueprint_pipeline.website_preparation_retry import FIELD, SCHEMA
    return {"status": "failed_retryable", "last_error": "clean_plate: refused", "capture_id": "walkthrough-x",
            "source_payload_sha256": "sha256:" + "1" * 64, "producer_delivery_key": "sha256:" + "2" * 64,
            FIELD: {"schema_version": SCHEMA, "source_payload_sha256": "sha256:" + "1" * 64,
                    "producer_delivery_key": "sha256:" + "2" * 64, "error": "clean_plate: refused", "kind": kind,
                    "retry_count": 7, "next_attempt_at": "2026-11-12T10:00:00Z", "release_commit": "a" * 40}}


def test_a_job_whose_timed_retries_are_spent_is_told_so(monkeypatch):
    from blueprint_pipeline.website_preparation_retry import REFUSAL, RETRIES_SPENT
    monkeypatch.delenv(outcome.ENDINGS_ENV, raising=False)
    assert outcome.retries_spent(_spent_ledger(RETRIES_SPENT), task_context_digest=CONTEXT) is None
    monkeypatch.setenv(outcome.ENDINGS_ENV, "true")
    said = outcome.retries_spent(_spent_ledger(RETRIES_SPENT), task_context_digest=CONTEXT)
    assert said["reason"] == outcome.RETRIES_SPENT and said["category"] == "processing_failure"
    # A job still on its timed schedule is not.
    assert outcome.retries_spent(_spent_ledger(REFUSAL), task_context_digest=CONTEXT) is None
