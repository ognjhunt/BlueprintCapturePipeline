"""A navigation intake re-issued for the same capture keeps the capture's identity, byte for byte in both runtimes.

The library-aisle job (capture-5b067498, 2026-11-08): its splat was rebuilt, so its scene half is derived
again from the new Atlas result (``website_held_handoff_replay``, ``atlas_result_superseded``). The website
and the Pipeline intake keep one intake per submission, so the new scene half's intake is the capture's
intake re-issued under its own submission: the marker names the capture, the parent submission and the
Atlas result, and the submission is the prefix and the digest of the rest. The WebApp reads the same
shared vector (``server/tests/website-navigation-reissue.test.ts``).
"""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_preparation_revision_identity.py
#   src/blueprint_pipeline/task_evaluation_scene_intent_contracts.py
#   src/blueprint_pipeline/website_navigation_intake.py
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.task_evaluation_scene_intake import stage_scene_intent
from blueprint_pipeline.task_evaluation_scene_intent_contracts import SceneIntakeError, validate_request
from blueprint_pipeline.website_navigation_intake import OUTBOX_FILENAME, OUTBOX_SCHEMA, reissue_parent
from blueprint_pipeline.website_preparation_revision_identity import (
    FIELD, REISSUE_FIELD, REISSUE_PREFIX, capture_id, reissue, renewal, requires_retained_authority, revision)

VECTOR = json.loads((Path(__file__).parent / "fixtures/website-navigation-reissue.v1.json").read_text())


def test_the_shared_vector_round_trips_and_keeps_the_capture():
    original, reissued, again = VECTOR["original"], VECTOR["reissued"], VECTOR["reissued_again"]
    assert reissue(original) is None and capture_id(original) == original["submission_id"]
    assert reissue(reissued) == {"capture_id": original["submission_id"],
                                 "parent_submission_id": original["submission_id"],
                                 "atlas_result_digest": "sha256:" + "a" * 64}
    assert reissue(again)["parent_submission_id"] == reissued["submission_id"]
    assert capture_id(reissued) == capture_id(again) == original["submission_id"]
    assert revision(reissued) is None and renewal(reissued) is None
    # The capture's own consent: a confirmed brief, read by the website, never retained preparation authority.
    assert not requires_retained_authority(reissued) and reissued["consent"]["task_confirmed"] is True
    # The submission is the prefix and the cross-runtime digest of everything else.
    body = {key: value for key, value in reissued.items() if key != "submission_id"}
    assert reissued["submission_id"] == REISSUE_PREFIX + cross_runtime_canonical_digest(body)[7:]
    # Only the scene the re-issue names, its marker and its identity differ from the original.
    def strip(value):
        return {key: item for key, item in value.items() if key not in {"submission_id", "source", "task"}}
    assert strip(reissued) == strip(original)
    assert {key: item for key, item in reissued["task"].items()
            if key not in {REISSUE_FIELD, "scene_revision_digest"}} == {
        key: item for key, item in original["task"].items() if key != "scene_revision_digest"}


@pytest.mark.parametrize("forgery", ["capture_id", "parent", "submission_id", "purpose", "policies", "atlas_result",
                                     "prefix_without_field", "extra_key", "rigid_task", "with_revision",
                                     "changed_scene", "null_marker"])
def test_a_changed_reissue_is_refused(forgery):
    value = copy.deepcopy(VECTOR["reissued"])
    field = value["task"][REISSUE_FIELD]
    if forgery == "capture_id":
        field["capture_id"] = "other-capture"
    elif forgery == "parent":
        field["parent_submission_id"] = "someone-else"
    elif forgery == "submission_id":
        value["submission_id"] = REISSUE_PREFIX + "0" * 64
    elif forgery == "purpose":
        value["execution"]["purpose"] = "evaluation"
    elif forgery == "policies":
        value["execution"]["policy_candidates"] = ["policy-a"]
    elif forgery == "atlas_result":
        field["atlas_result_digest"] = "atlas-result"
    elif forgery == "prefix_without_field":
        del value["task"][REISSUE_FIELD]
    elif forgery == "extra_key":
        field["note"] = "x"
    elif forgery == "rigid_task":
        value["task"]["strategy"] = "pick_and_place"
    elif forgery == "with_revision":
        value["task"]["subject"] = {FIELD: {"capture_id": field["capture_id"],
                                            "corrections_digest": "sha256:" + "a" * 64}}
    elif forgery == "changed_scene":
        value["task"]["scene_revision_digest"] = "sha256:" + "9" * 64
    else:
        value["task"][REISSUE_FIELD] = None
    with pytest.raises(ValueError):
        capture_id(value)


def test_the_pipeline_intake_stages_a_reissue_as_its_own_intent_and_refuses_a_forged_one(tmp_path):
    original, reissued = VECTOR["original"], VECTOR["reissued"]
    assert validate_request(reissued, now=100) == reissued
    staged = [stage_scene_intent(value=value, queue_root=tmp_path, authenticated_client="blueprint-webapp",
                                 trusted_clients={"blueprint-webapp"}, now=100) for value in (original, reissued)]
    # Its own submission, so its own intent: never the original's idempotency conflict.
    assert staged[0]["intent_id"] != staged[1]["intent_id"]
    assert stage_scene_intent(value=reissued, queue_root=tmp_path, authenticated_client="blueprint-webapp",
                              trusted_clients={"blueprint-webapp"}, now=100)["intent_id"] == staged[1]["intent_id"]
    forged = copy.deepcopy(reissued)
    forged["task"][REISSUE_FIELD]["capture_id"] = "other-capture"
    with pytest.raises(SceneIntakeError, match="scene_intake_navigation_reissue_invalid"):
        validate_request(forged, now=100)
    prefixed = copy.deepcopy(original)
    prefixed["submission_id"] = REISSUE_PREFIX + "0" * 64
    with pytest.raises(SceneIntakeError, match="scene_intake_navigation_reissue_invalid"):
        validate_request(prefixed, now=100)


def _outbox(directory: Path, request: dict) -> None:
    value = {"schema_version": OUTBOX_SCHEMA, "request": request, "outbox_digest": ""}
    value["outbox_digest"] = canonical_digest(value, digest_field="outbox_digest")
    directory.mkdir(parents=True)
    (directory / OUTBOX_FILENAME).write_text(json.dumps(value))


def test_the_parent_is_the_latest_intake_the_capture_holds_for_an_earlier_record(tmp_path):
    original, reissued, again = VECTOR["original"], VECTOR["reissued"], VECTOR["reissued_again"]
    capture = original["submission_id"]
    own = tmp_path / "navigation_scene_inputs/own/record.json"
    # No earlier intake: the capture's own submission, not a re-issue.
    assert reissue_parent(own, capture_id=capture) is None
    _outbox(tmp_path / "navigation_scene_inputs/first", original)
    assert reissue_parent(own, capture_id=capture) == capture
    _outbox(tmp_path / "navigation_scene_inputs/second", reissued)
    assert reissue_parent(own, capture_id=capture) == reissued["submission_id"]
    # This record's own outbox is never its parent.
    _outbox(own.parent, again)
    assert reissue_parent(own, capture_id=capture) == reissued["submission_id"]
    # Two re-issues of the same parent leave the lineage unknown: the capture's own submission is named.
    fork = copy.deepcopy(reissued)
    fork["task"][REISSUE_FIELD]["atlas_result_digest"] = "sha256:" + "c" * 64
    fork["submission_id"] = REISSUE_PREFIX + cross_runtime_canonical_digest(
        {key: value for key, value in fork.items() if key != "submission_id"})[7:]
    _outbox(tmp_path / "navigation_scene_inputs/fork", fork)
    assert reissue_parent(own, capture_id=capture) == capture
