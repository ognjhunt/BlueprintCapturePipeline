"""A completed preparation re-issued under a renewed grant keeps its capture's identity.

capture-447ae4f4 (2026-11-02): the site's grant expired and its intent was
revoked by a WebApp bug. The sponsor renewed the grant; the re-issued request
must stay in the capture's job so every sibling shares the original native
limits, and the WebApp must read the same identity (shared vector).
"""
import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline import website_preparation_revision as revisions
from blueprint_pipeline.website_preparation_contracts import preparation_consent_valid
from blueprint_pipeline.website_preparation_revision_identity import (
    RENEWAL_FIELD, capture_id, renewal, requires_retained_authority)

VECTOR = json.loads((Path(__file__).parent / "fixtures/website-preparation-renewal.v1.json").read_text())


def test_the_shared_vector_round_trips_and_keeps_the_capture():
    original, renewed, again = VECTOR["original"], VECTOR["renewed"], VECTOR["renewed_again"]
    assert renewal(original) is None
    assert renewal(renewed) == {"capture_id": original["submission_id"],
                                "parent_submission_id": original["submission_id"]}
    assert capture_id(renewed) == capture_id(again) == original["submission_id"]
    assert renewal(again)["parent_submission_id"] == renewed["submission_id"]
    assert requires_retained_authority(renewed) and requires_retained_authority(again)
    # Only the consent window, expiry, renewal field and identity differ.
    def strip(value):
        return {k: v for k, v in value.items() if k not in {"submission_id", "consent", "execution", "task"}}
    assert strip(renewed) == strip(original)
    assert {k: v for k, v in renewed["task"]["subject"].items() if k != RENEWAL_FIELD} == original["task"]["subject"]


@pytest.mark.parametrize("forgery", ["capture_id", "parent", "submission_id", "purpose", "policies",
                                     "prefix_without_field", "extra_key", "with_revision"])
def test_a_changed_renewal_is_refused(forgery):
    value = copy.deepcopy(VECTOR["renewed"])
    field = value["task"]["subject"][RENEWAL_FIELD]
    if forgery == "capture_id":
        field["capture_id"] = "other-capture"
    elif forgery == "parent":
        field["parent_submission_id"] = "someone-else"
    elif forgery == "submission_id":
        value["submission_id"] = "website-renewal-" + "0" * 64
    elif forgery == "purpose":
        value["execution"]["purpose"] = "evaluation"
    elif forgery == "policies":
        value["execution"]["policy_candidates"] = ["policy-a"]
    elif forgery == "prefix_without_field":
        del value["task"]["subject"][RENEWAL_FIELD]
    elif forgery == "extra_key":
        field["note"] = "x"
    else:
        value["task"]["subject"][revisions.FIELD] = {"capture_id": field["capture_id"],
                                                     "corrections_digest": "sha256:" + "a" * 64}
    with pytest.raises(ValueError):
        capture_id(value)


def test_a_confirmed_task_is_valid_preparation_consent_only_as_a_derivative():
    renewed = copy.deepcopy(VECTOR["renewed"])
    assert preparation_consent_valid(renewed) is True  # the vector's task is unconfirmed
    original = copy.deepcopy(VECTOR["original"])
    original["consent"]["task_confirmed"] = True
    assert preparation_consent_valid(original) is False


def test_a_renewal_workspace_resolves_to_its_capture(tmp_path):
    base = tmp_path / "pipeline" / "website_scene_preparation"
    plain = base / "renewals" / ("b" * 64) / "preparation.json"
    corrected = base / "revisions" / ("a" * 64) / "renewals" / ("b" * 64) / "preparation.json"
    assert revisions.capture_root_for_preparation(plain) == tmp_path
    assert revisions.capture_root_for_preparation(corrected) == tmp_path
    assert revisions.capture_root_for_preparation(base / "renewals" / "short" / "preparation.json") is None
