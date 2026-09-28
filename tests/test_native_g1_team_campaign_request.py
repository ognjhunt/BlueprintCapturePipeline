"""A signed team choice must stay on one task packet before any paid work."""

from __future__ import annotations

from copy import deepcopy

import pytest

from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
from blueprint_pipeline.native_g1_development_campaign import _movement_handoff
from blueprint_pipeline.native_g1_team_campaign_request import (
    SCHEMA,
    validate_g1_team_campaign_request,
)
from tests.test_native_g1_development_campaign import _book_handoff


NOW = 1_790_400_000.0
OWNER = {"user_id": "team-user", "organization_id": "team-org"}


def _request(tmp_path, monkeypatch):
    book = _book_handoff(tmp_path, monkeypatch)
    movement = _movement_handoff(book)
    setup = book["setup"]
    request = {
        "schema_version": SCHEMA,
        "run_id": "g1-team-841757-1",
        "owner": OWNER,
        "scene_id": setup["scene_id"],
        "task_id": setup["task_id"],
        "source_packet_receipt_digest": setup["source_packet_receipt_digest"],
        "robot_preset_id": "unitree_g1_dex3_sonic_v1",
        "book_handoff": book,
        "movement_handoff": movement,
        "authorization": {
            "maximum_cost_usd": 12,
            "hard_ttl_seconds": 14400,
            "expires_at_epoch": NOW + 3600,
            "retry_cap": 0,
        },
        "claim_ceiling": "development_only",
        "public_redistribution_authorized": False,
    }
    request["request_digest"] = cross_runtime_canonical_digest(
        request, digest_field="request_digest"
    )
    return setup, request


def _reseal(request):
    request["request_digest"] = cross_runtime_canonical_digest(
        request, digest_field="request_digest"
    )
    return request


def test_exact_team_choices_are_admitted_without_execution(tmp_path, monkeypatch):
    setup, request = _request(tmp_path, monkeypatch)
    admitted = validate_g1_team_campaign_request(
        request, trusted_setup=setup, authenticated_owner=OWNER, now_epoch=NOW
    )
    assert admitted == request
    assert admitted["book_handoff"]["choice"]["objective_id"] == "task_success"
    assert admitted["movement_handoff"]["choice"]["objective_id"] == "g1_navigation_goal"
    assert admitted["claim_ceiling"] == "development_only"


@pytest.mark.parametrize("change, error", [
    ("other_owner", "binding"),
    ("other_packet", "binding"),
    ("swapped_objectives", "policy_or_packet"),
    ("over_budget", "authorization"),
    ("expired", "authorization"),
    ("retry", "authorization"),
    ("public", "binding"),
    ("extra_field", "shape"),
    ("stale_digest", "binding"),
])
def test_refuses_unbound_or_unapproved_campaign(
    tmp_path, monkeypatch, change, error
):
    setup, request = _request(tmp_path, monkeypatch)
    request = deepcopy(request)
    if change == "other_owner":
        request["owner"]["user_id"] = "different"
    elif change == "other_packet":
        request["source_packet_receipt_digest"] = "sha256:" + "9" * 64
    elif change == "swapped_objectives":
        request["book_handoff"], request["movement_handoff"] = (
            request["movement_handoff"], request["book_handoff"]
        )
    elif change == "over_budget":
        request["authorization"]["maximum_cost_usd"] = 12.01
    elif change == "expired":
        request["authorization"]["expires_at_epoch"] = NOW
    elif change == "retry":
        request["authorization"]["retry_cap"] = 1
    elif change == "public":
        request["public_redistribution_authorized"] = True
    elif change == "extra_field":
        request["candidate_override"] = "unchecked"
    elif change == "stale_digest":
        request["run_id"] = "changed"
    if change != "stale_digest":
        _reseal(request)
    pattern = "g1_team_campaign_request_" + error if error in {"binding", "shape"} else "g1_team_campaign_" + error
    with pytest.raises(ValueError, match=pattern):
        validate_g1_team_campaign_request(
            request, trusted_setup=setup, authenticated_owner=OWNER, now_epoch=NOW
        )
