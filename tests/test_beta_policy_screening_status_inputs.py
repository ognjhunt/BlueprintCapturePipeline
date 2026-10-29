"""Status summaries refuse malformed revisions and unsafe ledger inputs."""

import json
import os

import pytest

from blueprint_pipeline import beta_policy_screening_status as status
from tests.test_beta_policy_screening_status import _ledger


@pytest.mark.parametrize("revision,published", [(True, 1), (1, True), (1, 2)])
def test_malformed_campaign_revision_cannot_be_reported_as_known_coverage(tmp_path, revision, published):
    _ledger(tmp_path, {"fixture-private-campaign": {"revision": revision, "published_revision": published,
        "created_at": "1970-01-01T00:01:40+00:00", "updated_at": "1970-01-01T00:05:00+00:00",
        "candidates": {"fixture": {"status": "completed", "blockers": []}}}})
    result = status.build_worker_status(tmp_path, now=1000)
    assert result["status"] == "unknown", f"malformed campaign metadata was accepted as {result['status']}"
    assert result["coverage"] == "unavailable"


def test_unknown_candidate_state_is_typed_unknown_and_never_idle(tmp_path):
    _ledger(tmp_path, {"fixture-private-campaign": {"revision": 1, "published_revision": 1,
        "candidates": {"fixture": {"status": "not-a-canonical-state", "blockers": []}}}})
    assert status.build_worker_status(tmp_path, now=1000)["reason"] == "screening_campaign_state_invalid"


@pytest.mark.parametrize("unsafe", ["symlink", "fifo"])
def test_unsafe_ledger_type_fails_closed_without_blocking(tmp_path, unsafe):
    ledger = tmp_path / "screening_ledger.v1.json"
    target = tmp_path / "private-fixture.json"
    if unsafe == "symlink":
        target.write_text("private remote body fixture")
        ledger.symlink_to(target)
    else:
        os.mkfifo(ledger)
    result = status.build_worker_status(tmp_path, now=1000)
    assert result["status"] == "unknown" and result["reason"] == "screening_ledger_unavailable"
    assert "private remote" not in json.dumps(result)


def test_status_symlink_is_not_followed_or_replaced(tmp_path):
    _ledger(tmp_path, {})
    target = tmp_path / "private-status-target.json"
    target.write_bytes(b"retained target must stay exact")
    (tmp_path / status.STATUS_FILENAME).symlink_to(target)
    result = status.write_worker_status(tmp_path)
    assert result["status"] == "unknown" and result["reason"] == "screening_status_path_unsafe"
    assert target.read_bytes() == b"retained target must stay exact"
    assert (tmp_path / status.STATUS_FILENAME).is_symlink()


def test_prior_failure_is_rebuilt_from_enums_without_private_fields_or_guessed_retry(tmp_path):
    _ledger(tmp_path, {})
    (tmp_path / status.STATUS_FILENAME).write_text(json.dumps({
        "schema_version": status.SCHEMA_VERSION,
        "last_successful_publication_at_epoch": "private-token-instead-of-time",
        "last_publication_failure": {"category": "auth", "owner": "private-customer-body",
            "next_action": "https://private-url/?token=secret", "observed_at_epoch": "private-error",
            "next_retry_at_epoch": float("nan")}}))
    value = status.write_worker_status(tmp_path)
    failure = value["last_publication_failure"]
    assert value["last_successful_publication_at_epoch"] is None
    assert failure == {"category": "auth", "owner": "pipeline_transport",
        "next_action": "check_existing_sync_credential", "next_retry_at_epoch": None,
        "observed_at_epoch": None}
    assert "private" not in json.dumps(value) and "https://" not in json.dumps(value)
    assert json.loads((tmp_path / status.STATUS_FILENAME).read_text()) == value
