"""Time-only extension records of a scene intent's window, as the retired operator-door writer sealed them.

No person extends a window any more; these records, where they already exist, still count for every reader."""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_execution_window.py
#   src/blueprint_pipeline/task_evaluation_scene_execution_window_evidence.py
#   src/blueprint_pipeline/website_preparation_authority.py
#   src/blueprint_pipeline/website_preparation_contracts.py
#   src/blueprint_pipeline/task_evaluation_scene_lifecycle_finished.py

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from blueprint_pipeline import task_evaluation_scene_execution_window as window
from blueprint_pipeline import task_evaluation_scene_intake as intake
from tests import scene_window_extension_records as records
from tests.test_task_evaluation_scene_intake import attempt, request, stage

RIGHTS = {"consent_revoked": False, "derived_scene_generation_allowed": True}
REQUEST_ID = "20261031T120000Z-extend-scene-execution-window-0000abcd"


def extend(root, owner_record, **overrides):
    args = dict(queue_root=root, intent_id=owner_record["intent_id"], intent_digest=owner_record["intent_digest"],
                expires_at_epoch=2000, authorization_reference="founder approval 2026-10-31 (time only)",
                ack=window.ACK, door_caller="cloud-scene-runs", door_request_id=REQUEST_ID,
                trusted_clients={"webapp"}, capture_rights=RIGHTS, apply=True, now=1001)
    args.update(overrides)
    return records.extend_scene_execution_window_by_operator(**args)


def _records(root, owner_record):
    return sorted((root / owner_record["intent_id"] / window.DIRECTORY).glob("*.json"))


def _intent(root, owner_record):
    return intake._read(root / owner_record["intent_id"] / "intent.json", "intent_digest")


def test_operator_extension_reopens_an_expired_intent_inside_its_original_caps(tmp_path):
    owner = stage(tmp_path)
    original = (tmp_path / owner["intent_id"] / "intent.json").read_bytes()
    with pytest.raises(intake.SceneIntakeError, match="authority_expired"):
        attempt(tmp_path, owner, now=1100)
    result = extend(tmp_path, owner)
    assert result["status"] == "execution_window_extended"
    record = result["extension"]
    assert record["authorized_via"] == "operator_door" and record["door_caller"] == "cloud-scene-runs"
    assert record["door_request_id"] == REQUEST_ID and record["scope"] == "execution_time_only"
    assert record["authorization_reference"] == "founder approval 2026-10-31 (time only)"
    # The door never claims the WebApp's issuer identity; it only binds to it.
    assert "authenticated_issuer" not in record and record["intent_authenticated_issuer"] == "webapp"
    intent = _intent(tmp_path, owner)
    assert record["unchanged_execution_bounds"] == records.operator_bounds(intent)
    assert record["unchanged_execution_bounds"]["execution"] == {
        key: value for key, value in request()["execution"].items() if key != "expires_at_epoch"}
    assert (tmp_path / owner["intent_id"] / "intent.json").read_bytes() == original
    assert window.effective_execution_expiry(tmp_path / owner["intent_id"], intent) == 2000
    # Spend and attempt caps are exactly the sealed ones: two attempts, USD 4 in all.
    attempt(tmp_path, owner, "a1", now=1100)
    attempt(tmp_path, owner, "a2", now=1100)
    with pytest.raises(intake.SceneIntakeError, match="attempt_cap_exhausted"):
        attempt(tmp_path, owner, "a3", now=1101)
    assert intent["request"]["execution"]["max_total_spend_usd"] == 4


def test_plan_checks_everything_and_writes_nothing(tmp_path):
    owner = stage(tmp_path)
    planned = extend(tmp_path, owner, apply=False)
    assert planned["status"] == "execution_window_extension_planned"
    assert planned["extension"]["expires_at_epoch"] == 2000 and planned["previous_expires_at_epoch"] == 1000
    assert not _records(tmp_path, owner)


@pytest.mark.parametrize(("change", "code"), [
    (dict(expires_at_epoch=1001 + 7 * 86400 + 1), "expiry_out_of_bounds"),
    (dict(expires_at_epoch=1001), "expiry_out_of_bounds"),
    (dict(expires_at_epoch=900), "expiry_out_of_bounds"),
    (dict(expires_at_epoch=True), "expiry_out_of_bounds"),
    (dict(expires_at_epoch=float("inf")), "expiry_out_of_bounds"),
    # Within 7 days of now but more than 7 days past the sealed expiry.
    (dict(now=1000 + 6 * 86400, expires_at_epoch=1000 + 7 * 86400 + 1), "expiry_beyond_original_window"),
    (dict(intent_digest="sha256:" + "0" * 64), "intent_digest_mismatch"),
    (dict(capture_rights={**RIGHTS, "consent_revoked": True}), "capture_rights_revoked"),
    (dict(capture_rights={**RIGHTS, "derived_scene_generation_allowed": False}), "capture_rights_revoked"),
    (dict(capture_rights={}), "capture_rights_revoked"),
    (dict(ack="yes"), "extension_unacknowledged"),
    (dict(authorization_reference=" "), "authorization_reference_invalid"),
    (dict(authorization_reference="x" * 501), "authorization_reference_invalid"),
    (dict(authorization_reference="line\nbreak"), "authorization_reference_invalid"),
    (dict(door_caller="../root"), "extension_request_invalid"),
    (dict(door_request_id="20261031T120000Z-deploy-0000abcd"), "extension_request_invalid"),
    (dict(trusted_clients={"other"}), "intent_issuer_not_trusted"),
])
def test_operator_extension_refusals_write_nothing(tmp_path, change, code):
    owner = stage(tmp_path)
    with pytest.raises(ValueError, match=code):
        extend(tmp_path, owner, **change)
    assert not _records(tmp_path, owner)


def test_revoked_intent_cannot_be_renewed(tmp_path):
    owner = stage(tmp_path)
    intake.revoke_scene_intent(queue_root=tmp_path, intent_id=owner["intent_id"], intent_digest=owner["intent_digest"],
                               owner=request()["owner"], now=103)
    with pytest.raises(ValueError, match="intent_revoked"):
        extend(tmp_path, owner)
    assert not _records(tmp_path, owner)


def test_equal_or_earlier_expiry_is_already_covered_and_names_the_covering_record(tmp_path):
    owner = stage(tmp_path)
    first = extend(tmp_path, owner)["extension"]
    for expiry in (2000, 1500):
        again = extend(tmp_path, owner, expires_at_epoch=expiry, now=1002)
        assert again["status"] == "execution_window_already_covers_request"
        assert again["expires_at_epoch"] == 2000 and again["covering_extension"] == first
    assert len(_records(tmp_path, owner)) == 1
    later = extend(tmp_path, owner, expires_at_epoch=3000, now=1003)
    assert later["status"] == "execution_window_extended" and len(_records(tmp_path, owner)) == 2
    assert window.effective_execution_expiry(tmp_path / owner["intent_id"], _intent(tmp_path, owner)) == 3000


@pytest.mark.parametrize("mutation", ["bounds", "issuer_claim", "beyond_original", "caller", "reference"])
def test_resealed_operator_records_fail_every_reader_closed(tmp_path, mutation):
    from blueprint_pipeline.task_evaluation_scene_intent_contracts import _seal
    owner = stage(tmp_path)
    record = extend(tmp_path, owner)["extension"]
    [path] = _records(tmp_path, owner)
    value = {key: item for key, item in record.items() if key != "extension_digest"}
    if mutation == "bounds":
        value["unchanged_execution_bounds"] = copy.deepcopy(value["unchanged_execution_bounds"])
        value["unchanged_execution_bounds"]["execution"]["max_total_spend_usd"] = 100
    elif mutation == "issuer_claim":
        value["authenticated_issuer"] = "webapp"
    elif mutation == "beyond_original":
        value["expires_at_epoch"] = 1000 + 7 * 86400 + 1
        value["issued_at_epoch"] = 1000 + 6 * 86400
    elif mutation == "caller":
        value["door_caller"] = ""
    else:
        value["authorization_reference"] = ""
    sealed = _seal(value, "extension_digest")
    path.unlink()
    (path.parent / (sealed["extension_digest"][7:] + ".json")).write_text(json.dumps(sealed))
    with pytest.raises(ValueError, match="scene_execution_window_extension_invalid"):
        window.effective_execution_expiry(tmp_path / owner["intent_id"], _intent(tmp_path, owner))
    with pytest.raises(ValueError):
        attempt(tmp_path, owner, now=1100)


def test_owner_review_reader_honours_the_operator_extension(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_owner_authority import reopen_scene_intent
    owner = stage(tmp_path)
    monkeypatch.setenv(intake.ROOT_ENV, str(tmp_path))
    monkeypatch.setenv(intake.CLIENTS_ENV, "webapp")
    path = tmp_path / owner["intent_id"] / "intent.json"
    reference = {"path": str(path), "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                 "size_bytes": path.stat().st_size}
    with pytest.raises(ValueError, match="expired"):
        reopen_scene_intent(reference, now=1100)
    extend(tmp_path, owner)
    assert reopen_scene_intent(reference, now=1100)["intent_digest"] == owner["intent_digest"]
    with pytest.raises(ValueError, match="expired"):
        reopen_scene_intent(reference, now=2000)


def test_lifecycle_finished_reader_accepts_the_same_operator_record(tmp_path):
    from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget
    from blueprint_pipeline.task_evaluation_scene_intent_contracts import _seal
    from blueprint_pipeline.task_evaluation_scene_lifecycle_finished import finished
    owner = stage(tmp_path)
    record = extend(tmp_path, owner)["extension"]
    intent = _intent(tmp_path, owner)
    base = "/intents/" + owner["intent_id"]

    def observe(value, now):
        rows = [{"role": "intent", "path": base + "/intent.json", "value": intent},
                {"role": "extensions", "value": value,
                 "path": base + "/execution-window-extensions/" + value["extension_digest"][7:] + ".json"}]
        return finished({"chain_validated": True, "projection_state": "current"}, rows,
                        {"intent_root": "/intents"}, owner["intent_id"], now,
                        ReferenceCollectionBudget(monotonic=lambda: 0), extensions_observed=True)

    assert observe(record, 2000 + 7 * 86400 - 1)["status"] == "unknown"
    assert observe(record, 2000 + 7 * 86400)["status"] == "expired_grace_elapsed"
    forged = _seal({**{k: v for k, v in record.items() if k != "extension_digest"}, "door_caller": "../x"},
                   "extension_digest")
    with pytest.raises(ValueError):
        observe(forged, 2000 + 7 * 86400)


# --- retained preparation authority (task_confirmed false) ------------------------------------


def test_retained_preparation_authority_honours_the_same_effective_expiry(tmp_path, monkeypatch):
    from blueprint_pipeline import website_task_context as context
    from tests.test_website_preparation_contract_v2 import NOW, retained_source

    helper, task, grant, value = retained_source(tmp_path)
    expiry = grant["expires_at_epoch"]
    clock = [NOW]
    monkeypatch.setattr(helper, "current_time", lambda: clock[0])
    reads = []

    def signed_read(**kwargs):
        reads.append(kwargs["operation"])
        return copy.deepcopy(task if kwargs["operation"] == "task-context" else grant)

    monkeypatch.setattr(context, "website_webapp_request", signed_read)
    root = tmp_path / "intents"
    staged = intake.stage_scene_intent(value=value, queue_root=root, authenticated_client="blueprint-webapp",
                                       trusted_clients={"blueprint-webapp"}, now=NOW)
    common = dict(queue_root=root, intent_id=staged["intent_id"], attempt_id="scene-configuration-" + "c" * 24,
                  source_commit="a" * 40, runtime_digest="sha256:" + "b" * 64, input_digest="sha256:" + "c" * 64,
                  provider="vast", maximum_spend_usd=1)
    clock[0] = expiry + 60
    with pytest.raises(ValueError, match="scene_intake_preparation_authority_invalid"):
        intake.reserve_scene_attempt(**common, now=expiry + 60)
    records.extend_scene_execution_window_by_operator(queue_root=root, intent_id=staged["intent_id"],
        intent_digest=staged["intent_digest"], expires_at_epoch=expiry + 3600,
        authorization_reference="founder approval (time only)", ack=window.ACK, door_caller="cloud-scene-runs",
        door_request_id=REQUEST_ID, trusted_clients={"blueprint-webapp"}, capture_rights=RIGHTS, apply=True,
        now=expiry + 30)
    reserved = intake.reserve_scene_attempt(**common, now=expiry + 60)
    assert reserved["status"] == "reserved" and reserved["maximum_spend_usd"] == grant["max_total_spend_usd"]
    # The grant's caps are untouched: one paid attempt, no more.
    with pytest.raises(ValueError):
        intake.reserve_scene_attempt(**{**common, "attempt_id": "scene-configuration-" + "d" * 24}, now=expiry + 61)
    # Only non-creating grant reads happened; nothing renewed or re-issued the grant.
    assert "scene-sponsorship" in reads and grant == json.loads(json.dumps(grant))
    # Past the extended window the intent is held again.
    clock[0] = expiry + 3600
    with pytest.raises(ValueError):
        intake.reserve_scene_attempt(**{**common, "attempt_id": "scene-configuration-" + "e" * 24},
                                     now=expiry + 3600)


def test_grant_check_keeps_its_own_issued_window_and_seven_day_cap():
    from blueprint_pipeline.website_preparation_contracts import validate_preparation_authority
    from tests.test_website_preparation_contract_v2 import VECTOR

    task = copy.deepcopy(VECTOR["task_context_response_preparation"])
    grant = copy.deepcopy(VECTOR["sponsorship_response"])
    expiry = grant["expires_at_epoch"]
    with pytest.raises(ValueError):
        validate_preparation_authority(task_context=task, authority=grant, now=expiry + 1)
    assert validate_preparation_authority(task_context=task, authority=grant, now=expiry + 1,
                                          valid_until=expiry + 3600) == grant
    # Never past raw expiry + 7 days, whatever is asked.
    with pytest.raises(ValueError):
        validate_preparation_authority(task_context=task, authority=grant, now=expiry + 7 * 86400,
                                       valid_until=expiry + 30 * 86400)
    for bad in (True, float("nan"), "later"):
        with pytest.raises(ValueError):
            validate_preparation_authority(task_context=task, authority=grant, now=expiry - 1, valid_until=bad)
