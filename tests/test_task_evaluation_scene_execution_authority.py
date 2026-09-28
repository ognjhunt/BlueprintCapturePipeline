"""New paid admissions reopen owner consent; historical closeout remains available."""

import pytest

from blueprint_pipeline import task_evaluation_scene_execution_authority as authority
from blueprint_pipeline import task_evaluation_scene_intake as intake
from tests.test_task_evaluation_scene_intake import request, stage, attempt


@pytest.fixture
def bound(tmp_path, monkeypatch):
    intent = stage(tmp_path)
    reserved = attempt(tmp_path, intent)
    monkeypatch.setenv(intake.ROOT_ENV, str(tmp_path))
    monkeypatch.setenv(intake.CLIENTS_ENV, "webapp")
    profile = {**authority.bind_scene_attempt(reserved), "source_commit": reserved["source_commit"],
               "allocator": {"max_spend_usd": 2, "argv": ["--provider", "vast"]}}
    return tmp_path, intent, profile


def test_exact_bound_attempt_admits_and_legacy_keeps_original_gate(bound):
    root, intent, profile = bound
    assert authority.scene_execution_authority_blockers(profile, now=102) == []
    assert authority.scene_execution_authority_blockers({}, now=102) == []
    assert authority.scene_execution_authority_blockers({"scene_intent_digest": intent["intent_digest"]}) == [
        "scene_execution_owner_binding_missing"]


def test_revoked_or_expired_consent_stops_new_admission_not_structural_readback(bound):
    root, intent, profile = bound
    assert authority.scene_execution_authority_blockers(profile, now=1001) == ["scene_execution_owner_expired"]
    intake.revoke_scene_intent(queue_root=root, intent_id=intent["intent_id"], intent_digest=intent["intent_digest"],
                               owner=request()["owner"], now=103)
    assert authority.scene_execution_authority_blockers(profile, now=104) == ["scene_execution_owner_revoked"]
    assert authority.scene_execution_authority_blockers(profile, reopen_records=False, now=104) == []


def test_changed_release_spend_or_policy_pair_refused(bound):
    root, intent, profile = bound
    assert authority.scene_execution_authority_blockers(profile, source_commit="a" * 40, now=102) == [
        "scene_execution_owner_binding_invalid"]
    assert authority.scene_execution_authority_blockers(profile, maximum_spend_usd=2.01, now=102) == [
        "scene_execution_owner_reservation_insufficient"]
    assert authority.scene_execution_authority_blockers(profile, provider="runpod", now=102) == [
        "scene_execution_owner_provider_mismatch"]
    pair = request()["execution"]["policy_candidates"]
    assert authority.scene_execution_authority_blockers({**profile, "scene_policy_candidates": pair[::-1]}, now=102) == []
    wrong = [{**pair[0], "artifact_digest": "sha256:" + "a" * 64}, pair[1]]
    assert authority.scene_execution_authority_blockers({**profile, "scene_policy_candidates": wrong}, now=102) == [
        "scene_execution_owner_policy_pair_mismatch"]


def test_issuer_is_rechecked_and_missing_store_cannot_be_legacy(bound, monkeypatch):
    root, intent, profile = bound
    monkeypatch.setenv(intake.CLIENTS_ENV, "other-service")
    assert authority.scene_execution_authority_blockers(profile, now=102) == [
        "scene_execution_owner_issuer_not_authorized"]
    monkeypatch.delenv(intake.ROOT_ENV)
    assert authority.scene_execution_authority_blockers(profile, now=102) == [
        "scene_execution_owner_store_missing"]
