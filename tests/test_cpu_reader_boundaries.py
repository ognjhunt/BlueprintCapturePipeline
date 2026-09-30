"""Existing CPU consumers retain transport, evidence and publication semantics."""

import json
from pathlib import Path

import pytest

from blueprint_pipeline import control_plane_capacity_evidence as capacity
from blueprint_pipeline import provider_credentials, provider_transport, vast_api_transport
from blueprint_pipeline import provider_credit_admission as credit
from blueprint_pipeline import task_evaluation_scene_configuration_activation_identity as identity
from blueprint_pipeline import task_evaluation_scene_intent_storage as storage
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def test_default_credit_reader_uses_existing_fixture_key_and_exact_rest_transport(
    tmp_path, monkeypatch,
):
    key = tmp_path / "vast-key"
    key.write_text("fixture-account-key\n")
    monkeypatch.setenv("VAST_API_KEY_FILE", str(key))
    monkeypatch.setenv(provider_credentials.PROVIDER_SECRETS_DIR_ENV, str(tmp_path / "absent"))
    calls = []

    def request(**kwargs):
        calls.append(kwargs)
        return 200, {"credit": 3.0, "email": "private@example.invalid"}

    monkeypatch.setattr(provider_transport, "provider_json_request", request)
    observation = credit.observe_vast_credit(now=100)
    assert calls == [{
        "url": "https://console.vast.ai/api/v0/users/current/",
        "method": "GET",
        "headers": {"Authorization": "Bearer fixture-account-key", "Content-Type": "application/json"},
        "body_json": None,
        "timeout_seconds": 10,
    }]
    assert observation["observation_digest"] == canonical_digest(
        observation, digest_field="observation_digest",
    )
    assert observation["provider_mutations_performed"] == 0
    assert "fixture-account-key" not in json.dumps(observation)
    assert "private@example.invalid" not in json.dumps(observation)
    assert credit.credit_admission(observation, required_usd=2, now=100)["status"] == "admitted"
    assert credit.credit_admission(observation, required_usd=2.01, now=100)["status"] == "blocked"


@pytest.mark.parametrize("failure", [OSError("fixture-account-key"), RuntimeError("private-account")])
def test_default_credit_transport_failure_retains_only_screened_evidence(monkeypatch, failure):
    def request(**_kwargs):
        raise failure

    monkeypatch.setattr(provider_transport, "provider_json_request", request)
    observation = credit.observe_vast_credit(api_key="fixture-account-key", now=100)
    assert observation["blockers"] == ["provider_credit_transport_failed"]
    assert observation["credit_usd"] is None
    assert "fixture-account-key" not in json.dumps(observation)
    assert "private-account" not in json.dumps(observation)


def test_vast_transport_preserves_absolute_url_payload_default_timeout_and_inventory_retry(monkeypatch):
    calls = []
    monkeypatch.setattr(provider_transport, "provider_json_request", lambda **kwargs: calls.append(kwargs) or (200, {}))
    vast_api_transport._api_json(method="POST", path="https://fixture.invalid/action", api_key="fixture", payload={"id": 1})
    assert calls[0]["url"] == "https://fixture.invalid/action"
    assert calls[0]["body_json"] == {"id": 1}
    assert calls[0]["timeout_seconds"] == 30
    assert "read_retry" not in calls[0]
    vast_api_transport._api_json(method="get", path="/instances/", api_key="fixture")
    assert calls[1]["url"] == "https://console.vast.ai/api/v0/instances/"
    assert calls[1]["read_retry"] is not None


def test_configured_missing_key_remains_authoritative_over_developer_fallback(tmp_path, monkeypatch):
    developer = tmp_path / "home"
    (developer / ".blueprint-secrets").mkdir(parents=True)
    (developer / ".blueprint-secrets" / "vast_api_key").write_text("fixture-developer-key")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: developer))
    monkeypatch.delenv("VAST_API_KEY_FILE", raising=False)
    monkeypatch.setenv(provider_credentials.PROVIDER_SECRETS_DIR_ENV, str(tmp_path / "configured-absent"))
    assert provider_credentials._read_secret("vast_api_key") is None


def test_attention_summary_refuses_symlink_nonobject_and_oversize_bytes(tmp_path):
    target = tmp_path / "summary.json"
    target.write_text('{"status":"measured"}')
    assert capacity._read_attention_summary(target) == {"status": "measured"}
    link = tmp_path / "link.json"
    link.symlink_to(target)
    assert capacity._read_attention_summary(link) == {"status": "unreadable"}
    target.write_text("[]")
    assert capacity._read_attention_summary(target) == {"status": "unreadable"}
    target.write_bytes(b" " * (128 * 1024 + 1))
    assert capacity._read_attention_summary(target) == {"status": "unreadable"}
    assert capacity._read_attention_summary(tmp_path / "absent.json") is None


def test_bounded_activation_identity_preserves_identifier_errors_and_stability():
    assert identity._activation_id("fixture-preparation") == "fixture-activation-auto"
    assert identity._bounded_launch_id("fixture-activation") == "fixture-activation-launch"
    long_id = "x" * 191
    bounded = identity._bounded_launch_id(long_id)
    assert len(bounded) <= 192
    assert bounded == identity._bounded_launch_id(long_id)
    assert bounded.endswith("-launch")
    with pytest.raises(identity.SceneConfigurationActivationAutomationError, match="identifier_invalid"):
        identity._activation_id("unsafe/path")


def test_scene_exclusive_writer_preserves_checkpoint_refusal_before_bytes(tmp_path, monkeypatch):
    from blueprint_pipeline.control_plane_lane_experiment_errors import OwnerTargetVersionError
    from blueprint_pipeline.task_evaluation_scene_intent_contracts import SceneIntakeError

    def refuse():
        raise OwnerTargetVersionError("experiment_publisher_input_limit")

    monkeypatch.setattr(storage, "_publisher_checkpoint", refuse)
    destination = tmp_path / "intent.json"
    with pytest.raises(SceneIntakeError, match="experiment_publisher_input_limit"):
        storage.write_exclusive(destination, {"intent": "fixture"})
    assert not destination.exists()


def test_scene_exclusive_writer_keeps_original_locked_writer_and_immutable_bytes(tmp_path):
    from blueprint_pipeline.task_evaluation_launch_preparation_storage import (
        _write_launch_preparation_record_exclusive_locked,
    )

    assert storage._write_exclusive_native is _write_launch_preparation_record_exclusive_locked
    destination = tmp_path / "intent.json"
    storage.write_exclusive(destination, {"intent": "fixture"})
    before = destination.read_bytes()
    with pytest.raises(FileExistsError):
        storage.write_exclusive(destination, {"intent": "changed"})
    assert destination.read_bytes() == before
    assert destination.stat().st_mode & 0o777 == 0o440
