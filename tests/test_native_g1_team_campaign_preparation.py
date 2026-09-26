"""Accepted G1 team requests become verified controller inputs without spend."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest as digest
from blueprint_pipeline.native_g1_team_campaign_intake import stage_g1_team_campaign
from blueprint_pipeline.native_g1_team_campaign_preparation import prepare_g1_team_campaign
from tests.test_native_g1_team_campaign_intake import _registry
from tests.test_native_g1_team_campaign_request import _request, _reseal


COMMIT = "a" * 40


def _accepted(tmp_path, monkeypatch):
    setup, request = _request(tmp_path, monkeypatch)
    moment = time.time()
    request["authorization"]["expires_at_epoch"] = moment + 1800
    _reseal(request)
    registry, binding = _registry(tmp_path, request)
    monkeypatch.setattr(
        "blueprint_pipeline.native_g1_team_campaign_intake.make_packet_planning_setup",
        lambda **_: setup,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.native_g1_team_campaign_preparation.make_packet_planning_setup",
        lambda **_: setup,
    )
    queue = tmp_path / "queue"
    receipt = stage_g1_team_campaign(
        value=request, registry_path=registry, queue_root=queue,
        authenticated_client="blueprint-webapp", trusted_clients={"blueprint-webapp"},
        now_epoch=moment,
    )
    return registry, binding, queue / receipt["intent_id"] / "intent.json", request


def test_preparation_rechecks_binding_and_builds_one_no_spend_bundle(tmp_path, monkeypatch):
    registry, binding, intent, request = _accepted(tmp_path, monkeypatch)
    calls = []
    def build(**kwargs):
        calls.append(kwargs)
        assert kwargs["book_handoff"].is_file()
        assert kwargs["movement_handoff"].is_file()
        assert json.loads(kwargs["book_handoff"].read_text()) == request["book_handoff"]
        assert json.loads(kwargs["movement_handoff"].read_text()) == request["movement_handoff"]
        kwargs["job_dir"].mkdir()
        (kwargs["job_dir"] / "native_g1_provider_bundle.v1.json").write_text("{}")
        return {"bundle_sha256": "sha256:" + "1" * 64}
    monkeypatch.setattr(
        "blueprint_pipeline.native_g1_team_campaign_preparation.build_g1_provider_bundle", build,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.native_g1_team_campaign_preparation.load_verified_g1_provider_bundle",
        lambda *_args, **_kwargs: {"bundle_sha256": "sha256:" + "1" * 64},
    )
    args = dict(
        intent_path=intent, registry_path=registry, work_root=tmp_path / "work",
        implementation_commit=COMMIT,
    )
    prepared = prepare_g1_team_campaign(**args)
    assert prepared["status"] == "bundle_prepared_not_executed"
    assert prepared["provider_mutation_performed"] is False
    assert prepared["preparation_digest"] == digest(prepared, digest_field="preparation_digest")
    assert calls[0]["manipulation_packet"] == Path(binding["manipulation_packet_dir"])
    assert calls[0]["runtime_source_receipt"] == Path(binding["runtime_source_receipt_path"])
    assert prepare_g1_team_campaign(**args) == prepared
    assert len(calls) == 1


def test_preparation_rejects_changed_registry_before_bundle(tmp_path, monkeypatch):
    registry, _, intent, _ = _accepted(tmp_path, monkeypatch)
    value = json.loads(registry.read_text())
    value["bindings"][0]["task_id"] = "changed"
    value["registry_digest"] = digest(value, digest_field="registry_digest")
    registry.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(
        "blueprint_pipeline.native_g1_team_campaign_preparation.build_g1_provider_bundle",
        lambda **_: pytest.fail("bundle must not be built"),
    )
    with pytest.raises(ValueError, match="registry_changed"):
        prepare_g1_team_campaign(
            intent_path=intent, registry_path=registry, work_root=tmp_path / "work",
            implementation_commit=COMMIT,
        )
