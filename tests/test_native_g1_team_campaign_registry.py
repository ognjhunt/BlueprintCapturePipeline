"""Operator registry creation must bind real packets to authenticated teams."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest as digest
from blueprint_pipeline.native_g1_team_campaign_intake import list_g1_team_campaign_setups
from blueprint_pipeline.native_g1_team_campaign_registry import build_g1_team_campaign_registry
from tests.test_native_g1_team_campaign_intake import _registry
from tests.test_native_g1_team_campaign_request import OWNER, _request


def test_builds_catalog_readable_by_the_authenticated_owner(tmp_path, monkeypatch):
    setup, request = _request(tmp_path, monkeypatch)
    source, binding = _registry(tmp_path, request)
    monkeypatch.setattr(
        "blueprint_pipeline.native_g1_team_campaign_registry.make_packet_planning_setup",
        lambda **_: setup,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.native_g1_team_campaign_intake.make_packet_planning_setup",
        lambda **_: setup,
    )
    registry = build_g1_team_campaign_registry([binding])
    assert registry["registry_digest"] == digest(registry, digest_field="registry_digest")
    source.write_text(json.dumps(registry), encoding="utf-8")
    assert list_g1_team_campaign_setups(registry_path=source, owner=OWNER)["setups"] == [setup]


@pytest.mark.parametrize("change, error", [
    ("owner", "owner_invalid"),
    ("duplicate", "binding_unavailable"),
    ("packet", "binding_packet_mismatch"),
    ("missing_path", "binding_path_invalid"),
])
def test_rejects_unsafe_or_mismatched_registry(tmp_path, monkeypatch, change, error):
    setup, request = _request(tmp_path, monkeypatch)
    _, binding = _registry(tmp_path, request)
    monkeypatch.setattr(
        "blueprint_pipeline.native_g1_team_campaign_registry.make_packet_planning_setup",
        lambda **_: setup,
    )
    binding = deepcopy(binding)
    if change == "owner":
        binding["owner"] = {"user_id": ""}
    elif change == "packet":
        binding["task_id"] = "other-task"
    elif change == "missing_path":
        binding["movement_packet_dir"] = str(tmp_path / "missing")
    bindings = [binding, deepcopy(binding)] if change == "duplicate" else [binding]
    with pytest.raises(ValueError, match=error):
        build_g1_team_campaign_registry(bindings)
