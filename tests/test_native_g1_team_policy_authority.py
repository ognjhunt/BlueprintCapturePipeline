"""Reopened team approval must still bind the exact immutable request."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_g1_team_policy_authority import (
    verify_g1_team_policy_authority,
)
from blueprint_pipeline.native_g1_team_policy_run_intake import stage_g1_team_policy_run
from tests.test_native_g1_team_policy_approval import _approval
from tests.test_native_g1_team_policy_run_intake import NOW, _setup


def _authority(tmp_path: Path, monkeypatch):
    setup, request, registry, _ = _setup(tmp_path, monkeypatch)
    receipt = stage_g1_team_policy_run(
        value=request, registry_path=registry, queue_root=tmp_path / "queue",
        authenticated_client="blueprint-webapp", trusted_clients={"blueprint-webapp"},
        now_epoch=NOW,
    )
    profile = request["policy_profile"]
    approval = _approval(setup, profile, {
        "mode": "container", "profile_digest": profile["profile_digest"],
        "image_ref": profile["delivery"]["image_ref"], "gpu_device": 0,
    })
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(json.dumps(approval), encoding="utf-8")
    monkeypatch.setattr(
        "blueprint_pipeline.native_g1_team_policy_authority.make_packet_planning_setup",
        lambda *, source_packet_dir: setup,
    )
    args = {
        "intent_path": tmp_path / "queue" / receipt["intent_id"] / "intent.json",
        "registry_path": registry,
        "approval_path": approval_path,
        "trusted_clients": {"blueprint-webapp"},
        "now_epoch": NOW,
    }
    return args, approval


def test_reopens_exact_operator_approval_before_dispatch(tmp_path: Path, monkeypatch) -> None:
    args, approval = _authority(tmp_path, monkeypatch)
    verified = verify_g1_team_policy_authority(**args)
    assert verified["operator_approval"] == approval
    assert verified["intent"]["provider_mutation_performed"] is False
    assert verified["trusted_setup"]["scene_id"] == verified["registry_binding"]["scene_id"]


def test_rejects_changed_registry_expiry_and_delivery_approval(
    tmp_path: Path, monkeypatch
) -> None:
    args, approval = _authority(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="registry_changed"):
        changed = json.loads(args["registry_path"].read_text())
        changed["bindings"] = []
        changed["registry_digest"] = canonical_digest(changed, digest_field="registry_digest")
        args["registry_path"].write_text(json.dumps(changed), encoding="utf-8")
        verify_g1_team_policy_authority(**args)

    second = tmp_path / "second"
    second.mkdir()
    args, approval = _authority(second, monkeypatch)
    with pytest.raises(ValueError, match="authorization_invalid"):
        verify_g1_team_policy_authority(**{**args, "now_epoch": NOW + 3600})
    changed_approval = {
        **approval,
        "runtime_binding": {**approval["runtime_binding"], "image_ref":
                            "registry.example.org/team/g1@sha256:" + "f" * 64},
    }
    changed_approval["approval_digest"] = canonical_digest(
        changed_approval, digest_field="approval_digest"
    )
    args["approval_path"].write_text(json.dumps(changed_approval), encoding="utf-8")
    with pytest.raises(ValueError, match="approval_binding_invalid"):
        verify_g1_team_policy_authority(**args)
    with pytest.raises(ValueError, match="intent_invalid"):
        verify_g1_team_policy_authority(**{**args, "trusted_clients": set()})
