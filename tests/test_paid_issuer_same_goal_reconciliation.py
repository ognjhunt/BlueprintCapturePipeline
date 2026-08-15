"""Every retry-capable paid issuer must carry the shared billing proof."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fake_binding(lane: str) -> dict[str, object]:
    return {
        "prior_terminal_attempts": [
            {
                "result_path": "/retained/result.json",
                "result_sha256": "sha256:" + "1" * 64,
                "receipt_digest": "sha256:" + "2" * 64,
                "result": {
                    "path": "/retained/result.json",
                    "size_bytes": 1,
                    "sha256": "sha256:" + "1" * 64,
                },
                "estimated_cost_usd": 0.9,
                "actual_provider_charge_usd": 0.025,
                "provider_instance_id": 123,
                "reconciliation_entry_digest": "sha256:" + "3" * 64,
            }
        ],
        "reconciliation": {
            "path": "/retained/reconciliation.json",
            "size_bytes": 1,
            "sha256": "sha256:" + "4" * 64,
            "receipt_digest": "sha256:" + "5" * 64,
            "entry_count": 1,
            "total_cost_usd": 0.025,
            "lane": lane,
        },
        "actual_total_usd": 0.025,
    }


def test_content_agents_issuer_carries_shared_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    issuer = _load("issue_content_agents_paid_attempt_authority")
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("{}", encoding="utf-8")
    receipt_sha = "sha256:" + hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps({"bundle_receipt_sha256": receipt_sha}), encoding="utf-8")
    monkeypatch.setattr(
        issuer,
        "resolve_host_resident_bundle_receipt",
        lambda _path: {"blockers": [], "receipt": {"bundle_sha256": "sha256:" + "a" * 64}},
    )
    monkeypatch.setattr(issuer, "validate_content_agents_paid_attempt_authority", lambda *_a, **_k: {})
    binding = _fake_binding("content_agents")
    seen: dict[str, object] = {}

    def bind(**kwargs):
        seen.update(kwargs)
        return binding

    monkeypatch.setattr(issuer, "bind_lane_prior_spend", bind)
    authority = issuer.issue_content_agents_paid_attempt_authority(
        bundle_receipt_path=receipt_path,
        config_preflight_path=preflight,
        authorized_by="user",
        authority_reference="goal",
        max_hourly_rate_usd=1.0,
        hard_cap_usd=1.0,
        hard_ttl_seconds=3600,
        prior_result_paths=(tmp_path / "prior.json",),
        prior_spend_reconciliation_path=tmp_path / "reconciliation.json",
    )
    assert seen["lane"] == "content_agents"
    assert authority["prior_actual_provider_spend_usd"] == 0.025
    assert authority["prior_spend_reconciliation"] == binding["reconciliation"]


def test_simready_issuer_refuses_nonzero_prior_without_shared_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    issuer = _load("issue_simready_isaac_paid_attempt_authority")
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        issuer,
        "resolve_host_resident_bundle_receipt",
        lambda _path: {"blockers": [], "receipt": {"bundle_sha256": "sha256:" + "a" * 64}},
    )
    prior = tmp_path / "prior.json"
    prior.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="prior_spend_reconciliation_required"):
        issuer.issue_simready_isaac_paid_attempt_authority(
            bundle_receipt_path=receipt_path,
            authorized_by="user",
            authority_reference="goal",
            max_hourly_rate_usd=1.0,
            hard_cap_usd=1.0,
            hard_ttl_seconds=3600,
            prior_result_paths=(prior,),
        )


def test_gaussian_issuer_cannot_omit_or_tamper_shared_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    issuer = _load("issue_gaussian_excision_paid_attempt_authority")
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("{}", encoding="utf-8")
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"receipt_digest": "sha256:" + "7" * 64}), encoding="utf-8")
    monkeypatch.setattr(
        issuer,
        "resolve_host_resident_bundle_receipt",
        lambda _path: {"blockers": [], "receipt": {"bundle_sha256": "sha256:" + "a" * 64}},
    )
    monkeypatch.setattr(issuer, "validate_gaussian_excision_paid_attempt_authority", lambda *_a, **_k: {})
    monkeypatch.setattr(
        issuer,
        "bind_lane_prior_spend",
        lambda **_kwargs: (_ for _ in ()).throw(
            ValueError("same_goal_spend_binding_invalid")
        ),
    )
    with pytest.raises(ValueError, match="same_goal_spend_binding_invalid"):
        issuer.issue_gaussian_excision_paid_attempt_authority(
            bundle_receipt_path=receipt_path,
            authorized_by="user",
            authority_reference="goal",
            prior_attempt_receipt_path=prior,
            prior_spend_reconciliation_path=tmp_path / "reconciliation.json",
        )
