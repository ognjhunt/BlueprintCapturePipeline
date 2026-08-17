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


def test_simready_issuer_carries_the_complete_five_attempt_campaign_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scene 840920 starts after five retained SimReady allocations ($0.247)."""

    issuer = _load("issue_simready_isaac_paid_attempt_authority")
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("{}", encoding="utf-8")
    bundle = {
        "bundle_sha256": "sha256:" + "a" * 64,
        "probe_spec_sha256": "sha256:" + "b" * 64,
        "scene_id": "840920",
        "task_id": "task_a_washer_door_open",
        "asset_id": "840920_simready_washer_candidate",
        "validation_mode": "commanded_articulation",
        "candidate_usd_sha256": "sha256:" + "c" * 64,
        "native_probe_manifest_sha256": "sha256:" + "d" * 64,
        "native_probe_manifest_digest": "sha256:" + "e" * 64,
        "predecessor_binding_digest": "sha256:" + "f" * 64,
    }
    monkeypatch.setattr(
        issuer,
        "resolve_host_resident_bundle_receipt",
        lambda _path: {"blockers": [], "receipt": bundle},
    )
    monkeypatch.setattr(
        issuer, "validate_simready_isaac_paid_attempt_authority", lambda *_a, **_k: {}
    )
    prior = [
        {
            "result": {
                "path": f"/retained/simready/{index}/result.json",
                "size_bytes": 1,
                "sha256": "sha256:" + str(index) * 64,
            }
        }
        for index in range(1, 6)
    ]
    reconciliation = {
        "path": "/retained/simready/reconciliation.json",
        "size_bytes": 1,
        "sha256": "sha256:" + "6" * 64,
        "receipt_digest": "sha256:" + "7" * 64,
        "entry_count": 5,
        "total_cost_usd": 0.247,
        "lane": "simready_isaac",
    }
    monkeypatch.setattr(
        issuer,
        "bind_lane_prior_spend",
        lambda **_kwargs: {
            "prior_terminal_attempts": prior,
            "reconciliation": reconciliation,
            "actual_total_usd": 0.247,
        },
    )

    authority = issuer.issue_simready_isaac_paid_attempt_authority(
        bundle_receipt_path=receipt_path,
        authorized_by="local_fixture_audit",
        authority_reference="post_merge_no_spend_executability_audit",
        max_hourly_rate_usd=1.2,
        hard_cap_usd=2.0,
        hard_ttl_seconds=6000,
        prior_result_paths=tuple(tmp_path / f"prior-{index}.json" for index in range(5)),
        prior_spend_reconciliation_path=tmp_path / "reconciliation.json",
    )

    assert len(authority["prior_terminal_attempts"]) == 5
    assert authority["prior_actual_provider_spend_usd"] == 0.247
    assert authority["prior_spend_reconciliation"]["entry_count"] == 5
    assert authority["task_id"] == "task_a_washer_door_open"


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


def test_gaussian_issuer_binds_segment_contribution_sweep_purpose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    issuer = _load("issue_gaussian_excision_paid_attempt_authority")
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        issuer,
        "resolve_host_resident_bundle_receipt",
        lambda _path: {
            "blockers": [],
            "receipt": {
                "bundle_sha256": "sha256:" + "a" * 64,
                "execution_purpose": "released_code_segment_contribution_sweep",
            },
        },
    )
    monkeypatch.setattr(
        issuer,
        "bind_lane_prior_spend",
        lambda **_kwargs: {
            "prior_terminal_attempts": [],
            "reconciliation": None,
            "actual_total_usd": 0.0,
        },
    )
    captured: dict[str, object] = {}

    def capture_validate(value, *_args, **_kwargs):
        captured.update(value)
        return value

    monkeypatch.setattr(
        issuer, "validate_gaussian_excision_paid_attempt_authority", capture_validate
    )

    authority = issuer.issue_gaussian_excision_paid_attempt_authority(
        bundle_receipt_path=receipt_path,
        authorized_by="user",
        authority_reference="goal",
    )

    assert authority["purpose"] == "released_code_segment_contribution_sweep"
    assert captured["purpose"] == "released_code_segment_contribution_sweep"


def test_gaussian_first_attempt_carries_cross_freeze_same_lane_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    issuer = _load("issue_gaussian_excision_paid_attempt_authority")
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("{}", encoding="utf-8")
    task_a_result = tmp_path / "task-a-result.json"
    task_a_result.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        issuer,
        "resolve_host_resident_bundle_receipt",
        lambda _path: {
            "blockers": [],
            "receipt": {
                "bundle_sha256": "sha256:" + "a" * 64,
                "execution_purpose": "released_code_segment_contribution_sweep",
            },
        },
    )
    binding = _fake_binding("gaussian_excision")
    seen: dict[str, object] = {}

    def bind(**kwargs):
        seen.update(kwargs)
        return binding

    monkeypatch.setattr(issuer, "bind_lane_prior_spend", bind)
    captured: dict[str, object] = {}

    def capture_validate(value, *_args, **_kwargs):
        captured.update(value)
        return value

    monkeypatch.setattr(
        issuer, "validate_gaussian_excision_paid_attempt_authority", capture_validate
    )

    authority = issuer.issue_gaussian_excision_paid_attempt_authority(
        bundle_receipt_path=receipt_path,
        authorized_by="user",
        authority_reference="task-b",
        prior_spend_result_paths=(task_a_result,),
        prior_spend_reconciliation_path=tmp_path / "reconciliation.json",
    )

    assert seen["prior_result_paths"] == (task_a_result,)
    assert authority["paid_attempt_ordinal"] == 1
    assert "previous_attempt_receipt_digest" not in authority
    assert authority["prior_actual_provider_spend_usd"] == 0.025
    assert captured["prior_spend_reconciliation"] == binding["reconciliation"]


def _issue_gaussian_with_prior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prior_doc: dict[str, object]
) -> dict[str, object]:
    """The 2026-08-15 chain 0813 -> 22:30 -> next needed ordinal 3; the issuer
    hardcoded `2` for any prior, so a third corrective attempt could never be
    authorized (`previous_attempt_ordinal_mismatch`)."""

    issuer = _load("issue_gaussian_excision_paid_attempt_authority")
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("{}", encoding="utf-8")
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps(prior_doc), encoding="utf-8")
    monkeypatch.setattr(
        issuer,
        "resolve_host_resident_bundle_receipt",
        lambda _path: {"blockers": [], "receipt": {"bundle_sha256": "sha256:" + "a" * 64}},
    )
    captured: dict[str, object] = {}

    def capture_validate(value, *_a, **_k):
        captured.update(value)
        return value

    monkeypatch.setattr(
        issuer, "validate_gaussian_excision_paid_attempt_authority", capture_validate
    )
    monkeypatch.setattr(
        issuer, "bind_lane_prior_spend", lambda **_kwargs: _fake_binding("gaussian_excision")
    )
    issuer.issue_gaussian_excision_paid_attempt_authority(
        bundle_receipt_path=receipt_path,
        authorized_by="user",
        authority_reference="goal",
        prior_attempt_receipt_path=prior,
        prior_spend_reconciliation_path=tmp_path / "reconciliation.json",
    )
    return captured


def test_gaussian_issuer_extends_ordinal_past_second_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _issue_gaussian_with_prior(
        tmp_path,
        monkeypatch,
        {"receipt_digest": "sha256:" + "7" * 64, "paid_attempt_ordinal": 2},
    )
    assert authority["paid_attempt_ordinal"] == 3


def test_gaussian_issuer_pins_legacy_prior_to_second_ordinal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _issue_gaussian_with_prior(
        tmp_path, monkeypatch, {"receipt_digest": "sha256:" + "7" * 64}
    )
    assert authority["paid_attempt_ordinal"] == 2


def test_gaussian_issuer_refuses_invalid_prior_ordinal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="previous_attempt_ordinal_invalid"):
        _issue_gaussian_with_prior(
            tmp_path,
            monkeypatch,
            {"receipt_digest": "sha256:" + "7" * 64, "paid_attempt_ordinal": 0},
        )
