"""Selected dispatch binds bytes, live rights, budget and private credentials."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import native_g1_team_dispatch_preflight as preflight
from blueprint_pipeline.native_g1_team_provider_bundle import build_g1_team_provider_bundle
from tests.test_native_g1_team_policy_credentials import TOKEN, _binding
from tests.test_native_g1_team_provider_bundle import COMMIT, _inputs


def _prepared(tmp_path, monkeypatch):
    # Fixtures use real signed-intent, profile and operator authority validators.
    credential_args, registry, secret = _binding(tmp_path, monkeypatch)
    args = {
        "job_dir": tmp_path / "bundle", "execution_packet_path": tmp_path / "execution/native_g1_team_policy_execution_packet.v1.json",
        "authority_arguments": credential_args["authority_arguments"],
        "scene_packet_root": tmp_path / "scene", "publisher_source": tmp_path / "publisher",
        "runtime_source_receipt": tmp_path / "runtime.json", "sonic_asset_dir": tmp_path / "sonic",
        "expected_implementation_commit": COMMIT,
    }
    bundle = build_g1_team_provider_bundle(**args)
    request = json.loads(args["execution_packet_path"].read_text())["request"]
    return {
        "bundle_receipt_path": Path(bundle["receipt_path"]),
        "authority_arguments": args["authority_arguments"],
        "expected_implementation_commit": COMMIT,
        "credential_registry_path": credential_args["registry_path"],
        "max_hourly_rate_usd": 1.0,
        "hard_cap_usd": request["authorization"]["maximum_cost_usd"],
        "hard_ttl_seconds": request["authorization"]["hard_ttl_seconds"],
    }, bundle, registry, secret


def test_endpoint_inputs_bound_but_not_paid_admitted(tmp_path, monkeypatch):
    args, bundle, _, secret = _prepared(tmp_path, monkeypatch)
    plan = preflight.verify_g1_team_dispatch_inputs(**args)
    receipt = plan.safe_receipt()
    assert receipt["status"] == "verified_inputs_not_spend_admitted"
    assert receipt["bundle_sha256"] == bundle["bundle_sha256"]
    assert receipt["canonical_allocator_required"] is True
    assert receipt["provider_mutation_performed"] is False
    assert plan.runtime_secret_file_paths() == {"BLUEPRINT_G1_TEAM_CREDENTIAL_FILE": secret}
    assert TOKEN not in str(receipt) + repr(plan)
    assert str(secret) not in str(receipt) + repr(plan)
    assert plan.recheck() == receipt


@pytest.mark.parametrize("field,value", [
    ("hard_cap_usd", 13), ("hard_cap_usd", True), ("hard_cap_usd", float("nan")),
    ("hard_ttl_seconds", 14401), ("hard_ttl_seconds", True),
    ("max_hourly_rate_usd", 0), ("max_hourly_rate_usd", float("inf")),
])
def test_invalid_or_over_authorized_budgets_refuse_before_resolution(tmp_path, monkeypatch, field, value):
    args, _, _, _ = _prepared(tmp_path, monkeypatch)
    monkeypatch.setattr(preflight, "resolve_g1_team_policy_credential", lambda **kwargs: pytest.fail("secret resolved before budget validation"))
    with pytest.raises(ValueError, match="budget_invalid"):
        preflight.verify_g1_team_dispatch_inputs(**{**args, field: value})


def test_budget_cannot_exceed_signed_request_even_below_global_cap(tmp_path, monkeypatch):
    args, _, _, _ = _prepared(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="budget_invalid"):
        preflight.verify_g1_team_dispatch_inputs(**{**args, "hard_cap_usd": args["hard_cap_usd"] + 0.01})
    with pytest.raises(ValueError, match="budget_invalid"):
        preflight.verify_g1_team_dispatch_inputs(**{**args, "hard_ttl_seconds": args["hard_ttl_seconds"] + 1})


def test_mode_capability_gap_refuses_before_credential_or_paid_staging(tmp_path, monkeypatch):
    args, _ = _inputs(tmp_path, monkeypatch)
    bundle = build_g1_team_provider_bundle(**args)
    monkeypatch.setattr(preflight, "resolve_g1_team_policy_credential", lambda **kwargs: pytest.fail("container secret resolution"))
    with pytest.raises(ValueError, match="paired_policy_runtime_required"):
        preflight.verify_g1_team_dispatch_inputs(
            bundle_receipt_path=Path(bundle["receipt_path"]),
            authority_arguments=args["authority_arguments"], expected_implementation_commit=COMMIT,
            credential_registry_path=None, max_hourly_rate_usd=1, hard_cap_usd=1,
            hard_ttl_seconds=1800,
        )


def test_mutation_recheck_rejects_changed_bundle_or_credential(tmp_path, monkeypatch):
    args, bundle, _, secret = _prepared(tmp_path, monkeypatch)
    plan = preflight.verify_g1_team_dispatch_inputs(**args)
    secret.write_text("replacement-token")
    with pytest.raises(ValueError, match="changed"):
        plan.recheck()
    secret.write_text(TOKEN + "\n")
    plan = preflight.verify_g1_team_dispatch_inputs(**args)
    with Path(bundle["bundle_path"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="bundle_bytes_invalid"):
        plan.recheck()
