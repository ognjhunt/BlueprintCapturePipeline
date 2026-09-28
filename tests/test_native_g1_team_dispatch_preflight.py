"""Selected dispatch binds bytes, live rights, budget and private credentials."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import native_g1_team_dispatch_preflight as preflight
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest as digest
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


def _synthetic_fetcher(calls, *, after=None):
    from blueprint_pipeline.core.security_controls import BoundedHttpResponse
    from blueprint_pipeline.native_g1_team_policy_conformance import TASK
    def fetcher(url, **options):
        request = json.loads(options["data"])
        assert options["headers"]["Authorization"] == "Bearer " + TOKEN
        calls.append(request)
        if request["kind"] == "infer":
            import base64
            assert request["observation"]["state"] == [0.0] * 64
            assert base64.b64decode(request["observation"]["images"]["front"]["data_b64"]) == bytes(480 * 640 * 3)
            assert request["task"] == TASK
        action = [0.0] * 40
        action[3:9] = [1, 0, 0, 1, 0, 0]
        response = {key: request[key] for key in ("protocol", "profile_digest", "request_id")}
        response.update({"ok": True} if request["kind"] == "reset" else {"action_chunk": [action]})
        if after is not None:
            after()
        return BoundedHttpResponse(body=json.dumps(response).encode(), status=200,
                                   content_type="application/json", final_url=url)
    return fetcher


def test_selected_synthetic_probe_uses_bound_private_token_and_sends_no_site_input(tmp_path, monkeypatch):
    args, _, _, secret = _prepared(tmp_path, monkeypatch)
    plan = preflight.verify_g1_team_dispatch_inputs(**args)
    calls = []
    result = plan.probe_synthetic_endpoint(fetcher=_synthetic_fetcher(calls))
    assert [call["kind"] for call in calls] == ["reset", "infer"]
    assert result["status"] == "synthetic_wire_compatible_before_allocation"
    assert result["selected_input_receipt_digest"] == plan.safe_receipt()["receipt_digest"]
    assert result["synthetic_conformance"]["site_policy_query_count"] == 0
    assert result["synthetic_conformance"]["runtime_identity_verified"] is False
    assert TOKEN not in str(result)
    assert str(secret) not in str(result)
    assert result["receipt_digest"] == digest(result, digest_field="receipt_digest")


def test_selected_synthetic_probe_refuses_changed_secret_before_contact(tmp_path, monkeypatch):
    args, _, _, secret = _prepared(tmp_path, monkeypatch)
    plan = preflight.verify_g1_team_dispatch_inputs(**args)
    secret.write_text("replacement-token")
    with pytest.raises(ValueError, match="synthetic_preflight_failed"):
        plan.probe_synthetic_endpoint(fetcher=lambda *args, **kwargs: pytest.fail("changed token sent"))


def test_selected_synthetic_probe_reopens_authority_after_endpoint_call(tmp_path, monkeypatch):
    args, _, _, _ = _prepared(tmp_path, monkeypatch)
    plan = preflight.verify_g1_team_dispatch_inputs(**args)
    approval = args["authority_arguments"]["approval_path"]
    def revoke():
        if approval.exists():
            approval.unlink()
    with pytest.raises(ValueError, match="synthetic_preflight_failed"):
        plan.probe_synthetic_endpoint(fetcher=_synthetic_fetcher([], after=revoke))


def test_selected_synthetic_probe_redacts_transport_exception_and_cause(tmp_path, monkeypatch):
    args, _, _, _ = _prepared(tmp_path, monkeypatch)
    plan = preflight.verify_g1_team_dispatch_inputs(**args)
    def refuse(*args, **kwargs):
        raise RuntimeError("untrusted HTTP response " + TOKEN)
    with pytest.raises(ValueError, match="synthetic_preflight_failed") as error:
        plan.probe_synthetic_endpoint(fetcher=refuse)
    assert str(error.value) == "g1_team_endpoint_synthetic_preflight_failed"
    assert error.value.__suppress_context__ is True


def test_selected_synthetic_probe_refuses_invalid_actions(tmp_path, monkeypatch):
    from blueprint_pipeline.core.security_controls import BoundedHttpResponse
    args, _, _, _ = _prepared(tmp_path, monkeypatch)
    plan = preflight.verify_g1_team_dispatch_inputs(**args)
    valid = _synthetic_fetcher([])
    def fetcher(url, **options):
        response = valid(url, **options)
        if json.loads(options["data"])["kind"] == "infer":
            value = json.loads(response.body)
            value["action_chunk"] = [[0.0] * 39]
            return BoundedHttpResponse(body=json.dumps(value).encode(), status=200,
                                       content_type="application/json", final_url=url)
        return response
    with pytest.raises(ValueError, match="synthetic_preflight_failed"):
        plan.probe_synthetic_endpoint(fetcher=fetcher)
