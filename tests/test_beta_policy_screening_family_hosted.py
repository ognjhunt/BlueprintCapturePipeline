"""Native family policy delivery preserves private origins and inclusive charges."""
from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline.beta_policy_screening_hosted import request_limit, validate_hosted_delivery
from tests.test_task_family_policy_execution import _scene, _spec


def _hosted(tmp_path):
    spec = _spec(_scene())
    credential = tmp_path / "native-policy-token"
    credential.write_text("hermetic-private-native-policy-token")
    credential.chmod(0o600)
    settings = {"credential_file": str(credential), "credential_origin": "https://policy.example",
        "provider_hard_cap_usd": "4.00", "hosted_inference_maximum_usd": "1.00",
        "hosted_inference_payment": "blueprint", "hosted_max_cost_per_request_usd": "0.01",
        "hosted_http_request_limit": request_limit(spec, 10)}
    return spec, settings


def test_native_family_hosted_delivery_uses_exact_native_spec_and_bounded_http(tmp_path, monkeypatch):
    from blueprint_pipeline.droid_https_policy_client import DroidHttpsPolicySpec

    # The native schema must never be coerced into the DROID action/robot shape.
    monkeypatch.setattr(DroidHttpsPolicySpec, "validate", lambda _self: pytest.fail("DROID coercion"))
    spec, settings = _hosted(tmp_path)
    validate_hosted_delivery(spec, settings, episodes=10)
    assert settings["hosted_http_request_limit"] == 80


@pytest.mark.parametrize("change,reason", [
    ("origin", "origin_or_identity_mismatch"), ("checkpoint", "origin_or_identity_mismatch"),
    ("horizon", "origin_or_identity_mismatch"), ("backend", "interface_not_supported"),
    ("public_secret", "credential_not_private"), ("query", "query_parameters_forbidden"),
    ("request_count", "request_bound_mismatch"), ("charges", "cost_below_request_bound"),
    ("unknown_payment", "payment_unproven"),
])
def test_native_delivery_refuses_unbound_origin_identity_credentials_and_costs(tmp_path, change, reason):
    spec, settings = _hosted(tmp_path)
    if change == "origin":
        settings["credential_origin"] = "https://different.example"
    elif change == "checkpoint":
        spec["checkpoint_digest"] = "sha256:" + "f" * 64
    elif change == "horizon":
        spec["open_loop_horizon"] = 2
    elif change == "backend":
        spec["evaluation_backend"] = "robolab.v1"
    elif change == "public_secret":
        Path(settings["credential_file"]).chmod(0o644)
    elif change == "query":
        spec["policy_spec"]["endpoint_url"] += "?key=do-not-retain"
    elif change == "request_count":
        settings["hosted_http_request_limit"] -= 1
    elif change == "charges":
        settings["hosted_inference_maximum_usd"] = "0.01"
    else:
        settings["hosted_inference_payment"] = "unproven"
    with pytest.raises(ValueError, match="beta_screening_hosted_" + reason):
        validate_hosted_delivery(spec, settings, episodes=10)


def test_native_delivery_rejects_invalid_native_action_schema_before_credentials(tmp_path):
    spec, settings = _hosted(tmp_path)
    spec["policy_spec"]["action_bounds"] = [[True, 1]]
    Path(settings["credential_file"]).unlink()
    with pytest.raises(ValueError, match="task_family_policy_action_contract_invalid"):
        validate_hosted_delivery(spec, settings, episodes=10)
