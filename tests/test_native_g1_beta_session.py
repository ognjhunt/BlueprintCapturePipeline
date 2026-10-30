from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from blueprint_pipeline import beta_policy_screening_g1 as g1
from blueprint_pipeline import native_g1_beta_session as session
from blueprint_pipeline.beta_policy_candidate_catalog import load_default_catalog, seal_catalog
from blueprint_pipeline.beta_policy_screening_plan import build_screening_plan
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_beta_policy_screening_g1 import builder_inputs, DP
from tests.test_beta_policy_screening_plan import admitted, scene


def inputs(tmp_path, monkeypatch):
    kwargs, *_ = builder_inputs(tmp_path, monkeypatch)
    spec = g1.candidate_execution_spec(**kwargs)
    catalog = load_default_catalog()
    configured = next(row for row in catalog["candidates"] if row["candidate_id"] == DP)
    catalog = seal_catalog({**catalog, "candidates": [configured]})
    binding, quote = admitted(configured, maximum="4.00")
    execution = binding["execution_spec"]
    execution.update(backend=g1.BACKEND, hard_ttl_seconds=1800, runtime_digest=spec["runtime_identity_digest"],
        task_success_contract_digest=spec["task_success_contract_digest"], minimum_gpu_memory_gb=48)
    binding.update(backend=g1.BACKEND, runtime_digest=spec["runtime_identity_digest"], admission_digest=canonical_digest(execution))
    quote["valid_until"] = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    quote["quote_digest"] = canonical_digest(quote, digest_field="quote_digest")
    current = scene()
    current.update(**spec["source"], site_id=spec["site_id"], scene_id=spec["site_id"],
                   task_success_contract_digest=spec["task_success_contract_digest"])
    plan = build_screening_plan(catalog, current, campaign_id="native-g1-beta-test", now=datetime.now(timezone.utc),
        capabilities={"candidate_bindings": {DP: binding}}, cost_quotes={DP: quote})
    candidate = plan["candidates"][0]
    return plan, candidate, spec


def test_session_is_exact_selected_candidate_budget_and_one_shot(tmp_path, monkeypatch):
    plan, candidate, spec = inputs(tmp_path, monkeypatch)
    authority = session.build_session_authority(plan=plan, candidate=candidate, spec=spec)
    assert authority["candidate_ids"] == [DP]
    assert len(authority["cells"]) == 10
    assert authority["resource_name"] == session.RESOURCE_PREFIX + authority["run_id"]
    assert len(authority["run_id"]) == 32
    receipt = session.consume_session_authority_once(authority, consumption_path=tmp_path / "consumed.json", now=100)
    assert receipt["hard_deadline_epoch"] == 100 + authority["hard_ttl_seconds"]
    again = session.consume_session_authority_once(authority, consumption_path=tmp_path / "consumed.json", now=200)
    assert again["status"] == "already_consumed" and again["maximum_provider_allocations"] == 0


@pytest.mark.parametrize("field,value", [("hard_cap_usd", 16), ("hard_ttl_seconds", 14401),
    ("maximum_provider_allocations", 2), ("retry_cap", 1), ("public_media_publication_authorized", True),
    ("automatic_retry_authorized", True), ("resource_name", "arbitrary-resource")])
def test_resealed_budget_scope_or_name_escalation_is_rejected(tmp_path, monkeypatch, field, value):
    plan, candidate, spec = inputs(tmp_path, monkeypatch)
    row = deepcopy(session.build_session_authority(plan=plan, candidate=candidate, spec=spec))
    row[field] = value
    row["authority_digest"] = canonical_digest(row, digest_field="authority_digest")
    with pytest.raises(ValueError):
        session.validate_session_authority(row)


def test_transfer_quote_accounts_selected_exact_models_and_output():
    from blueprint_pipeline.native_g1_beta_assets import ORDER
    dp = session.transfer_quote(candidate_ids=[ORDER[0]], runtime_packet_bytes=200, bundle_bytes=100)
    pi = session.transfer_quote(candidate_ids=[ORDER[1]], runtime_packet_bytes=200, bundle_bytes=100)
    assert dp["expected_provider_download_bytes"] == 300 + dp["selected_model_download_bytes"] + dp["setup_download_allowance_bytes"] + dp["external_dependency_download_bytes"]
    assert pi["selected_model_download_bytes"] > dp["selected_model_download_bytes"]
    assert dp["expected_provider_upload_bytes"] == session.MAX_OUTPUT_BYTES


def test_generated_authority_uses_real_parent_child_provider_and_fleet_contracts(tmp_path, monkeypatch):
    from tests.test_native_g1_beta_watchdog import exercise_native_g1_watchdog
    plan, candidate, spec = inputs(tmp_path, monkeypatch)
    authority = session.build_session_authority(plan=plan, candidate=candidate, spec=spec)
    exercise_native_g1_watchdog(tmp_path, monkeypatch, authority["resource_name"])
