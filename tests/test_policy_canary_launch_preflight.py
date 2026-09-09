"""A canary dry run must expose the same deterministic refusal as execute."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from blueprint_pipeline import policy_canary_allocator_lane as lane


@pytest.mark.parametrize("execute", [False, True])
@pytest.mark.parametrize("fault", ["watchdog_name", "billing", "runtime_window"])
def test_launch_refusal_precedes_consumption_staging_and_transport(tmp_path, monkeypatch, execute, fault):
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.delenv(lane.SPEND_ADMISSION_LOCK_PATH_ENV, raising=False)
    authority_path = tmp_path / "authority.json"
    bundle_path = tmp_path / "bundle.json"
    authority_path.write_text("{}")
    bundle_path.write_text("{}")
    ttl = 600 if fault == "runtime_window" else 1800
    authority = {"hard_cap_usd": 3.78, "hard_ttl_seconds": ttl,
        "authority_digest": "sha256:" + "a" * 64,
        "resource_name": "blueprint-native-task-policy-canary-" + ("short" if fault == "watchdog_name" else "1" * 32)}
    bundle = {"container_image": "nvcr.io/nvidia/isaac-sim@sha256:" + "2" * 64,
        "bundle_sha256": "sha256:" + "3" * 64, "runtime_inputs_digest": "sha256:" + "4" * 64}
    monkeypatch.setattr(lane, "validate_session_authority", lambda _: authority)
    monkeypatch.setattr(lane, "validate_provider_bundle", lambda *a, **kw: bundle)
    monkeypatch.setattr(lane, "_vast_credential_file_present", lambda: True)
    if fault != "billing":
        monkeypatch.setattr(lane, "require_pre_spend_preflight", lambda **kw: {"status": "PASS"})
    monkeypatch.setattr(lane, "run_native_task_arena_policy_canary_session_vast",
        lambda **kw: pytest.fail("Refused preflight reached provider transport"))
    args = SimpleNamespace(provider="vast", execute=execute, adp_job_dir=str(tmp_path / "job"),
        native_task_arena_policy_canary_session_authority=str(authority_path),
        native_task_arena_policy_canary_session_bundle_receipt=str(bundle_path),
        adp_max_spend_usd=3.78, adp_hard_ttl_seconds=ttl, adp_max_hourly_rate_usd=.8,
        admission_out=str(tmp_path / "admission.json"), adapter_output=str(tmp_path / "result.json"))
    assert lane.run_policy_canary_allocator_lane(args, ([], {"orchestrator_source_commit": "5" * 40})) == 2
    result = json.loads((tmp_path / "result.json").read_text())
    expected = {"watchdog_name": "independent_vast_watchdog_exact_resource_name_invalid",
        "runtime_window": "adp_arena_cumulative_budget_below_minimum_live_window",
        "billing": "spend_admission:spend_admission_lock_schema_invalid"}[fault]
    assert expected in result["blockers"]
    assert result["provider_mutations_performed"] == 0
    assert not (tmp_path / "job/policy_canary_session_consumption.json").exists()
    assert not (tmp_path / "job/attempts").exists()


def test_dry_preflight_collects_independent_refusals_in_one_pass(tmp_path, monkeypatch):
    monkeypatch.setenv("INVOCATION_ID", "test-service")
    monkeypatch.delenv("BLUEPRINT_VAST_WATCHDOG_CALLER_EXIT_SURVIVAL", raising=False)
    monkeypatch.delenv(lane.SPEND_ADMISSION_LOCK_PATH_ENV, raising=False)
    monkeypatch.setattr(lane, "_vast_credential_file_present", lambda: False)
    args = SimpleNamespace(adp_job_dir=str(tmp_path), adp_max_spend_usd=3.78,
        adp_hard_ttl_seconds=600, adp_max_hourly_rate_usd=.8)
    blockers = lane._launch_environment_blockers(args,
        {"resource_name": "blueprint-native-task-policy-canary-short"}, {"container_image": "unbound"})
    assert "independent_vast_watchdog_exact_resource_name_invalid" in blockers
    assert "adp_arena_cumulative_budget_below_minimum_live_window" in blockers
    assert "independent_vast_watchdog_caller_exit_survival_unproven" in blockers
    assert any("spend_admission" in code for code in blockers)
