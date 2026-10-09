"""Synthetic discovery/admission tests; no live provider or sink credentials."""
import json
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from tests.test_daily_research_knowledge import enable_v3, policy_bundle, v3
from tests.test_daily_research_runner import DAY, NOW, FakeAPI
from tests.test_daily_research_runner import fixture as runner_fixture
from tools.daily_research import adaptive, discovery, render
from tools.daily_research import consumer as consumer_module
from tools.daily_research import runner as runner_module
from tools.daily_research.runner import (
    Refusal,
    canonical,
    configuration,
    digest,
    keys,
    phase_runtime_seconds,
    prompt,
    validate_output,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture(tmp_path):
    yield from runner_fixture.__wrapped__(tmp_path)


def coverage(count=10):
    return {"search_queries": 22, "pages_opened": 17, "branches_checked": ["operator task", "incumbent automation"],
            "rejection_reasons": ["Already deployed: learning contact only"], "stop_reason": "Evidence scope covered",
            "shortfall_reason": None if count >= 10 else "Only the supported subset survived source and duplicate checks"}


def result(count=10):
    _, _, policy, ctx = policy_bundle()
    out = v3(ctx)
    example = out["candidates"][0]
    out["candidates"] = []
    for number in range(count):
        candidate = deepcopy(example)
        candidate.update(organization=f"Synthetic operator {number}", organization_url=f"https://operator{number}.example/",
                         site=f"Synthetic site {number}", task=f"Synthetic task {number}")
        for evidence in candidate["evidence"]:
            if evidence["classification"] == "operator":
                evidence["url"] = f"https://operator{number}.example/tasks"
        out["candidates"].append(candidate)
    out["coverage"] = coverage(count)
    return out, ctx, policy


def test_ten_v3_rows_pass_schema_and_exact_dedupe_counts_new_only():
    out, ctx, policy = result()
    schema = json.loads((ROOT / "tools/daily_research/daily-research.v3.schema.json").read_text())
    Draft202012Validator(schema).validate(out)
    accepted, duplicates = validate_output(out, DAY, keys(out["candidates"][0]), contract_version=3,
                                          knowledge_context=ctx, refresh_policy=policy, observed_at=NOW)
    assert len(accepted) == 9 and len(duplicates) == 1
    assert all(c["qualification_status"] == "unqualified" for c in accepted)
    legacy = deepcopy(out)
    legacy.pop("coverage")
    legacy.pop("refresh_policy_hash")
    legacy["schema_version"] = "blueprint.daily-research.v2"
    with pytest.raises(Refusal, match="output_date_or_count_invalid"):
        validate_output(legacy, DAY, set(), contract_version=2, knowledge_context=ctx, observed_at=NOW)


def test_shortfall_and_resource_ceiling_are_honest_not_padded():
    out, ctx, policy = result(2)
    discovery.validate_coverage(out["coverage"], 2)
    out["coverage"]["shortfall_reason"] = None
    with pytest.raises(Refusal, match="shortfall_reason_required"):
        validate_output(out, DAY, set(), contract_version=3, knowledge_context=ctx, refresh_policy=policy, observed_at=NOW)
    out, ctx, policy = result(101)
    accepted, _ = validate_output(out, DAY, set(), contract_version=3, knowledge_context=ctx, refresh_policy=policy, observed_at=NOW)
    assert len(accepted) == 101  # Retention is bounded by bytes, not a prospect quota.


def test_adaptive_prompt_uses_skill_paths_and_does_not_inherit_scan_caps():
    _, _, _, ctx = policy_bundle()
    text = prompt(DAY, ctx, 3, adaptive=True)
    assert "defined, evidence-based scope" in text and "references/prospect-contract.md" in text
    assert "at least 10 NEW" not in text and "no minimum or maximum prospect quota" in text
    assert "credible routing contact" in text and "counterevidence" in text
    assert "2025-2026" in text and "never count them as site prospects" in text
    assert "at most two searches" not in text and "up to THREE" not in text and "under 800 words" not in text
    assert "Existing robot deployments are separate" in text and "first" in text
    assert "Unknown buyer interest" in text and "never a pretend read" in text
    assert "Blueprint agents own research QA and publication" in text
    assert "UNTRUSTED DATA" in text and "Snapshot data JSON string:" in text


def test_new_daily_example_is_disabled_without_a_numeric_budget():
    cfg = json.loads((ROOT / "tools/daily_research/adaptive-daily.config.example.json").read_text())
    assert configuration(cfg)["enabled"] is False and cfg["soft_target_usd"] is None
    assert cfg["max_runtime_seconds"] - cfg["qa_reserved_seconds"] == 1200
    assert configuration({**cfg, "soft_target_usd": 25})["soft_target_usd"] == 25
    for bad in (None, "1800", True, []):
        with pytest.raises(Refusal, match="approved_envelope"):
            configuration({**cfg, "max_runtime_seconds": bad})
    with pytest.raises(Refusal, match="phase_envelope"):
        configuration({**cfg, "qa_reserved_seconds": 1800})


def test_owner_approved_adaptive_envelope_is_the_single_runtime_bound():
    cfg = json.loads((ROOT / "tools/daily_research/adaptive-daily.config.example.json").read_text())
    assert runner_module.MAX_ADAPTIVE_RUNTIME_SECONDS == 14400
    approved = configuration({**cfg, "max_runtime_seconds": 3600, "qa_reserved_seconds": 900})
    assert approved["max_runtime_seconds"] - approved["qa_reserved_seconds"] == 2700
    for minutes in (60,120,180,240):
        configured = configuration({**cfg,"max_runtime_seconds":minutes *60,"qa_reserved_seconds":900})
        assert configured["max_runtime_seconds"] ==minutes *60
    # Rows and configs admitted under the earlier 30-minute envelope stay valid.
    assert configuration({**cfg, "max_runtime_seconds": 1800, "qa_reserved_seconds": 600})["max_runtime_seconds"] == 1800
    with pytest.raises(Refusal, match="^approved_envelope_mismatch$"):
        configuration({**cfg, "max_runtime_seconds": 14401, "qa_reserved_seconds": 900})
    with pytest.raises(Refusal, match="^adaptive_phase_envelope_invalid$"):
        configuration({**cfg, "max_runtime_seconds": 3600, "qa_reserved_seconds": 3600})
    # The non-adaptive cap is unchanged.
    legacy = {key: value for key, value in cfg.items() if key not in {"discovery_profile", "qa_reserved_seconds"}}
    assert configuration({**legacy, "max_runtime_seconds": 180})["max_runtime_seconds"] == 180
    for runtime in (181, 1800, 3600):
        with pytest.raises(Refusal, match="^approved_envelope_mismatch$"):
            configuration({**legacy, "max_runtime_seconds": runtime})
    for field, phase in (("research_runtime_seconds", "research"), ("total_runtime_seconds", "qa")):
        assert phase_runtime_seconds({field: 3600}, {}, phase) == 3600
        with pytest.raises(Refusal, match="^pinned_phase_envelope_invalid$"):
            phase_runtime_seconds({field: 14401}, {}, phase)


def test_sixty_minute_row_pins_forty_five_minute_research_and_shared_total(fixture, tmp_path):
    runner, api, ledger = fixture
    enable_v3(runner, tmp_path)
    runner.config.update(discovery_profile="adaptive-sites-v1", max_runtime_seconds=3600, qa_reserved_seconds=900)
    out, _, _ = result()
    api.raw, api.tool_count = canonical(out).encode(), 17
    api.turn_status = "in_progress"
    row = runner.start_or_resume()
    assert row["state"] == "running" and len(api.payloads) == 1
    assert row["total_runtime_seconds"] == 3600 and row["research_runtime_seconds"] == 2700
    assert consumer_module.qa_deadline(row, runner.config) == NOW + timedelta(seconds=3600)
    assert render.observation_seconds(row, {}, "research", NOW) == 2730
    assert render.observation_seconds(row, {}, "qa", NOW + timedelta(seconds=2700)) == 930
    # The old 1800-second envelope no longer cancels a 60-minute row.
    runner.clock = lambda: NOW + timedelta(seconds=2699)
    assert runner.start_or_resume()["state"] == "running" and not api.cancellations
    runner.clock = lambda: NOW + timedelta(seconds=2700)
    assert runner.start_or_resume()["state"] == "cancel_pending"
    assert len(api.cancellations) == 1 and len(api.payloads) == 1
    assert ledger.get(DAY)["research_runtime_seconds"] == 2700


def test_raising_the_envelope_never_extends_an_already_admitted_row(fixture, tmp_path):
    runner, api, ledger = fixture
    enable_v3(runner, tmp_path)
    runner.config.update(discovery_profile="adaptive-sites-v1", max_runtime_seconds=1800, qa_reserved_seconds=600)
    out, _, _ = result()
    api.raw, api.tool_count = canonical(out).encode(), 17
    api.turn_status = "in_progress"
    row = runner.start_or_resume()
    assert (row["total_runtime_seconds"], row["research_runtime_seconds"]) == (1800, 1200)
    # The 3600-second config applies only to rows admitted after the change.
    runner.config.update(max_runtime_seconds=3600, qa_reserved_seconds=900)
    assert consumer_module.qa_deadline(ledger.get(DAY), runner.config) == NOW + timedelta(seconds=1800)
    runner.clock = lambda: NOW + timedelta(seconds=1200)
    assert runner.start_or_resume()["state"] == "cancel_pending"
    assert len(api.cancellations) == 1 and len(api.payloads) == 1
    assert (ledger.get(DAY)["total_runtime_seconds"], ledger.get(DAY)["research_runtime_seconds"]) == (1800, 1200)


def test_adaptive_activity_and_deadline_are_pinned_not_old_scan_caps(fixture, tmp_path):
    runner, api, ledger = fixture
    enable_v3(runner, tmp_path)
    runner.config.update(discovery_profile="adaptive-sites-v1", max_runtime_seconds=1800, qa_reserved_seconds=600)
    out, _, _ = result()
    api.raw, api.tool_count = canonical(out).encode(), 17
    api.turn_status = "in_progress"
    row = runner.start_or_resume()
    assert row["state"] == "running" and row["research_runtime_seconds"] == 1200
    assert len(api.payloads) == 1 and row["total_runtime_seconds"] == 1800
    runner.clock = lambda: NOW + timedelta(seconds=300)
    assert runner.start_or_resume()["state"] == "running" and not api.cancellations
    # A changed process config cannot extend or shorten this admitted row.
    runner.config["max_runtime_seconds"] = 180
    runner.clock = lambda: NOW + timedelta(seconds=1200)
    assert runner.start_or_resume()["state"] == "cancel_pending"
    assert len(api.cancellations) == 1 and len(api.payloads) == 1
    assert ledger.get(DAY)["cleanup_required"] is True


def test_completed_adaptive_scan_retains_ten_rows_and_real_coverage(fixture, tmp_path):
    runner, api, _ = fixture
    enable_v3(runner, tmp_path)
    runner.config.update(discovery_profile="adaptive-sites-v1", max_runtime_seconds=1800, qa_reserved_seconds=600)
    out, _, _ = result()
    api.raw, api.tool_count = canonical(out).encode(), 17
    row = runner.start_or_resume()
    assert row["state"] == "awaiting_review" and len(row["packet"]["candidates"]) == 10
    assert row["packet"]["coverage"]["search_queries"] == 22
    assert row["packet"]["discovery_counts"]["semantic_and_deployment_qa_pending"] is True
    assert row["packet"]["discovery_counts"]["shortfall"] == 0
    assert len(api.payloads) == 1 and not api.cancellations


def test_observer_allows_admitted_phases_and_preserves_legacy_bounds():
    row = {"started_at": NOW.isoformat(), "discovery_profile": "adaptive-sites-v1",
           "research_runtime_seconds": 1200, "total_runtime_seconds": 1800}
    assert render.observation_seconds(row, {}, "research", NOW) == 1230
    assert render.observation_seconds(row, {}, "qa", NOW + timedelta(seconds=1200)) == 630
    assert render.observation_seconds(row, {}, "qa", NOW + timedelta(seconds=1900)) == 30
    assert render.observation_seconds({"started_at": NOW.isoformat()}, {}, "research", NOW) == 300
    assert render.observation_seconds(None, {}, "qa", NOW) == 200


def test_unpinned_legacy_row_is_not_extended_by_adaptive_config(fixture):
    runner, api, ledger = fixture
    api.turn_status = "in_progress"
    row = runner.start_or_resume()
    row.pop("research_runtime_seconds")
    row.pop("total_runtime_seconds")
    ledger.put(row)
    runner.config.update(max_runtime_seconds=1800, qa_reserved_seconds=600)
    assert phase_runtime_seconds(row, runner.config, "research") == 180
    assert phase_runtime_seconds(row, runner.config, "qa") == 180
    runner.clock = lambda: NOW + timedelta(seconds=181)
    assert runner.start_or_resume()["state"] == "cancel_pending"
    assert len(api.cancellations) == 1 and len(api.payloads) == 1


def prepared_inputs(fixture, tmp_path, monkeypatch):
    runner, api, ledger = fixture
    enable_v3(runner, tmp_path)
    row = runner.start_or_resume()
    row = deepcopy(row)
    row.update(date="2026-10-01", state="failed", session_id=adaptive.SESSION, qa=None, delivery={})
    monkeypatch.setattr(adaptive, "DAILY_STATUS_SHA", digest(row))
    monkeypatch.setattr(adaptive, "DAILY_ARTIFACT_SHA", row["raw_output_digest"])
    cfg = json.loads(adaptive.PROFILE.read_text())
    # Identity-only prep never uses CRM contact/email fields.
    crm = deepcopy(row["crm_snapshot"])
    crm["values"].append(["BP-000002", "MealPro", "Facility", "7433 Greenback", "PRIVATE NAME", "PRIVATE EMAIL",
                          "", "", "", "https://meal.example/", "", "", "", "", "Plating"])
    return cfg, row, crm, api, ledger


def test_preparation_has_zero_provider_calls_and_preserves_original_failed_row(fixture, tmp_path, monkeypatch):
    cfg, row, crm, api, ledger = prepared_inputs(fixture, tmp_path, monkeypatch)
    before, calls, durable = deepcopy(row), deepcopy(api.calls), ledger.get(DAY)
    intent = adaptive.prepare(cfg, row, crm, "a" * 40, now=NOW)
    assert intent["state"] == "prepared_disabled" and intent["provider_calls"] == 0
    assert intent["durable_intent_and_claim_required_before_post"] is True
    assert intent["hard_total_cap_verified"] is False and len(intent["admission_blockers"]) == 3
    assert row == before and ledger.get(DAY) == durable and api.calls == calls
    assert "PRIVATE NAME" not in canonical(intent) and "PRIVATE EMAIL" not in canonical(intent)
    assert "BP-000002" in canonical(intent) and intent["event"]["type"] == "agent.session.input.message"
    assert intent["request_digest"] == digest(intent["event"])
    with pytest.raises(Refusal, match="unchanged"):
        adaptive.prepare(cfg, {**row, "cleanup_required": False}, crm, "a" * 40)
    with pytest.raises(Refusal, match="disabled"):
        adaptive.prepare({**cfg, "enabled": True}, row, crm, "a" * 40)


def test_exact_session_receipts_require_idle_usable_standard_and_original_root(fixture, tmp_path, monkeypatch):
    _, row, _, _, _ = prepared_inputs(fixture, tmp_path, monkeypatch)
    session = {"id":row["session_id"], "status":"idle", "metadata":row["metadata"],
               "agent":deepcopy(FakeAPI().agent), "environment":{"id":row["environment_id"], "type":"openai_hosted", "container_size":"small"}}
    session["agent"]["service_tier"] = "default"
    env = {"id":row["environment_id"], "status":"connected"}
    turns = [{"id":row["turn_id"], "status":"completed", "subagent_id":None}]
    assert adaptive.session_blockers(row, session, env, turns) == []
    assert adaptive.session_blockers(row, session, {**env,"status":"expired"}, turns) == ["same_hosted_environment_not_usable_small"]
    assert adaptive.session_blockers(row, {**session,"status":"in_progress"}, env, turns) == ["same_session_not_idle"]
    fast = deepcopy(session)
    fast["agent"]["service_tier"] = "fast"
    assert adaptive.session_blockers(row, fast, env, turns) == ["standard_service_tier_unverified"]
    assert adaptive.session_blockers(row, session, env, turns*2) == ["same_session_initial_turn_scope_mismatch"]


def test_cost_estimate_counts_reasoning_once_and_never_claims_total_ceiling():
    estimate = discovery.estimated_model_cost({"input_tokens":100000, "output_tokens":10000, "reasoning_tokens":9000})
    assert estimate["estimate_usd"] == "0.65" and estimate["hard_total_cap"] is False
    assert "tool_fees" in estimate["excludes"]
    assert discovery.estimated_model_cost(None)["known"] is False
    assert discovery.estimated_model_cost({"input_tokens":True, "output_tokens":1})["known"] is False
