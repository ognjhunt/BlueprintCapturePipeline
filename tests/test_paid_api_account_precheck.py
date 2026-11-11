"""WP-23 / G48: API-priced calls reachable from automatic jobs ask the account-wide monthly limit first.

A refusal spends nothing, reserves nothing, dispatches nothing, and is never a state an
operator must clear: the next pass (or the next run of the stage) asks again by itself.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import paid_resource_admission

REACHED = "account_monthly_spend_limit_reached"


@pytest.fixture()
def limit(monkeypatch):
    """Switch the real account-limit check between refusing and admitting."""

    state = {"blockers": [REACHED]}
    monkeypatch.setattr(paid_resource_admission, "_account_monthly_limit_blockers", lambda: list(state["blockers"]))
    for name in ("BLUEPRINT_GCP_MONTHLY_BUDGET_REPORT", "BLUEPRINT_GCP_BUDGET_GUARD_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    return state


def test_launch_supervisor_waits_uncached_and_asks_again(tmp_path, limit, monkeypatch):
    from blueprint_pipeline import task_evaluation_launch_supervisor as supervisor

    calls = []

    class Invoker:
        def invoke(self, spec, prompt):
            calls.append(prompt)
            return SimpleNamespace(output={"summary": "ok", "recommended_profile_id": None,
                                           "human_decision_request": None, "blockers": []},
                                   provider="openai", model="m", sdk_version="x", trace_id="t")

    snapshot = {"snapshot_digest": "sha256:" + "a" * 64, "admissible_profile_ids": []}
    waiting = supervisor.run_launch_supervisor(snapshot=snapshot, output_dir=tmp_path, invoker=Invoker(), enabled=True)
    assert waiting["status"] == supervisor.ACCOUNT_LIMIT_WAIT_STATUS and waiting["blockers"] == [REACHED]
    assert calls == [] and not list(tmp_path.glob("*.json"))  # nothing invoked, nothing cached
    limit["blockers"] = []
    later = supervisor.run_launch_supervisor(snapshot=snapshot, output_dir=tmp_path, invoker=Invoker(), enabled=True)
    assert later["status"] != supervisor.ACCOUNT_LIMIT_WAIT_STATUS and len(calls) == 1


def test_agent_task_is_not_started_while_refused_and_starts_on_a_later_pass(limit, monkeypatch):
    from blueprint_pipeline.agent_execution import production
    from blueprint_pipeline.agent_execution import supervision
    from blueprint_pipeline.agent_execution.contracts import AgentExecutionError

    monkeypatch.setattr(supervision, "ownership_event", lambda *_args: None)
    started = []
    service = SimpleNamespace(
        validate_admission=lambda task: None, _ownership=lambda record: {},
        journal=SimpleNamespace(record_event=lambda *_args: None),
        service=SimpleNamespace(enqueue=lambda task: started.append(task) or {"state": "queued"}))
    record = SimpleNamespace(task=SimpleNamespace(task_id="task-1", run_id="run-1"), supervision=None)
    with pytest.raises(AgentExecutionError, match=production.ACCOUNT_LIMIT_WAIT):
        production.ProductionAgentService._enqueue_record(service, record, _run_lock_held=True)
    assert started == []  # nothing admitted with the provider; autostart skips it and asks again
    limit["blockers"] = []
    production.ProductionAgentService._enqueue_record(service, record, _run_lock_held=True)
    assert started == [record.task]


def test_removal_analysis_without_a_website_job_blocks_the_stage_by_name(limit, monkeypatch):
    from blueprint_pipeline import clean_plate_removal_analysis_gemini as analysis

    monkeypatch.setattr(analysis, "_analyze_removal_targets", lambda **_kw: pytest.fail("Gemini called"))
    plan = analysis.analyze_removal_targets(video_path="/nonexistent.mp4")
    assert plan["status"] == "blocked" and plan["blockers"] == [REACHED]
    limit["blockers"] = []
    monkeypatch.setattr(analysis, "_analyze_removal_targets", lambda **_kw: {"status": "completed"})
    assert analysis.analyze_removal_targets(video_path="/nonexistent.mp4") == {"status": "completed"}


def test_background_review_without_a_website_job_refuses_before_the_review(limit, monkeypatch, tmp_path):
    from blueprint_pipeline import website_image_completion as completion

    monkeypatch.setattr(completion, "_camera_changed_review", lambda *_args: None)
    monkeypatch.setattr(completion, "_verify_completed_background", lambda **_kw: pytest.fail("Gemini called"))
    with pytest.raises(paid_resource_admission.PaidResourceAdmissionBlocked):
        completion.verify_completed_background(frames=[], original_frames=[], plan={}, output_root=tmp_path)


def test_optional_phase2_skill_is_skipped_and_says_why(limit, monkeypatch):
    from blueprint_pipeline.agent_runtime import openai_phase2

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    runner = openai_phase2._OpenAISDKRunner(config=openai_phase2.OpenAIPhase2Config(mode="sdk"))
    assert runner("qualification_weakness_summarizer", {}) is None
    assert runner.runtime_metadata()["openai_phase2_last_provider_call"]["account_limit_blockers"] == [REACHED]
    codex = openai_phase2.CodexOpenAIPhase2Runner(config=openai_phase2.OpenAIPhase2Config(mode="codex_cli"),
                                                  repo_root=Path("."))
    monkeypatch.setattr(openai_phase2.shutil, "which", lambda _name: pytest.fail("codex looked up"))
    assert codex("qualification_weakness_summarizer", {}) is None


def test_arena_live_operator_gate_asks_the_limit_last(limit, monkeypatch):
    from blueprint_pipeline import arena_result_ingest as arena

    monkeypatch.setattr(arena, "_env_truthy", lambda _name: True)
    monkeypatch.setattr(arena, "_module_available", lambda _names: True)
    monkeypatch.setattr(arena, "codex_cli_path", lambda *_args: "/usr/bin/codex")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert arena._live_agents_blockers(True) == [REACHED]
    assert arena._live_codex_blockers(True) == [REACHED]
    assert arena._live_agents_blockers(False) == ["missing_cli_allow_live_agents_sdk"]  # its own gates first
    limit["blockers"] = []
    assert arena._live_agents_blockers(True) == [] and arena._live_codex_blockers(True) == []


def test_operator_runtime_refuses_before_any_live_call(limit):
    from blueprint_pipeline import agent_operator_runtime as runtime

    config = runtime.OperatorRunConfig(adapter="a", model="m", prompt="p", plan_context={})
    for run in (runtime.run_agents_sdk_operator, runtime.run_codex_cli_operator):
        with pytest.raises(paid_resource_admission.PaidResourceAdmissionBlocked):
            run(config)


@pytest.mark.parametrize("module", ["hosted_model_inference_preflight", "hosted_image_generation_preflight"])
def test_hosted_probe_is_a_blocked_record_with_no_request(limit, tmp_path, module):
    import importlib

    preflight = importlib.import_module(f"blueprint_pipeline.{module}")
    sent = []
    kwargs = {"output_path": tmp_path / "preflight.json",
              "http_post": lambda *args: sent.append(args) or (500, b"{}")}
    if module == "hosted_model_inference_preflight":
        kwargs.update(backend=next(iter(preflight.BACKENDS)), secret_loader=lambda _backend: ("key", "test"))
        record = preflight.materialize_hosted_model_inference_preflight(**kwargs)
    else:
        kwargs.update(secret_loader=lambda: ("key", "test"))
        record = preflight.materialize_hosted_image_generation_preflight(**kwargs)
    assert sent == [] and REACHED in record["blockers"]
    assert REACHED in json.dumps(json.loads((tmp_path / "preflight.json").read_text()))
