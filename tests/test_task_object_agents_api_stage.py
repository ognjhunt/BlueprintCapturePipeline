"""Hermetic admission and scene-stage routing for opt-in managed authoring."""
from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline.agent_execution.contracts import AgentExecutionError, digest
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import task_evaluation_scene_configuration_astra_driver as driver
from blueprint_pipeline import task_object_agents_api_stage as managed
from tests.test_astra_automatic_resume import authoring_fixture  # noqa: F401
from tests.test_task_evaluation_scene_configuration_astra_driver import retained, component  # noqa: F401
from tests.test_task_object_agent_session import agent_fixture  # noqa: F401
from scripts.write_scene_agents_api_project_guard import write_guard


def _guard(tmp_path: Path) -> tuple[dict, Path, dict]:
    guard = {"schema_version": managed.GUARD_SCHEMA, "project_id": "proj_fixture",
        "credential_id": "key_fixture", "dashboard_hard_limit_enabled": True,
        "disclosure_scope": managed.DISCLOSURE_SCOPE,
        "budget_policy": "project_guard_accepted_uncertainty",
        "session_retention": "until_deleted", "trace_retention": "provider_default",
        "provider_api_region": "us", "observed_at": 1000, "expires_at": 2800,
        "spend_limit": {"object": "project.spend_limit", "currency": "USD",
                        "interval": "month", "threshold_amount": 500}}
    path = tmp_path / "project-guard.json"
    path.write_text(json.dumps(guard))
    path.chmod(0o600)
    policy = {"schema_version": "scene_configuration_agents_api_policy.v1",
        "disclosure_scope": managed.DISCLOSURE_SCOPE,
        "session_retention": "until_deleted", "trace_retention": "provider_default",
        "region": "us", "budget_policy": "project_guard_accepted_uncertainty",
        "project_guard_receipt_digest": digest(guard), "ttl_seconds": 300,
        "maximum_review_cycles": 3}
    return guard, path, policy


def test_project_guard_requires_fresh_exact_project_and_hard_limit(tmp_path):
    guard, path, policy = _guard(tmp_path)
    kwargs = dict(policy=policy, guard_file=path, project_id="proj_fixture",
                  credential_id="key_fixture", maximum_cost_usd=5.0,
                  deadline=1350, now=1100)
    assert managed.validate_managed_asset_guard(**kwargs) == digest(guard)
    with pytest.raises(AgentExecutionError, match="project_guard_not_admitted"):
        managed.validate_managed_asset_guard(**{**kwargs, "project_id": "other"})
    with pytest.raises(AgentExecutionError, match="project_guard_not_admitted"):
        managed.validate_managed_asset_guard(**{**kwargs, "maximum_cost_usd": 4.99})
    with pytest.raises(AgentExecutionError, match="project_guard_not_admitted"):
        managed.validate_managed_asset_guard(**{**kwargs, "now": 2800})


def test_operator_guard_writer_records_digest_without_exceeding_stage_cap(tmp_path):
    tmp_path.chmod(0o700)
    path = tmp_path / "private-guard.json"
    kwargs = dict(output=path, project_id="proj_fixture", credential_id="key_fixture",
                  hard_limit_usd="5.00", stage_cap_usd="5.00",
                  expires_in_seconds=3600, observed_at=1000.0)
    receipt_digest = write_guard(**kwargs)
    assert path.stat().st_mode & 0o777 == 0o600
    assert receipt_digest == digest(json.loads(path.read_text()))
    assert json.loads(path.read_text())["spend_limit"]["threshold_amount"] == 500
    with pytest.raises(ValueError, match="hard_limit_exceeds_stage_cap"):
        write_guard(**{**kwargs, "output": tmp_path / "blocked.json", "stage_cap_usd": "4.99"})
    assert not (tmp_path / "blocked.json").exists()


def test_packaging_requires_deleted_bound_managed_session_receipt(tmp_path):
    path = tmp_path / "stage-receipt.json"
    value = {"schema_version": "task_asset_agents_api_stage_receipt.v1",
        "provider": "openai", "model": "gpt-6-sol", "runtime": "openai_agents_api",
        "session_cleanup": "deleted", "result_digest": "sha256:" + "a" * 64}
    value["receipt_digest"] = canonical_digest(value)
    path.write_text(json.dumps(value))
    assert driver._managed_authoring_receipt(path, value) == value
    value["session_cleanup"] = "unknown"
    path.write_text(json.dumps(value))
    with pytest.raises(driver.AstraStageError, match="receipt_invalid"):
        driver._managed_authoring_receipt(path, value)


def test_rejected_review_continues_same_managed_session_with_bounded_revision(
        tmp_path, agent_fixture, monkeypatch):  # noqa: F811
    guard, guard_path, policy = _guard(tmp_path)
    calls = []

    class FakeRuntime:
        def __init__(self, **_kwargs):
            self.tasks = {}

        def start(self, task):
            calls.append(("start", task.task_id))
            self.tasks[task.task_id] = task

        def continue_task(self, task):
            assert task.parent_task_id in self.tasks
            calls.append(("continue", task.task_id))
            self.tasks[task.task_id] = task

        def step(self, task_id):
            return {"state": "completed", "session_id": "same-session",
                    "task": self.tasks[task_id].model_dump(mode="json")}

        def cleanup(self, task_id):
            calls.append(("cleanup", task_id))
            return {"cleanup_state": "deleted"}

    def review(self, *, task_state, invoker):
        index = len([row for row in calls if row[0] == "review"]) + 1
        calls.append(("review", task_state["task"]["task_id"]))
        if index == 1:
            value = {"accepted": False, "review": {"blockers": ["missing handle"]}}
            path = self.journal_root / "review-001-result.json"
            path.write_text(json.dumps(value))
            return value
        return {"accepted": True, "result": {"result_digest": "sha256:" + "a" * 64,
                                               "model": "gpt-6-sol"}}

    monkeypatch.setattr(managed, "OpenAIAgentsRuntime", FakeRuntime)
    monkeypatch.setattr(managed.AgentsAPIAssetTools, "review", review)
    f = agent_fixture
    result = managed.run_managed_asset_authoring(
        request_value=f.kwargs["request_value"], output_root=f.kwargs["output_root"],
        budget_root=tmp_path / "budget", cad_executor=f.kwargs["cad_executor"],
        blender_runner=f.kwargs["blender_runner"],
        blender_executable=f.kwargs["blender_executable"], review_invoker=object(),
        policy=policy, authority_digest="sha256:" + "b" * 64,
        source_commit="c" * 40, project_id="proj_fixture", credential_id="key_fixture",
        guard_file=guard_path, maximum_cost_usd=5.0,
        transport=SimpleNamespace(project_id="proj_fixture"), clock=lambda: 1100,
        sleep=lambda _seconds: None)
    assert result["model"] == "gpt-6-sol"
    assert [row[0] for row in calls] == ["start", "review", "continue", "review", "cleanup"]
    receipt = json.loads((tmp_path / "budget/agents_api_stage_receipt.json").read_text())
    assert receipt["session_id"] == "same-session"
    assert receipt["review_cycles"] == 2
    assert receipt["project_guard_digest"] == digest(guard)


@pytest.mark.parametrize("fault", ["review", "step_once", "step_terminal", "step_stuck"])
def test_managed_exception_cancels_and_deletes_only_a_settled_session(
        tmp_path, agent_fixture, monkeypatch, fault):  # noqa: F811
    _guard_value, guard_path, policy = _guard(tmp_path)
    events = []

    class FakeRuntime:
        def __init__(self, **_kwargs):
            self.state = "queued"
            self.failed_once = False

        def start(self, task):
            self.state = "running"
            events.append("start")

        def step(self, task_id):
            events.append("step")
            if fault == "step_terminal" and not self.failed_once:
                self.failed_once = True
                self.state = "completed"
                raise RuntimeError("injected step failure")
            if fault in {"step_once", "step_stuck"} and (fault == "step_stuck" or not self.failed_once):
                self.failed_once = True
                raise RuntimeError("injected step failure")
            self.state = "cancelled" if self.state == "cancelling" else "completed"
            return {"state": self.state, "session_id": "session-fixture"}

        def inspect(self, task_id):
            events.append("inspect")
            return {"state": self.state}

        def cancel(self, task_id):
            events.append("cancel")
            self.state = "cancelling"

        def cleanup(self, task_id):
            assert self.state in {"completed", "cancelled"}
            events.append("cleanup")
            return {"cleanup_state": "deleted"}

    def review(self, *, task_state, invoker):
        events.append("review")
        raise RuntimeError("injected review failure")

    monkeypatch.setattr(managed, "OpenAIAgentsRuntime", FakeRuntime)
    monkeypatch.setattr(managed.AgentsAPIAssetTools, "review", review)
    f = agent_fixture
    kwargs = dict(request_value=f.kwargs["request_value"], output_root=f.kwargs["output_root"],
        budget_root=tmp_path / "budget", cad_executor=f.kwargs["cad_executor"],
        blender_runner=f.kwargs["blender_runner"], blender_executable=f.kwargs["blender_executable"],
        review_invoker=object(), policy=policy, authority_digest="sha256:" + "b" * 64,
        source_commit="c" * 40, project_id="proj_fixture", credential_id="key_fixture",
        guard_file=guard_path, maximum_cost_usd=5.0,
        transport=SimpleNamespace(project_id="proj_fixture"), clock=lambda: 1100,
        sleep=lambda _seconds: None)
    if fault == "step_stuck":
        with pytest.raises(AgentExecutionError, match="exception_cleanup_unresolved"):
            managed.run_managed_asset_authoring(**kwargs)
        assert "cancel" in events and "cleanup" not in events
    else:
        with pytest.raises(RuntimeError, match="injected .* failure"):
            managed.run_managed_asset_authoring(**kwargs)
        assert events[-1] == "cleanup"
        assert ("cancel" in events) == (fault == "step_once")
    assert not (tmp_path / "budget/agents_api_stage_receipt.json").exists()


def test_signed_agents_api_stage_routes_to_managed_authoring_then_existing_packaging(tmp_path, monkeypatch):
    guard, guard_path, policy = _guard(tmp_path)
    rights = {"schema_version": "website_native_rights_admission.v1",
        "execution_authority": {"allowed_providers": ["openai"]},
        "private_provider_processing_allowed": True, "provider_training_allowed": False,
        "consent": {"provider_terms_reference": "owner-approved-fixture"}}
    rights["digest"] = canonical_digest(rights, digest_field="digest")
    config = {"authoring_agent_runtime": "openai_agents_api", "authoring_model_provider": "openai",
        "authoring_model": "gpt-6-sol", "source_observation_kind": "website_capture_frames",
        "agents_api_policy": policy}
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    skill = tmp_path / "cad/text-to-cad/skills/cad/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("fixture skill")
    key = tmp_path / "key"
    key.write_text("fixture-no-network")
    key.chmod(0o600)
    stage_input_path = tmp_path / "stage-input.json"
    stage_input_path.write_text("{}")
    values = {driver._INPUT_ENV: str(stage_input_path),
        "BLUEPRINT_SCENE_CONFIGURATION_AUTHORING_RUNTIME": "openai_agents_api",
        "BLUEPRINT_SCENE_CONFIGURATION_AUTHORITY_DIGEST": "sha256:" + "a" * 64,
        "BLUEPRINT_SCENE_CONFIGURATION_AGENTS_API_PROJECT_GUARD_FILE": str(guard_path),
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_CONTENT_AGENTS_MAX_COST_USD": "5",
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_COST_USD": "10",
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_REQUESTS": "8",
        "OPENAI_PROJECT_ID": "proj_fixture"}
    events = []
    monkeypatch.setattr(driver, "scene_configuration_openai_stage_scope", lambda *_args, **_kw:
        {"api_key_file": str(key), "api_key_id": "key_fixture"})
    monkeypatch.setattr(driver, "_stage_sdk_environment", lambda *_args: nullcontext())
    monkeypatch.setattr(driver, "budgeted_invoker", lambda **_kw:
        (object(), SimpleNamespace(manifest=lambda: {"review_calls": 1})))
    monkeypatch.setattr(managed, "run_managed_asset_authoring", lambda **kw:
        events.append(("managed", kw["policy"]["project_guard_receipt_digest"])) or
        {"result_digest": "sha256:" + "b" * 64, "model": "gpt-6-sol"})
    monkeypatch.setattr(driver, "_finish_component", lambda **kw:
        events.append(("package", kw["authored"]["model"])) or {"status": "candidate"})

    class Gate:
        def reserve(self): events.append(("reserve", None))
        def complete(self, **kwargs): events.append(("complete", kwargs["provider_call_performed"]))

    request = SimpleNamespace(run_id="run-fixture", model_dump=lambda **_kw: {"run_id": "run-fixture"})
    result = driver._execute_agents_api_stage(values=values,
        stage_input={"run_id": "run-fixture", "source_commit": "c" * 40},
        rights=rights, request=request, articulated=False, plan=None, part_requests=None,
        runtime=runtime, authored_root=runtime / "authoring", output=tmp_path / "delivery",
        physics_bounds={}, configuration=config, source_record={"digest": "sha256:" + "d" * 64},
        rights_record={}, cad_runtime={}, cad_root=tmp_path / "cad", verified_sources={},
        blender={"executable": "fixture-blender"}, sandbox=object(),
        result_path=tmp_path / "result.json", package_candidate=object(),
        cost_gate_factory=lambda **_kw: Gate())
    assert result == {"status": "candidate"}
    assert events == [("reserve", None), ("managed", digest(guard)),
                      ("complete", True), ("package", "gpt-6-sol")]
