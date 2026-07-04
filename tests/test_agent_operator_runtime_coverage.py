from __future__ import annotations

import builtins
import subprocess
import sys
import types
from pathlib import Path

import pytest

from blueprint_pipeline import agent_operator_runtime as runtime


pytestmark = pytest.mark.slow


def test_operator_runtime_env_paths_and_ledgers(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(runtime.AGENT_EXTERNAL_ACTIONS_ENV, "yes")
    monkeypatch.setenv(runtime.AGENT_SPEND_ACTIONS_ENV, "off")

    assert runtime.env_truthy(runtime.AGENT_EXTERNAL_ACTIONS_ENV) is True
    assert runtime.env_truthy(runtime.AGENT_SPEND_ACTIONS_ENV) is False
    assert runtime.string(None) == ""
    assert runtime.string("  value  ") == "value"
    assert runtime.module_available(["definitely_missing_blueprint_module"]) is False
    assert runtime.module_available(["json"]) is True
    assert runtime.codex_cli_path(str(tmp_path / "missing-codex")) is None
    fake_codex = tmp_path / "codex"
    fake_codex.write_text("#!/bin/sh\n", encoding="utf-8")
    assert runtime.codex_cli_path(str(fake_codex)) == str(fake_codex)
    monkeypatch.setattr(runtime.shutil, "which", lambda value: f"/usr/bin/{value}")
    assert runtime.codex_cli_path("") == "/usr/bin/codex"

    assert runtime.external_action_gates() == {
        "external_actions_env": runtime.AGENT_EXTERNAL_ACTIONS_ENV,
        "external_actions_allowed": True,
        "spend_actions_env": runtime.AGENT_SPEND_ACTIONS_ENV,
        "spend_actions_allowed": False,
    }
    assert runtime.proof_effect(summary="kept advisory", deterministic_artifacts_required=["manifest.json"]) == {
        "summary": "kept advisory",
        "proof_booleans_mutable_by_agent": False,
        "direct_proof_booleans_set_true": [],
        "requires_deterministic_accepted_artifacts": True,
        "deterministic_artifacts_required": ["manifest.json"],
    }

    blocked = runtime.blocked_operator_ledger(
        adapter="codex",
        blockers=["missing_gate"],
        command_chosen="codex exec",
        proof_artifacts_required=["accepted.json"],
    )
    assert blocked["operator_mode"] == "live_operator_blocked"
    assert blocked["decisions"][0]["reason"] == "missing_gate"
    assert blocked["commands_chosen"] == ["codex exec"]
    assert blocked["proof_effect"]["deterministic_artifacts_required"] == ["accepted.json"]

    normalized = runtime.normalize_operator_output(
        {
            "summary": "choose command",
            "decisions": [{"decision": "inspect"}],
            "tool_call_summaries": [{"tool": "x"}],
            "commands_chosen": ["python -m pytest"],
            "refusals": ["none"],
            "blockers": ["blocked"],
            "raw_result_type": "fake",
        }
    )
    assert normalized["final_output"] == "choose command"
    assert normalized["raw_result_type"] == "fake"
    assert runtime.normalize_operator_output(" done ") == {
        "final_output": "done",
        "decisions": [],
        "tool_call_summaries": [],
        "commands_chosen": [],
        "refusals": [],
        "blockers": [],
        "raw_result_type": "str",
    }

    completed = runtime.completed_operator_ledger(
        adapter="agents",
        output={"final_output": "all set", "raw_result_type": "sdk"},
        default_command="python -m pytest",
        proof_artifacts_required=["proof.json"],
    )
    assert completed["operator_mode"] == "live_operator"
    assert completed["commands_chosen"] == ["python -m pytest"]
    assert completed["decisions"][0]["summary"] == "all set"
    assert completed["proof_effect"]["deterministic_artifacts_required"] == ["proof.json"]


def test_operator_runtime_executor_shortcuts_normalize_output() -> None:
    config = runtime.OperatorRunConfig(
        adapter="fake",
        model="model",
        prompt="inspect",
        plan_context={"root": "capture"},
        executor=lambda prompt, context: {
            "final_response": f"{prompt}:{context['root']}",
            "commands_chosen": ["cmd"],
        },
    )

    for runner in [
        runtime.run_agents_sdk_operator,
        runtime.run_codex_sdk_operator,
        runtime.run_codex_cli_operator,
    ]:
        output = runner(config)
        assert output["final_output"] == "inspect:capture"
        assert output["commands_chosen"] == ["cmd"]


def test_agents_sdk_operator_missing_and_success(monkeypatch) -> None:
    config = runtime.OperatorRunConfig(adapter="agents", model="gpt", prompt="plan", plan_context={})
    real_import = builtins.__import__

    def import_missing(name, *args, **kwargs):
        if name == "agents":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_missing)
    with pytest.raises(RuntimeError, match="missing_openai_agents_sdk"):
        runtime.run_agents_sdk_operator(config)
    monkeypatch.setattr(builtins, "__import__", real_import)

    agents_module = types.ModuleType("agents")

    class FakeAgent:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeRunner:
        @staticmethod
        async def run(agent, prompt):
            assert agent.kwargs["name"] == "agents"
            assert prompt == "plan"
            return types.SimpleNamespace(
                final_output="operator summary",
                new_items=[
                    types.SimpleNamespace(item_type="tool_call", raw_item=types.SimpleNamespace(tool_name="shell")),
                    types.SimpleNamespace(type="message", raw_item=types.SimpleNamespace(name="assistant")),
                    types.SimpleNamespace(type="", raw_item=types.SimpleNamespace()),
                ],
            )

    agents_module.Agent = FakeAgent
    agents_module.Runner = FakeRunner
    monkeypatch.setitem(sys.modules, "agents", agents_module)

    output = runtime.run_agents_sdk_operator(config)

    assert output["final_output"] == "operator summary"
    assert output["tool_call_summaries"] == [
        {"index": 0, "item_type": "tool_call", "tool_name": "shell"},
        {"index": 1, "item_type": "message", "tool_name": "assistant"},
    ]
    assert output["raw_result_type"] == "SimpleNamespace"


def test_codex_sdk_operator_missing_and_success(monkeypatch) -> None:
    config = runtime.OperatorRunConfig(
        adapter="codex",
        model="gpt",
        prompt="inspect",
        plan_context={},
        sandbox="workspace-write",
    )
    real_import = builtins.__import__

    def import_missing(name, *args, **kwargs):
        if name == "openai_codex":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_missing)
    with pytest.raises(RuntimeError, match="missing_codex_sdk"):
        runtime.run_codex_sdk_operator(config)
    monkeypatch.setattr(builtins, "__import__", real_import)

    codex_module = types.ModuleType("openai_codex")
    calls: list[dict[str, object]] = []

    class FakeSandbox:
        read_only = "ro"
        workspace_write = "ww"

    class FakeThread:
        def run(self, prompt):
            assert prompt == "inspect"
            return types.SimpleNamespace(
                final_response="codex response",
                new_items=[types.SimpleNamespace(type="tool", raw_item=types.SimpleNamespace(name="edit"))],
            )

    class FakeCodex:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def thread_start(self, **kwargs):
            calls.append(kwargs)
            return FakeThread()

    codex_module.Codex = FakeCodex
    codex_module.Sandbox = FakeSandbox
    monkeypatch.setitem(sys.modules, "openai_codex", codex_module)

    output = runtime.run_codex_sdk_operator(config)

    assert calls == [{"model": "gpt", "sandbox": "ww"}]
    assert output["final_output"] == "codex response"
    assert output["tool_call_summaries"] == [{"index": 0, "item_type": "tool", "tool_name": "edit"}]


def test_codex_cli_operator_success_and_failures(monkeypatch, tmp_path: Path) -> None:
    config = runtime.OperatorRunConfig(
        adapter="codex-cli",
        model="gpt",
        prompt="plan",
        plan_context={},
        sandbox="invalid",
        cwd=str(tmp_path),
        timeout_seconds=0,
    )

    monkeypatch.setattr(runtime, "codex_cli_path", lambda _bin: None)
    with pytest.raises(RuntimeError, match="missing_codex_cli"):
        runtime.run_codex_cli_operator(config)

    monkeypatch.setattr(runtime, "codex_cli_path", lambda _bin: "/usr/bin/codex")

    def raise_os_error(*_args, **_kwargs):
        raise OSError("no exec")

    monkeypatch.setattr(runtime.subprocess, "run", raise_os_error)
    with pytest.raises(RuntimeError, match="codex_cli_operator_execution_failed:OSError"):
        runtime.run_codex_cli_operator(config)

    def failing_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(["codex"], 2, stdout="bad")

    monkeypatch.setattr(runtime.subprocess, "run", failing_run)
    with pytest.raises(RuntimeError, match="codex_cli_operator_execution_failed$"):
        runtime.run_codex_cli_operator(config)

    def successful_run(command, **kwargs):
        assert command[command.index("--sandbox") + 1] == "read-only"
        assert command[command.index("--cd") + 1] == str(tmp_path.resolve())
        assert command[-3:] == ["--model", "gpt", "-"]
        assert kwargs["input"] == "plan"
        assert kwargs["timeout"] == 1
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(" final operator output ", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="stdout fallback")

    monkeypatch.setattr(runtime.subprocess, "run", successful_run)
    output = runtime.run_codex_cli_operator(config)

    assert output["final_output"] == "final operator output"
    assert output["tool_call_summaries"] == [
        {"tool_name": "codex_exec", "transport": "codex_cli_host_oauth", "exit_code": 0}
    ]
    assert output["raw_result_type"] == "codex_cli_exec"

    def stdout_run(command, **_kwargs):
        return subprocess.CompletedProcess(command, 0, stdout=" stdout only ")

    monkeypatch.setattr(runtime.subprocess, "run", stdout_run)
    assert runtime.run_codex_cli_operator(config)["final_output"] == "stdout only"
