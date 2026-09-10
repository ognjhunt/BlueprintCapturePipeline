from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import astra_cad_skill_runtime as runtime


class FakeInvoker:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def invoke(self, spec, payload):
        self.calls.append((spec, payload))
        if self.fail:
            raise RuntimeError("preserved provider failure")
        return SimpleNamespace(output=runtime._TextOutput(content="# complete candidate\nx = 1\n"),
                               usage={"input_tokens": 10, "output_tokens": 8}, cost_usd=0.01,
                               cost_status="observed", trace_id="fake", provider="openai",
                               sdk_version="fake", model="gpt-6-astra")


class FakeRunner:
    def __init__(self):
        self.ready = False
        self.calls = []

    def preflight(self):
        self.ready = True

    def __call__(self, argv, **kwargs):
        assert self.ready
        self.calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")


def test_requires_real_sandbox_preflight_before_model_or_files(tmp_path):
    invoker = FakeInvoker()
    with pytest.raises(runtime.AstraCADRuntimeBlocked, match="sandboxed_subprocess_runner_required"):
        runtime.execute_mac_candidate("exact box", tmp_path / "out", tmp_path / "mac", tmp_path / "cad",
                                      invoker, expected_dimensions_mm=(1, 2, 3))
    assert not invoker.calls and not (tmp_path / "out").exists()


def test_sdk_bridge_preserves_exact_brief_and_has_one_turn(tmp_path):
    invoker = FakeInvoker()
    bridge = runtime._SDKChatBridge(invoker, tmp_path, "width 12.34567 mm", "shared-parent", 4000, 512, 1, "tray")
    bridge.create(messages=[{"role": "user", "content": "make JSON"}], model="qwen", max_tokens=99999)
    spec, payload = invoker.calls[0]
    assert spec.model == "gpt-6-astra" and spec.reasoning_effort == "high"
    assert spec.max_turns == 1 and spec.max_output_tokens == 512 and not spec.tool_bindings
    assert spec.run_id == "shared-parent" and spec.capability == "astra_cad_candidate:tray"
    assert json.loads(payload)["object_label"] == "tray"
    assert json.loads(payload)["immutable_original_brief"] == "width 12.34567 mm"
    with pytest.raises(runtime.AstraCADRuntimeBlocked, match="invocation_budget_exhausted"):
        bridge.create(messages=[])
    assert len(invoker.calls) == 1


def test_failed_sdk_call_is_counted_and_input_ceiling_precedes_spend(tmp_path):
    invoker = FakeInvoker(fail=True)
    bridge = runtime._SDKChatBridge(invoker, tmp_path, "brief", "run", 1000, 512, 1)
    with pytest.raises(RuntimeError, match="preserved provider failure"):
        bridge.create(messages=[])
    with pytest.raises(runtime.AstraCADRuntimeBlocked, match="budget_exhausted"):
        bridge.create(messages=[])
    assert json.loads((tmp_path / "invocations.json").read_text())[0]["failure"] == "RuntimeError"
    bridge = runtime._SDKChatBridge(invoker, tmp_path, "x" * 1001, "run", 1000, 512, 1)
    with pytest.raises(runtime.AstraCADRuntimeBlocked, match="input_token_ceiling"):
        bridge.create(messages=[])
    assert len(invoker.calls) == 1


@pytest.fixture
def fake_graph(monkeypatch, tmp_path):
    nodes = SimpleNamespace()
    graph = SimpleNamespace(get_default_initial_state=lambda: {})
    seen = {}

    def planner(state):
        nodes._llm_client().create(messages=[{"role": "user", "content": state["user_request"]}])
        return {"cad_brief": SimpleNamespace(part_name="box")}

    def architect(state):
        nodes._llm_client().create(messages=[{"role": "user", "content": "architecture"}])
        return {"architect_plan": {"dimensions": [12.34567, 20, 30]}}

    def coder(state):
        nodes._llm_client().create(messages=[{"role": "user", "content": "code"}])
        root = Path.cwd()
        nodes.subprocess.run(["python", "candidate.py"], env={"SECRET": "must disappear"},
                             cwd="/unsafe", timeout=999)
        for name in ("candidate.step", "candidate.stl", "candidate.py"):
            (root / name).write_text("candidate")
        return {"current_step_path": str(root / "candidate.step"),
                "current_stl_path": str(root / "candidate.stl")}

    def qa(state):
        root = Path.cwd()
        for _ in range(3):
            seen.setdefault("repairs", []).append(nodes._run_repair_on_script(
                script_path=str(root / "candidate.py"), error_details=["repair"], user_request="exact"))
        return {"error_type": "none"}

    for name, fn in (("node_spec_planner", planner), ("node_geometric_architect", architect),
                     ("node_python_coder", coder), ("node_autonomous_skill_loop", qa)):
        setattr(nodes, name, fn)
    nodes._call_llm_json_with_retry = lambda *a, **kw: kw
    nodes._extract_code_from_llm_response = lambda value: value

    def invoke(state, config):
        assert graph._MAX_SELF_RETRIES == 1 and config["recursion_limit"] == 8
        assert nodes._CFG_MAX_EXEC_RETRIES == 0
        assert nodes._node_python_coder_deterministic() is None
        assert nodes._optimize_print_orientation(Path("same.stl"))[0] == Path("same.stl")
        for name in ("OpenAI", "_fill_unsupported_with_aider", "generate_initial_solution", "_run_direct_repair_fallback"):
            with pytest.raises(runtime.AstraCADRuntimeBlocked, match="bypass_denied"):
                getattr(nodes, name)()
        assert nodes._call_llm_json_with_retry(max_retries=99)["max_retries"] == 1
        for name in ("node_spec_planner", "node_geometric_architect", "node_python_coder", "node_autonomous_skill_loop"):
            state.update(getattr(graph, name)(state))
        return state

    graph.build_graph = lambda: SimpleNamespace(invoke=invoke)

    @contextmanager
    def source_modules(*args):
        cwd = Path.cwd()
        try:
            yield graph, nodes
        finally:
            import os
            os.chdir(cwd)

    monkeypatch.setattr(runtime, "_source_modules", source_modules)
    monkeypatch.setattr(runtime, "_verify_source", lambda root, sha: {"commit": sha})
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(runtime, "_read_step", lambda *args: {"passed": True, "measured_dimensions_mm": [12.34567, 20, 30]})
    return seen


def test_graph_routes_models_and_execution_through_bounded_supplied_adapters(tmp_path, fake_graph):
    invoker, runner = FakeInvoker(), FakeRunner()
    root = tmp_path / "out"
    result = runtime.execute_mac_candidate("12.34567 x 20 x 30 mm", root, tmp_path / "mac", tmp_path / "cad",
                                           invoker, expected_dimensions_mm=(12.34567, 20, 30),
                                           subprocess_runner=runner)
    assert result["passed"] and result["graph_invoked"]
    assert len(invoker.calls) == 5 and fake_graph["repairs"] == [True, True, False]
    assert len(runner.calls) == 1
    assert runner.calls[0][1]["cwd"] == str(root)
    assert set(runner.calls[0][1]["env"]) == {"PYTHONPATH"}
    assert runner.calls[0][1]["timeout"] == 120
    assert (root / "repair-2-before.py").is_file()
    assert json.loads((root / "parameters.json").read_text())["expected_dimensions_mm"][0] == 12.34567


def test_step_readback_rejects_rounded_dimensions(tmp_path):
    build123d = pytest.importorskip("build123d")
    step = tmp_path / "box.step"
    build123d.export_step(build123d.Box(12.34567, 20, 30), str(step))
    assert runtime._read_step(step, (12.34567, 20, 30), 0.000001)["passed"]
    assert not runtime._read_step(step, (12.35, 20, 30), 0.000001)["passed"]


def test_dirty_source_refuses_before_model(tmp_path, monkeypatch):
    calls = []

    def output(argv, **kw):
        calls.append(argv)
        return runtime.MAC_COMMIT + "\n" if argv[-1] == "HEAD" else " M nodes.py\n"

    monkeypatch.setattr(runtime.subprocess, "check_output", output)
    with pytest.raises(runtime.AstraCADRuntimeBlocked, match="exact_clean_pin"):
        runtime._verify_source(tmp_path, runtime.MAC_COMMIT)
    assert len(calls) == 2
