from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import stat
from types import SimpleNamespace
import zipfile

import pytest

from blueprint_pipeline import astra_cad_skill_runtime as runtime
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


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
    skill = tmp_path / "cad/skills/cad/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("# Fixture CAD instructions")
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


@pytest.fixture
def archived_sources(tmp_path, monkeypatch):
    license_bytes = b"MIT fixture license\n"
    specs = tuple({**spec, "license_sha256": "sha256:" + hashlib.sha256(license_bytes).hexdigest()}
                  for spec in runtime.SOURCE_SPECS)
    monkeypatch.setattr(runtime, "SOURCE_SPECS", specs)
    receipt = {"schema_version": "task_evaluation_cad_skill_component_source.v1", "status": "pinned_sources_packaged",
               "scene_specific_source": False, "skill_count": 10, "sources": [], "receipt_digest": ""}
    supplied = {}
    receipt_path = tmp_path / "cad_skill_source_receipt.json"
    for name, spec in zip(("cad", "mac"), specs, strict=True):
        root = tmp_path / name
        root.mkdir()
        files = {"LICENSE": license_bytes, "source.py": b"# exact fixture source\n"}
        archive_path = tmp_path / f"{name}.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            for relative, data in files.items():
                archive.writestr(relative, data)
                (root / relative).write_bytes(data)
        archive_hash = "sha256:" + runtime._digest(archive_path)
        receipt["sources"].append({**spec, "skills": list(spec["skills"]), "archive_sha256": archive_hash})
        supplied[name] = {"root": str(root), "commit": spec["commit"], "source_diff": "",
            "tracked_file_sha256": {relative: hashlib.sha256(data).hexdigest() for relative, data in files.items()},
            "source_receipt_path": str(receipt_path), "archive_path": str(archive_path), "archive_sha256": archive_hash}

    def save_receipt():
        receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
        receipt_path.write_text(json.dumps(receipt))
        for row in supplied.values():
            row["source_receipt_digest"] = receipt["receipt_digest"]

    def rebind_archive(name):
        digest = "sha256:" + runtime._digest(Path(supplied[name]["archive_path"]))
        supplied[name]["archive_sha256"] = digest
        next(row for row in receipt["sources"] if row["commit"] == supplied[name]["commit"])["archive_sha256"] = digest
        save_receipt()

    save_receipt()
    return SimpleNamespace(root=tmp_path, supplied=supplied, receipt=receipt,
                           save_receipt=save_receipt, rebind_archive=rebind_archive)


def test_verified_archives_need_no_git_and_preserve_full_byte_inventory(archived_sources, monkeypatch):
    bundle = archived_sources
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda *a, **kw: pytest.fail("archive verification must not use git"))
    results = runtime.verify_cad_sources(bundle.root / "mac", bundle.root / "cad", bundle.supplied)
    for name in ("mac", "cad"):
        assert results[name]["admission"] == "sealed_component_archive"
        assert results[name]["tracked_file_sha256"] == bundle.supplied[name]["tracked_file_sha256"]
        assert results[name]["source_receipt_digest"] == bundle.receipt["receipt_digest"]


@pytest.mark.parametrize("field", ["commit", "tree", "license", "license_sha256", "repository"])
def test_recomputed_receipt_cannot_change_a_pinned_source_identity(archived_sources, field):
    bundle = archived_sources
    bundle.receipt["sources"][0][field] = "changed"
    bundle.save_receipt()
    with pytest.raises(runtime.AstraCADRuntimeBlocked, match="receipt_pins_invalid"):
        runtime.verify_cad_sources(bundle.root / "mac", bundle.root / "cad", bundle.supplied)


def test_changed_archive_refuses_even_when_caller_recomputes_archive_hash(archived_sources):
    bundle = archived_sources
    archive = Path(bundle.supplied["mac"]["archive_path"])
    with archive.open("ab") as stream:
        stream.write(b"changed")
    bundle.supplied["mac"]["archive_sha256"] = "sha256:" + runtime._digest(archive)
    with pytest.raises(runtime.AstraCADRuntimeBlocked, match="archive_digest_mismatch"):
        runtime.verify_cad_sources(bundle.root / "mac", bundle.root / "cad", bundle.supplied)


@pytest.mark.parametrize("mutation", ["changed", "extra", "missing"])
def test_recomputed_caller_inventory_cannot_admit_different_extracted_bytes(archived_sources, mutation):
    bundle = archived_sources
    root = bundle.root / "mac"
    if mutation == "changed":
        (root / "source.py").write_text("# changed")
    elif mutation == "extra":
        (root / "extra.py").write_text("# untracked module")
    else:
        (root / "source.py").unlink()
    bundle.supplied["mac"]["tracked_file_sha256"] = {p.name: runtime._digest(p) for p in root.iterdir()}
    with pytest.raises(runtime.AstraCADRuntimeBlocked, match="file_inventory_mismatch"):
        runtime.verify_cad_sources(root, bundle.root / "cad", bundle.supplied)


@pytest.mark.parametrize("name", ["../escape.py", "/absolute.py", "a/../../escape.py", "a\\escape.py", "./hidden.py"])
def test_unsafe_archive_paths_refuse_even_with_rebound_receipt(archived_sources, name):
    bundle = archived_sources
    with zipfile.ZipFile(bundle.supplied["mac"]["archive_path"], "a") as archive:
        archive.writestr(name, "# never extract")
    bundle.rebind_archive("mac")
    with pytest.raises(runtime.AstraCADRuntimeBlocked, match="archive_path_invalid"):
        runtime.verify_cad_sources(bundle.root / "mac", bundle.root / "cad", bundle.supplied)


def test_archive_symlink_and_extracted_symlink_are_forbidden(archived_sources):
    bundle = archived_sources
    link = bundle.root / "mac/link.py"
    link.symlink_to(bundle.root / "cad/source.py")
    with pytest.raises(runtime.AstraCADRuntimeBlocked, match="extracted_path_invalid"):
        runtime.verify_cad_sources(bundle.root / "mac", bundle.root / "cad", bundle.supplied)
    link.unlink()
    info = zipfile.ZipInfo("link.py")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(bundle.supplied["mac"]["archive_path"], "a") as archive:
        archive.writestr(info, "../cad/source.py")
    bundle.rebind_archive("mac")
    with pytest.raises(runtime.AstraCADRuntimeBlocked, match="archive_link_or_special_file"):
        runtime.verify_cad_sources(bundle.root / "mac", bundle.root / "cad", bundle.supplied)


def test_license_bytes_must_match_pinned_license_digest(archived_sources):
    bundle = archived_sources
    archive = Path(bundle.supplied["mac"]["archive_path"])
    (bundle.root / "mac/LICENSE").write_text("different license")
    with zipfile.ZipFile(archive, "w") as archive:
        for path in (bundle.root / "mac").iterdir():
            archive.write(path, path.name)
    bundle.rebind_archive("mac")
    with pytest.raises(runtime.AstraCADRuntimeBlocked, match="license_bytes_mismatch"):
        runtime.verify_cad_sources(bundle.root / "mac", bundle.root / "cad", bundle.supplied)


def test_archive_drift_is_rejected_before_any_model_call(archived_sources):
    bundle = archived_sources
    (bundle.root / "mac/source.py").write_text("changed bytes")
    invoker = FakeInvoker()
    with pytest.raises(runtime.AstraCADRuntimeBlocked, match="file_inventory_mismatch"):
        runtime.execute_mac_candidate("box", bundle.root / "out", bundle.root / "mac", bundle.root / "cad",
            invoker, expected_dimensions_mm=(1, 2, 3), subprocess_runner=FakeRunner(), verified_sources=bundle.supplied)
    assert invoker.calls == []


def test_source_change_after_graph_cannot_leave_a_passing_receipt(tmp_path, fake_graph, monkeypatch):
    calls = []
    def verify(*args):
        calls.append(True)
        return {"state": "before" if len(calls) == 1 else "changed"}
    monkeypatch.setattr(runtime, "verify_cad_sources", verify)
    output = tmp_path / "out"
    with pytest.raises(runtime.AstraCADRuntimeBlocked, match="source_changed_during_execution"):
        runtime.execute_mac_candidate("box", output, tmp_path / "mac", tmp_path / "cad", FakeInvoker(),
            expected_dimensions_mm=(12.34567, 20, 30), subprocess_runner=FakeRunner())
    receipt = json.loads((output / "candidate-receipt.json").read_text())
    assert receipt["passed"] is False and receipt["source_unchanged_after"] is False
    assert len(calls) == 2
