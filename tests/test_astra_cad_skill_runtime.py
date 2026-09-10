from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
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
    seen = {"nodes": nodes}

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
        if seen.get("exercise_fallback_cli"):
            nodes.subprocess.run(["python", str(root / "temp_design_0.py")], cwd=str(root))
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
        assert callable(nodes._node_python_coder_deterministic)
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


def test_sealed_python_loader_survives_checkmesh_and_cad_cli_subprocesses(tmp_path, fake_graph, monkeypatch):
    loader = tmp_path / "sealed_python_runtime"
    loader.mkdir()
    monkeypatch.setenv("PYTHONPATH", str(loader))
    fake_graph["exercise_fallback_cli"] = True
    runner = FakeRunner()
    runtime.execute_mac_candidate("box", tmp_path / "out", tmp_path / "mac", tmp_path / "cad", FakeInvoker(),
        expected_dimensions_mm=(12.34567, 20, 30), subprocess_runner=runner)
    assert len(runner.calls) == 2
    assert str(loader) in runner.calls[0][1]["env"]["PYTHONPATH"].split(os.pathsep)
    assert "skills/cad/scripts/step" in runner.calls[1][0][1]
    assert str(loader) in runner.calls[1][1]["env"]["PYTHONPATH"].split(os.pathsep)
    assert set(runner.calls[1][1]["env"]) == {"PYTHONPATH"}


def _stub_step_reader(monkeypatch, path, dimensions, *, solid_count=1, volume=1.0,
                      valid=True, validity_is_method=False):
    """Fake only the external STEP kernel; production readback computes every verdict."""
    path.write_bytes(b"retained STEP parser fixture")
    imported = []
    shape = SimpleNamespace(
        bounding_box=lambda: SimpleNamespace(size=SimpleNamespace(
            X=dimensions[0], Y=dimensions[1], Z=dimensions[2])),
        is_valid=(lambda: valid) if validity_is_method else valid,
        volume=volume, solids=lambda: [object() for _ in range(solid_count)],
    )
    def import_step(actual_path):
        assert actual_path == str(path)
        assert Path(actual_path).read_bytes() == b"retained STEP parser fixture"
        imported.append(actual_path)
        return shape
    monkeypatch.setitem(sys.modules, "build123d", SimpleNamespace(import_step=import_step))
    versions = {"build123d": "fixture-build123d", "fixture-ocp": "fixture-kernel"}
    monkeypatch.setattr(runtime.importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(runtime.importlib.metadata, "packages_distributions", lambda: {"OCP": ["fixture-ocp"]})
    return imported


def test_step_readback_rejects_rounded_dimensions(tmp_path, monkeypatch):
    step = tmp_path / "box.step"
    imported = _stub_step_reader(monkeypatch, step, (12.34567, 20, 30))
    exact = runtime._read_step(step, (12.34567, 20, 30), 0.000001)
    assert exact["passed"] and exact["measured_dimensions_mm"] == [12.34567, 20.0, 30.0]
    assert exact["build123d"] == "fixture-build123d"
    assert exact["kernel_versions"] == {"fixture-ocp": "fixture-kernel"}
    assert not runtime._read_step(step, (12.35, 20, 30), 0.000001)["passed"]
    assert imported == [str(step), str(step)]


@pytest.mark.parametrize(("dimensions", "volume", "valid"), [
    ((1, 2, 3), 6, False),
    ((1, 2, 3), 0, True),
    ((float("nan"), 2, 3), 6, True),
    ((1, float("inf"), 3), 6, True),
])
def test_step_readback_refuses_invalid_nonpositive_or_nonfinite_kernel_outputs(
        tmp_path, monkeypatch, dimensions, volume, valid):
    step = tmp_path / "invalid.step"
    _stub_step_reader(monkeypatch, step, dimensions, volume=volume, valid=valid,
                      validity_is_method=True)
    assert runtime._read_step(step, (1, 2, 3), 0.000001)["passed"] is False


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


def test_adoption_requires_matching_completed_sdk_output_and_exact_parameters(tmp_path):
    from pydantic import BaseModel

    class Brief(BaseModel):
        part_name: str
        user_request_raw: str

    class Plan(BaseModel):
        width: float

    source = tmp_path / 'prior'
    source.mkdir()
    budget = tmp_path / 'budget'
    completions = budget / 'inference_reservations/completed'
    completions.mkdir(parents=True)
    parameters = {'brief': 'exact 12.34567 mm', 'expected_dimensions_mm': [12.34567, 20, 30],
                  'run_id': 'same-global-run', 'object_label': 'book'}
    runtime._save(source / 'parameters.json', parameters)
    for index, (node, field, output) in enumerate((
        ('node_spec_planner', 'cad_brief', {'part_name': 'book', 'user_request_raw': 'book'}),
        ('node_geometric_architect', 'architect_plan', {'width': 12.34567}),
    )):
        raw = json.dumps(output)
        (source / f'invocation-{index:02d}-output.txt').write_text(raw)
        retained = dict(output)
        if index == 0:
            retained['user_request_raw'] = parameters['brief']
        runtime._save(source / f'node-{node}.json', {field: retained})
        completion = {'run_id': parameters['run_id'], 'model': 'gpt-6-astra', 'provider': 'openai',
                      'capability': 'astra_cad_candidate:book',
                      'structured_output_digest': canonical_digest({'content': raw})}
        completion['inference_completion_digest'] = canonical_digest(completion, digest_field='inference_completion_digest')
        runtime._save(completions / f'{index}.json', completion)
    nodes = SimpleNamespace(CADBrief=Brief, ArchitectPlan=Plan)
    result, receipt = runtime._adopt_completed_phases(source, budget, parameters, nodes)
    assert result['node_spec_planner']['cad_brief'].user_request_raw == parameters['brief']
    assert len(receipt['phases']) == 2
    with pytest.raises(runtime.AstraCADRuntimeBlocked, match='parameters_mismatch'):
        runtime._adopt_completed_phases(source, budget, {**parameters, 'brief': 'changed'}, nodes)
    (source / 'invocation-01-output.txt').write_text('{"width":12.35}')
    with pytest.raises(runtime.AstraCADRuntimeBlocked, match='missing_sdk_completion'):
        runtime._adopt_completed_phases(source, budget, parameters, nodes)


def test_compact_coder_prompt_preserves_curved_geometry_and_no_duplicate_brief():
    prompt = runtime._compact_coder_prompt(plan_json=json.dumps({'radius': None, 'control_points': [],
        'notes': 'degree-five Bezier, no overshoot', 'width': 12.34567}), previous_feedback='',
        user_request='giant duplicated brief', step_path='/scratch/model.step', stl_path='/scratch/model.stl')
    result = json.loads(prompt)
    assert result['architect_plan'] == {'notes': 'degree-five Bezier, no overshoot', 'width': 12.34567}
    assert result['step_path'] == '/scratch/model.step'
    assert 'giant duplicated brief' not in prompt


def test_step_readback_rejects_disconnected_solids_even_with_exact_envelope(tmp_path, monkeypatch):
    path = tmp_path / "disconnected.step"
    _stub_step_reader(monkeypatch, path, (3, 1, 1), solid_count=2, volume=2,
                      validity_is_method=True)
    result = runtime._read_step(path, (3, 1, 1), 0.01)
    assert result["measured_dimensions_mm"] == [3.0, 1.0, 1.0]
    assert result["solid_count"] == 2 and result["valid"] and not result["passed"]


def test_adopted_coder_uses_no_invoker_and_retains_pending_raw_program(tmp_path):
    invoker = FakeInvoker(fail=True)
    bridge = runtime._SDKChatBridge(invoker, tmp_path, 'exact', 'run', 4000, 512, 1)
    raw = 'from build123d import *\ndef gen_step():\n    return Box(1,2,3)\n'
    bridge.adopted_coder_output = raw
    for _ in range(2):
        response = bridge.create(messages=[{'role': 'system', 'content': runtime._CODER_SYSTEM_PROMPT}])
        assert response.choices[0].message.content == raw
        assert bridge.pending_coder_output == raw
    assert invoker.calls == [] and bridge.calls == []


def test_upstream_shim_is_retained_but_raw_coder_source_runs_via_skill(tmp_path, fake_graph):
    nodes = fake_graph['nodes']
    raw = '# complete candidate\nx = 1\n'

    def coder(state):
        nodes._llm_client().create(messages=[{'role': 'system', 'content': runtime._CODER_SYSTEM_PROMPT}])
        root = Path.cwd()
        path = root / 'temp_design_0.py'
        path.write_text('# destructive shim\n' + raw)
        nodes.subprocess.run(['/python', str(path)], capture_output=True)
        assert path.read_text() == raw
        assert path.with_suffix('.upstream-compat.py').read_text() == '# destructive shim\n' + raw
        (root / 'candidate.py').write_text(raw)
        for name in ('temp_output_0.step', 'temp_output_0.stl'):
            (root / name).write_text('fixture')
        return {'current_step_path': str(root / 'temp_output_0.step'),
                'current_stl_path': str(root / 'temp_output_0.stl')}

    nodes.node_python_coder = coder
    runner = FakeRunner()
    result = runtime.execute_mac_candidate('box', tmp_path / 'out', tmp_path / 'mac', tmp_path / 'cad',
        FakeInvoker(), expected_dimensions_mm=(12.34567,20,30), subprocess_runner=runner, repair_budget=0)
    assert result['passed']
    argv = runner.calls[0][0]
    assert str(tmp_path / 'cad/skills/cad/scripts/step') == argv[1]
    assert '--output' in argv and '--stl' in argv


def test_coder_adoption_binds_completed_output_to_original_budgeted_input(tmp_path):
    source = tmp_path / 'prior'
    source.mkdir()
    budget = tmp_path / 'budget/inference_reservations'
    (budget / 'completed').mkdir(parents=True)
    (budget / 'reserved').mkdir()
    params = {'brief': 'exact original brief', 'expected_dimensions_mm': [1,2,3],
              'run_id': 'global-run', 'object_label': 'book'}
    request = {'object_label': 'book', 'immutable_original_brief': params['brief'],
               'upstream_messages': [{'role': 'system', 'content': runtime._CODER_SYSTEM_PROMPT}]}
    raw = 'from build123d import *\ndef gen_step():\n    return Box(1,2,3)\n'
    runtime._save(source / 'parameters.json', params)
    runtime._save(source / 'invocation-00-input.json', request)
    (source / 'invocation-00-output.txt').write_text(raw)
    reservation = {'reservation_id': 'reservation',
                   'input_digest': canonical_digest({'input_text': json.dumps(request)})}
    reservation['inference_reservation_digest'] = canonical_digest(reservation, digest_field='inference_reservation_digest')
    completion = {'reservation_id': 'reservation', 'run_id': params['run_id'], 'provider': 'openai',
                  'model': 'gpt-6-astra', 'capability': 'astra_cad_candidate:book',
                  'structured_output_digest': canonical_digest({'content': raw})}
    completion['inference_completion_digest'] = canonical_digest(completion, digest_field='inference_completion_digest')
    runtime._save(budget / 'reserved/receipt.json', reservation)
    runtime._save(budget / 'completed/receipt.json', completion)
    adopted, receipt = runtime._adopt_completed_coder(source, budget.parent, params)
    assert adopted == raw and receipt['source_output_sha256'] == runtime._digest(source / 'invocation-00-output.txt')
    request['upstream_messages'].append({'role': 'user', 'content': 'tampered request'})
    runtime._save(source / 'invocation-00-input.json', request)
    with pytest.raises(runtime.AstraCADRuntimeBlocked, match='reservation_mismatch'):
        runtime._adopt_completed_coder(source, budget.parent, params)
