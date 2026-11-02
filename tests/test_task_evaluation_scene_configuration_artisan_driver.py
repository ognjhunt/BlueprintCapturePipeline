"""Hermetic stage orchestration with real final USD; no framework/live proof."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import time
import zipfile

from PIL import Image
import pytest

from blueprint_pipeline import task_evaluation_scene_configuration_artisan_driver as driver
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_object_authoring_contracts import file_record
from tests.test_task_object_artisan_contracts import source_request
from tests.test_task_object_artisan_exports import construction_request, usd_fixture, export_fixture


@dataclass
class FixtureGate:
    output_root: Path
    events: list

    def __post_init__(self):
        self.output_root.mkdir(parents=True, exist_ok=True)

    @property
    def reservation_path(self):
        return self.output_root / "reservation.json"

    def reserve(self):
        self.events.append("reserve_original_pool")
        self.reservation_path.write_text('{"test_fixture":true}')

    def _validated_reservation(self):
        assert json.loads(self.reservation_path.read_text()) == {"test_fixture": True}

    def complete(self, **kwargs):
        self.events.append(("completion", str(self.output_root), kwargs))


@pytest.fixture
def stage_fixture(tmp_path, monkeypatch):
    stage, deps, frames = source_request(tmp_path, frames=21)
    stage["stage"] = {"stage_id": "stage-3", "adapter": {"id": "content_agents_rigid_replacement"}}
    stage["toolchain_digest"] = "sha256:" + "c" * 64
    stage["configuration"]["metric_envelope"].update(minimum_xyz_m=[-0.5, -0.5, 0], maximum_xyz_m=[0.5, 0.5, 1])
    stage["configuration"]["required_output"] = construction_request()["configuration"]["required_output"]
    stage["configuration"]["provider_disclosure"] = {"derived_views_and_metric_envelope": True, "provider_training": False,
                                                     "public_redistribution": False}
    rights = tmp_path / "rights.json"
    rights.write_text(json.dumps({"status": "admitted_for_internal_development", "private_provider_processing_allowed": True,
                                 "provider_training_allowed": False, "public_redistribution_allowed": False}))
    stage["construction_envelope"]["materialized_references"] = [{"contract_path": "scene.rights.admission",
        "materialized_path": str(rights), **file_record(rights)}]
    toolchain, package, output = tmp_path / "toolchain", tmp_path / "toolchain/package", tmp_path / "output"
    for path in (package / "artisan_source", package / "runtime/site_packages", package / "runtime/blender", package / "bridge", output):
        path.mkdir(parents=True)
    key = tmp_path / "scoped-key"
    key.write_text("synthetic-key-never-sent")
    key.chmod(0o600)
    input_path, dependencies = tmp_path / "input.json", tmp_path / "dependencies.json"
    input_path.write_text(json.dumps(stage))
    dependencies.write_text(json.dumps(deps))
    environment = {driver.INPUT: str(input_path), driver.DEPENDENCIES: str(dependencies), driver.OUTPUT: str(output),
        driver.RESULT: str(output / "result.json"), driver.PACKAGE: str(package),
        "BLUEPRINT_TASK_EVALUATION_SCENE_CONFIGURATION_TOOLCHAIN_ROOT": str(toolchain),
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_CONTENT_AGENTS_MAX_COST_USD": "12",
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_COST_USD": "20", "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_REQUESTS": "32",
        driver.STAGE_DEADLINE_EPOCH_ENV: str(time.time() + 1800), driver.PARENT_DEADLINE_EPOCH_ENV: str(time.time() + 3600)}
    from blueprint_pipeline import task_evaluation_scene_configuration_stage_tool as stage_tool
    from blueprint_pipeline import task_evaluation_scene_configuration_builtin_producers as producers
    from blueprint_pipeline import task_object_artisan_hooks as hooks
    monkeypatch.setattr(stage_tool, "_validate_input", lambda value, **kwargs: value)
    monkeypatch.setattr(stage_tool, "_validate_dependencies", lambda value: value)
    monkeypatch.setattr(driver, "source_frames", lambda *_: ({"role": "synthetic source handoff fixture"}, frames))
    monkeypatch.setattr(driver, "validate_artisan_runtime", lambda *_: {"backend": "artisan_v1", "test_fixture": True})
    monkeypatch.setattr(producers, "_validate_toolchain", lambda **_: ({"toolchain_digest": stage["toolchain_digest"]}, {}))
    monkeypatch.setattr(driver, "scene_configuration_openai_stage_scope", lambda *_args, **_kwargs: {"api_key_file": str(key)})
    class Preflight:
        def preflight(self): pass
        def __call__(self, *_args, **_kwargs): return SimpleNamespace(returncode=0)
    monkeypatch.setattr(hooks, "sandbox", lambda *_: Preflight())
    events, calls = [], []
    def gate_factory(**kwargs):
        return FixtureGate(kwargs["output_root"], events)
    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        runtime = output / "artisan_runtime"
        context = json.loads((runtime / "bridge_context.v1.json").read_text())
        workspace = Path(context["sandbox"]["write_root"])
        workspace.mkdir(parents=True, exist_ok=True)
        terminal, manifest = export_fixture(workspace)
        usd = usd_fixture(workspace / "geometry")
        exported = workspace / "exports/object.usd"
        shutil.copyfile(usd, exported)
        record = file_record(exported)
        manifest["artifacts"]["object.usd"].update(sha256=record["sha256"][7:], size_bytes=record["size_bytes"])
        (workspace / "exports/export_manifest.json").write_text(json.dumps(manifest))
        terminal["paths"]["root"] = str(workspace)
        terminal["usage"] = {"calls": [], "test_fixture": True}
        (runtime / "terminal.v1.json").write_text(json.dumps(terminal))
        return SimpleNamespace(returncode=0, stdout="synthetic upstream terminal", stderr="")
    return SimpleNamespace(environment=environment, runner=runner, gate=gate_factory, events=events, calls=calls,
                           output=output, key=key, package=package, frames=frames)


def test_invokes_real_cli_shape_with_every_view_and_adopts_completed_bytes_without_runtime(stage_fixture):
    fixture = stage_fixture
    result = driver.execute_artisan_component(environment=fixture.environment, runner=fixture.runner, cost_gate_factory=fixture.gate)
    argv, kwargs = fixture.calls[0]
    assert argv[1:4] == ["-S", "-m", "blueprint_pipeline.task_object_artisan_bridge"]
    assert argv.count("--image") == 21 and argv[argv.index("--max-cost-usd") + 1] == "12.0"
    assert kwargs["timeout"] < 1800 - 600
    assert "OPENAI_API_KEY" not in kwargs["env"]
    assert result["backend"] == "artisan_v1" and result["status"] == "completed"
    fixture.key.unlink()
    shutil.rmtree(fixture.package)
    def forbidden(*_args, **_kwargs): pytest.fail("Completed adoption cannot perform inference/runtime setup")
    adopted = driver.execute_artisan_component(environment=fixture.environment, runner=forbidden, cost_gate_factory=forbidden)
    assert adopted == result and len(fixture.calls) == 1
    Path(fixture.environment[driver.RESULT]).unlink()
    assert driver.execute_artisan_component(environment=fixture.environment, runner=forbidden, cost_gate_factory=forbidden) == result


def test_interruption_preserves_original_pool_workspace_and_deadline(stage_fixture):
    fixture = stage_fixture
    def interrupted(argv, **kwargs):
        fixture.calls.append((argv, kwargs))
        context = json.loads((fixture.output / "artisan_runtime/bridge_context.v1.json").read_text())
        checkpoint = Path(context["sandbox"]["write_root"]).parent / "pipeline_state.json"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_text("{}")
        raise RuntimeError("synthetic controller interruption before any provider call")
    with pytest.raises(RuntimeError, match="controller interruption"):
        driver.execute_artisan_component(environment=fixture.environment, runner=interrupted, cost_gate_factory=fixture.gate)
    retained = (fixture.output / "artisan_runtime/bridge_context.v1.json").read_bytes()
    result = driver.execute_artisan_component(environment=fixture.environment, runner=fixture.runner, cost_gate_factory=fixture.gate)
    argv = fixture.calls[-1][0]
    assert "resume" in argv and "--prompt-file" not in argv
    assert (fixture.output / "artisan_runtime/bridge_context.v1.json").read_bytes() == retained
    assert fixture.events.count("reserve_original_pool") == 1
    assert result["status"] == "completed"


def test_real_component_output_passes_next_stage_and_changed_artifact_is_refused(stage_fixture):
    from blueprint_pipeline.task_evaluation_scene_configuration_stage_tool import (
        _validate_component_result, TaskEvaluationSceneConfigurationStageToolError,
    )
    fixture = stage_fixture
    result = driver.execute_artisan_component(environment=fixture.environment, runner=fixture.runner, cost_gate_factory=fixture.gate)
    arguments = dict(adapter_id="content_agents_rigid_replacement", stage_id="stage-3",
                     output_root=fixture.output, diagnostic_only=False)
    assert _validate_component_result(result, **arguments) == result["artifacts"]
    assert all("digest" in row and "sha256" not in row for row in result["artifacts"])
    Path(result["artifacts"][0]["path"]).write_bytes(b"changed asset")
    with pytest.raises(TaskEvaluationSceneConfigurationStageToolError, match="component_artifact_invalid"):
        _validate_component_result(result, **arguments)


def test_legacy_completed_hash_keys_are_repaired_without_regeneration_and_original_bytes_retained(stage_fixture):
    fixture = stage_fixture
    result = driver.execute_artisan_component(environment=fixture.environment, runner=fixture.runner, cost_gate_factory=fixture.gate)
    legacy = {**result, "artifacts": [
        {"role": row["role"], "path": row["path"], "sha256": row["digest"], "size_bytes": row["size_bytes"]}
        for row in result["artifacts"]]}
    legacy["result_digest"] = canonical_digest(legacy, digest_field="result_digest")
    original = (json.dumps(legacy, indent=2) + "\n").encode()
    result_path = Path(fixture.environment[driver.RESULT])
    result_path.write_bytes(original)
    fixture.key.unlink()
    shutil.rmtree(fixture.package)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Legacy adoption cannot perform inference or reserve spend")

    assert driver.execute_artisan_component(environment=fixture.environment, runner=forbidden, cost_gate_factory=forbidden) == result
    history = result_path.with_name(f"{result_path.stem}.legacy-sha256-{legacy['result_digest'][7:]}.json")
    assert history.read_bytes() == original
    assert driver.execute_artisan_component(environment=fixture.environment, runner=forbidden, cost_gate_factory=forbidden) == result
    assert len(fixture.calls) == 1 and fixture.events.count("reserve_original_pool") == 1


def test_cached_result_with_conflicting_hash_aliases_is_not_migrated(stage_fixture):
    from blueprint_pipeline.task_object_authoring_contracts import AuthoringContractError
    fixture = stage_fixture
    result = driver.execute_artisan_component(environment=fixture.environment, runner=fixture.runner, cost_gate_factory=fixture.gate)
    result["artifacts"][0]["sha256"] = "sha256:" + "0" * 64
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    result_path = Path(fixture.environment[driver.RESULT])
    result_path.write_text(json.dumps(result))
    original = result_path.read_bytes()
    with pytest.raises(AuthoringContractError, match="authoring_source_bytes_changed"):
        driver.execute_artisan_component(environment=fixture.environment, runner=fixture.runner, cost_gate_factory=fixture.gate)
    assert result_path.read_bytes() == original and not list(result_path.parent.glob("*.legacy-sha256-*.json"))
    assert len(fixture.calls) == 1


def test_explicit_candidate_passes_existing_rigid_static_boundary_and_rejects_changed_reviewed_program(stage_fixture):
    from blueprint_pipeline.task_object_artisan_candidate import qualify_artisan_asset_static
    from blueprint_pipeline.task_object_authoring_contracts import AuthoringContractError
    fixture = stage_fixture
    result = driver.execute_artisan_component(environment=fixture.environment, runner=fixture.runner, cost_gate_factory=fixture.gate)
    paths = {row["role"]: Path(row["path"]) for row in result["artifacts"]}
    receipt = json.loads(paths["replacement_authoring_receipt"].read_text())
    graph = json.loads(paths["replacement_graph_spec"].read_text())
    static = qualify_artisan_asset_static(asset_path=paths["replacement_asset"], graph_spec=graph, authoring_receipt=receipt,
        replacement_identity=receipt["replacement_identity"], output_path=fixture.output / "static.json")
    assert static["authored_structure_statically_qualified"] is True
    assert static["claim_boundary"]["native_simulator_import_qualified"] is False
    Path(receipt["upstream_artifacts"]["program"]["path"]).write_text("changed reviewed program")
    with pytest.raises(AuthoringContractError, match="source_bytes_changed"):
        driver.execute_artisan_component(environment=fixture.environment, runner=fixture.runner, cost_gate_factory=fixture.gate)
    assert len(fixture.calls) == 1


def test_confined_import_preflight_allows_cold_imports_of_the_sealed_runtime(stage_fixture, monkeypatch):
    """The sealed runtime has no bytecode and is read-only, so imports compile cold.

    The release builder measured these imports past 60 s on the 4-core control
    plane under load, the host the CPU prestage runs stage 3 on. A 60 s check
    would refuse a healthy runtime before the first model call.
    """
    from blueprint_pipeline import task_object_artisan_hooks as hooks
    from blueprint_pipeline.asset_authoring_sandbox import SandboxedAssetRunner

    seen = []

    class Preflight:
        def preflight(self):
            pass

        def __call__(self, argv, **kwargs):
            seen.append((argv, kwargs))
            return SimpleNamespace(returncode=0)

    monkeypatch.setattr(hooks, "sandbox", lambda *_: Preflight())
    driver.execute_artisan_component(environment=stage_fixture.environment, runner=stage_fixture.runner,
                                     cost_gate_factory=stage_fixture.gate)
    [(argv, kwargs)] = seen
    assert "import sdk, cadquery" in argv[-1]
    assert kwargs["timeout"] == driver.CONFINED_PREFLIGHT_TIMEOUT_SECONDS >= 300
    # The landlock backend caps any sandboxed call at 900 s; stay inside it.
    assert "min(float(timeout), 900)" in Path(SandboxedAssetRunner.__call__.__code__.co_filename).read_text()
    assert driver.CONFINED_PREFLIGHT_TIMEOUT_SECONDS <= 900


def test_sandbox_can_read_and_load_the_interpreter_it_runs(stage_fixture, monkeypatch):
    """capture-447ae4f4: Kit's Python sits in a tree only its group may read.

    The Landlock launcher clears every capability, so ARTISAN must make that
    runtime readable first (as the Astra backend does) and hand the sandbox
    the interpreter's loader path; otherwise its preflight fails with
    Permission denied: '/isaac-sim/kit/python'.
    """
    import sys
    from blueprint_pipeline import asset_runtime_permissions
    prepared = []
    monkeypatch.setattr(asset_runtime_permissions, "prepare_runtime_code_access",
                        lambda roots: prepared.append([str(p) for p in roots]) or {"status": "ready", "changed_count": 0})
    fixture = stage_fixture
    driver.execute_artisan_component(environment=fixture.environment, runner=fixture.runner, cost_gate_factory=fixture.gate)
    context = json.loads((fixture.output / "artisan_runtime/bridge_context.v1.json").read_text())["sandbox"]
    base = str(Path(sys.base_prefix).resolve())
    assert base in context["read_roots"] and prepared == [context["read_roots"]]
    assert context["library_executables"] == [str(Path(sys.executable).resolve())]
    assert isinstance(context["library_environment"], dict)
    assert json.loads((fixture.output / "artisan_runtime/runtime_code_access.json").read_text())["status"] == "ready"


def test_controller_keeps_the_interpreter_loader_path(stage_fixture, monkeypatch):
    """Kit's Python finds libpython only through its launcher's library path.

    The controller runs this interpreter with a scrubbed environment. Without
    that path it would exit 127 before its first import (lookahead of
    capture-447ae4f4's next GPU run, 2026-11-01).
    """
    from blueprint_pipeline import asset_python_runtime
    real = asset_python_runtime.sandboxed_interpreter()
    monkeypatch.setattr(asset_python_runtime, "sandboxed_interpreter",
                        lambda: {**real, "library_environment": {"LD_LIBRARY_PATH": "/isaac-sim/kit/python/lib"}})
    fixture = stage_fixture
    fixture.environment.update(HTTPS_PROXY="http://proxy.invalid:3128", AWS_SECRET_ACCESS_KEY="ambient-never-inherited")
    driver.execute_artisan_component(environment=fixture.environment, runner=fixture.runner, cost_gate_factory=fixture.gate)
    [(argv, kwargs)] = fixture.calls
    environment = kwargs["env"]
    assert argv[:4] == [str(Path(real["executable"])), "-S", "-m", "blueprint_pipeline.task_object_artisan_bridge"]
    assert environment["LD_LIBRARY_PATH"] == "/isaac-sim/kit/python/lib"
    assert environment["PYTHONPATH"].split(os.pathsep) == driver.package_import_roots(fixture.package.resolve())
    assert environment["HTTPS_PROXY"] == "http://proxy.invalid:3128"
    assert "AWS_SECRET_ACCESS_KEY" not in environment


def _sealed_blender(package):
    sealed = package / "runtime/blender"
    (sealed / "5.0/scripts/startup").mkdir(parents=True)
    (sealed / "5.0/scripts/startup/bl_ui.py").write_text("x = 1\n")
    (sealed / "blender").write_text("#!/bin/sh\n")
    for path in sealed.rglob("*"):
        path.chmod(0o555 if path.is_dir() or path.name == "blender" else 0o444)
    sealed.chmod(0o555)
    return sealed


def test_root_stage_runs_a_private_blender_and_leaves_the_sealed_tree_untouched(stage_fixture, monkeypatch, tmp_path):
    """Root writes past read-only modes; Blender's own bytecode must not reach the toolchain.

    Upstream review renders run Blender from the controller, outside the
    sandbox, and Blender's Python ignores PYTHONDONTWRITEBYTECODE. On the GPU
    host (container root) a ``__pycache__`` in the sealed tree would fail stage
    5's byte-for-byte toolchain revalidation.
    """
    monkeypatch.setattr(driver, "_bypasses_read_only_modes", lambda: True)
    fixture = stage_fixture
    sealed = _sealed_blender(fixture.package)
    before = sorted(str(p.relative_to(sealed)) for p in sealed.rglob("*"))
    driver.execute_artisan_component(environment=fixture.environment, runner=fixture.runner, cost_gate_factory=fixture.gate)
    context = json.loads((fixture.output / "artisan_runtime/bridge_context.v1.json").read_text())["sandbox"]
    private = Path(context["blender"])
    assert private == fixture.output.resolve() / "artisan_runtime/packaged_blender/tree/blender"
    assert str(private.parent) in context["read_roots"] and str(sealed.resolve()) not in context["read_roots"]
    # Linked, not copied: no second 1.8 GB tree on the GPU host's disk.
    assert private.stat().st_ino == (sealed / "blender").stat().st_ino
    cache = private.parent / "5.0/scripts/startup/__pycache__"
    cache.mkdir()
    (cache / "bl_ui.cpython-311.pyc").write_bytes(b"blender-wrote-this")
    assert sorted(str(p.relative_to(sealed)) for p in sealed.rglob("*")) == before
    # The output archive leaves the tree behind.
    from blueprint_pipeline.task_evaluation_scene_configuration_output_archive import write_output_archive
    write_output_archive(fixture.output, tmp_path / "output.zip")
    with zipfile.ZipFile(tmp_path / "output.zip") as archive:
        assert not any("packaged_blender" in name for name in archive.namelist())
        assert any(name.endswith("bridge_context.v1.json") for name in archive.namelist())


def test_unprivileged_stage_runs_the_sealed_blender(stage_fixture, monkeypatch):
    """Where read-only modes hold (the control-plane prestage), nothing is linked."""
    monkeypatch.setattr(driver, "_bypasses_read_only_modes", lambda: False)
    fixture = stage_fixture
    sealed = _sealed_blender(fixture.package)
    driver.execute_artisan_component(environment=fixture.environment, runner=fixture.runner, cost_gate_factory=fixture.gate)
    context = json.loads((fixture.output / "artisan_runtime/bridge_context.v1.json").read_text())["sandbox"]
    assert context["blender"] == str(sealed.resolve() / "blender")
    assert not (fixture.output / "artisan_runtime/packaged_blender").exists()


@pytest.mark.parametrize("changed_frame", [False, True])
def test_a_continued_source_resumes_its_upstream_run_and_meters_only_the_remainder(stage_fixture, monkeypatch, tmp_path,
                                                                                  changed_frame):
    """capture-447ae4f4: the successor resumes the source's upstream run at its saved
    phase; the carried spend counts against the cumulative cap, never against new money."""
    from blueprint_pipeline import task_evaluation_artisan_continuation as continuation
    fixture = stage_fixture
    input_path = Path(fixture.environment[driver.INPUT])
    stage = json.loads(input_path.read_text())
    source_stage = {**stage, "run_id": "source-run"}
    input_path.write_text(json.dumps(source_stage))
    runtime = fixture.output / "artisan_runtime"

    def paid_then_stopped(argv, **kwargs):
        context = json.loads((runtime / "bridge_context.v1.json").read_text())
        workspace = Path(context["sandbox"]["write_root"])
        (workspace.parent / "pipeline_state.json").parent.mkdir(parents=True, exist_ok=True)
        (workspace.parent / "pipeline_state.json").write_text('{"phase": "P5_review"}')
        (workspace / "candidate").mkdir(parents=True, exist_ok=True)
        (workspace / "candidate/object.usd").write_text("#usda 1.0")
        Path(context["usage_file"]).write_text(json.dumps({
            "schema_version": "artisan_upstream_usage.v1", "maximum_cost_usd": context["maximum_cost_usd"],
            "maximum_requests": context["maximum_requests"], "deadline_epoch": context["deadline_epoch"],
            "calls": [{"id": "1", "status": "priced", "cost_usd": 2.5}, {"id": "2", "status": "priced", "cost_usd": 2.5}]}))
        raise RuntimeError("synthetic authority exhausted in review")
    with pytest.raises(RuntimeError, match="authority exhausted"):
        driver.execute_artisan_component(environment=fixture.environment, runner=paid_then_stopped,
                                         cost_gate_factory=fixture.gate)
    source_request = json.loads((runtime / "blueprint_request.v1.json").read_text())
    upstream = "pl_" + source_request["request_digest"][7:39]
    (runtime / "terminal.v1.json").write_text(json.dumps({"run_id": upstream, "phase": "P5_review"}))
    # Pack what selection would carry: the workspace in place, run-bound records relocated.
    archive = tmp_path / "retained_runtime.zip"
    inventory = []
    with zipfile.ZipFile(archive, "w") as packed:
        for path in sorted(p for p in runtime.rglob("*") if p.is_file()):
            relative = path.relative_to(runtime).as_posix()
            kind = continuation._classify(relative)
            if kind is None:
                continue
            name = relative if kind == "keep" else continuation.RELOCATED_ROOT + relative
            packed.write(path, name)
            inventory.append({"relative_path": name, **{k: v for k, v in file_record(path).items() if k != "path"}})
    usage_bytes = (runtime / "blueprint_usage.json").read_bytes()
    shutil.rmtree(runtime)  # a new provider host
    monkeypatch.setattr(continuation, "ORIGINAL_ROOT", str(runtime))
    lineage = {"source_run_id": "source-run", "successor_run_id": stage["run_id"]}
    configuration = stage["configuration"]
    descriptor = {"schema_version": continuation.SCHEMA_VERSION, "source_run_id": "source-run",
                  "successor_run_id": stage["run_id"], "original_runtime_root": str(runtime),
                  "source_request_digest": source_request["request_digest"], "source_upstream_run_id": upstream,
                  "source_terminal_phase": "P5_review",
                  "semantic_request_digest": canonical_digest(continuation.semantic_request(source_request)),
                  "semantic_configuration_digest": canonical_digest(continuation.semantic_configuration(configuration)),
                  "carried_spend": {"cost_usd": 5.0, "call_count": 2, "largest_call_usd": 2.5,
                                    "usage_sha256": "sha256:" + hashlib.sha256(usage_bytes).hexdigest()},
                  "owner_intent_lineage": lineage,
                  "retained_runtime_archive": {k: v for k, v in file_record(archive).items() if k != "path"},
                  "retained_files": sorted(inventory, key=lambda row: row["relative_path"]),
                  "new_spend_authorized": False}
    descriptor["adoption_digest"] = canonical_digest(descriptor, digest_field="adoption_digest")
    descriptor_path = tmp_path / "descriptor.json"
    descriptor_path.write_text(json.dumps(descriptor))
    successor = json.loads(json.dumps(stage))
    successor["construction_envelope"]["artisan_continuation"] = {
        "verified_lineage": lineage,
        "descriptor": {"materialized_path": str(descriptor_path), **file_record(descriptor_path)},
        "runtime_archive": {"materialized_path": str(archive), **file_record(archive)}}
    input_path.write_text(json.dumps(successor))
    reservations = []

    def gate(**kwargs):
        reservations.append(kwargs["max_cost_usd"])
        return fixture.gate(**kwargs)
    if changed_frame:
        # The successor's own stage 2 produced a different view: never resume
        # incompatible work, and never stop the job for it.
        Image.new("RGB", (4, 4), color=(250, 1, 1)).save(fixture.frames[0])
    result = driver.execute_artisan_component(environment=fixture.environment, runner=fixture.runner,
                                              cost_gate_factory=gate)
    argv, _ = fixture.calls[-1]
    if changed_frame:
        assert "resume" not in argv and "--prompt-file" in argv
        assert argv[argv.index("--max-cost-usd") + 1] == "12.0" and reservations == [12.0]
        rejected = json.loads((fixture.output / "artisan_continuation_rejected.json").read_text())
        assert rejected["reason"] == "scientific_inputs_changed" and rejected["fresh_authoring"] is True
        assert (fixture.output / "artisan_continuation_rejected_runtime/workspace").is_dir()
        assert result["status"] == "completed"
        return
    assert argv[argv.index("resume") + 1] == upstream and "--prompt-file" not in argv
    # The configuration's $15 cumulative cap: upstream may reach it; this attempt meters $10.
    assert argv[argv.index("--max-cost-usd") + 1] == "15.0"
    context = json.loads((runtime / "bridge_context.v1.json").read_text())
    assert context["maximum_cost_usd"] == 10.0 and reservations == [10.0]
    record = json.loads((runtime / continuation.RELOCATED_ROOT / "adoption.v1.json").read_text())
    assert record["carried_cost_usd"] == 5.0 and record["increment_usd"] == 10.0
    assert record["successor_request_digest"] != record["source_request_digest"]
    assert result["status"] == "completed"
