"""Hermetic stage orchestration with real final USD; no framework/live proof."""
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import time

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
