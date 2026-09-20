"""Stage-3 evidence translation, parent admission, and truthful candidate delivery."""
from pathlib import Path
from types import SimpleNamespace
import inspect
import json
import os

import pytest

from blueprint_pipeline import task_evaluation_scene_configuration_astra_driver as driver
from blueprint_pipeline import task_evaluation_scene_configuration_content_agents_driver as legacy
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_object_astra_authoring import validate_request


@pytest.fixture
def retained(tmp_path):
    image, mesh, rights_path = (tmp_path / name for name in ("source.png", "source.usda", "rights.json"))
    image.write_bytes(b"\x89PNG\r\n\x1a\nretained-source")
    mesh.write_text("#usda 1.0\n")
    rights = {"status": "admitted_for_internal_development", "private_provider_processing_allowed": True,
              "provider_training_allowed": False, "public_redistribution_allowed": False}
    rights_path.write_text(json.dumps(rights))
    config = {"schema_version": "rigid_replacement_authoring_configuration.v1", "authoring_backend": driver.BACKEND,
        "replacement_identity": {"id": "source_book", "version": "v1"},
        "authoring_target": "Recreate the owner-selected open book with its visible pages and cover.",
        "source_object_identity": "publisher-instance-1234", "metric_envelope": {
            "minimum_xyz_m": [0.0, 0.0, 0.0], "maximum_xyz_m": [.295304002, .397696028, .0211374],
            "maximum_dimension_relative_error": .05},
        "required_output": {"mass_kg_bounds": [.1, 1.0], "static_friction_bounds": [.3, .8],
            "dynamic_friction_bounds": [.2, .6], "restitution_bounds": [0.0, .15]},
        "provider_disclosure": {"derived_views_and_metric_envelope": True, "provider_training": False,
            "public_redistribution": False}}
    envelope = {"run_id": "shared-stage-run", "expected_production_commit": "a" * 40, "materialized_references": [],
                "envelope_digest": ""}
    envelope["envelope_digest"] = canonical_digest(envelope, digest_field="envelope_digest")
    stage_input = {"schema_version": "task_evaluation_scene_configuration_stage_production_input.v1",
        "run_id": "shared-stage-run", "source_commit": "a" * 40, "construction_source_commit": "a" * 40,
        "configuration_sha256": "sha256:" + "c" * 64, "toolchain_digest": "sha256:" + "d" * 64,
        "stage": {"adapter": {"id": driver._ADAPTER_ID}, "stage_id": "03-author", "execution_class": "gpu_canary"},
        "construction_envelope": envelope, "configuration": config}
    record = driver._file_record(mesh)
    return SimpleNamespace(input=stage_input, config=config, rights=rights, rights_path=rights_path,
                           source=record, image=image, root=tmp_path)


def test_request_preserves_exact_geometry_and_has_no_invented_physical_prior(retained):
    request = driver.build_authoring_request(retained.input, retained.source, [retained.image], retained.rights)
    assert request.dimensions_m == (.295304002, .397696028, .0211374)
    assert request.dimension_uncertainty_m == tuple(d * .05 for d in request.dimensions_m)
    assert request.dimension_authority == "source_geometry"
    assert request.physical_review_input.proposed is None
    assert request.physical_review_input.measured.mass_kg is None
    assert request.physical_review_input.appearance == "unknown"
    assert request.source_frames[0].path == str(retained.image)
    assert request.source_frames[0].sha256 == driver._sha256(retained.image)
    assert "uncertainty proxy" in request.physical_review_input.dimensions.x_m.uncertainty
    assert validate_request(request.model_dump(mode="json")) == request


def test_supplied_primary_evidence_and_measured_values_remain_bound_to_retained_bytes(retained):
    measured = retained.root / "measured.txt"
    measured.write_text("Measured mass 0.35 kg +/- 0.01 kg.")
    digest = driver._sha256(measured)
    retained.input["construction_envelope"]["materialized_references"] = [{"materialized_path": str(measured),
        "digest": digest, "full_byte_service_account_readback_passed": True}]
    evidence = {"evidence_id": "owner_scale", "uri": "https://owner.example/retained-scale-record",
        "sha256": digest.removeprefix("sha256:"), "kind": "physical_measurement", "excerpt": measured.read_text()}
    retained.config["physical_evidence"] = [evidence]
    mass = {"value": .35, "basis": "measured", "interval": {"lower": .34, "upper": .36},
        "rationale": "Owner scale", "uncertainty": "Scale precision", "evidence_ids": ["owner_scale"]}
    retained.config["measured_physical_properties"] = {"mass_kg": mass, "static_friction": None,
        "dynamic_friction": None, "restitution": None}
    request = driver.build_authoring_request(retained.input, retained.source, [retained.image], retained.rights)
    assert request.physical_review_input.measured.mass_kg.model_dump() == mass
    assert request.physical_review_input.evidence[-1].model_dump() == evidence
    measured.write_text("changed")
    with pytest.raises(driver.AstraStageError, match="evidence_digest_mismatch"):
        driver.build_authoring_request(retained.input, retained.source, [retained.image], retained.rights)


@pytest.mark.parametrize("mutation", ["rights", "disclosure", "uncertainty", "owner", "export_tolerance"])
def test_invalid_input_refuses_before_authoring(retained, mutation):
    if mutation == "rights":
        retained.rights["private_provider_processing_allowed"] = False
    elif mutation == "disclosure":
        retained.config["provider_disclosure"]["provider_training"] = True
    elif mutation == "uncertainty":
        retained.config["dimension_uncertainty_m"] = [0, 0, 0]
    elif mutation == "export_tolerance":
        retained.config["maximum_export_error_m"] = .001
    else:
        retained.config["authoring_target"] = ""
    with pytest.raises(driver.AstraStageError):
        driver.build_authoring_request(retained.input, retained.source, [retained.image], retained.rights)


@pytest.mark.parametrize("reasoning_effort", ["medium", "high"])
def test_shared_stage_invoker_denies_extra_calls_wrong_identity_or_unbounded_tools(reasoning_effort):
    seen = []
    invoker = driver._StageInvoker(SimpleNamespace(invoke=lambda *args: seen.append(args)), "shared", 1)
    spec = SimpleNamespace(run_id="other", model="gpt-6-astra", max_turns=1, tool_bindings=(),
                           max_output_tokens=12000, max_input_tokens=80000, reasoning_effort=reasoning_effort)
    with pytest.raises(driver.AstraStageError):
        invoker.invoke(spec, "input")
    spec.run_id = "shared"
    invoker.invoke(spec, "input")
    with pytest.raises(driver.AstraStageError):
        invoker.invoke(spec, "input")
    assert len(seen) == 1


def test_stage_sdk_disables_unreserved_http_retries_and_restores_client(tmp_path):
    from agents.models import _openai_shared
    previous = _openai_shared.get_default_openai_client()
    key = tmp_path / "stage-key"
    key.write_text("fixture-no-network")
    with pytest.raises(RuntimeError, match="fixture failure"):
        with driver._stage_sdk_environment(key):
            client = _openai_shared.get_default_openai_client()
            assert client.max_retries == 0
            assert client.timeout == 600
            assert os.environ["OPENAI_API_KEY_FILE"] == str(key)
            raise RuntimeError("fixture failure")
    assert _openai_shared.get_default_openai_client() is previous


@pytest.fixture
def component(retained, monkeypatch):
    root = retained.root
    output, toolchain, package, blender = (root / name for name in ("output", "toolchain", "toolchain/component", "blender"))
    for path in (output, package, blender):
        path.mkdir(parents=True, exist_ok=True)
    for name in driver._CAD_PACKAGE_FILES:
        (package / name).write_text("sealed-test-fixture")
    input_path, deps_path, result_path, key = (root / name for name in ("input.json", "deps.json", "output/result.json", "stage-key"))
    input_path.write_text(json.dumps(retained.input))
    dependency = {"schema_version": "task_evaluation_scene_configuration_stage_result.v1", "status": "completed",
                  "output_artifacts": [{"role": "source_object_candidate_mesh", **retained.source}], "stage_result_digest": ""}
    dependency["stage_result_digest"] = canonical_digest(dependency, digest_field="stage_result_digest")
    deps_path.write_text(json.dumps([dependency]))
    key.write_text("fake-key-never-sent")
    key.chmod(0o600)
    environment = {driver._INPUT_ENV: str(input_path), driver._DEPENDENCIES_ENV: str(deps_path),
        driver._OUTPUT_ENV: str(output), driver._RESULT_ENV: str(result_path), driver._PACKAGE_ENV: str(package),
        driver.TOOLCHAIN_ROOT_ENV: str(toolchain), driver.BLENDER_ROOT_ENV: str(blender),
        "OPENAI_CONTENT_AGENTS_API_KEY_FILE": str(key), "OPENAI_CONTENT_AGENTS_API_KEY_ID": "fake-id",
        "BLUEPRINT_OPENAI_CONTENT_AGENTS_COST_SCOPE_ATTESTATION_FILE": str(root / "attestation.json"),
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_CONTENT_AGENTS_MAX_COST_USD": "9",
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_COST_USD": "20", "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_REQUESTS": "8"}
    events, seen = [], {}
    monkeypatch.setattr(driver, "_validate_toolchain", lambda **kw: ({"toolchain_digest": retained.input["toolchain_digest"]}, {}))
    monkeypatch.setattr(driver, "_reference_frames", lambda *args: [retained.image])
    monkeypatch.setattr(driver, "_materialized", lambda *args, **kw: (driver._file_record(retained.rights_path), retained.rights_path))
    monkeypatch.setattr(driver.importlib.util, "find_spec", lambda name: object())
    def mac(*args, verified_sources=None, **kwargs):
        raise AssertionError("fake authoring does not execute MAC")
    monkeypatch.setattr(driver, "execute_mac_candidate", mac)
    monkeypatch.setattr(driver, "verify_cad_sources", lambda *args: {})

    def materialize(runtime):
        cadroot = runtime / "cad_authoring"
        for name in ("Multi-Agent-CAD", "text-to-cad"):
            (cadroot / name).mkdir(parents=True)
            (cadroot / name / "source.py").write_text("# pinned source")
        skill = cadroot / "text-to-cad/skills/cad/SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text("# Fixture CAD instructions")
        return {"root": str(cadroot), "receipt_digest": "sha256:" + "e" * 64,
                "source_commits": {"multi-agent-cad": "f" * 40, "text-to-cad": "b" * 40}}

    monkeypatch.setattr(driver, "_materialize_cad_skill_runtime", materialize)

    class Sandbox:
        def __init__(self, **kw): seen["sandbox"] = kw
        def preflight(self): events.append("sandbox")
        def __call__(self, *args, **kw):
            if "--python-expr" in args[0]:
                events.append("blender_preflight")
                seen["blender_probe"] = kw
                (kw["cwd"] / "runtime_probe.blend").write_bytes(b"BLENDER-fixture")
                (kw["cwd"] / "runtime_probe.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")
            else:
                events.append("cad_preflight")
                seen["cad_probe"] = kw
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    class Gate:
        def reserve(self): events.append("reserve")
        def complete(self, **kw):
            events.append("complete")
            seen["completion"] = kw

    def gate(**kw):
        seen["gate"] = kw
        return Gate()
    def invoker(**kw):
        seen["budget"] = kw
        return SimpleNamespace(invoke=lambda *args: events.append("sdk")), SimpleNamespace(manifest=lambda: {"reservation_count": 1})

    def authoring(**kw):
        assert events[-1] == "reserve"
        seen["request"] = kw["request_value"]
        assert os.environ["OPENAI_API_KEY_FILE"] == str(key)
        spec = SimpleNamespace(run_id=retained.input["run_id"], model="gpt-6-astra", max_turns=1,
            tool_bindings=(), max_output_tokens=12000, max_input_tokens=80000, reasoning_effort="high")
        kw["invoker"].invoke(spec, "fake")
        result = {"result_digest": "sha256:" + "2" * 64, "model": "gpt-6-astra"}
        (kw["output_root"] / "result.json").write_text(json.dumps(result))
        return result

    def pack(*, request, authoring_result, output_root, physics_bounds):
        events.append("package")
        asset = output_root / "astra.usdz"
        asset.write_bytes(b"packaged-fixture")
        return {"asset": driver.file_record(asset), "physics_completion": {
            "schema_version": legacy.PHYSICS_COMPLETION_SCHEMA_VERSION,
            "collision_dimensions_m": list(request.dimensions_m), "completion_digest": ""}}

    return SimpleNamespace(environment=environment, events=events, seen=seen, kwargs={"environment": environment,
        "cost_gate_factory": gate, "invoker_factory": invoker, "authoring_executor": authoring,
        "sandbox_factory": Sandbox, "package_candidate": pack,
        "blender_validator": lambda *args, **kw: {"executable": str(blender / "blender"), "version": "5.2.1"}})


def test_stage_reserves_parent_gate_then_seals_existing_roles_without_nvidia_claims(component):
    previous = os.environ.get("OPENAI_API_KEY_FILE")
    result = driver.execute_astra_component(**component.kwargs)
    assert component.events == ["sandbox", "cad_preflight", "blender_preflight", "reserve", "sdk", "complete", "package"]
    assert component.seen["budget"]["maximum_cost_usd"] == 9
    assert component.seen["gate"]["stage"] == "content_agents"
    assert component.seen["budget"]["run_id"] == component.seen["request"]["run_id"] == "shared-stage-run"
    assert component.seen["completion"]["provider_call_performed"] is True
    assert {r["role"] for r in result["artifacts"]} == {"replacement_asset", "replacement_authoring_receipt", "replacement_graph_spec"}
    receipt = json.loads(Path(next(row["path"] for row in result["artifacts"] if row["role"] == "replacement_authoring_receipt")).read_text())
    assert receipt["status"] == "authored_candidate_pending_qualification"
    assert receipt["authoring_backend"] == driver.BACKEND and receipt["model"] == "gpt-6-astra"
    assert "content_agents_runtime_result" not in receipt
    assert receipt["physics_authority_granted"] is False
    assert os.environ.get("OPENAI_API_KEY_FILE") == previous


def test_initial_disclosure_refusal_precedes_model_gate_and_runtime(component, retained):
    rights = json.loads(retained.rights_path.read_text())
    rights["private_provider_processing_allowed"] = False
    retained.rights_path.write_text(json.dumps(rights))
    with pytest.raises(driver.AstraStageError, match="astra_derived_disclosure_not_admitted"):
        driver.execute_astra_component(**component.kwargs)
    assert component.events == []
    assert "gate" not in component.seen and "budget" not in component.seen


def test_stage_preserves_sealed_external_python_runtime_in_sandbox(component, tmp_path, monkeypatch):
    installed = tmp_path / "sealed-python-runtime"
    installed.mkdir()
    monkeypatch.setenv("PYTHONPATH", str(installed))
    driver.execute_astra_component(**component.kwargs)
    assert installed in component.seen["sandbox"]["read_roots"]
    assert str(installed) in component.seen["cad_probe"]["env"]["PYTHONPATH"].split(os.pathsep)


def test_stage_admits_vendor_python_shared_library_paths(component, tmp_path, monkeypatch):
    libs = tmp_path / 'vendor-python-lib'
    libs.mkdir()
    monkeypatch.setenv('LD_LIBRARY_PATH', ':'+str(libs)+':relative:'+str(libs))
    driver.execute_astra_component(**component.kwargs)
    assert libs in component.seen['sandbox']['read_roots']
    paths = component.seen['sandbox']['library_environment']['LD_LIBRARY_PATH'].split(os.pathsep)
    assert paths.count(str(libs)) == 1 and '' not in paths and 'relative' not in paths
    assert component.seen['sandbox']['library_executables'] == [Path(driver.sys.executable)]


def test_stage_finds_kit_libpython_when_parent_loader_environment_is_absent(component, tmp_path, monkeypatch):
    kit = tmp_path / 'kit'
    prefix = kit / 'python'
    prefix.mkdir(parents=True)
    (kit / f'libpython{driver.sys.version_info.major}.{driver.sys.version_info.minor}.so.1.0').write_bytes(b'fixture')
    monkeypatch.setattr(driver.sys, 'base_prefix', str(prefix))
    monkeypatch.delenv('LD_LIBRARY_PATH', raising=False)
    driver.execute_astra_component(**component.kwargs)
    assert component.seen['sandbox']['library_environment']['LD_LIBRARY_PATH'] == str(kit)
    assert kit in component.seen['sandbox']['read_roots']


def test_authoring_failure_closes_parent_cost_receipt(component):
    def failure(**kw): raise RuntimeError("fixture failure before invocation")
    component.kwargs["authoring_executor"] = failure
    with pytest.raises(RuntimeError, match="fixture failure"):
        driver.execute_astra_component(**component.kwargs)
    assert component.events == ["sandbox", "cad_preflight", "blender_preflight", "reserve", "complete"]
    assert component.seen["completion"] == {"provider_call_performed": False, "runtime_result_digest": None,
                                             "runtime_exception_type": "RuntimeError"}


def test_real_authoring_accepts_root_after_runtime_probes(component):
    from blueprint_pipeline.task_object_astra_authoring import execute_asset_authoring

    def authoring(**kwargs):
        root = kwargs['output_root']
        assert {p.name for p in root.iterdir()} <= {'tmp', 'xdg', 'cache'}
        assert not (root / 'tmp' / 'runtime-probe').exists()
        class StopBeforeModel:
            def invoke(self, *args, **kw):
                raise RuntimeError('reached_first_model_boundary')
        kwargs['invoker'] = StopBeforeModel()
        return execute_asset_authoring(**kwargs)

    component.kwargs['authoring_executor'] = authoring
    with pytest.raises(RuntimeError, match='reached_first_model_boundary'):
        driver.execute_astra_component(**component.kwargs)


def test_cad_import_failure_refuses_before_reservation_or_model(component):
    class BrokenSandbox:
        def __init__(self, **kw): pass
        def preflight(self): pass
        def __call__(self, *args, **kw):
            return SimpleNamespace(returncode=1, stdout="", stderr="missing build123d")
    component.kwargs["sandbox_factory"] = BrokenSandbox
    with pytest.raises(driver.AstraStageError, match="sandboxed_cad_runtime_preflight_failed"):
        driver.execute_astra_component(**component.kwargs)
    assert component.events == []
    assert "budget" not in component.seen and "gate" not in component.seen


def test_archive_admission_api_absence_refuses_before_model(component, monkeypatch):
    monkeypatch.setattr(driver, "execute_mac_candidate", lambda *args: None)
    with pytest.raises(driver.AstraStageError, match="mac_archive_admission_unavailable"):
        driver.execute_astra_component(**component.kwargs)
    assert component.events == [] and "budget" not in component.seen


def test_archive_source_verification_refusal_precedes_stage_spend(component, monkeypatch):
    def refuse(*args):
        raise driver.AstraStageError("source archive drift")
    monkeypatch.setattr(driver, "verify_cad_sources", refuse)
    with pytest.raises(driver.AstraStageError, match="source archive drift"):
        driver.execute_astra_component(**component.kwargs)
    assert component.events == [] and "budget" not in component.seen


def test_absent_blender_environment_uses_own_sealed_component_before_stage_spend(component, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_configuration_astra_runtime as materializer
    component.environment.pop(driver.BLENDER_ROOT_ENV)
    observed = []
    def materialize(package_root, destination_root, *, runner):
        component.events.append("packaged_blender")
        observed.append((Path(package_root), Path(destination_root)))
        return {"executable": str(destination_root / "blender"), "version": "5.2.1"}
    def forbidden(*args, **kwargs):
        pytest.fail("absent environment must use packaged materializer")
    monkeypatch.setattr(materializer, "materialize_packaged_blender_runtime", materialize)
    component.kwargs["blender_validator"] = forbidden
    result = driver.execute_astra_component(**component.kwargs)
    assert result["status"] == "completed"
    assert observed[0][0] == Path(component.environment[driver._PACKAGE_ENV])
    assert observed[0][1].name == "packaged_blender"
    assert component.events.index("packaged_blender") < component.events.index("reserve")


def test_backend_dispatch_is_explicit_and_legacy_remains_default(tmp_path, monkeypatch):
    path = tmp_path / "input.json"
    path.write_text(json.dumps({"configuration": {"authoring_backend": driver.BACKEND}}))
    seen = []
    monkeypatch.setattr(driver, "execute_astra_component", lambda **kwargs: seen.append(kwargs) or {"selected": "astra"})
    assert legacy.execute_content_agents_component(environment={driver._INPUT_ENV: str(path)}) == {"selected": "astra"}
    path.write_text(json.dumps({"configuration": {}}))
    with pytest.raises(legacy.TaskEvaluationSceneConfigurationContentAgentsError, match="environment_missing"):
        legacy.execute_content_agents_component(environment={driver._INPUT_ENV: str(path)})
    assert len(seen) == 1


def test_runtime_preflight_uses_same_bootstrap_without_model_calls(tmp_path, monkeypatch):
    package = tmp_path/'package'
    package.mkdir()
    for name in driver._CAD_PACKAGE_FILES:
        (package/name).write_bytes(b'sealed-input')
    observed = []
    monkeypatch.setattr(driver, 'prepare_astra_execution_runtime', lambda **kwargs: observed.append(kwargs))
    monkeypatch.setattr(driver, 'budgeted_invoker', lambda **_: pytest.fail('model call during runtime preflight'))
    output = tmp_path/'preflight'
    driver.preflight_astra_execution_runtime(package=package, output_root=output, environment={})
    assert len(observed) == 1 and observed[0]['package'] == package
    assert all((output/name).read_bytes() == b'sealed-input' for name in driver._CAD_PACKAGE_FILES)
    assert json.loads((output/'runtime_preflight.json').read_text())['model_calls_performed'] == 0


def test_missing_sandboxed_blender_output_refuses_before_model_reservation(component):
    original = component.kwargs["sandbox_factory"]
    class MissingOutput(original):
        def __call__(self, argv, **kwargs):
            if "--python-expr" in argv:
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            return super().__call__(argv, **kwargs)
    component.kwargs["sandbox_factory"] = MissingOutput
    with pytest.raises(driver.AstraStageError, match="sandboxed_blender_runtime_preflight_failed"):
        driver.execute_astra_component(**component.kwargs)
    assert "reserve" not in component.events


def test_stage_invoker_admits_the_cad_output_budget_and_refuses_above_it():
    """The architect emits every section in one reply; 12000 truncated a real part.

    Scene 840938 object 219 planned 13 loft sections, the reply was cut mid-string
    and failed to parse, the coder node never ran, and the graph exported no
    STEP/STL. The boundary must admit the budget the CAD path actually asks for
    and still refuse anything above it.
    """

    assert driver.CAD_MAX_OUTPUT_TOKENS == 20000
    seen = []
    invoker = driver._StageInvoker(
        SimpleNamespace(invoke=lambda *args: seen.append(args)), "run", 4
    )

    def spec(tokens):
        return SimpleNamespace(
            run_id="run", model="gpt-6-astra", max_turns=1, tool_bindings=(),
            max_output_tokens=tokens, max_input_tokens=80000, reasoning_effort="high",
        )

    invoker.invoke(spec(driver.CAD_MAX_OUTPUT_TOKENS), "input")
    invoker.invoke(spec(12000), "input")
    assert len(seen) == 2
    with pytest.raises(driver.AstraStageError):
        invoker.invoke(spec(driver.CAD_MAX_OUTPUT_TOKENS + 1), "input")
    assert len(seen) == 2


def test_cad_output_budget_stays_inside_the_runtime_contract():
    """The runtime refuses a budget over 32000; the CAD path must stay under it."""

    from blueprint_pipeline import astra_cad_skill_runtime as runtime

    source = inspect.getsource(runtime.execute_mac_candidate)
    assert "max_output_tokens <= 32_000" in source
    assert 256 <= driver.CAD_MAX_OUTPUT_TOKENS <= 32_000


def test_cad_output_budget_fits_the_observed_stage_cost_reservation():
    """The budget must not project past the content-agents stage cost cap.

    Reserving `input + tokens * $0.00005` per request, the 2026-09-15 run put
    $0.827/$0.540/$0.316 of input against a $5.00 cap. A budget that projects
    over the cap is refused before any CAD is produced, which costs a whole
    GPU rental and an attempt slot.
    """

    observed_input_usd = (0.82717, 0.53957, 0.31573)
    usd_per_output_token = 0.00005
    stage_cap_usd = 5.00

    projected = sum(
        cost + driver.CAD_MAX_OUTPUT_TOKENS * usd_per_output_token
        for cost in observed_input_usd
    )
    assert projected < stage_cap_usd, f"projects {projected:.2f} over the {stage_cap_usd} cap"
