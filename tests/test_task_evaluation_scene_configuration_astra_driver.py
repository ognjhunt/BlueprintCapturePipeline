"""Stage-3 evidence translation, parent admission, and truthful candidate delivery."""
from pathlib import Path
from types import SimpleNamespace
import inspect
import json
import os
import subprocess

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


def test_articulated_configuration_refuses_before_any_inference(retained):
    stage_input = json.loads(json.dumps(retained.input))
    stage_input["configuration"]["schema_version"] = "articulated_replacement_authoring_configuration.v1"
    with pytest.raises(driver.AstraStageError, match="astra_authoring_configuration_kind_unsupported"):
        driver.build_authoring_request(stage_input, retained.source, [retained.image], retained.rights)


def test_articulated_depth_hypothesis_binds_original_frame_and_source_receipt(retained, monkeypatch):
    from tests.test_task_object_articulated_packaging import _depth_prior, _thin_website_cabinet
    from blueprint_pipeline import website_native_inputs

    monkeypatch.setattr(website_native_inputs, "validate_website_authoring_disclosure", lambda **_: None)
    config = _thin_website_cabinet()
    config.update(authoring_backend=driver.BACKEND,
                  provider_disclosure={"derived_views_and_metric_envelope": True,
                                       "provider_training": False, "public_redistribution": False})
    hypothesis = _depth_prior(config)
    hypothesis["basis"] = "original_capture_frames"
    hypothesis["evidence_frame_sha256s"] = [driver._sha256(retained.image)]
    config["development_geometry_hypothesis"] = hypothesis
    config["mechanism"]["estimated_usable_stroke_m"] = hypothesis["estimated_usable_stroke_m"]
    config["mechanism"]["joint_limits"] = [0.0, hypothesis["estimated_usable_stroke_m"]]
    config["required_output"]["mass_kg_bounds"] = [4.0, 30.0]
    config["required_output"]["task_part_mass_kg_bounds"] = [0.5, 9.0]
    retained.input["configuration"] = config
    plan, requests = driver.build_articulated_authoring_requests(
        retained.input, retained.source, [retained.image], retained.rights)
    assert plan["source_geometry_receipt"]["source_candidate_digest"] == retained.source["digest"]
    assert plan["source_geometry_receipt"]["construction_envelope_digest"] == retained.input["construction_envelope"]["envelope_digest"]
    assert requests["carcass"].dimensions_m[0] == pytest.approx(0.55)
    assert requests["carcass"].dimension_authority == "estimated"
    assert requests["carcass"].physical_review_input.dimensions.x_m.interval.lower == pytest.approx(0.45)
    assert requests["carcass"].physical_review_input.dimensions.x_m.interval.upper == pytest.approx(0.65)
    assert "development_geometry_hypothesis" in requests["carcass"].construction_constraints
    config["development_geometry_hypothesis"]["evidence_frame_sha256s"] = ["sha256:" + "0" * 64]
    with pytest.raises(driver.AstraStageError, match="depth_hypothesis_frame_mismatch"):
        driver.build_articulated_authoring_requests(
            retained.input, retained.source, [retained.image], retained.rights)


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


def test_admitted_32_request_limit_reaches_author_and_reviewers_without_raising_spend(component):
    component.environment["BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_REQUESTS"] = "32"
    component.environment["BLUEPRINT_SCENE_CONFIGURATION_OPENAI_CONTENT_AGENTS_MAX_COST_USD"] = "5"
    original = component.kwargs['authoring_executor']

    def author(**kwargs):
        invoker = kwargs['invoker']
        assert invoker.maximum_calls == 32
        result = original(**kwargs)
        spec = SimpleNamespace(run_id=invoker.run_id, model='gpt-6-astra', max_turns=1,
            tool_bindings=(), max_output_tokens=12000, max_input_tokens=80000, reasoning_effort='medium')
        for _ in range(31):
            invoker.invoke(spec, 'bounded author or independent review')
        with pytest.raises(driver.AstraStageError, match='inference_boundary_refused'):
            invoker.invoke(spec, 'over the configured limit')
        return result

    component.kwargs['authoring_executor'] = author
    driver.execute_astra_component(**component.kwargs)
    assert component.events.count('sdk') == 32
    assert component.seen['budget']['maximum_cost_usd'] == 5
    assert component.seen['gate']['max_cost_usd'] == 5


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


def test_cad_import_timeout_is_bounded_and_reported_without_launcher_arguments(component):
    class SlowSandbox:
        def __init__(self, **kw): pass
        def preflight(self): pass
        def __call__(self, argv, **kw):
            assert kw["timeout"] == 180
            raise subprocess.TimeoutExpired(argv, kw["timeout"])

    component.kwargs["sandbox_factory"] = SlowSandbox
    with pytest.raises(driver.AstraStageError, match="^astra_sandboxed_cad_runtime_preflight_timeout$"):
        driver.execute_astra_component(**component.kwargs)
    failure = Path(component.environment[driver._OUTPUT_ENV]) / "astra_cad_blender_runtime/cad_runtime_preflight_failure.json"
    assert json.loads(failure.read_text()) == {
        "status": "timed_out", "timeout_seconds": 180, "sandboxed_execution": True,
    }
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


def _articulated_configuration():
    from blueprint_pipeline.task_evaluation_scene_configuration_submission_records import (
        articulated_stage_three_configuration,
    )
    from tests.test_task_object_articulated_packaging import MECHANISM, PHYSICS
    return articulated_stage_three_configuration(
        scene_id="scene-1", replacement_identity={"id": "source_cabinet", "version": "v1"},
        source_instance_id="cabinet-1", authoring_target="three-drawer wood cabinet with silver handles",
        source_min=[-0.21, -0.275, 0.0], source_max=[0.21, 0.275, 0.62], dimension_tolerance=0.2,
        physics_bounds=PHYSICS, mechanism=MECHANISM)


def test_articulated_configuration_authors_each_part_and_seals_one_assembly(component, retained):
    """Two bounded part sessions, one composed articulation, checkpointed parts on retry."""
    from tests.test_task_object_articulated_packaging import _part, open_front_shell
    configuration = _articulated_configuration()
    retained.input["configuration"] = configuration
    retained.config.clear()
    retained.config.update(configuration)
    Path(component.environment[driver._INPUT_ENV]).write_text(json.dumps(retained.input))
    authored_parts = []

    def author(**kw):
        assert component.events[-1] in {"reserve", "sdk"}
        request_value = kw["request_value"]
        authored_parts.append(request_value["object_id"])
        spec = SimpleNamespace(run_id=retained.input["run_id"], model="gpt-6-astra", max_turns=1,
            tool_bindings=(), max_output_tokens=12000, max_input_tokens=80000, reasoning_effort="high")
        kw["invoker"].invoke(spec, "fake")
        part_id = request_value["object_id"].rsplit("__", 1)[1]
        mass, density = (12.0, (60.0, 140.0)) if part_id == "carcass" else (2.0, (30.0, 120.0))
        bounds = {"mass_kg": [4.0, 40.0] if part_id == "carcass" else [0.5, 6.0], "static_friction": [0.3, 0.8],
                  "dynamic_friction": [0.2, 0.6], "restitution": [0.0, 0.2]}
        cavities = json.loads(request_value["construction_constraints"]).get(
            "interior_cavities_must_stay_hollow_and_open_front")
        _, result, _ = _part(kw["output_root"] / "fixture", object_id=request_value["object_id"],
                             dimensions=request_value["dimensions_m"], mass_kg=mass, density=density, bounds=bounds,
                             mesh=open_front_shell(request_value["dimensions_m"], cavities) if cavities else None)
        result["request_digest"] = request_value["request_digest"]
        result["model"] = "gpt-6-astra"
        result["result_digest"] = canonical_digest(result, digest_field="result_digest")
        (kw["output_root"] / "result.json").write_text(json.dumps(result))
        return result

    component.kwargs["authoring_executor"] = author
    component.kwargs["package_candidate"] = None  # the real articulated packager composes the USD
    result = driver.execute_astra_component(**component.kwargs)
    assert authored_parts == ["source_cabinet__carcass", "source_cabinet__drawer"]
    assert component.events.count("sdk") == 2 and component.events[-1] == "complete"
    receipt = json.loads(Path(result["artifacts"][1]["path"]).read_text())
    assert receipt["part_models"] == {"carcass": "gpt-6-astra", "drawer": "gpt-6-astra"}
    assert result["asset_kind"] == "articulated_assembly"
    artifacts = {row["role"]: Path(row["path"]) for row in result["artifacts"]}
    assert set(artifacts) == {"replacement_asset", "replacement_authoring_receipt", "replacement_graph_spec"}
    receipt = json.loads(artifacts["replacement_authoring_receipt"].read_text())
    assert receipt["schema_version"] == driver.ARTICULATED_RECEIPT_SCHEMA_VERSION
    assert receipt["model"] == result["model"] == "gpt-6-astra"
    assert receipt["status"] == "authored_candidate_pending_qualification" and receipt["physics_authority_granted"] is False
    assert set(receipt["part_authoring_results"]) == {"carcass", "drawer"}
    completion = receipt["candidate_physics_completion"]
    assert completion["metric_envelope_validation"]["status"] == "within_preregistered_metric_envelope"
    assert completion["task_joint_prim_path"] == "/Asset/joints/task_part_joint"
    graph = json.loads(artifacts["replacement_graph_spec"].read_text())
    assert graph["schema_version"] == driver.ARTICULATED_GRAPH_SCHEMA_VERSION
    assert [j["joint_type"] for j in graph["articulation_graph"]["joints"]] == ["fixed", "prismatic", "fixed"]
    assert graph["fixed_base_body_prim_path"] == "/Asset/links/carcass"
    from pxr import Usd, UsdPhysics
    stage = Usd.Stage.Open(str(artifacts["replacement_asset"]))
    assert stage.GetDefaultPrim().HasAPI(UsdPhysics.ArticulationRootAPI)
    # A second run of the same job adopts both completed parts without a model call.
    events_before = len(component.events)
    result_path = Path(component.environment[driver._RESULT_ENV])
    result_path.unlink()
    again = driver.execute_astra_component(**component.kwargs)
    assert again["asset_kind"] == "articulated_assembly"
    assert component.events[events_before:].count("sdk") == 0
    assert len(authored_parts) == 2


def test_a_refused_part_does_not_stop_the_parts_after_it(component, retained):
    """2026-09-27 website dishwasher: the door's refusal left every later part unbuilt."""
    from blueprint_pipeline.task_object_astra_authoring import AssetAuthoringError
    configuration = _articulated_configuration()
    retained.input["configuration"] = configuration
    retained.config.clear()
    retained.config.update(configuration)
    Path(component.environment[driver._INPUT_ENV]).write_text(json.dumps(retained.input))
    attempted = []

    def author(**kw):
        object_id = kw["request_value"]["object_id"]
        attempted.append(object_id)
        raise AssetAuthoringError("authoring_session_context_ceiling_exceeded" if object_id.endswith("carcass")
                                  else "authoring_independent_review_limit_reached")

    component.kwargs["authoring_executor"] = author
    component.kwargs["package_candidate"] = None
    with pytest.raises(Exception, match="articulated_parts_failed:carcass=authoring_session_context_ceiling_"
                                        "exceeded;drawer=authoring_independent_review_limit_reached"):
        driver.execute_astra_component(**component.kwargs)
    # The first refusal no longer stops the second part from being authored.
    assert attempted == ["source_cabinet__carcass", "source_cabinet__drawer"]
    [failures] = list(retained.root.rglob("part_failures.json"))
    record = json.loads(failures.read_text())
    assert record["assembly_packaged"] is False and set(record["failed_parts"]) == {"carcass", "drawer"}
    assert set(record["part_spend"]) == {"carcass", "drawer"}
    assert {row["status"] for row in record["part_spend"].values()} == {"failed"}


def test_a_budget_refused_part_is_recorded_and_the_stage_still_attempts_the_next(component, retained):
    """2026-09-28: the second part's budget refusal aborted the stage and six parts were never tried."""
    from blueprint_pipeline.task_evaluation_supervisor.agents_sdk import AgentsSDKInvocationBlocked
    configuration = _articulated_configuration()
    retained.input["configuration"] = configuration
    retained.config.clear()
    retained.config.update(configuration)
    Path(component.environment[driver._INPUT_ENV]).write_text(json.dumps(retained.input))
    attempted = []

    def author(**kw):
        object_id = kw["request_value"]["object_id"]
        attempted.append(object_id)
        raise AgentsSDKInvocationBlocked("agents_sdk_inference_budget_ceiling_exceeded")

    component.kwargs["authoring_executor"] = author
    component.kwargs["package_candidate"] = None
    with pytest.raises(Exception, match="articulated_parts_failed:carcass=agents_sdk_inference_budget_ceiling_"
                                        "exceeded;drawer=agents_sdk_inference_budget_ceiling_exceeded"):
        driver.execute_astra_component(**component.kwargs)
    assert attempted == ["source_cabinet__carcass", "source_cabinet__drawer"]
    [failures] = list(retained.root.rglob("part_failures.json"))
    record = json.loads(failures.read_text())
    assert record["part_spend"]["drawer"]["exception_type"] == "AgentsSDKInvocationBlocked"
    assert record["part_spend"]["drawer"]["reason"] == "agents_sdk_inference_budget_ceiling_exceeded"
    # The parent cost gate still closes after the recorded refusals.
    assert component.events[-1] == "complete"


class _PoolInvoker:
    """Reserve the worst case before a call and reconcile to actual after, like the SDK invoker.

    ``input_value`` carries ``(worst_case_usd, actual_usd)`` for the fake call.
    """

    def __init__(self, maximum_cost_usd):
        from blueprint_pipeline.task_evaluation_supervisor.agents_sdk import AgentsSDKInvocationBlocked
        self.blocked = AgentsSDKInvocationBlocked
        self.config = SimpleNamespace(max_inference_cost_usd=maximum_cost_usd)
        self._reserved_cost_usd = 0.0

    def invoke(self, spec, input_value):
        worst, actual = input_value
        if self._reserved_cost_usd + worst > self.config.max_inference_cost_usd:
            raise self.blocked("agents_sdk_inference_budget_ceiling_exceeded")
        self._reserved_cost_usd += actual


def _pool_parts(tmp_path, names, *, maximum_cost_usd, maximum_calls=32, deadline_epoch=None):
    """N generic parts authored against one shared stage pool; no family or object semantics."""
    requests = {name: SimpleNamespace(request_digest="sha256:" + str(i) * 64,
                                      model_dump=lambda mode="json", name=name: {"object_id": "task_object__" + name})
                for i, name in enumerate(names, 1)}
    invoker = driver._StageInvoker(_PoolInvoker(maximum_cost_usd), "pool-run", maximum_calls,
                                   deadline_epoch=deadline_epoch)
    spec = SimpleNamespace(run_id="pool-run", model="gpt-6-astra", max_turns=1, tool_bindings=(),
                           max_output_tokens=12000, max_input_tokens=80000, reasoning_effort="medium")
    authored_root = tmp_path / "authoring"
    authored_root.mkdir()

    def run(calls_by_part, *, before_part=None):
        attempted = []

        def author(**kw):
            name = kw["request_value"]["object_id"].rsplit("__", 1)[1]
            attempted.append(name)
            if before_part:
                before_part(name)
            for call in calls_by_part[name]:
                kw["invoker"].invoke(spec, call)
            return {"model": "gpt-6-astra", "result_digest": "sha256:" + "a" * 64}

        kwargs = dict(plan={"required_parts": []}, part_requests=requests, authored_root=authored_root,
                      runtime=tmp_path / "runtime", prior_roots=[], invoker=invoker, sandbox=None,
                      blender={"executable": "blender"}, cad_root=tmp_path, verified_sources={},
                      authoring_instructions="", configuration={}, authoring_executor=author, mac_executor=None)
        return driver._author_articulated_parts(**kwargs), attempted

    return run, authored_root, invoker


def test_parts_draw_unequally_on_one_shared_stage_pool_and_record_their_spend(tmp_path):
    names = ["part_a", "part_b", "part_c", "part_d", "part_e"]
    run, root, invoker = _pool_parts(tmp_path, names, maximum_cost_usd=25.0)
    # One expensive part, several cheap ones: no part is limited to an equal share.
    calls = {"part_a": [(1.4, 0.3)] * 10, "part_b": [(1.4, 1.0)], "part_c": [(1.4, 0.25)] * 2,
             "part_d": [(1.4, 5.0)] * 2, "part_e": [(1.4, 0.5)]}
    authored, attempted = run(calls)
    assert attempted == names
    spend = authored["part_spend"]
    assert {name: spend[name]["spend_usd"] for name in names} == {
        "part_a": 3.0, "part_b": 1.0, "part_c": 0.5, "part_d": 10.0, "part_e": 0.5}
    assert {name: spend[name]["model_calls"] for name in names} == {
        "part_a": 10, "part_b": 1, "part_c": 2, "part_d": 2, "part_e": 1}
    assert {row["status"] for row in spend.values()} == {"authored"}
    assert spend["part_e"]["pool_remaining_after_usd"] == 10.0
    sealed = json.loads((root / "part_spend.json").read_text())
    assert sealed["pool_sharing"] == "one_shared_stage_pool" and sealed["parts"] == spend
    assert sealed["stage_pool_at_end"]["maximum_cost_usd"] == 25.0
    assert sealed["stage_pool_at_end"]["reserved_cost_usd"] == 15.0
    assert sealed["stage_pool_at_end"]["requests_used"] == invoker.calls == 16
    assert json.loads((root / "result.json").read_text())["part_spend"] == spend
    assert not (root / "part_failures.json").exists()


def test_a_part_refused_by_the_pool_is_recorded_and_later_parts_use_what_is_left(tmp_path):
    names = ["part_a", "part_b", "part_c"]
    run, root, _ = _pool_parts(tmp_path, names, maximum_cost_usd=5.0)
    # part_a leaves $0.50; part_b's $1.40 worst case is refused; part_c's $0.30 one fits.
    calls = {"part_a": [(1.4, 1.5)] * 3, "part_b": [(1.4, 0.2)], "part_c": [(0.3, 0.3)]}
    with pytest.raises(driver.AssetAuthoringError,
                       match="^articulated_parts_failed:part_b=agents_sdk_inference_budget_ceiling_exceeded$"):
        run(calls)
    record = json.loads((root / "part_failures.json").read_text())
    assert record["failed_parts"] == {"part_b": "agents_sdk_inference_budget_ceiling_exceeded"}
    assert record["authored_parts"] == ["part_a", "part_c"] and record["assembly_packaged"] is False
    spend = record["part_spend"]
    assert spend["part_a"]["spend_usd"] == 4.5 and spend["part_a"]["status"] == "authored"
    assert spend["part_b"] | {"elapsed_seconds": 0} == {
        "status": "failed", "reason": "agents_sdk_inference_budget_ceiling_exceeded",
        "exception_type": "AgentsSDKInvocationBlocked", "pool_remaining_before_usd": 0.5,
        # The refused request still counts against the stage request limit; it spent nothing.
        "spend_usd": 0.0, "model_calls": 1, "pool_remaining_after_usd": 0.5, "elapsed_seconds": 0}
    assert spend["part_c"]["status"] == "authored" and spend["part_c"]["spend_usd"] == 0.3
    assert record["stage_pool_at_end"]["remaining_cost_usd"] == 0.2
    assert json.loads((root / "part_spend.json").read_text())["parts"] == spend


def test_a_fully_used_pool_records_every_remaining_part_as_not_attempted(tmp_path):
    names = ["part_a", "part_b", "part_c"]
    run, root, _ = _pool_parts(tmp_path, names, maximum_cost_usd=3.0)
    with pytest.raises(driver.AssetAuthoringError, match="part_b=not_attempted:stage_budget_exhausted;"
                                                         "part_c=not_attempted:stage_budget_exhausted"):
        _, attempted = run({"part_a": [(1.5, 1.5)] * 2, "part_b": [], "part_c": []})
    record = json.loads((root / "part_failures.json").read_text())
    assert record["authored_parts"] == ["part_a"]
    for name in ("part_b", "part_c"):
        assert record["part_spend"][name] == {"status": "not_attempted", "reason": "stage_budget_exhausted",
            "spend_usd": 0.0, "model_calls": 0, "pool_remaining_before_usd": 0.0}


def test_a_spent_request_limit_records_the_remaining_parts_as_not_attempted(tmp_path):
    run, root, _ = _pool_parts(tmp_path, ["part_a", "part_b"], maximum_cost_usd=25.0, maximum_calls=3)
    with pytest.raises(driver.AssetAuthoringError, match="^articulated_parts_failed:"
                                                         "part_b=not_attempted:stage_request_limit_exhausted$"):
        run({"part_a": [(0.1, 0.1)] * 3, "part_b": [(0.1, 0.1)]})
    record = json.loads((root / "part_failures.json").read_text())
    assert record["part_spend"]["part_a"]["model_calls"] == 3
    assert record["stage_pool_at_end"]["requests_used"] == 3


def test_stage_time_stops_the_running_part_and_later_parts_are_recorded(tmp_path, monkeypatch):
    clock = {"now": 1_000.0}
    monkeypatch.setattr(driver, "_now", lambda: clock["now"])
    names = ["part_a", "part_b", "part_c"]
    run, root, _ = _pool_parts(tmp_path, names, maximum_cost_usd=25.0, deadline_epoch=2_000.0)

    def advance(name):
        # part_b starts before the closeout point, then runs past it between calls.
        clock["now"] += 900.0

    with pytest.raises(driver.AssetAuthoringError,
                       match="^articulated_parts_failed:part_b=astra_stage_time_exhausted;"
                             "part_c=not_attempted:stage_time_exhausted$"):
        run({"part_a": [(0.2, 0.2)], "part_b": [(0.2, 0.2)], "part_c": [(0.2, 0.2)]}, before_part=advance)
    record = json.loads((root / "part_failures.json").read_text())
    spend = record["part_spend"]
    assert spend["part_a"]["status"] == "authored" and spend["part_a"]["spend_usd"] == 0.2
    assert spend["part_b"]["status"] == "failed" and spend["part_b"]["exception_type"] == "AstraStageError"
    assert spend["part_b"]["model_calls"] == 0 and spend["part_b"]["spend_usd"] == 0.0
    assert spend["part_c"]["reason"] == "stage_time_exhausted"
    assert record["stage_pool_at_end"]["deadline_epoch"] == 2_000.0


def test_stage_deadline_is_the_producer_deadline_less_closeout_and_fails_closed():
    from blueprint_pipeline.task_evaluation_scene_configuration_runtime_budget import (
        ASTRA_AUTHORING_CLOSEOUT_RESERVE_SECONDS, STAGE_DEADLINE_EPOCH_ENV,
    )
    assert driver._authoring_deadline_epoch({}) is None
    assert driver._authoring_deadline_epoch({STAGE_DEADLINE_EPOCH_ENV: "10000"}) == (
        10000 - ASTRA_AUTHORING_CLOSEOUT_RESERVE_SECONDS)
    for raw in ("soon", "nan", "inf"):
        with pytest.raises(driver.AstraStageError, match="astra_stage_deadline_invalid"):
            driver._authoring_deadline_epoch({STAGE_DEADLINE_EPOCH_ENV: raw})


@pytest.mark.parametrize("articulated", [True, False])
def test_only_articulated_authoring_binds_the_stage_clock(component, retained, articulated):
    from blueprint_pipeline.task_evaluation_scene_configuration_runtime_budget import STAGE_DEADLINE_EPOCH_ENV
    component.environment[STAGE_DEADLINE_EPOCH_ENV] = str(driver._now() + 7_800)
    if articulated:
        configuration = _articulated_configuration()
        retained.input["configuration"] = configuration
        retained.config.clear()
        retained.config.update(configuration)
        Path(component.environment[driver._INPUT_ENV]).write_text(json.dumps(retained.input))
    seen = []

    def author(**kw):
        seen.append(kw["invoker"].deadline_epoch)
        raise driver.AssetAuthoringError("fixture_stop")

    component.kwargs["authoring_executor"] = author
    component.kwargs["package_candidate"] = None
    with pytest.raises(driver.AssetAuthoringError):
        driver.execute_astra_component(**component.kwargs)
    expected = float(component.environment[STAGE_DEADLINE_EPOCH_ENV]) - 600
    assert seen and all((value == expected) if articulated else value is None for value in seen)


def test_stage_cap_is_the_shared_astra_constant_not_a_literal(component):
    from blueprint_pipeline.task_evaluation_scene_configuration_runtime_budget import MAX_ASTRA_AUTHORING_SPEND_USD
    assert MAX_ASTRA_AUTHORING_SPEND_USD == 25.0
    source = inspect.getsource(driver)
    assert "min(15.0" not in source and source.count("min(MAX_ASTRA_AUTHORING_SPEND_USD") == 2
    component.environment["BLUEPRINT_SCENE_CONFIGURATION_OPENAI_CONTENT_AGENTS_MAX_COST_USD"] = "30"
    component.environment["BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_COST_USD"] = "40"
    driver.execute_astra_component(**component.kwargs)
    assert component.seen["budget"]["maximum_cost_usd"] == 25.0
    assert component.seen["gate"]["max_cost_usd"] == 25.0


# Computed on origin/main 8f22c8801 with this exact fixture (paths normalized,
# request_digest dropped): a legacy drawer's plan and part requests must not
# move, or an already-bought carcass is bought again.
LEGACY_DRAWER_GOLDEN = {
    "plan": "sha256:5f264447d01a683f03e8add84e88c17ad326e17c2ae602bf4155343840cc39f2",
    "carcass": "sha256:745142dc8f6a68411a4ff8343cb15cae81414625741af3fa3f08ae5d502f8d85",
    "drawer": "sha256:9ccdd4ea76e43c23a59a564be5db148541e3b2157e38319443178446ade18d35",
}


def test_legacy_drawer_plan_and_part_requests_match_origin_main(retained):
    retained.input["configuration"] = _articulated_configuration()
    assert "assembly_family" not in retained.input["configuration"]
    plan, requests = driver.build_articulated_authoring_requests(
        retained.input, retained.source, [retained.image], retained.rights)

    def normalized(value):
        return canonical_digest(json.loads(json.dumps(value).replace(str(retained.root), "<root>")))

    observed = {"plan": normalized(plan)}
    for part_id, request in requests.items():
        value = request.model_dump(mode="json")
        value.pop("request_digest")
        observed[part_id] = normalized(value)
    assert observed == LEGACY_DRAWER_GOLDEN
    assert "interior_cavities_must_stay_hollow_and_open_front" not in requests["carcass"].construction_constraints


def test_articulated_configuration_refuses_rigid_phase_adoption(component, retained):
    configuration = _articulated_configuration()
    configuration["astra_phase_adoption"] = {"prior_runtime": "/nonexistent", "schema_version": "x"}
    retained.input["configuration"] = configuration
    Path(component.environment[driver._INPUT_ENV]).write_text(json.dumps(retained.input))
    with pytest.raises(driver.AstraStageError, match="astra_articulated_phase_adoption_unsupported"):
        driver.execute_astra_component(**component.kwargs)


def test_completed_articulated_successor_bypasses_model_and_cad_runtime(component, retained, monkeypatch):
    from blueprint_pipeline import task_evaluation_partial_astra_successor as successor
    configuration = _articulated_configuration()
    configuration["authoring_agent_runtime"] = "openai_agents_api"
    retained.input["configuration"] = configuration
    descriptor = retained.root / "descriptor.json"
    archive = retained.root / "retained.zip"
    archive.write_bytes(b"retained-fixture")
    primary = Path(component.environment[driver._OUTPUT_ENV]) / "astra_cad_blender_runtime"
    descriptor.write_text(json.dumps({"adoption_kind": "completed_articulated_agents_api",
        "original_runtime_root": str(primary), "adoption_digest": "sha256:" + "a" * 64}))

    def reference(path):
        record = driver.file_record(path)
        return {"materialized_path": str(path), "digest": record["sha256"],
                "size_bytes": record["size_bytes"]}

    envelope = retained.input["construction_envelope"]
    envelope["partial_astra_successor"] = {"descriptor": reference(descriptor),
        "runtime_archive": reference(archive), "verified_lineage": {"source_run_id": "source"}}
    envelope["envelope_digest"] = canonical_digest(envelope, digest_field="envelope_digest")
    Path(component.environment[driver._INPUT_ENV]).write_text(json.dumps(retained.input))
    observed = {}

    def restore(**kwargs):
        observed["restored"] = kwargs
        kwargs["original_root"].mkdir(parents=True)

    def prepare(**kwargs):
        observed["prepared"] = kwargs
        return {"source_part_requests": kwargs["part_requests"],
                "authored": {"model": "gpt-6-sol", "parts": {}},
                "lineage": {"new_provider_calls": 0}}

    def finish(**kwargs):
        observed["finished"] = kwargs
        return {"status": "completed", "adopted": True}

    monkeypatch.setattr(successor, "restore_partial_astra", restore)
    monkeypatch.setattr(successor, "prepare_completed_articulated_successor", prepare)
    monkeypatch.setattr(driver, "_finish_articulated_component", finish)
    monkeypatch.setattr(driver, "prepare_astra_execution_runtime",
                        lambda **_kwargs: pytest.fail("CAD runtime was repeated"))
    component.kwargs["authoring_executor"] = lambda **_kwargs: pytest.fail("model authoring was repeated")
    assert driver.execute_astra_component(**component.kwargs)["adopted"] is True
    assert observed["restored"]["original_root"] == primary
    assert observed["finished"]["adoption_lineage"]["new_provider_calls"] == 0
    assert observed["finished"]["part_requests"] == observed["prepared"]["part_requests"]


def _dishwasher_stage(retained, monkeypatch):
    from tests.test_articulated_hinged_door_appliance import _frame, dishwasher
    from blueprint_pipeline import website_native_inputs

    monkeypatch.setattr(website_native_inputs, "validate_website_authoring_disclosure", lambda **_: None)
    from PIL import Image
    # Real bounded PNGs: frames are checked against the provider's request bound.
    first, second = retained.root / "closed.png", retained.root / "open.png"
    Image.new("RGB", (64, 48), (200, 200, 200)).save(first)
    Image.new("RGB", (64, 48), (90, 90, 90)).save(second)
    frames = [
        _frame("f_closed", driver._sha256(first), "closed", ["door_outer", "handle", "control_panel"]),
        _frame("f_open", driver._sha256(second), "open", ["tub_interior", "upper_rack", "lower_rack"],
               view="front-high", reason="interior observed with the door open")]
    config = dishwasher(frames=frames)
    config.update(authoring_backend=driver.BACKEND,
                  provider_disclosure={"derived_views_and_metric_envelope": True,
                                       "provider_training": False, "public_redistribution": False})
    retained.input["configuration"] = config
    return config, [first, second]


def test_hinged_appliance_briefs_caption_each_frame_from_its_reference_row(retained, monkeypatch):
    config, references = _dishwasher_stage(retained, monkeypatch)
    plan, requests = driver.build_articulated_authoring_requests(
        retained.input, retained.source, references, retained.rights)
    assert set(requests) == {"body", "door", "upper_rack", "lower_rack", "cutlery_basket"}
    frames = requests["door"].source_frames
    assert [frame.sha256 for frame in frames] == [driver._sha256(path) for path in references]
    assert "dishwasher door is closed" in frames[0].description
    assert "Door outer panel, Door handle, Control panel" in frames[0].description
    assert "is open" in frames[1].description and "Stainless tub interior" in frames[1].description
    assert "interior observed with the door open" in frames[1].description
    brief = json.dumps([request.model_dump(mode="json") for request in requests.values()]).lower()
    assert "wood-grain" not in brief and "silver bar handles" not in brief and "interior is unobserved" not in brief
    body = json.loads(requests["body"].construction_constraints)
    assert body["assembly_family"] == "hinged_door_appliance" and body["hinge_edge"] == "bottom"
    assert body["interior_cavities_must_stay_hollow_and_open_front"] == plan["interior_cavities"]
    assert "bay_count" not in body
    door = json.loads(requests["door"].construction_constraints)
    assert {row["part_id"] for row in door["required_parts_on_this_part"]} == {
        "door_outer", "door_inner", "handle", "control_panel", "brand_label"}
    assert "Door handle" in requests["door"].physical_review_input.material_description
    bounds = driver._articulated_physics_bounds(config, plan)
    assert bounds["upper_rack"]["mass_kg"] == [0.3, 4.0] and bounds["door"]["mass_kg"] == [2.0, 12.0]
    del config["required_output"]["fixed_part_mass_kg_bounds"]
    with pytest.raises(driver.AstraStageError, match="astra_articulated_fixed_part_mass_bounds_invalid"):
        driver.build_articulated_authoring_requests(retained.input, retained.source, references, retained.rights)


def _cabinet_stage(retained, monkeypatch, estimates=None):
    """A side-hinged storage cabinet with two fixed shelves, each seen in a different frame."""
    from tests.test_articulated_part_evidence import cabinet_frames, storage_cabinet
    from blueprint_pipeline import website_native_inputs

    monkeypatch.setattr(website_native_inputs, "validate_website_authoring_disclosure", lambda **_: None)
    from PIL import Image
    references = []
    for index, shade in enumerate((200, 120, 60)):
        path = retained.root / f"cabinet_{index}.png"
        Image.new("RGB", (64, 48), (shade, shade, shade)).save(path)
        references.append(path)
    frames = [{**row, "sha256": driver._sha256(path)} for row, path in zip(cabinet_frames(), references)]
    config = storage_cabinet(frames=frames, estimates=estimates)
    config.update(authoring_backend=driver.BACKEND,
                  provider_disclosure={"derived_views_and_metric_envelope": True,
                                       "provider_training": False, "public_redistribution": False})
    retained.input["configuration"] = config
    return config, references


def test_each_part_brief_leads_with_the_frames_that_show_it(retained, monkeypatch):
    """2026-09-27 website capture: every part was briefed with the same whole-object frames."""
    _config, references = _cabinet_stage(retained, monkeypatch)
    plan, requests = driver.build_articulated_authoring_requests(
        retained.input, retained.source, references, retained.rights)
    digests = [driver._sha256(path) for path in references]
    for request in requests.values():  # The same bounded frame set for every part; only order and captions differ.
        assert sorted(frame.sha256 for frame in request.source_frames) == sorted(digests)
    upper = requests["upper_shelf"]
    assert [frame.sha256 for frame in upper.source_frames] == [digests[1], digests[0], digests[2]]
    assert upper.source_frames[0].description.startswith("Shows this part (Upper shelf): reproduce its observed")
    assert all(frame.description.startswith("Whole-object context; this part is not visible here.")
               for frame in upper.source_frames[1:])
    lower = requests["lower_shelf"]
    assert [frame.sha256 for frame in lower.source_frames] == [digests[2], digests[0], digests[1]]
    # A part carried by a link is shown by any frame naming it: the door leads with both frames showing it.
    assert [frame.sha256 for frame in requests["door"].source_frames] == [digests[0], digests[2], digests[1]]
    constraints = json.loads(upper.construction_constraints)
    evidence = constraints["part_evidence"]
    assert evidence["frames_showing_this_part"] == ["f_upper"]
    assert evidence["whole_object_context_frames"] == ["f_front", "f_lower"]
    assert evidence["dimension_bases"]["upper_shelf"]["basis"] == "template_prior"
    description = upper.physical_review_input.object_description
    assert "Frames f_upper show this part. Reproduce this part's observed construction, pattern, spacing, " \
           "colours and materials from those frames" in description
    assert "template prior of the assembly family, not observed" in description
    for axis in ("x_m", "y_m", "z_m"):
        dimension = getattr(upper.physical_review_input.dimensions, axis)
        assert dimension.rationale.startswith("template_prior:")
        assert dimension.evidence_ids == ["retained_source_geometry"]
    # The body and door keep their envelope rationale; neither is a template-placed part.
    assert requests["body"].physical_review_input.dimensions.x_m.rationale.startswith("Part envelope derived")
    for part_id, feature in (("body", "left_side"), ("door", "handle")):
        assert json.loads(requests[part_id].construction_constraints)["part_evidence"]["dimension_bases"] == {
            feature: _brief_basis(plan, feature)}


def _brief_basis(plan, part_id):
    """A plan basis as a brief carries it: link, feature and frames already travel on the part rows."""
    return {key: value for key, value in plan["part_dimension_bases"][part_id].items()
            if key not in {"link_id", "feature", "appearance_frame_ids"}}


def test_part_frame_evidence_is_family_agnostic_and_leaves_frameless_configurations_alone():
    from blueprint_pipeline.task_object_articulated_packaging import assembly_contract, plan_articulated_assembly
    from tests.test_articulated_hinged_door_appliance import _frame

    config = _articulated_configuration()
    frames = [_frame("f_front", "sha256:" + "a" * 64, "closed", ["carcass", "middle_drawer"]),
              _frame("f_top", "sha256:" + "b" * 64, "open", ["top_drawer"], view="top_down")]
    config.update(assembly_family="stacked_drawer_cabinet", reference_frames=frames,
                  source_observation_kind="website_capture_frames", required_parts=[
                      {"part_id": "carcass", "label": "Cabinet body", "role": "body", "observed_frame_ids": ["f_front"]},
                      {"part_id": "middle_drawer", "label": "Middle drawer", "role": "task_part",
                       "observed_frame_ids": ["f_front"]},
                      {"part_id": "top_drawer", "label": "Top drawer", "role": "fixed_interior",
                       "observed_frame_ids": ["f_top"]}])
    plan = plan_articulated_assembly(config)
    contract = assembly_contract(config, "stacked_drawer_cabinet")
    source = [{"path": f"/f/{row['frame_id']}.png", "sha256": row["sha256"], "role": "observed_source",
               "description": row["frame_id"]} for row in frames]
    carcass = [row for row in plan["required_parts"] if row["link_id"] == "carcass"]
    ordered, evidence = driver.part_frame_evidence(source, contract, carcass, plan["part_dimension_bases"])
    assert [row["description"].rsplit(" ", 1)[1] for row in ordered] == ["f_front", "f_top"]
    assert evidence["frames_showing_this_part"] == ["f_front"] and "dimension_bases" not in evidence
    # The shared drawer solid is instanced in every bay, so any frame naming a drawer shows it.
    links = {row["link_id"] for row in plan["links"] if row["part_id"] == "drawer"}
    drawer = [row for row in plan["required_parts"] if row["link_id"] in links]
    ordered, evidence = driver.part_frame_evidence(source, contract, drawer, plan["part_dimension_bases"])
    assert evidence["frames_showing_this_part"] == ["f_front", "f_top"]
    assert evidence["dimension_bases"]["top_drawer"]["basis"] == "template_prior"
    # Without contract frames (a legacy drawer), frames pass through untouched and no evidence is added.
    legacy = assembly_contract(_articulated_configuration(), "stacked_drawer_cabinet")
    assert driver.part_frame_evidence(source, legacy, drawer, {}) == (source, None)


def test_observed_part_box_sizes_its_brief_with_its_own_basis_and_uncertainty(retained, monkeypatch):
    from blueprint_pipeline.task_object_articulated_packaging import plan_articulated_assembly
    from tests.test_articulated_part_evidence import _estimate, _link_box, storage_cabinet

    lo, hi = _link_box(plan_articulated_assembly(storage_cabinet()), "lower_shelf")
    box = ([lo[0], lo[1], lo[2] + 0.02], [hi[0] - 0.05, hi[1], lo[2] + 0.05])
    _config, references = _cabinet_stage(retained, monkeypatch,
                                         estimates=[_estimate("lower_shelf", ["f_lower"], *box, (0.02, 0.01, 0.015))])
    plan, requests = driver.build_articulated_authoring_requests(
        retained.input, retained.source, references, retained.rights)
    lower = requests["lower_shelf"]
    assert lower.dimensions_m == pytest.approx([b - a for a, b in zip(*box)], abs=1e-5)
    assert list(lower.dimension_uncertainty_m) == [0.02, 0.01, 0.015]
    dimension = lower.physical_review_input.dimensions.z_m
    assert dimension.rationale.startswith("observed_estimate_from_frames: frames f_lower, clamped inside the tub")
    assert dimension.evidence_ids == ["retained_source_geometry", "retained_source_view_2"]
    assert dimension.interval.lower == pytest.approx(lower.dimensions_m[2] - 0.015)
    basis = json.loads(lower.construction_constraints)["part_evidence"]["dimension_bases"]["lower_shelf"]
    assert basis["basis"] == "observed_estimate_from_frames" and basis["frame_ids"] == ["f_lower"]
    assert basis == _brief_basis(plan, "lower_shelf") and basis["uncertainty_m"] == [0.02, 0.01, 0.015]
    assert "estimated from frames f_lower" in lower.physical_review_input.object_description
    # The unestimated sibling stays an explicit prior.
    assert requests["upper_shelf"].physical_review_input.dimensions.x_m.rationale.startswith("template_prior:")


def test_hinged_appliance_frames_must_match_the_retained_frames(retained, monkeypatch):
    _config, references = _dishwasher_stage(retained, monkeypatch)
    with pytest.raises(driver.AstraStageError, match="reference_frames_disagree_with_retained_frames"):
        driver.build_articulated_authoring_requests(retained.input, retained.source, references[:1], retained.rights)


def test_created_object_without_frames_plans_but_refuses_an_observed_brief(retained, monkeypatch):
    config, _references = _dishwasher_stage(retained, monkeypatch)
    config.update(source_observation_kind="not_captured_created_from_description", reference_frames=[],
                  body_depth={"value_m": 0.58, "basis": "owner_or_catalog_specified", "frame_ids": []})
    for row in config["required_parts"]:
        row["observed_frame_ids"] = []
    assert driver.articulated_frame_descriptions(config, []) == []
    with pytest.raises(driver.AstraStageError, match="created_object_authoring_unsupported"):
        driver.build_articulated_authoring_requests(retained.input, retained.source, [], retained.rights)


def test_finish_refuses_thin_slab_envelope_and_uncarried_required_parts(tmp_path):
    from tests.test_articulated_hinged_door_appliance import dishwasher
    from blueprint_pipeline.task_object_articulated_packaging import plan_articulated_assembly, required_parts_by_link

    config = dishwasher()
    plan = plan_articulated_assembly(config)
    output = (tmp_path / "out").resolve()

    def attempt(completion):
        def package(**_):
            output.mkdir(parents=True, exist_ok=True)
            asset = output / "asset.usdz"
            asset.write_bytes(b"usdz")
            return {"asset": driver._file_record(asset), "physics_completion": completion}
        driver._finish_articulated_component(
            plan=plan, part_requests={}, authored={"parts": {}}, output=output, physics_bounds={},
            configuration=config, source_record=None, stage_input=None, rights_record=None, cad_runtime=None,
            blender=None, authored_root=None, result_path=None, package_candidate=package)

    base = {"schema_version": driver.ARTICULATED_COMPLETION_SCHEMA_VERSION,
            "required_parts_by_link": required_parts_by_link(plan)}
    with pytest.raises(driver.AstraStageError, match="astra_articulated_assembly_envelope_mismatch"):
        attempt({**base, "collision_dimensions_m": [0.16, 0.6, 0.85]})
    carried = required_parts_by_link(plan)
    del carried["door"]["handle"]
    with pytest.raises(driver.AstraStageError, match="required_part_unplanned:handle"):
        attempt({**base, "required_parts_by_link": carried, "collision_dimensions_m": [0.61, 0.6, 0.85]})
