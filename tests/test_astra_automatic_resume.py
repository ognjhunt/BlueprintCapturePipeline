"""Restart the real authoring orchestration using only completed retained phases."""
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import task_object_astra_authoring as author
from blueprint_pipeline import task_evaluation_scene_configuration_astra_driver as driver
from blueprint_pipeline import task_evaluation_scene_configuration_astra_phase_adoption as adoption
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_supervisor.inference_reservations import InferenceReservationAudit
from blueprint_pipeline.task_object_physical_property_review import PhysicalPropertyReviewProposal
from tests.test_task_evaluation_scene_configuration_astra_driver import retained, component  # noqa: F401


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class FixtureInvoker:
    def __init__(self, request, budget, *, fail=None, first_review_passed=True):
        self.request, self.budget, self.fail = request, budget, fail
        self.first_review_passed = first_review_passed
        self.calls = []

    def invoke(self, spec, inputs):
        capability = spec.capability.removeprefix(self.request.object_id + "_")
        self.calls.append(capability)
        if capability == self.fail:
            raise RuntimeError("interrupted before the next provider reservation")
        if capability == "source_analysis":
            output = author.VisualBrief(object_identity="book", observed_parts=["pages"], appearance_requirements=["opaque"],
                unknown_regions=["underside"], cad_brief_markdown="Exact source dimensions", proposed_material="paper", proposed_appearance="opaque")
        elif capability == "physical_property_review":
            def value(number, low, high):
                return dict(value=number, basis="estimated", interval=dict(lower=low, upper=high),
                    rationale="Fixture estimate", uncertainty="Fixture interval", evidence_ids=["retained_source_geometry"])
            mass = math.prod(self.request.dimensions_m) * 200
            output = PhysicalPropertyReviewProposal.model_validate(dict(object_id=self.request.object_id,
                dimensions=self.request.physical_review_input.dimensions.model_dump(mode="json"),
                properties=dict(mass_kg=value(mass, mass*.9, mass*1.1), static_friction=value(.5,.4,.6),
                    dynamic_friction=value(.3,.25,.35), restitution=value(.05,0,.1)),
                optical_material=dict(name="paper", transmission=0, opacity=1),
                mass_model=dict(method="density_fill", density_kg_m3=dict(lower=180,upper=220),
                    envelope_fill_fraction=dict(lower=.95,upper=1), sheet_count=None, sheet_area_m2=None,
                    grammage_g_m2=None, cover_mass_kg=None, rationale="Fixture", uncertainty="Fixture range", evidence_ids=["retained_source_geometry"]),
                review_rationale="Independent fixture review"))
        elif capability.startswith("blender_author"):
            output = author.BlenderProgram(program="# fixture geometry\n", explanation="fixture", generated_surface_assumptions=["unseen underside"])
        else:
            output = author.AppearanceReview(source_object_recognizable=True, source_color_and_material_preserved=True,
                opaque_surfaces_opaque=True, required_parts_present=self.first_review_passed or capability.endswith("_1"),
                no_obvious_geometry_artifacts=True, blockers=[], repair_instructions="", unobserved_surface_limitations=["underside"])
        policy = {"policy_digest": "sha256:" + "b" * 64}
        identity = {"run_id": spec.run_id, "capability": spec.capability, "model": "gpt-6-astra",
            "input_digest": canonical_digest(inputs), "max_turns": 1, "max_output_tokens": 12000,
            "cache_policy_digest": policy["policy_digest"]}
        reservation = {**identity, "schema_version": "task_evaluation_inference_reservation.v1",
            "reservation_id": canonical_digest(identity), "projected_max_cost_usd": 1.4,
            "cache_policy": policy, "breakpoint_digests": []}
        reservation["inference_reservation_digest"] = canonical_digest(reservation, digest_field="inference_reservation_digest")
        completion = {"schema_version": "task_evaluation_inference_completion.v1", "reservation_id": reservation["reservation_id"],
            "run_id": spec.run_id, "capability": spec.capability, "model": "gpt-6-astra", "provider": "openai",
            "cache_policy": policy, "breakpoint_digests": [], "projected_max_cost_usd": 1.4,
            "reconciled_actual_cost_usd": .1, "released_reservation_usd": 1.3,
            "structured_output_digest": canonical_digest(output.model_dump(mode="json"))}
        completion["inference_completion_digest"] = canonical_digest(completion, digest_field="inference_completion_digest")
        audit = InferenceReservationAudit(run_root=self.budget, run_id=spec.run_id)
        audit.record_reservation(reservation)
        audit.record_completion(completion)
        return SimpleNamespace(output=output, model="gpt-6-astra", provider="openai", usage={}, cost_usd=.1, cost_status="fixture")


@pytest.fixture
def authoring_fixture(retained, component):  # noqa: F811 - shared pytest fixtures
    request = driver.build_authoring_request(retained.input, retained.source, [retained.image], retained.rights)
    runtime = Path(component.environment[driver._OUTPUT_ENV]) / "astra_cad_blender_runtime"
    runtime.mkdir()
    package = Path(component.environment[driver._PACKAGE_ENV])
    for name in adoption.SOURCE_ARCHIVES:
        (runtime / name).write_bytes((package / name).read_bytes())
    source = {**retained.source, "role": "source_object_candidate_mesh"}
    _save(runtime / "stage_source_binding.json", driver._stage_source_binding(request, retained.input, source, retained.rights_path))
    executed = []
    def cad(*, brief, output_root, dimensions_m):
        executed.append("cad")
        output_root.mkdir()
        step, stl = output_root / "part.step", output_root / "part.stl"
        step.write_text("fixture exact STEP")
        stl.write_text("fixture exact STL")
        readback = {"passed": True, "valid": True, "solid_count": 1,
            "measured_dimensions_mm": [v*1000 for v in dimensions_m], "expected_dimensions_mm": [v*1000 for v in dimensions_m],
            "absolute_tolerance_mm": request.maximum_export_error_m*1000, "volume_mm3": math.prod(dimensions_m)*1e9,
            "build123d": "fixture", "kernel_versions": {"fixture": "1"}}
        _save(output_root / "step-readback.json", readback)
        result = {"passed": True, "source_unchanged_after": True, "readback": readback,
            "step": author.file_record(step), "stl": author.file_record(stl), "step_path": str(step), "stl_path": str(stl),
            "artifacts": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output_root.iterdir()}}
        _save(output_root / "candidate-receipt.json", result)
        return result
    class Blender:
        def preflight(self): pass
        def __call__(self, argv, *, cwd, **kwargs):
            executed.append("blender")
            cwd = Path(cwd)
            for name in ("candidate.usdc", "candidate.blend", "perspective.png", "top.png", "side.png", "final_visual_mesh.json"):
                (cwd / name).write_bytes(b"fixture diagnostic bytes")
            _save(cwd / "geometry_readback.json", {"dimensions_m": list(request.dimensions_m), "minimum_z_m": 0,
                "center_xy_m": [0,0], "materials": [{"alpha": 1,"transmission": 0}]})
            receipt = {"schema_version": "final_visual_mesh_receipt.v1",
                "source_cad_stl_sha256": author.file_record(cwd / "candidate.stl")["sha256"],
                "author_program_sha256": author.file_record(cwd / "asset_program.py")["sha256"],
                "candidate_usd_sha256": author.file_record(cwd / "candidate.usdc")["sha256"],
                "mesh_sha256": author.file_record(cwd / "final_visual_mesh.json")["sha256"]}
            receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
            _save(cwd / "final_visual_mesh_receipt.json", receipt)
            return SimpleNamespace(returncode=0, stdout="fixture", stderr="")
    return SimpleNamespace(request=request, runtime=runtime, package=package, component=component,
        cad=cad, blender=Blender(), executed=executed)


def execute(fixture, invoker, root, **kwargs):
    return author.execute_asset_authoring(request_value=fixture.request.model_dump(mode="json"), output_root=root,
        invoker=invoker, mac_executor=fixture.cad, blender_runner=fixture.blender, blender_executable="fixture", **kwargs)


@pytest.mark.parametrize("failure", ["physical_property_review", "independent_visual_review_0"])
def test_automatic_resume_skips_completed_models_cad_and_blender_with_unknown_appearance(authoring_fixture, failure):
    f = authoring_fixture
    invoker = FixtureInvoker(f.request, f.runtime / "inference", fail=failure)
    with pytest.raises(RuntimeError, match="interrupted"):
        execute(f, invoker, f.runtime / "authoring")
    original_physical_input = (f.runtime / "authoring/physical_review_input.json").read_bytes()
    prior_inventory = adoption._inventory(f.runtime)
    descriptor = adoption.materialize_automatic_phase_adoption(prior_runtime=f.runtime)
    resumed = f.runtime.parent / "resume"
    prepared = adoption.prepare_phase_adoption(value=descriptor, request_value=f.request.model_dump(mode="json"),
        package=f.package, budget_root=resumed / "inference")
    next_invoker = FixtureInvoker(f.request, resumed / "inference")
    result = execute(f, next_invoker, resumed / "authoring", **prepared["authoring_kwargs"])
    assert result["status"] == "candidate_authored_pending_native_qualification"
    assert f.executed.count("cad") == 1
    assert f.executed.count("blender") == 1
    assert "source_analysis" not in next_invoker.calls
    if failure.startswith("independent"):
        assert next_invoker.calls == ["independent_visual_review_0"]
    assert (resumed / "authoring/physical_review_input.json").read_bytes() == original_physical_input
    assert adoption._inventory(f.runtime) == prior_inventory


def test_no_cost_production_replay_automatically_reuses_completed_authoring(authoring_fixture):
    f = authoring_fixture
    execute(f, FixtureInvoker(f.request, f.runtime / "inference"), f.runtime / "authoring")
    def forbidden(*args, **kwargs):
        pytest.fail("completed authoring must not invoke model, CAD, Blender, sandbox, or external cost gate")
    args = {**f.component.kwargs, "no_cost_replay": True, "cost_gate_factory": forbidden,
        "invoker_factory": forbidden, "sandbox_factory": forbidden, "blender_validator": forbidden, "authoring_executor": forbidden}
    result = driver.execute_astra_component(**args)
    assert result["status"] == "completed"
    assert f.executed == ["cad", "blender"]
    assert any(path.name == "no_cost_authoring_adoption.json" for path in f.runtime.parent.rglob("*.json"))


def test_resuming_the_second_visual_round_does_not_reset_the_repair_allowance(authoring_fixture):
    f = authoring_fixture
    with pytest.raises(RuntimeError, match="interrupted"):
        execute(f, FixtureInvoker(f.request, f.runtime / "inference", fail="independent_visual_review_1", first_review_passed=False),
                f.runtime / "authoring")
    descriptor = adoption.materialize_automatic_phase_adoption(prior_runtime=f.runtime)
    resumed = f.runtime.parent / "second-round-resume"
    prepared = adoption.prepare_phase_adoption(value=descriptor, request_value=f.request.model_dump(mode="json"),
        package=f.package, budget_root=resumed / "inference")
    invoker = FixtureInvoker(f.request, resumed / "inference")
    execute(f, invoker, resumed / "authoring", **prepared["authoring_kwargs"])
    assert invoker.calls == ["independent_visual_review_1"]
    assert not (resumed / "authoring/appearance-00").exists()
    assert f.executed == ["cad", "blender", "blender"]


def test_no_cost_replay_refuses_incomplete_authoring_without_any_runtime_call(authoring_fixture):
    f = authoring_fixture
    with pytest.raises(RuntimeError):
        execute(f, FixtureInvoker(f.request, f.runtime / "inference", fail="physical_property_review"), f.runtime / "authoring")
    with pytest.raises(driver.AstraStageError, match="requires_completed_authoring"):
        driver.execute_astra_component(**f.component.kwargs, no_cost_replay=True)
    assert f.executed == ["cad"]


def test_completed_authoring_is_reusable_after_an_interrupted_packaging_attempt(authoring_fixture):
    f = authoring_fixture
    execute(f, FixtureInvoker(f.request, f.runtime / "inference"), f.runtime / "authoring")
    calls = []
    original = f.component.kwargs["package_candidate"]
    def package(**kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("packaging interruption")
        return original(**kwargs)
    args = {**f.component.kwargs, "package_candidate": package, "no_cost_replay": True}
    with pytest.raises(RuntimeError, match="packaging interruption"):
        driver.execute_astra_component(**args)
    result = driver.execute_astra_component(**args)
    assert result["status"] == "completed"
    assert f.executed == ["cad", "blender"]
    assert len(calls) == 2


def test_no_cost_replay_preserves_the_original_stage_root(authoring_fixture, tmp_path):
    from blueprint_pipeline.task_evaluation_astra_authoring_replay import replay_completed_astra_authoring
    f = authoring_fixture
    execute(f, FixtureInvoker(f.request, f.runtime / "inference"), f.runtime / "authoring")
    stage = f.runtime.parent
    for key, name in ((driver._INPUT_ENV, "stage_production_input.v1.json"), (driver._DEPENDENCIES_ENV, "dependency_results.v1.json")):
        (stage / name).write_bytes(Path(f.component.environment[key]).read_bytes())
    before = adoption._inventory(f.runtime)
    def invoke(**kwargs):
        return driver.execute_astra_component(**{**f.component.kwargs, **kwargs})
    result = replay_completed_astra_authoring(retained_stage_root=stage, replay_root=tmp_path / "no-cost-replay",
        toolchain_root=Path(f.component.environment[driver.TOOLCHAIN_ROOT_ENV]), component_root=f.package, executor=invoke)
    assert result["status"] == "completed"
    assert adoption._inventory(f.runtime) == before
    assert f.executed == ["cad", "blender"]


def test_automatic_replay_refuses_changed_rights_receipt_even_if_booleans_still_allow_it(authoring_fixture):
    f = authoring_fixture
    execute(f, FixtureInvoker(f.request, f.runtime / "inference"), f.runtime / "authoring")
    binding = json.loads((f.runtime / "stage_source_binding.json").read_text())
    rights = Path(binding["rights_admission"]["path"])
    value = json.loads(rights.read_text()) | {"additional_unbound_admission": "different grant"}
    rights.write_text(json.dumps(value))
    with pytest.raises(driver.AstraStageError, match="source_or_rights_binding_changed"):
        driver.execute_astra_component(**f.component.kwargs, no_cost_replay=True)
    assert f.executed == ["cad", "blender"]
