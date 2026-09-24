import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_native_background import (
    construction_rights_admission, derived_stage_three_configuration, prepare_construction_stages,
)
from blueprint_pipeline.website_drawer_depth_prior import PRIOR as DRAWER_DEPTH_PRIOR
from blueprint_pipeline.website_native_inputs import INPUT_STATUS, validate_website_native_inputs
from blueprint_pipeline.website_object_observations import _record
from blueprint_pipeline.website_scene_runtime_inputs import prepare_website_runtime_inputs
from blueprint_pipeline.task_evaluation_scene_configuration_render_inputs import materialize_scene_configuration_render_inputs
from blueprint_pipeline.task_evaluation_scene_configuration_source_preflight import validate_scene_configuration_source_preflight
from blueprint_pipeline.task_evaluation_scene_configuration_stage_configuration import validate_immutable_stage_configurations
from blueprint_pipeline.task_evaluation_scene_configuration_disclosure import render_inputs_disclosure_is_coherent
from blueprint_pipeline.task_evaluation_scene_configuration_submission_records import recipe
from blueprint_pipeline.task_evaluation_scene_construction_recipe import validate_scene_construction_recipe
from tests.test_website_native_appearance import inputs


def test_cc48_drawer_depth_prior_is_bound_to_its_own_source_evidence():
    from blueprint_pipeline.website_drawer_depth_prior import CC48_PRIOR, prior_for
    from blueprint_pipeline.task_object_articulated_packaging import derived_website_cabinet_depth_hypothesis

    configuration = {
        "schema_version": "articulated_replacement_authoring_configuration.v1",
        "scene_id": CC48_PRIOR["scene_id"], "replacement_identity": CC48_PRIOR["subject_identity"],
        "source_observation_kind": "website_capture_frames",
        "metric_envelope": {"minimum_xyz_m": [-0.8676723447340478, -0.04660110532186394, 0.75],
                            "maximum_xyz_m": [-0.28922411491134925, 0.04660110532186394, 1.531543853290867]},
        "mechanism": {"joint_type": "prismatic", "estimated_front_normal_world":
                      [0.11635973738420832, -0.9932071342453588, 0.0],
                      "estimated_usable_stroke_m": 0.1199},
    }
    frames = [{"role": "observed_source", "sha256": digest}
              for digest in CC48_PRIOR["original_frame_sha256s"]]
    assert prior_for(scene_id=configuration["scene_id"],
                     subject_identity=configuration["replacement_identity"]) is CC48_PRIOR
    hypothesis = derived_website_cabinet_depth_hypothesis(configuration, frames, 0.6)
    assert hypothesis["estimated_depth_m"] == 0.55
    assert hypothesis["estimated_usable_stroke_m"] == 0.4125
    assert hypothesis["estimated_minimum_opening_m"] == 0.2475
    assert hypothesis["prior_comparison"]["reference_models_are_exact_match"] is False
    assert prior_for(scene_id=configuration["scene_id"],
                     subject_identity=DRAWER_DEPTH_PRIOR["subject_identity"]) is None


def test_exact_drawer_successor_changes_only_stage_three_and_freezes_stroke(tmp_path, monkeypatch):
    from tests.test_task_object_articulated_packaging import _thin_website_cabinet
    from blueprint_pipeline import website_drawer_depth_prior

    original = _thin_website_cabinet()
    original["scene_id"] = DRAWER_DEPTH_PRIOR["scene_id"]
    original["replacement_identity"] = DRAWER_DEPTH_PRIOR["subject_identity"]
    observed = {"schema_version": "website_object_observations.v1",
                "preparation_digest": DRAWER_DEPTH_PRIOR["preparation_digest"],
                "frames": [{"frame_id": str(i), "image_basis": "original_capture",
                            "image": {"digest": digest}}
                           for i, digest in enumerate(DRAWER_DEPTH_PRIOR["original_frame_sha256s"])],
                "digest": ""}
    observed["digest"] = canonical_digest(observed, digest_field="digest")
    manifest_path = tmp_path / "observations.json"
    manifest_path.write_text(json.dumps(observed))
    record = _record(manifest_path)
    prior = copy.deepcopy(DRAWER_DEPTH_PRIOR)
    prior["observation_manifest_digest"] = record["digest"]
    monkeypatch.setattr(website_drawer_depth_prior, "PRIOR", prior)
    runtime = {"object_authoring": {"configuration": original, "observation_manifest": record},
               "stage_one_marker": "completed", "stage_two_marker": "completed"}
    preparation = {"digest": DRAWER_DEPTH_PRIOR["preparation_digest"],
                   "development_test": {"kind": "development_drawer_fixture",
                                        "captured_scene_evaluation_allowed": False},
                   "authoring_inputs": {"source_frames": [
                       {"role": "observed_source", "sha256": digest}
                       for digest in DRAWER_DEPTH_PRIOR["original_frame_sha256s"]]},
                   "intake_request": {"task": {"success": {
                       "minimum_opening_fraction_of_estimated_stroke": 0.6}}}}
    before = json.dumps(runtime, sort_keys=True)
    successor = derived_stage_three_configuration(runtime=runtime, preparation=preparation)
    assert derived_stage_three_configuration(runtime=runtime, preparation=preparation) == successor
    assert json.dumps(runtime, sort_keys=True) == before
    assert successor["metric_envelope"] == original["metric_envelope"]
    assert successor["development_geometry_hypothesis"]["estimated_depth_m"] == 0.55
    assert successor["mechanism"]["joint_limits"] == [0.0, 0.4125]
    assert successor["development_geometry_hypothesis"]["estimated_minimum_opening_m"] == 0.2475
    assert successor["required_output"]["mass_kg_bounds"] == [4.0, 30.0]
    assert successor["required_output"]["task_part_mass_kg_bounds"] == [0.5, 9.0]
    observed["frames"][0]["image"]["digest"] = "sha256:" + "0" * 64
    manifest_path.write_text(json.dumps(observed))
    assert derived_stage_three_configuration(runtime=runtime, preparation=preparation) == original


def test_second_scene_requires_its_own_preparation_observation_and_identity(tmp_path, monkeypatch):
    from tests.test_task_object_articulated_packaging import _thin_website_cabinet
    from blueprint_pipeline import website_drawer_depth_prior
    from blueprint_pipeline.task_object_articulated_packaging import plan_articulated_assembly

    second = copy.deepcopy(DRAWER_DEPTH_PRIOR)
    second.update(scene_id="site-capture-second-development",
                  preparation_digest="sha256:" + "b" * 64,
                  subject_identity={"id": "website-subject-second", "version": "v1"})
    observed = {"schema_version": "website_object_observations.v1",
                "preparation_digest": second["preparation_digest"],
                "frames": [{"frame_id": str(i), "image_basis": "original_capture",
                            "image": {"digest": digest}}
                           for i, digest in enumerate(second["original_frame_sha256s"])],
                "digest": ""}
    observed["digest"] = canonical_digest(observed, digest_field="digest")
    manifest_path = tmp_path / "observations.json"
    write_json(manifest_path, observed)
    record = _record(manifest_path)
    second["observation_manifest_digest"] = record["digest"]
    monkeypatch.setattr(website_drawer_depth_prior, "ADDITIONAL_PRIORS", (second,))

    original = _thin_website_cabinet()
    original.update(scene_id=second["scene_id"], replacement_identity=second["subject_identity"])
    runtime = {"object_authoring": {"configuration": original, "observation_manifest": record}}
    preparation = {"digest": second["preparation_digest"],
                   "development_test": {"kind": "development_drawer_fixture",
                                        "captured_scene_evaluation_allowed": False},
                   "intake_request": {"task": {"success": {
                       "minimum_opening_fraction_of_estimated_stroke": 0.6}}}}
    successor = derived_stage_three_configuration(runtime=runtime, preparation=preparation)
    assert successor["development_geometry_hypothesis"]["estimated_depth_m"] == 0.55
    assert plan_articulated_assembly(successor)["task_joint"]["limits_m"] == [0.0, 0.4125]
    assert derived_stage_three_configuration(
        runtime=runtime, preparation={**preparation, "digest": DRAWER_DEPTH_PRIOR["preparation_digest"]}) == original
    assert derived_stage_three_configuration(
        runtime={"object_authoring": {"configuration": original,
                                      "observation_manifest": {**record, "digest": "sha256:" + "0" * 64}}},
        preparation=preparation) == original
    assert website_drawer_depth_prior.prior_for(
        scene_id=DRAWER_DEPTH_PRIOR["scene_id"],
        subject_identity=DRAWER_DEPTH_PRIOR["subject_identity"]) is website_drawer_depth_prior.PRIOR


def packet(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    args, preparation, _ = inputs(tmp_path)
    preparation_path = args["output_root"] / "preparation.json"
    runtime = prepare_website_runtime_inputs(preparation=preparation, base_scene=args["base_scene"],
        source_geometry=args["source_geometry"], task_masks=args["task_masks"], output_root=tmp_path / "native")
    construction = prepare_construction_stages(runtime_inputs_path=tmp_path / "native/runtime_inputs.json",
                                               preparation_path=preparation_path)
    rights = tmp_path / "rights.json"
    write_json(rights, construction_rights_admission(preparation=preparation,
        task_context=args["task_context"], now=args["now"]))
    request = {"run_id": "website-development", "scene": {"website_native_inputs": {"frames": []},
               "rights": {"provider_disclosure_scope": "derived_only"}}}
    refs = construction["references"] + [
        {"contract_path": "scene.source_manifest", **_record(preparation_path)},
        {"contract_path": "scene.geometry.validation", **_record(Path(runtime["collision"]["normalization_path"]))},
        {"contract_path": "scene.rights.admission", **_record(rights)}]
    rows = []
    for i, row in enumerate(refs):
        reference = {"uri": f"gs://test-bucket/website/{i}{Path(row['path']).suffix}", "digest": row["digest"], "size_bytes": row["size_bytes"]}
        bound = request
        parts = row["contract_path"].split(".")
        for part in parts[:-1]:
            bound = bound.setdefault(part, {})
        if isinstance(bound, list):
            assert int(parts[-1]) == len(bound)
            bound.append(reference)
        else:
            bound[parts[-1]] = reference
        rows.append({"contract_path": row["contract_path"], **reference, "materialized_path": row["path"],
                     "full_byte_service_account_readback_passed": True})
    config_rows, config_refs, configurations = [], [], {}
    for i, config in enumerate(construction["configurations"]):
        path = tmp_path / f"configuration-{i}.json"
        write_json(path, config)
        record = _record(path)
        reference = {"uri": f"gs://test-bucket/configuration/{i}.json", "digest": record["digest"],
                     "size_bytes": record["size_bytes"]}
        config_refs.append(reference)
        config_rows.append({"contract_path": f"construction.recipe.stage_sequence.{i}.configuration", **reference,
                            "materialized_path": record["path"], "full_byte_service_account_readback_passed": True})
        configurations[f"stage-{i + 1}"] = config
    compiled = recipe(recipe_id="website-recipe", team_namespace="website-development",
        scene_identity=construction["scene_identity"], task_identity={"id": "pick-place", "version": "v1"},
        subject_identity=construction["subject_identity"], output_identity={"id": "website-revision", "version": "v1"},
        source_manifest_digest=request["scene"]["source_manifest"]["digest"],
        rights_admission_digest=request["scene"]["rights"]["admission"]["digest"],
        stage_configuration_references=config_refs, supplemental_destination=None)
    for i, stage in enumerate(construction["stage_sequence"]):
        compiled["stage_sequence"][i].update(stage)
    compiled["recipe_digest"] = canonical_digest(compiled, digest_field="recipe_digest")
    validate_scene_construction_recipe(compiled)
    return {"run_id": request["run_id"], "expected_production_commit": "a" * 40, "request": request, "recipe": compiled,
            "materialized_references": rows, "stage_configuration_references": config_rows}, configurations


def test_website_six_stage_inputs_pass_real_preflight_and_need_no_reconstructed_object_render(tmp_path):
    envelope, configs = packet(tmp_path)
    validate_immutable_stage_configurations(envelope=envelope, configurations=configs)
    validate_website_native_inputs(envelope=envelope, configurations=configs, require_render_inputs=False)
    def never(**_):
        pytest.fail("website prepared background must not render an object-present reconstruction")
    render = materialize_scene_configuration_render_inputs(envelope=envelope, stage_one_configuration=configs["stage-1"],
        output_root=tmp_path / "method-inputs", renderer=never, runtime_resolver=never, splat_decoder=never)
    assert render["status"] == INPUT_STATUS
    assert render_inputs_disclosure_is_coherent(render)
    assert render["renderer_qualified"] is False
    envelope["render_inputs_result"] = render
    validate_scene_configuration_source_preflight(envelope=envelope, configurations=configs)


@pytest.mark.parametrize("portable", [False, True])
def test_actual_astra_request_accepts_bound_website_consent_before_gpu(tmp_path, portable):
    from blueprint_pipeline.website_native_inputs import preflight_website_authoring_request
    envelope, configs = packet(tmp_path)
    if portable:
        envelope["stage_configuration_references"] = [
            {"stage_id": stage["stage_id"], **{key: row[key] for key in ("materialized_path", "digest", "size_bytes")}}
            for stage, row in zip(envelope["recipe"]["stage_sequence"], envelope["stage_configuration_references"], strict=True)]
    request = preflight_website_authoring_request(envelope=envelope, configurations=configs)
    assert request.dimension_authority == "estimated"
    assert request.private_provider_processing_allowed is True
    assert request.provider_training_allowed is False
    assert request.public_redistribution_allowed is False
    assert request.physical_review_input.measured.mass_kg is None
    assert len(request.source_frames) == len(envelope["request"]["scene"]["website_native_inputs"]["frames"])


def test_astra_website_disclosure_cannot_substitute_unbound_rights(tmp_path):
    from blueprint_pipeline.website_native_inputs import validate_website_authoring_disclosure
    envelope, configs = packet(tmp_path)
    with pytest.raises(ValueError, match="authoring_disclosure_binding_invalid"):
        validate_website_authoring_disclosure(envelope=envelope, configuration=configs["stage-3"],
            rights={"status": "admitted_for_internal_development", "private_provider_processing_allowed": True,
                    "provider_training_allowed": False, "public_redistribution_allowed": False})


def test_cpu_source_preflight_runs_the_actual_authoring_request_contract(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_configuration_astra_driver as driver
    envelope, configs = packet(tmp_path)
    envelope["render_inputs_result"] = materialize_scene_configuration_render_inputs(
        envelope=envelope, stage_one_configuration=configs["stage-1"], output_root=tmp_path / "method-inputs")
    def reject(*args):
        raise driver.AstraStageError("astra_cad_export_tolerance_unsupported")
    monkeypatch.setattr(driver, "build_authoring_request", reject)
    with pytest.raises(ValueError, match="astra_cad_export_tolerance_unsupported"):
        validate_scene_configuration_source_preflight(envelope=envelope, configurations=configs)


def test_cpu_source_preflight_uses_two_part_authoring_for_drawer(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_configuration_astra_driver as driver
    from blueprint_pipeline.website_native_inputs import preflight_website_authoring_request

    envelope, configs = packet(tmp_path)
    configs["stage-3"]["schema_version"] = driver.ARTICULATED_AUTHORING_SCHEMA_VERSION
    observed = {}

    def articulated(stage_input, source, frames, rights):
        observed.update(stage_input=stage_input, source=source, frames=frames, rights=rights)
        return {"parts": ["carcass", "drawer"]}

    def rigid(*_args):
        pytest.fail("an articulated fixture must not use the one-solid CAD brief")

    monkeypatch.setattr(driver, "build_articulated_authoring_requests", articulated)
    monkeypatch.setattr(driver, "build_authoring_request", rigid)
    assert preflight_website_authoring_request(envelope=envelope, configurations=configs) == {
        "parts": ["carcass", "drawer"]
    }
    assert observed["stage_input"]["configuration"] is configs["stage-3"]
    assert observed["source"]["path"]
    assert len(observed["frames"]) == len(envelope["request"]["scene"]["website_native_inputs"]["frames"])
    assert observed["rights"]["private_provider_processing_allowed"] is True


@pytest.mark.parametrize("change", ["subject", "frame_binding", "legacy_adapter", "render_binding", "configuration"])
def test_preflight_cannot_adopt_different_task_or_input_contract(tmp_path, change):
    envelope, configs = packet(tmp_path)
    render = materialize_scene_configuration_render_inputs(envelope=envelope, stage_one_configuration=configs["stage-1"],
        output_root=tmp_path / "method-inputs")
    envelope["render_inputs_result"] = render
    if change == "subject":
        envelope["recipe"]["subject_identity"]["id"] = "another-object"
    elif change == "frame_binding":
        envelope["request"]["scene"]["website_native_inputs"]["frames"][0]["digest"] = "sha256:" + "f" * 64
    elif change == "legacy_adapter":
        configs["stage-1"]["schema_version"] = "observed_appearance_object_removal_configuration.v1"
    elif change == "render_binding":
        render["website_binding"]["captured_frame_count"] += 1
        render["result_digest"] = canonical_digest(render, digest_field="result_digest")
    else:
        configs["stage-3"]["construction_constraints"]["rebuild_only_this_subject"] = False
    with pytest.raises(ValueError, match="website_native_inputs"):
        validate_scene_configuration_source_preflight(envelope=envelope, configurations=configs)


def test_capture_derivative_upload_requires_its_own_rights_admission(tmp_path):
    envelope, configs = packet(tmp_path)
    row = next(r for r in envelope["materialized_references"] if r["contract_path"] == "scene.rights.admission")
    path = Path(row["materialized_path"])
    rights = json.loads(path.read_text())
    rights["provider_disclosure"]["captured_frame_derivatives_allowed"] = False
    write_json(path, rights)
    row.update(digest=_record(path)["digest"], size_bytes=path.stat().st_size)
    envelope["request"]["scene"]["rights"]["admission"].update(digest=row["digest"], size_bytes=row["size_bytes"])
    envelope["recipe"]["rights_admission_digest"] = row["digest"]
    with pytest.raises(ValueError, match="capture_derivative_disclosure_not_admitted"):
        validate_website_native_inputs(envelope=envelope, configurations=configs, require_render_inputs=False)


@pytest.mark.parametrize("change", ["context", "consent", "expired", "held"])
def test_native_rights_producer_requires_bound_current_authority(tmp_path, change):
    args, preparation, _ = inputs(tmp_path)
    context, now = copy.deepcopy(args["task_context"]), args["now"]
    if change == "context":
        context["capture_rights"]["derived_scene_generation_allowed"] = False
        context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    elif change == "consent":
        preparation["intake_request"]["consent"]["accepted_by"] = "another-owner"
        preparation["digest"] = canonical_digest(preparation, digest_field="digest")
    elif change == "expired":
        now += 90000
    else:
        preparation["status"] = "needs_input"
        preparation["blockers"] = ["task_destination_pose_required"]
        preparation["digest"] = canonical_digest(preparation, digest_field="digest")
    with pytest.raises(ValueError):
        construction_rights_admission(preparation=preparation, task_context=context, now=now)


def test_website_disclosure_cannot_claim_qualified_renders_or_raw_video_upload(tmp_path):
    envelope, configs = packet(tmp_path)
    render = materialize_scene_configuration_render_inputs(envelope=envelope, stage_one_configuration=configs["stage-1"],
        output_root=tmp_path / "method-inputs")
    for key in ("renderer_qualified", "physical_truth_claimed", "raw_capture_video_in_provider_packet", "provider_render_required"):
        changed = copy.deepcopy(render)
        changed[key] = True
        assert not render_inputs_disclosure_is_coherent(changed)


def test_website_bundle_relocates_and_hydrates_actual_capture_inputs(tmp_path, monkeypatch):
    import shutil
    import zipfile
    from blueprint_pipeline.task_evaluation_scene_configuration_bundle import build_scene_configuration_provider_bundle
    from blueprint_pipeline.task_evaluation_scene_configuration_provider_preflight import scene_configuration_bundle_contract
    from blueprint_pipeline.task_evaluation_scene_configuration_adapters import SceneConfigurationAdapterRegistry
    from blueprint_pipeline.task_evaluation_scene_configuration_builtin_adapters import builtin_scene_configuration_adapter_handlers
    from blueprint_pipeline.task_evaluation_scene_configuration_content_agents_driver import _reference_frames
    from scripts.task_evaluation_scene_configuration_provider_runner import _hydrate_envelope
    from tests.astra_toolchain_fixture import astra_toolchain_fixture
    control = tmp_path / "control"
    envelope, configs = packet(control)
    render = materialize_scene_configuration_render_inputs(envelope=envelope, stage_one_configuration=configs["stage-1"],
        output_root=control / "method-inputs")
    commit = "a" * 40
    envelope.update(schema_version="task_evaluation_scene_construction_envelope.v1", expected_production_commit=commit,
        orchestration_id="website-preparation", recipe_digest=envelope["recipe"]["recipe_digest"], render_inputs_result=render)
    envelope["envelope_digest"] = canonical_digest(envelope, digest_field="envelope_digest")
    path = control / "envelope.json"
    write_json(path, envelope)
    built = build_scene_configuration_provider_bundle(construction_envelope_path=path,
        toolchain_root=astra_toolchain_fixture(tmp_path / "toolchain", commit, monkeypatch),
        repository_root=Path(__file__).resolve().parents[1],
        output_root=tmp_path / "bundle", expected_source_commit=commit)
    with zipfile.ZipFile(built["bundle_path"]) as archive:
        _, _, blockers = scene_configuration_bundle_contract(archive)
        assert not blockers
        archive.extractall(tmp_path / "worker")  # Our locally constructed hermetic fixture archive.
    shutil.rmtree(control)
    runtime = tmp_path / "worker/provider_runtime"
    portable = json.loads((runtime / "input/portable_construction_envelope.v1.json").read_text())
    hydrated = _hydrate_envelope(runtime, portable)
    assert hydrated["render_inputs_result"]["status"] == INPUT_STATUS
    assert hydrated["provider_disclosure_receipt"]["captured_frame_derivatives_in_provider_bundle"] is True
    assert hydrated["provider_disclosure_receipt"]["derived_rendered_views_in_provider_bundle"] is False
    registry = SceneConfigurationAdapterRegistry(builtin_scene_configuration_adapter_handlers())
    results = []
    for stage, row in zip(hydrated["recipe"]["stage_sequence"][:2], hydrated["stage_configuration_references"][:2], strict=True):
        config_path = Path(row["materialized_path"])
        results.append(registry.execute(stage=stage, envelope=hydrated, configuration=json.loads(config_path.read_text()),
            configuration_path=config_path, dependency_results=tuple(results), output_root=tmp_path / stage["stage_id"]))
    assert _reference_frames({"configuration": configs["stage-3"]}, results)
