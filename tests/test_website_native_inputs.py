import contextlib
import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_native_background import (
    construction_rights_admission, derived_stage_three_configuration, prepare_construction_stages,
)
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


# The three development drawer fixtures that held exact-scene prior records before
# the prior became a cabinet category prior. Their sealed hypotheses carry
# ``website_drawer_depth_prior.v1`` and the category values; both stay unchanged.
_RETAINED_FRAMES = [
    "sha256:0205303c4faa9892114349af39bc5908b07c145dcc58eaeedd9b5fb74e4336af",
    "sha256:7c711762f672a41fd90803b4c5c658a81ee23e9dc3b0fcc71dd57ed243bfcc4a",
    "sha256:aba7b9e216a351510bf07b16bc8362ec0a68306c94306bbf705f60a0e8874157",
    "sha256:afba0d20718721e00312255a4be823a42af42f36e6640598008eee2e2bbe01f9",
    "sha256:b6f03d21cf7d274c7cc4228fca289e242e9eeb347f97986c4af39d2e00812a25",
]
# (scene, preparation, observation manifest, subject) and the canonical digest of the
# exact-scene record it held (frames in sorted order), computed before the change.
_RETAINED_BINDINGS = [
    ("site-capture-7d655e25-34f2-4961-ae9c-6da11621ce8f-development",
     "sha256:cfec5e38e056748c73d2d93660aae81c1f703e81f6bf8041dd88a3914d64cdb6",
     "sha256:028cc3a1ff6c0093d13850859e3220bede6b2ef919a0a985570d858b56e91681",
     "website-subject-0300250645e1def96386",
     "sha256:0ac1a1092a0216f499154b1ae4d71bccd79015b020db934d41940cd59a826fae"),
    ("site-capture-bc15f409-09b7-438c-9891-519ba24d728f-development",
     "sha256:2e1254a642a9a3a77de7f4ae421cc59e68ca8ef8b75da6d14617207afaba1f59",
     "sha256:5d780038e8d92460645b15bccf2146a3dcc2436dd230ea1b236dfdb07a494e73",
     "website-subject-cc9b93603a73eb7cbace",
     "sha256:7763d75a9dc568041402c473ef3729a3a2e4fed56dc111d93238812dc536f898"),
    ("site-capture-cc484026-f6ba-4b9e-9a16-aca0a0163b8d-development",
     "sha256:6943e1f45cab6f7764a52a60e00ad4e24c8d974ab40c1161e0b878f7a2caabf7",
     "sha256:4d3d7498376ca741fe0a11dcf1b80a0a9fe4d1a39a0f6fa788892d34a45475c2",
     "website-subject-aa9aaf64b6fe08ffe2bd",
     "sha256:b5379aafeecd559b2a00e324cf6808a5637aecc0400f763fb643121a66959a28"),
]
# The cc48 fixture's re-derived hypothesis, computed before the change.
_RETAINED_CC48_HYPOTHESIS_DIGEST = "sha256:6777df5c671f4416f3d65bdbdc653b905df30124a6dccbb85c6d612dd665dc9d"


def _cc48_configuration():
    scene_id, _, _, subject, _ = _RETAINED_BINDINGS[2]
    return {
        "schema_version": "articulated_replacement_authoring_configuration.v1",
        "scene_id": scene_id, "replacement_identity": {"id": subject, "version": "v1"},
        "source_observation_kind": "website_capture_frames",
        "metric_envelope": {"minimum_xyz_m": [-0.8676723447340478, -0.04660110532186394, 0.75],
                            "maximum_xyz_m": [-0.28922411491134925, 0.04660110532186394, 1.531543853290867]},
        "mechanism": {"joint_type": "prismatic", "estimated_front_normal_world":
                      [0.11635973738420832, -0.9932071342453588, 0.0],
                      "estimated_usable_stroke_m": 0.1199},
    }


@pytest.mark.parametrize("binding", _RETAINED_BINDINGS, ids=["7d65", "bc15", "cc48"])
def test_retained_drawer_fixture_records_are_unchanged_by_the_category_prior(binding):
    from blueprint_pipeline.website_drawer_depth_prior import prior_for

    scene_id, preparation_digest, observation_digest, subject, record_digest = binding
    record = prior_for({"scene_id": scene_id, "replacement_identity": {"id": subject, "version": "v1"},
                        "mechanism": {"joint_type": "prismatic"}},
                       preparation_digest=preparation_digest, observation_manifest_digest=observation_digest,
                       original_frame_sha256s=reversed(_RETAINED_FRAMES))
    assert record["schema_version"] == "website_drawer_depth_prior.v1"
    assert canonical_digest(record) == record_digest


def test_cabinet_prior_binds_each_scene_and_uses_no_site_readings():
    from blueprint_pipeline.website_drawer_depth_prior import prior_for
    from blueprint_pipeline.task_object_articulated_packaging import (
        derived_website_cabinet_depth_hypothesis, plan_articulated_assembly)
    from blueprint_pipeline.task_object_authoring_contracts import AssetAuthoringError
    from tests.test_task_object_articulated_packaging import _thin_website_cabinet

    configuration = _cc48_configuration()
    frames = [{"role": "observed_source", "sha256": digest} for digest in _RETAINED_FRAMES]
    hypothesis = derived_website_cabinet_depth_hypothesis(configuration, frames, 0.6)
    # A retained fixture re-derives byte for byte the hypothesis its stage 3 sealed.
    assert canonical_digest(hypothesis) == _RETAINED_CC48_HYPOTHESIS_DIGEST
    # The single video and public cabinet specifications only: no site-reported size enters.
    assert hypothesis["estimated_depth_m"] == 0.55
    assert hypothesis["estimated_usable_stroke_m"] == 0.4125
    assert hypothesis["estimated_minimum_opening_m"] == 0.2475
    assert "owner_reported_dimensions_m" not in hypothesis["prior_comparison"]
    assert hypothesis["prior_comparison"]["reference_models_are_exact_match"] is False
    # The same cabinet in any other scene gets the same prior, bound to that scene.
    elsewhere = {**configuration, "scene_id": "site-capture-elsewhere-development",
                 "replacement_identity": {"id": "website-subject-elsewhere", "version": "v1"}}
    record = prior_for(elsewhere, preparation_digest="sha256:" + "c" * 64,
                       observation_manifest_digest="sha256:" + "d" * 64,
                       original_frame_sha256s=["sha256:" + "f" * 64, "sha256:" + "e" * 64])
    assert {key: record[key] for key in ("scene_id", "preparation_digest", "observation_manifest_digest",
                                         "subject_identity", "original_frame_sha256s")} == {
        "scene_id": "site-capture-elsewhere-development", "preparation_digest": "sha256:" + "c" * 64,
        "observation_manifest_digest": "sha256:" + "d" * 64,
        "subject_identity": {"id": "website-subject-elsewhere", "version": "v1"},
        "original_frame_sha256s": ["sha256:" + "e" * 64, "sha256:" + "f" * 64]}
    assert record["claim_ceiling"] == "development_only" and record["nominal_depth_m"] == 0.55
    assert record["limitations"] == ("Comparison models differ from the filmed cabinet; rear geometry and"
                                     " physical mass remain unmeasured.")
    assert derived_website_cabinet_depth_hypothesis(elsewhere, frames, 0.6) == hypothesis
    # Unknown binding stays unknown, and a caller's edit never reaches the next record.
    unbound = prior_for(elsewhere)
    assert unbound["preparation_digest"] is None and unbound["original_frame_sha256s"] == []
    unbound["manufacturer_examples"].clear()
    assert len(prior_for(elsewhere)["manufacturer_examples"]) == 4
    # Not a drawer cabinet, or nothing to bind to: no prior.
    hinged = {**elsewhere, "assembly_family": "hinged_door_appliance",
              "mechanism": {**elsewhere["mechanism"], "joint_type": "revolute"}}
    rigid = {key: value for key, value in elsewhere.items() if key != "mechanism"}
    assert prior_for(hinged) is None and prior_for(rigid) is None
    assert prior_for({**elsewhere, "assembly_family": "sliding_tray"}) is None
    assert prior_for({**elsewhere, "scene_id": ""}) is None
    assert prior_for({**elsewhere, "replacement_identity": None}) is None
    assert derived_website_cabinet_depth_hypothesis(hinged, frames, 0.6) is None
    authored = _thin_website_cabinet()
    authored.update(scene_id=configuration["scene_id"], replacement_identity=configuration["replacement_identity"],
                    metric_envelope=configuration["metric_envelope"])
    authored["mechanism"]["estimated_front_normal_world"] = configuration["mechanism"]["estimated_front_normal_world"]
    authored["mechanism"]["estimated_usable_stroke_m"] = 0.4125
    authored["mechanism"]["joint_limits"] = [0.0, 0.4125]
    authored["required_output"]["mass_kg_bounds"] = [4.0, 30.0]
    authored["required_output"]["task_part_mass_kg_bounds"] = [0.5, 9.0]
    authored["development_geometry_hypothesis"] = hypothesis
    plan = plan_articulated_assembly(authored)
    # Width and height stay the video's own; only the unseen depth comes from the prior.
    assert plan["assembly_dimensions_m"] == {
        "depth_x": 0.55, "width_y": plan["source_geometry"]["projected_width_m"],
        "height_z": plan["source_geometry"]["height_m"], "authority": "development_only_depth_hypothesis"}
    stale = {**hypothesis, "prior_comparison": {**hypothesis["prior_comparison"],
                                                 "owner_reported_dimensions_m": {"depth": 0.508}}}
    with pytest.raises(AssetAuthoringError, match="articulated_cabinet_depth_hypothesis_invalid"):
        plan_articulated_assembly({**authored, "development_geometry_hypothesis": stale})


def _drawer_fixture(tmp_path, *, scene_id, preparation_digest, frames):
    from tests.test_task_object_articulated_packaging import _thin_website_cabinet

    observed = {"schema_version": "website_object_observations.v1",
                "preparation_digest": preparation_digest,
                "frames": [{"frame_id": str(i), "image_basis": "original_capture", "image": {"digest": digest}}
                           for i, digest in enumerate(frames)],
                "digest": ""}
    observed["digest"] = canonical_digest(observed, digest_field="digest")
    manifest_path = tmp_path / "observations.json"
    write_json(manifest_path, observed)
    original = _thin_website_cabinet()
    original.update(scene_id=scene_id,
                    replacement_identity={"id": "website-subject-" + scene_id, "version": "v1"})
    runtime = {"object_authoring": {"configuration": original, "observation_manifest": _record(manifest_path)},
               "stage_one_marker": "completed", "stage_two_marker": "completed"}
    preparation = {"digest": preparation_digest,
                   "development_test": {"kind": "development_drawer_fixture",
                                        "captured_scene_evaluation_allowed": False},
                   "intake_request": {"task": {"success": {
                       "minimum_opening_fraction_of_estimated_stroke": 0.6}}}}
    return original, runtime, preparation, observed, manifest_path


def test_drawer_successor_changes_only_stage_three_and_freezes_stroke(tmp_path):
    original, runtime, preparation, observed, manifest_path = _drawer_fixture(
        tmp_path, scene_id="site-capture-first-development", preparation_digest="sha256:" + "a" * 64,
        frames=_RETAINED_FRAMES)
    before = json.dumps(runtime, sort_keys=True)
    successor = derived_stage_three_configuration(runtime=runtime, preparation=preparation)
    assert derived_stage_three_configuration(runtime=runtime, preparation=preparation) == successor
    assert json.dumps(runtime, sort_keys=True) == before
    assert successor["metric_envelope"] == original["metric_envelope"]
    assert successor["development_geometry_hypothesis"]["estimated_depth_m"] == 0.55
    assert successor["development_geometry_hypothesis"]["evidence_frame_sha256s"] == _RETAINED_FRAMES
    assert successor["mechanism"]["joint_limits"] == [0.0, 0.4125]
    assert successor["development_geometry_hypothesis"]["estimated_minimum_opening_m"] == 0.2475
    assert successor["required_output"]["mass_kg_bounds"] == [4.0, 30.0]
    assert successor["required_output"]["task_part_mass_kg_bounds"] == [0.5, 9.0]
    observed["frames"][0]["image"]["digest"] = "sha256:" + "0" * 64
    manifest_path.write_text(json.dumps(observed))
    assert derived_stage_three_configuration(runtime=runtime, preparation=preparation) == original


def test_any_scene_drawer_fixture_derives_only_from_its_own_evidence(tmp_path):
    from blueprint_pipeline.task_object_articulated_packaging import plan_articulated_assembly

    frames = ["sha256:" + "e" * 64, "sha256:" + "f" * 64]
    original, runtime, preparation, observed, manifest_path = _drawer_fixture(
        tmp_path, scene_id="site-capture-second-development", preparation_digest="sha256:" + "b" * 64,
        frames=frames)
    successor = derived_stage_three_configuration(runtime=runtime, preparation=preparation)
    assert successor["development_geometry_hypothesis"]["estimated_depth_m"] == 0.55
    assert successor["development_geometry_hypothesis"]["evidence_frame_sha256s"] == frames
    assert plan_articulated_assembly(successor)["task_joint"]["limits_m"] == [0.0, 0.4125]
    record = runtime["object_authoring"]["observation_manifest"]
    # Another preparation's observations, a changed manifest, a frame that is not an
    # original capture, a captured-scene run or a subject that is not a drawer cabinet
    # each keep the original configuration.
    assert derived_stage_three_configuration(
        runtime=runtime, preparation={**preparation, "digest": "sha256:" + "c" * 64}) == original
    assert derived_stage_three_configuration(
        runtime={"object_authoring": {"configuration": original,
                                      "observation_manifest": {**record, "digest": "sha256:" + "0" * 64}}},
        preparation=preparation) == original
    assert derived_stage_three_configuration(
        runtime=runtime, preparation={**preparation, "development_test": {
            "kind": "development_drawer_fixture", "captured_scene_evaluation_allowed": True}}) == original
    hinged = {**original, "assembly_family": "hinged_door_appliance",
              "mechanism": {**original["mechanism"], "joint_type": "revolute"}}
    assert derived_stage_three_configuration(
        runtime={"object_authoring": {"configuration": hinged, "observation_manifest": record}},
        preparation=preparation) == hinged
    observed["frames"][0]["image_basis"] = "generated_view"
    observed["digest"] = canonical_digest(observed, digest_field="digest")
    write_json(manifest_path, observed)
    assert derived_stage_three_configuration(
        runtime={"object_authoring": {"configuration": original, "observation_manifest": _record(manifest_path)}},
        preparation=preparation) == original


def packet(tmp_path, *, authoring_backend=None, freezing=contextlib.nullcontext):
    """``freezing`` wraps the first preparation, which freezes stage 3."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    with freezing():
        args, preparation, _ = inputs(tmp_path, authoring_backend=authoring_backend)
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
    envelope, configs = packet(tmp_path, authoring_backend="astra_cad_blender_v1")
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
    envelope, configs = packet(tmp_path, authoring_backend="astra_cad_blender_v1")
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

    envelope, configs = packet(tmp_path, authoring_backend="astra_cad_blender_v1")
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
    envelope, configs = packet(control, authoring_backend="astra_cad_blender_v1")
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


def test_construction_without_a_support_under_the_subject_is_a_named_refusal(tmp_path):
    # 2026-10-30 capture-447ae4f4: compile flagged support_surface_not_found_under_subject and
    # construction then failed with "'NoneType' object is not subscriptable".
    args, preparation, _ = inputs(tmp_path)
    unsupported = {**preparation, "support": None}
    unsupported["digest"] = canonical_digest(unsupported, digest_field="digest")
    preparation_path = args["output_root"] / "preparation.json"
    write_json(preparation_path, unsupported)
    prepare_website_runtime_inputs(preparation=unsupported, base_scene=args["base_scene"],
        source_geometry=args["source_geometry"], task_masks=args["task_masks"], output_root=tmp_path / "native")
    with pytest.raises(ValueError, match="^website_native_construction_support_missing$"):
        prepare_construction_stages(runtime_inputs_path=tmp_path / "native/runtime_inputs.json",
                                    preparation_path=preparation_path)


def test_the_owner_deployed_artisan_cap_raises_a_frozen_website_stage_three(tmp_path, monkeypatch):
    """capture-447ae4f4, 2026-11-02: the owner raised the per-object cap to $25; the
    retry still froze $15 because stage 3 is reused from the first preparation."""
    from blueprint_pipeline import website_owner_authoring_cap as owner_cap
    release = tmp_path / "task-evaluation-scene-configuration-release.env"
    release.write_text("BLUEPRINT_TASK_EVALUATION_SCENE_CONFIGURATION_TOOLCHAIN_ROOT=/x\n"
                       "BLUEPRINT_ARTISAN_AUTHORING_MAX_COST_USD=25\n")
    monkeypatch.setattr(owner_cap, "RELEASE_ENVIRONMENT_FILE", release)
    envelope, configs = packet(tmp_path / "raised", authoring_backend="artisan_v1")
    assert configs["stage-3"]["authoring_budget"]["maximum_cost_usd"] == 25.0
    validate_website_native_inputs(envelope=envelope, configurations=configs, require_render_inputs=False)
    # A lower deployed value never lowers a frozen cap; no file changes nothing.
    release.write_text("BLUEPRINT_ARTISAN_AUTHORING_MAX_COST_USD=5\n")
    assert packet(tmp_path / "lower", authoring_backend="artisan_v1")[1]["stage-3"]["authoring_budget"][
        "maximum_cost_usd"] == 12.0
    release.unlink()
    assert packet(tmp_path / "absent", authoring_backend="artisan_v1")[1]["stage-3"]["authoring_budget"][
        "maximum_cost_usd"] == 12.0


ASTRA_MODELS = {"provider": "openai", "model": "gpt-6-astra", "thinking": "high",
                "compaction_model": "gpt-5.6-sol", "image_model": "gpt-6-astra",
                "image_tool_model": "gpt-image-2.5-sunburst"}


def test_a_stage_three_frozen_under_earlier_models_runs_on_the_release_models_and_their_cap(tmp_path, monkeypatch):
    """Owner direction 2026-11-02: ARTISAN moves from gpt-6-astra to gpt-6.1-sol and its cap
    from $25 to $12. capture-447ae4f4's stage 3 froze gpt-6-astra at $25; stage 3 refuses any
    models but the release's, so its retry is rebound to them and to their lower cap."""
    from blueprint_pipeline import task_object_authoring_backend as backend
    from blueprint_pipeline import website_owner_authoring_cap as owner_cap
    from blueprint_pipeline.task_evaluation_scene_configuration_stage_configuration import _stage_three_refusal
    release = tmp_path / "task-evaluation-scene-configuration-release.env"
    release.write_text("BLUEPRINT_ARTISAN_AUTHORING_MAX_COST_USD=12\n")
    monkeypatch.setattr(owner_cap, "RELEASE_ENVIRONMENT_FILE", release)

    @contextlib.contextmanager
    def under_astra():
        with monkeypatch.context() as frozen:
            frozen.setattr(backend, "ARTISAN_MODELS", ASTRA_MODELS)
            frozen.setenv(backend.ARTISAN_BUDGET_ENV, "25")
            yield
    envelope, configs = packet(tmp_path / "rebound", authoring_backend="artisan_v1", freezing=under_astra)
    runtime = json.loads((tmp_path / "rebound/native/runtime_inputs.json").read_text())
    frozen = runtime["object_authoring"]["configuration"]
    assert frozen["authoring_models"] == ASTRA_MODELS and frozen["authoring_budget"]["maximum_cost_usd"] == 25.0
    third = configs["stage-3"]
    assert third["authoring_models"] == backend.ARTISAN_MODELS == {
        "provider": "openai", "model": "gpt-6.1-sol", "thinking": "max", "compaction_model": "gpt-6.1-sol",
        "image_model": "gpt-6.1-sol", "image_tool_model": "gpt-image-2.5-sunburst"}
    assert third["authoring_budget"]["maximum_cost_usd"] == 12.0
    assert {k: v for k, v in third.items() if k not in {"authoring_models", "authoring_budget"}} == {
        k: v for k, v in frozen.items() if k not in {"authoring_models", "authoring_budget"}}
    validate_immutable_stage_configurations(envelope=envelope, configurations=configs)
    validate_website_native_inputs(envelope=envelope, configurations=configs, require_render_inputs=False)
    assert _stage_three_refusal(frozen, envelope) == "artisan_framework_or_budget_binding"
    assert _stage_three_refusal(third, envelope) is None
    # Without a deployed value the model's own cap applies.
    release.unlink()
    assert packet(tmp_path / "absent", authoring_backend="artisan_v1", freezing=under_astra)[1]["stage-3"][
        "authoring_budget"]["maximum_cost_usd"] == backend.ARTISAN_AUTHORING_CAP_USD == 12.0


def test_only_a_raised_cumulative_artisan_cap_passes_the_stage_three_rederivation():
    from blueprint_pipeline.task_object_authoring_backend import ARTISAN_AUTHORING_CAP_USD, ARTISAN_MODELS
    from blueprint_pipeline.website_owner_authoring_cap import same_except_owner_cap, with_owner_artisan_cap
    derived = {"authoring_backend": "artisan_v1", "replacement_identity": {"id": "tray"},
               "authoring_budget": {"schema_version": "task_object_authoring_budget.v1", "mode": "finite_cap",
                                    "maximum_cost_usd": 15.0,
                                    "scope": "whole_authoring_stage_cumulative_including_resume"}}
    derived["authoring_models"] = dict(ARTISAN_MODELS)
    raised = with_owner_artisan_cap(derived, 25.0)
    assert raised["authoring_budget"]["maximum_cost_usd"] == 25.0 and same_except_owner_cap(raised, derived)
    assert with_owner_artisan_cap(derived, 10.0) == derived and same_except_owner_cap(derived, derived)
    assert not same_except_owner_cap(derived, raised)  # never lowered
    assert not same_except_owner_cap({**raised, "replacement_identity": {"id": "other"}}, derived)
    astra = {**derived, "authoring_backend": "astra_cad_blender_v1"}
    assert with_owner_artisan_cap(astra, 25.0) == astra
    assert not same_except_owner_cap({**astra, "authoring_budget": raised["authoring_budget"]}, astra)
    # Frozen under earlier models: rebound to the release's models and their cap, even a lower one.
    earlier = {**derived, "authoring_models": ASTRA_MODELS,
               "authoring_budget": {**derived["authoring_budget"], "maximum_cost_usd": 25.0}}
    rebound = with_owner_artisan_cap(earlier, 12.0)
    assert rebound == {**earlier, "authoring_models": ARTISAN_MODELS,
                       "authoring_budget": {**earlier["authoring_budget"], "maximum_cost_usd": 12.0}}
    assert same_except_owner_cap(rebound, earlier)
    assert with_owner_artisan_cap(earlier, None)["authoring_budget"]["maximum_cost_usd"] == ARTISAN_AUTHORING_CAP_USD
    assert not same_except_owner_cap(earlier, rebound)  # never back to earlier models
    assert not same_except_owner_cap({**rebound, "authoring_models": {**ARTISAN_MODELS, "thinking": "high"}}, earlier)
    assert not same_except_owner_cap({**rebound, "replacement_identity": {"id": "other"}}, earlier)
    assert not same_except_owner_cap({**rebound, "authoring_budget": {**rebound["authoring_budget"],
                                                                      "maximum_cost_usd": 0}}, earlier)
