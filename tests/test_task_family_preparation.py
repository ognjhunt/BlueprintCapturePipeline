"""Known families compile and execute without entering rigid authoring."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from blueprint_pipeline import native_task_runtime_contract as native
from blueprint_pipeline import website_scene_handoff as handoff
from blueprint_pipeline import website_task_preparation as website
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.job_capability_routing import route_job_capabilities
from blueprint_pipeline.task_family_scoring import FAMILY_ALIASES, seal_task_family_contract
from blueprint_pipeline.website_task_context_contracts import retains_task_family_preparation
from tests.test_task_evaluation_task_family_native_adapter import Policy, Runtime, _runtime_contract
from tests.test_website_task_preparation import _arguments


class CountedRuntime(Runtime):
    def __init__(self):
        super().__init__()
        self.close_count = 0

    def close(self):
        self.close_count += 1
        super().close()


def _arguments_for_family(tmp_path, family="button_press"):
    arguments = _arguments(tmp_path)
    context = arguments["task_context"]
    context["task_family"] = family
    context["description"] = "Run the original multi-object job."
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    return arguments


def _evaluation_inputs(arguments):
    context = arguments["task_context"]
    contract = seal_task_family_contract(task_family="button_press",
        scope={"scene_id": context["scene_id"], "task_id": "website-" + context["context_digest"][7:27],
            "task_context_digest": context["context_digest"]}, runtime_identity_digest="sha256:" + "a" * 64,
        stages=[{"stage_id": "stage1", "task_family": "button_press", "minimum_hold_seconds": 0,
            "criteria": {"minimum_button_travel_m": .01, "maximum_contact_force_n": 20}}],
        reset={"stage1": {"button_travel_m": 0, "button_activated": False, "contact_force_n": 0}})
    template = _runtime_contract()
    kwargs = {key: template[key] for key in ("task_state_binding", "assets", "robot_base_pose_world",
        "robot_joint_reset_positions_rad", "robot_configuration", "cameras", "scenario_cell_id",
        "scenario_instance_digest", "seed", "scenario_context_kind")}
    kwargs = copy.deepcopy(kwargs)
    base, geometry, masks = (arguments[key] for key in ("base_scene", "source_geometry", "task_masks"))
    frame = {"source_geometry_digest": geometry["digest"], "collision_mesh_digest": base["collision_mesh_digest"],
        "runtime_up_axis": base["up_axis"], "runtime_meters_per_unit": base["meters_per_unit"],
        "source_to_runtime": np.eye(4).tolist()}
    frame["registration_digest"] = canonical_digest(frame, digest_field="registration_digest")
    kwargs["task_state_binding"]["source_scene_binding"] = {
        "task_context_digest": context["context_digest"], "source_geometry_digest": geometry["digest"],
        "task_masks_digest": masks["digest"], "splat_digest": base["splat_digest"],
        "collision_mesh_digest": base["collision_mesh_digest"], "frame_registration": frame}
    kwargs["assets"].extend({"asset_id": role, "semantic_role": role, "content_digest": digest,
        "reset_state": {"fixed": True}} for role, digest in (("scene_appearance", base["splat_digest"]),
            ("scene_collision", base["collision_mesh_digest"])))
    return {"task_success_contract": contract, "native_runtime_inputs": kwargs}


@pytest.mark.parametrize("family", FAMILY_ALIASES)
def test_known_families_have_real_profiles_and_request_specific_inputs(tmp_path, monkeypatch, family):
    args = _arguments_for_family(tmp_path, family)
    monkeypatch.setattr(website, "register_source_to_runtime", lambda **_: pytest.fail("No rigid coercion"))
    result = website.compile_website_scene_preparation(**args)
    assert result["job_disposition"] == "accepted" and result["status"] == "needs_input"
    assert result["blockers"] == ["website_task_family_success_contract_required",
                                  "website_task_family_native_runtime_inputs_required"]
    assert "native_task_preparation_gap" not in result and result["intake_request"] is None
    inputs = result["task_evaluation_inputs"]
    assert inputs["task_family"] == family and inputs["source_targets"] == args["task_masks"]["targets"]
    assert inputs["runtime_adapter"] == "task_evaluation_task_family_native_adapter"
    assert website.native_task_preparation_gap(args["task_context"], inferred_profile_id="rigid_pick_place") is None
    route = route_job_capabilities(family)
    assert route["status"] == "needs_input" and route["capability_gap"] is None
    assert "native_runtime_contract" in route["missing_inputs"]


def test_existing_materializer_and_loader_dispatch_family_contract(tmp_path):
    args = _arguments_for_family(tmp_path)
    inputs = _evaluation_inputs(args)
    task = inputs["task_success_contract"]
    path = tmp_path / "native.json"
    contract = native.materialize_native_task_runtime_contract(scene_id=task["scope"]["scene_id"],
        task_id=task["scope"]["task_id"], task_spec={"task_kind": "button_press", "task_success_contract": task},
        destination=path, **inputs["native_runtime_inputs"])
    assert contract["schema_version"] == "task_family_native_runtime_contract.v1"
    assert native.load_native_task_runtime_contract(path) == contract


def test_compiled_website_family_is_consumed_by_existing_handoff_module(tmp_path):
    args = _arguments_for_family(tmp_path)
    args["evaluation_inputs"] = _evaluation_inputs(args)
    result = website.compile_website_scene_preparation(**args)
    assert result["status"] == "task_evaluation_ready" and result["blockers"] == []
    runtime = CountedRuntime()
    runtime.native_runtime_contract = result["task_evaluation_inputs"]["native_runtime_contract"]
    runtime.native_runtime_contract_digest = runtime.native_runtime_contract["contract_digest"]
    receipt = handoff.execute_prepared_task_family(preparation_path=args["output_root"] / "preparation.json",
        runtime=runtime, policy=Policy(), output_dir=tmp_path / "episode", episode_id="family1",
        maximum_steps=2, control_frequency_hz=1)
    assert receipt["status"] == "completed" and receipt["task_succeeded"] and runtime.closed
    assert runtime.close_count == 1
    assert receipt["physical_validation"] is False


@pytest.mark.parametrize("corruption", ["source", "frame", "assets"])
def test_family_runtime_requires_exact_source_assets_and_frame(tmp_path, corruption):
    args = _arguments_for_family(tmp_path)
    inputs = _evaluation_inputs(args)
    runtime = inputs["native_runtime_inputs"]
    if corruption == "source":
        runtime["task_state_binding"]["source_scene_binding"]["splat_digest"] = "sha256:" + "f" * 64
        expected = "source_scene_binding_mismatch"
    elif corruption == "frame":
        runtime["task_state_binding"]["source_scene_binding"]["frame_registration"]["runtime_meters_per_unit"] = 1000
        expected = "frame_registration_invalid"
    else:
        runtime["assets"] = runtime["assets"][:1]
        result = website.compile_website_scene_preparation(**args, evaluation_inputs=inputs)
        assert result["status"] == "needs_input"
        assert "website_task_family_scene_collision_asset_required" in result["blockers"]
        return
    with pytest.raises(ValueError, match=expected):
        website.compile_website_scene_preparation(**args, evaluation_inputs=inputs)


def test_family_route_cannot_promote_unavailable_humanoid_or_mismatched_interfaces(tmp_path):
    args = _arguments_for_family(tmp_path)
    inputs = _evaluation_inputs(args)
    inputs["native_runtime_inputs"]["robot_configuration"]["embodiment_id"] = "boston_dynamics_atlas"
    result = website.compile_website_scene_preparation(**args, evaluation_inputs=inputs)
    route = route_job_capabilities("button_press", embodiment_ids=["boston_dynamics_atlas"],
        native_runtime_contracts={"boston_dynamics_atlas": result["task_evaluation_inputs"]["native_runtime_contract"]})
    assert route["status"] == "capability_gap" and route["configurations"] == []
    assert any(row["id"] == "boston_dynamics_atlas" for row in route["unavailable"])


def test_family_preparation_survives_handoff_without_legacy_intake(tmp_path):
    args = _arguments_for_family(tmp_path)
    result = website.compile_website_scene_preparation(**args)
    context = args["task_context"]
    value = {"schema_version": "website_scene_handoff.v1", "scene_id": context["scene_id"],
        "capture_id": context["capture_id"], "visual_reconstruction_ready": True, "simulator_ready": False,
        "status": result["status"], "blockers": result["blockers"], "job_disposition": "accepted",
        "task_evaluation_inputs": result["task_evaluation_inputs"]}
    value["digest"] = canonical_digest(value, digest_field="digest")
    assert retains_task_family_preparation(value, context)
    value["task_evaluation_inputs"]["scope"]["task_context_digest"] = "sha256:" + "f" * 64
    value["digest"] = canonical_digest(value, digest_field="digest")
    assert not retains_task_family_preparation(value, context)


def test_ready_family_handoff_revalidates_the_same_source_frame(tmp_path):
    args = _arguments_for_family(tmp_path)
    result = website.compile_website_scene_preparation(**args, evaluation_inputs=_evaluation_inputs(args))
    context = args["task_context"]
    value = {"schema_version": "website_scene_handoff.v1", "scene_id": context["scene_id"],
        "capture_id": context["capture_id"], "visual_reconstruction_ready": True, "simulator_ready": False,
        "status": result["status"], "blockers": [], "job_disposition": "accepted",
        "source_geometry_digest": args["source_geometry"]["digest"],
        "task_evaluation_inputs": result["task_evaluation_inputs"]}
    value["digest"] = canonical_digest(value, digest_field="digest")
    assert retains_task_family_preparation(value, context)
    runtime = value["task_evaluation_inputs"]["native_runtime_contract"]
    frame = runtime["task_state_binding"]["source_scene_binding"]["frame_registration"]
    frame["source_to_runtime"][0][0] = 0
    frame["registration_digest"] = canonical_digest(frame, digest_field="registration_digest")
    runtime["contract_digest"] = canonical_digest(runtime, digest_field="contract_digest")
    value["digest"] = canonical_digest(value, digest_field="digest")
    assert not retains_task_family_preparation(value, context)
    assert not retains_task_family_preparation(None, context)
    assert not retains_task_family_preparation(value, None)


def test_prepared_family_rechecks_scene_bytes_before_policy_use(tmp_path):
    args = _arguments_for_family(tmp_path)
    result = website.compile_website_scene_preparation(**args, evaluation_inputs=_evaluation_inputs(args))
    runtime = CountedRuntime()
    runtime.native_runtime_contract_digest = result["task_evaluation_inputs"]["native_runtime_contract"]["contract_digest"]
    (tmp_path / "world.ply").write_bytes(b"changed-scene")
    with pytest.raises(ValueError, match="source_scene_changed"):
        handoff.execute_prepared_task_family(preparation_path=args["output_root"] / "preparation.json",
            runtime=runtime, policy=Policy(), output_dir=tmp_path / "episode", episode_id="family1",
            maximum_steps=2, control_frequency_hz=1)
    assert runtime.actions == []
    assert runtime.closed and runtime.close_count == 1


def test_prepared_family_closes_runtime_on_native_contract_refusal(tmp_path):
    args = _arguments_for_family(tmp_path)
    website.compile_website_scene_preparation(**args, evaluation_inputs=_evaluation_inputs(args))
    runtime = CountedRuntime()
    runtime.native_runtime_contract_digest = "sha256:" + "f" * 64
    with pytest.raises(ValueError, match="runtime_scene_mismatch"):
        handoff.execute_prepared_task_family(preparation_path=args["output_root"] / "preparation.json",
            runtime=runtime, policy=Policy(), output_dir=tmp_path / "episode", episode_id="family1",
            maximum_steps=2, control_frequency_hz=1)
    assert runtime.actions == [] and runtime.closed and runtime.close_count == 1


def test_actual_capture_handoff_compiles_family_without_rigid_authoring(tmp_path, monkeypatch):
    args = _arguments_for_family(tmp_path, "cloth_folding")
    args["source_geometry"]["geometry_backend"] = "fixture-native-depth"
    args["source_geometry"]["digest"] = canonical_digest(args["source_geometry"], digest_field="digest")
    args["task_masks"]["source_geometry_digest"] = args["source_geometry"]["digest"]
    args["task_masks"]["digest"] = canonical_digest(args["task_masks"], digest_field="digest")
    pipeline = tmp_path / "pipeline"
    pipeline.mkdir()
    removal = pipeline / "removal.json"
    write_json(removal, args["removal_manifest"])
    atlas = {"status": "ready", "base_scene": args["base_scene"], "source_geometry": args["source_geometry"],
        "source_pose_digest": "sha256:" + "b" * 64, "background_pose_digest": "sha256:" + "c" * 64,
        "alignment": "anchored"}
    monkeypatch.setattr("blueprint_pipeline.website_atlas_scene_source.atlas_scene_source", lambda **_: atlas)
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.bind_task_masks_to_geometry", lambda **kw: kw["task_masks"])
    monkeypatch.setattr("blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage",
        lambda **_: pytest.fail("No rigid/articulated authoring for cloth"))
    result = handoff.prepare_website_scene_handoff(descriptor={"scene_id": args["task_context"]["scene_id"],
        "capture_id": args["task_context"]["capture_id"], "metadata": {"site_task_context": args["task_context"]}},
        clean_plate={"privacy_verified": True, "status": "objects_removed", "removal_manifest_path": str(removal),
            "task_masks": args["task_masks"]}, capture_root=tmp_path, now=args["now"])
    assert result["status"] == "needs_input" and result["job_disposition"] == "accepted"
    assert retains_task_family_preparation(result, args["task_context"])
    assert result["task_evaluation_inputs"]["task_family"] == "cloth_folding"
    assert "native_task_preparation_gap" not in result and result["simulator_ready"] is False
    assert json.loads((pipeline / "website_scene_preparation/handoff.json").read_text()) == result


@pytest.mark.parametrize("missing", ["runtime_selection", "authority", "expired_authority"])
def test_prepared_family_never_infers_dispatch_or_paid_authority(tmp_path, missing):
    args = _arguments_for_family(tmp_path)
    inputs = _evaluation_inputs(args)
    inputs["task_family_runtime_selection"] = {
        "runtime_id": "registered_engine", "runtime_catalog_digest": "sha256:" + "b" * 64}
    args["spend"] = copy.deepcopy(args["spend"])
    if missing == "runtime_selection":
        del inputs["task_family_runtime_selection"]
        expected = "website_task_family_qualified_runtime_selection_required"
    elif missing == "authority":
        args["spend"] = {}
        expected = "website_scene_execution_authority_required"
    else:
        args["spend"]["expires_at_epoch"] = args["now"]
        expected = "scene_intake_authority_expiry_invalid"
    result = website.compile_website_scene_preparation(**args, evaluation_inputs=inputs)
    assert result["status"] == "task_evaluation_ready"
    assert result["intake_request"] is None and expected in result["dispatch_blockers"]
    assert result["job_disposition"] == "accepted"
    assert result["task_evaluation_inputs"]["automatic_paid_execution_authorized"] is False
