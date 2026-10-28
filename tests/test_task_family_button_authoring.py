# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_family_button_authoring.py
#   src/blueprint_pipeline/task_family_preparation.py
#   src/blueprint_pipeline/task_family_authoring.py
#   src/blueprint_pipeline/website_scene_handoff.py
#   src/blueprint_pipeline/website_task_semantics.py
"""Ordinary measured-layout development fixtures, never runtime qualification.

No native_task_evaluation, supplied success contract, finished native profile,
provider, SDK or real policy episode is used by this authoring coverage.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline import task_family_button_authoring as producer
from blueprint_pipeline import website_task_preparation as preparation
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_family_native_contracts import validate_task_family_native_runtime_contract
from blueprint_pipeline.task_object_authoring_contracts import file_record
from blueprint_pipeline.website_task_semantics import task_semantics_from_receipt
from tests.test_task_family_native_runtime import _seal_catalog
from tests.test_website_task_preparation import NOW, RUNTIME_ROTATION, RUNTIME_SCALE, RUNTIME_TRANSLATION, _arguments

URDF = '''<robot name="development_only_button">
<link name="body"><inertial><mass value="0.1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
<collision><geometry><box size="0.03 0.03 0.02"/></geometry></collision></link>
<link name="cap"><inertial><mass value="0.01"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
<collision><geometry><box size="0.02 0.02 0.01"/></geometry></collision></link>
<joint name="button_joint" type="prismatic"><parent link="body"/><child link="cap"/>
<origin xyz="0 0 0.02" rpy="0 0 0"/><axis xyz="0 0 1"/>
<limit lower="0" upper="0.02" effort="30" velocity="0.05"/></joint></robot>'''


def _seal(value, field):
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _fixture(tmp_path, monkeypatch, *, sequence=False, general=False):
    args = _arguments(tmp_path)
    context, geometry, masks, scene = [args[key] for key in ("task_context", "source_geometry", "task_masks", "base_scene")]
    context.update(task_family="general_task" if general else "multi_step" if sequence else "button_press",
                   description="Press the red button, then press the green button." if sequence else "Press the red button.")
    geometry.update(binding={"source_video_digest": "sha256:" + "e" * 64}, geometry_backend="fixture-native-rgbd",
                    metric_measurement_proven=True, unit="meters", scale_status="fixture_metric_source")
    _seal(geometry, "digest")
    masks["source_geometry_digest"] = geometry["digest"]
    target = masks["targets"][0]
    target.update(target_id="red", semantic_label="red button")
    masks["targets"] = [target]
    if sequence:
        masks["targets"].append({**copy.deepcopy(target), "target_id": "green", "semantic_label": "green button"})
    matrix = np.eye(4)
    matrix[:3, :3] = RUNTIME_ROTATION * RUNTIME_SCALE
    matrix[:3, 3] = RUNTIME_TRANSLATION
    frame = _seal({"source_geometry_digest": geometry["digest"], "collision_mesh_digest": scene["collision_mesh_digest"],
        "runtime_up_axis": scene["up_axis"], "runtime_meters_per_unit": scene["meters_per_unit"],
        "source_to_runtime": matrix.tolist()}, "registration_digest")
    registration = tmp_path / "measured-layout-registration.json"
    registration.write_text(json.dumps({"claim_ceiling": "development_only", "frame_registration": frame,
                                        "source_video_digest": geometry["binding"]["source_video_digest"]}))
    configuration = {"embodiment_id": "droid_franka_robotiq_2f85", "controller": "droid_absolute_joint_position_v1",
                     "observation_schema_id": "droid_two_camera_robot_state_v1", "action_schema_id": "droid_absolute_joint_position_v1"}
    pose = {"position_world_m": [0.2, 0.5, 1.5], "orientation_xyzw": [0, 0, 0, 1]}
    masks["registered_layout"] = _seal({"schema_version": producer.LAYOUT_SCHEMA,
        "source_geometry_digest": geometry["digest"], "collision_mesh_digest": scene["collision_mesh_digest"],
        "source_video_digest": geometry["binding"]["source_video_digest"], "metric_registration_verified": True,
        "registration_evidence": file_record(registration), "frame_registration": frame,
        "targets": {row["target_id"]: {"asset_id": "engineering-button", "pose_source": pose,
                    "source_frame_ids": ["frame-0"]} for row in masks["targets"]},
        "robot": {"configuration": configuration, "base_pose_source": pose,
                  "joint_reset_positions_rad": {"panda_joint1": 0}},
        "camera_frame_ids": ["frame-0"]}, "layout_digest")
    _seal(masks, "digest")
    specs = ([{"operation": "button_press", "description": "Press the red button", "task_quote": "Press the red button"},
              {"operation": "button_press", "description": "Press the green button", "task_quote": "press the green button"}]
             if sequence else None)
    operation_ids = ["operation-" + str(index).zfill(3) + "-" + canonical_digest(step)[7:15]
                     for index, step in enumerate(specs, 1)] if specs else ["operation-001"]
    facts = _seal({"schema_version": producer.FACTS_SCHEMA, "capture_id": context["capture_id"],
        "source_video_digest": geometry["binding"]["source_video_digest"],
        "task_description_digest": canonical_digest({"description": context["description"]}), "confirmed": True,
        "task_success_criteria_digest": canonical_digest({"success_criteria": context.get("success_criteria")}),
        "confirmation_reference": "development-owner-fact-correction",
        "operations": [{"operation_id": key, "minimum_button_travel_m": .01,
                        "maximum_contact_force_n": 20, "minimum_hold_seconds": .2} for key in operation_ids]}, "facts_digest")
    context["operator_answers"] = {"family_task_facts": facts}
    _seal(context, "context_digest")
    if sequence:
        binding = {"task_context_digest": context["context_digest"], "source_video_digest": geometry["binding"]["source_video_digest"]}
        proposal = {"status": "proposed", "task_semantics": {"operative_job": context["description"], "task_family": "multi_step",
            "steps": specs, "covers_entire_job": True, "unresolved_aspects": [], "preparation_notes": []}}
        args["task_semantics"] = task_semantics_from_receipt(receipt={"status": "completed", "binding": binding,
            "binding_digest": canonical_digest(binding), "proposal": proposal, "proposal_digest": canonical_digest(proposal)},
            task_context=context, source_video_digest=geometry["binding"]["source_video_digest"])
    library = tmp_path / "operator-library"
    library.mkdir()
    model, evidence = library / "button.urdf", library / "engineering-evidence.json"
    model.write_text(URDF)
    evidence.write_text(json.dumps({"claim_ceiling": "development_only", "source": "engineering-fixture-model",
                                  "physics_validation_claimed": False}))
    row = {"runtime_id": "fixture-engine", "runtime_identity_digest": "sha256:" + "a" * 64,
        "engine_id": "hermetic-fixture-not-executable-qualification", "factory_id": "gymnasium_native_task_family.v1",
        "task_families": ["button_press", "multi_step"], "robot_interfaces": [configuration],
        "policy_interfaces": ["fixture-never-queried"], "gymnasium_environment_id": "Fixture/Buttons",
        "registration_module": "fixture_driver", "entry_point": "fixture_driver:Environment",
        "runtime_sources": [{"module": "fixture_driver", "sha256": "sha256:" + "b" * 64}],
        "authoring_assets": [{"asset_id": "engineering-button", "task_family": "button_press",
            "source": file_record(model), "engineering_evidence": file_record(evidence),
            "rights_reference": "CC0 engineering fixture; never production evidence",
            "contact_material": {"static_friction": .5, "dynamic_friction": .4, "restitution": .1},
            "button_mechanics": {"rest_position_m": 0, "reset_position_m": 0, "press_direction": 1,
                "activation_sensor_id": "button-switch", "reset_sensor_active": False, "reset_contact_force_n": 0}}]}
    catalog = library / "catalog.json"
    _seal_catalog({"runtime_catalog_path": catalog}, row, qualification_changes={"expires_at_epoch": NOW + 3600})
    monkeypatch.setenv("BLUEPRINT_TASK_FAMILY_RUNTIME_CATALOG_PATH", str(catalog))
    return args, row


def test_ordinary_registered_layout_authors_validated_native_inputs_without_injection(tmp_path, monkeypatch):
    args, row = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(preparation, "register_source_to_runtime", lambda **_: pytest.fail("No estimated rigid registration"))
    result = preparation.compile_website_scene_preparation(**args)
    inputs = result["task_evaluation_inputs"]
    assert result["status"] == "intake_ready", result["blockers"]
    assert inputs["authoring"]["request"]["native_inputs_authored"] is True
    native = validate_task_family_native_runtime_contract(inputs["native_runtime_contract"])
    assert native["assets"][0]["native_joint_id"] == "button_joint"
    assert native["assets"][0]["joint_axis_local"] == [0, 0, 1]
    assert native["assets"][0]["content_digest"] == row["authoring_assets"][0]["source"]["sha256"]
    assert native["assets"][0]["reset_state"]["pose_world"]["position_world_m"] == pytest.approx([2.2, 3.5, 4.5])
    assert native["cameras"][0]["role"] == "source_reference"
    assert native["cameras"][0]["intrinsics"] == args["source_geometry"]["frames"][0]["intrinsics"]
    assert inputs["task_success_contract"]["stages"][0]["criteria"] == {
        "minimum_button_travel_m": .01, "maximum_contact_force_n": 20}
    assert inputs["automatic_paid_execution_authorized"] is False
    assert result["claim_ceiling"] == "development_only" and result["provider_mutation_performed"] is False


def test_ordered_buttons_keep_source_operations_and_independent_asset_bindings(tmp_path, monkeypatch):
    args, _ = _fixture(tmp_path, monkeypatch, sequence=True)
    result = preparation.compile_website_scene_preparation(**args)
    inputs = result["task_evaluation_inputs"]
    assert result["status"] == "intake_ready"
    stages = inputs["task_success_contract"]["stages"]
    assert [row["stage_id"] for row in stages] == [row["operation_id"] for row in
        args["task_context"]["operator_answers"]["family_task_facts"]["operations"]]
    native = inputs["native_runtime_contract"]
    bindings = native["task_state_binding"]["stage_bindings"]
    assert len({row["asset_roles"]["button"] for row in bindings.values()}) == 2
    assert list(native["task_spec"]["task_success_contract"]["reset"]) == [row["stage_id"] for row in stages]


def _reseal_facts(args):
    _seal(args["task_context"]["operator_answers"]["family_task_facts"], "facts_digest")
    _seal(args["task_context"], "context_digest")


def _rebind_semantics(args, *, steps=None):
    receipt = copy.deepcopy(args["task_semantics"]["planner_receipt"])
    receipt["binding"]["task_context_digest"] = args["task_context"]["context_digest"]
    receipt["binding_digest"] = canonical_digest(receipt["binding"])
    if steps is not None:
        receipt["proposal"]["task_semantics"]["steps"] = steps
    receipt["proposal_digest"] = canonical_digest(receipt["proposal"])
    args["task_semantics"] = task_semantics_from_receipt(receipt=receipt, task_context=args["task_context"],
        source_video_digest=args["source_geometry"]["binding"]["source_video_digest"])


@pytest.mark.parametrize("change", ["no_ids", "wrong_order", "wrong_count", "unconfirmed", "copied_quote"])
def test_repeated_family_route_requires_source_bound_operation_identity_and_order(tmp_path, monkeypatch, change):
    from blueprint_pipeline.website_task_semantics import resolved_task_profile

    args, _ = _fixture(tmp_path, monkeypatch, sequence=True, general=True)
    facts = args["task_context"]["operator_answers"]["family_task_facts"]
    steps = None
    if change == "no_ids":
        for answer in facts["operations"]:
            answer.pop("operation_id")
    elif change == "wrong_order":
        facts["operations"].reverse()
    elif change == "wrong_count":
        facts["operations"].pop()
    elif change == "unconfirmed":
        facts["confirmed"] = False
    else:
        steps = copy.deepcopy(args["task_semantics"]["semantics"]["steps"])
        steps[1] = copy.deepcopy(steps[0])
        for index, (answer, step) in enumerate(zip(facts["operations"], steps), 1):
            answer["operation_id"] = "operation-" + str(index).zfill(3) + "-" + canonical_digest(step)[7:15]
    _reseal_facts(args)
    _rebind_semantics(args, steps=steps)
    assert resolved_task_profile(args["task_semantics"], task_context=args["task_context"],
        source_video_digest=args["source_geometry"]["binding"]["source_video_digest"]) is None
    held = preparation.compile_website_scene_preparation(**args)
    assert held["status"] == "needs_input" and held["intake_request"] is None
    assert held["native_task_preparation_gap"]["code"] == "website_task_operation_coverage_unresolved"
    assert "task_evaluation_inputs" not in held
    assert not (args["output_root"] / "family_authoring" / "stages").exists()


@pytest.mark.parametrize("declared", ["general_task", "multi_step", "multi-step"])
def test_repeated_family_packet_cannot_reverse_native_stage_identities(tmp_path, monkeypatch, declared):
    from blueprint_pipeline.common import write_json
    from blueprint_pipeline.task_family_scoring import seal_task_family_contract
    from blueprint_pipeline.website_task_family_packet import materialize_website_task_family_packet

    args, _ = _fixture(tmp_path, monkeypatch, sequence=True, general=declared == "general_task")
    args["task_context"]["task_family"] = declared
    _seal(args["task_context"], "context_digest")
    _rebind_semantics(args)
    result = preparation.compile_website_scene_preparation(**args)
    inputs = result["task_evaluation_inputs"]
    original = inputs["task_success_contract"]
    reversed_contract = seal_task_family_contract(task_family="multi_step", scope=original["scope"],
        stages=list(reversed(original["stages"])), reset=original["reset"],
        runtime_identity_digest=original["runtime_identity_digest"])
    supplied = copy.deepcopy(inputs["authoring"]["request"]["producer_result"]["outputs"])
    supplied["task_success_contract"] = reversed_contract
    with pytest.raises(ValueError, match="semantic_stages_mismatch"):
        preparation.compile_website_scene_preparation(**args, evaluation_inputs=supplied)
    # A stored preparation cannot evade the same check by resealing both
    # native/intake digests rather than re-entering the compiler.
    inputs["task_success_contract"] = reversed_contract
    native = inputs["native_runtime_contract"]
    native["task_spec"]["task_success_contract"] = reversed_contract
    _seal(native, "contract_digest")
    result["intake_request"]["task"].update(task_success_contract_digest=reversed_contract["contract_digest"],
        native_runtime_contract_digest=native["contract_digest"])
    _seal(result, "digest")
    runtime_path = tmp_path / "reversed-native.json"
    write_json(runtime_path, native)
    reference = file_record(runtime_path)
    output = tmp_path / "reversed-packet"
    with pytest.raises(ValueError, match="semantic_stages_mismatch"):
        materialize_website_task_family_packet(preparation=result, runtime_inputs_path=runtime_path,
            output_root=output, intent_reference=reference, attempt_reference=reference,
            release_reference=reference, task_context=args["task_context"])
    assert not output.exists()


def test_explicit_multistep_cannot_drop_the_retained_operation_reading(tmp_path, monkeypatch):
    args, _ = _fixture(tmp_path, monkeypatch, sequence=True)
    result = preparation.compile_website_scene_preparation(**args)
    supplied = result["task_evaluation_inputs"]["authoring"]["request"]["producer_result"]["outputs"]
    args.pop("task_semantics")
    with pytest.raises(ValueError, match="semantic_stages_mismatch"):
        preparation.compile_website_scene_preparation(**args, evaluation_inputs=supplied)


def test_multistep_packet_requires_original_context_even_if_semantics_field_is_dropped(tmp_path, monkeypatch):
    from blueprint_pipeline.common import write_json
    from blueprint_pipeline.website_task_family_packet import materialize_website_task_family_packet

    args, _ = _fixture(tmp_path, monkeypatch, sequence=True)
    result = preparation.compile_website_scene_preparation(**args)
    result["task_evaluation_inputs"].pop("task_semantics")
    _seal(result, "digest")
    runtime_path = tmp_path / "native-without-context.json"
    write_json(runtime_path, result["task_evaluation_inputs"]["native_runtime_contract"])
    reference = file_record(runtime_path)
    output = tmp_path / "packet-without-context"
    with pytest.raises(ValueError, match="semantics_context_required"):
        materialize_website_task_family_packet(preparation=result, runtime_inputs_path=runtime_path,
            output_root=output, intent_reference=reference, attempt_reference=reference, release_reference=reference)
    assert not output.exists()


def test_missing_pass_mark_asks_one_normal_fact_and_adopts_completed_source_stage(tmp_path, monkeypatch):
    args, _ = _fixture(tmp_path, monkeypatch)
    answer = args["task_context"]["operator_answers"]["family_task_facts"]["operations"][0]
    del answer["minimum_hold_seconds"]
    _reseal_facts(args)
    held = preparation.compile_website_scene_preparation(**args)
    request = held["task_evaluation_inputs"]["authoring"]["request"]
    assert held["status"] == "needs_input"
    assert request["next_action"]["quantity"] == "minimum_hold_seconds"
    assert request["next_action"]["owner"] == "task_owner"
    sources = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in
               (args["output_root"] / "family_authoring" / "stages").glob("*/source-*.json")}
    assert len(sources) == 1
    answer["minimum_hold_seconds"] = .2
    _reseal_facts(args)
    prepared = preparation.compile_website_scene_preparation(**args)
    assert prepared["status"] == "intake_ready"
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == before for path, before in sources.items())


@pytest.mark.parametrize("quantity", ["minimum_button_travel_m", "maximum_contact_force_n"])
def test_missing_physical_button_threshold_is_engineering_dependency(tmp_path, monkeypatch, quantity):
    args, _ = _fixture(tmp_path, monkeypatch)
    del args["task_context"]["operator_answers"]["family_task_facts"]["operations"][0][quantity]
    _reseal_facts(args)
    held = preparation.compile_website_scene_preparation(**args)
    next_action = held["task_evaluation_inputs"]["authoring"]["request"]["next_action"]
    assert held["status"] == "needs_input" and next_action["quantity"] == quantity
    assert next_action["category"] == "source_measurement" and next_action["owner"] == "scene_preparation"
    assert "engineering/source evidence" in next_action["question"]


@pytest.mark.parametrize("boundary", ["source-button-", "operation-1", "evidence-inputs", "native-inputs", "authoring-result"])
def test_restart_after_each_authoring_stage_adopts_identical_bytes(tmp_path, monkeypatch, boundary):
    args, _ = _fixture(tmp_path, monkeypatch)
    original = producer._retain
    stopped = []

    def interrupted(root, kind, value):
        reference = original(root, kind, value)
        if kind.startswith(boundary) and not stopped:
            stopped.append(kind)
            raise RuntimeError("fixture crash after retained stage")
        return reference

    monkeypatch.setattr(producer, "_retain", interrupted)
    with pytest.raises(RuntimeError, match="fixture crash"):
        preparation.compile_website_scene_preparation(**args)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in
              (args["output_root"] / "family_authoring" / "stages").glob("*/*.json")}
    assert stopped and before
    monkeypatch.setattr(producer, "_retain", original)
    result = preparation.compile_website_scene_preparation(**args)
    assert result["status"] == "intake_ready"
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == data for path, data in before.items())
    assert preparation.compile_website_scene_preparation(**args) == result


def test_estimated_layout_cannot_be_promoted_from_visible_bounds(tmp_path, monkeypatch):
    args, _ = _fixture(tmp_path, monkeypatch)
    layout = args["task_masks"]["registered_layout"]
    layout["metric_registration_verified"] = False
    _seal(layout, "layout_digest")
    _seal(args["task_masks"], "digest")
    result = preparation.compile_website_scene_preparation(**args)
    request = result["task_evaluation_inputs"]["authoring"]["request"]
    assert result["status"] == "needs_input" and result["task_evaluation_inputs"]["native_runtime_contract"] is None
    assert request["next_action"]["quantity"] == "registered_target_layout"
    assert request["next_action"]["category"] == "source_measurement"
    assert "visible bounds" in request["next_action"]["question"]


def test_changed_asset_bytes_refuse_without_overwriting_completed_work(tmp_path, monkeypatch):
    args, row = _fixture(tmp_path, monkeypatch)
    prepared = preparation.compile_website_scene_preparation(**args)
    previous = Path(prepared["task_evaluation_inputs"]["authoring"]["reference"]["path"])
    before = previous.read_bytes()
    Path(row["authoring_assets"][0]["source"]["path"]).write_text(URDF + " ")
    with pytest.raises(ValueError, match="authoring_source_bytes_changed"):
        preparation.compile_website_scene_preparation(**args)
    assert previous.read_bytes() == before


def test_unknown_contact_material_is_engineering_dependency_not_customer_engine_json(tmp_path, monkeypatch):
    args, row = _fixture(tmp_path, monkeypatch)
    row["authoring_assets"][0].pop("contact_material", None)
    _seal_catalog({"runtime_catalog_path": tmp_path / "operator-library" / "catalog.json"}, row,
                  qualification_changes={"expires_at_epoch": NOW + 3600})
    result = preparation.compile_website_scene_preparation(**args)
    request = result["task_evaluation_inputs"]["authoring"]["request"]
    assert result["status"] == "needs_input"
    assert request["next_action"]["quantity"] == "button_contact_material"
    assert request["next_action"]["owner"] == "scene_preparation"
    assert request["next_action"]["category"] == "source_measurement"
    assert "friction" in request["next_action"]["question"]


def _replace_library_model(tmp_path, row, content):
    path = Path(row["authoring_assets"][0]["source"]["path"])
    path.write_text(content)
    row["authoring_assets"][0]["source"] = file_record(path)
    _seal_catalog({"runtime_catalog_path": tmp_path / "operator-library" / "catalog.json"}, row,
                  qualification_changes={"expires_at_epoch": NOW + 3600})


@pytest.mark.parametrize("before,after", [
    ('size="0.03 0.03 0.02"', 'size="0.03 -0.03 0.02"'),
    ('parent link="body"', 'parent link="unrelated"'),
    ('effort="30"', 'effort="nan"'),
])
def test_malformed_engineering_model_refuses_before_retaining_native_inputs(tmp_path, monkeypatch, before, after):
    args, row = _fixture(tmp_path, monkeypatch)
    _replace_library_model(tmp_path, row, URDF.replace(before, after))
    with pytest.raises(ValueError, match="website_button_authoring_button_model"):
        preparation.compile_website_scene_preparation(**args)
    assert not list(args["output_root"].glob("family_authoring/stages/*/native-inputs.json"))


def test_unimplemented_asset_import_keeps_accepted_job_with_specific_software_gap(tmp_path, monkeypatch):
    args, row = _fixture(tmp_path, monkeypatch)
    _replace_library_model(tmp_path, row, URDF.replace('<box size="0.03 0.03 0.02"/>',
        '<mesh filename="body.stl"/>'))
    held = preparation.compile_website_scene_preparation(**args)
    request = held["task_evaluation_inputs"]["authoring"]["request"]
    assert held["status"] == "needs_input" and held["job_disposition"] == "accepted"
    assert request["next_action"]["quantity"] == "button_model_import_support"
    assert request["next_action"]["category"] == "runtime_integration"
    assert request["next_action"]["owner"] == "runtime_operator"
    assert held["task_evaluation_inputs"]["native_runtime_contract"] is None


def test_over_limit_sequence_preserves_all_operations_as_specific_preparation_gap(tmp_path, monkeypatch):
    args, _ = _fixture(tmp_path, monkeypatch, sequence=True)
    steps = [{"operation": "button_press", "description": "Press button " + str(index),
              "task_quote": "Press button " + str(index)} for index in range(1, 10)]
    args["task_context"]["description"] = ", then ".join(step["task_quote"] for step in steps) + "."
    _seal(args["task_context"], "context_digest")
    _rebind_semantics(args, steps=steps)
    held = preparation.compile_website_scene_preparation(**args)
    request = held["task_evaluation_inputs"]["authoring"]["request"]
    assert held["status"] == "needs_input" and held["job_disposition"] == "accepted"
    assert len(request["operations"]) == 9
    assert [row["source_operation"] for row in request["operations"]] == steps
    assert request["next_action"]["quantity"] == "button_sequence_support"
    assert request["next_action"]["category"] == "runtime_integration"
    assert held["task_evaluation_inputs"]["native_runtime_contract"] is None


def test_changed_retained_frame_bytes_refuse_before_native_inputs(tmp_path, monkeypatch):
    args, _ = _fixture(tmp_path, monkeypatch)
    Path(args["source_geometry"]["frames"][0]["geometry_path"]).write_bytes(b"changed-depth")
    with pytest.raises(ValueError, match="source_frame_bytes_changed"):
        preparation.compile_website_scene_preparation(**args)


def test_changed_job_pass_mark_cannot_reuse_older_normal_fact_confirmation(tmp_path, monkeypatch):
    args, _ = _fixture(tmp_path, monkeypatch)
    args["task_context"]["success_criteria"] = {"successDefinition": "Hold the button active for two seconds.",
        "successRate": None, "cycleTimeSeconds": None, "unknown": False}
    _seal(args["task_context"], "context_digest")
    with pytest.raises(ValueError, match="owner_facts_source_changed"):
        preparation.compile_website_scene_preparation(**args)


def test_unqualified_engine_can_retain_static_inputs_but_cannot_create_intake(tmp_path, monkeypatch):
    args, row = _fixture(tmp_path, monkeypatch)
    row.pop("qualification")
    catalog = {"schema_version": "task_family_native_runtime_catalog.v1", "runtimes": [row]}
    _seal(catalog, "catalog_digest")
    (tmp_path / "operator-library" / "catalog.json").write_text(json.dumps(catalog))
    result = preparation.compile_website_scene_preparation(**args)
    assert result["status"] == "task_evaluation_ready"
    assert result["task_evaluation_inputs"]["authoring"]["request"]["native_inputs_authored"] is True
    assert result["task_evaluation_inputs"]["native_runtime_contract"]
    assert result["intake_request"] is None
    assert "task_family_native_runtime_qualification_required" in result["dispatch_blockers"]


@pytest.mark.parametrize("sequence,general", [(False, False), (True, False), (True, True)])
def test_capture_handoff_and_existing_packet_reader_consume_authored_inputs_without_native_metadata(tmp_path, monkeypatch, sequence, general):
    from blueprint_pipeline.common import write_json
    from blueprint_pipeline.native_task_arena_bundle import verify_native_task_arena_packet
    from blueprint_pipeline.website_scene_handoff import prepare_website_scene_handoff
    from blueprint_pipeline.website_task_family_packet import materialize_website_task_family_packet

    args, _ = _fixture(tmp_path, monkeypatch, sequence=sequence, general=general)
    pipeline = tmp_path / "pipeline"
    pipeline.mkdir()
    removal = pipeline / "removal.json"
    write_json(removal, args["removal_manifest"])
    if sequence:
        semantics = args["task_semantics"]
        job = pipeline / "atlas"
        write_json(job / "selection_binding.json", {"planner_receipt": semantics["planner_receipt"]})
        write_json(job / "task_semantics" / (semantics["digest"][7:] + ".json"), semantics)
    monkeypatch.setenv("BLUEPRINT_WEBSITE_SCENE_BINDING_ROOT", str(tmp_path / "bindings"))
    monkeypatch.setattr("blueprint_pipeline.website_atlas_scene_source.atlas_scene_source", lambda **_: {
        "status": "ready", "base_scene": args["base_scene"], "source_geometry": args["source_geometry"],
        "source_pose_digest": "sha256:" + "b" * 64, "background_pose_digest": "sha256:" + "c" * 64,
        "alignment": "fixture-measured-registration"})
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.bind_task_masks_to_geometry", lambda **_: args["task_masks"])
    context = args["task_context"]
    result = prepare_website_scene_handoff(descriptor={"scene_id": context["scene_id"], "capture_id": context["capture_id"],
        "metadata": {"site_task_context": context, "website_scene_execution_authority": args["spend"]}},
        clean_plate={"privacy_verified": True, "status": "objects_removed", "removal_manifest_path": str(removal),
                     "task_masks": args["task_masks"]}, capture_root=tmp_path, now=args["now"])
    assert result["status"] == "intake_ready", result
    assert result["simulator_ready"] is False and result["provider_mutation_performed"] is False
    prepared = json.loads(Path(result["preparation_path"]).read_text())
    native_path = pipeline / "authored-native.json"
    write_json(native_path, prepared["task_evaluation_inputs"]["native_runtime_contract"])
    record = file_record(native_path)
    packet = materialize_website_task_family_packet(preparation=prepared, runtime_inputs_path=native_path,
        output_root=tmp_path / "authored-packet", intent_reference=record, attempt_reference=record,
        release_reference=record, task_context=context)
    _root, receipt, _ = verify_native_task_arena_packet(packet["packet_root"])
    assert receipt["native_application_claimed"] is False and receipt["policy_episode_claimed"] is False
