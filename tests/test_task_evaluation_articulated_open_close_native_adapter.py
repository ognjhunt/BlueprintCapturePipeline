"""The configured drawer task compiles to a native contract and a real scene plan, on CPU.

This drives the whole articulated chain against actually composed USD bytes:
assembly -> static qualification -> native adapter -> native runtime contract ->
native arena scene plan. Nothing here rents a GPU; the point is that a defect in
the articulated contract surfaces here instead of on a rented one.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_scene_plan import _articulation_plan
from blueprint_pipeline.native_task_arena_packet import _validated_scenario_context
from blueprint_pipeline.native_task_runtime_contract import (
    NativeTaskRuntimeContractError, materialize_native_task_runtime_contract,
)
from blueprint_pipeline.task_evaluation_articulated_open_close_native_adapter import (
    SCHEMA_VERSION, TaskEvaluationArticulatedOpenCloseNativeAdapterError,
    adapt_articulated_open_close_task_template,
)
from blueprint_pipeline.task_evaluation_scene_configuration_articulated_static_qualification import (
    qualify_scene_configuration_articulated_asset_static,
)
from blueprint_pipeline.task_evaluation_scene_configuration_submission_records import (
    articulated_open_close_task_records,
)
from tests.test_task_evaluation_configured_scene_revision import revision

DEFINITION = "scene.configured_revision.task_template.definition"
SUCCESS = "scene.configured_revision.task_template.success_criteria"
EXECUTION = "scene.configured_revision.task_template.execution"
SUPPORT = "scene.configured_revision.registration.support_plane"
SOURCE_OBJECT = "scene.configured_revision.replacement.source_object"
STATIC = "scene.configured_revision.replacement.static_qualification"
NATIVE_IMPORT = "scene.configured_revision.replacement.native_import_qualification"
SUPPORT_TOP_Z_M = 0.0
OPENING_FRACTION = 0.6


def _assembly(tmp_path: Path):
    """Compose the real assembly and statically qualify it, exactly as stages 3-4 do."""
    from tests.test_task_evaluation_scene_configuration_articulated_static_qualification import _sealed
    asset, graph_spec, authoring = _sealed(tmp_path / "assembly")
    static = qualify_scene_configuration_articulated_asset_static(
        asset_path=asset, graph_spec=graph_spec, authoring_receipt=authoring,
        replacement_identity={"id": "website-subject-cab", "version": "v1"},
        output_path=tmp_path / "static_qualification.json")
    return asset, graph_spec, static


def _native_import(identity: dict) -> dict:
    value = {
        "schema_version": "task_evaluation_replacement_native_import_result.v1",
        "status": "qualified", "replacement_identity": dict(identity),
        "native_simulator_import_qualified": True, "blockers": [],
        "asset_kind": "articulated_assembly", "result_digest": "",
    }
    value["result_digest"] = canonical_digest(value, digest_field="result_digest")
    return value


def _case(tmp_path: Path, *, opening_fraction: float = OPENING_FRACTION, assembly=None, mechanism=None):
    """``assembly`` (asset, graph_spec, static) and ``mechanism`` default to the drawer cabinet."""
    asset, graph_spec, static = assembly or _assembly(tmp_path)
    plan = graph_spec["assembly_plan"]
    configured = revision()
    identity = {"id": "website-subject-cab", "version": "v1"}
    configured["replacement"]["identity"] = identity
    configured["task_template"]["identity"] = {"id": "website-drawer-open", "version": "v1"}
    static = json.loads(json.dumps(static))
    static["replacement_identity"] = identity
    static["result_digest"] = ""
    static["result_digest"] = canonical_digest(static, digest_field="result_digest")
    dims = plan["assembly_dimensions_m"]
    lower = [-dims["depth_x"] / 2, -dims["width_y"] / 2, 0.0]
    upper = [dims["depth_x"] / 2, dims["width_y"] / 2, dims["height_z"]]
    template, success, execution = articulated_open_close_task_records(
        task_identity=configured["task_template"]["identity"], object_identity=identity,
        start_center=[0.4, 0.1, dims["height_z"] / 2], source_min=lower, source_max=upper,
        mechanism=mechanism or {
            "part_label": "middle drawer", "joint_type": "prismatic",
            "estimated_usable_stroke_m": plan["task_joint"]["limits_m"][1],
            "estimated_front_normal_world": plan["assembly_frame"]["estimated_front_normal_world"],
            "lock_status": "unknown", "travel_authority": "object_prior_estimate"},
        success={"control_frequency_hz": 15, "maximum_episode_seconds": 30,
                 "minimum_opening_fraction_of_estimated_stroke": opening_fraction,
                 "minimum_hold_seconds": 1.0, "maximum_retries": 0},
        resolved_seed=1)
    template["instruction"] = "Open the middle drawer of the three-drawer wood-front cabinet."
    template["instruction_subject_label"] = "three-drawer wood cabinet"
    template["visible_target_label"] = "middle drawer"
    documents = {
        DEFINITION: template, SUCCESS: success, EXECUTION: execution,
        SUPPORT: {"schema_version": "task_evaluation_support_plane_input.v1",
                  "status": "frozen_candidate_pending_production_validation",
                  "sage_prim_path": "/Root", "top_z_m": SUPPORT_TOP_Z_M,
                  "bounds_min_xyz_m": [-1.0, -1.0, -0.02], "bounds_max_xyz_m": [1.0, 1.0, SUPPORT_TOP_Z_M]},
        SOURCE_OBJECT: {"schema_version": "task_evaluation_source_object_selection.v1",
                        "status": "frozen_before_scene_configuration_run",
                        "aabb_min_xyz_m": lower, "aabb_max_xyz_m": upper},
        STATIC: static, NATIVE_IMPORT: _native_import(identity),
    }
    references: dict[str, dict] = {}
    for index, (contract_path, document) in enumerate(documents.items()):
        payload = (json.dumps(document, sort_keys=True) + "\n").encode("utf-8")
        path = tmp_path / f"reference-{index}.json"
        path.write_bytes(payload)
        reference = {"uri": f"s3://blueprint-production-inputs/{path.name}",
                     "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
                     "size_bytes": len(payload)}
        references[contract_path] = {"contract_path": contract_path, **reference,
                                     "materialized_path": str(path),
                                     "full_byte_service_account_readback_passed": True}
        section, field = {
            DEFINITION: ("task_template", "definition"), SUCCESS: ("task_template", "success_criteria"),
            EXECUTION: ("task_template", "execution"), SUPPORT: ("registration", "support_plane"),
            SOURCE_OBJECT: ("replacement", "source_object"), STATIC: ("replacement", "static_qualification"),
            NATIVE_IMPORT: ("replacement", "native_import_qualification"),
        }[contract_path]
        configured[section][field] = reference
    configured["revision_digest"] = ""
    configured["revision_digest"] = canonical_digest(configured, digest_field="revision_digest")
    return {"asset": asset, "plan": plan, "configured": configured,
            "references": references, "documents": documents, "static": static}


def test_configured_drawer_task_adapts_to_a_passive_joint_native_definition(tmp_path: Path) -> None:
    case = _case(tmp_path)
    result = adapt_articulated_open_close_task_template(
        configured_revision=case["configured"], materialized_references=case["references"])
    assert result["schema_version"] == SCHEMA_VERSION and result["status"] == "adapted"
    assert result["native_task_kind"] == "articulated_open_close"
    assert result["external_task_kind"] == "articulated_manipulation"
    assert result["adapter_digest"] == canonical_digest(result, digest_field="adapter_digest")
    assert result["claim_boundary"] == {"task_joint_is_passive": True, "policy_must_open_the_part": True,
                                        "joint_travel_is_measured": False,
                                        "simulator_execution_is_not_physical_truth": True}
    definition = result["native_task_definition"]
    spec = definition["task_spec"]
    assert spec["schema_version"] == "adp_task_spec.v2" and spec["task_kind"] == "articulated_open_close"
    assert "destination_position_bounds_world_m" not in spec and "minimum_lift_m" not in spec
    stroke = case["plan"]["task_joint"]["limits_m"][1]
    threshold = spec["executable_opening_threshold"]
    assert threshold["target_joint_id"] == "task_part_joint" and threshold["joint_type"] == "prismatic"
    assert threshold["success_interval"][0] == pytest.approx(OPENING_FRACTION * stroke, abs=1e-4)
    assert threshold["success_interval"][1] == pytest.approx(stroke, abs=1e-6)
    assert threshold["authority"] == "frozen_from_static_qualification_of_exact_asset_bytes"
    assert threshold["travel_is_measured"] is False
    # One second of hold at 15 Hz is fifteen settled samples.
    assert threshold["hold_window_samples"] == 15 == spec["settle_window_samples"]
    affordance = spec["interaction_affordance"]
    assert affordance["contact_link_id"] == "drawer_1"
    assert affordance["contact_body_prim_paths"] == ["/Asset/links/drawer_1"]
    assert affordance["handle_prim_paths"] == ["/Asset/links/drawer_1/collision/handle"]
    assert affordance["task_joint_drive_forbidden"] is True
    assert affordance["affordance_digest"] == canonical_digest(affordance, digest_field="affordance_digest")
    # The pull direction is the joint's own opening axis; approach is its inverse.
    assert affordance["pull_unit_asset_root"] == [1.0, 0.0, 0.0]
    assert affordance["approach_unit_asset_root"] == [-1.0, 0.0, 0.0]
    bindings = {row["joint_id"]: row for row in definition["task_joint_bindings"]}
    assert set(bindings) == {"task_part_joint", "drawer_0_fixed", "drawer_2_fixed"}
    assert bindings["task_part_joint"]["readback_kind"] == "native_coordinate"
    assert bindings["task_part_joint"]["native_joint_name"] == "task_part_joint"
    assert all(bindings[j]["readback_kind"] == "fixed_joint_static" for j in ("drawer_0_fixed", "drawer_2_fixed"))
    assert all(bindings[j]["static_qualification_digest"] == case["static"]["result_digest"]
               for j in ("drawer_0_fixed", "drawer_2_fixed"))
    state = definition["task_state_binding"]
    assert state["schema_version"] == "native_articulated_graph_task_state_binding.v1"
    assert set(state["link_native_body_names"]) == {"carcass", "drawer_0", "drawer_1", "drawer_2"}
    assert state["interaction_affordance_digest"] == affordance["affordance_digest"]
    # The assembly's own base lands on the registered support top, not below it.
    assert definition["task_object_pose_world"]["position_world_m"][2] == pytest.approx(SUPPORT_TOP_Z_M, abs=1e-9)
    assert definition["task_object_reset_joint_positions"] == {"task_part_joint": 0.0}
    execution = result["native_episode_execution"]
    assert execution["control_decimation"] == 8 and execution["maximum_step_count"] == 450


def test_adapter_refuses_a_threshold_that_contradicts_the_qualified_joint_limit(tmp_path: Path) -> None:
    """The template may declare a fraction; the executable interval comes from the bytes."""
    case = _case(tmp_path)
    references = copy.deepcopy(case["references"])
    document = copy.deepcopy(case["documents"][DEFINITION])
    document["success"]["minimum_opening_fraction_of_usable_travel"] = 0.95
    payload = (json.dumps(document, sort_keys=True) + "\n").encode("utf-8")
    path = tmp_path / "definition-rewritten.json"
    path.write_bytes(payload)
    configured = copy.deepcopy(case["configured"])
    reference = {"uri": "s3://blueprint-production-inputs/definition-rewritten.json",
                 "digest": "sha256:" + hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
    references[DEFINITION] = {"contract_path": DEFINITION, **reference, "materialized_path": str(path),
                              "full_byte_service_account_readback_passed": True}
    configured["task_template"]["definition"] = reference
    configured["revision_digest"] = ""
    configured["revision_digest"] = canonical_digest(configured, digest_field="revision_digest")
    with pytest.raises(TaskEvaluationArticulatedOpenCloseNativeAdapterError,
                       match="success_threshold_disagrees_with_qualified_limit|configured_task_documents_invalid"):
        adapt_articulated_open_close_task_template(
            configured_revision=configured, materialized_references=references)


def test_adapter_refuses_an_unqualified_or_driven_assembly(tmp_path: Path) -> None:
    case = _case(tmp_path)
    for mutate, expected in (
        (lambda d: d.__setitem__("status", "authored_structure_statically_qualified_with_findings"),
         "executable_geometry_missing"),
        (lambda d: d["articulation_graph"]["joints"].__setitem__(
            [row["role"] for row in d["articulation_graph"]["joints"]].index("target"),
            {**next(r for r in d["articulation_graph"]["joints"] if r["role"] == "target"),
             "drive": {"drive_type": "force", "stiffness": 500.0, "damping": 5.0, "maximum_force": 50.0}}),
         "articulation_graph_invalid"),
    ):
        static = copy.deepcopy(case["static"])
        mutate(static)
        static["result_digest"] = ""
        static["result_digest"] = canonical_digest(static, digest_field="result_digest")
        payload = (json.dumps(static, sort_keys=True) + "\n").encode("utf-8")
        path = tmp_path / f"static-{expected}.json"
        path.write_bytes(payload)
        references = copy.deepcopy(case["references"])
        configured = copy.deepcopy(case["configured"])
        reference = {"uri": f"s3://blueprint-production-inputs/{path.name}",
                     "digest": "sha256:" + hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
        references[STATIC] = {"contract_path": STATIC, **reference, "materialized_path": str(path),
                              "full_byte_service_account_readback_passed": True}
        configured["replacement"]["static_qualification"] = reference
        configured["revision_digest"] = ""
        configured["revision_digest"] = canonical_digest(configured, digest_field="revision_digest")
        with pytest.raises(TaskEvaluationArticulatedOpenCloseNativeAdapterError, match=expected):
            adapt_articulated_open_close_task_template(
                configured_revision=configured, materialized_references=references)


def _runtime_contract(case, adapted, tmp_path: Path, asset_directory: Path):
    """Stage the real assembly bytes and freeze the native runtime contract."""
    import shutil
    asset_directory.mkdir(parents=True, exist_ok=True)
    task_object = asset_directory / "task_object.usdz"
    shutil.copyfile(case["asset"], task_object)
    scene_collision = asset_directory / "scene_collision.usda"
    appearance = asset_directory / "scene_appearance.usda"
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateNew(str(scene_collision))
    root = UsdGeom.Xform.Define(stage, "/Scene")
    stage.SetDefaultPrim(root.GetPrim())
    floor = UsdGeom.Cube.Define(stage, "/Scene/floor").GetPrim()
    UsdPhysics.CollisionAPI.Apply(floor)
    stage.GetRootLayer().Save()
    appearance.write_text("#usda 1.0\n(\n    defaultPrim = \"Scene\"\n)\ndef Xform \"Scene\" {}\n")

    def digest(path: Path) -> str:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    definition = adapted["native_task_definition"]
    spec = definition["task_spec"]
    assets = [
        {"asset_id": "scene_appearance", "semantic_role": "scene_appearance", "filename": appearance.name,
         "sha256": digest(appearance), "size_bytes": appearance.stat().st_size,
         "object_type": "RIGID", "pose_world": {"position_world_m": [0.0, 0.0, 0.0],
                                                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
         "reset_state": {"position_world_m": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                         "joint_positions": {}}},
        {"asset_id": "scene_collision", "semantic_role": "scene_collision", "filename": scene_collision.name,
         "sha256": digest(scene_collision), "size_bytes": scene_collision.stat().st_size,
         "object_type": "RIGID", "pose_world": {"position_world_m": [0.0, 0.0, 0.0],
                                                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
         "reset_state": {"position_world_m": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                         "joint_positions": {}}},
        {"asset_id": spec["subject_asset_id"], "semantic_role": "task_object", "filename": task_object.name,
         "sha256": digest(task_object), "size_bytes": task_object.stat().st_size,
         "object_type": "ARTICULATION", "task_subject": True,
         "pose_world": definition["task_object_pose_world"],
         "reset_state": {**definition["task_object_pose_world"],
                         "joint_positions": definition["task_object_reset_joint_positions"]}},
    ]
    return assets, task_object, scene_collision


def materialize_contract_and_plan(case, adapted, tmp_path: Path):
    """Freeze the native runtime contract and articulation plan for one adapted task, on CPU."""
    assets, task_object, scene_collision = _runtime_contract(
        case, adapted, tmp_path, tmp_path / "staged")
    definition = adapted["native_task_definition"]
    execution = adapted["native_episode_execution"]
    def pose(translation):
        tx, ty, tz = translation
        return [1.0, 0.0, 0.0, tx, 0.0, 1.0, 0.0, ty, 0.0, 0.0, 1.0, tz, 0.0, 0.0, 0.0, 1.0]

    intrinsics = {"fx": 240.0, "fy": 240.0, "cx": 160.0, "cy": 120.0, "width": 320, "height": 240}
    cameras = [
        {"camera_id": "external", "role": "external", "policy_input": True, "scoring_input": False,
         "pose_frame": "world", "parent_prim_path": "{ENV_REGEX_NS}",
         "frame_from_camera_matrix": pose([1.2, -0.8, 0.9]),
         "optical_convention": "opencv", "intrinsics": dict(intrinsics)},
        {"camera_id": "wrist", "role": "wrist", "policy_input": True, "scoring_input": False,
         "pose_frame": "robot_body", "parent_prim_path": "{ENV_REGEX_NS}/Robot/panda_hand",
         "frame_from_camera_matrix": pose([0.0, 0.0, 0.05]),
         "optical_convention": "opencv", "intrinsics": dict(intrinsics)},
        {"camera_id": "overview", "role": "overview", "policy_input": False, "scoring_input": False,
         "pose_frame": "world", "parent_prim_path": "{ENV_REGEX_NS}",
         "frame_from_camera_matrix": pose([0.0, -1.6, 1.4]),
         "optical_convention": "opencv", "intrinsics": dict(intrinsics)},
    ]
    from blueprint_pipeline.native_task_runtime_contract import DROID_FRANKA_RESET_JOINT_NAMES
    try:
        contract = materialize_native_task_runtime_contract(
            scene_id="site-capture-drawer", task_id=definition["identity"]["id"],
            task_spec=definition["task_spec"],
            task_joint_bindings=definition["task_joint_bindings"],
            task_state_binding=definition["task_state_binding"],
            assets=assets,
            robot_base_pose_world={"position_world_m": [-0.5, 0.0, 0.0],
                                   "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
            robot_joint_reset_positions_rad=dict.fromkeys(DROID_FRANKA_RESET_JOINT_NAMES, 0.0),
            cameras=cameras,
            scenario_cell_id=execution["scenario"]["cell_id"],
            scenario_instance_digest=execution["scenario"]["instance_digest"],
            seed=execution["scenario"]["seed"])
    except NativeTaskRuntimeContractError as exc:  # pragma: no cover - surfaced as the assertion
        pytest.fail("articulated runtime contract refused: " + ";".join(exc.errors))
    plan = _articulation_plan(contract, task_object_asset_path=task_object,
                              scene_collision_asset_path=scene_collision)
    return contract, plan


def test_adapted_drawer_task_freezes_a_native_contract_and_scene_plan(tmp_path: Path) -> None:
    """End to end on CPU: qualified bytes -> adapter -> runtime contract -> articulation plan."""
    case = _case(tmp_path)
    adapted = adapt_articulated_open_close_task_template(
        configured_revision=case["configured"], materialized_references=case["references"])
    scenario = adapted["native_episode_execution"]["scenario"]
    assert _validated_scenario_context(scenario) == scenario
    contract, plan = materialize_contract_and_plan(case, adapted, tmp_path)
    assert contract["task_kind"] == "articulated_open_close"
    assert contract["runtime_readback_required"]["task_joint_indices"] is True
    assert contract["scoring_contract"]["policy_may_grade_itself"] is False
    sample = contract["task_sample_binding"]
    assert set(sample["joint_ids"]) == {"task_part_joint", "drawer_0_fixed", "drawer_2_fixed"}
    assert sample["native_coordinate_joint_ids"] == ["task_part_joint"]
    assert sorted(sample["fixed_joint_ids"]) == ["drawer_0_fixed", "drawer_2_fixed"]
    subject = next(row for row in contract["objects"] if row.get("task_subject") is True)
    assert subject["object_type"] == "ARTICULATION"
    assert subject["reset_state"]["joint_positions"] == {"task_part_joint": 0.0}
    assert plan["graph_articulation"] is True
    assert plan["task_joint_reset_positions_rad"] == {"task_part_joint": 0.0}
    assert plan["interaction_link_native_body_name"] == "drawer_1"
    assert plan["task_contact_body_paths"] == ["{ENV_REGEX_NS}/task_object/links/drawer_1"]
    assert set(plan["task_joint_prim_paths"]) == {"task_part_joint", "drawer_0_fixed", "drawer_2_fixed"}
    assert plan["state_thresholds"]["root_translation_tolerance_m"] == pytest.approx(0.02)
    assert plan["gpu_collision_qualification"]["status"] == "qualified"
    assert {row["logical_sensor_id"] for row in plan["contact_sensors"]} == {
        "task_robot_contact", "task_scene_contact", "robot_task_forbidden_collision", "robot_scene_contact"}


def test_runtime_contract_refuses_a_rigid_spawn_of_the_assembly(tmp_path: Path) -> None:
    """An articulated task whose subject spawns RIGID would never expose a joint to read."""
    case = _case(tmp_path)
    adapted = adapt_articulated_open_close_task_template(
        configured_revision=case["configured"], materialized_references=case["references"])
    assets, _task_object, _scene = _runtime_contract(case, adapted, tmp_path, tmp_path / "staged")
    for row in assets:
        if row.get("task_subject"):
            row["object_type"] = "RIGID"
            row["reset_state"] = {k: v for k, v in row["reset_state"].items() if k != "joint_positions"}
            row["reset_state"]["joint_positions"] = {}
    definition = adapted["native_task_definition"]
    from blueprint_pipeline.native_task_runtime_contract import DROID_FRANKA_RESET_JOINT_NAMES
    with pytest.raises(NativeTaskRuntimeContractError) as error:
        materialize_native_task_runtime_contract(
            scene_id="site-capture-drawer", task_id=definition["identity"]["id"],
            task_spec=definition["task_spec"],
            task_joint_bindings=definition["task_joint_bindings"],
            task_state_binding=definition["task_state_binding"], assets=assets,
            robot_base_pose_world={"position_world_m": [-0.5, 0.0, 0.0],
                                   "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
            robot_joint_reset_positions_rad=dict.fromkeys(DROID_FRANKA_RESET_JOINT_NAMES, 0.0),
            cameras=[], scenario_cell_id="cell", scenario_instance_digest="sha256:" + "a" * 64, seed=1)
    # The subject may not spawn rigid for an articulated task, and a rigid
    # subject may not carry joint reset state: either way the contract refuses
    # before anything reaches a simulator.
    message = str(error.value)
    assert ("articulated_spawn_required" in message
            or "composition_invalid" in message
            or "reset_state" in message), message
