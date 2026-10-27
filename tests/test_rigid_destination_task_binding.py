"""Qualified supplemental containers retain the same task from probe to packet."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.rigid_destination_task_binding import (
    BUNDLE_CONTRACT, destination_references, stage_revision_destination,
)
from blueprint_pipeline.rigid_robot_free_probe_plan import physics_plan, probe_context
from blueprint_pipeline.rigid_robot_free_probe_recorder import record_rigid_robot_free_probe
from blueprint_pipeline.task_evaluation_rigid_destination_geometry import derive_rigid_destination_geometry
from blueprint_pipeline import task_evaluation_scene_rigid_robot_free_readiness as readiness
from blueprint_pipeline.task_evaluation_rigid_relocation_native_adapter import (
    STATIC_QUALIFICATION_CONTRACT_PATH, SUPPORT_PLANE_CONTRACT_PATH,
)
from blueprint_pipeline.team_candidate_quick10_registry import scene_facts
from blueprint_pipeline.team_scene_packet import TeamScenePacketError, materialize_team_scene
from tests.test_rigid_robot_free_probe import FlatSupport, binding
from tests.test_task_evaluation_native_arena_episode_compiler import _destination_case
from tests.test_task_evaluation_rigid_destination_geometry import LIMITS
from tests.test_team_scene_packet import _box, _scene, _stage


def _sha(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value, digest_field=None):
    if digest_field:
        value[digest_field] = canonical_digest(value, digest_field=digest_field)
    Path(path).write_text(json.dumps(value, sort_keys=True) + "\n")
    return Path(path)


def _reference(path, name):
    return {"uri": "s3://fixture/" + name, "digest": _sha(path), "size_bytes": Path(path).stat().st_size}


def _simready(identity):
    return {"schema_version": "task_evaluation_passive_destination_simready.v1",
        "destination_identity": identity, "intended_support_prim_paths": ["/Tray"],
        "intended_support_collision_prim_paths": ["/Tray/Bottom"],
        "interior_bounds_body_frame_m": {"minimum": [-.11, -.11, .005], "maximum": [.11, .11, .2]}}


def _case(tmp_path):
    from pxr import UsdPhysics

    scene = _scene(tmp_path, strategy="pick_and_place")
    revision, documents = scene["revision"], scene["documents"]
    static_path = Path(documents[STATIC_QUALIFICATION_CONTRACT_PATH])
    static = json.loads(static_path.read_text())
    static["observed_structure"]["collision_bounds_body_frame_m"] = {
        "minimum": [-0.0638749, -0.0658936, 0.0], "maximum": [0.0638749, 0.0658936, 0.127638]}
    static_path = _write(tmp_path / "subject-static-updated.json", static, "result_digest")
    documents[STATIC_QUALIFICATION_CONTRACT_PATH] = static_path
    revision["replacement"]["static_qualification"] = _reference(static_path, "subject-static.json")
    request, refs, _ = _destination_case(tmp_path, subject_identity=revision["replacement"]["identity"],
        subject_static_path=static_path, subject_static=static,
        subject_scoring_transform={"position_m": [0.0, 0.0, 0.063819], "orientation_xyzw": [0., 0., 0., 1.]},
        configured_scene_collision_path=tmp_path / "room.usda",
        configured_scene_support_plane_path=documents[SUPPORT_PLANE_CONTRACT_PATH])
    destination = request["task"]["destination"]
    destination["pose_world"]["position_world_m"][2] = 0.7545
    destination["native_probe"] = {"schema_version": "task_evaluation_rigid_destination_native_probe_configuration.v1",
        "placement_support_scene_prim_paths": ["/Root/Support"], "qualification_limits": dict(LIMITS),
        "settle_sample_count": 3, "settle_steps_per_sample": 60}
    asset = Path(refs["task.destination.asset"]["materialized_path"])
    asset.unlink()
    stage = _stage(asset, "/Tray")
    UsdPhysics.RigidBodyAPI.Apply(stage.GetDefaultPrim())
    UsdPhysics.MassAPI.Apply(stage.GetDefaultPrim()).CreateMassAttr(0.4)
    _box(stage, "/Tray/Bottom", (-.12, -.12, 0.), (.12, .12, .005))
    _box(stage, "/Tray/Left", (-.12, -.12, .005), (-.11, .12, .2))
    _box(stage, "/Tray/Right", (.11, -.12, .005), (.12, .12, .2))
    _box(stage, "/Tray/Front", (-.11, -.12, .005), (.11, -.11, .2))
    _box(stage, "/Tray/Back", (-.11, .11, .005), (.11, .12, .2))
    stage.GetRootLayer().Save()
    destination_static_path = Path(refs["task.destination.static_qualification"]["materialized_path"])
    destination_static = json.loads(destination_static_path.read_text())
    destination_static["replacement_usd"].update(sha256=_sha(asset), size_bytes=asset.stat().st_size)
    destination_static["observed_structure"].update(collision_bounds_body_frame_m={
        "minimum": [-.12, -.12, 0.], "maximum": [.12, .12, .2]},
        collision_prim_paths=["/Tray/" + name for name in ("Bottom", "Left", "Right", "Front", "Back")])
    _write(destination_static_path, destination_static, "result_digest")
    native_path = Path(refs["task.destination.native_import_qualification"]["materialized_path"])
    native = json.loads(native_path.read_text())
    native.update(asset_digest=_sha(asset), static_qualification_digest=_sha(destination_static_path))
    _write(native_path, native, "result_digest")
    simready = _simready(destination["identity"])
    geometry = derive_rigid_destination_geometry(subject_identity=revision["replacement"]["identity"],
        destination_identity=destination["identity"], relation="inside", pose_world=destination["pose_world"],
        subject_static_qualification=static, subject_static_qualification_digest=_sha(static_path),
        subject_scoring_transform={"position_m": [0., 0., .063819], "orientation_xyzw": [0., 0., 0., 1.]},
        destination_static_qualification=destination_static,
        destination_static_qualification_digest=_sha(destination_static_path),
        destination_simready_result=simready, qualification_limits=LIMITS)
    geometry_path = _write(refs["task.destination.geometry"]["materialized_path"], geometry)
    for field in ("asset", "rights_admission", "static_qualification", "native_import_qualification", "geometry"):
        key = "task.destination." + field
        path = geometry_path if field == "geometry" else Path(refs[key]["materialized_path"])
        documents[key] = path
        destination[field] = _reference(path, field)
    revision["task_template"]["destination"] = destination
    revision["revision_digest"] = canonical_digest(revision, digest_field="revision_digest")
    documents[BUNDLE_CONTRACT] = scene["configured_scene_bundle_path"]
    placement_path = Path(refs["task.destination.placement_qualification"]["materialized_path"])
    placement = json.loads(placement_path.read_text())
    placement.update(configured_scene_revision_digest=revision["revision_digest"], destination_asset_digest=_sha(asset),
        destination_static_qualification_digest=_sha(destination_static_path),
        destination_native_import_qualification_digest=_sha(native_path), destination_geometry_digest=geometry["geometry_digest"],
        pose_world=destination["pose_world"])
    _write(placement_path, placement, "placement_qualification_digest")
    profile_path = scene["droid_profile_path"]
    profile = json.loads(profile_path.read_text())
    profile["destination_execution_binding"] = {"execution_commit": "a" * 40,
        "runtime_identity": {"id": "native-arena", "version": "v1"}}
    _write(profile_path, profile, "request_digest")
    scene["droid_profile_digest"] = _sha(profile_path)
    return scene, placement_path


def test_qualified_container_has_identical_probe_and_policy_task_and_retained_receipt(tmp_path):
    scene, placement = _case(tmp_path)
    kwargs = {key: value for key, value in scene.items() if key not in ("profile", "store", "revision_reference")}
    with pytest.raises(TeamScenePacketError, match="placement_qualification_required"):
        materialize_team_scene(**kwargs, output_root=tmp_path / "missing-placement")
    compiled = materialize_team_scene(**kwargs, output_root=tmp_path / "prepared",
        destination_placement_qualification_path=placement)
    facts = scene_facts({"team_scene_inputs_path": compiled["path"], "open_loop_horizon": 8})
    task = readiness.rigid_task_for_revision(revision=scene["revision"], documents=scene["documents"])
    assert task["task_spec"] == facts["scene"]["task_spec"]
    spec, pose = task["task_spec"], task["task_spec"]["destination_pose_world"]

    class ContainerStandIn(FlatSupport):
        def _ground(self):
            return spec["target_position_world_m"][2] if self.pose[0] > 3.1 else super()._ground()

        def read_sample(self):
            return {**super().read_sample(), "destination_pose_world": pose}

    trace = record_rigid_robot_free_probe(scene=ContainerStandIn(spec["start_pose_world"]),
        plan=physics_plan(probe_context(spec)), binding=binding(spec, scene["revision"]["revision_digest"]))
    from blueprint_pipeline import task_evaluation_configured_controls_deferred_inputs as inputs
    refs = {**inputs._document_references(scene["revision"]), **destination_references(scene["revision"]),
        BUNDLE_CONTRACT: scene["revision"]["configured_scene_bundle"]}
    store = {ref["uri"]: Path(scene["documents"][key]).read_bytes() for key, ref in refs.items()}
    receipt = readiness.rigid_robot_free_readiness_for_revision(revision=scene["revision"],
        output_root=tmp_path / "retained", physics_probe=trace, fetcher=lambda ref: store[ref["uri"]])
    assert receipt["status"] == "passed", receipt.get("probe_error")
    assert receipt["task_spec_digest"] == canonical_digest(facts["scene"]["task_spec"])
    assert task["destination_probe"]["asset_digest"] == _sha(scene["documents"]["task.destination.asset"])
    assert Path(task["destination_probe"]["asset_path"]).suffix == ".usda"


def test_construction_destination_plan_is_identical_after_immutable_publication(tmp_path):
    from blueprint_pipeline.rigid_robot_free_construction_probe import (
        SOURCE_ALIASES, DESTINATION_SOURCE_ALIASES, TRACE_FIELD, prepare_construction_probe,
        published_construction_probe,
    )
    from blueprint_pipeline.task_evaluation_rigid_relocation_native_adapter import NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH

    scene, _placement = _case(tmp_path)
    revision, documents = scene["revision"], scene["documents"]
    native_path = tmp_path / "subject-native-with-construction.json"
    native = json.loads(Path(documents[NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH]).read_text())
    native["asset_digest"] = _sha(tmp_path / "replacement.usda")
    native["result_digest"] = canonical_digest(native, digest_field="result_digest")
    aliases = {**SOURCE_ALIASES, **DESTINATION_SOURCE_ALIASES}
    rows = [{"contract_path": alias, "materialized_path": str(documents[key]),
        **_reference(documents[key], alias), "full_byte_service_account_readback_passed": True}
        for key, alias in aliases.items()]
    simready = _write(tmp_path / "destination-simready.json", _simready(revision["task_template"]["destination"]["identity"]))
    rows.append({"contract_path": "construction.recipe.supplemental_destination.simready_result",
        "materialized_path": str(simready), **_reference(simready, "simready"),
        "full_byte_service_account_readback_passed": True})
    envelope = {"run_id": revision["configuration_run_id"], "team_namespace": revision["team_namespace"],
        "request": {"scene": {"identity": revision["scene_identity"]}, "task": {
            "strategy": "pick_and_place", "destination": revision["task_template"]["destination"]}},
        "materialized_references": rows}
    prepared = prepare_construction_probe(envelope=envelope, asset_path=tmp_path / "replacement.usda",
        static_path=Path(documents[STATIC_QUALIFICATION_CONTRACT_PATH]), native_import=native)
    task = readiness.rigid_task_for_revision(revision=revision, documents=documents)
    assert prepared["plan"] == physics_plan(probe_context(task["task_spec"]))
    raw = {"status": "recorded", "plan": prepared["plan"], "binding": prepared["binding"],
        "samples": [], "trace_digest": ""}
    raw["trace_digest"] = canonical_digest(raw, digest_field="trace_digest")
    native[TRACE_FIELD] = raw
    _write(native_path, native, "result_digest")
    documents[NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH] = native_path
    revision["replacement"]["native_import_qualification"] = _reference(native_path, "subject-native-with-construction")
    revision["revision_digest"] = canonical_digest(revision, digest_field="revision_digest")
    published_task = readiness.rigid_task_for_revision(revision=revision, documents=documents)
    projected = published_construction_probe(revision=revision, documents=documents, task=published_task)
    assert projected["construction_trace_digest"] == raw["trace_digest"]
    assert projected["plan"] == physics_plan(probe_context(published_task["task_spec"]))
    # A source list cannot conceal a trace simulated with another asset.
    raw["binding"]["destination_asset_digest"] = "sha256:" + "f" * 64
    raw["trace_digest"] = canonical_digest(raw, digest_field="trace_digest")
    native[TRACE_FIELD] = raw
    _write(native_path, native, "result_digest")
    with pytest.raises(ValueError, match="destination_geometry_mismatch"):
        published_construction_probe(revision=revision, documents=documents, task=published_task)


@pytest.mark.parametrize("runtime", [{}, {"id": "native-arena", "version": ""},
    {"id": "native-arena", "version": "v1", "unadmitted": True}])
def test_malformed_expected_runtime_cannot_admit_a_container(tmp_path, runtime):
    scene, placement = _case(tmp_path)
    with pytest.raises(ValueError, match="execution_binding_required"):
        stage_revision_destination(revision=scene["revision"], documents=scene["documents"], task_spec={},
            collision_path=tmp_path / "room.usda", output_root=tmp_path, require_placement=True,
            placement_qualification_path=placement,
            expected_execution_binding={"execution_commit": "a" * 40, "runtime_identity": runtime})
