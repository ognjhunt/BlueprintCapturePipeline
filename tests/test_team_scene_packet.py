"""A prepared scene compiles its own half of a team evaluation: no robot placement, no prior run (beta L5)."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline import task_evaluation_scene_rigid_robot_free_readiness as rigid
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_runtime_contract import DROID_FRANKA_RESET_JOINT_NAMES
from blueprint_pipeline.team_candidate_quick10_registry import scene_facts
from blueprint_pipeline.team_scene_packet import (
    STATION_FOOTPRINT_HALF_EXTENT_M,
    TeamScenePacketError,
    materialize_team_scene,
    read_team_scene_inputs,
)
from tests.test_native_task_arena_packet import _camera
from tests.test_rigid_robot_free_probe import rigid_case

SUPPORT_TOP_Z = 0.7545  # the fixture revision's registered support


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _box(stage: Any, path: str, low: tuple[float, ...], high: tuple[float, ...]) -> None:
    from pxr import UsdGeom, UsdPhysics

    mesh = UsdGeom.Mesh.Define(stage, path)
    (x0, y0, z0), (x1, y1, z1) = low, high
    mesh.CreatePointsAttr([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                           (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)])
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr([0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7])
    mesh.CreateExtentAttr([low, high])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())


def _stage(path: Path, root: str) -> Any:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, root).GetPrim())
    return stage


def _bundle(tmp_path: Path) -> Path:
    """The configured bundle: a room (floor and the registered support) shown as mesh, and the task object."""

    from pxr import UsdPhysics

    room = _stage(tmp_path / "room.usda", "/Root")
    _box(room, "/Root/Floor", (-2.0, -14.0, 0.0), (9.0, 4.0, 0.0))
    _box(room, "/Root/Support", (2.5, -9.5, 0.0), (4.5, -1.4, SUPPORT_TOP_Z))
    room.GetRootLayer().Save()
    subject = _stage(tmp_path / "replacement.usda", "/Asset")
    UsdPhysics.RigidBodyAPI.Apply(subject.GetDefaultPrim())
    UsdPhysics.MassAPI.Apply(subject.GetDefaultPrim()).CreateMassAttr(0.3)
    _box(subject, "/Asset/body", (-0.0638749, -0.0658936, 0.0), (0.0638749, 0.0658936, 0.127638))
    subject.GetRootLayer().Save()
    payloads = {"appearance.usda": (tmp_path / "room.usda").read_bytes(),
                "collision.usda": (tmp_path / "room.usda").read_bytes(),
                "replacement.usda": (tmp_path / "replacement.usda").read_bytes()}
    rows = [{"role": name.split(".")[0], "relative_path": name, "digest": _sha(data), "size_bytes": len(data)}
            for name, data in payloads.items()]
    manifest = {"schema_version": "task_evaluation_configured_scene_bundle_candidate.v1",
                "status": "assembled_pending_control_plane_publication", "robot_neutral": True,
                "robot_specific_base_registration_included": False, "assets": rows, "manifest_digest": ""}
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    path = tmp_path / "configured-scene.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
        archive.writestr("configured_scene_bundle_candidate.v1.json", json.dumps(manifest))
    return path


def _scene(tmp_path: Path) -> dict[str, Any]:
    """The rigid fixture revision, published with a configured bundle whose collision is a real room."""

    sources = tmp_path / "sources"
    sources.mkdir()
    configured, references, _docs, _spec = rigid_case(sources)
    bundle = _bundle(tmp_path)
    configured["configured_scene_bundle"] = {"uri": "s3://scenes/configured-scene.zip",
                                             "digest": _sha(bundle.read_bytes()), "size_bytes": bundle.stat().st_size}
    configured["revision_digest"] = canonical_digest(configured, digest_field="revision_digest")
    store = {row["uri"]: Path(row["materialized_path"]).read_bytes() for row in references.values()}
    documents = rigid.fetch_revision_documents(revision=configured, output_root=tmp_path / "work",
                                               fetcher=lambda reference: store[reference["uri"]])
    revision_bytes = json.dumps(configured, sort_keys=True).encode()
    store["s3://scenes/revision.json"] = revision_bytes
    profile = {"schema_version": "native_task_arena_packet_request.v1",
               "cameras": [_camera(role) for role in ("external", "wrist", "overview")],
               "robot_joint_reset_positions_rad": {name: 0.01 * index
                                                   for index, name in enumerate(DROID_FRANKA_RESET_JOINT_NAMES)},
               "request_digest": ""}
    profile["request_digest"] = canonical_digest(profile, digest_field="request_digest")
    profile_path = tmp_path / "droid_profile.json"
    profile_path.write_text(json.dumps(profile))
    runtime = tmp_path / "runtime_source.json"
    runtime.write_text(json.dumps({"schema_version": "native_task_runtime_source_packet.v1",
                                   "packet_sha256": "sha256:" + "4" * 64, "packet_size_bytes": 1024,
                                   "redistribution_permitted": True}))
    return {"revision": configured, "documents": documents, "configured_scene_bundle_path": bundle,
            "droid_profile_path": profile_path, "droid_profile_digest": _sha(profile_path.read_bytes()),
            "runtime_source_receipt_path": runtime, "profile": profile, "store": store,
            "revision_reference": {"uri": "s3://scenes/revision.json", "digest": _sha(revision_bytes),
                                   "size_bytes": len(revision_bytes)}}


def _compile(tmp_path: Path, scene: dict[str, Any]) -> tuple[dict, dict]:
    inputs = {key: value for key, value in scene.items() if key not in ("profile", "store", "revision_reference")}
    compiled = materialize_team_scene(**inputs, output_root=tmp_path / "team-scene")
    packet = Path(compiled["path"]).parent / "native-task-packet"
    return compiled, json.loads((packet / "native_task_arena_packet_request.v1.json").read_text())


def test_a_prepared_scene_compiles_its_own_half_with_no_robot_placement_or_prior_run(tmp_path) -> None:
    scene = _scene(tmp_path)
    compiled, request = _compile(tmp_path, scene)
    # The station: a mount level with the support, on clear floor beside the table, facing the object.
    station = compiled["robot_station"]
    x, y, z = request["robot_base_pose_world"]["position_world_m"]
    start = request["task_spec"]["start_pose_world"]
    assert z == pytest.approx(SUPPORT_TOP_Z) and station["floor_z_m"] == pytest.approx(0.0, abs=1e-6)
    assert x + STATION_FOOTPRINT_HALF_EXTENT_M[0] <= 2.5 + 1e-9  # clear of the table's edge
    assert math.hypot(x - start[0], y - start[1]) < 0.8  # the closest clear spot, within an arm's reach
    assert station["yaw_rad"] == pytest.approx(math.atan2(start[1] - y, start[0] - x))
    # Blueprint's DROID Franka stands in it, from the operator's profile; a team's robot replaces it at run time.
    assert request["robot_joint_reset_positions_rad"] == scene["profile"]["robot_joint_reset_positions_rad"]
    roles = {camera["role"]: camera for camera in request["cameras"]}
    assert set(roles) == {"external", "wrist", "overview"}
    assert roles["wrist"] == scene["profile"]["cameras"][1] and roles["overview"]["policy_input"] is False
    # The scene's own bytes and task, from the published revision alone.
    assert [row["semantic_role"] for row in request["assets"]] == ["scene_appearance", "scene_collision", "task_object"]
    assert request["task_spec"]["task_kind"] == "rigid_pick_place"
    assert request["request_digest"] == canonical_digest(request, digest_field="request_digest")
    # The record the team registry reads: the packet is its own construction lineage, controls pending.
    half = read_team_scene_inputs(compiled["path"])
    assert half["construction_result"] == half["base_native_packet"]
    assert half["scene_revision_digest"] == scene["revision"]["revision_digest"]
    assert half["control_diagnostic"]["mode"] == "nonblocking_diagnostic_pending"
    assert half["task_success_contract"]["contract_digest"] == half["task_success_contract_digest"]
    # The team registry reads it as a whole scene half: the nominal cell resolves for the DROID interface.
    facts = scene_facts({"team_scene_inputs_path": compiled["path"], "open_loop_horizon": 8})
    assert facts["prompt"] and facts["max_policy_queries"] > 0 and facts["inputs"] == half
    # Changed packet bytes are never read as the sealed scene.
    receipt = Path(half["base_native_packet"]["path"])
    receipt.chmod(0o644)
    receipt.write_text("{}")
    with pytest.raises(TeamScenePacketError, match="base_native_packet_changed"):
        read_team_scene_inputs(compiled["path"])


def test_a_scene_is_refused_by_name_before_any_packet(tmp_path) -> None:
    scene = _scene(tmp_path)
    (tmp_path / "other.zip").write_bytes(b"not the published bundle")
    with pytest.raises(TeamScenePacketError, match="configured_bundle_binding_invalid"):
        _compile(tmp_path, {**scene, "configured_scene_bundle_path": tmp_path / "other.zip"})
    with pytest.raises(TeamScenePacketError, match="droid_profile_binding_invalid"):
        _compile(tmp_path, {**scene, "droid_profile_digest": "sha256:" + "0" * 64})
    destination = {**scene["revision"], "task_template": {**scene["revision"]["task_template"],
                                                          "destination": {"geometry": {}}}}
    with pytest.raises(TeamScenePacketError, match="destination_body_unsupported"):
        _compile(tmp_path, {**scene, "revision": destination})
    assert not (tmp_path / "team-scene").exists()


def test_a_completed_source_launch_is_compiled_and_recorded_where_the_team_registry_reads_it(
        tmp_path, monkeypatch) -> None:
    from blueprint_pipeline import task_evaluation_configured_controls_progression_worker as source
    from blueprint_pipeline.team_candidate_quick10_registry import (
        REGISTRY_ENV,
        REGISTRY_SCHEMA,
        offer_eligibility,
        profile_for_capture,
    )
    from blueprint_pipeline.team_scene_packet import prepare_team_scene

    scene = _scene(tmp_path)
    run, capture, operator = tmp_path / "launch", tmp_path / "capture", tmp_path / "operator"
    for directory in (run, capture, operator):
        directory.mkdir()
    (capture / "capture_descriptor.json").write_text(json.dumps({"scene_id": "site-1"}))
    (operator / "salt.key").write_text("k" * 48)
    (operator / "salt.key").chmod(0o600)
    # The operator decides only who may fund a run, at what rate, and the salt; the scene writes its own row.
    registry = {"schema_version": REGISTRY_SCHEMA, "profiles": [], "derived_profiles": {
        "root": str(operator / "derived"), "scored_matrix_salt_key_path": str(operator / "salt.key"),
        "defaults": {"allowed_founder_approvers": ["founder@blueprint"], "max_hourly_rate_usd": 0.8,
                     "open_loop_horizon": 8}}, "registry_digest": ""}
    registry["registry_digest"] = canonical_digest(registry, digest_field="registry_digest")
    (operator / "registry.json").write_text(json.dumps(registry))
    (operator / "registry.json").chmod(0o644)
    monkeypatch.setenv(REGISTRY_ENV, str(operator / "registry.json"))
    monkeypatch.setattr(source, "_validate_source", lambda root: ({
        "configured_scene_revision_reference": scene["revision_reference"],
        "configured_scene_revision_digest": scene["revision"]["revision_digest"]}, {}, {}))

    prepared = prepare_team_scene(
        run_root=run, capture_root=capture, scenario_id="nominal", output_root=tmp_path / "prepared",
        fetcher=lambda reference: scene["store"][reference["uri"]],
        **{key: scene[key] for key in ("configured_scene_bundle_path", "droid_profile_path", "droid_profile_digest",
                                       "runtime_source_receipt_path")})
    assert prepared["registry"]["status"] == "written"
    profile = profile_for_capture(capture)
    assert profile["team_scene_inputs_path"] == prepared["path"] and profile["robot_free_probe_root"] == str(run)
    # Listed for every team: its probe is still undecided, so the first cell of the first shard records it.
    facts, blockers = offer_eligibility(profile)
    assert blockers == [] and facts["probe"] is None and facts["inputs"] == read_team_scene_inputs(prepared["path"])
