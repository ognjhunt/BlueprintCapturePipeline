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
from blueprint_pipeline.droid_exterior_cameras import ARENA_EXTERIOR_INTRINSICS
from blueprint_pipeline.native_task_arena_runtime import camera_runtime_parameters
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


def _scene(tmp_path: Path, *, strategy: str = "planar_push") -> dict[str, Any]:
    """The rigid fixture revision, published with a configured bundle whose collision is a real room."""

    sources = tmp_path / "sources"
    sources.mkdir()
    configured, references, _docs, _spec = rigid_case(sources, strategy=strategy)
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
    with pytest.raises(TeamScenePacketError, match="destination_placement_qualification_required"):
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
        scene_facts,
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
    monkeypatch.setenv("BLUEPRINT_RIGID_ROBOT_FREE_PROBE_QUEUE", str(tmp_path / "probe-queue"))
    monkeypatch.setattr(source, "_validate_source", lambda root: ({
        "configured_scene_revision_reference": scene["revision_reference"],
        "configured_scene_revision_digest": scene["revision"]["revision_digest"]}, {}, {}))

    prepared = prepare_team_scene(
        run_root=run, capture_root=capture, scenario_id="nominal", output_root=tmp_path / "prepared",
        fetcher=lambda reference: scene["store"][reference["uri"]],
        **{key: scene[key] for key in ("configured_scene_bundle_path", "droid_profile_path", "droid_profile_digest",
                                       "runtime_source_receipt_path")})
    assert prepared["registry"]["status"] == "written"
    assert prepared["robot_free_probe"]["status"] == "queued"
    assert prepared["robot_free_probe"]["queued"] is True
    assert prepared["robot_free_probe"]["lane_enabled"] is False
    pending = json.loads(Path(prepared["robot_free_probe"]["path"]).read_text())
    assert pending["capture_root"] == str(capture)
    assert pending["probe_request_digest"] == prepared["robot_free_probe"]["probe_request_digest"]
    profile = profile_for_capture(capture)
    assert profile["team_scene_inputs_path"] == prepared["path"] and profile["robot_free_probe_root"] == str(run)
    # The compiled scene is readable, but a paid team offer waits for its actual passed probe.
    facts, blockers = offer_eligibility(profile)
    assert facts is None and blockers == ["team_evaluation_robot_free_probe_receipt_missing"]
    assert scene_facts(profile)["inputs"] == read_team_scene_inputs(prepared["path"])


def _articulated_scene(tmp_path: Path, *, owner_confirmed: bool = True) -> dict[str, Any]:
    """The configured drawer-cabinet revision, standing on a room's floor, published with its bundle."""

    from pxr import UsdGeom

    from tests.test_task_evaluation_articulated_open_close_native_adapter import DEFINITION, _case

    case = _case(tmp_path / "case")
    configured, references = case["configured"], case["references"]
    if owner_confirmed:
        reference = references[DEFINITION]
        template = json.loads(Path(reference["materialized_path"]).read_text())
        template["owner_success_contract_authority"] = {
            "accepted_by": "owner-1", "authority_reference": "owner-chat:drawer-60-percent",
            "basis": "owner_confirmed_proposed_rule", "confirmation_status": "confirmed"}
        payload = (json.dumps(template, sort_keys=True) + "\n").encode()
        Path(reference["materialized_path"]).write_bytes(payload)
        reference.update(digest=_sha(payload), size_bytes=len(payload))
        configured["task_template"]["definition"] = {key: reference[key] for key in ("uri", "digest", "size_bytes")}
    room = _stage(tmp_path / "room.usda", "/Root")
    _box(room, "/Root/Floor", (-3.0, -3.0, 0.0), (3.0, 3.0, 0.0))
    _box(room, "/Root/BackWall", (-0.4, -3.0, 0.0), (-0.3, 3.0, 1.5))  # behind the cabinet's back
    UsdGeom.SetStageMetersPerUnit(room, 1.0)
    room.GetRootLayer().Save()
    asset = Path(case["asset"])
    payloads = {"appearance.usda": (tmp_path / "room.usda").read_bytes(),
                "collision.usda": (tmp_path / "room.usda").read_bytes(), "replacement" + asset.suffix: asset.read_bytes()}
    rows = [{"role": name.split(".")[0], "relative_path": name, "digest": _sha(data), "size_bytes": len(data)}
            for name, data in payloads.items()]
    manifest = {"schema_version": "task_evaluation_configured_scene_bundle_candidate.v1",
                "status": "assembled_pending_control_plane_publication", "robot_neutral": True,
                "robot_specific_base_registration_included": False, "assets": rows, "manifest_digest": ""}
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    bundle = tmp_path / "configured-scene.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
        archive.writestr("configured_scene_bundle_candidate.v1.json", json.dumps(manifest))
    configured["configured_scene_bundle"] = {"uri": "s3://scenes/configured-scene.zip",
                                             "digest": _sha(bundle.read_bytes()), "size_bytes": bundle.stat().st_size}
    configured["revision_digest"] = canonical_digest(configured, digest_field="revision_digest")
    store = {row["uri"]: Path(row["materialized_path"]).read_bytes() for row in references.values()}
    documents = rigid.fetch_revision_documents(revision=configured, output_root=tmp_path / "work",
                                               fetcher=lambda reference: store[reference["uri"]])
    (tmp_path / "rigid").mkdir()
    rigid_scene = _scene(tmp_path / "rigid")
    return {**{key: rigid_scene[key] for key in ("droid_profile_path", "droid_profile_digest",
                                                 "runtime_source_receipt_path", "profile")},
            "revision": configured, "documents": documents, "configured_scene_bundle_path": bundle}


def test_a_drawer_task_compiles_with_a_station_clear_of_the_drawers_travel(tmp_path) -> None:
    from blueprint_pipeline.adp_articulated_task_success_contract import validate_task_success_contract
    from blueprint_pipeline.scene_physics_preflight import require_scene_physics_admission

    scene = _articulated_scene(tmp_path)
    compiled, request = _compile(tmp_path, scene)
    packet = Path(compiled["path"]).parent / "native-task-packet"
    # The scene's own drawer task, as the website episode compiler builds it.
    assert request["task_spec"]["task_kind"] == "articulated_open_close"
    subject = request["assets"][2]
    assert subject["object_type"] == "ARTICULATION" and subject["reset_state"]["joint_positions"] == {
        "task_part_joint": 0.0}
    assert request["articulated_kinematic_path_receipt"]["schema_version"]
    # A floor cabinet puts the arm's mount on the floor, clear of the drawer's whole travel.
    station = compiled["robot_station"]
    assert station["position_world_m"][2] == pytest.approx(station["floor_z_m"], abs=1e-6)
    assert station["task_motion_clearance"]["reached_beyond_task_range"] is False
    report = require_scene_physics_admission(packet, sample_spacing_m=0.01)
    assert report["robot_clearance"]["blockers"] == [] and report["robot_clearance"]["advisories"] == []
    # The owner's confirmed criteria become the sealed contract, read by the team registry as a whole half.
    half = read_team_scene_inputs(compiled["path"])
    contract = validate_task_success_contract(half["task_success_contract"], task_kind="articulated_open_close")
    assert contract["provenance"]["author_source"] == "task_owner"
    facts = scene_facts({"team_scene_inputs_path": compiled["path"], "open_loop_horizon": 8})
    assert facts["prompt"] and facts["max_policy_queries"] > 0 and facts["scene"]["task_spec"]["task_kind"] == (
        "articulated_open_close")


def test_a_drawer_task_whose_owner_contract_is_unconfirmed_is_refused(tmp_path) -> None:
    scene = _articulated_scene(tmp_path, owner_confirmed=False)
    with pytest.raises(TeamScenePacketError, match="team_scene_owner_contract_invalid"):
        _compile(tmp_path, scene)


def test_right_exterior_keeps_authored_camera_and_distinguishes_estimated_from_measured_visibility(tmp_path, monkeypatch) -> None:
    from blueprint_pipeline import scene_camera_visibility

    scene = _scene(tmp_path)
    # Without measurable splats, keep the actual authored sensor and record the unknown visibility.
    compiled, request = _compile(tmp_path, scene)
    layout = json.loads(Path(compiled["path"]).read_text())["camera_layout"]
    assert layout["exterior_right"]["status"] == "estimated" and "measured_on_packet" not in layout
    assert layout["exterior_right"]["visibility_measured"] is False
    assert layout["exterior_right"]["simulation_estimate"] is True
    assert "estimated_from_packet" in layout
    assert layout["exterior_right"]["reason"].startswith("droid_exterior_right_unmeasured:")
    assert {camera["role"] for camera in request["cameras"]} == {"external", "wrist", "overview"}
    # The returned helper request is the original measurement packet; execution
    # is bound to the newly materialized packet with the independent third view.
    estimated = json.loads(Path(compiled["path"]).read_text())
    estimated_packet = Path(estimated["base_native_packet"]["path"]).parent
    estimated_plan = json.loads((estimated_packet / "native_task_arena_scene_plan.v1.json").read_text())
    assert {camera["role"] for camera in estimated_plan["cameras"]} == {
        "external", "wrist", "overview", "exterior_right"}
    # Where Arena's own mount sees the task clear of the scene, the packet is compiled once more with it.
    measured: list[Path] = []
    monkeypatch.setattr(scene_camera_visibility, "packet_view_evaluator",
                        lambda packet: lambda row: measured.append(Path(packet)) or [])
    inputs = {key: value for key, value in scene.items() if key not in ("profile", "store", "revision_reference")}
    three = materialize_team_scene(**inputs, output_root=tmp_path / "three-view")
    record = json.loads(Path(three["path"]).read_text())
    layout = record["camera_layout"]
    assert layout["exterior_right"]["status"] == "placed" and layout["exterior_right"]["placement"] == "arena_mount"
    assert measured == [Path(layout["measured_on_packet"]["path"]).parent]
    assert measured[0].name == "native-task-packet"
    packet = Path(record["base_native_packet"]["path"]).parent
    assert packet.name == "native-task-packet-three-view"
    plan = json.loads((packet / "native_task_arena_scene_plan.v1.json").read_text())
    cameras = {camera["role"]: camera for camera in plan["cameras"]}
    assert set(cameras) == {"external", "wrist", "overview", "exterior_right"}
    assert cameras["exterior_right"]["policy_input"] is True and cameras["overview"]["policy_input"] is False
    # Arena's own exterior optics, the size the left exterior keeps in a DROID canary, and every
    # compiled camera converts to a runtime camera (a refused one would block the two-view pair too).
    assert cameras["exterior_right"]["intrinsics"] == ARENA_EXTERIOR_INTRINSICS
    for camera in plan["cameras"]:
        assert camera_runtime_parameters(camera, robot_id="franka_panda")["role"] == camera["role"]
    # The station and the scene's own task are the same either way; the team registry reads it the same.
    assert three["robot_station"] == compiled["robot_station"]
    assert read_team_scene_inputs(three["path"])["base_native_packet"] == record["base_native_packet"]
    # A poor view is an evaluation limitation, not a reason to remove a
    # supported policy's sensor before the task can be attempted.
    monkeypatch.setattr(scene_camera_visibility, "packet_view_evaluator",
                        lambda packet: lambda row: ["scene_camera_contact_occluded_by_scene:exterior_right:100%"])
    limited = materialize_team_scene(**inputs, output_root=tmp_path / "limited-view")
    limited_record = json.loads(Path(limited["path"]).read_text())
    limitation = limited_record["camera_layout"]["exterior_right"]
    assert limitation["status"] == "limited" and limitation["visibility_measured"] is True
    assert limitation["clear_view_found"] is False and limitation["camera_available"] is True
    assert limitation["reason"] == "droid_exterior_right_no_clear_view"
    limited_packet = Path(limited_record["base_native_packet"]["path"]).parent
    limited_plan = json.loads((limited_packet / "native_task_arena_scene_plan.v1.json").read_text())
    assert {camera["role"] for camera in limited_plan["cameras"]} == set(cameras)


@pytest.mark.parametrize("rise", [-SUPPORT_TOP_Z, 0.3])
def test_a_task_ending_on_another_surface_faces_the_whole_region_from_the_lower_surface(
        tmp_path, monkeypatch, rise) -> None:
    # Floor to desk (or table to shelf): the mount is level with the lower surface and the
    # station faces the region from the start to the target, not the start alone.
    from blueprint_pipeline import task_evaluation_rigid_relocation_native_adapter as adapter_module
    real = adapter_module.adapt_rigid_relocation_task_template

    def with_destination_surface(**kwargs):
        adapter = real(**kwargs)
        documents = adapter["source_documents"]["documents"]
        documents["definition"]["destination_support_plane"] = {**documents["support_plane"],
                                                                "top_z_m": SUPPORT_TOP_Z + rise}
        return adapter
    monkeypatch.setattr(adapter_module, "adapt_rigid_relocation_task_template", with_destination_surface)
    compiled, request = _compile(tmp_path, _scene(tmp_path))
    station = compiled["robot_station"]
    x, y, z = request["robot_base_pose_world"]["position_world_m"]
    assert z == pytest.approx(min(SUPPORT_TOP_Z, SUPPORT_TOP_Z + rise), abs=0.03)
    start, target = request["task_spec"]["start_pose_world"], request["task_spec"]["target_position_world_m"]
    middle = [(start[i] + target[i]) / 2.0 for i in (0, 1)]
    assert station["yaw_rad"] == pytest.approx(math.atan2(middle[1] - y, middle[0] - x), abs=1e-6)


def test_a_pick_and_place_packet_stands_where_the_bundle_gate_finds_a_carry(tmp_path) -> None:
    # The packet's station search and the bundle's own gate measure the same carry: they agree.
    from blueprint_pipeline.robot_arm_path_clearance import packet_arm_path_clearance

    compiled, request = _compile(tmp_path, _scene(tmp_path, strategy="pick_and_place"))
    record = json.loads(Path(compiled["path"]).read_text())
    search = record["robot_station_search"]
    assert search["rule"] == "first_ranked_stance_that_carries" and search["results"]
    assert record["robot_station"]["stance_digest"] == search["results"][search["chosen_rank"]]["stance_digest"]
    gate = packet_arm_path_clearance(Path(compiled["path"]).parent / "native-task-packet")
    assert gate["status"] == ("clear" if search["cleared"] else "blocked")
    assert request["robot_base_pose_world"]["position_world_m"] == record["robot_station"]["position_world_m"]


def bodies(tmp_path: Path) -> dict[str, dict[str, Any]]:
    """Three registered team bodies: a long and a short fixed arm, and a wheeled humanoid free on the floor."""

    from blueprint_pipeline.team_embodiment import validate_embodiment_manifest
    from tests.test_team_embodiment import manifest, wheeled_manifest, write_robot_usd

    arm = write_robot_usd(tmp_path / "robot.usda")
    wheeled = write_robot_usd(tmp_path / "wheeled.usda", anchored=False, wheels=True)
    return {"long": validate_embodiment_manifest(manifest(arm, control_hz=60)),
            "short": validate_embodiment_manifest(manifest(arm, control_hz=60, embodiment_id="short-arm", reach_m=0.2)),
            "wheeled": validate_embodiment_manifest({**wheeled_manifest(wheeled), "embodiment_id": "acme-rover"})}


def test_each_registered_body_gets_its_own_station_from_its_own_manifest(tmp_path, monkeypatch) -> None:
    from blueprint_pipeline import robot_arm_path_clearance as clearance

    # Say the Franka carries only from the third closest stance; no other body inherits that choice.
    def franka_carries_from_third(stances, **_):
        attempts = [(row, {"status": "clear" if index == 2 else "blocked", "blockers": []})
                    for index, row in enumerate(stances[:3])]
        return stances[2], attempts[2][1], attempts
    monkeypatch.setattr(clearance, "first_carrying_stance", franka_carries_from_third)
    scene = _scene(tmp_path, strategy="pick_and_place")
    inputs = {key: value for key, value in scene.items() if key not in ("profile", "store", "revision_reference")}
    franka_only = materialize_team_scene(**inputs, output_root=tmp_path / "franka-only")
    body = bodies(tmp_path)
    compiled = materialize_team_scene(**inputs, output_root=tmp_path / "with-bodies",
                                      embodiment_manifests=list(body.values()))
    alone, record = (json.loads(Path(row["path"]).read_text()) for row in (franka_only, compiled))
    # The Franka's station and search are what a Franka-only compile makes; only the bodies' stations are added.
    assert "embodiment_stations" not in alone and set(record) - set(alone) == {"embodiment_stations"}
    assert record["robot_station"] == alone["robot_station"] == franka_only["robot_station"]
    assert record["robot_station_search"] == alone["robot_station_search"]
    assert alone["robot_station_search"]["chosen_rank"] == 2
    assert set(record["embodiment_stations"]) == {row["embodiment_digest"] for row in body.values()}
    stations = {row["embodiment_id"]: row for row in record["embodiment_stations"].values()}
    task = json.loads((Path(compiled["path"]).parent / "native-task-packet"
                       / "native_task_arena_packet_request.v1.json").read_text())["task_spec"]
    points = [task["start_pose_world"][:3], task["target_position_world_m"]]
    # A long fixed arm takes the closest clear stance its own reach covers: not the Franka's third.
    long = stations["acme-arm-v2"]
    assert long["status"] == "chosen" and long["blockers"] == [] and long["base_mode"] == "fixed"
    assert long["station"]["position_world_m"] != record["robot_station"]["position_world_m"]
    assert long["station"]["position_world_m"][2] == pytest.approx(SUPPORT_TOP_Z)
    assert long["reach"]["reach_m"] == 3.0 and long["reach"]["task_points_world_m"] == points
    assert long["reach"]["farthest_task_point_m"] == pytest.approx(
        max(math.dist(long["station"]["position_world_m"], point) for point in points))
    # A 20 cm arm reaches the task from no clear stance on this floor: measured, and named.
    short = stations["short-arm"]
    assert short["status"] == "out_of_reach" and short["blockers"] == ["team_embodiment_station_out_of_reach"]
    assert short["reach"]["clear_stances_measured"] > 1 and short["reach"]["farthest_task_point_m"] > 0.2
    # A free base stands on its own footprint at its own root height, on the floor; its reach is undeclared.
    rover = stations["acme-rover"]
    assert rover["status"] == "chosen" and rover["base_mode"] == "free"
    assert rover["station"]["floor_z_m"] == pytest.approx(0.0, abs=1e-6)
    assert rover["station"]["position_world_m"][2] == pytest.approx(0.9)
    assert rover["station"]["footprint_half_extent_m"] == [0.3, 0.3]
    assert rover["reach"] is None and rover["unassessed"] == ["embodiment_free_base_reach_not_modeled"]
    # Each declared tool is screened on its own opening; no grip force is declared, so none is assumed.
    for row in stations.values():
        [grip] = row["grippers"]
        assert grip["gripper"]["full_stroke_m"] == 0.08 and grip["gripper"]["grip_force_n"] is None
        assert grip["sensitivity"] in {"blocked_by_estimate", "hold_unassessed_grip_force_not_declared"}
    assert read_team_scene_inputs(compiled["path"])["construction_result"] == record["base_native_packet"]


def test_a_bodys_station_keeps_off_a_moving_part_and_leaves_its_unmodelled_body_unassessed(tmp_path) -> None:
    scene = _articulated_scene(tmp_path)
    inputs = {key: value for key, value in scene.items() if key != "profile"}
    compiled = materialize_team_scene(**inputs, output_root=tmp_path / "team-scene",
                                      embodiment_manifests=[bodies(tmp_path)["long"]])
    [row] = json.loads(Path(compiled["path"]).read_text())["embodiment_stations"].values()
    # The drawer's travel is kept off the mount; the arm on it has no volume model here, so it is not measured.
    assert row["status"] == "chosen" and row["station"]["position_world_m"][2] == pytest.approx(
        row["station"]["floor_z_m"], abs=1e-6)
    assert row["station"]["task_motion_clearance"]["robot_id"] == "team_embodiment"
    assert row["station"]["task_motion_clearance"]["reached_beyond_task_range"] is False
    assert row["unassessed"] == ["embodiment_body_task_motion_clearance_unmodeled"]
    [grip] = row["grippers"]
    assert grip["joint_type"] == "prismatic" and grip["gripper"]["full_stroke_m"] == 0.08

