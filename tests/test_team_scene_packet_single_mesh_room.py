"""capture-447's shape through the team packet: a website room whose collider is one scanned mesh.

The website records bind the support to ``/Root`` (the room is one mesh), the tray stands on an oval dining
table, and the collider is ``/Root/mesh_0000_GLTF`` past the obstacle-box index's size cap. The packet's
Franka station stands on clear floor off the table, never inside it.
"""
from __future__ import annotations

import copy
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("pxr")

from blueprint_pipeline.decision_evidence_contracts import canonical_digest  # noqa: E402
from blueprint_pipeline.native_task_runtime_contract import DROID_FRANKA_RESET_JOINT_NAMES  # noqa: E402
from blueprint_pipeline.team_embodiment_stance import PEDESTAL_CLEARANCE_M  # noqa: E402
from tests.test_native_task_arena_packet import _camera  # noqa: E402
from tests.test_task_evaluation_owner_success_confirmation import OWNER, confirm, publish  # noqa: E402
from tests.test_task_evaluation_rigid_relocation_native_adapter import (  # noqa: E402
    DEFINITION,
    EXECUTION,
    SOURCE_OBJECT,
    STATIC,
    SUCCESS,
    SUPPORT,
    _case,
    _rewrite,
)
from tests.test_team_embodiment_stance_room_mesh import (  # noqa: E402
    ROOM,
    START,
    SUPPORT_TOP,
    TABLE_TOP,
    TRAY_HALF,
    _in_oval,
    footprint_samples,
    write_scanned_room,
)
from tests.test_team_scene_packet import _sha  # noqa: E402

SCALE = 1.103344777840341  # capture-447's estimated metres per unit, Y up
REGISTRATION = [[SCALE, 0.0, 0.0, 0.0], [0.0, 0.0, SCALE, 0.0], [0.0, -SCALE, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
SUPPORT_MIN, SUPPORT_MAX = [-1.00, -1.06, 0.77], [0.29, 0.51, SUPPORT_TOP]
DISC, RADIUS = [-0.2277, 0.0998, 0.8039], 0.39


def _appearance(path: Path) -> None:
    """A standard 3DGS splat of the room's surfaces, stored Y up at registration scale."""

    from blueprint_pipeline.atlas_depth_splat import PLY_PROPERTIES
    from blueprint_pipeline.gaussian_splat_decode import SplatData
    from blueprint_pipeline.particlefield_usd import write_particlefield_usd

    rng = np.random.default_rng(447)
    (x0, y0), (x1, y1) = (-2.5, -2.6), (1.8, 2.0)
    gx, gy = np.meshgrid(np.arange(x0, x1, 0.05), np.arange(y0, y1, 0.05))
    floor = np.c_[gx.ravel(), gy.ravel(), np.zeros(gx.size)]
    floor = floor[~_in_oval(floor)]
    tx, ty = np.meshgrid(np.arange(SUPPORT_MIN[0], SUPPORT_MAX[0], 0.02), np.arange(SUPPORT_MIN[1], SUPPORT_MAX[1], 0.02))
    top = np.c_[tx.ravel(), ty.ravel()]
    top = np.c_[top[_in_oval(top)], np.full(int(_in_oval(top).sum()), TABLE_TOP)]
    world = np.vstack([floor, top]) + rng.normal(0.0, 0.003, (len(floor) + len(top), 3))
    stored = world @ np.linalg.inv(np.asarray(REGISTRATION)[:3, :3])
    count = len(stored)
    splat = SplatData(count=count, xyz=stored.astype(np.float32), opacity=np.full(count, 3.0, np.float32),
                      f_dc=np.zeros((count, 3), np.float32),
                      scales=np.tile(np.log([0.012 / SCALE, 0.012 / SCALE, 0.002 / SCALE]), (count, 1)).astype(np.float32),
                      quats=np.tile([1.0, 0.0, 0.0, 0.0], (count, 1)).astype(np.float32), properties=PLY_PROPERTIES)
    assert write_particlefield_usd(splat, path, layer_transform_row_major=REGISTRATION)["status"] == "completed"


def _tray(path: Path) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Asset").GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(stage.GetDefaultPrim())
    UsdPhysics.MassAPI.Apply(stage.GetDefaultPrim()).CreateMassAttr(0.9)
    (hx, hy, hz) = TRAY_HALF
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/body")
    mesh.CreatePointsAttr([(-hx, -hy, 0.0), (hx, -hy, 0.0), (hx, hy, 0.0), (-hx, hy, 0.0),
                           (-hx, -hy, 2 * hz), (hx, -hy, 2 * hz), (hx, hy, 2 * hz), (-hx, hy, 2 * hz)])
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr([0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7])
    mesh.CreateExtentAttr([(-hx, -hy, 0.0), (hx, hy, 2 * hz)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    stage.GetRootLayer().Save()


def _bundle(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _appearance(root / "appearance.usdc")
    write_scanned_room(root / "collision.usda")
    _tray(root / "replacement.usda")
    payloads = {name: (root / name).read_bytes() for name in ("appearance.usdc", "collision.usda", "replacement.usda")}
    rows = [{"role": name.split(".")[0], "relative_path": name, "digest": _sha(data), "size_bytes": len(data)}
            for name, data in payloads.items()]
    manifest = {"schema_version": "task_evaluation_configured_scene_bundle_candidate.v1",
                "status": "assembled_pending_control_plane_publication", "robot_neutral": True,
                "robot_specific_base_registration_included": False, "assets": rows, "manifest_digest": ""}
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    path = root / "configured-scene.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
        archive.writestr("configured_scene_bundle_candidate.v1.json", json.dumps(manifest))
    return path


def _capture447(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """capture-447's published job shape: website records, the oval table, the support bound to ``/Root``."""

    from blueprint_pipeline import task_evaluation_scene_configuration_submission_records as records
    from blueprint_pipeline.task_evaluation_surface_target import derive_surface_target, surface_execution_limits

    root.mkdir(parents=True)
    _launch, configured, references, docs = _case(root)
    support = records.support_plane_input(scene_id="site-capture-447-synthetic", instance_id="website-support",
        semantic_label="wooden dining table", sage_prim_path="/Root", bounds_min=SUPPORT_MIN, bounds_max=SUPPORT_MAX)
    support.update(authority="registered_estimated_capture_and_reconstruction", physical_scale_measured=False,
                   source_face_indices=[0, 1, 2])
    lower = [START[i] - TRAY_HALF[i] for i in range(3)]
    upper = [START[i] + TRAY_HALF[i] for i in range(3)]
    source = {**docs[SOURCE_OBJECT], "center_xyz_m": START, "aabb_min_xyz_m": lower, "aabb_max_xyz_m": upper}
    target = derive_surface_target(destination={"kind": "green_region", "relation": "on", "visible_label": "green spot",
        "position_world_m": DISC, "orientation_xyzw": [0, 0, 0, 1], "radius_m": RADIUS}, support=support,
        source_min=lower, source_max=upper, support_instance_id="website-support", height_tolerance_m=0.072)
    success = surface_execution_limits(success={"control_frequency_hz": 15, "maximum_episode_seconds": 40,
        "minimum_lift_m": 0.05, "pregrasp_clearance_m": 0.1, "minimum_planar_displacement_m": 0.1,
        "maximum_final_planar_target_error_m": 0.05, "maximum_retries": 0, "maximum_regrasps": 0}, support=support)
    definition, criteria, execution = records.pick_and_place_task_records(
        task_identity=configured["task_template"]["identity"], object_identity=configured["replacement"]["identity"],
        start_center=START, target_center=[DISC[0], DISC[1], DISC[2] + TRAY_HALF[2]], source_min=lower,
        source_max=upper, grasp_axis=2, grasp_sign=1.0, jaw_axis=1, success=success, resolved_seed=1)
    definition.update(instruction="pick up the wooden tray from the dining table and place it at the other end of "
                      "the same table", instruction_subject_label="wooden tray", visible_target_label="green spot",
                      surface_target=target, dimension_authority="estimated", physical_world_truth_claimed=False,
                      placement_uncertainty_m=0.072, scale_authority="registration_estimate")
    definition["success"]["surface_target"] = criteria["surface_target"] = target
    definition["owner_success_contract_authority"] = {
        "confirmation_status": "proposal_only", "accepted_by": OWNER["user_id"],
        "authority_reference": "scene-intent:sha256:" + "f" * 64,
        "basis": "fixed_development_control_not_translated_owner_target", "scorer_translation_verified": False,
        "owner_success_criteria": {"basis": "owner_stated_target", "status": "not_supplied", "targets": None}}
    static = copy.deepcopy(docs[STATIC])
    static["observed_structure"]["center_of_mass_m"] = [0.0, 0.0, TRAY_HALF[2]]
    static["observed_structure"]["collision_bounds_body_frame_m"] = {
        "minimum": [-TRAY_HALF[0], -TRAY_HALF[1], 0.0], "maximum": [TRAY_HALF[0], TRAY_HALF[1], 2 * TRAY_HALF[2]]}
    static["result_digest"] = canonical_digest(static, digest_field="result_digest")
    for key, document in ((SUPPORT, support), (SOURCE_OBJECT, source), (DEFINITION, definition), (SUCCESS, criteria),
                          (EXECUTION, execution), (STATIC, static)):
        _rewrite(tmp_path=root, configured=configured, references=references, contract_path=key, document=document)
    configured["task_template"]["surface_target"] = target
    configured["revision_digest"] = canonical_digest(configured, digest_field="revision_digest")
    return configured, references


def _inputs(tmp_path: Path, monkeypatch: Any) -> dict[str, Any]:
    """capture-447's shape: ``materialize_team_scene``'s inputs, but for its output root."""

    from blueprint_pipeline import task_evaluation_scene_rigid_robot_free_readiness as rigid

    configured, references = _capture447(tmp_path / "case")
    bundle = _bundle(tmp_path / "bundle")
    configured["configured_scene_bundle"] = {"uri": "s3://scenes/configured-scene.zip",
                                             "digest": _sha(bundle.read_bytes()), "size_bytes": bundle.stat().st_size}
    configured["revision_digest"] = canonical_digest(configured, digest_field="revision_digest")
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
    job = publish(tmp_path, monkeypatch, configured=configured, references=references)
    confirm(job)
    documents = rigid.fetch_revision_documents(revision=job.revision, output_root=tmp_path / "work",
                                               fetcher=lambda reference: job.store[reference["uri"]])
    return dict(revision=job.revision, documents=documents, configured_scene_bundle_path=bundle,
                droid_profile_path=profile_path, droid_profile_digest=_sha(profile_path.read_bytes()),
                runtime_source_receipt_path=runtime,
                owner_success_confirmation_scope={"intent": job.intent, "directory": job.directory})


def _compile(tmp_path: Path, monkeypatch: Any, **extra: Any) -> dict[str, Any]:
    """capture-447's shape through ``materialize_team_scene``; the sealed record."""

    from blueprint_pipeline.team_scene_packet import materialize_team_scene

    compiled = materialize_team_scene(**_inputs(tmp_path, monkeypatch), output_root=tmp_path / "team-scene", **extra)
    return json.loads(Path(compiled["path"]).read_text())


def test_the_team_packet_stands_the_franka_off_a_single_mesh_rooms_table(tmp_path, monkeypatch) -> None:
    record = _compile(tmp_path, monkeypatch)
    station = record["robot_station"]
    # The mount is level with the support, on floor clear of the table by the pedestal's clearance.
    assert station["position_world_m"][2] == pytest.approx(SUPPORT_TOP)
    assert not _in_oval(footprint_samples(station, grow_m=PEDESTAL_CLEARANCE_M)).any(), station["position_world_m"]
    assert station["search"]["obstacle_clearance_m"] == PEDESTAL_CLEARANCE_M
    assert station["room_mesh_clearance"]["room_mesh_prims"] == ["/Root/mesh_0000_GLTF"]
    assert station["room_mesh_clearance"]["occupied_stances"] > 0
    x, y, _ = station["position_world_m"]
    assert ROOM[0][0] < x < ROOM[1][0] and ROOM[0][1] < y < ROOM[1][1]
    assert math.hypot(x - START[0], y - START[1]) <= 1.6
    # The carry search measured the room-clear stances (it measures the arm, not the footprint).
    from blueprint_pipeline.team_scene_packet import STATION_RULE

    assert record["robot_station_search"]["rule"] == STATION_RULE
    assert record["robot_station_search"]["stances_measured"] >= record["robot_station_search"]["stances_in_reach"]
    # The packet the team runs places the robot where the record says.
    request = json.loads((Path(record["base_native_packet"]["path"]).parent
                          / "native_task_arena_packet_request.v1.json").read_text())
    assert request["robot_base_pose_world"]["position_world_m"] == station["position_world_m"]


def test_the_team_packet_reads_a_text_room_collider_once_and_never_touches_it(tmp_path, monkeypatch) -> None:
    # capture-447's collider is a 292 MB USDA text file. OpenUSD parses it again at each open that finds no
    # stage of it alive, transiently at about twelve times its size, and the packet opened it 22 times: two
    # killed compiles. Now the compile reads it once, into a binary copy every reader opens instead.
    import hashlib

    from blueprint_pipeline import team_scene_packet
    from blueprint_pipeline import usd_binary_read_cache as cache
    from tests.test_team_embodiment_stance_room_mesh import text_parses
    from tests.test_team_scene_packet import bodies

    monkeypatch.setattr(cache, "LARGE_ARRAY_BYTES", 64)  # the fixture room's arrays are small
    copies: list[tuple[str, str, str | None]] = []
    make = team_scene_packet.binary_copy

    def recorded(source: Any, directory: Any) -> Any:
        copy = make(source, directory)
        copies.append((Path(source).name, copy.suffix, cache.copy_method(source)))
        return copy

    monkeypatch.setattr(team_scene_packet, "binary_copy", recorded)
    _, parsed = text_parses(monkeypatch, "collision.usda")
    copy_opens, _ = text_parses(monkeypatch, "collision.usdc")
    record = _compile(tmp_path, monkeypatch, embodiment_manifests=list(bodies(tmp_path).values()))
    assert copies == [("collision.usda", ".usdc", "numpy_large_arrays")]
    assert parsed == []  # no reader parsed the text, the packet's staged hard link included
    assert len(copy_opens) > 10  # every search, carried stance and packet step read the copy
    # The published collider is untouched, and everything recorded still names it.
    published = tmp_path / "team-scene" / "configured-scene" / "collision.usda"
    digest = "sha256:" + hashlib.sha256(published.read_bytes()).hexdigest()
    assert digest == "sha256:" + hashlib.sha256(
        (tmp_path / "bundle" / "collision.usda").read_bytes()).hexdigest()
    assert record["robot_station"]["scene_collision_sha256"] == digest
    assert {row["station"]["scene_collision_sha256"] for row in record["embodiment_stations"].values()
            if row["station"]} == {digest}
    receipt = json.loads((Path(record["base_native_packet"]["path"])).read_text())
    [binding] = [row for row in receipt["source_bindings"] if row["semantic_role"] == "scene_collision"]
    assert binding["staged_sha256"] == binding["source"]["sha256"] == digest
    assert not (tmp_path / "team-scene" / ".collider-read-cache").exists()


def test_the_binary_copy_changes_no_byte_the_compile_writes(tmp_path, monkeypatch) -> None:
    # The same compile into the same root, reading the text collider and then the compile's binary copy: every
    # file it leaves (the team record, both packets' requests, plans, receipts and staged assets) is identical.
    import hashlib
    import shutil

    from blueprint_pipeline import team_scene_packet
    from blueprint_pipeline import usd_binary_read_cache as cache
    from tests.test_team_scene_packet import bodies

    inputs = {**_inputs(tmp_path, monkeypatch), "embodiment_manifests": list(bodies(tmp_path).values())}
    root = tmp_path / "team-scene"

    def compiled() -> dict[str, str]:
        team_scene_packet.materialize_team_scene(**inputs, output_root=root)
        files = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(root.rglob("*")) if path.is_file()}
        shutil.rmtree(root)
        return files

    with monkeypatch.context() as text_only:
        text_only.setattr(team_scene_packet, "binary_copy", lambda source, directory: Path(source))
        from_text = compiled()
    monkeypatch.setattr(cache, "LARGE_ARRAY_BYTES", 64)
    from_copy = compiled()
    assert from_copy == from_text and "team_scene_inputs.v1.json" in from_copy
    assert not any(".collider-read-cache" in name for name in from_copy)
