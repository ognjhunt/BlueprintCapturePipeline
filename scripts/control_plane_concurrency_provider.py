"""Explicit external scene-provider fixtures for developmental Plan 13c runs.

ADP-009D/day 28. Native import and visual review are fictional provider outputs.
The real publication, byte readback and queue finalization validators consume
them; these fixtures cannot establish qualification, physical or paid proof.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from scripts.control_plane_concurrency_fixture import FilesystemObjectStore


def file_record(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return {"digest": "sha256:" + digest.hexdigest(), "size_bytes": path.stat().st_size}


def fixture_publisher(object_root: Path):
    store = FilesystemObjectStore(object_root)

    def publish(*, path: Path, object_name: str) -> dict:
        key = "task-evaluation/fixture-publications/" + object_name
        expected = file_record(path)
        try:
            store.head_object(Bucket="blueprint", Key=key)
        except KeyError:
            with path.open("rb") as source:
                store.put_object(Bucket="blueprint", Key=key, Body=source,
                                 ContentLength=expected["size_bytes"])
        digest, size = hashlib.sha256(), 0
        with store.get_object(Bucket="blueprint", Key=key)["Body"] as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        observed = {"digest": "sha256:" + digest.hexdigest(), "size_bytes": size}
        if observed != expected:
            raise ValueError("harness_fixture_publication_readback_mismatch")
        return {"uri": "s3://blueprint/" + key, **expected,
                "full_byte_service_account_readback_passed": True,
                "readback_digest": observed["digest"], "readback_size_bytes": observed["size_bytes"]}

    return publish


def _json(path: Path, value: dict, field: str | None = None) -> None:
    if field:
        value[field] = canonical_digest(value, digest_field=field)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write("\n")


def fixture_scene_artifacts(*, envelope: dict, output_root: Path) -> list[dict]:
    """Derive fixture assets from this envelope's immutable source geometry."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    output_root.mkdir(parents=True, mode=0o700)
    refs = {row["contract_path"]: row for row in envelope["materialized_references"]}
    selection_path = Path(refs["task.subject.source_object"]["materialized_path"])
    selection = json.loads(selection_path.read_text())
    dimensions = [high - low for low, high in zip(selection["aabb_min_xyz_m"], selection["aabb_max_xyz_m"], strict=True)]
    appearance, collision, replacement = [output_root / (name + ".usda")
                                           for name in ("appearance", "collision", "replacement")]
    # The provider fixture changes only its returned assets. The production
    # input, declared task and preceding request digests remain unchanged.
    original = Path(refs["scene.geometry.collision"]["materialized_path"])
    # Preparation CAS paths have no extension; USD chooses its reader from
    # an asset extension. Preserve the exact validated bytes in fixture work.
    source_usd = output_root / "source-collision.usda"
    shutil.copyfile(original, source_usd)
    stage = Usd.Stage.Open(str(source_usd))
    stage.GetRootLayer().Export(str(collision))
    stage = Usd.Stage.Open(str(collision))
    subject_path = "/Root" + selection["source_object_id"]
    if not stage.RemovePrim(subject_path):
        raise ValueError("harness_fixture_subject_prim_missing")
    stage.GetRootLayer().Save()
    stage.GetRootLayer().Export(str(appearance))
    stage = None
    body = Usd.Stage.CreateNew(str(replacement))
    UsdGeom.SetStageUpAxis(body, "Z")
    UsdGeom.SetStageMetersPerUnit(body, 1.0)
    root = UsdGeom.Xform.Define(body, "/Asset")
    body.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass.CreateMassAttr(0.3)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, dimensions[2] / 2))
    cube = UsdGeom.Cube.Define(body, "/Asset/Collision")
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, dimensions[2] / 2))
    cube.AddScaleOp().Set(Gf.Vec3d(*dimensions))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    body.GetRootLayer().Save()
    body = None
    identity = envelope["request"]["task"]["subject"]["identity"]
    static = {"schema_version": "task_evaluation_rigid_replacement_static_qualification.v1",
              "status": "authored_structure_statically_qualified", "replacement_identity": identity,
              "fixture_provider": True, "claim_ceiling": "development_only",
              "observed_structure": {"center_of_mass_m": [0.0, 0.0, dimensions[2] / 2],
                  "collision_bounds_body_frame_m": {"minimum": [-dimensions[0] / 2, -dimensions[1] / 2, 0.0],
                                                     "maximum": [dimensions[0] / 2, dimensions[1] / 2, dimensions[2]]},
                  "rigid_body_paths": ["/Asset"]}}
    native = {"schema_version": "task_evaluation_replacement_native_import_result.v1", "status": "qualified",
              "replacement_identity": identity, "native_simulator_import_qualified": True,
              "fixture_provider": True, "claim_ceiling": "development_only", "blockers": []}
    _json(output_root / "static-receipt.json", static, "result_digest")
    _json(output_root / "native-receipt.json", native, "result_digest")
    _json(output_root / "appearance-receipt.json", {"fixture_provider": True, "status": "completed"}, "result_digest")
    _json(output_root / "collision-receipt.json", {"fixture_provider": True, "status": "completed",
            "removed_source_prim_path": subject_path, "source_geometry": file_record(original)}, "result_digest")
    _json(output_root / "assembly-receipt.json", {"fixture_provider": True,
            "request_digest": envelope["request_digest"] if "request_digest" in envelope else canonical_digest(envelope["request"]),
            "source_envelope_digest": envelope["envelope_digest"]}, "result_digest")
    assets = [{"role": role, "relative_path": path.name, **file_record(path)}
              for role, path in (("appearance", appearance), ("collision", collision), ("replacement", replacement))]
    _json(output_root / "bundle-candidate.json", {
        "schema_version": "task_evaluation_configured_scene_bundle_candidate.v1",
        "status": "assembled_pending_control_plane_publication", "robot_neutral": True,
        "robot_specific_base_registration_included": False, "assets": assets}, "manifest_digest")
    frames = envelope["render_inputs_result"]["derived_frames"]
    thumbnail = output_root / "thumbnail.png"
    shutil.copyfile(Path(frames[3]["path"]), thumbnail)
    _json(output_root / "review.json", {
        "schema_version": "task_evaluation_artifixer_ai_visual_review.v1", "status": "accepted",
        "fixture_provider": True, "claim_ceiling": "development_only", "review_frame_count": len(frames),
        "task_thumbnail_is_exact_review_frame": True,
        "task_thumbnail_selection": {"camera_id": "fixture-03", "frame_sha256": file_record(thumbnail)["digest"],
                                     "rationale": "Deterministic external reviewer fixture."},
        "reviewer": {"kind": "ai", "identity": "artifixer-independent-vision-reviewer-v1",
                     "runtime": "openai_agents_sdk", "model": "gpt-6.1-sol"}}, "receipt_digest")
    roles = {
        "configured_appearance_without_source_object": appearance,
        "appearance_removal_receipt": output_root / "appearance-receipt.json",
        "configured_collision_without_source_object": collision,
        "collision_excision_receipt": output_root / "collision-receipt.json",
        "statically_qualified_replacement_asset": replacement,
        "static_qualification_receipt": output_root / "static-receipt.json",
        "native_qualified_replacement_asset": replacement,
        "native_import_qualification_receipt": output_root / "native-receipt.json",
        "configured_scene_bundle_candidate_manifest": output_root / "bundle-candidate.json",
        "scene_assembly_receipt": output_root / "assembly-receipt.json",
        "appearance_visual_review_receipt": output_root / "review.json",
        "configured_task_thumbnail": thumbnail,
    }
    return [{"fixture_provider": True, "fixture_input_digest": envelope["envelope_digest"],
             "actual_provider_calls": 0,
             "output_artifacts": [{"role": role, "path": str(path), **file_record(path)} for role, path in roles.items()]}]
