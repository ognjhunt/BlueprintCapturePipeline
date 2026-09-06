"""Supplied mesh bytes are usable input, never fabricated camera or physical proof."""

import hashlib
import json

import pytest

from blueprint_pipeline.provided_scene_mesh import (
    ProvidedSceneMeshImportAdapter, inspect_mesh, method_profile,
)
from blueprint_pipeline.reconstruction_capability import ReconstructionContractError
from blueprint_pipeline.capture_upload_intake import process_capture_upload_submission
from tests.test_capture_upload_intake import _submission, _opener


USD = b'''#usda 1.0
(metersPerUnit = 1
 upAxis = "Z")
def Mesh "Book" {
    point3f[] points = [(0,0,0), (1,0,0), (0,1,1)]
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0,1,2]
}
'''


def test_mesh_upload_requires_no_invented_source_capture_binding(tmp_path):
    submission = _submission(USD)
    submission["request"].update(capture_authority_profile="provided_scene_mesh", source_type="provided_scene_mesh",
        original_file={"original_filename": "room.usda", "size_bytes": len(USD), "media_type": "application/octet-stream"},
        available_sensor_streams=[{"stream_type": "provided_geometry", "status": "available"}],
        coordinate_frame_declaration={"meters_per_unit": 1, "up_axis": "Z"},
        permitted_evidence_uses=["appearance_review"])
    receipt = process_capture_upload_submission(submission, store_root=tmp_path,
        allowed_hosts=["download.example.test"], transfer_opener=_opener(USD, {}),
        malware_scanner=lambda path: {"status": "passed", "scanner": "fixture-scanner"})
    assert receipt["admission_status"] == "accepted"
    envelope = json.loads((tmp_path / receipt["artifact_reference"]["uri"] / "capture_intake_envelope.json").read_text())
    assert "source_capture_binding" not in envelope
    assert envelope["capture_authority_profile"] == "provided_scene_mesh"
    assert receipt["claim_ceiling"]["metric_geometry"] is False
    assert receipt["claim_ceiling"]["captured_observation_review"] is False
    from blueprint_pipeline.reconstruction_control_plane import (
        prepare_reconstruction_plan, authorize_reconstruction_plan, execute_reconstruction_plan,
    )
    from blueprint_pipeline.provided_scene_mesh import ADAPTER
    state = tmp_path / "reconstruction"
    planned = prepare_reconstruction_plan(state_root=state, capture_store_root=tmp_path,
        capture_session_id=submission["capture_session_id"], intake_id=submission["request"]["intake_id"],
        requested_claim_types=["appearance_review"], idempotency_key="mesh-plan")
    assert planned["authorization_candidates"][0]["adapter_reference"] == ADAPTER
    authorize_reconstruction_plan(state_root=state, plan_id=planned["plan_id"],
        reconstruction_plan_digest=planned["reconstruction_plan"]["reconstruction_plan_digest"],
        authorized_adapter_references=[ADAPTER], actor={"role": "customer", "identity": "fixture-owner"},
        idempotency_key="mesh-authorize")
    executed = execute_reconstruction_plan(state_root=state, capture_store_root=tmp_path, plan_id=planned["plan_id"])
    assert executed["state"] == "partial", executed.get("errors")
    assert executed["errors"] == []
    assert executed["results"][0]["claim_ceiling"]["provided_mesh_imported"] is True
    assert executed["missing_representations"]  # import is not renderer qualification
    assert executed["results"][0]["claim_ceiling"]["physical_task_success"] is False


def test_real_usd_mesh_import_retains_declared_units_and_exact_objects(tmp_path):
    path = tmp_path / "room.usda"
    path.write_bytes(USD)
    digest = "sha256:" + hashlib.sha256(USD).hexdigest()
    result = ProvidedSceneMeshImportAdapter().execute(intake_id="mesh-1", capture_digest=digest,
        capture_root=tmp_path, asset_relative_path="room.usda", original_filename="room.usda",
        output_root=tmp_path / "out", rights_and_retention={"local_processing_allowed": True},
        coordinate_frame_declaration={"meters_per_unit": 1, "up_axis": "Z"})
    assert result["validation_metrics"]["objects"][0]["source_object_id"] == "/Book"
    assert result["validation_metrics"]["objects"][0]["face_count"] == 1
    assert result["claim_ceiling"]["metric_geometry"] is False
    assert result["claim_ceiling"]["raw_capture_authority"] is False
    assert result["method_profile_digest"] == method_profile(execution_authorized=True)["method_profile_digest"]


@pytest.mark.parametrize("declaration", [{}, {"meters_per_unit": float("nan"), "up_axis": "Z"},
    {"meters_per_unit": 100, "up_axis": "Z"}, {"meters_per_unit": 1, "up_axis": "Y"}])
def test_missing_or_conflicting_units_fail(tmp_path, declaration):
    path = tmp_path / "room.usda"
    path.write_bytes(USD)
    with pytest.raises(ReconstructionContractError):
        inspect_mesh(path, original_filename=path.name, coordinate_frame_declaration=declaration)


def test_external_usd_dependency_is_refused_before_composition(tmp_path):
    path = tmp_path / "room.usda"
    path.write_text('#usda 1.0\n(subLayers = [@/etc/passwd@])\n')
    with pytest.raises(ReconstructionContractError, match="external_dependency_forbidden"):
        inspect_mesh(path, original_filename=path.name,
                     coordinate_frame_declaration={"meters_per_unit": 1, "up_axis": "Z"})


def test_real_glb_mesh_import(tmp_path):
    import trimesh
    path = tmp_path / "room.glb"
    path.write_bytes(trimesh.Scene(trimesh.creation.box()).export(file_type="glb"))
    result = inspect_mesh(path, original_filename=path.name,
                          coordinate_frame_declaration={"meters_per_unit": 1, "up_axis": "Y"})
    assert len(result["objects"]) == 1
    assert result["objects"][0]["face_count"] == 12
    assert result["objects"][0]["world_aabb_max_m"] == [0.5, 0.5, 0.5]
