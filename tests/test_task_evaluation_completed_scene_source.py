"""Finished asset intake binds real owned bytes and exact mesh object names."""
import json
import pytest

from blueprint_pipeline.task_evaluation_completed_scene_source import bind_completed_scene_source
from blueprint_pipeline.capture_upload_intake import process_capture_upload_submission
from tests.test_capture_upload_intake import _submission, _opener
from tests.test_task_evaluation_scene_intake import request, stage
from tests.test_provided_scene_splat import splat_bytes

MESH = b'''#usda 1.0
(metersPerUnit = 1; upAxis = "Z")
def Mesh "Book" {
 point3f[] points = [(0,0,0.75),(0.15,0,0.75),(0.15,0.2,0.75),(0,0.2,0.75),
                       (0,0,0.77),(0.15,0,0.77),(0.15,0.2,0.77),(0,0.2,0.77)]
 int[] faceVertexCounts = [4,4,4,4,4,4]
 int[] faceVertexIndices = [0,3,2,1,4,5,6,7,0,1,5,4,1,2,6,5,2,3,7,6,3,0,4,7]
}
def Mesh "Table" {
 point3f[] points = [(-1,-1,0.7),(2,-1,0.7),(2,2,0.7),(-1,2,0.7),
                       (-1,-1,0.75),(2,-1,0.75),(2,2,0.75),(-1,2,0.75)]
 int[] faceVertexCounts = [4,4,4,4,4,4]
 int[] faceVertexIndices = [0,3,2,1,4,5,6,7,0,1,5,4,1,2,6,5,2,3,7,6,3,0,4,7]
}
'''


def upload(root, data, identifier, profile, filename):
    submission = _submission(data)
    submission["capture_session_id"] = identifier
    submission["request"].update(intake_id=identifier, idempotency_key=identifier,
        capture_authority_profile=profile, source_type=profile,
        original_file={"original_filename": filename, "size_bytes": len(data), "media_type": "application/octet-stream"},
        available_sensor_streams=[{"stream_type": "provided_geometry", "status": "available"}],
        coordinate_frame_declaration={"meters_per_unit": 1, "up_axis": "Z"})
    receipt = process_capture_upload_submission(submission, store_root=root,
        allowed_hosts=["download.example.test"], transfer_opener=_opener(data, {}),
        malware_scanner=lambda _: {"status": "passed", "scanner": "fixture"})
    return submission, receipt


def test_paired_completed_assets_bind_without_capture_or_trainer(tmp_path):
    store = tmp_path / "store"
    submitted, appearance = upload(store, splat_bytes(), "appearance1", "provided_scene_splat", "room.ply")
    _, collision = upload(store, MESH, "collision1", "provided_scene_mesh", "room.usda")
    owner = request()
    owner["owner"] = {"user_id": submitted["customer_id"], "organization_id": submitted["organization_id"]}
    owner["consent"].update(accepted_by=owner["owner"]["user_id"], rights_reference=appearance["envelope_digest"])
    owner["source"] = {"kind": "gaussian_splat", "binding_id": "appearance1", "content_digest": appearance["capture_digest"],
        "collision_mesh": {"binding_id": "collision1", "content_digest": collision["capture_digest"],
            "rights_reference": collision["envelope_digest"], "frame_relation": "owner_declared_common_frame"}}
    owner["task"].update(subject={"description": "Book"}, support={"description": "Table"})
    accepted = stage(tmp_path / "intents", owner)
    intent = json.loads((tmp_path / "intents" / accepted["intent_id"] / "intent.json").read_text())
    config = {"capture_store_root": str(store), "factory_output_root": str(tmp_path / "out")}
    bound = bind_completed_scene_source(intent=intent, config=config)
    assert bound["status"] == "source_task_objects_bound", bound
    assert bound["object_bindings"]["subject"]["source_object_id"] == "/Book"
    assert bound["provider_reconstruction_started"] is False
    assert bound["physical_registration_proven"] is False
    assert bind_completed_scene_source(intent=intent, config=config) == bound
    intent["request"]["owner"]["user_id"] = "another-owner"
    with pytest.raises(ValueError, match="owner_or_digest_mismatch"):
        bind_completed_scene_source(intent=intent, config=config)
