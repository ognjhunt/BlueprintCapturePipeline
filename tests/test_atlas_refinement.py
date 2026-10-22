"""A refinement cannot change source identity or introduce hidden review pixels."""
import json

from PIL import Image
import pytest

from blueprint_pipeline.atlas_refinement import load_refinement_feedback
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file


def test_refinement_binds_parent_source_selection_and_preview_pixels(tmp_path):
    parent = tmp_path / "parent"
    (parent / "pose").mkdir(parents=True)
    selection = parent / "pose/atlas_view_selection.json"
    selection.write_text(json.dumps({"frames": [{"frame_id": "visible"}], "review": {"same_scene_state": True}}))
    result = {"source_video_digest": "source", "task_context_digest": "context", "splat_available": True,
              "claim_ceiling": "development_only"}
    result["digest"] = canonical_digest(result, digest_field="digest")
    (parent / "result.json").write_text(json.dumps(result))
    image = tmp_path / "preview.png"
    Image.new("RGB", (20, 30), "white").save(image)
    value = {"schema_version": "atlas_reconstruction_feedback.v1", "parent_root": str(parent),
        "parent_result_digest": result["digest"], "parent_selection_sha256": _sha256_file(selection),
        "previews": [{"relative_path": image.name, "sha256": _sha256_file(image), "source_frame_id": "visible"}]}
    path = tmp_path / "feedback.json"
    def save():
        value["digest"] = canonical_digest(value, digest_field="digest")
        path.write_text(json.dumps(value))
    save()
    loaded = load_refinement_feedback(path, source_digest="source", context_digest="context")
    assert loaded["prior_selected_frame_ids"] == ["visible"]
    assert loaded["preview_authority"].startswith("derived_reconnaissance_only")
    with pytest.raises(ValueError, match="parent_changed"):
        load_refinement_feedback(path, source_digest="different-source", context_digest="context")
    value["previews"][0]["source_frame_id"] = "hidden"
    save()
    with pytest.raises(ValueError, match="preview_invalid"):
        load_refinement_feedback(path, source_digest="source", context_digest="context")
    value["previews"][0]["source_frame_id"] = "visible"
    save()
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="preview_invalid"):
        load_refinement_feedback(path, source_digest="source", context_digest="context")
