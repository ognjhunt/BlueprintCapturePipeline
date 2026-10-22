from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline import website_scene_geometry as geometry


def test_phone_rotation_preserves_a_pixel_mapping(tmp_path):
    source = tmp_path / "original.png"
    Image.new("RGB", (28, 14), "red").save(source)
    before = source.read_bytes()
    result = geometry._prepare_image(source, tmp_path / "prepared.png", -90)
    transform = np.asarray(result["source_to_geometry_pixels"])
    sx, sy = result["width"] / 14, result["height"] / 28
    assert np.allclose(transform @ [0, 0, 1], [13 * sx + (sx - 1) / 2, (sy - 1) / 2, 1])
    assert result["height"] > result["width"]
    assert source.read_bytes() == before


@pytest.fixture
def geometry_case(tmp_path, monkeypatch):
    source = tmp_path / "raw.mov"
    source.write_bytes(b"original video")
    calls = []

    def decode(self, **kwargs):
        root = kwargs["output_root"]
        root.mkdir(parents=True)
        rows = []
        for i in range(2):
            image = root / f"{i}.png"
            Image.new("RGB", (28, 14), (i * 30, 0, 0)).save(image)
            rows.append({"candidate_relative_path": image.name, "frame_id": str(i),
                         "frame_digest": geometry._sha256_file(image), "t_video_sec": float(i)})
        (root / "candidate.json").write_text(json.dumps({"heldout_pixels_included": False, "frames": rows}))
        (root / "index.json").write_text(json.dumps({"stream_metadata": {"display_rotation_degrees": -90}}))
        return {"asset_references": {"candidate_dataset_manifest": {"relative_path": "candidate.json"},
                                     "decoded_observation_index": {"relative_path": "index.json"}}}

    monkeypatch.setattr(geometry.LocalDecodedObservationAdapter, "execute", decode)
    kwargs = dict(source_video=source, output_root=tmp_path / "out", capture_id="website-capture")
    return kwargs, calls


def test_source_views_wait_for_an_admitted_atlas_pose(geometry_case, monkeypatch):
    kwargs, calls = geometry_case
    result = geometry.run_website_scene_geometry(**kwargs)
    assert result["status"] == "pending"
    assert result["backend"] == "atlas"
    assert result["actual_provider_input_count"] == 0
    assert result["provider_mutation_performed"] is False
    assert result["heldout_pixels_included"] is False
    assert result["metric_measurement_proven"] is False
    assert result["blockers"] == ["website_atlas_admitted_pose_operation_required"]
    assert all("geometry_path" not in row for row in result["frames"])
    assert geometry.run_website_scene_geometry(**kwargs) == result
    assert not calls


def test_atlas_pose_result_cannot_switch_source_binding(geometry_case, monkeypatch):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    kwargs, calls = geometry_case
    pending = geometry.run_website_scene_geometry(**kwargs)
    pose = {"schema_version": "website_atlas_pose_result.v1", "endpoint": pending["endpoint"],
            "binding": {**pending["binding"], "source_video_digest": "sha256:" + "0" * 64},
            "operation": {"id": "synthetic-operation", "done": True}}
    pose["digest"] = canonical_digest(pose, digest_field="digest")
    (kwargs["output_root"] / "atlas_posed_rgbd_result.json").write_text(json.dumps(pose))
    with pytest.raises(ValueError, match="website_atlas_pose_result_binding_mismatch"):
        geometry.run_website_scene_geometry(**kwargs)
    assert not (kwargs["output_root"] / "atlas_generation_inputs.json").exists()
    assert not calls


@pytest.mark.parametrize("include_targets", [False, True])
def test_completed_atlas_pose_does_not_require_generation_or_imply_native_geometry(
        geometry_case, monkeypatch, include_targets):
    from copy import deepcopy
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    kwargs, calls = geometry_case
    pending = geometry.run_website_scene_geometry(**kwargs)
    # Synthetic returned bundle; this is not a real provider receipt or usage.
    camera = {"intrinsics": {"width": 640, "height": 480, "fx": 500, "fy": 500,
                             "cx": 320, "cy": 240},
              "extrinsics": {"coordinateSystem": "rdf", "position": [0, 0, 0],
                             "quaternion": [0, 0, 0, 1]}}
    pose = {"schema_version": "website_atlas_pose_result.v1", "endpoint": pending["endpoint"],
            "binding": pending["binding"],
            "operation": {"id": "synthetic-pose", "done": True,
                          "response": {"frames": [
                              {"imageAsset": {"assetId": f"synthetic-rgb-{i}"},
                               "depth": {"depthAsset": {"assetId": f"synthetic-depth-{i}"}},
                               "camera": deepcopy(camera)} for i in range(2)]}}}
    if include_targets:
        target = deepcopy(camera)
        target["intrinsics"].update(width=1280, height=720)
        pose["target_cameras"] = [target]
    pose["digest"] = canonical_digest(pose, digest_field="digest")
    (kwargs["output_root"] / "atlas_posed_rgbd_result.json").write_text(json.dumps(pose))
    result = geometry.run_website_scene_geometry(**kwargs)
    assert result["status"] == "pending" and result["posed_rgbd_available"] is True
    assert result["native_geometry_available"] is False
    assert result["actual_provider_input_count"] is None
    assert result["returned_pose_frame_count"] == 2
    assert ("generation_request" in result) is include_targets
    assert (kwargs["output_root"] / ("atlas_generation_inputs.json" if include_targets
                                     else "atlas_posed_rgbd_handoff.json")).is_file()
    assert geometry.run_website_scene_geometry(**kwargs) == result
    assert not calls


def test_cpu_preparation_exports_only_candidate_views(geometry_case):
    kwargs, calls = geometry_case
    inputs = geometry.prepare_website_geometry_inputs(**kwargs)
    root = kwargs["output_root"] / "worker_inputs"
    assert inputs["heldout_pixels_included"] is False
    assert not calls
    assert all(not Path(row["image_path"]).is_absolute() for row in inputs["frames"])
    assert len(list(root.rglob("*.png"))) == 4
    assert geometry.prepare_website_geometry_inputs(**kwargs) == inputs


def test_visual_source_frames_do_not_invent_geometry(geometry_case):
    kwargs, calls = geometry_case
    frames = geometry.prepare_website_source_frames(**kwargs)
    assert calls == []
    assert frames["geometry_available"] is False
    assert all(Path(row["source_image_path"]).is_file() and "geometry_path" not in row for row in frames["frames"])
    inputs = json.loads((kwargs["output_root"] / "worker_inputs/geometry_inputs.json").read_text())
    assert frames["geometry_input_digest"] == inputs["digest"]


@pytest.mark.parametrize("preparation", [False, True])
def test_a_website_task_context_is_checked_before_any_source_frame_is_decoded(geometry_case, monkeypatch, preparation):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    kwargs, calls = geometry_case
    task = {"capture_id": kwargs["capture_id"], "confirmed": True}
    if preparation:
        task.update(schema_version="website_site_task_context.v1", request_id="req1", scene_id="site-req1",
                    purpose="scene_preparation", description="Move the carton", confirmed=False, confirmed_at=None)
        task["context_digest"] = canonical_digest(task, digest_field="context_digest")
    assert geometry.run_website_scene_geometry(**kwargs, task_context=task)["status"] == "pending"
    if preparation:
        assert task["confirmed"] is False
        monkeypatch.setattr(geometry, "prepare_website_geometry_inputs", lambda **_kwargs: pytest.fail("must reject before decode"))
        for changed, reason in [({"request_id": None}, "identity_mismatch"),
                                ({"scene_id": " "}, "identity_mismatch"),
                                ({"purpose": "evaluation"}, "purpose_mismatch"),
                                ({"confirmed": "false"}, "confirmation_invalid"),
                                ({"confirmed_at": "2026-10-09T23:00:00Z"}, "confirmation_invalid")]:
            invalid = {**task, **changed}
            invalid["context_digest"] = canonical_digest(invalid, digest_field="context_digest")
            with pytest.raises(ValueError, match=reason):
                geometry.run_website_scene_geometry(**kwargs, task_context=invalid)
        with pytest.raises(ValueError, match="digest_mismatch"):
            geometry.run_website_scene_geometry(**kwargs, task_context={**task, "description": "Changed"})
    with pytest.raises(ValueError, match="task_capture_mismatch"):
        geometry.run_website_scene_geometry(**kwargs, task_context={**task, "capture_id": "other"})
