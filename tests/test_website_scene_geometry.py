from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline import website_scene_geometry as geometry


def _prediction(width=28, height=14):
    pose = np.eye(4)[None]
    pose[0, 0, 3] = 2.5
    return {
        "depth_z": np.full((1, height, width, 1), 1.25),
        "conf": np.full((1, height, width), 2.0),
        "mask": np.ones((1, height, width, 1)),
        "intrinsics": np.array([[[20, 0, width / 2], [0, 20, height / 2], [0, 0, 1]]]),
        "camera_poses": pose,
    }


def test_native_estimates_keep_the_camera_transform_and_invalid_pixels(tmp_path):
    prediction = _prediction()
    prediction["depth_z"][0, 0, 0, 0] = np.nan
    result = geometry._write_prediction(prediction, {"width": 28, "height": 14}, tmp_path / "geometry.npz")
    with np.load(result["geometry_path"]) as data:
        assert not data["valid_mask"][0, 0]
        assert data["depth_m"][0, 0] == 0
        assert data["depth_m"][1, 1] == 1.25
        # Confidence is retained as the model's score, not mislabeled probability.
        assert data["confidence"][1, 1] == 2.0
    assert result["world_from_camera"][0][3] == 2.5
    assert result["camera_from_world"][0][3] == -2.5


@pytest.mark.parametrize("failure", ["shape", "pose", "intrinsics", "mask", "empty"])
def test_bad_predictions_are_rejected_without_synthetic_replacement(tmp_path, failure):
    prediction = _prediction()
    if failure == "shape":
        prediction["conf"] = np.zeros((1, 7, 14))
    elif failure == "pose":
        prediction["camera_poses"][0, 0, 0] = 2
    elif failure == "intrinsics":
        prediction["intrinsics"][0, 0, 0] = -1
    elif failure == "mask":
        prediction["mask"][0, 0, 0, 0] = np.nan
    else:
        prediction["mask"][:] = 0
    with pytest.raises(ValueError, match="mapanything_"):
        geometry._write_prediction(prediction, {"width": 28, "height": 14}, tmp_path / "geometry.npz")
    assert not (tmp_path / "geometry.npz").exists()


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


def test_missing_local_weights_do_not_trigger_a_download(tmp_path, monkeypatch):
    monkeypatch.setenv("BLUEPRINT_MAPANYTHING_MODEL_PATH", str(tmp_path / "missing"))
    monkeypatch.setattr(geometry, "prepare_website_geometry_inputs", lambda **_kwargs: {})
    monkeypatch.setattr(geometry, "_infer", lambda *_a, **_k: pytest.fail("must not infer"))
    with pytest.raises(ValueError, match="local_checkpoint_missing"):
        geometry.run_website_scene_geometry(source_video=tmp_path / "raw.mov", output_root=tmp_path / "out", capture_id="c")


@pytest.fixture
def geometry_case(tmp_path, monkeypatch):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"test weights")
    (weights / "config.json").write_text("{}")
    monkeypatch.setenv("BLUEPRINT_MAPANYTHING_MODEL_PATH", str(weights))
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

    def infer(paths, **kwargs):
        calls.append(paths)
        assert kwargs["model_path"] == weights
        with Image.open(paths[0]) as image:
            width, height = image.size
        return [_prediction(width, height) for _ in paths]

    monkeypatch.setattr(geometry.LocalDecodedObservationAdapter, "execute", decode)
    monkeypatch.setattr(geometry, "_infer", infer)
    kwargs = dict(source_video=source, output_root=tmp_path / "out", capture_id="website-capture")
    return kwargs, calls


def test_website_geometry_uses_original_views_and_reuses_bound_estimates(geometry_case):
    kwargs, calls = geometry_case
    result = geometry.run_website_scene_geometry(**kwargs)
    assert result["status"] == "estimated"
    assert result["scale_status"] == "model_estimated"
    assert result["metric_measurement_proven"] is False
    assert result["physical_evidence"] is False
    assert result["claim_ceiling"] == "development_only"
    assert result["binding"]["model_id"] == "facebook/map-anything-apache"
    assert len(result["frames"]) == 2
    assert geometry.run_website_scene_geometry(**kwargs) == result
    assert len(calls) == 1
    Path(result["frames"][0]["source_image_path"]).write_bytes(b"edited pixels")
    with pytest.raises(ValueError, match="artifact_changed"):
        geometry.run_website_scene_geometry(**kwargs)


def test_cpu_preparation_needs_no_checkpoint_and_exports_only_candidate_views(geometry_case, monkeypatch):
    kwargs, calls = geometry_case
    monkeypatch.setenv("BLUEPRINT_MAPANYTHING_MODEL_PATH", "/missing-model")
    inputs = geometry.prepare_website_geometry_inputs(**kwargs)
    root = kwargs["output_root"] / "worker_inputs"
    assert inputs["heldout_pixels_included"] is False
    assert not calls
    assert all(not Path(row["image_path"]).is_absolute() for row in inputs["frames"])
    assert len(list(root.rglob("*.png"))) == 4
    assert geometry.prepare_website_geometry_inputs(**kwargs) == inputs


def test_visual_source_frames_do_not_infer_or_invent_geometry(geometry_case, monkeypatch):
    kwargs, calls = geometry_case
    monkeypatch.setenv("BLUEPRINT_MAPANYTHING_MODEL_PATH", "/missing-model")
    frames = geometry.prepare_website_source_frames(**kwargs)
    assert calls == []
    assert frames["geometry_available"] is False
    assert all(Path(row["source_image_path"]).is_file() and "geometry_path" not in row for row in frames["frames"])
    inputs = json.loads((kwargs["output_root"] / "worker_inputs/geometry_inputs.json").read_text())
    assert frames["geometry_input_digest"] == inputs["digest"]


def test_worker_inputs_and_outputs_survive_host_path_changes(geometry_case, tmp_path, monkeypatch):
    import shutil

    kwargs, calls = geometry_case
    inputs = geometry.prepare_website_geometry_inputs(**kwargs)
    worker_inputs = tmp_path / "other-host" / "inputs"
    shutil.copytree(kwargs["output_root"] / "worker_inputs", worker_inputs)
    worker_output = tmp_path / "other-host" / "output"
    geometry.infer_website_geometry_inputs(input_manifest=worker_inputs / "geometry_inputs.json",
                                           output_root=worker_output, model_path=tmp_path / "weights")
    received = tmp_path / "received"
    shutil.move(worker_output, received)
    shutil.rmtree(worker_inputs)
    monkeypatch.setenv("BLUEPRINT_WEBSITE_GEOMETRY_RESULT", str(received / "source_geometry.json"))
    monkeypatch.setenv("BLUEPRINT_MAPANYTHING_MODEL_PATH", "/missing-model")
    result = geometry.run_website_scene_geometry(**kwargs)
    assert result["binding"]["input_digest"] == inputs["digest"]
    assert len(calls) == 1
    assert all(Path(row["geometry_path"]).is_relative_to(received) for row in result["frames"])
    assert result["metric_measurement_proven"] is False
    assert geometry.run_website_scene_geometry(**kwargs) == result


@pytest.mark.parametrize("mutation", ["source", "input", "frame", "escape", "measurement"])
def test_returned_worker_result_cannot_switch_capture_or_upgrade_authority(geometry_case, mutation):
    kwargs, _ = geometry_case
    geometry.run_website_scene_geometry(**kwargs)
    root = kwargs["output_root"]
    path = root / "source_geometry.json"
    document = json.loads(path.read_text())
    inputs = json.loads((root / "worker_inputs" / "geometry_inputs.json").read_text())
    if mutation == "source":
        document["binding"]["source_video_digest"] = "0" * 64
    elif mutation == "input":
        document["binding"]["input_digest"] = "0" * 64
    elif mutation == "frame":
        document["frames"][0]["timestamp_seconds"] += 1
    elif mutation == "escape":
        document["frames"][0]["image_path"] = "../other.png"
    else:
        document["metric_measurement_proven"] = True
    document["digest"] = geometry.canonical_digest(document, digest_field="digest")
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="website_(geometry_result|source_geometry_artifact_escape)"):
        geometry.load_website_geometry_result(manifest_path=path, inputs=inputs)


def test_inference_requests_the_upstream_validity_mask(tmp_path, monkeypatch):
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace
    image_path = tmp_path / "view.png"
    Image.new("RGB", (28, 14)).save(image_path)
    class Model:
        def to(self, device): return self
        def eval(self): return self
        def infer(self, views, **kwargs):
            # Upstream creates the mask key only inside its apply_mask branch.
            prediction = _prediction()
            if not kwargs["apply_mask"]:
                prediction.pop("mask")
            return [prediction]
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, "mapanything.models", SimpleNamespace(MapAnything=SimpleNamespace(from_pretrained=lambda *a, **kw: Model())))
    monkeypatch.setitem(sys.modules, "mapanything.utils.image", SimpleNamespace(load_images=lambda paths, **kw: [{} for _ in paths]))
    prediction = geometry._infer([str(image_path)], model_path=tmp_path)[0]
    result = geometry._write_prediction(prediction, {"width": 28, "height": 14}, tmp_path / "prediction.npz")
    assert result["valid_pixel_fraction"] == 1.0


def test_confirmed_website_task_uses_controller_allocator_instead_of_manual_result_override(geometry_case, monkeypatch):
    from blueprint_pipeline import paid_resource_allocator as allocator
    kwargs, calls = geometry_case
    invoked = []
    monkeypatch.setenv("BLUEPRINT_WEBSITE_GEOMETRY_RESULT", "/manual/result.json")
    def dispatch(**kwargs):
        invoked.append(kwargs)
        return {"status": "estimated", "controller": True}
    monkeypatch.setattr(allocator, "run_sponsored_website_geometry", dispatch)
    task = {"capture_id": kwargs["capture_id"], "confirmed": True}
    assert geometry.run_website_scene_geometry(**kwargs, task_context=task)["controller"] is True
    assert invoked[0]["task_context"] == task
    assert not calls
    with pytest.raises(ValueError, match="task_capture_mismatch"):
        geometry.run_website_scene_geometry(**kwargs, task_context={**task, "capture_id": "other"})


@pytest.fixture
def task_geometry_case(geometry_case, monkeypatch):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    kwargs, _ = geometry_case
    original = geometry.prepare_website_geometry_inputs(**kwargs)
    inputs = dict(original, frames=[{**row, "frame_id": f"decoded-{i * 2:09d}"}
                                    for i, row in enumerate(original["frames"])])
    inputs["digest"] = canonical_digest(inputs, digest_field="digest")
    registry = [{"source_frame_id": f"decoded-{i:09d}", "model_frame_index": i,
                 "decoded_pts_seconds": i / 30, "width": 14, "height": 28} for i in range(6)]
    mask = {"source_frame_id": "decoded-000000003", "width": 14, "height": 28,
            "runs": [{"start": 0, "length": 10}]}
    masks = {"source_video_digest": inputs["binding"]["source_video_digest"],
             "source_frame_registry": registry, "binding": {"geometry_input_digest": inputs["digest"]},
             "targets": [{"source_track": {"observations": [mask]}}]}
    masks["digest"] = canonical_digest(masks, digest_field="digest")
    split = {"capture_digest": inputs["binding"]["source_video_digest"],
             "assignments": [{"frame_id": "decoded-000000004", "split": "held_out"}]}
    split["split_digest"] = canonical_digest(split, digest_field="split_digest")
    (kwargs["output_root"] / "source/frozen_split_manifest.json").write_text(json.dumps(split))
    calls = []
    def decode(**kw):
        calls.append(kw["indexes"])
        kw["frame_root"].mkdir(parents=True, exist_ok=True)
        rows = []
        for i in kw["indexes"]:
            frame_id = registry[i]["source_frame_id"]
            path = kw["frame_root"] / f"{frame_id}.png"
            Image.new("RGB", (28, 14), "blue").save(path)
            rows.append({"frame_id": frame_id, "t_video_sec": i / 30, "digest": geometry._sha256_file(path)})
        return rows
    monkeypatch.setattr("blueprint_pipeline.local_reconstruction_adapters._extract_frames", decode)
    return {k: kwargs[k] for k in ("source_video", "output_root")} | {"inputs": inputs, "task_masks": masks}, calls


def test_geometry_adds_exact_observed_task_frame_and_reuses_bound_inputs(task_geometry_case):
    kwargs, calls = task_geometry_case
    result, root = geometry.prepare_task_geometry_inputs(**kwargs)
    assert calls == [[3]]
    assert len(result["frames"]) == 3
    assert "decoded-000000003" in {row["frame_id"] for row in result["frames"]}
    assert result["binding"]["tracking_input_digest"] == kwargs["inputs"]["digest"]
    assert result["binding"]["task_masks_digest"] == kwargs["task_masks"]["digest"]
    assert result["heldout_pixels_included"] is False
    assert geometry.prepare_task_geometry_inputs(**kwargs) == (result, root)
    assert calls == [[3]]
    assert len(kwargs["inputs"]["frames"]) == 2


def test_geometry_cannot_take_a_task_mask_from_heldout_pixels(task_geometry_case):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    kwargs, calls = task_geometry_case
    masks = kwargs["task_masks"]
    masks["targets"][0]["source_track"]["observations"][0]["source_frame_id"] = "decoded-000000004"
    masks["digest"] = canonical_digest(masks, digest_field="digest")
    with pytest.raises(ValueError, match="observation_missing"):
        geometry.prepare_task_geometry_inputs(**kwargs)
    assert calls == []


def test_already_observed_task_keeps_the_original_geometry_batch(task_geometry_case):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    kwargs, calls = task_geometry_case
    masks = kwargs["task_masks"]
    masks["targets"][0]["source_track"]["observations"][0]["source_frame_id"] = "decoded-000000002"
    masks["digest"] = canonical_digest(masks, digest_field="digest")
    assert geometry.prepare_task_geometry_inputs(**kwargs) == (kwargs["inputs"], kwargs["output_root"])
    assert calls == []


def test_different_capture_masks_cannot_change_geometry_inputs(task_geometry_case):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    kwargs, calls = task_geometry_case
    kwargs["task_masks"]["source_video_digest"] = "other-video"
    kwargs["task_masks"]["digest"] = canonical_digest(kwargs["task_masks"], digest_field="digest")
    with pytest.raises(ValueError, match="input_mismatch"):
        geometry.prepare_task_geometry_inputs(**kwargs)
    assert calls == []
