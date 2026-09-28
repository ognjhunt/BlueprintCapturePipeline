"""The deferred static objects use exact retained paid SAM evidence only."""

from __future__ import annotations

import copy
import json

import pytest

from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.meta_sam31 import PROFILE
from blueprint_pipeline.website_task_masks import complete_retained_static_task_masks


def _track(track_id: str, label: str, start: int) -> dict:
    return {"track_id": track_id, "label": label, "observations": [
        {"source_frame_id": frame, "width": 4, "height": 4,
         "runs": [{"start": start, "length": 2}, {"start": start + 4, "length": 2}]}
        for frame in ("frame-0", "frame-1")]}


def _target(target_id: str, label: str, box: list[float], effect: str, disposition: str) -> dict:
    return {"target_id": target_id, "semantic_label": label, "segmentation_prompt": label,
            "task_effect": effect, "disposition": disposition,
            "spatial_evidence": [{"timestamp_seconds": 0,
                                  "box_xywh_normalized": box}]}


def _fixture(tmp_path, *, continuous=False):
    video_path = tmp_path / "mask_view_plan" / "1" / "clip.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"retained scene-specific selected original views")
    source = {"geometry_available": False, "geometry_input_digest": "sha256:source-input",
              "binding": {"source_video_digest": "sha256:scene-video"},
              "frames": [{"frame_id": frame, "timestamp_seconds": index / 30, "width": 4, "height": 4}
                         for index, frame in enumerate(("frame-0", "frame-1"))]}
    source["digest"] = canonical_digest(source, digest_field="digest")
    plan = {"task_context_sha256": "sha256:task", "targets": [
        _target("cabinet", "under-desk cabinet", [0, 0, .5, .5], "manipulated", "remove"),
        _target("backpack", "teal backpack", [.5, 0, .5, .5], "static_obstacle", "keep"),
        _target("desk", "desk", [0, .5, .5, .5], "static_obstacle", "keep")]}
    sparse = [{"source_frame_id": frame, "model_frame_index": index, "decoded_pts_seconds": index / 30,
               "width": 4, "height": 4, "retained_video_digest": "sha256:scene-video"}
              for index, frame in enumerate(("frame-0", "frame-1"))]
    video = {"path": str(video_path), "sha256": _sha256_file(video_path),
             "source_video_digest": "sha256:scene-video", "encoding": "selected_original_views_lossless_h264_v1"}
    view_plan = {"binding": {"source_geometry_digest": source["digest"],
                             "source_video_digest": "sha256:scene-video", "task_context_sha256": "sha256:task"},
                 "sparse_registry": sparse, "source_frame_registry": sparse, "video": video}
    view_plan["digest"] = canonical_digest(view_plan, digest_field="digest")
    write_json(video_path.parent / "view_plan.json", view_plan)
    mask_binding = {"source_frames_digest": source["digest"], "geometry_input_digest": "sha256:source-input",
                    "task_context_sha256": "sha256:task", "task_targets": plan["targets"],
                    "mask_input_pixels": "continuous_source_video_v1" if continuous else "selected_original_views_v1",
                    "profile_digest": canonical_digest(PROFILE)}
    if not continuous:
        mask_binding["view_plan_digest"] = view_plan["digest"]
    mask_root = tmp_path / "task_masks" / canonical_digest(mask_binding)[7:23]
    mask_root.mkdir(parents=True)
    if continuous:
        retained_video = mask_root / "continuous-upright.mp4"
        retained_video.write_bytes(video_path.read_bytes())
        video = {**video, "path": str(retained_video),
                 "encoding": "upright_h264_crf18_veryfast_threads2_all_source_frames_v2"}
        receipt = {"registry": sparse, "video": video}
        receipt["digest"] = canonical_digest(receipt, digest_field="digest")
        write_json(mask_root / "continuous-video.json", receipt)
    initial = {"status": "object_removal_ready", "binding": mask_binding,
               "source_video_digest": "sha256:scene-video", "source_frame_registry": sparse,
               "deferred_target_ids": ["backpack", "desk"],
               "targets": [{"target_id": "cabinet", "source_track": _track("cabinet-1", "cabinet", 0)}]}
    initial["digest"] = canonical_digest(initial, digest_field="digest")
    write_json(mask_root / "task_masks.object_removal.json", initial)
    prompts = [{"prompt_id": target["target_id"], "output_label": target["target_id"],
                "text": target["segmentation_prompt"]} for target in plan["targets"]]
    provider_binding = {"profile": PROFILE, "frame_registry": sparse, "frame_artifacts": [],
                        "prompts": prompts,
                        "continuous_video": {key: video[key] for key in ("sha256", "source_video_digest", "encoding")},
                        "video_transport": "meta_files_v1"}
    provider_digest = canonical_digest(provider_binding)
    paid_root = mask_root / provider_digest[7:]
    paid_root.mkdir()
    write_json(paid_root / "binding.json", provider_binding)
    tracks = [_track("cabinet-1", "cabinet", 0), _track("backpack-1", "backpack", 2),
              _track("desk-1", "desk", 8), _track("desk-2", "desk", 2),
              _track("desk-3", "desk", 10), _track("desk-4", "desk", 0)]
    receipts = []
    for index, prompt in enumerate(prompts):
        response_path = paid_root / f"response-{index}.json"
        write_json(response_path, {"binding_digest": provider_digest, "clip_digest": video["sha256"],
                                   "response": {"status": "completed"}})
        response_digest = _sha256_file(response_path)
        receipts.append({"path": str(response_path), "sha256": response_digest})
        parsed = {"binding": {"response_digest": response_digest, "request_digest": provider_digest,
                              "parser_revision": 1, "profile": PROFILE},
                  "tracks": [track for track in tracks if track["label"] == prompt["output_label"]]}
        parsed["digest"] = canonical_digest(parsed, digest_field="digest")
        write_json(paid_root / f"parsed-response-{index}.json", parsed)
    result = {"schema_version": "website_meta_sam31_tracks.v1", "status": "completed",
              "binding_digest": provider_digest, "profile": PROFILE,
              "tracks": tracks, "responses": receipts}
    write_json(paid_root / "tracks.json", result)
    return {"plan": plan, "source_geometry": source, "task_masks": initial,
            "output_root": tmp_path / "task_masks", "view_plan_root": tmp_path / "mask_view_plan"}, paid_root


def test_spatially_selected_static_tracks_reuse_exact_retained_receipts(tmp_path):
    kwargs, _ = _fixture(tmp_path)
    complete = complete_retained_static_task_masks(**kwargs)
    assert complete["status"] == "completed"
    assert [(target["target_id"], target["source_track"]["track_id"]) for target in complete["targets"]] == [
        ("cabinet", "cabinet-1"), ("backpack", "backpack-1"), ("desk", "desk-1")]
    assert complete_retained_static_task_masks(**kwargs) == complete
    assert json.loads(next(kwargs["output_root"].glob("*/task_masks.json")).read_text()) == complete


def test_continuous_video_response_is_reused_without_a_new_provider_request(tmp_path):
    kwargs, _ = _fixture(tmp_path, continuous=True)
    complete = complete_retained_static_task_masks(**kwargs)
    assert complete["status"] == "completed"
    assert {row["target_id"] for row in complete["targets"]} == {"cabinet", "backpack", "desk"}
    assert complete["retained_static_source"]["retained_video_receipt_digest"].startswith("sha256:")


@pytest.mark.parametrize("change,error", [
    ("scene", "binding_invalid"), ("plan", "binding_invalid"),
    ("provider_binding", "provider_binding_changed"), ("response", "provider_receipts_invalid"),
    ("tracks", "provider_tracks_changed"), ("ambiguous", "track_ambiguous"),
])
def test_retained_reuse_fails_closed_for_cross_scene_or_unproved_tracks(tmp_path, change, error):
    kwargs, paid_root = _fixture(tmp_path)
    if change == "scene":
        kwargs["source_geometry"] = copy.deepcopy(kwargs["source_geometry"])
        kwargs["source_geometry"]["binding"]["source_video_digest"] = "sha256:other-scene"
        kwargs["source_geometry"]["digest"] = canonical_digest(kwargs["source_geometry"], digest_field="digest")
    elif change == "plan":
        kwargs["plan"] = copy.deepcopy(kwargs["plan"])
        kwargs["plan"]["targets"][1]["spatial_evidence"][0]["box_xywh_normalized"] = [0, 0, .5, .5]
    elif change == "provider_binding":
        binding_path = paid_root / "binding.json"
        binding = json.loads(binding_path.read_text())
        binding["continuous_video"]["source_video_digest"] = "sha256:other-scene"
        write_json(binding_path, binding)
    elif change == "response":
        (paid_root / "response-1.json").write_text('{"changed":true}')
    elif change == "tracks":
        result_path = paid_root / "tracks.json"
        result = json.loads(result_path.read_text())
        result["tracks"][1]["track_id"] = "substituted"
        write_json(result_path, result)
    else:
        result_path = paid_root / "tracks.json"
        result = json.loads(result_path.read_text())
        result["tracks"][3] = copy.deepcopy(result["tracks"][2])
        result["tracks"][3]["track_id"] = "desk-2"
        write_json(result_path, result)
        parsed_path = paid_root / "parsed-response-2.json"
        parsed = json.loads(parsed_path.read_text())
        parsed["tracks"][1] = result["tracks"][3]
        parsed["digest"] = canonical_digest(parsed, digest_field="digest")
        write_json(parsed_path, parsed)
    with pytest.raises(ValueError, match=error):
        complete_retained_static_task_masks(**kwargs)
