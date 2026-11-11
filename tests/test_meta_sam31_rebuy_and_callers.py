"""A clip re-buy survives a refused reservation or a crash at any step, and every caller goes through one.

Also: the scale references' segmentation inside the source pose stays an optional step (its stop is
recorded and the job goes on), and a request that names an uploaded file keeps ``origin/main``'s
answer rules exactly.
"""
from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
from urllib.error import HTTPError

import numpy as np
import pytest

from blueprint_pipeline import meta_sam31 as sam
from blueprint_pipeline import website_task_context

from .test_meta_sam31_inline_clip import CONTEXT, Provider, _frames, _job, _mask, _reservations


class Stop(Exception):
    """Ends a caller right after its SAM call: only the SAM calls are under test."""


def _partial_then_too_large(args, monkeypatch, root, *, failing="object 1"):
    """A first run with one prompt refused, then a lower ceiling the retained clip no longer fits."""
    holds = _reservations(monkeypatch)
    provider = Provider({failing: [413]})
    with pytest.raises(ValueError, match="meta_sam_http_413"):
        sam.run_meta_sam31(**args, opener=provider)
    body = sam._inline_body_bytes(root / "retained-frames.mp4", args["prompts"])
    monkeypatch.setattr(sam, "INLINE_REQUEST_CEILING_BYTES", body - 1)
    return holds, provider


# --- point 1: the re-buy is recorded before anything moves ------------------------------------

def test_a_refused_re_buy_reservation_never_falls_back_to_the_original_binding(tmp_path, monkeypatch):
    args, digest, root = _job(tmp_path, monkeypatch)
    holds, provider = _partial_then_too_large(args, monkeypatch, root)
    reserve = website_task_context.reserve_website_sam_spend

    def refused(**kwargs):
        raise ValueError("website_control_preparation-spend_http_409:website_scene_preparation_budget_exhausted")

    monkeypatch.setattr(website_task_context, "reserve_website_sam_spend", refused)
    with pytest.raises(ValueError, match="budget_exhausted"):
        sam.run_meta_sam31(**args, opener=provider)
    monkeypatch.setattr(website_task_context, "reserve_website_sam_spend", reserve)
    assert sam.run_meta_sam31(**args, opener=provider)["status"] == "completed"
    assert [hold["allocation_binding_digest"] == digest for hold in holds] == [True, False]
    assert holds[1]["request_count"] == 3


@pytest.mark.parametrize("crash_after", ["mkdir", "record", "first_move", "all_moves"])
def test_a_crash_at_any_step_of_the_set_aside_is_finished_and_re_bought_once(tmp_path, monkeypatch, crash_after):
    args, digest, root = _job(tmp_path, monkeypatch)
    holds, provider = _partial_then_too_large(args, monkeypatch, root)
    real_replace, real_mkdir, moves = os.replace, Path.mkdir, []

    def replace(src, dst):
        real_replace(src, dst)
        if "superseded-clip" in str(dst):
            if str(dst).endswith(sam.SET_ASIDE_RECORD) and crash_after == "record":
                raise KeyboardInterrupt("killed")
            if not str(dst).endswith(sam.SET_ASIDE_RECORD):
                moves.append(dst)
                if crash_after == "first_move" and len(moves) == 1:
                    raise KeyboardInterrupt("killed")

    def mkdir(self, *a, **k):
        real_mkdir(self, *a, **k)
        if crash_after == "mkdir" and "superseded-clip" in str(self):
            raise KeyboardInterrupt("killed")

    monkeypatch.setattr(sam.os, "replace", replace)
    monkeypatch.setattr(Path, "mkdir", mkdir)
    if crash_after == "all_moves":
        monkeypatch.setattr(website_task_context, "reserve_website_sam_spend",
                            lambda **_: (_ for _ in ()).throw(KeyboardInterrupt("killed")))
    with pytest.raises(KeyboardInterrupt):
        sam.run_meta_sam31(**args, opener=provider)
    monkeypatch.setattr(sam.os, "replace", real_replace)
    monkeypatch.setattr(Path, "mkdir", real_mkdir)
    holds_before = len(holds)
    if crash_after == "all_moves":
        holds = _reservations(monkeypatch)
        holds_before = 0
    assert sam.run_meta_sam31(**args, opener=provider)["status"] == "completed"
    new = holds[holds_before:]
    assert len(new) == 1 and new[0]["allocation_binding_digest"] != digest and new[0]["request_count"] == 3
    responses = [json.loads((root / f"response-{i}.json").read_text()) for i in range(3)]
    assert len({row["clip_digest"] for row in responses}) == 1
    # A later run buys nothing more.
    count = len(holds)
    assert sam.run_meta_sam31(**args, opener=lambda *a, **k: pytest.fail("rebought"))["status"] == "completed"
    assert len(holds) == count


# --- the background step and the task masks, each through one re-buy -------------------------

def test_the_background_step_goes_through_one_re_buy(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_background_automation as background

    monkeypatch.setenv("META_MODEL_API_KEY", "fixture-meta-secret")
    rows, artifacts = _frames(tmp_path, 2, 64, noise=12)
    frames = [{"frame_id": row["source_frame_id"], "intrinsics": {"width": 64, "height": 64},
               "image_path": art["path"], "image_digest": art["sha256"], "rgb": np.zeros((64, 64, 3), np.uint8)}
              for row, art in zip(rows, artifacts)]
    objects = [{"target_id": "tray", "segmentation_prompt": "wooden tray", "disposition": "keep"},
               {"target_id": "oven", "segmentation_prompt": "double oven", "disposition": "keep"}]
    monkeypatch.setattr(background, "object_evidence", lambda **kw: {row["target_id"]: [] for row in objects})
    holds = _reservations(monkeypatch)
    provider = Provider({"double oven": [413]})
    real = sam.run_meta_sam31

    def through(**kw):
        result = real(**kw, opener=provider)
        raise Stop(result)

    monkeypatch.setattr(sam, "run_meta_sam31", through)
    arguments = dict(frames=frames, objects=objects, inputs={"frames": []}, task_context=dict(CONTEXT),
                     source_digest="sha256:" + "e" * 64, root=tmp_path / "background")
    with pytest.raises(ValueError, match="meta_sam_http_413"):
        background.segment_objects(**arguments)
    root = next((tmp_path / "background").iterdir())
    monkeypatch.setattr(sam, "INLINE_REQUEST_CEILING_BYTES",
                        sam._inline_body_bytes(root / "retained-frames.mp4", [{"text": "double oven"}]) - 1)
    with pytest.raises(Stop):
        background.segment_objects(**arguments)
    assert provider.sent == ["wooden tray", "double oven", "wooden tray", "double oven"] and len(holds) == 2
    with pytest.raises(Stop):
        background.segment_objects(**arguments)
    assert len(provider.sent) == 4 and len(holds) == 2


def test_the_task_masks_without_a_source_video_go_through_one_re_buy(tmp_path, monkeypatch):
    from blueprint_pipeline import website_task_masks as masks

    monkeypatch.setenv("META_MODEL_API_KEY", "fixture-meta-secret")
    monkeypatch.setenv("BLUEPRINT_WEBSITE_SAM31_PROVIDER", "meta")
    rows, artifacts = _frames(tmp_path, 2, 64, noise=12)
    frames = [{"frame_id": row["source_frame_id"], "timestamp_seconds": float(index), "width": 64, "height": 64,
               "source_image_path": art["path"], "source_image_digest": art["sha256"],
               "display_rotation_degrees": 0}
              for index, (row, art) in enumerate(zip(rows, artifacts))]
    targets = [{"target_id": "tray", "semantic_label": "wooden tray", "task_effect": "manipulated",
                "disposition": "remove", "spatial_evidence": [{"timestamp_seconds": 0.0,
                                                               "box_xywh_normalized": [0, 0, 0.5, 0.5]}]},
               {"target_id": "table", "semantic_label": "dining table", "task_effect": "static_contact",
                "disposition": "keep", "spatial_evidence": [{"timestamp_seconds": 1.0,
                                                             "box_xywh_normalized": [0, 0, 1, 1]}]}]
    holds = _reservations(monkeypatch)
    provider = Provider({"dining table": [413]})
    real = sam.run_meta_sam31

    def through(**kw):
        raise Stop(real(**kw, opener=provider))

    monkeypatch.setattr(masks, "run_meta_sam31", through)
    arguments = dict(plan={"targets": targets, "task_context_sha256": "task"},
                     source_geometry={"digest": "source", "geometry_available": False,
                                      "binding": {"source_video_digest": "video"}, "frames": frames},
                     output_root=tmp_path / "masks", task_context=dict(CONTEXT))
    with pytest.raises(ValueError, match="meta_sam_http_413"):
        masks.run_website_task_masks(**arguments)
    root = next(path.parent for path in (tmp_path / "masks").rglob("retained-frames.mp4"))
    monkeypatch.setattr(sam, "INLINE_REQUEST_CEILING_BYTES",
                        sam._inline_body_bytes(root / "retained-frames.mp4", [{"text": "dining table"}]) - 1)
    with pytest.raises(Stop):
        masks.run_website_task_masks(**arguments)
    assert provider.sent == ["wooden tray", "dining table", "wooden tray", "dining table"] and len(holds) == 2
    with pytest.raises(Stop):
        masks.run_website_task_masks(**arguments)
    assert len(provider.sent) == 4 and len(holds) == 2


# --- point 2: the optional segmentation inside the source pose --------------------------------

@pytest.mark.parametrize("code", ["meta_sam_http_413", "meta_sam_http_503", "meta_sam_http_502", "meta_sam_http_429"])
def test_a_provider_answer_in_the_source_poses_reference_segmentation_is_recorded_not_raised(tmp_path, monkeypatch, code):
    # The windowed re-pose records this as repose_unavailable and keeps the first pose; the preserved
    # scene keeps building with the stop as a blocker. Neither may fail or hold the job.
    from blueprint_pipeline import atlas_context_fov, atlas_reference_masks, atlas_scene_measurements
    from blueprint_pipeline import atlas_source_pose_inputs, paid_resource_allocator
    from blueprint_pipeline import website_atlas_pipeline as pipeline

    inputs = {"frames": [{"frame_id": "a"}], "binding": {"source_video_digest": "video"}}
    (tmp_path / "atlas_posed_rgbd_result.json").write_text("{}")
    monkeypatch.setattr(atlas_source_pose_inputs, "prepare_source_pose", lambda root: (root, inputs))
    monkeypatch.setattr(paid_resource_allocator, "run_sponsored_website_atlas_pose", lambda **kw: {"status": "completed"})
    monkeypatch.setattr(atlas_scene_measurements, "load_views", lambda *a: ({"digest": "pose"}, [{"frame_id": "a"}]))
    monkeypatch.setattr(atlas_context_fov, "verify_pose_context_fov", lambda **kw: None)
    from blueprint_pipeline import atlas_scene_automation
    monkeypatch.setattr(atlas_scene_automation, "measurement_plan", lambda review: {"scale_references": []})

    def refused(**kw):
        raise ValueError(code)

    monkeypatch.setattr(atlas_reference_masks, "segment_scale_references", refused)
    policy = {"pose_input_cost_usd": 0.01, "rate_card_reference": "r", "reviewed_by": "b", "reviewed_on": "o"}
    result = pipeline._source_pose(pose_root=tmp_path, inputs=inputs, review={"automation": {
        "scale_references": [], "objects": []}}, policy=policy, task_context={"capture_id": "c"},
        check=lambda *_: None, experimental=False, max_polls=1)
    assert result == {"status": "needs_input", "reason": code}


# --- point 4: a request that names an uploaded file keeps main's rules ------------------------

def _continuous(tmp_path, monkeypatch):
    args, _, _ = _job(tmp_path, monkeypatch)
    rows = [{**args["frame_registry"][0], "source_frame_id": f"f{i}"} for i in range(3)]
    artifacts = [{**args["frame_artifacts"][0], "source_frame_id": row["source_frame_id"]} for row in rows]
    source = sam.encode_clip(registry=rows, artifacts=artifacts, root=tmp_path)
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
    registry, video = sam.prepare_continuous_video(source=source, source_digest=_sha256_file(source),
                                                   root=tmp_path / "continuous")
    return {**args, "frame_registry": registry, "frame_artifacts": [], "video_artifact": video}


class UploadProvider:
    def __init__(self, status):
        self.calls, self.status, self.uploads = [], status, 0

    def __call__(self, request, **_kwargs):
        if request.full_url == "https://api.meta.ai/v1/files":
            self.uploads += 1
            self.calls.append("upload")
            return BytesIO(json.dumps({"id": f"file-u{self.uploads}"}).encode())
        if request.get_method() == "DELETE":
            self.calls.append("delete:" + request.full_url.rsplit("/", 1)[1])
            return BytesIO(b'{"deleted":true}')
        payload = json.loads(request.data)
        text, media = payload["input"][0]["content"][0]["text"], payload["input"][0]["content"][1]
        self.calls.append((text, media["file_id"]))
        if text == "object 1" and self.status is not None:
            status, self.status = self.status, None
            raise HTTPError(sam.ENDPOINT, status, "Answered", {}, None)
        answer = {"status": "completed", "usage": {"video_frames_processed": 3},
                  "output": [{"type": "message", "content": [{"type": "output_text",
                                                               "text": _mask(64).replace("<0f>", "<2f>")}]}]}
        return BytesIO(json.dumps(answer).encode())


#: What ``origin/main`` (49759e2e0) does with each status on a run that uploads, observed by running
#: this same scenario against main's ``meta_sam31``: the second run's outcome and its provider calls.
MAIN_SECOND_RUN = {
    # Resent naming the run's first upload, as main does (main's own fault: that file was deleted
    # when the first run ended; the tracker lists it).
    **{status: ("completed", [("object 1", "file-u1"), ("object 2", "file-u1"), "delete:file-u1"])
       for status in (400, 401, 403, 409, 413, 415, 422, 429, 503)},
    404: ("completed", ["upload", ("object 1", "file-u2"), ("object 2", "file-u2"), "delete:file-u2"]),
    **{status: ("meta_sam_submission_requires_reconciliation", []) for status in (408, 425, 500, 502, 504)},
}


@pytest.mark.parametrize("status", sorted(MAIN_SECOND_RUN))
def test_a_run_that_uploads_handles_every_status_as_main_does(tmp_path, monkeypatch, status):
    args = _continuous(tmp_path, monkeypatch)
    _reservations(monkeypatch)
    provider = UploadProvider(status)
    with pytest.raises(ValueError, match=f"meta_sam_http_{status}"):
        sam.run_meta_sam31(**args, opener=provider)
    first = list(provider.calls)
    try:
        outcome = sam.run_meta_sam31(**args, opener=provider)["status"]
    except ValueError as exc:
        outcome = str(exc)
    expected_outcome, expected_calls = MAIN_SECOND_RUN[status]
    assert first == ["upload", ("object 0", "file-u1"), ("object 1", "file-u1"), "delete:file-u1"]
    assert (outcome, provider.calls[len(first):]) == (expected_outcome, expected_calls)
