"""Several manipulated objects (plus people) are removed from website views end to end."""
from __future__ import annotations

import hashlib
import json
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import binary_dilation

from blueprint_pipeline import website_image_completion as completion
from blueprint_pipeline import website_image_repair_agent as repair
from blueprint_pipeline import website_object_removal as removal
from blueprint_pipeline import website_task_masks as masks
from blueprint_pipeline.clean_plate_removal_analysis_gemini import (
    build_removal_plan, parse_removal_plan_response, validate_removal_plan)
from blueprint_pipeline.clean_plate_stage import _build_removal_manifest
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.semantic_teacher_image_edit_worker import _normalized_usage

TASK = "Put the cup and the bowl in the dishwasher and close its door."
SIZE = 32
BOXES = {"cup": (2, 2), "bowl": (2, 20), "person": (12, 26)}  # (row, col) of a 4x4 square


def _row(target_id, label, effect, *, role, target_class="movable_object", disposition="keep",
         rebuild="none", quote="", **extra):
    return {"target_id": target_id, "semantic_label": label, "segmentation_prompt": label,
            "target_class": target_class, "target_role": role, "task_effect": effect,
            "decision_reason": "visible in the task area", "task_basis_quote": quote,
            "clarification_question": "", "placement_relation": "", "articulated_part": "",
            "articulation_kind": "", "disposition": disposition, "rebuild_intent": rebuild, "confidence": 0.9,
            "spatial_evidence": [{"timestamp_seconds": 1.0, "box_xywh_normalized": [0.1, 0.1, 0.2, 0.2]}],
            **extra}


def _analysis_rows():
    removed = dict(role="task_object", disposition="remove", rebuild="rebuild_and_compose")
    return [_row("cup", "white cup", "manipulated", quote="the cup", **removed),
            _row("bowl", "blue bowl", "manipulated", quote="the bowl", **removed),
            _row("dishwasher", "dishwasher", "manipulated", quote="close its door", **removed,
                 articulated_part="door", articulation_kind="revolute"),
            _row("counter", "counter", "static_contact", role="support"),
            _row("person", "person", "privacy", role="person", target_class="person", disposition="remove")]


def _plan():
    targets = parse_removal_plan_response(json.dumps({"targets": _analysis_rows()}), strict=True,
                                          task_description=TASK)
    plan = build_removal_plan(targets=targets, status="completed", model="fixture", processing="static")
    return {**plan, "task_context_sha256": "task"}


def test_removal_plan_and_manifest_carry_every_manipulated_object():
    plan = _plan()
    assert validate_removal_plan(plan) == []
    assert plan["movable_removal_count"] == 3 and plan["person_target_count"] == 1
    manifest = _build_removal_manifest(plan)
    # One entry, and one compose-back slot, per removed object; never the person or the kept support.
    assert [row["target_id"] for row in manifest["entries"]] == ["cup", "bowl", "dishwasher"]
    assert manifest["removed_target_count"] == 3
    slots = [row["compose_back"] for row in manifest["entries"]]
    assert all(slot == {"replacement_asset_id": None, "pose_world": None,
                        "replacement_asset_frame_registration_uri": None} for slot in slots)
    slots[0]["replacement_asset_id"] = "cup-asset"
    assert slots[1]["replacement_asset_id"] is None and slots[2]["replacement_asset_id"] is None
    assert manifest["entries"][2]["articulation_kind"] == "revolute"


def _square(frame_id, row, col, *, size=4, extent=SIZE):
    return {"source_frame_id": frame_id, "height": extent, "width": extent,
            "runs": [{"start": (row + r) * extent + col, "length": size} for r in range(size)]}


def _box(row, col, size=4):
    return [col / SIZE, row / SIZE, size / SIZE, size / SIZE]


def test_tracking_selects_every_object_and_removal_masks_union_them_with_the_person(tmp_path, monkeypatch):
    plan = _plan()
    targets = [dict(row) for row in plan["targets"] if row["target_id"] != "dishwasher"]
    for row in targets:
        if row["target_id"] in ("cup", "bowl"):
            row["spatial_evidence"] = [{"timestamp_seconds": 1 / 30, "box_xywh_normalized": _box(*BOXES[row["target_id"]])}]
        elif row["target_id"] == "counter":
            row["spatial_evidence"] = [{"timestamp_seconds": 1 / 30, "box_xywh_normalized": [0, 28 / SIZE, 1, 4 / SIZE]}]
        else:
            row["spatial_evidence"] = []
    counter = {"source_frame_id": "anchor", "height": SIZE, "width": SIZE,
               "runs": [{"start": 28 * SIZE, "length": 4 * SIZE}]}
    tracks = [{"track_id": f"{name}-1", "label": name, "observations": [_square("anchor", *BOXES[name])]}
              for name in ("cup", "bowl", "person")]
    tracks.append({"track_id": "counter-1", "label": "counter", "observations": [counter]})
    registry = [{"source_frame_id": "anchor", "decoded_pts_seconds": 1 / 30, "width": SIZE, "height": SIZE}]
    calls = []
    monkeypatch.setenv("BLUEPRINT_WEBSITE_SAM31_PROVIDER", "meta")
    monkeypatch.setattr(masks, "prepare_continuous_video", lambda **kw: (registry, {"path": "prepared.mp4"}))
    monkeypatch.setattr(masks, "run_meta_sam31", lambda **kw: calls.append(kw["prompts"]) or {"tracks": tracks})
    result = masks.run_website_task_masks(
        plan={"targets": targets, "task_context_sha256": "task"},
        source_geometry={"digest": "source", "geometry_available": False, "binding": {"source_video_digest": "video"},
                         "frames": [{"frame_id": "anchor", "timestamp_seconds": 1 / 30, "width": SIZE, "height": SIZE}]},
        source_video=tmp_path / "source.mov", output_root=tmp_path / "masks")
    # One continuous-video call carries every concept; every target keeps its own track.
    assert len(calls) == 1 and [p["text"] for p in calls[0]] == ["white cup", "blue bowl", "counter", "person"]
    by_id = {row["target_id"]: row for row in result["targets"]}
    assert set(by_id) == {"cup", "bowl", "counter", "person"}
    assert {by_id[name]["track"]["track_id"] for name in ("cup", "bowl", "counter")} == {"cup-1", "bowl-1", "counter-1"}
    assert [row["target_id"] for row in result["targets"] if row["task_effect"] == "manipulated"] == ["cup", "bowl"]
    assert by_id["counter"]["disposition"] == "keep"

    source = tmp_path / "anchor.png"
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8)).save(source)
    frame = {"frame_id": "anchor", "source_image_path": str(source), "source_image_digest": _sha256_file(source),
             "display_rotation_degrees": 0}
    prepared = removal.prepare_object_removal_frames(frames=[frame], task_masks=result, output_root=tmp_path / "edit")[0]
    expected = np.zeros((SIZE, SIZE), dtype=bool)
    for row, col in BOXES.values():
        expected[row:row + 4, col:col + 4] = True
    expected = binary_dilation(expected, iterations=3)
    mask = np.asarray(Image.open(prepared["remaining_mask_path"])) == 255
    np.testing.assert_array_equal(mask, expected)
    # The kept support is never erased.
    assert not mask[28:].any()
    assert prepared["removed_task_object_ids"] == ["cup", "bowl"]


def test_two_objects_of_one_class_share_a_concept_but_never_a_track(tmp_path, monkeypatch):
    left = {"target_id": "cup-left", "semantic_label": "cup", "segmentation_prompt": "white cup",
            "task_effect": "manipulated", "disposition": "remove",
            "spatial_evidence": [{"timestamp_seconds": 1 / 30, "box_xywh_normalized": _box(*BOXES["cup"])}]}
    right = {**left, "target_id": "cup-right",
             "spatial_evidence": [{"timestamp_seconds": 1 / 30, "box_xywh_normalized": _box(*BOXES["bowl"])}]}
    tracks = [{"track_id": f"cup-{i}", "label": "cup-left", "observations": [_square("anchor", *BOXES[name])]}
              for i, name in enumerate(("bowl", "cup"))]
    registry = [{"source_frame_id": "anchor", "decoded_pts_seconds": 1 / 30, "width": SIZE, "height": SIZE}]
    calls = []
    monkeypatch.setenv("BLUEPRINT_WEBSITE_SAM31_PROVIDER", "meta")
    monkeypatch.setattr(masks, "prepare_continuous_video", lambda **kw: (registry, {"path": "prepared.mp4"}))
    monkeypatch.setattr(masks, "run_meta_sam31", lambda **kw: calls.append(kw["prompts"]) or {"tracks": tracks})
    result = masks.run_website_task_masks(
        plan={"targets": [left, right], "task_context_sha256": "task"},
        source_geometry={"digest": "source", "geometry_available": False, "binding": {"source_video_digest": "video"},
                         "frames": [{"frame_id": "anchor", "timestamp_seconds": 1 / 30, "width": SIZE, "height": SIZE}]},
        source_video=tmp_path / "source.mov", output_root=tmp_path / "masks")
    assert len(calls[0]) == 1
    assert {row["target_id"]: row["track"]["track_id"] for row in result["targets"]} == {
        "cup-left": "cup-1", "cup-right": "cup-0"}


def _views(tmp_path, count):
    frames = []
    for index in range(count):
        path = tmp_path / f"view-{index}.png"
        Image.fromarray(np.random.default_rng(index).integers(0, 255, (16, 16, 3), dtype=np.uint8)).save(path)
        frames.append({"frame_id": f"f{index}", "image_path": str(path), "image_digest": _sha256_file(path),
                       "remaining_pixel_count": 0})
    return frames


def _observed(*indexes):
    return {"observations": [{"source_frame_id": f"f{i}"} for i in indexes]}


def test_every_manipulated_object_keeps_its_own_work_views(tmp_path):
    frames = _views(tmp_path, 20)
    # The dishwasher is in every view; the cup and bowl only briefly. The union
    # of observations spans the whole clip, so first/last of the union alone
    # would say nothing about the cup or the bowl.
    task_masks = {"targets": [
        {"target_id": "dishwasher", "task_effect": "manipulated", "track": _observed(*range(20))},
        {"target_id": "cup", "task_effect": "manipulated", "track": _observed(5, 6)},
        {"target_id": "bowl", "task_effect": "manipulated", "source_track": _observed(12, 13), "track": _observed()},
        {"target_id": "person", "task_effect": "privacy", "track": _observed(9)},
        {"target_id": "counter", "task_effect": "static_contact", "track": _observed(3)}]}
    selected = {row["frame_id"] for row in removal.select_reconstruction_frames(frames=frames, task_masks=task_masks,
                                                                                limit=3)}
    assert selected == {"f0", "f5", "f12"}
    selected = {row["frame_id"] for row in removal.select_reconstruction_frames(frames=frames, task_masks=task_masks,
                                                                                limit=8)}
    assert {"f0", "f19", "f5", "f6", "f12", "f13"} <= selected and len(selected) == 8
    with pytest.raises(ValueError, match="frame_limit_below_task_objects"):
        removal.select_reconstruction_frames(frames=frames, task_masks=task_masks, limit=2)


def test_single_object_view_selection_is_unchanged(tmp_path):
    frames = _views(tmp_path, 20)
    frames[19] = {**frames[19], "image_digest": frames[4]["image_digest"], "image_path": frames[4]["image_path"]}
    task_masks = {"targets": [{"task_effect": "manipulated", "track": _observed(4, 7, 19)},
                              {"target_id": "person", "task_effect": "privacy", "track": _observed(0)}]}
    selected = removal.select_reconstruction_frames(frames=frames, task_masks=task_masks, limit=5)
    # Same-image last anchor is dropped exactly as before; context fills the rest.
    assert "f4" in {row["frame_id"] for row in selected} and "f19" not in {row["frame_id"] for row in selected}
    assert len(selected) == 5


def test_short_lived_views_of_every_object_are_decoded(tmp_path, monkeypatch):
    video = tmp_path / "video.mov"
    video.write_bytes(b"source")
    registry = [{"source_frame_id": f"decoded-{i:09d}", "decoded_pts_seconds": i / 30} for i in range(90)]
    existing = [{"frame_id": registry[i]["source_frame_id"], "timestamp_seconds": i / 30,
                 "display_rotation_degrees": 0} for i in range(0, 90, 10)]
    geometry = {"frames": existing, "binding": {"source_video_digest": _sha256_file(video)}}
    task_masks = {"source_video_digest": _sha256_file(video), "source_frame_registry": registry, "targets": [
        {"task_effect": "manipulated", "source_track": {"observations": [
            {"source_frame_id": registry[i]["source_frame_id"]} for i in (43, 44, 45)]}},
        {"task_effect": "manipulated", "source_track": {"observations": [
            {"source_frame_id": registry[i]["source_frame_id"]} for i in (71, 72, 73)]}},
        {"task_effect": "privacy", "target_class": "person", "source_track": {"observations": [
            {"source_frame_id": registry[i]["source_frame_id"]} for i in (15, 16)]}}]}
    indexes = []

    def extract(**kw):
        indexes.extend(kw["indexes"])
        return [{"frame_id": registry[i]["source_frame_id"], "t_video_sec": i / 30, "digest": f"digest-{i}"}
                for i in kw["indexes"]]
    monkeypatch.setattr(removal, "_extract_frames", extract)
    monkeypatch.setattr(removal.shutil, "which", lambda _: "/ffmpeg")
    removal.reconstruction_source_frames(source_geometry=geometry, task_masks=task_masks, source_video=video,
                                         limit=8, output_root=tmp_path / "frames")
    assert indexes == [43, 45, 71, 73]


def _edit_inputs(tmp_path, counts, objects):
    frames = []
    for index, (count, shown) in enumerate(zip(counts, objects)):
        image, mask = tmp_path / f"source-{index}.png", tmp_path / f"mask-{index}.png"
        Image.new("RGB", (10, 20), ("red", "green", "blue")[index]).save(image)
        pixels = np.zeros((20, 10), dtype=np.uint8)
        pixels.reshape(-1)[:count] = 255
        Image.fromarray(pixels).save(mask)
        frames.append({"frame_id": str(index), "image_path": str(image), "image_digest": _sha256_file(image),
                       "remaining_mask_path": str(mask), "remaining_mask_digest": _sha256_file(mask),
                       "remaining_pixel_count": count, "removed_task_object_ids": shown})
    return frames


def test_edit_prompt_names_every_object_and_the_view_showing_most_objects_anchors(tmp_path, monkeypatch):
    from blueprint_pipeline.paid_resource_admission import require_paid_resource_admission
    targets = _plan()["targets"]
    prompt = completion._completion_prompt(targets)
    single = completion._completion_prompt([row for row in targets if row["target_id"] not in ("bowl", "dishwasher")])
    assert prompt.startswith(single.split(" Scene-specific targets")[0])
    names = json.loads(prompt.split(completion.MULTI_OBJECT_PROMPT)[1])
    assert [row["target_id"] for row in names["remove"]] == ["cup", "bowl", "dishwasher"]
    assert [row["target_id"] for row in names["keep"]] == ["counter"]
    assert "Remove every one of them" in prompt and "remove every person" in prompt

    # Frame 0 shows the most removal pixels (the dishwasher alone); frame 2 shows both small objects.
    frames = _edit_inputs(tmp_path, (40, 5, 20), (["dishwasher"], [], ["cup", "bowl"]))
    _, _, digest = completion._validated_backend(completion.REGISTRY_PATH, backend_id=completion.BACKEND_ID)
    binding = completion.completion_binding(frames, task_digest="task", backend_digest=digest, targets=targets)
    assert binding["reference_policy"] == "edited_anchor_most_task_objects"
    assert [row["removed_task_object_ids"] for row in binding["frames"]] == [["dishwasher"], [], ["cup", "bowl"]]
    admission = {"schema_version": "paid_lane_admission.v1", "status": "admitted", "blockers": [],
                 "resource_class": "openai_api_candidate", "external_disclosure_allowed": True,
                 "allocation_binding_digest": canonical_digest(binding), "maximum_cost_usd": 1.0}
    calls = []

    def edit(**kwargs):
        calls.append(kwargs)
        stream = BytesIO()
        Image.new("RGB", kwargs["expected_size"], ("white", "black", "gray")[len(calls) - 1]).save(stream, format="PNG")
        return {"succeeded": True, "generated": stream.getvalue(),
                "usage": _normalized_usage({"output_tokens_details": {"image_tokens": 100}})}
    monkeypatch.setattr(completion, "_execute_frame_request", edit)
    outputs = completion.complete_background_images(
        frames=frames, task_digest="task", output_root=tmp_path / "edits", admission=admission,
        admission_grant=require_paid_resource_admission(admission, resource_class="openai_api_candidate",
                                                        expected_schema_version="paid_lane_admission.v1"),
        token="test", targets=targets)
    first = np.asarray(Image.open(BytesIO(calls[0]["image_bytes"])).convert("RGB"))
    assert np.all(first[first.any(axis=-1)] == [0, 0, 255])
    anchor = Path(outputs[2]["image_path"]).read_bytes()
    assert [call["reference_images"] for call in calls[1:]] == [[anchor], [anchor]]
    assert all(call["prompt"] == prompt for call in calls)


# The dishwasher scene's retained paid edits and reviews are content addressed.
# These digests were computed on origin/main (including its #2199 review prompt).
SINGLE_BINDING_DIGEST = "sha256:852a8de192a120f450bb5c2e03e2aea4df43d6a1ee4e638852786432154009af"
SINGLE_REPAIR_BINDING_DIGEST = "sha256:0903c71068138cff0197884eb750175e8ed6fcaac057607a08616a0b7b720dae"
REVIEW_PROMPT_SHA256 = "14bc55f4b4b5fdeb0149834387716880bf1f779ce1c07fd5ef9231aa5c1b643e"
PROMPT_SHA256 = "f3ec969f5d959f4a5696ebc1afeb73f197cea8ee94681b38eedc6b55502f026e"


def test_single_object_edit_and_review_requests_are_byte_identical_to_before(tmp_path):
    frames = [{"frame_id": f"decoded-{i:09d}", "image_digest": "sha256:" + str(i) * 64,
               "remaining_mask_digest": "sha256:" + "ab" * 32, "edge_feather_pixels": 3,
               "remaining_pixel_count": 10 * i} for i in range(3)]
    targets = [{"target_id": "dishwasher", "semantic_label": "dishwasher", "task_effect": "manipulated",
                "disposition": "remove", "target_class": "movable_object", "articulated_part": "door",
                "articulation_kind": "revolute"},
               {"target_id": "person", "semantic_label": "person", "task_effect": "privacy", "disposition": "remove",
                "target_class": "person"},
               {"target_id": "counter", "semantic_label": "counter", "task_effect": "static_contact",
                "disposition": "keep", "target_class": "fixed_clutter"}]
    assert canonical_digest(completion.completion_binding(
        frames, task_digest="task", backend_digest="backend", targets=targets)) == SINGLE_BINDING_DIGEST
    assert canonical_digest(completion.completion_binding(
        frames, task_digest="task", backend_digest="backend", targets=targets, repair_instruction="Keep the mat",
        reference_digest="sha256:ref")) == SINGLE_REPAIR_BINDING_DIGEST
    assert hashlib.sha256(completion.PROMPT.encode()).hexdigest() == PROMPT_SHA256
    assert hashlib.sha256(completion.REVIEW_PROMPT.encode()).hexdigest() == REVIEW_PROMPT_SHA256
    assert completion._review_prompt(targets) is completion.REVIEW_PROMPT
    # A single-object frame record gains no per-object field.
    source = tmp_path / "view.png"
    Image.new("RGB", (8, 8)).save(source)
    prepared = removal.prepare_object_removal_frames(
        frames=[{"frame_id": "f0", "source_image_path": str(source), "source_image_digest": _sha256_file(source),
                 "display_rotation_degrees": 0}],
        task_masks={"targets": [{"target_id": "dishwasher", "task_effect": "manipulated", "disposition": "remove",
                                 "track": {"observations": [_square("f0", 0, 0, size=2, extent=8)]}},
                                {"target_id": "person", "target_class": "person", "task_effect": "privacy",
                                 "disposition": "remove", "track": {"observations": []}}]},
        output_root=tmp_path / "edit")[0]
    assert "removed_task_object_ids" not in prepared


def _fake_review(monkeypatch, answer):
    import google
    sent = []

    def generate(**kwargs):
        sent.append(kwargs["contents"])
        return NS(candidates=[NS(finish_reason="STOP")], text=json.dumps(answer))

    class Client:
        def __init__(self, **kwargs):
            self.models = NS(generate_content=generate)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    fake = NS(Client=Client, types=NS(Part=NS(from_bytes=lambda **kw: kw), GenerateContentConfig=NS,
                                      HttpOptions=NS, HttpRetryOptions=NS))
    monkeypatch.setattr(google, "genai", fake, raising=False)
    monkeypatch.setitem(sys.modules, "google.genai", fake)
    monkeypatch.setattr(completion, "_api_key", lambda: ("fixture-key", "fixture"))
    return sent


BLOCKED = {"consistent_background": True, "task_objects_removed": False, "unrelated_objects_preserved": True,
           "reason": "bowl remains", "remaining_task_object_frame_ids": ["1"]}


@pytest.mark.parametrize("objects,error", [
    ([{"frame_id": "1", "target_id": "bowl"}, {"frame_id": "1", "target_id": "person"}], None),
    (None, "review_objects_invalid"),
    ([{"frame_id": "1", "target_id": "counter"}], "review_objects_invalid"),
    ([{"frame_id": "0", "target_id": "bowl"}], "review_objects_invalid"),
    ([{"frame_id": "1", "target_id": "bowl"}, {"frame_id": "1", "target_id": "bowl"}], "review_objects_invalid"),
    ([["1", "bowl"]], "review_objects_invalid"),
])
def test_review_names_which_of_several_objects_each_view_still_shows(tmp_path, monkeypatch, objects, error):
    answer = dict(BLOCKED, **({} if objects is None else {"remaining_task_objects": objects}))
    sent = _fake_review(monkeypatch, answer)
    frames = _edit_inputs(tmp_path, (5, 5), ([], []))
    plan = {"task_context_sha256": "task", "targets": _plan()["targets"]}
    args = dict(frames=frames, original_frames=frames, plan=plan, output_root=tmp_path / "review",
                retain_result=False)
    if error:
        with pytest.raises(ValueError, match=error):
            completion._verify_completed_background(**args)
        return
    result = completion._verify_completed_background(**args)
    assert result["status"] == "blocked"
    assert result["review"]["remaining_task_objects"] == objects
    assert sent[0][0].startswith(completion.REVIEW_PROMPT.removesuffix("Targets: "))
    assert completion.MULTI_OBJECT_REVIEW_PROMPT in sent[0][0]
    assert result["binding"]["review_prompt"] != completion.REVIEW_PROMPT


def test_single_object_review_needs_no_object_names(tmp_path, monkeypatch):
    sent = _fake_review(monkeypatch, BLOCKED)
    frames = _edit_inputs(tmp_path, (5, 5), ([], []))
    targets = [row for row in _plan()["targets"] if row["target_id"] not in ("bowl", "dishwasher")]
    result = completion._verify_completed_background(
        frames=frames, original_frames=frames, plan={"task_context_sha256": "task", "targets": targets},
        output_root=tmp_path / "review", retain_result=False)
    assert result["status"] == "blocked"
    assert sent[0][0] == completion.REVIEW_PROMPT + json.dumps(targets, sort_keys=True)


def test_repair_copies_the_accepted_view_showing_most_objects(tmp_path, monkeypatch):
    selected, sources = [], []
    for index, shown in enumerate((["cup"], ["cup", "bowl"], ["bowl"])):
        path = tmp_path / f"edited-{index}.png"
        Image.new("RGB", (8, 8), "white").save(path)
        record = {"frame_id": f"f{index}", "image_path": str(path), "image_digest": _sha256_file(path),
                  "original_image_path": str(path), "original_image_digest": _sha256_file(path),
                  "removed_task_object_ids": shown}
        sources.append(dict(record, remaining_pixel_count=5))
        selected.append(dict(record, generated_pixels_present=True, generated_pixel_count=64 * (3 - index),
                             remaining_pixel_count=0))
    references = []
    monkeypatch.setattr(repair, "plan_image_repairs", lambda **kw: {
        "plan": {"repairs": [{"frame_id": "f2", "instruction": "Remove the bowl"}], "summary": ""},
        "plan_digest": "sha256:plan"})
    monkeypatch.setattr(completion, "complete_background_images",
                        lambda **kw: references.append(kw["repair_reference_path"]) or [dict(kw["frames"][0])])
    monkeypatch.setattr(completion, "verify_completed_background", lambda **kw: {"status": "passed"})
    repair.repair_rejected_views(selected=selected, object_removal_frames=sources, original_frames=sources,
                                 plan={"targets": [], "task_context_sha256": "task"},
                                 failed_review={"status": "blocked"}, output_root=tmp_path, task_context={})
    # Frame 0 has the largest generated count, but frame 1 shows both removed objects.
    assert references == [Path(selected[1]["image_path"])]
