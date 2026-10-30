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

from blueprint_pipeline import website_image_completion as completion
from blueprint_pipeline import website_image_repair_agent as repair
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


# Retained v1 paid edits and unchanged role-based prompts keep their byte pins
# when fresh edits adopt the explicitly versioned v2 cleanup policy.
SINGLE_BINDING_DIGEST = "sha256:245a78a0958bbeded2208c0c50d67cd70382c44e2b551ca1f8521d032c780193"
SINGLE_REPAIR_BINDING_DIGEST = "sha256:914d6283367ff4485d9a870a694e83d36e842cb7be0a5ff0ad4dd69d1f486191"
REVIEW_PROMPT_SHA256 = "5ff92c802eeb52156f7220512861e255435f9e77d3d0d08697507a8f92103b20"
PROMPT_SHA256 = "b097d4151e62cf690386c603cd534d53a2c71177d123a200b94607265e2fef48"


def test_single_object_prompts_and_retained_v1_requests_remain_byte_identical(tmp_path):
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
        frames, task_digest="task", backend_digest="backend", targets=targets,
        output_policy=completion.LEGACY_REMOVAL_ZONE_BINDING)) == SINGLE_BINDING_DIGEST
    assert canonical_digest(completion.completion_binding(
        frames, task_digest="task", backend_digest="backend", targets=targets, repair_instruction="Keep the mat",
        reference_digest="sha256:ref", output_policy=completion.LEGACY_REMOVAL_ZONE_BINDING)) == SINGLE_REPAIR_BINDING_DIGEST
    assert canonical_digest(completion.completion_binding(
        frames, task_digest="task", backend_digest="backend", targets=targets)) == (
            "sha256:d397275b97f4281f834379faf0ea7ff86229f47cd31da8551667cb8404ec5dc0")
    assert hashlib.sha256(completion.PROMPT.encode()).hexdigest() == PROMPT_SHA256
    assert hashlib.sha256(completion.REVIEW_PROMPT.encode()).hexdigest() == REVIEW_PROMPT_SHA256
    assert completion._review_prompt(targets) is completion.REVIEW_PROMPT


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
    result = completion._verify_completed_background(**args)
    if error:
        # Retained as a blocked result, never a raise: the paid answer is kept and
        # the receipt never becomes uncertain.
        assert result["status"] == "blocked" and result["blocker"] == "website_image_completion_" + error
        return
    assert result["status"] == "blocked" and "blocker" not in result
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
        mask_path = tmp_path / f"mask-{index}.png"
        mask = np.zeros((8, 8), np.uint8)
        mask.flat[:5] = 255
        Image.fromarray(mask).save(mask_path)
        record = {"frame_id": f"f{index}", "image_path": str(path), "image_digest": _sha256_file(path),
                  "original_image_path": str(path), "original_image_digest": _sha256_file(path),
                  "remaining_mask_path": str(mask_path), "remaining_mask_digest": _sha256_file(mask_path),
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
