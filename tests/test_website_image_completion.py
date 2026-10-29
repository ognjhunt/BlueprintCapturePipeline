import json
from io import BytesIO
from pathlib import Path

import numpy as np
from blueprint_pipeline.paid_resource_admission import require_paid_resource_admission



import pytest
from PIL import Image

from blueprint_pipeline import website_image_completion as completion
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.semantic_teacher_image_edit_worker import _multipart, _normalized_usage


def test_preserved_capture_review_keeps_objects_instead_of_requesting_their_absence():
    prompt = completion._review_prompt([], frames=[{'novel_view': True}], context_basis='preserved_capture')
    assert 'No object removal or image inpainting is requested' in prompt
    assert 'Preserve every photographed object' in prompt and 'object-absent' not in prompt
    assert 'not geometry, collision, metric or physical validation' in prompt
    with pytest.raises(ValueError, match='preserved_targets_invalid'):
        completion._review_prompt([{'disposition': 'remove'}], context_basis='preserved_capture')


def _grant(admission):
    return require_paid_resource_admission(admission, resource_class=admission["resource_class"],
                                           expected_schema_version="paid_lane_admission.v1")


def test_read_only_image_adoption_cannot_buy_a_missing_or_partial_batch(tmp_path, monkeypatch):
    frames, admission = _inputs(tmp_path)
    monkeypatch.setattr(completion, "_execute_frame_request", lambda **_: pytest.fail("bought cached image"))
    with pytest.raises(ValueError, match="cached_batch_missing"):
        completion.complete_background_images(frames=frames, task_digest="task", output_root=tmp_path / "out",
                                              admission=admission, token="", require_cached=True)
    root = next((tmp_path / "out").glob("[0-9a-f]" * 64))
    (root / "0.json").write_text(json.dumps({"status": "submitting"}))
    with pytest.raises(ValueError, match="cached_batch_incomplete"):
        completion.complete_background_images(frames=frames, task_digest="task", output_root=tmp_path / "out",
                                              admission=admission, token="", require_cached=True)


def _inputs(tmp_path):
    frames = []
    for index in range(2):
        image, mask = tmp_path / f"source-{index}.png", tmp_path / f"mask-{index}.png"
        Image.new("RGB", (10, 20), "red").save(image)
        pixels = np.zeros((20, 10), dtype=np.uint8)
        pixels[5:10, 3:6] = 255
        Image.fromarray(pixels).save(mask)
        frames.append({"frame_id": str(index), "image_path": str(image), "image_digest": _sha256_file(image),
                       "remaining_mask_path": str(mask), "remaining_mask_digest": _sha256_file(mask),
                       "remaining_pixel_count": 15})
    _backend, _execution, digest = completion._validated_backend(completion.REGISTRY_PATH, backend_id=completion.BACKEND_ID)
    admission = {"schema_version": "paid_lane_admission.v1", "status": "admitted", "blockers": [],
                 "resource_class": "openai_api_candidate", "external_disclosure_allowed": True,
                 "allocation_binding_digest": canonical_digest(completion.completion_binding(frames, task_digest="task", backend_digest=digest)),
                 "maximum_cost_usd": 1.0}
    return frames, admission


@pytest.mark.parametrize("mask_guidance,region_guide", [(False, False), (True, False), (True, True)])
def test_completion_sends_plain_frame_keeps_whole_edit_references_first_edit_and_reuses_paid_outputs(tmp_path, monkeypatch, mask_guidance, region_guide):
    frames, admission = _inputs(tmp_path)
    if region_guide:
        for frame in frames:
            frame.update(mask_guide_basis='agent_grounded_region', mask_guide_evidence_digest='retained-evidence')
    if mask_guidance:
        backend = completion._validated_backend(completion.REGISTRY_PATH, backend_id=completion.BACKEND_ID)[2]
        admission["allocation_binding_digest"] = canonical_digest(completion.completion_binding(
            frames, task_digest="task", backend_digest=backend, mask_guidance=True))
    calls = []

    def edit(**kwargs):
        calls.append(kwargs)
        stream = BytesIO()
        Image.new("RGB", kwargs["expected_size"], "blue").save(stream, format="PNG")
        return {"succeeded": True, "generated": stream.getvalue(), "usage": _normalized_usage({"output_tokens_details": {"image_tokens": 100}})}

    monkeypatch.setattr(completion, "_execute_frame_request", edit)
    # An edit whose camera cannot be verified stays whole (see the removal-zone test).
    from blueprint_pipeline import edited_view_camera as camera_check
    monkeypatch.setattr(camera_check, "edited_view_camera", lambda *_: {"status": "unverifiable"})
    args = dict(frames=frames, task_digest="task", output_root=tmp_path / "edits", admission=admission,
                admission_grant=_grant(admission), token="test", mask_guidance=mask_guidance)
    outputs = completion.complete_background_images(**args)
    assert len(calls) == 2
    assert calls[0]["reference_images"][0] == Path(frames[1]["image_path"]).read_bytes()
    assert calls[1]["reference_images"][0] == Path(outputs[0]["image_path"]).read_bytes()
    assert all(len(call["reference_images"]) == (2 if mask_guidance else 1) for call in calls)
    if mask_guidance:
        assert ("source-grounded region target-guide" if region_guide else "SAM target-guide overlay") in calls[0]["prompt"]
        assert calls[0]["reference_images"][-1] == completion._sam_guide(
            Image.open(frames[0]["image_path"]).convert("RGB"), Image.open(frames[0]["remaining_mask_path"]).convert("L"))
        receipt = json.loads(Path(outputs[0]["completion_receipt"]).read_text())
        assert receipt["target_guidance"] == ("source_grounded_region_overlay_v1" if region_guide else "sam_overlay_v1")
        assert _sha256_file(Path(receipt["guide_image_path"])) == receipt["guide_image_digest"]
    # The first image is plain; SAM guidance is a separate reference. The API
    # mask field and silhouette paste-back remain unused.
    assert all(call["mask_bytes"] is None for call in calls)
    sent = np.asarray(Image.open(BytesIO(calls[0]["image_bytes"])).convert("RGB"))
    assert sent.shape == (1536, 768, 3) and np.all(sent == [255, 0, 0])
    assert "shadows" in calls[0]["prompt"] and "Change nothing else" in calls[0]["prompt"]
    # Contents of a removed object must go with it rather than float in mid-air.
    assert "anything inside them or resting on them" in calls[0]["prompt"]
    for original, output in zip(frames, outputs):
        image = np.asarray(Image.open(output["image_path"]))
        # An unverifiable camera keeps the whole edited frame: no paste-back.
        assert np.all(image == [0, 0, 255])
        assert _sha256_file(Path(original["image_path"])) == original["image_digest"]
        assert output["generated_pixel_count"] == 10 * 20
        assert output["generated_region"] == "full_frame"
        assert output["view_consistency"] == "requires_review"
    assert completion.complete_background_images(**args) == outputs
    assert len(calls) == 2
    if not mask_guidance:
        assert completion.complete_background_images(**{**args, "mask_guidance": True}) == outputs
        path = Path(outputs[0]["completion_receipt"])
        receipt = json.loads(path.read_text())
        receipt["status"] = "submitting"
        path.write_text(json.dumps(receipt))
        with pytest.raises(ValueError, match="requires_reconciliation"):
            completion.complete_background_images(**{**args, "mask_guidance": True})
        assert len(calls) == 2


def _admission_for(frames, backend_id):
    _, _, digest = completion._validated_backend(completion.REGISTRY_PATH, backend_id=backend_id)
    return {"schema_version": "paid_lane_admission.v1", "status": "admitted", "blockers": [],
            "resource_class": "openai_api_candidate", "external_disclosure_allowed": True,
            "allocation_binding_digest": canonical_digest(completion.completion_binding(
                frames, task_digest="task", backend_digest=digest)), "maximum_cost_usd": 1.0}


def _blue_edit(calls):
    def edit(**kwargs):
        calls.append(kwargs)
        stream = BytesIO()
        Image.new("RGB", kwargs["expected_size"], "blue").save(stream, format="PNG")
        return {"succeeded": True, "generated": stream.getvalue(),
                "usage": _normalized_usage({"output_tokens_details": {"image_tokens": 100}})}
    return edit


def test_website_edits_use_xhigh_and_a_finished_high_batch_is_reused_without_spending(tmp_path, monkeypatch):
    frames, _ = _inputs(tmp_path)
    legacy_id = completion.LEGACY_BACKEND_IDS[-1]
    current_id = completion.BACKEND_ID
    registry = completion._validated_backend(completion.REGISTRY_PATH, backend_id=completion.BACKEND_ID)[1]
    assert registry["default_options"] == {"output_format": "png", "quality": "xhigh"}
    assert registry["website_canvas_policy"] == "native_aspect_v1"
    assert registry["model_snapshot"] == "gpt-image-2.5-sunburst-2026-09-08"
    calls = []
    monkeypatch.setattr(completion, "_execute_frame_request", _blue_edit(calls))
    # A batch the earlier high-quality row finished...
    monkeypatch.setattr(completion, "BACKEND_ID", legacy_id)
    legacy_admission = _admission_for(frames, legacy_id)
    finished = completion.complete_background_images(
        frames=frames, task_digest="task", output_root=tmp_path / "edits", admission=legacy_admission,
        admission_grant=_grant(legacy_admission), token="test")
    assert [call["execution"]["default_options"]["quality"] for call in calls] == ["high", "high"]
    monkeypatch.undo()
    # ...is read back after the switch with no spend authority and no provider call.
    monkeypatch.setattr(completion, "_execute_frame_request", lambda **_: pytest.fail("must not spend"))
    assert completion.complete_background_images(frames=frames, task_digest="task", output_root=tmp_path / "edits",
                                                 admission={}, token="") == finished
    # A partial high batch is never finished at the old setting: new edits are xhigh.
    other = tmp_path / "partial"
    monkeypatch.setattr(completion, "_execute_frame_request", _blue_edit(calls))
    monkeypatch.setattr(completion, "BACKEND_ID", legacy_id)
    completion.complete_background_images(frames=frames, task_digest="task", output_root=other,
                                          admission=legacy_admission, admission_grant=_grant(legacy_admission),
                                          token="test")
    next(other.glob("*/1.json")).unlink()
    monkeypatch.setattr(completion, "BACKEND_ID", current_id)
    calls.clear()
    admission = _admission_for(frames, completion.BACKEND_ID)
    completion.complete_background_images(frames=frames, task_digest="task", output_root=other,
                                          admission=admission, admission_grant=_grant(admission), token="test")
    assert [call["execution"]["default_options"]["quality"] for call in calls] == ["xhigh", "xhigh"]


@pytest.mark.parametrize("size,expected", [
    ((720, 1280), (864, 1536)), ((1080, 1920), (864, 1536)),
    ((1280, 720), (1536, 864)), ((640, 480), (1360, 1024)), ((900, 900), (1024, 1024)),
])
def test_custom_canvas_keeps_all_source_corners_without_a_letterbox(size, expected):
    source = Image.new("RGB", size, "gray")
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for (x, y), color in zip([(0, 0), (size[0] - 50, 0), (0, size[1] - 50),
                              (size[0] - 50, size[1] - 50)], colors):
        source.paste(color, (x, y, x + 50, y + 50))
    canvas, box = completion._canvas(source)
    assert canvas.size == expected and box == (0, 0, *expected)
    assert all(value % 16 == 0 for value in expected)
    assert 655_360 <= canvas.width * canvas.height <= 8_294_400
    assert canvas.width / canvas.height == pytest.approx(size[0] / size[1], rel=.01)
    assert [canvas.getpixel(p) for p in [(0, 0), (canvas.width - 1, 0),
        (0, canvas.height - 1), (canvas.width - 1, canvas.height - 1)]] == colors


@pytest.mark.parametrize("uncertain", [False, True])
def test_camera_repair_uses_current_canvas_but_never_bypasses_uncertain_legacy_edit(tmp_path, monkeypatch, uncertain):
    frames, _ = _inputs(tmp_path)
    current_id = completion.BACKEND_ID
    legacy_id = completion.LEGACY_BACKEND_IDS[0]
    calls = []
    monkeypatch.setattr(completion, "_execute_frame_request", _blue_edit(calls))
    monkeypatch.setattr(completion, "BACKEND_ID", legacy_id)
    old_admission = _admission_for(frames, legacy_id)
    kwargs = dict(frames=frames, task_digest="task", output_root=tmp_path / "out", token="test")
    old = completion.complete_background_images(**kwargs, admission=old_admission, admission_grant=_grant(old_admission))
    assert [call["expected_size"] for call in calls] == [(1024, 1536)] * 2
    monkeypatch.setattr(completion, "BACKEND_ID", current_id)
    if uncertain:
        path = Path(old[0]["completion_receipt"])
        receipt = json.loads(path.read_text())
        receipt["status"] = "submitting"
        path.write_text(json.dumps(receipt))
        with pytest.raises(ValueError, match="requires_reconciliation"):
            completion.complete_background_images(**kwargs, admission={}, require_original_camera=True)
        assert len(calls) == 2
    else:
        # Ordinary adoption retains the exact old bytes. A framing correction
        # explicitly requires the new canvas rather than reusing its failed one.
        assert completion.complete_background_images(**kwargs, admission={}) == old
        admission = _admission_for(frames, current_id)
        new = completion.complete_background_images(**kwargs, admission=admission,
            admission_grant=_grant(admission), require_original_camera=True)
        assert len(calls) == 4 and [c["expected_size"] for c in calls[2:]] == [(768, 1536)] * 2
        assert new[0]["completion_receipt"] != old[0]["completion_receipt"]
        assert completion.complete_background_images(**kwargs, admission={}, require_original_camera=True) == new
        assert len(calls) == 4


def test_unsupported_source_aspect_refuses_before_provider_request(tmp_path, monkeypatch):
    frames, _ = _inputs(tmp_path)
    frame = frames[0]
    Image.new("RGB", (1600, 100), "red").save(frame["image_path"])
    frame["image_digest"] = _sha256_file(Path(frame["image_path"]))
    admission = _admission_for(frames, completion.BACKEND_ID)
    monkeypatch.setattr(completion, "_execute_frame_request", lambda **_: pytest.fail("invalid canvas reached provider"))
    with pytest.raises(ValueError, match="source_aspect_unsupported"):
        completion.complete_background_images(frames=frames, task_digest="task", output_root=tmp_path / "out",
            admission=admission, admission_grant=_grant(admission), token="test")


def test_largest_removal_view_is_the_edited_anchor_for_every_other_view(tmp_path, monkeypatch):
    frames = []
    for index, count in enumerate((5, 20, 10)):
        image, mask = tmp_path / f"source-{index}.png", tmp_path / f"mask-{index}.png"
        Image.new("RGB", (10, 20), ("red", "green", "blue")[index]).save(image)
        pixels = np.zeros((20, 10), dtype=np.uint8)
        pixels.reshape(-1)[:count] = 255
        Image.fromarray(pixels).save(mask)
        frames.append({"frame_id": str(index), "image_path": str(image), "image_digest": _sha256_file(image),
                       "remaining_mask_path": str(mask), "remaining_mask_digest": _sha256_file(mask),
                       "remaining_pixel_count": count})
    _, _, digest = completion._validated_backend(completion.REGISTRY_PATH, backend_id=completion.BACKEND_ID)
    admission = {"schema_version": "paid_lane_admission.v1", "status": "admitted", "blockers": [],
                 "resource_class": "openai_api_candidate", "external_disclosure_allowed": True,
                 "allocation_binding_digest": canonical_digest(completion.completion_binding(
                     frames, task_digest="task", backend_digest=digest)), "maximum_cost_usd": 1.0}
    calls = []

    def edit(**kwargs):
        calls.append(kwargs)
        stream = BytesIO()
        Image.new("RGB", kwargs["expected_size"], ("white", "black", "gray")[len(calls) - 1]).save(stream, format="PNG")
        return {"succeeded": True, "generated": stream.getvalue(),
                "usage": _normalized_usage({"output_tokens_details": {"image_tokens": 100}})}

    monkeypatch.setattr(completion, "_execute_frame_request", edit)
    outputs = completion.complete_background_images(frames=frames, task_digest="task", output_root=tmp_path / "edits",
                                                    admission=admission, admission_grant=_grant(admission), token="test")
    # Frame 1 shows the most of the object: it is edited first, against an original view.
    first = np.asarray(Image.open(BytesIO(calls[0]["image_bytes"])).convert("RGB"))
    assert np.all(first[first.any(axis=-1)] == [0, 128, 0])
    assert calls[0]["reference_images"] == [Path(frames[0]["image_path"]).read_bytes()]
    # Every other view references that edited anchor, never the previous edit.
    anchor = Path(outputs[1]["image_path"]).read_bytes()
    assert [call["reference_images"] for call in calls[1:]] == [[anchor], [anchor]]
    assert [output["frame_id"] for output in outputs] == ["0", "1", "2"]
    # The rules name the open bay for built-in objects, in the edit and in the review.
    assert "leave the empty bay it occupied open to its full depth" in calls[0]["prompt"]
    assert "never fill that space with cabinets, doors, drawers, panels" in calls[0]["prompt"]
    assert "not new cabinets, doors, drawers or panels" in completion.REVIEW_PROMPT
    # Objects next to the removed one stay, and a covered rug or mat continues underneath.
    assert "even where they touch or sit next to the removed object" in calls[0]["prompt"]
    assert "continue it underneath" in calls[0]["prompt"]
    # People and body parts go with the task objects, in the edit and in the review.
    assert "remove every person and every part of a person" in calls[0]["prompt"]
    assert "Treat any person or part of a person" in completion.REVIEW_PROMPT
    assert "not a reason to reject" not in completion.REVIEW_PROMPT


@pytest.mark.parametrize("fault", ["binding", "rights", "budget", "changed_source"])
def test_completion_holds_before_provider_mutation(tmp_path, monkeypatch, fault):
    frames, admission = _inputs(tmp_path)
    if fault == "binding":
        admission["allocation_binding_digest"] = "other"
    elif fault == "rights":
        admission["external_disclosure_allowed"] = False
    elif fault == "budget":
        admission["maximum_cost_usd"] = 0.01
    else:
        Image.new("RGB", (10, 20), "green").save(frames[0]["image_path"])
    monkeypatch.setattr(completion, "_execute_frame_request", lambda **_: pytest.fail("must not spend"))
    with pytest.raises((ValueError, RuntimeError)):
        completion.complete_background_images(frames=frames, task_digest="task", output_root=tmp_path / "edits", admission=admission, admission_grant=_grant(admission), token="test")


def test_uncertain_provider_response_is_not_purchased_again(tmp_path, monkeypatch):
    frames, admission = _inputs(tmp_path)
    calls = []

    def fail(**_kwargs):
        calls.append(1)
        return {"succeeded": False, "blocker": "timeout"}

    monkeypatch.setattr(completion, "_execute_frame_request", fail)
    args = dict(frames=frames, task_digest="task", output_root=tmp_path / "edits", admission=admission, admission_grant=_grant(admission), token="test")
    with pytest.raises(ValueError, match="timeout"):
        completion.complete_background_images(**args)
    with pytest.raises(ValueError, match="requires_reconciliation"):
        completion.complete_background_images(**args)
    assert len(calls) == 1
    receipt = next((tmp_path / "edits").rglob("0.json"))
    assert json.loads(receipt.read_text())["status"] == "submitting"


def test_reference_multipart_keeps_mask_on_first_image():
    body = _multipart(fields={"prompt": "test"}, image_bytes=b"FIRST", mask_bytes=b"MASK", boundary="test", reference_images=[b"REFERENCE"])
    assert body.count(b'name="image[]"') == 2
    assert body.index(b"FIRST") < body.index(b"REFERENCE") < body.index(b"MASK")


def test_multipart_without_mask_sends_no_mask_part():
    body = _multipart(fields={"prompt": "test"}, image_bytes=b"FIRST", mask_bytes=None, boundary="test",
                      reference_images=[b"REFERENCE"])
    assert b'name="mask"' not in body
    assert body.index(b"FIRST") < body.index(b"REFERENCE")


def test_admission_dictionary_cannot_authorize_image_spend(tmp_path, monkeypatch):
    frames, admission = _inputs(tmp_path)
    monkeypatch.setattr(completion, "_execute_frame_request", lambda **_: pytest.fail("must not spend"))
    with pytest.raises(RuntimeError, match="grant_missing"):
        completion.complete_background_images(frames=frames, task_digest="task", output_root=tmp_path / "edits",
                                              admission=admission, token="test")


@pytest.mark.parametrize("website", [False, True])
@pytest.mark.parametrize("novel", [False, True])
def test_background_review_keeps_client_alive_and_prices_all_novel_capture_context(tmp_path, monkeypatch, website, novel):
    import sys
    from types import SimpleNamespace as NS
    import google

    state = {"closed": False, "calls": 0}
    requests = []
    def generate(**kwargs):
        assert not state["closed"]
        state["calls"] += 1
        requests.append(kwargs)
        return NS(candidates=[NS(finish_reason="STOP")], text=json.dumps({
            "consistent_background": True, "task_objects_removed": True,
            "people_absent": False, "unrelated_objects_preserved": True,
            "unrelated_object_loss_frame_ids": [],
            "remaining_task_object_frame_ids": []}))
    class Client:
        def __init__(self, **kwargs):
            self.models = NS(generate_content=generate)
        def __enter__(self):
            return self
        def __exit__(self, *_):
            state["closed"] = True
        def __del__(self):
            state["closed"] = True

    types = NS(Part=NS(from_bytes=lambda **kw: kw), GenerateContentConfig=NS,
               HttpOptions=NS, HttpRetryOptions=NS)
    fake = NS(Client=Client, types=types)
    monkeypatch.setattr(google, "genai", fake, raising=False)
    monkeypatch.setitem(sys.modules, "google.genai", fake)
    monkeypatch.setattr(completion, "_api_key", lambda: ("fixture-key", "fixture"))
    frames, _ = _inputs(tmp_path)
    if novel:
        frames = [{**frame, "novel_view": True} for frame in frames]
    raw_path = tmp_path / "unselected-capture.png"
    Image.new("RGB", (20, 10), "blue").save(raw_path)
    raw = {"frame_id": "unselected-capture", "image_path": str(raw_path), "image_digest": _sha256_file(raw_path)}
    args = dict(frames=frames, original_frames=[*frames, raw], plan={"task_context_sha256": "task", "targets": []},
                output_root=tmp_path / "review")
    reservations = []
    quotes = []
    if website:
        from blueprint_pipeline import website_gemini_receipts as receipts
        from blueprint_pipeline.decision_evidence_contracts import canonical_digest
        from hashlib import sha256
        task = {"schema_version": "website_site_task_context.v1", "request_id": "req", "scene_id": "scene",
                "capture_id": "capture", "confirmed": True, "confirmed_at": "2026-09-19", "description": "Move box"}
        task["context_digest"] = canonical_digest(task, digest_field="context_digest")
        args["task_context"] = task
        args["plan"]["task_context_sha256"] = sha256(json.dumps(task, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        def reserve(**kwargs):
            reservations.append(kwargs)
            admission = {"schema_version": "paid_lane_admission.v1", "status": "admitted",
                         "resource_class": kwargs["resource_class"], "blockers": [],
                         "allocation_binding_digest": kwargs["binding_digest"]}
            return admission, _grant(admission)
        monkeypatch.setattr(receipts, "reserve_website_preparation_spend", reserve)
        real_quote = receipts.gemini_quote
        def quote(**kwargs):
            quotes.append(kwargs)
            return real_quote(**kwargs)
        monkeypatch.setattr(receipts, "gemini_quote", quote)
    assert completion.verify_completed_background(**args)["status"] == "passed"
    assert state == {"closed": True, "calls": 1}
    images = [item for item in requests[0]["contents"] if isinstance(item, dict)]
    assert len(images) == 4 + int(novel)
    assert ("raw capture context unselected-capture" in requests[0]["contents"]) is novel
    assert any(item["data"] == raw_path.read_bytes() for item in images) is novel
    if website:
        monkeypatch.setattr(completion, "_api_key", lambda: (None, None))
        assert len(reservations) == 1
        assert reservations[0]["provider"] == "google"
        request = json.loads(next((args["output_root"] / "gemini_reviews").glob("*.json")).read_text())["request"]
        assert len(request["binding"]["image_digests"]) == len(images)
        assert request["binding"].get("capture_context_frame_ids", []) == ([raw["frame_id"]] if novel else [])
        assert quotes[0]["input_tokens"] >= 1120 * len(images)
    assert completion.verify_completed_background(**args)["status"] == "passed"
    assert state["calls"] == 1
    if website and novel:
        Image.new("RGB", (20, 10), "green").save(raw_path)
        with pytest.raises(ValueError, match="review_source_changed"):
            completion.verify_completed_background(**args)
        assert len(reservations) == 1


@pytest.mark.parametrize("camera_ids,novel,expected", [
    ([], False, "passed"), (None, False, "blocked"), (["0"], False, "blocked"),
    (["missing"], False, "blocked"), (["0", "0"], False, "blocked"), ([1], False, "blocked"),
    (None, True, "passed"),
])
def test_visual_review_must_resolve_source_camera_changes_and_retains_invalid_answers(
        tmp_path, monkeypatch, camera_ids, novel, expected):
    import sys
    from types import SimpleNamespace as NS

    response = {"consistent_background": True, "task_objects_removed": True,
                "unrelated_objects_preserved": True, "remaining_task_object_frame_ids": [],
                "unrelated_object_loss_frame_ids": [], "camera_changed_frame_ids": camera_ids}
    calls = []
    class Client:
        def __init__(self, **_):
            self.models = self
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return NS(candidates=[NS(finish_reason="STOP")], text=json.dumps(response))
    types = NS(Part=NS(from_bytes=lambda **kw: kw), GenerateContentConfig=NS,
               HttpOptions=NS, HttpRetryOptions=NS)
    monkeypatch.setitem(sys.modules, "google.genai", NS(Client=Client, types=types))
    monkeypatch.setattr(completion, "_api_key", lambda: ("fixture-key", "fixture"))
    frames, _ = _inputs(tmp_path)
    edited = [{**frame, "generated_pixels_present": True, "novel_view": novel} for frame in frames]
    args = dict(frames=edited, original_frames=frames,
                plan={"task_context_sha256": "task", "targets": []}, output_root=tmp_path / "review")
    first = completion.verify_completed_background(**args)
    assert first["status"] == expected
    assert first["review"] == response
    assert (completion.CAMERA_REVIEW_PROMPT in first["binding"]["review_prompt"]) is not novel
    assert completion.verify_completed_background(**args) == first
    assert len(calls) == 1  # A blocked/invalid answer remains a completed paid review.


def test_http_failure_retains_sanitized_provider_evidence_and_never_retries(tmp_path, monkeypatch):
    frames, admission = _inputs(tmp_path)
    calls = []
    failure = {"http_status": 429, "provider_error_code": "rate_limit_exceeded",
               "provider_request_id": "req_fixture", "raw_provider_body_recorded": False}
    def request(**_):
        calls.append(True)
        return {"succeeded": False, "blocker": "semantic_teacher_provider_http_error",
                "provider_failure": failure}
    monkeypatch.setattr(completion, "_execute_frame_request", request)
    args = dict(frames=frames, task_digest="task", output_root=tmp_path / "edits",
                admission=admission, admission_grant=_grant(admission), token="fixture-key")
    with pytest.raises(ValueError, match="semantic_teacher_provider_http_error"):
        completion.complete_background_images(**args)
    receipt = json.loads(next((tmp_path / "edits").glob("*/0.json")).read_text())
    assert receipt["provider_failure"] == failure
    assert receipt["status"] == "submitting" and receipt["requires_reconciliation"] is True
    with pytest.raises(ValueError, match="requires_reconciliation"):
        completion.complete_background_images(**args)
    assert calls == [True]


@pytest.mark.parametrize("diagnosed_ids", [["0"], [], ["0", "1"], ["missing"]])
def test_consistency_diagnosis_is_exactly_one_retained_view(tmp_path, monkeypatch, diagnosed_ids):
    import sys
    from hashlib import sha256
    from types import SimpleNamespace as NS
    import google
    from blueprint_pipeline import website_gemini_receipts as receipts

    frames, _ = _inputs(tmp_path)
    task = {"request_id": "request", "scene_id": "scene", "capture_id": "capture",
            "context_digest": "sha256:task-context"}
    task_digest = sha256(json.dumps(task, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    calls = []
    def retained(**kwargs):
        calls.append(kwargs)
        return kwargs["invoke"]()
    def generate(**kwargs):
        assert kwargs["model"] == completion.DEFAULT_MODEL
        assert len(kwargs["contents"]) == 1 + 4 * len(frames)
        return NS(candidates=[NS(finish_reason="STOP")], text=json.dumps({
            "inconsistent_background_frame_ids": diagnosed_ids,
            "visual_evidence": "one generated panel conflicts with the other view"}))
    class Client:
        def __init__(self, **kwargs):
            self.models = NS(generate_content=generate)
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
    fake = NS(Client=Client, types=NS(Part=NS(from_bytes=lambda **kw: kw),
        GenerateContentConfig=NS, HttpOptions=NS, HttpRetryOptions=NS))
    monkeypatch.setattr(google, "genai", fake, raising=False)
    monkeypatch.setitem(sys.modules, "google.genai", fake)
    monkeypatch.setattr(completion, "_api_key", lambda: ("fixture-key", "fixture"))
    monkeypatch.setattr(receipts, "retained_gemini_call", retained)
    args = dict(frames=frames, original_frames=frames,
        plan={"task_context_sha256": task_digest}, failed_review={"status": "blocked",
            "request_digest": "sha256:prior", "review": {"consistent_background": False,
                "task_objects_removed": True, "people_absent": True,
                "unrelated_objects_preserved": True, "remaining_task_object_frame_ids": []}},
        output_root=tmp_path / "review", task_context=task)
    if diagnosed_ids == ["0"]:
        result = completion.diagnose_inconsistent_background(**args)
        assert result["diagnosis"]["inconsistent_background_frame_ids"] == ["0"]
        assert calls[0]["binding"]["prior_review_digest"] == "sha256:prior"
        assert calls[0]["binding"]["kind"] == "background_consistency_diagnosis"
        assert calls[0]["binding"]["revision"] == 2
        assert calls[0]["binding"]["max_output_tokens"] == 8192
        assert calls[0]["maximum_cost_usd"] > 0
    else:
        with pytest.raises(ValueError, match="diagnosis_invalid"):
            completion.diagnose_inconsistent_background(**args)
    assert len(calls) == 1


@pytest.mark.parametrize("prior_status,expected_revision", [("completed", 1), ("submitting", 2)])
def test_consistency_diagnosis_preserves_or_supersedes_exact_prior_receipt(
        tmp_path, monkeypatch, prior_status, expected_revision):
    from hashlib import sha256
    from blueprint_pipeline import website_gemini_receipts as receipts
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    frames, _ = _inputs(tmp_path)
    task = {"request_id": "request", "scene_id": "scene", "capture_id": "capture",
            "context_digest": "sha256:task-context"}
    image_digests = [digest for frame in frames
                     for digest in (frame["image_digest"], frame["image_digest"])]
    prior_binding = {"kind": "background_consistency_diagnosis", "model": completion.DEFAULT_MODEL,
        "prior_review_digest": "sha256:prior", "image_digests": image_digests,
        "frame_ids": [frame["frame_id"] for frame in frames],
        "prompt": completion.CONSISTENCY_DIAGNOSIS_PROMPT,
        "media_resolution": "MEDIA_RESOLUTION_HIGH", "revision": 1, "max_output_tokens": 1024}
    text_bytes = len(completion.CONSISTENCY_DIAGNOSIS_PROMPT.encode()) + 1024
    text_bytes += sum(len(str(frame["frame_id"]).encode()) * 2 + 32 for frame in frames)
    prior_request = {"binding": prior_binding, "task_context_digest": task["context_digest"],
        "maximum_cost_usd": receipts.gemini_quote(model=completion.DEFAULT_MODEL,
            input_tokens=text_bytes + 1120 * len(image_digests), max_output_tokens=1024)}
    prior_digest = canonical_digest(prior_request)
    review_dir = tmp_path / "review" / "gemini_reviews"
    review_dir.mkdir(parents=True)
    (review_dir / f"{prior_digest[7:]}.json").write_text(json.dumps({
        "status": prior_status, "request_digest": prior_digest, "request": prior_request}))
    seen = []
    def retained(**kwargs):
        seen.append(kwargs)
        return {"status": "completed", "binding": kwargs["binding"],
                "diagnosis": {"inconsistent_background_frame_ids": [frames[0]["frame_id"]],
                              "visual_evidence": "invented panel"}}
    monkeypatch.setattr(receipts, "retained_gemini_call", retained)
    result = completion.diagnose_inconsistent_background(
        frames=frames, original_frames=frames,
        plan={"task_context_sha256": sha256(json.dumps(task, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()},
        failed_review={"status": "blocked", "request_digest": "sha256:prior", "review": {
            "consistent_background": False, "task_objects_removed": True,
            "people_absent": True, "unrelated_objects_preserved": True,
            "remaining_task_object_frame_ids": []}},
        output_root=tmp_path / "review", task_context=task)
    assert result["status"] == "completed"
    assert seen[0]["binding"]["revision"] == expected_revision
    assert seen[0]["binding"]["max_output_tokens"] == (1024 if expected_revision == 1 else 8192)
    if prior_status == "submitting":
        assert seen[0]["binding"]["supersedes_incomplete_request_digest"] == prior_digest


@pytest.mark.parametrize("prior_version,prior_status,expected_revision", [
    (1, "completed", 1), (1, "submitting", 3), (1, None, 3),
    (2, "completed", 2), (2, "submitting", 3)])
def test_background_review_preserves_or_supersedes_exact_prior_receipt(
        tmp_path, monkeypatch, prior_version, prior_status, expected_revision):
    from hashlib import sha256
    from blueprint_pipeline import website_gemini_receipts as receipts
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    frames, _ = _inputs(tmp_path)
    task = {"request_id": "request", "scene_id": "scene", "capture_id": "capture",
            "context_digest": "sha256:task-context"}
    plan = {"task_context_sha256": sha256(json.dumps(task, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "targets": []}
    image_digests = [digest for frame in frames for digest in (frame["image_digest"], frame["image_digest"])]
    prior_binding = {"kind": "background_review", "revision": 1, "model": completion.DEFAULT_MODEL,
        "max_output_tokens": 2048, "image_digests": image_digests, "frame_ids": [frame["frame_id"] for frame in frames],
        "targets": [], "prompt": completion.REVIEW_PROMPT, "media_resolution": "MEDIA_RESOLUTION_HIGH"}
    text_bytes = len((completion.REVIEW_PROMPT + json.dumps([], sort_keys=True)).encode()) + 2048
    text_bytes += sum(len(str(frame["frame_id"]).encode()) * 2 + 32 for frame in frames)
    prior_request = {"binding": prior_binding, "task_context_digest": task["context_digest"],
        "maximum_cost_usd": receipts.gemini_quote(model=completion.DEFAULT_MODEL,
            input_tokens=text_bytes + 1120 * len(image_digests), max_output_tokens=2048)}
    prior_digest = canonical_digest(prior_request)
    if prior_version == 2:
        prior_binding = {**prior_binding, "revision": 2, "timeout_seconds": 300,
                         "supersedes_incomplete_request_digest": None}
        prior_request = {**prior_request, "binding": prior_binding}
        prior_digest = canonical_digest(prior_request)
    if prior_status:
        review_dir = tmp_path / "review" / "gemini_reviews"
        review_dir.mkdir(parents=True)
        (review_dir / f"{prior_digest[7:]}.json").write_text(json.dumps({
            "status": prior_status, "request_digest": prior_digest, "request": prior_request}))
    seen, timeouts, token_limits = [], [], []
    def retained(**kwargs):
        seen.append(kwargs)
        kwargs["invoke"]()
        return {"status": "passed", "binding": kwargs["binding"]}
    monkeypatch.setattr(receipts, "retained_gemini_call", retained)
    monkeypatch.setattr(completion, "_verify_completed_background", lambda **kw:
                        (timeouts.append(kw["timeout_ms"]), token_limits.append(kw["max_output_tokens"])))
    completion.verify_completed_background(frames=frames, original_frames=frames, plan=plan,
                                           output_root=tmp_path / "review", task_context=task)
    binding = seen[0]["binding"]
    assert binding["revision"] == expected_revision
    if expected_revision in {1, 2}:
        # Completed reviews are reused exactly, including their old quote.
        assert binding == prior_binding and timeouts == [120_000 if expected_revision == 1 else 300_000]
        assert token_limits == [2048]
    else:
        # One separately bounded v3 request preserves earlier incomplete calls.
        assert binding["timeout_seconds"] == 300 and timeouts == [300_000]
        assert binding["max_output_tokens"] == 8192 and token_limits == [8192]
        assert binding["supersedes_incomplete_request_digest"] == (prior_digest if prior_status else None)
        assert not prior_status or json.loads(next((tmp_path / "review" / "gemini_reviews").glob("*.json"))
                                              .read_text())["status"] == prior_status


@pytest.mark.parametrize("finish_reason", ["MAX_TOKENS", "STOP"])
def test_review_retains_incomplete_reason_and_accepts_a_guided_single_target_only_after_review(
        tmp_path, monkeypatch, finish_reason):
    from types import SimpleNamespace
    import sys
    frames, _ = _inputs(tmp_path)
    monkeypatch.setattr(completion, "_api_key", lambda: ("non-secret", "test"))

    class Client:
        def __init__(self, **_):
            self.models = self

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def generate_content(self, **kwargs):
            assert kwargs["config"].max_output_tokens == 8192
            return SimpleNamespace(candidates=[SimpleNamespace(finish_reason=finish_reason)], text=json.dumps({
                "consistent_background": True, "task_objects_removed": True, "unrelated_objects_preserved": True,
                "remaining_task_object_frame_ids": [], "unrelated_object_loss_frame_ids": []}))

    types = SimpleNamespace(
        Part=SimpleNamespace(from_bytes=lambda **_: "image"),
        HttpOptions=lambda **kw: SimpleNamespace(**kw), HttpRetryOptions=lambda **kw: SimpleNamespace(**kw),
        GenerateContentConfig=lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setitem(sys.modules, "google.genai", SimpleNamespace(Client=Client, types=types))
    if finish_reason == "STOP":
        selected = [{**frame, "target_guidance": "sam_overlay_v1"} for frame in frames]
        result = completion._verify_completed_background(frames=selected, original_frames=frames,
            plan={"targets": [{"task_effect": "manipulated", "disposition": "remove"}],
                  "task_context_sha256": "task"}, output_root=tmp_path / "review",
            retain_result=False, max_output_tokens=8192)
        assert result["status"] == "passed" and "same camera field of view" in result["binding"]["review_prompt"]
        return
    with pytest.raises(ValueError, match="review_incomplete"):
        completion._verify_completed_background(frames=frames, original_frames=frames,
            plan={"targets": [], "task_context_sha256": "task"}, output_root=tmp_path / "review",
            retain_result=False, max_output_tokens=8192)
    result = json.loads(next((tmp_path / "review").glob("review-incomplete-*.json")).read_text())
    assert result["finish_reason"] == "MAX_TOKENS" and result["approves_views"] is False


@pytest.mark.parametrize("source_digest", [None, "sha256:" + "f" * 64])
def test_controller_reserves_image_edits_and_restart_reuses_outputs_without_new_grant(tmp_path, monkeypatch, source_digest):
    from blueprint_pipeline import website_task_context as control
    frames, admission = _inputs(tmp_path)
    reservations, calls = [], []

    def reserve(**kwargs):
        reservations.append(kwargs)
        return admission, _grant(admission)

    def edit(**kwargs):
        calls.append(kwargs)
        stream = BytesIO()
        Image.new("RGB", kwargs["expected_size"], "blue").save(stream, format="PNG")
        return {"succeeded": True, "generated": stream.getvalue(),
                "usage": _normalized_usage({"output_tokens_details": {"image_tokens": 100}})}

    settlements = []

    def webapp(**kwargs):
        assert kwargs["operation"] == "preparation-settlement"
        settlements.append(kwargs["payload"]["settlement"])
        return {**kwargs["payload"]["settlement"], "status": "settled"}

    monkeypatch.setattr(control, "reserve_website_preparation_spend", reserve)
    monkeypatch.setattr(control, "website_webapp_request", webapp)
    monkeypatch.setattr(completion, "_execute_frame_request", edit)
    context = {"context_digest": "task", "capture_id": "cap", "request_id": "req", "scene_id": "scene"}
    task_digest = completion.sha256(json.dumps(context, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    _, _, backend_digest = completion._validated_backend(completion.REGISTRY_PATH, backend_id=completion.BACKEND_ID)
    binding = completion.completion_binding(frames, task_digest=task_digest, backend_digest=backend_digest)
    if source_digest:
        binding["source_video_digest"] = source_digest
    admission["allocation_binding_digest"] = canonical_digest(binding)
    args = dict(frames=frames, task_digest=task_digest, task_context=context,
                output_root=tmp_path / "edits", admission={}, token="test", source_video_digest=source_digest)
    outputs = completion.complete_background_images(**args)
    assert len(reservations) == 1 and len(calls) == 2
    # The completed batch releases its quote down to the receipted charge, once.
    assert len(settlements) == 1
    assert settlements[0]["allocation_binding_digest"] == admission["allocation_binding_digest"]
    assert settlements[0]["provider"] == "openai" and settlements[0]["completed_request_count"] == 2
    receipts = sorted((tmp_path / "edits").rglob("[0-9].json"))
    assert settlements[0]["provider_charge_amount_usd"] == round(sum(json.loads(p.read_text())["cost_usd"] for p in receipts), 6)
    assert reservations[0]["provider"] == "openai"
    assert reservations[0]["binding_digest"] == admission["allocation_binding_digest"]
    assert reservations[0]["request_count"] == 2
    assert reservations[0]["source_video_digest"] == source_digest
    # No key or admission survives the controller restart. Reuse needs neither.
    assert completion.complete_background_images(**{**args, "token": ""}) == outputs
    assert len(reservations) == 1 and len(calls) == 2 and len(settlements) == 1
    Image.new("RGB", (10, 20), "green").save(outputs[0]["image_path"])
    with pytest.raises(ValueError, match="output_changed"):
        completion.complete_background_images(**args)
    assert len(reservations) == 1 and len(calls) == 2


def test_prior_completed_batch_is_settled_before_a_new_reservation(tmp_path, monkeypatch):
    from blueprint_pipeline import website_task_context as control
    output_root = tmp_path / "edits"
    done, partial = output_root / ("a" * 64), output_root / ("b" * 64)
    done.mkdir(parents=True)
    partial.mkdir()
    for index, cost in enumerate((0.07, 0.06)):
        (done / f"{index}.json").write_text(json.dumps({"status": "completed", "request_digest": "sha256:" + "a" * 64,
                                                        "cost_usd": cost}))
    (partial / "0.json").write_text(json.dumps({"status": "completed", "request_digest": "sha256:" + "b" * 64,
                                                "cost_usd": 0.07}))
    (partial / "1.json").write_text(json.dumps({"status": "submitting", "request_digest": "sha256:" + "b" * 64}))
    (output_root / "gemini_reviews").mkdir()
    order = []

    def webapp(**kwargs):
        order.append(("settle", kwargs["payload"]["settlement"]))
        return {**kwargs["payload"]["settlement"], "status": "settled"}

    frames, admission = _inputs(tmp_path)
    context = {"context_digest": "task", "capture_id": "cap", "request_id": "req", "scene_id": "scene"}
    task_digest = completion.sha256(json.dumps(context, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    _, _, backend_digest = completion._validated_backend(completion.REGISTRY_PATH, backend_id=completion.BACKEND_ID)
    admission["allocation_binding_digest"] = canonical_digest(completion.completion_binding(
        frames, task_digest=task_digest, backend_digest=backend_digest))

    def reserve(**kwargs):
        order.append(("reserve", kwargs["binding_digest"]))
        return admission, _grant(admission)

    def edit(**kwargs):
        stream = BytesIO()
        Image.new("RGB", kwargs["expected_size"], "blue").save(stream, format="PNG")
        return {"succeeded": True, "generated": stream.getvalue(),
                "usage": _normalized_usage({"output_tokens_details": {"image_tokens": 100}})}

    monkeypatch.setattr(control, "reserve_website_preparation_spend", reserve)
    monkeypatch.setattr(control, "website_webapp_request", webapp)
    monkeypatch.setattr(completion, "_execute_frame_request", edit)
    completion.complete_background_images(frames=frames, task_digest=task_digest, task_context=context,
                                          output_root=output_root, admission={}, token="test")
    # The earlier finished batch is released before the new quote is reserved; the
    # batch with an uncertain request keeps its full hold.
    assert order[0][0] == "settle" and order[0][1]["allocation_binding_digest"] == "sha256:" + "a" * 64
    assert order[0][1]["completed_request_count"] == 2 and order[0][1]["provider_charge_amount_usd"] == 0.13
    assert order[1] == ("reserve", admission["allocation_binding_digest"])
    settled = {row[1]["allocation_binding_digest"] for row in order if row[0] == "settle"}
    assert "sha256:" + "b" * 64 not in settled
    assert json.loads((done / "settlement.json").read_text())["status"] == "settled"
    assert [row[0] for row in order].count("settle") == 2  # the old batch once, the new batch once


def test_another_controller_cannot_rebuy_reserved_image_work(tmp_path, monkeypatch):
    from blueprint_pipeline import website_task_context as control
    frames, _ = _inputs(tmp_path)
    def reserved(**kwargs):
        return _grant({"schema_version": "paid_lane_admission.v1", "status": "already_reserved",
                       "resource_class": "openai_api_candidate", "blockers": []})
    monkeypatch.setattr(control, "reserve_website_preparation_spend", reserved)
    monkeypatch.setattr(completion, "_execute_frame_request", lambda **_: pytest.fail("duplicate spend"))
    context = {"context_digest": "task"}
    task_digest = completion.sha256(json.dumps(context, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with pytest.raises(RuntimeError):
        completion.complete_background_images(frames=frames, task_digest=task_digest,
            task_context=context, output_root=tmp_path / "edits", admission={}, token="test")


def test_only_a_scene_with_transients_extends_the_bound_edit_and_review_prompts():
    from blueprint_pipeline import website_image_completion as completion

    still = [{"target_id": "obj", "semantic_label": "dishwasher", "task_effect": "manipulated",
              "disposition": "remove"}]
    working = [*still, {"target_id": "transient-0", "semantic_label": "person", "task_effect": "transient",
                        "disposition": "remove"}]
    assert completion.TRANSIENT_PROMPT not in completion._completion_prompt(still)
    assert completion.TRANSIENT_REVIEW_PROMPT not in completion._review_prompt(still)
    assert completion._review_prompt(still) == completion.REVIEW_PROMPT
    edit = completion._completion_prompt(working)
    assert completion.TRANSIENT_PROMPT in edit and '"task_effect": "transient"' in edit
    review = completion._review_prompt(working)
    assert completion.TRANSIENT_REVIEW_PROMPT in review and review.endswith("Targets: ")
    # A person is not one of several task objects; the single-object prompts stay.
    assert completion.MULTI_OBJECT_PROMPT not in edit


def test_removal_zone_reaches_one_object_size_beyond_the_mask_with_a_soft_edge():
    editable = np.zeros((400, 600), dtype=bool)
    editable[180:220, 280:320] = True                      # 40 x 40: reach 40 px
    alpha = completion.removal_zone_alpha(editable)
    row = alpha[200]
    assert row[300] == 1 and row[320 + 39] == 1            # mask and its reach
    assert 0 < row[320 + 52] < 1                           # the 24 px soft edge
    assert row[320 + 40 + 24] == 0 and row[0] == 0         # photographed pixels beyond
    tiny = np.zeros((400, 600), dtype=bool)
    tiny[200, 300] = True
    assert completion.removal_zone_alpha(tiny)[200, 300 + 32] == 1      # at least 32 px
    assert completion.removal_zone_alpha(np.ones((400, 600), dtype=bool)).min() == 1


def _zone_inputs(tmp_path):
    image, mask = tmp_path / "source.png", tmp_path / "mask.png"
    Image.new("RGB", (240, 160), "red").save(image)
    pixels = np.zeros((160, 240), dtype=np.uint8)
    pixels[70:90, 110:130] = 255
    Image.fromarray(pixels).save(mask)
    frames = [{"frame_id": "0", "image_path": str(image), "image_digest": _sha256_file(image),
               "remaining_mask_path": str(mask), "remaining_mask_digest": _sha256_file(mask),
               "remaining_pixel_count": 400}]
    digest = completion._validated_backend(completion.REGISTRY_PATH, backend_id=completion.BACKEND_ID)[2]
    admission = {"schema_version": "paid_lane_admission.v1", "status": "admitted", "blockers": [],
                 "resource_class": "openai_api_candidate", "external_disclosure_allowed": True,
                 "allocation_binding_digest": canonical_digest(completion.completion_binding(
                     frames, task_digest="task", backend_digest=digest)),
                 "maximum_cost_usd": 1.0}
    return frames, admission


@pytest.mark.parametrize("camera", ["kept", "moved"])
def test_edit_keeps_editor_pixels_only_in_the_removal_zone_when_its_camera_held(tmp_path, monkeypatch, camera):
    # 2026-10-29 capture-4b60431e: the full-frame edit removed the tray and also
    # erased an unrelated object. Photographed pixels now stay outside the zone.
    from blueprint_pipeline import edited_view_camera as camera_check
    frames, admission = _zone_inputs(tmp_path)

    def edit(**kwargs):
        stream = BytesIO()
        Image.new("RGB", kwargs["expected_size"], "blue").save(stream, format="PNG")
        return {"succeeded": True, "generated": stream.getvalue(),
                "usage": _normalized_usage({"output_tokens_details": {"image_tokens": 100}})}

    checked = []
    monkeypatch.setattr(completion, "_execute_frame_request", edit)
    monkeypatch.setattr(camera_check, "edited_view_camera",
                        lambda *paths: checked.append(paths) or {"status": camera})
    args = dict(frames=frames, task_digest="task", output_root=tmp_path / "edits", admission=admission,
                admission_grant=_grant(admission), token="test")
    [output] = completion.complete_background_images(**args)
    image = np.asarray(Image.open(output["image_path"]).convert("RGB"))
    receipt = json.loads(Path(output["completion_receipt"]).read_text())
    full_frame = Path(output["completion_receipt"]).with_name("0-full-frame.png")
    # The camera check sees the editor's raw frame, never the composite.
    assert checked == [(Path(frames[0]["image_path"]), full_frame, Path(frames[0]["remaining_mask_path"]))]
    assert np.all(np.asarray(Image.open(full_frame).convert("RGB")) == [0, 0, 255])
    assert receipt["full_frame_image_digest"] == _sha256_file(full_frame) and receipt["edited_view_camera"] == camera
    assert "removal_zone_composite_v1" == completion.completion_binding(
        frames, task_digest="task", backend_digest=receipt["backend_digest"])["output_policy"]
    if camera == "kept":
        assert np.all(image[70:90, 110:130] == [0, 0, 255])        # the removal itself
        assert np.all(image[:10] == [255, 0, 0]) and np.all(image[:, :40] == [255, 0, 0])
        zone = completion.removal_zone_alpha(np.asarray(Image.open(frames[0]["remaining_mask_path"])) == 255)
        assert output["generated_region"] == receipt["generated_region"] == "removal_zone"
        assert output["generated_pixel_count"] == int(np.count_nonzero(zone)) < 240 * 160
        assert output["full_frame_image_path"] == str(full_frame)
        assert output["full_frame_image_digest"] == receipt["full_frame_image_digest"]
    else:
        # A reframed edit stays whole so the camera review still re-edits it.
        assert np.all(image == [0, 0, 255])
        assert output["generated_region"] == "full_frame" and "full_frame_image_path" not in output
        assert output["generated_pixel_count"] == 240 * 160
    assert completion.complete_background_images(**args) == [output]
