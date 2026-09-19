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


def _grant(admission):
    return require_paid_resource_admission(admission, resource_class=admission["resource_class"],
                                           expected_schema_version="paid_lane_admission.v1")


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


def test_completion_preserves_unmasked_pixels_references_first_edit_and_reuses_paid_outputs(tmp_path, monkeypatch):
    frames, admission = _inputs(tmp_path)
    calls = []

    def edit(**kwargs):
        calls.append(kwargs)
        stream = BytesIO()
        Image.new("RGB", kwargs["expected_size"], "blue").save(stream, format="PNG")
        return {"succeeded": True, "generated": stream.getvalue(), "usage": _normalized_usage({"output_tokens_details": {"image_tokens": 100}})}

    monkeypatch.setattr(completion, "_execute_frame_request", edit)
    args = dict(frames=frames, task_digest="task", output_root=tmp_path / "edits", admission=admission, admission_grant=_grant(admission), token="test")
    outputs = completion.complete_background_images(**args)
    assert len(calls) == 2
    assert calls[0]["reference_images"] == [Path(frames[1]["image_path"]).read_bytes()]
    assert calls[1]["reference_images"] == [Path(outputs[0]["image_path"]).read_bytes()]
    for original, output in zip(frames, outputs):
        image = np.asarray(Image.open(output["image_path"]))
        editable = np.asarray(Image.open(original["remaining_mask_path"])) == 255
        assert np.all(image[editable] == [0, 0, 255])
        assert np.all(image[~editable] == [255, 0, 0])
        assert _sha256_file(Path(original["image_path"])) == original["image_digest"]
        assert output["generated_pixel_count"] == 15
        assert output["view_consistency"] == "requires_review"
    assert completion.complete_background_images(**args) == outputs
    assert len(calls) == 2


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


def test_edit_feathers_background_margin_but_preserves_outside_and_replaces_core(tmp_path, monkeypatch):
    frames, admission = _inputs(tmp_path)
    for frame in frames:
        frame["edge_feather_pixels"] = 2
    _, _, digest = completion._validated_backend(completion.REGISTRY_PATH, backend_id=completion.BACKEND_ID)
    admission["allocation_binding_digest"] = canonical_digest(completion.completion_binding(
        frames, task_digest="task", backend_digest=digest))
    def edit(**kwargs):
        stream = BytesIO()
        Image.new("RGB", kwargs["expected_size"], "blue").save(stream, format="PNG")
        return {"succeeded": True, "generated": stream.getvalue(),
                "usage": _normalized_usage({"output_tokens_details": {"image_tokens": 100}})}
    monkeypatch.setattr(completion, "_execute_frame_request", edit)
    outputs = completion.complete_background_images(frames=frames, task_digest="task", output_root=tmp_path / "edits",
        admission=admission, admission_grant=_grant(admission), token="test")
    pixels = np.asarray(Image.open(outputs[0]["image_path"]))
    np.testing.assert_array_equal(pixels[5, 3], [128, 0, 128])
    np.testing.assert_array_equal(pixels[7, 4], [0, 0, 255])
    np.testing.assert_array_equal(pixels[4, 3], [255, 0, 0])


def test_admission_dictionary_cannot_authorize_image_spend(tmp_path, monkeypatch):
    frames, admission = _inputs(tmp_path)
    monkeypatch.setattr(completion, "_execute_frame_request", lambda **_: pytest.fail("must not spend"))
    with pytest.raises(RuntimeError, match="grant_missing"):
        completion.complete_background_images(frames=frames, task_digest="task", output_root=tmp_path / "edits",
                                              admission=admission, token="test")


def test_background_review_keeps_client_alive_until_response(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace as NS
    import google

    state = {"closed": False, "calls": 0}
    def generate(**kwargs):
        assert not state["closed"]
        state["calls"] += 1
        return NS(candidates=[NS(finish_reason="STOP")], text=json.dumps({
            "consistent_background": True, "task_objects_removed": True,
            "people_absent": True, "unrelated_objects_preserved": True}))
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
    args = dict(frames=frames, original_frames=frames, plan={"task_context_sha256": "task", "targets": []},
                output_root=tmp_path / "review")
    assert completion.verify_completed_background(**args)["status"] == "passed"
    assert state == {"closed": True, "calls": 1}
    assert completion.verify_completed_background(**args)["status"] == "passed"
    assert state["calls"] == 1
