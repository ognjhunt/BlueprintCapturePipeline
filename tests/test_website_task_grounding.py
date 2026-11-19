"""ADP-009B/day 14: exact-image evidence resolves coarse video localization."""
from __future__ import annotations

import enum
import json

import pytest
from blueprint_pipeline.website_task_grounding import validate_grounding


def candidate(**overrides):
    return {"visible": True, "confidence": 0.92, "box_xywh_normalized": [0.2, 0.3, 0.2, 0.1],
            "segmentation_prompt": "white book", "reason": "The object directly beneath the blue case.", **overrides}


def test_grounding_refines_visual_evidence_without_rewriting_the_task():
    target = {"target_id": "support", "disposition": "keep", "task_effect": "static_contact",
              "semantic_label": "white container under blue case", "spatial_evidence": [{"timestamp_seconds": 8}]}
    result = validate_grounding(candidate(), target=target, timestamp=8.133, frame_id="decoded-244", image_digest="sha256:image")
    assert result["target_id"] == "support" and result["disposition"] == "keep"
    assert result["semantic_label"] == target["semantic_label"]
    assert result["spatial_evidence"] == [{"timestamp_seconds": 8.133, "box_xywh_normalized": [0.2, 0.3, 0.2, 0.1]}]
    assert result["segmentation_prompt"] == "white book"
    assert result["grounding"]["source_frame_id"] == "decoded-244"
    assert result["grounding"]["image_digest"] == "sha256:image"


@pytest.mark.parametrize("changes", [{"visible": False}, {"confidence": 0.4}, {"confidence": True},
    {"box_xywh_normalized": [0.9, 0.2, 0.3, 0.2]}, {"box_xywh_normalized": [0, 0, float("nan"), 1]},
    {"segmentation_prompt": ""}, {"reason": ""}])
def test_uncertain_or_invalid_grounding_cannot_authorize_a_mask(changes):
    with pytest.raises(ValueError, match="grounding"):
        validate_grounding(candidate(**changes), target={}, timestamp=8, frame_id="frame", image_digest="image")


def test_concept_recovery_binds_unedited_crop_and_keeps_full_frame_coordinates(tmp_path, monkeypatch):
    from PIL import Image
    from blueprint_pipeline import website_task_grounding as module
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file

    video = tmp_path / "video.mp4"
    video.write_bytes(b"retained video")
    def extract(args, **kwargs):
        Image.new("RGB", (100, 200), "white").save(args[-1])
    monkeypatch.setattr(module.subprocess, "run", extract)
    calls = []
    def retain(**kwargs):
        calls.append(kwargs)
        return {"observation": candidate()}
    monkeypatch.setattr(module, "retained_gemini_call", retain)
    target = {"target_id": "support", "disposition": "keep", "spatial_evidence": [
        {"timestamp_seconds": 8, "box_xywh_normalized": [0.2, 0.3, 0.2, 0.1]}]}
    target["grounding"] = {"source_image_path": "/private/local/path.png", "image_digest": "sha256:source"}
    result = module.ground_task_target(target=target, tracks=[],
        registry=[{"source_frame_id": "frame", "model_frame_index": 240, "decoded_pts_seconds": 8}],
        video={"path": str(video), "sha256": _sha256_file(video)},
        task_context={"description": "Keep the support under the blue object"}, output_root=tmp_path / "ground",
        failed_segmentation_prompt="white container")
    binding = calls[0]["binding"]
    recovery = binding["concept_recovery"]
    assert recovery["failed_prompt"] == "white container"
    crop = tmp_path / "ground" / "support-240-concept.png"
    assert recovery["crop_digest"] == _sha256_file(crop)
    assert recovery["crop_box_pixels"] == (16, 55, 44, 84)
    assert "FIRST, full image" in binding["prompt"]
    assert "/private/local/path.png" not in binding["prompt"]
    assert result["target_id"] == "support" and result["disposition"] == "keep"
    assert result["spatial_evidence"][0]["box_xywh_normalized"] == [0.2, 0.3, 0.2, 0.1]


# --- test job itest-917ca852 (2026-11-19): the look's answer never finished ----------------------------------------
# The clean plate's segmenter found no track for the analysis noun, so it fell back on this exact-frame look. The look
# asked Gemini 3.8 for a box and a noun with 2048 output tokens at its default thinking effort, which shares that
# limit. Twice the call ran about 20 s and ended without STOP; the answer was raised away inside the receipt, which
# stayed "submitting", was counted as spent and was sent again under a new reservation, and the job stopped there.

def _context():
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    value = {"schema_version": "website_site_task_context.v1", "request_id": "itest", "scene_id": "site-itest",
             "capture_id": "walkthrough-itest", "confirmed": True, "confirmed_at": "2026-11-19T00:00:00Z",
             "description": "Open the top drawer under the hob."}
    value["context_digest"] = canonical_digest(value, digest_field="context_digest")
    return value


ASSEMBLY = {"target_id": "cabinet_assembly", "semantic_label": "kitchen cabinet assembly with drawers",
            "segmentation_prompt": "cabinet unit with drawers", "task_effect": "manipulated", "disposition": "remove",
            "articulated_part": "top drawer", "articulation_kind": "prismatic",
            "spatial_evidence": [{"timestamp_seconds": 3.5, "box_xywh_normalized": [0.573, 0.737, 0.244, 0.263]}]}
REGISTRY = [{"source_frame_id": "decoded-000000084", "model_frame_index": 84, "decoded_pts_seconds": 3.5,
             "width": 20, "height": 20}]


class Provider:
    """Gemini, its client and the website's reservations, faked; each answer is taken in turn."""

    def __init__(self, monkeypatch, *answers):
        import sys
        import time
        from types import SimpleNamespace as NS

        from blueprint_pipeline import website_gemini_receipts as receipts
        from blueprint_pipeline import website_task_grounding as grounding
        from blueprint_pipeline.paid_resource_admission import require_paid_resource_admission
        self.answers, self.sent, self.reservations = list(answers), [], []
        provider = self

        class Client:
            def __init__(self, **_):
                self.models = self

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def generate_content(self, **kwargs):
                provider.sent.append(kwargs)
                return provider.answers.pop(0)

        types = NS(Part=NS(from_bytes=lambda **kw: kw), GenerateContentConfig=lambda **kw: kw,
                   ThinkingConfig=lambda **kw: kw, HttpOptions=lambda **kw: kw, HttpRetryOptions=lambda **kw: kw)
        monkeypatch.setitem(sys.modules, "google.genai", NS(Client=Client, types=types))
        monkeypatch.setattr(grounding, "_api_key", lambda: ("fixture-key", "fixture"))

        def reserve(**kwargs):
            provider.reservations.append(kwargs)
            admission = {"schema_version": "paid_lane_admission.v1", "status": "admitted",
                         "resource_class": kwargs["resource_class"], "blockers": [], "provider": "google",
                         "allocation_binding_digest": kwargs["binding_digest"], "expires_at_epoch": time.time() + 600,
                         "maximum_cost_usd": kwargs["maximum_cost_usd"]}
            return admission, require_paid_resource_admission(
                admission, resource_class=kwargs["resource_class"], expected_schema_version="paid_lane_admission.v1")
        monkeypatch.setattr(receipts, "reserve_website_preparation_spend", reserve)


class FinishReason(str, enum.Enum):
    """As the SDK's own: a string enum, equal to its name."""
    STOP = "STOP"
    MAX_TOKENS = "MAX_TOKENS"
    SAFETY = "SAFETY"


def answer(finish="STOP", text=None, **usage):
    from types import SimpleNamespace as NS
    if text is None:
        text = json.dumps(candidate(segmentation_prompt="black drawer cabinet",
                                    box_xywh_normalized=[0.57, 0.7, 0.2, 0.3]))
    return NS(candidates=[NS(finish_reason=FinishReason(finish))] if finish else [], text=text,
              usage_metadata=NS(**usage) if usage else None)


def _ground(tmp_path, monkeypatch, **kwargs):
    from PIL import Image

    from blueprint_pipeline import website_task_grounding as module
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
    video = tmp_path / "video.mp4"
    video.write_bytes(b"retained video")
    monkeypatch.setattr(module.subprocess, "run",
                        lambda args, **_: Image.new("RGB", (20, 20), "black").save(args[-1]))
    return module.ground_task_target(target=ASSEMBLY, tracks=[], registry=REGISTRY,
                                     video={"path": str(video), "sha256": _sha256_file(video)},
                                     task_context=_context(), output_root=tmp_path / "grounding", **kwargs)



def test_an_unfinished_answer_is_kept_with_its_reason_and_never_bought_again(tmp_path, monkeypatch):
    from blueprint_pipeline import website_task_grounding as module
    provider = Provider(monkeypatch, answer("MAX_TOKENS", text='{"visi', candidates_token_count=40,
                                            thoughts_token_count=8152, prompt_token_count=4100))
    for _ in range(3):
        with pytest.raises(ValueError, match=f"^{module.INCOMPLETE}$"):
            _ground(tmp_path, monkeypatch)
    # One reservation and one call; every later run reads the kept answer at $0.
    assert len(provider.reservations) == 1 and len(provider.sent) == 1
    [receipt] = [json.loads(path.read_text()) for path in (tmp_path / "grounding/receipts").glob("*.json")]
    assert receipt["status"] == "completed"
    assert receipt["result"] == {"observation": None, "finish_reason": "MAX_TOKENS", "partial_output": '{"visi',
                                 "token_usage": {"prompt_token_count": 4100, "candidates_token_count": 40,
                                                 "thoughts_token_count": 8152}}
    # The look is asked at low thinking effort with room for its answer, and the receipt binds both.
    config = provider.sent[0]["config"]
    assert config["thinking_config"] == {"thinking_level": "low"} and config["max_output_tokens"] == 8192
    binding = receipt["request"]["binding"]
    assert (binding["thinking_level"], binding["max_output_tokens"]) == ("low", 8192)
    assert receipt["request"]["maximum_cost_usd"] == module.gemini_quote(
        model=binding["model"], input_tokens=len(binding["prompt"].encode()) + 3168, max_output_tokens=8192)


@pytest.mark.parametrize(("response", "reason"), [
    (answer(None, text=""), "no_candidates"), (answer("SAFETY", text=""), "SAFETY"),
    (answer("STOP", text="not json"), "STOP_unparseable")])
def test_every_answer_without_a_finished_observation_is_kept_as_none(response, reason):
    from blueprint_pipeline.website_task_grounding import grounding_answer
    kept = grounding_answer(response)
    assert kept["observation"] is None and kept["finish_reason"] == reason


def test_a_finished_answer_grounds_the_object_with_the_refined_noun(tmp_path, monkeypatch):
    provider = Provider(monkeypatch, answer())
    grounded = _ground(tmp_path, monkeypatch)
    assert grounded["segmentation_prompt"] == "black drawer cabinet"
    assert grounded["spatial_evidence"] == [{"timestamp_seconds": 3.5, "box_xywh_normalized": [0.57, 0.7, 0.2, 0.3]}]
    assert grounded["target_id"] == ASSEMBLY["target_id"] and len(provider.sent) == 1


def _older_form(tmp_path, monkeypatch, outcome, times):
    """The look as the release before this one sent it (2048 tokens, default effort), run ``times`` times."""
    from blueprint_pipeline import website_gemini_receipts as receipts
    from blueprint_pipeline import website_task_grounding as module
    seen = {}
    monkeypatch.setattr(module, "retained_gemini_call", lambda **kw: seen.update(kw) or {"observation": candidate()})
    _ground(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "retained_gemini_call", receipts.retained_gemini_call)
    current = seen["binding"]
    older = {key: current[key] for key in ("kind", "model", "prompt", "source_video_digest", "frame", "image_digest",
                                           "media_resolution")} | {"max_output_tokens": 2048}
    cost = receipts.gemini_quote(model=current["model"], input_tokens=len(current["prompt"].encode()) + 3168,
                                 max_output_tokens=2048)
    assert seen["earlier_requests"] == [(older, cost)]

    def invoke():
        if outcome == "unanswered":
            raise ValueError(module.INCOMPLETE)  # the older release raised its unfinished answer away
        return {"observation": candidate(segmentation_prompt="drawer cabinet")}
    for _ in range(times):
        try:
            receipts.retained_gemini_call(output_root=tmp_path / "grounding/receipts", binding=older,
                                          task_context=_context(), maximum_cost_usd=cost, preflight=lambda: None,
                                          invoke=invoke)
        except ValueError as exc:
            assert str(exc) in {module.INCOMPLETE, receipts.EXHAUSTED}


@pytest.mark.parametrize(("outcome", "times", "bought"), [
    ("answered", 1, 0),      # a job grounded before keeps its noun and its masks: nothing is bought
    ("unanswered", 1, 1),    # counted as spent; the look is asked once in the current form
    ("unanswered", 3, 1),    # job itest-917ca852's chain, its bound spent: the current form is still asked once
])
def test_a_look_an_older_release_sent_is_reused_or_asked_once_in_the_current_form(tmp_path, monkeypatch, outcome,
                                                                                  times, bought):
    provider = Provider(monkeypatch, answer())
    _older_form(tmp_path, monkeypatch, outcome, times)
    before = len(provider.reservations)
    for _ in range(2):
        grounded = _ground(tmp_path, monkeypatch)
    assert len(provider.sent) == bought and len(provider.reservations) - before == bought
    assert grounded["segmentation_prompt"] == ("drawer cabinet" if outcome == "answered" else "black drawer cabinet")
