"""Dense initial completion preserves its evidence while respecting the existing call cap."""
import base64
from copy import deepcopy
from io import BytesIO
import json
import math
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from blueprint_pipeline import atlas_scene_planner as planner
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.openai_prompt_cache import worst_case_reservation_usd


def dense_initial_inputs(tmp_path, rotation):
    image = tmp_path / "fixture-source.png"
    Image.new("RGB", (1280, 720), "white").save(image)
    sha = _sha256_file(image)
    survey = {"source_video_digest": "sha256:" + "a" * 64, "heldout_pixels_included": False,
              "display_rotation_degrees": rotation, "frames": [{"frame_id": f"conditioned-{index}",
                  "source_image_path": str(image), "source_image_digest": sha} for index in range(3)]}
    survey["digest"] = canonical_digest(survey, digest_field="digest")
    path = tmp_path / "survey.json"
    path.write_text(json.dumps(survey))
    cameras = [f"gap-camera-{index:04}-source-conditioned-work-area" for index in range(163)]
    completion = {"scope": "missing_surfaces_within_captured_scene",
        "candidate_camera_ids": [{"camera_id": name, "source_frame_id": "conditioned-0", "purpose": "gap_target",
            "source_gap_ids": [name + "-gap-1"], "pose_rub": [.1234, .2345, .3456, 0., 0., 0., 1.],
            "aim": [.4567, .5678, .6789]} for name in cameras],
        "gap_survey": {"views": [{"camera_id": name, "gaps": [
            {"label": str(index), "pixel_count": 100, "box_xywh_normalized": [.1234, .2345, .3456, .4567]}
            for index in range(7)]} for name in cameras]},
        "previews": [{"path": str(image), "sha256": sha, "source_frame_id": f"source-{index}",
            "camera_id": cameras[index % len(cameras)], "timestamp_seconds": index / 10,
            "kind": "original_capture_source_check_only" if index < 141 else "geometry_gap_survey"}
            for index in range(242)]}
    return path, survey, completion


@pytest.mark.parametrize("rotation,repair", [(0, None), (45, {"error_type": "schema", "error_message": "Return all required fields"})])
def test_initial_completion_fits_without_view_loss_or_token_undercount(tmp_path, monkeypatch, rotation, repair):
    from blueprint_pipeline import paid_resource_admission as admission, website_task_context as control
    path, survey, completion = dense_initial_inputs(tmp_path, rotation)
    original = deepcopy(completion)
    policy = {"model": "gpt-6.1-sol", "maximum_review_cost_usd": 1.}
    reservations, calls = [], []
    monkeypatch.setattr(admission, "require_paid_resource_admission_grant", lambda *a, **k: None)
    monkeypatch.setattr(control, "reserve_website_preparation_spend", lambda **kw: (reservations.append(kw) or {}, object()))
    monkeypatch.setattr(control, "website_webapp_request", lambda **kw: {"status": "settled"})
    context = {"description": "Move the tray to the other end of the same table", "context_digest": "sha256:" + "b" * 64,
               "request_id": "test-request", "capture_id": "test-capture", "scene_id": "test-scene"}
    with pytest.raises(ValueError, match="token_budget_exceeded"):
        planner.preflight_review_input(survey, context, planner.completion_summary(completion),
            model=policy["model"], maximum=1., output_tokens=4000,
            instructions=planner.COMPLETION_INSTRUCTIONS, schema=planner.CompletionProposal)

    class Invoker:
        config = SimpleNamespace(max_inference_cost_usd=1.)

        def configure_reservation_audit(self, **_kwargs):
            pass

        def invoke(self, spec, payload):
            calls.append((spec, payload))
            return SimpleNamespace(output={"status": "proposed", "selected_camera_ids": [
                completion["candidate_camera_ids"][0]["camera_id"]], "gap_assessment": ["Complete nearby floor"],
                "remaining_unknowns": ["Hidden room surfaces"]}, cost_usd=.03,
                                   usage={"input_tokens": 100}, model=spec.model, sdk_version="offline-fixture")

    options = dict(survey_path=path, task_context=context, configuration={"policy": policy},
        output_root=tmp_path / "plans", invoker=Invoker(), release_blockers=lambda: ("c" * 40, []),
        feedback=None, completion=completion, scaling=None, recovery=None, truncation_retry=None,
        validation_repair=repair)
    receipt = planner._review_attempt(**options)
    assert planner._review_attempt(**options) == receipt
    assert len(reservations) == len(calls) == 1 and reservations[0]["maximum_cost_usd"] == 1.
    spec, payload = calls[0]
    assert worst_case_reservation_usd(model=policy["model"], input_token_ceiling=spec.max_input_tokens,
                                      max_output_tokens=spec.max_output_tokens, cache_policy=None) <= 1.
    labels = [json.loads(p["text"]) for p in payload[0]["content"] if p["type"] == "input_text"]
    feedback = next(row["reconstruction_feedback"] for row in labels if "reconstruction_feedback" in row)
    restored = [dict(zip(feedback["candidate_camera_columns"], row, strict=True)) for row in feedback["candidate_camera_ids"]]
    expected = planner.completion_summary(completion)
    assert restored == expected["candidate_camera_ids"]
    assert [dict(zip(feedback["gap_survey"]["view_columns"], row, strict=True)) for row in
            feedback["gap_survey"]["views"]] == [{"outline": False, **row} for row in expected["gap_survey"]["views"]]
    assert [row["original_capture_frame"] for row in labels if "original_capture_frame" in row] == [
        row["source_frame_id"] for row in completion["previews"][:141]]
    tiles = [tile for row in labels for tile in row.get("derived_reconstruction_previews", [])]
    assert [(row["camera_id"], row["source_frame_id"]) for row in tiles] == [
        (row["camera_id"], row["source_frame_id"]) for row in completion["previews"][141:]]
    encoded_tokens = 24_000 + len(spec.instructions.encode()) + len(json.dumps(spec.output_type.model_json_schema()).encode())
    for part in payload[0]["content"]:
        if part["type"] == "input_text":
            encoded_tokens += len(part["text"].encode())
        else:
            with Image.open(BytesIO(base64.b64decode(part["image_url"].partition(",")[2]))) as image:
                encoded_tokens += 3 * math.ceil(image.width / 32) * math.ceil(image.height / 32) + 256
    assert spec.max_input_tokens >= encoded_tokens
    assert not spec.hosted_tools and spec.max_hosted_tool_calls_per_turn == 0
    if repair:
        assert spec.instructions.count("Return all required fields") == 1
    assert completion == original
    assert _sha256_file(Path(survey["frames"][0]["source_image_path"])) == survey["frames"][0]["source_image_digest"]


@pytest.mark.parametrize("model,maximum", [("gpt-6.1-sol", .03), ("gpt-5.6-sol", 1.)])
def test_initial_completion_that_still_exceeds_cap_never_reserves(tmp_path, monkeypatch, model, maximum):
    from blueprint_pipeline import website_task_context as control
    path, _, completion = dense_initial_inputs(tmp_path, 0)
    monkeypatch.setattr(control, "reserve_website_preparation_spend", lambda **_: pytest.fail("over-cap reservation"))
    with pytest.raises(ValueError, match="token_budget_exceeded"):
        planner.review_survey(survey_path=path, task_context={"description": "Review all views"},
            configuration={"policy": {"model": model, "maximum_review_cost_usd": maximum}},
            completion=completion, output_root=tmp_path / "plans")


@pytest.mark.parametrize("caps", [[1., 4.], [.03, 1., 4., 5.]])
def test_new_compaction_reuses_completed_higher_cap_receipt_before_buying_again(tmp_path, monkeypatch, caps):
    from blueprint_pipeline import paid_resource_admission as admission, website_task_context as control
    from blueprint_pipeline.website_atlas_pipeline import REVIEW_CAPS

    path, _, completion = dense_initial_inputs(tmp_path, 0)
    reservations, calls, settlements = [], [], []
    monkeypatch.setattr(admission, "require_paid_resource_admission_grant", lambda *a, **k: None)
    monkeypatch.setattr(control, "reserve_website_preparation_spend", lambda **kw: (reservations.append(kw) or {}, object()))
    monkeypatch.setattr(control, "website_webapp_request", lambda **kw: (settlements.append(kw) or {"status": "settled"}))

    class Invoker:
        config = SimpleNamespace(max_inference_cost_usd=4.)

        def configure_reservation_audit(self, **_kwargs):
            pass

        def invoke(self, spec, payload):
            calls.append(spec)
            return SimpleNamespace(output={"status": "proposed", "selected_camera_ids": [
                completion["candidate_camera_ids"][0]["camera_id"]], "gap_assessment": ["Complete nearby floor"],
                "remaining_unknowns": ["Hidden room surfaces"]}, cost_usd=.03,
                                   usage={"input_tokens": 100}, model=spec.model, sdk_version="offline-fixture")

    policy = {"model": "gpt-6.1-sol", "maximum_review_cost_usd": 4.}
    options = dict(survey_path=path, task_context={"description": "Move the tray", "context_digest": "sha256:" + "b" * 64,
        "request_id": "test-request", "capture_id": "test-capture", "scene_id": "test-scene"},
        configuration={"policy": policy}, output_root=tmp_path / "plans", invoker=Invoker(),
        release_blockers=lambda: ("c" * 40, []), feedback=None, completion=completion, scaling=None,
        recovery=None, truncation_retry=None)
    # Before the new compact fit, the $1 cap refused this input and a later cap
    # bought it. Its retained receipt still binds that original $4 allowance.
    bought = planner._review_attempt(**options)
    assert bought["binding"]["maximum_cost_usd"] == 4.
    resumed = {**options, "configuration": {"policy": {**policy, "maximum_review_cost_usd": caps[-1]}, REVIEW_CAPS: caps}}
    assert planner._review_attempt(**resumed) == bought
    assert len(reservations) == len(calls) == len(settlements) == 1
    assert reservations[0]["maximum_cost_usd"] == 4.
