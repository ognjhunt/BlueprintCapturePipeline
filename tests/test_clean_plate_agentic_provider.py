"""Actual provider request shape and fail-closed agentic trace; no live calls."""
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from blueprint_pipeline.clean_plate_removal_analysis_gemini import (
    DEFAULT_MODEL,
    _invoke_agentic_video,
    _provider_error_blocker,
    parse_removal_plan_response,
)
from blueprint_pipeline.capture_audio_minimization import AudioRemovalError
from blueprint_pipeline.core.handoff_lease import HandoffLeaseError, bind_handoff_lease
from tests.capture_audio_fakes import VIDEO_ONLY_MARKER, FakeMediaTools


@pytest.fixture(autouse=True)
def media_tools(monkeypatch) -> FakeMediaTools:
    """Hermetic ffmpeg/ffprobe: the provider receives a video-only copy."""

    return FakeMediaTools().install(monkeypatch)


def invoke(tmp_path, steps, finish="completed", processing="agentic"):
    path = tmp_path / "source.mov"
    path.write_bytes(b"source video")
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return NS(status=finish, steps=steps)

    media = NS(name="files/test", uri="https://provider.test/file", state=NS(name="ACTIVE"))
    deleted = []
    files = NS(upload=lambda **_: media, delete=lambda **kwargs: deleted.append(kwargs))
    genai = NS(Client=lambda **_: NS(interactions=NS(create=generate), files=files))
    types = NS(HttpOptions=NS, HttpRetryOptions=NS)
    try:
        result = _invoke_agentic_video(api_key="fake", model=DEFAULT_MODEL,
                                      processing=processing, video_path=path,
                                      genai=genai, types=types)
    finally:
        if processing in {"agentic", "static"}:
            assert deleted == [{"name": "files/test"}]
    return result, calls


def trace():
    return [NS(type="processing_call", id="inspect-1"),
            NS(type="processing_result", call_id="inspect-1")]


def output(text):
    return NS(type="model_output", content=[NS(type="text", text=text)])


def test_requests_agentic_38_and_filters_thoughts(tmp_path):
    result, calls = invoke(tmp_path, [*trace(), NS(type="thought", text="private"),
                                    output('{"targets":[]}')])
    assert result["text"] == '{"targets":[]}'
    assert result["video_processing"]["media_tool_calls"] == 1
    assert result["video_processing"]["media_tool_responses"] == 1
    assert len(calls) == 1
    assert calls[0]["model"] == "gemini-3.8-flash"
    assert calls[0]["generation_config"]["thinking_level"] == "low"
    assert calls[0]["store"] is False
    assert calls[0]["input"][0] == {"type": "video", "processing": "agentic",
        "uri": "https://provider.test/file", "mime_type": "video/quicktime"}


def test_static_answer_does_not_masquerade_as_agentic(tmp_path):
    with pytest.raises(ValueError, match="agentic_trace_missing"):
        invoke(tmp_path, [output('{"targets":[]}')])


def test_visual_segmentation_prompt_survives_normalization():
    import json
    row = {"semantic_label": "small container beside picture", "target_class": "movable_object",
           "segmentation_prompt": "blue object", "disposition": "remove"}
    assert parse_removal_plan_response(json.dumps([row]))[0]["segmentation_prompt"] == "blue object"
    del row["segmentation_prompt"]
    assert parse_removal_plan_response(json.dumps([row]))[0]["segmentation_prompt"] == row["semantic_label"]


def test_partial_json_not_accepted_as_completed_analysis(tmp_path):
    with pytest.raises(ValueError, match="analysis_incomplete"):
        invoke(tmp_path, [*trace(), output("{}")], finish="incomplete")


def test_unmatched_video_processing_result_is_not_an_agentic_receipt(tmp_path):
    steps = trace()
    steps[1].call_id = "different-call"
    with pytest.raises(ValueError, match="agentic_trace_missing"):
        invoke(tmp_path, [*steps, output('{"targets":[]}')])


def test_provider_tool_loop_error_is_distinct_from_credit_or_key_failure():
    error = ValueError("Error code: 400 - Model generated too many tool calls. Please retry the request.")
    assert _provider_error_blocker(error) == "gemini_clean_plate_analysis_incomplete_too_many_tool_calls"


def test_static_single_pass_has_explicit_mode_and_no_agentic_trace(tmp_path):
    result, calls = invoke(tmp_path, [output('{"targets":[]}')], processing="static")
    assert calls[0]["input"][0]["processing"] == {"type": "static", "fps": 2}
    assert result["video_processing"]["mode"] == "static"
    assert result["video_processing"]["media_tool_calls"] == 0
    schema = calls[0]["response_format"]["schema"]
    import jsonschema
    jsonschema.validate({"targets": []}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"targets": [{"target_class": "movable_object", "disposition": "remove"}]}, schema)
    assert {"target_role", "task_effect", "task_basis_quote", "decision_reason"} <= set(schema["properties"]["targets"]["items"]["required"])


@pytest.mark.parametrize(("duration", "expected"), [(13.525, "static"), (300, "static"), (300.1, "agentic")])
def test_duration_selects_mode_without_changing_task_semantics(monkeypatch, tmp_path, duration, expected):
    from blueprint_pipeline import clean_plate_removal_analysis_gemini as module
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: NS(stdout='{"format":{"duration":%s}}' % duration))
    assert module._video_processing(tmp_path / "video.mov", "auto") == (expected, duration)
    prompt = module._video_prompt(expected, {"description": "Move only the blue box"})
    assert "Move only the blue box" in prompt
    assert "Movability alone NEVER warrants removal" in prompt
    if expected == "static":
        assert "media-processing tool calls" not in prompt


def test_unreadable_duration_cannot_silently_choose_an_expensive_mode(monkeypatch, tmp_path):
    from blueprint_pipeline import clean_plate_removal_analysis_gemini as module
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: NS(stdout='{"format":{}}'))
    with pytest.raises(ValueError, match="video_duration_unavailable"):
        module._video_processing(tmp_path / "video.mov", "auto")


@pytest.mark.parametrize("payload", ["not json", "{}", '{"targets":[{}]}'])
def test_malformed_analysis_cannot_become_an_empty_clean_scene(payload):
    with pytest.raises(ValueError, match="removal_analysis_"):
        parse_removal_plan_response(payload, strict=True)


def test_support_table_cannot_be_removed_as_generic_movable_clutter():
    import json
    target = {"target_id": "table", "target_class": "fixed_clutter",
              "target_role": "support", "disposition": "remove"}
    with pytest.raises(ValueError, match="non_task_removal"):
        parse_removal_plan_response(json.dumps({"targets": [target]}), strict=True)


def test_task_removal_requires_timestamped_confident_source_observation():
    import json
    target = {"target_id": "box", "target_class": "movable_object",
              "target_role": "task_object", "disposition": "remove", "confidence": 0.95,
              "task_effect": "manipulated", "rebuild_intent": "rebuild_and_compose",
              "decision_reason": "The robot transfers this box.", "task_basis_quote": "Move the box"}
    with pytest.raises(ValueError, match="evidence_missing"):
        parse_removal_plan_response(json.dumps({"targets": [target]}), strict=True, task_description="Move the box")
    target["spatial_evidence"] = [{"timestamp_seconds": 2.0}]
    parsed = parse_removal_plan_response(json.dumps({"targets": [target]}), strict=True, task_description="Move the box")
    assert parsed[0]["rebuild_intent"] == "rebuild_and_compose"


def test_movable_chair_stays_for_box_task_but_is_rebuilt_for_chair_task():
    import json
    chair = {"target_id": "chair", "semantic_label": "office chair",
             "target_class": "movable_object", "target_role": "background",
             "task_effect": "unrelated", "disposition": "keep", "rebuild_intent": "none",
             "decision_reason": "Chair is outside the carton transfer work area."}
    kept = parse_removal_plan_response(json.dumps({"targets": [chair]}), strict=True,
                                      task_description="Move the box")
    assert kept[0]["disposition"] == "keep"
    assert kept[0]["collision_required"] is False
    moved = dict(chair, target_role="task_object", task_effect="manipulated",
                 disposition="remove", rebuild_intent="rebuild_and_compose", confidence=0.95,
                 decision_reason="The task explicitly moves this chair.",
                 task_basis_quote="Move the chair", spatial_evidence=[{"timestamp_seconds": 3}])
    rebuilt = parse_removal_plan_response(json.dumps({"targets": [moved]}), strict=True,
                                         task_description="Move the chair to the marked area")
    assert rebuilt[0]["collision_required"] is True
    with pytest.raises(ValueError, match="task_basis_missing"):
        parse_removal_plan_response(json.dumps({"targets": [moved]}), strict=True,
                                    task_description="Move the box")


@pytest.mark.parametrize("effect,role", [("static_contact", "support"), ("static_obstacle", "obstacle")])
def test_contact_geometry_does_not_require_appearance_removal(effect, role):
    import json
    row = {"target_id": "table", "target_class": "movable_object", "target_role": role,
           "task_effect": effect, "disposition": "keep", "rebuild_intent": "none",
           "decision_reason": "The table stays fixed throughout this task."}
    parsed = parse_removal_plan_response(json.dumps({"targets": [row]}), strict=True)
    assert parsed[0]["collision_required"] is True
    assert parsed[0]["disposition"] == "keep"


def test_uncertain_task_object_requires_question_and_cannot_be_removed():
    import json
    row = {"target_id": "tote", "target_class": "movable_object", "target_role": "task_object",
           "task_effect": "uncertain", "disposition": "keep", "rebuild_intent": "none",
           "decision_reason": "Two totes match the task description."}
    with pytest.raises(ValueError, match="uncertainty_requires_question"):
        parse_removal_plan_response(json.dumps({"targets": [row]}), strict=True)
    row["clarification_question"] = "Which of the two totes should the robot move?"
    parsed = parse_removal_plan_response(json.dumps({"targets": [row]}), strict=True)
    assert parsed[0]["clarification_question"] == row["clarification_question"]
    row["disposition"] = "remove"
    with pytest.raises(ValueError, match="uncertainty_requires_question"):
        parse_removal_plan_response(json.dumps({"targets": [row]}), strict=True)


def test_unrelated_movable_object_cannot_be_silently_rebuilt():
    import json
    row = {"target_id": "book", "target_class": "movable_object", "target_role": "background",
           "task_effect": "unrelated", "disposition": "keep", "rebuild_intent": "rebuild_and_compose",
           "decision_reason": "Unrelated to the task."}
    with pytest.raises(ValueError, match="task_effect_conflict"):
        parse_removal_plan_response(json.dumps({"targets": [row]}), strict=True)


def test_placement_destination_is_preserved_as_static_contact():
    import json
    target = {"target_id": "tray", "target_class": "movable_object", "target_role": "destination",
              "task_effect": "static_contact", "disposition": "keep", "rebuild_intent": "none",
              "semantic_label": "destination tray", "decision_reason": "The robot places the box here.",
              "placement_relation": "inside", "task_basis_quote": "inside the tray",
              "confidence": 0.9, "spatial_evidence": [{"timestamp_seconds": 2, "box_xywh_normalized": [0.2, 0.2, 0.2, 0.2]}]}
    parsed = parse_removal_plan_response(json.dumps({"targets": [target]}), strict=True, task_description="Move box inside the tray")
    assert parsed[0]["target_role"] == "destination"
    assert parsed[0]["placement_relation"] == "inside"
    assert parsed[0]["disposition"] == "keep"
    assert parsed[0]["collision_required"] is True


def test_destination_relation_cannot_be_invented_from_movability():
    import json
    target = {"target_id": "tray", "target_class": "movable_object", "target_role": "destination",
              "task_effect": "static_contact", "disposition": "keep", "rebuild_intent": "none",
              "decision_reason": "A tray is visible."}
    with pytest.raises(ValueError, match="destination_relation_required"):
        parse_removal_plan_response(json.dumps({"targets": [target]}), strict=True,
                                    task_description="Move the small object")


_TASK = {"confirmed": True, "description": "Open the dishwasher door and close it again."}


def _door(**overrides):
    row = {"target_id": "dishwasher", "semantic_label": "dishwasher", "target_class": "movable_object",
           "target_role": "task_object", "task_effect": "manipulated", "disposition": "remove",
           "rebuild_intent": "rebuild_and_compose", "confidence": 0.95,
           "decision_reason": "The task opens this door.", "task_basis_quote": "Open the dishwasher door",
           "spatial_evidence": [{"timestamp_seconds": 2.0}], "articulation_kind": "revolute",
           "articulated_part": "dishwasher door"}
    return {**row, **overrides}


def _rack(**overrides):
    row = {"target_id": "lower_rack", "semantic_label": "lower rack", "target_class": "movable_object",
           "target_role": "obstacle", "task_effect": "static_obstacle", "disposition": "keep",
           "rebuild_intent": "none", "decision_reason": "The rack stays inside the tub.",
           "articulation_kind": "prismatic", "articulated_part": "lower rack"}
    return {**row, **overrides}


def _analysis(tmp_path, monkeypatch, *, first, repaired):
    import json

    import sys
    import types as module_types

    from blueprint_pipeline import clean_plate_removal_analysis_gemini as analysis
    # The provider calls are faked below; the SDK itself need not be installed.
    google = sys.modules.get("google") or module_types.ModuleType("google")
    genai = module_types.ModuleType("google.genai")
    genai.types = module_types.ModuleType("google.genai.types")
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setattr(google, "genai", genai, raising=False)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", genai.types)
    video = tmp_path / "walkthrough.mp4"
    video.write_bytes(b"video")
    monkeypatch.setenv(analysis.GATE_ENV, "1")
    monkeypatch.setattr(analysis, "_api_key", lambda: ("fixture", "fixture"))
    monkeypatch.setattr(analysis, "_video_processing", lambda path, mode: ("static", 10.0))
    def initial(**_kwargs):
        if isinstance(first, Exception):
            raise first
        return {"text": json.dumps({"targets": first}), "video_processing": {"mode": "static"}}

    monkeypatch.setattr(analysis, "_invoke_agentic_video", initial)
    repairs = []

    def repair(**kwargs):
        repairs.append(kwargs)
        if isinstance(repaired, Exception):
            raise repaired
        return {"text": json.dumps({"targets": repaired})}

    monkeypatch.setattr(analysis, "_invoke_text_repair", repair)
    plan = analysis._analyze_removal_targets(video_path=video, task_context=_TASK)
    return plan, repairs


def test_a_mechanism_hint_on_a_static_target_is_repaired_in_one_turn(tmp_path, monkeypatch):
    from blueprint_pipeline.clean_plate_removal_analysis_gemini import _repair_prompt
    plan, repairs = _analysis(tmp_path, monkeypatch, first=[_door(), _rack()],
                              repaired=[_door(), _rack(articulation_kind="", articulated_part="")])
    assert plan["status"] == "completed", plan["blockers"]
    assert plan["validation_repair"] == {"turns": 1, "repaired": True,
                                         "failed_check": "removal_analysis_articulation_hint_invalid"}
    assert [row["articulation_kind"] for row in plan["targets"]] == ["revolute", ""]
    (call,) = repairs
    assert call["error"] == "removal_analysis_articulation_hint_invalid"
    prompt = _repair_prompt(call["prompt"], call["previous_text"], call["error"])
    # Located feedback: only the offending static rack is named, not the door.
    assert '"target_id": "lower_rack"' in prompt and '"target_id": "dishwasher"' not in prompt.split("Original task")[0]


def test_a_repair_that_still_breaks_a_rule_stays_blocked(tmp_path, monkeypatch):
    plan, repairs = _analysis(tmp_path, monkeypatch, first=[_door(), _rack()], repaired=[_door(), _rack()])
    assert plan["status"] == "blocked" and len(repairs) == 1
    assert plan["blockers"] == ["removal_analysis_articulation_hint_invalid",
                                "validation_repair_failed:removal_analysis_articulation_hint_invalid"]


def test_a_valid_first_answer_makes_no_repair_call(tmp_path, monkeypatch):
    plan, repairs = _analysis(tmp_path, monkeypatch, first=[_door()], repaired=[])
    assert plan["status"] == "completed" and repairs == [] and "validation_repair" not in plan


def test_the_repair_turn_resends_text_only_under_the_same_schema():
    from blueprint_pipeline.clean_plate_removal_analysis_gemini import RESPONSE_SCHEMA, _invoke_text_repair
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return NS(status="completed", steps=[output('{"targets":[]}')])

    genai = NS(Client=lambda **_: NS(interactions=NS(create=create),
                                     files=NS(upload=lambda **_: pytest.fail("no media in a repair turn"))))
    result = _invoke_text_repair(api_key="fake", model=DEFAULT_MODEL, genai=genai,
                                 types=NS(HttpOptions=NS, HttpRetryOptions=NS), prompt="Find targets.",
                                 previous_text='{"targets":[{"target_id":"x"}]}', error="removal_analysis_target_invalid")
    assert result == {"text": '{"targets":[]}'}
    (call,) = calls
    assert call["store"] is False and call["response_format"]["schema"] == RESPONSE_SCHEMA
    assert [item["type"] for item in call["input"]] == ["text"]
    assert "Failed check: removal_analysis_target_invalid" in call["input"][0]["text"]
    assert '{"targets":[{"target_id":"x"}]}' in call["input"][0]["text"]


def test_the_retained_decision_names_its_repair_policy_and_reserves_for_it(tmp_path, monkeypatch):
    from blueprint_pipeline import clean_plate_removal_analysis_gemini as analysis
    from blueprint_pipeline import website_gemini_receipts
    video = tmp_path / "walkthrough.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(analysis, "_video_processing", lambda path, mode: ("static", 10.0))
    captured = {}
    monkeypatch.setattr(website_gemini_receipts, "retained_gemini_call", lambda **kwargs: captured.update(kwargs) or {})
    analysis.analyze_removal_targets(video_path=video, output_root=tmp_path / "gemini",
                                     task_context={**_TASK, "capture_id": "walkthrough-x"})
    # A decision retained under the no-repair policy is a different request: it is never reused.
    assert captured["binding"]["validation_repair_turns"] == 1
    single = website_gemini_receipts.gemini_quote(model=captured["binding"]["model"], input_tokens=1, max_output_tokens=8192)
    assert captured["maximum_cost_usd"] > 2 * single


def _recording_genai(uploads):
    media = NS(name="files/test", uri="https://provider.test/file", state=NS(name="ACTIVE"))

    def upload(*, file, config):
        uploads.append((file, Path(file).read_bytes(), config))
        return media

    def generate(**_):
        return NS(status="completed", steps=[output('{"targets":[]}')])

    files = NS(upload=upload, get=lambda **_: media, delete=lambda **_: None)
    return NS(Client=lambda **_: NS(interactions=NS(create=generate), files=files))


def test_task_video_analysis_uploads_a_video_only_copy(tmp_path, media_tools):
    """Website free-beta privacy minimization: the site's video reaches the provider without sound."""
    source = tmp_path / "website-upload.mov"
    source.write_bytes(b"site-video-with-voices")
    uploads = []

    result = _invoke_agentic_video(api_key="fake", model=DEFAULT_MODEL, processing="static",
                                   video_path=source, genai=_recording_genai(uploads),
                                   types=NS(HttpOptions=NS, HttpRetryOptions=NS))

    [(uploaded_path, uploaded_bytes, config)] = uploads
    assert uploaded_path != str(source)
    assert uploaded_bytes == VIDEO_ONLY_MARKER + b"site-video-with-voices"
    assert config == {"mime_type": "video/quicktime"}
    assert "-an" in media_tools.ffmpeg_commands()[0]
    assert source.read_bytes() == b"site-video-with-voices"
    assert result["video_processing"]["upload_audio_removal"]["audio_removed"] is True


@pytest.mark.parametrize("tools", [lambda: FakeMediaTools(ffmpeg=False), lambda: FakeMediaTools(output_audio_streams=1)])
def test_task_video_analysis_uploads_nothing_without_a_video_only_copy(monkeypatch, tmp_path, tools):
    tools().install(monkeypatch)
    source = tmp_path / "website-upload.mov"
    source.write_bytes(b"site-video-with-voices")
    uploads = []

    with pytest.raises(AudioRemovalError) as error:
        _invoke_agentic_video(api_key="fake", model=DEFAULT_MODEL, processing="static",
                              video_path=source, genai=_recording_genai(uploads),
                              types=NS(HttpOptions=NS, HttpRetryOptions=NS))

    assert uploads == []
    assert _provider_error_blocker(error.value) == "gemini_clean_plate_upload_audio_removal_failed"


@pytest.mark.parametrize("lost_during", ["audio_copy", "upload", "processing_poll"])
def test_lease_loss_stops_next_provider_effect_but_still_deletes_uploaded_media(tmp_path, monkeypatch, lost_during):
    from contextlib import contextmanager
    from blueprint_pipeline import clean_plate_removal_analysis_gemini as analysis

    source = tmp_path / "source.mov"
    source.write_bytes(b"source video")
    lost = False
    calls = []

    def check(boundary):
        if lost:
            raise HandoffLeaseError("lost", boundary)

    @contextmanager
    def audio_copy(path):
        nonlocal lost
        if lost_during == "audio_copy":
            lost = True
        yield path, {"audio_removed": True}

    active = NS(name="files/existing", uri="https://provider.test/file", state=NS(name="ACTIVE"))

    def upload(**_kwargs):
        nonlocal lost
        calls.append("upload")
        if lost_during == "upload":
            lost = True
        return (NS(name=active.name, state=NS(name="PROCESSING"))
                if lost_during == "processing_poll" else active)

    def poll(**_kwargs):
        nonlocal lost
        calls.append("poll")
        lost = True
        return active

    def generate(**_kwargs):
        calls.append("generate")
        return NS(status="completed", steps=[output('{"targets":[]}')])

    files = NS(upload=upload, get=poll, delete=lambda **kwargs: calls.append(("delete", kwargs["name"])))
    genai = NS(Client=lambda **_: NS(files=files, interactions=NS(create=generate)))
    monkeypatch.setattr(analysis, "audio_free_upload_copy", audio_copy)
    monkeypatch.setattr(analysis.time, "sleep", lambda _seconds: None)
    with bind_handoff_lease(check), pytest.raises(HandoffLeaseError):
        _invoke_agentic_video(api_key="fake", model=DEFAULT_MODEL, processing="static",
            video_path=source, genai=genai, types=NS(HttpOptions=NS, HttpRetryOptions=NS))
    expected = [] if lost_during == "audio_copy" else ["upload"]
    if lost_during == "processing_poll":
        expected.append("poll")
    if lost_during != "audio_copy":
        expected.append(("delete", active.name))
    assert calls == expected


def test_lease_loss_cannot_start_another_paid_repair_turn():
    from blueprint_pipeline.clean_plate_removal_analysis_gemini import _invoke_text_repair
    calls = []
    genai = NS(Client=lambda **_: NS(interactions=NS(create=lambda **kwargs: calls.append(kwargs)
        or NS(status="completed", steps=[output('{"targets":[]}')]))))

    def lost(boundary):
        raise HandoffLeaseError("lost", boundary)

    with bind_handoff_lease(lost), pytest.raises(HandoffLeaseError):
        _invoke_text_repair(api_key="fake", model=DEFAULT_MODEL, genai=genai,
            types=NS(HttpOptions=NS, HttpRetryOptions=NS), prompt="Find targets.",
            previous_text="{}", error="removal_analysis_target_invalid")
    assert calls == []


@pytest.mark.parametrize("during_repair", [False, True])
def test_analysis_propagates_lease_loss_instead_of_retaining_a_blocked_model_answer(tmp_path, monkeypatch, during_repair):
    error = HandoffLeaseError("lost", "provider")
    with pytest.raises(HandoffLeaseError) as raised:
        _analysis(tmp_path, monkeypatch,
            first=[_door(), _rack()] if during_repair else error, repaired=error)
    assert raised.value is error
