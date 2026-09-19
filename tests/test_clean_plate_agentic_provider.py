"""Actual provider request shape and fail-closed agentic trace; no live calls."""
from types import SimpleNamespace as NS

import pytest

from blueprint_pipeline.clean_plate_removal_analysis_gemini import (
    DEFAULT_MODEL,
    _invoke_agentic_video,
    parse_removal_plan_response,
)


def invoke(tmp_path, parts, finish="STOP", processing="agentic"):
    path = tmp_path / "source.mov"
    path.write_bytes(b"source video")
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return NS(candidates=[NS(finish_reason=finish, content=NS(parts=parts))])

    media = NS(name="files/test", uri="https://provider.test/file", state=NS(name="ACTIVE"))
    deleted = []
    files = NS(upload=lambda **_: media, delete=lambda **kwargs: deleted.append(kwargs))
    genai = NS(Client=lambda **_: NS(models=NS(generate_content=generate), files=files))
    types = NS(Part=NS(from_uri=lambda *, file_uri, mime_type: NS(
        file_data=NS(file_uri=file_uri, mime_type=mime_type))),
        GenerateContentConfig=NS, HttpOptions=NS, HttpRetryOptions=NS, ThinkingConfig=NS)
    try:
        result = _invoke_agentic_video(api_key="fake", model=DEFAULT_MODEL,
                                      processing=processing, video_path=path,
                                      genai=genai, types=types)
    finally:
        if processing == "agentic":
            assert deleted == [{"name": "files/test"}]
    return result, calls


def trace():
    return [NS(tool_call=NS(tool_type="MEDIA_PROCESSING")),
            NS(tool_response=NS(tool_type="MEDIA_PROCESSING"))]


def test_requests_agentic_38_and_filters_thoughts(tmp_path):
    result, calls = invoke(tmp_path, [*trace(), NS(thought=True, text="private"),
                                    NS(text='{"targets":[]}')])
    assert result["text"] == '{"targets":[]}'
    assert result["video_processing"]["media_tool_calls"] == 1
    assert result["video_processing"]["media_tool_responses"] == 1
    assert len(calls) == 1
    assert calls[0]["model"] == "gemini-3.8-flash"
    assert calls[0]["config"].thinking_config.thinking_level == "LOW"
    part = calls[0]["contents"][0]
    assert part.media_processing == "AGENTIC"
    assert part.file_data.file_uri == "https://provider.test/file"
    assert part.file_data.mime_type == "video/quicktime"


def test_static_answer_does_not_masquerade_as_agentic(tmp_path):
    with pytest.raises(ValueError, match="agentic_trace_missing"):
        invoke(tmp_path, [NS(text='{"targets":[]}')])


def test_partial_json_not_accepted_as_completed_analysis(tmp_path):
    with pytest.raises(ValueError, match="analysis_incomplete"):
        invoke(tmp_path, [*trace(), NS(text="{}")], finish="MAX_TOKENS")


def test_static_override_is_rejected_before_provider_call(tmp_path):
    with pytest.raises(ValueError, match="agentic_processing_required"):
        invoke(tmp_path, [], processing="static")


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
              "confidence": 0.9, "spatial_evidence": [{"timestamp_seconds": 2, "box_xywh_normalized": [0.2, 0.2, 0.2, 0.2]}]}
    parsed = parse_removal_plan_response(json.dumps({"targets": [target]}), strict=True, task_description="Move box to tray")
    assert parsed[0]["target_role"] == "destination"
    assert parsed[0]["disposition"] == "keep"
    assert parsed[0]["collision_required"] is True
