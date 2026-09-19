"""Actual provider request shape and fail-closed agentic trace; no live calls."""
from types import SimpleNamespace as NS

import pytest

from blueprint_pipeline.clean_plate_removal_analysis_gemini import (
    DEFAULT_MODEL,
    _invoke_agentic_video,
)


def invoke(tmp_path, parts, finish="STOP", processing="agentic"):
    path = tmp_path / "source.mov"
    path.write_bytes(b"source video")
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return NS(candidates=[NS(finish_reason=finish, content=NS(parts=parts))])

    genai = NS(Client=lambda **_: NS(models=NS(generate_content=generate)))
    types = NS(Part=NS, Blob=NS, GenerateContentConfig=NS, HttpOptions=NS, HttpRetryOptions=NS)
    result = _invoke_agentic_video(api_key="fake", model=DEFAULT_MODEL,
                                  processing=processing, video_path=path,
                                  genai=genai, types=types)
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
    part = calls[0]["contents"][0]
    assert part.media_processing == "AGENTIC"
    assert part.inline_data.data == b"source video"
    assert part.inline_data.mime_type == "video/quicktime"


def test_static_answer_does_not_masquerade_as_agentic(tmp_path):
    with pytest.raises(ValueError, match="agentic_trace_missing"):
        invoke(tmp_path, [NS(text='{"targets":[]}')])


def test_partial_json_not_accepted_as_completed_analysis(tmp_path):
    with pytest.raises(ValueError, match="analysis_incomplete"):
        invoke(tmp_path, [*trace(), NS(text="{}")], finish="MAX_TOKENS")


def test_static_override_is_rejected_before_provider_call(tmp_path):
    with pytest.raises(ValueError, match="agentic_processing_required"):
        invoke(tmp_path, [], processing="static")
