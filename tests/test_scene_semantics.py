"""Tests for scene semantics inference."""

from __future__ import annotations

from pathlib import Path

from blueprint_pipeline import scene_semantics


def _make_frames(tmp_path: Path, count: int = 3) -> Path:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        (frames_dir / f"frame_{idx:05d}.jpg").write_bytes(b"fake-jpg")
    return frames_dir


def test_scene_semantics_explicit_hint_short_circuits_gemini(monkeypatch, tmp_path: Path) -> None:
    """Explicit hint is honored without invoking Gemini."""
    frames_dir = _make_frames(tmp_path)

    def _should_not_call_gemini(**_kwargs):
        raise AssertionError("Gemini should not be called for explicit hints")

    monkeypatch.setattr(scene_semantics, "_infer_with_gemini", _should_not_call_gemini)

    report = scene_semantics.infer_scene_semantics(
        frames_dir=frames_dir,
        requested_environment="warehouse",
    )
    assert report["resolved_environment"] == "warehouse"
    assert report["environment_source"] == "explicit_hint"
    assert report["prompt_source"] == "explicit_hint"
    assert report["environment_confidence"] == 1.0
    assert "shelf" in report["detection_prompts"]
    assert "bed" not in report["detection_prompts"]


def test_scene_semantics_gemini_success_for_auto(monkeypatch, tmp_path: Path) -> None:
    frames_dir = _make_frames(tmp_path)

    monkeypatch.setattr(
        scene_semantics,
        "_infer_with_gemini",
        lambda **_kwargs: scene_semantics._GeminiResult(
            environment="bedroom",
            confidence=0.91,
            model="gemini-3.0-pro",
            raw_text='{"room_type":"bedroom","confidence":0.91}',
            detected_objects=[],  # No objects enumerated — falls back to env prompts
        ),
    )

    report = scene_semantics.infer_scene_semantics(
        frames_dir=frames_dir,
        requested_environment="auto",
    )
    assert report["resolved_environment"] == "bedroom"
    assert report["environment_source"] == "gemini_video_inference"
    assert report["prompt_source"] == "gemini_video_inference"
    assert report["environment_confidence"] == 0.91
    assert report["explicit_hint"] is None


def test_scene_semantics_falls_back_when_gemini_unavailable(monkeypatch, tmp_path: Path) -> None:
    frames_dir = _make_frames(tmp_path)
    monkeypatch.setattr(scene_semantics, "_infer_with_gemini", lambda **_kwargs: None)

    report = scene_semantics.infer_scene_semantics(
        frames_dir=frames_dir,
        requested_environment="auto",
    )
    assert report["resolved_environment"] == "default"
    assert report["environment_source"] == "local_auto_fallback"
    assert report["prompt_source"] == "auto_fallback"
    assert report["fallback_reason"]

    out_path = tmp_path / "scene_semantics_report.json"
    scene_semantics.write_scene_semantics_report(out_path, report)
    assert out_path.is_file()


def test_scene_semantics_ignores_non_dict_detected_objects(monkeypatch, tmp_path: Path) -> None:
    """Non-dict Gemini object entries should be ignored instead of crashing."""
    frames_dir = _make_frames(tmp_path)

    monkeypatch.setattr(
        scene_semantics,
        "_infer_with_gemini",
        lambda **_kwargs: scene_semantics._GeminiResult(
            environment="warehouse",
            confidence=0.82,
            model="gemini-3.0-pro",
            raw_text='{"room_type":"warehouse","confidence":0.82}',
            detected_objects=[
                "not-a-dict",  # malformed Gemini item
                {"object_id": "tool_chest", "sam_prompt": "tool chest"},
            ],
        ),
    )

    report = scene_semantics.infer_scene_semantics(
        frames_dir=frames_dir,
        requested_environment="auto",
    )
    assert report["resolved_environment"] == "warehouse"
    assert report["prompt_source"] == "gemini_object_enumeration"
    assert report["detection_prompts"] == ["tool chest"]
