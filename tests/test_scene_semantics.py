"""Tests for Gemini-first scene semantics inference."""

from __future__ import annotations

from pathlib import Path

from blueprint_pipeline import scene_semantics


def _make_frames(tmp_path: Path, count: int = 3) -> Path:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        (frames_dir / f"frame_{idx:05d}.jpg").write_bytes(b"fake-jpg")
    return frames_dir


def test_scene_semantics_gemini_overrides_explicit_hint(monkeypatch, tmp_path: Path) -> None:
    """Gemini always runs, even when an explicit environment is requested."""
    frames_dir = _make_frames(tmp_path)

    monkeypatch.setattr(
        scene_semantics,
        "_infer_with_gemini",
        lambda **_kwargs: scene_semantics._GeminiResult(
            environment="bedroom",
            confidence=0.95,
            model="gemini-3.0-pro",
            raw_text='{"room_type":"bedroom","confidence":0.95}',
            detected_objects=[
                {"object_id": "wooden_bed", "category": "Furniture", "sam_prompt": "wooden bed"},
                {"object_id": "blue_suitcase", "category": "Container", "sam_prompt": "blue suitcase"},
            ],
        ),
    )

    # Request warehouse explicitly, but Gemini sees bedroom
    report = scene_semantics.infer_scene_semantics(
        frames_dir=frames_dir,
        requested_environment="warehouse",
    )
    assert report["resolved_environment"] == "bedroom"
    assert report["environment_source"] == "gemini_video_inference"
    assert report["prompt_source"] == "gemini_object_enumeration"
    assert report["explicit_hint"] == "warehouse"
    assert "wooden bed" in report["detection_prompts"]
    assert "blue suitcase" in report["detection_prompts"]


def test_scene_semantics_explicit_hint_fallback_when_gemini_unavailable(monkeypatch, tmp_path: Path) -> None:
    """When Gemini is unavailable, explicit hint is used as fallback."""
    frames_dir = _make_frames(tmp_path)
    monkeypatch.setattr(scene_semantics, "_infer_with_gemini", lambda **_kwargs: None)

    report = scene_semantics.infer_scene_semantics(
        frames_dir=frames_dir,
        requested_environment="bedroom",
    )
    assert report["resolved_environment"] == "bedroom"
    assert report["environment_source"] == "explicit_hint_fallback"
    assert report["prompt_source"] == "explicit_hint_fallback"
    assert report["environment_confidence"] == 0.7
    assert "bed" in report["detection_prompts"]
    assert "gemini_unavailable" in report["fallback_reason"]


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
