from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from pathlib import Path

from blueprint_pipeline.scene_semantics import infer_capture_fidelity_review, infer_scene_semantics


def test_capture_fidelity_review_prefers_raw_video(monkeypatch, tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    frames_dir = capture_root / "frames"
    frames_dir.mkdir(parents=True)
    raw_video = capture_root / "walkthrough.mov"
    raw_video.write_bytes(b"video")

    calls: list[str] = []

    def fake_video(**_kwargs):
        calls.append("video")
        return {
            "status": "succeeded",
            "model": "gemini-video",
            "raw_text": "{}",
            "confidence": 0.9,
            "summary": "video ok",
            "scores": {
                "coverage": 0.9,
                "visual_clarity": 0.9,
                "lighting_stability": 0.9,
                "motion_stability": 0.9,
                "task_understanding": 0.9,
                "world_model_fitness": 0.9,
                "payout_quality": 0.9,
            },
            "bonus_signals": {
                "complete_coverage": {"score": 1.0, "reason": "video"},
                "multi_pass": {"score": 1.0, "reason": "video"},
                "lidar_depth": {"score": 1.0, "reason": "video"},
                "steady_walkthrough": {"score": 1.0, "reason": "video"},
            },
            "blur_assessment": {"status": "good", "score": 0.9, "summary": "sharp", "impact": "low"},
            "lighting_assessment": {"status": "good", "score": 0.9, "summary": "stable", "impact": "low"},
            "motion_speed_assessment": {"status": "good", "score": 0.9, "summary": "steady pace", "impact": "low"},
            "doubling_back_assessment": {"status": "good", "score": 0.9, "summary": "helpful revisits", "impact": "low"},
            "coverage_completeness_assessment": {"status": "good", "score": 0.95, "summary": "full coverage", "impact": "low"},
            "task_zone_completeness_assessment": {"status": "good", "score": 0.95, "summary": "task zone covered", "impact": "low"},
            "occlusion_and_hidden_zone_assessment": {"status": "good", "score": 0.9, "summary": "low occlusion", "impact": "low"},
            "depth_and_spatial_conditioning_assessment": {"status": "good", "score": 0.95, "summary": "strong depth", "impact": "low"},
            "missing_views": [],
            "blur_observations": [],
            "lighting_observations": [],
            "occlusion_observations": [],
            "task_scope_notes": [],
            "blocker_summaries": [],
            "recapture_recommendations": [],
            "world_model_recommendation": "good_candidate",
            "payout_recommendation": "bonus",
            "video_file_name": "files/123",
            "video_file_uri": "uri://123",
        }

    monkeypatch.setattr("blueprint_pipeline.scene_semantics._infer_capture_review_with_gemini_video", fake_video)

    review = infer_capture_fidelity_review(
        capture_root=capture_root,
        raw_video_path=raw_video,
        keyframe_path=None,
        descriptor={"capture_modality": "iphone_arkit_lidar"},
        qa_report={},
        task_hypothesis_report=None,
    )

    assert calls == ["video"]
    assert review["review_mode"] == "video_file_upload"
    assert review["provenance"]["video_file_name"] == "files/123"
    assert review["assessments"]["blur"]["status"] == "good"


def test_capture_fidelity_review_fails_without_video_success(monkeypatch, tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    capture_root.mkdir(parents=True)
    raw_video = capture_root / "walkthrough.mov"
    raw_video.write_bytes(b"video")

    calls: list[str] = []

    monkeypatch.setattr(
        "blueprint_pipeline.scene_semantics._infer_capture_review_with_gemini_video",
        lambda **_kwargs: calls.append("video") or None,
    )

    review = infer_capture_fidelity_review(
        capture_root=capture_root,
        raw_video_path=raw_video,
        keyframe_path=None,
        descriptor={"capture_modality": "iphone_arkit_lidar"},
        qa_report={},
        task_hypothesis_report=None,
    )

    assert calls == ["video"]
    assert review["status"] == "failed"
    assert review["review_mode"] == "video_file_upload"
    assert review["findings"]["blocker_summaries"] == ["Gemini raw-video review is unavailable or failed"]


def test_raw_video_review_polls_uploaded_file_until_active(monkeypatch, tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    capture_root.mkdir(parents=True)
    raw_video = capture_root / "walkthrough.mov"
    raw_video.write_bytes(b"video")

    class FakeClient:
        def __init__(self, *_args, **_kwargs) -> None:
            self.files = self
            self.models = self
            self.get_calls = 0
            self.generated = False

        def upload(self, *, file: str):
            assert file.endswith("walkthrough.mov")
            return SimpleNamespace(name="files/123", uri="uri://123", state=SimpleNamespace(name="PROCESSING"))

        def get(self, *, name: str):
            self.get_calls += 1
            assert name == "files/123"
            return SimpleNamespace(name="files/123", uri="uri://123", state=SimpleNamespace(name="ACTIVE"))

        def generate_content(self, *, model: str, contents, config):
            self.generated = True
            assert contents[0].fileData.fileUri == "uri://123"
            assert contents[0].videoMetadata.fps == 5.0
            assert isinstance(contents[1], str)
            return SimpleNamespace(text='{"summary":"ok","confidence":0.8,"scores":{"coverage":0.8},"bonus_signals":{"complete_coverage":{"score":0.8}}}')

    fake_client = FakeClient()
    fake_types = SimpleNamespace(
        FileData=lambda **kwargs: SimpleNamespace(**kwargs),
        VideoMetadata=lambda **kwargs: SimpleNamespace(**kwargs),
        Part=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    fake_genai = SimpleNamespace(Client=lambda api_key: fake_client, types=fake_types)
    fake_google = ModuleType("google")
    fake_google.genai = fake_genai

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)

    review = infer_capture_fidelity_review(
        capture_root=capture_root,
        raw_video_path=raw_video,
        keyframe_path=None,
        descriptor={"capture_modality": "iphone_arkit_lidar"},
        qa_report={},
        task_hypothesis_report=None,
    )

    assert fake_client.get_calls == 1
    assert fake_client.generated is True
    assert review["review_mode"] == "video_file_upload"
    assert review["provenance"]["video_analysis_fps"] == 5.0


def test_scene_semantics_prefers_raw_video_at_five_fps(monkeypatch, tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    frames_dir = capture_root / "frames"
    frames_dir.mkdir(parents=True)
    raw_video = capture_root / "walkthrough.mov"
    raw_video.write_bytes(b"video")

    calls: list[str] = []

    def fake_video(**_kwargs):
        calls.append("video")
        return SimpleNamespace(
            environment="warehouse",
            confidence=0.92,
            model="gemini-video",
            raw_text='{"room_type":"warehouse"}',
            detected_objects=[{"sam_prompt": "blue tote"}],
        )

    monkeypatch.setattr("blueprint_pipeline.scene_semantics._infer_with_gemini_video", fake_video)

    report = infer_scene_semantics(
        frames_dir=frames_dir,
        raw_video_path=raw_video,
        requested_environment="auto",
    )

    assert calls == ["video"]
    assert report["resolved_environment"] == "warehouse"
    assert report["gemini_inference_mode"] == "video_file_upload"
    assert report["gemini_video_analysis_fps"] == 5.0
    assert report["detection_prompts"] == ["blue tote"]


def test_scene_semantics_uses_local_fallback_when_video_fails(monkeypatch, tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    frames_dir = capture_root / "frames"
    frames_dir.mkdir(parents=True)
    raw_video = capture_root / "walkthrough.mov"
    raw_video.write_bytes(b"video")

    calls: list[str] = []

    monkeypatch.setattr(
        "blueprint_pipeline.scene_semantics._infer_with_gemini_video",
        lambda **_kwargs: calls.append("video") or None,
    )

    report = infer_scene_semantics(
        frames_dir=frames_dir,
        raw_video_path=raw_video,
        requested_environment="auto",
    )

    assert calls == ["video"]
    assert report["resolved_environment"] == "default"
    assert report["environment_source"] == "local_auto_fallback"
    assert report["fallback_reason"] == "gemini_video_unavailable_or_failed"
