from __future__ import annotations

import builtins
import json
import sys
from types import ModuleType, SimpleNamespace
from pathlib import Path

from blueprint_pipeline import scene_semantics as semantics
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


def _fake_genai_module(client):
    return SimpleNamespace(
        Client=lambda api_key: client,
        types=SimpleNamespace(
            FileData=lambda **kwargs: SimpleNamespace(**kwargs),
            VideoMetadata=lambda **kwargs: SimpleNamespace(**kwargs),
            Part=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    )


def test_scene_semantics_edge_branches(monkeypatch, tmp_path: Path) -> None:
    video = tmp_path / "walkthrough.mov"
    video.write_bytes(b"video")
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()

    assert semantics._normalize_environment("") == "default"
    assert semantics._normalize_environment("factory") == "manufacturing"
    assert semantics._extract_json_object("no json here") == {}
    assert semantics._extract_json_object('prefix {"room_type":"warehouse"} suffix') == {
        "room_type": "warehouse"
    }
    assert semantics._extract_json_object("prefix {bad json} suffix") == {}

    assert semantics._extract_response_text(SimpleNamespace(candidates=None)) == ""
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(content=SimpleNamespace(parts="not-list")),
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(thought=True, text="hidden reasoning"),
                        SimpleNamespace(text="final text"),
                    ]
                )
            ),
        ]
    )
    assert semantics._extract_response_text(response) == "final text"
    assert (
        semantics._extract_response_text(
            SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[]))])
        )
        == ""
    )

    monkeypatch.setenv("GEMINI_VIDEO_ANALYSIS_FPS", "not-a-number")
    assert semantics._gemini_video_analysis_fps() == 5.0
    monkeypatch.setenv("GEMINI_VIDEO_ANALYSIS_FPS", "99")
    assert semantics._gemini_video_analysis_fps() == 24.0

    class UploadRaises:
        files = None

        def __init__(self) -> None:
            self.files = self

        def upload(self, *, file: str):
            raise RuntimeError(file)

    assert semantics._upload_gemini_video_file(UploadRaises(), video, 5) is None

    class UploadFailed:
        files = None

        def __init__(self) -> None:
            self.files = self

        def upload(self, *, file: str):
            return SimpleNamespace(name="files/1", state=SimpleNamespace(name="FAILED"))

    assert semantics._upload_gemini_video_file(UploadFailed(), video, 5) is None

    class UploadTimeout:
        files = None

        def __init__(self) -> None:
            self.files = self

        def upload(self, *, file: str):
            return SimpleNamespace(name="files/1", state=SimpleNamespace(name="PROCESSING"))

        def get(self, *, name: str):
            raise AssertionError("timeout should happen before polling")

    times = iter([0.0, 6.0])
    monkeypatch.setattr(semantics.time, "time", lambda: next(times))
    assert semantics._upload_gemini_video_file(UploadTimeout(), video, 1) is None

    class UploadGetRaises:
        files = None

        def __init__(self) -> None:
            self.files = self

        def upload(self, *, file: str):
            return SimpleNamespace(name="files/1", state=SimpleNamespace(name="PROCESSING"))

        def get(self, *, name: str):
            raise RuntimeError(name)

    times = iter([0.0, 1.0])
    monkeypatch.setattr(semantics.time, "time", lambda: next(times))
    monkeypatch.setattr(semantics.time, "sleep", lambda *_args, **_kwargs: None)
    assert semantics._upload_gemini_video_file(UploadGetRaises(), video, 10) is None

    assert semantics._extract_json_array("```json\n[{\"a\": 1}]\n```") == [{"a": 1}]
    assert semantics._extract_json_array('{"items": [1, 2]}') == [1, 2]
    assert semantics._extract_json_array("prefix [1, 2] suffix") == [1, 2]
    assert semantics._extract_json_array("prefix [bad] suffix") == []

    report_path = tmp_path / "reports" / "scene.json"
    semantics.write_scene_semantics_report(report_path, {"ok": True})
    assert json.loads(report_path.read_text(encoding="utf-8")) == {"ok": True}
    assert semantics._string_list("a") == ["a"]
    assert semantics._string_list(["a", "a", " b "]) == ["a", "b"]
    assert "Capture context" in semantics._gemini_capture_review_prompt(
        descriptor={},
        qa_report={},
        task_hypothesis_report=None,
        capture_context={"site_id": "site-1"},
    )

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_API_KEY", raising=False)
    assert semantics._infer_with_gemini_video(raw_video_path=video, timeout_sec=5) is None
    assert (
        semantics._infer_capture_review_with_gemini_video(
            raw_video_path=video,
            descriptor={},
            qa_report={},
            task_hypothesis_report=None,
            capture_context=None,
            timeout_sec=5,
        )
        is None
    )

    explicit = infer_scene_semantics(
        frames_dir=frames_dir,
        raw_video_path=None,
        requested_environment="bed room",
    )
    assert explicit["environment_source"] == "explicit_hint"
    assert explicit["resolved_environment"] == "bedroom"

    missing_video = infer_scene_semantics(
        frames_dir=frames_dir,
        raw_video_path=None,
        requested_environment="auto",
    )
    assert missing_video["fallback_reason"] == "raw_walkthrough_video_missing"

    monkeypatch.setattr(
        semantics,
        "_infer_with_gemini_video",
        lambda **_kwargs: semantics._GeminiResult(
            environment="kitchen",
            confidence=0.4,
            model="gemini",
            raw_text="{}",
            detected_objects=["skip", {"sam_prompt": ""}],
        ),
    )
    fallback_prompts = infer_scene_semantics(
        frames_dir=frames_dir,
        raw_video_path=video,
        requested_environment="auto",
    )
    assert fallback_prompts["prompt_source"] == "gemini_object_enumeration"
    assert fallback_prompts["detection_prompts"] == semantics._PROMPTS_BY_ENV["kitchen"]

    monkeypatch.setattr(
        semantics,
        "_infer_with_gemini_video",
        lambda **_kwargs: semantics._GeminiResult(
            environment="warehouse",
            confidence=0.6,
            model="gemini",
            raw_text="{}",
            detected_objects=[{"object_id": 42}],
        ),
    )
    non_string_prompt = infer_scene_semantics(
        frames_dir=frames_dir,
        raw_video_path=video,
        requested_environment="auto",
    )
    assert non_string_prompt["detection_prompts"] == ["42"]

    monkeypatch.setattr(
        semantics,
        "_infer_with_gemini_video",
        lambda **_kwargs: semantics._GeminiResult(
            environment="warehouse",
            confidence=0.6,
            model="gemini",
            raw_text="{}",
            detected_objects=[],
        ),
    )
    no_objects = infer_scene_semantics(
        frames_dir=frames_dir,
        raw_video_path=video,
        requested_environment="auto",
    )
    assert no_objects["prompt_source"] == "gemini_video_inference"

    no_video_review = infer_capture_fidelity_review(
        capture_root=tmp_path,
        raw_video_path=None,
        keyframe_path=None,
        descriptor={},
        qa_report={},
    )
    assert no_video_review["findings"]["blocker_summaries"] == [
        "raw walkthrough video is missing"
    ]


def test_scene_semantics_gemini_import_and_model_edges(monkeypatch, tmp_path: Path) -> None:
    video = tmp_path / "walkthrough.mov"
    video.write_bytes(b"video")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    fake_google = ModuleType("google")
    monkeypatch.setitem(sys.modules, "google", fake_google)
    assert semantics._infer_with_gemini_video(raw_video_path=video, timeout_sec=5) is None
    assert (
        semantics._infer_capture_review_with_gemini_video(
            raw_video_path=video,
            descriptor={},
            qa_report={},
            task_hypothesis_report=None,
            capture_context=None,
            timeout_sec=5,
        )
        is None
    )

    original_import = builtins.__import__

    def raise_on_google(name, *args, **kwargs):
        if name == "google":
            raise RuntimeError("import exploded")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as import_patch:
        import_patch.setattr(builtins, "__import__", raise_on_google)
        assert semantics._infer_with_gemini_video(raw_video_path=video, timeout_sec=5) is None
        assert (
            semantics._infer_capture_review_with_gemini_video(
                raw_video_path=video,
                descriptor={},
                qa_report={},
                task_hypothesis_report=None,
                capture_context=None,
                timeout_sec=5,
            )
            is None
        )

    class ActiveFiles:
        def upload(self, *, file: str):
            return SimpleNamespace(
                name="files/1",
                uri="uri://1",
                mime_type="video/mp4",
                state=SimpleNamespace(name="ACTIVE"),
            )

    class UploadRaisingFiles:
        def upload(self, *, file: str):
            raise RuntimeError(file)

    class UploadNoneClient:
        files = UploadRaisingFiles()
        models = SimpleNamespace(generate_content=lambda **_kwargs: None)

    fake_google.genai = _fake_genai_module(UploadNoneClient())
    assert semantics._infer_with_gemini_video(raw_video_path=video, timeout_sec=5) is None
    assert (
        semantics._infer_capture_review_with_gemini_video(
            raw_video_path=video,
            descriptor={},
            qa_report={},
            task_hypothesis_report=None,
            capture_context=None,
            timeout_sec=5,
        )
        is None
    )

    class RaisingModelsClient:
        files = ActiveFiles()

        class models:
            @staticmethod
            def generate_content(**_kwargs):
                raise RuntimeError("model unavailable")

    fake_google.genai = _fake_genai_module(RaisingModelsClient())
    assert semantics._infer_with_gemini_video(raw_video_path=video, timeout_sec=5) is None
    assert (
        semantics._infer_capture_review_with_gemini_video(
            raw_video_path=video,
            descriptor={},
            qa_report={},
            task_hypothesis_report=None,
            capture_context=None,
            timeout_sec=5,
        )
        is None
    )

    class EmptyTextClient:
        files = ActiveFiles()

        class models:
            @staticmethod
            def generate_content(**_kwargs):
                return SimpleNamespace(text="")

    fake_google.genai = _fake_genai_module(EmptyTextClient())
    assert semantics._infer_with_gemini_video(raw_video_path=video, timeout_sec=5) is None
    assert (
        semantics._infer_capture_review_with_gemini_video(
            raw_video_path=video,
            descriptor={},
            qa_report={},
            task_hypothesis_report=None,
            capture_context=None,
            timeout_sec=5,
        )
        is None
    )

    class NonJsonReviewClient:
        files = ActiveFiles()

        class models:
            @staticmethod
            def generate_content(**_kwargs):
                return SimpleNamespace(text="not json")

    fake_google.genai = _fake_genai_module(NonJsonReviewClient())
    assert (
        semantics._infer_capture_review_with_gemini_video(
            raw_video_path=video,
            descriptor={},
            qa_report={},
            task_hypothesis_report=None,
            capture_context=None,
            timeout_sec=5,
        )
        is None
    )

    class SuccessClient:
        files = ActiveFiles()

        class models:
            @staticmethod
            def generate_content(**_kwargs):
                return SimpleNamespace(
                    text=json.dumps(
                        {
                            "environment": "factory",
                            "confidence": "not-a-number",
                            "objects": [{"sam_prompt": "red bin"}, "skip"],
                        }
                    )
                )

    fake_google.genai = _fake_genai_module(SuccessClient())
    result = semantics._infer_with_gemini_video(raw_video_path=video, timeout_sec=5)
    assert result is not None
    assert result.environment == "manufacturing"
    assert result.confidence == 0.0
    assert result.detected_objects == [{"sam_prompt": "red bin"}]


def test_scene_semantics_retries_transient_and_deletes_uploaded_file(monkeypatch, tmp_path: Path) -> None:
    video = tmp_path / "walkthrough.mov"
    video.write_bytes(b"video")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(semantics.time, "sleep", lambda *_args, **_kwargs: None)

    class RetryClient:
        def __init__(self) -> None:
            self.files = self
            self.models = self
            self.calls = 0
            self.deleted: list[str] = []

        def upload(self, *, file: str):
            return SimpleNamespace(
                name="files/retry",
                uri="uri://retry",
                mime_type="video/mp4",
                state=SimpleNamespace(name="ACTIVE"),
            )

        def delete(self, *, name: str):
            self.deleted.append(name)

        def generate_content(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("RESOURCE_EXHAUSTED 429")
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "room_type": "kitchen",
                        "confidence": True,
                        "objects": [{"sam_prompt": "fridge handle"}],
                    }
                )
            )

    client = RetryClient()
    fake_google = ModuleType("google")
    fake_google.genai = _fake_genai_module(client)
    monkeypatch.setitem(sys.modules, "google", fake_google)

    result = semantics._infer_with_gemini_video(raw_video_path=video, timeout_sec=5)

    assert result is not None
    assert result.environment == "kitchen"
    assert result.confidence == 0.0
    assert client.calls == 2
    assert client.deleted == ["files/retry"]
