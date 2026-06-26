from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path


from blueprint_pipeline import rollout_vision_label_openai as vision


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_rollout_vision_private_helpers_parse_and_normalize(tmp_path: Path) -> None:
    assert vision._string(None) == ""
    assert vision._truthy(" YES ") is True
    assert vision._truthy("no") is False
    assert vision._mapping({"a": 1}) == {"a": 1}
    assert vision._mapping(["bad"]) == {}
    assert vision._read_mapping(tmp_path / "missing.json") == {}
    _write_json(tmp_path / "mapping.json", {"ok": True})
    assert vision._read_mapping(tmp_path / "mapping.json") == {"ok": True}
    _write_json(tmp_path / "list.json", [])
    assert vision._read_mapping(tmp_path / "list.json") == {}

    clips = vision._clip_by_attempt({"clips": ["bad", {"attempt_id": ""}, {"attempt_id": "a1", "clip_id": "clip"}]})
    assert clips == {"a1": {"attempt_id": "a1", "clip_id": "clip"}}
    assert vision._safe_stem(" ../bad id! ") == "bad-id"
    assert vision._safe_stem("!!!") == "attempt"

    image = tmp_path / "keyframe.png"
    image.write_bytes(b"png-bytes")
    assert vision._data_url(image).startswith("data:image/png;base64,")
    assert vision._parse_json_text('```json\n{"ok": true}\n```') == {"ok": True}
    assert vision._parse_json_text('prefix {"ok": 1} suffix') == {"ok": 1}
    assert vision._parse_json_text("[1, 2]") == {}


def test_rollout_vision_keyframe_extraction_edges(monkeypatch, tmp_path: Path) -> None:
    assert vision._extract_keyframe(output_dir=tmp_path, clip={}, clip_id="missing") == {
        "status": "blocked",
        "reason": "missing_clip_path",
        "path": None,
    }
    assert vision._extract_keyframe(output_dir=tmp_path, clip={"clip_path": "missing.mov"}, clip_id="missing") == {
        "status": "blocked",
        "reason": "clip_path_not_found",
        "path": str(tmp_path / "missing.mov"),
    }

    clip = tmp_path / "clip.mov"
    clip.write_bytes(b"video")

    def raise_not_found(*_args, **_kwargs):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(vision.subprocess, "run", raise_not_found)
    assert vision._extract_keyframe(output_dir=tmp_path, clip={"clip_path": str(clip)}, clip_id="clip 1")["reason"] == "missing_ffmpeg"

    def raise_timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("ffmpeg", 30)

    monkeypatch.setattr(vision.subprocess, "run", raise_timeout)
    assert vision._extract_keyframe(output_dir=tmp_path, clip={"clip_path": str(clip)}, clip_id="clip 1")["reason"] == "ffmpeg_timeout"

    monkeypatch.setattr(
        vision.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(["ffmpeg"], 1, stderr="bad frame"),
    )
    failed = vision._extract_keyframe(output_dir=tmp_path, clip={"clip_path": str(clip)}, clip_id="clip 1")
    assert failed["reason"] == "ffmpeg_failed"
    assert failed["stderr"] == "bad frame"

    def success_run(command, **_kwargs):
        Path(command[-1]).write_bytes(b"jpg")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(vision.subprocess, "run", success_run)
    completed = vision._extract_keyframe(output_dir=tmp_path, clip={"clip_path": str(clip)}, clip_id="clip 1")
    assert completed["status"] == "completed"
    assert completed["relative_path"] == "vision_keyframes/clip-1.jpg"


def test_rollout_vision_openai_and_fallback_label(monkeypatch, tmp_path: Path) -> None:
    keyframe = tmp_path / "frame.jpg"
    keyframe.write_bytes(b"jpg")

    openai_module = types.ModuleType("openai")

    class FakeResponses:
        def create(self, **kwargs):
            assert kwargs["model"] == "vision-model"
            content = kwargs["input"][0]["content"]
            assert content[0]["type"] == "input_text"
            assert content[1]["image_url"].startswith("data:image/jpeg;base64,")
            return types.SimpleNamespace(output_text='{"object_state": "open", "threshold_miss": true}')

    class FakeOpenAI:
        def __init__(self) -> None:
            self.responses = FakeResponses()

    openai_module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", openai_module)

    payload = vision._openai_label(
        model="vision-model",
        label={"label_id": "label-1", "attempt_id": "attempt-1", "failure_categories": ["contact"]},
        clip={"clip_id": "clip-1", "scenario_id": "scenario"},
        keyframe_path=keyframe,
    )
    assert payload == {"object_state": "open", "threshold_miss": True}

    fallback = vision._fallback_label(
        label={"label_id": "", "attempt_id": "attempt-1", "failure_categories": ["contact"], "threshold_miss": False},
        clip={"clip_path": "clips/attempt.mov"},
        keyframe={"status": "completed", "relative_path": "vision_keyframes/attempt.jpg"},
        model="vision-model",
        openai_payload={
            "object_state": "open",
            "contact": "none",
            "occlusion": "clear",
            "threshold_miss": True,
            "failure_evidence": ["visible miss"],
            "confidence": 0.8,
        },
    )
    assert fallback["vision_label_id"] == "vision_attempt-1"
    assert fallback["failure_evidence"] == ["visible miss"]
    assert fallback["confidence"] == 0.8
    assert fallback["evidence_refs"] == ["clips/attempt.mov", "vision_keyframes/attempt.jpg"]
    assert fallback["visual_evidence_used"] is True


def test_build_openai_rollout_vision_labels_blocked_and_completed(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv(vision.GATE_ENV, raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv(vision.MODEL_ENV, raising=False)

    blocked = vision.build_openai_rollout_vision_labels(output_dir=tmp_path)
    assert blocked["status"] == "blocked_review_required"
    assert blocked["blockers"] == sorted(
        ["missing_failure_labels", "missing_openai_api_key", f"missing_env_{vision.GATE_ENV}"]
    )
    assert (tmp_path / vision.OUTPUT_FILENAME).is_file()

    _write_json(
        tmp_path / "failure_labels.json",
        {"labels": [{"label_id": "label-1", "attempt_id": "attempt-1", "threshold_miss": True, "failure_categories": ["miss"]}]},
    )
    _write_json(tmp_path / "clips_manifest.json", {"clips": [{"attempt_id": "attempt-1", "clip_id": "clip-1", "clip_path": "clip.mov"}]})

    no_visual = vision.build_openai_rollout_vision_labels(output_dir=tmp_path, require_visual_evidence=True)
    assert "missing_visual_evidence_keyframes" in no_visual["blockers"]
    assert no_visual["label_count"] == 0

    skipped_for_blockers = vision.build_openai_rollout_vision_labels(
        output_dir=tmp_path,
        require_visual_evidence=False,
        max_labels=1,
    )
    assert skipped_for_blockers["keyframes"][0]["reason"] == "clip_path_not_found"
    assert skipped_for_blockers["label_count"] == 0

    monkeypatch.setenv(vision.GATE_ENV, "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv(vision.MODEL_ENV, "env-model")
    monkeypatch.setattr(
        vision,
        "_extract_keyframe",
        lambda **_kwargs: {"status": "completed", "path": str(tmp_path / "frame.jpg"), "relative_path": "frame.jpg"},
    )
    monkeypatch.setattr(
        vision,
        "_openai_label",
        lambda **_kwargs: {"object_state": "closed", "contact": "none", "threshold_miss": False},
    )

    completed = vision.build_openai_rollout_vision_labels(output_dir=tmp_path, max_labels=1)
    assert completed["status"] == "completed_review_required"
    assert completed["model"] == "env-model"
    assert completed["label_count"] == 1
    assert completed["labels"][0]["object_state"] == "closed"
    assert completed["visual_evidence_used"] is True


def test_rollout_vision_main_reports_status(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(
        vision,
        "build_openai_rollout_vision_labels",
        lambda **kwargs: {"status": "blocked_review_required", "blockers": ["missing"], "kwargs": kwargs},
    )
    assert vision.main(["--output-dir", str(tmp_path), "--model", "m", "--max-labels", "2", "--allow-missing-visual-evidence"]) == 1
    output = capsys.readouterr().out
    assert f"manifest={tmp_path.resolve() / vision.OUTPUT_FILENAME}" in output
    assert "status=blocked_review_required" in output
    assert "blockers=1" in output

    monkeypatch.setattr(
        vision,
        "build_openai_rollout_vision_labels",
        lambda **_kwargs: {"status": "completed_review_required", "blockers": []},
    )
    assert vision.main(["--output-dir", str(tmp_path)]) == 0
