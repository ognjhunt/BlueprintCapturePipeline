from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline.capture_bridge import CaptureDescriptor
from blueprint_pipeline.ios_manifest import IOSManifest
from blueprint_pipeline.local_capture import LocalCaptureContext
from blueprint_pipeline import object_index_stage as oi
from blueprint_pipeline.object_index_stage import _existing_index_is_reusable


def _descriptor(raw_prefix_uri: str = "gs://bucket/scenes/scene/captures/capture/raw") -> CaptureDescriptor:
    return CaptureDescriptor(
        schema_version="v1",
        scene_id="scene",
        capture_id="capture",
        capture_source="iphone",
        capture_tier="external_alpha",
        raw_prefix_uri=raw_prefix_uri,
        frames_index_uri=f"{raw_prefix_uri}/arkit/frames.jsonl",
        environment_type_hint="warehouse",
        swap_focus=["kitchen"],
        metadata={
            "task_statement": "Open the cabinet and move the tote to the shelf",
            "workflow_context": "inventory organize workflow",
            "task_zone": {"label": "warehouse aisle"},
            "privacy_restrictions": ["faces_blurred"],
            "success_criteria": ["cabinet opened"],
            "known_blockers": ["tight aisle"],
        },
    )


def _manifest(video_uri: str = "") -> IOSManifest:
    return IOSManifest.from_dict(
        {
            "scene_id": "scene",
            "video_uri": video_uri,
            "intended_space_type": "office",
            "width": 640,
            "height": 480,
        }
    )


def _keyframe(tmp_path: Path, frame_index: int = 0, *, translation: list[float] | None = None) -> oi._Keyframe:
    image_path = tmp_path / f"frame_{frame_index}.png"
    oi._ensure_png(image_path)
    return oi._Keyframe(
        frame_index=frame_index,
        timestamp=float(frame_index),
        image_width=100,
        image_height=80,
        image_path=image_path,
        intrinsics=[1.0, 2.0],
        camera_translation=translation or [0.0, 0.0, 0.0],
        motion_score=0.1,
    )


def _capture_tree(tmp_path: Path) -> tuple[Path, LocalCaptureContext]:
    capture_root = tmp_path / "storage" / "bucket" / "scenes" / "scene" / "captures" / "capture"
    raw_root = capture_root / "raw"
    raw_root.mkdir(parents=True)
    context = LocalCaptureContext(
        capture_root=capture_root,
        raw_root=raw_root,
        pipeline_root=capture_root / "pipeline",
        descriptor_path=capture_root / "capture_descriptor.json",
        raw_complete_path=raw_root / "capture_upload_complete.json",
        storage_root=tmp_path / "storage",
        bucket="bucket",
        scene_id="scene",
        capture_id="capture",
    )
    oi.write_json(context.descriptor_path, _descriptor(context.raw_prefix_uri).to_dict())
    oi.write_json(
        raw_root / "manifest.json",
        {
            "scene_id": "scene",
            "video_uri": "walkthrough.mp4",
            "intended_space_type": "warehouse",
            "width": 640,
            "height": 480,
        },
    )
    return capture_root, context


def test_existing_index_is_not_reused_when_empty_and_runtime_was_missing() -> None:
    reusable = _existing_index_is_reusable(
        loaded=[],
        report={
            "status": "built",
            "object_count": 0,
            "empty_index_cause": "runtime_missing",
            "runtime_preflight": {
                "backends": {
                    "yolo_world": {
                        "support_level": "required",
                        "status": "runtime_missing",
                    }
                }
            },
        },
    )

    assert reusable is False


def test_existing_index_is_reused_when_zero_objects_were_a_real_result() -> None:
    reusable = _existing_index_is_reusable(
        loaded=[],
        report={
            "status": "built",
            "object_count": 0,
            "empty_index_cause": "zero_detections",
            "runtime_preflight": {
                "backends": {
                    "yolo_world": {
                        "support_level": "required",
                        "status": "configured",
                    }
                }
            },
        },
    )

    assert reusable is True


def test_object_index_basic_file_keyframe_and_prompt_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert oi._safe_float("3.5") == 3.5
    assert oi._safe_float(object(), default=7.0) == 7.0
    assert oi._safe_int("4") == 4
    assert oi._safe_int(object(), default=9) == 9
    assert oi._string_list(" door ", ["door", " shelf "], {"ignored": True}, ("bin",)) == ["door", "shelf", "bin"]
    assert oi._slug("  Shelf / Door!! ") == "shelf_door"
    assert oi._slug("!!!") == "object"

    assert oi._optional_json(tmp_path / "missing.json") == {}
    non_mapping = tmp_path / "list.json"
    non_mapping.write_text("[1, 2]", encoding="utf-8")
    assert oi._optional_json(non_mapping) == {}
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text('\n{"a": 1}\nnot-json\n[1]\n{"b": 2}\n', encoding="utf-8")
    assert oi._jsonl(jsonl) == [{"a": 1}, {"b": 2}]
    assert oi._jsonl(tmp_path / "missing.jsonl") == []

    context = SimpleNamespace(raw_root=tmp_path, storage_root=tmp_path / "storage")
    for rel in ("walkthrough.mov", "walkthrough.mp4", "recording.mov", "recording.mp4"):
        for existing in tmp_path.glob("*.mov"):
            existing.unlink()
        for existing in tmp_path.glob("*.mp4"):
            existing.unlink()
        candidate = tmp_path / rel
        candidate.write_bytes(b"video")
        assert oi._resolve_video_path(context, _manifest()) == candidate
        candidate.unlink()
    relative_video = tmp_path / "nested.mp4"
    relative_video.write_bytes(b"video")
    assert oi._resolve_video_path(context, _manifest("nested.mp4")) == relative_video
    monkeypatch.setattr(oi, "resolve_gs_uri_to_path", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad gs")))
    assert oi._resolve_video_path(context, _manifest("gs://bucket/video.mp4")) is None
    assert oi._resolve_video_path(context, _manifest("missing.mp4")) is None

    assert oi._translation_from_matrix(list(range(16))) == [12.0, 13.0, 14.0]
    assert oi._translation_from_matrix([[0, 0], [0, 0, 5], [0], []]) == [0.0, 0.0, 0.0]
    assert oi._translation_from_matrix([[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, 0, 1]]) == [1.0, 2.0, 3.0]
    assert oi._translation_from_matrix("bad") == [0.0, 0.0, 0.0]
    assert oi._nearest_motion_score(1.0, []) == 0.0
    assert oi._nearest_motion_score(
        1.1,
        [
            {"timestamp": 1.0, "rotationRate": {"x": 3}, "userAcceleration": {"y": 4}},
            {"timestamp": 9.0, "rotationRate": {"x": 99}},
        ],
    ) == 7.0
    prompt_bank = oi._build_prompt_bank(
        _descriptor(),
        {"taskSteps": ["turn on the sink right handle"]},
        {},
        "kitchen",
    )
    assert "sink right handle" in prompt_bank["task_specific"]
    assert "right handle" in prompt_bank["task_specific"]
    assert "water stream" in prompt_bank["task_specific"]

    raw_root = tmp_path / "capture" / "raw"
    arkit_root = raw_root / "arkit"
    arkit_root.mkdir(parents=True)
    (arkit_root / "frames.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "frameIndex": index,
                    "timestamp": float(index),
                    "imageResolution": [320 + index, 240 + index],
                    "intrinsics": [index, index + 1],
                    "cameraTransform": list(range(16)),
                }
            )
            for index in range(4)
        ),
        encoding="utf-8",
    )
    (arkit_root / "poses.jsonl").write_text(
        json.dumps({"frameIndex": 2, "transform": [[0, 0, 0, 7], [0, 0, 0, 8], [0, 0, 0, 9], [0, 0, 0, 1]]}),
        encoding="utf-8",
    )
    (raw_root / "motion.jsonl").write_text(
        json.dumps({"timestamp": 0.0, "rotationRate": {"x": 1}, "userAcceleration": {"z": 2}}),
        encoding="utf-8",
    )
    keyframes = oi._sample_keyframes(
        context=SimpleNamespace(raw_root=raw_root, raw_prefix_uri="gs://bucket/raw", storage_root=tmp_path),
        max_keyframes=2,
        artifact_dir=tmp_path / "artifacts",
    )
    assert len(keyframes) == 2
    assert keyframes[0].image_width == 320
    assert keyframes[0].motion_score == pytest.approx(3.0)

    empty_raw = tmp_path / "empty" / "raw"
    empty_raw.mkdir(parents=True)
    (tmp_path / "video.mp4").write_bytes(b"video")
    monkeypatch.setattr(oi, "load_raw_manifest", lambda *_args, **_kwargs: _manifest("video.mp4"))
    monkeypatch.setattr(oi, "_resolve_video_path", lambda *_args, **_kwargs: tmp_path / "video.mp4")
    monkeypatch.setattr(oi, "_ffprobe_duration_seconds", lambda _path: 6.0)
    video_keyframes = oi._sample_keyframes(
        context=SimpleNamespace(raw_root=empty_raw, raw_prefix_uri="gs://bucket/raw", storage_root=tmp_path),
        max_keyframes=2,
        artifact_dir=tmp_path / "video-keyframes",
    )
    assert [item.timestamp for item in video_keyframes] == [2.0, 4.0]
    monkeypatch.setattr(oi, "_ffprobe_duration_seconds", lambda _path: 0.0)
    zero_duration_keyframes = oi._sample_keyframes(
        context=SimpleNamespace(raw_root=empty_raw, raw_prefix_uri="gs://bucket/raw", storage_root=tmp_path),
        max_keyframes=2,
        artifact_dir=tmp_path / "zero-video-keyframes",
    )
    assert [item.timestamp for item in zero_duration_keyframes] == [0.0]
    monkeypatch.setattr(oi, "_resolve_video_path", lambda *_args, **_kwargs: None)
    assert oi._sample_keyframes(
        context=SimpleNamespace(raw_root=empty_raw, raw_prefix_uri="gs://bucket/raw", storage_root=tmp_path),
        max_keyframes=2,
        artifact_dir=tmp_path / "no-video-keyframes",
    ) == []

    descriptor = _descriptor()
    assert oi._infer_environment(descriptor, _manifest()) == "warehouse"
    assert oi._infer_environment(
        CaptureDescriptor(**{**descriptor.__dict__, "environment_type_hint": "kitchen prep", "swap_focus": []}),
        _manifest(),
    ) == "kitchen"
    assert oi._infer_environment(
        CaptureDescriptor(**{**descriptor.__dict__, "environment_type_hint": "bedroom", "swap_focus": []}),
        _manifest(),
    ) == "bedroom"
    assert oi._infer_environment(
        CaptureDescriptor(**{**descriptor.__dict__, "environment_type_hint": "", "swap_focus": []}),
        _manifest(),
    ) == "office"
    assert oi._infer_environment(
        CaptureDescriptor(
            **{
                **descriptor.__dict__,
                "environment_type_hint": "",
                "swap_focus": [],
                "metadata": {"task_zone": 7},
            }
        ),
        IOSManifest.from_dict({"scene_id": "scene", "video_uri": "", "intended_space_type": "unknown"}),
    ) == "default"
    prompt_bank = oi._build_prompt_bank(
        descriptor,
        {"workflowName": "open drawer", "taskSteps": ["inventory tote"]},
        {"captureSource": "iphone"},
        "warehouse",
    )
    assert "drawer" in prompt_bank["task_specific"]
    assert "rack" in prompt_bank["all"]
    assert oi._maybe_expand_prompt_bank(
        runner=None,
        descriptor=descriptor,
        intake={},
        capture_context={},
        prompt_bank=prompt_bank,
    ) == (prompt_bank, None)

    calls: list[tuple[str, dict[str, object]]] = []

    def runner(name: str, payload: dict[str, object]) -> dict[str, object]:
        calls.append((name, payload))
        return {"additional_prompts": ["valve", "drawer"]}

    expanded, response = oi._maybe_expand_prompt_bank(
        runner=runner,
        descriptor=descriptor,
        intake={"workflowName": "inspect"},
        capture_context={"captureSource": "app"},
        prompt_bank=prompt_bank,
    )
    assert calls[0][0] == "prompt_bank_expander"
    assert "valve" in expanded["all"]
    assert response == {"additional_prompts": ["valve", "drawer"]}
    assert oi._maybe_expand_prompt_bank(
        runner=lambda *_args: ["bad"],
        descriptor=descriptor,
        intake={},
        capture_context={},
        prompt_bank=prompt_bank,
    )[1] is None


def test_run_object_index_stage_all_backends_skipped_emits_empty_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_root, context = _capture_tree(tmp_path)
    arkit_root = context.raw_root / "arkit"
    arkit_root.mkdir(parents=True)
    (arkit_root / "frames.jsonl").write_text(
        json.dumps(
            {
                "frameIndex": 0,
                "timestamp": 0.0,
                "imageResolution": [16, 12],
                "intrinsics": [1.0, 1.0],
                "cameraTransform": list(range(16)),
            }
        ),
        encoding="utf-8",
    )
    for name in (
        "OBJECT_INDEX_YOLO_WORLD_COMMAND",
        "OBJECT_INDEX_GROUNDING_DINO_COMMAND",
        "OBJECT_INDEX_SAM3_COMMAND",
        "OBJECT_INDEX_SPLAT_ANALYZER_COMMAND",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(oi, "resolve_local_capture_context", lambda _capture_root: context)
    monkeypatch.setattr(oi, "_command_from_env", lambda _name: "")

    result = oi.run_object_index_stage(capture_root=capture_root, force_rebuild=True)
    build_report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))

    assert result["status"] == "built"
    assert result["object_count"] == 0
    assert build_report["empty_index_cause"] == "backend_skipped"
    assert Path(result["manifest_path"]).is_file()
    assert Path(result["report_path"]).is_file()
    grounding_hints = context.raw_root / "object_grounding_hints.json"
    assert grounding_hints.is_file()
    grounding_payload = json.loads(grounding_hints.read_text(encoding="utf-8"))
    assert grounding_payload["grounded_objects"] == []
    assert grounding_payload["manipulation_candidates"] == []
    assert grounding_payload["articulation_hints"] == []
    assert grounding_payload["tasks"] == []


def test_object_index_subprocess_backend_and_detection_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run_success(command, **_kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="12.5\n", stderr="")

    monkeypatch.setattr(oi.subprocess, "run", run_success)
    assert oi._ffprobe_duration_seconds(tmp_path / "video.mp4") == 12.5
    monkeypatch.setattr(oi.subprocess, "run", lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, stdout="", stderr="bad"))
    assert oi._ffprobe_duration_seconds(tmp_path / "video.mp4") == 0.0
    monkeypatch.setattr(oi.subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("missing")))
    assert oi._ffprobe_duration_seconds(tmp_path / "video.mp4") == 0.0

    frame = tmp_path / "frame.png"
    oi._ensure_png(frame)
    assert frame.is_file()
    existing = oi._Keyframe(0, 0.0, 10, 10, frame, [], [0.0, 0.0, 0.0], 0.0)
    oi._extract_keyframe_images(None, [existing])
    missing_frame = tmp_path / "missing.png"
    oi._extract_keyframe_images(None, [oi._Keyframe(1, 0.0, 10, 10, missing_frame, [], [0.0, 0.0, 0.0], 0.0)])
    assert missing_frame.is_file()
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    ffmpeg_frame = tmp_path / "ffmpeg.png"
    monkeypatch.setattr(oi.subprocess, "run", lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, stdout="", stderr="bad"))
    oi._extract_keyframe_images(video, [oi._Keyframe(2, -1.0, 10, 10, ffmpeg_frame, [], [0.0, 0.0, 0.0], 0.0)])
    assert ffmpeg_frame.is_file()
    os_error_frame = tmp_path / "os-error.png"
    monkeypatch.setattr(oi.subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("missing")))
    oi._extract_keyframe_images(video, [oi._Keyframe(3, 1.0, 10, 10, os_error_frame, [], [0.0, 0.0, 0.0], 0.0)])
    assert os_error_frame.is_file()

    monkeypatch.setenv("OBJECT_INDEX_YOLO_WORLD_COMMAND", "custom {INPUT_JSON}")
    assert oi._command_from_env("OBJECT_INDEX_YOLO_WORLD_COMMAND") == "custom {INPUT_JSON}"
    monkeypatch.delenv("OBJECT_INDEX_YOLO_WORLD_COMMAND")
    assert "object_index_yolo_world_runner.py" in oi._command_from_env("OBJECT_INDEX_YOLO_WORLD_COMMAND")
    assert "object_index_splat_analyzer_runner.py" in oi._command_from_env("OBJECT_INDEX_SPLAT_ANALYZER_COMMAND")
    assert oi._command_from_env("UNKNOWN_COMMAND") == ""
    assert oi._module_available("json") is True
    monkeypatch.setattr(oi.importlib.util, "find_spec", lambda _name: (_ for _ in ()).throw(RuntimeError("boom")))
    assert oi._module_available("json") is False
    monkeypatch.setenv("SAM3_WEIGHTS_PATH", str(tmp_path / "sam.pt"))
    assert oi._default_sam3_weights_path() == tmp_path / "sam.pt"

    monkeypatch.setattr(oi, "_module_available", lambda name: name == "torch")
    requirements = oi._backend_runtime_requirements("yolo_world")
    assert requirements["missing_modules"] == ["ultralytics"]
    sam_requirements = oi._backend_runtime_requirements("sam3")
    assert sam_requirements["support_level"] == "optional"
    assert str(tmp_path / "sam.pt") in sam_requirements["missing_paths"]
    splat_requirements = oi._backend_runtime_requirements("splat_analyzer")
    assert splat_requirements["support_level"] == "optional"
    assert splat_requirements["required_modules"] == []
    assert oi._backend_runtime_requirements("other")["required_modules"] == []

    monkeypatch.setattr(
        oi,
        "_backend_runtime_requirements",
        lambda backend_name: {
            "support_level": "required",
            "required_modules": [],
            "missing_modules": [],
            "required_paths": [],
            "missing_paths": [],
        },
    )
    assert oi._backend_preflight_status(backend_name="yolo_world", command_template="")["reason"] == "command_not_configured"
    assert oi._backend_preflight_status(backend_name="yolo_world", command_template='"unterminated')["status"] == "invalid"
    assert oi._backend_preflight_status(backend_name="yolo_world", command_template="   ")["reason"] == "empty_command"
    assert oi._backend_preflight_status(backend_name="yolo_world", command_template="")["configured"] is False
    missing = oi._backend_preflight_status(backend_name="yolo_world", command_template="/not/a/real/exe {INPUT_JSON}")
    assert missing["status"] == "missing"
    monkeypatch.setattr(
        oi,
        "_backend_runtime_requirements",
        lambda backend_name: {
            "support_level": "optional" if backend_name == "sam3" else "required",
            "required_modules": ["x"],
            "missing_modules": ["x"],
            "required_paths": [],
            "missing_paths": [],
        },
    )
    assert oi._backend_preflight_status(backend_name="sam3", command_template="python -m x")["status"] == "optional_unavailable"
    assert oi._backend_preflight_status(backend_name="yolo_world", command_template="python -m x")["status"] == "runtime_missing"
    monkeypatch.setattr(
        oi,
        "_backend_runtime_requirements",
        lambda backend_name: {
            "support_level": "required",
            "required_modules": [],
            "missing_modules": [],
            "required_paths": ["/missing/weights.pt"],
            "missing_paths": ["/missing/weights.pt"],
        },
    )
    assert oi._backend_preflight_status(backend_name="yolo_world", command_template="python -m x")["reason"] == "missing_paths:/missing/weights.pt"
    monkeypatch.setattr(
        oi,
        "_backend_runtime_requirements",
        lambda backend_name: {
            "support_level": "optional",
            "required_modules": [],
            "missing_modules": [],
            "required_paths": ["/missing/optional.pt"],
            "missing_paths": ["/missing/optional.pt"],
        },
    )
    assert oi._backend_preflight_status(backend_name="sam3", command_template="python -m x")["status"] == "optional_unavailable"

    assert oi._payload_detection_count({"detections": [1, 2]}) == 2
    assert oi._payload_detection_count({"items": [1]}) == 1
    assert oi._payload_detection_count({"objects": [1, 2, 3]}) == 3
    assert oi._payload_detection_count([1]) == 1
    assert oi._payload_detection_count("bad") == 0

    assert oi._run_backend_command(backend_name="b", command_template="", input_payload={}, output_dir=tmp_path)["status"] == "skipped"
    assert oi._run_backend_command(backend_name="b", command_template='"bad', input_payload={}, output_dir=tmp_path)["status"] == "failed"
    assert oi._run_backend_command(backend_name="b", command_template="   ", input_payload={}, output_dir=tmp_path)["reason"] == "empty_command"
    monkeypatch.setattr(oi.subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("no launch")))
    assert oi._run_backend_command(backend_name="b", command_template="cmd", input_payload={}, output_dir=tmp_path)["reason"].startswith("failed_to_launch")

    def run_writes_mapping(command, **_kwargs):
        Path(command[-1]).write_text(
            json.dumps({"backend_status": "ok", "detections": [], "stderr_tail": "first\nlast"}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(oi.subprocess, "run", run_writes_mapping)
    empty = oi._run_backend_command(backend_name="mapped", command_template="cmd {INPUT_JSON} {OUTPUT_JSON}", input_payload={"x": 1}, output_dir=tmp_path)
    assert empty["status"] == "empty"
    assert empty["reason"] == "no_detections"

    def run_writes_reason(command, **_kwargs):
        Path(command[-1]).write_text(
            json.dumps({"backend_status": "failed", "reason": "model unavailable"}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(oi.subprocess, "run", run_writes_reason)
    assert oi._run_backend_command(
        backend_name="reason",
        command_template="cmd {INPUT_JSON} {OUTPUT_JSON}",
        input_payload={},
        output_dir=tmp_path / "reason",
    )["reason"] == "model unavailable"

    def run_writes_invalid(command, **_kwargs):
        Path(command[-1]).write_text("{bad", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(oi.subprocess, "run", run_writes_invalid)
    assert oi._run_backend_command(backend_name="bad-json", command_template="cmd {INPUT_JSON} {OUTPUT_JSON}", input_payload={}, output_dir=tmp_path)["reason"].startswith("invalid_output_json")

    monkeypatch.setattr(
        oi.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps([{"label": "door", "bbox": [0, 1, 2, 3]}]),
            stderr="",
        ),
    )
    stdout_payload = oi._run_backend_command(backend_name="stdout", command_template="cmd {INPUT_JSON} {OUTPUT_JSON}", input_payload={}, output_dir=tmp_path / "stdout")
    assert stdout_payload["payload"]["detections"][0]["label"] == "door"
    monkeypatch.setattr(
        oi.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, stdout="[]", stderr=""),
    )
    stdout_empty = oi._run_backend_command(backend_name="stdout-empty", command_template="cmd {INPUT_JSON} {OUTPUT_JSON}", input_payload={}, output_dir=tmp_path / "stdout-empty")
    assert stdout_empty["status"] == "empty"
    assert stdout_empty["reason"] == "no_detections"
    monkeypatch.setattr(
        oi.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 1, stdout="ignored", stderr="backend failed"),
    )
    failed = oi._run_backend_command(backend_name="failed", command_template="cmd {INPUT_JSON} {OUTPUT_JSON}", input_payload={}, output_dir=tmp_path / "failed")
    assert failed["status"] == "failed"
    assert oi._backend_reason_indicates_runtime_missing("torch_not_installed") is True
    assert oi._backend_reason_indicates_runtime_missing("") is False

    assert oi._bbox_xyxy({"bbox": [-1, -2, 120, 90]}, width=100, height=80) == [0.0, 0.0, 100.0, 80.0]
    assert oi._bbox_xyxy({"bbox": [1, 2, 3]}, width=100, height=80) is None
    keyframes = {0: _keyframe(tmp_path, 0)}
    detections, manip, artic, tasks = oi._normalize_detection_payload(
        backend_name="backend",
        payload={
            "items": [
                {"frameIndex": 0, "name": "Cabinet", "confidence": 0.7, "bbox": [1, 2, 20, 30]},
                {"frame_index": 99, "label": "skip", "bbox": [0, 0, 1, 1]},
                {"frame_index": 0, "label": "", "bbox": [0, 0, 1, 1]},
                {"frame_index": 0, "label": "NoBox"},
                "bad",
            ],
            "manipulation_candidates": [{"id": "m"}, "bad"],
            "articulation_hints": [{"id": "a"}],
            "tasks": [{"id": "t"}],
        },
        keyframes_by_index=keyframes,
    )
    assert detections[0]["label"] == "Cabinet"
    assert manip == [{"id": "m"}]
    assert artic == [{"id": "a"}]
    assert tasks == [{"id": "t"}]
    assert oi._normalize_detection_payload(backend_name="b", payload={"detections": "bad"}, keyframes_by_index={})[0] == []


def test_object_index_object_synthesis_llm_and_grounding_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _descriptor()
    existing = oi._normalize_existing_objects(
        {
            "objects": [
                {
                    "id": "known",
                    "label": "Door",
                    "boundingBox": {"center": [1, 2, 3], "extents": [0, 0.3], "axes": [[1]], "orientationQuaternion": [0, 1, 0, 0]},
                    "confidence": 0.6,
                    "all_crops": ["crop.png"],
                    "task_relevance": {"score": 0.5},
                    "articulation_hints": {"interactive": True},
                    "evidence_frames": [1],
                    "source_prompts": ["door"],
                    "provenance": {"source": "legacy"},
                    "mean_box_px": {"area": 12},
                },
                {"name": ""},
                "bad",
            ]
        }
    )
    assert existing[0]["boundingBox"]["extents"][0] == oi._MIN_BOX_EXTENT
    assert oi._normalize_existing_objects({"objects": "bad"}) == []

    assert oi._iou2d([0, 0, 10, 10], [5, 5, 15, 15]) > 0
    assert oi._iou2d([0, 0, 1, 1], [5, 5, 6, 6]) == 0.0
    detections = [
        {"frame_index": 0, "label": "Door", "score": 0.4, "bbox_xyxy": [0, 0, 10, 10]},
        {"frame_index": 0, "label": "Door", "score": 0.9, "bbox_xyxy": [0, 0, 10, 10]},
        {"frame_index": 1, "label": "Door", "score": 0.5, "bbox_xyxy": [0, 0, 10, 10]},
        {"frame_index": 0, "label": "Shelf", "score": 0.8, "bbox_xyxy": [50, 0, 90, 40]},
    ]
    deduped = oi._dedupe_same_frame(detections)
    assert [(item["frame_index"], item["label"]) for item in deduped] == [(0, "Door"), (0, "Shelf"), (1, "Door")]
    assert oi._center_from_bbox([0, 0, 10, 20], 20, 40) == [0.25, 0.25]
    assert oi._box_area([1, 1, 1, 1]) == 1.0
    assert oi._label_bucket("refrigerator door") == "door"
    assert oi._label_bucket("") == "object"

    keyframes = {
        0: _keyframe(tmp_path, 0, translation=[1.0, 2.0, 3.0]),
        1: _keyframe(tmp_path, 1, translation=[0.0, 0.0, 0.0]),
    }
    clustered = oi._cluster_detections(
        [
            {"frame_index": 0, "label": "Door", "score": 0.8, "bbox_xyxy": [0, 0, 20, 20], "world_center": [1, 1, 1]},
            {"frame_index": 1, "label": "door", "score": 0.7, "bbox_xyxy": [1, 1, 21, 21], "world_center": [1.2, 1, 1]},
            {"frame_index": 99, "label": "Door", "score": 0.6, "bbox_xyxy": [0, 0, 20, 20]},
            {"frame_index": 0, "label": "Shelf", "score": 0.5, "bbox_xyxy": [50, 10, 90, 40]},
        ],
        keyframes,
    )
    assert len(clustered) == 2
    center_clustered = oi._cluster_detections(
        [
            {"frame_index": 0, "label": "Box", "score": 0.8, "bbox_xyxy": [0, 0, 20, 20]},
            {"frame_index": 1, "label": "Box", "score": 0.7, "bbox_xyxy": [1, 1, 21, 21]},
        ],
        keyframes,
    )
    assert len(center_clustered) == 1

    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    crop = tmp_path / "crop.png"
    monkeypatch.setattr(oi.subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("ffmpeg missing")))
    oi._copy_crop(source, crop, [0, 0, 2, 2])
    assert crop.read_bytes() == b"source"
    crop.unlink()
    monkeypatch.setattr(oi.subprocess, "run", lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, stdout="", stderr="bad"))
    oi._copy_crop(source, crop, [0, 0, 2, 2])
    assert crop.read_bytes() == b"source"
    crop.unlink()

    def run_crop(command, **_kwargs):
        Path(command[-1]).write_bytes(b"cropped")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(oi.subprocess, "run", run_crop)
    oi._copy_crop(source, crop, [0, 0, 2, 2])
    assert crop.read_bytes() == b"cropped"

    assert oi._task_relevance("cabinet", ["cabinet"], descriptor)["score"] > 0.6
    assert oi._task_relevance("box", [], descriptor)["score"] > 0.2
    assert oi._articulation_hints("door")["kind"] == "door"
    assert oi._articulation_hints("drawer")["interactive"] is True
    assert oi._articulation_hints("cabinet")["kind"] == "cabinet"
    assert oi._articulation_hints("fridge")["kind"] == "refrigerator_door"
    assert oi._articulation_hints("shelf")["interactive"] is False

    provided_bbox = oi._synthesized_bbox(
        [{"world_center": [1, 2, 3], "world_extents": [0.1, 0.2, 0.3], "bbox_xyxy": [0, 0, 10, 10]}],
        keyframes_by_index=keyframes,
        cluster_index=1,
        label="cabinet",
    )
    assert provided_bbox["center"] == [1.0, 2.0, 3.0]
    translated_bbox = oi._synthesized_bbox(
        [{"frame_index": 0, "bbox_xyxy": [0, 0, 20, 20]}],
        keyframes_by_index=keyframes,
        cluster_index=1,
        label="cabinet",
    )
    assert translated_bbox["center"][2] == 4.5
    fallback_bbox = oi._synthesized_bbox([], keyframes_by_index=keyframes, cluster_index=3, label="unknown")
    assert fallback_bbox["center"] == [1.05, 0.0, 0.0]
    missing_keyframe_bbox = oi._synthesized_bbox(
        [{"frame_index": 99, "bbox_xyxy": [0, 0, 10, 10]}],
        keyframes_by_index=keyframes,
        cluster_index=4,
        label="unknown",
    )
    assert missing_keyframe_bbox["center"] == [1.4, 0.0, 0.0]

    raw_root = tmp_path / "raw"
    crops_dir = raw_root / "object_index_artifacts" / "crops"
    crops_dir.mkdir(parents=True)
    existing_crop = raw_root / "existing.png"
    existing_crop.write_bytes(b"crop")
    monkeypatch.setattr(oi, "_copy_crop", lambda _frame, path, _bbox: path.write_bytes(b"generated"))
    objects = oi._build_objects(
        clusters=[
            [
                {
                    "frame_index": 0,
                    "label": "Cabinet",
                    "score": 0.9,
                    "bbox_xyxy": [0, 0, 20, 20],
                    "source_prompt": "cabinet",
                    "source": "unit",
                    "crop_path": str(existing_crop),
                },
                {
                    "frame_index": 1,
                    "label": "Cabinet",
                    "score": 0.7,
                    "bbox_xyxy": [2, 2, 22, 22],
                    "source_prompt": "drawer",
                    "source": "unit",
                },
            ],
            [
                {
                    "frame_index": 99,
                    "label": "Shelf",
                    "score": 0.6,
                    "bbox_xyxy": [0, 0, 20, 20],
                    "source_prompt": "shelf",
                    "source": "unit",
                }
            ],
            [],
        ],
        keyframes_by_index=keyframes,
        descriptor=descriptor,
        raw_root=raw_root,
        crops_dir=crops_dir,
    )
    assert objects[0]["reference_crop"] == "existing.png"
    assert objects[0]["mean_confidence"] == 0.7
    assert objects[0]["provenance"]["privacy_penalty_applied"] is True

    assert oi._apply_llm_task_relevance(runner=None, descriptor=descriptor, objects=objects) is None
    assert oi._apply_llm_task_relevance(runner=lambda *_args: ["bad"], descriptor=descriptor, objects=objects) is None
    response = oi._apply_llm_task_relevance(
        runner=lambda name, payload: {
            "scores": [
                {"object_id": objects[0]["id"], "score": 0.95, "matched_terms": ["cabinet", "open"], "reason": "task"}
            ]
        },
        descriptor=descriptor,
        objects=objects,
    )
    assert response is not None
    assert objects[0]["task_relevance"]["score"] == 0.95
    oi._apply_llm_task_relevance(
        runner=lambda name, payload: {"scores": [{"object_id": "missing", "score": 1.0}]},
        descriptor=descriptor,
        objects=objects,
    )

    assert oi._apply_llm_articulation_priors(runner=None, descriptor=descriptor, objects=objects) is None
    assert oi._apply_llm_articulation_priors(runner=lambda *_args: ["bad"], descriptor=descriptor, objects=objects) is None
    oi._apply_llm_articulation_priors(
        runner=lambda name, payload: {
            "articulation_priors": [
                {"instance_id": objects[0]["id"], "interactive": True, "kind": "cabinet_door", "confidence": 0.98, "reason": "llm"}
            ]
        },
        descriptor=descriptor,
        objects=objects,
    )
    assert objects[0]["articulation_hints"]["kind"] == "cabinet_door"
    oi._apply_llm_articulation_priors(
        runner=lambda name, payload: {"articulation_priors": [{"object_id": "missing", "interactive": True}]},
        descriptor=descriptor,
        objects=objects,
    )
    assert oi._llm_target_resolution(runner=None, descriptor=descriptor, objects=objects) is None
    assert oi._llm_target_resolution(runner=lambda *_args: {"tasks": [{"task_id": "t"}]}, descriptor=descriptor, objects=objects) == {"tasks": [{"task_id": "t"}]}
    assert oi._llm_target_resolution(runner=lambda *_args: "bad", descriptor=descriptor, objects=objects) is None

    grounding = oi._grounding_payload_from_objects(
        objects,
        descriptor,
        {"status": "built"},
    )
    assert grounding["backend_status"] == "ok"
    assert grounding["manipulation_candidates"][0]["instance_id"] == objects[0]["id"]
    assert grounding["articulation_hints"][0]["reason"] == "cabinet_door"
    assert grounding["tasks"][0]["task_id"] == "open_close_primary"
    assert oi._grounding_payload_from_objects(["bad", *objects], descriptor, {})["grounded_objects"][0]["label"] == objects[0]["label"]
    assert oi._grounding_payload_from_objects([], descriptor, {})["backend_status"] == "empty"


def test_object_index_legacy_reuse_writers_stage_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    descriptor = _descriptor("gs://bucket/scenes/scene/captures/capture/raw")
    descriptor_path = tmp_path / "capture_descriptor.json"
    oi._write_descriptor_updates(descriptor_path, descriptor, "gs://bucket/raw/object_index.json")
    assert json.loads(descriptor_path.read_text(encoding="utf-8"))["object_index_uri"].endswith("object_index.json")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    oi._write_manifest_updates(manifest_path)
    assert "object_index_uri" not in json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_text('{"scene_id": "scene"}', encoding="utf-8")
    oi._write_manifest_updates(manifest_path)
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["object_index_uri"] == "object_index.json"
    missing_manifest = tmp_path / "missing_manifest.json"
    oi._write_manifest_updates(missing_manifest)
    assert not missing_manifest.exists()

    _, context = _capture_tree(tmp_path / "legacy")
    monkeypatch.setattr(
        oi,
        "load_object_index",
        lambda *_args, **_kwargs: [
            {
                "id": "legacy_obj",
                "label": "Cabinet",
                "boundingBox": {"center": [0, 0, 0], "extents": [1, 1, 1]},
                "task_relevance": {"score": 0.9},
                "articulation_hints": {"interactive": True, "kind": "cabinet", "confidence": 0.7},
            }
        ],
    )
    assert oi._canonicalize_legacy_index(context=context, descriptor=descriptor) is None
    legacy_index = context.raw_root / "arkit" / "objects" / "index.json"
    legacy_index.parent.mkdir(parents=True)
    legacy_index.write_text('{"objects": []}', encoding="utf-8")
    canonicalized = oi._canonicalize_legacy_index(context=context, descriptor=descriptor)
    assert canonicalized is not None
    assert canonicalized["status"] == "canonicalized_legacy"

    reused = oi._canonicalize_legacy_index(context=context, descriptor=descriptor)
    assert reused is not None
    assert reused["status"] == "reused"
    (context.raw_root / "object_index_build_report.json").write_text(
        json.dumps({"object_count": 0, "empty_index_cause": "runtime_missing"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(oi, "load_object_index", lambda *_args, **_kwargs: [])
    assert oi._canonicalize_legacy_index(context=context, descriptor=descriptor) is None

    assert _existing_index_is_reusable(loaded=[{"id": "x"}], report=None) is True
    assert _existing_index_is_reusable(loaded=[], report=None) is False
    assert _existing_index_is_reusable(loaded=[], report={"object_count": "bad"}) is False
    assert _existing_index_is_reusable(loaded=[], report={"object_count": 1}) is False
    assert _existing_index_is_reusable(
        loaded=[],
        report={"object_count": 0, "runtime_preflight": {"backends": {"b": {"support_level": "required", "status": "optional_unavailable"}}}},
    ) is False
    assert _existing_index_is_reusable(
        loaded=[],
        report={"object_count": 0, "runtime_preflight": {"backends": {"b": "bad"}}},
    ) is True

    monkeypatch.setattr(oi, "_canonicalize_legacy_index", lambda **_kwargs: {"status": "reused", "object_count": 1})
    assert oi.run_object_index_stage(capture_root=context.capture_root, force_rebuild=False)["status"] == "reused"

    capture_root, stage_context = _capture_tree(tmp_path / "stage")
    monkeypatch.setattr(oi, "resolve_local_capture_context", lambda _capture_root: stage_context)
    monkeypatch.setattr(oi, "load_raw_manifest", lambda *_args, **_kwargs: _manifest("walkthrough.mp4"))
    monkeypatch.setattr(oi, "build_capture_enrichment_runner", lambda **_kwargs: enrichment_runner)
    monkeypatch.setattr(oi, "_sample_keyframes", lambda **_kwargs: [_keyframe(tmp_path, 0)])
    monkeypatch.setattr(oi, "_extract_keyframe_images", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(oi, "_resolve_video_path", lambda *_args, **_kwargs: stage_context.raw_root / "walkthrough.mp4")
    monkeypatch.setattr(oi, "_command_from_env", lambda name: f"{name.lower()} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setattr(
        oi,
        "_backend_preflight_status",
        lambda backend_name, command_template: {
            "status": "ready",
            "support_level": "optional" if backend_name in {"sam3", "splat_analyzer"} else "required",
        },
    )

    def fake_backend(backend_name: str, **_kwargs):
        if backend_name == "yolo_world":
            return {
                "status": "ok",
                "backend": backend_name,
                "payload": {
                    "detections": [
                        {
                            "frame_index": 0,
                            "label": "Cabinet",
                            "score": 0.8,
                            "bbox": [0, 0, 20, 20],
                            "source_prompt": "cabinet",
                        }
                    ],
                    "manipulation_candidates": [{"instance_id": "backend-candidate"}],
                    "articulation_hints": [{"instance_id": "backend-hint"}],
                    "tasks": [{"task_id": "backend-task"}],
                },
            }
        return {"status": "skipped", "backend": backend_name, "reason": "no-op"}

    def enrichment_runner(name: str, payload: dict[str, object]) -> dict[str, object]:
        if name == "prompt_bank_expander":
            return {"additional_prompts": ["hinge"]}
        if name == "task_relevance_ranker":
            object_id = payload["objects"][0]["object_id"]  # type: ignore[index]
            return {"scores": [{"object_id": object_id, "score": 0.95, "matched_terms": ["cabinet"], "reason": "match"}]}
        if name == "articulation_prior_writer":
            object_id = payload["objects"][0]["object_id"]  # type: ignore[index]
            return {"articulation_priors": [{"object_id": object_id, "interactive": True, "kind": "cabinet", "confidence": 0.91}]}
        if name == "workflow_target_resolver":
            return {
                "manipulation_candidates": [{"instance_id": "llm-candidate"}],
                "articulation_hints": [{"instance_id": "llm-hint"}],
                "tasks": [{"task_id": "llm-task"}],
            }
        return {}

    monkeypatch.setattr(oi, "_run_backend_command", fake_backend)
    monkeypatch.setattr(oi, "_copy_crop", lambda _frame, path, _bbox: path.write_bytes(b"crop"))
    result = oi.run_object_index_stage(capture_root=capture_root, force_rebuild=True)
    assert result["status"] == "built"
    assert result["object_count"] == 1
    report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
    assert report["prompt_bank"]["all"][-1] == "hinge"
    assert report["llm_enrichment"]["workflow_target_resolver"]["tasks"][0]["task_id"] == "llm-task"
    grounding = json.loads((stage_context.raw_root / "object_grounding_hints.json").read_text(encoding="utf-8"))
    assert grounding["tasks"][0]["task_id"] == "llm-task"

    def existing_object_backend(backend_name: str, **_kwargs):
        if backend_name == "splat_analyzer":
            return {
                "status": "ok",
                "backend": backend_name,
                "payload": {
                    "objects": [
                        {
                            "id": "existing",
                            "label": "Door",
                            "boundingBox": {"center": [0, 0, 0], "extents": [1, 1, 1]},
                            "task_relevance": {"score": 0.9},
                            "articulation_hints": {"interactive": True, "kind": "door", "confidence": 0.8},
                        }
                    ],
                    "scene_relationship_candidates": [
                        {
                            "subject_id": "existing",
                            "object_id": "target",
                            "relationship": "near",
                            "confidence": 0.6,
                        }
                    ],
                },
            }
        return {"status": "ok", "backend": backend_name, "payload": {"detections": []}}

    monkeypatch.setattr(oi, "_run_backend_command", existing_object_backend)
    existing_result = oi.run_object_index_stage(capture_root=capture_root, force_rebuild=True)
    assert existing_result["object_count"] == 1
    existing_grounding = json.loads((stage_context.raw_root / "object_grounding_hints.json").read_text(encoding="utf-8"))
    assert existing_grounding["scene_relationship_candidates"][0]["relationship"] == "near"

    empty_cases = [
        (
            "runtime-missing",
            [{"status": "failed", "backend": "yolo_world", "reason": "torch_not_installed"}],
            None,
            "runtime_missing",
        ),
        (
            "backend-skipped",
            [{"status": "skipped", "backend": "yolo_world", "reason": "command_not_configured"}],
            None,
            "backend_skipped",
        ),
        (
            "zero-detections",
            [{"status": "ok", "backend": "yolo_world", "payload": {"detections": []}}],
            None,
            "zero_detections",
        ),
        (
            "all-filtered",
            [{"status": "ok", "backend": "yolo_world", "payload": {"detections": [{"frame_index": 0, "label": "Box", "bbox": [0, 0, 10, 10]}]}}],
            lambda **_kwargs: [],
            "all_filtered",
        ),
    ]
    original_build_objects = oi._build_objects
    for suffix, reports, build_objects, expected in empty_cases:
        report_iter = iter(
            [
                *reports,
                {"status": "ok", "backend": "grounding_dino", "payload": {"detections": []}},
                {"status": "skipped", "backend": "sam3", "reason": "sam3_not_installed"},
                {"status": "skipped", "backend": "splat_analyzer", "reason": "missing_local_splat_asset"},
            ]
        )
        monkeypatch.setattr(oi, "_run_backend_command", lambda **_kwargs: next(report_iter))
        monkeypatch.setattr(oi, "_build_objects", build_objects or original_build_objects)
        empty_result = oi.run_object_index_stage(capture_root=capture_root, force_rebuild=True)
        empty_report = json.loads(Path(empty_result["report_path"]).read_text(encoding="utf-8"))
        assert empty_report["empty_index_cause"] == expected
    monkeypatch.setattr(oi, "_build_objects", original_build_objects)

    monkeypatch.setattr(oi, "run_object_index_stage", lambda **_kwargs: {"manifest_path": "m", "report_path": "r", "object_count": 2})
    assert oi.ensure_object_index_stage(capture_root=capture_root)["object_count"] == 2
    assert oi.main(["--capture-root", str(capture_root), "--force-rebuild"]) == 0
    assert "object_count=2" in capsys.readouterr().out
    monkeypatch.setattr(oi, "run_object_index_stage", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    assert oi.main(["--capture-root", str(capture_root)]) == 1
    assert "FAILED: boom" in capsys.readouterr().out


def test_normalize_detection_payload_tolerates_malformed_backend_reports(tmp_path: Path) -> None:
    """Malformed/partial backend payloads normalize to empty-but-valid output, no raise."""
    keyframes = {0: _keyframe(tmp_path, 0)}

    # Missing detections/objects keys entirely -> empty tuple, no raise.
    detections, manip, artic, tasks = oi._normalize_detection_payload(
        backend_name="b", payload={}, keyframes_by_index=keyframes
    )
    assert detections == []
    assert (manip, artic, tasks) == ([], [], [])

    # detections / items not a list -> ignored, empty detections.
    assert oi._normalize_detection_payload(
        backend_name="b", payload={"detections": "bad"}, keyframes_by_index=keyframes
    )[0] == []
    assert oi._normalize_detection_payload(
        backend_name="b", payload={"items": {"x": 1}}, keyframes_by_index=keyframes
    )[0] == []

    # Detections missing bbox / label / score, plus non-mapping rows, all dropped.
    detections, manip, artic, tasks = oi._normalize_detection_payload(
        backend_name="b",
        payload={
            "detections": [
                "string-row",
                42,
                None,
                {"frame_index": 0, "label": "NoBox"},  # missing bbox -> dropped
                {"frame_index": 0, "bbox": [0, 0, 10, 10]},  # missing label -> dropped
                {"frame_index": 0, "label": "", "bbox": [0, 0, 10, 10]},  # blank label -> dropped
                {"frame_index": 7, "label": "OffFrame", "bbox": [0, 0, 10, 10]},  # unknown keyframe -> dropped
                {"frame_index": 0, "label": "GoodNoScore", "bbox": [1, 2, 9, 9]},  # kept, score defaults to 0.0
            ],
            # Non-list grounding collections must normalize to empty lists.
            "manipulation_candidates": "bad",
            "articulation_hints": 7,
            "tasks": {"not": "a list"},
        },
        keyframes_by_index=keyframes,
    )
    assert [item["label"] for item in detections] == ["GoodNoScore"]
    assert detections[0]["score"] == 0.0
    assert (manip, artic, tasks) == ([], [], [])

    # Mixed grounding collections: only mapping entries survive.
    _, manip, artic, tasks = oi._normalize_detection_payload(
        backend_name="b",
        payload={
            "manipulation_candidates": [{"id": "m"}, "bad", 3],
            "articulation_hints": [{"id": "a"}, None],
            "tasks": [{"id": "t"}, ["nested"]],
        },
        keyframes_by_index=keyframes,
    )
    assert manip == [{"id": "m"}]
    assert artic == [{"id": "a"}]
    assert tasks == [{"id": "t"}]


def test_run_object_index_stage_normalizes_mixed_malformed_backend_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mix of ok-but-malformed / failed / skipped backends yields empty-but-valid artifacts.

    Reuses the stage stubbing pattern (resolve context, keyframes, _run_backend_command)
    and asserts the stage does not raise and records empty_index_cause.
    """
    capture_root, stage_context = _capture_tree(tmp_path / "stage")
    monkeypatch.setattr(oi, "resolve_local_capture_context", lambda _capture_root: stage_context)
    monkeypatch.setattr(oi, "load_raw_manifest", lambda *_args, **_kwargs: _manifest("walkthrough.mp4"))
    monkeypatch.setattr(oi, "build_capture_enrichment_runner", lambda **_kwargs: None)
    monkeypatch.setattr(oi, "_sample_keyframes", lambda **_kwargs: [_keyframe(tmp_path, 0)])
    monkeypatch.setattr(oi, "_extract_keyframe_images", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(oi, "_resolve_video_path", lambda *_args, **_kwargs: stage_context.raw_root / "walkthrough.mp4")
    monkeypatch.setattr(oi, "_command_from_env", lambda name: f"{name.lower()} {{INPUT_JSON}} {{OUTPUT_JSON}}")
    monkeypatch.setattr(
        oi,
        "_backend_preflight_status",
        lambda backend_name, command_template: {
            "status": "ready",
            "support_level": "optional" if backend_name in {"sam3", "splat_analyzer"} else "required",
        },
    )

    def mixed_backend(backend_name: str, **_kwargs):
        if backend_name == "yolo_world":
            # ok status but the payload is structurally malformed in every field.
            return {
                "status": "ok",
                "backend": backend_name,
                "payload": {
                    "detections": "not-a-list",
                    "manipulation_candidates": {"bad": True},
                    "articulation_hints": 5,
                    "tasks": "nope",
                },
            }
        if backend_name == "grounding_dino":
            # ok status, detections present but each row is missing bbox/label/score.
            return {
                "status": "ok",
                "backend": backend_name,
                "payload": {
                    "detections": [
                        {"frame_index": 0, "label": "NoBox"},
                        {"frame_index": 0, "bbox": [0, 0, 5, 5]},
                        "garbage-row",
                        99,
                    ]
                },
            }
        if backend_name == "sam3":
            return {"status": "failed", "backend": backend_name, "reason": "model unavailable"}
        return {"status": "skipped", "backend": backend_name, "reason": "missing_local_splat_asset"}

    monkeypatch.setattr(oi, "_run_backend_command", mixed_backend)
    monkeypatch.setattr(oi, "_copy_crop", lambda _frame, path, _bbox: path.write_bytes(b"crop"))

    result = oi.run_object_index_stage(capture_root=capture_root, force_rebuild=True)

    assert result["status"] == "built"
    assert result["object_count"] == 0
    report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
    # No detections survived normalization -> empty index, valid cause recorded.
    assert report["object_count"] == 0
    assert report["empty_index_cause"] in {"zero_detections", "all_filtered", "backend_skipped"}
    # Manifest + grounding artifacts are still written and structurally valid.
    assert Path(result["manifest_path"]).is_file()
    manifest_payload = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest_payload["objects"] == []
    grounding = json.loads((stage_context.raw_root / "object_grounding_hints.json").read_text(encoding="utf-8"))
    assert grounding["grounded_objects"] == []
    assert grounding["manipulation_candidates"] == []
    assert grounding["articulation_hints"] == []
    assert grounding["tasks"] == []
