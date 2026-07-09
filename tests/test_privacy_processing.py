from __future__ import annotations

import json
import shutil
import subprocess
import urllib.error
from io import BytesIO
from pathlib import Path

import pytest

import blueprint_pipeline.privacy_processing as pp
from blueprint_pipeline.privacy_processing import (
    _deepprivacy_command_template,
    _vip_command_template,
    run_privacy_postprocess,
)


def _write_video(path: Path, payload: bytes = b"fake-video") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_real_test_video(path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg not installed")
    path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=96x64:rate=5",
            "-t",
            "1",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.skip(f"ffmpeg test source failed: {proc.stderr[-200:]}")


def _depth_anything_result() -> dict[str, object]:
    return {
        "status": "succeeded",
        "source": "depth_anything",
        "provider": "depth_anything_3",
        "model_name": "da3metric-large",
        "depth_prefix_uri": "gs://bucket/scenes/scene-1/captures/cap-1/pipeline/privacy_depth/depth",
        "confidence_prefix_uri": "gs://bucket/scenes/scene-1/captures/cap-1/pipeline/privacy_depth/confidence",
        "depth_manifest_uri": "gs://bucket/scenes/scene-1/captures/cap-1/pipeline/privacy_depth/depth_manifest.json",
        "confidence_manifest_uri": "gs://bucket/scenes/scene-1/captures/cap-1/pipeline/privacy_depth/confidence_manifest.json",
        "depth_manifest_path": "/tmp/depth_manifest.json",
        "confidence_manifest_path": "/tmp/confidence_manifest.json",
        "frame_count": 12,
    }


def _capture_paths(tmp_path: Path, name: str = "cap-1") -> tuple[Path, Path, Path]:
    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / name
    pipeline_dir = capture_root / "pipeline"
    raw_video = capture_root / "raw" / "walkthrough.mov"
    _write_video(raw_video)
    return capture_root, pipeline_dir, raw_video


def _run_postprocess_for(
    capture_root: Path,
    pipeline_dir: Path,
    raw_video: Path | None,
) -> dict[str, object]:
    return run_privacy_postprocess(
        bucket="bucket",
        scene_id="scene-1",
        capture_id=capture_root.name,
        capture_root=capture_root,
        pipeline_dir=pipeline_dir,
        raw_video_path=raw_video,
    )


def test_privacy_command_templates_accept_privacy_prefixed_env(monkeypatch) -> None:
    monkeypatch.delenv("VIP_COMMAND", raising=False)
    monkeypatch.delenv("DEEPPRIVACY2_COMMAND", raising=False)
    monkeypatch.setenv("PRIVACY_VIP_COMMAND", "vip-runner --input {INPUT_VIDEO}")
    monkeypatch.setenv(
        "PRIVACY_DEEPPRIVACY2_COMMAND",
        "deepprivacy2-runner --input {INPUT_VIDEO}",
    )

    assert _vip_command_template() == "vip-runner --input {INPUT_VIDEO}"
    assert _deepprivacy_command_template() == "deepprivacy2-runner --input {INPUT_VIDEO}"


def test_default_privacy_redaction_prompt_covers_industrial_pii_classes(monkeypatch) -> None:
    monkeypatch.delenv("PRIVACY_REDACTION_TARGET_CLASSES", raising=False)

    classes = pp._privacy_redaction_target_classes()

    assert "person" in classes
    assert "badge_id" in classes
    assert "screen" in classes
    assert "whiteboard" in classes
    assert "license_plate" in classes
    assert "shipping_label" in classes
    assert "badge_id" in pp._privacy_redaction_prompt()


def test_privacy_postprocess_non_arkit_passthrough_still_generates_depth(monkeypatch, tmp_path: Path) -> None:
    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "cap-1"
    pipeline_dir = capture_root / "pipeline"
    raw_video = capture_root / "raw" / "walkthrough.mov"
    _write_video(raw_video)

    monkeypatch.setenv("PRIVACY_PIPELINE_ENABLED", "true")
    monkeypatch.setattr(
        "blueprint_pipeline.privacy_processing._run_sam3",
        lambda **_kwargs: {
            "status": "succeeded",
            "people_detected": False,
            "people_count": 0,
            "mask_paths": [],
        },
    )
    depth_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "blueprint_pipeline.privacy_processing._run_depth_anything",
        lambda **kwargs: depth_calls.append(kwargs) or _depth_anything_result(),
    )

    result = run_privacy_postprocess(
        bucket="bucket",
        scene_id="scene-1",
        capture_id="cap-1",
        capture_root=capture_root,
        pipeline_dir=pipeline_dir,
        raw_video_path=raw_video,
    )

    assert result["status"] == "no_people_detected"
    assert result["depth_source"] == "depth_anything"
    assert result["depth_conditioning"]["depth_manifest_uri"].endswith("/pipeline/privacy_depth/depth_manifest.json")
    assert result["world_model_video_uri"] == result["privacy_processed_video_uri"]
    assert depth_calls
    assert (capture_root / "privacy" / "final_walkthrough.mov").is_file()
    manifest = json.loads((pipeline_dir / "privacy_processing_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "no_people_detected"
    assert manifest["depth_source"] == "depth_anything"
    assert "badge_id" in manifest["redaction_target_classes"]
    assert "screen" in manifest["redaction_target_classes"]


def test_privacy_postprocess_placeholder_runner_url_fails_closed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "cap-1"
    pipeline_dir = capture_root / "pipeline"
    raw_video = capture_root / "raw" / "walkthrough.mov"
    _write_video(raw_video)

    monkeypatch.setenv("PRIVACY_PIPELINE_ENABLED", "true")
    monkeypatch.setenv("PRIVACY_SAM3_URL", "REPLACE_ME_PRIVACY_SAM3_URL")

    result = run_privacy_postprocess(
        bucket="bucket",
        scene_id="scene-1",
        capture_id="cap-1",
        capture_root=capture_root,
        pipeline_dir=pipeline_dir,
        raw_video_path=raw_video,
    )
    manifest = json.loads((pipeline_dir / "privacy_processing_manifest.json").read_text(encoding="utf-8"))
    verification = json.loads((pipeline_dir / "privacy_verification_report.json").read_text(encoding="utf-8"))

    assert result["status"] == "failed_closed"
    assert result["reason"] == "runner_url_invalid_or_placeholder"
    assert manifest["status"] == "failed_closed"
    assert verification["initial_detection"]["reason"] == "runner_url_invalid_or_placeholder"


def test_privacy_postprocess_local_full_frame_redaction_writes_final_walkthrough(
    monkeypatch,
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "cap-1"
    pipeline_dir = capture_root / "pipeline"
    raw_video = capture_root / "raw" / "walkthrough.mov"
    _write_real_test_video(raw_video)

    monkeypatch.setenv("PRIVACY_PIPELINE_ENABLED", "true")
    monkeypatch.setenv("PRIVACY_LOCAL_FULL_FRAME_REDACTION_ENABLED", "true")
    monkeypatch.setattr(
        "blueprint_pipeline.privacy_processing._run_sam3",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("SAM3 should not run")),
    )

    result = run_privacy_postprocess(
        bucket="bucket",
        scene_id="scene-1",
        capture_id="cap-1",
        capture_root=capture_root,
        pipeline_dir=pipeline_dir,
        raw_video_path=raw_video,
    )

    assert result["status"] == "full_frame_redacted_local_proof"
    assert result["mode"] == "full_frame_redaction"
    assert result["local_repo_proof_only"] is True
    assert result["production_review_required"] is True
    assert result["proof_boundary"]["local_full_frame_redaction_executed"] is True
    assert result["proof_boundary"]["live_privacy_service_proven"] is False
    assert (capture_root / "privacy" / "final_walkthrough.mov").is_file()
    manifest = json.loads((pipeline_dir / "privacy_processing_manifest.json").read_text(encoding="utf-8"))
    verification = json.loads((pipeline_dir / "privacy_verification_report.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "full_frame_redacted_local_proof"
    assert verification["status"] == "full_frame_redacted_local_proof"


def test_privacy_postprocess_uses_anonymized_fallback(monkeypatch, tmp_path: Path) -> None:
    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "cap-1"
    pipeline_dir = capture_root / "pipeline"
    raw_video = capture_root / "raw" / "walkthrough.mov"
    _write_video(raw_video)

    monkeypatch.setenv("PRIVACY_PIPELINE_ENABLED", "true")
    sam3_results = iter(
        [
            {
                "status": "succeeded",
                "people_detected": True,
                "people_count": 2,
                "mask_paths": ["mask-1.png"],
            },
            {
                "status": "succeeded",
                "people_detected": True,
                "people_count": 1,
                "mask_paths": ["mask-2.png"],
            },
            {
                "status": "succeeded",
                "people_detected": True,
                "people_count": 1,
                "mask_paths": [],
            },
        ]
    )
    monkeypatch.setattr(
        "blueprint_pipeline.privacy_processing._run_sam3",
        lambda **_kwargs: next(sam3_results),
    )
    monkeypatch.setattr(
        "blueprint_pipeline.privacy_processing._run_depth_anything",
        lambda **_kwargs: _depth_anything_result(),
    )

    def _vip(**kwargs):
        output = kwargs["output_video"]
        _write_video(output, b"vip-video")
        return {
            "status": "succeeded",
            "output_video": str(output),
            "depth_source": "depth_anything",
        }

    def _deepprivacy(**kwargs):
        output = kwargs["output_video"]
        _write_video(output, b"deepprivacy-video")
        return {
            "status": "succeeded",
            "output_video": str(output),
            "face_anonymized_segments": ["segment-1"],
        }

    monkeypatch.setattr("blueprint_pipeline.privacy_processing._run_vip", _vip)
    monkeypatch.setattr("blueprint_pipeline.privacy_processing._run_deepprivacy2", _deepprivacy)

    result = run_privacy_postprocess(
        bucket="bucket",
        scene_id="scene-1",
        capture_id="cap-1",
        capture_root=capture_root,
        pipeline_dir=pipeline_dir,
        raw_video_path=raw_video,
    )

    assert result["status"] == "face_anonymized_fallback"
    assert result["fallback_used"] is True
    assert result["face_anonymized_segments"] == ["segment-1"]
    assert result["depth_source"] == "depth_anything"
    assert (capture_root / "privacy" / "final_walkthrough.mov").is_file()


def test_privacy_postprocess_prefers_arkit_depth_for_vip(monkeypatch, tmp_path: Path) -> None:
    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "cap-1"
    pipeline_dir = capture_root / "pipeline"
    raw_video = capture_root / "raw" / "walkthrough.mov"
    depth_dir = capture_root / "raw" / "arkit" / "depth"
    confidence_dir = capture_root / "raw" / "arkit" / "confidence"
    _write_video(raw_video)
    depth_dir.mkdir(parents=True, exist_ok=True)
    confidence_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("PRIVACY_PIPELINE_ENABLED", "true")
    sam3_results = iter(
        [
            {"status": "succeeded", "people_detected": True, "people_count": 1, "mask_paths": ["mask-1.png"]},
            {"status": "succeeded", "people_detected": False, "people_count": 0, "mask_paths": []},
        ]
    )
    monkeypatch.setattr(
        "blueprint_pipeline.privacy_processing._run_sam3",
        lambda **_kwargs: next(sam3_results),
    )
    monkeypatch.setattr(
        "blueprint_pipeline.privacy_processing._run_depth_anything",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("Depth Anything should not run for ARKit captures")),
    )

    def _vip(**kwargs):
        assert kwargs["arkit_depth_prefix_uri"] == "gs://bucket/scenes/scene-1/captures/cap-1/raw/arkit/depth"
        assert kwargs["arkit_confidence_prefix_uri"] == "gs://bucket/scenes/scene-1/captures/cap-1/raw/arkit/confidence"
        assert kwargs["depth_manifest_uri"] is None
        assert kwargs["confidence_manifest_uri"] is None
        output = kwargs["output_video"]
        _write_video(output, b"vip-video")
        return {"status": "succeeded", "output_video": str(output), "depth_source": "arkit"}

    monkeypatch.setattr("blueprint_pipeline.privacy_processing._run_vip", _vip)

    result = run_privacy_postprocess(
        bucket="bucket",
        scene_id="scene-1",
        capture_id="cap-1",
        capture_root=capture_root,
        pipeline_dir=pipeline_dir,
        raw_video_path=raw_video,
    )

    assert result["status"] == "person_removed"
    assert result["depth_source"] == "arkit"


def test_privacy_postprocess_uses_depth_anything_for_non_arkit_capture(monkeypatch, tmp_path: Path) -> None:
    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "cap-1"
    pipeline_dir = capture_root / "pipeline"
    raw_video = capture_root / "raw" / "walkthrough.mov"
    _write_video(raw_video)

    monkeypatch.setenv("PRIVACY_PIPELINE_ENABLED", "true")
    sam3_results = iter(
        [
            {"status": "succeeded", "people_detected": True, "people_count": 1, "mask_paths": ["mask-1.png"]},
            {"status": "succeeded", "people_detected": False, "people_count": 0, "mask_paths": []},
        ]
    )
    monkeypatch.setattr(
        "blueprint_pipeline.privacy_processing._run_sam3",
        lambda **_kwargs: next(sam3_results),
    )
    monkeypatch.setattr(
        "blueprint_pipeline.privacy_processing._run_depth_anything",
        lambda **_kwargs: _depth_anything_result(),
    )

    def _vip(**kwargs):
        assert kwargs["arkit_depth_prefix_uri"] is None
        assert kwargs["arkit_confidence_prefix_uri"] is None
        assert kwargs["depth_manifest_uri"] == "gs://bucket/scenes/scene-1/captures/cap-1/pipeline/privacy_depth/depth_manifest.json"
        assert kwargs["confidence_manifest_uri"] == "gs://bucket/scenes/scene-1/captures/cap-1/pipeline/privacy_depth/confidence_manifest.json"
        output = kwargs["output_video"]
        _write_video(output, b"vip-video")
        return {
            "status": "succeeded",
            "output_video": str(output),
            "depth_source": "depth_anything",
        }

    monkeypatch.setattr("blueprint_pipeline.privacy_processing._run_vip", _vip)

    result = run_privacy_postprocess(
        bucket="bucket",
        scene_id="scene-1",
        capture_id="cap-1",
        capture_root=capture_root,
        pipeline_dir=pipeline_dir,
        raw_video_path=raw_video,
    )

    assert result["status"] == "person_removed"
    assert result["depth_source"] == "depth_anything"


def test_privacy_processing_private_helper_edges(monkeypatch, tmp_path: Path) -> None:
    assert pp._string_list("mask.png") == ["mask.png"]
    assert pp._string_list(7) == ["7"]

    monkeypatch.setenv("PRIVACY_TIMEOUT", "bad")
    assert pp._timeout_env("PRIVACY_TIMEOUT", default=9) == 9
    monkeypatch.setenv("PRIVACY_TIMEOUT", "0")
    assert pp._timeout_env("PRIVACY_TIMEOUT", default=9) == 9
    monkeypatch.setenv("PRIVACY_TIMEOUT", "12")
    assert pp._timeout_env("PRIVACY_TIMEOUT", default=9) == 12

    assert pp._render_command("runner --in {INPUT_VIDEO}", {"INPUT_VIDEO": "a b.mov"}) == [
        "runner",
        "--in",
        "a",
        "b.mov",
    ]
    assert pp._load_json(tmp_path / "missing.json") == {}
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{bad", encoding="utf-8")
    assert pp._load_json(invalid_json) == {}

    class Completed:
        def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    output_json = tmp_path / "command-output.json"
    assert pp._run_json_command(
        command_template="",
        substitutions={"OUTPUT_JSON": output_json},
        timeout_seconds=1,
    ) == {"status": "failed", "reason": "empty_command"}

    def failed_run(*_args: object, **_kwargs: object) -> Completed:
        output_json.write_text('{"detail":"from-json"}', encoding="utf-8")
        return Completed(2, stdout="out" * 2000, stderr="err" * 2000)

    monkeypatch.setattr(pp.subprocess, "run", failed_run)
    failed = pp._run_json_command(
        command_template="runner --out {OUTPUT_JSON}",
        substitutions={"OUTPUT_JSON": output_json},
        timeout_seconds=1,
    )
    assert failed["reason"] == "command_failed:2"
    assert failed["detail"] == "from-json"
    assert len(failed["stdout"]) == 4000

    def succeeded_run(*_args: object, **_kwargs: object) -> Completed:
        output_json.unlink(missing_ok=True)
        return Completed(0, stdout="ok", stderr="")

    monkeypatch.setattr(pp.subprocess, "run", succeeded_run)
    assert pp._run_json_command(
        command_template="runner --out {OUTPUT_JSON}",
        substitutions={"OUTPUT_JSON": output_json},
        timeout_seconds=1,
    )["status"] == "succeeded"

    def succeeded_with_payload(*_args: object, **_kwargs: object) -> Completed:
        output_json.write_text('{"status":"payload"}', encoding="utf-8")
        return Completed(0)

    monkeypatch.setattr(pp.subprocess, "run", succeeded_with_payload)
    assert pp._run_json_command(
        command_template="runner --out {OUTPUT_JSON}",
        substitutions={"OUTPUT_JSON": output_json},
        timeout_seconds=1,
    ) == {"status": "payload"}

    monkeypatch.setenv("PRIVACY_RUNNER_TOKEN", "token-1")
    assert pp._http_runner_headers()["Authorization"] == "Bearer token-1"
    assert pp._runner_url_invalid_reason("") == "runner_url_missing"
    assert pp._runner_url_invalid_reason("not-a-url") == "runner_url_invalid_or_placeholder"
    assert pp._runner_url_invalid_reason("https://replace_me.example") == (
        "runner_url_invalid_or_placeholder"
    )
    assert pp._runner_url_invalid_reason("https://runner.example") is None

    class FakeResponse:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return self._payload

    responses = iter(
        [
            FakeResponse(b'{"status":"ok"}'),
            urllib.error.HTTPError(
                "https://runner.example",
                503,
                "unavailable",
                {},
                BytesIO(b"down"),
            ),
            urllib.error.URLError("offline"),
            FakeResponse(b"not json"),
            FakeResponse(b"[1]"),
        ]
    )

    def fake_urlopen(*_args: object, **_kwargs: object):
        item = next(responses)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(pp.urllib_request, "urlopen", fake_urlopen)
    assert pp._run_http_json(url="https://runner.example", body={}, timeout_seconds=1) == {
        "status": "ok"
    }
    assert pp._run_http_json(url="https://runner.example", body={}, timeout_seconds=1)[
        "reason"
    ] == "http_error:503"
    assert pp._run_http_json(url="https://runner.example", body={}, timeout_seconds=1)[
        "reason"
    ] == "http_unreachable:offline"
    assert pp._run_http_json(url="https://runner.example", body={}, timeout_seconds=1)[
        "reason"
    ] == "http_invalid_json"
    assert pp._run_http_json(url="https://runner.example", body={}, timeout_seconds=1)[
        "reason"
    ] == "http_non_object_json"


def test_privacy_http_runner_preserves_runner_token_and_adds_cloud_run_id_token(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"status":"ok"}'

    def fake_urlopen(request: object, **_kwargs: object) -> FakeResponse:
        captured["headers"] = dict(request.headers)  # type: ignore[attr-defined]
        return FakeResponse()

    def add_cloud_run_header(headers: dict[str, str], *, url: str) -> dict[str, str]:
        assert url == "https://runner.example/run"
        merged = dict(headers)
        merged["X-Serverless-Authorization"] = "Bearer google-id-token"
        return merged

    monkeypatch.setenv("PRIVACY_RUNNER_TOKEN", "runner-token")
    monkeypatch.setattr(pp, "cloud_run_id_token_headers", add_cloud_run_header)
    monkeypatch.setattr(pp.urllib_request, "urlopen", fake_urlopen)

    assert pp._run_http_json(url="https://runner.example/run", body={}, timeout_seconds=1) == {
        "status": "ok"
    }
    headers = captured["headers"]
    assert headers["Authorization"] == "Bearer runner-token"  # type: ignore[index]
    assert headers["X-serverless-authorization"] == "Bearer google-id-token"  # type: ignore[index]


def test_privacy_processing_runner_wrapper_edges(monkeypatch, tmp_path: Path) -> None:
    input_video = tmp_path / "raw" / "walkthrough.mov"
    _write_video(input_video)
    masks_dir = tmp_path / "masks"
    output_json = tmp_path / "runner.json"
    output_video = tmp_path / "out.mov"

    monkeypatch.delenv("PRIVACY_SAM3_URL", raising=False)
    monkeypatch.delenv("PRIVACY_SAM3_COMMAND", raising=False)
    monkeypatch.delenv("SAM3_COMMAND", raising=False)
    assert pp._run_sam3(
        input_video=input_video,
        input_video_uri="gs://bucket/in.mov",
        output_json=output_json,
        output_json_uri="gs://bucket/out.json",
        masks_dir=masks_dir,
        masks_prefix_uri="gs://bucket/masks",
        stage_name="initial",
    )["reason"] == "sam3_runner_not_configured"

    monkeypatch.setenv("PRIVACY_SAM3_URL", "https://sam3.example")
    monkeypatch.setattr(
        pp,
        "_run_http_json",
        lambda **_kwargs: {
            "status": "succeeded",
            "people_detected": False,
            "people_count": 2,
            "mask_paths": "mask.png",
        },
    )
    sam3 = pp._run_sam3(
        input_video=input_video,
        input_video_uri="gs://bucket/in.mov",
        output_json=output_json,
        output_json_uri="gs://bucket/out.json",
        masks_dir=masks_dir,
        masks_prefix_uri="gs://bucket/masks",
        stage_name="initial",
    )
    assert sam3["people_detected"] is True
    assert sam3["mask_paths"] == ["mask.png"]
    monkeypatch.delenv("PRIVACY_SAM3_URL", raising=False)
    monkeypatch.setenv("PRIVACY_SAM3_COMMAND", "sam3 --out {OUTPUT_JSON}")
    monkeypatch.setattr(
        pp,
        "_run_json_command",
        lambda **_kwargs: {
            "status": "succeeded",
            "people_detected": True,
            "people_count": 1,
            "mask_paths": ["mask-command.png"],
        },
    )
    assert pp._run_sam3(
        input_video=input_video,
        input_video_uri="gs://bucket/in.mov",
        output_json=output_json,
        output_json_uri="gs://bucket/out.json",
        masks_dir=masks_dir,
        masks_prefix_uri="gs://bucket/masks",
        stage_name="initial",
    )["mask_paths"] == ["mask-command.png"]

    monkeypatch.delenv("PRIVACY_VIP_URL", raising=False)
    monkeypatch.delenv("PRIVACY_VIP_COMMAND", raising=False)
    monkeypatch.delenv("VIP_COMMAND", raising=False)
    assert pp._run_vip(
        input_video=input_video,
        input_video_uri="gs://bucket/in.mov",
        masks_dir=masks_dir,
        masks_prefix_uri="gs://bucket/masks",
        output_video=output_video,
        output_video_uri="gs://bucket/out.mov",
        output_json=output_json,
        output_json_uri="gs://bucket/out.json",
        arkit_depth_prefix_uri=None,
        arkit_confidence_prefix_uri=None,
        depth_manifest_uri=None,
        confidence_manifest_uri=None,
    )["reason"] == "vip_runner_not_configured"
    monkeypatch.setenv("PRIVACY_VIP_COMMAND", "vip --out {OUTPUT_JSON}")
    monkeypatch.setattr(pp, "_run_json_command", lambda **_kwargs: {"status": "succeeded"})
    vip = pp._run_vip(
        input_video=input_video,
        input_video_uri="gs://bucket/in.mov",
        masks_dir=masks_dir,
        masks_prefix_uri="gs://bucket/masks",
        output_video=output_video,
        output_video_uri="gs://bucket/out.mov",
        output_json=output_json,
        output_json_uri="gs://bucket/out.json",
        arkit_depth_prefix_uri=None,
        arkit_confidence_prefix_uri=None,
        depth_manifest_uri=None,
        confidence_manifest_uri=None,
    )
    assert vip["status"] == "succeeded"
    assert vip["output_video_uri"] == "gs://bucket/out.mov"
    output_video.write_bytes(b"remote-vip")
    monkeypatch.setattr(pp, "_run_json_command", lambda **_kwargs: {"status": "failed"})
    assert pp._run_vip(
        input_video=input_video,
        input_video_uri="gs://bucket/in.mov",
        masks_dir=masks_dir,
        masks_prefix_uri="gs://bucket/masks",
        output_video=output_video,
        output_video_uri="gs://bucket/out.mov",
        output_json=output_json,
        output_json_uri="gs://bucket/out.json",
        arkit_depth_prefix_uri=None,
        arkit_confidence_prefix_uri=None,
        depth_manifest_uri=None,
        confidence_manifest_uri=None,
    )["status"] == "succeeded"
    output_video.unlink()
    assert pp._run_vip(
        input_video=input_video,
        input_video_uri="gs://bucket/in.mov",
        masks_dir=masks_dir,
        masks_prefix_uri="gs://bucket/masks",
        output_video=output_video,
        output_video_uri="gs://bucket/out.mov",
        output_json=output_json,
        output_json_uri="gs://bucket/out.json",
        arkit_depth_prefix_uri=None,
        arkit_confidence_prefix_uri=None,
        depth_manifest_uri=None,
        confidence_manifest_uri=None,
    )["reason"] == "vip_output_missing"
    monkeypatch.setenv("PRIVACY_VIP_URL", "https://vip.example")
    monkeypatch.delenv("PRIVACY_VIP_COMMAND", raising=False)
    output_video.write_bytes(b"vip-url")
    monkeypatch.setattr(pp, "_run_http_json", lambda **_kwargs: {"status": "succeeded"})
    assert pp._run_vip(
        input_video=input_video,
        input_video_uri="gs://bucket/in.mov",
        masks_dir=masks_dir,
        masks_prefix_uri="gs://bucket/masks",
        output_video=output_video,
        output_video_uri="gs://bucket/out.mov",
        output_json=output_json,
        output_json_uri="gs://bucket/out.json",
        arkit_depth_prefix_uri=None,
        arkit_confidence_prefix_uri=None,
        depth_manifest_uri=None,
        confidence_manifest_uri=None,
    )["depth_source"] == "depth_anything"

    monkeypatch.delenv("PRIVACY_DEPTH_ANYTHING_URL", raising=False)
    monkeypatch.delenv("PRIVACY_DEPTH_ANYTHING_COMMAND", raising=False)
    monkeypatch.delenv("DEPTH_ANYTHING_COMMAND", raising=False)
    monkeypatch.delenv("PRIVACY_VIP_URL", raising=False)
    assert pp._run_depth_anything(
        input_video=input_video,
        input_video_uri="gs://bucket/in.mov",
        depth_dir=tmp_path / "depth",
        depth_prefix_uri="gs://bucket/depth",
        confidence_dir=tmp_path / "confidence",
        confidence_prefix_uri="gs://bucket/confidence",
        depth_manifest_path=tmp_path / "depth_manifest.json",
        depth_manifest_uri="gs://bucket/depth_manifest.json",
        confidence_manifest_path=tmp_path / "confidence_manifest.json",
        confidence_manifest_uri="gs://bucket/confidence_manifest.json",
    )["reason"] == "depth_anything_runner_not_configured"
    monkeypatch.setenv("PRIVACY_DEPTH_ANYTHING_COMMAND", "depth --out {DEPTH_MANIFEST}")
    monkeypatch.setattr(pp, "_run_json_command", lambda **_kwargs: {"status": "failed"})
    assert pp._run_depth_anything(
        input_video=input_video,
        input_video_uri="gs://bucket/in.mov",
        depth_dir=tmp_path / "depth",
        depth_prefix_uri="gs://bucket/depth",
        confidence_dir=tmp_path / "confidence",
        confidence_prefix_uri="gs://bucket/confidence",
        depth_manifest_path=tmp_path / "depth_manifest.json",
        depth_manifest_uri="gs://bucket/depth_manifest.json",
        confidence_manifest_path=tmp_path / "confidence_manifest.json",
        confidence_manifest_uri="gs://bucket/confidence_manifest.json",
    )["reason"] == "depth_anything_failed"
    depth_manifest = tmp_path / "depth_manifest.json"
    confidence_manifest = tmp_path / "confidence_manifest.json"
    depth_manifest.write_text("{}", encoding="utf-8")
    confidence_manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        pp,
        "_run_json_command",
        lambda **_kwargs: {"status": "succeeded", "frame_count": 4},
    )
    depth = pp._run_depth_anything(
        input_video=input_video,
        input_video_uri="gs://bucket/in.mov",
        depth_dir=tmp_path / "depth",
        depth_prefix_uri="gs://bucket/depth",
        confidence_dir=tmp_path / "confidence",
        confidence_prefix_uri="gs://bucket/confidence",
        depth_manifest_path=depth_manifest,
        depth_manifest_uri="gs://bucket/depth_manifest.json",
        confidence_manifest_path=confidence_manifest,
        confidence_manifest_uri="gs://bucket/confidence_manifest.json",
    )
    assert depth["status"] == "succeeded"
    assert depth["frame_count"] == 4
    monkeypatch.setenv("PRIVACY_DEPTH_ANYTHING_URL", "https://depth.example")
    monkeypatch.setattr(
        pp,
        "_run_http_json",
        lambda **_kwargs: {"status": "succeeded", "frame_count": 5},
    )
    assert pp._run_depth_anything(
        input_video=input_video,
        input_video_uri="gs://bucket/in.mov",
        depth_dir=tmp_path / "depth",
        depth_prefix_uri="gs://bucket/depth",
        confidence_dir=tmp_path / "confidence",
        confidence_prefix_uri="gs://bucket/confidence",
        depth_manifest_path=depth_manifest,
        depth_manifest_uri="gs://bucket/depth_manifest.json",
        confidence_manifest_path=confidence_manifest,
        confidence_manifest_uri="gs://bucket/confidence_manifest.json",
    )["frame_count"] == 5

    monkeypatch.delenv("PRIVACY_DEEPPRIVACY2_URL", raising=False)
    monkeypatch.delenv("PRIVACY_DEEPPRIVACY2_COMMAND", raising=False)
    monkeypatch.delenv("DEEPPRIVACY2_COMMAND", raising=False)
    output_video.unlink(missing_ok=True)
    assert pp._run_deepprivacy2(
        input_video=input_video,
        input_video_uri="gs://bucket/in.mov",
        output_video=output_video,
        output_video_uri="gs://bucket/deep.mov",
        output_json=output_json,
        output_json_uri="gs://bucket/deep.json",
    )["reason"] == "deepprivacy2_runner_not_configured"
    monkeypatch.setenv("PRIVACY_DEEPPRIVACY2_COMMAND", "deep --out {OUTPUT_JSON}")
    monkeypatch.setattr(
        pp,
        "_run_json_command",
        lambda **_kwargs: {"status": "succeeded", "face_anonymized_segments": "segment-1"},
    )
    deepprivacy = pp._run_deepprivacy2(
        input_video=input_video,
        input_video_uri="gs://bucket/in.mov",
        output_video=output_video,
        output_video_uri="gs://bucket/deep.mov",
        output_json=output_json,
        output_json_uri="gs://bucket/deep.json",
    )
    assert deepprivacy["output_video_uri"] == "gs://bucket/deep.mov"
    assert deepprivacy["face_anonymized_segments"] == ["segment-1"]
    output_video.write_bytes(b"deepprivacy")
    monkeypatch.setenv("PRIVACY_DEEPPRIVACY2_URL", "https://deep.example")
    monkeypatch.delenv("PRIVACY_DEEPPRIVACY2_COMMAND", raising=False)
    monkeypatch.setattr(
        pp,
        "_run_http_json",
        lambda **_kwargs: {"status": "succeeded", "face_anonymized_segments": ["url-segment"]},
    )
    assert pp._run_deepprivacy2(
        input_video=input_video,
        input_video_uri="gs://bucket/in.mov",
        output_video=output_video,
        output_video_uri="gs://bucket/deep.mov",
        output_json=output_json,
        output_json_uri="gs://bucket/deep.json",
    )["face_anonymized_segments"] == ["url-segment"]
    monkeypatch.delenv("PRIVACY_DEEPPRIVACY2_URL", raising=False)
    monkeypatch.setenv("PRIVACY_DEEPPRIVACY2_COMMAND", "deep --out {OUTPUT_JSON}")
    output_video.write_bytes(b"deepprivacy-local")
    monkeypatch.setattr(pp, "_run_json_command", lambda **_kwargs: {"status": "failed"})
    assert pp._run_deepprivacy2(
        input_video=input_video,
        input_video_uri="gs://bucket/in.mov",
        output_video=output_video,
        output_video_uri="gs://bucket/deep.mov",
        output_json=output_json,
        output_json_uri="gs://bucket/deep.json",
    )["status"] == "succeeded"
    output_video.unlink()
    assert pp._run_deepprivacy2(
        input_video=input_video,
        input_video_uri="gs://bucket/in.mov",
        output_video=output_video,
        output_video_uri="gs://bucket/deep.mov",
        output_json=output_json,
        output_json_uri="gs://bucket/deep.json",
    )["reason"] == "deepprivacy2_output_missing"


def test_privacy_processing_remote_output_and_copy_fallback(monkeypatch, tmp_path: Path) -> None:
    capture_root = tmp_path / "storage" / "bucket" / "scenes" / "scene-1" / "captures" / "cap-1"
    remote = capture_root / "privacy" / "remote.mov"
    destination = capture_root / "privacy" / "local.mov"
    _write_video(remote, b"remote")
    pp._ensure_remote_output_local(
        capture_root=capture_root,
        output_uri="gs://bucket/scenes/scene-1/captures/cap-1/privacy/remote.mov",
        destination=destination,
    )
    assert destination.read_bytes() == b"remote"

    source = tmp_path / "copy-source.mov"
    dest = tmp_path / "copy-dest.mov"
    _write_video(source, b"copy")
    monkeypatch.setattr(pp.shutil, "which", lambda _name: None)
    assert pp._copy_or_remux_video(source, dest)["mode"] == "copy"
    assert dest.read_bytes() == b"copy"

    remux_dest = tmp_path / "remux-dest.mov"

    class RemuxCompleted:
        returncode = 0
        stdout = ""
        stderr = ""

    def remux_run(args: list[str], **_kwargs: object) -> RemuxCompleted:
        Path(args[-1]).write_bytes(b"remux")
        return RemuxCompleted()

    monkeypatch.setattr(pp.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(pp.subprocess, "run", remux_run)
    assert pp._copy_or_remux_video(source, remux_dest)["mode"] == "remux"

    monkeypatch.setattr(pp.shutil, "which", lambda _name: None)
    assert pp._run_local_full_frame_redaction(source, tmp_path / "redacted.mov")[
        "reason"
    ] == "ffmpeg_not_found"

    class FailedRedaction:
        returncode = 3
        stderr = "ffmpeg failed"

    monkeypatch.setattr(pp.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(pp.subprocess, "run", lambda *_args, **_kwargs: FailedRedaction())
    failed = pp._run_local_full_frame_redaction(source, tmp_path / "redacted-failed.mov")
    assert failed["reason"] == "ffmpeg_redaction_failed:3"


def test_privacy_postprocess_fail_closed_branches(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PRIVACY_PIPELINE_ENABLED", "true")

    missing_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "missing"
    missing_result = _run_postprocess_for(missing_root, missing_root / "pipeline", None)
    assert missing_result["reason"] == "raw_video_missing"

    disabled_root, disabled_pipeline, disabled_raw = _capture_paths(tmp_path, "disabled")
    monkeypatch.delenv("PRIVACY_PIPELINE_ENABLED", raising=False)
    disabled = _run_postprocess_for(disabled_root, disabled_pipeline, disabled_raw)
    assert disabled["reason"] == "privacy_pipeline_disabled"
    monkeypatch.setenv("PRIVACY_PIPELINE_ENABLED", "true")

    redaction_root, redaction_pipeline, redaction_raw = _capture_paths(tmp_path, "redaction")
    monkeypatch.setenv("PRIVACY_LOCAL_FULL_FRAME_REDACTION_ENABLED", "true")
    monkeypatch.setattr(
        pp,
        "_run_local_full_frame_redaction",
        lambda *_args, **_kwargs: {"status": "failed", "reason": "local_failed"},
    )
    redaction = _run_postprocess_for(redaction_root, redaction_pipeline, redaction_raw)
    assert redaction["reason"] == "local_failed"
    monkeypatch.delenv("PRIVACY_LOCAL_FULL_FRAME_REDACTION_ENABLED", raising=False)

    initial_root, initial_pipeline, initial_raw = _capture_paths(tmp_path, "initial")
    monkeypatch.setattr(pp, "_run_sam3", lambda **_kwargs: {"status": "failed", "reason": "sam3_failed"})
    initial = _run_postprocess_for(initial_root, initial_pipeline, initial_raw)
    assert initial["reason"] == "sam3_failed"

    depth_root, depth_pipeline, depth_raw = _capture_paths(tmp_path, "depth")
    monkeypatch.setattr(
        pp,
        "_run_sam3",
        lambda **_kwargs: {"status": "succeeded", "people_detected": False, "people_count": 0},
    )
    monkeypatch.setattr(pp, "_run_depth_anything", lambda **_kwargs: {"status": "failed", "reason": "depth_failed"})
    depth = _run_postprocess_for(depth_root, depth_pipeline, depth_raw)
    assert depth["reason"] == "depth_failed"

    vip_root, vip_pipeline, vip_raw = _capture_paths(tmp_path, "vip")
    monkeypatch.setattr(
        pp,
        "_run_sam3",
        lambda **_kwargs: {"status": "succeeded", "people_detected": True, "people_count": 1},
    )
    monkeypatch.setattr(pp, "_run_depth_anything", lambda **_kwargs: _depth_anything_result())
    monkeypatch.setattr(pp, "_run_vip", lambda **_kwargs: {"status": "failed", "reason": "vip_failed"})
    vip = _run_postprocess_for(vip_root, vip_pipeline, vip_raw)
    assert vip["reason"] == "vip_failed"

    vip_verify_root, vip_verify_pipeline, vip_verify_raw = _capture_paths(tmp_path, "vip-verify")
    sam3_results = iter(
        [
            {"status": "succeeded", "people_detected": True, "people_count": 1},
            {"status": "failed", "reason": "vip_verify_failed"},
        ]
    )
    monkeypatch.setattr(pp, "_run_sam3", lambda **_kwargs: next(sam3_results))

    def vip_success(**kwargs):
        _write_video(kwargs["output_video"], b"vip")
        return {"status": "succeeded", "output_video": str(kwargs["output_video"])}

    monkeypatch.setattr(pp, "_run_vip", vip_success)
    vip_verify = _run_postprocess_for(vip_verify_root, vip_verify_pipeline, vip_verify_raw)
    assert vip_verify["reason"] == "vip_verify_failed"

    deep_root, deep_pipeline, deep_raw = _capture_paths(tmp_path, "deep")
    sam3_results = iter(
        [
            {"status": "succeeded", "people_detected": True, "people_count": 1},
            {"status": "succeeded", "people_detected": True, "people_count": 1},
        ]
    )
    monkeypatch.setattr(pp, "_run_sam3", lambda **_kwargs: next(sam3_results))
    monkeypatch.setattr(pp, "_run_vip", vip_success)
    monkeypatch.setattr(pp, "_run_deepprivacy2", lambda **_kwargs: {"status": "failed", "reason": "deep_failed"})
    deep = _run_postprocess_for(deep_root, deep_pipeline, deep_raw)
    assert deep["reason"] == "deep_failed"

    fallback_root, fallback_pipeline, fallback_raw = _capture_paths(tmp_path, "fallback")
    sam3_results = iter(
        [
            {"status": "succeeded", "people_detected": True, "people_count": 1},
            {"status": "succeeded", "people_detected": True, "people_count": 1},
            {"status": "failed", "reason": "fallback_verify_failed"},
        ]
    )
    monkeypatch.setattr(pp, "_run_sam3", lambda **_kwargs: next(sam3_results))

    def deep_success(**kwargs):
        _write_video(kwargs["output_video"], b"deep")
        return {
            "status": "succeeded",
            "output_video": str(kwargs["output_video"]),
            "face_anonymized_segments": ["segment"],
        }

    monkeypatch.setattr(pp, "_run_deepprivacy2", deep_success)
    fallback = _run_postprocess_for(fallback_root, fallback_pipeline, fallback_raw)
    assert fallback["reason"] == "fallback_verify_failed"

    segments_root, segments_pipeline, segments_raw = _capture_paths(tmp_path, "segments")
    sam3_results = iter(
        [
            {"status": "succeeded", "people_detected": True, "people_count": 1},
            {"status": "succeeded", "people_detected": True, "people_count": 1},
            {"status": "succeeded", "people_detected": False, "people_count": 0},
        ]
    )
    monkeypatch.setattr(pp, "_run_sam3", lambda **_kwargs: next(sam3_results))

    def deep_without_segments(**kwargs):
        _write_video(kwargs["output_video"], b"deep")
        return {"status": "succeeded", "output_video": str(kwargs["output_video"])}

    monkeypatch.setattr(pp, "_run_deepprivacy2", deep_without_segments)
    segments = _run_postprocess_for(segments_root, segments_pipeline, segments_raw)
    assert segments["reason"] == "deepprivacy2_face_segments_missing"


def test_privacy_postprocess_fails_closed_with_no_runners_configured(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """With every runner URL/command unset and local redaction disabled, the
    pipeline must fail closed at the first detection backend and must never
    publish a final walkthrough derived from the raw capture.

    No backend, GPU, or ffmpeg invocation is allowed: ffmpeg-touching helpers
    are stubbed to raise so the test both avoids the ffmpeg dependency and
    proves no raw-capture passthrough/redaction occurs.
    """

    capture_root, pipeline_dir, raw_video = _capture_paths(tmp_path, "no-runners")

    monkeypatch.setenv("PRIVACY_PIPELINE_ENABLED", "true")
    # Fail-closed is the default, but assert it explicitly for this edge.
    monkeypatch.setenv("PRIVACY_FAIL_CLOSED", "true")
    # Local full-frame redaction must be OFF so we exercise the runner path.
    monkeypatch.delenv("PRIVACY_LOCAL_FULL_FRAME_REDACTION_ENABLED", raising=False)
    monkeypatch.delenv("BLUEPRINT_PRIVACY_LOCAL_FULL_FRAME_REDACTION", raising=False)

    # Clear every runner URL and command (PRIVACY_-prefixed + legacy bare).
    for var in (
        "PRIVACY_SAM3_URL",
        "PRIVACY_SAM3_COMMAND",
        "SAM3_COMMAND",
        "PRIVACY_VIP_URL",
        "PRIVACY_VIP_COMMAND",
        "VIP_COMMAND",
        "PRIVACY_DEPTH_ANYTHING_URL",
        "PRIVACY_DEPTH_ANYTHING_COMMAND",
        "DEPTH_ANYTHING_COMMAND",
        "PRIVACY_DEEPPRIVACY2_URL",
        "PRIVACY_DEEPPRIVACY2_COMMAND",
        "DEEPPRIVACY2_COMMAND",
    ):
        monkeypatch.delenv(var, raising=False)

    # Guard: ffmpeg / raw-passthrough helpers must never run. If any fires it
    # would mean a final walkthrough was being built from the raw capture.
    def _forbidden(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("ffmpeg/raw passthrough must not run when failing closed")

    monkeypatch.setattr(pp, "_copy_or_remux_video", _forbidden)
    monkeypatch.setattr(pp, "_run_local_full_frame_redaction", _forbidden)
    # Belt-and-suspenders: even if a helper slipped through, ffmpeg is absent.
    monkeypatch.setattr(pp.shutil, "which", lambda _name: None)

    result = run_privacy_postprocess(
        bucket="bucket",
        scene_id="scene-1",
        capture_id="no-runners",
        capture_root=capture_root,
        pipeline_dir=pipeline_dir,
        raw_video_path=raw_video,
    )

    # Documented fail-closed status + the first-backend not-configured reason.
    assert result["status"] == "failed_closed"
    assert result["reason"] == "sam3_runner_not_configured"
    assert result["fail_closed"] is True
    # No privacy-safe walkthrough was selected, and the raw capture was retained.
    assert result["privacy_processed_video_uri"] is None
    assert result["world_model_video_uri"] is None
    assert result["raw_retained"] is True

    # Crucially: no final walkthrough exists, so nothing leaks from raw capture.
    assert not (capture_root / "privacy" / "final_walkthrough.mov").exists()

    # The persisted manifest and verification report agree on fail-closed.
    manifest = json.loads((pipeline_dir / "privacy_processing_manifest.json").read_text(encoding="utf-8"))
    verification = json.loads((pipeline_dir / "privacy_verification_report.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed_closed"
    assert manifest["reason"] == "sam3_runner_not_configured"
    assert verification["status"] == "failed_closed"
    assert verification["initial_detection"]["reason"] == "sam3_runner_not_configured"
