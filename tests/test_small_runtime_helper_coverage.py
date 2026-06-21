from __future__ import annotations

import io
import json
import runpy
import sys
from pathlib import Path
from urllib import error as urllib_error

import pytest

from blueprint_pipeline.agent_runtime import skill_sync
from blueprint_pipeline.common import PipelineError
from blueprint_pipeline.geometry_sources import (
    _load_jsonl,
    _manifest_relative_path,
    _parse_intrinsics_from_arkit_row,
    _safe_float,
    _safe_int,
    _zero_pad_frame_id,
    load_capture_geometry,
    resolve_geometry_source,
)
from blueprint_pipeline.local_capture import resolve_local_capture_context
from blueprint_pipeline.video_to_world_client import run_video_to_world_provider


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) if not isinstance(row, str) else row for row in rows) + "\n\n",
        encoding="utf-8",
    )


def _capture_context(tmp_path: Path):
    capture_root = tmp_path / "storage" / "bucket" / "scenes" / "site-1" / "captures" / "cap-1"
    capture_root.mkdir(parents=True)
    return resolve_local_capture_context(capture_root)


class _Response:
    def __init__(self, body: object) -> None:
        self._body = body if isinstance(body, bytes) else str(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return self._body


def test_video_to_world_client_payload_success_and_runner_helpers(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(req.headers)
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _Response(json.dumps({"status": "succeeded", "artifact_uri": "gs://bucket/out"}))

    monkeypatch.setenv("VIDEO_TO_WORLD_URL", "https://runner.example/root/")
    monkeypatch.delenv("VIDEO_TO_WORLD_RUNNER_TOKEN", raising=False)
    monkeypatch.setenv("PRIVACY_RUNNER_TOKEN", "privacy-token")
    monkeypatch.setenv("VIDEO_TO_WORLD_TIMEOUT_SECONDS", "bad-int")
    monkeypatch.setattr("blueprint_pipeline.video_to_world_client.urllib_request.urlopen", fake_urlopen)

    payload = run_video_to_world_provider(
        video_path=tmp_path / "raw.mov",
        video_uri="gs://bucket/raw.mov",
        geometry_root=tmp_path / "geometry",
        dynamic_mask_manifest_path=tmp_path / "masks" / "manifest.json",
        dynamic_mask_manifest_uri="gs://bucket/capture/masks/manifest.json",
        provider="da3",
        model="depth-anything-3",
        execution_mode="remote",
        video_probe={"fps": 24},
    )

    assert payload["artifact_uri"] == "gs://bucket/out"
    assert captured["url"] == "https://runner.example/root/run"
    assert captured["timeout"] == 7200
    assert captured["headers"]["Authorization"] == "Bearer privacy-token"
    assert captured["payload"] == {
        "input_video_path": str(tmp_path / "raw.mov"),
        "input_video_uri": "gs://bucket/raw.mov",
        "geometry_root_path": str(tmp_path / "geometry"),
        "geometry_root_uri": "gs://bucket/capture",
        "dynamic_mask_manifest_path": str(tmp_path / "masks" / "manifest.json"),
        "dynamic_mask_manifest_uri": "gs://bucket/capture/masks/manifest.json",
        "provider": "da3",
        "model": "depth-anything-3",
        "execution_mode": "remote",
        "video_probe": {"fps": 24},
    }


@pytest.mark.parametrize(
    ("body", "error_match"),
    [
        ("not-json", "video_to_world_invalid_json"),
        (json.dumps(["bad"]), "video_to_world_invalid_payload"),
        (json.dumps({"status": "failed", "reason": "model_crashed"}), "model_crashed"),
        (json.dumps({}), "video_to_world_failed"),
    ],
)
def test_video_to_world_client_response_failures(monkeypatch, tmp_path: Path, body: str, error_match: str) -> None:
    monkeypatch.setenv("VIDEO_TO_WORLD_URL", "https://runner.example")
    monkeypatch.setenv("VIDEO_TO_WORLD_TIMEOUT_SECONDS", "4")
    monkeypatch.setattr(
        "blueprint_pipeline.video_to_world_client.urllib_request.urlopen",
        lambda *_args, **_kwargs: _Response(body),
    )

    with pytest.raises(RuntimeError, match=error_match):
        run_video_to_world_provider(
            video_path=tmp_path / "raw.mov",
            video_uri="gs://bucket/raw.mov",
            geometry_root=tmp_path / "geometry",
            dynamic_mask_manifest_path=tmp_path / "masks.json",
            dynamic_mask_manifest_uri="gs://bucket/masks/manifest.json",
            provider="provider",
            model="model",
            execution_mode="remote",
            video_probe={},
        )


@pytest.mark.parametrize(
    "urlopen_error, error_match",
    [
        (
            urllib_error.HTTPError(
                "https://runner.example/run",
                503,
                "unavailable",
                {},
                io.BytesIO(b"service unavailable with extra diagnostic text"),
            ),
            "video_to_world_http_503:service unavailable",
        ),
        (urllib_error.URLError("network down"), "video_to_world_unreachable:network down"),
    ],
)
def test_video_to_world_client_transport_failures(monkeypatch, tmp_path: Path, urlopen_error: Exception, error_match: str) -> None:
    monkeypatch.setenv("VIDEO_TO_WORLD_URL", "https://runner.example")

    def raise_error(*_args, **_kwargs):
        raise urlopen_error

    monkeypatch.setattr("blueprint_pipeline.video_to_world_client.urllib_request.urlopen", raise_error)

    with pytest.raises(RuntimeError, match=error_match):
        run_video_to_world_provider(
            video_path=tmp_path / "raw.mov",
            video_uri="gs://bucket/raw.mov",
            geometry_root=tmp_path / "geometry",
            dynamic_mask_manifest_path=tmp_path / "masks.json",
            dynamic_mask_manifest_uri="gs://bucket/masks/manifest.json",
            provider="provider",
            model="model",
            execution_mode="remote",
            video_probe={},
        )


def test_video_to_world_client_blocks_without_runner_url(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("VIDEO_TO_WORLD_URL", raising=False)

    with pytest.raises(RuntimeError, match="video_to_world_runner_not_configured"):
        run_video_to_world_provider(
            video_path=tmp_path / "raw.mov",
            video_uri="gs://bucket/raw.mov",
            geometry_root=tmp_path / "geometry",
            dynamic_mask_manifest_path=tmp_path / "masks.json",
            dynamic_mask_manifest_uri="gs://bucket/masks/manifest.json",
            provider="provider",
            model="model",
            execution_mode="remote",
            video_probe={},
        )


def _build_skillpack_repo(root: Path) -> None:
    _write_json(
        root / "skillpacks" / "ops" / "skillpack_manifest.json",
        {"name": "ops", "source_root": "skill_src", "skills": ["drive", "", "dock"]},
    )
    for skill in ["drive", "dock"]:
        skill_dir = root / "skill_src" / skill
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(f"# {skill}\n", encoding="utf-8")


def test_skill_sync_copies_skillpacks_and_main_reports_success(tmp_path: Path, capsys) -> None:
    _build_skillpack_repo(tmp_path)

    assert skill_sync._repo_root().name == "BlueprintCapturePipeline"
    result = skill_sync.sync_skill_pack(tmp_path)

    assert result["schema_version"] == "v1"
    assert result["skill_count"] == 2
    assert result["skillpacks"] == ["ops"]
    assert result["skills"] == ["drive", "dock"]
    for target in [tmp_path / ".claude" / "skills", tmp_path / ".agents" / "skills"]:
        assert (target / "drive" / "SKILL.md").read_text(encoding="utf-8") == "# drive\n"
        assert (target / "dock" / "SKILL.md").read_text(encoding="utf-8") == "# dock\n"

    assert skill_sync.main(["--repo-root", str(tmp_path)]) == 0
    assert "[skill-sync] synced 2 skills" in capsys.readouterr().out


def test_skill_sync_validates_missing_and_duplicate_skillpacks(tmp_path: Path, capsys) -> None:
    with pytest.raises(PipelineError, match="No skill pack manifests"):
        skill_sync.load_skillpack_manifests(tmp_path)

    _write_json(
        tmp_path / "skillpacks" / "a" / "skillpack_manifest.json",
        {"name": "a", "source_root": "a_src", "skills": ["drive"]},
    )
    _write_json(
        tmp_path / "skillpacks" / "b" / "skillpack_manifest.json",
        {"name": "b", "source_root": "b_src", "skills": ["drive"]},
    )
    with pytest.raises(PipelineError, match="Duplicate skill 'drive'"):
        skill_sync.sync_skill_pack(tmp_path)

    (tmp_path / "skillpacks" / "b" / "skillpack_manifest.json").unlink()
    assert skill_sync.main(["--repo-root", str(tmp_path)]) == 1
    assert "[skill-sync] FAILED: Skill source root is missing" in capsys.readouterr().out


def test_skill_sync_validates_missing_declared_skill_and_main_guard(tmp_path: Path, monkeypatch) -> None:
    _write_json(
        tmp_path / "skillpacks" / "ops" / "skillpack_manifest.json",
        {"name": "ops", "source_root": "skill_src", "skills": ["missing"]},
    )
    (tmp_path / "skill_src").mkdir()

    with pytest.raises(PipelineError, match="Skill source is missing"):
        skill_sync.sync_skill_pack(tmp_path)

    monkeypatch.setattr(sys, "argv", ["skill_sync", "--repo-root", str(tmp_path)])
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("blueprint_pipeline.agent_runtime.skill_sync", run_name="__main__")
    assert excinfo.value.code == 1


def test_geometry_source_private_helpers_cover_invalid_inputs(tmp_path: Path) -> None:
    class DigitText:
        def __str__(self) -> str:
            return "42"

    assert _load_jsonl(tmp_path / "missing.jsonl") == []
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text('\n{"ok": true}\nnot-json\n[1, 2]\n{"second": 2}\n', encoding="utf-8")
    assert _load_jsonl(jsonl) == [{"ok": True}, {"second": 2}]

    assert _zero_pad_frame_id(None, fallback=7) == "000007"
    assert _zero_pad_frame_id("") == "000000"
    assert _zero_pad_frame_id("42") == "000042"
    assert _zero_pad_frame_id(DigitText()) == "000042"
    assert _zero_pad_frame_id("frame-A", fallback=9) == "frame-A"
    assert _safe_float("bad", 1.25) == 1.25
    assert _safe_int(object(), 11) == 11

    assert _parse_intrinsics_from_arkit_row(
        {"intrinsics": {"fx": "1.5", "fy": 2, "cx": 3, "cy": 4, "width": "640", "height": "480"}}
    ) == {"fx": 1.5, "fy": 2.0, "cx": 3.0, "cy": 4.0, "width": 640, "height": 480}
    assert _parse_intrinsics_from_arkit_row({"intrinsics": {"fx": 1}}) is None
    assert _parse_intrinsics_from_arkit_row({"intrinsics": [1, 0, 0, 0, 2, 0, 3, 4, 1], "imageResolution": [640, 480]}) == {
        "fx": 1.0,
        "fy": 2.0,
        "cx": 3.0,
        "cy": 4.0,
        "width": 640,
        "height": 480,
    }
    assert _parse_intrinsics_from_arkit_row({"intrinsics": [1, 2]}) is None

    _write_json(tmp_path / "manifest.json", {"frames": ["bad", {"frame_id": "000002", "depth_path": "d/2.png"}]})
    assert _manifest_relative_path(tmp_path / "manifest.json", "depth_path", "000002") == "d/2.png"
    assert _manifest_relative_path(tmp_path / "manifest.json", "depth_path", "000003") is None
    _write_json(tmp_path / "not_frames.json", {"frames": {}})
    assert _manifest_relative_path(tmp_path / "not_frames.json", "depth_path", "000002") is None


def test_geometry_sources_load_pipeline_geometry_and_descriptor_fallbacks(tmp_path: Path) -> None:
    context = _capture_context(tmp_path)
    descriptor = {
        "geometry_source": "descriptor-source",
        "coordinate_frame_session_id": "session-123",
        "quality": {"geometry_ready": True},
    }
    assert resolve_geometry_source(context=context, descriptor=descriptor) == "descriptor-source"

    geometry_root = context.pipeline_root / "geometry"
    _write_json(
        geometry_root / "geometry_summary.json",
        {
            "geometry_source": "video_to_world",
            "fallback_used": True,
            "fallback_kind": "local_sfm",
            "provider_native_result": True,
            "contract_ready_for_world_model": True,
            "internal_fallback_ready": True,
            "geometry_live_ready": True,
            "external_market_ready": False,
            "site_faithful_market_ready": False,
            "ready_for_world_model": True,
        },
    )
    _write_json(geometry_root / "camera" / "intrinsics.json", {"fx": 100.0})
    _write_jsonl(
        geometry_root / "frames" / "frame_index.jsonl",
        [
            {
                "frame_index": 3,
                "timestamp_seconds": "1.25",
                "sharpness_score": 88,
                "world_mapping_status": "mapped",
                "anchor_observations": [{"anchor": "door"}],
                "image_path": "frames/000003.jpg",
                "depth_path": "depth/000003.png",
                "confidence_path": "conf/000003.png",
                "pose_confidence": "0.8",
            }
        ],
    )
    _write_jsonl(
        geometry_root / "camera" / "poses.jsonl",
        [{"frame_index": 3, "frame_id": "frame-3", "timestamp": "1.30", "world_from_camera": [1, 0, 0]}],
    )

    geometry = load_capture_geometry(context=context, descriptor=descriptor)

    assert geometry["source"] == "video_to_world"
    assert geometry["coordinate_frame_session_id"] == "session-123"
    assert geometry["fallback_used"] is True
    assert geometry["ready_for_world_model"] is True
    assert geometry["poses"] == [{"frame_id": "frame-3", "frame_index": 3, "timestamp": 1.3, "T_world_camera": [1, 0, 0]}]
    assert geometry["frame_meta"]["frame-3"]["anchorObservations"] == [{"anchor": "door"}]
    assert geometry["intrinsics"] == {"fx": 100.0}


def test_geometry_sources_load_arkit_and_arcore_geometry(tmp_path: Path) -> None:
    context = _capture_context(tmp_path)
    arkit_root = context.raw_root / "arkit"
    (arkit_root / "depth").mkdir(parents=True)
    (arkit_root / "confidence").mkdir(parents=True)
    (arkit_root / "depth" / "000005.png").write_bytes(b"depth")
    (arkit_root / "confidence" / "000005.png").write_bytes(b"confidence")
    _write_json(arkit_root / "intrinsics.json", {"camera": "arkit"})
    _write_jsonl(
        arkit_root / "frames.jsonl",
        [
            {
                "frameIndex": 5,
                "timestamp": "2.5",
                "intrinsics": [10, 0, 0, 0, 11, 0, 12, 13, 1],
                "imageResolution": [1920, 1080],
                "trackingState": "normal",
                "sharpnessScore": 91,
                "relocalizationEvent": True,
                "worldMappingStatus": "mapped",
                "anchorObservations": ["counter"],
            }
        ],
    )
    _write_jsonl(arkit_root / "poses.jsonl", [{"frameIndex": 5, "timestamp": "2.55", "transform": [0, 1, 0]}])

    arkit_geometry = load_capture_geometry(
        context=context,
        descriptor={"metadata": {"capture_topology": {"captureSessionId": "arkit-session"}}, "geometry_ready": True},
    )

    assert arkit_geometry["source"] == "arkit"
    assert arkit_geometry["coordinate_frame_session_id"] == "arkit-session"
    assert arkit_geometry["frame_meta"]["000005"]["depth_path"].endswith("000005.png")
    assert arkit_geometry["frame_meta"]["000005"]["intrinsics_payload"]["width"] == 1920
    assert arkit_geometry["poses"][0]["T_world_camera"] == [0, 1, 0]
    assert arkit_geometry["ready_for_world_model"] is True

    (arkit_root / "poses.jsonl").unlink()
    arcore_root = context.raw_root / "arcore"
    _write_json(arcore_root / "session_intrinsics.json", {"camera": "arcore"})
    _write_json(arcore_root / "depth_manifest.json", {"frames": [{"frame_id": "000009", "depth_path": "depth/9.png"}]})
    _write_json(arcore_root / "confidence_manifest.json", {"frames": [{"frame_id": "000009", "confidence_path": "conf/9.png"}]})
    _write_jsonl(
        arcore_root / "frames.jsonl",
        [{"frame_index": "9", "t_capture_sec": "4.5", "tracking_state": "TRACKING"}],
    )
    _write_jsonl(
        arcore_root / "poses.jsonl",
        [{"frame_index": 9, "frame_id": "9", "timestamp_seconds": "4.6", "T_world_camera": [9]}],
    )

    arcore_geometry = load_capture_geometry(
        context=context,
        descriptor={"quality": {"world_model_candidate": True}, "coordinate_frame_session_id": "arcore-session"},
    )

    assert arcore_geometry["source"] == "arcore"
    assert arcore_geometry["coordinate_frame_session_id"] == "arcore-session"
    assert arcore_geometry["frame_meta"]["000009"]["depth_path"] == "depth/9.png"
    assert arcore_geometry["frame_meta"]["000009"]["confidence_path"] == "conf/9.png"
    assert arcore_geometry["frame_meta"]["000009"]["pose_confidence"] == 1.0
    assert arcore_geometry["poses"][0]["frame_id"] == "000009"
    assert arcore_geometry["ready_for_world_model"] is True
