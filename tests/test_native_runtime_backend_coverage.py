from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
pytest.importorskip("PIL")
from PIL import Image

from blueprint_pipeline import native_runtime_backend as nrb
from blueprint_pipeline.native_runtime_backend import NativeRuntimeConfig, NativeWorldModelRuntimeStore


def _payload(*, launchable: bool = True) -> dict:
    return {
        "spec": {
            "schema_version": "v1",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "canonical_package_uri": "gs://bucket/site-worlds/site-1/canonical.json",
            "runtime_eligibility": {
                "launchable": launchable,
                "default_backend": "native_world_model",
                "blockers": [] if launchable else ["runtime_blocked"],
            },
        },
        "registration": {
            "schema_version": "v1",
            "site_world_id": "siteworld-1",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
        },
        "health": {
            "schema_version": "v1",
            "site_world_id": "siteworld-1",
            "launchable": launchable,
            "status": "healthy" if launchable else "blocked",
            "blockers": [] if launchable else ["health_blocked"],
        },
    }


def _store(tmp_path: Path) -> NativeWorldModelRuntimeStore:
    return NativeWorldModelRuntimeStore(
        NativeRuntimeConfig(
            root_dir=tmp_path / "runtime",
            base_url="http://127.0.0.1:8791",
            ws_base_url="ws://127.0.0.1:8791",
        )
    )


def _write_site_index(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "reference_id": "ref-1",
                "frame_uri": "gs://bucket/frames/000001.jpg",
                "T_world_camera": np.eye(4).tolist(),
                "intrinsics": {
                    "width": 320,
                    "height": 240,
                    "fx": 500.0,
                    "fy": 500.0,
                    "cx": 160.0,
                    "cy": 120.0,
                },
                "reference_frame": "site_world",
                "camera_frame": "head_rgb",
                "translation_unit": "m",
                "reprojection_error_px": 0.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_native_runtime_low_level_helper_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / "missing.bin"
    assert nrb._read_bytes_if_available(missing) is None

    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert nrb._json_read(list_json) == {}

    assert nrb._optional_existing_path("gs://bucket/key") is None
    existing = tmp_path / "existing"
    existing.write_text("ok", encoding="utf-8")
    assert nrb._optional_existing_path(str(existing)) == existing.resolve()

    monkeypatch.setattr(nrb, "_module_available", lambda name: name == "torch")
    monkeypatch.setenv("BLUEPRINT_NATIVE_RUNTIME_BACKEND", "cosmos_wam")
    monkeypatch.setenv("NATIVE_WORLD_MODEL_READY", "1")
    monkeypatch.setenv("NATIVE_WORLD_MODEL_CHECKPOINT_READY", "1")
    readiness = nrb._runtime_readiness()
    assert readiness["ready"] is False
    assert "missing_native_runtime_packages" in readiness["notes"]

    blockers = nrb._runtime_blockers(
        {"runtime_eligibility": {"blockers": ["runtime_blocked", "", "health_blocked"]}},
        {"blockers": ["health_blocked", " "]},
    )
    assert blockers == ["health_blocked", "runtime_blocked"]

    assert nrb._parse_gs_uri("https://example.invalid/file") is None
    assert nrb._parse_gs_uri("gs://") is None
    assert nrb._parse_gs_uri("gs://bucket/path/file") == ("bucket", "path/file")
    assert nrb._bucket_from_site_world({"siteWorldRegistrationUri": "gs://camel/path"}) == "camel"
    monkeypatch.setenv("GCS_BUCKET", " ")
    assert nrb._bucket_from_site_world({}) == "vast-local"

    monkeypatch.setenv("GCS_ROOT", str(tmp_path / "configured-storage"))
    assert nrb._default_storage_root() == (tmp_path / "configured-storage").resolve()

    monkeypatch.delenv("COSMOS_OFFICIAL_REPO_ROOT", raising=False)
    assert nrb._find_cosmos_repo() is None

    repo = tmp_path / "cosmos-repo"
    inference = repo / "examples" / "inference.py"
    python_bin = repo / ".venv" / "bin" / "python"
    inference.parent.mkdir(parents=True)
    python_bin.parent.mkdir(parents=True)
    inference.write_text("print('ok')\n", encoding="utf-8")
    python_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    python_bin.chmod(0o755)
    monkeypatch.setenv("COSMOS_OFFICIAL_REPO_ROOT", str(repo))
    assert nrb._find_cosmos_repo() == (repo, python_bin)

    monkeypatch.setattr(nrb.os, "access", lambda _path, _mode: False)
    assert nrb._find_cosmos_repo() is None
    monkeypatch.setattr(
        nrb.os,
        "access",
        lambda path, _mode: str(path).endswith("examples/inference.py"),
    )
    assert nrb._find_cosmos_repo() is None
    monkeypatch.setattr(nrb.os, "access", os.access)
    python_bin.chmod(0o644)
    assert nrb._find_cosmos_repo() is None

    def raise_is_file(self: Path) -> bool:
        if str(self).endswith("examples/inference.py"):
            raise OSError("stat failed")
        return Path.exists(self) and not Path.is_dir(self)

    monkeypatch.setattr(nrb.Path, "is_file", raise_is_file)
    assert nrb._find_cosmos_repo() is None
    monkeypatch.undo()

    assert nrb._action_to_delta_T({"type": "turn_left", "magnitude": 90})[0, 0] == pytest.approx(0.0)
    assert nrb._action_to_delta_T({"type": "move_backward", "magnitude": 2.0})[2, 3] == 2.0
    assert nrb._action_to_delta_T({"type": "move_up", "magnitude": 1.25})[1, 3] == -1.25
    assert nrb._action_to_delta_T({"type": "move_down", "magnitude": 1.5})[1, 3] == 1.5
    assert nrb._action_to_delta_T({"type": "turn_right", "magnitude": 90})[0, 0] == pytest.approx(0.0)
    list_delta = nrb._action_to_delta_T([1.0, 2.0, 0.5])
    assert list_delta[2, 3] == -1.0
    assert list_delta[0, 3] == 2.0

    applied = nrb._apply_action(np.eye(4), {"type": "move_forward", "magnitude": 0.75})
    assert applied[2, 3] == pytest.approx(-0.75)

    index_path = tmp_path / "site_reference_index.jsonl"
    _write_site_index(index_path)
    assert nrb._pose_from_site_index(index_path).shape == (4, 4)
    assert nrb._pose_from_site_index(tmp_path / "bad.jsonl") is None

    intrinsics, height, width = nrb._intrinsics_from_site_index(index_path)
    assert intrinsics["width"] == 320
    assert (height, width) == (240, 320)
    with pytest.raises(RuntimeError, match="site_index_camera_calibration_unreadable"):
        nrb._intrinsics_from_site_index(tmp_path / "missing.jsonl")

    invalid_index = tmp_path / "invalid-camera-index.jsonl"
    invalid_index.write_text(
        json.dumps(
            {
                "T_world_camera": np.eye(4).tolist(),
                "intrinsics": {"width": 320, "height": 240, "fx": 500.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="site_index_camera_calibration_blocked"):
        nrb._intrinsics_from_site_index(invalid_index)

    bad_manifest = tmp_path / "bad_site_reference_manifest.json"
    bad_manifest.write_text(json.dumps({"schema_version": "bad"}), encoding="utf-8")
    empty_index = tmp_path / "empty_site_reference_index.jsonl"
    empty_index.write_text("", encoding="utf-8")
    adapter = nrb._site_reference_runtime_adapter(
        site_id="site-1",
        manifest_path=bad_manifest,
        site_index_path=empty_index,
        storage_root=tmp_path,
        bucket="bucket",
        runtime_readiness={"ready": False, "notes": []},
    )
    assert "site_reference_manifest_schema_invalid" in adapter["blockers"]
    assert "site_reference_query_empty" in adapter["blockers"]

    arkit_index = tmp_path / "arkit_index.jsonl"
    arkit_index.write_text(json.dumps({"geometry_source": "arkit"}) + "\n", encoding="utf-8")
    assert nrb._site_reference_geometry_state(arkit_index)["state"] == "not_applicable"
    vtw_index = tmp_path / "vtw_index.jsonl"
    vtw_index.write_text(json.dumps({"geometry_source": "video_to_world"}) + "\n", encoding="utf-8")
    assert nrb._site_reference_geometry_state(vtw_index)["state"] == "ready"
    blank_index = tmp_path / "blank_index.jsonl"
    blank_index.write_text("\n" + json.dumps({"geometry_source": "local_sfm"}) + "\n", encoding="utf-8")
    # local_sfm is synthetic/fallback geometry and is never treated as usable
    # (degraded-but-indexable); it's fully blocked like any other non-live source.
    assert nrb._site_reference_geometry_state(blank_index)["state"] == "blocked"

    original_open = Path.open

    def raise_open(self: Path, *args, **kwargs):
        if self == blank_index:
            raise OSError("unreadable")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(nrb.Path, "open", raise_open)
    assert nrb._site_reference_geometry_state(blank_index)["blockers"] == ["site_reference_index_unreadable"]
    monkeypatch.setattr(nrb.Path, "open", original_open)

    bad_pose_index = tmp_path / "bad_pose_index.jsonl"
    bad_pose_index.write_text(json.dumps({"T_world_camera": [1, 2, 3]}) + "\n", encoding="utf-8")
    assert nrb._site_reference_query_count(site_index_path=bad_pose_index, storage_root=tmp_path, bucket="bucket") == 0
    monkeypatch.setattr(
        "blueprint_pipeline.synthesis.retrieval_query.query_site",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("query failed")),
    )
    assert nrb._site_reference_query_count(site_index_path=index_path, storage_root=tmp_path, bucket="bucket") == 0

    storage_root = tmp_path / "storage"
    flat_index = storage_root / "sites" / "site-1" / "reference_memory" / "site_reference_index.jsonl"
    _write_site_index(flat_index)
    assert nrb._resolve_site_index_path("site-1", "scene-1", "capture-1", storage_root, "bucket") == flat_index

    manifest = storage_root / "bucket" / "sites" / "site-1" / "reference_memory" / "site_reference_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}", encoding="utf-8")
    assert nrb._resolve_site_reference_manifest_path("site-1", storage_root, "bucket") == manifest
    assert nrb._resolve_site_id({"site_id": "direct-site"}, "scene-1", "capture-1", storage_root, "bucket") == "direct-site"

    descriptor = storage_root / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1" / "capture_descriptor.json"
    descriptor.parent.mkdir(parents=True, exist_ok=True)
    descriptor.write_text(json.dumps({"metadata": {"site_identity": {"site_id": "descriptor-site"}}}), encoding="utf-8")
    assert nrb._resolve_site_id({}, "scene-1", "capture-1", storage_root, "bucket") == "descriptor-site"


def test_runtime_readiness_is_scoped_to_the_selected_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BLUEPRINT_NATIVE_RUNTIME_BACKEND", "site_splat")
    monkeypatch.delenv("NATIVE_WORLD_MODEL_SYNTHESIS_MODE", raising=False)
    monkeypatch.delenv("COSMOS_OFFICIAL_REPO_ROOT", raising=False)
    monkeypatch.setattr(
        nrb,
        "_module_available",
        lambda name: name in {"numpy", "PIL"},
    )

    readiness = nrb._runtime_readiness()
    assert readiness["ready"] is True
    assert readiness["selected_runtime_path"] == "splat_only"
    assert readiness["cosmos_ready"] is False
    assert readiness["notes"] == []

    monkeypatch.setattr(nrb, "_module_available", lambda name: name == "numpy")
    readiness = nrb._runtime_readiness()
    assert readiness["ready"] is False
    assert readiness["notes"] == ["missing_site_splat_runtime_packages"]


def test_native_runtime_store_state_and_rollout_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    monkeypatch.setenv("BLUEPRINT_NATIVE_RUNTIME_BACKEND", "cosmos_wam")

    monkeypatch.setattr(nrb, "_runtime_readiness", lambda: {"ready": False})
    assert store.prewarm_runtime()["status"] == "skipped"

    from blueprint_pipeline.synthesis import cosmos_inference

    monkeypatch.setattr(nrb, "_runtime_readiness", lambda: {"ready": True})
    monkeypatch.setattr(cosmos_inference, "prewarm_cosmos_model", lambda: {"backend": "fake"})
    assert store.prewarm_runtime()["status"] == "ready"

    def raise_prewarm() -> dict:
        raise RuntimeError("boom")

    monkeypatch.setattr(cosmos_inference, "prewarm_cosmos_model", raise_prewarm)
    assert store.prewarm_runtime()["status"] == "failed"

    with pytest.raises(RuntimeError, match="site_world_id"):
        store.register_site_world_package(spec={}, registration={}, health={})
    with pytest.raises(FileNotFoundError):
        store.load_site_world("missing")
    with pytest.raises(FileNotFoundError):
        store.load_site_world_health("missing")
    with pytest.raises(FileNotFoundError):
        store._load_session_state("missing")

    blocked_payload = _payload(launchable=False)
    store.register_site_world_package(**blocked_payload)
    with pytest.raises(RuntimeError, match="health_blocked"):
        store.create_session("siteworld-1")

    assert store._claim_generation_owner("s1") == (True, "s1")
    assert store._claim_generation_owner("s2") == (False, "s1")
    store._release_generation_owner("s2")
    assert store._generation_owner_session_id == "s1"
    store._release_generation_owner("s1")
    assert store._generation_owner_session_id is None

    bad_pose = store._session_target_pose({"camera_pose_matrix": [["bad"]]}, None)
    assert bad_pose.shape == (4, 4)

    pose = {"x": 1.0, "y": 2.0, "z": 3.0, "yaw": 0.25, "pitch": 0.1}
    assert store._fallback_pose_update(pose, {"type": "move_forward", "magnitude": 2.0})["z"] == 1.0
    assert store._fallback_pose_update(pose, {"type": "move_backward", "magnitude": 2.0})["z"] == 5.0
    assert store._fallback_pose_update(pose, {"type": "move_up", "magnitude": 0.5})["y"] == 1.5
    assert store._fallback_pose_update(pose, {"type": "move_down", "magnitude": 0.5})["y"] == 2.5
    assert store._fallback_pose_update(pose, {"type": "turn_right", "magnitude": 90})["yaw"] < 0.25
    assert store._fallback_pose_update(pose, [1.0, 2.0, 0.5])["x"] == 3.0

    assert store._latest_render_path({}) is None
    render = tmp_path / "render.png"
    render.write_bytes(b"png")
    assert store._latest_render_path({"latest_render_path": str(render)}) == render
    assert store._latest_render_path({"latest_render_path": str(tmp_path / "missing.png")}) is None

    monkeypatch.delenv("BLUEPRINT_NATIVE_RUNTIME_BACKEND", raising=False)
    monkeypatch.setenv("NATIVE_WORLD_MODEL_SYNTHESIS_MODE", "forced_mode")
    with pytest.raises(ValueError, match="native_runtime_backend_unknown"):
        store._live_synthesis_mode()
    monkeypatch.delenv("NATIVE_WORLD_MODEL_SYNTHESIS_MODE", raising=False)
    monkeypatch.setattr(nrb, "_runtime_readiness", lambda: {"ready": True})
    assert store._live_synthesis_mode() == "splat_only"
    monkeypatch.setenv("BLUEPRINT_NATIVE_RUNTIME_BACKEND", "cosmos_wam")
    assert store._live_synthesis_mode() == "cosmos_i2w"
    monkeypatch.setenv("NATIVE_WORLD_MODEL_ENABLE_TRUTHFUL_PREVIEW", "1")
    assert store._uses_truthful_preview() is True
    monkeypatch.setenv("NATIVE_WORLD_MODEL_ENABLE_TRUTHFUL_PREVIEW", "0")
    assert store._uses_truthful_preview() is False
    assert store._preview_generation_mode() == "cosmos_i2w"
    monkeypatch.delenv("NATIVE_WORLD_MODEL_ENABLE_TRUTHFUL_PREVIEW", raising=False)

    rollout = store._rollout_defaults()
    store._replace_chunk(rollout, {"chunk_id": "chunk-0001", "chunk_index": 1})
    store._replace_chunk(rollout, {"chunk_id": "chunk-0000", "chunk_index": 0})
    rollout["active_chunk_id"] = "chunk-0000"
    assert store._chunk_record(rollout, "chunk-0001")["chunk_index"] == 1
    assert store._chunk_record(rollout, "missing") is None
    assert store._current_chunk(rollout)["chunk_id"] == "chunk-0000"
    assert store._current_chunk({"active_chunk_id": ""}) is None
    assert store._should_queue_more_chunks({"queued_chunk_ids": ["chunk-queued"]}) is False

    now = nrb._utc_now_ms()
    playback = {
        "rollout": {
            **store._rollout_defaults(),
            "active_chunk_id": "chunk-0000",
            "buffered_chunk_ids": ["chunk-0000", "chunk-0001"],
            "chunks": [
                {"chunk_id": "chunk-0000", "chunk_index": 0, "activated_at_ms": now - 10, "duration_ms": 1},
                {"chunk_id": "chunk-0001", "chunk_index": 1, "duration_ms": 1, "media_type": "video/mp4", "render_source": "next"},
            ],
        }
    }
    store._refresh_rollout_playback(playback)
    assert playback["rollout"]["active_chunk_id"] == "chunk-0001"
    assert playback["rollout"]["status"] == "playing"

    missing_active = {
        "rollout": {
            **store._rollout_defaults(),
            "active_chunk_id": "chunk-0000",
            "buffered_chunk_ids": ["other-chunk"],
            "chunks": [{"chunk_id": "chunk-0000", "chunk_index": 0, "activated_at_ms": now - 10, "duration_ms": 1}],
        }
    }
    store._refresh_rollout_playback(missing_active)
    assert missing_active["rollout"]["underrun"] is True

    underrun = {
        "rollout": {
            **store._rollout_defaults(),
            "active_chunk_id": "chunk-0000",
            "buffered_chunk_ids": ["chunk-0000"],
            "chunks": [{"chunk_id": "chunk-0000", "chunk_index": 0, "activated_at_ms": now - 10, "duration_ms": 1}],
        }
    }
    store._refresh_rollout_playback(underrun)
    assert underrun["rollout"]["underrun"] is True

    first_buffer = {
        "rollout": {
            **store._rollout_defaults(),
            "buffered_chunk_ids": ["chunk-0000"],
            "chunks": [{"chunk_id": "chunk-0000", "chunk_index": 0, "media_type": "image/png", "render_source": "bootstrap"}],
        }
    }
    store._refresh_rollout_playback(first_buffer)
    assert first_buffer["rollout"]["active_chunk_id"] == "chunk-0000"

    queued = {"rollout": {**store._rollout_defaults(), "queued_chunk_ids": ["chunk-0002"]}}
    store._refresh_rollout_playback(queued)
    assert queued["rollout"]["status"] == "buffering"

    empty_after_chunks = {"rollout": {**store._rollout_defaults(), "chunk_count": 1}}
    store._refresh_rollout_playback(empty_after_chunks)
    assert empty_after_chunks["rollout"]["status"] == "underrun"

    index_path = tmp_path / "pose_index.jsonl"
    _write_site_index(index_path)
    assert store._session_target_pose({"camera_pose_matrix": [[1]]}, index_path).shape == (4, 4)

    monkeypatch.setattr(
        "blueprint_pipeline.synthesis.retrieval_query.query_site",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("query failed")),
    )
    assert store._query_references_for_pose(
        site_index_path=index_path,
        target_T_world_camera=np.eye(4),
        storage_root=tmp_path,
        bucket="bucket",
        k=1,
    ) == []


def test_native_runtime_generation_and_media_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    store.register_site_world_package(**_payload())
    session_id = "session-coverage"
    index_path = (
        tmp_path
        / "bucket"
        / "sites"
        / "site-1"
        / "reference_memory"
        / "site_reference_index.jsonl"
    )
    _write_site_index(index_path)
    state = {
        "session_id": session_id,
        "site_world_id": "siteworld-1",
        "scene_id": "scene-1",
        "capture_id": "capture-1",
        "site_id": "site-1",
        "site_index_path": str(index_path),
        "storage_root": str(tmp_path),
        "storage_bucket": "bucket",
        "status": "ready",
        "step_count": 0,
        "step_index": 0,
        "pose": {},
        "camera_pose_matrix": np.eye(4).tolist(),
        "rollout": store._rollout_defaults(),
    }
    store._store_session_state(session_id, state)

    from blueprint_pipeline.synthesis import cosmos_inference, synthesize

    def fail_synthesize(**_kwargs):
        raise RuntimeError("synthesis boom")

    monkeypatch.setattr(synthesize, "synthesize_view", fail_synthesize)
    failed_state = {**state, "pending_step_index": 1}
    store._store_session_state(session_id, failed_state)
    store._synthesize_step_async(
        session_id=session_id,
        step_index=1,
        site_id="site-1",
        storage_root=tmp_path,
        bucket="bucket",
        target_T_world_camera=np.eye(4),
        target_intrinsics={"fx": 1.0},
        target_h=8,
        target_w=8,
    )
    assert store._load_session_state(session_id)["synthesis_status"] == "failed"

    stale_state = {**state, "pending_step_index": 2, "latest_synthesis": None}
    store._store_session_state(session_id, stale_state)
    store._synthesize_step_async(
        session_id=session_id,
        step_index=1,
        site_id="site-1",
        storage_root=tmp_path,
        bucket="bucket",
        target_T_world_camera=np.eye(4),
        target_intrinsics={"fx": 1.0},
        target_h=8,
        target_w=8,
    )
    assert store._load_session_state(session_id)["latest_synthesis"] is None

    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"video")
    monkeypatch.setattr(store, "_extract_tail_frame", lambda _video, output: output.write_bytes(b"tail"))
    assert store._copy_video_to_chunk(session_id=session_id, source_path=tmp_path / "missing.mp4", chunk_id="chunk-missing", chunk_index=0, render_source="x") is None
    copied = store._copy_video_to_chunk(
        session_id=session_id,
        source_path=source_video,
        chunk_id="chunk-0000",
        chunk_index=0,
        render_source="copied",
    )
    assert copied and copied["media_type"] == "video/mp4"

    store_for_tail = _store(tmp_path / "tail-store")
    def raise_run(*_args, **_kwargs):
        raise OSError("ffmpeg missing")

    monkeypatch.setattr(nrb.subprocess, "run", raise_run)
    store_for_tail._extract_tail_frame(source_video, tmp_path / "tail.png")
    assert store._image_to_mp4(tmp_path / "image.png", tmp_path / "out.mp4", 100) is False
    assert store._convert_to_fmp4(source_video, tmp_path / "converted.mp4") is False

    def ok_run(args, **_kwargs):
        Path(args[-1]).write_bytes(b"media")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(nrb.subprocess, "run", ok_run)
    image = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(image)
    assert store._image_to_mp4(image, tmp_path / "still.mp4", 100) is True
    assert store._convert_to_fmp4(source_video, tmp_path / "converted.mp4") is True

    monkeypatch.setattr(store, "_find_prebuilt_cosmos_video", lambda _site_world_id: source_video)
    bootstrap = store._bootstrap_video_chunk(session_id, "siteworld-1")
    assert bootstrap and bootstrap["render_source"] == "bootstrap_prebuilt_video"

    frame = tmp_path / "conditioning.png"
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(frame)
    monkeypatch.setattr(store, "_find_prebuilt_cosmos_video", lambda _site_world_id: None)
    monkeypatch.setattr(store, "_find_conditioning_frame", lambda _site_world_id: frame)
    monkeypatch.setattr(store, "_image_to_mp4", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(store, "_extract_tail_frame", lambda _video, output: output.write_bytes(b"tail"))
    video_conditioning = store._bootstrap_video_chunk(session_id, "siteworld-1")
    assert video_conditioning and video_conditioning["tail_path"]
    monkeypatch.setattr(store, "_image_to_mp4", lambda *_args, **_kwargs: False)
    conditioning = store._bootstrap_video_chunk(session_id, "siteworld-1")
    assert conditioning and conditioning["media_type"] == "image/png"

    runtime_video = tmp_path / "runtime.mp4"
    runtime_video.write_bytes(b"runtime")
    monkeypatch.setattr(nrb, "_find_cosmos_repo", lambda: (tmp_path / "repo", tmp_path / "repo" / "python"))
    monkeypatch.setattr(store, "_find_conditioning_frame", lambda _site_world_id: tmp_path / "future-frame.jpg")
    monkeypatch.setattr(store, "_run_cosmos_inference_sync", lambda **_kwargs: [])
    nrb._json_write(store._cosmos_status_path(session_id), {"video": str(runtime_video)})
    runtime_bootstrap = store._bootstrap_video_chunk(session_id, "siteworld-1")
    assert runtime_bootstrap and runtime_bootstrap["render_source"] == "bootstrap_runtime_video"

    store._store_session_state(session_id, {**state, "rollout": {**store._rollout_defaults(), "buffered_chunk_ids": ["chunk-0000"]}})
    store._ensure_initial_rollout_chunk(session_id, "siteworld-1")
    assert store._load_session_state(session_id)["rollout"]["buffered_chunk_ids"] == ["chunk-0000"]

    store._store_session_state(session_id, {**state, "status": "ready", "rollout": store._rollout_defaults()})
    monkeypatch.setattr(store, "_bootstrap_video_chunk", lambda *_args: None)
    store._ensure_initial_rollout_chunk(session_id, "siteworld-1")
    assert store._load_session_state(session_id)["rollout"]["status"] == "buffering"

    pending_bootstrap = {**state, "status": "synthesizing", "pending_step_index": 1, "rollout": store._rollout_defaults()}
    store._store_session_state(session_id, pending_bootstrap)
    store._ensure_initial_rollout_chunk(session_id, "siteworld-1")
    assert store._load_session_state(session_id)["pending_step_index"] == 1

    ready_chunk = {"chunk_id": "chunk-ready", "chunk_index": 0, "media_path": str(source_video), "render_source": "ready"}
    store._store_session_state(session_id, {**state, "status": "ready", "rollout": store._rollout_defaults()})
    monkeypatch.setattr(store, "_bootstrap_video_chunk", lambda *_args: ready_chunk)
    store._ensure_initial_rollout_chunk(session_id, "siteworld-1")
    assert store._load_session_state(session_id)["rollout"]["active_chunk_id"] == "chunk-ready"

    def racing_bootstrap(*_args):
        racing_state = store._load_session_state(session_id)
        racing_state["rollout"] = {**store._rollout_defaults(), "buffered_chunk_ids": ["already-ready"]}
        store._store_session_state(session_id, racing_state)
        return ready_chunk

    store._store_session_state(session_id, {**state, "status": "ready", "rollout": store._rollout_defaults()})
    monkeypatch.setattr(store, "_bootstrap_video_chunk", racing_bootstrap)
    store._ensure_initial_rollout_chunk(session_id, "siteworld-1")
    assert store._load_session_state(session_id)["rollout"]["buffered_chunk_ids"] == ["already-ready"]

    refinement_state = {
        **state,
        "rollout": {
            **store._rollout_defaults(),
            "chunks": [{"chunk_id": "chunk-refine", "chunk_index": 0, "media_path": "preview.png", "preview_media_path": "preview.png"}],
        },
    }
    store._store_session_state(session_id, refinement_state)
    store._generation_owner_session_id = "other-session"
    store._refine_chunk_async(
        session_id=session_id,
        chunk_id="chunk-refine",
        chunk_index=0,
        site_id="site-1",
        storage_root=tmp_path,
        bucket="bucket",
        target_T_world_camera=np.eye(4),
        lookahead_T_world_camera=np.eye(4),
        target_intrinsics={"fx": 1.0},
        target_h=8,
        target_w=8,
        previous_tail_path=None,
        lookahead_ref_uris=[],
        grounding_refs=[],
        lookahead_refs=[],
        horizon=[],
        control={},
    )
    skipped = store._load_session_state(session_id)["rollout"]
    assert skipped["refinement_status"] == "skipped"
    assert store.drain_media_events(session_id)[-1]["event"] == "chunk_refinement_skipped"

    store._generation_owner_session_id = None
    monkeypatch.setattr(cosmos_inference, "load_cosmos_model", lambda: (_ for _ in ()).throw(RuntimeError("model down")))
    store._store_session_state(session_id, refinement_state)
    store._refine_chunk_async(
        session_id=session_id,
        chunk_id="chunk-refine",
        chunk_index=0,
        site_id="site-1",
        storage_root=tmp_path,
        bucket="bucket",
        target_T_world_camera=np.eye(4),
        lookahead_T_world_camera=np.eye(4),
        target_intrinsics={"fx": 1.0},
        target_h=8,
        target_w=8,
        previous_tail_path=None,
        lookahead_ref_uris=[],
        grounding_refs=[],
        lookahead_refs=[],
        horizon=[],
        control={},
    )
    assert store._load_session_state(session_id)["rollout"]["refinement_status"] == "failed"

    def no_output_synthesize(**_kwargs):
        return {"status": "completed", "reason": "missing output"}

    monkeypatch.setattr(synthesize, "synthesize_view", no_output_synthesize)
    monkeypatch.setattr(cosmos_inference, "load_cosmos_model", lambda: {"backend": "fake"})
    monkeypatch.setattr(cosmos_inference, "describe_cosmos_model", lambda _model: {"worker_backend": "fake-worker"})
    store._generation_owner_session_id = None
    store._store_session_state(session_id, refinement_state)
    store._refine_chunk_async(
        session_id=session_id,
        chunk_id="chunk-refine",
        chunk_index=0,
        site_id="site-1",
        storage_root=tmp_path,
        bucket="bucket",
        target_T_world_camera=np.eye(4),
        lookahead_T_world_camera=np.eye(4),
        target_intrinsics={"fx": 1.0},
        target_h=8,
        target_w=8,
        previous_tail_path=None,
        lookahead_ref_uris=[],
        grounding_refs=[],
        lookahead_refs=[],
        horizon=[],
        control={},
    )
    assert store._load_session_state(session_id)["rollout"]["refinement_status"] == "failed"

    def successful_synthesize(**kwargs):
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        refined_png = output_path
        refined_png.write_bytes(b"png")
        return {"status": "completed", "output_path": str(refined_png.with_suffix(".raw"))}

    monkeypatch.setattr(cosmos_inference, "load_cosmos_model", lambda: {"backend": "fake"})
    monkeypatch.setattr(cosmos_inference, "describe_cosmos_model", lambda _model: {"worker_backend": "fake-worker"})
    monkeypatch.setattr(synthesize, "synthesize_view", successful_synthesize)
    monkeypatch.setattr(store, "_image_to_mp4", lambda *_args, **_kwargs: False)
    store._generation_owner_session_id = None
    store._store_session_state(session_id, refinement_state)
    store._refine_chunk_async(
        session_id=session_id,
        chunk_id="chunk-refine",
        chunk_index=0,
        site_id="site-1",
        storage_root=tmp_path,
        bucket="bucket",
        target_T_world_camera=np.eye(4),
        lookahead_T_world_camera=np.eye(4),
        target_intrinsics={"fx": 1.0},
        target_h=8,
        target_w=8,
        previous_tail_path=None,
        lookahead_ref_uris=[],
        grounding_refs=[],
        lookahead_refs=[],
        horizon=[],
        control={},
    )
    assert store._load_session_state(session_id)["rollout"]["refinement_status"] == "completed"

    mp4_refined = tmp_path / "refined.mp4"
    mp4_refined.write_bytes(b"mp4")
    fmp4_refined = tmp_path / "refined_fmp4.mp4"

    def synthesize_mp4(**_kwargs):
        return {"status": "completed", "video_path": str(mp4_refined)}

    monkeypatch.setattr(synthesize, "synthesize_view", synthesize_mp4)
    monkeypatch.setattr(store, "_convert_to_fmp4", lambda _input, output: bool(output.write_bytes(b"fmp4")))
    promote_state = {
        **state,
        "rollout": {
            **store._rollout_defaults(),
            "active_chunk_id": "other",
            "chunks": [{"chunk_id": "chunk-refine", "chunk_index": 0, "media_path": "preview.png", "preview_media_path": "preview.png"}],
        },
    }
    store._generation_owner_session_id = None
    store._store_session_state(session_id, promote_state)
    store._refine_chunk_async(
        session_id=session_id,
        chunk_id="chunk-refine",
        chunk_index=0,
        site_id="site-1",
        storage_root=tmp_path,
        bucket="bucket",
        target_T_world_camera=np.eye(4),
        lookahead_T_world_camera=np.eye(4),
        target_intrinsics={"fx": 1.0},
        target_h=8,
        target_w=8,
        previous_tail_path=None,
        lookahead_ref_uris=[],
        grounding_refs=[],
        lookahead_refs=[],
        horizon=[],
        control={},
    )
    promoted_chunk = store._load_session_state(session_id)["rollout"]["chunks"][0]
    assert promoted_chunk["media_path"] == str(fmp4_refined.resolve())

    store._generation_owner_session_id = None
    store._store_session_state(session_id, {**state, "rollout": store._rollout_defaults()})
    store._refine_chunk_async(
        session_id=session_id,
        chunk_id="missing-chunk",
        chunk_index=0,
        site_id="site-1",
        storage_root=tmp_path,
        bucket="bucket",
        target_T_world_camera=np.eye(4),
        lookahead_T_world_camera=np.eye(4),
        target_intrinsics={"fx": 1.0},
        target_h=8,
        target_w=8,
        previous_tail_path=None,
        lookahead_ref_uris=[],
        grounding_refs=[],
        lookahead_refs=[],
        horizon=[],
        control={},
    )

    queued_state = {**state, "rollout": {**store._rollout_defaults(), "queued_chunk_ids": ["chunk-existing"]}}
    store._store_session_state(session_id, queued_state)
    store._generate_next_chunk(session_id)
    assert store._load_session_state(session_id)["rollout"]["queued_chunk_ids"] == ["chunk-existing"]

    monkeypatch.setattr(store, "_preview_generation_mode", lambda: "splat_only")
    def png_synthesize(**kwargs):
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"png")
        return {"status": "completed", "output_path": str(output_path), "mode": kwargs["mode"]}

    monkeypatch.setattr(synthesize, "synthesize_view", png_synthesize)
    monkeypatch.setattr(store, "_image_to_mp4", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(store, "_convert_to_fmp4", lambda *_args, **_kwargs: False)
    monkeypatch.setenv("NATIVE_WORLD_MODEL_ENABLE_COSMOS_REFINEMENT", "0")
    png_session_id = "session-png-generation"
    store._store_session_state(png_session_id, {**state, "session_id": png_session_id, "rollout": store._rollout_defaults()})
    store._generate_next_chunk(png_session_id)
    generated = store._load_session_state(png_session_id)["rollout"]["chunks"][-1]
    assert generated["media_type"] == "image/png"
    store._release_generation_owner(png_session_id)

    monkeypatch.setattr(store, "_preview_generation_mode", lambda: "cosmos_i2w")
    monkeypatch.setattr(cosmos_inference, "load_cosmos_model", lambda: {"backend": "fake"})
    monkeypatch.setattr(cosmos_inference, "describe_cosmos_model", lambda _model: {"backend": "fake"})
    monkeypatch.setattr(synthesize, "synthesize_view", fail_synthesize)
    cosmos_fail_session_id = "session-cosmos-failure"
    store._store_session_state(cosmos_fail_session_id, {**state, "session_id": cosmos_fail_session_id, "rollout": store._rollout_defaults()})
    store._generate_next_chunk(cosmos_fail_session_id)
    assert store._load_session_state(cosmos_fail_session_id)["rollout"]["status"] == "failed"
    assert store._generation_owner_session_id is None

    owner_busy_session_id = "session-owner-busy"
    store._generation_owner_session_id = "other-owner"
    store._store_session_state(owner_busy_session_id, {**state, "session_id": owner_busy_session_id, "rollout": store._rollout_defaults()})
    store._generate_next_chunk(owner_busy_session_id)
    assert "generation_worker_owned_by" in store._load_session_state(owner_busy_session_id)["failure_reason"]
    store._generation_owner_session_id = None

    storage_root = tmp_path / "step-storage"
    step_index = storage_root / "bucket" / "sites" / "site-1" / "reference_memory" / "site_reference_index.jsonl"
    _write_site_index(step_index)
    reset_state = {
        **state,
        "site_world_id": "siteworld-1",
        "site_id": "site-1",
        "storage_root": str(storage_root),
        "storage_bucket": "bucket",
        "step_count": 3,
        "rollout": {**store._rollout_defaults(), "active_chunk_id": "old"},
    }
    store._store_session_state("reset-session", reset_state)

    class DummyThread:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def start(self):
            return None

    monkeypatch.setattr(nrb.threading, "Thread", DummyThread)
    reset = store.reset_session(
        "reset-session",
        task_id="task-2",
        scenario_id="scenario-2",
        start_state_id="start-2",
    )
    assert reset["step_count"] == 0
    assert reset["task_id"] == "task-2"
    assert reset["camera_pose_matrix"] == np.eye(4).tolist()

    monkeypatch.setattr(store, "_query_references_for_pose", lambda **_kwargs: [{"frame_id": "ref"}])
    monkeypatch.setattr(store, "_queue_chunk_generation", lambda _session_id: None)
    step_state = {**reset_state, "site_index_path": None, "rollout": store._rollout_defaults()}
    store._store_session_state("step-session", step_state)
    stepped = store.step_session("step-session", action={"type": "move_forward", "magnitude": 1.0})
    assert stepped["pending_step_index"] == 4

    control_state = {**reset_state, "site_index_path": None, "rollout": store._rollout_defaults()}
    store._store_session_state("control-session", control_state)
    controlled = store.control_session("control-session", control={"vx": 0.1, "durationMs": 500})
    assert controlled["rollout"]["lookahead_anchor"] == {"frame_id": "ref"}


def test_native_runtime_cosmos_file_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store(tmp_path)
    store.register_site_world_package(**_payload())
    gcs_root = tmp_path / "gcs"
    monkeypatch.setenv("GCS_ROOT", str(gcs_root))

    pipeline_base = gcs_root / "vast-local" / "scenes" / "scene-1" / "captures" / "capture-1" / "pipeline"
    prebuilt_mp4 = pipeline_base / "cosmos_single_capture_smoke" / "renders" / "video_bootstrap_0000.mp4"
    prebuilt_mp4.parent.mkdir(parents=True)
    prebuilt_mp4.write_bytes(b"mp4")
    assert store._find_prebuilt_cosmos_video("siteworld-1") is None
    monkeypatch.setenv("NATIVE_WORLD_MODEL_ALLOW_PREBUILT_BOOTSTRAP_VIDEO", "1")
    assert store._allow_prebuilt_bootstrap_video() is True
    assert store._find_prebuilt_cosmos_video("siteworld-1") == prebuilt_mp4
    assert store._find_prebuilt_cosmos_video("missing-site") is None

    prebuilt_mp4.unlink()
    fallback = gcs_root / "manual_cosmos_probe_official" / "blueprint_probe.mp4"
    fallback.parent.mkdir(parents=True)
    fallback.write_bytes(b"fallback")
    assert store._find_prebuilt_cosmos_video("siteworld-1") == fallback
    fallback.unlink()
    assert store._find_prebuilt_cosmos_video("siteworld-1") is None

    conditioning = pipeline_base / "cosmos_single_capture_smoke" / "video_bootstrap_frames" / "frame_0000.jpg"
    conditioning.parent.mkdir(parents=True)
    Image.new("RGB", (6, 6), color=(1, 2, 3)).save(conditioning)
    assert store._find_conditioning_frame("siteworld-1") == conditioning
    assert store._find_conditioning_frame("missing-site") is None

    original_is_file = Path.is_file

    def raise_for_candidates(self: Path) -> bool:
        text = str(self)
        if "video_bootstrap_0000" in text or "blueprint_probe.mp4" in text or "frame_0000.jpg" in text:
            raise OSError("stat failed")
        return original_is_file(self)

    monkeypatch.setattr(nrb.Path, "is_file", raise_for_candidates)
    assert store._find_prebuilt_cosmos_video("siteworld-1") is None
    assert store._find_conditioning_frame("siteworld-1") is None
    monkeypatch.setattr(nrb.Path, "is_file", original_is_file)

    def extract_success(args, **_kwargs):
        frames_dir = Path(args[-2]).parent
        frames_dir.mkdir(parents=True, exist_ok=True)
        (frames_dir / "frame_0001.png").write_bytes(b"frame")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(nrb.subprocess, "run", extract_success)
    assert store._extract_frames_from_video(tmp_path / "video.mp4", tmp_path / "frames-success")
    monkeypatch.setattr(nrb.subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("missing")))
    assert store._extract_frames_from_video(tmp_path / "video.mp4", tmp_path / "frames") == []

    single_frames = store._extract_single_frame(conditioning, tmp_path / "single")
    assert single_frames and single_frames[0].is_file()

    invalid = tmp_path / "invalid.jpg"
    invalid.write_text("not image", encoding="utf-8")
    fallback_frames = store._extract_single_frame(invalid, tmp_path / "fallback")
    assert fallback_frames and fallback_frames[0].is_file()
    monkeypatch.setattr(nrb.shutil, "copy2", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("copy failed")))
    assert store._extract_single_frame(invalid, tmp_path / "copy-failed") == []

    existing_frame = store._cosmos_frames_dir("session-existing") / "frame_0001.png"
    existing_frame.parent.mkdir(parents=True)
    existing_frame.write_bytes(b"png")
    assert store._ensure_cosmos_frames("session-existing", "siteworld-1") == [existing_frame]

    monkeypatch.setattr(store, "_find_prebuilt_cosmos_video", lambda _site_world_id: conditioning)
    monkeypatch.setattr(store, "_extract_single_frame", lambda _path, frames_dir: [frames_dir / "frame_0001.png"])
    assert store._ensure_cosmos_frames("session-prebuilt", "siteworld-1")
    assert store._cosmos_status_path("session-prebuilt").is_file()

    monkeypatch.setattr(store, "_find_prebuilt_cosmos_video", lambda _site_world_id: prebuilt_mp4)
    monkeypatch.setattr(store, "_extract_frames_from_video", lambda _path, frames_dir: [frames_dir / "frame_0001.png"])
    assert store._ensure_cosmos_frames("session-prebuilt-mp4", "siteworld-1")

    monkeypatch.setattr(store, "_find_prebuilt_cosmos_video", lambda _site_world_id: None)
    monkeypatch.setattr(store, "_find_conditioning_frame", lambda _site_world_id: conditioning)
    assert store._ensure_cosmos_frames("session-conditioning", "siteworld-1")

    monkeypatch.setattr(store, "_find_conditioning_frame", lambda _site_world_id: tmp_path / "future-conditioning.jpg")
    monkeypatch.setattr(nrb, "_find_cosmos_repo", lambda: (tmp_path / "repo", tmp_path / "repo" / "python"))
    original_run_cosmos_inference_sync = store._run_cosmos_inference_sync
    monkeypatch.setattr(store, "_run_cosmos_inference_sync", lambda **_kwargs: [tmp_path / "generated.png"])
    assert store._ensure_cosmos_frames("session-infer", "siteworld-1") == [tmp_path / "generated.png"]
    monkeypatch.setattr(store, "_run_cosmos_inference_sync", original_run_cosmos_inference_sync)

    recheck_frame = store._cosmos_frames_dir("session-recheck") / "frame_0001.png"

    class CreateFramesOnEnter:
        def __enter__(self):
            recheck_frame.parent.mkdir(parents=True, exist_ok=True)
            recheck_frame.write_bytes(b"frame")
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(nrb, "_cosmos_session_lock", lambda _session_id: CreateFramesOnEnter())
    assert store._ensure_cosmos_frames("session-recheck", "siteworld-1") == [recheck_frame]

    adapter = pipeline_base / "cosmos_training_export" / "checkpoints" / "adapter_model.safetensors"
    adapter.parent.mkdir(parents=True)
    adapter.write_bytes(b"adapter")
    monkeypatch.delenv("COSMOS_LORA_CHECKPOINT_PATH", raising=False)
    assert store._find_lora_adapter("siteworld-1") == adapter
    explicit_adapter = tmp_path / "explicit.safetensors"
    explicit_adapter.write_bytes(b"adapter")
    monkeypatch.setenv("COSMOS_LORA_CHECKPOINT_PATH", str(explicit_adapter))
    assert store._find_lora_adapter("siteworld-1") == explicit_adapter
    monkeypatch.setenv("COSMOS_LORA_CHECKPOINT_PATH", str(tmp_path / "missing.safetensors"))
    assert store._find_lora_adapter("siteworld-1") is None
    monkeypatch.delenv("COSMOS_LORA_CHECKPOINT_PATH", raising=False)
    assert store._find_lora_adapter("missing-site") is None

    def raise_adapter_is_file(self: Path) -> bool:
        if str(self).endswith("adapter_model.safetensors"):
            raise OSError("stat failed")
        return original_is_file(self)

    monkeypatch.setattr(nrb.Path, "is_file", raise_adapter_is_file)
    assert store._find_lora_adapter("siteworld-1") is None
    monkeypatch.setattr(nrb.Path, "is_file", original_is_file)

    repo = tmp_path / "repo"
    python_bin = repo / ".venv" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    cond_frame = tmp_path / "cond.png"
    Image.new("RGB", (4, 4)).save(cond_frame)

    monkeypatch.setattr(nrb.subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("no python")))
    assert store._run_cosmos_inference_sync("session-run-oserror", (repo, python_bin), cond_frame, tmp_path / "frames-oserror") == []

    monkeypatch.setattr(nrb.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=1))
    assert store._run_cosmos_inference_sync("session-run-fail", (repo, python_bin), cond_frame, tmp_path / "frames-fail") == []

    def success_run(args, **_kwargs):
        out_dir = Path(args[args.index("-o") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        sample_name = next(out_dir.glob("cosmos_*.json")).stem
        (out_dir / f"{sample_name}.mp4").write_bytes(b"video")
        return SimpleNamespace(returncode=0)

    def fake_extract(video_path: Path, frames_dir: Path) -> list[Path]:
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame = frames_dir / "frame_0001.png"
        frame.write_bytes(b"frame")
        return [frame]

    monkeypatch.setattr(nrb.subprocess, "run", success_run)
    monkeypatch.setattr(store, "_convert_to_fmp4", lambda input_path, output_path: bool(output_path.write_bytes(b"fmp4")))
    monkeypatch.setattr(store, "_extract_frames_from_video", fake_extract)
    frames = store._run_cosmos_inference_sync("session-run-ok", (repo, python_bin), cond_frame, tmp_path / "frames-ok", lora_adapter=explicit_adapter)
    assert frames and store._cosmos_status_path("session-run-ok").is_file()

    store._store_session_state("render-session", {"session_id": "render-session", "site_world_id": "siteworld-1", "step_count": 0, "runtime_backend_selected": "native"})
    assert store._render_png("render-session", "head_rgb", refine_mode="preview").startswith(b"\x89PNG")

    explorer_live = tmp_path / "explorer-live.png"
    explorer_live.write_bytes(b"live")
    store._store_session_state(
        "explorer-live",
        {
            "session_id": "explorer-live",
            "site_world_id": "siteworld-1",
            "step_count": 0,
            "pose": {},
            "latest_render_path": str(explorer_live),
            "rollout": store._rollout_defaults(),
        },
    )
    live_result = store.explorer_render(
        "explorer-live",
        camera_id="head_rgb",
        pose={"x": 1},
        viewport_width=100,
        viewport_height=100,
        refine_mode="live",
    )
    assert Path(live_result["frame_path"]).read_bytes() == b"live"

    cosmos_frame = tmp_path / "explorer-cosmos.png"
    cosmos_frame.write_bytes(b"cosmos")
    store._store_session_state(
        "explorer-cosmos",
        {
            "session_id": "explorer-cosmos",
            "site_world_id": "siteworld-1",
            "step_count": 0,
            "pose": {},
            "rollout": store._rollout_defaults(),
        },
    )
    monkeypatch.setattr(store, "_ensure_cosmos_frames", lambda *_args: [cosmos_frame])
    cosmos_result = store.explorer_render(
        "explorer-cosmos",
        camera_id="head_rgb",
        pose={"x": 2},
        viewport_width=100,
        viewport_height=100,
        refine_mode=None,
    )
    assert Path(cosmos_result["frame_path"]).read_bytes() == b"cosmos"

    monkeypatch.setenv("SITE_WORLD_RUNTIME_SERVICE_HOST", "0.0.0.0")
    monkeypatch.setenv("SITE_WORLD_RUNTIME_SERVICE_PORT", "9999")
    monkeypatch.setenv("SITE_WORLD_RUNTIME_PUBLIC_BASE_URL", "https://runtime.example")
    monkeypatch.delenv("SITE_WORLD_RUNTIME_PUBLIC_WS_BASE_URL", raising=False)
    monkeypatch.setenv("SITE_WORLD_NATIVE_RUNTIME_ROOT", str(tmp_path / "runtime-root"))
    config = nrb.native_runtime_config_from_env()
    assert config.ws_base_url == "wss://runtime.example"
    assert config.root_dir == tmp_path / "runtime-root"


def test_runtime_info_reports_truthful_selected_render_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    for name in (
        "NATIVE_WORLD_MODEL_OUTPUT_PROFILE",
        "NATIVE_WORLD_MODEL_ENABLE_TRUTHFUL_PREVIEW",
        "NATIVE_WORLD_MODEL_SYNTHESIS_MODE",
        "BLUEPRINT_NATIVE_RUNTIME_BACKEND",
        "NATIVE_WORLD_MODEL_ENABLE_COSMOS_REFINEMENT",
        "NATIVE_WORLD_MODEL_PRODUCTION_GRADE",
    ):
        monkeypatch.delenv(name, raising=False)

    def _readiness(*, ready: bool) -> dict:
        return {
            "ready": ready,
            "package_ready": ready,
            "model_ready": ready,
            "checkpoint_ready": ready,
            "packages": {"torch": ready},
            "model_dir": "",
            "checkpoint_path": "",
            "cosmos_repo": "",
            "notes": [] if ready else ["native_model_not_provisioned"],
        }

    # Default truthful-preview profile without a configured model must report
    # the splat_only fallback, never a hard-coded cosmos identity.
    monkeypatch.setattr(nrb, "_runtime_readiness", lambda: _readiness(ready=False))
    info = store.runtime_info(service_version="test")
    assert info["model_identity"]["model_family"] == "site_splat_truthful_preview"
    assert info["model_identity"]["selected_runtime_path"] == "splat_only"
    assert info["state_guarantees"]["render_source"] == "truthful_preview_splat"
    assert info["state_guarantees"]["async_cosmos_refinement_enabled"] is False
    assert info["readiness"]["selected_runtime_path"] == "splat_only"
    assert info["engine_identity"]["selected_runtime_path"] == "splat_only"

    # A ready cosmos runtime with truthful preview disabled selects cosmos_i2w.
    monkeypatch.setenv("NATIVE_WORLD_MODEL_ENABLE_TRUTHFUL_PREVIEW", "0")
    monkeypatch.setenv("BLUEPRINT_NATIVE_RUNTIME_BACKEND", "cosmos_wam")
    monkeypatch.setattr(nrb, "_runtime_readiness", lambda: _readiness(ready=True))
    info = store.runtime_info(service_version="test")
    assert info["model_identity"]["model_family"] == "cosmos_i2w_native"
    assert info["state_guarantees"]["render_source"] == "cosmos_i2w"
    assert info["readiness"]["selected_runtime_path"] == "cosmos_i2w"

    # Explicitly requesting cosmos_i2w without a ready runtime is reported as
    # unconfigured instead of pretending a cosmos model is present.
    monkeypatch.setenv("NATIVE_WORLD_MODEL_SYNTHESIS_MODE", "cosmos_i2w")
    monkeypatch.setattr(nrb, "_runtime_readiness", lambda: _readiness(ready=False))
    info = store.runtime_info(service_version="test")
    assert info["model_identity"]["model_family"] == "unconfigured"
    assert info["state_guarantees"]["render_source"] == "unconfigured"
    assert info["readiness"]["selected_runtime_path"] == "unconfigured"


def test_neutral_runtime_does_not_implicitly_enable_cosmos_refinement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    monkeypatch.setattr(
        nrb,
        "_runtime_readiness",
        lambda: {"ready": True, "cosmos_ready": True},
    )
    monkeypatch.setenv("BLUEPRINT_NATIVE_RUNTIME_BACKEND", "site_splat")
    monkeypatch.delenv("NATIVE_WORLD_MODEL_SYNTHESIS_MODE", raising=False)
    monkeypatch.delenv("NATIVE_WORLD_MODEL_ENABLE_COSMOS_REFINEMENT", raising=False)

    assert store._uses_truthful_preview() is True
    assert store._cosmos_refinement_enabled() is False

    monkeypatch.setenv("NATIVE_WORLD_MODEL_ENABLE_COSMOS_REFINEMENT", "true")
    assert store._cosmos_refinement_enabled() is True

    monkeypatch.delenv("NATIVE_WORLD_MODEL_ENABLE_COSMOS_REFINEMENT", raising=False)
    monkeypatch.setenv("BLUEPRINT_NATIVE_RUNTIME_BACKEND", "cosmos_wam")
    assert store._cosmos_refinement_enabled() is True
