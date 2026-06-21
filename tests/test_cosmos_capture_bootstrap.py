from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from blueprint_pipeline.synthesis import cosmos_capture_bootstrap as bootstrap


def _write_bootstrap_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    video = tmp_path / "video.mp4"
    poses = tmp_path / "poses.jsonl"
    intrinsics = tmp_path / "intrinsics.json"
    video.write_bytes(b"video")
    poses.write_text(
        "".join(
            json.dumps(
                {
                    "t_capture_sec": float(index),
                    "T_world_camera": [
                        [1, 0, 0, index],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ],
                }
            )
            + "\n"
            for index in range(4)
        ),
        encoding="utf-8",
    )
    intrinsics.write_text(json.dumps({"fx": 1, "fy": 1, "cx": 1, "cy": 1}), encoding="utf-8")
    return video, poses, intrinsics


def test_cosmos_capture_bootstrap_helpers_and_sources(monkeypatch, tmp_path: Path) -> None:
    existing = tmp_path / "local.mov"
    existing.write_bytes(b"video")
    assert bootstrap._optional_existing_path(existing) == existing.resolve()
    assert bootstrap._optional_existing_path("gs://bucket/video.mov") is None
    assert bootstrap._optional_existing_path(tmp_path / "missing.mov") is None
    assert bootstrap._existing_path_from_candidates(None, existing) == existing.resolve()
    assert bootstrap._existing_path_from_candidates(None) is None

    storage = tmp_path / "storage"
    gs_path = storage / "bucket" / "video.mov"
    gs_path.parent.mkdir(parents=True)
    gs_path.write_bytes(b"video")
    context = SimpleNamespace(storage_root=storage, raw_root=tmp_path / "raw")
    assert bootstrap._resolved_gs_path(context, "not-gs") is None
    assert bootstrap._resolved_gs_path(context, "gs://bucket/video.mov") == gs_path
    assert bootstrap._resolved_gs_path(context, "gs://bucket/missing.mov") is None

    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text('\n{"a": 1}\n[]\n{"transform": [[1]]}\n', encoding="utf-8")
    assert bootstrap._read_jsonl(tmp_path / "missing.jsonl") == []
    assert bootstrap._read_jsonl(jsonl) == [{"a": 1}, {"transform": [[1]]}]
    assert bootstrap._read_pose_rows(jsonl) == [{"transform": [[1]]}]

    def fake_ffprobe(command, **_kwargs):
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"streams": [{"nb_frames": "7"}]}))

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_ffprobe)
    assert bootstrap._ffprobe_total_frames(existing) == 7
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, stdout=json.dumps({"streams": []})))
    assert bootstrap._ffprobe_total_frames(existing) is None
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, stdout=json.dumps({"streams": [{"nb_frames": "N/A"}]})))
    assert bootstrap._ffprobe_total_frames(existing) is None
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("ffprobe bad")))
    assert bootstrap._ffprobe_total_frames(existing) is None

    frame_path = tmp_path / "frame.jpg"

    def fake_ffmpeg(command, **_kwargs):
        Path(command[-1]).write_bytes(b"jpg")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_ffmpeg)
    assert bootstrap._extract_frame_ffmpeg(video_path=existing, frame_index=-1, frame_path=frame_path) is True
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("ffmpeg bad")))
    assert bootstrap._extract_frame_ffmpeg(video_path=existing, frame_index=0, frame_path=tmp_path / "missing-frame.jpg") is False

    assert bootstrap.resolve_video_bootstrap_sources(context=context, conditioning_bundle={}) == {}
    raw = context.raw_root
    (raw / "arkit").mkdir(parents=True)
    (raw / "walkthrough.mp4").write_bytes(b"video")
    (raw / "arkit" / "poses.jsonl").write_text("{}", encoding="utf-8")
    (raw / "arkit" / "intrinsics.json").write_text("{}", encoding="utf-8")
    raw_sources = bootstrap.resolve_video_bootstrap_sources(context=context, conditioning_bundle={})
    assert raw_sources["origin"] == "raw_capture_assets"

    local_sources = bootstrap.resolve_video_bootstrap_sources(
        context=context,
        conditioning_bundle={
            "raw_video_uri": "gs://bucket/video.mov",
            "arkit": {"poses_uri": "gs://bucket/poses.jsonl", "intrinsics_uri": "gs://bucket/intrinsics.json"},
            "local_paths": {
                "raw_video_path": str(existing),
                "arkit_poses_path": str(raw / "arkit" / "poses.jsonl"),
                "arkit_intrinsics_path": str(raw / "arkit" / "intrinsics.json"),
            },
        },
    )
    assert local_sources["origin"] == "conditioning_bundle"


def test_extract_video_bootstrap_records_cv2_and_ffmpeg_edges(monkeypatch, tmp_path: Path) -> None:
    video, poses, intrinsics = _write_bootstrap_inputs(tmp_path)
    sources = {"origin": "test", "video_path": str(video), "poses_path": str(poses), "intrinsics_path": str(intrinsics), "source_video_uri": "source"}
    assert bootstrap.extract_video_bootstrap_records(
        bootstrap_sources={"video_path": str(tmp_path / "missing.mp4"), "poses_path": str(poses), "intrinsics_path": str(intrinsics)},
        export_root=tmp_path / "missing",
        max_frames=2,
    ) == []
    no_pose = tmp_path / "no_pose.jsonl"
    no_pose.write_text('{"x": 1}\n', encoding="utf-8")
    assert bootstrap.extract_video_bootstrap_records(
        bootstrap_sources={"video_path": str(video), "poses_path": str(no_pose), "intrinsics_path": str(intrinsics)},
        export_root=tmp_path / "no-pose",
        max_frames=2,
    ) == []

    release_calls: list[str] = []

    class FakeCapture:
        def __init__(self, _path: str) -> None:
            self.path = _path

        def isOpened(self) -> bool:
            return True

        def get(self, _prop) -> int:
            return 4

        def set(self, _prop, _value) -> None:
            return None

        def read(self):
            return True, object()

        def release(self) -> None:
            release_calls.append("released")

    cv2_mod = ModuleType("cv2")
    cv2_mod.CAP_PROP_FRAME_COUNT = 1
    cv2_mod.CAP_PROP_POS_FRAMES = 2
    cv2_mod.VideoCapture = FakeCapture

    def write_frame(path: str, _frame) -> bool:
        Path(path).write_bytes(b"jpg")
        return True

    cv2_mod.imwrite = write_frame
    monkeypatch.setitem(sys.modules, "cv2", cv2_mod)
    records = bootstrap.extract_video_bootstrap_records(
        bootstrap_sources=sources,
        export_root=tmp_path / "cv2-success",
        max_frames=3,
    )
    assert len(records) == 3
    assert records[0]["source_mode"] == "video_bootstrap"
    assert release_calls

    assert bootstrap.extract_video_bootstrap_records(
        bootstrap_sources=sources,
        export_root=tmp_path / "one-frame",
        max_frames=1,
    ) == []

    class ClosedCapture(FakeCapture):
        def isOpened(self) -> bool:
            return False

    cv2_mod.VideoCapture = ClosedCapture
    def extract_frame(**kwargs) -> bool:
        Path(kwargs["frame_path"]).write_bytes(b"jpg")
        return True

    monkeypatch.setattr(bootstrap, "_extract_frame_ffmpeg", extract_frame)
    fallback_records = bootstrap.extract_video_bootstrap_records(
        bootstrap_sources=sources,
        export_root=tmp_path / "ffmpeg-success",
        max_frames=2,
    )
    assert len(fallback_records) == 2

    class FailingReadCapture(FakeCapture):
        def read(self):
            return False, None

    cv2_mod.VideoCapture = FailingReadCapture
    assert bootstrap.extract_video_bootstrap_records(
        bootstrap_sources=sources,
        export_root=tmp_path / "read-fail",
        max_frames=2,
    ) == []

    cv2_mod.VideoCapture = FakeCapture
    cv2_mod.imwrite = lambda _path, _frame: False
    assert bootstrap.extract_video_bootstrap_records(
        bootstrap_sources=sources,
        export_root=tmp_path / "write-fail",
        max_frames=2,
    ) == []

    monkeypatch.setitem(sys.modules, "cv2", None)
    monkeypatch.setattr(bootstrap, "_extract_frame_ffmpeg", lambda **_kwargs: False)
    assert bootstrap.extract_video_bootstrap_records(
        bootstrap_sources=sources,
        export_root=tmp_path / "ffmpeg-fail",
        max_frames=2,
    ) == []
