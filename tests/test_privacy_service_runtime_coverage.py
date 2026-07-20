from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from blueprint_pipeline import privacy_service_runtime as psr


def _write(path: Path, payload: bytes = b"data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


class _FakeBlob:
    def __init__(self, name: str, payload: bytes = b"blob") -> None:
        self.name = name
        self.payload = payload
        self.uploaded_from: str | None = None

    def download_to_filename(self, filename: str) -> None:
        _write(Path(filename), self.payload)

    def upload_from_filename(self, filename: str) -> None:
        self.uploaded_from = filename


class _FakeBucket:
    def __init__(self, blobs: list[_FakeBlob] | None = None) -> None:
        self._blobs = blobs or []
        self.uploaded: dict[str, _FakeBlob] = {}

    def list_blobs(self, prefix: str):
        return [blob for blob in self._blobs if blob.name.startswith(prefix)]

    def blob(self, key: str) -> _FakeBlob:
        blob = _FakeBlob(key)
        self.uploaded[key] = blob
        return blob


class _FakeStorageClient:
    def __init__(self, bucket: _FakeBucket) -> None:
        self._bucket = bucket

    def bucket(self, _name: str) -> _FakeBucket:
        return self._bucket


class _FakeCapture:
    def __init__(self, frames: list[np.ndarray], *, opened: bool = True, fps: float = 10.0) -> None:
        self.frames = list(frames)
        self.opened = opened
        self.fps = fps
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def get(self, prop):
        if prop == 5:
            return self.fps
        if prop == 3:
            return 4
        if prop == 4:
            return 3
        return 0

    def release(self) -> None:
        self.released = True


class _FakeWriter:
    def __init__(self, path: str, opened: bool = True) -> None:
        self.path = Path(path)
        self.opened = opened
        self.frames: list[np.ndarray] = []

    def isOpened(self) -> bool:
        return self.opened

    def write(self, frame) -> None:
        self.frames.append(frame)
        _write(self.path, b"video")

    def release(self) -> None:
        return None


def _install_fake_cv2(monkeypatch: pytest.MonkeyPatch, *, frames: list[np.ndarray] | None = None, opened: bool = True, writer_opened: bool = True, masks: dict[str, np.ndarray] | None = None):
    fake = ModuleType("cv2")
    fake.COLOR_BGR2RGB = 1
    fake.CAP_PROP_FPS = 5
    fake.CAP_PROP_FRAME_WIDTH = 3
    fake.CAP_PROP_FRAME_HEIGHT = 4
    fake.IMREAD_GRAYSCALE = 0
    fake.IMREAD_UNCHANGED = -1
    fake.INPAINT_TELEA = 1
    capture_frames = frames if frames is not None else [np.zeros((3, 4, 3), dtype=np.uint8)]
    fake.VideoCapture = lambda _path: _FakeCapture(capture_frames, opened=opened)
    fake.VideoWriter = lambda path, *_args: _FakeWriter(path, opened=writer_opened)
    fake.VideoWriter_fourcc = lambda *_args: 0
    fake.cvtColor = lambda frame, _code: frame
    fake.imwrite = lambda path, image: (_write(Path(path), np.asarray(image).tobytes() or b"mask"), True)[1]
    fake.imread = lambda path, _flag=None: (masks or {}).get(Path(path).name, np.ones((3, 4), dtype=np.uint8))
    fake.dilate = lambda mask, _kernel, iterations=1: mask
    fake.inpaint = lambda frame, _mask, _radius, _method: np.full_like(frame, 7)
    monkeypatch.setitem(sys.modules, "cv2", fake)
    return fake


@pytest.fixture(autouse=True)
def _reset_model_runtime_cache():
    # SCALE2-06: loaded model runtimes are cached in-process for warm
    # instances; tests that swap fake backends need a clean cache.
    psr._MODEL_RUNTIME_CACHE.clear()
    yield
    psr._MODEL_RUNTIME_CACHE.clear()


def _install_fake_sam3(monkeypatch: pytest.MonkeyPatch, *, output: object, model_error: bool = False):
    # Each installed fake is a new model world; drop any runtime cached from
    # a previous fake so the scenario under test actually loads this one.
    psr._MODEL_RUNTIME_CACHE.clear()
    sam3_pkg = ModuleType("sam3")
    sam3_model_pkg = ModuleType("sam3.model")
    processor_mod = ModuleType("sam3.model.sam3_image_processor")
    builder_mod = ModuleType("sam3.model_builder")

    class Processor:
        def __init__(self, _model) -> None:
            return None

        def set_image(self, image):
            return {"image": image}

        def set_text_prompt(self, *, state, prompt):
            if isinstance(output, Exception):
                raise output
            return output

    def build_model(**_kwargs):
        if model_error:
            raise RuntimeError("model failed")
        return object()

    processor_mod.Sam3Processor = Processor
    builder_mod.build_sam3_image_model = build_model
    monkeypatch.setitem(sys.modules, "sam3", sam3_pkg)
    monkeypatch.setitem(sys.modules, "sam3.model", sam3_model_pkg)
    monkeypatch.setitem(sys.modules, "sam3.model.sam3_image_processor", processor_mod)
    monkeypatch.setitem(sys.modules, "sam3.model_builder", builder_mod)


def _install_fake_geometry_da3(monkeypatch: pytest.MonkeyPatch, *, depth: np.ndarray | None = None):
    module = ModuleType("blueprint_pipeline.geometry_da3")
    module._infer_depth_with_runtime = lambda _runtime, _rgb: depth
    module._normalized_confidence = lambda depth_array: np.ones_like(depth_array, dtype=np.float32)
    module._load_da3_runtime = lambda _name: (SimpleNamespace(), ["loaded"])
    monkeypatch.setitem(sys.modules, "blueprint_pipeline.geometry_da3", module)
    return module


def test_privacy_runtime_materialization_copy_and_subprocess_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GCS_ROOT", str(tmp_path / "gcs"))
    assert psr._gcs_root() == tmp_path / "gcs"
    assert psr._json_path(tmp_path / "json", "out.json") == tmp_path / "json" / "out.json"
    assert psr._string(" x ") == "x"
    assert psr._to_jsonable({"a": "p"}) == {"a": "p"}

    storage_module = ModuleType("google.cloud.storage")
    storage_module.Client = lambda: "client"
    google_module = ModuleType("google")
    cloud_module = ModuleType("google.cloud")
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud_module)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", storage_module)
    assert psr._storage_client() == "client"

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"http"

    monkeypatch.setattr(psr.urllib_request, "urlopen", lambda _req, timeout: FakeResponse())
    assert psr._download_http_to_path("https://example.com/model", tmp_path / "download.bin").read_bytes() == b"http"

    local = _write(tmp_path / "local.mov", b"video")
    assert psr._materialize_input_file(uri="", local_hint=str(local), working_path=tmp_path / "work.mov") == local
    with pytest.raises(FileNotFoundError, match="missing_input_video"):
        psr._materialize_input_file(uri="", local_hint="", working_path=tmp_path / "work.mov")
    monkeypatch.setattr(psr, "ensure_local_uri_path", lambda uri, **_kwargs: _write(tmp_path / "gcs.mov", b"gcs"))
    assert psr._materialize_input_file(uri="gs://bucket/video.mov", local_hint="", working_path=tmp_path / "work.mov").name == "gcs.mov"
    assert psr._materialize_input_file(uri="https://example.com/video.mov", local_hint="", working_path=tmp_path / "http.mov").read_bytes() == b"http"
    assert psr._materialize_input_file(uri=str(local), local_hint="", working_path=tmp_path / "work.mov") == local
    with pytest.raises(FileNotFoundError, match="input_file_not_found"):
        psr._materialize_input_file(uri=str(tmp_path / "missing.mov"), local_hint="", working_path=tmp_path / "work.mov")

    prefix = tmp_path / "prefix"
    _write(prefix / "a.txt")
    assert psr._materialize_prefix(uri="", local_hint=str(prefix), working_dir=tmp_path / "unused") == prefix
    with pytest.raises(FileNotFoundError, match="missing_input_prefix"):
        psr._materialize_prefix(uri="", local_hint="", working_dir=tmp_path / "prefix-work")
    with pytest.raises(FileNotFoundError, match="input_prefix_not_found"):
        psr._materialize_prefix(uri=str(tmp_path / "missing-prefix"), local_hint="", working_dir=tmp_path / "prefix-work")

    mounted_prefix = tmp_path / "mounted-prefix"
    mounted_prefix.mkdir()
    monkeypatch.setattr(psr, "resolve_gs_uri_to_path", lambda uri, root: mounted_prefix)
    assert psr._materialize_prefix(uri="gs://bucket/prefix", local_hint="", working_dir=tmp_path / "prefix-work") == mounted_prefix
    download_bucket = _FakeBucket([_FakeBlob("remote/prefix/a.txt", b"a"), _FakeBlob("remote/prefix/")])
    monkeypatch.setattr(psr, "resolve_gs_uri_to_path", lambda uri, root: tmp_path / "missing-mounted")
    monkeypatch.setattr(psr, "_storage_client", lambda: _FakeStorageClient(download_bucket))
    downloaded = psr._materialize_prefix(uri="gs://bucket/remote/prefix", local_hint="", working_dir=tmp_path / "downloaded-prefix")
    assert (downloaded / "a.txt").read_bytes() == b"a"
    with pytest.raises(FileNotFoundError, match="prefix_not_found"):
        psr._materialize_prefix(uri="gs://bucket/empty/prefix", local_hint="", working_dir=tmp_path / "empty-prefix")

    model = _write(tmp_path / "model.pt")
    assert psr._materialize_model_path(str(model), working_dir=tmp_path, default_name="model.pt") == str(model)
    assert psr._materialize_model_path("", working_dir=tmp_path, default_name="model.pt") == ""
    assert psr._materialize_model_path("gs://bucket/model.pt", working_dir=tmp_path, default_name="model.pt").endswith("gcs.mov")
    assert psr._materialize_model_path("https://example.com/model.pt", working_dir=tmp_path, default_name="model-http.pt").endswith("model-http.pt")
    assert psr._materialize_model_path("hf:model", working_dir=tmp_path, default_name="model.pt") == "hf:model"

    copied_file = tmp_path / "copied" / "file.txt"
    psr._copy_file_to_uri(local, str(copied_file))
    assert copied_file.read_bytes() == b"video"
    psr._copy_file_to_uri(local, "")
    mounted_file = tmp_path / "mounted" / "file.txt"
    monkeypatch.setattr(psr, "resolve_gs_uri_to_path", lambda uri, root: mounted_file)
    psr._copy_file_to_uri(local, "gs://bucket/mounted/file.txt")
    assert mounted_file.read_bytes() == b"video"
    upload_bucket = _FakeBucket()
    monkeypatch.setattr(psr, "resolve_gs_uri_to_path", lambda uri, root: tmp_path / "no-mount-file" / "file.txt")
    monkeypatch.setattr(psr, "ensure_dir", lambda path: (_ for _ in ()).throw(RuntimeError("mount failed")) if "no-mount-file" in str(path) else path.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(psr, "_storage_client", lambda: _FakeStorageClient(upload_bucket))
    psr._copy_file_to_uri(local, "gs://bucket/upload/file.txt")
    assert upload_bucket.uploaded["upload/file.txt"].uploaded_from == str(local)

    source_dir = tmp_path / "source-dir"
    _write(source_dir / "nested" / "a.txt", b"a")
    assert psr._copy_directory_to_uri(source_dir, "") == []
    local_refs = psr._copy_directory_to_uri(source_dir, str(tmp_path / "dir-copy"))
    assert len(local_refs) == 1
    mounted_dir = tmp_path / "mounted-dir"
    monkeypatch.setattr(psr, "resolve_gs_uri_to_path", lambda uri, root: mounted_dir)
    assert psr._copy_directory_to_uri(source_dir, "gs://bucket/mounted-dir") == ["gs://bucket/mounted-dir/nested/a.txt"]
    upload_dir_bucket = _FakeBucket()
    monkeypatch.setattr(psr, "resolve_gs_uri_to_path", lambda uri, root: tmp_path / "not-created" / "child")
    monkeypatch.setattr(psr, "ensure_dir", lambda path: (_ for _ in ()).throw(RuntimeError("no mount")) if "not-created" in str(path) else path.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(psr, "_storage_client", lambda: _FakeStorageClient(upload_dir_bucket))
    assert psr._copy_directory_to_uri(source_dir, "gs://bucket/upload-dir") == ["gs://bucket/upload-dir/nested/a.txt"]

    monkeypatch.setattr(psr, "ensure_dir", lambda path: path.mkdir(parents=True, exist_ok=True))
    payload_path = tmp_path / "payload" / "out.json"
    psr._write_payload_json({"ok": True}, payload_path, "")
    assert json.loads(payload_path.read_text(encoding="utf-8")) == {"ok": True}
    assert psr._read_json_object(payload_path) == {"ok": True}
    assert psr._read_json_object(tmp_path / "missing.json") == {}
    assert psr._join_output_reference("gs://bucket/prefix/", "/a.txt") == "gs://bucket/prefix/a.txt"
    assert psr._join_output_reference("", "a") == ""
    assert psr._frames_with_suffix(source_dir / "nested", {".txt"}) == [source_dir / "nested" / "a.txt"]
    assert psr._frames_with_suffix(None, {".txt"}) == []

    monkeypatch.setattr(psr.shutil, "which", lambda name: None)
    assert psr._ffprobe_duration(local) == 0.0
    monkeypatch.setattr(psr.shutil, "which", lambda name: "/bin/ffprobe")
    monkeypatch.setattr(psr.subprocess, "run", lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, stdout="", stderr="bad"))
    assert psr._ffprobe_duration(local) == 0.0
    monkeypatch.setattr(psr.subprocess, "run", lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, stdout="bad", stderr=""))
    assert psr._ffprobe_duration(local) == 0.0
    monkeypatch.setattr(psr.subprocess, "run", lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, stdout="1.25", stderr=""))
    assert psr._ffprobe_duration(local) == 1.25

    video_only = _write(tmp_path / "video-only.mp4", b"video-only")
    output = tmp_path / "merged.mp4"
    monkeypatch.setattr(psr.shutil, "which", lambda name: None)
    psr._merge_audio_if_possible(source_video=local, video_only_path=video_only, output_video=output)
    assert output.read_bytes() == b"video-only"
    output.unlink()
    monkeypatch.setattr(psr.shutil, "which", lambda name: "/bin/ffmpeg")
    monkeypatch.setattr(psr.subprocess, "run", lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, stdout="", stderr="bad"))
    psr._merge_audio_if_possible(source_video=local, video_only_path=video_only, output_video=output)
    assert output.read_bytes() == b"video-only"


def test_privacy_runtime_sam3_and_depth_backends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = _write(tmp_path / "video.mov", b"video")
    result = psr._run_sam3_backend(
        input_video=video,
        masks_dir=tmp_path / "masks-unavailable",
        prompt="person",
        stage_name="stage",
        weights_path="",
    )
    assert result["reason"].startswith("sam3_runtime_unavailable")

    _install_fake_cv2(monkeypatch, frames=[np.zeros((3, 4, 3), dtype=np.uint8), np.zeros((3, 4, 3), dtype=np.uint8)])
    _install_fake_sam3(monkeypatch, output={"masks": np.ones((1, 3, 4), dtype=np.float32), "scores": np.array([0.9])})
    monkeypatch.setenv("PRIVACY_SAM3_FRAME_STRIDE", "2")
    monkeypatch.setenv("PRIVACY_SAM3_SCORE_THRESHOLD", "0.5")
    masks_dir = tmp_path / "masks"
    _write(masks_dir / "old.png", b"old")
    succeeded = psr._run_sam3_backend(
        input_video=video,
        masks_dir=masks_dir,
        prompt="",
        stage_name="stage",
        weights_path="weights.pt",
    )
    assert succeeded["status"] == "succeeded"
    assert succeeded["frames_scanned"] == 1
    assert not (masks_dir / "old.png").exists()
    assert Path(succeeded["mask_paths"][0]).is_file()

    _install_fake_sam3(monkeypatch, output={}, model_error=True)
    assert psr._run_sam3_backend(input_video=video, masks_dir=tmp_path / "model-fail", prompt="person", stage_name="", weights_path="")["reason"].startswith("sam3_model_load_failed")
    _install_fake_cv2(monkeypatch, opened=False)
    _install_fake_sam3(monkeypatch, output={})
    assert psr._run_sam3_backend(input_video=video, masks_dir=tmp_path / "video-fail", prompt="person", stage_name="", weights_path="")["reason"].startswith("video_open_failed")
    _install_fake_cv2(monkeypatch)
    _install_fake_sam3(monkeypatch, output=RuntimeError("infer failed"))
    assert psr._run_sam3_backend(input_video=video, masks_dir=tmp_path / "infer-fail", prompt="person", stage_name="", weights_path="")["reason"].startswith("sam3_inference_failed")
    _install_fake_sam3(monkeypatch, output={"masks": np.zeros((0,), dtype=np.float32), "scores": None})
    empty_masks = psr._run_sam3_backend(input_video=video, masks_dir=tmp_path / "empty-masks", prompt="person", stage_name="", weights_path="")
    assert empty_masks["people_detected"] is False

    result = psr._run_depth_anything_backend(
        input_video=video,
        depth_dir=tmp_path / "depth-unavailable",
        confidence_dir=tmp_path / "confidence-unavailable",
        depth_anything_model_path="",
    )
    assert result["reason"].startswith("depth_anything_runtime_unavailable")

    _install_fake_cv2(monkeypatch, frames=[np.zeros((3, 4, 3), dtype=np.uint8)])
    _install_fake_geometry_da3(monkeypatch, depth=np.ones((3, 4), dtype=np.float32))
    monkeypatch.setattr(psr, "_load_depth_anything_runtime", lambda _path: (SimpleNamespace(), ["warn"]))
    depth_result = psr._run_depth_anything_backend(
        input_video=video,
        depth_dir=tmp_path / "depth",
        confidence_dir=tmp_path / "confidence",
        depth_anything_model_path="model",
    )
    assert depth_result["status"] == "succeeded"
    assert depth_result["frame_count"] == 1
    assert Path(depth_result["depth_artifacts"][0]["path"]).is_file()
    monkeypatch.setattr(psr, "_load_depth_anything_runtime", lambda _path: (None, ["missing"]))
    assert psr._run_depth_anything_backend(input_video=video, depth_dir=tmp_path / "depth-missing", confidence_dir=tmp_path / "conf-missing", depth_anything_model_path="")["warnings"] == ["missing"]
    _install_fake_cv2(monkeypatch, opened=False)
    monkeypatch.setattr(psr, "_load_depth_anything_runtime", lambda _path: (SimpleNamespace(), []))
    assert psr._run_depth_anything_backend(input_video=video, depth_dir=tmp_path / "depth-video-fail", confidence_dir=tmp_path / "conf-video-fail", depth_anything_model_path="")["reason"].startswith("video_open_failed")
    _install_fake_cv2(monkeypatch, frames=[np.zeros((3, 4, 3), dtype=np.uint8)])
    _install_fake_geometry_da3(monkeypatch, depth=None)
    monkeypatch.setattr(psr, "_load_depth_anything_runtime", lambda _path: (SimpleNamespace(), []))
    assert psr._run_depth_anything_backend(input_video=video, depth_dir=tmp_path / "depth-infer-fail", confidence_dir=tmp_path / "conf-infer-fail", depth_anything_model_path="")["reason"] == "depth_anything_inference_failed"
    _install_fake_cv2(monkeypatch, frames=[])
    assert psr._run_depth_anything_backend(input_video=video, depth_dir=tmp_path / "depth-empty", confidence_dir=tmp_path / "conf-empty", depth_anything_model_path="")["reason"] == "video_empty"


def test_privacy_runtime_depth_maps_vip_and_deepprivacy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = _write(tmp_path / "video.mov", b"video")
    manifest = {
        "artifacts": [
            {"frame_index": 0, "path": str(_write(tmp_path / "depth.npy", np.ones((2, 2), dtype=np.float32).tobytes())), "relative_path": "depth.npy"},
            "bad",
            {"frame_index": 1, "path": str(tmp_path / "missing.npy"), "uri": "", "relative_path": "missing.npy"},
        ]
    }
    artifact_map = psr._materialize_artifact_map(manifest=manifest, working_dir=tmp_path / "artifact-work")
    assert 0 in artifact_map
    assert psr._materialize_artifact_map(manifest=None, working_dir=tmp_path) == {}

    npy = tmp_path / "frame.npy"
    np.save(npy, np.ones((2, 2), dtype=np.float32))
    assert psr._read_depth_frame(npy).shape == (2, 2)
    npy_video_shape = tmp_path / "frame_video_shape.npy"
    np.save(npy_video_shape, np.ones((3, 4), dtype=np.float32))
    _install_fake_cv2(monkeypatch, masks={"depth.png": np.ones((2, 2, 1), dtype=np.uint16)})
    assert psr._read_depth_frame(tmp_path / "depth.png").shape == (2, 2)
    _install_fake_cv2(monkeypatch, masks={"missing.png": None})
    with pytest.raises(FileNotFoundError):
        psr._read_depth_frame(tmp_path / "missing.png")

    frame = np.zeros((3, 4, 3), dtype=np.uint8)
    assert np.array_equal(psr._inpaint_with_depth(frame=frame, mask=np.zeros((3, 4), dtype=np.uint8), depth_map=None, confidence_map=None), frame)
    _install_fake_cv2(monkeypatch)
    mask = np.ones((3, 4), dtype=np.uint8)
    inpainted = psr._inpaint_with_depth(
        frame=frame,
        mask=mask,
        depth_map=np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.float32),
        confidence_map=np.zeros((3, 4), dtype=np.float32),
    )
    assert inpainted.sum() > 0

    monkeypatch.setitem(sys.modules, "cv2", None)
    assert psr._run_vip_backend(
        input_video=video,
        masks_dir=tmp_path / "masks",
        output_video=tmp_path / "out.mov",
        preferred_depth_source="depth_anything",
        arkit_depth_dir=None,
        arkit_confidence_dir=None,
        precomputed_depth_frames=None,
        precomputed_confidence_frames=None,
        vip_model_path="",
        depth_anything_model_path="",
    )["reason"].startswith("vip_runtime_unavailable")

    masks_dir = tmp_path / "masks"
    _write(masks_dir / "frame_000000.png", b"mask")
    arkit_depth = tmp_path / "arkit-depth"
    arkit_conf = tmp_path / "arkit-conf"
    arkit_depth.mkdir()
    arkit_conf.mkdir()
    np.save(arkit_depth / "depth_000000.npy", np.ones((3, 4), dtype=np.float32))
    np.save(arkit_conf / "confidence_000000.npy", np.ones((3, 4), dtype=np.float32))
    _install_fake_cv2(monkeypatch, frames=[np.zeros((3, 4, 3), dtype=np.uint8)])
    monkeypatch.setattr(psr, "_merge_audio_if_possible", lambda source_video, video_only_path, output_video: _write(output_video, b"merged"))
    vip_arkit = psr._run_vip_backend(
        input_video=video,
        masks_dir=masks_dir,
        output_video=tmp_path / "vip-arkit.mov",
        preferred_depth_source="arkit",
        arkit_depth_dir=arkit_depth,
        arkit_confidence_dir=arkit_conf,
        precomputed_depth_frames=None,
        precomputed_confidence_frames=None,
        vip_model_path="",
        depth_anything_model_path="",
    )
    assert vip_arkit["status"] == "succeeded"
    assert vip_arkit["depth_source"] == "arkit"

    _install_fake_cv2(monkeypatch, frames=[np.zeros((3, 4, 3), dtype=np.uint8)])
    precomputed = {0: npy_video_shape}
    vip_precomputed = psr._run_vip_backend(
        input_video=video,
        masks_dir=masks_dir,
        output_video=tmp_path / "vip-precomputed.mov",
        preferred_depth_source="depth_anything",
        arkit_depth_dir=None,
        arkit_confidence_dir=None,
        precomputed_depth_frames=precomputed,
        precomputed_confidence_frames={0: npy_video_shape},
        vip_model_path="",
        depth_anything_model_path="",
    )
    assert vip_precomputed["used_precomputed_depth"] is True
    _install_fake_cv2(monkeypatch, frames=[np.zeros((3, 4, 3), dtype=np.uint8)])
    assert psr._run_vip_backend(
        input_video=video,
        masks_dir=masks_dir,
        output_video=tmp_path / "vip-missing-precomputed.mov",
        preferred_depth_source="depth_anything",
        arkit_depth_dir=None,
        arkit_confidence_dir=None,
        precomputed_depth_frames={},
        precomputed_confidence_frames=None,
        vip_model_path="",
        depth_anything_model_path="",
    )["reason"] == "depth_anything_runtime_unavailable"
    monkeypatch.setattr(psr, "_load_depth_anything_runtime", lambda _path: (SimpleNamespace(_blueprint_infer_depth=lambda _runtime, _rgb: None), []))
    _install_fake_cv2(monkeypatch, frames=[np.zeros((3, 4, 3), dtype=np.uint8)])
    assert psr._run_vip_backend(
        input_video=video,
        masks_dir=masks_dir,
        output_video=tmp_path / "vip-depth-fail.mov",
        preferred_depth_source="depth_anything",
        arkit_depth_dir=None,
        arkit_confidence_dir=None,
        precomputed_depth_frames=None,
        precomputed_confidence_frames=None,
        vip_model_path="",
        depth_anything_model_path="",
    )["reason"] == "depth_anything_inference_failed"
    _install_fake_cv2(monkeypatch, opened=False)
    monkeypatch.setattr(psr, "_load_depth_anything_runtime", lambda _path: (SimpleNamespace(_blueprint_infer_depth=lambda _runtime, _rgb: np.ones((3, 4), dtype=np.float32)), []))
    assert psr._run_vip_backend(
        input_video=video,
        masks_dir=masks_dir,
        output_video=tmp_path / "vip-video-fail.mov",
        preferred_depth_source="depth_anything",
        arkit_depth_dir=None,
        arkit_confidence_dir=None,
        precomputed_depth_frames=None,
        precomputed_confidence_frames=None,
        vip_model_path="",
        depth_anything_model_path="",
    )["reason"].startswith("video_open_failed")
    _install_fake_cv2(monkeypatch, writer_opened=False)
    assert psr._run_vip_backend(
        input_video=video,
        masks_dir=masks_dir,
        output_video=tmp_path / "vip-writer-fail.mov",
        preferred_depth_source="depth_anything",
        arkit_depth_dir=None,
        arkit_confidence_dir=None,
        precomputed_depth_frames=None,
        precomputed_confidence_frames=None,
        vip_model_path="",
        depth_anything_model_path="",
    )["reason"].startswith("video_writer_failed")

    assert psr._deepprivacy2_repo_dir().as_posix().endswith("deepprivacy2")
    assert psr._run_deepprivacy2_backend(input_video=video, output_video=tmp_path / "dp2.mov", deepprivacy2_model_path="")["reason"].startswith("deepprivacy2_repo_missing")
    repo = tmp_path / "deepprivacy2"
    _write(repo / "anonymize.py", b"script")
    _write(repo / "configs" / "anonymizers" / "face.py", b"config")
    monkeypatch.setenv("DEEPPRIVACY2_REPO_DIR", str(repo))
    monkeypatch.setattr(psr.subprocess, "run", lambda command, **_kwargs: subprocess.CompletedProcess(command, 1, stdout="out", stderr="err"))
    failed = psr._run_deepprivacy2_backend(input_video=video, output_video=tmp_path / "dp2-failed.mov", deepprivacy2_model_path="model")
    assert failed["reason"] == "deepprivacy2_command_failed:1"

    def run_dp2(command, **_kwargs):
        _write(Path(command[command.index("-o") + 1]), b"dp2")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(psr.subprocess, "run", run_dp2)
    monkeypatch.setattr(psr, "_ffprobe_duration", lambda _path: 2.5)
    succeeded = psr._run_deepprivacy2_backend(input_video=video, output_video=tmp_path / "dp2-ok.mov", deepprivacy2_model_path="model")
    assert succeeded["status"] == "succeeded"
    assert succeeded["face_anonymized_segments"] == ["0.0-2.5"]


def test_privacy_runtime_dispatcher_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert psr.execute_privacy_service_request("unknown", {})["reason"] == "unsupported_runner_kind:unknown"
    assert psr.execute_privacy_service_request("sam3", {})["reason"] == "missing_input_video"
    video = _write(tmp_path / "input.mov", b"video")
    monkeypatch.setattr(psr, "_json_path", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("json path failed")))

    def fake_sam3(**kwargs):
        _write(kwargs["masks_dir"] / "frame_000000.png", b"mask")
        return {"status": "succeeded", "mask_paths": []}

    monkeypatch.setattr(psr, "_run_sam3_backend", fake_sam3)
    sam3 = psr.execute_privacy_service_request(
        "sam3",
        {
            "input_video_path": str(video),
            "masks_dir_path": str(tmp_path / "mask-output"),
            "output_json_path": str(tmp_path / "sam3.json"),
        },
    )
    assert sam3["status"] == "succeeded"
    assert sam3["mask_paths"]

    monkeypatch.setattr(psr, "_run_depth_anything_backend", lambda **_kwargs: {"status": "failed", "reason": "depth failed"})
    depth_failed = psr.execute_privacy_service_request(
        "vip",
        {"input_video_path": str(video), "depth_generation_only": True},
    )
    assert depth_failed["reason"] == "depth failed"
    monkeypatch.setattr(psr, "_materialize_prefix", lambda **_kwargs: (_ for _ in ()).throw(FileNotFoundError("missing masks")))
    vip_missing_masks = psr.execute_privacy_service_request("vip", {"input_video_path": str(video)})
    assert vip_missing_masks["reason"] == "missing masks"

    masks = tmp_path / "masks-dispatch"
    _write(masks / "frame_000000.png", b"mask")
    monkeypatch.setattr(psr, "_materialize_prefix", lambda **kwargs: masks)
    monkeypatch.setattr(psr, "_materialize_input_file", lambda **kwargs: video if "manifest" not in str(kwargs.get("working_path")) else (_ for _ in ()).throw(FileNotFoundError("manifest missing")))
    monkeypatch.setattr(psr, "_run_vip_backend", lambda **kwargs: (_write(kwargs["output_video"], b"vip"), {"status": "succeeded", "output_video": str(kwargs["output_video"])})[1])
    vip = psr.execute_privacy_service_request(
        "vip",
        {
            "input_video_path": str(video),
            "masks_dir_path": str(masks),
            "arkit_depth_prefix_uri": "gs://bucket/missing-depth",
            "arkit_confidence_prefix_uri": "gs://bucket/missing-confidence",
            "depth_manifest_uri": "gs://bucket/missing-depth-manifest",
            "confidence_manifest_uri": "gs://bucket/missing-confidence-manifest",
            "output_video_path": str(tmp_path / "vip-out.mov"),
        },
    )
    assert vip["status"] == "succeeded"
    monkeypatch.setattr(psr, "_run_deepprivacy2_backend", lambda **kwargs: (_write(kwargs["output_video"], b"dp2"), {"status": "succeeded", "output_video": str(kwargs["output_video"])})[1])
    dp2 = psr.execute_privacy_service_request(
        "deepprivacy2",
        {"input_video_path": str(video), "output_video_path": str(tmp_path / "dp2-out.mov")},
    )
    assert dp2["status"] == "succeeded"
    monkeypatch.setenv("PRIVACY_PIPELINE_ENABLED", "false")
    assert psr.privacy_service_enabled() is False


def test_privacy_runtime_remaining_branch_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_prefix = tmp_path / "local-prefix-uri"
    local_prefix.mkdir()
    assert psr._materialize_prefix(uri=str(local_prefix), local_hint="", working_dir=tmp_path / "unused-prefix") == local_prefix

    geometry_module = _install_fake_geometry_da3(monkeypatch, depth=np.ones((2, 2), dtype=np.float32))
    runtime = SimpleNamespace()
    geometry_module._load_da3_runtime = lambda _name: (runtime, ["ok"])
    loaded_runtime, warnings = psr._load_depth_anything_runtime("model-path")
    assert loaded_runtime is runtime
    assert loaded_runtime._blueprint_infer_depth is geometry_module._infer_depth_with_runtime
    assert warnings == ["ok"]
    assert psr.os.environ["DA3_MODEL_PATH"] == "model-path"

    video = _write(tmp_path / "video.mov", b"video")

    class FakeTensor:
        def __init__(self, value) -> None:
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def reshape(self, *_args):
            return self

        def tolist(self):
            return list(np.asarray(self.value).reshape(-1))

        def numpy(self):
            return np.asarray(self.value)

    _install_fake_cv2(monkeypatch, frames=[np.zeros((3, 4, 3), dtype=np.uint8)])
    _install_fake_sam3(monkeypatch, output={"masks": None})
    assert psr._run_sam3_backend(input_video=video, masks_dir=tmp_path / "sam-none", prompt="person", stage_name="", weights_path="")["frames_with_people"] == 0

    _install_fake_cv2(monkeypatch, frames=[np.zeros((3, 4, 3), dtype=np.uint8)])
    _install_fake_sam3(
        monkeypatch,
        output={
            "masks": FakeTensor(np.ones((1, 1, 3, 4), dtype=np.float32)),
            "scores": FakeTensor(np.array([0.9], dtype=np.float32)),
        },
    )
    tensor_result = psr._run_sam3_backend(input_video=video, masks_dir=tmp_path / "sam-tensor", prompt="person", stage_name="", weights_path="")
    assert tensor_result["frames_with_people"] == 1

    _install_fake_cv2(monkeypatch, frames=[np.zeros((3, 4, 3), dtype=np.uint8)])
    _install_fake_sam3(monkeypatch, output={"masks": np.ones((3, 4), dtype=np.float32), "scores": [0.9]})
    assert psr._run_sam3_backend(input_video=video, masks_dir=tmp_path / "sam-2d", prompt="person", stage_name="", weights_path="")["frames_with_people"] == 1

    _install_fake_cv2(monkeypatch, frames=[np.zeros((3, 4, 3), dtype=np.uint8)])
    _install_fake_sam3(monkeypatch, output={"masks": np.ones((4,), dtype=np.float32), "scores": [0.9]})
    assert psr._run_sam3_backend(input_video=video, masks_dir=tmp_path / "sam-1d", prompt="person", stage_name="", weights_path="")["frames_with_people"] == 0

    _install_fake_cv2(monkeypatch, frames=[np.zeros((3, 4, 3), dtype=np.uint8)])
    _install_fake_sam3(monkeypatch, output={"masks": np.ones((1, 3, 4), dtype=np.float32), "scores": [0.01]})
    assert psr._run_sam3_backend(input_video=video, masks_dir=tmp_path / "sam-low-score", prompt="person", stage_name="", weights_path="")["frames_with_people"] == 0

    monkeypatch.setitem(sys.modules, "cv2", None)
    assert psr._run_depth_anything_backend(input_video=video, depth_dir=tmp_path / "depth-import", confidence_dir=tmp_path / "conf-import", depth_anything_model_path="")["reason"].startswith("depth_anything_runtime_unavailable")

    local_hint = _write(tmp_path / "artifact-local.npy", b"local")
    monkeypatch.setattr(psr, "_materialize_input_file", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("no materialize")))
    fallback_map = psr._materialize_artifact_map(
        manifest={"artifacts": [{"frame_index": 3, "path": str(local_hint), "relative_path": "artifact-local.npy"}]},
        working_dir=tmp_path / "artifact-fallback",
    )
    assert fallback_map[3] == local_hint

    masks_dir = tmp_path / "vip-masks"
    _write(masks_dir / "frame_000001.png", b"mask-late")
    _install_fake_cv2(monkeypatch, frames=[np.zeros((3, 4, 3), dtype=np.uint8), np.zeros((3, 4, 3), dtype=np.uint8)])
    monkeypatch.setattr(psr, "_load_depth_anything_runtime", lambda _path: (SimpleNamespace(_blueprint_infer_depth=lambda _runtime, _rgb: np.ones((3, 4), dtype=np.float32)), []))
    monkeypatch.setattr(psr, "_merge_audio_if_possible", lambda source_video, video_only_path, output_video: _write(output_video, b"vip"))
    vip_missing_first_mask = psr._run_vip_backend(
        input_video=video,
        masks_dir=masks_dir,
        output_video=tmp_path / "vip-missing-first-mask.mov",
        preferred_depth_source="depth_anything",
        arkit_depth_dir=None,
        arkit_confidence_dir=None,
        precomputed_depth_frames=None,
        precomputed_confidence_frames=None,
        vip_model_path="",
        depth_anything_model_path="",
    )
    assert vip_missing_first_mask["frames_processed"] == 2

    empty_mask_dir = tmp_path / "vip-empty-mask"
    _write(empty_mask_dir / "frame_000000.png", b"empty")
    _install_fake_cv2(
        monkeypatch,
        frames=[np.zeros((3, 4, 3), dtype=np.uint8)],
        masks={"frame_000000.png": np.zeros((3, 4), dtype=np.uint8)},
    )
    assert psr._run_vip_backend(
        input_video=video,
        masks_dir=empty_mask_dir,
        output_video=tmp_path / "vip-empty-mask.mov",
        preferred_depth_source="depth_anything",
        arkit_depth_dir=None,
        arkit_confidence_dir=None,
        precomputed_depth_frames=None,
        precomputed_confidence_frames=None,
        vip_model_path="",
        depth_anything_model_path="",
    )["masks_used"] == 0

    missing_precomputed_masks = tmp_path / "vip-precomputed-missing-mask"
    _write(missing_precomputed_masks / "frame_000000.png", b"mask")
    _install_fake_cv2(monkeypatch, frames=[np.zeros((3, 4, 3), dtype=np.uint8)])
    missing_precomputed = psr._run_vip_backend(
        input_video=video,
        masks_dir=missing_precomputed_masks,
        output_video=tmp_path / "vip-missing-precomputed.mov",
        preferred_depth_source="depth_anything",
        arkit_depth_dir=None,
        arkit_confidence_dir=None,
        precomputed_depth_frames={1: tmp_path / "unused.npy"},
        precomputed_confidence_frames=None,
        vip_model_path="",
        depth_anything_model_path="",
    )
    assert missing_precomputed["reason"] == "precomputed_depth_missing:frame_000000"

    dispatcher_masks = tmp_path / "dispatcher-masks"
    _write(dispatcher_masks / "frame_000000.png", b"mask")
    monkeypatch.setattr(psr, "_materialize_input_file", lambda **kwargs: video)

    def materialize_prefix_for_dispatch(**kwargs):
        working_dir = str(kwargs.get("working_dir"))
        if "arkit_depth" in working_dir or "arkit_confidence" in working_dir:
            raise FileNotFoundError("missing arkit")
        return dispatcher_masks

    monkeypatch.setattr(psr, "_materialize_prefix", materialize_prefix_for_dispatch)
    monkeypatch.setattr(psr, "_run_vip_backend", lambda **kwargs: (_write(kwargs["output_video"], b"vip"), {"status": "succeeded", "output_video": str(kwargs["output_video"])})[1])
    dispatched = psr.execute_privacy_service_request(
        "vip",
        {
            "input_video_path": str(video),
            "masks_dir_path": str(dispatcher_masks),
            "arkit_depth_dir_path": str(tmp_path / "missing-depth"),
            "arkit_confidence_dir_path": str(tmp_path / "missing-confidence"),
        },
    )
    assert dispatched["status"] == "succeeded"
